from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from PIL import Image
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from dc_nerf.pipelines.base_pipeline import ModifiedVanillaPipeline
from dc_nerf.data.datamanagers.dc_datamanager import DCDataManagerConfig
from dc_nerf.data.datamanagers.dc_splat_datamanager import DCSplatDataManagerConfig
from dc.dc import DC, DCConfig, tensor_to_pil, DC
from dc.utils.imageutil import merge_images
from dc.utils.sysutil import clean_gpu
from dc.utils.free_lunch import register_free_upblock2d, register_free_crossattn_upblock2d

cmap = plt.get_cmap("viridis")


@dataclass
class DCPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: DCPipeline)
    datamanager: Union[DCDataManagerConfig, DCSplatDataManagerConfig] = DCDataManagerConfig()

    # DC configs.
    dc: DCConfig = DCConfig()
    dc_device: Optional[Union[torch.device, str]] = None

    dc_loss_mult: float = 1.0
    change_view_step: int = 1
    log_step: int = 10


class DCPipeline(ModifiedVanillaPipeline):
    config: DCPipelineConfig

    def __init__(
        self,
        config: DCPipelineConfig,
        device: Union[str, torch.device],
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler, **kwargs)

        # Construct DC
        self.dc_device = (
            torch.device(device) if self.config.dc_device is None else torch.device(self.config.dc_device)
        )
        self.config.dc.device = self.dc_device
        self.use_wandb = kwargs.get("wandb_enabled", False)
        
        self.dc = DC(self.config.dc, use_wandb=self.use_wandb)
        # Caching source's x0
        self.src_x0s = dict()
        self.current_spot = None

        # Load cached foreground masks for masked Perp-Neg, keyed by image stem name
        self.cached_masks = {}
        if self.config.dc.depth_masked_perp_neg and self.config.dc.depth_mask_source == "cached":
            from pathlib import Path as P
            mask_dir = P(self.config.dc.cached_mask_dir)
            if mask_dir.exists():
                for img_path in self.datamanager.train_dataparser_outputs.image_filenames:
                    stem = img_path.stem
                    mask_path = mask_dir / f"{stem}.png"
                    if mask_path.exists():
                        mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0
                        self.cached_masks[stem] = torch.tensor(mask).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                print(f"[Masked PN] Loaded {len(self.cached_masks)} cached masks from {mask_dir}")
            else:
                print(f"[Masked PN] WARNING: cached_mask_dir '{mask_dir}' not found, falling back to depth")
                self.config.dc.depth_mask_source = "depth"
        
    def get_current_rendering(self, step):
        if getattr(self, "current_spot", None) is None or step % self.config.change_view_step == 0:
            self.current_spot = np.random.randint(len(self.datamanager.train_dataparser_outputs.image_filenames))
        current_spot = self.current_spot
        current_index = self.datamanager.image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index:current_index+1].to(self.device)
        # Get the image stem for correct cached mask lookup
        current_stem = self.datamanager.train_dataparser_outputs.image_filenames[current_index].stem
        camera_outputs = self.model.diff_get_outputs_for_camera(current_camera)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B,3,H,W]
        # Extract depth for depth-masked Perp-Neg (free — already computed by renderer)
        depth_map = camera_outputs.get("depth", None)
        if depth_map is not None:
            depth_map = depth_map.detach().unsqueeze(dim=0).permute(0, 3, 1, 2)  # [1,1,H,W]

        # delete to free up memory
        del camera_outputs
        del current_camera
        clean_gpu()

        return rendered_image, current_spot, current_stem, depth_map

    def get_train_loss_dict(self, step: int):
        loss_dict = dict()

        rendered_image, current_spot, current_stem, depth_map = self.get_current_rendering(step)
        # get original image from dataloader
        original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
        original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)

        h, w = original_image.shape[2:]
        l = min(h, w)
        h = int(h * 512 / l)
        w = int(w * 512 / l)  # resize an image such that the smallest length is 512.
        original_image_512 = F.interpolate(original_image, size=(h, w), mode="bilinear")
        rendered_image_512 = F.interpolate(rendered_image, size=(h, w), mode="bilinear")

        if current_spot not in self.src_x0s.keys():
            with torch.no_grad():
                src_x0 = self.dc.encode_image(original_image_512.to(self.dc_device))
                self.src_x0s[current_spot] = src_x0.clone().cpu()
        else:
            src_x0 = self.src_x0s[current_spot].to(self.dc_device)

        x0 = self.dc.encode_image(rendered_image_512.to(self.dc_device))
        src_emb = self.dc.encode_src_image(original_image_512.to(self.dc_device))

        del rendered_image_512
        del original_image_512
        clean_gpu()

        # Build foreground mask for masked Perp-Neg
        depth_mask = None
        if self.config.dc.depth_masked_perp_neg and self.config.dc.perp_neg:
            source = self.config.dc.depth_mask_source
            use_cached = source == "cached" and current_stem in self.cached_masks
            if use_cached:
                # Precomputed mask (e.g. from Grounded-SAM), resize to 512-space
                depth_mask = F.interpolate(self.cached_masks[current_stem], size=(h, w), mode="nearest")
                depth_mask = depth_mask.to(self.dc_device)
            elif depth_map is not None:
                # Renderer depth, percentile-normalized
                d_flat = depth_map.reshape(-1)
                d_lo = torch.quantile(d_flat, 0.05)
                d_hi = torch.quantile(d_flat, 0.95)
                depth_clipped = depth_map.clamp(d_lo, d_hi)
                depth_norm = (depth_clipped - d_lo) / (d_hi - d_lo + 1e-8)
                mask_pixel = (depth_norm < self.config.dc.depth_mask_threshold).float()
                depth_mask = F.interpolate(mask_pixel, size=(h, w), mode="nearest")
                depth_mask = depth_mask.to(self.dc_device)
            # Post-process mask in image-space (512-space)
            if depth_mask is not None:
                # Optional hard dilation (expand binary region)
                d = self.config.dc.perp_neg_mask_dilate
                if d > 0:
                    depth_mask = F.max_pool2d(depth_mask, kernel_size=2*d+1, stride=1, padding=d)
                # Optional Gaussian blur (soft falloff: 1.0 at core → 0.0 far away)
                sigma = self.config.dc.perp_neg_mask_blur
                if sigma > 0:
                    k = int(6 * sigma) | 1  # kernel covers ±3σ, forced odd
                    depth_mask = TF.gaussian_blur(depth_mask, kernel_size=k, sigma=sigma)

        dic = self.dc(tgt_x0=x0, src_x0=src_x0, src_emb=src_emb, return_dict=True, step=step, current_spot=current_spot, depth_mask=depth_mask)
        grad = dic["grad"].cpu()
        grad_mask = dic.get("grad_mask", None)
        self_grad_mask = dic.get("self_grad_mask", None)
        cross_attention_mask = dic.get("cross_attention_mask", None)
        loss = dic["loss"] * self.config.dc_loss_mult
        loss = loss.to(self.device)
        loss_dict["dc_loss"] = loss
        
        vis_grad = self.visualize_grad(grad, w, h)
        
        min_size = 128
        original_image_resized = TF.resize(original_image[0], min_size)
        vis_grad_resized = TF.resize(vis_grad, min_size)
        rendered_image_resized = TF.resize(rendered_image[0], min_size)
        
        if self.use_wandb:
            import wandb
            wandb.log({
                "grad": wandb.Image(vis_grad_resized),
                "original_image": wandb.Image(original_image_resized.permute(1, 2, 0).detach().cpu().numpy()),
                "rendered_image": wandb.Image(rendered_image_resized.permute(1, 2, 0).detach().cpu().numpy()),
            }, step=step, commit=False) if step % self.config.log_step == 0 else None
            
            wandb.log({
                "dc_loss": loss.item(),
            }, step=step, commit=False)

        # Save depth mask visualization for debugging
        if depth_mask is not None and step % self.config.log_step == 0:
            mask_vis = F.interpolate(depth_mask.cpu(), size=(h, w), mode="nearest")
            mask_img = Image.fromarray((mask_vis[0, 0].numpy() * 255).astype(np.uint8))
            mask_img.save(self.base_dir / f"logging/{step}_depth_mask.png")

        # Save self-derived relevance mask visualization for debugging
        if grad_mask is not None and step % self.config.log_step == 0:
            grad_mask_vis = F.interpolate(grad_mask.cpu(), size=(h, w), mode="bilinear", align_corners=False)
            grad_mask_img = Image.fromarray((grad_mask_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
            grad_mask_img.save(self.base_dir / f"logging/{step}_gradient_mask.png")

        if self_grad_mask is not None and step % self.config.log_step == 0:
            self_grad_mask_vis = F.interpolate(self_grad_mask.cpu(), size=(h, w), mode="bilinear", align_corners=False)
            self_grad_mask_img = Image.fromarray((self_grad_mask_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
            self_grad_mask_img.save(self.base_dir / f"logging/{step}_self_mask.png")

        if cross_attention_mask is not None and step % self.config.log_step == 0:
            cross_attention_mask_vis = F.interpolate(cross_attention_mask.cpu(), size=(h, w), mode="bilinear", align_corners=False)
            cross_attention_mask_img = Image.fromarray((cross_attention_mask_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
            cross_attention_mask_img.save(self.base_dir / f"logging/{step}_cross_attention_mask.png")

        # logging
        if step % self.config.log_step == 0:
            self.log_images(rendered_image, original_image, grad, step)

        return None, loss_dict, dict()

    @torch.no_grad()
    def log_images(self, rendered_image, original_image, grad, step):
        edit_img = tensor_to_pil(rendered_image)
        orig_img = tensor_to_pil(original_image)

        w, h = edit_img.size
        
        vis_grad = grad if isinstance(grad, Image.Image) else self.visualize_grad(grad, w, h)
        
        img = merge_images([orig_img, edit_img, vis_grad])
        img.save(self.base_dir / f"logging/{step}.png")
        
    def visualize_grad(self, grad, w, h):
        vis_grad = grad.norm(dim=1).clone().detach().cpu()
        vis_grad = vis_grad / vis_grad.max()
        vis_grad = vis_grad.clamp(0, 1).squeeze().numpy()
        vis_grad = cmap(vis_grad)[..., :3]
        vis_grad = Image.fromarray((vis_grad * 255).astype(np.uint8))
        vis_grad = vis_grad.resize((w, h), resample=Image.Resampling.NEAREST)
        
        return vis_grad

    # to enable backprop.
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.model.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.model.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
