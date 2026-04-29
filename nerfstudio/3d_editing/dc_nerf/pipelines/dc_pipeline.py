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
from dc.dc import DC, DCConfig, tensor_to_pil
from dc.mask_voxel_cache import MaskVoxelCache
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

    # 3D voxel-cache for cross-view-consistent localization (Stage 1 prototype).
    # When enabled, per-view diffusion masks are scattered into a coarse 3D
    # voxel grid via depth backprojection, EMA-aggregated across views, and
    # queried back per-view to override (or blend with) the internal mask.
    # See `nerfstudio/dc/mask_voxel_cache.py` for the math + paper references.
    mask_voxel_cache_enabled: bool = False
    mask_voxel_cache_resolution: int = 128
    mask_voxel_cache_ema_beta: float = 0.9
    # Warmup ramp for the external-mask blend factor inside DC.__call__.
    # `blend(step) = clamp((step - start) / (end - start), 0, 1) * max_blend`.
    # During [0, start] iterations the cache is built but not yet used in the
    # gradient. During [start, end] the cache phases in. After `end`, blend
    # stays at `max_blend`.
    mask_voxel_cache_warmup_start: int = 700
    mask_voxel_cache_warmup_end: int = 1500
    mask_voxel_cache_max_blend: float = 0.5
    mask_voxel_cache_accumulation_threshold: float = 0.3


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
        # Caching source's x0 and IP2P image-conditioning latent per view.
        self.src_x0s = dict()
        self.src_encodeds = dict()
        self.current_spot = None

        # Optional 3D voxel-cache for cross-view-consistent localization.
        # Initialized lazily on first iteration so we have access to the
        # scene bounding box from the dataparser.
        self.mask_voxel_cache: Optional[MaskVoxelCache] = None

    def get_current_rendering(self, step):
        if getattr(self, "current_spot", None) is None or step % self.config.change_view_step == 0:
            self.current_spot = np.random.randint(len(self.datamanager.train_dataparser_outputs.image_filenames))
        current_spot = self.current_spot
        current_index = self.datamanager.image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index:current_index+1].to(self.device)
        camera_outputs = self.model.diff_get_outputs_for_camera(current_camera)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B,3,H,W]

        # When the voxel cache is active we also need the rendered depth and
        # the camera-space ray bundle (origins + directions in world frame) so
        # we can backproject pixels to 3D world points. We compute them here
        # under the same render call rather than re-rendering later.
        depth_world = None
        accumulation_world = None
        ray_origins = None
        ray_directions = None
        if self.config.mask_voxel_cache_enabled:
            depth_t = camera_outputs.get("depth", None)
            if depth_t is not None:
                depth_world = depth_t.detach().clone()  # [H, W, 1]
                accumulation_t = camera_outputs.get("accumulation", None)
                if accumulation_t is not None:
                    accumulation_world = accumulation_t.detach().clone()  # [H, W, 1]
                # Generate world-space rays (Nerfstudio convention: rays are
                # already in world frame, so depth-along-ray gives world points
                # directly via r(t) = o + t·d — same parametric form as the
                # NeRF volume-rendering integral).
                rays = current_camera.generate_rays(camera_indices=0, keep_shape=True)
                ray_origins = rays.origins.detach().clone()  # [H, W, 3]
                ray_directions = rays.directions.detach().clone()  # [H, W, 3]

        # delete to free up memory
        del camera_outputs
        del current_camera
        clean_gpu()

        return rendered_image, current_spot, depth_world, accumulation_world, ray_origins, ray_directions

    def _ensure_voxel_cache(self):
        """Lazy-initialize the voxel cache from the dataparser scene box.

        Reference: `dataparser_outputs.scene_box.aabb` is the standard
        Nerfstudio scene-bounds tensor, shape [2, 3] = [[x_min, y_min, z_min],
        [x_max, y_max, z_max]]. Pad slightly so rays grazing the boundary
        still fall inside.
        """
        if self.mask_voxel_cache is not None:
            return
        if not self.config.mask_voxel_cache_enabled:
            return
        scene_box = self.datamanager.train_dataparser_outputs.scene_box
        aabb = scene_box.aabb.to(self.device)  # [2, 3]
        bbox_min = aabb[0]
        bbox_max = aabb[1]
        # Inflate by 5% per side to absorb edge cases at the bbox boundary.
        extent = bbox_max - bbox_min
        bbox_min = bbox_min - 0.05 * extent
        bbox_max = bbox_max + 0.05 * extent
        self.mask_voxel_cache = MaskVoxelCache(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=self.config.mask_voxel_cache_resolution,
            ema_beta=self.config.mask_voxel_cache_ema_beta,
            device=self.device,
        )

    def _voxel_cache_warmup_blend(self, step: int) -> float:
        """Linear ramp from 0 → max_blend over [warmup_start, warmup_end].

        Same shape as your existing `gradient_mask_warmup` pattern, just
        applied to the external-mask blend rather than to the mask itself.
        """
        s = self.config.mask_voxel_cache_warmup_start
        e = self.config.mask_voxel_cache_warmup_end
        if e <= s:
            return float(self.config.mask_voxel_cache_max_blend)
        progress = max(0, step - s) / max(1, e - s)
        progress = min(max(progress, 0.0), 1.0)
        return float(progress * self.config.mask_voxel_cache_max_blend)

    def get_train_loss_dict(self, step: int):
        loss_dict = dict()

        rendered_image, current_spot, depth_world, accumulation_world, ray_origins, ray_directions = (
            self.get_current_rendering(step)
        )
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

        if current_spot not in self.src_encodeds:
            with torch.no_grad():
                src_emb = self.dc.encode_src_image(original_image_512.to(self.dc_device))
                src_encoded = src_emb.latent_dist.mode()
                self.src_encodeds[current_spot] = src_encoded.clone().cpu()
        else:
            src_encoded = self.src_encodeds[current_spot].to(self.dc_device)

        x0 = self.dc.encode_image(rendered_image_512.to(self.dc_device))

        del rendered_image_512
        del original_image_512
        clean_gpu()

        # ----------------------------------------------------------------
        # 3D voxel-cache query: produce an external mask consistent across
        # views by reading the EMA-aggregated 3D mask field at the world
        # points each pixel of the current view sees. The math:
        #     r(t) = o + t · d         (NeRF Eq. 1 ray parameterization)
        # with `t = depth_world` from the differentiable renderer, gives
        # the world-space surface points per pixel. We then look up the
        # voxel grid at those points → external mask in image space.
        # The lookup operates at the latent (DDS-mask) resolution.
        # ----------------------------------------------------------------
        external_grad_mask = None
        external_grad_mask_valid = None
        external_mask_blend = 0.0
        mask_world_points = None
        mask_world_points_valid = None
        if (
            self.config.mask_voxel_cache_enabled
            and depth_world is not None
            and ray_origins is not None
            and ray_directions is not None
        ):
            self._ensure_voxel_cache()
            mask_h, mask_w = x0.shape[-2:]
            # Move spatial maps to [B, C, H, W] for F.interpolate, then to mask resolution.
            d = depth_world.to(self.dc_device)
            if d.dim() == 2:
                d = d.unsqueeze(-1)  # [H, W, 1]
            d = d.permute(2, 0, 1).unsqueeze(0).float()  # [1, 1, H, W]
            if accumulation_world is None:
                acc = torch.ones_like(d)
            else:
                acc = accumulation_world.to(self.dc_device)
                if acc.dim() == 2:
                    acc = acc.unsqueeze(-1)
                acc = acc.permute(2, 0, 1).unsqueeze(0).float()  # [1, 1, H, W]
            o = ray_origins.to(self.dc_device).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
            dr = ray_directions.to(self.dc_device).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
            d = F.interpolate(d, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
            acc = F.interpolate(acc, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
            o = F.interpolate(o, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
            dr = F.interpolate(dr, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
            # Renormalize directions: bilinear interpolation breaks unit norm and
            # NeRF rays are conventionally unit-direction, so r(t) = o + t·d̂.
            dr = dr / dr.norm(dim=1, keepdim=True).clamp_min(1e-8)
            # Backproject: world points where each ray reaches `depth_world`.
            world_points = o + d * dr  # [1, 3, mask_h, mask_w]
            mask_world_points = world_points.permute(0, 2, 3, 1).reshape(-1, 3)  # [N, 3]
            mask_world_points_valid = (
                (acc > float(self.config.mask_voxel_cache_accumulation_threshold))
                & torch.isfinite(d)
                & (d > 0.0)
            ).reshape(-1)
            # Query the cache. `cache_valid` is true only for observed, in-bounds,
            # sufficiently accumulated voxels; invalid pixels fall back to the
            # internal per-view mask inside DC.__call__.
            queried, cache_valid = self.mask_voxel_cache.query(
                mask_world_points,
                in_bounds=mask_world_points_valid,
                return_valid=True,
            )
            external_grad_mask = queried.view(1, 1, mask_h, mask_w).to(
                device=x0.device, dtype=x0.dtype
            )
            external_grad_mask_valid = cache_valid.view(1, 1, mask_h, mask_w).to(device=x0.device)
            external_mask_blend = self._voxel_cache_warmup_blend(step)

        dic = self.dc(
            tgt_x0=x0,
            src_x0=src_x0,
            src_encoded=src_encoded,
            return_dict=True,
            step=step,
            current_spot=current_spot,
            external_grad_mask=external_grad_mask,
            external_grad_mask_valid=external_grad_mask_valid,
            external_mask_blend=external_mask_blend,
        )

        # ----------------------------------------------------------------
        # 3D voxel-cache update: scatter the fresh internal per-view DC mask
        # into the voxel grid at the same world points we queried from. The
        # teacher is intentionally the pre-voxel-blend hybrid mask, so after
        # warmup the cache keeps learning from the diffusion mask instead of
        # feeding its own values back into itself.
        #
        # Reference: same scatter-mean-then-EMA pattern used in
        # Panoptic Lifting (Siddiqui et al., CVPR 2023, §3.1) for cross-
        # frame label accumulation, adapted here to soft mask values.
        # ----------------------------------------------------------------
        if (
            self.config.mask_voxel_cache_enabled
            and self.mask_voxel_cache is not None
            and mask_world_points is not None
            and dic.get("internal_grad_mask", None) is not None
        ):
            new_mask = dic["internal_grad_mask"].detach().to(self.device).reshape(-1)
            self.mask_voxel_cache.update(
                mask_world_points,
                new_mask,
                in_bounds=mask_world_points_valid,
            )
            if self.use_wandb and step % self.config.log_step == 0:
                import wandb
                valid_ratio = (
                    float(external_grad_mask_valid.float().mean().item())
                    if external_grad_mask_valid is not None
                    else 0.0
                )
                wandb.log(
                    {
                        "dc_debug/voxel_cache_occupancy": float(self.mask_voxel_cache.occupancy),
                        "dc_debug/voxel_cache_blend": float(external_mask_blend),
                        "dc_debug/voxel_cache_valid_ratio": valid_ratio,
                    },
                    step=step,
                    commit=False,
                )
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

        if external_grad_mask is not None and step % self.config.log_step == 0:
            voxel_mask_vis = F.interpolate(external_grad_mask.detach().cpu(), size=(h, w), mode="bilinear", align_corners=False)
            voxel_mask_img = Image.fromarray((voxel_mask_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
            voxel_mask_img.save(self.base_dir / f"logging/{step}_voxel_cache_mask.png")

        if external_grad_mask_valid is not None and step % self.config.log_step == 0:
            voxel_valid_vis = F.interpolate(
                external_grad_mask_valid.detach().float().cpu(),
                size=(h, w),
                mode="nearest",
            )
            voxel_valid_img = Image.fromarray((voxel_valid_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
            voxel_valid_img.save(self.base_dir / f"logging/{step}_voxel_cache_valid.png")

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
