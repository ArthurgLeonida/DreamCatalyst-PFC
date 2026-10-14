"""Step-3 editing pipeline: render one view, compute the DC loss, update the NeRF.

Upstream DreamCatalyst structure (get_current_rendering, get_train_loss_dict,
log_images, visualize_grad, get_outputs_for_camera_ray_bundle), plus the
optional 3D voxel cache of Part 2, which shares the edit mask across views.
With the cache enabled, every step:

  1. backprojects the latent-resolution pixels of the view to 3D with the
     rendered depth,
  2. reads the cache at those points (DC's external mask),
  3. runs DC,
  4. writes this view's raw edit mask into the cache.

The cache is built once ``mask_voxel_cache_bbox_observe_steps`` steps have
observed the scene's surface points, on their bounding box.
"""

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
from dc.method_config import VOXEL_CACHE_PARAMS
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

    # 3D voxel cache (Part 2). Values come from method_config.VOXEL_CACHE_PARAMS;
    # see the comments there and docs/VoxelCacheExplained.md.
    mask_voxel_cache_enabled: bool = VOXEL_CACHE_PARAMS["mask_voxel_cache_enabled"]
    mask_voxel_cache_resolution: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_resolution"]
    mask_voxel_cache_ema_beta_camera_factor: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_ema_beta_camera_factor"]
    mask_voxel_cache_warmup_start: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_warmup_start"]
    mask_voxel_cache_warmup_end: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_warmup_end"]
    mask_voxel_cache_max_blend: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_max_blend"]
    mask_voxel_cache_accumulation_threshold: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_accumulation_threshold"]
    mask_voxel_cache_observation_fraction: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_observation_fraction"]
    mask_voxel_cache_min_observations_floor: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_min_observations_floor"]
    mask_voxel_cache_min_observations_cap: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_min_observations_cap"]
    mask_voxel_cache_max_variance: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_max_variance"]
    mask_voxel_cache_variance_decay: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_variance_decay"]
    mask_voxel_cache_mass_threshold: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_mass_threshold"]
    mask_voxel_cache_scale_normalize_quantile: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_scale_normalize_quantile"]
    mask_voxel_cache_bbox_observe_steps: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_bbox_observe_steps"]
    mask_voxel_cache_bbox_observe_quantile: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_bbox_observe_quantile"]
    mask_voxel_cache_bbox_inflation: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_bbox_inflation"]
    mask_voxel_cache_angular_freeze_patience: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_angular_freeze_patience"]
    mask_voxel_cache_angular_freeze_warmup: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_angular_freeze_warmup"]


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
        if self.config.dc.localization_enabled:
            print(
                f"[self-mask] per-view EMA beta = {self.dc.self_mask_ema_beta:.6f} "
                f"(memory = {self.config.dc.gradient_mask_ema_memory_visits} visits of each view)"
            )
        if self.config.mask_voxel_cache_enabled and not self.config.dc.cross_attention_mask_enabled:
            raise ValueError(
                "mask_voxel_cache_enabled requires cross_attention_mask_enabled: "
                "cache additions are gated by the cross-attention mask"
            )

        # Caching source's x0 and IP2P image-conditioning latent per view.
        self.src_x0s = dict()
        self.src_encodeds = dict()
        self.current_spot = None

        # Voxel cache, built lazily once the bounding box has been observed.
        self.mask_voxel_cache: Optional[MaskVoxelCache] = None
        self.mask_voxel_cache_start_step: Optional[int] = None
        self._observed_pts_min: Optional[torch.Tensor] = None
        self._observed_pts_max: Optional[torch.Tensor] = None
        self._observed_pts_count: int = 0

    def get_current_rendering(self, step):
        """Render the current training view (a new random one every change_view_step).

        Also returns the rendered depth and accumulation when the voxel cache
        needs them to backproject pixels (else None).
        """
        if getattr(self, "current_spot", None) is None or step % self.config.change_view_step == 0:
            self.current_spot = np.random.randint(len(self.datamanager.train_dataparser_outputs.image_filenames))
        current_spot = self.current_spot
        current_index = self.datamanager.image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index:current_index+1].to(self.device)
        camera_outputs = self.model.diff_get_outputs_for_camera(current_camera)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B,3,H,W]

        depth_world = None
        accumulation_world = None
        if self.config.mask_voxel_cache_enabled:
            depth_t = camera_outputs.get("depth", None)
            if depth_t is not None:
                depth_world = depth_t.detach().clone()  # [H, W, 1]
                accumulation_t = camera_outputs.get("accumulation", None)
                if accumulation_t is not None:
                    accumulation_world = accumulation_t.detach().clone()  # [H, W, 1]

        del camera_outputs
        del current_camera
        clean_gpu()

        return rendered_image, current_spot, depth_world, accumulation_world

    # ------------------------------------------------------------------
    # 3D voxel cache (Part 2)
    # ------------------------------------------------------------------

    def _num_train_views(self) -> int:
        return max(1, len(self.datamanager.train_dataparser_outputs.image_filenames))

    def _cache_edit_step(self, step: int) -> int:
        """Zero-based step of the edit run.

        The trainer step resumes from the reconstruction checkpoint (e.g.
        30000), but the cache starts empty with the edit, so its warmup and
        freeze schedules count from the first edit step.
        """
        if self.mask_voxel_cache_start_step is None:
            self.mask_voxel_cache_start_step = int(step)
        return max(0, int(step) - self.mask_voxel_cache_start_step)

    def _cache_warmup_blend(self, edit_step: int) -> float:
        """Fusion strength: a linear ramp from 0 to max_blend over [warmup_start, warmup_end].

        The cache only starts to act once it has accumulated multi-view evidence.
        """
        start = self.config.mask_voxel_cache_warmup_start
        end = self.config.mask_voxel_cache_warmup_end
        if end <= start:
            return float(self.config.mask_voxel_cache_max_blend)
        progress = max(0, edit_step - start) / max(1, end - start)
        progress = min(max(progress, 0.0), 1.0)
        return float(progress * self.config.mask_voxel_cache_max_blend)

    def _cache_ema_beta(self) -> float:
        """Decay of the grid mean, from the camera count: 1 - 1 / (factor * N_cam)."""
        factor = max(float(self.config.mask_voxel_cache_ema_beta_camera_factor), 1e-6)
        beta = 1.0 - 1.0 / (factor * float(self._num_train_views()))
        return min(max(beta, 0.0), 0.9999)

    def _cache_min_observations(self) -> int:
        """Count gate: clamp(ceil(fraction * N_cam), floor, cap) unique views."""
        fraction = min(max(float(self.config.mask_voxel_cache_observation_fraction), 0.0), 1.0)
        floor = max(int(self.config.mask_voxel_cache_min_observations_floor), 1)
        cap = max(int(self.config.mask_voxel_cache_min_observations_cap), floor)
        threshold = int(np.ceil(float(self._num_train_views()) * fraction))
        return min(max(threshold, floor), cap)

    def _backproject_view(self, current_spot, depth_world, accumulation_world, mask_shape):
        """World points seen by each latent-resolution pixel of the current view.

        Rays are generated directly at the latent resolution (the centers of the
        coarse pixels) and followed to the rendered depth: x = o + D * d. A
        point is valid only if its ray hits a surface: accumulation above the
        threshold, finite and positive depth.

        Returns (points [N, 3], directions [N, 3], valid [N], accumulation [1, 1, h, w]).
        """
        mask_h, mask_w = mask_shape
        H_cam, W_cam = int(depth_world.shape[0]), int(depth_world.shape[1])
        current_index = self.datamanager.image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.train_dataparser_outputs.cameras[
            current_index:current_index + 1
        ].to(self.device)
        # Latent pixel centers in camera pixels; nerfstudio coords are (y, x).
        yy = (torch.arange(mask_h, device=self.device, dtype=torch.float32) + 0.5) * (H_cam / mask_h)
        xx = (torch.arange(mask_w, device=self.device, dtype=torch.float32) + 0.5) * (W_cam / mask_w)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1)  # [mask_h, mask_w, 2]
        rays = current_camera.generate_rays(camera_indices=0, coords=coords, keep_shape=True)
        origins = rays.origins.detach()  # [mask_h, mask_w, 3]
        directions = rays.directions.detach()  # [mask_h, mask_w, 3]
        del current_camera, rays

        depth = depth_world.to(self.dc_device)
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)
        depth = depth.permute(2, 0, 1).unsqueeze(0).float()  # [1, 1, H, W]
        if accumulation_world is None:
            accumulation = torch.ones_like(depth)
        else:
            accumulation = accumulation_world.to(self.dc_device)
            if accumulation.dim() == 2:
                accumulation = accumulation.unsqueeze(-1)
            accumulation = accumulation.permute(2, 0, 1).unsqueeze(0).float()
        depth = F.interpolate(depth, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
        accumulation = F.interpolate(accumulation, size=(mask_h, mask_w), mode="bilinear", align_corners=False)

        origins = origins.permute(2, 0, 1).unsqueeze(0).to(self.dc_device).float()
        directions = directions.permute(2, 0, 1).unsqueeze(0).to(self.dc_device).float()
        points = origins + depth * directions  # [1, 3, mask_h, mask_w]
        valid = (
            (accumulation > float(self.config.mask_voxel_cache_accumulation_threshold))
            & torch.isfinite(depth)
            & (depth > 0.0)
        ).reshape(-1)
        return (
            points.permute(0, 2, 3, 1).reshape(-1, 3),
            directions.permute(0, 2, 3, 1).reshape(-1, 3),
            valid,
            accumulation,
        )

    def _observe_bbox(self, points: torch.Tensor, valid: torch.Tensor) -> None:
        """Grow the observed bounding box with this view's valid surface points.

        Per view, the (q, 1 - q) quantiles of each axis clip far-depth outliers
        from low-confidence rays (q = 0 takes the plain min/max). The box is
        the union of those per-view ranges.
        """
        pts = points.reshape(-1, 3).to(self.device).float()
        pts = pts[valid.reshape(-1).to(self.device)]
        # Drop non-finite rows (infinite depths give infinite points).
        pts = pts[torch.isfinite(pts).all(dim=-1)]
        if pts.numel() == 0:
            return

        q = min(max(float(self.config.mask_voxel_cache_bbox_observe_quantile), 0.0), 0.49)
        if q <= 0.0:
            cur_min = pts.min(dim=0).values
            cur_max = pts.max(dim=0).values
        else:
            cur_min = torch.quantile(pts, q, dim=0)
            cur_max = torch.quantile(pts, 1.0 - q, dim=0)
        if self._observed_pts_min is None:
            self._observed_pts_min = cur_min
            self._observed_pts_max = cur_max
        else:
            self._observed_pts_min = torch.minimum(self._observed_pts_min, cur_min)
            self._observed_pts_max = torch.maximum(self._observed_pts_max, cur_max)
        self._observed_pts_count += 1

    def _build_voxel_cache(self) -> None:
        """Create the cache on the observed box, inflated by mask_voxel_cache_bbox_inflation.

        The observed box contains exactly the surface points the cache will
        index; boxes from camera positions or the dataparser can sit in another
        frame and leave most points out. The box is printed once so it can be
        checked against the camera positions.
        """
        if self._observed_pts_min is None or self._observed_pts_max is None:
            return

        inflation = float(self.config.mask_voxel_cache_bbox_inflation)
        center = 0.5 * (self._observed_pts_min + self._observed_pts_max)
        half_extent = 0.5 * (self._observed_pts_max - self._observed_pts_min) * (1.0 + inflation)
        bbox_min = center - half_extent
        bbox_max = center + half_extent
        ema_beta = self._cache_ema_beta()

        print(
            f"[voxel cache] observed bbox, inflation={inflation:.2f}, "
            f"resolution={self.config.mask_voxel_cache_resolution}, "
            f"ema_beta={ema_beta:.6f}"
        )
        print(f"[voxel cache] bbox_min = {bbox_min.tolist()}")
        print(f"[voxel cache] bbox_max = {bbox_max.tolist()}")
        print(f"[voxel cache] extent   = {(bbox_max - bbox_min).tolist()}")

        self.mask_voxel_cache = MaskVoxelCache(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=self.config.mask_voxel_cache_resolution,
            ema_beta=ema_beta,
            num_views=self._num_train_views(),
            variance_decay=float(self.config.mask_voxel_cache_variance_decay),
            device=self.device,
        )

    def _scale_normalize_cache_mask(self, cache_mask, valid):
        """Stretch the cache read's [1 - q, q] quantile range over valid pixels to [0, 1].

        The cache value is a multi-view mean, compressed toward mid-range, while
        the 2D mask is a sharp near-indicator. Without the stretch the cache
        reads below the 2D mask over the whole edit region and the
        positive-only fusion never fires there. Invalid pixels carry zero
        confidence downstream, so their stretched value never reaches the
        gradient.
        """
        sel = valid.reshape(-1).bool()
        if int(sel.sum().item()) < 16:
            return cache_mask  # too few observed pixels for a stable range
        vals = cache_mask.reshape(-1)[sel].float()
        q = min(max(float(self.config.mask_voxel_cache_scale_normalize_quantile), 0.5), 0.999)
        lo = torch.quantile(vals, 1.0 - q).to(cache_mask.dtype)
        hi = torch.quantile(vals, q).to(cache_mask.dtype)
        span = (hi - lo).clamp_min(1e-4)
        return ((cache_mask - lo) / span).clamp(0.0, 1.0)

    def _query_voxel_cache(self, points, valid, x0):
        """Read the cache for this view at the latent resolution of ``x0``.

        Returns a dict with the scale-matched cache mask, its validity and
        confidence (what DC fuses), plus the per-pixel view counts and variance
        and the count-gate threshold (for logging).
        """
        mask_h, mask_w = x0.shape[-2:]
        min_observations = self._cache_min_observations()
        values, cache_valid, confidence, counts, variance = self.mask_voxel_cache.query(
            points,
            in_bounds=valid,
            min_observations=min_observations,
            max_variance=self.config.mask_voxel_cache_max_variance,
            mass_threshold=float(self.config.mask_voxel_cache_mass_threshold),
        )
        cache_valid = cache_valid.view(1, 1, mask_h, mask_w).to(device=x0.device)
        cache_mask = values.view(1, 1, mask_h, mask_w).to(device=x0.device, dtype=x0.dtype)
        return {
            "mask": self._scale_normalize_cache_mask(cache_mask, cache_valid),
            "valid": cache_valid,
            "confidence": confidence.view(1, 1, mask_h, mask_w).to(device=x0.device, dtype=x0.dtype),
            "count": counts.view(1, 1, mask_h, mask_w),
            "variance": variance.view(1, 1, mask_h, mask_w),
            "min_observations": min_observations,
        }

    def _update_voxel_cache(self, dic, points, valid, directions, view_id, step, edit_step, return_written):
        """Write this view's raw edit mask into the cache, then try the angular freeze.

        The cache input is S's raw magnitude, not the final mask: the final
        mask already contains the cache and the attention restriction, which
        would feed back into the observations.
        Returns (per-voxel values written or None, cache input) for logging.
        """
        cache_input = dic["self_grad_mask_raw"].detach().to(self.device).reshape(-1)
        written = self.mask_voxel_cache.update(
            points,
            cache_input,
            in_bounds=valid,
            view_id=view_id,
            ray_directions=directions,
            return_per_voxel_mean=return_written,
        )

        if not self.mask_voxel_cache.angular_denominator_is_frozen:
            min_views = self._cache_min_observations()
            frozen_value = self.mask_voxel_cache.try_auto_freeze_angular_denominator(
                edit_step=edit_step,
                min_views=min_views,
                patience=int(self.config.mask_voxel_cache_angular_freeze_patience),
                warmup_steps=int(self.config.mask_voxel_cache_angular_freeze_warmup),
            )
            if frozen_value is not None:
                print(
                    f"[voxel-cache] auto-froze angular denominator at "
                    f"edit_step={edit_step} (global={step}): "
                    f"value={frozen_value:.5f} captured at peak "
                    f"edit_step={self.mask_voxel_cache.angular_peak_step} (min_views={min_views})"
                )
        return written, cache_input

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def get_train_loss_dict(self, step: int):
        loss_dict = dict()
        log_this_step = step % self.config.log_step == 0

        rendered_image, current_spot, depth_world, accumulation_world = (
            self.get_current_rendering(step)
        )
        # The train-camera slot is the unique view id: image_idx can refer to
        # non-contiguous dataset indices after the train/eval split.
        current_view_id = int(current_spot)
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

        # 3D voxel cache: read it for this view before DC, write this view after.
        view_points = None
        cache_query = None
        cache_blend = 0.0
        edit_step = None
        if self.config.mask_voxel_cache_enabled and depth_world is not None:
            edit_step = self._cache_edit_step(step)
            view_points = self._backproject_view(current_spot, depth_world, accumulation_world, x0.shape[-2:])
            points, _, valid, accumulation = view_points

            if self.mask_voxel_cache is None:
                self._observe_bbox(points, valid)
                if self._observed_pts_count >= int(self.config.mask_voxel_cache_bbox_observe_steps):
                    self._build_voxel_cache()
                if self.use_wandb and log_this_step:
                    import wandb
                    wandb.log(
                        {"dc_debug/voxel_cache_observing_count": int(self._observed_pts_count)},
                        step=step,
                        commit=False,
                    )

            if self.mask_voxel_cache is not None:
                cache_query = self._query_voxel_cache(points, valid, x0)
                cache_blend = self._cache_warmup_blend(edit_step)

            if self.use_wandb and log_this_step:
                self._log_backprojection(step, points, valid, accumulation)

        dic = self.dc(
            tgt_x0=x0,
            src_x0=src_x0,
            src_encoded=src_encoded,
            return_dict=True,
            step=step,
            current_spot=current_spot,
            external_grad_mask=cache_query["mask"] if cache_query is not None else None,
            external_grad_mask_valid=cache_query["valid"] if cache_query is not None else None,
            external_grad_mask_confidence=cache_query["confidence"] if cache_query is not None else None,
            external_mask_blend=cache_blend,
        )

        if cache_query is not None and dic["self_grad_mask_raw"] is not None:
            points, directions, valid, _ = view_points
            written, cache_input = self._update_voxel_cache(
                dic,
                points,
                valid,
                directions,
                view_id=current_view_id,
                step=step,
                edit_step=edit_step,
                return_written=self.use_wandb and log_this_step,
            )
            if self.use_wandb and log_this_step:
                self._log_voxel_cache_state(step, edit_step, cache_blend, cache_query, written, cache_input)

        grad = dic["grad"].cpu()
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

        # logging
        if log_this_step:
            self._save_dc_masks(step, (h, w), dic)
            if cache_query is not None:
                self._log_voxel_cache_read(step, (h, w), cache_query)
            self.log_images(rendered_image, original_image, grad, step)

        return None, loss_dict, dict()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _save_mask_png(self, mask, size, name, step, mode="bilinear"):
        """Save a [1, 1, h, w] map in [0, 1] as logging/{step}_{name}.png at ``size``.

        Returns the PIL image, so the caller can also send it to WandB.
        """
        mask = mask.detach().float().cpu()
        if mode == "nearest":
            vis = F.interpolate(mask, size=size, mode="nearest")
        else:
            vis = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
        image = Image.fromarray((vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
        image.save(self.base_dir / f"logging/{step}_{name}.png")
        return image

    def _save_dc_masks(self, step, size, dic):
        """Save the final edit mask, the self-mask S and the attention mask A as PNGs."""
        for key, name in (
            ("grad_mask", "gradient_mask"),
            ("self_grad_mask", "self_mask"),
            ("cross_attention_mask", "cross_attention_mask"),
        ):
            if dic.get(key) is not None:
                self._save_mask_png(dic[key], size, name, step)

    def _log_backprojection(self, step, points, valid, accumulation):
        """WandB: extent of the backprojected points, valid-ray share, share inside the cache box."""
        import wandb

        with torch.no_grad():
            pts_min = points.min(dim=0).values
            pts_max = points.max(dim=0).values
            payload = {
                "dc_debug/voxel_pts_min_x": float(pts_min[0]),
                "dc_debug/voxel_pts_min_y": float(pts_min[1]),
                "dc_debug/voxel_pts_min_z": float(pts_min[2]),
                "dc_debug/voxel_pts_max_x": float(pts_max[0]),
                "dc_debug/voxel_pts_max_y": float(pts_max[1]),
                "dc_debug/voxel_pts_max_z": float(pts_max[2]),
                "dc_debug/voxel_acc_filter_pass_frac": float(valid.float().mean()),
                "dc_debug/voxel_mean_accumulation": float(accumulation.float().mean()),
            }
            if self.mask_voxel_cache is not None:
                bbox_lo = self.mask_voxel_cache.bbox_min
                bbox_hi = self.mask_voxel_cache.bbox_max
                norm = (points - bbox_lo) / (bbox_hi - bbox_lo).clamp_min(1e-8)
                inside = ((norm >= 0.0) & (norm < 1.0)).all(dim=-1).float().mean()
                payload["dc_debug/voxel_pure_in_bbox_frac"] = float(inside)
        wandb.log(payload, step=step, commit=False)

    def _log_voxel_cache_state(self, step, edit_step, blend, cache_query, written, cache_input):
        """WandB: cache statistics after this step's update (counts, variance, angular freeze)."""
        import wandb

        cache = self.mask_voxel_cache
        min_observations = cache_query["min_observations"]
        frozen = cache.frozen_angular_denominator
        payload = {
            "dc_debug/voxel_cache_occupancy": float(cache.occupancy),
            "dc_debug/voxel_cache_blend": float(blend),
            "dc_debug/voxel_cache_valid_ratio": float(cache_query["valid"].float().mean().item()),
            "dc_debug/voxel_cache_edit_step": float(edit_step or 0),
            "dc_debug/voxel_cache_ema_beta": float(cache.ema_beta),
            "dc_debug/voxel_cache_mean_observation_count": float(cache.mean_observation_count),
            "dc_debug/voxel_cache_mean_observed_variance": float(cache.mean_observed_variance),
            "dc_debug/voxel_cache_mean_angular_factor": float(cache.mean_angular_factor),
            "dc_debug/voxel_cache_mean_angular_factor_trusted": float(
                cache.mean_angular_factor_at(min_views=min_observations)
            ),
            "dc_debug/voxel_cache_angular_denominator_frozen": float(frozen if frozen is not None else 0.0),
            "dc_debug/voxel_cache_angular_peak_value": float(cache.angular_peak_value),
            "dc_debug/voxel_cache_angular_peak_step": float(cache.angular_peak_step),
            "dc_debug/voxel_cache_min_observations": float(min_observations),
        }
        if written is not None and written.numel() > 0:
            written = written.detach().float().cpu()
            payload["dc_debug/voxel_cache_update_hist"] = wandb.Histogram(written.numpy(), num_bins=50)
            payload["dc_debug/voxel_cache_update_mean"] = float(written.mean().item())
        cache_input = cache_input.detach().float().cpu()
        if cache_input.numel() > 0:
            payload["dc_debug/voxel_cache_input_mask_hist"] = wandb.Histogram(cache_input.numpy(), num_bins=50)
            payload["dc_debug/voxel_cache_input_mask_mean"] = float(cache_input.mean().item())
        wandb.log(payload, step=step, commit=False)

    def _log_voxel_cache_read(self, step, size, cache_query):
        """Save (PNG) and log (WandB) the cache read for this view.

        value, confidence, the variance map (variance / max_variance: bright
        means the views disagree), value * confidence, and the validity mask.
        """
        cache_mask = cache_query["mask"].detach().float().clamp(0.0, 1.0)
        confidence = cache_query["confidence"].detach().float().clamp(0.0, 1.0)
        variance = cache_query["variance"].detach().float()
        trusted = cache_mask * confidence
        max_variance = float(self.config.mask_voxel_cache_max_variance)
        variance_scale = max_variance if max_variance > 1e-8 else 1.0

        mask_img = self._save_mask_png(cache_mask, size, "voxel_cache_mask", step)
        confidence_img = self._save_mask_png(confidence, size, "voxel_cache_confidence", step)
        variance_img = self._save_mask_png((variance / variance_scale).clamp(0.0, 1.0), size, "voxel_cache_variance", step)
        trusted_img = self._save_mask_png(trusted, size, "voxel_cache_trusted_mask", step)
        valid_img = self._save_mask_png(cache_query["valid"], size, "voxel_cache_valid", step, mode="nearest")
        if not self.use_wandb:
            return

        import wandb

        min_size = 128
        count = cache_query["count"].detach().float()
        wandb.log(
            {
                "dc_debug/voxel_cache_mask": wandb.Image(
                    TF.resize(mask_img, min_size), caption=f"step={step} | queried voxel-cache mask"
                ),
                "dc_debug/voxel_cache_mask_mean": float(cache_mask.mean().item()),
                "dc_debug/voxel_cache_mask_max": float(cache_mask.max().item()),
                "dc_debug/voxel_cache_mask_coverage_0.5": float((cache_mask > 0.5).float().mean().item()),
                "dc_debug/voxel_cache_confidence": wandb.Image(
                    TF.resize(confidence_img, min_size),
                    caption=f"step={step} | confidence: count x variance x angular x mass",
                ),
                "dc_debug/voxel_cache_confidence_mean": float(confidence.mean().item()),
                "dc_debug/voxel_cache_confidence_coverage_0.5": float((confidence > 0.5).float().mean().item()),
                "dc_debug/voxel_cache_query_count_mean": float(count.mean().item()),
                "dc_debug/voxel_cache_query_view_frac_mean": float(
                    (count / float(self._num_train_views())).mean().item()
                ),
                "dc_debug/voxel_cache_query_count_max": float(count.max().item()),
                "dc_debug/voxel_cache_query_variance_mean": float(variance.mean().item()),
                "dc_debug/voxel_cache_variance_map": wandb.Image(
                    TF.resize(variance_img, min_size),
                    caption=f"step={step} | cross-view variance / max_variance (bright = views disagree; >= gate is damped)",
                ),
                "dc_debug/voxel_cache_trusted_mask": wandb.Image(
                    TF.resize(trusted_img, min_size), caption=f"step={step} | voxel mask times confidence"
                ),
                "dc_debug/voxel_cache_trusted_mask_mean": float(trusted.mean().item()),
                "dc_debug/voxel_cache_trusted_mask_coverage_0.5": float((trusted > 0.5).float().mean().item()),
                "dc_debug/voxel_cache_valid_mask": wandb.Image(
                    TF.resize(valid_img, min_size), caption=f"step={step} | voxel-cache observed/in-bounds mask"
                ),
            },
            step=step,
            commit=False,
        )

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
