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

    mask_voxel_cache_enabled: bool = VOXEL_CACHE_PARAMS["mask_voxel_cache_enabled"]
    mask_voxel_cache_measure_only: bool = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_measure_only", False
    )
    mask_voxel_cache_scale_normalize: bool = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_scale_normalize", False
    )
    mask_voxel_cache_scale_normalize_quantile: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_scale_normalize_quantile", 0.95
    )
    mask_voxel_cache_trilinear: bool = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_trilinear", False
    )
    mask_voxel_cache_resolution: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_resolution"]
    mask_voxel_cache_ema_beta: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_ema_beta"]
    mask_voxel_cache_ema_beta_auto: bool = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_ema_beta_auto", False
    )
    mask_voxel_cache_ema_beta_camera_factor: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_ema_beta_camera_factor", 2.0
    )
    mask_voxel_cache_warmup_start: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_warmup_start"]
    mask_voxel_cache_warmup_end: int = VOXEL_CACHE_PARAMS["mask_voxel_cache_warmup_end"]
    mask_voxel_cache_max_blend: float = VOXEL_CACHE_PARAMS["mask_voxel_cache_max_blend"]
    mask_voxel_cache_accumulation_threshold: float = VOXEL_CACHE_PARAMS[
        "mask_voxel_cache_accumulation_threshold"
    ]
    mask_voxel_cache_update_threshold: float = VOXEL_CACHE_PARAMS[
        "mask_voxel_cache_update_threshold"
    ]
    mask_voxel_cache_update_source: str = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_update_source", "internal"
    )
    mask_voxel_cache_confidence_enabled: bool = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_confidence_enabled", False
    )
    mask_voxel_cache_min_observations: int = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_min_observations", 1
    )
    mask_voxel_cache_min_observations_auto: bool = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_min_observations_auto", False
    )
    mask_voxel_cache_observation_fraction: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_observation_fraction", 0.05
    )
    mask_voxel_cache_min_observations_floor: int = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_min_observations_floor", 2
    )
    mask_voxel_cache_min_observations_cap: int = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_min_observations_cap", 8
    )
    mask_voxel_cache_max_variance: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_max_variance", 0.0
    )
    mask_voxel_cache_variance_decay: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_variance_decay", 0.0
    )
    mask_voxel_cache_variance_peak_decay: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_variance_peak_decay", 0.0
    )
    mask_voxel_cache_angular_power: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_angular_power", 0.0
    )
    mask_voxel_cache_angular_relative: bool = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_angular_relative", False
    )
    mask_voxel_cache_mass_threshold: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_mass_threshold", 0.0
    )
    mask_voxel_cache_mass_power: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_mass_power", 0.0
    )
    mask_voxel_cache_angular_freeze_patience: int = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_angular_freeze_patience", 100
    )
    mask_voxel_cache_angular_freeze_warmup: int = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_angular_freeze_warmup", 50
    )
    mask_voxel_cache_min_angular_factor: float = VOXEL_CACHE_PARAMS.get(
        "mask_voxel_cache_min_angular_factor", 0.0
    )
    mask_voxel_cache_bbox_source: str = VOXEL_CACHE_PARAMS["mask_voxel_cache_bbox_source"]
    mask_voxel_cache_bbox_observe_steps: int = VOXEL_CACHE_PARAMS[
        "mask_voxel_cache_bbox_observe_steps"
    ]
    mask_voxel_cache_bbox_observe_quantile: float = VOXEL_CACHE_PARAMS[
        "mask_voxel_cache_bbox_observe_quantile"
    ]
    mask_voxel_cache_bbox_inflation: float = VOXEL_CACHE_PARAMS[
        "mask_voxel_cache_bbox_inflation"
    ]


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
        if getattr(self.config.dc, "gradient_mask_ema_beta_auto", False):
            n_cameras = max(
                1,
                len(self.datamanager.train_dataparser_outputs.image_filenames),
            )
            factor = max(
                float(self.config.dc.gradient_mask_ema_beta_camera_factor), 1e-6
            )
            auto_beta = 1.0 - 1.0 / (factor * float(n_cameras))
            auto_beta = min(max(auto_beta, 0.0), 0.9999)
            self.config.dc.gradient_mask_ema_beta = auto_beta
            self.dc.config.gradient_mask_ema_beta = auto_beta
            print(
                f"[self-mask] auto EMA beta = {auto_beta:.6f} "
                f"(N_cam={n_cameras}, factor={factor})"
            )
        # Caching source's x0 and IP2P image-conditioning latent per view.
        self.src_x0s = dict()
        self.src_encodeds = dict()
        self.current_spot = None

        self.mask_voxel_cache: Optional[MaskVoxelCache] = None
        self.mask_voxel_cache_start_step: Optional[int] = None
        self.mask_voxel_cache_effective_ema_beta: Optional[float] = None
        self.mask_voxel_cache_effective_min_observations: Optional[int] = None

        self._observed_pts_min: Optional[torch.Tensor] = None
        self._observed_pts_max: Optional[torch.Tensor] = None
        self._observed_pts_count: int = 0

    def get_current_rendering(self, step):
        if getattr(self, "current_spot", None) is None or step % self.config.change_view_step == 0:
            self.current_spot = np.random.randint(len(self.datamanager.train_dataparser_outputs.image_filenames))
        current_spot = self.current_spot
        current_index = self.datamanager.image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index:current_index+1].to(self.device)
        camera_outputs = self.model.diff_get_outputs_for_camera(current_camera)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  # [B,3,H,W]

        depth_world = None
        accumulation_world = None
        if self._voxel_cache_active():
            depth_t = camera_outputs.get("depth", None)
            if depth_t is not None:
                depth_world = depth_t.detach().clone()  # [H, W, 1]
                accumulation_t = camera_outputs.get("accumulation", None)
                if accumulation_t is not None:
                    accumulation_world = accumulation_t.detach().clone()  # [H, W, 1]

        # delete to free up memory
        del camera_outputs
        del current_camera
        clean_gpu()

        return rendered_image, current_spot, depth_world, accumulation_world

    def _voxel_cache_active(self) -> bool:
        """Whether the voxel cache should be built and updated this run.

        True when the cache is enabled (it influences the gradient) OR when
        `mask_voxel_cache_measure_only` is set (passive cache-off control: the
        cache accumulates cross-view statistics but never blends into the
        gradient). Gates depth capture, lazy build, and the update path, while
        the query/blend stays gated on `mask_voxel_cache_enabled` alone.
        """
        return bool(self.config.mask_voxel_cache_enabled) or bool(
            getattr(self.config, "mask_voxel_cache_measure_only", False)
        )

    def _scale_normalize_cache_mask(self, ext, valid):
        """Contrast-stretch the queried voxel-cache mask to [0,1] over its
        observed voxels before it is fused with the sharp 2D mask in DC.

        The cache value is a multi-view MEAN (compressed toward mid-range, with
        a 0.5 fallback for under-observed voxels), while the 2D hybrid mask is a
        sharp [0,1] indicator. DC's bidirectional fusion differences the two
        (``ext - grad_mask``); because the compressed cache sits below the 2D
        peaks across the whole edit region, the subtractive ("negative
        correction") term fires on the genuine edit, not just background — a
        scale artifact, not real 3D disagreement. Rescaling the cache's active
        range (its ``[1-q, q]`` percentiles over observed voxels) to [0,1] makes
        the comparison like-for-like, so the down-term only cleans background.

        Selection uses the observed/in-bounds mask (not confidence) so the low
        end of the range captures background voxels; invalid/fallback pixels
        carry zero confidence downstream, so their post-stretch value never
        reaches the gradient.
        """
        if valid is None:
            return ext
        sel = valid.reshape(-1).bool()
        if int(sel.sum().item()) < 16:
            return ext  # too few observed voxels to estimate a stable range
        vals = ext.reshape(-1)[sel].float()
        q = float(getattr(self.config, "mask_voxel_cache_scale_normalize_quantile", 0.95))
        q = min(max(q, 0.5), 0.999)
        lo = torch.quantile(vals, 1.0 - q).to(ext.dtype)
        hi = torch.quantile(vals, q).to(ext.dtype)
        span = (hi - lo).clamp_min(1e-4)
        return ((ext - lo) / span).clamp(0.0, 1.0)

    def _ensure_voxel_cache(self):
        """Lazy-initialize the voxel cache.

        Two bbox sources are supported, selected by
        `config.mask_voxel_cache_bbox_source`:

        - "cameras" (default, robust):
            Derive from the AABB of camera positions
            (`cameras.camera_to_worlds[..., :3, 3]`), inflated by
            `bbox_inflation`. By construction, rays generated from these
            cameras are in the same coordinate frame, so backprojected
            world points cannot fall outside the bbox due to a frame
            mismatch.

        - "scene_box":
            Use `dataparser_outputs.scene_box.aabb`. Fast but assumes the
            dataparser keeps scene_box in the same frame as the cameras —
            which several Nerfstudio dataparsers do not (some normalize
            cameras to a unit cube while leaving `scene_box.aabb` in raw
            world units, or vice versa). When this assumption fails, most
            backprojected points fall outside the bbox and the cache
            populates extremely sparsely (the symptom on the prior run:
            occupancy ~1.2%, valid_ratio ~0.5%).

        The chosen bbox is printed once at init so you can eyeball it
        against the camera positions and any depth statistics in WandB.
        """
        if self.mask_voxel_cache is not None:
            return
        if not self._voxel_cache_active():
            return

        source = str(self.config.mask_voxel_cache_bbox_source).lower()
        inflation = float(self.config.mask_voxel_cache_bbox_inflation)
        cameras = self.datamanager.train_dataparser_outputs.cameras

        if source == "observed":
            if self._observed_pts_min is None or self._observed_pts_max is None:
                return
            bbox_min = self._observed_pts_min.clone()
            bbox_max = self._observed_pts_max.clone()
        elif source == "cameras":
            # Camera-position AABB. cameras.camera_to_worlds is [..., 3, 4];
            # the last column is the translation (= camera origin in world).
            # NOTE: empirically wrong for object-centric capture — left here
            # only as an explicit opt-in for debugging.
            c2w = cameras.camera_to_worlds.to(self.device)
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0)
            cam_pos = c2w[..., :3, 3].reshape(-1, 3).float()
            bbox_min = cam_pos.min(dim=0).values
            bbox_max = cam_pos.max(dim=0).values
        elif source == "scene_box":
            scene_box = self.datamanager.train_dataparser_outputs.scene_box
            aabb = scene_box.aabb.to(self.device).float()
            bbox_min = aabb[0]
            bbox_max = aabb[1]
        else:
            raise ValueError(
                f"Unknown mask_voxel_cache_bbox_source={source!r}; "
                f"expected 'observed', 'cameras', or 'scene_box'."
            )

        center = 0.5 * (bbox_min + bbox_max)
        half_extent = 0.5 * (bbox_max - bbox_min) * (1.0 + inflation)
        bbox_min = center - half_extent
        bbox_max = center + half_extent

        ema_beta = self._effective_voxel_cache_ema_beta()
        self.mask_voxel_cache_effective_ema_beta = ema_beta

        print(
            f"[voxel cache] source={source}, inflation={inflation:.2f}, "
            f"resolution={self.config.mask_voxel_cache_resolution}, "
            f"ema_beta={ema_beta:.6f}"
        )
        print(f"[voxel cache] bbox_min = {bbox_min.tolist()}")
        print(f"[voxel cache] bbox_max = {bbox_max.tolist()}")
        print(f"[voxel cache] extent   = {(bbox_max - bbox_min).tolist()}")
        n_views = len(self.datamanager.train_dataparser_outputs.image_filenames)

        self.mask_voxel_cache = MaskVoxelCache(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            resolution=self.config.mask_voxel_cache_resolution,
            ema_beta=ema_beta,
            num_views=n_views if self.config.mask_voxel_cache_confidence_enabled else None,
            variance_decay=float(
                getattr(self.config, "mask_voxel_cache_variance_decay", 0.0)
            ),
            variance_peak_decay=float(
                getattr(self.config, "mask_voxel_cache_variance_peak_decay", 0.0)
            ),
            device=self.device,
        )

    def _effective_voxel_cache_ema_beta(self) -> float:
        """Return the manual or camera-count-aware voxel-cache EMA beta."""
        if not self.config.mask_voxel_cache_ema_beta_auto:
            return min(max(float(self.config.mask_voxel_cache_ema_beta), 0.0), 0.9999)

        n_cameras = max(
            1,
            len(self.datamanager.train_dataparser_outputs.image_filenames),
        )
        factor = max(float(self.config.mask_voxel_cache_ema_beta_camera_factor), 1e-6)
        beta = 1.0 - 1.0 / (factor * float(n_cameras))
        return min(max(beta, 0.0), 0.9999)

    def _effective_voxel_cache_min_observations(self) -> int:
        """Return the manual or camera-count-aware voxel trust threshold."""
        if not self.config.mask_voxel_cache_min_observations_auto:
            return max(int(self.config.mask_voxel_cache_min_observations), 1)

        n_cameras = max(
            1,
            len(self.datamanager.train_dataparser_outputs.image_filenames),
        )
        fraction = min(max(float(self.config.mask_voxel_cache_observation_fraction), 0.0), 1.0)
        floor = max(int(self.config.mask_voxel_cache_min_observations_floor), 1)
        cap = max(int(self.config.mask_voxel_cache_min_observations_cap), floor)
        threshold = int(np.ceil(float(n_cameras) * fraction))
        return min(max(threshold, floor), cap)

    def _observe_points(self, points_world: torch.Tensor, valid: torch.Tensor) -> None:
        """Accumulate robust per-axis bounds of valid backprojected world points.

        Called during the bbox-observation window (when bbox_source="observed"
        and the cache hasn't been built yet). Once enough iterations have been
        observed, `_ensure_voxel_cache()` will pick up these accumulated bounds.

        Per-iteration robustification:
          - With `bbox_observe_quantile = 0` we take the literal per-axis
            min/max for each iteration (sensitive to far-depth outliers).
          - With `bbox_observe_quantile = q ∈ (0, 0.5)` we take the (q, 1-q)
            quantile per iteration, which clips long-tail outliers from
            low-confidence rays while preserving the legitimate surface
            extent. Across iterations we accumulate the cross-iteration
            min(of-low-quantile) and max(of-high-quantile), giving a bbox
            that tightly fits the union of well-supported surface regions.
        """
        valid = valid.reshape(-1).to(self.device)
        pts = points_world.reshape(-1, 3).to(self.device).float()
        pts = pts[valid]
        # Drop any non-finite rows (NaN/inf depths produce inf points).
        pts = pts[torch.isfinite(pts).all(dim=-1)]
        if pts.numel() == 0:
            return

        q = float(self.config.mask_voxel_cache_bbox_observe_quantile)
        q = min(max(q, 0.0), 0.49)
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

    def _voxel_cache_edit_step(self, step: int) -> int:
        """Return a zero-based edit-local step.

        Nerfstudio's trainer step can resume from the reconstruction checkpoint
        step (for example 30000), but the voxel cache starts empty at the
        beginning of the edit run. Warmup must therefore use edit-local time,
        not the global trainer step.
        """
        if self.mask_voxel_cache_start_step is None:
            self.mask_voxel_cache_start_step = int(step)
        return max(0, int(step) - self.mask_voxel_cache_start_step)

    def _voxel_cache_warmup_blend(self, edit_step: int) -> float:
        """Linear ramp from 0 → max_blend over [warmup_start, warmup_end].

        Same shape as your existing `gradient_mask_warmup` pattern, just
        applied to the external-mask blend rather than to the mask itself.
        """
        s = self.config.mask_voxel_cache_warmup_start
        e = self.config.mask_voxel_cache_warmup_end
        if e <= s:
            return float(self.config.mask_voxel_cache_max_blend)
        progress = max(0, edit_step - s) / max(1, e - s)
        progress = min(max(progress, 0.0), 1.0)
        return float(progress * self.config.mask_voxel_cache_max_blend)

    def get_train_loss_dict(self, step: int):
        loss_dict = dict()

        rendered_image, current_spot, depth_world, accumulation_world = (
            self.get_current_rendering(step)
        )
        # Use the train-camera slot as the unique-view id. The datamanager's
        # `image_idx` can refer to original dataset indices after train/eval
        # splitting, which may be non-contiguous; `current_spot` is guaranteed
        # to be in [0, num_train_views).
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
        external_grad_mask_confidence = None
        external_mask_blend = 0.0
        mask_world_points = None
        mask_world_points_valid = None
        mask_ray_directions = None
        voxel_cache_query_count = None
        voxel_cache_query_variance = None
        voxel_cache_edit_step = None
        if (
            self._voxel_cache_active()
            and depth_world is not None
        ):
            voxel_cache_edit_step = self._voxel_cache_edit_step(step)
            # For non-observed sources, build the cache eagerly (existing flow).
            # For "observed" source, defer building until enough world-point
            # samples are accumulated below.
            if str(self.config.mask_voxel_cache_bbox_source).lower() != "observed":
                self._ensure_voxel_cache()
            mask_h, mask_w = x0.shape[-2:]
            H_cam, W_cam = int(depth_world.shape[0]), int(depth_world.shape[1])

            # Generate rays directly at mask (latent) resolution. Each latent
            # pixel center maps to a sub-pixel position halfway through its
            # (H_cam/mask_h) × (W_cam/mask_w) image footprint. This replaces
            # the previous "render at full resolution, interpolate down"
            # path — bilinear-interpolating unit ray directions then
            # renormalizing produces vectors that don't correspond to any
            # actual ray at the resampled position. Nerfstudio convention:
            # coords[..., 0] is y, coords[..., 1] is x.
            current_index = self.datamanager.image_batch["image_idx"][current_spot]
            current_camera = self.datamanager.train_dataparser_outputs.cameras[
                current_index:current_index + 1
            ].to(self.device)
            yy = (
                torch.arange(mask_h, device=self.device, dtype=torch.float32) + 0.5
            ) * (H_cam / mask_h)
            xx = (
                torch.arange(mask_w, device=self.device, dtype=torch.float32) + 0.5
            ) * (W_cam / mask_w)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            coords = torch.stack([grid_y, grid_x], dim=-1)  # [mask_h, mask_w, 2]
            rays_lr = current_camera.generate_rays(
                camera_indices=0, coords=coords, keep_shape=True
            )
            o = rays_lr.origins.detach()      # [mask_h, mask_w, 3]
            dr = rays_lr.directions.detach()  # [mask_h, mask_w, 3]
            del current_camera, rays_lr

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
            d = F.interpolate(d, size=(mask_h, mask_w), mode="bilinear", align_corners=False)
            acc = F.interpolate(acc, size=(mask_h, mask_w), mode="bilinear", align_corners=False)

            # Reshape rays to [1, 3, mask_h, mask_w] for arithmetic with d, then
            # backproject. World points where each ray reaches `depth_world`.
            o = o.permute(2, 0, 1).unsqueeze(0).to(self.dc_device).float()
            dr = dr.permute(2, 0, 1).unsqueeze(0).to(self.dc_device).float()
            world_points = o + d * dr  # [1, 3, mask_h, mask_w]
            mask_world_points = world_points.permute(0, 2, 3, 1).reshape(-1, 3)  # [N, 3]
            mask_ray_directions = dr.permute(0, 2, 3, 1).reshape(-1, 3)  # [N, 3]
            mask_world_points_valid = (
                (acc > float(self.config.mask_voxel_cache_accumulation_threshold))
                & torch.isfinite(d)
                & (d > 0.0)
            ).reshape(-1)

            if (
                str(self.config.mask_voxel_cache_bbox_source).lower() == "observed"
                and self.mask_voxel_cache is None
            ):
                self._observe_points(mask_world_points, mask_world_points_valid)
                if (
                    self._observed_pts_count
                    >= int(self.config.mask_voxel_cache_bbox_observe_steps)
                ):
                    self._ensure_voxel_cache()
                if self.use_wandb and step % self.config.log_step == 0:
                    import wandb
                    wandb.log(
                        {
                            "dc_debug/voxel_cache_observing_count": int(self._observed_pts_count),
                        },
                        step=step,
                        commit=False,
                    )

            if self.mask_voxel_cache is not None:
                min_observations = (
                    self._effective_voxel_cache_min_observations()
                    if self.config.mask_voxel_cache_confidence_enabled
                    else 1
                )
                self.mask_voxel_cache_effective_min_observations = min_observations
                if self.config.mask_voxel_cache_enabled:
                    (
                        queried,
                        cache_valid,
                        cache_confidence,
                        cache_count,
                        cache_variance,
                    ) = self.mask_voxel_cache.query(
                        mask_world_points,
                        in_bounds=mask_world_points_valid,
                        return_valid=True,
                        return_stats=True,
                        min_observations=min_observations,
                        max_variance=(
                            self.config.mask_voxel_cache_max_variance
                            if self.config.mask_voxel_cache_confidence_enabled
                            else None
                        ),
                        angular_power=float(self.config.mask_voxel_cache_angular_power),
                        min_angular_factor=float(
                            self.config.mask_voxel_cache_min_angular_factor
                        ),
                        angular_relative=bool(
                            self.config.mask_voxel_cache_angular_relative
                        ),
                        mass_threshold=float(
                            self.config.mask_voxel_cache_mass_threshold
                        ),
                        mass_power=float(
                            self.config.mask_voxel_cache_mass_power
                        ),
                        trilinear=bool(
                            getattr(self.config, "mask_voxel_cache_trilinear", False)
                        ),
                    )
                    external_grad_mask = queried.view(1, 1, mask_h, mask_w).to(
                        device=x0.device, dtype=x0.dtype
                    )
                    external_grad_mask_valid = cache_valid.view(1, 1, mask_h, mask_w).to(
                        device=x0.device
                    )
                    external_grad_mask_confidence = cache_confidence.view(1, 1, mask_h, mask_w).to(
                        device=x0.device, dtype=x0.dtype
                    )
                    if getattr(self.config, "mask_voxel_cache_scale_normalize", False):
                        external_grad_mask = self._scale_normalize_cache_mask(
                            external_grad_mask, external_grad_mask_valid
                        )
                    voxel_cache_query_count = cache_count.view(1, 1, mask_h, mask_w)
                    voxel_cache_query_variance = cache_variance.view(1, 1, mask_h, mask_w)
                    external_mask_blend = self._voxel_cache_warmup_blend(voxel_cache_edit_step)

            if self.use_wandb and step % self.config.log_step == 0:
                import wandb
                with torch.no_grad():
                    pts = mask_world_points
                    pts_min = pts.min(dim=0).values
                    pts_max = pts.max(dim=0).values
                    acc_pass = mask_world_points_valid.float().mean()
                    mean_acc = acc.float().mean() if acc is not None else torch.tensor(1.0)
                    payload = {
                        "dc_debug/voxel_pts_min_x": float(pts_min[0]),
                        "dc_debug/voxel_pts_min_y": float(pts_min[1]),
                        "dc_debug/voxel_pts_min_z": float(pts_min[2]),
                        "dc_debug/voxel_pts_max_x": float(pts_max[0]),
                        "dc_debug/voxel_pts_max_y": float(pts_max[1]),
                        "dc_debug/voxel_pts_max_z": float(pts_max[2]),
                        "dc_debug/voxel_acc_filter_pass_frac": float(acc_pass),
                        "dc_debug/voxel_mean_accumulation": float(mean_acc),
                    }
                    if self.mask_voxel_cache is not None:
                        bbox_lo = self.mask_voxel_cache.bbox_min
                        bbox_hi = self.mask_voxel_cache.bbox_max
                        norm = (pts - bbox_lo) / (bbox_hi - bbox_lo).clamp_min(1e-8)
                        pure_in_bbox = ((norm >= 0.0) & (norm < 1.0)).all(dim=-1).float().mean()
                        payload["dc_debug/voxel_pure_in_bbox_frac"] = float(pure_in_bbox)
                    wandb.log(payload, step=step, commit=False)

        dic = self.dc(
            tgt_x0=x0,
            src_x0=src_x0,
            src_encoded=src_encoded,
            return_dict=True,
            step=step,
            current_spot=current_spot,
            external_grad_mask=external_grad_mask,
            external_grad_mask_valid=external_grad_mask_valid,
            external_grad_mask_confidence=external_grad_mask_confidence,
            external_mask_blend=external_mask_blend,
        )

        if (
            self._voxel_cache_active()
            and self.mask_voxel_cache is not None
            and mask_world_points is not None
            and dic.get("internal_grad_mask", None) is not None
        ):
            update_source = str(self.config.mask_voxel_cache_update_source).lower()
            if update_source not in {"internal", "raw_self", "raw_attn"}:
                raise ValueError(
                    f"Unknown mask_voxel_cache_update_source={update_source!r}; "
                    "expected 'internal', 'raw_self', or 'raw_attn'."
                )
            if update_source == "internal" and not getattr(
                self, "_warned_internal_update_source", False
            ):
                print(
                    "[voxel-cache] WARNING: mask_voxel_cache_update_source='internal' "
                    "is deprecated. The CA-mask schedule leaks into cache "
                    "observations. Prefer 'raw_self' for cleaner cross-view "
                    "aggregation. (This warning fires once per run.)"
                )
                self._warned_internal_update_source = True
            if update_source == "raw_attn" and dic.get("cross_attention_mask") is not None:
                cache_mask_input = dic["cross_attention_mask"]
                if cache_mask_input.shape[-2:] != (mask_h, mask_w):
                    cache_mask_input = F.interpolate(
                        cache_mask_input.float(),
                        size=(mask_h, mask_w),
                        mode="bilinear",
                        align_corners=False,
                    )
            elif update_source == "raw_self" and dic.get("self_grad_mask_raw") is not None:
                cache_mask_input = dic["self_grad_mask_raw"]
            else:
                cache_mask_input = dic["internal_grad_mask"]
            new_mask = cache_mask_input.detach().to(self.device).reshape(-1)
            log_this_step = (
                self.use_wandb and step % self.config.log_step == 0
            )
            per_voxel_mean_snapshot = self.mask_voxel_cache.update(
                mask_world_points,
                new_mask,
                in_bounds=mask_world_points_valid,
                value_threshold=self.config.mask_voxel_cache_update_threshold,
                view_id=current_view_id,
                return_per_voxel_mean=log_this_step,
                ray_directions=mask_ray_directions,
            )
            if (
                self.config.mask_voxel_cache_angular_relative
                and not self.mask_voxel_cache.angular_denominator_is_frozen
            ):
                edit_step_for_freeze = (
                    voxel_cache_edit_step
                    if voxel_cache_edit_step is not None
                    else self._voxel_cache_edit_step(step)
                )
                freeze_min_views = (
                    self.mask_voxel_cache_effective_min_observations
                    if self.mask_voxel_cache_effective_min_observations is not None
                    else self._effective_voxel_cache_min_observations()
                )
                frozen_value = (
                    self.mask_voxel_cache.try_auto_freeze_angular_denominator(
                        edit_step=edit_step_for_freeze,
                        min_views=freeze_min_views,
                        patience=int(
                            self.config.mask_voxel_cache_angular_freeze_patience
                        ),
                        warmup_steps=int(
                            self.config.mask_voxel_cache_angular_freeze_warmup
                        ),
                    )
                )
                if frozen_value is not None:
                    peak_step = self.mask_voxel_cache.angular_peak_step
                    print(
                        f"[voxel-cache] auto-froze angular denominator at "
                        f"edit_step={edit_step_for_freeze} (global={step}): "
                        f"value={frozen_value:.5f} captured at peak "
                        f"edit_step={peak_step} (min_views={freeze_min_views})"
                    )
            if log_this_step:
                import wandb
                valid_ratio = (
                    float(external_grad_mask_valid.float().mean().item())
                    if external_grad_mask_valid is not None
                    else 0.0
                )
                payload = {
                    "dc_debug/voxel_cache_occupancy": float(self.mask_voxel_cache.occupancy),
                    "dc_debug/voxel_cache_blend": float(external_mask_blend),
                    "dc_debug/voxel_cache_measure_only": (
                        1.0
                        if (
                            getattr(self.config, "mask_voxel_cache_measure_only", False)
                            and not self.config.mask_voxel_cache_enabled
                        )
                        else 0.0
                    ),
                    "dc_debug/voxel_cache_valid_ratio": valid_ratio,
                    "dc_debug/voxel_cache_edit_step": float(voxel_cache_edit_step or 0),
                    "dc_debug/voxel_cache_ema_beta": float(
                        self.mask_voxel_cache_effective_ema_beta
                        if self.mask_voxel_cache_effective_ema_beta is not None
                        else self.mask_voxel_cache.ema_beta
                    ),
                    "dc_debug/voxel_cache_mean_observation_count": float(
                        self.mask_voxel_cache.mean_observation_count
                    ),
                    "dc_debug/voxel_cache_mean_geom_observation_count": float(
                        self.mask_voxel_cache.mean_geom_observation_count
                    ),
                    "dc_debug/voxel_cache_mean_observed_variance": float(
                        self.mask_voxel_cache.mean_observed_variance
                    ),
                    "dc_debug/voxel_cache_mean_observed_variance_peak": float(
                        self.mask_voxel_cache.mean_observed_variance_peak
                    ),
                    "dc_debug/voxel_cache_mean_angular_factor": float(
                        self.mask_voxel_cache.mean_angular_factor
                    ),
                    "dc_debug/voxel_cache_mean_angular_factor_trusted": float(
                        self.mask_voxel_cache.mean_angular_factor_at(
                            min_views=min_observations
                        )
                    ),
                    "dc_debug/voxel_cache_angular_denominator_frozen": float(
                        self.mask_voxel_cache.frozen_angular_denominator
                        if self.mask_voxel_cache.frozen_angular_denominator is not None
                        else 0.0
                    ),
                    "dc_debug/voxel_cache_angular_peak_value": float(
                        self.mask_voxel_cache.angular_peak_value
                    ),
                    "dc_debug/voxel_cache_angular_peak_step": float(
                        self.mask_voxel_cache.angular_peak_step
                    ),
                    "dc_debug/voxel_cache_min_observations": float(
                        self.mask_voxel_cache_effective_min_observations
                        if self.mask_voxel_cache_effective_min_observations is not None
                        else self._effective_voxel_cache_min_observations()
                    ),
                }
                if (
                    per_voxel_mean_snapshot is not None
                    and per_voxel_mean_snapshot.numel() > 0
                ):
                    snap = per_voxel_mean_snapshot.detach().float().cpu()
                    payload["dc_debug/voxel_cache_update_hist"] = wandb.Histogram(
                        snap.numpy(), num_bins=50
                    )
                    threshold = float(
                        self.config.mask_voxel_cache_update_threshold
                    )
                    payload["dc_debug/voxel_cache_update_mean_pre_gate"] = float(
                        snap.mean().item()
                    )
                    payload["dc_debug/voxel_cache_update_above_threshold_frac"] = float(
                        (snap >= threshold).float().mean().item()
                    ) if threshold > 0.0 else 1.0
                src_2d = new_mask.detach().float().cpu()
                if src_2d.numel() > 0:
                    payload["dc_debug/voxel_cache_input_mask_hist"] = wandb.Histogram(
                        src_2d.numpy(), num_bins=50
                    )
                    payload["dc_debug/voxel_cache_input_mask_mean"] = float(
                        src_2d.mean().item()
                    )
                    payload["dc_debug/voxel_cache_input_mask_source"] = (
                        2.0 if update_source == "raw_attn"
                        else 1.0 if update_source == "raw_self"
                        else 0.0
                    )
                wandb.log(payload, step=step, commit=False)
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
            if self.use_wandb:
                import wandb
                voxel_mask_stats = external_grad_mask.detach().float().clamp(0.0, 1.0)
                wandb.log(
                    {
                        "dc_debug/voxel_cache_mask": wandb.Image(
                            TF.resize(voxel_mask_img, min_size),
                            caption=f"step={step} | queried voxel-cache mask",
                        ),
                        "dc_debug/voxel_cache_mask_mean": float(voxel_mask_stats.mean().item()),
                        "dc_debug/voxel_cache_mask_max": float(voxel_mask_stats.max().item()),
                        "dc_debug/voxel_cache_mask_coverage_0.5": float(
                            (voxel_mask_stats > 0.5).float().mean().item()
                        ),
                    },
                    step=step,
                    commit=False,
                )

        if external_grad_mask_confidence is not None and step % self.config.log_step == 0:
            voxel_conf_vis = F.interpolate(
                external_grad_mask_confidence.detach().float().cpu(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            voxel_conf_img = Image.fromarray((voxel_conf_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
            voxel_conf_img.save(self.base_dir / f"logging/{step}_voxel_cache_confidence.png")
            if self.use_wandb:
                import wandb
                conf = external_grad_mask_confidence.detach().float().clamp(0.0, 1.0)
                payload = {
                    "dc_debug/voxel_cache_confidence": wandb.Image(
                        TF.resize(voxel_conf_img, min_size),
                        caption=f"step={step} | confidence from count + variance",
                    ),
                    "dc_debug/voxel_cache_confidence_mean": float(conf.mean().item()),
                    "dc_debug/voxel_cache_confidence_coverage_0.5": float(
                        (conf > 0.5).float().mean().item()
                    ),
                }
                if voxel_cache_query_count is not None:
                    payload["dc_debug/voxel_cache_query_count_mean"] = float(
                        voxel_cache_query_count.detach().float().mean().item()
                    )
                    n_views = max(
                        1,
                        len(self.datamanager.train_dataparser_outputs.image_filenames),
                    )
                    query_count = voxel_cache_query_count.detach().float()
                    payload["dc_debug/voxel_cache_query_view_frac_mean"] = float(
                        (query_count / float(n_views)).mean().item()
                    )
                    payload["dc_debug/voxel_cache_query_count_max"] = float(
                        query_count.max().item()
                    )
                if voxel_cache_query_variance is not None:
                    payload["dc_debug/voxel_cache_query_variance_mean"] = float(
                        voxel_cache_query_variance.detach().float().mean().item()
                    )
                    max_var_gate = float(
                        getattr(self.config, "mask_voxel_cache_max_variance", 0.0)
                    )
                    var_scale = max_var_gate if max_var_gate > 1e-8 else 1.0
                    voxel_var_vis = F.interpolate(
                        (voxel_cache_query_variance.detach().float().cpu() / var_scale).clamp(0.0, 1.0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    voxel_var_img = Image.fromarray(
                        (voxel_var_vis[0, 0].numpy() * 255).astype(np.uint8)
                    )
                    voxel_var_img.save(self.base_dir / f"logging/{step}_voxel_cache_variance.png")
                    peak_held = (
                        float(
                            getattr(
                                self.config,
                                "mask_voxel_cache_variance_peak_decay",
                                0.0,
                            )
                        )
                        > 0.0
                    )
                    var_kind = "peak-held cross-view variance" if peak_held else "cross-view variance"
                    payload["dc_debug/voxel_cache_variance_map"] = wandb.Image(
                        TF.resize(voxel_var_img, min_size),
                        caption=f"step={step} | {var_kind} / max_variance (bright = views disagree; >= gate is damped)",
                    )
                if external_grad_mask is not None:
                    trusted = (
                        external_grad_mask.detach().float().clamp(0.0, 1.0)
                        * external_grad_mask_confidence.detach().float().clamp(0.0, 1.0)
                    )
                    trusted_vis = F.interpolate(
                        trusted.cpu(),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    trusted_img = Image.fromarray(
                        (trusted_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
                    )
                    trusted_img.save(self.base_dir / f"logging/{step}_voxel_cache_trusted_mask.png")
                    payload["dc_debug/voxel_cache_trusted_mask"] = wandb.Image(
                        TF.resize(trusted_img, min_size),
                        caption=f"step={step} | voxel mask times confidence",
                    )
                    payload["dc_debug/voxel_cache_trusted_mask_mean"] = float(trusted.mean().item())
                    payload["dc_debug/voxel_cache_trusted_mask_coverage_0.5"] = float(
                        (trusted > 0.5).float().mean().item()
                    )
                    internal_mask = dic.get("internal_grad_mask", None)
                    if internal_mask is not None:
                        internal_mask = internal_mask.detach().float().clamp(0.0, 1.0)
                        cache_mask = external_grad_mask.detach().float().clamp(0.0, 1.0)
                        cache_conf = external_grad_mask_confidence.detach().float().clamp(0.0, 1.0)
                        if cache_mask.shape[-2:] != internal_mask.shape[-2:]:
                            cache_mask = F.interpolate(
                                cache_mask,
                                size=internal_mask.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        if cache_conf.shape[-2:] != internal_mask.shape[-2:]:
                            cache_conf = F.interpolate(
                                cache_conf,
                                size=internal_mask.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        negative_correction = (internal_mask - cache_mask).clamp_min(0.0) * cache_conf
                        negative_correction_vis = F.interpolate(
                            negative_correction.cpu(),
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        negative_correction_img = Image.fromarray(
                            (negative_correction_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
                        )
                        negative_correction_img.save(
                            self.base_dir / f"logging/{step}_voxel_cache_negative_correction.png"
                        )
                        payload["dc_debug/voxel_cache_negative_correction"] = wandb.Image(
                            TF.resize(negative_correction_img, min_size),
                            caption=f"step={step} | max(0, internal - cache) times confidence",
                        )
                        payload["dc_debug/voxel_cache_negative_correction_mean"] = float(
                            negative_correction.mean().item()
                        )
                        payload["dc_debug/voxel_cache_negative_correction_coverage_0.05"] = float(
                            (negative_correction > 0.05).float().mean().item()
                        )
                wandb.log(payload, step=step, commit=False)

        if external_grad_mask_valid is not None and step % self.config.log_step == 0:
            voxel_valid_vis = F.interpolate(
                external_grad_mask_valid.detach().float().cpu(),
                size=(h, w),
                mode="nearest",
            )
            voxel_valid_img = Image.fromarray((voxel_valid_vis[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8))
            voxel_valid_img.save(self.base_dir / f"logging/{step}_voxel_cache_valid.png")
            if self.use_wandb:
                import wandb
                wandb.log(
                    {
                        "dc_debug/voxel_cache_valid_mask": wandb.Image(
                            TF.resize(voxel_valid_img, min_size),
                            caption=f"step={step} | voxel-cache observed/in-bounds mask",
                        ),
                    },
                    step=step,
                    commit=False,
                )

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
