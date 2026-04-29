"""3D voxel-grid cache for cross-view mask aggregation.

Stage-1 prototype of the "lift the mask into 3D" idea. Per-view diffusion
masks are projected into a coarse voxel grid via the rendered depth, EMA-
aggregated across views, and queried back per-view to provide a
3D-consistent localization signal to the DDS gradient.

References (where each piece comes from):
    - Volume rendering equation (the depth-along-ray semantics that lets
      us backproject pixel + depth → 3D point):
      Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance
      Fields for View Synthesis," ECCV 2020. Eq. 1–3.
    - Pinhole-camera depth-backprojection: standard Hartley & Zisserman
      "Multiple View Geometry" pinhole model.
    - Lifting per-view 2D signals into a 3D scene-level field for cross-
      view aggregation: Semantic NeRF (Zhi et al., ICCV 2021, §3.2 — the
      "3D-consistent semantics from 2D supervision" argument). The voxel
      cache is the non-parametric / cache-based variant of this idea.
    - Per-frame label accumulation into a 3D scene field via running
      averages: Panoptic Lifting (Siddiqui et al., CVPR 2023, §3.1).
    - Closest direct precedent for diffusion-derived mask aggregation
      into a 3D representation: RoMaP (Kim et al., SNU 2025), which
      attaches per-Gaussian mask attributes for masked SDS editing in
      3DGS. The voxel-grid form here is the NeRF-friendly cousin.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


class MaskVoxelCache:
    """Coarse 3D voxel grid storing EMA-aggregated mask values.

    Stage-1 prototype: no learned MLP, no gradient flow through the cache.
    Mask values come from per-view diffusion-derived masks, are scattered
    into the grid via depth backprojection, and read back into image space
    via the same backprojection on subsequent views. By construction, two
    views observing the same 3D point read the same cached mask value, so
    the rendered mask is cross-view-consistent up to voxel-grid
    discretization.
    """

    def __init__(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        resolution: int = 128,
        ema_beta: float = 0.9,
        fallback_value: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            bbox_min, bbox_max: [3] world-space scene bounding box.
                Read from `dataparser_outputs.scene_box.aabb`.
            resolution: per-axis grid resolution. 128 → 128**3 ≈ 2M voxels,
                ~8MB at fp32. Coarser = cheaper but blurrier across-views.
            ema_beta: EMA decay applied to existing voxel values. Higher
                values mean slower change. Standard value 0.9 (matches the
                running-average pattern used in Panoptic Lifting §3.1).
            fallback_value: returned for voxels never observed. 0.5 is
                "uncertain"; safer than 0.0 (would over-suppress edits in
                unseen regions) or 1.0 (would over-edit them).
            device: torch device. Defaults to current CUDA device.
        """
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bbox_min = bbox_min.to(self.device).float()
        self.bbox_max = bbox_max.to(self.device).float()
        self.resolution = int(resolution)
        self.ema_beta = float(ema_beta)
        self.fallback_value = float(fallback_value)

        V = self.resolution
        # Grid of mask values, initialized to the fallback. Shape [V, V, V].
        self.grid = torch.full(
            (V, V, V),
            fill_value=self.fallback_value,
            device=self.device,
            dtype=torch.float32,
        )
        # Tracks which voxels have been observed at least once.
        self.observed = torch.zeros(
            (V, V, V),
            dtype=torch.bool,
            device=self.device,
        )

    # -----------------------------------------------------------------
    # Spatial coordinate conversion
    # -----------------------------------------------------------------

    def _world_to_voxel(self, points_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map world-space points to voxel indices.

        Uses the standard normalized-grid mapping
            idx = floor((x - bbox_min) / (bbox_max - bbox_min) * V).
        Out-of-bounds points are clamped to the grid surface.

        Returns:
            idx: [..., 3] long tensor of voxel indices in [0, V).
            in_bounds: [...] bool tensor flagging which points fell inside
                the bounding box before clamping. Useful for ignoring
                far-away rays (sky / infinite depth).
        """
        normalized = (points_world - self.bbox_min) / (
            self.bbox_max - self.bbox_min
        ).clamp_min(1e-8)
        in_bounds = ((normalized >= 0.0) & (normalized < 1.0)).all(dim=-1)
        idx = (normalized * self.resolution).long().clamp_(0, self.resolution - 1)
        return idx, in_bounds

    # -----------------------------------------------------------------
    # Depth-backprojection helper
    # -----------------------------------------------------------------

    @staticmethod
    def backproject_via_rays(
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Backproject (origin, direction, depth) → world-space 3D points.

        Uses the parametric ray equation r(t) = o + t·d. This is exactly
        the form used inside NeRF's volume rendering (Mildenhall et al.,
        ECCV 2020, Eq. 1) and avoids any camera-convention ambiguity
        because we consume Nerfstudio's already-world-space rays.

        Args:
            ray_origins: [H, W, 3] world-space ray origins.
            ray_directions: [H, W, 3] world-space ray directions
                (unit-norm or not — Nerfstudio rays are unit-norm).
            depth: [H, W] or [H, W, 1] depth along the ray.

        Returns:
            [H, W, 3] world-space points where each ray reaches `depth`.
        """
        if depth.dim() == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        return ray_origins + depth.unsqueeze(-1) * ray_directions

    # -----------------------------------------------------------------
    # Update — scatter per-view mask into voxels
    # -----------------------------------------------------------------

    def update(
        self,
        points_world: torch.Tensor,
        mask_values: torch.Tensor,
        in_bounds: Optional[torch.Tensor] = None,
    ) -> None:
        """EMA-update voxels at backprojected 3D positions.

        Within-batch duplicates (multiple pixels falling in the same voxel
        in this view) are averaged before the EMA step. This avoids the
        last-write-wins indexing artifact and matches the per-frame label-
        accumulation pattern in Panoptic Lifting (Siddiqui et al., CVPR
        2023, §3.1).

        Voxels observed for the first time take the input value directly
        (no EMA blending against the fallback).

        Args:
            points_world: [N, 3] world-space points.
            mask_values: [N] mask values in [0, 1].
            in_bounds: [N] optional bool mask; if provided, points marked
                False are skipped.
        """
        points_world = points_world.reshape(-1, 3).to(self.device)
        mask_values = mask_values.reshape(-1).to(self.device).float().clamp(0.0, 1.0)
        if points_world.numel() == 0:
            return

        idx, default_in_bounds = self._world_to_voxel(points_world)  # [N, 3], [N]
        if in_bounds is None:
            valid = default_in_bounds
        else:
            valid = in_bounds.reshape(-1).to(self.device).bool() & default_in_bounds
        idx = idx[valid]
        mask_values = mask_values[valid]
        if idx.numel() == 0:
            return

        V = self.resolution
        flat = idx[:, 0] * V * V + idx[:, 1] * V + idx[:, 2]  # [N]

        # Within-batch averaging: per-voxel mean of mask values from this view.
        # Equivalent to scatter_reduce(reduce="mean") but using sum / count for
        # backwards-compatibility across PyTorch versions.
        n_voxels = V * V * V
        sums = torch.zeros(n_voxels, device=self.device, dtype=torch.float32)
        counts = torch.zeros(n_voxels, device=self.device, dtype=torch.float32)
        sums.index_add_(0, flat, mask_values)
        counts.index_add_(0, flat, torch.ones_like(mask_values))

        touched = counts > 0  # [n_voxels]
        if not touched.any():
            return
        per_voxel_mean = torch.zeros_like(sums)
        per_voxel_mean[touched] = sums[touched] / counts[touched]

        # EMA blend into existing grid for already-observed voxels;
        # direct copy for first-time-observed voxels.
        flat_grid = self.grid.view(-1)
        flat_observed = self.observed.view(-1)

        existing_and_touched = touched & flat_observed
        first_time = touched & (~flat_observed)

        if existing_and_touched.any():
            beta = self.ema_beta
            flat_grid[existing_and_touched] = (
                beta * flat_grid[existing_and_touched]
                + (1.0 - beta) * per_voxel_mean[existing_and_touched]
            )
        if first_time.any():
            flat_grid[first_time] = per_voxel_mean[first_time]
            flat_observed[first_time] = True

    # -----------------------------------------------------------------
    # Query — read voxels back at backprojected 3D positions
    # -----------------------------------------------------------------

    def query(
        self,
        points_world: torch.Tensor,
        in_bounds: Optional[torch.Tensor] = None,
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Look up mask values at world-space positions.

        Returns `fallback_value` for voxels never observed. Out-of-bounds
        points (if `in_bounds` is provided) are also returned as fallback.
        If `return_valid=True`, also returns a bool tensor indicating which
        lookups came from observed, in-bounds voxels.
        """
        original_shape = points_world.shape[:-1]
        flat_points = points_world.reshape(-1, 3).to(self.device)
        idx, default_in_bounds = self._world_to_voxel(flat_points)
        if in_bounds is None:
            in_bounds = default_in_bounds
        else:
            in_bounds = in_bounds.reshape(-1).to(self.device) & default_in_bounds

        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
        values = self.grid[ix, iy, iz]
        observed = self.observed[ix, iy, iz]

        # Replace unobserved or out-of-bounds with fallback.
        valid = observed & in_bounds
        values = torch.where(
            valid, values, torch.full_like(values, self.fallback_value)
        )
        values = values.view(*original_shape)
        if return_valid:
            return values, valid.view(*original_shape)
        return values

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    @property
    def occupancy(self) -> float:
        """Fraction of voxels that have been observed at least once."""
        return float(self.observed.float().mean().item())
