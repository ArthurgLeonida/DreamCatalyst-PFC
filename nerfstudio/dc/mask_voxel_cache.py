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
        num_views: Optional[int] = None,
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
        self.num_views = int(num_views) if num_views is not None and int(num_views) > 0 else None

        V = self.resolution
        n_voxels = V * V * V
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
        # Per-voxel cross-view consistency statistics. Each update contributes
        # one per-view mean mask value per touched voxel. Welford statistics
        # let us estimate disagreement between different views that hit the
        # same 3D location without storing a history.
        self.update_count = torch.zeros(
            (V, V, V),
            dtype=torch.int32,
            device=self.device,
        )
        self.unique_view_count = torch.zeros(
            (V, V, V),
            dtype=torch.int32,
            device=self.device,
        )
        self.view_observed = (
            torch.zeros(
                (n_voxels, self.num_views),
                dtype=torch.bool,
                device=self.device,
            )
            if self.num_views is not None
            else None
        )
        self.running_mean = torch.zeros(
            (V, V, V),
            device=self.device,
            dtype=torch.float32,
        )
        self.running_m2 = torch.zeros(
            (V, V, V),
            device=self.device,
            dtype=torch.float32,
        )
        # Sum of unit ray directions over all observations for each voxel.
        # `‖view_dir_sum / unique_view_count‖` is the resultant length of a
        # set of unit vectors; it tends to 1 when all observations come from
        # parallel rays (no angular diversity) and to 0 when observations are
        # spherically distributed (maximal diversity). This is the standard
        # circular-statistics resultant-length measure (Fisher, 1953, "Dispersion
        # on a sphere"). We use 1 − resultant_length as the per-voxel angular
        # coverage factor in `query`, which down-weights confidence on voxels
        # observed only from a narrow cone of viewpoints — the failure mode
        # diagnosed empirically on elf (65 cameras, but voxels overwhelmingly
        # observed from clustered directions; cross-view variance ≈ 0.005 with
        # near-saturated confidence but minimal real correspondence info).
        self.view_dir_sum = torch.zeros(
            (V, V, V, 3),
            device=self.device,
            dtype=torch.float32,
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
        value_threshold: float = 0.0,
        view_id: Optional[int] = None,
        return_per_voxel_mean: bool = False,
        ray_directions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
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
            value_threshold: optional confidence gate, applied to the
                per-voxel mean of mask values from this view (not to
                raw pixel values). Voxels whose averaged evidence falls
                below the threshold are skipped — neither the EMA grid
                nor the Welford cross-view statistics are updated from
                this view. Prevents stale low priors from accumulating
                in voxels where the model hasn't yet started editing
                (e.g. stormtrooper helmet voxels during iters 0–1400,
                where the per-view mask is low because the model is
                busy on the body) and prevents those early under-
                edited samples from inflating the running variance.
                Gating the per-voxel mean (rather than raw pixels)
                ensures that a voxel only contributes a variance sample
                when this view's evidence for that voxel is itself
                confident. Set 0.0 to disable.
            view_id: optional integer camera index. When provided and
                the cache was constructed with `num_views`, the Welford
                cross-view statistics increment only the first time
                this view observes each voxel — preventing repeat
                visits to the same training view from inflating the
                effective sample size and deflating variance.
            return_per_voxel_mean: if True, returns the per-voxel mean
                values (one scalar per touched voxel) BEFORE the
                value_threshold gate filters them out. Intended for
                histogram diagnostics — lets the caller see the full
                distribution of evidence this view contributed,
                including the values the threshold rejected. Returns
                None otherwise.
        """
        points_world = points_world.reshape(-1, 3).to(self.device)
        mask_values = mask_values.reshape(-1).to(self.device).float().clamp(0.0, 1.0)
        empty_return: Optional[torch.Tensor] = (
            mask_values.new_empty(0) if return_per_voxel_mean else None
        )
        if points_world.numel() == 0:
            return empty_return

        idx, default_in_bounds = self._world_to_voxel(points_world)  # [N, 3], [N]
        if in_bounds is None:
            valid = default_in_bounds
        else:
            valid = in_bounds.reshape(-1).to(self.device).bool() & default_in_bounds
        if ray_directions is not None:
            rd_full = ray_directions.reshape(-1, 3).to(self.device).float()
            rd_full = rd_full / rd_full.norm(dim=1, keepdim=True).clamp_min(1e-8)
            rd_filtered = rd_full[valid]
        else:
            rd_filtered = None

        idx = idx[valid]
        mask_values = mask_values[valid]
        if idx.numel() == 0:
            return empty_return

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

        # Per-voxel mean ray direction from this view. Aggregate the unit
        # ray directions of all pixels mapping to each voxel and renormalize.
        # The result represents "the direction this view observed this voxel
        # from"; the running sum across views drives the angular-diversity
        # factor in `query`.
        per_voxel_view_dir: Optional[torch.Tensor] = None
        if rd_filtered is not None:
            dir_sums = torch.zeros(n_voxels, 3, device=self.device, dtype=torch.float32)
            dir_sums.index_add_(0, flat, rd_filtered)
            dir_norms = dir_sums.norm(dim=1, keepdim=True).clamp_min(1e-8)
            per_voxel_view_dir = dir_sums / dir_norms  # [n_voxels, 3] unit vectors

        touched = counts > 0  # [n_voxels]
        if not touched.any():
            return empty_return
        per_voxel_mean = torch.zeros_like(sums)
        per_voxel_mean[touched] = sums[touched] / counts[touched]
        # Pre-gate snapshot for diagnostics: per-voxel mean values from this
        # view BEFORE the value_threshold gate filters them out.
        pre_gate_snapshot = (
            per_voxel_mean[touched].detach().clone()
            if return_per_voxel_mean
            else None
        )

        # Confidence gate: only learn from per-voxel evidence above the
        # threshold. The gate is on the per-voxel mean (the actual sample
        # this view contributes to the cache and to the cross-view
        # statistics) rather than on raw pixel values — pixel-level gating
        # would still let a voxel with mostly-low evidence contribute a
        # poisoned sample after the per-pixel filter. Gating the per-voxel
        # mean prevents early-iteration under-edited views (e.g. helmet
        # voxels during iters 0–1400 where the model hasn't started
        # editing yet) from inflating the running variance.
        if value_threshold > 0.0:
            confident_voxels = per_voxel_mean >= float(value_threshold)
            touched = touched & confident_voxels
            if not touched.any():
                return pre_gate_snapshot if return_per_voxel_mean else None

        # EMA blend into existing grid for already-observed voxels;
        # direct copy for first-time-observed voxels.
        flat_grid = self.grid.view(-1)
        flat_observed = self.observed.view(-1)
        flat_update_count = self.update_count.view(-1)
        flat_unique_view_count = self.unique_view_count.view(-1)
        flat_running_mean = self.running_mean.view(-1)
        flat_running_m2 = self.running_m2.view(-1)

        touched_idx = touched.nonzero(as_tuple=True)[0]
        touched_values = per_voxel_mean[touched_idx]
        flat_update_count[touched_idx] += 1

        stats_idx = touched_idx
        stats_values = touched_values
        if self.view_observed is not None and view_id is not None:
            vid = int(view_id)
            if 0 <= vid < self.num_views:
                is_new_view = ~self.view_observed[touched_idx, vid]
                if is_new_view.any():
                    stats_idx = touched_idx[is_new_view]
                    stats_values = touched_values[is_new_view]
                    self.view_observed[stats_idx, vid] = True
                    flat_unique_view_count[stats_idx] += 1
                else:
                    stats_idx = touched_idx[:0]
                    stats_values = touched_values[:0]
        else:
            flat_unique_view_count[touched_idx] += 1

        # Cross-view correspondence statistics: every time a view maps to a
        # voxel, compare its per-view mean against previous observations of
        # that same voxel. Low variance means the same 3D location receives
        # consistent mask evidence across views; high variance means this voxel
        # is unreliable for suppressing or amplifying the 2D mask.
        if stats_idx.numel() > 0:
            old_count = (flat_unique_view_count[stats_idx].float() - 1.0).clamp_min(0.0)
            new_count = old_count + 1.0
            old_mean = flat_running_mean[stats_idx]
            delta = stats_values - old_mean
            new_mean = old_mean + delta / new_count
            delta2 = stats_values - new_mean
            flat_running_mean[stats_idx] = new_mean
            flat_running_m2[stats_idx] = flat_running_m2[stats_idx] + delta * delta2

        # Angular-diversity statistics: accumulate the per-voxel mean unit
        # ray direction from this view into a per-voxel running sum, but
        # only when this view is genuinely new for that voxel (same gating
        # as the Welford stats above). The resultant length
        # ‖sum / unique_view_count‖ ∈ [0, 1] measures clustering of the
        # observing directions: 1 = parallel rays (no angular diversity),
        # 0 = spherically uniform (maximal diversity). 1 − resultant_length
        # becomes the per-voxel angular coverage factor in `query`.
        if per_voxel_view_dir is not None and stats_idx.numel() > 0:
            flat_view_dir_sum = self.view_dir_sum.view(-1, 3)
            flat_view_dir_sum[stats_idx] = (
                flat_view_dir_sum[stats_idx] + per_voxel_view_dir[stats_idx]
            )

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

        return pre_gate_snapshot if return_per_voxel_mean else None

    # -----------------------------------------------------------------
    # Query — read voxels back at backprojected 3D positions
    # -----------------------------------------------------------------

    def query(
        self,
        points_world: torch.Tensor,
        in_bounds: Optional[torch.Tensor] = None,
        return_valid: bool = False,
        return_stats: bool = False,
        min_observations: int = 1,
        max_variance: Optional[float] = None,
        angular_power: float = 0.0,
        min_angular_factor: float = 0.0,
        angular_relative: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Look up mask values at world-space positions.

        Returns `fallback_value` for voxels never observed. Out-of-bounds
        points (if `in_bounds` is provided) are also returned as fallback.
        If `return_valid=True`, also returns a bool tensor indicating which
        lookups came from observed, in-bounds voxels.
        If `return_stats=True`, also returns a confidence map, observation
        counts, and cross-view variance estimates. Confidence is zero until a
        voxel has at least `min_observations`; if `max_variance` is provided,
        it then decays linearly to zero as variance approaches that threshold.

        `angular_power` (≥ 0) raises the per-voxel angular-diversity factor
        `(1 − ‖mean_view_dir‖)` to that power before multiplying into the
        confidence. 0.0 disables the angular gate. 1.0 multiplies linearly.
        Larger values make the gate steeper (only voxels observed from very
        widely-separated viewpoints stay confident). `min_angular_factor`
        sets a floor so the gate never collapses confidence to zero on
        single-observation voxels (where the resultant length is 1 by
        construction and the factor would otherwise be 0).

        `angular_relative` (default False) switches the factor from absolute
        to scene-relative. With it enabled, each voxel's factor is divided
        by the scene-wide mean angular factor (over all observed voxels with
        ≥ 2 unique views), then clamped to [0, 1]. This is what makes the
        gate work consistently across capture geometries: a captured-on-a-
        horizontal-arc scene (clown) has an absolute mean factor of ~0.05
        and a forward-facing capture (elf) has ~0.01; without normalization
        both look "almost zero" and the gate collapses the cache on both.
        With normalization, each voxel is judged against its own scene's
        typical diversity — voxels above the scene mean keep most of their
        confidence, voxels below get damped, regardless of the absolute
        scale set by the capture rig.
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
        counts = self.unique_view_count[ix, iy, iz].float()
        m2 = self.running_m2[ix, iy, iz]
        variance = torch.where(
            counts > 1.0,
            m2 / (counts - 1.0).clamp_min(1.0),
            torch.zeros_like(m2),
        )

        # Replace unobserved or out-of-bounds with fallback.
        valid = observed & in_bounds
        min_observations = max(int(min_observations), 1)
        enough_views = counts >= float(min_observations)
        if max_variance is not None and float(max_variance) > 0.0:
            var_confidence = (1.0 - variance / float(max_variance)).clamp(0.0, 1.0)
        else:
            var_confidence = torch.ones_like(values)

        # Angular-diversity factor. The resultant length of the per-voxel
        # sum of unit view-directions, divided by the count, lies in [0, 1].
        # 1 means "all observations came from parallel rays" — the cache has
        # no real correspondence evidence and should be down-trusted.
        # 0 means "observations spread uniformly on the sphere" — maximal
        # information from triangulation. (1 − resultant) is the diversity.
        if float(angular_power) > 0.0:
            dir_sum = self.view_dir_sum[ix, iy, iz]  # [N, 3]
            safe_counts = counts.clamp_min(1.0)
            resultant_len = (dir_sum.norm(dim=-1) / safe_counts).clamp(0.0, 1.0)
            angular_factor = (1.0 - resultant_len).clamp(0.0, 1.0)

            # Scene-relative normalization. Cameras for a given dataset live
            # on a roughly-fixed-shape rig (e.g. a horizontal orbit), so the
            # absolute angular factor saturates near a scene-specific
            # ceiling. Dividing by that ceiling (here, the scene mean over
            # multi-view voxels) gives a factor that is comparable across
            # scenes: ratio > 1 means "better triangulated than average for
            # this scene"; ratio < 1 means "worse." Without this, a clown
            # voxel with factor 0.06 looks identically untrustworthy to an
            # elf voxel with factor 0.01, even though clown's voxel is at
            # the high end of its scene's distribution.
            if angular_relative:
                scene_mean = float(self.mean_angular_factor)
                if scene_mean > 1e-6:
                    angular_factor = (angular_factor / scene_mean).clamp(0.0, 1.0)

            if float(min_angular_factor) > 0.0:
                floor = float(min_angular_factor)
                angular_factor = angular_factor.clamp_min(floor)
            angular_confidence = angular_factor.pow(float(angular_power))
        else:
            angular_confidence = torch.ones_like(values)

        confidence = (
            valid.float()
            * enough_views.float()
            * var_confidence
            * angular_confidence
        )

        values = torch.where(
            valid, values, torch.full_like(values, self.fallback_value)
        )
        values = values.view(*original_shape)
        confidence = confidence.view(*original_shape)
        counts = counts.view(*original_shape)
        variance = variance.view(*original_shape)
        if return_valid:
            valid = valid.view(*original_shape)
            if return_stats:
                return values, valid, confidence, counts, variance
            return values, valid
        if return_stats:
            return values, confidence, counts, variance
        return values

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    @property
    def occupancy(self) -> float:
        """Fraction of voxels that have been observed at least once."""
        return float(self.observed.float().mean().item())

    @property
    def mean_observation_count(self) -> float:
        """Mean number of unique camera observations over observed voxels."""
        observed = self.observed
        if not observed.any():
            return 0.0
        return float(self.unique_view_count[observed].float().mean().item())

    @property
    def mean_observed_variance(self) -> float:
        """Mean cross-view variance over voxels with at least two unique views."""
        counts = self.unique_view_count.float()
        valid = counts > 1.0
        if not valid.any():
            return 0.0
        variance = self.running_m2[valid] / (counts[valid] - 1.0).clamp_min(1.0)
        return float(variance.mean().item())

    @property
    def mean_angular_factor(self) -> float:
        """Mean angular-diversity factor (1 − ‖mean_view_dir‖) over voxels
        observed by at least two unique views.

        Close to 0 means observing rays are clustered in direction (cache
        has narrow-cone evidence); close to 1 means rays span a wide
        angular range (cache has true triangulation evidence). On dense
        captures this rises smoothly toward ~0.5; on captures with
        clustered cameras it stays below ~0.2 and is the principal signal
        that the cache should be down-trusted on that scene.
        """
        counts = self.unique_view_count.float()
        mask = counts > 1.0
        if not mask.any():
            return 0.0
        dir_sum = self.view_dir_sum[mask]  # [M, 3]
        resultant = (dir_sum.norm(dim=-1) / counts[mask]).clamp(0.0, 1.0)
        return float((1.0 - resultant).mean().item())
