"""World-space voxel cache that shares the edit mask across views (Part 2).

Each iteration the pipeline backprojects the current view's edit-magnitude
mask (``raw_self``) to 3D with the rendered depth and calls ``update``; later
views read the accumulated value back with ``query``. Per voxel the cache keeps:

    grid                EMA mean of the mask: the value read back
    observed            whether any view has written the voxel
    unique_view_count   number of distinct views that observed it (n)
    view_observed       which views did, so revisiting a camera is not a new sample
    running_mean/_m2    exponentially weighted mean and variance across those views
    view_dir_sum        sum of their unit viewing directions

``query`` returns the value and a confidence in [0, 1], the product of four
linear gates: enough views (count), agreement across views (variance), spread
of viewing directions (angular), and edit mass (the value itself).

Background: lifting per-view 2D signals into a 3D field follows Semantic NeRF
(Zhi et al., ICCV 2021) and Panoptic Lifting (Siddiqui et al., CVPR 2023); this
is a non-parametric grid variant. Full walkthrough: docs/VoxelCacheExplained.md.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch


class MaskVoxelCache:
    """Coarse voxel grid of cross-view mask statistics (no learned parameters)."""

    def __init__(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        resolution: int,
        ema_beta: float,
        num_views: int,
        variance_decay: float,
        fallback_value: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            bbox_min, bbox_max: [3] world-space corners of the box the grid spans.
            resolution: voxels per axis V; the grid holds V**3 cells.
            ema_beta: decay of the grid mean; every touch keeps ema_beta of the old value.
            num_views: number of training cameras (view ids are in [0, num_views)).
            variance_decay: weight of the newest view in the exponentially weighted
                cross-view mean and variance, in (0, 1); memory of about 1/variance_decay
                views. Down-weights early views, whose edit was still immature.
            fallback_value: value returned where nothing has been observed.
            device: torch device for all state.
        """
        if not 0.0 < float(variance_decay) < 1.0:
            raise ValueError("variance_decay must be in (0, 1)")
        if int(num_views) <= 0:
            raise ValueError("num_views must be positive")

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bbox_min = bbox_min.to(self.device).float()
        self.bbox_max = bbox_max.to(self.device).float()
        self.resolution = int(resolution)
        self.ema_beta = float(ema_beta)
        self.variance_decay = float(variance_decay)
        self.num_views = int(num_views)
        self.fallback_value = float(fallback_value)

        V = self.resolution
        shape = (V, V, V)
        self.grid = torch.full(shape, self.fallback_value, device=self.device, dtype=torch.float32)
        self.observed = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.unique_view_count = torch.zeros(shape, dtype=torch.int32, device=self.device)
        self.view_observed = torch.zeros((V ** 3, self.num_views), dtype=torch.bool, device=self.device)
        self.running_mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        # Holds the exponentially weighted variance itself (not a sum of squares).
        self.running_m2 = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.view_dir_sum = torch.zeros((*shape, 3), dtype=torch.float32, device=self.device)

        # Scene-relative angular normalization: the denominator is frozen at the
        # peak of the trusted population's mean angular factor
        # (try_auto_freeze_angular_denominator), so the same voxel is judged the
        # same way for the rest of training.
        self._frozen_angular_denominator: Optional[float] = None
        self._angular_peak_value: float = 0.0
        self._angular_peak_step: int = -1

    # -----------------------------------------------------------------
    # Grid indexing
    # -----------------------------------------------------------------

    def _world_to_voxel(self, points_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest-voxel indices of world points.

        idx = floor((x - bbox_min) / (bbox_max - bbox_min) * V), clamped to the
        grid. Also returns which points were inside the box before clamping.
        """
        normalized = (points_world - self.bbox_min) / (self.bbox_max - self.bbox_min).clamp_min(1e-8)
        in_bounds = ((normalized >= 0.0) & (normalized < 1.0)).all(dim=-1)
        idx = (normalized * self.resolution).long().clamp_(0, self.resolution - 1)
        return idx, in_bounds

    def _trilinear_corners(
        self, flat_points: torch.Tensor
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (ix, iy, iz, weight) for the 8 voxel centers around each point.

        Voxel centers sit at (index + 0.5) / V in normalized coordinates; points
        beyond the outermost centers are clamped to them.
        """
        V = self.resolution
        normalized = (flat_points - self.bbox_min) / (self.bbox_max - self.bbox_min).clamp_min(1e-8)
        c = (normalized * V - 0.5).clamp(0.0, float(V - 1))
        i0 = c.floor().long().clamp_(0, V - 1)
        i1 = (i0 + 1).clamp_(0, V - 1)
        frac = (c - i0.float()).clamp(0.0, 1.0)

        for cx in (0, 1):
            wx = frac[:, 0] if cx else (1.0 - frac[:, 0])
            gx = i1[:, 0] if cx else i0[:, 0]
            for cy in (0, 1):
                wy = frac[:, 1] if cy else (1.0 - frac[:, 1])
                gy = i1[:, 1] if cy else i0[:, 1]
                for cz in (0, 1):
                    wz = frac[:, 2] if cz else (1.0 - frac[:, 2])
                    gz = i1[:, 2] if cz else i0[:, 2]
                    yield gx, gy, gz, wx * wy * wz

    def _trilinear_observed_read(self, flat_points: torch.Tensor) -> torch.Tensor:
        """Trilinear read of the grid mean, counting only observed corners.

        Unobserved corners hold the fallback value; weighting them out keeps it
        from bleeding into the result. Points with no observed corner read the
        fallback.
        """
        num = torch.zeros(flat_points.shape[0], device=self.device, dtype=torch.float32)
        den = torch.zeros_like(num)
        for gx, gy, gz, w in self._trilinear_corners(flat_points):
            obs = self.observed[gx, gy, gz].float()
            num = num + w * obs * self.grid[gx, gy, gz]
            den = den + w * obs
        return torch.where(
            den > 1e-6,
            num / den.clamp_min(1e-6),
            torch.full_like(num, self.fallback_value),
        )

    def _trilinear_variance_read(
        self, flat_points: torch.Tensor, nearest_variance: torch.Tensor
    ) -> torch.Tensor:
        """Trilinear read of the cross-view variance, so the agreement gate is
        continuous across voxel boundaries like the value it gates.

        Unlike the value read: a corner contributes only with at least two
        views (with one view the variance is undefined, not zero), and only the
        within-voxel variance is averaged. The spread between the corners'
        means is a spatial gradient, not view disagreement. Points with no
        eligible corner keep the nearest-voxel variance.
        """
        den = torch.zeros(flat_points.shape[0], device=self.device, dtype=torch.float32)
        within = torch.zeros_like(den)
        for gx, gy, gz, w in self._trilinear_corners(flat_points):
            counts_c = self.unique_view_count[gx, gy, gz].float()
            eligible = (self.observed[gx, gy, gz] & (counts_c > 1.0)).float()
            w_eff = w * eligible
            den = den + w_eff
            within = within + w_eff * self.running_m2[gx, gy, gz]
        smoothed = within / den.clamp_min(1e-6)
        return torch.where(den > 1e-6, smoothed, nearest_variance)

    # -----------------------------------------------------------------
    # Update: write one view's mask into the grid
    # -----------------------------------------------------------------

    def update(
        self,
        points_world: torch.Tensor,
        mask_values: torch.Tensor,
        in_bounds: torch.Tensor,
        view_id: int,
        ray_directions: torch.Tensor,
        return_per_voxel_mean: bool = False,
    ) -> Optional[torch.Tensor]:
        """Write one view's mask values at their backprojected points.

        Pixels of this view are first averaged per voxel (one sample per view and
        voxel, so dense regions do not dominate). Then:
          - the grid mean takes an EMA step; a voxel's first touch copies the value;
          - only on this view's FIRST observation of a voxel, the cross-view
            statistics (count, EW mean/variance, direction sum) are updated, so
            revisiting a camera never counts as a new, agreeing view.

        Args:
            points_world: [N, 3] world points; mask_values: [N] in [0, 1];
            in_bounds: [N] validity (surface hit); view_id: camera slot;
            ray_directions: [N, 3] viewing directions of the pixels.

        Returns the per-voxel values this view wrote when ``return_per_voxel_mean``
        (for histograms), else None.
        """
        points_world = points_world.reshape(-1, 3).to(self.device)
        mask_values = mask_values.reshape(-1).to(self.device).float().clamp(0.0, 1.0)
        empty = mask_values.new_empty(0) if return_per_voxel_mean else None
        if points_world.numel() == 0:
            return empty

        idx, inside = self._world_to_voxel(points_world)
        valid = in_bounds.reshape(-1).to(self.device).bool() & inside
        directions = ray_directions.reshape(-1, 3).to(self.device).float()
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        idx = idx[valid]
        mask_values = mask_values[valid]
        directions = directions[valid]
        if idx.numel() == 0:
            return empty

        V = self.resolution
        n_voxels = V ** 3
        flat = idx[:, 0] * V * V + idx[:, 1] * V + idx[:, 2]

        # Per-voxel mean of this view's pixels, and its mean viewing direction.
        sums = torch.zeros(n_voxels, device=self.device, dtype=torch.float32)
        counts = torch.zeros(n_voxels, device=self.device, dtype=torch.float32)
        sums.index_add_(0, flat, mask_values)
        counts.index_add_(0, flat, torch.ones_like(mask_values))
        dir_sums = torch.zeros(n_voxels, 3, device=self.device, dtype=torch.float32)
        dir_sums.index_add_(0, flat, directions)
        view_dirs = dir_sums / dir_sums.norm(dim=1, keepdim=True).clamp_min(1e-8)

        touched_idx = (counts > 0).nonzero(as_tuple=True)[0]
        if touched_idx.numel() == 0:
            return empty
        touched_values = sums[touched_idx] / counts[touched_idx]
        snapshot = touched_values.detach().clone() if return_per_voxel_mean else None

        # Cross-view statistics: only voxels this view has never observed before.
        vid = int(view_id)
        if not 0 <= vid < self.num_views:
            raise ValueError(f"view_id {vid} is outside [0, {self.num_views})")
        is_new_view = ~self.view_observed[touched_idx, vid]
        if is_new_view.any():
            new_idx = touched_idx[is_new_view]
            self.view_observed[new_idx, vid] = True
            self._update_cross_view_statistics(new_idx, touched_values[is_new_view], view_dirs[new_idx])

        # Grid mean: an EMA step on every touch, revisits included.
        flat_grid = self.grid.view(-1)
        flat_observed = self.observed.view(-1)
        seen_before = flat_observed[touched_idx]
        old_idx = touched_idx[seen_before]
        first_idx = touched_idx[~seen_before]
        if old_idx.numel() > 0:
            flat_grid[old_idx] = (
                self.ema_beta * flat_grid[old_idx]
                + (1.0 - self.ema_beta) * touched_values[seen_before]
            )
        if first_idx.numel() > 0:
            # First observation: copy instead of blending with the fallback.
            flat_grid[first_idx] = touched_values[~seen_before]
            flat_observed[first_idx] = True

        return snapshot

    def _update_cross_view_statistics(
        self, voxel_idx: torch.Tensor, values: torch.Tensor, view_dirs: torch.Tensor
    ) -> None:
        """Add one new view's sample to the count, EW mean/variance and direction sum.

        Exponentially weighted mean and variance (Finch 2009) with new-sample
        weight alpha = variance_decay; a voxel's first view seeds mean = value and
        variance = 0.
        """
        flat_count = self.unique_view_count.view(-1)
        flat_mean = self.running_mean.view(-1)
        flat_var = self.running_m2.view(-1)
        flat_dirs = self.view_dir_sum.view(-1, 3)

        flat_count[voxel_idx] += 1
        first = flat_count[voxel_idx] == 1
        alpha = self.variance_decay
        old_mean = flat_mean[voxel_idx]
        delta = values - old_mean
        ew_incr = alpha * delta
        flat_mean[voxel_idx] = old_mean + torch.where(first, delta, ew_incr)
        flat_var[voxel_idx] = torch.where(
            first,
            torch.zeros_like(old_mean),
            (1.0 - alpha) * (flat_var[voxel_idx] + delta * ew_incr),
        )
        flat_dirs[voxel_idx] = flat_dirs[voxel_idx] + view_dirs

    # -----------------------------------------------------------------
    # Query: read the cache back at a view's points
    # -----------------------------------------------------------------

    def query(
        self,
        points_world: torch.Tensor,
        in_bounds: torch.Tensor,
        min_observations: int,
        max_variance: float,
        mass_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the cache at world points.

        Returns (values, valid, confidence, counts, variance), each shaped like
        ``points_world[..., 0]``:
            values      observed-weighted trilinear mean; fallback where invalid
            valid       nearest voxel observed and the point inside the box
            confidence  valid * count gate * variance gate * angular gate * mass gate
            counts      unique views of the nearest voxel
            variance    cross-view variance (trilinear over eligible corners)

        Gates (all linear, in [0, 1]):
            count     1 if counts >= min_observations else 0
            variance  1 - variance / max_variance
            angular   (1 - |mean viewing direction|) relative to the scene's frozen mean
            mass      value / mass_threshold (the model barely edits here)
        """
        original_shape = points_world.shape[:-1]
        flat_points = points_world.reshape(-1, 3).to(self.device)
        idx, inside = self._world_to_voxel(flat_points)
        in_bounds = in_bounds.reshape(-1).to(self.device) & inside
        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]

        values = self._trilinear_observed_read(flat_points)
        observed = self.observed[ix, iy, iz]
        counts = self.unique_view_count[ix, iy, iz].float()
        m2 = self.running_m2[ix, iy, iz]
        # Variance is undefined (stored as 0) below two views.
        nearest_variance = torch.where(counts > 1.0, m2, torch.zeros_like(m2))
        variance = self._trilinear_variance_read(flat_points, nearest_variance)

        valid = observed & in_bounds
        enough_views = counts >= float(max(int(min_observations), 1))
        if float(max_variance) > 0.0:
            var_confidence = (1.0 - variance / float(max_variance)).clamp(0.0, 1.0)
        else:
            var_confidence = torch.ones_like(values)
        angular_confidence = self._angular_confidence(ix, iy, iz, counts, min_observations)
        if float(mass_threshold) > 1e-6:
            mass_confidence = (values / float(mass_threshold)).clamp(0.0, 1.0)
        else:
            mass_confidence = torch.ones_like(values)

        confidence = (
            valid.float()
            * enough_views.float()
            * var_confidence
            * angular_confidence
            * mass_confidence
        )
        values = torch.where(valid, values, torch.full_like(values, self.fallback_value))
        return (
            values.view(*original_shape),
            valid.view(*original_shape),
            confidence.view(*original_shape),
            counts.view(*original_shape),
            variance.view(*original_shape),
        )

    def _angular_confidence(self, ix, iy, iz, counts, min_observations) -> torch.Tensor:
        """Angular gate: how spread out the viewing directions of a voxel are.

        A = 1 - |sum of unit directions| / n: 0 when every view looked from the
        same direction (no triangulation), toward 1 when directions spread over
        the sphere. Divided by the scene's mean A over trusted voxels (frozen at
        its peak), so a capture rig with little absolute spread is judged
        against itself; clamped to [0, 1].
        """
        resultant = (self.view_dir_sum[ix, iy, iz].norm(dim=-1) / counts.clamp_min(1.0)).clamp(0.0, 1.0)
        angular = (1.0 - resultant).clamp(0.0, 1.0)
        if self._frozen_angular_denominator is not None:
            scene_mean = float(self._frozen_angular_denominator)
        else:
            scene_mean = float(self.mean_angular_factor_at(min_views=min_observations))
        if scene_mean > 1e-6:
            angular = (angular / scene_mean).clamp(0.0, 1.0)
        return angular

    # -----------------------------------------------------------------
    # Angular denominator freeze and diagnostics
    # -----------------------------------------------------------------

    def mean_angular_factor_at(self, min_views: int = 2) -> float:
        """Mean angular factor (1 - resultant length) over voxels with >= min_views views.

        Restricting to the trusted population (min_views = the count gate's
        threshold) matters: averaging all two-view voxels would drag the mean
        toward 0, because late edge voxels seen by two nearby cameras dominate.
        """
        min_views = max(int(min_views), 2)
        counts = self.unique_view_count.float()
        trusted = counts >= float(min_views)
        if not trusted.any():
            return 0.0
        resultant = (self.view_dir_sum[trusted].norm(dim=-1) / counts[trusted]).clamp(0.0, 1.0)
        return float((1.0 - resultant).mean().item())

    def try_auto_freeze_angular_denominator(
        self,
        edit_step: int,
        min_views: int,
        patience: int,
        warmup_steps: int,
        min_value: float = 1e-3,
    ) -> Optional[float]:
        """Freeze the angular denominator at the peak of the trusted mean.

        After ``warmup_steps`` edit steps, tracks the running maximum of
        ``mean_angular_factor_at(min_views)``. When it has not improved for
        ``patience`` steps, freezes it (if above ``min_value``). A new peak after
        a freeze thaws it, and the patience check fires again on the new peak.
        Returns the frozen value on the call that froze it, else None.
        """
        if edit_step < int(warmup_steps):
            return None
        current = self.mean_angular_factor_at(min_views=min_views)
        if current > self._angular_peak_value:
            self._angular_peak_value = float(current)
            self._angular_peak_step = int(edit_step)
            self._frozen_angular_denominator = None
            return None
        if self._frozen_angular_denominator is not None:
            return None
        if (
            edit_step - self._angular_peak_step >= int(patience)
            and self._angular_peak_value > float(min_value)
        ):
            self._frozen_angular_denominator = float(self._angular_peak_value)
            return self._frozen_angular_denominator
        return None

    @property
    def angular_denominator_is_frozen(self) -> bool:
        return self._frozen_angular_denominator is not None

    @property
    def frozen_angular_denominator(self) -> Optional[float]:
        return self._frozen_angular_denominator

    @property
    def angular_peak_value(self) -> float:
        """Highest trusted mean angular factor seen so far."""
        return float(self._angular_peak_value)

    @property
    def angular_peak_step(self) -> int:
        """Edit step at which ``angular_peak_value`` was reached."""
        return int(self._angular_peak_step)

    @property
    def mean_angular_factor(self) -> float:
        """Mean angular factor over every voxel with at least two views (diagnostic)."""
        return self.mean_angular_factor_at(min_views=2)

    @property
    def occupancy(self) -> float:
        """Fraction of voxels observed at least once."""
        return float(self.observed.float().mean().item())

    @property
    def mean_observation_count(self) -> float:
        """Mean number of unique views over observed voxels."""
        if not self.observed.any():
            return 0.0
        return float(self.unique_view_count[self.observed].float().mean().item())

    @property
    def mean_observed_variance(self) -> float:
        """Mean cross-view variance over voxels with at least two views."""
        multi_view = self.unique_view_count > 1
        if not multi_view.any():
            return 0.0
        return float(self.running_m2[multi_view].mean().item())
