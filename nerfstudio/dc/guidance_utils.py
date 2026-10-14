"""Stateless guidance formulas used by ``DC.__call__``, one per function.

    compute_tag_eta            adaptive TAG strength for the current timestep
    apply_tag                  tangential amplification of a noise prediction
    compute_stg_scale          bump-scheduled STG scale, attenuated by (1 - s)
    compute_edit_strength      scalar edit strength s in [0, 1]
    apply_source_blend         restrict the DDS delta to a mask
    compute_preserve_weight    DreamCatalyst psi plus the outside-mask anchor
    apply_latent_mean_anchor   pull per-channel latent means toward the source
    compute_cache_gate         semantic gate for voxel-cache additions
"""

import math
from typing import Optional

import torch


def compute_tag_eta(eta_max: float, t_normalized: float) -> float:
    """Adaptive TAG strength: eta(t) = 1 + (eta_max - 1) * t^(1/e).

    Strongest at high noise and relaxing toward the identity (eta = 1) as the
    timestep anneals. The 1/e exponent is the one in DreamCatalyst's w_DDS, so
    the two decay together. eta_max = 1 disables TAG.
    """
    return 1.0 + (eta_max - 1.0) * t_normalized ** (1 / math.e)


def apply_tag(noise_pred: torch.Tensor, latents_noisy: torch.Tensor, eta: float) -> torch.Tensor:
    """Tangential Amplifying Guidance (TAG).

    Splits the prediction into its component along the noisy latent (radial)
    and the rest (tangential), and scales only the tangential part by eta.
    eta = 1 returns the prediction unchanged.
    """
    if eta == 1.0:
        return noise_pred

    v = latents_noisy / (latents_noisy.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8)
    noise_parallel = (noise_pred * v).sum(dim=(1, 2, 3), keepdim=True) * v
    noise_tangential = noise_pred - noise_parallel
    return noise_parallel + eta * noise_tangential


def compute_stg_scale(
    *,
    base_scale: float,
    iteration: int,
    max_iteration: int,
    start_ratio: float,
    end_ratio: float,
    bump_peak_ratio: float,
    current_edit_strength: float,
) -> float:
    """STG scale for this iteration: a triangular bump times (1 - s).

    Over training progress p = iteration / max_iteration the scale is zero
    outside [start_ratio, end_ratio], rises linearly to ``base_scale`` at
    start + bump_peak_ratio * (end - start), and falls back to zero at
    end_ratio, so the last stretch of training runs without STG. The result is
    multiplied by (1 - s): bold edits (high edit strength s) get less STG.
    """
    max_iteration = max(int(max_iteration), 1)
    progress = min(max(iteration / max_iteration, 0.0), 1.0)
    start_ratio = float(start_ratio)
    end_ratio = float(end_ratio)

    if progress <= start_ratio or progress >= end_ratio or end_ratio <= start_ratio:
        scale = 0.0
    else:
        peak_ratio = min(max(float(bump_peak_ratio), 0.0), 1.0)
        peak_time = start_ratio + peak_ratio * (end_ratio - start_ratio)
        if progress <= peak_time:
            span = max(peak_time - start_ratio, 1e-8)
            scale = float(base_scale) * (progress - start_ratio) / span
        else:
            span = max(end_ratio - peak_time, 1e-8)
            scale = float(base_scale) * (end_ratio - progress) / span

    s = min(max(float(current_edit_strength), 0.0), 1.0)
    return scale * (1.0 - s)


def compute_edit_strength(eps_tgt: torch.Tensor, eps_src: torch.Tensor) -> float:
    """Edit strength s = ||eps_tgt - eps_src|| / (||eps_tgt|| + ||eps_src||) in [0, 1].

    Bounded by the triangle inequality. Computed from the raw IP2P predictions
    (before TAG/STG), so it measures how strongly the target branch disagrees
    with the source branch, relative to their magnitudes: low for
    identity-preserving edits (elf), high for structural ones (stormtrooper).
    Used as (1 - s) to attenuate STG and the outside-mask anchor.
    """
    delta = (eps_tgt - eps_src).detach().float()
    tgt = eps_tgt.detach().float()
    src = eps_src.detach().float()
    delta_norm = delta.flatten(1).norm(dim=1)
    denom = (tgt.flatten(1).norm(dim=1) + src.flatten(1).norm(dim=1)).clamp_min(1e-8)
    return float((delta_norm / denom).mean().item())


def apply_source_blend(
    eps_tgt: torch.Tensor,
    eps_src: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Source-blended localization: eps_src + M * (eps_tgt - eps_src).

    The DDS delta (eps_tgt - eps_src) survives only where the mask is high and
    is exactly zero where M = 0. It masks the edit force only; the
    preservation term of the gradient is not masked.
    """
    return eps_src + mask * (eps_tgt - eps_src)


def compute_preserve_weight(
    *,
    psi: float,
    preservation_mask: Optional[torch.Tensor],
    outside_mask_anchor_weight: float,
    edit_strength: float,
):
    """Preservation weight: psi + w_out * (1 - s) * (1 - P).

    psi is DreamCatalyst's global weight. The outside-mask anchor adds w_out
    where the preservation mask P (the source-instruction mask) says the edit
    does not belong, attenuated by (1 - s) so structural edits are held less
    tightly. Returns the scalar psi when there is no mask or w_out = 0.
    """
    if preservation_mask is None or outside_mask_anchor_weight <= 0:
        return psi
    s = min(max(float(edit_strength), 0.0), 1.0)
    return psi + outside_mask_anchor_weight * (1.0 - s) * (1.0 - preservation_mask)


def apply_latent_mean_anchor(
    grad: torch.Tensor,
    tgt_x0: torch.Tensor,
    src_x0: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Add weight * (mean(tgt) - mean(src)) per channel to the gradient.

    Pulls the per-channel spatial mean of the edited latent toward the
    source's, countering the brightness/saturation drift of TAG.
    """
    if weight <= 0:
        return grad
    tgt_mean = tgt_x0.mean(dim=(2, 3), keepdim=True)
    src_mean = src_x0.mean(dim=(2, 3), keepdim=True)
    return grad + weight * (tgt_mean - src_mean).expand_as(grad)


def compute_cache_gate(
    ca_mask: Optional[torch.Tensor],
    self_mask: torch.Tensor,
) -> torch.Tensor:
    """Semantic gate for voxel-cache additions: G = max(A, S) * A.

    The cache may only add support where the current view confirms it, and the
    cross-attention mask A must be part of that confirmation: the self-mask S
    alone cannot authorize an addition, because the cache is built from S.
    Without an attention map the gate is zero (no additions).
    """
    if ca_mask is None:
        return torch.zeros_like(self_mask)
    return torch.maximum(ca_mask, self_mask).clamp(0.0, 1.0) * ca_mask
