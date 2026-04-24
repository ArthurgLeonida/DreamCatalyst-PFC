import math
from typing import Optional

import torch


def compute_tag_eta(eta_tag: float, t_normalized: float, adaptive_tag: bool) -> float:
    """Return the TAG eta for the current timestep."""
    if not adaptive_tag:
        return eta_tag
    return 1.0 + (eta_tag - 1.0) * t_normalized ** (1 / math.e)


def apply_tag(noise_pred: torch.Tensor, latents_noisy: torch.Tensor, eta: float) -> torch.Tensor:
    """Apply tangential amplification around the noisy latent direction."""
    if eta == 1.0:
        return noise_pred

    v = latents_noisy / (latents_noisy.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8)
    noise_parallel = (noise_pred * v).sum(dim=(1, 2, 3), keepdim=True) * v
    noise_tangential = noise_pred - noise_parallel
    return noise_parallel + eta * noise_tangential


def apply_stg(noise_pred: torch.Tensor, noise_pred_weak: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply STG extrapolation from weak-model prediction to full prediction."""
    if scale <= 0:
        return noise_pred
    return noise_pred + scale * (noise_pred - noise_pred_weak)


def apply_perp_neg(eps_tgt: torch.Tensor, eps_src: torch.Tensor, alpha: float) -> torch.Tensor:
    """Orthogonalize target prediction with respect to the source prediction."""
    src_norm_sq = (eps_src * eps_src).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    projection = (eps_tgt * eps_src).sum(dim=(1, 2, 3), keepdim=True) / src_norm_sq
    return eps_tgt - alpha * projection * eps_src


def apply_source_blend(eps_tgt: torch.Tensor, eps_src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Localize DDS by falling back to the source branch outside the mask."""
    return eps_src + mask * (eps_tgt - eps_src)


def compute_stg_scale(
    *,
    base_scale: float,
    iteration: int,
    max_iteration: int,
    schedule_enabled: bool,
    mode: str,
    start_ratio: float,
    end_ratio: float,
    bump_peak_ratio: float,
    coverage_adaptive: bool,
    current_mask_coverage: Optional[float],
) -> float:
    """Return scheduled STG scale, optionally attenuated by robust current-step coverage."""
    if not schedule_enabled:
        scale = float(base_scale)
    else:
        max_iteration = max(int(max_iteration), 1)
        progress = min(max(iteration / max_iteration, 0.0), 1.0)
        start_ratio = float(start_ratio)
        end_ratio = float(end_ratio)
        mode = str(mode).lower()

        if mode == "growth":
            if progress <= start_ratio:
                scale = 0.0
            elif progress >= end_ratio or end_ratio <= start_ratio:
                scale = float(base_scale)
            else:
                scale = float(base_scale) * (progress - start_ratio) / (end_ratio - start_ratio)
        elif mode == "bump":
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
        else:
            if end_ratio <= start_ratio:
                scale = float(base_scale) if progress < end_ratio else 0.0
            elif progress <= start_ratio:
                scale = float(base_scale)
            elif progress >= end_ratio:
                scale = 0.0
            else:
                decay_progress = (progress - start_ratio) / (end_ratio - start_ratio)
                scale = float(base_scale) * (1.0 - decay_progress)

    if coverage_adaptive and current_mask_coverage is not None:
        coverage = min(max(float(current_mask_coverage), 0.0), 1.0)
        scale = scale * (1.0 - coverage)

    return scale


def compute_mask_coverage(mask: Optional[torch.Tensor]) -> Optional[float]:
    """Return effective-support coverage via the Herfindahl-inverse statistic.

    coverage = mean(M)**2 / mean(M**2)  ∈ [0, 1]

    Interpretation: "effective fraction of pixels participating in the mask
    mass." For a binary mask, coverage equals the area fraction of mask=1
    pixels. For a smooth mask, it captures shape concentration — a tight
    hotspot gives low coverage, diffuse support gives high coverage.

    Motivation: the self-mask and cross-attention mask are both
    percentile-normalized inside `normalize_relevance_map`, which fixes their
    raw means to a narrow band across scene types and destroys magnitude-based
    discrimination. The Herfindahl inverse survives percentile normalization
    because it is a shape statistic — the denominator mean(M**2) varies with
    how the (near-constant) L1 mass is distributed spatially.

    Used by the coverage-adaptive STG scale and outside-mask anchor weight as
    `multiplier = (1 - coverage)`: localized edits (elf face, low coverage)
    keep STG/anchor near full strength; creative edits (stormtrooper body,
    high coverage) attenuate both — the target universal-config behavior.
    """
    if mask is None:
        return None

    mask = mask.detach().float().clamp(0.0, 1.0)
    mean_m = mask.mean()
    mean_m_sq = (mask * mask).mean()
    if mean_m_sq.item() <= 1e-12:
        return 0.0
    return float((mean_m * mean_m / mean_m_sq).item())


def compute_preserve_weight(
    *,
    psi: float,
    psi_late_multiplier: float,
    t_normalized: float,
    grad_mask: Optional[torch.Tensor],
    outside_mask_anchor_weight: float,
    outside_mask_anchor_coverage_adaptive: bool,
    mask_coverage: Optional[float] = None,
):
    """Compute DreamCatalyst preservation weight plus optional outside-mask anchor.

    `mask_coverage` is a precomputed robust scalar coverage of `grad_mask`.
    When None (e.g. during mask warmup), coverage-adaptive attenuation is
    skipped, matching the STG scheduler's behavior.
    """
    psi_schedule_factor = 1.0 + (psi_late_multiplier - 1.0) * (1.0 - t_normalized)
    preserve_weight = psi * psi_schedule_factor

    if grad_mask is not None and outside_mask_anchor_weight > 0:
        w_out_effective = outside_mask_anchor_weight
        if outside_mask_anchor_coverage_adaptive and mask_coverage is not None:
            coverage = min(max(float(mask_coverage), 0.0), 1.0)
            w_out_effective = w_out_effective * (1.0 - coverage)
        preserve_weight = preserve_weight + w_out_effective * (1.0 - grad_mask)

    return preserve_weight


def apply_latent_mean_anchor(
    grad: torch.Tensor,
    tgt_x0: torch.Tensor,
    src_x0: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Pull per-channel spatial means of edited latent toward source latent."""
    if weight <= 0:
        return grad
    tgt_mean = tgt_x0.mean(dim=(2, 3), keepdim=True)
    src_mean = src_x0.mean(dim=(2, 3), keepdim=True)
    return grad + weight * (tgt_mean - src_mean).expand_as(grad)
