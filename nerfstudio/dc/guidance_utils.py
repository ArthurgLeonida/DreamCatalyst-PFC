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
    edit_strength_adaptive: bool,
    current_edit_strength: Optional[float],
) -> float:
    """Return scheduled STG scale, optionally attenuated by current-step edit strength.

    `current_edit_strength` is the ratio `||eps_tgt − eps_src|| / (||eps_tgt|| + ||eps_src||)`
    in [0, 1] — high for creative/structural edits, low for identity-preserving edits.
    When enabled, `scale *= (1 − edit_strength)` so bold edits attenuate STG more.
    """
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

    if edit_strength_adaptive and current_edit_strength is not None:
        s = min(max(float(current_edit_strength), 0.0), 1.0)
        scale = scale * (1.0 - s)

    return scale


def compute_edit_strength(eps_tgt: torch.Tensor, eps_src: torch.Tensor) -> float:
    """Return edit strength ∈ [0, 1] from the raw target/source noise predictions.

        s = ||eps_tgt − eps_src|| / (||eps_tgt|| + ||eps_src||)

    Bounded in [0, 1] by the triangle inequality. Time-invariant: numerator
    and denominator scale together with the noise-prediction magnitude at the
    current timestep, so the ratio cancels timestep effects.

    Physically: "how much the target's noise prediction diverges from the
    source's, relative to their combined magnitude." Low for identity-preserving
    edits (e.g. face / elf — target mostly agrees with source); high for
    structural edits (e.g. person → stormtrooper — target pulls the latent to a
    different rendering).

    Motivation over mask-based coverage: both the self-mask and the
    cross-attention mask are percentile-normalized in `normalize_relevance_map`,
    which forces their magnitudes into a narrow band across scenes. Neither raw
    mean nor the Herfindahl-inverse shape statistic discriminates scene types
    robustly in the direction the adaptation needs. `eps_tgt − eps_src` is the
    raw DDS delta before any mask is built — the clean scene-level signal.

    Used by the adaptive STG scale and outside-mask anchor weight as
    `multiplier = (1 − s)`: identity edits keep STG/anchor near full strength;
    structural edits attenuate both.
    """
    delta = (eps_tgt - eps_src).detach().float()
    tgt = eps_tgt.detach().float()
    src = eps_src.detach().float()
    delta_norm = delta.flatten(1).norm(dim=1)
    denom = (tgt.flatten(1).norm(dim=1) + src.flatten(1).norm(dim=1)).clamp_min(1e-8)
    return float((delta_norm / denom).mean().item())


def compute_preserve_weight(
    *,
    psi: float,
    grad_mask: Optional[torch.Tensor],
    outside_mask_anchor_weight: float,
    outside_mask_anchor_edit_strength_adaptive: bool,
    edit_strength: Optional[float] = None,
):
    """Compute DreamCatalyst preservation weight plus optional outside-mask anchor.

    `edit_strength` ∈ [0, 1] is the precomputed scalar from `compute_edit_strength`.
    When the adaptive flag is on and edit_strength is provided, the anchor weight
    is attenuated by `(1 − edit_strength)` so creative edits relax the background
    anchor while identity edits keep it near full strength.
    """
    preserve_weight = psi

    if grad_mask is not None and outside_mask_anchor_weight > 0:
        w_out_effective = outside_mask_anchor_weight
        if outside_mask_anchor_edit_strength_adaptive and edit_strength is not None:
            s = min(max(float(edit_strength), 0.0), 1.0)
            w_out_effective = w_out_effective * (1.0 - s)
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
