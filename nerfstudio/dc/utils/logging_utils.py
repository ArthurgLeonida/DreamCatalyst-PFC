"""WandB debug payload of one DC iteration (keys under dc_debug/)."""

from typing import Optional

import torch


def make_wandb_image(image_tensor, tensor_to_pil_fn, resize_image_fn, caption: str, min_size: int = 256):
    """Convert a tensor image or mask into a compact WandB image."""
    import wandb

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.detach().float().cpu().clamp(0.0, 1.0)
    return wandb.Image(
        resize_image_fn(tensor_to_pil_fn(image_tensor), min_size=min_size),
        caption=caption,
    )


def summarize_mask(mask: torch.Tensor):
    """Mean, max and fraction of pixels above 0.5 of a mask."""
    mask = mask.detach().float()
    return {
        "mean": mask.mean().item(),
        "max": mask.max().item(),
        "coverage_0.5": (mask > 0.5).float().mean().item(),
    }


def batch_l2_norm_mean(tensor: torch.Tensor) -> float:
    """Return the mean L2 norm across the batch for arbitrary-shaped tensors."""
    return tensor.detach().float().flatten(1).norm(dim=1).mean().item()


def log_dc_debug_to_wandb(
    *,
    step: int,
    current_spot: int,
    t: torch.Tensor,
    t_normalized: float,
    eta_tag_current: float,
    current_stg_scale: float,
    current_edit_strength: float,
    w_dds: float,
    preserve_weight,
    eps_tgt: torch.Tensor,
    eps_src: torch.Tensor,
    eps_tgt_for_grad: torch.Tensor,
    grad: torch.Tensor,
    pred_x0_tgt: torch.Tensor,
    pred_x0_src: torch.Tensor,
    tensor_to_pil_fn,
    resize_image_fn,
    masks=None,
    stg_perturbation: Optional[torch.Tensor] = None,
    source_blend_enabled: bool = False,
    self_mask_ema_beta: Optional[float] = None,
):
    """Log the scalars, predictions and masks of one DC iteration.

    ``masks`` is the iteration's ``dc.dc.LocalizationMasks`` (None when
    localization is off, as in the DreamCatalyst baseline).
    """
    import wandb

    caption = (
        f"step={step} | spot={current_spot} | "
        f"t={t.item()} | t_norm={t_normalized:.3f}"
    )

    def add_image(name, image):
        payload[f"dc_debug/{name}"] = make_wandb_image(image, tensor_to_pil_fn, resize_image_fn, caption)

    def add_mask(name, mask):
        add_image(name, mask)
        for key, value in summarize_mask(mask).items():
            payload[f"dc_debug/{name}_{key}"] = value

    preserve_weight_mean = (
        preserve_weight.mean().item()
        if isinstance(preserve_weight, torch.Tensor)
        else float(preserve_weight)
    )
    payload = {
        "dc_debug/current_spot": int(current_spot),
        "dc_debug/timestep": float(t.item()),
        "dc_debug/timestep_normalized": float(t_normalized),
        "dc_debug/eta_tag_current": float(eta_tag_current),
        "dc_debug/stg_scale_current": float(current_stg_scale),
        "dc_debug/edit_strength": float(current_edit_strength),
        "dc_debug/w_dds": float(w_dds),
        "dc_debug/preserve_weight_mean": preserve_weight_mean,
        # Delta norms before and after source blending, and of the final gradient.
        "dc_debug/raw_eps_delta_norm": batch_l2_norm_mean(eps_tgt - eps_src),
        "dc_debug/effective_eps_delta_norm": batch_l2_norm_mean(eps_tgt_for_grad - eps_src),
        "dc_debug/grad_norm": batch_l2_norm_mean(grad),
    }
    if self_mask_ema_beta is not None:
        payload["dc_debug/self_mask_ema_beta"] = float(self_mask_ema_beta)
    add_image("tgt_pred_x0", pred_x0_tgt)
    add_image("src_pred_x0", pred_x0_src)

    # STG's effective contribution against the TAG-only DDS delta, both through
    # the same source-blend mask. Zero while STG is off.
    edit_mask = masks.final if masks is not None else None
    stg_delta = stg_perturbation if stg_perturbation is not None else torch.zeros_like(eps_tgt)
    base_delta = eps_tgt - eps_src - stg_delta
    if source_blend_enabled and edit_mask is not None:
        stg_delta = edit_mask * stg_delta
        base_delta = edit_mask * base_delta
    stg_norm = batch_l2_norm_mean(stg_delta)
    base_norm = batch_l2_norm_mean(base_delta)
    payload["dc_debug/stg_effective_delta_norm"] = stg_norm
    payload["dc_debug/tag_only_effective_delta_norm"] = base_norm
    payload["dc_debug/stg_to_tag_only_dds_ratio"] = stg_norm / max(base_norm, 1e-8)

    if masks is not None:
        add_mask("self_mask", masks.self_mask)
        add_mask("preservation_mask", masks.source_mask)
        if masks.ca_mask is not None:
            add_mask("cross_attention_mask", masks.ca_mask)
        add_mask("internal_mask", masks.internal)
        add_mask("final_mask", masks.final)
        if masks.veto_floor_addition is not None:
            # Support the floor min(S, S_src) restored over S * A: should light up
            # body parts the edit tokens miss (hands), not a halo of background.
            payload["dc_debug/ca_veto_floor_addition_mean"] = (
                masks.veto_floor_addition.float().mean().item()
            )
            add_image("ca_veto_floor_addition", masks.veto_floor_addition)
        # What the voxel cache added on top of the internal mask (zero without it).
        cache_addition = (masks.final - masks.internal).clamp_min(0.0)
        payload["dc_debug/cache_addition_mean"] = cache_addition.detach().float().mean().item()
        add_image("cache_addition", cache_addition)

    wandb.log(payload, step=step, commit=False)
