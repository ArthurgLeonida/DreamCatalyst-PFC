# Custom DreamCatalyst novelty parameters
# Edit this file to configure all novelty settings in one place.
# These values are unpacked into every DCConfig instantiation.
#
# Suggested tuning order:
# 1. Localization branch (main current research direction)
# 2. TAG branch (edit-strength novelties)
# 3. STG branch (optional extra UNet pass)
# 4. Perp-Neg branch (older creative-editing branch)

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # 1. Localization branch
    # ---------------------------------------------------------------------
    psi=0.075,
    source_blend_localization_enabled=True,
    gradient_mask_enabled=False,
    outside_mask_anchor_weight=0.05, # Default = 0.05
    # Coverage-adaptive outside-mask anchor: scales outside_mask_anchor_weight by
    # (1 − per-sample mean(grad_mask)). Face (small coverage) keeps the bg anchor tight;
    # stormtrooper (large coverage) automatically loosens it. Enables one anchor value
    # to work across identity-preserving and creative-transform scenes.
    outside_mask_anchor_coverage_adaptive=False,

    # These parameters matter whenever source_blend_localization_enabled,
    # gradient_mask_enabled, or outside_mask_anchor_weight is active.
    gradient_mask_blur=2.0,      # latent-space Gaussian blur sigma
    gradient_mask_gamma=1.2,     # >1 = tighter mask, <1 = broader mask
    gradient_mask_ema_beta=0,  # temporal smoothing per view; 0 disables EMA
    gradient_mask_warmup=0,     # during warmup the mask is all ones
    # Optional semantic prior from target-token cross-attention. This is meant
    # to tighten the self-mask when methods like STG / Perp-Neg spread the
    # raw eps_tgt - eps_src delta across the whole image.
    cross_attention_mask_enabled=False,
    cross_attention_mask_keywords="",
    cross_attention_mask_prompt="",
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=1.0,
    cross_attention_mask_blur=0.0,
    cross_attention_mask_gamma=1.0,
    # M1: ignore M_self and use M_attn alone (use when ||eps_tgt - eps_src|| anti-localizes the edit,
    # e.g. IP2P on face scenes where image conditioning collapses the delta on the target).
    # Note: on IP2P face scenes tested so far, M1 underperforms W — use only when diagnostic says so.
    cross_attention_mask_only=False,
    # M2: treat (1 - M_self) as the "model-agreement" mask and intersect with M_attn.
    # Note: catastrophically failed on face/elf (CLIPd -0.02); kept as an ablation switch.
    invert_self_mask=False,

    # DaCapo-inspired ψ schedule (Huang et al., CVPR 2025).
    # psi_late_multiplier=1.0 disables the schedule (current DreamCatalyst behavior).
    # >1 grows preservation as t decreases: edit commits early, preservation tightens late.
    psi_late_multiplier=1.0,

    # N2: latent-mean anchor. Adds λ·(mean(tgt_x0) - mean(src_x0)) to the final grad,
    # counteracting TAG-driven brightness/saturation drift without a text negative prompt.
    # 0.0 disables. Start around 0.005-0.02.
    latent_mean_anchor_weight=0.0,

    # ---------------------------------------------------------------------
    # 2. TAG branch
    # ---------------------------------------------------------------------
    # eta_tag=1.0 disables TAG.
    eta_tag=1.0,
    adaptive_tag=False,
    asymmetric_tag=False,
    # Optional post-TAG negative-prompt regularizer. This does not change CFG;
    # it subtracts a negative semantic direction after TAG is applied.
    tag_negative_prompt="",
    tag_negative_strength=0.0,

    # ---------------------------------------------------------------------
    # 3. STG branch
    # ---------------------------------------------------------------------
    stg_enabled=False,
    stg_scale=0.5,
    stg_skip_layers=[2],
    stg_schedule_enabled=False,
    stg_decay_start_ratio=0.0,
    stg_decay_end_ratio=0.35,
    # "decay" (original: STG early, off late) or "growth" (STG off early, on late).
    # Use "growth" on creative-transform scenes (stormtrooper) so TAG commits the edit
    # first and STG only refines structure once new geometry has emerged.
    stg_schedule_mode="decay",
    # Coverage-adaptive STG: multiply stg_scale by (1 − previous iteration mask coverage).
    # Small edit regions (face) keep STG near base; large edit regions (stormtrooper)
    # fade STG toward 0. Pairs with outside_mask_anchor_coverage_adaptive for a single
    # config that self-adjusts to both identity-preserving and creative-transform scenes.
    stg_coverage_adaptive=False,

    # ---------------------------------------------------------------------
    # 4. Perp-Neg branch
    # ---------------------------------------------------------------------
    # Global Perp-Neg orthogonalizes eps_tgt with respect to eps_src.
    perp_neg=False,

    # These only have an effect when perp_neg=True.
    depth_masked_perp_neg=True,
    depth_mask_source="cached",              # "depth" or "cached"
    depth_mask_threshold=0.5,                # only used for depth_mask_source="depth"
    cached_mask_dir="data/face_processed/masks",
    perp_neg_mask_dilate=0,                  # hard expansion in image pixels
    perp_neg_mask_blur=30.0,                 # Gaussian blur sigma in image pixels
    perp_neg_alpha=1.0,                      # PN strength (try 0.5-0.75 if too aggressive)
)
