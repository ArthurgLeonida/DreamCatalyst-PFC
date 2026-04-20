# Custom DreamCatalyst novelty parameters
# Edit this file to configure all novelty settings in one place.
# These values are unpacked into every DCConfig instantiation.
#
# Suggested tuning order:
# 1. Localization branch (main current research direction)
# 2. TAG branch (edit-strength novelties)
# 3. STG branch (structural amplification with schedule / coverage adaptation)
# 4. Perp-Neg branch (optional creative-editing projection)

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # 1. Localization branch
    # ---------------------------------------------------------------------
    psi=0.075,
    source_blend_localization_enabled=True,
    gradient_mask_enabled=False,
    outside_mask_anchor_weight=0.05,
    # Coverage-adaptive outside-mask anchor: scales outside_mask_anchor_weight by
    # (1 − per-sample mean(grad_mask)). Face (small coverage) keeps the bg anchor tight;
    # stormtrooper (large coverage) loosens it automatically. Lets one anchor value
    # work across identity-preserving and creative-transform scenes.
    outside_mask_anchor_coverage_adaptive=False,

    # Self-mask post-processing
    gradient_mask_blur=1.0,      # latent-space Gaussian blur sigma
    gradient_mask_gamma=1.2,     # >1 = tighter mask, <1 = broader mask
    gradient_mask_ema_beta=0.0,  # per-view temporal smoothing; 0 disables
    gradient_mask_warmup=0,      # during warmup the mask is all ones

    # Cross-attention semantic prior. Token selection is auto-derived from src/tgt
    # prompts (target-only words minus stopwords) to stay fully general across scenes.
    cross_attention_mask_enabled=False,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=1.0,
    cross_attention_mask_blur=0.0,
    cross_attention_mask_gamma=1.0,

    # M1 (ablation): ignore M_self, use M_attn alone. Refuted on IP2P face (CLIPd 0.129
    # vs W 0.143). Kept as a clean ablation switch.
    cross_attention_mask_only=False,
    # M2 (ablation): intersect (1 − M_self) with M_attn. Catastrophic on face
    # (CLIPd −0.02, no edit). Kept as a clean ablation switch.
    invert_self_mask=False,

    # DaCapo-inspired ψ schedule (Huang et al., CVPR 2025). =1.0 disables.
    # >1 grows preservation as t decreases: edit commits early, preservation tightens late.
    psi_late_multiplier=1.0,

    # N2: latent-mean anchor. Adds λ·(mean(tgt_x0) − mean(src_x0)) to the final grad,
    # counteracting TAG-driven brightness/saturation drift without a negative text prompt.
    # 0.0 disables. 0.005 is the empirically best value on face.
    latent_mean_anchor_weight=0.0,

    # ---------------------------------------------------------------------
    # 2. TAG branch
    # ---------------------------------------------------------------------
    eta_tag=1.0,            # 1.0 disables TAG.
    adaptive_tag=False,
    asymmetric_tag=False,
    # Post-TAG negative-prompt regularizer (subtracts a negative semantic direction after TAG).
    # Exploratory: did not convincingly beat non-neg runs on face.
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
    stg_decay_end_ratio=1.0,
    # Schedule shape:
    #   "decay"  → STG at stg_scale early, fades to 0 by stg_decay_end_ratio.
    #              Best on identity-preserving edits (face).
    #   "growth" → STG at 0 early, ramps to stg_scale between start and end.
    #              Lets TAG commit the edit first; STG refines structure late.
    #   "bump"   → triangle: 0 → stg_scale between start and peak, back to 0 by end.
    #              Lets STG help mid-phase structural commitment without freezing
    #              view-dependent inconsistencies at the end of training.
    stg_schedule_mode="decay",
    # Only used in "bump" mode. Fraction within [start, end] where STG peaks.
    # 0.5 = symmetric triangle; >0.5 = faster rise, slower decay.
    stg_bump_peak_ratio=0.5,
    # Coverage-adaptive STG: multiply stg_scale by (1 − previous iteration mask coverage).
    # Face (small coverage) keeps STG near base; stormtrooper (large coverage) fades STG
    # toward 0. Pairs with outside_mask_anchor_coverage_adaptive for a single config
    # self-adjusting across identity-preserving and creative-transform scenes.
    stg_coverage_adaptive=False,

    # ---------------------------------------------------------------------
    # 4. Perp-Neg branch (optional)
    # ---------------------------------------------------------------------
    perp_neg=False,
    perp_neg_alpha=1.0,
)
