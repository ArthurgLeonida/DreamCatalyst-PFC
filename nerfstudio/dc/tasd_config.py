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
    outside_mask_anchor_weight=0.2,
    outside_mask_anchor_edit_strength_adaptive=True,

    gradient_mask_blur=1.0,
    gradient_mask_gamma=1.2,
    gradient_mask_ema_beta=0.0,
    gradient_mask_warmup=0,

    cross_attention_mask_enabled=True,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=0.7,
    cross_attention_mask_blur=0.5,
    cross_attention_mask_gamma=1.2,

    latent_mean_anchor_weight=0.005,

    # How an externally-supplied mask (3D voxel cache) is fused with the
    # internal hybrid mask. See DCConfig docstring for the math; "screen"
    # is additive-only support that preserves per-view edit-signal peaks
    # while supplying cross-view consensus where the internal mask is weak.
    external_mask_fusion="screen",

    # ---------------------------------------------------------------------
    # 2. TAG branch
    # ---------------------------------------------------------------------
    eta_tag=1.25,            # 1.0 disables TAG.
    adaptive_tag=True,
    asymmetric_tag=True,

    # ---------------------------------------------------------------------
    # 3. STG branch
    # ---------------------------------------------------------------------
    stg_enabled=True,
    stg_scale=2,
    stg_skip_layers=[2],
    stg_schedule_enabled=True,
    stg_schedule_start_ratio=0.4,
    stg_schedule_end_ratio=0.7,
    # Schedule shape:
    #   "decay"  → STG at stg_scale early, fades to 0 by stg_schedule_end_ratio.
    #              Best on identity-preserving edits (face).
    #   "growth" → STG at 0 early, ramps to stg_scale between start and end.
    #              Lets TAG commit the edit first; STG refines structure late.
    #   "bump"   → triangle: 0 → stg_scale between start and peak, back to 0 by end.
    #              Lets STG help mid-phase structural commitment without freezing
    #              view-dependent inconsistencies at the end of training.
    stg_schedule_mode="bump",
    stg_bump_peak_ratio=0.5,
    stg_edit_strength_adaptive=True,

    # ---------------------------------------------------------------------
    # 4. Perp-Neg branch (optional)
    # ---------------------------------------------------------------------
    perp_neg=False,
    perp_neg_alpha=1.0,
)
