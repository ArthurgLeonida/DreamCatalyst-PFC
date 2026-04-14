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

    # These parameters matter whenever source_blend_localization_enabled,
    # gradient_mask_enabled, or outside_mask_anchor_weight is active.
    gradient_mask_blur=2.0,      # latent-space Gaussian blur sigma
    gradient_mask_gamma=1.2,     # >1 = tighter mask, <1 = broader mask
    gradient_mask_ema_beta=0,  # temporal smoothing per view; 0 disables EMA
    gradient_mask_warmup=0,     # during warmup the mask is all ones

    # ---------------------------------------------------------------------
    # 2. TAG branch
    # ---------------------------------------------------------------------
    # eta_tag=1.0 disables TAG.
    eta_tag=1.0,
    adaptive_tag=False,
    asymmetric_tag=False,

    # ---------------------------------------------------------------------
    # 3. STG branch
    # ---------------------------------------------------------------------
    stg_enabled=False,
    stg_scale=0.5,
    stg_skip_layers=[2],

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
