# Custom DreamCatalyst novelty parameters
# Edit this file to configure all novelty settings in one place.
# These values are unpacked into every DCConfig instantiation.

DC_CUSTOM_PARAMS = dict(
    # TAG (Tangential Amplified Guidance)
    eta_tag=1.0,
    adaptive_tag=False,
    asymmetric_tag=False,
    # Perpendicular Gradient Projection (Perp-Neg) — orthogonalize eps_tgt w.r.t. eps_src
    perp_neg=False,
    # Foreground-Masked Perp-Neg — restrict PN subtraction to foreground
    depth_masked_perp_neg=True,
    depth_mask_source="cached",      # "depth" or "cached"
    depth_mask_threshold=0.5,        # for depth source only
    cached_mask_dir="data/face_processed/masks",  # for cached source only
    perp_neg_mask_dilate=0,          # hard expansion in image-pixels (0=off)
    perp_neg_mask_blur=30.0,         # Gaussian blur sigma in image-pixels (soft falloff; 0=off)
    perp_neg_alpha=1.0,              # PN strength (1.0=full, try 0.5-0.75 if too aggressive)
    # STG (Self-attention skip guidance)
    stg_enabled=False,
    stg_scale=0.5,
    stg_skip_layers=[2],
    # Self-derived relevance masking — localize the final DDS gradient using
    # the model's own tgt/src discrepancy. Off by default to preserve current behavior.
    gradient_mask_enabled=False,
    gradient_mask_blur=3.0,          # latent-space Gaussian blur sigma
    gradient_mask_ema_beta=0.9,      # temporal smoothing across iterations
    gradient_mask_gamma=1.0,         # >1 sharpens the mask
    gradient_mask_warmup=50,         # wait N steps before applying the EMA mask
    source_blend_localization_enabled=True,  # blend eps_tgt toward eps_src outside the relevance mask
    outside_mask_anchor_weight=0.0,  # extra x0/source anchoring outside the relevance mask
)
