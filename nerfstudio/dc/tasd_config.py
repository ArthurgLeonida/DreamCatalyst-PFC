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
    # Depth-Masked Perp-Neg — restrict projection to foreground using renderer depth map
    depth_masked_perp_neg=False,
    depth_mask_threshold=0.5,
    # STG (Self-attention skip guidance)
    stg_enabled=False,
    stg_scale=0.5,
    stg_skip_layers=[2],
)
