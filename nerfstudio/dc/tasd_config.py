# Custom DreamCatalyst novelty parameters
# Edit this file to configure all novelty settings in one place.
# These values are unpacked into every DCConfig instantiation.

DC_CUSTOM_PARAMS = dict(
    # TAG (Tangential Amplified Guidance)
    eta_tag=1.0,
    adaptive_tag=False,
    asymmetric_tag=False,
    # Conflict-Free Guidance (project out conflicting component)
    conflict_free=False,
    # STG (Self-attention skip guidance)
    stg_enabled=False,
    stg_scale=1.0,
    stg_skip_layers=[1, 2],
)
