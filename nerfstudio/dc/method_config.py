# Central DreamCatalyst method parameters.
# DC_CUSTOM_PARAMS → DCConfig
# VOXEL_CACHE_PARAMS → DCPipelineConfig

# Experiment: dc_baseline
# DreamCatalyst baseline — all Part 1 / Part 2 novelties disabled.
# Reproduces the closest in-repo approximation to upstream DreamCatalyst:
# raw DDS + IP2P + FreeU + delta/gamma w_DDS schedule + psi preservation.
# No TAG, no STG/PAG, no self-mask EMA gating, no source blend, no CA mask,
# no outside-mask anchor, no latent-mean anchor, no voxel cache.
#
# NOTE: not bit-identical to upstream KAIST DreamCatalyst — the upstream
# perpendicular-projection step was removed from this repo on 2026-05-18.
# Use this as a "minus our additions" baseline, not an upstream reproduction.

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # 1. Localization branch — OFF
    # ---------------------------------------------------------------------
    psi=0.075,
    source_blend_localization_enabled=False,
    outside_mask_anchor_weight=0.0,
    outside_mask_anchor_edit_strength_adaptive=False,
    outside_mask_anchor_edit_strength_power=1.0,
    outside_mask_anchor_schedule_enabled=False,
    outside_mask_anchor_schedule_power=0.5,
    outside_mask_anchor_schedule_direction="decay",

    gradient_mask_blur=0.0,
    gradient_mask_gamma=1.0,
    gradient_mask_ema_beta=0.99,
    gradient_mask_ema_beta_auto=False,
    gradient_mask_ema_beta_camera_factor=2.0,
    gradient_mask_warmup=0,

    cross_attention_mask_enabled=False,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=0.0,
    cross_attention_mask_blur=0.0,
    cross_attention_mask_gamma=1.0,
    cross_attention_mask_weight_schedule_enabled=False,
    cross_attention_mask_weight_schedule_power=0.5,

    latent_mean_anchor_weight=0.0,

    external_mask_fusion="bidirectional",
    external_mask_screen_attn_gate_strength=1.0,
    external_mask_interp_suppression_ratio=0.3,
    external_mask_negative_variance_power=0.0,
    external_mask_screen_self_boost_lambda=1.0,

    # ---------------------------------------------------------------------
    # 2. TAG branch — OFF (eta_tag=1 → identity)
    # ---------------------------------------------------------------------
    eta_tag=1.0,
    adaptive_tag=False,
    asymmetric_tag=False,

    # ---------------------------------------------------------------------
    # 3. STG / PAG branch — OFF
    # ---------------------------------------------------------------------
    stg_enabled=False,
    stg_scale=0.0,
    stg_skip_layers=[2],
    stg_schedule_enabled=False,
    stg_schedule_start_ratio=0.4,
    stg_schedule_end_ratio=0.7,
    stg_schedule_mode="bump",
    stg_bump_peak_ratio=0.5,
    stg_edit_strength_adaptive=False,
    stg_tag_compose_mode="parallel",
    stg_weak_method="stg",
)


# ---------------------------------------------------------------------
# 4. 3D voxel-cache localization — OFF
# ---------------------------------------------------------------------
VOXEL_CACHE_PARAMS = dict(
    mask_voxel_cache_enabled=False,
    mask_voxel_cache_resolution=64,

    mask_voxel_cache_ema_beta=0.99,
    mask_voxel_cache_ema_beta_auto=True,
    mask_voxel_cache_ema_beta_camera_factor=2.0,

    mask_voxel_cache_warmup_start=1300,
    mask_voxel_cache_warmup_end=2500,
    mask_voxel_cache_max_blend=0.4,
    mask_voxel_cache_accumulation_threshold=0.3,
    mask_voxel_cache_update_threshold=0.2,

    mask_voxel_cache_confidence_enabled=True,
    mask_voxel_cache_min_observations=3,
    mask_voxel_cache_min_observations_auto=True,
    mask_voxel_cache_observation_fraction=0.10,
    mask_voxel_cache_min_observations_floor=5,
    mask_voxel_cache_min_observations_cap=12,
    mask_voxel_cache_max_variance=0.04,

    mask_voxel_cache_bbox_source="observed",  # observed | cameras | scene_box
    mask_voxel_cache_bbox_observe_steps=50,
    mask_voxel_cache_bbox_observe_quantile=0.05,
    mask_voxel_cache_bbox_inflation=0.2,

    mask_voxel_cache_update_source="raw_self",  # raw_self | internal

    mask_voxel_cache_angular_power=1.0,
    mask_voxel_cache_min_angular_factor=0.0,
    mask_voxel_cache_angular_relative=True,
    mask_voxel_cache_angular_freeze_patience=100,
    mask_voxel_cache_angular_freeze_warmup=50,

    mask_voxel_cache_mass_threshold=0.0,
    mask_voxel_cache_mass_power=0.0,
)
