# Central DreamCatalyst method parameters — the single source of truth for
# which knobs are active in a run. Scene selection happens in scripts/edit.sh.
#
# DC_CUSTOM_PARAMS → DCConfig. The Part-1 "standard" universal 2D config:
#   source-blend localization + hybrid self/CA mask + adaptive asymmetric TAG
#   + STG bump schedule + outside-mask and latent-mean anchors.
#
# VOXEL_CACHE_PARAMS → DCPipelineConfig. The Part-2 3D voxel cache: lifts the
# raw self-mask (per-frame p95-normalized edit magnitude) into a voxel grid for
# cross-view-consistent localization. Positive-only fusion, observed-weighted
# trilinear readback, decayed cross-view variance, and a confidence gate
# (count + variance + angular diversity + mass) with scale matching and a
# warmup-ramped blend. Full walkthrough: docs/VoxelCacheExplained.md.
#
# The agreement gate max_variance is scene-sensitive: tune it against the
# variance map (dc_debug/voxel_cache_variance_map), above the wanted-edit
# variance and below the over-edit variance.
#
# ---------------------------------------------------------------------------
# The three evaluated configurations
# ---------------------------------------------------------------------------
# As committed, this file is the Part-2 "voxel cache" configuration. The other
# two are obtained by editing only the values listed below:
#
# 1. Part-1 "standard" (universal 2D config, cache off):
#      stg_scale=3.5, mask_voxel_cache_enabled=False
#
# 2. "dc baseline" (upstream DreamCatalyst: raw DDS + IP2P + FreeU + psi/delta/gamma):
#      source_blend_localization_enabled=False,
#      outside_mask_anchor_weight=0.0,
#      outside_mask_anchor_edit_strength_adaptive=False,
#      cross_attention_mask_enabled=False,
#      latent_mean_anchor_weight=0.0,
#      eta_tag=1.0, adaptive_tag=False, asymmetric_tag=False,
#      stg_enabled=False, stg_scale=0.0,
#      gradient_mask_ema_beta=0.99, gradient_mask_ema_beta_auto=False,
#      mask_voxel_cache_enabled=False
#    (with all localization flags off no relevance mask is built, so the
#    remaining mask-shaping values — blur/gamma/quantile — are inert)

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # 1. Localization branch
    # ---------------------------------------------------------------------
    psi=0.075,
    source_blend_localization_enabled=True,

    outside_mask_anchor_weight=0.15,
    outside_mask_anchor_edit_strength_adaptive=True,

    gradient_mask_blur=0.5,
    gradient_mask_gamma=1.2,
    gradient_mask_ema_beta=0.0,
    gradient_mask_ema_beta_auto=True,
    gradient_mask_ema_beta_camera_factor=2.0,
    gradient_mask_raw_norm_quantile=0.95,

    cross_attention_mask_enabled=True,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_blur=0.5,
    cross_attention_mask_gamma=1.2,
    cross_attention_mask_weight_schedule_power=0.75,

    latent_mean_anchor_weight=0.005,

    # ---------------------------------------------------------------------
    # 2. TAG branch — ON (adaptive + asymmetric; eta_tag>1 amplifies tangential)
    # ---------------------------------------------------------------------
    eta_tag=1.2,
    adaptive_tag=True,
    asymmetric_tag=True,

    # ---------------------------------------------------------------------
    # 3. STG branch
    # ---------------------------------------------------------------------
    stg_enabled=True,
    stg_scale=3.0,  # 3.5 reproduces the cache-off Part-1 results
    stg_skip_layers=[2],
    stg_schedule_start_ratio=0,
    stg_schedule_end_ratio=0.8,
    stg_bump_peak_ratio=0.5,
)


# ---------------------------------------------------------------------
# 4. 3D voxel-cache localization
# ---------------------------------------------------------------------
VOXEL_CACHE_PARAMS = dict(
    mask_voxel_cache_enabled=True,
    mask_voxel_cache_resolution=64,

    mask_voxel_cache_ema_beta_camera_factor=2.0,

    mask_voxel_cache_warmup_start=500,
    mask_voxel_cache_warmup_end=1200,
    mask_voxel_cache_max_blend=0.2,
    mask_voxel_cache_accumulation_threshold=0.30,

    mask_voxel_cache_observation_fraction=0.10,
    mask_voxel_cache_min_observations_floor=5,
    mask_voxel_cache_min_observations_cap=12,
    mask_voxel_cache_max_variance=0.02,
    mask_voxel_cache_variance_decay=0.2,

    mask_voxel_cache_bbox_observe_steps=50,
    mask_voxel_cache_bbox_observe_quantile=0.05,
    mask_voxel_cache_bbox_inflation=0.2,

    mask_voxel_cache_angular_freeze_patience=100,
    mask_voxel_cache_angular_freeze_warmup=50,

    mask_voxel_cache_mass_threshold=0.18,

    mask_voxel_cache_scale_normalize_quantile=0.95,
)
