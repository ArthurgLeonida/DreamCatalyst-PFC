# Central DreamCatalyst method parameters.
# DC_CUSTOM_PARAMS → DCConfig
# VOXEL_CACHE_PARAMS → DCPipelineConfig

# Experiment: voxel MaxVar0035 (full gates) on top of the standard Part-1 config.
# DC_CUSTOM_PARAMS = the proposed universal 2D config (source-blend + CA mask +
# adaptive/asymmetric TAG + STG bump + outside/latent anchors) — the same config
# that produced the Part-1 "standard" results.
# VOXEL_CACHE_PARAMS = 3D cache ON, MaxVar0035 full-gates: max_variance=0.035,
# mass 0.18/pow1.0, warmup 300/1100, full negative correction (interp 0.3) and
# angular gate (power 1.0). Used for the Part-2 bear/elf headroom experiments;
# the scene is selected via the scripts/edit.sh argument, not here.

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # 1. Localization branch
    # ---------------------------------------------------------------------
    psi=0.075,
    source_blend_localization_enabled=True,
    source_blend_floor=0.0,  # 0.0 = hard gate; ~0.1 lets edits grow into M≈0 regions

    outside_mask_anchor_weight=0.15,
    outside_mask_anchor_edit_strength_adaptive=True,
    outside_mask_anchor_edit_strength_power=1.0,
    outside_mask_anchor_schedule_enabled=False,
    outside_mask_anchor_schedule_power=0.75,
    outside_mask_anchor_schedule_direction="growth",

    gradient_mask_blur=0.5,
    gradient_mask_gamma=1.2,
    gradient_mask_ema_beta=0.0,
    gradient_mask_ema_beta_auto=True,
    gradient_mask_ema_beta_camera_factor=2.0,
    gradient_mask_warmup=0,

    cross_attention_mask_enabled=True,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=1.0,
    cross_attention_mask_blur=0.5,
    cross_attention_mask_gamma=1.2,
    cross_attention_mask_weight_schedule_enabled=True,
    cross_attention_mask_weight_schedule_power=0.75, # Most of the experiments use 0.75

    latent_mean_anchor_weight=0.005,

    external_mask_fusion="bidirectional",
    external_mask_screen_attn_gate_strength=1.0,
    external_mask_interp_suppression_ratio=0.3,
    external_mask_negative_variance_power=0.0,
    external_mask_screen_self_boost_lambda=1.0,

    # ---------------------------------------------------------------------
    # 2. TAG branch — OFF (eta_tag=1 → identity)
    # ---------------------------------------------------------------------
    eta_tag=1.2,
    adaptive_tag=True,
    asymmetric_tag=True,

    # ---------------------------------------------------------------------
    # 3. STG / PAG branch
    # ---------------------------------------------------------------------
    stg_enabled=True,
    stg_scale=3.5,
    stg_skip_layers=[2],
    stg_schedule_enabled=True,
    stg_schedule_start_ratio=0, #  0.3 for decay
    stg_schedule_end_ratio=0.8, # 0.8 for decay
    stg_schedule_mode="bump", # bump | growth | decay
    stg_bump_peak_ratio=0.5,
    stg_edit_strength_adaptive=True,
    stg_tag_compose_mode="parallel",
    stg_weak_method="stg",
)


# ---------------------------------------------------------------------
# 4. 3D voxel-cache localization
# ---------------------------------------------------------------------
VOXEL_CACHE_PARAMS = dict(
    mask_voxel_cache_enabled=True,
    # Passive cache-off control (Part-2 A/B): when True AND enabled=False, the
    # cache is built and its cross-view Welford variance is logged, but it is
    # never blended into the DDS gradient -- the edit trajectory matches the
    # standard run exactly. Lets you compare voxel_cache_mean_observed_variance
    # cache-on vs cache-off on the statistic the cache actually manipulates (the
    # training-time latent mask), which EditMaskVariance_3D in eval cannot see.
    mask_voxel_cache_measure_only=False,
    mask_voxel_cache_resolution=64,

    mask_voxel_cache_ema_beta=0.99,
    mask_voxel_cache_ema_beta_auto=True,
    mask_voxel_cache_ema_beta_camera_factor=2.0,

    mask_voxel_cache_warmup_start=300,
    mask_voxel_cache_warmup_end=1100,
    mask_voxel_cache_max_blend=0.4,
    mask_voxel_cache_accumulation_threshold=0.30,
    mask_voxel_cache_update_threshold=0.0,

    mask_voxel_cache_confidence_enabled=True,
    mask_voxel_cache_min_observations=8,
    mask_voxel_cache_min_observations_auto=True,
    mask_voxel_cache_observation_fraction=0.10,
    mask_voxel_cache_min_observations_floor=5,
    mask_voxel_cache_min_observations_cap=12,      # saturates the auto-rule at N_cam > 120 for cross-scene consistency
    mask_voxel_cache_max_variance=0.035,           # MaxVar0035 (best clown config); looser gate = denser field, matters on high-variance scenes (bear/elf)

    mask_voxel_cache_bbox_source="observed",  # observed | cameras | scene_box
    mask_voxel_cache_bbox_observe_steps=50,
    mask_voxel_cache_bbox_observe_quantile=0.05,
    mask_voxel_cache_bbox_inflation=0.2,

    mask_voxel_cache_update_source="raw_self",     # avoid CA-schedule leakage into cache observations

    mask_voxel_cache_angular_power=1.0,    # was 1.0
    mask_voxel_cache_min_angular_factor=0.0,
    mask_voxel_cache_angular_relative=True,
    mask_voxel_cache_angular_freeze_patience=100,
    mask_voxel_cache_angular_freeze_warmup=50,

    mask_voxel_cache_mass_threshold=0.18,   # was 0.15 | 0.18 | 0.22
    mask_voxel_cache_mass_power=1.0,       # was 1.0 | 1.5 | 2.0
)
