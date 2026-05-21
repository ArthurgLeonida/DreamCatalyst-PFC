# Central DreamCatalyst method parameters.
# DC_CUSTOM_PARAMS → DCConfig
# VOXEL_CACHE_PARAMS → DCPipelineConfig

# Experiment: st_PAG_TAG115_STG25_CAsch05_bg015
# PAG (Perturbed-Attention Guidance) ablation: replaces STG's QK-skip
# perturbation with PAG's identity-self-attention perturbation. Same scale,
# same skip-layers, same schedule — only the weak-prediction mechanism
# differs. Tests whether the symmetric-body ghost is driven by STG's
# specific cross-attention amplification pattern (which PAG should avoid).
# Same config as ghost-present baseline `CAsch075_TAG115_STG25_bg015`
# (CLIP_dir 0.169, helmet+ghost) except stg_weak_method "stg" → "pag".

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # 1. Localization branch
    # ---------------------------------------------------------------------
    psi=0.075,
    source_blend_localization_enabled=True,
    outside_mask_anchor_weight=0.15,
    outside_mask_anchor_edit_strength_adaptive=True,
    outside_mask_anchor_edit_strength_power=1.0,

    gradient_mask_blur=0.5,
    gradient_mask_gamma=1.2,
    gradient_mask_ema_beta=0.99,
    gradient_mask_ema_beta_auto=True,
    gradient_mask_warmup=0,

    cross_attention_mask_enabled=True,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=1.0,
    cross_attention_mask_blur=0.5,
    cross_attention_mask_gamma=1.2,
    cross_attention_mask_weight_schedule_enabled=True,
    cross_attention_mask_weight_schedule_power=0.5,

    latent_mean_anchor_weight=0.005,

    external_mask_fusion="bidirectional",  # bidirectional | screen
    external_mask_screen_attn_gate_strength=1.0,
    external_mask_interp_suppression_ratio=0.3,
    external_mask_negative_variance_power=0.0,
    external_mask_screen_self_boost_lambda=1.0,

    # ---------------------------------------------------------------------
    # 2. TAG branch
    # ---------------------------------------------------------------------
    eta_tag=1.15,
    adaptive_tag=True,
    asymmetric_tag=True,

    # ---------------------------------------------------------------------
    # 3. STG / PAG branch
    # ---------------------------------------------------------------------
    stg_enabled=True,
    stg_scale=2.5,
    stg_skip_layers=[2],
    stg_schedule_enabled=True,
    stg_schedule_start_ratio=0.4,
    stg_schedule_end_ratio=0.7,
    stg_schedule_mode="bump",  # decay | growth | bump
    stg_bump_peak_ratio=0.5,
    stg_edit_strength_adaptive=True,
    stg_tag_compose_mode="parallel",  # sequential | parallel
    stg_weak_method="pag",  # stg | pag  ← THE TEST KNOB
)


# ---------------------------------------------------------------------
# 4. 3D voxel-cache localization
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
