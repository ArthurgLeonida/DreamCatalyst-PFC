# Central DreamCatalyst method parameters.
# DC_CUSTOM_PARAMS → DCConfig
# VOXEL_CACHE_PARAMS → DCPipelineConfig

# DC_CUSTOM_PARAMS = universal 2D config (source-blend + CA mask + adaptive/asymmetric
# TAG + STG bump + outside/latent anchors) — the Part-1 "standard" config.
# VOXEL_CACHE_PARAMS = the Part-2 3D cache, lifting raw_self (edit force) into a voxel
# grid for cross-view-consistent localization. Active package:
#   - raw_self p95-normalized (gradient_mask_raw_norm_quantile=0.95) — robust per-frame
#     scale that avoids single-pixel-max jitter inflating cross-view variance.
#   - positive-only fusion (external_mask_interp_suppression_ratio=0.0): the cache adds
#     agreed-upon force, never subtracts.
#   - observed-weighted trilinear readback (mask_voxel_cache_trilinear=True) — converts
#     mask-level consistency into rendered multi-view consistency (the elf MV_cos gain).
#   - decayed/EW cross-view variance (mask_voxel_cache_variance_decay=0.2): recency-
#     weighted, principled for the non-stationary edit; re-tune max_variance per scene.
#   - agreement gate max_variance: TUNE PER SCENE against the variance map
#     (dc_debug/voxel_cache_variance_map), in the gap between the wanted-edit variance
#     (low) and the over-edit-region variance (high), above the former.
#   - confidence gate (count + variance + angular diversity + mass), scale-matching,
#     warmup-ramped blend (max_blend).
# STG note: the cache config uses stg_scale=3.0 (vs 3.5 for the cache-off Part-1 runs)
# as a design rebalancing — the cache adds agreement-gated localization support, so the
# guidance amplifier is reduced to avoid double-amplifying the edit signal. The
# editability difference is within the run-to-run noise floor; this is a config choice,
# not a measured win. Set stg_scale=3.5 to reproduce Part-1.
# Scene is selected via the scripts/edit.sh argument, not here.

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
    gradient_mask_raw_norm_quantile=0.95,

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
    external_mask_interp_suppression_ratio=0.0,
    external_mask_negative_variance_power=0.0,
    external_mask_screen_self_boost_lambda=1.0,

    # ---------------------------------------------------------------------
    # 2. TAG branch — ON (adaptive + asymmetric; eta_tag>1 amplifies tangential)
    # ---------------------------------------------------------------------
    eta_tag=1.2,
    adaptive_tag=True,
    asymmetric_tag=True,

    # ---------------------------------------------------------------------
    # 3. STG / PAG branch
    # ---------------------------------------------------------------------
    stg_enabled=True,
    stg_scale=3.0,  # cache-on (Part 2) rebalancing; use 3.5 for cache-off Part-1 runs
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
    mask_voxel_cache_measure_only=False,
    mask_voxel_cache_resolution=64,

    mask_voxel_cache_ema_beta=0.99,
    mask_voxel_cache_ema_beta_auto=True,
    mask_voxel_cache_ema_beta_camera_factor=2.0,

    mask_voxel_cache_warmup_start=500, # clown override: gentle/late warmup (universal: 100)
    mask_voxel_cache_warmup_end=1200, # clown override (universal: 1100)
    mask_voxel_cache_max_blend=0.2,   # validated package value — 0.4 doubles the cache push and accelerates the arm over-edit feedback loop; keep 0.2 while fighting over-edits
    mask_voxel_cache_accumulation_threshold=0.30,
    mask_voxel_cache_update_threshold=0.0,

    mask_voxel_cache_confidence_enabled=True,
    mask_voxel_cache_min_observations=8,
    mask_voxel_cache_min_observations_auto=True,
    mask_voxel_cache_observation_fraction=0.10,
    mask_voxel_cache_min_observations_floor=5,
    mask_voxel_cache_min_observations_cap=12,      # saturates the auto-rule at N_cam > 120 for cross-scene consistency
    mask_voxel_cache_max_variance=0.02,           # tune per scene from the variance map: set in the wanted-edit↔over-edit gap (head ≈ dark, arms ≈ bright)
    mask_voxel_cache_variance_decay=0.2,

    mask_voxel_cache_bbox_source="observed",  # observed | cameras | scene_box
    mask_voxel_cache_bbox_observe_steps=50,
    mask_voxel_cache_bbox_observe_quantile=0.05,
    mask_voxel_cache_bbox_inflation=0.2,

    mask_voxel_cache_update_source="raw_self", # raw_self | internal | raw_attn 
    mask_voxel_cache_trilinear=True, # Test with True

    mask_voxel_cache_angular_power=1.0,    # was 1.0
    mask_voxel_cache_min_angular_factor=0.0,
    mask_voxel_cache_angular_relative=True,
    mask_voxel_cache_angular_freeze_patience=100,
    mask_voxel_cache_angular_freeze_warmup=50,

    mask_voxel_cache_mass_threshold=0.18,   # was 0.15 | 0.18 | 0.22
    mask_voxel_cache_mass_power=1.0,       # was 1.0 | 1.5 | 2.0

    mask_voxel_cache_scale_normalize=True,
    mask_voxel_cache_scale_normalize_quantile=0.95,
)