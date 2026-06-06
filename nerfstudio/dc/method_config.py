# Central DreamCatalyst method parameters.
# DC_CUSTOM_PARAMS → DCConfig
# VOXEL_CACHE_PARAMS → DCPipelineConfig

# Experiment: voxel "view-invariant force" package on top of the standard Part-1 config.
# DC_CUSTOM_PARAMS = the proposed universal 2D config (source-blend + CA mask +
# adaptive/asymmetric TAG + STG bump + outside/latent anchors) — the same config
# that produced the Part-1 "standard" results.
# VOXEL_CACHE_PARAMS = 3D cache ON, lifting raw_self (edit force). The package
# targets the helps-clown/hurts-faces split by extracting only the view-invariant
# component of the edit force:
#   1. raw_self normalized by p95 (gradient_mask_raw_norm_quantile=0.95) — kills
#      single-pixel-max jitter that faked cross-view variance.
#   2. positive-only fusion (external_mask_interp_suppression_ratio=0.0) — the
#      cache may add agreed-upon force, never subtract (was 0.3, eroded detail).
#   3. tighter agreement gate (max_variance=0.02, was 0.035) — only genuinely
#      view-consistent voxels contribute. Sweep down per measure-only baseline.
#   mass 0.18/pow1.0, angular gate power 1.0, warmup 300/1100 unchanged.
#   trilinear cache read implemented but OFF (isolate the above first).
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
    # Robust per-frame scale for the raw_self cache input: divide by p95 instead
    # of the single-pixel max so one hot pixel can't deflate the whole frame and
    # inject spurious cross-view variance into the voxel cache. 1.0 = legacy max.
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
    # Positive-only cache fusion: 0.0 disables the negative (subtractive) branch.
    # With raw_self the cache mean sits below this view's true peak on view-
    # dependent regions, so a negative branch erodes legitimate edit force
    # (stormtrooper armor, elf eyes). 0.0 lets the cache only ADD agreed-upon
    # force, never subtract. (Was 0.3.) See VoxelCacheExplained §5b.
    external_mask_interp_suppression_ratio=0.0,
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
    mask_voxel_cache_max_variance=0.02,            # tightened from 0.035: with the robust raw_self norm (p95) spurious variance drops, so the gate should only pass genuinely view-invariant force. Sweep {0.02, 0.015, 0.01} after reading measure-only baseline variance per scene.

    mask_voxel_cache_bbox_source="observed",  # observed | cameras | scene_box
    mask_voxel_cache_bbox_observe_steps=50,
    mask_voxel_cache_bbox_observe_quantile=0.05,
    mask_voxel_cache_bbox_inflation=0.2,

    mask_voxel_cache_update_source="raw_self", # raw_self | internal | raw_attn  -- lift the edit-force (||eps_tgt-eps_src||) mask, not the semantic CA mask
    # Observed-weighted trilinear cache read. Implemented and ready; kept OFF so
    # the first experiment isolates the raw_self-normalization + fusion + gate
    # changes. Flip to True to ablate the discretization fix (no density penalty,
    # unlike raising resolution). See dc_pipeline mask_voxel_cache_trilinear.
    mask_voxel_cache_trilinear=False,

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
