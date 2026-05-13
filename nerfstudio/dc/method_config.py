# Central DreamCatalyst method parameters.
#
# Edit this file to configure the active Step-3 editing method. Values in
# DC_CUSTOM_PARAMS are unpacked into every DCConfig instantiation. Values in
# VOXEL_CACHE_PARAMS are imported by DCPipelineConfig as pipeline-level
# defaults.
#
# Suggested tuning order:
# 1. Localization branch
# 2. TAG branch
# 3. STG branch
# 4. Perp-Neg branch
# 5. 3D voxel-cache localization

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # 1. Localization branch
    # ---------------------------------------------------------------------
    psi=0.075,
    source_blend_localization_enabled=True,
    gradient_mask_enabled=False,
    outside_mask_anchor_weight=0.18,
    outside_mask_anchor_edit_strength_adaptive=True,

    gradient_mask_blur=0.5,
    gradient_mask_gamma=1.2,
    gradient_mask_ema_beta=0,
    gradient_mask_warmup=0,

    cross_attention_mask_enabled=True,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=1,
    cross_attention_mask_blur=0.5,
    cross_attention_mask_gamma=1.2,
    cross_attention_mask_weight_schedule_enabled=True,
    cross_attention_mask_weight_schedule_power=0.5,

    latent_mean_anchor_weight=0.005,

    external_mask_fusion="bidirectional", # bidirectional or screen
    external_mask_screen_attn_gate_strength=1.0, # Gates positive cache support in screen/bidirectional
    external_mask_interp_suppression_ratio=1.0,  # For bidirectional fusion only
    
    # Which signal opens positive voxel-cache support in screen/bidirectional modes. Options:
    #   "ca"            : M_attn (semantic; late-confirmation signal)
    #   "self"          : M_self (responsive to raw DDS delta; circular risk)
    #   "hybrid_max"    : max(M_self, M_attn) — most aggressive, recovers
    #                     late-forming features (helmet) but may amplify
    #                     per-view artifacts
    #   "hybrid_mean"   : 0.5(M_self + M_attn) — averaged; can underperform
    #                     pure CA when M_self < M_attn in target region
    #   "self_boost"    : M_attn + λ · max(M_self − M_attn, 0) — monotone over
    #                     CA (gate ≥ M_attn always); self contributes only
    #                     where it discovers signal CA missed. Best universal
    #                     candidate for late-forming features (helmet).

    external_mask_screen_gate_source="self_boost",
    external_mask_screen_self_boost_lambda=1.0, # For self_boost gate source only

    # ---------------------------------------------------------------------
    # 2. TAG branch
    # ---------------------------------------------------------------------
    eta_tag=1.25,  # 1.0 disables TAG.
    adaptive_tag=True,
    asymmetric_tag=True,

    # ---------------------------------------------------------------------
    # 3. STG branch
    # ---------------------------------------------------------------------
    stg_enabled=True,
    stg_scale=2.5,
    stg_skip_layers=[2],
    stg_schedule_enabled=True,
    stg_schedule_start_ratio=0.4,
    stg_schedule_end_ratio=0.7,
    # Schedule shape:
    #   "decay"  -> STG at stg_scale early, fades to 0 by end_ratio.
    #   "growth" -> STG at 0 early, ramps to stg_scale between start/end.
    #   "bump"   -> triangle: 0 -> stg_scale -> 0 inside the window.
    stg_schedule_mode="bump",
    stg_bump_peak_ratio=0.5,
    stg_edit_strength_adaptive=True,
    # How STG and TAG compose. See DCConfig docstring for the algebra.
    #   "sequential" : TAG runs on the STG-amplified signal. Compounds an
    #                  extra s·(η−1)·(eps_full−eps_weak)_⊥ tangential boost
    #                  onto the STG direction.
    #   "parallel"   : TAG and STG act independently on the raw CFG signal,
    #                  then are summed. Each operator has a bounded effect.
    #                  Cleaner separation for the writeup.
    stg_tag_compose_mode="parallel",

    # ---------------------------------------------------------------------
    # 4. Perp-Neg branch (optional)
    # ---------------------------------------------------------------------
    perp_neg=False,
    perp_neg_alpha=1.0,
)


# 3D voxel-cache localization parameters.
VOXEL_CACHE_PARAMS = dict(
    mask_voxel_cache_enabled=True,
    mask_voxel_cache_resolution=64,
    # Manual fallback if `mask_voxel_cache_ema_beta_auto=False`.
    mask_voxel_cache_ema_beta=0.99,
    # Camera-count-aware EMA. With 65 cameras and factor=2:
    # beta = 1 - 1 / (2 * 65) = 0.9923.
    # This keeps the cache stable across views without freezing it as hard as
    # fixed beta=0.999 on medium-sized captures.
    mask_voxel_cache_ema_beta_auto=True,
    mask_voxel_cache_ema_beta_camera_factor=2.0,
    mask_voxel_cache_warmup_start=1300,
    mask_voxel_cache_warmup_end=2500,
    mask_voxel_cache_max_blend=0.4,
    mask_voxel_cache_accumulation_threshold=0.3,
    mask_voxel_cache_update_threshold=0.25,
    # Confidence-aware 3D mask trust. The cache still learns every confident
    # observation, but fusion only trusts voxels after enough repeated
    # world-space observations agree. This is the NeRF-friendly version of a
    # TransSplat-style residual / agreement prior and a cheap correspondence
    # consistency proxy.
    mask_voxel_cache_confidence_enabled=True,
    # Manual fallback if `mask_voxel_cache_min_observations_auto=False`.
    mask_voxel_cache_min_observations=3,
    # Camera-count-aware trust threshold:
    # min_obs = clamp(ceil(N_cameras * fraction), floor, cap).
    # With 65 cameras and fraction=0.05 this gives 4 observations.
    mask_voxel_cache_min_observations_auto=True,
    mask_voxel_cache_observation_fraction=0.10,
    mask_voxel_cache_min_observations_floor=5,
    mask_voxel_cache_min_observations_cap=12,
    # Variance is measured on soft mask values in [0, 1]. 0.04 means a
    # standard deviation of 0.2; voxels above this are treated as unreliable.
    mask_voxel_cache_max_variance=0.04,
    mask_voxel_cache_bbox_source="observed",
    mask_voxel_cache_bbox_observe_steps=50,
    mask_voxel_cache_bbox_observe_quantile=0.05,
    mask_voxel_cache_bbox_inflation=0.2,
)
