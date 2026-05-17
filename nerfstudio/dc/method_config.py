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
    # Reverted to the table-era value (0.2) for the universal-baseline
    # comparison. The drift to 0.18 came after the table was generated.
    outside_mask_anchor_weight=0.2,
    outside_mask_anchor_edit_strength_adaptive=True,

    gradient_mask_blur=0.5,
    gradient_mask_gamma=1.2,
    # Self-mask EMA: stabilizes per-view ||eps_tgt - eps_src|| across
    # iterations. The table-era runs used hand-picked values (0.9 or
    # 0.85) that were probably too low — the same logic that justifies
    # the voxel cache's camera-count-aware β (each revisit contributes
    # 1/(c·N_cam)) applies here. With auto=True, the pipeline resolves
    # β = 1 − 1/(camera_factor · N_cameras) at construction.
    #   N_cam=65, factor=2 → β ≈ 0.9923
    #   N_cam=365, factor=2 → β ≈ 0.9986
    gradient_mask_ema_beta=0.99,
    gradient_mask_ema_beta_auto=True,
    gradient_mask_warmup=0,

    cross_attention_mask_enabled=True,
    cross_attention_mask_layers=[1, 2],
    # Reverted to table-era value (0.7). Drifted to 1 after the table
    # was generated; the CA weight schedule mechanism was added later
    # and is disabled below to match the table's setup.
    cross_attention_mask_weight=0.7,
    cross_attention_mask_blur=0.5,
    cross_attention_mask_gamma=1.2,
    # CA weight schedule did not exist when the table was produced.
    # Disabled here so the cache+universal comparison is apples-to-apples.
    cross_attention_mask_weight_schedule_enabled=False,
    cross_attention_mask_weight_schedule_power=0.5,

    latent_mean_anchor_weight=0.005,

    external_mask_fusion="bidirectional", # bidirectional or screen
    external_mask_screen_attn_gate_strength=1.0, # Gates positive cache support in screen/bidirectional
    external_mask_interp_suppression_ratio=0.3,  # For bidirectional fusion only
    # Extra variance/confidence exponent on the bidirectional negative
    # branch. Both branches inherit cache confidence through `blend_map`,
    # but symmetric gating treats subtraction and addition as equally
    # destructive — empirically the negative branch erodes high-frequency
    # edit detail (stormtrooper armor) while the positive branch is
    # harmless on the same regions. This knob asks the negative branch
    # for stricter agreement. 0.0 disables. 1.0 squares confidence on
    # the negative branch (since blend_map already has one factor).
    # 2.0 cubes it — only very-high-confidence voxels survive.
    external_mask_negative_variance_power=0.0,
    
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
    # Reverted to the table-era value (2.0). Drifted to 2.5 after the
    # stg_tag_compose_mode mechanism was introduced; with sequential
    # composition (the table-era behavior) the effective edit strength
    # is higher, so a lower scale is the right calibration.
    stg_scale=2.0,
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
    # Reverted to "sequential" to match the table-era behavior (the
    # compose_mode knob didn't exist then; sequential was the implicit
    # implementation). Use "parallel" only after ablation-confirming it
    # against this baseline.
    stg_tag_compose_mode="sequential",

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
    mask_voxel_cache_bbox_source="observed",
    mask_voxel_cache_bbox_observe_steps=50,
    mask_voxel_cache_bbox_observe_quantile=0.05,
    mask_voxel_cache_bbox_inflation=0.2,
    # Source for the cache's 2D mask input.
    #   "internal" : post-percentile / EMA / blur mask (current behavior).
    #                Value-compressed; update histogram is unimodal near 0.
    #   "raw_self" : raw per-sample max-normalized ||eps_tgt - eps_src||.
    #                Preserves within-view foreground/background contrast
    #                better than percentile normalization; noisier per-view
    #                but denoised by cross-view aggregation.
    mask_voxel_cache_update_source="raw_self",
    mask_voxel_cache_angular_power=1.0,
    mask_voxel_cache_min_angular_factor=0.0,
    mask_voxel_cache_angular_relative=True,
    # Legacy hardcoded freeze step. Unused now — kept for backward
    # compatibility with older runs in the experiment log. The auto-freeze
    # below supersedes it: it tracks the trusted curve's running max and
    # snapshots the denominator when no improvement is seen for `patience`
    # edit-steps, which catches the scene-specific peak instead of guessing
    # a fixed step that may over- or under-shoot across rigs.
    mask_voxel_cache_angular_freeze_step=2500,
    # Scene-adaptive auto-freeze knobs.
    #   patience: edit-steps of no-improvement before the peak is locked in.
    #             Lower = freezes sooner, more sensitive to noise on the
    #             trusted curve. Higher = more robust, but may miss the
    #             peak if drift starts gradually.
    #   warmup:   edit-steps at the start of the edit during which the
    #             auto-freeze does not track. Avoids treating the cache's
    #             transient first observations as a "peak."
    # 500/50 catches clown's edit-step ~100 peak (freezes around step ~600)
    # and is short enough to act on elf before its slower drift erodes the
    # denominator significantly.
    mask_voxel_cache_angular_freeze_patience=100,
    mask_voxel_cache_angular_freeze_warmup=50,
    # Mass gate (C_mass): explicit replacement for the implicit content
    # coupling that the value-gated angular factor had pre-Fix-B. Damps
    # the cache's confidence on voxels whose cached mean value is below
    # `mass_threshold`. Without this gate, the cache contributes to
    # regions the model isn't actively editing (stormtrooper hand /
    # crotch, elf eyes), producing geometric artifacts and blurring
    # fine detail. With it, the cache's influence is restricted to
    # regions where the diffusion model has committed real edit signal.
    #   threshold = 0.3 → start damping below cache-mean 0.3
    #   power     = 1.0 → linear ramp inside the damped region
    #   power     = 0.0 → gate disabled (legacy)
    mask_voxel_cache_mass_threshold=0.0,
    mask_voxel_cache_mass_power=0.0,
)
