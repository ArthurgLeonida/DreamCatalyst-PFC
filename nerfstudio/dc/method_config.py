# Central DreamCatalyst method parameters.
# DC_CUSTOM_PARAMS → DCConfig
# VOXEL_CACHE_PARAMS → DCPipelineConfig

# Experiment: voxel "view-invariant force" package on top of the standard Part-1 config.
# DC_CUSTOM_PARAMS = universal 2D config (source-blend + CA mask + adaptive/asymmetric
# TAG + STG bump + outside/latent anchors) — the Part-1 "standard" config.
# VOXEL_CACHE_PARAMS = 3D cache ON, lifting raw_self (edit force). Active package:
#   - raw_self p95-normalized (gradient_mask_raw_norm_quantile=0.95) — kills
#     single-pixel-max jitter that faked cross-view variance.
#   - positive-only fusion (external_mask_interp_suppression_ratio=0.0) — cache adds
#     agreed force, never subtracts. Negative branch is redundant when the over-edit
#     region is variance-distinguished (clown A/B); p_neg only weakens suppression
#     (do not use it for over-edits).
#   - observed-weighted trilinear readback (mask_voxel_cache_trilinear=True) —
#     converts mask-level consistency into rendered multi-view consistency (the elf
#     MV_cos gain), no resolution/density penalty.
#   - decayed/EW cross-view variance (mask_voxel_cache_variance_decay=0.2) — KEPT;
#     recency-weighting is principled (non-stationary edit) and improved elf. The EW
#     bias is a uniform rescale, so it preserves the variance separation.
#   - agreement gate max_variance. TUNE PER SCENE against the variance map
#     (dc_debug/voxel_cache_variance_map): set it in the gap between the wanted-edit
#     variance (low) and the over-edit-region variance (high), above the former.
#   - variance PEAK-HOLD (mask_voxel_cache_variance_peak_decay) — implemented but
#     OFF (0.0): the warmup transient dominates the run-wide peak and zeroes
#     confidence body-wide (clown A/B 2026-06-10). Kept in code as documented
#     mechanism history (VoxelCacheExplained §6c).
#   mass 0.18/pow1.0 ON, angular power 1.0 relative, warmup 100->1100 universal
#   (current values are the clown override: 500->1200 gentle/late).
# Per-scene status: elf FIXED (cache now neutral-to-better + MV_cos up; trilinear is
# the lever). clown ARM RATE RESTORED TO BASE: the bundle max_blend 0.2 (0.4 doubled
# the push) + max_variance 0.015 + warmup 500->1200 brings arm failures back to the
# ~1/3 cache-off base rate (n=3); the variance map confirms the gate separates arms
# (bright/gated) from head (dark/trusted) — the residual 1/3 is the 2D process, not
# the cache. Next: confirm cache aliveness (confidence coverage) + MV_cos/EMV vs
# cache-off, +3 seeds. einstein/stormtrooper/bear pending.
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
    external_mask_interp_suppression_ratio=0.0,
    external_mask_negative_variance_power=0.0,
    external_mask_screen_self_boost_lambda=1.0,
    # LAST-TRY clown experiment (2026-06-10): contested-region suppression.
    # fused -= warmup_blend * ratio * contested * M, where contested =
    # valid * (n>=n_min) * min(1, var/max_variance) from the voxel cache.
    # Unlike the up/down branches (confidence-gated -> abstain on distrusted
    # voxels), this ACTIVELY damps the 2D mask where views disagree (the
    # arms), which also strengthens the (1-M)-scaled anchors there (skin
    # recovery). Max damping at ratio=1.0 is warmup_blend=0.2, i.e. <=20%.
    # Kill criteria: (a) dc_debug/voxel_cache_contested_map must light up the
    # arms (if not, mechanism falsified -> stop, set 0.0); (b) if map is
    # right but arm rate doesn't drop at n=3, one follow-up at 2.0, then stop.
    external_mask_contested_suppression_ratio=1.0,

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
    mask_voxel_cache_max_variance=0.015,           # clown: inside the head↔arm gap read off the variance map (was 0.020; head ≈ dark, arms ≈ bright)
    mask_voxel_cache_variance_decay=0.2,
    # Peak-hold of the cross-view variance: the gate sees the WORST
    # disagreement each voxel ever showed. TESTED ON CLOWN 2026-06-10 (n=3,
    # vs n=3 latch-off) and RETIRED at 0.0: the peak is dominated by the
    # immature-edit warmup transient (mean peak ~0.03 >> gate 0.015), which
    # zeroes confidence body-wide — the cache silently no-ops and runs
    # revert to the 2D base rate. With the late warmup (500->1200) the
    # stats accumulate uncontaminated and the plain gate already separates
    # arms (bright) from head (dark), so the latch solves an already-solved
    # problem. Do not re-enable without a post-maturity start for the peak.
    mask_voxel_cache_variance_peak_decay=0.0,

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
