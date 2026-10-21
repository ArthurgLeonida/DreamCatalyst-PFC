"""Method configuration: the single source of truth for every editing run.

DC_CUSTOM_PARAMS   -> DCConfig (nerfstudio/dc/dc.py): guidance, localization, preservation.
VOXEL_CACHE_PARAMS -> DCPipelineConfig (dc_pipeline.py): the 3D voxel cache (Part 2).

As committed, this is the final 3D configuration. The other two evaluated
configurations change only the values below:

  2D (Part 1, no voxel cache):
      mask_voxel_cache_enabled=False

  DreamCatalyst baseline (upstream: DDS + IP2P + FreeU + psi):
      source_blend_localization_enabled=False, outside_mask_anchor_weight=0.0,
      cross_attention_mask_enabled=False, latent_mean_anchor_weight=0.0,
      eta_tag=1.0, stg_scale=0.0, mask_voxel_cache_enabled=False
      (with every localization switch off no mask is built, so the remaining
      mask settings are inert)

One optional variant is implemented but was not evaluated (off in all three):

  Edit-release anchor:
      outside_mask_anchor_edit_release=True (requires cross_attention_mask_enabled)
      The outside-mask anchor is weighted by 1 - max(S_src, M_internal)
      instead of 1 - S_src, so it is released at least as much as the
      pre-cache edit mask drives the edit, and shapes the edit adds (a
      helmet) stop fighting it. The price: drift that A confirms also
      releases it, so the brake is no longer fixed per view. The cache still
      cannot release it.

Nothing is tuned per scene. In particular mask_voxel_cache_max_variance is one
global value for every scene and edit. That is a stated limitation: it is the
most scene-sensitive setting, read off dc_debug/voxel_cache_variance_map.

Symbols: S = target-instruction mask, S_src = source-instruction mask,
A = cross-attention mask, M = edit mask, s = edit strength in [0, 1].
"""

DC_CUSTOM_PARAMS = dict(
    # ---------------------------------------------------------------------
    # Guidance amplification (target branch only)
    # ---------------------------------------------------------------------
    eta_tag=1.2,                      # adaptive TAG ceiling; 1.0 = off
    stg_scale=3.0,                    # STG peak scale, times (1 - s); 0.0 = off
    stg_skip_layers=[2],              # up-blocks whose self-attention STG skips
    stg_schedule_start_ratio=0.0,     # STG bump over training progress: zero
    stg_schedule_end_ratio=0.8,       # outside [start, end], peak at
    stg_bump_peak_ratio=0.5,          # start + ratio * (end - start)

    # ---------------------------------------------------------------------
    # Localization: where the edit may act, M = max(S * A, min(S, S_src))
    # ---------------------------------------------------------------------
    source_blend_localization_enabled=True,  # restrict the DDS delta to M
    gradient_mask_gamma=1.2,                 # S and S_src: sharpening exponent
    gradient_mask_blur=0.5,                  # S and S_src: Gaussian sigma (latent px)
    gradient_mask_raw_norm_quantile=0.95,    # voxel-cache input: S over this per-frame quantile
    gradient_mask_ema_memory_visits=5.0,     # S: per-view EMA memory, in visits of that view
    localization_source_timestep_ratio=0.5,  # S_src: noise level (fraction of training steps)
    localization_source_num_samples=4,       # S_src: noise draws averaged per view
    localization_source_seed=42,             # S_src: private RNG seed (training RNG untouched)

    cross_attention_mask_enabled=True,       # A: attention of the edit tokens, full strength
    cross_attention_mask_layers=[1, 2],      # A: up-blocks read (16x16 and 32x32)
    cross_attention_mask_gamma=1.2,
    cross_attention_mask_blur=0.5,

    # ---------------------------------------------------------------------
    # Preservation: what stays like the source
    # ---------------------------------------------------------------------
    psi=0.075,                               # DreamCatalyst preservation weight (upstream)
    outside_mask_anchor_weight=0.25,         # extra weight outside S_src, times (1 - s)
    outside_mask_anchor_edit_release=False,  # optional, not evaluated: anchor mask max(S_src, M_internal)
    latent_mean_anchor_weight=0.005,         # per-channel latent-mean anchor
)


VOXEL_CACHE_PARAMS = dict(
    mask_voxel_cache_enabled=True,

    # Grid and backprojection
    mask_voxel_cache_resolution=64,              # voxels per axis
    mask_voxel_cache_accumulation_threshold=0.50,  # a ray must hit a surface to count
    mask_voxel_cache_bbox_observe_steps=50,      # steps observing points before building the grid
    mask_voxel_cache_bbox_observe_quantile=0.05, # per-view outlier clip of the observed box
    mask_voxel_cache_bbox_inflation=0.2,         # box margin for geometry the edit grows

    # Statistics
    mask_voxel_cache_ema_beta_camera_factor=0.5, # grid-mean decay 1 - 1 / (factor * N_cam)
    mask_voxel_cache_variance_decay=0.2,         # weight of the newest view in the cross-view variance

    # Confidence gates (each in [0, 1], multiplied)
    mask_voxel_cache_observation_fraction=0.10,  # count gate: ceil(fraction * N_cam) views,
    mask_voxel_cache_min_observations_floor=5,   #   clamped to [floor, cap]
    mask_voxel_cache_min_observations_cap=12,
    mask_voxel_cache_max_variance=0.02,          # variance gate: 1 - variance / max_variance (global)
    mask_voxel_cache_angular_freeze_warmup=50,   # angular gate: track the scene mean after this many
    mask_voxel_cache_angular_freeze_patience=100,  #   steps, freeze it 100 steps after its peak
    mask_voxel_cache_mass_threshold=0.18,        # mass gate: min(1, cached value / threshold)

    # Fusion: M += blend * confidence * G * max(cache - M, 0), positive only
    mask_voxel_cache_scale_normalize_quantile=0.95,  # stretch the cache read to [0, 1] first
    mask_voxel_cache_warmup_start=500,           # blend ramps 0 -> max_blend over
    mask_voxel_cache_warmup_end=1200,            #   [start, end] edit steps
    mask_voxel_cache_max_blend=0.2,
)
