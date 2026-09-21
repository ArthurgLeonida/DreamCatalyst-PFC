# Localization controls and experiment plan

These opt-in controls address hypotheses raised by the clown arm overedit and
the stormtrooper's need for stronger STG. They have not been validated by training
runs. Every new default preserves the previous guidance and localization path.
No project code or tests were executed locally, per `AGENTS.md`.

Configure them in `nerfstudio/dc/method_config.py`: all controls below belong to
`DC_CUSTOM_PARAMS`; cache enablement still belongs to `VOXEL_CACHE_PARAMS`.
Use fresh edits from the same reconstruction for comparisons. Runtime masks and
their EMA state are not checkpointed, as with the existing mask implementation.

## Controls and why they exist

| Control | Default | Suggested first ablation |
|---|---|---|
| `gradient_mask_ema_mode` | `"legacy_camera"` | `"per_view"` |
| `gradient_mask_ema_memory_visits` | `2.0` | Keep `2.0` |
| `gradient_mask_source` | `"dds"` | `"target_instruction"`, then independently `"source_instruction"` |
| `localization_source_timestep_ratio` | `0.5` | Keep `0.5` |
| `localization_source_num_samples` | `4` | Keep `4` |
| `localization_source_seed` | `42` | Hold fixed across comparisons |
| `cross_attention_mask_weight_min` | `0.0` | `0.25` |
| `external_mask_ca_gate_weight` | `0.0` | `1.0` |
| `outside_mask_anchor_mask_source` | `"final"` | `"internal"`; optionally `"source"` later |

### 1. Measure self-mask EMA memory in updates of one camera

The original automatic beta is `1 - 1/(camera_factor * N_cam)`, but the EMA is
stored **per camera** and updated only when that camera is sampled. With 300
cameras and factor 2, beta is about 0.99833. After nine further updates of that
camera, the first mask still contributes about 98.5% of its EMA. That can retain
an early error for essentially the whole edit.

With `gradient_mask_ema_mode="per_view"`, the pipeline instead resolves:

```text
beta = exp(-1 / gradient_mask_ema_memory_visits)
```

Memory 2 gives beta about 0.60653; after two subsequent camera updates an old
contribution has decayed to about 36.8%. This fixes the mismatch between the
specified memory and the clock on which this EMA is actually updated. It may
also expose more noisy masks, so better image quality remains an experimental
question. The unit is a call for that camera, including consecutive calls if
`change_view_step` exceeds 1.

`gradient_mask_ema_beta_auto=True` is required for either automatic mode.
With it disabled, `gradient_mask_ema_beta` remains the manual override and the
mode/memory knobs are inert. The voxel cache's EMA is unchanged: it has a
different update process. Effective self-mask beta is printed and logged.

### 2. Separate instruction evidence from the edited image's drift

`gradient_mask_source` selects the signal **before** the existing normalization,
EMA, gamma, and blur:

- `"dds"`: existing `||eps_CFG_tgt(z_edit) - eps_image_CFG_src(z_src)||`.
  Both conditioning and the noisy latent differ, so this includes accumulated
  changes in the render as well as the requested instruction.
- `"target_instruction"`: `||eps(z_edit, I_src, instruction) -
  eps(z_edit, I_src, empty_text)||`. Both predictions use the same noisy edited
  latent and the same source image conditioning. It reuses the two target
  sub-batches already computed by IP2P: **no extra UNet calls**. This removes
  the explicit edited-versus-source image difference, but remains conditioned
  on the evolving render and the DDS noise schedule.
- `"source_instruction"`: the same instruction-versus-empty-text contrast on
  a **noisy source latent**, at a fixed localization noise level. It averages
  channel-L2 magnitudes over `localization_source_num_samples` independent noise
  draws before normalization. Maps are computed lazily on each camera's first
  use, then kept on CPU and reused throughout the edit. This removes render
  drift and DDS timestep annealing from the self-mask's evidence.

The fixed-source noise index is
`round(localization_source_timestep_ratio * (num_train_timesteps - 1))`.
Thus 0.5 means roughly timestep 500 in a 1000-step diffusion training schedule,
not 50% of editing progress. The extra work is four two-condition UNet calls
per newly encountered view with the suggested settings, not four per iteration.
The source latent is the same per-view cached VAE sample used by DDS.

Localization noise uses a private generator seeded by
`localization_source_seed + camera_slot`, leaving the training noise stream
unchanged. Changing the training seed still changes VAE samples and optimization;
the fixed localization seed does not make whole runs identical. The source-map
cache is invalidated when prompts or its noise settings change, and maps are
rebuilt on shape mismatch.

The instruction contrast is motivated by the prompt-aware scoring in
[LatentEditor, section 3.2.2](https://arxiv.org/html/2312.09313v2#S3.SS2.SSS2).
Contrasting diffusion predictions for localization is also the principle behind
[DiffEdit](https://arxiv.org/abs/2210.11427). The choices here are adaptations to
this DDS pipeline, not a reproduction or an established improvement.

The cache input remains `self_grad_mask_raw`, but its underlying evidence now
follows the chosen source. It retains the same per-frame robust quantile
normalization, with no CA schedule, EMA, TAG, STG, or cache fusion folded into
the observation. First compare these choices **with the voxel cache off**.

None of these signals is a ground-truth semantic mask. Quantile normalization
can amplify weak evidence. A fixed-source mask can also omit space needed for
a new helmet, hair, or costume. Check edit completion as well as preservation.

### 3. Allow semantic restriction from the beginning

The hybrid mask still uses `M_internal = S * ((1-w) + w*A)`. The new schedule is:

```text
w = min_weight + (1-min_weight) * progress^(schedule_power * e)
```

`cross_attention_mask_weight_min=0` is exactly the previous schedule. A floor
of 0.25 introduces partial CA restriction immediately. A floor of 1 gives full
CA restriction throughout. The existing power knob still controls the ramp.
This can prevent some early background changes, but cannot correct a misleading
attention map. Full early CA may suppress structural edits; use 0.25 first and
treat 1 as a stronger diagnostic, not a recommended universal setting.

### 4. Require semantic support for positive cache additions

Previously the gate was `G=max(A,S)`, so a strong self-mask could authorize a
cache boost even when CA rejected that location. The optional gate is:

```text
G = max(A,S) * ((1-lambda) + lambda*A)
M_final = M_internal + blend * confidence * G * max(M_cache - M_internal, 0)
```

Here `lambda=external_mask_ca_gate_weight`. Zero reproduces the previous gate.
At 1, a location with zero CA support cannot receive a cache addition. Missing
CA maps are treated as zero support for this additional requirement; enabling
the knob with CA disabled raises a configuration error.

The fusion remains positive-only. This does not subtract from the current 2D
mask and does not restore the retired contested-region suppression mechanism.
It can still reinforce unwanted arms if attention supports the arms; agreement
and attention are evidence, not proof of edit correctness.

### 5. Prevent cache boosts from automatically weakening preservation

The outside-mask anchor still uses `psi + w_out*(1-s)*(1-P)` when adaptive
scaling is enabled. Only the mask `P` becomes selectable:

- `"final"`: previous behavior, using the mask after cache fusion.
- `"internal"`: the 2D hybrid mask before cache fusion. A cache boost increases
  the edit force without simultaneously relaxing the preservation weight.
- `"source"`: the fixed-source instruction mask described above, with the same
  normalization/gamma/blur but no per-view EMA, CA schedule, or cache fusion.
  This allows preservation support to stay fixed even if the edit mask evolves.
  The source relevance computation is shared when the self-mask also uses it.

`"internal"` isolates the cache/anchor coupling most cleanly. `"source"` is a
separate, stronger hypothesis and can resist desired structural changes. The
adaptive scalar `(1-s)` still evolves in both modes; only spatial support is
decoupled. With zero outside-anchor weight these options have no gradient effect.

## Diagnostics

New WandB fields under `dc_debug/`:

- `self_mask_ema_beta`: effective beta, including pipeline resolution.
- `internal_mask`, `preservation_mask`, their means/maxima/coverage: inspect
  where editing is allowed and where preservation is relaxed.
- `cache_addition`, `cache_addition_mean`: actual nonnegative addition to the
  internal mask, after confidence, validity, and the configured gate.
- `stg_effective_delta_norm`, `tag_only_effective_delta_norm`,
  `stg_to_tag_only_dds_ratio`: compare the STG residual with the TAG-only DDS
  delta, applying the same source-blend mask to both when enabled. STG-off steps
  report zero STG contribution. A large ratio with a tiny denominator should
  be read alongside both norms; it does not prove that STG caused overediting.

These are training diagnostics, not independent evaluation metrics. STG's
schedule and formula are unchanged; nominal scale alone cannot reveal its
effective magnitude or whether its spatial support is appropriate.

## Experiments, in order

Keep the original prompts initially: source `"a photo of a person"`, targets
`"Turn him into a Clown"` and `"Turn him into a Storm Trooper"`. Preserve the
exact capitalization/wording of previous runs if it differs. Hold reconstruction,
downscale, 3000-step budget, guidance scale, and all unlisted knobs constant.
Do not use the historical Part-1 STG 3.5 configuration as the cache-off control
for a cache-on run at 3.0.

Use seed 42 for initial screening on **both** edits. The following full screening
sequence is 26 runs if you try every listed variant and reuse the controls.
Stop an experimental branch if it clearly prevents the intended edit, recording
that failure. Re-run the surviving candidates **and their controls** with seeds
43 and 44 before drawing conclusions; a single successful clown is insufficient.

### A. Establish whether the cache worsens overediting: 8 screening runs

With every new knob at its default, run the full 2x2 comparison on both edits:

| Variant | `stg_scale` | `mask_voxel_cache_enabled` |
|---|---:|---|
| A0 | 3.0 | False |
| A1 | 3.0 | True |
| A2 | 2.5 | False |
| A3 | 2.5 | True |

This answers the still-open matched-scale cache question and preserves the
known clown/stormtrooper tradeoff as a control. Existing runs can replace cells
only if prompts, seed, reconstruction, budget, and all settings match.

### B. Test localization at STG 3.0, cache off: 8 additional screening runs

Start each row from A0; the table lists **only** its deltas:

| Variant | Deltas from A0 | Question |
|---|---|---|
| B1 | `gradient_mask_ema_mode="per_view"` | Does stale mask memory explain persistence? |
| B2 | `gradient_mask_source="target_instruction"` | Does removing the cross-latent difference help? |
| B3 | Both B1 and B2 changes | Does fresher instruction evidence work best together? |
| B4 | `gradient_mask_source="source_instruction"` | Does fixed-source, fixed-noise localization help? |

Keep memory 2, source noise ratio 0.5, and four samples. In B4 the input mask
is fixed per view, so changing EMA memory is not a useful separate ablation.
B4 changes both the reference latent and the localization timestep strategy;
attribute any gain to that combined strategy, not to one of those factors alone.

### C. Test early semantic restriction: 2 additional screening runs

Take the best common cache-off setting from A0/B1-B4 and change only
`cross_attention_mask_weight_min=0.25`. Compare both edits with its floor-zero
control. If useful, keep it as one shared choice. Trying floor 1 is optional
and adds two runs; check whether it blocks the helmet.

### D. Reintroduce the cache: 8 additional screening runs

Take the selected common 2D setting at STG 3.0. Its cache-off result is the
control. Turn the cache on with the unchanged global `max_variance=0.02`, blend
0.2, and warmup 500-1200, then test:

| Variant | `external_mask_ca_gate_weight` | `outside_mask_anchor_mask_source` |
|---|---:|---|
| D0 | 0.0 | `"final"` |
| D1 | 1.0 | `"final"` |
| D2 | 0.0 | `"internal"` |
| D3 | 1.0 | `"internal"` |

D1 isolates the gate; D2 isolates preservation coupling; D3 tests their
interaction. If evolving preservation support remains a problem, optionally
compare `"source"` against `"internal"` with the gate and every other parameter
held fixed. Do not change the variance threshold to rescue individual scenes.

### E. Confirm one shared configuration

Repeat finalists and matched controls across seeds 42/43/44, then evaluate the
same selected settings on all five scenes. Keep the best configuration common
to clown and stormtrooper before extending it; do not choose one setting for
each scene. After localization is selected, a matched STG 2.5 versus 3.0 check
can establish whether the original strength tradeoff has improved.

Compare background preservation, edit completion, and multi-view consistency
separately. A nearly unchanged scene can look excellent on preservation and
mask variance while failing the instruction.

Record clown unwanted-arm paint and stormtrooper helmet completion using fixed
views and explicit criteria set before comparing methods. Also report
`Background_LPIPS`, `Background_PSNR`, `CLIP_direction_local`,
`EditMaskVariance_3D`, and the magnitude/context metrics. A whole-person
evaluation region includes the arms: it cannot by itself detect this failure.
Use separate, source-defined arm/background diagnostic regions, identical across
methods; never use the training mask to define the evaluation region. The broad
clown instruction does not explicitly prohibit arm changes, so report this as
the desired preservation criterion rather than an objectively implied mask.

Inspect early, middle, and final renders from the same cameras where available.
Training mask logs use randomly sampled cameras: compare `current_spot` before
attributing a visual difference to time. Set checkpoint retention/logging for
any intermediate evaluation before starting the server run.

## Running the comparisons

Edit the central config for each variant; method flags are intentionally not
duplicated as shell environment overrides. `scripts/edit.sh` now accepts `SEED`
(default 42) and prints the new localization settings. Nerfstudio records the
method config in each run's `config.yml`.

Example on the lab server, after applying B3's deltas and disabling the cache:

```bash
# Replace these with the same dataset, reconstruction, and downscale used
# for the control. "clown" in RUN_NAME identifies the edit, not the dataset.
SCENE=your_scene
LOAD_DIR=outputs/your_scene/nerfacto/your_timestamp/nerfstudio_models
DOWNSCALE=1

for seed in 42 43 44; do
    SEED="$seed" RUN_NAME="${SCENE}_clown_B3_stg3_s${seed}" \
        bash scripts/edit.sh "$SCENE" "a photo of a person" \
        "Turn him into a Clown" "$LOAD_DIR" 3000 nerf "$DOWNSCALE"
done
```

Repeat with the stormtrooper target and a distinct run name. The wrapper's
evaluation remains enabled by default (`EVAL_AFTER_EDIT=1`). For screening, run
only seed 42 first. Preserve each run's config and compare matching seeds.

Before a full sweep, do a short server smoke run for the default path, B3, B4,
and D3; check config parsing, finite losses/masks, expected effective beta,
source-mask reuse, and positive-only additions. A short run does not validate
quality, and must not be included as a 3000-step result. To exercise cache
fusion, the smoke run must reach the configured cache warmup or use explicitly
temporary earlier warmup settings. Restore all experimental settings afterward.
