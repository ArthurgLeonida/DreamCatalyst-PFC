# Localization controls and experiment plan

These opt-in controls address hypotheses raised by the clown arm overedit and
the stormtrooper's need for stronger STG. A/B screening was reported on
2026-09-21 and the C results on 2026-09-22. The evidence and revised next steps
are recorded below. These are screening results, not yet evidence for a
universal winning configuration.
The implementation defaults still preserve the previous guidance and localization
path. This update changes the experiment guide only; no project code or tests
were executed locally, per `AGENTS.md`.

Configure them in `nerfstudio/dc/method_config.py`: all controls below belong to
`DC_CUSTOM_PARAMS`; cache enablement still belongs to `VOXEL_CACHE_PARAMS`.
Use fresh edits from the same reconstruction for comparisons. Runtime masks and
their EMA state are not checkpointed, as with the existing mask implementation.

## Current decision after C (2026-09-22)

**Use B4 (source instruction mask, CA floor 0) at STG 3.0 as the shared working
reference for D0.** The researcher now confirms that B4 was slightly better
than C4 on the stormtrooper hand and that all experiments used the same fixed
training seeds. Together with the clown preservation regressions under C4,
this supports advancing B4 without another B4/C4 screening sweep. C4 remains a
recorded ablation, not the new default. B4 is not yet a validated universal winner.

C4 improves stormtrooper background preservation, but the clown repeats show
more out-of-region change, lower background PSNR, and lower multi-view cosine
than both exported B4 runs. C3 is also mixed: it improves local CLIP and reported
mask variance while worsening stormtrooper background LPIPS. Do not expand the
CA-floor or STG sweep, or add a mask mixture, on this evidence.

Next: run the small D0 cache off/on comparison on B4 for both edits, retaining
the existing fixed training seed setup. Robustness across distinct training
seeds is reserved for confirmation of the selected setting. The full D1-D3 sweep remains
conditional on an observed cache benefit with a specific remaining failure.

### New export and run accounting

Source: `wandb_export_2026-09-22T01_43_57.406-03_00.csv`, 16 rows: nine
stormtrooper rows and seven clown rows. The seven stormtrooper rows from the
previous export are retained, with one C3 and one C4 added. Clown has one B3,
one C3, **two distinct B4 runs**, and three C4 runs. Do not deduplicate by Name:

| Exported name | Run timestamp from region-mask path | Role |
|---|---|---|
| `clown_standard_B4` | `2026-09-21_215632` | B4 earlier run |
| `clown_standard_B4` | `2026-09-21_231738` | B4 later run |
| `clown_standard_C4` | `2026-09-21_231848` | C4 first run |
| `clown_standard_C4_1` | `2026-09-22_031926` | C4 repeat |
| `clown_standard_C4_2` | `2026-09-22_031945` | C4 repeat |

The two B4 records have identical exported config/region settings but different
metrics and checkpoint paths. The researcher confirms that the repeats used the
same fixed training seeds: `_1` and `_2` are repeat labels, not different-seed
evidence. Keep these runs as measurements of variation under the fixed seed
setup. Their differences do not establish seed robustness or identify the cause
of the variation; nondeterministic execution or unrecorded runtime differences
would need separate investigation before attributing a cause.
The numeric training seed remains unexported, as do STG scale and cache enablement.
The C analysis uses the intended STG-3/cache-off protocol provisionally, rather
than claiming the CSV independently verifies it.

The exported C3/C4 controls match their definitions: target/per-view versus
source/legacy-camera masks, CA floor 0.25, gate 0, anchor `final`, memory 2,
source noise ratio 0.5, four source samples, and localization seed 42. All new
runs report `cv_trilinear=True`; this is inert when the cache is disabled.
Region settings and reported mean coverage match within each edit: 0.21535645
for stormtrooper and 0.21257580 for clown. View counts, shared EMV3D grid
provenance, and exact mask identity are still absent from the export.

### Stormtrooper: B-to-C comparison

One exported run per cell. Names `st_standard_standard_C3` and
`st_standard_standard_C4` are shown as C3/C4 here; the repeated `standard` is
part of the exported name. Values are rounded, not significance estimates.

| Metric | B3 | C3 | B4 | C4 |
|---|---:|---:|---:|---:|
| Global CLIP direction, higher | 0.19826 | 0.19665 | 0.18530 | 0.18755 |
| Local CLIP direction, higher | 0.17767 | 0.17900 | 0.18249 | 0.18077 |
| CLIP image similarity, higher | 0.71049 | 0.70082 | 0.69272 | 0.69552 |
| SSIM, higher | 0.83407 | 0.83428 | 0.83725 | 0.84133 |
| LPIPS, lower | 0.29760 | 0.30166 | 0.30275 | 0.29886 |
| Background LPIPS, lower | 0.16180 | 0.16555 | 0.16050 | 0.15824 |
| Background PSNR, higher | 27.8732 | 27.8277 | 28.0186 | 28.4171 |
| Localization ratio, higher | 0.84281 | 0.84872 | 0.84144 | 0.84654 |
| Multi-view cosine, higher | 0.90576 | 0.90527 | 0.90674 | 0.90479 |
| EMV3D raw, lower | 0.006231 | 0.004343 | 0.005749 | 0.005608 |
| EMV3D normalized, lower | 0.52990 | 0.49512 | 0.51964 | 0.53242 |
| Edit magnitude in region (context) | 0.37739 | 0.37873 | 0.38405 | 0.36691 |
| Edit magnitude out of region (context) | 0.07182 | 0.06891 | 0.07391 | 0.06810 |

- **C4 versus B4:** background LPIPS decreases 1.41%, PSNR increases 0.399 dB,
  and out-of-region change decreases 7.85%. SSIM and whole-image LPIPS improve.
  Local CLIP decreases 0.95% and in-region magnitude decreases 4.46%; inspect
  whether this is beneficial restraint or incomplete editing. Multi-view cosine
  decreases and normalized EMV3D increases 2.46%, despite a small decrease in
  raw variance. C4 is a preservation candidate, not a consistency winner.
- **C3 versus B3:** local CLIP increases 0.75%, but background LPIPS worsens
  2.31% and whole-image LPIPS worsens 1.36%. Raw/normalized EMV3D decrease
  30.30%/6.56%. This does not establish that the early-CA background problem was
  solved. Keep C3 as an ablation result, without prioritizing another C3 sweep.

### Clown: all exported repeats retained

B3 and C3 each have one row. B4 and C4 below are **descriptive means of two and
three fixed-seed repeats**, respectively, not independent-seed confidence
estimates. The common seed controls the seed choice, but does not create a
one-to-one pairing between the two B4 and three C4 runs. Do not pair repeats
arbitrarily by their order in the export.

| Metric | B3 (1 row) | C3 (1 row) | B4 (2 rows) | C4 (3 rows) |
|---|---:|---:|---:|---:|
| Global CLIP direction, higher | 0.13845 | 0.14179 | 0.13500 | 0.13486 |
| Local CLIP direction, higher | 0.13474 | 0.13751 | 0.13076 | 0.13005 |
| CLIP image similarity, higher | 0.76592 | 0.75715 | 0.75698 | 0.75711 |
| SSIM, higher | 0.93177 | 0.93008 | 0.92857 | 0.92875 |
| LPIPS, lower | 0.20135 | 0.20133 | 0.20508 | 0.20259 |
| Background LPIPS, lower | 0.14291 | 0.14199 | 0.14211 | 0.14192 |
| Background PSNR, higher | 31.3531 | 31.1116 | 31.2716 | 30.8379 |
| Localization ratio, higher | 0.79543 | 0.79939 | 0.82363 | 0.80315 |
| Multi-view cosine, higher | 0.91748 | 0.91455 | 0.91919 | 0.91341 |
| EMV3D raw, lower | 0.008207 | 0.006696 | 0.005143 | 0.007013 |
| EMV3D normalized, lower | 0.34277 | 0.33389 | 0.34916 | 0.30021 |
| Edit magnitude in region (context) | 0.20834 | 0.22332 | 0.24932 | 0.24028 |
| Edit magnitude out of region (context) | 0.05501 | 0.05736 | 0.05436 | 0.05968 |

The most repeat-consistent C4 findings are regressions relative to **both** B4
rows, regardless of which B4 checkpoint is used:

| Quantity | B4 range (2 rows) | C4 range (3 rows) |
|---|---:|---:|
| Background PSNR, higher | 31.2606-31.2825 | 30.6548-31.0136 |
| Out-of-region edit magnitude | 0.05344-0.05529 | 0.05735-0.06181 |
| Localization ratio, higher | 0.81721-0.83004 | 0.79565-0.80901 |
| Multi-view cosine, higher | 0.91846-0.91992 | 0.91162-0.91602 |
| EMV3D raw, lower | 0.005017-0.005268 | 0.006912-0.007096 |

C4's mean background LPIPS is only 0.13% lower, while mean out-of-region change
is 9.78% higher and background PSNR is 0.434 dB lower. Whole-image LPIPS improves
1.21%, and normalized EMV3D is 14.02% lower even though raw EMV3D is 36.37%
higher. These metrics measure different aspects: normalized EMV3D divides by
squared eligible-voxel edit mass. It can decrease while absolute disagreement
increases. Report both, together with grid/mass provenance; do not call the
normalized improvement alone better geometry or uniformly better consistency.

C3 improves local CLIP over B3 by 2.05% and reduces background LPIPS slightly,
but background PSNR falls 0.242 dB, out-of-region change rises 4.27%, and
multi-view cosine falls. Neither mask family shows a clean cross-edit win from
the CA floor. A smaller mask at a given iteration is not sufficient evidence
of better final background preservation after the whole optimization changes.

Follow-up visual verdict: **B4 was slightly better than C4 on the hand**. This
supports retaining floor 0 for the next stage, alongside the clown metric
evidence. No separate verdict about the clown arm overedit was provided; a
whole-person evaluation region cannot settle that question. Continue scoring
arm paint, hand/helmet completion, and view consistency in D0 rather than
assuming the visual failure is solved by the aggregate metrics.

## Earlier B screening record (2026-09-21)

At this stage B3 and B4 were advanced to C. B4 was the leading visual candidate
for the stormtrooper hand, B3 the global-CLIP reference, and B2 the preservation
reference. The following section retains that earlier evidence; the current
decision is the post-C assessment above.

### Evidence available

- **A:** the researcher reports that STG 3.0 was best and the cache was neutral.
  The A-run export was not provided with this update, so this is a reported
  result, not a numeric comparison reconstructed from the attached CSV.
- **B:** `wandb_export_2026-09-21T18_43_30.795-03_00.csv` contains seven
  stormtrooper rows: `st_standard_B1` through `st_standard_B4`, plus
  `st_standard`, `st_standard_1`, and `st_standard_2`. There is one exported row
  per B variant, no per-seed aggregation, and no clown rows in this export.
- **Images:** the researcher confirms the identities `st_B3.png` and
  `st_B4.png` and identifies B4 as the result with the smaller stormtrooper hand.
  B3 shows a larger reddish hand/remnant; B4 shows a smaller, darker rendered
  hand. This supports a difference in rendered appearance; a single RGB view
  cannot establish a 3D geometry change.
- **Run conditions:** the researcher confirms that the B runs used the cache
  disabled and the same training seed. The numeric training seed was not given;
  this is one matched-seed comparison, not independent seed replication.

The following values are copied from that export and rounded. The standard
column is the arithmetic mean of the three named standard rows, **not** a
confirmed matched-seed baseline. Bold marks the best B variant per metric;
it is descriptive and does not establish statistical significance.

| Metric | Standard mean (3 rows) | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|---:|
| CLIP direction, global, higher | 0.17062 | 0.18667 | 0.18642 | **0.19826** | 0.18530 |
| CLIP image similarity, higher | 0.68224 | 0.70830 | 0.70993 | **0.71049** | 0.69272 |
| CLIP direction, local, higher | 0.16945 | 0.17094 | 0.17517 | 0.17767 | **0.18249** |
| SSIM, higher | 0.83877 | 0.84031 | **0.84168** | 0.83407 | 0.83725 |
| LPIPS, lower | 0.29503 | **0.29364** | 0.29629 | 0.29760 | 0.30275 |
| Background LPIPS, lower | 0.15429 | 0.16036 | **0.15746** | 0.16180 | 0.16050 |
| Background PSNR, higher | 28.5560 | 28.1088 | **28.6064** | 27.8732 | 28.0186 |
| Edit localization ratio, higher | 0.84309 | 0.84751 | **0.85095** | 0.84281 | 0.84144 |
| Multi-view CLIP cosine, higher | 0.90120 | **0.91260** | 0.90332 | 0.90576 | 0.90674 |
| EditMaskVariance_3D, lower | 0.005425 | 0.005070 | **0.004490** | 0.006231 | 0.005749 |
| EditMaskVariance_3D_normalized, lower | 0.54927 | 0.50897 | **0.46215** | 0.52990 | 0.51964 |

Interpretation:

- All four B variants exceed all three standard rows on **global CLIP
  direction**, but not on every metric. Every B variant has worse background
  LPIPS than every exported standard row (standard range 0.15373-0.15459).
  B3/B4 also have worse raw edit-mask variance and whole-image LPIPS than those
  standard rows. This is an editability/preservation tradeoff, not uniform gain.
- B3 leads global CLIP direction and CLIP image similarity. B4 leads **local**
  CLIP direction among B variants, and beats B3 on SSIM, background LPIPS/PSNR,
  multi-view cosine, and both edit-mask variance values. Thus the visual
  preference for B4 is supported by part of the metric suite as well.
- B2 has the best B-family background preservation, localization ratio, SSIM,
  and mask variance; B1 has the best whole-image LPIPS and multi-view cosine.
  Preserve those results as useful controls rather than calling them failures.
- Whole-image LPIPS and CLIP image similarity reward resemblance to the source.
  Desired hand replacement/shrinkage can cost preservation score inside the
  edited region. That does not excuse background damage, and the present data
  do not show how much of B4's LPIPS difference comes from the hand.
- Global and local CLIP directions use different image composites and are not
  additive. B3's global advantage cannot be attributed numerically to background
  leakage by subtracting its local score. `CLIP_direction_bg` near zero also
  does not prove a clean background; use background LPIPS/PSNR and renders.

### Why the hand result is informative

One plausible explanation is that B4 keeps edit support at the original hand
location while that hand changes. B3 recomputes instruction relevance on the
evolving render: if relevance fades around an incompletely replaced hand, the
same mask can both reduce the DDS edit force and increase the outside-mask
anchor toward the source. B4's source-only evidence could avoid that particular
feedback. This is a **hypothesis**, not a conclusion from the screenshots.

B4 also changes localization timestep and averages four noise realizations, so
the current comparison does not isolate source anchoring as the cause. Its
self-mask evidence is fixed, but its CA mask, CA schedule, and adaptive guidance
still evolve. Inspect masks at the same camera/hand region over time before
claiming which mechanism enabled the change. Do not add a B3/B4 mask mixture
or further mechanisms before testing the existing candidates.

### Provenance to retain or confirm

The CSV confirms B1/B3 use `per_view` EMA with beta 0.60653, B2/B4 use
`legacy_camera` with beta 0.9984127, B3 uses `target_instruction`, and B4 uses
`source_instruction` with noise ratio 0.5, four samples, and localization seed 42.
All B rows have CA floor 0, additional cache-CA gate 0, and anchor mask `final`.

The export omits `machine.seed`, `pipeline.dc.stg_scale`, and
`pipeline.mask_voxel_cache_enabled`. Cache off and a common training seed are
now confirmed by the researcher. Record the actual numeric seed from the saved
`config.yml`; localization seed 42 is **not** evidence of training seed 42.
STG 3.0 remains the working setting from the B protocol and A decision; include
its saved value in the next export for independent verification.
All B rows export `mask_voxel_cache_cv_trilinear=True`, unlike the repository
default False. It is inert with cache off; with cache on, hold the verified
choice constant within comparisons and record it explicitly.

The B rows have matching reported region settings (`union`,
`person | storm trooper`) and mean coverage 0.21535645. The older standard rows
omit region provenance. Before making a formal improvement claim, verify their
evaluation versions, identical source-defined region masks and valid-view sets,
and matching EMV3D bbox/resolution/min-observations. The CSV lacks the grid
provenance needed to certify cross-run variance comparability. Re-evaluate
existing checkpoints with shared masks/grid if needed; this requires no retraining.

The exported STG contribution and cache-addition summaries are zero for B1-B4.
They are not time averages: STG is scheduled off late, while zero cache addition
is expected with the confirmed cache-off setting. In future cache-on runs, one
zero summary value cannot establish inactivity throughout training. Inspect
histories if investigating their effect.

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

## Experiment definitions and updated next steps

Keep the original prompts initially: source `"a photo of a person"`, targets
`"Turn him into a Clown"` and `"Turn him into a Storm Trooper"`. Preserve the
exact capitalization/wording of previous runs if it differs. Hold reconstruction,
downscale, 3000-step budget, guidance scale, and all unlisted knobs constant.
Do not use the historical Part-1 STG 3.5 configuration as the cache-off control
for a cache-on run at 3.0.

Stages A, B, and C are retained below as definitions of completed screening,
not a request to restart them. The current export covers both edits through C;
the researcher has confirmed fixed training seeds across runs and B4's slight
hand advantage. The next work is the small D0 cache comparison on B4. Reuse the
existing B4 cache-off runs, and record the numeric training seed in future exports.

### A. Completed screening: STG strength and cache

Reference matrix for A, with every new knob at its default:

| Variant | `stg_scale` | `mask_voxel_cache_enabled` |
|---|---:|---|
| A0 | 3.0 | False |
| A1 | 3.0 | True |
| A2 | 2.5 | False |
| A3 | 2.5 | True |

Reported result: STG 3.0 was best; cache impact was neutral. Keep 3.0 fixed in
the next stages and leave the cache off while selecting localization. Neutrality
here applies to the tested configuration, not necessarily a new mask source.

### B. Completed screening: localization at STG 3.0, cache off

Reference definitions for B; the table lists **only** the deltas from A0:

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

### C. Completed screening: early semantic restriction on both finalists

Reference matrix for the C comparison, at STG 3.0 with cache off. Each candidate
retains its B EMA setting; the only intended change is the CA floor.

| Variant | `gradient_mask_source` | `gradient_mask_ema_mode` | `cross_attention_mask_weight_min` |
|---|---|---|---:|
| B3 control | `"target_instruction"` | `"per_view"` | 0.0 |
| C3 | `"target_instruction"` | `"per_view"` | 0.25 |
| B4 control | `"source_instruction"` | `"legacy_camera"` | 0.0 |
| C4 | `"source_instruction"` | `"legacy_camera"` | 0.25 |

For every row keep `gradient_mask_ema_beta_auto=True`, memory 2, source noise
ratio 0.5, source sample count 4, `localization_source_seed=42`,
`external_mask_ca_gate_weight=0`, and `outside_mask_anchor_mask_source="final"`.
Keep all remaining guidance/localization settings identical to the matching B
control. Source settings are inert for B3/C3 under this anchor choice.

Observed result: C4 improves stormtrooper background preservation but has
clown regressions in PSNR, out-of-region magnitude, localization ratio, and
multi-view cosine. C3 has no uniform preservation gain either. Detailed values
and repeat ranges are in the post-C record above. The CA floor remains an
experimental candidate; do not promote 0.25 universally or start a floor-1 sweep
based on the stormtrooper result alone.

### C-confirm. Clarification received: advance B4 to D0

The researcher confirmed B4's slight hand advantage over C4 and that all runs
used the same fixed training seeds. Choose B4 as the shared working base for
both edits in D0. A new 12-cell B4/C4 confirmation grid is not required before
this small cache screen; distinct-seed validation remains necessary before a
final universal-config claim and is handled in E. Keep B3/C3/C4 as recorded
ablations rather than expanding their sweeps now.

For D0 and later validation, compare identical frontal and oblique/side views
against the source/reconstruction. Score the hand's remaining source-colored
remnant, attachment/proportion, silhouette, and stability across views. For a
claim of actual geometry reduction, also inspect depth/accumulation at matched
views; dark recoloring alone can make a hand appear smaller. Inspect helmet and
body completion as well so a clean hand does not hide a failed edit elsewhere.

Use a predeclared visual success criterion plus local CLIP, background
LPIPS/PSNR, and both raw/normalized EMV3D alongside multi-view cosine. Make paired
comparisons per seed and edit, then report their mean and spread; a separate
average of unmatched runs does not isolate the configuration effect. Do not
choose weights after seeing scores to manufacture one scalar winner. Retain B2
as the stormtrooper preservation reference in the result table.

### D. Next: B4 plus D0 cache off/on, then conditional gate ablations

Compare B4 with cache off versus on at the same STG 3.0, fixed training seed
setup, prompts, and localization settings on both edits. The cache
observations now use the selected mask evidence, so A's neutrality need not
carry over. Keep global `max_variance=0.02`, blend 0.2, warmup 500-1200, and a
fixed, recorded variance-read mode; do not mix `cv_trilinear=True` and False
within the comparison.

For the first D0 screen, add **one cache-on run per edit** at an existing
control's training seed: two new runs if both cache-off controls exist. Leave
`external_mask_ca_gate_weight=0` and `outside_mask_anchor_mask_source="final"`.
Use B4 in both scenes: `gradient_mask_source="source_instruction"`,
`gradient_mask_ema_mode="legacy_camera"`, and
`cross_attention_mask_weight_min=0`. Keep source noise ratio 0.5, four source
samples, and localization seed 42. The only active method change from the B4
control is `mask_voxel_cache_enabled=True`; label it in the run name.
The exported experiments use `cv_trilinear=True`; retain that for this immediate
comparison if it is the recorded server configuration, while explicitly noting
that historical reported Part-2 results used False. The repository currently
defaults to False, so set and record the intended value rather than assuming it.

Judge the initial cache effect against the observed fixed-seed B4 variation,
not just one conveniently chosen B4 checkpoint. If its magnitude is comparable
to repeat variation, call the screen inconclusive and repeat the cache-on run
under the same conditions before treating a small difference as an improvement.

If the cache is still neutral, retain the simpler cache-off setting for Part 1
and report the neutral Part-2 result. Check confidence and actual cache-addition
histories before spending runs on alternative gates. If the cache improves
consistency but causes unwanted edits or preservation loss, proceed with the
following original D ablations, reusing the cache-on control:

| Variant | `external_mask_ca_gate_weight` | `outside_mask_anchor_mask_source` |
|---|---:|---|
| D0 | 0.0 | `"final"` |
| D1 | 1.0 | `"final"` |
| D2 | 0.0 | `"internal"` |
| D3 | 1.0 | `"internal"` |

D1 isolates the gate; D2 isolates preservation coupling; D3 tests their
interaction. With cache off, `"internal"` and `"final"` are the same mask, and
the cache gate has no effect: those would be duplicate experiments. The
`"source"` preservation option is a separate hypothesis, deferred while the
D0 comparison remains open. Do not change the variance threshold to rescue
individual scenes.

Repeat any promising cache-on result and its cache-off control at two additional
distinct training seeds before extending the selected common setting to all
five scenes. A lower variance accompanied by failed hand/helmet/arm criteria
does not count as a successful cache result.

### E. Confirm one shared configuration

Repeat finalists and matched controls across at least three distinct training
seeds (42/43/44 if 42 was the original seed), then evaluate the same selected
settings on all five scenes. Keep the best configuration common
to clown and stormtrooper before extending it; do not choose one setting for
each scene. Preserve the STG 3.0 decision unless new evidence gives a concrete
reason to revisit it; a new broad STG sweep is not the next priority.

Compare background preservation, edit completion, and multi-view consistency
separately. A nearly unchanged scene can look excellent on preservation and
mask variance while failing the instruction.

Record clown unwanted-arm paint and stormtrooper helmet/hand completion using
fixed views and explicit criteria set before comparing methods. Also report
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

Example on the lab server, for the next D0 screen. Set B4 in `DC_CUSTOM_PARAMS`,
enable the cache in `VOXEL_CACHE_PARAMS`, and retain the D0 gate/anchor and other
cache parameters listed above. Use the existing cache-off control's fixed
training seed (replace the example 42 if the numeric seed differs).

```bash
# Replace these with the same dataset, reconstruction, and downscale used
# for the control. "stormtrooper" identifies the edit, not the dataset.
SCENE=your_scene
LOAD_DIR=outputs/your_scene/nerfacto/your_timestamp/nerfstudio_models
DOWNSCALE=1

for seed in 42; do
    SEED="$seed" RUN_NAME="${SCENE}_stormtrooper_B4_D0_cacheon_stg3_s${seed}" \
        bash scripts/edit.sh "$SCENE" "a photo of a person" \
        "Turn him into a Storm Trooper" "$LOAD_DIR" 3000 nerf "$DOWNSCALE"
done
```

Repeat for the clown target with a distinct run name, and compare to the
existing matched cache-off control. The wrapper's evaluation remains enabled
by default (`EVAL_AFTER_EDIT=1`). Extend the seed list for cache confirmation
after screening. Preserve each run's config and compare matching seeds. Export
run ID/path, `machine.seed`, STG scale, cache enablement/read mode,
evaluation region/view/grid provenance, and the
metric values together so the next comparison is reproducible.

For any newly used branch, a short server smoke run can check config parsing,
finite losses/masks, expected effective beta, source-mask reuse, and positive-only
additions. There is no need to rerun smoke checks for already completed B runs.
A short run does not validate quality, and must not be included as a 3000-step
result. To exercise cache fusion, the smoke run must reach the configured cache
warmup or use explicitly temporary earlier warmup settings. Restore all
experimental settings afterward.
