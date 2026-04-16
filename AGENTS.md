# AGENTS.md

## Project Snapshot

This repo is an undergraduate thesis project extending DreamCatalyst for text-driven 3D scene editing.

Pipeline:
- COLMAP
- Nerfstudio reconstruction (`nerfacto` / `splatfacto`)
- DreamCatalyst DDS editing (`dc` / `dc_splat`)
- optional refinement (`dc_refinement` / `dc_splat_refinement`)

The main research direction is now:
- stronger DDS guidance
- better localization / background preservation
- cleaner multi-scene evaluation and professor-facing presentation

For older historical notes, see `CLAUDE.md`.
This file is the current handoff summary.

## Most Important Files

- `nerfstudio/dc/dc.py`
  Main guidance/orchestration file. Core DDS flow still lives here.
- `nerfstudio/dc/attention_utils.py`
  STG attention-skip helpers and cross-attention capture processors.
- `nerfstudio/dc/localization_utils.py`
  Localization helpers: mask normalization, token selection, and cross-attention mask construction.
- `nerfstudio/dc/utils/logging_utils.py`
  WandB debug payload construction for DDS runs.
- `nerfstudio/dc/tasd_config.py`
  Central novelty config bundle used by the editing methods.
- `nerfstudio/3d_editing/dc_nerf/dc_config.py`
  Method configs that import `DC_CUSTOM_PARAMS`.
- `nerfstudio/3d_editing/dc_nerf/pipelines/dc_pipeline.py`
  Step 3 editing pipeline and image / mask logging.
- `scripts/edit.sh`
  Step 3 runner.
- `scripts/refine.sh`
  Step 4 runner with rep auto-detection.
- `scripts/evaluate.py`
  Offline metric evaluation.
- `docs/research_summary/research_summary.tex`
  Professor-facing research summary.

## Novelty / Paper Map

This section is intentionally search-friendly: if a fresh model needs more theory, it can search by the cited paper names directly.

### Guidance / DDS

- **TAG**
  - local implementation: tangential amplification of `noise_pred`
  - based on: **TAG** (Cho et al., arXiv 2510.04533)

- **Adaptive TAG**
  - local implementation: anneal `eta_tag` with timestep
  - based on: original extension inspired by **TAG**

- **Asymmetric TAG**
  - local implementation: apply TAG only to `eps_tgt`, not `eps_src`
  - based on: original extension built on **TAG**

- **STG**
  - local implementation: paper-faithful weak-model path via `STGIdentityValueAttnProcessor`
  - based on: **STG / Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling** (Hyung et al., CVPR 2025)
  - important note: do not revert this to returning the hook input; that was discussed and rejected

- **Perpendicular projection / Perp-Neg**
  - local implementation: orthogonalize `eps_tgt` with respect to `eps_src`
  - based on:
    - **PCGrad** (Yu et al., NeurIPS 2020)
    - **Perp-Neg** (Armandpour et al., ICML 2023)

- **Foreground-masked Perp-Neg**
  - local implementation: restrict the projection to foreground pixels
  - based on:
    - **PCGrad**
    - **Perp-Neg**
    - conceptual masking inspiration from **RoMaP**

### Localization

- **Self-derived latent relevance mask**
  - local implementation: build `M` from `||eps_tgt - eps_src||`
  - based on:
    - **LatentEditor** (delta-score localization)
    - conceptually aligned with **LENeRF**

- **Direct gradient masking**
  - local implementation: ablation where the final DDS gradient is multiplied by `M`
  - based on: local ablation of the self-derived mask branch

- **Source-blended localization**
  - local implementation:
    - `eps_tgt_loc = eps_src + M * (eps_tgt - eps_src)`
  - based on:
    - most directly inspired by **LatentEditor**
    - conceptually supported by **FoI**
    - conceptually supported by **ZONE**

- **Cross-attention semantic mask**
  - local implementation:
    - record target-token cross-attention maps from selected `attn2` modules in the UNet up-blocks
    - aggregate them into a latent-space mask
    - use that mask as a semantic prior to tighten the self-derived mask:
      - `M_hybrid = M_self * ((1 - w) + w * M_attn)`
  - based on:
    - **Prompt-to-Prompt** (cross-attention carries word-to-region layout information)
    - **What the DAAM** (aggregate diffusion cross-attention into attribution maps)
    - **From Text to Mask** (diffusion attention distilled into localization / segmentation maps)

- **Outside-mask background anchor**
  - local implementation: strengthen the existing `x0` preservation term with `(1 - M)`
  - based on:
    - local extension of the source-blend branch
    - conceptually aligned with preservation / anchoring ideas in **RoMaP**

### Other recent changes

- `dc.py` was refactored so the main guidance flow is smaller:
  - STG and cross-attention helper classes moved to `attention_utils.py`
  - localization/mask helper functions moved to `localization_utils.py`
  - WandB debug logging moved to `utils/logging_utils.py`
- `psi` is configurable from `tasd_config.py`
- post-TAG negative-prompt regularizer exists:
  - `tag_negative_prompt`
  - `tag_negative_strength`
  - it subtracts a post-TAG semantic direction
- cross-attention mask branch now exists:
  - `cross_attention_mask_enabled`
  - `cross_attention_mask_keywords`
  - `cross_attention_mask_layers`
  - `cross_attention_mask_weight`
  - it is intended to stop STG / Perp-Neg from contaminating the self-mask through `||eps_tgt - eps_src||`
- `edit.sh` now supports automatic evaluation after editing
- `evaluate.py` was patched to load the edited checkpoint directly from the current run folder instead of relying on `eval_setup`

### Useful related papers to search directly

- **DreamCatalyst**
- **DDS / Delta Denoising Score**
- **InstructPix2Pix**
- **TAG**
- **STG**
- **PCGrad**
- **Perp-Neg**
- **LatentEditor**
- **LENeRF**
- **FoI**
- **ZONE**
- **CustomNeRF**
- **RoMaP**
- **Prompt-to-Prompt**
- **What the DAAM**
- **From Text to Mask**

## Current Experimental Conclusions

### 1. Stormtrooper / person scene

This is the strongest evidence for the localization branch.

Best pure source-blend result so far:
- `src_blend_loc`
- `blur=2`
- `ema=0`
- `gamma=1.2`
- `warmup=0`
- `GS=7.5`

Metrics vs baseline:
- CLIP Direction: `0.1640 -> 0.1925`
- CLIP Image Sim: `0.6349 -> 0.7119`
- SSIM: `0.7951 -> 0.8354`
- LPIPS: `0.3394 -> 0.3000`
- MV Pairwise Cos Sim: `0.9224 -> 0.9243`

Interpretation:
- source-blended localization is a real win on this hard scene
- it improves editability and preservation at the same time

Best practical combined stormtrooper run:
- `Full TAG 1.15 + src_blend_loc + outside_mask_anchor_weight=0.05`

Representative metrics:
- Visual quality: `8.1/10`
- CLIP Direction: `0.1919`
- CLIP Image Sim: `0.7078`
- SSIM: `0.8428`
- LPIPS: `0.2915`
- MV Pairwise Cos Sim: `0.9194`

Important qualitative finding:
- self-mask at high guidance can create a ghost / duplicate stormtrooper
- source-blended localization removes that ghost at the same high guidance
- remaining issue becomes wall staining / low-frequency background drift

### 2. Face / Tolkien elf scene

This scene is not a clean win for source-blended localization.

After fixing the downscale mismatch, the face results became more reliable.

Main conclusion:
- Full TAG / TAG-only variants are still the safest face results
- source-blended localization often weakens the edit or makes it look dirty / monochromatic on face
- lowering `psi` did **not** help face
- `blur=1` and `gamma=0.9` were the least bad source-blend-only face settings, but still not convincing winners

Working interpretation:
- source-blend is currently a hard-scene localization contribution
- it is not yet a universal replacement for TAG on easy face scenes

### 3. Downscale mismatch was real

A previous reproduction mismatch came from inconsistent downscale between:
- reconstruction
- editing
- refinement

This affected qualitative behavior on the face scene, including the brown-hair vs green-hair discrepancy.

Current rule:
- keep the same downscale consistently across reconstruction, edit, and refine when doing fair comparisons

## Negative Prompt Branch: Current Status

The post-TAG negative prompt idea is still exploratory.

Important facts:
- it is applied **after TAG**, not inside CFG
- the earlier CFG-based version was intentionally removed
- current instruction-style prompts seem more promising than adjective lists because the model is IP2P-based

Current conclusion:
- the negative prompt branch has **not** convincingly solved the TAG brightness / saturation artifact
- larger values can introduce weird color spill, including green stain on clothes
- do not assume this branch is a success yet

Reasonable interpretation:
- TAG brightness seems to be only partly text-semantic
- so a prompt-based correction may be too indirect

## Evaluation / Script Gotchas

### `edit.sh`

Current behavior:
- launches Step 3 edit
- then automatically runs evaluation unless disabled

Useful environment knobs:
- `RUN_NAME`
- `PROJECT_NAME`
- `VIS_MODE`
- `EVAL_AFTER_EDIT`
- `EVAL_DEVICE`

### `refine.sh`

Safer than before:
- infers `nerf` vs `splat` from `LOAD_DIR` when `rep=auto`
- supports explicit downscale

### `evaluate.py`

This recently needed a fix because the old automatic evaluation path could re-enter a trainer-like setup and appear to start editing again.

The intended fixed behavior is:
- read `config.yml` from the finished run
- load the edited checkpoint from:
  - `<run_dir>/nerfstudio_models`
- render views
- compute metrics
- save `metrics.json` into the same run folder

If a fresh thread sees evaluation behaving like training again, this is the first thing to inspect.

## Research Summary Status

`docs/research_summary/research_summary.tex` is already heavily updated.

Current story:
- DreamCatalyst extensions for guidance and localization
- strong stormtrooper evidence for source-blended localization
- honest face-scene discussion
- concise professor-facing tone

Current stormtrooper figure strategy:
- matched ghost-comparison row:
  - `person_frame165.png`
  - `person_st_self-mask_b2e0g1.2w0_GS10.png`
  - `person_st_sbl_b2e0g1.2w0_GS10.png`
- matched best-result row:
  - `person_unedited_frame1.png`
  - `person_st_src-blend-loc_b2e0g1.png`

Important:
- the summary should stay concise
- do not bloat it with every ablation

## Recommended Next Technical Directions

Highest-value next steps:
- better localization signal than the current hand-tuned self-mask
- attention-grounded localization / hybrid mask research
- explicit background-preservation regularization
- cleaner cross-scene validation

Lower-value rabbit holes right now:
- long face-only source-blend sweeps
- pushing the post-TAG negative prompt too hard
- trying to make one single method dominate every scene before the evidence supports it

## Quick Orientation For A New Thread

If starting from scratch in a new thread, assume:

1. `dc.py` is the main file to inspect first.
2. If the question is about STG or cross-attention capture internals, inspect `attention_utils.py`.
3. If the question is about self-mask / attention-mask construction, inspect `localization_utils.py`.
4. If the question is about DC-side WandB debugging, inspect `utils/logging_utils.py`.
5. Stormtrooper is the best current demonstration of the localization contribution.
6. Face is the best stress test for “does this actually look natural?”
7. `source_blend_localization_enabled=True` is the main localization branch, but it is not universally dominant yet.
8. `Full TAG 1.15` remains important, especially on face.
9. Negative-prompt regularization is exploratory and currently weak.
10. Cross-attention masking is now implemented as a semantic prior for the self-mask, but is still unvalidated experimentally.
11. Automatic evaluation was recently patched and should be treated carefully until re-validated on the server.
