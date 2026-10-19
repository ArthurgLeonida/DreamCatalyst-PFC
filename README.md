# DreamCatalyst-PFC

> **Extending DreamCatalyst for text-driven 3D editing** — adaptive guidance and spatial localization in neural radiance fields.

Text-driven 3D scene editing built on top of **DreamCatalyst** (DDS-based score distillation with InstructPix2Pix). This is the undergraduate final project (PFC) of **Arthur Gislon Leonida** at **UFSC**, also presented as the final project for the **Computer Vision** course (Prof. **Aldo von Wangenheim**). The work focuses on **NeRF-based** reconstructions; the 3D Gaussian Splatting path is supported for completeness but is not the primary target.

> Base method — Kim et al. *"DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation"*. ICLR 2025. [arXiv:2407.11394](https://arxiv.org/abs/2407.11394)

The work is organized as **two independent contributions** sharing one repository:

- **Part 1 — Guidance, localization and preservation.** One universal configuration, applied unchanged to every scene: adaptive asymmetric TAG, STG with a bump schedule, an edit mask built from the instruction contrast and cross-attention with an agreement floor, source-blended localization, a background anchor weighted by a fixed source-side mask, and a latent-mean anchor. No per-scene tuning, keyword lists or precomputed segmentation.
- **Part 2 — 3D voxel cache.** An optional non-parametric world-space voxel grid that aggregates the per-view edit evidence across cameras and returns it to later views. Layered on top of the Part 1 configuration with every other value unchanged. See [`docs/VoxelCacheExplained.md`](docs/VoxelCacheExplained.md).

A short write-up of the method and results is in [`research_summary.pdf`](research_summary.pdf).

## Results at a glance

Final configuration (code since 2026-10-13), **9 edits on 6 scenes**, **3 runs per configuration**, mean over edits, NVIDIA H100, guidance scales pinned at the DreamCatalyst values (`7.5` text, `1.5` image). A per-edit difference counts only when it exceeds the combined run spread `sqrt(σ_A² + σ_B²)`; the count columns give the number of edits on which each side is better, with the number beyond that spread in parentheses.

Edits: face → Tolkien elf, face → skull, fangzhou → knight helmet, yuseung → Batman, yuseung → Joker, person → clown, person → stormtrooper, bear statue → panda, red velvet cake → chocolate cake (CO3D).

**Part 1 — DreamCatalyst → universal configuration (2D)**

| Metric | DreamCatalyst | Proposed | Proposed better | DreamCatalyst better |
|---|---|---|---|---|
| CLIP-dir ↑ (editability) | 0.158 | 0.160 | 5 (3) | 4 (1) |
| CLIP-I ↑ | 0.677 | 0.710 | 8 (7) | 1 (1) |
| SSIM ↑ | 0.818 | 0.860 | 9 (9) | 0 |
| LPIPS ↓ | 0.289 | 0.233 | 9 (9) | 0 |
| MV-cos ↑ | 0.923 | 0.925 | 5 (3) | 4 (2) |
| CLIP-dir-local ↑ (in the edit region) | 0.174 | 0.173 | 5 (5) | 4 (3) |
| CLIP-dir-bg → 0 (background leakage) | 0.001 | −0.001 | 5 (2) | 4 (0) |
| Edit localization ratio ↑ | 0.660 | 0.694 | 7 (6) | 2 (2) |
| Background LPIPS ↓ | 0.169 | 0.131 | 9 (9) | 0 |
| Background PSNR ↑ (dB) | 25.07 | 28.86 | 9 (8) | 0 |

Headline: **better preservation at unchanged whole-image editability.** SSIM, LPIPS and background LPIPS improve on all nine edits beyond the run spread, background PSNR rises by 3.8 dB, and the mean change outside the edit region falls by 44%, while CLIP-direction is tied (0.158 vs 0.160). Restricted to the edit region, editability rises on five edits and falls on four (bear, cake, clown, fangzhou), where the baseline changes the edit region more.

**Part 2 — 2D configuration → + voxel cache (3D)**

The cache is **metric-neutral in aggregate**: every mean changes by less than the typical run-to-run spread (SSIM 0.860 and `EditMaskVariance_3D` 0.018 with and without it, CLIP-dir 0.160 → 0.159). Of the 99 per-edit comparisons (11 metrics × 9 edits), 19 exceed the run spread, 11 in favour of the cache (six on stormtrooper) and 8 against (three on clown). Its visible effect is qualitative and cross-view: on the Batman edit without the cache the hands are gloved in one view and bare in another; with the cache they are gloved in both (one run).

The undergraduate monograph reports an earlier configuration (8 edits on 5 scenes) that this code no longer reproduces; the numbers above are those of the final configuration.

## Pipeline

```
Photos/Video ──► COLMAP ──► Nerfacto (NeRF) ──► DreamCatalyst (edit) ──► Refinement (optional, not evaluated)
                  Step 1       Step 2                 Step 3                      Step 4
```

3DGS alternative: substitute `splatfacto` in Step 2 and `dc_splat` / `dc_splat_refinement` in Steps 3–4.

All contributions live in **Step 3**.

## Setup

```bash
bash setup.sh ns        # creates conda env 'dreamcatalyst_ns', installs everything
conda activate dreamcatalyst_ns
```

## Step 1 — Process data (COLMAP)

```bash
bash scripts/process_data.sh <scene>            # from images in data/<scene>/images/
bash scripts/process_data.sh <scene> video      # from video
```

Output: `data/<scene>_processed/` with `transforms.json`.

> **Robust model selection.** COLMAP's mapper can emit several disconnected sub-models (`sparse/0`, `sparse/1`, …) numbered by creation order, not size. `ns-process-data` reads `sparse/0` by default, which is occasionally a tiny fragment while the real reconstruction sits elsewhere. `process_data.sh` automatically picks the model with the most registered images and regenerates `transforms.json` from it — no COLMAP recompute.

## Step 2 — Train NeRF reconstruction

```bash
bash scripts/train.sh <scene> 30000 1 nerfacto
# equivalent to: ns-train nerfacto ... nerfstudio-data --data data/<scene>_processed
```

Output: `outputs/<scene>/nerfacto/<timestamp>/`

> The third argument is the downscale (`1`, `2`, or `4`); use the same value for reconstruction, editing, and refinement. For the 3DGS path, pass `splatfacto` instead of `nerfacto` as the fourth argument.

## Step 3 — Edit with DreamCatalyst (DDS)

```bash
RUN_NAME=face_elf bash scripts/edit.sh face \
    "a photo of a face" \
    "Turn him into the Tolkien Elf" \
    outputs/face/nerfacto/<timestamp>/nerfstudio_models/ \
    3000 nerf 1
# Usage: bash scripts/edit.sh <scene> <src_prompt> <tgt_instruction> <load_dir> [max_iters=3000] [rep=nerf] [downscale=1]
# rep: nerf (default for this project) or splat
```

Loads the reconstruction from Step 2 and optimizes the NeRF (or Gaussians) with DDS guidance and the extensions described in [Method](#method). The configuration comes entirely from `nerfstudio/dc/method_config.py`; the script only sets runtime options.

- `max_iters` is forwarded to `--pipeline.dc.max-iteration`, so the timestep annealing, the TAG schedule and the STG bump span the whole run.
- Environment options: `RUN_NAME` (experiment name, default `<scene>_dc_edit`), `SEED` (default `42`), `EVAL_AFTER_EDIT` (default `1`: runs `scripts/evaluate.py eval` on the finished run; set `0` to skip), `EVAL_DEVICE`, `VIS_MODE` (default `wandb`), `PROJECT_NAME`, `CUDA_VISIBLE_DEVICES` (picked automatically when unset).
- Output: `outputs/<RUN_NAME>/dc/<timestamp>/` (`dc_splat` for the 3DGS path).

**Prompts.** The source prompt is a short caption of what the scene contains (`"a photo of a face"`, `"a photo of a bear statue"`). The target is an **InstructPix2Pix instruction** (`"Turn him into a clown"`, `"Put a knight's helmet on him"`, `"Make it chocolate cake"`). The cross-attention edit tokens are derived automatically: the words of the instruction that are neither stopwords nor present in the source prompt. Name the new concept explicitly in the instruction.

**IP2P model:** Step 3 uses `timbrooks/instruct-pix2pix` via a CLI override in `scripts/edit.sh`. The `DCConfig.sd_pretrained_model_or_path` default stays at `runwayml/stable-diffusion-v1-5` — do not change it (refinement depends on SD 1.5).

### Render after editing

```bash
ns-render interpolate \
    --load-config outputs/<RUN_NAME>/dc/<timestamp>/config.yml \
    --output-path renders/<RUN_NAME>_edited.mp4
```

## Step 4 — Refinement (supported, not evaluated)

```bash
RUN_NAME=my_refine bash scripts/refine.sh <scene> \
    "a photo of a Tolkien elf" \
    outputs/<RUN_NAME>/dc/<timestamp>/nerfstudio_models/ \
    30000
```

Uses SDEdit (SD 1.5) to refine the current edited renders and retrains the NeRF against the resulting detached 2D targets. This step was part of the original DreamCatalyst pipeline and is kept for completeness, but it is **not part of the experimental evaluation**: every reported number comes from the Step 3 checkpoint. The option remains available for anyone who wants to produce polished final renders.

## Method

The DDS orchestration lives in `nerfstudio/dc/dc.py`; one formula per function in `nerfstudio/dc/guidance_utils.py`; mask shaping and edit-token selection in `nerfstudio/dc/localization_utils.py`; attention processors in `nerfstudio/dc/attention_utils.py`; the voxel cache in `nerfstudio/dc/mask_voxel_cache.py` and `nerfstudio/3d_editing/dc_nerf/pipelines/dc_pipeline.py`. **Every value is set in `nerfstudio/dc/method_config.py`**: `DC_CUSTOM_PARAMS` (unpacked into `DCConfig`) and `VOXEL_CACHE_PARAMS` (loaded by `DCPipelineConfig`). That file is the single source of truth; the values below are copied from it.

One iteration edits one view, in this order:

1. IP2P noise predictions for the current render (target branch) and the source image (source branch), with the same noise draw.
2. Localization signals, read from these predictions **before any amplification**. A mask built after TAG/STG would grow wherever the amplifier is strongest and feed back into its own gate.
3. Amplification of the target prediction only (TAG and STG).
4. Gradient: `g = w_DDS(t) · M ⊙ (ε̃_tgt − ε_src) + e^t · ψ_eff ⊙ (x₀ − x₀_src) + latent-mean anchor`, where `x₀` and `x₀_src` are the clean latents of the current render and of the source image.

### Localization: where the edit may act

| Signal | Definition | Settings |
|---|---|---|
| `S`, target-instruction mask | `‖ε(render, I_src, instruction) − ε(render, I_src, "")‖` per latent pixel: what the instruction still demands of the current render, read from IP2P passes that are already computed. Normalized per frame to its p5–p95 window, smoothed by a per-view EMA, sharpened and blurred. | `gradient_mask_ema_memory_visits=5.0` (β = e^(−1/5) per visit of that view), `gradient_mask_gamma=1.2`, `gradient_mask_blur=0.5` |
| `S_src`, source-instruction mask | The same contrast on the source latent noised at t = 500, averaged over 4 noise draws from a private generator, computed once per view and cached. | `localization_source_timestep_ratio=0.5`, `localization_source_num_samples=4`, `localization_source_seed=42` |
| `A`, cross-attention mask | Attention of the edit tokens in UNet up-blocks 1 and 2, at full strength from the first iteration (Prompt-to-Prompt, DAAM, LEDITS++). | `cross_attention_mask_layers=[1, 2]`, `cross_attention_mask_gamma=1.2`, `cross_attention_mask_blur=0.5` |
| `M`, edit mask | `M = max(S·A, min(S, S_src))`. Attention restricts `S`, but cannot veto a region that both instruction masks mark (the agreement floor). | — |
| Source-blended localization | `ε_tgt^loc = ε_src + M ⊙ (ε̃_tgt − ε_src)`: the DDS delta, including what TAG and STG added, vanishes outside `M`. Only the edit force is masked (LatentEditor, FoI, ZONE). | `source_blend_localization_enabled=True` |

### Guidance: how strongly it acts

| Mechanism | Definition | Settings |
|---|---|---|
| Edit strength `s` | `s = ‖ε_tgt − ε_src‖ / (‖ε_tgt‖ + ‖ε_src‖) ∈ [0, 1]` on the guided predictions before amplification. Scales STG and the background anchor by `(1 − s)`, so one configuration serves identity-preserving and structural edits. | — |
| Adaptive asymmetric TAG (Cho et al.) | Scales the tangential part of `ε_tgt` (relative to the noisy latent) by `η(t) = 1 + (η_max − 1) · t^(1/e)`; the source branch is left unchanged. | `eta_tag=1.2` (`1.0` = off) |
| STG with a bump schedule (Hyung et al.) | Adds `s_stg · a(p) · (1 − s) · (ε_tgt − ε_weak)`, where `ε_weak` comes from a pass whose up-block-2 self-attention is replaced by an identity-value processor. `a(p)` rises from 0 at the start of training to 1 at 40% and returns to 0 at 80%, so the last fifth runs without it. Composed in parallel with TAG. | `stg_scale=3.0` (`0.0` = off), `stg_skip_layers=[2]`, `stg_schedule_start_ratio=0.0`, `stg_schedule_end_ratio=0.8`, `stg_bump_peak_ratio=0.5` |

### Preservation: what stays like the source

| Mechanism | Definition | Settings |
|---|---|---|
| DreamCatalyst term | `e^t · ψ · (x₀ − x₀_src)` (upstream). | `psi=0.075` |
| Outside-mask anchor (aligned with RoMaP) | `ψ_eff = ψ + w_out · (1 − s) · (1 − S_src)`. Weighted by the fixed source-side mask, so neither accidental drift nor the voxel cache can relax the restoring force. | `outside_mask_anchor_weight=0.25` |
| Latent-mean anchor (aligned with Piva, Stable Score Distillation) | Adds `λ · (mean(x₀) − mean(x₀_src))` per latent channel, against the slow brightness and saturation drift that TAG causes. | `latent_mean_anchor_weight=0.005` |

### 3D voxel cache (Part 2)

A 64³ world-space grid, bounded by the points the field renders during the first 50 edit steps, stores the instruction contrast of every latent pixel (normalized per frame by its 0.95 quantile) at its backprojected surface point, using only rays whose accumulation exceeds 0.5. Each voxel keeps an EMA mean (`β = 1 − 1/(0.5 · N_cam)`), an exponentially weighted cross-view variance (newest-view weight 0.2, updated once per view), its distinct-view count and the spread of its viewing directions. Later views read it back with observed-weighted trilinear interpolation, stretch the read to [0, 1], and add support to `M` through a positive-only fusion:

```
M_final = M + α(t) · C̃ · G · max(M_3D − M, 0),   G = max(A, S) · A
C̃ = count gate · (1 − σ² / 0.02)₊ · angular-spread gate · mass gate (threshold 0.18)
α(t): 0 → 0.2 over edit steps 500 → 1200
```

The gate `G` requires attention to confirm every addition, and `σ²_max = 0.02` is one global value for every scene and edit. Full design, with a knob-to-symbol map: [`docs/VoxelCacheExplained.md`](docs/VoxelCacheExplained.md).

### Supported configurations

The code reproduces exactly three configurations; nothing is tuned per scene.

| Configuration | How to obtain it in `method_config.py` |
|---|---|
| 3D (Part 2) | as committed |
| 2D (Part 1) | `mask_voxel_cache_enabled=False` |
| DreamCatalyst baseline | `source_blend_localization_enabled=False`, `outside_mask_anchor_weight=0.0`, `cross_attention_mask_enabled=False`, `latent_mean_anchor_weight=0.0`, `eta_tag=1.0`, `stg_scale=0.0`, `mask_voxel_cache_enabled=False` |

Mechanisms tried and removed during development (Perp-Neg, a scheduled cross-attention weight, a self-mask built from the DDS delta, a camera-count EMA, contested-region suppression, a peak-held variance latch, among others) are no longer in the code. The screening that selected the final masks is recorded in [`docs/LocalizationAblations.md`](docs/LocalizationAblations.md).

## Evaluation

`scripts/evaluate.py eval` loads the finished checkpoint from `<run_dir>/nerfstudio_models`, renders every training view, compares it with the captured image, and writes `metrics.json` into the run folder. `scripts/edit.sh` runs it automatically.

**Whole-image metrics**

- `CLIP_direction` (↑) — editability (CLIP ViT-L/14 directional similarity)
- `CLIP_img_sim` (↑), `SSIM` (↑), `LPIPS` (↓, VGG) — preservation
- `MultiView_pairwise_cos_sim` (↑) — whether the edited scene looks like the same object from every camera
- `EditMaskVariance_3D` (↓) — the rendered edit magnitude `‖I_edit − I_src‖` backprojected into a shared 64³ grid; mean cross-view variance over voxels seen by at least 3 views. A weak edit lowers it for the wrong reason, so read it with `CLIP_direction`.

**Region-restricted metrics** (the PIE-Bench protocol, applied per view)

- `CLIP_direction_local` (↑) — editability inside the edit region
- `CLIP_direction_bg` (→ 0) — whether the background moved toward the target text
- `Edit_Localization_Ratio` (↑) — share of the change that landed inside the region (0.5 = uniform)
- `Background_LPIPS` (↓), `Background_PSNR` (↑) — preservation outside the region
- `Edit_Magnitude_in_region` / `_out_region` (context, not scores) and `Region_valid_views` / `Region_mask_mean_coverage` (provenance)

The region is segmented by CLIPSeg (`CIDAS/clipseg-rd64-refined`) from the **source** frames only, so it is identical for every method compared on a scene, and dilated by 0.05 of the image diagonal (one global value, never set per scene). The suite is skipped cleanly when CLIPSeg is unavailable and can be turned off with `--no-localized-metrics`.

```bash
bash scripts/evaluate.sh <run_dir>/config.yml "<src_prompt>" "<tgt_instruction>" [output_dir]
DRY_RUN=1 bash scripts/reevaluate_all.sh                 # plan a re-score of every finished run, no retraining
bash scripts/reevaluate_all.sh                           # re-score; keeps the old metrics.json as metrics.prev.json
python scripts/collect_metrics.py --out metrics.csv      # one CSV row per run (reads JSON only, no GPU)
python scripts/evaluate.py push-wandb <run_dir> [...]    # replay metrics.json into its WandB run, no re-render
```

Each configuration is run three times per edit, and a difference is called real only when it exceeds the combined run spread. The training seed is set with `SEED` (default 42) and recorded as `machine.seed` in each run's `config.yml`. Comparisons across runs must use the **same `downscale`** for reconstruction, editing and evaluation, and the same reconstruction checkpoint per scene.

## Datasets

| Dataset | Link |
|---|---|
| 🗂️ Dataset — Instruct-NeRF2NeRF | [Google Drive](https://drive.google.com/drive/folders/1v4MLNoSwxvSlWb26xvjxeoHpgjhi_s-s) |
| 🗂️ Dataset — PDS scenes | [OneDrive](https://onedrive.live.com/?id=%2Fpersonal%2F0ce615b143fc4bdc%2FDocuments%2Frelease_data%2FPDS%2Fdata&listurl=%2Fpersonal%2F0ce615b143fc4bdc%2FDocuments) |
| 🗂️ Dataset — CO3D (cake, sequence `374_42274_84517`) | [GitHub](https://github.com/facebookresearch/co3d) |

Please cite the original sources if you use these scenes.

## Environment

| Component | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.1.2+cu118 |
| CUDA | 11.8 |
| Nerfstudio | 1.0.2 |
| diffusers | 0.27.2 |
| COLMAP | ≤ 3.9.1 |

## References

```bibtex
@inproceedings{kim2025dreamcatalyst,
  title     = {DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation},
  author    = {Jiwook Kim and Seonho Lee and Jaeyo Shin and Jiho Choi and Hyunjung Shim},
  booktitle = {ICLR},
  year      = {2025},
  url       = {https://arxiv.org/abs/2407.11394},
}

@inproceedings{hertz2023dds,
  title     = {Delta Denoising Score},
  author    = {Amir Hertz and Kfir Aberman and Daniel Cohen-Or},
  booktitle = {ICCV},
  year      = {2023},
  url       = {https://arxiv.org/abs/2304.07090},
}

@inproceedings{brooks2023instructpix2pix,
  title     = {InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author    = {Tim Brooks and Aleksander Holynski and Alexei A. Efros},
  booktitle = {CVPR},
  year      = {2023},
  url       = {https://arxiv.org/abs/2211.09800},
}

@inproceedings{instructnerf2023,
  title     = {Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions},
  author    = {Haque, Ayaan and Tancik, Matthew and Efros, Alexei and Holynski, Aleksander and Kanazawa, Angjoo},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023},
}

@inproceedings{koo2024pds,
  title     = {Posterior Distillation Sampling},
  author    = {Juil Koo and Chanho Park and Minhyuk Sung},
  booktitle = {CVPR},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.13831},
}

@inproceedings{cho2026tag,
  title     = {TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling},
  author    = {Hyunmin Cho and Donghoon Ahn and Susung Hong and Jee Eun Kim and Seungryong Kim and Kyong Hwan Jin},
  booktitle = {ICML},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.04533},
}

@inproceedings{hyung2025stg,
  title     = {Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling},
  author    = {Junha Hyung and Kinam Kim and Susung Hong and Min-Jung Kim and Jaegul Choo},
  booktitle = {CVPR},
  year      = {2025},
  url       = {https://arxiv.org/abs/2411.18664},
}

@inproceedings{khalid2024latenteditor,
  title     = {{LatentEditor}: Text Driven Local Editing of 3D Scenes},
  author    = {Umar Khalid and Hasan Iqbal and Nazmul Karim and Muhammad Tayyab and Jing Hua and Chen Chen},
  booktitle = {ECCV},
  year      = {2024},
}

@inproceedings{hertz2023p2p,
  title     = {Prompt-to-Prompt Image Editing with Cross-Attention Control},
  author    = {Amir Hertz and Ron Mokady and Jay Tenenbaum and Kfir Aberman and Yael Pritch and Daniel Cohen-Or},
  booktitle = {ICLR},
  year      = {2023},
}

@inproceedings{tang2023daam,
  title     = {What the DAAM: Interpreting Stable Diffusion Using Cross Attention},
  author    = {Raphael Tang and Linqing Liu and Akshat Pandey and Zhiying Jiang and Gefei Yang and Karun Kumar and Pontus Stenetorp and Jimmy Lin and Ferhan Ture},
  booktitle = {ACL},
  year      = {2023},
}

@inproceedings{couairon2023diffedit,
  title     = {DiffEdit: Diffusion-based Semantic Image Editing with Mask Guidance},
  author    = {Guillaume Couairon and Jakob Verbeek and Holger Schwenk and Matthieu Cord},
  booktitle = {ICLR},
  year      = {2023},
  url       = {https://arxiv.org/abs/2210.11427},
}

@inproceedings{brack2024leditspp,
  title     = {{LEDITS++}: Limitless Image Editing using Text-to-Image Models},
  author    = {Manuel Brack and Felix Friedrich and Katharina Kornmeier and Linoy Tsaban and Patrick Schramowski and Kristian Kersting and Apolinario Passos},
  booktitle = {CVPR},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.16711},
}

@inproceedings{huang2025dacapo,
  title     = {{DaCapo}: Score Distillation as Stacked Bridge for Fast and High-Quality 3D Editing},
  author    = {Yufei Huang and Bangyan Liao and Yuqi Hu and Haitao Lin and Lirong Wu and Siyuan Li and Cheng Tan and Zicheng Liu and Yunfan Liu and Zelin Zang and Chang Yu and Zhen Lei},
  booktitle = {CVPR},
  year      = {2025},
}

@article{le2024piva,
  title     = {Preserving Identity with Variational Score for General-purpose 3D Editing},
  author    = {Duong H. Le and Tuan Pham and Aniruddha Kembhavi and Stephan Mandt and Wei-Chiu Ma and Jiasen Lu},
  journal   = {arXiv preprint arXiv:2406.08953},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.08953},
}

@article{ssd2025,
  title     = {Stable Score Distillation},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.09168},
}

@inproceedings{kim2025romap,
  title     = {{RoMaP}: Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling},
  author    = {Hayeon Kim and Ji Ha Jang and Se Young Chun},
  booktitle = {ICCV},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.11061},
}

@inproceedings{ju2024piebench,
  title     = {{PnP} Inversion: Boosting Diffusion-Based Editing with 3 Lines of Code},
  author    = {Xuan Ju and Ailing Zeng and Yuxuan Bian and Shaoteng Liu and Qiang Xu},
  booktitle = {ICLR},
  year      = {2024},
}

@inproceedings{luddecke2022clipseg,
  title     = {Image Segmentation Using Text and Image Prompts},
  author    = {Timo L{\"u}ddecke and Alexander S. Ecker},
  booktitle = {CVPR},
  year      = {2022},
}

@inproceedings{miao2025uds,
  title     = {Rethinking Score Distilling Sampling for 3D Editing and Generation},
  author    = {Miao and collaborators},
  booktitle = {ICML},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.01888},
}
```
