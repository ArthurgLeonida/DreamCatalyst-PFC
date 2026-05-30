# DreamCatalyst-PFC

Text-driven 3D scene editing built on top of **DreamCatalyst** (DDS-based score distillation). This undergraduate thesis project (UFSC, PFC) focuses on **NeRF-based** reconstructions — the 3D Gaussian Splatting path is supported for completeness but is not the primary target.

> Kim et al. *"DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation"*. ICLR 2025. [arXiv:2407.11394](https://arxiv.org/abs/2407.11394)

The work is organized as **two independent contributions** sharing one repository:

- **Part 1 — Guidance & localization.** Universal-config improvements to DreamCatalyst's DDS guidance: TAG family, STG with scheduling, self-derived relevance mask, source-blended localization, cross-attention semantic mask, outside-mask background anchor, latent-mean anchor. Evaluated as a single config across multiple scenes.
- **Part 2 — 3D voxel cache.** An optional non-parametric voxel grid that aggregates per-view diffusion masks across views to enforce 3D consistency. Layered on top of the Part 1 config. See [`docs/VoxelCacheExplained.md`](docs/VoxelCacheExplained.md).

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
bash scripts/train.sh <scene> 30000 nerf
# equivalent to: ns-train nerfacto ... nerfstudio-data --data data/<scene>_processed
```

Output: `outputs/<scene>/nerfacto/<timestamp>/`

> For the 3DGS path, pass `splat` instead of `nerf`.

## Step 3 — Edit with DreamCatalyst (DDS)

```bash
RUN_NAME=my_edit bash scripts/edit.sh <scene> \
    "a photo of a sks man" \
    "a photo of a Tolkien elf" \
    outputs/<scene>/nerfacto/<timestamp>/nerfstudio_models/ \
    3000 nerf
# Usage: bash scripts/edit.sh <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep]
# rep: nerf (default for this project) or splat
```

Loads the reconstruction from Step 2 and optimizes the NeRF (or Gaussians) toward the target prompt using DDS guidance with the extensions listed in [Novelties](#novelties).

`max_iters` is forwarded to `--pipeline.dc.max-iteration` so the timestep schedule covers the full training budget. `EVAL_AFTER_EDIT=true` automatically invokes `scripts/evaluate.py` on the finished run.

**Prompt guidelines:**
- Describe the **full scene**, not just the object.
- Keep source and target prompts as close as possible — only change the edited element.
- Prefix with `"a photo of"` to anchor the diffusion model to photorealistic outputs.

**IP2P model:** Step 3 uses `timbrooks/instruct-pix2pix` via a CLI override in `scripts/edit.sh`. The `DCConfig.sd_pretrained_model_or_path` default stays at `runwayml/stable-diffusion-v1-5` — do not change it (refinement depends on SD 1.5).

Output: `outputs/<scene>/dc/<timestamp>/`

### Render after editing

```bash
ns-render interpolate \
    --load-config outputs/<scene>/dc/<timestamp>/config.yml \
    --output-path renders/<scene>_edited.mp4
```

## Step 4 — Refinement (supported, not evaluated in this thesis)

```bash
RUN_NAME=my_refine bash scripts/refine.sh <scene> \
    "a photo of a Tolkien elf" \
    outputs/<scene>/dc/<timestamp>/nerfstudio_models/ \
    30000
```

Uses SDEdit (SD 1.5, 20 denoising steps with `skip=7`) to produce edited 2D images and retrains the NeRF against them. This step was part of the original DreamCatalyst pipeline and is kept here for completeness, but it is **not part of the current experimental evaluation**: Step 3 is where the contributions live, and refinement adds significant runtime without altering the scientific claim. The option remains available for anyone who wants to produce polished final renders.

## Novelties

The main DDS orchestration lives in `nerfstudio/dc/dc.py`; reusable novelty math lives in `nerfstudio/dc/guidance_utils.py`; **all knobs are configured centrally in `nerfstudio/dc/method_config.py`** — `DC_CUSTOM_PARAMS` (Part 1, unpacked into `DCConfig`) and `VOXEL_CACHE_PARAMS` (Part 2, loaded by `DCPipelineConfig`). That file is the single source of truth for which mechanisms are active in any run; consult it rather than any copy here.

### Localization branch — Part 1 (main research direction)

| Novelty | Config | Description |
|---|---|---|
| **Source-blended localization** | `source_blend_localization_enabled` | Replaces the DDS target with `eps_src + M·(eps_tgt − eps_src)`, so the edit signal vanishes outside the mask. Motivated by LatentEditor / FoI / ZONE. |
| **Outside-mask background anchor** | `outside_mask_anchor_weight` | Strengthens the preservation term by `w_out · (1 − M)`, tightening `x0` outside the mask. Conceptually aligned with RoMaP. |
| **Edit-strength-adaptive anchor** | `outside_mask_anchor_edit_strength_adaptive` | Scales `w_out` by `(1 − s)`, where `s = ‖eps_tgt − eps_src‖ / (‖eps_tgt‖ + ‖eps_src‖) ∈ [0, 1]` is the scene-level edit strength from the raw pre-guidance noise predictions. Time-invariant (numerator and denominator scale together with timestep). Low on identity-preserving edits (face / elf), high on structural edits (person → stormtrooper). |
| **Cross-attention semantic mask** | `cross_attention_mask_enabled` + `cross_attention_mask_{layers,weight,gamma,blur}` | Aggregates target-token cross-attention from selected UNet up-blocks, fuses with the self-mask as `M_hybrid = M_self · ((1 − w) + w · M_attn)`. Target-token selection is auto-derived from src/tgt prompts (no manual keyword overrides). Based on Prompt-to-Prompt, What the DAAM, DiffEdit, LEDITS++. |
| **Latent-mean anchor (N2)** | `latent_mean_anchor_weight` | Adds `λ · (mean(tgt_x0) − mean(src_x0))` per channel onto the final gradient — penalizes VAE-latent channel-mean drift and counteracts TAG brightness/saturation artifacts directly in latent statistics. `=0.0` disables. Conceptually aligned with Piva and Stable Score Distillation. |

### TAG branch — Part 1 (edit strength)

| Novelty | Config | Description |
|---|---|---|
| **TAG** | `eta_tag` | Amplifies the tangential component of `noise_pred` with respect to the noisy latent. `eta_tag=1.0` disables. Based on TAG (Cho et al., 2024). |
| **Adaptive TAG** | `adaptive_tag` | Anneals `η(t) = 1 + (eta_tag − 1) · t_norm^(1/e)` so amplification is strongest at high noise and decays toward 1.0 near the clean regime. |
| **Asymmetric TAG** | `asymmetric_tag` | Applies TAG only to the target branch, leaving `eps_src` at `η = 1.0`. |

### STG branch — Part 1 (structural amplification)

| Novelty | Config | Description |
|---|---|---|
| **STG** | `stg_enabled`, `stg_scale`, `stg_skip_layers` | Runs a weak UNet pass via `STGIdentityValueAttnProcessor` on selected up-blocks and amplifies `eps = eps_full + s · (eps_full − eps_weak)`. Based on STG (Hyung et al., CVPR 2025). Target-branch only. |
| **STG schedule** | `stg_schedule_enabled`, `stg_schedule_mode`, `stg_schedule_{start,end}_ratio`, `stg_bump_peak_ratio` | Three shapes: `"decay"` (STG early, off late — strongest on monotonic edits where every step pushes the same direction), `"growth"` (STG off early, on late), `"bump"` (triangle: STG peaks mid-phase and returns to 0 before the end). |
| **Edit-strength-adaptive STG** | `stg_edit_strength_adaptive` | Multiplies the scheduled STG scale by `(1 − s)`, where `s` is the same per-step edit strength used by the anchor. Identity edits keep STG near full strength; structural edits fade STG automatically. |
| **STG/TAG composition** | `stg_tag_compose_mode` | `"parallel"` applies TAG and STG additively to the raw CFG prediction (no cross-amplification); `"sequential"` nests them (TAG amplifies STG's perturbation). |

### 3D voxel cache — Part 2 (multi-view consistency)

| Novelty | Config | Description |
|---|---|---|
| **Voxel-grid mask cache** | `mask_voxel_cache_enabled` + `mask_voxel_cache_*` | Backprojects per-view diffusion masks into a coarse 3D voxel grid via rendered depth, EMA-aggregates across views, and queries them back per view to provide a 3D-consistent localization signal. Confidence gates on cross-view Welford variance, angular triangulation diversity, and cached mask mass. Full design in [`docs/VoxelCacheExplained.md`](docs/VoxelCacheExplained.md). |

## Evaluation

`scripts/evaluate.py` loads the finished edited checkpoint from `<run_dir>/nerfstudio_models`, renders the evaluation views, computes metrics, and writes `metrics.json` into the same run folder. Reported metrics:

- `CLIP_direction` (↑) — editability
- `CLIP_img_sim` (↑) — content preservation
- `SSIM` (↑) — identity preservation
- `LPIPS` (↓) — perceptual distance
- `MultiView_pairwise_cos_sim` (↑) — multi-view consistency
- `EditMaskVariance_3D` (↓) — direct 3D mask-consistency metric for the Part 2 cache

Comparisons across runs must use the **same `downscale`** for reconstruction, edit, and (when run) refinement; inconsistent downscales have produced spurious qualitative differences in the past.

## Datasets

Scenes are sourced from prior 3D-editing work and re-processed locally through `scripts/process_data.sh`:

- **DreamCatalyst** release (which redistributes scenes originally introduced for **Posterior Distillation Sampling**, PDS).
- **Instruct-NeRF2NeRF** (e.g. the `campsite` / `bear` / outdoor captures).

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

@article{cho2024tag,
  title     = {Tangential Amplified Guidance for Score Distillation Sampling},
  author    = {Juhyung Cho and Jaehyeok Shim and Seungryong Kim},
  journal   = {arXiv preprint arXiv:2510.04533},
  year      = {2024},
}

@inproceedings{hyung2025stg,
  title     = {Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling},
  author    = {Minyoung Hyung and Jaegul Choo},
  booktitle = {CVPR},
  year      = {2025},
}

@inproceedings{yu2020pcgrad,
  title     = {Gradient Surgery for Multi-Task Learning},
  author    = {Tianhe Yu and Saurabh Kumar and Abhishek Gupta and Sergey Levine and Karol Hausman and Chelsea Finn},
  booktitle = {NeurIPS},
  year      = {2020},
}

@inproceedings{armandpour2023perpneg,
  title     = {Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus Problem and Beyond},
  author    = {Mohammadreza Armandpour and Ali Sadeghian and Huangjie Zheng and Amir Sadeghian and Mingyuan Zhou},
  booktitle = {ICML},
  year      = {2023},
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
  title     = {{DaCapo}: Score Distillation as Stacked Bridge for Fast and High-quality 3D Editing},
  author    = {Huang and collaborators},
  booktitle = {CVPR},
  year      = {2025},
}

@article{piva2024,
  title     = {Preserving Identity with Variational Score for General-purpose 3D Editing},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.08953},
}

@article{ssd2025,
  title     = {Stable Score Distillation},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.09168},
}

@article{kim2025romap,
  title     = {{RoMaP}: Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling},
  author    = {Hayeon Kim and Ji Ha Jang and Se Young Chun},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.11061},
}

@inproceedings{miao2025uds,
  title     = {Rethinking Score Distilling Sampling for 3D Editing and Generation},
  author    = {Miao and collaborators},
  booktitle = {ICML},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.01888},
}
```
