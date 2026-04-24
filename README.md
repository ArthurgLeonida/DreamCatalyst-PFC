# DreamCatalyst-PFC

Text-driven 3D scene editing built on top of **DreamCatalyst** (DDS-based score distillation). This undergraduate thesis project (UFSC, PFC) focuses on **NeRF-based** reconstructions — the 3D Gaussian Splatting path is supported for completeness but is not the primary target.

> Kim et al. *"DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation"*. ICLR 2025. [arXiv:2407.11394](https://arxiv.org/abs/2407.11394)

## Pipeline

```
Photos/Video ──► COLMAP ──► Nerfacto (NeRF) ──► DreamCatalyst (edit) ──► Refinement (optional, not evaluated)
                  Step 1       Step 2                 Step 3                      Step 4
```

3DGS alternative: substitute `splatfacto` in Step 2 and `dc_splat` / `dc_splat_refinement` in Steps 3–4.

## Research scope

The contributions of this work are concentrated in **Step 3**. Extensions target:
- stronger DDS guidance (TAG family, STG, Perp-Neg),
- better localization / background preservation (self-derived relevance mask, source-blended localization, cross-attention semantic mask, outside-mask background anchor),
- cleaner multi-scene evaluation.

Stormtrooper / person is the strongest localization showcase; face / elf is the naturalness and preservation stress test.

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

Uses SDEdit (SD 1.5, 20 denoising steps with `skip=7`) to produce edited 2D images and retrains the NeRF against them. This step was part of the original DreamCatalyst pipeline and is kept here for completeness, but it is **not part of the current experimental evaluation**. The reasoning matches the original paper's evaluation protocol: Step 3 is where the contributions live, and refinement adds significant runtime without altering the scientific claim. The option remains available for anyone who wants to produce polished final renders.

## Novelties

The main DDS orchestration lives in `nerfstudio/dc/dc.py`; reusable novelty math lives in `nerfstudio/dc/guidance_utils.py`; settings are configured centrally in `nerfstudio/dc/tasd_config.py` (`DC_CUSTOM_PARAMS`). Every novelty defaults to *off*, so enabling it produces a clean ablation.

### Localization branch (main research direction)

| Novelty | Config | Description |
|---|---|---|
| **Self-derived relevance mask** | `gradient_mask_enabled` | Builds a soft mask `M` from the per-pixel norm of `eps_tgt − eps_src` (pre-TAG / pre-STG / pre-PN snapshot). Inspired by LatentEditor. |
| **Source-blended localization** | `source_blend_localization_enabled` | Replaces the DDS target with `eps_src + M·(eps_tgt − eps_src)`, so the edit signal vanishes outside the mask. Motivated by LatentEditor / FoI / ZONE. |
| **Outside-mask background anchor** | `outside_mask_anchor_weight` | Strengthens the preservation term by `w_out · (1 − M)`, tightening `x0` outside the mask. Conceptually aligned with RoMaP. |
| **Coverage-adaptive anchor** | `outside_mask_anchor_coverage_adaptive` | Scales `w_out` by `(1 − mean(M))`. Identity-preserving scenes (small coverage, e.g. face) keep the bg anchor tight; creative-transform scenes (large coverage, e.g. stormtrooper) loosen automatically. Lets one anchor value work across scene types. |
| **Cross-attention semantic mask** | `cross_attention_mask_enabled` + `cross_attention_mask_{layers,weight,gamma,blur}` | Aggregates target-token cross-attention from selected UNet up-blocks, fuses with the self-mask as `M_hybrid = M_self · ((1 − w) + w · M_attn)`. Target-token selection is auto-derived from src/tgt prompts (no manual keyword overrides). Based on Prompt-to-Prompt, What the DAAM, DiffEdit, LEDITS++. |
| **ψ schedule** | `psi_late_multiplier` | Temporal schedule on the preservation weight: `preserve_weight = ψ · (1 + (psi_late_multiplier − 1) · (1 − t_norm))`. Edit commits early, preservation tightens late. `=1.0` disables. Based on DaCapo (Huang et al., CVPR 2025). |
| **Latent-mean anchor (N2)** | `latent_mean_anchor_weight` | Adds `λ · (mean(tgt_x0) − mean(src_x0))` per channel onto the final gradient — penalizes VAE-latent channel-mean drift and counteracts TAG brightness/saturation artifacts directly in latent statistics. `=0.0` disables. Conceptually aligned with Piva and Stable Score Distillation. |

### TAG branch (edit strength)

| Novelty | Config | Description |
|---|---|---|
| **TAG** | `eta_tag` | Amplifies the tangential component of `noise_pred` with respect to the noisy latent. `eta_tag=1.0` disables. Based on TAG (Cho et al., 2024). |
| **Adaptive TAG** | `adaptive_tag` | Anneals `η(t) = 1 + (eta_tag − 1) · t_norm^(1/e)` so amplification is strongest at high noise and decays toward 1.0 near the clean regime. |
| **Asymmetric TAG** | `asymmetric_tag` | Applies TAG only to the target branch, leaving `eps_src` at `η = 1.0`. |
### STG branch (structural amplification)

| Novelty | Config | Description |
|---|---|---|
| **STG** | `stg_enabled`, `stg_scale`, `stg_skip_layers` | Runs a weak UNet pass via `STGIdentityValueAttnProcessor` on selected up-blocks and amplifies `eps = eps_full + s · (eps_full − eps_weak)`. Based on STG (Hyung et al., CVPR 2025). Target-branch only. |
| **STG schedule** | `stg_schedule_enabled`, `stg_schedule_mode`, `stg_schedule_{start,end}_ratio`, `stg_bump_peak_ratio` | Three shapes: `"decay"` (STG early, off late — best on identity-preserving edits), `"growth"` (STG off early, on late — lets TAG commit the edit first), `"bump"` (triangle: STG peaks mid-phase and returns to 0 before the end — prevents late STG from locking in view-dependent partial-state inconsistencies on creative edits). |
| **Coverage-adaptive STG** | `stg_coverage_adaptive` | Multiplies the scheduled STG scale by `(1 − current_coverage)` using the same-view mask built from clean pre-guidance `eps_raw`. Self-mask coverage is a proxy for scene type: small coverage (face) keeps STG near its scheduled value; large coverage (stormtrooper) fades STG toward 0 automatically, preventing it from pulling `eps_tgt` back toward the current source-image structure during creative restructuring. |

### Perp-Neg branch (creative direction separation)

| Novelty | Config | Description |
|---|---|---|
| **Perpendicular Gradient Projection** | `perp_neg`, `perp_neg_alpha` | Orthogonalizes `eps_tgt` with respect to `eps_src` via Gram-Schmidt. Kept as an optional global branch; earlier depth/cached-mask foreground variants were retired in favor of self-mask + CA-mask localization, which subsume their role. Based on PCGrad (Yu et al., NeurIPS 2020) and Perp-Neg (Armandpour et al., ICML 2023). |

### Central config

```python
# nerfstudio/dc/tasd_config.py
DC_CUSTOM_PARAMS = dict(
    # Localization
    psi=0.075,
    source_blend_localization_enabled=True,
    gradient_mask_enabled=False,
    outside_mask_anchor_weight=0.2,
    outside_mask_anchor_coverage_adaptive=True,
    gradient_mask_blur=1.0,
    gradient_mask_gamma=1.2,
    gradient_mask_ema_beta=0.0,
    gradient_mask_warmup=0,
    cross_attention_mask_enabled=True,
    cross_attention_mask_layers=[1, 2],
    cross_attention_mask_weight=0.7,
    cross_attention_mask_blur=0.5,
    cross_attention_mask_gamma=1.2,
    psi_late_multiplier=1.0,         # DaCapo-inspired ψ schedule (1.0 = off)
    latent_mean_anchor_weight=0.005, # N2: latent-mean anchor (0.0 = off)

    # TAG
    eta_tag=1.25,
    adaptive_tag=True,
    asymmetric_tag=True,

    # STG
    stg_enabled=True,
    stg_scale=2,
    stg_skip_layers=[2],
    stg_schedule_enabled=True,
    stg_schedule_start_ratio=0.4,
    stg_schedule_end_ratio=0.7,
    stg_schedule_mode="bump",        # "decay" | "growth" | "bump"
    stg_bump_peak_ratio=0.5,         # only used in "bump" mode
    stg_coverage_adaptive=True,

    # Perp-Neg (optional; earlier depth/cached-mask variants retired)
    perp_neg=False,
    perp_neg_alpha=1.0,
)
```

## Evaluation

`scripts/evaluate.py` loads the finished edited checkpoint from `<run_dir>/nerfstudio_models`, renders the evaluation views, computes metrics, and writes `metrics.json` into the same run folder. Reported metrics:

- `CLIP_direction` (↑) — editability
- `CLIP_img_sim` (↑) — content preservation
- `SSIM` (↑) — identity preservation
- `LPIPS` (↓) — perceptual distance
- `MultiView_pairwise_cos_sim` (↑) — multi-view consistency

Comparisons across runs must use the **same `downscale`** for reconstruction, edit, and (when run) refinement; inconsistent downscales have produced spurious qualitative differences in the past.

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

@inproceedings{koo2024pds,
  title     = {Posterior Distillation Sampling},
  author    = {Juil Koo and Chanho Park and Minhyuk Sung},
  booktitle = {CVPR},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.13831},
}
```
