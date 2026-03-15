# DreamCatalyst-PFC

Text-driven 3D scene editing using **3D Gaussian Splatting** + **DreamCatalyst** (DDS guidance).

> Kim et al. *"DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation"*. ICLR 2025. [arXiv:2407.11394](https://arxiv.org/abs/2407.11394)

## Pipeline overview

```
Photos/Video ──► COLMAP ──► Splatfacto (3DGS) ──► DreamCatalyst (edit) ──► Refinement (SDEdit)
                  Step 1        Step 2                  Step 3                  Step 4
```

## Setup

```bash
bash setup.sh ns        # creates conda env 'dreamcatalyst_ns', installs everything
conda activate dreamcatalyst_ns
```

## Step 1 — Process data (COLMAP)

```bash
# From images (place them in data/<scene>/images/)
bash scripts/process_data.sh <scene>

# From video
bash scripts/process_data.sh <scene> video
```

Output: `data/<scene>_processed/` with `transforms.json`.

## Step 2 — Train 3DGS reconstruction

```bash
bash scripts/train.sh <scene> 30000
# equivalent to: ns-train splatfacto ... nerfstudio-data --data data/<scene>_processed
```

Output: `outputs/<scene>/splatfacto/<timestamp>/`

## Step 3 — Edit with DreamCatalyst (DDS)

```bash
bash scripts/edit.sh bicycle \
    "a photo of a bicycle leaning against a bench" \
    "a photo of a motorcycle leaning against a bench" \
    outputs/bicycle/splatfacto/<timestamp>/nerfstudio_models/ \
    3000
# Usage: bash scripts/edit.sh <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep]
# rep: splat (default) or nerf
# Default: 3000 iterations
```

Loads the reconstruction from Step 2 and optimizes the Gaussians toward the target prompt using DDS guidance. Each iteration renders a view, computes guidance loss, and backprops into the splat parameters.

The `max_iters` argument is synced to `--pipeline.dc.max-iteration` so the timestep schedule covers the full training range.

**Prompt guidelines:**
- Describe the **full scene**, not just the object: `"a photo of a bicycle leaning against a bench"` not `"a bicycle"`.
- Keep the source and target prompts as similar as possible — only change the edited element.
- Prefix with `"a photo of"` to anchor the diffusion model to photorealistic outputs.

Output: `outputs/<scene>/dc_splat/<timestamp>/`

### Render after editing (check before refinement)

```bash
ns-render interpolate \
    --load-config outputs/<scene>/dc_splat/<timestamp>/config.yml \
    --output-path renders/<scene>_edited.mp4
```

## Step 4 — Refinement

```bash
bash scripts/refine.sh bicycle \
    "a photo of a motorcycle leaning against a bench" \
    outputs/bicycle/dc_splat/<timestamp>/nerfstudio_models/ \
    30000
# Usage: bash scripts/refine.sh <scene> <tgt_prompt> <load_dir> [max_iters]
# Default: 30000 iterations
```

Uses SDEdit to produce edited 2D images, then retrains the Gaussians against them. Cleans up floater artifacts from Step 3. **Do not skip this step** — it significantly improves output quality.

### Render after refinement

```bash
ns-render interpolate \
    --load-config outputs/<scene>/dc_splat_refinement/<timestamp>/config.yml \
    --output-path renders/<scene>_refined.mp4
```

## Export & monitor

```bash
ns-export gaussian-splat \
    --load-config outputs/<scene>/dc_splat_refinement/<timestamp>/config.yml \
    --output-dir exports/<scene>

# Monitor training in real time
tensorboard --logdir outputs/ --port 6006 --bind_all
```

## Fixes over original DreamCatalyst repo

The [original repo](https://github.com/kaist-cvml/DreamCatalyst) ships `runwayml/stable-diffusion-v1-5` in `DCConfig`, but the `__call__` method constructs 8-channel UNet inputs with 3-way CFG — this is the InstructPix2Pix architecture, not SD 1.5 (4-channel, 2-way CFG). Running the original config as-is crashes.

| Fix | File | Description |
|---|---|---|
| **Model path** | `dc.py` DCConfig | Changed to `timbrooks/instruct-pix2pix` to match the 8-channel UNet input the code actually builds. |
| **`run_sdedit` channel mismatch** | `dc.py` | The original `run_sdedit` passes 4-channel input (`[xt]*2`) to the UNet, but IP2P expects 8 channels. Added `image_cond` parameter and concatenates it to produce `[B, 8, H, W]` input. |
| **Refinement pipeline** | `refinement_pipeline.py` | Now computes and passes `image_cond` via `encode_src_image()` when calling `run_sdedit`. Without this, Step 4 crashes on the first SDEdit call. |
| **Hardcoded `max_iteration`** | `dc.py` DCConfig | Was hardcoded to 3000 in `__init__`. Now a config field (`max_iteration`) so the timestep curriculum schedule syncs with the actual `--max-num-iterations` value. |
| **`edit.sh` iteration sync** | `scripts/edit.sh` | Passes `--pipeline.dc.max-iteration` matching `--max-num-iterations` so shorter/longer runs don't break the timestep schedule. |

## Novelties

This project extends DreamCatalyst's DDS guidance with modifications to the noise prediction step. All are configured in `nerfstudio/dc/tasd_config.py` and applied to every method config automatically.

| # | Novelty | Config | Description | Status |
|---|---|---|---|---|
| 1 | **TAG** | `eta_tag=1.15` | Amplifies the tangential component of the noise prediction relative to the noisy latent, improving detail and reducing oversaturation. Based on TAG (Cho et al., 2024). `eta_tag=1.0` disables it. | Done |
| 2 | **Adaptive TAG** | `adaptive_tag=True` | Anneals η from `eta_tag` at high noise to 1.0 at low noise: `η(t) = 1 + (eta_tag - 1) * t_normalized`. Stronger amplification when the signal is noisiest, tapering off as denoising progresses. Original contribution. | Done |
| 3 | **Asymmetric TAG** | `asymmetric_tag=True` | Applies TAG only to the target branch of DDS, leaving the source branch unmodified (`η=1.0`). This amplifies the editing direction without disturbing source reconstruction. Original contribution. | Done |
| 4 | **STG** | `stg_enabled=True` | Runs a second "weak" UNet pass with self-attention zeroed out in selected up_blocks, then blends: `eps = eps_weak + stg_scale * (eps_full - eps_weak)`. Applied only to target branch. Based on STG (Hyung et al., CVPR 2025). | Done |
| 5 | **Conflict-Free Guidance** | `conflict_free=True` | Projects out the component of `eps_tgt` parallel to `eps_src` before the DDS delta, making the two guidance signals orthogonal. Based on Devil in Detail (Jo et al., CVPR 2025). | Done |

```python
# nerfstudio/dc/tasd_config.py
DC_CUSTOM_PARAMS = dict(
    eta_tag=1.15,         # 1.0 = disabled
    adaptive_tag=True,    # anneal η with timestep
    asymmetric_tag=True,  # TAG only on target branch
    conflict_free=False,  # project out conflicting components
    stg_enabled=False,    # self-attention skip guidance
    stg_scale=1.0,        # STG blend strength
    stg_skip_layers=[1, 2],  # which up_blocks to skip
)
```

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
  title     = {Self-Guidance: Improve Deep Diffusion Model via Self-Guidance},
  author    = {Minyoung Hyung and Jaegul Choo},
  booktitle = {CVPR},
  year      = {2025},
}

@inproceedings{jo2025devil,
  title     = {Devil in the Details: Towards Conflict-Free Guidance for Image Editing},
  author    = {Seonho Jo and Jaegul Choo},
  booktitle = {CVPR},
  year      = {2025},
}
```