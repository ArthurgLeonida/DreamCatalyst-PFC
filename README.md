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
ns-train dc_splat \
    --max-num-iterations 3000 \
    --pipeline.dc.src-prompt "a photo of a bicycle leaning against a bench" \
    --pipeline.dc.tgt-prompt "a photo of a motorcycle leaning against a bench" \
    --load-dir outputs/<scene>/splatfacto/<timestamp>/nerfstudio_models/ \
    nerfstudio-data --data data/<scene>_processed
```

This loads the reconstruction from Step 2 and optimizes the Gaussians toward the target prompt using DDS guidance. Each iteration renders a view, computes guidance loss, and backprops into the splat parameters.

Output: `outputs/<scene>/dc_splat/<timestamp>/`

## Step 4 — Refinement (optional)

```bash
ns-train dc_splat_refinement \
    --max-num-iterations 30000 \
    --pipeline.dc.tgt-prompt "a photo of a motorcycle leaning against a bench" \
    --load-dir outputs/<scene>/dc_splat/<timestamp>/nerfstudio_models/ \
    nerfstudio-data --data data/<scene>_processed
```

Uses SDEdit to produce edited 2D images, then retrains the Gaussians against them. Cleans up artifacts from Step 3.

## Export & view

```bash
ns-export gaussian-splat \
    --load-config outputs/<scene>/dc_splat/<timestamp>/config.yml \
    --output-dir exports/<scene>

# Monitor training
tensorboard --logdir outputs/ --port 6006 --bind_all
```

## Novelties

This project extends DreamCatalyst's DDS guidance with modifications to the noise prediction step. All are configured in `nerfstudio/dc/tasd_config.py` and applied to every method config automatically.

| # | Novelty | Config | Description | Status |
|---|---|---|---|---|
| 1 | **TAG** | `eta_tag=1.15` | Amplifies the tangential component of the noise prediction relative to the noisy latent, improving detail and reducing oversaturation. Based on TAG (Cho et al., 2024). `eta_tag=1.0` disables it. | Done |
| 2 | **Adaptive TAG** | `adaptive_tag=True` | Anneals η from `eta_tag` at high noise to 1.0 at low noise: `η(t) = 1 + (eta_tag - 1) * t_normalized`. Stronger amplification when the signal is noisiest, tapering off as denoising progresses. Original contribution. | Done |
| 3 | **Asymmetric TAG** | `asymmetric_tag=True` | Applies TAG only to the target branch of DDS, leaving the source branch unmodified (`η=1.0`). This amplifies the editing direction without disturbing source reconstruction. Original contribution. | Done |
| 4 | **STG** | `stg_enabled` | Skips self-attention in UNet up_blocks for classifier-free guidance without a negative prompt. Based on STG (Hyung et al., CVPR 2025). | TODO |
| 5 | **Conflict-Free Guidance** | `conflict_free` | Projects out conflicting components between text and image guidance vectors. Based on Devil in Detail (Jo et al., CVPR 2025). | TODO |

```python
# nerfstudio/dc/tasd_config.py
DC_CUSTOM_PARAMS = dict(
    eta_tag=1.15,         # 1.0 = disabled
    adaptive_tag=True,    # anneal η with timestep
    asymmetric_tag=True,  # TAG only on target branch
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