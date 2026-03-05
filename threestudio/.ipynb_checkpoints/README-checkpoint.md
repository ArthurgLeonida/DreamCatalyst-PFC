# DreamCatalyst-PFC

A thesis reimplementation of **DreamCatalyst** (ICLR 2025) using **Nerfstudio** and **3D Gaussian Splatting** as the primary pipeline.

> Jiwook Kim et al. *"DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation"*. ICLR 2025. [arXiv:2407.11394](https://arxiv.org/abs/2407.11394)

---

## How it works

DreamCatalyst injects diffusion-guided editing losses (SDS / InstructPix2Pix) into an existing 3DGS scene, editing it toward a text target while preserving identity.

| Component | Role |
|---|---|
| **Nerfstudio** | Training loop, data pipeline, Splatfacto/NeRF models, `ns-train` CLI |
| **dc_nerf** (`3d_editing/`) | DreamCatalyst methods — registers `dc`, `dc_splat`, `dc_refinement`, `dc_splat_refinement` with `ns-train` |
| **diffusers + CLIP** | Diffusion guidance (SD, IP2P) |

---

## Project structure

```
DreamCatalyst-PFC/
├── setup.sh                        # One-shot environment setup (ns | gs | all)
├── scripts/
│   ├── process_data.sh             # COLMAP data processing wrapper
│   ├── train.sh                    # Training launcher
│   └── pick_gpu.py                 # Auto-select least-busy GPU
├── nerfstudio/                     # DreamCatalyst fork (dc/ + 3d_editing/ only)
│   ├── dc/                         # Base dc package
│   ├── 3d_editing/                 # dc_nerf package — registers ns-train methods
│   └── setup.py
├── threestudio/                    # GaussianEditor variant (optional)
├── pyproject.toml
└── data/                           # Your datasets (gitignored)
```

---

## 1. Setup (Linux / HPC)

```bash
git clone https://github.com/ArthurgLeonida/DreamCatalyst-PFC.git
cd DreamCatalyst-PFC
chmod +x setup.sh
bash setup.sh ns       # Nerfstudio pipeline (main thesis pipeline) ← recommended
bash setup.sh gs       # GaussianEditor/threestudio pipeline (optional)
bash setup.sh all      # Both environments
```

### `dreamcatalyst_ns` — Nerfstudio pipeline

`setup.sh ns` creates the `dreamcatalyst_ns` conda environment and installs everything in order:

1. Create conda env — Python 3.9
2. Install COLMAP ≤ 3.9.1 and FFmpeg via `conda-forge`
3. Install PyTorch 2.1.2 + CUDA 11.8 — pins `numpy==1.26.4`
4. Build and install **tinycudann** from source (NVlabs) — forces `CUDA_HOME` to conda env to avoid system nvcc mismatch; requires `setuptools<70`
5. Install upstream **nerfstudio 1.0.2**
6. Install DreamCatalyst `dc` package (`nerfstudio/`) and `dc_nerf` package (`nerfstudio/3d_editing/`) — pins `huggingface_hub<0.24` for `diffusers==0.27.2` compatibility
7. Verify all components and `ns-train` method registration

After setup, activate and verify:

```bash
conda activate dreamcatalyst_ns
ns-train --help   # should list: dc, dc_refinement, dc_splat, dc_splat_refinement
```

> **HPC tip:** If `conda activate` fails in JupyterLab, prefix with `eval "$(conda shell.bash hook)"`.

### `dreamcatalyst_gs` — GaussianEditor pipeline (optional)

`setup.sh gs` creates the `dreamcatalyst_gs` conda environment (Python 3.8, CUDA 11.7, PyTorch 2.0.1).
> ⚠️ The H100 (sm_90) is **not supported** by PyTorch 2.0.1+CUDA 11.7 — use `dreamcatalyst_ns` for H100 runs.

### Known build fixes baked into `setup.sh`

| Issue | Fix |
|---|---|
| System nvcc 12.4 vs PyTorch CUDA 11.8 | `export CUDA_HOME` to conda env before any CUDA build |
| `setuptools>=70` removed `pkg_resources` | Pin `setuptools<70` before building tinycudann |
| pip partial clone can't resolve tags | Full `git clone` of tiny-cuda-nn + local install |
| `dc`/`dc_splat` not registered in `ns-train` | `pip install -e nerfstudio/3d_editing` (not just `nerfstudio/`) |
| `cached_download` removed in `huggingface_hub>=0.24` | Pin `huggingface_hub<0.24` after `diffusers==0.27.2` |
| `glm/glm.hpp` missing (gs env) | `conda install -c conda-forge glm` |
| `cub/cub.cuh`, `thrust/complex.h` missing (gs env) | Install full `cuda-toolkit` + `cuda-libraries-dev` |

---

## 2. Data Preparation

Choose one of the options below.

### Option A — Mip-NeRF 360 (Real-World Scenes)
Real-world scenes (garden, kitchen, room, counter, bicycle, etc.) captured with a DSLR. Good for benchmarking against published results.

```bash
cd data/
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip
```

### Option B — NeRF Synthetic (Blender Scenes)
8 synthetic objects (lego, chair, hotdog, drums, etc.) rendered at 800×800 — no COLMAP needed.

```bash
ns-download-data blender --save-dir data/blender
```

### Option C — Custom Images
Place images at `data/<scene>/images/` (50–100 JPGs, multi-height orbit), then run:

```bash
bash scripts/process_data.sh chair          # from image folder
bash scripts/process_data.sh chair video    # from video file
```

---

## 3. Training

### Step 1 — Reconstruct scene with Splatfacto

```bash
conda activate dreamcatalyst_ns
ns-train splatfacto --data data/<scene>
```

### Step 2 — Edit with DreamCatalyst

```bash
# 3DGS editing (recommended)
ns-train dc_splat     --data data/<scene>     --load-dir outputs/<scene>/splatfacto/<timestamp>/nerfstudio_models/     --pipeline.dc.src_prompt "a man wearing a white shirt"     --pipeline.dc.tgt_prompt "a man wearing iron man armor"

# NeRF editing
ns-train dc     --data data/<scene>     --load-dir outputs/<scene>/nerfacto/<timestamp>/nerfstudio_models/     --pipeline.dc.src_prompt "source description"     --pipeline.dc.tgt_prompt "target description"

# Refinement pass (after dc_splat)
ns-train dc_splat_refinement     --data data/<scene>     --load-dir outputs/<scene>/dc_splat/<timestamp>/nerfstudio_models/
```

**Monitor** with TensorBoard:
```bash
tensorboard --logdir outputs/ --port 6006 --bind_all &
# Open http://<host>:6006
```

---

## Environment

| Component | Version |
|---|---|
| OS | Linux (HPC) |
| GPU | H100 80 GB HBM3 (cluster) |
| Python | 3.9 |
| PyTorch | 2.1.2+cu118 |
| CUDA | 11.8 |
| nerfstudio | 1.0.2 |
| dc_nerf | 1.0.0 |
| gsplat | 0.1.6 |
| tinycudann | latest (built from source) |
| diffusers | 0.27.2 |
| huggingface_hub | 0.23.x |
| numpy | 1.26.4 |
| COLMAP | ≤ 3.9.1 |
| FFmpeg | any recent |

---

## BibTeX

```bibtex
@inproceedings{kim2025dreamcatalyst,
  title     = {DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation},
  author    = {Jiwook Kim and Seonho Lee and Jaeyo Shin and Jiho Choi and Hyunjung Shim},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2407.11394},
}
```