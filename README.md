# DreamCatalyst-NS

A clean reimplementation of **DreamCatalyst** (ICLR 2025) on top of Nerfstudio, focusing on **3D Gaussian Splatting** (Splatfacto) as the scene representation. The original DreamCatalyst already uses Nerfstudio — this project rebuilds it cleanly with modern dependencies and a GS-first approach, with a path toward a custom architecture.

> **Citation**: This project is based on the methodology from:
> 
> Jiwook Kim, Seonho Lee, Jaeyo Shin, Jiho Choi, Hyunjung Shim. *"DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation"*. ICLR 2025. [arXiv:2407.11394](https://arxiv.org/abs/2407.11394)

## Project Structure

```
DreamCatalyst-PFC/
├── pyproject.toml                       # Package config + Nerfstudio entry-point
├── README.md
├── requirements.txt                     # Pip dependencies
├── setup.sh                             # One-shot Linux/HPC setup
├── train.sh                             # Training launcher
├── scripts/
│   └── process_data.sh                  # COLMAP data processing wrapper
│
├── dream_catalyst_ns/                   # ── Custom package ──
│   ├── __init__.py
│   ├── dream_config.py                  # MethodSpecification (registers with ns-train)
│   └── dream_pipeline.py                # Custom Pipeline (SDS + IP2P)
│
├── nerfstudio/                          # ── Embedded Nerfstudio v1.1.5 (trimmed) ──
│   ├── pyproject.toml
│   └── nerfstudio/
│       ├── cameras/                     # Camera models & optimizers
│       ├── configs/                     # Method & dataparser configs (Splatfacto only)
│       ├── data/                        # Dataparsers, datamanagers, datasets
│       ├── engine/                      # Trainer, optimizers, schedulers, callbacks
│       ├── models/                      # base_model + splatfacto only
│       ├── model_components/            # Losses, bilateral grid, renderers
│       ├── pipelines/                   # VanillaPipeline + DynamicBatch
│       ├── process_data/                # COLMAP utilities (for dataparsers)
│       └── utils/                       # Math, profiling, rich, writer, etc.
│
├── threestudio/                         # ── Embedded threestudio v0.2.3 (trimmed) ──
│   ├── setup.py
│   ├── launch.py
│   └── threestudio/
│       ├── models/
│       │   ├── guidance/                # ⭐ SDS, InstructPix2Pix, SDI guidance
│       │   └── prompt_processors/       # Text prompt encoding
│       ├── systems/                     # DreamFusion, InstructNeRF2NeRF, SDI systems
│       ├── data/                        # Camera sampling, multiview datasets
│       └── utils/                       # Config, ops, loss, typing, etc.
│
└── data/                                # Your datasets (git-ignored)
    └── chair/
        ├── images/                      # Raw images
        └── ...                          # COLMAP outputs, downscaled copies
```

### What was trimmed

Both `nerfstudio/` and `threestudio/` are embedded as **local editable packages**, stripped down to only the modules needed for DreamCatalyst:

| Removed from Nerfstudio | Reason |
|--------------------------|--------|
| `viewer/`, `viewer_legacy/` | Interactive viewer — not needed for headless training |
| `exporter/` | Mesh/point cloud export |
| `scripts/`, `plugins/` | CLI entrypoints — we use our own |
| `generative/` | Built-in SD wrappers — threestudio handles this |
| `fields/`, `field_components/` | Neural fields — Splatfacto uses explicit Gaussians |
| All NeRF models | Only `splatfacto.py` + `base_model.py` kept |
| All dataparsers except `nerfstudio` + `colmap` | Unused dataset formats |
| `docs/`, `tests/`, `colab/` | Development artifacts |

| Removed from threestudio | Reason |
|---------------------------|--------|
| `models/geometry/`, `renderers/`, `materials/`, `background/` | Nerfstudio handles 3D representation |
| `models/exporters/` | Not needed |
| Most guidance modules | Only `stable_diffusion`, `instructpix2pix`, `sdi` kept |
| Most systems | Only `base`, `dreamfusion`, `instructnerf2nerf`, `sdi` kept |
| `extern/`, `load/`, `custom/` | Zero123 / demo data |
| `docker/`, `docs/`, notebooks, `gradio_app.py` | Development artifacts |
| Most YAML configs | Only `dreamfusion-sd.yaml`, `instructnerf2nerf.yaml`, `sdi.yaml` kept |

---

## 1. Installation (Linux / HPC — recommended)

### Quick start (one command)

```bash
git clone https://github.com/ArthurgLeonida/DreamCatalyst-PFC.git
cd DreamCatalyst-PFC
chmod +x setup.sh
./setup.sh          # creates conda env '3d_edit', installs everything
conda activate 3d_edit
```

### Manual installation

#### 1a. Create conda environment

```bash
conda create -n 3d_edit python=3.10 -y
conda activate 3d_edit
```

#### 1b. Install PyTorch with CUDA 11.8

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

#### 1c. Install local frameworks

```bash
# Install the embedded nerfstudio (editable)
pip install -e ./nerfstudio

# Install the embedded threestudio (editable)
pip install -e ./threestudio

# Install remaining dependencies
pip install -r requirements.txt
```

#### 1d. Install this project

```bash
pip install -e .

# Verify dream-catalyst is registered
ns-train --help | grep dream-catalyst
```

#### 1e. System dependencies

- **COLMAP** — `apt install colmap` or `module load colmap` (version ≤ 3.9.1 for NS 1.1.5)
- **FFmpeg** — `apt install ffmpeg`

---

## 2. Data Processing

### 2a. Place your images

```
data/chair/images/
├── IMG_0001.jpg
├── IMG_0002.jpg
├── ...
```

**Tips**: 50–100 images, walk around the object, multiple heights, good lighting.

### 2b. Run `ns-process-data`

```bash
# From images (COLMAP):
ns-process-data images --data data/chair/images --output-dir data/chair

# From video:
ns-process-data video --data data/chair/video.mp4 --output-dir data/chair
```

Or use the wrapper script:
```bash
bash scripts/process_data.sh chair
```

### 2c. Verify

```bash
python -c "import json; d=json.load(open('data/chair/transforms.json')); print('Frames:', len(d['frames']))"
```

---

## 3. Training

### 3a. Verify with vanilla Splatfacto

```bash
ns-train splatfacto --data data/chair --max-num-iterations 500 --vis tensorboard
```

With downscaling:
```bash
ns-train splatfacto --max-num-iterations 500 --vis tensorboard \
    nerfstudio-data --data data/chair --downscale-factor 4
```

### 3b. Run with DreamCatalyst pipeline

```bash
ns-train dream-catalyst --data data/chair --vis tensorboard
```

Or use the training script:
```bash
bash train.sh chair              # 500 iters, splatfacto
bash train.sh chair 30000 dream  # 30k iters, dream-catalyst
```

### 3c. View results

```bash
tensorboard --logdir outputs/
```

---

## Environment

| Component    | Version / Notes                                          |
|--------------|----------------------------------------------------------|
| OS           | Linux (HPC) / Windows 11 (local dev)                     |
| GPU          | H100 80 GB (cluster) / RTX 3050 6 GB (local)             |
| Python       | 3.10                                                     |
| PyTorch      | 2.1.2+cu118                                              |
| CUDA         | 11.8 (via PyTorch wheels)                                |
| Nerfstudio   | 1.1.5 (local, trimmed)                                   |
| threestudio  | 0.2.3 (local, trimmed)                                   |
| gsplat       | 1.4.0                                                    |
| diffusers    | ≥ 0.36.0                                                 |
| COLMAP       | ≤ 3.9.1 (system binary)                                  |
| FFmpeg       | any recent version (system binary)                       |

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
