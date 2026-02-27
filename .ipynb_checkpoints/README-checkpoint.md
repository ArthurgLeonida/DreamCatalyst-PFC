# DreamCatalyst-NS

A clean reimplementation of **DreamCatalyst** (ICLR 2025) on top of **Nerfstudio**, using **3D Gaussian Splatting** (Splatfacto) as the scene representation.

> Jiwook Kim et al. *"DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation"*. ICLR 2025. [arXiv:2407.11394](https://arxiv.org/abs/2407.11394)

---

## How it works

The `dream_catalyst_ns/` package is a **Nerfstudio plugin** — no modifications to upstream code needed.

| Component | Role |
|---|---|
| **Nerfstudio** | Training loop, data pipeline, Splatfacto model, `ns-train` CLI |
| **threestudio** | Diffusion guidance (SDS, InstructPix2Pix) |
| **dream_catalyst_ns** | Injects SDS/IP2P losses into the Splatfacto training loop |

---

## Project structure

```
DreamCatalyst-PFC/
├── scripts/
│   ├── process_data.sh             # COLMAP data processing wrapper
│   ├── train.sh                    # Training launcher
│   ├── setup.sh                    # One-shot Linux/HPC setup
│   └── pick_gpu.py                 # Auto-select least-busy GPU
├── dream_catalyst_ns/
│   ├── dream_config.py             # Registers 'dream-catalyst' with ns-train
│   └── dream_pipeline.py           # Custom pipeline (SDS + IP2P skeleton)
├── pyproject.toml
├── requirements.txt
└── data/                           # Your datasets (gitignored)
```

---

## 1. Setup (Linux / HPC)

```bash
git clone https://github.com/ArthurgLeonida/DreamCatalyst-PFC.git
cd DreamCatalyst-PFC
chmod +x setup.sh
./setup.sh            # creates conda env '3d_edit', installs everything
conda activate 3d_edit
```

`setup.sh` handles everything in order:
1. Create conda env — Python 3.10
2. Install COLMAP ≤ 3.9.1 and FFmpeg via `conda-forge`
3. Install PyTorch 2.1.2 + CUDA 11.8
4. Clone and install **nerfstudio v1.1.5**
5. Clone and install **threestudio**
6. Install `requirements.txt` + this project
7. Verify all components

> **HPC tip:** If `conda` is not found, run `module load anaconda` (or `miniconda`) first.
> If `conda activate` fails in JupyterLab, prefix it with `eval "$(conda shell.bash hook)"`.

---

## 2. Data preparation

Place your images at `data/<scene>/images/` (50–100 JPGs, walk around the object at multiple heights), then run:

```bash
# From images
bash scripts/process_data.sh chair

# From a video file
bash scripts/process_data.sh chair video
```

This calls `ns-process-data` with COLMAP and writes output to `data/chair_processed/`.

**Verify** the result:
```bash
ls data/chair_processed/transforms.json
ls data/chair_processed/images_4/
```

---

## 3. Training

```bash
bash train.sh chair              # splatfacto, 500 iters (quick test)
bash train.sh chair 30000        # splatfacto, full run
bash train.sh chair 30000 dream  # DreamCatalyst pipeline, full run
```

`train.sh` automatically selects the least-busy GPU via `scripts/pick_gpu.py`.

**Monitor** with TensorBoard:
```bash
tensorboard --logdir outputs/ --port 6006 --bind_all &
# Open http://<host>:6006
```

**Export** the trained splat:
```bash
ns-export gaussian-splat \
    --load-config outputs/chair/dream-catalyst/<DATE>/config.yml \
    --output-dir exports/chair
```

---

## Environment

| Component   | Version                                    |
|-------------|--------------------------------------------|
| OS          | Linux (HPC) / Windows 11 (local dev)       |
| GPU         | H100 80 GB (cluster) / RTX 3050 6 GB (laptop) |
| Python      | 3.10                                       |
| PyTorch     | 2.1.2+cu118                                |
| CUDA        | 11.8                                       |
| Nerfstudio  | 1.1.5 (cloned by `setup.sh`)               |
| threestudio | latest (cloned by `setup.sh`)              |
| gsplat      | 1.4.0                                      |
| diffusers   | ≥ 0.27.0, < 0.31.0                         |
| COLMAP      | ≤ 3.9.1 (installed by `setup.sh`)          |
| FFmpeg      | any recent (installed by `setup.sh`)       |

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
