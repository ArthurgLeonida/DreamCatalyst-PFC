# DreamCatalyst-NS

A clean reimplementation of **DreamCatalyst** (ICLR 2025) on top of Nerfstudio, focusing on **3D Gaussian Splatting** (Splatfacto) as the scene representation. The original DreamCatalyst already uses Nerfstudio — this project rebuilds it cleanly with modern dependencies and a GS-first approach, with a path toward a custom architecture.

## Project Structure

```
PFC_3D_Edit/
├── pyproject.toml                       # Package config + Nerfstudio entry-point
├── README.md
├── data/                                # Your datasets go here
│   └── chair/
│       └── images/                      # Raw photos of the chair
│           ├── frame_00001.jpg
│           ├── frame_00002.jpg
│           └── ...
└── dream_catalyst_ns/                   # Python package
    ├── __init__.py
    ├── dream_config.py                  # MethodSpecification (registers with ns-train)
    └── dream_pipeline.py                # Custom Pipeline (skeleton)
```

---

## 1. Installation

### Prerequisites (one-time, on the machine with the RTX 3050)

```powershell
# 1. Create a fresh conda environment with Python 3.10
conda create -n 3d_edit python=3.10 -y
conda activate 3d_edit

# 2. Install PyTorch with CUDA 11.8 (works with RTX 3050)
#    See https://pytorch.org/get-started/locally/ for other CUDA versions.
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3. Verify CUDA works
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True

# 4. Install Nerfstudio
pip install nerfstudio
```

> **No GPU on this machine?** Steps 1-3 still work — PyTorch installs fine,
> it just won't detect a GPU. You can write/edit code here and run training
> later on the laptop with the RTX 3050.

### Install this project

```powershell
conda activate 3d_edit
cd E:\Documentos_HD\Estudo\UFSC\9FASE\PFC\PFC_3D_Edit

# Install the package in editable mode
pip install -e .

# Re-generate CLI tab-completion (optional but nice)
ns-install-cli
```

After this, `ns-train --help` should list **`dream-catalyst`** as an available method.

---

## 2. Data Processing (Chair Dataset)

### 2a. Place your images

Put your photos in a folder. The recommended layout is:

```
data/chair/images/
├── IMG_0001.jpg
├── IMG_0002.jpg
├── ...
```

### 2b. Install COLMAP

```powershell
conda install -c conda-forge colmap
# Verify:
colmap -h
```

### 2c. Run `ns-process-data`

**Option A – From a folder of images (COLMAP):**

```powershell
ns-process-data images --data data/chair/images --output-dir data/chair_processed
```

This will:
1. Run COLMAP feature extraction + matching + sparse reconstruction
2. Generate `data/chair_processed/transforms.json` with camera poses
3. Copy/symlink images into `data/chair_processed/images/`

**Option B – From a Polycam export (.zip):**

If you captured with Polycam on an iPhone with LiDAR:

```powershell
ns-process-data polycam --data data/chair/polycam_export.zip --output-dir data/chair_processed
```

**Option C – From a video:**

```powershell
ns-process-data video --data data/chair/video.mp4 --output-dir data/chair_processed
```

### Tips for better results
- **50-100 images** is a good range for a single object
- Ensure significant overlap between views (walk around the object)
- Good, even lighting; avoid motion blur
- COLMAP can be slow – be patient on the first run

---

## 3. Training

### 3a. Verify the base Splatfacto model works

Test that your processed data is correct with vanilla Gaussian Splatting:

```powershell
ns-train splatfacto --data data/chair_processed
```

### 3b. Run with DreamCatalyst pipeline

```powershell
ns-train dream-catalyst --data data/chair_processed
```

You should see `[DreamCatalyst] Pipeline initialised – custom code is running!` in the console, and every 100 steps: `[DreamCatalyst] Training step ...`.

---

## 4. Next Steps (Implementing the real logic)

1. **Load InstructPix2Pix** in `DreamCatalystPipeline.__init__`:
   ```python
   from diffusers import StableDiffusionInstructPix2PixPipeline
   self.ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
       "timbrooks/instruct-pix2pix", torch_dtype=torch.float16,
       safety_checker=None,
   ).to("cuda")
   self.ip2p.set_progress_bar_config(disable=True)
   ```

2. **Render full images** — Splatfacto already does this! `model_outputs["rgb"]` is a full image.

3. **Compute SDS loss** in `get_train_loss_dict` using the rendered Gaussian Splat image + the diffusion model.

4. **Add the DreamCatalyst modifications**: creative catalyst sampling, optimised noise schedules, etc.

5. **Future**: Design a custom architecture with your professor — the pipeline is ready to be extended.

---

## Environment

| Component | Version |
|-----------|---------|
| OS        | Windows 11 |
| GPU       | RTX 3050 (8 GB VRAM) |
| Python    | 3.10 |
| PyTorch   | 2.1.2+cu118 |
| CUDA      | 11.8 (via PyTorch) |
| Nerfstudio| latest (pip) |
