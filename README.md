# DreamCatalyst-NS

A clean reimplementation of **DreamCatalyst** (ICLR 2025) on top of Nerfstudio, focusing on **3D Gaussian Splatting** (Splatfacto) as the scene representation. The original DreamCatalyst already uses Nerfstudio — this project rebuilds it cleanly with modern dependencies and a GS-first approach, with a path toward a custom architecture.

## Project Structure

```
DreamCatalyst-PFC/
├── pyproject.toml                       # Package config + Nerfstudio entry-point
├── README.md
├── data/                                # Your datasets go here (git-ignored)
│   └── chair/
│       ├── images/                      # Raw + COLMAP-copied images
│       ├── images_2/                    # 2× downscaled
│       ├── images_4/                    # 4× downscaled
│       ├── images_8/                    # 8× downscaled
│       ├── colmap/                      # COLMAP sparse reconstruction
│       ├── transforms.json              # Camera poses (Nerfstudio format)
│       └── sparse_pc.ply               # Sparse point cloud
└── dream_catalyst_ns/                   # Python package
    ├── __init__.py
    ├── dream_config.py                  # MethodSpecification (registers with ns-train)
    └── dream_pipeline.py                # Custom Pipeline
```

---

## 1. Installation

### 1a. Create conda environment

```powershell
conda create -n 3d_edit python=3.10 -y
conda activate 3d_edit
```

### 1b. Install PyTorch with CUDA 11.8

```powershell
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True
```

### 1c. Install Nerfstudio

```powershell
pip install nerfstudio
```

### 1d. Install FFmpeg (system binary)

FFmpeg is required by `ns-process-data` for image metadata extraction.

1. Download from <https://www.gyan.dev/ffmpeg/builds/> — get **`ffmpeg-release-essentials.zip`**
2. Extract to a permanent location (e.g. `C:\ffmpeg\`)
3. Add the `bin\` folder to your **PATH** environment variable:
   - Win + S → "Environment Variables" → User variables → Path → New →
     `C:\ffmpeg\ffmpeg-X.X-essentials_build\bin`
4. Restart your terminal and verify:

```powershell
ffmpeg -version
```

### 1e. Install COLMAP 3.9.1 (system binary)

> ⚠️ **COLMAP ≥ 3.13 is NOT compatible** with Nerfstudio 1.1.5
> (the `--SiftExtraction.use_gpu` flag was removed). Use **COLMAP 3.9.1**.

1. Download **COLMAP 3.9.1** (CUDA) from [GitHub Releases](https://github.com/colmap/colmap/releases/tag/3.9.1) — get `COLMAP-3.9.1-windows-cuda.zip`
2. Extract to a permanent location (e.g. `C:\colmap\`)
3. Add the folder to your **PATH** (same process as FFmpeg above)
4. Restart your terminal and verify:

```powershell
colmap help
# Should show: COLMAP 3.9.1
```

### 1f. Install this project

```powershell
conda activate 3d_edit
cd <path-to-this-repo>

# Install in editable mode
pip install -e .

# Re-generate CLI tab-completion (optional)
ns-install-cli
```

After this, `ns-train --help` should list **`dream-catalyst`** as an available method.

---

## 2. Data Processing

### 2a. Place your images

Put your photos in a folder:

```
data/chair/images/
├── IMG_0001.jpg
├── IMG_0002.jpg
├── ...
```

**Tips for good captures:**
- **50–100 images** is a good range for a single object
- Walk around the object; ensure significant overlap between views
- Capture from multiple heights (eye level, low angle, slightly above)
- Good, even lighting; avoid motion blur

### 2b. Run `ns-process-data`

**From a folder of images (COLMAP):**

```powershell
ns-process-data images --data data/chair/images --output-dir data/chair
```

This will:
1. Run COLMAP feature extraction + matching + sparse reconstruction
2. Generate `data/chair/transforms.json` with camera poses
3. Create downscaled image copies (`images_2/`, `images_4/`, `images_8/`)
4. Generate `sparse_pc.ply` (initial point cloud for Splatfacto)

> ⏱️ COLMAP can take 30–60 minutes depending on image count and resolution.

**From a video:**

```powershell
ns-process-data video --data data/chair/video.mp4 --output-dir data/chair
```

### 2c. Verify the processed dataset

After processing, run this quick check:

```powershell
python -c "import json; d=json.load(open('data/chair/transforms.json')); print('Frames:', len(d['frames'])); print('Resolution: %dx%d' % (d['w'], d['h']))"
```

Also confirm these files/folders exist:
- `data/chair/transforms.json`
- `data/chair/sparse_pc.ply`
- `data/chair/images_2/`, `images_4/`, `images_8/`

You should see a frame count close to your original image count.

---

## 3. Training

### 3a. Verify with vanilla Splatfacto

Test that the data is correct before using the custom pipeline:

```powershell
ns-train splatfacto --data data/chair --max-num-iterations 500
```

If cameras and images load without errors, the data is good.

### 3b. Run with DreamCatalyst pipeline

```powershell
ns-train dream-catalyst --data data/chair
```

---

## Environment

| Component  | Version / Notes                                          |
|------------|----------------------------------------------------------|
| OS         | Windows 11                                               |
| GPU        | NVIDIA RTX 3050 (8 GB VRAM) / H100 (university cluster)  |
| Python     | 3.10                                                     |
| PyTorch    | 2.1.2+cu118                                              |
| CUDA       | 11.8 (via PyTorch wheels)                                |
| Nerfstudio | 1.1.5                                                    |
| gsplat     | 1.4.0                                                    |
| diffusers  | 0.36.0                                                   |
| COLMAP     | 3.9.1 (system binary)                                    |
| FFmpeg     | system binary                                            |
