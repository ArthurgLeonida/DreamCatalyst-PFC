#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-PFC — Environment Setup Script
# ==============================================================================
#  Usage:
#    chmod +x setup.sh
#    bash setup.sh [ns|gs|all]   (default: ns)
#
#  ns  → dreamcatalyst_ns  (Nerfstudio, Python 3.9, CUDA 11.8) ← thesis pipeline
#  gs  → dreamcatalyst_gs  (GaussianEditor, Python 3.8, CUDA 11.7) ← optional
#  all → both environments
#
#  Prerequisites:
#    - conda available in PATH
#    - CUDA drivers ≥ 11.8 (check: nvidia-smi)
#    - Internet access (git clone, pip, conda)
# ==============================================================================

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
TARGET="${1:-ns}"
if [[ "$TARGET" != "ns" && "$TARGET" != "gs" && "$TARGET" != "all" ]]; then
    echo "Usage: bash setup.sh [ns|gs|all]"
    exit 1
fi

# ── Resolve project root ─────────────────────────────────────────────────────
# pwd -P resolves bind-mounts to their real path (e.g. /lapix → /home/jovyan).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${SCRIPT_DIR}"

if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found at ${PROJECT_ROOT}"
    echo "       Run this script from the project root."
    exit 1
fi

NERFSTUDIO_DIR="${PROJECT_ROOT}/nerfstudio"
THREESTUDIO_DIR="${PROJECT_ROOT}/threestudio"

echo "============================================================"
echo "  DreamCatalyst-PFC — Environment Setup"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Target       : ${TARGET}"
echo "============================================================"

verify_import() {
    local module="$1"
    python -c "
import importlib, sys
m = importlib.import_module('${module}')
f = getattr(m, '__file__', None)
if f is None:
    print('  ✗  ${module}.__file__ is None — ghost namespace package!')
    sys.exit(1)
print(f'  ✓  ${module}: {f}')
"
}

# ==============================================================================
#  ENVIRONMENT 1 — dreamcatalyst_ns
#  Nerfstudio-based | Python 3.9 | CUDA 11.8 | gsplat 0.1.6
#  Main thesis pipeline: COLMAP → splatfacto → DreamCatalyst
# ==============================================================================
setup_ns() {
    echo ""
    echo "============================================================"
    echo "  [ENV 1/2] dreamcatalyst_ns  (Nerfstudio pipeline)"
    echo "============================================================"

    # ── Check DreamCatalyst fork ──────────────────────────────────────────────
    if [ ! -d "${NERFSTUDIO_DIR}/3d_editing" ]; then
        echo "  nerfstudio/3d_editing not found — re-cloning DreamCatalyst fork..."
        rm -rf "${NERFSTUDIO_DIR}"
        git clone --depth 1 https://github.com/kaist-cvml/DreamCatalyst.git /tmp/_dc_src
        cp -r /tmp/_dc_src/nerfstudio "${NERFSTUDIO_DIR}"
        rm -rf /tmp/_dc_src
        echo "  ✓  DreamCatalyst nerfstudio fork cloned."
    else
        echo "  ✓  DreamCatalyst fork detected (3d_editing/ present)."
    fi

    # ── 1. Create conda env ───────────────────────────────────────────────────
    echo ""
    echo "[1/7] Creating conda env 'dreamcatalyst_ns' (Python 3.9)..."
    if conda info --envs | grep -q "^dreamcatalyst_ns "; then
        echo "  Already exists, skipping creation."
    else
        conda create -n dreamcatalyst_ns python=3.9 -y
    fi

    eval "$(conda shell.bash hook)"
    conda activate dreamcatalyst_ns
    SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"

    echo "  Python : $(python --version)"
    echo "  Site   : ${SITE}"

    # ── 2. COLMAP + FFmpeg ────────────────────────────────────────────────────
    echo ""
    echo "[2/7] Installing COLMAP ≤3.9.1 and FFmpeg via conda-forge..."
    conda install -y -c conda-forge "colmap<=3.9.1" ffmpeg

    colmap_ver=$(colmap -h 2>&1 | grep -oP 'COLMAP \K[0-9.]+' | head -1 || echo 'NOT FOUND')
    ffmpeg_ver=$(ffmpeg -version 2>&1 | grep -oP 'version \K[^ ]+' | head -1 || echo 'NOT FOUND')
    echo "  COLMAP : ${colmap_ver}"
    echo "  FFmpeg : ${ffmpeg_ver}"

    # ── 3. PyTorch 2.1.2 + CUDA 11.8 + NumPy pin ─────────────────────────────
    echo ""
    echo "[3/7] Installing PyTorch 2.1.2+cu118..."
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118

    conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

    # Pin NumPy <2 — PyTorch 2.1.2 was compiled against NumPy 1.x
    pip install "numpy<2"

    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  ✓  PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPU: {torch.cuda.get_device_name(0)}')
"

    # ── 4. tinycudann ─────────────────────────────────────────────────────────
    # - setuptools>=70 removed pkg_resources — pin before build.
    # - System nvcc may differ from CUDA 11.8 — force conda's nvcc via CUDA_HOME.
    # - Build AND import verify run inside the subshell: LD_LIBRARY_PATH is
    #   needed at import time too (.so resolution), not just at compile time.
    # - Subshell prevents CUDA_HOME / LD_LIBRARY_PATH from leaking into the
    #   parent shell (which would break git SSH and other tools after setup).
    echo ""
    echo "[4/7] Installing tinycudann from source (NVlabs)..."
    echo "      This will take 5–15 minutes. Please wait..."
    pip install "setuptools<70" wheel ninja

    (
        export CUDA_HOME="/root/anaconda3/envs/dreamcatalyst_ns"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
        echo "  nvcc: $(nvcc --version 2>&1 | grep release)"
    
        rm -rf /tmp/tiny-cuda-nn
        git clone https://github.com/NVlabs/tiny-cuda-nn.git /tmp/tiny-cuda-nn
        cd /tmp/tiny-cuda-nn
        git submodule update --init --recursive
        pip install --no-build-isolation ./bindings/torch
        rm -rf /tmp/tiny-cuda-nn
    )
    # Verify outside subshell — torch uses its own bundled MKL cleanly
    python -c "import tinycudann; print('  ✓  tinycudann OK')"

    # ── 5. Upstream nerfstudio 1.0.2 ─────────────────────────────────────────
    echo ""
    echo "[5/7] Installing upstream nerfstudio 1.0.2..."
    pip install "nerfstudio==1.0.2"
    verify_import "nerfstudio"

    # ── 6. DreamCatalyst dc + dc_nerf (3d_editing) ───────────────────────────
    echo ""
    echo "[6/7] Installing DreamCatalyst dc + 3d_editing packages..."
    cd "${NERFSTUDIO_DIR}"
    pip install -e .
    pip install -e "${NERFSTUDIO_DIR}/3d_editing"
    cd "${PROJECT_ROOT}"
    pip install numpy==1.26.4
    pip install gsplat==0.1.6
    # diffusers==0.27.2 uses cached_download removed in huggingface_hub>=0.24
    pip install "huggingface_hub<0.24"

    # ── 7. Verify ─────────────────────────────────────────────────────────────
    echo ""
    echo "[7/7] Verifying dreamcatalyst_ns..."
    python -c "
import torch, numpy as np, gsplat, sys
from importlib.metadata import version as V
print(f'  ✓  Python       {sys.version.split()[0]}')
print(f'  ✓  PyTorch      {torch.__version__}')
print(f'  ✓  CUDA         {torch.version.cuda}')
print(f'  ✓  GPU          {torch.cuda.get_device_name(0)}')
print(f'  ✓  NumPy        {np.__version__}')
print(f'  ✓  gsplat       {gsplat.__version__}')
print(f'  ✓  nerfstudio   {V(\"nerfstudio\")}')
print(f'  ✓  dc_nerf      {V(\"dc_nerf\")}')
"
    command -v colmap &>/dev/null \
        && echo "  ✓  COLMAP       ${colmap_ver}" \
        || echo "  ✗  WARNING: colmap not in PATH"

    ns-train -h 2>&1 | grep -qE "\bdc\b" \
        && echo "  ✓  ns-train dc       registered" \
        || echo "  ✗  WARNING: 'dc' not found in ns-train"
    ns-train -h 2>&1 | grep -qE "\bdc_splat\b" \
        && echo "  ✓  ns-train dc_splat registered" \
        || echo "  ✗  WARNING: 'dc_splat' not found in ns-train"

    echo ""
    echo "  ✅  dreamcatalyst_ns setup complete!"
    echo ""
    echo "  Quick start:"
    echo "    conda activate dreamcatalyst_ns"
    echo "    ns-train splatfacto --data data/<scene>"
    echo "    ns-train dc_splat --data data/<scene> \\"
    echo "        --load-dir outputs/<scene>/splatfacto/<timestamp>/nerfstudio_models/ \\"
    echo "        --pipeline.dc.src_prompt 'source description' \\"
    echo "        --pipeline.dc.tgt_prompt 'target description'"
}

# ==============================================================================
#  ENVIRONMENT 2 — dreamcatalyst_gs
#  GaussianEditor-based | Python 3.8 | CUDA 11.7 | PyTorch 2.0.1
#  Optional — only needed for the GaussianEditor variant of DreamCatalyst
# ==============================================================================
setup_gs() {
    echo ""
    echo "============================================================"
    echo "  [ENV 2/2] dreamcatalyst_gs  (GaussianEditor pipeline)"
    echo "============================================================"

    # ── Check threestudio fork ────────────────────────────────────────────────
    if [ ! -d "${THREESTUDIO_DIR}/gaussiansplatting" ]; then
        echo "  threestudio/gaussiansplatting not found — re-cloning DreamCatalyst fork..."
        rm -rf "${THREESTUDIO_DIR}"
        git clone --depth 1 https://github.com/kaist-cvml/DreamCatalyst.git /tmp/_dc_src2
        cp -r /tmp/_dc_src2/threestudio "${THREESTUDIO_DIR}"
        rm -rf /tmp/_dc_src2
        echo "  ✓  DreamCatalyst threestudio fork cloned."
    else
        echo "  ✓  DreamCatalyst threestudio fork detected."
    fi

    # ── 1. Create conda env ───────────────────────────────────────────────────
    echo ""
    echo "[1/5] Creating conda env 'dreamcatalyst_gs' (Python 3.8)..."
    if conda info --envs | grep -q "^dreamcatalyst_gs "; then
        echo "  Already exists, skipping creation."
    else
        conda create -n dreamcatalyst_gs python=3.8 -y
    fi

    eval "$(conda shell.bash hook)"
    conda activate dreamcatalyst_gs
    echo "  Python : $(python --version)"

    # ── 2. CUDA 11.7 + PyTorch 2.0.1 ─────────────────────────────────────────
    echo ""
    echo "[2/5] Installing CUDA 11.7 toolkit and PyTorch 2.0.1..."
    conda install -y -c "nvidia/label/cuda-11.7.0" \
        cuda-nvcc \
        cuda-toolkit \
        cuda-libraries-dev \
        libcufft-dev \
        libcurand-dev
    conda install -y -c conda-forge glm
    pip install ninja cmake "setuptools<70"
    conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        pytorch-cuda=11.7 -c pytorch -c nvidia

    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  ✓  PyTorch {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPU: {torch.cuda.get_device_name(0)}')
"

    # ── 3. Gaussian Splatting submodules ──────────────────────────────────────
    # Same subshell pattern as tinycudann above:
    # - CUDA_HOME scoped to build + import verify only
    # - prevents LD_LIBRARY_PATH from leaking into parent shell
    echo ""
    echo "[3/5] Installing Gaussian Splatting submodules..."
    (
        export CUDA_HOME="/root/anaconda3/envs/dreamcatalyst_gs"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
        echo "  nvcc: $(nvcc --version 2>&1 | grep release)"

        cd "${THREESTUDIO_DIR}/gaussiansplatting"
        rm -rf submodules/diff-gaussian-rasterization/build \
               submodules/simple-knn/build
        pip install --no-build-isolation submodules/diff-gaussian-rasterization
        pip install --no-build-isolation submodules/simple-knn

        # Verify inside subshell — LD_LIBRARY_PATH still active for .so resolution
        python -c "import diff_gaussian_rasterization; print('  ✓  diff-gaussian-rasterization OK')"
        python -c "import simple_knn; print('  ✓  simple-knn OK')"
    )

    # ── 4. Required packages ──────────────────────────────────────────────────
    echo ""
    echo "[4/5] Installing required packages..."
    cd "${THREESTUDIO_DIR}"
    pip install tqdm plyfile mediapipe diffusers==0.27.2
    pip install "huggingface_hub<0.24"
    pip install -r requirements_all.txt
    cd "${PROJECT_ROOT}"

    # ── 5. Verify ─────────────────────────────────────────────────────────────
    echo ""
    echo "[5/5] Verifying dreamcatalyst_gs..."
    python -c "
import torch, diffusers, sys
print(f'  ✓  Python    {sys.version.split()[0]}')
print(f'  ✓  PyTorch   {torch.__version__}')
print(f'  ✓  CUDA      {torch.version.cuda}')
print(f'  ✓  GPU       {torch.cuda.get_device_name(0)}')
print(f'  ✓  diffusers {diffusers.__version__}')
"

    echo ""
    echo "  ✅  dreamcatalyst_gs setup complete!"
    echo ""
    echo "  Quick start:"
    echo "    conda activate dreamcatalyst_gs"
    echo "    cd ${THREESTUDIO_DIR}"
    echo "    python launch.py --config configs/edit-dc.yaml --train --gpu 0 \\"
    echo "        system.seg_prompt='a man' \\"
    echo "        system.prompt_processor.prompt='Turn him into a Batman' \\"
    echo "        data.source=../colmap/yuseung \\"
    echo "        system.gs_source=../scene/yuseung/point_cloud/iteration_30000/point_cloud.ply"
}

# ==============================================================================
#  Main
# ==============================================================================
case "$TARGET" in
    ns)  setup_ns ;;
    gs)  setup_gs ;;
    all) setup_ns; setup_gs ;;
esac

echo ""
echo "============================================================"
echo "  All requested environments ready."
echo ""
echo "  Environments:"
if [[ "$TARGET" == "ns"  || "$TARGET" == "all" ]]; then
    echo "    conda activate dreamcatalyst_ns   ← thesis pipeline (use this)"
fi
if [[ "$TARGET" == "gs"  || "$TARGET" == "all" ]]; then
    echo "    conda activate dreamcatalyst_gs   ← GaussianEditor variant"
fi
echo "============================================================"