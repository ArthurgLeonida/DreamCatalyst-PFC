#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Linux / HPC setup script
# ==============================================================================
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh [env_name]       (default: 3d_edit)
#
#  Prerequisites on the server:
#    - conda  (or mamba / micromamba)
#    - CUDA drivers ≥ 11.8  (check with: nvidia-smi)
#    - COLMAP 3.9.1          (apt install colmap  OR  module load colmap)
#    - FFmpeg                (apt install ffmpeg  OR  module load ffmpeg)
# ==============================================================================

set -euo pipefail

ENV_NAME="${1:-3d_edit}"
PYTHON_VERSION="3.10"

# Pinned upstream versions
NERFSTUDIO_TAG="v1.1.5"

echo "============================================"
echo " DreamCatalyst-NS — Environment Setup"
echo " Conda env: ${ENV_NAME}"
echo "============================================"

# ── 1. Create / activate conda environment ──────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[1/6] Conda env '${ENV_NAME}' already exists. Activating..."
else
    echo "[1/6] Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "  Python: $(python --version)"
echo "  pip:    $(pip --version)"

# ── 2. Install PyTorch with CUDA 11.8 ───────────────────────────────────────
echo "[2/6] Installing PyTorch 2.1.2+cu118..."
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Pin NumPy <2 — PyTorch 2.1.2 was compiled against NumPy 1.x
pip install "numpy<2"

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  ✓ PyTorch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')
"

# ── 3. Clone & install upstream nerfstudio ──────────────────────────────────
echo "[3/6] Setting up nerfstudio ${NERFSTUDIO_TAG}..."
if [ -d "nerfstudio" ]; then
    echo "  nerfstudio/ already exists, skipping clone."
else
    git clone --branch "${NERFSTUDIO_TAG}" --depth 1 \
        https://github.com/nerfstudio-project/nerfstudio.git
fi
pip install -e ./nerfstudio

# ── 4. Clone & install upstream threestudio ─────────────────────────────────
echo "[4/6] Setting up threestudio..."
if [ -d "threestudio" ]; then
    echo "  threestudio/ already exists, skipping clone."
else
    git clone --depth 1 \
        https://github.com/threestudio-project/threestudio.git
fi
pip install -e ./threestudio

# ── 5. Install remaining requirements + this project ────────────────────────
echo "[5/6] Installing requirements.txt + dream_catalyst_ns..."
pip install -r requirements.txt
pip install -e .

# ── 6. Verify installation ─────────────────────────────────────────────────
echo "[6/6] Verifying installation..."

python -c "
import numpy as np
import torch
from importlib.metadata import version as pkg_version
import gsplat
import diffusers
print(f'  ✓ NumPy        {np.__version__}')
print(f'  ✓ PyTorch      {torch.__version__}')
print(f'  ✓ CUDA         {torch.version.cuda}')
print(f'  ✓ GPU          {torch.cuda.get_device_name(0)}')
print(f'  ✓ Nerfstudio   {pkg_version(\"nerfstudio\")}')
print(f'  ✓ gsplat       {gsplat.__version__}')
print(f'  ✓ diffusers    {diffusers.__version__}')
"

# Check that dream-catalyst is registered
if ns-train --help 2>&1 | grep -q "dream-catalyst"; then
    echo "  ✓ dream-catalyst method registered with ns-train"
else
    echo "  ✗ WARNING: dream-catalyst not found in ns-train --help"
    echo "    Try: pip install -e . && ns-install-cli"
fi

# Check threestudio
python -c "
import threestudio
print('  ✓ threestudio %s (%d modules registered)' % (threestudio.__version__, len(threestudio.__modules__)))
" || echo "  ⚠ WARNING: threestudio import failed"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Upload your data to data/<scene>/images/"
echo "   2. Run:  bash scripts/process_data.sh <scene>"
echo "   3. Run:  bash train.sh <scene> [iters]"
echo "============================================"
