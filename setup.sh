#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Linux / HPC setup script
# ==============================================================================
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
#
#  Prerequisites on the server:
#    - conda  (or mamba / micromamba)
#    - CUDA drivers ≥ 11.8  (check with: nvidia-smi)
#    - COLMAP  (apt install colmap  OR  module load colmap)
#    - FFmpeg  (apt install ffmpeg  OR  module load ffmpeg)
# ==============================================================================

set -euo pipefail

ENV_NAME="${1:-3d_edit}"
PYTHON_VERSION="3.10"

echo "============================================"
echo " DreamCatalyst-NS — Environment Setup"
echo " Conda env: ${ENV_NAME}"
echo "============================================"

# ── 1. Create conda environment ─────────────────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Conda env '${ENV_NAME}' already exists. Activating..."
else
    echo "[1/5] Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate (works in scripts with conda init)
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[INFO] Python: $(python --version)"
echo "[INFO] pip:    $(pip --version)"

# ── 2. Install PyTorch with CUDA 11.8 ───────────────────────────────────────
echo "[2/5] Installing PyTorch 2.1.2+cu118..."
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  ✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# ── 3. Install requirements ────────────────────────────────────────────────
echo "[3/5] Installing requirements.txt..."
pip install -r requirements.txt

# ── 4. Install this project in editable mode ────────────────────────────────
echo "[4/5] Installing dream_catalyst_ns (editable)..."
pip install -e .

# ── 5. Verify installation ─────────────────────────────────────────────────
echo "[5/5] Verifying installation..."

python -c "
import torch, nerfstudio, gsplat, diffusers
print(f'  ✓ PyTorch      {torch.__version__}')
print(f'  ✓ CUDA         {torch.version.cuda}')
print(f'  ✓ GPU          {torch.cuda.get_device_name(0)}')
print(f'  ✓ Nerfstudio   {nerfstudio.__version__}')
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

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Upload your data to data/<scene>/images/"
echo "   2. Run:  bash scripts/process_data.sh <scene>"
echo "   3. Run:  bash train.sh <scene> [iters]"
echo "============================================"
