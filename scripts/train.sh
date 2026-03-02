#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Training script (Linux / H100)
# ==============================================================================
#  Usage:
#    bash scripts/train.sh bicycle                    # splatfacto, 500 iters (quick test)
#    bash scripts/train.sh bicycle 30000              # splatfacto, full training  
#    bash scripts/train.sh bicycle 30000 2 dc_splat   # DreamCatalyst + 2x downscale
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene_name> [max_iters] [downscale] [method]}"
MAX_ITERS="${2:-500}"
DOWNSCALE="${3:-auto}"           # auto | 1 | 2 | 4
METHOD="${4:-splatfacto}"        # default: splatfacto (matches README)
DATA_DIR="data/${SCENE}_processed"

# Normalize aliases for DreamCatalyst
if [ "${METHOD}" = "dream" ] || [ "${METHOD}" = "dream-catalyst" ]; then
    METHOD="dc_splat"
fi

# ── Auto-select least-busy GPU ───────────────────────────────────────────────
echo "[train.sh] Selecting best available GPU..."
GPU_ID=$(python scripts/pick_gpu.py 2>/dev/null | grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9]+' || echo "0")
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================"
echo " Training: ${METHOD}"
echo " Scene:    ${SCENE}"
echo " Data:     ${DATA_DIR}"
echo " Iters:    ${MAX_ITERS}"
echo " Downscale:${DOWNSCALE}"
echo " GPU idx:  ${CUDA_VISIBLE_DEVICES}"
echo " GPU name: $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'unknown')"
echo "============================================"

if [ ! -f "${DATA_DIR}/transforms.json" ]; then
    echo "ERROR: ${DATA_DIR}/transforms.json not found."
    echo "Run:  bash scripts/process_data.sh ${SCENE}"
    exit 1
fi

# ── Build ns-train command ────────────────────────────────────────────────────
if [ "${METHOD}" = "splatfacto" ]; then
    MP_FLAG="False"
else
    MP_FLAG="True"
fi

# Base command
CMD=(ns-train "${METHOD}" \
        --max-num-iterations "${MAX_ITERS}" \
        --mixed-precision "${MP_FLAG}" \
        --vis tensorboard \
        --experiment-name "${SCENE}" \
        nerfstudio-data \
            --data "${DATA_DIR}")

# Add downscale flag if specified
if [ "${DOWNSCALE}" != "auto" ]; then
    CMD+=(--pipeline.datamanager.dataparser.downscale-factor "${DOWNSCALE}")
fi

# Run training
"${CMD[@]}"

echo ""
echo "============================================"
echo " Training complete!"
echo " Outputs in: outputs/${SCENE}/${METHOD}/<timestamp>/"
echo "============================================"
