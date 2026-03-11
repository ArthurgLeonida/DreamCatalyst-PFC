#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Training script (Step 2: initialization)
# ==============================================================================
#  Usage:
#    bash scripts/train.sh bicycle                    # splatfacto, 500 iters (quick test)
#    bash scripts/train.sh bicycle 30000              # splatfacto, full training
#    bash scripts/train.sh bicycle 30000 2            # splatfacto + 2x downscale
#
#  This script is for initialization only (splatfacto / nerfacto).
#  For editing (Step 3), use:  bash scripts/edit.sh
#  For refinement (Step 4), use:  bash scripts/refine.sh
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene_name> [max_iters] [downscale] [method]}"
MAX_ITERS="${2:-500}"
DOWNSCALE="${3:-auto}"           # auto | 1 | 2 | 4
METHOD="${4:-splatfacto}"        # splatfacto | nerfacto
DATA_DIR="data/${SCENE}_processed"

# Block editing methods — they need edit.sh / refine.sh
case "${METHOD}" in
    dc_splat|dc|dream|dream-catalyst)
        echo "ERROR: '${METHOD}' is an editing method. Use scripts/edit.sh instead."
        exit 1
        ;;
    dc_splat_refinement|dc_refinement)
        echo "ERROR: '${METHOD}' is a refinement method. Use scripts/refine.sh instead."
        exit 1
        ;;
esac

# ── Auto-select least-busy GPU ───────────────────────────────────────────────
echo "[train.sh] Selecting best available GPU..."
GPU_ID=$(python scripts/pick_gpu.py 2>/dev/null | tail -1 || echo "0")
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

# Base command
CMD=(ns-train "${METHOD}" \
        --max-num-iterations "${MAX_ITERS}" \
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
