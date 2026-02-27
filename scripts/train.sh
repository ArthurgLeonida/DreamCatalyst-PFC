#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Training script (Linux / H100)
# ==============================================================================
#  Usage:
#    bash train.sh chair                     # vanilla splatfacto, 500 iters (test)
#    bash train.sh chair 30000               # vanilla splatfacto, full training
#    bash train.sh chair 30000 dream         # dream-catalyst method
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene_name> [max_iters] [method]}"
MAX_ITERS="${2:-500}"
METHOD="${3:-splatfacto}"
DATA_DIR="data/${SCENE}_processed"

if [ "${METHOD}" = "dream" ] || [ "${METHOD}" = "dream-catalyst" ]; then
    METHOD="dream-catalyst"
fi

# ── Auto-select least-busy GPU ───────────────────────────────────────────────
echo "[train.sh] Selecting best available GPU..."
eval "$(python scripts/pick_gpu.py 2>&1 | tail -1 | sed -n 's/.*CUDA_VISIBLE_DEVICES=\([0-9]*\)/export CUDA_VISIBLE_DEVICES=\1/p')"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

echo "============================================"
echo " Training: ${METHOD}"
echo " Scene:    ${SCENE}"
echo " Data:     ${DATA_DIR}"
echo " Iters:    ${MAX_ITERS}"
echo " GPU idx:  ${CUDA_VISIBLE_DEVICES}"
echo " GPU name: $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'unknown')"
echo "============================================"

if [ ! -f "${DATA_DIR}/transforms.json" ]; then
    echo "ERROR: ${DATA_DIR}/transforms.json not found."
    echo "Run:  bash scripts/process_data.sh ${SCENE}"
    exit 1
fi

# ── Run training ─────────────────────────────────────────────────────────────
ns-train "${METHOD}" \
    --max-num-iterations "${MAX_ITERS}" \
    --mixed-precision True \
    --vis tensorboard \
    --experiment-name "${SCENE}" \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.use-scale-regularization True \
    nerfstudio-data \
        --data "${DATA_DIR}"

echo ""
echo "============================================"
echo " Training complete!"
echo " Outputs in: outputs/${SCENE}/${METHOD}/"
echo ""
echo " View results:"
echo "   tensorboard --logdir outputs/"
echo "   CONFIG=\$(find outputs/${SCENE} -name config.yml | tail -1)"
echo "   ns-export gaussian-splat --load-config \"\$CONFIG\" --output-dir exports/${SCENE}/"
echo "============================================"
