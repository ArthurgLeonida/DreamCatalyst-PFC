#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Refinement script (Step 4: SDEdit)
# ==============================================================================
#  Usage:
#    bash scripts/refine.sh <scene> <tgt_prompt> <load_dir> [max_iters]
#
#  Example:
#    bash scripts/refine.sh bicycle \
#        "a photo of a motorcycle leaning against a bench" \
#        outputs/bicycle/dc_splat/2026-03-06_120000/nerfstudio_models/
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene> <tgt_prompt> <load_dir> [max_iters]}"
TGT_PROMPT="${2:?Missing tgt_prompt}"
LOAD_DIR="${3:?Missing load_dir (path to dc_splat nerfstudio_models/)}"
MAX_ITERS="${4:-30000}"
DATA_DIR="data/${SCENE}_processed"

# ── Auto-select least-busy GPU ───────────────────────────────────────────────
echo "[refine.sh] Selecting best available GPU..."
GPU_ID=$(python scripts/pick_gpu.py 2>/dev/null | grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9]+' || echo "0")
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================"
echo " Refinement: dc_splat_refinement"
echo " Scene:      ${SCENE}"
echo " Data:       ${DATA_DIR}"
echo " Iters:      ${MAX_ITERS}"
echo " Tgt:        ${TGT_PROMPT}"
echo " Load from:  ${LOAD_DIR}"
echo " GPU idx:    ${CUDA_VISIBLE_DEVICES}"
echo " GPU name:   $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'unknown')"
echo "============================================"

if [ ! -f "${DATA_DIR}/transforms.json" ]; then
    echo "ERROR: ${DATA_DIR}/transforms.json not found."
    exit 1
fi

if [ ! -d "${LOAD_DIR}" ]; then
    echo "ERROR: ${LOAD_DIR} not found."
    echo "Run editing first:  bash scripts/edit.sh ${SCENE} ..."
    exit 1
fi

ns-train dc_splat_refinement \
    --max-num-iterations "${MAX_ITERS}" \
    --mixed-precision False \
    --vis tensorboard \
    --experiment-name "${SCENE}" \
    --load-dir "${LOAD_DIR}" \
    --pipeline.dc.tgt-prompt "${TGT_PROMPT}" \
    --optimizers.xyz.optimizer.lr 1.6e-5 \
    --optimizers.scaling.optimizer.lr 0.001 \
    --optimizers.opacity.optimizer.lr 0.01 \
    pipeline.datamanager:dc-splat-data-manager-config \
        --pipeline.datamanager.dataparser.data "${DATA_DIR}"

echo ""
echo "============================================"
echo " Refinement complete!"
echo " Outputs in: outputs/${SCENE}/dc_splat_refinement/<timestamp>/"
echo "============================================"
