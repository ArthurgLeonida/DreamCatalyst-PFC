#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Refinement script (Step 4: SDEdit)
# ==============================================================================
#  Usage:
#    bash scripts/refine.sh <scene> <tgt_prompt> <load_dir> [max_iters] [rep]
#
#  rep: splat (default) or nerf
#
#  Examples:
#    bash scripts/refine.sh bicycle \
#        "a photo of a motorcycle leaning against a bench" \
#        outputs/bicycle/dc_splat/2026-03-06_120000/nerfstudio_models/
#
#    bash scripts/refine.sh bicycle \
#        "a photo of a motorcycle" \
#        outputs/bicycle/dc/.../nerfstudio_models/ 15000 nerf
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene> <tgt_prompt> <load_dir> [max_iters] [rep]}"
TGT_PROMPT="${2:?Missing tgt_prompt}"
LOAD_DIR="${3:?Missing load_dir (path to edited model nerfstudio_models/)}"
MAX_ITERS="${4:-30000}"
REP="${5:-splat}"        # splat | nerf
DATA_DIR="data/${SCENE}_processed"

# ── Resolve method from representation ────────────────────────────────────────
case "${REP}" in
    splat|3dgs|gaussian)
        METHOD="dc_splat_refinement"
        ;;
    nerf|nerfacto)
        METHOD="dc_refinement"
        ;;
    *)
        echo "ERROR: Unknown representation '${REP}'. Use 'splat' or 'nerf'."
        exit 1
        ;;
esac

# ── Auto-select least-busy GPU ───────────────────────────────────────────────
echo "[refine.sh] Selecting best available GPU..."
GPU_ID=$(python scripts/pick_gpu.py 2>/dev/null | tail -1 || echo "0")
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================"
echo " Refinement: ${METHOD}"
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

ns-train "${METHOD}" \
    --max-num-iterations "${MAX_ITERS}" \
    --mixed-precision False \
    --vis tensorboard \
    --experiment-name "${SCENE}" \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --pipeline.dc.tgt-prompt "${TGT_PROMPT}"

echo ""
echo "============================================"
echo " Refinement complete!"
echo " Outputs in: outputs/${SCENE}/${METHOD}/<timestamp>/"
echo "============================================"
