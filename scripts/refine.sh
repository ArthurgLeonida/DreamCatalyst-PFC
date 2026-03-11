#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Refinement script (Step 4: SDEdit)
# ==============================================================================
#  Usage:
#    bash scripts/refine.sh <scene> <tgt_prompt> <load_dir> [max_iters] [rep]
#
#  rep: splat (default) or nerf
#
#  For NeRF, two GPUs are auto-selected (NeRF model + diffusion model).
#  Override with: CUDA_VISIBLE_DEVICES=5,7 bash scripts/refine.sh ...
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
        NUM_GPUS=1
        ;;
    nerf|nerfacto)
        METHOD="dc_refinement"
        NUM_GPUS=2       # NeRF needs 2 GPUs: model + diffusion
        ;;
    *)
        echo "ERROR: Unknown representation '${REP}'. Use 'splat' or 'nerf'."
        exit 1
        ;;
esac

# ── Auto-select GPU(s) unless CUDA_VISIBLE_DEVICES is already set ────────────
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[refine.sh] Selecting ${NUM_GPUS} best available GPU(s)..."
    GPU_IDS=$(python scripts/pick_gpu.py "${NUM_GPUS}" 2>/dev/null | tail -1 || echo "0")
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

echo "============================================"
echo " Refinement: ${METHOD}"
echo " Scene:      ${SCENE}"
echo " Data:       ${DATA_DIR}"
echo " Iters:      ${MAX_ITERS}"
echo " Tgt:        ${TGT_PROMPT}"
echo " Load from:  ${LOAD_DIR}"
echo " GPUs:       ${CUDA_VISIBLE_DEVICES}"
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

# ── Build ns-train command ────────────────────────────────────────────────────
CMD=(ns-train "${METHOD}" \
    --max-num-iterations "${MAX_ITERS}" \
    --mixed-precision False \
    --vis tensorboard \
    --experiment-name "${SCENE}" \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --pipeline.dc.tgt-prompt "${TGT_PROMPT}")

# For NeRF: offload diffusion model to second GPU
if [ "${NUM_GPUS}" -ge 2 ]; then
    CMD+=(--pipeline.dc-device cuda:1)
fi

"${CMD[@]}"

echo ""
echo "============================================"
echo " Refinement complete!"
echo " Outputs in: outputs/${SCENE}/${METHOD}/<timestamp>/"
echo "============================================"
