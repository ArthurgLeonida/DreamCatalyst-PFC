#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Editing script (Step 3: DDS guidance)
# ==============================================================================
#  Usage:
#    bash scripts/edit.sh <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep]
#
#  rep: splat (default) or nerf
#
#  For NeRF, two GPUs are auto-selected (NeRF model + diffusion model)
#  since nerfacto + IP2P exceeds single-GPU memory.
#  Override with: CUDA_VISIBLE_DEVICES=5,7 bash scripts/edit.sh ...
#
#  Examples:
#    bash scripts/edit.sh bicycle \
#        "a photo of a bicycle leaning against a bench" \
#        "a photo of a motorcycle leaning against a bench" \
#        outputs/bicycle/splatfacto/2026-03-02_045741/nerfstudio_models/
#
#    bash scripts/edit.sh bicycle \
#        "a photo of a bicycle" "a photo of a motorcycle" \
#        outputs/bicycle/nerfacto/.../nerfstudio_models/ 3000 nerf
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep]}"
SRC_PROMPT="${2:?Missing src_prompt}"
TGT_PROMPT="${3:?Missing tgt_prompt}"
LOAD_DIR="${4:?Missing load_dir (path to init model nerfstudio_models/)}"
MAX_ITERS="${5:-3000}"
REP="${6:-splat}"        # splat | nerf
DATA_DIR="data/${SCENE}_processed"

# ── Resolve method from representation ────────────────────────────────────────
case "${REP}" in
    splat|3dgs|gaussian)
        METHOD="dc_splat"
        NUM_GPUS=1
        DM_CONFIG="dc-splat-data-manager-config"
        ;;
    nerf|nerfacto)
        METHOD="dc"
        NUM_GPUS=1       # 80GB H100 can easily fit both!
        DM_CONFIG="dc-data-manager-config"
        ;;
    *)
        echo "ERROR: Unknown representation '${REP}'. Use 'splat' or 'nerf'."
        exit 1
        ;;
esac

# ── Auto-select GPU(s) unless CUDA_VISIBLE_DEVICES is already set ────────────
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[edit.sh] Selecting ${NUM_GPUS} best available GPU(s)..."
    GPU_IDS=$(python scripts/pick_gpu.py "${NUM_GPUS}" 2>/dev/null | tail -1 || echo "0")
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

echo "============================================"
echo " Editing:   ${METHOD}"
echo " Scene:     ${SCENE}"
echo " Data:      ${DATA_DIR}"
echo " Iters:     ${MAX_ITERS}"
echo " Src:       ${SRC_PROMPT}"
echo " Tgt:       ${TGT_PROMPT}"
echo " Load from: ${LOAD_DIR}"
echo " GPUs:      ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

if [ ! -f "${DATA_DIR}/transforms.json" ]; then
    echo "ERROR: ${DATA_DIR}/transforms.json not found."
    echo "Run:  bash scripts/process_data.sh ${SCENE}"
    exit 1
fi

if [ ! -d "${LOAD_DIR}" ]; then
    echo "ERROR: ${LOAD_DIR} not found."
    echo "Train first:  bash scripts/train.sh ${SCENE} 30000"
    exit 1
fi

# ── Build ns-train command ────────────────────────────────────────────────────
CMD=(ns-train "${METHOD}" \
    --machine.seed 42 \
    --max-num-iterations "${MAX_ITERS}" \
    --mixed-precision False \
    --vis tensorboard \
    --experiment-name "${SCENE}" \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --pipeline.dc.src-prompt "${SRC_PROMPT}" \
    --pipeline.dc.tgt-prompt "${TGT_PROMPT}" \
    --pipeline.dc.max-iteration "${MAX_ITERS}" \
    --pipeline.dc.guidance-scale 12.5 \
    --pipeline.dc-device "cuda:0" \
    --pipeline.dc.sd-pretrained-model-or-path timbrooks/instruct-pix2pix \
    pipeline.datamanager:"${DM_CONFIG}" \
    --pipeline.datamanager.dataparser.downscale-factor 2)

# For NeRF: offload diffusion model to second GPU if available
# Count how many GPUs are actually visible (comma-separated list)
ACTUAL_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
if [ "${NUM_GPUS}" -ge 2 ] && [ "${ACTUAL_GPUS}" -ge 2 ]; then
    CMD+=(--pipeline.dc-device cuda:1)
elif [ "${NUM_GPUS}" -ge 2 ] && [ "${ACTUAL_GPUS}" -lt 2 ]; then
    echo "WARNING: NeRF editing needs 2 GPUs but only ${ACTUAL_GPUS} available."
    echo "         This will likely OOM. Set CUDA_VISIBLE_DEVICES=X,Y manually."
    echo "         Proceeding on single GPU anyway..."
fi

"${CMD[@]}"

echo ""
echo "============================================"
echo " Editing complete!"
echo " Outputs in: outputs/${SCENE}/${METHOD}/<timestamp>/"
echo "============================================"
