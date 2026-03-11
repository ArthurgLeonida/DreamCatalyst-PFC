#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Editing script (Step 3: DDS guidance)
# ==============================================================================
#  Usage:
#    bash scripts/edit.sh <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep]
#
#  rep: splat (default) or nerf
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
        ;;
    nerf|nerfacto)
        METHOD="dc"
        ;;
    *)
        echo "ERROR: Unknown representation '${REP}'. Use 'splat' or 'nerf'."
        exit 1
        ;;
esac

# ── Auto-select least-busy GPU ───────────────────────────────────────────────
echo "[edit.sh] Selecting best available GPU..."
GPU_ID=$(python scripts/pick_gpu.py 2>/dev/null | tail -1 || echo "0")
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================"
echo " Editing:   ${METHOD}"
echo " Scene:     ${SCENE}"
echo " Data:      ${DATA_DIR}"
echo " Iters:     ${MAX_ITERS}"
echo " Src:       ${SRC_PROMPT}"
echo " Tgt:       ${TGT_PROMPT}"
echo " Load from: ${LOAD_DIR}"
echo " GPU idx:   ${CUDA_VISIBLE_DEVICES}"
echo " GPU name:  $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'unknown')"
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

ns-train "${METHOD}" \
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
    --pipeline.dc.guidance-scale 7.5 \
    --pipeline.dc.sd-pretrained-model-or-path timbrooks/instruct-pix2pix

echo ""
echo "============================================"
echo " Editing complete!"
echo " Outputs in: outputs/${SCENE}/${METHOD}/<timestamp>/"
echo "============================================"
