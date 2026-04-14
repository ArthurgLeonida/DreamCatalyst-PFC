#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS - Refinement script (Step 4: SDEdit)
# ==============================================================================
#  Usage:
#    bash scripts/refine.sh <scene> <tgt_prompt> <load_dir> [max_iters] [rep] [downscale]
#
#  rep: auto (default), splat, or nerf
#  downscale: auto (default), 1, 2, or 4
#  If rep is omitted, the script infers the representation from <load_dir>.
#  The downscale should match the scale used for the reconstruction/editing run.
#
#  Examples:
#    bash scripts/refine.sh bicycle \
#        "a photo of a motorcycle leaning against a bench" \
#        outputs/bicycle/dc_splat/2026-03-06_120000/nerfstudio_models/
#
#    bash scripts/refine.sh face \
#        "A photo of a Tolkien Elf" \
#        outputs/face/dc/2026-04-13_141652/nerfstudio_models/ 30000 nerf 2
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene> <tgt_prompt> <load_dir> [max_iters] [rep] [downscale]}"
TGT_PROMPT="${2:?Missing tgt_prompt}"
LOAD_DIR="${3:?Missing load_dir (path to edited model nerfstudio_models/)}"
MAX_ITERS="${4:-30000}"
REP_INPUT="${5:-auto}"   # auto | splat | nerf
DOWNSCALE="${6:-auto}"   # auto | 1 | 2 | 4
DATA_DIR="data/${SCENE}_processed"

detect_rep_from_load_dir() {
    case "${LOAD_DIR}" in
        */dc_splat/*|*/dc_splat_refinement/*)
            echo "splat"
            ;;
        */dc/*|*/dc_refinement/*)
            echo "nerf"
            ;;
        *)
            return 1
            ;;
    esac
}

case "${REP_INPUT}" in
    auto|"")
        if ! REP=$(detect_rep_from_load_dir); then
            echo "ERROR: Could not infer representation from LOAD_DIR='${LOAD_DIR}'."
            echo "       Please pass the 5th argument explicitly: 'nerf' or 'splat'."
            exit 1
        fi
        ;;
    splat|3dgs|gaussian)
        REP="splat"
        ;;
    nerf|nerfacto)
        REP="nerf"
        ;;
    *)
        echo "ERROR: Unknown representation '${REP_INPUT}'. Use 'auto', 'splat', or 'nerf'."
        exit 1
        ;;
esac

case "${DOWNSCALE}" in
    auto|1|2|4)
        ;;
    *)
        echo "ERROR: Unknown downscale '${DOWNSCALE}'. Use 'auto', 1, 2, or 4."
        exit 1
        ;;
esac

if DETECTED_REP=$(detect_rep_from_load_dir 2>/dev/null); then
    if [ "${DETECTED_REP}" != "${REP}" ]; then
        echo "ERROR: Representation mismatch."
        echo "       LOAD_DIR suggests '${DETECTED_REP}', but rep='${REP_INPUT}' was requested."
        echo "       LOAD_DIR='${LOAD_DIR}'"
        exit 1
    fi
fi

case "${REP}" in
    splat)
        METHOD="dc_splat_refinement"
        NUM_GPUS=1
        DM_CONFIG="dc-splat-data-manager-config"
        ;;
    nerf)
        METHOD="dc_refinement"
        NUM_GPUS=1       # 80GB H100 can fit refinement on a single GPU
        DM_CONFIG="dc-data-manager-config"
        ;;
esac

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
echo " Rep:        ${REP}"
echo " Downscale:  ${DOWNSCALE}"
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

CMD=(ns-train "${METHOD}" \
    --max-num-iterations "${MAX_ITERS}" \
    --mixed-precision False \
    --vis tensorboard \
    --experiment-name "${SCENE}" \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --pipeline.dc.tgt-prompt "${TGT_PROMPT}" \
    pipeline.datamanager:"${DM_CONFIG}")

if [ "${DOWNSCALE}" != "auto" ]; then
    CMD+=(--pipeline.datamanager.dataparser.downscale-factor "${DOWNSCALE}")
fi

"${CMD[@]}"

echo ""
echo "============================================"
echo " Refinement complete!"
echo " Outputs in: outputs/${SCENE}/${METHOD}/<timestamp>/"
echo "============================================"
