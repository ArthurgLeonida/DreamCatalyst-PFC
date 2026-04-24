#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Editing script (Step 3: DDS guidance)
# ==============================================================================
#  Usage:
#    bash scripts/edit.sh <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep] [downscale]
#
#  rep: splat (default) or nerf
#  downscale: must match Step 2 training downscale (default: 1)
#  After editing, metrics are automatically evaluated and saved inside the
#  experiment folder as metrics.json. Disable with: EVAL_AFTER_EDIT=0 bash ...
#
#  WandB-aware auto-eval:
#  when VIS_MODE=wandb, edit.sh stores WandB files under the method folder and
#  evaluate.py resumes the same training run to attach eval metrics to it.
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

SCENE="${1:?Usage: $0 <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep] [downscale]}"
SRC_PROMPT="${2:?Missing src_prompt}"
TGT_PROMPT="${3:?Missing tgt_prompt}"
LOAD_DIR="${4:?Missing load_dir (path to init model nerfstudio_models/)}"
MAX_ITERS="${5:-3000}"
REP="${6:-nerf}"        # splat | nerf
DOWN_SCALE="${7:-1}"
DATA_DIR="data/${SCENE}_processed"
VIS_MODE="${VIS_MODE:-wandb}"
PROJECT_NAME="${PROJECT_NAME:-dreamcatalyst-pfc}"
EXPERIMENT_NAME="${RUN_NAME:-${SCENE}_dc_edit}"
EVAL_AFTER_EDIT="${EVAL_AFTER_EDIT:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
RUN_DIR=""
TRAIN_LOG=""

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

BASE_OUTPUT_DIR="outputs/${EXPERIMENT_NAME}/${METHOD}"
TRAIN_WANDB_DIR=""
if [ "${VIS_MODE}" = "wandb" ]; then
    TRAIN_WANDB_DIR="${WANDB_DIR:-${BASE_OUTPUT_DIR}/wandb}"
    mkdir -p "${TRAIN_WANDB_DIR}"
fi

extract_run_dir_from_train_log() {
    local log_path="$1"
    local config_path=""

    if [ ! -f "${log_path}" ]; then
        return 0
    fi

    config_path="$(grep -oE 'outputs/[^[:space:]]+/config\.yml' "${log_path}" | tail -n 1 || true)"
    if [ -z "${config_path}" ]; then
        return 0
    fi

    dirname "${config_path}"
}

run_training() {
    if [ -n "${TRAIN_WANDB_DIR}" ]; then
        env WANDB_DIR="${TRAIN_WANDB_DIR}" "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
}

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
echo " Downscale: ${DOWN_SCALE}"
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
    --vis "${VIS_MODE}" \
    --project-name "${PROJECT_NAME}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --data "${DATA_DIR}" \
    --load-dir "${LOAD_DIR}" \
    --pipeline.dc.src-prompt "${SRC_PROMPT}" \
    --pipeline.dc.tgt-prompt "${TGT_PROMPT}" \
    --pipeline.dc.max-iteration "${MAX_ITERS}" \
    --pipeline.dc.guidance-scale 7.5 \
    --pipeline.dc-device "cuda:0" \
    --pipeline.dc.sd-pretrained-model-or-path timbrooks/instruct-pix2pix \
    pipeline.datamanager:"${DM_CONFIG}" \
    --pipeline.datamanager.dataparser.downscale-factor "${DOWN_SCALE}")

TRAIN_LOG="$(mktemp -t edit-train-XXXXXX.log)"
echo "[edit.sh] Capturing training log to ${TRAIN_LOG}"

run_training 2>&1 | tee "${TRAIN_LOG}"

RUN_DIR="$(extract_run_dir_from_train_log "${TRAIN_LOG}")"
if [ -n "${RUN_DIR}" ]; then
    echo "[edit.sh] Resolved run directory: ${RUN_DIR}"
fi

if [ "${EVAL_AFTER_EDIT}" = "1" ]; then
    echo ""
    echo "============================================"
    echo " Running evaluation..."
    echo "============================================"

    if [ -z "${RUN_DIR}" ] || [ ! -f "${RUN_DIR}/config.yml" ]; then
        echo "WARNING: Could not locate the new experiment directory for evaluation."
        echo "         Expected under: ${BASE_OUTPUT_DIR}/<timestamp>/config.yml"
        echo "         Training log: ${TRAIN_LOG}"
    else
        EVAL_CMD=(python scripts/evaluate.py eval \
            --config "${RUN_DIR}/config.yml" \
            --src-prompt "${SRC_PROMPT}" \
            --tgt-prompt "${TGT_PROMPT}" \
            --output-dir "${RUN_DIR}" \
            --device "${EVAL_DEVICE}")

        if [ "${VIS_MODE}" = "wandb" ]; then
            EVAL_CMD+=(--log-wandb)
            if [ -n "${TRAIN_WANDB_DIR}" ]; then
                EVAL_CMD+=(--wandb-dir "${TRAIN_WANDB_DIR}")
            fi
        fi

        if env -u WANDB_MODE "${EVAL_CMD[@]}"; then
            if [ "${VIS_MODE}" = "wandb" ]; then
                echo " Metrics saved to: ${RUN_DIR}/metrics.json. WandB logging attempted."
            else
                echo " Metrics saved to: ${RUN_DIR}/metrics.json."
            fi
        else
            echo "WARNING: Evaluation failed, but the edit run completed successfully."
            echo "         You can retry manually with:"
            printf '         bash scripts/evaluate.sh %q %q %q %q\n' \
                "${RUN_DIR}/config.yml" \
                "${SRC_PROMPT}" \
                "${TGT_PROMPT}" \
                "${RUN_DIR}"
        fi
    fi
fi

echo ""
echo "============================================"
echo " Editing complete!"
if [ -n "${RUN_DIR}" ]; then
    echo " Outputs in: ${RUN_DIR}"
else
    echo " Outputs in: ${BASE_OUTPUT_DIR}/<timestamp>/"
fi
echo "============================================"
