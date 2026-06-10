#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/edit.sh <scene> <src_prompt> <tgt_prompt> <load_dir> [max_iters] [rep] [downscale]
#
# Runtime knobs kept here:
#   RUN_NAME, PROJECT_NAME, VIS_MODE, EVAL_AFTER_EDIT, EVAL_DEVICE, CUDA_VISIBLE_DEVICES
#
# Method knobs live in:
#   nerfstudio/dc/method_config.py

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

case "${REP}" in
    splat|3dgs|gaussian)
        METHOD="dc_splat"
        NUM_GPUS=1
        DM_CONFIG="dc-splat-data-manager-config"
        ;;
    nerf|nerfacto)
        METHOD="dc"
        NUM_GPUS=1
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

print_voxel_config() {
    python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("nerfstudio/dc/method_config.py")
spec = importlib.util.spec_from_file_location("method_config", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

voxel = module.VOXEL_CACHE_PARAMS
dc = module.DC_CUSTOM_PARAMS
enabled = 1 if voxel.get("mask_voxel_cache_enabled") else 0
ca_enabled = 1 if dc.get("cross_attention_mask_enabled") else 0

print(f" CA mask:   {ca_enabled}")
if ca_enabled:
    branch = "\u251c\u2500"
    last = "\u2514\u2500"
    arrow = "\u2192"
    sched = 1 if dc.get("cross_attention_mask_weight_schedule_enabled") else 0
    print(f"   {branch} weight:     {dc['cross_attention_mask_weight']}")
    print(f"   {branch} gamma:      {dc['cross_attention_mask_gamma']}")
    if sched:
        print(
            f"   {last} schedule:   "
            f"reverse_tag 0{arrow}{dc['cross_attention_mask_weight']} "
            f"(power={dc['cross_attention_mask_weight_schedule_power']})"
        )
    else:
        print(f"   {last} schedule:   off")
print(f" Voxel 3D:  {enabled}")
if enabled:
    branch = "\u251c\u2500"
    last = "\u2514\u2500"
    arrow = "\u2192"
    print(f"   {branch} res:        {voxel['mask_voxel_cache_resolution']}")
    if voxel.get("mask_voxel_cache_ema_beta_auto", False):
        factor = voxel.get("mask_voxel_cache_ema_beta_camera_factor", 2.0)
        print(f"   {branch} ema:        auto 1-1/({factor}*Ncam)")
    else:
        print(f"   {branch} ema:        {voxel['mask_voxel_cache_ema_beta']}")
    print(
        f"   {branch} bbox src:   "
        f"{voxel['mask_voxel_cache_bbox_source']} "
        f"(q={voxel['mask_voxel_cache_bbox_observe_quantile']}, "
        f"infl={voxel['mask_voxel_cache_bbox_inflation']})"
    )
    print(f"   {branch} acc thr:    {voxel['mask_voxel_cache_accumulation_threshold']}")
    print(f"   {branch} upd thr:    {voxel['mask_voxel_cache_update_threshold']}")
    if voxel.get("mask_voxel_cache_confidence_enabled", False):
        if voxel.get("mask_voxel_cache_min_observations_auto", False):
            obs = (
                f"ceil(Ncam*{voxel['mask_voxel_cache_observation_fraction']}) "
                f"[{voxel['mask_voxel_cache_min_observations_floor']},"
                f"{voxel['mask_voxel_cache_min_observations_cap']}]"
            )
        else:
            obs = str(voxel["mask_voxel_cache_min_observations"])
        vdecay = voxel.get('mask_voxel_cache_variance_decay', 0.0)
        var_mode = f"EW(a={vdecay})" if vdecay and vdecay > 0.0 else "Welford"
        vpeak = voxel.get('mask_voxel_cache_variance_peak_decay', 0.0)
        if vpeak and vpeak > 0.0:
            var_mode += f"+peak({vpeak})"
        print(
            f"   {branch} trust:      "
            f"obs>={obs}, "
            f"var<={voxel['mask_voxel_cache_max_variance']} [{var_mode}]"
        )
    else:
        print(f"   {branch} trust:      off")
    print(
        f"   {branch} blend:      "
        f"{voxel['mask_voxel_cache_max_blend']} "
        f"(warmup {voxel['mask_voxel_cache_warmup_start']}{arrow}"
        f"{voxel['mask_voxel_cache_warmup_end']})"
    )
    print(f"   {branch} fusion:     {dc['external_mask_fusion']}")
    print(f"   {branch} gate str:   {dc['external_mask_screen_attn_gate_strength']}")
    neg_var_p = dc.get('external_mask_negative_variance_power', 0.0)
    if neg_var_p > 0.0:
        print(f"   {branch} neg var p:  {neg_var_p}")
    contested_r = dc.get('external_mask_contested_suppression_ratio', 0.0)
    if contested_r > 0.0:
        print(f"   {branch} contested:  {contested_r} (suppress M where var>=gate)")
    print(f"   {branch} update src: {voxel.get('mask_voxel_cache_update_source', 'internal')}")
    ang_p = voxel.get('mask_voxel_cache_angular_power', 0.0)
    if ang_p > 0.0:
        ang_floor = voxel.get('mask_voxel_cache_min_angular_factor', 0.0)
        ang_rel = voxel.get('mask_voxel_cache_angular_relative', False)
        rel_str = "relative" if ang_rel else "absolute"
        print(f"   {branch} ang gate:   power={ang_p} floor={ang_floor} ({rel_str})")
    else:
        print(f"   {branch} ang gate:   off")
    mass_p = voxel.get('mask_voxel_cache_mass_power', 0.0)
    mass_t = voxel.get('mask_voxel_cache_mass_threshold', 0.0)
    if mass_p > 0.0 and mass_t > 0.0:
        print(f"   {branch} mass gate:  power={mass_p} threshold={mass_t}")
    else:
        print(f"   {branch} mass gate:  off")
    print(f"   {last} stg+tag:    {dc.get('stg_tag_compose_mode', 'sequential')}")
PY
}

run_training() {
    if [ -n "${TRAIN_WANDB_DIR}" ]; then
        env WANDB_DIR="${TRAIN_WANDB_DIR}" "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
}

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
print_voxel_config
echo "============================================"

if [ ! -f "${DATA_DIR}/transforms.json" ]; then
    echo "ERROR: ${DATA_DIR}/transforms.json not found."
    echo "Run: bash scripts/process_data.sh ${SCENE}"
    exit 1
fi

if [ ! -d "${LOAD_DIR}" ]; then
    echo "ERROR: ${LOAD_DIR} not found."
    echo "Train first: bash scripts/train.sh ${SCENE} 30000"
    exit 1
fi

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
            echo " Metrics saved to: ${RUN_DIR}/metrics.json."
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
