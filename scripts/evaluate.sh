#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Evaluation script
# ==============================================================================
#  Usage:
#    bash scripts/evaluate.sh <config_yml> <src_prompt> <tgt_prompt> [output_dir]
#
#  Compare:
#    bash scripts/evaluate.sh --compare eval_results/exp1 eval_results/exp2 ...
#
#  Examples:
#    bash scripts/evaluate.sh \
#        outputs/bicycle/dc_splat/2026-03-11_025213/config.yml \
#        "a photo of a bicycle" \
#        "a photo of a motorcycle" \
#        eval_results/bicycle_fulltag_1.1
#
#    bash scripts/evaluate.sh --compare \
#        eval_results/bicycle_fulltag_1.1 \
#        eval_results/bicycle_adatag_1.1 \
#        eval_results/bicycle_cfgfree
# ==============================================================================

set -euo pipefail

if [ "${1:-}" = "--compare" ]; then
    shift
    python scripts/evaluate.py compare "$@"
    exit 0
fi

CONFIG="${1:?Usage: $0 <config_yml> <src_prompt> <tgt_prompt> [output_dir]}"
SRC_PROMPT="${2:?Missing src_prompt}"
TGT_PROMPT="${3:?Missing tgt_prompt}"

# Auto-generate output dir from config path if not provided
if [ -n "${4:-}" ]; then
    OUTPUT_DIR="$4"
else
    # Extract: outputs/<scene>/<method>/<timestamp> -> eval_results/<scene>_<method>_<timestamp>
    REL_PATH=$(dirname "$(dirname "${CONFIG}")")
    OUTPUT_DIR="eval_results/$(echo "${REL_PATH}" | sed 's|outputs/||' | tr '/' '_')"
fi

# Auto-select GPU
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_ID=$(python scripts/pick_gpu.py 1 2>/dev/null | tail -1 || echo "0")
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

echo "============================================"
echo " Evaluating: ${CONFIG}"
echo " Src:        ${SRC_PROMPT}"
echo " Tgt:        ${TGT_PROMPT}"
echo " Output:     ${OUTPUT_DIR}"
echo " GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

python scripts/evaluate.py eval \
    --config "${CONFIG}" \
    --src-prompt "${SRC_PROMPT}" \
    --tgt-prompt "${TGT_PROMPT}" \
    --output-dir "${OUTPUT_DIR}"
