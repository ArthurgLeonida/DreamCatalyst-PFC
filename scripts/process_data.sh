#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Data processing script (Linux)
# ==============================================================================
#  Usage:
#    bash scripts/process_data.sh <scene_name> [images|video]
#
#  Examples:
#    bash scripts/process_data.sh hero
#    bash scripts/process_data.sh hero video
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene_name> [images|video]}"
SOURCE_TYPE="${2:-images}"
DATA_DIR="data/${SCENE}"
OUTPUT_DIR="data/${SCENE}_processed"
IMAGE_COUNT=0   # initialized here so it's always in scope

echo "============================================"
echo " Processing scene: ${SCENE}"
echo " Source type:      ${SOURCE_TYPE}"
echo " Input:            ${DATA_DIR}"
echo " Output:           ${OUTPUT_DIR}"
echo "============================================"

# Verify prerequisites
command -v colmap          >/dev/null 2>&1 || { echo "ERROR: colmap not found."; exit 1; }
command -v ffmpeg          >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found."; exit 1; }
command -v ns-process-data >/dev/null 2>&1 || { echo "ERROR: ns-process-data not found. Run: conda activate dreamcatalyst_ns"; exit 1; }

# Normalize image extensions to lowercase
# Fixes nerfstudio silently dropping .JPG/.PNG files (uppercase not matched)
normalize_extensions() {
    local dir="$1"
    find "${dir}" -type f \( -name "*.JPG" -o -name "*.JPEG" -o -name "*.PNG" \) | \
    while read -r f; do
        lower="${f%.*}.$(echo "${f##*.}" | tr '[:upper:]' '[:lower:]')"
        [ "$f" != "$lower" ] && mv "$f" "$lower" && echo "  Renamed: $(basename "$f") -> $(basename "$lower")"
    done
}

# COLMAP's mapper can emit several disconnected sub-models (sparse/0,
# sparse/1, ...), numbered by CREATION ORDER, not size. ns-process-data
# always reads sparse/0, which is sometimes a tiny fragment while the real
# reconstruction sits in another folder (observed on the campsite scene:
# sparse/0 had 4 images, sparse/1 had all 174). This picks the model with
# the most registered images and, if it isn't sparse/0, regenerates
# transforms.json from it — no COLMAP recompute.
select_best_colmap_model() {
    local sparse_dir="${OUTPUT_DIR}/colmap/sparse"
    [ -d "${sparse_dir}" ] || return 0

    local best_idx="" best_count=-1
    for d in "${sparse_dir}"/*/; do
        [ -d "${d}" ] || continue
        local idx count
        idx="$(basename "${d}")"
        count="$(colmap model_analyzer --path "${d}" 2>&1 \
            | sed -n 's/.*Registered images:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)"
        count="${count:-0}"
        echo "  COLMAP model ${idx}: ${count} registered images"
        if [ "${count}" -gt "${best_count}" ]; then
            best_count="${count}"
            best_idx="${idx}"
        fi
    done

    [ -n "${best_idx}" ] || { echo "WARNING: no COLMAP sub-models found."; return 0; }
    echo "  Largest model: sparse/${best_idx} (${best_count} registered)"

    if [ "${best_idx}" != "0" ]; then
        echo "  sparse/0 was not the largest — regenerating transforms.json from sparse/${best_idx}..."
        ns-process-data images \
            --data "${OUTPUT_DIR}/images" \
            --output-dir "${OUTPUT_DIR}" \
            --skip-colmap \
            --skip-image-processing \
            --colmap-model-path "colmap/sparse/${best_idx}" \
        || ns-process-data images \
            --data "${OUTPUT_DIR}/images" \
            --output-dir "${OUTPUT_DIR}" \
            --skip-colmap \
            --colmap-model-path "colmap/sparse/${best_idx}"
    fi
}

if [ "${SOURCE_TYPE}" = "video" ]; then
    VIDEO_FILE=$(find "${DATA_DIR}" -maxdepth 1 \( -name "*.mp4" -o -name "*.mov" -o -name "*.avi" \) | head -1)
    if [ -z "${VIDEO_FILE}" ]; then
        echo "ERROR: No video file found in ${DATA_DIR}/"
        exit 1
    fi
    echo "Video file: ${VIDEO_FILE}"

    ns-process-data video \
        --data "${VIDEO_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-downscales 3 \
        --matching-method exhaustive

else
    if [ ! -d "${DATA_DIR}/images" ]; then
        echo "ERROR: ${DATA_DIR}/images/ does not exist"
        exit 1
    fi

    normalize_extensions "${DATA_DIR}/images"

    IMAGE_COUNT=$(find "${DATA_DIR}/images" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    echo "Found ${IMAGE_COUNT} images"
    if [ "${IMAGE_COUNT}" -eq 0 ]; then
        echo "ERROR: No images found in ${DATA_DIR}/images/"
        exit 1
    fi

    # Wipe stale output to avoid COLMAP reusing a corrupt database
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "Removing stale output dir: ${OUTPUT_DIR}"
        rm -rf "${OUTPUT_DIR}"
    fi

    ns-process-data images \
        --data "${DATA_DIR}/images" \
        --output-dir "${OUTPUT_DIR}" \
        --matching-method exhaustive \
        --no-gpu # Fixed to run on the server, GPU was not working

fi

# Guard against COLMAP's multi-model numbering quirk (see function comment).
echo ""
echo "Selecting best COLMAP model..."
select_best_colmap_model

echo ""
echo "============================================"
echo " Processing complete!"
echo " Output: ${OUTPUT_DIR}"
echo ""
echo " Verify:"
echo "   ls ${OUTPUT_DIR}/transforms.json"
echo "   ls ${OUTPUT_DIR}/images/"
if [ -f "${OUTPUT_DIR}/transforms.json" ]; then
    FRAMES=$(python -c "import json; print(len(json.load(open('${OUTPUT_DIR}/transforms.json'))['frames']))" 2>/dev/null || echo "?")
    if [ "${IMAGE_COUNT}" -gt 0 ]; then
        echo "   Frames in transforms.json: ${FRAMES} / ${IMAGE_COUNT} input images"
    else
        echo "   Frames in transforms.json: ${FRAMES}"
    fi
fi
echo "============================================"
