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

echo ""
echo "============================================"
echo " Processing complete!"
echo " Output: ${OUTPUT_DIR}"
echo ""
echo " Verify:"
echo "   ls ${OUTPUT_DIR}/transforms.json"
echo "   ls ${OUTPUT_DIR}/images/"
if [ "${IMAGE_COUNT}" -gt 0 ]; then
    REGISTERED=$(grep -v '^#' "${OUTPUT_DIR}/colmap/sparse/0/images.txt" 2>/dev/null \
        | grep -v '^$' | wc -l | awk '{print int($1/2)}')
    echo "   COLMAP registered: ${REGISTERED} / ${IMAGE_COUNT} images"
fi
echo "============================================"
