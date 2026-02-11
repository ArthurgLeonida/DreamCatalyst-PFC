#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS — Data processing script (Linux)
# ==============================================================================
#  Usage:
#    bash scripts/process_data.sh chair
#    bash scripts/process_data.sh chair video    # if using a video source
# ==============================================================================

set -euo pipefail

SCENE="${1:?Usage: $0 <scene_name> [images|video]}"
SOURCE_TYPE="${2:-images}"
DATA_DIR="data/${SCENE}"
OUTPUT_DIR="data/${SCENE}_processed"

echo "============================================"
echo " Processing scene: ${SCENE}"
echo " Source type:      ${SOURCE_TYPE}"
echo " Input:            ${DATA_DIR}"
echo " Output:           ${OUTPUT_DIR}"
echo "============================================"

# Verify prerequisites
command -v colmap >/dev/null 2>&1 || { echo "ERROR: colmap not found. Install it first."; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found. Install it first."; exit 1; }

if [ "${SOURCE_TYPE}" = "video" ]; then
    VIDEO_FILE=$(find "${DATA_DIR}" -maxdepth 1 -name "*.mp4" -o -name "*.mov" -o -name "*.avi" | head -1)
    if [ -z "${VIDEO_FILE}" ]; then
        echo "ERROR: No video file found in ${DATA_DIR}/"
        exit 1
    fi
    echo "Video file: ${VIDEO_FILE}"
    ns-process-data video --data "${VIDEO_FILE}" --output-dir "${OUTPUT_DIR}"
else
    if [ ! -d "${DATA_DIR}/images" ]; then
        echo "ERROR: ${DATA_DIR}/images/ does not exist"
        exit 1
    fi
    IMAGE_COUNT=$(find "${DATA_DIR}/images" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    echo "Found ${IMAGE_COUNT} images"
    ns-process-data images --data "${DATA_DIR}/images" --output-dir "${OUTPUT_DIR}"
fi

echo ""
echo "============================================"
echo " Processing complete!"
echo " Output: ${OUTPUT_DIR}"
echo ""
echo " Verify:"
echo "   ls ${OUTPUT_DIR}/transforms.json"
echo "   ls ${OUTPUT_DIR}/sparse_pc.ply"
echo "   ls ${OUTPUT_DIR}/images_4/"
echo "============================================"
