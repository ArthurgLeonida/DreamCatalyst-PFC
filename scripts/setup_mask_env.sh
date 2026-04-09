#!/bin/bash
# Setup a separate conda env for mask generation.
# This avoids breaking dreamcatalyst_ns which pins old transformers/huggingface_hub.
#
# Usage:
#   bash scripts/setup_mask_env.sh
#   conda activate mask_gen
#   python scripts/generate_masks.py --image-dir data/face_processed/images_2 --prompt "person"

set -e

ENV_NAME="mask_gen"

echo "Creating conda env '${ENV_NAME}' with Python 3.10..."
conda create -n "${ENV_NAME}" python=3.10 -y

echo "Installing dependencies..."
conda run -n "${ENV_NAME}" pip install \
    torch torchvision \
    transformers>=4.42.0 \
    segment-anything-py \
    pillow \
    numpy

echo ""
echo "Done! To generate masks:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/generate_masks.py --image-dir data/face_processed/images_2 --prompt \"person\""
echo ""
echo "Then switch back to dreamcatalyst_ns for training:"
echo "  conda activate dreamcatalyst_ns"
