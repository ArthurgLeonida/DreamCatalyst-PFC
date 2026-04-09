#!/usr/bin/env python3
"""
Generate foreground masks for all training images using GroundingDINO + SAM.
Masks are saved as binary PNGs (255=foreground, 0=background).

Used by Foreground-Masked Perp-Neg (N6) to restrict the Gram-Schmidt
projection to the foreground object only, preventing background destruction.

IMPORTANT: Run this in the 'mask_gen' conda env, NOT 'dreamcatalyst_ns'.
           Setup: bash scripts/setup_mask_env.sh

Usage:
    # Default: masks saved to <image-dir>/../masks/
    python scripts/generate_masks.py \
        --image-dir data/face_processed/images_2 \
        --prompt "person"

    # Custom output dir:
    python scripts/generate_masks.py \
        --image-dir data/face_processed/images_2 \
        --prompt "person" \
        --output-dir data/face_processed/masks_custom

    # rembg backend (no text prompt needed, simpler):
    python scripts/generate_masks.py \
        --image-dir data/face_processed/images_2 \
        --backend rembg
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ── GPU selection (same pattern as edit.sh / pick_gpu.py) ──────────────────────

def auto_select_gpu(device: str) -> str:
    """Pick the least-busy GPU if CUDA_VISIBLE_DEVICES is not already set."""
    if device != "cuda":
        return device
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[GPU] Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        return device
    try:
        out = subprocess.check_output(
            [sys.executable, "scripts/pick_gpu.py", "1"],
            stderr=subprocess.PIPE,
        ).decode().strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = out
        print(f"[GPU] Auto-selected GPU {out}")
    except Exception as e:
        print(f"[GPU] Auto-selection failed ({e}), using default")
    return device


# ── Image discovery ────────────────────────────────────────────────────────────

def iter_image_paths(image_dir):
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )


# ── Backend: GroundingDINO + SAM ───────────────────────────────────────────────

def generate_masks_grounded_sam(image_dir, output_dir, prompt, device="cuda"):
    """Generate masks using GroundingDINO (text-prompted detector) + SAM."""
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from transformers import SamModel, SamProcessor

    print("Loading GroundingDINO...")
    gd_id = "IDEA-Research/grounding-dino-tiny"
    gd_processor = AutoProcessor.from_pretrained(gd_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(device)

    print("Loading SAM...")
    sam_id = "facebook/sam-vit-base"
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(device)

    image_paths = iter_image_paths(image_dir)
    print(f"Processing {len(image_paths)} images with prompt: '{prompt}'")

    # GroundingDINO expects prompt ending with "."
    prompt_text = prompt if prompt.strip().endswith(".") else f"{prompt.strip()}."

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")

        # Detect bounding boxes
        gd_inputs = gd_processor(images=image, text=prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gd_outputs = gd_model(**gd_inputs)
        results = gd_processor.post_process_grounded_object_detection(
            gd_outputs,
            gd_inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )
        boxes = results[0]["boxes"]

        if len(boxes) == 0:
            print(f"  WARNING: No detection in {img_path.name}, saving empty mask")
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            # SAM segmentation from bounding boxes
            sam_inputs = sam_processor(
                image, input_boxes=[boxes.cpu().tolist()], return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)
            masks = sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )
            # Union all detection masks into one binary mask
            combined = torch.zeros(image.height, image.width, dtype=torch.bool)
            for m in masks[0]:
                # m shape varies across SAM versions; flatten to [N, H, W] then union
                m_2d = m.reshape(-1, m.shape[-2], m.shape[-1]).any(dim=0)
                combined |= m_2d
            mask = (combined.numpy() * 255).astype(np.uint8)

        out_path = output_dir / f"{img_path.stem}.png"
        Image.fromarray(mask).save(out_path)
        fg_pct = (mask > 0).sum() / mask.size * 100
        print(f"  {img_path.name} -> {out_path.name} ({fg_pct:.1f}% foreground)")

    print(f"\nDone! {len(image_paths)} masks saved to {output_dir}")


# ── Backend: rembg ─────────────────────────────────────────────────────────────

def generate_masks_rembg(image_dir, output_dir):
    """Generate masks using rembg (U2Net-based background removal). No text prompt needed."""
    from rembg import remove

    image_paths = iter_image_paths(image_dir)
    print(f"Processing {len(image_paths)} images with rembg...")

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        result = remove(image)
        alpha = np.array(result)[:, :, 3]
        mask = ((alpha > 128) * 255).astype(np.uint8)

        out_path = output_dir / f"{img_path.stem}.png"
        Image.fromarray(mask).save(out_path)
        fg_pct = (mask > 0).sum() / mask.size * 100
        print(f"  {img_path.name} -> {out_path.name} ({fg_pct:.1f}% foreground)")

    print(f"\nDone! {len(image_paths)} masks saved to {output_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate foreground masks for Foreground-Masked Perp-Neg (N6)"
    )
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Training images (e.g. data/face_processed/images_2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for masks (default: <image-dir>/../masks/)")
    parser.add_argument("--prompt", type=str, default="person",
                        help="Text prompt for detection (default: 'person')")
    parser.add_argument("--backend", type=str, default="grounded-sam",
                        choices=["grounded-sam", "rembg"],
                        help="Segmentation backend (default: grounded-sam)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Default output: <scene_processed>/masks/ (sibling of images_2/, images_4/, etc.)
    if args.output_dir is None:
        args.output_dir = str(Path(args.image_dir).parent / "masks")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.device = auto_select_gpu(args.device)

    print(f"Image dir:  {args.image_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Backend:    {args.backend}")
    print()

    if args.backend == "grounded-sam":
        generate_masks_grounded_sam(args.image_dir, output_dir, args.prompt, args.device)
    elif args.backend == "rembg":
        generate_masks_rembg(args.image_dir, output_dir)


if __name__ == "__main__":
    main()
