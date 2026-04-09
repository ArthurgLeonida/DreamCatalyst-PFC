#!/usr/bin/env python3
"""
Generate foreground masks for all training images using Grounded-SAM.
Masks are saved as binary PNGs (255=foreground, 0=background).

These masks are used by Depth-Masked Perp-Neg (N6) to restrict the
Gram-Schmidt projection to the foreground object only, preventing
background texture destruction.

Usage:
    python scripts/generate_masks.py \
        --image-dir data/face_processed/images_2 \
        --prompt "person" \
        --output-dir data/face_processed/masks

    # Quick test with rembg (no text prompt needed):
    python scripts/generate_masks.py \
        --image-dir data/face_processed/images_2 \
        --output-dir data/face_processed/masks \
        --backend rembg

Install (in a SEPARATE env to avoid breaking dreamcatalyst_ns):
    # Option A: Grounded-SAM (best quality, text-prompted)
    conda create -n mask_gen python=3.10 -y
    conda activate mask_gen
    pip install torch torchvision transformers pillow numpy

    # Option B: rembg (simplest, no text prompt)
    pip install rembg pillow numpy
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate_masks_grounded_sam(image_dir, output_dir, prompt, device="cuda"):
    """Generate masks using GroundingDINO + SAM from HuggingFace transformers."""
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from transformers import SamModel, SamProcessor

    print(f"Loading GroundingDINO...")
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)

    print(f"Loading SAM...")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    print(f"Processing {len(image_paths)} images with prompt: '{prompt}'")

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")

        # GroundingDINO: detect object bounding box
        gd_inputs = gd_processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gd_outputs = gd_model(**gd_inputs)
        results = gd_processor.post_process_grounded_object_detection(
            gd_outputs,
            gd_inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],  # (H, W)
        )

        boxes = results[0]["boxes"]
        if len(boxes) == 0:
            print(f"  WARNING: No detection in {img_path.name}, saving empty mask")
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            # SAM: segment from bounding box(es)
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
            # Combine all detected object masks into one binary mask
            combined = torch.zeros(image.height, image.width, dtype=torch.bool)
            for m in masks[0]:  # masks for first image, one per detection box
                # m may be [num_scores, H, W] or [1, num_scores, H, W] — flatten to [H, W]
                m_flat = m.reshape(-1, m.shape[-2], m.shape[-1])  # [N, H, W]
                combined |= m_flat.any(dim=0)  # union across all score variants
            mask = (combined.numpy() * 255).astype(np.uint8)

        out_path = output_dir / f"{img_path.stem}.png"
        Image.fromarray(mask).save(out_path)
        n_fg = (mask > 0).sum() / mask.size * 100
        print(f"  {img_path.name} -> {out_path.name} ({n_fg:.1f}% foreground)")

    print(f"Done! {len(image_paths)} masks saved to {output_dir}")


def generate_masks_rembg(image_dir, output_dir):
    """Generate masks using rembg (U2Net-based background removal)."""
    from rembg import remove

    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    print(f"Processing {len(image_paths)} images with rembg...")

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        # rembg returns RGBA; alpha channel is the foreground mask
        result = remove(image)
        alpha = np.array(result)[:, :, 3]
        # Binarize (rembg alpha can be soft)
        mask = ((alpha > 128) * 255).astype(np.uint8)

        out_path = output_dir / f"{img_path.stem}.png"
        Image.fromarray(mask).save(out_path)
        n_fg = (mask > 0).sum() / mask.size * 100
        print(f"  {img_path.name} -> {out_path.name} ({n_fg:.1f}% foreground)")

    print(f"Done! {len(image_paths)} masks saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate foreground masks for masked Perp-Neg")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory with training images (e.g. data/face_processed/images_2)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for mask PNGs (e.g. data/face_processed/masks)")
    parser.add_argument("--prompt", type=str, default="person",
                        help="Text prompt for Grounded-SAM detection (default: 'person')")
    parser.add_argument("--backend", type=str, default="grounded-sam",
                        choices=["grounded-sam", "rembg"],
                        help="Segmentation backend (default: grounded-sam)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "grounded-sam":
        generate_masks_grounded_sam(args.image_dir, output_dir, args.prompt, args.device)
    elif args.backend == "rembg":
        generate_masks_rembg(args.image_dir, output_dir)


if __name__ == "__main__":
    main()
