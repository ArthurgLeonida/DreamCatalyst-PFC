#!/usr/bin/env python3
# ==============================================================================
#  DreamCatalyst-NS — Blur detection script
# ==============================================================================
#  Usage:
#    python scripts/check_blur.py chair
#    python scripts/check_blur.py chair --threshold 150.0
# ==============================================================================

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


def compute_blur_score(img_path: Path) -> float | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    parser = argparse.ArgumentParser(description="Check blur in a scene's image dataset.")
    parser.add_argument("scene",                   help="Scene name (looks in data/<scene>/images/)")
    parser.add_argument("--threshold", type=float, default=100.0, help="Blur threshold (default: 100.0)")
    args = parser.parse_args()

    data_dir    = Path("data") / args.scene / "images"
    report_path = Path("data") / args.scene / "blur_report.txt"
    extensions  = {".jpg", ".jpeg", ".png"}

    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        sys.exit(1)

    images = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in extensions])
    if not images:
        print(f"ERROR: No images found in {data_dir}")
        sys.exit(1)

    print("============================================")
    print(f" Scene     : {args.scene}")
    print(f" Image dir : {data_dir}")
    print(f" Threshold : {args.threshold}")
    print(f" Images    : {len(images)}")
    print("============================================\n")

    scores  = []
    skipped = []

    for img_path in images:
        score = compute_blur_score(img_path)
        if score is None:
            print(f"  WARNING: Could not read {img_path.name}, skipping")
            skipped.append(img_path.name)
            continue
        scores.append((img_path, score))

    if not scores:
        print("ERROR: No images could be read.")
        sys.exit(1)

    scores.sort(key=lambda x: x[1])
    blurry  = [(p, s) for p, s in scores if s < args.threshold]
    values  = [s for _, s in scores]
    pct     = len(blurry) / len(scores) * 100

    print(f"  Total checked  : {len(scores)}")
    print(f"  Skipped        : {len(skipped)}")
    print(f"  Blurry images  : {len(blurry)} ({pct:.1f}%)")
    print(f"  Average score  : {np.mean(values):.2f}")
    print(f"  Sharpest       : {scores[-1][1]:.2f}  ({scores[-1][0].name})")
    print(f"  Blurriest      : {scores[0][1]:.2f}  ({scores[0][0].name})")

    if blurry:
        print(f"\n  --- Blurry images (score < {args.threshold}) ---")
        for p, s in blurry:
            print(f"  {s:8.2f}  {p.name}")
    else:
        print("\n  All images are sharp!")

    # Save report
    with open(report_path, "w") as f:
        f.write(f"Blur report — scene: {args.scene}\n")
        f.write(f"Threshold : {args.threshold}\n")
        f.write(f"Total     : {len(scores)} | Blurry: {len(blurry)} ({pct:.1f}%) | Skipped: {len(skipped)}\n\n")
        f.write(f"{'Score':>10}  {'Status':<10}  Filename\n")
        f.write("-" * 55 + "\n")
        for p, s in scores:
            status = "BLURRY" if s < args.threshold else "OK"
            f.write(f"{s:10.2f}  {status:<10}  {p.name}\n")

    print(f"\n  Report saved to: {report_path}")
    print("\n============================================")


if __name__ == "__main__":
    main()
