#!/usr/bin/env python3
# ==============================================================================
#  DreamCatalyst-NS — Blur detection script
# ==============================================================================
#  Usage:
#    python scripts/check_blur.py hero
#    python scripts/check_blur.py hero --threshold 80
#    python scripts/check_blur.py hero --threshold 50 --delete
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
    parser.add_argument("scene",                    help="Scene name (looks in data/<scene>/images/)")
    parser.add_argument("--threshold", type=float,  default=None, help="Blur threshold (default: 100.0)")
    parser.add_argument("--delete",    action="store_true",        help="Delete blurry images (requires --threshold)")
    args = parser.parse_args()

    # --delete requires --threshold to be explicitly set
    if args.delete and args.threshold is None:
        parser.error("--delete requires --threshold to be explicitly provided.")

    threshold = args.threshold if args.threshold is not None else 100.0

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
    print(f" Threshold : {threshold}")
    print(f" Images    : {len(images)}")
    print(f" Delete    : {'yes' if args.delete else 'no'}")
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
    blurry  = [(p, s) for p, s in scores if s < threshold]
    values  = [s for _, s in scores]
    pct     = len(blurry) / len(scores) * 100

    print(f"  Total checked  : {len(scores)}")
    print(f"  Skipped        : {len(skipped)}")
    print(f"  Blurry images  : {len(blurry)} ({pct:.1f}%)")
    print(f"  Average score  : {np.mean(values):.2f}")
    print(f"  Sharpest       : {scores[-1][1]:.2f}  ({scores[-1][0].name})")
    print(f"  Blurriest      : {scores[0][1]:.2f}  ({scores[0][0].name})")

    if blurry:
        print(f"\n  --- Blurry images (score < {threshold}) ---")
        for p, s in blurry:
            print(f"  {s:8.2f}  {p.name}")
    else:
        print("\n  All images are sharp!")

    # Delete blurry images if requested
    deleted  = []
    d_failed = []
    if args.delete and blurry:
        print(f"\n  --- Deleting {len(blurry)} blurry image(s) ---")
        for p, s in blurry:
            try:
                p.unlink()
                deleted.append(p.name)
                print(f"  DELETED  {p.name}  (score: {s:.2f})")
            except OSError as e:
                d_failed.append(p.name)
                print(f"  FAILED   {p.name}  ({e})")

    # Save report
    with open(report_path, "w") as f:
        f.write(f"Blur report — scene: {args.scene}\n")
        f.write(f"Threshold : {threshold}\n")
        f.write(f"Total     : {len(scores)} | Blurry: {len(blurry)} ({pct:.1f}%) | Skipped: {len(skipped)}\n")
        if args.delete:
            f.write(f"Deleted   : {len(deleted)} | Failed: {len(d_failed)}\n")
        f.write("\n")
        f.write(f"{'Score':>10}  {'Status':<10}  Filename\n")
        f.write("-" * 55 + "\n")
        for p, s in scores:
            if s < threshold:
                status = "DELETED" if p.name in deleted else ("FAILED" if p.name in d_failed else "BLURRY")
            else:
                status = "OK"
            f.write(f"{s:10.2f}  {status:<10}  {p.name}\n")

    print(f"\n  Report saved to: {report_path}")
    print("\n============================================")


if __name__ == "__main__":
    main()
