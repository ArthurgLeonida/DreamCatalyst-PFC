#!/usr/bin/env python3
"""
Environment diagnostics for DreamCatalyst-NS.

Usage:
    python scripts/diagnose_env.py

Run this on BOTH your laptop and VLAB server, then compare outputs.
"""

import subprocess
import sys
import os
from pathlib import Path


def run(cmd: str) -> tuple[str, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip()


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. System ────────────────────────────────────────────────────────────────
section("System")
out, _ = run("uname -a")
print(f"OS:       {out or sys.platform}")
out, _ = run("hostname")
print(f"Hostname: {out}")

# ── 2. Python & PyTorch ──────────────────────────────────────────────────────
section("Python & PyTorch")
print(f"Python: {sys.version}")
try:
    import torch
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # Safely get CUDA version — avoid submodule attribute quirks
        try:
            cuda_ver = torch.version.cuda  # type: ignore[attr-defined]
        except Exception:
            try:
                import torch.version as _tv
                cuda_ver = getattr(_tv, "cuda", "unknown")
            except Exception:
                cuda_ver = "unknown"
        print(f"CUDA version:   {cuda_ver}")
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: NOT INSTALLED")
except Exception as e:
    print(f"PyTorch error: {e}")

# ── 3. NumPy ─────────────────────────────────────────────────────────────────
section("NumPy")
try:
    import numpy as np
    ver = getattr(np, "__version__", None) or getattr(getattr(np, "version", None), "version", "unknown")
    print(f"NumPy: {ver}")
    if ver != "unknown" and int(ver.split(".")[0]) >= 2:
        print("  WARNING: NumPy >= 2 — may cause compatibility issues!")
    else:
        print("  ✓ NumPy < 2 (correct)")
except ImportError:
    print("NumPy: NOT INSTALLED")
except Exception as e:
    print(f"NumPy error: {e}")

# ── 4. COLMAP ────────────────────────────────────────────────────────────────
section("COLMAP")
path_out, _ = run("which colmap")
print(f"Binary path: {path_out or 'NOT FOUND'}")

# Try to get version
ver_out, ver_err = run("colmap version 2>&1")
help_out, help_err = run("colmap help 2>&1 | head -5")
print(f"colmap version output:\n  {ver_out or ver_err}")
print(f"colmap help (first 5 lines):\n  {help_out}")

# List available commands to spot API differences
cmd_out, _ = run("colmap help 2>&1 | grep -E 'feature_extractor|exhaustive_matcher|vocab_tree'")
print(f"Key commands available:\n  {cmd_out or '(none matched)'}")

# ── 5. COLMAP GPU support ────────────────────────────────────────────────────
section("COLMAP GPU / SIFT Config")
# Try feature extraction with invalid args to see what COLMAP says about GPU
out, err = run("colmap feature_extractor --help 2>&1 | grep -iE 'gpu|use_gpu|cuda' | head -10")
print(f"GPU-related flags in feature_extractor:\n{out or '(none found)'}")

# ── 6. FFmpeg ────────────────────────────────────────────────────────────────
section("FFmpeg")
path_out, _ = run("which ffmpeg")
print(f"Binary path: {path_out or 'NOT FOUND'}")
out, _ = run("ffmpeg -version 2>&1 | head -3")
print(f"Version:\n  {out}")

# ── 7. Conda env ─────────────────────────────────────────────────────────────
section("Conda Environment")
out, _ = run("conda info --envs 2>&1")
print(out)

out, _ = run("conda list 2>&1 | grep -iE 'colmap|ffmpeg|nerfstudio|gsplat|torch|numpy|cuda|diffusers|transformers'")
print(f"Key conda packages:\n{out or '(nothing matched — try: conda list)'}")

# ── 8. Nerfstudio ────────────────────────────────────────────────────────────
section("Nerfstudio")
try:
    from importlib.metadata import version as pkg_version
    print(f"Nerfstudio: {pkg_version('nerfstudio')}")
except Exception:
    print("Nerfstudio: not found via importlib")

# 'which' works on Linux; on Windows use 'where'
which_cmd = "which" if sys.platform != "win32" else "where"
path_out, _ = run(f"{which_cmd} ns-process-data 2>&1")
print(f"ns-process-data path: {path_out or 'NOT FOUND'}")
path_out, _ = run(f"{which_cmd} ns-train 2>&1")
print(f"ns-train path:        {path_out or 'NOT FOUND'}")

# Check the colmap_utils.py API naming
ns_root = Path(__file__).parent.parent / "nerfstudio"
utils_files = list(ns_root.rglob("colmap_utils.py"))
print(f"\ncolmap_utils.py files found: {len(utils_files)}")
for f in utils_files:
    content = f.read_text()
    has_sift = "SiftExtraction" in content
    has_feature = "FeatureExtraction" in content
    has_exhaustive = "exhaustive" in content.lower()
    print(f"  {f}")
    print(f"    SiftExtraction:    {has_sift}")
    print(f"    FeatureExtraction: {has_feature}")
    print(f"    exhaustive ref:    {has_exhaustive}")

# ── 9. Image data check ──────────────────────────────────────────────────────
section("Dataset Check (data/Hero/images)")
hero_dir = Path(__file__).parent.parent / "data" / "Hero" / "images"
if hero_dir.exists():
    imgs = sorted(hero_dir.glob("*"))
    supported = [f for f in imgs if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    other = [f for f in imgs if f not in supported]
    print(f"Total files:     {len(imgs)}")
    print(f"Supported (jpg/jpeg/png): {len(supported)}")
    print(f"Unsupported:     {len(other)}")
    if other:
        print(f"  Unsupported files: {[f.name for f in other[:5]]}")
    if supported:
        # Check image sizes
        try:
            from PIL import Image
            sizes = set()
            for p in supported[:5]:
                with Image.open(p) as im:
                    sizes.add(im.size)
            print(f"Sample sizes (first 5): {sizes}")
        except ImportError:
            print("  (Pillow not available for size check)")
else:
    print(f"Directory not found: {hero_dir}")

# ── 10. COLMAP output check ──────────────────────────────────────────────────
section("COLMAP Output Check (data/Hero_processed)")
hero_proc = Path(__file__).parent.parent / "data" / "Hero_processed"
if hero_proc.exists():
    transforms = hero_proc / "transforms.json"
    if transforms.exists():
        import json
        d = json.load(open(transforms))
        print(f"transforms.json: ✓  ({len(d.get('frames', []))} frames)")
    else:
        print("transforms.json: ✗ NOT FOUND (COLMAP may have failed)")

    sparse_dir = hero_proc / "sparse"
    colmap_db = hero_proc / "colmap" / "database.db"
    for p in [sparse_dir, colmap_db]:
        exists = "✓" if p.exists() else "✗"
        print(f"  {exists} {p}")

    # Check if COLMAP produced any models
    for sparse_model in ["0", "1"]:
        model_dir = hero_proc / "sparse" / sparse_model
        if model_dir.exists():
            files = list(model_dir.glob("*"))
            print(f"  sparse/{sparse_model}/: {[f.name for f in files]}")
else:
    print(f"Directory not found: {hero_proc}")
    print("  → COLMAP has not been run yet on Hero scene.")

print("\n" + "="*60)
print("  Diagnostics complete. Compare this output with your laptop.")
print("="*60 + "\n")
