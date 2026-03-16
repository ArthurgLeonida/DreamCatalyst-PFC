#!/usr/bin/env python3
"""
Evaluate a DreamCatalyst editing experiment.

Renders all views from a trained checkpoint, computes metrics against
the original (unedited) images, and saves results to JSON.

Metrics:
  - CLIP_text_sim:   cosine similarity between edited image and target prompt
  - CLIP_direction:  directional CLIP similarity (editing faithfulness)
  - CLIP_img_sim:    cosine similarity between original and edited image (identity)
  - SSIM:            structural similarity (identity preservation)
  - LPIPS:           perceptual distance (lower = more similar to original)
  - Multi-view consistency: std of per-view CLIP embeddings (lower = more consistent)

Usage:
  python scripts/evaluate.py \
      --config outputs/bicycle/dc_splat/<timestamp>/config.yml \
      --src-prompt "a photo of a bicycle" \
      --tgt-prompt "a photo of a motorcycle" \
      [--output-dir eval_results/bicycle_exp001]

  # Compare multiple experiments:
  python scripts/evaluate.py --compare eval_results/exp1 eval_results/exp2 ...
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def load_clip_model(device):
    """Load CLIP model for text-image similarity."""
    import clip
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess


def clip_encode_image(model, images, device):
    """Encode a batch of PIL images with CLIP. Images in [0,1] tensor [B,C,H,W]."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    images = F.interpolate(images.float(), size=224, mode="bicubic", align_corners=False)
    images = (images - mean) / std
    with torch.no_grad():
        features = model.encode_image(images)
    return features / features.norm(dim=1, keepdim=True)


def clip_encode_text(model, texts, device):
    """Encode text prompts with CLIP."""
    import clip
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    return features / features.norm(dim=1, keepdim=True)


def compute_ssim(img1, img2):
    """Compute SSIM between two [H,W,3] float32 numpy arrays in [0,1]."""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)


def compute_lpips(img1_tensor, img2_tensor, lpips_model):
    """Compute LPIPS between two [1,C,H,W] tensors in [0,1]."""
    # LPIPS expects [-1, 1]
    with torch.no_grad():
        return lpips_model(img1_tensor * 2 - 1, img2_tensor * 2 - 1).item()


def render_all_views(config_path, device):
    """Load a nerfstudio checkpoint and render all training views.
    Returns list of (rendered_image_tensor, gt_image_tensor) pairs.
    rendered images are [1,C,H,W] in [0,1].
    """
    import yaml

    # Load config to find data path and model
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Use ns-process to load the pipeline
    from nerfstudio.utils.eval_utils import eval_setup
    _, pipeline, _, _ = eval_setup(Path(config_path))
    pipeline.eval()

    rendered_images = []
    gt_images = []
    image_names = []

    # Get all cameras from the datamanager
    dataparser_outputs = pipeline.datamanager.dataparser.get_dataparser_outputs(
        split="train"
    )
    dataset = pipeline.datamanager.train_dataset

    for i in range(len(dataset)):
        camera = dataset.cameras[i : i + 1].to(device)

        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        # Rendered image: [H, W, 3]
        rendered = outputs["rgb"].cpu()
        # GT image: [H, W, 3]
        gt = dataset[i]["image"].cpu()

        # Convert to [1, C, H, W]
        rendered_tensor = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        gt_tensor = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

        rendered_images.append(rendered_tensor)
        gt_images.append(gt_tensor)

        fname = Path(dataset.image_filenames[i]).stem if hasattr(dataset, "image_filenames") else f"view_{i:04d}"
        image_names.append(fname)

    pipeline.train()
    return rendered_images, gt_images, image_names, pipeline


def evaluate_experiment(config_path, src_prompt, tgt_prompt, output_dir, device="cuda"):
    """Run full evaluation on a single experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered").mkdir(exist_ok=True)

    print(f"Loading checkpoint from {config_path}...")
    rendered_images, gt_images, image_names, pipeline = render_all_views(config_path, device)
    num_views = len(rendered_images)
    print(f"Rendered {num_views} views.")

    # Save rendered images
    for name, rendered in zip(image_names, rendered_images):
        img = (rendered.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(output_dir / f"rendered/{name}.png")

    # ── LPIPS ──
    print("Computing LPIPS...")
    import lpips as lpips_lib
    lpips_model = lpips_lib.LPIPS(net="vgg").to(device)
    lpips_scores = []
    for rendered, gt in zip(rendered_images, gt_images):
        # Resize gt to match rendered if needed
        if gt.shape[2:] != rendered.shape[2:]:
            gt = F.interpolate(gt, size=rendered.shape[2:], mode="bilinear", align_corners=False)
        score = compute_lpips(rendered.to(device), gt.to(device), lpips_model)
        lpips_scores.append(score)
    del lpips_model
    torch.cuda.empty_cache()

    # ── SSIM ──
    print("Computing SSIM...")
    ssim_scores = []
    for rendered, gt in zip(rendered_images, gt_images):
        r_np = rendered.squeeze(0).permute(1, 2, 0).numpy()
        if gt.shape[2:] != rendered.shape[2:]:
            gt = F.interpolate(gt, size=rendered.shape[2:], mode="bilinear", align_corners=False)
        g_np = gt.squeeze(0).permute(1, 2, 0).numpy()
        ssim_scores.append(compute_ssim(r_np, g_np))

    # ── CLIP metrics ──
    print("Computing CLIP metrics...")
    clip_model, _ = load_clip_model(device)

    # Encode text prompts
    src_text_feat = clip_encode_text(clip_model, [src_prompt], device)
    tgt_text_feat = clip_encode_text(clip_model, [tgt_prompt], device)

    clip_text_sims = []      # edited image vs target text
    clip_directions = []     # directional similarity
    clip_img_sims = []       # original vs edited image similarity
    edited_features_all = [] # for multi-view consistency

    for rendered, gt in zip(rendered_images, gt_images):
        edited_feat = clip_encode_image(clip_model, rendered.to(device), device)
        if gt.shape[2:] != rendered.shape[2:]:
            gt = F.interpolate(gt, size=rendered.shape[2:], mode="bilinear", align_corners=False)
        orig_feat = clip_encode_image(clip_model, gt.to(device), device)

        # Text similarity: edited image vs target prompt
        clip_text_sims.append(
            F.cosine_similarity(edited_feat, tgt_text_feat).item()
        )

        # Directional similarity: (img_edit - img_orig) vs (text_tgt - text_src)
        img_delta = edited_feat - orig_feat
        text_delta = tgt_text_feat - src_text_feat
        if img_delta.norm() > 1e-8 and text_delta.norm() > 1e-8:
            clip_directions.append(
                F.cosine_similarity(img_delta, text_delta).item()
            )
        else:
            clip_directions.append(0.0)

        # Image similarity: original vs edited (identity preservation)
        clip_img_sims.append(
            F.cosine_similarity(orig_feat, edited_feat).item()
        )

        edited_features_all.append(edited_feat)

    # ── Multi-view consistency ──
    # Measure how consistent the CLIP embeddings are across views.
    # Lower std = more consistent editing across views.
    all_feats = torch.cat(edited_features_all, dim=0)  # [N, D]
    mv_consistency_std = all_feats.std(dim=0).mean().item()

    # Also compute pairwise cosine similarity mean
    cos_sim_matrix = F.cosine_similarity(
        all_feats.unsqueeze(0), all_feats.unsqueeze(1), dim=2
    )
    # Exclude diagonal
    mask = ~torch.eye(num_views, dtype=torch.bool, device=device)
    mv_pairwise_mean = cos_sim_matrix[mask].mean().item()

    del clip_model
    torch.cuda.empty_cache()

    # ── Aggregate results ──
    results = {
        "config": str(config_path),
        "src_prompt": src_prompt,
        "tgt_prompt": tgt_prompt,
        "num_views": num_views,
        "metrics": {
            "CLIP_text_sim": {
                "mean": float(np.mean(clip_text_sims)),
                "std": float(np.std(clip_text_sims)),
            },
            "CLIP_direction": {
                "mean": float(np.mean(clip_directions)),
                "std": float(np.std(clip_directions)),
            },
            "CLIP_img_sim": {
                "mean": float(np.mean(clip_img_sims)),
                "std": float(np.std(clip_img_sims)),
            },
            "SSIM": {
                "mean": float(np.mean(ssim_scores)),
                "std": float(np.std(ssim_scores)),
            },
            "LPIPS": {
                "mean": float(np.mean(lpips_scores)),
                "std": float(np.std(lpips_scores)),
            },
            "MultiView_consistency_std": float(mv_consistency_std),
            "MultiView_pairwise_cos_sim": float(mv_pairwise_mean),
        },
        "per_view": {
            name: {
                "CLIP_text_sim": float(clip_text_sims[i]),
                "CLIP_direction": float(clip_directions[i]),
                "CLIP_img_sim": float(clip_img_sims[i]),
                "SSIM": float(ssim_scores[i]),
                "LPIPS": float(lpips_scores[i]),
            }
            for i, name in enumerate(image_names)
        },
    }

    # Save
    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Evaluation: {Path(config_path).parent.parent.name}")
    print("=" * 60)
    m = results["metrics"]
    print(f"  CLIP text sim (edit quality):  {m['CLIP_text_sim']['mean']:.4f} +/- {m['CLIP_text_sim']['std']:.4f}")
    print(f"  CLIP direction (edit faithf.): {m['CLIP_direction']['mean']:.4f} +/- {m['CLIP_direction']['std']:.4f}")
    print(f"  CLIP img sim (identity):       {m['CLIP_img_sim']['mean']:.4f} +/- {m['CLIP_img_sim']['std']:.4f}")
    print(f"  SSIM (identity):               {m['SSIM']['mean']:.4f} +/- {m['SSIM']['std']:.4f}")
    print(f"  LPIPS (perceptual dist):       {m['LPIPS']['mean']:.4f} +/- {m['LPIPS']['std']:.4f}")
    print(f"  MV consistency (feat std):     {m['MultiView_consistency_std']:.6f}")
    print(f"  MV pairwise cos sim:           {m['MultiView_pairwise_cos_sim']:.4f}")
    print("=" * 60)
    print(f"  Results saved to: {results_path}")

    return results


def compare_experiments(dirs):
    """Load and compare metrics from multiple experiment directories."""
    all_results = []
    for d in dirs:
        metrics_path = Path(d) / "metrics.json"
        if not metrics_path.exists():
            print(f"WARNING: {metrics_path} not found, skipping.")
            continue
        with open(metrics_path) as f:
            all_results.append(json.load(f))

    if not all_results:
        print("No results to compare.")
        return

    # Print comparison table
    header = f"{'Experiment':<30} {'CLIP_txt':>9} {'CLIP_dir':>9} {'CLIP_img':>9} {'SSIM':>7} {'LPIPS':>7} {'MV_cos':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in all_results:
        name = Path(r["config"]).parent.parent.name
        m = r["metrics"]
        print(
            f"{name:<30} "
            f"{m['CLIP_text_sim']['mean']:>9.4f} "
            f"{m['CLIP_direction']['mean']:>9.4f} "
            f"{m['CLIP_img_sim']['mean']:>9.4f} "
            f"{m['SSIM']['mean']:>7.4f} "
            f"{m['LPIPS']['mean']:>7.4f} "
            f"{m['MultiView_pairwise_cos_sim']:>7.4f}"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Evaluate DreamCatalyst editing experiments")
    subparsers = parser.add_subparsers(dest="command")

    # Evaluate a single experiment
    eval_parser = subparsers.add_parser("eval", help="Evaluate a single experiment")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to config.yml")
    eval_parser.add_argument("--src-prompt", type=str, required=True, help="Source prompt")
    eval_parser.add_argument("--tgt-prompt", type=str, required=True, help="Target prompt")
    eval_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    eval_parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    # Compare multiple experiments
    cmp_parser = subparsers.add_parser("compare", help="Compare multiple experiments")
    cmp_parser.add_argument("dirs", nargs="+", help="Directories containing metrics.json")

    args = parser.parse_args()

    if args.command == "eval":
        evaluate_experiment(args.config, args.src_prompt, args.tgt_prompt, args.output_dir, args.device)
    elif args.command == "compare":
        compare_experiments(args.dirs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
