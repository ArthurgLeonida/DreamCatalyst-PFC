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
  python scripts/evaluate.py eval \
      --config outputs/bicycle/dc_splat/<timestamp>/config.yml \
      --src-prompt "a photo of a bicycle" \
      --tgt-prompt "a photo of a motorcycle" \
      [--output-dir eval_results/bicycle_exp001]

  # Attach the metrics to the original WandB run if available:
  python scripts/evaluate.py eval ... --log-wandb
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml


WANDB_RUN_DIR_RE = re.compile(r"(?:offline-)?run-\d{8}_\d{6}-([a-z0-9]+)", re.IGNORECASE)
WANDB_RUN_FILE_RE = re.compile(r"(?:offline-)?run-([a-z0-9]+)\.wandb$", re.IGNORECASE)


def load_clip_model(device):
    """Load CLIP model for text-image similarity."""
    import clip
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()
    return model


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


@contextmanager
def temporary_env_var(name, value):
    """Temporarily set an environment variable for the duration of a context."""
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def load_experiment_config(config_path):
    """Load the serialized Nerfstudio config object for an experiment."""
    return yaml.load(Path(config_path).read_text(), Loader=yaml.Loader)


def infer_experiment_name(config_path, config=None):
    """Best-effort experiment name, preferring the serialized config."""
    if config is not None:
        experiment_name = getattr(config, "experiment_name", None)
        if experiment_name:
            return experiment_name

    config_path = Path(config_path)
    if len(config_path.parents) >= 3:
        return config_path.parents[2].name
    return config_path.parent.name


def infer_project_name(config=None, default="dreamcatalyst-pfc"):
    """Best-effort WandB project name, preferring the serialized config."""
    if config is not None:
        project_name = getattr(config, "project_name", None)
        if project_name:
            return project_name
    return default


def load_wandb_run_metadata(config_path):
    """Load per-run WandB metadata written during training, when available."""
    metadata_path = Path(config_path).parent / "wandb_run.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:
        print(f"WARNING: Failed to parse {metadata_path}: {exc}")
        return None


def extract_wandb_run_id_from_path(path):
    """Parse a WandB run id from common local directory/file naming schemes."""
    path = Path(path)
    for candidate in (str(path), path.name):
        match = WANDB_RUN_DIR_RE.search(candidate)
        if match:
            return match.group(1)
        match = WANDB_RUN_FILE_RE.search(candidate)
        if match:
            return match.group(1)
    return None


def discover_wandb_run_id(wandb_dir):
    """Best-effort local WandB run id discovery from a run storage directory."""
    if wandb_dir is None:
        return None

    wandb_dir = Path(wandb_dir)
    if not wandb_dir.exists():
        return None

    candidate_paths = []
    if wandb_dir.is_file():
        candidate_paths.append(wandb_dir)
    else:
        patterns = [
            "latest-run",
            "run-*",
            "offline-run-*",
            "**/run-*.wandb",
            "**/offline-run-*.wandb",
            "**/wandb-metadata.json",
            "**/wandb-settings.json",
        ]
        for pattern in patterns:
            candidate_paths.extend(wandb_dir.glob(pattern))

    candidate_paths = sorted(
        {path.resolve() if path.exists() else path for path in candidate_paths},
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )

    for path in candidate_paths:
        run_id = extract_wandb_run_id_from_path(path)
        if run_id:
            return run_id
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            run_id = payload.get("run_id") or payload.get("id")
            if run_id:
                return run_id

    return None


def infer_wandb_dir(config_path):
    """Infer the method-level WandB directory used by edit.sh for this run."""
    candidate = Path(config_path).parent.parent / "wandb"
    return candidate if candidate.exists() else None


def flatten_wandb_metrics(metrics, num_views):
    """Flatten nested metrics into WandB-friendly scalar keys."""
    flattened = {"eval/num_views": num_views}
    for key, value in metrics.items():
        if isinstance(value, dict) and "mean" in value:
            flattened[f"eval/{key}"] = value["mean"]
            if "std" in value:
                flattened[f"eval/{key}_std"] = value["std"]
        else:
            flattened[f"eval/{key}"] = value
    return flattened


def log_results_to_wandb(results, config_path, config, metrics_path, run_id=None, wandb_project=None):
    """Log evaluation metrics into an existing WandB run when possible."""
    try:
        import wandb
    except Exception as exc:
        print(f"WARNING: Failed to import wandb: {exc}")
        return

    project_name = wandb_project or infer_project_name(config)
    experiment_name = infer_experiment_name(config_path, config)
    flattened_metrics = flatten_wandb_metrics(results["metrics"], results["num_views"])

    run = None
    attached = False
    if run_id:
        try:
            print(f"Attaching evaluation metrics to WandB run {run_id}...")
            run = wandb.init(
                project=project_name,
                id=run_id,
                resume="must",
                job_type="evaluation",
            )
            attached = True
        except Exception as exc:
            print(f"WARNING: Failed to attach to WandB run {run_id}: {exc}")

    if run is None:
        run_name = f"{experiment_name}_eval"
        print(f"Logging evaluation metrics to a new WandB run: {run_name}")
        run = wandb.init(
            project=project_name,
            name=run_name,
            job_type="evaluation",
        )

    run.summary["eval/config"] = str(config_path)
    run.summary["eval/metrics_path"] = str(metrics_path)
    run.summary["eval/attached_to_existing_run"] = attached
    for key, value in flattened_metrics.items():
        run.summary[key] = value

    wandb.finish()


def load_pipeline_from_experiment(config_path, device, disable_wandb_during_load=True):
    """Load the edited checkpoint directly from the experiment folder.

    This intentionally avoids Nerfstudio's eval_setup helper because some
    installed versions may reuse the original training load_dir and spin up a
    fresh trainer/W&B session instead of loading the edited run checkpoint.
    """
    from nerfstudio.configs.method_configs import all_methods

    config_path = Path(config_path)
    config = load_experiment_config(config_path)

    # Restore the datamanager target in the same way Nerfstudio eval_setup does.
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target

    # Force evaluation to load the checkpoint produced by this edited run.
    checkpoint_dir = config_path.parent / "nerfstudio_models"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    config.load_dir = checkpoint_dir

    # Be extra defensive about side effects during evaluation.
    if hasattr(config, "vis"):
        config.vis = "tensorboard"

    env_value = "disabled" if disable_wandb_during_load else None
    with temporary_env_var("WANDB_MODE", env_value):
        pipeline = config.pipeline.setup(device=device, test_mode="test")
        pipeline.eval()

        checkpoint_steps = sorted(
            int(path.stem.split("-")[1]) for path in checkpoint_dir.glob("step-*.ckpt")
        )
        if not checkpoint_steps:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        load_step = checkpoint_steps[-1]
        load_path = checkpoint_dir / f"step-{load_step:09d}.ckpt"
        loaded_state = torch.load(load_path, map_location="cpu")
        pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    return pipeline, load_path, load_step, config


def render_all_views(config_path, device, disable_wandb_during_load=True):
    """Load an edited nerfstudio checkpoint and render all training views.
    Returns list of (rendered_image_tensor, gt_image_tensor) pairs.
    rendered images are [1,C,H,W] in [0,1].
    """
    pipeline, checkpoint_path, load_step, config = load_pipeline_from_experiment(
        config_path,
        device,
        disable_wandb_during_load=disable_wandb_during_load,
    )
    print(f"Loaded edited checkpoint from {checkpoint_path} (step {load_step}).")

    rendered_images = []
    gt_images = []
    image_names = []

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

    return rendered_images, gt_images, image_names, config


def evaluate_experiment(
    config_path,
    src_prompt,
    tgt_prompt,
    output_dir,
    device="cuda",
    log_wandb=False,
    wandb_run_id=None,
    wandb_dir=None,
    wandb_project=None,
):
    """Run full evaluation on a single experiment."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered").mkdir(exist_ok=True)

    print(f"Loading checkpoint from {config_path}...")
    rendered_images, gt_images, image_names, config = render_all_views(
        config_path,
        device,
        disable_wandb_during_load=True,
    )
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
    clip_model = load_clip_model(device)

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

    # ── W&B Logging ──
    if log_wandb:
        metadata = load_wandb_run_metadata(config_path) or {}
        resolved_wandb_dir = Path(wandb_dir) if wandb_dir else infer_wandb_dir(config_path)
        resolved_run_id = wandb_run_id or metadata.get("run_id") or discover_wandb_run_id(resolved_wandb_dir)
        resolved_project_name = wandb_project or metadata.get("project") or infer_project_name(config)
        log_results_to_wandb(
            results,
            config_path,
            config,
            results_path,
            run_id=resolved_run_id,
            wandb_project=resolved_project_name,
        )

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Evaluation: {infer_experiment_name(config_path, config)}")
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
    eval_parser.add_argument("--log-wandb", action="store_true", help="Log evaluation metrics to Weights & Biases")
    eval_parser.add_argument("--wandb-run-id", type=str, default=None, help="Attach metrics to an existing WandB run id")
    eval_parser.add_argument("--wandb-dir", type=str, default=None, help="Directory containing local WandB run files")
    eval_parser.add_argument("--wandb-project", type=str, default=None, help="Override the WandB project name")

    args = parser.parse_args()

    if args.command == "eval":
        evaluate_experiment(
            args.config,
            args.src_prompt,
            args.tgt_prompt,
            args.output_dir,
            args.device,
            log_wandb=args.log_wandb,
            wandb_run_id=args.wandb_run_id,
            wandb_dir=args.wandb_dir,
            wandb_project=args.wandb_project,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
