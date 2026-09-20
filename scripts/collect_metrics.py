#!/usr/bin/env python3
"""
Collect every metrics.json under an outputs root into one CSV.

One row per run, one column per metric. Column naming mirrors
``flatten_wandb_metrics`` in scripts/evaluate.py, which is also what produced
the WandB exports in docs/ExtendingDreamCatalystPFC/*.csv:

    eval/<Metric>          mean (or the scalar, for non-mean/std metrics)
    eval/<Metric>_std      std, when the metric records one
    eval/<Metric>_<i>      element i, for vector metrics (the EMV3D bboxes)

so a CSV produced here drops into the same tables as a WandB export.

Usage:
  python scripts/collect_metrics.py
  python scripts/collect_metrics.py --outputs-root outputs --out metrics_summary.csv
  python scripts/collect_metrics.py --include 'clown|bear' --sort-by Name
  python scripts/collect_metrics.py --wandb-style --out wandb_like.csv

  # Sanity check that re-evaluation did not move the pre-existing numbers:
  #   collect the snapshots reevaluate_all.sh left behind, then diff the
  #   shared columns against the fresh CSV.
  python scripts/collect_metrics.py --metrics-filename metrics.prev.json \
      --out metrics_before.csv

Reads nothing but JSON: no torch, no CLIP, no GPU.
"""

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re
import sys


# Paper/table order. Anything not listed still shows up, alphabetically, after
# these. The new region-restricted metrics sit next to the global
# CLIP_direction they are meant to explain, never replacing it in the export.
PREFERRED_METRIC_ORDER = [
    "CLIP_direction",
    "CLIP_direction_local",
    "CLIP_direction_bg",
    "Edit_Localization_Ratio",
    "Edit_Magnitude_in_region",
    "Edit_Magnitude_out_region",
    "Background_LPIPS",
    "Background_PSNR",
    "CLIP_img_sim",
    "SSIM",
    "LPIPS",
    "CLIP_text_sim",
    "MultiView_pairwise_cos_sim",
    "MultiView_consistency_std",
    "Region_valid_views",
    "Region_mask_mean_coverage",
    "EditMaskVariance_3D",
    "EditMaskVariance_3D_normalized",
]

# Provenance of the region mask, from the top-level "region" block that
# evaluate.py records. Worth carrying into the CSV: the phrase the segmenter was
# given is the first thing to check when a localized number looks odd.
REGION_INFO_KEYS = [
    "region_phrase",
    "region_phrase_source",
    # The anchor and the two candidate phrases are the audit trail for "is this
    # region even the right object". Segmenting the SOURCE for a target concept
    # that the source does not contain is the main way this suite can go wrong,
    # and region_source_phrase / region_target_phrase show at a glance which
    # query was actually used.
    "region_anchor",
    "region_source_phrase",
    "region_target_phrase",
    "region_mask_source",
    "clipseg_model",
    "region_norm_quantile",
    "region_binarize_threshold",
    "region_min_prob_scale",
    "region_dilate_fraction",
    "region_feather_fraction",
    "region_prob_mean",
]

IDENTITY_COLUMNS = [
    "Name",
    "method",
    "timestamp",
    "num_views",
    "src_prompt",
    "tgt_prompt",
    "run_dir",
    "metrics_path",
]


def flatten_metrics(metrics, num_views, keep_non_numeric=True):
    """Flatten a metrics dict into ``eval/``-prefixed scalar columns.

    Mirrors ``flatten_wandb_metrics`` in scripts/evaluate.py so the columns
    match the WandB exports. The one deliberate difference: non-numeric values
    (the EMV3D bbox-source string) are kept by default, because a CSV can hold
    them and they say whether two runs share a voxel partition. ``--wandb-style``
    drops them again.
    """
    flat = {}
    if num_views is not None:
        flat["eval/num_views"] = num_views
    if not isinstance(metrics, dict):
        return flat

    for key, value in metrics.items():
        if isinstance(value, dict) and "mean" in value:
            flat["eval/%s" % key] = value["mean"]
            if "std" in value:
                flat["eval/%s_std" % key] = value["std"]
            # How many views the mean is over. Only the region-restricted
            # metrics carry it, and only they need it: they are averaged over
            # the views that had a usable region, while CLIP_direction is
            # averaged over every view. When this column is below num_views,
            # the local and global numbers describe different view sets.
            if "n" in value:
                flat["eval/%s_n" % key] = value["n"]
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            flat["eval/%s" % key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        ):
            for i, item in enumerate(value):
                flat["eval/%s_%d" % (key, i)] = item
        elif keep_non_numeric and isinstance(value, str):
            flat["eval/%s" % key] = value
    return flat


def derive_identity(metrics_path, payload):
    """Recover <experiment>/<method>/<timestamp> for a run.

    Prefers the ``config`` path recorded inside metrics.json, so results
    collected into a separate OUT_ROOT still report the run they came from.
    The experiment name is the same one edit.sh passes as --experiment-name,
    which is what the WandB "Name" column holds.
    """
    run_dir = None
    config = payload.get("config") if isinstance(payload, dict) else None
    if isinstance(config, str) and config.strip():
        run_dir = PurePosixPath(config.strip().replace("\\", "/")).parent
    if run_dir is None or str(run_dir) in ("", "."):
        run_dir = PurePosixPath(metrics_path.parent.as_posix())

    parts = run_dir.parts
    timestamp = parts[-1] if len(parts) >= 1 else ""
    method = parts[-2] if len(parts) >= 2 else ""
    name = parts[-3] if len(parts) >= 3 else timestamp
    return name, method, timestamp, str(run_dir)


def collect_rows(root, metrics_filename, include_re, exclude_re, keep_non_numeric):
    rows = []
    failures = []
    for metrics_path in sorted(root.rglob(metrics_filename)):
        key_path = metrics_path.as_posix()
        if include_re and not include_re.search(key_path):
            continue
        if exclude_re and exclude_re.search(key_path):
            continue

        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append((key_path, str(exc)))
            continue
        if not isinstance(payload, dict) or "metrics" not in payload:
            failures.append((key_path, "no 'metrics' object"))
            continue

        name, method, timestamp, run_dir = derive_identity(metrics_path, payload)
        row = {
            "Name": name,
            "method": method,
            "timestamp": timestamp,
            "num_views": payload.get("num_views", ""),
            "src_prompt": payload.get("src_prompt", ""),
            "tgt_prompt": payload.get("tgt_prompt", ""),
            "run_dir": run_dir,
            "metrics_path": key_path,
        }
        row.update(
            flatten_metrics(
                payload.get("metrics"),
                payload.get("num_views"),
                keep_non_numeric=keep_non_numeric,
            )
        )
        region = payload.get("region")
        if keep_non_numeric and isinstance(region, dict):
            for key in REGION_INFO_KEYS:
                if key in region and region[key] is not None:
                    row["region/%s" % key] = region[key]
        rows.append(row)

    return rows, failures


def disambiguate_names(rows):
    """Make Name unique, so a repeated experiment name cannot silently collide."""
    counts = {}
    for row in rows:
        counts[row["Name"]] = counts.get(row["Name"], 0) + 1
    for row in rows:
        if counts.get(row["Name"], 0) > 1 and row.get("timestamp"):
            row["Name"] = "%s/%s" % (row["Name"], row["timestamp"])
    return rows


def order_columns(rows, wandb_style, include_std):
    present = set()
    for row in rows:
        present.update(row.keys())

    columns = []
    if wandb_style:
        columns.append("Name")
    else:
        columns.extend([c for c in IDENTITY_COLUMNS if c in present])

    used = set(columns)
    if not wandb_style:
        for key in REGION_INFO_KEYS:
            column = "region/%s" % key
            if column in present:
                columns.append(column)
                used.add(column)

    for key in PREFERRED_METRIC_ORDER:
        for candidate in ("eval/%s" % key, "eval/%s_std" % key):
            if candidate.endswith("_std") and not include_std:
                continue
            if candidate in present and candidate not in used:
                columns.append(candidate)
                used.add(candidate)

    rest = []
    for column in present:
        if column in used or not column.startswith("eval/"):
            continue
        if column.endswith("_std") and not include_std:
            continue
        if not wandb_style and column == "eval/num_views":
            continue  # already reported as the num_views identity column
        rest.append(column)
    columns.extend(sorted(rest))
    return columns


def main():
    parser = argparse.ArgumentParser(
        description="Collect every metrics.json under an outputs root into one CSV."
    )
    parser.add_argument("--outputs-root", type=str, default="outputs",
                        help="Root directory to walk (default: outputs)")
    parser.add_argument("--out", type=str, default="metrics_summary.csv",
                        help="CSV to write, or '-' for stdout (default: metrics_summary.csv)")
    parser.add_argument("--metrics-filename", type=str, default="metrics.json",
                        help="Filename to collect (default: metrics.json). Use "
                             "metrics.prev.json to read the snapshots that "
                             "reevaluate_all.sh takes before it overwrites a run.")
    parser.add_argument("--include", type=str, default=None,
                        help="Only paths matching this regex")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Drop paths matching this regex")
    parser.add_argument("--wandb-style", action="store_true",
                        help="Emit only Name + numeric eval/* columns, matching "
                             "the WandB exports in docs/ExtendingDreamCatalystPFC/")
    parser.add_argument("--no-std", action="store_true",
                        help="Drop the *_std columns")
    parser.add_argument("--sort-by", type=str, default="Name",
                        help="Column to sort rows by (default: Name)")
    parser.add_argument("--quote-minimal", action="store_true",
                        help="Quote only when needed (default quotes every field, "
                             "like the existing CSV exports)")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    if not root.is_dir():
        print("ERROR: outputs root '%s' not found." % root, file=sys.stderr)
        return 2

    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    rows, failures = collect_rows(
        root,
        args.metrics_filename,
        include_re,
        exclude_re,
        keep_non_numeric=not args.wandb_style,
    )

    for path, reason in failures:
        print("WARNING: skipping %s (%s)" % (path, reason), file=sys.stderr)

    if not rows:
        print("No %s found under %s." % (args.metrics_filename, root), file=sys.stderr)
        return 1

    rows = disambiguate_names(rows)
    columns = order_columns(rows, args.wandb_style, include_std=not args.no_std)

    sort_key = args.sort_by if args.sort_by in columns else "Name"
    rows.sort(key=lambda r: str(r.get(sort_key, "")))

    quoting = csv.QUOTE_MINIMAL if args.quote_minimal else csv.QUOTE_ALL
    if args.out == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore",
                                restval="", quoting=quoting, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    else:
        out_path = Path(args.out)
        if out_path.parent and str(out_path.parent) not in ("", "."):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore",
                                    restval="", quoting=quoting, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote %d run(s) x %d column(s) to %s" % (len(rows), len(columns), out_path))

        missing = [
            key for key in
            ("CLIP_direction_local", "CLIP_direction_bg", "Edit_Localization_Ratio",
             "Background_LPIPS", "Background_PSNR", "Edit_Magnitude_in_region")
            if "eval/%s" % key not in columns
        ]
        if missing:
            print("NOTE: no run carries %s yet. Re-evaluate with "
                  "scripts/reevaluate_all.sh to add them." % ", ".join(missing))

    return 0


if __name__ == "__main__":
    sys.exit(main())
