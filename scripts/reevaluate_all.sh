#!/usr/bin/env bash
# ==============================================================================
#  DreamCatalyst-NS - Batch re-evaluation of finished runs
# ==============================================================================
#  Recomputes scripts/evaluate.py metrics for every finished run under an
#  outputs root, WITHOUT retraining anything. Each run checkpoint is re-rendered
#  and its metrics.json is rewritten, which is all that is needed to obtain the
#  region-restricted metrics (CLIP_direction_local, CLIP_direction_bg,
#  Edit_Localization_Ratio, Background_LPIPS, Background_PSNR) on experiments
#  that finished before those metrics existed.
#
#  Usage:
#    bash scripts/reevaluate_all.sh [outputs_root] [extra evaluate.py args...]
#
#  Examples:
#    DRY_RUN=1 bash scripts/reevaluate_all.sh                  # show the plan
#    bash scripts/reevaluate_all.sh                            # re-eval outputs/
#    INCLUDE_REGEX='clown|bear' bash scripts/reevaluate_all.sh
#    FORCE=1 bash scripts/reevaluate_all.sh outputs
#    PARALLEL=2 bash scripts/reevaluate_all.sh
#    REGION_MASK_DIR=data/region_masks bash scripts/reevaluate_all.sh
#    bash scripts/reevaluate_all.sh outputs --region-phrase-mode diff
#
#  Any extra positional flag is forwarded verbatim to evaluate.py, which is how
#  the per-run region knobs are reached (--region-phrase-mode, --clipseg-model,
#  --region-norm-quantile). Avoid --region-prompt in a batch: one hand-written
#  phrase cannot be right for every scene in the sweep.
#
#  Runtime knobs (env):
#    OUTPUTS_ROOT      Root to walk                        (default: outputs)
#    DRY_RUN=1         Print what would run, run nothing   (default: 0)
#    FORCE=1           Re-evaluate even when the new keys  (default: 0)
#                      are already in metrics.json
#    PARALLEL=N        N evaluations at a time, one GPU    (default: 1)
#                      each; clamped to the GPUs available
#    DEVICE            --device passed to evaluate.py      (default: cuda)
#    LIMIT=N           Stop after N eligible runs          (default: 0 = all)
#    INCLUDE_REGEX     Only runs whose path matches        (default: all)
#    EXCLUDE_REGEX     Drop runs whose path matches        (default: none)
#    PROMPT_MAP        JSON prompt fallback, used only     (default: none)
#                      when a run records no prompts
#    OUT_ROOT          Write results to <OUT_ROOT>/<slug>/ (default: in place)
#    LOG_DIR           Per-run stdout logs    (default: <root>/_reevaluate_logs/<ts>)
#    VERBOSE=1         Also tee per-run stdout to console  (default: 0)
#    BACKUP_METRICS=0  Skip the one-time snapshot of the   (default: 1)
#                      old metrics.json to metrics.prev.json
#    REUSE_BBOX=0      Re-derive the EMV3D voxel bbox      (default: 1 = reuse
#                      instead of reusing the recorded one  the recorded bbox)
#    SKIP_MASK_VARIANCE=1  Pass --disable-mask-variance    (default: 0)
#                      NOTE: metrics.json is REWRITTEN, so skipping EMV3D drops
#                      the EditMaskVariance_3D keys from the new file.
#    REGION_MASK_DIR   Precomputed region masks            (default: segmenter)
#    REGION_BINARIZE_THRESHOLD  Binarize the region mask   (default: soft)
#    REGION_ANCHOR     union|source|target: which phrase   (default: evaluate.py
#                      the SOURCE frame is segmented for    default = union)
#    REGION_DILATE_FRAC  Grow R by this fraction of the    (default: 0 = off)
#                      image diagonal. Set it for the scenes whose edit grows
#                      geometry outside the source object (fangzhou helmet,
#                      person->stormtrooper); use INCLUDE_REGEX to give those
#                      scenes their own pass so the value does not leak into
#                      scenes that do not need it.
#    DISABLE_REGION_METRICS=1   Old metrics only           (default: 0)
#    REGION_METRICS_ARGS  Extra raw args for the new metrics
#    LOG_WANDB=1       Push metrics into the WandB run     (default: 0)
#    REQUIRED_KEYS     Metric keys that mark a run as already done
#    DONE_MARKER_KEY   Key that alone proves the region pass already ran
#                      (default: Region_valid_views)
#
#  Extra positional args after the root are forwarded verbatim to evaluate.py.
#
#  Method knobs live in nerfstudio/dc/method_config.py; this script never
#  touches them, it only re-reads finished checkpoints.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

EVAL_SCRIPT="scripts/evaluate.py"

# Positional: [outputs_root] [extra evaluate.py args...]
PASSTHROUGH_ARGS=()
if [ -n "${1:-}" ] && [[ "${1}" != --* ]]; then
    OUTPUTS_ROOT="${1}"
    if [ "$#" -gt 1 ]; then
        PASSTHROUGH_ARGS=("${@:2}")
    fi
else
    OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
    if [ "$#" -gt 0 ]; then
        PASSTHROUGH_ARGS=("${@:1}")
    fi
fi

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
PARALLEL="${PARALLEL:-1}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"
INCLUDE_REGEX="${INCLUDE_REGEX:-}"
EXCLUDE_REGEX="${EXCLUDE_REGEX:-}"
PROMPT_MAP="${PROMPT_MAP:-}"
OUT_ROOT="${OUT_ROOT:-}"
VERBOSE="${VERBOSE:-0}"
BACKUP_METRICS="${BACKUP_METRICS:-1}"
REUSE_BBOX="${REUSE_BBOX:-1}"
SKIP_MASK_VARIANCE="${SKIP_MASK_VARIANCE:-0}"
REGION_MASK_DIR="${REGION_MASK_DIR:-}"
REGION_BINARIZE_THRESHOLD="${REGION_BINARIZE_THRESHOLD:-}"
REGION_ANCHOR="${REGION_ANCHOR:-}"
REGION_DILATE_FRAC="${REGION_DILATE_FRAC:-}"
DISABLE_REGION_METRICS="${DISABLE_REGION_METRICS:-0}"
REGION_METRICS_ARGS="${REGION_METRICS_ARGS:-}"
LOG_WANDB="${LOG_WANDB:-0}"
REQUIRED_KEYS="${REQUIRED_KEYS:-CLIP_direction_local CLIP_direction_bg Edit_Localization_Ratio Background_LPIPS Background_PSNR Edit_Magnitude_in_region}"
# Deliberately a key that ONLY the corrected region pass emits. Region_valid_views
# was also written by the first version of the region code, whose masks could be
# fabricated all-ones maps, so using it here would mark those runs as done and
# silently keep the bad numbers.
DONE_MARKER_KEY="${DONE_MARKER_KEY:-Edit_Magnitude_in_region}"

RUN_STAMP="$(date +%Y-%m-%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${OUTPUTS_ROOT}/_reevaluate_logs/${RUN_STAMP}}"
STATUS_DIR="${LOG_DIR}/status"

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "ERROR: ${EVAL_SCRIPT} not found (repo root resolved to ${REPO_ROOT})."
    exit 1
fi
if [ ! -d "${OUTPUTS_ROOT}" ]; then
    echo "ERROR: outputs root '${OUTPUTS_ROOT}' not found."
    exit 1
fi

# ── Flag capability detection ────────────────────────────────────────────────
# The region metrics land in evaluate.py separately from this runner, so the
# exact flag spellings are discovered from the source rather than assumed. An
# unsupported flag is reported and dropped instead of crashing every run.
eval_supports_flag() {
    grep -qF -- "$1" "${EVAL_SCRIPT}"
}

first_supported_flag() {
    local candidate
    for candidate in "$@"; do
        if eval_supports_flag "${candidate}"; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

REGION_DISABLE_FLAG="$(first_supported_flag --no-localized-metrics --disable-region-metrics --no-region-metrics --skip-region-metrics --disable-localized-metrics || true)"
REGION_ENABLE_FLAG="$(first_supported_flag --region-metrics --enable-region-metrics --with-region-metrics || true)"

REGION_METRICS_SUPPORTED=0
if [ -n "${REGION_DISABLE_FLAG}" ] || [ -n "${REGION_ENABLE_FLAG}" ] \
   || eval_supports_flag --region-mask-dir; then
    REGION_METRICS_SUPPORTED=1
fi

BBOX_FLAG_SUPPORTED=0
if eval_supports_flag --mask-variance-bbox-min; then
    BBOX_FLAG_SUPPORTED=1
elif [ "${REUSE_BBOX}" = "1" ]; then
    echo "WARNING: --mask-variance-bbox-min not supported; the EMV3D voxel grid"
    echo "         will be re-derived per run and will not match the old numbers."
fi

# ── Shared evaluate.py arguments ─────────────────────────────────────────────
COMMON_ARGS=()
if [ "${SKIP_MASK_VARIANCE}" = "1" ]; then
    if eval_supports_flag --disable-mask-variance; then
        COMMON_ARGS+=(--disable-mask-variance)
    else
        echo "WARNING: --disable-mask-variance not supported; EMV3D stays enabled."
    fi
fi
if [ "${LOG_WANDB}" = "1" ]; then
    COMMON_ARGS+=(--log-wandb)
fi

if [ "${DISABLE_REGION_METRICS}" = "1" ]; then
    if [ -n "${REGION_DISABLE_FLAG}" ]; then
        COMMON_ARGS+=("${REGION_DISABLE_FLAG}")
    else
        echo "WARNING: DISABLE_REGION_METRICS=1 but evaluate.py exposes no disable flag."
    fi
else
    if [ -n "${REGION_ENABLE_FLAG}" ] && [ -z "${REGION_DISABLE_FLAG}" ]; then
        # Opt-in style: the new metrics only run when explicitly requested.
        COMMON_ARGS+=("${REGION_ENABLE_FLAG}")
    fi
    if [ -n "${REGION_MASK_DIR}" ]; then
        if eval_supports_flag --region-mask-dir; then
            COMMON_ARGS+=(--region-mask-dir "${REGION_MASK_DIR}")
        else
            echo "WARNING: --region-mask-dir not supported by ${EVAL_SCRIPT}; ignoring."
        fi
    fi
    if [ -n "${REGION_BINARIZE_THRESHOLD}" ]; then
        if eval_supports_flag --region-binarize-threshold; then
            COMMON_ARGS+=(--region-binarize-threshold "${REGION_BINARIZE_THRESHOLD}")
        else
            echo "WARNING: --region-binarize-threshold not supported; ignoring."
        fi
    fi
    if [ -n "${REGION_ANCHOR}" ]; then
        if eval_supports_flag --region-anchor; then
            COMMON_ARGS+=(--region-anchor "${REGION_ANCHOR}")
        else
            echo "WARNING: --region-anchor not supported; ignoring."
        fi
    fi
    if [ -n "${REGION_DILATE_FRAC}" ]; then
        if eval_supports_flag --region-dilate-frac; then
            COMMON_ARGS+=(--region-dilate-frac "${REGION_DILATE_FRAC}")
        else
            echo "WARNING: --region-dilate-frac not supported; ignoring."
        fi
    fi
    if [ -n "${REGION_METRICS_ARGS}" ]; then
        # Deliberately word-split: this knob carries whole flag/value pairs.
        # shellcheck disable=SC2206
        COMMON_ARGS+=(${REGION_METRICS_ARGS})
    fi
fi
if [ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]; then
    COMMON_ARGS+=("${PASSTHROUGH_ARGS[@]}")
fi

# ── Discover runs and resolve prompts ────────────────────────────────────────
# One Python pass over the tree: find run dirs, read back src/tgt prompts, read
# back the recorded EMV3D bbox, and decide already-done vs eligible. Records use
# the ASCII unit/record separators, so prompt text may contain anything else.
discover_runs() {
    OUTPUTS_ROOT="${OUTPUTS_ROOT}" \
    REQUIRED_KEYS="${REQUIRED_KEYS}" \
    DONE_MARKER_KEY="${DONE_MARKER_KEY}" \
    FORCE="${FORCE}" \
    REUSE_BBOX="${REUSE_BBOX}" \
    INCLUDE_REGEX="${INCLUDE_REGEX}" \
    EXCLUDE_REGEX="${EXCLUDE_REGEX}" \
    PROMPT_MAP="${PROMPT_MAP}" \
    OUT_ROOT="${OUT_ROOT}" \
    LIMIT="${LIMIT}" \
    python - <<'PY'
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

import yaml

US = "\x1f"   # field separator
RS = "\x1e"   # record separator

try:  # a non-ASCII prompt under an LC_ALL=C shell would otherwise raise
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

root = Path(os.environ["OUTPUTS_ROOT"])
required = [k for k in os.environ.get("REQUIRED_KEYS", "").split() if k]
done_marker = os.environ.get("DONE_MARKER_KEY", "").strip()
force = os.environ.get("FORCE", "0") == "1"
reuse_bbox = os.environ.get("REUSE_BBOX", "1") == "1"
include = os.environ.get("INCLUDE_REGEX", "").strip()
exclude = os.environ.get("EXCLUDE_REGEX", "").strip()
prompt_map_path = os.environ.get("PROMPT_MAP", "").strip()
out_root = os.environ.get("OUT_ROOT", "").strip()
try:
    limit = int(os.environ.get("LIMIT", "0") or 0)
except ValueError:
    limit = 0


class TolerantLoader(yaml.SafeLoader):
    """Read a Nerfstudio config.yml without importing nerfstudio.

    config.yml is a pyyaml dump of live config objects, so it is full of
    !!python/object tags. Every unknown tag collapses to a plain dict, list or
    scalar, which is all the prompt lookup needs, and it avoids a multi-second
    nerfstudio import per run.
    """


def _any_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        try:
            return loader.construct_mapping(node, deep=True)
        except Exception:
            return {}
    if isinstance(node, yaml.SequenceNode):
        try:
            return loader.construct_sequence(node, deep=True)
        except Exception:
            return []
    try:
        return loader.construct_scalar(node)
    except Exception:
        return None


TolerantLoader.add_multi_constructor("", _any_tag)

PROMPT_RE = re.compile(r"^\s*(src_prompt|tgt_prompt)\s*:\s*(.+?)\s*$")


def _clean_scalar(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return text.strip()


def prompts_from_yaml_text(text):
    """Regex fallback for a config.yml that pyyaml refuses to parse."""
    found = {}
    for line in text.splitlines():
        match = PROMPT_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), _clean_scalar(match.group(2))
        if value and key not in found:
            found[key] = value
    if "src_prompt" in found and "tgt_prompt" in found:
        return found["src_prompt"], found["tgt_prompt"]
    return None


def prompts_from_config(config_path):
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    try:
        cfg = yaml.load(text, Loader=TolerantLoader)
    except Exception:
        cfg = None

    if isinstance(cfg, dict):
        # Preferred location first: pipeline.dc.{src,tgt}_prompt.
        node = cfg.get("pipeline")
        node = node.get("dc") if isinstance(node, dict) else None
        if isinstance(node, dict):
            src, tgt = node.get("src_prompt"), node.get("tgt_prompt")
            if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
                return src.strip(), tgt.strip()

        # Otherwise the first node carrying both, breadth-first.
        queue = deque([cfg])
        visited = 0
        while queue and visited < 20000:
            current = queue.popleft()
            visited += 1
            if isinstance(current, dict):
                src, tgt = current.get("src_prompt"), current.get("tgt_prompt")
                if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
                    return src.strip(), tgt.strip()
                queue.extend(current.values())
            elif isinstance(current, (list, tuple)):
                queue.extend(current)

    return prompts_from_yaml_text(text)


def load_prompt_map(path):
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print("WARNING: could not read PROMPT_MAP %s: %s" % (path, exc), file=sys.stderr)
        return {}
    if not isinstance(payload, dict):
        print("WARNING: PROMPT_MAP %s is not a JSON object; ignoring." % path, file=sys.stderr)
        return {}

    table = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            src = value.get("src") or value.get("src_prompt")
            tgt = value.get("tgt") or value.get("tgt_prompt")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            src, tgt = value
        else:
            continue
        if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
            table[str(key)] = (src.strip(), tgt.strip())
    return table


def prompts_from_map(table, run_dir, experiment_name):
    if not table:
        return None
    if experiment_name in table:
        return table[experiment_name]
    key_path = run_dir.as_posix()
    # Longest matching key wins, so a per-rep key beats a per-scene key.
    best = None
    for key, value in table.items():
        if key and key in key_path and (best is None or len(key) > len(best[0])):
            best = (key, value)
    return best[1] if best else None


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def bbox_field(metrics, key):
    value = metrics.get(key) if isinstance(metrics, dict) else None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            # Fixed-point, never exponent form: argparse only accepts a leading
            # "-" as a value when it matches its negative-number pattern, so a
            # repr like "-1e-05" would be parsed as an option and break the run.
            return " ".join(format(float(v), ".10f") for v in value)
        except (TypeError, ValueError):
            return ""
    return ""


def slugify(run_dir):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.as_posix()).strip("_")
    return slug or "run"


prompt_table = load_prompt_map(prompt_map_path)
include_re = re.compile(include) if include else None
exclude_re = re.compile(exclude) if exclude else None

run_dirs = []
for config_path in root.rglob("config.yml"):
    run_dir = config_path.parent
    if not (run_dir / "nerfstudio_models").is_dir():
        continue
    run_dirs.append(run_dir)
run_dirs.sort(key=lambda p: p.as_posix())

emitted = 0
for run_dir in run_dirs:
    key_path = run_dir.as_posix()
    if include_re and not include_re.search(key_path):
        continue
    if exclude_re and exclude_re.search(key_path):
        continue

    parts = run_dir.parts
    experiment_name = parts[-3] if len(parts) >= 3 else run_dir.name

    out_dir = Path(out_root) / slugify(run_dir) if out_root else run_dir

    run_metrics = read_json(run_dir / "metrics.json")
    out_metrics = run_metrics if out_dir == run_dir else read_json(out_dir / "metrics.json")

    # Prompts: the run's own metrics.json, then its config.yml, then the map.
    prompts = None
    source = "-"
    if isinstance(run_metrics, dict):
        src, tgt = run_metrics.get("src_prompt"), run_metrics.get("tgt_prompt")
        if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
            prompts = (src.strip(), tgt.strip())
            source = "metrics.json"
    if prompts is None:
        prompts = prompts_from_config(run_dir / "config.yml")
        if prompts is not None:
            source = "config.yml"
    if prompts is None:
        prompts = prompts_from_map(prompt_table, run_dir, experiment_name)
        if prompts is not None:
            source = "prompt-map"

    bbox_min = bbox_max = ""
    if reuse_bbox:
        # The run's own metrics.json first, then whatever sits in the output
        # dir: with OUT_ROOT set, or when the run was evaluated by hand into
        # eval_results/, the run dir holds no metrics.json at all and the only
        # recorded bbox is the one under out_dir. A run with no bbox anywhere
        # falls back to a per-run grid, which the caller warns about, because
        # a re-derived grid makes EditMaskVariance_3D incomparable with the
        # number the run originally reported.
        for candidate in (run_metrics, out_metrics):
            if not isinstance(candidate, dict):
                continue
            metrics = candidate.get("metrics", {})
            bbox_min = bbox_field(metrics, "EditMaskVariance_3D_bbox_min")
            bbox_max = bbox_field(metrics, "EditMaskVariance_3D_bbox_max")
            if bbox_min and bbox_max:
                break
            bbox_min = bbox_max = ""

    if prompts is None:
        status = "skip_noprompt"
        reason = "no src/tgt prompt in metrics.json or config.yml"
        src = tgt = ""
    else:
        src, tgt = prompts
        reason = ""
        done = False
        if isinstance(out_metrics, dict):
            have = out_metrics.get("metrics", {})
            if isinstance(have, dict):
                # evaluate.py drops a localized entry whose every view was
                # degenerate, so requiring all five keys would re-run such a
                # run forever. The marker key is written whenever the region
                # pass ran at all, and it settles that case.
                if done_marker and done_marker in have:
                    done = True
                elif required:
                    done = all(k in have for k in required)
        if done and not force:
            status = "skip_done"
            reason = "new metric keys already present"
        else:
            status = "ok"

    if status == "ok":
        if limit > 0 and emitted >= limit:
            # Emitted rather than dropped, so the printed buckets still add up
            # to the number of runs that were walked.
            status = "skip_limit"
            reason = "LIMIT=%d already reached" % limit
        else:
            emitted += 1

    record = US.join([
        run_dir.as_posix(),
        out_dir.as_posix(),
        slugify(run_dir),
        status,
        reason,
        source,
        src,
        tgt,
        bbox_min,
        bbox_max,
    ])
    sys.stdout.write(record + RS)
PY
}

# ── Banner ───────────────────────────────────────────────────────────────────
echo "============================================"
echo " Batch re-evaluation (no retraining)"
echo " Root:        ${OUTPUTS_ROOT}"
echo " Eval script: ${EVAL_SCRIPT}"
if [ "${REGION_METRICS_SUPPORTED}" = "1" ]; then
    if [ "${DISABLE_REGION_METRICS}" = "1" ]; then
        echo " Region:      disabled by request (${REGION_DISABLE_FLAG:-no flag})"
    elif [ -n "${REGION_MASK_DIR}" ]; then
        echo " Region:      precomputed masks from ${REGION_MASK_DIR}"
    else
        echo " Region:      segmenter (masks cached per run under region_masks/)"
    fi
else
    echo " Region:      NOT SUPPORTED by ${EVAL_SCRIPT} yet - old metrics only"
fi
if [ "${REUSE_BBOX}" = "1" ]; then
    echo " EMV3D bbox:  reuse the bbox recorded in each metrics.json"
else
    echo " EMV3D bbox:  re-derive per run (NOT comparable with the old numbers)"
fi
if [ -n "${OUT_ROOT}" ]; then
    echo " Out:         ${OUT_ROOT}/<slug>"
else
    echo " Out:         in place (metrics.json is rewritten)"
fi
echo " Logs:        ${LOG_DIR}"
echo " Parallel:    ${PARALLEL}"
echo " Dry run:     ${DRY_RUN}"
echo "============================================"

# ── Collect the plan ─────────────────────────────────────────────────────────
RUN_DIRS=()
OUT_DIRS=()
SLUGS=()
SRCS=()
TGTS=()
BBOX_MINS=()
BBOX_MAXS=()
SOURCES=()
SKIPPED_DONE=()
SKIPPED_NOPROMPT=()
SKIPPED_LIMIT=()
NO_BBOX=()

# Materialize the plan first: a process substitution would swallow a discovery
# failure and make an empty plan look like "nothing to do".
PLAN_FILE="$(mktemp -t reeval-plan-XXXXXX)"
trap 'rm -f "${PLAN_FILE}"' EXIT
if ! discover_runs > "${PLAN_FILE}"; then
    echo "ERROR: run discovery failed; see the traceback above."
    exit 1
fi

while IFS=$'\x1f' read -r -d $'\x1e' r_run r_out r_slug r_status r_reason r_source r_src r_tgt r_bmin r_bmax; do
    case "${r_status}" in
        ok)
            RUN_DIRS+=("${r_run}")
            OUT_DIRS+=("${r_out}")
            SLUGS+=("${r_slug}")
            SRCS+=("${r_src}")
            TGTS+=("${r_tgt}")
            BBOX_MINS+=("${r_bmin}")
            BBOX_MAXS+=("${r_bmax}")
            SOURCES+=("${r_source}")
            if [ -z "${r_bmin}" ] || [ -z "${r_bmax}" ]; then
                NO_BBOX+=("${r_run}")
            fi
            ;;
        skip_done)
            SKIPPED_DONE+=("${r_run}")
            ;;
        skip_noprompt)
            SKIPPED_NOPROMPT+=("${r_run} (${r_reason})")
            ;;
        skip_limit)
            SKIPPED_LIMIT+=("${r_run}")
            ;;
    esac
done < "${PLAN_FILE}"

TOTAL="${#RUN_DIRS[@]}"
echo "Eligible runs: ${TOTAL}"
if [ "${FORCE}" = "1" ]; then
    echo "Already done:  ${#SKIPPED_DONE[@]} (FORCE=1, so these are queued too)"
else
    echo "Already done:  ${#SKIPPED_DONE[@]}"
fi
echo "No prompts:    ${#SKIPPED_NOPROMPT[@]}"
if [ "${#SKIPPED_LIMIT[@]}" -gt 0 ]; then
    echo "Over LIMIT:    ${#SKIPPED_LIMIT[@]} (LIMIT=${LIMIT})"
fi

# EMV3D comparability. The banner above promises the recorded bbox is reused;
# say so honestly when there is nothing to reuse. A run whose grid is
# re-derived gets a DIFFERENT voxel partition from the one its published
# EditMaskVariance_3D was computed on, so that one metric stops being
# comparable with the old table while every other metric stays fine.
if [ "${REUSE_BBOX}" = "1" ] && [ "${BBOX_FLAG_SUPPORTED}" = "1" ] \
   && [ "${#NO_BBOX[@]}" -gt 0 ]; then
    echo ""
    echo "WARNING: ${#NO_BBOX[@]} of ${TOTAL} eligible run(s) record no EMV3D bbox"
    echo "         (no metrics.json next to the checkpoint, or an older one that"
    echo "         predates the bbox keys). Their voxel grid is re-derived, so"
    echo "         their EditMaskVariance_3D will NOT be comparable with the"
    echo "         previously published value. Every other metric is unaffected."
    echo "         First: ${NO_BBOX[0]}"
    echo "         Pass SKIP_MASK_VARIANCE=1 to leave EMV3D out of this pass, or"
    echo "         --mask-variance-bbox-min/--mask-variance-bbox-max explicitly."
fi
echo ""

if [ "${TOTAL}" -eq 0 ]; then
    if [ "${#SKIPPED_NOPROMPT[@]}" -gt 0 ]; then
        echo "Runs skipped for missing prompts:"
        printf '  %s\n' "${SKIPPED_NOPROMPT[@]}"
        echo "  Supply them with PROMPT_MAP=<file.json>."
    fi
    echo "Nothing to do."
    exit 0
fi

# ── GPU selection (same pattern as edit.sh / evaluate.sh) ────────────────────
GPU_POOL=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_POOL <<< "${CUDA_VISIBLE_DEVICES}"
elif [ "${DRY_RUN}" = "1" ]; then
    GPU_POOL=("0")
else
    echo "[reevaluate_all.sh] Selecting ${PARALLEL} best available GPU(s)..."
    GPU_LIST="$(python scripts/pick_gpu.py "${PARALLEL}" 2>/dev/null | tail -1 || echo "")"
    if [ -z "${GPU_LIST}" ]; then
        GPU_LIST="0"
    fi
    IFS=',' read -r -a GPU_POOL <<< "${GPU_LIST}"
fi

if [ "${PARALLEL}" -gt "${#GPU_POOL[@]}" ]; then
    echo "NOTE: only ${#GPU_POOL[@]} GPU(s) available; lowering PARALLEL from ${PARALLEL}."
    PARALLEL="${#GPU_POOL[@]}"
fi
if [ "${PARALLEL}" -lt 1 ]; then
    PARALLEL=1
fi
if [ "${PARALLEL}" -gt 1 ] && [ "${LOG_WANDB}" = "1" ]; then
    echo "NOTE: LOG_WANDB=1 with PARALLEL>1 opens several WandB sessions at once."
fi

if [ "${DRY_RUN}" != "1" ]; then
    mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
fi

# ── One evaluation ───────────────────────────────────────────────────────────
run_one() {
    local run_dir="$1" out_dir="$2" slug="$3" src="$4" tgt="$5"
    local bbox_min="$6" bbox_max="$7" gpu="$8" index="$9"
    local prompt_source="${10}"
    local log_file="${LOG_DIR}/${slug}.log"
    local cmd=(python "${EVAL_SCRIPT}" eval
        --config "${run_dir}/config.yml"
        --src-prompt "${src}"
        --tgt-prompt "${tgt}"
        --output-dir "${out_dir}"
        --device "${DEVICE}")

    if [ "${BBOX_FLAG_SUPPORTED}" = "1" ] && [ -n "${bbox_min}" ] && [ -n "${bbox_max}" ]; then
        local -a bmin=() bmax=()
        read -r -a bmin <<< "${bbox_min}"
        read -r -a bmax <<< "${bbox_max}"
        cmd+=(--mask-variance-bbox-min "${bmin[@]}")
        cmd+=(--mask-variance-bbox-max "${bmax[@]}")
    fi
    if [ "${#COMMON_ARGS[@]}" -gt 0 ]; then
        cmd+=("${COMMON_ARGS[@]}")
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        printf '[%d/%d] %s  (prompts from %s)\n' \
            "${index}" "${TOTAL}" "${run_dir}" "${prompt_source}"
        printf '        CUDA_VISIBLE_DEVICES=%s ' "${gpu}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    # Keep the pre-existing numbers reachable: evaluate.py rewrites metrics.json
    # in place, and this snapshot is written once, so repeated re-runs can never
    # overwrite the original table values.
    if [ "${BACKUP_METRICS}" = "1" ] \
       && [ -f "${out_dir}/metrics.json" ] \
       && [ ! -f "${out_dir}/metrics.prev.json" ]; then
        cp "${out_dir}/metrics.json" "${out_dir}/metrics.prev.json"
    fi

    printf '[%d/%d] GPU %s  %s  (prompts from %s)\n' \
        "${index}" "${TOTAL}" "${gpu}" "${run_dir}" "${prompt_source}"
    local status=0
    # env options must precede the NAME=VALUE assignments, otherwise GNU env
    # stops option parsing at the assignment and runs "-u" as the command.
    if [ "${VERBOSE}" = "1" ]; then
        env -u WANDB_MODE CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" 2>&1 | tee "${log_file}" || status=$?
    else
        env -u WANDB_MODE CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log_file}" 2>&1 || status=$?
    fi

    echo "${status}" > "${STATUS_DIR}/${slug}.status"
    if [ "${status}" -eq 0 ]; then
        printf '        OK   -> %s/metrics.json\n' "${out_dir}"
    else
        printf '        FAIL (exit %s) -> %s\n' "${status}" "${log_file}"
    fi
    return 0
}

# ── Drive the queue ──────────────────────────────────────────────────────────
# Batched rather than a rolling pool: evaluations of one scene take comparable
# time, and a batch barrier keeps the GPU assignment unambiguous (one job per
# GPU, never two jobs on the same device).
i=0
FIRST_JOB_DONE=0
while [ "${i}" -lt "${TOTAL}" ]; do
    batch=1
    if [ "${PARALLEL}" -gt 1 ] && [ "${FIRST_JOB_DONE}" -eq 1 ]; then
        batch="${PARALLEL}"
    fi

    slot=0
    while [ "${slot}" -lt "${batch}" ] && [ "${i}" -lt "${TOTAL}" ]; do
        gpu="${GPU_POOL[$((slot % ${#GPU_POOL[@]}))]}"
        if [ "${batch}" -eq 1 ]; then
            run_one "${RUN_DIRS[$i]}" "${OUT_DIRS[$i]}" "${SLUGS[$i]}" \
                    "${SRCS[$i]}" "${TGTS[$i]}" \
                    "${BBOX_MINS[$i]}" "${BBOX_MAXS[$i]}" \
                    "${gpu}" "$((i + 1))" "${SOURCES[$i]}"
        else
            run_one "${RUN_DIRS[$i]}" "${OUT_DIRS[$i]}" "${SLUGS[$i]}" \
                    "${SRCS[$i]}" "${TGTS[$i]}" \
                    "${BBOX_MINS[$i]}" "${BBOX_MAXS[$i]}" \
                    "${gpu}" "$((i + 1))" "${SOURCES[$i]}" &
        fi
        slot=$((slot + 1))
        i=$((i + 1))
    done
    wait
    # The first evaluation always runs alone, so a cold CLIP/CLIPSeg weight
    # download happens once before any concurrency touches the same HF cache.
    FIRST_JOB_DONE=1
done

# ── Summary ──────────────────────────────────────────────────────────────────
SUCCEEDED=()
FAILED=()
if [ "${DRY_RUN}" != "1" ]; then
    idx=0
    while [ "${idx}" -lt "${TOTAL}" ]; do
        status_file="${STATUS_DIR}/${SLUGS[$idx]}.status"
        if [ -f "${status_file}" ] && [ "$(cat "${status_file}")" = "0" ]; then
            SUCCEEDED+=("${RUN_DIRS[$idx]}")
        else
            FAILED+=("${RUN_DIRS[$idx]}")
        fi
        idx=$((idx + 1))
    done

    {
        printf 'run_dir\tstatus\tlog_dir\n'
        for run in ${SUCCEEDED[@]+"${SUCCEEDED[@]}"}; do
            printf '%s\tok\t%s\n' "${run}" "${LOG_DIR}"
        done
        for run in ${FAILED[@]+"${FAILED[@]}"}; do
            printf '%s\tfailed\t%s\n' "${run}" "${LOG_DIR}"
        done
        for run in ${SKIPPED_DONE[@]+"${SKIPPED_DONE[@]}"}; do
            printf '%s\tskipped_done\t-\n' "${run}"
        done
        for run in ${SKIPPED_NOPROMPT[@]+"${SKIPPED_NOPROMPT[@]}"}; do
            printf '%s\tskipped_noprompt\t-\n' "${run}"
        done
        for run in ${SKIPPED_LIMIT[@]+"${SKIPPED_LIMIT[@]}"}; do
            printf '%s\tskipped_limit\t-\n' "${run}"
        done
    } > "${LOG_DIR}/summary.tsv"
fi

echo ""
echo "============================================"
if [ "${DRY_RUN}" = "1" ]; then
    echo " Dry run: ${TOTAL} run(s) would be evaluated."
else
    echo " Succeeded: ${#SUCCEEDED[@]}"
    echo " Failed:    ${#FAILED[@]}"
fi
echo " Skipped (already have the new keys): ${#SKIPPED_DONE[@]}"
echo " Skipped (no prompts resolved):       ${#SKIPPED_NOPROMPT[@]}"
if [ "${#SKIPPED_NOPROMPT[@]}" -gt 0 ]; then
    printf '   %s\n' "${SKIPPED_NOPROMPT[@]}"
    echo "   Supply them with PROMPT_MAP=<file.json>."
fi
if [ "${#SKIPPED_LIMIT[@]}" -gt 0 ]; then
    echo " Skipped (over LIMIT=${LIMIT}):          ${#SKIPPED_LIMIT[@]}"
fi
if [ "${DRY_RUN}" != "1" ] && [ "${#FAILED[@]}" -gt 0 ]; then
    echo " Failed runs:"
    printf '   %s\n' "${FAILED[@]}"
    echo " Logs in: ${LOG_DIR}"
fi
if [ "${DRY_RUN}" != "1" ]; then
    echo ""
    echo " Summary table: ${LOG_DIR}/summary.tsv"
    echo " Collect every metrics.json into one CSV:"
    echo "   python scripts/collect_metrics.py --outputs-root ${OUTPUTS_ROOT} --out metrics_summary.csv"
fi
echo "============================================"

if [ "${DRY_RUN}" != "1" ] && [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
