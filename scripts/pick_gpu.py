"""
Auto-GPU selector — picks the least-busy GPU(s) on multi-GPU nodes.

Usage (in Python):
    from scripts.pick_gpu import pick_idle_gpu, pick_idle_gpus
    pick_idle_gpu()          # sets CUDA_VISIBLE_DEVICES automatically
    pick_idle_gpus(n=2)      # returns list of N best GPU indices

Usage (from shell):
    python scripts/pick_gpu.py          # prints best GPU index
    python scripts/pick_gpu.py 2        # prints two best GPU indices, comma-separated
"""

import os
import subprocess
import sys


def _query_gpus():
    """Query all GPUs via nvidia-smi, return list of (index, mem_used_MiB, util_pct)."""
    query = "index,memory.used,utilization.gpu"
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    ).decode("utf-8")

    gpus = []
    for line in out.strip().splitlines():
        idx_str, mem_str, util_str = [x.strip() for x in line.split(",")]
        gpus.append((int(idx_str), int(mem_str), int(util_str)))

    if not gpus:
        raise RuntimeError("No GPUs found via nvidia-smi")
    return gpus


def pick_idle_gpus(
    n: int = 1,
    mem_threshold_mb: int = 1000,
    util_threshold_pct: int = 10,
):
    """
    Pick the N least-busy GPUs.

    Returns:
        List of GPU indices sorted by (memory, utilization).
    """
    gpus = _query_gpus()

    # Prefer idle GPUs, fall back to all
    idle = [g for g in gpus if g[1] < mem_threshold_mb and g[2] < util_threshold_pct]
    candidates = idle or gpus

    ranked = sorted(candidates, key=lambda x: (x[1], x[2]))
    selected = ranked[:n]

    for idx, mem, util in selected:
        print(
            f"[auto-gpu] GPU {idx} (mem={mem} MiB, util={util}%)",
            file=sys.stderr,
        )

    return [g[0] for g in selected]


def pick_idle_gpu(
    mem_threshold_mb: int = 1000,
    util_threshold_pct: int = 10,
    set_env: bool = True,
) -> int:
    """Pick a single idle GPU and optionally set CUDA_VISIBLE_DEVICES."""
    result = pick_idle_gpus(n=1, mem_threshold_mb=mem_threshold_mb, util_threshold_pct=util_threshold_pct)
    best_idx = result[0]

    if set_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)

    return best_idx


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    indices = pick_idle_gpus(n=n)
    # Print comma-separated indices to stdout (info goes to stderr)
    print(",".join(str(i) for i in indices))
