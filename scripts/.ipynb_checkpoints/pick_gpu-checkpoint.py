"""
Auto-GPU selector — picks the least-busy GPU on multi-GPU nodes.

Usage (in Python):
    from scripts.pick_gpu import pick_idle_gpu
    pick_idle_gpu()          # sets CUDA_VISIBLE_DEVICES automatically

Usage (from shell, before training):
    python -c "from scripts.pick_gpu import pick_idle_gpu; pick_idle_gpu()"
"""

import os
import subprocess


def pick_idle_gpu(
    mem_threshold_mb: int = 1000,
    util_threshold_pct: int = 10,
    set_env: bool = True,
) -> int:
    """
    Pick an 'idle' GPU using nvidia-smi and optionally set CUDA_VISIBLE_DEVICES.

    Heuristic:
      - Query index, memory.used, utilization.gpu for all GPUs.
      - Prefer ones with memory < mem_threshold_mb and util < util_threshold_pct.
      - Among candidates, choose the one with smallest (mem, util).
      - If no GPU passes the thresholds, choose the globally smallest (mem, util).

    Returns:
        The chosen GPU index (original global index).
    """
    query = "index,memory.used,utilization.gpu"
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    ).decode("utf-8")

    gpus = []
    for line in out.strip().splitlines():
        idx_str, mem_str, util_str = [x.strip() for x in line.split(",")]
        idx = int(idx_str)
        mem = int(mem_str)  # MiB
        util = int(util_str)  # %
        gpus.append((idx, mem, util))

    if not gpus:
        raise RuntimeError("No GPUs found via nvidia-smi")

    # 'Idle-ish' candidates
    idle = [g for g in gpus if g[1] < mem_threshold_mb and g[2] < util_threshold_pct]
    candidates = idle or gpus

    # Pick GPU with smallest (memory, util)
    best_idx, best_mem, best_util = sorted(candidates, key=lambda x: (x[1], x[2]))[0]

    if set_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
        print(
            f"[auto-gpu] Picked GPU {best_idx} (mem={best_mem} MiB, util={best_util}%) "
            f"→ CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
        )
    else:
        print(
            f"[auto-gpu] Best GPU appears to be {best_idx} "
            f"(mem={best_mem} MiB, util={best_util}%)"
        )

    return best_idx


if __name__ == "__main__":
    pick_idle_gpu()
