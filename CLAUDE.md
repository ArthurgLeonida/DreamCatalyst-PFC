# CLAUDE.md — DreamCatalyst-PFC

## Project
Undergraduate thesis: text-driven 3D scene editing via 3DGS + diffusion models.
Pipeline: COLMAP → Nerfstudio splatfacto → DreamCatalyst (DDS editing).

## Environment
- Conda env: `dreamcatalyst_ns` (Python 3.9, PyTorch 2.1.2+cu118, CUDA 11.8)
- Setup: `bash setup.sh ns`
- PyTorch only. No TensorFlow.

## Key files
- `nerfstudio/dc/dc.py` — all guidance logic, DDS loss, novelties. This is THE file to edit.
- `nerfstudio/dc/dc_unet.py` — CustomUNet2DConditionModel (read-only, unless implementing STG hooks).
- `nerfstudio/dc/utils/free_lunch.py` — FreeU registration.
- There is NO separate `guidance/` folder. Everything is in dc.py.

## Coding rules
- New hyperparameters go in `DCConfig` dataclass with defaults that reproduce original behavior (e.g., `eta_tag=1.0` = no-op).
- Access config via `self.config.param_name`, never bare variable names.
- `compute_posterior_mean` must be called AFTER any noise_pred modification, not before.
- In `run_sdedit`, use `xt.clone()` when saving previous state for delta computation.

## Novelties (TASD)
Five modifications to DDS guidance in dc.py. See `docs/TASD_implementation_guide.md` for full details.

1. **TAG in SDS/DDS** [DONE] — tangential amplification of noise_pred after CFG. Config: `eta_tag`. Based on: TAG (Cho et al., arXiv 2510.04533, Seungryong Kim's lab @ KAIST).
2. **Adaptive TAG** [DONE] — anneal η with `t_normalized`. Config: `adaptive_tag`. Original contribution inspired by TAG §6.
3. **Asymmetric TAG** [DONE] — apply TAG only to target branch. Config: `asymmetric_tag`. Original contribution.
4. **STG** [TODO] — skip self-attention in up_blocks for weak-model guidance. Config: `stg_enabled`. Based on: STG (Hyung et al., CVPR 2025, Jaegul Choo's lab @ KAIST).
5. **Conflict-Free Guidance** [TODO] — project out conflicting text/image guidance. Config: `conflict_free`. Based on: Devil in Detail (Jo et al., CVPR 2025, Jaegul Choo's lab @ KAIST).

## Build gotchas
- System nvcc 12.4 vs PyTorch CUDA 11.8 → CUDA_HOME must point to conda env.
- `setuptools>=70` breaks tinycudann → pin `setuptools<70`.
- `huggingface_hub>=0.24` removes `cached_download` → pin `<0.24`.
- CUDA builds must run in subshells to avoid LD_LIBRARY_PATH leaking.
- COLMAP must be ≤3.9.1 (uses SiftExtraction/SiftMatching API).