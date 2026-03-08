# CLAUDE.md — DreamCatalyst-PFC

## Project
Undergraduate thesis: text-driven 3D scene editing via 3DGS + diffusion models.
Pipeline: COLMAP → Nerfstudio splatfacto → DreamCatalyst (DDS editing) → Refinement (SDEdit).

## Environment
- Conda env: `dreamcatalyst_ns` (Python 3.9, PyTorch 2.1.2+cu118, CUDA 11.8)
- Setup: `bash setup.sh ns`
- PyTorch only. No TensorFlow.
- Server has NVIDIA H100 80GB HBM3.

## Key files
- `nerfstudio/dc/dc.py` — all guidance logic, DDS loss, novelties. This is THE file to edit.
- `nerfstudio/dc/dc_unet.py` — CustomUNet2DConditionModel (read-only, unless implementing STG hooks).
- `nerfstudio/dc/tasd_config.py` — TASD novelty params (eta_tag, adaptive_tag, asymmetric_tag). Unpacked into DCConfig via `**DC_CUSTOM_PARAMS`.
- `nerfstudio/dc/utils/free_lunch.py` — FreeU registration.
- `nerfstudio/3d_editing/dc_nerf/dc_config.py` — all method configs (dc_splat, dc_splat_refinement, etc). Imports DC_CUSTOM_PARAMS.
- `nerfstudio/3d_editing/dc_nerf/pipelines/dc_pipeline.py` — Step 3 editing pipeline.
- `nerfstudio/3d_editing/dc_nerf/pipelines/refinement_pipeline.py` — Step 4 refinement pipeline.
- `scripts/edit.sh` — Step 3 wrapper. Passes `--pipeline.dc.max-iteration` synced to `--max-num-iterations`.
- `scripts/refine.sh` — Step 4 wrapper.
- There is NO separate `guidance/` folder. Everything is in dc.py.

## Fixes over original DreamCatalyst repo (https://github.com/kaist-cvml/DreamCatalyst)
The original repo has several bugs that prevent it from working end-to-end:

1. **Model path mismatch**: Original config has `runwayml/stable-diffusion-v1-5` (4-ch UNet) but `__call__` builds 8-channel input with 3-way CFG (InstructPix2Pix architecture). Fixed: changed to `timbrooks/instruct-pix2pix`.
2. **`run_sdedit` channel mismatch**: Passes 4-ch input (`[xt]*2`) to UNet, but IP2P expects 8 channels. Fixed: added `image_cond` parameter, concatenates `[xt_input, image_cond]` along dim=1 for 8-ch input.
3. **Refinement pipeline missing image_cond**: `refinement_pipeline.py` now computes `image_cond = encode_src_image(resized_img).latent_dist.mode()` and passes it to `run_sdedit`.
4. **Refinement resize mismatch**: Original resized `edit_img` to match `rendered_image` but assigned to `datamanager.image_batch` (different size). Fixed: resize to `datamanager.image_batch[current_spot].shape[:2]`.
5. **Hardcoded `max_iteration=3000`**: Broke timestep schedule when using non-3000 iterations. Fixed: added `max_iteration` field to `DCConfig`, `edit.sh` syncs it with `--max-num-iterations`.
6. **Refinement uses original images**: `input_img = original_image` feeds unedited photos from disk through SDEdit, erasing the Step 3 edit. Fixed: `input_img = rendered_image` so SDEdit starts from the edited model's rendering.

## Coding rules
- New hyperparameters go in `DCConfig` dataclass with defaults that reproduce original behavior (e.g., `eta_tag=1.0` = no-op).
- Access config via `self.config.param_name`, never bare variable names.
- `compute_posterior_mean` must be called AFTER any noise_pred modification, not before.
- In `run_sdedit`, use `xt.clone()` when saving previous state for delta computation.
- `--mixed-precision False` is required for Steps 3 and 4 (FP16 corrupts DDS gradients).

## Editing tips (learned from experiments)
- **Prompts must describe the full scene**: `"a photo of a yellow LEGO bulldozer on a wooden table"` not `"a yellow LEGO bulldozer"`. Vague prompts destroy geometry.
- **Guidance scale**: Default 7.5 may be too weak. 12.5 worked well for material changes.
- **Geometry LRs**: Reduce xyz/scaling/opacity LRs to prevent floaters while allowing color changes:
  - `--optimizers.xyz.optimizer.lr 1.6e-5` (10x lower than default)
  - `--optimizers.scaling.optimizer.lr 0.001` (5x lower)
  - `--optimizers.opacity.optimizer.lr 0.01` (5x lower)
- **TAG novelties**: Can cause floaters. Disable (`eta_tag=1.0`) when debugging other issues. Re-enable gradually (1.05 → 1.10 → 1.15).
- **Step 4 refinement**: 30000 iters, ~1h on H100. Most iters are cheap photometric loss; SDEdit runs every 10 steps with only 2-4 denoising passes.

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