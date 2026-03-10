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

## Two-model architecture (CRITICAL)
The original DreamCatalyst uses TWO different diffusion models:
- **Step 3 (editing via DDS)**: `timbrooks/instruct-pix2pix` — 8-ch UNet, 3-way IP2P CFG. Passed via CLI in `edit.sh`.
- **Step 4 (refinement via SDEdit)**: `runwayml/stable-diffusion-v1-5` — 4-ch UNet, 2-way text CFG. Uses DCConfig default.
- The config default is SD 1.5. Editing overrides it to IP2P at runtime. Refinement uses the default.
- `run_sdedit` uses 4-channel input (no image conditioning) — designed for SD 1.5 only.
- `__call__` uses 8-channel input (with image conditioning) — designed for IP2P only.
- **NEVER** change the DCConfig default to IP2P — it breaks refinement.

## Fixes over original DreamCatalyst repo (https://github.com/kaist-cvml/DreamCatalyst)
Verified changes (original repo has bugs/missing features):

1. **~~Model path mismatch~~** [NOT A BUG — two-model design]: Config default `runwayml/stable-diffusion-v1-5` is intentional for refinement. Step 3 overrides to IP2P via CLI `--pipeline.dc.sd-pretrained-model-or-path timbrooks/instruct-pix2pix`.
2. **~~`run_sdedit` channel mismatch~~** [NOT A BUG]: `run_sdedit` uses 4-ch input for SD 1.5 refinement. This is correct. Do NOT add image_cond.
3. **~~Refinement pipeline missing image_cond~~** [NOT A BUG]: Refinement uses SD 1.5 with 4-ch text-only denoising. No image_cond needed.
4. **~~Refinement resize mismatch~~** [REVERTED]: Original resizes to `rendered_image.size()` — kept as original.
5. **Hardcoded `max_iteration=3000`**: Broke timestep schedule when using non-3000 iterations. Fixed: added `max_iteration` field to `DCConfig`, `edit.sh` syncs it with `--max-num-iterations`.
6. **Refinement SDEdit input**: Original repo correctly uses `input_img = original_image` — SDEdit starts from clean original photos, adds light noise, and denoises with the edit prompt to produce clean, multi-view consistent targets.
7. **~~Refinement SDEdit skip/steps mismatch~~** [MISDIAGNOSED]: The original refinement config already sets `num_inference_steps=20` in DCConfig, so skip is correctly computed from 20.
8. **`encode_src_image` unnormalized input** [INTENTIONAL, NOT A BUG]: The original passes [0,1] to VAE (no `2*x-1`). This produces weaker/noisier image conditioning, which is actually DESIRED — it lets the text guidance dominate in the DDS difference. Adding proper normalization makes image conditioning too strong, drowning out the editing signal. **Reverted to original behavior.**
9. **Asymmetric CFG spatial leakage** [KNOWN ISSUE]: The original uses full 3-way IP2P CFG for tgt branch but only 2-way (image+uncond, no text) for src branch. This means `eps_tgt - eps_src = gs*(text_tgt - image)`, which applies text influence globally. Attempted fix (symmetric CFG for both branches) FAILED — `gs*(text_tgt - text_src)` signal is too weak, produces only noise/artifacts with no edit. The asymmetric formulation is correct; spatial leakage should be addressed via masking (restrict gradient to target Gaussians), not by changing the CFG formula.

## Coding rules
- New hyperparameters go in `DCConfig` dataclass with defaults that reproduce original behavior (e.g., `eta_tag=1.0` = no-op).
- Access config via `self.config.param_name`, never bare variable names.
- `compute_posterior_mean` must be called AFTER any noise_pred modification, not before.
- In `run_sdedit`, use `xt.clone()` when saving previous state for delta computation.
- `--mixed-precision False` is required for Steps 3 and 4 (FP16 corrupts DDS gradients).

## Editing tips (learned from experiments)
- **Guidance scale**: Default 7.5 may be too weak. 12.5 worked well for material changes.
- **TAG novelties**: Can cause floaters. Disable (`eta_tag=1.0`) when debugging other issues. Re-enable gradually (1.05 → 1.10 → 1.15).

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