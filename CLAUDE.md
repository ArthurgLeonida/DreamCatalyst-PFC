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
- `nerfstudio/dc/dc_unet.py` — CustomUNet2DConditionModel (read-only). STG hooks are registered dynamically from dc.py, no modification needed.
- `nerfstudio/dc/tasd_config.py` — TASD novelty params (eta_tag, adaptive_tag, asymmetric_tag, perp_neg, stg_enabled, stg_scale, stg_skip_layers). Unpacked into DCConfig via `**DC_CUSTOM_PARAMS`.
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
Verified changes:

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
- **Guidance scale is scene-dependent**: there is no single best CFG scale for all scenes.
- **Face / Tolkien elf**: `guidance_scale=12.5` is currently the best known setting. It gives stronger and better-looking edits than `7.5` on the face scene.
- **Stormtrooper / full-body person**: `guidance_scale=7.5` is currently the safest known setting. It reduced ghost-subject leakage compared with `12.5`, although the edit became somewhat weaker/softer.
- **Do NOT force one CFG scale for all experiments**. Current best practice is:
  - use **12.5** for the face/elf benchmark
  - use **7.5** as the stormtrooper baseline
  - treat kitchen/object scenes as unresolved and test around `7.5 -> 10.0 -> 12.5`
- **Next guidance-scale experiments**:
  - face/elf: keep `12.5` as the reference; do not lower it unless specifically testing a localization branch
  - stormtrooper: compare `7.5` vs `10.0`, then test whether self-derived masking makes `10.0` or `12.5` usable without ghosting
  - kitchen: start from `7.5` or `10.0`; only use `12.5` if localization/masking is active
- **Interpretation**: higher guidance helps edit strength, but also amplifies the risk of global leakage, duplicate subject emergence, and background smoothing. The point of self-derived masking is not to make CFG arbitrarily large; it is to make mid/high guidance more spatially selective.
- **Current preferred localization variant**: the self-derived mask branch now supports both direct gradient masking and **source-blended localization**. Prefer `source_blend_localization_enabled=True` as the first serious experiment; keep pure `gradient_mask_enabled=True` as the ablation/baseline. Source blending is more principled because low-mask regions naturally fall back toward `eps_src` instead of only getting their final gradient attenuated.
- **TAG novelties**: Can cause floaters. Disable (`eta_tag=1.0`) when debugging other issues. Re-enable gradually (1.05 → 1.10 → 1.15).
- **NeRF (dc) memory**: NeRF editing needs ~77 GiB VRAM for differentiable full-image rendering + IP2P. Use `--pipeline.dc-device cuda:1` to offload diffusion to a second GPU, or use 3DGS (dc_splat) which is more memory-efficient.
- **STG**: Adds ~50% per-iteration time. Formula matches paper Eq 13: `ε̃ = ε_full + w·(ε_full − ε_weak)`. `stg_scale=0` = off. Hook zeroes `attn1` output; with residual connection this IS the identity skip (correct STG-A behavior). Paper default `w=2.0` is too aggressive when combined with IP2P 3-way CFG at `guidance_scale=12.5` — start with `stg_scale=0.5` and `stg_skip_layers=[2]`, sweep up. ⚠️ Old formula `ε_weak + λ·(ε_full − ε_weak)` had λ=1 as no-op — all prior STG experiments with `stg_scale=1.0` had zero STG effect.
- **Perp-Neg**: Zero overhead. Boosts creative edits (elf cloak) but destroys background. Use Foreground-Masked PN (N6) with cached masks + soft blur for the best tradeoff.
- **Kitchen scene lesson**: kitchen is a strong localization stress test. Even improved prompts can still leak realism/material changes into table, placemats, and nearby objects. Treat this as a true DDS/IP2P spatial-selectivity problem, not only a prompt-writing issue.
- **Stormtrooper scene lesson**: this scene can show strong edit capability, but may create a duplicate/ghost stormtrooper in another region. Interpret this as weak instance selectivity / subject duplication, not just “bad generation.” Also note that the previous bug with only 4 images in eval outputs was due to COLMAP and has been solved.
- **Fine-structure loss**: even when the main subject edit looks good, wall seam lines and floor patterns can get washed out. This is likely low-amplitude global smoothing from repeated DDS optimization, not necessarily a gross localization failure.
- **Face dataset dimensions (downscale-factor 2)**: Original images in `images_2/` are 497×369 (W×H). Pipeline resizes smallest-side to 512 → `[1, 3, 512, 689]`. VAE encodes (8× spatial compression) → latent `[1, 4, 64, 86]`. UNet spatial progression: 64×86 → 32×43 → 16×22 → 8×11 (mid) → 16×22 → 32×43 → 64×86. The first CrossAttnUpBlock2D (Up 1, where STG hooks act) operates at **16×22×1280ch**. Latent is NOT 64×64 — aspect ratio propagates through the entire network.

## Novelties (TASD)

Modifications to DDS guidance in `dc.py`. Novelties N1–N7 are done; N8–N10 are TODO in priority order.

---

### ✅ DONE

1. **TAG in DDS** [DONE] — tangential amplification of `noise_pred` after CFG.
   * **Config:** `eta_tag` (default 1.0 = no-op).
   * **Based on:** TAG (Cho et al., arXiv 2510.04533, Seungryong Kim's lab @ KAIST).

2. **Adaptive TAG** [DONE] — anneal η with `t_normalized` over training.
   * **Config:** `adaptive_tag` (default False).
   * **Formula:** `η(t̂) = 1 + (η_max − 1) · t̂^(1/e)` — concave decay (stays strong longer than linear, drops steeply near end). Exponent `1/e` mirrors DreamCatalyst's own `w_DDS` schedule for natural coupling.
   * Original contribution inspired by TAG §6.
   * ⚠️ Previously used linear decay `η(t̂) = 1 + (η_max − 1)·t̂`; updated to `t̂^(1/e)` to keep amplification strong during the high-noise (structural) phase and decay later.

3. **Asymmetric TAG** [DONE] — apply TAG only to `eps_tgt`, not `eps_src`.
   * **Config:** `asymmetric_tag` (default False).
   * Original contribution. Preserves source branch identity signal.

4. **STG** [DONE] — skip self-attention in `up_blocks` for implicit weak-model guidance. Replaces or augments CFG with a structure-preserving perturbation.
   * **Config:** `stg_enabled` (default False), `stg_scale` (default 2.0 = paper STG-A default), `stg_skip_layers` (default [2] — single later up_block per paper).
   * **Implementation:** `_run_unet_with_skipped_attn()` registers forward hooks on `unet.up_blocks[i].attentions[j].transformer_blocks[k].attn1` that zero out self-attn output. Runs a second "weak" UNet pass, then blends per paper Eq 13: `eps_stg = eps_full + stg_scale*(eps_full - eps_weak)`. `stg_scale=0` → off, `stg_scale=1.0` → paper default (STG-R). Applied ONLY to `eps_tgt`. Hooks cleaned up in `finally` block.
   * **Based on:** STG (Hyung et al., CVPR 2025, Jaegul Choo's lab @ KAIST).
   * ⚠️ Only works with IP2P (Step 3). Do NOT apply in `run_sdedit`.
   * ⚠️ Adds ~50% per-iteration time (extra UNet forward pass, but `torch.no_grad()`).
   * **Future experiment — Adaptive STG scale:** apply the same decay schedule as Adaptive TAG to `stg_scale`. `stg_scale_current = stg_scale * t̂^(1/e)`. STG at full power during late (low-noise) timesteps fights fine-detail sharpening; decaying it mirrors the same intuition as Adaptive TAG.

5. **Perpendicular Gradient Projection (Perp-Neg)** [DONE] — Gram-Schmidt orthogonalization of `eps_tgt` w.r.t. `eps_src` before computing the DDS delta. Forces the editing signal to be perpendicular to the source identity signal, preventing the edit from destroying underlying geometry.
   * **Config:** `perp_neg` (default False).
   * **Implementation:** after the tgt/src loop, before gradient computation: `eps_tgt = eps_tgt - (dot(eps_tgt, eps_src) / dot(eps_src, eps_src)) * eps_src`. Applied after TAG (both branches computed first, then projection).
   * **Based on:** PCGrad (Yu et al., NeurIPS 2020) — conflicting gradient projection for multi-task learning; adapted to diffusion guidance by Perp-Neg (Armandpour et al., ICML 2023).
   * ⚠️ Previously misattributed to "Devil in Detail" (Jo et al., CVPR 2025), which proposes a different technique (excluding image prompts from negative CFG branch in self-attention KV injection for 2D image prompting — unrelated to DDS gradient projection).

6. **Foreground-Masked Perp-Neg** [DONE] — spatially restrict the Gram-Schmidt projection to foreground pixels, preserving background identity.
   * **Config:** `depth_masked_perp_neg` (default False), `depth_mask_source` (`"depth"` or `"cached"`, default `"depth"`), `depth_mask_threshold` (default 0.5, for depth source), `cached_mask_dir` (str, for cached source). Requires `perp_neg=True`.
   * **Two mask sources:**
     * `"cached"` (recommended): precomputed per-view masks from Grounded-SAM or rembg, generated offline via `scripts/generate_masks.py`. Sharp object boundaries, zero training cost. Masks loaded at pipeline init by matching image stem names.
     * `"depth"`: renderer depth map, percentile-normalized (5th–95th to remove outlier Gaussians), binarized at threshold. Works for indoor scenes with depth separation. Noisier boundaries than cached.
   * **Accumulation rejected:** 3DGS fills the entire scene with Gaussians (even sky/background), so accumulation ≈ 1.0 everywhere — useless for masking.
   * **Implementation:** `dc_pipeline.py` builds the mask (cached: load + resize to 512-space; depth: extract + normalize + binarize + resize). Inside the Perp-Neg block in `dc.py`, mask is resized to match `tgt_x0.shape[2:]` (robust to VAE rounding). Projection scalar is computed from **foreground pixels only** (masked dot products), then subtraction is also gated by the mask: `eps_tgt = eps_tgt - projection * eps_src * mask`. Background (mask=0) keeps original `eps_tgt` untouched.
   * **Why masked-global (not per-pixel or full-global):** per-pixel projection (`sum over dim=1` = 4 channels) is too noisy — produces salt-and-pepper artifacts. Full-global projection is dominated by background pixels (where `eps_tgt ≈ eps_src`, pushing projection toward 1.0), which over-subtracts in foreground. Masked-global uses only foreground elements (~8K if foreground is 40%) — stable and accurate.
   * **Based on:** original contribution combining PCGrad/Perp-Neg spatial gating with offline zero-shot segmentation. Inspired by the masking approach in RoMaP (Kim, Jang & Chun, 2025).
   * ⚠️ Cached masks: zero training cost. Depth masks: zero model cost (depth already computed by renderer).
   * ⚠️ Mask is `.detach()`ed — no gradients flow through the mask to the NeRF density field.
   * **Script:** `scripts/generate_masks.py` — supports `grounded-sam` (GroundingDINO + SAM, text-prompted) and `rembg` (U2Net, automatic) backends. Run in a separate env to avoid dependency conflicts.

7. **Self-Derived Relevance Masking** [DONE / exploratory] — localize the final DDS gradient using the model's own tgt/src discrepancy.
   * **Config:** `gradient_mask_enabled`, `gradient_mask_blur`, `gradient_mask_ema_beta`, `gradient_mask_gamma`, `gradient_mask_warmup`, `source_blend_localization_enabled`.
   * **Implementation:** derive `R = ||eps_tgt - eps_src||_2` per spatial location, percentile-normalize it (5th/95th), smooth it with EMA per `current_spot`, optionally sharpen it with `gamma`, optionally blur it in latent space, and then use the resulting soft mask in one of two ways:
     * **Ablation / direct masking:** `grad_masked = grad * M`
     * **Preferred current form:** `eps_tgt_loc = eps_src + M * (eps_tgt - eps_src)`
   * **Why source blending is preferred:** low-mask regions naturally fall back toward the source branch, which is a cleaner way to reduce floor/wall/background drift than only attenuating the already-computed final DDS gradient.
   * **Why it exists:** recent kitchen and stormtrooper failures suggest the biggest remaining issue is spatial selectivity. Kitchen leaks into nearby surfaces/objects; stormtrooper can instantiate a duplicate "ghost" subject even when the main edit works.
   * **Based on:** inspired by **LatentEditor** (delta-score localization), conceptually aligned with **LENeRF** (3D-localized editing), and complementary to **CustomNeRF** (local-global editing schedule).
   * **Important caution:** this is **LatentEditor-inspired**, not literally the same signal as LatentEditor's conditional-vs-empty delta. Here the relevance comes from `eps_tgt - eps_src` in DreamCatalyst's asymmetric DDS setup.

---

### 🟡 TODO — Medium Priority (professor's recommended papers)

8. **Depth Regularization** [TODO] — add a monocular depth consistency loss to the DreamCatalyst loss `L_DC` to anchor Gaussian positions during editing. Prevents floaters introduced by TAG at high η.
   * **Config:** `depth_reg_weight` (default 0.0), `depth_model` (default `"midas"`).
   * **Implementation:** in `dc.py` loss computation, load MiDaS depth prediction for the current render and add `depth_reg_weight * L1(depth_pred, depth_anchor)` where `depth_anchor` is computed once from the original unedited scene.
   * **Based on:** Depth-Regularized Optimization for 3DGS (Chung et al., CVPRW 2024).
   * ⚠️ Adds ~1.2s per iteration. Use `depth_reg_freq=10` (every 10 iters) to amortize.

9. **3D-GALP Part Masking** [TODO] — restrict DDS gradient to Gaussians that belong to the target semantic region, leaving background untouched.
   * **Config:** `galp_enabled` (default False), `galp_seg_prompt` (str, e.g. `"face"`).
   * **Implementation:** run a zero-shot segmentation (e.g. Grounded-SAM) on the source render to get a 2D mask per camera. Project mask into 3D via splatting opacity weights to identify "target Gaussians". Zero out `∇θ L_DC` for non-target Gaussians before the Adam step.
   * **Based on:** RoMaP (Kim, Jang & Chun, 2025).
   * ⚠️ Requires `groundingdino` + `segment-anything` in the env. Heavy dependency.
   * ⚠️ 200 MB VRAM overhead. Run with `--pipeline.datamanager.train-num-rays-per-batch 2048`.

10. **GAP³D Cross-View Attention Prior** [TODO] — enforce multi-view consistency by injecting cross-attention keys/values from a second camera view into the current view's UNet forward pass during guidance.
   * **Config:** `gap3d_enabled` (default False), `gap3d_num_views` (default 4).
   * **Implementation:** at each iteration, sample `gap3d_num_views` additional cameras, render and encode them, then patch `unet.up_blocks` cross-attention to attend over the multi-view feature set. This requires modifying `dc_unet.py` (the one read-only file — make a backup first).
   * **Based on:** InterGSEdit (Wen et al., 2025).
   * ⚠️ 300 MB VRAM overhead per extra view. Start with `gap3d_num_views=2`.
   * ⚠️ Most complex novelty. Implement AFTER N4–N8 are validated.

---

### 🟢 TODO — Low Priority (optional, bonus contribution)

11. **Anchor L1 Loss (RoMaP-style)** [TODO] — add a part-level L1 anchor term that penalizes deviation of edited Gaussians from their original positions, preventing over-deformation of geometry during aggressive edits.
    * **Config:** `anchor_l1_weight` (default 0.0), `anchor_l1_region` (`"full"` or `"background"`).
    * **Implementation:** cache `theta_0` (original Gaussian positions `mu_i`) before training starts. Add `anchor_l1_weight * mean(|mu_i - mu_i_0|)` to `L_DC`. When `galp_enabled=True`, apply only to non-target Gaussians (background anchor).
    * **Based on:** RoMaP (Kim, Jang & Chun, 2025).
    * ⚠️ Much simpler than N8 (3D-GALP). Can be implemented independently in ~20 lines.

---

### Implementation Order Recommendation

1. ~~N4 (STG) + N5 (Perp-Neg)~~ [DONE]
2. ~~N6 (Depth-Masked Perp-Neg)~~ [DONE]
3. ~~N7 (Self-Derived Relevance Masking)~~ [DONE / exploratory; prefer source-blended localization before bigger branches]
4. N8 (Depth Reg) + N11 (Anchor L1) [lightweight preservation losses]
5. N9 (3D-GALP masking) [needs Grounded-SAM setup]
6. N10 (GAP³D) [most complex, needs dc_unet.py edit]

### Current Best Interpretation Of Failures

- **Face**: best success case; subject dominance hides much of the global leakage.
- **Kitchen**: strongest evidence of localized editability failure; realism/material changes leak into table, placemats, and nearby objects.
- **Stormtrooper**: main subject edit can work well, but duplicate/ghost subject emergence reveals weak instance selectivity.
- **Wall/floor line loss**: even without obvious ghosting, repeated DDS optimization can wash out fine non-target structure via low-amplitude global smoothing.

So the main remaining bottleneck is:

> strong but spatially under-constrained editing

not simply "stronger guidance needed."

---

### 📚 Related Work (NOT dc.py novelties — cite in thesis only)

**OmniSplat** (Lee et al., arXiv 2412.16604, Kyoung Mu Lee's lab @ SNU) 

What it is: Feed-forward 3DGS reconstruction from omnidirectional (360°) images using Yin-Yang grid decomposition to bridge the domain gap with perspective-trained encoders. Relevance to your project: NOT applicable as a dc.py novelty. Your pipeline uses perspective images + COLMAP + nerfstudio. OmniSplat replaces Steps 1-2 entirely. 

Cite in: Chapter 2 (Related Work §2.1), Chapter 7 (Future Work — 360° capture).

**C3G** (An, Jung et al., KAIST AI, 2025, Seungryong Kim's lab — same as TAG)

What it is: Compact 3DGS using only 2K learnable query tokens instead of per-pixel Gaussians. 65× fewer Gaussians, lower VRAM, competitive reconstruction quality. Includes C3G-F for view-invariant feature lifting without autoencoder compression. Relevance to your project: NOT applicable as a dc.py novelty, but highly relevant as a future pipeline evolution. Running DDS editing on 2K compact Gaussians instead of millions would: (a) reduce floater risk, (b) make N9 Anchor L1 trivial, (c) make N8 GAP³D unnecessary since C3G-F already provides view-invariant features. Same lab as TAG (N1) — strengthens the KAIST-connection narrative of your thesis.

Cite in: Chapter 2 (Related Work §2.1), Chapter 7 (Future Work — compact base repr).

**LatentEditor** (Khalid et al., 2024)

What it is: 2D instruction-based image editing that derives a localization signal from the model's own conditional-vs-empty prediction delta. Relevance to your project: the cleanest inspiration for self-derived relevance masking in DreamCatalyst, especially for kitchen/stormtrooper-style leakage.

Cite in: Chapter 2 (Related Work §2.2), Chapter 5 (new localization branch).

**LENeRF** (KAIST, Jaegul Choo's group, 2023)

What it is: localized NeRF editing via a learned 3D attention field distilled from CLIP-style masks. Relevance to your project: strongest conceptual support for interpreting kitchen as a localized-editability failure and stormtrooper as a weak instance-selectivity problem.

Cite in: Chapter 2 (Related Work §2.2), Chapter 7 (Future Work — learned 3D attention).

**CustomNeRF** (He et al., CVPR 2024)

What it is: adaptive source-driven 3D scene editing with Local-Global Iterative Editing (LGIE). Relevance to your project: excellent next-step inspiration if self-derived masking helps but remains too conservative; suggests alternating localized and global editing instead of only masking.

Cite in: Chapter 2 (Related Work §2.2), Chapter 7 (Future Work — local-global scheduling).

**ED-NeRF** (KAIST, 2024)

What it is: latent-space NeRF editing that argues DDS is better suited for editing than SDS. Relevance to your project: supports staying in a latent-space/DDS editing framework rather than switching immediately to a different backbone.

Cite in: Chapter 2 (Related Work §2.2), Chapter 7 (Future Work — NeRF-focused editing extensions).

**Blending-NeRF** (SNU, 2024)

What it is: localized NeRF editing with support for structural additions/replacements. Relevance to your project: especially relevant to kitchen-like object replacement and cloak-like localized structural edits, but more invasive than the current timeline allows.

Cite in: Chapter 2 (Related Work §2.2), Chapter 7 (Future Work — localized structural editing).

## Build gotchas
- System nvcc 12.4 vs PyTorch CUDA 11.8 → CUDA_HOME must point to conda env.
- `setuptools>=70` breaks tinycudann → pin `setuptools<70`.
- `huggingface_hub>=0.24` removes `cached_download` → pin `<0.24`.
- CUDA builds must run in subshells to avoid LD_LIBRARY_PATH leaking.
- COLMAP must be ≤3.9.1 (uses SiftExtraction/SiftMatching API).
