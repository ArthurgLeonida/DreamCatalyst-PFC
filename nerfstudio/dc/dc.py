from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from PIL import Image
from typing import List, Dict, Optional
from dc.attention_utils import (
    run_unet_with_cross_attention_capture,
    run_unet_with_pag,
    run_unet_with_skipped_attn,
)
from dc.dc_unet import CustomUNet2DConditionModel
from dc.guidance_utils import (
    apply_latent_mean_anchor,
    apply_source_blend,
    apply_stg,
    apply_tag,
    compute_bg_anchor_schedule_factor,
    compute_ca_mask_weight,
    compute_edit_strength,
    compute_gate_signal,
    compute_preserve_weight,
    compute_stg_scale,
    compute_tag_eta,
)
from dc.localization_utils import (
    apply_mask_postprocessing,
    build_cross_attention_relevance_mask,
    get_cross_attention_token_indices,
    normalize_relevance_map,
)
from dc.utils.free_lunch import register_free_upblock2d_in, register_free_crossattn_upblock2d_in
from dc.utils.logging_utils import log_dc_debug_to_wandb
import math


@dataclass
class DCConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v1-5"
    
    num_inference_steps: int = 500
    min_step_ratio: float = 0.2
    max_step_ratio: float = 0.9

    src_prompt: str = "a photo of a sks man"
    tgt_prompt: str = "a photo of a Batman"

    log_step: int = 10
    guidance_scale: float = 7.5
    device: torch.device = torch.device("cuda")
    image_guidance_scale: float = 1.5

    psi: float = 0.075
    chi = math.log(0.1)
    delta = 0.2
    gamma = 0.8

    freeu_b1: float=1.1
    freeu_b2: float=1.1
    freeu_s1: float=0.9
    freeu_s2: float=0.2

    max_iteration: int = 3000

    eta_tag: float = 1.25
    adaptive_tag: bool = True
    asymmetric_tag: bool = True


    stg_enabled: bool = True
    stg_scale: float = 2.0
    stg_skip_layers: List[int] = field(default_factory=lambda: [2])
    stg_schedule_enabled: bool = True
    stg_schedule_start_ratio: float = 0.4
    stg_schedule_end_ratio: float = 0.7
    stg_schedule_mode: str = "bump"
    stg_bump_peak_ratio: float = 0.5
    stg_edit_strength_adaptive: bool = True
    # Weak-model perturbation method. "stg" preserves the current behavior
    # (skip QK attention, V passes through). "pag" replaces self-attention
    # with the identity over spatial positions, perturbing spatial coherence
    # instead of cross-attention. PAG amplifies structural-edit signal while
    # leaving the symmetric-body semantic prior untouched, addressing STG's
    # ghost-feature artifact on invention scenes (stormtrooper helmet).
    # Shares stg_scale / stg_skip_layers / stg_schedule_* knobs.
    stg_weak_method: str = "stg"  # stg | pag
    # How STG and TAG compose when both are active.
    #   "sequential" (default, current behavior):
    #       eps_stg = eps_full + s · (eps_full − eps_weak)
    #       eps_out = TAG(eps_stg)            # TAG sees the STG-amplified signal
    #     Algebraically: eps_out = eps_full + (η−1)·eps_full_⊥
    #                            + s·(eps_full − eps_weak)
    #                            + s·(η−1)·(eps_full − eps_weak)_⊥
    #     The last term is the *implicit* extra TAG-tangential boost on the
    #     STG perturbation that compounding produces.
    #   "parallel":
    #       eps_out = TAG(eps_full) + s · (eps_full − eps_weak)
    #     STG and TAG act independently on the raw CFG prediction. STG
    #     contributes its full direction without TAG's tangential boost.
    #     Algebraically: eps_out = eps_full + (η−1)·eps_full_⊥
    #                            + s·(eps_full − eps_weak)
    #     The difference seq − par = s·(η−1)·(eps_full − eps_weak)_⊥ is
    #     exactly the compounding term above.
    stg_tag_compose_mode: str = "sequential"

    # Self-derived relevance masking
    gradient_mask_blur: float = 0.5
    gradient_mask_ema_beta: float = 0.99
    gradient_mask_ema_beta_auto: bool = True
    gradient_mask_ema_beta_camera_factor: float = 2.0
    gradient_mask_gamma: float = 1.2
    gradient_mask_warmup: int = 0
    # Per-frame normalization divisor for the RAW relevance mask (the
    # ``self_grad_mask_raw`` / ``raw_self`` variant fed to the voxel cache).
    #   1.0  = divide by the per-frame max (legacy). A single hot pixel then
    #          rescales the whole frame, so the same 3D point reads different
    #          normalized values across views purely because each frame's max
    #          differs — spurious cross-view variance the cache mistakes for
    #          real inconsistency.
    #   q in (0.5, 1.0) = divide by that per-frame quantile (e.g. 0.95 → p95).
    #          A robust scale: preserves absolute foreground/background
    #          structure (still a monotone linear rescale, so the §10 contrast
    #          fix holds) but is insensitive to single-pixel outliers.
    # Only affects ``self_grad_mask_raw``; the DDS gradient mask keeps its
    # percentile normalization. Default 1.0 reproduces prior behavior exactly.
    gradient_mask_raw_norm_quantile: float = 1.0
    source_blend_localization_enabled: bool = True
    # Leaky source-blend gate. 0.0 = original hard gate (edit zeroed where M=0).
    # A small floor (e.g. 0.1) lets a fraction of the edit drive reach M≈0
    # regions so novel structure can grow into empty space and bootstrap its
    # own mask. 1.0 = no localization. See apply_source_blend.
    source_blend_floor: float = 0.0

    outside_mask_anchor_weight: float = 0.2
    outside_mask_anchor_edit_strength_adaptive: bool = True
    outside_mask_anchor_edit_strength_power: float = 1.0
    # Temporal schedule on the outside-mask anchor weight. When enabled, scales
    # the (already edit-strength-adapted) anchor by a t-dependent factor.
    # "decay": factor goes 1 → 0 across the noise schedule (high t → low t),
    #          protecting background strongly during coarse-structure denoising
    #          and relaxing during refinement.
    # "growth": reverse (0 → 1). Disabled by default; factor = 1 reproduces
    # prior behavior.
    outside_mask_anchor_schedule_enabled: bool = False
    outside_mask_anchor_schedule_power: float = 0.5
    outside_mask_anchor_schedule_direction: str = "decay"  # decay | growth

    cross_attention_mask_enabled: bool = True
    cross_attention_mask_layers: List[int] = field(default_factory=lambda: [1, 2])
    cross_attention_mask_weight: float = 0.7
    cross_attention_mask_blur: float = 0.5
    cross_attention_mask_gamma: float = 1.2
    cross_attention_mask_weight_schedule_enabled: bool = False
    cross_attention_mask_weight_schedule_power: float = 0.5

    latent_mean_anchor_weight: float = 0.005

    external_mask_fusion: str = "bidirectional"  # bidirectional | screen
    external_mask_screen_attn_gate_strength: float = 1.0
    external_mask_screen_self_boost_lambda: float = 1.0
    external_mask_interp_suppression_ratio: float = 0.3
    external_mask_negative_variance_power: float = 0.0
    # Contested-region suppression: damp the 2D mask where the voxel cache
    # has enough cross-view evidence AND high cross-view variance (the
    # "contested" map from the pipeline). Unlike the up/down branches —
    # which are multiplied by confidence and therefore abstain exactly
    # where the gate distrusts — this term uses distrust as an ACTIVE
    # suppression signal: fused -= blend · ratio · contested · M.
    # Suppressing M both damps the edit force and strengthens the
    # (1 − M)-scaled preservation anchors, pulling contested regions back
    # toward the source. 0.0 = off (legacy behavior).
    external_mask_contested_suppression_ratio: float = 0.0


class DC(object):
    def __init__(self, config: DCConfig, use_wandb=False):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.unet = CustomUNet2DConditionModel.from_pretrained(
            config.sd_pretrained_model_or_path,
            subfolder="unet"
        ).to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")
    
        self.use_wandb = use_wandb

        self.iteration = 0
        self.max_iteration = config.max_iteration
        self.gradient_mask_ema: Dict[int, torch.Tensor] = {}

        b1 = self.config.freeu_b1
        b2 = self.config.freeu_b2
        s1= self.config.freeu_s1
        s2= self.config.freeu_s2

        register_free_upblock2d_in(self.unet, b1, b2, s1, s2)
        register_free_crossattn_upblock2d_in(self.unet, b1, b2, s1, s2)

    def _build_gradient_relevance_mask(self, eps_tgt, eps_src, current_spot):
        """Build a soft latent-space relevance mask from the DDS delta.

        The mask is:
        1. derived from ||eps_tgt - eps_src||_2 per spatial location,
        2. percentile-normalized per sample,
        3. temporally smoothed via EMA per view,
        4. optionally sharpened and blurred,
        5. detached before use so it does not backprop through the mask.

        Also returns a raw, max-normalized variant of the relevance map (no
        percentile pass, no EMA, no post-processing). The percentile pass
        is the right choice for the DDS gradient mask — it keeps the spatial
        rank ordering and is invariant to noise-level scale — but it
        compresses absolute confidence toward the median, which makes
        downstream consumers that need foreground/background discrimination
        (e.g. the voxel cache) lose signal. The raw variant preserves
        absolute structure for those consumers.
        """
        relevance = (eps_tgt - eps_src).norm(dim=1, keepdim=True) # B, 1, H, W
        # Per-sample max-normalization: preserves absolute structure within a
        # frame while keeping values in [0, 1]. Each frame is rescaled by its
        # own max so values are comparable across the cache's spatial extent.
        B = relevance.shape[0]
        flat_relevance = relevance.view(B, -1)
        q_norm = float(getattr(self.config, "gradient_mask_raw_norm_quantile", 1.0))
        if q_norm >= 1.0:
            # Legacy per-frame max: one outlier pixel can rescale the frame.
            per_sample_scale = flat_relevance.amax(dim=1)
        else:
            # Robust per-frame scale: divide by the q-quantile so a single hot
            # pixel can no longer deflate the whole frame's mask (a major source
            # of spurious cross-view variance once these masks are aggregated
            # into the 3D voxel cache). Pixels above the quantile saturate to 1.
            q_norm = min(max(q_norm, 0.5), 0.999)
            per_sample_scale = torch.quantile(flat_relevance, q_norm, dim=1)
        per_sample_scale = per_sample_scale.clamp_min(1e-8).view(B, 1, 1, 1)
        raw_mask = (relevance / per_sample_scale).clamp(0.0, 1.0).detach()

        normalized = normalize_relevance_map(relevance)

        prev_mask = self.gradient_mask_ema.get(current_spot)
        if prev_mask is None or prev_mask.shape != normalized.shape:
            ema_mask = normalized.detach()
        else:
            beta = self.config.gradient_mask_ema_beta
            ema_mask = beta * prev_mask + (1.0 - beta) * normalized.detach()
        self.gradient_mask_ema[current_spot] = ema_mask

        if self.iteration < self.config.gradient_mask_warmup:
            mask = torch.ones_like(normalized)
        else:
            mask = ema_mask

        mask = apply_mask_postprocessing(
            mask,
            gamma=self.config.gradient_mask_gamma,
            sigma=self.config.gradient_mask_blur,
        )

        return mask.detach(), raw_mask

        
    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        """
        device = self.device
        beta_t = self.scheduler.betas[t].to(device)
        alpha_t = self.scheduler.alphas[t].to(device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        mean_func = c0 * pred_x0 + c1 * xt
        
        return mean_func, pred_x0

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * 0.18215
    
    def encode_src_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor.float()
        return self.vae.encode(x)

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latent(self, latent):
        x = self.vae.decode(latent / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)

    def _get_current_stg_scale(self, current_edit_strength=None, iteration=None) -> float:
        """Return scheduled STG scale, optionally attenuated by current edit strength."""
        if iteration is None:
            iteration = self.iteration
        return compute_stg_scale(
            base_scale=self.config.stg_scale,
            iteration=iteration,
            max_iteration=self.max_iteration,
            schedule_enabled=self.config.stg_schedule_enabled,
            mode=self.config.stg_schedule_mode,
            start_ratio=self.config.stg_schedule_start_ratio,
            end_ratio=self.config.stg_schedule_end_ratio,
            bump_peak_ratio=self.config.stg_bump_peak_ratio,
            edit_strength_adaptive=self.config.stg_edit_strength_adaptive,
            current_edit_strength=current_edit_strength,
        )

    def dc_timestep_sampling(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = (
            len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)

        idx = torch.full((batch_size,), (max_step-min_step)*((self.max_iteration-self.iteration)/self.max_iteration) + min_step, dtype=torch.long, device="cpu")

        timestep_noralized = idx[0].item() / len(timesteps)
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()
        return t, t_prev, timestep_noralized

    def __call__(
        self,
        tgt_x0,
        src_x0,
        src_encoded,
        tgt_prompt=None,
        src_prompt=None,
        reduction="mean",
        return_dict=False,
        step=0,
        current_spot=0,
        external_grad_mask=None,
        external_grad_mask_valid=None,
        external_grad_mask_confidence=None,
        external_grad_mask_contested=None,
        external_mask_blend=0.0,
    ):
        device = self.device
        scheduler = self.scheduler

        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        tgt_text_embedding, src_text_embedding = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
        uncond_embedding = self.null_text_feature

        batch_size = tgt_x0.shape[0]
        t, t_prev, t_normalized = self.dc_timestep_sampling(batch_size)
        eta_tag_current = compute_tag_eta(
            self.config.eta_tag,
            t_normalized,
            self.config.adaptive_tag,
        )

        noise = torch.randn_like(tgt_x0)

        eps = dict()
        eps_raw = dict()
        pred_x0s = dict()
        noisy_latents = dict()
        base_text_embeddings_by_name = dict()
        base_latent_model_inputs = dict()
        target_cross_attention_mask = None
        target_cross_attention_token_indices = None
        if self.config.cross_attention_mask_enabled:
            target_cross_attention_token_indices = get_cross_attention_token_indices(
                self.tokenizer,
                self.tgt_prompt,
                src_prompt=self.src_prompt,
            )

        uncond_image_latent = torch.zeros_like(src_encoded)

        # Phase 1: clean CFG predictions only. 
        for latent, cond_text_embedding, name in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"]
        ):
            latents_noisy = scheduler.add_noise(latent, noise, t)
            noisy_latents[name] = latents_noisy

            base_text_embeddings = torch.cat([cond_text_embedding, uncond_embedding, uncond_embedding], dim=0)
            base_text_embeddings = torch.cat([base_text_embeddings, base_text_embeddings], dim=1)
            base_latent_image = torch.cat([src_encoded, src_encoded, uncond_image_latent], dim=0)
            base_latent_model_input = torch.cat([latents_noisy] * 3, dim=0)
            base_latent_model_input = torch.cat([base_latent_model_input, base_latent_image], dim=1)
            base_text_embeddings_by_name[name] = base_text_embeddings
            base_latent_model_inputs[name] = base_latent_model_input

            timestep_input = torch.cat([t] * 3).to(device)

            if name == "tgt" and self.config.cross_attention_mask_enabled:
                noise_pred, target_cross_attention_mask = run_unet_with_cross_attention_capture(
                    self.unet,
                    self.config.cross_attention_mask_layers,
                    target_cross_attention_token_indices,
                    base_latent_model_input,
                    timestep_input,
                    base_text_embeddings,
                    conditioned_batch_size=batch_size,
                    build_attention_mask_fn=lambda maps: build_cross_attention_relevance_mask(
                        maps,
                        gamma=self.config.cross_attention_mask_gamma,
                        sigma=self.config.cross_attention_mask_blur,
                        target_shape=latents_noisy.shape[-2:],
                    ),
                )
            else:
                noise_pred = self.unet.forward(
                    base_latent_model_input,
                    timestep_input,
                    encoder_hidden_states=base_text_embeddings,
                ).sample

            if name == "tgt":
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_image) + \
                    self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            else:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)

            eps_raw[name] = noise_pred.detach().clone()

        iteration_for_stg = self.iteration
        self.iteration += 1
        current_cross_attention_mask_weight = compute_ca_mask_weight(
            self.config.cross_attention_mask_weight,
            t_normalized,
            self.config.cross_attention_mask_weight_schedule_enabled,
            self.config.min_step_ratio,
            self.config.max_step_ratio,
            self.config.cross_attention_mask_weight_schedule_power,
        )

        grad_mask = None
        self_grad_mask = None
        needs_self_mask = (
            self.config.source_blend_localization_enabled
            or self.config.outside_mask_anchor_weight > 0
            or self.config.cross_attention_mask_enabled
        )

        self_grad_mask_raw: Optional[torch.Tensor] = None
        if needs_self_mask:
            self_grad_mask, self_grad_mask_raw = self._build_gradient_relevance_mask(
                eps_raw["tgt"], eps_raw["src"], current_spot
            )
            grad_mask = self_grad_mask

        if target_cross_attention_mask is not None:
            target_shape = grad_mask.shape[-2:] if grad_mask is not None else eps_raw["tgt"].shape[-2:]
            if target_cross_attention_mask.shape[-2:] != target_shape:
                target_cross_attention_mask = F.interpolate(
                    target_cross_attention_mask,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            if grad_mask is None:
                grad_mask = target_cross_attention_mask
            else:
                weight = current_cross_attention_mask_weight
                self_mask = grad_mask
                grad_mask = self_mask * ((1.0 - weight) + weight * target_cross_attention_mask)
            grad_mask = grad_mask.clamp(0.0, 1.0)

        internal_grad_mask = grad_mask.detach().clone() if grad_mask is not None else None

        # EXTERNAL MASK
        if external_grad_mask is not None:
            blend = min(max(float(external_mask_blend), 0.0), 1.0)
            ext = external_grad_mask
            target_shape = grad_mask.shape[-2:] if grad_mask is not None else ext.shape[-2:]
            if ext.shape[-2:] != target_shape:
                ext = F.interpolate(
                    ext,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            ext = ext.to(device=eps_raw["tgt"].device, dtype=eps_raw["tgt"].dtype).clamp(0.0, 1.0)
            valid = None
            if external_grad_mask_valid is not None:
                valid = external_grad_mask_valid
                if valid.dim() == 2:
                    valid = valid.unsqueeze(0).unsqueeze(0)
                elif valid.dim() == 3:
                    valid = valid.unsqueeze(0)
                valid = valid.to(device=eps_raw["tgt"].device, dtype=torch.float32)
                if valid.shape[-2:] != target_shape:
                    valid = F.interpolate(valid, size=target_shape, mode="nearest")
                valid = valid > 0.5
            confidence = None
            if external_grad_mask_confidence is not None:
                confidence = external_grad_mask_confidence
                if confidence.dim() == 2:
                    confidence = confidence.unsqueeze(0).unsqueeze(0)
                elif confidence.dim() == 3:
                    confidence = confidence.unsqueeze(0)
                confidence = confidence.to(device=eps_raw["tgt"].device, dtype=eps_raw["tgt"].dtype)
                if confidence.shape[-2:] != target_shape:
                    confidence = F.interpolate(
                        confidence,
                        size=target_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                confidence = confidence.clamp(0.0, 1.0)
            contested = None
            if external_grad_mask_contested is not None:
                contested = external_grad_mask_contested
                if contested.dim() == 2:
                    contested = contested.unsqueeze(0).unsqueeze(0)
                elif contested.dim() == 3:
                    contested = contested.unsqueeze(0)
                contested = contested.to(
                    device=eps_raw["tgt"].device, dtype=eps_raw["tgt"].dtype
                )
                if contested.shape[-2:] != target_shape:
                    contested = F.interpolate(
                        contested,
                        size=target_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                contested = contested.clamp(0.0, 1.0)
            blend_tensor = blend if confidence is None else blend * confidence
            if grad_mask is None:
                grad_mask = ext
            else:
                mode = str(self.config.external_mask_fusion).lower()
                
                # Check and compute gating signal once, before branching on fusion mode
                gate_signal = None
                if mode in ["bidirectional", "screen"]:
                    if target_cross_attention_mask is None and self_grad_mask is None:
                        if mode == "bidirectional":
                            raise ValueError(
                                "external_mask_fusion='bidirectional' requires at least one of "
                                "target_cross_attention_mask or self_grad_mask to be available for "
                                "gate computation. Enable cross_attention_mask or self-derived mask."
                            )
                    else:
                        target_shape = grad_mask.shape[-2:]

                        def _to_target(m):
                            if m is None:
                                return None
                            if m.shape[-2:] != target_shape:
                                m = F.interpolate(
                                    m,
                                    size=target_shape,
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            return m.to(device=grad_mask.device, dtype=grad_mask.dtype).clamp(0.0, 1.0)

                        target_ca = _to_target(target_cross_attention_mask)
                        sm = _to_target(self_grad_mask)

                        gate_signal = compute_gate_signal(
                            target_ca=target_ca,
                            sm=sm,
                            self_boost_lambda=self.config.external_mask_screen_self_boost_lambda,
                        )

                if mode == "bidirectional":
                    diff = ext - grad_mask
                    gate = 1.0

                    if gate_signal is not None:
                        strength = min(max(float(self.config.external_mask_screen_attn_gate_strength), 0.0), 1.0)
                        gate = (1.0 - strength) + strength * gate_signal

                    if torch.is_tensor(blend_tensor):
                        blend_map = blend_tensor
                    else:
                        blend_map = torch.full_like(diff, float(blend_tensor))

                    up = blend_map * gate * diff.clamp_min(0.0)
                    neg_var_power = float(
                        getattr(self.config, "external_mask_negative_variance_power", 0.0)
                    )
                    if neg_var_power > 0.0 and confidence is not None:
                        neg_extra = confidence.clamp(0.0, 1.0).pow(neg_var_power)
                    else:
                        neg_extra = 1.0
                    down = (
                        blend_map
                        * neg_extra
                        * float(getattr(self.config, "external_mask_interp_suppression_ratio", 0.4))
                        * diff.clamp_max(0.0)
                    )
                    # Contested-region suppression. Uses the scalar warmup
                    # blend (NOT blend_map): blend_map carries confidence,
                    # which is ~0 exactly where the variance gate distrusts —
                    # the regions this term must act on. The damp clamp bounds
                    # this term to at most M; the downstream mask clamp
                    # guarantees the final [0, 1] range.
                    contested_ratio = float(
                        getattr(
                            self.config,
                            "external_mask_contested_suppression_ratio",
                            0.0,
                        )
                    )
                    if contested_ratio > 0.0 and contested is not None:
                        damp = (blend * contested_ratio * contested).clamp(0.0, 1.0)
                        down_contested = -damp * grad_mask
                    else:
                        down_contested = 0.0
                    fused = grad_mask + up + down + down_contested
                elif mode == "screen":
                    contribution = blend_tensor * ext * (1.0 - grad_mask)

                    if gate_signal is not None:
                        strength = float(self.config.external_mask_screen_attn_gate_strength)
                        strength = min(max(strength, 0.0), 1.0)
                        gate = (1.0 - strength) + strength * gate_signal
                        contribution = contribution * gate
                    fused = grad_mask + contribution
                else:
                    raise ValueError(
                        f"Unknown external_mask_fusion={mode!r}; expected "
                        f"'bidirectional' or 'screen'."
                    )
                grad_mask = torch.where(valid, fused, grad_mask) if valid is not None else fused
            grad_mask = grad_mask.clamp(0.0, 1.0)

        current_edit_strength = compute_edit_strength(eps_raw["tgt"], eps_raw["src"])
        current_stg_scale = self._get_current_stg_scale(
            current_edit_strength=current_edit_strength,
            iteration=iteration_for_stg,
        )

        # Phase 2: apply guidance novelties from cached eps_raw after the clean current mask exists.
        compose_mode = str(self.config.stg_tag_compose_mode).lower()
        if compose_mode not in ("sequential", "parallel"):
            raise ValueError(
                f"Unknown stg_tag_compose_mode={compose_mode!r}; expected "
                f"'sequential' or 'parallel'."
            )
        for name in ["tgt", "src"]:
            eps_full = eps_raw[name]
            latents_noisy = noisy_latents[name]
            eta_current = eta_tag_current if (name == "tgt" or not self.config.asymmetric_tag) else 1.0
            stg_active = (
                name == "tgt"
                and self.config.stg_enabled
                and current_stg_scale > 0
            )

            if stg_active:
                weak_method = str(self.config.stg_weak_method).lower()
                if weak_method == "pag":
                    weak_runner = run_unet_with_pag
                elif weak_method == "stg":
                    weak_runner = run_unet_with_skipped_attn
                else:
                    raise ValueError(
                        f"Unknown stg_weak_method={weak_method!r}; expected "
                        f"'stg' or 'pag'."
                    )
                weak_pred = weak_runner(
                    self.unet,
                    self.device,
                    self.config.stg_skip_layers,
                    base_latent_model_inputs["tgt"],
                    t,
                    base_text_embeddings_by_name["tgt"],
                )
                weak_text, weak_image, weak_uncond = weak_pred.chunk(3)
                noise_pred_weak = weak_uncond + self.config.guidance_scale * (weak_text - weak_image) + \
                    self.config.image_guidance_scale * (weak_image - weak_uncond)
                if compose_mode == "sequential":
                    # eps_out = TAG(eps_full + s·(eps_full − eps_weak))
                    noise_pred = apply_stg(eps_full, noise_pred_weak, current_stg_scale)
                    noise_pred = apply_tag(noise_pred, latents_noisy, eta_current)
                else:  # "parallel"
                    # eps_out = TAG(eps_full) + s·(eps_full − eps_weak)
                    # TAG and STG act independently on the raw CFG prediction.
                    tag_out = apply_tag(eps_full, latents_noisy, eta_current)
                    stg_perturbation = current_stg_scale * (eps_full - noise_pred_weak)
                    noise_pred = tag_out + stg_perturbation
            else:
                # STG inactive on this branch — TAG only.
                noise_pred = apply_tag(eps_full, latents_noisy, eta_current)

            _, pred_x0 = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
            eps[name] = noise_pred
            pred_x0s[name] = pred_x0

        eps_tgt_for_grad = eps["tgt"]
        if self.config.source_blend_localization_enabled and grad_mask is not None:
            eps_tgt_for_grad = apply_source_blend(
                eps["tgt"], eps["src"], grad_mask, floor=self.config.source_blend_floor
            )

        bg_anchor_schedule_factor = compute_bg_anchor_schedule_factor(
            t_normalized=t_normalized,
            schedule_enabled=self.config.outside_mask_anchor_schedule_enabled,
            min_step_ratio=self.config.min_step_ratio,
            max_step_ratio=self.config.max_step_ratio,
            schedule_power=self.config.outside_mask_anchor_schedule_power,
            direction=self.config.outside_mask_anchor_schedule_direction,
        )

        preserve_weight = compute_preserve_weight(
            psi=self.config.psi,
            grad_mask=grad_mask,
            outside_mask_anchor_weight=self.config.outside_mask_anchor_weight,
            outside_mask_anchor_edit_strength_adaptive=self.config.outside_mask_anchor_edit_strength_adaptive,
            outside_mask_anchor_edit_strength_power=self.config.outside_mask_anchor_edit_strength_power,
            edit_strength=current_edit_strength,
            schedule_factor=bg_anchor_schedule_factor,
        )

        w_DDS = self.config.delta + self.config.gamma * (t_normalized ** (1/math.e))
        grad = (
            w_DDS * (eps_tgt_for_grad - eps["src"])
            + math.exp(t_normalized) * preserve_weight * (tgt_x0 - src_x0)
        )

        grad = apply_latent_mean_anchor(
            grad,
            tgt_x0,
            src_x0,
            self.config.latent_mean_anchor_weight,
        )

        grad = torch.nan_to_num(grad)
        
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size 
        
        
        if self.use_wandb and step % self.config.log_step == 0:
            log_dc_debug_to_wandb(
                step=step,
                current_spot=current_spot,
                t=t,
                t_normalized=t_normalized,
                eta_tag_current=eta_tag_current,
                current_stg_scale=current_stg_scale if self.config.stg_enabled else 0.0,
                current_edit_strength=current_edit_strength,
                w_dds=w_DDS,
                preserve_weight=preserve_weight,
                eps_tgt=eps["tgt"],
                eps_src=eps["src"],
                eps_tgt_for_grad=eps_tgt_for_grad,
                grad=grad,
                pred_x0_tgt=self.decode_latent(pred_x0s["tgt"]),
                pred_x0_src=self.decode_latent(pred_x0s["src"]),
                grad_mask=grad_mask,
                self_grad_mask=self_grad_mask,
                cross_attention_mask=target_cross_attention_mask,
                cross_attention_mask_weight_current=current_cross_attention_mask_weight,
                tensor_to_pil_fn=tensor_to_pil,
                resize_image_fn=resize_image,
            )
        
        if return_dict:
            dic = {
                "loss": loss,
                "grad": grad,
                "t": t,
                "grad_mask": grad_mask,
                "internal_grad_mask": internal_grad_mask,
                "self_grad_mask": self_grad_mask,
                "self_grad_mask_raw": self_grad_mask_raw,
                "cross_attention_mask": target_cross_attention_mask,
                "cross_attention_mask_weight": current_cross_attention_mask_weight,
                "edit_strength": current_edit_strength,
                "stg_scale": current_stg_scale if self.config.stg_enabled else 0.0,
            }
            return dic
        else:
            return loss

    def run_sdedit(self, x0, tgt_prompt=None, num_inference_steps=20, skip=7, eta=0):
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        reversed_timesteps = reversed(scheduler.timesteps)

        S = num_inference_steps - skip
        t = reversed_timesteps[S - 1]
        noise = torch.randn_like(x0)

        xt = scheduler.add_noise(x0, noise, t)

        self.update_text_features(None, tgt_prompt=tgt_prompt)
        tgt_text_embedding = self.tgt_text_feature
        null_text_embedding = self.null_text_feature
        text_embeddings = torch.cat([tgt_text_embedding, null_text_embedding], dim=0)

        op = timesteps[-S:]

        for t in op:
            xt_input = torch.cat([xt] * 2)
            noise_pred = self.unet.forward(
                xt_input,
                torch.cat([t[None]] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            xt = self.reverse_step(noise_pred, t, xt, eta=eta)

        return xt

    def reverse_step(self, model_output, timestep, sample, eta=0, variance_noise=None):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        variance = self.get_variance(timestep)
        model_output_direction = model_output
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z
        return prev_sample

    def get_variance(self, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance


def tensor_to_pil(img):
    if img.ndim == 4:
        img = img[0]
    img = img.cpu().permute(1, 2, 0).detach().numpy()
    
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def pil_to_tensor(img, device="cpu"):
    device = torch.device(device)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img[None].transpose(0, 3, 1, 2))
    img = img.to(device)
    return img


def resize_image(image, min_size):
    if min(image.size) < min_size:
        image = image.resize((min_size, min_size))
    return image
