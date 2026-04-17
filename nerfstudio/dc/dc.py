from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from PIL import Image
from typing import List, Dict
from dc.attention_utils import (
    run_unet_with_cross_attention_capture,
    run_unet_with_skipped_attn,
)
from dc.dc_unet import CustomUNet2DConditionModel
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

    # Maximum iterations (must match --max-num-iterations for correct timestep schedule)
    max_iteration: int = 3000

    # TAG (Tangential Amplified Guidance) — eta_tag=1.0 disables TAG
    eta_tag: float = 1.0
    adaptive_tag: bool = False
    asymmetric_tag: bool = False
    tag_negative_prompt: str = ""
    tag_negative_strength: float = 0.0

    # Perpendicular Gradient Projection (Perp-Neg) — orthogonalize eps_tgt w.r.t. eps_src
    perp_neg: bool = False
    # Foreground-Masked Perp-Neg — restrict PN subtraction to foreground
    depth_masked_perp_neg: bool = False
    depth_mask_source: str = "depth"  # "depth" (renderer depth) or "cached" (precomputed masks)
    depth_mask_threshold: float = 0.5  # for depth source only
    cached_mask_dir: str = ""  # for cached source only
    perp_neg_mask_dilate: int = 0  # dilate binary mask by N image-pixels (hard expansion)
    perp_neg_mask_blur: float = 0.0  # Gaussian blur sigma in image-pixels (soft falloff; 0=off)
    perp_neg_alpha: float = 1.0  # PN subtraction strength (1.0 = full, 0.5 = half)

    # STG (Spatiotemporal Skip Guidance) — replace CFG with structure-preserving perturbation
    stg_enabled: bool = False
    stg_scale: float = 0.5
    stg_skip_layers: List[int] = field(default_factory=lambda: [2])
    stg_schedule_enabled: bool = False
    stg_decay_start_ratio: float = 0.0
    stg_decay_end_ratio: float = 1.0

    # Self-derived relevance masking — localize the DDS gradient using the
    # model's own tgt/src prediction discrepancy.
    gradient_mask_enabled: bool = False
    gradient_mask_blur: float = 3.0
    gradient_mask_ema_beta: float = 0.9
    gradient_mask_gamma: float = 1.0
    gradient_mask_warmup: int = 50
    source_blend_localization_enabled: bool = False
    outside_mask_anchor_weight: float = 0.0

    # Cross-attention-based relevance masking — use cross-attention maps
    cross_attention_mask_enabled: bool = False
    cross_attention_mask_keywords: str = ""
    cross_attention_mask_prompt: str = ""
    cross_attention_mask_layers: List[int] = field(default_factory=lambda: [1, 2])
    cross_attention_mask_weight: float = 1.0
    cross_attention_mask_blur: float = 0.0
    cross_attention_mask_gamma: float = 1.0

    # M1 — CA-only localization: ignore M_self, use cross-attention as the sole mask.
    # Useful on scenes (e.g. IP2P faces) where ||eps_tgt - eps_src|| anti-localizes the edit.
    cross_attention_mask_only: bool = False
    # M2 — inverted self-mask: use (1 - M_self) as the self-mask component before fusion.
    # Rationale: on IP2P, high ||eps_tgt - eps_src|| marks model-disagreement regions
    # (often NOT the edit target); inverting reframes M_self as a "model-agreement" mask.
    invert_self_mask: bool = False

    # DaCapo-inspired ψ schedule (Huang et al., CVPR 2025).
    # preserve_weight = psi * (1 + (psi_late_multiplier - 1) * (1 - t_normalized))
    # psi_late_multiplier=1.0 disables the schedule (current behavior).
    # >1 grows preservation as t decreases: coarse/edit early, fine/preserve late.
    psi_late_multiplier: float = 1.0

    # N2: latent-mean anchor. Adds λ · (mean(tgt_x0) - mean(src_x0)) to the final grad,
    # minimizing channel-mean drift in VAE latent space. Targets the TAG brightness /
    # saturation artifact without relying on text-semantic negative prompts.
    latent_mean_anchor_weight: float = 0.0


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

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt
        self.tag_negative_prompt = self.config.tag_negative_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")
        self.tag_negative_text_feature = (
            self.encode_text(self.tag_negative_prompt) if self.tag_negative_prompt else None
        )
    
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
        """
        relevance = (eps_tgt - eps_src).norm(dim=1, keepdim=True) # B, 1, H, W
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

        return mask.detach()

        
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

    def get_tag_negative_text_embedding(self):
        """Return the cached text embedding for the post-TAG negative regularizer."""
        return self.tag_negative_text_feature

    def _get_current_stg_scale(self) -> float:
        """Return the STG scale for the current edit iteration.

        By default STG stays constant for the whole run. When the schedule is
        enabled, it linearly decays from stg_scale to 0 between
        stg_decay_start_ratio and stg_decay_end_ratio of the total edit budget.
        """
        base_scale = self.config.stg_scale
        if not self.config.stg_schedule_enabled:
            return base_scale

        max_iteration = max(int(self.max_iteration), 1)
        progress = min(max(self.iteration / max_iteration, 0.0), 1.0)
        decay_start = float(self.config.stg_decay_start_ratio)
        decay_end = float(self.config.stg_decay_end_ratio)

        if decay_end <= decay_start:
            return base_scale if progress < decay_end else 0.0
        if progress <= decay_start:
            return base_scale
        if progress >= decay_end:
            return 0.0

        decay_progress = (progress - decay_start) / (decay_end - decay_start)
        return base_scale * (1.0 - decay_progress)

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
        src_emb,
        tgt_prompt=None,
        src_prompt=None,
        reduction="mean",
        return_dict=False,
        step=0,
        current_spot=0,
        depth_mask=None,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        tgt_text_embedding, src_text_embedding = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
        uncond_embedding = self.null_text_feature

        batch_size = tgt_x0.shape[0]
        t, t_prev, t_normalized = self.dc_timestep_sampling(batch_size)

        # Adaptive TAG: anneal η from eta_tag (high noise) → 1.0 (low noise)
        # ====================================================================================
        if self.config.adaptive_tag:
            eta_tag_current = 1.0 + (self.config.eta_tag - 1.0) * t_normalized ** (1/math.e) # Maybe test with other exponents too
        else:
            eta_tag_current = self.config.eta_tag
        # ====================================================================================

        noise = torch.randn_like(tgt_x0)
        
        eps = dict()
        eps_raw = dict()  # post-CFG snapshot, pre-STG/TAG/PN, used only for mask construction
        pred_x0s = dict()
        noisy_latents = dict()
        cross_attention_mask = None
        cross_attention_token_indices = None
        if self.config.cross_attention_mask_enabled:
            cross_attention_token_indices = get_cross_attention_token_indices(
                self.tokenizer,
                self.tgt_prompt,
                explicit_keywords=self.config.cross_attention_mask_keywords,
                cross_attention_prompt=self.config.cross_attention_mask_prompt,
                src_prompt=self.src_prompt,
            )
         
        for latent, cond_text_embedding, name in zip(
            [tgt_x0, src_x0], [tgt_text_embedding, src_text_embedding], ["tgt", "src"]
        ):
            latents_noisy = scheduler.add_noise(latent, noise, t)
            src_encoded = src_emb.latent_dist.mode()
            
            uncond_image_latent = torch.zeros_like(src_encoded)
            base_text_embeddings = torch.cat([cond_text_embedding, uncond_embedding, uncond_embedding], dim=0)
            base_text_embeddings = torch.cat([base_text_embeddings, base_text_embeddings], dim=1)
            base_latent_image = torch.cat([src_encoded, src_encoded, uncond_image_latent], dim=0)
            base_latent_model_input = torch.cat([latents_noisy] * 3, dim=0)
            base_latent_model_input = torch.cat([base_latent_model_input, base_latent_image], dim=1)
            
            if name == "tgt":
                tag_negative_correction = None
                if self.config.tag_negative_strength > 0 and self.get_tag_negative_text_embedding() is not None:
                    neg_text_embedding = self.get_tag_negative_text_embedding()
                    text_embeddings = torch.cat([cond_text_embedding, neg_text_embedding, uncond_embedding, uncond_embedding], dim=0)
                    text_embeddings = torch.cat([text_embeddings, text_embeddings], dim=1)
                    latent_image = torch.cat([src_encoded, src_encoded, src_encoded, uncond_image_latent], dim=0)
                    latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                    latent_model_input = torch.cat([latent_model_input, latent_image], dim=1)
                    timestep_input = torch.cat([t] * 4).to(device)

                    if self.config.cross_attention_mask_enabled:
                        noise_pred, cross_attention_mask = run_unet_with_cross_attention_capture(
                            self.unet,
                            self.config.cross_attention_mask_layers,
                            cross_attention_token_indices,
                            latent_model_input,
                            timestep_input,
                            text_embeddings,
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
                            latent_model_input,
                            timestep_input,
                            encoder_hidden_states=text_embeddings,
                        ).sample

                    noise_pred_text, neg_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(4)
                    
                    noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_image) + \
                        self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    
                    noise_pred_neg_cfg = noise_pred_uncond + self.config.guidance_scale * (neg_text - noise_pred_image) + \
                        self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    
                    tag_negative_correction = noise_pred_neg_cfg - noise_pred
                else:
                    text_embeddings = base_text_embeddings
                    latent_model_input = base_latent_model_input
                    timestep_input = torch.cat([t] * 3).to(device)
                    if self.config.cross_attention_mask_enabled:
                        noise_pred, cross_attention_mask = run_unet_with_cross_attention_capture(
                            self.unet,
                            self.config.cross_attention_mask_layers,
                            cross_attention_token_indices,
                            latent_model_input,
                            timestep_input,
                            text_embeddings,
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
                            latent_model_input,
                            timestep_input,
                            encoder_hidden_states=text_embeddings,
                        ).sample
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_image) + \
                        self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)

                # Pre-STG, pre-TAG, pre-PN snapshot used downstream for clean mask construction.
                eps_raw["tgt"] = noise_pred.detach().clone()

                # STG: amplify structural signal beyond full model (paper Eq 13)
                # ==============================================================================
                current_stg_scale = self._get_current_stg_scale()
                if self.config.stg_enabled and current_stg_scale > 0:
                    weak_pred = run_unet_with_skipped_attn(
                        self.unet,
                        self.device,
                        self.config.stg_skip_layers,
                        base_latent_model_input,
                        t,
                        base_text_embeddings,
                    )
                    weak_text, weak_image, weak_uncond = weak_pred.chunk(3)
                    noise_pred_weak = weak_uncond + self.config.guidance_scale * (weak_text - weak_image) + \
                        self.config.image_guidance_scale * (weak_image - weak_uncond)
                    noise_pred = noise_pred + current_stg_scale * (noise_pred - noise_pred_weak)
                # ==============================================================================
            else:
                text_embeddings = base_text_embeddings
                latent_model_input = base_latent_model_input

                noise_pred = self.unet.forward(
                    latent_model_input,
                    torch.cat([t] * 3).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                eps_raw["src"] = noise_pred.detach().clone()

            # TAG: amplify tangential component of noise prediction
            # ====================================================================================
            if self.config.asymmetric_tag:
                eta_current = eta_tag_current if name == "tgt" else 1.0
            else:
                eta_current = eta_tag_current
            
            v = latents_noisy / (latents_noisy.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8)
            noise_parallel = (noise_pred * v).sum(dim=(1, 2, 3), keepdim=True) * v
            noise_tangential = noise_pred - noise_parallel
            noise_pred = noise_parallel + eta_current * noise_tangential

            # Post-TAG negative-prompt regularizer
            if name == "tgt" and self.config.tag_negative_strength > 0 and tag_negative_correction is not None:
                noise_pred = noise_pred - self.config.tag_negative_strength * tag_negative_correction
            # ====================================================================================

            _, pred_x0 = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)

            eps[name] = noise_pred
            pred_x0s[name] = pred_x0
            noisy_latents[name] = latents_noisy

        # Perpendicular Gradient Projection (Perp-Neg): orthogonalize eps_tgt w.r.t. eps_src
        # ====================================================================================
        if self.config.perp_neg:
            # Always compute projection globally (keeps creative editing signal)
            src_norm_sq = (eps["src"] * eps["src"]).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            projection = (eps["tgt"] * eps["src"]).sum(dim=(1, 2, 3), keepdim=True) / src_norm_sq
            alpha = self.config.perp_neg_alpha
            if self.config.depth_masked_perp_neg and depth_mask is not None:
                # Masked application: subtract only in foreground, background keeps eps_tgt
                mask = F.interpolate(depth_mask, size=tgt_x0.shape[2:], mode="nearest")
                eps["tgt"] = eps["tgt"] - alpha * projection * eps["src"] * mask
            else:
                # Standard global Perp-Neg
                eps["tgt"] = eps["tgt"] - alpha * projection * eps["src"]
        # ====================================================================================

        self.iteration += 1
        
        grad_mask = None
        self_grad_mask = None

        # M1 short-circuit: when cross_attention_mask_only is set, skip the self-mask build.
        needs_self_mask = (
            self.config.gradient_mask_enabled
            or self.config.source_blend_localization_enabled
            or self.config.outside_mask_anchor_weight > 0
            or self.config.cross_attention_mask_enabled
        ) and not self.config.cross_attention_mask_only

        if needs_self_mask:
            self_grad_mask = self._build_gradient_relevance_mask(
                eps_raw["tgt"], eps_raw["src"], current_spot
            )
            if self.config.invert_self_mask:
                # M2: flip so high values mark model-agreement (often the real edit region on IP2P).
                self_grad_mask = (1.0 - self_grad_mask).clamp(0.0, 1.0)
            grad_mask = self_grad_mask

        if cross_attention_mask is not None:
            target_shape = grad_mask.shape[-2:] if grad_mask is not None else eps["tgt"].shape[-2:]
            if cross_attention_mask.shape[-2:] != target_shape:
                cross_attention_mask = F.interpolate(
                    cross_attention_mask,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            if self.config.cross_attention_mask_only:
                # M1: cross-attention is the sole localization signal.
                grad_mask = cross_attention_mask
            elif grad_mask is None:
                grad_mask = cross_attention_mask
            elif self.config.invert_self_mask:
                # M2: intersection of (1 - M_self) and M_attn — both factors ≤ 1, order invariant.
                grad_mask = grad_mask * cross_attention_mask
            else:
                weight = float(self.config.cross_attention_mask_weight)
                weight = min(max(weight, 0.0), 1.0)
                self_mask = grad_mask
                grad_mask = self_mask * ((1.0 - weight) + weight * cross_attention_mask)
            grad_mask = grad_mask.clamp(0.0, 1.0)

        eps_tgt_for_grad = eps["tgt"]
        if self.config.source_blend_localization_enabled and grad_mask is not None:
            eps_tgt_for_grad = eps["src"] + grad_mask * (eps["tgt"] - eps["src"])

        # DaCapo-inspired temporal schedule on ψ: coarse/edit at high t, fine/preserve at low t.
        # psi_late_multiplier=1.0 (default) → no change from DreamCatalyst behavior.
        psi_schedule_factor = 1.0 + (self.config.psi_late_multiplier - 1.0) * (1.0 - t_normalized)
        preserve_weight = self.config.psi * psi_schedule_factor
        if grad_mask is not None and self.config.outside_mask_anchor_weight > 0:
            preserve_weight = preserve_weight + self.config.outside_mask_anchor_weight * (1.0 - grad_mask)

        w_DDS = self.config.delta + self.config.gamma * (t_normalized ** (1/math.e))
        grad = (
            w_DDS * (eps_tgt_for_grad - eps["src"])
            + math.exp(t_normalized) * preserve_weight * (tgt_x0 - src_x0)
        )

        if self.config.gradient_mask_enabled and grad_mask is not None:
            grad = grad * grad_mask

        # N2: latent-mean anchor. Adds a bias that drives mean(tgt_x0) toward mean(src_x0) per
        # channel, counteracting TAG/CFG-driven brightness & saturation drift in VAE latent space.
        if self.config.latent_mean_anchor_weight > 0:
            tgt_mean = tgt_x0.mean(dim=(2, 3), keepdim=True)
            src_mean = src_x0.mean(dim=(2, 3), keepdim=True)
            grad = grad + self.config.latent_mean_anchor_weight * (tgt_mean - src_mean).expand_as(grad)

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
                cross_attention_mask=cross_attention_mask,
                tensor_to_pil_fn=tensor_to_pil,
                resize_image_fn=resize_image,
            )
        
        if return_dict:
            dic = {
                "loss": loss,
                "grad": grad,
                "t": t,
                "grad_mask": grad_mask,
                "self_grad_mask": self_grad_mask,
                "cross_attention_mask": cross_attention_mask,
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
