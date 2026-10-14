"""DreamCatalyst editing loss (DDS on InstructPix2Pix) with the PFC extensions.

Upstream DreamCatalyst (KAIST) is kept as it was: the first block of DCConfig,
the DC methods from ``compute_posterior_mean`` to ``get_variance``, and the
image helpers at the end of the file. Everything else implements the method of
the paper. One ``DC.__call__`` is one editing iteration on one view:

  1. IP2P noise predictions for the current render (target) and the source.
  2. Localization masks, from those clean predictions (before amplification):
       S      target-instruction mask: where the instruction, applied to the
              current render, still differs from reconstructing the source
       S_src  source-instruction mask: the same contrast on the source image,
              computed once per view
       A      cross-attention mask of the edit tokens
       M      = max(S * A, min(S, S_src)), plus optional voxel-cache support
  3. Amplification of the target prediction only: adaptive TAG and scheduled STG.
  4. Gradient = w_DDS * M * (eps_tgt - eps_src)                (localized DDS)
              + e^t * (psi + w_out (1 - s)(1 - S_src)) * (z - z_src)  (preservation)
              + latent-mean anchor.

Every extension can be switched off in method_config.py, which reproduces
upstream DreamCatalyst (the "dc baseline" values listed there).
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from PIL import Image

from dc.attention_utils import (
    run_unet_with_cross_attention_capture,
    run_unet_with_skipped_attn,
)
from dc.dc_unet import CustomUNet2DConditionModel
from dc.guidance_utils import (
    apply_latent_mean_anchor,
    apply_source_blend,
    apply_tag,
    compute_cache_gate,
    compute_edit_strength,
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


@dataclass
class DCConfig:
    # ------------------------------------------------------------------
    # Upstream DreamCatalyst
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Guidance amplification (target branch only)
    # ------------------------------------------------------------------
    # Adaptive TAG ceiling: eta(t) = 1 + (eta_tag - 1) * t^(1/e). 1.0 disables TAG.
    eta_tag: float = 1.2
    # STG peak scale, attenuated by (1 - s). 0.0 disables STG.
    stg_scale: float = 3.0
    # UNet up-blocks whose self-attention the STG weak pass skips.
    stg_skip_layers: List[int] = field(default_factory=lambda: [2])
    # Bump schedule over training progress: zero outside [start, end], peak at
    # start + peak_ratio * (end - start).
    stg_schedule_start_ratio: float = 0.0
    stg_schedule_end_ratio: float = 0.8
    stg_bump_peak_ratio: float = 0.5

    # ------------------------------------------------------------------
    # Localization: where the edit may act
    # ------------------------------------------------------------------
    # Restrict the DDS delta to the edit mask M.
    source_blend_localization_enabled: bool = True
    # Shaping of S and S_src: gamma sharpening, then a Gaussian blur (latent px).
    gradient_mask_gamma: float = 1.2
    gradient_mask_blur: float = 0.5
    # Voxel-cache input: S's magnitude divided by this per-frame quantile.
    gradient_mask_raw_norm_quantile: float = 0.95
    # Per-view EMA of S, memory counted in visits of that view:
    # beta = exp(-1 / memory).
    gradient_mask_ema_memory_visits: float = 5.0
    # S_src: noise level (fraction of the training timesteps), number of noise
    # draws averaged per view, and the seed of its private RNG.
    localization_source_timestep_ratio: float = 0.5
    localization_source_num_samples: int = 4
    localization_source_seed: int = 42
    # A: cross-attention mask of the edit tokens, read from these up-blocks.
    cross_attention_mask_enabled: bool = True
    cross_attention_mask_layers: List[int] = field(default_factory=lambda: [1, 2])
    cross_attention_mask_gamma: float = 1.2
    cross_attention_mask_blur: float = 0.5

    # ------------------------------------------------------------------
    # Preservation: what stays like the source
    # ------------------------------------------------------------------
    # Extra preservation outside S_src, attenuated by (1 - s). 0.0 disables it.
    outside_mask_anchor_weight: float = 0.25
    # Per-channel latent-mean anchor. 0.0 disables it.
    latent_mean_anchor_weight: float = 0.005

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Check the PFC settings. DC calls this again: the CLI can edit fields."""
        memory = self.gradient_mask_ema_memory_visits
        if not math.isfinite(memory) or memory <= 0:
            raise ValueError("gradient_mask_ema_memory_visits must be finite and positive")
        if not 0.0 < self.localization_source_timestep_ratio < 1.0:
            raise ValueError("localization_source_timestep_ratio must be in (0, 1)")
        if self.localization_source_num_samples < 1:
            raise ValueError("localization_source_num_samples must be at least 1")
        if self.localization_source_seed < 0:
            raise ValueError("localization_source_seed must be nonnegative")

    @property
    def localization_enabled(self) -> bool:
        """Whether any mechanism needs the localization masks (all off = upstream DC)."""
        return (
            self.source_blend_localization_enabled
            or self.outside_mask_anchor_weight > 0
            or self.cross_attention_mask_enabled
        )


@dataclass
class LocalizationMasks:
    """The masks of one iteration, each [B, 1, h, w] in [0, 1] at latent resolution."""

    self_mask: torch.Tensor  # S: target-instruction mask (per-view EMA, gamma, blur)
    self_mask_raw: torch.Tensor  # S's magnitude over its 95th percentile: voxel-cache input
    source_mask: torch.Tensor  # S_src: source-instruction mask, also the preservation mask
    ca_mask: Optional[torch.Tensor]  # A: cross-attention mask (None when disabled)
    internal: torch.Tensor  # max(S * A, min(S, S_src)): the edit mask before the cache
    final: torch.Tensor  # edit mask M after voxel-cache fusion (= internal without it)
    veto_floor_addition: Optional[torch.Tensor]  # support min(S, S_src) restored over S * A


def _match_mask_shape(mask: torch.Tensor, shape, mode: str = "bilinear") -> torch.Tensor:
    """Resize a [B, 1, h, w] map to ``shape``; a no-op when it already matches."""
    if mask.shape[-2:] == shape:
        return mask
    if mode == "nearest":
        return F.interpolate(mask, size=shape, mode="nearest")
    return F.interpolate(mask, size=shape, mode="bilinear", align_corners=False)


class DC(object):
    def __init__(self, config: DCConfig, use_wandb=False):
        self.config = config
        # The CLI can modify fields after the dataclass was built: validate again.
        config.validate()
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
        # Per-view EMA of S, keyed by camera slot.
        self.self_mask_ema: Dict[int, torch.Tensor] = {}
        self.self_mask_ema_beta = math.exp(-1.0 / config.gradient_mask_ema_memory_visits)
        # S_src relevance per camera slot (on CPU), valid for the signature below.
        self.source_relevance_cache: Dict[int, torch.Tensor] = {}
        self.source_relevance_signature = None

        b1 = self.config.freeu_b1
        b2 = self.config.freeu_b2
        s1= self.config.freeu_s1
        s2= self.config.freeu_s2

        register_free_upblock2d_in(self.unet, b1, b2, s1, s2)
        register_free_crossattn_upblock2d_in(self.unet, b1, b2, s1, s2)

    # ------------------------------------------------------------------
    # IP2P predictions
    # ------------------------------------------------------------------

    def _ip2p_unet(self, latents_noisy, cond_text_embedding, src_encoded, t, ca_token_indices=None):
        """One UNet pass on the IP2P batch [text + image, image only, unconditional].

        Returns (pred_text, pred_image, pred_uncond, model_input, text_embeddings,
        ca_mask). The model input and text embeddings are reused by the STG weak
        pass; ca_mask is the cross-attention mask A when ``ca_token_indices`` is
        given, else None.
        """
        uncond_embedding = self.null_text_feature
        text_embeddings = torch.cat([cond_text_embedding, uncond_embedding, uncond_embedding], dim=0)
        # Same conditioning layout as the upstream IP2P path.
        text_embeddings = torch.cat([text_embeddings, text_embeddings], dim=1)
        image_latents = torch.cat([src_encoded, src_encoded, torch.zeros_like(src_encoded)], dim=0)
        model_input = torch.cat([torch.cat([latents_noisy] * 3, dim=0), image_latents], dim=1)
        timestep_input = torch.cat([t] * 3).to(self.device)

        ca_mask = None
        if ca_token_indices is not None:
            noise_pred, ca_mask = run_unet_with_cross_attention_capture(
                self.unet,
                self.config.cross_attention_mask_layers,
                ca_token_indices,
                model_input,
                timestep_input,
                text_embeddings,
                conditioned_batch_size=latents_noisy.shape[0],
                build_attention_mask_fn=lambda maps: build_cross_attention_relevance_mask(
                    maps,
                    gamma=self.config.cross_attention_mask_gamma,
                    sigma=self.config.cross_attention_mask_blur,
                    target_shape=latents_noisy.shape[-2:],
                ),
            )
        else:
            noise_pred = self.unet.forward(
                model_input,
                timestep_input,
                encoder_hidden_states=text_embeddings,
            ).sample

        pred_text, pred_image, pred_uncond = noise_pred.chunk(3)
        return pred_text, pred_image, pred_uncond, model_input, text_embeddings, ca_mask

    def _ip2p_guidance(self, pred_text, pred_image, pred_uncond, use_text: bool):
        """IP2P classifier-free guidance; the source branch drops the text term."""
        if use_text:
            return pred_uncond + self.config.guidance_scale * (pred_text - pred_image) + \
                self.config.image_guidance_scale * (pred_image - pred_uncond)
        return pred_uncond + self.config.image_guidance_scale * (pred_image - pred_uncond)

    # ------------------------------------------------------------------
    # Localization masks
    # ------------------------------------------------------------------

    def _target_instruction_mask(self, relevance, current_spot):
        """S from the target-instruction relevance, plus the voxel-cache input.

        relevance = ||eps(render, I_src, instruction) - eps(render, I_src, "")||
        per latent pixel. Returns (mask, raw):
            mask  p5-p95 normalized, per-view EMA, gamma, blur (enters M)
            raw   divided by its per-frame 95th percentile and clipped to [0, 1].
                  The cache needs absolute foreground/background contrast,
                  which the rank normalization of ``mask`` erases.
        """
        B = relevance.shape[0]
        # A robust quantile rather than the max, so one hot pixel cannot rescale
        # the whole frame (spurious cross-view variance in the cache).
        q = min(max(float(self.config.gradient_mask_raw_norm_quantile), 0.5), 0.999)
        scale = torch.quantile(relevance.view(B, -1), q, dim=1)
        scale = scale.clamp_min(1e-8).view(B, 1, 1, 1)
        raw = (relevance / scale).clamp(0.0, 1.0).detach()

        normalized = normalize_relevance_map(relevance).detach()
        previous = self.self_mask_ema.get(current_spot)
        if previous is not None and previous.shape == normalized.shape:
            # EMA over this view's own visits (a view's first visit seeds it).
            beta = self.self_mask_ema_beta
            normalized = beta * previous + (1.0 - beta) * normalized
        self.self_mask_ema[current_spot] = normalized

        mask = apply_mask_postprocessing(
            normalized,
            gamma=self.config.gradient_mask_gamma,
            sigma=self.config.gradient_mask_blur,
        )
        return mask.detach(), raw

    @torch.no_grad()
    def _source_instruction_relevance(self, src_x0, src_encoded, current_spot):
        """S_src relevance: the instruction contrast on the noisy SOURCE latent.

        ||eps(z_src_t, I_src, instruction) - eps(z_src_t, I_src, "")|| at a fixed
        timestep, with channel magnitudes averaged over
        ``localization_source_num_samples`` noise draws. It does not depend on
        the edit's progress, so it is computed on a view's first visit and
        cached on CPU. The noise comes from a private generator seeded per view,
        so the training RNG stream is untouched.
        """
        # Changed prompts or settings make every cached map stale.
        signature = (
            self.src_prompt, self.tgt_prompt,
            self.config.localization_source_timestep_ratio,
            self.config.localization_source_num_samples,
            self.config.localization_source_seed,
        )
        if signature != self.source_relevance_signature:
            self.source_relevance_cache.clear()
            self.source_relevance_signature = signature
        expected_shape = (src_x0.shape[0], 1, *src_x0.shape[-2:])
        cached = self.source_relevance_cache.get(current_spot)
        if cached is not None and cached.shape == expected_shape:
            return cached.to(device=src_x0.device)

        num_train_steps = self.scheduler.config.num_train_timesteps
        timestep = int(round(
            self.config.localization_source_timestep_ratio * (num_train_steps - 1)
        ))
        t_local = torch.full((src_x0.shape[0],), timestep, device=src_x0.device, dtype=torch.long)
        generator = torch.Generator(device=src_x0.device)
        generator.manual_seed((self.config.localization_source_seed + int(current_spot)) % (2**63 - 1))
        embeddings = torch.cat([self.tgt_text_feature, self.null_text_feature], dim=0)
        # Same conditioning layout as the IP2P DDS path.
        embeddings = torch.cat([embeddings, embeddings], dim=1)
        image_condition = torch.cat([src_encoded, src_encoded], dim=0)
        relevance = torch.zeros(expected_shape, device=src_x0.device, dtype=torch.float32)
        for _ in range(self.config.localization_source_num_samples):
            local_noise = torch.randn(
                src_x0.shape, device=src_x0.device, dtype=src_x0.dtype, generator=generator,
            )
            noisy_source = self.scheduler.add_noise(src_x0, local_noise, t_local)
            model_input = torch.cat(
                [torch.cat([noisy_source, noisy_source], dim=0), image_condition], dim=1,
            )
            pred = self.unet.forward(
                model_input,
                torch.cat([t_local, t_local]),
                encoder_hidden_states=embeddings,
            ).sample
            pred_text, pred_image = pred.chunk(2)
            # Average magnitudes, not signed vectors that could cancel.
            relevance += (pred_text.float() - pred_image.float()).norm(dim=1, keepdim=True)
        relevance = relevance / self.config.localization_source_num_samples
        self.source_relevance_cache[current_spot] = relevance.cpu()
        return relevance

    def _source_instruction_mask(self, src_x0, src_encoded, current_spot):
        """S_src: the cached source relevance shaped like S (no EMA: it is fixed per view)."""
        relevance = self._source_instruction_relevance(src_x0, src_encoded, current_spot)
        return apply_mask_postprocessing(
            normalize_relevance_map(relevance),
            gamma=self.config.gradient_mask_gamma,
            sigma=self.config.gradient_mask_blur,
        ).detach()

    def _localization_masks(self, target_relevance, ca_mask, src_x0, src_encoded, current_spot):
        """Build S, S_src and the edit mask M = max(S * A, min(S, S_src)).

        A restricts S at full strength from the first step, which keeps the edit
        off the background. A can only subtract, so it can also veto parts of
        the requested edit that the edit tokens do not attend to (a
        stormtrooper's hands). The floor min(S, S_src) lifts that veto exactly
        where both instruction masks mark the region. M never exceeds S.
        """
        self_mask, self_mask_raw = self._target_instruction_mask(target_relevance, current_spot)
        source_mask = self._source_instruction_mask(src_x0, src_encoded, current_spot)

        internal = self_mask
        veto_floor_addition = None
        if ca_mask is not None:
            ca_mask = _match_mask_shape(ca_mask, self_mask.shape[-2:])
            hybrid = (self_mask * ca_mask).clamp(0.0, 1.0)
            internal = torch.maximum(hybrid, torch.minimum(self_mask, source_mask)).clamp(0.0, 1.0)
            veto_floor_addition = (internal - hybrid).clamp_min(0.0).detach()

        return LocalizationMasks(
            self_mask=self_mask,
            self_mask_raw=self_mask_raw,
            source_mask=source_mask,
            ca_mask=ca_mask,
            internal=internal,
            final=internal,
            veto_floor_addition=veto_floor_addition,
        )

    def _fuse_voxel_cache(self, masks, cache_mask, cache_valid, cache_confidence, cache_blend):
        """Add voxel-cache support to the internal mask (positive only).

            M = M_internal + blend * C * G * max(M_cache - M_internal, 0)
            G = max(A, S) * A

        The cache adds edit support only where its cross-view consensus exceeds
        the current view's mask, scaled by the warmup blend, the per-pixel cache
        confidence C and the semantic gate G. It never subtracts. Pixels
        without a valid cache read keep M_internal. The cache inputs are
        [1, 1, h, w] maps at latent resolution.
        """
        mask = masks.internal
        shape = mask.shape[-2:]
        cache_mask = _match_mask_shape(cache_mask, shape).to(device=mask.device, dtype=mask.dtype).clamp(0.0, 1.0)
        valid = _match_mask_shape(cache_valid.to(device=mask.device, dtype=torch.float32), shape, mode="nearest") > 0.5
        confidence = _match_mask_shape(cache_confidence, shape).to(device=mask.device, dtype=mask.dtype).clamp(0.0, 1.0)

        ca_mask = masks.ca_mask.clamp(0.0, 1.0) if masks.ca_mask is not None else None
        gate = compute_cache_gate(ca_mask, masks.self_mask.clamp(0.0, 1.0))
        blend = min(max(float(cache_blend), 0.0), 1.0)

        fused = mask + (blend * confidence) * gate * (cache_mask - mask).clamp_min(0.0)
        return torch.where(valid, fused, mask).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Guidance amplification
    # ------------------------------------------------------------------

    def _guided_target_prediction(self, eps_tgt, latents_noisy, eta, stg_scale, model_input, text_embeddings, t):
        """Target prediction after amplification: TAG(eps) + s_stg * (eps - eps_weak).

        TAG and STG act in parallel on the same IP2P prediction. eps_weak comes
        from a UNet pass with self-attention skipped in ``stg_skip_layers``.
        Returns (prediction, STG term), the STG term being None when STG is off
        at this step.
        """
        guided = apply_tag(eps_tgt, latents_noisy, eta)
        if stg_scale <= 0:
            return guided, None

        weak_text, weak_image, weak_uncond = run_unet_with_skipped_attn(
            self.unet,
            self.device,
            self.config.stg_skip_layers,
            model_input,
            t,
            text_embeddings,
        ).chunk(3)
        eps_weak = self._ip2p_guidance(weak_text, weak_image, weak_uncond, use_text=True)
        stg_term = stg_scale * (eps_tgt - eps_weak)
        return guided + stg_term, stg_term

    # ------------------------------------------------------------------
    # Upstream DreamCatalyst helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # One editing iteration
    # ------------------------------------------------------------------

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
        external_mask_blend=0.0,
    ):
        """One editing iteration on one view (steps 1-4 of the module docstring).

        Args:
            tgt_x0: latent of the current render (the loss's gradient flows here).
            src_x0: latent of the source image; src_encoded: its IP2P image condition.
            step: trainer step (logging cadence); current_spot: camera slot of the
                view, which keys the per-view caches.
            external_grad_mask, external_grad_mask_valid,
            external_grad_mask_confidence: the voxel-cache read for this view,
                [1, 1, h, w] at latent resolution (None without the cache).
            external_mask_blend: the cache's warmup blend.

        Returns the loss, or with ``return_dict`` a dict with the loss, gradient
        and masks (mask entries are None when localization is off).
        """
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        batch_size = tgt_x0.shape[0]
        t, t_prev, t_normalized = self.dc_timestep_sampling(batch_size)
        # STG follows the iteration count before this step increments it.
        iteration = self.iteration
        self.iteration += 1
        eta_tag_current = compute_tag_eta(self.config.eta_tag, t_normalized)

        # The same noise for both branches: DDS cancels it in eps_tgt - eps_src.
        noise = torch.randn_like(tgt_x0)
        noisy_tgt = self.scheduler.add_noise(tgt_x0, noise, t)
        noisy_src = self.scheduler.add_noise(src_x0, noise, t)

        # 1. IP2P predictions. The target pass also records A.
        ca_token_indices = None
        if self.config.cross_attention_mask_enabled:
            ca_token_indices = get_cross_attention_token_indices(
                self.tokenizer,
                self.tgt_prompt,
                src_prompt=self.src_prompt,
            )
        tgt_text, tgt_image, tgt_uncond, tgt_model_input, tgt_text_embeddings, ca_mask = self._ip2p_unet(
            noisy_tgt, self.tgt_text_feature, src_encoded, t, ca_token_indices
        )
        src_text, src_image, src_uncond, _, _, _ = self._ip2p_unet(
            noisy_src, self.src_text_feature, src_encoded, t
        )
        eps_tgt_raw = self._ip2p_guidance(tgt_text, tgt_image, tgt_uncond, use_text=True).detach().clone()
        eps_src = self._ip2p_guidance(src_text, src_image, src_uncond, use_text=False).detach().clone()

        # 2. Localization, from the clean predictions (TAG/STG would otherwise
        #    feed their own amplification back into the masks).
        masks = None
        if self.config.localization_enabled:
            target_relevance = (
                tgt_text.detach().float() - tgt_image.detach().float()
            ).norm(dim=1, keepdim=True)
            masks = self._localization_masks(target_relevance, ca_mask, src_x0, src_encoded, current_spot)
            if external_grad_mask is not None:
                masks.final = self._fuse_voxel_cache(
                    masks,
                    external_grad_mask,
                    external_grad_mask_valid,
                    external_grad_mask_confidence,
                    external_mask_blend,
                )

        # 3. Amplify the target prediction; the source prediction stays as is.
        edit_strength = compute_edit_strength(eps_tgt_raw, eps_src)
        stg_scale = compute_stg_scale(
            base_scale=self.config.stg_scale,
            iteration=iteration,
            max_iteration=self.max_iteration,
            start_ratio=self.config.stg_schedule_start_ratio,
            end_ratio=self.config.stg_schedule_end_ratio,
            bump_peak_ratio=self.config.stg_bump_peak_ratio,
            current_edit_strength=edit_strength,
        )
        eps_tgt, stg_term = self._guided_target_prediction(
            eps_tgt_raw, noisy_tgt, eta_tag_current, stg_scale, tgt_model_input, tgt_text_embeddings, t
        )

        # 4. Gradient: localized DDS delta + preservation + latent-mean anchor.
        edit_mask = masks.final if masks is not None else None
        preservation_mask = masks.source_mask if masks is not None else None
        eps_tgt_for_grad = eps_tgt
        if self.config.source_blend_localization_enabled and edit_mask is not None:
            eps_tgt_for_grad = apply_source_blend(eps_tgt, eps_src, edit_mask)

        preserve_weight = compute_preserve_weight(
            psi=self.config.psi,
            preservation_mask=preservation_mask,
            outside_mask_anchor_weight=self.config.outside_mask_anchor_weight,
            edit_strength=edit_strength,
        )
        w_DDS = self.config.delta + self.config.gamma * (t_normalized ** (1/math.e))
        grad = (
            w_DDS * (eps_tgt_for_grad - eps_src)
            + math.exp(t_normalized) * preserve_weight * (tgt_x0 - src_x0)
        )
        grad = apply_latent_mean_anchor(grad, tgt_x0, src_x0, self.config.latent_mean_anchor_weight)
        grad = torch.nan_to_num(grad)

        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size

        if self.use_wandb and step % self.config.log_step == 0:
            _, pred_x0_tgt = self.compute_posterior_mean(noisy_tgt, eps_tgt, t, t_prev)
            _, pred_x0_src = self.compute_posterior_mean(noisy_src, eps_src, t, t_prev)
            log_dc_debug_to_wandb(
                step=step,
                current_spot=current_spot,
                t=t,
                t_normalized=t_normalized,
                eta_tag_current=eta_tag_current,
                current_stg_scale=stg_scale,
                current_edit_strength=edit_strength,
                w_dds=w_DDS,
                preserve_weight=preserve_weight,
                eps_tgt=eps_tgt,
                eps_src=eps_src,
                eps_tgt_for_grad=eps_tgt_for_grad,
                grad=grad,
                pred_x0_tgt=self.decode_latent(pred_x0_tgt),
                pred_x0_src=self.decode_latent(pred_x0_src),
                tensor_to_pil_fn=tensor_to_pil,
                resize_image_fn=resize_image,
                masks=masks,
                stg_perturbation=stg_term,
                source_blend_enabled=self.config.source_blend_localization_enabled,
                self_mask_ema_beta=self.self_mask_ema_beta,
            )

        if not return_dict:
            return loss
        return {
            "loss": loss,
            "grad": grad,
            "t": t,
            "grad_mask": edit_mask,
            "internal_grad_mask": masks.internal if masks is not None else None,
            "preservation_mask": preservation_mask,
            "self_grad_mask": masks.self_mask if masks is not None else None,
            "self_grad_mask_raw": masks.self_mask_raw if masks is not None else None,
            "cross_attention_mask": masks.ca_mask if masks is not None else None,
            "edit_strength": edit_strength,
            "stg_scale": stg_scale,
        }

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
