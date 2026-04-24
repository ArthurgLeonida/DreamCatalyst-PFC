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
from dc.guidance_utils import (
    apply_latent_mean_anchor,
    apply_perp_neg,
    apply_source_blend,
    apply_stg,
    apply_tag,
    compute_mask_coverage,
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

    eta_tag: float = 1.0
    adaptive_tag: bool = False
    asymmetric_tag: bool = False

    perp_neg: bool = False
    perp_neg_alpha: float = 1.0

    stg_enabled: bool = False
    stg_scale: float = 0.5
    stg_skip_layers: List[int] = field(default_factory=lambda: [2])
    stg_schedule_enabled: bool = False
    stg_schedule_start_ratio: float = 0.0
    stg_schedule_end_ratio: float = 1.0
    stg_schedule_mode: str = "decay"
    stg_bump_peak_ratio: float = 0.5
    stg_coverage_adaptive: bool = False

    # Self-derived relevance masking
    gradient_mask_enabled: bool = False
    gradient_mask_blur: float = 3.0
    gradient_mask_ema_beta: float = 0.9
    gradient_mask_gamma: float = 1.0
    gradient_mask_warmup: int = 50
    source_blend_localization_enabled: bool = False

    outside_mask_anchor_weight: float = 0.0
    outside_mask_anchor_coverage_adaptive: bool = False

    cross_attention_mask_enabled: bool = False
    cross_attention_mask_layers: List[int] = field(default_factory=lambda: [1, 2])
    cross_attention_mask_weight: float = 1.0
    cross_attention_mask_blur: float = 0.0
    cross_attention_mask_gamma: float = 1.0

    # DaCapo-inspired ψ schedule (Huang et al., CVPR 2025).
    psi_late_multiplier: float = 1.0

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

    def _get_current_stg_scale(self, current_mask_coverage=None, iteration=None) -> float:
        """Return scheduled STG scale, optionally using current same-view coverage."""
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
            coverage_adaptive=self.config.stg_coverage_adaptive,
            current_mask_coverage=current_mask_coverage,
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
        cross_attention_mask = None
        cross_attention_token_indices = None
        if self.config.cross_attention_mask_enabled:
            cross_attention_token_indices = get_cross_attention_token_indices(
                self.tokenizer,
                self.tgt_prompt,
                src_prompt=self.src_prompt,
            )

        uncond_image_latent = torch.zeros_like(src_encoded)

        # Phase 1: clean CFG predictions only. These eps_raw tensors are the
        # localization source of truth and are intentionally pre-STG/pre-TAG/pre-PN.
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

            if name == "tgt":
                timestep_input = torch.cat([t] * 3).to(device)
                if self.config.cross_attention_mask_enabled:
                    noise_pred, cross_attention_mask = run_unet_with_cross_attention_capture(
                        self.unet,
                        self.config.cross_attention_mask_layers,
                        cross_attention_token_indices,
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
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_image) + \
                    self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            else:
                timestep_input = torch.cat([t] * 3).to(device)
                noise_pred = self.unet.forward(
                    base_latent_model_input,
                    timestep_input,
                    encoder_hidden_states=base_text_embeddings,
                ).sample
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)

            eps_raw[name] = noise_pred.detach().clone()

        # STG schedule uses the pre-increment iteration so progress starts at 0
        # on the first call. After the increment, self.iteration drives the mask
        # warmup check below: with gradient_mask_warmup=N, the self-mask stays as
        # all-ones for post-increment iterations 1..N-1 and switches to the real
        # EMA mask at post-increment iteration N.
        iteration_for_stg = self.iteration
        self.iteration += 1

        grad_mask = None
        self_grad_mask = None
        needs_self_mask = (
            self.config.gradient_mask_enabled
            or self.config.source_blend_localization_enabled
            or self.config.outside_mask_anchor_weight > 0
            or self.config.cross_attention_mask_enabled
            or self.config.stg_coverage_adaptive
        )

        if needs_self_mask:
            self_grad_mask = self._build_gradient_relevance_mask(
                eps_raw["tgt"], eps_raw["src"], current_spot
            )
            grad_mask = self_grad_mask

        if cross_attention_mask is not None:
            target_shape = grad_mask.shape[-2:] if grad_mask is not None else eps_raw["tgt"].shape[-2:]
            if cross_attention_mask.shape[-2:] != target_shape:
                cross_attention_mask = F.interpolate(
                    cross_attention_mask,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            if grad_mask is None:
                grad_mask = cross_attention_mask
            else:
                weight = float(self.config.cross_attention_mask_weight)
                weight = min(max(weight, 0.0), 1.0)
                self_mask = grad_mask
                grad_mask = self_mask * ((1.0 - weight) + weight * cross_attention_mask)
            grad_mask = grad_mask.clamp(0.0, 1.0)

        mask_is_warmup = needs_self_mask and self.iteration < self.config.gradient_mask_warmup
        current_mask_coverage = None if mask_is_warmup else compute_mask_coverage(grad_mask)
        current_stg_scale = self._get_current_stg_scale(
            current_mask_coverage=current_mask_coverage,
            iteration=iteration_for_stg,
        )

        # Phase 2: apply guidance novelties from cached eps_raw after the clean current mask exists.
        for name in ["tgt", "src"]:
            noise_pred = eps_raw[name]
            latents_noisy = noisy_latents[name]

            if name == "tgt" and self.config.stg_enabled and current_stg_scale > 0:
                weak_pred = run_unet_with_skipped_attn(
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
                noise_pred = apply_stg(noise_pred, noise_pred_weak, current_stg_scale)

            eta_current = eta_tag_current if (name == "tgt" or not self.config.asymmetric_tag) else 1.0
            noise_pred = apply_tag(noise_pred, latents_noisy, eta_current)

            _, pred_x0 = self.compute_posterior_mean(latents_noisy, noise_pred, t, t_prev)
            eps[name] = noise_pred
            pred_x0s[name] = pred_x0

        if self.config.perp_neg:
            eps["tgt"] = apply_perp_neg(eps["tgt"], eps["src"], self.config.perp_neg_alpha)

        eps_tgt_for_grad = eps["tgt"]
        if self.config.source_blend_localization_enabled and grad_mask is not None:
            eps_tgt_for_grad = apply_source_blend(eps["tgt"], eps["src"], grad_mask)

        preserve_weight = compute_preserve_weight(
            psi=self.config.psi,
            psi_late_multiplier=self.config.psi_late_multiplier,
            t_normalized=t_normalized,
            grad_mask=grad_mask,
            outside_mask_anchor_weight=self.config.outside_mask_anchor_weight,
            outside_mask_anchor_coverage_adaptive=self.config.outside_mask_anchor_coverage_adaptive,
            mask_coverage=current_mask_coverage,
        )

        w_DDS = self.config.delta + self.config.gamma * (t_normalized ** (1/math.e))
        grad = (
            w_DDS * (eps_tgt_for_grad - eps["src"])
            + math.exp(t_normalized) * preserve_weight * (tgt_x0 - src_x0)
        )

        if self.config.gradient_mask_enabled and grad_mask is not None:
            grad = grad * grad_mask

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
                stg_mask_coverage=current_mask_coverage,
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
                "mask_coverage": current_mask_coverage,
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
