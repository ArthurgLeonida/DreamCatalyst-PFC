from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import DDIMScheduler, DiffusionPipeline
from jaxtyping import Float
from PIL import Image
from typing import List, Dict
from dc.dc_unet import CustomUNet2DConditionModel
from dc.utils.free_lunch import register_free_upblock2d_in, register_free_crossattn_upblock2d_in
import math


class STGIdentityValueAttnProcessor:
    """Paper-faithful STG-A processor.

    The STG paper describes attention skip by replacing the attention matrix
    A with the identity I so that SA'(Q, K, V) = IV = V. This processor
    implements that directly in diffusers' attention path by skipping the
    query/key attention-score computation while preserving the value and output
    projections.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # STG-A: replace A with I so the branch output becomes the projected
        # value path rather than A @ V.
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)
        hidden_states = attn.batch_to_head_dim(value)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


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

    # STG (Self-attention skip guidance) — replace CFG with structure-preserving perturbation
    stg_enabled: bool = False
    stg_scale: float = 0.5
    stg_skip_layers: List[int] = field(default_factory=lambda: [2])

    # Self-derived relevance masking — localize the DDS gradient using the
    # model's own tgt/src prediction discrepancy.
    gradient_mask_enabled: bool = False
    gradient_mask_blur: float = 3.0
    gradient_mask_ema_beta: float = 0.9
    gradient_mask_gamma: float = 1.0
    gradient_mask_warmup: int = 50
    source_blend_localization_enabled: bool = False
    outside_mask_anchor_weight: float = 0.0


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
        flat = relevance.flatten(1) # B, H*W
        p5 = torch.quantile(flat, 0.05, dim=1, keepdim=True).view(-1, 1, 1, 1)
        p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True).view(-1, 1, 1, 1)
        normalized = ((relevance - p5) / (p95 - p5 + 1e-8)).clamp(0.0, 1.0)

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

        gamma = self.config.gradient_mask_gamma
        if gamma != 1.0:
            mask = mask.clamp_min(0.0).pow(gamma)

        sigma = self.config.gradient_mask_blur
        if sigma > 0:
            kernel_size = max(3, int(round(6 * sigma + 1)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask = TF.gaussian_blur(mask, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        return mask.clamp(0.0, 1.0).detach()

        
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

    def _run_unet_with_skipped_attn(self, latent_model_input, t, text_embeddings):
        """Run UNet forward pass with STG-A attention skip in selected up_blocks.

        Preferred path: temporarily swap attn1's processor with a paper-faithful
        processor that replaces the attention map A with I so SA'(Q,K,V)=IV=V.

        Fallback path: if processor swapping is unavailable, zero the branch
        output so the surrounding transformer residual behaves like an identity
        skip at the block level.
        """
        hooks = []
        original_processors = []
        stg_processor = STGIdentityValueAttnProcessor()

        def _zero_attn_branch_output(module, inputs, output):
            """Approximate STG-A when a processor-level override is unavailable."""
            return torch.zeros_like(output)

        try:
            for layer_idx in self.config.stg_skip_layers:
                block = self.unet.up_blocks[layer_idx]
                if not hasattr(block, "attentions"):
                    continue
                for attn_module in block.attentions:
                    for transformer_block in attn_module.transformer_blocks:
                        attn = transformer_block.attn1
                        if hasattr(attn, "processor"):
                            original_processors.append((attn, attn.processor))
                            if hasattr(attn, "set_processor"):
                                attn.set_processor(stg_processor)
                            else:
                                attn.processor = stg_processor
                        else:
                            hook = attn.register_forward_hook(_zero_attn_branch_output)
                            hooks.append(hook)

            with torch.no_grad():
                output = self.unet.forward(
                    latent_model_input,
                    torch.cat([t] * 3).to(self.device),
                    encoder_hidden_states=text_embeddings,
                )
            return output.sample
        finally:
            for attn, processor in original_processors:
                if hasattr(attn, "set_processor"):
                    attn.set_processor(processor)
                else:
                    attn.processor = processor
            for hook in hooks:
                hook.remove()

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
        pred_x0s = dict()
        noisy_latents = dict()
        
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
                    text_embeddings = torch.cat(
                        [cond_text_embedding, neg_text_embedding, uncond_embedding, uncond_embedding], dim=0
                    )
                    text_embeddings = torch.cat([text_embeddings, text_embeddings], dim=1)
                    latent_image = torch.cat([src_encoded, src_encoded, src_encoded, uncond_image_latent], dim=0)
                    latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                    latent_model_input = torch.cat([latent_model_input, latent_image], dim=1)

                    noise_pred = self.unet.forward(
                        latent_model_input,
                        torch.cat([t] * 4).to(device),
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

                    noise_pred = self.unet.forward(
                        latent_model_input,
                        torch.cat([t] * 3).to(device),
                        encoder_hidden_states=text_embeddings,
                    ).sample
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_image) + \
                        self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)

                # STG: amplify structural signal beyond full model (paper Eq 13)
                # eps_stg = eps_full + stg_scale * (eps_full - eps_weak)
                # stg_scale=0 → no effect, stg_scale=1.0 → paper default (STG-R)
                # ==============================================================================
                if self.config.stg_enabled:
                    weak_pred = self._run_unet_with_skipped_attn(
                        base_latent_model_input, t, base_text_embeddings
                    )
                    weak_text, weak_image, weak_uncond = weak_pred.chunk(3)
                    noise_pred_weak = weak_uncond + self.config.guidance_scale * (weak_text - weak_image) + \
                        self.config.image_guidance_scale * (weak_image - weak_uncond)
                    noise_pred = noise_pred + self.config.stg_scale * (noise_pred - noise_pred_weak)
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
        if (
            self.config.gradient_mask_enabled
            or self.config.source_blend_localization_enabled
            or self.config.outside_mask_anchor_weight > 0
        ):
            grad_mask = self._build_gradient_relevance_mask(
                eps["tgt"], eps["src"], current_spot
            )

        eps_tgt_for_grad = eps["tgt"]
        if self.config.source_blend_localization_enabled and grad_mask is not None:
            # Source-blended localization:
            # low-mask regions naturally fall back toward the source branch,
            # reducing DDS pressure on background / non-target structure.
            eps_tgt_for_grad = eps["src"] + grad_mask * (eps["tgt"] - eps["src"])

        preserve_weight = self.config.psi
        if grad_mask is not None and self.config.outside_mask_anchor_weight > 0:
            # Strengthen x0/source anchoring outside the edit region so smooth
            # backgrounds (walls, floors) remain closer to the original scene.
            preserve_weight = preserve_weight + self.config.outside_mask_anchor_weight * (1.0 - grad_mask)

        w_DDS = self.config.delta + self.config.gamma * (t_normalized ** (1/math.e))
        grad = (
            w_DDS * (eps_tgt_for_grad - eps["src"])
            + math.exp(t_normalized) * preserve_weight * (tgt_x0 - src_x0)
        )

        if self.config.gradient_mask_enabled and grad_mask is not None:
            grad = grad * grad_mask

        grad = torch.nan_to_num(grad)
        
        target = (tgt_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size 
        
        
        if self.use_wandb:
            import wandb
            wandb.log({
                f"target_prediction_x0_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(pred_x0s["tgt"])), min_size=256), caption=f"{t.item()}"),
                f"source_prediction_x0_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(pred_x0s["src"])), min_size=256), caption=f"{t.item()}"),
                f"target_noise_prediction_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(eps["tgt"])), min_size=256), caption=f"{t.item()}"),
                f"source_noise_prediction_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(eps["src"])), min_size=256), caption=f"{t.item()}"),
                f"target_noisy_latents_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(noisy_latents["tgt"])), min_size=256), caption=f"{t.item()}"),
                f"source_noisy_latents_{current_spot}": wandb.Image(resize_image(tensor_to_pil(self.decode_latent(noisy_latents["src"])), min_size=256), caption=f"{t.item()}"),
            }, step=step, commit=False) if step % self.config.log_step == 0 else None
        
        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t, "grad_mask": grad_mask}
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
