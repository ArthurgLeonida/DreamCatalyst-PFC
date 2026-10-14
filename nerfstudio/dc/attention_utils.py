"""Attention hooks for the two extra UNet passes.

STG: re-run the UNet with the self-attention of selected up-blocks replaced by
     the identity (``STGIdentityValueAttnProcessor``) to get a "weak"
     prediction; STG extrapolates away from it.
CA:  run the target UNet pass while recording the cross-attention maps of the
     edit tokens (``CrossAttentionCaptureProcessor``) to build the semantic
     mask A.

Both swap diffusers attention processors inside ``_swapped_processors``, which
always restores the originals.
"""

import math
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

import torch


class STGIdentityValueAttnProcessor:
    """Self-attention with the attention matrix replaced by the identity (STG-A).

    The STG paper skips attention by setting A = I, so SA(Q, K, V) = V. This
    processor skips the query/key scores and keeps the value and output
    projections, which is exactly that.
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

        # A = I: the output is the projected values themselves.
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


class CrossAttentionMapRecorder:
    """Collect the cross-attention maps of selected text tokens.

    For each recorded layer, averages the attention probabilities over the
    chosen tokens and the heads, keeping only the first
    ``conditioned_batch_size`` items (the text-conditioned part of the IP2P
    batch), and stores one [B, 1, h, w] map.
    """

    def __init__(
        self,
        token_indices: List[int],
        conditioned_batch_size: int,
        reference_spatial_shape: Optional[Tuple[int, int]] = None,
    ):
        self.token_indices = sorted(set(token_indices))
        self.conditioned_batch_size = conditioned_batch_size
        self.reference_spatial_shape = reference_spatial_shape
        self.maps: List[torch.Tensor] = []

    def _infer_spatial_shape(self, query_tokens: int) -> Optional[Tuple[int, int]]:
        """Recover (h, w) from the number of query tokens.

        diffusers hands the processor flattened [B, h*w, C] states, so the
        layer's grid is the divisor pair of ``query_tokens`` whose aspect ratio
        is closest to the latent's (both orientations, for portrait frames).
        """
        if self.reference_spatial_shape is None:
            side = int(round(math.sqrt(query_tokens)))
            if side * side != query_tokens:
                return None
            return (side, side)

        ref_h, ref_w = self.reference_spatial_shape
        if ref_h <= 0 or ref_w <= 0:
            return None

        target_ratio = ref_w / ref_h
        best_shape = None
        best_error = None
        for divisor in range(1, int(math.sqrt(query_tokens)) + 1):
            if query_tokens % divisor != 0:
                continue
            other = query_tokens // divisor
            for (h, w) in ((divisor, other), (other, divisor)):
                ratio_error = abs(math.log((w / h) / target_ratio))
                if best_error is None or ratio_error < best_error:
                    best_error = ratio_error
                    best_shape = (h, w)
        return best_shape

    def record(
        self,
        attention_probs: torch.Tensor,
        num_heads: int,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Store the token-averaged map of one attention call."""
        if not self.token_indices:
            return

        batch_times_heads, query_tokens, key_tokens = attention_probs.shape
        if num_heads <= 0 or batch_times_heads % num_heads != 0:
            return

        batch_size = batch_times_heads // num_heads
        if batch_size < self.conditioned_batch_size:
            return

        valid_token_indices = [idx for idx in self.token_indices if 0 <= idx < key_tokens]
        if not valid_token_indices:
            return

        attn = attention_probs.view(batch_size, num_heads, query_tokens, key_tokens)
        attn = attn[: self.conditioned_batch_size]
        token_map = attn[..., valid_token_indices].mean(dim=-1).mean(dim=1)

        if spatial_shape is None:
            spatial_shape = self._infer_spatial_shape(query_tokens)
            if spatial_shape is None:
                return

        token_map = token_map.view(self.conditioned_batch_size, 1, *spatial_shape)
        self.maps.append(token_map.detach())


class CrossAttentionCaptureProcessor:
    """The default attention computation, plus a hand-off of the probabilities to a recorder."""

    def __init__(self, recorder: CrossAttentionMapRecorder):
        self.recorder = recorder

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
        spatial_shape = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_shape = (height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if hasattr(attn, "prepare_attention_mask"):
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.recorder.record(attention_probs, getattr(attn, "heads", 1), spatial_shape=spatial_shape)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, *spatial_shape)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


def _up_block_attentions(unet, layer_indices: List[int], which: str) -> List[torch.nn.Module]:
    """Attention modules of the given up-blocks: "attn1" (self) or "attn2" (cross).

    Out-of-range indices and blocks without attention (up_blocks[0] in SD 1.5)
    are skipped.
    """
    modules = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(unet.up_blocks):
            continue
        for attn_module in getattr(unet.up_blocks[layer_idx], "attentions", []):
            for transformer_block in attn_module.transformer_blocks:
                modules.append(getattr(transformer_block, which))
    return modules


@contextmanager
def _swapped_processors(attention_modules: List[torch.nn.Module], processor):
    """Temporarily give every module in ``attention_modules`` the same processor."""
    originals = [(attn, attn.processor) for attn in attention_modules]
    try:
        for attn, _ in originals:
            attn.set_processor(processor)
        yield
    finally:
        for attn, original in originals:
            attn.set_processor(original)


def run_unet_with_skipped_attn(unet, device, skip_layers, latent_model_input, t, text_embeddings):
    """STG weak pass: the UNet with self-attention skipped in ``skip_layers``.

    ``latent_model_input`` and ``text_embeddings`` are the 3-way IP2P batch of
    the regular target pass, so the weak prediction is combined the same way.
    """
    modules = _up_block_attentions(unet, skip_layers, "attn1")
    with _swapped_processors(modules, STGIdentityValueAttnProcessor()), torch.no_grad():
        output = unet.forward(
            latent_model_input,
            torch.cat([t] * 3).to(device),
            encoder_hidden_states=text_embeddings,
        )
    return output.sample


def run_unet_with_cross_attention_capture(
    unet,
    cross_attention_layers,
    token_indices,
    latent_model_input,
    t,
    text_embeddings,
    conditioned_batch_size,
    build_attention_mask_fn: Callable[[List[torch.Tensor]], Optional[torch.Tensor]],
):
    """One regular UNet pass that also records the edit tokens' cross-attention.

    Returns (noise prediction, mask), where the mask is
    ``build_attention_mask_fn`` applied to the recorded maps, or None when
    there are no tokens to record. The prediction is identical to a plain
    forward pass: the capture processor computes attention the default way.
    """
    if not token_indices:
        output = unet.forward(latent_model_input, t, encoder_hidden_states=text_embeddings)
        return output.sample, None

    recorder = CrossAttentionMapRecorder(
        token_indices=token_indices,
        conditioned_batch_size=conditioned_batch_size,
        reference_spatial_shape=latent_model_input.shape[-2:],
    )
    modules = _up_block_attentions(unet, cross_attention_layers, "attn2")
    with _swapped_processors(modules, CrossAttentionCaptureProcessor(recorder)):
        output = unet.forward(latent_model_input, t, encoder_hidden_states=text_embeddings)
    return output.sample, build_attention_mask_fn(recorder.maps)
