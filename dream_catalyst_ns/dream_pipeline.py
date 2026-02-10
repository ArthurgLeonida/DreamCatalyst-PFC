"""
DreamCatalyst Custom Pipeline
==============================
Skeleton that inherits from VanillaPipeline.
Override ``get_train_loss_dict`` to inject custom SDS / IP2P logic later.

This pipeline is designed for **Gaussian Splatting** (Splatfacto).
Splatfacto uses full-image rendering (not ray bundles), so the datamanager
yields ``(camera, batch)`` pairs where ``camera`` is a ``Cameras`` object.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DreamCatalystPipelineConfig(VanillaPipelineConfig):
    """Configuration for the DreamCatalyst pipeline."""

    _target: Type = field(default_factory=lambda: DreamCatalystPipeline)
    """Target class to instantiate."""
    datamanager: DataManagerConfig = field(default_factory=FullImageDatamanagerConfig)
    """Specifies the datamanager config (full-image for Gaussian Splatting)."""
    model: ModelConfig = field(default_factory=SplatfactoModelConfig)
    """Specifies the model config (Splatfacto by default)."""

    # ── Future knobs (uncomment / extend as needed) ──
    # edit_prompt: str = "make it look like a golden chair"
    # guidance_scale: float = 7.5
    # image_guidance_scale: float = 1.5
    # sds_loss_weight: float = 1.0
    # ip2p_model_name: str = "timbrooks/instruct-pix2pix"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class DreamCatalystPipeline(VanillaPipeline):
    """Custom pipeline for DreamCatalyst-style 3D editing with Gaussian Splatting.

    Right now this is a *skeleton* – it simply wraps VanillaPipeline and adds
    a print statement in ``get_train_loss_dict`` so you can verify that your
    custom code path is being executed.

    Later you will:
      1. Load an InstructPix2Pix / Stable-Diffusion model in ``__init__``.
      2. Override ``get_train_loss_dict`` to compute an SDS loss
         (or the DreamCatalyst variant) using the rendered Gaussian Splat image.
      3. Implement the "creative catalyst" sampling from the DreamCatalyst paper.
    """

    config: DreamCatalystPipelineConfig

    def __init__(
        self,
        config: DreamCatalystPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        # ------------------------------------------------------------------
        # TODO: initialise diffusion guidance model here, e.g.
        #   from diffusers import StableDiffusionInstructPix2PixPipeline
        #   self.ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        #       "timbrooks/instruct-pix2pix", torch_dtype=torch.float16,
        #       safety_checker=None,
        #   ).to(device)
        #   self.ip2p.set_progress_bar_config(disable=True)
        # ------------------------------------------------------------------
        print("\n[DreamCatalyst] Pipeline initialised – Gaussian Splatting mode!\n")

    # ------------------------------------------------------------------
    # Training step override
    # ------------------------------------------------------------------
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """Override the training loss computation.

        In Splatfacto the datamanager yields full images:
          - ``camera``: a Cameras object (single training view)
          - ``batch``:  dict with ``"image"`` [H, W, 3], optionally ``"mask"``, etc.

        The model's ``get_outputs(camera)`` renders a full image and returns
        ``model_outputs["rgb"]`` of shape [1, H, W, 3] (or [H, W, 3]).

        Currently delegates to the parent (VanillaPipeline), which already
        handles the Splatfacto flow correctly, and adds a proof-of-life print.
        """
        # ── 1. Call the parent to get the standard Gaussian Splatting losses ──
        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)

        # ── 2. Print proof-of-life every 100 steps ──
        if step % 100 == 0:
            print(f"[DreamCatalyst] Training step {step} – "
                  f"recon loss = {sum(v.item() for v in loss_dict.values()):.4f}")

        # ── 3. (TODO) Compute SDS / IP2P loss and add it ──
        # The rendered image is available as model_outputs["rgb"]  → [H, W, 3]
        # The original GT image is in batch["image"]               → [H, W, 3]
        #
        # DreamCatalyst workflow:
        #   rendered = model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        #   sds_loss = self._compute_sds_loss(rendered, step)
        #   loss_dict["sds_loss"] = self.config.sds_loss_weight * sds_loss

        return model_outputs, loss_dict, metrics_dict

    # ------------------------------------------------------------------
    # Placeholder for future SDS logic
    # ------------------------------------------------------------------
    # def _compute_sds_loss(
    #     self,
    #     rendered_image: torch.Tensor,   # [1, 3, H, W]
    #     step: int,
    # ) -> torch.Tensor:
    #     """Compute the Score Distillation Sampling loss (DreamCatalyst variant).
    #
    #     Steps:
    #       1. Sample a random noise level t ~ U(t_min, t_max)
    #       2. Add noise to the rendered image: x_t = √ᾱ_t * x + √(1-ᾱ_t) * ε
    #       3. Run the IP2P U-Net to predict noise: ε_pred = unet(x_t, t, text, image)
    #       4. Compute classifier-free guidance with both text and image conditioning
    #       5. SDS gradient: ∇ = w(t) * (ε_pred - ε)
    #       6. Loss = (∇.detach() * rendered_image).sum()   (stop-gradient trick)
    #     """
    #     pass
