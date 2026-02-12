__modules__ = {}
__version__ = "0.2.3"


def register(name):
    def decorator(cls):
        if name in __modules__:
            raise ValueError(
                f"Module {name} already exists! Names of extensions conflict!"
            )
        else:
            __modules__[name] = cls
        return cls

    return decorator


def find(name):
    if ":" in name:
        main_name, sub_name = name.split(":")
        if "," in sub_name:
            name_list = sub_name.split(",")
        else:
            name_list = [sub_name]
        name_list.append(main_name)
        NewClass = type(
            f"{main_name}.{sub_name}",
            tuple([__modules__[name] for name in name_list]),
            {},
        )
        return NewClass
    return __modules__[name]


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)

debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


# Import subpackages.
# Order matters: base modules first, then modules with @register decorators.
import importlib


def _safe_import(module_path: str) -> None:
    """Import a module, logging but not crashing on ImportError."""
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        logger.warning(f"Could not import {module_path}: {e}")


# 1. Data modules (no cross-deps on models/systems)
_safe_import("threestudio.data.multiview")
_safe_import("threestudio.data.uncond")

# 2. Prompt processor BASE only (provides PromptProcessorOutput, no @register)
_safe_import("threestudio.models.prompt_processors.base")

# 3. Guidance modules (depend on prompt_processors.base)
_safe_import("threestudio.models.guidance.instructpix2pix_guidance")
_safe_import("threestudio.models.guidance.stable_diffusion_guidance")
_safe_import("threestudio.models.guidance.stable_diffusion_sdi_guidance")

# 4. Concrete prompt processors (use @threestudio.register — register/find must be ready)
_safe_import("threestudio.models.prompt_processors.stable_diffusion_prompt_processor")

# 5. System modules (depend on everything above)
_safe_import("threestudio.systems.dreamfusion")
_safe_import("threestudio.systems.instructnerf2nerf")
_safe_import("threestudio.systems.sdi")
