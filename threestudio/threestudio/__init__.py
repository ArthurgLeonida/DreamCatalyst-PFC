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

try:
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
except ImportError:
    debug = logger.debug
    info = logger.info
    warn = logger.warning


# Lazy module loading: modules are NOT imported at package init time.
# Call threestudio.ensure_loaded() once before using threestudio.find().
import importlib

_loaded = False


def _safe_import(module_path: str) -> None:
    """Import a module, logging but not crashing on errors."""
    try:
        importlib.import_module(module_path)
    except Exception as e:
        logger.warning(f"Could not import {module_path}: {e}")


def ensure_loaded() -> None:
    """Import all threestudio submodules so @register decorators fire.
    Safe to call multiple times — only runs once."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    # Order matters: base modules first, then @register-decorated modules.

    # 1. Data modules
    _safe_import("threestudio.data.multiview")
    _safe_import("threestudio.data.uncond")

    # 2. Prompt processor BASE (provides PromptProcessorOutput, no @register)
    _safe_import("threestudio.models.prompt_processors.base")

    # 3. Concrete prompt processors (use @threestudio.register)
    _safe_import("threestudio.models.prompt_processors.stable_diffusion_prompt_processor")

    # 4. Guidance modules (depend on prompt_processors.base, use @register)
    _safe_import("threestudio.models.guidance.instructpix2pix_guidance")
    _safe_import("threestudio.models.guidance.stable_diffusion_guidance")
    _safe_import("threestudio.models.guidance.stable_diffusion_sdi_guidance")

    # 5. System modules (depend on everything above)
    _safe_import("threestudio.systems.dreamfusion")
    _safe_import("threestudio.systems.instructnerf2nerf")
    _safe_import("threestudio.systems.sdi")
