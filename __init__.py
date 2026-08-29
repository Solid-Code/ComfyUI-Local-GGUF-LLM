from pathlib import Path

# Keep exactly one version of each frontend module. This matters when users
# extract an update over an existing custom-node directory because ComfyUI loads
# every JS module found under WEB_DIRECTORY.
_js_dir = Path(__file__).resolve().parent / "web" / "js"
_FRONTENDS = {
    "local_llm_server": "local_llm_server_v099.js",
    "prompt_enhancer": "prompt_enhancer_dom_v0624.js",
}
if _js_dir.is_dir():
    for _candidate in _js_dir.glob("*.js"):
        name = _candidate.name
        keep = True
        if name.startswith("local_llm_server"):
            keep = name == _FRONTENDS["local_llm_server"]
        elif name.startswith("prompt_enhancer"):
            keep = name == _FRONTENDS["prompt_enhancer"]
        elif name.startswith("h3_shot_generator"):
            # Migration from v0.18.40-and-earlier combined builds: H3 now lives
            # in the sibling ComfyUI-H3-Shot-Generator package.
            keep = False
        if not keep:
            try:
                _candidate.unlink()
            except OSError:
                pass

# Also remove the legacy backend module when this split package is extracted
# over an older combined install. It is intentionally no longer imported or
# registered by this package.
_legacy_h3_backend = Path(__file__).resolve().parent / "h3_shot_generator.py"
if _legacy_h3_backend.is_file():
    try:
        _legacy_h3_backend.unlink()
    except OSError:
        pass

from .nodes import LocalGGUFLLMAPI
from .service import SERVICE, SAMPLER_PRESET_FIELDS, LocalLLMServiceAPI, LocalLLMGenerate, LocalLLMSettings
from .prompt_enhancer import LocalLLMPromptEnhancer

# Stable in-process integration point for sibling custom-node packages.  This
# deliberately exposes only the reusable Local LLM service and the sampler
# override field names; consumers do not need to import this package by its
# filesystem folder name (which contains hyphens).
import sys
import types

_bridge = types.ModuleType("comfyui_local_gguf_llm_bridge")
_bridge.SERVICE = SERVICE
_bridge.SAMPLER_PRESET_FIELDS = SAMPLER_PRESET_FIELDS
_bridge.API_VERSION = 1
sys.modules["comfyui_local_gguf_llm_bridge"] = _bridge

# LocalGGUFLLM remains the internal engine behind the persistent service and is
# intentionally not registered as a standalone ComfyUI node.
NODE_CLASS_MAPPINGS = {
    "LocalLLMGenerate": LocalLLMGenerate,
    "LocalLLMSettings": LocalLLMSettings,
    "LocalLLMPromptEnhancer": LocalLLMPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalLLMGenerate": "Local LLM Generate",
    "LocalLLMSettings": "Local LLM Settings",
    "LocalLLMPromptEnhancer": "Local LLM Prompt Enhancer",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "LocalGGUFLLMAPI",
    "LocalLLMServiceAPI",
    "LocalLLMSettings",
    "LocalLLMGenerate",
    "LocalLLMPromptEnhancer",
    "SERVICE",
]
