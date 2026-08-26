from pathlib import Path

# Keep exactly one version of each frontend module. This matters when users
# extract an update over an existing custom-node directory because ComfyUI loads
# every JS module found under WEB_DIRECTORY.
_js_dir = Path(__file__).resolve().parent / "web" / "js"
_FRONTENDS = {
    "local_llm_server": "local_llm_server_v088.js",
    "prompt_enhancer": "prompt_enhancer_dom_v0612.js",
}
if _js_dir.is_dir():
    for _candidate in _js_dir.glob("*.js"):
        name = _candidate.name
        keep = True
        if name.startswith("local_llm_server"):
            keep = name == _FRONTENDS["local_llm_server"]
        elif name.startswith("prompt_enhancer"):
            keep = name == _FRONTENDS["prompt_enhancer"]
        if not keep:
            try:
                _candidate.unlink()
            except OSError:
                pass

from .nodes import LocalGGUFLLMAPI
from .service import SERVICE, LocalLLMServiceAPI, LocalLLMGenerate, LocalLLMSettings
from .prompt_enhancer import LocalLLMPromptEnhancer

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
    "SERVICE",
]
