from .nodes import LocalGGUFLLMAPI
from .service import SERVICE, LocalLLMServiceAPI, GetLocalLLMService, LocalLLMServiceGenerate

# Only expose the persistent-service workflow nodes.  LocalGGUFLLM remains an
# internal engine used by the service and is intentionally not registered as a
# standalone ComfyUI node.
NODE_CLASS_MAPPINGS = {
    "GetLocalLLMService": GetLocalLLMService,
    "LocalLLMServiceGenerate": LocalLLMServiceGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetLocalLLMService": "Get Local LLM Service",
    "LocalLLMServiceGenerate": "Local LLM Service Generate",
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
