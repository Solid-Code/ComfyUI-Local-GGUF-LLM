from copy import deepcopy

# Neutral sampler values intentionally disable samplers/penalties that a model's
# official guidance does not specify. Model presets override only the settings
# actually recommended for that family/mode.
MODEL_NEUTRAL = {
    "thinking_mode": "Auto",
    "reasoning_effort": "Auto",
    "preserve_thinking": False,
    "chat_format": "Auto (GGUF embedded)",
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "tfs_z": 1.0,
    "mirostat_mode": "Off",
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
}


def _m(**kwargs):
    x = deepcopy(MODEL_NEUTRAL)
    x.update(kwargs)
    return x


# The values below are based on current upstream model cards/generation configs.
# Unspecified samplers are left at neutral values instead of inventing settings.
MODEL_PRESETS = {
    "Generic Chat": _m(temperature=0.7, top_p=0.9, top_k=40),

    # Qwen3.8-27B official recommendations.
    "Qwen3.8 Thinking": _m(
        thinking_mode="Enabled",
        reasoning_effort="XHigh",
        preserve_thinking=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
    ),
    "Qwen3.8 Non-Thinking": _m(
        thinking_mode="Disabled",
        reasoning_effort="Auto",
        preserve_thinking=True,
        temperature=0.7,
        top_p=0.80,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    ),

    # Qwen3.5 official recommendations.
    "Qwen3.5 Thinking - General": _m(
        thinking_mode="Enabled",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    ),
    "Qwen3.5 Thinking - Coding": _m(
        thinking_mode="Enabled",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
    ),
    "Qwen3.5 Non-Thinking - General": _m(
        thinking_mode="Disabled",
        temperature=0.7,
        top_p=0.80,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    ),
    "Qwen3.5 Non-Thinking - Reasoning": _m(
        thinking_mode="Disabled",
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        min_p=0.0,
        presence_penalty=2.0,
        repeat_penalty=1.0,
    ),

    # Qwen3 official recommendations. For GGUF quantizations, presence 1.5 is
    # useful against repetition; Qwen explicitly recommends it for quantized models.
    "Qwen3 Thinking": _m(
        thinking_mode="Enabled",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    ),
    "Qwen3 Non-Thinking": _m(
        thinking_mode="Disabled",
        temperature=0.7,
        top_p=0.80,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repeat_penalty=1.0,
    ),

    # GPT-OSS uses Harmony from the embedded GGUF template. llama.cpp guidance
    # recommends neutral penalties/samplers with temperature/top-p at 1.0.
    "GPT-OSS 20B": _m(
        reasoning_effort="Medium",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repeat_penalty=1.0,
    ),

    # Mistral only specifies temperature for these releases. Other samplers remain neutral.
    "Mistral Small 3.2 24B": _m(thinking_mode="Disabled", temperature=0.15),
    "Ministral 3 Instruct": _m(thinking_mode="Disabled", temperature=0.05),
    "Ministral 3 8B Reasoning": _m(temperature=0.7),
    "Ministral 3 14B Reasoning": _m(temperature=1.0),

    # Gemma 3 generation config: top-k 64 / top-p .95; AI Studio guidance uses temp 1.0.
    "Gemma 3 Instruct": _m(
        thinking_mode="Disabled",
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    ),

    # Meta generation config does not specify top-k; keep it disabled.
    "Llama 3.1/3.2 Instruct": _m(
        thinking_mode="Disabled",
        temperature=0.6,
        top_p=0.9,
    ),

    # DeepSeek R1 distill generation config specifies temperature/top-p only.
    "DeepSeek R1 Distill": _m(
        temperature=0.6,
        top_p=0.95,
    ),

    # Microsoft requires ChatML plus these exact sampling values.
    "Phi-4 Reasoning": _m(
        chat_format="chatml",
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    ),

    # NVIDIA Nemotron 3 Nano generation config / guidance.
    "Nemotron 3 Nano - Reasoning": _m(
        temperature=1.0,
        top_p=1.0,
    ),
    "Nemotron 3 Nano - Tool Use": _m(
        temperature=0.6,
        top_p=0.95,
    ),
}

# Compatibility alias for workflows saved with v0.2.x. Keep it out of Auto
# detection, but allow the saved combo value to resolve without silently falling
# back to user-entered numbers.
MODEL_PRESETS["Qwen3.5 Non-Thinking"] = deepcopy(MODEL_PRESETS["Qwen3.5 Non-Thinking - General"])

# UI/backend capabilities are separate from sampling values. This prevents us
# from showing controls that a model family does not actually support.
MODEL_PRESET_META = {
    "Generic Chat": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Qwen3.8 Thinking": {"thinking_switch": True, "reasoning_efforts": ["Low", "Medium", "XHigh"], "preserve_thinking": True, "reasoning_output": True},
    "Qwen3.8 Non-Thinking": {"thinking_switch": True, "reasoning_efforts": ["Low", "Medium", "XHigh"], "preserve_thinking": True, "reasoning_output": True},
    "Qwen3.5 Thinking - General": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Qwen3.5 Thinking - Coding": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Qwen3.5 Non-Thinking - General": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Qwen3.5 Non-Thinking - Reasoning": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Qwen3 Thinking": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Qwen3 Non-Thinking": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "GPT-OSS 20B": {"thinking_switch": False, "reasoning_efforts": ["Low", "Medium", "High"], "preserve_thinking": False, "reasoning_output": True},
    "Mistral Small 3.2 24B": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": False},
    "Ministral 3 Instruct": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": False},
    "Ministral 3 8B Reasoning": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Ministral 3 14B Reasoning": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Gemma 3 Instruct": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": False},
    "Llama 3.1/3.2 Instruct": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": False},
    "DeepSeek R1 Distill": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Phi-4 Reasoning": {"thinking_switch": False, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Nemotron 3 Nano - Reasoning": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
    "Nemotron 3 Nano - Tool Use": {"thinking_switch": True, "reasoning_efforts": [], "preserve_thinking": False, "reasoning_output": True},
}
MODEL_PRESET_META["Qwen3.5 Non-Thinking"] = deepcopy(MODEL_PRESET_META["Qwen3.5 Non-Thinking - General"])

# Central capability metadata.  Generation-control flags remain at the top level
# for backward compatibility with the existing frontend; modality/handler fields
# are the authoritative source for multimodal UI and validation.
#
# `vision=None` means "unknown/try generic MTMD" rather than "unsupported".  This
# preserves compatibility with newly released VLM GGUFs before they are added to
# the registry, while known text-only families can hide/reject vision controls.
_GENERIC_CAPABILITY = {
    **MODEL_PRESET_META["Generic Chat"],
    "text": True,
    "vision": None,
    "audio": False,
    "embeddings": None,
    "mtp": None,
    "mmproj_required_for_vision": None,
    "preferred_chat_handlers": [],
    "capability_confidence": "unknown",
    "implemented": {
        "text": True,
        "vision": True,
        "audio": False,
        "embeddings": False,
        "mtp": False,
    },
}


def _cap(preset, *, vision=False, audio=False, handlers=(), confidence="known", mmproj_required=None):
    x = deepcopy(MODEL_PRESET_META[preset])
    x.update({
        "text": True,
        "vision": vision,
        "audio": audio,
        "embeddings": None,
        "mtp": None,
        "mmproj_required_for_vision": (bool(vision) if mmproj_required is None else mmproj_required),
        "preferred_chat_handlers": list(handlers),
        "capability_confidence": confidence,
        "implemented": {
            "text": True,
            "vision": True,
            "audio": False,
            "embeddings": False,
            "mtp": False,
        },
    })
    return x


FAMILY_CAPABILITIES = {
    # Qwen unified/VL families.  Qwen3.5/3.6/3.8 use the newer Qwen35 handler
    # when present, with MTMD as a generic fallback.
    "qwen3.8": _cap("Qwen3.8 Thinking", vision=True, handlers=("Qwen35ChatHandler", "Qwen3VLChatHandler")),
    "qwen3.6": _cap("Qwen3.5 Thinking - General", vision=True, handlers=("Qwen35ChatHandler", "Qwen3VLChatHandler")),
    "qwen3.5": _cap("Qwen3.5 Thinking - General", vision=True, handlers=("Qwen35ChatHandler", "Qwen3VLChatHandler")),
    "qwen3": _cap("Qwen3 Thinking", vision=False),
    "qwen3-vl": _cap("Qwen3 Thinking", vision=True, handlers=("Qwen3VLChatHandler",)),
    "qwen2.5-vl": _cap("Generic Chat", vision=True, handlers=("Qwen25VLChatHandler",)),
    "qwen3-asr": _cap("Generic Chat", vision=False, audio=True, handlers=("Qwen3ASRChatHandler",)),

    # Google / Meta / Mistral / reasoning text families.
    "gemma4": _cap("Generic Chat", vision=True, handlers=("Gemma4ChatHandler",)),
    "gemma3": _cap("Gemma 3 Instruct", vision=True, handlers=("Gemma3ChatHandler",)),
    "gpt-oss": _cap("GPT-OSS 20B", vision=False),
    "mistral-small-3.2": _cap("Mistral Small 3.2 24B", vision=True),
    "ministral-3-instruct": _cap("Ministral 3 Instruct", vision=False),
    "ministral-3-reasoning": _cap("Ministral 3 8B Reasoning", vision=False),
    "deepseek-r1": _cap("DeepSeek R1 Distill", vision=False),
    "phi4-reasoning": _cap("Phi-4 Reasoning", vision=False),
    "nemotron3": _cap("Nemotron 3 Nano - Reasoning", vision=False),
    "llama3-instruct": _cap("Llama 3.1/3.2 Instruct", vision=False),

    # Additional VLM families mirrored from the actively maintained
    # ComfyUI-llama-cpp_vlm handler coverage.
    "glm4.6v": _cap("Generic Chat", vision=True, handlers=("GLM46VChatHandler",)),
    "glm4.1v": _cap("Generic Chat", vision=True, handlers=("GLM41VChatHandler",)),
    "lfm2-vl": _cap("Generic Chat", vision=True, handlers=("LFM2VLChatHandler",)),
    "lfm2.5-vl": _cap("Generic Chat", vision=True, handlers=("LFM25VLChatHandler",)),
    "granite-docling": _cap("Generic Chat", vision=True, handlers=("GraniteDoclingChatHandler",)),
    "deepseek-ocr": _cap("Generic Chat", vision=True, handlers=("MTMDChatHandler",)),
    "paddleocr-vl": _cap("Generic Chat", vision=True, handlers=("PaddleOCRChatHandler",)),
    "step3-vl": _cap("Generic Chat", vision=True, handlers=("Step3VLChatHandler",)),
    "minicpm-v2.6": _cap("Generic Chat", vision=True, handlers=("MiniCPMv26ChatHandler",)),
    "minicpm-v4.5": _cap("Generic Chat", vision=True, handlers=("MiniCPMv45ChatHandler",)),
    "minicpm-v4.6": _cap("Generic Chat", vision=True, handlers=("MiniCPMV46ChatHandler",)),
    "moondream2": _cap("Generic Chat", vision=True, handlers=("MoondreamChatHandler",)),
    "nanollava": _cap("Generic Chat", vision=True, handlers=("NanoLlavaChatHandler",)),
    "llama3-vision": _cap("Generic Chat", vision=True, handlers=("Llama3VisionAlphaChatHandler",)),
    "llava": _cap("Generic Chat", vision=True, handlers=("Llava16ChatHandler", "Llava15ChatHandler")),
}


MEMORY_COMMON = {
    "context_size": 32768,
    "kv_cache_k": "q8_0",
    "kv_cache_v": "q5_0",
    "kv_cache_location": "GPU",
    "gpu_layers": -1,
    "flash_attention": True,
    "prompt_batch_size": 2048,
    "memory_batch_size": 512,
    "use_mmap": True,
    "use_mlock": False,
}

MEMORY_PRESETS = {
    "Balanced": deepcopy(MEMORY_COMMON),
    "Maximum Quality": {**MEMORY_COMMON, "context_size": 16384, "kv_cache_k": "f16", "kv_cache_v": "f16"},
    "High Quality KV": {**MEMORY_COMMON, "kv_cache_k": "q8_0", "kv_cache_v": "q8_0"},
    "Low KV Memory": {**MEMORY_COMMON, "kv_cache_k": "q8_0", "kv_cache_v": "q4_0"},
    "Minimum KV Memory": {**MEMORY_COMMON, "kv_cache_k": "q4_0", "kv_cache_v": "q4_0"},
    "CPU KV Cache": {**MEMORY_COMMON, "kv_cache_k": "q8_0", "kv_cache_v": "q8_0", "kv_cache_location": "CPU"},
    "CPU / Low VRAM": {
        **MEMORY_COMMON,
        "context_size": 16384,
        "kv_cache_k": "q8_0",
        "kv_cache_v": "q8_0",
        "kv_cache_location": "CPU",
        "gpu_layers": 0,
        "flash_attention": False,
        "prompt_batch_size": 512,
        "memory_batch_size": 256,
    },
}


def capabilities_for_family(family):
    return deepcopy(FAMILY_CAPABILITIES.get(family, _GENERIC_CAPABILITY))


def public_presets():
    return {
        "model": deepcopy(MODEL_PRESETS),
        "model_meta": deepcopy(MODEL_PRESET_META),
        "memory": deepcopy(MEMORY_PRESETS),
    }
