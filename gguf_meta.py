import os
import re
import struct

# GGUF metadata value types
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

_SIZES = {
    UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4,
    FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8, FLOAT64: 8,
}
_FMT = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I", INT32: "<i",
    FLOAT32: "<f", BOOL: "<?", UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}

DEFAULT_WANTED = {
    "general.name",
    "general.basename",
    "general.architecture",
    "general.finetune",
    "general.size_label",
    "tokenizer.chat_template",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    # Memory-planning metadata. Architecture-specific prefixes vary, so
    # read_gguf_metadata also accepts these suffixes below.
    "general.file_type",
}

MEMORY_KEY_SUFFIXES = (
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".context_length",
)


def _read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected EOF while reading GGUF metadata")
    return b


def _u64(f):
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _string(f):
    n = _u64(f)
    if n > 128 * 1024 * 1024:
        raise ValueError(f"Unreasonable GGUF string length: {n}")
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip_value(f, typ):
    if typ in _SIZES:
        f.seek(_SIZES[typ], os.SEEK_CUR)
        return
    if typ == STRING:
        n = _u64(f)
        f.seek(n, os.SEEK_CUR)
        return
    if typ == ARRAY:
        elem_type = struct.unpack("<I", _read_exact(f, 4))[0]
        n = _u64(f)
        if elem_type in _SIZES:
            f.seek(_SIZES[elem_type] * n, os.SEEK_CUR)
        else:
            for _ in range(n):
                _skip_value(f, elem_type)
        return
    raise ValueError(f"Unsupported GGUF metadata type {typ}")


def _read_value(f, typ):
    if typ in _FMT:
        return struct.unpack(_FMT[typ], _read_exact(f, _SIZES[typ]))[0]
    if typ == STRING:
        return _string(f)
    if typ == ARRAY:
        elem_type = struct.unpack("<I", _read_exact(f, 4))[0]
        n = _u64(f)
        # Metadata arrays can be huge (token vocab). Avoid retaining them.
        if n > 4096:
            if elem_type in _SIZES:
                f.seek(_SIZES[elem_type] * n, os.SEEK_CUR)
            else:
                for _ in range(n):
                    _skip_value(f, elem_type)
            return f"<array:{n}>"
        return [_read_value(f, elem_type) for _ in range(n)]
    raise ValueError(f"Unsupported GGUF metadata type {typ}")


def read_gguf_metadata(path, wanted=None):
    wanted = set(wanted or DEFAULT_WANTED)
    out = {}
    with open(path, "rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError("Not a GGUF file")
        version = struct.unpack("<I", _read_exact(f, 4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        _tensor_count = _u64(f)
        kv_count = _u64(f)
        if kv_count > 1_000_000:
            raise ValueError("Unreasonable GGUF metadata count")
        for _ in range(kv_count):
            key = _string(f)
            typ = struct.unpack("<I", _read_exact(f, 4))[0]
            if key in wanted or key.endswith(MEMORY_KEY_SUFFIXES):
                out[key] = _read_value(f, typ)
            else:
                _skip_value(f, typ)
    return out


def detect_family(metadata, filename):
    text = " ".join(str(metadata.get(k, "")) for k in (
        "general.name", "general.basename", "general.finetune", "general.architecture"
    )) + " " + os.path.basename(filename)
    raw = text.lower()
    s = raw.replace("_", "-")

    # Do not normalize Qwen release dots into dashes for these checks: a name
    # such as Qwen3-8B is Qwen3 8B, not Qwen3.8. This distinction is important
    # because Qwen3.8 uses different recommended sampling/template behavior.
    if re.search(r"qwen3\.8(?:[^0-9]|$)", raw):
        return "qwen3.8"
    if re.search(r"qwen3\.6(?:[^0-9]|$)", raw):
        return "qwen3.6"
    if re.search(r"qwen3\.5(?:[^0-9]|$)", raw):
        return "qwen3.5"
    if "qwen3-vl" in s:
        return "qwen3-vl"
    if "qwen2.5-vl" in s or "qwen2-5-vl" in s:
        return "qwen2.5-vl"
    if "qwen3-asr" in s or "qwen3 asr" in s:
        return "qwen3-asr"
    if "qwen3" in s:
        return "qwen3"
    if "gpt-oss" in s:
        return "gpt-oss"
    if "mistral-small-3.2" in s or "mistral small 3.2" in s:
        return "mistral-small-3.2"
    if "ministral-3" in s or "ministral 3" in s:
        if "reason" in s:
            return "ministral-3-reasoning"
        return "ministral-3-instruct"
    if "gemma-4" in s or "gemma4" in s:
        return "gemma4"
    if "gemma-3" in s or "gemma3" in s:
        return "gemma3"
    if "glm-4.6v" in s or "glm4.6v" in s or "glm-4-6v" in s:
        return "glm4.6v"
    if "glm-4.1v" in s or "glm4.1v" in s or "glm-4-1v" in s:
        return "glm4.1v"
    if "lfm2.5-vl" in s or "lfm2-5-vl" in s:
        return "lfm2.5-vl"
    if "lfm2-vl" in s:
        return "lfm2-vl"
    if "granite-docling" in s or "granite docling" in s:
        return "granite-docling"
    if "deepseek-ocr" in s or "deepseek ocr" in s:
        return "deepseek-ocr"
    if "paddleocr-vl" in s or "paddleocr vl" in s:
        return "paddleocr-vl"
    if "step3-vl" in s or "step3 vl" in s:
        return "step3-vl"
    if "minicpm-v4.6" in s or "minicpm v4.6" in s:
        return "minicpm-v4.6"
    if "minicpm-v4.5" in s or "minicpm v4.5" in s:
        return "minicpm-v4.5"
    if "minicpm-v2.6" in s or "minicpm v2.6" in s:
        return "minicpm-v2.6"
    if "moondream2" in s or "moondream-2" in s:
        return "moondream2"
    if "nanollava" in s or "nano-llava" in s:
        return "nanollava"
    if "llama3-vision" in s or "llama-3-vision" in s or "llama3 vision" in s:
        return "llama3-vision"
    if "llava" in s:
        return "llava"
    if "deepseek-r1" in s or "deepseek r1" in s:
        return "deepseek-r1"
    if "phi-4" in s and "reason" in s:
        return "phi4-reasoning"
    if "nemotron-3" in s or "nemotron 3" in s:
        return "nemotron3"
    if "llama-3.2" in s or "llama 3.2" in s or "llama-3.1" in s or "llama 3.1" in s:
        return "llama3-instruct"
    return str(metadata.get("general.architecture", "generic") or "generic").lower()


def available_model_presets(metadata, filename):
    """Return only presets that are meaningful for the detected model family.

    The backend INPUT_TYPES still advertises the full superset for workflow/API
    compatibility; the frontend narrows the dropdown to these choices after
    reading the selected GGUF metadata.
    """
    family = detect_family(metadata, filename)
    s = (str(metadata.get("general.name", "")) + " " + os.path.basename(filename)).lower()
    if family == "qwen3.8":
        return ["Qwen3.8 Thinking", "Qwen3.8 Non-Thinking"]
    if family in ("qwen3.5", "qwen3.6"):
        return [
            "Qwen3.5 Thinking - General",
            "Qwen3.5 Thinking - Coding",
            "Qwen3.5 Non-Thinking - General",
            "Qwen3.5 Non-Thinking - Reasoning",
        ]
    if family in ("qwen3", "qwen3-vl"):
        return ["Qwen3 Thinking", "Qwen3 Non-Thinking"]
    if family == "gpt-oss":
        return ["GPT-OSS 20B"]
    if family == "mistral-small-3.2":
        return ["Mistral Small 3.2 24B"]
    if family == "ministral-3-instruct":
        return ["Ministral 3 Instruct"]
    if family == "ministral-3-reasoning":
        # Do not offer the wrong size-specific reasoning preset when size can be
        # resolved from the model name. If it cannot, expose both.
        if "14b" in s:
            return ["Ministral 3 14B Reasoning"]
        if "8b" in s:
            return ["Ministral 3 8B Reasoning"]
        return ["Ministral 3 8B Reasoning", "Ministral 3 14B Reasoning"]
    if family == "gemma3":
        return ["Gemma 3 Instruct"]
    if family == "deepseek-r1":
        return ["DeepSeek R1 Distill"]
    if family == "phi4-reasoning":
        return ["Phi-4 Reasoning"]
    if family == "nemotron3":
        return ["Nemotron 3 Nano - Reasoning", "Nemotron 3 Nano - Tool Use"]
    if family == "llama3-instruct":
        return ["Llama 3.1/3.2 Instruct"]
    return ["Generic Chat"]


def recommended_model_preset(metadata, filename):
    family = detect_family(metadata, filename)
    s = (str(metadata.get("general.name", "")) + " " + os.path.basename(filename)).lower()
    if family == "qwen3.8":
        return "Qwen3.8 Thinking"
    if family in ("qwen3.5", "qwen3.6"):
        return "Qwen3.5 Thinking - General"
    if family == "qwen3":
        return "Qwen3 Thinking"
    if family == "gpt-oss":
        return "GPT-OSS 20B"
    if family == "mistral-small-3.2":
        return "Mistral Small 3.2 24B"
    if family == "ministral-3-instruct":
        return "Ministral 3 Instruct"
    if family == "ministral-3-reasoning":
        return "Ministral 3 14B Reasoning" if "14b" in s else "Ministral 3 8B Reasoning"
    if family == "gemma3":
        return "Gemma 3 Instruct"
    if family == "deepseek-r1":
        return "DeepSeek R1 Distill"
    if family == "phi4-reasoning":
        return "Phi-4 Reasoning"
    if family == "nemotron3":
        return "Nemotron 3 Nano - Reasoning"
    if family == "llama3-instruct":
        return "Llama 3.1/3.2 Instruct"
    return "Generic Chat"
