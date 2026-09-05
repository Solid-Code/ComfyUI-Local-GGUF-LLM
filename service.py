"""Global persistent Local LLM service for ComfyUI.

The service is intentionally process-global rather than workflow-global.  It keeps
one llama.cpp model/context alive and exposes it to lightweight ComfyUI nodes and
an OpenAI-compatible HTTP endpoint.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import secrets
import statistics
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import folder_paths

from .nodes import (
    LocalGGUFLLM,
    _AUTO_VISION,
    _NONE,
    _cleanup_llm,
    _suspend_llm_native,
    _native_operation_snapshot,
    _native_operation_owned_by_current_thread,
    _sync_cuda_device,
    _find_matching_mmproj,
    _validate_mmproj_pair,
    _gpu_choices,
    _metadata_for,
    _model_lists,
    _estimate_native_vram_components,
    _speculative_runtime_support,
    _resolve_speculative_mode,
    _mtp_layer_count,
    _full_path,
    _gpu_index,
    _reload_vram_target_bytes,
    _MODEL_CACHE,
    _MODEL_LOCK,
    LocalLLMInterrupted,
)
from .gguf_meta import detect_family, recommended_model_preset, available_model_presets
from .presets import MEMORY_PRESETS, MODEL_PRESETS, capabilities_for_family, public_presets
from .version import PACKAGE_VERSION, BRIDGE_API_VERSION, VRAM_POLICY_VERSION, VRAM_COORDINATION_MODE
from .vram_coordination import GPUMemoryLeaseManager

log = logging.getLogger(__name__)

STATE_STOPPED = "stopped"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_PROCESSING = "processing"
STATE_GENERATING = "generating"
STATE_WAITING_COMFY = "waiting_comfy"
STATE_RELOADING = "reloading"
STATE_STOPPING = "stopping"
STATE_TUNING = "tuning"
STATE_ERROR = "error"

LOAD_FIELDS = {
    "model", "vision_model", "memory_preset", "context_size", "kv_cache_k", "kv_cache_v",
    "kv_cache_location", "gpu_layers", "flash_attention", "prompt_batch_size",
    "memory_batch_size", "use_mmap", "use_mlock", "speculative_mode",
    "ngram_pred_tokens", "ngram_size", "ngram_mode", "ngram_min_hits",
    "ngram_max_entries_per_key", "ngram_sync_check_tokens", "mtp_draft_tokens", "mtp_p_min",
    "split_mode", "main_gpu", "tensor_split",
    "threads", "threads_batch", "op_offload", "swa_full", "rope_freq_base", "rope_freq_scale",
    "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx",
    "verbose", "vram_policy",
}

SERVER_ONLY_FIELDS = {
    "startup_mode", "external_api_enabled", "api_key", "allow_buffered_streaming",
    "log_prompt_content", "log_response_content", "show_status_indicator",
}


# Common context sizes exposed by the global server UI.  The list is filtered
# against the selected GGUF's native training context when that metadata is
# available.  This avoids accidental odd values (e.g. 31000) and prevents the
# server from silently requesting a context larger than the model advertises.
CONTEXT_SIZE_STEPS = (
    1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768,
    49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288,
    786432, 1048576,
)


# Reusable presets used by the lightweight service client node.
#
# Presets are intentionally split by concern so sampler settings, system prompts,
# and user prompts can be mixed independently.  The files live beside the GGUF
# models rather than inside workflows:
#
#   models/llm/local_LLM_presets/
#       sampler/         -> JSON generation settings
#       system_prompts/  -> plain UTF-8 text
#       prompts/         -> plain UTF-8 text
#
# The standard ComfyUI seed is deliberately NOT part of sampler presets.
SAMPLER_PRESET_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "max_tokens",
)
SAMPLER_PRESET_SCHEMA = "local_llm_sampler_preset"
SAMPLER_PRESET_VERSION = 1
PRESET_ROOT_DIR = Path(folder_paths.models_dir) / "LLM" / "local_LLM_presets"
SAMPLER_PRESET_DIR = PRESET_ROOT_DIR / "sampler"
SYSTEM_PROMPT_PRESET_DIR = PRESET_ROOT_DIR / "system_prompts"
PROMPT_PRESET_DIR = PRESET_ROOT_DIR / "prompts"
MEMORY_PRESET_DIR = PRESET_ROOT_DIR / "memory"  # legacy v0.17 and earlier
SETTINGS_PRESET_DIR = PRESET_ROOT_DIR / "settings"

MEMORY_PRESET_FIELDS = (
    "context_size", "kv_cache_k", "kv_cache_v", "kv_cache_location", "gpu_layers",
    "flash_attention", "prompt_batch_size", "memory_batch_size", "use_mmap", "use_mlock",
    "main_gpu", "split_mode", "tensor_split", "speculative_mode",
    "ngram_pred_tokens", "ngram_size", "ngram_mode", "ngram_min_hits",
    "ngram_max_entries_per_key", "ngram_sync_check_tokens", "mtp_draft_tokens", "mtp_p_min",
)
MEMORY_PRESET_SCHEMA = "local_llm_memory_preset"
MEMORY_PRESET_VERSION = 1


# Complete presets intentionally cover the LLM runtime itself, not server/admin
# state such as API keys, startup behavior, logging, or UI preferences.  They
# replace the old memory-only preset concept in the user interface.
COMPLETE_SETTINGS_PRESET_FIELDS = (
    "model", "vision_model", "model_preset", "thinking_mode", "reasoning_effort",
    "preserve_thinking",
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "max_tokens",
    "vision_max_images", "vision_max_frames", "vision_max_edge",
    "context_size", "kv_cache_k", "kv_cache_v", "kv_cache_location", "gpu_layers",
    "flash_attention", "prompt_batch_size", "memory_batch_size", "use_mmap", "use_mlock",
    "prompt_cache_mode", "speculative_mode", "ngram_pred_tokens", "ngram_size",
    "ngram_mode", "ngram_min_hits", "ngram_max_entries_per_key",
    "ngram_sync_check_tokens", "mtp_draft_tokens", "mtp_p_min",
    "split_mode", "main_gpu", "tensor_split", "vram_policy",
)
COMPLETE_SETTINGS_PRESET_SCHEMA = "local_llm_complete_settings_preset"
COMPLETE_SETTINGS_PRESET_VERSION = 1


_RESERVED_PRESET_NAMES = frozenset({"default", "custom", "server default"})
_BUILTIN_MEMORY_PRESET_NAMES = frozenset(str(name).lower() for name in MEMORY_PRESETS)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a small UTF-8 settings/preset file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _ensure_preset_dirs():
    for path in (PRESET_ROOT_DIR, SAMPLER_PRESET_DIR, SYSTEM_PROMPT_PRESET_DIR, PROMPT_PRESET_DIR, MEMORY_PRESET_DIR, SETTINGS_PRESET_DIR):
        path.mkdir(parents=True, exist_ok=True)


_ensure_preset_dirs()


def _safe_preset_name(name: str) -> str:
    name = str(name or "").strip()
    # Keep filenames portable between WSL/Linux and Windows-backed model trees.
    name = ''.join('_' if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in name)
    name = name.rstrip('. ').strip()
    if not name:
        raise ValueError("Preset name cannot be empty")
    if name.lower() in _RESERVED_PRESET_NAMES:
        raise ValueError(f"'{name}' is reserved; choose another preset name")
    if len(name) > 96:
        name = name[:96].rstrip()
    return name


def _sampler_preset_path(name: str) -> Path:
    return SAMPLER_PRESET_DIR / (_safe_preset_name(name) + ".json")


def _validate_sampler_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError("sampler preset settings must be an object")
    out = {}
    for key in SAMPLER_PRESET_FIELDS:
        if key not in settings:
            continue
        value = settings[key]
        if key in {"top_k", "max_tokens"}:
            value = int(value)
        else:
            value = float(value)
        out[key] = value
    if not out:
        raise ValueError("sampler preset contains no generation settings")
    return out


def _load_sampler_preset_file(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        settings = data.get("settings", data)
        settings = _validate_sampler_settings(settings)
        name = str(data.get("name") or path.stem).strip() or path.stem
        return {
            "name": name,
            "settings": settings,
            "path": str(path),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
    except Exception as e:
        log.warning("[Local LLM Server] Ignoring invalid sampler preset %s: %s", path, e)
        return None


def sampler_presets() -> dict[str, dict[str, Any]]:
    _ensure_preset_dirs()
    found = {}
    for path in sorted(SAMPLER_PRESET_DIR.glob("*.json"), key=lambda x: x.name.lower()):
        item = _load_sampler_preset_file(path)
        if item is not None:
            found[item["name"]] = item
    return found


def sampler_preset_names() -> list[str]:
    return sorted(sampler_presets(), key=str.lower)


def load_sampler_preset(name: str) -> dict[str, Any] | None:
    return copy.deepcopy((sampler_presets().get(str(name)) or {}).get("settings"))


def save_sampler_preset(name: str, settings: dict[str, Any]) -> dict[str, Any]:
    clean_name = _safe_preset_name(name)
    clean = _validate_sampler_settings(settings)
    path = _sampler_preset_path(clean_name)
    payload = {
        "schema": SAMPLER_PRESET_SCHEMA,
        "schema_version": SAMPLER_PRESET_VERSION,
        "name": clean_name,
        "settings": clean,
    }
    _atomic_write_json(path, payload)
    return {"name": clean_name, "settings": copy.deepcopy(clean), "path": str(path)}


def delete_sampler_preset(name: str) -> dict[str, Any]:
    """Delete one user sampler preset without ever accepting an arbitrary path."""
    requested = str(name or "").strip()
    if requested.lower() in _RESERVED_PRESET_NAMES:
        raise ValueError(f"'{requested or 'Default'}' is a built-in selector and cannot be deleted")
    item = sampler_presets().get(requested)
    if item is None:
        raise ValueError(f"Sampler preset '{requested}' was not found")
    path = Path(str(item.get("path") or ""))
    # Resolve both sides and require the file to live directly in our user preset directory.
    if path.resolve().parent != SAMPLER_PRESET_DIR.resolve():
        raise ValueError("Refusing to delete a sampler preset outside the Local LLM preset directory")
    path.unlink()
    return {"name": requested, "path": str(path)}


def _text_preset_dir(kind: str) -> Path:
    if kind == "system_prompts":
        return SYSTEM_PROMPT_PRESET_DIR
    if kind == "prompts":
        return PROMPT_PRESET_DIR
    raise ValueError(f"Unknown text preset kind: {kind}")


def text_presets(kind: str) -> dict[str, dict[str, Any]]:
    directory = _text_preset_dir(kind)
    directory.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in sorted(directory.glob("*.txt"), key=lambda x: x.name.lower()):
        try:
            name = path.stem
            found[name] = {
                "name": name,
                "text": path.read_text(encoding="utf-8"),
                "path": str(path),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
        except Exception as e:
            log.warning("[Local LLM Server] Ignoring unreadable %s preset %s: %s", kind, path, e)
    return found


def text_preset_names(kind: str) -> list[str]:
    return sorted(text_presets(kind), key=str.lower)


def load_text_preset(kind: str, name: str) -> str | None:
    item = text_presets(kind).get(str(name))
    return None if item is None else str(item.get("text", ""))


def save_text_preset(kind: str, name: str, text: str) -> dict[str, Any]:
    clean_name = _safe_preset_name(name)
    directory = _text_preset_dir(kind)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (clean_name + ".txt")
    value = str(text if text is not None else "")
    _atomic_write_text(path, value)
    return {"name": clean_name, "text": value, "path": str(path)}


def delete_text_preset(kind: str, name: str) -> dict[str, Any]:
    """Delete one user text preset without accepting an arbitrary path."""
    requested = str(name or "").strip()
    if requested.lower() in _RESERVED_PRESET_NAMES:
        raise ValueError(f"'{requested or 'Custom'}' is a built-in selector and cannot be deleted")
    directory = _text_preset_dir(kind)
    item = text_presets(kind).get(requested)
    if item is None:
        label = "System prompt" if kind == "system_prompts" else "Prompt"
        raise ValueError(f"{label} preset '{requested}' was not found")
    path = Path(str(item.get("path") or ""))
    if path.resolve().parent != directory.resolve():
        raise ValueError("Refusing to delete a text preset outside the Local LLM preset directory")
    path.unlink()
    return {"name": requested, "path": str(path)}


def _validate_memory_preset_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError("memory preset settings must be an object")
    out = {}
    for key in MEMORY_PRESET_FIELDS:
        if key not in settings:
            continue
        value = settings[key]
        if key in {"context_size", "gpu_layers", "prompt_batch_size", "memory_batch_size",
                   "ngram_pred_tokens", "ngram_size", "ngram_min_hits",
                   "ngram_max_entries_per_key", "ngram_sync_check_tokens", "mtp_draft_tokens"}:
            value = int(value)
        elif key in {"mtp_p_min"}:
            value = float(value)
        elif key in {"flash_attention", "use_mmap", "use_mlock"}:
            value = bool(value)
        else:
            value = str(value)
        out[key] = value
    if not out:
        raise ValueError("memory preset contains no supported settings")
    return out


def memory_presets() -> dict[str, dict[str, Any]]:
    """User memory/performance presets stored beside the other Local LLM presets."""
    _ensure_preset_dirs()
    found = {}
    for path in sorted(MEMORY_PRESET_DIR.glob("*.json"), key=lambda x: x.name.lower()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            settings = _validate_memory_preset_settings(data.get("settings", data))
            name = str(data.get("name") or path.stem).strip() or path.stem
            found[name] = {
                "name": name, "settings": settings, "path": str(path),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
        except Exception as e:
            log.warning("[Local LLM Server] Ignoring invalid memory preset %s: %s", path, e)
    return found


def save_memory_preset(name: str, settings: dict[str, Any]) -> dict[str, Any]:
    clean_name = _safe_preset_name(name)
    if clean_name.lower() in _BUILTIN_MEMORY_PRESET_NAMES:
        raise ValueError(f"'{clean_name}' is a built-in memory preset and cannot be overwritten from the UI")
    clean = _validate_memory_preset_settings(settings)
    path = MEMORY_PRESET_DIR / (clean_name + ".json")
    payload = {
        "schema": MEMORY_PRESET_SCHEMA, "schema_version": MEMORY_PRESET_VERSION,
        "name": clean_name, "settings": clean,
    }
    _atomic_write_json(path, payload)
    return {"name": clean_name, "settings": copy.deepcopy(clean), "path": str(path)}


def _validate_complete_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError("complete settings preset must be an object")
    int_fields = {
        "top_k", "max_tokens", "vision_max_images", "vision_max_frames", "vision_max_edge",
        "context_size", "gpu_layers", "prompt_batch_size", "memory_batch_size",
        "ngram_pred_tokens", "ngram_size", "ngram_min_hits",
        "ngram_max_entries_per_key", "ngram_sync_check_tokens", "mtp_draft_tokens",
    }
    float_fields = {
        "temperature", "top_p", "min_p", "repeat_penalty",
        "presence_penalty", "frequency_penalty", "mtp_p_min",
    }
    bool_fields = {"preserve_thinking", "flash_attention", "use_mmap", "use_mlock"}
    out = {}
    for key in COMPLETE_SETTINGS_PRESET_FIELDS:
        if key not in settings:
            continue
        value = settings[key]
        try:
            if key in bool_fields:
                value = bool(value)
            elif key in int_fields:
                value = int(value)
            elif key in float_fields:
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for complete preset field '{key}': {value!r}")
        out[key] = value
    if not out:
        raise ValueError("complete settings preset contains no supported settings")
    return out


def complete_settings_presets() -> dict[str, dict[str, Any]]:
    _ensure_preset_dirs()
    found = {}
    for path in sorted(SETTINGS_PRESET_DIR.glob("*.json"), key=lambda x: x.name.lower()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            clean = _validate_complete_settings(data.get("settings", data))
            name = str(data.get("name") or path.stem).strip() or path.stem
            found[name] = {
                "name": name, "settings": clean, "path": str(path),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
        except Exception as e:
            log.warning("[Local LLM Server] Ignoring invalid complete settings preset %s: %s", path, e)
    return found


def complete_settings_preset_names() -> list[str]:
    return sorted(complete_settings_presets(), key=str.lower)


def load_complete_settings_preset(name: str) -> dict[str, Any] | None:
    item = complete_settings_presets().get(str(name))
    return None if item is None else copy.deepcopy(item.get("settings") or {})


def save_complete_settings_preset(name: str, settings: dict[str, Any]) -> dict[str, Any]:
    clean_name = _safe_preset_name(name)
    clean = _validate_complete_settings(settings)
    path = SETTINGS_PRESET_DIR / (clean_name + ".json")
    payload = {
        "schema": COMPLETE_SETTINGS_PRESET_SCHEMA,
        "schema_version": COMPLETE_SETTINGS_PRESET_VERSION,
        "name": clean_name,
        "settings": clean,
    }
    _atomic_write_json(path, payload)
    return {"name": clean_name, "settings": copy.deepcopy(clean), "path": str(path)}


def delete_complete_settings_preset(name: str) -> dict[str, Any]:
    requested = str(name or "").strip()
    if requested.lower() in _RESERVED_PRESET_NAMES:
        raise ValueError(f"'{requested or 'Custom'}' is reserved and cannot be deleted")
    item = complete_settings_presets().get(requested)
    if item is None:
        raise ValueError(f"Complete settings preset '{requested}' was not found")
    path = Path(str(item.get("path") or ""))
    if path.resolve().parent != SETTINGS_PRESET_DIR.resolve():
        raise ValueError("Refusing to delete a preset outside the Local LLM settings preset directory")
    path.unlink()
    return {"name": requested, "path": str(path)}


# Backward-compatible aliases for v0.8 internal names.  New code and files use
# the sampler-specific terminology above.
REQUEST_PRESET_FIELDS = SAMPLER_PRESET_FIELDS
REQUEST_PRESET_DIR = SAMPLER_PRESET_DIR
request_presets = sampler_presets
request_preset_names = sampler_preset_names
load_request_preset = load_sampler_preset
save_request_preset = save_sampler_preset


def _native_context_length(metadata: dict[str, Any]) -> int | None:
    if not metadata:
        return None
    preferred = []
    fallback = []
    for key, value in metadata.items():
        if not str(key).endswith(".context_length"):
            continue
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        # Prefer the architecture's main context_length over rope/original
        # context metadata when both happen to be present.
        if "rope" in str(key).lower() or "original" in str(key).lower():
            fallback.append(n)
        else:
            preferred.append(n)
    vals = preferred or fallback
    return max(vals) if vals else None


def _context_options_for_model(model_name: str | None):
    md = {}
    if model_name and model_name != "No GGUF models found":
        try:
            md = _metadata_for(model_name)
        except Exception:
            md = {}
    native = _native_context_length(md)
    if native:
        options = [n for n in CONTEXT_SIZE_STEPS if n <= native]
        if native not in options:
            options.append(native)
        options = sorted(set(options))
        if not options:
            options = [native]
    else:
        # Unknown/community GGUF: still constrain the control to standard sizes.
        options = list(CONTEXT_SIZE_STEPS)
    return options, native


def _normalize_context_size(value, model_name: str | None) -> int:
    options, _native = _context_options_for_model(model_name)
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 32768
    if requested in options:
        return requested
    lower = [n for n in options if n <= requested]
    if lower:
        return max(lower)
    return min(options)


def _spec_default(spec):
    typ = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in opts:
        return copy.deepcopy(opts["default"])
    if isinstance(typ, (list, tuple)):
        return copy.deepcopy(typ[0]) if typ else None
    if typ == "BOOLEAN":
        return False
    if typ == "INT":
        return 0
    if typ == "FLOAT":
        return 0.0
    if typ == "STRING":
        return ""
    return None


def _node_defaults() -> dict[str, Any]:
    schema = LocalGGUFLLM.INPUT_TYPES()["required"]
    return {name: _spec_default(spec) for name, spec in schema.items()}


def _model_capabilities_with_runtime(metadata, family):
    caps = copy.deepcopy(capabilities_for_family(family))
    runtime = _speculative_runtime_support()
    mtp_layers = _mtp_layer_count(metadata or {})
    caps["mtp"] = bool(mtp_layers > 0)
    caps["mtp_layers"] = int(mtp_layers)
    implemented = copy.deepcopy(caps.get("implemented") or {})
    implemented["mtp"] = bool(mtp_layers > 0 and runtime.get("mtp"))
    implemented["ngram_speculative"] = bool(runtime.get("ngram"))
    caps["implemented"] = implemented
    return caps, runtime


def _resolve_service_speculative(cfg, metadata, runtime=None):
    """Resolve UI/estimator speculative mode with current native-MTP constraints."""
    support = runtime or _speculative_runtime_support()
    spec = _resolve_speculative_mode(cfg.get("speculative_mode", "Off"), metadata, support)
    requested = str(spec.get("requested") or "Off")
    try:
        gpu_layers = int(cfg.get("gpu_layers", -1))
    except Exception:
        gpu_layers = -1
    if spec.get("effective") == "MTP" and gpu_layers >= 0:
        if requested == "Auto" and support.get("ngram"):
            spec["effective"] = "N-gram"
            spec["reason"] = "Native MTP requires full GPU offload (GPU layers = -1); Auto fell back to N-gram."
        else:
            spec["effective"] = "Off"
            spec["reason"] = "Native MTP requires GPU layers = -1 (full GPU offload) with the supported bridge."
    try:
        if spec.get("effective") == "MTP" and int(cfg.get("context_size") or 0) < int(cfg.get("mtp_draft_tokens") or 2) + 1:
            if requested == "Auto" and support.get("ngram"):
                spec["effective"] = "N-gram"
                spec["reason"] = "Context size is too small for the configured MTP draft block; Auto fell back to N-gram."
            else:
                spec["effective"] = "Off"
                spec["reason"] = "Context size must be at least MTP draft tokens + 1."
    except Exception:
        pass
    return spec


def _default_config() -> dict[str, Any]:
    d = _node_defaults()
    # The global service defaults to a hot, yieldable native context: while
    # resident it behaves like the persistent server, but ComfyUI can evict it
    # automatically through its normal VRAM pressure path.
    d["model_retention"] = "ComfyUI Managed"
    # The service now stores concrete memory values; memory-only presets are a
    # legacy workflow compatibility feature, not part of global configuration.
    d["memory_preset"] = "Custom"
    d["system_prompt"] = "You are a helpful assistant."
    d["prompt"] = ""
    d.update({
        "startup_mode": "On Demand",
        "external_api_enabled": False,
        "api_key": "",
        # OpenAI-compatible live SSE streaming. Keep the legacy config key for
        # backward compatibility with existing saved server settings/UI.
        "allow_buffered_streaming": True,
        # Content logging is privacy-sensitive and therefore opt-in.
        "log_prompt_content": False,
        "log_response_content": False,
        # Small draggable ComfyUI-native status box; frontend-only and safe to
        # toggle without reloading the GGUF.
        "show_status_indicator": True,
        "vram_policy": "Auto Yield to ComfyUI",
    })
    return d


def _config_path() -> Path:
    try:
        root = Path(folder_paths.get_user_directory())
    except Exception:
        root = Path(getattr(folder_paths, "base_path", os.getcwd())) / "user" / "default"
    root.mkdir(parents=True, exist_ok=True)
    return root / "local_llm_server.json"


def _load_config_file() -> dict[str, Any]:
    config = _default_config()
    path = _config_path()
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update(saved)
        except Exception as e:
            log.warning("[Local LLM Server] Could not read %s: %s", path, e)
    # Migrate old/invalid model names gracefully.
    models, vision = _model_lists()
    if config.get("model") not in models and models and models[0] != "No GGUF models found":
        config["model"] = models[0]
    if config.get("vision_model") not in vision:
        config["vision_model"] = _NONE
    config["context_size"] = _normalize_context_size(config.get("context_size"), config.get("model"))
    config["memory_preset"] = "Custom"
    if config.get("vram_policy") not in {"Auto Yield to ComfyUI", "Keep Resident"}:
        config["vram_policy"] = "Auto Yield to ComfyUI"
    config["model_retention"] = (
        "ComfyUI Managed" if config.get("vram_policy") == "Auto Yield to ComfyUI"
        else "Persistent (Driver Managed)"
    )
    return config


def _write_config_file(config: dict[str, Any]):
    path = _config_path()
    _atomic_write_json(path, config)


def _normalized_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        role = str(m.get("role", "")).strip()
        if not role:
            raise ValueError("each message requires a role")
        # Preserve OpenAI-style text/multimodal content blocks. llama-cpp-python
        # chat handlers understand these directly when a compatible mmproj is active.
        out.append({k: copy.deepcopy(v) for k, v in m.items() if k in {"role", "content", "name", "tool_calls", "tool_call_id"}})
        out[-1]["role"] = role
        if "content" not in out[-1]:
            out[-1]["content"] = ""
    return out


class LocalLLMServiceAPI:
    API_TYPE = "LOCAL_LLM_SERVICE_API"
    API_VERSION = 2

    def __init__(self, manager: "LocalLLMServiceManager"):
        self._manager = manager

    def status(self):
        return self._manager.status()

    def get_settings(self):
        return self._manager.get_config()

    def is_loaded(self):
        return bool(self.status().get("model_loaded"))

    def generate(self, prompt=None, system_prompt=None, messages=None, image=None, video_frames=None, client="ComfyUI node", **overrides):
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": str(system_prompt)})
            messages.append({"role": "user", "content": "" if prompt is None else str(prompt)})
        return self._manager.generate_messages(messages, image=image, video_frames=video_frames, client=client, overrides=overrides)

    def query(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate_text(self, *args, **kwargs):
        return self.generate(*args, **kwargs)["response"]

    def gpu_handoff(self, reason="external-gpu-handoff"):
        """Wait for all native LLM work, yield residency, and return diagnostics."""
        return self._manager.gpu_handoff(reason=reason)


class _RequestCancelToken:
    """Generation-scoped cancellation token backed by the service stop epoch."""
    def __init__(self, manager, epoch):
        self._manager = manager
        self._epoch = int(epoch)
    def is_set(self):
        with self._manager._state_lock:
            return int(self._manager._stop_epoch) != self._epoch


class LocalLLMServiceManager:
    def __init__(self):
        self._config_lock = threading.RLock()
        self._generation_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._config = _load_config_file()
        self._state = STATE_STOPPED
        self._error = None
        self._api = None
        self._restart_required = False
        self._queue_count = 0
        self._requests_total = 0
        self._current_client = None
        self._current_started = None
        self._current_phase_started = None
        self._current_completion_tokens = 0
        self._current_tokens_per_second = 0.0
        self._current_prompt_tokens = 0
        self._current_prompt_tokens_per_second = 0.0
        self._last_progress_emit = 0.0
        self._last_progress_console = 0.0
        self._model_loaded = False
        self._last_result = None
        self._last_generation_seconds = None
        self._logs = deque(maxlen=300)
        self._service_api = LocalLLMServiceAPI(self)
        self._tuner_lock = threading.RLock()
        self._tuner_thread = None
        self._tuner_cancel = threading.Event()
        self._stop_epoch = 0
        self._stop_pending = threading.Event()
        self._stop_worker_lock = threading.RLock()
        self._stop_thread = None
        self._tuner = {
            "state": "idle", "phase": "Idle", "progress": 0.0, "candidate_index": 0,
            "candidate_total": 0, "current_candidate": None, "results": [],
            "recommendation": None, "error": None, "started_at": None, "finished_at": None,
            "profile": "Quick", "message": "Ready to benchmark the saved server configuration.",
        }
        self._log(
            f"Service manager initialized — Local GGUF LLM v{PACKAGE_VERSION} • "
            f"bridge API {BRIDGE_API_VERSION} • VRAM policy v{VRAM_POLICY_VERSION}"
        )

    @property
    def api(self):
        return self._service_api

    def _log(self, text, level="info"):
        entry = {"time": time.time(), "level": level, "message": str(text)}
        with self._state_lock:
            self._logs.append(entry)
        getattr(log, level if hasattr(log, level) else "info")("[Local LLM Server] %s", text)

    def logs(self):
        with self._state_lock:
            return list(self._logs)

    def _clear_current_request_locked(self, *, phase: bool = False, metrics: bool = False) -> None:
        """Clear live request fields while ``_state_lock`` is held.

        ``phase`` and ``metrics`` preserve the previous call-site semantics; this
        helper only centralizes assignments that had been repeated in several
        lifecycle paths.
        """
        self._current_client = None
        self._current_started = None
        if phase:
            self._current_phase_started = None
        if metrics:
            self._current_completion_tokens = 0
            self._current_tokens_per_second = 0.0
            self._current_prompt_tokens = 0
            self._current_prompt_tokens_per_second = 0.0

    @staticmethod
    def _safe_content_for_log(value, limit=8000):
        """Serialize request content without dumping image payloads or unbounded text."""
        def scrub(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if k == "image_url":
                        out[k] = "<image omitted>"
                    elif k == "url" and isinstance(v, str) and v.startswith("data:image"):
                        out[k] = "<image omitted>"
                    else:
                        out[k] = scrub(v)
                return out
            if isinstance(obj, list):
                return [scrub(x) for x in obj]
            return obj
        try:
            text = json.dumps(scrub(value), ensure_ascii=False)
        except Exception:
            text = str(value)
        if len(text) > limit:
            text = text[:limit] + "… <truncated>"
        return text

    def _progress(self, update):
        """Receive load/prompt/decode phase transitions and throttled generation progress."""
        now = time.perf_counter()
        phase = str(update.get("phase") or "generating").lower()
        try:
            tokens = max(0, int(update.get("completion_tokens") or 0))
            speed = float(update.get("tokens_per_second") or 0.0)
            prompt_tokens = max(0, int(update.get("prompt_tokens") or 0))
            prompt_speed = float(update.get("prompt_tokens_per_second") or 0.0)
        except Exception:
            return
        elapsed = float(update.get("elapsed") or 0.0)
        with self._state_lock:
            previous_state = self._state
            stopping = self._stop_pending.is_set() or previous_state == STATE_STOPPING
            if phase == "loading":
                if not stopping:
                    self._state = STATE_LOADING
                self._current_phase_started = time.time()
                self._current_completion_tokens = 0
                self._current_tokens_per_second = 0.0
            elif phase == "processing":
                if not stopping:
                    self._state = STATE_PROCESSING
                self._current_phase_started = time.time()
                self._current_completion_tokens = 0
                self._current_tokens_per_second = 0.0
            else:
                if previous_state != STATE_GENERATING:
                    self._current_phase_started = time.time()
                # No token stream event arrives until prompt evaluation has finished;
                # generation progress is therefore the prompt->decode transition.
                if not stopping:
                    self._state = STATE_GENERATING
                self._current_completion_tokens = tokens
                self._current_tokens_per_second = speed
                self._current_prompt_tokens = prompt_tokens
                self._current_prompt_tokens_per_second = prompt_speed
            emit_due = phase in {"loading", "processing"} or now - self._last_progress_emit >= 0.35
            console_due = phase == "generating" and now - self._last_progress_console >= 1.0
            if emit_due:
                self._last_progress_emit = now
            if console_due:
                self._last_progress_console = now
        if console_due:
            log.info(
                "[Local LLM Server] Generating%s: %.1f tok/s • %d tokens • %.1fs",
                f" for {self._current_client}" if self._current_client else "",
                speed, tokens, elapsed,
            )
        if emit_due:
            self._emit()


    def _emit(self):
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("local_llm_server_status", self.status())
        except Exception:
            pass

    def get_config(self):
        with self._config_lock:
            return copy.deepcopy(self._config)

    def public_config(self):
        c = self.get_config()
        c["api_key_set"] = bool(c.get("api_key"))
        return c

    def request_generation_defaults(self):
        """Return the effective sampler values used by an unmodified request.

        The global model preset is resolved here so the lightweight Generate node
        can display the exact values the server will use instead of generic widget
        defaults. max_tokens remains the server's explicit request limit.
        """
        cfg = self.get_config()
        values = {k: copy.deepcopy(cfg.get(k)) for k in REQUEST_PRESET_FIELDS}
        model = cfg.get("model")
        preset = cfg.get("model_preset")
        if preset == "Auto (Detected)":
            try:
                md = _metadata_for(model) if model and model != "No GGUF models found" else {}
                preset = recommended_model_preset(md, model or "")
            except Exception:
                preset = "Generic Chat"
        if preset in MODEL_PRESETS and cfg.get("model_preset") != "Custom":
            model_values = MODEL_PRESETS[preset]
            for key in REQUEST_PRESET_FIELDS:
                if key in model_values:
                    values[key] = copy.deepcopy(model_values[key])
        # Normalize types for frontend/backend stability.
        for key in ("top_k", "max_tokens"):
            try:
                values[key] = int(values[key])
            except Exception:
                pass
        for key in set(REQUEST_PRESET_FIELDS) - {"top_k", "max_tokens"}:
            try:
                values[key] = float(values[key])
            except Exception:
                pass
        return values

    def request_preset_catalog(self):
        # Backward-compatible sampler-only view used by v0.8 clients.
        presets = sampler_presets()
        return {
            "directory": str(SAMPLER_PRESET_DIR),
            "default": self.request_generation_defaults(),
            "names": ["Default", "Custom", *sorted(presets, key=str.lower)],
            "presets": {name: copy.deepcopy(item["settings"]) for name, item in presets.items()},
        }

    def node_preset_catalog(self):
        samplers = sampler_presets()
        systems = text_presets("system_prompts")
        prompts = text_presets("prompts")
        complete = complete_settings_presets()
        builtins = public_presets()
        return {
            "root_directory": str(PRESET_ROOT_DIR),
            "model_runtime_presets": copy.deepcopy(builtins.get("model") or {}),
            "settings": {
                "directory": str(SETTINGS_PRESET_DIR),
                "names": ["Current Server", "Custom", *sorted(complete, key=str.lower)],
                "deletable_names": sorted(complete, key=str.lower),
                "presets": {name: copy.deepcopy(item["settings"]) for name, item in complete.items()},
            },
            "sampler": {
                "directory": str(SAMPLER_PRESET_DIR),
                "default": self.request_generation_defaults(),
                "names": ["Default", "Custom", *sorted(samplers, key=str.lower)],
                "deletable_names": sorted(samplers, key=str.lower),
                "presets": {name: copy.deepcopy(item["settings"]) for name, item in samplers.items()},
            },
            "system_prompts": {
                "directory": str(SYSTEM_PROMPT_PRESET_DIR),
                "names": ["Custom", *sorted(systems, key=str.lower)],
                "deletable_names": sorted(systems, key=str.lower),
                "presets": {name: str(item.get("text", "")) for name, item in systems.items()},
            },
            "prompts": {
                "directory": str(PROMPT_PRESET_DIR),
                "names": ["Custom", *sorted(prompts, key=str.lower)],
                "deletable_names": sorted(prompts, key=str.lower),
                "presets": {name: str(item.get("text", "")) for name, item in prompts.items()},
            },
        }

    def update_config(self, patch: dict[str, Any]):
        if not isinstance(patch, dict):
            raise TypeError("configuration payload must be an object")
        defaults = _default_config()
        allowed = set(defaults)
        clean = {k: copy.deepcopy(v) for k, v in patch.items() if k in allowed}
        with self._config_lock:
            before = copy.deepcopy(self._config)
            self._config.update(clean)
            # Global service configuration uses concrete memory values.  Keep the
            # old field pinned to Custom so LocalGGUFLLM cannot re-apply a legacy
            # memory preset over user/preset values.
            self._config["memory_preset"] = "Custom"
            self._config["context_size"] = _normalize_context_size(
                self._config.get("context_size"), self._config.get("model")
            )
            if self._config.get("vram_policy") not in {"Auto Yield to ComfyUI", "Keep Resident"}:
                self._config["vram_policy"] = "Auto Yield to ComfyUI"
            self._config["model_retention"] = (
                "ComfyUI Managed" if self._config.get("vram_policy") == "Auto Yield to ComfyUI"
                else "Persistent (Driver Managed)"
            )
            # External access is opt-in and receives a strong bearer key by default.
            # This prevents accidentally exposing an unauthenticated GPU endpoint
            # when ComfyUI itself is listening on a LAN interface.
            if self._config.get("external_api_enabled") and not str(self._config.get("api_key") or "").strip():
                self._config["api_key"] = "sk-local-" + secrets.token_urlsafe(24)
            _write_config_file(self._config)
            if self._state in {STATE_READY, STATE_PROCESSING, STATE_GENERATING, STATE_WAITING_COMFY}:
                # Autosave can issue several small/full configuration updates in
                # succession. Once load-affecting settings make the resident
                # native context stale, a later sampler/UI-only update must not
                # clear that pending reload. It is cleared only after Start/Reload
                # successfully constructs a context from the current config.
                self._restart_required = self._restart_required or any(
                    before.get(k) != self._config.get(k) for k in LOAD_FIELDS
                )
        self._log("Configuration updated" + ("; new load settings will apply on next request" if self._restart_required else ""))
        self._emit()
        return self.public_config()

    def regenerate_api_key(self):
        key = "sk-local-" + secrets.token_urlsafe(24)
        self.update_config({"api_key": key})
        return key

    def _call_args(self, config=None):
        cfg = self.get_config() if config is None else copy.deepcopy(config)
        args = _node_defaults()
        args.update({k: v for k, v in cfg.items() if k in args})
        args["model_retention"] = (
            "ComfyUI Managed" if cfg.get("vram_policy", "Auto Yield to ComfyUI") == "Auto Yield to ComfyUI"
            else "Persistent (Driver Managed)"
        )
        # These are only placeholders when messages_override is supplied.
        args["system_prompt"] = cfg.get("system_prompt", "")
        args["prompt"] = cfg.get("prompt", "")
        return args


    def _comfy_running_jobs(self):
        """Return currently executing ComfyUI prompt jobs without mutating the queue."""
        try:
            from server import PromptServer
            server = PromptServer.instance
            queue = getattr(server, "prompt_queue", None)
            if queue is None:
                return []
            getter = getattr(queue, "get_current_queue_volatile", None) or getattr(queue, "get_current_queue", None)
            if getter is None:
                return []
            running, _pending = getter()
            return list(running or [])
        except Exception:
            return []

    def _wait_for_comfy_idle(self, reason="LLM GPU request", cancel_event=None):
        """Serialize external/native LLM GPU work behind active ComfyUI execution.

        Calling ComfyUI's memory manager from another thread while diffusion CUDA
        kernels are still executing can invalidate allocations that those kernels
        are using.  External service requests therefore wait until the active
        ComfyUI prompt finishes before loading/reloading or running llama.cpp.
        Workflow-owned Local LLM Generate calls bypass this guard because
        they already execute serially inside that same ComfyUI prompt.
        """
        logged = False
        started = time.perf_counter()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise LocalLLMInterrupted("Local LLM operation stopped by user")
            running = self._comfy_running_jobs()
            if not running:
                elapsed = time.perf_counter() - started
                if logged:
                    self._log(f"ComfyUI is idle; continuing {reason} (waited {elapsed:.3f}s)")
                return elapsed
            with self._state_lock:
                self._state = STATE_WAITING_COMFY
            if not logged:
                self._log(f"Waiting for active ComfyUI workflow to finish before {reason}")
                logged = True
            self._emit()
            time.sleep(0.10)

    @staticmethod
    def _is_workflow_client(client):
        return str(client or "").startswith("ComfyUI Local LLM Generate")

    @contextmanager
    def _generation_slot(self, *, workflow_owned: bool, cancel_event=None, reason: str = "servicing the LLM request"):
        """Acquire the service generation lock without deadlocking ComfyUI.

        External requests must wait for ComfyUI to become idle, but they must *not*
        hold ``_generation_lock`` while doing so: an active H3/Enhancer workflow
        may need that same lock to perform the blocking Local-LLM GPU handoff before
        its queue item can finish.  Holding the lock during the idle wait creates a
        classic cycle (external request waits for ComfyUI; ComfyUI waits for the
        external request's generation lock).

        We therefore wait outside the lock, acquire it, and re-check ComfyUI. If
        another workflow became active while we were competing for the lock, release
        it and repeat. Workflow-owned callers are already serialized by ComfyUI and
        bypass the idle check.
        """
        comfy_wait_seconds = 0.0
        generation_lock_wait_seconds = 0.0
        acquired = False
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise LocalLLMInterrupted("Local LLM request cancelled while waiting for its generation turn")
                if not workflow_owned:
                    comfy_wait_seconds += float(
                        self._wait_for_comfy_idle(reason, cancel_event=cancel_event) or 0.0
                    )
                lock_started = time.perf_counter()
                while not acquired:
                    if cancel_event is not None and cancel_event.is_set():
                        raise LocalLLMInterrupted("Local LLM request cancelled while waiting for its generation turn")
                    acquired = self._generation_lock.acquire(timeout=0.10)
                generation_lock_wait_seconds += time.perf_counter() - lock_started
                if cancel_event is not None and cancel_event.is_set():
                    raise LocalLLMInterrupted("Local LLM request cancelled while waiting for its generation turn")
                if workflow_owned or not self._comfy_running_jobs():
                    break
                # ComfyUI started/restarted while this external caller was waiting
                # for the generation lock. Never sleep for ComfyUI idle while the
                # lock is held; release and retry instead.
                self._generation_lock.release()
                acquired = False
            yield {
                "comfy_wait_seconds": float(comfy_wait_seconds),
                "generation_lock_wait_seconds": float(generation_lock_wait_seconds),
            }
        finally:
            if acquired:
                self._generation_lock.release()

    def _warm_load(self, cancel_event=None):
        args = self._call_args()
        model = args.get("model")
        if not model or model == "No GGUF models found":
            raise FileNotFoundError("No GGUF model is configured. Put a GGUF in ComfyUI/models/llm and select it in Local LLM Server.")
        # One-token warmup deliberately exercises the exact chat handler/template
        # path that real callers will use. This guarantees Start means actually
        # loaded/ready rather than only configured.
        args.update({
            "max_tokens": 1,
            "seed": 0,
            "messages_override": [{"role": "user", "content": "Respond with OK."}],
            "cancel_event": cancel_event,
        })
        response, thinking, info_json, tokens, api = LocalGGUFLLM().generate(**args)
        self._api = api
        try:
            info = json.loads(info_json)
        except Exception:
            info = {"raw": info_json}
        self._last_result = {"response": response, "thinking": thinking, "info": info, "tokens": int(tokens)}

    def start(self, wait_for_comfy=True):
        # Capture the current stop epoch so a Stop click can cancel both a
        # ComfyUI-idle wait and the warm-load request without destroying native
        # llama.cpp state concurrently.
        with self._state_lock:
            if self._state == STATE_STOPPING:
                return self.status()
            epoch = self._stop_epoch
        cancel_token = _RequestCancelToken(self, epoch)
        with self._generation_slot(
            workflow_owned=not bool(wait_for_comfy),
            cancel_event=cancel_token,
            reason="loading the LLM",
        ):
            if cancel_token.is_set():
                raise LocalLLMInterrupted("Local LLM start stopped by user")
            with self._state_lock:
                if self._state in {STATE_RELOADING, STATE_PROCESSING, STATE_GENERATING, STATE_STOPPING, STATE_TUNING}:
                    return self.status()
                if self._state == STATE_READY and self._api is not None and not self._restart_required and self._resident_snapshot():
                    return self.status()
                self._state = STATE_LOADING
                self._error = None
            self._emit()
            self._log("Loading persistent GGUF service")
            try:
                self._warm_load(cancel_event=cancel_token)
                if cancel_token.is_set():
                    raise LocalLLMInterrupted("Local LLM start stopped by user")
                with self._state_lock:
                    self._state = STATE_READY
                    self._model_loaded = True
                    self._restart_required = False
                    self._error = None
                self._log("Model loaded; service ready")
            except LocalLLMInterrupted:
                with self._state_lock:
                    if self._state != STATE_STOPPING:
                        self._state = STATE_READY if self._api is not None else STATE_STOPPED
                    self._error = None
                self._log("LLM start interrupted by Stop")
                raise
            except Exception as e:
                with self._state_lock:
                    self._state = STATE_ERROR
                    self._model_loaded = False
                    self._error = f"{type(e).__name__}: {e}"
                    self._api = None
                self._log(self._error, "error")
                log.exception("[Local LLM Server] Service start failed")
                raise
            finally:
                self._emit()
        return self.status()

    def gpu_handoff(self, reason="external-gpu-handoff"):
        """Establish a blocking native-LLM -> other-GPU-owner handoff boundary.

        The service generation lock serializes service callers. The lower-level
        process-global native gate additionally waits for facade/direct-engine work
        that did not enter through this service. When this method returns, no
        top-level llama.cpp context is resident in either Auto-Yield or Keep
        Resident mode. The service/API configuration remains ready for on-demand
        reconstruction after the other GPU owner has finished.
        """
        reason = str(reason or "external-gpu-handoff")
        if _native_operation_owned_by_current_thread():
            raise RuntimeError(
                "Local LLM GPU handoff cannot be initiated re-entrantly from inside an active native llama.cpp operation. "
                "Return from generation first, then request the handoff."
            )
        with self._generation_lock:
            self._log(f"GPU handoff requested: {reason}; waiting for native LLM ownership barrier")
            started = time.perf_counter()
            native = _suspend_llm_native(reason=reason, heavy_cleanup=False)
            with self._state_lock:
                if self._state not in {STATE_STOPPED, STATE_STOPPING, STATE_ERROR}:
                    self._state = STATE_READY if self._api is not None else STATE_STOPPED
                    self._error = None
                # `_model_loaded` historically means the service has a usable model
                # configuration/API; live residency is reported separately by
                # `_resident_snapshot()` / `vram_yielded`.
                self._model_loaded = bool(self._api is not None and self._state != STATE_STOPPED)
                self._clear_current_request_locked(phase=True, metrics=True)
            native_after = self._resident_snapshot()
            if native_after:
                raise RuntimeError("Local LLM GPU handoff returned while a native llama.cpp context was still resident.")
            elapsed = time.perf_counter() - started
            self._log(
                "Native GGUF GPU handoff complete "
                f"(path={native.get('path', 'unknown')}, total={elapsed:.3f}s, "
                f"released_accounted={int(native.get('released_accounted_bytes') or 0) / (1024*1024):.1f} MiB)"
            )
            self._emit()
            return {
                "ok": True,
                "reason": reason,
                "elapsed": elapsed,
                "native": copy.deepcopy(native),
                "native_operation": _native_operation_snapshot(),
                "status": self.status(),
            }

    def suspend(self, reason="manual-suspend"):
        """Yield native residency while keeping the service configured and ready.

        v0.18.54 routes Suspend through the same blocking handoff contract used by
        sibling GPU-heavy node packs. This now works for both Auto Yield and Keep
        Resident modes and cannot return early while another native LLM caller is
        still active.
        """
        self.gpu_handoff(reason=str(reason or "manual-suspend"))
        return self.status()

    def _stop_worker_main(self):
        """Finish teardown only after active native work has safely unwound."""
        try:
            with self._tuner_lock:
                tuner_thread = self._tuner_thread
            if tuner_thread is not None and tuner_thread.is_alive() and tuner_thread is not threading.current_thread():
                tuner_thread.join()

            with self._generation_lock:
                self._log("Stop barrier reached; unloading persistent GGUF")
                try:
                    _cleanup_llm()
                finally:
                    with self._state_lock:
                        self._api = None
                        self._model_loaded = False
                        self._state = STATE_STOPPED
                        self._error = None
                        self._restart_required = False
                        self._clear_current_request_locked(phase=True, metrics=True)
                    self._stop_pending.clear()
                    self._emit()
                    self._log("Local LLM service stopped and native model unloaded")
        except Exception as e:
            with self._state_lock:
                self._state = STATE_ERROR
                self._error = f"{type(e).__name__}: {e}"
            self._stop_pending.clear()
            self._log(self._error, "error")
            self._emit()
        finally:
            with self._stop_worker_lock:
                if self._stop_thread is threading.current_thread():
                    self._stop_thread = None

    def stop(self):
        """Request a global LLM interrupt immediately and tear down safely.

        This call never waits behind `_generation_lock`. Advancing the stop epoch
        cancels the active request and all LLM requests already queued behind it.
        A background worker unloads llama.cpp only after those calls unwind, which
        avoids the CUDA/context corruption caused by destroying a live context from
        another thread. GPU prompt-prefill can only stop at llama.cpp's next safe
        Python/sampling boundary; current llama.cpp abort callbacks are CPU-only.
        """
        with self._state_lock:
            if self._state == STATE_STOPPED and not self._stop_pending.is_set():
                return self.status()
            if not self._stop_pending.is_set():
                self._stop_epoch += 1
                self._stop_pending.set()
            self._state = STATE_STOPPING
            self._error = None
        self._tuner_cancel.set()
        self._emit()
        self._log("Stop requested; interrupting active/queued LLM work")

        with self._stop_worker_lock:
            if self._stop_thread is None or not self._stop_thread.is_alive():
                worker = threading.Thread(target=self._stop_worker_main, name="LocalLLMStopWorker", daemon=True)
                self._stop_thread = worker
                worker.start()
        return self.status()

    def reload(self, wait_for_comfy=True):
        if self.tuner_status().get("running"):
            raise RuntimeError("Cancel the performance tuner before reloading the service")
        with self._state_lock:
            if self._state == STATE_STOPPING:
                return self.status()
            epoch = self._stop_epoch
        cancel_token = _RequestCancelToken(self, epoch)
        with self._generation_slot(
            workflow_owned=not bool(wait_for_comfy),
            cancel_event=cancel_token,
            reason="reloading the LLM",
        ):
            if cancel_token.is_set():
                raise LocalLLMInterrupted("Local LLM reload stopped by user")
            with self._state_lock:
                self._state = STATE_RELOADING
                self._error = None
            self._emit()
            try:
                _cleanup_llm()
                self._api = None
                if cancel_token.is_set():
                    raise LocalLLMInterrupted("Local LLM reload stopped by user")
                self._warm_load(cancel_event=cancel_token)
                if cancel_token.is_set():
                    raise LocalLLMInterrupted("Local LLM reload stopped by user")
                with self._state_lock:
                    self._state = STATE_READY
                    self._model_loaded = True
                    self._restart_required = False
                    self._error = None
                self._log("Model reloaded")
            except LocalLLMInterrupted:
                with self._state_lock:
                    if self._state != STATE_STOPPING:
                        self._state = STATE_READY if self._api is not None else STATE_STOPPED
                    self._error = None
                self._log("LLM reload interrupted by Stop")
                raise
            except Exception as e:
                with self._state_lock:
                    self._state = STATE_ERROR
                    self._model_loaded = False
                    self._error = f"{type(e).__name__}: {e}"
                    self._api = None
                self._log(self._error, "error")
                log.exception("[Local LLM Server] Service reload failed")
                raise
            finally:
                self._emit()
        return self.status()

    def _resident_snapshot(self):
        """Lock-free residency snapshot for status/UI and fast request checks.

        ComfyUI may trigger the thin native-residency yield hook outside the service request path.
        Reading these Python references is safe enough for an advisory snapshot and
        intentionally avoids the native model lock held throughout generation.
        """
        try:
            adapter = _MODEL_CACHE.get("managed_adapter")
            if adapter is not None:
                return getattr(adapter, "llm", None) is not None
            return _MODEL_CACHE.get("llm") is not None
        except Exception:
            return bool(self._model_loaded)

    def _ensure_started(self, workflow_owned=False):
        with self._state_lock:
            state = self._state
            has_api = self._api is not None
            restart_required = self._restart_required
        # READY + configured API remains request-ready even when ComfyUI has
        # evicted the native context.  The actual LocalGGUFLLM.generate() call
        # below will reload the yielded native context exactly once and then perform the
        # requested generation; do not waste time on a separate warm-up reload.
        if state == STATE_READY and has_api and not restart_required:
            return
        startup = self.get_config().get("startup_mode", "On Demand")
        if state == STATE_STOPPED and startup == "Off":
            raise RuntimeError("Local LLM Server is stopped. Start it from the LLM sidebar panel.")
        # ``_ensure_started`` is called only from ``generate_messages`` after that
        # caller has already acquired `_generation_slot`. External callers were
        # therefore checked for ComfyUI idleness *before* taking the generation
        # lock; do not perform a second idle wait from inside the re-entrant lock,
        # which could recreate the external-request <-> workflow handoff deadlock.
        if restart_required and state == STATE_READY:
            self.reload(wait_for_comfy=False)
        else:
            # A READY service with an evicted/yielded model is reloaded on demand.
            self.start(wait_for_comfy=False)

    def generate_messages(self, messages, image=None, video_frames=None, client="unknown", overrides=None, token_callback=None, runtime_config=None, config_snapshot=None):
        messages = _normalized_messages(messages)
        overrides = dict(overrides or {})
        # API/client request fields may not change persistent allocation settings.
        # Workflow Settings nodes use the separate runtime_config channel below;
        # those values are request-local and never mutate the modal/server config.
        forbidden = sorted(set(overrides) & LOAD_FIELDS)
        if forbidden:
            raise ValueError("Request cannot override server load/memory setting(s): " + ", ".join(forbidden))

        persistent_cfg = self.get_config()
        # Internal callers such as Prompt Enhancer batches may pin the complete
        # runtime configuration that existed when a multi-request operation
        # began.  This is request-local: autosaved modal settings remain the
        # authoritative persistent config and become active after the batch.
        snapshot_clean = copy.deepcopy(config_snapshot) if isinstance(config_snapshot, dict) and config_snapshot else None
        effective_cfg = copy.deepcopy(snapshot_clean if snapshot_clean is not None else persistent_cfg)
        runtime_clean = {}
        if isinstance(runtime_config, dict) and runtime_config:
            runtime_clean = _validate_complete_settings(runtime_config)
            effective_cfg.update(copy.deepcopy(runtime_clean))
            effective_cfg["memory_preset"] = "Custom"
            effective_cfg["context_size"] = _normalize_context_size(
                effective_cfg.get("context_size"), effective_cfg.get("model")
            )
            if effective_cfg.get("vram_policy") not in {"Auto Yield to ComfyUI", "Keep Resident"}:
                effective_cfg["vram_policy"] = "Auto Yield to ComfyUI"
            effective_cfg["model_retention"] = (
                "ComfyUI Managed" if effective_cfg.get("vram_policy") == "Auto Yield to ComfyUI"
                else "Persistent (Driver Managed)"
            )
        request_local_config = snapshot_clean is not None or bool(runtime_clean)
        temporary_load_change = bool(request_local_config) and any(
            persistent_cfg.get(key) != effective_cfg.get(key)
            for key in LOAD_FIELDS
        )

        with self._state_lock:
            if self._state == STATE_STOPPING:
                raise LocalLLMInterrupted("Local LLM service is stopping")
            request_epoch = self._stop_epoch
            self._queue_count += 1
        cancel_token = _RequestCancelToken(self, request_epoch)
        self._emit()
        queued_at = time.perf_counter()
        workflow_owned = self._is_workflow_client(client)
        queue_registration_pending = True
        try:
            with self._generation_slot(
                workflow_owned=workflow_owned,
                cancel_event=cancel_token,
                reason="servicing the LLM request",
            ) as slot:
                generation_lock_acquired = time.perf_counter()
                queue_wait_seconds = generation_lock_acquired - queued_at
                comfy_wait_seconds = float(slot.get("comfy_wait_seconds") or 0.0)
                with self._state_lock:
                    self._queue_count = max(0, self._queue_count - 1)
                    queue_registration_pending = False
                if cancel_token.is_set():
                    raise LocalLLMInterrupted("Queued Local LLM request cancelled by Stop")
                ensure_started_at = time.perf_counter()
                if temporary_load_change or snapshot_clean is not None:
                    # Request-local Settings and Prompt Enhancer batch snapshots
                    # must not consume the persistent modal's pending-reload flag.
                    # A batch therefore keeps using its pinned native signature
                    # even if the user autosaves a different model midway through
                    # it; LocalGGUFLLM switches/reloads only when this pinned
                    # request itself actually needs native residency.
                    with self._state_lock:
                        request_state = self._state
                    if request_state == STATE_STOPPED and persistent_cfg.get("startup_mode", "On Demand") == "Off":
                        raise RuntimeError("Local LLM Server is stopped. Start it from the LLM sidebar panel.")
                else:
                    self._ensure_started(workflow_owned=workflow_owned)
                ensure_started_seconds = time.perf_counter() - ensure_started_at
                if cancel_token.is_set():
                    raise LocalLLMInterrupted("Local LLM request cancelled by Stop")
                request_needs_reload = temporary_load_change or not self._resident_snapshot()
                self._log(
                    "PERF request preflight: "
                    f"queue_wait={queue_wait_seconds:.3f}s • "
                    f"comfy_wait={comfy_wait_seconds:.3f}s • "
                    f"ensure_started={ensure_started_seconds:.3f}s • "
                    f"resident_before_call={not request_needs_reload}"
                )
                with self._state_lock:
                    if cancel_token.is_set():
                        raise LocalLLMInterrupted("Local LLM request cancelled by Stop")
                    self._state = STATE_LOADING if request_needs_reload else STATE_PROCESSING
                    self._model_loaded = True
                    self._current_client = str(client)
                    self._current_started = time.time()
                    self._current_phase_started = self._current_started
                    self._current_completion_tokens = 0
                    self._current_tokens_per_second = 0.0
                    self._current_prompt_tokens = 0
                    self._current_prompt_tokens_per_second = 0.0
                    self._last_progress_emit = 0.0
                    self._last_progress_console = 0.0
                    self._requests_total += 1
                self._emit()
                self._log(f"Processing request from {client}")
                cfg = effective_cfg
                if cfg.get("log_prompt_content"):
                    self._log("Prompt content: " + self._safe_content_for_log(messages))
                started = time.perf_counter()
                # Re-enter the canonical node execution path with the effective
                # request configuration. Normally this is the authoritative modal
                # configuration; a connected Settings node may supply a temporary
                # workflow-only runtime overlay without persisting it.
                args = self._call_args(cfg)
                args["messages_override"] = messages
                args.update(overrides)
                cfg_model = str(cfg.get("model") or "")
                cfg_model_md = _metadata_for(cfg_model) if cfg_model and cfg_model != "No GGUF models found" else {}
                family = detect_family(cfg_model_md or {}, cfg_model) if cfg_model else "unknown"
                caps = capabilities_for_family(family)
                ignored_vision_inputs = []
                if caps.get("vision") is False:
                    if image is not None:
                        ignored_vision_inputs.append("image(s)")
                        image = None
                    if video_frames is not None:
                        ignored_vision_inputs.append("video_frames")
                        video_frames = None
                    if ignored_vision_inputs:
                        self._log(
                            f"Ignoring {' and '.join(ignored_vision_inputs)} input for detected text-only model family '{family}'.",
                            level="warning",
                        )
                if image is not None:
                    args["image"] = image
                if video_frames is not None:
                    args["video_frames"] = video_frames
                args["progress_callback"] = self._progress
                args["token_callback"] = token_callback
                args["cancel_event"] = cancel_token
                node_call_started = time.perf_counter()
                try:
                    response, thinking, info_json, tokens, api = LocalGGUFLLM().generate(**args)
                except Exception as first_exc:
                    with self._state_lock:
                        emitted_tokens = int(self._current_completion_tokens or 0)
                    recoverable_decode = bool(
                        getattr(first_exc, "local_gguf_decode_context_invalidated", False)
                    )
                    if not recoverable_decode or emitted_tokens > 0:
                        raise

                    recovery_path = str(
                        getattr(first_exc, "local_gguf_decode_recovery_path", "fresh-context")
                    )
                    self._log(
                        "Native llama.cpp decode context failed before output; "
                        f"discarded poisoned context via {recovery_path} and retrying request once.",
                        "warning",
                    )
                    with self._state_lock:
                        self._state = STATE_LOADING
                        self._current_phase_started = time.time()
                        self._current_completion_tokens = 0
                        self._current_tokens_per_second = 0.0
                        self._current_prompt_tokens = 0
                        self._current_prompt_tokens_per_second = 0.0
                    self._emit()
                    response, thinking, info_json, tokens, api = LocalGGUFLLM().generate(**args)
                node_call_seconds = time.perf_counter() - node_call_started
                self._api = api
                try:
                    parsed_info = json.loads(info_json)
                except Exception:
                    parsed_info = {"raw": info_json}
                result = {
                    "response": response,
                    "thinking": thinking,
                    "info": parsed_info,
                    "tokens": int(tokens),
                }
                response_chars = len(str(response or ""))
                thinking_chars = len(str(thinking or ""))
                self._log(f"Output received: {response_chars} final chars • {thinking_chars} reasoning chars")
                if response_chars == 0 and thinking_chars > 0:
                    self._log(
                        "Request produced reasoning-only output and no final content. "
                        "Clients with reasoning display disabled can appear blank; check Thinking Mode, max_tokens, and stop sequences.",
                        "warning",
                    )
                if cfg.get("log_response_content"):
                    self._log("Response content: " + self._safe_content_for_log(response))
                elapsed = time.perf_counter() - started
                with self._state_lock:
                    self._last_generation_seconds = elapsed
                    self._last_result = copy.deepcopy(result)
                    self._state = STATE_READY
                    self._clear_current_request_locked(phase=True, metrics=True)
                    self._error = None
                info = result.get("info") or {}
                speed = info.get("tokens_per_second")
                spec = info.get("speculative") or {}
                if spec.get("effective") and spec.get("effective") != "Off":
                    stats = spec.get("stats") or {}
                    drafted = stats.get("drafted_tokens_normalized")
                    accepted = stats.get("accepted_tokens_normalized")
                    rate = stats.get("acceptance_rate")
                    stat_text = ""
                    if drafted is not None or accepted is not None:
                        stat_text = f" • drafted={int(drafted or 0)} • accepted={int(accepted or 0)}"
                    if rate is not None:
                        stat_text += f" • acceptance={float(rate)*100:.1f}%"
                    self._log(
                        f"Speculative decoding: requested={spec.get('requested')} • effective={spec.get('effective')} "
                        f"• implementation={spec.get('implementation') or 'n/a'}{stat_text}"
                    )
                prompt_cache = info.get("prompt_cache") or {}
                if prompt_cache:
                    reused = int(prompt_cache.get("reused_tokens") or 0)
                    total_prompt = int(prompt_cache.get("prompt_tokens") or 0)
                    evaluated = int(prompt_cache.get("evaluated_tokens") or 0)
                    reuse_pct = float(prompt_cache.get("reuse_percent") or 0.0)
                    saved = float(prompt_cache.get("estimated_seconds_saved") or 0.0)
                    uncached_rate = float(prompt_cache.get("uncached_prompt_tokens_per_second") or 0.0)
                    effective_rate = float(prompt_cache.get("effective_prompt_tokens_per_second") or 0.0)
                    self._log(
                        f"Prompt cache: requested={prompt_cache.get('requested', 'Auto')} • "
                        f"effective={prompt_cache.get('effective', 'Auto')} • "
                        f"hit={bool(prompt_cache.get('hit'))} • reused={reused}/{total_prompt} ({reuse_pct:.1f}%) • "
                        f"evaluated={evaluated} • eval_rate={uncached_rate:.1f}t/s • effective_rate={effective_rate:.1f}t/s • "
                        f"est_saved={saved:.3f}s • scope={prompt_cache.get('scope', 'resident-context')} • "
                        f"resident_ctx={int(prompt_cache.get('resident_tokens_before') or 0)}→{int(prompt_cache.get('resident_tokens_start') or 0)}→{int(prompt_cache.get('resident_tokens_after') or 0)}"
                    )
                gpu_backend = info.get("gpu_backend") or {}
                preload = gpu_backend.get("preload_memory") or {}
                last_unload = gpu_backend.get("last_unload") or {}
                self._log(
                    "PERF lifecycle: "
                    f"node_call={node_call_seconds:.3f}s • "
                    f"load_total={float(info.get('load_seconds') or 0):.3f}s • "
                    f"load_path={gpu_backend.get('load_path', 'n/a')} • "
                    f"native_load={float(gpu_backend.get('native_load_seconds') or 0):.3f}s • "
                    f"coordination={gpu_backend.get('coordination_mode', 'native-direct')} • "
                    f"signature={gpu_backend.get('signature_id', 'n/a')} • "
                    f"comfy_loader={float(gpu_backend.get('comfy_load_models_gpu_seconds') or 0):.3f}s • "
                    f"handoff={float(preload.get('handoff_seconds') or 0):.3f}s({preload.get('strategy', 'none')}) • "
                    f"prompt={float(info.get('prompt_eval_seconds') or 0):.3f}s • "
                    f"decode={float(info.get('generation_seconds') or 0):.3f}s • "
                    f"prior_yield_close={float(last_unload.get('close_seconds') or 0):.3f}s"
                    f"({last_unload.get('reason', 'none')})"
                )
                room = preload.get("room_details") or {}
                self._log(
                    "PERF GPU lease: "
                    f"runtime={float(room.get('runtime_target_bytes') or preload.get('runtime_target_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"headroom={float(room.get('headroom_bytes') or preload.get('headroom_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"free_target={float(room.get('free_target_bytes') or preload.get('free_target_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"request={float(room.get('request_target_bytes') or preload.get('request_target_bytes') or 0) / (1024*1024):.1f}MiB "
                    f"coop_request={float(room.get('cooperative_request_target_bytes') or 0) / (1024*1024):.1f}MiB "
                    f"torch_slack={float(room.get('cooperative_torch_slack_bytes') or 0) / (1024*1024):.1f}MiB "
                    f"coop_retry={float(room.get('cooperative_retry_request_target_bytes') or 0) / (1024*1024):.1f}MiB "
                    f"source={preload.get('target_source', 'n/a')} • "
                    f"estimated={float(preload.get('estimated_vram_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"observed={float(preload.get('observed_vram_bytes') or gpu_backend.get('observed_vram_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"raw_before={float(room.get('raw_free_before_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"raw_after_cache={float(room.get('raw_free_after_cache_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"raw_after_coop={float(room.get('raw_free_after_cooperative_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"raw_after={float(room.get('raw_free_after_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"strategy={room.get('strategy', 'none')} • "
                    f"cache_reclaim={bool(room.get('cache_reclaim_called', False))}({room.get('cache_reclaim_stage', 'none')}) • "
                    f"cooperative={bool(room.get('cooperative_eviction_called', False))} "
                    f"unloaded={int(room.get('cooperative_unloaded_count') or 0)} • "
                    f"aimdo_cleanup={bool(room.get('aimdo_cleanup_called', False))} • "
                    f"exclusive={bool(room.get('exclusive_eviction_called', False))} • "
                    f"final_sync={bool(room.get('final_sync_called', False))} • "
                    f"satisfied={bool(room.get('satisfied', True))} • "
                    f"lease_time={float(room.get('elapsed_seconds') or 0):.3f}s"
                )
                # Detailed but compact lease telemetry. This is diagnostic-only:
                # it exposes the accounting mismatch and stage costs without
                # changing admission policy or adding per-token logging.
                stages = room.get("stage_memory") or {}
                timings = room.get("stage_timings_seconds") or {}
                unloaded_models = room.get("unloaded_models") or {}

                def _mib(v):
                    return float(v or 0) / (1024 * 1024)

                def _stage(name):
                    value = stages.get(name) or {}
                    models = value.get("loaded_models") or {}
                    return (
                        f"{name}[raw={_mib(value.get('raw_free_bytes')):.1f},"
                        f"logical={_mib(value.get('comfy_logical_free_bytes')):.1f},"
                        f"slack={_mib(value.get('torch_reclaimable_bytes')):.1f},"
                        f"used={_mib(value.get('raw_used_bytes')):.1f},"
                        f"models={int(models.get('same_device_count') or 0)}/{int(models.get('count') or 0)}]"
                    )

                if stages:
                    ordered_names = [
                        "before", "after_pre_cache", "after_cooperative",
                        "after_cooperative_retry", "after_aimdo_cleanup",
                        "after_aimdo_retry", "after_exclusive", "after_final_sync",
                    ]
                    present = [name for name in ordered_names if name in stages]
                    self._log("PERF GPU accounting: " + " → ".join(_stage(name) for name in present))

                    before_stage = stages.get("before") or {}
                    coop_stage = stages.get("after_cooperative") or {}
                    final_stage = stages.get("after_final_sync") or stages.get("after_exclusive") or stages.get("after_aimdo_retry") or stages.get("after_aimdo_cleanup") or coop_stage or before_stage
                    self._log(
                        "PERF GPU deltas: "
                        f"target_shortfall_before={max(0.0, _mib(room.get('free_target_bytes')) - _mib(before_stage.get('raw_free_bytes'))):.1f}MiB • "
                        f"logical_minus_raw_before={_mib(before_stage.get('logical_minus_raw_bytes')):.1f}MiB • "
                        f"raw_gain_coop={(_mib(coop_stage.get('raw_free_bytes')) - _mib(before_stage.get('raw_free_bytes'))):.1f}MiB • "
                        f"slack_delta_coop={(_mib(coop_stage.get('torch_reclaimable_bytes')) - _mib(before_stage.get('torch_reclaimable_bytes'))):.1f}MiB • "
                        f"raw_gain_total={(_mib(final_stage.get('raw_free_bytes')) - _mib(before_stage.get('raw_free_bytes'))):.1f}MiB • "
                        f"logical_margin_final={(_mib(final_stage.get('comfy_logical_free_bytes')) - _mib(room.get('free_target_bytes'))):.1f}MiB"
                    )

                if timings:
                    timing_order = [
                        "raw_probe", "comfy_probe", "cache_flush_pre-cooperative",
                        "cooperative_free_memory", "cooperative_retry_free_memory",
                        "aimdo_transient_cleanup", "aimdo_retry_free_memory",
                        "exclusive_free_memory", "final_sync",
                    ]
                    timing_parts = [f"{name}={float(timings.get(name) or 0):.3f}s" for name in timing_order if name in timings]
                    if timing_parts:
                        self._log("PERF GPU stages: " + " • ".join(timing_parts))

                if unloaded_models or stages:
                    before_models = ((stages.get("before") or {}).get("loaded_models") or {})
                    final_models = ((stages.get("after_final_sync") or stages.get("after_exclusive") or stages.get("after_aimdo_retry") or stages.get("after_aimdo_cleanup") or stages.get("after_cooperative_retry") or stages.get("after_cooperative") or {}).get("loaded_models") or {})
                    model_parts = []
                    if before_models.get("same_device_labels"):
                        model_parts.append("before=" + ",".join(str(x) for x in before_models.get("same_device_labels") or []))
                    for stage_name in ("cooperative", "cooperative_retry", "aimdo_retry", "exclusive"):
                        labels = unloaded_models.get(stage_name) or []
                        if labels:
                            model_parts.append(stage_name + "_unloaded=" + ",".join(str(x) for x in labels))
                    if final_models.get("same_device_labels"):
                        model_parts.append("after=" + ",".join(str(x) for x in final_models.get("same_device_labels") or []))
                    if model_parts:
                        self._log("PERF GPU models: " + " • ".join(model_parts))

                headroom_sources = room.get("headroom_sources") or {}
                if headroom_sources:
                    self._log(
                        "PERF GPU policy inputs: "
                        f"llama_margin={_mib(headroom_sources.get('llama_default_margin_bytes')):.1f}MiB • "
                        f"comfy_reserved={_mib(headroom_sources.get('comfy_reserved_bytes')):.1f}MiB • "
                        f"dynamic_headroom={_mib(headroom_sources.get('dynamic_vram_extra_headroom_bytes')):.1f}MiB • "
                        f"configured_comfy={_mib(headroom_sources.get('configured_comfy_headroom_bytes')):.1f}MiB • "
                        f"request_granularity={_mib(room.get('request_granularity_bytes')):.1f}MiB • "
                        f"aimdo={bool(room.get('aimdo', False))}"
                    )

                estimate_components = preload.get("estimate_components") or {}
                if estimate_components:
                    self._log(
                        "PERF VRAM estimate: "
                        f"weights={_mib(estimate_components.get('weights_bytes')):.1f}MiB • "
                        f"kv={_mib(estimate_components.get('kv_cache_bytes')):.1f}MiB • "
                        f"compute={_mib(estimate_components.get('compute_batch_bytes')):.1f}MiB • "
                        f"vision={_mib(estimate_components.get('vision_bytes')):.1f}MiB • "
                        f"speculative={_mib(estimate_components.get('speculative_bytes')):.1f}MiB • "
                        f"total={_mib(estimate_components.get('total_bytes')):.1f}MiB • "
                        f"ctx={int(estimate_components.get('context_size') or 0)} • "
                        f"batch={int(estimate_components.get('n_batch') or 0)}/{int(estimate_components.get('n_ubatch') or 0)} • "
                        f"gpu_layers={int(estimate_components.get('gpu_layers') or 0)}"
                    )

                observed_delta = gpu_backend.get("vram_delta_mib_by_gpu") or {}
                observed_before = gpu_backend.get("vram_free_before_mib_by_gpu") or {}
                observed_after = gpu_backend.get("vram_free_after_mib_by_gpu") or {}
                if observed_delta or observed_before or observed_after:
                    gpu_keys = sorted(set(observed_delta) | set(observed_before) | set(observed_after), key=lambda x: int(x) if str(x).isdigit() else str(x))
                    gpu_parts = []
                    for key in gpu_keys:
                        gpu_parts.append(
                            f"gpu{key}[before={float(observed_before.get(key) or 0):.1f},"
                            f"after={float(observed_after.get(key) or 0):.1f},"
                            f"delta={float(observed_delta.get(key) or 0):.1f}]"
                        )
                    estimate_mib = float(preload.get("estimated_vram_bytes") or 0) / (1024 * 1024)
                    observed_total_mib = float(gpu_backend.get("total_vram_delta_mib") or 0)
                    ratio = (observed_total_mib / estimate_mib) if estimate_mib > 0 else 0.0
                    self._log(
                        "PERF native VRAM observed: " + " ".join(gpu_parts) +
                        f" • observed_total={observed_total_mib:.1f}MiB • estimate_ratio={ratio:.3f}"
                    )

                self._log(
                    "PERF GGUF load I/O: "
                    f"mode={gpu_backend.get('load_mode', 'n/a')} • "
                    f"native={float(gpu_backend.get('native_load_seconds') or 0):.3f}s • "
                    f"major_faults={int(gpu_backend.get('major_faults_delta') or 0)} • "
                    f"minor_faults={int(gpu_backend.get('minor_faults_delta') or 0)} • "
                    f"block_inputs={int(gpu_backend.get('block_inputs_delta') or 0)} • "
                    f"cache_hint={gpu_backend.get('page_cache_hint', 'n/a')} • "
                    f"page_cache={float(gpu_backend.get('page_cache_after_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"WSL={bool(gpu_backend.get('wsl', False))}"
                )
                self._log(f"Request complete in {elapsed:.2f}s" + (f" at {speed} tok/s" if speed is not None else ""))
                return result
        except LocalLLMInterrupted as e:
            with self._state_lock:
                if self._state != STATE_STOPPING:
                    self._state = STATE_READY if self._api is not None else STATE_STOPPED
                self._error = None
                self._clear_current_request_locked(phase=True, metrics=True)
            self._log(f"LLM request interrupted: {e}")
            raise
        except Exception as e:
            with self._state_lock:
                # A request error should not mark the whole service unusable if the
                # model is still resident. Keep READY when possible.
                self._state = STATE_READY if self._api is not None and self._api.is_loaded() else STATE_ERROR
                self._error = f"{type(e).__name__}: {e}"
                self._clear_current_request_locked(metrics=True)
            self._log(self._error, "error")
            log.exception("[Local LLM Server] Request failed")
            raise
        finally:
            if queue_registration_pending:
                with self._state_lock:
                    self._queue_count = max(0, self._queue_count - 1)
            self._emit()

    def vram_estimate(self, patch=None):
        """Return a live, non-mutating estimate for the selected memory settings.

        `patch` may contain unsaved modal values.  The same component estimator
        used by Auto-Yield first-load planning is used here so UI estimates and
        actual handoff targets share one source of truth.
        """
        cfg_saved = self.get_config()
        cfg = copy.deepcopy(cfg_saved)
        if isinstance(patch, dict):
            # The saved config already contains the complete supported schema;
            # avoid rebuilding LocalGGUFLLM.INPUT_TYPES/model lists on every live
            # estimator refresh.
            allowed = set(cfg_saved)
            cfg.update({k: copy.deepcopy(v) for k, v in patch.items() if k in allowed})
        cfg["context_size"] = _normalize_context_size(cfg.get("context_size"), cfg.get("model"))

        model = str(cfg.get("model") or "")
        if not model or model == "No GGUF models found":
            return {"available": False, "reason": "No GGUF model selected"}

        md = _metadata_for(model)
        family = detect_family(md, model)
        caps = capabilities_for_family(family)
        model_path = _full_path(model)

        vision_choice = str(cfg.get("vision_model") or _NONE)
        resolved_vision = None
        if caps.get("vision") is not False:
            if vision_choice == _AUTO_VISION:
                resolved_vision = _find_matching_mmproj(model)
            elif vision_choice != _NONE:
                resolved_vision = vision_choice
        mmproj_path = _full_path(resolved_vision) if resolved_vision else None

        try:
            gpu_layers = int(cfg.get("gpu_layers", -1))
        except Exception:
            gpu_layers = -1
        try:
            main_gpu_index = _gpu_index(cfg.get("main_gpu", 0))
        except Exception:
            main_gpu_index = 0

        speculative_runtime = _speculative_runtime_support()
        speculative = _resolve_service_speculative(cfg, md, speculative_runtime)
        components = _estimate_native_vram_components(
            model_path=model_path,
            mmproj_path=mmproj_path,
            metadata=md,
            gpu_layers=gpu_layers,
            n_ctx=int(cfg.get("context_size") or 0),
            kv_k=str(cfg.get("kv_cache_k") or "f16"),
            kv_v=str(cfg.get("kv_cache_v") or "f16"),
            kv_location=str(cfg.get("kv_cache_location") or "GPU"),
            n_batch=int(cfg.get("prompt_batch_size") or 2048),
            n_ubatch=int(cfg.get("memory_batch_size") or 512),
            flash_attention=bool(cfg.get("flash_attention")),
            speculative_mode=str(speculative.get("effective") or "Off"),
        )

        raw_free = None
        total_vram = None
        torch_allocated = None
        torch_reserved = None
        if gpu_layers != 0:
            try:
                import torch
                if torch.cuda.is_available() and main_gpu_index < int(torch.cuda.device_count()):
                    with torch.cuda.device(main_gpu_index):
                        free_b, total_b = torch.cuda.mem_get_info()
                        raw_free = int(free_b)
                        total_vram = int(total_b)
                        torch_allocated = int(torch.cuda.memory_allocated(main_gpu_index))
                        torch_reserved = int(torch.cuda.memory_reserved(main_gpu_index))
            except Exception:
                pass

        # A measured native allocation is valid for display only when the saved
        # load-affecting settings still match the unsaved modal values.  Otherwise
        # keep it visible as prior information but do not use it to calibrate the
        # proposed configuration.
        match_fields = set(LOAD_FIELDS) - {"memory_preset"}
        saved_settings_match = all(cfg.get(k) == cfg_saved.get(k) for k in match_fields)
        resident_ctl = None
        diagnostics = {}
        resident_now = False
        with _MODEL_LOCK:
            resident_ctl = _MODEL_CACHE.get("managed_adapter")
            diagnostics = copy.deepcopy(_MODEL_CACHE.get("load_diagnostics") or {})
            resident_now = bool(_MODEL_CACHE.get("llm") is not None or (resident_ctl is not None and getattr(resident_ctl, "llm", None) is not None))

        measured = 0
        measured_source = None
        if resident_ctl is not None:
            try:
                measured = int(resident_ctl.observed_vram_bytes or 0)
            except Exception:
                measured = 0
            if measured > 0:
                measured_source = "measured native allocation"
        if measured <= 0:
            try:
                per_gpu = diagnostics.get("vram_delta_bytes_by_gpu") or {}
                measured = int(per_gpu.get(str(main_gpu_index), 0) or 0)
            except Exception:
                measured = 0
            if measured <= 0:
                try:
                    measured = int(float(diagnostics.get("total_vram_delta_mib") or 0) * 1024 * 1024)
                except Exception:
                    measured = 0
            if measured > 0:
                measured_source = "verified load delta"

        measured_applies = bool(measured > 0 and saved_settings_match)
        try:
            last_info = copy.deepcopy((self._last_result or {}).get("info") or {})
            measured_includes_vision = bool(last_info.get("vision_active"))
        except Exception:
            measured_includes_vision = False
        base_total = int(components["base_total_bytes"])
        total_with_vision = int(components["total_bytes"])
        vision_bytes = int(components["vision_bytes"])
        reload_target, reload_target_source = _reload_vram_target_bytes(
            base_total, measured if measured_applies else 0
        )
        try:
            import comfy.model_management as mm
            import comfy.memory_management as cmm
            lease_plan = GPUMemoryLeaseManager(
                mm, aimdo_enabled=bool(getattr(cmm, "aimdo_enabled", False))
            ).plan(reload_target).as_dict()
        except Exception:
            lease_plan = {
                "runtime_target_bytes": int(reload_target),
                "headroom_bytes": 1024 * 1024 * 1024,
                "free_target_bytes": int(reload_target + 1024 * 1024 * 1024),
                "request_target_bytes": int(reload_target + 1024 * 1024 * 1024),
            }
        reload_lease_free_target = int(lease_plan.get("free_target_bytes") or reload_target)
        try:
            vision_plan = GPUMemoryLeaseManager(
                mm, aimdo_enabled=bool(getattr(cmm, "aimdo_enabled", False))
            ).plan(total_with_vision).as_dict()
            vision_lease_free_target = int(vision_plan.get("free_target_bytes") or total_with_vision)
        except Exception:
            vision_lease_free_target = int(total_with_vision + 1024 * 1024 * 1024)

        # Current free VRAM already excludes a resident LLM.  If the same saved
        # configuration is resident, its projected base headroom is therefore the
        # current free value.  A configuration change/vision activation first
        # closes that context, so credit its measured allocation back before
        # subtracting the new estimate.
        release_credit = int(measured if resident_now and measured > 0 else 0)
        if raw_free is None:
            base_headroom = None
            vision_headroom = None
        elif resident_now and saved_settings_match:
            base_headroom = int(raw_free)
            vision_headroom = int(raw_free + release_credit - total_with_vision) if vision_bytes > 0 else int(raw_free)
        else:
            available_after_old_close = int(raw_free + release_credit)
            base_headroom = int(available_after_old_close - reload_lease_free_target)
            vision_headroom = int(available_after_old_close - vision_lease_free_target)

        headroom = vision_headroom if vision_bytes > 0 else base_headroom
        warning = None
        if str(cfg.get("speculative_mode") or "Off") in {"N-gram", "MTP"} and speculative.get("effective") == "Off":
            warning = str(speculative.get("reason") or "Selected speculative-decoding mode is unavailable.")
        elif gpu_layers == 0:
            warning = "GPU layers is 0; native model weights are configured for CPU."
        elif headroom is not None and headroom < 0:
            warning = "Current free VRAM is below the estimated requirement; Auto Yield may need to evict ComfyUI models before loading."
        elif headroom is not None and headroom < 1024 * 1024 * 1024:
            warning = "Estimated VRAM headroom is below 1 GiB."

        return {
            "available": True,
            "model": model,
            "family": family,
            "gpu_index": int(main_gpu_index),
            "split_mode": cfg.get("split_mode"),
            "components": components,
            "resolved_vision": resolved_vision,
            "vision_optional": bool(vision_bytes > 0),
            "speculative": speculative,
            "speculative_support": speculative_runtime,
            "raw_free_bytes": raw_free,
            "total_vram_bytes": total_vram,
            "torch_allocated_bytes": torch_allocated,
            "torch_reserved_bytes": torch_reserved,
            "torch_reclaimable_bytes": (max(0, int(torch_reserved - torch_allocated)) if torch_reserved is not None and torch_allocated is not None else None),
            "measured_bytes": int(measured),
            "measured_source": measured_source,
            "measured_applies": measured_applies,
            "measured_includes_vision": bool(measured_includes_vision),
            "resident": bool(resident_now),
            "saved_settings_match": bool(saved_settings_match),
            "reload_target_bytes": int(reload_target),
            "reload_target_source": reload_target_source,
            "reload_lease_free_target_bytes": int(reload_lease_free_target),
            "reload_lease_headroom_bytes": int(lease_plan.get("headroom_bytes") or 0),
            "reload_lease_request_target_bytes": int(lease_plan.get("request_target_bytes") or reload_lease_free_target),
            "projected_base_headroom_bytes": base_headroom,
            "projected_vision_headroom_bytes": vision_headroom,
            "projected_headroom_bytes": headroom,
            "warning": warning,
            "note": "Vision/mmproj memory is included only in the with-vision total and is allocated when a vision request activates the projector.",
        }

    # ------------------------------------------------------------------
    # Performance tuner
    # ------------------------------------------------------------------

    @staticmethod
    def _tuner_prompt():
        # Deliberately fixed and text-only so every candidate receives the same
        # prompt-evaluation workload.  The content is moderately repetitive to
        # represent prompt-enhancement / structured-writing use without being a
        # synthetic single-token loop that would unfairly favor N-gram decoding.
        blocks = [
            "A visual prompt should identify the primary subject, the important physical details, the environment, the action, the composition, the camera relationship, the lighting, and any constraints that must remain unchanged. Prefer concrete descriptions over vague quality adjectives. Preserve the requested subject and action while resolving ambiguity with specific but neutral details.",
            "When revising text, keep names, quantities, ordering, and explicit restrictions intact. Improve clarity by grouping related details and removing accidental repetition. Do not invent a new scene or change the requested outcome. Describe spatial relationships, materials, motion, and lighting only when they help make the instruction easier to execute consistently.",
            "For a short generation request, useful structure is subject first, then action, environment, framing, lighting, and finishing details. Camera language should be physically coherent. Motion should have a clear direction and cause. If the source request is simple, the improved version should remain concise rather than expanding into unrelated cinematic language.",
            "The benchmark text is intentionally stable across trials. It contains natural prose, repeated concepts, punctuation, and several sentence lengths so prompt processing is representative of ordinary local-LLM workflow requests rather than a tiny microbenchmark. The response itself is used only for timing and is discarded after each measured run.",
        ]
        text = "\n\n".join(blocks * 3)
        return [{"role": "user", "content": text + "\n\nSummarize the practical rules above as a compact checklist using complete sentences."}]

    def _set_tuner(self, **patch):
        with self._tuner_lock:
            self._tuner.update(copy.deepcopy(patch))
        self._emit()

    def tuner_status(self):
        with self._tuner_lock:
            data = copy.deepcopy(self._tuner)
            thread = self._tuner_thread
        data["running"] = bool(thread is not None and thread.is_alive())
        data["cancel_requested"] = bool(self._tuner_cancel.is_set())
        return data

    @staticmethod
    def _tuner_patch_label(patch):
        names = {
            "prompt_batch_size": "batch", "memory_batch_size": "ubatch",
            "flash_attention": "flash", "speculative_mode": "spec",
            "ngram_pred_tokens": "ngram-draft", "mtp_draft_tokens": "mtp-draft",
            "use_mmap": "mmap", "use_mlock": "mlock", "kv_cache_location": "kv-location",
            "gpu_layers": "gpu-layers", "kv_cache_k": "kv-k", "kv_cache_v": "kv-v",
        }
        order = (
            "prompt_batch_size", "memory_batch_size", "flash_attention",
            "use_mmap", "use_mlock", "kv_cache_location", "gpu_layers",
            "kv_cache_k", "kv_cache_v", "speculative_mode",
            "ngram_pred_tokens", "mtp_draft_tokens",
        )
        parts = []
        for key in order:
            if key not in patch:
                continue
            value = patch[key]
            if isinstance(value, bool):
                value = "on" if value else "off"
            parts.append(f"{names.get(key, key)}={value}")
        return "Baseline" if not parts else " • ".join(parts)

    def _tuner_unload_current(self, reason="tuner-cycle"):
        """Yield the current native context and return full wall-clock teardown timing.

        The tuner intentionally models the same lightweight close path used by
        Suspend/Auto-Yield, including the device synchronization that manual
        Suspend performs before close.  Heavy Stop/Unload cleanup is not used.
        """
        ctl = _MODEL_CACHE.get("managed_adapter")
        if ctl is None or getattr(ctl, "llm", None) is None:
            return {
                "total_seconds": 0.0, "sync_seconds": 0.0, "close_seconds": 0.0,
                "released_accounted_bytes": 0,
            }
        started = time.perf_counter()
        sync_seconds = 0.0
        load_device = getattr(ctl, "load_device", None)
        if getattr(load_device, "type", None) == "cuda":
            sync_started = time.perf_counter()
            _sync_cuda_device(load_device)
            sync_seconds = time.perf_counter() - sync_started
        released = int(ctl._unload_native(reason=reason, heavy_cleanup=False) or 0)
        wall = time.perf_counter() - started
        diag = copy.deepcopy(getattr(ctl, "_last_unload_diagnostics", {}) or {})
        return {
            "total_seconds": float(wall),
            "sync_seconds": float(sync_seconds),
            "close_seconds": float(diag.get("close_seconds") or 0.0),
            "native_unload_seconds": float(diag.get("total_seconds") or 0.0),
            "gc_seconds": float(diag.get("gc_seconds") or 0.0),
            "soft_empty_cache_seconds": float(diag.get("soft_empty_cache_seconds") or 0.0),
            "released_accounted_bytes": released,
        }

    @staticmethod
    def _tuner_fit_screen(candidate_cfg, estimate, safety_headroom_bytes):
        """Return the tuner's non-mutating physical-fit screen.

        This intentionally does *not* use live projected headroom.  Live headroom
        answers "would it fit without reclaiming anything right now?" whereas a
        tuner candidate is loaded through the normal GPU lease manager, which may
        reclaim ComfyUI/DynamicVRAM residency first.

        For a single GPU we can prove only one useful pre-load impossibility:
        estimated native runtime + required free-space guard > physical VRAM.
        For multi-GPU split modes, aggregate GGUF estimates cannot prove per-device
        placement, so the real loader is the authoritative admission check.
        """
        estimate = estimate or {}
        components = estimate.get("components") or {}
        runtime_bytes = int(estimate.get("reload_target_bytes") or components.get("base_total_bytes") or 0)
        loader_headroom_bytes = int(estimate.get("reload_lease_headroom_bytes") or 0)
        guard_bytes = max(int(safety_headroom_bytes or 0), loader_headroom_bytes)
        total_vram = estimate.get("total_vram_bytes")
        total_vram = None if total_vram is None else int(total_vram)
        split_mode = str((candidate_cfg or {}).get("split_mode") or "None (single GPU)")
        current_projected = estimate.get("projected_headroom_bytes")
        current_projected = None if current_projected is None else int(current_projected)

        physical_headroom = None
        required_bytes = None
        if total_vram is not None and runtime_bytes > 0:
            physical_headroom = total_vram - runtime_bytes
            required_bytes = runtime_bytes + guard_bytes

        if split_mode != "None (single GPU)":
            basis = "runtime-loader-authoritative-multi-gpu"
            allowed = True
        elif total_vram is None or runtime_bytes <= 0:
            basis = "runtime-loader-authoritative-no-capacity-proof"
            allowed = True
        else:
            basis = "single-gpu-physical-capacity"
            allowed = required_bytes <= total_vram

        reason = None
        if not allowed:
            reason = (
                f"Estimated native runtime {runtime_bytes/(1024**3):.2f} GiB plus "
                f"{guard_bytes/(1024**3):.2f} GiB required free headroom exceeds "
                f"the GPU's {total_vram/(1024**3):.2f} GiB physical VRAM."
            )

        return {
            "allowed": bool(allowed),
            "basis": basis,
            "reason": reason,
            "runtime_bytes": int(runtime_bytes),
            "loader_headroom_bytes": int(loader_headroom_bytes),
            "guard_bytes": int(guard_bytes),
            "required_bytes": (None if required_bytes is None else int(required_bytes)),
            "total_vram_bytes": total_vram,
            "physical_headroom_bytes": (None if physical_headroom is None else int(physical_headroom)),
            "live_projected_headroom_bytes": current_projected,
        }

    @staticmethod
    def _tuner_trial_summary(records):
        """Aggregate repeated full-cycle tuner trials with robust stability metrics.

        Screening can use one short trial, but sustained validation uses several
        load -> inference -> unload cycles.  The recommendation is ranked by a
        robust upper score (median + scaled MAD), so a setting that is fast once
        but periodically stalls does not beat a consistently fast setting.
        """
        rows = [dict(r) for r in (records or []) if isinstance(r, dict) and float(r.get("score_seconds") or 0) > 0]
        if not rows:
            return {}

        def vals(key, positive=False):
            out=[]
            for row in rows:
                try:
                    v=float(row.get(key))
                except Exception:
                    continue
                if positive and v <= 0:
                    continue
                out.append(v)
            return out

        def med(key, positive=False):
            v=vals(key, positive)
            return float(statistics.median(v)) if v else 0.0

        scores=vals("score_seconds", True)
        score_med=float(statistics.median(scores)) if scores else 0.0
        score_mad=float(statistics.median([abs(x-score_med) for x in scores])) if len(scores)>1 else 0.0
        # 1.4826 scales MAD to a standard-deviation-equivalent measure. For a
        # small validation set a single severe stall can still leave MAD small,
        # so the upper-quartile score is also part of the conservative rank.
        if len(scores) > 1:
            score_q75=float(statistics.quantiles(scores, n=4, method="inclusive")[2])
        else:
            score_q75=score_med
        robust=float(max(score_med + 1.4826*score_mad, score_q75))
        decode=vals("decode_tps", True)
        load=vals("load_seconds")
        native=vals("native_load_seconds")
        block=[int(max(0, r.get("block_inputs", 0) or 0)) for r in rows]
        major=[int(max(0, r.get("major_faults", 0) or 0)) for r in rows]
        storage=sum(1 for r in rows if int(r.get("block_inputs", 0) or 0)>0 or int(r.get("major_faults", 0) or 0)>0)
        pc_before=next((r.get("page_cache_before_bytes") for r in rows if r.get("page_cache_before_bytes") is not None), None)
        pc_after=next((r.get("page_cache_after_bytes") for r in reversed(rows) if r.get("page_cache_after_bytes") is not None), None)
        mem_before=next((r.get("mem_available_before_bytes") for r in rows if r.get("mem_available_before_bytes") is not None), None)
        mem_after=next((r.get("mem_available_after_bytes") for r in reversed(rows) if r.get("mem_available_after_bytes") is not None), None)
        return {
            "trial_count": len(rows),
            "score_median_seconds": score_med,
            "score_mad_seconds": score_mad,
            "score_q75_seconds": score_q75,
            "robust_score_seconds": robust,
            "score_worst_seconds": max(scores) if scores else 0.0,
            "score_best_seconds": min(scores) if scores else 0.0,
            "score_dispersion_ratio": ((1.4826*score_mad/score_med) if score_med>0 else 0.0),
            "prompt_tps_median": med("prompt_tps", True),
            "decode_tps_median": med("decode_tps", True),
            "decode_tps_floor": min(decode) if decode else 0.0,
            "decode_tps_ceiling": max(decode) if decode else 0.0,
            "prompt_seconds_median": med("prompt_seconds"),
            "decode_seconds_median": med("decode_seconds"),
            "fixed_work_seconds_median": med("fixed_work_seconds"),
            "load_seconds_median": med("load_seconds"),
            "load_seconds_worst": max(load) if load else 0.0,
            "native_load_seconds_median": med("native_load_seconds"),
            "native_load_seconds_worst": max(native) if native else 0.0,
            "unload_seconds_median": med("unload_seconds"),
            "unload_sync_seconds_median": med("unload_sync_seconds"),
            "cycle_fixed_seconds_median": med("cycle_fixed_seconds"),
            "cycle_wall_seconds_median": med("cycle_wall_seconds"),
            "storage_backed_trials": int(storage),
            "block_inputs_total": int(sum(block)),
            "block_inputs_worst": int(max(block) if block else 0),
            "major_faults_total": int(sum(major)),
            "page_cache_delta_bytes": (None if pc_before is None or pc_after is None else int(pc_after)-int(pc_before)),
            "mem_available_delta_bytes": (None if mem_before is None or mem_after is None else int(mem_after)-int(mem_before)),
        }

    @classmethod
    def _tuner_merge_validation_results(cls, first, second, *, label="Validation • Baseline"):
        """Combine baseline validation bracketing the finalists.

        Measuring baseline both before and after finalists controls for gradual
        host/page-cache/thermal drift.  The merged baseline is the reference used
        by final selection; it is not an additional native benchmark run.
        """
        rows=[]
        for item in (first, second):
            rows.extend(copy.deepcopy((item or {}).get("trial_records") or []))
        summary=cls._tuner_trial_summary(rows)
        merged=copy.deepcopy(first or second or {})
        merged.update({
            "status": "ok" if rows else "error",
            "label": label,
            "validation": True,
            "validation_role": "baseline-combined",
            "trial_records": rows,
            "trials": int(summary.get("trial_count") or 0),
        })
        mapping={
            "prompt_tps":"prompt_tps_median", "decode_tps":"decode_tps_median",
            "prompt_seconds":"prompt_seconds_median", "decode_seconds":"decode_seconds_median",
            "fixed_work_seconds":"fixed_work_seconds_median", "load_seconds":"load_seconds_median",
            "native_load_seconds":"native_load_seconds_median", "unload_seconds":"unload_seconds_median",
            "unload_sync_seconds":"unload_sync_seconds_median", "cycle_fixed_seconds":"cycle_fixed_seconds_median",
            "cycle_wall_seconds":"cycle_wall_seconds_median", "score_seconds":"score_median_seconds",
        }
        for dst,src in mapping.items():
            if src in summary: merged[dst]=summary[src]
        merged.update(summary)
        return merged

    def _tuner_candidate(self, base_cfg, patch, *, output_tokens, trials, safety_headroom_bytes, score_mode):
        if self._tuner_cancel.is_set():
            raise InterruptedError("Performance tuner cancelled")

        candidate_cfg = copy.deepcopy(base_cfg)
        candidate_cfg.update(patch)
        estimate = self.vram_estimate(patch)
        headroom = estimate.get("projected_headroom_bytes")
        label = self._tuner_patch_label({k: candidate_cfg.get(k) for k in patch})
        fit = self._tuner_fit_screen(candidate_cfg, estimate, safety_headroom_bytes)

        if patch and not fit["allowed"]:
            return {
                "status": "skipped", "label": label, "patch": copy.deepcopy(patch),
                "reason": str(fit.get("reason") or "Candidate cannot physically fit on the selected GPU."),
                "estimated_vram_bytes": int((estimate.get("components") or {}).get("base_total_bytes") or 0),
                "projected_headroom_bytes": (None if headroom is None else int(headroom)),
                "projected_physical_headroom_bytes": fit.get("physical_headroom_bytes"),
                "tuner_fit_guard_bytes": int(fit.get("guard_bytes") or 0),
                "tuner_fit_required_bytes": fit.get("required_bytes"),
                "tuner_fit_basis": fit.get("basis"),
            }

        live_projected = fit.get("live_projected_headroom_bytes")
        if patch and live_projected is not None and int(live_projected) < int(safety_headroom_bytes):
            physical = fit.get("physical_headroom_bytes")
            physical_text = "unknown" if physical is None else f"{int(physical)/(1024**3):.2f}GiB"
            self._log(
                "TUNER fit gate: "
                f"{label} • live_projected={int(live_projected)/(1024**3):.2f}GiB is reclaimable • "
                f"physical_after_runtime={physical_text} • guard={int(fit.get('guard_bytes') or 0)/(1024**3):.2f}GiB • "
                f"basis={fit.get('basis')} • decision=benchmark"
            )

        physical_headroom = fit.get("physical_headroom_bytes")
        fit_guard_bytes = fit.get("guard_bytes")
        fit_required_bytes = fit.get("required_bytes")
        split_mode = str(candidate_cfg.get("split_mode") or "None (single GPU)")
        mode = "Inference Only" if str(score_mode).lower().startswith("inference") else "ComfyUI Cycle"

        args = self._call_args()
        args.update(patch)
        args.update({
            "messages_override": self._tuner_prompt(),
            "prompt_cache_mode": "Off",
            "max_tokens": int(output_tokens),
            "seed": 8675309,
            "progress_callback": None,
            "token_callback": None,
            "cancel_event": self._tuner_cancel,
        })

        # Prime backend/JIT and ordinary page-cache state once. Every measured
        # trial still begins yielded and includes a real native reload.
        warm_args = dict(args)
        warm_args["max_tokens"] = min(16, int(output_tokens))
        warm_started = time.perf_counter()
        _response, _thinking, info_json, _tokens, api = LocalGGUFLLM().generate(**warm_args)
        warm_wall = time.perf_counter() - warm_started
        self._api = api
        try:
            warm_info = json.loads(info_json)
        except Exception:
            warm_info = {}
        warm_unload = self._tuner_unload_current("tuner-warmup-yield")

        prompt_rates=[]; decode_rates=[]; prompt_seconds=[]; decode_seconds=[]
        load_seconds=[]; native_load_seconds=[]; unload_seconds=[]; unload_sync_seconds=[]
        completion_tokens=[]; inference_seconds=[]; cycle_wall_seconds=[]; acceptance_rates=[]
        trial_records=[]
        measured_vram=0
        last_info={}

        for trial in range(max(1, int(trials))):
            if self._tuner_cancel.is_set():
                raise InterruptedError("Performance tuner cancelled")
            cycle_started=time.perf_counter()
            info={}; tokens=0; generate_wall=0.0
            p_rate=d_rate=p_sec=d_sec=load_sec=native_sec=0.0
            p_tok=c_tok=0
            gpu={}
            unload={}
            try:
                started=time.perf_counter()
                _response, _thinking, info_json, tokens, api = LocalGGUFLLM().generate(**args)
                generate_wall=time.perf_counter()-started
                self._api=api
                try: info=json.loads(info_json)
                except Exception: info={}
                last_info=info

                p_rate=float(info.get("prompt_tokens_per_second") or 0.0)
                d_rate=float(info.get("tokens_per_second") or 0.0)
                p_sec=float(info.get("prompt_eval_seconds") or 0.0)
                d_sec=float(info.get("generation_seconds") or 0.0)
                p_tok=int(info.get("prompt_eval_tokens", info.get("prompt_tokens")) or 0)
                c_tok=int(info.get("completion_tokens") or tokens or 0)
                if p_sec<=0 and p_rate>0 and p_tok>0: p_sec=p_tok/p_rate
                if d_sec<=0 and d_rate>0 and c_tok>0: d_sec=c_tok/d_rate

                load_sec=float(info.get("load_seconds") or 0.0)
                gpu=info.get("gpu_backend") or {}
                native_sec=float(gpu.get("native_load_seconds") or 0.0)
                if load_sec<=0:
                    load_sec=max(0.0, generate_wall-max(0.0,p_sec)-max(0.0,d_sec))

                if p_rate>0: prompt_rates.append(p_rate)
                if d_rate>0: decode_rates.append(d_rate)
                if p_sec>=0: prompt_seconds.append(p_sec)
                if d_sec>=0: decode_seconds.append(d_sec)
                load_seconds.append(load_sec); native_load_seconds.append(native_sec)
                completion_tokens.append(c_tok)
                inference_seconds.append((p_sec+d_sec) if (p_sec>0 or d_sec>0) else max(0.0,generate_wall-load_sec))
                spec_stats=((info.get("speculative") or {}).get("stats") or {})
                rate=spec_stats.get("acceptance_rate")
                if isinstance(rate,(int,float)): acceptance_rates.append(float(rate))
                measured_vram=max(measured_vram,int(gpu.get("observed_vram_bytes") or 0))
            finally:
                unload=self._tuner_unload_current(f"tuner-trial-{trial + 1}-yield")
                u=float(unload.get("total_seconds") or 0.0)
                us=float(unload.get("sync_seconds") or 0.0)
                unload_seconds.append(u); unload_sync_seconds.append(us)
                cycle_wall=time.perf_counter()-cycle_started
                cycle_wall_seconds.append(cycle_wall)

            fixed=p_sec
            if d_rate>0:
                fixed += float(output_tokens)/d_rate
            elif d_sec>0:
                fixed += d_sec
            else:
                fixed=max(0.0,generate_wall-load_sec)
            cycle_fixed=float(load_sec+fixed+u)
            trial_score=float(fixed if mode=="Inference Only" else cycle_fixed)
            trial_records.append({
                "trial": int(trial+1), "output_tokens": int(output_tokens),
                "prompt_tps": float(p_rate), "decode_tps": float(d_rate),
                "prompt_seconds": float(p_sec), "decode_seconds": float(d_sec),
                "fixed_work_seconds": float(fixed), "load_seconds": float(load_sec),
                "native_load_seconds": float(native_sec), "unload_seconds": float(u),
                "unload_sync_seconds": float(us), "cycle_fixed_seconds": cycle_fixed,
                "cycle_wall_seconds": float(cycle_wall), "score_seconds": trial_score,
                "major_faults": int(gpu.get("major_faults_delta") or 0),
                "minor_faults": int(gpu.get("minor_faults_delta") or 0),
                "block_inputs": int(gpu.get("block_inputs_delta") or 0),
                "page_cache_before_bytes": gpu.get("page_cache_before_bytes"),
                "page_cache_after_bytes": gpu.get("page_cache_after_bytes"),
                "mem_available_before_bytes": gpu.get("mem_available_before_bytes"),
                "mem_available_after_bytes": gpu.get("mem_available_after_bytes"),
                "page_cache_hint": gpu.get("page_cache_hint"),
            })

        summary=self._tuner_trial_summary(trial_records)
        prompt_rate=float(summary.get("prompt_tps_median") or (statistics.median(prompt_rates) if prompt_rates else 0.0))
        decode_rate=float(summary.get("decode_tps_median") or (statistics.median(decode_rates) if decode_rates else 0.0))
        prompt_sec=float(summary.get("prompt_seconds_median") or (statistics.median(prompt_seconds) if prompt_seconds else 0.0))
        decode_sec=float(summary.get("decode_seconds_median") or (statistics.median(decode_seconds) if decode_seconds else 0.0))
        load_sec=float(summary.get("load_seconds_median") or (statistics.median(load_seconds) if load_seconds else 0.0))
        native_load_sec=float(summary.get("native_load_seconds_median") or (statistics.median(native_load_seconds) if native_load_seconds else 0.0))
        unload_sec=float(summary.get("unload_seconds_median") or (statistics.median(unload_seconds) if unload_seconds else 0.0))
        unload_sync_sec=float(summary.get("unload_sync_seconds_median") or (statistics.median(unload_sync_seconds) if unload_sync_seconds else 0.0))
        fixed_work_seconds=float(summary.get("fixed_work_seconds_median") or 1e9)
        cycle_fixed_seconds=float(summary.get("cycle_fixed_seconds_median") or (load_sec+fixed_work_seconds+unload_sec))
        score_seconds=float(summary.get("score_median_seconds") or (fixed_work_seconds if mode=="Inference Only" else cycle_fixed_seconds))

        tradeoffs=[]; quality_tradeoff=False
        if candidate_cfg.get("kv_cache_k") != base_cfg.get("kv_cache_k") or candidate_cfg.get("kv_cache_v") != base_cfg.get("kv_cache_v"):
            quality_tradeoff=True
            tradeoffs.append("KV precision changed; lower-bit KV formats can slightly affect output quality.")
        if bool(candidate_cfg.get("use_mlock")) and not bool(base_cfg.get("use_mlock")):
            tradeoffs.append("mlock pins mapped model pages in system RAM and increases RAM pressure.")
        if not bool(candidate_cfg.get("use_mmap", True)):
            tradeoffs.append("mmap is disabled; repeated reloads copy/read model pages instead of using normal mapped-page reuse.")
        if str(candidate_cfg.get("kv_cache_location")) != str(base_cfg.get("kv_cache_location")):
            tradeoffs.append("KV cache placement changed; CPU placement saves VRAM but can reduce throughput.")
        if int(candidate_cfg.get("gpu_layers",-1)) != int(base_cfg.get("gpu_layers",-1)):
            tradeoffs.append("GPU-layer offload changed; partial offload trades VRAM for CPU/PCIe work.")

        spec=last_info.get("speculative") or {}
        result={
            "status":"ok", "label":label, "patch":copy.deepcopy(patch),
            "prompt_tps":prompt_rate, "decode_tps":decode_rate,
            "prompt_seconds":prompt_sec, "decode_seconds":decode_sec,
            "fixed_work_seconds":fixed_work_seconds,
            "load_seconds":load_sec, "native_load_seconds":native_load_sec,
            "unload_seconds":unload_sec, "unload_sync_seconds":unload_sync_sec,
            "cycle_fixed_seconds":cycle_fixed_seconds,
            "cycle_wall_seconds":float(summary.get("cycle_wall_seconds_median") or (statistics.median(cycle_wall_seconds) if cycle_wall_seconds else 0.0)),
            "score_mode":mode, "score_seconds":score_seconds,
            "median_completion_tokens":int(statistics.median(completion_tokens)) if completion_tokens else 0,
            "warmup_load_seconds":float(warm_info.get("load_seconds") or 0.0),
            "warmup_native_load_seconds":float((warm_info.get("gpu_backend") or {}).get("native_load_seconds") or 0.0),
            "warmup_wall_seconds":float(warm_wall),
            "warmup_unload_seconds":float(warm_unload.get("total_seconds") or 0.0),
            "measured_vram_bytes":int(measured_vram),
            "estimated_vram_bytes":int((estimate.get("components") or {}).get("base_total_bytes") or 0),
            "projected_headroom_bytes":(None if headroom is None else int(headroom)),
            "projected_physical_headroom_bytes":(None if physical_headroom is None else int(physical_headroom)),
            "tuner_fit_guard_bytes":(None if fit_guard_bytes is None else int(fit_guard_bytes)),
            "tuner_fit_required_bytes":(None if fit_required_bytes is None else int(fit_required_bytes)),
            "tuner_fit_basis":("single-gpu-physical-capacity" if split_mode=="None (single GPU)" else "runtime-loader-authoritative"),
            "speculative_effective":str(spec.get("effective") or "Off"),
            "speculative_implementation":spec.get("implementation"),
            "acceptance_rate":(float(statistics.median(acceptance_rates)) if acceptance_rates else None),
            "tradeoffs":tradeoffs, "quality_tradeoff":bool(quality_tradeoff),
            "trials":max(1,int(trials)), "trial_records":trial_records,
        }
        result.update(summary)
        return result

    @staticmethod
    def _tuner_choose(current, candidates, minimum_gain=0.01):
        valid = [x for x in candidates if x and x.get("status") == "ok" and float(x.get("score_seconds") or 0) > 0]
        if not valid:
            return current
        best = min(valid, key=lambda x: float(x.get("score_seconds") or 1e99))
        if current is None or current.get("status") != "ok":
            return best
        cur = float(current.get("score_seconds") or 1e99)
        new = float(best.get("score_seconds") or 1e99)
        # Tiny differences are usually run-to-run noise. Keep the simpler/current
        # setting unless the candidate clears a small measurable threshold.
        return best if new < cur * (1.0 - float(minimum_gain)) else current

    def _run_tuner(self, options, original_resident, original_yielded):
        base_cfg = self.get_config()
        profile = str(options.get("profile") or "Quick").title()
        standard = profile == "Standard"
        # Stage 1 is deliberately cheap screening. Stage 2 re-tests only the
        # strongest complete configurations with longer, repeated full cycles.
        # This prevents a single short burst from becoming the recommendation.
        screening_trials = 2 if standard else 1
        screening_tokens = 96 if standard else 64
        validation_trials = 6 if standard else 4
        validation_tokens = 256 if standard else 192
        validation_baseline_half = 3 if standard else 2
        validation_finalists = 3 if standard else 2
        safety_mib = max(256, int(options.get("safety_headroom_mib") or 1024))
        safety_bytes = safety_mib * 1024 * 1024
        score_mode = "Inference Only" if str(options.get("score_mode") or "").lower().startswith("inference") else "ComfyUI Cycle"
        tune_batches = bool(options.get("tune_batches", True))
        tune_flash = bool(options.get("tune_flash_attention", True))
        tune_spec = bool(options.get("tune_speculative", True))
        tune_memory = bool(options.get("tune_memory", True))
        tune_kv_precision = bool(options.get("tune_kv_precision", False))
        results = []
        seen = set()
        planned = 1
        if tune_batches:
            planned += 8 if standard else 5
        if tune_flash:
            planned += 1
        if tune_memory:
            planned += 7 if standard else 5
        if tune_kv_precision:
            planned += 5 if standard else 3
        if tune_spec:
            planned += 5 if standard else 2
        # Validation baseline before + finalists + baseline after. The combined
        # baseline summary is derived from those trials and does not run again.
        planned += 2 + validation_finalists

        def patch_key(patch):
            return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in patch.items()))

        def run_candidate(patch, phase):
            key = patch_key(patch)
            if key in seen:
                for item in results:
                    if patch_key(item.get("patch") or {}) == key:
                        return item
            seen.add(key)
            idx = len(results) + 1
            self._set_tuner(
                phase=phase, candidate_index=idx, candidate_total=max(planned, idx),
                progress=min(0.96, max(0.02, idx / max(planned, 1))),
                current_candidate=self._tuner_patch_label(patch),
                message=f"Benchmarking {self._tuner_patch_label(patch)}",
                results=results,
            )
            try:
                item = self._tuner_candidate(
                    base_cfg, patch, output_tokens=screening_tokens, trials=screening_trials,
                    safety_headroom_bytes=safety_bytes, score_mode=score_mode,
                )
            except InterruptedError:
                raise
            except Exception as e:
                item = {
                    "status": "error", "label": self._tuner_patch_label(patch),
                    "patch": copy.deepcopy(patch), "reason": f"{type(e).__name__}: {e}",
                }
            results.append(item)
            self._set_tuner(results=results)
            if item.get("status") == "ok":
                self._log(
                    "TUNER candidate: "
                    f"{item['label']} • load={item['load_seconds']:.3f}s • prompt={item['prompt_tps']:.1f}t/s • "
                    f"decode={item['decode_tps']:.2f}t/s • unload={item['unload_seconds']:.3f}s • "
                    f"inference={item['fixed_work_seconds']:.3f}s • cycle={item['cycle_fixed_seconds']:.3f}s • "
                    f"score({score_mode})={item['score_seconds']:.3f}s • VRAM={item['measured_vram_bytes']/(1024**3):.2f}GiB"
                )
            elif item.get("status") == "skipped":
                self._log(f"TUNER skipped: {item['label']} • {item.get('reason')}", "warning")
            else:
                self._log(f"TUNER candidate failed: {item['label']} • {item.get('reason')}", "warning")
            return item

        def run_validation(patch, phase, *, role, trial_count):
            idx = len(results) + 1
            base_label = self._tuner_patch_label(patch)
            display_label = f"Validation • {base_label}"
            self._set_tuner(
                phase=phase, candidate_index=idx, candidate_total=max(planned, idx),
                progress=min(0.985, max(0.02, idx / max(planned, 1))),
                current_candidate=display_label,
                message=f"Sustained validation: {base_label} ({trial_count} cycles × {validation_tokens} tokens)",
                results=results,
            )
            try:
                item = self._tuner_candidate(
                    base_cfg, patch, output_tokens=validation_tokens, trials=trial_count,
                    safety_headroom_bytes=safety_bytes, score_mode=score_mode,
                )
            except InterruptedError:
                raise
            except Exception as e:
                item = {
                    "status": "error", "label": display_label,
                    "patch": copy.deepcopy(patch), "reason": f"{type(e).__name__}: {e}",
                }
            item["screen_label"] = item.get("label") or base_label
            item["label"] = display_label
            item["validation"] = True
            item["validation_role"] = role
            item["validation_output_tokens"] = int(validation_tokens)
            results.append(item)
            self._set_tuner(results=results)
            if item.get("status") == "ok":
                self._log(
                    "TUNER validation: "
                    f"{base_label} • cycles={item.get('trials', 0)} • "
                    f"median={float(item.get('score_seconds') or 0):.3f}s • "
                    f"robust={float(item.get('robust_score_seconds') or item.get('score_seconds') or 0):.3f}s • "
                    f"decode_median={float(item.get('decode_tps') or 0):.2f}t/s • "
                    f"decode_floor={float(item.get('decode_tps_floor') or 0):.2f}t/s • "
                    f"load_worst={float(item.get('load_seconds_worst') or item.get('load_seconds') or 0):.3f}s • "
                    f"storage_trials={int(item.get('storage_backed_trials') or 0)}/{int(item.get('trials') or 0)}"
                )
            else:
                self._log(f"TUNER validation failed: {base_label} • {item.get('reason')}", "warning")
            return item

        try:
            with self._generation_slot(
                workflow_owned=False,
                cancel_event=self._tuner_cancel,
                reason="running the Local LLM performance tuner",
            ):
                with self._state_lock:
                    self._state = STATE_TUNING
                    self._error = None
                self._emit()
                self._log(
                    f"Performance tuner started ({profile}, optimize={score_mode}, "
                    f"screen={screening_trials}x{screening_tokens}tok, "
                    f"validate={validation_trials}x{validation_tokens}tok, finalists={validation_finalists}, "
                    f"safety={safety_mib}MiB, memory={tune_memory}, kv_precision={tune_kv_precision})"
                )

                baseline = run_candidate({}, "Baseline full cycle")
                if baseline.get("status") != "ok":
                    raise RuntimeError("Baseline benchmark failed; tuner cannot compare candidates")
                current = baseline
                best_patch = {}

                if tune_batches:
                    base_batch = max(1, int(base_cfg.get("prompt_batch_size") or 2048))
                    base_ubatch = max(1, int(base_cfg.get("memory_batch_size") or 512))
                    ub_values = [base_ubatch, 512, 1024] if not standard else [base_ubatch, 256, 512, 1024, 2048]
                    ub_ceiling = min(int(base_cfg.get("context_size") or 32768), max(base_batch, 2048 if standard else 1024))
                    ub_values = sorted(set(v for v in ub_values if 64 <= v <= max(64, ub_ceiling)))
                    group = [current]
                    for ub in ub_values:
                        p = dict(best_patch)
                        p["memory_batch_size"] = int(ub)
                        p["prompt_batch_size"] = max(base_batch, int(ub))
                        group.append(run_candidate(p, "Tune micro-batch"))
                    current = self._tuner_choose(current, group)
                    best_patch = dict(current.get("patch") or {})

                    selected_ub = int(best_patch.get("memory_batch_size", base_ubatch))
                    context = max(selected_ub, int(base_cfg.get("context_size") or 32768))
                    batch_values = [base_batch, 2048, 4096] if not standard else [base_batch, 512, 1024, 2048, 4096, 8192]
                    batch_values = sorted(set(v for v in batch_values if selected_ub <= v <= context))
                    group = [current]
                    for batch in batch_values:
                        p = dict(best_patch)
                        p["prompt_batch_size"] = int(batch)
                        group.append(run_candidate(p, "Tune prompt batch"))
                    current = self._tuner_choose(current, group)
                    best_patch = dict(current.get("patch") or {})

                if tune_flash:
                    current_flash = bool(best_patch.get("flash_attention", base_cfg.get("flash_attention", True)))
                    opposite = dict(best_patch)
                    opposite["flash_attention"] = not current_flash
                    tested = run_candidate(opposite, "Tune Flash Attention")
                    current = self._tuner_choose(current, [current, tested])
                    best_patch = dict(current.get("patch") or {})

                if tune_memory:
                    # mmap affects load/reload behavior; mlock can improve page
                    # residency at the cost of system-RAM pressure. Both are tested
                    # independently around the current winner.
                    for key, phase in (("use_mmap", "Tune mmap"), ("use_mlock", "Tune mlock")):
                        now = bool(best_patch.get(key, base_cfg.get(key)))
                        p = dict(best_patch)
                        p[key] = not now
                        tested = run_candidate(p, phase)
                        current = self._tuner_choose(current, [current, tested])
                        best_patch = dict(current.get("patch") or {})

                    # GPU vs CPU KV placement can substantially change VRAM and
                    # prompt/decode performance without changing KV precision.
                    now_loc = str(best_patch.get("kv_cache_location", base_cfg.get("kv_cache_location") or "GPU"))
                    other_loc = "CPU" if now_loc.upper() == "GPU" else "GPU"
                    p = dict(best_patch)
                    p["kv_cache_location"] = other_loc
                    tested = run_candidate(p, "Tune KV placement")
                    current = self._tuner_choose(current, [current, tested])
                    best_patch = dict(current.get("patch") or {})

                    # Explore useful partial-offload points, but never force pure
                    # CPU inference. Context size stays fixed so memory tuning
                    # cannot win by silently reducing configured capacity.
                    est = self.vram_estimate(best_patch)
                    blocks = int((est.get("components") or {}).get("block_count") or 0)
                    cur_layers = int(best_patch.get("gpu_layers", base_cfg.get("gpu_layers", -1)))
                    effective_spec = str(current.get("speculative_effective") or "Off")
                    if blocks > 4 and effective_spec != "MTP":
                        layer_values = {cur_layers, -1, max(1, blocks - max(2, blocks // 8))}
                        if standard:
                            layer_values.add(max(1, int(round(blocks * 0.75))))
                        group = [current]
                        for layers in sorted(layer_values):
                            if layers == cur_layers:
                                continue
                            p = dict(best_patch)
                            p["gpu_layers"] = int(layers)
                            group.append(run_candidate(p, "Tune GPU layer offload"))
                        current = self._tuner_choose(current, group)
                        best_patch = dict(current.get("patch") or {})

                if tune_kv_precision:
                    # KV quantization can alter output slightly, so it is a
                    # separately opt-in search rather than part of normal memory
                    # tuning. Symmetric q8/q4/f16 cover the useful broad tradeoffs;
                    # Standard also tests q8 keys + q4 values.
                    cur_k = str(best_patch.get("kv_cache_k", base_cfg.get("kv_cache_k") or "q8_0"))
                    cur_v = str(best_patch.get("kv_cache_v", base_cfg.get("kv_cache_v") or "q8_0"))
                    pairs = {(cur_k, cur_v), ("q8_0", "q8_0"), ("q4_0", "q4_0"), ("f16", "f16")}
                    if standard:
                        pairs.add(("q8_0", "q4_0"))
                    group = [current]
                    for k_type, v_type in sorted(pairs):
                        if (k_type, v_type) == (cur_k, cur_v):
                            continue
                        p = dict(best_patch)
                        p["kv_cache_k"] = k_type
                        p["kv_cache_v"] = v_type
                        group.append(run_candidate(p, "Tune KV precision"))
                    current = self._tuner_choose(current, group)
                    best_patch = dict(current.get("patch") or {})

                if tune_spec:
                    md = _metadata_for(str(base_cfg.get("model") or ""))
                    support = _speculative_runtime_support()
                    modes = ["Off"]
                    if support.get("ngram"):
                        modes.append("N-gram")
                    # Give native MTP a fair chance even if memory tuning selected
                    # partial GPU offload: MTP itself requires full GPU offload, so
                    # probe and benchmark that combined valid configuration.
                    mtp_cfg = copy.deepcopy(base_cfg)
                    mtp_cfg.update(best_patch)
                    mtp_cfg["speculative_mode"] = "MTP"
                    mtp_cfg["gpu_layers"] = -1
                    mtp_resolved = _resolve_service_speculative(mtp_cfg, md, support)
                    if mtp_resolved.get("effective") == "MTP":
                        modes.append("MTP")
                    group = [current]
                    for mode in modes:
                        p = dict(best_patch)
                        p["speculative_mode"] = mode
                        if mode == "MTP":
                            p["gpu_layers"] = -1
                        group.append(run_candidate(p, "Tune speculative decoding"))
                    current = self._tuner_choose(current, group)
                    best_patch = dict(current.get("patch") or {})

                    if standard and current.get("status") == "ok":
                        effective = str(current.get("speculative_effective") or best_patch.get("speculative_mode") or "Off")
                        if effective == "N-gram":
                            vals = sorted(set([int(base_cfg.get("ngram_pred_tokens") or 10), 5, 10, 16, 24]))
                            group = [current]
                            for value in vals:
                                p = dict(best_patch)
                                p["speculative_mode"] = "N-gram"
                                p["ngram_pred_tokens"] = int(value)
                                group.append(run_candidate(p, "Tune N-gram draft size"))
                            current = self._tuner_choose(current, group)
                            best_patch = dict(current.get("patch") or {})
                        elif effective == "MTP":
                            vals = sorted(set([int(base_cfg.get("mtp_draft_tokens") or 2), 1, 2, 3, 4]))
                            group = [current]
                            for value in vals:
                                p = dict(best_patch)
                                p["speculative_mode"] = "MTP"
                                p["gpu_layers"] = -1
                                p["mtp_draft_tokens"] = int(value)
                                group.append(run_candidate(p, "Tune MTP draft size"))
                            current = self._tuner_choose(current, group)
                            best_patch = dict(current.get("patch") or {})

                # ----------------------------------------------------------
                # Sustained validation
                # ----------------------------------------------------------
                # Short screening is allowed to be noisy. Build a small finalist
                # set from the greedy winner plus the best complete configurations
                # seen anywhere in screening, then benchmark those configurations
                # repeatedly with a much longer decode. Baseline is bracketed
                # before and after the finalists to expose host/page-cache/thermal
                # drift instead of giving the first or last configuration an
                # ordering advantage.
                screen_ok = [r for r in results if r.get("status") == "ok" and not r.get("validation")]
                ranked = sorted(screen_ok, key=lambda r: float(r.get("score_seconds") or 1e99))
                finalist_patches = []
                finalist_keys = set()

                def add_finalist(patch):
                    patch = copy.deepcopy(patch or {})
                    if not patch:
                        return
                    key = patch_key(patch)
                    if key in finalist_keys:
                        return
                    finalist_keys.add(key)
                    finalist_patches.append(patch)

                add_finalist(best_patch)
                for item in ranked:
                    if len(finalist_patches) >= validation_finalists:
                        break
                    add_finalist(item.get("patch") or {})

                baseline_pre = run_validation(
                    {}, "Validate baseline (before finalists)",
                    role="baseline-pre", trial_count=validation_baseline_half,
                )
                if baseline_pre.get("status") != "ok":
                    raise RuntimeError("Sustained baseline validation failed before finalists")

                validated = []
                for patch in finalist_patches:
                    validated.append(run_validation(
                        patch, "Validate finalist", role="finalist",
                        trial_count=validation_trials,
                    ))

                baseline_post = run_validation(
                    {}, "Validate baseline (after finalists)",
                    role="baseline-post", trial_count=validation_baseline_half,
                )
                if baseline_post.get("status") != "ok":
                    raise RuntimeError("Sustained baseline validation failed after finalists")

                validation_baseline = self._tuner_merge_validation_results(
                    baseline_pre, baseline_post, label="Validation • Baseline (combined)"
                )
                results.append(validation_baseline)
                self._set_tuner(results=results)

                baseline_robust = float(validation_baseline.get("robust_score_seconds") or validation_baseline.get("score_seconds") or 0.0)
                valid_finalists = [
                    r for r in validated
                    if r.get("status") == "ok" and float(r.get("robust_score_seconds") or r.get("score_seconds") or 0) > 0
                ]

                chosen = validation_baseline
                if valid_finalists and baseline_robust > 0:
                    # Rank by sustained robust performance. If multiple finalists
                    # are within 1% (our existing noise floor), prefer the one that
                    # produced fewer storage-backed reloads and lower dispersion.
                    best_robust = min(float(r.get("robust_score_seconds") or r.get("score_seconds") or 1e99) for r in valid_finalists)
                    near = [
                        r for r in valid_finalists
                        if float(r.get("robust_score_seconds") or r.get("score_seconds") or 1e99) <= best_robust * 1.01
                    ]
                    finalist_choice = min(
                        near,
                        key=lambda r: (
                            int(r.get("storage_backed_trials") or 0),
                            float(r.get("score_dispersion_ratio") or 0.0),
                            float(r.get("robust_score_seconds") or r.get("score_seconds") or 1e99),
                            int(r.get("measured_vram_bytes") or 0),
                        ),
                    )
                    finalist_robust = float(finalist_choice.get("robust_score_seconds") or finalist_choice.get("score_seconds") or 1e99)
                    # A recommendation must clear the same 1% measurable-gain
                    # floor against the *validated* baseline, not merely screening.
                    if finalist_robust < baseline_robust * 0.99:
                        chosen = finalist_choice

                current = chosen
                best_patch = dict(current.get("patch") or {})
                baseline_score = float(validation_baseline.get("score_seconds") or 0.0)
                best_score = float(current.get("score_seconds") or baseline_score or 1.0)
                best_robust_score = float(current.get("robust_score_seconds") or best_score)
                improvement = max(0.0, (baseline_robust - best_robust_score) / baseline_robust) if baseline_robust > 0 else 0.0

                self._log(
                    "TUNER sustained decision: "
                    f"baseline_robust={baseline_robust:.3f}s • "
                    f"chosen={current.get('label')} • chosen_robust={best_robust_score:.3f}s • "
                    f"gain={improvement*100:.2f}% • patch={best_patch or 'current settings'}"
                )

                recommendation_patch = {}
                final_cfg = copy.deepcopy(base_cfg)
                final_cfg.update(best_patch)
                recommendation_fields = (
                    "prompt_batch_size", "memory_batch_size", "flash_attention",
                    "use_mmap", "use_mlock", "kv_cache_location", "gpu_layers",
                    "kv_cache_k", "kv_cache_v", "speculative_mode",
                    "ngram_pred_tokens", "mtp_draft_tokens",
                )
                for key in recommendation_fields:
                    if final_cfg.get(key) != base_cfg.get(key):
                        recommendation_patch[key] = copy.deepcopy(final_cfg.get(key))
                recommendation = {
                    "patch": recommendation_patch,
                    "settings": {k: copy.deepcopy(final_cfg.get(k)) for k in MEMORY_PRESET_FIELDS if k in final_cfg},
                    "label": current.get("label"),
                    "score_mode": score_mode,
                    "baseline_score_seconds": baseline_score,
                    "best_score_seconds": best_score,
                    "baseline_robust_score_seconds": baseline_robust,
                    "best_robust_score_seconds": best_robust_score,
                    "validated": True,
                    "validation_trials": int(current.get("trials") or 0),
                    "validation_tokens": int(validation_tokens),
                    "decode_tps_floor": current.get("decode_tps_floor"),
                    "load_seconds_worst": current.get("load_seconds_worst"),
                    "score_dispersion_ratio": current.get("score_dispersion_ratio"),
                    "storage_backed_trials": current.get("storage_backed_trials"),
                    "baseline_validation_trials": int(validation_baseline.get("trials") or 0),
                    "baseline_fixed_work_seconds": float(validation_baseline.get("fixed_work_seconds") or 0.0),
                    "best_fixed_work_seconds": float(current.get("fixed_work_seconds") or 0.0),
                    "baseline_cycle_seconds": float(validation_baseline.get("cycle_fixed_seconds") or 0.0),
                    "best_cycle_seconds": float(current.get("cycle_fixed_seconds") or 0.0),
                    "improvement_percent": improvement * 100.0,
                    "load_seconds": current.get("load_seconds"),
                    "unload_seconds": current.get("unload_seconds"),
                    "prompt_tps": current.get("prompt_tps"),
                    "decode_tps": current.get("decode_tps"),
                    "measured_vram_bytes": current.get("measured_vram_bytes"),
                    "speculative_effective": current.get("speculative_effective"),
                    "acceptance_rate": current.get("acceptance_rate"),
                    "tradeoffs": copy.deepcopy(current.get("tradeoffs") or []),
                    "quality_tradeoff": bool(current.get("quality_tradeoff")),
                }
                score_name = "ComfyUI cycle" if score_mode == "ComfyUI Cycle" else "inference"
                self._set_tuner(
                    state="complete", phase="Complete", progress=1.0,
                    current_candidate=None, results=results, recommendation=recommendation,
                    error=None, finished_at=time.time(),
                    message=(
                        f"Complete. Sustained validation selected {self._tuner_patch_label(best_patch)}; "
                        f"robust {score_name} improved {improvement*100:.1f}% over the bracketed baseline."
                    ),
                )
                self._log(
                    f"Performance tuner complete: {improvement*100:.1f}% validated robust {score_name} improvement • "
                    f"validation={int(current.get('trials') or 0)}x{validation_tokens}tok • "
                    f"recommendation={recommendation_patch or 'current settings'}"
                )
        except (InterruptedError, LocalLLMInterrupted):
            self._set_tuner(
                state="cancelled", phase="Cancelled", progress=0.0, current_candidate=None,
                results=results, finished_at=time.time(), message="Cancelled after the current benchmark trial.",
            )
            self._log("Performance tuner cancelled")
        except Exception as e:
            self._set_tuner(
                state="error", phase="Error", current_candidate=None, results=results,
                error=f"{type(e).__name__}: {e}", finished_at=time.time(), message=str(e),
            )
            self._log(f"Performance tuner failed: {type(e).__name__}: {e}", "error")
        finally:
            # Restore the user's saved load signature rather than leaving the last
            # benchmark candidate resident. This does not save any tuner result.
            try:
                if self._stop_pending.is_set():
                    # Global Stop owns final cleanup. Do not start a restore/warmup
                    # after the tuner was cancelled by Stop.
                    return
                if self._api is not None:
                    restore_args = self._call_args()
                    restore_args.update({
                        "messages_override": [{"role": "user", "content": "Respond with OK."}],
                        "max_tokens": 1, "seed": 0, "progress_callback": None, "token_callback": None,
                    })
                    _r, _t, _i, _n, api = LocalGGUFLLM().generate(**restore_args)
                    self._api = api
                    ctl = _MODEL_CACHE.get("managed_adapter")
                    if original_yielded:
                        if ctl is not None and getattr(ctl, "llm", None) is not None:
                            ctl._unload_native(reason="tuner-restore-yielded", heavy_cleanup=False)
                    elif ctl is not None and getattr(ctl, "llm", None) is not None:
                        # Do not leave the tuner's restore/warmup prompt as the next
                        # user's reusable resident prefix. The model remains loaded.
                        reset = getattr(ctl.llm, "reset", None)
                        if callable(reset):
                            reset()
                with self._state_lock:
                    self._state = STATE_READY
                    self._model_loaded = bool(original_resident or original_yielded or self._api is not None)
                    self._error = None
            except Exception as restore_error:
                restore_message = f"Tuner finished but restoring the saved configuration failed: {restore_error}"
                with self._state_lock:
                    self._state = STATE_ERROR
                    self._error = restore_message
                self._set_tuner(state="error", phase="Restore error", error=restore_message, message=restore_message, finished_at=time.time())
                self._log(self._error, "error")
            finally:
                self._emit()

    def start_tuner(self, options=None):
        options = dict(options or {})
        with self._tuner_lock:
            if self._tuner_thread is not None and self._tuner_thread.is_alive():
                raise RuntimeError("Performance tuner is already running")
        with self._state_lock:
            if self._state != STATE_READY or self._api is None:
                raise RuntimeError("Start the Local LLM service before running the performance tuner")
            if self._restart_required:
                raise RuntimeError("Save/reload the current server configuration before benchmarking")
        original_resident = bool(self._resident_snapshot())
        original_yielded = bool(not original_resident and self._api is not None)
        profile = str(options.get("profile") or "Quick").title()
        if profile not in {"Quick", "Standard"}:
            profile = "Quick"
        options["profile"] = profile
        self._tuner_cancel.clear()
        with self._tuner_lock:
            self._tuner = {
                "state": "running", "phase": "Preparing", "progress": 0.01,
                "candidate_index": 0, "candidate_total": 0, "current_candidate": None,
                "results": [], "recommendation": None, "error": None,
                "started_at": time.time(), "finished_at": None, "profile": profile,
                "message": "Preparing benchmark candidates…",
                "options": copy.deepcopy(options),
            }
            thread = threading.Thread(
                target=self._run_tuner,
                args=(copy.deepcopy(options), original_resident, original_yielded),
                name="LocalLLMPerformanceTuner", daemon=True,
            )
            self._tuner_thread = thread
            thread.start()
        return self.tuner_status()

    def cancel_tuner(self):
        with self._tuner_lock:
            running = self._tuner_thread is not None and self._tuner_thread.is_alive()
        if running:
            self._tuner_cancel.set()
            self._set_tuner(message="Cancellation requested; waiting for the current native benchmark call to finish…")
        return self.tuner_status()

    def status(self):
        with self._state_lock:
            state = self._state
            error = self._error
            queue_count = self._queue_count
            requests_total = self._requests_total
            current_client = self._current_client
            current_started = self._current_started
            current_phase_started = self._current_phase_started
            current_completion_tokens = self._current_completion_tokens
            current_tokens_per_second = self._current_tokens_per_second
            current_prompt_tokens = self._current_prompt_tokens
            current_prompt_tokens_per_second = self._current_prompt_tokens_per_second
            model_loaded = self._model_loaded
            last = copy.deepcopy(self._last_result)
            last_seconds = self._last_generation_seconds
            restart_required = self._restart_required
        cfg = self.get_config()
        resident_now = self._resident_snapshot() if state not in {STATE_STOPPED, STATE_ERROR, STATE_LOADING, STATE_RELOADING, STATE_PROCESSING} else bool(model_loaded)
        yielded = bool(state == STATE_READY and self._api is not None and not resident_now)
        info = (last or {}).get("info") or {}
        model = cfg.get("model")
        md = _metadata_for(model) if model and model != "No GGUF models found" else {}
        family = detect_family(md, model or "") if md else "unknown"
        capabilities, speculative_support = _model_capabilities_with_runtime(md, family)
        speculative = _resolve_service_speculative(cfg, md, speculative_support)
        gpu_backend = info.get("gpu_backend") or {}
        preload = gpu_backend.get("preload_memory") or {}
        last_unload = gpu_backend.get("last_unload") or {}
        return {
            "state": state,
            "active": state not in {STATE_STOPPED, STATE_ERROR},
            # Never acquire the native llama model lock here. Status must remain
            # responsive while generation holds that lock so the modal can be
            # opened/updated during inference.
            "model_loaded": bool(resident_now),
            "vram_yielded": yielded,
            "native_operation": _native_operation_snapshot(),
            "restart_required": bool(restart_required),
            "error": error,
            "model": model,
            "vision_model": cfg.get("vision_model"),
            "family": family,
            "capabilities": capabilities,
            "speculative_support": speculative_support,
            "speculative": speculative,
            "speculative_mode": cfg.get("speculative_mode", "Off"),
            "speculative_effective": speculative.get("effective", "Off"),
            "prompt_cache_mode": cfg.get("prompt_cache_mode", "Auto"),
            "model_preset": cfg.get("model_preset"),
            "memory_preset": cfg.get("memory_preset"),
            "vram_policy": cfg.get("vram_policy", "Auto Yield to ComfyUI"),
            "context_size": cfg.get("context_size"),
            "kv_cache_k": cfg.get("kv_cache_k"),
            "kv_cache_v": cfg.get("kv_cache_v"),
            "main_gpu": cfg.get("main_gpu"),
            "queue_count": queue_count,
            "requests_total": requests_total,
            "current_client": current_client,
            "current_seconds": (time.time() - current_started) if current_started else None,
            "current_phase_seconds": (time.time() - current_phase_started) if current_phase_started else None,
            "current_completion_tokens": current_completion_tokens,
            "current_tokens_per_second": (float(current_tokens_per_second or 0.0) if state == STATE_GENERATING else 0.0),
            "current_prompt_tokens": current_prompt_tokens,
            "current_prompt_tokens_per_second": float(current_prompt_tokens_per_second or 0.0),
            "last_generation_seconds": info.get("generation_seconds"),
            "last_prompt_eval_seconds": info.get("prompt_eval_seconds"),
            "last_inference_seconds": info.get("request_generation_seconds"),
            "last_load_seconds": info.get("load_seconds"),
            "last_load_path": gpu_backend.get("load_path"),
            "last_native_load_seconds": gpu_backend.get("native_load_seconds"),
            "last_comfy_load_models_seconds": gpu_backend.get("comfy_load_models_gpu_seconds"),
            "last_vram_handoff_seconds": preload.get("handoff_seconds"),
            "last_yield_close_seconds": last_unload.get("close_seconds"),
            "last_yield_total_seconds": last_unload.get("total_seconds"),
            # Total wall-clock request time intentionally includes an on-demand
            # Auto-Yield reload. Prompt/decode timings below never do.
            "last_request_seconds": last_seconds,
            "last_total_seconds": last_seconds,
            "last_tokens_per_second": info.get("tokens_per_second"),
            "last_average_tokens_per_second": info.get("tokens_per_second"),
            "last_prompt_tokens_per_second": info.get("prompt_tokens_per_second"),
            "last_prompt_tokens": info.get("prompt_eval_tokens", info.get("prompt_tokens")),
            "last_prompt_cache": copy.deepcopy(info.get("prompt_cache") or {}),
            "last_prompt_cache_hit": bool((info.get("prompt_cache") or {}).get("hit")),
            "last_prompt_cache_reused_tokens": (info.get("prompt_cache") or {}).get("reused_tokens"),
            "last_prompt_cache_reuse_percent": (info.get("prompt_cache") or {}).get("reuse_percent"),
            "last_prompt_cache_saved_seconds": (info.get("prompt_cache") or {}).get("estimated_seconds_saved"),
            "last_completion_tokens": info.get("completion_tokens"),
            "last_tokens": info.get("completion_tokens"),
            "last_total_tokens": info.get("total_tokens"),
            "external_api_enabled": bool(cfg.get("external_api_enabled")),
            "startup_mode": cfg.get("startup_mode"),
            "api_key_set": bool(cfg.get("api_key")),
            "buffered_streaming": bool(cfg.get("allow_buffered_streaming")),
            "log_prompt_content": bool(cfg.get("log_prompt_content")),
            "log_response_content": bool(cfg.get("log_response_content")),
            "api_path": "/local-llm/v1",
        }


SERVICE = LocalLLMServiceManager()


def catalog():
    models, vision = _model_lists()
    user_memory = memory_presets()  # legacy files remain readable for old workflows
    complete = complete_settings_presets()
    preset_payload = public_presets()
    preset_payload["memory"].update({name: copy.deepcopy(item["settings"]) for name, item in user_memory.items()})
    return {
        "models": models,
        "vision": vision,
        "gpus": _gpu_choices(),
        "model_presets": ["Auto (Detected)", "Custom"] + list(MODEL_PRESETS.keys()),
        "memory_presets": ["Custom"] + list(MEMORY_PRESETS.keys()) + sorted(user_memory, key=str.lower),
        "memory_preset_files": {name: item.get("path") for name, item in user_memory.items()},
        "settings_presets": {
            "directory": str(SETTINGS_PRESET_DIR),
            "names": ["Custom", *sorted(complete, key=str.lower)],
            "deletable_names": sorted(complete, key=str.lower),
            "presets": {name: copy.deepcopy(item["settings"]) for name, item in complete.items()},
        },
        "context_sizes": list(CONTEXT_SIZE_STEPS),
        "request_presets": SERVICE.request_preset_catalog(),
        "node_presets": SERVICE.node_preset_catalog(),
        "presets": preset_payload,
    }


def model_info(name):
    if not name or name == "No GGUF models found":
        return {
            "metadata": {}, "family": "unknown", "recommended_preset": "Generic Chat",
            "available_presets": ["Generic Chat"], "capabilities": capabilities_for_family("generic"),
            "matching_vision": None,
            "native_context": None,
            "context_sizes": list(CONTEXT_SIZE_STEPS),
        }
    md = _metadata_for(name)
    family = detect_family(md, name)
    context_sizes, native_context = _context_options_for_model(name)
    matching_vision = _find_matching_mmproj(name)
    capabilities, speculative_support = _model_capabilities_with_runtime(md, family)
    speculative = _resolve_service_speculative(SERVICE.get_config(), md, speculative_support)
    return {
        "metadata": {k: (v[:2000] + "…" if isinstance(v, str) and len(v) > 2000 else v) for k, v in md.items()},
        "family": family,
        "recommended_preset": recommended_model_preset(md, name),
        "available_presets": available_model_presets(md, name),
        "capabilities": capabilities,
        "speculative_support": speculative_support,
        "speculative": speculative,
        "matching_vision": matching_vision,
        "matching_vision_validation": (
            _validate_mmproj_pair(name, matching_vision) if matching_vision else None
        ),
        "native_context": native_context,
        "context_sizes": context_sizes,
    }


def maybe_autostart():
    if SERVICE.get_config().get("startup_mode") != "Auto Start":
        return
    def worker():
        # Give ComfyUI its normal startup path a moment to settle before allocating
        # a potentially large native CUDA model.
        time.sleep(1.5)
        try:
            SERVICE.start()
        except Exception:
            pass
    threading.Thread(target=worker, name="LocalLLMAutostart", daemon=True).start()


# ---------------------------------------------------------------------------
# Lightweight ComfyUI nodes backed by the global service
# ---------------------------------------------------------------------------

class LocalLLMSettings:
    """Reusable complete Local LLM runtime settings for workflow generation.

    Complete presets are load-only from the node.  Creation/deletion is owned by
    the server Presets tab so workflows cannot accidentally mutate the shared
    preset library.
    """

    EXPOSED_FIELDS = (
        "model", "vision_model", "model_preset", "thinking_mode", "reasoning_effort",
        "preserve_thinking",
        "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
        "presence_penalty", "frequency_penalty", "max_tokens",
        "vision_max_images", "vision_max_frames", "vision_max_edge",
        "context_size", "kv_cache_k", "kv_cache_v", "kv_cache_location", "gpu_layers",
        "flash_attention", "prompt_batch_size", "memory_batch_size", "use_mmap", "use_mlock",
        "prompt_cache_mode", "speculative_mode", "ngram_pred_tokens", "ngram_size",
        "ngram_mode", "ngram_min_hits", "ngram_max_entries_per_key",
        "ngram_sync_check_tokens", "mtp_draft_tokens", "mtp_p_min",
        "split_mode", "main_gpu", "tensor_split", "vram_policy",
    )

    @classmethod
    def INPUT_TYPES(cls):
        source = LocalGGUFLLM.INPUT_TYPES()["required"]
        required = {
            "settings_preset": (["Current Server", "Custom", *complete_settings_preset_names()], {
                "default": "Current Server",
                "tooltip": "Load the current Local LLM server settings or a saved Settings Preset. Editing any field switches this node to Custom.",
            }),
        }
        for key in cls.EXPOSED_FIELDS:
            if key == "vram_policy":
                required[key] = (["Auto Yield to ComfyUI", "Keep Resident"], {"default": "Auto Yield to ComfyUI"})
            else:
                required[key] = copy.deepcopy(source[key])
        return {"required": required}

    RETURN_TYPES = ("LOCAL_LLM_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = "LLM/Local Service"
    DESCRIPTION = "Reusable model, generation, vision, and memory/performance settings for Local LLM Generate and Prompt Enhancer. Seed remains request-local to the consuming node."

    @classmethod
    def IS_CHANGED(cls, settings_preset="Current Server", **kwargs):
        name = str(settings_preset or "Current Server")
        payload = {"settings_preset": name, "values": {k: kwargs.get(k) for k in cls.EXPOSED_FIELDS if k in kwargs}}
        if name == "Current Server":
            current = SERVICE.get_config()
            payload["current_server_settings"] = {k: copy.deepcopy(current.get(k)) for k in cls.EXPOSED_FIELDS if k in current}
        elif name != "Custom":
            item = complete_settings_presets().get(name)
            payload["preset_settings"] = copy.deepcopy((item or {}).get("settings"))
            payload["preset_mtime_ns"] = (item or {}).get("mtime_ns")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def build(self, settings_preset="Current Server", **kwargs):
        name = str(settings_preset or "Current Server")
        values = {k: copy.deepcopy(kwargs.get(k)) for k in self.EXPOSED_FIELDS if k in kwargs}
        if name == "Current Server":
            current = SERVICE.get_config()
            values.update({k: copy.deepcopy(current.get(k)) for k in self.EXPOSED_FIELDS if k in current})
        elif name != "Custom":
            saved = load_complete_settings_preset(name)
            if saved is not None:
                # Saved preset is authoritative while selected.  Preserve any
                # future preset fields not yet surfaced as node widgets too.
                values.update(copy.deepcopy(saved))
        clean = _validate_complete_settings(values)
        return ({
            "schema": "local_llm_complete_request_settings",
            "schema_version": 3,
            "settings_preset": name,
            "sampling_mode": "Custom",
            "server_config": clean,
            **{k: copy.deepcopy(clean.get(k, kwargs.get(k))) for k in SAMPLER_PRESET_FIELDS},
            "vision_max_images": int(clean.get("vision_max_images", kwargs.get("vision_max_images", 4))),
            "vision_max_frames": int(clean.get("vision_max_frames", kwargs.get("vision_max_frames", 24))),
            "vision_max_edge": int(clean.get("vision_max_edge", kwargs.get("vision_max_edge", 1536))),
        },)


class LocalLLMGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt_preset": (["Custom", *text_preset_names("system_prompts")], {"default": "Custom", "tooltip": "Reusable system prompts from models/LLM/local_LLM_presets/system_prompts. Editing the text switches this selector to Custom."}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True, "dynamicPrompts": False}),
                "prompt_preset": (["Custom", *text_preset_names("prompts")], {"default": "Custom", "tooltip": "Reusable prompts from models/LLM/local_LLM_presets/prompts. Editing the text switches this selector to Custom."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Standard ComfyUI seed for this Generate node. Control After Generate supports fixed, increment, decrement, or randomize. Seed is request-local and is never supplied or overridden by Local LLM Settings."}),
            },
            "optional": {
                "settings": ("LOCAL_LLM_SETTINGS", {"tooltip": "Optional Local LLM Settings node. When connected, it supplies model/runtime, sampler, and vision configuration. Seed remains owned by this Generate node. When disconnected, the current Local LLM server configuration is used."}),
                "image": ("IMAGE", {"tooltip": "One still image or an IMAGE batch."}),
                "video_frames": ("IMAGE", {"tooltip": "Ordered video frames as an IMAGE batch; sampled evenly according to the active Local LLM settings."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "thinking", "info")
    FUNCTION = "generate"
    CATEGORY = "LLM/Local Service"
    DESCRIPTION = "Generate through the persistent Local LLM Server. LLM/runtime settings come from an optional Local LLM Settings node (or the current server configuration); seed remains a per-Generate-node control."

    @classmethod
    def IS_CHANGED(cls, system_prompt_preset="Custom", prompt_preset="Custom", **kwargs):
        # Generation has no local model/sampler widgets, but it deliberately owns
        # its per-request seed. Cache invalidation follows the connected Settings
        # payload (or live server config) plus this node's seed.
        linked_settings = kwargs.get("settings") if isinstance(kwargs.get("settings"), dict) else None
        payload = {"settings_node": bool(linked_settings), "seed": int(kwargs.get("seed", 0))}
        if linked_settings:
            payload["settings"] = copy.deepcopy(linked_settings)
        else:
            cfg = SERVICE.get_config()
            payload["server"] = {
                key: copy.deepcopy(cfg.get(key))
                for key in LocalLLMSettings.EXPOSED_FIELDS
                if key in cfg
            }
            payload["generation_defaults"] = SERVICE.request_generation_defaults()

        for kind, selected in (("system_prompts", system_prompt_preset), ("prompts", prompt_preset)):
            name = str(selected or "Custom")
            payload[kind] = {"name": name}
            if name != "Custom":
                item = text_presets(kind).get(name)
                payload[kind]["text"] = None if item is None else item.get("text", "")
                payload[kind]["mtime_ns"] = None if item is None else item.get("mtime_ns")

        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def generate(self, system_prompt_preset, system_prompt, prompt_preset, prompt, seed,
                 image=None, video_frames=None, settings=None):
        linked_settings = settings if isinstance(settings, dict) else None
        runtime_config = None
        overrides = {}

        if linked_settings:
            server_patch = linked_settings.get("server_config")
            # Current Server follows the live modal/service configuration. Custom
            # and named Settings presets provide a request-local runtime snapshot.
            if (
                str(linked_settings.get("settings_preset") or "Current Server") != "Current Server"
                and isinstance(server_patch, dict)
                and server_patch
            ):
                runtime_config = copy.deepcopy(server_patch)

            for key in (*SAMPLER_PRESET_FIELDS, "vision_max_images", "vision_max_frames", "vision_max_edge"):
                if key in linked_settings:
                    overrides[key] = copy.deepcopy(linked_settings.get(key))

        # Seed is intentionally request-local to Generate. This keeps standard
        # ComfyUI Control After Generate behavior available even when a Settings
        # node owns the LLM/runtime configuration.
        overrides["seed"] = int(seed)

        effective_system_prompt = system_prompt
        if str(system_prompt_preset or "Custom") != "Custom":
            saved = load_text_preset("system_prompts", system_prompt_preset)
            if saved is not None:
                effective_system_prompt = saved

        effective_prompt = prompt
        if str(prompt_preset or "Custom") != "Custom":
            saved = load_text_preset("prompts", prompt_preset)
            if saved is not None:
                effective_prompt = saved

        messages = []
        if effective_system_prompt:
            messages.append({"role": "system", "content": str(effective_system_prompt)})
        messages.append({"role": "user", "content": "" if effective_prompt is None else str(effective_prompt)})

        result = SERVICE.generate_messages(
            messages,
            image=image,
            video_frames=video_frames,
            client="ComfyUI Local LLM Generate",
            overrides=overrides,
            runtime_config=runtime_config,
        )
        info = copy.deepcopy(result.get("info") or {})
        info["request_presets"] = {
            "system_prompt": str(system_prompt_preset or "Custom"),
            "prompt": str(prompt_preset or "Custom"),
            "complete_settings": str((linked_settings or {}).get("settings_preset") or "Current Server"),
            "settings_node": bool(linked_settings),
        }
        return (
            result.get("response", ""),
            result.get("thinking", ""),
            json.dumps(info, indent=2),
        )


# ---------------------------------------------------------------------------
# HTTP routes: management + OpenAI-compatible API
# ---------------------------------------------------------------------------

try:
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    def _json_error(message, status=400):
        return web.json_response({"error": {"message": str(message), "type": "local_llm_server_error"}}, status=status)

    _SSE_HEADERS = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    def _sse_response():
        return web.StreamResponse(status=200, headers=dict(_SSE_HEADERS))

    def _token_queue_callback(loop, token_queue):
        def on_token(event):
            try:
                loop.call_soon_threadsafe(token_queue.put_nowait, dict(event or {}))
            except Exception:
                pass
        return on_token

    async def _finish_sse(stream):
        await stream.write(b"data: [DONE]\n\n")
        try:
            await stream.write_eof()
        except Exception:
            pass

    def _authorized(request):
        cfg = SERVICE.get_config()
        if not cfg.get("external_api_enabled"):
            return False, "External Local LLM API is disabled"
        expected = str(cfg.get("api_key") or "")
        if not expected:
            return True, None
        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {expected}":
            return True, None
        return False, "Invalid or missing API key"

    @routes.get("/local_llm_server/status")
    async def local_llm_server_status(request):
        return web.json_response(SERVICE.status())

    @routes.get("/local_llm_server/config")
    async def local_llm_server_config_get(request):
        return web.json_response(SERVICE.public_config())

    @routes.put("/local_llm_server/config")
    async def local_llm_server_config_put(request):
        try:
            data = await request.json()
            # update_config emits a ComfyUI websocket status event via send_sync.
            # Running it directly on aiohttp's event-loop thread can stall the
            # HTTP response even though the file/log write already completed.
            updated = await asyncio.to_thread(SERVICE.update_config, data)
            return web.json_response({"config": updated, "status": SERVICE.status()})
        except Exception as e:
            return _json_error(e)

    @routes.get("/local_llm_server/catalog")
    async def local_llm_server_catalog(request):
        return web.json_response(catalog())

    @routes.get("/local_llm_server/model_info")
    async def local_llm_server_model_info(request):
        try:
            return web.json_response(model_info(request.query.get("model", "")))
        except Exception as e:
            return _json_error(e)

    @routes.post("/local_llm_server/vram_estimate")
    async def local_llm_server_vram_estimate(request):
        try:
            data = await request.json()
            estimate = await asyncio.to_thread(SERVICE.vram_estimate, data if isinstance(data, dict) else {})
            return web.json_response(estimate)
        except Exception as e:
            return _json_error(e)

    @routes.get("/local_llm_server/tuner/status")
    async def local_llm_server_tuner_status(request):
        return web.json_response(SERVICE.tuner_status())

    @routes.post("/local_llm_server/tuner/start")
    async def local_llm_server_tuner_start(request):
        try:
            body = await request.json()
            status = await asyncio.to_thread(SERVICE.start_tuner, body if isinstance(body, dict) else {})
            return web.json_response(status)
        except Exception as e:
            return _json_error(e, 409 if "already running" in str(e).lower() else 400)

    @routes.post("/local_llm_server/tuner/cancel")
    async def local_llm_server_tuner_cancel(request):
        try:
            return web.json_response(await asyncio.to_thread(SERVICE.cancel_tuner))
        except Exception as e:
            return _json_error(e)

    @routes.get("/local_llm_server/settings_presets")
    async def local_llm_server_settings_presets_get(request):
        return web.json_response(catalog().get("settings_presets") or {})

    @routes.post("/local_llm_server/settings_presets")
    async def local_llm_server_settings_presets_post(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            source = body.get("settings")
            if source is None:
                source = SERVICE.get_config()
            saved = await asyncio.to_thread(save_complete_settings_preset, body.get("name"), source)
            SERVICE._log(f"Saved Local LLM complete settings preset: {saved['name']}")
            return web.json_response({"saved": saved, "catalog": catalog()})
        except Exception as e:
            return _json_error(e)

    @routes.delete("/local_llm_server/settings_presets")
    async def local_llm_server_settings_presets_delete(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            deleted = await asyncio.to_thread(delete_complete_settings_preset, body.get("name"))
            SERVICE._log(f"Deleted Local LLM complete settings preset: {deleted['name']}")
            return web.json_response({"deleted": deleted, "catalog": catalog()})
        except Exception as e:
            return _json_error(e)

    # Legacy endpoint retained for older workflows/frontends; the current UI no
    # longer exposes memory-only presets.
    @routes.post("/local_llm_server/memory_presets")
    async def local_llm_server_memory_presets_post(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            saved = await asyncio.to_thread(save_memory_preset, body.get("name"), body.get("settings") or {})
            return web.json_response({"saved": saved, "catalog": catalog()})
        except Exception as e:
            return _json_error(e)

    @routes.get("/local_llm_server/logs")
    async def local_llm_server_logs(request):
        return web.json_response({"logs": SERVICE.logs()})

    @routes.get("/local_llm_server/request_presets")
    async def local_llm_server_request_presets_get(request):
        return web.json_response(SERVICE.request_preset_catalog())

    @routes.post("/local_llm_server/request_presets")
    async def local_llm_server_request_presets_post(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            saved = await asyncio.to_thread(
                save_request_preset,
                body.get("name"),
                body.get("settings") or {},
            )
            SERVICE._log(f"Saved Local LLM request preset: {saved['name']}")
            return web.json_response({
                "saved": saved,
                "catalog": SERVICE.request_preset_catalog(),
            })
        except Exception as e:
            return _json_error(e)

    @routes.get("/local_llm_server/node_presets")
    async def local_llm_server_node_presets_get(request):
        return web.json_response(SERVICE.node_preset_catalog())

    @routes.post("/local_llm_server/node_presets")
    async def local_llm_server_node_presets_post(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            kind = str(body.get("kind") or "").strip()
            if kind == "sampler":
                saved = await asyncio.to_thread(
                    save_sampler_preset, body.get("name"), body.get("settings") or {}
                )
                label = "sampler"
            elif kind in {"system_prompts", "prompts"}:
                saved = await asyncio.to_thread(
                    save_text_preset, kind, body.get("name"), body.get("text", "")
                )
                label = "system prompt" if kind == "system_prompts" else "prompt"
            else:
                raise ValueError("kind must be sampler, system_prompts, or prompts")
            SERVICE._log(f"Saved Local LLM {label} preset: {saved['name']}")
            return web.json_response({
                "saved": saved,
                "catalog": SERVICE.node_preset_catalog(),
            })
        except Exception as e:
            return _json_error(e)

    @routes.delete("/local_llm_server/node_presets")
    async def local_llm_server_node_presets_delete(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("request body must be an object")
            kind = str(body.get("kind") or "").strip()
            if kind == "sampler":
                deleted = await asyncio.to_thread(delete_sampler_preset, body.get("name"))
                label = "sampler"
            elif kind in {"system_prompts", "prompts"}:
                deleted = await asyncio.to_thread(delete_text_preset, kind, body.get("name"))
                label = "system prompt" if kind == "system_prompts" else "prompt"
            else:
                raise ValueError("kind must be sampler, system_prompts, or prompts")
            SERVICE._log(f"Deleted Local LLM {label} preset: {deleted['name']}")
            return web.json_response({"deleted": deleted, "catalog": SERVICE.node_preset_catalog()})
        except Exception as e:
            return _json_error(e)


    @routes.post("/local_llm_server/start")
    async def local_llm_server_start(request):
        try:
            status = await asyncio.to_thread(SERVICE.start)
            return web.json_response(status)
        except LocalLLMInterrupted:
            return web.json_response(SERVICE.status())
        except Exception as e:
            return _json_error(e, 500)

    @routes.post("/local_llm_server/suspend")
    async def local_llm_server_suspend(request):
        try:
            status = await asyncio.to_thread(SERVICE.suspend)
            return web.json_response(status)
        except Exception as e:
            return _json_error(e, 500)

    @routes.post("/local_llm_server/stop")
    async def local_llm_server_stop(request):
        try:
            status = await asyncio.to_thread(SERVICE.stop)
            return web.json_response(status)
        except Exception as e:
            return _json_error(e, 500)

    @routes.post("/local_llm_server/reload")
    async def local_llm_server_reload(request):
        try:
            status = await asyncio.to_thread(SERVICE.reload)
            return web.json_response(status)
        except LocalLLMInterrupted:
            return web.json_response(SERVICE.status())
        except Exception as e:
            return _json_error(e, 500)

    @routes.post("/local_llm_server/api_key/regenerate")
    async def local_llm_server_api_key(request):
        try:
            key = await asyncio.to_thread(SERVICE.regenerate_api_key)
            return web.json_response({"api_key": key})
        except Exception as e:
            return _json_error(e)

    @routes.get("/local-llm/v1/models")
    async def local_llm_openai_models(request):
        ok, reason = _authorized(request)
        if not ok:
            return _json_error(reason, 401 if SERVICE.get_config().get("api_key") else 403)
        cfg = SERVICE.get_config()
        model = str(cfg.get("model") or "local-llm")
        configured_ctx = int(_normalize_context_size(cfg.get("context_size"), model) or 0)
        try:
            md = _metadata_for(model) if model and model != "No GGUF models found" else {}
        except Exception:
            md = {}
        native_ctx = _native_context_length(md or {})
        # OpenAI does not standardize a context-length field on /v1/models.
        # Advertise the common local-server spellings while retaining a strict
        # OpenAI-compatible core; clients that do not recognize them ignore them.
        item = {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "local-llm-server",
            "context_length": configured_ctx,
            "max_context_length": configured_ctx,
            "n_ctx": configured_ctx,
            "meta": {
                "context_length": configured_ctx,
                "n_ctx": configured_ctx,
                "n_ctx_train": int(native_ctx) if native_ctx else None,
            },
        }
        if native_ctx:
            item["n_ctx_train"] = int(native_ctx)
        return web.json_response({"object": "list", "data": [item]})

    def _request_overrides(body):
        mapping = {
            "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
            "repeat_penalty": "repeat_penalty", "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty", "max_tokens": "max_tokens", "seed": "seed",
            "reasoning_effort": "reasoning_effort",
        }
        out = {}
        for src, dst in mapping.items():
            if src in body and body[src] is not None:
                out[dst] = body[src]
        stop = body.get("stop")
        if isinstance(stop, str):
            # Preserve an OpenAI stop string as one exact sequence.
            out["stop_sequences"] = [str(stop).replace("\\|", "|")]
        elif isinstance(stop, list):
            # Keep the native array form. Serializing with ``|`` corrupts special
            # tokens such as <|im_end|> and <|im_start|>.
            out["stop_sequences"] = [str(x).replace("\\|", "|") for x in stop if x is not None and str(x) != ""]
        # Extensions used by several local OpenAI clients.
        if "thinking_mode" in body:
            out["thinking_mode"] = body["thinking_mode"]
        if "preserve_thinking" in body:
            out["preserve_thinking"] = bool(body["preserve_thinking"])
        return out

    def _openai_response(result, model):
        now = int(time.time())
        message = {"role": "assistant", "content": result.get("response", "")}
        if result.get("thinking"):
            message["reasoning_content"] = result.get("thinking")
        info = result.get("info") or {}
        usage = {
            "prompt_tokens": int(info.get("prompt_tokens") or 0),
            "completion_tokens": int(info.get("completion_tokens") or 0),
            "total_tokens": int(result.get("tokens") or info.get("total_tokens") or 0),
        }
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": now,
            "model": model,
            # ``text`` is a compatibility mirror. Strict OpenAI chat clients use
            # ``message.content``; older/local clients sometimes still read
            # ``choices[0].text`` even on a chat-completions endpoint.
            "choices": [{"index": 0, "message": message, "text": message.get("content", ""), "finish_reason": "stop"}],
            "usage": usage,
        }

    @routes.post("/local-llm/v1/chat/completions")
    async def local_llm_openai_chat(request):
        ok, reason = _authorized(request)
        if not ok:
            return _json_error(reason, 401 if SERVICE.get_config().get("api_key") else 403)
        try:
            body = await request.json()
            messages = _normalized_messages(body.get("messages"))
            requested_model = str(body.get("model") or SERVICE.get_config().get("model") or "local-llm")
            client = request.headers.get("X-Client-Name") or request.headers.get("User-Agent") or "OpenAI-compatible client"
            overrides = _request_overrides(body)
            SERVICE._log(
                f"OpenAI API {request.method} {request.path} • stream={bool(body.get('stream'))} • client={client}"
            )

            if body.get("stream"):
                if not SERVICE.get_config().get("allow_buffered_streaming"):
                    return _json_error("Streaming is disabled", 400)

                # True SSE streaming: LocalGGUFLLM already consumes llama.cpp's
                # iterator for live perf/status. Mirror those same deltas to the
                # OpenAI client instead of swallowing them and rebuilding one
                # buffered response after generation completes.
                rid = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                stream = _sse_response()
                await stream.prepare(request)

                async def write_chunk(delta, finish_reason=None, usage=None):
                    choice = {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                    # Compatibility mirror for local clients that select a text-
                    # completion parser even though they call /chat/completions.
                    # SillyTavern's Custom/OpenAI parser prefers delta.content, so
                    # this does not duplicate text there.
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        choice["text"] = delta.get("content", "")
                    chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": requested_model,
                        "choices": [choice],
                    }
                    if usage is not None:
                        chunk["usage"] = usage
                    data = "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                    await stream.write(data.encode("utf-8"))

                await write_chunk({"role": "assistant"})

                loop = asyncio.get_running_loop()
                token_queue = asyncio.Queue()
                visible_chars = 0
                reasoning_chars = 0

                # Called from the llama.cpp worker thread. The callback only
                # transfers a copied event onto aiohttp's event loop.
                on_token = _token_queue_callback(loop, token_queue)

                generation_task = asyncio.create_task(asyncio.to_thread(
                    SERVICE.generate_messages,
                    messages,
                    image=None,
                    video_frames=None,
                    client=client,
                    overrides=overrides,
                    token_callback=on_token,
                ))

                disconnected = False
                while True:
                    if generation_task.done() and token_queue.empty():
                        break
                    try:
                        event = await asyncio.wait_for(token_queue.get(), timeout=0.10)
                    except asyncio.TimeoutError:
                        continue
                    text = str(event.get("text") or "")
                    if not text:
                        continue
                    try:
                        if event.get("type") == "reasoning":
                            reasoning_chars += len(text)
                            await write_chunk({"reasoning_content": text})
                        else:
                            visible_chars += len(text)
                            await write_chunk({"content": text})
                    except (ConnectionResetError, BrokenPipeError, RuntimeError):
                        disconnected = True
                        break

                # Even if the browser disconnects, let the native request unwind
                # cleanly so model/context state and locks are not abandoned. A
                # global Stop is an intentional finish, not a transport failure.
                try:
                    result = await generation_task
                except LocalLLMInterrupted:
                    if not disconnected:
                        await write_chunk({}, finish_reason="stop")
                        await _finish_sse(stream)
                    SERVICE._log(f"OpenAI SSE interrupted by Stop for {client}")
                    return stream
                payload = _openai_response(result, requested_model)
                final_content = str(payload["choices"][0]["message"].get("content") or "")
                final_reasoning = str(payload["choices"][0]["message"].get("reasoning_content") or "")

                if not disconnected:
                    # Compatibility fallback for handlers that do not expose token
                    # deltas even though their final completion object contains text.
                    if visible_chars == 0 and final_content:
                        await write_chunk({"content": final_content})
                        visible_chars += len(final_content)
                    if reasoning_chars == 0 and final_reasoning:
                        await write_chunk({"reasoning_content": final_reasoning})
                        reasoning_chars += len(final_reasoning)

                    # Never leave OpenAI clients with a completely blank assistant
                    # message. A reasoning-only completion is a model/template/stop
                    # condition, not a transport success; surface that fact visibly
                    # while still preserving the reasoning_content stream separately.
                    if visible_chars == 0 and final_reasoning:
                        notice = "[Model produced reasoning but no final response text. Try increasing max tokens, disabling thinking, or reviewing stop sequences.]"
                        await write_chunk({"content": notice})
                        visible_chars += len(notice)

                    usage = payload.get("usage") if bool((body.get("stream_options") or {}).get("include_usage")) else None
                    await write_chunk({}, finish_reason="stop", usage=usage)
                    await _finish_sse(stream)

                SERVICE._log(
                    f"OpenAI SSE sent {visible_chars} visible chars • {reasoning_chars} reasoning chars to {client}"
                    + (" (client disconnected)" if disconnected else "")
                )
                return stream

            result = await asyncio.to_thread(
                SERVICE.generate_messages,
                messages,
                image=None,
                video_frames=None,
                client=client,
                overrides=overrides,
            )
            return web.json_response(_openai_response(result, requested_model))
        except LocalLLMInterrupted as e:
            return _json_error(e, 409)
        except Exception as e:
            return _json_error(e, 500)

    @routes.post("/local-llm/v1/completions")
    async def local_llm_openai_completions(request):
        ok, reason = _authorized(request)
        if not ok:
            return _json_error(reason, 401 if SERVICE.get_config().get("api_key") else 403)
        try:
            body = await request.json()
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            requested_model = str(body.get("model") or SERVICE.get_config().get("model") or "local-llm")
            client = request.headers.get("X-Client-Name") or request.headers.get("User-Agent") or "completion client"
            overrides = _request_overrides(body)
            SERVICE._log(
                f"OpenAI API {request.method} {request.path} • stream={bool(body.get('stream'))} • client={client}"
            )

            if body.get("stream"):
                if not SERVICE.get_config().get("allow_buffered_streaming"):
                    return _json_error("Streaming is disabled", 400)

                rid = "cmpl-" + uuid.uuid4().hex
                created = int(time.time())
                stream = _sse_response()
                await stream.prepare(request)

                async def write_text_chunk(text="", finish_reason=None, usage=None):
                    chunk = {
                        "id": rid,
                        "object": "text_completion",
                        "created": created,
                        "model": requested_model,
                        "choices": [{
                            "text": str(text or ""),
                            "index": 0,
                            "finish_reason": finish_reason,
                        }],
                    }
                    if usage is not None:
                        chunk["usage"] = usage
                    await stream.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))

                loop = asyncio.get_running_loop()
                token_queue = asyncio.Queue()
                visible_chars = 0

                on_token = _token_queue_callback(loop, token_queue)

                generation_task = asyncio.create_task(asyncio.to_thread(
                    SERVICE.generate_messages,
                    [{"role": "user", "content": str(prompt)}],
                    image=None,
                    video_frames=None,
                    client=client,
                    overrides=overrides,
                    token_callback=on_token,
                ))

                disconnected = False
                while True:
                    if generation_task.done() and token_queue.empty():
                        break
                    try:
                        event = await asyncio.wait_for(token_queue.get(), timeout=0.10)
                    except asyncio.TimeoutError:
                        continue
                    if event.get("type") == "reasoning":
                        continue
                    text = str(event.get("text") or "")
                    if not text:
                        continue
                    try:
                        await write_text_chunk(text)
                        visible_chars += len(text)
                    except (ConnectionResetError, BrokenPipeError, RuntimeError):
                        disconnected = True
                        break

                try:
                    result = await generation_task
                except LocalLLMInterrupted:
                    if not disconnected:
                        await write_text_chunk("", finish_reason="stop")
                        await _finish_sse(stream)
                    SERVICE._log(f"OpenAI text SSE interrupted by Stop for {client}")
                    return stream
                final_content = str(result.get("response") or "")
                info = result.get("info") or {}
                usage = {
                    "prompt_tokens": int(info.get("prompt_tokens") or info.get("prompt_eval_tokens") or 0),
                    "completion_tokens": int(info.get("completion_tokens") or 0),
                    "total_tokens": int(result.get("tokens") or info.get("total_tokens") or 0),
                }

                if not disconnected:
                    if visible_chars == 0 and final_content:
                        await write_text_chunk(final_content)
                        visible_chars += len(final_content)
                    await write_text_chunk("", finish_reason="stop", usage=(usage if bool((body.get("stream_options") or {}).get("include_usage")) else None))
                    await _finish_sse(stream)

                SERVICE._log(
                    f"OpenAI text SSE sent {visible_chars} visible chars to {client}"
                    + (" (client disconnected)" if disconnected else "")
                )
                return stream

            result = await asyncio.to_thread(
                SERVICE.generate_messages,
                [{"role": "user", "content": str(prompt)}],
                image=None,
                video_frames=None,
                client=client,
                overrides=overrides,
            )
            info = result.get("info") or {}
            response_text = str(result.get("response") or "")
            SERVICE._log(f"OpenAI text response returned {len(response_text)} visible chars to {client}")
            return web.json_response({
                "id": "cmpl-" + uuid.uuid4().hex,
                "object": "text_completion",
                "created": int(time.time()),
                "model": requested_model,
                "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": int(info.get("prompt_tokens") or info.get("prompt_eval_tokens") or 0),
                    "completion_tokens": int(info.get("completion_tokens") or 0),
                    "total_tokens": int(result.get("tokens") or info.get("total_tokens") or 0),
                },
            })
        except LocalLLMInterrupted as e:
            return _json_error(e, 409)
        except Exception as e:
            return _json_error(e, 500)

except Exception as route_error:
    log.warning("[Local LLM Server] HTTP routes unavailable: %s", route_error)


maybe_autostart()
