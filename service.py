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
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import folder_paths

from .nodes import (
    LocalGGUFLLM,
    _AUTO_VISION,
    _NONE,
    _cleanup_llm,
    _find_matching_mmproj,
    _validate_mmproj_pair,
    _gpu_choices,
    _metadata_for,
    _model_lists,
    _MODEL_CACHE,
)
from .gguf_meta import detect_family, recommended_model_preset, available_model_presets
from .presets import MEMORY_PRESETS, MODEL_PRESETS, capabilities_for_family, public_presets

log = logging.getLogger(__name__)

STATE_STOPPED = "stopped"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_PROCESSING = "processing"
STATE_GENERATING = "generating"
STATE_WAITING_COMFY = "waiting_comfy"
STATE_RELOADING = "reloading"
STATE_STOPPING = "stopping"
STATE_ERROR = "error"

LOAD_FIELDS = {
    "model", "vision_model", "memory_preset", "context_size", "kv_cache_k", "kv_cache_v",
    "kv_cache_location", "gpu_layers", "flash_attention", "prompt_batch_size",
    "memory_batch_size", "use_mmap", "use_mlock", "split_mode", "main_gpu", "tensor_split",
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


def _ensure_preset_dirs():
    for path in (PRESET_ROOT_DIR, SAMPLER_PRESET_DIR, SYSTEM_PROMPT_PRESET_DIR, PROMPT_PRESET_DIR):
        path.mkdir(parents=True, exist_ok=True)


_ensure_preset_dirs()


def _safe_preset_name(name: str) -> str:
    name = str(name or "").strip()
    # Keep filenames portable between WSL/Linux and Windows-backed model trees.
    name = ''.join('_' if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in name)
    name = name.rstrip('. ').strip()
    if not name:
        raise ValueError("Preset name cannot be empty")
    if name.lower() in {"default", "custom", "server default"}:
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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return {"name": clean_name, "settings": copy.deepcopy(clean), "path": str(path)}


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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)
    return {"name": clean_name, "text": value, "path": str(path)}


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


def _default_config() -> dict[str, Any]:
    d = _node_defaults()
    # The global service defaults to a hot, yieldable native context: while
    # resident it behaves like the persistent server, but ComfyUI can evict it
    # automatically through its normal VRAM pressure path.
    d["model_retention"] = "ComfyUI Managed"
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
    if config.get("vram_policy") not in {"Auto Yield to ComfyUI", "Keep Resident"}:
        config["vram_policy"] = "Auto Yield to ComfyUI"
    config["model_retention"] = (
        "ComfyUI Managed" if config.get("vram_policy") == "Auto Yield to ComfyUI"
        else "Persistent (Driver Managed)"
    )
    return config


def _write_config_file(config: dict[str, Any]):
    path = _config_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


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
    API_VERSION = 1

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
        self._log("Service manager initialized")

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
            if phase == "loading":
                self._state = STATE_LOADING
                self._current_phase_started = time.time()
                self._current_completion_tokens = 0
                self._current_tokens_per_second = 0.0
            elif phase == "processing":
                self._state = STATE_PROCESSING
                self._current_phase_started = time.time()
                self._current_completion_tokens = 0
                self._current_tokens_per_second = 0.0
            else:
                if previous_state != STATE_GENERATING:
                    self._current_phase_started = time.time()
                # No token stream event arrives until prompt evaluation has finished;
                # generation progress is therefore the prompt->decode transition.
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
        return {
            "root_directory": str(PRESET_ROOT_DIR),
            "sampler": {
                "directory": str(SAMPLER_PRESET_DIR),
                "default": self.request_generation_defaults(),
                "names": ["Default", "Custom", *sorted(samplers, key=str.lower)],
                "presets": {name: copy.deepcopy(item["settings"]) for name, item in samplers.items()},
            },
            "system_prompts": {
                "directory": str(SYSTEM_PROMPT_PRESET_DIR),
                "names": ["Custom", *sorted(systems, key=str.lower)],
                "presets": {name: str(item.get("text", "")) for name, item in systems.items()},
            },
            "prompts": {
                "directory": str(PROMPT_PRESET_DIR),
                "names": ["Custom", *sorted(prompts, key=str.lower)],
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
                self._restart_required = any(before.get(k) != self._config.get(k) for k in LOAD_FIELDS)
        self._log("Configuration saved" + ("; model reload required" if self._restart_required else ""))
        self._emit()
        return self.public_config()

    def regenerate_api_key(self):
        key = "sk-local-" + secrets.token_urlsafe(24)
        self.update_config({"api_key": key})
        return key

    def _call_args(self):
        cfg = self.get_config()
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

    def _wait_for_comfy_idle(self, reason="LLM GPU request"):
        """Serialize external/native LLM GPU work behind active ComfyUI execution.

        Calling ComfyUI's memory manager from another thread while diffusion CUDA
        kernels are still executing can invalidate allocations that those kernels
        are using.  External service requests therefore wait until the active
        ComfyUI prompt finishes before loading/reloading or running llama.cpp.
        Workflow-owned Local LLM Service Generate calls bypass this guard because
        they already execute serially inside that same ComfyUI prompt.
        """
        logged = False
        started = time.perf_counter()
        while True:
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
        return str(client or "").startswith("ComfyUI Local LLM Service Generate")

    def _warm_load(self):
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
        })
        response, thinking, info_json, tokens, api = LocalGGUFLLM().generate(**args)
        self._api = api
        try:
            info = json.loads(info_json)
        except Exception:
            info = {"raw": info_json}
        self._last_result = {"response": response, "thinking": thinking, "info": info, "tokens": int(tokens)}

    def start(self, wait_for_comfy=True):
        with self._state_lock:
            if self._state in {STATE_LOADING, STATE_RELOADING, STATE_PROCESSING, STATE_GENERATING, STATE_WAITING_COMFY}:
                return self.status()
            if self._state == STATE_READY and self._api is not None and not self._restart_required and self._resident_snapshot():
                return self.status()
            self._state = STATE_LOADING
            self._error = None
        self._emit()
        self._log("Loading persistent GGUF service")
        try:
            if wait_for_comfy:
                self._wait_for_comfy_idle("loading the LLM")
            with self._state_lock:
                self._state = STATE_LOADING
            self._emit()
            self._warm_load()
            with self._state_lock:
                self._state = STATE_READY
                self._model_loaded = True
                self._restart_required = False
                self._error = None
            self._log("Model loaded; service ready")
        except Exception as e:
            with self._state_lock:
                self._state = STATE_ERROR
                self._model_loaded = False
                self._error = f"{type(e).__name__}: {e}"
                self._api = None
            self._log(self._error, "error")
            raise
        finally:
            self._emit()
        return self.status()

    def stop(self):
        with self._state_lock:
            if self._state == STATE_STOPPED:
                return self.status()
            self._state = STATE_STOPPING
        self._emit()
        self._log("Stopping service and unloading persistent GGUF")
        try:
            _cleanup_llm()
        finally:
            with self._state_lock:
                self._api = None
                self._model_loaded = False
                self._state = STATE_STOPPED
                self._error = None
                self._restart_required = False
                self._current_client = None
                self._current_started = None
            self._emit()
        return self.status()

    def reload(self, wait_for_comfy=True):
        with self._state_lock:
            self._state = STATE_RELOADING
            self._error = None
        self._emit()
        try:
            if wait_for_comfy:
                self._wait_for_comfy_idle("reloading the LLM")
            with self._state_lock:
                self._state = STATE_RELOADING
            self._emit()
            _cleanup_llm()
            self._api = None
            self._warm_load()
            with self._state_lock:
                self._state = STATE_READY
                self._model_loaded = True
                self._restart_required = False
            self._log("Model reloaded")
        except Exception as e:
            with self._state_lock:
                self._state = STATE_ERROR
                self._model_loaded = False
                self._error = f"{type(e).__name__}: {e}"
                self._api = None
            self._log(self._error, "error")
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
        if restart_required and state == STATE_READY:
            self.reload(wait_for_comfy=not workflow_owned)
        else:
            # A READY service with an evicted/yielded model is reloaded on demand.
            self.start(wait_for_comfy=not workflow_owned)

    def generate_messages(self, messages, image=None, video_frames=None, client="unknown", overrides=None, token_callback=None):
        messages = _normalized_messages(messages)
        overrides = dict(overrides or {})
        # API/client request fields may not change persistent allocation settings.
        forbidden = sorted(set(overrides) & LOAD_FIELDS)
        if forbidden:
            raise ValueError("Request cannot override server load/memory setting(s): " + ", ".join(forbidden))

        with self._state_lock:
            self._queue_count += 1
        self._emit()
        queued_at = time.perf_counter()
        try:
            with self._generation_lock:
                generation_lock_acquired = time.perf_counter()
                queue_wait_seconds = generation_lock_acquired - queued_at
                with self._state_lock:
                    self._queue_count = max(0, self._queue_count - 1)
                # External API clients must not run llama.cpp CUDA work concurrently
                # with an executing ComfyUI workflow.  In particular, a managed
                # on-demand reload may call mm.free_memory(), which is unsafe while
                # diffusion kernels are actively using those allocations.
                workflow_owned = self._is_workflow_client(client)
                comfy_wait_seconds = 0.0
                if not workflow_owned:
                    comfy_wait_seconds = float(self._wait_for_comfy_idle("servicing the LLM request") or 0.0)
                ensure_started_at = time.perf_counter()
                self._ensure_started(workflow_owned=workflow_owned)
                ensure_started_seconds = time.perf_counter() - ensure_started_at
                request_needs_reload = not self._resident_snapshot()
                self._log(
                    "PERF request preflight: "
                    f"queue_wait={queue_wait_seconds:.3f}s • "
                    f"comfy_wait={comfy_wait_seconds:.3f}s • "
                    f"ensure_started={ensure_started_seconds:.3f}s • "
                    f"resident_before_call={not request_needs_reload}"
                )
                with self._state_lock:
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
                cfg = self.get_config()
                if cfg.get("log_prompt_content"):
                    self._log("Prompt content: " + self._safe_content_for_log(messages))
                started = time.perf_counter()
                # Re-enter the canonical node execution path with the CURRENT
                # server configuration on every request. Persistent load settings
                # still reuse the same native llama.cpp context, while edits to
                # sampling/model presets take effect immediately without a reload.
                args = self._call_args()
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
                node_call_started = time.perf_counter()
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
                    self._current_client = None
                    self._current_started = None
                    self._current_phase_started = None
                    self._current_completion_tokens = 0
                    self._current_tokens_per_second = 0.0
                    self._current_prompt_tokens = 0
                    self._current_prompt_tokens_per_second = 0.0
                    self._error = None
                info = result.get("info") or {}
                speed = info.get("tokens_per_second")
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
                )
                room = preload.get("room_details") or {}
                self._log(
                    "PERF handoff: "
                    f"target={float(preload.get('requested_release_bytes') or 0) / (1024*1024):.1f}MiB "
                    f"source={preload.get('target_source', 'n/a')} • "
                    f"estimated={float(preload.get('estimated_vram_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"observed={float(preload.get('observed_vram_bytes') or gpu_backend.get('observed_vram_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"raw_before={float(room.get('raw_free_before_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"reclaimable={float(room.get('torch_reclaimable_before_bytes') or 0) / (1024*1024):.1f}MiB • "
                    f"cache_probe={'skipped' if room.get('cache_probe_skipped') else ('run' if room.get('cache_probe_called') else 'not-needed')} • "
                    f"sync={float(room.get('pre_release_sync_seconds') or 0):.3f}s • "
                    f"evict={float(room.get('release_seconds') or 0):.3f}s • "
                    f"cache_flush={float(room.get('cache_probe_seconds') or room.get('soft_empty_cache_seconds') or 0):.3f}s"
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
        except Exception as e:
            with self._state_lock:
                self._queue_count = max(0, self._queue_count - 1)
                # A request error should not mark the whole service unusable if the
                # model is still resident. Keep READY when possible.
                self._state = STATE_READY if self._api is not None and self._api.is_loaded() else STATE_ERROR
                self._error = f"{type(e).__name__}: {e}"
                self._current_client = None
                self._current_started = None
                self._current_completion_tokens = 0
                self._current_tokens_per_second = 0.0
                self._current_prompt_tokens = 0
                self._current_prompt_tokens_per_second = 0.0
            self._log(self._error, "error")
            raise
        finally:
            self._emit()

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
        capabilities = capabilities_for_family(family)
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
            "restart_required": bool(restart_required),
            "error": error,
            "model": model,
            "vision_model": cfg.get("vision_model"),
            "family": family,
            "capabilities": capabilities,
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
    return {
        "models": models,
        "vision": vision,
        "gpus": _gpu_choices(),
        "model_presets": ["Auto (Detected)", "Custom"] + list(MODEL_PRESETS.keys()),
        "memory_presets": ["Custom"] + list(MEMORY_PRESETS.keys()),
        "context_sizes": list(CONTEXT_SIZE_STEPS),
        "request_presets": SERVICE.request_preset_catalog(),
        "node_presets": SERVICE.node_preset_catalog(),
        "presets": public_presets(),
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
    return {
        "metadata": {k: (v[:2000] + "…" if isinstance(v, str) and len(v) > 2000 else v) for k, v in md.items()},
        "family": family,
        "recommended_preset": recommended_model_preset(md, name),
        "available_presets": available_model_presets(md, name),
        "capabilities": capabilities_for_family(family),
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

class GetLocalLLMService:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("LOCAL_LLM_SERVICE_API",)
    RETURN_NAMES = ("api",)
    FUNCTION = "get_service"
    CATEGORY = "LLM/Local Service"
    DESCRIPTION = "Return a live facade for the global persistent Local LLM Server."

    def get_service(self):
        return (SERVICE.api,)


class LocalLLMServiceGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt_preset": (["Custom", *text_preset_names("system_prompts")], {"default": "Custom", "tooltip": "Reusable system prompts from models/LLM/local_LLM_presets/system_prompts. Editing the text switches this selector to Custom."}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True, "dynamicPrompts": False}),
                "prompt_preset": (["Custom", *text_preset_names("prompts")], {"default": "Custom", "tooltip": "Reusable prompts from models/LLM/local_LLM_presets/prompts. Editing the text switches this selector to Custom."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "sampling_mode": (["Default", "Custom", *sampler_preset_names()], {"default": "Default", "tooltip": "Default mirrors the current global server generation defaults. Editing a sampler setting switches this selector to Custom. Saved samplers are loaded from models/LLM/local_LLM_presets/sampler."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 10000}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 262144}),
                "vision_max_images": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Maximum still images accepted from the IMAGE batch."}),
                "vision_max_frames": ("INT", {"default": 24, "min": 1, "max": 1024, "tooltip": "Evenly sampled frames from the Video Frames IMAGE batch."}),
                "vision_max_edge": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One still image or an IMAGE batch."}),
                "video_frames": ("IMAGE", {"tooltip": "Ordered video frames as an IMAGE batch; sampled evenly."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("response", "thinking", "info", "tokens")
    FUNCTION = "generate"
    CATEGORY = "LLM/Local Service"
    DESCRIPTION = "Generate through the global persistent Local LLM Server with independent sampler, system-prompt, and prompt presets."

    @classmethod
    def IS_CHANGED(cls, sampling_mode="Default", system_prompt_preset="Custom", prompt_preset="Custom", **kwargs):
        # This node depends on global server state and preset files that are not
        # fully represented by its serialized widget values. Include the effective
        # default/preset contents so changes on disk invalidate cached responses.
        mode = str(sampling_mode or "Default")
        payload = {"sampler_mode": mode}
        for key in ("vision_max_images", "vision_max_frames", "vision_max_edge"):
            if key in kwargs:
                payload[key] = kwargs.get(key)
        if mode == "Default":
            payload["sampler"] = SERVICE.request_generation_defaults()
        elif mode != "Custom":
            preset = sampler_presets().get(mode)
            payload["sampler"] = copy.deepcopy((preset or {}).get("settings"))
            payload["sampler_mtime_ns"] = (preset or {}).get("mtime_ns")

        for kind, selected in (("system_prompts", system_prompt_preset), ("prompts", prompt_preset)):
            name = str(selected or "Custom")
            payload[kind] = {"name": name}
            if name != "Custom":
                item = text_presets(kind).get(name)
                payload[kind]["text"] = None if item is None else item.get("text", "")
                payload[kind]["mtime_ns"] = None if item is None else item.get("mtime_ns")

        # Model/template configuration affects output even when request-level
        # sampler and prompt fields are custom.
        cfg = SERVICE.get_config()
        payload["server"] = {
            "model": cfg.get("model"),
            "vision_model": cfg.get("vision_model"),
            "model_preset": cfg.get("model_preset"),
            "thinking_mode": cfg.get("thinking_mode"),
            "reasoning_effort": cfg.get("reasoning_effort"),
            "preserve_thinking": cfg.get("preserve_thinking"),
            "chat_format": cfg.get("chat_format"),
            "custom_chat_format": cfg.get("custom_chat_format"),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def generate(self, system_prompt_preset, system_prompt, prompt_preset, prompt,
                 sampling_mode, temperature, top_p, top_k, min_p, repeat_penalty,
                 presence_penalty, frequency_penalty, max_tokens, vision_max_images, vision_max_frames,
                 vision_max_edge, seed, image=None, video_frames=None):
        mode = str(sampling_mode or "Default")

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

        # The standard ComfyUI seed is always request-local. It is intentionally
        # not part of a sampler preset.
        overrides = {
            "seed": seed,
            "vision_max_images": vision_max_images,
            "vision_max_frames": vision_max_frames,
            "vision_max_edge": vision_max_edge,
        }
        if mode == "Default":
            # Send no sampler overrides: the global server remains authoritative.
            pass
        elif mode == "Custom":
            overrides.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "max_tokens": max_tokens,
            })
        else:
            saved = load_sampler_preset(mode)
            if saved is None:
                # A workflow can outlive a deleted preset. Fall back to the
                # workflow's serialized visible sampler values.
                saved = {
                    "temperature": temperature, "top_p": top_p, "top_k": top_k,
                    "min_p": min_p, "repeat_penalty": repeat_penalty,
                    "presence_penalty": presence_penalty, "frequency_penalty": frequency_penalty,
                    "max_tokens": max_tokens,
                }
            overrides.update(saved)

        result = SERVICE.api.generate(
            system_prompt=effective_system_prompt,
            prompt=effective_prompt,
            image=image,
            video_frames=video_frames,
            client="ComfyUI Local LLM Service Generate",
            **overrides,
        )
        info = copy.deepcopy(result.get("info") or {})
        info["request_presets"] = {
            "sampler": mode,
            "system_prompt": str(system_prompt_preset or "Custom"),
            "prompt": str(prompt_preset or "Custom"),
        }
        return (
            result.get("response", ""),
            result.get("thinking", ""),
            json.dumps(info, indent=2),
            int(result.get("tokens") or 0),
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

    @routes.post("/local_llm_server/start")
    async def local_llm_server_start(request):
        try:
            status = await asyncio.to_thread(SERVICE.start)
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
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model", "created": 0, "owned_by": "local-llm-server"}]})

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
                stream = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
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

                def on_token(event):
                    # Called from the llama.cpp worker thread. Never touch aiohttp
                    # objects here; only transfer the immutable event to the loop.
                    try:
                        loop.call_soon_threadsafe(token_queue.put_nowait, dict(event or {}))
                    except Exception:
                        pass

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

                # Even if the browser disconnects, let the native request finish
                # cleanly so model/context state and locks are not abandoned.
                result = await generation_task
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
                    await stream.write(b"data: [DONE]\n\n")
                    try:
                        await stream.write_eof()
                    except Exception:
                        pass

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
                stream = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream; charset=utf-8",
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
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

                def on_token(event):
                    try:
                        loop.call_soon_threadsafe(token_queue.put_nowait, dict(event or {}))
                    except Exception:
                        pass

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

                result = await generation_task
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
                    await stream.write(b"data: [DONE]\n\n")
                    try:
                        await stream.write_eof()
                    except Exception:
                        pass

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
        except Exception as e:
            return _json_error(e, 500)

except Exception as route_error:
    log.warning("[Local LLM Server] HTTP routes unavailable: %s", route_error)


maybe_autostart()
