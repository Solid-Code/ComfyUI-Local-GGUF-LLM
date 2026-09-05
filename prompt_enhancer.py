from __future__ import annotations

import copy

import asyncio
import json
import logging
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import folder_paths

log = logging.getLogger(__name__)

NODE_VERSION = "0.6.29-alpha"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_DIR = PACKAGE_DIR / "templates" / "default"
USER_TEMPLATE_DIR = Path(folder_paths.models_dir) / "LLM" / "local_LLM_presets" / "prompt_enhancer"
USER_PROMPT_SET_DIR = USER_TEMPLATE_DIR / "prompt_sets"
PROMPT_SET_NONE = "Unsaved"

DEFAULT_TEMPLATE_ORDER = [
    "Krea 2 - Image",
    "MiniMax H3 - T2VA",
    "MiniMax H3 - I2VA",
    "MiniMax H3 - FL2VA",
    "MiniMax H3 - L2VA",
    "MiniMax H3 - Ref2VA",
]
DEFAULT_SELECTION = "Default / Krea 2 - Image"

LOCAL_LLM_SAMPLER_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "max_tokens",
)
LOCAL_LLM_VISION_FIELDS = ("vision_max_images", "vision_max_frames", "vision_max_edge")

# Manual Enhance is armed by the browser, then ComfyUI partially executes only
# this node (and the dependencies required to produce its connected media).  The
# pending request stores only text/metadata; IMAGE/VIDEO tensors are never kept
# in a long-lived global cache.
_PENDING_LOCK = threading.Lock()
_PENDING_REQUESTS: dict[str, dict[str, Any]] = {}
_PENDING_TTL_SECONDS = 300.0

# A manual Enhance batch pins the complete Local LLM runtime configuration that
# existed when the user pressed Enhance.  Modal autosaves are still accepted
# while the batch runs, but they are deliberately not observed by later items in
# that batch.  Sessions are backend-only and expire defensively if a browser is
# closed/reloaded before it can send the normal batch_end request.
_BATCH_LOCK = threading.Lock()
_BATCH_SESSIONS: dict[str, dict[str, Any]] = {}
_BATCH_TTL_SECONDS = 2 * 60 * 60.0

# A normal workflow can be queued while a manual Enhance batch is still the
# active ComfyUI queue item. ComfyUI serializes that later workflow immediately,
# so its Prompt Enhancer widgets may represent the pre-batch (or an intermediate
# progress) array. Keep the completed batch result briefly in the backend and
# reconcile those already-queued snapshots when they eventually execute.
_MANUAL_HISTORY_LOCK = threading.Lock()
_MANUAL_HISTORY_STATES: dict[str, dict[str, Any]] = {}
_MANUAL_HISTORY_TTL_SECONDS = 30 * 60.0

# Prompt Cycle used to advance only when the browser processed ComfyUI's
# `executed` event. Background tabs can throttle/delay that JavaScript, causing
# repeated workflow runs to serialize the same active index. Keep the live
# workflow cursor in the backend instead; the frontend now mirrors this state
# for display but is not part of the correctness path.
_PROMPT_CYCLE_LOCK = threading.Lock()
_PROMPT_CYCLE_STATES: dict[str, dict[str, Any]] = {}
_PROMPT_CYCLE_TTL_SECONDS = 24 * 60 * 60.0


def _display_name_from_path(path: Path) -> str:
    return path.stem.strip()


def _safe_user_filename(name: str) -> str:
    """Turn a user-visible template name into a safe local .txt filename."""
    name = str(name or "").strip()
    name = re.sub(r"^(?:Default|User)\s*/\s*", "", name, flags=re.IGNORECASE)
    name = name.replace("\x00", "")
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        raise ValueError("Enter a template name before saving.")
    if name in {".", ".."}:
        raise ValueError("Invalid template name.")
    return name[:120]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig").strip()


def _atomic_write(path: Path, text: str) -> None:
    """Atomically replace a user-owned prompt/template file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _template_records() -> list[dict[str, Any]]:
    """Return protected built-ins followed by editable user templates."""
    records: list[dict[str, Any]] = []

    default_paths: dict[str, Path] = {}
    if DEFAULT_TEMPLATE_DIR.is_dir():
        for path in DEFAULT_TEMPLATE_DIR.glob("*.txt"):
            if path.is_file():
                default_paths[_display_name_from_path(path)] = path

    ordered_default_names = [n for n in DEFAULT_TEMPLATE_ORDER if n in default_paths]
    ordered_default_names.extend(sorted(n for n in default_paths if n not in ordered_default_names))
    for name in ordered_default_names:
        path = default_paths[name]
        try:
            text = _read_text(path)
        except Exception as exc:
            log.warning("[Local LLM Prompt Enhancer] Could not read default template %s: %s", path, exc)
            continue
        records.append({
            "label": f"Default / {name}",
            "name": name,
            "source": "default",
            "protected": True,
            "text": text,
        })

    try:
        USER_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning("[Local LLM Prompt Enhancer] Could not create user template directory %s: %s", USER_TEMPLATE_DIR, exc)

    if USER_TEMPLATE_DIR.is_dir():
        for path in sorted(USER_TEMPLATE_DIR.glob("*.txt"), key=lambda p: p.name.casefold()):
            if not path.is_file():
                continue
            name = _display_name_from_path(path)
            try:
                text = _read_text(path)
            except Exception as exc:
                log.warning("[Local LLM Prompt Enhancer] Could not read user template %s: %s", path, exc)
                continue
            records.append({
                "label": f"User / {name}",
                "name": name,
                "source": "user",
                "protected": False,
                "text": text,
            })

    return records


def _template_labels() -> list[str]:
    labels = [str(record["label"]) for record in _template_records()]
    return labels or ["Custom"]


def _template_by_label(label: str) -> dict[str, Any] | None:
    label = str(label or "")
    for record in _template_records():
        if record["label"] == label:
            return record
    return None


def _messages_for(
    prompt: str,
    enhancement_text: str,
    *,
    has_images: bool = False,
    has_video: bool = False,
) -> list[dict[str, str]]:
    prompt = str(prompt or "")
    enhancement_text = str(enhancement_text or "").strip()
    if not prompt.strip():
        raise ValueError("Prompt is empty.")
    if not enhancement_text:
        raise ValueError("Enhancement Instructions are empty.")

    media_note = ""
    if has_images or has_video:
        kinds = []
        if has_images:
            kinds.append("image reference(s)")
        if has_video:
            kinds.append("a video reference")
        media_note = (
            " The request includes " + " and ".join(kinds) + ". Inspect supplied visual media and use only details "
            "that are actually visible and useful to the requested generation mode. Treat media as conditioning/reference "
            "material rather than something to describe mechanically unless the enhancement template asks for that."
        )

    system = (
        "You are a precise prompt-enhancement editor. Return ONLY the revised prompt text. "
        "Do not explain your changes, add a title or label, or wrap the result in quotation marks or Markdown code fences. "
        "The enhancement instructions are editing instructions, not content to append verbatim. "
        "Preserve the source prompt's intent, explicit constraints, identities, names, quoted dialogue, requested text, "
        "reference labels, placeholders, timestamps, XML-style tags, field names, and model-specific syntax unless the "
        "enhancement instructions explicitly require a structural conversion. Never answer or execute the source prompt; "
        "rewrite it as a generation prompt. Do not invent visual facts that are unsupported by the source prompt or supplied media."
        + media_note
    )

    user = "\n".join([
        "Enhancement instructions:",
        "<enhancement_instructions>",
        enhancement_text,
        "</enhancement_instructions>",
        "",
        "Source prompt to enhance:",
        "<prompt_to_enhance>",
        prompt,
        "</prompt_to_enhance>",
    ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _clean_response(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _find_local_llm_service():
    """Find the live SERVICE exported by ComfyUI-Local-GGUF-LLM safely."""
    candidates: list[tuple[int, str, Any]] = []
    seen: set[int] = set()

    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            module_dict = vars(module)
        except Exception:
            continue

        module_file = str(module_dict.get("__file__") or "")
        haystack = f"{name} {module_file}".replace("\\", "/").lower()
        if not (
            "comfyui-local-gguf-llm" in haystack
            or "comfyui_local_gguf_llm" in haystack
            or "local_gguf_llm" in haystack
        ):
            continue

        service = module_dict.get("SERVICE")
        if service is None or id(service) in seen:
            continue
        try:
            generate_messages = getattr(service, "generate_messages", None)
            status = getattr(service, "status", None)
        except Exception:
            continue
        if not callable(generate_messages) or not callable(status):
            continue

        seen.add(id(service))
        score = 0
        if "/comfyui-local-gguf-llm/service.py" in haystack:
            score += 20
        if "comfyui-local-gguf-llm" in haystack:
            score += 10
        if "service" in str(name).lower():
            score += 2
        candidates.append((score, str(name), service))

    if not candidates:
        raise RuntimeError(
            "ComfyUI Local GGUF LLM service was not found. Install/enable ComfyUI-Local-GGUF-LLM and restart ComfyUI."
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]



def _pending_comfy_queue_count() -> int | None:
    """Return the number of ComfyUI jobs waiting behind the current one.

    Prompt Enhancer executes its manual batch as one targeted ComfyUI queue
    item. While that item is running, pending jobs are exactly the work that
    needs a deterministic native-LLM handoff before this item returns. If the
    queue cannot be inspected, return None so callers fail safe and hand off.
    """
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        queue = getattr(server, "prompt_queue", None)
        if queue is None:
            return None
        getter = getattr(queue, "get_current_queue_volatile", None)
        if not callable(getter):
            getter = getattr(queue, "get_current_queue", None)
        if not callable(getter):
            return None
        _running, pending = getter()
        return len(pending or [])
    except Exception:
        return None


def _handoff_after_enhancer_batch(service, *, reason: str) -> bool:
    """Yield immediately only when another ComfyUI job is already waiting.

    When the queue is empty, leaving the resident context hot is safe in Auto
    Yield mode: any later GPU-heavy ComfyUI load crosses the existing free_memory
    pressure hook and acquires native ownership there. If queue inspection fails,
    preserve the older fail-safe behavior and hand off unconditionally.
    """
    pending = _pending_comfy_queue_count()
    if pending == 0:
        log.info("[Local LLM Prompt Enhancer] Handoff decision: pending=0 -> keep resident GGUF hot")
        return False
    if pending is None:
        log.info("[Local LLM Prompt Enhancer] Handoff decision: queue state unavailable -> fail-safe GPU handoff (%s)", reason)
    else:
        log.info("[Local LLM Prompt Enhancer] Handoff decision: pending=%d -> GPU handoff (%s)", int(pending), reason)
    service.gpu_handoff(reason=reason)
    return True

def _find_local_llm_export(name: str):
    """Safely find a named export from the loaded Local GGUF extension."""
    candidates: list[tuple[int, str, Any]] = []
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            module_dict = vars(module)
        except Exception:
            continue
        module_file = str(module_dict.get("__file__") or "")
        haystack = f"{module_name} {module_file}".replace("\\", "/").lower()
        if not (
            "comfyui-local-gguf-llm" in haystack
            or "comfyui_local_gguf_llm" in haystack
            or "local_gguf_llm" in haystack
        ):
            continue
        value = module_dict.get(name)
        if value is None:
            continue
        score = 0
        if "/comfyui-local-gguf-llm/service.py" in haystack:
            score += 20
        if "comfyui-local-gguf-llm" in haystack:
            score += 10
        if "service" in str(module_name).lower():
            score += 2
        candidates.append((score, str(module_name), value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _settings_runtime_config(settings: Any) -> dict[str, Any] | None:
    """Return a request-local complete Settings-node override when selected."""
    linked = settings if isinstance(settings, dict) else None
    if not linked or str(linked.get("settings_preset") or "Current Server") == "Current Server":
        return None
    patch = linked.get("server_config")
    if not isinstance(patch, dict) or not patch:
        return None
    return copy.deepcopy(patch)


def _settings_overrides(settings: Any, fallback_seed: Any = 0) -> dict[str, Any]:
    """Translate LOCAL_LLM_SETTINGS into request overrides while keeping seed request-local."""
    linked = settings if isinstance(settings, dict) else None
    overrides: dict[str, Any] = {"seed": _normalize_seed(fallback_seed)}
    if not linked:
        return overrides
    for key in LOCAL_LLM_VISION_FIELDS:
        if key in linked:
            try:
                overrides[key] = int(linked[key])
            except (TypeError, ValueError, OverflowError):
                pass

    mode = str(linked.get("sampling_mode") or "Default")
    if mode == "Default":
        # No sampler overrides: the Local LLM server's current defaults remain authoritative.
        return overrides

    if mode == "Custom":
        sampler = linked
    else:
        sampler = None
        loader = _find_local_llm_export("load_sampler_preset")
        if callable(loader):
            try:
                sampler = loader(mode)
            except Exception as exc:
                log.warning("[Local LLM Prompt Enhancer] Could not load sampler preset %r: %s", mode, exc)
        # Match Local LLM Generate: a deleted/missing preset falls back to the
        # visible serialized values carried by the Settings node.
        if not isinstance(sampler, dict):
            sampler = linked

    for key in LOCAL_LLM_SAMPLER_FIELDS:
        if key not in sampler:
            continue
        try:
            overrides[key] = int(sampler[key]) if key in {"top_k", "max_tokens"} else float(sampler[key])
        except (TypeError, ValueError, OverflowError):
            continue
    return overrides


def _effective_seed(settings: Any, fallback_seed: Any = 0) -> int:
    # Seed is owned by Prompt Enhancer itself. Local LLM Settings never overrides it.
    return _normalize_seed(fallback_seed)


def _save_user_template(name: str, text: str) -> dict[str, Any]:
    clean_name = _safe_user_filename(name)
    text = str(text or "").strip()
    if not text:
        raise ValueError("Enhancement Instructions are empty; there is nothing to save.")

    USER_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    destination = USER_TEMPLATE_DIR / f"{clean_name}.txt"
    if destination.resolve().parent != USER_TEMPLATE_DIR.resolve():
        raise ValueError("Invalid template path.")

    _atomic_write(destination, text.rstrip() + "\n")
    return {
        "label": f"User / {clean_name}",
        "name": clean_name,
        "source": "user",
        "protected": False,
        "text": text,
    }


def _delete_user_template(label: str) -> dict[str, Any]:
    record = _template_by_label(str(label or ""))
    if record is None:
        raise ValueError("Select a saved user template to delete.")
    if bool(record.get("protected")) or str(record.get("source")) != "user":
        raise ValueError("Built-in default templates cannot be deleted.")

    clean_name = _safe_user_filename(str(record.get("name") or ""))
    path = USER_TEMPLATE_DIR / f"{clean_name}.txt"
    if path.resolve().parent != USER_TEMPLATE_DIR.resolve():
        raise ValueError("Invalid template path.")
    if not path.is_file():
        raise ValueError(f"Template '{clean_name}' was not found.")
    path.unlink()
    return {"label": f"User / {clean_name}", "name": clean_name}


def _prompt_set_records() -> list[dict[str, Any]]:
    USER_PROMPT_SET_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in sorted(USER_PROMPT_SET_DIR.glob("*.json"), key=lambda p: p.name.casefold()):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            prompts = data.get("prompts") if isinstance(data, dict) else None
            if not isinstance(prompts, list):
                continue
            prompts = [str(item) for item in prompts]
            index = _normalize_history_index(data.get("active_index", 0), len(prompts))
            records.append({"name": path.stem, "prompts": prompts, "active_index": index})
        except Exception as exc:
            log.warning("[Local LLM Prompt Enhancer] Could not read prompt array %s: %s", path, exc)
    return records


def _prompt_set_labels() -> list[str]:
    return [PROMPT_SET_NONE, *[str(record["name"]) for record in _prompt_set_records()]]


def _save_prompt_set(name: str, prompts: Any, active_index: Any = 0) -> dict[str, Any]:
    clean_name = _safe_user_filename(name)
    if not isinstance(prompts, list):
        raise ValueError("Prompt set must contain a prompt list.")
    clean_prompts = [str(item) for item in prompts]
    if not clean_prompts:
        raise ValueError("The prompt set is empty.")
    index = _normalize_history_index(active_index, len(clean_prompts))
    USER_PROMPT_SET_DIR.mkdir(parents=True, exist_ok=True)
    destination = USER_PROMPT_SET_DIR / f"{clean_name}.json"
    payload = {"name": clean_name, "prompts": clean_prompts, "active_index": index}
    _atomic_write(destination, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def _load_prompt_set(name: str) -> dict[str, Any]:
    clean_name = _safe_user_filename(name)
    path = USER_PROMPT_SET_DIR / f"{clean_name}.json"
    if not path.is_file():
        raise ValueError(f"Prompt set '{clean_name}' was not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if not isinstance(prompts, list):
        raise ValueError("Saved prompt set is invalid.")
    prompts = [str(item) for item in prompts]
    index = _normalize_history_index(data.get("active_index", 0), len(prompts))
    return {"name": clean_name, "prompts": prompts, "active_index": index}


def _delete_prompt_set(name: str) -> dict[str, Any]:
    clean_name = _safe_user_filename(name)
    path = USER_PROMPT_SET_DIR / f"{clean_name}.json"
    if path.resolve().parent != USER_PROMPT_SET_DIR.resolve():
        raise ValueError("Invalid prompt set path.")
    if not path.is_file():
        raise ValueError(f"Prompt Set '{clean_name}' was not found.")
    path.unlink()
    return {"name": clean_name}


def _enhancer_state_key(node_id: Any = None, state_id: Any = None, runtime_scope: Any = None) -> str:
    """Per-workflow enhancer key, with legacy fallbacks for older clients.

    ``state_id`` is serialized with the node so it can identify the enhancer
    across normal tab remounts. ``runtime_scope`` is browser-session/workflow
    ownership supplied by the frontend and prevents an imported workflow/image
    carrying the same serialized state id from inheriting another tab's live
    Prompt Enhancer cursor/request state.
    """
    stable = str(state_id or "").strip()
    scope = str(runtime_scope or "").strip()
    if stable and scope:
        return f"{scope}\x1f{stable}"
    if stable:
        return stable
    return str(node_id or "").strip()


def _prune_pending_locked(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    stale = [
        node_id for node_id, item in _PENDING_REQUESTS.items()
        if now - float(item.get("created", 0.0)) > _PENDING_TTL_SECONDS
    ]
    for node_id in stale:
        _PENDING_REQUESTS.pop(node_id, None)


def _prune_batches_locked(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    stale = [
        batch_id for batch_id, item in _BATCH_SESSIONS.items()
        if now - float(item.get("created", 0.0)) > _BATCH_TTL_SECONDS
    ]
    for batch_id in stale:
        _BATCH_SESSIONS.pop(batch_id, None)


def _begin_batch() -> str:
    service = _find_local_llm_service()
    get_config = getattr(service, "get_config", None)
    if not callable(get_config):
        raise RuntimeError("Local LLM service does not expose a configuration snapshot API.")
    server_config = copy.deepcopy(get_config())
    batch_id = uuid.uuid4().hex
    with _BATCH_LOCK:
        _prune_batches_locked()
        _BATCH_SESSIONS[batch_id] = {
            "created": time.monotonic(),
            "server_config": server_config,
            # Connected Local LLM Settings is captured lazily on the first
            # partial execution because its serialized value exists only inside
            # the ComfyUI node execution path.
            "settings_captured": False,
            "settings_runtime_config": None,
            "settings_overrides": {},
        }
    return batch_id


def _end_batch(batch_id: Any) -> bool:
    batch_id = str(batch_id or "").strip()
    if not batch_id:
        return False
    with _BATCH_LOCK:
        return _BATCH_SESSIONS.pop(batch_id, None) is not None


def _batch_request_snapshot(batch_id: Any, settings: Any, seed: Any) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve immutable batch settings while allowing the seed to advance.

    The modal/server configuration is captured at batch_begin.  If a Settings
    node is connected, its request-local runtime/sampler values are captured on
    the first item and reused for the rest of the batch.  Seed remains per-item
    because ComfyUI's standard control_after_generate lifecycle intentionally
    advances it between batch entries.
    """
    batch_id = str(batch_id or "").strip()
    if not batch_id:
        return _settings_overrides(settings, seed), _settings_runtime_config(settings), None

    with _BATCH_LOCK:
        _prune_batches_locked()
        session = _BATCH_SESSIONS.get(batch_id)
        if session is None:
            raise RuntimeError("Prompt Enhancer batch configuration expired. Start the batch again.")

        if not bool(session.get("settings_captured")):
            runtime = _settings_runtime_config(settings)
            first_overrides = _settings_overrides(settings, seed)
            first_overrides.pop("seed", None)
            session["settings_runtime_config"] = copy.deepcopy(runtime) if runtime is not None else None
            session["settings_overrides"] = copy.deepcopy(first_overrides)
            session["settings_captured"] = True

        server_snapshot = copy.deepcopy(session.get("server_config") or {})
        runtime_snapshot = copy.deepcopy(session.get("settings_runtime_config"))
        overrides = copy.deepcopy(session.get("settings_overrides") or {})

    overrides["seed"] = _normalize_seed(seed)
    return overrides, runtime_snapshot, server_snapshot


def _normalize_seed(seed: Any) -> int:
    try:
        value = int(seed)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Seed must be an integer.")
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Seed must be between 0 and 18446744073709551615.")
    return value


def _arm_request(
    node_id: str,
    prompt: str,
    enhancement_text: str,
    seed: Any = 0,
    batch_id: Any = "",
    seeds: Any = None,
    batch_count: Any = 1,
    prompt_history_json: Any = "[]",
    prompt_history_index: Any = 0,
    enhanced_prompt: Any = "",
    overwrite_enhanced: Any = False,
    state_id: Any = "",
    runtime_scope: Any = "",
) -> str:
    node_id = str(node_id or "").strip()
    if not node_id:
        raise ValueError("Missing Prompt Enhancer node id.")
    # Validate now so a malformed request fails before ComfyUI queues anything.
    _messages_for(prompt, enhancement_text)

    try:
        requested_count = int(batch_count)
    except (TypeError, ValueError, OverflowError):
        requested_count = 1
    requested_count = max(1, min(64, requested_count))

    seed_values: list[int] = []
    if isinstance(seeds, list):
        for value in seeds[:64]:
            seed_values.append(_normalize_seed(value))
    if not seed_values:
        seed_values = [_normalize_seed(seed)]
    if len(seed_values) < requested_count:
        # Old/stale frontends only send one seed. Repeating it is the safest
        # compatibility behavior; the current frontend sends the complete
        # standard Control After Generate seed sequence for the whole batch.
        seed_values.extend([seed_values[-1]] * (requested_count - len(seed_values)))
    seed_values = seed_values[:requested_count]

    token = uuid.uuid4().hex
    key = _enhancer_state_key(node_id, state_id, runtime_scope)
    with _PENDING_LOCK:
        _prune_pending_locked()
        _PENDING_REQUESTS[key] = {
            "token": token,
            "node_id": node_id,
            "prompt": str(prompt or ""),
            "enhancement_text": str(enhancement_text or ""),
            "seed": seed_values[0],
            "seeds": seed_values,
            "batch_count": len(seed_values),
            "batch_id": str(batch_id or "").strip(),
            "prompt_history_json": str(prompt_history_json or "[]"),
            "prompt_history_index": prompt_history_index,
            "enhanced_prompt": str(enhanced_prompt or ""),
            "overwrite_enhanced": bool(overwrite_enhanced),
            "state_id": str(state_id or "").strip(),
            "runtime_scope": str(runtime_scope or "").strip(),
            "created": time.monotonic(),
        }
    return token


def _pop_request(node_id: Any, state_id: Any = "", runtime_scope: Any = "") -> dict[str, Any] | None:
    key = _enhancer_state_key(node_id, state_id, runtime_scope)
    stable_fallback = _enhancer_state_key(node_id, state_id, "")
    fallback = _enhancer_state_key(node_id, "", "")
    with _PENDING_LOCK:
        _prune_pending_locked()
        pending = _PENDING_REQUESTS.pop(key, None)
        if pending is None and stable_fallback and stable_fallback != key:
            pending = _PENDING_REQUESTS.pop(stable_fallback, None)
        if pending is None and fallback and fallback not in {key, stable_fallback}:
            pending = _PENDING_REQUESTS.pop(fallback, None)
        return pending


def _video_frames(video: Any):
    if video is None:
        return None
    # Native current ComfyUI VIDEO objects expose get_components().images.
    get_components = getattr(video, "get_components", None)
    if callable(get_components):
        components = get_components()
        frames = getattr(components, "images", None)
        if frames is None and isinstance(components, dict):
            frames = components.get("images")
        if frames is None:
            raise ValueError("Connected VIDEO did not provide image frames.")
        return frames
    # Compatibility: if a custom node supplies an IMAGE batch to a VIDEO-typed
    # bridge, accepting the tensor directly is harmless for the Local GGUF API.
    if hasattr(video, "shape"):
        return video
    raise TypeError("Unsupported VIDEO object. Connect a native ComfyUI VIDEO or convert it to IMAGE frames first.")


def _run_enhancement(
    prompt: str,
    enhancement_text: str,
    *,
    images: Any = None,
    video: Any = None,
    workflow_owned: bool = False,
    seed: Any = 0,
    settings: Any = None,
    batch_id: Any = "",
    yield_after: bool = False,
) -> dict[str, Any]:
    frames = _video_frames(video) if video is not None else None
    messages = _messages_for(
        prompt,
        enhancement_text,
        has_images=images is not None,
        has_video=frames is not None,
    )
    service = _find_local_llm_service()
    # When invoked from the partial ComfyUI execution started by the Enhance
    # button, use the service's workflow-client prefix so VRAM arbitration does
    # not wait on the very workflow that is currently executing this node.
    client = "ComfyUI Local LLM Generate - Prompt Enhancer" if workflow_owned else "Prompt Enhancer"
    overrides, runtime_config, config_snapshot = _batch_request_snapshot(batch_id, settings, seed)
    # Prompt Enhancer entries are independent rewrite requests, not turns in one
    # chat conversation. Keeping the previous completion in llama.cpp's resident
    # KV context can make later entries progressively more expensive to decode
    # even though the native model itself remains hot. Reset only the KV context
    # for each enhancement; this preserves resident GGUF/model allocations while
    # paying the very small prompt-prefill cost again.
    overrides["prompt_cache_mode"] = "Off"
    result = None
    try:
        result = service.generate_messages(
            messages,
            image=images,
            video_frames=frames,
            client=client,
            overrides=overrides,
            runtime_config=runtime_config,
            config_snapshot=config_snapshot,
        )
        revised = _clean_response((result or {}).get("response", ""))
        if not revised:
            thinking = str((result or {}).get("thinking", "") or "").strip()
            if thinking:
                raise RuntimeError(
                    "The Local LLM produced reasoning but no final prompt text. Review Thinking Mode, max tokens, or stop sequences."
                )
            raise RuntimeError("The Local LLM returned an empty prompt.")
        return {
            "prompt": revised,
            "tokens": int((result or {}).get("tokens") or 0),
            "info": (result or {}).get("info") or {},
        }
    finally:
        # Manual Enhance is commonly followed immediately by a GPU-heavy ComfyUI
        # workflow. Establish the v2 Local-LLM GPU handoff at a controlled
        # boundary *before* returning control to the UI rather than making the
        # next workflow discover native llama.cpp residency inside mm.free_memory().
        # Multi-item batches keep the model hot between items and hand off once in
        # batch_end instead. The handoff now works for both Auto Yield and
        # driver-managed/Keep Resident configurations.
        if bool(yield_after):
            try:
                log.info("[Local LLM Prompt Enhancer] Manual enhance requested deterministic post-call GPU handoff")
                service.gpu_handoff(reason="prompt-enhancer-complete")
            except Exception as suspend_exc:
                log.warning(
                    "[Local LLM Prompt Enhancer] Post-enhance GPU handoff failed: %s",
                    suspend_exc,
                )


def _publish_manual_batch_progress(
    node_id: Any,
    token: str,
    index: int,
    total: int,
    result: dict[str, Any],
    *,
    state_id: Any = "",
    runtime_scope: Any = "",
    used_images: bool = False,
    used_video: bool = False,
    used_settings: bool = False,
) -> None:
    """Best-effort browser progress while one ComfyUI queue item owns the batch.

    The final executed payload always contains the complete result list, so this
    event is only for responsive UI/history updates and is safe to miss (for
    example while a browser tab is throttled).
    """
    try:
        from server import PromptServer

        server = getattr(PromptServer, "instance", None)
        sender = getattr(server, "send_sync", None)
        if not callable(sender):
            return
        sender(
            "local_llm_prompt_enhancer_batch_progress",
            {
                "node_id": str(node_id or ""),
                "token": str(token or ""),
                "state_id": str(state_id or "").strip(),
                "runtime_scope": str(runtime_scope or "").strip(),
                "index": int(index),
                "total": int(total),
                "prompt": str((result or {}).get("prompt") or ""),
                "tokens": int((result or {}).get("tokens") or 0),
                "used_images": bool(used_images),
                "used_video": bool(used_video),
                "used_settings": bool(used_settings),
            },
        )
    except Exception:
        # Never fail an enhancement because a UI progress event could not be
        # delivered; the normal ComfyUI executed event remains authoritative.
        pass


def _parse_prompt_history(value: Any, enhanced_prompt: str = "") -> list[str]:
    try:
        raw = json.loads(str(value or "[]"))
    except Exception:
        raw = []
    if not isinstance(raw, list):
        raw = []
    history = [str(item) for item in raw if isinstance(item, (str, int, float))]
    if not history and str(enhanced_prompt or "").strip():
        history = [str(enhanced_prompt)]
    return history


def _normalize_history_index(value: Any, count: int) -> int:
    if count <= 0:
        return 0
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError):
        index = 0
    return max(0, min(index, count - 1))


def _manual_history_signature(history: list[str], index: int) -> tuple[tuple[str, ...], int]:
    clean = [str(item) for item in history]
    active = _normalize_history_index(index, len(clean))
    return tuple(clean), active


def _prune_manual_history_locked(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    stale = [
        node_id for node_id, state in _MANUAL_HISTORY_STATES.items()
        if now - float(state.get("created", 0.0)) > _MANUAL_HISTORY_TTL_SECONDS
    ]
    for node_id in stale:
        _MANUAL_HISTORY_STATES.pop(node_id, None)


def _store_manual_batch_history(node_id: Any, pending: dict[str, Any], prompts: list[str]) -> None:
    """Remember all pre/final batch history stages for already-queued workflows.

    While this single batch queue item is running the browser may receive zero,
    some, or all progress events before the user queues the next workflow. Each
    queued workflow therefore can contain any intermediate prompt-array stage.
    All such stages are safe to upgrade to the authoritative completed array.
    """
    key = str(node_id or "").strip()
    if not key:
        return

    history = _parse_prompt_history(pending.get("prompt_history_json", "[]"), pending.get("enhanced_prompt", ""))
    index = _normalize_history_index(pending.get("prompt_history_index", 0), len(history))
    visible = str(pending.get("enhanced_prompt", "") or "")
    if history:
        history[index] = visible

    accepted: set[tuple[tuple[str, ...], int]] = {_manual_history_signature(history, index)}
    overwrite = bool(pending.get("overwrite_enhanced", False))
    for batch_index, value in enumerate(prompts):
        text = str(value or "")
        if not text.strip():
            continue
        if batch_index == 0 and overwrite and history:
            history[index] = text
        else:
            history.append(text)
            index = len(history) - 1
        accepted.add(_manual_history_signature(history, index))

    with _MANUAL_HISTORY_LOCK:
        _prune_manual_history_locked()
        _MANUAL_HISTORY_STATES[key] = {
            "created": time.monotonic(),
            "accepted": accepted,
            "history": list(history),
            "index": _normalize_history_index(index, len(history)),
        }


def _reconcile_queued_manual_history(
    node_id: Any,
    history: list[str],
    index: int,
    visible_enhanced: str,
) -> tuple[list[str], int, str, bool]:
    """Upgrade a workflow snapshot queued while its Enhance batch was active."""
    key = str(node_id or "").strip()
    clean = [str(item) for item in history]
    active = _normalize_history_index(index, len(clean))
    visible = str(visible_enhanced or "")
    if clean:
        clean[active] = visible
    incoming = _manual_history_signature(clean, active)

    with _MANUAL_HISTORY_LOCK:
        _prune_manual_history_locked()
        state = _MANUAL_HISTORY_STATES.get(key)
        if not state or incoming not in state.get("accepted", set()):
            return clean, active, visible, False
        final_history = [str(item) for item in state.get("history", [])]
        final_index = _normalize_history_index(state.get("index", 0), len(final_history))

    final_visible = final_history[final_index] if final_history else visible
    return final_history, final_index, final_visible, True


def _parse_shuffle_state(value: Any, count: int, current_index: int) -> list[int]:
    try:
        raw = json.loads(str(value or "[]"))
    except Exception:
        raw = []
    if not isinstance(raw, list):
        return []
    seen = set()
    result = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= index < count and index != current_index and index not in seen:
            seen.add(index)
            result.append(index)
    return result


def _next_prompt_index(
    mode: str,
    count: int,
    current_index: int,
    shuffle_state: Any,
) -> tuple[int, list[int]]:
    if count <= 1:
        return current_index if count else 0, []

    mode = str(mode or "fixed").strip().lower()
    if mode == "increment":
        return (current_index + 1) % count, []
    if mode == "decrement":
        return (current_index - 1) % count, []

    # Prompt selection randomness is intentionally independent from the LLM
    # generation seed. Random/Shuffle behave like an internal Randomize control:
    # each new random choice / shuffle bag is seeded from fresh OS entropy.
    rng = random.SystemRandom()
    if mode == "random":
        return rng.randrange(count), []

    if mode == "shuffle":
        bag = _parse_shuffle_state(shuffle_state, count, current_index)
        if not bag:
            bag = [i for i in range(count) if i != current_index]
            rng.shuffle(bag)
        next_index = bag.pop(0) if bag else current_index
        return next_index, bag

    return current_index, _parse_shuffle_state(shuffle_state, count, current_index)


def _prompt_preset_names() -> list[str]:
    """Return shared Local LLM prompt-preset names without owning that storage here."""
    loader = _find_local_llm_export("text_preset_names")
    if not callable(loader):
        return []
    try:
        return [str(name) for name in loader("prompts")]
    except Exception as exc:
        log.warning("[Local LLM Prompt Enhancer] Could not enumerate shared prompt presets: %s", exc)
        return []


def _normalize_cycle_revision(value: Any) -> int:
    try:
        revision = int(value)
    except (TypeError, ValueError, OverflowError):
        revision = 0
    return max(0, min(revision, 0x7FFFFFFF))


def _prompt_cycle_signature(mode: str, history: list[str], revision: Any = 0) -> tuple[str, tuple[str, ...], int]:
    return (
        str(mode or "fixed").strip().lower(),
        tuple(str(item) for item in history),
        _normalize_cycle_revision(revision),
    )


def _clear_prompt_cycle_state(unique_id: Any, revision: Any = None) -> None:
    """Clear an old cursor without letting a delayed browser reset erase a newer run.

    When ``revision`` is supplied, a state already created with that same
    revision is newer than (or concurrent with) the reset request and must be
    preserved. Calls from backend-owned lifecycle changes omit the revision and
    clear unconditionally.
    """
    key = str(unique_id or "").strip()
    if not key:
        return
    requested_revision = None if revision is None else _normalize_cycle_revision(revision)
    with _PROMPT_CYCLE_LOCK:
        state = _PROMPT_CYCLE_STATES.get(key)
        if state is None:
            return
        if requested_revision is not None and _normalize_cycle_revision(state.get("revision", 0)) == requested_revision:
            return
        _PROMPT_CYCLE_STATES.pop(key, None)


def _prune_prompt_cycle_states_locked(now: float) -> None:
    stale = [
        key for key, state in _PROMPT_CYCLE_STATES.items()
        if now - float(state.get("updated_at") or 0.0) > _PROMPT_CYCLE_TTL_SECONDS
    ]
    for key in stale:
        _PROMPT_CYCLE_STATES.pop(key, None)


def _prompt_cycle_state_snapshot(
    unique_id: Any,
    mode: str,
    history: list[str],
    revision: Any = 0,
) -> dict[str, Any]:
    """Read the authoritative live cycle cursor without advancing it.

    A ComfyUI workflow-tab switch can make the browser miss the execution UI
    payload for a node that is no longer in the currently mounted graph.  The
    frontend uses this snapshot when the node becomes visible again.  State is
    returned only when the caller's mode/history exactly matches the signature
    that produced the backend cursor, so an old cursor can never overwrite a
    newly edited prompt array.
    """
    key = str(unique_id or "").strip()
    count = len(history)
    if not key or count <= 0:
        return {"valid": False}

    signature = _prompt_cycle_signature(mode, history, revision)
    now = time.monotonic()
    with _PROMPT_CYCLE_LOCK:
        _prune_prompt_cycle_states_locked(now)
        state = _PROMPT_CYCLE_STATES.get(key)
        if not state or state.get("signature") != signature:
            return {"valid": False}
        next_index = _normalize_history_index(state.get("next_index", 0), count)
        shuffle = [
            int(value) for value in (state.get("shuffle") or [])
            if isinstance(value, int) and 0 <= value < count and value != next_index
        ]
        return {
            "valid": True,
            "next_index": next_index,
            "shuffle": shuffle,
        }


def _advance_prompt_cycle_backend(
    unique_id: Any,
    mode: str,
    history: list[str],
    requested_index: int,
    shuffle_state: Any,
    revision: Any = 0,
) -> tuple[int, int, list[int]]:
    """Return (index used now, index for next run, next shuffle bag).

    The serialized UI index is authoritative when the history/mode changes or
    after an explicit frontend reset. Otherwise the backend cursor is
    authoritative, which makes rapid/auto-queued runs independent of browser
    focus and `executed` event timing.
    """
    count = len(history)
    requested = _normalize_history_index(requested_index, count)
    if count <= 0:
        _clear_prompt_cycle_state(unique_id)
        return 0, 0, []

    key = str(unique_id or "").strip()
    # UNIQUE_ID should always be present in ComfyUI. Keep a stateless fallback
    # for unusual direct/test invocations rather than sharing a global key.
    if not key:
        next_index, next_shuffle = _next_prompt_index(mode, count, requested, shuffle_state)
        return requested, next_index, next_shuffle

    cycle_revision = _normalize_cycle_revision(revision)
    signature = _prompt_cycle_signature(mode, history, cycle_revision)
    now = time.monotonic()
    with _PROMPT_CYCLE_LOCK:
        _prune_prompt_cycle_states_locked(now)
        state = _PROMPT_CYCLE_STATES.get(key)
        if not state or state.get("signature") != signature:
            current_index = requested
            current_shuffle: Any = shuffle_state
        else:
            current_index = _normalize_history_index(state.get("next_index", requested), count)
            current_shuffle = state.get("shuffle", [])

        next_index, next_shuffle = _next_prompt_index(
            mode, count, current_index, current_shuffle
        )
        _PROMPT_CYCLE_STATES[key] = {
            "signature": signature,
            "revision": cycle_revision,
            "next_index": next_index,
            "shuffle": list(next_shuffle),
            "updated_at": now,
        }
    return current_index, next_index, list(next_shuffle)


def _effective_prompt_text(prompt_preset: Any, prompt: Any) -> str:
    """Resolve a shared Prompt Preset exactly like Local LLM Generate."""
    source = str(prompt or "")
    name = str(prompt_preset or "Custom")
    if name == "Custom":
        return source
    loader = _find_local_llm_export("load_text_preset")
    if not callable(loader):
        return source
    try:
        saved = loader("prompts", name)
    except Exception as exc:
        log.warning("[Local LLM Prompt Enhancer] Could not load prompt preset %r: %s", name, exc)
        return source
    return source if saved is None else str(saved)


class LocalLLMPromptEnhancer:
    """Manual multimodal prompt enhancement through the persistent Local GGUF service."""

    @classmethod
    def INPUT_TYPES(cls):
        labels = _template_labels()
        default_selection = DEFAULT_SELECTION if DEFAULT_SELECTION in labels else labels[0]
        default_record = _template_by_label(default_selection)
        default_text = str((default_record or {}).get("text") or "")
        prompt_set_labels = _prompt_set_labels()
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": " ",
                        "tooltip": (
                            "Original/source prompt. Enhance never changes this field. "
                            "Use the ↑ control to explicitly replace it with the active Enhanced Prompt."
                        ),
                    },
                ),
                "enhanced_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": " ",
                        "tooltip": (
                            "Editable active entry from the generated prompt array. "
                            "Use ← / → to browse entries, × to delete the active entry, and Undo to restore the last array edit."
                        ),
                    },
                ),
                "prompt_set": (
                    prompt_set_labels,
                    {
                        "default": PROMPT_SET_NONE,
                        "tooltip": (
                            "Named saved set for the enhanced-prompt array. Selecting a set loads its prompts and active entry. "
                            "Save Prompt Set stores the current array and refreshes this selector automatically."
                        ),
                    },
                ),
                "prompt_cycle": (
                    ["fixed", "increment", "decrement", "shuffle", "random"],
                    {
                        "default": "fixed",
                        "tooltip": (
                            "How the stored enhanced-prompt array advances after a normal workflow run when Enhance with Workflow is off. "
                            "Fixed keeps the active entry; Increment/Decrement wrap; Shuffle avoids repeats until the bag is exhausted; "
                            "Random chooses freely."
                        ),
                    },
                ),
                "overwrite_enhanced": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "overwrite active",
                        "label_off": "add new",
                        "tooltip": (
                            "Controls where a newly generated enhancement is stored. Enabled replaces the active enhanced prompt; "
                            "disabled appends a new prompt and makes it active."
                        ),
                    },
                ),
                "enhancement_preset": (
                    labels,
                    {
                        "default": default_selection,
                        "tooltip": (
                            "Loads a protected Krea 2 / MiniMax H3 enhancement template or a user template "
                            "into the editable Enhancement Instructions field."
                        ),
                    },
                ),
                "enhancement_text": (
                    "STRING",
                    {
                        "default": default_text,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": " ",
                        "tooltip": (
                            "Editable instructions sent to the local LLM when Enhance is clicked or when "
                            "Enhance with Workflow is enabled. Selecting a preset loads its text here; edits remain local until saved."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Standard ComfyUI request seed for prompt enhancement. Prompt Cycle Random/Shuffle use "
                            "fresh internal randomness and do not depend on this seed. The linked control supports fixed, "
                            "increment, decrement, and randomize behavior."
                        ),
                    },
                ),
                "enhance_with_workflow": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "enabled",
                        "label_off": "disabled",
                        "tooltip": (
                            "When enabled, normal workflow execution runs the Local GGUF enhancement using the current Prompt, "
                            "Enhancement Instructions, seed, and connected image/video inputs, then sends the fresh enhanced prompt downstream. "
                            "When disabled, the active stored enhanced prompt is used and Prompt Cycle controls the next active entry."
                        ),
                    },
                ),
                "prompt_history_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                "prompt_history_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                    },
                ),
                "prompt_cycle_revision": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                    },
                ),
                "prompt_shuffle_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                # Keep this appended after the existing serialized widgets so old
                # Prompt Enhancer workflows retain their widget-value positions.
                # The frontend renders this selector above Prompt.
                "prompt_preset": (
                    ["Custom", *_prompt_preset_names()],
                    {
                        "default": "Custom",
                        "tooltip": (
                            "Reusable prompts shared with Local LLM Generate from "
                            "models/LLM/local_LLM_presets/prompts. Editing Prompt switches this selector to Custom."
                        ),
                    },
                ),
                # Stable frontend-owned identity used to reconcile background
                # executions when this workflow is not currently mounted. Keep
                # appended after legacy serialized widgets for compatibility.
                "prompt_state_id": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                # Browser-session workflow ownership. This is deliberately
                # overwritten by the frontend whenever a workflow is mounted;
                # serialized values from workflow images are never authoritative.
                "prompt_runtime_scope": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional still image or IMAGE batch used as visual reference while enhancing. "
                            "The Enhance button partially executes this node so connected image tensors are available immediately."
                        ),
                    },
                ),
                "video": (
                    "VIDEO",
                    {
                        "tooltip": (
                            "Optional native ComfyUI VIDEO reference. Frames are extracted and passed to the Local GGUF vision model "
                            "using the server's current vision frame sampling limits."
                        ),
                    },
                ),
                "settings": (
                    "LOCAL_LLM_SETTINGS",
                    {
                        "tooltip": (
                            "Optional Local LLM Settings node. When connected, it supplies sampler, vision-limit, and runtime values. "
                            "Seed remains owned by this Prompt Enhancer node and its local Seed controls stay active."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "prompt")
    FUNCTION = "output_prompts"
    # Manual media-aware enhancement uses ComfyUI partial execution targeted at
    # this node. ComfyUI only accepts terminal/output nodes as partial-execution
    # targets, so mark this as an output node. Normal workflow execution remains
    # passive unless Enhance with Workflow is enabled.
    OUTPUT_NODE = True
    CATEGORY = "prompt/Local LLM"
    DESCRIPTION = (
        "Standalone multimodal prompt enhancer using the persistent ComfyUI Local GGUF LLM service. "
        "Prompt stays unchanged while enhancement writes to an editable Enhanced Prompt. Use the manual Enhance control, "
        "or enable Enhance with Workflow to generate a fresh enhanced prompt during normal workflow execution."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # The manual Enhance action arms an out-of-band request that is not part
        # of serialized widget inputs. Force this lightweight node to execute so
        # a repeated enhancement with identical graph values cannot be satisfied
        # from ComfyUI's output cache.
        return float("nan")

    def output_prompts(
        self,
        prompt: str,
        enhanced_prompt: str = "",
        prompt_set: str = PROMPT_SET_NONE,
        prompt_cycle: str = "fixed",
        overwrite_enhanced: bool = False,
        enhancement_preset: str = "",
        enhancement_text: str = "",
        seed: int = 0,
        enhance_with_workflow: bool = False,
        prompt_history_json: str = "[]",
        prompt_history_index: int = 0,
        prompt_cycle_revision: int = 0,
        prompt_shuffle_json: str = "[]",
        prompt_preset: str = "Custom",
        prompt_state_id: str = "",
        prompt_runtime_scope: str = "",
        settings: Any = None,
        images: Any = None,
        video: Any = None,
        unique_id: Any = None,
    ):
        source = _effective_prompt_text(prompt_preset, prompt)
        history = _parse_prompt_history(prompt_history_json, enhanced_prompt)
        ui_active_index = _normalize_history_index(prompt_history_index, len(history))
        visible_enhanced = str(enhanced_prompt or "")
        # Normal ComfyUI workflows may have been queued while this node's manual
        # Enhance batch was still executing. Their serialized widget snapshot can
        # therefore contain the pre-batch or an intermediate prompt array. Upgrade
        # only a recognized batch stage to the authoritative completed array; all
        # unrelated/manual edits remain untouched.
        state_key = _enhancer_state_key(unique_id, prompt_state_id, prompt_runtime_scope)
        history, ui_active_index, visible_enhanced, _queued_batch_reconciled = _reconcile_queued_manual_history(
            state_key, history, ui_active_index, visible_enhanced
        )
        effective_seed = _effective_seed(settings, seed)

        # A manual media/settings-aware Enhance request always takes priority. The
        # browser explicitly armed this targeted execution so connected media
        # can reach the Local GGUF service.
        pending = _pop_request(unique_id, prompt_state_id, prompt_runtime_scope)
        if pending is not None:
            _clear_prompt_cycle_state(state_key)
            token = str(pending.get("token") or "")
            batch_id = str(pending.get("batch_id", "") or "").strip()
            raw_seeds = pending.get("seeds")
            if isinstance(raw_seeds, list) and raw_seeds:
                batch_seeds = [_normalize_seed(value) for value in raw_seeds[:64]]
            else:
                batch_seeds = [_normalize_seed(pending.get("seed", effective_seed))]

            batch_results: list[dict[str, Any]] = []
            try:
                # The complete manual batch intentionally runs *inside this one
                # ComfyUI partial-execution queue item*. No later enhancement is
                # appended to the queue by JavaScript, so a normal workflow can
                # safely wait behind the batch as a single atomic GPU owner.
                for batch_index, batch_seed in enumerate(batch_seeds):
                    result = _run_enhancement(
                        source,
                        str(pending.get("enhancement_text") or ""),
                        images=images,
                        video=video,
                        workflow_owned=True,
                        seed=batch_seed,
                        settings=settings,
                        batch_id=batch_id,
                        yield_after=False,
                    )
                    batch_results.append(result)
                    _publish_manual_batch_progress(
                        unique_id,
                        token,
                        batch_index,
                        len(batch_seeds),
                        result,
                        state_id=pending.get("state_id") or prompt_state_id,
                        runtime_scope=pending.get("runtime_scope") or prompt_runtime_scope,
                        used_images=images is not None,
                        used_video=video is not None,
                        used_settings=isinstance(settings, dict),
                    )
            finally:
                # Yield native llama.cpp VRAM before this ComfyUI queue item is
                # allowed to finish. Therefore the next queued diffusion job can
                # never begin while the enhancer is still tearing down GPU state.
                if batch_id:
                    _end_batch(batch_id)
                try:
                    service = _find_local_llm_service()
                    _handoff_after_enhancer_batch(service, reason="prompt-enhancer-batch-complete")
                except Exception as suspend_exc:
                    log.warning(
                        "[Local LLM Prompt Enhancer] In-job post-batch GPU handoff failed: %s",
                        suspend_exc,
                    )

            prompts = [str(item.get("prompt") or "") for item in batch_results]
            token_counts = [int(item.get("tokens") or 0) for item in batch_results]
            # Preserve the completed history server-side so workflows that were
            # already added to ComfyUI's queue during this batch can consume the
            # finished enhancements even though their prompt snapshot was taken
            # earlier. This does not delay or intercept queue submission.
            _store_manual_batch_history(state_key, pending, prompts)
            payload = {
                "mode": "manual_batch",
                "token": token,
                "state_id": str(pending.get("state_id") or prompt_state_id or ""),
                "runtime_scope": str(pending.get("runtime_scope") or prompt_runtime_scope or ""),
                "prompt": prompts[-1] if prompts else "",
                "prompts": prompts,
                "tokens": sum(token_counts),
                "token_counts": token_counts,
                "batch_count": len(prompts),
                "overwrite": bool(pending.get("overwrite_enhanced", False)),
                "used_images": images is not None,
                "used_video": video is not None,
                "used_settings": isinstance(settings, dict),
            }
            final_manual_prompt = prompts[-1] if prompts and prompts[-1].strip() else (visible_enhanced if visible_enhanced.strip() else source)
            return {
                "ui": {"prompt_enhancer": [json.dumps(payload, ensure_ascii=False)]},
                "result": (final_manual_prompt, source),
            }

        if bool(enhance_with_workflow):
            _clear_prompt_cycle_state(state_key)
            result = _run_enhancement(
                source,
                str(enhancement_text or ""),
                images=images,
                video=video,
                workflow_owned=True,
                seed=effective_seed,
                settings=settings,
            )
            revised = str(result["prompt"] or "")
            payload = {
                "mode": "workflow",
                "state_id": str(prompt_state_id or ""),
                "runtime_scope": str(prompt_runtime_scope or ""),
                "prompt": revised,
                "tokens": result["tokens"],
                "used_images": images is not None,
                "used_video": video is not None,
                "used_settings": isinstance(settings, dict),
                "overwrite": bool(overwrite_enhanced),
            }
            return {
                "ui": {"prompt_enhancer": [json.dumps(payload, ensure_ascii=False)]},
                "result": (revised, source),
            }

        active_index, next_index, next_shuffle = _advance_prompt_cycle_backend(
            state_key,
            prompt_cycle,
            history,
            ui_active_index,
            prompt_shuffle_json,
            prompt_cycle_revision,
        )
        active_enhanced = history[active_index] if history else visible_enhanced
        effective = active_enhanced if active_enhanced.strip() else source
        payload = {
            "mode": "cycle",
            "state_id": str(prompt_state_id or ""),
            "runtime_scope": str(prompt_runtime_scope or ""),
            "active_index": active_index,
            "next_index": next_index,
            "shuffle": next_shuffle,
            "used_settings": isinstance(settings, dict),
            "backend_owned": True,
        }
        return {
            "ui": {"prompt_enhancer": [json.dumps(payload, ensure_ascii=False)]},
            "result": (effective, source),
        }


try:
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    def _json_error(message: Any, status: int = 400):
        return web.json_response(
            {"error": {"message": str(message), "type": "local_llm_prompt_enhancer_error"}},
            status=status,
        )

    @routes.get("/local_llm_prompt_enhancer/templates")
    async def local_llm_prompt_enhancer_templates(_request):
        try:
            return web.json_response({
                "templates": _template_records(),
                "default": DEFAULT_SELECTION,
                "user_directory": str(USER_TEMPLATE_DIR),
                "default_directory": str(DEFAULT_TEMPLATE_DIR),
            })
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Template listing failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/save_template")
    async def local_llm_prompt_enhancer_save_template(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            name = str(body.get("name") or "")
            text = str(body.get("text") or "")
            saved = await asyncio.to_thread(_save_user_template, name, text)
            log.info("[Local LLM Prompt Enhancer] Saved user template: %s", saved["label"])
            # Return a fresh list so the frontend can update its dropdown without
            # a second refresh button or manual reload.
            return web.json_response({
                "saved": saved,
                "templates": _template_records(),
                "default": DEFAULT_SELECTION,
            })
        except ValueError as exc:
            return _json_error(exc, 400)
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Template save failed")
            return _json_error(exc, 500)


    @routes.post("/local_llm_prompt_enhancer/delete_template")
    async def local_llm_prompt_enhancer_delete_template(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            deleted = await asyncio.to_thread(_delete_user_template, str(body.get("label") or ""))
            log.info("[Local LLM Prompt Enhancer] Deleted user template: %s", deleted["label"])
            return web.json_response({
                "deleted": deleted,
                "templates": _template_records(),
                "default": DEFAULT_SELECTION,
            })
        except ValueError as exc:
            return _json_error(exc, 400)
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Template delete failed")
            return _json_error(exc, 500)

    @routes.get("/local_llm_prompt_enhancer/prompt_sets")
    async def local_llm_prompt_enhancer_prompt_sets(_request):
        try:
            records = await asyncio.to_thread(_prompt_set_records)
            return web.json_response({"sets": [{"name": r["name"], "count": len(r["prompts"]), "active_index": r["active_index"]} for r in records]})
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Prompt set listing failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/save_prompt_set")
    async def local_llm_prompt_enhancer_save_prompt_set(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            saved = await asyncio.to_thread(
                _save_prompt_set,
                str(body.get("name") or ""),
                body.get("prompts"),
                body.get("active_index", 0),
            )
            return web.json_response({
                "saved": saved,
                "sets": [{"name": r["name"], "count": len(r["prompts"]), "active_index": r["active_index"]} for r in _prompt_set_records()],
            })
        except ValueError as exc:
            return _json_error(exc, 400)
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Prompt set save failed")
            return _json_error(exc, 500)


    @routes.post("/local_llm_prompt_enhancer/delete_prompt_set")
    async def local_llm_prompt_enhancer_delete_prompt_set(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            deleted = await asyncio.to_thread(_delete_prompt_set, str(body.get("name") or ""))
            return web.json_response({
                "deleted": deleted,
                "sets": [{"name": r["name"], "count": len(r["prompts"]), "active_index": r["active_index"]} for r in _prompt_set_records()],
            })
        except ValueError as exc:
            return _json_error(exc, 400)
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Prompt Set delete failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/load_prompt_set")
    async def local_llm_prompt_enhancer_load_prompt_set(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            loaded = await asyncio.to_thread(_load_prompt_set, str(body.get("name") or ""))
            return web.json_response({"set": loaded})
        except ValueError as exc:
            return _json_error(exc, 400)
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Prompt set load failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/batch_begin")
    async def local_llm_prompt_enhancer_batch_begin(_request):
        try:
            batch_id = await asyncio.to_thread(_begin_batch)
            return web.json_response({"batch_id": batch_id})
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Batch snapshot failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/batch_end")
    async def local_llm_prompt_enhancer_batch_end(request):
        try:
            body = await request.json()
            ended = _end_batch((body or {}).get("batch_id", ""))
            if ended:
                try:
                    service = _find_local_llm_service()
                    await asyncio.to_thread(
                        _handoff_after_enhancer_batch,
                        service,
                        reason="prompt-enhancer-batch-complete",
                    )
                except Exception as suspend_exc:
                    log.warning(
                        "[Local LLM Prompt Enhancer] Post-batch GPU handoff failed: %s",
                        suspend_exc,
                    )
            return web.json_response({"ended": bool(ended)})
        except Exception as exc:
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/arm")
    async def local_llm_prompt_enhancer_arm(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            token = _arm_request(
                str(body.get("node_id") or ""),
                str(body.get("prompt") or ""),
                str(body.get("enhancement_text") or ""),
                body.get("seed", 0),
                body.get("batch_id", ""),
                body.get("seeds"),
                body.get("batch_count", 1),
                body.get("prompt_history_json", "[]"),
                body.get("prompt_history_index", 0),
                body.get("enhanced_prompt", ""),
                body.get("overwrite_enhanced", False),
                body.get("state_id", ""),
                body.get("runtime_scope", ""),
            )
            return web.json_response({"token": token})
        except ValueError as exc:
            return _json_error(exc, 400)
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Arm request failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/cycle_reset")
    async def local_llm_prompt_enhancer_cycle_reset(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            revision = body.get("revision") if isinstance(body, dict) and "revision" in body else None
            _clear_prompt_cycle_state(_enhancer_state_key(body.get("node_id"), body.get("state_id"), body.get("runtime_scope")), revision)
            return web.json_response({"ok": True})
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Prompt-cycle reset failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/cycle_state")
    async def local_llm_prompt_enhancer_cycle_state(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            history_raw = body.get("history", [])
            if not isinstance(history_raw, list):
                raise TypeError("history must be an array")
            history = [str(item) for item in history_raw]
            snapshot = _prompt_cycle_state_snapshot(
                _enhancer_state_key(body.get("node_id"), body.get("state_id"), body.get("runtime_scope")),
                str(body.get("mode") or "fixed"),
                history,
            )
            return web.json_response(snapshot)
        except (TypeError, ValueError) as exc:
            return _json_error(exc, 400)
        except Exception as exc:
            log.exception("[Local LLM Prompt Enhancer] Prompt-cycle state read failed")
            return _json_error(exc, 500)

    @routes.post("/local_llm_prompt_enhancer/cancel")
    async def local_llm_prompt_enhancer_cancel(request):
        try:
            body = await request.json()
            node_id = str((body or {}).get("node_id") or "")
            state_id = str((body or {}).get("state_id") or "")
            runtime_scope = str((body or {}).get("runtime_scope") or "")
            token = str((body or {}).get("token") or "")
            key = _enhancer_state_key(node_id, state_id, runtime_scope)
            stable_fallback = _enhancer_state_key(node_id, state_id, "")
            fallback = _enhancer_state_key(node_id, "", "")
            removed = False
            with _PENDING_LOCK:
                for candidate_key in dict.fromkeys([key, stable_fallback, fallback]):
                    if not candidate_key:
                        continue
                    current = _PENDING_REQUESTS.get(candidate_key)
                    if current and (not token or str(current.get("token")) == token):
                        _PENDING_REQUESTS.pop(candidate_key, None)
                        removed = True
                        break
            return web.json_response({"cancelled": removed})
        except Exception as exc:
            return _json_error(exc, 500)

    # v0.18.52 keeps the old text-only direct HTTP execution path retired because it
    # ran llama.cpp outside ComfyUI's prompt queue and could overlap a newly
    # started diffusion workflow. The frontend now uses the same arm + targeted
    # partial-execution transport for text, image, video, and Settings requests.
    # Keep a guarded endpoint only so stale cached v0.18.47 JavaScript fails safe
    # instead of recreating the GPU race.
    @routes.post("/local_llm_prompt_enhancer/run")
    async def local_llm_prompt_enhancer_run(_request):
        return _json_error(
            "Direct Prompt Enhancer execution was retired for GPU safety. Refresh ComfyUI so the current frontend can use targeted queued execution.",
            409,
        )

except Exception as route_error:
    log.warning("[Local LLM Prompt Enhancer] HTTP routes unavailable: %s", route_error)
