"""Pure OpenAI-compatible generation-control helpers.

Keep request-field extraction and diagnostics outside ``service.py`` so the API
contract can be tested without importing ComfyUI, aiohttp, or the model runtime.
The server still owns seed allocation and transport details.
"""
from __future__ import annotations

import threading

# OpenAI-standard fields plus commonly supported local-LLM extensions.  The
# destination names are the canonical LocalGGUFLLM generation arguments.
OPENAI_GENERATION_FIELD_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "typical_p": "typical_p",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "max_tokens": "max_tokens",
    "reasoning_effort": "reasoning_effort",
    "tfs_z": "tfs_z",
    "mirostat_mode": "mirostat_mode",
    "mirostat_tau": "mirostat_tau",
    "mirostat_eta": "mirostat_eta",
}

# Fields worth printing in the resolved sampler diagnostic. ``max_tokens`` is
# technically an output limit rather than a sampler, but including it makes the
# client/server boundary visible in one line.
OPENAI_DIAGNOSTIC_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "tfs_z",
    "mirostat_mode",
    "mirostat_tau",
    "mirostat_eta",
    "max_tokens",
)


class OpenAIRequestSequence:
    """Small thread-safe process-local request sequence for log correlation."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


def extract_openai_generation_overrides(body: dict | None) -> dict:
    """Extract explicit request-local generation controls.

    Missing and ``null`` values intentionally do not override the server/model
    preset.  Zero and false-like numeric values remain explicit overrides.
    """
    if not isinstance(body, dict):
        return {}
    out = {}
    for source_name, destination_name in OPENAI_GENERATION_FIELD_MAP.items():
        if source_name in body and body[source_name] is not None:
            out[destination_name] = body[source_name]
    return out


def openai_control_sources(body: dict | None) -> dict[str, str]:
    """Describe whether each effective control came from API or server/preset."""
    body = body if isinstance(body, dict) else {}
    sources: dict[str, str] = {}
    for source_name, destination_name in OPENAI_GENERATION_FIELD_MAP.items():
        sources[destination_name] = (
            "api" if source_name in body and body[source_name] is not None else "server/preset"
        )
    # These are effective runtime controls but have no accepted OpenAI request
    # field in this service. Include them when they appear in runtime info.
    for name in OPENAI_DIAGNOSTIC_FIELDS:
        sources.setdefault(name, "server/preset")
    return sources


def effective_openai_sampling(info: dict | None, sources: dict | None = None) -> dict[str, dict]:
    """Return effective values paired with their request/server source.

    The runtime result is authoritative for values.  Source labels answer the
    separate question of whether the HTTP request explicitly supplied a value.
    """
    info = info if isinstance(info, dict) else {}
    sources = sources if isinstance(sources, dict) else {}
    effective = info.get("effective_model_settings") or {}
    independent = info.get("independent_output_controls") or {}
    result: dict[str, dict] = {}
    for name in OPENAI_DIAGNOSTIC_FIELDS:
        if name == "max_tokens":
            value = independent.get(name)
        else:
            value = effective.get(name)
        if value is None:
            continue
        result[name] = {
            "value": value,
            "source": str(sources.get(name) or "server/preset"),
        }
    return result


def format_openai_sampling(snapshot: dict | None) -> list[str]:
    """Format a sampling snapshot as stable ``name=value[source]`` fields."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    fields: list[str] = []
    for name in OPENAI_DIAGNOSTIC_FIELDS:
        item = snapshot.get(name)
        if not isinstance(item, dict) or "value" not in item:
            continue
        fields.append(f"{name}={item['value']}[{item.get('source') or 'server/preset'}]")
    return fields
