"""Pure generation-control resolution shared by runtime tests.

Model presets provide defaults for model-family-specific sampling/reasoning
behavior. Request-local callers may explicitly override some of those controls;
those explicit values must be applied after the preset, not before it.
"""
from __future__ import annotations

import copy

MODEL_PRESET_EXECUTION_FIELDS = (
    "thinking_mode",
    "reasoning_effort",
    "preserve_thinking",
    "chat_format",
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
)

REQUEST_GENERATION_OVERRIDE_FIELDS = frozenset(MODEL_PRESET_EXECUTION_FIELDS)


def resolve_generation_controls(
    *,
    model_preset: str,
    model_preset_resolved: str,
    current: dict,
    model_presets: dict,
    request_overrides: dict | None = None,
) -> dict:
    """Return effective controls with precedence current < preset < request."""
    resolved = {
        key: copy.deepcopy(current.get(key))
        for key in MODEL_PRESET_EXECUTION_FIELDS
    }

    if model_preset_resolved in model_presets and model_preset != "Custom":
        preset = model_presets[model_preset_resolved]
        for key in MODEL_PRESET_EXECUTION_FIELDS:
            if key in preset:
                resolved[key] = copy.deepcopy(preset[key])

    if isinstance(request_overrides, dict):
        for key, value in request_overrides.items():
            if key in REQUEST_GENERATION_OVERRIDE_FIELDS:
                resolved[key] = copy.deepcopy(value)

    return resolved
