from __future__ import annotations

import unittest

from generation_controls import resolve_generation_controls


CURRENT = {
    "thinking_mode": "Enabled",
    "reasoning_effort": "medium",
    "preserve_thinking": True,
    "chat_format": "Auto",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "typical_p": 1.0,
    "repeat_penalty": 1.05,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "tfs_z": 1.0,
    "mirostat_mode": "Off",
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
}

PRESETS = {
    "Family Preset": {
        "thinking_mode": "Disabled",
        "temperature": 0.05,
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.1,
    }
}


class GenerationControlTests(unittest.TestCase):
    def test_model_preset_applies_when_request_has_no_override(self):
        result = resolve_generation_controls(
            model_preset="Auto (Detected)",
            model_preset_resolved="Family Preset",
            current=CURRENT,
            model_presets=PRESETS,
        )
        self.assertEqual(result["temperature"], 0.05)
        self.assertEqual(result["top_p"], 0.8)
        self.assertEqual(result["thinking_mode"], "Disabled")
        self.assertEqual(result["min_p"], CURRENT["min_p"])

    def test_explicit_request_controls_win_after_model_preset(self):
        result = resolve_generation_controls(
            model_preset="Auto (Detected)",
            model_preset_resolved="Family Preset",
            current=CURRENT,
            model_presets=PRESETS,
            request_overrides={
                "temperature": 1.1,
                "top_p": 0.97,
                "reasoning_effort": "high",
            },
        )
        self.assertEqual(result["temperature"], 1.1)
        self.assertEqual(result["top_p"], 0.97)
        self.assertEqual(result["reasoning_effort"], "high")
        # Fields not explicitly supplied by the request still come from preset.
        self.assertEqual(result["top_k"], 20)
        self.assertEqual(result["thinking_mode"], "Disabled")

    def test_custom_model_preset_leaves_current_values_then_request_overrides(self):
        result = resolve_generation_controls(
            model_preset="Custom",
            model_preset_resolved="Family Preset",
            current=CURRENT,
            model_presets=PRESETS,
            request_overrides={"temperature": 0.9},
        )
        self.assertEqual(result["top_p"], CURRENT["top_p"])
        self.assertEqual(result["temperature"], 0.9)

    def test_unknown_request_fields_are_ignored(self):
        result = resolve_generation_controls(
            model_preset="Auto (Detected)",
            model_preset_resolved="Family Preset",
            current=CURRENT,
            model_presets=PRESETS,
            request_overrides={"temperature": 1.0, "seed": 123, "model": "bad"},
        )
        self.assertEqual(result["temperature"], 1.0)
        self.assertNotIn("seed", result)
        self.assertNotIn("model", result)


if __name__ == "__main__":
    unittest.main()
