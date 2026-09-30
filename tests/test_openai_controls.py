from __future__ import annotations

import unittest

from generation_controls import resolve_generation_controls
from openai_controls import (
    OpenAIRequestSequence,
    effective_openai_sampling,
    extract_openai_generation_overrides,
    format_openai_sampling,
    openai_control_sources,
)


SERVER = {
    "thinking_mode": "Enabled",
    "reasoning_effort": "low",
    "preserve_thinking": True,
    "chat_format": "Auto",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
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

SILLYTAVERN_BODY = {
    "temperature": 0.88,
    "frequency_penalty": 0.22,
    "presence_penalty": 0.78,
    "top_p": 1.0,
    "max_tokens": 4096,
    "seed": -1,
    "stream": True,
}


class OpenAIControlTests(unittest.TestCase):
    def test_sillytavern_request_extracts_only_explicit_generation_controls(self):
        overrides = extract_openai_generation_overrides(SILLYTAVERN_BODY)
        self.assertEqual(overrides["temperature"], 0.88)
        self.assertEqual(overrides["top_p"], 1.0)
        self.assertEqual(overrides["frequency_penalty"], 0.22)
        self.assertEqual(overrides["presence_penalty"], 0.78)
        self.assertEqual(overrides["max_tokens"], 4096)
        self.assertNotIn("seed", overrides)
        self.assertNotIn("stream", overrides)
        self.assertNotIn("top_k", overrides)

    def test_sillytavern_request_wins_only_where_it_explicitly_supplies_values(self):
        overrides = extract_openai_generation_overrides(SILLYTAVERN_BODY)
        effective = resolve_generation_controls(
            model_preset="Custom",
            model_preset_resolved="Custom",
            current=SERVER,
            model_presets={},
            request_overrides=overrides,
        )
        self.assertEqual(effective["temperature"], 0.88)
        self.assertEqual(effective["top_p"], 1.0)
        self.assertEqual(effective["frequency_penalty"], 0.22)
        self.assertEqual(effective["presence_penalty"], 0.78)
        # SillyTavern did not send these, so server values remain authoritative.
        self.assertEqual(effective["top_k"], 20)
        self.assertEqual(effective["min_p"], 0.05)
        self.assertEqual(effective["typical_p"], 1.0)

    def test_sillytavern_source_map_makes_hidden_server_fallbacks_visible(self):
        sources = openai_control_sources(SILLYTAVERN_BODY)
        for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "max_tokens"):
            self.assertEqual(sources[name], "api")
        for name in ("top_k", "min_p", "typical_p", "repeat_penalty"):
            self.assertEqual(sources[name], "server/preset")

    def test_null_control_is_server_fallback_but_zero_is_explicit_api_value(self):
        body = {"temperature": 0, "top_p": None, "top_k": 0}
        overrides = extract_openai_generation_overrides(body)
        sources = openai_control_sources(body)
        self.assertEqual(overrides["temperature"], 0)
        self.assertEqual(overrides["top_k"], 0)
        self.assertNotIn("top_p", overrides)
        self.assertEqual(sources["temperature"], "api")
        self.assertEqual(sources["top_k"], "api")
        self.assertEqual(sources["top_p"], "server/preset")

    def test_local_extension_typical_p_is_request_override_when_explicit(self):
        overrides = extract_openai_generation_overrides({"typical_p": 0.93})
        self.assertEqual(overrides, {"typical_p": 0.93})
        self.assertEqual(openai_control_sources({"typical_p": 0.93})["typical_p"], "api")

    def test_advanced_local_sampler_extensions_are_extracted_explicitly(self):
        body = {
            "tfs_z": 0.9,
            "mirostat_mode": "v2",
            "mirostat_tau": 4.5,
            "mirostat_eta": 0.2,
        }
        overrides = extract_openai_generation_overrides(body)
        self.assertEqual(overrides, body)
        sources = openai_control_sources(body)
        for name in body:
            self.assertEqual(sources[name], "api")

    def test_effective_snapshot_uses_runtime_values_and_request_sources(self):
        info = {
            "effective_model_settings": {
                "temperature": 0.88,
                "top_p": 1.0,
                "top_k": 20,
                "min_p": 0.05,
                "typical_p": 1.0,
                "repeat_penalty": 1.05,
                "presence_penalty": 0.78,
                "frequency_penalty": 0.22,
            },
            "independent_output_controls": {"max_tokens": 4096},
        }
        snapshot = effective_openai_sampling(info, openai_control_sources(SILLYTAVERN_BODY))
        self.assertEqual(snapshot["temperature"], {"value": 0.88, "source": "api"})
        self.assertEqual(snapshot["top_k"], {"value": 20, "source": "server/preset"})
        self.assertEqual(snapshot["max_tokens"], {"value": 4096, "source": "api"})
        formatted = format_openai_sampling(snapshot)
        self.assertIn("temperature=0.88[api]", formatted)
        self.assertIn("top_p=1.0[api]", formatted)
        self.assertIn("top_k=20[server/preset]", formatted)
        self.assertIn("max_tokens=4096[api]", formatted)

    def test_request_sequence_is_monotonic(self):
        seq = OpenAIRequestSequence()
        self.assertEqual([seq.next(), seq.next(), seq.next()], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
