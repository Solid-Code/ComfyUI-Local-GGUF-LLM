from __future__ import annotations

import unittest

from api_diagnostics import OpenAIRepeatTracker, openai_request_signature, stable_text_hash


class ApiDiagnosticsTests(unittest.TestCase):
    def test_signature_ignores_seed_and_stream_but_not_prompt(self):
        base = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.88}
        a = openai_request_signature({**base, "seed": 1, "stream": False})
        b = openai_request_signature({**base, "seed": 2, "stream": True})
        c = openai_request_signature({**base, "messages": [{"role": "user", "content": "different"}]})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_repeat_tracker_flags_same_visible_output_under_different_seeds(self):
        tracker = OpenAIRepeatTracker(limit=8)
        first = tracker.observe(request_signature="abc", request_id=1, sampler_seed=111, response="same", thinking="r1")
        second = tracker.observe(request_signature="abc", request_id=2, sampler_seed=222, response="same", thinking="r2")
        self.assertIsNone(first["matched_previous"])
        self.assertEqual(second["matched_previous"]["request_id"], 1)
        self.assertFalse(second["thinking_same_as_previous"])

    def test_repeat_tracker_does_not_flag_same_seed_or_different_request(self):
        tracker = OpenAIRepeatTracker(limit=8)
        tracker.observe(request_signature="abc", request_id=1, sampler_seed=111, response="same")
        same_seed = tracker.observe(request_signature="abc", request_id=2, sampler_seed=111, response="same")
        different_request = tracker.observe(request_signature="def", request_id=3, sampler_seed=222, response="same")
        self.assertIsNone(same_seed["matched_previous"])
        self.assertIsNone(different_request["matched_previous"])

    def test_hash_is_stable_and_content_sensitive(self):
        self.assertEqual(stable_text_hash("x"), stable_text_hash("x"))
        self.assertNotEqual(stable_text_hash("x"), stable_text_hash("y"))


if __name__ == "__main__":
    unittest.main()
