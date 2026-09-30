from __future__ import annotations

import asyncio
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "local_gguf_testpkg"


def load_prompt_enhancer_module():
    # Load prompt_enhancer.py as a synthetic package module so its relative
    # imports work without importing the real ComfyUI-facing __init__.py.
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = tempfile.gettempdir()
    sys.modules["folder_paths"] = folder_paths

    service = types.ModuleType(f"{PACKAGE_NAME}.service")
    service.SERVICE = types.SimpleNamespace()
    service.load_sampler_preset = lambda *_args, **_kwargs: None
    service.load_text_preset = lambda *_args, **_kwargs: None
    service.text_preset_names = lambda *_args, **_kwargs: []
    sys.modules[f"{PACKAGE_NAME}.service"] = service

    class FakeRoutes:
        def _decorator(self, _path):
            def decorator(function):
                return function
            return decorator

        post = _decorator
        get = _decorator

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=FakeRoutes()))
    sys.modules["server"] = server

    name = f"{PACKAGE_NAME}.prompt_enhancer"
    spec = importlib.util.spec_from_file_location(name, ROOT / "prompt_enhancer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PE = load_prompt_enhancer_module()


class PromptEnhancerIntegrationTests(unittest.TestCase):
    def setUp(self):
        PE.PROMPT_CYCLE_STORE._states.clear()
        PE._PENDING_ENHANCEMENTS.clear()
        PE._MANUAL_HISTORY_STATES.clear()

    def run_node(
        self,
        *,
        mode="fixed",
        index=0,
        revision=1,
        history=None,
        enhanced="A",
        state_id="node-4",
        scope="wf-a",
        queue_seq=0,
        shuffle_json="[]",
    ):
        if history is None:
            history = ["A", "B", "C"]
        node = PE.LocalLLMPromptEnhancer()
        return node.output_prompts(
            prompt="source",
            enhanced_prompt=enhanced,
            prompt_cycle=mode,
            prompt_history_json=json.dumps(history),
            prompt_history_index=index,
            prompt_cycle_revision=revision,
            prompt_shuffle_json=shuffle_json,
            prompt_state_id=state_id,
            prompt_runtime_scope=scope,
            prompt_cycle_queue_seq=queue_seq,
            unique_id="4",
        )

    def test_fixed_uses_selected_xy(self):
        fixed = self.run_node(mode="fixed", index=2, enhanced="C")
        self.assertEqual(fixed["result"], ("C", "source"))
        payload = json.loads(fixed["ui"]["prompt_enhancer"][0])
        self.assertEqual(payload["mode"], "fixed")
        self.assertEqual(payload["active_index"], 2)
        self.assertFalse(payload["backend_owned"])

    def test_queue_serialized_xy_is_the_cycle_authority(self):
        indices = [0, 1, 2, 0, 1, 2, 0]
        outputs = [
            self.run_node(mode="increment", index=index, queue_seq=sequence)["result"][0]
            for sequence, index in enumerate(indices, 1)
        ]
        self.assertEqual(outputs, ["A", "B", "C", "A", "B", "C", "A"])

    def test_backend_does_not_secretly_advance_repeated_serialized_xy(self):
        outputs = [
            self.run_node(mode="increment", index=0, queue_seq=sequence)["result"][0]
            for sequence in range(1, 5)
        ]
        self.assertEqual(outputs, ["A", "A", "A", "A"])

    def test_queue_transport_reports_next_cursor_without_changing_selected_xy(self):
        transport = json.dumps({
            "__queue_cycle_v1": True,
            "next_index": 2,
            "shuffle": [],
        })
        result = self.run_node(
            mode="increment",
            index=1,
            enhanced="B",
            queue_seq=7,
            shuffle_json=transport,
        )
        self.assertEqual(result["result"][0], "B")
        payload = json.loads(result["ui"]["prompt_enhancer"][0])
        self.assertEqual(payload["active_index"], 1)
        self.assertEqual(payload["next_index"], 2)
        self.assertEqual(payload["queue_seq"], 7)
        self.assertTrue(payload["queue_prepared"])
        self.assertFalse(payload["backend_owned"])

    def test_queue_sequence_is_diagnostic_only(self):
        for sequence in (1, 2, 99):
            with self.subTest(sequence=sequence):
                result = self.run_node(mode="increment", index=2, enhanced="C", queue_seq=sequence)
                self.assertEqual(result["result"][0], "C")

    def test_positive_queue_sequence_marks_browser_prepared_without_transport_object(self):
        result = self.run_node(mode="increment", index=1, enhanced="B", queue_seq=3, shuffle_json="[]")
        payload = json.loads(result["ui"]["prompt_enhancer"][0])
        self.assertEqual(result["result"][0], "B")
        self.assertTrue(payload["queue_prepared"])
        self.assertEqual(payload["queue_seq"], 3)

    def test_legacy_headless_cycle_gets_one_step_next_cursor(self):
        result = self.run_node(mode="increment", index=1, enhanced="B", shuffle_json="[]")
        payload = json.loads(result["ui"]["prompt_enhancer"][0])
        self.assertEqual(result["result"][0], "B")
        self.assertEqual(payload["next_index"], 2)
        self.assertFalse(payload["queue_prepared"])

    def test_runtime_scope_no_longer_controls_normal_cycle_selection(self):
        a = self.run_node(mode="increment", index=0, scope="wf-a", queue_seq=1)
        b = self.run_node(mode="increment", index=2, enhanced="C", scope="wf-b", queue_seq=2)
        c = self.run_node(mode="increment", index=1, enhanced="B", state_id="", scope="", queue_seq=3)
        self.assertEqual([a["result"][0], b["result"][0], c["result"][0]], ["A", "C", "B"])

    def test_fixed_is_cacheable_and_queue_prepared_cycles_have_unique_signatures(self):
        fixed = PE.LocalLLMPromptEnhancer.IS_CHANGED(
            prompt_cycle="fixed",
            enhance_with_workflow=False,
            unique_id="4",
            prompt_state_id="node-4",
            prompt_runtime_scope="wf-a",
        )
        self.assertEqual(fixed, "idle")

        for mode in ("increment", "decrement", "shuffle", "random"):
            with self.subTest(mode=mode):
                legacy = PE.LocalLLMPromptEnhancer.IS_CHANGED(
                    prompt_cycle=mode,
                    enhance_with_workflow=False,
                    prompt_cycle_revision=7,
                    prompt_cycle_queue_seq=0,
                )
                self.assertTrue(math.isnan(legacy))
                first = PE.LocalLLMPromptEnhancer.IS_CHANGED(
                    prompt_cycle=mode,
                    enhance_with_workflow=False,
                    prompt_cycle_revision=7,
                    prompt_cycle_queue_seq=1,
                    prompt_history_index=0,
                )
                second = PE.LocalLLMPromptEnhancer.IS_CHANGED(
                    prompt_cycle=mode,
                    enhance_with_workflow=False,
                    prompt_cycle_revision=7,
                    prompt_cycle_queue_seq=2,
                    prompt_history_index=1,
                )
                self.assertEqual(first, f"cycle:{mode}:7:1:0")
                self.assertEqual(second, f"cycle:{mode}:7:2:1")
                self.assertNotEqual(first, second)

        self.assertTrue(math.isnan(PE.LocalLLMPromptEnhancer.IS_CHANGED(prompt_cycle="fixed", enhance_with_workflow=True)))

    def test_string_false_does_not_enable_workflow_enhancement(self):
        self.assertEqual(
            PE.LocalLLMPromptEnhancer.IS_CHANGED(
                prompt_cycle="fixed",
                enhance_with_workflow="false",
                prompt_cycle_queue_seq=0,
            ),
            "idle",
        )
        self.assertEqual(
            PE.LocalLLMPromptEnhancer.IS_CHANGED(
                prompt_cycle="increment",
                enhance_with_workflow="false",
                prompt_cycle_queue_seq=1,
                prompt_cycle_revision=0,
                prompt_history_index=0,
            ),
            "cycle:increment:0:1:0",
        )

    def test_queue_transport_parser_rejects_unmarked_data(self):
        self.assertIsNone(PE._queue_cycle_transport("[]", 3, 0))
        self.assertIsNone(PE._queue_cycle_transport('{"next_index":2}', 3, 0))
        self.assertEqual(
            PE._queue_cycle_transport('{"__queue_cycle_v1":true,"next_index":2,"shuffle":[0,1]}', 3, 0),
            (2, [0, 1]),
        )

    def test_runtime_state_key_requires_both_scope_and_state_id(self):
        self.assertEqual(PE._enhancer_state_key("4", "node-4", "wf-a"), "wf-a\x1fnode-4")
        self.assertEqual(PE._enhancer_state_key("4", "node-4", ""), "")
        self.assertEqual(PE._enhancer_state_key("4", "", "wf-a"), "")


if __name__ == "__main__":
    unittest.main()
