from __future__ import annotations

import asyncio
import copy
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

    class FakePromptServerInstance:
        def __init__(self):
            self.routes = FakeRoutes()
            self.on_prompt_handlers = []

        def add_on_prompt_handler(self, handler):
            self.on_prompt_handlers.append(handler)

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=FakePromptServerInstance())
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


    def submission_payload(
        self,
        *,
        workflow_id="workflow-a",
        mode="increment",
        index=0,
        revision=1,
        epoch=0,
        runtime_scope=None,
        state_id="node-4",
        history=None,
        enhanced="A",
        partial=False,
        enhance_with_workflow=False,
    ):
        if history is None:
            history = ["A", "B", "C"]
        if runtime_scope is None:
            runtime_scope = workflow_id
        payload = {
            "client_id": "client-a",
            "workflow_id": workflow_id,
            "extra_data": {
                "extra_pnginfo": {
                    "workflow": {"id": workflow_id},
                },
            },
            "prompt": {
                "4": {
                    "class_type": "LocalLLMPromptEnhancer",
                    "inputs": {
                        "prompt": "source",
                        "enhanced_prompt": enhanced,
                        "prompt_cycle": mode,
                        "enhance_with_workflow": enhance_with_workflow,
                        "prompt_history_json": json.dumps(history),
                        "prompt_history_index": index,
                        "prompt_cycle_revision": revision,
                        "prompt_shuffle_json": "[]",
                        "prompt_state_id": state_id,
                        "prompt_runtime_scope": runtime_scope,
                        "prompt_cycle_queue_seq": 0,
                        "prompt_cycle_epoch": epoch,
                    },
                },
            },
        }
        if partial:
            payload["partial_execution_targets"] = ["4"]
        return payload

    def execute_prepared_submission(self, submission):
        inputs = dict(submission["prompt"]["4"]["inputs"] )
        return PE.LocalLLMPromptEnhancer().output_prompts(
            unique_id="4",
            **inputs,
        )

    def test_prompt_cycle_does_not_register_server_admission_handler(self):
        from server import PromptServer
        self.assertEqual(PromptServer.instance.on_prompt_handlers, [])

    def test_native_cycle_counter_maps_run_x_n_to_history_entries(self):
        counters = list(range(7))
        outputs = [
            self.run_node(mode="increment", counter=counter)["result"][0]
            for counter in counters
        ]
        self.assertEqual(outputs, ["A", "B", "C", "A", "B", "C", "A"])

    def test_native_cycle_counter_supports_negative_decrement_wrap(self):
        counters = [0, -1, -2, -3, -4]
        outputs = [
            self.run_node(mode="decrement", counter=counter)["result"][0]
            for counter in counters
        ]
        self.assertEqual(outputs, ["A", "C", "B", "A", "C"])

    def test_ten_prompt_native_counter_cycles_all_items(self):
        history = [f"P{i}" for i in range(1, 11)]
        outputs = [
            self.run_node(mode="increment", history=history, enhanced=history[0], counter=counter)["result"][0]
            for counter in range(12)
        ]
        self.assertEqual(outputs, history + history[:2])

    def run_node(
        self,
        *,
        mode="fixed",
        index=0,
        revision=1,
        epoch=0,
        history=None,
        enhanced="A",
        state_id="node-4",
        scope="wf-a",
        queue_seq=0,
        shuffle_json="[]",
        counter=0,
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
            prompt_cycle_epoch=epoch,
            prompt_shuffle_json=shuffle_json,
            prompt_state_id=state_id,
            prompt_runtime_scope=scope,
            prompt_cycle_queue_seq=queue_seq,
            prompt_cycle_counter=counter,
            unique_id="4",
        )

    def test_fixed_uses_selected_xy(self):
        fixed = self.run_node(mode="fixed", index=2, enhanced="C")
        self.assertEqual(fixed["result"], ("C", "source"))
        payload = json.loads(fixed["ui"]["prompt_enhancer"][0])
        self.assertEqual(payload["mode"], "fixed")
        self.assertEqual(payload["active_index"], 2)
        self.assertFalse(payload["backend_owned"])

    def test_prompt_history_index_no_longer_drives_nonfixed_cycle(self):
        # Browser-visible X/Y may be stale; the native counter serialized into
        # each queue item is the sole non-fixed selection authority.
        outputs = [
            self.run_node(mode="increment", index=0, counter=counter)["result"][0]
            for counter in range(4)
        ]
        self.assertEqual(outputs, ["A", "B", "C", "A"])

    def test_runtime_scope_does_not_control_native_cycle_selection(self):
        a = self.run_node(mode="increment", index=0, scope="wf-a", counter=0)
        b = self.run_node(mode="increment", index=0, scope="wf-b", counter=2)
        c = self.run_node(mode="increment", index=0, state_id="", scope="", counter=1)
        self.assertEqual([a["result"][0], b["result"][0], c["result"][0]], ["A", "C", "B"])

    def test_fixed_is_cacheable_and_native_cycle_counter_is_signature(self):
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
                first = PE.LocalLLMPromptEnhancer.IS_CHANGED(
                    prompt_cycle=mode,
                    enhance_with_workflow=False,
                    prompt_cycle_counter=10,
                )
                second = PE.LocalLLMPromptEnhancer.IS_CHANGED(
                    prompt_cycle=mode,
                    enhance_with_workflow=False,
                    prompt_cycle_counter=11,
                )
                self.assertEqual(first, f"cycle:{mode}:10")
                self.assertEqual(second, f"cycle:{mode}:11")
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
                prompt_cycle_counter=3,
            ),
            "cycle:increment:3",
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
