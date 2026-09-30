from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ENHANCER = ROOT / "prompt_enhancer.py"
FRONTEND = ROOT / "web" / "js" / "prompt_enhancer_dom_v0654.js"
INPUT_HELPERS = ROOT / "web" / "js" / "prompt_enhancer_input_slots.js"
INIT = ROOT / "__init__.py"
SERVICE = ROOT / "service.py"
NODES = ROOT / "nodes.py"
VERSION = ROOT / "version.py"


class SourceContractTests(unittest.TestCase):
    def test_cycle_state_endpoint_passes_revision_to_backend_snapshot(self):
        source = PROMPT_ENHANCER.read_text(encoding="utf-8")
        route = source[source.index('@routes.post("/local_llm_prompt_enhancer/cycle_state")') :]
        route = route[: route.index('@routes.post("/local_llm_prompt_enhancer/cancel")')]
        self.assertIn('body.get("revision", 0)', route)

    def test_prompt_enhancer_uses_extracted_cycle_store_instead_of_duplicate_global_state(self):
        source = PROMPT_ENHANCER.read_text(encoding="utf-8")
        self.assertIn("from .prompt_cycle import", source)
        self.assertNotIn("_PROMPT_CYCLE_STATES", source)
        self.assertNotIn("random.SystemRandom", source)

    def test_frontend_never_calls_legacy_cycle_reset_endpoint(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertNotIn("/local_llm_prompt_enhancer/cycle_reset", source)

    def test_normal_frontend_queueing_has_no_backend_cycle_sync_authority(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertNotIn("syncPromptCycleFromBackend", source)
        self.assertNotIn("/local_llm_prompt_enhancer/cycle_state", source)
        self.assertNotIn("__promptEnhancerCycleSyncAt", source)

    def test_frontend_contains_no_input_topology_mutation(self):
        source = FRONTEND.read_text(encoding="utf-8") + "\n" + INPUT_HELPERS.read_text(encoding="utf-8")
        forbidden = [
            r"\.inputs\s*\.\s*splice\s*\(",
            r"\.inputs\s*\.\s*unshift\s*\(",
            r"\.inputs\s*\.\s*push\s*\(",
            r"\.inputs\s*\.\s*sort\s*\(",
            r"\.inputs\s*\.\s*reverse\s*\(",
            r"\.target_slot\s*=",
            r"\.targetSlot\s*=",
        ]
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, source))

    def test_prompt_cycle_uses_native_comfy_control_after_generate(self):
        backend = PROMPT_ENHANCER.read_text(encoding="utf-8")
        frontend = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('"prompt_cycle_counter": (', backend)
        self.assertIn('"control_after_generate": True', backend)
        self.assertIn("function installNativePromptCycleControl", frontend)
        self.assertIn("promptCycleControlWidget", frontend)
        self.assertIn("originalBefore", frontend)
        self.assertIn("originalAfter", frontend)
        self.assertIn("promptCycleIndexFromCounter", frontend)
        self.assertNotIn("_register_prompt_cycle_submit_handler(PromptServer.instance)", backend)

    def test_backend_maps_native_counter_by_modulo_without_shared_cursor(self):
        source = PROMPT_ENHANCER.read_text(encoding="utf-8")
        cycle_block = source[source.index("# Non-fixed Prompt Cycle uses the same queue lifecycle as a ComfyUI") :]
        cycle_block = cycle_block[: cycle_block.index("\n\n\ntry:\n    from aiohttp")]
        self.assertIn("active_index = (cycle_counter % len(history)) if history else 0", cycle_block)
        self.assertIn('"native_control": True', cycle_block)
        self.assertNotIn("PROMPT_CYCLE_STORE.advance(", cycle_block)

    def test_runtime_cycle_counter_is_appended_and_hidden(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('"prompt_cycle_counter"', source)
        self.assertIn("hideNativeWidgetForPanel(promptCycleControlWidget(node))", source)
        self.assertNotIn("installQueueOwnedPromptCycle", source)

    def test_frontend_helper_contains_no_legacy_cycle_planner(self):
        helper = (ROOT / "web" / "js" / "prompt_enhancer_queue_state.js").read_text(encoding="utf-8")
        self.assertNotIn("planPromptCycleQueueItem", helper)
        self.assertNotIn("nextPromptCycleQueueSequence", helper)
        self.assertNotIn("normalizeCycleMode", helper)

    def test_unused_enhancement_seed_is_neutralized(self):
        source = FRONTEND.read_text(encoding="utf-8")
        helper = (ROOT / "web" / "js" / "prompt_enhancer_queue_state.js").read_text(encoding="utf-8")
        self.assertIn("function installExecutionSerialization", source)
        self.assertIn("serializeEnhancementSeed", source)
        self.assertIn("shouldAdvanceEnhancementSeed", source)
        self.assertIn("promptEnhancerManual: true", source)
        self.assertIn("normalizeWidgetBoolean", helper)

    def test_shuffle_deck_phase_survives_runtime_state_remounts(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn("promptShuffleStarted:", source)
        self.assertIn('node.__promptEnhancerShuffleStarted = state.promptCycle === "shuffle"', source)

    def test_workflow_toggle_never_uses_truthiness_for_string_false(self):
        source = FRONTEND.read_text(encoding="utf-8")
        self.assertIn('normalizeWidgetBoolean(widget(node, "enhance_with_workflow")?.value)', source)
        self.assertNotIn('!!widget(node, "enhance_with_workflow")?.value', source)


    def test_openai_routes_pop_internal_sampler_seed_before_generation(self):
        source = SERVICE.read_text(encoding="utf-8")
        self.assertEqual(source.count('sampler_seed = int(overrides.pop("_openai_sampler_seed"))'), 2)
        self.assertIn('X-Local-LLM-Sampler-Seed', source)
        self.assertEqual(source.count('_sse_response(seed_headers)'), 2)

    def test_openai_routes_log_resolved_sampler_and_have_no_duplicate_model_assignment(self):
        source = SERVICE.read_text(encoding="utf-8")
        self.assertEqual(source.count('_log_openai_sampling_result('), 5)  # definition + four route calls
        self.assertIn('"X-Local-LLM-Seed-Mode": str(mode)', source)
        self.assertIn('"X-Local-LLM-Request-Id"]', source)
        self.assertEqual(source.count('control_sources = openai_control_sources(body)'), 2)
        self.assertEqual(source.count('request_id = _OPENAI_REQUEST_SEQUENCE.next()'), 2)
        duplicate = (
            'requested_model = str(body.get("model") or SERVICE.get_config().get("model") or "local-llm")\n'
            '            requested_model = str(body.get("model") or SERVICE.get_config().get("model") or "local-llm")'
        )
        self.assertNotIn(duplicate, source)

    def test_runtime_info_exposes_advanced_sampler_values_for_api_diagnostics(self):
        source = NODES.read_text(encoding="utf-8")
        block = source[source.index('"effective_model_settings": {'):]
        block = block[:block.index('"independent_output_controls": {')]
        for name in ("tfs_z", "mirostat_mode", "mirostat_tau", "mirostat_eta"):
            with self.subTest(name=name):
                self.assertIn(f'"{name}": {name}', block)

    def test_request_generation_overrides_are_forwarded_to_runtime(self):
        service_source = SERVICE.read_text(encoding="utf-8")
        node_source = NODES.read_text(encoding="utf-8")
        self.assertIn('args["request_generation_overrides"]', service_source)
        self.assertIn('request_overrides=request_generation_overrides', node_source)
        # Preset resolution and request precedence live in the pure tested helper.
        self.assertIn('resolve_generation_controls(', node_source)

    def test_openai_native_sampler_seed_is_carried_directly_to_generation(self):
        service_source = SERVICE.read_text(encoding="utf-8")
        node_source = NODES.read_text(encoding="utf-8")
        self.assertEqual(service_source.count('overrides["native_sampler_seed_override"] = sampler_seed'), 2)
        self.assertIn('native_sampler_seed_override=None', node_source)
        self.assertIn('"llama_seed_applied": seed_object_after_set', node_source)
        self.assertIn('"seed_kwarg_forwarded": seed_kwarg_forwarded', node_source)


    def test_registry_scanner_sensitive_paths_use_plain_code_patterns(self):
        node_source = NODES.read_text(encoding="utf-8")
        enhancer_source = PROMPT_ENHANCER.read_text(encoding="utf-8")
        # Low-level llama.cpp access stays lazy/optional without dynamic imports.
        self.assertNotIn("import importlib", node_source)
        self.assertIn("from llama_cpp import llama_cpp as low_level", node_source)
        # Pending manual work has an enhancement-specific name, avoiding an
        # accidental collision with network-client naming conventions.
        self.assertIn("_PENDING_ENHANCEMENTS", enhancer_source)
        # Fatal decode recognition is semantic rather than tied to a textual
        # Python-call representation from an exception message.
        fatal_start = node_source.index("def _is_fatal_decode_error")
        fatal_end = node_source.index("\ndef _discard_failed_native_context", fatal_start)
        fatal_source = node_source[fatal_start:fatal_end]
        self.assertIn('"llama" not in message or "decode" not in message', fatal_source)


    def test_llm_models_and_presets_share_canonical_uppercase_root(self):
        paths_source = (ROOT / "paths.py").read_text(encoding="utf-8")
        node_source = NODES.read_text(encoding="utf-8")
        service_source = SERVICE.read_text(encoding="utf-8")
        enhancer_source = PROMPT_ENHANCER.read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn('LLM_FOLDER_KEY = "LLM"', paths_source)
        self.assertIn('LLM_MODEL_DIR = Path(folder_paths.models_dir) / LLM_FOLDER_KEY', paths_source)
        self.assertIn('PRESET_ROOT_DIR = LLM_MODEL_DIR / "local_LLM_presets"', paths_source)
        self.assertIn('register_llm_model_folder()', node_source)
        self.assertIn('from .paths import PRESET_ROOT_DIR', service_source)
        self.assertIn('from .paths import PRESET_ROOT_DIR', enhancer_source)
        self.assertNotIn('models/llm', node_source + service_source + enhancer_source)
        self.assertIn('ComfyUI/models/LLM/', readme)
        self.assertIn('ComfyUI/models/llm/', readme)  # migration note for pre-0.18.86 installs

    def test_cpu_only_is_a_hard_runtime_offload_boundary(self):
        source = NODES.read_text(encoding="utf-8")
        self.assertIn('if cpu_only:', source)
        self.assertIn('gpu_layers = 0', source)
        self.assertIn('kv_cache_location = "CPU"', source)
        self.assertIn('op_offload = "Disabled"', source)
        self.assertIn('vision_use_gpu = False if cpu_only else', source)
        self.assertIn('offload_kqv=(kv_cache_location == "GPU")', source)

    def test_cpu_auto_threads_and_numa_are_passed_to_llama_constructor(self):
        source = NODES.read_text(encoding="utf-8")
        self.assertIn('cpu_runtime_profile(threads, threads_batch, numa_mode, cpu_only=cpu_only)', source)
        self.assertIn('n_threads=effective_threads', source)
        self.assertIn('n_threads_batch=effective_threads_batch', source)
        self.assertIn('numa=_llama_numa_strategy(llama_cpp, effective_numa_mode)', source)

    def test_cpu_load_diagnostics_do_not_require_cuda_snapshots(self):
        source = NODES.read_text(encoding="utf-8")
        verified_start = source.index('def _load_llama_verified')
        verified_end = source.index('\ndef _load_llama_fast', verified_start)
        verified = source[verified_start:verified_end]
        self.assertIn('before = _torch_cuda_snapshot() if requested_gpu else {}', verified)
        self.assertIn('after = _torch_cuda_snapshot() if requested_gpu else {}', verified)

    def test_cpu_memory_presets_define_first_class_cpu_only_profiles(self):
        source = (ROOT / "presets.py").read_text(encoding="utf-8")
        self.assertIn('"CPU Only (Auto / High Core)"', source)
        self.assertIn('"compute_mode": "CPU Only"', source)
        self.assertIn('"numa_mode": "Auto"', source)
        self.assertIn('"op_offload": "Disabled"', source)

    def test_server_ui_exposes_one_click_cpu_auto_profile(self):
        source = (ROOT / "web" / "js" / "local_llm_server_v102.js").read_text(encoding="utf-8")
        self.assertIn('data-action="cpu-auto-profile"', source)
        self.assertIn('compute_mode:"CPU Only"', source)
        self.assertIn('threads:0,threads_batch:0', source)
        self.assertIn('kv_cache_location:"CPU",gpu_layers:0,op_offload:"Disabled"', source)
        self.assertIn('const memoryOwned=new Set(["compute_mode","numa_mode"', source)

    def test_service_auto_cpu_detection_applies_before_runtime_load(self):
        source = SERVICE.read_text(encoding="utf-8")
        self.assertIn('def _service_cpu_only(cfg):', source)
        self.assertIn('if _service_cpu_only(cfg):\n        gpu_layers = 0', source)
        self.assertIn('estimator_cpu_only = _service_cpu_only(cfg)', source)
        self.assertIn('cpu_only=_service_cpu_only(cfg)', source)

    def test_release_version_advanced_past_01880(self):
        source = VERSION.read_text(encoding="utf-8")
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('PACKAGE_VERSION = "0.18.103-alpha"', source)
        self.assertIn('version = "0.18.103a0"', project)

    def test_frontend_registry_references_cache_distinct_current_file_and_preserves_helper_module(self):
        source = INIT.read_text(encoding="utf-8")
        self.assertIn('"local_llm_server": "local_llm_server_v102.js"', source)
        self.assertIn('"prompt_enhancer": "prompt_enhancer_dom_v0654.js"', source)
        self.assertTrue(INPUT_HELPERS.is_file())
        # Cleanup is limited to versioned prompt_enhancer_dom_v*.js files; the
        # unversioned helper therefore survives extract-over-existing upgrades.
        self.assertIn('name.startswith("prompt_enhancer_dom_v")', source)

    def test_readme_matches_current_package_scope(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("**Version 0.18.103-alpha**", readme)
        self.assertIn("**Local LLM Generate**", readme)
        self.assertIn("**Local LLM Settings**", readme)
        self.assertIn("**Local LLM Prompt Enhancer**", readme)
        self.assertIn("Prompt Enhancer version: 0.6.54-alpha", readme)
        self.assertIn('NODE_VERSION = "0.6.54-alpha"', PROMPT_ENHANCER.read_text(encoding="utf-8"))
        self.assertNotIn("H3_SEQUENCE", readme)
        self.assertNotIn("The node is a planning/orchestration layer", readme)


if __name__ == "__main__":
    unittest.main()
