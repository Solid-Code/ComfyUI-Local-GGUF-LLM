"""Run all dependency-free regression tests from the custom-node root."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])
    node = shutil.which("node")
    if node:
        run([node, "--test", "tests/frontend/input_slots.test.mjs", "tests/frontend/prompt_enhancer_queue_state.test.mjs", "tests/frontend/source_contracts.test.mjs"])
        run([node, "--check", "web/js/local_llm_server_v102.js"])
        run([node, "--check", "web/js/prompt_enhancer_dom_v0654.js"])
        run([node, "--check", "web/js/prompt_enhancer_input_slots.js"])
        run([node, "--check", "web/js/prompt_enhancer_queue_state.js"])
    else:
        print("Node.js not found; frontend runtime tests skipped.", flush=True)
    run([sys.executable, "-m", "py_compile", "paths.py", "cpu_tuning.py", "prompt_cycle.py", "prompt_enhancer.py", "seed_utils.py", "generation_controls.py", "openai_controls.py", "api_diagnostics.py", "service.py", "nodes.py", "vram_coordination.py", "gguf_meta.py", "presets.py", "version.py"])
    print("All available regression checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
