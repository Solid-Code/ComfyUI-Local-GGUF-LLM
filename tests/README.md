# Regression tests

Run from the `ComfyUI-Local-GGUF-LLM` directory:

```bash
python tests/run_all.py
```

The Python suite uses only the standard library. If Node.js is installed, the runner also executes the frontend input-slot tests and JavaScript syntax checks. The tests do not require ComfyUI, CUDA, a GGUF model, or `llama-cpp-python`.

The current suite protects the high-risk Prompt Enhancer behavior: Fixed/Increment/Decrement/Shuffle/Random, rapid queueing, revision invalidation, workflow isolation, TTL cleanup, concurrent backend cursor access, stale-result guards, and preservation of ComfyUI input slot topology.
