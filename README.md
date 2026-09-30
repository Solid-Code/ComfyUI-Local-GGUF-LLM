# ComfyUI Local GGUF LLM

> **0.18.100 CPU-first runtime:** CPU-only inference is now a first-class mode. `CPU Only` hard-disables model-layer, KV-cache, operator, and vision-projector GPU offload; Auto CPU threading is topology/affinity aware; NUMA Auto is available for multi-node CPU systems.

**Version 0.18.100-alpha**  
Persistent local GGUF inference for ComfyUI, with workflow nodes, multimodal prompt support, VRAM-aware model residency, a performance tuner, Prompt Enhancer, and an optional OpenAI-compatible API.

## What is in this package

This package currently registers exactly three ComfyUI nodes:

| Node | Purpose |
| --- | --- |
| **Local LLM Generate** | Send a system prompt, prompt, optional images/video frames, and a request-local seed to the persistent Local LLM service. |
| **Local LLM Settings** | Carry reusable model/sampling/vision/runtime settings into Generate or Prompt Enhancer. It can follow the current server configuration or load a saved Complete Settings Preset. |
| **Local LLM Prompt Enhancer** | Generate and manage enhanced prompts with optional image/video reference input, prompt history, Prompt Sets, templates, batching, and workflow-time cycling. |

The package also installs the **Local LLM Server** UI inside ComfyUI. The server owns the active llama.cpp model/context and is shared by all three nodes and by the OpenAI-compatible API.

> **Package scope:** the old H3 Shot Generator is no longer part of this package. It lives in the separate `ComfyUI-H3-Shot-Generator` package. MiniMax H3 enhancement templates remain here because Prompt Enhancer can still write H3 prompts.

## Architecture

```text
                         ┌──────────────────────┐
                         │  Local LLM Server    │
                         │  one shared llama.cpp│
                         │  model/context       │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
      Local LLM Generate    Local LLM Prompt      OpenAI-compatible API
              │                 Enhancer          /v1/chat/completions
              │                     │              /v1/completions
              └──────────┬──────────┘
                         │
                Local LLM Settings
                 (optional snapshot)
```

The workflow nodes do **not** independently load duplicate GGUF models. They submit work to the same process-global service.

## Requirements

- ComfyUI
- Python 3.10 or newer
- `llama-cpp-python` installed in the **same Python environment as ComfyUI**
- A CPU-capable `llama-cpp-python` build for CPU-only inference
- A GPU-enabled `llama-cpp-python` build only when GPU or mixed CPU/GPU inference is desired

`requirements.txt` intentionally installs nothing. This package does not automatically install or replace `llama-cpp-python`, because doing so can replace a working CUDA build with an incompatible or CPU-only wheel.

Verify the binding visible to ComfyUI with the Python executable used by ComfyUI:

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

Feature availability such as particular multimodal chat handlers, quantized KV formats, Flash Attention, N-gram speculative decoding, or native MTP depends on the installed `llama-cpp-python` / llama.cpp build.

## Installation

Extract or clone the package to:

```text
ComfyUI/custom_nodes/ComfyUI-Local-GGUF-LLM/
```

Then restart ComfyUI. After updating from an older build, hard-refresh the browser (`Ctrl+F5`) if the previous JavaScript UI is still cached.

Do not install a second standalone copy of the Local LLM Prompt Enhancer beside this package; Prompt Enhancer is bundled here.

## Model layout

GGUF model files are discovered recursively through the shared ComfyUI `LLM` model folder, whose canonical path is:

```text
ComfyUI/models/LLM/
```

Example:

```text
ComfyUI/models/LLM/
├── Qwen/
│   ├── Qwen3.8-27B-Q4_K_M.gguf
│   └── mmproj-Qwen3.8-27B-F16.gguf
├── Gemma/
│   ├── gemma-3-12b-it-Q4_K_M.gguf
│   └── mmproj-gemma-3-12b-f16.gguf
└── Mistral/
    └── mistral-small-24b.gguf
```

Files whose names look like `mmproj`, `vision-proj`, or `projector` are presented as vision projectors rather than text-model choices.

Package presets live under the same canonical `LLM` root:

```text
ComfyUI/models/LLM/local_LLM_presets/
```

Using one `models/LLM` root avoids parallel `llm`/`LLM` directories on case-sensitive Linux/WSL filesystems and interoperates with other ComfyUI LLM nodes that register the `LLM` model category.

If upgrading from a release that stored GGUF files in lowercase `ComfyUI/models/llm/`, move those model files into `ComfyUI/models/LLM/` once. Presets already used the uppercase root in recent releases.

## Quick start

1. Open the **Local LLM Server** from its ComfyUI sidebar launcher or floating status control.
2. In **Model**, choose a GGUF model.
3. For a multimodal model, choose the matching **Vision / mmproj** projector or use `Auto` when a safe match is available.
4. Leave **Model Preset** on `Auto (Detected)` unless you want to set sampling behavior manually.
5. In **Memory**, choose a context size and VRAM policy. For a ComfyUI machine that also runs diffusion/video models, start with **Auto Yield to ComfyUI**.
6. Use **On Demand** startup unless you specifically want the LLM loaded at ComfyUI startup.
7. Add **Local LLM Generate** to a workflow and enter a prompt.

The service automatically loads on the first request when startup mode is **On Demand**.

# Local LLM Server

The server UI is the authoritative place for the global model/runtime configuration. It contains seven tabs.

## Server

Shows live state and request activity:

- loaded model and service state
- decode speed in tokens/second
- generated token count
- queued requests and total request count
- current client
- **Start**, **Suspend**, and **Stop / Unload** controls
- startup mode: `Off`, `On Demand`, or `Auto Start`
- optional draggable floating status indicator

**Suspend** releases the native model/context while retaining the configuration so the next request can reload it. **Stop / Unload** also acts as the package-wide Local LLM interrupt. It cancels the current request generation epoch and unloads safely when the active native llama.cpp call reaches a safe return boundary; it does not trigger ComfyUI's global workflow interrupt.

## Presets

The Presets tab manages **Complete Settings Presets**. A complete preset stores the LLM runtime configuration, including:

- model and vision projector
- model preset / reasoning behavior
- sampler values
- vision limits
- context / KV-cache configuration
- GPU offload and split settings
- prompt-prefix cache mode
- speculative decoding settings
- VRAM policy

Server administration is deliberately excluded: API keys, startup mode, content logging, and interface preferences are not part of Complete Settings Presets.

Preset creation and deletion are centralized in this tab. **Local LLM Settings** can load these presets but does not delete them.

## Model

The Model tab controls the current model and normal generation behavior:

- GGUF model
- Vision / mmproj projector
- detected model family and capabilities
- still-image, video-frame, and image-edge limits
- model preset
- thinking mode and reasoning effort where supported
- preserve-thinking-history behavior where supported
- temperature, top-p, top-k, min-p
- repeat, presence, and frequency penalties
- default max tokens

`Auto (Detected)` uses GGUF metadata plus the filename to select the closest known model family/preset. Unknown/community models fall back to generic behavior rather than being rejected.

Current tuned/detected families include presets for Qwen 3.8, Qwen 3.5, Qwen 3, GPT-OSS 20B, Mistral Small 3.2, Ministral 3, Gemma 3, Llama 3.1/3.2 Instruct, DeepSeek R1 Distill, Phi-4 Reasoning, and Nemotron 3 Nano.

Vision capability metadata also recognizes a broader set of supported llama.cpp multimodal families/handlers, including Qwen VL variants, Gemma 3/4, GLM vision models, LFM vision models, MiniCPM, LLaVA, Llama 3 Vision, Moondream, OCR-oriented models, and others. Actual support still depends on the handlers present in the installed binding and on a compatible projector.

## Memory

The Memory tab contains the model-residency and performance controls.

### Live VRAM estimate

The UI estimates and displays:

- model-weight VRAM
- KV-cache VRAM
- compute / batch working memory
- speculative-decoding memory
- vision/mmproj memory when applicable
- measured native residency for a previously verified matching configuration
- current GPU free memory
- projected headroom and Auto-Yield reload target

The estimate is intentionally conservative until the exact configuration has completed a verified load.

### VRAM policy

**Auto Yield to ComfyUI** is designed for shared diffusion/video + LLM systems. The LLM stays resident while there is room, but its native context can be fully closed when ComfyUI needs GPU memory and recreated on the next LLM request.

**Keep Resident** avoids voluntary LLM eviction and is appropriate when enough VRAM exists for the other models in the workflow.

The package uses a GPU-memory lease/coordination layer rather than blindly loading llama.cpp into whatever memory happens to be free.

### Context and KV cache

The panel exposes:

- context size, constrained to normal steps and the GGUF's advertised native context when known
- independent K and V KV-cache formats
- GPU or CPU KV-cache location
- GPU layers (`-1` = full supported offload)
- Flash Attention
- prompt batch (`n_batch`) and micro-batch (`n_ubatch`)
- mmap / mlock
- main GPU
- multi-GPU split mode and tensor split

Lower-bit KV formats can save significant memory but may change quality/performance slightly. CPU KV saves VRAM but is usually slower.

### CPU-only and high-core-count CPUs

**Compute Mode** controls the hardware boundary:

- `Auto` — use GPU/mixed mode only when ComfyUI can see a CUDA/ROCm accelerator and the installed llama.cpp binding supports GPU offload; otherwise use CPU-only mode.
- `GPU / Mixed` — allow llama.cpp GPU offload according to GPU Layers, KV placement, operator offload, and vision settings.
- `CPU Only` — hard-disable all native GPU allocations owned by this package: model layers use `n_gpu_layers=0`, KV stays on CPU, operator/KQV offload is disabled, and an mmproj/vision projector stays on CPU.

CPU thread fields accept `0` for **Auto**. Auto uses the physical-core count for token-generation/decode threads and the logical CPU count for prompt/batch threads, while respecting the CPU affinity visible to the ComfyUI process. On a fully exposed Threadripper 3990X (64 cores / 128 threads), this resolves to **64 generation threads and 128 prompt/batch threads**.

**NUMA = Auto** stays disabled on a single exposed NUMA node. In CPU-only mode, if the OS exposes more than one NUMA node to the process, Auto selects llama.cpp's `Distribute` strategy. Explicit `Disabled`, `Distribute`, `Isolate`, and `Numactl` choices remain available for manual tuning.

For CPU-only use, keep **mmap enabled** unless there is a specific reason not to. Leave **mlock disabled** by default; enable it only when the system has sufficient RAM and you intentionally want to pin model pages. The Memory tab's **Use CPU-only Auto** button applies a high-core-count baseline in one step: the CPU hard boundary, Auto threading/NUMA, CPU KV, mmap on, mlock off, and 2048/512 prompt/micro batches. The built-in **CPU Only (Auto / High Core)** memory preset applies the same baseline for legacy/internal memory-preset consumers.

CPU-only loads skip CUDA synchronization and VRAM-verification snapshots, so a machine without a working CUDA runtime does not need GPU diagnostics merely to load a GGUF.

### Prompt Prefix Cache

`Auto` can reuse an **exact token prefix already present in the current resident llama.cpp KV context**. It is not a second RAM cache.

The reusable prefix is cleared or bypassed by operations that invalidate the resident context, including Suspend, Stop / Unload, model reload, and vision requests.

### Speculative decoding

Modes:

- `Off`
- `Auto`
- `N-gram`
- `MTP`

Speculative decoding is target-verified: draft tokens are accepted only when the target model verifies them.

`Auto` prefers native embedded MTP when the GGUF and installed binding genuinely support it; otherwise it can fall back to N-gram. The current native MTP path requires full GPU offload (`GPU layers = -1`). Unsupported providers are disabled rather than silently emulated.

## Tuner

The built-in Performance Tuner benchmarks the actual saved server configuration and can test combinations of:

- prompt/micro batch sizes
- Flash Attention
- speculative decoding
- mmap / mlock
- GPU/CPU KV placement
- useful GPU-layer offload points
- optional KV precision variants

Profiles:

- **Quick** — shorter screening followed by sustained validation of the strongest candidates
- **Standard** — wider screening and longer validation

Scoring modes:

- **ComfyUI Cycle** — includes warm reload, fixed prompt/generation work, and Suspend-style unload; useful when the LLM frequently yields to ComfyUI
- **Inference Only** — focuses on prompt processing + generation without load/unload time

The tuner never silently reduces the configured context size. A recommendation can be applied directly or saved as a Complete Settings Preset.

## API

The API tab controls the optional OpenAI-compatible server:

- enable/disable external API access
- allow/disallow streaming requests
- display/copy the API base URL
- configure, reveal, or regenerate the API key

The API is disabled by default.

## Logs and privacy

The service keeps a bounded runtime log for status/diagnostics. Prompt and response **content are not logged by default**. Content logging must be explicitly enabled in the Logs tab.

# Workflow nodes

## Local LLM Generate

**Category:** `LLM/Local Service`

Inputs:

- System Prompt Preset
- editable System Prompt
- Prompt Preset
- editable Prompt
- request-local Seed + standard ComfyUI Control After Generate
- optional `LOCAL_LLM_SETTINGS`
- optional `IMAGE` / image batch
- optional `video_frames` as an ordered IMAGE batch

Outputs:

```text
response : STRING
thinking : STRING
info     : STRING (formatted JSON diagnostics)
```

When `settings` is disconnected, Generate uses the **current server configuration**. When connected, the Settings object supplies its runtime/sampler/vision snapshot. The Generate node's seed always remains request-local and is never replaced by Local LLM Settings.

Prompt and System Prompt presets are stored as editable text files and can be saved/deleted from the Generate node UI.

### Image and video-frame input

`image` accepts one still or an IMAGE batch. `video_frames` accepts ordered frames as an IMAGE batch and samples them evenly using the active Vision Max Frames limit. Inputs are downscaled to the active Vision Max Edge before being encoded for the multimodal handler.

A compatible multimodal GGUF + projector/handler is required. Connecting media does not make a text-only model multimodal.

## Local LLM Settings

**Category:** `LLM/Local Service`

Output:

```text
LOCAL_LLM_SETTINGS
```

The Settings node is deliberately a **request configuration carrier**, not another model server.

Its preset selector supports:

- **Current Server** — resolves against the live global server configuration
- **Custom** — use the serialized values in this Settings node
- any saved Complete Settings Preset

The compact node UI directly exposes thinking/reasoning, sampler values, and vision limits. Model selection, projector selection, and detailed memory/offload values are carried by the current server or Complete Settings Preset instead of being duplicated as a large second server panel inside the workflow.

Editing a preset-owned visible field changes the node to `Custom`. Seed is intentionally absent from Local LLM Settings; each consuming request node owns its own seed.

## Local LLM Prompt Enhancer

**Prompt Enhancer version: 0.6.51-alpha**  
**Category:** `prompt/Local LLM`

Prompt Enhancer uses the same persistent Local LLM service as Generate.

Inputs include:

- original Prompt
- editable Enhanced Prompt
- Prompt Set
- Prompt Cycle
- Add New / Overwrite behavior
- Enhancement Preset and editable Enhancement Instructions
- request-local enhancement seed
- Enhance with Workflow
- shared Prompt Preset
- optional `image(s)` IMAGE input
- optional native ComfyUI `VIDEO` input
- optional `LOCAL_LLM_SETTINGS`

Outputs:

```text
enhanced_prompt : STRING
prompt          : STRING  # unchanged original prompt
```

### Manual Enhance

The node's **Enhance Prompt** action can generate 1–64 enhancements in one batch. A batch greater than 1 appends results rather than repeatedly overwriting the same entry.

A manual batch pins the server/runtime configuration that existed when the batch started, so changing the server modal halfway through a batch does not make later items use a different model configuration.

When image/video inputs are connected, manual Enhance uses ComfyUI partial execution to obtain only the dependencies required for this node. Media tensors are not kept in a long-lived global cache.

### Enhanced-prompt history

Enhancements are stored as an editable prompt array. The UI supports:

- previous / next prompt
- direct active-index editing
- delete active entry
- clear all
- Undo / Redo, up to 20 array states in each direction

The active prompt is the one returned by `enhanced_prompt` when workflow-time enhancement is disabled.

### Prompt Sets

Prompt Sets save and restore the complete enhanced-prompt array and active index.

Stored at:

```text
ComfyUI/models/LLM/local_LLM_presets/prompt_enhancer/prompt_sets/
```

### Enhancement templates

Bundled protected templates currently include:

- Krea 2 - Image
- MiniMax H3 - T2VA
- MiniMax H3 - I2VA
- MiniMax H3 - FL2VA
- MiniMax H3 - L2VA
- MiniMax H3 - Ref2VA

User templates are stored as text files in:

```text
ComfyUI/models/LLM/local_LLM_presets/prompt_enhancer/
```

The MiniMax templates are prompt-writing templates only; the H3 shot-planning/generation node pack is separate.

### Prompt Cycle

When **Enhance with Workflow** is off, Prompt Cycle chooses how the stored enhanced-prompt array advances during normal workflow execution:

- `fixed` — stay on the selected entry
- `increment` — advance and wrap
- `decrement` — move backward and wrap
- `shuffle` — true no-repeat deck behavior until the deck is exhausted
- `random` — choose freely

Fixed mode is cacheable. Active cycle modes deliberately re-execute because each queued item can select a different stored prompt. During normal ComfyUI queueing, the frontend freezes the exact X/Y index into `prompt_history_index` while **each Run x N API prompt is serialized**. The backend uses that serialized index directly; it does not advance a shared cycle cursor at execution time. This makes the chosen prompt independent of queue execution timing and backend cursor state.

The queue snapshot also marks the request so backend history reconciliation cannot overwrite the selected array entry with the stale visible Enhanced Prompt text from a different X/Y position. Shuffle/Random planning is performed per queued item and is independent of the LLM generation seed. Headless/legacy submissions without queue metadata use the serialized X/Y as the active item and calculate only a one-step next-index value for UI feedback.

### Enhance with Workflow

When enabled, every normal workflow execution calls the LLM and creates a fresh enhancement before returning downstream text.

When disabled, workflow execution does not call the LLM for enhancement; it returns the selected stored Enhanced Prompt and applies Prompt Cycle if enabled.

# Preset storage

Current package-owned preset layout:

```text
ComfyUI/models/LLM/local_LLM_presets/
├── settings/          # Complete Settings Presets
├── prompts/           # shared user prompts
├── system_prompts/    # shared system prompts
├── sampler/           # legacy/user sampler presets remain readable
└── prompt_enhancer/
    ├── *.txt           # user enhancement templates
    └── prompt_sets/    # saved enhanced-prompt arrays
```

Small preset/config writes use same-directory temporary files followed by atomic replacement to avoid partial files and fixed-temp-name collisions.

The global server configuration itself is stored in the ComfyUI user directory as:

```text
local_llm_server.json
```

# Thinking and reasoning

Where the model/template exposes reasoning, ComfyUI requests return reasoning separately from final visible text:

- `thinking` output on Local LLM Generate
- `reasoning_content` on OpenAI-compatible chat responses/stream chunks

The runtime also handles common reasoning-tag/channel forms and the prefilled-thinking behavior used by supported Qwen templates. The final `response` output is kept separate from extracted reasoning.

# OpenAI-compatible API

Enable the API from **Local LLM Server → API**.

Base URL shown by the UI:

```text
http://<comfy-host>:<port>/local-llm/v1
```

Endpoints:

```text
GET  /local-llm/v1/models
POST /local-llm/v1/chat/completions
POST /local-llm/v1/completions
```

This is a focused local compatibility layer, not a complete implementation of every OpenAI API feature.

## Single active model

The service has one configured active GGUF model. `/v1/models` advertises that model and its configured usable context window.

The `model` string supplied by a client is accepted for OpenAI-client compatibility and response labeling; it does **not** route to or hot-swap an arbitrary GGUF. Select the real model in the Local LLM Server configuration.

`/v1/models` additionally exposes local-client context metadata:

- `context_length`
- `max_context_length`
- `n_ctx`
- `n_ctx_train` when native GGUF metadata is available

## Request control precedence

Generation-control precedence is:

```text
server/current values
        ↓
model preset / Auto-detected preset
        ↓
explicit request-local API fields
```

Only fields explicitly sent by the API client override the server/preset. Missing or `null` generation controls fall back to the server/preset value.

Supported request controls include the normal/local fields used by common OpenAI-compatible clients:

- `temperature`
- `top_p`
- `max_tokens`
- `presence_penalty`
- `frequency_penalty`
- `stop`
- `seed`
- `stream`

Common local-LLM extensions are also accepted:

- `top_k`
- `min_p`
- `typical_p`
- `repeat_penalty`
- `reasoning_effort`
- `thinking_mode`
- `preserve_thinking`
- `tfs_z`
- `mirostat_mode`
- `mirostat_tau`
- `mirostat_eta`

Fields not implemented by this service are not silently turned into unrelated behavior. For example, the current API is not an `n`-completion fan-out service and does not implement OpenAI logprobs/logit-bias semantics.

## Seed behavior

The API treats seeds as follows:

```text
seed omitted  -> fresh random request seed
seed: null    -> fresh random request seed
seed: -1      -> fresh random request seed
seed >= 0     -> explicit deterministic request seed
```

ComfyUI exposes a 64-bit seed, while llama.cpp sampling ultimately uses a deterministic 32-bit seed. The package maps the full 64-bit value into that native seed space and reserves llama.cpp's default/random sentinel.

Automatically randomized API requests keep a bounded recent history of native sampler seeds so a newly generated random request is redrawn if it would immediately recycle a recently used effective seed. Explicit client seeds remain reproducible and are never rewritten for uniqueness.

API responses include diagnostic headers:

```text
X-Local-LLM-Request-Id
X-Local-LLM-Seed
X-Local-LLM-Sampler-Seed
X-Local-LLM-Seed-Mode
```

Seed modes are `explicit`, `random-omitted`, `random-null`, or `random-sentinel`.

The server log also records the requested runtime seed, the seed read back from the live Llama object after `set_seed()`, sampler provenance, and hashes for repeated-output diagnosis. Different valid seeds can still legitimately converge to identical text when the token distribution is strongly peaked; identical text alone is not evidence of a reused seed.

## Streaming

Both chat completions and text completions support SSE streaming when streaming is enabled in the API tab.

Chat streaming forwards visible text through `delta.content` and reasoning through `delta.reasoning_content`. A final usage object is included when the request asks for `stream_options.include_usage`.

## Example request

```bash
curl http://127.0.0.1:8188/local-llm/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-llm",
    "messages": [{"role": "user", "content": "Explain KV cache quantization briefly."}],
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": -1,
    "stream": false
  }'
```

If the API key is blank, the server permits requests once the external API itself is enabled. Use an API key whenever the ComfyUI server is reachable by anything beyond a trusted local environment.

# In-process bridge for sibling custom nodes

On package import, a stable process-local module is registered as:

```python
import comfyui_local_gguf_llm_bridge as local_llm
```

It exposes:

- `SERVICE`
- `SAMPLER_PRESET_FIELDS`
- `API_VERSION` (currently 2)
- `PACKAGE_VERSION`
- `VRAM_POLICY_VERSION`
- `VRAM_COORDINATION_MODE`

`SERVICE.api` provides a small integration surface for sibling custom-node packages, including status/settings access, text/message generation, and an explicit GPU handoff. This avoids depending on this package's hyphenated filesystem folder name or importing private implementation helpers.

# Tests

The package includes a dependency-light regression suite covering Prompt Cycle behavior, workflow isolation, input-slot stability, OpenAI request precedence and seed plumbing, repeated-output diagnostics, and source contracts.

Run it from the custom-node directory:

```bash
python tests/run_all.py
```

When Node.js is available, the runner also executes the frontend tests and JavaScript syntax checks.

The v0.18.99 package currently runs:

- **77 Python tests**
- **26 JavaScript tests**
- Python compilation checks
- JavaScript syntax checks for all shipped frontend modules

Live CUDA/model performance still depends on the user's ComfyUI, driver, GPU, GGUF, and `llama-cpp-python` build and therefore cannot be fully represented by the dependency-free unit suite.

# Troubleshooting

## No models appear

Confirm the GGUF is below:

```text
ComfyUI/models/LLM/
```

Then restart ComfyUI. The model list is recursive.

## Vision input is rejected or ignored

Check all three conditions:

1. the GGUF family is actually multimodal,
2. a compatible mmproj/projector is selected,
3. the installed `llama-cpp-python` build contains a compatible chat handler / MTMD path.

## ComfyUI needs VRAM occupied by the LLM

Use **Auto Yield to ComfyUI**. The service can close the native context when ComfyUI needs the memory and warm-reload it on the next LLM request. **Suspend** can also yield manually.

## Context shown by an external client does not match the server

The actual llama.cpp context is the **Context Size configured in Local LLM Server / Local LLM Settings**, not whatever an external UI assumes. Query `/local-llm/v1/models` or inspect the server panel to see the configured value.

## An API reroll produces the same response

Check the `OpenAI API request` and `OpenAI sampling resolved` log lines. Compare:

- request ID
- seed mode
- request seed
- sampler seed
- applied sampler seed
- temperature / top-p / top-k and their `[api]` vs `[server/preset]` provenance
- response and reasoning hashes

A temperature of 0 or top-k of 1 makes the output effectively seed-independent. Distinct seeds can also produce the same output when the model distribution strongly favors the same tokens.

## Prompt Enhancer runs again when nothing changed

With **Enhance with Workflow** disabled, the Prompt Enhancer seed is excluded from normal workflow variability and its Control After Generate state does not advance. Fixed mode is cacheable. Increment/Decrement/Shuffle/Random deliberately re-execute the Prompt Enhancer because each queued API prompt can carry a different serialized X/Y selection. Downstream nodes may still reuse cached work when a cycled prompt eventually returns to identical output and all of their effective inputs are unchanged.

# Updating

For manual updates:

1. Stop ComfyUI.
2. **Delete the existing `ComfyUI-Local-GGUF-LLM` directory completely.** Do not merge/extract a new release over an older folder; obsolete frontend or helper files can otherwise survive and make debugging unreliable.
3. Extract/copy the new complete `ComfyUI-Local-GGUF-LLM` directory into `custom_nodes`.
4. Start ComfyUI.
5. Hard-refresh the browser (`Ctrl+F5`) if the old frontend remains cached.

The package removes obsolete versioned frontend files from older releases at import time. It also removes the legacy bundled H3 Shot Generator backend/frontend left by old combined builds.

# v0.18.100-alpha release notes

CPU-only inference is now a first-class runtime mode. **Compute Mode = CPU Only** is a hard boundary: model layers are forced to `n_gpu_layers=0`, KV/KQV offload stays on CPU, operator offload is disabled, and multimodal projector execution is kept on CPU. CPU-only model loads also skip CUDA synchronization and VRAM verification snapshots. `Auto` selects CPU-only behavior when ComfyUI has no visible CUDA/ROCm accelerator or the installed llama.cpp binding explicitly reports that GPU offload is unavailable.

CPU Auto tuning is topology and affinity aware. `CPU Threads = 0` resolves to the physical-core count for token generation, while `CPU Batch Threads = 0` resolves to the logical CPUs available to the ComfyUI process. A fully exposed 64-core/128-thread Threadripper 3990X therefore resolves to 64 generation threads and 128 prompt-processing threads. Explicit thread values are preserved.

NUMA is now passed directly to supported llama.cpp bindings. **NUMA = Auto** enables `Distribute` only for CPU-only inference when more than one NUMA node is exposed; otherwise it remains disabled. Explicit `Disabled`, `Distribute`, `Isolate`, and `Numactl` modes are available. The server UI exposes Compute Mode, CPU generation/batch threads, NUMA mode, and operator offload, reports the resolved CPU profile, and provides a **Use CPU-only Auto** button for a one-step high-core-count baseline. The new **CPU Only (Auto / High Core)** memory preset provides the same baseline to legacy/internal memory-preset consumers.

The service-side estimator and speculative-decoding resolver now treat `Auto` as CPU-only on no-CUDA hosts before model load. Native MTP therefore does not advertise itself as active when its full-GPU requirement cannot be met. The Local LLM Server frontend moved to `local_llm_server_v102.js` to force a cache-distinct update.

# v0.18.99-alpha release notes

Prompt Cycle now uses ComfyUI's actual per-item queue lifecycle instead of custom `serializeValue()` overrides or an execution-time backend cursor. The existing Seed **Control After Generate** widget is the lifecycle carrier because ComfyUI creates it as a real node widget and calls widget `beforeQueued` before `graphToPrompt()` and `afterQueued` after each queued item is submitted.

For every non-Fixed Run × N item, `beforeQueued` plans exactly one cycle entry and writes the selected `prompt_history_index`, matching `enhanced_prompt`, and a unique `prompt_cycle_queue_seq` into ordinary schema widgets **before** ComfyUI builds that API prompt. `afterQueued` commits exactly one next-entry advance. JavaScript does not use completion events to advance queue-prepared cycle items, so a fast Prompt Enhancer execution cannot move X/Y while later batch items are still being serialized.

Workflow serialization is also read-only with respect to the stored prompt array while a cycle item is being prepared. This closes the remaining path where `graph.serialize()`/`onSerialize` could reconcile the visible editor while a Run × N item was being frozen. The backend simply consumes the X/Y already present in that queued graph; it does not reinterpret the item through shared cursor state.

`prompt_cycle_queue_seq` is a monotonic cache discriminator only; it never selects the prompt. Positive queue values produce distinct `IS_CHANGED` signatures, while legacy/headless clients retain the non-cacheable fallback. The frontend moved to `prompt_enhancer_dom_v0651.js`, and the node version is `0.6.51-alpha`.

# v0.18.97-alpha release notes

Prompt Cycle was rebuilt around one authority: the Python backend state machine. `Increment`, `Decrement`, `Shuffle`, and `Random` no longer depend on ComfyUI widget `beforeQueued` / `afterQueued` callbacks, no longer carry a frontend cycle cursor, and no longer use a queue-prepared snapshot path. Every non-fixed execution is intentionally non-cacheable and consumes exactly one backend cycle entry.

The legacy `prompt_cycle_queue_seq` input remains only for workflow/schema compatibility and is forced to `0` by the v0.6.48 frontend. Positive values from older cached frontends are ignored by the backend. Runtime scope, state id, cycle revision, and shuffle state remain API-only values so each workflow/tab has an isolated cursor without persisting runtime ownership into workflow/image metadata.

The visible X/Y selector is updated from the backend execution result. A single queued run should therefore visibly move after that Prompt Enhancer execution completes; Run × N advances once per executed item and the UI follows those execution results.

The Prompt Enhancer frontend moved to `prompt_enhancer_dom_v0649.js` to force a cache-distinct load.

# v0.18.95-alpha release notes

Prompt Cycle queueing was rebuilt rather than patched on top of the 0.18.90-0.18.93 implementations. For normal browser queueing, there is now one owner of cycle progression: the hidden `prompt_cycle_queue_seq` widget. Its `beforeQueued` callback freezes the current X/Y selection into that queue item and serializes a unique monotonic nonce. Its `afterQueued` callback is the only place that advances the live selector to the entry for the next item.

The old frontend private cursor and frontend-to-backend cycle synchronization path were removed from normal queue operation. The backend `PromptCycleStore` remains only as a compatibility fallback for legacy/headless submissions that arrive without the queue nonce. Queue-prepared submissions always trust the X/Y index serialized into that specific item, so backend state cannot advance them a second time.

Each queue-prepared non-Fixed item now has a distinct `IS_CHANGED` signature using its queue nonce. That prevents several Run x N items from collapsing into one cached Prompt Enhancer execution merely because the surrounding workflow snapshot is otherwise identical. A failed queue submission does not consume a cycle entry because advancement happens only after acceptance.

Shuffle was also rebuilt as a deck. The selected entry is consumed once, every other entry is emitted once before a new deck begins, and a deck-boundary repeat is avoided. The runtime journal now remembers whether Shuffle has started even when the remaining deck is momentarily empty, so workflow-tab remounts do not accidentally restart the first deck.

The Prompt Enhancer frontend was moved to `prompt_enhancer_dom_v0647.js` to force a cache-distinct load. The queue lifecycle carrier is now the required `Prompt Cycle` widget rather than the hidden queue nonce widget. The dependency-free suite is run as part of release validation.

# v0.18.93-alpha release notes

This release corrects the Prompt Cycle regressions in 0.18.90-0.18.92. Current ComfyUI queues each batch item with one `beforeQueued -> graphToPrompt -> afterQueued` lifecycle. Prompt Cycle now prepares exactly one immutable cycle selection in `beforeQueued`, serializes that selection into the queued API prompt, and commits the next private cursor only in `afterQueued` after the item is accepted. Cycling no longer advances from `serializeValue()`, so extra prompt serialization cannot consume a second cycle step.

The lifecycle hook is attached to the **current** `prompt_cycle` widget rather than being marked only on the node. If Nodes 2.0 replaces that widget object after extension initialization, the `promptQueueing` event repairs the hook before ComfyUI starts the per-item queue loop.

The workflow-enhancement toggle now uses explicit boolean normalization throughout the queue path. Values such as the string `"false"` are treated as false instead of JavaScript truthy strings; this was the reason the callback-based 0.18.90/0.18.91 design could bypass cycling completely on affected frontends.

When **Enhance with Workflow** is disabled, the Prompt Enhancer LLM seed still serializes as a stable zero and runtime-only cycle bookkeeping is omitted from the API prompt where possible, preserving ComfyUI cache reuse when the effective enhanced prompt and all downstream inputs are unchanged.
