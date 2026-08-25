
## v0.10.6-alpha UI cleanup

- Removed the standalone **Local GGUF LLM** node from ComfyUI registration. The internal engine remains in the package and is used by the persistent Local LLM service. Public workflow nodes are now **Get Local LLM Service** and **Local LLM Service Generate**.
- Removed benchmark wording from the visible `Use mmap` setting.
- Added concise tooltips only to advanced vision/memory/load controls where the setting can materially affect VRAM, load behavior, or compatibility.
## v0.10.6-alpha — floating status layout/layering

- Floating status box now sizes to its status text up to a 460 px viewport-safe cap; long metric lines wrap instead of being truncated with an ellipsis.
- The second and third lines continue to use the full box width beneath the robot/header row.
- Status updates re-clamp the movable box so content growth cannot push it off-screen.
- The floating box now uses the workflow/canvas UI layer (`z-index: 2`) instead of a top-level overlay layer, so ComfyUI sidebars, menus, dialogs, and popovers naturally render above it. The previous modal-detection/z-index shim was removed.

- Sidebar/menu robot animation now pulses **only** during loading/reloading, prompt processing, or token generation. Fixed a stale `processing` CSS class that could leave the icon flashing after the request finished; stopping/ready/yielded/error states are static.
## v0.10.2-alpha — resident request family-detection fix

- Fixed a v0.10.1 regression in the Local LLM Service request path where `detect_family()` was called with only the filename after llama-cpp-python/KV compatibility work. The helper requires `(metadata, filename)`, so resident-model requests could fail before inference with `TypeError: detect_family() missing 1 required positional argument: 'filename'`.
- The service now uses the same metadata-aware family detection as the loader before applying capability-aware vision soft-ignore behavior.

## v0.10.1-alpha — llama-cpp-python KV enum compatibility

- Fixed false `KV cache type ... is not supported` errors after llama-cpp-python upgrades that no longer re-export a GGML enum at package scope. KV types are now resolved from both the package and low-level binding, with ABI-stable upstream enum values as a final compatibility fallback.
- `q4_1` remains supported by current upstream llama.cpp; this change prevents wrapper-layout differences from being misreported as backend feature removal.

## v0.10.0-alpha — profiled fast VRAM handoff

- **Smart cache-flush gate:** before calling ComfyUI `soft_empty_cache()`, the LLM now measures raw CUDA free VRAM and ComfyUI/PyTorch reclaimable allocator cache. It skips the synchronized cache flush entirely when that cache cannot cover the actual native-VRAM shortfall.
- **Observed-VRAM reload target:** first load still uses the conservative GGUF/KV/mmproj estimator, but subsequent Auto-Yield reloads use the measured native CUDA allocation plus a fixed 256 MiB safety margin. The old extra 5% + ComfyUI inference-reserve stacking is removed.
- **Target-GPU AIMDO full unload:** AIMDO fallback now requests a full unload only on the llama.cpp target GPU instead of calling global `unload_all_models()` across every accelerator.
- **No redundant post-eviction CUDA sync:** the handoff keeps the safety synchronization before residency changes but avoids an additional unconditional sync after ComfyUI's own unload/cache path.
- **Warm/cold GGUF diagnostics:** first loads and fast reloads now log mmap/no-mmap mode, major/minor page-fault deltas, block-input deltas, Linux page-cache size, a warm/storage-backed cache hint, and WSL detection.
- **Server Logs performance breakdown:** the LLM panel now records preload target/source, estimated vs observed VRAM, raw free memory, reclaimable PyTorch cache, whether the cache probe was skipped, CUDA-sync time, ComfyUI eviction time, cache-flush time, native GGUF load time, and page-fault/cache diagnostics.
- **Accurate resident-reuse logging:** a request that reuses an already-resident context now reports `native_load=0` and `load_path=resident-reuse` instead of repeating stale timing from the previous reload.
- The `Use mmap` / `Use mlock` controls remain available; logs report `mmap`, `no-mmap`, or `mmap+mlock` to help diagnose reload behavior.


- For detected text-only models, Service Generate now **blocks new vision connections in the UI** but **soft-ignores existing connected `image(s)` / `video_frames` inputs** at runtime, logging a warning instead of raising an error.
## v0.9.9-alpha — simplified all-or-nothing native memory lifecycle

- **Removed llama.cpp from ComfyUI's `load_models_gpu()` / partial-loading lifecycle.** Auto Yield now keeps a normal native `Llama(...)` context outside `LoadedModel`/`ModelPatcher`; ComfyUI coordination is a thin layer around load/yield only.
- **Strict native load signature:** the full allocation signature (model/mmproj file state + native load kwargs + vision behavior/template controls) gets a stable short signature id. If the signature matches and the context is resident, the load path is a true no-op: no ComfyUI loader calls, no CUDA verification snapshots, and no memory handoff.
- **Whole-context yield only:** when ComfyUI needs driver-visible VRAM, the free-memory hook waits on the native model lock, synchronizes the target GPU, fully closes `llm`, drops strong references, and only then lets ComfyUI perform its own eviction. No fake partial GGUF unload semantics remain.
- **Direct cached reload:** after a successful first verified load, a yielded context is recreated directly with cached `Llama(**load_kwargs)` settings; it is not re-registered or routed through `load_models_gpu()`.
- **Cleanup only on demonstrated failure:** normal Auto Yield is `llm.close()` + reference release. `gc.collect()` runs only if driver-visible VRAM still does not satisfy the requested target (or essentially no native VRAM returned); `soft_empty_cache()` is an additional fallback only when the target is still unmet. Explicit Stop/Reload/model-change cleanup remains thorough.
- **Raw-VRAM-aware ComfyUI yield:** the reverse handoff now compares ComfyUI's requested memory against raw CUDA-driver free memory when available, so reclaimable PyTorch cache cannot hide a real native-allocation conflict.
- Expanded `[Local GGUF LLM PERF]` logging with signature ids, strict resident reuse/no-op events, direct reload paths, full-close timing, free VRAM before/after close, fallback-GC/cache use, handoff strategy, and explicit confirmation that ComfyUI loader registration is disabled.

## v0.9.6-alpha — fast Auto-Yield lifecycle + performance instrumentation

- **Fast managed reload path:** the first load after a model/settings change still performs full GPU-offload verification, CUDA VRAM snapshots, backend-library inspection, and llama.cpp system diagnostics. Auto-Yield reloads of that same verified allocation skip those heavyweight checks and recreate `Llama(...)` directly from cached load settings.
- **Fast Auto-Yield unload:** normal ComfyUI-pressure eviction closes the native llama.cpp context and drops references immediately, without unconditional `gc.collect()` or PyTorch `soft_empty_cache()`. A one-shot GC fallback runs only if driver-visible VRAM fails to increase after closing a substantial native allocation. Explicit Stop/Reload/model-change cleanup remains heavy and thorough.
- **Cache-first VRAM handoff:** before unloading ComfyUI models, the LLM reload path first releases unused PyTorch allocator cache and checks raw CUDA-driver free VRAM. If that alone is enough, no ComfyUI model unload occurs. AIMDO still uses the clean full-model unload when actual model eviction is required.
- Added always-on `[Local GGUF LLM PERF]` lifecycle logs with timings for VRAM room checks, cache release, CUDA synchronization, ComfyUI model eviction, `load_models_gpu`, native GGUF construction, first-load verification snapshots/diagnostics, Auto-Yield close time, fallback GC, and total load/unload time.
- Service request logs now break out queue wait, ComfyUI-idle wait, service-start overhead, total node call, load path (`verified-first-load` vs `fast-reload`), native load time, ComfyUI `load_models_gpu` time, VRAM handoff time, prompt evaluation, decode, and prior-yield close time.

## v0.9.4-alpha — SillyTavern compatibility + endpoint diagnostics

- Logs the exact OpenAI route and `stream` mode used by each client request.
- Chat responses mirror visible text in both `delta.content`/`message.content` and legacy `choices[].text` for parser compatibility.
- `/v1/completions` and `/local-llm/v1/completions` now support real SSE streaming instead of returning JSON to a streaming request.
- Completion/chat SSE logs now report the exact number of visible characters sent to the client.

## v0.9.3-alpha — SillyTavern/OpenAI live-stream fix

- OpenAI-compatible chat completions now use **true SSE streaming** from llama.cpp to the client instead of consuming the native stream internally and returning one buffered SSE body afterward.
- Streams standard `delta.content` and `delta.reasoning_content` events and finishes with an OpenAI-style stop chunk plus `[DONE]`.
- Accepts both dict and Pydantic/OpenAI-style llama-cpp-python stream objects and legacy `choice.text` stream chunks.
- SillyTavern escaped special-token stops such as `<\|im_end\|>` are normalized back to `<|im_end|>` before inference.
- Adds an explicit visible warning if a reasoning model returns reasoning but **zero final response text**, so clients no longer appear to silently succeed with a blank message.
- Server logs now report exactly how many visible and reasoning characters were sent over SSE.

## v0.9.2-alpha — AIMDO-safe two-way VRAM handoff

- Fixed the managed LLM/ComfyUI VRAM handoff for DynamicVRAM/AIMDO. ComfyUI's generic eviction order can partially reduce a VBAR model before reaching the all-or-nothing llama.cpp adapter; the plugin now yields the resident managed LLM first when ComfyUI has a real memory shortfall.
- When the LLM itself needs more VRAM and AIMDO is active, the plugin prefers a clean full ComfyUI model unload while idle instead of driving VBAR residency toward a zero watermark with targeted partial eviction. This avoids leaving pinned pages above `watermark 0`.
- CUDA work is synchronized before residency changes, and ComfyUI's own free-memory accounting is preferred over raw driver free-memory counters.
- Non-AIMDO builds retain targeted `free_memory()` behavior.

## v0.9.1-alpha — OpenAI/SillyTavern response fix

- Fixed the v0.9.0 OpenAI endpoint regression introduced when `video_frames` was added to `generate_messages()`: chat/completions and completions now call the service with named arguments, so the client name, sampler overrides, and video input can no longer shift into the wrong parameters.
- OpenAI `stop` arrays are now preserved as arrays end-to-end. Special-token stops such as `<|im_end|>` and `<|im_start|>` are no longer corrupted by the ComfyUI widget's pipe-delimited serialization format.
- The widget stop parser now correctly supports escaped literal pipes (`\|`) while retaining legacy pipe-delimited presets.
- Added non-content diagnostics after every service request (`final chars` vs `reasoning chars`) and a warning for reasoning-only responses, making blank-client responses immediately distinguishable from transport failures.

## v0.9.0-alpha — Phase 1 multimodal compatibility + capability metadata

- Added a central **model capability registry** used by both backend validation and frontend UI. It reports text/vision/audio/embeddings/MTP capability state, whether vision needs an mmproj, preferred llama.cpp vision handlers, and implementation status.
- Expanded model-family-aware VLM handling for current Qwen VL/unified families, Gemma 3/4, GLM-V, LFM-VL, MiniCPM-V, LLaVA variants, Granite Docling, PaddleOCR-VL, Step3-VL, Moondream/NanoLLaVA, Llama 3 Vision, and related llama-cpp handlers when present in the installed binding.
- **Auto mmproj matching is stricter and safer.** Known family mismatches are rejected; same-folder proximity is only a tie-breaker and can no longer select an unrelated projector by itself. Unknown/new families remain allowed when compatibility cannot yet be proven.
- Added **native multi-image input**: an IMAGE batch is preserved in order and can send multiple images in one multimodal request.
- Added **Video Frames** input: an ordered IMAGE batch is treated as video frames and evenly sampled up to `vision_max_frames`, with the same aspect-preserving `vision_max_edge` preprocessing.
- Added capability-aware UI: known text-only models hide/disable vision preprocessing/projector controls while unknown families remain available for forward-compatible Generic MTMD use.
- Added model-info/status capability metadata plus matching-projector validation details for UI/API consumers.
- Corrected service timing semantics after Auto Yield: **Loading**, **Processing**, and **Generating** are separate phases. Prompt/decode tok/s come only from llama.cpp inference counters; model reload time is reported separately. The floating `Last:` line labels full wall time as `total` and appends `load X.Xs` when a reload occurred.

## v0.8.7-alpha

- Fixed a CUDA race when an external/on-demand LLM request tries to load or run while ComfyUI is actively diffusing. External LLM GPU work now waits for ComfyUI's running prompt queue to become idle before model load/reload/generation, preventing `free_memory()` or llama.cpp allocations from racing active diffusion kernels.
- Workflow-owned `Local LLM Service Generate` calls are exempt from the external wait because they already execute serially inside the active ComfyUI prompt.
- The status indicator shows **Waiting for ComfyUI…** while an external request is safely queued behind active workflow execution.

# ComfyUI Local GGUF LLM

### v0.8.5 alpha — bidirectional VRAM handoff + last-request status

- **Auto Yield to ComfyUI is now explicitly bidirectional.** Before a managed GGUF is first loaded or reloaded after eviction, the LLM path checks device free VRAM and asks `comfy.model_management.free_memory()` to release enough ComfyUI-managed cache when necessary.
- The managed LLM remains registered with ComfyUI, so ComfyUI can evict the complete llama.cpp GGUF context, KV cache, and projector allocation when another workflow needs VRAM. The next LLM request reloads it on demand.
- Generation keeps the native model lock for the whole request, preventing a ComfyUI eviction from destroying the llama.cpp context while it is actively generating.
- The floating robot status box keeps **live tok/s** on the top line (`0.0 t/s` while idle) and now adds a persistent **Last:** line showing **completion tokens**, **average generation tok/s**, and **total time spent on the last request**.
- Added explicit status API aliases: `last_tokens`, `last_total_tokens`, `last_average_tokens_per_second`, and `last_request_seconds`.

### v0.8.4 alpha — locked Generate node sizing

- Fixed workflow/tab deserialization being mistaken for a newly created Generate node.
- Saved Generate node width/height are captured before frontend widget installation and reasserted across delayed render frames.
- Tab changes and page refreshes no longer allow preset button installation to auto-fit a loaded node.

### v0.8.3 alpha — stable Generate node sizing

- Fixed **Local LLM Service Generate** changing size during preset/widget edits, workflow tab switches/restores, and workflow execution.
- Canvas redraws no longer call `computeSize()` / `setSize()` or replace the widget array.
- A newly created Generate node may grow once to fit the three preset-save buttons; loaded nodes preserve their saved size exactly.

### v0.8.2 alpha — cleaner Generate node preset UI

- Removed the **include prompt/response in info** control completely. The `info` output is metadata/statistics only; prompt/response logging remains controlled globally in the Local LLM Server UI and stays off by default.
- Each preset save action now sits **directly below its matching selector**:
  - System prompt preset → Save system prompt preset → System Prompt
  - Prompt preset → Save prompt preset → Prompt
  - Sampler preset → Save sampler preset → sampler controls
- The three preset libraries remain independent under:

```text
models/LLM/local_LLM_presets/
├── sampler/
├── system_prompts/
└── prompts/
```

- Added name-keyed Generate-node widget persistence so the frontend-only save buttons can live between normal ComfyUI widgets without shifting saved settings.
- Removed the legacy v0.8.0 sampler-preset migration path; this alpha now treats the structured folders above as authoritative.

### v0.8.1 alpha — split prompt/sampler presets + robust sidebar robot

The lightweight **Local LLM Service Generate** node now keeps three independent preset libraries under the Local LLM model root:

```text
models/LLM/local_LLM_presets/
├── sampler/          # JSON sampler presets
├── system_prompts/   # plain UTF-8 .txt files
└── prompts/          # plain UTF-8 .txt files
```

- **Sampler preset**: `Default`, `Custom`, plus saved sampler JSON files. `Default` mirrors the effective global server defaults and sends no sampler overrides.
- **System prompt preset**: `Custom` plus saved `.txt` system prompts. Editing the System Prompt changes only this selector to `Custom`.
- **Prompt preset**: `Custom` plus saved `.txt` prompts. Editing Prompt changes only this selector to `Custom`.
- Separate **save sampler preset**, **save system prompt preset**, and **save prompt preset** actions write to their corresponding folders.
- The standard ComfyUI seed remains request-local and is not stored in sampler presets.
- v0.8.0 sampler JSON files from the legacy `Local_LLM_Presets/` folder are copied forward non-destructively into `local_LLM_presets/sampler/` when possible.
- Fixed the disappearing sidebar robot again by making the robot a CSS pseudo-element on the registered **LLM launcher button itself**. It no longer relies on ComfyUI creating an `<i>`, `<svg>`, Iconify class, or any other particular child icon DOM. The robot still carries the live status color/pulse.

### v0.8.0 alpha — request presets for Global LLM Generate

The lightweight **Local LLM Service Generate** node now has a real request-preset system:

- `Default` mirrors the effective generation defaults from the running global Local LLM Server (including the resolved model preset) and sends no sampler overrides.
- Changing temperature, Top P/K, Min P, penalties, or max tokens automatically switches the node to `Custom`.
- Named custom presets can be saved from the node with **save custom preset**.
- Presets are stored as readable JSON files under `ComfyUI/models/llm/Local_LLM_Presets/` (the active registered LLM model root).
- Saved preset names automatically appear in the same preset dropdown and are enforced by the backend as well as mirrored into the visible widgets.
- The standard ComfyUI seed is intentionally not stored in presets.
- Changes to server defaults or a saved preset are included in the node cache fingerprint so cached text is invalidated correctly.

## Previous releases

### v0.7.9 alpha — reliable sidebar robot icon

- Removed the injected custom sidebar robot SVG that bypassed ComfyUI's native icon slot and could sit a few pixels off-center.
- The sidebar now uses the `icon-[lucide--bot]` icon created by `registerSidebarTab` itself, so its position/size is controlled entirely by ComfyUI's standard sidebar layout.
- The native robot icon still carries all live status colors and pulse states; queue/error badges are unchanged.

### v0.7.7 alpha — live-only tok/s

- Runtime **tok/s is now strictly live**: it shows the current generation rate only while the service is actively generating and resets to `0.00` immediately afterward.
- Removed the UI fallback to the previous request's completed-generation speed.
- The floating status monitor likewise shows `0.0 t/s` whenever the LLM is not actively generating.
- Historical request speed is still retained internally for logs/diagnostics, but is no longer presented as the current rate.

### v0.7.6 alpha — sidebar icon sizing/alignment

- The LLM sidebar launcher now explicitly follows ComfyUI's standard vertical sidebar layout: the robot status icon is centered **above** the `LLM` label instead of sitting beside it.
- Reduced the robot status icon to a 20×20 px visual footprint so it matches nearby ComfyUI sidebar icons without crowding the label.
- The robot remains the live status indicator (gray/green/pulsing green/amber/blue/red).

### v0.7.5 alpha — robot status indicator + floating monitor

- The **LLM sidebar icon is now a simple robot head and is itself the live status indicator**: gray = stopped, green = ready, pulsing green = generating, amber = loading/reloading, blue = yielded to ComfyUI, red = error.
- The sidebar badge is now reserved for queue count or `!` on error; normal ready/generating state no longer uses a redundant dot badge.
- Added a small **draggable floating LLM status box** showing the robot state, current tokens/sec while generating, generated-token count, queue count, and yielded/loading/error state.
- Clicking the floating status box opens the full Local LLM Server modal. Dragging it moves it without opening the modal.
- Floating-box position persists in the browser independently of workflows/tabs.
- Added **Server → Interface → Show floating status box**, saved globally in `local_llm_server.json`. It can be toggled without reloading the model.

### v0.7.4 VRAM auto-yield

The global server now defaults to **Auto Yield to ComfyUI**. The llama.cpp context stays hot and is reused without re-entering ComfyUI model loading on every request. It is registered once with ComfyUI's model manager so `free_memory()` can evict the whole GGUF + KV context only when another ComfyUI model actually needs VRAM. A later LLM request reloads it automatically. Choose **Keep Resident** if you explicitly want KoboldCPP-style pinned residency instead.


## v0.7.4 alpha — live throughput, privacy controls, generation-safe modal

- The LLM sidebar modal can now be opened and used **while generation is active**. Status no longer waits on llama.cpp's native model lock.
- The sidebar launcher is kept clickable while ComfyUI is executing; model-conflicting actions remain disabled inside the modal.
- Global service generation now consumes llama.cpp streaming chunks internally so it can report **live tokens/sec** without exposing partial response text.
- The ComfyUI console prints a throttled live line during generation, approximately once per second, e.g. `Generating for SillyTavern: 42.7 tok/s • 318 tokens • 7.4s`.
- The modal Runtime panel shows current tok/s and current generated-token count while inference is active. (As of v0.7.7, tok/s resets to zero when idle instead of falling back to the previous request.)
- Added privacy settings: **Log prompt content** and **Log response content**, both OFF by default. Image data URLs are always omitted and logged text is bounded.
- `Local LLM Service Generate` always shows **System Prompt** and **Prompt** so they remain editable. The privacy toggle is now **include prompt/response in info**, OFF by default; when enabled it adds the System Prompt, Prompt, and Response to the `info` output. Global log-content controls remain separate.
- Added midpoint context choices above 4K: 6K, 12K, 24K, 48K, 96K, 192K, 384K, and 768K, still capped by the selected GGUF's advertised native context.
- External OpenAI responses remain buffered in this alpha; internal streaming is used for live service telemetry.

## v0.7.1 alpha — explicit Save/Reload + model-aware context sizes

- **Save** now only persists configuration. It never reloads the model.
- **Reload Model** is a separate action and is disabled when no model is loaded, while generation/loading is active, or while there are unsaved edits.
- **Start** also uses saved configuration only; save edited values before starting.
- Fixed a possible `Saving…` hang by running configuration updates off ComfyUI/aiohttp's event-loop thread before emitting websocket status.
- Added unsaved-change tracking and clear button enable/disable states.
- Context is now a constrained selector rather than a free-form integer. The options use standard context sizes and are capped to the selected GGUF's native `*.context_length` metadata when available. The backend validates/clamps context too, so API/config edits cannot bypass the limit.
- Opening the modal no longer reapplies model/memory presets over saved values; presets apply only when selected or when changing models while a preset remains active.


## v0.7.0 alpha — global persistent server + sidebar modal

This release begins the transition from a workflow-owned GGUF node to a **global persistent LLM service** that feels more like running KoboldCPP inside ComfyUI. The existing `Local GGUF LLM` direct node is retained for compatibility.

### New global UI

- Adds an **LLM** item to ComfyUI's left sidebar using the supported `registerSidebarTab` extension API.
- Selecting it opens a popup server-management modal.
- Sidebar state indicator: gray = stopped, amber pulse = loading/reloading, green = ready, green pulse = generating, red = error. Queue count/error also use the sidebar badge when the frontend refreshes it.
- Modal tabs: **Server**, **Model**, **Memory**, **API**, and **Logs**. No collapsible widgets are used.
- Start and Reload save the values currently visible in the modal before loading the model.

### Persistent service behavior

- One process-global service owns the llama.cpp model/context. It remains alive independently of workflows.
- Default/only global residency policy is **Auto Yield to ComfyUI**. ComfyUI sees the remaining device-wide VRAM while Windows/WDDM/NVIDIA or the platform driver handles residency pressure.
- Service states: `stopped`, `loading`, `ready`, `generating`, `reloading`, `stopping`, `error`.
- Requests are serialized through one generation lock, so ComfyUI nodes and external clients cannot concurrently corrupt one llama.cpp context.
- Load-affecting changes mark **Reload required**. Sampling/model-preset changes take effect on the next request without recreating the model.
- Startup modes: **Off**, **On Demand**, **Auto Start**. Auto Start is off by default.
- Configuration persists globally in the active ComfyUI user directory as `local_llm_server.json`.

### OpenAI-compatible API

When enabled, the same persistent model is available at:

```text
/local-llm/v1/models
/local-llm/v1/chat/completions
/local-llm/v1/completions
```

The base URL for SillyTavern is:

```text
http(s)://<your-comfy-host>:<port>/local-llm/v1
```

- External API is **disabled by default**.
- Enabling it automatically creates a strong `Bearer` API key if none exists.
- OpenAI-style multi-turn `messages` are passed through to the GGUF chat template instead of being flattened into one user prompt.
- OpenAI `image_url` content blocks can activate the configured/auto-detected mmproj.
- `stream=true` is accepted in this alpha and returned using valid SSE framing, but it is **buffered until generation completes**. True token-by-token streaming is the next server milestone.

### New service nodes

`Get Local LLM Service` returns `LOCAL_LLM_SERVICE_API`, a live facade over the global singleton.

`Local LLM Service Generate` sends a workflow request through the global server without loading another GGUF. Its preset selector uses **Default** (mirrors the current effective server defaults), **Custom**, or any named JSON preset saved in `models/llm/Local_LLM_Presets/`. It uses standard ComfyUI 64-bit seed / control-after-generate behavior.

### Current alpha limits

- One resident model/profile at a time.
- True incremental token streaming and request abort are not implemented yet.
- The sidebar launcher uses the public custom-sidebar registration API; the popup itself is a self-contained modal so it does not depend on workflow/node UI state.

---


## v0.6.0 live linked-node API

- Replaces the old fifth `settings` output with **`api`** of ComfyUI type **`LOCAL_GGUF_LLM_API`**.
- The API is a stable facade, not the raw `llama_cpp.Llama` object, so downstream nodes remain valid if the native model is reloaded, evicted, or its vision handler changes.
- Query effective settings at any time with `get_settings()`, `get_model_settings()`, `get_generation_settings()`, `get_memory_settings()`, `get_prompting_settings()`, `get_advanced_settings()`, or `get("memory.context_size")`. Every settings query returns a copy so downstream nodes cannot mutate the source configuration accidentally.
- Query live residency with `status()` / `is_loaded()`.
- Linked custom nodes can reuse the configured/persistent LLM with `api.generate(...)`, `api.query(...)`, or `api.generate_text(...)`. The call is serialized through locks and reuses the existing native llama.cpp context whenever load-affecting settings still match.
- API generation accepts temporary request-time overrides (sampling, thinking mode, max tokens, seed, stop sequences, etc.) without mutating the source node. Model/KV/memory allocation settings are intentionally read-only through the API.
- The API does **not** retain the source `IMAGE` tensor, preventing cached API outputs from pinning a large image tensor. A downstream node may pass `image=...` explicitly for vision inference.
- The original output order is preserved for the first four outputs: `response`, `thinking`, `info`, `tokens`; output 5 is now `api`.
- The persistent native cache key also includes model/mmproj path + size + nanosecond mtime, so replacing a GGUF under the same filename reloads the actual native model rather than only invalidating ComfyUI's text-output cache.

### Downstream node input

```python
"llm_api": ("LOCAL_GGUF_LLM_API",)
```

Example queries:

```python
model = llm_api.get("model.name")
context = llm_api.get("memory.context_size")
settings = llm_api.get_generation_settings()
status = llm_api.status()
```

Example generation from a linked node:

```python
result = llm_api.generate(
    prompt="Rewrite this as a concise video prompt.",
    temperature=0.4,
    max_tokens=512,
)
text = result["response"]
```

`generate_text(...)` is available when only the final response string is needed.

## v0.5.9 persistent native memory policy

- New nodes default to **Persistent (Driver Managed)** model retention. The llama.cpp model, context, KV cache and optional mmproj stay alive between executions and are **not** registered in ComfyUI's `current_loaded_models` eviction list.
- Persistent mode behaves more like a long-running KoboldCPP/llama.cpp server: ComfyUI sees the device-wide free VRAM that remains, while Windows/WDDM/NVIDIA (or the platform driver) controls physical residency under pressure.
- ComfyUI is consulted only before the **first** persistent native load, and only if current free VRAM is below the conservative load estimate. If enough VRAM is already free, no ComfyUI models are proactively unloaded.
- After the native model is resident, normal ComfyUI memory pressure does **not** destroy/recreate the LLM. It unloads only when model/load-affecting settings change, `Unload Cached LLM Now` is pressed, or `Unload After Run` is selected.
- **Model management is now independent of Memory Preset.** Selecting/changing a KV/context memory preset no longer silently changes the retention policy or flips the memory preset to Custom when you alter retention.
- `ComfyUI Managed` remains available as an optional compatibility/low-VRAM mode, and `Unload After Run` remains available for deterministic release.
- Runtime `info` now reports `persistent_native` and whether a conditional first-load ComfyUI release was actually needed.

## v0.5.8 standard ComfyUI seed management

- Replaces the custom `seed = -1` random sentinel with ComfyUI's standard unsigned 64-bit `seed` widget (`0..0xFFFFFFFFFFFFFFFF`).
- Enables the normal **control after generate** selector: Fixed, Increment, Decrement, and Randomize.
- Fixed seed + unchanged inputs remains cacheable; changing the seed through the standard control naturally reruns the node.
- External GGUF/mmproj file fingerprinting remains in place, so replacing a model under the same filename invalidates cached output.
- llama.cpp uses a 32-bit sampler seed. The node deterministically maps the full 64-bit ComfyUI seed to a non-random uint32 and exposes both `seed` and `llama_seed` through `api.get_generation_settings()` in v0.6.0.
- Legacy workflows saved with `seed = -1` are migrated to seed `0` on load.

## v0.5.7 deterministic ComfyUI output caching

- Fixed seeds (`seed >= 0`) now use normal ComfyUI output caching. If the model, prompt, system prompt, settings, seed, and upstream inputs are unchanged, the LLM node is not executed again and ComfyUI reuses its cached outputs.
- Adds an external-file fingerprint for the selected model GGUF and resolved mmproj using real path + file size + nanosecond modification time. Replacing/updating a GGUF under the same filename invalidates the cached output automatically.
- The fingerprint does **not** hash the entire GGUF file, avoiding multi-gigabyte disk reads on every queue.
- Model residency is independent from output caching: ComfyUI may evict the native llama.cpp model from VRAM while still retaining the previous text outputs. An identical fixed-seed queue can therefore be satisfied from the output cache without reloading the GGUF.

## v0.5.6 typed settings output (historical; replaced by API in v0.6.0)

- Adds a fifth output named **`settings`** with ComfyUI type **`LOCAL_GGUF_LLM_SETTINGS`**.
- The output is a Python dictionary intended for direct connection to another custom node; it is not a JSON string.
- It contains the **effective** configuration after Model and Memory presets have been resolved, so downstream nodes see the settings actually used for inference.
- The payload is versioned (`schema=local_gguf_llm_settings`; v0.5.8 emits `schema_version=2`) and grouped into `model`, `generation`, `memory`, `prompting`, and `advanced` sections.
- Existing output indices are preserved: `response`, `thinking`, `info`, and `tokens` stay in their original positions; `settings` is appended as output 5.

A downstream custom node can accept it with:

```python
"settings": ("LOCAL_GGUF_LLM_SETTINGS",)
```

Example access:

```python
ctx = settings["memory"]["context_size"]
kv_k = settings["memory"]["kv_cache_k"]
temp = settings["generation"]["temperature"]
model = settings["model"]["name"]
```

## v0.5.5 fixed-layout UI

- Removes the collapsible-section feature completely. There are no collapse headers, collapse callbacks, section-hide reasons, or saved collapse state.
- Uses a normal fixed ComfyUI widget layout; only capability-driven fields such as custom chat format, tensor split, and Mirostat detail fields are conditionally hidden.
- Removes legacy `local_gguf_ui` / `local_gguf_sections` workflow properties on the next save.
- Keeps frontend-only Refresh/Unload buttons at the very end of the widget list instead of inserting them between backend settings, reducing positional serialization risk.
- Renames the frontend JavaScript file so the old collapsing implementation cannot be reused from browser cache.
- Keeps model presets, memory presets, named-value persistence, GPU-name detection, ComfyUI-managed memory, vision support, System Prompt, and Prompt unchanged.

## v0.5.4 collapse/UI-state fix

- Fixes collapsible section buttons after workflow load/tab switching by resolving live `node.properties` at click time instead of retaining pre-config state.
- Synchronizes widget `hidden`, combo `values`, and canvas-only flags with ComfyUI Nodes 2.0 reactive widget state as well as Classic LiteGraph.
- Keeps the simplified Prompting section: only System Prompt and Prompt.

## v0.5.3 simplified prompting

- Removes the Prompt Preset system completely. There are no MiniMax H3, Krea 2, prompt-detail, target-profile, or hidden prompt-preset controls in the node or backend.
- Prompting contains exactly two controls: **System Prompt** and **Prompt**.
- The LLM receives the System Prompt exactly as entered; the node no longer appends target-specific runtime instructions.
- Model presets and Memory / VRAM presets remain independent and unchanged.
- v0.5.2 name-keyed workflow persistence is retained; older saved prompt-preset fields are ignored during migration and disappear on the next save.

## v0.5.2 workflow persistence / tab-switch fix

- Fixes settings shifting, reverting, becoming `null`, or causing node-validation errors after browser refresh, workflow reload, copy/paste, and workflow-tab changes.
- Root cause: frontend-only collapsible-section headers and utility buttons were inserted between normal widgets with `widget.serialize = false`. Current ComfyUI serializes `widgets_values` by full widget index but restores legacy arrays with a compact counter, so skipped UI widgets can shift every later value.
- Section headers and utility buttons are now **workflow-serializable but API-prompt-excluded**: `widget.serialize = true`, `widget.options.serialize = false`. This keeps positional ordering stable while ensuring these UI controls are never sent to the Python node.
- Adds a second, name-keyed persistence layer (`local_gguf_values_v1`) for all server-defined settings. Current `widgets_values_named` is also written explicitly. This makes the node resilient to future UI widget reordering.
- During workflow load, saved settings are authoritative. The frontend no longer reapplies Model/Memory presets from `nodeCreated` or `loadedGraphNode`. Presets are applied only to a genuinely new node or when the user actively changes a preset/model.
- Passive model/GPU refresh after workflow restore updates available dropdown choices and capability visibility without changing saved values.
- Existing v0.5.1 workflows with correct `widgets_values_named` are restored by name before dynamic UI logic runs. The new serializer then self-heals them on the next save.
- Collapsed-section state continues to live in workflow properties and now survives tab changes without causing widget-value shifts.

## v0.5.1 reasoning output normalization

- `thinking` output is now independent of `thinking in response`: it always returns every reasoning trace the backend/model exposes.
- `Strip` now removes reasoning markup from `response` while preserving the extracted text on `thinking`.
- Supports structured `reasoning_content` / `reasoning` / `thinking` / `analysis` fields, content-block reasoning, multiple `<think>` blocks, Qwen delimiter-only `</think>` output, unfinished reasoning due to token limits, Gemma thought channels, and GPT-OSS/Harmony-style analysis channels.
- Adds `thinking_extraction` diagnostics to the `info` output.

## v0.5.0 ComfyUI memory management + verified GPU offload

- **`ComfyUI Managed` is now the default model-management mode.** The GGUF LLM is registered with ComfyUI's model manager through a native-allocation adapter. ComfyUI can keep it resident while memory is available and evict the complete llama.cpp context automatically when another model needs VRAM. The next LLM execution reloads it transparently.
- llama.cpp cannot migrate already-created GGUF layers between GPU and CPU the way ComfyUI can move PyTorch weights, so managed partial-unload requests intentionally become **whole-context eviction**. This releases model VRAM, KV cache, and active mmproj resources together.
- Model-management choices are now `ComfyUI Managed`, `Keep Loaded / Pinned`, and `Unload After Run`. `Keep Loaded / Pinned` intentionally stays outside ComfyUI eviction; `Unload After Run` deterministically frees the native context after every generation.
- Before a first managed load, the node estimates native VRAM from GGUF file size, GPU-layer count, context/KV metadata and KV quantization. After loading, device-wide free VRAM is measured and the adapter switches to the **actual native allocation** for later ComfyUI accounting.
- **GPU offload is verified, not assumed.** If `gpu_layers != 0` but model loading consumes less than 8 MiB of visible GPU VRAM, the node stops with a diagnostic error instead of silently running the LLM on CPU. The error includes the llama.cpp GPU-offload capability hint, system-info string, and detected CUDA/backend libraries.
- Successful runs expose `gpu_backend.gpu_offload_verified`, per-GPU native VRAM deltas, CUDA/HIP runtime info and llama.cpp system info in the `info` JSON output.
- New nodes default to `None (single GPU)` split mode. ComfyUI Managed mode still rejects multi-GPU Layer/Row/Tensor split when multiple GPUs are visible because one manager entry cannot accurately account for allocations across devices. Multi-GPU llama.cpp is available with `Persistent (Driver Managed)` or `Unload After Run`.
- Old workflows using `Keep Loaded` or `Keep Loaded / Pinned` migrate to `Persistent (Driver Managed)`.

## v0.4.1 GPU selector improvement

- `main GPU` is now a real dropdown labeled with the accelerator name and detected VRAM, e.g. `0 — NVIDIA GeForce RTX 4090 (24.0 GiB)`.
- GPU numbering comes from PyTorch first, so it follows the same logical device order ComfyUI sees (including `CUDA_VISIBLE_DEVICES` remapping).
- Falls back to `nvidia-smi`, then backend device 0, if PyTorch cannot enumerate GPUs.
- Existing workflows that stored `main_gpu` as an integer are migrated in the frontend to the matching named GPU without changing the numeric index sent to llama.cpp.
- `Refresh Models / GPUs` refreshes both GGUF lists and GPU labels.
- The backend still passes only the numeric logical GPU index to llama.cpp.

## v0.4.0 vision compatibility fix

- Fixes `No compatible multimodal handler was found` on current upstream `llama-cpp-python` builds. The node previously searched for `GenericMTMDChatHandler` and family-specific handlers but accidentally omitted upstream's generic `MTMDChatHandler`.
- Supports both current API families:
  - upstream/older builds: `MTMDChatHandler(clip_model_path=...)`
  - newer MTMD/JamePeng-style builds: `GenericMTMDChatHandler` / `MTMDChatHandler` with `mmproj_path=...`
- Falls back to older dedicated handlers such as Qwen/Gemma/LLaVA when generic MTMD is unavailable.
- A selected Vision/mmproj no longer forces multimodal initialization for a text-only run. The projector is activated only when the ComfyUI `IMAGE` input is actually connected.
- Hybrid Qwen vision models use `ctx_checkpoints=0` automatically when the installed binding exposes that option.
- Vision errors now report the installed llama-cpp-python version and detected handler classes.


**Version 0.3.0**

A single comprehensive ComfyUI node for running local GGUF LLMs directly from `ComfyUI/models/llm/` with optional vision/mmproj input.

## Features

- Recursive model selector for `models/llm/**/*.gguf`
- Separate Vision/mmproj selector with `None` and `Auto (matching mmproj)`
- Optional ComfyUI `IMAGE` input (supports image batches)
- GGUF metadata detection without loading the full model
- Independent **Model** and **Memory / VRAM** preset systems
- Built-in model presets for Qwen3.8, Qwen3.5, Qwen3, GPT-OSS 20B, Mistral/Ministral, Gemma 3, Llama 3.x, DeepSeek R1 Distill, Phi-4 Reasoning, and Nemotron 3 Nano
- Independent K/V KV-cache quantization (`f16`, `q8_0`, `q5_0`, `q4_0`, etc., depending on your llama.cpp build)
- GPU/CPU KV cache selection
- Context size, GPU layers, flash attention, n_batch/n_ubatch, mmap/mlock
- Multi-GPU split mode / tensor split
- Model management: `Persistent (Driver Managed)` (default), optional `ComfyUI Managed`, or `Unload After Run`
- Simple prompting with only **System Prompt** and **Prompt**
- Model/Memory preset values remain editable; changing a preset-owned setting automatically switches that preset to **Custom**
- Manual Refresh Models and Unload Cached LLM buttons
- Outputs: response, thinking, info JSON, token count, live linked-node API facade


## 0.3.0 model-preset correctness pass

This release separates **model-recommended inference behavior** from local output/task controls. In particular:

- Qwen3.8 Thinking uses the current official sampler: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=0`, `repeat_penalty=1` with `reasoning_effort=XHigh`.
- Qwen3.8 Non-Thinking uses `0.7 / 0.80 / 20 / 0 / 1.5 / 1` and a real `enable_thinking=false` chat-template argument.
- Qwen3.5 has separate General Thinking, Coding Thinking, General Non-Thinking, and Reasoning Non-Thinking presets.
- `thinking in response` (Strip/Keep), `max tokens`, and `seed` are **not model-preset-owned settings**. They no longer force the model preset to Custom and selecting a model preset does not overwrite them.
- The Model Preset dropdown is now **family-specific after GGUF detection**. For example, Qwen3.8 exposes only Auto, Custom, Qwen3.8 Thinking, and Qwen3.8 Non-Thinking; changing to a different model family discards incompatible preset choices and returns to Auto.
- `preserve thinking (history)` is distinct from output stripping: it controls historical reasoning in supported Qwen3.8 templates. This node is currently single-turn, so it only becomes operationally important once prior assistant history is supplied.
- Text-only Qwen thinking/reasoning template controls are injected per run without reloading the model. Vision handler behavior is cached with a stable key, so unchanged vision runs reuse the model while a genuine template-mode change reloads when required for correctness.
- Vision chat-handler resources are explicitly closed before model unload to work around current llama-cpp-python mtmd cleanup issues.

`max_tokens` intentionally remains task-specific. Official reasoning-model cards often recommend very large output budgets, but automatically forcing 32K-262K generation for a short generation task would be undesirable.

## 0.2.0 frontend compatibility fixes

This release rewrites the UI integration around ComfyUI's supported extension hooks instead of prototype hijacking. Controls are marked `canvasOnly` before widget construction so dynamic widget behavior uses the mutable LiteGraph widget state reliably on current Vue Nodes 2.0 frontends.

It also:

- refreshes model and vision combo values without losing a saved selection
- uses `nodeCreated` / `loadedGraphNode` extension hooks for stable lifecycle handling
- filters llama-cpp-python constructor options for version compatibility while refusing to silently ignore explicitly requested KV-cache controls
- fixes Qwen family detection so `Qwen3-8B` is not mistaken for `Qwen3.8`

After updating the custom-node folder, restart ComfyUI and hard-refresh the browser (`Ctrl+F5`) so the old JavaScript extension is not cached.

## Install

Copy this directory into:

```text
ComfyUI/custom_nodes/ComfyUI-Local-GGUF-LLM/
```

Restart ComfyUI.

### llama-cpp-python

This package intentionally does **not** auto-install `llama-cpp-python`, because blindly installing from PyPI can replace a CUDA build with a CPU-only wheel.

Install a build appropriate for your environment into the same Python environment ComfyUI uses. Verify:

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

For CUDA, use the wheel/build instructions for your installed CUDA/Python version.

## Model layout

```text
ComfyUI/models/llm/
├── Qwen/
│   ├── Qwen3.8-27B-Q4_K_M.gguf
│   └── mmproj-Qwen3.8-27B-F16.gguf
├── GPT-OSS/
│   └── gpt-oss-20b-Q4_K_M.gguf
└── Gemma/
    ├── gemma-3-12b-it-Q4_K_M.gguf
    └── mmproj-gemma-3-12b-f16.gguf
```

Subfolders are supported.

## Presets

### Model preset

Controls only **model-recommended chat-template behavior and sampling**. `Auto (Detected)` reads GGUF metadata + filename and applies the closest built-in family preset. Editing a preset-owned sampler/template setting changes the selector to `Custom` without discarding your values.

The following are intentionally independent of Model Preset: `thinking in response` (Strip/Keep), `max tokens (task limit)`, and `seed`. `preserve thinking (history)` is a Qwen3.8 chat-history feature and is not the same thing as keeping `<think>` text in the `response` output.

### Memory preset

Built-ins include Balanced, Maximum Quality, High Quality KV, Low KV Memory, Minimum KV Memory, CPU KV Cache, and CPU / Low VRAM. Editing context/KV/offload/load controls changes the selector to `Custom`.

Default Balanced KV is:

```text
K = q8_0
V = q5_0
```

### Prompting

The Prompting section has only two multiline text fields:

- **System Prompt** — passed as the system-role message exactly as entered.
- **Prompt** — passed as the user-role message.

There is no prompting preset layer or automatic diffusion-model prompt rewriting.

## Notes on vision

`llama-cpp-python` multimodal APIs are evolving. This node detects the installed API at runtime:

1. Prefer a modern `mmproj_path`/generic multimodal path when the installed `Llama` exposes it.
2. Otherwise try dedicated model-family handlers such as Qwen/Gemma handlers.
3. If neither is available, return a clear update/build error rather than silently ignoring the image.

The exact vision models supported therefore depend on the llama.cpp/llama-cpp-python build you install.

## KV cache formats

The UI offers common llama.cpp cache types. Availability depends on the installed build. Unsupported formats fail with an explicit message rather than silently falling back.

## Multi-GPU tensor split

Enter comma-separated proportions, for example:

```text
1,1
```

or

```text
0.7,0.3
```

Leave blank for llama.cpp defaults.

## Stop sequences

Separate multiple sequences with `|`. Literal `\\n` is converted to a newline.
