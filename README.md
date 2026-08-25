# ComfyUI Local GGUF LLM

A persistent local GGUF LLM service for ComfyUI, powered by `llama-cpp-python`.

The **LLM sidebar icon opens the server as a modal**; it is intentionally not a persistent sidebar panel.

The extension is designed to keep a local language model available for ComfyUI workflows without forcing the GGUF to remain in VRAM when diffusion or video models need the GPU. It provides a global LLM service, a workflow generation node, optional vision support, model/memory presets, prompt presets, performance logging, and an optional OpenAI-compatible HTTP endpoint.

## What it does

- Loads `.gguf` language models from `ComfyUI/models/llm/`.
- Keeps the selected model persistent between LLM requests when practical.
- Can automatically yield the complete native llama.cpp GPU allocation when ComfyUI needs VRAM, then reload it on the next LLM request.
- Reuses the same resident model when the model and load settings have not changed.
- Uses `mmap` by default so Linux/WSL can reuse GGUF pages already cached in system RAM during reloads.
- Supports text-only and supported multimodal/VLM GGUF models.
- Supports one image, an image batch, or sampled video frames for compatible vision models.
- Separates prompt processing speed from generation speed in the status display.
- Provides detailed load, unload, VRAM handoff, page-cache, and inference performance logs.

## Requirements

- ComfyUI
- Python 3.10 or newer
- `llama-cpp-python` installed in the same Python environment used by ComfyUI
- A CUDA-enabled `llama-cpp-python` build if GPU inference is desired

This extension intentionally does not install or replace `llama-cpp-python`. Installing the wrong wheel can replace a CUDA-enabled build with a CPU-only build.

To verify the version available to ComfyUI:

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

Use the llama-cpp-python installation/build instructions appropriate for your operating system, Python version, CUDA version, and GPU.

## Installation

Place the extension at:

```text
ComfyUI/custom_nodes/ComfyUI-Local-GGUF-LLM/
```

Then restart ComfyUI and refresh the browser. After updating the frontend files, a hard refresh may be useful if the browser has cached an older JavaScript file.

## Model folders

Place GGUF files anywhere below:

```text
ComfyUI/models/llm/
```

Subfolders are supported. For example:

```text
ComfyUI/models/llm/
├── Qwen/
│   ├── Qwen3-30B-Q4_K_M.gguf
│   └── mmproj-Qwen3-VL-F16.gguf
├── Gemma/
│   ├── gemma-3-12b-it-Q4_K_M.gguf
│   └── mmproj-gemma-3-12b-f16.gguf
└── Mistral/
    └── mistral-24b-Q4_K_M.gguf
```

The model selector searches recursively through the folder.

## Basic setup

Open the **LLM** item in the ComfyUI side menu. This opens the Local LLM Server panel.

For a basic text model:

1. Select the GGUF under **Model**.
2. Leave **Vision / mmproj** at `None` or `Auto` unless the model supports vision.
3. Leave **VRAM policy** at `Auto Yield to ComfyUI` unless you specifically want the LLM to remain resident.
4. Choose a Model Preset or use `Auto (Detected)`.
5. Choose a Memory / VRAM preset.
6. Save the server settings.
7. Start or reload the model if required.
8. Add **Local LLM Service Generate** to the workflow.
9. Enter a System Prompt and Prompt, then run the workflow.

The service can also run in **On Demand** mode, where the model is loaded when it is first needed.

## Workflow nodes

### Local LLM Service Generate

This is the node to use for normal ComfyUI LLM generation.

Inputs include:

- **System Prompt Preset** — optional saved system prompt.
- **System Prompt** — system-role text.
- **Prompt Preset** — optional saved user prompt.
- **Prompt** — user-role text.
- **Sampling Mode** — use the global server defaults, custom values, or a saved sampler preset.
- **Temperature / Top P / Top K / Min P / penalties** — request-level sampling controls when using Custom mode.
- **Max Tokens** — output-token limit for the request.
- **Seed** — standard ComfyUI request seed.
- **image(s)** — optional single IMAGE or IMAGE batch for a supported vision model.
- **video_frames** — optional ordered IMAGE batch treated as frames from a video.

Outputs:

- **response** — normal final-answer text.
- **thinking** — separated reasoning/thinking text when the model/template provides it.
- **info** — JSON metadata and performance information.
- **tokens** — number of completion tokens generated.

### Get Local LLM Service

**This node is reserved for future workflow integration.**

It exposes a `LOCAL_LLM_SERVICE_API` handle, but there are currently no companion nodes that consume that handle, so it does not provide useful workflow functionality yet. Use **Local LLM Service Generate** for current workflows.

This future workflow API node is separate from the optional HTTP/OpenAI-compatible endpoint described later in this README.

## Persistent model and VRAM behavior

### Auto Yield to ComfyUI

This is the recommended/default VRAM policy for a shared ComfyUI GPU.

While the LLM is resident, llama.cpp owns its native GPU allocation. The GGUF is deliberately kept outside ComfyUI's partial ModelPatcher loading system.

When ComfyUI needs GPU memory:

```text
ComfyUI memory pressure
        ↓
Local LLM yields
        ↓
llm.close()
        ↓
native GGUF/KV/context VRAM is released
        ↓
ComfyUI continues loading its model
```

On the next LLM request:

```text
LLM request
    ↓
check available VRAM
    ↓
make room through ComfyUI only if needed
    ↓
recreate the native llama.cpp context
    ↓
process the real request
```

The extension uses a strict load signature. If the same model and load settings are already resident, the load path is a no-op.

After the first verified load, subsequent reloads use the measured native VRAM requirement for that exact configuration rather than relying only on a conservative estimate.

### Keep Resident

Keeps the LLM loaded until it is manually stopped/reloaded or the configuration changes. This can reduce LLM latency but leaves less VRAM available for diffusion/video models.

Use this only when you know the LLM and the rest of the workflow comfortably fit together.

## mmap and warm reloads

**Use mmap** is normally recommended.

With mmap enabled, the GGUF is memory-mapped. Linux/WSL may keep recently used GGUF pages in the filesystem page cache after the native GPU context is closed. A later reload can therefore read much of the model from system RAM instead of storage.

This is why a second load can be much faster than the first even though the GPU model was fully unloaded.

mmap does not permanently pin those pages. The operating system can reclaim them under RAM pressure.

**Use mlock** keeps mapped model pages locked in system RAM. This can preserve RAM residency more aggressively, but it also increases system-memory pressure. It is normally best left off unless there is a specific reason to use it.

## Important memory settings

### Context size

Maximum context allocated for the llama.cpp context. Larger contexts consume more KV-cache memory.

The server filters the available choices against detected model metadata when the GGUF advertises a native context limit.

### KV cache K / V

Controls the data type used for the key and value KV caches. Lower-bit formats save memory; higher-precision formats use more memory and may be faster or more accurate depending on the model and backend.

Available formats depend on the installed llama.cpp/llama-cpp-python build.

### KV cache location

Controls whether the KV cache is kept on the GPU or CPU where supported. GPU KV is normally faster; CPU KV saves VRAM.

### GPU layers

Controls how many model layers llama.cpp offloads to the GPU. Full GPU offload is normally fastest when the model fits.

### Prompt batch (`n_batch`)

The logical maximum number of prompt tokens processed in one evaluation batch. It affects prompt/context ingestion rather than normal one-token-at-a-time decoding.

### Micro batch (`n_ubatch`)

The physical batch processed at once. Larger values can improve prompt-processing throughput but can require more temporary VRAM.

### Flash Attention

Enables llama.cpp Flash Attention when supported by the model/backend. It can improve performance and can affect KV-cache compatibility.

### Split mode / Main GPU / Tensor split

Advanced multi-GPU controls. Leave these at their defaults for a single-GPU system.

For a custom tensor split, use comma-separated proportions, for example:

```text
1,1
```

or:

```text
0.7,0.3
```

## Model presets

**Auto (Detected)** reads GGUF metadata and the filename to identify the model family and select an appropriate built-in behavior preset when possible.

Model presets can define model-specific chat-template/thinking behavior and recommended sampler values.

If a preset-owned setting is changed manually, the preset changes to **Custom** rather than silently overwriting the edited value.

Task-specific controls such as the request seed and output-token limit remain request-local.

## Memory presets

Memory presets group settings such as:

- context size
- KV-cache types
- KV-cache location
- GPU layers
- prompt and micro batch sizes
- mmap/mlock behavior
- other native llama.cpp memory/offload controls

Use a built-in preset as a starting point, then adjust individual values if needed. Editing a preset-owned value changes the selector to **Custom**.

## Prompt and sampler presets

Saved presets are stored under:

```text
ComfyUI/models/LLM/local_LLM_presets/
├── sampler/
├── system_prompts/
└── prompts/
```

The three preset types are independent.

- **sampler/** contains saved sampler settings.
- **system_prompts/** contains reusable system prompts.
- **prompts/** contains reusable user prompts.

On the Generate node, `Default` sampling means the current global Local LLM Server generation defaults. `Custom` uses the sampler controls serialized in that workflow node.

## Vision and multimodal input

Vision support depends on both the GGUF model and the multimodal support available in the installed llama-cpp-python build.

The extension detects known model families and selects compatible handlers when available. The Vision/mmproj selector can also automatically locate a matching projector.

### image(s)

The **image(s)** input accepts either:

- one ComfyUI IMAGE, or
- an IMAGE batch containing multiple still images.

Images are kept in batch order and sent to the model up to **Vision Max Images**.

### video_frames

The **video_frames** input accepts an ordered IMAGE batch representing frames from a video.

The extension evenly samples frames across that batch up to **Vision Max Frames**. This allows a long sequence of frames to be summarized without sending every frame to the VLM.

### Text-only models

When a model is confidently detected as text-only:

- `image(s)` and `video_frames` are visibly marked unavailable.
- new connections to those sockets are blocked by the UI.
- existing connections are preserved so switching models does not damage the workflow.
- if existing image/video data reaches the node, it is ignored and a warning is logged rather than failing the workflow.

Vision content embedded directly inside an external chat-message payload is not silently discarded; using that with a known text-only model is treated as an invalid request.

## Thinking and reasoning output

For models/templates that expose reasoning separately, the Generate node returns:

- final answer in **response**
- reasoning in **thinking**

The server status and performance calculations distinguish prompt processing from generated-token decoding.

## Status indicator

A draggable status box can be enabled from the LLM server panel.

Its robot/status behavior is:

- **Ready** — green, static.
- **Loading / Reloading** — amber, flashing.
- **Processing** — green, flashing.
- **Generating** — green, flashing.
- **Yielded to ComfyUI** — static.
- **Waiting for ComfyUI** — static.
- **Stopped / Error** — static.

The box shows live generation tok/s while generating. When idle, the live rate returns to zero.

The last-request line can show:

- completion tokens
- average generation tok/s
- prompt-processing tok/s
- total request time
- model load time when a reload occurred

The floating box is kept on the workflow/canvas UI layer so ComfyUI sidebars, menus, dialogs, and popovers appear above it.

## Performance logs

The **Logs** section of the Local LLM Server panel includes detailed timing for model lifecycle and inference.

Useful entries include:

- queue wait
- wait for active ComfyUI execution
- VRAM handoff time
- raw CUDA free memory
- reclaimable PyTorch cache
- whether cache cleanup was skipped or used
- ComfyUI model-eviction time
- native GGUF load time
- observed native VRAM allocation
- mmap/no-mmap mode
- major/minor page-fault changes
- Linux page-cache information
- prompt-processing speed
- generation speed
- native context close/yield time

These logs are intended to distinguish ComfyUI handoff overhead from llama.cpp model construction, storage/page-cache behavior, and actual inference speed.

Prompt and response content logging is disabled by default and can be enabled separately.

## Optional HTTP API

The global service includes an optional OpenAI-compatible HTTP interface for external clients. It is disabled by default and can be enabled in the LLM server panel.

This HTTP interface is independent of the **Get Local LLM Service** workflow node.

The server supports chat/completion-style requests and streaming response transport. External GPU requests wait while ComfyUI is actively executing GPU work so llama.cpp cannot race a diffusion/video workload during VRAM handoff.

If external access is enabled, configure an API key in the server panel as appropriate for your environment.

## Startup modes

The server supports startup behavior from the LLM panel, including:

- **On Demand** — load when first needed.
- **Auto Start** — load shortly after ComfyUI starts.
- **Off** — remain stopped until started manually.

For a machine that frequently switches between diffusion/video work and LLM work, **On Demand + Auto Yield to ComfyUI** is generally the most flexible combination.

## Updating models or settings

Changes to model-allocation settings require the native context to be reloaded. Request-local prompt and sampler changes do not require a model reload.

If the exact native load signature is unchanged and the model is still resident, the extension reuses the existing context.

If the model was yielded to ComfyUI, the next request reconstructs the model using the cached verified load configuration.

## Troubleshooting

### Model loads on CPU instead of GPU

Verify that the `llama-cpp-python` installed in ComfyUI's Python environment was built with the desired CUDA backend. The extension cannot turn a CPU-only llama-cpp-python installation into a CUDA build.

### KV-cache type is reported unsupported

KV formats depend on the installed llama.cpp binding. The extension checks current high-level and low-level llama-cpp-python enum locations, but a backend still has to support the requested cache type.

Try a more broadly supported cache type such as `f16`, `q8_0`, or `q4_0` if necessary.

### Reload is much slower than previous reloads

Check the performance logs for major page faults, block input, and the mmap cache hint. A warm mmap reload can use GGUF pages already cached in RAM; a cold reload may have to read those pages from storage again.

### ComfyUI needs the GPU while the LLM is loaded

Use **Auto Yield to ComfyUI**. The service will fully close the native llama.cpp context when ComfyUI has a real VRAM shortfall and recreate it on the next request.

### Vision input does nothing

Confirm that:

- the selected model actually supports vision
- a compatible mmproj is selected or auto-detected when required
- the installed llama-cpp-python build contains the required multimodal support

Known text-only models intentionally ignore existing connected image/video inputs and log a warning.

## Recommended starting configuration

For a single NVIDIA GPU shared between ComfyUI generation and a local LLM:

- Startup mode: **On Demand**
- VRAM policy: **Auto Yield to ComfyUI**
- Model preset: **Auto (Detected)**
- Memory preset: **Balanced** or another preset appropriate for the model/context
- Use mmap: **On**
- Use mlock: **Off**
- Flash Attention: **On** when supported
- Prompt batch: **2048**
- Micro batch: **512**

Then adjust context size, KV-cache format, and GPU layers according to model size and available VRAM.
