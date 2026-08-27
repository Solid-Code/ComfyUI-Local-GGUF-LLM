# ComfyUI Local GGUF LLM

A single ComfyUI custom-node package for running a persistent local GGUF LLM and using it directly from workflows.

This package includes:

- **Local LLM Generate** — send prompts, images, and sampled video frames to the persistent local LLM.
- **Local LLM Settings** — reusable model/generation settings with loadable Complete Settings Presets. Memory/performance tuning remains in the Local LLM service panel rather than crowding the workflow node.
- **Local LLM Prompt Enhancer** — bundled **v0.6.12-alpha** prompt-enhancement node with prompt history, Prompt Sets, enhancement templates, IMAGE/VIDEO references, and workflow-driven enhancement.
- **Local LLM Server panel** — model loading, presets, memory/VRAM controls, status, performance information, and the optional OpenAI-compatible API.


## Requirements

- ComfyUI
- Python 3.10 or newer
- `llama-cpp-python` installed in the same Python environment used by ComfyUI
- A CUDA-enabled `llama-cpp-python` build when GPU inference is desired

This package intentionally does not install or replace `llama-cpp-python`, because installing the wrong wheel can replace a working CUDA build with a CPU-only build.

Verify the copy visible to ComfyUI with:

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

## Installation

Extract the package so the folder is:

```text
ComfyUI/custom_nodes/ComfyUI-Local-GGUF-LLM/
```

Restart ComfyUI, then hard-refresh the browser if an older frontend is still cached.

Do not install the standalone `ComfyUI-Local-LLM-Prompt-Enhancer` beside this package. Prompt Enhancer v0.6.12 is already bundled here.

## GGUF model folders

Place model GGUF files anywhere below:

```text
ComfyUI/models/llm/
```

Subfolders are supported, for example:

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

The Local LLM model selectors search this folder recursively.

## First-time server setup

Open **LLM** in the ComfyUI side menu to open the Local LLM Server panel.

For a normal first setup:

1. Select the GGUF under **Model**.
2. Select the matching **Vision / mmproj** only when the model supports vision. Otherwise use `None` or `Auto`.
3. Select a model preset or use **Auto (Detected)**.
4. Adjust memory/VRAM settings directly, or load a **Complete Settings Preset** from the **Presets** tab.
5. For a machine that also runs diffusion/video models, **Auto Yield to ComfyUI** is the recommended VRAM policy.
6. Save the server settings.
7. Start the model, or use **On Demand** so it loads on the first request.

The server is global and persistent. Workflow nodes send requests to that service instead of creating a new llama.cpp model for every node execution.

## Local LLM Generate

Add **Local LLM Generate** from the Local LLM node category.

The node provides:

- System Prompt preset and editable System Prompt
- Prompt preset and editable Prompt
- Temperature
- Top P
- Top K
- Min P
- Repeat Penalty
- Presence Penalty
- Frequency Penalty
- Max Tokens
- vision image/frame limits
- vision maximum edge size
- Seed with standard ComfyUI Control After Generate behavior
- optional `LOCAL_LLM_SETTINGS`
- optional `IMAGE`
- optional video frames as an `IMAGE` batch

Outputs:

- `response`
- `thinking`
- `info`
- `tokens`

When a **Local LLM Settings** node is connected, its complete model/runtime settings, sampler values, vision limits, and seed become authoritative for that workflow request. Prompt text and media still come from Local LLM Generate.

### Number controls

The custom DOM numeric controls mirror Nodes 2.0 behavior:

- large touch/clickable `−` and `+` controls
- drag or touch-scrub left/right to change the value
- min/max range fill when a real range exists
- exact step snapping
- direct click-to-edit
- keyboard stepping

The underlying native ComfyUI widgets remain authoritative for serialization and execution.

## Complete Settings Presets and Local LLM Settings

The server panel has a dedicated **Presets** tab for Complete Settings Presets. A complete preset stores the LLM runtime configuration: model and vision projector, model behavior/sampling, and memory/KV/offload/speculative settings. API keys, startup mode, logging, and interface preferences are intentionally excluded. The Presets tab is the only place that creates or deletes Complete Settings Presets and shows a readable summary of the selected preset.

Use **Local LLM Settings** when a workflow should own a reusable LLM configuration. The node can **load** Complete Settings Presets from the same preset library, but it does not save or delete them. Model, vision-projector, and model-preset selection come from the Complete Settings Preset rather than separate selectors on the workflow node. Thinking/reasoning and sampler controls remain directly adjustable; editing any visible preset-owned field changes the node to **Custom**. Detailed memory/performance tuning is managed in the Local LLM service panel and carried into the node when a Complete Settings Preset is loaded.

It outputs:

```text
LOCAL_LLM_SETTINGS
```

Connect that output to the `settings` input on **Local LLM Generate** or **Local LLM Prompt Enhancer**. When connected, the Settings node is authoritative for model/runtime settings, sampler values, vision limits, and seed.

## Local LLM Prompt Enhancer

The bundled **Local LLM Prompt Enhancer v0.6.12-alpha** uses the same persistent Local GGUF service. No second LLM server or second model load is required.

### Main workflow

1. Add **Local LLM Prompt Enhancer**.
2. Enter the original text in **Prompt**.
3. Choose an **Enhancement Preset** or edit **Enhancement Instructions**.
4. Set the number beside **Enhance Prompt** to `1` for a single result or higher for a batch, then click **Enhance Prompt**. Batch counts above `1` automatically switch **Overwrite Enhanced** to **Add New** and lock that choice until the count returns to `1`.
5. Edit the resulting **Enhanced Prompt** if needed.
6. Connect `enhanced_prompt` downstream to the node that should receive the enhanced text.

The original `prompt` output is also available unchanged.

### Optional media

Prompt Enhancer accepts:

- `images` — still image or IMAGE batch
- `video` — native ComfyUI VIDEO input
- `settings` — Local LLM Settings

Manual Enhance can partially execute the dependencies needed to make connected media available to the LLM. The selected local model must support the media type and have the appropriate vision/mmproj configuration.

### Enhanced Prompt history

Generated enhanced prompts are stored as an editable array.

The history control uses:

```text
− | X / Y | + | × | Undo | Redo | Clear All
```

- `X` is the editable/scrubbable active prompt index.
- `Y` is the read-only number of stored prompts.
- `×` deletes the active entry.
- Undo/Redo keep up to 20 array states in each direction.
- **Clear All** empties the array.

### Prompt Sets

Prompt Sets save and restore the complete enhanced-prompt array and active index.

They are stored under:

```text
ComfyUI/models/LLM/local_LLM_presets/prompt_enhancer/prompt_sets/
```

### Enhancement templates

Built-in templates are included for:

- Krea 2 Image
- MiniMax H3 T2VA
- MiniMax H3 I2VA
- MiniMax H3 FL2VA
- MiniMax H3 L2VA
- MiniMax H3 Ref2VA

User enhancement templates are stored under:

```text
ComfyUI/models/LLM/local_LLM_presets/prompt_enhancer/
```

Built-in templates are protected from deletion through the node UI.

### Prompt Cycle

When **Enhance with Workflow** is disabled, Prompt Cycle controls how the stored enhanced-prompt array advances after normal workflow execution:

- `fixed`
- `increment`
- `decrement`
- `shuffle`
- `random`

Shuffle and Random use fresh internal randomness and do not use the LLM generation seed.

### Enhance with Workflow

Enable **Enhance with Workflow** when every normal workflow execution should generate a fresh enhancement before sending text downstream.

When disabled, normal workflow execution uses the currently selected stored Enhanced Prompt instead of calling the LLM again.

## Preset folders

Local LLM presets are stored below:

```text
ComfyUI/models/LLM/local_LLM_presets/
├── settings/          # Complete Settings Presets
├── prompts/
├── system_prompts/
├── sampler/           # legacy/user sampler files remain readable
└── prompt_enhancer/
    └── prompt_sets/
```

**Local LLM Settings** loads Complete Settings Presets from `settings/`. Complete preset creation and deletion is intentionally centralized in the server panel's **Presets** tab.

## Vision input

For **Local LLM Generate**:

- `image` accepts a still IMAGE or IMAGE batch.
- `video_frames` accepts ordered frames as an IMAGE batch and samples them according to the configured frame limit.

For **Local LLM Prompt Enhancer**:

- `images` accepts still IMAGE references.
- `video` accepts native ComfyUI VIDEO.

Vision input requires a compatible multimodal model and mmproj/projector configuration. A text-only GGUF cannot use image/video input simply because the node socket is connected.

## Thinking / reasoning

The service exposes the final response and reasoning separately to ComfyUI nodes when the model/template provides reasoning.

The OpenAI-compatible chat endpoint also supports structured `reasoning_content`. For models such as Qwen where the chat template prefills the opening `<think>` and generation begins with reasoning text followed by `</think>`, the server recognizes the prefilled-thinking form and separates reasoning from final content instead of leaking the closing tag into the response.

## OpenAI-compatible API

The optional external API is configured from the Local LLM Server panel.

Endpoints:

```text
GET  /local-llm/v1/models
POST /local-llm/v1/chat/completions
POST /local-llm/v1/completions
```

`/local-llm/v1/chat/completions` supports streaming responses. Configure the API key and external-access options in the LLM panel before using an external client such as SillyTavern.

External GPU requests coordinate with ComfyUI GPU execution so the persistent LLM does not intentionally race a diffusion/video workload during VRAM handoff.

## Model residency and ComfyUI VRAM

The service is designed to coexist with normal ComfyUI model workloads.

- **Auto Yield to ComfyUI** allows the LLM to release its native context when ComfyUI needs the GPU, then reconstruct it on the next LLM request.
- **Keep Resident** prioritizes avoiding LLM reloads and is appropriate when enough VRAM remains for the rest of the workflow.
- **On Demand** loads the model on first use.
- **Auto Start** loads shortly after ComfyUI starts.

Changing model-allocation settings requires a native model reload. Request-local prompts and sampler values do not.

## Updating

When updating this package:

1. Replace/extract over `ComfyUI/custom_nodes/ComfyUI-Local-GGUF-LLM/`.
2. Restart ComfyUI.
3. Hard-refresh the browser if necessary.

The package removes stale versioned Local LLM and Prompt Enhancer frontend modules at import time so an old JS file left by an overwrite-style update is not loaded alongside the current one.
