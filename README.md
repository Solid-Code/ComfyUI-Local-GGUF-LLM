# ComfyUI Local GGUF LLM

A single ComfyUI custom-node package for running a persistent local GGUF LLM and using it directly from workflows.

This package includes:

- **Local LLM Generate** — send prompts, images, and sampled video frames to the persistent local LLM.
- **Local LLM Settings** — reusable model/generation settings with loadable Complete Settings Presets. Memory/performance tuning remains in the Local LLM service panel rather than crowding the workflow node.
- **Local LLM Prompt Enhancer** — bundled **v0.6.32-alpha** prompt-enhancement node with prompt history, Prompt Sets, enhancement templates, IMAGE/VIDEO references, and workflow-driven enhancement.
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

Do not install the standalone `ComfyUI-Local-LLM-Prompt-Enhancer` beside this package. Prompt Enhancer v0.6.35-alpha is already bundled here.

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
- optional `LOCAL_LLM_SETTINGS`
- optional `IMAGE`
- optional video frames as an `IMAGE` batch

Outputs:

- `response`
- `thinking`
- `info`

Local LLM Generate no longer duplicates model, sampler, or vision-limit controls. Connect **Local LLM Settings** when the workflow should own those values. **Seed + Control After Generate remain on Generate** as per-request controls. Local LLM Settings no longer contains or overrides seed. If `settings` is left disconnected, Generate uses the current Local LLM server/modal configuration. Prompt text, media, and seed remain owned by Local LLM Generate.

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




The node is a planning/orchestration layer. Enter the complete video idea in plain English and click **Generate Shots**. The Local LLM returns a validated, versioned plan and the node renders the result as horizontally scrollable shot cards.

### Sequence settings

- **Max Shot** — hard maximum for one H3 generation segment, up to 15 seconds.
- **Target Length** — `0` means Auto. Set a value when the complete sequence needs a requested total duration.
- **Width / Height** — explicit output geometry.
- **Start Frame Ratio / MP** — preserves the designated starting-image aspect ratio (or the first connected image as a fallback) and derives width/height from the target megapixel count.
- **Seed** — request-local seed used only when planning/regenerating shots.

### Dynamic H3 references

The frontend starts with one socket each for IMAGE, VIDEO, and AUDIO. As the final slot of a media type is connected or configured, the next slot appears automatically, up to the supported H3 planner limits:

- 9 images
- 3 videos
- 3 audio clips

Every visible reference gets a small **Role** selector and **Label** field. Labels tell the Local LLM what a connected asset represents; roles provide stronger routing hints such as first frame, last frame, subject identity, motion/camera reference, voice, music, paired video soundtrack, or source timeline audio.

`first_frame`, `last_frame`, and `source_timeline` are sequence-global roles and may each be assigned to only one connected asset. **Source audio timeline** is carried in the sequence separately so a downstream continuation chain can receive it once and preserve/slice it across shots; it is not automatically treated as a per-shot `<Audio N>` reference.

Audio references can also be marked as the soundtrack paired to Video 1/2/3 so Node Expansion can route them to H3's paired video-audio input rather than treating them as standalone audio.

The planner uses stable workflow asset IDs such as `image_1` and creates **shot-local** H3 bindings such as `<Picture 1>`, `<Video 1>`, and `<Audio 1>`. This matters because each shot may use a different subset of connected assets while still keeping valid contiguous H3 reference numbers.

### Shot cards

Each generated card supports:

- drag to reorder
- direct script editing
- direct duration editing
- per-shot guidance text
- **per-shot reference editing**: swap a bound asset in place, or add/remove Ref2VA bindings without an LLM call
- **reference pins**: pin any connected asset so the Local LLM must keep it bound when that shot is regenerated or when the full plan is regenerated; endpoint-conditioned modes can accept a new reference as a regen pin without corrupting their fixed input shape
- **shot lock**: locked shots are immutable anchors during Generate Shots replanning and their card controls/regeneration remain disabled until unlocked
- **STALE indicator**: a generated card is marked stale when the master description, sequence settings, connected-reference topology, or asset label/role changes after that shot was generated
- ↻ regenerate only that shot while sending the original request and complete current sequence as continuity context
- duplicate
- delete

Regeneration preserves the card's stable workflow ID. Pinned assets are enforced again by backend validation, and locked shots are restored verbatim even if the Local LLM attempts to rewrite them. The Local LLM returns a tolerant tagged transport format rather than embedding long H3 scripts inside JSON strings. The backend parses `<shot>`, `<seconds>`, `<model_mode>`, `<binding>`, and `<h3_script>` blocks into the same normalized internal plan, then validates durations, media bindings, model mode, reference limits, label numbering, and local timing. `plan_json` and `H3_SEQUENCE` remain structured JSON/Python workflow state; only the LLM response boundary uses tags. A compact tag-format repair request is attempted if wrapper structure is malformed.

### Output contract

The node outputs:

```text
H3_SEQUENCE  (displayed as "H3 Sequence")
```

`H3_SEQUENCE` is deliberately a **planned sequence**, not the live continuation `H3_CHAIN`. It contains the original request, validated shot plan, geometry, connected media objects plus metadata, reference limits, and optional source-audio asset ID. A downstream H3 Node Expansion/generation node should consume this sequence, build the first H3 conditioning/generation, and then create/advance the runtime `H3_CHAIN` as actual latent/audio history exists.

Each shot also stores the requested `seconds`, canonical H3 `frame_count`, and resulting `actual_seconds`. H3 uses 24 fps and the `17n+5` temporal frame grid; Node Expansion should use `frame_count` as authoritative.

This keeps planning state separate from generated AV state and avoids pretending a pre-generation object already contains continuation history.

### Local LLM media visibility

Still images and sampled video frames are sent to a compatible multimodal Local LLM. If a connected **Local LLM Settings** node sets vision limits below the number of connected references, the Shot Generator errors rather than silently hiding references from the planner.

Audio objects are preserved in `H3_SEQUENCE`, but the current GGUF vision path does not audition the waveform. The planner receives the audio's label, role, and duration metadata and is explicitly told not to invent unheard content.

## Local LLM Prompt Enhancer

The bundled **Local LLM Prompt Enhancer v0.6.15-alpha** uses the same persistent Local GGUF service. No second LLM server or second model load is required.

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

- `image(s)` — still image or IMAGE batch
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

Prompt Cycle execution state is mirrored from ComfyUI's global `executed` event, so cycling does not depend on the Prompt Enhancer node being selected or mounted by the renderer.

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

- `image(s)` accepts still IMAGE references.
- `video` accepts native ComfyUI VIDEO.


- dynamic `image_N` sockets provide still H3 references.
- dynamic `video_N` sockets provide native VIDEO references sampled for Local LLM planning.
- dynamic `audio_N` sockets are preserved for downstream H3 use; only label/role/duration metadata is available to the current Local LLM planner.

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

`GET /local-llm/v1/models` also advertises the **configured usable context window** through `context_length`, `max_context_length`, and `n_ctx`. When the GGUF exposes its native/training context metadata, `n_ctx_train` is included separately. These are compatibility extensions: strict OpenAI clients can ignore them, while local clients that understand them can auto-size their context window.

External GPU requests coordinate with ComfyUI GPU execution so the persistent LLM does not intentionally race a diffusion/video workload during VRAM handoff.

## Model residency and ComfyUI VRAM

The service is designed to coexist with normal ComfyUI model workloads.

- **Auto Yield to ComfyUI** allows the LLM to release its native context when ComfyUI needs the GPU, then reconstruct it on the next LLM request.
- **Keep Resident** prioritizes avoiding LLM reloads and is appropriate when enough VRAM remains for the rest of the workflow.
- **On Demand** loads the model on first use.
- **Auto Start** loads shortly after ComfyUI starts.

Changing model-allocation settings requires a native model reload. Request-local prompts and sampler values do not.

### Stop / Unload

**Stop / Unload** is a global Local LLM interrupt and remains available while the service is loading, waiting for ComfyUI, evaluating a prompt, generating, running an Enhance batch, or running the performance tuner. A Stop request immediately cancels the active/queued Local LLM request generation epoch; native llama.cpp teardown is deferred until the active native call reaches a safe return boundary. This avoids destroying a live CUDA context from another thread. On GPU, a large native model load or prompt-prefill cannot be forcibly torn down mid-kernel, so Stop becomes effective at the next safe llama.cpp/Python boundary and then unloads the model. It does not invoke ComfyUI's global workflow interrupt.