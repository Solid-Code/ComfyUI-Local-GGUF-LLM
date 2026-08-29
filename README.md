# Local GGUF LLM — v0.18.52 alpha


## v0.18.52 alpha

- Prompt Cycle now carries a serialized revision so an explicit X/Y selection wins immediately even if Run is queued before the asynchronous backend reset arrives; stale/background UI still follows the backend-owned cursor.
- Delayed cycle-reset requests can no longer erase a newer cursor created by an already-started workflow run.
- A manual Enhance batch now reports its final generated prompt on the `enhanced_prompt` output when its enhancer-only queue item completes.

- Manual Prompt Enhancer queue jobs are now physically pruned to the enhancer node and its upstream dependencies. This prevents unrelated workflow branches from running even on ComfyUI frontend builds that drop partial-execution targeting metadata.

- **Normal ComfyUI queue behavior restored:** Prompt Enhancer no longer wraps/intercepts `app.queuePrompt()`. While the one-item Enhance batch is running, clicking Run immediately adds normal workflow jobs to ComfyUI's queue behind it.
- **The Enhance batch owns its full queue lifetime:** all internal LLM iterations plus the final native llama.cpp VRAM suspend happen before the single Prompt Enhancer queue item returns, so the next queued diffusion job cannot begin early.
- **Queued-snapshot reconciliation:** workflows queued during the batch may have serialized the Prompt Enhancer before all progress updates reached the browser. The backend now recognizes pre-batch/intermediate history snapshots and upgrades them to the completed batch history when those queued jobs execute, without delaying queue submission.
- Removed the v0.18.49 informational “Workflow queued until…” Run gate/notification.
- Prompt Enhancer frontend: **v0.6.26-alpha**.


## v0.18.49 alpha

- **One queue item per Enhance batch:** batch size 1–64 now executes as a single targeted ComfyUI partial-execution job. Individual LLM iterations run internally inside that job instead of JavaScript appending another queue item after each completion.
- **Safe queued Run ordering:** if normal ComfyUI Run is clicked while the Enhance batch is active, that Run request waits until the single batch job has finished, its generated prompts have updated the node, and llama.cpp has yielded VRAM. The workflow is then submitted as the next queue item using the newly updated prompt state.
- **In-job LLM yield:** Prompt Enhancer now suspends/yields the native GGUF allocation before the one batch queue item returns, eliminating the gap where the next diffusion item could start during llama.cpp teardown.
- **Live batch UI remains:** each internally completed enhancement emits a best-effort progress event so generated prompts can populate the history while the one queue item is still running. The final executed payload contains the full batch and repairs any progress event missed by a background/throttled browser tab.
- Batch seed generation still follows the node's standard **Control After Generate** lifecycle; all seeds are captured up front and passed to the one backend batch job.
- The old direct HTTP generation path remains retired for GPU safety.
- Prompt Enhancer frontend: **v0.6.23-alpha**.

## v0.18.48 alpha

- Manual Prompt Enhancer requests moved to ComfyUI targeted partial execution so text-only enhancement can no longer run llama.cpp outside ComfyUI's execution queue.
- Added the browser-side Enhance/Run gate and post-batch Auto-Yield safety path that v0.18.49 consolidates into one batch queue item.

## v0.18.46 alpha
- Complete Settings **Load Preset** is disabled and rendered grey when the selected preset already exactly matches the current live settings. Selecting a different preset enables Load Preset again.

This package contains the reusable Local LLM service, settings/generation nodes, OpenAI-compatible API, and Prompt Enhancer. The H3 Shot Generator is distributed separately and uses this package through a small in-process bridge.

# ComfyUI Local GGUF LLM

A single ComfyUI custom-node package for running a persistent local GGUF LLM and using it directly from workflows.

This package includes:

- **Local LLM Generate** — send prompts, images, and sampled video frames to the persistent local LLM.
- **Local LLM Settings** — reusable model/generation settings with loadable Complete Settings Presets. Memory/performance tuning remains in the Local LLM service panel rather than crowding the workflow node.
- **Local LLM Prompt Enhancer** — bundled **v0.6.26-alpha** prompt-enhancement node with prompt history, Prompt Sets, enhancement templates, IMAGE/VIDEO references, and workflow-driven enhancement.
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

Do not install the standalone `ComfyUI-Local-LLM-Prompt-Enhancer` beside this package. Prompt Enhancer v0.6.26-alpha is already bundled here.

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

## Updating

When updating this package:

1. Replace/extract over `ComfyUI/custom_nodes/ComfyUI-Local-GGUF-LLM/`.
2. Restart ComfyUI.
3. Hard-refresh the browser if necessary.




### v0.18.40 alpha

- H3 sequential planning now performs one visual-reference analysis pass before generating unlocked shots. Connected image/video references are inspected once and condensed into a reusable `reference_context`; normal later shot calls are text-only and reuse that context instead of reprocessing every image/video.
- The reference context is stored in the internal plan/H3 Sequence so the accepted sequence retains the visual facts that informed planning. Audio is still not auditioned by the Local LLM; audio labels/roles remain metadata guidance.
- Added targeted fresh-vision fallback: a next-shot response may return `<vision_request>image_2,video_1</vision_request>` when a genuinely necessary visual fact is absent/ambiguous. The backend then repeats only that shot request with only those requested visual assets attached. Audio and unknown assets are rejected, and a shot may request fresh vision only once.
- Added detailed H3 planner performance logging. Console output now reports the one-time reference-analysis wall/prompt/decode/load time, each shot's primary LLM time, targeted fresh-vision time, tag-format repair time/count, H3 semantic-repair time, and total shot time. Formatting and semantic validation failures are explicitly logged when they trigger a repair.
- Existing per-card Regenerate remains a direct visual one-shot request; `plan_json` and downstream `H3_SEQUENCE` remain structured JSON/Python data.


### v0.18.39 alpha

- Progressive updates contain the already parsed, H3-validated, duration-snapped plan, so the card shown in the UI is the same normalized shot used as context for the next Local LLM request.
- The status line advances from `Shot N ready` to `Generating Shot N+1` and the shot strip follows the newest arriving card.
- The progressive partial plan is written to `plan_json`, so cancelling or failing after several successful shots preserves the shots that already completed.
- The frontend planning timeout is refreshed after every accepted shot, making the timeout per-shot/progress rather than a five-minute cap on an entire multi-shot sequence.
- Final node execution still returns the complete structured `H3_SEQUENCE`; the websocket progress path is UI-only and cannot make backend generation fail if no browser is connected.

### v0.18.37 alpha

- Every next-shot request receives the complete original video request, all connected references, and the full set of already accepted/normalized shots as continuity context.
- Added `<sequence_complete>true|false</sequence_complete>` to the LLM transport so the model decides after each accepted shot whether another outer shot is needed. A completion-only response with no `<shot>` is supported when the accepted sequence is already finished.
- Each generated shot is parsed, H3-validated, duration-snapped, and endpoint-normalized before it becomes context for the following request. A malformed response therefore affects only one shot request rather than the complete sequence.
- Locked shots are inserted directly as authoritative history without asking the LLM to reproduce them; pinned current/future positions remain hard sequential-planning constraints.
- Per-card regeneration remains a single Local LLM request. Internal `plan_json` and `H3_SEQUENCE` remain unchanged structured data.

### v0.18.36 alpha

- `generate_all` now returns `<h3_sequence>` with `<shot>` blocks; single-shot regeneration returns `<h3_regeneration>` with one `<shot>`.
- Each shot uses simple `<title>`, `<seconds>`, `<model_mode>`, `<input_bindings>`, and `<h3_script>` wrappers; bindings use child tags and plain labels such as `Picture 1`, which the backend normalizes to `<Picture 1>`.
- H3 tags such as `<Subject 1>`, `<Picture 1>`, `<Audio 1>`, and `<d>...</d>` remain verbatim inside `<h3_script>` and are not treated as transport markup.
- Added tolerant tagged-response parsing plus one deterministic tag-format repair pass. H3 semantic validation/repair remains separate from transport repair.
- Internal `plan_json`, shot-card state, and `H3_SEQUENCE` remain structured JSON/Python data, so the UI and downstream sequence contract are unchanged.

### v0.18.35 alpha

- Added balanced-object extraction plus deterministic cleanup for common preambles/fences, trailing commas, and literal control characters inside JSON strings.
- Split JSON syntax recovery from H3 semantic validation: syntax failures now use a compact deterministic repair request at temperature 0 without re-sending vision media.
- Syntax repair receives the exact parser line, column, character offset, and nearby text and may retry twice before returning a clean node error.
- H3 semantic repair now starts from parseable JSON; if the semantic repair introduces malformed JSON, syntax recovery is applied again before final validation.
- Planner instructions now explicitly require proper escaping of long `script` values, commas, line breaks, quotes, and backslashes.

### v0.18.34 alpha

- Intermediate local time marks are now driven by the Local LLM's understanding of meaningful action, camera, dialogue, audio, reference, and state-change boundaries rather than a fixed cadence.
- Planner instructions explicitly treat timestamps as a semantic temporal storyboard and discourage start/end-only scripts when a shot contains multiple phases.
- Backend keeps only structural timeline checks: local `00:00.000` start, chronological unique timing cues, in-range intermediate marks, and exact snapped H3 terminal timing/alignment.

### v0.18.33 alpha

- Planner instructions target one meaningful local `At MM:SS.mmm,` action beat about every 2.0–2.5 seconds, with intermediate beats required for shots longer than 2.5 seconds.
- Backend validation computes a minimum marker count from each shot's snapped H3 `actual_seconds`, so sparse timelines trigger the existing LLM repair pass.
- Backend also rejects clustered timing when any untimed interval exceeds approximately 3.25 seconds, preventing a model from satisfying the count with meaningless early/late marker clusters.
- Exact terminal timestamp and FL2VA/L2VA endpoint snapping remain unchanged.

### v0.18.32 alpha

- The planner then chooses the shortest realistic requested `seconds` value within `MAX_SHOT_SECONDS` and writes the final localized H3 timeline against that duration.
- JSON still emits `seconds` before `script` as a schema requirement; this no longer dictates the model's internal planning order.
- Single-shot regeneration now preserves the existing duration by default but may change it when guidance or materially revised action genuinely requires a different runtime.
- Existing backend H3 frame-grid snapping and exact terminal timestamp/alignment normalization remain the final timing authority.

### v0.18.31 alpha

- Every outer shot now treats its own `seconds` field as its authoritative requested local runtime and must write explicit `At MM:SS.mmm,` action beats beginning at local `00:00.000`.
- Regeneration explicitly supplies the current requested shot duration and current snapped H3 duration to the Local LLM.
- Backend validation requires a localized start and terminal time mark, rejects cumulative/out-of-range timelines, and rewrites the terminal action beat to the exact snapped H3 `actual_seconds`.
- FL2VA/L2VA endpoint alignment is normalized after frame snapping to the official two-decimal `S.SS-second mark`.

### v0.18.30 alpha
- H3 References now shows only media sockets that are actually connected. Autogrow placeholder sockets remain available on the node edge but no longer appear as `(+)` rows in the internal reference panel.

### v0.18.28 alpha
- Added per-shot reference editing: swap bindings directly and add/remove Ref2VA references without regenerating the script.
- Added persistent per-shot reference pins enforced as hard constraints during single-shot regeneration and full Generate Shots replanning.
- Added persistent shot locking; locked shots cannot be edited/regenerated in the card UI and are preserved as immutable positional anchors during replanning.
- Added per-shot **STALE** tracking for changes to the master request, sequence settings, asset labels/roles, and connected media topology.

### v0.18.27 alpha
- Added explicit first-frame, last-frame, and source-audio-timeline roles for downstream H3 expansion/continuation routing.
- Local LLM multimodal helpers now accept lists of still-image batches and video-frame batches so the H3 planner can inspect multiple independent connected references.

### v0.18.26 alpha
- Removed the redundant Local LLM Settings-connected seed status text from the Prompt Enhancer UI. Seed behavior is unchanged.

### v0.18.25 alpha
- Removed Seed / Control After Generate from Local LLM Settings.
- Generate and Prompt Enhancer now own their seeds independently. Connecting Settings no longer disables or overrides Prompt Enhancer seed controls.
