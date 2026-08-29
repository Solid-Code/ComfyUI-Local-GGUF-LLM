import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXTENSION_NAME = "LocalLLM.PromptEnhancer";
const FRONTEND_VERSION = "0.6.21-alpha";
console.info(`[Local LLM Prompt Enhancer] frontend ${FRONTEND_VERSION}`);
const NODE_CLASS = "LocalLLMPromptEnhancer";
const DEFAULT_PRESET = "Default / Krea 2 - Image";
const PROMPT_SET_NONE = "Unsaved";
const ENHANCE_TIMEOUT_MS = 5 * 60 * 1000;
const ENHANCE_BATCH_MIN = 1;
const ENHANCE_BATCH_MAX = 64;

let graphConfigureDepth = 0;
const pendingNodes = new Map();

function widget(node, name) {
  return node?.widgets?.find((w) => w?.name === name) || null;
}

function redraw(node) {
  try { node?.setDirtyCanvas?.(true, true); } catch (_) {}
  try { app?.graph?.setDirtyCanvas?.(true, true); } catch (_) {}
}

function markWorkflowChanged(node) {
  try { node?.graph?.change?.(); } catch (_) {}
  try { app?.graph?.setDirtyCanvas?.(true, true); } catch (_) {}
}

function copySize(node) {
  if (!node?.size || node.size.length < 2) return null;
  return [Number(node.size[0]) || 0, Number(node.size[1]) || 0];
}

function restoreSize(node, size) {
  if (!node || !size) return;
  try { node.setSize?.([size[0], size[1]]); }
  catch (_) { node.size = [size[0], size[1]]; }
}

function restoreLoadedWidthAndAutosize(node, saved) {
  if (!node || !saved) return;
  // Width is a genuine user choice and should survive reload/copy. Height is
  // derived from the persisted textarea heights + current responsive wrapping,
  // so restoring an old node height would fight the DOM panel auto-sizer.
  const current = copySize(node) || [440, 120];
  const width = Math.max(320, Number(saved[0]) || current[0] || 440);
  restoreSize(node, [width, current[1]]);

  const settle = () => {
    pinDomWidgetFullWidth(node.__promptEnhancerDomPanelWidget);
    clampDomPanelToNode(node);
    scheduleDomPanelHeight(node);
  };
  requestAnimationFrame(() => {
    settle();
    requestAnimationFrame(settle);
  });
}

function setWidgetOption(w, key, value) {
  if (!w) return;
  w.options ||= {};
  w.options[key] = value;
}

function setWidgetValue(w, value, invokeCallback = true) {
  if (!w) return;
  w.value = value;
  if (invokeCallback) {
    try { w.callback?.(value); } catch (_) {}
  }
}

function notify(severity, summary, detail) {
  const service = app?.extensionManager?.toast;
  try {
    if (service?.add) {
      service.add({ severity, summary, detail, life: severity === "error" ? 7000 : 3500 });
      return;
    }
  } catch (_) {}
  if (severity === "error") console.error(`[${summary}] ${detail}`);
  else if (severity === "warn") console.warn(`[${summary}] ${detail}`);
  else console.log(`[${summary}] ${detail}`);
}

async function getJSON(path) {
  const response = await api.fetchApi(path, { method: "GET" });
  let data = null;
  try { data = await response.json(); } catch (_) {}
  if (!response.ok) {
    const message = data?.error?.message || data?.message || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return data || {};
}

async function postJSON(path, body, options = {}) {
  const response = await api.fetchApi(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    ...options,
  });
  let data = null;
  try { data = await response.json(); } catch (_) {}
  if (!response.ok) {
    const message = data?.error?.message || data?.message || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return data || {};
}

function publishEnhancementBatchState(batchId, active, total = 0) {
  const id = String(batchId || "");
  if (!id) return;
  const registry = window.__localLLMEnhancementBatches ||= new Map();
  if (active) registry.set(id, { total: Number(total) || 0 });
  else registry.delete(id);
  try {
    window.dispatchEvent(new CustomEvent("local-llm-enhance-batch-state", {
      detail: { batchId: id, active: !!active, total: Number(total) || 0 },
    }));
  } catch (_) {}
}

async function beginEnhancementBatch(total) {
  const data = await postJSON("/local_llm_prompt_enhancer/batch_begin", {});
  const batchId = String(data?.batch_id || "");
  if (!batchId) throw new Error("Could not snapshot Local LLM settings for the enhancement batch.");
  publishEnhancementBatchState(batchId, true, total);
  return batchId;
}

async function endEnhancementBatch(batchId) {
  const id = String(batchId || "");
  if (!id) return;
  publishEnhancementBatchState(id, false, 0);
  try { await postJSON("/local_llm_prompt_enhancer/batch_end", { batch_id: id }); } catch (_) {}
}

function resetBackendPromptCycle(node) {
  const nodeId = String(node?.id ?? "");
  if (!nodeId) return;
  // Explicit user navigation/mode changes must win over any in-flight state
  // read started before this reset. Suppress readback until the backend has
  // acknowledged the reset so a stale response cannot jump the visible index.
  const epoch = (Number(node.__promptEnhancerCycleSyncEpoch) || 0) + 1;
  node.__promptEnhancerCycleSyncEpoch = epoch;
  node.__promptEnhancerCycleResetPending = true;
  void postJSON("/local_llm_prompt_enhancer/cycle_reset", { node_id: nodeId })
    .catch(() => {})
    .finally(() => {
      if (Number(node.__promptEnhancerCycleSyncEpoch) === epoch) {
        node.__promptEnhancerCycleResetPending = false;
      }
    });
}

async function syncPromptCycleFromBackend(node, { force = false } = {}) {
  if (!node || node.comfyClass !== NODE_CLASS || node.__promptEnhancerCycleResetPending) return false;
  if (!!widget(node, "enhance_with_workflow")?.value) return false;

  const mode = String(widget(node, "prompt_cycle")?.value ?? "fixed").trim().toLowerCase();
  if (mode === "fixed") return false;
  const history = promptHistory(node);
  if (!history.length) return false;

  const now = performance.now();
  const last = Number(node.__promptEnhancerCycleSyncAt) || 0;
  if (!force && now - last < 250) return false;
  if (node.__promptEnhancerCycleSyncPromise) return node.__promptEnhancerCycleSyncPromise;
  node.__promptEnhancerCycleSyncAt = now;

  const epoch = Number(node.__promptEnhancerCycleSyncEpoch) || 0;
  const nodeId = String(node.id ?? "");
  const request = postJSON("/local_llm_prompt_enhancer/cycle_state", {
    node_id: nodeId,
    mode,
    history,
  }).then((data) => {
    if (!data?.valid || !node || node.__promptEnhancerCycleResetPending) return false;
    if ((Number(node.__promptEnhancerCycleSyncEpoch) || 0) !== epoch) return false;

    // Re-check the signature after the async hop. If the user edited the array
    // or changed cycle mode while this request was in flight, ignore it.
    const currentMode = String(widget(node, "prompt_cycle")?.value ?? "fixed").trim().toLowerCase();
    const currentHistory = promptHistory(node);
    if (currentMode !== mode || JSON.stringify(currentHistory) !== JSON.stringify(history)) return false;

    const nextIndex = Math.max(0, Math.min(Math.trunc(Number(data.next_index) || 0), currentHistory.length - 1));
    const shuffle = Array.isArray(data.shuffle) ? data.shuffle : [];
    if (promptHistoryIndex(node, currentHistory) === nextIndex &&
        JSON.stringify(promptShuffleState(node)) === JSON.stringify(shuffle)) {
      return true;
    }
    setPromptHistoryState(node, currentHistory, nextIndex, shuffle, { backendSync: false });
    return true;
  }).catch(() => false).finally(() => {
    if (node.__promptEnhancerCycleSyncPromise === request) node.__promptEnhancerCycleSyncPromise = null;
  });
  node.__promptEnhancerCycleSyncPromise = request;
  return request;
}

function currentGraphPromptEnhancerNodes() {
  const nodes = [];
  const seen = new Set();
  const visit = (graph) => {
    if (!graph || seen.has(graph)) return;
    seen.add(graph);
    for (const candidate of graph._nodes || []) {
      if (!candidate) continue;
      if (candidate.comfyClass === NODE_CLASS) nodes.push(candidate);
      visit(candidate.subgraph || candidate.graph?.subgraph);
    }
  };
  visit(app?.rootGraph);
  visit(app?.graph);
  return nodes;
}

function syncCurrentGraphPromptCycles({ force = false } = {}) {
  for (const node of currentGraphPromptEnhancerNodes()) {
    void syncPromptCycleFromBackend(node, { force });
  }
}

let graphActivationSyncGeneration = 0;

function reconcilePromptCyclesAfterGraphActivation() {
  // ComfyUI workflow-tab switches are asynchronous. Mouse/focus events happen
  // before app.graph has necessarily been replaced, so they are not a reliable
  // signal that the newly selected workflow is ready. afterConfigureGraph() is
  // the authoritative lifecycle point; these passes only allow our async node
  // panel initialization/rendering to settle after the graph itself is active.
  const generation = ++graphActivationSyncGeneration;
  const sync = () => {
    if (generation !== graphActivationSyncGeneration) return;
    syncCurrentGraphPromptCycles({ force: true });
  };
  queueMicrotask(sync);
  requestAnimationFrame(sync);
  setTimeout(sync, 80);
  setTimeout(sync, 250);
}

function setBusy(node, busy, active = null) {
  if (!node) return;
  node.__promptEnhancerBusy = !!busy;
  node.__promptEnhancerActiveAction = busy ? active : null;
  scheduleDomPanelSync(node);
  redraw(node);
}

function normalizeEnhanceBatchCount(value) {
  const number = Math.trunc(Number(value));
  if (!Number.isFinite(number)) return ENHANCE_BATCH_MIN;
  return Math.max(ENHANCE_BATCH_MIN, Math.min(ENHANCE_BATCH_MAX, number));
}

function enhanceBatchCount(node) {
  if (!node) return ENHANCE_BATCH_MIN;
  const current = Number(node.__promptEnhancerBatchCount);
  if (Number.isFinite(current)) return normalizeEnhanceBatchCount(current);
  const saved = node?.properties?.[PERSISTENCE_STATE_KEY]?.batchCount;
  const count = normalizeEnhanceBatchCount(saved ?? ENHANCE_BATCH_MIN);
  node.__promptEnhancerBatchCount = count;
  return count;
}

function enforceBatchAddNew(node, { mark = true } = {}) {
  if (!node || enhanceBatchCount(node) <= 1) return false;
  const overwrite = widget(node, "overwrite_enhanced");
  if (!overwrite || !overwrite.value) return false;
  setWidgetValue(overwrite, false, false);
  if (mark && !node.__promptEnhancerRestoringState) {
    persistEnhancementState(node);
    markWorkflowChanged(node);
  }
  scheduleDomPanelSync(node);
  redraw(node);
  return true;
}

function enforceBatchWorkflowDisabled(node, { mark = true } = {}) {
  if (!node || enhanceBatchCount(node) <= 1) return false;
  const workflow = widget(node, "enhance_with_workflow");
  if (!workflow || !workflow.value) return false;
  setWidgetValue(workflow, false, false);
  if (mark && !node.__promptEnhancerRestoringState) {
    persistEnhancementState(node);
    markWorkflowChanged(node);
  }
  scheduleDomPanelSync(node);
  redraw(node);
  return true;
}

function setEnhanceBatchCount(node, value) {
  if (!node || node.__promptEnhancerBusy) return;
  const next = normalizeEnhanceBatchCount(value);
  const changed = enhanceBatchCount(node) !== next;
  node.__promptEnhancerBatchCount = next;
  // A batch must append every result and workflow-time enhancement is a
  // single-result mode. Force both settings immediately when the count crosses
  // above one, then keep both controls locked until it returns to one.
  const forcedAddNew = enforceBatchAddNew(node, { mark: false });
  const forcedWorkflowDisabled = enforceBatchWorkflowDisabled(node, { mark: false });
  persistEnhancementState(node);
  if (changed || forcedAddNew || forcedWorkflowDisabled) markWorkflowChanged(node);
  scheduleDomPanelSync(node);
  redraw(node);
}

function parseJSONList(value) {
  try {
    const parsed = JSON.parse(String(value ?? "[]"));
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function promptHistory(node) {
  const raw = parseJSONList(widget(node, "prompt_history_json")?.value);
  return raw.map((item) => String(item ?? ""));
}

function promptHistoryIndex(node, list = null) {
  const history = list || promptHistory(node);
  if (!history.length) return 0;
  const raw = Number(widget(node, "prompt_history_index")?.value ?? 0);
  const index = Number.isFinite(raw) ? Math.trunc(raw) : 0;
  return Math.max(0, Math.min(index, history.length - 1));
}

function promptShuffleState(node) {
  return parseJSONList(widget(node, "prompt_shuffle_json")?.value)
    .map((v) => Number(v))
    .filter((v) => Number.isInteger(v) && v >= 0);
}

function arraySnapshot(node) {
  return {
    history: promptHistory(node),
    index: promptHistoryIndex(node),
    shuffle: promptShuffleState(node),
  };
}

const ARRAY_UNDO_LIMIT = 20;

function pushArrayStack(stack, state) {
  stack.push(state);
  if (stack.length > ARRAY_UNDO_LIMIT) stack.shift();
}

function pushArrayUndo(node) {
  node.__promptEnhancerArrayUndo ||= [];
  pushArrayStack(node.__promptEnhancerArrayUndo, arraySnapshot(node));
  // A new edit creates a new history branch, just like a normal editor.
  node.__promptEnhancerArrayRedo = [];
}

function setPromptHistoryState(node, history, index = 0, shuffle = [], { mark = true, backendSync = mark } = {}) {
  if (!node) return;
  const list = Array.isArray(history) ? history.map((item) => String(item ?? "")) : [];
  const clamped = list.length ? Math.max(0, Math.min(Math.trunc(Number(index) || 0), list.length - 1)) : 0;
  const validShuffle = Array.isArray(shuffle)
    ? shuffle.map((v) => Number(v)).filter((v) => Number.isInteger(v) && v >= 0 && v < list.length && v !== clamped)
    : [];

  node.__promptEnhancerSyncingHistory = true;
  try {
    setWidgetValue(widget(node, "prompt_history_json"), JSON.stringify(list), false);
    setWidgetValue(widget(node, "prompt_history_index"), clamped, false);
    setWidgetValue(widget(node, "prompt_shuffle_json"), JSON.stringify(validShuffle), false);
    setWidgetValue(widget(node, "enhanced_prompt"), list.length ? list[clamped] : "", false);
  } finally {
    node.__promptEnhancerSyncingHistory = false;
  }
  persistEnhancementState(node);
  if (mark) markWorkflowChanged(node);
  if (backendSync) resetBackendPromptCycle(node);
  scheduleDomPanelSync(node);
  redraw(node);
}

function syncVisibleEnhancedIntoHistory(node) {
  if (!node || node.__promptEnhancerSyncingHistory) return;
  const enhanced = String(widget(node, "enhanced_prompt")?.value ?? "");
  const history = promptHistory(node);
  let index = promptHistoryIndex(node, history);
  if (!history.length) {
    if (!enhanced.trim()) return;
    history.push(enhanced);
    index = 0;
  } else {
    history[index] = enhanced;
  }
  setPromptHistoryState(node, history, index, [], { mark: false });
}

function applyGeneratedPrompt(node, revised, overwriteOverride = null) {
  const text = String(revised ?? "");
  if (!text.trim()) return false;
  pushArrayUndo(node);
  const history = promptHistory(node);
  let index = promptHistoryIndex(node, history);
  const overwrite = overwriteOverride == null
    ? !!widget(node, "overwrite_enhanced")?.value
    : !!overwriteOverride;
  if (overwrite && history.length) {
    history[index] = text;
  } else {
    history.push(text);
    index = history.length - 1;
  }
  setPromptHistoryState(node, history, index, []);
  return true;
}

function cycleStoredPrompt(node, delta) {
  if (!node || node.__promptEnhancerBusy) return;
  syncVisibleEnhancedIntoHistory(node);
  const history = promptHistory(node);
  if (!history.length) {
    notify("warn", "Local LLM Prompt Enhancer", "No enhanced prompts are stored yet.");
    return;
  }
  const current = promptHistoryIndex(node, history);
  const next = (current + Number(delta) + history.length) % history.length;
  setPromptHistoryState(node, history, next, []);
}

function selectStoredPromptIndex(node, oneBasedIndex) {
  if (!node || node.__promptEnhancerBusy) return;
  syncVisibleEnhancedIntoHistory(node);
  const history = promptHistory(node);
  if (!history.length) {
    scheduleDomPanelSync(node);
    return;
  }
  const requested = Math.trunc(Number(oneBasedIndex));
  const clampedOneBased = Number.isFinite(requested)
    ? Math.max(1, Math.min(requested, history.length))
    : promptHistoryIndex(node, history) + 1;
  setPromptHistoryState(node, history, clampedOneBased - 1, []);
}

function deleteActivePrompt(node) {
  if (!node || node.__promptEnhancerBusy) return;
  syncVisibleEnhancedIntoHistory(node);
  const history = promptHistory(node);
  if (!history.length) {
    notify("warn", "Local LLM Prompt Enhancer", "No enhanced prompt to delete.");
    return;
  }
  pushArrayUndo(node);
  const index = promptHistoryIndex(node, history);
  history.splice(index, 1);
  const next = history.length ? Math.min(index, history.length - 1) : 0;
  setPromptHistoryState(node, history, next, []);
}

function clearPromptArray(node) {
  if (!node || node.__promptEnhancerBusy) return;
  syncVisibleEnhancedIntoHistory(node);
  const history = promptHistory(node);
  if (!history.length) {
    notify("warn", "Local LLM Prompt Enhancer", "The enhanced prompt array is already empty.");
    return;
  }
  pushArrayUndo(node);
  setPromptHistoryState(node, [], 0, []);

  const setWidget = widget(node, "prompt_set");
  if (setWidget && String(setWidget.value ?? "") !== PROMPT_SET_NONE) {
    node.__promptEnhancerSettingPromptSet = true;
    try { setWidgetValue(setWidget, PROMPT_SET_NONE, false); }
    finally { node.__promptEnhancerSettingPromptSet = false; }
  }
  persistEnhancementState(node);
  markWorkflowChanged(node);
  redraw(node);
}

async function savePromptSet(node) {
  if (!node || node.__promptEnhancerBusy) return;
  syncVisibleEnhancedIntoHistory(node);
  const history = promptHistory(node);
  if (!history.length) {
    notify("warn", "Local LLM Prompt Enhancer", "There are no enhanced prompts to save as a Prompt Set.");
    return;
  }

  const setWidget = widget(node, "prompt_set");
  const selected = String(setWidget?.value ?? PROMPT_SET_NONE);
  const suggested = selected !== PROMPT_SET_NONE ? selected : "";
  const entered = globalThis.prompt?.("Save Prompt Set as:", suggested);
  if (entered === null || entered === undefined) return;
  const name = String(entered).trim();
  if (!name) {
    notify("warn", "Local LLM Prompt Enhancer", "Prompt Set name cannot be empty.");
    return;
  }

  setBusy(node, true, "save_prompt_set");
  try {
    const data = await postJSON("/local_llm_prompt_enhancer/save_prompt_set", {
      name,
      prompts: history,
      active_index: promptHistoryIndex(node, history),
    });
    const savedName = String(data?.saved?.name || name);
    applyPromptSetRecords(node, data, { selectName: savedName });
    notify("success", "Local LLM Prompt Enhancer", `Prompt Set saved: ${savedName}`);
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  } finally {
    setBusy(node, false);
  }
}

async function deletePromptSet(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const setWidget = widget(node, "prompt_set");
  const selected = String(setWidget?.value ?? PROMPT_SET_NONE).trim();
  if (!selected || selected === PROMPT_SET_NONE) {
    notify("warn", "Local LLM Prompt Enhancer", "Select a saved Prompt Set to delete.");
    return;
  }

  const confirmed = globalThis.confirm?.(`Delete Prompt Set "${selected}"?\n\nThis deletes the saved set file. The current prompts in the node will remain.`);
  if (!confirmed) return;

  setBusy(node, true, "delete_prompt_set");
  try {
    const data = await postJSON("/local_llm_prompt_enhancer/delete_prompt_set", { name: selected });
    applyPromptSetRecords(node, data, { selectName: PROMPT_SET_NONE });
    markWorkflowChanged(node);
    notify("success", "Local LLM Prompt Enhancer", `Prompt Set deleted: ${selected}`);
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  } finally {
    setBusy(node, false);
  }
}

async function loadPromptSet(node, name) {
  if (!node || node.__promptEnhancerBusy) return;
  const cleanName = String(name ?? "").trim();
  if (!cleanName || cleanName === PROMPT_SET_NONE) return;
  try {
    syncVisibleEnhancedIntoHistory(node);
    const loaded = await postJSON("/local_llm_prompt_enhancer/load_prompt_set", { name: cleanName });
    const item = loaded?.set || {};
    pushArrayUndo(node);
    setPromptHistoryState(
      node,
      Array.isArray(item.prompts) ? item.prompts : [],
      Number(item.active_index) || 0,
      []
    );
    notify("success", "Local LLM Prompt Enhancer", `Prompt Set loaded: ${cleanName}`);
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  }
}

function applyPromptSetRecords(node, data, { selectName = null } = {}) {
  const records = Array.isArray(data?.sets) ? data.sets : [];
  const names = records.map((item) => String(item?.name ?? "")).filter(Boolean);
  node.__promptEnhancerPromptSets = new Map(records.map((item) => [String(item.name), item]));

  const setWidget = widget(node, "prompt_set");
  const labels = [PROMPT_SET_NONE, ...names];
  updateComboValues(setWidget, labels);

  let next = selectName != null ? String(selectName) : String(setWidget?.value ?? PROMPT_SET_NONE);
  if (!labels.includes(next)) next = PROMPT_SET_NONE;
  if (setWidget && String(setWidget.value ?? "") !== next) {
    node.__promptEnhancerSettingPromptSet = true;
    try { setWidgetValue(setWidget, next, false); }
    finally { node.__promptEnhancerSettingPromptSet = false; }
  }
  persistEnhancementState(node);
  scheduleDomPanelSync(node);
  redraw(node);
}

async function refreshPromptSets(node, { selectName = null } = {}) {
  if (!node) return;
  try {
    const data = await getJSON("/local_llm_prompt_enhancer/prompt_sets");
    applyPromptSetRecords(node, data, { selectName });
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  }
}

function undoPromptArray(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const undo = node.__promptEnhancerArrayUndo || [];
  if (!undo.length) {
    notify("warn", "Local LLM Prompt Enhancer", "Nothing to undo in the enhanced prompt array this session.");
    return;
  }
  node.__promptEnhancerArrayRedo ||= [];
  pushArrayStack(node.__promptEnhancerArrayRedo, arraySnapshot(node));
  const state = undo.pop();
  setPromptHistoryState(node, state.history || [], state.index || 0, state.shuffle || []);
}

function redoPromptArray(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const redo = node.__promptEnhancerArrayRedo || [];
  if (!redo.length) {
    notify("warn", "Local LLM Prompt Enhancer", "Nothing to redo in the enhanced prompt array this session.");
    return;
  }
  node.__promptEnhancerArrayUndo ||= [];
  pushArrayStack(node.__promptEnhancerArrayUndo, arraySnapshot(node));
  const state = redo.pop();
  setPromptHistoryState(node, state.history || [], state.index || 0, state.shuffle || []);
}

function initializePromptHistory(node) {
  let history = promptHistory(node);
  let index = promptHistoryIndex(node, history);
  const visible = String(widget(node, "enhanced_prompt")?.value ?? "");
  if (!history.length && visible.trim()) {
    history = [visible];
    index = 0;
  }
  setPromptHistoryState(node, history, index, promptShuffleState(node), { mark: false });
}

function templateMap(node) {
  return node?.__promptEnhancerTemplates || new Map();
}

function selectedTemplate(node) {
  const presetWidget = widget(node, "enhancement_preset");
  return templateMap(node).get(String(presetWidget?.value ?? "")) || null;
}

function loadSelectedTemplate(node) {
  const record = selectedTemplate(node);
  const textWidget = widget(node, "enhancement_text");
  if (!record || !textWidget) return false;
  setWidgetValue(textWidget, String(record.text ?? ""), true);
  persistEnhancementState(node);
  markWorkflowChanged(node);
  redraw(node);
  return true;
}

function updateComboValues(combo, labels) {
  if (!combo) return;
  combo.options ||= {};
  combo.options.values = labels;
}

function applyTemplateRecords(node, data, { loadSelected = false, selectLabel = null } = {}) {
  const records = Array.isArray(data?.templates) ? data.templates : [];
  const map = new Map(records.map((r) => [String(r.label), r]));
  const labels = records.map((r) => String(r.label));
  node.__promptEnhancerTemplates = map;

  const presetWidget = widget(node, "enhancement_preset");
  updateComboValues(presetWidget, labels.length ? labels : ["Custom"]);

  let next = selectLabel ? String(selectLabel) : String(presetWidget?.value ?? "");
  if (!map.has(next)) {
    const preferred = String(data?.default || DEFAULT_PRESET);
    next = map.has(preferred) ? preferred : (labels[0] || "Custom");
  }
  if (presetWidget && String(presetWidget.value ?? "") !== next) {
    setWidgetValue(presetWidget, next, false);
  }
  if (loadSelected) loadSelectedTemplate(node);
  scheduleDomPanelSync(node);
  redraw(node);
}

async function refreshTemplates(node, { loadSelected = false, selectLabel = null } = {}) {
  if (!node) return;
  try {
    const data = await getJSON("/local_llm_prompt_enhancer/templates");
    applyTemplateRecords(node, data, { loadSelected, selectLabel });
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  }
}

function applyPromptPresetCatalog(node, data, { loadSelected = false, selectName = null } = {}) {
  const section = data?.prompts || {};
  const names = Array.isArray(section.names) && section.names.length ? section.names.map(String) : ["Custom"];
  const presets = section.presets && typeof section.presets === "object" ? section.presets : {};
  const deletable = new Set(Array.isArray(section.deletable_names) ? section.deletable_names.map(String) : Object.keys(presets));
  node.__promptEnhancerPromptPresetCatalog = { names, presets, deletable };

  const selector = widget(node, "prompt_preset");
  updateComboValues(selector, names);
  let next = selectName == null ? String(selector?.value ?? "Custom") : String(selectName);
  if (!names.includes(next)) next = "Custom";
  if (selector && String(selector.value ?? "") !== next) setWidgetValue(selector, next, false);

  if (loadSelected && next !== "Custom" && Object.prototype.hasOwnProperty.call(presets, next)) {
    node.__promptEnhancerApplyingPromptPreset = true;
    try { setWidgetValue(widget(node, "prompt"), String(presets[next] ?? ""), false); }
    finally { node.__promptEnhancerApplyingPromptPreset = false; }
  }
  persistEnhancementState(node);
  scheduleDomPanelSync(node);
  redraw(node);
}

async function refreshPromptPresets(node, { loadSelected = false, selectName = null } = {}) {
  if (!node) return;
  try {
    const data = await getJSON("/local_llm_server/node_presets");
    applyPromptPresetCatalog(node, data, { loadSelected, selectName });
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  }
}

async function applyPromptPresetSelection(node, selection) {
  if (!node) return;
  const name = String(selection || "Custom");
  if (name === "Custom") {
    persistEnhancementState(node);
    scheduleDomPanelSync(node);
    return;
  }
  await refreshPromptPresets(node, { loadSelected: true, selectName: name });
}

async function savePromptPreset(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const selector = widget(node, "prompt_preset");
  if (!selector) return;
  if (String(selector.value || "Custom") !== "Custom") {
    notify("warn", "Local LLM Prompt Preset", "Edit Prompt first; edited text automatically switches this selector to Custom.");
    return;
  }
  const entered = window.prompt("Name this prompt preset:", "");
  if (entered === null) return;
  const name = String(entered).trim();
  if (!name) {
    notify("warn", "Local LLM Prompt Preset", "Prompt preset name cannot be empty.");
    return;
  }
  setBusy(node, true, "save_prompt_preset");
  try {
    const result = await postJSON("/local_llm_server/node_presets", {
      kind: "prompts", name, text: String(widget(node, "prompt")?.value ?? ""),
    });
    const data = result?.catalog || await getJSON("/local_llm_server/node_presets");
    const savedName = String(result?.saved?.name || name);
    applyPromptPresetCatalog(node, data, { loadSelected: true, selectName: savedName });
    markWorkflowChanged(node);
    notify("success", "Local LLM Prompt Preset", `Saved ${savedName}.`);
  } catch (error) {
    notify("error", "Local LLM Prompt Preset", error?.message || String(error));
  } finally {
    setBusy(node, false);
  }
}

async function deletePromptPreset(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const selector = widget(node, "prompt_preset");
  const name = String(selector?.value || "Custom").trim();
  if (!name || name === "Custom") {
    notify("warn", "Local LLM Prompt Preset", "Select a saved prompt preset to delete.");
    return;
  }
  let catalog = node.__promptEnhancerPromptPresetCatalog;
  if (!catalog) {
    try {
      const data = await getJSON("/local_llm_server/node_presets");
      applyPromptPresetCatalog(node, data);
      catalog = node.__promptEnhancerPromptPresetCatalog;
    } catch (error) {
      notify("error", "Local LLM Prompt Preset", error?.message || String(error));
      return;
    }
  }
  if (!catalog?.deletable?.has(name)) {
    notify("warn", "Local LLM Prompt Preset", `“${name}” is not a deletable user preset.`);
    return;
  }
  if (!globalThis.confirm?.(`Delete prompt preset “${name}”?\n\nThe current Prompt text will remain.`)) return;
  setBusy(node, true, "delete_prompt_preset");
  try {
    const response = await api.fetchApi("/local_llm_server/node_presets", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "prompts", name }),
    });
    let result = null;
    try { result = await response.json(); } catch (_) {}
    if (!response.ok) throw new Error(result?.error?.message || result?.message || `HTTP ${response.status}`);
    const data = result?.catalog || await getJSON("/local_llm_server/node_presets");
    applyPromptPresetCatalog(node, data, { loadSelected: false, selectName: "Custom" });
    markWorkflowChanged(node);
    notify("success", "Local LLM Prompt Preset", `Deleted ${name}.`);
  } catch (error) {
    notify("error", "Local LLM Prompt Preset", error?.message || String(error));
  } finally {
    setBusy(node, false);
  }
}

function clearEnhanceTransport(node) {
  if (!node) return;
  if (node.__promptEnhancerTimeout) clearTimeout(node.__promptEnhancerTimeout);
  node.__promptEnhancerTimeout = null;
  node.__promptEnhancerPendingToken = null;
  node.__promptEnhancerJobId = null;
  node.__promptEnhancerRequestController = null;
  pendingNodes.delete(String(node.id));
}

function settleEnhanceExecution(node, error = null, data = null) {
  const waiter = node?.__promptEnhancerExecutionWaiter;
  if (!waiter) return false;
  node.__promptEnhancerExecutionWaiter = null;
  if (error) waiter.reject(error);
  else waiter.resolve(data);
  return true;
}

function clearEnhancePending(node) {
  if (!node) return;
  clearEnhanceTransport(node);
  settleEnhanceExecution(node, new DOMException("Enhancement cancelled.", "AbortError"));
  node.__promptEnhancerCancelRequested = false;
  node.__promptEnhancerOverwriteAtStart = null;
  node.__promptEnhancerBatchProgress = null;
  if (node.__promptEnhancerManualSeedSource) node.__promptEnhancerManualSeedSource = null;
  setBusy(node, false);
}

function parseEnhanceExecution(output) {
  const values = output?.prompt_enhancer;
  const raw = Array.isArray(values) ? values[0] : values;
  if (!raw) return null;
  if (typeof raw === "object") return raw;
  try { return JSON.parse(String(raw)); }
  catch (_) { return null; }
}

function handleEnhanceExecuted(node, output) {
  const data = parseEnhanceExecution(output);
  if (!data) return false;

  const mode = String(data.mode || "manual");
  if (mode === "cycle") {
    syncVisibleEnhancedIntoHistory(node);
    const history = promptHistory(node);
    const nextIndex = history.length
      ? Math.max(0, Math.min(Number(data.next_index ?? 0), history.length - 1))
      : 0;
    const shuffle = Array.isArray(data.shuffle) ? data.shuffle : [];
    setPromptHistoryState(node, history, nextIndex, shuffle, { backendSync: false });
    node.__promptEnhancerCycleSyncAt = performance.now();
    return true;
  }

  const revised = String(data.prompt ?? "");
  if (mode === "workflow") {
    if (!revised.trim()) {
      notify("error", "Local LLM Prompt Enhancer", "The workflow enhancement returned an empty prompt.");
      return true;
    }
    applyGeneratedPrompt(node, revised, data.overwrite);
    return true;
  }

  const pendingToken = String(node?.__promptEnhancerPendingToken || "");
  if (!pendingToken || String(data.token || "") !== pendingToken) return false;

  if (!revised.trim()) {
    const error = new Error("The Local LLM returned an empty prompt.");
    clearEnhanceTransport(node);
    if (!settleEnhanceExecution(node, error)) {
      clearEnhancePending(node);
      notify("error", "Local LLM Prompt Enhancer", error.message);
    }
    return true;
  }

  // Batched media/settings enhancement waits for this partial-execution result
  // before queuing the next item. Hand the payload back to the batch runner and
  // keep the node busy instead of finalizing the whole action here.
  if (node.__promptEnhancerExecutionWaiter) {
    clearEnhanceTransport(node);
    settleEnhanceExecution(node, null, data);
    redraw(node);
    return true;
  }

  // Compatibility fallback for a single already-armed request created by an
  // older frontend instance during a hot reload.
  applyGeneratedPrompt(node, revised, node.__promptEnhancerOverwriteAtStart);
  clearEnhancePending(node);

  const media = [];
  if (data.used_images) media.push("image(s)");
  if (data.used_video) media.push("video");
  const suffix = media.length ? ` using ${media.join(" + ")}` : "";
  notify("success", "Local LLM Prompt Enhancer", `Enhancement complete${suffix}.`);
  redraw(node);
  return true;
}

async function cancelArmed(node, token) {
  try {
    await postJSON("/local_llm_prompt_enhancer/cancel", {
      node_id: String(node?.id ?? ""),
      token: String(token || ""),
    });
  } catch (_) {}
}

function applyPromptEnhancerInputLabels(node) {
  for (const input of node?.inputs || []) {
    if (input?.name === "images") input.label = "image(s)";
  }
}

function hasConnectedMedia(node) {
  return (node?.inputs || []).some((input) =>
    (input?.name === "images" || input?.name === "video") && input?.link != null
  );
}

function hasConnectedSettings(node) {
  return (node?.inputs || []).some((input) => input?.name === "settings" && input?.link != null);
}

function inputOriginNode(node, inputName) {
  const input = (node?.inputs || []).find((item) => item?.name === inputName);
  if (!input || input.link == null) return null;
  const graph = node?.graph || app?.graph;
  const link = graph?.links?.[input.link] || graph?.links?.get?.(input.link);
  const originId = link?.origin_id ?? link?.originId;
  if (originId == null) return null;
  return graph?.getNodeById?.(originId) || null;
}

function setSeedOverrideUI(node) {
  if (!node) return;
  // Settings never owns seed. Keep Prompt Enhancer's Seed + Control After
  // Generate active regardless of whether LOCAL_LLM_SETTINGS is connected.
  const connected = hasConnectedSettings(node);
  for (const w of [widget(node, "seed"), seedControlWidget(node)]) {
    if (!w) continue;
    try {
      const el = w.element;
      if (el && "disabled" in el) el.disabled = false;
      if (el) el.setAttribute?.("aria-disabled", "false");
    } catch (_) {}
    try {
      if (w.options && Object.isExtensible(w.options)) w.options.readOnly = false;
    } catch (_) {}
  }
  node.__promptEnhancerSettingsConnected = connected;
  scheduleDomPanelSync(node);
  redraw(node);
}

function wrapSettingsConnectionUI(node) {
  if (!node || node.__promptEnhancerSettingsConnectionWrapped) return;
  node.__promptEnhancerSettingsConnectionWrapped = true;
  const original = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = original?.apply(this, args);
    queueMicrotask(() => setSeedOverrideUI(this));
    return result;
  };
  setSeedOverrideUI(node);
  requestAnimationFrame(() => setSeedOverrideUI(node));
}

function seedControlWidget(node) {
  const seedWidget = widget(node, "seed");
  return (seedWidget?.linkedWidgets || []).find((w) =>
    typeof w?.beforeQueued === "function" && typeof w?.afterQueued === "function"
  ) || null;
}

function beginSeedControlCycle(node) {
  // Manual Enhance bypasses app.queuePrompt(), so drive this Prompt Enhancer's
  // own standard control_after_generate lifecycle. Settings never owns seed.
  const sourceNode = node;
  if (!sourceNode) return 0;
  const control = seedControlWidget(sourceNode);
  try { control?.beforeQueued?.({ isPartialExecution: false }); } catch (error) { console.warn(error); }

  const seedWidget = widget(sourceNode, "seed");
  const raw = Number(seedWidget?.value ?? 0);
  const seed = Number.isFinite(raw) ? Math.max(0, Math.trunc(raw)) : 0;
  if (seedWidget && seedWidget.value !== seed) setWidgetValue(seedWidget, seed, true);
  node.__promptEnhancerManualSeedSource = sourceNode;

  if (sourceNode === node) persistEnhancementState(node);
  markWorkflowChanged(sourceNode);
  redraw(sourceNode);
  redraw(node);
  return seed;
}

function finishSeedControlCycle(node) {
  const sourceNode = node?.__promptEnhancerManualSeedSource || node;
  const control = seedControlWidget(sourceNode);
  try { control?.afterQueued?.({ isPartialExecution: false }); } catch (error) { console.warn(error); }
  if (sourceNode === node) persistEnhancementState(node);
  markWorkflowChanged(sourceNode);
  redraw(sourceNode);
  redraw(node);
  if (node) node.__promptEnhancerManualSeedSource = null;
}

async function cancelEnhancement(node) {
  if (!node || node.__promptEnhancerActiveAction !== "enhance") return;
  node.__promptEnhancerCancelRequested = true;

  const token = String(node.__promptEnhancerPendingToken || "");
  const jobId = String(node.__promptEnhancerJobId || "");
  const controller = node.__promptEnhancerRequestController;

  try { controller?.abort?.(); } catch (_) {}

  // Remove an armed-but-not-yet-executing request from the enhancer backend.
  // An empty token intentionally means "cancel the current request for this node".
  await cancelArmed(node, token);

  if (jobId) {
    // Current ComfyUI exposes per-job cancellation. It safely removes a queued
    // partial run or interrupts it if it has already started. Fall back to the
    // older interrupt endpoint when running against an older frontend/runtime.
    try {
      if (typeof api?.cancelJob === "function") await api.cancelJob(jobId);
      else if (typeof api?.interrupt === "function") await api.interrupt(jobId);
    } catch (_) {
      try { await api?.interrupt?.(jobId); } catch (_) {}
    }
  }

  clearEnhanceTransport(node);
  settleEnhanceExecution(node, new DOMException("Enhancement cancelled.", "AbortError"));
  notify("info", "Local LLM Prompt Enhancer", "Enhancement cancelled.");
}

async function runTextOnlyEnhancementOnce(node, prompt, enhancementText, seed, batchId = "") {
  const controller = new AbortController();
  node.__promptEnhancerRequestController = controller;
  const data = await postJSON(
    "/local_llm_prompt_enhancer/run",
    { prompt, enhancement_text: enhancementText, seed, batch_id: batchId },
    { signal: controller.signal },
  );
  node.__promptEnhancerRequestController = null;
  if (node.__promptEnhancerCancelRequested) throw new DOMException("Enhancement cancelled.", "AbortError");
  const revised = String(data?.prompt ?? "");
  if (!revised.trim()) throw new Error("The Local LLM returned an empty prompt.");
  return { ...data, prompt: revised };
}

async function runMediaEnhancementOnce(node, prompt, enhancementText, seed, batchId = "") {
  let token = "";
  try {
    const controller = new AbortController();
    node.__promptEnhancerRequestController = controller;
    const armed = await postJSON(
      "/local_llm_prompt_enhancer/arm",
      {
        node_id: String(node.id),
        prompt,
        enhancement_text: enhancementText,
        seed,
        batch_id: batchId,
      },
      { signal: controller.signal },
    );
    token = String(armed?.token || "");
    if (!token) throw new Error("Prompt Enhancer could not arm the request.");
    if (node.__promptEnhancerCancelRequested) {
      await cancelArmed(node, token);
      throw new DOMException("Enhancement cancelled.", "AbortError");
    }

    node.__promptEnhancerPendingToken = token;
    pendingNodes.set(String(node.id), node);

    if (typeof app?.graphToPrompt !== "function" || typeof api?.queuePrompt !== "function") {
      throw new Error("This ComfyUI frontend does not support targeted execution required for connected media/settings enhancement.");
    }

    const executionResult = new Promise((resolve, reject) => {
      node.__promptEnhancerExecutionWaiter = { token, resolve, reject };
    });

    // Queue directly through the stable API so we receive the actual prompt/job
    // id. The execution waiter above is installed first so even a very fast local
    // partial run cannot finish before the browser is ready to consume it.
    const promptData = await app.graphToPrompt();
    if (node.__promptEnhancerCancelRequested) {
      await cancelArmed(node, token);
      throw new DOMException("Enhancement cancelled.", "AbortError");
    }
    const queued = await api.queuePrompt(0, promptData, {
      partialExecutionTargets: [String(node.id)],
    });
    const jobId = String(queued?.prompt_id || "");
    if (!jobId) {
      const message = queued?.error?.message || "ComfyUI rejected the Prompt Enhancer partial execution.";
      throw new Error(message);
    }
    node.__promptEnhancerJobId = jobId;
    node.__promptEnhancerRequestController = null;

    // Match normal ComfyUI seed-control behavior once this individual batch item
    // is accepted. The next item captures the resulting seed before it queues.
    finishSeedControlCycle(node);

    if (node.__promptEnhancerCancelRequested) {
      try { await api?.cancelJob?.(jobId); } catch (_) { try { await api?.interrupt?.(jobId); } catch (_) {} }
      await cancelArmed(node, token);
      throw new DOMException("Enhancement cancelled.", "AbortError");
    }

    node.__promptEnhancerTimeout = setTimeout(async () => {
      if (String(node.__promptEnhancerPendingToken || "") !== token) return;
      const timedOutJob = String(node.__promptEnhancerJobId || "");
      await cancelArmed(node, token);
      if (timedOutJob) {
        try { await api?.cancelJob?.(timedOutJob); } catch (_) { try { await api?.interrupt?.(timedOutJob); } catch (_) {} }
      }
      clearEnhanceTransport(node);
      settleEnhanceExecution(node, new Error("Enhancement did not complete before the request timeout."));
    }, ENHANCE_TIMEOUT_MS);

    return await executionResult;
  } catch (error) {
    if (token) await cancelArmed(node, token);
    clearEnhanceTransport(node);
    // If the failure occurred after the waiter was installed, settle it so no
    // promise remains pending while the batch runner handles the same exception.
    if (node.__promptEnhancerExecutionWaiter) {
      const waiter = node.__promptEnhancerExecutionWaiter;
      node.__promptEnhancerExecutionWaiter = null;
      // Resolve the internal waiter defensively; the thrown error below is the
      // authoritative failure path for this function.
      waiter.resolve(null);
    }
    throw error;
  }
}

async function enhance(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const promptWidget = widget(node, "prompt");
  const instructionsWidget = widget(node, "enhancement_text");
  const prompt = String(promptWidget?.value ?? "");
  const enhancementText = String(instructionsWidget?.value ?? "");

  if (!prompt.trim()) {
    notify("warn", "Local LLM Prompt Enhancer", "Enter a Prompt first.");
    return;
  }
  if (!enhancementText.trim()) {
    notify("warn", "Local LLM Prompt Enhancer", "Enhancement Instructions are empty.");
    return;
  }

  const count = enhanceBatchCount(node);
  if (count > 1) enforceBatchAddNew(node);

  node.__promptEnhancerCancelRequested = false;
  node.__promptEnhancerOverwriteAtStart = count > 1 ? false : !!widget(node, "overwrite_enhanced")?.value;
  node.__promptEnhancerBatchProgress = { current: 0, total: count };
  setBusy(node, true, "enhance");

  const usesPartialExecution = hasConnectedMedia(node) || hasConnectedSettings(node);
  let completed = 0;
  let mediaSummary = "";
  let batchId = "";

  try {
    // Only multi-item Enhance runs need a pinned session. The backend keeps the
    // full server/model configuration opaque and immutable for this batch while
    // the modal remains free to autosave settings for the next request.
    if (count > 1) batchId = await beginEnhancementBatch(count);

    for (let index = 0; index < count; index += 1) {
      if (node.__promptEnhancerCancelRequested) break;
      node.__promptEnhancerBatchProgress = { current: index + 1, total: count };
      scheduleDomPanelSync(node);

      const seed = beginSeedControlCycle(node);
      let data;
      if (usesPartialExecution) {
        data = await runMediaEnhancementOnce(node, prompt, enhancementText, seed, batchId);
      } else {
        data = await runTextOnlyEnhancementOnce(node, prompt, enhancementText, seed, batchId);
        finishSeedControlCycle(node);
      }

      if (node.__promptEnhancerCancelRequested) break;
      const revised = String(data?.prompt ?? "");
      if (!revised.trim()) throw new Error("The Local LLM returned an empty prompt.");
      applyGeneratedPrompt(node, revised, node.__promptEnhancerOverwriteAtStart);
      completed += 1;

      if (usesPartialExecution && !mediaSummary) {
        const media = [];
        if (data?.used_images) media.push("image(s)");
        if (data?.used_video) media.push("video");
        mediaSummary = media.length ? ` using ${media.join(" + ")}` : "";
      }
    }

    if (!node.__promptEnhancerCancelRequested && completed > 0) {
      const noun = completed === 1 ? "Enhancement" : `${completed} enhancements`;
      notify("success", "Local LLM Prompt Enhancer", `${noun} complete${mediaSummary}.`);
    }
  } catch (error) {
    if (!node.__promptEnhancerCancelRequested && error?.name !== "AbortError") {
      notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
    }
  } finally {
    await endEnhancementBatch(batchId);
    clearEnhanceTransport(node);
    node.__promptEnhancerExecutionWaiter = null;
    node.__promptEnhancerCancelRequested = false;
    node.__promptEnhancerOverwriteAtStart = null;
    node.__promptEnhancerBatchProgress = null;
    if (node.__promptEnhancerManualSeedSource) node.__promptEnhancerManualSeedSource = null;
    setBusy(node, false);
  }
}

function toggleEnhance(node) {
  if (node?.__promptEnhancerBusy && node.__promptEnhancerActiveAction === "enhance") {
    void cancelEnhancement(node);
    return;
  }
  void enhance(node);
}

function useEnhancedAsPrompt(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const promptWidget = widget(node, "prompt");
  const enhancedWidget = widget(node, "enhanced_prompt");
  const enhanced = String(enhancedWidget?.value ?? "");
  if (!enhanced.trim()) {
    notify("warn", "Local LLM Prompt Enhancer", "Enhanced Prompt is empty.");
    return;
  }

  node.__promptEnhancerPromptHistory ||= [];
  node.__promptEnhancerPromptHistory.push(String(promptWidget?.value ?? ""));
  if (node.__promptEnhancerPromptHistory.length > 20) node.__promptEnhancerPromptHistory.shift();
  setWidgetValue(promptWidget, enhanced, true);
  markWorkflowChanged(node);
  redraw(node);
}

function undoPromptPromotion(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const history = node.__promptEnhancerPromptHistory || [];
  if (!history.length) {
    notify("warn", "Local LLM Prompt Enhancer", "Nothing to undo from the ↑ action this session.");
    return;
  }
  const promptWidget = widget(node, "prompt");
  setWidgetValue(promptWidget, history.pop(), true);
  markWorkflowChanged(node);
  redraw(node);
}

async function saveTemplate(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const textWidget = widget(node, "enhancement_text");
  const text = String(textWidget?.value ?? "");
  const selected = selectedTemplate(node);

  if (!text.trim()) {
    notify("warn", "Local LLM Prompt Enhancer", "Enhancement Instructions are empty.");
    return;
  }

  const suggested = selected?.source === "user" ? String(selected.name || "") : "";
  const entered = window.prompt("Save enhancement template as:", suggested);
  if (entered === null) return;
  const name = String(entered).trim();
  if (!name) {
    notify("warn", "Local LLM Prompt Enhancer", "Template name cannot be empty.");
    return;
  }

  setBusy(node, true, "save");
  try {
    const data = await postJSON("/local_llm_prompt_enhancer/save_template", { name, text });
    const savedLabel = String(data?.saved?.label || "");
    // Save returns the freshly rescanned template list, so the dropdown updates
    // immediately, so no separate rescan control is needed.
    applyTemplateRecords(node, data, { loadSelected: true, selectLabel: savedLabel });
    notify("success", "Local LLM Prompt Enhancer", `${savedLabel || "Template"} saved.`);
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  } finally {
    setBusy(node, false);
  }
}

async function deleteTemplate(node) {
  if (!node || node.__promptEnhancerBusy) return;
  const selected = selectedTemplate(node);
  if (!selected) {
    notify("warn", "Local LLM Prompt Enhancer", "Select a saved user template to delete.");
    return;
  }
  if (selected.protected || selected.source !== "user") {
    notify("warn", "Local LLM Prompt Enhancer", "Built-in default templates are protected and cannot be deleted.");
    return;
  }

  const label = String(selected.label || `User / ${selected.name || "Template"}`);
  const confirmed = globalThis.confirm?.(`Delete enhancement template "${label}"?\n\nThis cannot be undone.`);
  if (!confirmed) return;

  setBusy(node, true, "delete_template");
  try {
    const data = await postJSON("/local_llm_prompt_enhancer/delete_template", { label });
    const fallback = String(data?.default || DEFAULT_PRESET);
    applyTemplateRecords(node, data, { loadSelected: true, selectLabel: fallback });
    markWorkflowChanged(node);
    notify("success", "Local LLM Prompt Enhancer", `${label} deleted.`);
  } catch (error) {
    notify("error", "Local LLM Prompt Enhancer", error?.message || String(error));
  } finally {
    setBusy(node, false);
  }
}

const TEXTAREA_KEYS = ["prompt", "enhanced", "instructions"];

function normalizeTextareaHeight(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 80) return 0;
  return Math.max(80, Math.min(1600, Math.round(n)));
}

function textareaHeightSnapshot(node) {
  const prior = node?.__promptEnhancerTextareaHeights || {};
  const controls = node?.__promptEnhancerPanelControls || {};
  const result = {};
  for (const key of TEXTAREA_KEYS) {
    const element = controls[key];
    const measured = normalizeTextareaHeight(element?.getBoundingClientRect?.().height || element?.offsetHeight);
    const fallback = normalizeTextareaHeight(prior[key]);
    const value = measured || fallback;
    if (value) result[key] = value;
  }
  return result;
}

function applyTextareaHeights(node, heights, { schedule = true } = {}) {
  if (!node) return;
  const normalized = {};
  for (const key of TEXTAREA_KEYS) {
    const h = normalizeTextareaHeight(heights?.[key]);
    if (h) normalized[key] = h;
  }
  node.__promptEnhancerTextareaHeights = normalized;
  const controls = node.__promptEnhancerPanelControls || {};
  node.__promptEnhancerApplyingTextareaHeights = true;
  try {
    for (const key of TEXTAREA_KEYS) {
      const element = controls[key];
      if (!element) continue;
      const h = normalized[key];
      element.style.height = h ? `${h}px` : "";
    }
  } finally {
    node.__promptEnhancerApplyingTextareaHeights = false;
  }
  if (schedule) scheduleDomPanelHeight(node);
}

function captureTextareaHeights(node) {
  if (!node || node.__promptEnhancerApplyingTextareaHeights || !node.__promptEnhancerPanelControls) return;
  const next = textareaHeightSnapshot(node);
  const previous = node.__promptEnhancerTextareaHeights || {};
  const changed = TEXTAREA_KEYS.some((key) => Number(next[key] || 0) !== Number(previous[key] || 0));
  if (!changed) return;
  node.__promptEnhancerTextareaHeights = next;
  persistEnhancementState(node);
  markWorkflowChanged(node);
  scheduleDomPanelHeight(node);
}

const PERSISTENCE_STATE_VERSION = 1;
const PERSISTENCE_STATE_KEY = "local_llm_prompt_enhancer_state";

function normalizedHistorySnapshot(node) {
  let history = promptHistory(node);
  let index = promptHistoryIndex(node, history);
  const enhanced = String(widget(node, "enhanced_prompt")?.value ?? "");

  // The visible editor is authoritative for the active array entry. Do this
  // during every persistence snapshot as well as in the widget callback so a
  // refresh/copy immediately after typing cannot lose the last edit.
  if (history.length) {
    history[index] = enhanced;
  } else if (enhanced.trim()) {
    history = [enhanced];
    index = 0;
  } else {
    index = 0;
  }

  const shuffle = promptShuffleState(node)
    .filter((v) => Number.isInteger(v) && v >= 0 && v < history.length && v !== index);
  return { history, index, shuffle, enhanced };
}

function enhancementStateSnapshot(node) {
  const normalized = normalizedHistorySnapshot(node);
  const seedWidget = widget(node, "seed");
  const seedRaw = Number(seedWidget?.value ?? 0);
  const seed = Number.isFinite(seedRaw) ? Math.max(0, Math.trunc(seedRaw)) : 0;
  const control = seedControlWidget(node);

  return {
    version: PERSISTENCE_STATE_VERSION,
    prompt: String(widget(node, "prompt")?.value ?? ""),
    promptPreset: String(widget(node, "prompt_preset")?.value ?? "Custom"),
    enhancedPrompt: normalized.enhanced,
    promptSet: String(widget(node, "prompt_set")?.value ?? PROMPT_SET_NONE),
    promptHistory: normalized.history,
    promptHistoryIndex: normalized.index,
    promptShuffle: normalized.shuffle,
    promptCycle: String(widget(node, "prompt_cycle")?.value ?? "fixed"),
    overwriteEnhanced: !!widget(node, "overwrite_enhanced")?.value,
    batchCount: enhanceBatchCount(node),
    enhancementPreset: String(widget(node, "enhancement_preset")?.value ?? ""),
    enhancementText: String(widget(node, "enhancement_text")?.value ?? ""),
    seed,
    seedControl: control ? String(control.value ?? "") : "",
    enhanceWithWorkflow: !!widget(node, "enhance_with_workflow")?.value,
    textareaHeights: textareaHeightSnapshot(node),
  };
}

function clonePersistenceState(state) {
  return {
    version: Number(state?.version ?? PERSISTENCE_STATE_VERSION),
    prompt: String(state?.prompt ?? ""),
    promptPreset: String(state?.promptPreset ?? "Custom"),
    enhancedPrompt: String(state?.enhancedPrompt ?? ""),
    promptSet: String(state?.promptSet ?? PROMPT_SET_NONE),
    promptHistory: Array.isArray(state?.promptHistory) ? state.promptHistory.map((v) => String(v ?? "")) : [],
    promptHistoryIndex: Number(state?.promptHistoryIndex ?? 0),
    promptShuffle: Array.isArray(state?.promptShuffle) ? state.promptShuffle.map((v) => Number(v)) : [],
    promptCycle: String(state?.promptCycle ?? "fixed"),
    overwriteEnhanced: !!state?.overwriteEnhanced,
    batchCount: normalizeEnhanceBatchCount(state?.batchCount ?? ENHANCE_BATCH_MIN),
    enhancementPreset: String(state?.enhancementPreset ?? ""),
    enhancementText: String(state?.enhancementText ?? ""),
    seed: Number(state?.seed ?? 0),
    seedControl: String(state?.seedControl ?? ""),
    enhanceWithWorkflow: !!state?.enhanceWithWorkflow,
    textareaHeights: Object.fromEntries(
      TEXTAREA_KEYS.map((key) => [key, normalizeTextareaHeight(state?.textareaHeights?.[key])]).filter(([, value]) => value)
    ),
  };
}

function syncNativeStateWidgets(node, state) {
  if (!node || !state) return;
  const history = Array.isArray(state.promptHistory) ? state.promptHistory : [];
  const index = history.length
    ? Math.max(0, Math.min(Math.trunc(Number(state.promptHistoryIndex) || 0), history.length - 1))
    : 0;
  const shuffle = Array.isArray(state.promptShuffle)
    ? state.promptShuffle.filter((v) => Number.isInteger(v) && v >= 0 && v < history.length && v !== index)
    : [];

  node.__promptEnhancerSyncingHistory = true;
  try {
    setWidgetValue(widget(node, "prompt_history_json"), JSON.stringify(history), false);
    setWidgetValue(widget(node, "prompt_history_index"), index, false);
    setWidgetValue(widget(node, "prompt_shuffle_json"), JSON.stringify(shuffle), false);
  } finally {
    node.__promptEnhancerSyncingHistory = false;
  }
}

function persistEnhancementState(node, serializedData = null) {
  if (!node || node.__promptEnhancerRestoringState) return null;
  const state = enhancementStateSnapshot(node);
  syncNativeStateWidgets(node, state);

  node.properties ||= {};
  node.properties[PERSISTENCE_STATE_KEY] = clonePersistenceState(state);

  if (serializedData && typeof serializedData === "object") {
    serializedData.properties ||= {};
    serializedData.properties[PERSISTENCE_STATE_KEY] = clonePersistenceState(state);
  }
  return state;
}

function restoreEnhancementState(node) {
  const raw = node?.properties?.[PERSISTENCE_STATE_KEY];
  if (!raw || typeof raw !== "object" || Number(raw.version) !== PERSISTENCE_STATE_VERSION) return false;
  const state = clonePersistenceState(raw);

  node.__promptEnhancerRestoringState = true;
  node.__promptEnhancerSyncingHistory = true;
  try {
    setWidgetValue(widget(node, "prompt_preset"), state.promptPreset || "Custom", false);
    setWidgetValue(widget(node, "prompt"), state.prompt, false);
    setWidgetValue(widget(node, "prompt_set"), state.promptSet || PROMPT_SET_NONE, false);
    setWidgetValue(widget(node, "prompt_cycle"), state.promptCycle || "fixed", false);
    node.__promptEnhancerBatchCount = normalizeEnhanceBatchCount(state.batchCount ?? ENHANCE_BATCH_MIN);
    setWidgetValue(widget(node, "overwrite_enhanced"), node.__promptEnhancerBatchCount > 1 ? false : state.overwriteEnhanced, false);
    setWidgetValue(widget(node, "enhancement_preset"), state.enhancementPreset, false);
    setWidgetValue(widget(node, "enhancement_text"), state.enhancementText, false);
    setWidgetValue(widget(node, "seed"), Math.max(0, Math.trunc(Number(state.seed) || 0)), false);
    setWidgetValue(widget(node, "enhance_with_workflow"), node.__promptEnhancerBatchCount > 1 ? false : state.enhanceWithWorkflow, false);

    const history = state.promptHistory;
    const index = history.length
      ? Math.max(0, Math.min(Math.trunc(Number(state.promptHistoryIndex) || 0), history.length - 1))
      : 0;
    const shuffle = state.promptShuffle
      .filter((v) => Number.isInteger(v) && v >= 0 && v < history.length && v !== index);
    setWidgetValue(widget(node, "prompt_history_json"), JSON.stringify(history), false);
    setWidgetValue(widget(node, "prompt_history_index"), index, false);
    setWidgetValue(widget(node, "prompt_shuffle_json"), JSON.stringify(shuffle), false);
    setWidgetValue(widget(node, "enhanced_prompt"), history.length ? history[index] : state.enhancedPrompt, false);

    const control = seedControlWidget(node);
    if (control && state.seedControl) setWidgetValue(control, state.seedControl, false);
    applyTextareaHeights(node, state.textareaHeights, { schedule: false });
  } finally {
    node.__promptEnhancerSyncingHistory = false;
    node.__promptEnhancerRestoringState = false;
  }
  redraw(node);
  return true;
}

function wrapNodeSerialization(node) {
  if (!node || node.__promptEnhancerSerializationWrapped) return;
  node.__promptEnhancerSerializationWrapped = true;
  const oldSerialize = node.onSerialize;
  node.onSerialize = function (data) {
    const result = oldSerialize?.call(this, data);
    // Always stamp current widget values into the exact object being copied,
    // saved, tabbed, or autosaved. This is the final authority for persistence.
    persistEnhancementState(this, data);
    return result;
  };
}

function wrapSimpleStatePersistence(node, name) {
  const w = widget(node, name);
  if (!w || w.__promptEnhancerStatePersistenceWrapped) return;
  w.__promptEnhancerStatePersistenceWrapped = true;
  const original = w.callback;
  w.callback = function (...args) {
    const result = original?.apply(this, args);
    if (!node.__promptEnhancerRestoringState) {
      persistEnhancementState(node);
      markWorkflowChanged(node);
    }
    return result;
  };
}

function wrapSeedControlPersistence(node) {
  const control = seedControlWidget(node);
  if (!control || control.__promptEnhancerStatePersistenceWrapped) return;
  control.__promptEnhancerStatePersistenceWrapped = true;

  const originalCallback = control.callback;
  if (typeof originalCallback === "function") {
    control.callback = function (...args) {
      const result = originalCallback.apply(this, args);
      if (!node.__promptEnhancerRestoringState) {
        persistEnhancementState(node);
        markWorkflowChanged(node);
      }
      return result;
    };
  }

  const originalBeforeQueued = control.beforeQueued;
  if (typeof originalBeforeQueued === "function") {
    control.beforeQueued = function (...args) {
      if (hasConnectedSettings(node)) return;
      return originalBeforeQueued.apply(this, args);
    };
  }

  const originalAfterQueued = control.afterQueued;
  if (typeof originalAfterQueued === "function") {
    control.afterQueued = function (...args) {
      if (hasConnectedSettings(node)) return;
      const result = originalAfterQueued.apply(this, args);
      if (!node.__promptEnhancerRestoringState) {
        persistEnhancementState(node);
        markWorkflowChanged(node);
      }
      return result;
    };
  }
}

function wrapEnhancementTextPersistence(node) {
  const textWidget = widget(node, "enhancement_text");
  if (!textWidget || textWidget.__promptEnhancerPersistenceWrapped) return;
  textWidget.__promptEnhancerPersistenceWrapped = true;
  const original = textWidget.callback;
  textWidget.callback = function (...args) {
    const result = original?.apply(this, args);
    if (!node.__promptEnhancerRestoringState) {
      persistEnhancementState(node);
      markWorkflowChanged(node);
    }
    return result;
  };
}

function wrapEnhancedPromptHistory(node) {
  const enhancedWidget = widget(node, "enhanced_prompt");
  if (!enhancedWidget || enhancedWidget.__promptEnhancerHistoryWrapped) return;
  enhancedWidget.__promptEnhancerHistoryWrapped = true;
  const original = enhancedWidget.callback;
  enhancedWidget.callback = function (...args) {
    const result = original?.apply(this, args);
    if (!node.__promptEnhancerSyncingHistory && !node.__promptEnhancerRestoringState) {
      syncVisibleEnhancedIntoHistory(node);
      persistEnhancementState(node);
      markWorkflowChanged(node);
    }
    return result;
  };
}

function hideInternalWidget(w) {
  if (!w) return;
  w.__promptEnhancerHiddenInternal = true;
  w.hidden = true;
  w.options ||= {};
  w.options.hidden = true;
  // Keep the original widget type intact so workflow/prompt serialization remains
  // native and stable. Legacy layout also needs a zero-size fallback.
  w.computeSize = () => [0, 0];
  w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
}

function wrapPresetCallback(node) {
  const presetWidget = widget(node, "enhancement_preset");
  if (!presetWidget || presetWidget.__promptEnhancerWrapped) return;
  presetWidget.__promptEnhancerWrapped = true;
  const original = presetWidget.callback;
  presetWidget.callback = function (...args) {
    const result = original?.apply(this, args);
    if (!node.__promptEnhancerSettingPreset) loadSelectedTemplate(node);
    persistEnhancementState(node);
    markWorkflowChanged(node);
    return result;
  };
}

function wrapPromptPresetCallbacks(node) {
  const selector = widget(node, "prompt_preset");
  if (selector && !selector.__promptEnhancerPromptPresetWrapped) {
    selector.__promptEnhancerPromptPresetWrapped = true;
    const original = selector.callback;
    selector.callback = function (...args) {
      const result = original?.apply(this, args);
      const value = String(this?.value ?? args?.[0] ?? "Custom");
      if (!node.__promptEnhancerApplyingPromptPreset && !node.__promptEnhancerRestoringState) {
        void applyPromptPresetSelection(node, value);
        markWorkflowChanged(node);
      }
      persistEnhancementState(node);
      scheduleDomPanelSync(node);
      return result;
    };
  }

  const promptWidget = widget(node, "prompt");
  if (promptWidget && !promptWidget.__promptEnhancerPromptPresetTextWrapped) {
    promptWidget.__promptEnhancerPromptPresetTextWrapped = true;
    const original = promptWidget.callback;
    promptWidget.callback = function (...args) {
      const result = original?.apply(this, args);
      if (!node.__promptEnhancerApplyingPromptPreset && !node.__promptEnhancerRestoringState) {
        const preset = widget(node, "prompt_preset");
        if (preset && String(preset.value ?? "Custom") !== "Custom") setWidgetValue(preset, "Custom", false);
        persistEnhancementState(node);
        markWorkflowChanged(node);
      }
      scheduleDomPanelSync(node);
      return result;
    };
  }
}

function wrapPromptSetCallback(node) {
  const setWidget = widget(node, "prompt_set");
  if (!setWidget || setWidget.__promptEnhancerPromptSetWrapped) return;
  setWidget.__promptEnhancerPromptSetWrapped = true;
  const original = setWidget.callback;
  setWidget.callback = function (...args) {
    const result = original?.apply(this, args);
    if (!node.__promptEnhancerSettingPromptSet) {
      const name = String(this?.value ?? PROMPT_SET_NONE);
      if (name !== PROMPT_SET_NONE) void loadPromptSet(node, name);
    }
    persistEnhancementState(node);
    markWorkflowChanged(node);
    return result;
  };
}

function setWidgetDisplayLabel(w, label) {
  if (!w) return;
  w.label = label;
  w.options ||= {};
  w.options.label = label;
}


function panelComboValues(w) {
  if (!w) return [];
  let values = w.options?.values;
  if (typeof values === "function") {
    try { values = values(); } catch (_) { values = []; }
  }
  if (!Array.isArray(values)) return [];
  return values.map((v) => String(v ?? ""));
}

function panelStyle(el, styles) {
  Object.assign(el.style, styles || {});
  return el;
}

function panelElement(tag, className = "") {
  const el = document.createElement(tag);
  if (className) el.className = className;
  return el;
}

function panelHasScrollableOverflow(el) {
  if (!(el instanceof HTMLElement)) return false;
  try {
    const style = getComputedStyle(el);
    const scrollY = /^(auto|scroll|overlay)$/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1;
    const scrollX = /^(auto|scroll|overlay)$/.test(style.overflowX) && el.scrollWidth > el.clientWidth + 1;
    return scrollY || scrollX;
  } catch (_) {
    return false;
  }
}

// Current ComfyUI Nodes 2.0 lets a DOM subtree opt out of canvas wheel handling
// with data-capture-wheel="true". Mark only genuinely overflowing controls;
// ordinary DOM regions continue to forward wheel input to the workspace canvas.
function panelWrapScrollableControl(scrollEl) {
  const shell = panelElement("div", "pe-wheel-capture");
  shell.tabIndex = -1;
  shell.__promptEnhancerWheelScrollElement = scrollEl;
  shell.appendChild(scrollEl);

  const syncCapture = () => {
    if (panelHasScrollableOverflow(scrollEl)) shell.dataset.captureWheel = "true";
    else shell.removeAttribute("data-capture-wheel");
  };
  scrollEl.__promptEnhancerSyncWheelCapture = syncCapture;

  // Hovering a scrollable field is enough for that field to own the wheel.
  // Do not move keyboard focus just to make scrolling work: native textarea/
  // scroll-container wheel behavior does not require focus, and focus stealing
  // was the reason some fields still leaked wheel input back to the workspace.
  shell.addEventListener("pointerenter", syncCapture);
  scrollEl.addEventListener("input", () => requestAnimationFrame(syncCapture));
  requestAnimationFrame(syncCapture);
  return shell;
}

function panelShouldCaptureWheel(event, root) {
  const target = event.target instanceof Element ? event.target : null;
  const capture = target?.closest?.(".pe-wheel-capture");
  if (!capture || !root.contains(capture)) return false;
  const scrollEl = capture.__promptEnhancerWheelScrollElement;
  const scrollable = !!scrollEl && panelHasScrollableOverflow(scrollEl);
  if (scrollable) capture.dataset.captureWheel = "true";
  else capture.removeAttribute("data-capture-wheel");
  return scrollable;
}

// DOM widgets are separate browser hit targets from LiteGraph's <canvas> in
// Classic mode. Merely allowing a wheel event to bubble therefore cannot make
// the canvas see it. Current ComfyUI uses the same forwarding pattern for Vue
// UI surfaces: consume the original event and dispatch an equivalent wheel
// event directly at the real canvas element. This also works as a fallback when
// a Nodes 2.0 host does not forward a particular custom DOM subtree itself.
function panelForwardWheelToCanvas(event) {
  const canvasEl = app?.canvas?.canvas;
  if (!(canvasEl instanceof HTMLCanvasElement)) return false;
  event.preventDefault();
  event.stopPropagation();
  canvasEl.dispatchEvent(new WheelEvent("wheel", {
    clientX: event.clientX,
    clientY: event.clientY,
    screenX: event.screenX,
    screenY: event.screenY,
    deltaX: event.deltaX,
    deltaY: event.deltaY,
    deltaZ: event.deltaZ,
    deltaMode: event.deltaMode,
    ctrlKey: event.ctrlKey,
    metaKey: event.metaKey,
    shiftKey: event.shiftKey,
    altKey: event.altKey,
    bubbles: false,
    cancelable: true,
  }));
  return true;
}

function panelHandleWheel(event, root) {
  // Any genuinely overflowing DOM field owns wheel input while hovered. Only
  // non-scrollable overlay regions are forwarded to the ComfyUI workspace.
  if (panelShouldCaptureWheel(event, root)) {
    event.stopPropagation();
    return;
  }
  panelForwardWheelToCanvas(event);
}

function hideNativeWidgetForPanel(w) {
  if (!w || w.__promptEnhancerDomHidden) return;
  w.__promptEnhancerDomHidden = true;
  w.hidden = true;
  w.options ||= {};
  w.options.hidden = true;
  // Keep the widget and its real value/callback alive for ComfyUI prompt and
  // workflow serialization; only remove its visual footprint.
  w.computeSize = () => [0, 0];
  w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
}

function hideNativeWidgetsForPanel(node) {
  if (!node?.__promptEnhancerDomPanelWidget) return;
  for (const name of [
    "prompt_preset",
    "prompt",
    "enhanced_prompt",
    "prompt_set",
    "prompt_cycle",
    "overwrite_enhanced",
    "enhancement_preset",
    "enhancement_text",
    "seed",
    "enhance_with_workflow",
    "prompt_history_json",
    "prompt_history_index",
    "prompt_shuffle_json",
  ]) {
    hideNativeWidgetForPanel(widget(node, name));
  }
  hideNativeWidgetForPanel(seedControlWidget(node));
}

function removeCopiedDomPanelWidget(node) {
  if (!node?.widgets) return;
  for (let i = node.widgets.length - 1; i >= 0; i -= 1) {
    const w = node.widgets[i];
    if (w?.name !== "prompt_enhancer_panel") continue;
    node.widgets.splice(i, 1);
    try { w.onRemove?.(); } catch (_) {}
  }
}

function setPanelSelectOptions(select, values, selected) {
  if (!select) return;
  const normalized = Array.isArray(values) ? values.map(String) : [];
  const signature = JSON.stringify(normalized);
  if (select.__promptEnhancerOptionsSignature !== signature) {
    select.__promptEnhancerOptionsSignature = signature;
    const fragment = document.createDocumentFragment();
    for (const value of normalized) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      fragment.appendChild(option);
    }
    select.replaceChildren(fragment);
  }
  const next = String(selected ?? "");
  if (select.value !== next) select.value = next;
}

function setPanelNativeValue(node, name, value) {
  const w = widget(node, name);
  if (!w) return;
  const batchLocked = enhanceBatchCount(node) > 1;
  const next = batchLocked && (name === "overwrite_enhanced" || name === "enhance_with_workflow") ? false : value;
  setWidgetValue(w, next, true);
  persistEnhancementState(node);
  markWorkflowChanged(node);
  if (name === "prompt_cycle" || name === "enhance_with_workflow") resetBackendPromptCycle(node);
  scheduleDomPanelSync(node);
}


const PE_INTEGER_FIELDS = new Set(["seed"]);

function panelNumericStep(w, name) {
  const exact = Number(w?.options?.step2);
  if (Number.isFinite(exact) && exact > 0) return exact;
  // Legacy ComfyUI numeric widgets store `step` at 10x the input-spec step.
  // Newer frontends expose the exact value as `step2`.
  const legacy = Number(w?.options?.step);
  if (Number.isFinite(legacy) && legacy > 0) return legacy / 10;
  const precision = Number(w?.options?.precision);
  if (Number.isInteger(precision) && precision >= 0 && precision <= 12) return precision === 0 ? 1 : 10 ** (-precision);
  return PE_INTEGER_FIELDS.has(name) ? 1 : 1;
}

function panelNumericPrecision(w, name, step = panelNumericStep(w, name)) {
  const configured = Number(w?.options?.precision);
  if (Number.isInteger(configured) && configured >= 0 && configured <= 12) return configured;
  if (PE_INTEGER_FIELDS.has(name)) return 0;
  if (!Number.isFinite(step) || step <= 0) return undefined;
  const value = String(step).toLowerCase();
  if (value.includes("e-")) {
    const n = Number(value.split("e-")[1]);
    return Number.isFinite(n) ? Math.min(12, Math.max(0, n)) : undefined;
  }
  const dot = value.indexOf(".");
  return dot < 0 ? 0 : Math.min(12, value.length - dot - 1);
}

function normalizePanelNumericValue(w, name, raw) {
  let value = Number(raw);
  if (!Number.isFinite(value)) value = Number(w?.value) || 0;
  const min = Number(w?.options?.min);
  const max = Number(w?.options?.max);
  const step = panelNumericStep(w, name);
  if (Number.isFinite(step) && step > 0) {
    const anchor = Number.isFinite(min) ? min : 0;
    value = anchor + Math.round((value - anchor) / step) * step;
    const precision = panelNumericPrecision(w, name, step);
    if (Number.isInteger(precision) && precision >= 0 && precision <= 12) value = Number(value.toFixed(precision));
  }
  if (Number.isFinite(min)) value = Math.max(min, value);
  if (Number.isFinite(max)) value = Math.min(max, value);
  if (PE_INTEGER_FIELDS.has(name)) value = Math.round(value);
  return value;
}

function formatPanelNumericValue(w, name, value = w?.value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value ?? "");
  const precision = panelNumericPrecision(w, name);
  return Number.isInteger(precision) && precision >= 0 ? n.toFixed(precision) : String(n);
}

function panelNumericPercent(w, value = w?.value) {
  const min = Number(w?.options?.min);
  const max = Number(w?.options?.max);
  const n = Number(value);
  if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(n) || max <= min) return null;
  return Math.max(0, Math.min(100, ((n - min) / (max - min)) * 100));
}

function setPanelNativeNumericValue(node, name, raw) {
  const w = widget(node, name);
  if (!w) return;
  setPanelNativeValue(node, name, normalizePanelNumericValue(w, name, raw));
}

function panelNumberStepIcon(kind) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = kind === "minus" ? '<path d="M5 12h14"/>' : '<path d="M12 5v14M5 12h14"/>';
  return svg;
}

function setPanelNumberEditing(input, editing) {
  const control = input?.__promptEnhancerNumberControl;
  const shell = control?.shell;
  if (!shell) return;
  shell.classList.toggle("pe-number-editing", !!editing);
  if (editing) {
    input.value = String(widget(input.__promptEnhancerNode, input.__promptEnhancerName)?.value ?? input.value);
    requestAnimationFrame(() => {
      try { input.focus({ preventScroll: true }); input.select(); } catch (_) {}
    });
  }
}

function syncPanelNumberControl(input, w, name) {
  const control = input?.__promptEnhancerNumberControl;
  if (!input || !control || !w) return;
  const editing = control.shell.classList.contains("pe-number-editing") && document.activeElement === input;
  if (!editing) {
    const value = formatPanelNumericValue(w, name, w.value);
    if (input.value !== value) input.value = value;
  }
  const pct = panelNumericPercent(w, w.value);
  if (pct == null) {
    control.fill.style.display = "none";
    control.fill.style.width = "0%";
  } else {
    control.fill.style.display = "block";
    control.fill.style.width = `${pct}%`;
  }
  const min = Number(w.options?.min);
  const max = Number(w.options?.max);
  const value = Number(w.value);
  const disabled = !!input.disabled;
  control.dec.disabled = disabled || (Number.isFinite(min) && Number.isFinite(value) && value <= min);
  control.inc.disabled = disabled || (Number.isFinite(max) && Number.isFinite(value) && value >= max);
  input.setAttribute("aria-valuenow", String(w.value ?? ""));
  if (Number.isFinite(min)) input.setAttribute("aria-valuemin", String(min)); else input.removeAttribute("aria-valuemin");
  if (Number.isFinite(max)) input.setAttribute("aria-valuemax", String(max)); else input.removeAttribute("aria-valuemax");
}

function setPanelNumberDisabled(input, disabled) {
  if (!input) return;
  input.disabled = !!disabled;
  const control = input.__promptEnhancerNumberControl;
  if (!control) return;
  control.shell.classList.toggle("pe-number-disabled", !!disabled);
  if (disabled && control.shell.classList.contains("pe-number-editing")) {
    control.shell.classList.remove("pe-number-editing");
    try { input.blur(); } catch (_) {}
  }
  syncPanelNumberControl(input, widget(input.__promptEnhancerNode, input.__promptEnhancerName), input.__promptEnhancerName);
}

function createPanelNumberControl(node, name, labelText) {
  const w = widget(node, name);
  if (!w) return null;
  const shell = panelElement("div", "pe-number");
  shell.title = w.options?.tooltip || labelText || name;
  const fill = panelElement("div", "pe-number-fill");
  const dec = panelElement("button", "pe-number-step pe-number-dec");
  dec.type = "button";
  dec.setAttribute("aria-label", `Decrease ${labelText || name}`);
  dec.appendChild(panelNumberStepIcon("minus"));
  const center = panelElement("div", "pe-number-center");
  const input = panelElement("input", "pe-number-input");
  input.type = "text";
  input.inputMode = PE_INTEGER_FIELDS.has(name) ? "numeric" : "decimal";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.setAttribute("role", "spinbutton");
  input.setAttribute("aria-label", labelText || name);
  const scrub = panelElement("div", "pe-number-scrub");
  scrub.setAttribute("aria-hidden", "true");
  const inc = panelElement("button", "pe-number-step pe-number-inc");
  inc.type = "button";
  inc.setAttribute("aria-label", `Increase ${labelText || name}`);
  inc.appendChild(panelNumberStepIcon("plus"));
  center.append(input, scrub);
  shell.append(fill, dec, center, inc);
  input.__promptEnhancerNode = node;
  input.__promptEnhancerName = name;
  input.__promptEnhancerNumberControl = { shell, fill, dec, inc, center, scrub };

  const nudge = (direction, multiplier = 1) => {
    if (input.disabled) return;
    const step = panelNumericStep(w, name);
    const current = Number(w.value) || 0;
    setPanelNativeNumericValue(node, name, current + direction * step * multiplier);
  };
  dec.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); nudge(-1); });
  inc.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); nudge(1); });

  const commit = () => {
    if (!shell.classList.contains("pe-number-editing")) return;
    setPanelNativeNumericValue(node, name, input.value);
    shell.classList.remove("pe-number-editing");
    syncPanelNumberControl(input, w, name);
  };
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); commit(); input.blur(); }
    else if (event.key === "Escape") { event.preventDefault(); shell.classList.remove("pe-number-editing"); syncPanelNumberControl(input, w, name); input.blur(); }
    else if (event.key === "ArrowUp") { event.preventDefault(); nudge(1); }
    else if (event.key === "ArrowDown") { event.preventDefault(); nudge(-1); }
    else if (event.key === "PageUp") { event.preventDefault(); nudge(1, 10); }
    else if (event.key === "PageDown") { event.preventDefault(); nudge(-1, 10); }
  });

  scrub.addEventListener("pointerdown", (event) => {
    if (input.disabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const pointerId = event.pointerId;
    const startX = event.clientX;
    const startValue = Number(w.value) || 0;
    const step = panelNumericStep(w, name);
    let appliedSteps = 0;
    let moved = false;
    shell.classList.add("pe-number-scrubbing");
    try { scrub.setPointerCapture(pointerId); } catch (_) {}
    const move = (ev) => {
      if (ev.pointerId !== pointerId) return;
      const dx = ev.clientX - startX;
      const totalSteps = Math.trunc(dx / 10);
      if (totalSteps !== 0) moved = true;
      if (totalSteps === appliedSteps) return;
      appliedSteps = totalSteps;
      setPanelNativeNumericValue(node, name, startValue + totalSteps * step);
      ev.preventDefault();
      ev.stopPropagation();
    };
    const finish = (ev, cancelled = false) => {
      if (ev.pointerId !== pointerId) return;
      scrub.removeEventListener("pointermove", move);
      scrub.removeEventListener("pointerup", up);
      scrub.removeEventListener("pointercancel", cancel);
      shell.classList.remove("pe-number-scrubbing");
      try { scrub.releasePointerCapture(pointerId); } catch (_) {}
      if (!cancelled && !moved && Math.abs(ev.clientX - startX) < 4) setPanelNumberEditing(input, true);
      ev.preventDefault();
      ev.stopPropagation();
    };
    const up = (ev) => finish(ev, false);
    const cancel = (ev) => finish(ev, true);
    scrub.addEventListener("pointermove", move);
    scrub.addEventListener("pointerup", up);
    scrub.addEventListener("pointercancel", cancel);
  });

  syncPanelNumberControl(input, w, name);
  return { shell, input };
}


function createEnhanceBatchNumberControl(node) {
  const shell = panelElement("div", "pe-number pe-batch-number");
  shell.title = `Number of enhanced prompts to generate (${ENHANCE_BATCH_MIN}-${ENHANCE_BATCH_MAX})`;
  const fill = panelElement("div", "pe-number-fill");
  const dec = panelElement("button", "pe-number-step pe-number-dec");
  dec.type = "button";
  dec.setAttribute("aria-label", "Decrease enhancement batch count");
  dec.appendChild(panelNumberStepIcon("minus"));
  const center = panelElement("div", "pe-number-center");
  const input = panelElement("input", "pe-number-input");
  input.type = "text";
  input.inputMode = "numeric";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.setAttribute("role", "spinbutton");
  input.setAttribute("aria-label", "Enhancement batch count");
  input.setAttribute("aria-valuemin", String(ENHANCE_BATCH_MIN));
  input.setAttribute("aria-valuemax", String(ENHANCE_BATCH_MAX));
  const scrub = panelElement("div", "pe-number-scrub");
  scrub.setAttribute("aria-hidden", "true");
  const inc = panelElement("button", "pe-number-step pe-number-inc");
  inc.type = "button";
  inc.setAttribute("aria-label", "Increase enhancement batch count");
  inc.appendChild(panelNumberStepIcon("plus"));
  center.append(input, scrub);
  shell.append(fill, dec, center, inc);

  const sync = () => {
    const count = enhanceBatchCount(node);
    if (!(shell.classList.contains("pe-number-editing") && document.activeElement === input)) {
      input.value = String(count);
    }
    input.setAttribute("aria-valuenow", String(count));
    fill.style.display = "block";
    fill.style.width = `${((count - ENHANCE_BATCH_MIN) / (ENHANCE_BATCH_MAX - ENHANCE_BATCH_MIN)) * 100}%`;
    const disabled = !!node.__promptEnhancerBusy;
    input.disabled = disabled;
    shell.classList.toggle("pe-number-disabled", disabled);
    dec.disabled = disabled || count <= ENHANCE_BATCH_MIN;
    inc.disabled = disabled || count >= ENHANCE_BATCH_MAX;
  };

  const nudge = (direction, multiplier = 1) => {
    if (input.disabled) return;
    setEnhanceBatchCount(node, enhanceBatchCount(node) + direction * multiplier);
    sync();
  };
  dec.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); nudge(-1); });
  inc.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); nudge(1); });

  const startEditing = () => {
    if (input.disabled) return;
    shell.classList.add("pe-number-editing");
    input.value = String(enhanceBatchCount(node));
    requestAnimationFrame(() => {
      try { input.focus({ preventScroll: true }); input.select(); } catch (_) {}
    });
  };
  const commit = () => {
    if (!shell.classList.contains("pe-number-editing")) return;
    setEnhanceBatchCount(node, input.value);
    shell.classList.remove("pe-number-editing");
    sync();
  };
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); commit(); input.blur(); }
    else if (event.key === "Escape") { event.preventDefault(); shell.classList.remove("pe-number-editing"); sync(); input.blur(); }
    else if (event.key === "ArrowUp") { event.preventDefault(); nudge(1); }
    else if (event.key === "ArrowDown") { event.preventDefault(); nudge(-1); }
    else if (event.key === "PageUp") { event.preventDefault(); nudge(1, 10); }
    else if (event.key === "PageDown") { event.preventDefault(); nudge(-1, 10); }
  });

  scrub.addEventListener("pointerdown", (event) => {
    if (input.disabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const pointerId = event.pointerId;
    const startX = event.clientX;
    const startValue = enhanceBatchCount(node);
    let appliedSteps = 0;
    let moved = false;
    shell.classList.add("pe-number-scrubbing");
    try { scrub.setPointerCapture(pointerId); } catch (_) {}
    const move = (ev) => {
      if (ev.pointerId !== pointerId) return;
      // New number controls use the same fast scrub feel as the Local LLM panel:
      // approximately one configured step per horizontal pixel.
      const totalSteps = Math.trunc(ev.clientX - startX);
      if (totalSteps !== 0) moved = true;
      if (totalSteps === appliedSteps) return;
      appliedSteps = totalSteps;
      setEnhanceBatchCount(node, startValue + totalSteps);
      sync();
      ev.preventDefault();
      ev.stopPropagation();
    };
    const finish = (ev, cancelled = false) => {
      if (ev.pointerId !== pointerId) return;
      scrub.removeEventListener("pointermove", move);
      scrub.removeEventListener("pointerup", up);
      scrub.removeEventListener("pointercancel", cancel);
      shell.classList.remove("pe-number-scrubbing");
      try { scrub.releasePointerCapture(pointerId); } catch (_) {}
      if (!cancelled && !moved && Math.abs(ev.clientX - startX) < 4) startEditing();
      ev.preventDefault();
      ev.stopPropagation();
    };
    const up = (ev) => finish(ev, false);
    const cancel = (ev) => finish(ev, true);
    scrub.addEventListener("pointermove", move);
    scrub.addEventListener("pointerup", up);
    scrub.addEventListener("pointercancel", cancel);
  });

  sync();
  return { shell, input, sync };
}

function createHistoryIndexNumberControl(node) {
  const shell = panelElement("div", "pe-number pe-history-number");
  shell.title = "Active enhanced-prompt index";
  const fill = panelElement("div", "pe-number-fill");
  const dec = panelElement("button", "pe-number-step pe-number-dec");
  dec.type = "button";
  dec.setAttribute("aria-label", "Previous enhanced prompt");
  dec.appendChild(panelNumberStepIcon("minus"));
  const center = panelElement("div", "pe-number-center");
  const input = panelElement("input", "pe-number-input pe-history-number-input");
  input.type = "text";
  input.inputMode = "numeric";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.setAttribute("role", "spinbutton");
  input.setAttribute("aria-label", "Active enhanced-prompt index");
  const denominator = panelElement("span", "pe-history-denominator");
  denominator.textContent = "/ 0";
  denominator.setAttribute("aria-hidden", "true");
  const scrub = panelElement("div", "pe-number-scrub");
  scrub.setAttribute("aria-hidden", "true");
  const inc = panelElement("button", "pe-number-step pe-number-inc");
  inc.type = "button";
  inc.setAttribute("aria-label", "Next enhanced prompt");
  inc.appendChild(panelNumberStepIcon("plus"));
  center.append(input, denominator, scrub);
  shell.append(fill, dec, center, inc);

  const state = () => {
    const history = promptHistory(node);
    const total = history.length;
    const current = total ? promptHistoryIndex(node, history) + 1 : 0;
    return { history, total, current };
  };

  const setEditing = (editing) => {
    shell.classList.toggle("pe-number-editing", !!editing);
    if (editing) {
      const { current } = state();
      input.value = String(current);
      requestAnimationFrame(() => {
        try { input.focus({ preventScroll: true }); input.select(); } catch (_) {}
      });
    }
  };

  const sync = () => {
    const { total, current } = state();
    const busy = !!node.__promptEnhancerBusy;
    const hasHistory = total > 0;
    const editing = shell.classList.contains("pe-number-editing") && document.activeElement === input;
    if (!editing) input.value = String(current);
    denominator.textContent = `/ ${total}`;
    input.disabled = busy || !hasHistory;
    dec.disabled = busy || !hasHistory || current <= 1;
    inc.disabled = busy || !hasHistory || current >= total;
    shell.classList.toggle("pe-number-disabled", busy || !hasHistory);
    input.setAttribute("aria-valuenow", String(current));
    input.setAttribute("aria-valuemin", hasHistory ? "1" : "0");
    input.setAttribute("aria-valuemax", String(total));
    const pct = total > 1 ? Math.max(0, Math.min(100, ((current - 1) / (total - 1)) * 100)) : (total === 1 ? 100 : 0);
    fill.style.display = hasHistory ? "block" : "none";
    fill.style.width = `${pct}%`;
  };

  const commit = () => {
    if (!shell.classList.contains("pe-number-editing")) return;
    selectStoredPromptIndex(node, input.value);
    shell.classList.remove("pe-number-editing");
    sync();
  };

  const nudge = (direction, multiplier = 1) => {
    const { total, current } = state();
    if (node.__promptEnhancerBusy || !total) return;
    selectStoredPromptIndex(node, current + direction * multiplier);
  };

  dec.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); nudge(-1); });
  inc.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); nudge(1); });
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); commit(); input.blur(); }
    else if (event.key === "Escape") { event.preventDefault(); shell.classList.remove("pe-number-editing"); sync(); input.blur(); }
    else if (event.key === "ArrowUp") { event.preventDefault(); nudge(1); }
    else if (event.key === "ArrowDown") { event.preventDefault(); nudge(-1); }
    else if (event.key === "PageUp") { event.preventDefault(); nudge(1, 10); }
    else if (event.key === "PageDown") { event.preventDefault(); nudge(-1, 10); }
  });

  scrub.addEventListener("pointerdown", (event) => {
    if (input.disabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const pointerId = event.pointerId;
    const startX = event.clientX;
    const startCurrent = state().current;
    let appliedSteps = 0;
    let moved = false;
    shell.classList.add("pe-number-scrubbing");
    try { scrub.setPointerCapture(pointerId); } catch (_) {}
    const move = (ev) => {
      if (ev.pointerId !== pointerId) return;
      const dx = ev.clientX - startX;
      const totalSteps = Math.trunc(dx / 10);
      if (totalSteps !== 0) moved = true;
      if (totalSteps === appliedSteps) return;
      appliedSteps = totalSteps;
      selectStoredPromptIndex(node, startCurrent + totalSteps);
      ev.preventDefault();
      ev.stopPropagation();
    };
    const finish = (ev, cancelled = false) => {
      if (ev.pointerId !== pointerId) return;
      scrub.removeEventListener("pointermove", move);
      scrub.removeEventListener("pointerup", up);
      scrub.removeEventListener("pointercancel", cancel);
      shell.classList.remove("pe-number-scrubbing");
      try { scrub.releasePointerCapture(pointerId); } catch (_) {}
      if (!cancelled && !moved && Math.abs(ev.clientX - startX) < 4) setEditing(true);
      ev.preventDefault();
      ev.stopPropagation();
    };
    const up = (ev) => finish(ev, false);
    const cancel = (ev) => finish(ev, true);
    scrub.addEventListener("pointermove", move);
    scrub.addEventListener("pointerup", up);
    scrub.addEventListener("pointercancel", cancel);
  });

  sync();
  return { shell, input, denominator, dec, inc, sync };
}

function panelButton(label, title, callback) {
  const button = panelElement("button", "pe-button");
  button.type = "button";
  button.textContent = label;
  button.title = title || label;
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    callback?.();
  });
  return button;
}

function panelIconButton(kind, title, callback) {
  const button = panelButton("", title, callback);
  button.classList.add("pe-icon-button");
  button.setAttribute("aria-label", title);
  if (kind === "delete") button.classList.add("pe-delete");

  // Inline SVG keeps the icon crisp and independent of OS emoji/font support.
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  if (kind === "save") {
    svg.innerHTML = '<path d="M5 3h12l2 2v16H5z"/><path d="M8 3v6h8V3"/><path d="M8 21v-7h8v7"/>';
  } else {
    svg.innerHTML = '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M7 7l1 14h8l1-14"/><path d="M10 11v6M14 11v6"/>';
  }
  button.appendChild(svg);
  return button;
}

function panelFieldLabel(text) {
  const label = panelElement("div", "pe-field-label");
  label.textContent = text;
  return label;
}


function pinDomWidgetFullWidth(domWidget) {
  if (!domWidget || domWidget.__promptEnhancerFullWidthPinned) return;
  domWidget.__promptEnhancerFullWidthPinned = true;
  // Current classic ComfyUI can occasionally leave a stale pixel width on a
  // DOM widget. The renderer prefers widget.width over the live node width,
  // which lets the HTML panel spill outside a resized node. Keep width
  // undefined so both classic LiteGraph and Nodes 2.0 fall back to node width.
  try {
    Object.defineProperty(domWidget, "width", {
      configurable: true,
      enumerable: true,
      get() { return undefined; },
      set(_value) {},
    });
  } catch (_) {
    try { domWidget.width = undefined; } catch (_) {}
  }
}

function measureDomPanelNodeHeight(node, panelHeight) {
  const domWidget = node?.__promptEnhancerDomPanelWidget;
  const margin = Math.max(0, Number(domWidget?.margin ?? domWidget?.options?.margin ?? 4) || 0);
  const y = Number(domWidget?.y);
  let geometryHeight = 0;
  // In classic mode the DOM widget begins below the input/output socket rows.
  // node.computeSize() can under-count that top offset after the native widgets
  // are visually hidden, so include the actual laid-out widget y position.
  if (Number.isFinite(y) && y >= 0) {
    geometryHeight = Math.ceil(y + panelHeight + margin);
  }

  let computedHeight = 0;
  try {
    const required = node?.computeSize?.();
    computedHeight = Number(required?.[1]) || 0;
  } catch (_) {}

  return Math.max(120, geometryHeight, computedHeight);
}

function clampDomPanelToNode(node) {
  const root = node?.__promptEnhancerPanelRoot;
  const content = node?.__promptEnhancerPanelContent;
  if (!root || !content) return;
  // Prevent min-content sizing from pushing the HTML outside a narrow classic
  // node. The DOM-widget host itself is kept full-width by pinDomWidgetFullWidth.
  for (const el of [root, content]) {
    el.style.minWidth = "0";
    el.style.maxWidth = "100%";
  }
}

function createPromptEnhancerDomPanel(node) {
  if (!node) return null;
  if (node.__promptEnhancerDomPanelWidget?.node === node) return node.__promptEnhancerDomPanelWidget;
  try { node.__promptEnhancerPanelResizeObserver?.disconnect?.(); } catch (_) {}
  try { node.__promptEnhancerTextareaResizeObserver?.disconnect?.(); } catch (_) {}
  node.__promptEnhancerPanelResizeObserver = null;
  node.__promptEnhancerTextareaResizeObserver = null;
  node.__promptEnhancerDomPanelWidget = null;
  node.__promptEnhancerPanelControls = null;
  node.__promptEnhancerPanelRoot = null;
  node.__promptEnhancerPanelContent = null;
  node.__promptEnhancerPanelMeasuredHeight = 0;
  if (typeof node.addDOMWidget !== "function") return null;

  // A DOMWidget can be cloned by the frontend during duplicate/copy. Never use
  // a copied panel: it may share the source node's HTMLElement. Rebuild a fresh
  // panel for this node and leave all state in the real backend widgets.
  removeCopiedDomPanelWidget(node);

  const root = panelElement("div", "local-llm-prompt-enhancer-panel");
  root.dataset.nodeId = String(node.id ?? "");
  root.setAttribute("role", "group");
  root.setAttribute("aria-label", "Local LLM Prompt Enhancer controls");
  panelStyle(root, {
    width: "100%",
    minWidth: "0",
    maxWidth: "100%",
    boxSizing: "border-box",
    display: "block",
    position: "relative",
    padding: "2px 3px 8px",
    color: "inherit",
    fontFamily: "inherit",
    fontSize: "13px",
    lineHeight: "1.25",
    userSelect: "text",
    touchAction: "manipulation",
    overflow: "visible",
  });

  // Keep touch/mouse interaction inside the form instead of dragging the node.
  for (const eventName of ["pointerdown", "mousedown", "touchstart", "click", "dblclick", "contextmenu"]) {
    root.addEventListener(eventName, (event) => event.stopPropagation());
  }
  // Wheel events targeted at a DOM widget do not naturally reach LiteGraph's
  // canvas in Classic mode. Forward all non-scrollable wheel input explicitly;
  // overflowing textareas remain native scroll surfaces.
  root.addEventListener("wheel", (event) => panelHandleWheel(event, root), { passive: false });

  const css = panelElement("style");
  css.textContent = `
    .local-llm-prompt-enhancer-panel * { box-sizing: border-box; }
    .local-llm-prompt-enhancer-panel,
    .local-llm-prompt-enhancer-panel .pe-content,
    .local-llm-prompt-enhancer-panel .pe-row,
    .local-llm-prompt-enhancer-panel .pe-field,
    .local-llm-prompt-enhancer-panel .pe-selector-actions,
    .local-llm-prompt-enhancer-panel .pe-seed-grid,
    .local-llm-prompt-enhancer-panel .pe-boolean-field,
    .local-llm-prompt-enhancer-panel .pe-toggle-group { min-width: 0; max-width: 100%; }
    .local-llm-prompt-enhancer-panel .pe-wheel-capture { display: block; width: 100%; min-width: 0; max-width: 100%; outline: none; }
    .local-llm-prompt-enhancer-panel .pe-textarea {
      width: 100%; min-width: 0; max-width: 100%; color: inherit; font: inherit;
      background: rgba(127,127,127,.12);
      border: 1px solid rgba(127,127,127,.42);
      border-radius: 6px; outline: none;
    }
    .local-llm-prompt-enhancer-panel .pe-select {
      width: 100%; min-width: 0; max-width: 100%; font: inherit;
      color: #f3f4f6 !important;
      background-color: #27292d !important;
      border: 1px solid #555a62;
      border-radius: 6px; outline: none;
      color-scheme: dark;
    }
    .local-llm-prompt-enhancer-panel .pe-select option,
    .local-llm-prompt-enhancer-panel .pe-select optgroup {
      color: #f3f4f6 !important;
      background-color: #27292d !important;
    }
    .local-llm-prompt-enhancer-panel .pe-textarea {
      display: block; min-height: 108px; padding: 8px 9px;
      resize: vertical; line-height: 1.35;
    }
    .local-llm-prompt-enhancer-panel .pe-textarea:focus,
    .local-llm-prompt-enhancer-panel .pe-select:focus {
      border-color: rgba(120,170,255,.9);
      box-shadow: 0 0 0 1px rgba(120,170,255,.32);
    }
    .local-llm-prompt-enhancer-panel .pe-row {
      display: flex; flex-wrap: wrap; gap: 6px; width: 100%; align-items: center;
    }
    .local-llm-prompt-enhancer-panel .pe-button {
      min-height: 38px; min-width: 42px; padding: 6px 10px;
      border: 1px solid rgba(127,127,127,.42); border-radius: 6px;
      background: rgba(127,127,127,.18); color: inherit; font: inherit;
      font-weight: 600; cursor: pointer; touch-action: manipulation;
    }
    .local-llm-prompt-enhancer-panel .pe-button:hover:not(:disabled) { filter: brightness(1.12); }
    .local-llm-prompt-enhancer-panel .pe-button:active:not(:disabled) { transform: translateY(1px); }
    .local-llm-prompt-enhancer-panel .pe-button:disabled { opacity: .42; cursor: default; }
    .local-llm-prompt-enhancer-panel .pe-enhance { flex: 1 1 150px; color: white; }
    .local-llm-prompt-enhancer-panel .pe-batch-number { flex: 0 0 112px; width: 112px; min-width: 104px; }
    .local-llm-prompt-enhancer-panel .pe-batch-number .pe-number-step { flex-basis: 32px; width: 32px; }
    .local-llm-prompt-enhancer-panel .pe-grow { flex: 1 1 135px; }
    .local-llm-prompt-enhancer-panel .pe-select { min-height: 38px; padding: 5px 8px; }
    .local-llm-prompt-enhancer-panel .pe-selector-actions {
      display: grid; grid-template-columns: 42px minmax(0, 1fr) 42px;
      gap: 6px; align-items: center; min-width: 0; width: 100%;
    }
    .local-llm-prompt-enhancer-panel .pe-icon-button {
      width: 42px; min-width: 42px; padding: 0; display: inline-flex;
      align-items: center; justify-content: center;
    }
    .local-llm-prompt-enhancer-panel .pe-icon-button svg {
      width: 19px; height: 19px; pointer-events: none;
      stroke: currentColor; fill: none; stroke-width: 1.8;
      stroke-linecap: round; stroke-linejoin: round;
    }
    .local-llm-prompt-enhancer-panel .pe-icon-button.pe-delete:not(:disabled) { color: #ffb4b4; }
    .local-llm-prompt-enhancer-panel .pe-number {
      position: relative; display: flex; align-items: stretch; width: 100%; min-width: 0;
      height: 30px; overflow: hidden; border: 1px solid rgba(127,127,127,.38);
      border-radius: 10px; background: rgba(127,127,127,.14); color: inherit;
      font-variant-numeric: tabular-nums; isolation: isolate; user-select: none;
    }
    .local-llm-prompt-enhancer-panel .pe-number-fill {
      position: absolute; z-index: 0; inset: 0 auto 0 0; width: 0; pointer-events: none;
      background: rgba(71,133,181,.38); transition: width .06s linear;
    }
    .local-llm-prompt-enhancer-panel .pe-number-step {
      position: relative; z-index: 2; flex: 0 0 32px; width: 32px; height: 100%; padding: 0;
      border: 0; border-radius: 0; background: transparent; color: rgba(235,235,235,.6);
      display: flex; align-items: center; justify-content: center; cursor: pointer; touch-action: manipulation;
    }
    .local-llm-prompt-enhancer-panel .pe-number-step:hover:not(:disabled) { background: rgba(255,255,255,.07); color: rgba(255,255,255,.82); }
    .local-llm-prompt-enhancer-panel .pe-number-step:active:not(:disabled) { background: rgba(255,255,255,.11); }
    .local-llm-prompt-enhancer-panel .pe-number-step:disabled { opacity: .28; cursor: default; }
    .local-llm-prompt-enhancer-panel .pe-number-step svg {
      width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 2;
      stroke-linecap: round; stroke-linejoin: round; pointer-events: none;
    }
    .local-llm-prompt-enhancer-panel .pe-number-center { position: relative; z-index: 2; min-width: 4ch; flex: 1 1 auto; height: 100%; }
    .local-llm-prompt-enhancer-panel .pe-number-input {
      position: absolute; inset: 0; width: 100%; height: 100%; min-width: 0; border: 0; outline: 0;
      background: transparent; color: #f5f5f5; font: inherit; font-size: 15px; font-weight: 500;
      text-align: center; padding: 0 5px; font-variant-numeric: tabular-nums; user-select: text;
    }
    .local-llm-prompt-enhancer-panel .pe-number-input:focus { outline: 0; box-shadow: none; }
    .local-llm-prompt-enhancer-panel .pe-number-scrub { position: absolute; z-index: 3; inset: 0; cursor: ew-resize; touch-action: pan-y; }
    .local-llm-prompt-enhancer-panel .pe-number-editing .pe-number-scrub { display: none; }
    .local-llm-prompt-enhancer-panel .pe-number-editing { border-color: rgba(120,170,255,.9); box-shadow: 0 0 0 1px rgba(120,170,255,.28); }
    .local-llm-prompt-enhancer-panel .pe-number-scrubbing { cursor: ew-resize; box-shadow: inset 0 0 0 1px rgba(120,170,255,.22); }
    .local-llm-prompt-enhancer-panel .pe-number-scrubbing .pe-number-fill { transition: none; }
    .local-llm-prompt-enhancer-panel .pe-number-disabled { opacity: .48; }
    .local-llm-prompt-enhancer-panel .pe-number-disabled .pe-number-scrub { cursor: default; }
    .local-llm-prompt-enhancer-panel .pe-field { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px; align-items: center; width: 100%; }
    .local-llm-prompt-enhancer-panel .pe-field-label { font-weight: 600; opacity: .9; min-width: 0; white-space: nowrap; }
    .local-llm-prompt-enhancer-panel .pe-boolean-field { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px; align-items: center; width: 100%; }
    .local-llm-prompt-enhancer-panel .pe-toggle-group {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 2px;
      width: 100%; padding: 2px; border-radius: 7px;
      border: 1px solid rgba(127,127,127,.42); background: rgba(127,127,127,.10);
    }
    .local-llm-prompt-enhancer-panel .pe-toggle-option {
      min-width: 0; min-height: 36px; border: 0; border-radius: 5px;
      background: transparent; color: inherit; font: inherit; font-weight: 600;
      padding: 5px 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .local-llm-prompt-enhancer-panel .pe-toggle-option.pe-active {
      background: rgba(79,156,255,.32); box-shadow: inset 0 0 0 1px rgba(110,175,255,.55);
      color: #f8fbff;
    }
    .local-llm-prompt-enhancer-panel .pe-toggle-option:disabled { opacity: .42; cursor: default; }
    .local-llm-prompt-enhancer-panel .pe-history-number {
      flex: 0 1 150px; width: 150px; min-width: 126px;
    }
    .local-llm-prompt-enhancer-panel .pe-history-number .pe-number-center {
      display: flex; align-items: center; justify-content: center; gap: 4px; padding: 0 7px;
    }
    .local-llm-prompt-enhancer-panel .pe-history-number-input {
      position: relative; inset: auto; width: 4ch; height: 100%; flex: 0 1 4.5ch;
      padding: 0; text-align: right;
    }
    .local-llm-prompt-enhancer-panel .pe-history-denominator {
      position: relative; z-index: 2; flex: 0 0 auto; opacity: .72; white-space: nowrap;
      pointer-events: none; font-weight: 500; user-select: none;
    }
    .local-llm-prompt-enhancer-panel .pe-seed-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 7px; }
    .local-llm-prompt-enhancer-panel .pe-mini-label { display: block; font-size: 11px; font-weight: 600; opacity: .75; margin: 0 0 3px 2px; white-space: nowrap; }
    .local-llm-prompt-enhancer-panel .pe-seed-disabled { opacity: .48; }
    .local-llm-prompt-enhancer-panel .pe-badge { font-size: 11px; opacity: .75; margin-left: auto; }
    .local-llm-prompt-enhancer-panel .pe-count { min-width: 42px; text-align: center; opacity: .72; font-variant-numeric: tabular-nums; }
    @media (max-width: 520px) {
      .local-llm-prompt-enhancer-panel .pe-button { min-height: 42px; }
      .local-llm-prompt-enhancer-panel .pe-toggle-option { min-height: 40px; }
      .local-llm-prompt-enhancer-panel .pe-history-number { height: 34px; }
      .local-llm-prompt-enhancer-panel .pe-selector-actions { grid-template-columns: 44px minmax(0, 1fr) 44px; }
      .local-llm-prompt-enhancer-panel .pe-icon-button { width: 44px; min-width: 44px; }
      .local-llm-prompt-enhancer-panel .pe-select { min-height: 42px; }
      .local-llm-prompt-enhancer-panel .pe-number { height: 34px; }
      .local-llm-prompt-enhancer-panel .pe-number-step { flex-basis: 36px; width: 36px; }
      .local-llm-prompt-enhancer-panel .pe-batch-number { flex: 0 0 132px; width: 132px; min-width: 132px; }
      .local-llm-prompt-enhancer-panel .pe-batch-number .pe-number-step { flex-basis: 36px; width: 36px; }
      .local-llm-prompt-enhancer-panel .pe-field,
      .local-llm-prompt-enhancer-panel .pe-boolean-field { grid-template-columns: 1fr; gap: 4px; }
      .local-llm-prompt-enhancer-panel .pe-seed-grid { grid-template-columns: 1fr; }
      .local-llm-prompt-enhancer-panel .pe-textarea { min-height: 120px; }
    }
  `;
  root.appendChild(css);

  const controls = {};

  const promptPresetField = panelElement("div", "pe-field");
  promptPresetField.appendChild(panelFieldLabel("Prompt Preset"));
  const promptPresetSelector = panelElement("div", "pe-selector-actions");
  controls.savePromptPreset = panelIconButton("save", "Save Prompt Preset", () => void savePromptPreset(node));
  controls.promptPreset = panelElement("select", "pe-select");
  controls.promptPreset.addEventListener("change", () => setPanelNativeValue(node, "prompt_preset", controls.promptPreset.value));
  controls.deletePromptPreset = panelIconButton("delete", "Delete selected Prompt Preset", () => void deletePromptPreset(node));
  promptPresetSelector.append(controls.savePromptPreset, controls.promptPreset, controls.deletePromptPreset);
  promptPresetField.appendChild(promptPresetSelector);
  root.appendChild(promptPresetField);

  controls.prompt = panelElement("textarea", "pe-textarea");
  controls.prompt.placeholder = "prompt";
  controls.prompt.spellcheck = true;
  controls.prompt.title = widget(node, "prompt")?.options?.tooltip || "Prompt";
  controls.prompt.addEventListener("input", () => setPanelNativeValue(node, "prompt", controls.prompt.value));
  root.appendChild(panelWrapScrollableControl(controls.prompt));

  const enhanceRow = panelElement("div", "pe-row");
  controls.enhance = panelButton("Enhance Prompt", "Enhance the current prompt", () => toggleEnhance(node));
  controls.enhance.classList.add("pe-enhance");
  controls.batch = createEnhanceBatchNumberControl(node);
  controls.promote = panelButton("↑", "Use Enhanced Prompt as Prompt", () => useEnhancedAsPrompt(node));
  controls.undoPromotion = panelButton("Undo", "Undo Prompt replacement", () => undoPromptPromotion(node));
  enhanceRow.append(controls.enhance);
  if (controls.batch?.shell) enhanceRow.append(controls.batch.shell);
  enhanceRow.append(controls.promote, controls.undoPromotion);
  root.appendChild(enhanceRow);

  const overwriteField = panelElement("div", "pe-boolean-field");
  overwriteField.appendChild(panelFieldLabel("Overwrite Enhanced"));
  const overwriteToggle = panelElement("div", "pe-toggle-group");
  overwriteToggle.setAttribute("role", "group");
  overwriteToggle.setAttribute("aria-label", "Overwrite Enhanced");
  controls.overwriteOff = panelButton("add new", "Append a new enhanced prompt", () => setPanelNativeValue(node, "overwrite_enhanced", false));
  controls.overwriteOn = panelButton("overwrite active", "Replace the active enhanced prompt", () => setPanelNativeValue(node, "overwrite_enhanced", true));
  controls.overwriteOff.classList.add("pe-toggle-option");
  controls.overwriteOn.classList.add("pe-toggle-option");
  overwriteToggle.append(controls.overwriteOff, controls.overwriteOn);
  overwriteField.appendChild(overwriteToggle);
  root.appendChild(overwriteField);

  controls.enhanced = panelElement("textarea", "pe-textarea");
  controls.enhanced.placeholder = "enhanced_prompt";
  controls.enhanced.spellcheck = true;
  controls.enhanced.title = widget(node, "enhanced_prompt")?.options?.tooltip || "Enhanced Prompt";
  controls.enhanced.addEventListener("input", () => setPanelNativeValue(node, "enhanced_prompt", controls.enhanced.value));
  root.appendChild(panelWrapScrollableControl(controls.enhanced));

  const historyRow = panelElement("div", "pe-row");
  const historyNumber = createHistoryIndexNumberControl(node);
  controls.historyIndex = historyNumber?.input || null;
  controls.historyTotal = historyNumber?.denominator || null;
  controls.historyNumber = historyNumber || null;
  controls.deletePrompt = panelButton("×", "Delete active enhanced prompt", () => deleteActivePrompt(node));
  controls.undoArray = panelButton("Undo", "Undo enhanced-prompt array change (multi-level)", () => undoPromptArray(node));
  controls.redoArray = panelButton("Redo", "Redo enhanced-prompt array change (multi-level)", () => redoPromptArray(node));
  controls.clear = panelButton("Clear All", "Clear all enhanced prompts", () => clearPromptArray(node));
  if (historyNumber?.shell) historyRow.appendChild(historyNumber.shell);
  historyRow.append(controls.deletePrompt, controls.undoArray, controls.redoArray, controls.clear);
  root.appendChild(historyRow);

  const promptSetField = panelElement("div", "pe-field");
  promptSetField.appendChild(panelFieldLabel("Prompt Set"));
  const promptSetSelector = panelElement("div", "pe-selector-actions");
  controls.savePromptSet = panelIconButton("save", "Save Prompt Set", () => void savePromptSet(node));
  controls.promptSet = panelElement("select", "pe-select");
  controls.promptSet.addEventListener("change", () => setPanelNativeValue(node, "prompt_set", controls.promptSet.value));
  controls.deletePromptSet = panelIconButton("delete", "Delete selected Prompt Set", () => void deletePromptSet(node));
  promptSetSelector.append(controls.savePromptSet, controls.promptSet, controls.deletePromptSet);
  promptSetField.appendChild(promptSetSelector);
  root.appendChild(promptSetField);

  const cycleField = panelElement("div", "pe-field");
  cycleField.appendChild(panelFieldLabel("Prompt Cycle"));
  controls.promptCycle = panelElement("select", "pe-select");
  controls.promptCycle.addEventListener("change", () => setPanelNativeValue(node, "prompt_cycle", controls.promptCycle.value));
  cycleField.appendChild(controls.promptCycle);
  root.appendChild(cycleField);

  const presetField = panelElement("div", "pe-field");
  presetField.appendChild(panelFieldLabel("Enhancement Preset"));
  const presetSelector = panelElement("div", "pe-selector-actions");
  controls.saveTemplate = panelIconButton("save", "Save Enhancement Template", () => void saveTemplate(node));
  controls.preset = panelElement("select", "pe-select");
  controls.preset.addEventListener("change", () => setPanelNativeValue(node, "enhancement_preset", controls.preset.value));
  controls.deleteTemplate = panelIconButton("delete", "Delete selected user template", () => void deleteTemplate(node));
  presetSelector.append(controls.saveTemplate, controls.preset, controls.deleteTemplate);
  presetField.appendChild(presetSelector);
  root.appendChild(presetField);

  controls.instructions = panelElement("textarea", "pe-textarea");
  controls.instructions.placeholder = "enhancement_text";
  controls.instructions.spellcheck = true;
  controls.instructions.title = widget(node, "enhancement_text")?.options?.tooltip || "Enhancement Instructions";
  controls.instructions.addEventListener("input", () => setPanelNativeValue(node, "enhancement_text", controls.instructions.value));
  root.appendChild(panelWrapScrollableControl(controls.instructions));

  controls.seedWrap = panelElement("div");
  const seedGrid = panelElement("div", "pe-seed-grid");
  const seedValueBox = panelElement("div");
  const seedValueLabel = panelElement("label", "pe-mini-label");
  seedValueLabel.textContent = "Seed";
  const seedNumber = createPanelNumberControl(node, "seed", "Seed");
  controls.seed = seedNumber?.input || null;
  seedValueBox.append(seedValueLabel);
  if (seedNumber?.shell) seedValueBox.appendChild(seedNumber.shell);
  const seedControlBox = panelElement("div");
  const seedControlLabel = panelElement("label", "pe-mini-label");
  seedControlLabel.textContent = "Control After Generate";
  controls.seedControl = panelElement("select", "pe-select");
  controls.seedControl.title = "Control After Generate";
  controls.seedControl.addEventListener("change", () => {
    const w = seedControlWidget(node);
    if (!w) return;
    setWidgetValue(w, controls.seedControl.value, true);
    persistEnhancementState(node);
    markWorkflowChanged(node);
    scheduleDomPanelSync(node);
  });
  seedControlBox.append(seedControlLabel, controls.seedControl);
  seedGrid.append(seedValueBox, seedControlBox);
  controls.seedWrap.appendChild(seedGrid);
  root.appendChild(controls.seedWrap);

  const workflowField = panelElement("div", "pe-boolean-field");
  workflowField.appendChild(panelFieldLabel("Enhance with Workflow"));
  const workflowToggle = panelElement("div", "pe-toggle-group");
  workflowToggle.setAttribute("role", "group");
  workflowToggle.setAttribute("aria-label", "Enhance with Workflow");
  controls.enhanceWorkflowOff = panelButton("disabled", "Use the stored enhanced-prompt array during workflow execution", () => setPanelNativeValue(node, "enhance_with_workflow", false));
  controls.enhanceWorkflowOn = panelButton("enabled", "Generate a fresh enhanced prompt during workflow execution", () => setPanelNativeValue(node, "enhance_with_workflow", true));
  controls.enhanceWorkflowOff.classList.add("pe-toggle-option");
  controls.enhanceWorkflowOn.classList.add("pe-toggle-option");
  workflowToggle.append(controls.enhanceWorkflowOff, controls.enhanceWorkflowOn);
  workflowField.appendChild(workflowToggle);
  root.appendChild(workflowField);

  // Keep the DOM-widget host independent from its allocated height. ComfyUI
  // stretches the host element to the widget slot; measuring root.scrollHeight
  // therefore feeds the allocated height back into the requested height and can
  // cause endless growth. Measure this non-stretching content wrapper instead.
  const content = panelElement("div", "pe-content");
  panelStyle(content, {
    width: "100%",
    minWidth: "0",
    maxWidth: "100%",
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    gap: "7px",
    height: "auto",
    minHeight: "0",
    flex: "0 0 auto",
  });
  for (const child of Array.from(root.children)) {
    if (child === css) continue;
    content.appendChild(child);
  }
  root.appendChild(content);

  node.__promptEnhancerPanelControls = controls;
  node.__promptEnhancerPanelRoot = root;
  node.__promptEnhancerPanelContent = content;
  applyTextareaHeights(node, node.__promptEnhancerTextareaHeights || node?.properties?.[PERSISTENCE_STATE_KEY]?.textareaHeights, { schedule: false });

  const panelHeight = () => measureDomPanelHeight(node);
  const domWidget = node.addDOMWidget("prompt_enhancer_panel", "prompt_enhancer_panel", root, {
    serialize: false,
    hideOnZoom: false,
    margin: 4,
    getValue: () => "",
    setValue: () => {},
    getMinHeight: panelHeight,
    getHeight: panelHeight,
    afterResize: () => scheduleDomPanelHeight(node),
  });
  pinDomWidgetFullWidth(domWidget);
  clampDomPanelToNode(node);
  // UI-only: exclude the panel from both workflow widgets_values and API prompt
  // serialization. The real hidden native widgets remain the sole state owners.
  domWidget.serialize = false;
  domWidget.options ||= {};
  domWidget.options.serialize = false;
  node.__promptEnhancerDomPanelWidget = domWidget;

  hideNativeWidgetsForPanel(node);
  syncDomPanel(node);

  if (typeof ResizeObserver !== "undefined") {
    const observer = new ResizeObserver(() => scheduleDomPanelHeight(node));
    observer.observe(content);
    node.__promptEnhancerPanelResizeObserver = observer;

    const textareaObserver = new ResizeObserver(() => {
      captureTextareaHeights(node);
      controls.prompt.__promptEnhancerSyncWheelCapture?.();
      controls.enhanced.__promptEnhancerSyncWheelCapture?.();
      controls.instructions.__promptEnhancerSyncWheelCapture?.();
    });
    textareaObserver.observe(controls.prompt);
    textareaObserver.observe(controls.enhanced);
    textareaObserver.observe(controls.instructions);
    node.__promptEnhancerTextareaResizeObserver = textareaObserver;
  }
  scheduleDomPanelHeight(node);
  return domWidget;
}

function measureDomPanelHeight(node) {
  const root = node?.__promptEnhancerPanelRoot;
  const content = node?.__promptEnhancerPanelContent;
  if (!root || !content) return 650;

  // content.scrollHeight is intrinsic because .pe-content is never stretched to
  // the node/widget height. Root.scrollHeight must not be used here.
  let contentHeight = Number(content.scrollHeight) || Number(content.offsetHeight) || 0;
  let verticalPadding = 10;
  try {
    const styles = getComputedStyle(root);
    verticalPadding =
      (parseFloat(styles.paddingTop) || 0) +
      (parseFloat(styles.paddingBottom) || 0);
  } catch (_) {}
  if (contentHeight <= 0) return Number(node.__promptEnhancerPanelMeasuredHeight) || 650;
  return Math.max(320, Math.ceil(contentHeight + verticalPadding + 4));
}

function scheduleDomPanelHeight(node) {
  if (!node?.__promptEnhancerPanelContent || node.__promptEnhancerPanelHeightPending) return;
  node.__promptEnhancerPanelHeightPending = true;
  requestAnimationFrame(() => {
    node.__promptEnhancerPanelHeightPending = false;
    const height = measureDomPanelHeight(node);
    const previous = Number(node.__promptEnhancerPanelMeasuredHeight) || 0;
    const currentSizeForMeasure = copySize(node);
    const currentWidth = Number(currentSizeForMeasure?.[0]) || 0;
    const previousWidth = Number(node.__promptEnhancerPanelMeasuredNodeWidth) || 0;
    node.__promptEnhancerPanelMeasuredHeight = height;
    node.__promptEnhancerPanelMeasuredNodeWidth = currentWidth;

    const root = node.__promptEnhancerPanelRoot;
    if (root) root.style.setProperty("--comfy-widget-min-height", `${height}px`);
    const w = node.__promptEnhancerDomPanelWidget;
    if (w?.options) {
      w.options.getMinHeight = () => Number(node.__promptEnhancerPanelMeasuredHeight) || height;
      w.options.getHeight = () => Number(node.__promptEnhancerPanelMeasuredHeight) || height;
    }

    // Keep the node frame matched to the live DOM-panel geometry. In classic
    // mode the widget's y offset includes socket/header body space that
    // computeSize() may miss after the native widgets are hidden. In Nodes 2.0
    // this resolves to the same or smaller requirement, so the max is harmless.
    // Width remains user-controlled; the DOM widget itself is pinned to follow
    // the current node width instead of retaining a stale classic-mode width.
    clampDomPanelToNode(node);
    pinDomWidgetFullWidth(w);
    if ((Math.abs(height - previous) >= 1 || previous === 0 || Math.abs(currentWidth - previousWidth) >= 1) && !node.__promptEnhancerAutoSizing) {
      try {
        const current = copySize(node);
        if (current) {
          const targetHeight = measureDomPanelNodeHeight(node, height);
          if (Math.abs(current[1] - targetHeight) >= 2) {
            node.__promptEnhancerAutoSizing = true;
            node.setSize?.([current[0], targetHeight]);
            requestAnimationFrame(() => {
              node.__promptEnhancerAutoSizing = false;
              // One extra pass lets classic LiteGraph settle widget.y after the
              // resize without creating a recursive growth loop.
              scheduleDomPanelHeight(node);
            });
          }
        }
      } catch (_) {
        node.__promptEnhancerAutoSizing = false;
      }
    }
    redraw(node);
  });
}

function scheduleDomPanelSync(node) {
  if (!node?.__promptEnhancerPanelControls || node.__promptEnhancerPanelSyncPending) return;
  node.__promptEnhancerPanelSyncPending = true;
  queueMicrotask(() => {
    node.__promptEnhancerPanelSyncPending = false;
    syncDomPanel(node);
  });
}

function syncDomPanel(node) {
  const c = node?.__promptEnhancerPanelControls;
  if (!c) return;

  const prompt = String(widget(node, "prompt")?.value ?? "");
  const promptPresetWidget = widget(node, "prompt_preset");
  setPanelSelectOptions(c.promptPreset, panelComboValues(promptPresetWidget), promptPresetWidget?.value ?? "Custom");
  const enhanced = String(widget(node, "enhanced_prompt")?.value ?? "");
  const instructions = String(widget(node, "enhancement_text")?.value ?? "");
  if (c.prompt.value !== prompt) c.prompt.value = prompt;
  if (c.enhanced.value !== enhanced) c.enhanced.value = enhanced;
  if (c.instructions.value !== instructions) c.instructions.value = instructions;
  c.prompt.__promptEnhancerSyncWheelCapture?.();
  c.enhanced.__promptEnhancerSyncWheelCapture?.();
  c.instructions.__promptEnhancerSyncWheelCapture?.();

  const promptSetWidget = widget(node, "prompt_set");
  setPanelSelectOptions(c.promptSet, panelComboValues(promptSetWidget), promptSetWidget?.value ?? PROMPT_SET_NONE);
  const cycleWidget = widget(node, "prompt_cycle");
  setPanelSelectOptions(c.promptCycle, panelComboValues(cycleWidget), cycleWidget?.value ?? "fixed");
  const presetWidget = widget(node, "enhancement_preset");
  setPanelSelectOptions(c.preset, panelComboValues(presetWidget), presetWidget?.value ?? DEFAULT_PRESET);

  const batchCount = enhanceBatchCount(node);
  if (batchCount > 1) {
    enforceBatchAddNew(node, { mark: false });
    enforceBatchWorkflowDisabled(node, { mark: false });
  }
  c.batch?.sync?.();
  const overwriteLocked = batchCount > 1;
  const workflowLocked = batchCount > 1;
  const overwriteEnabled = !!widget(node, "overwrite_enhanced")?.value;
  c.overwriteOff.classList.toggle("pe-active", !overwriteEnabled);
  c.overwriteOn.classList.toggle("pe-active", overwriteEnabled);
  c.overwriteOff.setAttribute("aria-pressed", String(!overwriteEnabled));
  c.overwriteOn.setAttribute("aria-pressed", String(overwriteEnabled));
  const workflowEnabled = !!widget(node, "enhance_with_workflow")?.value;
  c.enhanceWorkflowOff.classList.toggle("pe-active", !workflowEnabled);
  c.enhanceWorkflowOn.classList.toggle("pe-active", workflowEnabled);
  c.enhanceWorkflowOff.setAttribute("aria-pressed", String(!workflowEnabled));
  c.enhanceWorkflowOn.setAttribute("aria-pressed", String(workflowEnabled));

  const seedWidget = widget(node, "seed");
  syncPanelNumberControl(c.seed, seedWidget, "seed");
  const seedControl = seedControlWidget(node);
  const controlValues = panelComboValues(seedControl);
  setPanelSelectOptions(c.seedControl, controlValues.length ? controlValues : ["fixed", "increment", "decrement", "randomize"], seedControl?.value ?? "fixed");

  setPanelNumberDisabled(c.seed, false);
  c.seedControl.disabled = false;
  c.seedWrap.classList.toggle("pe-seed-disabled", false);

  const busy = !!node.__promptEnhancerBusy;
  const enhancing = busy && node.__promptEnhancerActiveAction === "enhance";
  const history = promptHistory(node);
  const hasHistory = history.length > 0 || !!enhanced.trim();
  c.historyNumber?.sync?.();

  const batchProgress = node.__promptEnhancerBatchProgress;
  c.enhance.textContent = enhancing
    ? (batchProgress?.total > 1 ? `Cancel ${batchProgress.current}/${batchProgress.total}` : "Cancel")
    : "Enhance Prompt";
  c.enhance.style.background = enhancing ? "#e0b33f" : "#2f9e44";
  c.enhance.style.color = enhancing ? "#201a00" : "#ffffff";
  c.enhance.disabled = busy && !enhancing;

  c.promptPreset.disabled = busy;
  c.savePromptPreset.disabled = busy;
  const promptPresetName = String(promptPresetWidget?.value ?? "Custom");
  c.deletePromptPreset.disabled = busy || promptPresetName === "Custom" || !node.__promptEnhancerPromptPresetCatalog?.deletable?.has(promptPresetName);
  c.promote.disabled = busy || !hasHistory;
  c.undoPromotion.disabled = busy || !(node.__promptEnhancerPromptHistory || []).length;
  c.deletePrompt.disabled = busy || !hasHistory;
  c.undoArray.disabled = busy || !(node.__promptEnhancerArrayUndo || []).length;
  c.redoArray.disabled = busy || !(node.__promptEnhancerArrayRedo || []).length;
  c.clear.disabled = busy || !hasHistory;
  c.savePromptSet.disabled = busy || !hasHistory;

  const promptSetName = String(promptSetWidget?.value ?? PROMPT_SET_NONE).trim();
  c.deletePromptSet.disabled = busy || !promptSetName || promptSetName === PROMPT_SET_NONE;
  c.promptSet.disabled = busy;
  c.promptCycle.disabled = busy;
  c.overwriteOff.disabled = busy || overwriteLocked;
  c.overwriteOn.disabled = busy || overwriteLocked;
  const overwriteLockHint = overwriteLocked ? "Batch count above 1 requires Add New. Set the batch count back to 1 to unlock this control." : "";
  c.overwriteOff.title = overwriteLockHint || "Append a new enhanced prompt";
  c.overwriteOn.title = overwriteLockHint || "Replace the active enhanced prompt";
  c.preset.disabled = busy;
  c.instructions.disabled = busy;
  c.enhanceWorkflowOff.disabled = busy || workflowLocked;
  c.enhanceWorkflowOn.disabled = busy || workflowLocked;
  const workflowLockHint = workflowLocked ? "Batch count above 1 disables Enhance with Workflow. Set the batch count back to 1 to unlock this control." : "";
  c.enhanceWorkflowOff.title = workflowLockHint || "Use the stored enhanced-prompt array during workflow execution";
  c.enhanceWorkflowOn.title = workflowLockHint || "Generate a fresh enhanced prompt during workflow execution";
  c.saveTemplate.disabled = busy;

  const selected = selectedTemplate(node);
  c.deleteTemplate.disabled = busy || !selected || selected.source !== "user" || !!selected.protected;
  scheduleDomPanelHeight(node);
}

function wrapDomPanelSyncCallbacks(node) {
  if (!node || node.__promptEnhancerDomSyncWrapped) return;
  node.__promptEnhancerDomSyncWrapped = true;
  const widgets = [
    "prompt_preset", "prompt", "enhanced_prompt", "prompt_set", "prompt_cycle", "overwrite_enhanced",
    "enhancement_preset", "enhancement_text", "seed", "enhance_with_workflow",
    "prompt_history_json", "prompt_history_index", "prompt_shuffle_json",
  ].map((name) => widget(node, name)).filter(Boolean);
  const control = seedControlWidget(node);
  if (control) widgets.push(control);
  for (const w of widgets) {
    if (w.__promptEnhancerDomSyncCallbackWrapped) continue;
    w.__promptEnhancerDomSyncCallbackWrapped = true;
    const original = w.callback;
    w.callback = function (...args) {
      const result = original?.apply(this, args);
      scheduleDomPanelSync(node);
      return result;
    };
  }
}

function wrapNodeExecuted(node) {
  // Prompt Enhancer execution mirroring is intentionally handled by the global
  // API `executed` listener below. Tying state changes to node.onExecuted made
  // Prompt Cycle depend on renderer lifecycle/mount state in some Nodes 2.0
  // situations (most visibly when the node was not selected). Keep this helper
  // as a no-op marker for hot-reload/backward compatibility.
  if (!node || node.__promptEnhancerExecutedWrapped) return;
  node.__promptEnhancerExecutedWrapped = true;
}

function executionGraphNodeById(graph, rawId, seen = new Set()) {
  if (!graph || rawId == null || seen.has(graph)) return null;
  seen.add(graph);
  const ids = [rawId, String(rawId)];
  const numeric = Number(rawId);
  if (Number.isFinite(numeric)) ids.push(numeric);
  for (const id of ids) {
    try {
      const direct = graph.getNodeById?.(id);
      if (direct) return direct;
    } catch (_) {}
  }
  for (const candidate of graph._nodes || []) {
    if (!candidate) continue;
    if (String(candidate.id) === String(rawId)) return candidate;
    const subgraph = candidate.subgraph || candidate.graph?.subgraph;
    const nested = executionGraphNodeById(subgraph, rawId, seen);
    if (nested) return nested;
  }
  return null;
}

function promptEnhancerNodeForExecutedDetail(detail) {
  const ids = [detail?.display_node, detail?.node].filter((value) => value != null);
  const roots = [app?.rootGraph, app?.graph].filter(Boolean);
  for (const id of ids) {
    for (const graph of roots) {
      const node = executionGraphNodeById(graph, id);
      if (node?.comfyClass === NODE_CLASS) return node;
    }
  }
  return null;
}

function handlePromptEnhancerExecutedEvent(detail) {
  const node = promptEnhancerNodeForExecutedDetail(detail);
  if (!node) return;
  try {
    handleEnhanceExecuted(node, detail?.output);
  } catch (error) {
    console.error("[Local LLM Prompt Enhancer] global execution sync failed", error);
  }
}

function moveSettingsInputToTop(node) {
  const inputs = node?.inputs;
  if (!Array.isArray(inputs)) return;
  const index = inputs.findIndex((input) => input?.name === "settings");
  if (index < 0) return;
  if (index > 0) {
    const [settingsInput] = inputs.splice(index, 1);
    inputs.unshift(settingsInput);
  }

  // LiteGraph links target numeric input slots. Keep every existing link aimed
  // at the same named input after changing only the visible socket order.
  const graph = node?.graph || app?.graph;
  inputs.forEach((input, slot) => {
    const linkIds = [];
    if (input?.link != null) linkIds.push(input.link);
    if (Array.isArray(input?.links)) linkIds.push(...input.links);
    for (const id of new Set(linkIds)) {
      const link = graph?.links?.[id] || graph?.links?.get?.(id);
      if (!link) continue;
      link.target_slot = slot;
      if ("targetSlot" in link) link.targetSlot = slot;
    }
  });
}

function installControls(node, isNew = false, preservedSize = null) {
  if (!node || node.__promptEnhancerInstalled) {
    if (preservedSize) restoreSize(node, preservedSize);
    return;
  }
  node.__promptEnhancerInstalled = true;
  moveSettingsInputToTop(node);
  node.__promptEnhancerPromptHistory ||= [];
  node.__promptEnhancerArrayUndo ||= [];
  node.__promptEnhancerBatchCount = normalizeEnhanceBatchCount(
    node?.properties?.[PERSISTENCE_STATE_KEY]?.batchCount ?? node.__promptEnhancerBatchCount ?? ENHANCE_BATCH_MIN
  );

  const promptSetWidget = widget(node, "prompt_set");
  const presetWidget = widget(node, "enhancement_preset");

  setWidgetDisplayLabel(widget(node, "prompt_preset"), "Prompt Preset");
  setWidgetDisplayLabel(promptSetWidget, "Prompt Set");
  setWidgetDisplayLabel(widget(node, "prompt_cycle"), "Prompt Cycle");
  setWidgetDisplayLabel(widget(node, "overwrite_enhanced"), "Overwrite Enhanced");
  setWidgetDisplayLabel(presetWidget, "Enhancement Preset");
  setWidgetDisplayLabel(widget(node, "seed"), "Seed");
  setWidgetDisplayLabel(widget(node, "enhance_with_workflow"), "Enhance with Workflow");
  const seedWidget = widget(node, "seed");
  for (const linked of seedWidget?.linkedWidgets || []) {
    if (String(linked?.name || "").toLowerCase().includes("control")) {
      setWidgetDisplayLabel(linked, "Control After Generate");
    }
  }
  for (const name of ["prompt_history_json", "prompt_history_index", "prompt_shuffle_json"]) {
    hideInternalWidget(widget(node, name));
  }

  wrapNodeSerialization(node);
  wrapPromptSetCallback(node);
  wrapPresetCallback(node);
  wrapPromptPresetCallbacks(node);
  wrapEnhancementTextPersistence(node);
  wrapEnhancedPromptHistory(node);
  for (const name of [
    "prompt_cycle",
    "overwrite_enhanced",
    "seed",
    "enhance_with_workflow",
  ]) {
    wrapSimpleStatePersistence(node, name);
  }
  wrapSeedControlPersistence(node);

  const panel = createPromptEnhancerDomPanel(node);
  if (panel) {
    hideNativeWidgetsForPanel(node);
    wrapDomPanelSyncCallbacks(node);
    requestAnimationFrame(() => {
      hideNativeWidgetsForPanel(node);
      wrapDomPanelSyncCallbacks(node);
      syncDomPanel(node);
      scheduleDomPanelHeight(node);
    });
  }

  // Execution-result mirroring is secondary to mounting the visible panel.
  // Install it only after createPromptEnhancerDomPanel() has run so a hook
  // regression can never prevent the node UI from appearing.
  wrapNodeExecuted(node);

  // Attach Settings connection UI only after the DOM panel exists. This keeps
  // renderer-specific native widget behavior from blocking panel creation.
  wrapSettingsConnectionUI(node);

  if (preservedSize) {
    restoreLoadedWidthAndAutosize(node, preservedSize);
  } else if (isNew) {
    try {
      const computed = node.computeSize?.();
      if (computed?.length >= 2) node.setSize?.([Math.max(440, computed[0]), computed[1]]);
    } catch (_) {}
  }
  redraw(node);
}

function promptEnhancerMenuItems(node) {
  if (!node || node.comfyClass !== NODE_CLASS) return [];

  const busy = !!node.__promptEnhancerBusy;
  const enhancing = busy && node.__promptEnhancerActiveAction === "enhance";
  const history = promptHistory(node);
  const hasHistory = history.length > 0 || !!String(widget(node, "enhanced_prompt")?.value ?? "").trim();
  const canArrayUndo = (node.__promptEnhancerArrayUndo || []).length > 0;
  const canPromptUndo = (node.__promptEnhancerPromptHistory || []).length > 0;
  const promptSetName = String(widget(node, "prompt_set")?.value ?? PROMPT_SET_NONE).trim();
  const canDeleteSet = !!promptSetName && promptSetName !== PROMPT_SET_NONE;
  const selected = selectedTemplate(node);
  const canDeleteTemplate = !!selected && selected.source === "user" && !selected.protected;

  const disabledWhileBusy = busy;
  return [
    null,
    {
      content: enhancing ? "Prompt Enhancer: Cancel Enhancement" : "Prompt Enhancer: Enhance Prompt",
      disabled: busy && !enhancing,
      callback: () => toggleEnhance(node),
    },
    {
      content: "Prompt Enhancer: Use Enhanced as Prompt (↑)",
      disabled: disabledWhileBusy || !hasHistory,
      callback: () => useEnhancedAsPrompt(node),
    },
    {
      content: "Prompt Enhancer: Undo Prompt Replacement",
      disabled: disabledWhileBusy || !canPromptUndo,
      callback: () => undoPromptPromotion(node),
    },
    null,
    {
      content: "Enhanced Prompts: Previous",
      disabled: disabledWhileBusy || !hasHistory,
      callback: () => cycleStoredPrompt(node, -1),
    },
    {
      content: "Enhanced Prompts: Next",
      disabled: disabledWhileBusy || !hasHistory,
      callback: () => cycleStoredPrompt(node, 1),
    },
    {
      content: "Enhanced Prompts: Delete Active",
      disabled: disabledWhileBusy || !hasHistory,
      callback: () => deleteActivePrompt(node),
    },
    {
      content: "Enhanced Prompts: Undo Array Change",
      disabled: disabledWhileBusy || !canArrayUndo,
      callback: () => undoPromptArray(node),
    },
    {
      content: "Enhanced Prompts: Clear All",
      disabled: disabledWhileBusy || !hasHistory,
      callback: () => clearPromptArray(node),
    },
    null,
    {
      content: "Prompt Set: Save Current",
      disabled: disabledWhileBusy || !hasHistory,
      callback: () => void savePromptSet(node),
    },
    {
      content: "Prompt Set: Delete Selected",
      disabled: disabledWhileBusy || !canDeleteSet,
      callback: () => void deletePromptSet(node),
    },
    {
      content: "Enhancement Template: Save Instructions",
      disabled: disabledWhileBusy,
      callback: () => void saveTemplate(node),
    },
    {
      content: "Enhancement Template: Delete Selected",
      disabled: disabledWhileBusy || !canDeleteTemplate,
      callback: () => void deleteTemplate(node),
    },
  ];
}

async function initializeNode(node, loaded = false, serializedSize = null) {
  const preservedSize = loaded ? (serializedSize || copySize(node)) : null;
  applyPromptEnhancerInputLabels(node);
  installControls(node, !loaded, preservedSize);

  // Restore explicit node state for both loaded workflows and copied/pasted or
  // duplicated nodes. A genuinely new node has no state object and simply keeps
  // the backend-provided defaults already present in its widgets.
  const restored = restoreEnhancementState(node);
  const instructions = widget(node, "enhancement_text");
  const shouldLoad = !restored && !loaded && !String(instructions?.value ?? "").trim();

  // Refresh only the available choices. Never reload a selected template or
  // Prompt Set during initialization, because that would overwrite the exact
  // serialized working values after a tab change, paste, duplicate, or refresh.
  await refreshTemplates(node, { loadSelected: shouldLoad });
  await refreshPromptPresets(node);
  await refreshPromptSets(node);
  initializePromptHistory(node);
  enforceBatchAddNew(node, { mark: false });
  persistEnhancementState(node);
  setSeedOverrideUI(node);
  scheduleDomPanelSync(node);
  void syncPromptCycleFromBackend(node, { force: true });
  requestAnimationFrame(() => {
    moveSettingsInputToTop(node);
    setSeedOverrideUI(node);
    hideNativeWidgetsForPanel(node);
    wrapDomPanelSyncCallbacks(node);
    syncDomPanel(node);
    scheduleDomPanelHeight(node);
  });
  if (preservedSize) restoreLoadedWidthAndAutosize(node, preservedSize);
}

function failPendingNodes(detail) {
  if (!pendingNodes.size) return;
  const message = detail?.exception_message || detail?.error?.message || "Partial execution failed before enhancement completed.";
  for (const node of [...pendingNodes.values()]) {
    const token = node.__promptEnhancerPendingToken;
    if (token) cancelArmed(node, token);
    clearEnhancePending(node);
  }
  notify("error", "Local LLM Prompt Enhancer", String(message));
}

app.registerExtension({
  name: EXTENSION_NAME,
  setup() {
    api.addEventListener?.("executed", ({ detail }) => handlePromptEnhancerExecutedEvent(detail));
    api.addEventListener?.("execution_error", ({ detail }) => failPendingNodes(detail));
    api.addEventListener?.("execution_interrupted", ({ detail }) => failPendingNodes(detail));

    // Browser focus/visibility recovery is still useful when the whole browser
    // tab was backgrounded. Internal ComfyUI workflow tabs are handled by the
    // official afterConfigureGraph lifecycle hook below, after app.graph is
    // actually the selected workflow.
    const reconcileBrowserActivation = () => reconcilePromptCyclesAfterGraphActivation();
    window.addEventListener("focus", reconcileBrowserActivation, { passive: true });
    window.addEventListener("pageshow", reconcileBrowserActivation, { passive: true });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") reconcileBrowserActivation();
    }, { passive: true });
  },
  beforeConfigureGraph() {
    graphConfigureDepth += 1;
  },
  afterConfigureGraph() {
    graphConfigureDepth = Math.max(0, graphConfigureDepth - 1);
    if (graphConfigureDepth === 0) {
      // This fires after ComfyUI has loaded/switched the workflow graph. Sync
      // immediately from the backend cursor so X / Y is correct on return,
      // without waiting for another generation to finish.
      reconcilePromptCyclesAfterGraphActivation();
    }
  },
  nodeCreated(node) {
    if (node?.comfyClass !== NODE_CLASS) return;
    if (graphConfigureDepth > 0) return;
    setTimeout(() => initializeNode(node, false), 0);
  },
  getNodeMenuItems(node) {
    return promptEnhancerMenuItems(node);
  },
  loadedGraphNode(node) {
    if (node?.comfyClass !== NODE_CLASS) return;
    const serializedSize = copySize(node);
    setTimeout(() => initializeNode(node, true, serializedSize), 0);
  },
});
