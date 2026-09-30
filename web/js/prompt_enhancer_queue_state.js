/**
 * Prompt Enhancer queue/API serialization helpers.
 *
 * Dependency-free so browser queue semantics can be regression-tested under
 * Node without loading ComfyUI.
 */

export function normalizeWidgetBoolean(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) && value !== 0;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["true", "1", "yes", "on", "enabled"].includes(normalized)) return true;
    if (["false", "0", "no", "off", "disabled", ""].includes(normalized)) return false;
  }
  return !!value;
}

export function serializeEnhancementSeed({ enhanceWithWorkflow = false, seed = 0 } = {}) {
  if (!normalizeWidgetBoolean(enhanceWithWorkflow)) return 0;
  return seed ?? 0;
}

export function shouldAdvanceEnhancementSeed({ enhanceWithWorkflow = false, manual = false } = {}) {
  return !!manual || normalizeWidgetBoolean(enhanceWithWorkflow);
}

export function nextPromptCycleQueueSequence(current = 0) {
  const raw = Number(current);
  const normalized = Number.isFinite(raw) ? Math.max(0, Math.trunc(raw)) : 0;
  return normalized >= 0x7ffffffe ? 1 : normalized + 1;
}


function normalizeCycleMode(value) {
  const mode = String(value ?? "fixed").trim().toLowerCase();
  return ["fixed", "increment", "decrement", "shuffle", "random"].includes(mode) ? mode : "fixed";
}

function normalizeHistoryIndex(value, count) {
  if (!Number.isFinite(Number(value)) || count <= 0) return 0;
  return Math.max(0, Math.min(Math.trunc(Number(value)), count - 1));
}

function shuffled(values, random) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const r = Math.max(0, Math.min(0.999999999999, Number(random?.() ?? Math.random())));
    const j = Math.floor(r * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Plan exactly one queued Prompt Cycle item.
 *
 * The returned selectedIndex is the value that must be serialized into
 * prompt_history_index for THIS API prompt. The returned state owns only future
 * queue serialization; it never depends on backend execution timing.
 */
export function planPromptCycleQueueItem(previousState, {
  mode = "fixed",
  history = [],
  currentIndex = 0,
  revision = 0,
  random = Math.random,
} = {}) {
  const items = Array.isArray(history) ? history.map((item) => String(item ?? "")) : [];
  const count = items.length;
  const normalizedMode = normalizeCycleMode(mode);
  const requested = normalizeHistoryIndex(currentIndex, count);
  const rev = Number.isFinite(Number(revision)) ? Math.max(0, Math.trunc(Number(revision))) : 0;
  const signature = JSON.stringify([normalizedMode, rev, items]);

  if (normalizedMode === "fixed" || count <= 0) {
    return {
      selectedIndex: requested,
      nextIndex: requested,
      shuffle: [],
      state: null,
      signature,
    };
  }

  const matches = previousState && previousState.signature === signature;
  let selectedIndex = matches
    ? normalizeHistoryIndex(previousState.nextIndex, count)
    : requested;
  let bag = matches && Array.isArray(previousState.shuffle)
    ? previousState.shuffle.map(Number).filter((v, i, a) => Number.isInteger(v) && v >= 0 && v < count && v !== selectedIndex && a.indexOf(v) === i)
    : [];
  let nextIndex = selectedIndex;

  if (count <= 1) {
    nextIndex = selectedIndex;
    bag = [];
  } else if (normalizedMode === "increment") {
    nextIndex = (selectedIndex + 1) % count;
    bag = [];
  } else if (normalizedMode === "decrement") {
    nextIndex = (selectedIndex - 1 + count) % count;
    bag = [];
  } else if (normalizedMode === "random") {
    const r = Math.max(0, Math.min(0.999999999999, Number(random?.() ?? Math.random())));
    nextIndex = Math.floor(r * count);
    bag = [];
  } else if (normalizedMode === "shuffle") {
    if (!bag.length) {
      if (matches) {
        // New deck: all cards are eligible, but avoid an immediate repeat at
        // the deck boundary when possible.
        bag = shuffled([...Array(count).keys()], random);
        if (bag[0] === selectedIndex) {
          const swap = bag.findIndex((value, index) => index > 0 && value !== selectedIndex);
          if (swap > 0) [bag[0], bag[swap]] = [bag[swap], bag[0]];
        }
      } else {
        // The user-visible X/Y selection is the first card of the first deck.
        bag = shuffled([...Array(count).keys()].filter((value) => value !== selectedIndex), random);
      }
    }
    nextIndex = bag.length ? bag.shift() : selectedIndex;
  }

  const state = {
    signature,
    nextIndex,
    shuffle: bag,
  };
  return {
    selectedIndex,
    nextIndex,
    shuffle: [...bag],
    state,
    signature,
  };
}
