import test from "node:test";
import assert from "node:assert/strict";

import {
  nextPromptCycleQueueSequence,
  planPromptCycleQueueItem,
  normalizeWidgetBoolean,
  serializeEnhancementSeed,
  shouldAdvanceEnhancementSeed,
} from "../../web/js/prompt_enhancer_queue_state.js";

test("widget boolean normalization treats string false as false", () => {
  for (const value of [false, 0, "false", "0", "off", "disabled", ""]) {
    assert.equal(normalizeWidgetBoolean(value), false, String(value));
  }
  for (const value of [true, 1, "true", "1", "on", "enabled"]) {
    assert.equal(normalizeWidgetBoolean(value), true, String(value));
  }
});

test("passive workflow execution neutralizes the enhancer seed", () => {
  assert.equal(serializeEnhancementSeed({ enhanceWithWorkflow: false, seed: 123456 }), 0);
  assert.equal(serializeEnhancementSeed({ enhanceWithWorkflow: "false", seed: 123456 }), 0);
});

test("workflow enhancement preserves the real enhancer seed", () => {
  assert.equal(serializeEnhancementSeed({ enhanceWithWorkflow: true, seed: 123456 }), 123456);
  assert.equal(serializeEnhancementSeed({ enhanceWithWorkflow: "true", seed: 123456 }), 123456);
});

test("enhancer seed control advances only for manual or workflow enhancement", () => {
  assert.equal(shouldAdvanceEnhancementSeed({ enhanceWithWorkflow: false, manual: false }), false);
  assert.equal(shouldAdvanceEnhancementSeed({ enhanceWithWorkflow: "false", manual: false }), false);
  assert.equal(shouldAdvanceEnhancementSeed({ enhanceWithWorkflow: true, manual: false }), true);
  assert.equal(shouldAdvanceEnhancementSeed({ enhanceWithWorkflow: false, manual: true }), true);
});

test("Prompt Cycle queue sequence advances once per serialized item and wraps safely", () => {
  let sequence = 0;
  const values = [];
  for (let i = 0; i < 8; i += 1) {
    sequence = nextPromptCycleQueueSequence(sequence);
    values.push(sequence);
  }
  assert.deepEqual(values, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(nextPromptCycleQueueSequence(0x7ffffffe), 1);
  assert.equal(nextPromptCycleQueueSequence(null), 1);
  assert.equal(nextPromptCycleQueueSequence("bad"), 1);
});


test("increment queue planning serializes every Run x N item explicitly", () => {
  let state = null;
  const selected = [];
  for (let i = 0; i < 8; i += 1) {
    const plan = planPromptCycleQueueItem(state, {
      mode: "increment",
      history: ["A", "B", "C"],
      currentIndex: 0,
      revision: 1,
    });
    selected.push(plan.selectedIndex);
    state = plan.state;
  }
  assert.deepEqual(selected, [0, 1, 2, 0, 1, 2, 0, 1]);
});

test("decrement queue planning wraps without backend state", () => {
  let state = null;
  const selected = [];
  for (let i = 0; i < 6; i += 1) {
    const plan = planPromptCycleQueueItem(state, {
      mode: "decrement",
      history: ["A", "B", "C"],
      currentIndex: 1,
      revision: 4,
    });
    selected.push(plan.selectedIndex);
    state = plan.state;
  }
  assert.deepEqual(selected, [1, 0, 2, 1, 0, 2]);
});

test("revision change reanchors queue planning to visible X/Y", () => {
  let state = null;
  let plan = planPromptCycleQueueItem(state, {
    mode: "increment", history: ["A", "B", "C"], currentIndex: 0, revision: 1,
  });
  state = plan.state;
  plan = planPromptCycleQueueItem(state, {
    mode: "increment", history: ["A", "B", "C"], currentIndex: 0, revision: 1,
  });
  assert.equal(plan.selectedIndex, 1);
  plan = planPromptCycleQueueItem(plan.state, {
    mode: "increment", history: ["A", "B", "C"], currentIndex: 2, revision: 2,
  });
  assert.equal(plan.selectedIndex, 2);
});

test("shuffle queue planning consumes a full deck before repeating", () => {
  const values = [0.9, 0.2, 0.7, 0.1, 0.8, 0.3, 0.6, 0.4];
  let n = 0;
  const random = () => values[(n++) % values.length];
  let state = null;
  const selected = [];
  for (let i = 0; i < 6; i += 1) {
    const plan = planPromptCycleQueueItem(state, {
      mode: "shuffle",
      history: ["A", "B", "C"],
      currentIndex: 0,
      revision: 1,
      random,
    });
    selected.push(plan.selectedIndex);
    state = plan.state;
  }
  assert.equal(new Set(selected.slice(0, 3)).size, 3);
  assert.equal(new Set(selected.slice(3, 6)).size, 3);
  assert.notEqual(selected[2], selected[3]);
});

test("random queue planning freezes the chosen index per queued item", () => {
  const values = [0.0, 0.99, 0.34];
  let n = 0;
  const random = () => values[(n++) % values.length];
  let state = null;
  const selected = [];
  for (let i = 0; i < 4; i += 1) {
    const plan = planPromptCycleQueueItem(state, {
      mode: "random",
      history: ["A", "B", "C"],
      currentIndex: 1,
      revision: 1,
      random,
    });
    selected.push(plan.selectedIndex);
    state = plan.state;
  }
  // First item is the visible selection; following items use the random nexts.
  assert.deepEqual(selected, [1, 0, 2, 1]);
});
