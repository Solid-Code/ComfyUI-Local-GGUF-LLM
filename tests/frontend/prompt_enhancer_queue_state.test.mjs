import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeWidgetBoolean,
  serializeEnhancementSeed,
  shouldAdvanceEnhancementSeed,
} from "../../web/js/prompt_enhancer_queue_state.js";

test("widget boolean normalization treats string false as false", () => {
  assert.equal(normalizeWidgetBoolean(false), false);
  assert.equal(normalizeWidgetBoolean("false"), false);
  assert.equal(normalizeWidgetBoolean("0"), false);
  assert.equal(normalizeWidgetBoolean("off"), false);
  assert.equal(normalizeWidgetBoolean(true), true);
  assert.equal(normalizeWidgetBoolean("true"), true);
  assert.equal(normalizeWidgetBoolean("1"), true);
});

test("passive workflow execution neutralizes the enhancer seed", () => {
  assert.equal(serializeEnhancementSeed({ enhanceWithWorkflow: false, seed: 1234 }), 0);
  assert.equal(serializeEnhancementSeed({ enhanceWithWorkflow: "false", seed: 1234 }), 0);
});

test("workflow enhancement preserves the real enhancer seed", () => {
  assert.equal(serializeEnhancementSeed({ enhanceWithWorkflow: true, seed: 1234 }), 1234);
});

test("enhancer seed control advances only for manual or workflow enhancement", () => {
  assert.equal(shouldAdvanceEnhancementSeed({ enhanceWithWorkflow: false, manual: false }), false);
  assert.equal(shouldAdvanceEnhancementSeed({ enhanceWithWorkflow: true, manual: false }), true);
  assert.equal(shouldAdvanceEnhancementSeed({ enhanceWithWorkflow: false, manual: true }), true);
});
