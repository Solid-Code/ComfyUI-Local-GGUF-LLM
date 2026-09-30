import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const frontend = fs.readFileSync(path.join(root, "web/js/prompt_enhancer_dom_v0651.js"), "utf8");
const helper = fs.readFileSync(path.join(root, "web/js/prompt_enhancer_input_slots.js"), "utf8");

test("Prompt Enhancer frontend never mutates input topology", () => {
  const source = `${frontend}\n${helper}`;
  const forbidden = [
    /\.inputs\s*\.\s*splice\s*\(/,
    /\.inputs\s*\.\s*unshift\s*\(/,
    /\.inputs\s*\.\s*push\s*\(/,
    /\.inputs\s*\.\s*sort\s*\(/,
    /\.inputs\s*\.\s*reverse\s*\(/,
    /\.target_slot\s*=/,
    /\.targetSlot\s*=/,
  ];
  for (const pattern of forbidden) assert.equal(pattern.test(source), false, String(pattern));
});

test("legacy asynchronous cycle reset request is absent", () => {
  assert.equal(frontend.includes("/local_llm_prompt_enhancer/cycle_reset"), false);
});

test("normal frontend queueing has no backend cycle sync authority", () => {
  assert.equal(frontend.includes("syncPromptCycleFromBackend"), false);
  assert.equal(frontend.includes("/local_llm_prompt_enhancer/cycle_state"), false);
  assert.equal(frontend.includes("__promptEnhancerCycleSyncAt"), false);
});

test("cycle completion is guarded by current mode and revision", () => {
  const start = frontend.indexOf("function cycleExecutionMatchesNode");
  assert.notEqual(start, -1);
  const end = frontend.indexOf("\n}\n", start) + 3;
  const body = frontend.slice(start, end);
  assert.match(body, /currentMode\s*!==\s*"fixed"/);
  assert.match(body, /currentMode\s*===\s*resultMode/);
  assert.match(body, /promptCycleRevision\(node\)\s*===\s*resultRevision/);
});

test("Prompt Cycle uses ComfyUI queue lifecycle and ordinary schema values", () => {
  assert.match(frontend, /function preparePromptCycleQueueItem/);
  assert.match(frontend, /function finalizePromptCycleQueueItem/);
  assert.match(frontend, /planPromptCycleQueueItem/);
  assert.match(frontend, /setWidgetValue\(widget\(node, "prompt_history_index"\), plan\.selectedIndex, false\)/);
  assert.match(frontend, /setWidgetValue\(queueWidget, sequence, false\)/);
  assert.match(frontend, /node\.__promptEnhancerQueueCycleState = prepared\.plan\.state/);
  assert.match(frontend, /addEventListener\?\.\("promptQueueing"/);
  assert.match(frontend, /wrapSeedControlPersistence\(node\)/);
  assert.match(frontend, /preparePromptCycleQueueItem\(node, context\)/);
  assert.match(frontend, /finalizePromptCycleQueueItem\(node, context\)/);
  assert.match(frontend, /if \(data\?\.queue_prepared\) return true;/);
  assert.match(frontend, /const queuePrepared = !!node\?\.__promptEnhancerPreparedCycleItem/);
  assert.match(frontend, /if \(!queuePrepared\) history\[index\] = enhanced/);
  assert.doesNotMatch(frontend, /indexWidget\.serializeValue = \(\) => \{/);
  assert.doesNotMatch(frontend, /queue\.serializeValue = \(\) => \{/);
});

test("runtime cycle fields stay serializable without schema positional holes", () => {
  for (const name of ["prompt_cycle_revision", "prompt_shuffle_json", "prompt_state_id", "prompt_runtime_scope"]) {
    assert.match(frontend, new RegExp(`"${name}"`));
  }
  assert.match(frontend, /w\.options\.serialize === false/);
  assert.match(frontend, /delete w\.options\.serialize/);
  assert.doesNotMatch(frontend, /w\.serialize = false/);
});

test("shuffle deck phase survives runtime-state remounts", () => {
  assert.match(frontend, /promptShuffleStarted:/);
  assert.match(frontend, /node\.__promptEnhancerShuffleStarted = state\.promptCycle === "shuffle"/);
});

test("unused enhancement seed is excluded from normal workflow variability", () => {
  assert.match(frontend, /function installExecutionSerialization/);
  assert.match(frontend, /serializeEnhancementSeed/);
  assert.match(frontend, /shouldAdvanceEnhancementSeed/);
  assert.match(frontend, /promptEnhancerManual: true/);
});

test("workflow toggle uses explicit boolean normalization", () => {
  assert.match(frontend, /normalizeWidgetBoolean\(widget\(node, "enhance_with_workflow"\)\?\.value\)/);
  assert.doesNotMatch(frontend, /!!widget\(node, "enhance_with_workflow"\)\?\.value/);
});
