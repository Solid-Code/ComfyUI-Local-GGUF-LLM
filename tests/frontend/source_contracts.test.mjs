import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const frontend = fs.readFileSync(path.join(root, "web/js/prompt_enhancer_dom_v0654.js"), "utf8");
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

test("Prompt Cycle uses native control-after-generate callbacks", () => {
  assert.match(frontend, /function installNativePromptCycleControl/);
  assert.match(frontend, /promptCycleControlWidget/);
  assert.match(frontend, /originalBefore/);
  assert.match(frontend, /originalAfter/);
  assert.match(frontend, /promptCycleIndexFromCounter/);
  assert.match(frontend, /nativeControlModeForPromptCycle/);
  assert.doesNotMatch(frontend, /installQueueOwnedPromptCycle/);
});

test("cycle completion never advances native queue state", () => {
  const marker = 'if (mode === "cycle") {';
  const start = frontend.indexOf(marker);
  assert.notEqual(start, -1);
  const end = frontend.indexOf("\n  }", start) + 4;
  const body = frontend.slice(start, end);
  assert.match(body, /native control_after_generate lifecycle already advanced/);
  assert.doesNotMatch(body, /setPromptHistoryState/);
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
