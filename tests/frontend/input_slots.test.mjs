import test from "node:test";
import assert from "node:assert/strict";

import {
  applyPromptEnhancerInputLabels,
  inputIsConnected,
  inputLink,
  inputSlotIndex,
  stabilizePromptEnhancerInputSlots,
} from "../../web/js/prompt_enhancer_input_slots.js";

function makeInputs() {
  return [
    { name: "images", type: "IMAGE", link: 10 },
    { name: "video", type: "VIDEO", link: 11 },
    { name: "settings", type: "LOCAL_LLM_SETTINGS", link: 12 },
  ];
}

test("stabilization changes labels only and preserves logical slot identity", () => {
  const inputs = makeInputs();
  const identities = [...inputs];
  const linksBefore = inputs.map((input) => input.link);
  const node = { inputs };

  stabilizePromptEnhancerInputSlots(node);

  assert.deepEqual(node.inputs.map((input) => input.name), ["images", "video", "settings"]);
  assert.deepEqual(node.inputs.map((input) => input.link), linksBefore);
  assert.equal(node.inputs[0], identities[0]);
  assert.equal(node.inputs[1], identities[1]);
  assert.equal(node.inputs[2], identities[2]);
  assert.equal(node.inputs[0].label, "image(s)");
  assert.equal(node.inputs[1].label, undefined);
  assert.equal(node.inputs[2].label, undefined);
});

test("input slot lookup follows backend-declared names", () => {
  const node = { inputs: makeInputs() };
  assert.equal(inputSlotIndex(node, "images"), 0);
  assert.equal(inputSlotIndex(node, "video"), 1);
  assert.equal(inputSlotIndex(node, "settings"), 2);
  assert.equal(inputSlotIndex(node, "missing"), -1);
});

test("getInputLink is preferred without mutating target_slot", () => {
  const links = [
    { id: 100, target_slot: 0 },
    { id: 101, target_slot: 1 },
    { id: 102, target_slot: 2 },
  ];
  const node = {
    inputs: makeInputs(),
    getInputLink(slot) { return links[slot]; },
  };

  const before = links.map((link) => link.target_slot);
  assert.equal(inputLink(node, "settings"), links[2]);
  assert.equal(inputIsConnected(node, "settings"), true);
  assert.deepEqual(links.map((link) => link.target_slot), before);
});

test("graph Map fallback resolves link ids without slot rewriting", () => {
  const links = new Map([
    [10, { id: 10, target_slot: 0 }],
    [11, { id: 11, target_slot: 1 }],
    [12, { id: 12, target_slot: 2 }],
  ]);
  const node = { inputs: makeInputs(), graph: { links } };
  assert.equal(inputLink(node, "video")?.id, 11);
  assert.equal(inputIsConnected(node, "images"), true);
  assert.equal(inputIsConnected(node, "settings"), true);
  assert.deepEqual([...links.values()].map((link) => link.target_slot), [0, 1, 2]);
});

test("graph object fallback works when node graph is not yet attached", () => {
  const fallback = {
    links: {
      10: { id: 10, target_slot: 0 },
      11: { id: 11, target_slot: 1 },
      12: { id: 12, target_slot: 2 },
    },
  };
  const node = { inputs: makeInputs() };
  assert.equal(inputLink(node, "settings", fallback)?.id, 12);
  assert.equal(inputIsConnected(node, "settings", fallback), true);
});

test("isInputConnected is honored when available", () => {
  const node = {
    inputs: makeInputs(),
    isInputConnected(slot) { return slot === 1; },
  };
  assert.equal(inputIsConnected(node, "images"), false);
  assert.equal(inputIsConnected(node, "video"), true);
  assert.equal(inputIsConnected(node, "settings"), false);
});

test("label helper is idempotent", () => {
  const node = { inputs: makeInputs() };
  applyPromptEnhancerInputLabels(node);
  applyPromptEnhancerInputLabels(node);
  assert.equal(node.inputs[0].label, "image(s)");
  assert.deepEqual(node.inputs.map((input) => input.name), ["images", "video", "settings"]);
});
