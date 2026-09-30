/**
 * Prompt Enhancer input-slot helpers.
 *
 * ComfyUI identifies links by numeric target slots.  These helpers therefore
 * never reorder node.inputs and never rewrite link.target_slot.  The backend
 * declaration remains the single source of truth for logical slot identity.
 */

export function applyPromptEnhancerInputLabels(node) {
  for (const input of node?.inputs || []) {
    if (input?.name === "images") input.label = "image(s)";
  }
}

export function inputSlotIndex(node, inputName) {
  return (node?.inputs || []).findIndex((item) => item?.name === inputName);
}

export function inputLink(node, inputName, graphFallback = null) {
  const slot = inputSlotIndex(node, inputName);
  if (slot < 0) return null;

  try {
    const link = node?.getInputLink?.(slot);
    if (link) return link;
  } catch (_) {}

  const linkId = node?.inputs?.[slot]?.link;
  if (linkId == null) return null;
  const graph = node?.graph || graphFallback;
  return graph?.links?.get?.(linkId) || graph?.links?.[linkId] || null;
}

export function inputIsConnected(node, inputName, graphFallback = null) {
  const slot = inputSlotIndex(node, inputName);
  if (slot < 0) return false;
  try {
    if (typeof node?.isInputConnected === "function") return !!node.isInputConnected(slot);
  } catch (_) {}
  return !!inputLink(node, inputName, graphFallback);
}

export function stabilizePromptEnhancerInputSlots(node) {
  // Labels are cosmetic.  The node.inputs array itself is intentionally left
  // untouched so serialized numeric slot ids remain stable for the node's life.
  applyPromptEnhancerInputLabels(node);
}
