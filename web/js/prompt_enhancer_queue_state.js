/**
 * Prompt Enhancer queue/API serialization helpers.
 *
 * Prompt Cycle uses its own native ComfyUI control_after_generate target.
 * This dependency-free helper intentionally contains only enhancement-seed
 * behavior so there is no second custom cycle state machine.
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
