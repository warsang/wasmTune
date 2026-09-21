// wasmtune — chat entry: framework-free define + capability helpers.
export { defineSiteChat } from "./SiteChat.js";
export { mountAssistant, fetchManifest, planFromManifest, resolveManifestUrl, resolveEntryArtifacts } from "./mount.mjs";
export { detectCapabilities, hasWebGPU, pickArtifact, pickArtifactForEntry, ggufUrl } from "./fallback.mjs";
export {
  detectHardware, memoryBudgetGB, manifestEntries, resolveRequirements,
  fitRequirements, fitReport, rankEntries, pickModel, smallerEntries, TIER_RANK,
} from "./hardware.mjs";
export { resolveChatOptions, defaultSystemPrompt, toWebLlmRequest, toWllamaRequest, DEFAULT_CHAT_OPTIONS } from "./options.mjs";
export { createLoopGuard, findLoopOnset, truncateAtLoop, stripThinkingBlocks, LOOP_FALLBACK } from "./guard.mjs";
