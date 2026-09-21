// wasmtune — chat entry: framework-free define + capability helpers.
export { defineSiteChat } from "./SiteChat.js";
export { mountAssistant, fetchManifest, planFromManifest, resolveManifestUrl } from "./mount.mjs";
export { detectCapabilities, hasWebGPU, pickArtifact, ggufUrl } from "./fallback.mjs";
export { resolveChatOptions, defaultSystemPrompt, toWebLlmRequest, toWllamaRequest, DEFAULT_CHAT_OPTIONS } from "./options.mjs";
export { createLoopGuard, findLoopOnset, truncateAtLoop, stripThinkingBlocks, LOOP_FALLBACK } from "./guard.mjs";
