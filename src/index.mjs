// wasmtune — public Node API.
export { loadConfig, normalizeConfig, validateConfig, defaultConfig, configModels } from "./config.mjs";
export { ALLOWLIST, LARGE_MODELS, lookupModel, lookupByBrowserId, isAllowedModel, assertAllowedModel, formatModels, requirementsFor, slugifyModel, tierForBytes, minMemoryForBytes } from "./models.mjs";
export { buildDataset } from "./dataset/index.mjs";
export { resolveBackend, detectPlatform } from "./train/router.mjs";
export { convertModel, pretrainedEntry, writeModelsManifest } from "./convert/to_mlc.mjs";
export { checkServing } from "./serve-check.mjs";
export { serve } from "./serve.mjs";

// Retrieval. The same API runs in Node (building an index at build time) and in
// the browser (embedding a query, retrieving, purging a session's chunks).
export {
  PUBLIC_SCOPE,
  buildIndex,
  chunksFromDocs,
  createRag,
  cosine,
  indexScopes,
  loadIndex,
  purgeScope,
  searchChunks,
  splitSections,
  windowText,
} from "./rag/index.mjs";
export { createLexicalEmbedder, createTransformersEmbedder, resolveEmbedder } from "./rag/embed.mjs";
export { buildGroundedUserMessage, citedSources, withRagSystemPrompt, DEFAULT_RAG_INSTRUCTIONS } from "./rag/prompt.mjs";
export { parseToolCall, stripToolCalls, toolInstruction, toolRe } from "./rag/action.mjs";
export { createToolRegistry } from "./rag/tools.mjs";
export { PUBLIC_SCOPE as RAG_PUBLIC_SCOPE, normalizeScope, scopeSet } from "./rag/scope.mjs";
export { cacheKey, clearIndexes, loadIndex as loadStoredIndex, saveIndex } from "./rag/store.mjs";
