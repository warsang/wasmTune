// wasmtune — chat generation options (shared by SiteChat.js + worker.js).
//
// Small models loop and ramble without decoding guardrails, so the defaults
// are conservative: low temperature, repetition penalty >= 1.1, capped
// length, and a brevity-first system prompt.

export const DEFAULT_CHAT_OPTIONS = {
  temperature: 0.3,
  repetitionPenalty: 1.15,
  presencePenalty: 0.3,
  frequencyPenalty: 0.5,
  maxTokens: 256,
  topP: 0.9,
};

export function defaultSystemPrompt(siteName = "this website") {
  return (
    `You are the ${siteName} assistant. Answer questions about the site's ` +
    `documentation and pages in 1-4 sentences. If the docs don't cover a ` +
    `question, say so instead of guessing. Never invent commands, addresses, ` +
    `or field names.`
  );
}

export function resolveChatOptions(overrides = {}) {
  const o = overrides ?? {};
  // Explicit keys win over defaults; camelCase wins over snake_case.
  const get = (...keys) => {
    for (const k of keys) if (o[k] !== undefined) return o[k];
    return undefined;
  };
  const num = (v, fallback, min, max) =>
    Number.isFinite(Number(v)) ? Math.min(max, Math.max(min, Number(v))) : fallback;
  const d = DEFAULT_CHAT_OPTIONS;
  return {
    temperature: num(get("temperature") ?? d.temperature, 0.3, 0, 2),
    repetitionPenalty: num(get("repetitionPenalty", "repetition_penalty") ?? d.repetitionPenalty, 1.15, 1, 2),
    presencePenalty: num(get("presencePenalty", "presence_penalty") ?? d.presencePenalty, d.presencePenalty, -2, 2),
    frequencyPenalty: num(get("frequencyPenalty", "frequency_penalty") ?? d.frequencyPenalty, d.frequencyPenalty, -2, 2),
    maxTokens: Math.floor(num(get("maxTokens", "max_tokens") ?? d.maxTokens, 256, 16, 4096)),
    topP: num(get("topP", "top_p") ?? d.topP, 0.9, 0, 1),
    systemPrompt: typeof get("systemPrompt", "system_prompt") === "string" && get("systemPrompt", "system_prompt").trim()
      ? get("systemPrompt", "system_prompt")
      : null,
    // Chat-template kwargs (e.g. {enable_thinking: false} for hybrid
    // reasoning models like Qwen3.5 / Gemma 4). Plain-object passthrough;
    // only sent on paths that support it (wllama).
    templateKwargs: (() => {
      const v = get("templateKwargs", "template_kwargs", "chat_template_kwargs");
      return v && typeof v === "object" && !Array.isArray(v) ? v : {};
    })(),
  };
}

// WebLLM (OpenAI-compatible) request fields.
export function toWebLlmRequest(messages, opts) {
  const o = resolveChatOptions(opts);
  return {
    messages,
    stream: true,
    temperature: o.temperature,
    top_p: o.topP,
    max_tokens: o.maxTokens,
    repetition_penalty: o.repetitionPenalty,
    presence_penalty: o.presencePenalty,
    frequency_penalty: o.frequencyPenalty,
  };
}

// wllama (OAI-compat layer) request fields. wllama exposes presence /
// frequency penalties; repeat penalty lives on the lower-level sampler,
// so presence+frequency are the portable loop killers here.
export function toWllamaRequest(messages, opts) {
  const o = resolveChatOptions(opts);
  const req = {
    messages,
    max_tokens: o.maxTokens,
    temperature: o.temperature,
    top_p: o.topP,
    presence_penalty: o.presencePenalty,
    frequency_penalty: o.frequencyPenalty,
  };
  if (Object.keys(o.templateKwargs).length) req.chat_template_kwargs = o.templateKwargs;
  return req;
}
