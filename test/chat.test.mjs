import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  resolveChatOptions,
  defaultSystemPrompt,
  toWebLlmRequest,
  toWllamaRequest,
  DEFAULT_CHAT_OPTIONS,
} from "../src/chat/options.mjs";

describe("chat options", () => {
  it("defaults are conservative (low temp, repetition penalty, capped length)", () => {
    assert.ok(DEFAULT_CHAT_OPTIONS.temperature <= 0.5);
    assert.ok(DEFAULT_CHAT_OPTIONS.repetitionPenalty >= 1.1);
    assert.ok(DEFAULT_CHAT_OPTIONS.maxTokens <= 512);
  });

  it("resolves overrides and clamps garbage", () => {
    const o = resolveChatOptions({ temperature: 0.7, maxTokens: 100 });
    assert.equal(o.temperature, 0.7);
    assert.equal(o.maxTokens, 100);
    assert.equal(o.repetitionPenalty, DEFAULT_CHAT_OPTIONS.repetitionPenalty);
    const bad = resolveChatOptions({ temperature: 99, maxTokens: -5, repetitionPenalty: 0 });
    assert.ok(bad.temperature <= 2);
    assert.ok(bad.maxTokens >= 16);
    assert.ok(bad.repetitionPenalty >= 1);
  });

  it("accepts snake_case aliases", () => {
    const o = resolveChatOptions({ max_tokens: 100, repetition_penalty: 1.2, top_p: 0.8 });
    assert.equal(o.maxTokens, 100);
    assert.equal(o.repetitionPenalty, 1.2);
    assert.equal(o.topP, 0.8);
  });

  it("default system prompt demands brevity and honesty", () => {
    const p = defaultSystemPrompt("Demo Docs");
    assert.ok(p.includes("Demo Docs"));
    assert.match(p, /1-4 sentences/i);
    assert.match(p, /say so|admit|unsure/i);
  });

  it("builds a WebLLM request with guardrails", () => {
    const req = toWebLlmRequest([{ role: "user", content: "hi" }], { temperature: 0.2 });
    assert.equal(req.stream, true);
    assert.equal(req.temperature, 0.2);
    assert.ok(req.repetition_penalty >= 1.1);
    assert.ok(req.max_tokens <= 4096);
  });

  it("templateKwargs defaults to {} and rejects non-objects", () => {
    assert.deepEqual(resolveChatOptions({}).templateKwargs, {});
    assert.deepEqual(resolveChatOptions({ templateKwargs: "nope" }).templateKwargs, {});
    assert.deepEqual(resolveChatOptions({ templateKwargs: ["x"] }).templateKwargs, {});
  });

  it("wllama request omits empty template kwargs, includes set ones", () => {
    const plain = toWllamaRequest([{ role: "user", content: "hi" }], {});
    assert.ok(!("chat_template_kwargs" in plain));
    assert.ok(plain.presence_penalty > 0 && plain.frequency_penalty > 0);
    const thinking = toWllamaRequest([], { templateKwargs: { enable_thinking: false } });
    assert.deepEqual(thinking.chat_template_kwargs, { enable_thinking: false });
  });
});
