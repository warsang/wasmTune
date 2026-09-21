import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ALLOWLIST, isAllowedModel, assertAllowedModel } from "../src/models.mjs";

describe("models allowlist", () => {
  it("contains only small browser-viable models", () => {
    assert.ok(ALLOWLIST.length >= 10);
    for (const m of ALLOWLIST) {
      // Every entry needs an HF id plus at least one browser path
      // (WebLLM prebuilt, ONNX community build, or GGUF for wllama).
      assert.ok(m.hf, JSON.stringify(m));
      assert.ok(m.webllm || m.onnx || m.gguf, JSON.stringify(m));
    }
  });

  it("accepts allowlisted ids case-insensitively", () => {
    assert.ok(isAllowedModel("qwen/qwen2.5-1.5b-instruct"));
  });

  it("rejects frontier-scale models by default", () => {
    assert.equal(isAllowedModel("meta-llama/Llama-3.1-405B"), false);
    assert.throws(() => assertAllowedModel("meta-llama/Llama-3.1-405B"));
  });

  it("gates 7-9B behind allowLarge", () => {
    assert.equal(isAllowedModel("Qwen/Qwen2.5-7B-Instruct"), false);
    assert.equal(isAllowedModel("Qwen/Qwen2.5-7B-Instruct", { allowLarge: true }), true);
  });
});
