import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { parseSynthSpec, parseSynthOutput, synthQaForChunk } from "../src/dataset/synth.mjs";

describe("synth spec", () => {
  it("parses provider:model specs", () => {
    assert.deepEqual(parseSynthSpec("openai:gpt-4o-mini"), { provider: "openai", model: "gpt-4o-mini" });
    assert.deepEqual(parseSynthSpec("ollama:qwen3-4b"), { provider: "ollama", model: "qwen3-4b" });
    assert.deepEqual(parseSynthSpec("mlx:mlx-community/Qwen3-4B-4bit"), { provider: "mlx", model: "mlx-community/Qwen3-4B-4bit" });
  });

  it("rejects bad specs", () => {
    assert.throws(() => parseSynthSpec("openai"), /provider:model/);
    assert.throws(() => parseSynthSpec("bogus:model"), /unknown synth provider/);
    assert.throws(() => parseSynthSpec("openai:"), /missing model/);
  });

  it("requires OPENAI_API_KEY for openai without key", async () => {
    const saved = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    try {
      await assert.rejects(
        synthQaForChunk("Some text. ".repeat(30), { provider: "openai", model: "gpt-4o-mini", siteName: "Demo" }),
        /OPENAI_API_KEY/,
      );
    } finally {
      if (saved !== undefined) process.env.OPENAI_API_KEY = saved;
    }
  });
});

describe("synth output parsing", () => {
  it("parses JSON arrays, tolerating fences", () => {
    const out = parseSynthOutput('```json\n[{"q": "What is X?", "a": "X is a thing."}, {"q": "Bad", "a": ""}]\n```');
    assert.equal(out.length, 1);
    assert.equal(out[0].q, "What is X?");
  });

  it("parses Q:/A: format", () => {
    const out = parseSynthOutput("Q: What is the PID?\nA: The PID is 1312.\n\nQ: Who?\nA: Nobody here.");
    assert.equal(out.length, 2);
    assert.match(out[0].a, /1312/);
  });

  it("returns [] for empty/garbage", () => {
    assert.deepEqual(parseSynthOutput(""), []);
    assert.deepEqual(parseSynthOutput("no questions here at all"), []);
  });
});
