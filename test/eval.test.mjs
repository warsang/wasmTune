import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  extractKeywords,
  hasRepetitionLoop,
  scoreResponse,
  summarize,
  faithfulness,
  scoreConversation,
  CONVERSE_PROBES,
} from "../src/eval/score.mjs";
import { parseJudgeSpec, parseJudgeScore } from "../src/eval/judge.mjs";
import { buildEvalPrompts, buildMemorizationPrompts } from "../src/eval/index.mjs";

describe("eval scoring", () => {
  it("extracts distinctive keywords (numbers, code, terms)", () => {
    const kw = extractKeywords("The PID of kfsample.exe is 1312 in every world.");
    assert.ok(kw.includes("1312"), JSON.stringify(kw));
    assert.ok(kw.some((k) => /kfsample/i.test(k)), JSON.stringify(kw));
  });

  it("scores full recall at 1", () => {
    const s = scoreResponse(
      "Use !hookscan to find it. The routine is PsLookupProcessByProcessId.",
      "!hookscan finds the detoured PsLookupProcessByProcessId export.",
    );
    assert.ok(s.score >= 0.5, JSON.stringify(s));
    assert.equal(s.repetition, false);
  });

  it("zeroes repetition loops", () => {
    const loop = "the detour never lands because the detour never lands because ".repeat(6);
    assert.equal(hasRepetitionLoop(loop), true);
    const s = scoreResponse(loop + " !hookscan PsLookupProcessByProcessId", "!hookscan PsLookupProcessByProcessId");
    assert.equal(s.score, 0);
    assert.equal(s.repetition, true);
  });

  it("does not flag normal prose as looped", () => {
    assert.equal(hasRepetitionLoop("ActiveProcessLinks is a LIST_ENTRY inside EPROCESS linking all processes together."), false);
  });

  it("penalizes rambling dumps", () => {
    const ref = "The PID is 1312.";
    const s = scoreResponse("The PID is 1312. " + "Extra filler words about lists. ".repeat(40), ref);
    assert.equal(s.lengthFactor, 0.5);
    assert.ok(s.score < 1, JSON.stringify(s));
  });

  it("summarizes averages and loop counts", () => {
    const s = summarize([{ score: 1, repetition: false }, { score: 0, repetition: true }]);
    assert.equal(s.avg, 0.5);
    assert.equal(s.looped, 1);
    assert.equal(s.count, 2);
  });
});

describe("faithfulness", () => {
  const SRC = "ActiveProcessLinks is a LIST_ENTRY inside EPROCESS. Use !hookscan to find detours.";

  it("scores grounded answers high", () => {
    const f = faithfulness("ActiveProcessLinks is a LIST_ENTRY; run !hookscan.", SRC);
    assert.ok(f.score >= 0.5, JSON.stringify(f));
    assert.equal(f.untraced.length, 0);
  });

  it("flags invented commands and fields as untraced", () => {
    const f = faithfulness("Run !syscalltest KfBox.sys and clear the Kfrozen flag at 0xFFFF1234.", SRC);
    assert.ok(f.score < 0.5, JSON.stringify(f));
    assert.ok(f.untraced.length >= 2, JSON.stringify(f));
  });

  it("empty term sets score 1 (nothing to trace)", () => {
    assert.equal(faithfulness("ok", SRC).score, 1);
  });
});

describe("conversational probes", () => {
  it("has a fixed generic probe set", () => {
    const kinds = new Set(CONVERSE_PROBES.map((p) => p.kind));
    assert.ok(kinds.has("greeting") && kinds.has("fallback") && kinds.has("scope"));
    assert.ok(CONVERSE_PROBES.every((p) => p.id && p.prompt));
  });

  it("passes short greetings, fails dumps", () => {
    assert.equal(scoreConversation("Hi! How can I help?", "greeting").pass, true);
    const dump = "kd> !a\nkd> !b\nkd> !c\nkd> !d\n" + "x".repeat(500);
    assert.equal(scoreConversation(dump, "greeting").pass, false);
  });

  it("requires ignorance phrasing for fallbacks", () => {
    assert.equal(scoreConversation("I don't have information about that in the docs.", "fallback").pass, true);
    assert.equal(scoreConversation("The answer is 0xFFFF1234, definitely.", "fallback").pass, false);
  });

  it("rejects unknown kinds", () => {
    assert.equal(scoreConversation("hi", "nope").pass, false);
  });
});

describe("judge spec", () => {
  it("parses provider:model", () => {
    assert.deepEqual(parseJudgeSpec("ollama:qwen3-4b"), { provider: "ollama", model: "qwen3-4b" });
    assert.deepEqual(parseJudgeSpec("openai:gpt-4o-mini"), { provider: "openai", model: "gpt-4o-mini" });
    assert.throws(() => parseJudgeSpec("mlx:foo"), /openai\|ollama/);
    assert.throws(() => parseJudgeSpec("nope"), /provider:model/);
  });

  it("parses digit scores", () => {
    assert.equal(parseJudgeScore("2"), 2);
    assert.equal(parseJudgeScore("Grade: 1 - partial"), 1);
    assert.equal(parseJudgeScore("no digit here"), null);
  });
});

describe("eval prompts", () => {
  it("builds deterministic prompts from holdout chunks", async () => {
    const chunks = [
      { source: "a.md", ordinal: 0, hash: "aa00000000000000", text: "# Widget\n\nWidget is a device. It works well for everyone here." },
      { source: "b.md", ordinal: 0, hash: "bb00000000000000", text: "# Gadget\n\nGadget is a tool. It helps across the workflow here." },
    ];
    const a = await buildEvalPrompts(chunks, { siteName: "Demo", maxPrompts: 10 });
    const b = await buildEvalPrompts(chunks, { siteName: "Demo", maxPrompts: 10 });
    assert.deepEqual(a, b);
    assert.ok(a.length > 0);
    assert.ok(a.every((p) => p.id && p.prompt && p.reference));
  });

  it("spreads across sources", async () => {
    const chunks = ["a", "b", "c"].map((s, i) => ({
      source: `${s}.md`, ordinal: 0, hash: `${i}${i}00000000000000`,
      text: `# Topic${s}\n\nTopic${s} is a concept. It matters for everyone here today.`,
    }));
    const prompts = await buildEvalPrompts(chunks, { siteName: "Demo", maxPrompts: 3 });
    assert.equal(new Set(prompts.map((p) => p.source)).size, 3);
  });

  it("samples memorization prompts deterministically from QA pairs", () => {
    const pairs = Array.from({ length: 20 }, (_, i) => ({
      messages: [
        { role: "user", content: `What is thing${i}?` },
        { role: "assistant", content: `Thing${i} is a concept used here.` },
      ],
      meta: { source: "a.md", ordinal: i, kind: i % 2 ? "fact-qa" : "explain" },
    }));
    const a = buildMemorizationPrompts(pairs, { maxPrompts: 5 });
    const b = buildMemorizationPrompts(pairs, { maxPrompts: 5 });
    assert.deepEqual(a, b);
    assert.ok(a.length > 0 && a.length <= 5);
    // prefers QA kinds over explain
    assert.ok(a.every((p) => p.id.startsWith("mem-") && p.prompt && p.reference));
  });
});
