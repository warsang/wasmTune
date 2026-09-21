import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { createLoopGuard, findLoopOnset, truncateAtLoop, stripThinkingBlocks, LOOP_FALLBACK } from "../src/chat/guard.mjs";
import { hasRepetitionLoop } from "../src/eval/score.mjs";
import { resolveChatOptions, toWllamaRequest } from "../src/chat/options.mjs";

const LOOP = "the detour never lands because ".repeat(4).trim();
const CLEAN = "ActiveProcessLinks is a LIST_ENTRY inside EPROCESS linking all processes together.";

describe("loop guard", () => {
  it("finds loop onset at the second occurrence", () => {
    const at = findLoopOnset(LOOP);
    assert.ok(at > 0, `onset=${at}`);
    // The cut point keeps one copy: everything before it is loop-free.
    assert.equal(hasRepetitionLoop(LOOP.slice(0, at)), false);
    assert.ok(truncateAtLoop(LOOP).length < LOOP.length);
  });

  it("returns -1 for clean text", () => {
    assert.equal(findLoopOnset(CLEAN), -1);
  });

  it("truncateAtLoop keeps one copy and drops the spiral", () => {
    const cut = truncateAtLoop(`Intro sentence here. ${LOOP} ${LOOP}`);
    assert.ok(cut.length < `Intro sentence here. ${LOOP} ${LOOP}`.length);
    assert.ok(cut.includes("Intro sentence here."));
  });

  it("guard trips mid-stream and exposes clean text + fallback", () => {
    const g = createLoopGuard({ checkEveryChars: 16, minChars: 40 });
    let r = g.feed("Intro sentence that is definitely long enough. ");
    assert.equal(r.tripped, false);
    r = g.feed(LOOP + " " + LOOP);
    assert.equal(r.tripped, true);
    assert.ok(r.text.length < 200, `clean=${r.text.length}`);
    assert.ok(r.fallback && r.fallback.length > 10);
    assert.equal(LOOP_FALLBACK, r.fallback);
    // stays tripped
    assert.equal(g.feed("more").tripped, true);
  });

  it("guard never trips on clean streams", () => {
    const g = createLoopGuard({ checkEveryChars: 8, minChars: 10 });
    for (const piece of ["ActiveProcessLinks ", "is a LIST_ENTRY ", "inside EPROCESS. ", "Drivers walk it."]) {
      assert.equal(g.feed(piece).tripped, false);
    }
    assert.equal(g.text.includes("EPROCESS"), true);
  });
});

describe("stripThinkingBlocks", () => {
  it("removes paired think tags and flags the leak", () => {
    const r = stripThinkingBlocks("<think>planning the greeting</think>Hello! How can I help?");
    assert.equal(r.text, "Hello! How can I help?");
    assert.equal(r.stripped, true);
  });

  it("drops unclosed thinking at stream cut", () => {
    const r = stripThinkingBlocks("Hello<think>wait, let me reconsider the tone");
    assert.equal(r.text, "Hello");
    assert.equal(r.stripped, true);
  });

  it("handles [Start thinking]…[End thinking] markers", () => {
    const r = stripThinkingBlocks("[Start thinking]\nHmm, greeting.\n[End thinking]\nHello there!");
    assert.equal(r.text, "Hello there!");
    assert.equal(r.stripped, true);
  });

  it("drops unbounded thinking mid-stream (answer arrives later)", () => {
    const r = stripThinkingBlocks("[Start thinking]\nHmm, greeting.");
    assert.equal(r.text, "");
    assert.equal(r.stripped, true);
  });

  it("leaves clean answers untouched", () => {
    const r = stripThinkingBlocks("Use !hookscan to find the detour.");
    assert.equal(r.text, "Use !hookscan to find the detour.");
    assert.equal(r.stripped, false);
  });

  it("does not eat legitimate angle-bracket prose", () => {
    const r = stripThinkingBlocks("Run <addr> through the debugger.");
    assert.equal(r.text, "Run <addr> through the debugger.");
    assert.equal(r.stripped, false);
  });
});

describe("wllama request options", () => {
  it("includes presence/frequency penalties for loop suppression", () => {
    const req = toWllamaRequest([{ role: "user", content: "hi" }], {});
    assert.ok(req.presence_penalty > 0, JSON.stringify(req));
    assert.ok(req.frequency_penalty > 0, JSON.stringify(req));
    assert.equal(req.temperature, 0.3);
  });

  it("resolves new penalty overrides", () => {
    const o = resolveChatOptions({ presencePenalty: 0.6, frequency_penalty: 0.8 });
    assert.equal(o.presencePenalty, 0.6);
    assert.equal(o.frequencyPenalty, 0.8);
  });
});
