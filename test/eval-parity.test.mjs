import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const ROOT = new URL("..", import.meta.url);
const read = (rel) => readFileSync(new URL(rel, ROOT), "utf8");

// The quantization gate is only as good as its comparison. These three bugs
// each made it compare two different things and attribute the difference to
// quantization:
//
//   1. eval-onnx decoded greedily while the browser samples with a repetition
//      penalty. Greedy is far more loop-prone: measured on one model, greedy
//      produced repetition loops that the widget's own settings never did, so
//      the gate was reporting loops the artifact does not have.
//   2. eval-onnx generated 96 tokens while `wasmtune eval` generated 128.
//      Longer answers score higher on keyword recall, so that asymmetry alone
//      can move the delta by more than the gate's threshold.
//   3. eval_onnx.py's flags were never forwarded from onnx-eval.mjs, so (1)
//      was invisible: the numbers were identical to greedy.
//
// Found because a retrained model's q8 "introduced 4 repetition loops" that
// turned out to be the decoder.

describe("eval-onnx: decodes like the widget", () => {
  it("forwards the decoding flags from the wrapper to python", () => {
    const s = read("scripts/onnx-eval.mjs");
    // Without these, onnx-eval.mjs silently defaults onnx to greedy.
    for (const f of [
      '"--temperature", String(TEMPERATURE)',
      '"--top-p", String(TOP_P)',
      '"--repetition-penalty", String(REPETITION_PENALTY)',
      '"--presence-penalty", String(PRESENCE_PENALTY)',
      '"--frequency-penalty", String(FREQUENCY_PENALTY)',
    ]) {
      assert.ok(s.includes(f), `onnx-eval.mjs must forward ${f}`);
    }
  });

  it("reads those flags off the command line", () => {
    const s = read("scripts/onnx-eval.mjs");
    for (const flag of ["temperature", "top-p", "repetition-penalty", "presence-penalty", "frequency-penalty"]) {
      assert.match(s, new RegExp(`get\\("${flag}"`), `must parse --${flag}`);
    }
  });

  it("fails loudly when it cannot run, instead of exiting 0", () => {
    // Exit 0 from a gate that scored nothing reads as "the artifact is fine".
    const s = read("scripts/onnx-eval.mjs");
    const exits = [...s.matchAll(/process\.exit\((\d+)\)/g)].map((m) => m[1]);
    assert.ok(exits.length >= 2, "expected the setup-failure exits to be explicit");
    for (const code of exits) {
      assert.notEqual(code, "0", `process.exit(${code}) must not report success`);
    }
  });

  it("looks for the tokenizer where export.py actually puts it", () => {
    // convert writes only the graph into onnx/. The tokenizer ships in the
    // training run's merged/ dir, a sibling of the graph.
    const s = read("scripts/onnx-eval.mjs");
    assert.match(s, /path\.join\(base, "run", "merged"\)/);
    assert.match(s, /path\.join\(base, "merged"\)/);
  });
});

describe("eval_onnx.py: sampled generation", () => {
  const py = () => read("python/eval_onnx.py");

  it("does not hard-code argmax", () => {
    const s = py();
    assert.doesNotMatch(
      s, /next_id = int\(np\.argmax\(logits\[0, -1\]\)\)/,
      "argmax is greedy decoding; the gate must decode like the widget does");
  });

  it("implements temperature, top-p and all three penalties", () => {
    const s = py();
    for (const arg of [
      "--temperature", "--top-p", "--repetition-penalty",
      "--presence-penalty", "--frequency-penalty", "--seed",
    ]) {
      assert.ok(s.includes(`"${arg}"`), `must accept ${arg}`);
    }
    assert.match(s, /def softmax/);
    assert.match(s, /Counter\(generated\)/, "frequency penalty needs token counts");
  });

  it("is deterministic given a seed", () => {
    // A gate that moves on its own cannot gate anything.
    const s = py();
    assert.match(s, /default_rng\(a\.seed\)/);
  });
});

describe("wasmtune eval: same decoding and budget as the artifact's gate", () => {
  it("passes the widget's chat decoding to eval_lm.py", () => {
    const s = read("src/eval/index.mjs");
    for (const f of [
      '"--temperature"', '"--top-p"', '"--repetition-penalty"',
      '"--presence-penalty"', '"--frequency-penalty"',
    ]) {
      assert.ok(s.includes(f), `eval must forward ${f} so fp16 and q8 decode alike`);
    }
  });

  it("generates no more tokens than eval-onnx does", () => {
    // eval-onnx defaults to 96. eval defaulted to 128, so the gate compared a
    // longer fp16 baseline against a shorter artifact - and longer answers
    // score higher on keyword recall, so the asymmetry alone could move the
    // delta past the threshold.
    const bin = read("bin/wasmtune.mjs");
    const evalLine = bin.match(/maxTokens: opts\.maxTokens \?\? [^\n]*/);
    const onnxLine = bin.match(/--max-tokens", String\(opts\.maxTokens \?\? (\d+)\)/);
    assert.ok(evalLine, "wasmtune eval must resolve a token budget");
    assert.ok(onnxLine, "eval-onnx must resolve a token budget");
    const evalBudget = Number(evalLine[0].match(/\?\? [^,\n]*?(\d+)\s*[,)]/)?.[1] ?? NaN);
    assert.equal(
      evalBudget, Number(onnxLine[1]),
      `eval (${evalLine[0].trim()}) and eval-onnx (${onnxLine[1]}) must share one budget`);
  });
});

describe("eval_lm.py: presence/frequency penalties", () => {
  it("does not rely on a closure over a function-local import", () => {
    // A class defined inside a function cannot see that function's locals:
    // class scope is skipped in method name resolution, so `torch` would be a
    // NameError at call time. The import must be inside the factory.
    const s = read("python/eval_lm.py");
    const fn = s.slice(s.indexOf("def penalties_from"), s.indexOf("def build_logits_processors"));
    assert.match(fn, /import torch/, "the penalty factory must import torch itself");
  });

  it("wires the processors into generate()", () => {
    const s = read("python/eval_lm.py");
    assert.match(s, /logits_processor=processors/);
    assert.match(s, /RepetitionPenaltyLogitsProcessor/);
  });
});
