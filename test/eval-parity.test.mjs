import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import pathMod from "node:path";
import { resolveDecoding, manifestChat } from "../src/eval/decoding.mjs";

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

  it("resolves one decoding config shared with eval-onnx", async () => {
    // eval-onnx and eval must agree, and both must match what the widget reads
    // - the manifest's chat block. config.chat is a fallback only.
    const bin = readFileSync(new URL("../bin/wasmtune.mjs", import.meta.url), "utf8");
    assert.match(bin, /import \{ resolveDecoding \} from "\.\.\/src\/eval\/decoding\.mjs"/);
    // Both commands call the one resolver.
    const calls = [...bin.matchAll(/resolveDecoding\(config, cwd, opts\)/g)].length;
    assert.ok(calls >= 2, `expected both eval and eval-onnx to call resolveDecoding, got ${calls}`);
    // And both use its budget rather than their own default.
    assert.match(bin, /maxTokens: decoding\.maxTokens \?\? 96/);
    assert.match(bin, /"--max-tokens", String\(decoding\.maxTokens \?\? 96\)/);
  });
});

describe("decoding: the widget's parameters are the source of truth", () => {
  const withManifest = async (chat, cfgChat = {}) => {
    const dir = await mkdtemp(pathMod.join(tmpdir(), "dec-"));
    await mkdir(pathMod.join(dir, "public", "models"), { recursive: true });
    await writeFile(
      pathMod.join(dir, "public", "models", "model-manifest.json"),
      JSON.stringify({ models: [{ id: "t", source: "tuned", artifacts: { onnx: { id: "a/b", dtype: "q8" } }, chat }] }),
    );
    return resolveDecoding({ chat: cfgChat, output: { webDir: "public/models" } }, dir, {});
  };

  it("reads the manifest entry the widget reads, not config.chat", async () => {
    const d = await withManifest(
      { maxTokens: 96, temperature: 0.2, repetitionPenalty: 1.25 },
      { temperature: 0.9, topP: 0.5, presencePenalty: 0.7 },
    );
    assert.equal(d.source, "manifest");
    assert.equal(d.temperature, 0.2);
    assert.equal(d.repetitionPenalty, 1.25);
    assert.equal(d.maxTokens, 96);
  });

  it("does not fill omitted keys from config.chat", async () => {
    // The widget reads only the manifest block. A key it omits is a key the
    // browser does not apply, so falling back to config.chat for it would
    // reintroduce the exact mismatch this fixes.
    const d = await withManifest(
      { maxTokens: 96, temperature: 0.2, repetitionPenalty: 1.25 },
      { topP: 0.9, presencePenalty: 0.3, frequencyPenalty: 0.5 },
    );
    assert.equal(d.topP, null, "top_p must be unset, so the browser uses its own default");
    assert.equal(d.presencePenalty, null);
    assert.equal(d.frequencyPenalty, null);
  });

  it("falls back to config.chat when there is no manifest entry", async () => {
    const dir = await mkdtemp(pathMod.join(tmpdir(), "dec-"));
    const d = resolveDecoding(
      { chat: { temperature: 0.4, topP: 0.8, maxTokens: 64 }, output: { webDir: "public/models" } },
      dir,
      {},
    );
    assert.equal(d.source, "config.chat");
    assert.equal(d.temperature, 0.4);
    assert.equal(d.topP, 0.8);
  });

  it("explicit flags still win over both", async () => {
    const dir = await mkdtemp(pathMod.join(tmpdir(), "dec-"));
    const d = resolveDecoding(
      { chat: { temperature: 0.4, maxTokens: 64 }, output: { webDir: "public/models" } },
      dir,
      { temperature: 0.05, maxTokens: 12 },
    );
    assert.equal(d.temperature, 0.05);
    assert.equal(d.maxTokens, 12);
  });

  it("returns null for an unreadable manifest rather than throwing", async () => {
    const dir = await mkdtemp(pathMod.join(tmpdir(), "dec-"));
    await mkdir(pathMod.join(dir, "public", "models"), { recursive: true });
    await writeFile(pathMod.join(dir, "public", "models", "model-manifest.json"), "{ not json");
    assert.equal(manifestChat({ output: { webDir: "public/models" } }, dir), null);
  });
});

describe("publish: finds what convert actually wrote", () => {
  it("looks for graphs in webDir first, not output.dir", () => {
    // convert writes to webDir/<slug>/onnx. publish looked at
    // output.dir/<slug>/onnx, which never exists, so every publish printed
    // "no onnx/ graphs" and silently published nothing.
    const s = read("bin/wasmtune.mjs");
    assert.match(s, /config\.output\.webDir, multi \? slug : "\."/);
    assert.match(s, /\.find\(\(d\) => graphsIn\(d\)\.length\)/);
  });

  it("finds the tokenizer in run/merged, not merged", () => {
    // export.py writes <output.dir>/<slug>/run/merged. The single hard-coded
    // "<output.dir>/<slug>/merged" path uploaded no tokenizer at all, so the
    // graph landed on the Hub with nothing to decode with.
    const s = read("bin/wasmtune.mjs");
    assert.match(s, /path\.join\(finetuneRoot, "run", "merged"\)/);
    assert.match(s, /find\(\(d\) => existsSync\(path\.join\(d, "tokenizer\.json"\)\)\)/);
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
