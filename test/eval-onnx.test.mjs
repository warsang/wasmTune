import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// `wasmtune eval` scores the merged fp16 weights. What the widget loads is a
// quantized ONNX graph. Those are different models and give different answers —
// the published one said "In RAM on first run" where the fp16 one said the
// correct credentials path — and no gate noticed, because none looked at the
// artifact that ships. `wasmtune eval-onnx` closes that loop.

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PY = readFileSync(path.join(ROOT, "python/eval_onnx.py"), "utf8");
const RUNNER = readFileSync(path.join(ROOT, "scripts/onnx-eval.mjs"), "utf8");

describe("eval-onnx: contract", () => {
  it("has the pieces the CLI wires together", () => {
    assert.ok(existsSync(path.join(ROOT, "python/eval_onnx.py")));
    assert.ok(existsSync(path.join(ROOT, "scripts/onnx-eval.mjs")));
    assert.match(RUNNER, /--onnx-dir/);
    assert.match(RUNNER, /--prompts/);
    assert.match(RUNNER, /--out/);
  });

  it("scores with the same scorer wasmtune eval uses", () => {
    // A second scorer in Python would produce numbers that look comparable and
    // are not, which is worse than no comparison at all.
    assert.match(RUNNER, /src\/eval\/score\.mjs/, "must import the Node scorer");
    for (const fn of ["scoreResponse", "faithfulness", "scoreConversation"]) {
      assert.match(RUNNER, new RegExp(fn), `missing ${fn}`);
    }
  });

  it("tokenizes with the same tokenizer the browser uses", () => {
    // Two tokenizers, two vocabularies, two different prompts.
    assert.match(RUNNER, /AutoTokenizer/);
    assert.match(RUNNER, /inlineChatTemplate/,
      "transformers.js ignores the sibling .jinja, so it must be inlined first");
    // apply_chat_template returns {input_ids: BigInt64Array}; a plain spread
    // walks object properties and yields nothing.
    assert.match(RUNNER, /v\?\.input_ids \?\? v/);
  });

  it("reads the prompts from the same holdout the training gate used", () => {
    // Prompts the model was trained on would score as memorisation.
    assert.match(RUNNER, /eval\.prompts\.jsonl/);
  });

  it("refuses a graph whose external data is missing", () => {
    // The first publish shipped a 4 MB graph with 487 MB of weights absent.
    assert.match(PY, /references missing external data/);
  });

  it("loud-skips when onnxruntime is missing", () => {
    assert.match(PY, /eval-onnx skipped: onnxruntime not installed/);
  });

  it("fails the gate when quantization costs more than the threshold", () => {
    const cli = readFileSync(path.join(ROOT, "bin/wasmtune.mjs"), "utf8");
    assert.match(cli, /eval-onnx FAILED/, "must fail on quantization regression");
    assert.match(cli, /WASMTUNE_QUANT_DROP/, "threshold must be tunable");
    assert.match(cli, /--force/, "an escape hatch must exist");
  });

  it("is dispatched as a CLI command", () => {
    const cli = readFileSync(path.join(ROOT, "bin/wasmtune.mjs"), "utf8");
    assert.match(cli, /if \(cmd === "eval-onnx"\) return cmdEvalOnnx/);
    assert.match(cli, /eval-onnx \[--config <path>\]/, "must appear in USAGE");
  });
});
