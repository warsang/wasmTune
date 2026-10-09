import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// transformers.js resolves onnx/model_<dtype>.onnx, and the dtype names are
// not 1:1 with the graph names. `wasmtune convert` currently exports whatever
// optimum writes (model.onnx, fp32), so a manifest that asks the worker for
// dtype q8 points at a file nobody produced: transformers.js 404s the graph,
// the worker falls through to engine="unavailable", and the visitor watches
// the model never load — with no hint that the cause is a filename.

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONVERT = readFileSync(path.join(ROOT, "src/convert/to_mlc.mjs"), "utf8");

const DTYPE_SUFFIX = {
  q4: "model_q4.onnx",
  q4f16: "model_q4f16.onnx",
  fp16: "model_fp16.onnx",
  fp32: "model.onnx",
  int8: "model_int8.onnx",
  uint8: "model_uint8.onnx",
  q8: "model_quantized.onnx",
};

describe("onnx dtype contract with transformers.js", () => {
  it("maps every dtype transformers.js can name to a real graph name", () => {
    // The trap is exactly one entry: dtype "q8" is served as "quantized", so
    // model_q8.onnx is a name transformers.js will never request. int8 and
    // uint8 DO map to themselves and must keep doing so.
    assert.equal(DTYPE_SUFFIX.q8, "model_quantized.onnx");
    assert.equal(DTYPE_SUFFIX.int8, "model_int8.onnx");
    assert.equal(DTYPE_SUFFIX.uint8, "model_uint8.onnx");
    for (const [dtype, name] of Object.entries(DTYPE_SUFFIX)) {
      assert.match(name, /^model(_[a-z0-9]+)?\.onnx$/, `${dtype} -> ${name}`);
    }
  });

  it("pins the gap that exists today", () => {
    // The failing state: convert exports fp32 and the manifest carries no
    // dtype, so the ONNX tier can never load the graph the widget asks for.
    assert.match(CONVERT, /"export", "onnx"/, "convert should call optimum");
    assert.match(CONVERT, /entry\.artifacts\.onnx = rel\(out\)/,
      "whatever optimum writes becomes the artifact");
    // No quantization step yet — this is the missing piece.
    assert.ok(
      !/quantize_onnx|onnx.*--quantize/i.test(CONVERT),
      "convert has no ONNX quantization step (this test documents the gap)",
    );
  });
});
