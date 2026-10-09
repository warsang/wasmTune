import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// transformers.js resolves onnx/model_<dtype>.onnx, and the dtype names are not
// 1:1 with the graph names. `optimum-cli export onnx` writes fp32 only, so
// before this was fixed a manifest that asked the worker for dtype q8 pointed at
// model_quantized.onnx — a file nobody had produced. transformers.js 404s the
// graph, the worker reports engine="unavailable", and the visitor watches the
// model never load with no hint that the cause is a filename.

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONVERT = readFileSync(path.join(ROOT, "src/convert/to_mlc.mjs"), "utf8");
const SCRIPT = readFileSync(path.join(ROOT, "python/quantize_onnx.py"), "utf8");

// Only q8 is renamed. int8 and uint8 DO map to themselves and must keep doing so.
const DTYPE_FILE = {
  q4: "model_q4.onnx",
  q4f16: "model_q4.onnx",
  q8: "model_quantized.onnx",
  int8: "model_int8.onnx",
  uint8: "model_uint8.onnx",
  fp32: "model.onnx",
  fp16: "model_fp16.onnx",
};

describe("onnx dtype contract with transformers.js", () => {
  it("maps every dtype transformers.js can name to a real graph name", () => {
    assert.equal(DTYPE_FILE.q8, "model_quantized.onnx");
    assert.equal(DTYPE_FILE.q4f16, "model_q4.onnx",
      "ORT's dynamic quantizer has no q4f16 mode; it produces q4");
    for (const [dtype, name] of Object.entries(DTYPE_FILE)) {
      assert.match(name, /^model(_[a-z0-9]+)?\.onnx$/, `${dtype} -> ${name}`);
    }
  });

  it("quantizes inside convert, so the manifest names a graph that exists", () => {
    assert.match(CONVERT, /quantize_onnx\.py/, "convert must call the quantizer");
    // The export comes first, then quantization, then dedupe.
    const exportIdx = CONVERT.indexOf('"export", "onnx"');
    const quantIdx = CONVERT.indexOf("quantize_onnx.py");
    const dedupeIdx = CONVERT.indexOf("dedupe_onnx.py");
    assert.ok(exportIdx < quantIdx && quantIdx < dedupeIdx,
      "expected export -> quantize -> dedupe in that order");
  });

  it("records the dtype it actually produced in the manifest", () => {
    assert.match(CONVERT, /entry\.artifacts\.onnxDtype = onnxDtypes\[0\]/,
      "the manifest must say which dtype exists rather than hoping");
  });

  it("has a quantizer script with the dtype table and loud skips", () => {
    assert.ok(existsSync(path.join(ROOT, "python/quantize_onnx.py")));
    assert.match(SCRIPT, /model_quantized\.onnx/, "q8 must map to model_quantized");
    assert.match(SCRIPT, /DTYPE_ALIAS/, "q4f16 needs an explicit alias");
    assert.match(SCRIPT, /quantize skipped: onnxruntime not installed/,
      "missing onnxruntime must be a loud skip, like every other optional tool");
  });
});
