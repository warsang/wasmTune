import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";

const CONVERT = new URL("../src/convert/to_mlc.mjs", import.meta.url);
const SCRIPT = new URL("../python/dedupe_onnx.py", import.meta.url);
const src = () => readFileSync(CONVERT, "utf8");
const py = () => readFileSync(SCRIPT, "utf8");

// torch.onnx with --task text-generation-with-past writes a RoPE sin/cos cache
// into every decoder layer. Those tables depend only on position and head dim,
// so a 24-layer export ships 48 byte-identical tensors: [32768, 64] fp32 x 2 x
// 24 = 403 MB of a 901 MB file. Merging them is bit-identical arithmetic and
// cut a real Qwen2.5-0.5B export from 901 MB to 511 MB.

describe("onnx dedupe: convert wiring", () => {
  it("runs the dedupe step after the optimum export", () => {
    const s = src();
    const exportIdx = s.indexOf('"export", "onnx"');
    const dedupeIdx = s.indexOf("dedupe_onnx.py");
    assert.ok(exportIdx !== -1, "optimum export not found");
    assert.ok(dedupeIdx !== -1, "dedupe step not wired in");
    assert.ok(dedupeIdx > exportIdx, "dedupe must run after the export");
  });

  it("passes --data-name so the graph can be renamed into place", () => {
    // The dedupe writes an external <name>.onnx_data. If the graph is renamed
    // afterwards without the data file matching, the graph's own reference to
    // it breaks — a 4 MB graph and 487 MB of orphaned weights.
    const s = src();
    assert.match(s, /"--data-name"/);
    assert.match(s, /`\$\{f\}_data`/);
  });

  it("resolves the venv from the finetune out dir, not webDir", () => {
    // runPython finds the venv via venvPaths(outDir). convertModel's outDir is
    // the *web* output dir, which has no venv, so the dedupe would silently
    // fall back to a bare python3.
    const s = src();
    assert.match(s, /const finetuneOut = path\.resolve\(mergedDir, "\.\.", "\.\."\)/);
    assert.match(s, /outDir: finetuneOut/);
  });

  it("treats a dedupe failure as non-fatal", () => {
    const s = src();
    const idx = s.indexOf("dedupe skipped for");
    assert.ok(idx !== -1, "dedupe failure must be reported as a note, not thrown");
    assert.ok(s.slice(Math.max(0, idx - 400), idx).includes("catch"),
      "dedupe must be wrapped in a catch");
  });
});

describe("onnx dedupe: script", () => {
  it("ships and exposes the CLI the converter calls", () => {
    assert.ok(existsSync(SCRIPT), "python/dedupe_onnx.py missing");
    const s = py();
    for (const flag of ['"--model"', '"--out"', '"--data-name"']) {
      assert.ok(s.includes(flag), `missing ${flag}`);
    }
  });

  it("keys unreadable tensors by name so they can never be merged", () => {
    // Merging two tensors that merely look identical would corrupt the graph.
    // A tensor whose bytes cannot be read must fall back to a unique key.
    const s = py();
    assert.match(s, /h\.update\(init\.name\.encode\(\)\)/);
  });

  it("loud-skips when onnx is unavailable instead of failing the build", () => {
    const s = py();
    assert.match(s, /dedupe skipped: onnx not installed/);
    // convert already loud-skips a missing optimum-cli the same way.
    assert.match(s, /return 0/);
  });
});
