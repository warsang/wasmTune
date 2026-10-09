import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  tokenFor, graphsIn, externalDataRefs, DTYPE_SUFFIX,
} from "../src/publish.mjs";

// Publishing must stay optional: the whole rest of the pipeline runs offline,
// and a missing token has to degrade to a clear skip rather than an error.

describe("publish: token resolution", () => {
  it("reads HF_TOKEN, then HUGGING_FACE_HUB_TOKEN, then gives up", () => {
    assert.equal(tokenFor(null, { HF_TOKEN: "hf_a" }), "hf_a");
    assert.equal(tokenFor(null, { HUGGING_FACE_HUB_TOKEN: "hf_b" }), "hf_b");
    assert.equal(tokenFor(null, {}), null);
    assert.equal(tokenFor(null, { HF_TOKEN: "" }), "");
  });

  it("prefers an explicit --token over the environment", () => {
    assert.equal(tokenFor("hf_cli", { HF_TOKEN: "hf_env" }), "hf_cli");
  });
});

describe("publish: graph discovery", () => {
  it("maps dtypes to the filenames transformers.js actually requests", () => {
    // The trap is one entry: dtype "q8" is served as "model_quantized.onnx", so
    // model_q8.onnx is a name transformers.js will never request. int8 and
    // uint8 DO map to themselves and must keep doing so.
    assert.equal(DTYPE_SUFFIX.q8, "model_quantized.onnx");
    assert.equal(DTYPE_SUFFIX.int8, "model_int8.onnx");
    assert.equal(DTYPE_SUFFIX.uint8, "model_uint8.onnx");
    for (const [dtype, name] of Object.entries(DTYPE_SUFFIX)) {
      assert.match(name, /^model(_[a-z0-9]+)?\.onnx$/, `${dtype} -> ${name}`);
    }
  });

  it("maps q4f16 onto q4, because ORT has no q4f16 dynamic mode", () => {
    // optimum/ort produce q4 for a "q4f16" request; claiming otherwise puts a
    // filename in the manifest that nothing will ever write.
    assert.equal(DTYPE_SUFFIX.q4f16, DTYPE_SUFFIX.q4);
  });

  it("returns an empty list for a missing onnx dir instead of throwing", async () => {
    assert.deepEqual(await graphsIn(path.join(tmpdir(), "definitely-not-here-wasmtune")), []);
  });

  it("finds only graphs that exist, smallest first", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-graphs-"));
    await mkdir(path.join(dir, "onnx"), { recursive: true });
    await writeFile(path.join(dir, "onnx", "model_quantized.onnx"), Buffer.alloc(400));
    await writeFile(path.join(dir, "onnx", "model_q4.onnx"), Buffer.alloc(100));
    await writeFile(path.join(dir, "onnx", "model_q4.onnx_data"), Buffer.alloc(50));
    await writeFile(path.join(dir, "onnx", "unrelated.txt"), "x");

    const found = await graphsIn(path.join(dir, "onnx"));
    assert.deepEqual(found.map((g) => g.dtype), ["q4", "q8"], "smallest first");
    assert.equal(found[0].bytes, 100);
  });
});

describe("publish: external data refs", () => {
  it("finds a single external-data file referenced by the graph", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-ext-"));
    const graph = path.join(dir, "model_q4.onnx");
    // Enough of a protobuf-ish blob for the ascii scan to pick the location up.
    await writeFile(graph, Buffer.concat([
      Buffer.from([0x0a, 0x14]),
      Buffer.from("model_q4.onnx_data\0", "latin1"),
      Buffer.alloc(32),
    ]));
    const refs = externalDataRefs(graph);
    assert.ok(refs.includes("model_q4.onnx_data"), `got ${JSON.stringify(refs)}`);
  });

  it("returns nothing for a self-contained graph", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-self-"));
    const graph = path.join(dir, "model.onnx");
    await writeFile(graph, Buffer.alloc(64, 7));
    assert.deepEqual(externalDataRefs(graph), []);
  });
});
