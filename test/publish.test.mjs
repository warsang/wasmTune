import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, mkdir, readFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
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

  it("ranks a self-contained quantized graph above a stub with external weights", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-graphs-"));
    await mkdir(path.join(dir, "onnx"), { recursive: true });
    // model_quantized.onnx is self-contained; model_q4.onnx is a tiny stub
    // whose weights live in model_q4.onnx_data.
    await writeFile(path.join(dir, "onnx", "model_quantized.onnx"), Buffer.alloc(400));
    await writeFile(path.join(dir, "onnx", "model_q4.onnx"), Buffer.from("model_q4.onnx_data".padEnd(100)));
    await writeFile(path.join(dir, "onnx", "model_q4.onnx_data"), Buffer.alloc(50));
    await writeFile(path.join(dir, "onnx", "unrelated.txt"), "x");

    const found = await graphsIn(path.join(dir, "onnx"));
    // Sorting by raw size picked the q4 stub and published a graph whose
    // weights were never uploaded - a 404 the moment anything loaded it.
    assert.deepEqual(found.map((g) => g.dtype), ["q8", "q4"]);
    assert.equal(found[0].selfContained, true);
    assert.equal(found[0].bytes, 400);
    assert.equal(found[1].selfContained, false);
    // totalBytes counts the external weights, so a stub is not "the smallest".
    assert.equal(found[1].totalBytes, 150);
  });

  it("puts fp32 last even though its graph file is the smallest", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-graphs-"));
    await mkdir(path.join(dir, "onnx"), { recursive: true });
    await writeFile(path.join(dir, "onnx", "model_quantized.onnx"), Buffer.alloc(400_000));
    await writeFile(path.join(dir, "onnx", "model.onnx"), Buffer.alloc(4_400));
    await writeFile(path.join(dir, "onnx", "model.onnx_data"), Buffer.alloc(4_000));

    const found = await graphsIn(path.join(dir, "onnx"));
    assert.deepEqual(found.map((g) => g.dtype), ["q8", "fp32"]);
  });
});

describe("publish: upload path", () => {
  const src = () => readFileSync(new URL("../src/publish.mjs", import.meta.url), "utf8");

  it("shells out to the streaming python uploader", async () => {
    // The Node path used PUT /api/models/{repo}/upload/main/{path}, which HF has
    // removed (it now 404s). Its replacement takes base64 in an NDJSON line,
    // and a 0.5 GB graph becomes a ~0.7 GB base64 string - past V8's
    // 0x1fffffe8 string cap. The LFS flow that survives at this size is what
    // huggingface_hub already implements, so publish streams through it.
    const s = src();
    assert.match(s, /"python", "hf_upload\.py"/);
    assert.match(s, /spawn\(py/);
    assert.doesNotMatch(s, /upload\/main/);
  });

  it("finds the venv python by walking up, not by guessing beside outDir", async () => {
    // The venv lives at the root of the output dir; per-tier subdirs have
    // neither a venv nor one of their own.
    const s = src();
    assert.match(s, /function venvPython/);
    assert.match(s, /const parent = path\.dirname\(dir\)/);
  });

  it("ships the uploader script in the npm tarball", async () => {
    const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
    assert.ok(pkg.files.includes("python/"), "python/ must ship, or publish has no uploader");
    const s = await readFile(new URL("../python/hf_upload.py", import.meta.url), "utf8");
    assert.match(s, /HfApi/);
    assert.match(s, /upload_file/);
  });
  it("does not prefix repo paths with the model slug", async () => {
    // A publish repo holds ONE model, and transformers.js resolves
    // <repo>/onnx/<dtype file> from artifacts.onnx.id. Adding a slug put the
    // graph at <repo>/<slug>/onnx/..., so publish verified a successful upload
    // that the manifest never pointed at.
    const s = readFileSync(new URL("../bin/wasmtune.mjs", import.meta.url), "utf8");
    assert.doesNotMatch(s, /const sub = multi \? `\$\{slug\}\/` : ""/);
    assert.match(s, /const sub = ""/);
  });
  it("never writes the model card to <cwd>/README.md", async () => {
    // The card is an upload payload. Writing it to <cwd>/README.md destroyed
    // the repo's own 611-line README as a side effect of `wasmtune publish`,
    // whose job is to upload a model.
    const s = await readFile(new URL("../src/publish.mjs", import.meta.url), "utf8");
    assert.match(s, /WASMTUNE_CARD_DIR/);
    assert.doesNotMatch(s, /const out = path\.join\(dir, "README\.md"\)/);
    assert.match(s, /"\.finetune", "model-card"/);
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
