import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const ROOT = new URL("..", import.meta.url);
const read = (rel) => readFileSync(new URL(rel, ROOT), "utf8");

describe("publish: revision pins the cache key", () => {
  it("threads the publish commit oid from upload to manifest to worker", () => {
    // transformers.js caches "<repo>/<path>@<revision>". With no revision it
    // caches by path alone, so replacing onnx/model_quantized.onnx in place
    // means every returning visitor keeps the previous weights and a re-publish
    // is indistinguishable from no publish at all.
    const pub = read("src/publish.mjs");
    assert.match(pub, /revision = null/);
    assert.match(pub, /\.\.\.\(revision \? \{ revision \} : \{\}\)/);
    // /^OID (\S+)$/m, on the source. The backslashes here are only so this
    // regexp survives living inside a JS string comparison.
    assert.match(pub, /\^OID \(/);

    const bin = read("bin/wasmtune.mjs");
    assert.match(bin, /publishedRevision = graphUpload\.oid/);
    assert.match(bin, /revision: publishedRevision/);

    const mount = read("src/chat/mount.mjs");
    assert.match(mount, /a\.onnx\?\.revision/);
    assert.match(mount, /setAttribute\("onnx-revision", plan\.onnxRevision\)/);

    const worker = read("src/chat/worker.js");
    assert.match(worker, /onnxRevision = null/);
    assert.match(worker, /revision: onnxRevisionResolved/);
  });

  it("has the uploader print the commit oid", () => {
    const s = read("python/hf_upload.py");
    assert.match(s, /print\(f"OID \{oid\}"\)/);
  });
});
