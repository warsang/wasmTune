import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { hashAndRename } from "../src/convert/to_mlc.mjs";
import { ggufUrl } from "../src/chat/fallback.mjs";

describe("content-hashed artifacts", () => {
  it("renames to stem.<8hex>.ext and returns url/sha256/bytes", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hash-"));
    const f = path.join(dir, "model.Q4_K_M.gguf");
    await writeFile(f, "fake-weights-bytes");
    const g = await hashAndRename(f, dir);
    assert.match(g.url, /^\/model\.Q4_K_M\.[0-9a-f]{8}\.gguf$/);
    assert.equal(g.sha256.length, 64);
    assert.ok(g.bytes > 0);
    assert.ok(g.short.length === 8);
    await stat(path.join(dir, path.basename(g.url)));
  });

  it("is deterministic: same bytes -> same name", async () => {
    const a = await mkdtemp(path.join(tmpdir(), "hash-"));
    const b = await mkdtemp(path.join(tmpdir(), "hash-"));
    await writeFile(path.join(a, "m.gguf"), "same-bytes");
    await writeFile(path.join(b, "m.gguf"), "same-bytes");
    const ga = await hashAndRename(path.join(a, "m.gguf"), a);
    const gb = await hashAndRename(path.join(b, "m.gguf"), b);
    assert.equal(ga.url, gb.url);
    assert.equal(ga.sha256, gb.sha256);
  });

  it("different bytes -> different names (cache-bust guarantee)", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "hash-"));
    await writeFile(path.join(dir, "v1.gguf"), "weights-v1");
    await writeFile(path.join(dir, "v2.gguf"), "weights-v2");
    const g1 = await hashAndRename(path.join(dir, "v1.gguf"), dir);
    const g2 = await hashAndRename(path.join(dir, "v2.gguf"), dir);
    assert.notEqual(g1.url, g2.url);
  });
});

describe("ggufUrl", () => {
  it("resolves object shape", () => {
    assert.equal(ggufUrl({ url: "/models/m.ab12cd34.gguf", sha256: "x" }), "/models/m.ab12cd34.gguf");
  });
  it("accepts legacy string shape", () => {
    assert.equal(ggufUrl("/models/m.gguf"), "/models/m.gguf");
  });
  it("null-safe", () => {
    assert.equal(ggufUrl(null), null);
    assert.equal(ggufUrl(undefined), null);
    assert.equal(ggufUrl({}), null);
  });
});
