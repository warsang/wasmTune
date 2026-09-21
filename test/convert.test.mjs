import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, stat, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { hashAndRename, pretrainedEntry, writeModelsManifest } from "../src/convert/to_mlc.mjs";
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

describe("multi-model manifest assembly", () => {
  it("builds pretrained entries from allowlist artifacts", () => {
    const e = pretrainedEntry({
      model: "Qwen/Qwen3-0.6B",
      webllm: "Qwen3-0.6B-q4f16_1-MLC",
      onnx: "onnx-community/Qwen3-0.6B-ONNX",
      chat: { temperature: 0.2, ignored: "x" },
    });
    assert.equal(e.id, "qwen3-0.6b");
    assert.equal(e.source, "pretrained");
    assert.equal(e.artifacts.webllm, "Qwen3-0.6B-q4f16_1-MLC");
    assert.equal(e.artifacts.onnx, "onnx-community/Qwen3-0.6B-ONNX");
    assert.equal(e.artifacts.gguf, undefined);
    assert.deepEqual(e.chat, { temperature: 0.2 });
  });

  it("writes models[] plus a legacy mirror of the primary entry", async () => {
    const webDir = await mkdtemp(path.join(tmpdir(), "manifest-"));
    const tuned = {
      id: "gemma-4-e4b", label: null, base: "google/gemma-4-E4B-it", source: "tuned",
      artifacts: { gguf: { url: "/models/gemma.abcdef12.gguf", sha256: "abcdef12", bytes: 10 } },
      chat: { templateKwargs: { enable_thinking: false } },
    };
    const pre = pretrainedEntry({ model: "Qwen/Qwen3-0.6B", webllm: "Qwen3-0.6B-q4f16_1-MLC", onnx: "onnx-community/Qwen3-0.6B-ONNX" });
    const { manifestPath, manifest } = await writeModelsManifest({ webDir, entries: [tuned, pre], notes: ["note-a"] });
    assert.equal(manifest.models.length, 2);
    assert.equal(manifest.version, "abcdef12");
    assert.equal(manifest.base, "google/gemma-4-E4B-it");
    assert.deepEqual(manifest.artifacts, tuned.artifacts);
    assert.deepEqual(manifest.chat, tuned.chat);
    assert.deepEqual(manifest.notes, ["note-a"]);
    const onDisk = JSON.parse(await readFile(manifestPath, "utf8"));
    assert.equal(onDisk.models[1].id, "qwen3-0.6b");
  });
});
