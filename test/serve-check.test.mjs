import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  readGgufArchitecture,
  wasmSupportsArch,
  shardSiblings,
  largestGgufTensors,
  checkServing,
  GIANT_TENSOR_ERROR_BYTES,
} from "../src/serve-check.mjs";

// Minimal valid GGUF: magic + version + 0 tensors + 1 KV (general.architecture).
function fakeGguf(arch) {
  const key = Buffer.from("general.architecture", "utf8");
  const val = Buffer.from(arch, "utf8");
  const head = Buffer.alloc(4 + 4 + 8 + 8);
  head.write("GGUF", 0, "latin1");
  head.writeUInt32LE(3, 4);
  head.writeBigUInt64LE(0n, 8); // n_tensors
  head.writeBigUInt64LE(1n, 16); // n_kv
  const kb = Buffer.alloc(8);
  kb.writeBigUInt64LE(BigInt(key.length), 0);
  const tb = Buffer.alloc(4);
  tb.writeUInt32LE(8, 0); // type string
  const vb = Buffer.alloc(8);
  vb.writeBigUInt64LE(BigInt(val.length), 0);
  return Buffer.concat([head, kb, key, tb, vb, val]);
}

describe("readGgufArchitecture", () => {
  it("reads general.architecture from a minimal GGUF", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "gguf-"));
    const f = path.join(dir, "m.gguf");
    await writeFile(f, fakeGguf("gemma4"));
    assert.equal(readGgufArchitecture(f), "gemma4");
  });

  it("rejects bad magic", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "gguf-"));
    const f = path.join(dir, "m.gguf");
    await writeFile(f, Buffer.from("NOPE____________"));
    assert.throws(() => readGgufArchitecture(f), /not a GGUF/);
  });
});

describe("largestGgufTensors", () => {
  // Minimal GGUF with 2 tensors: header + 1 KV + tensor infos.
  function fakeGgufTensors(tensors) {
    const parts = [];
    const head = Buffer.alloc(4 + 4 + 8 + 8);
    head.write("GGUF", 0, "latin1");
    head.writeUInt32LE(3, 4);
    head.writeBigUInt64LE(BigInt(tensors.length), 8);
    head.writeBigUInt64LE(1n, 16);
    parts.push(head);
    const pushStr = (s) => {
      const b = Buffer.from(s, "utf8");
      const lb = Buffer.alloc(8);
      lb.writeBigUInt64LE(BigInt(b.length), 0);
      parts.push(lb, b);
    };
    pushStr("general.architecture");
    const tb = Buffer.alloc(4);
    tb.writeUInt32LE(8, 0);
    parts.push(tb);
    pushStr("gemma4");
    for (const [name, dims] of tensors) {
      pushStr(name);
      const nb = Buffer.alloc(4);
      nb.writeUInt32LE(dims.length, 0);
      parts.push(nb);
      for (const d of dims) {
        const db = Buffer.alloc(8);
        db.writeBigUInt64LE(BigInt(d), 0);
        parts.push(db);
      }
      const tail = Buffer.alloc(4 + 8);
      tail.writeUInt32LE(0, 0); // dtype
      tail.writeBigUInt64LE(0n, 4); // offset
      parts.push(tail);
    }
    return Buffer.concat(parts);
  }

  it("ranks tensors by param count", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "gguf-"));
    const f = path.join(dir, "m.gguf");
    await writeFile(f, fakeGgufTensors([
      ["blk.0.attn_q.weight", [2560, 2560]],
      ["token_embd.weight", [2560, 262144]],
    ]));
    const top = largestGgufTensors(f, { top: 2 });
    assert.equal(top[0].name, "token_embd.weight");
    assert.equal(top[0].params, 2560 * 262144);
    assert.equal(top[1].name, "blk.0.attn_q.weight");
  });

  it("estimates byte size per tensor (Q2_K giant is fine, Q4_K fatal)", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "gguf-"));
    const f = path.join(dir, "m.gguf");
    await writeFile(f, fakeGgufTensors([["per_layer_token_embd.weight", [10752, 262144]]]));
    const top = largestGgufTensors(f, { top: 1 });
    // Default dtype in fixture is 0 (F32): 2.8B params * 4B = ~11GB -> error.
    assert.ok(top[0].bytes > GIANT_TENSOR_ERROR_BYTES, `${top[0].bytes}`);
  });

  it("checkServing errors on giant tensors", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "serve-"));
    const webDir = path.join(dir, "public", "models");
    await mkdir(webDir, { recursive: true });
    await writeFile(
      path.join(webDir, "big.gguf"),
      fakeGgufTensors([["per_layer_token_embd.weight", [10752, 262144]]]),
    );
    await writeFile(path.join(webDir, "model-manifest.json"), JSON.stringify({
      artifacts: { gguf: { url: "/models/big.gguf" } },
    }));
    const r = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(r.ok, false);
    assert.ok(r.errors.some((e) => e.includes("giant tensor") && e.includes("tensor-type")), JSON.stringify(r.errors));
  });
});

describe("wasmSupportsArch", () => {
  it("finds arch tokens incl. across chunk boundaries", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasm-"));
    const f = path.join(dir, "w.wasm");
    await writeFile(f, Buffer.concat([Buffer.alloc(2 ** 20, 0x41), Buffer.from("llama_model_gemma4"), Buffer.alloc(100, 0x42)]));
    assert.equal(await wasmSupportsArch(f, "gemma4"), true);
    assert.equal(await wasmSupportsArch(f, "qwen3_5"), false);
  });
});

describe("sharded artifacts", () => {
  it("requires every sibling shard present", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "serve-"));
    const webDir = path.join(dir, "public", "models");
    await mkdir(webDir, { recursive: true });
    const { writeFile: wf } = await import("node:fs/promises");
    // Minimal GGUFs: real header in shard 1 (arch readable), stubs after.
    const key = Buffer.from("general.architecture", "utf8");
    const val = Buffer.from("gemma4", "utf8");
    const mkHead = () => {
      const h = Buffer.alloc(4 + 4 + 8 + 8);
      h.write("GGUF", 0, "latin1"); h.writeUInt32LE(3, 4);
      h.writeBigUInt64LE(0n, 8); h.writeBigUInt64LE(1n, 16);
      const kb = Buffer.alloc(8); kb.writeBigUInt64LE(BigInt(key.length), 0);
      const tb = Buffer.alloc(4); tb.writeUInt32LE(8, 0);
      const vb = Buffer.alloc(8); vb.writeBigUInt64LE(BigInt(val.length), 0);
      return Buffer.concat([h, kb, key, tb, vb, val]);
    };
    await wf(path.join(webDir, "m-00001-of-00002.gguf"), mkHead());
    await wf(path.join(webDir, "m-00002-of-00002.gguf"), Buffer.alloc(10));
    await wf(path.join(webDir, "model-manifest.json"), JSON.stringify({
      artifacts: { gguf: { url: "/models/m-00001-of-00002.gguf" } },
    }));
    // stub the wllama package resolution: point cwd at a fixture
    const wasmDir = path.join(dir, "node_modules", "@wllama", "wllama", "esm", "wasm");
    await mkdir(wasmDir, { recursive: true });
    await wf(path.join(wasmDir, "wllama.wasm"), Buffer.from("llama_model_gemma4"));
    await wf(path.join(dir, "node_modules", "@wllama", "wllama", "package.json"), JSON.stringify({ name: "@wllama/wllama" }));
    const ok = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(ok.ok, true, JSON.stringify(ok));
    assert.deepEqual(ok.details.ggufShards, { present: 2, total: 2, bytes: ok.details.ggufShards.bytes });
    // Remove a shard -> hard error.
    const { rm } = await import("node:fs/promises");
    await rm(path.join(webDir, "m-00002-of-00002.gguf"));
    const bad = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(bad.ok, false);
    assert.ok(bad.errors.some((e) => e.includes("missing 1 shard")), JSON.stringify(bad.errors));
  });
});

describe("shardSiblings", () => {
  it("parses split names, ignores single files", () => {
    assert.deepEqual(shardSiblings("m-00001-of-00005.gguf"), { stem: "m", index: "00001", total: "00005", ext: ".gguf" });
    assert.equal(shardSiblings("model.gguf"), null);
  });
});

describe("checkServing", () => {
  async function fixture({ arch = "gemma4", size = 100, wasmHasArch = true } = {}) {
    const dir = await mkdtemp(path.join(tmpdir(), "serve-"));
    const webDir = path.join(dir, "public", "models");
    await mkdir(webDir, { recursive: true });
    const ggufPath = path.join(webDir, "m.ab12cd34.gguf");
    const head = fakeGguf(arch);
    const padding = Buffer.alloc(Math.max(0, size - head.length), 0);
    await writeFile(ggufPath, Buffer.concat([head, padding]));
    await writeFile(path.join(webDir, "model-manifest.json"), JSON.stringify({
      version: "ab12cd34",
      artifacts: { gguf: { url: "/models/m.ab12cd34.gguf", sha256: "x", bytes: size } },
    }));
    const wasmDir = path.join(dir, "node_modules", "@wllama", "wllama", "esm", "wasm");
    await mkdir(wasmDir, { recursive: true });
    await writeFile(path.join(wasmDir, "wllama.wasm"), wasmHasArch ? Buffer.from("…llama_model_gemma4…") : Buffer.from("…llama_model_qwen3…"));
    await writeFile(path.join(dir, "node_modules", "@wllama", "wllama", "package.json"), JSON.stringify({ name: "@wllama/wllama" }));
    return { dir, webDir };
  }

  it("passes a supported arch with resolvable URL", async () => {
    const { dir } = await fixture();
    const r = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(r.ok, true, JSON.stringify(r));
    assert.equal(r.details.ggufArch, "gemma4");
    assert.equal(r.details.runtimeSupportsArch, true);
  });

  it("errors on arch-vs-runtime mismatch", async () => {
    const { dir } = await fixture({ arch: "gemma4", wasmHasArch: false });
    const r = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(r.ok, false);
    assert.ok(r.errors.some((e) => e.includes("runtime mismatch") && e.includes("gemma4")), JSON.stringify(r.errors));
  });

  it("warns on oversized single files", async () => {
    const { dir, webDir } = await fixture();
    // Sparse-extend to 3GB without writing bytes (APFS/tmpfs friendly).
    const { open } = await import("node:fs/promises");
    const fh = await open(path.join(webDir, "m.ab12cd34.gguf"), "r+");
    await fh.truncate(3 * 1024 ** 3);
    await fh.close();
    const r = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(r.ok, true, JSON.stringify(r));
    assert.ok(r.warnings.some((w) => w.includes("llama-gguf-split")), JSON.stringify(r.warnings));
  });

  it("errors when the manifest is missing", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "serve-"));
    const r = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(r.ok, false);
    assert.ok(r.errors[0].includes("model-manifest.json"));
  });

  it("resolves mount-relative subdirectory URLs against webDir", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "serve-"));
    const webDir = path.join(dir, "public", "models");
    const sub = path.join(webDir, "kf-gemma4-abc12345");
    await mkdir(sub, { recursive: true });
    await writeFile(path.join(sub, "m-00001-of-00002.gguf"), fakeGguf("gemma4"));
    await writeFile(path.join(sub, "m-00002-of-00002.gguf"), Buffer.alloc(10));
    await writeFile(path.join(webDir, "model-manifest.json"), JSON.stringify({
      artifacts: { gguf: { url: "/models/kf-gemma4-abc12345/m-00001-of-00002.gguf" } },
    }));
    const r = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(r.ok, true, JSON.stringify(r));
    assert.ok(r.details.ggufFile.endsWith(path.join("kf-gemma4-abc12345", "m-00001-of-00002.gguf")));
  });

  it("errors on unresolvable artifact URLs", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "serve-"));
    const webDir = path.join(dir, "public", "models");
    await mkdir(webDir, { recursive: true });
    await writeFile(path.join(webDir, "model-manifest.json"), JSON.stringify({
      artifacts: { gguf: { url: "/models/ghost.gguf" } },
    }));
    const r = await checkServing({ cwd: dir, config: { output: { webDir: "./public/models" } } });
    assert.equal(r.ok, false);
    assert.ok(r.errors.some((e) => e.includes("ghost.gguf")), JSON.stringify(r.errors));
  });
});
