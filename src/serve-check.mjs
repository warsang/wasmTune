// wasmtune — serving validation (generic).
//
// Catches "model trains fine but never loads in the browser" BEFORE deploy:
//   1. arch-vs-runtime: the GGUF's general.architecture must appear in the
//      installed @wllama/wllama WASM binary (exact manual check, productized).
//   2. size policy: single-file artifacts >2GB warn (tab fetch/OOM risk) and
//      recommend llama-gguf-split sharding.
//   3. URL resolution: every manifest artifact URL must resolve to a real
//      file (mount-relative or absolute), else the widget 404s at load.
//   4. giant tensors: any single tensor >1.5B params fails — splits can't
//      divide one tensor, and multi-GB blob slices fail in the browser
//      (NotReadableError). Fix by targeted requant (llama-quantize
//      --tensor-type) or a smaller base model.

import { existsSync, openSync, readSync, closeSync, statSync, createReadStream } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { manifestEntries } from "./chat/hardware.mjs";

export const SINGLE_FILE_WARN_BYTES = 2 * 1024 ** 3;
// Browser blob slices fail around ~2GB (saw NotReadableError); a tensor's
// byte size (not param count) is what matters — a 2.8B-param tensor at Q2_K
// is 882MB and fine, at Q4_K it is 2.3GB and fatal.
export const GIANT_TENSOR_ERROR_BYTES = 2 * 1024 ** 3;
export const GIANT_TENSOR_WARN_BYTES = 1 * 1024 ** 3;

// Approx bytes/param by GGUF type enum.
export const GGUF_TYPE_BPP = {
  0: 4, 1: 2, 30: 2, 2: 0.56, 3: 0.63, 6: 0.65, 7: 0.7, 8: 1.05,
  10: 0.33, 11: 0.44, 12: 0.56, 13: 0.65, 14: 0.75, 15: 1.05,
  16: 0.31, 17: 0.33, 18: 0.39, 19: 0.2, 20: 0.5, 21: 0.44,
  22: 0.33, 23: 0.53, 24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 29: 0.22,
};

// Minimal GGUF KV reader: just enough to find general.architecture.
// Layout: magic(4) version(u32) n_tensors(u64) n_kv(u64), then KVs:
// key = u64len+bytes, type u32, value. Only string values are decoded;
// anything else is skipped by type. Reads at most `cap` bytes.
export function readGgufArchitecture(filePath, { cap = 262144 } = {}) {
  const fd = openSync(filePath, "r");
  try {
    const st = statSync(filePath);
    const buf = Buffer.alloc(Math.min(cap, st.size || cap));
    readSync(fd, buf, 0, buf.length, 0);
    if (buf.subarray(0, 4).toString("latin1") !== "GGUF") {
      throw new Error("not a GGUF file (bad magic)");
    }
    let off = 4 + 4 + 8; // magic + version + n_tensors
    const nKv = Number(buf.readBigUInt64LE(off));
    off += 8;
    const u64 = () => {
      const v = Number(buf.readBigUInt64LE(off));
      off += 8;
      return v;
    };
    const u32 = () => {
      const v = buf.readUInt32LE(off);
      off += 4;
      return v;
    };
    const raw = (n) => {
      const b = buf.subarray(off, off + n);
      off += n;
      return b;
    };
    const str = () => {
      const n = u64();
      if (off + n > buf.length) throw new Error("general.architecture beyond read cap");
      return raw(n).toString("utf8");
    };
    const skipValue = (type) => {
      // GGUF metadata types: 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,
      // 8 string,9 array,10 u64,11 i64,12 f64
      const sizes = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
      if (type === 8) {
        str();
        return;
      }
      if (type === 9) {
        const etype = u32();
        const len = u64();
        if (etype === 8) {
          for (let i = 0; i < len; i++) str();
          return;
        }
        const s = sizes[etype];
        if (s === undefined) throw new Error(`unsupported array etype ${etype}`);
        off += s * Number(len);
        return;
      }
      const s = sizes[type];
      if (s === undefined) throw new Error(`unsupported metadata type ${type}`);
      off += s;
    };
    for (let i = 0; i < nKv; i++) {
      const key = str();
      const type = u32();
      if (key === "general.architecture") {
        if (type !== 8) throw new Error("general.architecture is not a string");
        return str();
      }
      skipValue(type);
    }
    throw new Error("general.architecture not found in metadata");
  } finally {
    closeSync(fd);
  }
}

// Scan a GGUF's tensor index (first shard holds it for splits) and return
// the largest tensors by parameter count: [{name, params}]. Tensor info
// layout per entry: name(string) n_dims(u32) dims[n_dims](u64) dtype(u32)
// offset(u64). Reads header+index only, streaming-safe for multi-GB files.
export function largestGgufTensors(filePath, { top = 5, cap = 64 * 1024 * 1024 } = {}) {
  const fd = openSync(filePath, "r");
  try {
    const st = statSync(filePath);
    const buf = Buffer.alloc(Math.min(cap, st.size || cap));
    readSync(fd, buf, 0, buf.length, 0);
    if (buf.subarray(0, 4).toString("latin1") !== "GGUF") {
      throw new Error("not a GGUF file (bad magic)");
    }
    let off = 4 + 4;
    const nTensors = Number(buf.readBigUInt64LE(off));
    off += 8;
    const nKv = Number(buf.readBigUInt64LE(off));
    off += 8;
    const u64 = () => {
      const v = Number(buf.readBigUInt64LE(off));
      off += 8;
      return v;
    };
    const u32 = () => {
      const v = buf.readUInt32LE(off);
      off += 4;
      return v;
    };
    const need = (n, what) => {
      if (off + n > buf.length) throw new Error(`${what} beyond read cap (${(cap / 1e6).toFixed(0)}MB)`);
    };
    const str = () => {
      const n = u64();
      need(n, "string");
      const s = buf.subarray(off, off + n).toString("utf8");
      off += n;
      return s;
    };
    // Skip metadata KVs (same type table as readGgufArchitecture).
    const sizes = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
    const skipValue = (type) => {
      if (type === 8) {
        str();
        return;
      }
      if (type === 9) {
        const etype = u32();
        const len = u64();
        if (etype === 8) {
          for (let i = 0; i < len; i++) str();
          return;
        }
        const s = sizes[etype];
        if (s === undefined) throw new Error(`unsupported array etype ${etype}`);
        need(s * Number(len), "array value");
        off += s * Number(len);
        return;
      }
      const s = sizes[type];
      if (s === undefined) throw new Error(`unsupported metadata type ${type}`);
      need(s, "scalar value");
      off += s;
    };
    for (let i = 0; i < nKv; i++) {
      str();
      skipValue(u32());
    }
    // Tensor infos.
    const acc = [];
    for (let i = 0; i < nTensors; i++) {
      const name = str();
      const nDims = u32();
      need(nDims * 8, "tensor dims");
      let params = 1;
      for (let d = 0; d < nDims; d++) params *= u64();
      need(4, "tensor dtype");
      const dtype = u32();
      need(8, "tensor offset");
      off += 8; // offset
      const bpp = GGUF_TYPE_BPP[dtype] ?? 2;
      acc.push({ name, params, dtype, bytes: Math.round(params * bpp) });
    }
    acc.sort((a, b) => b.bytes - a.bytes);
    return acc.slice(0, top);
  } finally {
    closeSync(fd);
  }
}

// Stream-search a (large) WASM binary for an architecture token.
export async function wasmSupportsArch(wasmPath, arch) {
  const needle = Buffer.from(String(arch));
  let tail = Buffer.alloc(0);
  for await (const chunk of createReadStream(wasmPath, { highWaterMark: 1 << 20 })) {
    const hay = Buffer.concat([tail, chunk]);
    if (hay.includes(needle)) return true;
    tail = hay.subarray(Math.max(0, hay.length - needle.length + 1));
  }
  return false;
}

export function resolveWllamaWasm(cwd = process.cwd()) {
  try {
    const require = createRequire(import.meta.url);
    const pkgPath = require.resolve("@wllama/wllama/package.json", { paths: [cwd] });
    const wasm = path.join(path.dirname(pkgPath), "esm", "wasm", "wllama.wasm");
    return existsSync(wasm) ? wasm : null;
  } catch {
    return null;
  }
}

export function shardSiblings(filePath) {
  // llama-gguf-split names shards model-00001-of-00005.gguf.
  const m = path.basename(filePath).match(/^(.*)-(\d{5})-of-(\d{5})(\.[^.]+)$/);
  if (!m) return null;
  return { stem: m[1], index: m[2], total: m[3], ext: m[4] };
}

export async function checkServing({ cwd = process.cwd(), config, fetchImpl = globalThis.fetch } = {}) {
  const errors = [];
  const warnings = [];
  const details = {};
  const webDir = path.resolve(cwd, config?.output?.webDir ?? "./public/models");
  const manifestPath = path.join(webDir, "model-manifest.json");
  if (!existsSync(manifestPath)) {
    return { ok: false, errors: [`no model-manifest.json at ${manifestPath} — run "wasmtune convert" first`], warnings, details };
  }
  let manifest;
  try {
    const { readFile } = await import("node:fs/promises");
    manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  } catch (e) {
    return { ok: false, errors: [`unreadable manifest: ${e.message}`], warnings, details };
  }
  details.manifest = manifestPath;
  details.version = manifest.version ?? null;

  const entries = manifestEntries(manifest);
  const multi = entries.length > 1;
  const artifacts = [];
  entries.forEach((entry, i) => {
    const a = entry.artifacts ?? {};
    const tag = multi ? `[${entry.id ?? `model-${i}`}] ` : "";
    const hasAny = ["gguf", "mlc", "onnx", "webllm"].some((k) => a[k]);
    if (!hasAny) {
      errors.push(`${tag}entry "${entry.id ?? i}" has no browser artifacts (gguf/mlc/onnx/webllm) — this tier can never load`);
    }
    if (a.gguf) {
      const g = a.gguf;
      artifacts.push({ kind: "gguf", tag, first: i === 0, entryId: entry.id, url: typeof g === "string" ? g : g?.url });
    }
    if (a.mlc) artifacts.push({ kind: "mlc", tag, first: i === 0, entryId: entry.id, url: a.mlc.config });
    if (a.onnx) {
      const id = typeof a.onnx === "string" ? a.onnx : (a.onnx?.id ?? a.onnx?.repo ?? null);
      // Pretrained entries reference an HF/ONNX repo id, not a served file.
      const remoteId = !!id && !id.startsWith("/") && !/^https?:\/\//.test(id);
      artifacts.push({ kind: "onnx", tag, first: i === 0, entryId: entry.id, url: id, remoteId });
    }
    if (a.webllm) details[`webllm${i ? `#${entry.id}` : ""}`] = a.webllm; // prebuilt id: remote, nothing local to check
  });

  const wasmPath = resolveWllamaWasm(cwd);
  details.wllamaWasm = wasmPath;
  if (!wasmPath) warnings.push("@wllama/wllama not installed here — skipping arch-vs-runtime check");

  for (const a of artifacts) {
    const p = a.tag ?? "";
    const key = (base) => (a.first ? base : `${base}#${a.entryId}`);
    if (!a.url) {
      errors.push(`${p}${a.kind} artifact has no URL`);
      continue;
    }
    // Remote model ids (pretrained ONNX repos): nothing local to validate.
    if (a.remoteId) {
      details[key(`${a.kind}Id`)] = a.url;
      continue;
    }
    // 3. URL resolution.
    let local = null;
    if (/^https?:\/\//.test(a.url)) {
      try {
        const res = await fetchImpl(a.url, { method: "HEAD" });
        if (!res.ok) errors.push(`${p}${a.kind} URL does not resolve: ${a.url} (HTTP ${res.status})`);
        else details[key(`${a.kind}Url`)] = "remote-ok";
      } catch (e) {
        errors.push(`${p}${a.kind} URL fetch failed: ${a.url} (${e.message})`);
      }
      continue;
    }
    // Mount-relative URLs (/models/…, /static/…) resolve against webDir by
    // progressively stripping leading segments: the manifest doesn't know
    // where the consumer mounts the models dir.
    const segs = a.url.split("?")[0].split("/").filter(Boolean);
    const candidates = [path.join(cwd, segs.join(path.sep))];
    for (let i = 0; i < segs.length; i++) {
      candidates.push(path.join(webDir, ...segs.slice(i)));
    }
    local = candidates.find((c) => existsSync(c) && statSync(c).isFile()) ?? null;
    if (!local) {
      errors.push(`${p}${a.kind} URL does not resolve to a file: ${a.url} (tried ${candidates.join(", ")})`);
      continue;
    }
    details[key(`${a.kind}File`)] = local;

    if (a.kind !== "gguf") continue;
    const st = statSync(local);
    details[key("ggufBytes")] = st.size;
    // Sharded artifacts: every sibling must exist (wllama fetches them all).
    const split = shardSiblings(local);
    if (split) {
      const dir = path.dirname(local);
      const missing = [];
      let total = 0;
      for (let i = 1; i <= Number(split.total); i++) {
        const name = `${split.stem}-${String(i).padStart(5, "0")}-of-${split.total}${split.ext}`;
        const fp = path.join(dir, name);
        if (!existsSync(fp)) missing.push(name);
        else total += statSync(fp).size;
      }
      details[key("ggufShards")] = { present: Number(split.total) - missing.length, total: Number(split.total), bytes: total };
      if (missing.length) errors.push(`${p}gguf split missing ${missing.length} shard(s): ${missing.slice(0, 3).join(", ")}${missing.length > 3 ? "…" : ""}`);
    }
    // 2. Size policy.
    if (!shardSiblings(local) && st.size > SINGLE_FILE_WARN_BYTES) {
      warnings.push(
        `${p}gguf is ${(st.size / 1e9).toFixed(2)} GB single-file — tab fetch/OOM risk; ` +
          `consider llama-gguf-split sharding (wllama loads *-00001-of-*.gguf natively)`,
      );
    }
    // 1. Arch-vs-runtime.
    let arch;
    try {
      arch = readGgufArchitecture(local);
      details[key("ggufArch")] = arch;
    } catch (e) {
      errors.push(`${p}cannot read GGUF architecture from ${local}: ${e.message}`);
      continue;
    }
    if (wasmPath) {
      const supported = await wasmSupportsArch(wasmPath, arch);
      details[key("runtimeSupportsArch")] = supported;
      if (!supported) {
        errors.push(
          `${p}runtime mismatch: GGUF architecture "${arch}" not found in installed wllama WASM — ` +
            `the model will fail to load in the browser (upgrade @wllama/wllama or pick a supported arch)`,
        );
      }
    }
    // 4. Giant single tensors (splits can't divide one tensor; multi-GB
    // blob slices fail in the browser with NotReadableError). Splits carry
    // only their own tensors in the index, so scan every present shard.
    const splitForScan = shardSiblings(local);
    let scanFiles = [local];
    if (splitForScan) {
      const dir = path.dirname(local);
      scanFiles = [];
      for (let i = 1; i <= Number(splitForScan.total); i++) {
        const fp = path.join(dir, `${splitForScan.stem}-${String(i).padStart(5, "0")}-of-${splitForScan.total}${splitForScan.ext}`);
        if (existsSync(fp)) scanFiles.push(fp);
      }
    }
    try {
      const biggest = scanFiles
        .flatMap((fp) => {
          try {
            return largestGgufTensors(fp, { top: 3 });
          } catch {
            return [];
          }
        })
        .sort((a, b) => b.bytes - a.bytes)
        .slice(0, 3);
      details[key("largestTensors")] = biggest.map((t) => ({ name: t.name, bytes: t.bytes }));
      for (const t of biggest) {
        const gb = (t.bytes / 1e9).toFixed(2);
        if (t.bytes > GIANT_TENSOR_ERROR_BYTES) {
          errors.push(
            `${p}giant tensor "${t.name}" is ~${gb} GB in one tensor — ` +
              `no split can divide it and browser blob reads fail (saw NotReadableError). ` +
              `Requantize with: llama-quantize --tensor-type <name>=q2_k (or a smaller base model)`,
          );
        } else if (t.bytes > GIANT_TENSOR_WARN_BYTES) {
          warnings.push(
            `${p}large tensor "${t.name}" is ~${gb} GB — watch browser load memory`,
          );
        }
      }
    } catch (e) {
      warnings.push(`${p}tensor scan skipped: ${e.message}`);
    }
  }
  return { ok: errors.length === 0, errors, warnings, details };
}
