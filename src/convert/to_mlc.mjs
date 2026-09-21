// wasmtune — convert merged weights to browser formats.
// All external tools are optional and loud-skip when missing:
//   - MLC: `mlc_llm` (convert_weight -> gen_config -> compile) for WebLLM.
//     Install: pip install --pre -f https://mlc.ai/wheels mlc_llm_nightly_cpu mlc_ai_nightly_cpu
//   - GGUF: llama.cpp convert_hf_to_gguf.py + llama-quantize (wllama/Ollama).
//   - ONNX: `optimum-cli export onnx` for Transformers.js fallback.
// The manifest lets <site-chat> pick the best fitting artifact per device.

import { existsSync, createReadStream } from "node:fs";
import { mkdir, writeFile, rename, rm } from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { slugifyModel } from "../models.mjs";

const execAsync = promisify(execFile);

// Content-hash a file (streaming — artifacts are gigabytes) and rename it to
// `<stem>.<8hex><ext>`, e.g. `model.Q4_K_M.gguf` ->
// `model.Q4_K_M.a1b2c3d4.gguf`. A new build always yields a new URL, so
// browser caches (wllama IndexedDB, service workers) can never serve stale
// weights after a rebuild. Returns {url, sha256, bytes} with url web-relative.
export async function hashAndRename(filePath, webDir) {
  const sha256 = await new Promise((resolve, reject) => {
    const h = crypto.createHash("sha256");
    const s = createReadStream(filePath);
    s.on("data", (d) => h.update(d));
    s.on("end", () => resolve(h.digest("hex")));
    s.on("error", reject);
  });
  const { size } = await import("node:fs/promises").then((m) => m.stat(filePath));
  const short = sha256.slice(0, 8);
  const dir = path.dirname(filePath);
  const ext = path.extname(filePath);
  const stem = path.basename(filePath, ext);
  const hashed = path.join(dir, `${stem}.${short}${ext}`);
  if (hashed !== filePath) await rename(filePath, hashed);
  return { url: "/" + path.relative(webDir, hashed).split(path.sep).join("/"), sha256, bytes: size, short };
}

async function which(cmd) {
  try {
    const { stdout } = await execAsync(process.platform === "win32" ? "where" : "which", [cmd]);
    return stdout.trim().split("\n")[0] ?? null;
  } catch {
    return null;
  }
}

// `python -m mlc_llm` works even when no mlc_llm binary is on PATH.
async function pythonHasMlc() {
  for (const py of ["python3", "python"]) {
    try {
      await execAsync(py, ["-m", "mlc_llm", "--help"]);
      return "python-mlc";
    } catch {
      /* try next */
    }
  }
  return null;
}

// Serializable chat hints (subset of config.chat / entry.chat).
function chatHints(chat) {
  if (!chat || typeof chat !== "object") return null;
  const { templateKwargs, temperature, repetitionPenalty, presencePenalty, frequencyPenalty, maxTokens, topP } = chat;
  const out = {};
  if (templateKwargs && typeof templateKwargs === "object") out.templateKwargs = templateKwargs;
  for (const [k, v] of Object.entries({ temperature, repetitionPenalty, presencePenalty, frequencyPenalty, maxTokens, topP })) {
    if (Number.isFinite(Number(v))) out[k] = Number(v);
  }
  return Object.keys(out).length ? out : null;
}

// Build one manifest entry for an untrained (pretrained) allowlist model:
// served straight from its public browser artifacts, no local conversion.
export function pretrainedEntry({ model, label = null, gguf = null, chat = null, requirements = null, onnx = null, webllm = null }) {
  const artifacts = {};
  if (webllm) artifacts.webllm = webllm;
  if (onnx) artifacts.onnx = onnx;
  if (gguf) artifacts.gguf = gguf;
  return {
    id: slugifyModel(model),
    label: label ?? model,
    base: model,
    source: "pretrained",
    artifacts,
    ...(chatHints(chat) ? { chat: chatHints(chat) } : {}),
    ...(requirements ? { requirements } : {}),
  };
}

export async function convertModel({
  mergedDir,
  webDir,
  modelId,
  quantization = "q4f16_1",
  device = "webgpu",
  convTemplate = "auto",
  chat = {},
  subdir = null, // per-entry artifact directory (multi-model builds)
  label = null,
  source = "tuned",
  requirements = null,
  writeManifest = true,
} = {}) {
  await mkdir(webDir, { recursive: true });
  const outDir = subdir ? path.join(webDir, subdir) : webDir;
  const entry = {
    id: slugifyModel(modelId),
    label: label ?? modelId,
    base: modelId,
    source,
    artifacts: {},
    ...(chatHints(chat) ? { chat: chatHints(chat) } : {}),
    ...(requirements ? { requirements } : {}),
  };
  const notes = [];
  const rel = (p) => "/" + path.relative(webDir, p).split(path.sep).join("/");

  // 1. MLC — the WebLLM primary artifact.
  // Documented flow: convert_weight -> gen_config -> compile.
  const mlc = (await which("mlc_llm")) ?? (await pythonHasMlc());
  if (mlc && mergedDir && existsSync(mergedDir)) {
    const mlcDir = path.join(outDir, "mlc");
    try {
      await mkdir(mlcDir, { recursive: true });
      const run = typeof mlc === "string" && mlc !== "python-mlc"
        ? (args) => execAsync(mlc, args)
        : (args) => execAsync(process.execPath, ["-m", "mlc_llm", ...args]);
      await run(["convert_weight", mergedDir, "--quantization", quantization, "-o", mlcDir]);
      const genArgs = ["gen_config", mergedDir, "--quantization", quantization, "-o", mlcDir];
      if (convTemplate !== "auto") genArgs.push("--conv-template", convTemplate);
      await run(genArgs);
      const cfgPath = path.join(mlcDir, "mlc-chat-config.json");
      const libOut = path.join(mlcDir, "lib.wasm");
      await run(["compile", cfgPath, "--device", device, "-o", libOut]);
      const buildId = new Date().toISOString().replace(/[-:.]/g, "").slice(0, 15);
      entry.artifacts.mlc = { config: rel(cfgPath), lib: rel(libOut), quantization, device, buildId };
    } catch (e) {
      notes.push(`mlc failed: ${String(e.message ?? e).slice(0, 300)}`);
    }
  } else if (source !== "pretrained") {
    notes.push("mlc skipped: mlc_llm not installed (pip install --pre -f https://mlc.ai/wheels mlc_llm_nightly_cpu mlc_ai_nightly_cpu)");
  }

  // 2. GGUF (llama.cpp) — feeds Ollama + wllama fallback.
  // Needs convert_hf_to_gguf.py (from a llama.cpp checkout, LLAMA_CPP_DIR)
  // plus the llama-quantize binary.
  const quantize = await which("llama-quantize");
  const converter = process.env.LLAMA_CPP_DIR
    ? path.join(process.env.LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    : null;
  if (quantize && converter && existsSync(converter) && mergedDir && existsSync(mergedDir)) {
    try {
      const f16 = path.join(outDir, "model.f16.gguf");
      await execAsync("python3", [converter, mergedDir, "--outfile", f16, "--outtype", "f16"]);
      const out = path.join(outDir, "model.Q4_K_M.gguf");
      await execAsync(quantize, [f16, out, "Q4_K_M"]);
      await rm(f16, { force: true }); // 8GB intermediate; reproducible from mergedDir
      const g = await hashAndRename(out, webDir);
      entry.artifacts.gguf = { url: g.url, sha256: g.sha256, bytes: g.bytes };
    } catch (e) {
      notes.push(`gguf failed: ${String(e.message ?? e).slice(0, 300)}`);
    }
  } else if (source !== "pretrained") {
    notes.push("gguf skipped: need llama-quantize on PATH and LLAMA_CPP_DIR pointing at a llama.cpp checkout");
  }

  // 3. ONNX — Transformers.js fallback.
  const optimum = await which("optimum-cli");
  if (optimum && mergedDir && existsSync(mergedDir)) {
    try {
      const out = path.join(outDir, "onnx");
      await execAsync(optimum, ["export", "onnx", "--model", mergedDir, out]);
      entry.artifacts.onnx = rel(out);
    } catch (e) {
      notes.push(`onnx skipped: ${e.message}`);
    }
  } else if (source !== "pretrained") {
    notes.push("onnx skipped: optimum-cli not installed (pip install optimum[onnxruntime] for Transformers.js output)");
  }

  const version = entry.artifacts.gguf?.sha256?.slice(0, 8)
    ?? entry.artifacts.mlc?.buildId
    ?? new Date().toISOString().replace(/[-:.]/g, "").slice(0, 15);
  const manifest = {
    base: modelId,
    created: new Date().toISOString(),
    version,
    models: [entry],
    artifacts: entry.artifacts, // legacy mirror of models[0]
    notes,
  };
  if (entry.chat) manifest.chat = entry.chat; // legacy chat location

  let manifestPath = null;
  if (writeManifest) {
    manifestPath = path.join(webDir, "model-manifest.json");
    await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  }
  return { manifestPath, manifest, entry, notes };
}

// Write a combined multi-model manifest (models[] + legacy mirror of the
// primary entry + flattened notes).
export async function writeModelsManifest({ webDir, entries, notes = [] }) {
  await mkdir(webDir, { recursive: true });
  const primary = entries.find((e) => e.source === "tuned") ?? entries[0] ?? null;
  const version = primary?.artifacts?.gguf?.sha256?.slice(0, 8)
    ?? primary?.artifacts?.mlc?.buildId
    ?? new Date().toISOString().replace(/[-:.]/g, "").slice(0, 15);
  const manifest = {
    version,
    created: new Date().toISOString(),
    models: entries,
    ...(primary ? { base: primary.base ?? null, artifacts: primary.artifacts, ...(primary.chat ? { chat: primary.chat } : {}) } : {}),
    notes,
  };
  const manifestPath = path.join(webDir, "model-manifest.json");
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  return { manifestPath, manifest };
}
