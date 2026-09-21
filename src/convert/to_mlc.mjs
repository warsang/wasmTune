// wasmtune — convert merged weights to browser formats.
// All external tools are optional and loud-skip when missing:
//   - MLC: `mlc_llm` (convert_weight -> gen_config -> compile) for WebLLM.
//     Install: pip install --pre -f https://mlc.ai/wheels mlc_llm_nightly_cpu mlc_ai_nightly_cpu
//   - GGUF: llama.cpp convert_hf_to_gguf.py + llama-quantize (wllama/Ollama).
//   - ONNX: `optimum-cli export onnx` for Transformers.js fallback.
// The manifest lets <site-chat> pick the best available artifact.

import { existsSync, createReadStream } from "node:fs";
import { mkdir, writeFile, rename, rm } from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

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

export async function convertModel({ mergedDir, webDir, modelId, quantization = "q4f16_1", device = "webgpu", convTemplate = "auto", chat = {} } = {}) {
  await mkdir(webDir, { recursive: true });
  const manifest = {
    base: modelId,
    created: new Date().toISOString(),
    artifacts: {},
    notes: [],
  };
  // Widget decoding hints (generic): template kwargs for hybrid-reasoning
  // models, generation guardrails. Only the serializable subset is stored.
  if (chat && typeof chat === "object") {
    const { templateKwargs, temperature, repetitionPenalty, presencePenalty, frequencyPenalty, maxTokens, topP } = chat;
    const chatOut = {};
    if (templateKwargs && typeof templateKwargs === "object") chatOut.templateKwargs = templateKwargs;
    for (const [k, v] of Object.entries({ temperature, repetitionPenalty, presencePenalty, frequencyPenalty, maxTokens, topP })) {
      if (Number.isFinite(Number(v))) chatOut[k] = Number(v);
    }
    if (Object.keys(chatOut).length) manifest.chat = chatOut;
  }

  // 1. MLC — the WebLLM primary artifact.
  // Documented flow: convert_weight -> gen_config -> compile.
  const mlc = (await which("mlc_llm")) ?? (await pythonHasMlc());
  if (mlc && mergedDir && existsSync(mergedDir)) {
    const mlcDir = path.join(webDir, "mlc");
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
      manifest.artifacts.mlc = { dir: mlcDir, config: cfgPath, lib: libOut, quantization, device, buildId };
      manifest.version ??= buildId;
    } catch (e) {
      manifest.notes.push(`mlc failed: ${String(e.message ?? e).slice(0, 300)}`);
    }
  } else {
    manifest.notes.push("mlc skipped: mlc_llm not installed (pip install --pre -f https://mlc.ai/wheels mlc_llm_nightly_cpu mlc_ai_nightly_cpu)");
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
      const f16 = path.join(webDir, "model.f16.gguf");
      await execAsync("python3", [converter, mergedDir, "--outfile", f16, "--outtype", "f16"]);
      const out = path.join(webDir, "model.Q4_K_M.gguf");
      await execAsync(quantize, [f16, out, "Q4_K_M"]);
      await rm(f16, { force: true }); // 8GB intermediate; reproducible from mergedDir
      const g = await hashAndRename(out, webDir);
      manifest.artifacts.gguf = { url: g.url, sha256: g.sha256, bytes: g.bytes };
      manifest.version = g.short;
    } catch (e) {
      manifest.notes.push(`gguf failed: ${String(e.message ?? e).slice(0, 300)}`);
    }
  } else {
    manifest.notes.push("gguf skipped: need llama-quantize on PATH and LLAMA_CPP_DIR pointing at a llama.cpp checkout");
  }

  // 3. ONNX — Transformers.js fallback.
  const optimum = await which("optimum-cli");
  if (optimum) {
    try {
      const out = path.join(webDir, "onnx");
      await execAsync(optimum, ["export", "onnx", "--model", mergedDir, out]);
      manifest.artifacts.onnx = out;
    } catch (e) {
      manifest.notes.push(`onnx skipped: ${e.message}`);
    }
  } else {
    manifest.notes.push("onnx skipped: optimum-cli not installed (pip install optimum[onnxruntime] for Transformers.js output)");
  }

  const manifestPath = path.join(webDir, "model-manifest.json");
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  return { manifestPath, manifest };
}
