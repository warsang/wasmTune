// wasmtune — small-model allowlist + hardware requirements.
//
// Only models that BOTH (a) fine-tune on consumer hardware via LoRA/QLoRA
// and (b) have a realistic browser inference path (WebLLM MLC artifact and/or
// ONNX community artifact) are accepted. This keeps `wasmtune train` working
// on an M-series Mac and on a single NVIDIA card, and `wasmtune convert` +
// <site-chat> able to serve the result over WebGPU.
//
// Each entry also carries the structured requirements the browser-side picker
// (src/chat/hardware.mjs) uses to choose a model tier per device:
//   vramBytes          runtime footprint of the quantized browser build
//   minDeviceMemoryGB  conservative device-memory floor for a load
//   tier               tiny | small | mid | large (biggest fitting tier wins)
//   mobileOk           acceptable on phones/tablets
//   cpuOk              usable without WebGPU (WASM CPU inference)

const RAW = [
  {
    hf: "HuggingFaceTB/SmolLM2-135M-Instruct",
    webllm: "SmolLM2-135M-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-135M-Instruct-ONNX",
    params: "135M", vram: "~78MB", vramBytes: 8.2e7, context: "2K",
    family: "SmolLM2", mobileOk: true, cpuOk: true,
  },
  {
    hf: "HuggingFaceTB/SmolLM2-360M-Instruct",
    webllm: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-360M-Instruct-ONNX",
    params: "360M", vram: "~210MB", vramBytes: 2.2e8, context: "2K",
    family: "SmolLM2", mobileOk: true, cpuOk: true,
  },
  {
    hf: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    webllm: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-1.7B-Instruct-ONNX",
    params: "1.7B", vram: "~1GB", vramBytes: 1.07e9, context: "2K",
    family: "SmolLM2", mobileOk: false, cpuOk: false,
  },
  {
    hf: "Qwen/Qwen2.5-0.5B-Instruct",
    webllm: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Qwen2.5-0.5B-Instruct-ONNX",
    params: "0.5B", vram: "~278MB", vramBytes: 2.9e8, context: "4K",
    family: "Qwen", mobileOk: true, cpuOk: true,
  },
  {
    hf: "Qwen/Qwen2.5-1.5B-Instruct",
    webllm: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Qwen2.5-1.5B-Instruct-ONNX",
    params: "1.5B", vram: "~868MB", vramBytes: 9.1e8, context: "4K",
    family: "Qwen", mobileOk: false, cpuOk: false,
  },
  {
    hf: "Qwen/Qwen3-0.6B",
    webllm: "Qwen3-0.6B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-0.6B-ONNX",
    params: "0.6B", vram: "~350MB", vramBytes: 3.7e8, context: "4K",
    family: "Qwen", mobileOk: true, cpuOk: true,
  },
  {
    hf: "Qwen/Qwen3-1.7B",
    webllm: "Qwen3-1.7B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-1.7B-ONNX",
    params: "1.7B", vram: "~1.1GB", vramBytes: 1.15e9, context: "4K",
    family: "Qwen", mobileOk: false, cpuOk: false,
  },
  {
    hf: "Qwen/Qwen3-4B-Instruct-2507",
    mlx: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    webllm: "Qwen3-4B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-4B-ONNX",
    params: "4B", vram: "~2.2GB", vramBytes: 2.36e9, context: "32K",
    family: "Qwen", mobileOk: false, cpuOk: false,
    note: "best fine-tuned SLM (distil-labs 12x8 study); non-thinking instruct",
  },
  {
    hf: "Qwen/Qwen3-4B",
    webllm: "Qwen3-4B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-4B-ONNX",
    params: "4B", vram: "~2.2GB", vramBytes: 2.36e9, context: "4K",
    family: "Qwen", mobileOk: false, cpuOk: false,
  },
  {
    hf: "meta-llama/Llama-3.2-1B-Instruct",
    webllm: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    params: "1B", vram: "~712MB", vramBytes: 7.5e8, context: "4K",
    family: "Llama", license: "llama-community (accept terms)",
    mobileOk: true, cpuOk: false,
  },
  {
    hf: "meta-llama/Llama-3.2-3B-Instruct",
    webllm: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Llama-3.2-3B-Instruct-ONNX",
    params: "3B", vram: "~1.76GB", vramBytes: 1.85e9, context: "4K",
    family: "Llama", license: "llama-community (accept terms)",
    mobileOk: false, cpuOk: false,
  },
  {
    hf: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    webllm: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
    onnx: "onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX",
    params: "1.1B", vram: "~400MB", vramBytes: 4.2e8, context: "2K",
    family: "TinyLlama", mobileOk: true, cpuOk: true,
  },
  {
    hf: "microsoft/Phi-3.5-mini-instruct",
    webllm: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    onnx: "microsoft/Phi-3-mini-4k-instruct-onnx-web",
    params: "3.8B", vram: "~2.1GB", vramBytes: 2.25e9, context: "4K",
    family: "Phi", mobileOk: false, cpuOk: false,
  },
  {
    hf: "google/gemma-2-2b-it",
    webllm: "gemma-2-2b-it-q4f16_1-MLC",
    onnx: "onnx-community/gemma-2-2b-it-ONNX",
    params: "2B", vram: "~1.44GB", vramBytes: 1.55e9, context: "4K",
    family: "Gemma", license: "gemma terms", mobileOk: false, cpuOk: false,
  },
  {
    hf: "google/gemma-3-1b-it",
    webllm: "gemma-3-1b-it-q4f16_1-MLC",
    onnx: "onnx-community/gemma-3-1b-it-ONNX",
    params: "1B", vram: "~700MB", vramBytes: 7.5e8, context: "8K",
    family: "Gemma", license: "gemma terms", mobileOk: true, cpuOk: false,
  },
  {
    hf: "Qwen/Qwen3.5-4B",
    mlx: "mlx-community/Qwen3.5-4B-MLX-4bit",
    gguf: "unsloth/Qwen3.5-4B-GGUF",
    params: "4B", vram: "~3GB", vramBytes: 3.2e9, context: "262K",
    family: "Qwen", mobileOk: false, cpuOk: false,
    note: "hybrid thinking (select non-thinking via chat template); strongest sub-10B reasoning",
  },
  {
    hf: "google/gemma-4-E4B-it",
    mlx: "mlx-community/gemma-4-e4b-it-4bit",
    gguf: "ggml-org/gemma-4-E4B-it-GGUF",
    params: "4.5B eff", vram: "~3GB", vramBytes: 3.2e9, context: "128K",
    family: "Gemma", mobileOk: false, cpuOk: false,
    // Converted GGUF is ~3.9GB; the embedded q2 embedding tensor plus KV
    // cache needs real headroom — keep the 6GB floor.
    minDeviceMemoryGB: 6,
    note: "disable thinking via enable_thinking=false; VLM arch (mlx_lm text path)",
  },
];

// Coarse tier by browser runtime footprint. Bigger = preferred when it fits.
export function tierForBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n < 5e8) return "tiny";
  if (n < 1.6e9) return "small";
  if (n < 4.5e9) return "mid";
  return "large";
}

export function minMemoryForBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n <= 5e8) return 1;
  if (n <= 1.6e9) return 2;
  if (n <= 3.2e9) return 4;
  return 8;
}

export const ALLOWLIST = RAW.map((m) => ({
  ...m,
  tier: m.tier ?? tierForBytes(m.vramBytes),
  minDeviceMemoryGB: m.minDeviceMemoryGB ?? minMemoryForBytes(m.vramBytes),
}));

// Larger models that technically work but need a discrete GPU and a
// desktop browser. Gated behind --allow-large.
export const LARGE_MODELS = new Set([
  "Qwen/Qwen2.5-7B-Instruct",
  "Qwen/Qwen3-8B",
  "meta-llama/Llama-3.1-8B-Instruct",
  "mistralai/Mistral-7B-Instruct-v0.3",
  "google/gemma-2-9b-it",
]);

const byHf = new Map(ALLOWLIST.map((m) => [m.hf.toLowerCase(), m]));
const byBrowserId = new Map();
for (const m of ALLOWLIST) {
  for (const key of ["webllm", "onnx", "gguf", "mlx"]) {
    if (m[key]) byBrowserId.set(String(m[key]).toLowerCase(), m);
  }
  // Also match the last path segment (e.g. "SmolLM2-135M-Instruct-q4f16_1-MLC").
  if (m.webllm) byBrowserId.set(m.webllm.toLowerCase().split("/").pop(), m);
  if (m.gguf) byBrowserId.set(String(m.gguf).toLowerCase().split("/").pop(), m);
}

export function lookupModel(hf) {
  return byHf.get(String(hf ?? "").toLowerCase()) ?? null;
}

// Manifest entries may reference the model by HF id, WebLLM id, ONNX repo, or
// GGUF repo — all resolve to the same allowlist record.
export function lookupByBrowserId(id) {
  return byBrowserId.get(String(id ?? "").toLowerCase()) ?? null;
}

// Directory/id slug for a model: "google/gemma-4-E4B-it" -> "gemma-4-e4b-it".
export function slugifyModel(id) {
  return String(id ?? "")
    .toLowerCase()
    .split("/")
    .pop()
    .replace(/[^a-z0-9.]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// Structured hardware requirements for a model id (HF, WebLLM, ONNX, GGUF) or
// manifest entry base. Returns null when nothing is known — callers then fall
// back to artifact byte sizes.
export function requirementsFor(id) {
  const m = lookupModel(id) ?? lookupByBrowserId(id) ?? null;
  if (m) {
    return {
      hf: m.hf,
      params: m.params,
      vramBytes: m.vramBytes,
      minDeviceMemoryGB: m.minDeviceMemoryGB,
      tier: m.tier,
      needsWebGPU: !m.cpuOk,
      mobileOk: !!m.mobileOk,
      cpuOk: !!m.cpuOk,
      context: m.context,
      source: "lib",
    };
  }
  if (LARGE_MODELS.has(String(id))) {
    return {
      hf: String(id),
      params: "7-9B",
      vramBytes: 8e9,
      minDeviceMemoryGB: 8,
      tier: "large",
      needsWebGPU: true,
      mobileOk: false,
      cpuOk: false,
      context: null,
      source: "lib-large",
    };
  }
  return null;
}

export function isAllowedModel(hf, { allowLarge = false } = {}) {
  if (lookupModel(hf)) return true;
  if (allowLarge && LARGE_MODELS.has(String(hf))) return true;
  return false;
}

export function assertAllowedModel(hf, { allowLarge = false } = {}) {
  if (isAllowedModel(hf, { allowLarge })) return lookupModel(hf) ?? { hf };
  const names = ALLOWLIST.map((m) => m.hf).join("\n  - ");
  throw new Error(
    `model "${hf}" is not on the small-model allowlist (browser + consumer-GPU only).\n` +
      `Allowed:\n  - ${names}\n` +
      `Pick one of these, or re-run with --allow-large for 7-9B (needs 4GB+ VRAM, desktop browser).`,
  );
}

export function formatModels() {
  const rows = ALLOWLIST.map(
    (m) =>
      `${m.hf}\n    webllm: ${m.webllm ?? "-"}  onnx: ${m.onnx ?? "-"}  vram: ${m.vram}` +
      `  tier: ${m.tier} (min ${m.minDeviceMemoryGB}GB${m.mobileOk ? ", mobile-ok" : ""}${m.cpuOk ? ", cpu-ok" : ""})` +
      `${m.license ? `  license: ${m.license}` : ""}`,
  );
  return rows.join("\n");
}
