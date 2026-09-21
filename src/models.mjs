// wasmtune — small-model allowlist.
//
// Only models that BOTH (a) fine-tune on consumer hardware via LoRA/QLoRA
// and (b) have a realistic browser inference path (WebLLM MLC artifact and/or
// ONNX community artifact) are accepted. This keeps `finetune train` working
// on an M-series Mac and on a single NVIDIA card, and `finetune convert` +
// <site-chat> able to serve the result over WebGPU.

export const ALLOWLIST = [
  {
    hf: "HuggingFaceTB/SmolLM2-135M-Instruct",
    webllm: "SmolLM2-135M-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-135M-Instruct-ONNX",
    params: "135M", vram: "~78MB", context: "2K",
    family: "SmolLM2",
  },
  {
    hf: "HuggingFaceTB/SmolLM2-360M-Instruct",
    webllm: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-360M-Instruct-ONNX",
    params: "360M", vram: "~210MB", context: "2K",
    family: "SmolLM2",
  },
  {
    hf: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    webllm: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-1.7B-Instruct-ONNX",
    params: "1.7B", vram: "~1GB", context: "2K",
    family: "SmolLM2",
  },
  {
    hf: "Qwen/Qwen2.5-0.5B-Instruct",
    webllm: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Qwen2.5-0.5B-Instruct-ONNX",
    params: "0.5B", vram: "~278MB", context: "4K",
    family: "Qwen",
  },
  {
    hf: "Qwen/Qwen2.5-1.5B-Instruct",
    webllm: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Qwen2.5-1.5B-Instruct-ONNX",
    params: "1.5B", vram: "~868MB", context: "4K",
    family: "Qwen",
  },
  {
    hf: "Qwen/Qwen3-0.6B",
    webllm: "Qwen3-0.6B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-0.6B-ONNX",
    params: "0.6B", vram: "~350MB", context: "4K",
    family: "Qwen",
  },
  {
    hf: "Qwen/Qwen3-1.7B",
    webllm: "Qwen3-1.7B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-1.7B-ONNX",
    params: "1.7B", vram: "~1.1GB", context: "4K",
    family: "Qwen",
  },
  {
    hf: "Qwen/Qwen3-4B-Instruct-2507",
    mlx: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    webllm: "Qwen3-4B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-4B-ONNX",
    params: "4B", vram: "~2.2GB", context: "32K",
    family: "Qwen",
    note: "best fine-tuned SLM (distil-labs 12x8 study); non-thinking instruct",
  },
  {
    hf: "Qwen/Qwen3-4B",
    webllm: "Qwen3-4B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-4B-ONNX",
    params: "4B", vram: "~2.2GB", context: "4K",
    family: "Qwen",
  },
  {
    hf: "meta-llama/Llama-3.2-1B-Instruct",
    webllm: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    params: "1B", vram: "~712MB", context: "4K",
    family: "Llama", license: "llama-community (accept terms)",
  },
  {
    hf: "meta-llama/Llama-3.2-3B-Instruct",
    webllm: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Llama-3.2-3B-Instruct-ONNX",
    params: "3B", vram: "~1.76GB", context: "4K",
    family: "Llama", license: "llama-community (accept terms)",
  },
  {
    hf: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    webllm: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
    onnx: "onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX",
    params: "1.1B", vram: "~400MB", context: "2K",
    family: "TinyLlama",
  },
  {
    hf: "microsoft/Phi-3.5-mini-instruct",
    webllm: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    onnx: "microsoft/Phi-3-mini-4k-instruct-onnx-web",
    params: "3.8B", vram: "~2.1GB", context: "4K",
    family: "Phi",
  },
  {
    hf: "google/gemma-2-2b-it",
    webllm: "gemma-2-2b-it-q4f16_1-MLC",
    onnx: "onnx-community/gemma-2-2b-it-ONNX",
    params: "2B", vram: "~1.44GB", context: "4K",
    family: "Gemma", license: "gemma terms",
  },
  {
    hf: "google/gemma-3-1b-it",
    webllm: "gemma-3-1b-it-q4f16_1-MLC",
    onnx: "onnx-community/gemma-3-1b-it-ONNX",
    params: "1B", vram: "~700MB", context: "8K",
    family: "Gemma", license: "gemma terms",
  },
  {
    hf: "Qwen/Qwen3.5-4B",
    mlx: "mlx-community/Qwen3.5-4B-MLX-4bit",
    gguf: "unsloth/Qwen3.5-4B-GGUF",
    params: "4B", vram: "~3GB", context: "262K",
    family: "Qwen",
    note: "hybrid thinking (select non-thinking via chat template); strongest sub-10B reasoning",
  },
  {
    hf: "google/gemma-4-E4B-it",
    mlx: "mlx-community/gemma-4-e4b-it-4bit",
    gguf: "ggml-org/gemma-4-E4B-it-GGUF",
    params: "4.5B eff", vram: "~3GB", context: "128K",
    family: "Gemma",
    note: "disable thinking via enable_thinking=false; VLM arch (mlx_lm text path)",
  },
];

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

export function lookupModel(hf) {
  return byHf.get(String(hf ?? "").toLowerCase()) ?? null;
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
    (m) => `${m.hf}\n    webllm: ${m.webllm}  onnx: ${m.onnx}  vram: ${m.vram}${m.license ? `  license: ${m.license}` : ""}`,
  );
  return rows.join("\n");
}
