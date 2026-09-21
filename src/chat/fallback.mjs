// wasmtune — runtime capability detection + artifact picking for the chat widget.
// Order: WebLLM (WebGPU) -> wllama (GGUF, WebGPU or CPU) -> Transformers.js
// (ONNX, WebGPU or WASM).

import { manifestEntries } from "./hardware.mjs";

export async function detectCapabilities() {
  const webgpu = await hasWebGPU();
  return {
    webgpu,
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "node",
    crossOriginIsolated: typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : false,
    recommended: webgpu ? "webllm" : "transformers",
  };
}

export async function hasWebGPU() {
  try {
    if (typeof navigator === "undefined" || !navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

// Which artifact to load for one manifest entry on this device. The engine
// preference is unchanged from v1; multi-model manifests just give every
// entry its own artifacts.
export function pickArtifactForEntry(entry, caps) {
  const a = entry?.artifacts ?? {};
  if (caps?.webgpu) {
    if (a.mlc) return { kind: "webllm", ref: a.mlc };
    if (a.webllm) return { kind: "webllm-prebuilt", ref: a.webllm };
    if (a.gguf) return { kind: "wllama", ref: a.gguf };
    if (a.onnx) return { kind: "transformers", ref: a.onnx };
    return { kind: "none", ref: null };
  }
  // No WebGPU: ONNX (transformers.js WASM) first — the incumbent behavior —
  // then GGUF via wllama's CPU path.
  if (a.onnx) return { kind: "transformers", ref: a.onnx };
  if (a.gguf) return { kind: "wllama", ref: a.gguf };
  return { kind: "none", ref: null };
}

// Legacy single-artifact manifests: pick from the primary entry.
export function pickArtifact(manifest, caps) {
  const entry = manifestEntries(manifest)[0] ?? { artifacts: {} };
  return pickArtifactForEntry(entry, caps);
}

// Resolve a loadable URL from a manifest artifact. GGUF artifacts are
// `{url, sha256, bytes}` (content-hashed filename busts browser caches);
// pre-v3 manifests stored a bare path string — still accepted.
export function ggufUrl(artifact) {
  if (!artifact) return null;
  if (typeof artifact === "string") return artifact;
  return artifact.url ?? null;
}
