// wasmtune — runtime capability detection for the chat widget.
// Order: WebLLM (WebGPU) -> Transformers.js (ONNX, WebGPU/WASM) -> wllama (WASM).

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

export function pickArtifact(manifest, caps) {
  if (caps.webgpu && manifest?.artifacts?.mlc) return { kind: "webllm", ref: manifest.artifacts.mlc };
  if (manifest?.artifacts?.onnx) return { kind: "transformers", ref: manifest.artifacts.onnx };
  if (manifest?.artifacts?.gguf) return { kind: "wllama", ref: manifest.artifacts.gguf };
  return { kind: "none", ref: null };
}

// Resolve a loadable URL from a manifest artifact. GGUF artifacts are
// `{url, sha256, bytes}` (content-hashed filename busts browser caches);
// pre-v3 manifests stored a bare path string — still accepted.
export function ggufUrl(artifact) {
  if (!artifact) return null;
  if (typeof artifact === "string") return artifact;
  return artifact.url ?? null;
}
