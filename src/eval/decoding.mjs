// Resolve the decoding parameters that actually ship.
//
// `eval` and `eval-onnx` both took them from config.chat, but the widget never
// sees config.chat - it reads the `chat` block of the manifest entry it loads.
// When those two disagree - a hand-curated manifest, like the demo's, which
// omits topP/presencePenalty/frequencyPenalty - the gate scores the artifact
// under a different decoder than the browser uses. That is how a gate can
// report loops the widget never sees, and miss loops it does.
//
// So: read the manifest, fall back to config.chat, and let both evals share
// one answer.

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

/** The manifest's chat block for the tuned tier, or null if there isn't one. */
export function manifestChat(config, cwd = process.cwd()) {
  try {
    const p = path.resolve(
      cwd,
      config?.output?.webDir ?? "public/models",
      "model-manifest.json",
    );
    if (!existsSync(p)) return null;
    const manifest = JSON.parse(readFileSync(p, "utf8"));
    const entry = (manifest.models ?? []).find(
      (e) => e?.source === "tuned" && e?.artifacts?.onnx,
    );
    // Only trust it if it actually says something about decoding.
    if (!entry?.chat || typeof entry.chat !== "object") return null;
    return entry.chat;
  } catch {
    return null;
  }
}

/**
 * One decoding config for both evals, in the shape python expects.
 * `overrides` lets a caller pass explicit flags, which still win.
 *
 * When a manifest entry exists it is the whole answer: the widget reads that
 * block and nothing else, so a key it omits is a key the browser does not
 * apply. Falling back to config.chat for the omitted keys would reintroduce
 * exactly the mismatch this module exists to close.
 */
export function resolveDecoding(config, cwd = process.cwd(), overrides = {}) {
  const shipped = manifestChat(config, cwd);
  const fallback = config?.chat ?? {};
  const src = shipped ?? fallback;

  const pick = (...keys) => {
    for (const k of keys) {
      if (overrides[k] != null) return overrides[k];
      if (src?.[k] != null) return src[k];
    }
    // Only consult config.chat when there is no manifest to read.
    if (!shipped) {
      for (const k of keys) {
        if (fallback?.[k] != null) return fallback[k];
      }
    }
    return null;
  };

  return {
    temperature: pick("temperature"),
    topP: pick("topP", "top_p"),
    repetitionPenalty: pick("repetitionPenalty", "repetition_penalty"),
    presencePenalty: pick("presencePenalty", "presence_penalty"),
    frequencyPenalty: pick("frequencyPenalty", "frequency_penalty"),
    maxTokens: pick("maxTokens", "max_tokens"),
    source: shipped ? "manifest" : (Object.keys(fallback).length ? "config.chat" : "defaults"),
  };
}
