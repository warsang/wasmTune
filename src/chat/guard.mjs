// wasmtune — runtime repetition guard (generic).
//
// Decoding penalties reduce loops but can't guarantee their absence, so the
// widget watches the live token stream too. When an n-gram loop is detected,
// generation stops and the visible reply is truncated at the loop onset
// (optionally replaced with a fallback line). Reuses the eval detector.

import { hasRepetitionLoop } from "../eval/score.mjs";

export const LOOP_FALLBACK = "I started repeating myself — let me stop there. Could you rephrase the question?";

// Index in `text` where a loop starts (start of theGram's second occurrence),
// or -1 when no loop. Keeps one copy of the repeated span.
export function findLoopOnset(text, { n = 8, repeats = 3 } = {}) {
  const tokens = String(text ?? "").split(/(\s+)/);
  const words = [];
  const wordStart = [];
  let pos = 0;
  for (const t of tokens) {
    if (/^\s+$/.test(t) || !t) {
      pos += t.length;
      continue;
    }
    words.push(t.toLowerCase());
    wordStart.push(pos);
    pos += t.length;
  }
  const seen = new Map();
  for (let i = 0; i + n <= words.length; i++) {
    const gram = words.slice(i, i + n).join(" ");
    const at = seen.get(gram);
    if (at) {
      at.push(i);
      if (at.length >= repeats) return wordStart[at[1]];
    } else {
      seen.set(gram, [i]);
    }
  }
  return -1;
}

export function truncateAtLoop(text, opts) {
  const at = findLoopOnset(text, opts);
  if (at === -1) return String(text ?? "");
  return String(text).slice(0, at).trimEnd();
}

// Strip reasoning blocks from displayed text, backend-independent.
// Covers <think>…</think> (Qwen3.5/Gemma4/DeepSeek-R1 style), <thought>…,
// [Start thinking]… markers, and unclosed variants (stream cut off
// mid-thought). Returns { text, stripped } so callers can tell whether
// the model thought out loud (useful for eval + debugging).
export function stripThinkingBlocks(text) {
  let out = String(text ?? "");
  let stripped = false;
  // Paired tags (case-insensitive, multiline).
  for (const tag of ["think", "thought", "thinking"]) {
    const re = new RegExp(`<${tag}[^>]*>[\\s\\S]*?<\\/${tag}>`, "gi");
    if (re.test(out)) {
      stripped = true;
      out = out.replace(re, "");
    }
  }
  // Unclosed opening tag: drop it and everything after (stream was cut).
  const open = out.match(/<(think|thought|thinking)[^>]*>[\s\S]*$/i);
  if (open) {
    stripped = true;
    out = out.slice(0, open.index);
  }
  // Plain-text "[Start thinking]" … "[End thinking]" style markers.
  const startMarker = out.search(/\[\s*start\s+thinking\s*\]/i);
  if (startMarker !== -1) {
    stripped = true;
    const endMarker = out.slice(startMarker).search(/\[\s*end\s+thinking\s*\]/i);
    out = endMarker === -1
      ? out.slice(0, startMarker)
      : out.slice(0, startMarker) + out.slice(startMarker + endMarker).replace(/^\[\s*end\s+thinking\s*\]/i, "");
  }
  out = out.replace(/\n{3,}/g, "\n\n").trim();
  return { text: out, stripped };
}

export function createLoopGuard({ checkEveryChars = 64, tailChars = 1200, minChars = 200, detectOpts = {}, fallback = LOOP_FALLBACK } = {}) {
  let buf = "";
  let lastCheck = 0;
  let tripped = false;
  return {
    get tripped() {
      return tripped;
    },
    get text() {
      return buf;
    },
    feed(piece) {
      if (tripped) return { tripped: true, text: buf };
      buf += String(piece ?? "");
      if (buf.length - lastCheck < checkEveryChars || buf.length < minChars) {
        return { tripped: false, text: buf };
      }
      lastCheck = buf.length;
      const tail = buf.slice(-tailChars);
      if (hasRepetitionLoop(tail, detectOpts)) {
        tripped = true;
        const onset = findLoopOnset(tail, detectOpts);
        const cut = onset === -1 ? buf.length : buf.length - tail.length + onset;
        buf = buf.slice(0, cut).trimEnd();
        return { tripped: true, text: buf, fallback };
      }
      return { tripped: false, text: buf };
    },
  };
}
