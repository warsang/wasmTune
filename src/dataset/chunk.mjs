// wasmtune — deterministic chunking + dedup.

import crypto from "node:crypto";

export function chunkDocs(docs, { maxChars = 2000, overlap = 200, minChars = 400 } = {}) {
  const chunks = [];
  for (const doc of docs) {
    const parts = splitWindows(doc.text, maxChars, overlap);
    for (let i = 0; i < parts.length; i++) {
      const text = parts[i].trim();
      if (text.length < minChars) continue;
      chunks.push({
        source: doc.path,
        ordinal: i,
        text,
        hash: crypto.createHash("sha256").update(text).digest("hex").slice(0, 16),
      });
    }
  }
  // Deterministic dedup by hash, keep first occurrence.
  const seen = new Set();
  return chunks.filter((c) => {
    if (seen.has(c.hash)) return false;
    seen.add(c.hash);
    return true;
  });
}

export function splitWindows(text, maxChars, overlap) {
  if (text.length <= maxChars) return [text];
  const out = [];
  // Prefer paragraph boundaries.
  const paras = text.split(/\n\n+/);
  let cur = "";
  for (const p of paras) {
    if ((cur + "\n\n" + p).length <= maxChars) {
      cur = cur ? cur + "\n\n" + p : p;
    } else {
      if (cur) out.push(cur);
      if (p.length > maxChars) {
        // hard slice long paragraph with overlap
        let i = 0;
        while (i < p.length) {
          out.push(p.slice(i, i + maxChars));
          i += maxChars - overlap;
        }
        cur = "";
      } else {
        cur = overlap && out.length
          ? tailChars(out[out.length - 1], overlap) + "\n\n" + p
          : p;
        if (cur.length > maxChars) {
          out.push(cur.slice(0, maxChars));
          cur = cur.slice(maxChars - overlap);
        }
      }
    }
  }
  if (cur.trim()) out.push(cur);
  return out;
}

function tailChars(s, n) {
  return s.slice(Math.max(0, s.length - n));
}

export function approxTokens(chars) {
  return Math.ceil(chars / 4);
}
