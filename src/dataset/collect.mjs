// wasmtune — crawl a generic website/content folder into plain texts.

import { existsSync, statSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

export const TEXT_EXTS = new Set([".md", ".mdx", ".txt", ".html", ".htm", ".js", ".mjs", ".cjs", ".json", ".ts"]);
export const SKIP_DIRS = new Set([
  "node_modules", "dist", "build", ".git", ".finetune", ".finetune-venv",
  "vendor", "public/vendor", "__pycache__", ".next", "coverage",
]);

export async function collectTexts(dataDirOrDirs, { cwd = process.cwd(), maxBytes = 20_000_000 } = {}) {
  const dirs = Array.isArray(dataDirOrDirs) ? dataDirOrDirs : [dataDirOrDirs];
  const roots = dirs.map((d) => path.resolve(cwd, d));
  for (const root of roots) {
    if (!existsSync(root)) throw new Error(`dataDir not found: ${root}`);
  }
  const files = [];
  const stack = [];
  const seen = new Set();
  for (const root of roots) {
    try {
      // Allow dataDir entries to be individual files as well as folders.
      if (statSync(root).isFile()) {
        const ext = path.extname(root).toLowerCase();
        if (TEXT_EXTS.has(ext) && statSync(root).size <= 2_000_000) files.push(root);
        continue;
      }
    } catch {
      continue;
    }
    stack.push(root);
  }
  while (stack.length) {
    const dir = stack.pop();
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      if (SKIP_DIRS.has(e.name)) continue;
      const full = path.join(dir, e.name);
      if (seen.has(full)) continue;
      seen.add(full);
      if (e.isDirectory()) {
        stack.push(full);
      } else if (e.isFile()) {
        const ext = path.extname(e.name).toLowerCase();
        if (!TEXT_EXTS.has(ext)) continue;
        try {
          if (statSync(full).size > 2_000_000) continue;
        } catch {
          continue;
        }
        files.push(full);
      }
    }
  }
  files.sort();
  const docs = [];
  let total = 0;
  for (const full of files) {
    let raw;
    try {
      raw = await readFile(full, "utf8");
    } catch {
      continue;
    }
    const text = cleanText(raw, path.extname(full).toLowerCase());
    if (text.length < 200) continue; // skip stubs/empty pages
    if (total + text.length > maxBytes) break;
    total += text.length;
    // Path relative to the containing data root (supports multiple roots).
    const home = roots.find((r) => full.startsWith(r + path.sep) || full === r) ?? roots[0];
    const rel = path.relative(home, full);
    docs.push({
      path: rel || path.basename(full),
      text,
      hash: crypto.createHash("sha256").update(text).digest("hex").slice(0, 16),
    });
  }
  return { root: roots.length === 1 ? roots[0] : roots.join(", "), roots, docs, totalChars: total };
}

export function cleanText(raw, ext) {
  let t = String(raw ?? "");
  // Strip ESM import/export noise for .mjs/.js lesson bodies: keep template text.
  if (ext === ".html" || ext === ".htm") {
    t = t.replace(/<script[\s\S]*?<\/script>/gi, " ").replace(/<style[\s\S]*?<\/style>/gi, " ");
    t = t.replace(/<[^>]+>/g, " ");
  }
  // Unwrap JS string bodies crudely: keep printable runs.
  t = t.replace(/```/g, "\n```\n");
  t = t.replace(/[ \t]+/g, " ");
  t = t
    .split("\n")
    .map((l) => l.trimEnd())
    .filter((l) => {
      const s = l.trim();
      if (!s) return false;
      // drop pure import/export lines common in lesson index files
      if (/^(import|export)\s+.*from\s+["']/.test(s)) return false;
      return true;
    })
    .join("\n");
  t = t.replace(/\n{3,}/g, "\n\n").trim();
  // Decode common entities for html sources
  t = t.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
  return t;
}
