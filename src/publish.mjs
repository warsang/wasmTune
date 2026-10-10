// wasmtune — publish converted browser artifacts to the Hugging Face Hub.
//
// Deliberately optional and separate from `convert`. Weights for a usable chat
// model are 100 MB - 1 GB, which no static host will take, so the Hub is where
// they go; but a token must never be required to use the rest of the pipeline,
// and publishing must never happen implicitly on a build. `convert` stays
// offline and deterministic; `publish` is the one command that talks to a
// remote and needs a credential.

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const DEFAULT_ENDPOINT = "https://huggingface.co";
const HUB_TIMEOUT_MS = 30 * 60 * 1000;

const PKG_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// huggingface_hub lives in the venv the trainer bootstraps, so `convert` and
// `publish` agree on which interpreter runs. The venv sits at the root of the
// configured output dir, not beside every per-tier subdir, so walk up.
function venvPython(outDir) {
  if (!outDir) return null;
  const bin = process.platform === "win32" ? "Scripts" : "bin";
  const exe = process.platform === "win32" ? "python.exe" : "python";
  let dir = path.resolve(outDir);
  for (;;) {
    const candidate = path.join(dir, ".finetune-venv", bin, exe);
    if (existsSync(candidate)) return candidate;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

export function tokenFor(explicit = null, env = process.env) {
  return explicit ?? env.HF_TOKEN ?? env.HUGGING_FACE_HUB_TOKEN ?? null;
}

// transformers.js resolves `onnx/model_<dtype>.onnx`, and the dtype strings do
// not map 1:1 to suffixes: q8 is requested as "model_quantized.onnx". Getting
// this wrong produces a 404 on the graph that nothing in the pipeline notices.
// Only q8 is renamed; int8 and uint8 DO map to themselves.
export const DTYPE_SUFFIX = {
  q4: "model_q4.onnx",
  q4f16: "model_q4.onnx",
  fp16: "model_fp16.onnx",
  fp32: "model.onnx",
  int8: "model_int8.onnx",
  uint8: "model_uint8.onnx",
  q8: "model_quantized.onnx",
};

/** Which graphs a converted onnx/ dir actually contains, keyed by dtype. */
export function graphsIn(onnxDir) {
  if (!existsSync(onnxDir)) return [];
  let files;
  try { files = readdirSync(onnxDir); } catch { return []; }
  const found = [];
  const seen = new Set(); // q4 and q4f16 share a file; report it once
  for (const [dtype, file] of Object.entries(DTYPE_SUFFIX)) {
    if (seen.has(file) || !files.includes(file)) continue;
    seen.add(file);
    const abs = path.join(onnxDir, file);
    // A graph is only publishable if its external weights are beside it.
    // model.onnx is a 4 MB stub whose 4 GB live in model.onnx_data, so sorting
    // by size alone made publish pick the fp32 stub - a graph that 404s the
    // moment anything loads it. Score by total bytes actually present.
    const total = statSync(abs).size + externalDataRefs(abs)
      .filter((ref) => files.includes(ref))
      .reduce((sum, ref) => sum + statSync(path.join(onnxDir, ref)).size, 0);
    found.push({ dtype, file, bytes: statSync(abs).size, totalBytes: total, selfContained: total === statSync(abs).size });
  }
  // Rank by the dtype a browser can actually fetch, then by real size. A
  // self-contained quantized graph beats a stub whose weights are elsewhere,
  // and among equals the smaller total wins - that is the one a phone downloads.
  const dtypeRank = { q8: 0, q4: 1, q4f16: 2, int8: 3, uint8: 4, fp16: 5, fp32: 6 };
  return found.sort((a, b) => {
    if (a.selfContained !== b.selfContained) return a.selfContained ? -1 : 1;
    if (a.dtype !== b.dtype) {
      return (dtypeRank[a.dtype] ?? 99) - (dtypeRank[b.dtype] ?? 99);
    }
    return a.totalBytes - b.totalBytes;
  });
}

// External-data references travel with the graph. Collected from the protobuf
// without an onnx runtime: location strings are length-delimited ASCII fields.
export function externalDataRefs(graphPath) {
  const ascii = readFileSync(graphPath).toString("latin1");
  return [...new Set([...ascii.matchAll(/[\w.\-/]{1,120}\.(?:onnx_data|bin\d*|data)\b/g)].map((m) => m[0]))];
}

export function sha256File(p) {
  const h = crypto.createHash("sha256");
  h.update(readFileSync(p));
  return h.digest("hex");
}

async function hubFetch(url, init = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), HUB_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: ctl.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function ensureRepo({ repoId, token, endpoint = DEFAULT_ENDPOINT, privateRepo = false }) {
  const res = await hubFetch(`${endpoint}/api/repos/create`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({
      name: repoId.split("/")[1],
      type: "model",
      private: privateRepo,
      organization: repoId.includes("/") ? repoId.split("/")[0] : undefined,
    }),
  });
  if (res.ok) return { created: true };
  const body = await res.text();
  if (res.status === 409 || /already.*exist/i.test(body)) return { created: false };
  throw new Error(`create repo failed (HTTP ${res.status}): ${body.slice(0, 300)}`);
}

/** Upload one file to the Hub, streaming through huggingface_hub. */
export async function uploadFile({
  repoId, token, localPath, pathInRepo, summary = null,
  endpoint = DEFAULT_ENDPOINT, outDir = null, python = null,
}) {
  if (!existsSync(localPath)) {
    throw new Error(`upload ${pathInRepo} failed: ${localPath} does not exist`);
  }
  // The venv python: huggingface_hub lives there (optimum depends on it).
  const venvPy = venvPython(outDir);
  const py = python ?? (venvPy && existsSync(venvPy) ? venvPy
    : (existsSync("/usr/bin/python3") ? "/usr/bin/python3" : "python3"));
  const script = path.join(PKG_ROOT, "python", "hf_upload.py");
  const child = spawn(py, [
    script,
    "--file", localPath,
    "--repo", repoId,
    "--path-in-repo", pathInRepo,
    "--token", token ?? "",
    "--commit-message", summary ?? `wasmtune: publish ${pathInRepo}`,
  ], { stdio: ["ignore", "ignore", "inherit"] });
  await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`upload ${pathInRepo} failed (python exit ${code})`)));
  });
  return { pathInRepo, bytes: statSync(localPath).size };
}

/**
 * Verify the file in the repo is the file that was meant to be published.
 *
 * A silently no-op'd upload is the failure this closes: the earlier hand-rolled
 * scripts printed "Upload 0 LFS files" for unchanged content, and there was no
 * way to tell a successful replace from a skipped write. The Hub exposes an
 * LFS oid equal to the sha256 of the contents, so the comparison is exact.
 */
export async function verifyUploaded({ repoId, token, localPath, pathInRepo, sha256, endpoint = DEFAULT_ENDPOINT }) {
  const dir = path.dirname(pathInRepo);
  const res = await hubFetch(
    `${endpoint}/api/models/${repoId}/tree/main${dir === "." ? "" : `/${dir}`}`,
    { headers: { Authorization: `Bearer ${token}` } },
  );
  if (!res.ok) throw new Error(`list repo failed (HTTP ${res.status})`);
  for (const entry of await res.json()) {
    if (entry.path !== pathInRepo) continue;
    const oid = entry.lfs?.oid ?? entry.oid ?? null;
    if (oid === sha256) return { ok: true, size: entry.size, oid };
    return { ok: false, size: entry.size, oid, expected: sha256 };
  }
  throw new Error(`${pathInRepo} is not present in ${repoId}`);
}

/**
 * transformers.js reads the chat template from tokenizer_config.json's
 * `chat_template` field and does not look at a sibling chat_template.jinja. A
 * repo with the template only in the .jinja file loads fine but fails at
 * generation time with "Cannot use apply_chat_template() because
 * tokenizer.chat_template is not set", after which an instruct model continues
 * the system prompt instead of answering. Returns the corrected path, or null
 * when there is nothing to do.
 */
export function inlineChatTemplate({ dir, outDir = dir }) {
  const cfgPath = path.join(dir, "tokenizer_config.json");
  const jinjaPath = path.join(dir, "chat_template.jinja");
  if (!existsSync(cfgPath) || !existsSync(jinjaPath)) return null;
  let cfg;
  try { cfg = JSON.parse(readFileSync(cfgPath, "utf8")); } catch { return null; }
  if (typeof cfg.chat_template === "string" && cfg.chat_template.trim()) return null;
  cfg.chat_template = readFileSync(jinjaPath, "utf8");
  const out = path.join(outDir, "tokenizer_config.json");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(out, JSON.stringify(cfg, null, 2));
  return out;
}

function slugifyOnnxRepo(repoId) {
  return String(repoId).split("/").pop()
    ?.replace(/[^a-z0-9.-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase() ?? "onnx-tier";
}

function chatHints(chat) {
  if (!chat) return null;
  const keys = ["temperature", "maxTokens", "repetitionPenalty",
    "presencePenalty", "frequencyPenalty", "topP"];
  const out = Object.fromEntries(
    keys.filter((k) => chat[k] !== undefined).map((k) => [k, chat[k]]));
  return Object.keys(out).length ? out : null;
}

/**
 * Merge a published ONNX repo into the site's manifest instead of printing a
 * line for the host to paste. A publish that landed but was never wired up
 * looks exactly like never having run it, which is why this closes the loop.
 */
export function writeOnnxManifestEntry({
  manifestPath = "public/models/model-manifest.json",
  repoId,
  dtype = null,
  siteName = null,
  label = null,
  base = null,
  chat = null,
}) {
  const target = path.resolve(manifestPath);
  let manifest = { version: "0", models: [], notes: [] };
  if (existsSync(target)) {
    try { manifest = JSON.parse(readFileSync(target, "utf8")); } catch { /* start fresh */ }
  }
  if (!Array.isArray(manifest.models)) manifest.models = [];

  const entry = {
    id: slugifyOnnxRepo(repoId),
    label: label ?? `${repoId.split("/").pop()} (published ONNX)`,
    base: base ?? repoId,
    source: "tuned",
    ...(siteName ? { siteName } : {}),
    artifacts: { onnx: repoId, ...(dtype ? { onnxDtype: dtype } : {}) },
    ...(chatHints(chat) ? { chat: chatHints(chat) } : {}),
  };
  // The site name the weights were trained with, so the widget's prompt matches.
  if (siteName) manifest.dataset = { ...(manifest.dataset ?? {}), siteName };

  const at = manifest.models.findIndex((m) => {
    if (m.id === entry.id) return true;
    // A tier's ONNX repo is written as a bare string by this function but as
    // { id, dtype } by a hand-written manifest (which is what the ships one
    // is). Comparing only the string form made publish fall through to
    // unshift() and add a second tier for the same repo, leaving the old entry
    // - with its stale label and requirements - still in the ladder.
    const onnx = m?.artifacts?.onnx;
    const repo = typeof onnx === "string" ? onnx : (onnx?.id ?? onnx?.repo ?? null);
    return repo === repoId;
  });
  if (at === -1) manifest.models.unshift(entry);
  else {
    // Merge so the hand-curated identity of the tier (its id, label,
    // requirements and chat) survives a publish, while the repo, dtype and
    // site name come from this run. Order matters: existing wins for identity,
    // and artifacts is rebuilt explicitly below.
    const existing = manifest.models[at];
    const existingOnnx = existing?.artifacts?.onnx;
    const nested = typeof existingOnnx === "object" && existingOnnx !== null;
    manifest.models[at] = {
      ...entry,
      ...existing,
      artifacts: {
        ...existing.artifacts,
        onnx: nested
          ? { ...existingOnnx, id: repoId, ...(dtype ? { dtype } : {}) }
          : repoId,
      },
    };
  }

  manifest.created = new Date().toISOString();
  mkdirSync(path.dirname(target), { recursive: true });
  writeFileSync(target, JSON.stringify(manifest, null, 2));
  return { path: target, entry };
}

export async function writeModelCard({ dir, siteName, baseModel, dtype, repoId = null, evalReport = null }) {
  const lines = [
    "---",
    "library_name: transformers.js",
    ...(baseModel ? ["base_model: " + baseModel] : []),
    "tags:",
    "  - wasmtune",
    "  - onnx",
    "---",
    "",
    `# ${siteName} assistant (fine-tuned ${path.basename(baseModel ?? "site model")})`,
    "",
    "Fine-tuned with [wasmtune](https://github.com/warsang/wasmTune) on the pages of",
    `the content folder behind \`${siteName}\`, then exported to ONNX for in-browser`,
    "inference with transformers.js.",
    "",
    "```js",
    'import { pipeline } from "@huggingface/transformers";',
    `const gen = await pipeline("text-generation", "${repoId ?? "org/repo"}", { dtype: "${dtype ?? "q8"}" });`,
    "```",
    "",
    "Inference runs entirely on the visitor's device. The weights here are a",
    "static download; no inference server is involved.",
    "",
  ];
  if (evalReport) {
    lines.push(
      "## Eval",
      "",
      "```",
      `base   avg=${evalReport.base?.avg ?? "?"}`,
      `tuned  avg=${evalReport.tuned?.avg ?? "?"}`,
      `delta  ${evalReport.delta ?? "?"}`,
      "```",
      "",
      "Keyword recall only: a passing gate means training moved the model, not",
      "that the answers are correct. Check `faithfulness` and `untraced` in",
      "`eval.report.json` before relying on it.",
      "",
    );
  }
  // NOT <cwd>/README.md. publish used to write the Hub model card there, which
  // overwrote the repo's own README - 611 lines of documentation destroyed by a
  // command whose job is to upload a model. The card is an upload payload, so
  // it goes somewhere that is not part of the project.
  const cardDir = process.env.WASMTUNE_CARD_DIR
    ?? path.join(dir, ".finetune", "model-card");
  await mkdir(cardDir, { recursive: true });
  const out = path.join(cardDir, "README.md");
  await writeFile(out, lines.join("\n"));
  return out;
}
