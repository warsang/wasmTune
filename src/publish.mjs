// wasmtune — publish converted browser artifacts to the Hugging Face Hub.
//
// Deliberately optional and separate from `convert`. Weights for a usable chat
// model are 100 MB - 1 GB, which no static host will take, so the Hub is where
// they go; but a token must never be required to use the rest of the pipeline,
// and publishing must never happen implicitly on a build. `convert` stays
// offline and deterministic; `publish` is the one command that talks to a
// remote and needs a credential.

import { existsSync, readFileSync, statSync } from "node:fs";
import { readdir, writeFile } from "node:fs/promises";
import path from "node:path";

const DEFAULT_ENDPOINT = "https://huggingface.co";
const HUB_TIMEOUT_MS = 30 * 60 * 1000;

export function tokenFor(explicit = null, env = process.env) {
  return explicit ?? env.HF_TOKEN ?? env.HUGGING_FACE_HUB_TOKEN ?? null;
}

// transformers.js resolves `onnx/model_<dtype>.onnx`, and the dtype strings do
// not map 1:1 to suffixes: q8 is requested as "model_quantized.onnx". Getting
// this wrong produces a 404 on the graph that nothing in the pipeline notices.
export const DTYPE_SUFFIX = {
  q4: "model_q4.onnx",
  q4f16: "model_q4f16.onnx",
  fp16: "model_fp16.onnx",
  fp32: "model.onnx",
  int8: "model_int8.onnx",
  uint8: "model_uint8.onnx",
  q8: "model_quantized.onnx",
};

/** Which graphs a converted onnx/ dir actually contains, keyed by dtype. */
export async function graphsIn(onnxDir) {
  if (!existsSync(onnxDir)) return [];
  const files = await readdir(onnxDir);
  const found = [];
  for (const [dtype, file] of Object.entries(DTYPE_SUFFIX)) {
    if (files.includes(file)) {
      found.push({ dtype, file, bytes: statSync(path.join(onnxDir, file)).size });
    }
  }
  return found.sort((a, b) => a.bytes - b.bytes);
}

// External-data references have to travel with the graph. Collect them from the
// ONNX protobuf without needing an onnx runtime: the location strings are
// plain repeated fields.
export function externalDataRefs(graphPath) {
  const buf = readFileSync(graphPath);
  const refs = new Set();
  // Scan for printable runs ending in a data-file extension; the protobuf
  // stores `location` as a length-delimited ASCII string.
  const ascii = buf.toString("latin1");
  for (const m of ascii.matchAll(/[\w.\-/]{1,120}\.(?:onnx_data|bin\d*|data)\b/g)) {
    refs.add(m[0]);
  }
  return [...refs];
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
    body: JSON.stringify({ name: repoId.split("/")[1], type: "model", private: privateRepo, organization: repoId.includes("/") ? repoId.split("/")[0] : undefined }),
  });
  if (res.ok) return { created: true };
  const body = await res.text();
  if (res.status === 409 || /already.*exist/i.test(body)) return { created: false };
  throw new Error(`create repo failed (HTTP ${res.status}): ${body.slice(0, 300)}`);
}

/** Upload one file, letting the git-LFS path handle anything large. */
export async function uploadFile({ repoId, token, localPath, pathInRepo, endpoint = DEFAULT_ENDPOINT }) {
  const bytes = statSync(localPath).size;
  const res = await hubFetch(
    `${endpoint}/api/models/${repoId}/upload/main/${encodeURIComponent(pathInRepo).replace(/%2F/g, "/")}`,
    {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${token}`,
        "content-type": "application/octet-stream",
        "x-lfs-size": String(bytes),
      },
      body: readFileSync(localPath),
    },
  );
  if (!res.ok) {
    throw new Error(`upload ${pathInRepo} failed (HTTP ${res.status}): ${(await res.text()).slice(0, 200)}`);
  }
  return { pathInRepo, bytes };
}

/** README so the published repo explains what it is. */
export async function writeModelCard({ dir, siteName, baseModel, dtype, evalReport = null }) {
  const lines = [
    "---",
    "library_name: transformers.js",
    "base_model: " + baseModel,
    "tags:",
    "  - wasmtune",
    "  - onnx",
    "---",
    "",
    `# ${siteName} assistant (fine-tuned ${path.basename(baseModel)})`,
    "",
    "Fine-tuned with [wasmtune](https://github.com/warsang/wasmTune) on the pages of",
    `the content folder behind \`${siteName}\`, then exported to ONNX for in-browser`,
    "inference with transformers.js.",
    "",
    "```js",
    'import { pipeline } from "@huggingface/transformers";',
    `const gen = await pipeline("text-generation", "${"${REPO_ID}"}", { dtype: "${dtype}" });`,
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
      "Keyword-recall only: a passing gate means training moved the model, not",
      "that the answers are correct. Check `faithfulness` and `untraced` in",
      "`eval.report.json` before relying on it.",
      "",
    );
  }
  const out = path.join(dir, "README.md");
  await writeFile(out, lines.join("\n"));
  return out;
}
