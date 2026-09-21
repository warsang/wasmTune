// wasmtune — dataset orchestration: collect -> chunk -> sft/dpo/grpo jsonl.

import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { collectTexts } from "./collect.mjs";
import { chunkDocs, approxTokens } from "./chunk.mjs";
import { buildSftBlend, toJsonl, answerCharsStats, kindCounts } from "./sft.mjs";
import { sftToDpoSeed, sftToGrpoSeed } from "./dpo.mjs";
import { synthQa } from "./synth.mjs";
import { venvPaths } from "../train/venv.mjs";

export async function buildDataset(config, { cwd = process.cwd(), synth = null } = {}) {
  const outDir = path.resolve(cwd, config.output?.dir ?? "./.finetune");
  await mkdir(outDir, { recursive: true });

  const { root, docs, totalChars } = await collectTexts(config.dataDir, { cwd });
  const allChunks = chunkDocs(docs);
  // Deterministic holdout for `finetune eval`: ~5% of chunks by hash, never trained on.
  const { trainChunks, holdoutChunks } = splitHoldout(allChunks, 0.05);
  const siteName = config.dataset?.siteName ?? inferSiteName(cwd);
  const summary = config.dataset?.summary ?? inferSummary(docs);
  const maxPairs = config.dataset?.maxPairs ?? 5000;
  const synthSpec = synth ?? config.dataset?.synth ?? null;
  let synthPairs = [];
  if (synthSpec) {
    const { py } = venvPaths(outDir);
    const venvPy = existsSync(py) ? py : null;
    synthPairs = await synthQa(trainChunks, {
      spec: synthSpec,
      siteName,
      maxChunks: config.dataset?.synthMaxChunks ?? 200,
      venvPy,
      onProgress: (p) => {
        if (p.error) console.error(`[finetune] synth ${p.done}/${p.total} ${p.source}: ${p.error}`);
        else if (p.done % 25 === 0 || p.done === p.total) console.error(`[finetune] synth ${p.done}/${p.total} (${p.pairs} pairs)`);
      },
    });
    console.error(`[finetune] synth: ${synthPairs.length} pairs via ${synthSpec}`);
  }
  const sft = buildSftBlend(trainChunks, { siteName, summary, maxPairs, extraPairs: synthPairs });
  const dpoSeed = sftToDpoSeed(sft, { chunks: trainChunks });
  const grpoSeed = sftToGrpoSeed(sft);

  const sftPath = path.join(outDir, "dataset.sft.jsonl");
  const dpoPath = path.join(outDir, "dataset.dpo.jsonl");
  const grpoPath = path.join(outDir, "dataset.grpo.jsonl");
  const holdoutPath = path.join(outDir, "dataset.holdout.json");
  await writeFile(sftPath, toJsonl(sft));
  await writeFile(dpoPath, dpoSeed.map((r) => JSON.stringify(r)).join("\n") + "\n");
  await writeFile(grpoPath, grpoSeed.map((r) => JSON.stringify(r)).join("\n") + "\n");
  await writeFile(holdoutPath, JSON.stringify(holdoutChunks, null, 2));

  const report = {
    root,
    docs: docs.length,
    chunks: allChunks.length,
    trainChunks: trainChunks.length,
    holdoutChunks: holdoutChunks.length,
    sftPairs: sft.length,
    kinds: kindCounts(sft),
    answerChars: answerCharsStats(sft),
    dpoSeedPairs: dpoSeed.length,
    grpoSeedPrompts: grpoSeed.length,
    totalChars,
    approxTokens: approxTokens(totalChars),
    files: { sft: sftPath, dpo: dpoPath, grpo: grpoPath, holdout: holdoutPath },
  };
  await writeFile(path.join(outDir, "dataset.report.json"), JSON.stringify(report, null, 2));
  return report;
}

// Deterministic holdout split by chunk hash (stable across runs).
export function splitHoldout(chunks, fraction = 0.05) {
  const trainChunks = [];
  const holdoutChunks = [];
  for (const c of chunks) {
    const bucket = parseInt(String(c.hash ?? "00").slice(0, 2), 16) / 255;
    if (bucket < fraction) holdoutChunks.push(c);
    else trainChunks.push(c);
  }
  // Guarantee a non-empty holdout when chunks exist.
  if (!holdoutChunks.length && chunks.length) holdoutChunks.push(chunks[0]);
  return { trainChunks: trainChunks.filter((c) => !holdoutChunks.includes(c)), holdoutChunks };
}

function inferSummary(docs) {
  const first = docs.find((d) => d.text && d.text.trim().length > 100);
  if (!first) return null;
  const para = first.text.split(/\n\n+/).map((s) => s.trim()).find((s) => s.length > 80);
  return para ? para.replace(/\s+/g, " ").slice(0, 300) : null;
}

function inferSiteName(cwd) {
  const base = path.basename(path.resolve(cwd));
  return base.replace(/[-_]+/g, " ");
}
