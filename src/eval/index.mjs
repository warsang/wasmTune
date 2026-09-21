// wasmtune — `wasmtune eval` orchestration (generic).
//
// Builds eval prompts from the training holdout (never trained on), runs
// base + tuned models, scores with src/eval/score.mjs, and gates on
// regression: tuned must not score below base.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { chunksToQa } from "../dataset/qa.mjs";
import { evalLoopContrasts } from "../dataset/dpo.mjs";
import { collectTexts } from "../dataset/collect.mjs";
import { chunkDocs } from "../dataset/chunk.mjs";
import { scoreResponse, summarize, faithfulness, scoreConversation, CONVERSE_PROBES, hasRepetitionLoop } from "./score.mjs";
import { parseJudgeSpec, judgeRows } from "./judge.mjs";
import { ensureVenv, runPython } from "../train/venv.mjs";

export const DEFAULT_MAX_PROMPTS = 30;
export const DEFAULT_MAX_TOKENS = 128;
export const MEMO_KINDS = new Set(["fact-qa", "definition", "faq", "howto", "synth-qa", "code-qa"]);

// Memorization prompts come from TRAINING pairs: for site-QA tuning the
// goal is learning the corpus closed-book, so the gate measures whether
// tuned recalls seen facts better than base. Holdout prompts (unseen
// chunks) measure generalization only — neither model can know unseen
// facts, so they never gate.
export function buildMemorizationPrompts(sftPairs, { maxPrompts = DEFAULT_MAX_PROMPTS, sourceByKey = null } = {}) {
  const qa = sftPairs.filter((p) => MEMO_KINDS.has(p.meta?.kind));
  const pool = qa.length ? qa : sftPairs;
  const stride = Math.max(1, Math.floor(pool.length / maxPrompts));
  const out = [];
  for (let i = 0; i < pool.length && out.length < maxPrompts; i += stride) {
    const p = pool[i];
    const user = p.messages.find((m) => m.role === "user")?.content ?? "";
    const ref = p.messages.find((m) => m.role === "assistant")?.content ?? "";
    if (!user || !ref) continue;
    const key = `${p.meta?.source ?? ""}::${p.meta?.ordinal ?? 0}`;
    out.push({
      id: `mem-${out.length}`, prompt: user, reference: ref,
      source: p.meta?.source ?? "", kind: p.meta?.kind ?? "",
      sourceText: sourceByKey?.get(key)?.slice(0, 4000) ?? "",
    });
  }
  return out;
}

export async function buildEvalPrompts(holdoutChunks, { siteName = "this website", maxPrompts = 15 } = {}) {
  const qa = chunksToQa(holdoutChunks, { siteName, maxPairs: maxPrompts * 3 });
  // Prefer short references (crisp facts) and spread across sources.
  const seenSources = new Set();
  const picked = [];
  const rest = [];
  for (const p of qa) {
    const user = p.messages.find((m) => m.role === "user")?.content ?? "";
    const ref = p.messages.find((m) => m.role === "assistant")?.content ?? "";
    if (!user || !ref) continue;
    const row = { prompt: user, reference: ref, source: p.meta?.source ?? "", kind: p.meta?.kind ?? "" };
    if (!seenSources.has(row.source)) {
      seenSources.add(row.source);
      picked.push(row);
    } else {
      rest.push(row);
    }
  }
  return [...picked, ...rest].slice(0, maxPrompts).map((r, i) => ({ id: `eval-${i}`, ...r }));
}

// The prompts file doubles as the auto-derived golden set: deterministic,
// persisted, with references + source text attached for faithfulness scoring.
export async function runEval(config, { cwd = process.cwd(), backend = "mlx", maxPrompts = DEFAULT_MAX_PROMPTS, maxTokens = DEFAULT_MAX_TOKENS, skipTuned = false, python = "python3", judge = null } = {}) {
  const outDir = path.resolve(cwd, config.output?.dir ?? "./.finetune");
  await mkdir(outDir, { recursive: true });
  const holdoutPath = path.join(outDir, "dataset.holdout.json");
  if (!existsSync(holdoutPath)) {
    throw new Error(`no holdout found at ${holdoutPath} — run "wasmtune dataset" first (newer versions write dataset.holdout.json).`);
  }
  const holdoutChunks = JSON.parse(await readFile(holdoutPath, "utf8"));
  const sftPath = path.join(outDir, "dataset.sft.jsonl");
  if (!existsSync(sftPath)) {
    throw new Error(`no SFT dataset at ${sftPath} — run "wasmtune dataset" first.`);
  }
  const sftPairs = (await readFile(sftPath, "utf8")).split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  const siteName = config.dataset?.siteName
    ?? (path.basename(path.resolve(cwd)) || "this website").replace(/[-_]+/g, " ");
  // Re-crawl source chunks (fast, deterministic) so faithfulness scoring can
  // trace output terms back to the originating chunk text.
  const { docs: evalDocs } = await collectTexts(config.dataDir, { cwd });
  const sourceByKey = new Map();
  for (const c of chunkDocs(evalDocs)) {
    sourceByKey.set(`${c.source}::${c.ordinal ?? 0}`, c.text);
  }
  const memoPrompts = buildMemorizationPrompts(sftPairs, { maxPrompts, sourceByKey });
  const genChunks = holdoutChunks.map((c) => ({ ...c }));
  const genPrompts = (await buildEvalPrompts(genChunks, { siteName, maxPrompts: Math.max(5, Math.floor(maxPrompts / 2)) }))
    .map((p) => {
      const chunk = holdoutChunks.find((c) => c.source === p.source);
      return { ...p, sourceText: String(chunk?.text ?? "").slice(0, 4000) };
    });
  const convPrompts = CONVERSE_PROBES.map((p) => ({ ...p, reference: "", source: "<converse>", kind: `conv-${p.kind}` }));
  if (!memoPrompts.length) throw new Error("no eval prompts could be built (corpus too small?).");
  const allPrompts = [...memoPrompts, ...genPrompts, ...convPrompts];
  const promptsPath = path.join(outDir, "eval.prompts.jsonl");
  await writeFile(promptsPath, allPrompts.map((r) => JSON.stringify(r)).join("\n") + "\n");

  await ensureVenv({ outDir, backend, cwd, python });
  const modelForBackend = backend === "mlx" ? (config.training?.mlxModel ?? config.model) : config.model;
  const runOne = async (label, adapters) => {
    const outPath = path.join(outDir, `eval.out.${label}.json`);
    const args = ["--backend", backend === "mlx" ? "mlx" : "cuda",
      "--model", modelForBackend, "--prompts", promptsPath, "--out", outPath,
      "--max-tokens", String(maxTokens)];
    if (adapters) args.push("--adapters", adapters);
    await runPython({ outDir, script: "python/eval_lm.py", args, cwd });
    return JSON.parse(await readFile(outPath, "utf8"));
  };

  const byId = Object.fromEntries(allPrompts.map((p) => [p.id, p]));
  const scoreRows = (rows) => rows.map((r) => {
    const ref = byId[r.id];
    const out = r.output ?? "";
    if (ref?.id?.startsWith("conv-")) {
      const kind = ref.kind.replace(/^conv-/, "");
      const c = scoreConversation(out, kind);
      return { id: r.id, prompt: ref?.prompt ?? "", kind: ref.kind, score: c.pass ? 1 : 0, conversePass: c.pass, converseReason: c.reason, repetition: hasRepetitionLoop(out), outputChars: out.length, output: out.slice(0, 2000) };
    }
    const s = scoreResponse(out, ref?.reference ?? "");
    const f = faithfulness(out, ref?.sourceText ?? "");
    return { id: r.id, prompt: ref?.prompt ?? "", ...s, faithfulness: f.score, untraced: f.untraced, output: out.slice(0, 2000) };
  });
  const split = (rows) => {
    const isMem = (r) => r.id.startsWith("mem-");
    const isConv = (r) => r.id.startsWith("conv-");
    return { mem: rows.filter(isMem), gen: rows.filter((r) => !isMem(r) && !isConv(r)), conv: rows.filter(isConv) };
  };
  const faithAvg = (rows) => {
    const f = rows.filter((r) => typeof r.faithfulness === "number");
    return f.length ? Math.round((f.reduce((a, r) => a + r.faithfulness, 0) / f.length) * 1000) / 1000 : null;
  };
  const convRate = (rows) => {
    const c = rows.filter((r) => typeof r.conversePass === "boolean");
    return c.length ? Math.round((c.filter((r) => r.conversePass).length / c.length) * 1000) / 1000 : null;
  };

  const baseRows = scoreRows(await runOne("base", null));
  const baseSplit = split(baseRows);
  const baseKnowledge = [...baseSplit.mem, ...baseSplit.gen];
  const base = {
    model: modelForBackend,
    memorization: summarize(baseSplit.mem),
    generalization: summarize(baseSplit.gen),
    converse: { passRate: convRate(baseSplit.conv), count: baseSplit.conv.length },
    faithfulness: faithAvg(baseKnowledge),
    ...summarize(baseKnowledge),
    results: baseRows,
  };

  let tuned = null;
  const adaptersDir = path.join(outDir, "run", "adapters");
  if (!skipTuned && existsSync(adaptersDir)) {
    const tunedRows = scoreRows(await runOne("tuned", adaptersDir));
    const tunedSplit = split(tunedRows);
    const tunedKnowledge = [...tunedSplit.mem, ...tunedSplit.gen];
    tuned = {
      model: modelForBackend, adapters: adaptersDir,
      memorization: summarize(tunedSplit.mem),
      generalization: summarize(tunedSplit.gen),
      converse: { passRate: convRate(tunedSplit.conv), count: tunedSplit.conv.length },
      faithfulness: faithAvg(tunedKnowledge),
      ...summarize(tunedKnowledge),
      results: tunedRows,
    };
  }

  const memDelta = tuned ? Math.round((tuned.memorization.avg - base.memorization.avg) * 1000) / 1000 : null;
  const delta = tuned ? Math.round((tuned.avg - base.avg) * 1000) / 1000 : null;
  const failOnRegression = config.eval?.failOnRegression ?? true;
  const regression = tuned ? memDelta < 0 : false;

  // Optional LLM judge (advisory only, never gates). Reuses synth-style
  // provider specs: --judge ollama:qwen3-4b.
  let judgeReport = null;
  const judgeSpec = judge ?? config.eval?.judge ?? null;
  if (judgeSpec) {
    const { provider, model } = parseJudgeSpec(judgeSpec);
    console.error(`[wasmtune] judge: ${provider}:${model} (advisory, memorization rows)`);
    const memIds = new Set(memoPrompts.map((p) => p.id));
    const collectOutputs = async (label) => {
      const out = JSON.parse(await readFile(path.join(outDir, `eval.out.${label}.json`), "utf8"));
      return out.filter((r) => memIds.has(r.id)).map((r) => ({
        id: r.id, prompt: byId[r.id]?.prompt ?? "",
        reference: byId[r.id]?.reference ?? "", output: r.output ?? "",
      }));
    };
    const judgeOne = async (label) => judgeRows(await collectOutputs(label), {
      provider, model,
      onProgress: (p) => {
        if (p.done % 10 === 0 || p.done === p.total) console.error(`[wasmtune] judge ${label} ${p.done}/${p.total}`);
      },
    });
    judgeReport = { spec: judgeSpec, base: await judgeOne("base") };
    if (tuned) judgeReport.tuned = await judgeOne("tuned");
  }

  const report = {
    created: new Date().toISOString(),
    prompts: promptsPath,
    memoPrompts: memoPrompts.length,
    genPrompts: genPrompts.length,
    convPrompts: convPrompts.length,
    base, tuned, delta, memDelta, regression, gate: failOnRegression ? !regression : true,
    judge: judgeReport,
  };
  const reportPath = path.join(outDir, "eval.report.json");
  await writeFile(reportPath, JSON.stringify(report, null, 2));

  // Self-improvement loop: mined loop contrasts for the next DPO run.
  const promptsById = byId;
  const loops = evalLoopContrasts(report, promptsById, {});
  const loopsPath = path.join(outDir, "eval.dpo-loops.jsonl");
  await writeFile(loopsPath, loops.map((r) => JSON.stringify(r)).join("\n") + (loops.length ? "\n" : ""));
  report.dpoLoops = { count: loops.length, file: loopsPath };
  await writeFile(reportPath, JSON.stringify(report, null, 2));
  if (loops.length) console.error(`[wasmtune] eval: ${loops.length} loop contrasts -> ${loopsPath} (used by train --method dpo)`);
  return { reportPath, report };
}

export function formatEvalTable(report) {
  const lines = [];
  const row = (label, s) => `  ${label.padEnd(8)} avg=${String(s.avg).padEnd(6)} n=${s.count} looped=${s.looped} faith=${s.faithfulness ?? "-"} conv=${s.converse?.passRate ?? "-"}`;
  const mem = (label, s) => `  ${label.padEnd(8)} mem=${s.memorization.avg} gen=${s.generalization.avg}`;
  lines.push(row("base", report.base));
  lines.push(mem("base", report.base));
  if (report.tuned) {
    lines.push(row("tuned", report.tuned));
    lines.push(mem("tuned", report.tuned));
    lines.push(`  memo-delta=${report.memDelta >= 0 ? "+" : ""}${report.memDelta}${report.regression ? "  REGRESSION" : ""}`);
  } else {
    lines.push("  tuned: (no adapters found, base only)");
  }
  if (report.judge) {
    const j = (s) => s ? `${s.avg} (n=${s.count})` : "-";
    lines.push(`  judge base=${j(report.judge.base)} tuned=${j(report.judge.tuned)} [advisory]`);
  }
  return lines.join("\n");
}
