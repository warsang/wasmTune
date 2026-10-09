// Score ONNX-generated completions with the same scorer `wasmtune eval` uses.
//
// The alternative — a second scorer in Python — would produce numbers that look
// comparable and are not, which is worse than no comparison at all.
//
//   node scripts/score-onnx-results.mjs <completions.json> <prompts.jsonl> <out.json>

import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { scoreResponse, scoreConversation, faithfulness, hasRepetitionLoop, summarize } from "../src/eval/score.mjs";

const [, , completionsPath, promptsPath, outPath] = process.argv;
if (!completionsPath || !promptsPath || !outPath) {
  console.error("usage: node scripts/score-onnx-results.mjs <completions.json> <prompts.jsonl> <out.json>");
  process.exit(2);
}

const completions = JSON.parse(readFileSync(completionsPath, "utf8"));
const byId = {};
for (const line of readFileSync(promptsPath, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const p = JSON.parse(line);
  byId[p.id] = p;
}

const rows = completions.map((c) => {
  const ref = byId[c.id];
  const out = c.output ?? "";
  if (ref?.id?.startsWith("conv-")) {
    const kind = ref.kind.replace(/^conv-/, "");
    const s = scoreConversation(out, kind);
    return { id: c.id, prompt: ref?.prompt ?? "", kind: ref.kind, score: s.pass ? 1 : 0, conversePass: s.pass, converseReason: s.reason, repetition: hasRepetitionLoop(out), outputChars: out.length, output: out.slice(0, 2000) };
  }
  const s = scoreResponse(out, ref?.reference ?? "");
  const f = faithfulness(out, ref?.sourceText ?? "");
  return { id: c.id, prompt: ref?.prompt ?? "", ...s, faithfulness: f.score, untraced: f.untraced, output: out.slice(0, 2000) };
});

const isMem = (r) => r.id.startsWith("mem-");
const isConv = (r) => r.id.startsWith("conv-");
const mem = rows.filter(isMem);
const gen = rows.filter((r) => !isMem(r) && !isConv(r));
const conv = rows.filter(isConv);
const knowledge = [...mem, ...gen];

const faithAvg = () => {
  const f = rows.filter((r) => typeof r.faithfulness === "number");
  return f.length ? Math.round((f.reduce((a, r) => a + r.faithfulness, 0) / f.length) * 1000) / 1000 : null;
};
const convRate = () => {
  const c = rows.filter((r) => typeof r.conversePass === "boolean");
  return c.length ? Math.round((c.filter((r) => r.conversePass).length / c.length) * 1000) / 1000 : null;
};

const report = {
  source: "onnx",
  memorization: summarize(mem),
  generalization: summarize(gen),
  converse: { passRate: convRate(), count: conv.length },
  faithfulness: faithAvg(),
  ...summarize(knowledge),
  results: rows,
};
writeFileSync(outPath, JSON.stringify(report, null, 2));

// Human-readable summary on the way out, matching `wasmtune eval`'s shape.
const p = (x) => (x == null ? "  -  " : String(x).padStart(5));
console.error(`  onnx     avg=${p(report.avg)} n=${report.count} looped=${report.looped} ` +
  `faith=${p(report.faithfulness)} conv=${p(report.converse.passRate)}`);
console.error(`  onnx     mem=${p(report.memorization.avg)} gen=${p(report.generalization.avg)}`);
