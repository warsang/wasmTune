// wasmtune — DPO + GRPO dataset helpers.
//
// DPO needs {prompt, chosen, rejected} triplets that teach SHAPE, not just
// facts. Auto generators (no user files required):
//   - conciseness: chosen = short answer, rejected = full source-chunk dump
//     (teaches: answer, don't regurgitate — the #1 v1 failure).
//   - faithfulness: chosen = gold answer, rejected = gold with two entities
//     swapped (teaches: don't hallucinate plausible-sounding facts).
//   - humility: chosen = ignorance fallback, rejected = long dump answer
//     (teaches: say "not in the docs" for out-of-scope prompts).
// `pairsFile` (config dpo.pairsFile) still augments/overrides via loadDpoPairs.

import { readFile } from "node:fs/promises";
import path from "node:path";

const SHORT_ANSWER = 600;

function splitPair(p) {
  const user = p.messages.find((m) => m.role === "user")?.content ?? "";
  const gold = p.messages.find((m) => m.role === "assistant")?.content ?? "";
  return { user, gold };
}

// Swap the first two distinct "entities" (numbers, code spans, Capitalized
// terms) to make a plausible-but-wrong answer. Returns null when fewer
// than two distinct entities exist.
export function swapEntities(answer) {
  const re = /(`[^`]{1,40}`|\b\d[\d,.]*\b|\b[A-Z][A-Za-z0-9_.!/-]{2,})/g;
  const found = [];
  const seen = new Set();
  let m;
  while ((m = re.exec(answer)) !== null) {
    if (!seen.has(m[0])) {
      seen.add(m[0]);
      found.push({ text: m[0], index: m.index });
    }
    if (found.length >= 6) break;
  }
  if (found.length < 2) return null;
  const [a, b] = found;
  const first = a.index < b.index ? a : b;
  const second = a.index < b.index ? b : a;
  return (
    answer.slice(0, first.index) + second.text +
    answer.slice(first.index + first.text.length, second.index) + first.text +
    answer.slice(second.index + second.text.length)
  );
}

// Auto DPO seeds from SFT pairs (+ optional source chunks for dump mining).
// Round-robins the three shaping generators for a balanced set. `chunks`
// enables conciseness triplets (rejected = full source text); without them
// only faithfulness + humility triplets are produced.
export function sftToDpoSeed(sftPairs, { maxPairs = 2000, chunks = [] } = {}) {
  const bySource = new Map();
  for (const c of chunks ?? []) {
    if (!bySource.has(c.source)) bySource.set(c.source, []);
    bySource.get(c.source).push(c);
  }
  const concise = [];
  const faithful = [];
  const humble = [];
  for (const p of sftPairs) {
    const { user, gold } = splitPair(p);
    if (!user || !gold) continue;
    const kind = p.meta?.kind ?? "";
    // 1. Conciseness: short gold answer vs full source-chunk dump.
    if (gold.length <= SHORT_ANSWER && kind !== "converse") {
      const dump = (bySource.get(p.meta?.source) ?? [])
        .filter((c) => (c.ordinal ?? 0) === (p.meta?.ordinal ?? 0))
        .map((c) => c.text)
        .find((t) => t && t.length > gold.length * 2);
      if (dump) {
        concise.push({
          prompt: user, chosen: gold, rejected: dump.slice(0, 2000),
          meta: { ...p.meta, kind: "dpo-concise" },
        });
      }
    }
    // 2. Faithfulness: gold vs entity-swapped plausible lie.
    if (gold.length <= SHORT_ANSWER && gold.length >= 40 && kind !== "converse") {
      const lie = swapEntities(gold);
      if (lie && lie !== gold) {
        faithful.push({
          prompt: user, chosen: gold, rejected: lie,
          meta: { ...p.meta, kind: "dpo-faithful" },
        });
      }
    }
    // 3. Humility: out-of-scope prompt -> fallback chosen, dump rejected.
    if (kind === "converse" && /not covered|don't have information|can't find/i.test(gold)) {
      const dumpPair = sftPairs.find(
        (q) => (q.meta?.kind === "explain") && splitPair(q).gold.length > 500,
      );
      const dump = dumpPair ? splitPair(dumpPair).gold.slice(0, 1200) : gold + " " + gold;
      humble.push({
        prompt: user, chosen: gold, rejected: dump,
        meta: { ...p.meta, kind: "dpo-humble" },
      });
    }
  }
  // Round-robin for balance; deterministic order.
  const out = [];
  const buckets = [concise, faithful, humble].filter((b) => b.length);
  let i = 0;
  while (out.length < maxPairs) {
    let advanced = false;
    for (const b of buckets) {
      if (i < b.length && out.length < maxPairs) {
        out.push(b[i]);
        advanced = true;
      }
    }
    if (!advanced) break;
    i++;
  }
  return out;
}

// Loop contrasts mined from an eval report: every TUNED completion that
// looped becomes {prompt, chosen: reference answer, rejected: looped output}.
// This closes the self-improvement loop: eval failures become DPO data, so
// the next `wasmtune train --method dpo` directly suppresses regurgitation
// spirals. Generic — no site content involved.
export function evalLoopContrasts(report, promptsById, { maxPairs = 200, maxRejectedChars = 1500 } = {}) {
  const out = [];
  const results = report?.tuned?.results ?? [];
  for (const r of results) {
    if (out.length >= maxPairs) break;
    if (!r.repetition) continue;
    const ref = promptsById?.[r.id];
    const reference = ref?.reference ?? "";
    const prompt = ref?.prompt ?? r.prompt ?? "";
    if (!prompt || !reference) continue;
    out.push({
      prompt,
      chosen: reference.slice(0, 800),
      rejected: String(r.output ?? r.full ?? "").slice(0, maxRejectedChars),
      meta: { kind: "dpo-loop", evalId: r.id, source: ref.source ?? "" },
    });
  }
  return out;
}

export async function loadDpoPairs(pairsFile, cwd = process.cwd()) {
  if (!pairsFile) return [];
  const full = path.resolve(cwd, pairsFile);
  let raw;
  try {
    raw = await readFile(full, "utf8");
  } catch {
    return [];
  }
  return raw
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

// GRPO needs {prompt} + a reward function (user-supplied JS module exporting
// `reward(prompt, completion)` -> number). v1 seeds prompts from SFT users.
export function sftToGrpoSeed(sftPairs, { maxPrompts = 2000 } = {}) {
  const out = [];
  const seen = new Set();
  for (const p of sftPairs) {
    if (out.length >= maxPrompts) break;
    const user = p.messages.find((m) => m.role === "user")?.content ?? "";
    if (!user || seen.has(user)) continue;
    seen.add(user);
    out.push({ prompt: user, meta: { ...p.meta, kind: "seed-grpo" } });
  }
  return out;
}

export const DEFAULT_REWARD_MJS = `// wasmtune GRPO reward stub — replace with your own scoring.
// Export reward(prompt, completion) -> number (higher is better).
export function reward(prompt, completion) {
  if (!completion || !completion.trim()) return 0;
  // Prefer answers that reuse distinctive source terms.
  const terms = String(prompt).toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length > 5).slice(0, 8);
  const lower = String(completion).toLowerCase();
  let score = Math.min(completion.length / 500, 1);
  for (const t of terms) if (lower.includes(t)) score += 0.2;
  return score;
}
`;
