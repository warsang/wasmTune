// wasmtune — SFT blend (generic).
//
// Lesson learned the hard way: excerpt-regurgitation pairs teach the model to
// dump text, hallucinate fluently, and loop. So the blend is QA-heavy with
// SHORT answers, plus conversational seeds and a small concise-explain share:
//   ~60% extractive QA (fact-qa / definition / faq / howto)
//   ~10% conversational seeds (greetings, scope, ignorance fallbacks)
//   ~30% concise explain (capped targets with an explicit brevity instruction)
// Buckets are interleaved round-robin for a deterministic, reproducible order.

import { chunksToQa, compressAnswer, commandDensity } from "./qa.mjs";
import { conversationalSeeds } from "./converse.mjs";

export const BLEND = { qa: 0.6, converse: 0.1, explain: 0.3 };
export const EXPLAIN_MAX_CHARS = 800;

export function explainPairs(chunks, { siteName = "this website", maxPairs = 5000 } = {}) {
  const pairs = [];
  for (const c of chunks) {
    if (pairs.length >= maxPairs) break;
    if (commandDensity(c.text) >= 0.5) continue;
    const answer = compressAnswer(c.text, { maxChars: EXPLAIN_MAX_CHARS, maxSentences: 4 });
    if (answer.length < 60) continue;
    const excerpt = c.text.slice(0, 800);
    pairs.push({
      messages: [
        {
          role: "user",
          content: `Summarize the key points of this ${siteName} excerpt in 2-4 sentences:\n\n${excerpt}`,
        },
        { role: "assistant", content: answer },
      ],
      meta: { source: c.source, ordinal: c.ordinal, kind: "explain" },
    });
  }
  return pairs;
}

export function buildSftBlend(chunks, { siteName = "this website", summary = null, maxPairs = 5000, blend = BLEND, extraPairs = [] } = {}) {
  const qaTarget = Math.floor(maxPairs * (blend.qa ?? BLEND.qa));
  const convTarget = Math.floor(maxPairs * (blend.converse ?? BLEND.converse));
  const qa = chunksToQa(chunks, { siteName, maxPairs: qaTarget * 2 }).slice(0, qaTarget);
  const conv = conversationalSeeds({ siteName, summary }).slice(0, Math.max(convTarget, 1));
  const extra = extraPairs.slice(0, maxPairs);
  const explainTarget = Math.max(0, maxPairs - qa.length - conv.length - extra.length);
  const explain = explainPairs(chunks, { siteName, maxPairs: explainTarget * 2 }).slice(0, explainTarget);
  // Round-robin interleave for deterministic mixed order.
  const buckets = [qa, extra, conv, explain].filter((b) => b.length);
  const out = [];
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

export function answerCharsStats(pairs) {
  const lens = pairs
    .map((p) => p.messages.find((m) => m.role === "assistant")?.content?.length ?? 0)
    .sort((a, b) => a - b);
  if (!lens.length) return { count: 0, p50: 0, p90: 0, max: 0 };
  const q = (f) => lens[Math.min(lens.length - 1, Math.floor(f * lens.length))];
  return { count: lens.length, p50: q(0.5), p90: q(0.9), max: lens[lens.length - 1] };
}

export function kindCounts(pairs) {
  const counts = {};
  for (const p of pairs) {
    const k = p.meta?.kind ?? "unknown";
    counts[k] = (counts[k] ?? 0) + 1;
  }
  return counts;
}

export function toJsonl(rows) {
  return rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : "");
}
