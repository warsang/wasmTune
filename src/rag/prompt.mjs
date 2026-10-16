// wasmtune — turning retrieved passages into a prompt a 0.6-4B model can use.
//
// Two decisions here, both from the failure modes of small models:
//
// 1. The passages go in the *user* message, not in a second system message.
//    Chat templates for the sizes this package serves (Qwen, Gemma, SmolLM2)
//    commonly require strict role alternation and accept exactly one leading
//    system turn. A mid-conversation system message is either dropped or
//    rejected by the template, and the symptom is a model that answers as if
//    retrieval never happened.
//
// 2. Passages are numbered and the instruction is to cite the number. It gives
//    the reader a way to check the answer, and it gives the model a cheap,
//    well-defined job — copy the marker — which small models are much better at
//    than paraphrase-with-attribution.

export const DEFAULT_RAG_INSTRUCTIONS =
  "Answer from the numbered context below. Cite the numbers you used, like [1]. " +
  "If the context does not answer the question, say so plainly instead of guessing.";

export function formatPassages(passages) {
  return passages
    .map((p, i) => {
      const where = p.title ? ` — ${p.title}` : "";
      const url = p.url ? ` (${p.url})` : "";
      return `[${i + 1}]${where}${url}\n${p.text}`;
    })
    .join("\n\n");
}

/**
 * The user turn the model actually sees: context, then the question.
 *
 * Returns the question unchanged when nothing was retrieved — an empty context
 * block is worse than none, because it teaches the model that the grounding
 * section is sometimes noise.
 */
export function buildGroundedUserMessage(question, passages, { instructions = DEFAULT_RAG_INSTRUCTIONS } = {}) {
  if (!passages?.length) return question;
  return (
    `${instructions}\n\n<context>\n${formatPassages(passages)}\n</context>\n\n` +
    `Question: ${question}`
  );
}

/** The system prompt, with the grounding instruction appended once. */
export function withRagSystemPrompt(systemPrompt, { instructions = DEFAULT_RAG_INSTRUCTIONS } = {}) {
  const base = String(systemPrompt ?? "").trim();
  if (!base) return instructions;
  if (base.includes(instructions)) return base;
  return `${base} ${instructions}`;
}

/**
 * The sources line shown under an answer.
 *
 * Only the passages the answer actually cited, when it cited any; all of them
 * otherwise. Shown as text, not as an assertion that the answer is correct —
 * the reader is the one who checks.
 */
export function citedSources(answer, passages) {
  if (!passages?.length) return [];
  const cited = new Set();
  for (const m of String(answer ?? "").matchAll(/\[(\d{1,2})\]/g)) {
    const n = Number(m[1]);
    if (n >= 1 && n <= passages.length) cited.add(n - 1);
  }
  const indexes = cited.size ? [...cited].sort((a, b) => a - b) : passages.map((_, i) => i);
  const seen = new Set();
  return indexes
    .map((i) => passages[i])
    .filter((p) => p && !seen.has(p.url ?? p.id) && seen.add(p.url ?? p.id))
    .map((p) => ({ n: passages.indexOf(p) + 1, title: p.title ?? null, url: p.url ?? null }));
}
