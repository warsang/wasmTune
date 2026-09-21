// wasmtune — optional LLM-judge scoring (generic, advisory only).
//
// Keyword/faithfulness metrics are cheap and deterministic but can't judge
// correctness. `--judge provider:model` (openai:*|ollama:*, reusing the
// synth provider interface) scores a sample of answers 0-2. Never gates;
// reported alongside the deterministic metrics.

export function parseJudgeSpec(spec) {
  const i = String(spec ?? "").indexOf(":");
  if (i === -1) throw new Error(`bad --judge spec "${spec}" (expected provider:model, e.g. ollama:qwen3-4b)`);
  const provider = spec.slice(0, i).toLowerCase();
  const model = spec.slice(i + 1);
  if (!["openai", "ollama"].includes(provider)) {
    throw new Error(`judge supports openai|ollama providers (got "${provider}")`);
  }
  if (!model) throw new Error(`bad --judge spec "${spec}" (missing model)`);
  return { provider, model };
}

export const JUDGE_SYSTEM =
  "You grade short answers against a reference. Reply with exactly one digit:\n" +
  "2 = correct (matches the reference's key facts), " +
  "1 = partially correct, " +
  "0 = wrong, hallucinated, or non-answer. " +
  "Reply with the digit only.";

export function parseJudgeScore(text) {
  const m = String(text ?? "").match(/[012]/);
  return m ? Number(m[0]) : null;
}

export async function judgeOne({ prompt, reference, output, provider, model, apiKey = null }) {
  const baseUrl = provider === "openai" ? "https://api.openai.com/v1" : "http://localhost:11434/v1";
  const key = provider === "openai" ? (apiKey ?? process.env.OPENAI_API_KEY) : null;
  if (provider === "openai" && !key) throw new Error("judge with openai needs OPENAI_API_KEY env");
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(key ? { authorization: `Bearer ${key}` } : {}),
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        { role: "system", content: JUDGE_SYSTEM },
        { role: "user", content: `Question: ${prompt}\nReference answer: ${reference}\nModel answer: ${output}\nGrade (0/1/2):` },
      ],
    }),
  });
  if (!res.ok) throw new Error(`judge request failed: ${res.status}`.slice(0, 200));
  const data = await res.json();
  return parseJudgeScore(data.choices?.[0]?.message?.content ?? "");
}

export async function judgeRows(rows, { provider, model, maxRows = 30, onProgress = null } = {}) {
  const out = [];
  const targets = rows.slice(0, maxRows);
  let i = 0;
  for (const r of targets) {
    i++;
    try {
      const score = await judgeOne({ prompt: r.prompt, reference: r.reference, output: r.output, provider, model });
      out.push({ id: r.id, judge: score });
    } catch (e) {
      out.push({ id: r.id, judge: null, judgeError: String(e.message ?? e).slice(0, 200) });
    }
    onProgress?.({ done: i, total: targets.length });
  }
  const scored = out.filter((r) => typeof r.judge === "number");
  return {
    rows: out,
    avg: scored.length ? Math.round((scored.reduce((a, r) => a + r.judge, 0) / scored.length) * 1000) / 1000 : null,
    count: scored.length,
  };
}
