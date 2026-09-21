// wasmtune — optional LLM QA synthesis (generic).
//
// Heuristic pairs (qa.mjs) are the default. `--synth provider:model` adds
// higher-quality pairs from an LLM that reads each chunk:
//   - openai:gpt-4o-mini  -> OpenAI chat completions (needs OPENAI_API_KEY)
//   - ollama:qwen3-4b     -> local Ollama OpenAI-compatible endpoint (no key)
//   - mlx:<hf-or-mlx-id>  -> venv `mlx_lm.generate` per chunk (slow, offline)
// The model must only use facts stated in the chunk, keep answers <=4
// sentences, and include one explicitly-unanswerable pair per chunk.

export const SYNTH_SYSTEM = (siteName, pairsPerChunk) =>
  `You write training questions for the ${siteName} site assistant. ` +
  `Given a documentation excerpt, output a JSON array of exactly ${pairsPerChunk + 1} ` +
  `objects with "q" and "a" keys: ${pairsPerChunk} question/answer pairs whose ` +
  `answers use ONLY facts stated in the excerpt (each answer at most 4 sentences), ` +
  `plus one final pair where "q" is a plausible question the excerpt does NOT ` +
  `answer and "a" says the docs don't cover it. Output JSON only, no prose.`;

export function parseSynthOutput(text) {
  const raw = String(text ?? "").trim();
  if (!raw) return [];
  // Prefer a JSON array, tolerating fences and surrounding prose.
  const jsonMatch = raw.match(/\[[\s\S]*\]/);
  if (jsonMatch) {
    try {
      const arr = JSON.parse(jsonMatch[0]);
      if (Array.isArray(arr)) {
        return arr
          .filter((r) => r && typeof r.q === "string" && typeof r.a === "string")
          .map((r) => ({ q: r.q.trim(), a: r.a.trim() }))
          .filter((r) => r.q && r.a);
      }
    } catch {
      /* fall through to Q:/A: parsing */
    }
  }
  const out = [];
  let cur = null;
  for (const line of raw.split("\n")) {
    const qm = line.match(/^\s*(?:Q\d*|question)\s*[:.-]\s*(.+)$/i);
    const am = line.match(/^\s*(?:A\d*|answer)\s*[:.-]\s*(.+)$/i);
    if (qm) {
      if (cur?.q && cur?.a) out.push(cur);
      cur = { q: qm[1].trim(), a: "" };
    } else if (am && cur) {
      cur.a = (cur.a ? cur.a + " " : "") + am[1].trim();
    } else if (cur?.q && line.trim()) {
      cur.a = (cur.a ? cur.a + " " : "") + line.trim();
    }
  }
  if (cur?.q && cur?.a) out.push(cur);
  return out;
}

export function parseSynthSpec(spec) {
  const i = String(spec ?? "").indexOf(":");
  if (i === -1) throw new Error(`bad --synth spec "${spec}" (expected provider:model, e.g. openai:gpt-4o-mini)`);
  const provider = spec.slice(0, i).toLowerCase();
  const model = spec.slice(i + 1);
  if (!["openai", "ollama", "mlx"].includes(provider)) {
    throw new Error(`unknown synth provider "${provider}" (expected openai|ollama|mlx)`);
  }
  if (!model) throw new Error(`bad --synth spec "${spec}" (missing model)`);
  return { provider, model };
}

async function chatCompletions({ baseUrl, apiKey, model, system, user }) {
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}),
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });
  if (!res.ok) throw new Error(`synth request failed: ${res.status} ${await res.text().catch(() => "")}`.slice(0, 300));
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content ?? "";
  if (!text) throw new Error("synth request returned no content");
  return text;
}

async function synthViaMlx({ venvPy, model, system, chunkText }) {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const prompt = `${system}\n\nExcerpt:\n${chunkText.slice(0, 3000)}`;
  const { stdout } = await promisify(execFile)(
    venvPy, ["-m", "mlx_lm.generate", "--model", model, "--prompt", prompt, "--max-tokens", "800", "--temp", "0.2"],
    { maxBuffer: 4 * 1024 * 1024 },
  );
  // mlx_lm.generate prints stats around the completion; take the longest line block.
  return stdout;
}

export async function synthQaForChunk(chunkText, { provider, model, siteName, pairsPerChunk = 3, venvPy = null, apiKey = null }) {
  const system = SYNTH_SYSTEM(siteName, pairsPerChunk);
  let text;
  if (provider === "openai") {
    const key = apiKey ?? process.env.OPENAI_API_KEY;
    if (!key) throw new Error("openai synth needs OPENAI_API_KEY env");
    text = await chatCompletions({ baseUrl: "https://api.openai.com/v1", apiKey: key, model, system, user: chunkText.slice(0, 4000) });
  } else if (provider === "ollama") {
    text = await chatCompletions({ baseUrl: "http://localhost:11434/v1", model, system, user: chunkText.slice(0, 4000) });
  } else if (provider === "mlx") {
    if (!venvPy) throw new Error("mlx synth needs a venv python (run inside `finetune dataset`)");
    text = await synthViaMlx({ venvPy, model, system, chunkText });
  } else {
    throw new Error(`unknown synth provider "${provider}"`);
  }
  return parseSynthOutput(text);
}

export async function synthQa(chunks, { spec, siteName = "this website", pairsPerChunk = 3, maxChunks = 200, venvPy = null, onProgress = null } = {}) {
  const { provider, model } = parseSynthSpec(spec);
  const pairs = [];
  const seen = new Set();
  const targets = chunks.filter((c) => (c.text ?? "").length >= 400).slice(0, maxChunks);
  let i = 0;
  for (const c of targets) {
    i++;
    try {
      const rows = await synthQaForChunk(c.text, { provider, model, siteName, pairsPerChunk, venvPy });
      for (const r of rows) {
        const key = r.q.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        pairs.push({
          messages: [
            { role: "user", content: r.q },
            { role: "assistant", content: r.a.slice(0, 800) },
          ],
          meta: { source: c.source, ordinal: c.ordinal, kind: "synth-qa", provider, model },
        });
      }
    } catch (e) {
      onProgress?.({ done: i, total: targets.length, error: e.message, source: c.source });
      continue;
    }
    onProgress?.({ done: i, total: targets.length, pairs: pairs.length, source: c.source });
  }
  return pairs;
}
