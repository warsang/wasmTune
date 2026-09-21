// wasmtune — eval scoring (pure JS, deterministic, unit-tested).
//
// Generic signals computable from ANY site corpus (no site-specific content):
//   - recall: reference keywords present in output (memorization proxy)
//   - faithfulness: output's distinctive terms traceable to the source chunk
//     (hallucination proxy — invented commands/fields score ~0 without
//     needing to know which facts are true)
//   - repetition: 0 when the output loops (n-gram repeat)
//   - conciseness: penalty when output >> reference (rambling dump detector)
// score = recall * lengthFactor, forced to 0 on repetition.

export function extractKeywords(reference, { max = 8 } = {}) {
  const text = String(reference ?? "");
  const found = [];
  const seen = new Set();
  const push = (t) => {
    const k = t.toLowerCase();
    if (t && !seen.has(k) && k.length >= 2) {
      seen.add(k);
      found.push(t);
    }
  };
  // Numbers, code spans, and Capitalized terms first (most distinctive).
  const re = /(`[^`]{1,40}`|\b\d[\d,._-]*\b|\b[A-Z][A-Za-z0-9_.!/+:-]{2,})/g;
  let m;
  while ((m = re.exec(text)) !== null && found.length < max) push(m[0].replace(/`/g, ""));
  // Then distinctive long words.
  if (found.length < max) {
    const stop = new Set(("the,a,an,and,or,for,with,from,that,this,these,those,into,using,used,when,which,what,does,have,has,are,was,were,will,would,should,could,there,their,they,them,then,than,such,also,only,just,like,over,under,between,through,during,each,other,more,most,some,such,into,onto,upon,within,without,about,after,before,following,site,website,page,pages,following,explain,key,points").split(","));
    for (const w of text.toLowerCase().split(/[^a-z0-9_]+/)) {
      if (found.length >= max) break;
      if (w.length > 5 && !stop.has(w)) push(w);
    }
  }
  return found;
}

// True when output contains a repeated n-gram loop (the classic
// under-trained degeneration: same phrase 3+ times).
export function hasRepetitionLoop(output, { n = 8, repeats = 3 } = {}) {
  const tokens = String(output ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  if (tokens.length < n * repeats) return false;
  const counts = new Map();
  for (let i = 0; i + n <= tokens.length; i++) {
    const gram = tokens.slice(i, i + n).join(" ");
    const c = (counts.get(gram) ?? 0) + 1;
    if (c >= repeats) return true;
    counts.set(gram, c);
  }
  return false;
}

export function scoreResponse(output, reference) {
  const out = String(output ?? "");
  const ref = String(reference ?? "");
  const keywords = extractKeywords(ref);
  const lower = out.toLowerCase();
  const hits = keywords.filter((k) => lower.includes(k.toLowerCase()));
  const recall = keywords.length ? hits.length / keywords.length : (out.trim() ? 1 : 0);
  const repetition = hasRepetitionLoop(out);
  const lengthFactor = ref.length && out.length > ref.length * 3 ? 0.5 : 1;
  const score = repetition ? 0 : recall * lengthFactor;
  return {
    score: Math.round(score * 100) / 100,
    recall: Math.round(recall * 100) / 100,
    hits,
    missing: keywords.filter((k) => !lower.includes(k.toLowerCase())),
    keywords,
    repetition,
    lengthFactor,
    outputChars: out.length,
    referenceChars: ref.length,
  };
}

// Faithfulness: fraction of the OUTPUT's distinctive terms that appear in
// the source chunk. A grounded answer scores high even when the reference
// is short; invented commands/addresses/fields are untraced by construction.
export function faithfulness(output, sourceText, { max = 12 } = {}) {
  const terms = extractKeywords(output, { max });
  if (!terms.length) return { score: 1, traced: [], untraced: [], terms };
  const src = String(sourceText ?? "").toLowerCase();
  const traced = terms.filter((t) => src.includes(t.toLowerCase()));
  const untraced = terms.filter((t) => !src.includes(t.toLowerCase()));
  return {
    score: Math.round((traced.length / terms.length) * 100) / 100,
    traced,
    untraced,
    terms,
  };
}

// Conversational probes: generic behavioral checks needing no reference.
// kind "greeting": must be short and command-free (no transcript dumps).
// kind "fallback": must contain ignorance phrasing (no hallucinating).
// kind "scope": must be reasonably brief and command-free.
export function scoreConversation(output, kind) {
  const out = String(output ?? "");
  const lines = out.split("\n").map((l) => l.trim()).filter(Boolean);
  const cmdLines = lines.filter((l) => /^(!|kd>|gdb>|\(gdb\)|\$|#|>)/.test(l)).length;
  const density = lines.length ? cmdLines / lines.length : 0;
  if (kind === "greeting") {
    const pass = out.length <= 400 && density < 0.3;
    return { pass, reason: pass ? "short, no dump" : `len=${out.length} dump=${density.toFixed(2)}` };
  }
  if (kind === "fallback") {
    const pass = /(don't (have|know)|do not (have|know)|not (covered|in)|couldn't find|can't find|no information|don't cover)/i.test(out);
    return { pass, reason: pass ? "admits ignorance" : "no fallback phrasing" };
  }
  if (kind === "scope") {
    const pass = out.length <= 600 && density < 0.3;
    return { pass, reason: pass ? "brief, no dump" : `len=${out.length} dump=${density.toFixed(2)}` };
  }
  return { pass: false, reason: `unknown probe kind "${kind}"` };
}

// Fixed generic probe set: no site content needed.
export const CONVERSE_PROBES = [
  { id: "conv-greet-0", kind: "greeting", prompt: "Hi" },
  { id: "conv-greet-1", kind: "greeting", prompt: "Hey what's up?" },
  { id: "conv-fallback-0", kind: "fallback", prompt: "Who won the World Cup in 1503?" },
  { id: "conv-fallback-1", kind: "fallback", prompt: "What is the capital of Atlantis?" },
  { id: "conv-scope-0", kind: "scope", prompt: "What can you do?" },
];

export function summarize(scored) {
  const avg = (rows) => rows.length ? rows.reduce((a, r) => a + r.score, 0) / rows.length : 0;
  return {
    count: scored.length,
    avg: Math.round(avg(scored) * 1000) / 1000,
    looped: scored.filter((r) => r.repetition).length,
  };
}
