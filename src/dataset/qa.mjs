// wasmtune — extractive Q&A synthesis (generic, heuristic, dependency-free).
//
// Turns chunks into short-answer QA pairs instead of excerpt regurgitation:
//   - headings    -> "What is {Title}?" (section body compressed to <=4 sentences)
//   - definitions -> "What is {Term}?" / "What does {Term} mean?"
//   - FAQs        -> literal `...?` lines with the following paragraph as answer
//   - how-tos     -> imperative steps become "How do I ...?" (max 1 per chunk)
// Chunks dominated by commands/code get flagged (commandsOnly) so the SFT mix
// can downweight them and DPO can mine them as rejected examples.

export const ANSWER_MAX_CHARS = 600;
export const ANSWER_MAX_SENTENCES = 4;

const COMMAND_LINE = /^(\s*(kd>|gdb>|\(gdb\)|\$|#|>|!|0:|\s{0,3}\d+[.)]\s+`?)|.*`[^`]+`.*$)/;
const IMPERATIVE = /^(run|use|open|type|enter|click|install|configure|create|add|set|enable|disable|check|verify|ensure|replace|copy|paste|save|load|boot|start|stop|restart|select|choose|pick|go to|navigate|scroll|press|hit|write|edit|delete|remove|build|compile|compile|deploy|publish|import|export)\b/i;
const DEFINITION = /^(.{3,60}?)\s+(is|are|refers? to|means?)\s+(.+)$/;
const TERM_LIKE = /^(`[^`]+`|"[^"]+"|[A-Z][A-Za-z0-9_.!:+/-]*(\s+[A-Z][A-Za-z0-9_.!:+/-]*)*|[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z0-9_]+)+)/;

export function splitSentences(text) {
  return String(text ?? "")
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-Z0-9`"'(#])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export function compressAnswer(text, { maxChars = ANSWER_MAX_CHARS, maxSentences = ANSWER_MAX_SENTENCES } = {}) {
  const out = [];
  let len = 0;
  for (const s of splitSentences(text)) {
    if (out.length >= maxSentences) break;
    if (s.length < 10) continue;
    if (len + s.length + 1 > maxChars && out.length) break;
    out.push(s);
    len += s.length + 1;
  }
  return out.join(" ");
}

// Fraction of non-empty lines that look like commands/prompts/code usage.
// Markdown headings (# Title) are never commands, even though a bare `#`
// can mean a root shell prompt elsewhere.
export function commandDensity(text) {
  const lines = String(text ?? "").split("\n").map((l) => l.trim()).filter(Boolean);
  if (!lines.length) return 0;
  let hits = 0;
  for (const l of lines) {
    if (/^#{1,4}\s/.test(l)) continue;
    if (COMMAND_LINE.test(l)) hits++;
  }
  return hits / lines.length;
}

function sectionBody(lines, startIdx) {
  const body = [];
  for (let i = startIdx; i < lines.length; i++) {
    const l = lines[i];
    if (/^#{1,4}\s+\S/.test(l.trim()) && body.length) break;
    body.push(l);
  }
  return body.join("\n").trim();
}

function cleanTitle(raw) {
  return raw.replace(/^#{1,4}\s+/, "").replace(/[*_`#>]/g, "").trim().slice(0, 80);
}

function isJunkTitle(t) {
  if (!t || t.length < 3) return true;
  // Pure code/hex/address headings carry no question value.
  if (/^(0x[0-9a-fA-F]+|!?\w+\s+0x[0-9a-fA-F]+|```)/.test(t)) return true;
  // Fragments and generic filler ("...and why...", "The lab", "Next steps").
  if (/^\.\.\./.test(t)) return true;
  if (/^(the|a|an)\s+(lab|page|section|overview|introduction|next|rest|goal|result|details|notes|example|part)\b/i.test(t)) return true;
  // Must start like a nameable thing: uppercase, backtick, digit, or quote.
  if (!/^([A-Z0-9`"'(#]|!)/.test(t)) return true;
  return false;
}

// Fenced code blocks -> {question, answer} pairs (generic, any docs site).
// Topic = nearest preceding markdown heading, else the first distinctive
// prose line; answer = capped code block (never a transcript dump).
export function extractCodeExamples(text, { maxAnswerChars = 1200, maxPairs = 3 } = {}) {
  const out = [];
  const lines = String(text ?? "").split("\n");
  let lastHeading = null;
  let lastProse = null;
  for (let i = 0; i < lines.length && out.length < maxPairs; i++) {
    const t = lines[i].trim();
    const h = t.match(/^#{1,4}\s+(.+)$/);
    if (h) {
      lastHeading = cleanTitle(h[1]);
      continue;
    }
    if (t.length > 30 && !t.startsWith("```") && !/^(kd>|gdb>|\(gdb\)|\$|#|>)/.test(t) && !lastProse) {
      lastProse = t.slice(0, 120);
    }
    const fence = t.match(/^```(\w*)\s*$/);
    if (!fence) continue;
    const lang = (fence[1] || "").toLowerCase();
    const body = [];
    let j = i + 1;
    for (; j < lines.length && !lines[j].trim().startsWith("```"); j++) body.push(lines[j]);
    i = j;
    const code = body.join("\n").trim();
    // Skip debugger transcripts, shell sessions, and stubs.
    if (lang === "kd" || lang === "gdb" || lang === "console" || lang === "text") continue;
    if (/^(kd>|gdb>|\(gdb\)|\$\s)/m.test(code)) continue;
    if (code.length < 24 || code.length > maxAnswerChars * 2) continue;
    if (!/[;{}()=#<>]/.test(code)) continue; // not actually code
    const topic = (!isJunkTitle(lastHeading) && lastHeading)
      ? lastHeading
      : (lastProse ?? "this topic");
    const langName = lang || "code";
    out.push({
      question: `Show example ${langName} code for ${topic}`,
      answer: code.slice(0, maxAnswerChars),
    });
    lastProse = null;
  }
  return out;
}

export function chunksToQa(chunks, { siteName = "this website", maxPairs = 5000 } = {}) {
  const pairs = [];
  const seenQuestions = new Set();
  const push = (question, answer, meta) => {
    const q = question.trim().replace(/\s+/g, " ");
    const a = (answer ?? "").trim();
    if (!q || !a || a.length < 20) return;
    const key = q.toLowerCase();
    if (seenQuestions.has(key)) return;
    if (pairs.length >= maxPairs) return;
    seenQuestions.add(key);
    pairs.push({
      messages: [
        { role: "user", content: q },
        { role: "assistant", content: a },
      ],
      meta,
    });
  };

  for (const c of chunks) {
    if (pairs.length >= maxPairs) break;
    const src = { source: c.source, ordinal: c.ordinal };
    const density = commandDensity(c.text);
    const meta = (kind) => ({ ...src, kind, commandsOnly: density >= 0.5 });

    // 0. Fenced code blocks -> "Show example code" pairs. Runs BEFORE the
    // command-density skip: code-heavy chunks are skipped below, but their
    // code examples are exactly what code questions need.
    for (const ex of extractCodeExamples(c.text)) {
      if (pairs.length >= maxPairs) break;
      push(ex.question, ex.answer, meta("code-qa"));
    }

    if (density >= 0.5) continue; // command dumps teach chat nothing; DPO mines them later

    const text = c.text;
    const lines = text.split("\n");

    // 1. Headings -> "What is {Title}?"
    for (let i = 0; i < lines.length && pairs.length < maxPairs; i++) {
      const t = lines[i].trim();
      if (!/^#{1,3}\s+\S/.test(t)) continue;
      const title = cleanTitle(t);
      if (isJunkTitle(title)) continue;
      const body = compressAnswer(sectionBody(lines, i + 1).replace(/^#{1,4}\s+.*$/gm, ""));
      if (!body) continue;
      if (/\?\s*$/.test(title)) {
        push(title, body, meta("faq"));
      } else {
        push(`What is ${title} on ${siteName}?`, body, meta("fact-qa"));
      }
    }

    // 2. Definitions -> "What is {Term}?"
    for (const s of splitSentences(text)) {
      if (pairs.length >= maxPairs) break;
      if (s.length > 300) continue;
      const m = s.match(DEFINITION);
      if (!m) continue;
      const term = m[1].replace(/[*_`"]/g, "").trim();
      if (!TERM_LIKE.test(term) || term.length > 60) continue;
      const answer = compressAnswer(s, { maxChars: 400, maxSentences: 2 });
      push(`What is ${term}?`, answer, meta("definition"));
    }

    // 3. FAQ lines -> literal question + following paragraph.
    for (let i = 0; i < lines.length && pairs.length < maxPairs; i++) {
      const l = lines[i].trim();
      if (!/\?\s*$/.test(l) || l.length < 12 || l.length > 220) continue;
      if (/^(kd>|gdb>|\(gdb\)|\$|#|>)/.test(l)) continue; // debugger transcripts aren't FAQs
      const answer = compressAnswer(lines.slice(i + 1, i + 6).join("\n"));
      push(l, answer, meta("faq"));
    }

    // 4. How-to: >=2 imperative lines -> one pair per chunk.
    const steps = lines.map((l) => l.trim()).filter((l) => l.length > 8 && IMPERATIVE.test(l.replace(/^(\d+[.)]\s+|[-*]\s+|`)/, "")));
    if (steps.length >= 2) {
      const first = steps[0].replace(/^(\d+[.)]\s+|[-*]\s+|`)/, "").replace(/`$/g, "");
      const answer = steps.slice(0, 5).map((s, i) => `${i + 1}. ${s.replace(/^(\d+[.)]\s+|[-*]\s+)/, "").slice(0, 160)}`).join("\n");
      push(`How do I ${first.charAt(0).toLowerCase() + first.slice(1, 140)}`, answer, meta("howto"));
    }
  }
  return pairs;
}
