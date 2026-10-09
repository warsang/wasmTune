// Evaluate the published ONNX artifact with the same tokenizer and the same
// scorer the rest of the package uses.
//
//   node scripts/onnx-eval.mjs --onnx-dir <dir> --prompts <jsonl> --out <report.json>
//
// The split is deliberate: tokenizing and scoring happen in Node with
// @huggingface/transformers and src/eval/score.mjs — the same code the browser
// and `wasmtune eval` use — while only the arithmetic runs in Python's
// onnxruntime. A second tokenizer or a second scorer would produce numbers that
// look comparable and are not.
//
// Why this script exists: `wasmtune eval` scores the merged fp16 weights, and
// the artifact the widget loads is a quantized ONNX graph. Those are different
// models. The published one said "In RAM on first run" where the fp16 one said
// the correct credentials path, and nothing caught it because nothing looked at
// the artifact that ships.

import { readFileSync, writeFileSync, createWriteStream } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { inlineChatTemplate } from "../src/publish.mjs";
import { scoreResponse, scoreConversation, faithfulness, hasRepetitionLoop, summarize } from "../src/eval/score.mjs";

const execFileAsync = promisify(execFile);
const [, , ...args] = process.argv;
const get = (name, dflt) => {
  const i = args.findIndex((a) => a === `--${name}`);
  return i !== -1 && args[i + 1] ? args[i + 1] : dflt;
};

const ONNX_DIR = path.resolve(get("onnx-dir", ".finetune/onnx"));
const PROMPTS = path.resolve(get("prompts", ".finetune/eval.prompts.jsonl"));
const OUT = path.resolve(get("out", ".finetune/eval.onnx.json"));
const DTYPE = get("dtype", "q8");
const MAX_TOKENS = Number(get("max-tokens", 64));
const SYSTEM_PROMPT = get("system-prompt", null) || null;
const PYTHON = get("python", "python3");
const TOKENIZER_DIR = get("tokenizer-dir", null);

const DTYPE_FILE = { q4: "model_q4.onnx", q4f16: "model_q4.onnx", q8: "model_quantized.onnx", fp32: "model.onnx" };
const graphPath = path.join(ONNX_DIR, DTYPE_FILE[DTYPE] ?? DTYPE_FILE.q8);
if (!existsSync(graphPath)) {
  console.error(`onnx-eval: ${graphPath} not present — run \`wasmtune convert\` first`);
  process.exit(0);
}

function existsSync(p) {
  try { readFileSync(p); return true; } catch { return false; }
}

// The tokenizer directory: convert leaves it beside onnx/, produced by export.py.
function findTokenizerDir() {
  if (TOKENIZER_DIR) return TOKENIZER_DIR;
  const root = path.dirname(ONNX_DIR);
  for (const c of [ONNX_DIR, root, path.join(root, "merged"), path.join(root, "fp16")]) {
    if (existsSync(path.join(c, "tokenizer.json"))) return c;
  }
  return null;
}

const prompts = readFileSync(PROMPTS, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));

const run = async () => {
  const tokDir = findTokenizerDir();
  if (!tokDir) {
    console.error(`onnx-eval: no tokenizer.json found next to ${ONNX_DIR}; pass --tokenizer-dir`);
    process.exit(0);
  }

  // transformers.js reads the chat template from tokenizer_config.json and
  // ignores a sibling .jinja, so without this the same failure that broke the
  // HF publish breaks here: apply_chat_template throws and the prompt falls
  // back to a shape the fine-tune never saw.
  inlineChatTemplate({ dir: tokDir });

  const { AutoTokenizer } = await import("@huggingface/transformers");
  const tok = await AutoTokenizer.from_pretrained(tokDir);
  console.error(`onnx-eval: tokenizer from ${tokDir}`);

  // apply_chat_template returns {input_ids: BigInt64Array}; the fallback path
  // returns a Tensor whose .data holds the ids. A plain spread walks object
  // properties and yields nothing, and JSON cannot carry BigInt to Python, so
  // normalise both shapes to plain numbers here.
  const toIds = (v) => {
    const inner = v?.input_ids ?? v;
    const arr = inner?.data ?? inner ?? v?.data ?? [];
    return Array.from(arr, Number);
  };

  // 1. Tokenize exactly as the browser will (same chat template, same ids).
  const rows = [];
  for (const p of prompts) {
    const msgs = (SYSTEM_PROMPT ? [{ role: "system", content: SYSTEM_PROMPT }] : [])
      .concat([{ role: "user", content: p.prompt }]);
    let ids;
    try {
      ids = tok.apply_chat_template(msgs, { tokenize: true, add_generation_prompt: true });
    } catch {
      ids = tok(msgs.map((m) => m.content).join("\n")).input_ids;
    }
    rows.push({ id: p.id, input_ids: toIds(ids) });
  }
  writeFileSync(`${OUT}.tokens.json`, JSON.stringify(rows));

  // 2. Generate through the graph.
  await execFileAsync(PYTHON, [
    "python/eval_onnx.py",
    "--onnx-dir", ONNX_DIR,
    "--input-tokens", `${OUT}.tokens.json`,
    "--out", `${OUT}.completions.json`,
    "--dtype", DTYPE,
    "--max-tokens", String(MAX_TOKENS),
  ], { env: { ...process.env, EVAL_ONNX_EOS: String(tok.eos_token_id ?? -1) } });

  const completions = JSON.parse(readFileSync(`${OUT}.completions.json`, "utf8"));
  const byId = Object.fromEntries(prompts.map((p) => [p.id, p]));

  // 3. Score with the shared scorer — not a second implementation.
  const scored = completions.map((c) => {
    const ref = byId[c.id];
    const output = tok.decode(c.output_ids, { skip_special_tokens: true });
    if (ref?.id?.startsWith("conv-")) {
      const kind = ref.kind.replace(/^conv-/, "");
      const s = scoreConversation(output, kind);
      return { id: c.id, prompt: ref?.prompt ?? "", kind: ref.kind, score: s.pass ? 1 : 0, conversePass: s.pass, converseReason: s.reason, repetition: hasRepetitionLoop(output), outputChars: output.length, output: output.slice(0, 2000) };
    }
    const s = scoreResponse(output, ref?.reference ?? "");
    const f = faithfulness(output, ref?.sourceText ?? "");
    return { id: c.id, prompt: ref?.prompt ?? "", ...s, faithfulness: f.score, untraced: f.untraced, output: output.slice(0, 2000) };
  });

  const isMem = (r) => r.id.startsWith("mem-");
  const isConv = (r) => r.id.startsWith("conv-");
  const mem = scored.filter(isMem);
  const gen = scored.filter((r) => !isMem(r) && !isConv(r));
  const conv = scored.filter(isConv);
  const knowledge = [...mem, ...gen];

  const faithAvg = () => {
    const f = scored.filter((r) => typeof r.faithfulness === "number");
    return f.length ? Math.round((f.reduce((a, r) => a + r.faithfulness, 0) / f.length) * 1000) / 1000 : null;
  };
  const convRate = () => {
    const c = scored.filter((r) => typeof r.conversePass === "boolean");
    return c.length ? Math.round((c.filter((r) => r.conversePass).length / c.length) * 1000) / 1000 : null;
  };

  const report = {
    source: "onnx",
    artifact: graphPath,
    dtype: DTYPE,
    memorization: summarize(mem),
    generalization: summarize(gen),
    converse: { passRate: convRate(), count: conv.length },
    faithfulness: faithAvg(),
    ...summarize(knowledge),
    results: scored,
  };
  writeFileSync(OUT, JSON.stringify(report, null, 2));

  // Compare against the fp16 eval, which is the only thing this used to be
  // confused with.
  const fp16Path = path.join(path.dirname(OUT), "eval.report.json");
  let fp16 = null;
  try { fp16 = JSON.parse(readFileSync(fp16Path, "utf8")).tuned; } catch { /* absent */ }

  console.error(`  onnx     avg=${p(report.avg)} n=${report.count} looped=${report.looped} ` +
    `faith=${p(report.faithfulness)} conv=${p(report.converse.passRate)}`);
  console.error(`  onnx     mem=${p(report.memorization.avg)} gen=${p(report.generalization.avg)}`);
  if (fp16) {
    console.error(`  fp16     avg=${p(fp16.avg)} n=${fp16.count} looped=${fp16.looped} ` +
      `faith=${p(fp16.faithfulness)} conv=${p(fp16.converse?.passRate)}`);
    console.error(`  fp16     mem=${p(fp16.memorization?.avg)} gen=${p(fp16.generalization?.avg)}`);
    const d = (x, y) => (x == null || y == null ? "?" : (Number(y) - Number(x)).toFixed(3));
    console.error(`  quantization delta  avg=${d(fp16.avg, report.avg)}  ` +
      `faith=${d(fp16.faithfulness, report.faithfulness)}`);
  }
  console.error(`onnx-eval: wrote ${OUT}`);
};

const p = (x) => (x == null ? "  -  " : String(x).padStart(5));
run().catch((e) => { console.error(`onnx-eval failed: ${e.message}`); process.exit(1); });
