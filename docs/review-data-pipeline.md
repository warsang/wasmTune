# Review: what the last few days showed, and what wasmtune should absorb

Two separate pieces of work. First the research question, then an audit of
everything I ended up doing by hand that the library should have done.

---

## 1. Does converting HTML to Markdown help fine-tuning accuracy?

**Short answer: the evidence supports your hypothesis, but the useful form of
it is more specific than "markdown".**

### What the papers actually measure

**AICC / MinerU-HTML** — [arXiv:2511.16397](https://arxiv.org/abs/2511.16397) (Nov 2025) is the closest thing to a direct answer. It isolates *extraction quality* while holding filtering constant, which is exactly the variable you are asking about:

- Heuristic extractors (they name **Trafilatura**) "struggle to preserve
  document structure and frequently corrupt structured elements such as
  formulas, codes, and tables".
- Their pipeline "explicitly categorizes semantic elements **before converting
  to Markdown**" — i.e. Markdown is the target representation, chosen
  because it can carry structure.
- On MainWebBench (7,887 annotated pages): **81.8% ROUGE-N F1 vs Trafilatura's
  63.6%**, with 90.9% code-block and 94.0% formula preservation.
- Controlled pretraining run (62B tokens, **identical filtering** on both
  sides): **+1.08pp average accuracy across 13 benchmarks** from extraction
  alone.
- Their conclusion: "HTML extraction is a critical, often underestimated
  component of web corpus construction."

**Nemotron-CC-Math** — [arXiv:2508.15096](https://arxiv.org/abs/2508.15096) is
independent corroboration and adds boilerplate removal:
- "brittle extraction heuristics, **lossy HTML-to-text conversion**, and the
  failure to reliably preserve mathematical structure".
- They recover maths from MathJax/KaTeX/MathML, **remove boilerplate**, and
  standardise notation into LaTeX.
- Gains: **+4.8 to +12.6 on MATH**, **+4.6 to +14.3 on MBPP+**, while also
  improving general-domain MMLU.

**Precedents for shipping Markdown corpora:** the Greek government decisions
dataset ([arXiv:2512.05647](https://arxiv.org/abs/2512.05647)) is *published*
in Markdown with a reproducible extraction pipeline and explicitly pitched as
fine-tuning material. **HPLT 3.0** ([arXiv:2511.01066](https://arxiv.org/abs/2511.01066))
treats HTML text extraction as a first-class pipeline stage alongside quality
scores and register labels.

**A methodological caveat worth keeping:** the PDF-to-Markdown benchmark
([arXiv:2602.11960](https://arxiv.org/abs/2602.11960)) warns that benchmarks
often "over-penalize benign formatting and linearization choices ... that are
largely irrelevant for downstream use." The correct test is *downstream
usefulness*, not cosmetic fidelity.

### What this means for us, concretely

The mechanism is **structure preservation**, and Markdown is a good carrier of
it because HTML→Markdown is a near-lossless mapping for the things that
matter: `<h2>` → `##`, `<pre><code>` → fenced block, `<table>` → pipe table.

And this is not hypothetical for wasmtune — it is a live bug.
`src/dataset/collect.mjs` `cleanText()` does:

```js
t = t.replace(/<script[\s\S]*?<\/script>/gi, " ")
     .replace(/<style[\s\S]*?<\/style>/gi, " ")
     .replace(/<[^>]+>/g, " ");   // every tag, gone
```

So `dataDir: ./public_html/` produces exactly the failure the papers describe:

| what the paper says matters | what happens now |
|---|---|
| headings | flattened — `<h2>Rate limits</h2>` becomes a bare line with no level |
| code blocks | fences stripped, so API curl examples lose their delimiters |
| tables | pipe tables flatten to a single line of words |
| formulas | lost |
| boilerplate | **kept** — nav, footer, cookie banners and (for a site built like our demo) the entire tier-panel and comparison matrix all end up in the corpus |

That last row is the one that would bite hardest in practice. Point `dataDir`
at a typical marketing site and you train on the navigation menu 50 times.

**Recommendation:** convert HTML → Markdown in the `dataset` step, not strip
tags. The evidence says this is worth roughly 1pp on average benchmarks when
filtering is already equal — and for a 51-pair corpus generated from 6 pages,
the corpus *is* the whole quality budget, so the relative gain is much larger
than that number suggests.

---

## 2. Audit: everything I did by hand that the library should absorb

I went from "a couple of commands" to five distinct failure classes. Ranked by
how badly each one breaks the promise.

### Fixed during this session (so the workflow is less broken)

| gap | what I had to do | now |
|---|---|---|
| Training used a different chat format than inference | hand-write the format, retrain | trainers use `apply_chat_template` |
| Training used a different system prompt than the widget | inject it by hand | prompt comes from `options.mjs` via `train.args.json` |
| `workerUrl` / `cloudUrl` silently dropped | pass `workerUrl` and discover it 404'd | `connectedCallback` re-reads both |
| ONNX exports 43% larger than needed | dedupe by hand | dedupe runs inside `convert` |
| Published model | hand-rolled upload scripts | `wasmtune publish` |
| Merging adapters | `export.py` exists but nothing called it | `convert` calls it |
| Multiple WebLLM ids that don't exist | hand-check against the registry | CI would have caught it (no CI existed); test added |
| Tokenizer chat template | inlined by hand | `publish` inlines it |
| Windows test bug | nothing | CI added |

### Still open, in the order I'd fix them

**1. ONNX dtype contract (highest impact — breaks the out-of-the-box path)**

`convert` writes whatever optimum emits (`model.onnx`, fp32). The worker asks
transformers.js for a dtype. Ask for `q8` and it looks for
`model_quantized.onnx` — which nobody produced. Result: 404 → engine
`unavailable` → "model never loads", with nothing pointing at a filename.

A user who follows the docs end to end gets a broken ONNX tier today. The
whole reason I shipped a q8 graph is that I quantized it myself.

*Fix:* quantize inside `convert` (ORT `quantize_dynamic` + the existing
dedupe), name the files by dtype, and write the manifest entry with the dtype
that was actually produced.

**2. HTML → Markdown extraction** (section 1 above; evidence-backed)

**3. `dataset.siteName` vs the widget's `siteName`**

Two places to spell the site's name, and they are not linked. If they
disagree, the model is trained on a prompt it will never be given — which is
precisely the class of bug that cost two full retrain cycles here.

*Fix:* derive the widget's `siteName` default from the manifest, or have
`convert` warn when the two differ.

**4. `publish` is not wired where it needs to be**

`build` deliberately excludes it (correct — retraining is nondeterministic,
so a build hook would churn the published weights and re-download for every
client). But nothing closes the loop either: `publish` prints the manifest
line to paste rather than writing it. And nothing verifies the upload landed.

*Fix:* `publish` writes the manifest entry itself and verifies the hash of
what it uploaded.

**5. Quantization accuracy is a silent quality knob**

What ships (q8) is not what I measured locally (fp16). Nothing in the
workflow tells a user that quantizing just changed their model's answers.
That's the difference between "answers are 85% right" and "answers are 40%
right", invisible until someone asks it a question.

*Fix:* run `eval` against the **quantized** artifact, not the fp16 weights,
and surface the delta.

---

## The workflow we want vs the workflow we have

**Target (from the README's own promise):**

```bash
npm i -D wasmtune
npx wasmtune init && npx wasmtune build
```

```html
<script type="module">
  import { mountAssistant } from "wasmtune/chat";
  await mountAssistant({ target: "#chat", siteName: "Docs" });
</script>
<div id="chat"></div>
```

**What actually happened:**

```bash
# ... fix two trainer-format bugs, retrain twice
# ... pick a base model, discover 135M can't hold the facts
# ... hand-quantize to q8, hand-dedupe, verify bit-identical logits
# ... publish to HF by hand
# ... discover the ONNX filename the widget asks for is not the one convert writes
```

Nothing here is exotic — every step is the sort of thing a first-time user
would hit. That's the gap worth closing, and the four items in "still open"
are the list.

*199/199 tests pass; the two new ones document gaps rather than fix them.*
