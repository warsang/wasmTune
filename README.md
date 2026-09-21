# WasmTune

Generic local fine-tuning + WebGPU chat for **any website folder**.

1. Point a config at the folder holding your site content.
2. `npx wasmtune dataset && npx wasmtune train` — LoRA/QLoRA locally.
   - NVIDIA Linux/Windows → **Unsloth + TRL + PEFT** (CUDA kernels, fastest).
   - Apple Silicon (M1–M5) → **MLX** via `mlx-lm` / `unsloth-mlx` (unified memory).
3. `npx wasmtune convert` — merged weights → GGUF + MLC (`q4f16_1`) + ONNX.
4. Embed `<site-chat>` — **WebLLM (WebGPU) primary**, Transformers.js and wllama fallback. Served locally, no server inference. Ship several tiers (`models[]`) and the browser loads the best one its hardware can run.

Real Unsloth is CUDA-only and does not train on Metal — that is why this
package routes by platform instead of pretending one backend fits all.

## Quick start (any repo)

```bash
npm i -D wasmtune
npx wasmtune init
# edit wasmtune.config.json: dataDir, model, method (finetune.config.json also accepted)
npx wasmtune check
npx wasmtune dataset
npx wasmtune train
npx wasmtune eval    # gate: tuned must beat base
npx wasmtune convert
npx wasmtune serve
```

No config file? Pass the essentials as flags instead — same result:
```bash
npx wasmtune dataset --dataDir ./docs --model Qwen/Qwen2.5-1.5B-Instruct
npx wasmtune train --dataDir ./docs --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
```
Flaggable knobs: `--dataDir/--model/--method/--backend/--epochs/--lr/--batch-size/--quant/--out-dir/--web-dir`.
With a file present the same flags override it. Precedence: flags > file > defaults.
Exotic keys (LoRA alpha, GRPO rewards, judge providers) stay file-only.

## Data quality doctrine

Excerpt-regurgitation pairs teach models to dump text, hallucinate fluently,
and loop. So `wasmtune dataset` builds a QA-heavy blend with **short answers**:

- ~60% extractive Q&A (headings/definitions/FAQs/how-tos, answers ≤ ~600 chars)
- ~10% conversational seeds (greetings, scope, "not in the docs" fallbacks)
- ~30% concise summaries (capped, brevity-instructed)
- Optional `--synth provider:model` LLM pass (`openai:*`, `ollama:*`, `mlx:*`)
  for higher-quality pairs: `npx wasmtune dataset --synth ollama:qwen3-4b`

`dataset.report.json` proves the mix (`kinds`, `answerChars.p50/p90`).
DPO seeds are auto-generated shaping triplets (concise-vs-dump,
true-vs-entity-swapped, fallback-vs-dump) so SFT→DPO works with no manual files.

## Eval

`npx wasmtune eval` builds prompts from a deterministic 5% holdout
(`dataset.holdout.json`, never trained on), runs base + tuned adapters, and
scores keyword recall with repetition-loop detection and rambling penalties:

```
  base     avg=0.198  n=50 looped=0
  tuned    avg=0.260  n=50 looped=0
  delta=+0.062
```

Exit code is non-zero on regression (disable with `eval.failOnRegression=false).
Flags: `--max-prompts N`, `--max-tokens N`, `--skip-tuned`, `--backend ...`.

Embed (one call — manifest, engine pick, banner, and widget handled):

```html
<script type="module">
  import { mountAssistant } from 'wasmtune/chat';
  await mountAssistant({ target: '#chat', baseModel: 'Qwen3-0.6B-q4f16_1-MLC' });
</script>
<div id="chat"></div>
```

Or wire `<site-chat>` yourself:

```html
<script type="module">
  import { defineSiteChat } from 'wasmtune/chat';
  defineSiteChat({
    gguf: '/models/my-site-abc12345/my-site.Q4_K_M-00001-of-00009.gguf',
    temperature: 0.3, repetitionPenalty: 1.15,
    // Hybrid-reasoning models (Gemma 4, Qwen3.5) think by default; the
    // widget also strips thinking blocks at display as a backstop.
    templateKwargs: { enable_thinking: false },
  });
</script>
<site-chat></site-chat>
```

Framework adapters (props mirror `mountAssistant` options):

```jsx
// React
import { SiteChat } from 'wasmtune/chat/react';
<SiteChat manifestUrl="/models/model-manifest.json" temperature={0.3}
  onReady={({ engine }) => console.log("ready:", engine)}
  onLoadFailed={(detail) => report(detail)} />;
```

```vue
<!-- Vue -->
<script setup>import { SiteChat } from 'wasmtune/chat/vue';</script>
<template>
  <SiteChat manifest-url="/models/model-manifest.json"
    @ready="console.log" @load-failed="report" />
</template>
```

```svelte
<!-- Svelte -->
<script>import SiteChat from 'wasmtune/chat/svelte';</script>
<SiteChat on:ready={console.log} on:load-failed={report} />
```

See `templates/wasmtune.config.example.json` and the Config section below.

## Config

`wasmtune.config.json` (or `.js`/`.mjs` for GRPO reward functions):

```json
{
  "dataDir": "./docs",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "method": "sft",
  "training": {
    "backend": "auto",
    "lora": { "r": 16, "alpha": 16, "dropout": 0.05 },
    "quantization": "q4",
    "epochs": 2, "batchSize": 2, "gradAccum": 4,
    "lr": 0.0002, "maxSeqLen": 2048, "seed": 42
  },
  "chat": {
    "temperature": 0.3, "repetitionPenalty": 1.15,
    "maxTokens": 256, "systemPrompt": null
  },
  "output": { "dir": "./.finetune", "webDir": "./public/models" }
}
```

Serve several model tiers and let the browser pick the best one its hardware
can run (see "Model tiers & hardware detection"):

```jsonc
{
  "dataDir": "./docs",
  "models": [
    // trained by the pipeline (one run per entry, artifacts under .finetune/<slug>/)
    { "model": "google/gemma-4-E4B-it", "label": "Gemma 4 E4B" },
    // served as-is from its public browser build — no training, no GPU
    { "model": "Qwen/Qwen3-0.6B", "trained": false },
    // optional: a direct .gguf URL for a custom pretrained tier
    { "model": "Qwen/Qwen3-1.7B", "trained": false, "gguf": "https://example.com/qwen3-1.7b.Q4_K_M.gguf" }
  ],
  "model": "google/gemma-4-E4B-it", // optional; defaults to models[0]
  "method": "sft",
  "chat": { "temperature": 0.3 }
}
```

| Key | Meaning |
|---|---|
| `dataDir` | Folder crawled for FT data (md/mdx/txt/html/js/mjs/json). |
| `dataset.siteName` | Display name used in generated questions/prompts (default: folder name). |
| `dataset.summary` | One-line site summary for conversational seeds (default: inferred). |
| `model` | Base model HF id. Must be on the small-model allowlist (`npx wasmtune models`). |
| `models` | Optional tier list. `{ model, label?, trained?, gguf?, onnx?, webllm?, chat?, requirements? }`. `trained: false` entries are published from their public browser artifacts (no training). `requirements` overrides the lib's known hardware needs (e.g. `{ "minDeviceMemoryGB": 6 }`). |
| `method` | `sft` \| `dpo` \| `orpo` \| `grpo`. DPO/ORPO need triplets (`dpo.pairsFile` + auto seeds + eval-mined loops); GRPO needs `grpo.rewardFile`. ORPO needs no reference model — prefer it on memory-tight Macs. |
| `training.backend` | `auto` (recommended) \| `unsloth` \| `mlx`. `auto` picks MLX on darwin-arm64, Unsloth when CUDA is present. |
| `chat` | Widget decoding guardrails: `temperature` (default 0.3), `repetitionPenalty` (default 1.15), `maxTokens` (256), `systemPrompt` (default: brief, honesty-first prompt). Per-entry `models[].chat` wins over it. Passed to `<site-chat>` / worker. |

## Model tiers & hardware detection

Fine-tuning and serving are different hardware problems: the machine that
trains a 4B model is not the phone that opens your site. With `models[]` the
manifest ships several tiers, and the browser picks the **highest tier that
fits the visitor's device** — no server, no per-visitor config.

What the browser can know (and what it can't):

| Signal | Source | Notes |
|---|---|---|
| WebGPU | `navigator.gpu.requestAdapter()` | required for WebLLM and for models >1B |
| Device memory | `navigator.deviceMemory` | **Chromium-only, capped at 8**; Safari/Firefox fall back to a conservative estimate from cores + WebGPU + mobile flag |
| Cores / mobile | `hardwareConcurrency`, `userAgentData.mobile` | |
| GPU info | adapter `info` + `limits.maxBufferSize` | weak proxy, only used as a tiebreaker |

Per-model requirements come from the lib's own allowlist knowledge
(`src/models.mjs`: params, VRAM footprint, tier, mobile/CPU viability) — so
any allowlisted base model is automatically rated, whether it was fine-tuned
here or served pretrained. An entry's explicit `requirements` always wins;
for unknown bases the artifact byte size is used with a safety overhead.

Behavior:

- **Fits** → the best tier loads (MLC/WebLLM over WebGPU, else GGUF via
  wllama, else ONNX via Transformers.js).
- **Nothing fits** → the chat shows a blocking "hardware is below its
  requirements" message with the detected specs and a **Try anyway** button
  (opt-in, may OOM the tab). The app never silently falls back to a
  different model.
- **Chosen tier fails to load** (e.g. GPU OOM) → the load-failure banner
  gains a **Load smaller model** button when a smaller tier exists.
- Events: `site-chat-hw-mismatch` / `onHardwareMismatch` (React/Vue/Svelte)
  carry `{ reasons, required, detected }`. `mountAssistant({ allowForce })`
  picks the smallest tier up front, `preferModel` pins an entry id.

Limits to be honest about: browsers do not expose true RAM, so detection is
heuristic and deliberately conservative (a false "fits" would crash the
tab). `deviceMemory` is capped at 8 GB and missing on Safari/Firefox. The
"Try anyway" path exists for exactly these edges.

## Small-model allowlist

Only models that fit consumer GPUs *and* have a browser path are accepted:

SmolLM2-135M/360M/1.7B, Qwen2.5-0.5B/1.5B, Qwen3-0.6B/1.7B/4B,
Qwen3.5-4B, Llama-3.2-1B/3B, TinyLlama-1.1B, Phi-3.5-mini,
Gemma-2-2B, Gemma-3-1B, Gemma-4-E4B.

`npx wasmtune models` lists HF ids, WebLLM ids, and VRAM guidance.
`--allow-large` bypasses the gate for 7–9B with an explicit warning.

## Commands

- `init` — write a starter `wasmtune.config.json`.
- `check` — validate config + model + platform/CUDA report, plus serving
  validation: manifest artifact URLs resolve, GGUF architecture is supported
  by the installed wllama runtime, all split shards present, single-file
  size warning. Add `--skip-serve` to skip.
- `models` — print the allowlist.
- `dataset` — crawl `dataDir` → `.finetune/dataset.{sft,dpo,grpo}.jsonl` + report.
  `--synth provider:model` adds LLM-generated QA pairs.
- `train` — bootstrap `.finetune-venv`, pip install, run SFT/DPO/GRPO. With
  `models[]`, runs once per `trained` entry (artifacts under `.finetune/<slug>/`;
  `trained: false` tiers are skipped).
- `eval` — holdout prompts → base vs tuned scores + regression gate (per trained tier).
- `convert` — merged LoRA → GGUF + MLC + ONNX, then writes one manifest with a
  `models[]` tier list (wraps `mlc_llm`, `llama.cpp`, `optimum`; all optional,
  loud-skip if missing). Pretrained tiers are published from their public
  browser ids without conversion.
- `serve` — static preview server for the chat widget + converted model.
- `build` — one-shot `dataset → train → eval → convert`. The eval gate
  throws on regression, so a bad model fails the build instead of shipping.

## Deploy to browsers (the intended setup)

```jsonc
// your-site/package.json
{
  "devDependencies": { "wasmtune": "^0.2.0" },
  "scripts": {
    "prebuild": "wasmtune build",
    "build": "vite build"
  }
}
```

```html
<script type="module">
  import { defineSiteChat } from 'wasmtune/chat';
  defineSiteChat({
    gguf: '/models/my-site-abc12345/my-site.Q4_K_M-00001-of-00009.gguf',
    temperature: 0.3, repetitionPenalty: 1.15,
    // Hybrid-reasoning models (Gemma 4, Qwen3.5) think by default; the
    // widget also strips thinking blocks at display as a backstop.
    templateKwargs: { enable_thinking: false },
  });
</script>
<site-chat></site-chat>
```

1. `wasmtune.config.json` points at your content folder (`dataDir`).
2. Every `npm run build` re-crawls, fine-tunes, eval-gates, and converts.
   First visitor download fetches the weights once (IndexedDB-cached);
   after that inference runs fully on-device (WebGPU via WebLLM, else
   Transformers.js ONNX, else wllama WASM) — no servers, no API keys.
3. The `<site-chat>` widget ships a runtime loop-breaker: if generation
   spirals, the reply is truncated at the loop onset with a fallback line
   instead of showing the loop.

## Training backends

- `python/requirements-cuda.txt`: `unsloth`, `trl~=0.13`, `peft~=0.14`, `datasets`, `bitsandbytes`, `accelerate`.
- `python/requirements-mlx.txt`: `mlx-lm`, `unsloth-mlx~=0.3.5`, `datasets`, `huggingface-hub`.
- Node never imports torch. `src/train/venv.mjs` creates `.finetune-venv` and spawns the right `python/train_*.py`.

## Browser chat

`src/chat/SiteChat.js` defines `<site-chat>` (Shadow DOM, no framework).
`src/chat/worker.js` holds inference off the main thread.
Order: WebLLM (WebGPU) → wllama GGUF → Transformers.js ONNX → optional cloud URL.
Weights cache in IndexedDB/OPFS; first load downloads once, then offline.
`src/chat/hardware.mjs` detects the device and picks the manifest tier that
fits; the worker re-checks at init (standalone `<site-chat>` usage included).
Artifact load failures surface as a blocking banner (attempted URL, size,
error; retry / smaller-tier / opt-in base buttons) — the widget never silently
swaps in a different model. Displayed replies are thinking-stripped and
loop-guarded. Manifest shape (v2):

```jsonc
{
  "version": "ab12cd34",
  "models": [{
    "id": "gemma-4-e4b", "label": "Gemma 4 E4B",
    "base": "google/gemma-4-E4B-it", "source": "tuned",
    "artifacts": {
      "gguf": { "url": "/models/gemma-4-e4b/m.Q4_K_M.ab12cd34.gguf", "sha256": "…", "bytes": 123 },
      "mlc": { "config": "/models/gemma-4-e4b/mlc/mlc-chat-config.json", "lib": "/models/gemma-4-e4b/mlc/lib.wasm" },
      "onnx": "/models/gemma-4-e4b/onnx"
    },
    "requirements": { "minDeviceMemoryGB": 6 },   // optional override
    "chat": { "templateKwargs": { "enable_thinking": false } }
  }],
  "artifacts": { /* legacy mirror of models[0] */ }
}
```

## Known limitations

- **Hardware detection is heuristic.** Browsers don't expose true RAM:
  `navigator.deviceMemory` is Chromium-only and capped at 8 GB, and
  Safari/Firefox fall back to an estimate from cores + WebGPU + mobile flag.
  Detection is conservative by design; "Try anyway" covers the edges and the
  load-failure banner offers the next-smaller tier.
- **DPO on MLX is experimental.** The pipeline (auto seeds → merge →
  `unsloth_mlx` DPOTrainer → adapters → eval loop mining) runs end-to-end,
  but `unsloth-mlx~=0.3.5`'s reference-free DPO loss sits at chance
  (~0.69, no movement even at 10x lr) — an upstream optimizer dynamics
  issue, not a data issue.
- **Prefer ORPO for shaping on MLX** (`method: "orpo"`, same triplet
  format, no reference model, ~half the RAM). It trains with real loss
  dynamics — but keep the default lr (`8e-6`): 6x higher collapsed a 0.6B
  test model into loops within 40 steps, while default-lr held steady
  (looped=0, small positive delta).
- SFT is the validated path for facts; use ORPO for shaping once SFT lands.
- Multi-tier training runs are sequential and each tier gets its own dataset
  build under `.finetune/<slug>/` — fine for 2-3 tiers, linear cost beyond.

## License

MIT.
