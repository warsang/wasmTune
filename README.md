# WasmTune

Generic local fine-tuning + WebGPU chat for **any website folder**.

1. Point a config at the folder holding your site content.
2. `npx wasmtune dataset && npx wasmtune train` — LoRA/QLoRA locally.
   - NVIDIA Linux/Windows → **Unsloth + TRL + PEFT** (CUDA kernels, fastest).
   - Apple Silicon (M1–M5) → **MLX** via `mlx-lm` / `unsloth-mlx` (unified memory).
3. `npx wasmtune convert` — merged weights → GGUF + MLC (`q4f16_1`) + ONNX.
4. Embed `<site-chat>` — **WebLLM (WebGPU) primary**, Transformers.js and wllama fallback. Served locally, no server inference.

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

See `templates/wasmtune.config.example.json` and `finetune.config.schema.md` (below).

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

| Key | Meaning |
|---|---|
| `dataDir` | Folder crawled for FT data (md/mdx/txt/html/js/mjs/json). |
| `dataset.siteName` | Display name used in generated questions/prompts (default: folder name). |
| `dataset.summary` | One-line site summary for conversational seeds (default: inferred). |
| `model` | Base model HF id. Must be on the small-model allowlist (`npx wasmtune models`). |
| `method` | `sft` \| `dpo` \| `orpo` \| `grpo`. DPO/ORPO need triplets (`dpo.pairsFile` + auto seeds + eval-mined loops); GRPO needs `grpo.rewardFile`. ORPO needs no reference model — prefer it on memory-tight Macs. |
| `training.backend` | `auto` (recommended) \| `unsloth` \| `mlx`. `auto` picks MLX on darwin-arm64, Unsloth when CUDA is present. |
| `chat` | Widget decoding guardrails: `temperature` (default 0.3), `repetitionPenalty` (default 1.15), `maxTokens` (256), `systemPrompt` (default: brief, honesty-first prompt). Passed to `<site-chat>` / worker. |

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
- `train` — bootstrap `.finetune-venv`, pip install, run SFT/DPO/GRPO.
- `eval` — holdout prompts → base vs tuned scores + regression gate.
- `convert` — merge LoRA → GGUF + MLC + ONNX manifest (wraps `mlc_llm`, `llama.cpp`, `optimum`; all optional, loud-skip if missing).
- `serve` — static preview server for the chat widget + converted model.
- `build` — one-shot `dataset → train → eval → convert`. The eval gate
  throws on regression, so a bad model fails the build instead of shipping.

## Deploy to browsers (the intended setup)

```jsonc
// your-site/package.json
{
  "devDependencies": { "wasmtune": "^0.1.0" },
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
Artifact load failures surface as a blocking banner (attempted URL, size,
error; retry / opt-in base buttons) — the widget never silently swaps in a
different model. Displayed replies are thinking-stripped and loop-guarded.

## Known limitations

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

## License

MIT.
