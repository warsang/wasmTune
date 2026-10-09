# Evaluation of the artifact that ships

`wasmtune eval` scores the merged fp16 weights. The widget loads a quantized
ONNX graph. Those are different models, and until `wasmtune eval-onnx` existed
nothing measured the difference.

## What the command does

```bash
wasmtune convert   # exports, quantizes, dedupes
wasmtune eval      # scores the fp16 weights eval always scored
wasmtune eval-onnx # scores onnx/model_<dtype>.onnnx — what actually ships
```

The split is deliberate: tokenizing and scoring happen in Node with
`@huggingface/transformers` and `src/eval/score.mjs` — the same code the browser
and `wasmtune eval` use — while only the arithmetic runs in Python's
onnxruntime. A second tokenizer or a second scorer would produce numbers that
look comparable and are not, which is worse than no comparison at all.

Source: `scripts/onnx-eval.mjs` (orchestration, tokenize, score) and
`python/eval_onnx.py` (generation through the graph).

## Measured, 39 holdout prompts, same system prompt, q8 vs fp16

| metric | fp16 | onnx q8 | cost |
|---|---|---|---|
| avg | 0.158 | 0.111 | **−30%** |
| generalization | 0.231 | 0.098 | **−58%** |
| memorization | 0.129 | 0.116 | −10% |
| faithfulness | 0.326 | 0.327 | — |
| loops introduced | 0 | 2 | +2 |
| converse pass rate | 0.6 | 0.4 | −0.2 |

## What that means, and what it does not

Generalization — answering a prompt shaped like but not from the training set —
is where the damage is. Memorization mostly survives, so a quantized model
still reproduces what it saw and can no longer extend it. That is the failure
mode to watch for.

`--force` publishes anyway; `WASMTUNE_QUANT_DROP` moves the threshold from its
default of 0.1.

Honest limitations of the numbers above:

- **Keyword recall is not correctness.** "The maximum batch size is 1" scores
  on `batch`, `size` and a digit; `1` is not `1000`. `faithfulness` and
  `untraced` are the anti-gaming counters — the fraction of the output's
  distinctive terms that no reference used — and they are reported, not summed
  into the score.
- **The score moves with the token budget.** The same graph scored 0.111 at 96
  new tokens and 0.079 at 48. Compare evals at matched `--max-tokens` or you
  are measuring the budget, not the model.
- **Prompts come from the docs' own headings.** A faithful paraphraser and a
  memorizer score identically.
- **It scores what the model says, not what a user wanted.**

## The bug this replaced

The fine-tune was originally evaluated against fp16 CUDA weights. The browser
said "In RAM on first run" to *"Where are the keys stored?"* while the fp16 eval
said the correct `~/.config/lumen/credentials.json`. Training was not
regressed; quantization was. Nothing caught it because every gate stopped at the
weights.

This is also why `wasmtune eval` and `wasmtune eval-onnx` must share a scorer: a
python reimplementation would have shown the same regression as a *smaller*
number and the gap would still have been invisible.
