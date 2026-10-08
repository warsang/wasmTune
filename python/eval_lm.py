#!/usr/bin/env python3
"""wasmtune eval generation (backend-agnostic).

Reads prompts JSONL [{id, prompt}], writes [{id, output}] JSON.
  --backend mlx   : mlx_lm (adapter_path optional, Apple Silicon)
  --backend cuda  : transformers (+ peft when --adapters given)

Usage:
  python eval_lm.py --backend mlx --model <hf-id-or-mlx-id> [--adapters DIR] \
      --prompts eval.prompts.jsonl --out eval.out.json [--max-tokens 128]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx", choices=["mlx", "cuda"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    # Same system prompt the widget sends. Without it the base injects its own
    # default (SmolLM2's "You are a helpful AI assistant named SmolLM..."),
    # so the gate would score the tuned model under a prompt it was never
    # trained on and never served with.
    ap.add_argument("--system-prompt", default=None)
    a = ap.parse_args()

    prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]
    if a.backend == "mlx":
        rows = run_mlx(a, prompts)
    else:
        rows = run_cuda(a, prompts)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"eval: {len(rows)} completions -> {a.out}", file=sys.stderr)


def apply_template(tokenizer, prompt, system_prompt=None):
    try:
        if getattr(tokenizer, "chat_template", None):
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
                + [{"role": "user", "content": prompt}]
            # tokenize=False is load-bearing. apply_chat_template defaults to
            # tokenize=True and therefore returns input_ids, not text; feeding
            # those straight back into tok() raises "text input must be of type
            # str (single example), list[str] ...". That broke `wasmtune eval`
            # for every model that ships a chat template (i.e. every instruct
            # model), and eval is the regression gate `wasmtune build` depends
            # on.
            #
            # Disable hybrid thinking where supported (Qwen3.5, Gemma4):
            # eval measures answers, not reasoning traces.
            try:
                return tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                # Older/newer templates may not accept enable_thinking.
                return tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        pass
    return prompt


def run_mlx(a, prompts):
    from mlx_lm import load, generate
    kwargs = {"adapter_path": a.adapters} if a.adapters else {}
    print(f"loading {a.model} {kwargs} ...", file=sys.stderr, flush=True)
    model, tokenizer = load(a.model, **kwargs)
    rows = []
    for p in prompts:
        out = generate(model, tokenizer, prompt=apply_template(tokenizer, p["prompt"], a.system_prompt),
                       max_tokens=a.max_tokens, verbose=False)
        rows.append({"id": p["id"], "output": out})
        print(f"[{p['id']}] {len(out)} chars", file=sys.stderr, flush=True)
    return rows


def run_cuda(a, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model} ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", trust_remote_code=False)
    if a.adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapters)
        model = model.merge_and_unload()
    model.eval()
    rows = []
    for p in prompts:
        text = apply_template(tok, p["prompt"], a.system_prompt)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=a.max_tokens,
                                 do_sample=a.temperature > 0,
                                 temperature=max(a.temperature, 1e-6),
                                 pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"id": p["id"], "output": out})
        print(f"[{p['id']}] {len(out)} chars", file=sys.stderr, flush=True)
    return rows


if __name__ == "__main__":
    main()
