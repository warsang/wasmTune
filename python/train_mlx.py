#!/usr/bin/env python3
"""wasmtune MLX trainer (Apple Silicon M1-M5): SFT + DPO + GRPO.

Uses the Unsloth-compatible `unsloth_mlx` API when available, falling back to
plain `mlx_lm.lora` for SFT. Datasets are the same JSONL files Node wrote.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args-json", required=True)
    a = ap.parse_args()
    with open(a.args_json) as f:
        cfg = json.load(f)

    if sys.platform != "darwin":
        print("WARNING: backend=mlx outside macOS — MLX needs Apple Silicon.", file=sys.stderr)
    try:
        import mlx  # noqa: F401
    except Exception as e:
        print(f"ERROR: MLX import failed: {e}", file=sys.stderr)
        print("Run inside the auto-created .finetune-venv (pip install -r python/requirements-mlx.txt).", file=sys.stderr)
        sys.exit(2)

    model_id = cfg["model"]
    method = cfg.get("method", "sft")
    out = cfg["outDir"]
    os.makedirs(out, exist_ok=True)

    ds_path = {"sft": cfg["sftFile"], "dpo": cfg["dpoFile"], "orpo": cfg["dpoFile"], "grpo": cfg["grpoFile"]}[method]
    with open(ds_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    if cfg.get("dryRun"):
        rows = rows[:8]
        print(f"[dry-run] {method}: {len(rows)} rows on MLX ({model_id}), no optimizer steps", file=sys.stderr)
        with open(os.path.join(out, "dry_run.json"), "w") as f:
            json.dump({"method": method, "rows": len(rows), "model": model_id, "backend": "mlx"}, f, indent=2)
        return

    # Prefer the Unsloth-compatible surface so scripts stay portable,
    # unless config asks for plain mlx-lm (mlxImpl: "mlx-lm").
    force_mlx_lm = str(cfg.get("mlxImpl", "auto")).lower() == "mlx-lm"
    try:
        if force_mlx_lm:
            raise ImportError("mlxImpl=mlx-lm requested")
        from unsloth_mlx import FastLanguageModel, SFTTrainer  # type: ignore
        use_unsloth_mlx = True
    except Exception:
        use_unsloth_mlx = False

    if method == "sft" and not use_unsloth_mlx:
        run_mlx_lm_lora(cfg, rows, out)
        return

    try:
        # unsloth_mlx path mirrors the CUDA script's structure.
        # Forward the quantization flag: without load_in_4bit, unsloth_mlx
        # dequantizes a 4-bit base to fp16 (~4x memory) and OOMs small boxes.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=int(cfg.get("maxSeqLen", 2048)),
            load_in_4bit=bool(cfg.get("quant4", True)))
        lora = cfg.get("lora", {})
        model = FastLanguageModel.get_peft_model(
            model, r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 16)),
            lora_dropout=float(lora.get("dropout", 0.05)))

        if method == "sft":
            from unsloth_mlx import SFTTrainer as Trainer  # type: ignore
            trainer = Trainer(model=model, tokenizer=tokenizer, train_dataset=rows,
                              max_steps=int(cfg.get("maxSteps", -1) or -1),
                              learning_rate=float(cfg.get("lr", 2e-4)), seed=int(cfg.get("seed", 42)),
                              output_dir=out)
        elif method == "dpo":
            # NB: unsloth_mlx trainers read config ONLY from the Config
            # object — stray kwargs are silently ignored. Build it explicitly.
            from unsloth_mlx import DPOTrainer as Trainer  # type: ignore
            from unsloth_mlx.rl_trainers import DPOConfig  # type: ignore
            args = DPOConfig(
                beta=float(cfg.get("beta", 0.1)),
                output_dir=out,
                learning_rate=float(cfg.get("lr", 5e-6)),
                per_device_train_batch_size=int(cfg.get("batchSize", 2)),
                gradient_accumulation_steps=int(cfg.get("gradAccum", 4)),
                num_train_epochs=int(cfg.get("epochs", 1)),
                max_steps=int(cfg.get("maxSteps", -1) or -1),
                max_seq_length=int(cfg.get("maxSeqLen", 2048)),
                seed=int(cfg.get("seed", 42)),
            )
            trainer = Trainer(model=model, tokenizer=tokenizer, train_dataset=rows, args=args)
        elif method == "orpo":
            # ORPO needs no reference model (single-model odds-ratio loss),
            # so it halves DPO's memory AND sidesteps unsloth_mlx's broken
            # reference-free DPO dynamics. Same triplet format as DPO.
            from unsloth_mlx import ORPOTrainer as Trainer  # type: ignore
            from unsloth_mlx.rl_trainers import ORPOConfig  # type: ignore
            args = ORPOConfig(
                beta=float(cfg.get("beta", 0.1)),
                output_dir=out,
                learning_rate=float(cfg.get("lr", 8e-6)),
                per_device_train_batch_size=int(cfg.get("batchSize", 2)),
                gradient_accumulation_steps=int(cfg.get("gradAccum", 4)),
                num_train_epochs=int(cfg.get("epochs", 1)),
                max_steps=int(cfg.get("maxSteps", -1) or -1),
                max_seq_length=int(cfg.get("maxSeqLen", 2048)),
                seed=int(cfg.get("seed", 42)),
            )
            trainer = Trainer(model=model, tokenizer=tokenizer, train_dataset=rows, args=args)
        else:
            from unsloth_mlx import GRPOTrainer as Trainer  # type: ignore
            from unsloth_mlx.rl_trainers import GRPOConfig  # type: ignore
            args = GRPOConfig(
                output_dir=out,
                learning_rate=float(cfg.get("lr", 5e-6)),
                per_device_train_batch_size=int(cfg.get("batchSize", 2)),
                num_train_epochs=int(cfg.get("epochs", 1)),
                max_steps=int(cfg.get("maxSteps", -1) or -1),
                num_generations=int(cfg.get("numGenerations", 4)),
                max_seq_length=int(cfg.get("maxSeqLen", 2048)),
                seed=int(cfg.get("seed", 42)),
            )
            trainer = Trainer(model=model, tokenizer=tokenizer, train_dataset=rows, args=args)
        trainer.train()
        model.save_pretrained(os.path.join(out, "adapters"))
        with open(os.path.join(out, "train.report.json"), "w") as f:
            json.dump({"method": method, "model": model_id, "backend": "unsloth-mlx"}, f, indent=2)
        print(f"done -> {out}")
    except Exception as e:
        print(f"WARNING: unsloth_mlx path failed ({e}); falling back to mlx_lm.lora", file=sys.stderr)
        if method != "sft":
            raise
        run_mlx_lm_lora(cfg, rows, out)


def run_mlx_lm_lora(cfg, rows, out):
    """Plain mlx-lm LoRA path: ShareGPT rows -> train/valid.jsonl, then run
    `mlx_lm.lora` as a subprocess (real training)."""
    import subprocess
    model_id = cfg["model"]
    data_dir = os.path.join(out, "mlx_data")
    os.makedirs(data_dir, exist_ok=True)
    texts = []
    for r in rows:
        msgs = r.get("messages", [])
        parts = []
        for m in msgs:
            role = "user" if m.get("role") == "user" else "assistant"
            parts.append(f"<|{role}|>\n{m.get('content','')}")
        texts.append({"text": "\n".join(parts)})
    n_valid = max(1, len(texts) // 10)
    with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
        for t in texts[:-n_valid] or texts:
            f.write(json.dumps(t) + "\n")
    with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
        for t in texts[-n_valid:]:
            f.write(json.dumps(t) + "\n")
    max_steps = int(cfg.get("maxSteps", 0) or 0)
    iters = max_steps if max_steps > 0 else int(cfg.get("epochs", 2)) * max(1, len(texts) // int(cfg.get("batchSize", 2)))
    cmd = [sys.executable, "-m", "mlx_lm", "lora",
           "--model", model_id, "--train", "--data", data_dir,
           "--iters", str(iters),
           "--batch-size", str(cfg.get("batchSize", 2)),
           "--learning-rate", str(cfg.get("lr", 2e-4)),
           "--max-seq-length", str(cfg.get("maxSeqLen", 2048)),
           "--grad-accumulation-steps", str(cfg.get("gradAccum", 1)),
           "--num-layers", str(cfg.get("numLayers", 16)),
           "--adapter-path", os.path.join(out, "adapters")]
    # Gradient checkpointing trades compute for memory (no math change).
    if cfg.get("gradCheckpoint"):
        cmd.append("--grad-checkpoint")
    print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True)
    with open(os.path.join(out, "train.report.json"), "w") as f:
        json.dump({"method": "sft", "model": model_id, "backend": "mlx-lm",
                   "iters": iters, "data": data_dir}, f, indent=2)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
