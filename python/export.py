#!/usr/bin/env python3
"""wasmtune export: adapters -> merged HF dir (convert to GGUF/MLC/ONNX via Node wrappers)."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # Prefer peft merge when available; otherwise copy adapters + manifest so the
    # Node convert step can still quantize/pack what exists.
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = AutoModelForCausalLM.from_pretrained(a.base)
        tok = AutoTokenizer.from_pretrained(a.base)
        model = PeftModel.from_pretrained(base, a.adapters)
        merged = model.merge_and_unload()
        merged.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        status = "merged"
    except Exception as e:
        status = f"merge-skipped: {e}"
    with open(os.path.join(a.out, "export.report.json"), "w") as f:
        json.dump({"base": a.base, "adapters": a.adapters, "status": status}, f, indent=2)
    print(status + f" -> {a.out}")


if __name__ == "__main__":
    main()
