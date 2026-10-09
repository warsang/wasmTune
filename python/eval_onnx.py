#!/usr/bin/env python3
"""Greedy generation through the ONNX graph the browser will load.

Tokenization happens outside this script, in scripts/onnx-eval.mjs, using the
same @huggingface/transformers the browser uses. That is deliberate: a second
tokenizer implementation would score differently from the one under test, and
the local transformers/tokenizers install cannot read a Qwen2 tokenizer.json
anyway. This script therefore does arithmetic only — ids in, ids out.
"""
import argparse
import json
import os
import sys

# The dtype names transformers.js asks for do not map 1:1 to filenames.
DTYPE_FILE = {
    "q4": "model_q4.onnx", "q4f16": "model_q4.onnx", "q8": "model_quantized.onnx",
    "fp32": "model.onnx", "fp16": "model_fp16.onnx",
    "int8": "model_int8.onnx", "uint8": "model_uint8.onnx",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--input-tokens", required=True,
                    help="JSON [{id, input_ids}] written by scripts/onnx-eval.mjs")
    ap.add_argument("--out", required=True, help="JSON [{id, output_ids}]")
    ap.add_argument("--dtype", default="q8")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"eval-onnx skipped: onnxruntime not installed ({e}); pip install "
              f"'optimum[onnxruntime]'", file=sys.stderr)
        return 0

    fname = DTYPE_FILE.get(a.dtype)
    if not fname:
        print(f"eval-onnx skipped: unknown dtype {a.dtype}", file=sys.stderr)
        return 0
    graph = os.path.join(a.onnx_dir, fname)
    if not os.path.exists(graph):
        print(f"eval-onnx skipped: {graph} not present (run `wasmtune convert`)",
              file=sys.stderr)
        return 0

    # A graph whose external data is missing is worse than no graph at all: the
    # first publish shipped a 4 MB stub with 487 MB of weights left behind.
    with open(graph, "rb") as f:
        head = f.read(8_000_000).decode("latin1", "ignore")
    missing = [
        ref for ref in set(__import__("re").findall(r"[\w.\-/]{1,120}\.onnx_data\b", head))
        if not os.path.exists(os.path.join(a.onnx_dir, ref))
    ]
    if missing:
        print(f"eval-onnx skipped: {graph} references missing external data: "
              f"{', '.join(missing)}", file=sys.stderr)
        return 0

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(graph, so, providers=["CPUExecutionProvider"])
    import numpy as np

    inputs = {i.name: i for i in sess.get_inputs()}

    def dtype_of(meta):
        return {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                "tensor(int64)": np.int64, "tensor(int32)": np.int32}.get(str(meta.type), np.float32)

    def shape_of(meta):
        return [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]

    cases = json.load(open(a.input_tokens, encoding="utf-8"))
    print(f"eval-onnx: {len(cases)} prompts through {fname} "
          f"({os.path.getsize(graph)/1e6:.1f} MB)", file=sys.stderr)

    eos = int(os.environ.get("EVAL_ONNX_EOS", "-1"))
    out = []
    for case in cases:
        ids = list(case["input_ids"])
        generated = []
        for _ in range(a.max_tokens):
            attention = [1] * len(ids)
            feed = {}
            for name, meta in inputs.items():
                # ONNX dims are symbolic ('sequence_length'), so substitute the
                # real ones rather than feeding 1x1 for a 71-token prompt.
                if name == "input_ids":
                    feed[name] = ort.OrtValue.ortvalue_from_numpy(
                        np.array(ids, dtype=dtype_of(meta)).reshape([1, len(ids)]))
                elif name == "attention_mask":
                    feed[name] = ort.OrtValue.ortvalue_from_numpy(
                        np.array(attention, dtype=dtype_of(meta)).reshape([1, len(attention)]))
                elif name == "position_ids":
                    feed[name] = ort.OrtValue.ortvalue_from_numpy(
                        np.array(list(range(len(ids))), dtype=dtype_of(meta)).reshape([1, len(ids)]))
                else:
                    dims = shape_of(meta)
                    if name.startswith("past_key_values"):
                        dims[2] = 0                      # empty past
                    feed[name] = ort.OrtValue.ortvalue_from_numpy(np.zeros(dims, dtype=dtype_of(meta)))
            logits = sess.run(["logits"], feed)[0]
            next_id = int(np.argmax(logits[0, -1]))
            if eos >= 0 and next_id == eos:
                break
            ids.append(next_id)
            generated.append(next_id)
        out.append({ "id": case["id"], "output_ids": generated })
        print(f"  [{case['id']}] {len(generated)} tokens", file=sys.stderr)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"eval-onnx: wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
