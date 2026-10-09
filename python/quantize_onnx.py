#!/usr/bin/env python3
"""Quantize an ONNX export into the dtypes transformers.js can request.

transformers.js picks onnx/model_<dtype>.onnx by dtype string, and the names are
not a key===value tie: dtype "q8" is served as model_quantized.onnx. Exporting
fp32 only (which is what `optimum-cli export onnx` writes) and then pointing the
manifest at a q8 dtype 404s the graph, and the widget reports engine
"unavailable" with nothing pointing at a filename.

Quantizing here closes the loop in the workflow instead of leaving it to the
host: whatever this script writes, the manifest can name.

Invoked by `wasmtune convert` after the optimum export. Loud-skips when onnx or
onnxruntime is missing, like every other optional converter.
"""
import argparse
import os
import sys

# Defaults mirror resolveDtype(): WebGPU -> q4f16, WASM -> q4. ORT's dynamic
# quantizer has no q4f16 mode (that combination needs QDQ / static
# quantization), so q4f16 is accepted as an alias for q4 and the manifest says
# which one actually exists.
DTYPE_ALIAS = {"q4f16": "q4", "q4": "q4", "q8": "q8", "int8": "q8", "uint8": "q8"}
# transformers.js' own filenames, by the dtype it asks for.
DTYPE_FILE = {"q4": "model_q4.onnx", "q8": "model_quantized.onnx"}


def quantize(onnx_dir, dtypes):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src = os.path.join(onnx_dir, "model.onnx")
    if not os.path.exists(src):
        return {"quantized": [], "skipped": ["no onnx/model.onnx to quantize"]}

    seen, made = set(), []
    for raw in dtypes:
        dtype = DTYPE_ALIAS.get(str(raw).strip().lower())
        if dtype is None or dtype in seen:
            continue
        seen.add(dtype)
        target = os.path.join(onnx_dir, DTYPE_FILE[dtype])
        try:
            quantize_dynamic(
                src, target, weight_type=QuantType.QInt4 if dtype == "q4" else QuantType.QUInt8,
                per_channel=(dtype == "q4"), reduce_range=(dtype == "q4"),
                extra_options={"MatMulConstBOnly": True},
            )
            data = os.path.splitext(target)[0] + ".onnx_data"
            size = os.path.getsize(data) if os.path.exists(data) else os.path.getsize(target)
            made.append({"dtype": dtype, "file": DTYPE_FILE[dtype], "bytes": size})
        except Exception as e:
            print(f"  {dtype} failed: {type(e).__name__}: {e}", file=sys.stderr)

    return {"quantized": made, "skipped": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--dtypes", default="q4f16 q4 q8",
                    help="space-separated subset of {q4,q4f16,q8,int8,uint8}")
    a = ap.parse_args()
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        print(f"quantize skipped: onnxruntime not installed ({e}); pip install "
              f"'optimum[onnxruntime]'", file=sys.stderr)
        return 0

    r = quantize(a.onnx_dir, a.dtypes.split())
    for m in r["quantized"]:
        print(f"  quantized {m['dtype']}: {m['bytes']/1e6:.1f} MB -> {m['file']}",
              file=sys.stderr)
    for s in r["skipped"]:
        print(f"  note: {s}", file=sys.stderr)
    if not r["quantized"] and not r["skipped"]:
        print("  nothing quantized", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
