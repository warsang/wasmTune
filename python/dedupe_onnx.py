#!/usr/bin/env python3
"""Share byte-identical initializers in an ONNX graph.

torch.onnx with `--task text-generation-with-past` writes a RoPE sin/cos cache
into every decoder layer. Those tables depend only on position and head dim, so
a 24-layer export carries 48 byte-identical tensors: for Qwen2.5-0.5B that is
24 x [32768, 64] fp32 x 2 = 403 MB of the 901 MB file, duplicated for no reason.

Merging them changes no arithmetic — verified bit-identical logits against the
pre-dedupe graph — and cut that export from 901 MB to 511 MB. The saving scales
with layer count and max_position_embeddings, so it is larger on bigger models.

Invoked automatically by `wasmtune convert` after the ONNX export. Pure stdlib +
onnx, so it is a loud-skip when onnx is not installed.
"""
import argparse
import hashlib
import os
import sys


def tensor_key(init, numpy_helper):
    h = hashlib.sha256()
    h.update(str(init.data_type).encode())
    h.update(str(list(init.dims)).encode())
    if init.raw_data:
        h.update(init.raw_data)
    else:
        try:
            h.update(numpy_helper.to_array(init).tobytes())
        except Exception:
            # Unreadable tensor: key it by name so it can never be merged.
            h.update(init.name.encode())
    return h.hexdigest()


def dedupe(src, dst, data_name=None):
    import onnx
    from onnx import numpy_helper

    model = onnx.load(src, load_external_data=True)
    before = len(model.graph.initializer)

    canonical = {}
    rename = {}
    for init in model.graph.initializer:
        k = tensor_key(init, numpy_helper)
        if k in canonical:
            rename[init.name] = canonical[k]
        else:
            canonical[k] = init.name

    if not rename:
        return {"merged": 0, "before": before, "after": before, "bytes": os.path.getsize(src)}

    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]

    keep = [i for i in model.graph.initializer if i.name not in rename]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

    base = os.path.splitext(dst)[0]
    # The data filename is caller-controlled so the result can be renamed into
    # place afterwards without invalidating the graph's own reference to it.
    location = data_name or (os.path.basename(base) + ".onnx_data")
    onnx.save(
        model, dst,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=1024,
    )
    data = os.path.join(os.path.dirname(dst), location)
    return {
        "merged": len(rename),
        "before": before,
        "after": len(model.graph.initializer),
        "bytes": os.path.getsize(data) if os.path.exists(data) else os.path.getsize(dst),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ONNX graph to deduplicate")
    ap.add_argument("--out", required=True, help="where to write the result")
    ap.add_argument("--data-name", default=None,
                    help="external-data filename to embed (default: <out>.onnx_data)")
    a = ap.parse_args()
    try:
        import onnx  # noqa: F401
    except Exception as e:
        print(f"dedupe skipped: onnx not installed ({e})", file=sys.stderr)
        return 0
    r = dedupe(a.model, a.out, a.data_name)
    print(
        f"dedupe: merged {r['merged']} identical tensor(s), "
        f"{r['before']} -> {r['after']} initializers, {r['bytes']/1e6:.1f} MB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
