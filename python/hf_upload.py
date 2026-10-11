#!/usr/bin/env python3
"""Upload one file to the Hugging Face Hub, streaming.

wasmtune's publish used to do this from Node with
`PUT /api/models/{repo}/upload/main/{path}`. HF removed that endpoint, and its
replacement (`POST .../commit/main`) takes base64 in an NDJSON line - which Node
cannot produce for the graphs this publishes, because a base64 string of a
0.5 GB graph is ~0.7 GB and V8 caps strings at 0x1fffffe8 (~0.5 GB). So the
GraphQL-shaped option that survives at this size is the LFS flow, which means
batching, an upload href, and a finalising commit - three moving parts that
huggingface_hub already implements correctly.

This runs huggingface_hub from the venv instead. It is not a new dependency:
convert already requires it (it pulls in optimum), and this script is reached
only from `publish`, the one command that is allowed to touch the network.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--path-in-repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--commit-message", default="wasmtune: publish")
    ap.add_argument("--repo-type", default="model")
    a = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        print(f"hf upload skipped: huggingface_hub not installed ({e}); "
              f"pip install huggingface_hub", file=sys.stderr)
        return 3

    api = HfApi(token=a.token)
    info = api.upload_file(
        path_or_fileobj=a.file,
        repo_id=a.repo,
        path_in_repo=a.path_in_repo,
        repo_type=a.repo_type,
        revision=a.revision,
        commit_message=a.commit_message,
    )
    # The commit oid is the cache key. transformers.js caches a model by
    # "<repo>/<path>@<revision>"; without a revision it caches by path alone, so
    # replacing onnx/model_quantized.onnx in place leaves every returning visitor
    # holding the previous weights - a new upload is indistinguishable from no
    # upload at all. Publish puts this oid in the manifest so the URL changes.
    oid = getattr(info, "oid", None) or ""
    print(f"uploaded {a.path_in_repo} -> {a.repo}")
    print(f"OID {oid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
