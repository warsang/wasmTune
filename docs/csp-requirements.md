# Content-Security-Policy requirements

The widget runs a module worker, and local inference compiles WebAssembly and
builds its runtime in a blob worker. Under a policy of `default-src 'self'` every
one of those requests is blocked, and the widget reports:

```
no local engine (install @mlc-ai/web-llm for WebGPU chat)
```

That message names a dependency, which is misleading: it appears when the engine
import is **blocked**, not when the package is missing. Check the console for a
CSP violation before installing anything — that has cost more debugging time than
the bug itself.

## Why the browser first

Two conditions have to hold before any policy question matters:

1. **WebGPU.** The engine needs `navigator.gpu` and a usable adapter. Without it,
   no policy change helps, and the same message appears.
2. **A policy that allows the engine.** The directives below.

## The directives, and why each one is needed

| Directive | Needed because |
| --- | --- |
| `script-src` with the engine origin (or `'self'` when you host it) | the worker imports the engine |
| `'wasm-unsafe-eval'` in `script-src` | WebAssembly compilation is governed by `script-src` |
| `worker-src blob:` | the runtime is built in a blob worker |
| `connect-src` with the weight hosts and `blob:` | model shards are fetched cross-origin and read into blobs |
| `font-src data:` if you embed fonts | icon fonts are inlined as `data:` URIs |

A workable example, with the engine loaded from a CDN:

```
Content-Security-Policy:
  default-src 'self';
  script-src 'self' https://esm.sh 'wasm-unsafe-eval' 'unsafe-inline';
  worker-src 'self' blob:;
  connect-src 'self' blob: https://esm.sh https://huggingface.co https://*.huggingface.co https://cdn-lfs.huggingface.co https://raw.githubusercontent.com;
  font-src 'self' data:;
```

`'unsafe-inline'` in `script-src` is needed only if your page has an inline
bootstrap script whose hash changes on every build. If you can emit a stable
hash, prefer that.

## Two traps that present as unrelated bugs

**A module worker does not inherit the page's import map.** A bare specifier that
the page resolves will not resolve inside the worker, even though the package is
installed and importable from the page. If you copy the worker tree into your
site, rewrite bare specifiers to absolute URLs as part of that copy.

**A `.mjs` file has no MIME type on some hosts.** Nginx maps it; some Caddy and
static-host configurations do not, and serve it as `application/octet-stream`.
The browser then refuses it as a module with a MIME error. Ship the worker tree
as `.js`, or add the mapping.

Both traps produce console errors that point at your code rather than at the
configuration, which is why they are worth knowing before you start.

## If you cannot widen the policy

Two alternatives:

- **Self-host the engine and the weights** on the same origin, and rewrite the
  specifiers accordingly. Everything is then same-origin and allowed without
  exceptions.
- **Bundle the worker with your own build.** The demo page does this, which is
  why it works on a host that sends no CSP at all.
