# Recipe: adding chat to a static site

This is the end-to-end integration. The library is generic: none of these steps
names a host, a model, or a page. Replace the paths with your own.

## 1. Install

```sh
npm i wasmtune
# peer deps, only if you want the corresponding fallback:
npm i -D @mlc-ai/web-llm   # WebGPU primary (also needed to build the bundled worker)
npm i -D @wllama/wllama     # GGUF fallback (optional)
```

## 2. Mount the widget

```js
import { mountAssistant } from "wasmtune/chat";

const { element, ready } = await mountAssistant({
  target: "#chat",
  manifestUrl: "/models/model-manifest.json",
  // workerUrl defaults to the bundle that ships in the package
  // (src/chat/bundled/worker.js). Override it if you copy or rebuild it elsewhere:
  // workerUrl: "/wasmtune/bundled/worker.js",
  rag: { indexUrl: "/models/rag-index.json" },
  tools: myTools, // optional: page-registered JS functions the model can call
});
await ready; // resolves with the engine name once the worker is up
```

Tools are plain objects with `{ name, description, run(args, ctx) }`. The model
emits `<tool>name</tool>{...json...}` and the page runs it — that is the whole
point of having the model in the browser. Nothing is handed to a server.

## 3. Self-host the worker (the "works on any static host" path)

`mountAssistant` works out of the box against the worker bundle that ships in
the package. If you want to host it yourself, or pin a different engine version:

```sh
wasmtune bundle                      # copies src/chat/bundled/ -> ./wasmtune-worker/
wasmtune bundle --out public/chat    # choose the destination directory
wasmtune bundle --from-source        # rebuild from src/chat/worker.js (needs vite)
```

Then point `workerUrl` at `<out>/worker.js`. The directory must be served as
static files: `worker.js` dynamically imports its sibling `worker-*.js` chunks,
and those resolve relative to `worker.js`'s own URL.

## 4. Build the retrieval index

```sh
wasmtune rag index                 # writes public/models/rag-index.json (lexical-512)
wasmtune rag index --embed transformers:Xenova/paraphrase-multilingual-MiniLM-L12-v2
```

The index records its embedder id; `createRag` throws if the query-time embedder
does not match. A lexical hash and a learned embedder both produce numbers in
[0, 1], so a mismatch would silently stop being similarity.

## 5. Serve a model manifest

`/models/model-manifest.json` lists tiers with hardware requirements. A minimal
entry:

```json
{
  "models": [{
    "id": "smollm2-360m",
    "label": "SmolLM2 360M",
    "artifacts": { "webllm": "Qwen3-0.6B-q4f16_1-MLC" },
    "requirements": { "minDeviceMemoryGB": 2, "needsWebGPU": true }
  }]
}
```

`mountAssistant` detects the device and picks the highest tier that fits; a
device that fits none gets a blocking banner instead of a silent swap.

## 6. Content-Security-Policy

The widget runs a module worker and compiles WebAssembly, so a strict policy
blocks it. See `docs/csp-requirements.md` for the full directive list and the
two traps (a module worker does not inherit the page's import map; `.mjs` is
served as `application/octet-stream` where the host has no mapping) that present
as unrelated bugs. Symptom of a blocked request: `no local engine`, even with
the package installed.

## 7. Tools (optional)

```js
const tools = {
  registry: [
    { name: "submit_contact_form", description: "POST the contact form",
      parameters: { type: "object", properties: { name: { type: "string" }, email: { type: "string" }, message: { type: "string" } }, required: ["name","email","message"] } },
  ],
  run: async (name, args, ctx) => {
    if (name === "submit_contact_form") {
      const r = await fetch("/api/contact", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(args) });
      return { ok: r.ok };
    }
    return { ok: false, error: "unknown tool" };
  },
  instruction: () => "You may submit the contact form with submit_contact_form when the visitor asks to be contacted.",
};
```

Only registered names run. A throwing tool returns a failed call, never a crash.
Tool results are not auto-fed back into the model (no agent loop).

## 8. Access control (optional)

Scopes are the security boundary and are enforced inside search, before ranking.

```js
// signed in: load the user's documents under a per-user scope
await assistant.setRagScope(["public", `user:${userId}`]);
// on sign-out: drop everything that is not public
assistant.purgeRag(`user:${userId}`);   // deletes, does not hide
assistant.setRagScope(["public"]);
```

A chunk that loses its scope during indexing is refused rather than published.
'