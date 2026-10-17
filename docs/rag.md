# RAG: answering from a site's own pages

Fine-tuning teaches a model *how* to answer; retrieval decides *what it knows*.
For a website assistant that is the difference between "the copy changed and the
model is now wrong" and "re-index and it is right again". This module adds the
retrieval half, without leaving the browser.

```
wasmtune rag index        crawl dataDir → chunk → embed → rag-index.json
browser: fetch index, embed the query, cosine top-k, ground the prompt
```

No vector database, no reranker, no agentic retrieval: chunk, embed, cosine,
top-k. The same reasoning the rest of this package applies to models — a simple
baseline you can reason about beats a stack you cannot.

## Two ways to run it

**With fine-tuning.** Configure model tiers and a `rag` block. Retrieval grounds
the answer in current copy; the fine-tune shapes tone and format. They solve
different problems and compose.

**Without fine-tuning at all.** Tiers can be pretrained — no training run, no
LoRA, no GPU:

```json
{
  "dataDir": "./content",
  "rag": { "enabled": true, "embedder": "transformers", "topK": 4 },
  "models": [
    { "model": "Qwen/Qwen3-1.7B", "trained": false, "label": "desk" },
    { "model": "Qwen/Qwen3-0.6B", "trained": false, "label": "phone" },
    { "model": "HuggingFaceTB/SmolLM2-135M-Instruct", "trained": false, "label": "tiny" }
  ]
}
```

`wasmtune convert` then publishes a manifest the browser serves tier by tier,
and `wasmtune rag index` builds the corpus. The visitor's device picks the
biggest tier its hardware fits (see `src/chat/hardware.mjs`); nothing needs to
be trained to ship this.

## The embedder must match, and the index says which one it was

The index is built once (Node) and every query is embedded in the browser.
Vectors from different embedders are not comparable, so the index records the
embedder id and `createRag` **refuses** to run with another one rather than
returning confidently wrong results.

| spec | needs | notes |
| --- | --- | --- |
| `lexical` | nothing | hashing bag-of-words with IDF, deterministic, offline. Real retrieval, no second download. What the tests assert on. |
| `transformers:<id>` | `@huggingface/transformers` | sentence embeddings, mean-pooled and normalized. A multilingual model is the right default for a multilingual site. |

## Scopes are the security boundary

Every chunk carries a `scope`. Retrieval takes the scopes the caller may see and
filters **inside** the search, before ranking — a chunk from another scope can
never be returned however well it scores, and a caller cannot forget to filter
afterwards. Public site content uses `"public"`; session data uses one scope per
subject.

```js
const rag = createRag({ index, embedder, topK: 4 });

rag.setScope(["public"]);                    // anonymous visitor
rag.setScope(["public", "assessment:42"]);   // signed in, this subject only
rag.setScope([]);                            // signed out: retrieves nothing

rag.purgeScope("assessment:42");             // and the chunks are gone, not hidden
```

Purging is the second half of the boundary: on logout the session's chunks are
removed from the index, so a later question cannot retrieve them and a leak
needs two mistakes instead of one. `saveIndex` also refuses to persist anything
non-public to IndexedDB, so a session's findings cannot survive a page reload.

## Why the embedder is recorded and a mismatch is an error

An embedding is just a vector of numbers; the *positions* in it only mean
something relative to the model that produced it. Each embedder defines its own
axes, so two vectors are only comparable if they came from the same one.

This bites silently. The lexical embedder hashes tokens into 512 buckets; a
MiniLM sentence model has 384 learned dimensions. Their dot product is still a
number between 0 and 1, so nothing throws — it simply isn't similarity any more.
What you get is an assistant that retrieves the wrong passages and cites them
confidently, which reads as "the model is bad" rather than "the maths was
meaningless".

It matters here more than in a normal app because the two halves run in
different places at different times: the index is embedded once, in Node, at
`wasmtune rag index`; the query is embedded later, in the visitor's browser.
That is the easiest possible setup for the embedder to drift apart — a default
changed in the config, a browser build pinned to an older package, a model id
edited in one place only.

So:

- the index stores the embedder id that produced it (`embedder: "transformers:…"`);
- `loadIndex` builds the query embedder **from that id**, so the browser cannot
  accidentally choose a different one;
- `createRag` throws if the embedder it is given is not the one in the index.

Changing the embedder is a re-index, not a runtime setting. If you would rather
have a warning than a hard failure, that is a one-line change — but the failure
it prevents is a confident wrong answer, so the default is to refuse.

## Grounding, for models this small

The passages go into the **user** message, not a second system message: the chat
templates for the sizes this package serves commonly accept exactly one leading
system turn, and a mid-conversation one is dropped or rejected — with the
symptom being a model that answers as if retrieval never happened. Passages are
numbered and the instruction is to cite the number, which small models are much
better at than paraphrase-with-attribution.

## Actions: the model asks, the host decides

A 0.6–4B model in a browser cannot be trusted to perform side effects, so it
does not. If the instruction is enabled, the model may emit:

```
<tool>submit_contact_form</tool>{"name":"…","email":"…","message":"…"}
```

`parseToolCall` validates it — allowed name, real-looking email, minimum length,
unknown keys dropped — and hands it back. **Nothing is sent by the parser or the
widget.** The host shows it, confirms with the visitor, and posts it. That is
the difference between "the assistant drafted a message" and "a small model
emailed a stranger because the prompt drifted".

## Wiring it into the widget

```js
const { element } = await mountAssistant({
  target: "#chat",
  rag: { indexUrl: "/models/rag-index.json" },        // or a context you built
  tools: createToolRegistry({
    submit_contact_form: {
      run: (args) => post("/api/contact", args),      // your page's own code
      validate: (args) => args.email.includes("@") || "email is missing",
      description: "send the contact form",
    },
  }),
});
```

What that gives you:

- **retrieve before generate.** The request carries a grounded user turn (numbered
  passages, then the question); the visible log and the conversation history keep
  what the visitor actually typed, so a second question is not answered against
  the first question's context.
- **tools that run.** The model replies with `<tool>name</tool>{…}`, the registry
  validates and executes it *in the page*, the result is rendered as a line in the
  log, and a `site-chat-tool` event fires with `{ name, args, result }`.
- **scopes for session data.** `element.setRagScope(["public", "assessment:42"])`
  and `element.purgeRag("assessment:42")` — the signed-out state is `setRagScope([])`.
- **citations.** A sources line under the answer, listing the pages the passages
  came from.
- **an index that is not re-fetched.** Cached in IndexedDB per index URL and
  embedder. Re-publishing an index at the same URL needs a cache-busting query or
  `clearIndexes()`.

Retrieval failing is not the model failing: if the index cannot be read, the
widget says so on the event stream and answers from the model alone.

## What is not here yet

The demo and the site-side examples (a contact tool wired to a real endpoint, a
session-scoped corpus added and purged on login/logout), and the four-way
evaluation: base, base+RAG, tuned, tuned+RAG against one benchmark, split by
document rather than by generated question.

