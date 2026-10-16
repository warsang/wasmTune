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

`parseAction` validates it — allowed name, real-looking email, minimum length,
unknown keys dropped — and hands it back. **Nothing is sent by the parser or the
widget.** The host shows it, confirms with the visitor, and posts it. That is
the difference between "the assistant drafted a message" and "a small model
emailed a stranger because the prompt drifted".

## What is not here yet

The retrieval engine, the index builder, the CLI and the action protocol are
implemented and tested. Wiring them into `<site-chat>` — retrieve-then-generate
on submit, a `setRagScope`/`purgeRag` API for the host, a `site-chat-action`
event for the confirmation UI — is the next commit on this branch, followed by
the evaluation work (base vs base+RAG vs tuned vs tuned+RAG on the same benchmark).
