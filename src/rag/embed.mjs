// wasmtune — embeddings for retrieval, client-side.
//
// Two embedders, and the choice matters because the *same* one has to run on
// both sides of the wire: the index is built once (Node, at `wasmtune rag
// index`) and every query is embedded in the visitor's browser. Vectors from
// different embedders are not comparable, so the index records which one
// produced it and a query embedded with anything else is refused rather than
// silently scoring noise.
//
//   lexical              zero dependencies, deterministic, offline. Hashing
//                        bag-of-words with IDF. Real retrieval, no model: the
//                        fallback for a site that will not ship a second
//                        download, and the one the tests can assert on.
//   transformers:<id>    a small sentence model through @huggingface/
//                        transformers, which is a *devDependency* here and an
//                        optional peer in the browser. Multilingual ids matter
//                        for a site that serves four languages.
//
// The heavy import is dynamic and guarded on purpose: `wasmtune rag index
// --embed lexical` must work in a checkout with nothing installed, the same
// way every other offline command does.

const LEXICAL_DIMS = 512;

export function lexicalTokens(text) {
  return String(text ?? "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .split(/[^a-z0-9_]+/)
    .filter((t) => t.length > 2 && t.length < 24);
}

// FNV-1a, so the hashing is stable across runs and platforms — a Math.random
// projection or a Set iteration order would make the index fresh every build
// and the cache key meaningless.
function hash32(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function l2normalize(vec) {
  let sum = 0;
  for (const v of vec) sum += v * v;
  const norm = Math.sqrt(sum);
  if (!norm) return vec;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

/**
 * Hashing bag-of-words with sublinear term weighting.
 *
 * `idf` is supplied by the index build (document frequency is a property of the
 * corpus, not of a single string); queries pass an empty map and rely on the
 * cosine denominator plus the idf recorded at index time.
 */
export function createLexicalEmbedder({ dims = LEXICAL_DIMS, idf = null } = {}) {
  let weights = idf ? new Map(Object.entries(idf)) : null;
  return {
    id: `lexical-${dims}`,
    dims,
    async embed(texts) {
      const list = Array.isArray(texts) ? texts : [texts];
      return list.map((text) => {
        const vec = new Float32Array(dims);
        const counts = new Map();
        for (const token of lexicalTokens(text)) {
          counts.set(token, (counts.get(token) ?? 0) + 1);
        }
        for (const [token, count] of counts) {
          const weight = (1 + Math.log(count)) * (weights?.get(token) ?? 1);
          vec[hash32(token) % dims] += weight;
        }
        return l2normalize(vec);
      });
    },
    /** Document frequencies for a corpus, as a plain object for JSON. */
    fit(texts) {
      const df = new Map();
      for (const text of texts) {
        for (const token of new Set(lexicalTokens(text))) {
          df.set(token, (df.get(token) ?? 0) + 1);
        }
      }
      const n = Math.max(1, texts.length);
      const out = {};
      for (const [token, count] of df) out[token] = Math.log(1 + n / (1 + count));
      weights = new Map(Object.entries(out));
      return out;
    },
    /** A query embedder that ignores the fitted weights (the index carries them). */
    forQuery() {
      return createLexicalEmbedder({ dims });
    },
  };
}

/**
 * Sentence embeddings through transformers.js.
 *
 * Mean pooling over the last hidden state, then L2 — the standard recipe these
 * sentence models were trained under, and skipping it (CLS pooling, or no
 * normalization) costs recall in a way that looks like "the model is bad"
 * rather than "the pooling is wrong".
 */
export function createTransformersEmbedder({
  model = "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
  dtype = "q8",
  transformersUrl = null,
  loadPipeline = null,
} = {}) {
  let pipe = null;
  let dims = null;
  const ready = (async () => {
    let extract;
    if (loadPipeline) {
      extract = await loadPipeline(model, { dtype });
    } else {
      const spec = transformersUrl ?? "@huggingface/transformers";
      const mod = await import(spec);
      extract = await mod.pipeline("feature-extraction", model, { dtype });
    }
    pipe = extract;
    return pipe;
  })();

  return {
    id: `transformers:${model}`,
    get dims() {
      return dims;
    },
    async embed(texts) {
      const list = Array.isArray(texts) ? texts : [texts];
      const p = await ready;
      const out = await p(list, { pooling: "mean", normalize: true });
      // transformers.js returns a flat tensor; tolist() gives [n][dims].
      const rows = out.tolist();
      dims = rows[0]?.length ?? null;
      return rows.map((row) => Float32Array.from(row));
    },
  };
}

/**
 * Resolve an embedder spec to an implementation.
 *
 * `transformers:<id>` needs the optional package; when it is not installed the
 * error says so and names the escape hatch, because "Cannot find module" in a
 * RAG command reads like a broken install rather than a choice to make.
 */
export async function resolveEmbedder(spec, { idf = null, dtype = "q8", transformersUrl = null, loadPipeline = null } = {}) {
  const s = String(spec ?? "lexical");
  // Accept both spellings: the id this module emits ("lexical-512") and the
  // friendlier CLI form ("lexical:512"). They disagreed once, and the symptom
  // was loadIndex being unable to rebuild the embedder for an index it had
  // written itself — a "unknown embedder" error for a valid index.
  const lexical = s.match(/^lexical(?:[-:](\d+))?$/);
  if (lexical) {
    const dims = Number(lexical[1]) || LEXICAL_DIMS;
    return createLexicalEmbedder({ dims, idf });
  }
  if (s === "transformers" || s.startsWith("transformers:")) {
    const model = s.includes(":") ? s.slice(s.indexOf(":") + 1) : undefined;
    try {
      return createTransformersEmbedder({ ...(model ? { model } : {}), dtype, transformersUrl, loadPipeline });
    } catch (err) {
      throw new Error(
        `embedder "${s}" needs @huggingface/transformers (npm i -D @huggingface/transformers), ` +
        `or use --embed lexical for a zero-dependency index. (${err?.message ?? err})`,
      );
    }
  }
  throw new Error(`unknown embedder "${s}" (expected "lexical", "lexical:<dims>" or "transformers:<model-id>")`);
}

export function isTransformersSpec(spec) {
  return String(spec ?? "").startsWith("transformers");
}
