// wasmtune — RAG: index a corpus, retrieve per question, ground the prompt.
//
// The pipeline is deliberately small and entirely local: chunk, embed, cosine,
// top-k. No vector database, no reranker, no agentic retrieval — the same
// reasoning the rest of this package applies to models, that a simple baseline
// you can reason about beats a stack you cannot.
//
// ## Scopes are the security boundary
//
// Every chunk carries a `scope` string. Retrieval takes the scopes the caller is
// allowed to see and filters *inside* the search, before ranking — so a chunk
// from another account can never be returned, however well it scores, and a
// caller cannot forget to filter afterwards. Purge is the other half: when a
// session ends, its chunks are removed from the index rather than merely
// excluded, so a later question cannot retrieve them and a leak needs two
// mistakes instead of one.
//
// Public site content uses the scope it was indexed with ("public"); session
// data uses one scope per subject ("assessment:42").

export const PUBLIC_SCOPE = "public";

export function normalizeScope(scope) {
  return String(scope ?? PUBLIC_SCOPE).trim() || PUBLIC_SCOPE;
}

export function cosine(a, b) {
  let sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

/** Top-k by cosine, with the scope filter applied first. */
export function searchChunks(chunks, queryVec, { topK = 4, minScore = 0.05, scope = null } = {}) {
  const allowed = scope === null ? null : new Set((Array.isArray(scope) ? scope : [scope]).map(normalizeScope));
  const scored = [];
  for (const chunk of chunks) {
    if (!chunk.vector) continue;
    if (allowed && !allowed.has(normalizeScope(chunk.scope))) continue;
    const score = cosine(queryVec, chunk.vector);
    if (score < minScore) continue;
    scored.push({ ...chunk, score });
  }
  scored.sort((a, b) => b.score - a.score || String(a.id).localeCompare(String(b.id)));
  return scored.slice(0, topK);
}

/**
 * Build a retrieval index.
 *
 * `embedder` must be the same one the browser will embed queries with; its id
 * is recorded so a mismatch is a refusal rather than a bad answer.
 */
export async function buildIndex({ documents, embedder, scope = PUBLIC_SCOPE, onProgress = null }) {
  const chunks = documents.map((doc, i) => ({
    id: doc.id ?? `${doc.source ?? "doc"}#${doc.ordinal ?? i}`,
    text: doc.text,
    url: doc.url ?? null,
    title: doc.title ?? null,
    heading: doc.heading ?? null,
    scope: normalizeScope(doc.scope ?? scope),
  }));
  if (!chunks.length) return { version: 1, embedder: embedder.id, scope: normalizeScope(scope), chunks: [] };

  // A corpus-fitted embedder (lexical) needs every text before the first embed.
  const idf = typeof embedder.fit === "function" ? embedder.fit(chunks.map((c) => c.text)) : null;
  const vectors = await embedder.embed(chunks.map((c) => c.text));
  onProgress?.({ stage: "embedded", count: vectors.length });
  return {
    version: 1,
    embedder: embedder.id,
    idf,
    scope: normalizeScope(scope),
    chunks: chunks.map((c, i) => ({ ...c, vector: Array.from(vectors[i]) })),
  };
}

export function indexScopes(index) {
  return [...new Set((index?.chunks ?? []).map((c) => normalizeScope(c.scope)))];
}

/** Drop every chunk of a scope. Used when a session ends or a subject changes. */
export function purgeScope(index, scope) {
  const target = normalizeScope(scope);
  const before = index.chunks.length;
  index.chunks = index.chunks.filter((c) => normalizeScope(c.scope) !== target);
  return { removed: before - index.chunks.length, remaining: index.chunks.length };
}

/**
 * The retrieval context handed to the chat widget.
 *
 * `scope` may be a string or an array. It is read at call time, not captured,
 * so a host can narrow or widen what a session may see without rebuilding the
 * object — and set it to [] on logout, which is the intended way to make a
 * logged-out session see nothing session-scoped.
 */
export function createRag({ index, embedder, topK = 4, minScore = 0.05 } = {}) {
  if (!index) throw new Error("createRag needs an index");
  if (!embedder) throw new Error("createRag needs an embedder");
  // Exact id match, including the lexical dims: two embedders that both look
  // lexical still produce incomparable vectors if they project into different
  // spaces, and the failure mode is a confidently wrong answer rather than an
  // error. Cheap to check, so check.
  if (index.embedder && embedder.id && index.embedder !== embedder.id) {
    throw new Error(
      `index was built with "${index.embedder}" but this embedder is "${embedder.id}" — ` +
      `queries and documents must share an embedder, or the scores are meaningless`,
    );
  }
  let scope = [PUBLIC_SCOPE];
  return {
    index,
    embedder,
    get scope() {
      return scope;
    },
    setScope(next) {
      scope = (Array.isArray(next) ? next : [next]).filter(Boolean).map(normalizeScope);
      return scope;
    },
    async retrieve(question, { topK: k = topK, minScore: s = minScore, scope: overrideScope } = {}) {
      const [queryVec] = await embedder.embed([question]);
      const allowed = overrideScope ?? scope;
      if (!allowed || !allowed.length) return [];
      return searchChunks(index.chunks, queryVec, { topK: k, minScore: s, scope: allowed });
    },
    /** Embed and attach extra chunks (a session's own documents). */
    async addDocuments(documents, { scope: docScope = PUBLIC_SCOPE } = {}) {
      const built = await buildIndex({ documents, embedder, scope: docScope });
      index.chunks.push(...built.chunks);
      return built.chunks.length;
    },
    purgeScope(target) {
      return purgeScope(index, target);
    },
    scopes() {
      return indexScopes(index);
    },
  };
}

/**
 * Load an index JSON and return both the index and a *query* embedder for it.
 *
 * The lexical branch re-creates the embedder without the fitted weights: those
 * are applied when the corpus is embedded, and re-applying them to a query
 * would weight the query by document frequency it does not have.
 */
export async function loadIndex(indexJson, { resolveEmbedder: resolve, dtype = "q8", transformersUrl = null, loadPipeline = null } = {}) {
  const index = typeof indexJson === "string" ? JSON.parse(indexJson) : indexJson;
  const spec = index?.embedder ?? "lexical";
  // The whole spec, dims included: "lexical-512" and "lexical-128" are different
  // spaces, and building the default (512) for a 128-dim index would produce
  // vectors that can never match a stored chunk.
  const embedder = spec.startsWith("transformers")
    ? await resolve(spec, { dtype, transformersUrl, loadPipeline })
    : await resolve(spec, {});
  return { index, embedder };
}

// --- chunking ------------------------------------------------------------

const HEADING_RE = /^(#{1,6})\s+(.+)$/;

/**
 * Split a document into sections at its headings.
 *
 * This is how a chunk gets the right heading: the windowing happens *inside* a
 * section, so the section a chunk came from is a fact about how it was built
 * rather than something inferred afterwards. Inferring it (searching for the
 * heading above the window's offset) mislabels every overlapping window, which
 * is how a chunk from "## Hardware" ends up citing "# Services".
 */
export function splitSections(text) {
  const sections = [];
  let current = { heading: null, level: 0, lines: [] };
  for (const line of String(text ?? "").split(/\r?\n/)) {
    const m = line.match(HEADING_RE);
    if (m) {
      sections.push(current);
      current = { heading: m[2].trim(), level: m[1].length, lines: [] };
      continue;
    }
    current.lines.push(line);
  }
  sections.push(current);
  return sections
    .map((s) => ({ heading: s.heading, level: s.level, body: s.lines.join("\n").trim() }))
    .filter((s) => s.heading || s.body);
}

/**
 * Heading-aware chunks with the metadata a citation needs.
 *
 * `chunkDocs` (the training pipeline's chunker) is reused for the windowing
 * semantics — one chunker, one set of overlap rules — and this adds what
 * retrieval needs on top: the section heading is prepended to the indexed text
 * (so a heading-only question can match), and it is also carried as metadata so
 * the answer can say where it came from.
 */
export function chunksFromDocs(docs, { maxChars = 1200, overlap = 150, minChars = 200, urlFor = null } = {}) {
  const out = [];
  for (const doc of docs) {
    const text = String(doc.text ?? "");
    const sections = splitSections(text);
    const title = sections.find((s) => s.heading)?.heading ?? doc.title ?? null;
    let ordinal = 0;
    for (const section of sections) {
      if (!section.body) continue;
      for (const window of windowText(section.body, maxChars, overlap)) {
        const trimmed = window.trim();
        if (trimmed.length < minChars) continue;
        const body = section.heading ? `${"#".repeat(section.level || 2)} ${section.heading}\n\n${trimmed}` : trimmed;
        out.push({
          id: `${doc.path ?? doc.id ?? "doc"}#${ordinal}`,
          source: doc.path ?? doc.id ?? null,
          url: urlFor ? urlFor(doc) : doc.url ?? null,
          title,
          heading: section.heading ?? title,
          ordinal,
          text: body,
          // Carried through, never re-derived. A session document that lost its
          // scope here would be indexed as public content, which is the one
          // mistake that leaks one account's findings into another's answers.
          scope: doc.scope ?? null,
        });
        ordinal++;
      }
    }
  }
  return out;
}

export function windowText(text, maxChars, overlap) {
  if (text.length <= maxChars) return [text];
  const out = [];
  const paras = text.split(/\n\n+/);
  let cur = "";
  for (const para of paras) {
    if ((cur + "\n\n" + para).length <= maxChars) {
      cur = cur ? `${cur}\n\n${para}` : para;
      continue;
    }
    if (cur) out.push(cur);
    if (para.length > maxChars) {
      let i = 0;
      while (i < para.length) {
        out.push(para.slice(i, i + maxChars));
        i += maxChars - overlap;
      }
      cur = "";
    } else {
      cur = overlap && out.length ? `${out[out.length - 1].slice(-overlap)}\n\n${para}` : para;
      if (cur.length > maxChars) {
        out.push(cur.slice(0, maxChars));
        cur = cur.slice(maxChars - overlap);
      }
    }
  }
  if (cur.trim()) out.push(cur);
  return out;
}
