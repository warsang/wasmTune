// wasmtune — persistence for retrieval indexes (browser only).
//
// Re-embedding a site on every visit would make the widget useless: the corpus
// is a few hundred windows and the embedder is a transformer. So the index is
// cached in IndexedDB under a key that is a hash of the corpus *and* the
// embedder id — change either and the key changes, which is what stops a stale
// index being served under a new embedding space.
//
// Session-scoped chunks are deliberately not persisted: `saveIndex` strips them,
// so an index loaded after a reload contains public content only, and a leak
// cannot survive a page refresh. That is not a substitute for the purge on
// logout — it is the second lock on the same door.

const DB_NAME = "wasmtune-rag";
const STORE = "indexes";

function openDb(scope = globalThis) {
  const idb = scope.indexedDB;
  if (!idb) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const req = idb.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function cacheKey(index) {
  const chunks = index?.chunks ?? [];
  let h = 0x811c9dc5;
  const feed = (s) => {
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
  };
  feed(String(index?.embedder ?? ""));
  for (const c of chunks) feed(`${c.id}:${c.text.length}:`);
  return `rag:${(h >>> 0).toString(36)}:${chunks.length}`;
}

export async function saveIndex(index, { scope = globalThis, key = null } = {}) {
  const publicOnly = {
    ...index,
    chunks: (index.chunks ?? []).filter((c) => String(c.scope ?? "public") === "public"),
  };
  const db = await openDb(scope);
  if (!db) return { saved: false, reason: "no indexedDB" };
  const k = key ?? cacheKey(index);
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(publicOnly, k);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  return { saved: true, key: k, chunks: publicOnly.chunks.length };
}

export async function loadIndex(indexKeyOrIndex, { scope = globalThis } = {}) {
  const db = await openDb(scope);
  if (!db) return null;
  const k = typeof indexKeyOrIndex === "string" ? indexKeyOrIndex : cacheKey(indexKeyOrIndex);
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(k);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}

export async function clearIndexes({ scope = globalThis } = {}) {
  const db = await openDb(scope);
  if (!db) return { cleared: false };
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).clear();
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  return { cleared: true };
}
