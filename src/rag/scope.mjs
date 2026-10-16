// wasmtune — the scope primitive, in its own module.
//
// It lives here rather than in index.mjs because two files need it and only one
// of them is the search: `store.mjs` decides what may be persisted by comparing
// against the same constant, and a second copy of the string "public" is how a
// session's chunks end up written to IndexedDB while the search thinks they are
// filtered. One definition, imported everywhere.

export const PUBLIC_SCOPE = "public";

export function normalizeScope(scope) {
  return String(scope ?? PUBLIC_SCOPE).trim() || PUBLIC_SCOPE;
}

/** Names allowed to be retrieved. `null` means "no filter" — used by tests only. */
export function scopeSet(scope) {
  if (scope === null) return null;
  const list = Array.isArray(scope) ? scope : [scope];
  return new Set(list.map(normalizeScope));
}
