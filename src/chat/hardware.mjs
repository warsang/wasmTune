// wasmtune — client-side hardware detection + model tier picking.
//
// Runs in both window and worker scopes (no node imports, injectable for
// tests). The web cannot read exact RAM, so detection is a set of signals
// combined conservatively: WebGPU adapter presence/limits, Chromium's
// navigator.deviceMemory (capped at 8), CPU core count and the mobile flag.
// Per-model requirements come from the lib's own knowledge
// (src/models.mjs allowlist) — explicit manifest/config overrides win.

import { requirementsFor, tierForBytes, minMemoryForBytes } from "../models.mjs";

export const TIER_RANK = { tiny: 0, small: 1, mid: 2, large: 3 };

// ---------------------------------------------------------------------------
// Detection (async: WebGPU adapter request)
// ---------------------------------------------------------------------------

export async function detectHardware(scope = globalThis) {
  const nav = scope?.navigator ?? {};
  const hw = {
    webgpu: false,
    adapter: null, // { vendor, architecture, device, maxBufferSize, maxStorageBufferBindingSize }
    deviceMemoryGB: null, // Chromium only (capped at 8); null elsewhere
    cores: null,
    mobile: null,
    crossOriginIsolated: typeof scope?.crossOriginIsolated === "boolean" ? scope.crossOriginIsolated : false,
  };
  try {
    const dm = Number(nav.deviceMemory);
    if (Number.isFinite(dm) && dm > 0) hw.deviceMemoryGB = dm;
  } catch { /* unsupported */ }
  try {
    const c = Number(nav.hardwareConcurrency);
    if (Number.isFinite(c) && c > 0) hw.cores = c;
  } catch { /* unsupported */ }
  try {
    hw.mobile = nav.userAgentData?.mobile
      ?? /Mobi|Android|iPhone|iPad|Tablet/i.test(String(nav.userAgent ?? ""));
  } catch {
    hw.mobile = false;
  }
  try {
    if (nav.gpu?.requestAdapter) {
      const adapter = await nav.gpu.requestAdapter();
      if (adapter) {
        hw.webgpu = true;
        let info = adapter.info ?? null;
        if (!info && typeof adapter.requestAdapterInfo === "function") {
          try { info = await adapter.requestAdapterInfo(); } catch { info = null; }
        }
        hw.adapter = {
          vendor: info?.vendor ?? "",
          architecture: info?.architecture ?? "",
          device: info?.device ?? "",
          maxBufferSize: Number(adapter.limits?.maxBufferSize) || null,
          maxStorageBufferBindingSize: Number(adapter.limits?.maxStorageBufferBindingSize) || null,
        };
      }
    }
  } catch {
    hw.webgpu = false;
  }
  return hw;
}

// Estimated device-memory budget in GB. deviceMemory is authoritative when
// present (Chromium; values are capped at 8). Without it (Safari/Firefox)
// fall back to a deliberately conservative heuristic — a wrong "fits" here
// means the tab OOMs, so unknowns resolve low.
export function memoryBudgetGB(hw) {
  if (Number.isFinite(hw?.deviceMemoryGB) && hw.deviceMemoryGB > 0) return hw.deviceMemoryGB;
  const cores = Number(hw?.cores) || 0;
  if (hw?.mobile) return cores >= 8 ? 4 : 2;
  if (hw?.webgpu) {
    const maxBuf = Number(hw.adapter?.maxBufferSize) || 0;
    if (maxBuf >= 2 ** 30 || cores >= 8) return 8;
    return 4;
  }
  return cores >= 8 ? 4 : 2;
}

// ---------------------------------------------------------------------------
// Manifest shape helpers
// ---------------------------------------------------------------------------

// v2 manifests carry models[]; legacy manifests (single artifacts block) are
// reshaped into one implicit entry so every consumer has one code path.
export function manifestEntries(manifest) {
  if (Array.isArray(manifest?.models) && manifest.models.length) {
    return manifest.models.map((m, i) => ({ id: m.id ?? `model-${i}`, ...m }));
  }
  const artifacts = manifest?.artifacts ?? {};
  if (!Object.keys(artifacts).length) return [];
  return [{
    id: manifest?.id ?? "default",
    label: null,
    base: manifest?.base ?? null,
    source: "tuned",
    artifacts,
    chat: manifest?.chat ?? {},
  }];
}

export function entryArtifacts(entry) {
  return entry?.artifacts ?? {};
}

export function ggufOf(entry) {
  const g = entryArtifacts(entry).gguf;
  if (!g) return null;
  return typeof g === "string" ? g : (g.url ?? null);
}

function ggufBytes(entry) {
  const g = entryArtifacts(entry).gguf;
  if (!g || typeof g !== "object") return null;
  return Number(g.bytes) || null;
}

// ---------------------------------------------------------------------------
// Requirements + fit
// ---------------------------------------------------------------------------

// Resolve the effective requirements for a manifest entry:
// explicit entry.requirements > lib knowledge (allowlist lookup by base /
// artifact ids) > artifact-size estimate with safety overhead.
export function resolveRequirements(entry) {
  const explicit = entry?.requirements ?? {};
  const artifacts = entryArtifacts(entry);
  const lib = requirementsFor(entry?.base)
    ?? requirementsFor(entry?.id)
    ?? requirementsFor(artifacts.webllm)
    ?? requirementsFor(artifacts.onnx)
    ?? requirementsFor(artifacts.gguf)
    ?? null;
  const bytes = ggufBytes(entry);
  const estVram = bytes ? Math.round(bytes * 1.3) : null;
  const vramBytes = Number(explicit.vramBytes) || lib?.vramBytes || estVram || null;
  const minDeviceMemoryGB = Number(explicit.minDeviceMemoryGB)
    || lib?.minDeviceMemoryGB
    || (vramBytes ? minMemoryForBytes(vramBytes) : null);
  return {
    known: !!(lib || explicit.minDeviceMemoryGB || estVram),
    source: explicit.minDeviceMemoryGB ? "explicit" : lib?.source ?? (estVram ? "artifact-size" : "unknown"),
    hf: lib?.hf ?? entry?.base ?? null,
    params: explicit.params ?? lib?.params ?? null,
    vramBytes,
    minDeviceMemoryGB: minDeviceMemoryGB ?? 2,
    tier: explicit.tier ?? lib?.tier ?? tierForBytes(vramBytes ?? 2e9),
    needsWebGPU: explicit.needsWebGPU ?? lib?.needsWebGPU ?? ((vramBytes ?? 0) > 1e9),
    mobileOk: explicit.mobileOk ?? lib?.mobileOk ?? ((vramBytes ?? 0) < 1.5e9),
    cpuOk: explicit.cpuOk ?? lib?.cpuOk ?? ((vramBytes ?? 0) < 1e9),
  };
}

export function fitRequirements(req, hw) {
  const reasons = [];
  const budget = memoryBudgetGB(hw);
  if (req.needsWebGPU && !hw?.webgpu) {
    reasons.push("requires WebGPU, which this browser/device does not expose");
  }
  if (Number(req.minDeviceMemoryGB) > budget) {
    reasons.push(`needs ~${req.minDeviceMemoryGB} GB device memory (estimated available: ~${budget} GB)`);
  }
  if (hw?.mobile && !req.mobileOk) {
    reasons.push("too large to run comfortably on a mobile device");
  }
  return { ok: reasons.length === 0, reasons, budget };
}

export function fitReport(entry, hw) {
  const req = resolveRequirements(entry);
  return { ...fitRequirements(req, hw), req };
}

export function rankEntries(entries) {
  return [...entries].sort((a, b) => {
    const ra = resolveRequirements(a);
    const rb = resolveRequirements(b);
    const tier = (TIER_RANK[rb.tier] ?? 0) - (TIER_RANK[ra.tier] ?? 0);
    if (tier !== 0) return tier;
    return (rb.vramBytes ?? 0) - (ra.vramBytes ?? 0);
  });
}

// ---------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------

// Choose the highest tier that fits this device. Returns:
//   { entry, entryId, req, hw, budget, reasons: [], forced, candidates }
// or when nothing fits and force is not allowed:
//   { entry: null, none: true, reasons, required, smallest, hw, budget, candidates }
export function pickModel(manifest, hw, { allowForce = false, preferId = null } = {}) {
  const ranked = rankEntries(manifestEntries(manifest));
  const candidates = ranked.map((entry) => {
    const fit = fitReport(entry, hw);
    return { entry, entryId: entry.id, ...fit };
  });
  const budget = memoryBudgetGB(hw);

  if (preferId) {
    const hit = candidates.find((c) => c.entryId === preferId);
    if (hit && (hit.ok || allowForce)) {
      return {
        entry: hit.entry, entryId: hit.entryId, req: hit.req, hw, budget,
        reasons: hit.reasons, forced: hit.ok ? false : true, candidates,
      };
    }
  }
  const fit = candidates.find((c) => c.ok);
  if (fit) {
    return {
      entry: fit.entry, entryId: fit.entryId, req: fit.req, hw, budget,
      reasons: [], forced: false, candidates,
    };
  }
  const smallest = candidates[candidates.length - 1] ?? null;
  if (allowForce && smallest) {
    return {
      entry: smallest.entry, entryId: smallest.entryId, req: smallest.req, hw, budget,
      reasons: smallest.reasons, forced: true, candidates,
    };
  }
  return {
    entry: null, none: true, hw, budget, candidates,
    reasons: smallest?.reasons ?? ["no models are published in the manifest"],
    required: smallest?.req ?? null,
    smallest: smallest?.entry ?? null,
    smallestId: smallest?.entryId ?? null,
  };
}

// Entries ranked strictly below `entryId`, in descending order — used for the
// "load smaller model" recovery when a load fails (e.g. GPU OOM).
export function smallerEntries(manifest, entryId, hw = null) {
  const ranked = rankEntries(manifestEntries(manifest));
  const idx = ranked.findIndex((e) => e.id === entryId);
  if (idx === -1) return ranked.slice(1).map((entry) => ({ entry, entryId: entry.id }));
  return ranked.slice(idx + 1).map((entry) => ({ entry, entryId: entry.id }));
}
