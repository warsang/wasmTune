// wasmtune demo — mounts the real <site-chat> widget on a static docs page and
// renders the tier picker with the same functions the widget itself uses.
//
// Nothing here is simulated except the explicitly-labelled "simulate another
// visitor" control, which feeds invented specs into the real detect/pick code.

import {
  mountAssistant,
  detectHardware,
  memoryBudgetGB,
  manifestEntries,
  rankEntries,
  fitReport,
  pickModel,
  TIER_RANK,
} from "../src/chat/index.mjs";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const BASE = import.meta.env?.BASE_URL ?? "/";
const manifestUrl = (large) => `${BASE}models/model-manifest${large ? "-large" : ""}.json`;

// ── simulated visitor profiles ────────────────────────────────────────
// Only the `hardware` override is synthetic. The picker that consumes it is the
// library's own code path (detectHardware -> pickModel -> fitReport).
const PROFILES = {
  phone:   { webgpu: false, adapter: null, deviceMemoryGB: 2, cores: 4, mobile: true, crossOriginIsolated: false },
  laptop:  { webgpu: true, adapter: { vendor: "nvidia", architecture: "", device: "", maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 30 }, deviceMemoryGB: 4, cores: 8, mobile: false, crossOriginIsolated: false },
  desktop: { webgpu: true, adapter: { vendor: "nvidia", architecture: "ampere", device: "RTX 3060", maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 }, deviceMemoryGB: 8, cores: 16, mobile: false, crossOriginIsolated: false },
  big:     { webgpu: true, adapter: { vendor: "nvidia", architecture: "ada", device: "RTX 4090", maxBufferSize: 2 ** 33, maxStorageBufferBindingSize: 2 ** 33 }, deviceMemoryGB: 8, cores: 32, mobile: false, crossOriginIsolated: true },
};

const DEFAULT_TIER = "smollm2-135m"; // smallest: opens in ~80 MB instead of ~1 GB

let manifest = null;
let realHw = null;
let simKey = "";
let preferId = DEFAULT_TIER;
let large = false;
let mounted = null;
let engine = "not loaded";

// URL state: ?sim=phone shows what a phone visitor gets, ?tiers=large adds the
// 4B tier, ?tier=<id> pins a specific model. Makes a scenario linkable and
// shareable instead of something you can only reach by clicking.
function readUrlState() {
  const q = new URLSearchParams(location.search);
  const sim = q.get("sim");
  if (sim && PROFILES[sim]) { simKey = sim; $("sim").value = sim; }
  if (q.get("tiers") === "large") { large = true; $("big-tiers").checked = true; }
  const tier = q.get("tier");
  if (tier) preferId = tier;
}

const hwOf = () => (simKey ? PROFILES[simKey] : realHw);

const fmtBytes = (n) => {
  if (!Number.isFinite(n) || n <= 0) return "—";
  const mb = n / 1e6;
  return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`;
};

async function loadManifest() {
  const res = await fetch(manifestUrl(large), { cache: "no-store" });
  if (!res.ok) throw new Error(`manifest ${manifestUrl(large)} -> HTTP ${res.status}`);
  manifest = await res.json();
}

// ── hardware list ─────────────────────────────────────────────────────
function renderHw() {
  const hw = hwOf();
  const list = $("hw-list");
  list.innerHTML = "";
  if (!hw) return;

  const rows = [
    ["WebGPU", hw.webgpu ? "yes" : "no", hw.webgpu ? "yes" : "no"],
    ["GPU", hw.adapter?.device || hw.adapter?.architecture || hw.adapter?.vendor || "not reported", ""],
    ["max buffer", hw.adapter?.maxBufferSize ? fmtBytes(hw.adapter.maxBufferSize) : "—", ""],
    ["deviceMemory", hw.deviceMemoryGB ? `${hw.deviceMemoryGB} GB` : "not exposed", hw.deviceMemoryGB ? "" : "dim"],
    ["cores", hw.cores ? String(hw.cores) : "—", ""],
    ["mobile", hw.mobile ? "yes" : "no", ""],
    ["budget used", `~${memoryBudgetGB(hw)} GB`, ""],
    ["source", simKey ? `simulated (${simKey})` : "measured in this tab", ""],
  ];
  for (const [k, v, cls] of rows) {
    list.append(el("dt", null, k));
    list.append(el("dd", cls, v));
  }
  $("sim-note").hidden = !simKey;
  $("use-real").hidden = !simKey;
  $("hw-how").textContent = simKey ? "simulated specs" : "detected in this tab";
}

// ── tier ladder ───────────────────────────────────────────────────────
function renderTiers() {
  const hw = hwOf();
  const host = $("tier-list");
  host.innerHTML = "";

  // Two different questions, kept apart on purpose:
  //   best  — what this hardware should run, ignoring the current choice
  //   load  — what the widget is actually running right now
  const best = pickModel(manifest, hw, { allowForce: false });
  const bestId = best.entryId ?? null;

  for (const entry of rankEntries(manifestEntries(manifest))) {
    const fit = fitReport(entry, hw);
    const isBest = entry.id === bestId;
    const isLoaded = entry.id === preferId;
    const li = el("li", `tier ${fit.ok ? "fit" : "nofit"}${isLoaded ? " picked" : ""}`);

    li.append(el("span", "mark", fit.ok ? "✓" : "·"));

    const mid = el("div");
    const name = el("div", "tier-name", entry.label ?? entry.id);
    if (isLoaded) name.append(Object.assign(document.createElement("span"), { textContent: "  ← loaded" }));
    mid.append(name);

    const r = fit.req;
    mid.append(el("div", "tier-meta",
      `${r.params ?? "?"} · ~${fmtBytes(r.vramBytes)} · needs ${r.minDeviceMemoryGB} GB` +
      `${r.needsWebGPU ? " + WebGPU" : ""}${r.mobileOk ? " · mobile-ok" : ""} · reqs from ${r.source}`));
    if (!fit.ok) mid.append(el("div", "tier-why", fit.reasons.join("; ")));
    li.append(mid);

    const right = el("div", "tier-right");
    right.append(el("span", `tag tier-${r.tier}`, `${r.tier} · L${(TIER_RANK[r.tier] ?? 0) + 1}`));
    if (isBest) right.append(el("span", "tag", "best fit"));
    if (!isLoaded) {
      const btn = el("button", "tier-cta", isBest ? "load best" : "load this");
      btn.type = "button";
      btn.title = fit.ok
        ? `Load ${entry.label} (~${fmtBytes(r.vramBytes)} on first visit)`
        : `Force ${entry.label} — it does not fit these specs and may fail`;
      btn.onclick = () => { preferId = entry.id; syncUrl(); renderTiers(); renderPicked(); mount(); };
      right.append(btn);
    }
    li.append(right);
    host.append(li);
  }
}

// ── picked-tier summary ───────────────────────────────────────────────
function renderPicked() {
  const hw = hwOf();
  const host = $("picked");
  host.innerHTML = "";
  const best = pickModel(manifest, hw, { allowForce: false });
  const loaded = rankEntries(manifestEntries(manifest)).find((e) => e.id === preferId);

  if (!best.entry) {
    const b = el("div");
    b.innerHTML = `<b>No shipped tier fits this device.</b> <span class="bad">${(best.reasons ?? []).join("; ")}</span>`;
    host.append(b);
    host.append(el("div", "row2",
      `wasmtune does not silently swap in a different model here — the widget shows a blocking banner with the detected specs and a "Try anyway" button instead. Smallest tier is ${best.smallestId ?? "n/a"}.`));
    return;
  }

  const r = best.req;
  const b = el("div");
  b.innerHTML =
    `<b>Best tier for these specs:</b> ${best.entry.label ?? best.entryId} <span class="ok">(${best.entryId})</span>` +
    ` — ${r.params ?? "?"}, ~${fmtBytes(r.vramBytes)} footprint, needs ${r.minDeviceMemoryGB} GB` +
    `${r.needsWebGPU ? " and WebGPU" : ""}. Detected budget: ~${memoryBudgetGB(hw)} GB, WebGPU ${hw.webgpu ? "available" : "absent"}.`;
  host.append(b);

  const lrow = el("div", "row2");
  const big = loaded && (best.entryId !== loaded.id);
  lrow.textContent = big
    ? `Currently loaded: ${loaded.label} (${loaded.id}) — deliberately below the best fit so this page opens in a few seconds. Click "load best" to move up. Engine: ${engine}.`
    : `Currently loaded: ${loaded?.label ?? preferId} — the best tier these specs allow. Engine: ${engine}.`;
  host.append(lrow);
  host.append(el("div", "row2",
    "First visit downloads the weights once (~" + fmtBytes(loaded ? fitReport(loaded, hw).req.vramBytes : null) +
    ") and the browser caches them; later visits are offline. A real `wasmtune convert` run ships your tuned weights here instead of these public pretrained builds."));
}

// ── worker watchdog ────────────────────────────────────────────────────
// <site-chat> renders "loading model…" from its template and only replaces it
// when the worker posts a message. If the worker script 404s, throws on load,
// or never starts, the widget sits on that string forever with no error
// anywhere — which is exactly what a missing worker.js looked like. Watch the
// status line and the engine label, and say something useful if nothing moves.
function startWorkerWatchdog() {
  const el = document.querySelector("#chat site-chat");
  if (!el?.shadowRoot) return;
  const $s = (sel) => el.shadowRoot?.querySelector(sel);
  const status = $s(".status");
  const engine = $s(".engine");
  if (!status || !engine) return;

  let last = "";
  let changedAt = performance.now();
  const IDLE_MS = 25_000;
  const tick = setInterval(() => {
    const now = `${status.textContent}|${engine.textContent}`;
    if (now !== last) {
      last = now;
      changedAt = performance.now();
      return;
    }
    // Ready, failed, or answered: nothing to guard.
    if (/ready|unavailable|failed|below|error|smaller/i.test(now)) return;
    if (performance.now() - changedAt < IDLE_MS) return;

    clearInterval(tick);
    const stillLoading = /loading/i.test(status.textContent ?? "");
    status.textContent = stillLoading
      ? "the chat worker never started — worker.js may be missing from this deployment"
      : status.textContent;
    status.style.color = "var(--bad)";
    engine.textContent = "(worker failed)";
    const row = document.createElement("div");
    row.className = "row";
    row.dataset.workerError = "1";
    const b = document.createElement("b");
    b.textContent = "The chat worker did not start.";
    row.append(b);
    const detail = document.createElement("div");
    detail.style.cssText = "font-size:12px;opacity:.85;margin:4px 0";
    detail.textContent =
      `worker.js should be served next to index.html. Check the network tab for ` +
      `/worker.js — a 404 there means the build did not emit it. The model tier ` +
      `picker below still works; it does not need the worker.`;
    row.append(detail);
    el.shadowRoot.querySelector(".log")?.prepend(row);
    console.error("[wasmtune demo] chat worker appears not to have started", {
      status: status.textContent, engine: engine.textContent,
    });
  }, 3000);
}

// ── mounting ──────────────────────────────────────────────────────────
async function mount() {
  if (!manifest) return;
  if (mounted) {
    // SiteChat has no disconnectedCallback; stop the old worker explicitly so
    // switching tiers does not leave a GPU engine alive behind the widget.
    mounted.element._worker?.terminate?.();
    mounted.element.remove();
    mounted = null;
  }
  engine = "loading…";
  renderPicked();

  mounted = await mountAssistant({
    target: "#chat",
    manifestUrl: manifestUrl(large),
    // The chat worker is a separate build (vite.worker.config.js); SiteChat
    // cannot be told where to find it by static analysis, so we pass it in.
    workerUrl: `${BASE}worker.js`,
    siteName: "Lumen",
    title: "Lumen assistant",
    hardware: hwOf(),          // real detectHardware(), or the labelled simulation
    preferModel: preferId,
    allowForce: false,
    onEvent: (e) => {
      if (e.type === "site-chat-ready") engine = e.detail?.engine ?? "ready";
      else if (e.type === "site-chat-hw-mismatch") engine = "hardware mismatch";
      else if (e.type === "site-chat-load-failed") engine = `load failed (${e.detail?.kind ?? "?"})`;
      else if (e.type === "site-chat-error") engine = `error: ${e.detail?.message ?? "?"}`;
      renderPicked();
    },
  });
}

// ── controls ──────────────────────────────────────────────────────────
// Keep the address bar in sync so any scenario on screen can be linked to.
function syncUrl() {
  const q = new URLSearchParams();
  if (simKey) q.set("sim", simKey);
  if (large) q.set("tiers", "large");
  if (preferId && preferId !== DEFAULT_TIER) q.set("tier", preferId);
  const qs = q.toString();
  // globalThis.history explicitly — a bare `history` identifier in a browser
  // module is a collision waiting to happen.
  globalThis.history.replaceState(null, "", qs ? `?${qs}${location.hash}` : location.pathname + location.hash);
}

$("sim").onchange = (e) => {
  simKey = e.target.value;
  preferId = DEFAULT_TIER;
  syncUrl();
  renderHw();
  renderTiers();
  renderPicked();
  mount();
};
$("use-real").onclick = () => { $("sim").value = ""; $("sim").dispatchEvent(new Event("change")); };
$("big-tiers").onchange = async (e) => {
  large = e.target.checked;
  preferId = DEFAULT_TIER;
  syncUrl();
  try {
    await loadManifest();
  } catch (err) {
    $("picked").innerHTML = `<span class="bad">${String(err.message ?? err)}</span>`;
    return;
  }
  renderTiers();
  renderPicked();
  mount();
};

(async () => {
  readUrlState();
  try {
    realHw = await detectHardware();
  } catch {
    realHw = { webgpu: false, deviceMemoryGB: null, cores: null, mobile: false, adapter: null };
  }
  try {
    await loadManifest();
  } catch (err) {
    $("picked").innerHTML = `<span class="bad">${String(err.message ?? err)}</span>`;
    return;
  }
  renderHw();
  renderTiers();
  renderPicked();
  await mount();
  startWorkerWatchdog();
})();
