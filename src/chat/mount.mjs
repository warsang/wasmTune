// wasmtune — one-call chat mounting (generic).
//
// mountAssistant() is the whole consumer integration: fetch the manifest
// the `wasmtune convert` step wrote, detect this device's hardware, pick the
// highest model tier that fits, mount a <site-chat> element, and surface
// hardware/load failures as a blocking banner (never a silent model swap).
// Framework adapters delegate to it.

import { defineSiteChat } from "./SiteChat.js";
import { resolveChatOptions } from "./options.mjs";
import { detectHardware, manifestEntries, pickModel, resolveRequirements, smallerEntries } from "./hardware.mjs";

function currentBase() {
  return globalThis.location?.href
    ?? globalThis.window?.location?.href
    ?? "http://localhost/";
}

export function resolveManifestUrl(manifestUrl, base = null) {
  return new URL(manifestUrl, base ?? currentBase()).href;
}

// Serializable subset of manifest/entry chat hints.
function chatHints(from) {
  return Object.fromEntries(
    Object.entries({
      systemPrompt: from?.systemPrompt,
      temperature: from?.temperature,
      repetitionPenalty: from?.repetitionPenalty,
      presencePenalty: from?.presencePenalty,
      frequencyPenalty: from?.frequencyPenalty,
      maxTokens: from?.maxTokens,
      topP: from?.topP,
      ...(from?.templateKwargs ? { templateKwargs: from.templateKwargs } : {}),
    }).filter(([, v]) => v !== undefined),
  );
}

// Entry artifacts -> loadable forms. URLs resolve against the manifest URL so
// /models can mount anywhere. `model` is the prebuilt WebLLM id when the entry
// is served without fine-tuning.
export function resolveEntryArtifacts(entry, manifestHref) {
  const a = entry?.artifacts ?? {};
  const rawGguf = typeof a.gguf === "string" ? a.gguf : a.gguf?.url;
  const gguf = rawGguf ? new URL(rawGguf, manifestHref).href : null;
  const onnx = typeof a.onnx === "string" ? a.onnx : (a.onnx?.id ?? a.onnx?.repo ?? null);
  const model = a.webllm ?? null;
  let appConfig = null;
  if (a.mlc?.config && a.mlc?.lib) {
    appConfig = {
      model_list: [{
        model: new URL(a.mlc.config, manifestHref).href,
        model_lib: new URL(a.mlc.lib, manifestHref).href,
      }],
    };
  }
  return { gguf, onnx, model, appConfig, chat: entry?.chat ?? {} };
}

// Pure manifest -> load plan (unit-testable, no DOM): resolves artifact URLs
// against the manifest URL and merges manifest chat hints under explicit
// caller options. With `hw` provided, picks the best-fitting model tier;
// without it (legacy callers/tests) the primary entry is used ungated.
export function planFromManifest(manifest, manifestHref, overrides = {}, { hw = null, allowForce = false, preferId = null } = {}) {
  const o = overrides ?? {};
  const entries = manifestEntries(manifest);
  let chosen = null;
  let pick = null;
  let hwMismatch = null;
  let forceTier = null;
  let smaller = [];

  if (entries.length) {
    if (hw) {
      pick = pickModel(manifest, hw, { allowForce, preferId });
      if (pick.entry) {
        chosen = pick.entry;
        smaller = smallerEntries(manifest, pick.entryId, hw)
          .map(({ entry }) => ({ entryId: entry.id, ...resolveEntryArtifacts(entry, manifestHref) }));
        if (pick.forced) {
          hwMismatch = {
            forced: true,
            reasons: pick.reasons,
            required: pick.req,
            detected: detectedOf(hw, pick.budget),
            smallestId: pick.entryId,
          };
        }
      } else {
        hwMismatch = {
          forced: false,
          reasons: pick.reasons,
          required: pick.required,
          detected: detectedOf(hw, pick.budget),
          smallestId: pick.smallestId,
        };
        if (pick.smallest) {
          forceTier = { entryId: pick.smallestId, ...resolveEntryArtifacts(pick.smallest, manifestHref) };
        }
      }
    } else {
      chosen = entries[0];
    }
  }

  const resolved = chosen
    ? resolveEntryArtifacts(chosen, manifestHref)
    : { gguf: null, onnx: null, model: null, appConfig: null, chat: {} };
  const fromManifest = chatHints(manifest?.chat ?? {});
  const fromEntry = chatHints(chosen?.chat ?? {});
  const chatOpts = resolveChatOptions({ ...fromManifest, ...fromEntry, ...o });

  // NOTE: manifest.base (an HF id) is informational only — never pass it as
  // `model`. `model` only carries an explicit caller value or a prebuilt
  // WebLLM id from a pretrained entry's artifacts.
  const model = o.model ?? resolved.model ?? null;
  return {
    gguf: resolved.gguf,
    appConfig: resolved.appConfig,
    onnx: resolved.onnx,
    model,
    chatOpts,
    manifest,
    entries,
    entry: chosen,
    entryId: chosen?.id ?? null,
    entryRequirements: chosen ? resolveRequirements(chosen) : null,
    hw: hw ?? null,
    hwMismatch,
    forceTier,
    smaller,
  };
}

function detectedOf(hw, budget) {
  return {
    budgetGB: budget,
    webgpu: !!hw?.webgpu,
    deviceMemoryGB: hw?.deviceMemoryGB ?? null,
    cores: hw?.cores ?? null,
    mobile: hw?.mobile ?? null,
    adapter: hw?.adapter ?? null,
  };
}

export async function fetchManifest(manifestUrl, { fetchImpl = fetch } = {}) {
  const href = resolveManifestUrl(manifestUrl);
  const res = await fetchImpl(href, { cache: "no-store" });
  if (!res.ok) throw new Error(`manifest fetch failed: ${href} (HTTP ${res.status})`);
  return { manifest: await res.json(), href };
}

/**
 * Mount a fully-wired <site-chat> into `target`.
 *
 * @param {object} args
 * @param {HTMLElement|string} args.target container (element or selector, required)
 * @param {string} [args.manifestUrl="/models/model-manifest.json"]
 * @param {string|null} [args.baseModel] prebuilt id for the explicit opt-in base button
 * @param {string} [args.title="Site assistant"]
 * @param {string} [args.siteName] default system-prompt scope
 * @param {string|null} [args.workerUrl] custom worker build
 * @param {object|null} [args.hardware] override detected hardware (tests / advanced)
 * @param {boolean} [args.allowForce=false] auto-pick the smallest tier even if it doesn't fit
 * @param {string|null} [args.preferModel] force a specific manifest entry id
 * @param {function|null} [args.onEvent] (event) => {} for site-chat-* DOM events
 * @param {...} [args.*] chat options (temperature, maxTokens, templateKwargs, …; win over manifest)
 * @returns {Promise<{element: HTMLElement, ready: Promise<string>, plan: object}>} ready resolves with engine name
 */
export async function mountAssistant({
  target,
  manifestUrl = "/models/model-manifest.json",
  baseModel = null,
  title = "Site assistant",
  siteName = null,
  workerUrl = null,
  hardware = null,
  allowForce = false,
  preferModel = null,
  onEvent = null,
  ...chatOpts
} = {}) {
  if (typeof window === "undefined" || typeof document === "undefined" || typeof customElements === "undefined") {
    throw new Error("mountAssistant needs a browser DOM");
  }
  const host = typeof target === "string" ? document.querySelector(target) : target;
  if (!host) throw new Error("mountAssistant: target container not found");
  defineSiteChat();

  let hw = hardware;
  if (!hw) {
    try {
      hw = await detectHardware();
    } catch {
      hw = null;
    }
  }

  let plan;
  try {
    const { manifest, href } = await fetchManifest(manifestUrl);
    plan = planFromManifest(manifest, href, chatOpts, { hw, allowForce, preferId: preferModel });
  } catch (err) {
    // No manifest (pre-training state): mount an unconfigured element and
    // let the worker report; surface the cause for adapters/debugging.
    plan = {
      gguf: null, appConfig: null, onnx: null, model: chatOpts.model ?? null,
      chatOpts: resolveChatOptions(chatOpts), manifest: null,
      entry: null, entryId: null, entryRequirements: null,
      hw, hwMismatch: null, forceTier: null, smaller: [],
      manifestError: String(err?.message ?? err),
    };
  }

  const el = document.createElement("site-chat");
  if (title) el.setAttribute("title", title);
  if (plan.model) el.setAttribute("model", plan.model);
  if (plan.gguf) el.setAttribute("gguf", plan.gguf);
  if (plan.onnx) el.setAttribute("onnx", plan.onnx);
  if (baseModel) el.setAttribute("base-model", baseModel);
  if (workerUrl) el.setAttribute("worker-url", workerUrl);
  if (allowForce) el.setAttribute("allow-force", "");
  el.appConfig = plan.appConfig;
  el.chatOptsPatch = chatOpts;
  el.siteName = siteName;
  el.baseModel = baseModel;
  el.entryId = plan.entryId;
  el.entryRequirements = plan.entryRequirements;
  el.hw = plan.hw;
  if (plan.hwMismatch) {
    el.hwMismatch = plan.hwMismatch;
    el.forceTier = plan.forceTier;
  }
  if (plan.smaller?.length) el.smallerTiers = plan.smaller;
  if (onEvent) {
    el.addEventListener("site-chat-ready", onEvent);
    el.addEventListener("site-chat-error", onEvent);
    el.addEventListener("site-chat-load-failed", onEvent);
    el.addEventListener("site-chat-hw-mismatch", onEvent);
  }
  const ready = new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error("site-chat: timed out waiting for engine ready")), 10 * 60 * 1000);
    to.unref?.();
    el.addEventListener("site-chat-ready", (e) => {
      clearTimeout(to);
      resolve(e.detail?.engine ?? "unknown");
    }, { once: true });
    el.addEventListener("site-chat-load-failed", () => {
      // Banner is up; ready may still come via explicit opt-in. Do not reject.
    });
  });
  host.append(el);
  return { element: el, ready, plan };
}
