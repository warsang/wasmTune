// wasmtune — one-call chat mounting (generic).
//
// mountAssistant() is the whole consumer integration: fetch the manifest
// the `finetune convert` step wrote, pick the best engine for this browser,
// mount a <site-chat> element, and surface load failures as a blocking
// banner (never a silent model swap). Framework adapters delegate to it.

import { defineSiteChat } from "./SiteChat.js";
import { resolveChatOptions } from "./options.mjs";

function currentBase() {
  return globalThis.location?.href
    ?? globalThis.window?.location?.href
    ?? "http://localhost/";
}

export function resolveManifestUrl(manifestUrl, base = null) {
  return new URL(manifestUrl, base ?? currentBase()).href;
}

// Pure manifest -> load plan (unit-testable, no DOM): resolves artifact URLs
// against the manifest URL so /models can mount anywhere, and merges
// manifest chat hints under explicit caller options.
export function planFromManifest(manifest, manifestHref, overrides = {}) {
  const o = overrides ?? {};
  const ref = manifest?.artifacts?.gguf;
  const rawGguf = typeof ref === "string" ? ref : ref?.url;
  const gguf = rawGguf ? new URL(rawGguf, manifestHref).href : null;
  const fromManifest = manifest?.chat ?? {};
  const chatOpts = resolveChatOptions({
    ...(fromManifest.templateKwargs ? { templateKwargs: fromManifest.templateKwargs } : {}),
    ...Object.fromEntries(
      Object.entries({
        systemPrompt: fromManifest.systemPrompt,
        temperature: fromManifest.temperature,
        repetitionPenalty: fromManifest.repetitionPenalty,
        presencePenalty: fromManifest.presencePenalty,
        frequencyPenalty: fromManifest.frequencyPenalty,
        maxTokens: fromManifest.maxTokens,
        topP: fromManifest.topP,
      }).filter(([, v]) => v !== undefined),
    ),
    ...o,
  });
  let appConfig = null;
  // NOTE: manifest.base (an HF id) is informational only — never pass it as
  // `model`. The transformers fallback would try to load it as ONNX weights
  // instead of using its working default, and WebLLM would waste a lookup.
  // `model` is set only when the caller explicitly provides one.
  const model = o.model ?? null;
  if (manifest?.artifacts?.mlc?.config) {
    appConfig = {
      model_list: [{
        model: new URL(manifest.artifacts.mlc.config, manifestHref).href,
        model_lib: new URL(manifest.artifacts.mlc.lib, manifestHref).href,
      }],
    };
  }
  return { gguf, appConfig, model, chatOpts, manifest };
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
 * @param {function|null} [args.onEvent] (event) => {} for site-chat-* DOM events
 * @param {...} [args.*] chat options (temperature, maxTokens, templateKwargs, …; win over manifest)
 * @returns {Promise<{element: HTMLElement, ready: Promise<string>}>} ready resolves with engine name
 */
export async function mountAssistant({
  target,
  manifestUrl = "/models/model-manifest.json",
  baseModel = null,
  title = "Site assistant",
  siteName = null,
  workerUrl = null,
  onEvent = null,
  ...chatOpts
} = {}) {
  if (typeof window === "undefined" || typeof document === "undefined" || typeof customElements === "undefined") {
    throw new Error("mountAssistant needs a browser DOM");
  }
  const host = typeof target === "string" ? document.querySelector(target) : target;
  if (!host) throw new Error("mountAssistant: target container not found");
  defineSiteChat();

  let plan;
  try {
    const { manifest, href } = await fetchManifest(manifestUrl);
    plan = planFromManifest(manifest, href, chatOpts);
  } catch (err) {
    // No manifest (pre-training state): mount an unconfigured element and
    // let the worker report; surface the cause for adapters/debugging.
    plan = { gguf: null, appConfig: null, model: chatOpts.model ?? null, chatOpts: resolveChatOptions(chatOpts), manifest: null, manifestError: String(err?.message ?? err) };
  }

  const el = document.createElement("site-chat");
  if (title) el.setAttribute("title", title);
  if (plan.model) el.setAttribute("model", plan.model);
  if (plan.gguf) el.setAttribute("gguf", plan.gguf);
  if (baseModel) el.setAttribute("base-model", baseModel);
  if (workerUrl) el.setAttribute("worker-url", workerUrl);
  el.appConfig = plan.appConfig;
  el.chatOptsPatch = chatOpts;
  el.siteName = siteName;
  el.baseModel = baseModel;
  if (onEvent) {
    el.addEventListener("site-chat-ready", onEvent);
    el.addEventListener("site-chat-error", onEvent);
    el.addEventListener("site-chat-load-failed", onEvent);
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
