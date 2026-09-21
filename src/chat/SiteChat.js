// wasmtune — <site-chat> web component (WebLLM primary).
// Framework-free, Shadow DOM, works on any static site. Heavy inference lives
// in worker.js; this file owns UI + fallback messaging only.

const TEMPLATE = `
<style>
  :host { display: block; font-family: system-ui, sans-serif; }
  .wrap { border: 1px solid #333; border-radius: 12px; overflow: hidden; max-width: 420px; }
  .head { padding: 10px 12px; background: #111; color: #eee; font-size: 13px; display: flex; justify-content: space-between; }
  .log { height: 300px; overflow-y: auto; padding: 10px; background: #0b0b0b; color: #e8e8e8; font-size: 13px; }
  .row { margin: 6px 0; } .u { color: #9ecbff; } .a { color: #d7f0d7; }
  .form { display: flex; border-top: 1px solid #333; }
  input { flex: 1; padding: 10px; border: 0; background: #141414; color: #eee; }
  button { padding: 0 14px; border: 0; background: #2b6cb0; color: white; cursor: pointer; }
  .status { font-size: 11px; color: #888; padding: 6px 10px; background: #111; }
</style>
<div class="wrap">
  <div class="head"><span class="title">Site assistant</span><span class="engine">…</span></div>
  <div class="log" part="log"></div>
  <div class="status">loading model…</div>
  <form class="form"><input placeholder="Ask about this site…" /><button>Send</button></form>
</div>`;

// Inline mirror of guard.mjs stripThinkingBlocks (kept dependency-free so
// defineSiteChat works from any bundler; behavior is pinned by the shared
// test vectors in test/guard.test.mjs).
function stripThinkingInline(text) {
  let out = String(text ?? "");
  let stripped = false;
  for (const tag of ["think", "thought", "thinking"]) {
    const re = new RegExp(`<${tag}[^>]*>[\\s\\S]*?<\\/${tag}>`, "gi");
    if (re.test(out)) {
      stripped = true;
      out = out.replace(re, "");
    }
  }
  const open = out.match(/<(think|thought|thinking)[^>]*>[\s\S]*$/i);
  if (open) {
    stripped = true;
    out = out.slice(0, open.index);
  }
  const startMarker = out.search(/\[\s*start\s+thinking\s*\]/i);
  if (startMarker !== -1) {
    stripped = true;
    const endMarker = out.slice(startMarker).search(/\[\s*end\s+thinking\s*\]/i);
    out = endMarker === -1
      ? out.slice(0, startMarker)
      : out.slice(0, startMarker) + out.slice(startMarker + endMarker).replace(/^\[\s*end\s+thinking\s*\]/i, "");
  }
  out = out.replace(/\n{3,}/g, "\n\n").trim();
  return { text: out, stripped };
}

export function defineSiteChat({ model, gguf = null, workerUrl = null, title = "Site assistant", cloudUrl = null, systemPrompt = null, temperature, repetitionPenalty, presencePenalty, frequencyPenalty, maxTokens, topP, templateKwargs } = {}) {
  if (typeof window === "undefined" || typeof customElements === "undefined") {
    throw new Error("<site-chat> needs a browser DOM");
  }
  if (customElements.get("site-chat")) return;
  // Generation guardrails live in options.mjs (shared with worker.js), but
  // SiteChat.js must stay import-free so `defineSiteChat` works from any
  // bundler; the worker resolves the same defaults itself.
  const chatOpts = { systemPrompt, temperature, repetitionPenalty, presencePenalty, frequencyPenalty, maxTokens, topP, templateKwargs };
  class SiteChat extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: "open" });
      this.attachShadow ? null : null;
      this.shadowRoot.innerHTML = TEMPLATE;
      this.shadowRoot.querySelector(".title").textContent = this.getAttribute("title") || title;
      this._model = this.getAttribute("model") || model;
      this._gguf = this.getAttribute("gguf") || gguf;
      this._baseModel = this.getAttribute("base-model") || null;
      this._cloudUrl = this.getAttribute("cloud-url") || cloudUrl;
      this._workerUrl = this.getAttribute("worker-url") || workerUrl;
      // Per-element overrides, set as plain JS properties before append
      // by mountAssistant / framework adapters:
      //   el.appConfig (MLC custom build), el.chatOptsPatch (merged over
      //   define-time opts), el.baseModel (opt-in fallback id).
      // Fall back to attributes, then define-time options.
      this._chatOpts = chatOpts;
      this._readyFired = false;
      this._loadBlocked = false;
    }
    connectedCallback() {
      const $ = (s) => this.shadowRoot.querySelector(s);
      this._log = $(".log");
      this._status = $(".status");
      this._engine = $(".engine");
      // Pre-append JS props win over attributes, which win over define-time.
      // (Attributes may be set after createElement, so re-read them here.)
      if (this.model) this._model = this.model;
      else if (this.getAttribute("model")) this._model = this.getAttribute("model");
      if (this.gguf) this._gguf = this.gguf;
      else if (this.getAttribute("gguf")) this._gguf = this.getAttribute("gguf");
      if (this.appConfig !== undefined) this._appConfig = this.appConfig;
      this._appConfig = this._appConfig ?? null;
      if (this.chatOptsPatch) this._chatOpts = { ...this._chatOpts, ...this.chatOptsPatch };
      if (this.baseModel) this._baseModel = this.baseModel;
      if (this.getAttribute("base-model")) this._baseModel = this.getAttribute("base-model");
      if (this.siteName) this._siteName = this.siteName;
      if (this.getAttribute("site-name")) this._siteName = this.getAttribute("site-name");
      if (this.getAttribute("title")) this.shadowRoot.querySelector(".title").textContent = this.getAttribute("title");
      this._worker = this._spawnWorker();
      $("form").addEventListener("submit", (e) => {
        e.preventDefault();
        const input = $("input");
        const text = input.value.trim();
        if (!text) return;
        input.value = "";
        this.say("u", text);
        this._worker?.postMessage({ type: "chat", messages: this._history(), model: this._model, cloudUrl: this._cloudUrl, chatOpts: this._chatOpts, siteName: this._siteName });
      });
      this._historyCache = [];
    }
    _history() {
      return this._historyCache.slice(-12);
    }
    // Public API for framework adapters / automation: submit a user message.
    ask(text) {
      const input = this.shadowRoot?.querySelector("input");
      if (!input) return false;
      input.value = String(text ?? "");
      this.shadowRoot.querySelector("form")?.requestSubmit();
      return true;
    }
    get readyState() {
      return this._readyFired ? "ready" : (this._loadBlocked ? "load-failed" : "loading");
    }
    say(who, text) {
      const div = document.createElement("div");
      div.className = "row";
      div.innerHTML = `<span class="${who}">${who === "u" ? "you" : "assistant"}:</span> `;
      div.append(document.createTextNode(text));
      this._log.append(div);
      this._log.scrollTop = this._log.scrollHeight;
      this._historyCache.push({ role: who === "u" ? "user" : "assistant", content: text });
    }
    _spawnWorker(initOverride = null) {
      try {
        if (this._worker) this._worker.terminate();
        const url = this._workerUrl || new URL("./worker.js", import.meta.url);
        const w = new Worker(url, { type: "module" });
        w.onmessage = (e) => {
          const m = e.data || {};
          if (m.type === "status") this._status.textContent = m.text;
          else if (m.type === "engine") {
            this._engine.textContent = m.name;
            if (!this._readyFired) {
              this._readyFired = true;
              this.dispatchEvent(new CustomEvent("site-chat-ready", { bubbles: true, detail: { engine: m.name } }));
            }
          } else if (m.type === "token") this._appendToken(m.text);
          else if (m.type === "retract") this._retractStream(m.text, m.fallback);
          else if (m.type === "loadFailed") this._onArtifactFailed(m);
          else if (m.type === "done" && !m.looped) this._historyCache.push({ role: "assistant", content: m.full });
          else if (m.type === "error") {
            this._status.textContent = m.message;
            if (m.fallback) this.say("a", m.fallback);
            this.dispatchEvent(new CustomEvent("site-chat-error", { bubbles: true, detail: { message: m.message } }));
          }
        };
        const init = initOverride ?? {
          model: this._model, gguf: this._gguf, appConfig: this._appConfig, chatOpts: this._chatOpts,
        };
        this._initParams = init;
        this._wantsArtifact = !!(init.gguf || init.appConfig);
        // Strict when an artifact was requested: the worker stops at the
        // banner instead of racing it with further silent fallbacks.
        w.postMessage({ type: "init", ...init, strictArtifacts: this._wantsArtifact });
        return w;
      } catch (err) {
        this._status.textContent = `chat unavailable: ${err.message}`;
        return null;
      }
    }
    _onArtifactFailed(m) {
      // A manifest artifact failed to load. Never silently degrade: show a
      // blocking banner; the user retries or explicitly opts into base.
      if (!this._wantsArtifact) return; // plain prebuilt/base failure stays soft
      this._loadBlocked = true;
      this._showLoadError({ kind: m.kind, url: m.url, error: m.error });
      this.dispatchEvent(new CustomEvent("site-chat-load-failed", {
        bubbles: true, detail: { kind: m.kind, url: m.url, error: m.error },
      }));
    }
    _showLoadError({ kind, url, error }) {
      this._dismissLoadError();
      const div = document.createElement("div");
      div.className = "row";
      div.dataset.loadError = "1";
      const title = document.createElement("div");
      title.innerHTML = `<span class="a">assistant:</span> `;
      const b = document.createElement("b");
      b.textContent = "Site model failed to load — NOT using a fallback.";
      title.append(b);
      div.append(title);
      const detail = document.createElement("div");
      detail.style.cssText = "font-size:12px;opacity:.85;margin:4px 0";
      detail.textContent = `${kind ?? "model"}${url ? ` · ${url}` : ""} — ${error ?? "unknown error"}`;
      div.append(detail);
      const row = document.createElement("div");
      row.style.cssText = "display:flex;gap:8px;margin-top:4px";
      const retry = document.createElement("button");
      retry.textContent = "↻ Retry site model";
      retry.onclick = () => {
        this._dismissLoadError();
        this._loadBlocked = false;
        this._worker = this._spawnWorker();
      };
      row.append(retry);
      if (this._baseModel) {
        const base = document.createElement("button");
        base.textContent = "Use base model instead";
        base.onclick = () => {
          this._dismissLoadError();
          this._loadBlocked = false;
          this._wantsArtifact = false;
          this._worker = this._spawnWorker({ model: this._baseModel, chatOpts: this._chatOpts });
        };
        row.append(base);
      }
      div.append(row);
      this._log.append(div);
      this._log.scrollTop = this._log.scrollHeight;
      this._engine.textContent = "(load failed)";
      this._status.textContent = "site model failed — see banner above";
    }
    _dismissLoadError() {
      this.shadowRoot.querySelector('[data-load-error="1"]')?.remove();
    }
    _appendToken(t) {
      let last = this._log.lastChild;
      if (!last || !last.dataset || last.dataset.stream !== "1") {
        last = document.createElement("div");
        last.className = "row";
        last.dataset.stream = "1";
        last.dataset.raw = "";
        last.innerHTML = `<span class="a">assistant:</span> `;
        const span = document.createElement("span");
        last.append(span);
        this._log.append(last);
      }
      // Render the thinking-stripped view of the full raw stream so far:
      // reasoning models leak <think> blocks mid-stream otherwise.
      last.dataset.raw += t;
      last.lastChild.textContent = stripThinkingInline(last.dataset.raw).text;
      this._log.scrollTop = this._log.scrollHeight;
    }
    _retractStream(cleanText, fallback) {
      // A loop tripped mid-stream: replace the streaming bubble with the
      // pre-loop text plus a fallback line so users never see the spiral.
      let last = this._log.lastChild;
      if (last?.dataset?.stream === "1") last.remove();
      const div = document.createElement("div");
      div.className = "row";
      const shown = [cleanText, fallback].filter(Boolean).join("\n\n");
      div.innerHTML = `<span class="a">assistant:</span> `;
      div.append(document.createTextNode(shown));
      this._log.append(div);
      this._log.scrollTop = this._log.scrollHeight;
      this._historyCache.push({ role: "assistant", content: shown });
    }
  }
  customElements.define("site-chat", SiteChat);
}
