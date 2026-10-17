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
  /* Pending indicator. An answer can take tens of seconds — the ONNX path is a
     single non-streaming forward pass, and a first run also downloads weights —
     so an empty log is indistinguishable from a hang. */
  .pending .dots::after { content: ""; animation: sc-dots 1.4s steps(4, end) infinite; }
  @keyframes sc-dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }
  .pending .hint { color: #777; font-size: 11px; }
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

function mkButton(label, fn) {
  const b = document.createElement("button");
  b.textContent = label;
  b.style.cssText = "padding:6px 10px;border:1px solid #666;border-radius:6px;background:#1a1a1a;color:#eee;cursor:pointer;font-size:12px";
  b.onclick = fn;
  return b;
}

// --- retrieval + tools, inlined -------------------------------------------
//
// SiteChat.js stays import-free so defineSiteChat() works from any bundler, so
// the three small pieces it needs from src/rag live here as mirrors. Behaviour
// is pinned by the shared vectors in test/rag.test.mjs (grounding format, tool
// marker, citation indexes); the full implementations — chunking, embedders,
// the tool registry, the scope filter — are passed in as objects the host built.
//
// The marker is a factory, not a shared RegExp. A module-level /g regex carries
// `lastIndex` between uses, so the same object matched in one function and
// replaced in another can skip or miss depending on call order — the kind of bug
// that shows up as "the tool fired but the text still shows the marker".
// The payload branch accepts an array as well as an object, matching
// src/rag/action.mjs. Without it a reply like `<tool>name</tool>[1,2]` — which the
// canonical parser rejects as "must be an object" — would keep its raw marker
// visible in the stream and in the answer bubble, because this mirror could not
// match the tag at all.
const TOOL_MARKER_SRC = "<tool>\\s*([a-z0-9_.-]+)\\s*</tool>\\s*(\\{[^}]*\\}|\\[[^\\]]*\\])?";
const toolMarkerRe = () => new RegExp(TOOL_MARKER_SRC, "gi");

function stripToolMarkers(text) {
  return String(text ?? "").replace(toolMarkerRe(), "").replace(/\n{3,}/g, "\n\n").trim();
}

/** The user turn the model sees: numbered context, then the question. */
function groundQuestion(question, passages) {
  if (!passages?.length) return question;
  const context = passages
    .map((p, i) => {
      const where = p.title ? ` — ${p.title}` : "";
      const url = p.url ? ` (${p.url})` : "";
      return `[${i + 1}]${where}${url}\n${p.text}`;
    })
    .join("\n\n");
  return (
    "Answer from the numbered context below. Cite the numbers you used, like [1]. " +
    "If the context does not answer the question, say so plainly instead of guessing.\n\n" +
    `<context>\n${context}\n</context>\n\nQuestion: ${question}`
  );
}

function toolCallIn(text) {
  const matches = [...String(text ?? "").matchAll(toolMarkerRe())];
  if (matches.length !== 1) return null;
  const [match] = matches;
  let args = {};
  if (match[2]) {
    try {
      args = JSON.parse(match[2]);
    } catch {
      return null;
    }
  }
  return { name: match[1].toLowerCase(), args };
}

function citedSources(answer, passages) {
  if (!passages?.length) return [];
  const cited = new Set();
  for (const m of String(answer ?? "").matchAll(/\[(\d{1,2})\]/g)) {
    const n = Number(m[1]);
    if (n >= 1 && n <= passages.length) cited.add(n - 1);
  }
  const indexes = cited.size ? [...cited].sort((a, b) => a - b) : passages.map((_, i) => i);
  const seen = new Set();
  return indexes
    .map((i) => passages[i])
    .filter((p) => p && p.url && !seen.has(p.url) && seen.add(p.url))
    .map((p) => ({ url: p.url, title: p.title ?? null }));
}

export function defineSiteChat({ model, gguf = null, onnx = null, allowForce = false, workerUrl = null, title = "Site assistant", cloudUrl = null, systemPrompt = null, temperature, repetitionPenalty, presencePenalty, frequencyPenalty, maxTokens, topP, templateKwargs } = {}) {
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
      this.shadowRoot.innerHTML = TEMPLATE;
      this.shadowRoot.querySelector(".title").textContent = this.getAttribute("title") || title;
      this._model = this.getAttribute("model") || model;
      this._gguf = this.getAttribute("gguf") || gguf;
      this._onnx = this.getAttribute("onnx") || onnx;
      this._baseModel = this.getAttribute("base-model") || null;
      this._cloudUrl = this.getAttribute("cloud-url") || cloudUrl;
      this._workerUrl = this.getAttribute("worker-url") || workerUrl;
      this._allowForce = allowForce || this.hasAttribute("allow-force");
      // Per-element overrides, set as plain JS properties before append
      // by mountAssistant / framework adapters:
      //   el.appConfig (MLC custom build), el.chatOptsPatch (merged over
      //   define-time opts), el.baseModel (opt-in fallback id),
      //   el.entryId / el.entryRequirements (hardware gate), el.hwMismatch +
      //   el.forceTier (mount-time "too weak" state), el.smallerTiers
      //   (load-failure degrade candidates).
      // Fall back to attributes, then define-time options.
      this._chatOpts = chatOpts;
      this._readyFired = false;
      this._loadBlocked = false;
      this._hwShown = false;
      this._forceHw = false;
      this._smallerTiers = [];
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
      if (this.onnx) this._onnx = this.onnx;
      else if (this.getAttribute("onnx")) this._onnx = this.getAttribute("onnx");
      if (this.onnxDtype) this._onnxDtype = this.onnxDtype;
      else if (this.getAttribute("onnx-dtype")) this._onnxDtype = this.getAttribute("onnx-dtype");
      // The publish commit. Pins the cache key so a re-published model at the
      // same repo path is fetched instead of served from cache.
      if (this.onnxRevision) this._onnxRevision = this.onnxRevision;
      else if (this.getAttribute("onnx-revision")) this._onnxRevision = this.getAttribute("onnx-revision");
      if (this.transformersUrl) this._transformersUrl = this.transformersUrl;
      else if (this.getAttribute("transformers-url")) this._transformersUrl = this.getAttribute("transformers-url");
      if (this.appConfig !== undefined) this._appConfig = this.appConfig;
      this._appConfig = this._appConfig ?? null;
      if (this.chatOptsPatch) this._chatOpts = { ...this._chatOpts, ...this.chatOptsPatch };
      if (this.baseModel) this._baseModel = this.baseModel;
      if (this.getAttribute("base-model")) this._baseModel = this.getAttribute("base-model");
      if (this.siteName) this._siteName = this.siteName;
      if (this.getAttribute("site-name")) this._siteName = this.getAttribute("site-name");
      // worker-url and cloud-url are re-read here for the same reason as every
      // field above: mountAssistant() and the framework adapters createElement()
      // first and set attributes after, so the constructor saw nothing and the
      // worker silently fell back to `new URL("./worker.js", import.meta.url)`.
      // In a bundled app that resolves next to the *page* bundle, not the site
      // root, so a host that correctly passes workerUrl still 404'd its worker.
      if (this.workerUrl) this._workerUrl = this.workerUrl;
      if (this.getAttribute("worker-url")) this._workerUrl = this.getAttribute("worker-url");
      if (this.cloudUrl) this._cloudUrl = this.cloudUrl;
      if (this.getAttribute("cloud-url")) this._cloudUrl = this.getAttribute("cloud-url");
      if (this.getAttribute("allow-force") != null) this._allowForce = true;
      if (this.rag) this._rag = this.rag;
      if (this.tools) this._tools = this.tools;
      if (this.entryId) this._entryId = this.entryId;
      if (this.entryRequirements) this._entryRequirements = this.entryRequirements;
      if (this.hwMismatch) this._hwMismatch = this.hwMismatch;
      if (this.forceTier) this._forceTier = this.forceTier;
      if (this.smallerTiers?.length) this._smallerTiers = [...this.smallerTiers];
      if (this.getAttribute("title")) this.shadowRoot.querySelector(".title").textContent = this.getAttribute("title");
      $("form").addEventListener("submit", (e) => {
        e.preventDefault();
        const input = $("input");
        const text = input.value.trim();
        if (!text) return;
        input.value = "";
        this.say("u", text);
        this._showPending();
        void this._ask(text);
      });
      this._historyCache = [];
      if (this._hwMismatch) this._showHwMismatch(this._hwMismatch);
      else this._worker = this._spawnWorker();
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
    /**
     * Retrieval runs before the model does. The request carries the grounded
     * user turn; the visible log and the history keep what the visitor typed, so
     * a second question is not answered against the first question's context.
     */
    async _ask(text) {
      let passages = [];
      if (this._rag?.retrieve) {
        try {
          passages = await this._rag.retrieve(text);
        } catch (err) {
          // Retrieval failing is not the model failing: answer without context
          // rather than leaving the pending indicator up forever.
          this._status.textContent = `retrieval failed: ${err?.message ?? err}`;
        }
      }
      this._passages = passages;
      const messages = this._history().slice(0, -1);
      messages.push({ role: "user", content: groundQuestion(text, passages) });
      this._post(messages);
    }
    _post(messages = null) {
      this._worker?.postMessage({
        type: "chat",
        messages: messages ?? this._history(),
        model: this._model,
        onnx: this._onnx,
        cloudUrl: this._cloudUrl,
        chatOpts: this._chatOpts,
        siteName: this._siteName,
      });
    }
    /** Narrow or widen what this session may retrieve; [] is the signed-out state. */
    setRagScope(scope) {
      this._rag?.setScope?.(scope);
      return this._rag?.scope ?? null;
    }
    purgeRag(scope) {
      return this._rag?.purgeScope?.(scope) ?? null;
    }
    /**
     * A finished answer may name a tool. Running it is the point of having the
     * model in the page: the page does the thing, with its own code.
     */
    async _afterAnswer(answer) {
      const call = this._tools ? toolCallIn(answer) : null;
      if (!call) {
        this._showSources(answer);
        return;
      }
      let result;
      try {
        result = await this._tools.run(call.name, call.args, {
          element: this,
          siteName: this._siteName,
        });
      } catch (err) {
        result = { ok: false, name: call.name, error: String(err?.message ?? err) };
      }
      this._showToolResult(call, result);
      // An answer can both cite passages and ask for a tool. Showing the tool
      // result must not swallow the citations — they are how a reader checks the
      // part of the answer that came from the site.
      this._showSources(answer);
      this.dispatchEvent(new CustomEvent("site-chat-tool", { bubbles: true, detail: { ...call, result } }));
    }
    _showSources(answer) {
      const sources = citedSources(answer, this._passages ?? []);
      if (!sources.length) return;
      const div = document.createElement("div");
      div.className = "row sources";
      div.style.cssText = "font-size:11px;opacity:.8";
      div.append(document.createTextNode("sources: "));
      sources.forEach((source, i) => {
        if (i) div.append(document.createTextNode(" · "));
        const link = document.createElement("a");
        link.href = source.url;
        link.textContent = source.title ?? source.url;
        link.style.color = "inherit";
        div.append(link);
      });
      this._log.append(div);
      this._log.scrollTop = this._log.scrollHeight;
    }
    _showToolResult(call, result) {
      const div = document.createElement("div");
      div.className = "row tool";
      div.dataset.tool = call.name;
      div.style.cssText = "font-size:12px;opacity:.9";
      const summary = result?.ok
        ? typeof result.result === "string"
          ? result.result
          : result.result?.message ?? "done"
        : result?.error ?? "failed";
      div.textContent = `${result?.ok ? "✓" : "✕"} ${call.name}: ${summary}`;
      this._log.append(div);
      this._log.scrollTop = this._log.scrollHeight;
    }
    get readyState() {
      if (this._readyFired) return "ready";
      if (this._loadBlocked) return "load-failed";
      if (this._hwShown) return "hw-mismatch";
      return "loading";
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
    // "assistant: …" while the model works. Every terminal path clears it, so it
    // can never outlive the turn it belongs to.
    _showPending() {
      this._clearPending();
      const div = document.createElement("div");
      div.className = "row pending";
      div.dataset.pending = "1";
      div.innerHTML = `<span class="a">assistant:</span> <span class="dots"></span> `;
      const hint = document.createElement("span");
      hint.className = "hint";
      hint.textContent = this._readyFired
        ? "generating…"
        : "waiting for the model to load…";
      div.append(hint);
      this._log.append(div);
      this._log.scrollTop = this._log.scrollHeight;
    }
    _clearPending() {
      this.shadowRoot.querySelector('[data-pending="1"]')?.remove();
    }
    // Load a manifest tier: swap the artifact set and (re)spawn the worker.
    _loadTier(tier, { forceHw = false } = {}) {
      this._dismissHwError();
      this._dismissLoadError();
      this._loadBlocked = false;
      this._readyFired = false;
      if ("model" in tier) this._model = tier.model;
      if ("gguf" in tier) this._gguf = tier.gguf;
      if ("onnx" in tier) this._onnx = tier.onnx;
      if ("appConfig" in tier) this._appConfig = tier.appConfig;
      this._forceHw = !!forceHw || this._allowForce;
      this._worker = this._spawnWorker();
    }
    _spawnWorker(initOverride = null) {
      try {
        if (this._worker) this._worker.terminate();
        // Prefer the self-contained bundle that ships in the package: it inlines
        // the engine, so a plain static host needs no bundler and no
        // copy/rewrite step. Fall back to the source worker for unbuilt dev
        // checkouts, where the bundle has not been generated yet.
        const url = this._workerUrl
          || new URL("./bundled/worker.js", import.meta.url);
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
          else if (m.type === "hwMismatch") this._onHwMismatch(m);
          else if (m.type === "done") {
            // Terminal: clear the pending indicator first so it cannot outlive
            // the turn, including on the loop-tripped path.
            this._clearPending();
            if (!m.looped) this._historyCache.push({ role: "assistant", content: m.full });
            void this._afterAnswer(m.full);
          } else if (m.type === "error") {
            this._clearPending();
            this._status.textContent = m.message;
            if (m.fallback) this.say("a", m.fallback);
            this.dispatchEvent(new CustomEvent("site-chat-error", { bubbles: true, detail: { message: m.message } }));
          }
        };
        const init = initOverride ?? {
          model: this._model, gguf: this._gguf, onnx: this._onnx, onnxDtype: this._onnxDtype, onnxRevision: this._onnxRevision,
          transformersUrl: this._transformersUrl,
          appConfig: this._appConfig,
          // A registered tool set gets its instruction appended to the system
          // prompt, because a model cannot ask for a tool it has not been told
          // about — and the tool list is a property of this page, not of training.
          chatOpts: this._tools?.instruction
            ? { ...this._chatOpts, systemPrompt: [this._chatOpts?.systemPrompt, this._tools.instruction()].filter(Boolean).join(" ") }
            : this._chatOpts,
          entryId: this._entryId,
          requirements: this._entryRequirements, forceHw: !!this._forceHw,
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
      // blocking banner; the user retries, drops to a smaller tier, or
      // explicitly opts into base.
      if (!this._wantsArtifact) return; // plain prebuilt/base failure stays soft
      this._loadBlocked = true;
      this._showLoadError({ kind: m.kind, url: m.url, error: m.error });
      this.dispatchEvent(new CustomEvent("site-chat-load-failed", {
        bubbles: true, detail: { kind: m.kind, url: m.url, error: m.error },
      }));
    }
    // Worker-side hardware gate: the selected tier cannot run on this device.
    _onHwMismatch(m) {
      this._showHwMismatch({
        reasons: m.reasons ?? [],
        required: m.required ?? null,
        detected: m.detected ?? null,
      });
    }
    _showHwMismatch(detail) {
      this._clearPending();
      this._dismissHwError();
      const div = document.createElement("div");
      div.className = "row";
      div.dataset.hwError = "1";
      const title = document.createElement("div");
      title.innerHTML = `<span class="a">assistant:</span> `;
      const b = document.createElement("b");
      b.textContent = "This device can't run the local chat model — hardware is below its requirements.";
      title.append(b);
      div.append(title);

      const d = detail?.detected ?? {};
      const bits = [
        d.webgpu
          ? `WebGPU: yes${d.adapter?.vendor ? ` (${d.adapter.vendor}${d.adapter.architecture ? `/${d.adapter.architecture}` : ""})` : ""}`
          : "WebGPU: no",
        d.budgetGB ? `memory: ~${d.budgetGB} GB` : null,
        d.cores ? `${d.cores} cores` : null,
        d.mobile ? "mobile device" : null,
      ].filter(Boolean).join(" · ");
      const detailEl = document.createElement("div");
      detailEl.style.cssText = "font-size:12px;opacity:.85;margin:4px 0";
      detailEl.textContent = `${(detail?.reasons ?? ["requirements not met"]).join("; ")}${bits ? ` — detected: ${bits}` : ""}`;
      div.append(detailEl);

      const row = document.createElement("div");
      row.style.cssText = "display:flex;gap:8px;margin-top:4px";
      row.append(mkButton("Try anyway", () => {
        const tier = this._forceTier ?? {
          model: this._model, gguf: this._gguf, onnx: this._onnx, appConfig: this._appConfig,
        };
        this._loadTier(tier, { forceHw: true });
      }));
      if (this._baseModel) {
        row.append(mkButton("Use base model instead", () => {
          this._dismissHwError();
          this._loadBlocked = false;
          this._wantsArtifact = false;
          this._forceHw = false;
          this._worker = this._spawnWorker({ model: this._baseModel, chatOpts: this._chatOpts, onnxDtype: this._onnxDtype, onnxRevision: this._onnxRevision });
        }));
      }
      div.append(row);
      this._log.append(div);
      this._log.scrollTop = this._log.scrollHeight;
      this._engine.textContent = "(hardware mismatch)";
      this._status.textContent = "hardware below model requirements — try anyway or a smaller model";
      this._hwShown = true;
      this.dispatchEvent(new CustomEvent("site-chat-hw-mismatch", { bubbles: true, detail }));
    }
    _dismissHwError() {
      this.shadowRoot.querySelector('[data-hw-error="1"]')?.remove();
      this._hwShown = false;
    }
    _showLoadError({ kind, url, error }) {
      this._clearPending();
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
      row.append(mkButton("↻ Retry site model", () => {
        this._dismissLoadError();
        this._loadBlocked = false;
        this._worker = this._spawnWorker();
      }));
      // Hardware tiering: a smaller (lower-VRAM) tier often survives what
      // the current one couldn't (GPU OOM, blob-read failures).
      if (this._smallerTiers.length) {
        row.append(mkButton(`Load smaller model (${this._smallerTiers[0].entryId})`, () => {
          const next = this._smallerTiers.shift();
          this._loadTier(next);
        }));
      }
      if (this._baseModel) {
        row.append(mkButton("Use base model instead", () => {
          this._dismissLoadError();
          this._loadBlocked = false;
          this._wantsArtifact = false;
          this._forceHw = false;
          this._worker = this._spawnWorker({ model: this._baseModel, chatOpts: this._chatOpts, onnxDtype: this._onnxDtype, onnxRevision: this._onnxRevision });
        }));
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
      // First token of the turn: the pending indicator has done its job.
      this._clearPending();
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
      // reasoning models leak <think> blocks mid-stream otherwise. Tool markers
      // are stripped too — the marker is an instruction to the page, not
      // something the visitor should watch being typed.
      last.dataset.raw += t;
      last.lastChild.textContent = stripToolMarkers(stripThinkingInline(last.dataset.raw).text);
      this._log.scrollTop = this._log.scrollHeight;
    }
    _retractStream(cleanText, fallback) {
      this._clearPending();
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
