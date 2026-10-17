const A = {
  temperature: 0.3,
  repetitionPenalty: 1.15,
  presencePenalty: 0.3,
  frequencyPenalty: 0.5,
  maxTokens: 256,
  topP: 0.9
};
function K(e = "this website") {
  return `You are the ${e} assistant. Answer questions about the site's documentation and pages in 1-4 sentences. If the docs don't cover a question, say so instead of guessing. Never invent commands, addresses, or field names.`;
}
function k(e = {}) {
  const t = e ?? {}, n = (...a) => {
    for (const l of a) if (t[l] !== void 0) return t[l];
  }, r = (a, l, m, i) => Number.isFinite(Number(a)) ? Math.min(i, Math.max(m, Number(a))) : l, o = A;
  return {
    temperature: r(n("temperature") ?? o.temperature, 0.3, 0, 2),
    repetitionPenalty: r(n("repetitionPenalty", "repetition_penalty") ?? o.repetitionPenalty, 1.15, 1, 2),
    presencePenalty: r(n("presencePenalty", "presence_penalty") ?? o.presencePenalty, o.presencePenalty, -2, 2),
    frequencyPenalty: r(n("frequencyPenalty", "frequency_penalty") ?? o.frequencyPenalty, o.frequencyPenalty, -2, 2),
    maxTokens: Math.floor(r(n("maxTokens", "max_tokens") ?? o.maxTokens, 256, 16, 4096)),
    topP: r(n("topP", "top_p") ?? o.topP, 0.9, 0, 1),
    systemPrompt: typeof n("systemPrompt", "system_prompt") == "string" && n("systemPrompt", "system_prompt").trim() ? n("systemPrompt", "system_prompt") : null,
    // Chat-template kwargs (e.g. {enable_thinking: false} for hybrid
    // reasoning models like Qwen3.5 / Gemma 4). Plain-object passthrough;
    // only sent on paths that support it (wllama).
    templateKwargs: (() => {
      const a = n("templateKwargs", "template_kwargs", "chat_template_kwargs");
      return a && typeof a == "object" && !Array.isArray(a) ? a : {};
    })()
  };
}
function F(e, t) {
  const n = k(t);
  return {
    messages: e,
    stream: !0,
    temperature: n.temperature,
    top_p: n.topP,
    max_tokens: n.maxTokens,
    repetition_penalty: n.repetitionPenalty,
    presence_penalty: n.presencePenalty,
    frequency_penalty: n.frequencyPenalty
  };
}
function X(e, t) {
  const n = k(t), r = {
    messages: e,
    max_tokens: n.maxTokens,
    temperature: n.temperature,
    top_p: n.topP,
    presence_penalty: n.presencePenalty,
    frequency_penalty: n.frequencyPenalty
  };
  return Object.keys(n.templateKwargs).length && (r.chat_template_kwargs = n.templateKwargs), r;
}
function W(e, { n: t = 8, repeats: n = 3 } = {}) {
  const r = String(e ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  if (r.length < t * n) return !1;
  const o = /* @__PURE__ */ new Map();
  for (let a = 0; a + t <= r.length; a++) {
    const l = r.slice(a, a + t).join(" "), m = (o.get(l) ?? 0) + 1;
    if (m >= n) return !0;
    o.set(l, m);
  }
  return !1;
}
const U = "I started repeating myself — let me stop there. Could you rephrase the question?";
function q(e, { n: t = 8, repeats: n = 3 } = {}) {
  const r = String(e ?? "").split(/(\s+)/), o = [], a = [];
  let l = 0;
  for (const i of r) {
    if (/^\s+$/.test(i) || !i) {
      l += i.length;
      continue;
    }
    o.push(i.toLowerCase()), a.push(l), l += i.length;
  }
  const m = /* @__PURE__ */ new Map();
  for (let i = 0; i + t <= o.length; i++) {
    const u = o.slice(i, i + t).join(" "), c = m.get(u);
    if (c) {
      if (c.push(i), c.length >= n) return a[c[1]];
    } else
      m.set(u, [i]);
  }
  return -1;
}
function D(e, t) {
  const n = q(e, t);
  return n === -1 ? String(e ?? "") : String(e).slice(0, n).trimEnd();
}
function R(e) {
  let t = String(e ?? ""), n = !1;
  for (const a of ["think", "thought", "thinking"]) {
    const l = new RegExp(`<${a}[^>]*>[\\s\\S]*?<\\/${a}>`, "gi");
    l.test(t) && (n = !0, t = t.replace(l, ""));
  }
  const r = t.match(/<(think|thought|thinking)[^>]*>[\s\S]*$/i);
  r && (n = !0, t = t.slice(0, r.index));
  const o = t.search(/\[\s*start\s+thinking\s*\]/i);
  if (o !== -1) {
    n = !0;
    const a = t.slice(o).search(/\[\s*end\s+thinking\s*\]/i);
    t = a === -1 ? t.slice(0, o) : t.slice(0, o) + t.slice(o + a).replace(/^\[\s*end\s+thinking\s*\]/i, "");
  }
  return t = t.replace(/\n{3,}/g, `

`).trim(), { text: t, stripped: n };
}
function G({ checkEveryChars: e = 64, tailChars: t = 1200, minChars: n = 200, detectOpts: r = {}, fallback: o = U } = {}) {
  let a = "", l = 0, m = !1;
  return {
    get tripped() {
      return m;
    },
    get text() {
      return a;
    },
    feed(i) {
      if (m) return { tripped: !0, text: a };
      if (a += String(i ?? ""), a.length - l < e || a.length < n)
        return { tripped: !1, text: a };
      l = a.length;
      const u = a.slice(-t);
      if (W(u, r)) {
        m = !0;
        const c = q(u, r), p = c === -1 ? a.length : a.length - u.length + c;
        return a = a.slice(0, p).trimEnd(), { tripped: !0, text: a, fallback: o };
      }
      return { tripped: !1, text: a };
    }
  };
}
const $ = [
  {
    hf: "HuggingFaceTB/SmolLM2-135M-Instruct",
    // NOT ...-q4f16_1-MLC: web-llm's prebuilt registry never shipped a q4f16
    // build of the 135M, so that id fails at engine init with "Cannot find
    // model record in appConfig". Only these two exist, and both are
    // unquantized — which is why this entry's footprint is fp16-sized, not
    // the ~78MB a q4 build would need. The ONNX path below is unaffected.
    webllm: "SmolLM2-135M-Instruct-q0f16-MLC",
    onnx: "onnx-community/SmolLM2-135M-Instruct-ONNX",
    params: "135M",
    vram: "~270MB",
    vramBytes: 27e7,
    context: "2K",
    family: "SmolLM2",
    mobileOk: !0,
    cpuOk: !0
  },
  {
    hf: "HuggingFaceTB/SmolLM2-360M-Instruct",
    webllm: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-360M-Instruct-ONNX",
    params: "360M",
    vram: "~210MB",
    vramBytes: 22e7,
    context: "2K",
    family: "SmolLM2",
    mobileOk: !0,
    cpuOk: !0
  },
  {
    hf: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    webllm: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/SmolLM2-1.7B-Instruct-ONNX",
    params: "1.7B",
    vram: "~1GB",
    vramBytes: 107e7,
    context: "2K",
    family: "SmolLM2",
    mobileOk: !1,
    cpuOk: !1
  },
  {
    hf: "Qwen/Qwen2.5-0.5B-Instruct",
    webllm: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Qwen2.5-0.5B-Instruct-ONNX",
    params: "0.5B",
    vram: "~278MB",
    vramBytes: 29e7,
    context: "4K",
    family: "Qwen",
    mobileOk: !0,
    cpuOk: !0
  },
  {
    hf: "Qwen/Qwen2.5-1.5B-Instruct",
    webllm: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Qwen2.5-1.5B-Instruct-ONNX",
    params: "1.5B",
    vram: "~868MB",
    vramBytes: 91e7,
    context: "4K",
    family: "Qwen",
    mobileOk: !1,
    cpuOk: !1
  },
  {
    hf: "Qwen/Qwen3-0.6B",
    webllm: "Qwen3-0.6B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-0.6B-ONNX",
    params: "0.6B",
    vram: "~350MB",
    vramBytes: 37e7,
    context: "4K",
    family: "Qwen",
    mobileOk: !0,
    cpuOk: !0
  },
  {
    hf: "Qwen/Qwen3-1.7B",
    webllm: "Qwen3-1.7B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-1.7B-ONNX",
    params: "1.7B",
    vram: "~1.1GB",
    vramBytes: 115e7,
    context: "4K",
    family: "Qwen",
    mobileOk: !1,
    cpuOk: !1
  },
  {
    hf: "Qwen/Qwen3-4B-Instruct-2507",
    mlx: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    webllm: "Qwen3-4B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-4B-ONNX",
    params: "4B",
    vram: "~2.2GB",
    vramBytes: 236e7,
    context: "32K",
    family: "Qwen",
    mobileOk: !1,
    cpuOk: !1,
    note: "best fine-tuned SLM (distil-labs 12x8 study); non-thinking instruct"
  },
  {
    hf: "Qwen/Qwen3-4B",
    webllm: "Qwen3-4B-q4f16_1-MLC",
    onnx: "onnx-community/Qwen3-4B-ONNX",
    params: "4B",
    vram: "~2.2GB",
    vramBytes: 236e7,
    context: "4K",
    family: "Qwen",
    mobileOk: !1,
    cpuOk: !1
  },
  {
    hf: "meta-llama/Llama-3.2-1B-Instruct",
    webllm: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    params: "1B",
    vram: "~712MB",
    vramBytes: 75e7,
    context: "4K",
    family: "Llama",
    license: "llama-community (accept terms)",
    mobileOk: !0,
    cpuOk: !1
  },
  {
    hf: "meta-llama/Llama-3.2-3B-Instruct",
    webllm: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    onnx: "onnx-community/Llama-3.2-3B-Instruct-ONNX",
    params: "3B",
    vram: "~1.76GB",
    vramBytes: 185e7,
    context: "4K",
    family: "Llama",
    license: "llama-community (accept terms)",
    mobileOk: !1,
    cpuOk: !1
  },
  {
    hf: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    webllm: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
    onnx: "onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX",
    params: "1.1B",
    vram: "~400MB",
    vramBytes: 42e7,
    context: "2K",
    family: "TinyLlama",
    mobileOk: !0,
    cpuOk: !0
  },
  {
    hf: "microsoft/Phi-3.5-mini-instruct",
    webllm: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    onnx: "microsoft/Phi-3-mini-4k-instruct-onnx-web",
    params: "3.8B",
    vram: "~2.1GB",
    vramBytes: 225e7,
    context: "4K",
    family: "Phi",
    mobileOk: !1,
    cpuOk: !1
  },
  {
    hf: "google/gemma-2-2b-it",
    webllm: "gemma-2-2b-it-q4f16_1-MLC",
    onnx: "onnx-community/gemma-2-2b-it-ONNX",
    params: "2B",
    vram: "~1.44GB",
    vramBytes: 155e7,
    context: "4K",
    family: "Gemma",
    license: "gemma terms",
    mobileOk: !1,
    cpuOk: !1
  },
  {
    hf: "google/gemma-3-1b-it",
    // gemma3-*, not gemma-3-*: web-llm names this family without the hyphen,
    // so the older spelling failed at engine init.
    webllm: "gemma3-1b-it-q4f16_1-MLC",
    onnx: "onnx-community/gemma-3-1b-it-ONNX",
    params: "1B",
    vram: "~700MB",
    vramBytes: 75e7,
    context: "8K",
    family: "Gemma",
    license: "gemma terms",
    mobileOk: !0,
    cpuOk: !1
  },
  {
    hf: "Qwen/Qwen3.5-4B",
    mlx: "mlx-community/Qwen3.5-4B-MLX-4bit",
    gguf: "unsloth/Qwen3.5-4B-GGUF",
    params: "4B",
    vram: "~3GB",
    vramBytes: 32e8,
    context: "262K",
    family: "Qwen",
    mobileOk: !1,
    cpuOk: !1,
    note: "hybrid thinking (select non-thinking via chat template); strongest sub-10B reasoning"
  },
  {
    hf: "google/gemma-4-E4B-it",
    mlx: "mlx-community/gemma-4-e4b-it-4bit",
    gguf: "ggml-org/gemma-4-E4B-it-GGUF",
    params: "4.5B eff",
    vram: "~3GB",
    vramBytes: 32e8,
    context: "128K",
    family: "Gemma",
    mobileOk: !1,
    cpuOk: !1,
    // Converted GGUF is ~3.9GB; the embedded q2 embedding tensor plus KV
    // cache needs real headroom — keep the 6GB floor.
    minDeviceMemoryGB: 6,
    note: "disable thinking via enable_thinking=false; VLM arch (mlx_lm text path)"
  }
];
function E(e) {
  const t = Number(e) || 0;
  return t < 5e8 ? "tiny" : t < 16e8 ? "small" : t < 45e8 ? "mid" : "large";
}
function j(e) {
  const t = Number(e) || 0;
  return t <= 5e8 ? 1 : t <= 16e8 ? 2 : t <= 32e8 ? 4 : 8;
}
const I = $.map((e) => ({
  ...e,
  tier: e.tier ?? E(e.vramBytes),
  minDeviceMemoryGB: e.minDeviceMemoryGB ?? j(e.vramBytes)
})), z = /* @__PURE__ */ new Set([
  "Qwen/Qwen2.5-7B-Instruct",
  "Qwen/Qwen3-8B",
  "meta-llama/Llama-3.1-8B-Instruct",
  "mistralai/Mistral-7B-Instruct-v0.3",
  "google/gemma-2-9b-it"
]), H = new Map(I.map((e) => [e.hf.toLowerCase(), e])), O = /* @__PURE__ */ new Map();
for (const e of I) {
  for (const t of ["webllm", "onnx", "gguf", "mlx"])
    e[t] && O.set(String(e[t]).toLowerCase(), e);
  e.webllm && O.set(e.webllm.toLowerCase().split("/").pop(), e), e.gguf && O.set(String(e.gguf).toLowerCase().split("/").pop(), e);
}
function J(e) {
  return H.get(String(e ?? "").toLowerCase()) ?? null;
}
function V(e) {
  return O.get(String(e ?? "").toLowerCase()) ?? null;
}
function Q(e) {
  const t = J(e) ?? V(e) ?? null;
  return t ? {
    hf: t.hf,
    params: t.params,
    vramBytes: t.vramBytes,
    minDeviceMemoryGB: t.minDeviceMemoryGB,
    tier: t.tier,
    needsWebGPU: !t.cpuOk,
    mobileOk: !!t.mobileOk,
    cpuOk: !!t.cpuOk,
    context: t.context,
    source: "lib"
  } : z.has(String(e)) ? {
    hf: String(e),
    params: "7-9B",
    vramBytes: 8e9,
    minDeviceMemoryGB: 8,
    tier: "large",
    needsWebGPU: !0,
    mobileOk: !1,
    cpuOk: !1,
    context: null,
    source: "lib-large"
  } : null;
}
async function Y(e = globalThis) {
  const t = e?.navigator ?? {}, n = {
    webgpu: !1,
    adapter: null,
    // { vendor, architecture, device, maxBufferSize, maxStorageBufferBindingSize }
    deviceMemoryGB: null,
    // Chromium only (capped at 8); null elsewhere
    cores: null,
    mobile: null,
    crossOriginIsolated: typeof e?.crossOriginIsolated == "boolean" ? e.crossOriginIsolated : !1
  };
  try {
    const r = Number(t.deviceMemory);
    Number.isFinite(r) && r > 0 && (n.deviceMemoryGB = r);
  } catch {
  }
  try {
    const r = Number(t.hardwareConcurrency);
    Number.isFinite(r) && r > 0 && (n.cores = r);
  } catch {
  }
  try {
    n.mobile = t.userAgentData?.mobile ?? /Mobi|Android|iPhone|iPad|Tablet/i.test(String(t.userAgent ?? ""));
  } catch {
    n.mobile = !1;
  }
  try {
    if (t.gpu?.requestAdapter) {
      const r = await t.gpu.requestAdapter();
      if (r) {
        n.webgpu = !0;
        let o = r.info ?? null;
        if (!o && typeof r.requestAdapterInfo == "function")
          try {
            o = await r.requestAdapterInfo();
          } catch {
            o = null;
          }
        n.adapter = {
          vendor: o?.vendor ?? "",
          architecture: o?.architecture ?? "",
          device: o?.device ?? "",
          maxBufferSize: Number(r.limits?.maxBufferSize) || null,
          maxStorageBufferBindingSize: Number(r.limits?.maxStorageBufferBindingSize) || null
        };
      }
    }
  } catch {
    n.webgpu = !1;
  }
  return n;
}
function Z(e) {
  if (Number.isFinite(e?.deviceMemoryGB) && e.deviceMemoryGB > 0) return e.deviceMemoryGB;
  const t = Number(e?.cores) || 0;
  return e?.mobile ? t >= 8 ? 4 : 2 : e?.webgpu ? (Number(e.adapter?.maxBufferSize) || 0) >= 2 ** 30 || t >= 8 ? 8 : 4 : t >= 8 ? 4 : 2;
}
function ee(e, t) {
  const n = [], r = Z(t);
  return e.needsWebGPU && !t?.webgpu && n.push("requires WebGPU, which this browser/device does not expose"), Number(e.minDeviceMemoryGB) > r && n.push(`needs ~${e.minDeviceMemoryGB} GB device memory (estimated available: ~${r} GB)`), t?.mobile && !e.mobileOk && n.push("too large to run comfortably on a mobile device"), { ok: n.length === 0, reasons: n, budget: r };
}
function te(e) {
  return e ? typeof e == "string" ? e : e.url ?? null : null;
}
function M(e) {
  const t = D(e), { text: n, stripped: r } = R(t);
  return { full: n, thoughtOutLoud: r };
}
let x = null, w = "none", g = null, S = null, P = null, C = null;
async function T() {
  return import(C ?? "@huggingface/transformers");
}
self.onmessage = async (e) => {
  const t = e.data || {};
  if (t.type === "init") return ne(t);
  if (t.type === "chat") return ae(t);
};
let _ = !1;
async function ne({ model: e, appConfig: t, modelId: n, gguf: r, onnx: o = null, onnxDtype: a = null, onnxRevision: l = null, transformersUrl: m = null, wasmUrl: i, chatOpts: u, strictArtifacts: c, entryId: p = null, requirements: d = null, forceHw: v = !1 }) {
  g = k(u), _ = !!c, S = a, P = l, C = m;
  try {
    if (!v) {
      const f = d ?? Q(p) ?? Q(e);
      if (f) {
        const y = await Y(self), h = ee(f, y);
        if (!h.ok) {
          s({
            type: "hwMismatch",
            reasons: h.reasons,
            required: f,
            detected: {
              budgetGB: h.budget,
              webgpu: !!y.webgpu,
              deviceMemoryGB: y.deviceMemoryGB,
              cores: y.cores,
              mobile: y.mobile,
              adapter: y.adapter
            }
          }), s({ type: "status", text: "hardware below model requirements — see banner above" });
          return;
        }
      }
    }
    if (s({ type: "status", text: "checking WebGPU…" }), await se() && (e || t))
      try {
        const { CreateMLCEngine: f } = await import("./worker-index-BnBBEHDo.js"), y = n ?? e, h = {
          initProgressCallback: (L) => s({ type: "status", text: L.text ?? "loading…" }),
          ...t ? { appConfig: t } : {}
        };
        s({ type: "status", text: `loading ${y} (WebGPU, first run downloads once)…` }), x = await f(y, h), w = "webllm", s({ type: "engine", name: "WebLLM · WebGPU" }), s({ type: "status", text: "ready (local, private)" });
        return;
      } catch (f) {
        if (t && (s({ type: "loadFailed", kind: "mlc", error: String(f?.message ?? f) }), _)) {
          s({ type: "status", text: "site model failed — see banner above" });
          return;
        }
        s({ type: "status", text: `WebLLM unavailable (${f.message}), trying fallback…` });
      }
    const B = te(r);
    if (B)
      try {
        const { Wllama: f } = await import("@wllama/wllama"), y = i ?? (await import("./worker-wllama-Drvd9dNr.js")).default, h = new f({ default: y });
        await h.loadModelFromUrl(B, {
          n_threads: 4,
          n_gpu_layers: 99,
          progressCallback: ({ loaded: L, total: N }) => {
            N && s({ type: "status", text: `downloading site model… ${Math.round(L / N * 100)}%` });
          }
        }), x = { __kind: "wllama", __inst: h }, w = "wllama", s({ type: "engine", name: "wllama · GGUF" }), s({ type: "status", text: "ready (local, private)" });
        return;
      } catch (f) {
        if (s({ type: "loadFailed", kind: "gguf", url: B, error: String(f?.message ?? f) }), _) {
          s({ type: "status", text: "site model failed — see banner above" });
          return;
        }
        s({ type: "status", text: `wllama unavailable (${f.message}), trying fallback…` });
      }
    try {
      await T(), w = "transformers", s({ type: "engine", name: "Transformers.js" }), s({ type: "status", text: "ready (ONNX fallback)" });
      return;
    } catch {
    }
    w = "none", s({ type: "engine", name: "unavailable" }), s({ type: "status", text: "no local engine (install @mlc-ai/web-llm for WebGPU chat)" });
  } catch (b) {
    s({ type: "error", message: String(b.message ?? b), stack: String(b.stack ?? "") });
  }
}
function re(e, t) {
  return e.some((r) => r.role === "system") ? e : [{ role: "system", content: g?.systemPrompt ?? K(t) }, ...e];
}
async function ae({ messages: e, model: t, onnx: n = null, cloudUrl: r, chatOpts: o, siteName: a }) {
  o && (g = k(o)), g || (g = k());
  const l = re(e, a);
  try {
    if (w === "webllm" && x) {
      const m = await x.chat.completions.create(F(l, g));
      let i = "";
      const u = G();
      for await (const c of m) {
        const p = c.choices?.[0]?.delta?.content ?? "";
        if (p) {
          i += p;
          const d = u.feed(p);
          if (d.tripped) {
            s({ type: "retract", text: d.text, fallback: d.fallback }), s({ type: "done", full: d.text, looped: !0 });
            return;
          }
          s({ type: "token", text: p });
        }
      }
      s({ type: "done", ...M(i) });
      return;
    }
    if (w === "wllama" && x?.__inst) {
      const m = G();
      let i = 0, u = !1;
      const c = await x.__inst.createChatCompletion(
        X(l, g),
        {
          onNewToken: (d, v, b) => {
            if (u) return;
            const B = m.feed(b.slice(i));
            if (B.tripped) {
              u = !0, s({ type: "retract", text: B.text, fallback: B.fallback }), s({ type: "done", full: B.text, looped: !0 });
              return;
            }
            s({ type: "token", text: b.slice(i) }), i = b.length;
          }
        }
      );
      if (u) return;
      const p = c?.choices?.[0]?.message?.content ?? "";
      s({ type: "done", ...M(p) });
      return;
    }
    if (w === "transformers") {
      const { pipeline: m } = await T(), c = (await (await m(
        "text-generation",
        n ?? t ?? "onnx-community/SmolLM2-135M-Instruct-ONNX",
        {
          ...S ? { dtype: S } : {},
          ...P ? { revision: P } : {}
        }
      ))(l, {
        max_new_tokens: g.maxTokens,
        temperature: g.temperature,
        top_p: g.topP,
        repetition_penalty: g.repetitionPenalty
      }))?.[0]?.generated_text, p = Array.isArray(c) ? String(c.filter((v) => v?.role === "assistant").at(-1)?.content ?? "") : String(c ?? ""), d = M(p);
      s({ type: "token", text: d.full }), s({ type: "done", ...d });
      return;
    }
    if (r) {
      const i = await (await fetch(r, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: l, model: t })
      })).json(), u = i.reply ?? i.text ?? JSON.stringify(i), c = M(u);
      s({ type: "token", text: c.full }), s({ type: "done", ...c });
      return;
    }
    s({
      type: "error",
      message: "no local model loaded",
      fallback: "Chat model is not available in this browser yet. An admin needs to run `wasmtune convert` and serve the model files."
    });
  } catch (m) {
    s({ type: "error", message: String(m.message ?? m), stack: String(m.stack ?? "") });
  }
}
async function se() {
  try {
    return self.navigator?.gpu ? !!await self.navigator.gpu.requestAdapter() : !1;
  } catch {
    return !1;
  }
}
function s(e) {
  self.postMessage(e);
}
