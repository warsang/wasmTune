// wasmtune — chat WebWorker: WebLLM primary, Transformers.js + wllama fallback.
// Peer deps (@mlc-ai/web-llm, @huggingface/transformers, wllama) are loaded
// lazily at runtime so `npm i wasmtune` stays dependency-free.

import { resolveChatOptions, defaultSystemPrompt, toWebLlmRequest, toWllamaRequest } from "./options.mjs";
import { createLoopGuard, truncateAtLoop, stripThinkingBlocks } from "./guard.mjs";
import { ggufUrl } from "./fallback.mjs";

// Final text for history + done display: loop-truncated, thinking-stripped,
// thinking-leak flagged (eval + debugging signal).
function finalize(text) {
  const cut = truncateAtLoop(text);
  const { text: clean, stripped } = stripThinkingBlocks(cut);
  return { full: clean, thoughtOutLoud: stripped };
}

let engine = null;
let engineKind = "none";
let chatOpts = null;

self.onmessage = async (e) => {
  const msg = e.data || {};
  if (msg.type === "init") return init(msg);
  if (msg.type === "chat") return chat(msg);
};

let strictArtifacts = false;

async function init({ model, appConfig, modelId, gguf, wasmUrl, chatOpts: opts, strictArtifacts: strict }) {
  chatOpts = resolveChatOptions(opts);
  strictArtifacts = !!strict;
  try {
    post({ type: "status", text: "checking WebGPU…" });
    const hasGpu = await hasWebGPU();
    if (hasGpu && (model || appConfig)) {
      try {
        const { CreateMLCEngine } = await import("@mlc-ai/web-llm");
        // Custom fine-tuned build: caller passes the MLC appConfig
        // ({ model_list: [{ model: <config-url>, model_lib: <lib-url> }] })
        // plus the model id to load. Otherwise `model` is a prebuilt id.
        const target = modelId ?? model;
        const opts = {
          initProgressCallback: (p) => post({ type: "status", text: p.text ?? "loading…" }),
          ...(appConfig ? { appConfig } : {}),
        };
        post({ type: "status", text: `loading ${target} (WebGPU, first run downloads once)…` });
        engine = await CreateMLCEngine(target, opts);
        engineKind = "webllm";
        post({ type: "engine", name: "WebLLM · WebGPU" });
        post({ type: "status", text: "ready (local, private)" });
        return;
      } catch (err) {
        // Distinct loadFailed (kept alongside the soft status line): lets
        // the host show a blocking banner instead of silently degrading.
        // Only for custom builds (appConfig); a plain prebuilt id IS the
        // fallback, so its failure stays soft. Fall-through preserved
        // unless strictArtifacts was requested.
        if (appConfig) {
          post({ type: "loadFailed", kind: "mlc", error: String(err?.message ?? err) });
          if (strictArtifacts) {
            post({ type: "status", text: "site model failed — see banner above" });
            return;
          }
        }
        post({ type: "status", text: `WebLLM unavailable (${err.message}), trying fallback…` });
      }
    }
    // wllama GGUF fallback (runs fine-tuned GGUFs, incl. architectures
    // WebLLM can't compile). The URL is content-hashed by `finetune
    // convert`, so browser caches can never serve stale weights.
    // Explicit `gguf` param (manifest artifact or URL); never sniffed
    // from `model` so existing callers are unaffected.
    const gguf = ggufUrl(gguf);
    if (gguf) {
      try {
        const { Wllama } = await import("@wllama/wllama");
        const wasm = wasmUrl
          ?? (await import("@wllama/wllama/esm/wasm/wllama.wasm?url")).default;
        const inst = new Wllama({ default: wasm });
        await inst.loadModelFromUrl(gguf, {
          n_threads: 4,
          n_gpu_layers: 99,
          progressCallback: ({ loaded, total }) => {
            if (total) post({ type: "status", text: `downloading site model… ${Math.round((loaded / total) * 100)}%` });
          },
        });
        engine = { __kind: "wllama", __inst: inst };
        engineKind = "wllama";
        post({ type: "engine", name: "wllama · GGUF" });
        post({ type: "status", text: "ready (local, private)" });
        return;
      } catch (err) {
        post({ type: "loadFailed", kind: "gguf", url: gguf, error: String(err?.message ?? err) });
        if (strictArtifacts) {
          post({ type: "status", text: "site model failed — see banner above" });
          return;
        }
        post({ type: "status", text: `wllama unavailable (${err.message}), trying fallback…` });
      }
    }
    // Transformers.js ONNX fallback
    try {
      await import("@huggingface/transformers");
      engineKind = "transformers";
      post({ type: "engine", name: "Transformers.js" });
      post({ type: "status", text: "ready (ONNX fallback)" });
      return;
    } catch {
      // fall through
    }
    engineKind = "none";
    post({ type: "engine", name: "unavailable" });
    post({ type: "status", text: "no local engine (install @mlc-ai/web-llm for WebGPU chat)" });
  } catch (err) {
    post({ type: "error", message: String(err.message ?? err) });
  }
}

function withSystemPrompt(messages, siteName) {
  if (messages.some((m) => m.role === "system")) return messages;
  const prompt = chatOpts?.systemPrompt ?? defaultSystemPrompt(siteName);
  return [{ role: "system", content: prompt }, ...messages];
}

async function chat({ messages, model, cloudUrl, chatOpts: opts, siteName }) {
  if (opts) chatOpts = resolveChatOptions(opts);
  if (!chatOpts) chatOpts = resolveChatOptions();
  const withSystem = withSystemPrompt(messages, siteName);
  try {
    if (engineKind === "webllm" && engine) {
      const stream = await engine.chat.completions.create(toWebLlmRequest(withSystem, chatOpts));
      let full = "";
      const guard = createLoopGuard();
      for await (const chunk of stream) {
        const t = chunk.choices?.[0]?.delta?.content ?? "";
        if (t) {
          full += t;
          const g = guard.feed(t);
          if (g.tripped) {
            post({ type: "retract", text: g.text, fallback: g.fallback });
            post({ type: "done", full: g.text, looped: true });
            return;
          }
          post({ type: "token", text: t });
        }
      }
      post({ type: "done", ...finalize(full) });
      return;
    }
    if (engineKind === "wllama" && engine?.__inst) {
      // wllama streams cumulative text via onNewToken; the loop guard
      // watches the stream and retracts on trip (same UX as WebLLM path).
      // Post-trip tokens are swallowed (no throw: abort semantics inside
      // the callback are backend-specific and unreliable).
      const guard = createLoopGuard();
      let posted = 0;
      let tripped = false;
      const res = await engine.__inst.createChatCompletion(
        toWllamaRequest(withSystem, chatOpts),
        {
          onNewToken: (_token, _piece, currentText) => {
            if (tripped) return;
            const g = guard.feed(currentText.slice(posted));
            if (g.tripped) {
              tripped = true;
              post({ type: "retract", text: g.text, fallback: g.fallback });
              post({ type: "done", full: g.text, looped: true });
              return;
            }
            post({ type: "token", text: currentText.slice(posted) });
            posted = currentText.length;
          },
        },
      );
      if (tripped) return;
      const text = res?.choices?.[0]?.message?.content ?? "";
      post({ type: "done", ...finalize(text) });
      return;
    }
    if (engineKind === "transformers") {
      const { pipeline } = await import("@huggingface/transformers");
      const gen = await pipeline("text-generation", model ?? "onnx-community/SmolLM2-135M-Instruct-ONNX");
      const out = await gen(withSystem.map((m) => m.content).join("\n"), {
        max_new_tokens: chatOpts.maxTokens,
        temperature: chatOpts.temperature,
        top_p: chatOpts.topP,
        repetition_penalty: chatOpts.repetitionPenalty,
      });
      let text = out?.[0]?.generated_text ?? "";
      // Non-streaming path: truncate at loop onset if one formed.
      const fin = finalize(text);
      post({ type: "token", text: fin.full });
      post({ type: "done", ...fin });
      return;
    }
    if (cloudUrl) {
      const res = await fetch(cloudUrl, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: withSystem, model }),
      });
      const data = await res.json();
      const text = data.reply ?? data.text ?? JSON.stringify(data);
      const fin = finalize(text);
      post({ type: "token", text: fin.full });
      post({ type: "done", ...fin });
      return;
    }
    post({
      type: "error",
      message: "no local model loaded",
      fallback: "Chat model is not available in this browser yet. An admin needs to run `finetune convert` and serve the model files.",
    });
  } catch (err) {
    post({ type: "error", message: String(err.message ?? err) });
  }
}

async function hasWebGPU() {
  try {
    if (!self.navigator?.gpu) return false;
    return !!(await self.navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

function post(m) {
  self.postMessage(m);
}
