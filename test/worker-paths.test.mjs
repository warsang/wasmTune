import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { readFileSync } from "node:fs";
import { inlineChatTemplate } from "../src/publish.mjs";

const WORKER = new URL("../src/chat/worker.js", import.meta.url);
// Strip comments before scanning: the regression comment for this very bug
// quotes the offending line verbatim, so a naive scan matches the explanation.
const src = () =>
  readFileSync(WORKER, "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");

describe("worker: no self-shadowing declaration", () => {
  // This shipped broken and was invisible because the WebLLM branch returns
  // early on success. `const gguf = ggufUrl(gguf)` shadowed the destructured
  // `gguf` parameter with a binding declared in the same scope, so the
  // initializer read it before initialisation:
  //   ReferenceError: Cannot access 'gguf' before initialization
  // It threw on EVERY path that reached it — GGUF, ONNX, and any browser where
  // WebLLM failed to initialise — so the documented fallback chain never ran.

  it("does not declare a const from an identifier of the same name", () => {
    const s = src();
    // Match `const X = something(X)` / `let X = f(X)` where X also appears as
    // a destructured parameter, which is the exact shape of the bug.
    const params = new Set([...s.matchAll(/async function init\(\{([^}]*)\}/g)]
      .flatMap((m) => m[1].split(",").map((p) => p.split("=")[0].trim())));
    const offenders = [];
    for (const m of s.matchAll(/\b(?:const|let)\s+(\w+)\s*=\s*([\w$.]+)\((\w+)\)/g)) {
      const [, declared, , arg] = m;
      if (declared === arg && params.has(declared)) offenders.push(m[0]);
    }
    assert.deepEqual(offenders, [], `self-shadowing declaration(s): ${offenders.join("; ")}`);
  });

  it("still resolves the gguf url under a distinct name", () => {
    const s = src();
    assert.match(s, /const ggufTarget = ggufUrl\(gguf\)/);
    assert.doesNotMatch(s, /const gguf = ggufUrl\(gguf\)/);
    // And the resolved value is what gets loaded, not the raw param.
    assert.match(s, /loadModelFromUrl\(ggufTarget/);
  });});

describe("worker: transformers path", () => {
  it("honours an onnx dtype from the manifest", () => {
    const s = src();
    // transformers.js resolves onnx/model_<dtype>.onnx (q8 -> model_quantized.onnx),
    // so a published tier has to say which dtype it uploaded.
    assert.match(s, /onnxDtypeResolved \? \{ dtype: onnxDtypeResolved \}/);
  });

  it("passes messages through so the chat template is applied", () => {
    const s = src();
    // Joining message contents with "\n" makes an instruct model continue the
    // system prompt instead of answering, and it echoed the prompt verbatim.
    assert.doesNotMatch(s, /withSystem\.map\(\(m\) => m\.content\)\.join/);
    assert.match(s, /await gen\(withSystem,/);
    // A messages input returns a messages array, not a string.
    assert.match(s, /Array\.isArray\(g\)/);
  });

  it("loads transformers from a URL when the host supplies one", () => {
    const s = src();
    assert.match(s, /transformersUrlResolved \?\? "@huggingface\/transformers"/);
    // @vite-ignore is a directive carried in a comment, so it has to be checked
    // against the raw source rather than the comment-stripped copy.
    assert.match(readFileSync(WORKER, "utf8"), /@vite-ignore/);
  });
});

describe("publish: chat template inlining", () => {
  it("inlines chat_template.jinja into tokenizer_config.json", async () => {
    // transformers.js does not read the .jinja file; without the inline copy it
    // fails at generation time with "tokenizer.chat_template is not set".
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-tmpl-"));
    await writeFile(path.join(dir, "tokenizer_config.json"),
      JSON.stringify({ tokenizer_class: "GPT2Tokenizer", bos_token: "<s>" }));
    await writeFile(path.join(dir, "chat_template.jinja"), "{% for m in messages %}{{ m }}{% endfor %}");

    const out = await inlineChatTemplate({ dir });
    assert.ok(out, "expected a corrected file to be written");
    const cfg = JSON.parse(await readFile(out, "utf8"));
    assert.match(cfg.chat_template, /for m in messages/);
    assert.equal(cfg.tokenizer_class, "GPT2Tokenizer", "existing keys preserved");
  });

  it("leaves an already-present template alone", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-tmpl2-"));
    await writeFile(path.join(dir, "tokenizer_config.json"),
      JSON.stringify({ chat_template: "already here" }));
    await writeFile(path.join(dir, "chat_template.jinja"), "ignored");
    assert.equal(await inlineChatTemplate({ dir }), null);
  });

  it("does nothing when the jinja file is absent", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "wasmtune-tmpl3-"));
    await writeFile(path.join(dir, "tokenizer_config.json"), "{}");
    assert.equal(await inlineChatTemplate({ dir }), null);
  });
});
