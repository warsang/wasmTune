import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { Window } from "happy-dom";
import {
  mountAssistant,
  fetchManifest,
  planFromManifest,
  resolveManifestUrl,
} from "../src/chat/mount.mjs";

const MANIFEST = {
  base: "google/gemma-4-E4B-it",
  version: "abc123",
  artifacts: {
    gguf: {
      url: "/models/kf-gemma4-abc12345/kf-gemma4-e4b.Q4_K_M-00001-of-00009.gguf",
      sha256: "deadbeef",
      bytes: 123,
    },
  },
  chat: { templateKwargs: { enable_thinking: false }, temperature: 0.5 },
};

describe("resolveManifestUrl", () => {
  it("resolves relative to a base", () => {
    assert.equal(
      resolveManifestUrl("/models/m.json", "https://x.test/docs/"),
      "https://x.test/models/m.json",
    );
  });
});

describe("planFromManifest", () => {
  const href = "https://x.test/models/model-manifest.json";

  it("resolves gguf against the manifest URL (mount-independent)", () => {
    const plan = planFromManifest(MANIFEST, href, {});
    assert.equal(
      plan.gguf,
      "https://x.test/models/kf-gemma4-abc12345/kf-gemma4-e4b.Q4_K_M-00001-of-00009.gguf",
    );
  });

  it("accepts legacy string gguf artifacts", () => {
    const plan = planFromManifest({ artifacts: { gguf: "m.gguf" } }, href, {});
    assert.equal(plan.gguf, "https://x.test/models/m.gguf");
  });

  it("merges manifest chat hints under explicit overrides", () => {
    const plan = planFromManifest(MANIFEST, href, { temperature: 0.7 });
    assert.equal(plan.chatOpts.temperature, 0.7); // caller wins
    assert.deepEqual(plan.chatOpts.templateKwargs, { enable_thinking: false }); // manifest fills gaps
  });

  it("never leaks manifest.base into model", () => {
    const plan = planFromManifest(MANIFEST, href, {});
    assert.equal(plan.model, null);
    const explicit = planFromManifest(MANIFEST, href, { model: "foo/bar" });
    assert.equal(explicit.model, "foo/bar");
  });

  it("builds mlc appConfig with resolved URLs", () => {
    const plan = planFromManifest(
      { artifacts: { mlc: { config: "mlc/c.json", lib: "mlc/lib.wasm" } } },
      href, {},
    );
    assert.equal(plan.appConfig.model_list[0].model, "https://x.test/models/mlc/c.json");
    assert.equal(plan.appConfig.model_list[0].model_lib, "https://x.test/models/mlc/lib.wasm");
  });

  it("handles missing artifacts gracefully", () => {
    const plan = planFromManifest({}, href, {});
    assert.equal(plan.gguf, null);
    assert.equal(plan.appConfig, null);
  });
});

describe("fetchManifest", () => {
  it("returns manifest + href on 200", async () => {
    const fakeFetch = async (url, opts) => ({
      ok: true,
      json: async () => MANIFEST,
    });
    const { manifest, href } = await fetchManifest("/models/m.json", { fetchImpl: fakeFetch });
    assert.equal(href, "http://localhost/models/m.json");
    assert.equal(manifest.version, "abc123");
  });

  it("throws on non-200", async () => {
    const fakeFetch = async () => ({ ok: false, status: 404 });
    await assert.rejects(fetchManifest("/models/nope.json", { fetchImpl: fakeFetch }), /404/);
  });
});

describe("mountAssistant", () => {
  let window;
  const prev = {};

  beforeEach(async () => {
    window = new Window({ url: "https://x.test/docs/" });
    for (const k of ["window", "document", "customElements", "HTMLElement", "CustomEvent"]) {
      prev[k] = globalThis[k];
      globalThis[k] = window[k];
    }
    // happy-dom lacks Worker: element degrades gracefully, no throw.
    await window.happyDOM.waitUntilComplete();
  });

  afterEach(async () => {
    for (const k of Object.keys(prev)) globalThis[k] = prev[k];
    await window.happyDOM.close();
  });

  it("mounts a configured <site-chat> into the target", async () => {
    const realFetch = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: true, json: async () => MANIFEST });
    try {
      const host = document.createElement("div");
      document.body.append(host);
      const { element, plan } = await mountAssistant({
        target: host,
        manifestUrl: "/models/model-manifest.json",
        title: "Demo",
        temperature: 0.7,
      });
      assert.equal(element.tagName.toLowerCase(), "site-chat");
      assert.equal(element.getAttribute("title"), "Demo");
      assert.equal(
        element.getAttribute("gguf"),
        "https://x.test/models/kf-gemma4-abc12345/kf-gemma4-e4b.Q4_K_M-00001-of-00009.gguf",
      );
      assert.equal(plan.chatOpts.temperature, 0.7);
      assert.deepEqual(plan.chatOpts.templateKwargs, { enable_thinking: false });
      // Worker unavailable under happy-dom: element degrades, stays mounted.
      assert.ok(host.querySelector("site-chat"));
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  it("throws on missing target", async () => {
    await assert.rejects(mountAssistant({ target: "#nope" }), /target container not found/);
  });

  it("mounts unconfigured element when manifest fetch fails", async () => {
    const realFetch = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: false, status: 404 });
    try {
      const host = document.createElement("div");
      document.body.append(host);
      const { element, plan } = await mountAssistant({ target: host });
      assert.equal(element.tagName.toLowerCase(), "site-chat");
      assert.equal(plan.manifest, null);
      assert.match(plan.manifestError, /404/);
    } finally {
      globalThis.fetch = realFetch;
    }
  });
});
