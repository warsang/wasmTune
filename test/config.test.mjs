import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { normalizeConfig, validateConfig, configFromFlags, resolveConfig, configModels } from "../src/config.mjs";

describe("config", () => {
  it("accepts a minimal valid config shape", () => {
    const cfg = normalizeConfig({ dataDir: ".", model: "Qwen/Qwen2.5-1.5B-Instruct", method: "sft" });
    const errors = validateConfig(cfg, { cwd: new URL("../../", import.meta.url).pathname, allowLarge: false });
    // dataDir "." exists relative to package dir; only model/method shape asserted here
    assert.ok(!errors.some((e) => e.includes("method")));
    assert.ok(!errors.some((e) => e.includes("allowlist")));
  });

  it("accepts a multi-model tier list and derives the primary model", () => {
    const cfg = normalizeConfig({
      dataDir: ".",
      models: [
        { model: "google/gemma-4-E4B-it", trained: true },
        { model: "Qwen/Qwen3-0.6B", trained: false },
      ],
      method: "sft",
    });
    assert.equal(cfg.model, "google/gemma-4-E4B-it"); // models[0]
    const errors = validateConfig(cfg, { cwd: new URL("../../", import.meta.url).pathname, allowLarge: false });
    assert.deepEqual(errors, []);
    const entries = configModels(cfg);
    assert.equal(entries.length, 2);
    assert.equal(entries[0].trained, true);
    assert.equal(entries[1].trained, false);
  });

  it("rejects duplicate or non-allowlisted tier entries", () => {
    const cfg = normalizeConfig({
      dataDir: ".",
      models: [
        { model: "Qwen/Qwen3-0.6B" },
        { model: "qwen/qwen3-0.6B" },
        { model: "meta-llama/Llama-3.1-405B" },
      ],
    });
    const errors = validateConfig(cfg, { cwd: "/", allowLarge: false });
    assert.ok(errors.some((e) => e.includes("duplicate")), JSON.stringify(errors));
    assert.ok(errors.some((e) => e.includes("models[2]")), JSON.stringify(errors));
  });

  it("configModels merges entry chat over config chat", () => {
    const cfg = normalizeConfig({
      dataDir: ".", model: "Qwen/Qwen3-0.6B",
      chat: { temperature: 0.4 },
      models: [
        { model: "google/gemma-4-E4B-it", chat: { temperature: 0.2 } },
        { model: "Qwen/Qwen3-0.6B", trained: false },
      ],
    });
    const entries = configModels(cfg);
    assert.equal(entries[0].chat.temperature, 0.2); // entry wins
    assert.equal(entries[1].chat.temperature, 0.4); // config default survives
  });

  it("rejects unknown models without --allow-large", () => {
    const cfg = normalizeConfig({ dataDir: ".", model: "meta-llama/Llama-3.1-405B", method: "sft" });
    const errors = validateConfig(cfg, { cwd: "/", allowLarge: false });
    assert.ok(errors.some((e) => e.includes("allowlist")));
  });

  it("rejects bad method/backend", () => {
    const cfg = normalizeConfig({ dataDir: ".", model: "Qwen/Qwen2.5-1.5B-Instruct", method: "ppo" });
    cfg.training.backend = "tpu";
    const errors = validateConfig(cfg, { cwd: "/", allowLarge: true });
    assert.ok(errors.some((e) => e.includes("method")));
    assert.ok(errors.some((e) => e.includes("backend")));
  });
});

describe("configFromFlags", () => {
  const MODEL = "Qwen/Qwen2.5-1.5B-Instruct";

  it("builds a valid config from required flags only", () => {
    const { path: p, config } = configFromFlags({ dataDir: ".", model: MODEL }, "/");
    assert.equal(p, "<flags>");
    assert.equal(config.dataDir, ".");
    assert.equal(config.model, MODEL);
    assert.equal(config.method, "sft"); // default
    assert.equal(config.training.backend, "auto"); // default
  });

  it("applies essential-knob overrides", () => {
    const { config } = configFromFlags({
      dataDir: ".", model: MODEL, method: "dpo", backend: "mlx",
      epochs: 5, lr: 0.0001, batchSize: 4, quant: "q8",
      outDir: "./custom-ft", webDir: "./static/models",
    }, "/");
    assert.equal(config.method, "dpo");
    assert.equal(config.training.backend, "mlx");
    assert.equal(config.training.epochs, 5);
    assert.equal(config.training.lr, 0.0001);
    assert.equal(config.training.batchSize, 4);
    assert.equal(config.training.quantization, "q8");
    assert.equal(config.output.dir, "./custom-ft");
    assert.equal(config.output.webDir, "./static/models");
  });

  it("throws when required flags are missing", () => {
    assert.throws(() => configFromFlags({}, "/"), /--dataDir.*--model|--model.*--dataDir/);
    assert.throws(() => configFromFlags({ dataDir: "." }, "/"), /--model/);
    assert.throws(() => configFromFlags({ model: MODEL }, "/"), /--dataDir/);
  });

  it("validates flag values (bad model, bad numbers)", () => {
    assert.throws(
      () => configFromFlags({ dataDir: ".", model: "meta-llama/Llama-3.1-405B" }, "/"),
      /allowlist/,
    );
    assert.throws(
      () => configFromFlags({ dataDir: ".", model: MODEL, epochs: -2 }, "/"),
      /training\.epochs/,
    );
  });
});

describe("resolveConfig", () => {
  const MODEL = "Qwen/Qwen2.5-1.5B-Instruct";

  async function tmpSite(files = {}) {
    const dir = await mkdtemp(path.join(tmpdir(), "sft-cfg-"));
    await mkdir(path.join(dir, "docs"), { recursive: true });
    await writeFile(path.join(dir, "docs", "a.md"), "# Hi\n\nhello world ".repeat(20));
    for (const [name, content] of Object.entries(files)) {
      await writeFile(path.join(dir, name), content);
    }
    return dir;
  }

  it("uses flags when no file exists", async () => {
    const dir = await tmpSite();
    const { path: p, config } = await resolveConfig({ dataDir: "./docs", model: MODEL }, dir);
    assert.equal(p, "<flags>");
    assert.equal(config.training.epochs, 2);
  });

  it("prefers an explicit --config file", async () => {
    const dir = await tmpSite({
      "custom.json": JSON.stringify({ dataDir: "./docs", model: MODEL, method: "orpo" }),
    });
    const { path: p, config } = await resolveConfig({ config: "custom.json" }, dir);
    assert.ok(p.endsWith("custom.json"));
    assert.equal(config.method, "orpo");
  });

  it("merges flags over file values", async () => {
    const dir = await tmpSite({
      "finetune.config.json": JSON.stringify({
        dataDir: "./docs", model: MODEL, method: "sft",
        training: { epochs: 2, lr: 0.0002 },
      }),
    });
    const { path: p, config } = await resolveConfig(
      { epochs: 9, lr: 0.00001, outDir: "./flag-out" }, dir,
    );
    assert.ok(p.includes("+ flags"));
    assert.equal(config.method, "sft"); // untouched file value survives
    assert.equal(config.training.epochs, 9); // flag wins
    assert.equal(config.training.lr, 0.00001); // flag wins
    assert.equal(config.training.backend, "auto"); // file default survives
    assert.equal(config.output.dir, "./flag-out"); // flag wins
  });

  it("returns the file untouched when no flag overrides given", async () => {
    const dir = await tmpSite({
      "finetune.config.json": JSON.stringify({ dataDir: "./docs", model: MODEL }),
    });
    const { path: p, config } = await resolveConfig({}, dir);
    assert.ok(p.endsWith("finetune.config.json"));
    assert.ok(!p.includes("flags"));
    assert.equal(config.method, "sft");
  });

  it("rejects invalid merged results", async () => {
    const dir = await tmpSite({
      "finetune.config.json": JSON.stringify({ dataDir: "./docs", model: MODEL }),
    });
    await assert.rejects(
      resolveConfig({ model: "meta-llama/Llama-3.1-405B" }, dir),
      /allowlist/,
    );
  });
});
