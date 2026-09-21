// wasmtune — config load + validate. Generic: no site-specific defaults.

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { assertAllowedModel } from "./models.mjs";

export const CONFIG_NAMES = [
  "wasmtune.config.json",
  "wasmtune.config.js",
  "wasmtune.config.mjs",
  "finetune.config.json",
  "finetune.config.js",
  "finetune.config.mjs",
];

export const VALID_METHODS = new Set(["sft", "dpo", "orpo", "grpo"]);
export const VALID_BACKENDS = new Set(["auto", "unsloth", "mlx"]);

export function defaultConfig() {
  return {
    dataDir: "./docs",
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    method: "sft",
    training: {
      backend: "auto",
      lora: { r: 16, alpha: 16, dropout: 0.05, targetModules: "auto" },
      quantization: "q4",
      epochs: 2,
      batchSize: 2,
      gradAccum: 4,
      lr: 0.0002,
      maxSeqLen: 2048,
      seed: 42,
      maxSteps: 0,
    },
    dpo: { pairsFile: "./.finetune/dpo_pairs.jsonl", beta: 0.1 },
    grpo: { rewardFile: "./rewards.mjs", numGenerations: 4 },
    output: { dir: "./.finetune", webDir: "./public/models" },
  };
}

export async function findConfig(cwd = process.cwd(), explicit = null) {
  if (explicit) return path.resolve(cwd, explicit);
  for (const name of CONFIG_NAMES) {
    const p = path.resolve(cwd, name);
    if (existsSync(p)) return p;
  }
  return null;
}

export async function loadConfig(cwd = process.cwd(), explicit = null) {
  const found = await findConfig(cwd, explicit);
  if (!found) {
    throw new Error(
      `no config found (looked for ${CONFIG_NAMES.join(", ")}). Run "wasmtune init" first.`,
    );
  }
  let raw;
  if (found.endsWith(".json")) {
    raw = JSON.parse(await readFile(found, "utf8"));
  } else {
    const mod = await import(pathToFileURL(found).href);
    raw = mod.default ?? mod.config ?? mod;
  }
  const cfg = normalizeConfig(raw);
  const errors = validateConfig(cfg, { cwd });
  if (errors.length) throw new Error(`invalid config ${found}:\n- ${errors.join("\n- ")}`);
  return { path: found, config: cfg };
}

// Build a config purely from CLI flags (no file). Only the essential knobs
// are flaggable; exotic settings (LoRA alpha, GRPO rewards, judge providers)
// still need a file. Throws if the two required knobs are missing.
export function configFromFlags(opts = {}, cwd = process.cwd()) {
  const o = opts ?? {};
  const missing = [];
  if (!o.dataDir) missing.push("--dataDir <dir>");
  if (!o.model) missing.push("--model <hf-id>");
  if (missing.length) {
    throw new Error(
      `no config file found and required flags missing: ${missing.join(", ")}.\n` +
        `Either run "wasmtune init" or pass --dataDir and --model.`,
    );
  }
  const raw = { dataDir: o.dataDir, model: o.model };
  if (o.method) raw.method = o.method;
  const training = {};
  if (o.backend) training.backend = o.backend;
  if (o.epochs !== undefined) training.epochs = Number(o.epochs);
  if (o.lr !== undefined) training.lr = Number(o.lr);
  if (o.batchSize !== undefined) training.batchSize = Number(o.batchSize);
  if (o.quant) training.quantization = o.quant;
  if (Object.keys(training).length) raw.training = training;
  const output = {};
  if (o.outDir) output.dir = o.outDir;
  if (o.webDir) output.webDir = o.webDir;
  if (Object.keys(output).length) raw.output = output;
  const cfg = normalizeConfig(raw);
  const errors = validateConfig(cfg, { cwd, allowLarge: !!o.allowLarge });
  if (errors.length) throw new Error(`invalid flag-built config:\n- ${errors.join("\n- ")}`);
  return { path: "<flags>", config: cfg };
}

// Resolve configuration with precedence: explicit --config file >
// auto-discovered file > CLI flags. Merges flag overrides for the essential
// knobs on top of any file: flags > file > defaults.
export async function resolveConfig(opts = {}, cwd = process.cwd()) {
  const o = opts ?? {};
  const found = await findConfig(cwd, o.config ?? null);
  if (!found) return configFromFlags(o, cwd);
  const { path: p, config: fileCfg } = await loadConfig(cwd, o.config ?? null);
  const overrides = {};
  if (o.dataDir) overrides.dataDir = o.dataDir;
  if (o.model) overrides.model = o.model;
  if (o.method) overrides.method = o.method;
  const training = {};
  if (o.backend) training.backend = o.backend;
  if (o.epochs !== undefined) training.epochs = Number(o.epochs);
  if (o.lr !== undefined) training.lr = Number(o.lr);
  if (o.batchSize !== undefined) training.batchSize = Number(o.batchSize);
  if (o.quant) training.quantization = o.quant;
  if (Object.keys(training).length) {
    overrides.training = { ...fileCfg.training, ...training };
  }
  const output = {};
  if (o.outDir) output.dir = o.outDir;
  if (o.webDir) output.webDir = o.webDir;
  if (Object.keys(output).length) {
    overrides.output = { ...fileCfg.output, ...output };
  }
  if (!Object.keys(overrides).length) return { path: p, config: fileCfg };
  const merged = normalizeConfig({ ...fileCfg, ...overrides });
  const errors = validateConfig(merged, { cwd, allowLarge: !!o.allowLarge });
  if (errors.length) throw new Error(`invalid merged config (${p} + flags):\n- ${errors.join("\n- ")}`);
  return { path: `${p} + flags`, config: merged };
}

export function normalizeConfig(raw = {}) {
  const d = defaultConfig();
  return {
    dataDir: raw.dataDir ?? d.dataDir,
    model: raw.model ?? d.model,
    method: String(raw.method ?? d.method).toLowerCase(),
    training: {
      ...d.training,
      ...(raw.training ?? {}),
      lora: { ...d.training.lora, ...(raw.training?.lora ?? {}) },
    },
    dpo: { ...d.dpo, ...(raw.dpo ?? {}) },
    grpo: { ...d.grpo, ...(raw.grpo ?? {}) },
    dataset: { ...(raw.dataset ?? {}) },
    eval: { failOnRegression: true, ...(raw.eval ?? {}) },
    chat: { ...(raw.chat ?? {}) },
    output: { ...d.output, ...(raw.output ?? {}) },
  };
}

export function validateConfig(cfg, { cwd = process.cwd(), allowLarge = false } = {}) {
  const errors = [];
  const dirs = Array.isArray(cfg.dataDir) ? cfg.dataDir : [cfg.dataDir];
  if (!dirs.length || !dirs.every((d) => typeof d === "string" && d)) {
    errors.push("dataDir must be a folder path string or array of folder paths");
  } else {
    for (const d of dirs) {
      if (!existsSync(path.resolve(cwd, d))) {
        errors.push(`dataDir not found: ${d} (resolved from ${cwd})`);
      }
    }
  }
  if (!cfg.model || typeof cfg.model !== "string") {
    errors.push("model must be a Hugging Face id string");
  } else {
    try {
      assertAllowedModel(cfg.model, { allowLarge });
    } catch (e) {
      errors.push(e.message.split("\n")[0]);
    }
  }
  if (!VALID_METHODS.has(cfg.method)) {
    errors.push(`method must be one of ${[...VALID_METHODS].join("|")}`);
  }
  if (!VALID_BACKENDS.has(cfg.training?.backend)) {
    errors.push(`training.backend must be one of ${[...VALID_BACKENDS].join("|")}`);
  }
  const t = cfg.training ?? {};
  for (const k of ["epochs", "batchSize", "gradAccum", "maxSeqLen", "seed"]) {
    if (t[k] !== undefined && !(Number.isFinite(Number(t[k])) && Number(t[k]) >= 0)) {
      errors.push(`training.${k} must be a non-negative number`);
    }
  }
  if (cfg.method === "dpo" && !cfg.dpo?.pairsFile) errors.push("dpo.pairsFile is required for method=dpo");
  if (cfg.method === "grpo" && !cfg.grpo?.rewardFile) errors.push("grpo.rewardFile is required for method=grpo");
  return errors;
}
