#!/usr/bin/env node
// wasmtune — generic local fine-tune CLI for any website folder.
//   wasmtune init|check|models|dataset|train|eval|convert|serve|build
// Zero runtime deps; Python/torch live behind the auto-venv in train/.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(here, "..");
const { loadConfig, defaultConfig, validateConfig, resolveConfig } = await import("../src/config.mjs");
const { formatModels, assertAllowedModel } = await import("../src/models.mjs");
const { buildDataset } = await import("../src/dataset/index.mjs");
const { resolveBackend, detectPlatform, trainerFor } = await import("../src/train/router.mjs");
const { ensureVenv, runTrainer, venvPaths } = await import("../src/train/venv.mjs");
const { convertModel } = await import("../src/convert/to_mlc.mjs");
const { serve } = await import("../src/serve.mjs");
const { runEval, formatEvalTable } = await import("../src/eval/index.mjs");

const USAGE = `Usage: wasmtune <command> [options]
  ("finetune" works as a backwards-compatible alias.)

Commands:
  init [--config <path>] [--dataDir <dir>] [--model <hf-id>] [--method sft|dpo|orpo|grpo]
  check [--config <path>] [--allow-large] [--skip-serve]
  models
  dataset [--config <path>] [--synth provider:model]
  train [--config <path>] [--allow-large] [--backend auto|unsloth|mlx] [--max-steps N] [--dry-run] [--python <bin>]
  eval [--config <path>] [--backend auto|unsloth|mlx] [--max-prompts N] [--max-tokens N] [--skip-tuned] [--judge provider:model] [--python <bin>]
  convert [--config <path>]
  serve [--config <path>] [--port N]
  build [--config <path>] [train/eval/dataset flags...] — dataset→train→eval→convert; eval gate fails the build

Config: finetune.config.json (or .js/.mjs). See templates/wasmtune.config.example.json.
Config-free runs: pass --dataDir <dir> --model <hf-id> instead of a file, plus
any of --method/--backend/--epochs/--lr/--batch-size/--quant/--out-dir/--web-dir.
With a file present those same flags override it. Precedence: flags > file > defaults.
Docs: https://github.com/warsang/wasmTune`;

function parse(argv) {
  const out = { cmd: argv[2] ?? null, opts: {} };
  for (let i = 3; i < argv.length; i++) {
    const a = argv[i];
    const eq = a.indexOf("=");
    const flag = eq === -1 ? a : a.slice(0, eq);
    const inline = eq === -1 ? null : a.slice(eq + 1);
    const take = () => (inline !== null ? inline : (argv[++i] ?? ""));
    if (flag === "--config") out.opts.config = take();
    else if (flag === "--dataDir") out.opts.dataDir = take();
    else if (flag === "--model") out.opts.model = take();
    else if (flag === "--method") out.opts.method = take();
    else if (flag === "--backend") out.opts.backend = take();
    else if (flag === "--port") out.opts.port = Number(take()) || 8080;
    else if (flag === "--epochs") out.opts.epochs = Number(take());
    else if (flag === "--lr") out.opts.lr = Number(take());
    else if (flag === "--batch-size") out.opts.batchSize = Number(take());
    else if (flag === "--quant") out.opts.quant = take();
    else if (flag === "--out-dir") out.opts.outDir = take();
    else if (flag === "--web-dir") out.opts.webDir = take();
    else if (flag === "--max-steps") out.opts.maxSteps = Number(take()) || 0;
    else if (flag === "--max-prompts") out.opts.maxPrompts = Number(take()) || 50;
    else if (flag === "--max-tokens") out.opts.maxTokens = Number(take()) || 128;
    else if (flag === "--synth") out.opts.synth = take();
    else if (flag === "--judge") out.opts.judge = take();
    else if (flag === "--python") out.opts.python = take();
    else if (flag === "--allow-large" || flag === "--dry-run" || flag === "--skip-tuned" || flag === "--skip-serve") out.opts[flag.slice(2)] = true;
    else if (a === "--help" || a === "-h") {
      console.error(USAGE);
      process.exit(0);
    } else {
      console.error(`unknown flag "${a}"\n${USAGE}`);
      process.exit(2);
    }
  }
  return out;
}

async function cmdInit(opts, cwd) {
  const target = path.resolve(cwd, opts.config ?? "wasmtune.config.json");
  if (existsSync(target)) throw new Error(`refusing to overwrite existing ${target}`);
  const cfg = defaultConfig();
  if (opts.dataDir) cfg.dataDir = opts.dataDir;
  if (opts.model) cfg.model = opts.model;
  if (opts.method) cfg.method = opts.method;
  await writeFile(target, JSON.stringify(cfg, null, 2) + "\n");
  console.error(`wrote ${target}\nedit dataDir/model/method, then run: wasmtune check`);
}

async function cmdCheck(opts, cwd) {
  const { path: p, config } = await resolveConfig(opts, cwd);
  const errors = validateConfig(config, { cwd, allowLarge: !!opts["allow-large"] });
  if (errors.length) throw new Error(`invalid ${p}:\n- ${errors.join("\n- ")}`);
  const plat = await detectPlatform();
  let backend;
  try {
    backend = await resolveBackend(config.training.backend, {});
  } catch (e) {
    backend = `unavailable (${e.message.split(".")[0]})`;
  }
  const out = { config: p, model: config.model, method: config.method, platform: plat, backend };
  if (!opts["skip-serve"]) {
    const { checkServing } = await import("../src/serve-check.mjs");
    out.serving = await checkServing({ cwd, config });
  }
  console.log(JSON.stringify(out, null, 2));
}

async function cmdDataset(opts, cwd) {
  const { config } = await resolveConfig(opts, cwd);
  const report = await buildDataset(config, { cwd, synth: opts.synth ?? null });
  console.error(`dataset: ${report.docs} docs -> ${report.chunks} chunks -> ${report.sftPairs} sft pairs (~${report.approxTokens} tokens)`);
  console.error(`wrote ${report.files.sft}\n      ${report.files.dpo}\n      ${report.files.grpo}`);
}

async function cmdTrain(opts, cwd) {
  const { config } = await resolveConfig(opts, cwd);
  assertAllowedModel(config.model, { allowLarge: !!opts["allow-large"] });
  const backend = await resolveBackend(opts.backend ?? config.training.backend, {});
  const outDir = path.resolve(cwd, config.output.dir);
  await mkdir(outDir, { recursive: true });
  console.error(`[wasmtune] backend=${backend} model=${config.model} method=${config.method}`);
  console.error(`[wasmtune] trainer=${trainerFor(backend)}`);

  // Ensure dataset exists (cheap, deterministic) unless user points elsewhere.
  const sftFile = path.join(outDir, "dataset.sft.jsonl");
  if (!existsSync(sftFile)) {
    console.error("[wasmtune] no dataset found, building first…");
    await buildDataset(config, { cwd });
  }
  await ensureVenv({ outDir, backend, cwd, python: opts.python ?? "python3" });

  const argsJson = path.join(outDir, "train.args.json");
  const payload = {
    model: config.training.mlxModel ?? config.model,
    method: config.method,
    mlxImpl: config.training.mlxImpl ?? "auto",
    outDir: path.join(outDir, "run"),
    sftFile: path.join(outDir, "dataset.sft.jsonl"),
    dpoFile: path.resolve(cwd, config.dpo.pairsFile),
    grpoFile: path.join(outDir, "dataset.grpo.jsonl"),
    lora: config.training.lora,
    quant4: (config.training.quantization ?? "q4") === "q4",
    epochs: config.training.epochs,
    batchSize: config.training.batchSize,
    gradAccum: config.training.gradAccum,
    lr: config.training.lr,
    maxSeqLen: config.training.maxSeqLen,
    numLayers: config.training.numLayers ?? 16,
    gradCheckpoint: config.training.gradCheckpoint ?? false,
    seed: config.training.seed,
    maxSteps: opts.maxSteps ?? config.training.maxSteps ?? 0,
    dryRun: !!opts["dry-run"],
    beta: config.dpo.beta,
    numGenerations: config.grpo.numGenerations,
    rewardPy: null,
  };
  // DPO/ORPO merges three sources (deduped by prompt+rejected): the user's pairsFile,
  // the auto seeds from `wasmtune dataset`, and loop contrasts mined by
  // `wasmtune eval` — the self-improvement loop for regurgitation spirals.
  if (config.method === "dpo" || config.method === "orpo") {
    const { loadDpoPairs } = await import("../src/dataset/dpo.mjs");
    const seen = new Set();
    const merged = [];
    const sources = [
      existsSync(payload.dpoFile) ? payload.dpoFile : null,
      path.join(outDir, "dataset.dpo.jsonl"),
      path.join(outDir, "eval.dpo-loops.jsonl"),
    ];
    for (const f of sources) {
      if (!f || !existsSync(f)) continue;
      for (const row of await loadDpoPairs(f, cwd)) {
        // Key on prompt + FULL rejected text: the same question with
        // different rejected variants (concise vs faithful vs loop) teaches
        // complementary shaping signals, not duplicates.
        const key = String(row.prompt ?? "").trim().toLowerCase()
          + "\n@@@\n" + String(row.rejected ?? "").trim().toLowerCase();
        if (!row.prompt || seen.has(key)) continue;
        seen.add(key);
        merged.push(row);
      }
    }
    if (!merged.length) throw new Error("no DPO pairs found (run `wasmtune dataset` / `wasmtune eval` or set dpo.pairsFile)");
    payload.dpoFile = path.join(outDir, "train.dpo.jsonl");
    await writeFile(payload.dpoFile, merged.map((r) => JSON.stringify(r)).join("\n") + "\n");
    console.error(`[wasmtune] dpo: ${merged.length} triplets merged from ${sources.filter((f) => f && existsSync(f)).length} source(s)`);
  }
  await writeFile(argsJson, JSON.stringify(payload, null, 2));
  await runTrainer({ outDir, backend, trainerArgs: ["--args-json", argsJson], cwd });
  const { venv } = venvPaths(outDir);
  console.error(`[wasmtune] done. adapters: ${path.join(outDir, "run", "adapters")} (venv: ${venv})`);
  console.error(`[wasmtune] next: finetune convert`);
}

async function cmdEval(opts, cwd) {
  const { config } = await resolveConfig(opts, cwd);
  const backend = await resolveBackend(opts.backend ?? config.training.backend, {});
  const { reportPath, report } = await runEval(config, {
    cwd,
    backend,
    maxPrompts: opts.maxPrompts ?? 50,
    maxTokens: opts.maxTokens ?? 128,
    skipTuned: !!opts["skip-tuned"],
    judge: opts.judge ?? null,
    python: opts.python ?? "python3",
  });
  console.error(formatEvalTable(report));
  console.error(`wrote ${reportPath}`);
  if (!report.gate) {
    throw new Error("eval gate failed: tuned model regressed vs base (set eval.failOnRegression=false to allow)");
  }
}

async function cmdConvert(opts, cwd) {
  const { config } = await resolveConfig(opts, cwd);
  const mergedDir = path.join(path.resolve(cwd, config.output.dir), "run", "merged");
  const webDir = path.resolve(cwd, config.output.webDir);
  const { manifestPath, manifest } = await convertModel({ mergedDir, webDir, modelId: config.model, chat: config.chat ?? {} });
  console.error(`wrote ${manifestPath}`);
  for (const n of manifest.notes) console.error(`note: ${n}`);
}

async function cmdBuild(opts, cwd) {
  // One-shot pipeline for prebuild hooks: dataset -> train -> eval -> convert.
  // The eval gate throws on regression, failing the build so no bad model ships.
  console.error("[wasmtune] build: dataset");
  await cmdDataset(opts, cwd);
  console.error("[wasmtune] build: train");
  await cmdTrain(opts, cwd);
  console.error("[wasmtune] build: eval");
  await cmdEval(opts, cwd);
  console.error("[wasmtune] build: convert");
  await cmdConvert(opts, cwd);
  console.error("[wasmtune] build: done — model manifest ready for deploy");
}

async function cmdServe(opts, cwd) {
  const { config } = await loadConfig(cwd, opts.config ?? null).catch(() => ({ config: null }));
  const port = opts.port ?? 8080;
  const webDir = config ? path.resolve(cwd, config.output.webDir) : null;
  const { url } = await serve({ root: cwd, port, webDir });
  console.error(`serving ${cwd}${webDir ? ` + models ${webDir}` : ""} at ${url} (Ctrl+C to stop)`);
  await new Promise(() => {});
}

async function main() {
  const { cmd, opts } = parse(process.argv);
  const cwd = process.cwd();
  if (!cmd || cmd.startsWith("-")) {
    console.error(USAGE);
    process.exit(cmd && (cmd === "--help" || cmd === "-h") ? 0 : 2);
  }
  if (cmd === "init") return cmdInit(opts, cwd);
  if (cmd === "check") return cmdCheck(opts, cwd);
  if (cmd === "models") {
    console.log(formatModels());
    return;
  }
  if (cmd === "dataset") return cmdDataset(opts, cwd);
  if (cmd === "train") return cmdTrain(opts, cwd);
  if (cmd === "eval") return cmdEval(opts, cwd);
  if (cmd === "convert") return cmdConvert(opts, cwd);
  if (cmd === "build") return cmdBuild(opts, cwd);
  if (cmd === "serve") return cmdServe(opts, cwd);
  console.error(`unknown command "${cmd}"\n${USAGE}`);
  process.exit(2);
}

main().catch((e) => {
  console.error(`ERROR: ${e.message}`);
  process.exit(1);
});
