#!/usr/bin/env node
// wasmtune — generic local fine-tune CLI for any website folder.
//   wasmtune init|check|models|dataset|train|eval|convert|serve|build
// Zero runtime deps; Python/torch live behind the auto-venv in train/.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const execFileAsync = promisify(execFile);

// Run node with an argv array and no shell. execFile is wrong for this on
// Windows: it concatenates argv into one command line without quoting, so any
// argument containing a space (every --system-prompt) gets split and the child
// dies with an error that looks like a bug in the script.
function spawnNode(args, { cwd = process.cwd() } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { cwd, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`node ${args[0]} exited ${code}`)));
  });
}

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(here, "..");
const { loadConfig, defaultConfig, validateConfig, resolveConfig, configModels } = await import("../src/config.mjs");
const { formatModels, assertAllowedModel, lookupModel, slugifyModel } = await import("../src/models.mjs");
const { buildDataset } = await import("../src/dataset/index.mjs");
const { resolveBackend, detectPlatform, trainerFor } = await import("../src/train/router.mjs");
const { ensureVenv, runTrainer, runPython, venvPaths } = await import("../src/train/venv.mjs");

// A HF dir counts as usable only once it actually contains weights. The
// in-trainer merge writes config.json first and can then fail on save, leaving
// a directory that looks populated but has nothing to quantize.
function hasWeights(dir) {
  if (!existsSync(dir)) return false;
  try {
    return readdirSync(dir).some(
      (f) => /\.(safetensors|bin|pt|gguf)$/i.test(f) || f.endsWith(".safetensors.index.json"),
    );
  } catch {
    return false;
  }
}
const { convertModel, pretrainedEntry, writeModelsManifest } = await import("../src/convert/to_mlc.mjs");
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
  eval-onnx [--config <path>] [--dtype q4|q8] [--max-tokens N] [--python <bin>] [--onnx-dir <dir>] [--tokenizer-dir <dir>] [--force]
  convert [--config <path>]
  publish [--config <path>] --repo <user>/<name> [--token <hf-token>] [--private]
  serve [--config <path>] [--port N]
  build [--config <path>] [train/eval/dataset flags...] — dataset→train→eval→convert; eval gate fails the build

Config: wasmtune.config.json (or .js/.mjs). See templates/wasmtune.config.example.json.
Config-free runs: pass --dataDir <dir> --model <hf-id> instead of a file, plus
any of --method/--backend/--epochs/--lr/--batch-size/--quant/--out-dir/--web-dir.
With a file present those same flags override it. Precedence: flags > file > defaults.
Multi-model tiers: add "models": [{ model, trained?, gguf?, label?, chat?, requirements? }]
to the config. train/eval run per trained entry (.finetune/<slug>/); convert writes
one manifest with every tier, and the browser serves the best tier its hardware can run.
Docs: https://github.com/warsang/wasmTune

Publishing: weights are 100 MB - 1 GB, which no static host will take, so
"wasmtune publish" uploads the converted ONNX graph + tokenizer to the Hub and
prints the manifest line that points at it. It needs HF_TOKEN (or --token) and
is the only command that touches the network or a credential — dataset, train,
eval and convert all stay offline and work without an account. It is
intentionally not part of "build", so a build hook stays reproducible.`;

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
    else if (flag === "--onnx-dir") out.opts.onnxDir = take();
    else if (flag === "--tokenizer-dir") out.opts.tokenizerDir = take();
    else if (flag === "--force") out.opts.force = true;
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

// Per-entry working config: multi-entry builds isolate artifacts under
// .finetune/<slug>/; single-entry configs keep the legacy .finetune/ layout.
function entryConfig(config, entry, multi) {
  const relOut = multi ? path.join(config.output.dir, slugifyModel(entry.model)) : config.output.dir;
  const pinnedMlx = config.training.mlxModel && entry.model === config.model ? config.training.mlxModel : null;
  const mlxModel = pinnedMlx ?? lookupModel(entry.model)?.mlx ?? config.training.mlxModel ?? entry.model;
  return {
    sub: {
      ...config,
      model: entry.model,
      output: { ...config.output, dir: relOut },
      training: { ...config.training, mlxModel },
      chat: entry.chat ?? config.chat ?? {},
    },
    relOut,
  };
}

// Trained entries only; pretrained tiers are served straight from their
// public browser artifacts and never enter the training pipeline.
function trainedEntries(config) {
  return configModels(config).filter((e) => e.trained);
}

async function trainOne(config, outDir, opts, cwd) {
  const entry = configModels(config)[0];
  assertAllowedModel(config.model, { allowLarge: !!opts["allow-large"] });
  const backend = await resolveBackend(opts.backend ?? config.training.backend, {});
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
    // Render training text with the same system prompt the widget will send.
    // Bases inject their own default system message when a conversation has
    // none, so training without this primes the model on text it will never be
    // given and omits the text it will.
    systemPrompt: (await import("../src/chat/options.mjs"))
      .defaultSystemPrompt(config.dataset.siteName ?? path.basename(config.dataDir)),
    maxSteps: opts.maxSteps ?? config.training.maxSteps ?? 0,
    dryRun: !!opts["dry-run"],
    beta: config.dpo.beta,
    numGenerations: config.grpo.numGenerations,
    rewardPy: await resolveRewardFile(config, outDir, cwd),
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
  void entry;
}

async function cmdTrain(opts, cwd) {
  const { config } = await resolveConfig(opts, cwd);
  const entries = trainedEntries(config);
  if (!entries.length) {
    console.error("[wasmtune] every configured model is pretrained — nothing to train. Run `wasmtune convert` to publish them.");
    return;
  }
  const multi = entries.length > 1;
  for (const entry of entries) {
    const { sub, relOut } = entryConfig(config, entry, multi);
    if (multi) console.error(`[wasmtune] tier ${entry.model} -> ${relOut}`);
    await trainOne(sub, path.resolve(cwd, relOut), opts, cwd);
    if (multi) console.error(`[wasmtune] next: wasmtune convert`);
  }
}

async function cmdEval(opts, cwd) {
  const { config } = await resolveConfig(opts, cwd);
  const entries = trainedEntries(config);
  if (!entries.length) {
    console.error("[wasmtune] every configured model is pretrained — nothing to evaluate.");
    return;
  }
  const multi = entries.length > 1;
  for (const entry of entries) {
    const { sub } = entryConfig(config, entry, multi);
    if (multi) console.error(`[wasmtune] eval tier ${entry.model}`);
    const backend = await resolveBackend(opts.backend ?? sub.training.backend, {});
    const { reportPath, report } = await runEval(sub, {
      cwd,
      backend,
      maxPrompts: opts.maxPrompts ?? 50,
      // Must match what eval-onnx uses (and what the widget ships), or the
      // quantization gate compares a 128-token fp16 baseline against a
      // 96-token artifact. Longer answers score higher on keyword recall, so
      // that asymmetry alone can move the delta by more than the threshold.
      maxTokens: opts.maxTokens ?? sub.chat?.maxTokens ?? 96,
      skipTuned: !!opts["skip-tuned"],
      judge: opts.judge ?? null,
      python: opts.python ?? "python3",
    });
    console.error(formatEvalTable(report));
    console.error(`wrote ${reportPath}`);
    if (!report.gate) {
      throw new Error(`eval gate failed: tuned model regressed vs base (${entry.model}; set eval.failOnRegression=false to allow)`);
    }
  }
}

async function cmdConvert(opts, cwd) {
  const { config } = await resolveConfig(opts, cwd);
  const entries = configModels(config);
  const webDir = path.resolve(cwd, config.output.webDir);
  const multi = entries.length > 1;
  const outEntries = [];
  const notes = [];
  for (const entry of entries) {
    const slug = slugifyModel(entry.model);
    if (entry.trained) {
      const { relOut } = entryConfig(config, entry, multi);
      const runDir = path.join(path.resolve(cwd, relOut), "run");
      const mergedDir = path.join(runDir, "merged");
      const adaptersDir = path.join(runDir, "adapters");
      if (!hasWeights(mergedDir)) {
        // The in-trainer merge is best-effort: merging LoRA into a 4-bit
        // bnb model and calling save_pretrained() breaks on transformers 5.x,
        // which leaves merged/ holding only config.json. export.py loads the
        // base at full precision instead and produces the same result, so
        // produce the weights here rather than skipping the tier.
        if (!existsSync(adaptersDir)) {
          notes.push(`${slug}: no adapters at ${adaptersDir} and no merged weights — run "wasmtune train" first (skipped)`);
          continue;
        }
        console.error(`[wasmtune] merging adapters -> HF dir (${entry.model}) ...`);
        try {
          await ensureVenv({ outDir: path.resolve(cwd, relOut), backend: "unsloth", cwd, python: opts.python ?? "python3" });
          await runPython({
            outDir: path.resolve(cwd, relOut), script: "python/export.py",
            args: ["--adapters", adaptersDir, "--base", entry.model, "--out", mergedDir],
            cwd,
          });
        } catch (err) {
          notes.push(`${slug}: merge failed (${err.message}) — skipped`);
          continue;
        }
        if (!hasWeights(mergedDir)) {
          notes.push(`${slug}: merge produced no weights in ${mergedDir} — skipped`);
          continue;
        }
      }
      console.error(`[wasmtune] converting tuned tier ${entry.model}`);
      const { entry: built, notes: n } = await convertModel({
        mergedDir, webDir, modelId: entry.model, chat: entry.chat,
        subdir: multi ? slug : null, label: entry.label,
        siteName: config.dataset?.siteName ?? path.basename(config.dataDir),
        requirements: entry.requirements, writeManifest: false,
        venvOutDir: path.resolve(cwd, config.output.dir),
      });
      outEntries.push(built);
      notes.push(...n.map((x) => `${slug}: ${x}`));
    } else {
      const m = lookupModel(entry.model);
      if (!m) throw new Error(`unknown pretrained model "${entry.model}" (not on the allowlist)`);
      // A custom URL is required for GGUF tiers; allowlist gguf values are
      // HF repo ids (not loadable URLs) and are skipped with a note.
      const gguf = entry.gguf && /^https?:\/\/.+\.gguf(\?.*)?$/i.test(entry.gguf) ? entry.gguf : null;
      if (entry.gguf && !gguf) notes.push(`${slug}: gguf "${entry.gguf}" is not a direct .gguf URL — ignored`);
      const webllm = entry.webllm ?? m.webllm ?? null;
      const onnx = entry.onnx ?? m.onnx ?? null;
      if (!gguf && !webllm && !onnx) {
        notes.push(`${slug}: no browser artifact available (allowlist has no webllm/onnx build and no direct gguf URL) — tier will fail the serving check`);
      }
      console.error(`[wasmtune] publishing pretrained tier ${entry.model} (${webllm ?? onnx ?? gguf ?? "no artifact"})`);
      outEntries.push(pretrainedEntry({
        model: entry.model, label: entry.label, gguf,
        onnx, webllm,
        chat: entry.chat, requirements: entry.requirements,
      }));
    }
  }
  if (!outEntries.length) throw new Error("no model entries produced — nothing to publish");
  const { manifestPath, manifest } = await writeModelsManifest({ webDir, entries: outEntries, notes });
  console.error(`wrote ${manifestPath}`);
  console.error(`[wasmtune] manifest: ${manifest.models.length} model tier(s): ${manifest.models.map((m) => m.id).join(", ")}`);
  for (const n of manifest.notes) console.error(`note: ${n}`);
}

async function cmdBuild(opts, cwd) {
  // One-shot pipeline for prebuild hooks: dataset -> train -> eval -> convert.
  // The eval gate throws on regression, failing the build so no bad model ships.
  // Publishing is deliberately NOT part of this: it needs a credential and a
  // remote, and a build hook must stay reproducible offline. Run
  // `wasmtune publish` as a separate release step.
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

// Always hand the trainer a concrete reward module, so the default and the
// user's override travel the same path. Writing it into outDir means a host can
// open the file the run actually used and edit it, rather than guessing what
// applied.
//
// Must be a .py: it is imported inside the python training venv. The old
// template shipped "./rewards.mjs", which importlib cannot load, so the
// override silently never applied and every GRPO run used the fallback reward.
async function resolveRewardFile(config, outDir, cwd) {
  const explicit = config.grpo?.rewardFile && path.resolve(cwd, config.grpo.rewardFile);
  if (explicit && existsSync(explicit)) {
    if (!explicit.endsWith(".py")) {
      throw new Error(
        `grpo.rewardFile must be a .py module (it is imported by the python ` +
        `trainer), got: ${config.grpo.rewardFile}`);
    }
    return explicit;
  }
  const src = path.resolve(cwd, "python/reward.py");
  if (!existsSync(src)) return null;
  const dest = path.join(outDir, "default_reward.py");
  await writeFile(dest, await readFile(src, "utf8"));
  return dest;
}
/**
 * Upload converted browser artifacts to the Hugging Face Hub.
 *
 * Optional by design: without a token this skips loudly and the rest of the
 * pipeline is unaffected. Weights for a usable chat model run 100 MB - 1 GB,
 * which no static host will take, so the Hub is the practical home for them —
 * but requiring an account to fine-tune locally would be a worse trade, so the
 * dependency lives in this one command.
 */
/**
 * Upload converted browser artifacts to the Hugging Face Hub.
 *
 * Optional by design: without a token this skips loudly and the rest of the
 * pipeline is unaffected. Weights for a usable chat model run 100 MB - 1 GB,
 * which no static host will take, so the Hub is the practical home for them —
 * but requiring an account to fine-tune locally would be a worse trade, so the
 * dependency lives in this one command.
 *
 * Also closes the loop the earlier version left open: it printed a manifest
 * line for the host to paste and verified nothing, so a publish that landed but
 * was never wired up looked exactly like never having run it. This writes the
 * manifest entry and compares the sha256 of the local graph to the repo's LFS
 * oid.
 */
async function cmdPublish(opts, cwd) {
  const {
    graphsIn, tokenFor, ensureRepo, uploadFile, writeModelCard,
    inlineChatTemplate, writeOnnxManifestEntry, verifyUploaded,
    sha256File, externalDataRefs,
  } = await import("../src/publish.mjs");

  // Check the credential before anything else: a user with no token should get
  // the skip message, not a config-resolution error.
  const token = tokenFor(opts.token ?? null);
  if (!token) {
    console.error(
      "[wasmtune] publish skipped: no Hugging Face token.\n" +
      "  Set HF_TOKEN (or pass --token), then re-run `wasmtune publish`.\n" +
      "  Everything else — dataset, train, eval, convert — works offline without one.",
    );
    return;
  }

  const { config } = await resolveConfig(opts, cwd);

  const repoId = opts.repo ?? config.publish?.repo ?? null;
  if (!repoId) {
    throw new Error(
      "publish needs a repo id: pass --repo <user>/<name> or set publish.repo in the config",
    );
  }

  const entries = configModels(config);
  const multi = entries.length > 1;
  const trained = entries.filter((e) => e.trained);
  if (!trained.length) {
    console.error("[wasmtune] publish: no trained entries — everything configured is pretrained, nothing to publish");
    return;
  }

  const siteName = siteNameOf(config, cwd);
  console.error(`[wasmtune] publishing to ${repoId}`);
  await ensureRepo({ repoId, token, privateRepo: !!opts.private });

  let uploaded = 0;
  let publishedDtype = null;
  const skipped = [];

  for (const entry of trained) {
    const slug = slugifyModel(entry.model);
    const { relOut } = entryConfig(config, entry, multi);
    const onnxDir = path.join(path.resolve(cwd, relOut), "onnx");
    const graphs = graphsIn(onnxDir);
    if (!graphs.length) {
      skipped.push(`${slug}: no onnx/ graphs at ${onnxDir} — run \`wasmtune convert\` with optimum-cli installed`);
      continue;
    }
    // Smallest first: it is the one transformers.js will fetch on a phone.
    const chosen = graphs[0];
    publishedDtype = chosen.dtype;
    console.error(`[wasmtune] ${slug}: ${chosen.file} (${(chosen.bytes / 1e6).toFixed(1)} MB, dtype ${chosen.dtype})`);

    const sub = multi ? `${slug}/` : "";
    const graphPath = path.join(onnxDir, chosen.file);
    const graphSha = sha256File(graphPath);

    // External data first, so the graph never references a file that is not
    // there yet mid-upload.
    for (const ref of externalDataRefs(graphPath)) {
      const local = path.join(onnxDir, ref);
      if (!existsSync(local)) {
        console.error(`[wasmtune] WARNING: graph references ${ref} but it is not in ${onnxDir}`);
        continue;
      }
      await uploadFile({ repoId, token, localPath: local, pathInRepo: `${sub}onnx/${ref}` });
      uploaded++;
    }
    await uploadFile({ repoId, token, localPath: graphPath, pathInRepo: `${sub}onnx/${chosen.file}` });
    uploaded++;

    // The graph is useless without its tokenizer.
    const mergedDir = path.join(path.resolve(cwd, relOut), "merged");
    for (const f of ["config.json", "generation_config.json", "tokenizer.json",
                     "special_tokens_map.json", "vocab.json", "merges.txt"]) {
      const local = path.join(mergedDir, f);
      if (existsSync(local)) {
        await uploadFile({ repoId, token, localPath: local, pathInRepo: `${sub}${f}` });
        uploaded++;
      }
    }
    // transformers.js reads the chat template from tokenizer_config.json and
    // ignores the sibling .jinja, so the inlined copy wins and goes last.
    const inlined = inlineChatTemplate({ dir: mergedDir });
    if (inlined) {
      await uploadFile({ repoId, token, localPath: inlined, pathInRepo: `${sub}tokenizer_config.json` });
      uploaded++;
      console.error(`[wasmtune] ${slug}: inlined chat_template into tokenizer_config.json`);
    }

    // Verify the bytes that landed are the bytes convert produced. A no-op'd
    // upload is otherwise indistinguishable from a successful one, and the whole
    // point of publishing is that what the widget loads is what was trained here.
    const check = await verifyUploaded({
      repoId, token, localPath: graphPath,
      pathInRepo: `${sub}onnx/${chosen.file}`, sha256: graphSha,
    }).catch((e) => ({ ok: null, error: String(e?.message ?? e) }));
    if (check.ok === true) {
      console.error(`[wasmtune] verified ${chosen.file} in ${repoId} (sha256 ${graphSha.slice(0, 12)}…)`);
    } else if (check.ok === false) {
      console.error(`[wasmtune] WARNING: ${chosen.file} in ${repoId} does not match the local graph — the upload may not have landed`);
    } else if (check.error) {
      console.error(`[wasmtune] could not verify ${chosen.file}: ${check.error}`);
    }
  }

  if (!uploaded) {
    console.error("[wasmtune] publish: nothing to upload");
    for (const s of skipped) console.error(`  note: ${s}`);
    return;
  }

  const card = await writeModelCard({
    dir: cwd, siteName, baseModel: config.model,
    dtype: publishedDtype, repoId,
    evalReport: readEvalReport(cwd, config, multi, trained[0]),
  }).catch(() => null);
  if (card) {
    await uploadFile({ repoId, token, localPath: card, pathInRepo: "README.md" });
    uploaded++;
  }

  console.error(`[wasmtune] published ${uploaded} file(s) -> https://huggingface.co/${repoId}`);

  // Close the loop: write the manifest entry rather than asking the host to.
  const manifestRel = config.output?.webDir ?? "public/models";
  const manifestPath = path.join(cwd, manifestRel, "model-manifest.json");
  const { entry: written } = writeOnnxManifestEntry({
    manifestPath, repoId, dtype: publishedDtype, siteName,
    label: trained[0].label ?? null, base: config.model, chat: config.chat ?? null,
  });
  console.error(`[wasmtune] wrote ${written.id} -> ${manifestPath}`);
  console.error("[wasmtune] the site loads it with: await mountAssistant({ target: \"#chat\" })");
  for (const s of skipped) console.error(`note: ${s}`);
}

function siteNameOf(config, cwd) {
  return config.dataset?.siteName ?? path.basename(path.resolve(cwd));
}

function readEvalReport(cwd, config, multi, entry) {
  if (!entry) return null;
  const p = path.join(path.resolve(cwd, entryConfig(config, entry, multi).relOut), "eval.report.json");
  if (!existsSync(p)) return null;
  try {
    const r = JSON.parse(readFileSync(p, "utf8"));
    return { base: r.base, tuned: r.tuned, delta: r.delta };
  } catch { return null; }
}

/**
 * Evaluate the artifact that ships, not the weights eval scored.
 *
 * `wasmtune eval` scores the merged fp16 output of `export.py`. What the widget
 * loads is the quantized ONNX graph, and those are different models: measured on
 * the same 39 holdout prompts, q8 costs ~30% of average score, ~58% of
 * generalization score and introduces repetition loops that fp16 did not have.
 * Nothing caught that before, because no gate looked at the artifact.
 *
 * Optional by design: without onnxruntime installed this loud-skips, so a host
 * that has not installed the export tooling is not blocked.
 */
async function cmdEvalOnnx(opts, cwd) {
  const config = (await resolveConfig(opts, cwd)).config;
  const entries = configModels(config);
  const trained = entries.filter((e) => e.trained);
  if (!trained.length) {
    console.error("[wasmtune] eval-onnx: no trained entries to evaluate");
    return;
  }

  const multi = entries.length > 1;
  const webDir = path.resolve(cwd, config.output.webDir);
  const graphDir = path.resolve(cwd, opts.onnxDir ?? path.join(webDir, "onnx"));
  if (!existsSync(graphDir)) {
    console.error(
      "[wasmtune] eval-onnx skipped: no onnx/ under " + graphDir +
      " — run `wasmtune convert` with optimum-cli installed first " +
      "(or pass --onnx-dir)");
    return;
  }

  // `eval` and `eval-onnx` must agree on where a tier's files live. In multi
  // mode that is <output.dir>/<slug>, but a run that started before models[] was
  // added (or a host that runs `wasmtune eval` config-free) leaves them directly
  // in <output.dir>. Accept either rather than failing a gate because a path
  // convention changed mid-project.
  const perTier = path.resolve(cwd, entryConfig(config, trained[0], multi).relOut);
  const shared = path.resolve(cwd, config.output.dir);
  const outDir = [perTier, shared].find((d) =>
    existsSync(path.join(d, "eval.prompts.jsonl"))) ?? perTier;
  const promptsPath = path.join(outDir, "eval.prompts.jsonl");
  if (!existsSync(promptsPath)) {
    console.error(
      "[wasmtune] eval-onnx skipped: no eval.prompts.jsonl under " + perTier +
      " or " + shared +
      " — run `wasmtune eval` first (it builds the same holdout the training " +
      "gate used, so the two numbers are comparable)");
    process.exitCode = 2;
    return;
  }

  // The tokenizer is NOT beside the graph: convert emits only the graph into
  // onnx/, and export.py puts the tokenizer in the training run's merged/ dir.
  // Point at it explicitly rather than letting the script search, because the
  // search previously found nothing and exit 0 made that look like success.
  const tokenizerDir = opts.tokenizerDir
    ? path.resolve(cwd, opts.tokenizerDir)
    : [perTier, shared]
        .map((d) => path.join(d, "run", "merged"))
        .find((d) => existsSync(path.join(d, "tokenizer.json"))) ?? null;
  if (!tokenizerDir) {
    console.error(
      "[wasmtune] eval-onnx skipped: no tokenizer.json in a merged/ dir under " +
      perTier + " or " + shared + " — run `wasmtune convert` first, or pass " +
      "--tokenizer-dir");
    process.exitCode = 2;
    return;
  }

  const systemPrompt = (await import("../src/chat/options.mjs"))
    .defaultSystemPrompt(config.dataset?.siteName ?? path.basename(config.dataDir));

  console.error("[wasmtune] evaluating the quantized artifact that the widget loads");
  // spawn, not execFile: on Windows execFile joins argv into a command line
  // without quoting, and --system-prompt always contains spaces and an
  // apostrophe. The mangled line made the child die with
  // "readFileSync is not defined", which reads like a bug in the script rather
  // than in how it was invoked.
  await spawnNode([
    "scripts/onnx-eval.mjs",
    "--onnx-dir", graphDir,
    "--prompts", promptsPath,
    "--out", path.join(outDir, "eval.onnx.json"),
    "--dtype", opts.dtype ?? "q8",
    "--max-tokens", String(opts.maxTokens ?? 96),
    "--system-prompt", systemPrompt,
    "--python", opts.python ?? "python3",
    "--tokenizer-dir", tokenizerDir,
    // Decode the way the widget decodes. These used to be absent, so the ONNX
    // eval decoded greedily while the browser samples with a repetition
    // penalty — greedy is far more loop-prone, so the gate was measuring the
    // decoder rather than the artifact it claims to score.
    "--temperature", String(config.chat?.temperature ?? 0.3),
    "--top-p", String(config.chat?.topP ?? 1.0),
    "--repetition-penalty", String(config.chat?.repetitionPenalty ?? 1.0),
    "--presence-penalty", String(config.chat?.presencePenalty ?? 0.0),
    "--frequency-penalty", String(config.chat?.frequencyPenalty ?? 0.0),
  ], { cwd: process.cwd() });

  // Regression gate: compare against the fp16 eval the model was trained against.
  const fp16Path = path.join(outDir, "eval.report.json");
  const onnxPath = path.join(outDir, "eval.onnx.json");
  if (!existsSync(fp16Path) || !existsSync(onnxPath)) return;

  const fp16 = JSON.parse(readFileSync(fp16Path, "utf8")).tuned;
  const onnx = JSON.parse(readFileSync(onnxPath, "utf8"));
  if (!fp16 || !onnx) return;

  const drop = (a, b) => (a == null || b == null ? 0 : Number(a) - Number(b));
  const avgDrop = drop(fp16.avg, onnx.avg);
  const THRESHOLD = Number(process.env.WASMTUNE_QUANT_DROP ?? 0.1);
  // Quantization's characteristic failure is not a lower average, it is
  // degenerate output: the graph starts repeating a token forever. A model can
  // lose only 0.08 of average and still loop on 1 prompt in 9, which is a
  // visible failure in a chat widget and invisible to the average check.
  const loopDelta = Number(onnx.looped ?? 0) - Number(fp16.looped ?? 0);
  console.error(
    `[wasmtune] quantization cost: avg ${(fp16.avg ?? 0).toFixed(3)} -> ${(onnx.avg ?? 0).toFixed(3)}` +
    ` (${avgDrop >= 0 ? "-" : "+"}${Math.abs(avgDrop).toFixed(3)}),` +
    ` faithful ${(fp16.faithfulness ?? 0).toFixed(3)} -> ${(onnx.faithfulness ?? 0).toFixed(3)},` +
    ` looped ${fp16.looped ?? 0} -> ${onnx.looped ?? 0}`);

  if (avgDrop > THRESHOLD && !opts.force) {
    console.error(
      `[wasmtune] eval-onnx FAILED: the shipped q8 artifact scores ${avgDrop.toFixed(3)} below the\n` +
      `  fp16 weights it was quantized from (threshold ${THRESHOLD}). Publish anyway\n` +
      `  with --force, or raise WASMTUNE_QUANT_DROP if this loss is expected.`);
    process.exitCode = 1;
  }

  if (loopDelta > 0 && !opts.force) {
    console.error(
      `[wasmtune] eval-onnx FAILED: the quantized artifact loops on ${onnx.looped} prompt(s)\n` +
      `  where the fp16 weights looped on ${fp16.looped ?? 0}. A repetition loop is the\n` +
      `  failure mode quantization is known for, and no average-score check sees it.\n` +
      `  Publish anyway with --force.`);
    process.exitCode = 1;
  }
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
if (cmd === "eval-onnx") return cmdEvalOnnx(opts, cwd);
  if (cmd === "convert") return cmdConvert(opts, cwd);
  if (cmd === "publish") return cmdPublish(opts, cwd);
  if (cmd === "build") return cmdBuild(opts, cwd);
  if (cmd === "serve") return cmdServe(opts, cwd);
  console.error(`unknown command "${cmd}"\n${USAGE}`);
  process.exit(2);
}

main().catch((e) => {
  console.error(`ERROR: ${e.message}`);
  process.exit(1);
});
