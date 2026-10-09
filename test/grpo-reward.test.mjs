import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { validateConfig } from "../src/config.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// python/reward.py runs inside the training venv, so the test runs python
// rather than reimplementing the scoring — a second implementation would
// produce numbers that look comparable and are not.
function runPython(code, venv = ".onnx-venv") {
  const py = path.join(ROOT, venv, "Scripts", "python.exe");
  if (!existsSync(py)) return null; // no venv on this machine; skip silently
  return execFileSync(py, ["-c", code], { cwd: ROOT, encoding: "utf8" });
}

describe("python/reward.py", () => {
  const REF = "Keys are shown once and stored in ~/.config/lumen/credentials.json.";

  it("is loadable and exposes a batched TRL reward", () => {
    const out = runPython(`
import sys; sys.path.insert(0, "python")
from reward import reward, score
assert callable(reward) and callable(score)
print("ok")
`);
    if (out === null) return;
    assert.match(out, /ok/);
  });

  it("ranks a factual short answer above a vague one", () => {
    const out = runPython(`
import sys; sys.path.insert(0, "python")
from reward import reward
ref = "Keys are shown once and stored in ~/.config/lumen/credentials.json."
good = reward(["q"], ["They live in ~/.config/lumen/credentials.json"], reference=[ref])[0]
vague = reward(["q"], ["They live in a database"], reference=[ref])[0]
empty = reward(["q"], [""], reference=[ref])[0]
print(f"{good} {vague} {empty}")
`);
    if (out === null) return;
    const [good, vague, empty] = out.trim().split(/\s+/).map(Number);
    assert.ok(good > vague, `factual answer (${good}) must beat vague (${vague})`);
    assert.equal(empty, 0, "an empty completion must score 0");
  });

  it("penalises padding instead of rewarding it", () => {
    // The old default was min(len(c)/500, 1.0), which made padding the optimal
    // strategy. This is the regression it is guarding.
    const out = runPython(`
import sys; sys.path.insert(0, "python")
from reward import reward
ref = "Keys are shown once and stored in ~/.config/lumen/credentials.json."
correct = "in ~/.config/lumen/credentials.json"
padded = correct + (" the quick brown fox " * 60)
c = reward(["q"], [correct], reference=[ref])[0]
p = reward(["q"], [padded], reference=[ref])[0]
old_correct = min(len(correct) / 500, 1.0)
old_padded = min(len(padded) / 500, 1.0)
print(f"{c} {p} {old_correct} {old_padded}")
`);
    if (out === null) return;
    const [correct, padded, oldCorrect, oldPadded] = out.trim().split(/\s+/).map(Number);
    assert.ok(correct > padded, `correct (${correct}) must beat padded (${padded})`);
    assert.ok(
      oldPadded > oldCorrect,
      `expected the old reward to prefer padding (${oldPadded} > ${oldCorrect}) — ` +
      "if it does not, this test no longer describes the bug it guards",
    );
  });

  it("cannot separate two values that share tokens — and that is stated, not hidden", () => {
    // Keyword recall measures "did the right words appear", not "was the right
    // value produced". A confabulated path that happens to share tokens with
    // the real one scores the same as the real one.
    //
    // This is a real limit of the default reward, not a test to tune away:
    // it is why `wasmtune eval` also reports `untraced` tokens, and why the
    // README frames GRPO as shaping a property rather than recovering facts.
    // Recovering facts needs the fact to be in the corpus enough times.
    const out = runPython(`
import sys; sys.path.insert(0, "python")
from reward import reward
ref = "Keys are shown once and stored in ~/.config/lumen/credentials.json."
right = reward(["q"], ["They are in ~/.config/lumen/credentials.json"], reference=[ref])[0]
wrong = reward(["q"], ["They are in ~/.config/lumen/API_KEYS.env"], reference=[ref])[0]
print(right, wrong)
`);
    if (out === null) return;
    const [r, w] = out.trim().split(/\s+/).map(Number);
    assert.equal(
      r,
      w,
      `expected keyword recall to score both paths equally (${r} vs ${w}); ` +
      "if this differs, the reward's fidelity improved and this note is stale",
    );
  });

  it("scores by how many reference terms survive, not by one lucky hit", () => {
    // The designed signal: recall is a count. A completion carrying 3 of the
    // reference's 4 terms beats one carrying 1 of them. It is a coarse proxy —
    // it measures that the right words appeared, not that the right value was
    // produced — which is why it is framed as shaping elsewhere in this package.
    const out = runPython(`
import sys; sys.path.insert(0, "python")
from reward import reward, extract_keywords
ref = "Retries use exponential backoff starting at 200ms."
assert len(extract_keywords(ref)) == 4, extract_keywords(ref)
many = reward(["q"], ["Exponential backoff starts, retries use it"], reference=[ref])[0]
one = reward(["q"], ["Retries happen"], reference=[ref])[0]
zero = reward(["q"], ["Sure, no problem at all"], reference=[ref])[0]
print(many, one, zero)
`);
    if (out === null) return;
    const [many, one, zero] = out.trim().split(/\s+/).map(Number);
    assert.ok(many > one, `3 terms (${many}) must beat 1 term (${one})`);
    assert.equal(zero, 0, `a completion with none of the terms must score 0 (got ${zero})`);
  });
});

describe("grpo.rewardFile validation", () => {
  const cfg = (extra = {}) => ({
    dataDir: "./docs",
    model: "Qwen/Qwen2.5-0.5B-Instruct",
    training: { backend: "auto" },
    ...extra,
  });

  it("no longer requires a rewardFile, now that a default exists", () => {
    const errs = validateConfig(cfg({ method: "grpo" }));
    assert.equal(
      errs.filter((e) => /rewardFile/.test(e)).length,
      0,
      "grpo.rewardFile must be optional: the CLI falls back to python/reward.py",
    );
  });

  it("accepts a .py reward module", () => {
    const errs = validateConfig(cfg({ method: "grpo", grpo: { rewardFile: "./rewards.py" } }));
    assert.equal(errs.filter((e) => /rewardFile/.test(e)).length, 0);
  });

  it("rejects a .mjs reward module the python trainer cannot import", () => {
    // This was the dead template path: importlib cannot load .mjs, so the
    // override silently never applied and every GRPO run used the fallback.
    const errs = validateConfig(cfg({ method: "grpo", grpo: { rewardFile: "./rewards.mjs" } }));
    assert.match(errs.join("; "), /rewardFile must be a \.py/);
  });
});

describe("resolveRewardFile wiring", () => {
  it("ships a default reward.py the CLI can copy", () => {
    assert.ok(
      existsSync(path.join(ROOT, "python", "reward.py")),
      "python/reward.py must exist for the CLI to hand the trainer a default",
    );
  });

  it("writes the used reward module into outDir so a host can open it", async () => {
    // Mirrors what the CLI does on every train run, without importing the CLI.
    const dir = await mkdtemp(path.join(tmpdir(), "reward-"));
    await writeFile(path.join(dir, "rewards.py"), "def reward(prompts, completions, **kw):\n    return [0.0]\n");
    const outDir = path.join(dir, ".finetune");
    await mkdir(outDir, { recursive: true });
    const src = path.join(ROOT, "python", "reward.py");
    const dest = path.join(outDir, "default_reward.py");
    const { copyFile } = await import("node:fs/promises");
    await copyFile(src, dest);
    assert.ok(existsSync(dest));
    const { readFile } = await import("node:fs/promises");
    assert.match(await readFile(dest, "utf8"), /def reward\(/);
  });
});
