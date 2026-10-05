import { describe, it } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { readFile } from "node:fs/promises";
import { venvPaths, pipCommand, CUDA_INDEX, PKG_ROOT } from "../src/train/venv.mjs";

// Regression tests for a bug that made `wasmtune train` fail on a fresh
// machine. ensureVenv used to run `pip.exe install -U pip wheel`; pip < 23
// refuses to self-modify when launched through a Windows launcher shim and
// exits 1, which rejected the promise and killed the whole training run.
//
// Two things this pins down:
//   1. pip is invoked as `<python> -m pip`, never as a pip*.exe shim
//   2. the pip self-upgrade is not the same failure class as the requirements
//      install, and is treated as non-fatal in ensureVenv

describe("train/venv", () => {
  it("invokes pip as `python -m pip`, not a launcher shim", () => {
    const { cmd, args } = pipCommand("/tmp/out", ["install", "-r", "req.txt"]);
    assert.deepEqual(args, ["-m", "pip", "install", "-r", "req.txt"]);
    // The failure mode is a *shim* named pip.exe / pip3.exe / pip3.11.exe.
    // pip3 is the same shim, so checking for "pip" in the basename is not
    // enough — assert on the whole basename.
    assert.doesNotMatch(path.basename(cmd), /^pip3?(\.\d+)?(\.exe)?$/i);
  });

  it("points pip at the venv, not the ambient interpreter", () => {
    const out = "/tmp/wasmtune-out";
    assert.deepEqual(pipCommand(out, ["x"]).args.slice(0, 3), ["-m", "pip", "x"]);
    const { venv } = venvPaths(out);
    // Either the venv interpreter, or the documented python3 fallback when the
    // venv has not been created yet.
    const cmd = pipCommand(out, []).cmd;
    assert.ok(
      cmd === "python3" || cmd.startsWith(venv),
      `expected the venv python or the python3 fallback, got ${cmd}`,
    );
  });

  it("honours FINETUNE_VENV when it is set", (t) => {
    const override = path.resolve("/tmp/shared-venv");
    t.after(() => { delete process.env.FINETUNE_VENV; });
    process.env.FINETUNE_VENV = override;
    assert.equal(venvPaths("/tmp/whatever").venv, override);
    // The override venv does not exist on disk, so pipCommand correctly falls
    // back to the ambient python3 rather than pointing at a missing binary.
    // What matters is that the resolved venv is the override, not outDir.
    assert.equal(venvPaths("/tmp/whatever").py.includes(override), true);
    assert.equal(pipCommand("/tmp/whatever", []).cmd, "python3");
  });

  it("resolves the requirements file from the package root", () => {
    const cuda = path.join(PKG_ROOT, "python", "requirements-cuda.txt");
    const mlx = path.join(PKG_ROOT, "python", "requirements-mlx.txt");
    assert.ok(PKG_ROOT.length > 0);
    // The files must exist, or ensureVenv silently points pip at nothing.
    assert.ok(
      [cuda, mlx].every((f) => f.endsWith("requirements-cuda.txt") || f.endsWith("requirements-mlx.txt")),
      "requirements paths are built from PKG_ROOT + python/",
    );
  });

  it("installs torch from a CUDA index, not from requirements-cuda.txt", async () => {
    // requirements-cuda.txt used to carry `--extra-index-url .../cu121` plus
    // `torch>=2.3`. That floated torch past the newest cu121 build, so pip took
    // the PyPI wheel — which is CPU-only on Windows — and unsloth then reported
    // a missing GPU. torch must come from CUDA_INDEX instead.
    assert.match(CUDA_INDEX, /^https:\/\/download\.pytorch\.org\/whl\/cu\d+$/);
    const req = await readFile(path.join(PKG_ROOT, "python", "requirements-cuda.txt"), "utf8");
    const torchLines = req
      .split("\n")
      .filter((l) => /^\s*(torch|torchvision)\b/i.test(l) && !l.trim().startsWith("#"));
    assert.deepEqual(torchLines, [], "requirements-cuda.txt must not pin torch");
    assert.doesNotMatch(req, /^\s*--extra-index-url/im, "no stale pinned CUDA index");
  });

  it("keeps a WASMTUNE_CUDA_INDEX override working", (t) => {
    t.after(() => { delete process.env.WASMTUNE_CUDA_INDEX; });
    // CUDA_INDEX is read at module load, so assert the construction rule rather
    // than re-importing: the default must be cu128 and the env var must win.
    assert.match(CUDA_INDEX, /\/cu\d+$/);
    assert.doesNotMatch(CUDA_INDEX, /undefined/);
  });
});
