// wasmtune — auto-venv bootstrap. Node spawns Python; never imports torch.

import { existsSync } from "node:fs";
import { spawn, execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
// src/train -> package root
export const PKG_ROOT = path.resolve(here, "..", "..");

export function venvPaths(outDir) {
  // FINETUNE_VENV overrides the per-run venv (useful when the network is
  // flaky: reuse one good venv instead of bootstrapping per output dir).
  const override = process.env.FINETUNE_VENV;
  const venv = override ? path.resolve(override) : path.resolve(outDir, ".finetune-venv");
  const bin = process.platform === "win32" ? "Scripts" : "bin";
  const py = path.join(venv, bin, process.platform === "win32" ? "python.exe" : "python");
  const pip = path.join(venv, bin, process.platform === "win32" ? "pip.exe" : "pip");
  return { venv, py, pip };
}

/**
 * How pip must be invoked inside a venv: `<python> -m pip ...`.
 *
 * Never use the venv's `pip` / `pip3` / `pip3.x` shim. pip < 23 refuses to
 * modify itself when it is launched through a Windows launcher shim and exits
 * 1 with "ERROR: To modify pip, please run the following command: ...
 * -m pip install -U pip" — so a `pip install -U pip wheel` step run through
 * the shim hard-failed the entire training run on a fresh venv. `pip3` is the
 * same shim and fails identically; only `-m pip` works.
 *
 * (This is a self-modification guard, not a network problem:
 * PIP_DISABLE_PIP_VERSION_CHECK does not affect it.)
 */
export function pipCommand(outDir, args) {
  const { py } = venvPaths(outDir);
  return { cmd: existsSync(py) ? py : "python3", args: ["-m", "pip", ...args] };
}

/**
 * CUDA wheel index for torch. Override with WASMTUNE_CUDA_INDEX when your
 * driver needs a different one (e.g. cu126 on an older driver).
 *
 * cu128 carries torch up to 2.11 and still supports sm_75 (Turing), so it is
 * the safest default for consumer cards.
 */
export const CUDA_INDEX = `https://download.pytorch.org/whl/${process.env.WASMTUNE_CUDA_INDEX ?? "cu128"}`;

/** Does this venv already have a CUDA-enabled torch? */
export function torchIsCuda(outDir) {
  const { py } = venvPaths(outDir);
  if (!existsSync(py)) return false;
  try {
    const out = execFileSync(py, ["-c", "import torch;print(torch.cuda.is_available())"], {
      encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 60_000,
    });
    return out.trim() === "True";
  } catch {
    return false;
  }
}

function run(cmd, args, { cwd }) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`${cmd} ${args.join(" ")} exited ${code}`)),
    );
  });
}

export async function ensureVenv({ outDir, backend, cwd = process.cwd(), python = "python3" }) {
  const { venv } = venvPaths(outDir);
  if (!existsSync(venv)) {
    console.error(`[wasmtune] creating venv at ${venv} ...`);
    await run(python, ["-m", "venv", venv], { cwd });
  } else {
    console.error(`[wasmtune] reusing venv at ${venv}`);
  }
  if (process.env.FINETUNE_SKIP_INSTALL === "1") {
    console.error("[wasmtune] FINETUNE_SKIP_INSTALL=1, skipping pip install");
    return venvPaths(outDir);
  }
  const req = path.join(PKG_ROOT, "python", backend === "mlx" ? "requirements-mlx.txt" : "requirements-cuda.txt");
  // Upgrading pip is a convenience, not a prerequisite: an old pip still
  // resolves requirements fine, so a failure here must not abort training.
  const upgrade = pipCommand(outDir, ["install", "-U", "pip", "wheel"]);
  try {
    await run(upgrade.cmd, upgrade.args, { cwd });
  } catch (err) {
    console.error(`[wasmtune] pip self-upgrade skipped, continuing: ${err.message}`);
  }
  // torch first, from an explicit CUDA index. requirements-cuda.txt must not
  // resolve it, because the default PyPI wheel is CPU-only on Windows and the
  // failure then surfaces as unsloth claiming you have no GPU.
  if (backend !== "mlx") {
    console.error(`[wasmtune] installing CUDA torch from ${CUDA_INDEX} ...`);
    const torch = pipCommand(outDir, [
      "install", "torch", "torchvision", "--index-url", CUDA_INDEX,
    ]);
    await run(torch.cmd, torch.args, { cwd });
  }
  console.error(`[wasmtune] pip install -r ${path.basename(req)} (first run takes a while) ...`);
  const install = pipCommand(outDir, ["install", "-r", req]);
  await run(install.cmd, install.args, { cwd });
  // Fail here, with a fix, rather than 60 lines into the trainer where unsloth
  // reports "Unsloth cannot find any torch accelerator? You need a GPU." —
  // which is what you read when a CPU-only wheel got installed on a box that
  // very much has a GPU.
  if (backend !== "mlx" && !torchIsCuda(outDir)) {
    throw new Error(
      `torch in ${venv} has no CUDA support, so training cannot run even though ` +
      `an NVIDIA GPU was detected.\n` +
      `Install a CUDA build and check the result:\n` +
      `  ${py} -m pip install torch torchvision --index-url ${CUDA_INDEX}\n` +
      `  ${py} -c "import torch; print(torch.cuda.is_available())"\n` +
      `If your driver needs an older CUDA, set WASMTUNE_CUDA_INDEX (e.g. cu126).`,
    );
  }
  return venvPaths(outDir);
}

export async function runTrainer({ outDir, backend, trainerArgs = [], cwd = process.cwd() }) {
  const { py } = venvPaths(outDir);
  const script = path.join(PKG_ROOT, "python", backend === "mlx" ? "train_mlx.py" : "train_unsloth.py");
  const pyExe = existsSync(py) ? py : "python3";
  console.error(`[wasmtune] ${pyExe} ${path.basename(script)} ${trainerArgs.join(" ")}`);
  await run(pyExe, [script, ...trainerArgs], { cwd });
}

// Run any packaged python script (eval, export, ...) with the venv python.
export async function runPython({ outDir, script, args = [], cwd = process.cwd() }) {
  const { py } = venvPaths(outDir);
  const full = path.isAbsolute(script) ? script : path.join(PKG_ROOT, script);
  const pyExe = existsSync(py) ? py : "python3";
  console.error(`[wasmtune] ${pyExe} ${path.basename(full)} ${args.join(" ")}`);
  await run(pyExe, [full, ...args], { cwd });
}
