// wasmtune — auto-venv bootstrap. Node spawns Python; never imports torch.

import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
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
  const { venv, pip } = venvPaths(outDir);
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
  console.error(`[wasmtune] pip install -r ${path.basename(req)} (first run takes a while) ...`);
  await run(pip, ["install", "-U", "pip", "wheel"], { cwd });
  await run(pip, ["install", "-r", req], { cwd });
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
