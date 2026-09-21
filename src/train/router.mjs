// wasmtune — backend router: auto picks MLX on Apple Silicon,
// Unsloth on CUDA, else a loud actionable error.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import os from "node:os";

const execFileAsync = promisify(execFile);

export async function detectPlatform({ cudaOverride = null } = {}) {
  const platform = os.platform();
  const arch = os.arch();
  const isMacArm = platform === "darwin" && arch === "arm64";
  const cuda = cudaOverride ?? (await hasCuda());
  return { platform, arch, isMacArm, cuda };
}

export async function hasCuda() {
  // nvidia-smi present => CUDA-capable box (Unsloth floor: capability 7.0+).
  try {
    await execFileAsync("nvidia-smi", ["-L"], { timeout: 8000 });
    return true;
  } catch {
    return false;
  }
  // Note: torch.cuda.is_available() is re-checked inside Python before training.
}

export async function resolveBackend(requested = "auto", opts = {}) {
  const req = String(requested ?? "auto").toLowerCase();
  if (req === "unsloth" || req === "mlx") return req;
  if (req !== "auto") throw new Error(`unknown training.backend "${requested}" (expected auto|unsloth|mlx)`);
  const { isMacArm, cuda, platform } = await detectPlatform(opts);
  if (isMacArm && !cuda) return "mlx";
  if (cuda) return "unsloth";
  throw new Error(
    `no training backend available (platform=${platform} arch=${os.arch()} cuda=false). ` +
      `On Apple Silicon this auto-selects MLX; on Linux/Windows it needs an NVIDIA GPU ` +
      `(nvidia-smi not found). Options: run on a CUDA box, on an M-series Mac, or use the CUDA Dockerfile.`,
  );
}

export function trainerFor(backend) {
  if (backend === "mlx") return "train_mlx.py";
  if (backend === "unsloth") return "train_unsloth.py";
  throw new Error(`unknown backend "${backend}"`);
}
