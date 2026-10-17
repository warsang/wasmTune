import { defineConfig } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Builds the chat WebWorker into a self-contained bundle that ships inside the
// package, so `mountAssistant()` works on a plain static host with no bundler
// and no copy/rewrite step: the worker resolves `@mlc-ai/web-llm` at build
// time instead of at runtime inside a module worker (where a bare specifier
// cannot resolve — module workers do not inherit the page's import map).
//
// Run with `npm run build:worker`. Output: src/chat/bundled/ (worker.js plus
// its chunks), which _spawnWorker prefers when no workerUrl is given. The
// unbundled worker.js stays as the source for bundler users and for
// `--from-source`.
//
// Fallbacks (@wllama/wllama, @huggingface/transformers) stay external so the
// bundle stays small; a consumer that needs them passes a transformersUrl /
// wasmUrl, or bundles the worker itself with `wasmtune bundle --from-source`.

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    outDir: path.resolve(here, "src/chat/bundled"),
    emptyOutDir: true,
    target: "es2022",
    lib: {
      entry: path.resolve(here, "src/chat/worker.js"),
      formats: ["es"],
      fileName: () => "worker.js",
    },
    rollupOptions: {
      external: ["@wllama/wllama", "@huggingface/transformers"],
      output: {
        chunkFileNames: "worker-[name]-[hash].js",
      },
    },
  },
});
