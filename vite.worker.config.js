import { defineConfig } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Second half of the demo build: the chat WebWorker.
//
// Vite cannot statically analyse `new Worker(this._workerUrl || new URL(...))`
// inside src/chat/SiteChat.js, so the worker is built here as its own ES bundle
// and handed to the widget through the documented `workerUrl` option. Keeping
// it a real bundle is the point — the worker resolves the bare
// `import("@mlc-ai/web-llm")` to a real chunk, which is what makes the WebGPU
// path work on a static host with no bundler in the consumer's way.

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    // Absolute, not "../dist-demo". Without an explicit `root` this config's
    // root is the package directory, so a relative outDir resolved to the
    // package's PARENT — the worker was written outside the repo and never
    // reached the Pages artifact. The widget then 404'd on worker.js and the
    // chat sat on "loading model…" forever, because the worker never started
    // and so never posted a status.
    outDir: path.resolve(here, "dist-demo"),
    emptyOutDir: false, // the page build owns the directory
    target: "es2022",
    lib: {
      entry: "src/chat/worker.js",
      formats: ["es"],
      fileName: () => "worker.js",
    },
    rollupOptions: {
      // Only the primary engine is bundled. The GGUF and ONNX fallbacks stay
      // external so they fail at runtime and the widget reports "no local
      // engine" instead of half-working against a backend we do not serve.
      external: ["@wllama/wllama", "@huggingface/transformers"],
      output: {
        chunkFileNames: "worker-[name]-[hash].js",
      },
    },
  },
});