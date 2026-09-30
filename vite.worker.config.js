import { defineConfig } from "vite";

// Second half of the demo build: the chat WebWorker.
//
// Vite cannot statically analyse `new Worker(this._workerUrl || new URL(...))`
// inside src/chat/SiteChat.js, so the worker is built here as its own ES bundle
// and handed to the widget through the documented `workerUrl` option. Keeping
// it a real bundle is the point — the worker resolves the bare
// `import("@mlc-ai/web-llm")` to a real chunk, which is what makes the WebGPU
// path work on a static host with no bundler in the consumer's way.

export default defineConfig({
  build: {
    outDir: "../dist-demo",
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
