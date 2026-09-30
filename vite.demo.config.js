import { defineConfig } from "vite";

// Static promo/demo site for the npm README. Deployed to GitHub Pages by
// .github/workflows/pages.yml, so `base` has to match the project path.
export default defineConfig({
  root: "demo",
  base: process.env.DEMO_BASE ?? "/wasmTune/",
  publicDir: "public",
  build: {
    outDir: "../dist-demo",
    emptyOutDir: true,
    target: "es2022",
    chunkSizeWarningLimit: 8192, // @mlc-ai/web-llm ships a large runtime
    rollupOptions: {
      // wasmtune has zero dependencies by design: @mlc-ai/web-llm is an optional
      // peer that the chat worker imports lazily. The demo installs it so the
      // WebGPU path is real; the other optional peers stay external so the
      // worker fails honestly (and visibly) instead of pulling in a backend
      // the demo does not actually serve.
      external: ["@wllama/wllama", "@huggingface/transformers"],
    },
  },
  worker: {
    format: "es",
    rollupOptions: {
      external: ["@wllama/wllama", "@huggingface/transformers"],
    },
  },
});
