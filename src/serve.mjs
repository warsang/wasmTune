// wasmtune — tiny static preview server for `wasmtune serve`.
// Serves cwd + webDir model files with COOP/COEP so WebGPU workers + caching behave.

import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { existsSync } from "node:fs";

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".css": "text/css", ".wasm": "application/wasm",
  ".bin": "application/octet-stream", ".gguf": "application/octet-stream",
  ".so": "application/octet-stream",
};

export async function serve({ root = process.cwd(), port = 8080, webDir = null } = {}) {
  const server = http.createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent(new URL(req.url, "http://x").pathname);
      const candidates = [path.join(root, urlPath.slice(1))];
      if (webDir && urlPath.startsWith("/models/")) {
        candidates.unshift(path.join(webDir, urlPath.slice("/models/".length)));
      }
      for (const full of candidates) {
        if (existsSync(full)) {
          const st = await stat(full);
          const file = st.isDirectory() ? path.join(full, "index.html") : full;
          if (!existsSync(file)) continue;
          const body = await readFile(file);
          res.writeHead(200, {
            "content-type": MIME[path.extname(file).toLowerCase()] ?? "application/octet-stream",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "cache-control": "no-cache",
          });
          res.end(body);
          return;
        }
      }
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("not found");
    } catch (e) {
      res.writeHead(500, { "content-type": "text/plain" });
      res.end(String(e.message));
    }
  });
  await new Promise((resolve) => server.listen(port, resolve));
  return { server, url: `http://localhost:${port}` };
}
