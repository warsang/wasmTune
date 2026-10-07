import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ALLOWLIST } from "../src/models.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(here, "..");

// A `webllm` id is only usable if it exists in the prebuilt registry that
// @mlc-ai/web-llm ships. If it does not, CreateMLCEngine fails at init with
// "Cannot find model record in appConfig for <id>" — after the widget has
// already told the visitor it is loading a model. The 135M entry shipped a
// q4f16 id that web-llm never published, which broke the demo's default tier.
//
// web-llm is a devDependency of this repo for the demo, so when it is present
// we can check every advertised id against the real registry. When it is not
// installed (a consumer running `npm test` on the published package) the
// registry check is skipped rather than failing on a missing devDep.
describe("models: webllm ids exist in the web-llm prebuilt registry", () => {
  async function registryIds() {
    const entry = path.join(pkgRoot, "node_modules/@mlc-ai/web-llm/lib/index.js");
    let lib;
    try {
      lib = await readFile(entry, "utf8");
    } catch {
      return null;
    }
    return new Set(
      [...lib.matchAll(/model:\s*"([^"]+)"/g)].map((m) =>
        m[1].replace(/^https:\/\/huggingface\.co\/[^/]+\//, ""),
      ),
    );
  }

  it("every allowlist webllm id is a real prebuilt id", async (t) => {
    const ids = await registryIds();
    if (!ids) {
      t.skip("web-llm not installed; cannot verify against the registry");
      return;
    }
    const bad = ALLOWLIST
      .filter((m) => m.webllm && !ids.has(m.webllm))
      .map((m) => `${m.hf} -> ${m.webllm}`);
    assert.deepEqual(bad, [], `webllm ids not present in the registry:\n  ${bad.join("\n  ")}`);
  });

  it("anchors the specific id that used to be wrong", () => {
    const entry = ALLOWLIST.find((m) => m.hf === "HuggingFaceTB/SmolLM2-135M-Instruct");
    assert.ok(entry, "135M allowlist entry missing");
    assert.notEqual(
      entry.webllm,
      "SmolLM2-135M-Instruct-q4f16_1-MLC",
      "that build was never published by web-llm",
    );
    // Unquantized, so the footprint must not claim a q4-sized file.
    assert.ok(
      entry.vramBytes >= 2e8,
      `expected an fp16-sized footprint for a q0f16 build, got ${entry.vramBytes}`,
    );
  });

  it("rates every advertised webllm tier with a usable footprint", () => {
    for (const m of ALLOWLIST.filter((x) => x.webllm)) {
      assert.ok(m.vramBytes > 0, `${m.hf}: vramBytes must be positive`);
      assert.ok(m.minDeviceMemoryGB >= 1, `${m.hf}: minDeviceMemoryGB must be set`);
      assert.ok(["tiny", "small", "mid", "large"].includes(m.tier), `${m.hf}: bad tier ${m.tier}`);
    }
  });
});
