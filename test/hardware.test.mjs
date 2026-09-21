import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  detectHardware,
  memoryBudgetGB,
  manifestEntries,
  resolveRequirements,
  fitReport,
  rankEntries,
  pickModel,
  smallerEntries,
} from "../src/chat/hardware.mjs";
import { requirementsFor, slugifyModel } from "../src/models.mjs";

const HW_DESKTOP = { webgpu: true, deviceMemoryGB: 8, cores: 8, mobile: false, adapter: { maxBufferSize: 2 ** 30 } };
const HW_DESKTOP_NO_MEM = { webgpu: true, deviceMemoryGB: null, cores: 8, mobile: false, adapter: { maxBufferSize: 2 ** 30 } };
const HW_MOBILE = { webgpu: true, deviceMemoryGB: 4, cores: 8, mobile: true, adapter: { maxBufferSize: 1 << 28 } };
const HW_NOGPU = { webgpu: false, deviceMemoryGB: 4, cores: 4, mobile: false };
const HW_TINY = { webgpu: false, deviceMemoryGB: 0.5, cores: 2, mobile: true };

const TIERED = {
  version: "abc",
  models: [
    {
      id: "gemma-4-e4b", base: "google/gemma-4-E4B-it", source: "tuned",
      artifacts: { gguf: { url: "/models/gemma/m.gguf", bytes: 3.9e9 } },
    },
    {
      id: "qwen3-1.7b", base: "Qwen/Qwen3-1.7B", source: "pretrained",
      artifacts: { webllm: "Qwen3-1.7B-q4f16_1-MLC", onnx: "onnx-community/Qwen3-1.7B-ONNX" },
    },
    {
      id: "qwen3-0.6b", base: "Qwen/Qwen3-0.6B", source: "pretrained",
      artifacts: { webllm: "Qwen3-0.6B-q4f16_1-MLC", onnx: "onnx-community/Qwen3-0.6B-ONNX" },
    },
  ],
};

describe("memoryBudgetGB", () => {
  it("prefers navigator.deviceMemory when present", () => {
    assert.equal(memoryBudgetGB(HW_DESKTOP), 8);
    assert.equal(memoryBudgetGB(HW_MOBILE), 4);
  });

  it("falls back conservatively without deviceMemory (Safari/Firefox)", () => {
    assert.equal(memoryBudgetGB(HW_DESKTOP_NO_MEM), 8); // 8 cores + WebGPU
    assert.equal(memoryBudgetGB({ webgpu: true, cores: 4 }), 4);
    assert.equal(memoryBudgetGB({ mobile: true, cores: 8 }), 4);
    assert.equal(memoryBudgetGB({ mobile: true, cores: 4 }), 2);
    assert.equal(memoryBudgetGB({}), 2);
  });
});

describe("detectHardware", () => {
  it("reads navigator signals and the WebGPU adapter", async () => {
    const scope = {
      navigator: {
        deviceMemory: 8,
        hardwareConcurrency: 10,
        userAgent: "Mozilla/5.0 (Macintosh)",
        gpu: {
          requestAdapter: async () => ({
            info: { vendor: "apple", architecture: "metal-3" },
            limits: { maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 1 << 30 },
          }),
        },
      },
      crossOriginIsolated: true,
    };
    const hw = await detectHardware(scope);
    assert.equal(hw.webgpu, true);
    assert.equal(hw.deviceMemoryGB, 8);
    assert.equal(hw.cores, 10);
    assert.equal(hw.mobile, false);
    assert.equal(hw.crossOriginIsolated, true);
    assert.equal(hw.adapter.vendor, "apple");
  });

  it("degrades to no-WebGPU when the adapter request throws", async () => {
    const scope = {
      navigator: { userAgent: "Mozilla", gpu: { requestAdapter: async () => { throw new Error("denied"); } } },
    };
    const hw = await detectHardware(scope);
    assert.equal(hw.webgpu, false);
    assert.equal(hw.adapter, null);
  });

  it("detects mobile from userAgentData", async () => {
    const scope = { navigator: { userAgentData: { mobile: true }, userAgent: "desktop-like" } };
    const hw = await detectHardware(scope);
    assert.equal(hw.mobile, true);
  });
});

describe("manifestEntries", () => {
  it("passes v2 models[] through", () => {
    const entries = manifestEntries(TIERED);
    assert.equal(entries.length, 3);
    assert.equal(entries[0].id, "gemma-4-e4b");
  });

  it("reshapes legacy single-artifact manifests", () => {
    const entries = manifestEntries({
      base: "qwen/qwen3-0.6b",
      artifacts: { gguf: { url: "/models/m.gguf" } },
      chat: { temperature: 0.5 },
    });
    assert.equal(entries.length, 1);
    assert.equal(entries[0].id, "default");
    assert.equal(entries[0].base, "qwen/qwen3-0.6b");
    assert.equal(entries[0].artifacts.gguf.url, "/models/m.gguf");
    assert.equal(entries[0].chat.temperature, 0.5);
  });

  it("returns [] for an empty manifest", () => {
    assert.deepEqual(manifestEntries({}), []);
    assert.deepEqual(manifestEntries(null), []);
  });
});

describe("resolveRequirements", () => {
  it("uses lib knowledge for allowlisted bases", () => {
    const req = resolveRequirements({ base: "google/gemma-4-E4B-it" });
    assert.equal(req.source, "lib");
    assert.equal(req.tier, "mid");
    assert.equal(req.minDeviceMemoryGB, 6);
    assert.equal(req.needsWebGPU, true);
  });

  it("honors explicit manifest overrides", () => {
    const req = resolveRequirements({
      base: "google/gemma-4-E4B-it",
      requirements: { minDeviceMemoryGB: 12, tier: "large" },
    });
    assert.equal(req.source, "explicit");
    assert.equal(req.minDeviceMemoryGB, 12);
    assert.equal(req.tier, "large");
  });

  it("estimates from artifact bytes for unknown bases", () => {
    const req = resolveRequirements({
      base: "acme/unknown-model",
      artifacts: { gguf: { url: "/models/m.gguf", bytes: 1e9 } },
    });
    assert.equal(req.source, "artifact-size");
    assert.equal(req.vramBytes, 1.3e9);
    assert.equal(req.tier, "small");
    assert.equal(req.minDeviceMemoryGB, 2);
  });

  it("resolves prebuilt browser ids", () => {
    const req = resolveRequirements({ artifacts: { webllm: "Qwen3-0.6B-q4f16_1-MLC" } });
    assert.equal(req.hf, "Qwen/Qwen3-0.6B");
    assert.equal(req.cpuOk, true);
    assert.equal(req.minDeviceMemoryGB, 1);
  });
});

describe("fitReport", () => {
  it("reports each unmet requirement", () => {
    const fit = fitReport({ base: "google/gemma-4-E4B-it" }, HW_MOBILE);
    assert.equal(fit.ok, false);
    assert.ok(fit.reasons.some((r) => r.includes("device memory")), JSON.stringify(fit.reasons));
    assert.ok(fit.reasons.some((r) => r.includes("mobile")), JSON.stringify(fit.reasons));
  });

  it("requires WebGPU for models above the CPU threshold", () => {
    const fit = fitReport({ base: "Qwen/Qwen3-1.7B" }, HW_NOGPU);
    assert.equal(fit.ok, false);
    assert.ok(fit.reasons.some((r) => r.includes("WebGPU")), JSON.stringify(fit.reasons));
  });

  it("passes small CPU-viable models without WebGPU", () => {
    const fit = fitReport({ base: "Qwen/Qwen3-0.6B" }, HW_NOGPU);
    assert.equal(fit.ok, true, JSON.stringify(fit.reasons));
  });
});

describe("rankEntries", () => {
  it("orders by tier then footprint (biggest first)", () => {
    const ids = rankEntries(manifestEntries(TIERED)).map((e) => e.id);
    assert.deepEqual(ids, ["gemma-4-e4b", "qwen3-1.7b", "qwen3-0.6b"]);
  });
});

describe("pickModel", () => {
  it("picks the biggest tier that fits a desktop", () => {
    const pick = pickModel(TIERED, HW_DESKTOP);
    assert.equal(pick.entryId, "gemma-4-e4b");
    assert.equal(pick.forced, false);
    assert.deepEqual(pick.reasons, []);
  });

  it("drops to the mobile-viable tier on phones", () => {
    const pick = pickModel(TIERED, HW_MOBILE);
    assert.equal(pick.entryId, "qwen3-0.6b");
    assert.equal(pick.forced, false);
  });

  it("reports no fit with required vs detected details", () => {
    const pick = pickModel(TIERED, HW_TINY);
    assert.equal(pick.entry, null);
    assert.equal(pick.none, true);
    assert.equal(pick.budget, 0.5);
    assert.ok(pick.reasons.some((r) => r.includes("device memory")), JSON.stringify(pick.reasons));
    assert.equal(pick.smallestId, "qwen3-0.6b");
  });

  it("forces the smallest tier with allowForce", () => {
    const pick = pickModel(TIERED, HW_TINY, { allowForce: true });
    assert.equal(pick.entryId, "qwen3-0.6b");
    assert.equal(pick.forced, true);
    assert.ok(pick.reasons.length > 0);
  });

  it("honors preferId when it fits", () => {
    const pick = pickModel(TIERED, HW_DESKTOP, { preferId: "qwen3-1.7b" });
    assert.equal(pick.entryId, "qwen3-1.7b");
    assert.equal(pick.forced, false);
  });

  it("ignores preferId that does not fit without allowForce", () => {
    const pick = pickModel(TIERED, HW_MOBILE, { preferId: "gemma-4-e4b" });
    assert.equal(pick.entryId, "qwen3-0.6b");
  });

  it("works on legacy single-model manifests via lib requirements", () => {
    const legacy = { base: "Qwen/Qwen3-0.6B", artifacts: { gguf: { url: "/models/m.gguf" } } };
    const pick = pickModel(legacy, HW_DESKTOP);
    assert.equal(pick.entryId, "default");
    assert.equal(pick.entry.base, "Qwen/Qwen3-0.6B");
  });

  it("handles an empty manifest", () => {
    const pick = pickModel({}, HW_DESKTOP);
    assert.equal(pick.none, true);
    assert.match(pick.reasons[0], /no models/);
  });
});

describe("smallerEntries", () => {
  it("lists lower tiers in descending order", () => {
    const smaller = smallerEntries(TIERED, "gemma-4-e4b").map((s) => s.entryId);
    assert.deepEqual(smaller, ["qwen3-1.7b", "qwen3-0.6b"]);
  });

  it("returns the rest when the id is unknown", () => {
    const smaller = smallerEntries(TIERED, "ghost").map((s) => s.entryId);
    assert.deepEqual(smaller, ["qwen3-1.7b", "qwen3-0.6b"]);
  });
});

describe("models requirement exports", () => {
  it("requirementsFor returns null for unknown ids", () => {
    assert.equal(requirementsFor("acme/nope"), null);
    assert.equal(requirementsFor(null), null);
  });

  it("requirementsFor covers --allow-large models", () => {
    const req = requirementsFor("Qwen/Qwen2.5-7B-Instruct");
    assert.equal(req.source, "lib-large");
    assert.equal(req.needsWebGPU, true);
    assert.equal(req.minDeviceMemoryGB, 8);
  });

  it("slugifyModel produces stable entry ids", () => {
    assert.equal(slugifyModel("google/gemma-4-E4B-it"), "gemma-4-e4b-it");
    assert.equal(slugifyModel("Qwen/Qwen3-0.6B"), "qwen3-0.6b");
  });
});
