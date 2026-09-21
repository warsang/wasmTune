import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { buildDataset } from "../src/dataset/index.mjs";
import { chunkDocs } from "../src/dataset/chunk.mjs";
import { chunksToQa, compressAnswer, commandDensity, extractCodeExamples } from "../src/dataset/qa.mjs";
import { conversationalSeeds } from "../src/dataset/converse.mjs";

describe("dataset", () => {
  it("chunks deterministically with dedup", () => {
    const docs = [
      { path: "a.md", text: "hello world. ".repeat(200) },
      { path: "b.md", text: "hello world. ".repeat(200) },
    ];
    const once = chunkDocs(docs);
    const twice = chunkDocs(docs);
    assert.deepEqual(once, twice);
    // identical docs dedup to a single chunk set
    assert.ok(once.length > 0 && once.length === chunkDocs([docs[0]]).length);
  });

  it("builds sft/dpo/grpo jsonl from a folder", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "sft-"));
    const data = path.join(dir, "docs");
    await mkdir(data, { recursive: true });
    await writeFile(path.join(data, "page1.md"), "# Title\n\n" + "Some meaningful site content. ".repeat(60));
    await writeFile(path.join(data, "page2.html"), "<h1>Hi</h1><p>" + "More content here. ".repeat(60) + "</p>");
    const report = await buildDataset(
      { dataDir: data, model: "Qwen/Qwen2.5-1.5B-Instruct", method: "sft", output: { dir: path.join(dir, "out") } },
      { cwd: dir },
    );
    assert.equal(report.docs, 2);
    assert.ok(report.chunks >= 1);
    assert.ok(report.sftPairs >= 1);
    assert.ok(report.answerChars.p50 <= 800, `p50 too long: ${report.answerChars.p50}`);
    assert.ok(report.dpoSeedPairs >= 1);
    assert.ok(report.grpoSeedPrompts >= 1);
  });
});

describe("qa synthesis", () => {
  const chunk = (text, source = "a.md") => ({ source, ordinal: 0, text });

  it("turns headings into short-answer questions", () => {
    const pairs = chunksToQa([
      chunk("# ActiveProcessLinks\n\nActiveProcessLinks is a LIST_ENTRY inside EPROCESS that links all processes. Drivers walk it to enumerate processes. Extra sentence one. Extra sentence two. Extra sentence three. Extra sentence four. Extra sentence five."),
    ], { siteName: "TestSite" });
    const q = pairs.find((p) => p.meta.kind === "fact-qa");
    assert.ok(q, "expected a fact-qa pair");
    assert.match(q.messages[0].content, /ActiveProcessLinks/);
    const answer = q.messages[1].content;
    assert.ok(answer.length <= 600, `answer too long: ${answer.length}`);
    assert.ok(answer.length >= 20);
  });

  it("extracts definitions for term-like subjects", () => {
    const pairs = chunksToQa([
      chunk("EPROCESS is the kernel structure describing a process. It contains the PID and links."),
    ]);
    const def = pairs.find((p) => p.meta.kind === "definition");
    assert.ok(def, "expected a definition pair");
    assert.match(def.messages[0].content, /EPROCESS/);
  });

  it("uses literal FAQ lines with the following paragraph", () => {
    const pairs = chunksToQa([
      chunk("Some intro.\n\nWhat is the PID of kfsample?\n\nThe PID is 1312 in every world.\n\nMore text here to pad."),
    ]);
    const faq = pairs.find((p) => p.meta.kind === "faq");
    assert.ok(faq, "expected an faq pair");
    assert.match(faq.messages[1].content, /1312/);
  });

  it("skips command-dump chunks for QA", () => {
    const dump = Array.from({ length: 12 }, (_, i) => `kd> !cmd${i} arg`).join("\n")
      + "\nSome filler words here to pass length. ".repeat(4).trim();
    const pairs = chunksToQa([chunk(dump)]);
    assert.equal(pairs.length, 0);
    assert.ok(commandDensity(dump) >= 0.5);
  });

  it("dedupes identical questions across chunks", () => {
    const t = "# Widget\n\nWidget is a thing. It does stuff well enough here.";
    const pairs = chunksToQa([chunk(t, "a.md"), chunk(t, "b.md")]);
    const questions = pairs.map((p) => p.messages[0].content.toLowerCase());
    assert.equal(new Set(questions).size, questions.length);
  });

  it("compressAnswer caps sentences and chars", () => {
    const text = Array.from({ length: 20 }, (_, i) => `Sentence number ${i} with enough words to matter.`).join(" ");
    const out = compressAnswer(text);
    assert.ok(out.length <= 600);
    assert.ok(out.split(/(?<=[.!?])\s+/).length <= 4);
  });

  it("extracts fenced code with heading topic, skips transcripts", () => {
    const text = [
      "# Hello-world driver",
      "A minimal LKM looks like this:",
      "```c",
      "#include <linux/module.h>",
      "static int __init hello_init(void) { return 0; }",
      "module_init(hello_init);",
      "```",
      "And a debugger transcript:",
      "```kd",
      "kd> !process 0 0",
      "```",
    ].join("\n");
    const ex = extractCodeExamples(text);
    assert.equal(ex.length, 1);
    assert.match(ex[0].question, /c code for Hello-world driver/);
    assert.match(ex[0].answer, /linux\/module\.h/);
  });

  it("code-qa pairs survive command-dense chunks", () => {
    const dump = Array.from({ length: 12 }, (_, i) => `kd> !cmd${i} arg`).join("\n") +
      "\n# Loader\n```c\nint main(void) { return puts(\"hi\"); }\n```\n";
    const pairs = chunksToQa([{ source: "a.md", ordinal: 0, text: dump }]);
    assert.ok(pairs.some((p) => p.meta.kind === "code-qa"), JSON.stringify(pairs.map((p) => p.meta.kind)));
    assert.ok(!pairs.some((p) => p.meta.kind === "fact-qa"));
  });
});

describe("sft blend", () => {
  it("mixes qa, converse, and concise explain with short answers", async () => {
    const { buildSftBlend, answerCharsStats, kindCounts } = await import("../src/dataset/sft.mjs");
    const chunks = [
      { source: "a.md", ordinal: 0, text: "# Widget\n\nWidget is a device. It processes widgets efficiently and well.\n\nRun the widget daily. Restart the widget weekly." },
      { source: "b.md", ordinal: 0, text: "# Gadget\n\nGadget is a tool. It helps with gadgets across the entire workflow here.\n\nInstall the gadget first. Configure the gadget second." },
    ];
    const pairs = buildSftBlend(chunks, { siteName: "Demo", maxPairs: 100 });
    const kinds = kindCounts(pairs);
    assert.ok((kinds["fact-qa"] ?? 0) + (kinds.definition ?? 0) > 0, JSON.stringify(kinds));
    assert.ok((kinds.converse ?? 0) >= 5, JSON.stringify(kinds));
    const stats = answerCharsStats(pairs);
    assert.ok(stats.p90 <= 800, `p90 too long: ${stats.p90}`);
  });

  it("is deterministic across runs", async () => {
    const { buildSftBlend } = await import("../src/dataset/sft.mjs");
    const chunks = [{ source: "a.md", ordinal: 0, text: "# Widget\n\nWidget is a device that works well for everyone here." }];
    const a = JSON.stringify(buildSftBlend(chunks, { siteName: "Demo", maxPairs: 50 }));
    const b = JSON.stringify(buildSftBlend(chunks, { siteName: "Demo", maxPairs: 50 }));
    assert.equal(a, b);
  });
});

describe("dpo seeds", () => {
  const pair = (q, a, meta = {}) => ({
    messages: [
      { role: "user", content: q },
      { role: "assistant", content: a },
    ],
    meta,
  });

  it("teaches conciseness: short answer chosen, chunk dump rejected", async () => {
    const { sftToDpoSeed } = await import("../src/dataset/dpo.mjs");
    const gold = "ActiveProcessLinks is a LIST_ENTRY inside EPROCESS linking all processes.";
    const dump = gold + " " + "Additional dump context about lists and walking. ".repeat(30);
    const out = sftToDpoSeed(
      [pair("What is ActiveProcessLinks?", gold, { source: "a.md", ordinal: 0, kind: "fact-qa" })],
      { chunks: [{ source: "a.md", ordinal: 0, text: dump }] },
    );
    const concise = out.find((t) => t.meta.kind === "dpo-concise");
    assert.ok(concise, "expected a dpo-concise triplet");
    assert.equal(concise.chosen, gold);
    assert.ok(concise.rejected.length > gold.length * 2);
  });

  it("teaches faithfulness: entity-swapped lie rejected", async () => {
    const { sftToDpoSeed, swapEntities } = await import("../src/dataset/dpo.mjs");
    const gold = "The PID of kfsample.exe is 1312 in every world.";
    const lie = swapEntities("The PID of kfsample.exe is 1312 and the port is 8080.");
    assert.ok(lie && lie !== "The PID of kfsample.exe is 1312 and the port is 8080.");
    assert.ok(lie.includes("8080") && lie.includes("1312"));
    const out = sftToDpoSeed(
      [pair("What is the PID?", gold, { source: "a.md", ordinal: 0, kind: "fact-qa" })],
      { chunks: [] },
    );
    const faithful = out.find((t) => t.meta.kind === "dpo-faithful");
    assert.ok(faithful, "expected a dpo-faithful triplet");
    assert.equal(faithful.chosen, gold);
    assert.notEqual(faithful.rejected, gold);
  });

  it("returns [] for entity-poor answers without crashing", async () => {
    const { swapEntities } = await import("../src/dataset/dpo.mjs");
    assert.equal(swapEntities("hello there friend"), null);
  });
});

describe("dpo loop contrasts", () => {
  it("mines looped tuned outputs with references as chosen", async () => {
    const { evalLoopContrasts } = await import("../src/dataset/dpo.mjs");
    const report = {
      tuned: {
        results: [
          { id: "mem-0", repetition: true, output: "loop ".repeat(50) },
          { id: "mem-1", repetition: false, output: "fine answer here" },
        ],
      },
    };
    const byId = {
      "mem-0": { prompt: "What is X?", reference: "X is a thing.", source: "a.md" },
      "mem-1": { prompt: "What is Y?", reference: "Y is another.", source: "a.md" },
    };
    const out = evalLoopContrasts(report, byId, {});
    assert.equal(out.length, 1);
    assert.equal(out[0].prompt, "What is X?");
    assert.equal(out[0].chosen, "X is a thing.");
    assert.ok(out[0].rejected.includes("loop"));
    assert.equal(out[0].meta.kind, "dpo-loop");
  });

  it("returns [] with no loops or no tuned section", async () => {
    const { evalLoopContrasts } = await import("../src/dataset/dpo.mjs");
    assert.deepEqual(evalLoopContrasts({ tuned: null }, {}, {}), []);
    assert.deepEqual(evalLoopContrasts({ tuned: { results: [] } }, {}, {}), []);
  });
});

describe("holdout split", () => {
  it("holds out ~5% deterministically and never empty", async () => {
    const { splitHoldout } = await import("../src/dataset/index.mjs");
    const chunks = Array.from({ length: 200 }, (_, i) => ({
      source: "a.md", ordinal: i, text: `chunk ${i} `.repeat(30),
      // spread first bytes across 0x00-0xff (golden-ratio hash)
      hash: (((i * 2654435761) >>> 0) % 256).toString(16).padStart(2, "0") + "0".repeat(14),
    }));
    const { trainChunks, holdoutChunks } = splitHoldout(chunks, 0.05);
    assert.ok(holdoutChunks.length > 0 && holdoutChunks.length < 40);
    assert.equal(trainChunks.length + holdoutChunks.length, 200);
    const again = splitHoldout(chunks, 0.05);
    assert.deepEqual(again.holdoutChunks.map((c) => c.ordinal), holdoutChunks.map((c) => c.ordinal));
    // train and holdout never overlap
    const trainSet = new Set(trainChunks);
    assert.ok(holdoutChunks.every((c) => !trainSet.has(c)));
  });
});

describe("conversational seeds", () => {
  it("greets and states scope with the site name", () => {
    const seeds = conversationalSeeds({ siteName: "Demo Docs" });
    assert.ok(seeds.length >= 10);
    assert.ok(seeds.every((p) => p.meta.kind === "converse"));
    const hi = seeds.find((p) => /^hi$/i.test(p.messages[0].content));
    assert.ok(hi && hi.messages[1].content.includes("Demo Docs"));
    assert.ok(!hi.messages[1].content.includes("{site}"));
  });

  it("includes ignorance fallbacks without placeholders", () => {
    const seeds = conversationalSeeds({ siteName: "Demo Docs" });
    for (const p of seeds) {
      for (const m of p.messages) assert.ok(!m.content.includes("{site}"));
    }
  });
});
