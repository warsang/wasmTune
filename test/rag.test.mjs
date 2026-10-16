import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  buildIndex,
  chunksFromDocs,
  createRag,
  cosine,
  indexScopes,
  loadIndex,
  purgeScope,
  searchChunks,
  splitSections,
  windowText,
  PUBLIC_SCOPE,
} from "../src/rag/index.mjs";
import { createLexicalEmbedder, assertImportable, resolveEmbedder } from "../src/rag/embed.mjs";
import { buildGroundedUserMessage, citedSources, withRagSystemPrompt } from "../src/rag/prompt.mjs";
import { parseToolCall, toolInstruction } from "../src/rag/action.mjs";
import { createToolRegistry } from "../src/rag/tools.mjs";
import { saveIndex } from "../src/rag/store.mjs";
import { PUBLIC_SCOPE as SCOPE_CONSTANT } from "../src/rag/scope.mjs";

const DOCS = [
  {
    path: "services.md",
    text: [
      "# Services",
      "Merovea deploys sovereign AI systems for regulated European companies.",
      "",
      "## Hardware",
      "The machine is sized to the workload; there is no single recommended box.",
      "",
      "## Security audits",
      "Agent-assisted testing under a signed scope, with a human signing the report.",
    ].join("\n"),
  },
  {
    path: "pricing.md",
    text: "# Pricing\nEngagements are quoted per assessment after a scoping call.",
  },
];

describe("windowText", () => {
  it("returns short text whole and splits long text on paragraphs", () => {
    assert.deepEqual(windowText("short", 100, 10), ["short"]);
    const long = `${"a".repeat(80)}\n\n${"b".repeat(80)}`;
    const parts = windowText(long, 100, 10);
    assert.ok(parts.length >= 2, JSON.stringify(parts));
    for (const p of parts) assert.ok(p.length <= 120, `chunk longer than budget: ${p.length}`);
  });

  it("hard-slices a single paragraph that exceeds the budget", () => {
    const parts = windowText("x".repeat(250), 100, 20);
    assert.ok(parts.length >= 3);
  });
});

describe("chunksFromDocs", () => {
  it("keeps heading, title and url metadata for citations", () => {
    const chunks = chunksFromDocs(DOCS, { maxChars: 200, minChars: 20, urlFor: (d) => `/${d.path.replace(/\.md$/, "")}` });
    assert.ok(chunks.length >= 2);
    assert.equal(chunks[0].title, "Services");
    assert.equal(chunks[0].url, "/services");
    // Every section keeps its own heading rather than inheriting the document's
    // first one — the bug this asserts against mislabelled every window.
    assert.deepEqual(
      [...new Set(chunks.map((c) => c.heading))].sort(),
      ["Hardware", "Pricing", "Security audits", "Services"],
    );
    // The heading is inside the indexed text too, so a heading-shaped question
    // can match the section it names.
    const hardware = chunks.find((c) => c.heading === "Hardware");
    assert.match(hardware.text, /^## Hardware/);
  });

  it("drops windows too short to be worth embedding", () => {
    const chunks = chunksFromDocs([{ path: "a.md", text: "tiny" }], { minChars: 200 });
    assert.deepEqual(chunks, []);
  });

  it("splits sections on headings and keeps the preamble", () => {
    const sections = splitSections("# One\nalpha\n\n## Two\nbeta\n\ngamma");
    assert.deepEqual(sections.map((s) => s.heading), ["One", "Two"]);
    assert.equal(sections[1].body, "beta\n\ngamma");
    assert.equal(sections[1].level, 2);
  });
});

describe("lexical embedder", () => {
  it("is deterministic and L2-normalized", async () => {
    const e = createLexicalEmbedder({ dims: 64 });
    const [a] = await e.embed(["sovereign AI for European companies"]);
    const [b] = await e.embed(["sovereign AI for European companies"]);
    assert.deepEqual(Array.from(a), Array.from(b));
    const norm = Math.sqrt(Array.from(a).reduce((s, v) => s + v * v, 0));
    assert.ok(Math.abs(norm - 1) < 1e-6, `norm ${norm}`);
  });

  it("scores a related query above an unrelated one", async () => {
    const e = createLexicalEmbedder({ dims: 512 });
    const [q] = await e.embed(["hardware sizing for the workload"]);
    const [related] = await e.embed(["the machine is sized to the workload"]);
    const [unrelated] = await e.embed(["invoices and payment terms"]);
    assert.ok(cosine(q, related) > cosine(q, unrelated));
  });

  it("resolves through resolveEmbedder and rejects unknown specs", async () => {
    const e = await resolveEmbedder("lexical:128");
    assert.equal(e.id, "lexical-128");
    await assert.rejects(() => resolveEmbedder("telepathy"), /unknown embedder/);
  });
});

describe("buildIndex + searchChunks", () => {
  it("ranks the passage that answers the question first", async () => {
    const embedder = createLexicalEmbedder({ dims: 512 });
    const chunks = chunksFromDocs(DOCS, { maxChars: 200, minChars: 20 });
    const index = await buildIndex({ documents: chunks, embedder });
    const [q] = await embedder.embed(["how is the hardware sized?"]);
    const hits = searchChunks(index.chunks, q, { topK: 2, minScore: 0 });
    assert.ok(hits.length >= 1);
    assert.match(hits[0].text, /sized to the workload/);
  });

  it("respects topK and minScore", async () => {
    const embedder = createLexicalEmbedder({ dims: 256 });
    const index = await buildIndex({ documents: chunksFromDocs(DOCS, { minChars: 20 }), embedder });
    const [q] = await embedder.embed(["security audits"]);
    assert.equal(searchChunks(index.chunks, q, { topK: 1 }).length, 1);
    assert.deepEqual(searchChunks(index.chunks, q, { minScore: 1.1 }), []);
  });

  it("records the embedder so a mismatch can be refused", async () => {
    const embedder = createLexicalEmbedder({ dims: 64 });
    const index = await buildIndex({ documents: chunksFromDocs(DOCS, { minChars: 20 }), embedder });
    assert.equal(index.embedder, "lexical-64");
    assert.equal(index.version, 1);
  });
});

describe("scopes", () => {
  const docs = [
    { source: "public.md", text: "# Public\nSite copy anyone may read." },
    { source: "a1.md", text: "# Assessment 42\nAn IDOR was reproduced by the verifier.", scope: "assessment:42" },
    { source: "a2.md", text: "# Assessment 99\nA leaked key was reproduced here.", scope: "assessment:99" },
  ];

  async function indexWithScopes() {
    const embedder = createLexicalEmbedder({ dims: 512 });
    const chunks = chunksFromDocs(docs, { minChars: 10 });
    const index = await buildIndex({ documents: chunks, embedder });
    return { embedder, index };
  }

  it("never returns a chunk from a scope the caller may not see", async () => {
    const { embedder, index } = await indexWithScopes();
    const [q] = await embedder.embed(["leaked key reproduced"]);
    // The forbidden chunk is the best match by construction; scope filtering
    // has to win over ranking, not merely accompany it.
    const unscoped = searchChunks(index.chunks, q, { topK: 3, minScore: 0 });
    assert.equal(unscoped[0].scope, "assessment:99");

    const allowed = searchChunks(index.chunks, q, { topK: 3, minScore: 0, scope: ["public", "assessment:42"] });
    assert.ok(allowed.length >= 1);
    assert.ok(allowed.every((c) => c.scope !== "assessment:99"), JSON.stringify(allowed.map((c) => c.scope)));
  });

  it("an empty scope list retrieves nothing, which is the logged-out state", async () => {
    const { embedder, index } = await indexWithScopes();
    const [q] = await embedder.embed(["leaked key"]);
    assert.deepEqual(searchChunks(index.chunks, q, { scope: [] }), []);
  });

  it("purges a scope's chunks rather than only hiding them", async () => {
    const { index } = await indexWithScopes();
    assert.ok(indexScopes(index).includes("assessment:42"));
    const result = purgeScope(index, "assessment:42");
    assert.equal(result.removed, 1);
    assert.ok(!indexScopes(index).includes("assessment:42"));
    assert.ok(indexScopes(index).includes("assessment:99"), "other scopes must survive");
  });

  it("createRag narrows retrieval with setScope and returns nothing for a dead session", async () => {
    const { embedder, index } = await indexWithScopes();
    const rag = createRag({ index, embedder, minScore: 0 });
    rag.setScope(["public", "assessment:42"]);
    const hits = await rag.retrieve("leaked key reproduced");
    assert.ok(hits.every((h) => h.scope !== "assessment:99"));

    rag.setScope([]);
    assert.deepEqual(await rag.retrieve("leaked key reproduced"), []);
  });

  it("createRag refuses an embedder that did not build the index", async () => {
    const { index } = await indexWithScopes();
    const other = createLexicalEmbedder({ dims: 128 });
    assert.throws(() => createRag({ index, embedder: other }), /must share an embedder/);
  });

  it("adds session documents under their own scope and purges them on logout", async () => {
    const { embedder, index } = await indexWithScopes();
    const rag = createRag({ index, embedder, minScore: 0 });
    await rag.addDocuments([{ source: "live", text: "# Fresh\nA finding discovered moments ago.", scope: "assessment:7" }], { scope: "assessment:7" });

    rag.setScope(["public", "assessment:7"]);
    const hits = await rag.retrieve("finding discovered moments ago");
    assert.ok(hits.some((h) => h.scope === "assessment:7"), JSON.stringify(hits.map((h) => h.scope)));

    const purged = rag.purgeScope("assessment:7");
    assert.equal(purged.removed, 1);

    // Public content stays retrievable — a purge removes the session's chunks,
    // it does not blind the assistant.
    const after = await rag.retrieve("finding discovered moments ago");
    assert.equal(after.some((h) => h.scope === "assessment:7"), false, JSON.stringify(after.map((h) => h.scope)));
    assert.equal(after.some((h) => h.text.includes("moments ago")), false);
  });
});

describe("grounding prompt", () => {
  const passages = [
    { text: "The machine is sized to the workload.", title: "Hardware", url: "/services" },
    { text: "Quoted per assessment.", title: "Pricing", url: "/pricing" },
  ];

  it("numbers the passages and keeps the question", () => {
    const msg = buildGroundedUserMessage("How is hardware chosen?", passages);
    assert.match(msg, /\[1\] — Hardware \(\/services\)/);
    assert.match(msg, /\[2\] — Pricing \(\/pricing\)/);
    assert.match(msg, /<context>[\s\S]*<\/context>/);
    assert.match(msg, /Question: How is hardware chosen\?$/);
  });

  it("leaves the question alone when nothing was retrieved", () => {
    assert.equal(buildGroundedUserMessage("How is hardware chosen?", []), "How is hardware chosen?");
  });

  it("appends the grounding instruction to a system prompt once", () => {
    const once = withRagSystemPrompt("You are the assistant.");
    const twice = withRagSystemPrompt(once);
    assert.equal(once, twice);
    assert.match(once, /cite the numbers/i);
  });

  it("reports the sources the answer actually cited", () => {
    assert.deepEqual(citedSources("Sized per workload [1].", passages).map((s) => s.url), ["/services"]);
    assert.deepEqual(citedSources("Here is a summary.", passages).map((s) => s.url), ["/services", "/pricing"]);
    assert.deepEqual(citedSources("See [9].", passages).map((s) => s.url), ["/services", "/pricing"]);
  });
});

describe("tool calls", () => {
  it("parses one call and strips it from the visible answer", () => {
    const reply = 'Drafting that now.\n<tool>submit_contact_form</tool>{"name":"Ada"}';
    const { call, text } = parseToolCall(reply, { allowed: ["submit_contact_form"] });
    assert.deepEqual(call, { name: "submit_contact_form", args: { name: "Ada" } });
    assert.equal(text, "Drafting that now.");
  });

  it("treats two calls in one reply as no call at all", () => {
    // Picking one of two is the same class of mistake as trusting the first
    // line of a shell script.
    const reply = '<tool>a</tool>{"x":1} <tool>b</tool>{"y":2}';
    const { call, parseError, text } = parseToolCall(reply);
    assert.equal(call, null);
    assert.match(parseError, /more than one tool call/);
    assert.equal(text.includes("<tool>"), false);
  });

  it("refuses a tool the host did not offer", () => {
    const { call, parseError } = parseToolCall('<tool>rm_rf</tool>{}', { allowed: ["submit_contact_form"] });
    assert.equal(call, null);
    assert.match(parseError, /not available/);
  });

  it("refuses a malformed or non-object payload", () => {
    assert.match(parseToolCall('<tool>a</tool>{"x": }').parseError, /valid JSON/);
    assert.match(parseToolCall('<tool>a</tool>[1,2]').parseError, /must be an object/);
  });

  it("leaves an ordinary answer untouched", () => {
    const { call, text } = parseToolCall("We offer audits and deployment.");
    assert.equal(call, null);
    assert.equal(text, "We offer audits and deployment.");
  });

  it("builds an instruction only when tools exist", () => {
    assert.equal(toolInstruction([]), "");
    assert.match(toolInstruction(["submit_contact_form"]), /submit_contact_form/);
  });
});

describe("tool registry", () => {
  it("runs a registered function and returns its result", async () => {
    const seen = [];
    const tools = createToolRegistry({ submit_contact_form: (args) => { seen.push(args); return "sent"; } });
    const result = await tools.run("submit_contact_form", { name: "Ada" });
    assert.deepEqual(result, { ok: true, name: "submit_contact_form", result: "sent" });
    assert.deepEqual(seen, [{ name: "Ada" }]);
  });

  it("never runs an unregistered name", async () => {
    const tools = createToolRegistry({ a: () => "ran" });
    const result = await tools.run("b", {});
    assert.equal(result.ok, false);
    assert.match(result.error, /no tool named "b"/);
  });

  it("lets a tool reject or rewrite its arguments", async () => {
    const tools = createToolRegistry({
      strict: { run: (args) => args, validate: (args) => (args.email?.includes("@") ? undefined : "email is missing") },
      tidy: { run: (args) => args, validate: (args) => ({ name: String(args.name).trim() }) },
    });
    const rejected = await tools.run("strict", { email: "nope" });
    assert.equal(rejected.ok, false);
    assert.match(rejected.error, /email is missing/);

    const rewritten = await tools.run("tidy", { name: "  Ada  ", secret: "x" });
    assert.deepEqual(rewritten.result, { name: "Ada" });
  });

  it("reports a throwing tool instead of taking the widget down", async () => {
    const tools = createToolRegistry({ boom: () => { throw new Error("network down"); } });
    const result = await tools.run("boom", {});
    assert.equal(result.ok, false);
    assert.match(result.error, /network down/);
  });

  it("gives up on a tool that hangs", async () => {
    const tools = createToolRegistry({ slow: () => new Promise(() => {}) });
    const result = await tools.run("slow", {}, {}, { timeoutMs: 20 });
    assert.equal(result.ok, false);
    assert.match(result.error, /did not finish within 20ms/);
  });

  it("describes the available tools to the model", () => {
    const tools = createToolRegistry({
      book_call: { run: () => {}, description: "open the booking flow", parameters: { email: "string" } },
    });
    const instruction = tools.instruction();
    assert.match(instruction, /<tool>book_call<\/tool>/);
    assert.match(instruction, /open the booking flow/);
    assert.equal(createToolRegistry({}).instruction(), "");
  });
});

describe("loadIndex", () => {
  it("rebuilds a query embedder from the recorded spec", async () => {
    const embedder = createLexicalEmbedder({ dims: 128 });
    const index = await buildIndex({ documents: chunksFromDocs(DOCS, { minChars: 20 }), embedder });
    const loaded = await loadIndex(JSON.parse(JSON.stringify(index)), { resolveEmbedder });
    assert.equal(loaded.embedder.id, "lexical-128");
    const [q] = await loaded.embedder.embed(["pricing"]);
    assert.ok(searchChunks(loaded.index.chunks, q, { topK: 1, minScore: 0 }).length === 1);
  });

  it("refuses an index naming an embedder it cannot build", async () => {
    await assert.rejects(() => loadIndex({ embedder: "telepathy" }, { resolveEmbedder }), /unknown embedder/);
  });

  it("defaults a scope-less index to the public scope", async () => {
    const embedder = createLexicalEmbedder({ dims: 64 });
    const index = await buildIndex({ documents: chunksFromDocs(DOCS, { minChars: 20 }), embedder });
    assert.ok(index.chunks.every((c) => c.scope === PUBLIC_SCOPE));
  });
});

// ---------------------------------------------------------------------------
// Regressions. Each of these is a review comment on the first revision of this
// branch, and each names the failure it prevents.
// ---------------------------------------------------------------------------

describe("regressions from review", () => {
  it("terminates when overlap >= maxChars", () => {
    // The step was maxChars - overlap, so an overlap as large as the window made
    // it zero and the loop never advanced: a config file could hang the CLI.
    const parts = windowText("x".repeat(500), 100, 400);
    assert.ok(parts.length >= 5 && parts.length <= 500, `unexpected window count ${parts.length}`);
    for (const p of parts) assert.ok(p.length <= 100);
  });

  it("records retrieval hints in the index so --top-k is not a no-op", async () => {
    const embedder = createLexicalEmbedder({ dims: 256 });
    const index = await buildIndex({
      documents: chunksFromDocs(DOCS, { minChars: 20 }),
      embedder,
      retrieval: { topK: 1, minScore: 0 },
    });
    assert.deepEqual(index.retrieval, { topK: 1, minScore: 0 });
    const rag = createRag({ index, embedder });
    // topK comes from the index when the caller does not override it.
    assert.equal((await rag.retrieve("hardware is sized to the workload")).length, 1);
  });

  it("refuses import specs that are not https or a package name", async () => {
    assert.equal(assertImportable("https://cdn.example.com/transformers.mjs"), "https://cdn.example.com/transformers.mjs");
    assert.equal(assertImportable("@huggingface/transformers"), "@huggingface/transformers");
    for (const bad of ["file:///tmp/evil.mjs", "data:text/javascript,alert(1)", "./local.mjs", "../up.mjs"]) {
      assert.throws(() => assertImportable(bad), /must be an https URL or a package name/, bad);
    }
  });

  it("persists public chunks only, using the shared scope constant", async () => {
    // A fake IndexedDB that records what was written: the assertion is about the
    // filter, and the filter must agree with the search (one constant, imported
    // by both) or a session's chunks get written to disk.
    const written = new Map();
    const fakeScope = {
      indexedDB: {
        open() {
          const request = {};
          queueMicrotask(() => {
            request.result = {
              objectStoreNames: { contains: () => true },
              transaction() {
                return {
                  objectStore: () => ({ put: (value, key) => written.set(key, value), get: () => {}, clear: () => {} }),
                  set oncomplete(fn) { queueMicrotask(fn); },
                  set onerror(_fn) {},
                };
              },
            };
            request.onsuccess?.();
          });
          return request;
        },
      },
    };
    const embedder = createLexicalEmbedder({ dims: 64 });
    const index = await buildIndex({
      documents: chunksFromDocs(
        [
          { source: "public.md", text: "# Public\nSite copy anyone may read." },
          { source: "a.md", text: "# Session\nA session-only finding.", scope: "assessment:42" },
        ],
        { minChars: 10 },
      ),
      embedder,
    });
    const saved = await saveIndex(index, { scope: fakeScope });
    assert.equal(saved.saved, true);
    assert.equal(saved.chunks, 1);
    const persisted = [...written.values()][0];
    assert.ok(persisted.chunks.every((c) => c.scope === SCOPE_CONSTANT), JSON.stringify(persisted.chunks.map((c) => c.scope)));
  });
});
