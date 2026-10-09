import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { htmlToMarkdown } from "../src/dataset/markdown.mjs";

// A docs page converted to training text must keep the things that carry the
// meaning and drop the things that do not. The previous implementation (one
// regex removing every tag) kept neither half of that bargain: it flattened
// headings, code and tables into one stream, and kept the nav rail.

const DEMO_HTML = readFileSync(new URL("../demo/index.html", import.meta.url), "utf8");

describe("htmlToMarkdown: structure preservation", () => {
  const md = htmlToMarkdown(DEMO_HTML);

  it("keeps a code sample as a fenced block", () => {
    // The demo page has a curl example inside <pre><code>.
    assert.match(md, /```[\s\S]*curl\s+https:\/\/ingest/,
      "a <pre><code> block must survive as a fence, not become inline words");
  });

  it("keeps heading levels", () => {
    assert.match(md, /^#{1,6} /m, "expected at least one heading");
    // Docs title maps to the h1; section headings are h2.
    assert.match(md, /^# .*time-series store/m);
  });

  it("keeps a content link and its destination", () => {
    const md = htmlToMarkdown(
      '<p>See the <a href="https://github.com/warsang/wasmTune#readme">repo</a> for details.</p>');
    assert.match(md, /\[repo\]\(https:\/\/github\.com\/warsang\/wasmTune#readme\)/,
      "an absolute href in body content should survive as a markdown link");
  });

  it("keeps inline anchors out of link syntax when there is no href", () => {
    const md = htmlToMarkdown("<p>See <a>the docs</a> now.</p>");
    assert.ok(!/\[\]/.test(md), "an href-less anchor must not become an empty link");
    assert.match(md, /See the docs now\./);
  });

  it("drops the nav rail, the footer and the widget chrome", () => {
    assert.ok(!/GETTING STARTED/.test(md), "sidebar nav should be gone");
    assert.ok(!/Lumen is a fictional product/.test(md), "site footer should be gone");
    assert.ok(!/how this works/.test(md), "chat dock chrome should be gone");
    assert.ok(!/<site-chat>/.test(md), "no raw tags should survive");
  });

  it("stays well-formed: balanced fences, no stray tags", () => {
    const fences = (md.match(/^```/gm) ?? []).length;
    assert.equal(fences % 2, 0, `unbalanced code fences: ${fences}`);
    assert.ok(!/<[a-zA-Z][^>]*>/.test(md.replace(/```[\s\S]*?```/g, "")),
      "inline HTML tags leaked into the output");
  });
});

describe("htmlToMarkdown: units", () => {
  it("maps heading levels monotonically", () => {
    const out = htmlToMarkdown("<h1>a</h1><h3>b</h3><h6>c</h6>");
    assert.match(out, /^# a$/m);
    assert.match(out, /^### b$/m);
    assert.match(out, /^###### c$/m);
  });

  it("preserves table structure", () => {
    const md = htmlToMarkdown("<table><tr><th>Plan</th><th>Writes</th></tr><tr><td>Free</td><td>1000</td></tr></table>");
    assert.match(md, /\|\s*Plan\s*\|\s*Writes\s*\|/);
    assert.match(md, /\|\s*Free\s*\|\s*1000\s*\|/);
    assert.match(md, /\|\s*---\s*\|/);
  });

  it("keeps list markers and nesting", () => {
    const md = htmlToMarkdown("<ul><li>one</li><li>two</li></ul><ol><li>first</li><li>second</li></ol>");
    assert.match(md, /- one/);
    assert.match(md, /- two/);
    assert.match(md, /1\. first/);
    assert.match(md, /2\. second/);
  });

  it("keeps inline emphasis", () => {
    const md = htmlToMarkdown("<p>plain <strong>bold</strong> and <em>ital</em> and <code>code</code></p>");
    assert.match(md, /plain \*\*bold\*\* and \*ital\* and `code`/);
  });

  it("decodes entities instead of emitting &amp;", () => {
    assert.match(htmlToMarkdown("<p>a &amp; b &lt;tag&gt;</p>"), /a & b <tag>/);
  });

  it("drops script and style wholesale", () => {
    const md = htmlToMarkdown("<p>keep</p><script>var x = 'drop me';</script><style>.p{color:red}</style>");
    assert.match(md, /keep/);
    assert.ok(!/drop me/.test(md));
    assert.ok(!/color:red/.test(md));
  });

  it("returns an empty string for an empty document", () => {
    assert.equal(htmlToMarkdown(""), "");
    assert.equal(htmlToMarkdown("<script>x</script>"), "");
  });

  it("respects stripChrome:false for non-web HTML", () => {
    const frag = "<nav>menu</nav><p>body</p>";
    assert.doesNotMatch(htmlToMarkdown(frag), /menu/);
    assert.match(htmlToMarkdown(frag, { stripChrome: false }), /menu/);
  });
});
