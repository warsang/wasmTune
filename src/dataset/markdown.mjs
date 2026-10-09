// wasmtune — HTML -> Markdown for dataset preparation.
//
// Why this exists. Recent work (AICC/MinerU-HTML arXiv:2511.16397,
// Nemotron-CC-Math arXiv:2508.15096) holds corpus filtering constant and finds
// the HTML->text step alone worth ~1pp across 13 pretraining benchmarks, and up
// to +12 points in domains whose structure was being destroyed. Their diagnosis
// matches what the previous implementation did here: stripping every tag with
// one regex flattens headings, code fences and tables into a word stream and
// keeps nav/footer boilerplate, so a docs site gets trained on its navigation
// menu.
//
// Markdown is the carrier, not the mechanism: the mechanism is preserving the
// structure the page already had. Anchors, fences, pipe tables and heading
// levels are a near-lossless projection of the semantic elements docs HTML uses.
//
// Deliberately dependency-free (this package ships zero dependencies), so this
// is a tokenizer rather than a DOM. It is not a browser-grade parser: images
// are dropped, and nested markup inside table cells or list items is flattened.
// Tests pin the structure it does guarantee.

const DROP_ELEMENTS = /<(script|style|noscript|template|svg|iframe|canvas|video|audio)\b[^>]*>[\s\S]*?<\/\1\s*>/gi;
const DROP_COMMENTS = /<!--[\s\S]*?-->/g;
// Chrome a page renders around its real content. For a docs site that is the nav
// rail, the footer and whatever the first visitor's JS mounted.
const CHROME_ELEMENTS = /<(nav|footer|form|button|dialog|select|datalist|aside)\b[^>]*>[\s\S]*?<\/\1\s*>/gi;
const BOILERPLATE_ATTRS = /<(div|section|main|aside|ul|ol|span)\b[^>]*(?:class|id|role|aria-label)\s*=\s*"[^"]*(?:\b(?:nav|menu|sidebar|breadcrumb|footer|header|cookie|consent|banner|advert|promo|popup|modal|toolbar|pagination|social|share|related|newsletter|subscribe)[\w-]*)[^"]*"[^>]*>[\s\S]*?<\/\1\s*>/gi;

const VOID = new Set(["br", "hr", "img", "input", "meta", "link", "source", "area", "base", "col", "embed", "param", "track", "wbr"]);
const BLOCK = new Set([
  "p", "div", "section", "article", "main", "header", "h1", "h2", "h3", "h4", "h5", "h6",
  "ul", "ol", "li", "blockquote", "pre", "table", "thead", "tbody", "figure",
  "figcaption", "dl", "dt", "dd", "details", "summary",
]);
const INLINE_MAP = { strong: "**", b: "**", em: "*", i: "*", code: "`" };
const HEADING = /^h([1-6])$/;

function decodeEntities(s) {
  return String(s ?? "")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, " ").replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&amp;/g, "&");
}

function attrs(tagText) {
  const out = {};
  for (const m of tagText.slice(0, 500).matchAll(/([a-zA-Z-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))/g)) {
    out[m[1].toLowerCase()] = decodeEntities(m[2] ?? m[3] ?? m[4] ?? "");
  }
  return out;
}

const CELL_SPLIT = /<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi;

// A row is only knowable from its cells, so rows are collected and rendered once
// the table closes.
function extractTable(inner) {
  const rows = [];
  for (const tr of inner.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)) {
    const cells = [...tr[1].matchAll(CELL_SPLIT)]
      .map((m) => m[1].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
    if (cells.length) rows.push(cells);
  }
  if (!rows.length) return "";
  const cols = Math.max(...rows.map((r) => r.length));
  const pad = (r) => Array.from({ length: cols }, (_, i) => r[i] ?? "");
  const body = [`| ${pad(rows[0]).join(" | ")} |`, `| ${Array(cols).fill("---").join(" | ")} |`];
  for (const r of rows.slice(1)) body.push(`| ${pad(r).join(" | ")} |`);
  return body.join("\n");
}

/**
 * Convert an HTML document to structure-preserving Markdown.
 * @param {string} html raw HTML
 * @param {{stripChrome?: boolean}} [opts] drop nav/footer/aside boilerplate
 * @returns {string} Markdown
 */
export function htmlToMarkdown(html, { stripChrome = true } = {}) {
  let src = String(html ?? "");
  src = src.replace(DROP_COMMENTS, " ").replace(DROP_ELEMENTS, " ");
  if (stripChrome) src = src.replace(CHROME_ELEMENTS, " ").replace(BOILERPLATE_ATTRS, " ");

  const out = [];
  let line = [];
  const listStack = [];   // {ordered, counter}
  const anchors = [];     // open <a> hrefs, for the closing bracket
  let tableDepth = 0;
  let tableBuf = null;
  let preDepth = 0;

  const flushLine = () => {
    if (line.length) {
      out.push(line.join("").replace(/[ \t]+/g, " ").trim());
      line = [];
    }
  };
  const para = () => { flushLine(); out.push(""); };

  for (const tok of src.match(/<!--[\s\S]*?-->|<[a-zA-Z\/!][^>]*>|[^<]+/g) ?? []) {
    // Text node.
    if (tok[0] !== "<") {
      const text = decodeEntities(tok);
      if (tableBuf !== null) { tableBuf += text; continue; }
      if (preDepth > 0) { out.push(text.replace(/[ \t]+/g, " ").trimEnd()); continue; }
      if (!/\S/.test(text) && !line.length) continue;
      line.push(text.replace(/\s+/g, " "));
      continue;
    }

    const m = /^<(\/?)([a-zA-Z][a-zA-Z0-9-]*)((?:[^>"']|"[^"]*"|'[^']*')*)(\/?)>/.exec(tok);
    if (!m) continue;
    const [, closing, rawName, rawAttrs] = m;
    const tag = rawName.toLowerCase();

    if (tag === "pre") {
      if (!closing) { flushLine(); preDepth += 1; out.push("```"); }
      else if (preDepth > 0) {
        preDepth -= 1;
        while (out.length && !out[out.length - 1]) out.pop();
        out.push("```", "");
      }
      continue;
    }
    if (preDepth > 0) continue;

    if (tag === "table") {
      if (!closing) { flushLine(); tableDepth += 1; if (tableBuf === null) tableBuf = ""; }
      else if (tableDepth > 0) {
        tableDepth -= 1;
        const md = extractTable(tableBuf);
        if (md) { out.push(md, ""); }
        tableBuf = tableDepth === 0 ? null : tableBuf;
      }
      continue;
    }
    if (tableBuf !== null) { tableBuf += tok; continue; }

    if (VOID.has(tag)) {
      if (tag === "br") flushLine();
      else if (tag === "hr") { flushLine(); out.push("", "---", ""); }
      continue;
    }

    if (closing) {
      if (INLINE_MAP[tag]) line.push(INLINE_MAP[tag]);
      else if (tag === "a") {
        const href = anchors.pop() ?? "";
        line.push(/^https?:\/\//i.test(href) ? `](${href})` : "");
      } else if (tag === "li") flushLine();
      else if (BLOCK.has(tag)) para();
      if (listStack.length && (tag === "ul" || tag === "ol")) listStack.pop();
      continue;
    }

    const a = attrs(rawAttrs);
    const h = HEADING.exec(tag);

    if (h) { flushLine(); line.push("#".repeat(Math.min(6, Number(h[1]))) + " "); continue; }
    if (tag === "ul" || tag === "ol") { flushLine(); listStack.push({ ordered: tag === "ol", counter: 1 }); out.push(""); continue; }
    if (tag === "li") {
      flushLine();
      const st = listStack[listStack.length - 1];
      const marker = st?.ordered ? `${st.counter++}. ` : "- ";
      line.push(`${"  ".repeat(Math.max(0, listStack.length - 1))}${marker}`);
      continue;
    }
    if (tag === "blockquote") { flushLine(); line.push("> "); continue; }

    if (INLINE_MAP[tag]) line.push(INLINE_MAP[tag]);
    else if (tag === "a") {
      const href = a.href ?? "";
      anchors.push(href);
      line.push(/^https?:\/\//i.test(href) ? "[" : "");
    }
    else if (tag === "img" || tag === "td" || tag === "th" || tag === "tr") {
      /* images carry no QA signal; table cells are handled by the buffer */
    }
    else if (BLOCK.has(tag)) flushLine();
  }

  flushLine();
  return out.join("\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/^\n+/, "")
    .trim();
}
