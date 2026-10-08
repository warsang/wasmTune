import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Window } from "happy-dom";
import { defineSiteChat } from "../src/chat/SiteChat.js";

// An answer can take tens of seconds — the ONNX path is a single non-streaming
// forward pass, and a first run also downloads weights. Without an indicator an
// empty log is indistinguishable from a hang. The failure mode that actually
// matters is the opposite one: a spinner that never clears, which looks like a
// hung app forever. So the invariant under test is that every terminal message
// clears it, not merely that it appears.

const SRC = readFileSync(new URL("../src/chat/SiteChat.js", import.meta.url), "utf8");

describe("site-chat pending indicator", () => {
  let window;
  const prev = {};

  beforeEach(async () => {
    window = new Window({ url: "https://x.test/docs/" });
    for (const k of ["window", "document", "customElements", "HTMLElement", "CustomEvent"]) {
      prev[k] = globalThis[k];
      globalThis[k] = window[k];
    }
    await window.happyDOM.waitUntilComplete();
  });

  afterEach(async () => {
    for (const k of Object.keys(prev)) globalThis[k] = prev[k];
    await window.happyDOM.close();
  });

  function mount() {
    defineSiteChat({});
    const el = document.createElement("site-chat");
    el.setAttribute("model", "Qwen3-0.6B-q4f16_1-MLC");
    document.body.append(el);
    return el;
  }

  it("shows a pending row when the user submits", () => {
    const el = mount();
    el.ask("how long does setup take?");
    const pending = el.shadowRoot.querySelector('[data-pending="1"]');
    assert.ok(pending, "expected a pending row after submit");
    assert.match(pending.textContent, /assistant:/);
    // The dot animation lives on a CSS pseudo-element, so assert on the class.
    assert.ok(pending.querySelector(".dots"), "expected the animated dots span");
  });

  it("says it is still loading when no engine has reported ready yet", () => {
    const el = mount();
    el.ask("hi");
    const pending = el.shadowRoot.querySelector('[data-pending="1"]');
    assert.match(pending.textContent, /waiting for the model to load/);
  });

  it("clears the pending row on the first streamed token", () => {
    const el = mount();
    el.ask("hi");
    assert.ok(el.shadowRoot.querySelector('[data-pending="1"]'));
    el._appendToken("Hel");
    assert.equal(el.shadowRoot.querySelector('[data-pending="1"]'), null);
  });

  it("clears the pending row on a loop retraction", () => {
    const el = mount();
    el.ask("hi");
    el._retractStream("partial", "stopped");
    assert.equal(el.shadowRoot.querySelector('[data-pending="1"]'), null);
  });

  it("never stacks more than one pending row", () => {
    const el = mount();
    el.ask("one");
    el.ask("two");
    assert.equal(el.shadowRoot.querySelectorAll('[data-pending="1"]').length, 1);
  });

  // Static half: the DOM tests above cover the paths reachable without a worker,
  // but "done" and "error" only arrive as postMessage payloads.
  it("clears on every terminal worker message", () => {
    const terminal = ['m.type === "done"', 'm.type === "error"'];
    for (const t of terminal) {
      const idx = SRC.indexOf(t);
      assert.ok(idx !== -1, `message branch not found: ${t}`);
      const branch = SRC.slice(idx, idx + 400);
      assert.ok(
        branch.includes("_clearPending()"),
        `${t} must clear the pending indicator or a spinner can outlive the turn`,
      );
    }
  });

  it("clears from the hardware-mismatch and load-failure banners too", () => {
    // Match the definitions, not the call sites earlier in the file.
    for (const fn of ["_showHwMismatch(detail) {", "_showLoadError({ kind, url, error }) {"]) {
      const idx = SRC.indexOf(fn);
      assert.ok(idx !== -1, `definition not found: ${fn}`);
      assert.ok(SRC.slice(idx, idx + 220).includes("_clearPending()"),
        `${fn} must clear the pending indicator`);
    }
  });
});
