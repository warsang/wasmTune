// wasmtune — tools: JavaScript the page registers, and the model asks for.
//
// The point of running the model in the browser is that it can *do* things
// without handing the job to a server: fill a form, open a booking flow, put a
// draft in a compose box, call the site's own API. So a tool here is a plain
// function on the page. The model asks; the registry runs it.
//
// What the registry guarantees, because the model is small and its output is
// untrusted text:
//
//   - only registered names run. A reply that names anything else is a refused
//     call, not an eval.
//   - arguments are validated before the function is reached. A tool may bring
//     its own `validate`; whatever it returns is what `run` receives.
//   - a tool that throws, rejects or hangs is reported as a failed call with a
//     message the chat can show. It never takes the widget down with it, and it
//     never leaves the visitor looking at a spinner.
//   - results are returned to the caller (the widget), not fed back into the
//     model. Feeding a result back is how you get an agent loop; a site that
//     wants a second turn can ask for one explicitly.

const DEFAULT_TIMEOUT_MS = 15_000;

/**
 * @param {Record<string, Function|{run: Function, validate?: Function, description?: string, parameters?: object}>} tools
 */
export function createToolRegistry(tools = {}) {
  const entries = new Map();
  for (const [name, value] of Object.entries(tools ?? {})) {
    if (typeof value === "function") {
      entries.set(name, { name, run: value, validate: null, description: null, parameters: null });
      continue;
    }
    if (value && typeof value.run === "function") {
      entries.set(name, {
        name,
        run: value.run,
        validate: typeof value.validate === "function" ? value.validate : null,
        description: value.description ?? null,
        parameters: value.parameters ?? null,
      });
      continue;
    }
    throw new Error(`tool "${name}" must be a function or an object with a run() function`);
  }

  return {
    get size() {
      return entries.size;
    },
    names() {
      return [...entries.keys()];
    },
    has(name) {
      return entries.has(String(name ?? "").toLowerCase());
    },
    get(name) {
      return entries.get(String(name ?? "").toLowerCase()) ?? null;
    },
    /**
     * Validate then run. Always resolves — a failed tool is data, not an
     * exception, because the caller is a chat turn that still has to render.
     */
    async run(name, args = {}, ctx = {}, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
      const key = String(name ?? "").toLowerCase();
      const entry = entries.get(key);
      if (!entry) {
        return { ok: false, name: key, error: `no tool named "${name}" is registered on this page` };
      }
      let payload = args ?? {};
      if (entry.validate) {
        let verdict;
        try {
          verdict = entry.validate(payload);
        } catch (err) {
          return { ok: false, name: key, error: `arguments rejected: ${err?.message ?? err}` };
        }
        // A validator may return `false` (reject as-is), a string (the reason),
        // or an object (the payload to use, which is how a tool strips fields it
        // does not want to see).
        if (verdict === false) return { ok: false, name: key, error: "arguments rejected by the tool" };
        if (typeof verdict === "string") return { ok: false, name: key, error: verdict };
        if (verdict && typeof verdict === "object") payload = verdict;
      }
      try {
        const result = await withTimeout(entry.run(payload, ctx), timeoutMs, key);
        return { ok: true, name: key, result };
      } catch (err) {
        return { ok: false, name: key, error: String(err?.message ?? err) };
      }
    },
    /** The instruction that tells the model which tools exist. */
    instruction() {
      if (!entries.size) return "";
      const lines = [...entries.values()].map((e) => {
        const what = e.description ? ` — ${e.description}` : "";
        const params = e.parameters ? ` ${JSON.stringify(e.parameters)}` : "";
        return `  <tool>${e.name}</tool>${params}${what}`;
      });
      const first = [...entries.values()][0];
      const example = first
        ? `\nExample of a valid reply (nothing else on the line): <tool>${first.name}</tool>{"name":"Jane Doe","email":"jane@example.com","message":"I would like a pentest for my company"}`
        : "";
      return (
        "This page can perform actions for the visitor through you. When the visitor asks you " +
        "to perform one of the actions below, you MUST reply with exactly one line and nothing " +
        "else, in this exact form: <tool>name</tool>{\"argument\":\"value\"}. The page will run " +
        "it and report the result back to you. Do not describe the action in prose, do not " +
        "answer around it, and do not wrap it in code fences. If a required argument is missing " +
        "(for example a name or email), ask the visitor for it instead of inventing it. " +
        "Available actions:\n" + lines.join("\n") + example
      );
    },
  };
}

function withTimeout(promise, ms, name) {
  if (!Number.isFinite(ms) || ms <= 0) return promise;
  return new Promise((resolve, reject) => {
    // Deliberately NOT unref'd. An unref'd timer does not hold the event loop
    // open, so when the tool is the only pending work — a test, a CLI, a page
    // that has finished rendering — the loop drains, the timeout never fires,
    // and this promise never settles. Node 22 reports that as
    // "Promise resolution is still pending but the event loop has already
    // resolved" and cancels the rest of the file; Node 24 tolerated it, which is
    // how it got past a local run and onto CI.
    const timer = setTimeout(
      () => reject(new Error(`tool "${name}" did not finish within ${ms}ms`)),
      ms,
    );
    Promise.resolve(promise).then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}
