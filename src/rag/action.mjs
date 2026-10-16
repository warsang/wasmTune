// wasmtune — reading a tool call out of a model reply.
//
// The wire format is a single marker line rather than JSON mode, because JSON
// mode is engine-specific (WebLLM supports it, the wllama path does not) and a
// strict template is something these sizes follow far more reliably than a
// schema:
//
//   <tool>name</tool>{"argument":"value"}
//
// Parsing is deliberately paranoid. The reply is model output, and the whole
// point of this module is to decide whether the page should *run* something
// because of it.

export const TOOL_RE = /<tool>\s*([a-z0-9_.-]+)\s*<\/tool>\s*(\{[^}]*\}|\[[^\]]*\])?/gi;

/**
 * Parse a reply into zero or one tool call.
 *
 * Callers hand the result to a tool registry; this module never runs anything.
 * Two calls in one reply is treated as no call at all: a small model that
 * emits two is either looping or guessing, and picking one of them is the same
 * class of mistake as trusting the first `sudo` in a shell script.
 */
export function parseToolCall(reply, { allowed = null } = {}) {
  const text = String(reply ?? "");
  const matches = [...text.matchAll(TOOL_RE)];
  if (!matches.length) return { call: null, text: stripToolCalls(text) };
  if (matches.length > 1) {
    return {
      call: null,
      text: stripToolCalls(text),
      parseError: `more than one tool call in one reply (${matches.length})`,
    };
  }
  const [match] = matches;
  const name = match[1].toLowerCase();
  if (allowed && !allowed.includes(name)) {
    return { call: null, text: stripToolCalls(text), parseError: `tool "${name}" is not available` };
  }
  let args = {};
  if (match[2]) {
    try {
      const parsed = JSON.parse(match[2]);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) args = parsed;
      else {
        return { call: null, text: stripToolCalls(text), parseError: "tool arguments must be an object" };
      }
    } catch {
      // A malformed payload is a model mistake, not a host error: the answer
      // still shows, and the visitor can do the thing themselves.
      return { call: null, text: stripToolCalls(text), parseError: "tool arguments were not valid JSON" };
    }
  }
  return { call: { name, args }, text: stripToolCalls(text) };
}

export function stripToolCalls(text) {
  return String(text ?? "")
    .replace(TOOL_RE, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** The instruction that makes a model emit the marker. Empty when no tools exist. */
export function toolInstruction(names = []) {
  if (!names.length) return "";
  return (
    "If the visitor asks you to do something these tools can do, reply with a single line " +
    `<tool>name</tool>{"argument":"value"} and nothing else. Available: ${names.join(", ")}. ` +
    "If a required detail is missing, ask for it — never invent an address or a name."
  );
}
