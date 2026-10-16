// wasmtune — actions: letting the model *ask* for something to be done.
//
// A 0.6-4B model in a browser cannot be trusted to perform side effects. It can
// be trusted to say what it thinks should happen, and the host decides whether
// to do it. So this module parses an intent and hands it back — it never calls
// anything, and the widget never posts anything by itself. For the contact form
// that is the difference between "the assistant drafted a message for you to
// send" and "a small model emailed a stranger on your behalf because the prompt
// drifted".
//
// The wire format is a single marker line, chosen over JSON-mode because JSON
// mode is engine-specific (WebLLM supports it, the wllama path does not) and
// because a strict template is something these sizes follow far more reliably
// than a schema:
//
//   <tool>submit_contact_form</tool>{"name":"…","email":"…","message":"…"}

export const TOOL_RE = /<tool>\s*([a-z0-9_]+)\s*<\/tool>\s*(\{[\s\S]*?\})?/i;

export const CONTACT_ACTION = "submit_contact_form";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

/**
 * Validate and normalize a parsed action.
 *
 * Unknown keys are dropped rather than passed through: the host's handler should
 * receive exactly the fields it documented, and a model that invents
 * `{"bcc":"…"}` should not be able to reach code that forwards its input.
 */
export function validateAction(name, args, { allowed = [CONTACT_ACTION] } = {}) {
  if (!allowed.includes(name)) {
    return { ok: false, name, args, error: `action "${name}" is not allowed` };
  }
  if (name === CONTACT_ACTION) {
    const name_ = String(args?.name ?? "").trim();
    const email = String(args?.email ?? "").trim();
    const message = String(args?.message ?? "").trim();
    const problems = [];
    if (!name_) problems.push("name is missing");
    if (!EMAIL_RE.test(email)) problems.push("email is missing or not an address");
    if (message.length < 10) problems.push("message is too short to send");
    if (problems.length) return { ok: false, name, args: { name: name_, email, message }, error: problems.join("; ") };
    return {
      ok: true,
      name,
      args: {
        name: name_,
        email,
        message,
        company: args?.company ? String(args.company).trim() : null,
      },
    };
  }
  return { ok: true, name, args: args ?? {} };
}

/**
 * Pull an action out of a model reply.
 *
 * Returns `{ action: null, text }` for the ordinary case of an answer with no
 * tool call, so callers can always render something.
 */
export function parseAction(reply, { allowed = [CONTACT_ACTION] } = {}) {
  const text = String(reply ?? "");
  const m = text.match(TOOL_RE);
  if (!m) return { action: null, text: stripAction(text) };
  let args = null;
  if (m[2]) {
    try {
      args = JSON.parse(m[2]);
    } catch {
      // A malformed payload is a model mistake, not a host error: the answer
      // still shows, and the reader can send the form themselves.
      return { action: null, text: stripAction(text), parseError: "tool arguments were not valid JSON" };
    }
  }
  const validated = validateAction(m[1].toLowerCase(), args ?? {}, { allowed });
  return { action: validated, text: stripAction(text) };
}

export function stripAction(text) {
  return String(text ?? "")
    .replace(new RegExp(TOOL_RE.source, "gi"), "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** The instruction that makes the model emit the marker. Appended to the system prompt. */
export function actionInstruction({ actions = [CONTACT_ACTION] } = {}) {
  if (!actions.length) return "";
  return (
    `If the visitor asks to be contacted, wants to book a call, or asks you to send a message ` +
    `for them, collect their name, email and a one-or-two sentence message, then reply with a ` +
    `single line of the form <tool>${CONTACT_ACTION}</tool>{"name":"…","email":"…","message":"…"} ` +
    `and nothing else. Never invent an email address: ask for the missing field instead.`
  );
}
