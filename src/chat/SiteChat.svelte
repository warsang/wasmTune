<script>
  // wasmtune — Svelte adapter (thin wrapper; <site-chat> stays canonical).
  // Client-only: element mounts in onMount; SSR renders just the host div.
  import { onMount, onDestroy, createEventDispatcher } from "svelte";
  import { mountAssistant } from "./index.mjs";

  export let manifestUrl = "/models/model-manifest.json";
  export let baseModel = null;
  export let title = "Site assistant";
  export let siteName = null;
  export let workerUrl = null;
  export let chatOpts = {};

  /** Programmatic access: bind:this + ref.ask("hi") / ref.element */
  export const ref = {};
  export function ask(text) {
    return ref.element?.ask(text) ?? false;
  }

  const dispatch = createEventDispatcher();
  let host;
  let mounted = null;

  onMount(async () => {
    const onEvent = (e) => {
      if (e.type === "site-chat-ready") dispatch("ready", e.detail);
      else if (e.type === "site-chat-error") dispatch("error", e.detail);
      else if (e.type === "site-chat-load-failed") dispatch("load-failed", e.detail);
    };
    try {
      mounted = await mountAssistant({
        target: host,
        manifestUrl,
        baseModel,
        title,
        siteName,
        workerUrl,
        onEvent,
        ...(chatOpts ?? {}),
      });
      ref.element = mounted.element;
    } catch (err) {
      dispatch("error", { message: String(err?.message ?? err) });
    }
  });

  onDestroy(() => {
    mounted = null;
    ref.element = null;
    host?.replaceChildren();
  });
</script>

<div bind:this={host} data-site-chat-host=""></div>
