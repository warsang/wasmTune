// wasmtune — Vue adapter (thin wrapper; <site-chat> stays canonical).
// Render-function based: no SFC compiler needed. Client-only; safe to import
// in SSR (mounts in `mounted`, tears down in `unmounted`).

import { defineComponent, h, onMounted, onUnmounted, ref } from "vue";
import { mountAssistant } from "./index.mjs";

export const SiteChat = defineComponent({
  name: "SiteChat",
  props: {
    manifestUrl: { type: String, default: "/models/model-manifest.json" },
    baseModel: { type: String, default: null },
    title: { type: String, default: "Site assistant" },
    siteName: { type: String, default: null },
    workerUrl: { type: String, default: null },
    allowForce: { type: Boolean, default: false },
    preferModel: { type: String, default: null },
    chatOpts: { type: Object, default: () => ({}) },
  },
  emits: ["ready", "error", "load-failed", "hardware-mismatch"],
  setup(props, { emit, expose }) {
    const host = ref(null);
    const element = ref(null);

    const ask = (text) => element.value?.ask(text) ?? false;
    expose({
      ask,
      get element() {
        return element.value;
      },
      get readyState() {
        return element.value?.readyState ?? "loading";
      },
    });

    const onDomEvent = (e) => {
      if (e.type === "site-chat-ready") emit("ready", e.detail);
      else if (e.type === "site-chat-error") emit("error", e.detail);
      else if (e.type === "site-chat-load-failed") emit("load-failed", e.detail);
      else if (e.type === "site-chat-hw-mismatch") emit("hardware-mismatch", e.detail);
    };

    onMounted(async () => {
      try {
        const mounted = await mountAssistant({
          target: host.value,
          manifestUrl: props.manifestUrl,
          baseModel: props.baseModel,
          title: props.title,
          siteName: props.siteName,
          workerUrl: props.workerUrl,
          allowForce: props.allowForce,
          preferModel: props.preferModel,
          onEvent: onDomEvent,
          ...(props.chatOpts ?? {}),
        });
        element.value = mounted.element;
      } catch (err) {
        emit("error", { message: String(err?.message ?? err) });
      }
    });

    onUnmounted(() => {
      element.value = null;
      if (host.value) host.value.replaceChildren();
    });

    return () => h("div", { ref: host, "data-site-chat-host": "" });
  },
});

export default SiteChat;
