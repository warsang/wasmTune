// wasmtune — React adapter (thin wrapper; <site-chat> stays canonical).
// Client-only: renders nothing meaningful during SSR; mounts on effect.

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { mountAssistant } from "./index.mjs";

function toHandler(onEvent, map) {
  if (typeof onEvent !== "function") return undefined;
  return (e) => onEvent({ type: map[e.type] ?? e.type, detail: e.detail, element: e.target });
}

export const SiteChat = forwardRef(function SiteChat(
  {
    manifestUrl,
    baseModel = null,
    title = "Site assistant",
    siteName = null,
    workerUrl = null,
    onReady,
    onError,
    onLoadFailed,
    onEvent,
    ...chatOpts
  },
  ref,
) {
  const hostRef = useRef(null);
  const elRef = useRef(null);

  useImperativeHandle(ref, () => ({
    get element() {
      return elRef.current;
    },
    ask: (text) => elRef.current?.ask(text) ?? false,
    get readyState() {
      return elRef.current?.readyState ?? "loading";
    },
  }), []);

  useEffect(() => {
    let cancelled = false;
    const emit = (e) => {
      if (e.type === "site-chat-ready") onReady?.(e.detail);
      else if (e.type === "site-chat-error") onError?.(e.detail);
      else if (e.type === "site-chat-load-failed") onLoadFailed?.(e.detail);
      onEvent?.(e);
    };
    mountAssistant({
      target: hostRef.current,
      manifestUrl,
      baseModel,
      title,
      siteName,
      workerUrl,
      onEvent: toHandler(emit, {}),
      ...chatOpts,
    }).then(({ element }) => {
      if (!cancelled) elRef.current = element;
    }).catch((err) => {
      if (!cancelled) onError?.({ message: String(err?.message ?? err) });
    });
    return () => {
      cancelled = true;
      elRef.current = null;
      if (hostRef.current) hostRef.current.replaceChildren();
    };
    // Intentionally mount-once: model changes should remount via key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={hostRef} data-site-chat-host="" />;
});

export default SiteChat;
