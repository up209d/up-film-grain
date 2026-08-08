/** The render request lifecycle: what to ask for, when to ask, and cleaning up
 *  the object URLs afterwards.
 *
 *  Note what is *not* in the request: no zoom, no viewport, no pan. The server
 *  renders the whole source and the browser scales it, so navigating the
 *  preview never re-renders and never touches the network.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { DEBOUNCE_MS } from "../models/constants";
import type { Values } from "../models/types";
import {
  fetchSource,
  renderPreview,
  type ImageMeta,
  type ViewRequest,
} from "../services/api";

export function usePreview(opts: {
  meta: ImageMeta | null;
  applied: Values;
  supersample: number;
  referenceMp: number | null;
  scaleToRef: boolean;
  lut: string | null;
  onError: (msg: string | null) => void;
  onDevice: (d: string) => void;
}) {
  const { meta, applied, supersample, referenceMp, scaleToRef, lut } = opts;

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [renderMs, setRenderMs] = useState(0);
  /** Whether what is on screen is the full-resolution render or the proxy.
   *  Any parameter change drops back to the proxy, so this doubles as "is the
   *  1:1 render still current". */
  const [previewFull, setPreviewFull] = useState(false);
  const [renderingFull, setRenderingFull] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const urlsRef = useRef<string[]>([]);
  const srcUrlRef = useRef<string | null>(null);

  const errRef = useRef(opts.onError);
  errRef.current = opts.onError;
  const devRef = useRef(opts.onDevice);
  devRef.current = opts.onDevice;

  /** Track a *preview* object URL. Blobs stay in memory until revoked, so a
   *  long slider session would otherwise leak every intermediate render. */
  const track = (url: string) => {
    urlsRef.current.push(url);
    if (urlsRef.current.length > 4) {
      URL.revokeObjectURL(urlsRef.current.shift()!);
    }
    return url;
  };

  /** The source image is on its own lifecycle, not the preview LRU.
   *
   *  It used to share it, which was a slow-acting bug: the source is fetched
   *  once per upload while previews churn on every edit, so after four renders
   *  the source URL was the oldest entry and got revoked while an <img> was
   *  still pointing at it. The before/after compare went blank partway through
   *  a session, which looks like anything except an eviction policy. One URL,
   *  replaced explicitly. */
  const trackSource = (url: string | null) => {
    if (srcUrlRef.current) URL.revokeObjectURL(srcUrlRef.current);
    srcUrlRef.current = url;
    return url;
  };

  const viewBody = useCallback((): ViewRequest | null => {
    if (!meta) return null;
    return {
      id: meta.id,
      params: applied,
      supersample,
      reference_mp: scaleToRef ? referenceMp : null,
      lut,
    };
  }, [meta, applied, supersample, referenceMp, scaleToRef, lut]);

  // The untouched image is the same bytes for the life of an upload, so it is
  // fetched once here rather than riding along with every render.
  useEffect(() => {
    if (!meta) {
      setSourceUrl(trackSource(null));
      return;
    }
    const ac = new AbortController();
    fetchSource({ id: meta.id, params: {}, supersample: 1 }, ac.signal)
      .then((url) => setSourceUrl(trackSource(url)))
      .catch((e) => {
        if (e.name !== "AbortError") errRef.current(String(e.message ?? e));
      });
    return () => ac.abort();
  }, [meta]);

  /** One render path for both fidelities -- the only difference is `full`.
   *  Sharing `abortRef` is deliberate: touching a slider during a 1:1 render
   *  cancels it, because that render is about to be wrong anyway. */
  const render = useCallback((body: ViewRequest, full: boolean) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setRendering(true);
    setRenderingFull(full);
    renderPreview({ ...body, full }, ac.signal)
      .then((res) => {
        setPreviewUrl(track(res.url));
        setRenderMs(res.ms);
        setPreviewFull(res.full);
        if (res.device) devRef.current(res.device);
        errRef.current(null);
      })
      .catch((e) => {
        if (e.name !== "AbortError") errRef.current(String(e.message ?? e));
      })
      .finally(() => {
        if (!ac.signal.aborted) {
          setRendering(false);
          setRenderingFull(false);
        }
      });
  }, []);

  // Live edits always render the proxy. Full resolution is never automatic --
  // it is seconds of work, and firing it off every time a drag settles would
  // spend that on frames you are about to change again.
  useEffect(() => {
    const body = viewBody();
    if (!body) return;
    const t = setTimeout(() => render(body, false), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [viewBody, render]);

  const renderFull = () => {
    const body = viewBody();
    if (body) render(body, true);
  };

  /** Clear what is on screen -- called when a new photo is accepted, before
   *  its meta swaps in, so the old photo is never stretched to the new one's
   *  dimensions while the first render is in flight. */
  const clear = () => {
    setPreviewUrl(null);
    setSourceUrl(null);
    setPreviewFull(false);
    setRenderMs(0);
  };

  return {
    previewUrl, sourceUrl, rendering, renderMs, previewFull, renderingFull,
    renderFull, clear,
  };
}
