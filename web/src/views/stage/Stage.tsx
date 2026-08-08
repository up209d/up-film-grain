import {
  ArrowUturnLeftIcon,
  ArrowUturnRightIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "@heroicons/react/24/outline";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { type ExportJob, type ImageMeta } from "../../services/api";
import type { Compare } from "../../models/types";
import {
  FIT_PADDING,
  FIT_SNAP,
  FRAME_DEFAULT,
  FRAME_MAX,
  FRAME_SHADOW_BLUR,
  FRAME_SHADOW_DROP,
  FRAME_SHADOW_ROOM,
  PINCH_RATE,
  WHEEL_RATE,
  ZOOM_STEPS,
  boardTones,
} from "../../models/constants";

/** The preview canvas: two compare modes, and all of the zooming.
 *
 *  Zoom and pan are purely a display transform here. The server renders the
 *  whole source at full resolution once per parameter change and the browser
 *  scales it, so navigating costs nothing and never waits on a render. It also
 *  removes an entire class of bug: the old server-side crop had to snap its
 *  read origin to whole working pixels, because a crop starting mid-pixel
 *  resolves on a different grid phase than a whole-image downscale and shows a
 *  half-pixel shift on hard edges. Nothing samples anything here, so any zoom
 *  value is now safe -- the steps below are for feel, not correctness.
 *
 *  The trade is at the other end: fitting a full-resolution render into the
 *  viewport is a browser downscale, so fine grain averages away on screen.
 *  Zoom to 100% to judge grain; that view is exact. */
function Stage(props: {
  meta: ImageMeta | null;
  previewUrl: string | null;
  sourceUrl: string | null;
  compare: Compare;
  onCompare: (c: Compare) => void;
  split: number;
  onSplit: (v: number) => void;
  showBefore: boolean;
  onShowBefore: (v: boolean) => void;
  previewFull: boolean;
  onFile: (f: File) => void;
  rendering: boolean;
  job: ExportJob | null;
  /** Lightness of the chequerboard behind the photo, 0..100. A view control
   *  like the mount, and driven from the panel because it is something you set
   *  once for a photo rather than reach for while working. */
  bgLightness: number;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  /** Bottom-right overlay slot. Deliberately opaque to this component -- it
   *  carries the export controls, which have nothing to do with the view, and
   *  keeping them a `ReactNode` is what stops the stage from having to know
   *  about export jobs to give them a corner to sit in. */
  corner?: React.ReactNode;
}) {
  const { meta, previewUrl, sourceUrl, compare, previewFull, rendering, job } =
    props;
  // The two chequer tones, handed to CSS as custom properties so the gradient
  // itself stays in the stylesheet. Cast because `style` is typed to known CSS
  // properties and custom ones are not among them.
  const board = boardTones(props.bgLightness) as React.CSSProperties;
  // Holding B peeks at the original outright, which is the wipe pushed all the
  // way over rather than a separate mode -- so it is resolved here, once,
  // instead of every consumer remembering to check both.
  const split = props.showBefore ? 0 : props.split;
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ x: number; y: number; cx: number; cy: number } | null>(
    null,
  );
  // The wheel handler's memory of an in-flight zoom excursion that is
  // currently snap-displayed as Fit -- see the wheel handler below for why
  // this has to persist independent of the displayed zoom. Cleared wherever
  // Fit is set for a reason *other* than the wheel handler's own snap, so a
  // fresh "go to Fit" never inherits a stale excursion from a previous scroll.
  const wheelContRef = useRef<number | null>(null);
  // null = fit: follow the container instead of holding a fixed factor, so
  // resizing the window keeps the whole frame visible.
  const [zoom, setZoom] = useState<number | null>(null);
  const [center, setCenter] = useState({ x: 0.5, y: 0.5 });
  const [pane, setPane] = useState({ w: 0, h: 0 });
  // A mount border and a drop shadow around the photo. Purely a view control,
  // which is why it lives here and on the viewbar rather than in `params.py`:
  // it changes nothing about the render and nothing about an export. The point
  // is judging the picture, not decorating it -- a photograph butted straight
  // against a dark panel reads darker and flatter than it is, and the edge of
  // the frame stops being visible at all where the picture goes to black.
  const [frame, setFrame] = useState(true);
  const [frameWidth, setFrameWidth] = useState(FRAME_DEFAULT);
  // Both overlay bars sit on top of the photograph, so each folds away to just
  // its own chevron for an unobstructed look at the picture. Expanded is the
  // default: the controls are the reason the bars exist, and a collapsed bar
  // you did not collapse yourself reads as missing rather than as tidy.
  const [topBarOpen, setTopBarOpen] = useState(true);
  const [bottomBarOpen, setBottomBarOpen] = useState(true);

  // A new image inherits neither the old pan nor the old magnification -- a
  // corner crop of the last photo is never where you want to land. The bars
  // come back for the same reason: opening a photo is the start of a session
  // of work on it, and the controls should be in front of you for it.
  useEffect(() => {
    setCenter({ x: 0.5, y: 0.5 });
    wheelContRef.current = null;
    setZoom(null);
    setTopBarOpen(true);
    setBottomBarOpen(true);
  }, [props.meta?.id]);

  // A pane, not the stage: in side-by-side each image gets half the width, so
  // fit has to be computed against what the image is actually shown in.
  // Measured synchronously before paint as well as on resize, or the first
  // frame after an upload is laid out against a zero-width pane.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => {
      const n = el.querySelector(".pane");
      if (n) setPane({ w: n.clientWidth, h: n.clientHeight });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [compare, !!meta]);

  const iw = meta?.width ?? 1;
  const ih = meta?.height ?? 1;
  // Fit means "the whole thing is visible", and with a mount on, the mount is
  // part of the whole thing -- so the room it needs comes out of the fit before
  // the zoom is computed. Reserved on both axes because the border is drawn on
  // all four sides. Left out of `place()`'s clamping, which works in image
  // coordinates: the mount hangs outside the image box and never moves it.
  //
  // FIT_PADDING is added unconditionally, mount or no mount: Fit used to size
  // the image to the exact pane, leaving no margin to judge it against the
  // panel behind it.
  const inset = FIT_PADDING + (frame ? frameWidth + FRAME_SHADOW_ROOM : 0);
  const fitZoom =
    Math.min(
      Math.max(pane.w - 2 * inset, 1) / iw,
      Math.max(pane.h - 2 * inset, 1) / ih,
    ) || 1;
  const eff = zoom ?? fitZoom;
  const dw = iw * eff;
  const dh = ih * eff;
  const canPan = dw > pane.w + 1 || dh > pane.h + 1;

  // Live geometry for the wheel handler below. That listener is attached once
  // and by hand, so without this it would close over whatever zoom happened to
  // be current when it was attached and every notch would zoom from the same
  // starting point.
  const geom = useRef({
    eff: 1,
    fit: 1,
    iw: 1,
    ih: 1,
    pane: { w: 0, h: 0 },
    zoomIsNull: true,
  });
  geom.current = { eff, fit: fitZoom, iw, ih, pane, zoomIsNull: zoom === null };

  // Scroll to zoom, anchored on the pointer.
  //
  // Attached by hand rather than as an `onWheel` prop because React registers
  // wheel listeners as *passive*, where preventDefault is a no-op -- so the
  // React version would zoom the photo and scroll the page underneath it at
  // the same time.
  //
  // Anchoring is the part that makes it feel like anything: the image point
  // under the cursor has to still be under the cursor afterwards, or zooming
  // in on a detail walks it off the screen and you pan it back every time.
  // The anchor is read from the frame's own rect rather than recomputed from
  // `center`, so it accounts for the clamping in place() for free.
  useEffect(() => {
    const host = wrapRef.current?.querySelector(".panes");
    if (!host) return;

    const onWheel = (e: WheelEvent) => {
      const paneEl = (e.target as Element | null)?.closest?.(".pane");
      const frame = paneEl?.querySelector(".frame") as HTMLElement | null;
      if (!paneEl || !frame) return;
      e.preventDefault();

      const g = geom.current;
      // deltaY is in pixels, lines or pages depending on the browser and the
      // device; Firefox reports lines for a mouse wheel.
      const dy =
        e.deltaMode === 1
          ? e.deltaY * 16
          : e.deltaMode === 2
            ? e.deltaY * (g.pane.h || 400)
            : e.deltaY;
      const rate = e.ctrlKey ? PINCH_RATE : WHEEL_RATE;
      const hi = ZOOM_STEPS[ZOOM_STEPS.length - 1];
      // Same floor as the - button (changed 2026-08-04, on request): the wheel
      // used to bottom out at Fit and required the button for anything smaller,
      // which is a floor a continuous gesture should not have needed to defer to
      // a click for.
      const lo = ZOOM_STEPS[0];
      // The true starting point for this tick: the in-flight excursion if the
      // display is currently snapped to Fit and one is recorded, otherwise the
      // displayed value itself (there is nothing hidden to recover).
      const from =
        g.zoomIsNull && wheelContRef.current !== null
          ? wheelContRef.current
          : g.eff;
      const next = Math.min(hi, Math.max(lo, from * Math.exp(-dy * rate)));
      if (Math.abs(next - from) < 1e-6) return;

      const fr = frame.getBoundingClientRect();
      const pr = paneEl.getBoundingClientRect();
      // Where the cursor is on the image, 0..1, and where it is in the pane.
      const u = (e.clientX - fr.left) / fr.width;
      const v = (e.clientY - fr.top) / fr.height;
      const px = e.clientX - pr.left;
      const py = e.clientY - pr.top;
      // place() puts the image at left = pane.w/2 - center.x*dw, so holding
      // `u` at `px` across the zoom means center.x = u + (pane.w/2 - px)/dw.
      // Clamped to the image; place() clamps the placement again on top.
      const dw2 = g.iw * next;
      const dh2 = g.ih * next;
      setCenter({
        x: Math.min(1, Math.max(0, u + (g.pane.w / 2 - px) / dw2)),
        y: Math.min(1, Math.max(0, v + (g.pane.h / 2 - py) / dh2)),
      });
      // Two-sided now that scrolling out is not capped at Fit -- see FIT_SNAP.
      const snapped = Math.abs(next - g.fit) <= g.fit * FIT_SNAP;
      // Keep the true position alive under the snap so the *next* tick can
      // still tell it apart from a fresh arrival at Fit -- see wheelContRef.
      wheelContRef.current = snapped ? next : null;
      setZoom(snapped ? null : next);
    };

    host.addEventListener("wheel", onWheel as EventListener, {
      passive: false,
    });
    return () => host.removeEventListener("wheel", onWheel as EventListener);
  }, [compare, !!meta]);

  const stepZoom = (dir: 1 | -1) => {
    const next =
      dir > 0
        ? ZOOM_STEPS.find((z) => z > eff * 1.001)
        : [...ZOOM_STEPS].reverse().find((z) => z < eff * 0.999);
    if (next !== undefined) setZoom(next);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!canPan) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, cx: center.x, cy: center.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    // One screen pixel is 1/dw of the image's displayed width, so the same
    // delta moves both panes by the same fraction of the same image -- which
    // is what keeps the two sides locked together in side-by-side.
    setCenter({
      x: Math.min(1, Math.max(0, d.cx - (e.clientX - d.x) / dw)),
      y: Math.min(1, Math.max(0, d.cy - (e.clientY - d.y) / dh)),
    });
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  if (!meta) {
    return (
      <div
        className="stage empty"
        style={board}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f) props.onFile(f);
        }}
      >
        <div className="drop">
          <strong>Drop a photo here</strong>
          <span>or use “Open image”. JPEG or PNG, up to 30MB.</span>
        </div>
      </div>
    );
  }

  // Place the frame so the source point at `center` sits at the middle of the
  // pane, clamped so a pannable image cannot be dragged off its own edges and
  // a smaller-than-pane image simply centres.
  const place = (): React.CSSProperties => {
    const left =
      dw <= pane.w
        ? (pane.w - dw) / 2
        : Math.min(0, Math.max(pane.w - dw, pane.w / 2 - center.x * dw));
    const top =
      dh <= pane.h
        ? (pane.h - dh) / 2
        : Math.min(0, Math.max(pane.h - dh, pane.h / 2 - center.y * dh));
    return {
      width: dw,
      height: dh,
      left,
      top,
      // Past 2x the honest thing is to show the actual pixels rather than let
      // the browser invent smooth ones between them.
      imageRendering: eff >= 2 ? "pixelated" : "auto",
    };
  };

  /** The mount border and its shadow, as one element behind the images.
   *
   *  Its own element rather than a border on the `<img>`, for two reasons. The
   *  overlay mode draws the wipe by clipping the result image with `clipPath`,
   *  and clipPath clips a box-shadow with it -- so a ring on that image would
   *  lose whichever side was wiped away. And a CSS border would grow the box
   *  past the `dw x dh` that every coordinate in here is derived from, putting
   *  the pointer-anchored zoom half a border out and dragging `place()`'s
   *  clamping with it.
   *
   *  Drawn as two spread shadows rather than a border and a filter, so it
   *  occupies no layout at all: geometry stays exactly `place()`'s. The drop
   *  shadow carries the same spread as the border, or it would be laid down
   *  from the image's edge and sit *underneath* the opaque mount instead of
   *  around it. */
  const mount = () =>
    frame ? (
      <div
        className="mount"
        style={{
          ...place(),
          boxShadow:
            `0 0 0 ${frameWidth}px var(--mount), ` +
            `0 ${FRAME_SHADOW_DROP}px ${FRAME_SHADOW_BLUR}px ` +
            `${frameWidth}px rgba(0,0,0,.7)`,
        }}
      />
    ) : null;

  // The proxy resolves detail only up to its own resolution; past that the
  // browser is enlarging it and grain is not being shown honestly. Say so
  // rather than letting a soft preview read as a soft result.
  const proxyLimit = (meta?.proxy_width ?? 0) / iw;
  const softened = !previewFull && eff > proxyLimit * 1.05;

  // One bar for compare mode, the wipe, the zoom and the mount -- grouped left
  // to right in the order you reach for them, with the wipe next to the mode
  // that owns it.
  //
  // Undo and redo are the one thing here that changes the render rather than
  // the view, and they are on this bar anyway: undo is judged by looking at the
  // photo, so putting it in the panel would mean reaching away from the thing
  // you are watching. They lead the bar, ahead of their own separator, so they
  // read as a group that is not part of the view controls.
  const isExporting = job && job.status !== "done" && job.status !== "error";
  // The collapse chevron closes the bar, after everything it collapses.
  const barToggle = (
    <button
      className="seg icon bartoggle"
      onClick={() => setTopBarOpen((x) => !x)}
      title={topBarOpen ? "Collapse the view bar" : "Expand the view bar"}
      aria-label={topBarOpen ? "Collapse the view bar" : "Expand the view bar"}
      aria-expanded={topBarOpen}
    >
      {topBarOpen ? (
        <ChevronUpIcon aria-hidden="true" />
      ) : (
        <ChevronDownIcon aria-hidden="true" />
      )}
    </button>
  );
  const bar = (
    <div className={`viewbar${topBarOpen ? "" : " collapsed"}`}>
      {!topBarOpen ? null : (
        <>
          <div
            className={`spinner-placeholder ${rendering || isExporting ? "active" : ""}`}
          >
            <div className="spinner-icon" />
          </div>
          {softened && (
            <span
              className="fid"
              title="Enlarged beyond the proxy's resolution — press Render 1:1 to judge grain"
            >
              proxy
            </span>
          )}
          {/* Heroicons rather than the ↶/↷ arrows these were: the glyphs are a
          curved arrow with no home to return to and read as "rotate" at least
          as readily as "undo". The u-turn pair is the shape every editor uses
          for this, so it needs no title to be understood. */}
          <button
            className="seg icon"
            onClick={props.onUndo}
            disabled={!props.canUndo}
            title="Undo the last adjustment"
            aria-label="Undo"
          >
            <ArrowUturnLeftIcon aria-hidden="true" />
          </button>
          <button
            className="seg icon"
            onClick={props.onRedo}
            disabled={!props.canRedo}
            title="Redo"
            aria-label="Redo"
          >
            <ArrowUturnRightIcon aria-hidden="true" />
          </button>
          <span className="vsep" />
          <button
            className={compare === "overlay" ? "seg on" : "seg"}
            onClick={() => props.onCompare("overlay")}
            title="Wipe the result over the original"
          >
            Overlay
          </button>
          <button
            className={compare === "side" ? "seg on" : "seg"}
            onClick={() => props.onCompare("side")}
            title="Original and result side by side, panning and zooming together"
          >
            Side
          </button>
          {compare === "overlay" && (
            <>
              <button
                className={props.showBefore ? "swap on" : "swap"}
                onClick={() => props.onShowBefore(!props.showBefore)}
                title="Swap before/after — or hold B"
              >
                {props.showBefore ? "Before" : "After"}
              </button>
              <input
                className="wipe"
                type="range"
                min={0}
                max={1}
                step={0.001}
                value={props.split}
                disabled={props.showBefore}
                onChange={(e) => props.onSplit(Number(e.target.value))}
                title="Wipe"
              />
            </>
          )}
          <span className="vsep" />
          <button
            className={zoom === null ? "seg on" : "seg"}
            onClick={() => {
              wheelContRef.current = null;
              setZoom(null);
            }}
          >
            Fit
          </button>
          <button
            className="seg icon"
            onClick={() => stepZoom(-1)}
            disabled={eff <= ZOOM_STEPS[0] * 1.001}
            title="Zoom out"
          >
            −
          </button>
          <button
            className="seg zoomval"
            onClick={() => setZoom(eff === 1 ? null : 1)}
            title="Zoom to 1:1"
          >
            {Math.round(eff * 100)}%
          </button>
          <button
            className="seg icon"
            onClick={() => stepZoom(1)}
            disabled={eff >= ZOOM_STEPS[ZOOM_STEPS.length - 1] * 0.999}
            title="Zoom in"
          >
            +
          </button>
          <span className="vsep" />
          {/* Third group: the mount. A view control like everything else on this
          bar -- it changes nothing that gets rendered or exported. The width
          slider appears only with the frame on, matching how the wipe follows
          the mode that owns it, so the bar does not carry a dead control. */}
          <button
            className={frame ? "seg on" : "seg"}
            onClick={() => setFrame(!frame)}
            title="Show the photo on a mount, with a drop shadow"
          >
            Frame
          </button>
          {frame && (
            <input
              className="framew"
              type="range"
              min={0}
              max={FRAME_MAX}
              step={1}
              value={frameWidth}
              onChange={(e) => setFrameWidth(Number(e.target.value))}
              title={`Frame width — ${frameWidth}px on screen`}
            />
          )}
        </>
      )}
      {barToggle}
    </div>
  );

  const cornerbar = props.corner ? (
    <div className={`cornerbar${bottomBarOpen ? "" : " collapsed"}`}>
      {bottomBarOpen && props.corner}
      <button
        className="seg icon bartoggle"
        onClick={() => setBottomBarOpen((x) => !x)}
        title={
          bottomBarOpen ? "Collapse the export bar" : "Expand the export bar"
        }
        aria-label={
          bottomBarOpen ? "Collapse the export bar" : "Expand the export bar"
        }
        aria-expanded={bottomBarOpen}
      >
        {bottomBarOpen ? (
          <ChevronDownIcon aria-hidden="true" />
        ) : (
          <ChevronUpIcon aria-hidden="true" />
        )}
      </button>
    </div>
  ) : null;

  if (compare === "side") {
    return (
      <div
        className={`stage side ${canPan ? "grab" : ""}`}
        style={board}
        ref={wrapRef}
      >
        {bar}
        {/* Both panes read the same zoom and centre, so one drag anywhere in
            the stage moves both images identically. */}
        <div
          className="panes"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <div className="pane">
            <span className="tag">Before</span>
            {mount()}
            {sourceUrl && (
              <img
                className="frame"
                style={place()}
                src={sourceUrl}
                alt="before"
                draggable={false}
              />
            )}
          </div>
          <div className="pane">
            <span className="tag">After</span>
            {mount()}
            {previewUrl && (
              <img
                className="frame"
                style={place()}
                src={previewUrl}
                alt="after"
                draggable={false}
              />
            )}
          </div>
        </div>
        {cornerbar}
      </div>
    );
  }

  return (
    <div
      className={`stage ${canPan ? "grab" : ""}`}
      style={board}
      ref={wrapRef}
    >
      {bar}
      <div
        className="panes"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div className="pane">
          {mount()}
          {sourceUrl && (
            <img
              className="frame"
              style={place()}
              src={sourceUrl}
              alt="before"
              draggable={false}
            />
          )}
          {previewUrl && (
            <img
              className="frame"
              style={{
                ...place(),
                clipPath: `inset(0 0 0 ${(1 - split) * 100}%)`,
              }}
              src={previewUrl}
              alt="after"
              draggable={false}
            />
          )}
          {split > 0.001 && split < 0.999 && (
            <div
              className="divider"
              style={{
                left: `${(place().left as number) + (1 - split) * dw}px`,
              }}
            />
          )}
        </div>
      </div>
      {cornerbar}
    </div>
  );
}

export default Stage;
