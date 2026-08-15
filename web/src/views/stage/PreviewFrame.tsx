import { useEffect, useRef, useState } from "react";

/** The photo on the stage, drawn either by the browser or through a mip chain.
 *
 *  A browser asked to fit a 2400px image into a 400px box resamples it **once**.
 *  That single step is not a box average over the 6x6 source pixels behind each
 *  destination pixel -- it samples far fewer -- so content above the destination
 *  grid's Nyquist folds down instead of averaging away. Grain is made of
 *  essentially nothing else, and the fold moves as you zoom, which is what makes
 *  a zoomed-out preview crawl and shimmer in a way the file does not.
 *
 *  The filtered path draws the same picture through successive **exact
 *  halvings**. A 2x reduction averages each 2x2 block and can therefore alias
 *  nothing at all, and only the final step is a non-integer ratio -- held under
 *  2x, where the browser's own filter is correct.
 *
 *  **This changes the screen and never the file.** Nothing here touches a render
 *  or an export. The pixels are the ones already in `src`; only the filter that
 *  carries them to the display differs. It is a view control in the same sense
 *  the mount is, which is why it lives on the view bar and not in `params.py`.
 *
 *  Measured before it was built, and worth knowing before trusting it: on this
 *  engine's own output the two paths score the same. A flat plate rendered with
 *  Global Grain alone, downsampled 0.5x to 0.16x, measures an identical
 *  largest-spectral-peak ratio whether it goes through one naive step or through
 *  the halvings -- the layer carries no periodic component for either filter to
 *  fold. So this is here for the cases that measurement did not cover (a real
 *  browser's resampler is not the reference one that was simulated, and a
 *  photograph is not a flat plate), not because the artifact it removes was ever
 *  demonstrated in the render. */
export default function PreviewFrame(props: {
  src: string;
  alt: string;
  style: React.CSSProperties;
  /** Displayed size in CSS pixels -- `place()`'s `dw`/`dh`. Read here rather
   *  than off `style` so the drawing size never has to be parsed back out of a
   *  CSS string. */
  dw: number;
  dh: number;
  /** The view-bar toggle. False hands the picture straight to an `<img>`, which
   *  is the browser's own behaviour and the thing this is measured against. */
  filtered: boolean;
}) {
  const { src, alt, style, dw, dh, filtered } = props;
  // A retina display asks for twice the pixels, and downsampling to the CSS size
  // and letting the compositor double it back up would throw away exactly the
  // detail this component exists to preserve.
  const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
  const tw = Math.max(1, Math.round(dw * dpr));
  const th = Math.max(1, Math.round(dh * dpr));

  const [chain, setChain] = useState<HTMLCanvasElement[] | null>(null);
  const ref = useRef<HTMLCanvasElement | null>(null);

  // The chain is a function of the image alone, so it is built once per preview
  // and reused for every zoom. Rebuilding it per zoom would be the expensive
  // version of exactly the wrong thing -- the whole point is that the halvings
  // are shared.
  useEffect(() => {
    if (!filtered) {
      setChain(null);
      return;
    }
    let dead = false;
    const img = new Image();
    img.onload = () => {
      if (dead) return;
      const out: HTMLCanvasElement[] = [];
      let w = img.naturalWidth;
      let h = img.naturalHeight;
      if (!w || !h) return;
      const base = document.createElement("canvas");
      base.width = w;
      base.height = h;
      base.getContext("2d")?.drawImage(img, 0, 0);
      out.push(base);
      // Down to 8px rather than to 1: nothing on this stage is ever displayed
      // smaller, and the last few levels cost a canvas each to never be picked.
      while (w > 8 && h > 8) {
        const nw = Math.max(1, w >> 1);
        const nh = Math.max(1, h >> 1);
        const c = document.createElement("canvas");
        c.width = nw;
        c.height = nh;
        const g = c.getContext("2d");
        if (!g) break;
        g.imageSmoothingEnabled = true;
        g.imageSmoothingQuality = "high";
        g.drawImage(out[out.length - 1], 0, 0, nw, nh);
        out.push(c);
        w = nw;
        h = nh;
      }
      setChain(out);
    };
    img.src = src;
    return () => {
      dead = true;
    };
  }, [src, filtered]);

  // Only meaningful while the picture is being *reduced*. Enlarged past its own
  // resolution there is nothing to prefilter, and the `<img>` path keeps the
  // deliberate `imageRendering: pixelated` that `place()` sets past 2x.
  const shrinking = !!chain && tw < chain[0].width;

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !chain || !shrinking) return;
    // The smallest level still at least as large as the target, so the final
    // draw is a reduction of under 2x -- inside where the browser's filter is
    // right, which is the entire reason for the chain.
    let i = 0;
    while (i + 1 < chain.length && chain[i + 1].width >= tw) i++;
    cv.width = tw;
    cv.height = th;
    const g = cv.getContext("2d");
    if (!g) return;
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    g.drawImage(chain[i], 0, 0, tw, th);
  }, [chain, shrinking, tw, th]);

  if (!shrinking) {
    return (
      <img
        className="frame"
        style={style}
        src={src}
        alt={alt}
        draggable={false}
      />
    );
  }
  // `width`/`height` are set on the element in the effect above and deliberately
  // not passed as props: they are the backing store in device pixels, while
  // `style` carries the CSS size in layout pixels, and letting React own the
  // attributes would fight the effect for them every render.
  return (
    <canvas
      ref={ref}
      className="frame"
      style={style}
      role="img"
      aria-label={alt}
    />
  );
}
