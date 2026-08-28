import { useRef, useState } from "react";
import { createPortal } from "react-dom";

/** Slider title plus a help badge, sharing one tooltip.
 *
 *  The native `title` attribute this replaced was effectively invisible: it
 *  needs a long hover, cannot be triggered by keyboard, and vanishes on any
 *  movement. This one is positioned against the viewport rather than the
 *  scrolling panel, so it is never clipped by the panel's overflow.
 *
 *  The tip is rendered into `document.body` rather than in place. `position:
 *  fixed` resolves against the nearest ancestor with a filter, transform or
 *  `backdrop-filter` -- and `.cornerbar` has `backdrop-filter: blur(6px)`, so
 *  the Export and Cache tooltips were being laid out against a 30px-tall bar in
 *  the corner and landing off-screen. A portal has no such ancestor.
 *
 *  The whole title is the hover target, not just the badge -- the label already
 *  carries `cursor: help`, and a 15px circle is far too small a thing to have
 *  to find before the explanation will show. The badge stays as the visible
 *  affordance and as the keyboard/tap target, and is what the tooltip is
 *  positioned against so its placement does not depend on label length. */
function Help({ text, label }: { text: string; label: string }) {
  const [at, setAt] = useState<{
    top?: number;
    bottom?: number;
    left: number;
  } | null>(null);
  const ref = useRef<HTMLButtonElement | null>(null);

  const open = () => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    // Sit to the left of the panel, anchored to the badge. A tooltip's height
    // is not known until it has rendered, so clamping a `top` cannot keep a
    // long one on screen -- the cache readout runs to several hundred pixels
    // and hung off the bottom of the window from the corner bar. Anchoring by
    // the edge the badge is nearest instead means the browser does the work:
    // in the lower half of the window the tip grows *upwards* from a fixed
    // `bottom`, so it cannot reach the bottom edge whatever its height.
    const left = Math.max(8, r.left - 268);
    const below = window.innerHeight - r.bottom;
    setAt(
      r.top > window.innerHeight / 2
        ? { bottom: Math.max(8, below - 8), left }
        : { top: Math.max(8, r.top - 8), left },
    );
  };
  const close = () => setAt(null);

  if (!text) return <span className="title">{label}</span>;
  return (
    <span className="title" onMouseEnter={open} onMouseLeave={close}>
      {label}
      <button
        ref={ref}
        className="help"
        type="button"
        aria-label={`What does ${label} do?`}
        onFocus={open}
        onBlur={close}
        onClick={(e) => {
          e.preventDefault();
          if (at) close();
          else open();
        }}
      >
        ?
      </button>
      {at &&
        createPortal(
          <div
            className="tip"
            style={{ top: at.top, bottom: at.bottom, left: at.left }}
            role="tooltip"
          >
            {text}
          </div>,
          document.body,
        )}
    </span>
  );
}

export default Help;
