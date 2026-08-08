import { useRef, useState } from "react";

/** Slider title plus a help badge, sharing one tooltip.
 *
 *  The native `title` attribute this replaced was effectively invisible: it
 *  needs a long hover, cannot be triggered by keyboard, and vanishes on any
 *  movement. This one is positioned against the viewport rather than the
 *  scrolling panel, so it is never clipped by the panel's overflow.
 *
 *  The whole title is the hover target, not just the badge -- the label already
 *  carries `cursor: help`, and a 15px circle is far too small a thing to have
 *  to find before the explanation will show. The badge stays as the visible
 *  affordance and as the keyboard/tap target, and is what the tooltip is
 *  positioned against so its placement does not depend on label length. */
function Help({ text, label }: { text: string; label: string }) {
  const [at, setAt] = useState<{ top: number; left: number } | null>(null);
  const ref = useRef<HTMLButtonElement | null>(null);

  const open = () => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    // Sit to the left of the panel, vertically centred on the badge, clamped
    // so it cannot run off the top or bottom of the window.
    setAt({
      top: Math.min(Math.max(8, r.top - 8), window.innerHeight - 120),
      left: Math.max(8, r.left - 268),
    });
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
      {at && (
        <div className="tip" style={{ top: at.top, left: at.left }} role="tooltip">
          {text}
        </div>
      )}
    </span>
  );
}

export default Help;
