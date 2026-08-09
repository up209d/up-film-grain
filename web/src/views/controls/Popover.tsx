/** A button that opens a panel anchored to itself, and the dismissal rules that
 *  go with it.
 *
 *  Lifted out of `SectionMenu` on 2026-08-09, when the LUT picker needed the
 *  same shell. Nothing here is about *what* is in the panel -- that is the
 *  caller's children -- so the two menus in the app share the open/close
 *  behaviour without sharing their contents.
 *
 *  Two things it exists to get right, both of which were bugs before they were
 *  comments:
 *
 *  * Dismissal listens for `pointerdown`, not `click`, so the panel is gone by
 *    the time a control underneath it reacts. With `click` the panel is still
 *    on screen when the thing below it takes the press.
 *  * The containment test is what stops the trigger from closing and reopening
 *    in the same gesture: without it the outside handler closes the panel and
 *    the button's own `onClick` immediately reopens it.
 */

import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

/** Gap between trigger and panel, and the margin kept off the window edge. */
const GAP = 6;
const EDGE = 8;
/** Floor on the panel's height. A trigger jammed against the bottom of the
 *  window has almost no room on either side, and a panel clamped to 20px is no
 *  more usable than one running off the screen -- below this it is better to
 *  overhang slightly and stay scrollable. */
const MIN_PANEL = 180;
/** Ceiling on the panel's height, whatever room there is.
 *
 *  Fitting the window is a *constraint*, not a goal: on a tall display the LUT
 *  list would otherwise run 900px down the screen, which is a wall of names
 *  rather than a menu, and reading it means moving your eyes further than
 *  scrolling it does. 560 is about twenty rows — enough to see a folder's worth
 *  of LUTs at once, short enough to take in as one object. Whatever does not
 *  fit scrolls. */
const MAX_PANEL = 560;

/** Keys the browser would otherwise use to scroll the page. */
const SCROLL_KEYS = new Set([
  "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " ",
]);

/** Inside a text field every one of those keys means something else -- Home and
 *  End move the caret, and a swallowed space cannot be typed into a search box.
 *  The panel is welcome to act on them; the browser's scroll is what is being
 *  stopped, and that never applies while someone is typing. */
const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement &&
  (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);

export default function Popover(props: {
  /** What goes in the button. A string, or an icon element. */
  trigger: React.ReactNode;
  /** Rendered only while open, so a 300-item list costs nothing when shut. */
  children: (close: () => void) => React.ReactNode;
  /** Which way the panel *prefers* to open. A preference, not an instruction —
   *  it is overridden when the preferred side cannot fit the panel and the
   *  other side has more room. `up` is what anything in the corner bar wants:
   *  that bar sits 10px off the bottom of the stage. */
  drop?: "up" | "down";
  /** Which edge the panel lines up with. */
  align?: "left" | "right";
  disabled?: boolean;
  title?: string;
  ariaLabel?: string;
  /** Extra classes on the button. The shell owns layout, not appearance. */
  buttonClass?: string;
  /** Extra classes on the panel, for the few places that need a width. */
  panelClass?: string;
  /** ARIA role for the panel. `menu` is right for a list of items and wrong the
   *  moment there is a text field in it -- a menu may not contain a textbox --
   *  so a searchable panel says `dialog` and puts the `menu` on the list
   *  inside. */
  role?: "menu" | "dialog";
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
      } else if (SCROLL_KEYS.has(e.key) && !isTyping(e.target)) {
        // While a panel is open the navigation keys belong to it, and letting
        // one reach the browser scrolls whatever is behind it -- which for a
        // menu in the parameter sidebar means the sidebar walks away under the
        // open menu. Swallowed here rather than in the panel because focus is
        // not always inside it: a menu with no search box leaves focus on the
        // trigger, and a handler bound to the panel would never see the key.
        e.preventDefault();
      }
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // A disabled trigger that is somehow still open would leave a panel with no
  // way to shut it, which happens for real: the LUT list empties while its menu
  // is up if the server drops out.
  useEffect(() => {
    if (props.disabled) setOpen(false);
  }, [props.disabled]);

  /** Which side the panel actually opens on, and how tall it may be.
   *
   *  Measured rather than declared, and this is not a refinement -- a fixed
   *  direction is a bug you only see at some scroll positions. The LUT menu
   *  shipped `drop="down"` with a `60vh` list: with the trigger 30px off the
   *  bottom of an 800px window the panel ran 130px past the edge and 22 of its
   *  33 rows were unreachable, so every root LUT was pickable and nothing
   *  inside a folder was. The stated `drop` is kept unless the other side has
   *  genuinely more room, so the corner bar still opens upwards by default.
   *
   *  `max` is handed to the panel as an inline height cap -- the smaller of
   *  `MAX_PANEL` and the room actually available. The list inside it flexes, so
   *  the panel never exceeds either bound and the overflow becomes scroll
   *  instead of clipping. */
  const [place, setPlace] = useState({ drop: props.drop ?? "down", max: 0 });

  useLayoutEffect(() => {
    if (!open) return;
    const measure = () => {
      const r = ref.current?.getBoundingClientRect();
      if (!r) return;
      const below = window.innerHeight - r.bottom - GAP - EDGE;
      const above = r.top - GAP - EDGE;
      const pref = props.drop ?? "down";
      const [wanted, other] = pref === "up" ? [above, below] : [below, above];
      // Keep the preferred side when it can show a full-height panel -- once
      // both sides clear the ceiling the extra room buys nothing, and flipping
      // on a few pixels' difference would make the menu jump about as the panel
      // is scrolled. Only give it up when the other side is genuinely roomier.
      const keep = wanted >= MAX_PANEL || wanted >= other;
      const up = pref === "up" ? keep : !keep;
      setPlace({
        drop: up ? "up" : "down",
        max: Math.min(MAX_PANEL, Math.max(MIN_PANEL, up ? above : below)),
      });
    };
    measure();
    // `true` because the thing that scrolls is the parameter panel, not the
    // window -- a listener on `window` alone never fires and the panel keeps a
    // height measured for where the trigger used to be.
    window.addEventListener("scroll", measure, true);
    window.addEventListener("resize", measure);
    return () => {
      window.removeEventListener("scroll", measure, true);
      window.removeEventListener("resize", measure);
    };
  }, [open, props.drop]);

  const align = props.align ?? "left";

  return (
    <div className="menu" ref={ref}>
      <button
        type="button"
        className={`menu-btn${open ? " on" : ""}${
          props.buttonClass ? ` ${props.buttonClass}` : ""
        }`}
        onClick={() => setOpen((x) => !x)}
        disabled={props.disabled}
        title={props.title}
        aria-label={props.ariaLabel}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-controls={open ? panelId : undefined}
      >
        {props.trigger}
      </button>
      {open && (
        <div
          id={panelId}
          className={`menu-list menu-${place.drop} menu-${align}${
            props.panelClass ? ` ${props.panelClass}` : ""
          }`}
          style={{ maxHeight: place.max || undefined }}
          role={props.role ?? "menu"}
        >
          {props.children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
