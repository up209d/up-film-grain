/** A jump list of every pipeline section, on the export bar beside the
 *  collapse switch.
 *
 *  The panel is one long scroll of a dozen-odd sections and reaching one near
 *  the bottom means scrolling past everything above it. This is the index: the
 *  same list `GROUPS` defines, in pipeline order, and picking one scrolls the
 *  panel to it. Like the collapse switch it sits with, it changes nothing about
 *  the render -- which is why it belongs on this bar and not in the panel it
 *  drives.
 *
 *  It opens *upwards*: the bar is anchored to the bottom of the stage, so a
 *  menu dropped downwards from it would open off the window.
 */

import { Bars3Icon } from "@heroicons/react/24/outline";
import { useEffect, useRef, useState } from "react";

export default function SectionMenu(props: {
  groups: string[];
  onPick: (group: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  // Dismissal: a click anywhere else, or Escape. `pointerdown` rather than
  // `click` so the menu is gone by the time a control underneath it reacts,
  // and the containment test is what keeps the button itself from closing and
  // reopening in the same gesture.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="secmenu" ref={ref}>
      <button
        className={`seg icon secmenu-btn${open ? " on" : ""}`}
        onClick={() => setOpen((x) => !x)}
        disabled={!props.groups.length}
        title="Jump to a pipeline section"
        aria-label="Jump to a pipeline section"
        aria-expanded={open}
        aria-haspopup="menu"
      >
        <Bars3Icon aria-hidden="true" />
      </button>
      {open && (
        <div className="secmenu-list" role="menu">
          {props.groups.map((g) => (
            <button
              key={g}
              className="secmenu-item"
              role="menuitem"
              onClick={() => {
                props.onPick(g);
                setOpen(false);
              }}
            >
              {g}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
