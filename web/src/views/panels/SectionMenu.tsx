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
 *
 *  Deliberately **not** a `SelectMenu`, though it looks like one: these are
 *  commands, not values. There is nothing here to be "currently selected", so a
 *  component whose whole shape is `value`/`onPick` would be carrying a null the
 *  entire time. It shares the shell -- `Popover`, which was extracted from this
 *  file -- and that is the part that had the behaviour worth sharing.
 */

import { Bars3Icon } from "@heroicons/react/24/outline";

import Popover from "../controls/Popover";

export default function SectionMenu(props: {
  groups: string[];
  onPick: (group: string) => void;
}) {
  return (
    <Popover
      trigger={<Bars3Icon aria-hidden="true" />}
      drop="up"
      align="right"
      disabled={!props.groups.length}
      title="Jump to a pipeline section"
      ariaLabel="Jump to a pipeline section"
      buttonClass="seg icon secmenu-btn"
    >
      {(close) => (
        <div className="menu-scroll">
          {props.groups.map((g) => (
            <button
              type="button"
              key={g}
              className="menu-item"
              role="menuitem"
              onClick={() => {
                props.onPick(g);
                close();
              }}
            >
              <span className="menu-text">{g}</span>
            </button>
          ))}
        </div>
      )}
    </Popover>
  );
}
