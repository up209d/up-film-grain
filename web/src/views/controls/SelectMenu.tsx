/** The app's one way to choose from a list.
 *
 *  Every native `<select>` became one of these on 2026-08-09, on request: the
 *  browser's dropdown cannot be styled to match the panel, cannot group beyond
 *  one flat `<optgroup>` level, and cannot be searched -- and the LUT folder had
 *  just grown from 7 entries to 303 across nine subfolders, which is where all
 *  three of those stopped being cosmetic.
 *
 *  It is deliberately one component covering every case rather than a plain one
 *  and a searchable one. The list of things it has to serve is short and the
 *  differences between them are two booleans; two components would mean two
 *  places for the keyboard handling to drift.
 *
 *  **The value is a string here and a number at most call sites.** A discrete
 *  parameter's value is its index and the supersample menu's is a factor, so
 *  those callers stringify on the way in and parse on the way out. That is the
 *  same bargain `ParamControl` already made with `<select>`, kept in one place:
 *  a menu is a list of names, and the schema is where a value's type is decided.
 */

import {
  CheckIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from "@heroicons/react/24/outline";
import { useEffect, useMemo, useRef, useState } from "react";

import Popover from "./Popover";

export interface MenuItem {
  value: string;
  label: string;
  /** Dimmed suffix -- a cube size, a pixel dimension, "missing". Searched
   *  alongside the label, because "loaded" is a thing you would type. */
  hint?: string;
  /** Folder-style heading to file it under. Absent or "" means it renders
   *  above every group and is always visible. */
  group?: string;
}

/** Lower-cased, with every separator the LUT library uses flattened to a space.
 *  Applied to both sides of the comparison, so what the user types and what is
 *  on disk are punctuated the same way before they are matched. */
const norm = (s: string) => s.toLowerCase().replace(/[_\-/.]+/g, " ").trim();

export default function SelectMenu(props: {
  items: MenuItem[];
  /** The selected value, or null for a menu that holds no selection -- the
   *  preset list is a set of commands, not a state. */
  value: string | null;
  onPick: (value: string) => void;
  /** Trigger text when nothing is selected. */
  placeholder?: string;
  /** Show the filter box. Worth it past a screenful, pure noise below one. */
  searchable?: boolean;
  /** Render groups as collapsed headings rather than as plain separators. */
  collapseGroups?: boolean;
  drop?: "up" | "down";
  align?: "left" | "right";
  disabled?: boolean;
  title?: string;
  /** Extra classes on the trigger -- the two export menus are sized. */
  buttonClass?: string;
  panelClass?: string;
}) {
  const current = props.items.find((i) => i.value === props.value);
  const label = current ? current.label : props.placeholder ?? "—";

  return (
    <Popover
      trigger={
        <>
          <span className="menu-cur">{label}</span>
          {/* Drawn chevrons rather than the `▾` and `✓` glyphs these started
              as, here and on the group headings and the tick. A character
              renders at whatever weight the system font gives it, which at this
              size was a barely-visible smudge; an icon carries its own stroke
              width and holds it. (The panel's section headers still use a `›`
              -- they are 11px uppercase headings with nothing else competing
              for the eye, and they were never the complaint.) */}
          <ChevronDownIcon className="menu-caret" aria-hidden="true" />
        </>
      }
      drop={props.drop}
      align={props.align}
      // A search box is a textbox, and a textbox may not live inside a `menu`.
      // The `menu` moves to the list itself, where the items are.
      role={props.searchable ? "dialog" : "menu"}
      disabled={props.disabled || !props.items.length}
      title={props.title}
      buttonClass={`menu-select${props.buttonClass ? ` ${props.buttonClass}` : ""}`}
      panelClass={props.panelClass}
    >
      {(close) => (
        // Its own component so query, collapse state and the keyboard cursor
        // all mount fresh on every open. Reopening a menu onto last time's
        // half-typed filter is the behaviour nobody wants and everybody ships.
        <MenuBody
          items={props.items}
          value={props.value}
          searchable={props.searchable}
          collapseGroups={props.collapseGroups}
          onPick={(v) => {
            props.onPick(v);
            close();
          }}
        />
      )}
    </Popover>
  );
}

function MenuBody(props: {
  items: MenuItem[];
  value: string | null;
  searchable?: boolean;
  collapseGroups?: boolean;
  onPick: (value: string) => void;
}) {
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement | null>(null);
  const activeRef = useRef<HTMLButtonElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  /** Groups the user has opened by hand. Collapsed is the default and it is
   *  what makes 303 entries cheap -- nine headings and the seven root LUTs are
   *  all that is in the DOM until something is expanded. The group holding the
   *  current value starts open, so a selection is never hidden behind a
   *  heading you have to guess at. */
  const [opened, setOpened] = useState<Set<string>>(() => {
    const g = props.items.find((i) => i.value === props.value)?.group;
    return new Set(g ? [g] : []);
  });

  // `preventScroll` is load-bearing, not a nicety. The panel is absolutely
  // positioned inside the parameter sidebar, so an upward-opening menu puts its
  // search box *above* the sidebar's visible area -- and a plain `focus()` makes
  // the browser scroll every ancestor to reveal it, which threw the sidebar up
  // by as much as 694px and lost the reader's place. Focus belongs to the menu;
  // the scroll position behind it is none of its business.
  //
  // Falls back to the list when there is no search box. Without that, focus
  // stays on the trigger — which sits *outside* this element — so the keyboard
  // handler below never sees an arrow key, and the six menus that have no
  // search were the ones where arrows did nothing.
  useEffect(() => {
    (searchRef.current ?? scrollRef.current)?.focus({ preventScroll: true });
  }, []);

  const q = norm(query);

  /** The visible tree: loose items first, then a heading per group.
   *
   *  A query matches an item on its label or hint, *or* the whole group on its
   *  name -- typing a folder name is how you ask for that folder, and matching
   *  only leaf labels would return nothing for it. Anything matching is shown
   *  expanded regardless of `opened`: a search that hides its own results
   *  behind a collapsed heading is a search that does not work. */
  const { loose, groups } = useMemo(() => {
    // Every word must appear somewhere in the row, in any order, and the
    // separators the filenames use count as spaces. Typing `kodak portra` has
    // to find `gmic/negative_new/kodak_portra_400`: a literal substring test
    // finds nothing there, because the only thing between those two words on
    // disk is an underscore. `bw agfa` reaching `gmic/bw/agfa_apx_100` falls
    // out of the same rule -- the folder is searched alongside the name, so a
    // term can match either.
    const terms = q.split(/\s+/).filter(Boolean);
    const hit = (i: MenuItem) => {
      if (!terms.length) return true;
      const hay = norm(`${i.group ?? ""} ${i.label} ${i.hint ?? ""}`);
      return terms.every((t) => hay.includes(t));
    };

    const loose: MenuItem[] = [];
    const byGroup = new Map<string, MenuItem[]>();
    for (const i of props.items) {
      if (!hit(i)) continue;
      if (!i.group) loose.push(i);
      else {
        const list = byGroup.get(i.group);
        if (list) list.push(i);
        else byGroup.set(i.group, [i]);
      }
    }
    return { loose, groups: [...byGroup.entries()] };
  }, [props.items, q]);

  const isOpen = (g: string) => !!q || opened.has(g);

  /** Every item currently on screen, in render order -- what the arrow keys
   *  walk. Rebuilt from the same flags the render uses rather than tracked
   *  alongside it, so the cursor cannot address a row that is not there. */
  const visible = useMemo(() => {
    const out = [...loose];
    for (const [g, items] of groups) if (isOpen(g)) out.push(...items);
    return out;
  }, [loose, groups, opened, q]);

  /** value -> row number in `visible`. A map rather than an `indexOf` per row:
   *  every item asks, so the scan is quadratic and this list runs to 303. */
  const rowOf = useMemo(() => {
    const m = new Map<string, number>();
    visible.forEach((i, n) => m.set(i.value, n));
    return m;
  }, [visible]);

  const [active, setActive] = useState(0);
  // A filter that shortens the list would otherwise leave the cursor past its
  // end, and Enter would pick nothing.
  useEffect(() => setActive(0), [q]);

  /** Keep the keyboard cursor in view — by scrolling **only the list**.
   *
   *  `scrollIntoView` would be the obvious call and is the wrong one for the
   *  same reason the `focus` above needs `preventScroll`: it walks every
   *  scrollable ancestor, and one of those is the parameter sidebar this panel
   *  floats over. Nudging `scrollTop` by hand cannot touch anything outside the
   *  list.
   *
   *  Only the overshoot is corrected, never centred, so a mouse hover — which
   *  also sets `active` — does not yank the list out from under the pointer. */
  useEffect(() => {
    const el = activeRef.current;
    const box = scrollRef.current;
    if (!el || !box) return;
    const e = el.getBoundingClientRect();
    const b = box.getBoundingClientRect();
    if (e.top < b.top) box.scrollTop -= b.top - e.top;
    else if (e.bottom > b.bottom) box.scrollTop += e.bottom - b.bottom;
  }, [active]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!visible.length) return;
      const d = e.key === "ArrowDown" ? 1 : -1;
      setActive((a) => (a + d + visible.length) % visible.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const pick = visible[active];
      if (pick) props.onPick(pick.value);
    }
    // Escape is Popover's -- it listens on the document, so it fires whether or
    // not focus is still in here.
  };

  const row = (i: MenuItem) => {
    const idx = rowOf.get(i.value) ?? -1;
    return (
      <button
        type="button"
        key={i.value}
        ref={idx === active ? activeRef : undefined}
        className={`menu-item${i.value === props.value ? " sel" : ""}${
          idx === active ? " active" : ""
        }`}
        role="menuitemradio"
        aria-checked={i.value === props.value}
        onMouseEnter={() => idx >= 0 && setActive(idx)}
        onClick={() => props.onPick(i.value)}
      >
        {/* Always in the DOM, visible only when selected: the rows have to
            share a left edge whether or not one of them is ticked, and a tick
            that appears and shifts every label right is the flicker this
            avoids. */}
        <span className="menu-tick" aria-hidden="true">
          {i.value === props.value && <CheckIcon />}
        </span>
        <span className="menu-text">{i.label}</span>
        {i.hint && <em className="menu-hint">{i.hint}</em>}
      </button>
    );
  };

    // `menu-body` is not decoration: it sits between the panel and the scroll
    // box, and a plain div here silently breaks the height chain. `Popover`
    // caps the *panel*, the scroll box flexes into what is left -- and an
    // unstyled element in between has `flex: 0 1 auto` and `min-height: auto`,
    // so it keeps its natural height, the list never shrinks, and the panel
    // clips its overflow instead of scrolling it.
  return (
    <div className="menu-body" onKeyDown={onKeyDown}>
      {props.searchable && (
        <input
          ref={searchRef}
          className="menu-search"
          type="text"
          value={query}
          placeholder="Search…"
          spellCheck={false}
          autoComplete="off"
          onChange={(e) => setQuery(e.target.value)}
        />
      )}
      <div
        className="menu-scroll"
        ref={scrollRef}
        // Focusable but not tab-reachable: it exists to catch arrow keys when
        // there is no search box to hold focus, not to add a tab stop.
        tabIndex={-1}
        role={props.searchable ? "menu" : undefined}
      >
        {loose.map(row)}
        {groups.map(([g, items]) => {
          const open = isOpen(g);
          return (
            <div className="menu-group" key={g}>
              <button
                type="button"
                className={`menu-head${open ? " open" : ""}`}
                // Not a menuitem: it reveals rows rather than choosing one, and
                // announcing it as a choice would make the list read as 312
                // options when there are 303.
                aria-expanded={open}
                onClick={() =>
                  setOpened((s) => {
                    const next = new Set(s);
                    next.has(g) ? next.delete(g) : next.add(g);
                    return next;
                  })
                }
                // A search has forced this group open; collapsing it by hand
                // would fight the query rather than clear it.
                disabled={!!q}
              >
                <ChevronRightIcon
                  className={open ? "menu-chev open" : "menu-chev"}
                  aria-hidden="true"
                />
                <span className="menu-text">{g}</span>
                <em className="menu-hint">{items.length}</em>
              </button>
              {open && items.map(row)}
            </div>
          );
        })}
        {!loose.length && !groups.length && (
          <p className="menu-empty">Nothing matches “{query}”.</p>
        )}
      </div>
    </div>
  );
}
