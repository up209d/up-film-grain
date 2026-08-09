import { useMemo } from "react";

import { type LutInfo } from "../../services/api";
import SelectMenu, { type MenuItem } from "../controls/SelectMenu";

/** The "no LUT" entry's value.
 *
 *  Not the empty string, because `SelectMenu` reads a value it cannot find as
 *  "nothing is selected, show the placeholder" — and None *is* a selection, so
 *  it has to be findable. A backslash is what makes the sentinel
 *  collision-proof: a folder id is built with `as_posix()` so it can never
 *  contain one, and `lut.resolve_path` rejects the character outright, so no
 *  real LUT can ever answer to this. */
const NONE = "\\none";

/** The 3D LUT selector: everything in `luts/`, everything uploaded this
 *  session, and a button to add one.
 *
 *  Searchable and grouped since 2026-08-09, when the folder went from 7 entries
 *  to 303 across nine subfolders. Three things follow from that number, and all
 *  three are why this is not a plain menu:
 *
 *  * **Folders are groups, collapsed.** A LUT's id is its path relative to
 *    `luts/`, and the server reports the folder it came from as `group`, so the
 *    grouping is read off the listing rather than parsed back out of the id
 *    here. Root-level LUTs have no group and sit above the headings, always
 *    visible — those are the handful that came with the app.
 *  * **Search.** Nine collapsed headings are quick to scan and useless if you
 *    already know the name you want. The filter matches folder names too, so
 *    typing `instant` gets you the folder rather than nothing.
 *  * **The label is the bare filename**, not the path — the path is already the
 *    heading it sits under, and repeating it in every row would push the part
 *    that distinguishes them off the right edge.
 *
 *  A LUT that is selected but absent from the list gets its own entry rather
 *  than silently resetting the menu to None. That happens for real -- a preset
 *  file naming a `.cube` that has since been renamed, or an upload from a
 *  previous run, since those live in process memory. Showing "missing" is the
 *  honest state: the server renders with no LUT and zeroes the mix, so the
 *  picture is right, and the picker says why rather than looking like the
 *  preset had no LUT in it.
 *
 *  Sizes are shown for uploads because they were parsed on the way in. Folder
 *  entries do not report one -- listing them deliberately does not open them,
 *  so a library of 64-cubes costs nothing to browse. */
function LutPicker(props: {
  luts: LutInfo[];
  value: string | null;
  onPick: (id: string | null) => void;
  onLoadFile: () => void;
}) {
  const { luts, value } = props;
  const missing = !!value && !luts.some((l) => l.id === value);

  const items: MenuItem[] = useMemo(() => {
    const out: MenuItem[] = [{ value: NONE, label: "None" }];
    if (missing) {
      out.push({ value: value!, label: lutLabel(value!), hint: "missing" });
    }
    for (const l of luts) {
      out.push({
        value: l.id,
        label: l.name,
        hint: [l.size ? `${l.size}³` : "", l.source === "upload" ? "loaded" : ""]
          .filter(Boolean)
          .join(" · "),
        group: l.group,
      });
    }
    return out;
  }, [luts, value, missing]);

  return (
    <div className="slider lutpick">
      <div className="slabel">
        <span className="title">LUT</span>
        <button
          className="seg"
          onClick={props.onLoadFile}
          title="Load a .cube file from disk for this session"
        >
          Load .cube…
        </button>
      </div>
      <SelectMenu
        items={items}
        value={value ?? NONE}
        onPick={(v) => props.onPick(v === NONE ? null : v)}
        searchable
        collapseGroups
        title="Choose a 3D LUT"
      />
      {missing && (
        <p className="hint">
          This look wants a LUT called <strong>{lutLabel(value!)}</strong>, which
          is not in <code>luts/</code>
          {value!.startsWith("upload:")
            ? " — it was loaded from disk in an earlier session, so it has to be loaded again."
            : " — drop the .cube file in there, or load it from disk."}{" "}
          Nothing is being applied in the meantime.
        </p>
      )}
    </div>
  );
}

/** A LUT id as something to show a person. Uploads carry an opaque id, so there
 *  is nothing better to print for one that is no longer loaded. A folder id
 *  prints as its whole path, which is what a preset file records and therefore
 *  what you would go looking for on disk. */
const lutLabel = (id: string) =>
  id.startsWith("upload:") ? "a loaded file" : id;

export default LutPicker;
