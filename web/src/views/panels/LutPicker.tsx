import { type LutInfo } from "../../services/api";

/** The 3D LUT selector: everything in `luts/`, everything uploaded this
 *  session, and a button to add one.
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
 *  so a directory of 64-cubes costs nothing to browse. */
function LutPicker(props: {
  luts: LutInfo[];
  value: string | null;
  onPick: (id: string | null) => void;
  onLoadFile: () => void;
}) {
  const { luts, value } = props;
  const missing = !!value && !luts.some((l) => l.id === value);
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
      <select
        value={value ?? ""}
        onChange={(e) => props.onPick(e.target.value || null)}
      >
        <option value="">None</option>
        {missing && <option value={value!}>{lutLabel(value!)} — missing</option>}
        {luts.map((l) => (
          <option key={l.id} value={l.id}>
            {l.name}
            {l.size ? ` (${l.size}³)` : ""}
            {l.source === "upload" ? " — loaded" : ""}
          </option>
        ))}
      </select>
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
 *  is nothing better to print for one that is no longer loaded. */
const lutLabel = (id: string) =>
  id.startsWith("upload:") ? "a loaded file" : id;

export default LutPicker;
