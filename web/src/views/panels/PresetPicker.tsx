/** Picking a preset, the two whole-look buttons, and preset files.
 *
 *  "Original" and "Reset" are different questions and are deliberately both
 *  here: Original switches every stage off to show the untouched photo, Reset
 *  goes back to how the app opened -- which is the starting preset staged
 *  behind each section's mute button, not applied.
 */

import type { Schema } from "../../services/api";

export default function PresetPicker(props: {
  schema: Schema | null;
  isOriginal: boolean;
  onApplyPreset: (name: string) => void;
  onShowOriginal: () => void;
  onResetAll: () => void;
  onSaveFile: () => void;
  onLoadFile: (f: File) => void;
  fileRef: React.MutableRefObject<HTMLInputElement | null>;
  notice: string | null;
}) {
  return (
    <>
      <div className="row">
        <select
          defaultValue=""
          onChange={(e) => e.target.value && props.onApplyPreset(e.target.value)}
        >
          <option value="">Preset…</option>
          {props.schema?.presets.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        <button
          className="btn ghost"
          onClick={props.onShowOriginal}
          disabled={props.isOriginal}
          title="Switch every stage off — show the untouched photo"
        >
          Original
        </button>
        <button
          className="btn ghost"
          onClick={props.onResetAll}
          title="Back to the starting preset"
        >
          Reset
        </button>
      </div>

      <div className="row">
        <button
          className="btn ghost"
          onClick={props.onSaveFile}
          disabled={!props.schema}
          title="Write the current settings to a .json file"
        >
          Save to file…
        </button>
        <button
          className="btn ghost"
          onClick={() => props.fileRef.current?.click()}
          disabled={!props.schema}
          title="Load settings from a .json file"
        >
          Load file…
        </button>
        <input
          ref={props.fileRef}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            // Cleared so picking the same file twice still fires a change
            // event -- otherwise re-loading a file you just edited is a
            // no-op with no feedback.
            e.target.value = "";
            if (f) props.onLoadFile(f);
          }}
        />
      </div>
      {props.notice && <p className="note">{props.notice}</p>}
    </>
  );
}
