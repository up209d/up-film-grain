/** One parameter's control, generated from the schema.
 *
 *  This is the only place a control is built from a `Param`, which is the rule
 *  that matters: adding a parameter means adding a `Param` in `params.py` and
 *  nothing here.
 *
 *  A discrete parameter renders as a menu rather than a slider. It is still a
 *  plain number everywhere else -- in the schema, in the engine and in a preset
 *  file -- so this is the only place in the app that knows the difference, and
 *  that is why it is one component and not a Slider and a Dropdown: nothing
 *  else ever needs to choose between them.
 */

import type { Schema } from "../../services/api";
import Help from "./Help";

export default function ParamControl(props: {
  param: Schema["params"][number];
  value: number | undefined;
  onChange: (k: string, v: number) => void;
  onChangeNow: (k: string, v: number) => void;
  onCommit: () => void;
}) {
  const { param: p, value, onChange, onChangeNow, onCommit } = props;

  if (p.choices?.length) {
    return (
      <div className="slider">
        <div className="slabel">
          <Help text={p.help} label={p.label} />
        </div>
        <select
          value={String(value ?? p.default)}
          onChange={(e) => onChangeNow(p.key, Number(e.target.value))}
        >
          {p.choices.map((c, i) => (
            <option key={c} value={i}>
              {c}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <div className="slider">
      <div className="slabel">
        <Help text={p.help} label={p.label} />
        <input
          className="num"
          type="number"
          min={p.min}
          max={p.max}
          step={p.step}
          value={value ?? p.default}
          onChange={(e) => onChange(p.key, Number(e.target.value))}
          onKeyUp={onCommit}
          onBlur={onCommit}
        />
        {p.unit && <em>{p.unit}</em>}
      </div>
      <input
        type="range"
        min={p.min}
        max={p.max}
        step={p.step}
        value={value ?? p.default}
        onChange={(e) => onChange(p.key, Number(e.target.value))}
        // Pointer releases come from the window listener in `useValues`; these
        // cover the keyboard path (arrows nudge the thumb without ever
        // producing a pointer event).
        onKeyUp={onCommit}
        onBlur={onCommit}
      />
    </div>
  );
}
