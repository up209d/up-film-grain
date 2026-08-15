/** One parameter's control, generated from the schema.
 *
 *  This is the only place a control is built from a `Param`, which is the rule
 *  that matters: adding a parameter means adding a `Param` in `params.py` and
 *  nothing here.
 *
 *  A discrete parameter renders as a menu rather than a slider, and an on/off
 *  one as a checkbox. All three are still a plain number everywhere else -- in
 *  the schema, in the engine and in a preset file -- so this is the only place
 *  in the app that knows the difference, and that is why it is one component
 *  rather than a Slider, a Dropdown and a Checkbox: nothing else ever needs to
 *  choose between them, so three components could never vary independently.
 */

import type { Schema } from "../../services/api";
import Help from "./Help";
import SelectMenu from "./SelectMenu";

export default function ParamControl(props: {
  param: Schema["params"][number];
  value: number | undefined;
  onChange: (k: string, v: number) => void;
  onChangeNow: (k: string, v: number) => void;
  onCommit: () => void;
}) {
  const { param: p, value, onChange, onChangeNow, onCommit } = props;

  if (p.toggle) {
    return (
      <div className="slider">
        {/* `onChangeNow`, not `onChange`, and for the same reason the menu
            below uses it: a checkbox is a discrete action with no pointer
            release to follow. `useValues.setValue` defers the commit to a
            window `pointerup` listener that exists for slider drags, so a
            deferred toggle would apply a stale value or none at all. */}
        <label className="toggle">
          <input
            type="checkbox"
            checked={(value ?? p.default) >= 0.5}
            onChange={(e) => onChangeNow(p.key, e.target.checked ? 1 : 0)}
          />
          {/* `Help` renders the label with the schema's help text as its
              tooltip, so the explanation travels with the parameter rather
              than being restated here. */}
          <Help text={p.help} label={p.label} />
        </label>
      </div>
    );
  }

  if (p.choices?.length) {
    return (
      <div className="slider">
        <div className="slabel">
          <Help text={p.help} label={p.label} />
        </div>
        {/* The value is the *index*, stringified on the way in and parsed on
            the way out. `SelectMenu` deals in strings because a menu is a list
            of names; the schema is where a value's type is decided, and it says
            number. That conversion was always here -- `<select>` needed it
            too -- and this is still the only place in the app that knows a
            discrete parameter from a continuous one. */}
        <SelectMenu
          items={p.choices.map((c, i) => ({ value: String(i), label: c }))}
          value={String(value ?? p.default)}
          onPick={(v) => onChangeNow(p.key, Number(v))}
          title={p.label}
        />
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
