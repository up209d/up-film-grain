/** The generated control panel: one collapsible section per group, one control
 *  per parameter, built entirely from the schema.
 *
 *  Nothing here names a parameter. The one exception is the LUT picker, which
 *  is interleaved at `LUT_ANCHOR_KEY` -- a LUT is a named resource rather than
 *  a number, so there is nothing in the schema for the generator to pick up,
 *  and anchoring it to its own Mix slider keeps the section in pipeline order.
 */

import { Fragment } from "react";

import { LUT_ANCHOR_KEY, sectionDomId } from "../../models/constants";
import type { Values } from "../../models/types";
import type { LutInfo, Schema } from "../../services/api";
import ParamControl from "../controls/ParamControl";
import LutPicker from "./LutPicker";

export default function SliderPanel(props: {
  grouped: { group: string; params: Schema["params"] }[];
  values: Values;
  muted: Record<string, Values>;
  collapsed: Record<string, boolean>;
  onToggleCollapsed: (group: string) => void;
  onResetGroup: (group: string) => void;
  onToggleGroup: (group: string) => void;
  onChange: (k: string, v: number) => void;
  onChangeNow: (k: string, v: number) => void;
  onCommit: () => void;
  luts: LutInfo[];
  lut: string | null;
  onPickLut: (id: string | null) => void;
  onLoadLutFile: () => void;
}) {
  return (
    <div className="groups">
      {props.grouped.map(({ group, params }) => (
        // The id is what the section menu in the export bar scrolls to; it is
        // on the section rather than the header so the whole block, not just
        // its title, is what gets brought into view.
        <section key={group} id={sectionDomId(group)} className="group">
          <h3 onClick={() => props.onToggleCollapsed(group)}>
            <span className={props.collapsed[group] ? "chev" : "chev open"}>›</span>
            {group}
            {/* Both act on this section only, and both have to stop
                propagation -- the header itself toggles collapse, so
                without it either one would also fold the section shut. */}
            <button
              className="grpbtn"
              title={`Reset ${group} to the starting preset`}
              onClick={(e) => {
                e.stopPropagation();
                props.onResetGroup(group);
              }}
            >
              ↺
            </button>
            <button
              className={props.muted[group] ? "grpbtn on" : "grpbtn"}
              title={
                props.muted[group]
                  ? `Switch ${group} back on`
                  : `Switch ${group} off`
              }
              onClick={(e) => {
                e.stopPropagation();
                props.onToggleGroup(group);
              }}
            >
              {props.muted[group] ? "○" : "●"}
            </button>
          </h3>
          {!props.collapsed[group] &&
            params.map((p) => (
              <Fragment key={p.key}>
                {p.key === LUT_ANCHOR_KEY && (
                  <LutPicker
                    luts={props.luts}
                    value={props.lut}
                    onPick={props.onPickLut}
                    onLoadFile={props.onLoadLutFile}
                  />
                )}
                <ParamControl
                  param={p}
                  value={props.values[p.key]}
                  onChange={props.onChange}
                  onChangeNow={props.onChangeNow}
                  onCommit={props.onCommit}
                />
              </Fragment>
            ))}
        </section>
      ))}
    </div>
  );
}
