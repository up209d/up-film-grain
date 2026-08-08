/** Size scaling, and the arithmetic behind it spelled out.
 *
 *  A preset dialled in on a 24MP frame means something different on a 45MP one.
 *  The server rescales lengths by the **linear** ratio, and the readout here
 *  shows both ratios side by side because reaching for the megapixel one is the
 *  obvious mistake -- see `docs/presets.md`.
 *
 *  The factor can also be set by hand. Automatic is right about the arithmetic
 *  and does not always agree with the picture -- a crop, a photo that is not
 *  the size the look was judged at, or simply wanting coarser grain than the
 *  frame calls for -- and the on/off switch on its own only ever offered the
 *  computed answer or none of it. Manual is a *view* choice like the mount and
 *  the quality menu: it is not written into a saved preset, because the preset
 *  records what size its numbers were authored at, which is a fact about the
 *  file rather than a preference about this session.
 */

import {
  SCALE_MANUAL_MAX,
  SCALE_MANUAL_MIN,
  SCALE_MANUAL_STEP,
} from "../../models/constants";
import type { ImageMeta } from "../../services/api";
import Field from "../controls/Field";

export default function ScalePanel(props: {
  meta: ImageMeta | null;
  referenceMp: number | null;
  scaleToRef: boolean;
  /** Hand-set linear factor, or null to compute it from the sizes. */
  scaleOverride: number | null;
  onToggleScaleToRef: () => void;
  onSetFromPhoto: () => void;
  onScaleOverride: (v: number | null) => void;
}) {
  const { meta, referenceMp, scaleToRef, scaleOverride } = props;
  if (!meta) return null;

  const auto = referenceMp ? Math.sqrt(meta.megapixels / referenceMp) : 1;
  const manual = scaleOverride !== null;
  // What the render will actually use. The switch still wins over both: off
  // means nothing is scaled, however the factor was arrived at.
  const linear = manual ? scaleOverride : auto;
  const effective = scaleToRef && (manual || referenceMp) ? linear : 1;
  const clamp = (v: number) =>
    Math.min(SCALE_MANUAL_MAX, Math.max(SCALE_MANUAL_MIN, v));

  return (
    <>
      <Field label="Size scaling">
        <button
          className={scaleToRef && (manual || referenceMp) ? "seg on" : "seg"}
          onClick={props.onToggleScaleToRef}
          // A hand-set factor needs no preset reference to scale against, so
          // the switch is live even for a preset that records no size.
          disabled={!referenceMp && !manual}
          title="Rescale every length for this photo's size"
        >
          {!referenceMp && !manual ? "n/a" : scaleToRef ? "On" : "Off"}
        </button>
        <span className="val">{effective.toFixed(2)}×</span>
        <button
          className="seg"
          onClick={props.onSetFromPhoto}
          disabled={referenceMp === meta.megapixels}
          title="Record this photo's size as the size these settings were dialled in on"
        >
          Set from photo
        </button>
      </Field>

      <Field label="Scale factor">
        <button
          className={manual ? "seg" : "seg on"}
          onClick={() => props.onScaleOverride(null)}
          title="Compute the factor from the preset's size and this photo's"
        >
          Auto
        </button>
        <button
          className={manual ? "seg on" : "seg"}
          // Starts wherever automatic had it, so switching to manual moves
          // nothing -- you take the computed answer and adjust from it rather
          // than being dropped at 1.00x and having to find your way back.
          onClick={() => props.onScaleOverride(clamp(auto))}
          title="Set the factor by hand"
        >
          Manual
        </button>
        <span className="val">{manual ? `${linear.toFixed(2)}×` : "—"}</span>
      </Field>

      {manual && (
        <div className="fieldbody">
          <input
            type="range"
            min={SCALE_MANUAL_MIN}
            max={SCALE_MANUAL_MAX}
            step={SCALE_MANUAL_STEP}
            value={linear}
            onChange={(e) => props.onScaleOverride(Number(e.target.value))}
          />
          <input
            className="num"
            type="number"
            min={SCALE_MANUAL_MIN}
            max={SCALE_MANUAL_MAX}
            step={SCALE_MANUAL_STEP}
            value={linear}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n)) props.onScaleOverride(clamp(n));
            }}
          />
        </div>
      )}

      <p className="hint scalebox">
        <span>
          {manual ? (
            <>
              factor&nbsp;<strong>manual</strong>
            </>
          ) : (
            <>
              preset&nbsp;
              <strong>{referenceMp ? `${referenceMp}MP` : "—"}</strong>
            </>
          )}
        </span>
        <span>→</span>
        <span>
          photo&nbsp;<strong>{meta.megapixels}MP</strong>
        </span>
        <span>=</span>
        <span>
          <strong>{effective.toFixed(3)}×</strong>
        </span>
      </p>

      {!referenceMp && !manual && (
        <p className="hint">
          This preset does not record what size it was dialled in on, so nothing
          is scaled — it behaves exactly as it did before. If this photo is the
          size you dialled it in on, press <strong>Set from photo</strong> then{" "}
          <strong>Save to file…</strong> to stamp it at {meta.megapixels}MP. To
          retrofit every old preset at once, start the server with{" "}
          <code>FILM_GRAIN_DEFAULT_REFERENCE_MP=24</code>. Or press{" "}
          <strong>Manual</strong> above and set the factor yourself.
        </p>
      )}

      {manual && scaleToRef && (
        <p className="hint">
          Lengths are multiplied by {linear.toFixed(2)}× because you said so —
          the preset's own size, if it records one, is ignored while this is on.
          It is not saved into a preset file: <strong>Auto</strong> returns to{" "}
          {auto.toFixed(2)}×.
        </p>
      )}

      {!manual && referenceMp && scaleToRef && (
        <p className="hint">
          Lengths — clump size, every radius, jitter, speck and scratch size —
          are multiplied by the <strong>linear</strong> ratio{" "}
          {auto.toFixed(3)}×, not the megapixel ratio{" "}
          {(meta.megapixels / referenceMp).toFixed(2)}×: a photo with{" "}
          {(meta.megapixels / referenceMp).toFixed(2)}× the pixels is only{" "}
          {auto.toFixed(2)}× as wide. Amounts and mark counts are not scaled —
          they already mean the same thing at any size.
        </p>
      )}
    </>
  );
}
