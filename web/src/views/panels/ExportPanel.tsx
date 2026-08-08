/** Choosing what to write, and watching it render.
 *
 *  **Every export is full size** (2026-08-08, on request). The menu picks the
 *  supersample -- how finely the frame is rendered -- and nothing else. It
 *  replaced a three-way scale menu whose entries moved resolution and look
 *  together: "As previewed" wrote a smaller file *and* a coarser grain, because
 *  every length scales with the frame, so the one control was answering two
 *  questions at once and neither cleanly.
 *
 *  This sits over the photo, bottom right, rather than at the foot of the
 *  panel. The prose that used to run under the scale menu is behind the help
 *  badge instead: three paragraphs are a reasonable thing to scroll past in a
 *  side panel and an unreasonable thing to lay over the picture, and the words
 *  are worth keeping either way.
 *
 *  It is one row, matching the viewbar in the opposite corner (2026-08-08, on
 *  request) -- it was four stacked rows and stood far taller over the picture
 *  than the thing it is meant to pair with. Nothing was dropped to fit: the
 *  button's label moved to its tooltip, because the scale menu immediately to
 *  its left already says which of the three it will write, and repeating that
 *  in the button was the widest thing in the overlay.
 */

import { EXPORT_SUPERSAMPLES } from "../../services/api";
import type { ExportJob, ImageMeta } from "../../services/api";
import Help from "../controls/Help";

export default function ExportPanel(props: {
  meta: ImageMeta | null;
  exportSs: number;
  onExportSs: (s: number) => void;
  format: string;
  onFormat: (f: string) => void;
  onExport: () => void;
  job: ExportJob | null;
  /** Anything to sit at the right-hand end of the bar, past its separator.
   *  What goes in it is the caller's business. */
  headerAside?: React.ReactNode;
}) {
  const { meta, exportSs, job } = props;
  // Cost is roughly the square of the factor, which is the thing worth saying
  // out loud: 3x is 2.25x the work of 2x, and 1x is a quarter of it.
  const help =
    exportSs === 2
      ? `A full-size render of ${meta?.width}×${meta?.height} at 2× ` +
        "supersampling — the default, and what every preset was dialled in " +
        "against. Grain is rendered above the output grid and integrated down, " +
        "so each clump gets genuine partial-pixel coverage instead of a hard, " +
        "aliased footprint."
      : exportSs > 2
        ? "Renders finer than 2× and integrates down. Costs roughly the square " +
          "of the factor — 3× is 2.25× the work of 2× — for a modest gain in " +
          "how cleanly the smallest clumps resolve."
        : exportSs === 1
          ? "Renders at the output grid itself. Fast, but grain gets a hard, " +
            "aliased pixel footprint — the synthetic look supersampling exists " +
            "to avoid."
          : "Renders *below* the output and scales up, so the file is full " +
            "size but genuinely soft. For machines that cannot afford anything " +
            "else, not a look.";

  const action = `Export ${meta?.width}×${meta?.height} at ${exportSs}× supersampling`;

  return (
    <>
    <Help text={help} label="Export" />
      {/* Fixed width, so the bar does not resize as the menu is changed or a
          differently-sized photo is opened -- the labels carry pixel
          dimensions, and an overlay that shifts its own left edge while you
          are reading it is the reason the old one was sized rather than
          shrink-wrapped. */}
      <select
        className="xscale"
        value={exportSs}
        onChange={(e) => props.onExportSs(Number(e.target.value))}
      >
        {EXPORT_SUPERSAMPLES.map((s) => (
          <option key={s} value={s}>
            Full size{meta ? ` ${meta.width}×${meta.height}` : ""} / SS {s}×
          </option>
        ))}
      </select>
      <select
        className="xformat"
        value={props.format}
        onChange={(e) => props.onFormat(e.target.value)}
      >
        <option value="jpeg">JPEG 95</option>
        <option value="png16">PNG 16-bit</option>
        <option value="png8">PNG 8-bit</option>
      </select>
      <button
        className="btn primary export-go"
        onClick={props.onExport}
        disabled={!meta}
        title={action}
      >
        Export
      </button>
      {/* Progress inline; a failure as a sibling rather than inside `.job`, so
          the stylesheet can send the one that is a sentence to its own row
          without dragging the compact readout there with it. */}
      {job && job.status !== "done" && job.status !== "error" && (
        <div className="job">
          <div className="pbar">
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <span>
            {job.status} {Math.round(job.progress * 100)}%
          </span>
        </div>
      )}
      {job?.status === "error" && <span className="err">{job.error}</span>}
      {props.headerAside && (
        <>
          <span className="vsep" />
          {props.headerAside}
        </>
      )}
    </>
  );
}
