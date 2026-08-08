/** Choosing what to write, and watching it render.
 *
 *  **Every export is full size** (2026-08-08, on request). The menu picks the
 *  supersample -- how finely the frame is rendered -- and nothing else. It
 *  replaced a three-way scale menu whose entries moved resolution and look
 *  together: "As previewed" wrote a smaller file *and* a coarser grain, because
 *  every length scales with the frame, so the one control was answering two
 *  questions at once and neither cleanly.
 *
 *  **Every export is also the preview's look now** (2026-08-09, on request).
 *  The labels and the five factors are untouched; what each one renders is the
 *  previewed frame at that supersample, enlarged to the source's dimensions.
 *  So the file is the picture the settings were judged on rather than a 1:1
 *  render of the same numbers, which resolves finer, denser grain and is a
 *  different picture. Only the help text moved with it -- the menu still says
 *  the same five things, because it is still asking the same question.
 *
 *  **Plus a sixth entry, `Full size ... / 1:1 SS 1x`** (2026-08-09, on
 *  request): the real full-resolution render at 1x, for when the frame's own
 *  finest grain is what is wanted. Explicitly not the default -- 2x as
 *  previewed still is -- because it is the one file the preview cannot show
 *  you, and defaulting to it is the disagreement this section spent the day
 *  removing.
 *
 *  The menu's value is the option **key**, not the factor, since that entry
 *  arrived: `ss1` and `full` are both 1x and differ only in which tier they
 *  render, so a number can no longer say which one is selected.
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

import { EXPORT_OPTIONS, exportOption } from "../../services/api";
import type { ExportJob, ImageMeta } from "../../services/api";
import Help from "../controls/Help";

export default function ExportPanel(props: {
  meta: ImageMeta | null;
  exportKey: string;
  onExportKey: (k: string) => void;
  format: string;
  onFormat: (f: string) => void;
  onExport: () => void;
  job: ExportJob | null;
  /** Anything to sit at the right-hand end of the bar, past its separator.
   *  What goes in it is the caller's business. */
  headerAside?: React.ReactNode;
}) {
  const { meta, job } = props;
  const opt = exportOption(props.exportKey);
  const exportSs = opt.ss;
  const size = `${meta?.width}×${meta?.height}`;
  // Five entries render *what the preview renders* and enlarge it to the
  // source's dimensions, so the file matches the picture the settings were
  // judged on. The supersample is quality within that render, and cost is
  // roughly its square: 3× is 2.25× the work of 2×, 1× a quarter of it. The
  // sixth is the one that does not, and its help text has to say so plainly --
  // it is the only way to get a file whose grain the preview never showed.
  const enlarged = `, enlarged to ${size}`;
  const help = opt.full
    ? `A genuine ${size} render at 1× supersampling — the source's own grid, ` +
      "not the preview enlarged. It is the only entry whose pixels you have " +
      "not already seen: every length scales with the frame, so a full-" +
      "resolution render resolves finer, denser grain than the preview did. " +
      "That is a different picture, not a sharper one, which is why it is not " +
      "the default — use Render 1:1 to look at it before committing. At 1× " +
      "there is no supersampling either, so grain lands on a hard pixel " +
      "footprint."
    : exportSs === 2
      ? "The previewed frame rendered at 2× supersampling" + enlarged +
        " — the default, and what every preset was dialled in against. Grain " +
        "is rendered above the preview's grid and integrated down, so each " +
        "clump gets genuine partial-pixel coverage instead of a hard, aliased " +
        "footprint. It adds no detail beyond what the preview resolves; that " +
        "is the point — the file is the picture you judged."
      : exportSs > 2
        ? "Renders the previewed frame finer than 2× and integrates down" +
          enlarged + ". Costs roughly the square of the factor — 3× is 2.25× " +
          "the work of 2× — for a modest gain in how cleanly the smallest " +
          "clumps resolve."
        : exportSs === 1
          ? "Renders the previewed frame at its own grid" + enlarged +
            ". Fast, but grain gets a hard, aliased pixel footprint — the " +
            "synthetic look supersampling exists to avoid."
          : "Renders *below* the previewed frame and scales up" + enlarged +
            ", so the file is full size but genuinely soft. For machines that " +
            "cannot afford anything else, not a look.";

  const action = opt.full
    ? `Export a fresh ${size} render at 1× supersampling`
    : `Export as previewed at ${exportSs}× supersampling, enlarged to ${size}`;

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
        value={props.exportKey}
        onChange={(e) => props.onExportKey(e.target.value)}
      >
        {EXPORT_OPTIONS.map((o) => (
          <option key={o.key} value={o.key}>
            Full size{meta ? ` ${meta.width}×${meta.height}` : ""}
            {o.full ? " / 1:1 SS 1×" : ` / SS ${o.ss}×`}
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
