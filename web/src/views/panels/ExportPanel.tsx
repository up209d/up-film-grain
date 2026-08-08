/** Choosing what to write, and watching it render.
 *
 *  The three scales are not a size choice -- see the help text below and
 *  `docs/preview-and-export.md`. "As previewed" writes the proxy render itself
 *  because every length scales with the frame, so the same settings resolve
 *  finer grain at full size; a downscale of the 1:1 render would not match what
 *  is on screen.
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

import type { ExportJob, ExportScale, ImageMeta } from "../../services/api";
import Help from "../controls/Help";

export default function ExportPanel(props: {
  meta: ImageMeta | null;
  exportScale: ExportScale;
  onExportScale: (s: ExportScale) => void;
  format: string;
  onFormat: (f: string) => void;
  onExport: () => void;
  job: ExportJob | null;
  /** Anything to sit at the right-hand end of the bar, past its separator.
   *  What goes in it is the caller's business. */
  headerAside?: React.ReactNode;
}) {
  const { meta, exportScale, job } = props;
  // Below the proxy's own size every option renders the same pixels, so the
  // distinction the help text draws would be a lie.
  const sameEitherWay = !!meta && meta.proxy_width >= meta.width;

  const help = sameEitherWay
    ? "This photo is already smaller than the proxy, so every option renders " +
      "the same pixels."
    : exportScale === "preview"
      ? "Writes the proxy render itself — the grain you are looking at, not a " +
        "downscale of the 1:1 render. Every length scales with the frame, so " +
        "at full size the same settings resolve finer, denser grain; if the " +
        "preview is the look you want, this is the file that has it."
      : exportScale === "preview_full"
        ? `The proxy render enlarged to ${meta?.width}×${meta?.height} — a ` +
          "pixel match to what is on screen, not a fresh full-resolution " +
          "render. It adds no detail: zoomed in, the grain is the same softer " +
          'proxy texture, just bigger, not the finer grain "Full size" would ' +
          "resolve at this scale. Reach for this when the preview's look is " +
          "what you want to keep, in a file sized for printing or sharing at " +
          "full size."
        : "A fresh render of every pixel at full resolution. Lengths scale " +
          "with the frame, so grain resolves finer and denser than the proxy " +
          "preview shows — judge it with Render 1:1 before committing to it.";

  // What the button used to say. It is the tooltip now -- see the note above.
  const action =
    exportScale === "preview"
      ? "Export as previewed"
      : exportScale === "preview_full"
        ? "Export as previewed, full size"
        : "Export full size";

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
        value={exportScale}
        onChange={(e) => props.onExportScale(e.target.value as ExportScale)}
      >
        <option value="full">
          Full size{meta ? ` — ${meta.width}×${meta.height}` : ""}
        </option>
        <option value="preview">
          As previewed
          {meta ? ` — ${meta.proxy_width}×${meta.proxy_height}` : ""}
        </option>
        <option value="preview_full">
          As previewed, full size
          {meta ? ` — ${meta.width}×${meta.height}` : ""}
        </option>
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
