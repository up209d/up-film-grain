/** Starting an export and polling it to the download. */

import { useState } from "react";

import type { Values } from "../models/types";
import {
  downloadUrl,
  exportOption,
  exportStatus,
  startExport,
  EXPORT_DEFAULT_KEY,
  type ExportJob,
  type ImageMeta,
} from "../services/api";

export function useExport(opts: {
  meta: ImageMeta | null;
  values: Values;
  supersample: number;
  /** The edge the preview is rendering at. The export must render the same
   *  tier, or the file is not the picture the look was judged on. */
  proxyEdge: number;
  referenceMp: number | null;
  scaleToRef: boolean;
  lut: string | null;
  onError: (msg: string) => void;
}) {
  const [format, setFormat] = useState("jpeg");
  /** Which export entry is selected. **Not a size choice** -- every export is
   *  the source's own dimensions, so this is quality and tier alone.
   *
   *  A key rather than the factor since 2026-08-09, when a sixth entry arrived:
   *  five render the *preview tier* at their factor and enlarge it, so the file
   *  matches the frame the settings were judged on, and `full` renders the
   *  source itself at 1.0. `ss1` and `full` are both 1x, so the number can no
   *  longer say which one was picked.
   *
   *  Its own state rather than the Quality picker's, because the two are
   *  answering different questions: that one trades preview latency for
   *  fidelity while you work, and a file you are going to keep should not
   *  inherit whatever you left it on. Opens on the previewed frame at 2x, which
   *  is what every preset was dialled in against. */
  const [exportKey, setExportKey] = useState(EXPORT_DEFAULT_KEY);
  const [job, setJob] = useState<ExportJob | null>(null);

  const doExport = async () => {
    if (!opts.meta) return;
    try {
      const opt = exportOption(exportKey);
      const id = await startExport({
        id: opts.meta.id,
        params: opts.values,
        format,
        supersample: opt.ss,
        proxy_edge: opts.proxyEdge,
        full: opt.full,
        quality: 95,
        reference_mp: opts.scaleToRef ? opts.referenceMp : null,
        lut: opts.lut,
      });
      const poll = async () => {
        const s = await exportStatus(id);
        setJob(s);
        if (s.status === "done") {
          const a = document.createElement("a");
          a.href = downloadUrl(id);
          a.download = s.filename;
          a.click();
          return;
        }
        if (s.status !== "error") setTimeout(poll, 400);
      };
      poll();
    } catch (e: any) {
      opts.onError(String(e.message ?? e));
    }
  };

  return { format, setFormat, exportKey, setExportKey, job, setJob, doExport };
}
