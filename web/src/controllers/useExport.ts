/** Starting an export and polling it to the download. */

import { useState } from "react";

import type { Values } from "../models/types";
import {
  downloadUrl,
  exportStatus,
  startExport,
  type ExportJob,
  type ImageMeta,
} from "../services/api";

export function useExport(opts: {
  meta: ImageMeta | null;
  values: Values;
  supersample: number;
  referenceMp: number | null;
  scaleToRef: boolean;
  lut: string | null;
  onError: (msg: string) => void;
}) {
  const [format, setFormat] = useState("jpeg");
  /** How finely the export renders. **Not a size choice** -- every export is
   *  the source's own dimensions now, so this is quality alone.
   *
   *  Its own state rather than the Quality picker's, because the two are
   *  answering different questions: that one trades preview latency for
   *  fidelity while you work, and a file you are going to keep should not
   *  inherit whatever you left it on. Defaults to 2, which is what every preset
   *  was dialled in against. */
  const [exportSs, setExportSs] = useState(2);
  const [job, setJob] = useState<ExportJob | null>(null);

  const doExport = async () => {
    if (!opts.meta) return;
    try {
      const id = await startExport({
        id: opts.meta.id,
        params: opts.values,
        format,
        supersample: exportSs,
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

  return { format, setFormat, exportSs, setExportSs, job, setJob, doExport };
}
