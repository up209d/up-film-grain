/** Starting an export and polling it to the download. */

import { useState } from "react";

import type { Values } from "../models/types";
import {
  downloadUrl,
  exportStatus,
  startExport,
  type ExportJob,
  type ExportScale,
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
  /** Full resolution, or the proxy exactly as previewed. Not a size choice:
   *  the proxy renders every length at proxy scale, so its grain is the grain
   *  on screen. Downscaling a 1:1 export to the same pixels would not match.
   *
   *  Defaults to the enlarged proxy rather than a fresh full-size render,
   *  changed on request: what you dialled in is what the preview showed you, so
   *  the file that matches it is the one that starts from those pixels. A
   *  full-size render of the same numbers is a *different* picture -- finer,
   *  denser grain -- and having that be the default made the export quietly
   *  disagree with the screen. */
  const [exportScale, setExportScale] = useState<ExportScale>("preview_full");
  const [job, setJob] = useState<ExportJob | null>(null);

  const doExport = async () => {
    if (!opts.meta) return;
    try {
      const id = await startExport({
        id: opts.meta.id,
        params: opts.values,
        format,
        supersample: opts.supersample,
        quality: 95,
        scale: exportScale,
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

  return { format, setFormat, exportScale, setExportScale, job, setJob, doExport };
}
