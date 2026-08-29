/** The working geometry of the photograph, given the Prescaling Source values.
 *
 *  Prescaling resamples the photograph to a fixed megapixel count before the
 *  pipeline sees it, so `meta.width`/`height`/`megapixels` stop being the
 *  numbers this app should be quoting or sizing itself from -- they are facts
 *  about the *file*, and everything from the stage's zoom box to the export
 *  menu's label is about the *frame*. This is the one place that difference is
 *  worked out.
 *
 *  It is a deliberate mirror of `prescale_dims` and `Frame` in
 *  `server/models/upload.py`, with the server as the authority -- the same
 *  bargain `coerceValues` already makes with `sanitize()`. The rounding is
 *  written as `Math.floor(x + 0.5)` on both sides for one specific reason:
 *  Python's `round` is banker's rounding and `Math.round` is not, and they
 *  disagree on exactly the half-pixel case, which is the case a
 *  ratio-preserving resize is built around.
 *
 *  No React and no fetching, like everything else in here: these are the rules
 *  about what a size *is*.
 */

import type { ImageMeta } from "../services/api";
import type { Values } from "./types";

/** The section's name in `GROUPS`, and the only place the client spells it. */
export const PRESCALE_GROUP = "Prescaling Source";

export interface Geom {
  /** Pixel dimensions of the frame the pipeline renders. */
  width: number;
  height: number;
  /** Its megapixels, rounded the way `ImageMeta.megapixels` is. */
  megapixels: number;
  /** Dimensions of the proxy derived from *that* frame. */
  proxyWidth: number;
  proxyHeight: number;
  /** Linear factor from the file to the frame. 1 when prescaling is off. */
  factor: number;
  /** True when prescaling is on and actually changing the size. */
  prescaled: boolean;
}

/** Is prescaling asking for a working size at all? Mirrors `prescale_target`. */
function target(values: Values): number | null {
  if ((values.prescale ?? 0) < 0.5) return null;
  const mp = values.prescale_mp ?? 0;
  return mp > 0 ? mp : null;
}

function proxyOf(edge: number, h: number, w: number): [number, number] {
  // Mirrors `proxy_scale_at` on both `Upload` and `Frame`. The edge is a
  // property of the request rather than of the photograph, so it is passed in:
  // there is no single proxy on the server to read a size off, and a
  // photograph smaller than the edge is its own proxy at scale 1.
  const s = Math.min(1, edge / Math.max(h, w));
  return [Math.round(w * s), Math.round(h * s)];
}

export function prescaleGeom(
  meta: ImageMeta | null,
  values: Values,
  proxyEdge?: number,
): Geom | null {
  if (!meta) return null;
  const edge = proxyEdge ?? meta.proxy_edge_default;
  const mp = target(values);
  const [ipw, iph] = proxyOf(edge, meta.height, meta.width);
  const identity: Geom = {
    width: meta.width,
    height: meta.height,
    megapixels: meta.megapixels,
    proxyWidth: ipw,
    proxyHeight: iph,
    factor: 1,
    prescaled: false,
  };
  if (mp === null) return identity;

  const k = Math.sqrt((mp * 1e6) / (meta.width * meta.height));
  const w = Math.max(1, Math.floor(meta.width * k + 0.5));
  const h = Math.max(1, Math.floor(meta.height * k + 0.5));
  // A target that lands on the photograph's own size is not prescaling, and the
  // server agrees: `Upload.at()` hands the upload straight back there, so the
  // frame is the file and saying otherwise in the UI would be a lie.
  if (w === meta.width && h === meta.height) return identity;

  const [pw, ph] = proxyOf(edge, h, w);
  return {
    width: w,
    height: h,
    megapixels: Math.round((w * h) / 1e5) / 10,
    proxyWidth: pw,
    proxyHeight: ph,
    factor: k,
    prescaled: true,
  };
}

/** Where the *exported file* lands, which is a second question when prescaling
 *  is on: `prescale_output` 1 resamples the finished render back to the
 *  photograph's own dimensions. Mirrors the branch in `controllers/export.py`. */
export function exportDims(
  meta: ImageMeta | null,
  geom: Geom | null,
  values: Values,
): { width: number; height: number } | null {
  if (!meta || !geom) return null;
  const atPhotoSize = geom.prescaled && (values.prescale_output ?? 0) >= 0.5;
  return atPhotoSize
    ? { width: meta.width, height: meta.height }
    : { width: geom.width, height: geom.height };
}
