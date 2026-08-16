export interface Param {
  key: string;
  label: string;
  group: string;
  min: number;
  max: number;
  step: number;
  default: number;
  unit: string;
  help: string;
  /** Names for a discrete parameter, indexed by value. Non-empty means the
   *  control is a menu rather than a slider -- the value is still a number,
   *  so nothing else here has to care. */
  choices: string[];
  /** True for an on/off control, rendered as a checkbox. The value is still a
   *  number -- 0 or 1 -- so nothing else here has to care either. Declared
   *  explicitly because the client's `Param` is a hand-written mirror of the
   *  server dataclass rather than a generated one, and a field left out here is
   *  simply invisible: `spatial` has been shipping unread for exactly that
   *  reason. */
  toggle: boolean;
}

export interface Preset {
  name: string;
  values: Record<string, number>;
  /** Megapixels the preset was dialled in on, if it says. */
  reference_mp: number | null;
  /** 3D LUT the look wants, by id. A sibling of the values rather than one of
   *  them: a LUT is a resource identified by name, not a quantity, so it cannot
   *  be a number in the schema — see server/lut.py. */
  lut: string | null;
}

/** A 3D LUT the server can apply: either a `.cube` under `luts/` or one
 *  uploaded this session. `size` is the cube's grid resolution, and is null for
 *  folder entries because listing them deliberately does not parse them. */
export interface LutInfo {
  /** The path relative to `luts/` with the extension dropped, POSIX-separated
   *  — `UP-SuperPortra` at the root, `gmic/bw/agfa_apx_100` in a folder. This
   *  is what a preset file records, so it has to survive a restart. */
  id: string;
  /** The bare filename, for the picker to print. It is `id`'s last segment. */
  name: string;
  size: number | null;
  source: "folder" | "upload";
  /** Folder it came from, relative to `luts/`; `""` at the root and for every
   *  upload. The picker groups on it — reported rather than split back out of
   *  `id` here, because the id is a path and the server owns paths. */
  group: string;
}

export interface Schema {
  groups: string[];
  params: Param[];
  presets: Preset[];
  /** Preset to open on, and the one Reset returns to. Null when no file by
   *  that name exists, in which case the parameter defaults are the start. */
  default_preset: string | null;
  /** Values that switch every stage off. Rendering with these returns the
   *  source untouched, bit for bit. */
  neutral: Record<string, number>;
  /** Which keys in `neutral` are the *amounts* — the ones that decide whether
   *  a stage runs. Everything else (sizes, radii, seeds) is a shape, and may
   *  differ from its neutral value without the picture differing at all. This
   *  is the server's own `NEUTRAL_ZERO`; see the note beside it in
   *  `server/params/schema.py` for why it is shipped instead of guessed. */
  neutral_zero: string[];
  /** Fallback size for presets that do not record one. Null = no scaling. */
  default_reference_mp: number | null;
}

export interface ImageMeta {
  id: string;
  name: string;
  width: number;
  height: number;
  megapixels: number;
  proxy_width: number;
  proxy_height: number;
}

/** No view geometry: the server renders the whole frame and the browser does
 *  the scaling, so zoom and pan never reach the API. `full` picks proxy scale
 *  (fast, for live editing) or full resolution (exact, on demand). */
export interface ViewRequest {
  id: string;
  params: Record<string, number>;
  supersample: number;
  full?: boolean;
  /** Megapixels of the image these values were dialled in on. The server
   *  rescales every length by the *linear* ratio to the current image, so a
   *  preset keeps its look on a bigger or smaller photo. Omit for no scaling. */
  reference_mp?: number | null;
  /** Which 3D LUT to apply, by id. Beside the params rather than in them, for
   *  the same reason `reference_mp` is: it is not a number. An id the server
   *  cannot resolve is not an error — it renders with no LUT and zeroes the mix,
   *  so a preset naming a deleted file still loads. */
  lut?: string | null;
}

export interface RenderResult {
  url: string;
  ms: number;
  device: string;
  full: boolean;
}

async function fail(r: Response): Promise<never> {
  let detail = r.statusText;
  try {
    const j = await r.json();
    detail = j.detail ?? detail;
  } catch {
    /* body was not json */
  }
  throw new Error(detail);
}

export async function getSchema(): Promise<Schema> {
  const r = await fetch("/api/params");
  if (!r.ok) return fail(r);
  return r.json();
}

export async function getHealth(): Promise<{ device: string }> {
  const r = await fetch("/api/health");
  if (!r.ok) return fail(r);
  return r.json();
}

/** Every LUT the picker can offer. Its own request rather than a field on the
 *  schema because this list grows during a session — uploading one adds to it —
 *  while the parameter schema never changes. */
export async function getLuts(): Promise<LutInfo[]> {
  const r = await fetch("/api/luts");
  if (!r.ok) return fail(r);
  return (await r.json()).luts;
}

/** Hand a `.cube` to the server, which parses it now and keeps it for this
 *  session. Parsed on upload rather than at render time so a malformed file is
 *  reported while the file picker is still on screen. */
export async function uploadLut(file: File): Promise<LutInfo> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/lut", { method: "POST", body: fd });
  if (!r.ok) return fail(r);
  return r.json();
}

export async function uploadImage(file: File): Promise<ImageMeta> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  if (!r.ok) return fail(r);
  return r.json();
}

export async function renderPreview(
  body: ViewRequest,
  signal: AbortSignal,
): Promise<RenderResult> {
  const r = await fetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) return fail(r);
  const blob = await r.blob();
  return {
    url: URL.createObjectURL(blob),
    ms: Number(r.headers.get("X-Render-Ms") ?? 0),
    device: r.headers.get("X-Render-Device") ?? "",
    full: r.headers.get("X-Render-Full") === "1",
  };
}

export async function fetchSource(
  body: ViewRequest,
  signal: AbortSignal,
): Promise<string> {
  const r = await fetch("/api/source", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) return fail(r);
  return URL.createObjectURL(await r.blob());
}

export interface ExportJob {
  id: string;
  status: "queued" | "rendering" | "upscaling" | "encoding" | "done" | "error";
  progress: number;
  filename: string;
  width: number;
  height: number;
  size?: number;
  error?: string;
}

/** Supersample factors the export offers. **Every export is full size** since
 *  2026-08-08 — this picks how finely the frame is rendered, not how big the
 *  file is.
 *
 *  It replaced a three-way scale menu whose entries moved resolution and look
 *  together: "As previewed" wrote a smaller file *and* a coarser grain, because
 *  every length scales with the frame. Two questions on one control. Below 1
 *  the frame renders smaller than its output and is resampled back up; above,
 *  finer than the output grid and integrated down.
 *
 *  Since 2026-08-09 these five render the **preview tier** and are enlarged to
 *  the source's dimensions — see `EXPORT_OPTIONS` for the sixth entry, which is
 *  the one that does not. */
export const EXPORT_SUPERSAMPLES = [0.5, 1, 1.5, 2, 3] as const;

export interface ExportOption {
  /** Menu value. A string, not the factor, because the factor no longer
   *  identifies an entry on its own — `ss1` and `full` both render at 1×. */
  key: string;
  ss: number;
  /** Which tier gets rendered: the proxy the preview shows (`false`, then
   *  enlarged), or the source itself at 1.0 (`true`, nothing to enlarge). */
  full: boolean;
}

/** The export menu, in order.
 *
 *  Five preview-tier entries — the file is the picture you judged, enlarged —
 *  plus **Full size**, added 2026-08-09 on request: a genuine full-resolution
 *  render at 1× supersampling. That last one is the only entry whose pixels are
 *  not the preview's, and it is deliberately *not* the default: it resolves
 *  finer, denser grain than the screen showed, which is a different picture
 *  rather than a sharper one. */
export const EXPORT_OPTIONS: readonly ExportOption[] = [
  ...EXPORT_SUPERSAMPLES.map((ss) => ({ key: `ss${ss}`, ss, full: false })),
  { key: "full", ss: 1, full: true },
];

/** Opens on the previewed frame at 2×, unchanged. */
export const EXPORT_DEFAULT_KEY = "ss2";

export const exportOption = (key: string): ExportOption =>
  EXPORT_OPTIONS.find((o) => o.key === key) ??
  EXPORT_OPTIONS.find((o) => o.key === EXPORT_DEFAULT_KEY)!;

export async function startExport(body: {
  id: string;
  params: Record<string, number>;
  format: string;
  supersample: number;
  /** Render the source at 1.0 instead of the preview tier. Absent means the
   *  preview tier, which is what five of the six entries ask for. */
  full?: boolean;
  quality: number;
  reference_mp?: number | null;
  lut?: string | null;
}): Promise<string> {
  const r = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) return fail(r);
  return (await r.json()).job;
}

export async function exportStatus(job: string): Promise<ExportJob> {
  const r = await fetch(`/api/export/${job}`);
  if (!r.ok) return fail(r);
  return r.json();
}

export const downloadUrl = (job: string) => `/api/export/${job}/download`;
