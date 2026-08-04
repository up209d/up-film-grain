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

/** A 3D LUT the server can apply: either a `.cube` in the `luts/` folder or one
 *  uploaded this session. `size` is the cube's grid resolution, and is null for
 *  folder entries because listing them deliberately does not parse them. */
export interface LutInfo {
  id: string;
  name: string;
  size: number | null;
  source: "folder" | "upload";
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
  status: "queued" | "rendering" | "encoding" | "done" | "error";
  progress: number;
  filename: string;
  width: number;
  height: number;
  size?: number;
  error?: string;
}

/** Which render the export writes. `"full"` is the source at 1:1; `"preview"`
 *  is the working proxy — the same render a slider change produces, so the
 *  grain sits on the proxy's pixel grid rather than being a downscale of the
 *  full-resolution one. They are different looks, not different sizes. */
export type ExportScale = "full" | "preview";

export async function startExport(body: {
  id: string;
  params: Record<string, number>;
  format: string;
  supersample: number;
  quality: number;
  scale: ExportScale;
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
