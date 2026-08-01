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
}

export interface Preset {
  name: string;
  values: Record<string, number>;
  /** Megapixels the preset was dialled in on, if it says. */
  reference_mp: number | null;
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
  size?: number;
  error?: string;
}

export async function startExport(body: {
  id: string;
  params: Record<string, number>;
  format: string;
  supersample: number;
  quality: number;
  reference_mp?: number | null;
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
