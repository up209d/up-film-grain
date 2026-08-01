import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  downloadUrl,
  exportStatus,
  fetchSource,
  getHealth,
  getSchema,
  renderPreview,
  startExport,
  uploadImage,
  type ExportJob,
  type ImageMeta,
  type Schema,
  type ViewRequest,
} from "./api";

type Values = Record<string, number>;

/** How the before and after are shown against each other: stacked under a wipe
 *  (with B to swap outright), or in two panes that pan and zoom together. */
export type Compare = "overlay" | "side";

/** Renders are not started from the raw value stream -- see `applied` below --
 *  but pans, zooms and typed numbers still arrive in bursts, so requests wait
 *  this long to settle. Short, because a stale preview is worse than a late
 *  one. */
const DEBOUNCE_MS = 140;

/** Zoom stops for the +/- buttons. 1 is a real 1:1 pixel view.
 *
 *  These used to be constrained to clean fractions because the server cropped
 *  and resampled at the requested zoom, and an awkward factor put the read
 *  origin on a half pixel. Zooming is a pure browser transform now, so the
 *  list is free -- it is only about how the steps feel. */
const ZOOM_STEPS = [0.05, 0.1, 0.17, 0.25, 0.33, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8];

/** Marker written into saved preset files. Only used to make a hand-inspected
 *  file self-describing -- loading deliberately does not require it, so a bare
 *  `{"intensity": 40}` typed by hand still works. */
const PRESET_FORMAT = "film-grain-preset";

export default function App() {
  const [schema, setSchema] = useState<Schema | null>(null);
  /** What the panel shows: updated on every input event, so the number beside
   *  a slider tracks the thumb in real time. Nothing renders from this. */
  const [values, setValues] = useState<Values>({});
  /** What the renderer sees: `values` as of the last committed gesture. A fit
   *  preview is seconds of work, so rendering mid-drag only queues frames that
   *  are already stale by the time they arrive. */
  const [applied, setApplied] = useState<Values>({});
  const [meta, setMeta] = useState<ImageMeta | null>(null);
  const [device, setDevice] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [renderMs, setRenderMs] = useState(0);
  /** Whether what is on screen is the full-resolution render or the proxy.
   *  Any parameter change drops back to the proxy, so this doubles as "is the
   *  1:1 render still current". */
  const [previewFull, setPreviewFull] = useState(false);
  const [renderingFull, setRenderingFull] = useState(false);

  const [supersample, setSupersample] = useState(2);
  const [compare, setCompare] = useState<Compare>("overlay");
  const [split, setSplit] = useState(1); // 1 = fully processed
  const [showBefore, setShowBefore] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const [format, setFormat] = useState("jpeg");
  const [job, setJob] = useState<ExportJob | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const urlsRef = useRef<string[]>([]);
  const srcUrlRef = useRef<string | null>(null);
  const valuesRef = useRef<Values>(values);
  valuesRef.current = values;
  const presetFileRef = useRef<HTMLInputElement | null>(null);

  /** The starting point: the server's nominated default preset if that file
   *  exists, otherwise the raw parameter defaults. Shared by boot and Reset so
   *  the two cannot drift -- "reset" meaning something different from "how it
   *  opened" is its own small bug. */
  const startingValues = useCallback((s: Schema): Values => {
    const v: Values = {};
    for (const p of s.params) v[p.key] = p.default;
    const preset = s.presets.find((x) => x.name === s.default_preset);
    return preset ? { ...v, ...preset.values } : v;
  }, []);

  // ---------------------------------------------------------------- boot --
  useEffect(() => {
    getSchema()
      .then((s) => {
        setSchema(s);
        const v = startingValues(s);
        setValues(v);
        setApplied(v);
      })
      .catch((e) => setError(String(e.message ?? e)));
    getHealth()
      .then((h) => setDevice(h.device))
      .catch(() => undefined);
  }, []);

  /** Track a *preview* object URL. Blobs stay in memory until revoked, so a
   *  long slider session would otherwise leak every intermediate render. */
  const track = (url: string) => {
    urlsRef.current.push(url);
    if (urlsRef.current.length > 4) {
      URL.revokeObjectURL(urlsRef.current.shift()!);
    }
    return url;
  };

  /** The source image is on its own lifecycle, not the preview LRU.
   *
   *  It used to share it, which was a slow-acting bug: the source is fetched
   *  once per upload while previews churn on every edit, so after four renders
   *  the source URL was the oldest entry and got revoked while an <img> was
   *  still pointing at it. The before/after compare went blank partway through
   *  a session, which looks like anything except an eviction policy. One URL,
   *  replaced explicitly. */
  const trackSource = (url: string | null) => {
    if (srcUrlRef.current) URL.revokeObjectURL(srcUrlRef.current);
    srcUrlRef.current = url;
    return url;
  };

  // ------------------------------------------------------------- request --
  /** Note what is *not* in here: no zoom, no viewport, no pan. The server
   *  renders the whole source at full resolution and the browser scales it, so
   *  navigating the preview never re-renders and never touches the network. */
  const viewBody = useCallback((): ViewRequest | null => {
    if (!meta) return null;
    return { id: meta.id, params: applied, supersample };
  }, [meta, applied, supersample]);

  // The untouched image is the same bytes for the life of an upload, and it is
  // now a full-resolution PNG -- so it is fetched once here rather than riding
  // along with every render the way it did when both were crops of a view.
  useEffect(() => {
    if (!meta) {
      setSourceUrl(trackSource(null));
      return;
    }
    const ac = new AbortController();
    fetchSource({ id: meta.id, params: {}, supersample: 1 }, ac.signal)
      .then((url) => setSourceUrl(trackSource(url)))
      .catch((e) => {
        if (e.name !== "AbortError") setError(String(e.message ?? e));
      });
    return () => ac.abort();
  }, [meta]);

  /** One render path for both fidelities -- the only difference is `full`.
   *  Sharing `abortRef` is deliberate: touching a slider during a 1:1 render
   *  cancels it, because that render is about to be wrong anyway. */
  const render = useCallback(
    (body: ViewRequest, full: boolean) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setRendering(true);
      setRenderingFull(full);
      renderPreview({ ...body, full }, ac.signal)
        .then((res) => {
          setPreviewUrl(track(res.url));
          setRenderMs(res.ms);
          setPreviewFull(res.full);
          if (res.device) setDevice(res.device);
          setError(null);
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError(String(e.message ?? e));
        })
        .finally(() => {
          if (!ac.signal.aborted) {
            setRendering(false);
            setRenderingFull(false);
          }
        });
    },
    [],
  );

  // Live edits always render the proxy. Full resolution is never automatic --
  // it is seconds of work, and firing it off every time a drag settles would
  // spend that on frames you are about to change again.
  useEffect(() => {
    const body = viewBody();
    if (!body) return;
    const t = setTimeout(() => render(body, false), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [viewBody, render]);

  const renderFull = () => {
    const body = viewBody();
    if (body) render(body, true);
  };

  // -------------------------------------------------------------- upload --
  const onFile = async (file: File) => {
    setError(null);
    try {
      const m = await uploadImage(file);
      // Drop the outgoing photo's images *before* swapping meta in. They are
      // not merely stale: the stage sizes every layer from `meta`, so the old
      // photo would be stretched to the new one's dimensions until the first
      // render lands -- and if that render failed you were left looking at the
      // previous photo with nothing saying so.
      //
      // Only on success. A rejected file (wrong format, over the size cap)
      // must leave the session exactly as it was rather than clearing the
      // stage out from under you.
      setPreviewUrl(null);
      setSourceUrl(null);
      setPreviewFull(false);
      setRenderMs(0);
      setMeta(m);
      setJob(null);
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  };

  // -------------------------------------------------------------- export --
  const doExport = async () => {
    if (!meta) return;
    try {
      const id = await startExport({
        id: meta.id,
        params: values,
        format,
        supersample,
        quality: 95,
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
      setError(String(e.message ?? e));
    }
  };

  // --------------------------------------------------------------- panel --
  const grouped = useMemo(() => {
    if (!schema) return [];
    return schema.groups
      .map((g) => ({ group: g, params: schema.params.filter((p) => p.group === g) }))
      .filter((g) => g.params.length > 0);
  }, [schema]);

  const setValue = (k: string, v: number) => setValues((s) => ({ ...s, [k]: v }));

  /** Hand the live values to the renderer. Passing the ref's current object
   *  means an uncommitted gesture is a no-op: React bails out when the state
   *  is set to the identical reference, so this costs nothing when nothing
   *  moved. */
  const commit = useCallback(() => setApplied(valuesRef.current), []);

  // A drag that ends outside the slider -- release the mouse over the image,
  // or flick past the panel edge -- never delivers `pointerup` to the input,
  // so the release is caught on the window instead.
  useEffect(() => {
    window.addEventListener("pointerup", commit);
    window.addEventListener("pointercancel", commit);
    return () => {
      window.removeEventListener("pointerup", commit);
      window.removeEventListener("pointercancel", commit);
    };
  }, [commit]);

  // Hold B to peek at the original -- the fastest way to judge an adjustment.
  useEffect(() => {
    const isTyping = (t: EventTarget | null) =>
      t instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName);
    const down = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "b" && !e.repeat && !isTyping(e.target)) {
        setShowBefore(true);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "b") setShowBefore(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // Presets and reset are single discrete actions, not gestures, so they go
  // straight through to the renderer.
  const applyPreset = (name: string) => {
    const p = schema?.presets.find((x) => x.name === name);
    if (!p) return;
    const v = { ...values, ...p.values };
    setValues(v);
    setApplied(v);
  };

  const resetAll = () => {
    if (!schema) return;
    const v = startingValues(schema);
    setValues(v);
    setApplied(v);
  };

  /** Switch the whole pipeline off, so the preview is the untouched photo.
   *  Sizes, radii and seeds are left alone -- they are not what makes a stage
   *  run, and keeping them means turning a section back on returns you to what
   *  you had rather than to the factory numbers. */
  const showOriginal = () => {
    if (!schema) return;
    setValues(schema.neutral);
    setApplied(schema.neutral);
  };

  /** Switch one section off, same idea. Reaching for this is usually "is this
   *  section even earning its keep" -- so it toggles: press it again and the
   *  section comes back exactly as it was. */
  const [muted, setMuted] = useState<Record<string, Values>>({});

  const toggleGroup = (group: string) => {
    if (!schema) return;
    const keys = schema.params.filter((x) => x.group === group).map((x) => x.key);
    const v = { ...values };
    if (muted[group]) {
      for (const k of keys) v[k] = muted[group][k];
      setMuted((m) => {
        const n = { ...m };
        delete n[group];
        return n;
      });
    } else {
      const keep: Values = {};
      for (const k of keys) {
        keep[k] = values[k];
        v[k] = schema.neutral[k];
      }
      setMuted((m) => ({ ...m, [group]: keep }));
    }
    setValues(v);
    setApplied(v);
  };

  /** Reset one section to the starting preset -- the per-section counterpart of
   *  the Reset button, and a different question from the on/off switch: the
   *  switch asks "what does this section contribute", reset asks "put it back
   *  the way it shipped". Clears the muted state too, since a section that has
   *  just been given real values is not muted any more. */
  const resetGroup = (group: string) => {
    if (!schema) return;
    const start = startingValues(schema);
    const keys = schema.params.filter((x) => x.group === group).map((x) => x.key);
    const v = { ...values };
    for (const k of keys) v[k] = start[k];
    setMuted((m) => {
      const n = { ...m };
      delete n[group];
      return n;
    });
    setValues(v);
    setApplied(v);
  };

  const isOriginal = useMemo(
    () => !!schema && Object.keys(schema.neutral).every((k) => values[k] === schema.neutral[k]),
    [values, schema],
  );

  const groupActive = (group: string) =>
    !!schema &&
    schema.params.some(
      (x) => x.group === group && values[x.key] !== schema.neutral[x.key],
    );

  // ------------------------------------------------------- preset files --
  /** Coerce an arbitrary parsed object into a complete, in-range value set.
   *
   *  This mirrors `sanitize()` on the server: unknown keys are dropped, values
   *  are clamped, and anything missing falls back to its default rather than to
   *  whatever the sliders happen to be showing -- a preset file describes a
   *  whole look, so loading one must not leave stray state from the last one.
   *  It is why a file written before a slider's range changed still loads. */
  const coerce = (raw: unknown): { values: Values; dropped: string[] } => {
    if (!schema) throw new Error("Schema not loaded yet.");
    if (!raw || typeof raw !== "object") throw new Error("Not a JSON object.");
    // Accept both our own wrapper and a bare {key: value} map, so the file
    // stays hand-editable.
    const obj = raw as Record<string, unknown>;
    const src = (obj.values ?? obj) as Record<string, unknown>;
    if (!src || typeof src !== "object") throw new Error("No `values` object.");

    const v: Values = {};
    for (const p of schema.params) v[p.key] = p.default;
    const dropped: string[] = [];
    let matched = 0;
    for (const [k, val] of Object.entries(src)) {
      const p = schema.params.find((x) => x.key === k);
      const n = Number(val);
      if (!p || !Number.isFinite(n)) {
        dropped.push(k);
        continue;
      }
      v[k] = Math.min(p.max, Math.max(p.min, n));
      matched++;
    }
    if (!matched) throw new Error("No recognised parameters in that file.");
    return { values: v, dropped };
  };

  const savePreset = () => {
    if (!schema) return;
    const name = window.prompt("Preset name", "my-look")?.trim();
    if (!name) return;
    const file = {
      format: PRESET_FORMAT,
      version: 1,
      name,
      values: Object.fromEntries(
        // Written in schema order, not insertion order, so a hand-edited file
        // stays readable and two saves diff cleanly.
        schema.params.map((p) => [p.key, values[p.key] ?? p.default]),
      ),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(file, null, 2)], { type: "application/json" }),
    );
    const filename = `${name.replace(/[^\w.-]+/g, "-")}.json`;
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    // In the document and revoked on a later tick: Safari cancels the download
    // if the blob URL is revoked in the same task as the click, and Firefox
    // wants the anchor to actually be in the tree.
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    setError(null);
    // A browser download is silent, so say something -- otherwise the button
    // looks like it did nothing.
    setNotice(`Saved ${filename}`);
  };

  const loadPreset = async (file: File) => {
    try {
      const { values: v, dropped } = coerce(JSON.parse(await file.text()));
      setValues(v);
      setApplied(v); // discrete action -- render straight away
        setError(null);
      setNotice(
        dropped.length
          ? `Loaded ${file.name} — ignored unknown key${
              dropped.length > 1 ? "s" : ""
            }: ${dropped.join(", ")}`
          : `Loaded ${file.name}`,
      );
    } catch (e: any) {
      setNotice(null);
      setError(`Could not load ${file.name}: ${e.message ?? e}`);
    }
  };

  // -------------------------------------------------------------- render --
  return (
    <div className="app">
      <header className="bar">
        <div className="brand">
          <span className="dot" />
          Film Grain Engine
        </div>
        <label className="btn">
          Open image
          <input
            type="file"
            accept="image/jpeg,image/png"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              // Cleared so picking the *same* file again still fires a change
              // event. Without this, re-opening the photo you already have
              // loaded does nothing at all and reads as the app being stuck.
              e.target.value = "";
              if (f) onFile(f);
            }}
          />
        </label>
        {meta && (
          <span className="meta">
            {meta.name} · {meta.width}×{meta.height} · {meta.megapixels}MP
          </span>
        )}
        <div className="spacer" />
        <span className={`status ${rendering ? "busy" : ""}`}>
          {rendering ? "rendering…" : `${renderMs}ms`}
        </span>
        <span className="meta">{device}</span>
      </header>

      <main className="body">
        <Stage
          meta={meta}
          previewUrl={previewUrl}
          sourceUrl={sourceUrl}
          compare={compare}
          split={showBefore ? 0 : split}
          previewFull={previewFull}
          onFile={onFile}
        />

        <aside className="panel">
          {error && <div className="err">{error}</div>}

          <Field label="Compare">
            <div className="row">
              <button
                className={compare === "overlay" ? "seg on" : "seg"}
                onClick={() => setCompare("overlay")}
              >
                Overlay
              </button>
              <button
                className={compare === "side" ? "seg on" : "seg"}
                onClick={() => setCompare("side")}
              >
                Side by side
              </button>
            </div>
          </Field>
          {compare === "overlay" && (
            <Field label="Wipe">
              <button
                className={showBefore ? "swap on" : "swap"}
                onClick={() => setShowBefore((v) => !v)}
                disabled={!meta}
                title="Swap before/after — or hold B"
              >
                {showBefore ? "Before" : "After"}
              </button>
              <input
                type="range"
                min={0}
                max={1}
                step={0.001}
                value={split}
                disabled={showBefore}
                onChange={(e) => setSplit(Number(e.target.value))}
              />
            </Field>
          )}
          <Field label="Preview fidelity">
            <button
              className="btn"
              onClick={renderFull}
              disabled={!meta || previewFull || rendering}
              title="Render the whole frame at full resolution"
            >
              {renderingFull
                ? "Rendering 1:1…"
                : previewFull
                  ? "1:1 — up to date"
                  : "Render 1:1"}
            </button>
          </Field>
          <p className="hint">
            Editing renders a{" "}
            {meta ? `${meta.proxy_width}×${meta.proxy_height}` : "proxy"} proxy
            so sliders stay responsive. It predicts structure but cannot resolve
            the finest grain — <strong>Render 1:1</strong> for the exact
            exported pixels, and judge grain there at 100% zoom. Any adjustment
            drops back to the proxy.
          </p>
          <p className="hint">
            Zoom and pan live on the preview itself and never re-render.
          </p>

          <Field label="Quality">
            <select
              value={supersample}
              onChange={(e) => setSupersample(Number(e.target.value))}
            >
              <option value={1}>1× (fast, aliased grain)</option>
              <option value={2}>2× supersampled</option>
              <option value={3}>3× supersampled (slowest)</option>
            </select>
          </Field>

          <div className="row">
            <select
              defaultValue=""
              onChange={(e) => e.target.value && applyPreset(e.target.value)}
            >
              <option value="">Preset…</option>
              {schema?.presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
            <button
              className="btn ghost"
              onClick={showOriginal}
              disabled={isOriginal}
              title="Switch every stage off — show the untouched photo"
            >
              Original
            </button>
            <button className="btn ghost" onClick={resetAll} title="Back to the starting preset">
              Reset
            </button>
          </div>

          <div className="row">
            <button
              className="btn ghost"
              onClick={savePreset}
              disabled={!schema}
              title="Write the current settings to a .json file"
            >
              Save to file…
            </button>
            <button
              className="btn ghost"
              onClick={() => presetFileRef.current?.click()}
              disabled={!schema}
              title="Load settings from a .json file"
            >
              Load file…
            </button>
            <input
              ref={presetFileRef}
              type="file"
              accept="application/json,.json"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                // Cleared so picking the same file twice still fires a change
                // event -- otherwise re-loading a file you just edited is a
                // no-op with no feedback.
                e.target.value = "";
                if (f) loadPreset(f);
              }}
            />
          </div>
          {notice && <p className="note">{notice}</p>}

          <div className="groups">
            {grouped.map(({ group, params }) => (
              <section key={group} className="group">
                <h3
                  onClick={() =>
                    setCollapsed((c) => ({ ...c, [group]: !c[group] }))
                  }
                >
                  <span className={collapsed[group] ? "chev" : "chev open"}>›</span>
                  {group}
                  {/* Both act on this section only, and both have to stop
                      propagation -- the header itself toggles collapse, so
                      without it either one would also fold the section shut. */}
                  <button
                    className="grpbtn"
                    title={`Reset ${group} to the starting preset`}
                    onClick={(e) => {
                      e.stopPropagation();
                      resetGroup(group);
                    }}
                  >
                    ↺
                  </button>
                  <button
                    className={muted[group] ? "grpbtn on" : "grpbtn"}
                    title={
                      muted[group]
                        ? `Switch ${group} back on`
                        : `Switch ${group} off`
                    }
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleGroup(group);
                    }}
                  >
                    {muted[group] ? "○" : "●"}
                  </button>
                </h3>
                {!collapsed[group] &&
                  params.map((p) => (
                    <div className="slider" key={p.key}>
                      <div className="slabel">
                        <Help text={p.help} label={p.label} />
                        <input
                          className="num"
                          type="number"
                          min={p.min}
                          max={p.max}
                          step={p.step}
                          value={values[p.key] ?? p.default}
                          onChange={(e) =>
                            setValue(p.key, Number(e.target.value))
                          }
                          onKeyUp={commit}
                          onBlur={commit}
                        />
                        {p.unit && <em>{p.unit}</em>}
                      </div>
                      <input
                        type="range"
                        min={p.min}
                        max={p.max}
                        step={p.step}
                        value={values[p.key] ?? p.default}
                        onChange={(e) => setValue(p.key, Number(e.target.value))}
                        // Pointer releases come from the window listener above;
                        // these cover the keyboard path (arrows nudge the thumb
                        // without ever producing a pointer event).
                        onKeyUp={commit}
                        onBlur={commit}
                      />
                    </div>
                  ))}
              </section>
            ))}
          </div>

          <div className="export">
            <select value={format} onChange={(e) => setFormat(e.target.value)}>
              <option value="jpeg">JPEG 95</option>
              <option value="png16">PNG 16-bit</option>
              <option value="png8">PNG 8-bit</option>
            </select>
            <button className="btn primary" onClick={doExport} disabled={!meta}>
              Export full size
            </button>
          </div>
          {job && job.status !== "done" && (
            <div className="job">
              {job.status === "error" ? (
                <span className="err">{job.error}</span>
              ) : (
                <>
                  <div className="pbar">
                    <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
                  </div>
                  <span>
                    {job.status} {Math.round(job.progress * 100)}%
                  </span>
                </>
              )}
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}

/** Slider title plus a help badge, sharing one tooltip.
 *
 *  The native `title` attribute this replaced was effectively invisible: it
 *  needs a long hover, cannot be triggered by keyboard, and vanishes on any
 *  movement. This one is positioned against the viewport rather than the
 *  scrolling panel, so it is never clipped by the panel's overflow.
 *
 *  The whole title is the hover target, not just the badge -- the label already
 *  carries `cursor: help`, and a 15px circle is far too small a thing to have
 *  to find before the explanation will show. The badge stays as the visible
 *  affordance and as the keyboard/tap target, and is what the tooltip is
 *  positioned against so its placement does not depend on label length. */
function Help({ text, label }: { text: string; label: string }) {
  const [at, setAt] = useState<{ top: number; left: number } | null>(null);
  const ref = useRef<HTMLButtonElement | null>(null);

  const open = () => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    // Sit to the left of the panel, vertically centred on the badge, clamped
    // so it cannot run off the top or bottom of the window.
    setAt({
      top: Math.min(Math.max(8, r.top - 8), window.innerHeight - 120),
      left: Math.max(8, r.left - 268),
    });
  };
  const close = () => setAt(null);

  if (!text) return <span className="title">{label}</span>;
  return (
    <span className="title" onMouseEnter={open} onMouseLeave={close}>
      {label}
      <button
        ref={ref}
        className="help"
        type="button"
        aria-label={`What does ${label} do?`}
        onFocus={open}
        onBlur={close}
        onClick={(e) => {
          e.preventDefault();
          if (at) close();
          else open();
        }}
      >
        ?
      </button>
      {at && (
        <div className="tip" style={{ top: at.top, left: at.left }} role="tooltip">
          {text}
        </div>
      )}
    </span>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="fieldbody">{children}</div>
    </div>
  );
}

/** The preview canvas: two compare modes, and all of the zooming.
 *
 *  Zoom and pan are purely a display transform here. The server renders the
 *  whole source at full resolution once per parameter change and the browser
 *  scales it, so navigating costs nothing and never waits on a render. It also
 *  removes an entire class of bug: the old server-side crop had to snap its
 *  read origin to whole working pixels, because a crop starting mid-pixel
 *  resolves on a different grid phase than a whole-image downscale and shows a
 *  half-pixel shift on hard edges. Nothing samples anything here, so any zoom
 *  value is now safe -- the steps below are for feel, not correctness.
 *
 *  The trade is at the other end: fitting a full-resolution render into the
 *  viewport is a browser downscale, so fine grain averages away on screen.
 *  Zoom to 100% to judge grain; that view is exact. */
function Stage(props: {
  meta: ImageMeta | null;
  previewUrl: string | null;
  sourceUrl: string | null;
  compare: Compare;
  split: number;
  previewFull: boolean;
  onFile: (f: File) => void;
}) {
  const { meta, previewUrl, sourceUrl, compare, split, previewFull } = props;
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ x: number; y: number; cx: number; cy: number } | null>(null);
  // null = fit: follow the container instead of holding a fixed factor, so
  // resizing the window keeps the whole frame visible.
  const [zoom, setZoom] = useState<number | null>(null);
  const [center, setCenter] = useState({ x: 0.5, y: 0.5 });
  const [pane, setPane] = useState({ w: 0, h: 0 });

  // A new image inherits neither the old pan nor the old magnification -- a
  // corner crop of the last photo is never where you want to land.
  useEffect(() => {
    setCenter({ x: 0.5, y: 0.5 });
    setZoom(null);
  }, [props.meta?.id]);

  // A pane, not the stage: in side-by-side each image gets half the width, so
  // fit has to be computed against what the image is actually shown in.
  // Measured synchronously before paint as well as on resize, or the first
  // frame after an upload is laid out against a zero-width pane.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => {
      const n = el.querySelector(".pane");
      if (n) setPane({ w: n.clientWidth, h: n.clientHeight });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [compare, !!meta]);

  const iw = meta?.width ?? 1;
  const ih = meta?.height ?? 1;
  const fitZoom = Math.min(pane.w / iw, pane.h / ih) || 1;
  const eff = zoom ?? fitZoom;
  const dw = iw * eff;
  const dh = ih * eff;
  const canPan = dw > pane.w + 1 || dh > pane.h + 1;

  const stepZoom = (dir: 1 | -1) => {
    const next =
      dir > 0
        ? ZOOM_STEPS.find((z) => z > eff * 1.001)
        : [...ZOOM_STEPS].reverse().find((z) => z < eff * 0.999);
    if (next !== undefined) setZoom(next);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!canPan) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, cx: center.x, cy: center.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    // One screen pixel is 1/dw of the image's displayed width, so the same
    // delta moves both panes by the same fraction of the same image -- which
    // is what keeps the two sides locked together in side-by-side.
    setCenter({
      x: Math.min(1, Math.max(0, d.cx - (e.clientX - d.x) / dw)),
      y: Math.min(1, Math.max(0, d.cy - (e.clientY - d.y) / dh)),
    });
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  if (!meta) {
    return (
      <div
        className="stage empty"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f) props.onFile(f);
        }}
      >
        <div className="drop">
          <strong>Drop a photo here</strong>
          <span>or use “Open image”. JPEG or PNG, up to 30MB.</span>
        </div>
      </div>
    );
  }

  // Place the frame so the source point at `center` sits at the middle of the
  // pane, clamped so a pannable image cannot be dragged off its own edges and
  // a smaller-than-pane image simply centres.
  const place = (): React.CSSProperties => {
    const left =
      dw <= pane.w
        ? (pane.w - dw) / 2
        : Math.min(0, Math.max(pane.w - dw, pane.w / 2 - center.x * dw));
    const top =
      dh <= pane.h
        ? (pane.h - dh) / 2
        : Math.min(0, Math.max(pane.h - dh, pane.h / 2 - center.y * dh));
    return {
      width: dw,
      height: dh,
      left,
      top,
      // Past 2x the honest thing is to show the actual pixels rather than let
      // the browser invent smooth ones between them.
      imageRendering: eff >= 2 ? "pixelated" : "auto",
    };
  };

  // The proxy resolves detail only up to its own resolution; past that the
  // browser is enlarging it and grain is not being shown honestly. Say so
  // rather than letting a soft preview read as a soft result.
  const proxyLimit = (meta?.proxy_width ?? 0) / iw;
  const softened = !previewFull && eff > proxyLimit * 1.05;

  const bar = (
    <div className="viewbar">
      {softened && (
        <span className="fid" title="Enlarged beyond the proxy's resolution — press Render 1:1 to judge grain">
          proxy
        </span>
      )}
      <button
        className={zoom === null ? "seg on" : "seg"}
        onClick={() => setZoom(null)}
      >
        Fit
      </button>
      <button
        className="seg icon"
        onClick={() => stepZoom(-1)}
        disabled={eff <= ZOOM_STEPS[0] * 1.001}
        title="Zoom out"
      >
        −
      </button>
      <button
        className="seg zoomval"
        onClick={() => setZoom(eff === 1 ? null : 1)}
        title="Zoom to 1:1"
      >
        {Math.round(eff * 100)}%
      </button>
      <button
        className="seg icon"
        onClick={() => stepZoom(1)}
        disabled={eff >= ZOOM_STEPS[ZOOM_STEPS.length - 1] * 0.999}
        title="Zoom in"
      >
        +
      </button>
    </div>
  );

  if (compare === "side") {
    return (
      <div className={`stage side ${canPan ? "grab" : ""}`} ref={wrapRef}>
        {bar}
        {/* Both panes read the same zoom and centre, so one drag anywhere in
            the stage moves both images identically. */}
        <div
          className="panes"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <div className="pane">
            <span className="tag">Before</span>
            {sourceUrl && (
              <img className="frame" style={place()} src={sourceUrl} alt="before" draggable={false} />
            )}
          </div>
          <div className="pane">
            <span className="tag">After</span>
            {previewUrl && (
              <img className="frame" style={place()} src={previewUrl} alt="after" draggable={false} />
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`stage ${canPan ? "grab" : ""}`} ref={wrapRef}>
      {bar}
      <div
        className="panes"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div className="pane">
          {sourceUrl && (
            <img className="frame" style={place()} src={sourceUrl} alt="before" draggable={false} />
          )}
          {previewUrl && (
            <img
              className="frame"
              style={{ ...place(), clipPath: `inset(0 0 0 ${(1 - split) * 100}%)` }}
              src={previewUrl}
              alt="after"
              draggable={false}
            />
          )}
          {split > 0.001 && split < 0.999 && (
            <div
              className="divider"
              style={{ left: `${(place().left as number) + (1 - split) * dw}px` }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
