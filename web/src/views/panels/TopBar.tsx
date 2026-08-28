/** The app bar: open a photo, the seed-on-open switch, and the render readout. */

import filmGrain1x1 from "../../assets/film-grain-1x1.jpg";
import type { Geom } from "../../models/prescale";
import type { ImageMeta } from "../../services/api";

/** How long a preview may take before the bar says the config is heavy, in ms.
 *
 *  Two numbers because the two devices are two different machines: 5s on a GPU
 *  and 10s on a CPU are the budgets this app is held to. Keyed off the device
 *  string the server already returns rather than off a guess -- a CPU-only
 *  machine is not a slow GPU, and warning it at 5s would mean warning it
 *  always. */
const SLOW_MS_GPU = 5000;
const SLOW_MS_CPU = 10000;

/** Where the AGPL section 13 "Source" link in the bar points. A modified,
 *  network-served build is required to offer *its own* corresponding source, so
 *  a fork that deploys this publicly must change this constant -- it is a lone
 *  constant rather than an inline href for that reason. */
const SOURCE_URL = "https://github.com/up209d/up-film-grain";

/** The supersample factors the picker offers, lowest first. Mirrors
 *  `SUPERSAMPLES` in `server/models/upload.py`; the server clamps to its own
 *  list, so a drift here degrades to a nearby factor rather than an error. */
const SS_STEPS = [0.5, 1, 1.5, 2, 3];

export default function TopBar(props: {
  meta: ImageMeta | null;
  /** The frame being rendered, when it is not the file. Shown after the file's
   *  own dimensions rather than instead of them: the file's size is a fact
   *  about the photograph and this readout is where you look for it, while the
   *  frame's is what everything else in the app is now quoting, so a session
   *  with prescaling on needs both visible in one place. */
  geom?: Geom | null;
  device: string;
  rendering: boolean;
  renderMs: number;
  supersample: number;
  onSupersample: (v: number) => void;
  dropping: boolean;
  onDropping: (v: boolean) => void;
  onFile: (f: File) => void;
  randomizeSeedOnOpen: boolean;
  onRandomizeSeedOnOpen: (v: boolean) => void;
  onRandomizeSeeds: () => void;
}) {
  const { meta } = props;
  // The device string is the server's own words ("Apple GPU (MPS)", "CUDA",
  // "CPU"), so match on the one that means no accelerator rather than trying to
  // enumerate the others.
  const budget = /cpu/i.test(props.device) ? SLOW_MS_CPU : SLOW_MS_GPU;
  const slow = !props.rendering && props.renderMs > budget;
  // The next factor down, or null at the bottom of the list -- at 0.5x there is
  // nothing left to suggest and the warning should say so by omission rather
  // than offering a button that does nothing.
  const lower = SS_STEPS.filter((s) => s < props.supersample).pop() ?? null;
  return (
    <header className="bar">
      <div className="brand">
        <img src={filmGrain1x1} alt="Film grain" className="film-grain-icon" />
        Film Grain Engine
        {/* AGPL section 13: a network-served copy has to offer its source to
            the people using it, and the FSF's own suggested form is a "Source"
            link in the interface. Shipping it in the original rather than
            leaving it to forks is the cheap move -- it sets the norm, and a
            fork that strips the link is then visibly in breach instead of
            merely silent. If you deploy a MODIFIED build, this must point at
            your source, not at this repository. */}
        <a
          className="source-link"
          href={SOURCE_URL}
          target="_blank"
          rel="noreferrer noopener"
          title="Source code (AGPL-3.0) -- required by AGPL section 13"
        >
          Source
        </a>
      </div>
      <label
        className={`btn${props.dropping ? " dropping" : ""}`}
        // dragOver has to preventDefault on *every* event, not just the
        // first: the browser reads the absence of it as "this target does
        // not accept drops" and falls back to navigating to the file.
        onDragOver={(e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
          if (!props.dropping) props.onDropping(true);
        }}
        onDragLeave={() => props.onDropping(false)}
        onDrop={(e) => {
          e.preventDefault();
          props.onDropping(false);
          const f = e.dataTransfer.files?.[0];
          if (f) props.onFile(f);
        }}
      >
        Open image (drop here)
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
            if (f) props.onFile(f);
          }}
        />
      </label>
      <label
        className="checkfield"
        title={
          "Reroll the grain Seed and Texture Seed whenever a photo is " +
          "opened, so different photos don't render with the identical " +
          "grain and damage pattern. While it is on, a preset brings its " +
          "look but leaves those two seeds alone -- otherwise choosing one " +
          "would put every photo back on the seed it was saved with. Off " +
          "keeps whatever seeds are currently dialled in, presets included."
        }
      >
        <input
          type="checkbox"
          checked={props.randomizeSeedOnOpen}
          onChange={(e) => props.onRandomizeSeedOnOpen(e.target.checked)}
        />
        With random seed
      </label>
      {meta && (
        <span className="meta">
          {meta.name} · {meta.width}×{meta.height} · {meta.megapixels}MP
          {props.geom?.prescaled && (
            <> → {props.geom.width}×{props.geom.height} · {props.geom.megapixels}MP</>
          )}
        </span>
      )}
      {/* The same reroll the checkbox above does on open, on demand: a seed is
          not a quantity you dial in, it is one you keep drawing until the grain
          and the damage fall somewhere you like. Beside the filename because it
          is about the photo that is loaded, and hidden without one for the same
          reason. */}
      {meta && (
        <button
          className="btn ghost"
          onClick={props.onRandomizeSeeds}
          title="Draw a new grain Seed and Texture Seed, leaving every other value alone"
        >
          Random Seed
        </button>
      )}
      <div className="spacer" />
      <span className={`status ${props.rendering ? "busy" : ""}${slow ? " slow" : ""}`}>
        {props.rendering && <div className="spinner" />}
        {props.rendering ? "rendering…" : `${props.renderMs}ms`}
      </span>
      {slow && (
        <span className="meta warn">
          heavy config
          {lower !== null && (
            <>
              {" — "}
              <button
                className="linkish"
                onClick={() => props.onSupersample(lower)}
                title={`Render at ${lower}× instead of ${props.supersample}×. Supersampling costs roughly its square, so this is about ${Math.round((props.supersample / lower) ** 2 * 10) / 10}× less work.`}
              >
                try {lower}×
              </button>
            </>
          )}
        </span>
      )}
      <span className="meta">{props.device}</span>
    </header>
  );
}
