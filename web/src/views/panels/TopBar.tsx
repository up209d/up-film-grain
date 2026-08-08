/** The app bar: open a photo, the seed-on-open switch, and the render readout. */

import filmGrain1x1 from "../../assets/film-grain-1x1.jpg";
import type { ImageMeta } from "../../services/api";

export default function TopBar(props: {
  meta: ImageMeta | null;
  device: string;
  rendering: boolean;
  renderMs: number;
  dropping: boolean;
  onDropping: (v: boolean) => void;
  onFile: (f: File) => void;
  randomizeSeedOnOpen: boolean;
  onRandomizeSeedOnOpen: (v: boolean) => void;
  onRandomizeSeeds: () => void;
}) {
  const { meta } = props;
  return (
    <header className="bar">
      <div className="brand">
        <img src={filmGrain1x1} alt="Film grain" className="film-grain-icon" />
        Film Grain Engine
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
          "grain and damage pattern. Off keeps whatever seeds are " +
          "currently dialled in."
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
      <span className={`status ${props.rendering ? "busy" : ""}`}>
        {props.rendering && <div className="spinner" />}
        {props.rendering ? "rendering…" : `${props.renderMs}ms`}
      </span>
      <span className="meta">{props.device}</span>
    </header>
  );
}
