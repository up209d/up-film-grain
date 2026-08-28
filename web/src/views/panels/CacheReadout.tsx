/** How much the app is holding, and where — bottom right, beside Export.
 *
 *  It exists because "the app eats RAM" was a question nobody in the session
 *  could answer. The caches moved to the SSD (`server/engine/diskcache.py`) and
 *  the memory came back, but a fix you cannot see is indistinguishable from no
 *  fix: the point of this readout is that the next time the app feels heavy,
 *  the answer is on screen rather than in Activity Monitor.
 *
 *  **Two numbers, in this order, and the order is the argument.** `Disk` is
 *  what the caches are holding; `RAM` is what the process is resident in. They
 *  are shown together because either alone is misleading — a small disk figure
 *  with a large RAM one means the caches are not the problem, and a large disk
 *  figure with a small RAM one is the system working exactly as intended, which
 *  is a thing a user should be able to *see* rather than take on trust.
 *
 *  In the export bar rather than a panel section because it is not a parameter:
 *  it changes nothing about the render, and this is the overlay that already
 *  carries the other controls that do not (see `ExportPanel`).
 */

import Help from "../controls/Help";
import type { CacheStats } from "../../services/api";

/** Bytes as a short human string. Binary units, since these are memory and
 *  file sizes and every tool the user would check this against — Activity
 *  Monitor, Finder, `du` — reports the same way. */
function human(n: number | undefined): string {
  if (n === undefined || n < 0) return "--";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

/** The part names the server sends, in the order they are worth reading and
 *  spelled the way a user would say them. A name not listed still shows, under
 *  its own key — the server is free to add a store without this file changing,
 *  which is the same rule the parameter panel follows. */
const LABELS: Record<string, string> = {
  checkpoints: "Pipeline checkpoints",
  "grain-textures": "Grain textures",
  frames: "Source frames",
  exports: "Finished exports",
};

export default function CacheReadout(props: {
  stats: CacheStats | null;
  onClear: () => void;
  busy: boolean;
}) {
  const { stats } = props;
  if (!stats) return null;

  const lines = stats.parts
    .map((p) => {
      const label = LABELS[p.name] ?? p.name;
      const rate =
        p.hits === undefined || p.hits + (p.misses ?? 0) === 0
          ? ""
          : `, ${Math.round((100 * p.hits) / (p.hits + (p.misses ?? 0)))}% hit`;
      return `${label}: ${human(p.bytes)} in ${p.entries} ${
        p.entries === 1 ? "entry" : "entries"
      }${rate}`;
    })
    .join("\n");

  const help = stats.enabled
    ? "Everything the app caches lives on the SSD rather than in memory — " +
      "the pipeline's intermediate frames, the grain textures, the decoded " +
      "photograph and any finished export. Memory is handed back a couple of " +
      "seconds after rendering stops, so the RAM figure falls while you look " +
      "at the picture and rises while you drag.\n\n" +
      lines +
      `\n\nWritten this session ${human(stats.written)} · budget ` +
      `${human(stats.budget)} · peak RAM ${human(stats.memory.peak)}\n\n` +
      "Clear drops only what can be rebuilt; your photograph and any finished " +
      "export are kept. Opening a new photograph clears the old one's caches " +
      "on its own."
    : "There is no writable cache directory, so nothing is cached — renders " +
      "are correct but repeat every step each time. Point " +
      "FILM_GRAIN_CACHE_DIR at a writable folder to turn caching back on.";

  return (
    <>
      {/* No leading separator: `ExportPanel` already puts one before whatever
          it is handed, and `App` puts one after this. */}
      <Help text={help} label="Cache" />
      <span className="cachefig" title={stats.root ?? undefined}>
        {stats.enabled ? (
          <>
            <b>{human(stats.bytes)}</b> disk
            <span className="cachedot">·</span>
            <b>{human(stats.memory.rss)}</b> RAM
          </>
        ) : (
          <>off</>
        )}
      </span>
      <button
        className="seg"
        onClick={props.onClear}
        disabled={props.busy || !stats.enabled || stats.bytes === 0}
        title="Drop every cache that can be rebuilt. The photograph and any finished export are kept."
      >
        Clear
      </button>
    </>
  );
}
