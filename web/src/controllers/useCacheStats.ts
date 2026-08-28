/** What the caches are holding, polled while the picture is up.
 *
 *  A poll rather than a push. The numbers change on a render, on an upload and
 *  on the engine's own idle flush -- three producers, one of which is a timer
 *  in the server with no request attached to it -- so there is no single moment
 *  the client could be told about, and a websocket for a readout in the corner
 *  of a bar would be a lot of machinery for two numbers.
 */

import { useEffect, useState } from "react";

import { clearCache, getCacheStats, type CacheStats } from "../services/api";

/** How often to ask, in ms.
 *
 *  The endpoint reads counters the stores already keep -- it walks no
 *  directories and touches no disk -- so the cost is a round trip. Three
 *  seconds is chosen against the *server*, not the client: the idle flush fires
 *  two seconds after the last render, so this is slow enough not to be noise
 *  and fast enough that the drop it produces is visible while you are still
 *  looking at the bar. */
const EVERY_MS = 3000;

export function useCacheStats(opts: {
  /** Something that changes when the caches might have. Passing the render
   *  flag means the readout updates the moment a render finishes rather than up
   *  to `EVERY_MS` later, which is exactly when someone watching this is
   *  watching it. */
  pulse?: unknown;
  /** False while there is no photograph -- nothing to report and no reason to
   *  poll an idle server. */
  active: boolean;
}) {
  const [stats, setStats] = useState<CacheStats | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!opts.active) {
      setStats(null);
      return;
    }
    let alive = true;
    const tick = async () => {
      // Nothing is repainting a hidden tab, so nothing needs the numbers.
      // Skipped rather than unscheduled so the poll resumes on its own when the
      // tab comes back, without a second listener to wire up and tear down.
      if (document.visibilityState === "hidden") return;
      try {
        const s = await getCacheStats();
        if (alive) setStats(s);
      } catch {
        // A readout is not worth an error banner. The endpoint is read-only and
        // the next tick will either succeed or the app has larger problems that
        // something else is already reporting.
      }
    };
    tick();
    const id = window.setInterval(tick, EVERY_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [opts.active, opts.pulse]);

  const clear = async () => {
    setBusy(true);
    try {
      setStats(await clearCache());
    } catch {
      /* see above */
    } finally {
      setBusy(false);
    }
  };

  return { stats, clear, busy };
}
