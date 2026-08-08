/** The LUT selection.
 *
 *  It is its own state rather than a value in `values` because it is a name,
 *  not a number -- see `LUT_ANCHOR_KEY`. It travels with the values everywhere
 *  they go: into a render request, into an export, into a saved preset file,
 *  and back out of one.
 */

import { useRef } from "react";

import { LUT_ANCHOR_KEY } from "../models/constants";
import type { Values } from "../models/types";
import { uploadLut, type LutInfo } from "../services/api";

export function useLuts(opts: {
  luts: LutInfo[];
  setLuts: (f: (ls: LutInfo[]) => LutInfo[]) => void;
  setLut: (id: string | null) => void;
  valuesRef: { current: Values };
  setValueNow: (k: string, v: number) => void;
  liveFor: (k: string) => void;
  onError: (msg: string | null) => void;
  onNotice: (msg: string | null) => void;
}) {
  const fileRef = useRef<HTMLInputElement | null>(null);

  /** Pick a LUT (or clear it), and switch the stage on if it was off.
   *
   *  Bumping Mix from 0 to 1 is the one bit of behaviour here that is not
   *  mechanical, and it is deliberate: Mix ships at 0 like every other stage in
   *  the app, so a freshly-picked LUT would otherwise change nothing at all and
   *  read as broken. A Mix the user has already moved is left exactly where they
   *  put it — this only fires on a stage that is genuinely off. Clearing the LUT
   *  leaves Mix alone, so re-picking one returns you to the strength you had.
   *
   *  Both setters, one gesture: `setValue` then `commit()` would apply the
   *  *previous* value here, for the reason `setValueNow` documents — a menu has
   *  no pointerup to clean up after it. */
  const pickLut = (id: string | null) => {
    opts.setLut(id);
    if (id && (opts.valuesRef.current[LUT_ANCHOR_KEY] ?? 0) <= 0) {
      // Goes through setValueNow, so it also switches the section on. On a
      // fresh load every section is muted, and a LUT that renders while its own
      // section reads "off" is the incoherent state that pair exists to prevent.
      opts.setValueNow(LUT_ANCHOR_KEY, 1);
    } else if (id) {
      opts.liveFor(LUT_ANCHOR_KEY);
    }
    // Clearing it needs neither: `lut` is part of the render request, so
    // changing it re-fires the render effect on its own, and "no LUT" is not a
    // reason to switch a muted section back on.
  };

  const onLutFile = async (file: File) => {
    try {
      const info = await uploadLut(file);
      opts.setLuts((ls) => [...ls.filter((x) => x.id !== info.id), info]);
      pickLut(info.id);
      opts.onError(null);
      opts.onNotice(`Loaded LUT ${info.name} (${info.size}³)`);
    } catch (e: any) {
      opts.onNotice(null);
      opts.onError(`Could not load ${file.name}: ${e.message ?? e}`);
    }
  };

  return { fileRef, pickLut, onLutFile };
}
