/** Undo and redo over the value state.
 *
 *  Observed rather than driven. Every way of changing a value already exists in
 *  `useValues` -- a slider, a menu, a preset, a mute toggle, a LUT pick, a
 *  loaded file -- and a history that each of them had to remember to call would
 *  be one `push()` away from a hole in it at all times. So nothing calls in
 *  here: this watches the committed state and records the transitions it sees,
 *  which means a new way of editing is undoable the day it is written.
 *
 *  What it watches is the *committed* state, not the live one. `applied` is the
 *  value set the renderer has been handed, and it only moves when a gesture is
 *  finished -- so a slider drag is one step rather than one per pixel of
 *  travel, for free. `muted` deliberately does not trigger a step: mid-drag,
 *  touching a control in a muted section un-mutes it a render *before* the
 *  release commits, and treating that as its own transition would file a state
 *  that was never shown (section live, values not yet applied). It is still
 *  read at push time, so the step that does get filed carries the right one.
 *
 *  The known coarseness, stated rather than hidden: a typed number and an
 *  arrow-key nudge each commit per keystroke, so each is its own step. That is
 *  the same granularity the renderer sees, and collapsing it would mean
 *  guessing at where one edit ends.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { Snapshot } from "../models/types";

/** Steps kept. Snapshots are a few dozen numbers each, so the cap is about not
 *  growing without bound over a long session rather than about memory. */
export const HISTORY_LIMIT = 200;

/** Two snapshots are the same step when every field is the same *object*.
 *  Identity, not deep equality, on purpose: every path in `useValues` builds a
 *  fresh object for a real change, so a shared reference means nothing
 *  happened -- and it makes this cheap enough to run on every commit. */
function same(a: Snapshot, b: Snapshot): boolean {
  return (
    a.values === b.values &&
    a.muted === b.muted &&
    a.referenceMp === b.referenceMp &&
    a.lut === b.lut
  );
}

/** @param resetKey  Identity of the photo being edited. When it changes the
 *  history is cleared and the state at that moment becomes the new baseline:
 *  a step from the last photo is not something you can go back to, because the
 *  render it described is gone. Handled inside the same effect that records
 *  steps rather than as a `reset()` anyone can call, so that the open -- which
 *  swaps the photo *and* rerolls the seeds in one go -- cannot leave the reroll
 *  behind as the one undoable step on a freshly opened image. */
export function useHistory(
  current: Snapshot,
  restore: (s: Snapshot) => void,
  resetKey: string | null,
) {
  // The stacks are refs rather than state because `undo` has to read *and*
  // write them in one go and then call `restore`; a state updater that did
  // that would run its side effect twice under StrictMode. The counter is
  // what re-renders the buttons when the stacks change.
  const past = useRef<Snapshot[]>([]);
  const future = useRef<Snapshot[]>([]);
  const [, bump] = useState(0);
  const rerender = () => bump((n) => n + 1);

  // The last state we filed -- what the *next* change will push. Not derivable
  // from the stacks: it is the present, which by definition is on neither.
  const last = useRef<Snapshot | null>(null);
  // Set while our own restore is landing, so it does not read as a new edit.
  const restoring = useRef(false);
  // The photo the stacks belong to. `undefined` rather than null to start, so
  // the first run is a mismatch and takes the re-baseline path.
  const keyed = useRef<string | null | undefined>(undefined);

  useEffect(() => {
    // Nothing to remember until the schema has landed and seeded the values.
    // Without this the empty initial value set becomes the first step, and
    // undo would take you to a photo with no parameters at all.
    if (!Object.keys(current.values).length) return;

    const prev = last.current;
    last.current = current;

    // A different photo: start over from wherever the values have landed.
    // Checked before the push below, not beside it, because opening a photo
    // changes the key and the seeds in one commit and only one of those is a
    // step -- neither, as it turns out.
    if (keyed.current !== resetKey) {
      keyed.current = resetKey;
      restoring.current = false;
      if (past.current.length || future.current.length) {
        past.current = [];
        future.current = [];
        rerender();
      }
      return;
    }

    if (restoring.current) {
      restoring.current = false;
      return;
    }
    // The first state seen is the baseline -- how the app opened. There is
    // nothing before it to go back to.
    if (!prev || same(prev, current)) return;

    past.current = [...past.current, prev].slice(-HISTORY_LIMIT);
    future.current = [];
    rerender();
    // `muted` is read above but deliberately not a dependency -- see the file
    // comment. The other three are what defines a committed change.
  }, [current.values, current.referenceMp, current.lut, resetKey]);

  const undo = useCallback(() => {
    const prev = past.current[past.current.length - 1];
    if (!prev) return;
    past.current = past.current.slice(0, -1);
    if (last.current) future.current = [last.current, ...future.current];
    restoring.current = true;
    last.current = prev;
    restore(prev);
    rerender();
  }, [restore]);

  const redo = useCallback(() => {
    const next = future.current[0];
    if (!next) return;
    future.current = future.current.slice(1);
    if (last.current) past.current = [...past.current, last.current];
    restoring.current = true;
    last.current = next;
    restore(next);
    rerender();
  }, [restore]);

  return {
    undo,
    redo,
    canUndo: past.current.length > 0,
    canRedo: future.current.length > 0,
  };
}
