/** The value state and every way of changing it.
 *
 *  Two value sets, not one: `values` is what the panel shows and updates on
 *  every input event, `applied` is what the renderer sees and only moves when a
 *  gesture is committed. A fit preview is seconds of work, so rendering
 *  mid-drag only queues frames that are already stale by the time they arrive.
 *
 *  `muted` is the third piece and the one with the subtle rules: a muted
 *  section renders neutral while holding what it *would* be, so switching it
 *  back on returns you to what you had rather than to the factory numbers.
 *
 *  Undo/redo hangs off the bottom of this file and none of the mutators know
 *  about it -- `useHistory` watches `applied` and records what it sees, so an
 *  edit added here is undoable without being told to be. See that file.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  groupIndex, muteAll, neutralised, startingValues,
} from "../models/paramState";
import type { Snapshot, Values } from "../models/types";
import type { Schema } from "../services/api";
import { useHistory } from "./useHistory";

/** The seeds a photo gets its own draw of, with the section each belongs to.
 *  `global_seed` is deliberately not one of them: it is an *offset* on `seed`,
 *  so rerolling `seed` already reshuffles the global layer, and drawing both
 *  would throw away the one number a preset uses to say where it wants that
 *  layer relative to the rest. */
const SEED_KEYS: ReadonlyArray<readonly [string, string]> = [
  ["seed", "Grain Structure"],
  ["texture_seed", "Film Texture"],
];

export function useValues(
  schema: Schema | null,
  booted: Schema | null,
  imageId: string | null,
  /** Whether "With random seed" is on. Read only by `applyPreset`, which is
   *  the one action that would otherwise overwrite a freshly drawn seed with
   *  the fixed one baked into the preset file. */
  randomSeeds: boolean,
) {
  const [values, setValues] = useState<Values>({});
  const [applied, setApplied] = useState<Values>({});
  const [muted, setMuted] = useState<Record<string, Values>>({});
  const [referenceMp, setReferenceMp] = useState<number | null>(null);
  const [lut, setLut] = useState<string | null>(null);

  const valuesRef = useRef<Values>(values);
  valuesRef.current = values;

  // Seed from the schema the moment it lands. The starting preset's values are
  // held as "muted" rather than applied -- the app opens showing the untouched
  // photo, with every section's Stock look one click away on its own toggle.
  useEffect(() => {
    if (!booted) return;
    const start = startingValues(booted);
    setValues(booted.neutral);
    setApplied(booted.neutral);
    setReferenceMp(start.referenceMp);
    setLut(start.lut);
    setMuted(muteAll(booted, start.values));
  }, [booted]);

  const groupOf = groupIndex(schema);

  /** Touching a control in a muted section switches that section back on.
   *
   *  What gets applied is not just the edit: it is the section's *kept* values
   *  restored — exactly what clicking its own ● would do — with the edit laid on
   *  top. Without that, un-muting by editing would quietly discard whatever the
   *  rest of the section had been holding.
   *
   *  Split into a pure half and a side-effecting half on purpose: a `setMuted`
   *  call inside a `setValues` updater would run twice under StrictMode. */
  const keptFor = (k: string): Values | null => {
    const g = groupOf[k];
    return g && muted[g] ? muted[g] : null;
  };

  const liveFor = (k: string) => {
    const g = groupOf[k];
    if (!g || !muted[g]) return;
    setMuted((m) => {
      const n = { ...m };
      delete n[g];
      return n;
    });
  };

  const setValue = (k: string, v: number) => {
    const keep = keptFor(k);
    setValues((s) => ({ ...s, ...keep, [k]: v }));
    liveFor(k);
  };

  /** Hand the live values to the renderer. Passing the ref's current object
   *  means an uncommitted gesture is a no-op: React bails out when the state
   *  is set to the identical reference, so this costs nothing when nothing
   *  moved. */
  const commit = useCallback(() => setApplied(valuesRef.current), []);

  /** Set a value *and* render it, in one gesture.
   *
   *  `setValue` followed by `commit()` does not work and looks like it should:
   *  `commit` reads `valuesRef`, which is only refreshed during render, so
   *  called synchronously it applies the value from *before* the change.
   *  Sliders never noticed because their `pointerup` arrives a render later
   *  and commits the right thing; a menu has no second event, so a selection
   *  did nothing until the control lost focus. Building the next object here
   *  and handing it to both setters keeps them in step. */
  const setValueNow = (k: string, v: number) => {
    const next = { ...valuesRef.current, ...keptFor(k), [k]: v };
    setValues(next);
    setApplied(next);
    liveFor(k);
  };

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

  /** Reroll the grain and texture seeds for a freshly opened photo.
   *
   *  Deliberately *not* two `setValueNow` calls: each one independently reads
   *  `valuesRef.current` and hands a whole replacement object to `setValues`,
   *  so the second call's object is built without the first call's edit in it
   *  and silently drops it -- the same stale-snapshot trap `setValueNow`'s own
   *  comment documents for `commit()`. Both seeds are folded into one object
   *  here instead.
   *
   *  And deliberately *not* `setValueNow` at all: that un-mutes on the
   *  reasoning that a real edit means "I want this section live now," which
   *  does not hold for a reroll nobody asked for by name -- opening a photo
   *  silently switching a muted section back on would be a far bigger surprise
   *  than a repeated seed.
   *
   *  **The new seed goes into the live values *and* into a muted section's kept
   *  snapshot** (fixed 2026-08-16). It used to go into one or the other, on the
   *  reasoning that a muted section's live values are neutral and should stay
   *  that way -- but a seed is not an amount. It cannot switch a stage on, and
   *  while the section is muted it changes nothing, so there was never anything
   *  to protect; what the split actually did was hide the draw. Every session
   *  boots with *every* section muted, so on the first photo opened the reroll
   *  went nowhere the panel or the renderer could see it, and then `applyPreset`
   *  -- which drops the snapshots wholesale -- threw it away. Measured on the
   *  shipped build: open a photo, pick a look, and the render went out with
   *  `seed 1234` every single time, which is the whole feature not working.
   *  Writing both keeps them in step, so `toggleGroup` restoring the snapshot
   *  later restores the same number the panel has been showing. */
  const randomizeSeeds = () => {
    const rolls = SEED_KEYS.map(
      ([key, group]) =>
        [key, group, Math.floor(Math.random() * 10000)] as const,
    );

    const liveNext: Values = {};
    let mutedNext: Record<string, Values> | null = null;
    for (const [key, group, val] of rolls) {
      liveNext[key] = val;
      if (muted[group]) {
        const base: Record<string, Values> = mutedNext ?? muted;
        mutedNext = { ...base, [group]: { ...base[group], [key]: val } };
      }
    }

    if (mutedNext) setMuted(mutedNext);
    const next = { ...valuesRef.current, ...liveNext };
    setValues(next);
    setApplied(next);
  };

  // Presets and reset are single discrete actions, not gestures, so they go
  // straight through to the renderer.
  //
  // Picking a preset is the one thing that turns every section on: it is a
  // deliberate "use this whole look", unlike boot or Reset which stage the
  // preset's values behind each section's mute button instead.
  const applyPreset = (name: string) => {
    const p = schema?.presets.find((x) => x.name === name);
    if (!p) return;
    // A preset dialled in on a 24MP frame means something different on a 45MP
    // one; the server rescales lengths by the linear ratio, but only if it is
    // told what size the values were authored at.
    setReferenceMp(p.reference_mp ?? null);
    // The LUT is part of the look, so it comes along -- including its absence.
    // A preset with no LUT has to *clear* one that is selected, or the last
    // look's grade would keep riding under the new one.
    setLut(p.lut ?? null);
    const v = { ...values, ...p.values };
    // The look is the preset's; *where the grain and the damage fall* is this
    // photo's. Every shipped preset carries a fixed `seed` and `texture_seed`
    // -- whatever number happened to be dialled in when it was saved -- so
    // without this a preset re-pins them and "With random seed" quietly stops
    // meaning anything the moment you choose a look, which is exactly how the
    // feature was reported broken. Held back only while the switch is on:
    // turned off, a preset still reproduces its own grain exactly.
    if (randomSeeds) for (const [k] of SEED_KEYS) v[k] = values[k];
    setValues(v);
    setApplied(v);
    setMuted({});
  };

  // "How it opened" has to mean what boot shows, muted sections included --
  // otherwise Reset and a fresh load would disagree about the starting point.
  const resetAll = () => {
    if (!schema) return;
    const start = startingValues(schema);
    setValues(schema.neutral);
    setApplied(schema.neutral);
    setReferenceMp(start.referenceMp);
    setLut(start.lut);
    setMuted(muteAll(schema, start.values));
  };

  /** Switch the whole pipeline off, so the preview is the untouched photo.
   *  Sizes, radii and seeds are left alone -- they are not what makes a stage
   *  run, and keeping them means turning a section back on returns you to what
   *  you had rather than to the factory numbers.
   *
   *  That is what this has always said and it is only true since 2026-08-16:
   *  handing over `schema.neutral` wholesale put every shape back to its
   *  *default* as well, so Original was a partial Reset wearing the wrong name
   *  -- and once a seed is drawn per photo, a press of it silently undid the
   *  draw. Only the amounts are zeroed now, which is the same set the server
   *  reads to decide a render is a pass-through, so the picture is the source
   *  either way. */
  const showOriginal = () => {
    if (!schema) return;
    const v = neutralised(schema, values);
    setValues(v);
    setApplied(v);
  };

  /** Switch one section off, same idea. Reaching for this is usually "is this
   *  section even earning its keep" -- so it toggles: press it again and the
   *  section comes back exactly as it was. */
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
    const start = startingValues(schema).values;
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

  /** Replace the whole look at once -- what loading a preset file does. */
  const applyValues = (v: Values) => {
    setValues(v);
    setApplied(v); // discrete action -- render straight away
    // A loaded file is a whole look too, same as picking one from the menu --
    // every section goes live rather than staying behind its mute button.
    setMuted({});
  };

  /** Put a whole recorded state back. `values` and `applied` are set to the
   *  same object because a history step is a *committed* state by
   *  construction, so there is no uncommitted gesture to preserve -- and
   *  handing both the identical reference is what makes the window's
   *  `pointerup` commit a genuine no-op on the click that undid it. */
  const restore = useCallback((s: Snapshot) => {
    setValues(s.values);
    setApplied(s.values);
    setMuted(s.muted);
    setReferenceMp(s.referenceMp);
    setLut(s.lut);
  }, []);

  // Placed after every mutator on purpose: it observes rather than being
  // called, so nothing above it has to know the history exists.
  const history = useHistory(
    { values: applied, muted, referenceMp, lut },
    restore,
    imageId,
  );

  return {
    values, applied, muted, referenceMp, lut, valuesRef,
    setReferenceMp, setLut, setValue, setValueNow, commit,
    liveFor, randomizeSeeds, applyPreset, resetAll, showOriginal,
    toggleGroup, resetGroup, applyValues,
    undo: history.undo, redo: history.redo,
    canUndo: history.canUndo, canRedo: history.canRedo,
  };
}
