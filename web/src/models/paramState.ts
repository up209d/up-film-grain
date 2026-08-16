/** Pure functions over a schema and a value set.
 *
 *  No React in here on purpose: these are the rules about what a value set
 *  *is* -- where it starts, what "muted" holds, what a file is allowed to
 *  contain -- and they are the part worth reasoning about on their own. The
 *  hooks in `../controllers` own the state; this owns the arithmetic.
 */

import type { Schema } from "../services/api";
import type { Values } from "./types";

/** The starting point: the server's nominated default preset if that file
 *  exists, otherwise the raw parameter defaults. Shared by boot and Reset so
 *  the two cannot drift -- "reset" meaning something different from "how it
 *  opened" is its own small bug. */
export function startingValues(
  s: Schema,
): { values: Values; referenceMp: number | null; lut: string | null } {
  const v: Values = {};
  for (const p of s.params) v[p.key] = p.default;
  const preset = s.presets.find((x) => x.name === s.default_preset);
  // The reference size travels with the values, and so does the LUT name.
  // Returning them here rather than only in applyPreset is the point: boot
  // and Reset go through this path, so without it the app opened on Stock
  // with size scaling inert until you re-picked Stock from the dropdown by
  // hand.
  return preset
    ? {
        values: { ...v, ...preset.values },
        referenceMp: preset.reference_mp,
        lut: preset.lut ?? null,
      }
    : { values: v, referenceMp: null, lut: null };
}

/** Mute every section at once, the way pressing every section's own mute
 *  button would -- each group's *kept* values come from `src` (the starting
 *  preset), while the group's live values are neutral, exactly like
 *  `toggleGroup` does for one section. This is what boot and Reset show: the
 *  photo opens untouched, with the starting preset's whole look sitting
 *  behind the "○" buttons rather than applied. Picking a preset or loading a
 *  file is the only thing that clears this and turns every section on. */
export function muteAll(s: Schema, src: Values): Record<string, Values> {
  const m: Record<string, Values> = {};
  for (const g of s.groups) {
    const keys = s.params.filter((p) => p.group === g).map((p) => p.key);
    const keep: Values = {};
    for (const k of keys) keep[k] = src[k];
    m[g] = keep;
  }
  return m;
}

/** Coerce an arbitrary parsed object into a complete, in-range value set.
 *
 *  This mirrors `sanitize()` on the server: unknown keys are dropped, values
 *  are clamped, and anything missing falls back to its default rather than to
 *  whatever the sliders happen to be showing -- a preset file describes a
 *  whole look, so loading one must not leave stray state from the last one.
 *  It is why a file written before a slider's range changed still loads. */
export function coerceValues(
  s: Schema,
  raw: unknown,
): { values: Values; dropped: string[] } {
  if (!raw || typeof raw !== "object") throw new Error("Not a JSON object.");
  // Accept both our own wrapper and a bare {key: value} map, so the file
  // stays hand-editable.
  const obj = raw as Record<string, unknown>;
  const src = (obj.values ?? obj) as Record<string, unknown>;
  if (!src || typeof src !== "object") throw new Error("No `values` object.");

  const v: Values = {};
  for (const p of s.params) v[p.key] = p.default;
  const dropped: string[] = [];
  let matched = 0;
  for (const [k, val] of Object.entries(src)) {
    const p = s.params.find((x) => x.key === k);
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
}

/** Parameters bundled by group, in schema order, empty groups dropped. */
export function groupedParams(s: Schema | null) {
  if (!s) return [];
  return s.groups
    .map((g) => ({ group: g, params: s.params.filter((p) => p.group === g) }))
    .filter((g) => g.params.length > 0);
}

/** Which group each parameter belongs to. */
export function groupIndex(s: Schema | null): Record<string, string> {
  const m: Record<string, string> = {};
  for (const p of s?.params ?? []) m[p.key] = p.group;
  return m;
}

/** True when nothing is switched on, so a render would return its input.
 *
 *  The *amounts* only, mirroring `is_neutral()` on the server rather than
 *  comparing every key. A seed, a size or a radius can differ from its neutral
 *  value with every stage still switched off -- the picture is the source
 *  either way -- and reading those as a change is what made a photo whose seed
 *  had just been rerolled on open stop counting as untouched. The two sides now
 *  answer this question from the same list. */
export function isNeutral(s: Schema | null, values: Values): boolean {
  return !!s && s.neutral_zero.every((k) => values[k] === s.neutral[k]);
}

/** Switch every stage off, leaving the shapes alone.
 *
 *  What "Original" applies. Only the amounts are zeroed: sizes, radii and seeds
 *  keep whatever they are set to, so turning a section back on returns you to
 *  what you had rather than to the factory numbers -- and so a reroll on open
 *  is not undone by a press of Original. */
export function neutralised(s: Schema, values: Values): Values {
  const v = { ...values };
  for (const k of s.neutral_zero) v[k] = s.neutral[k];
  return v;
}

/** True when any parameter in a group differs from its neutral value. */
export function groupActive(
  s: Schema | null,
  values: Values,
  group: string,
): boolean {
  return (
    !!s &&
    s.params.some(
      (x) => x.group === group && values[x.key] !== s.neutral[x.key],
    )
  );
}
