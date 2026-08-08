/** Shared shapes the panel, the stage and the request builder all speak.
 *
 *  Parameter *values* are a flat key->number map because that is what the
 *  schema is: `params.py` is the single source of truth, so the client never
 *  names a parameter in a type. */
export type Values = Record<string, number>;

/** How the before and after are shown against each other: stacked under a wipe
 *  (with B to swap outright), or in two panes that pan and zoom together. */
export type Compare = "overlay" | "side";

/** One point in the edit history -- everything undo has to put back.
 *
 *  All four together, not just the values: a preset carries a reference size
 *  and a LUT alongside its numbers, and a muted section holds values that are
 *  not in `values` at all. Restoring three of the four would leave a state the
 *  user never had -- undoing a preset back to Stock with the previous look's
 *  LUT still riding under it, say. */
export type Snapshot = {
  values: Values;
  muted: Record<string, Values>;
  referenceMp: number | null;
  lut: string | null;
};
