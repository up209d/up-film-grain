/** Shared shapes the panel, the stage and the request builder all speak.
 *
 *  Parameter *values* are a flat key->number map because that is what the
 *  schema is: `params.py` is the single source of truth, so the client never
 *  names a parameter in a type. */
export type Values = Record<string, number>;

/** How the before and after are shown against each other: stacked under a wipe
 *  (with B to swap outright), or in two panes that pan and zoom together. */
export type Compare = "overlay" | "side";

/** Who a preset is by. Carried beside the values rather than in them for the
 *  same reason the LUT name is -- it is not a quantity -- and kept as one
 *  object rather than two loose strings so it travels as a single credit: a
 *  link with nobody's name on it is not attribution. Null means the look has
 *  no stated author, which is what anything dialled in here from scratch is
 *  until someone says otherwise. */
export type Author = {
  name: string;
  /** Where to find them. Optional even when the name is known. */
  link: string | null;
};

/** One point in the edit history -- everything undo has to put back.
 *
 *  All five together, not just the values: a preset carries a reference size, a
 *  LUT and its author's credit alongside its numbers, and a muted section holds
 *  values that are not in `values` at all. Restoring four of the five would
 *  leave a state the user never had -- undoing a preset back to Stock with the
 *  previous look's LUT still riding under it, say, or with the previous
 *  author's name still attached to what would be written out. */
export type Snapshot = {
  values: Values;
  muted: Record<string, Values>;
  referenceMp: number | null;
  lut: string | null;
  author: Author | null;
};
