/** Reading and writing preset files.
 *
 *  The file format is deliberately forgiving -- `coerceValues` accepts a bare
 *  `{"intensity": 40}` as readily as our own wrapper -- so what is written here
 *  stays hand-editable.
 */

import { useRef } from "react";

import { PRESET_FORMAT } from "../models/constants";
import { coerceValues } from "../models/paramState";
import type { Values } from "../models/types";
import type { ImageMeta, Schema } from "../services/api";

export function usePresetFile(opts: {
  schema: Schema | null;
  values: Values;
  meta: ImageMeta | null;
  referenceMp: number | null;
  lut: string | null;
  setReferenceMp: (v: number | null) => void;
  setLut: (v: string | null) => void;
  applyValues: (v: Values) => void;
  onError: (msg: string | null) => void;
  onNotice: (msg: string | null) => void;
}) {
  const fileRef = useRef<HTMLInputElement | null>(null);

  const savePreset = () => {
    const { schema, values, meta, referenceMp, lut } = opts;
    if (!schema) return;
    const name = window.prompt("Preset name", "my-look")?.trim();
    if (!name) return;
    const file = {
      format: PRESET_FORMAT,
      version: 1,
      name,
      // Stamped so the preset can be rescaled onto a different-sized photo.
      // Falls back to whatever it was loaded with, so re-saving a preset you
      // did not author here does not silently re-base it onto this image.
      reference_mp: referenceMp ?? meta?.megapixels ?? null,
      // A sibling of `values`, not one of them -- a LUT is named, not numbered.
      // An uploaded LUT's id will not resolve in a future session; that is the
      // honest thing to write, and the picker shows the name as missing rather
      // than the app pretending the grade is still there.
      lut,
      values: Object.fromEntries(
        // Written in schema order, not insertion order, so a hand-edited file
        // stays readable and two saves diff cleanly.
        schema.params.map((p) => [p.key, values[p.key] ?? p.default]),
      ),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(file, null, 2)], { type: "application/json" }),
    );
    const filename = `${name.replace(/[^\w.-]+/g, "-")}.json`;
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    // In the document and revoked on a later tick: Safari cancels the download
    // if the blob URL is revoked in the same task as the click, and Firefox
    // wants the anchor to actually be in the tree.
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    opts.onError(null);
    // A browser download is silent, so say something -- otherwise the button
    // looks like it did nothing.
    opts.onNotice(`Saved ${filename}`);
  };

  const loadPreset = async (file: File) => {
    try {
      if (!opts.schema) throw new Error("Schema not loaded yet.");
      const raw = JSON.parse(await file.text());
      const { values: v, dropped } = coerceValues(opts.schema, raw);
      if (typeof raw?.reference_mp === "number") opts.setReferenceMp(raw.reference_mp);
      // Set unconditionally, `null` included: a file with no LUT describes a
      // look that has none, and leaving the previous selection in place would
      // silently blend two grades.
      opts.setLut(typeof raw?.lut === "string" && raw.lut ? raw.lut : null);
      opts.applyValues(v);
      opts.onError(null);
      opts.onNotice(
        dropped.length
          ? `Loaded ${file.name} — ignored unknown key${
              dropped.length > 1 ? "s" : ""
            }: ${dropped.join(", ")}`
          : `Loaded ${file.name}`,
      );
    } catch (e: any) {
      opts.onNotice(null);
      opts.onError(`Could not load ${file.name}: ${e.message ?? e}`);
    }
  };

  return { fileRef, savePreset, loadPreset };
}
