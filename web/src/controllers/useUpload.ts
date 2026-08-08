/** Taking a photo in: the upload itself, and the drop-target highlight.
 *
 *  A drop is invisible otherwise -- there is nothing to tell you the button
 *  will take the file.
 */

import { useState } from "react";

import { uploadImage, type ImageMeta } from "../services/api";

export function useUpload(opts: {
  onAccepted: () => void;
  onMeta: (m: ImageMeta) => void;
  onError: (msg: string | null) => void;
}) {
  const [dropping, setDropping] = useState(false);

  const onFile = async (file: File) => {
    opts.onError(null);
    try {
      const m = await uploadImage(file);
      // Drop the outgoing photo's images *before* swapping meta in. They are
      // not merely stale: the stage sizes every layer from `meta`, so the old
      // photo would be stretched to the new one's dimensions until the first
      // render lands -- and if that render failed you were left looking at the
      // previous photo with nothing saying so.
      //
      // Only on success. A rejected file (wrong format, over the size cap)
      // must leave the session exactly as it was rather than clearing the
      // stage out from under you.
      opts.onAccepted();
      opts.onMeta(m);
    } catch (e: any) {
      opts.onError(String(e.message ?? e));
    }
  };

  return { dropping, setDropping, onFile };
}
