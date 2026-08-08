/** Boot: the parameter schema, the device name, and the LUT list.
 *
 *  Three independent fetches, deliberately not gated on each other -- a missing
 *  or unreadable `luts/` folder is not an error worth a banner, and the device
 *  readout is cosmetic, so only the schema can fail loudly.
 */

import { useEffect, useState } from "react";

import { getHealth, getLuts, getSchema, type LutInfo, type Schema } from "../services/api";

export function useSchema(onError: (msg: string) => void) {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [device, setDevice] = useState("");
  const [luts, setLuts] = useState<LutInfo[]>([]);
  /** Set once, when the schema lands, so whoever owns the value state can seed
   *  itself from it without this hook needing to know about values at all. */
  const [booted, setBooted] = useState<Schema | null>(null);

  useEffect(() => {
    getSchema()
      .then((s) => {
        setSchema(s);
        setBooted(s);
      })
      .catch((e) => onError(String(e.message ?? e)));
    getHealth()
      .then((h) => setDevice(h.device))
      .catch(() => undefined);
    getLuts()
      .then(setLuts)
      .catch(() => undefined);
    // Boot runs once; `onError` is a setter and stable enough not to re-fire it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { schema, device, setDevice, luts, setLuts, booted };
}
