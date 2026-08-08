/** Hold B to peek at the original -- the fastest way to judge an adjustment. */

import { useEffect, useState } from "react";

export function useBeforePeek() {
  const [showBefore, setShowBefore] = useState(false);

  useEffect(() => {
    const isTyping = (t: EventTarget | null) =>
      t instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName);
    const down = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "b" && !e.repeat && !isTyping(e.target)) {
        setShowBefore(true);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "b") setShowBefore(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  return { showBefore, setShowBefore };
}
