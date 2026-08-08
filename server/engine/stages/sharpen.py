"""Output sharpening -- deliberately the last thing before Film Texture.

Extracted from `render()` on 2026-08-08. Its own module rather than folded in
with the edge stages, because it is its own panel section and because
`docs/pipeline-order.md` lists its position as load-bearing: run before the
grain stages it would sharpen a clean image and leave the grain flat.
"""

from __future__ import annotations

import torch

from ..primitives import _blur


class SharpenMixin:
    """The blunt instrument, as against `acutance`'s edge-local effect."""

    def _sharpen(
        self, out: torch.Tensor, p: dict, scale: float,
    ) -> torch.Tensor:
        """Bit-identical to the inline version it replaced."""
        # 14. Output sharpening -- deliberately the last thing in the pipeline.
        #
        #     An unsharp mask amplifies whatever high-frequency content it
        #     finds, and by this point that is the grain as much as the image.
        #     That is the entire reason it sits here rather than earlier: it
        #     cranks the noise already present instead of generating any, so
        #     grain gains bite and the picture gains acutance from the same
        #     operation. Run before the grain stages it would sharpen a clean
        #     image and leave the grain flat, which is the opposite of the
        #     intent.
        #
        #     Distinct from `acutance`, which is an edge-local development
        #     effect extracted from the *pre-grain* base specifically so it
        #     sharpens the image without amplifying grain. This one is the
        #     blunt instrument, and it is applied to the unclamped signal so
        #     overshoot keeps its headroom until the final clamp.
        sh = p["sharpen"]
        if sh > 0.01:
            out = out + (out - _blur(out, max(0.3, p["sharpen_radius"] * scale))) * sh
        return out
