"""the 16-bit PNG writer, and upscale

Split out of the original single-function `verify.py` on 2026-08-08. The body is
the same code, in the same order -- see `docs/testing.md`.
"""

from __future__ import annotations

import numpy as np
import struct
import torch
import zlib
from server import imageio as iio
from tests.harness import Ctx, check, suite


@suite("imageio", "the 16-bit PNG writer, and upscale")
def run(cx: Ctx) -> None:
    eng, p, img = cx.eng, cx.p, cx.img
    print("\n16-bit PNG writer (Pillow cannot write these; we emit them by hand)")
    small = np.random.default_rng(0).random((37, 53, 3)).astype(np.float32)
    blob = iio.encode(small, "png16")
    pos, chunks = 8, {}
    ok_crc = True
    while pos < len(blob):
        ln = struct.unpack(">I", blob[pos:pos + 4])[0]
        tag = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + ln]
        crc = struct.unpack(">I", blob[pos + 8 + ln:pos + 12 + ln])[0]
        ok_crc &= crc == zlib.crc32(tag + data) & 0xFFFFFFFF
        chunks[tag] = chunks.get(tag, b"") + data
        pos += 12 + ln
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    raw = zlib.decompress(chunks[b"IDAT"])
    px = np.frombuffer(raw, np.uint8).reshape(h, w * 6 + 1)[:, 1:].reshape(h, w, 3, 2).astype(np.uint16)
    back = ((px[..., 0] << 8) | px[..., 1]).astype(np.float32) / 65535.0
    d = float(np.abs(back - small).max())
    check("chunk CRCs", ok_crc, "all valid")
    check("bit depth", depth == 16 and ctype == 2, f"depth={depth} colourtype={ctype}")
    check("precision", d < 5e-5, f"roundtrip {d:.2e} (8-bit floor would be 2e-3)")

    print("\nupscale (blow a render up to the source's own dimensions)")
    # A small gradient plate rather than noise -- noise has no structure for a
    # round trip to preserve, and a gradient can show an axis swap or an
    # off-by-one a shape-only check would miss.
    gy2, gx2 = np.mgrid[0:60, 0:90].astype(np.float32)
    small_plate = np.stack(
        [gx2 / 89.0, gy2 / 59.0, (gx2 + gy2) / 148.0], -1
    ).astype(np.float32)
    small_plate = np.ascontiguousarray(small_plate)

    up_same = iio.upscale(small_plate, 60, 90)
    check(
        "a no-op at the target size returns the same array",
        up_same is small_plate, "identity, not merely equal",
    )

    big = iio.upscale(small_plate, 240, 360)
    check(
        "upscale hits the exact requested size",
        big.shape == (240, 360, 3), f"got {big.shape}",
    )
    check(
        "upscale stays inside 0..1",
        float(big.min()) >= 0.0 and float(big.max()) <= 1.0,
        f"range {float(big.min()):.3f}..{float(big.max()):.3f} "
        "(bicubic can ring past the source's own range without the clamp)",
    )

    # Matches a direct call with the same arguments -- pins the choice of
    # bicubic/no-antialias/align_corners=False against a silent drift in any
    # one of them, since a symmetric gradient could pass a looser check with
    # any of the three wrong.
    import torch.nn.functional as _F  # noqa: E402
    t = torch.from_numpy(small_plate).permute(2, 0, 1).unsqueeze(0)
    ref = _F.interpolate(t, size=(240, 360), mode="bicubic", align_corners=False)
    ref = ref.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    d = float(np.abs(big - ref).max())
    check(
        "upscale matches a direct bicubic call", d < 1e-6,
        f"max delta {d:.2e}",
    )

    # A downscale/upscale round trip cannot recover detail, but on a smooth
    # gradient with no fine structure it should land close to the original --
    # a coarse sanity check that nothing is transposed, flipped or scaled
    # wrong, not a claim about image quality.
    down = iio.downscale(small_plate, 0.5)
    back_up = iio.upscale(down, 60, 90)
    d = float(np.abs(back_up - small_plate).max())
    check(
        "a downscale/upscale round trip approximates a smooth plate",
        d < 0.05, f"max delta {d:.2e}",
    )
