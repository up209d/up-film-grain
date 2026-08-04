"""Image decode/encode. Keeps everything float32 in 0..1 internally."""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

# Pillow refuses very large images by default as a decompression-bomb guard.
# Raise it: 45MP+ stills are the target, not an attack.
Image.MAX_IMAGE_PIXELS = 400_000_000

# Input is deliberately narrow for now. Since the preview became a
# full-resolution render (the client does all the scaling), every parameter
# change costs an export-grade pass over the whole image -- so the input bound
# is what keeps that affordable, in place of the proxy it replaced.
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
# MPO is a JPEG carrying more than one frame -- cameras emit it for burst and
# 3D captures, and the file still has a .jpg extension and a normal JPEG as its
# first frame. Rejecting it turned an ordinary photo away, so it is accepted and
# read as the JPEG it is.
INPUT_FORMATS = ("JPEG", "MPO", "PNG")

# What the rejection message names. MPO is deliberately absent: nobody thinks of
# their file as "an MPO", and listing it would suggest it needs converting.
INPUT_FORMAT_NAMES = ("JPEG", "PNG")
MAX_PIXELS = 120_000_000


# Both subclass ValueError so any caller that only knows about ValueError still
# behaves; the distinction exists so the API can answer 413 vs 415 rather than
# sniffing the message text.
class UploadTooLarge(ValueError):
    """Input is over the byte or pixel limit."""


class UnsupportedFormat(ValueError):
    """Input decoded, but is not a format we accept."""


# label -> (mime, extension, bit depth)
FORMATS = {
    "png16": ("image/png", "png", 16),
    "png8": ("image/png", "png", 8),
    "jpeg": ("image/jpeg", "jpg", 8),
}


def load_image(data: bytes) -> np.ndarray:
    """Decode to HxWx3 float32 in 0..1, honouring EXIF orientation.

    JPEG (including multi-frame MPO) and PNG only, size-capped. Both limits are
    checked here rather than at the endpoint so there is one place that decides
    what the engine accepts.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadTooLarge(
            f"File is {len(data) / 1024 / 1024:.1f}MB, over the "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024}MB limit."
        )

    im = Image.open(io.BytesIO(data))
    if im.format not in INPUT_FORMATS:
        raise UnsupportedFormat(
            f"{im.format or 'That file'} is not supported -- "
            f"{' and '.join(INPUT_FORMAT_NAMES)} only for now."
        )
    # Multi-frame inputs (MPO) open on frame 0 already, but say so explicitly:
    # the primary image is the photograph, the rest are the second eye of a
    # stereo pair or a burst sibling.
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    im = ImageOps.exif_transpose(im) or im

    if im.width * im.height > MAX_PIXELS:
        raise UploadTooLarge(
            f"Image is {im.width * im.height / 1e6:.0f}MP, over the "
            f"{MAX_PIXELS / 1e6:.0f}MP limit."
        )

    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.array(im).astype(np.float32) / 65535.0
        return np.ascontiguousarray(np.clip(np.stack([arr] * 3, -1), 0.0, 1.0))

    if im.mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGB")

    arr = np.array(im)
    if arr.dtype == np.uint16:
        f = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        f = arr.astype(np.float32) / 255.0
    else:
        f = arr.astype(np.float32)
        if f.max() > 1.5:
            f /= 255.0

    if f.ndim == 2:
        f = np.stack([f] * 3, axis=-1)
    elif f.shape[2] == 4:
        f = f[:, :, :3]  # alpha dropped; grain has no meaning on it
    elif f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)

    return np.ascontiguousarray(np.clip(f, 0.0, 1.0))


def _png16_rgb(u: np.ndarray) -> bytes:
    """Write a 16-bit RGB PNG.

    Pillow cannot do this -- it only writes 16-bit as single-band I;16 -- and
    8-bit is not enough for an export: grain is a low-amplitude high-frequency
    signal, and 8-bit quantisation visibly posterises it in smooth areas. The
    format is simple enough to emit directly.
    """
    h, w, _ = u.shape
    # PNG wants big-endian samples, each scanline prefixed with a filter byte.
    rows = np.ascontiguousarray(u.astype(">u2")).view(np.uint8).reshape(h, w * 6)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16bpc, colour type 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def encode(arr: np.ndarray, fmt: str = "png16", quality: int = 95) -> bytes:
    a = np.clip(arr, 0.0, 1.0)
    if fmt == "png16":
        return _png16_rgb((a * 65535.0 + 0.5).astype(np.uint16))

    u = (a * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    if fmt == "jpeg":
        # 4:4:4 -- chroma subsampling would smear the chroma grain away.
        Image.fromarray(u, "RGB").save(
            buf, format="JPEG", quality=int(quality), subsampling=0
        )
    else:
        Image.fromarray(u, "RGB").save(buf, format="PNG")
    return buf.getvalue()


PREVIEW_QUALITY = 95
PREVIEW_MEDIA_TYPE = "image/jpeg"


def encode_preview(arr: np.ndarray) -> bytes:
    """8-bit JPEG for on-screen preview.

    **4:4:4, and that is not optional** -- the default 4:2:0 would average away
    exactly the chroma grain the pipeline exists to produce.

    Was a `compress_level=1` PNG, which is a poor fit for this content: grain
    defeats PNG's predictor, so the "lightly compressed for latency" version came
    out both enormous *and* slow. Measured on a real photograph rendered through
    the `Stock` preset -- which matters, because a synthetic noise plate is a
    much harder case than real output and flatters the JPEG figure:

    | | size | encode |
    |---|---|---|
    | 2400px proxy, PNG level 1 | 10.4MB | 108ms |
    | 2400px proxy, JPEG q95 4:4:4 | **3.4MB** | **24ms** |

    So ~84ms of server time and ~7MB of wire per proxy preview, before the
    browser's decode of the difference -- and on a weak machine the zlib pass is
    the larger share of the two. 3.0x smaller, 4.5x faster.

    This is not a new quality judgement: JPEG 95 4:4:4 is already the shipped
    default for `/api/export`, measured at 100.2% of grain sigma. What it *does*
    cost is strict bit-equality between what is on screen and a preview-scale
    png export -- the two *renders* are still byte-for-byte identical (see
    `main._render_tier`), but the pixels displayed now go through JPEG. Judge
    grain from a 1:1 render, not from a pixel-peep diff of the preview.
    """
    u = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u, "RGB").save(
        buf, format="JPEG", quality=PREVIEW_QUALITY, subsampling=0,
    )
    return buf.getvalue()


def downscale(arr: np.ndarray, scale: float, device=None) -> np.ndarray:
    """Antialiased downscale in float32.

    Done in torch rather than Pillow so it stays in float and never round-trips
    through an 8-bit intermediate.
    """
    if scale >= 0.999:
        return arr
    h, w, _ = arr.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    # Any accelerator, not just CUDA. The old `== "cuda"` guard quietly ran a
    # 24MP bicubic-antialias downscale in CPU torch on every upload on this
    # machine, because the device here is `mps`.
    if device is not None and device.type in ("cuda", "mps"):
        t = t.to(device)
    out = F.interpolate(t, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
    out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(out)


def upscale(arr: np.ndarray, h: int, w: int, device=None) -> np.ndarray:
    """Plain bicubic upsample to an exact target size, in float32.

    Adds no detail -- it exists only to blow up an already-rendered image back
    to a photo's full pixel dimensions, for "export what I am looking at, at
    full size" rather than "re-render at full size". No `antialias`: that flag
    exists to fight aliasing when *discarding* samples, and there is nothing to
    discard going up. It is the same upsample `render_supersampled` already
    does for supersampling, not `downscale`'s antialiased one run backwards.

    A no-op, and returns the input array itself rather than a copy, when the
    size already matches -- the common case once a source is no bigger than
    the proxy long edge, where the caller must not pay for or introduce any
    deviation from a plain pass-through.
    """
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if device is not None and device.type in ("cuda", "mps"):
        t = t.to(device)
    out = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)
    out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(out)
