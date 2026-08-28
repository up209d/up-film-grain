"""Image decode/encode. Keeps everything float32 in 0..1 internally."""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, PngImagePlugin

from .engine.device import device_work

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


#: What every written file says made it. A constant rather than a literal at
#: three encode sites, and deliberately *not* versioned: the tag is there to say
#: which tool to open a question about, and a build number in it would only
#: promise a provenance trail this does not actually keep.
SOFTWARE = "UP Film Grain"


def _description(preset: dict | None) -> str | None:
    """The one line a file records about the look it was rendered with.

    `None` when no preset was used, and that is a *behaviour* rather than a
    fallback: an export dialled in by hand has no name to record, and inventing
    one ("custom") would put a fact in the file that is not true of it. Nothing
    is written in that case -- no EXIF block, no text chunk.

    The author travels with the name because a preset is someone's work and this
    is the only place the credit survives leaving the app; the file is what gets
    passed around, not the preset it came from.
    """
    if not preset or not preset.get("name"):
        return None
    out = f"Preset: {preset['name']}"
    if preset.get("author"):
        out += f" by {preset['author']}"
        if preset.get("author_link"):
            out += f" ({preset['author_link']})"
    if preset.get("lut"):
        out += f"; LUT: {preset['lut']}"
    return out


def _exif_bytes(desc: str) -> bytes:
    """An EXIF block carrying the description and the software tag.

    Built through Pillow's own `Image.Exif` rather than by hand: the IFD layout
    is offsets-into-a-blob and a hand-rolled one that is subtly wrong reads as a
    corrupt file to some viewers rather than as a missing tag.

    ASCII tags only. `UserComment` (0x9286) is the more usual home for free
    text, but it carries an encoding prefix that half of the tools that read it
    get wrong; `ImageDescription` is plain and every viewer shows it.
    """
    exif = Image.Exif()
    exif[0x010E] = desc      # ImageDescription
    exif[0x0131] = SOFTWARE  # Software
    return exif.tobytes()


def _png16_rgb(u: np.ndarray, desc: str | None = None) -> bytes:
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

    def text(key: str, value: str) -> bytes:
        """A `tEXt` chunk. PNG has no EXIF of its own that anything reads.

        Latin-1, which is what the spec allows in `tEXt`; a name or a link
        outside it degrades to `iTXt`-less ASCII rather than writing bytes no
        reader can decode.
        """
        payload = (key.encode("latin-1", "replace") + b"\x00"
                   + value.encode("latin-1", "replace"))
        return chunk(b"tEXt", payload)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16bpc, colour type 2
    # Before IDAT: the spec allows text chunks either side, and a reader that
    # streams stops caring once it has the pixels.
    meta = (text("Description", desc) + text("Software", SOFTWARE)) if desc else b""
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + meta
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def encode(arr: np.ndarray, fmt: str = "png16", quality: int = 95,
           preset: dict | None = None) -> bytes:
    """Encode a finished render, optionally stamped with the preset that made it.

    `preset` is the `load_presets` record -- name, author, author_link, lut --
    or `None`, which writes a file with no metadata at all. Nothing else about
    the render is recorded: the parameter values are the look and they belong in
    a preset file, not in half a kilobyte of EXIF nobody can load back.
    """
    a = np.clip(arr, 0.0, 1.0)
    desc = _description(preset)
    if fmt == "png16":
        return _png16_rgb((a * 65535.0 + 0.5).astype(np.uint16), desc)

    u = (a * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    if fmt == "jpeg":
        # 4:4:4 -- chroma subsampling would smear the chroma grain away.
        extra = {"exif": _exif_bytes(desc)} if desc else {}
        Image.fromarray(u, "RGB").save(
            buf, format="JPEG", quality=int(quality), subsampling=0, **extra
        )
    else:
        info = None
        if desc:
            info = PngImagePlugin.PngInfo()
            info.add_text("Description", desc)
            info.add_text("Software", SOFTWARE)
        Image.fromarray(u, "RGB").save(buf, format="PNG", pnginfo=info)
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


def _interp(arr: np.ndarray, h: int, w: int, antialias: bool,
            device=None) -> np.ndarray:
    """One bicubic resample to an exact size, in float32.

    The shared body of `downscale`, `upscale` and `resize_to`. They differ in
    exactly two things -- how the target size is arrived at, and whether
    `antialias` is passed -- and both of those are the caller's decision, so
    three copies of this were three places for the device handling and the
    clamp to drift apart.
    """
    # The one place in this module that touches the GPU, so the one place that
    # has to say so. `device_work` is what stops the engine's idle flush from
    # calling `empty_cache` underneath this -- an upload resamples on a request
    # thread holding no render lock at all, and on MPS that collision aborts the
    # process rather than merely racing. See `engine/device.py`.
    with device_work():
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        # Any accelerator, not just CUDA. The old `== "cuda"` guard quietly ran a
        # 24MP bicubic-antialias downscale in CPU torch on every upload on this
        # machine, because the device here is `mps`.
        if device is not None and device.type in ("cuda", "mps"):
            t = t.to(device)
        out = F.interpolate(
            t, size=(h, w), mode="bicubic", antialias=antialias,
            align_corners=False,
        )
        out = out.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(out)


def downscale(arr: np.ndarray, scale: float, device=None) -> np.ndarray:
    """Antialiased downscale in float32.

    Done in torch rather than Pillow so it stays in float and never round-trips
    through an 8-bit intermediate.
    """
    if scale >= 0.999:
        return arr
    h, w, _ = arr.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return _interp(arr, nh, nw, True, device)


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
    return _interp(arr, h, w, False, device)


def resize_to(arr: np.ndarray, h: int, w: int, device=None) -> np.ndarray:
    """Resample to an exact target size in either direction, in float32.

    `downscale` takes a factor and always antialiases; `upscale` takes a size
    and never does. This takes a size and decides, which is what a caller that
    does not know which way it is going needs -- prescaling a photograph to a
    fixed megapixel count (`models/upload.py`) and writing a prescaled export
    back at the file's own dimensions (`models/export_job.py`) are each an
    enlargement on one photograph and a reduction on the next.

    `antialias` is on as soon as *either* axis shrinks, for the reason `upscale`
    gives for omitting it read in the other direction: the flag exists to fight
    aliasing when samples are being discarded, and an axis that shrinks is
    discarding them however the other axis moves. Getting this wrong is not
    subtle on a rendered frame -- grain is nothing but content at the Nyquist
    limit, so an unfiltered reduction of it aliases into visible crawl.

    A no-op returning the input array itself when the size already matches, the
    same pass-through `upscale` offers and for the same reason: `Upload.at()`
    calls this on a photograph that is already the target size.
    """
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    shrinking = h < arr.shape[0] or w < arr.shape[1]
    return _interp(arr, h, w, shrinking, device)
