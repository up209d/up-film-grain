"""The SSD-backed caches: that they hold the right bytes and none in RAM.

`server/engine/diskcache.py` moved every large cache in the app out of process
memory. That change is unusual among the ones in this suite in that **its
failure modes are mostly invisible to every other module here**: the pipeline
renders the same pixels whether a checkpoint came from a dict or a file, so the
450 checks around this one would all pass on an implementation that silently
never cached anything, or that cached into RAM after all, or that answered one
engine's key with another engine's frame.

So each check here is aimed at a specific way the arrangement could look right
and be wrong:

* **a round trip that is not exact.** A cache whose value comes back to a
  different dtype, shape or device is a plausible wrong picture, which is the
  failure this codebase treats as the worst one it has.
* **caching that is not actually happening.** A store that quietly writes
  nothing is indistinguishable from a hit rate you never measured -- and it is
  the *comfortable* failure, because the app stays correct and merely slow.
* **caching that is happening in RAM.** The whole point is that the payload is
  not resident. Asserting on `nbytes` alone cannot see this; asserting the file
  exists and is the right size can.
* **two stores sharing a directory.** Keys carry no engine identity, so two
  engines in one process could answer each other -- which two in-memory dicts
  could never do, and which the check suite is the one place that builds.
* **eviction that drops too much or too little.** Both are invisible: too
  little wastes the disk, too much rebuilds silently.

Deliberately tiny arrays throughout. Every branch here -- the round trip, the
byte accounting, the LRU, the generation sweep, the file layout -- is
size-free, and this module lands beside the heavy ones under the parallel
runner, where a fixture of any size is multiplied by the pool. See the note at
the top of `tests/checks/prescale.py`.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from server.engine import diskcache as dc
from server.engine.grain_engine import GrainEngine
from tests.harness import Ctx, check, suite


def _store(share: float = 0.5) -> dc.DiskStore:
    return dc.DiskStore("check", share)


#: Bytes of `.npy` header in front of every stored array.
#:
#: The store accounts the *array's* bytes, not the file's, and that is the right
#: choice rather than an oversight: the figure the UI shows is "what this cache
#: is holding", which is the number that used to be resident, and a reader
#: comparing it against Activity Monitor should not have to know about a
#: container format. At 128 bytes against a 184MB frame the difference is
#: nothing; the checks below still spell it out rather than allowing slop,
#: because "roughly matches" would pass on a store that had lost track of a
#: whole small entry.
_NPY_HEADER = 128


@suite("diskcache", "the SSD-backed caches (nothing large stays in RAM)")
def run(cx: Ctx) -> None:
    dev = cx.dev

    # -- the store round trip ------------------------------------------------
    print("\nthe store's payload lives on the disk, not in the process")
    st = _store()
    check("a cache directory was created",
          dc.ROOT is not None and st.enabled,
          f"root {dc.ROOT}")

    t = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7) / 100.0
    t = t.to(dev)
    key = ("round-trip", 1, 2.5, str(dev))
    check("a miss is a miss before anything is stored",
          st.get(key, dev) is None and st.misses == 1,
          f"{st.misses} misses, {st.hits} hits")

    st.put(key, t)
    back = st.get(key, dev)
    # Exactness matters more here than anywhere else in this module: the value
    # being round-tripped is a *photograph mid-pipeline*, and a lossy or
    # reshaped return is the plausible-wrong-picture failure rather than a
    # performance one.
    ok = (
        back is not None
        and back.shape == t.shape
        and back.dtype == t.dtype
        and back.device.type == t.device.type
        and float((back.float() - t.float()).abs().max()) == 0.0
    )
    check("a stored tensor comes back bit-identical, same shape and device",
          ok,
          "maxdiff " + (f"{float((back.float() - t.float()).abs().max()):.2e}"
                        if back is not None else "no entry")
          + f", {tuple(back.shape) if back is not None else '-'} on "
          + f"{back.device.type if back is not None else '-'}")

    # The point of the whole module: the bytes are in a file. Asserting on
    # `nbytes` alone would pass just as happily on an in-memory dict wearing
    # this class's API.
    files = list(st._dir.glob("*.npy"))
    on_disk = sum(f.stat().st_size for f in files)
    check("the payload is a real file, sized like the tensor",
          len(files) == 1 and on_disk == st.nbytes + _NPY_HEADER > _NPY_HEADER,
          f"{len(files)} file, {on_disk} bytes on disk against "
          f"{st.nbytes} accounted plus a {_NPY_HEADER}-byte header")

    # `get` must hand out an independent tensor every time. The in-memory
    # version returned the same object to every caller, which is one careless
    # in-place op away from a corrupted cache entry; reading from a file makes
    # independence structural, and this pins that it stayed that way.
    a = st.get(key, dev)
    b = st.get(key, dev)
    a.add_(1.0)
    check("two gets are independent tensors, not one shared object",
          float((b.float() - t.float()).abs().max()) == 0.0,
          f"mutating one left the other at {float((b.float() - t.float()).abs().max()):.2e}")

    # -- byte accounting -----------------------------------------------------
    print("\nbyte accounting survives re-puts and evictions")
    st2 = _store()
    small = torch.zeros((1, 1, 40, 40), dtype=torch.float32, device=dev)
    for i in range(4):
        st2.put(("acct", i), small)
    n4 = st2.nbytes
    # Re-putting the same key is legal; counting its bytes twice is not, and the
    # symptom is a total that climbs while the entry count stays flat -- the
    # exact bug the in-memory version had to be fixed for.
    for i in range(4):
        st2.put(("acct", i), small)
    check("re-putting a key does not double-count its bytes",
          st2.nbytes == n4 and len(st2) == 4,
          f"{n4} bytes over 4 entries, {st2.nbytes} after re-putting all four")

    disk2 = sum(f.stat().st_size for f in st2._dir.glob("*.npy"))
    check("accounted bytes match the files on disk",
          disk2 == st2.nbytes + _NPY_HEADER * len(st2),
          f"{st2.nbytes} accounted over {len(st2)} entries, {disk2} on disk")

    # An entry over the cap is refused rather than admitted and immediately
    # evicting itself -- otherwise one large tile empties the store on every
    # pass and pays the bookkeeping for it.
    tiny = dc.DiskStore("check-tiny", 0.0)
    tiny.put(("too-big",), small)
    check("a store with no budget stores nothing",
          len(tiny) == 0 and tiny.nbytes == 0,
          f"{len(tiny)} entries")

    # -- eviction ------------------------------------------------------------
    print("\neviction removes the file, not just the index entry")
    st3 = _store()
    for i in range(5):
        st3.put(("ev", i), small)
    before = len(list(st3._dir.glob("*.npy")))
    dropped = st3.drop_if(lambda k: k[1] < 3)
    after = list(st3._dir.glob("*.npy"))
    check("dropped entries take their files with them",
          dropped == 3 and len(st3) == 2 and len(after) == 2,
          f"{before} files -> {len(after)} after dropping {dropped} of 5")
    disk3 = sum(f.stat().st_size for f in after)
    check("dropping updates the byte total",
          disk3 == st3.nbytes + _NPY_HEADER * len(st3),
          f"{st3.nbytes} accounted over {len(st3)} entries, {disk3} on disk")

    st3.clear()
    check("clear empties the directory",
          len(st3) == 0 and st3.nbytes == 0
          and not list(st3._dir.glob("*.npy")),
          f"{len(list(st3._dir.glob('*.npy')))} files left")

    # -- a vanished file is a miss, never a crash ----------------------------
    #
    # The disk is shared with the user, who may empty this folder while the app
    # is running, and with a full disk that can truncate a write. A store that
    # raised there would take a render down for something that has a correct
    # answer: recompute.
    st4 = _store()
    st4.put(("gone",), small)
    for f in st4._dir.glob("*.npy"):
        f.unlink()
    check("a file removed underneath the store reads as a miss",
          st4.get(("gone",), dev) is None and len(st4) == 0,
          "recovered and forgot the entry rather than raising")

    # -- two engines never share a directory ---------------------------------
    #
    # Keys carry the image id, the geometry and the device but no engine
    # identity, so two engines in one process can produce byte-identical keys.
    # In-memory dicts made that safe for free; two directories with the same
    # name would not, and this suite is the one place that builds more than one
    # engine.
    print("\ntwo engines in one process do not share cache files")
    e1, e2 = GrainEngine(dev), GrainEngine(dev)
    dirs = {str(e1._gg_cache._dir), str(e2._gg_cache._dir),
            str(e1.ckpt._store._dir), str(e2.ckpt._store._dir)}
    check("every store gets its own directory",
          len(dirs) == 4,
          f"{len(dirs)} distinct directories across 2 engines x 2 stores")

    shared_key = ("collide", 1)
    e1._gg_cache.put(shared_key, small)
    check("one engine's entry is invisible to another",
          e2._gg_cache.get(shared_key, dev) is None,
          "the identical key missed on the second engine, as it must")

    # -- a spilled array is mapped, not read ---------------------------------
    print("\nsource frames are mapped from disk, not held")
    arr = np.linspace(0, 1, 3 * 40 * 60, dtype=np.float32).reshape(40, 60, 3)
    sp = dc.Spill("check", arr)
    got = sp.array
    check("a spill returns the same values",
          got.shape == arr.shape and got.dtype == arr.dtype
          and float(np.abs(got - arr).max()) == 0.0,
          f"maxdiff {float(np.abs(got - arr).max()):.2e}")
    # `mmap_mode="c"` and not `"r"`: `torch.from_numpy` refuses a read-only
    # array, and `imageio._interp` -- which every resample in the app goes
    # through -- calls exactly that. A spill that came back read-only would
    # break every upload, so it is worth one check that it does not.
    check("the mapped array is writable, so torch will take it",
          got.flags.writeable and torch.from_numpy(got).shape == (40, 60, 3),
          "copy-on-write map, accepted by torch.from_numpy")
    check("it really is a memory map, not a plain read",
          isinstance(got, np.memmap),
          type(got).__name__)

    path = sp.path
    sp.unmap()
    check("unmapping keeps the file and the values",
          path.exists() and float(np.abs(sp.array - arr).max()) == 0.0,
          "re-mapped from disk after the flush dropped it")

    sp.release()
    check("releasing deletes the file",
          not path.exists(),
          f"{path.name} gone")

    # -- the engine's caches are the disk ones -------------------------------
    #
    # The wiring check. Everything above tests `DiskStore` in isolation; this
    # tests that the engine is actually using one, which is the thing a
    # refactor could quietly undo while every other check in the suite passed.
    print("\nthe engine renders through the disk stores")
    from server import params as P
    from tests.scene import scene

    eng = GrainEngine(dev)
    img = scene(160, 240)
    p = P.sanitize({"intensity": 30.0, "global_intensity": 9.0,
                    "halation": 0.5, "dust": 20.0})
    eng.render_image(img, p, 1.0, tile=4096, supersample=1,
                     checkpoint_id="dc:proxy")
    ck_files = list(eng.ckpt._store._dir.glob("*.npy"))
    gg_files = list(eng._gg_cache._dir.glob("*.npy"))
    check("a render writes its checkpoints to the disk",
          len(ck_files) > 0 and eng.ckpt.nbytes > 0,
          f"{len(ck_files)} checkpoint files, {eng.ckpt.nbytes / 1e6:.1f}MB")
    check("a render writes its grain textures to the disk",
          len(gg_files) > 0 and eng._gg_cache.nbytes > 0,
          f"{len(gg_files)} texture files, {eng._gg_cache.nbytes / 1e6:.1f}MB")

    warm = eng.render_image(img, p, 1.0, tile=4096, supersample=1,
                            checkpoint_id="dc:proxy")
    cold = GrainEngine(dev).render_image(img, p, 1.0, tile=4096, supersample=1)
    check("a warm disk cache renders the identical frame",
          float(np.abs(warm - cold).max()) == 0.0 and eng.ckpt.hits > 0,
          f"maxdiff {float(np.abs(warm - cold).max()):.2e} over "
          f"{eng.ckpt.hits} checkpoint hits")

    # -- reporting -----------------------------------------------------------
    print("\nthe readout the UI shows adds up")
    s = dc.stats()
    total = sum(x["bytes"] for x in s["parts"])
    check("reported total equals the sum of its parts",
          s["bytes"] == total,
          f"{s['bytes']} reported, {total} summed over {len(s['parts'])} parts")
    check("memory is reported and the peak is not below the current",
          s["memory"]["rss"] > 0 and s["memory"]["peak"] >= s["memory"]["rss"],
          f"rss {s['memory']['rss'] / 1e6:.0f}MB, "
          f"peak {s['memory']['peak'] / 1e6:.0f}MB")

    # `_budget_bytes` is clamped to a share of what the volume can spare, which
    # is the difference between a wish and a promise: a laptop with 6GB free
    # must not be told it may write 8GB of grain textures.
    have = os.statvfs(dc.ROOT) if hasattr(os, "statvfs") else None
    check("the budget never exceeds its share of free space",
          have is None
          or s["budget"] <= have.f_bavail * have.f_frsize * dc._FREE_SPACE_SHARE + 1,
          f"budget {s['budget'] / 1e9:.1f}GB")

    for st_ in (st, st2, st3, st4, tiny, e1, e2):
        (st_ if isinstance(st_, dc.DiskStore) else st_._gg_cache).clear()
