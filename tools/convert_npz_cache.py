#!/usr/bin/env python3
"""Convert the legacy .npz expert cache in k3-experts/ to the raw .bin span layout.

Why
---
tools/fetch_v2.py reads a .bin expert as one verbatim 17,547,264-byte shard span:
zero parse, page-aligned, straight into the pread pool at ~6.9 GB/s. A .npz goes
through np.load (zip directory walk + CRC32 + six allocations) at ~2.9 GB/s, and
the legacy files are ~15% of every routed selection, so they drag the per-token
expert read. Same bytes, same order — this is a pure repack:

    w1_p 5505024 | w1_s 344064 | w2_p 5505024 | w2_s 344064 | w3_p 5505024 | w3_s 344064

Safety
------
* One expert at a time per worker. Peak extra disk is workers x 17.5 MB, and each
  conversion NETS -1462 bytes (the zip container overhead). Never bulk-writes.
* .tmp -> fsync -> verify the tmp's bytes against the arrays that were read ->
  atomic rename -> re-verify the final size -> only then unlink the .npz. A
  corrupt or short .bin is therefore never visible under the real name, and the
  .npz is never removed until a byte-correct .bin has replaced it.
* Resumable by construction: the remaining .npz files ARE the work list. A crash
  leaves either an untouched .npz or a complete .bin, plus at worst an orphan
  .convtmp* (see --gc-tmp).
* Safe to run against a live cache. Readers see the rename atomically and prefer
  .bin (fetch_v2._has_bin / _cache_load); unlinking a .npz that a reader already
  opened is harmless on POSIX. Concurrent converters race harmlessly (same bytes,
  last rename wins, a lost unlink is ignored). Use --shard i/n to split the work.
* On macOS, F_NOCACHE is enabled by default on every read and write, so 222 GB
  of conversion traffic does not evict the page cache the resident spine
  depends on. --throttle-ms yields disk bandwidth back to a running inference.

Usage
-----
    # prove it on a few experts, destroying nothing:
    ./venv/bin/python tools/convert_npz_cache.py --selftest 5

    # dry run: what would happen
    ./venv/bin/python tools/convert_npz_cache.py --dry-run

    # sample conversion (really converts N experts, then stops)
    ./venv/bin/python tools/convert_npz_cache.py --limit 8

    # THE FULL RUN (12,676 experts, ~222 GB read + 222 GB written)
    ./venv/bin/python tools/convert_npz_cache.py --workers 4
"""
import argparse
import concurrent.futures
import hashlib
import os
import re
import sys
import threading
import time

try:
    # Only ever called behind a Darwin guard, for F_NOCACHE.
    import fcntl
except ImportError:  # Windows has no fcntl module
    fcntl = None

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_v2                                             # noqa: E402  (layout only)

# Hard-coded and cross-checked against fetch_v2 so a layout drift is a startup
# error, not 12,676 silently mis-packed experts.
EXPERT_SPAN = 17547264
_P, _S = 5505024, 344064
LAYOUT = [                                       # (npz key, offset, nbytes, shape)
    ("w1_p", 0,                _P, (3072, 1792)),
    ("w1_s", _P,               _S, (3072, 112)),
    ("w2_p", _P + _S,          _P, (3584, 1536)),
    ("w2_s", 2 * _P + _S,      _S, (3584, 96)),
    ("w3_p", 2 * (_P + _S),    _P, (3072, 1792)),
    ("w3_s", 3 * _P + 2 * _S,  _S, (3072, 112)),
]
assert EXPERT_SPAN == fetch_v2.EXPERT_SPAN, "EXPERT_SPAN drift vs fetch_v2"
assert LAYOUT == fetch_v2.LAYOUT, "LAYOUT drift vs fetch_v2"
assert sum(nb for _, _, nb, _ in LAYOUT) == EXPERT_SPAN

F_NOCACHE = 48                                   # <sys/fcntl.h>, Darwin
CHUNK = 8 << 20
_NPZ = re.compile(r"^(L(\d+)-E(\d+))\.npz$")


# --- io helpers -------------------------------------------------------------
def _nocache(fd, on):
    if on and sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except OSError:
            pass


def load_npz(path, nocache=True):
    """The six arrays of a legacy expert, validated. np.load walks the zip
    directory and checks each member's CRC32, so a rotten .npz raises here —
    before anything is written and long before anything is unlinked."""
    with open(path, "rb") as f:
        _nocache(f.fileno(), nocache)
        z = np.load(f)
        out = {}
        for key, _off, nb, shape in LAYOUT:
            a = z[key]
            if a.dtype != np.uint8 or a.shape != shape or a.nbytes != nb:
                raise ValueError(f"{path}: {key} is {a.dtype}{a.shape} ({a.nbytes} B), "
                                 f"expected uint8{shape} ({nb} B)")
            out[key] = np.ascontiguousarray(a)
        extra = set(z.files) - {k for k, _, _, _ in LAYOUT}
        if extra:
            raise ValueError(f"{path}: unexpected members {sorted(extra)}")
    return out


def _digest_arrays(arrs):
    h = hashlib.blake2b(digest_size=32)
    for key, _off, _nb, _shape in LAYOUT:
        h.update(memoryview(arrs[key]).cast("B"))
    return h.digest()


def _digest_file(path, nocache=True):
    h = hashlib.blake2b(digest_size=32)
    n = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        _nocache(fd, nocache)
        while True:
            b = os.read(fd, CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    finally:
        os.close(fd)
    return h.digest(), n


def write_span(tmp, arrs, nocache=True):
    """The span, in LAYOUT order, fsync'd. Returns bytes written."""
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    total = 0
    try:
        _nocache(fd, nocache)
        for key, off, nb, _shape in LAYOUT:
            mv = memoryview(arrs[key]).cast("B")
            assert off == total and len(mv) == nb
            while mv:
                w = os.write(fd, mv[:CHUNK])
                mv = mv[w:]
                total += w
        os.fsync(fd)
    finally:
        os.close(fd)
    return total


# --- one expert -------------------------------------------------------------
class Skip(Exception):
    pass


def convert_one(stem, root, nocache=True, fsync_dir=True, keep_npz=False):
    """Returns bytes converted. Raises on failure, having left the .npz intact."""
    npz_path = os.path.join(root, stem + ".npz")
    bin_path = os.path.join(root, stem + ".bin")
    tmp = f"{bin_path}.convtmp{os.getpid()}-{threading.get_ident()}"

    arrs = load_npz(npz_path, nocache)            # CRC-checked read
    want = _digest_arrays(arrs)
    try:
        n = write_span(tmp, arrs, nocache)
        if n != EXPERT_SPAN:
            raise IOError(f"{tmp}: wrote {n} B, expected {EXPERT_SPAN}")
        got, size = _digest_file(tmp, nocache)    # verify BEFORE it becomes visible
        if size != EXPERT_SPAN or got != want:
            raise IOError(f"{tmp}: readback mismatch (size {size}, "
                          f"digest {'ok' if got == want else 'BAD'})")
        os.replace(tmp, bin_path)                 # atomic; readers see all or nothing
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)                        # failure: leave the .npz alone

    st = os.stat(bin_path)
    if st.st_size != EXPERT_SPAN:                 # someone else clobbered it
        raise IOError(f"{bin_path}: size {st.st_size} after rename")
    if fsync_dir:
        dfd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(dfd)                         # rename durable before the unlink
        finally:
            os.close(dfd)
    if not keep_npz:
        try:
            os.unlink(npz_path)
        except FileNotFoundError:
            pass                                  # a concurrent converter won
    return EXPERT_SPAN


# --- selftest: convert to scratch and prove the dequant is bit-identical -----
def selftest(root, n, nocache=True):
    from mxfp4 import dequant_mxfp4
    stems = sorted(scan(root))[:n]
    if not stems:
        print("no .npz left in the cache — nothing to self-test")
        return 0
    scratch = os.path.join(root, f".selftest{os.getpid()}.bin")
    bad = 0
    try:
        for stem in stems:
            arrs = load_npz(os.path.join(root, stem + ".npz"), nocache)
            write_span(scratch, arrs, nocache)
            m = np.memmap(scratch, dtype=np.uint8, mode="r")
            if m.nbytes != EXPERT_SPAN:
                raise IOError(f"{scratch}: {m.nbytes} B")
            view = {k: m[off:off + nb].reshape(shape)
                    for k, off, nb, shape in LAYOUT}
            raw_ok = all(bytes(memoryview(view[k]).cast("B"))
                         == bytes(memoryview(arrs[k]).cast("B"))
                         for k, _, _, _ in LAYOUT)
            deq_ok = True
            for w in ("w1", "w2", "w3"):
                a = dequant_mxfp4(arrs[w + "_p"], arrs[w + "_s"])
                b = dequant_mxfp4(np.ascontiguousarray(view[w + "_p"]),
                                  np.ascontiguousarray(view[w + "_s"]))
                # tobytes(), not array_equal: bit-identity, so -0.0 != 0.0 and a
                # stray NaN cannot pass
                deq_ok &= (a.dtype == b.dtype and a.shape == b.shape
                           and a.tobytes() == b.tobytes())
            ok = raw_ok and deq_ok
            bad += not ok
            print(f"  {stem:14s} raw bytes {'==' if raw_ok else '!='}   "
                  f"dequant fp32 {'bit-identical' if deq_ok else 'DIFFERS'}   "
                  f"{'ok' if ok else 'FAIL'}")
            del view, m
    finally:
        if os.path.exists(scratch):
            os.unlink(scratch)
    print("SELFTEST PASS" if not bad else f"SELFTEST FAIL ({bad} bad)")
    return 0 if not bad else 1


# --- driver -----------------------------------------------------------------
def scan(root, shard=None):
    stems = []
    with os.scandir(root) as it:
        for de in it:
            m = _NPZ.match(de.name)
            if m:
                stems.append(m.group(1))
    stems.sort(key=lambda s: tuple(int(x) for x in re.findall(r"\d+", s)))
    if shard:
        i, n = shard
        stems = [s for k, s in enumerate(stems) if k % n == i]
    return stems


def gc_tmp(root, age_s=3600):
    now, gone = time.time(), 0
    with os.scandir(root) as it:
        for de in it:
            if ".convtmp" in de.name:
                try:
                    if now - de.stat().st_mtime > age_s:
                        os.unlink(de.path)
                        gone += 1
                except OSError:
                    pass
    return gone


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=fetch_v2.ECACHE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="stop after N experts (0 = all)")
    ap.add_argument("--shard", default="", metavar="i/n",
                    help="convert only stems where index %% n == i")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", type=int, default=0, metavar="N",
                    help="prove N conversions bit-identical, delete nothing, exit")
    ap.add_argument("--keep-npz", action="store_true",
                    help="write the .bin but do not unlink the .npz (disk grows!)")
    ap.add_argument("--no-nocache", dest="nocache", action="store_false",
                    help="allow conversion traffic into the page cache")
    ap.add_argument("--no-fsync-dir", dest="fsync_dir", action="store_false",
                    help="skip the per-file directory fsync (faster, weaker crash story)")
    ap.add_argument("--throttle-ms", type=float, default=0.0,
                    help="sleep between experts to leave disk for a running model")
    ap.add_argument("--gc-tmp", action="store_true",
                    help="remove orphan .convtmp* older than an hour, then exit")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise SystemExit(f"no such cache dir: {root}")
    if args.gc_tmp:
        print(f"removed {gc_tmp(root)} orphan .convtmp files")
        return 0
    if args.selftest:
        return selftest(root, args.selftest, args.nocache)

    shard = None
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        shard = (i, n)
    stems = scan(root, shard)
    if args.limit:
        stems = stems[:args.limit]
    total = len(stems)
    print(f"{root}: {total} .npz to convert"
          + (f" (shard {args.shard})" if shard else "")
          + f"  ~{total * EXPERT_SPAN / 1e9:.1f} GB, net disk {-1462 * total / 1e6:+.1f} MB, "
          f"peak extra {args.workers * EXPERT_SPAN / 1e6:.0f} MB")
    if args.dry_run or not total:
        for s in stems[:5]:
            print("  would convert", s)
        return 0

    done = failed = 0
    nbytes = 0
    lock = threading.Lock()
    t0 = time.time()

    def job(stem):
        nonlocal done, failed, nbytes
        try:
            n = convert_one(stem, root, args.nocache, args.fsync_dir, args.keep_npz)
        except Skip:
            return
        except Exception as e:
            with lock:
                failed += 1
            print(f"  FAIL {stem}: {type(e).__name__}: {e}", flush=True)
            return
        with lock:
            done += 1
            nbytes += n
            d = done
        if d % 100 == 0 or d == total:
            el = time.time() - t0
            print(f"  {d}/{total}  {nbytes/1e9:.1f} GB  {nbytes/1e9/max(el,1e-9):.2f} GB/s"
                  f"  {el:.0f}s elapsed", flush=True)
        if args.throttle_ms:
            time.sleep(args.throttle_ms / 1000.0)

    with concurrent.futures.ThreadPoolExecutor(max(1, args.workers)) as ex:
        list(ex.map(job, stems))

    el = time.time() - t0
    print(f"converted {done}/{total}, failed {failed}, {nbytes/1e9:.1f} GB in {el:.0f}s")
    left = len(scan(root))
    print(f".npz remaining in cache: {left}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
