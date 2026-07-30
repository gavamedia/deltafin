"""Low-level spine read primitives (fd cache, chunked preadv, OS I/O hints).

The int8 spine is 53 GB and is re-read from disk on every forward pass, so the
read is the single largest term in a decode token.  tools/spine_fast.py's
original reader does, per layer, `open(path); f.readinto(view)` for each of the
~14 .i8 payloads and ~14 .sc scale blobs, farmed out to
K3_SPINE_READ_THREADS worker threads with one work item per FILE.  Two things
are wrong with that shape:

  * ~2,500 open()/close() pairs per token.  The paths never change, the files
    are read-only for the life of the process, and macOS gives this process
    245,760 fds (kern.maxfilesperproc) against a soft RLIMIT_NOFILE already at
    1,048,576 -- so all 2,506 spine blobs can be opened once and kept.

  * ONE WORK ITEM PER FILE MEANS THE BIGGEST FILE IS THE CRITICAL PATH.  A KDA
    layer is 5 x 88.1 MB + 3 x 44.0 MB + 2 x 25.7 MB + 6.4 MB + change = 634 MB
    across 14 payloads.  Perfectly balanced over 8 threads that is 79 MB each --
    but no thread can finish before it has read a whole 88.1 MB file, so the
    makespan is >= 88.1 MB, i.e. at best 90% of the threads' time is useful and
    adding threads past 7 buys nothing at all.  Splitting every file into
    equal CHUNKS and letting the threads pull from a shared work list removes
    the imbalance: makespan <= ideal + one chunk.

Everything here is opt-in.  With K3_SPINE_FDCACHE=0 / K3_SPINE_CHUNK_MB=0 the
module is not consulted at all and spine_fast keeps its original reader.

Flags (all default to the pre-existing behaviour):
    K3_SPINE_FDCACHE=1     keep one fd per spine blob open for the process life
    K3_SPINE_CHUNK_MB=<n>  split reads into n-MiB work items (0 = per-file)
    K3_SPINE_RDADVISE=1    prefetch the next layer (Darwin F_RDADVISE;
                           Linux POSIX_FADV_WILLNEED)
    K3_SPINE_IOPOL=1       Darwin: setiopolicy_np(DISK, PROCESS, IMPORTANT) +
                           reader threads at QOS_CLASS_USER_INITIATED
    K3_SPINE_NOCACHE=1     keep completed spine ranges out of cache (Darwin
                           F_NOCACHE; Linux POSIX_FADV_DONTNEED; DIAGNOSTIC)
"""
import ctypes
import ctypes.util
import os
import struct
import sys
import threading

try:
    # Only ever called behind a Darwin guard, for F_NOCACHE and F_RDADVISE.
    import fcntl
except ImportError:  # Windows has no fcntl module
    fcntl = None

try:
    from runtime_platform import darwin_file_hints_enabled
except ImportError:  # imported as tools.spine_io instead of a top-level module
    from .runtime_platform import darwin_file_hints_enabled

# ---------------------------------------------------------------- flags -----
_IS_DARWIN = sys.platform == "darwin"
_IS_LINUX = sys.platform.startswith("linux")
FDCACHE = os.environ.get("K3_SPINE_FDCACHE", "0") == "1"
CHUNK_MB = float(os.environ.get("K3_SPINE_CHUNK_MB", "0") or 0)
CHUNK = int(CHUNK_MB * (1 << 20))
RDADVISE = os.environ.get("K3_SPINE_RDADVISE", "0") == "1"
IOPOL = darwin_file_hints_enabled(
    os.environ.get("K3_SPINE_IOPOL", "0") == "1"
)
NOCACHE = os.environ.get("K3_SPINE_NOCACHE", "0") == "1"
STREAM_TIER = os.environ.get("K3_SPINE_STREAM_NOCACHE", "0") == "1"

ENABLED = (FDCACHE or CHUNK > 0 or RDADVISE or NOCACHE
           or STREAM_TIER)

# ------------------------------------------------------- darwin constants ---
F_NOCACHE = 48
F_RDADVISE = 44          # struct radvisory { off_t ra_offset; int ra_count; }
F_RDAHEAD = 45

IOPOL_TYPE_DISK = 0
IOPOL_SCOPE_PROCESS = 0
IOPOL_SCOPE_THREAD = 1
# <sys/resource.h>: IOPOL_DEFAULT=0, IOPOL_IMPORTANT=1.  This must be 1:
# using 0 makes K3_SPINE_IOPOL silently reset the policy to default instead of
# protecting foreground model reads from background-I/O throttling.
IOPOL_IMPORTANT = 1

QOS_CLASS_USER_INITIATED = 0x19

_libc = None


def _lc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    return _libc


def set_process_io_policy():
    """Ask the kernel to treat this process's disk I/O as important rather than
    letting it inherit a throttled/utility policy (which a process launched from
    a background agent or a `nice`d shell can pick up).  No-op off Darwin."""
    if not IOPOL:
        return None
    try:
        rc = _lc().setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS,
                                  IOPOL_IMPORTANT)
        return rc == 0
    except Exception:
        return None


def set_thread_qos():
    """Reader threads created by Python inherit the creating thread's QoS, which
    can land them on E-cores.  The read path is a page-cache -> user-buffer
    memcpy in the kernel, so core class matters; ask for USER_INITIATED."""
    if not IOPOL:
        return None
    try:
        return _lc().pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) == 0
    except Exception:
        return None


# ------------------------------------------------------------- fd cache -----
# Read-only blobs that never change for the life of the process.  Opened once,
# never closed.  All reads go through preadv(), which is position-independent,
# so a single fd is safe to share across every reader thread with no seek state
# and no lock.
_FDS = ({}, {})            # [0] = normal (page-cached), [1] = F_NOCACHE
_FDS_LOCK = threading.Lock()
OPENS = 0


def _use_nocache(requested=False):
    """Whether a Darwin fd needs the persistent F_NOCACHE attribute."""
    return _IS_DARWIN and (requested or NOCACHE)


def _apply_nocache(fd, requested=False):
    """Apply Darwin F_NOCACHE, never command number 48 on another kernel."""
    if not _use_nocache(requested):
        return False
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return True


def _linux_fadvise(fd, offset, count, advice):
    """Best-effort Linux cache advice; unsupported libc/filesystems are safe."""
    if not _IS_LINUX:
        return False
    advise = getattr(os, "posix_fadvise", None)
    if advise is None or advice is None:
        return False
    try:
        advise(fd, int(offset), int(count), advice)
    except (OSError, ValueError):
        return False
    return True


def _drop_read_cache(fd, offset, count, requested=False):
    """Drop only a completed Linux read range when nocache was requested."""
    if not (requested or NOCACHE):
        return False
    return _linux_fadvise(
        fd, offset, count, getattr(os, "POSIX_FADV_DONTNEED", None)
    )


def fd_for(path, nocache=False):
    nc = 1 if _use_nocache(nocache) else 0
    tbl = _FDS[nc]
    fd = tbl.get(path)
    if fd is not None:
        return fd
    with _FDS_LOCK:
        fd = tbl.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            try:
                if nc:
                    _apply_nocache(fd, nocache)
            except Exception:
                os.close(fd)
                raise
            tbl[path] = fd
            global OPENS
            OPENS += 1
    return fd


def _open_transient(path, nocache=False):
    fd = os.open(path, os.O_RDONLY)
    try:
        _apply_nocache(fd, nocache)
    except Exception:
        os.close(fd)
        raise
    return fd


def close_all():
    with _FDS_LOCK:
        for tbl in _FDS:
            for fd in tbl.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            tbl.clear()


# ------------------------------------------------------------ readahead -----
def rdadvise(path, offset=0, count=0):
    """Ask the kernel to prefetch [offset, offset+count); 0 means to EOF."""
    if not (_IS_DARWIN or _IS_LINUX):
        return False
    try:
        fd = fd_for(path) if FDCACHE else _open_transient(path)
    except OSError:
        return False
    try:
        if not count:
            count = os.fstat(fd).st_size - offset
        if count <= 0:
            return False
        if _IS_DARWIN:
            # struct radvisory is {off_t; int;} -> 8 + 4 + 4 pad on arm64.
            fcntl.fcntl(
                fd, F_RDADVISE, struct.pack("qi4x", int(offset), int(count))
            )
            return True
        return _linux_fadvise(
            fd, offset, count, getattr(os, "POSIX_FADV_WILLNEED", None)
        )
    except OSError:
        return False
    finally:
        if not FDCACHE:
            try:
                os.close(fd)
            except OSError:
                pass


# --------------------------------------------------------------- reading ----
def _pread_into(fd, mv, off):
    """preadv into a preallocated slice.  os.pread would allocate and then need
    a second memcpy; preadv writes straight into our packed buffer."""
    n = len(mv)
    got = 0
    while got < n:
        r = os.preadv(fd, [mv[got:]], off + got)
        if r <= 0:
            raise OSError(f"short read: {got}/{n} @{off}")
        got += r
    return got


def read_one(path, mv, off=0, nocache=False):
    fd = fd_for(path, nocache) if FDCACHE else _open_transient(path, nocache)
    try:
        got = _pread_into(fd, mv, off)
        _drop_read_cache(fd, off, got, nocache)
        return got
    finally:
        if not FDCACHE:
            os.close(fd)


def make_jobs(files):
    """files: iterable of (path, file_offset, dest_memoryview).
    -> a work list split so no item exceeds CHUNK bytes.  Large files are
    emitted round-robin-ish (biggest first) so that with dynamic scheduling the
    tail of the list is small work items, which is what balances the makespan."""
    if CHUNK <= 0:
        return list(files)
    jobs = []
    for path, foff, mv in files:
        n = len(mv)
        if n <= CHUNK:
            jobs.append((path, foff, mv))
            continue
        o = 0
        while o < n:
            k = min(CHUNK, n - o)
            jobs.append((path, foff + o, mv[o:o + k]))
            o += k
    # Longest-processing-time-first: with a shared work list and greedy pulling,
    # descending order bounds the makespan at (ideal + smallest item) instead of
    # (ideal + largest item).
    jobs.sort(key=lambda j: -len(j[2]))
    return jobs


_QOS_DONE = threading.local()


def run_jobs(jobs, executor, threads, nocache=False):
    """Execute a work list across `threads` workers pulling from a shared
    iterator.  One future per THREAD, not per job: at 8 MiB chunks a layer is
    ~80 work items and ~7,400 per token, and a Future costs ~20 us to create
    and resolve, which would be 150 ms/token of pure bookkeeping.

    `nocache` marks this layer as STREAMING TIER -- see stream_tier() below."""
    if not jobs:
        return
    if threads <= 1 or len(jobs) == 1:
        _drain(iter(jobs), threading.Lock(), nocache)
        return
    it = iter(jobs)
    lock = threading.Lock()
    n = min(threads, len(jobs))
    futs = [executor.submit(_drain, it, lock, nocache) for _ in range(n - 1)]
    _drain(it, lock, nocache)           # the caller is a worker too
    for f in futs:
        f.result()                      # propagate any OSError


def _drain(it, lock, nocache=False):
    if IOPOL and not getattr(_QOS_DONE, "set", False):
        set_thread_qos()
        _QOS_DONE.set = True
    nxt = it.__next__
    while True:
        with lock:
            try:
                path, foff, mv = nxt()
            except StopIteration:
                return
        fd = fd_for(path, nocache) if FDCACHE else _open_transient(path, nocache)
        try:
            got = _pread_into(fd, mv, foff)
            _drop_read_cache(fd, foff, got, nocache)
        finally:
            if not FDCACHE:
                os.close(fd)


# ------------------------------------------------------- two-tier reading ---
# THE CYCLIC-SCAN PROBLEM.  Decode walks layers 0..91 in the same order every
# token, so the spine is a 53 GB working set scanned cyclically.  Under LRU that
# is the pathological case: whatever the page cache holds is exactly what gets
# evicted just before it is needed again, so a cache smaller than the working
# set converges on a ~0% hit rate no matter how big it is.  Measured on this
# machine, cold layer reads run at ~7.0 GB/s and page-cache-resident ones at
# ~27-38 GB/s, so those misses are worth ~6 s of every token.
#
# The fix is not more cache, it is a FIXED SUBSET. If layers 0..K are read
# normally and layers K+1..91 use the host's no-cache advice, the streaming tier
# stops flooding the cache, the resident tier stays resident, and the hit rate
# becomes (K+1)/92 instead of ~0. Darwin uses F_NOCACHE; Linux discards each
# completed range with POSIX_FADV_DONTNEED.
#
# This costs no anonymous memory, which is the whole point: the earlier explicit
# 29.7 GB heap cache achieved the same fixed-subset effect but paid for it in
# swappable pages and fell off an 8x cliff.  Clean file pages cannot be swapped;
# under pressure the kernel just drops them and we are back to 7 GB/s.
_RESIDENT = set()          # layer prefixes deliberately kept in the page cache
_RESIDENT_ACTIVE = False


def set_resident_tier(prefixes):
    """Declare the layer prefixes that should stay page-cache resident.  Until
    this is called every layer is treated as resident, i.e. today's behaviour."""
    global _RESIDENT_ACTIVE
    _RESIDENT.clear()
    _RESIDENT.update(prefixes)
    _RESIDENT_ACTIVE = True


def stream_tier(prefix):
    """True if this layer should use the host's streaming cache policy."""
    if not STREAM_TIER or not _RESIDENT_ACTIVE:
        return False
    return prefix not in _RESIDENT


def describe():
    return (f"fdcache={int(FDCACHE)} chunk_mb={CHUNK_MB:g} rdadvise={int(RDADVISE)} "
            f"iopol={int(IOPOL)} nocache={int(NOCACHE)} "
            f"stream_nocache={int(STREAM_TIER)}")
