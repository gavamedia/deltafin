"""Self-tuning, pressure-aware RAM cache for the int8 spine.

WHY THE OLD ONE WAS OFF BY DEFAULT
----------------------------------
kimi_run.py's first attempt sized a spine RAM cache from "free + inactive" and
took 29.7 GB on this 64 GB box.  A decode token went from 14 s to 122 s: the
inactive pages it counted as spare were the page cache that was already serving
the spine and the expert blobs, so claiming them as anonymous heap did not add
capacity, it MOVED capacity from a class the kernel can drop for free (clean
file pages) into a class it can only reclaim by compressing or swapping.  It
then compressed, and every token paid for it.  Two things follow, and this
module implements both:

  1. THE HEADROOM SIGNAL MUST BE `free`, NOT `free + inactive`.  `free` is the
     only class that can be taken without displacing something that is already
     doing useful work.  We additionally hold a reserve (K3_SPINE_CACHE_FLOOR_GB,
     default 8 GB) below which we stop growing, because the page cache still has
     to hold the expert payloads.

  2. GROWTH MUST BE WATCHED, NOT PREDICTED.  No static budget can know what else
     the machine will do while we run. Every admission re-reads the kernel's own
     accounting (Mach host_statistics64 on macOS; /proc on Linux); if the
     compressor or swapper starts moving, we stop growing and give memory back
     layer by layer. That is the difference between the 8x cliff and a cache
     that just quietly becomes smaller.

WHAT THE CACHE IS WORTH, MEASURED ON THIS MACHINE
-------------------------------------------------
Reading a layer's blobs cold off the NVMe:            ~7.0 GB/s
Reading the same blobs back out of the page cache:   ~27-38 GB/s
So an in-RAM layer is ~5x cheaper, and the 53 GB spine's ~7.5 s/token of read
collapses toward ~1.5 s for whatever fraction is resident.  The fraction is what
matters: access is a strict 0..91 cycle, so under LRU a working set larger than
RAM gets a ~0% hit rate (each layer is evicted just before it is needed again).
Caching a FIXED SUBSET instead converts that into a hit rate equal to the
fraction that fits, which is why a partial cache is still worth having -- but
only if the layers outside the subset can be prevented from evicting it, which
is what K3_SPINE_STREAM_NOCACHE (see spine_io) is for.

FLAGS (defaults reproduce today's behaviour exactly: no cache at all)
    K3_SPINE_CACHE_GB=<n>       hard ceiling on the cache, 0 = disabled
    K3_SPINE_CACHE_AUTO=1       grow toward the ceiling only while the kernel
                                says there is room; without _GB, cap growth by
                                queried physical/recommended working-set limits
    K3_SPINE_CACHE_FLOOR_GB=<n> stop growing with less than this much `free`
                                (default 8)
    K3_SPINE_CACHE_TOL_GB=<n>   how much new compressor traffic to tolerate
                                before shrinking (default 0.5)
    K3_SPINE_CACHE_SHRINK=<n>   layers to drop per shrink event (default 4)

ON A 128 GB MAC set K3_SPINE_CACHE_AUTO=1 and leave the ceiling alone (or
K3_SPINE_CACHE_GB=44 K3_SPINE_CACHE_AUTO=1 to be explicit): 53 GB of spine plus
~12 GB of hot expert page cache plus the model's own working set fits with room
to spare, and the watchdog will still claw it back if something else on the box
wants memory.  On 64 GB, K3_SPINE_CACHE_GB=6 K3_SPINE_CACHE_AUTO=1 is the
conservative setting -- it caches the first ~10 layers and cannot reach the
pressure point.
"""
import ctypes
import ctypes.util
import os
import struct
import sys
import threading

import apple_silicon

try:
    import runtime_platform
except ImportError:  # imported as tools.spine_cache instead of top-level
    from . import runtime_platform

_IS_DARWIN = sys.platform == "darwin"
_IS_LINUX = sys.platform.startswith("linux")
try:
    # Windows has no sysconf at all, which is an AttributeError rather than the
    # OSError a POSIX host raises for an unknown name.
    PAGE = int(os.sysconf("SC_PAGE_SIZE"))
except (AttributeError, OSError, ValueError):
    PAGE = 16384 if _IS_DARWIN else 4096
_libc = None


def _lc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    return _libc


# ------------------------------------------------------------- vm probe -----
# struct vm_statistics64, arm64 layout -- VERIFIED FIELD BY FIELD against
# vm_stat(1) on this machine (every name below matched exactly; the offsets of
# compressor_pages / external / internal are one natural_t later than the
# published header suggests, which is why they are pinned by measurement):
#   off   0  4 x natural_t  free active inactive wire
#   off  16  9 x uint64     zero_fill reactivations pageins pageouts faults
#                           cow lookups hits purges
#   off  88  2 x natural_t  purgeable speculative
#   off  96  4 x uint64     decompressions compressions swapins swapouts
#   off 128  4 x natural_t  compressor_pages throttled external internal
#   off 144  1 x uint64     total_uncompressed_pages_in_compressor
# = 152 bytes = 38 integer_t = HOST_VM_INFO64_COUNT.
_VM_FMT = "<4I9Q2I4Q4IQ"
_VM_KEYS = ("free", "active", "inactive", "wire",
            "zero_fill", "reactivations", "pageins", "pageouts", "faults",
            "cow", "lookups", "hits", "purges",
            "purgeable", "speculative",
            "decompressions", "compressions", "swapins", "swapouts",
            "compressor_pages", "throttled", "external", "internal",
            "uncompressed_in_compressor")
HOST_VM_INFO64 = 4
HOST_VM_INFO64_COUNT = 38


def _darwin_vm_snapshot():
    """Darwin kernel page accounting, ~2 us; same source vm_stat(1) reads."""
    lc = _lc()
    lc.mach_host_self.restype = ctypes.c_uint
    buf = (ctypes.c_uint32 * 64)()
    cnt = ctypes.c_uint32(HOST_VM_INFO64_COUNT)
    if lc.host_statistics64(lc.mach_host_self(), HOST_VM_INFO64,
                            ctypes.byref(buf), ctypes.byref(cnt)) != 0:
        return None
    return dict(zip(_VM_KEYS, struct.unpack_from(_VM_FMT, bytes(buf), 0)))


def _proc_fields(path, colon=False):
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle
            for line in lines:
                if colon:
                    name, sep, rest = line.partition(":")
                    if not sep:
                        continue
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    name, rest = parts[0], parts[1]
                token = rest.strip().split()
                try:
                    out[name] = int(token[0]) if token else 0
                except ValueError:
                    continue
    except OSError:
        return {}
    return out


def _linux_vm_snapshot(
    meminfo_path="/proc/meminfo",
    vmstat_path="/proc/vmstat",
    available_cap_bytes=None,
):
    """Linux page accounting normalized to the Darwin watchdog's key schema."""
    mem = _proc_fields(meminfo_path, colon=True)
    vm = _proc_fields(vmstat_path)
    if not mem:
        return None
    out = dict.fromkeys(_VM_KEYS, 0)
    free_bytes = max(0, mem.get("MemFree", 0)) * 1024
    file_bytes = max(
        0,
        mem.get("Cached", 0)
        + mem.get("SReclaimable", 0)
        - mem.get("Shmem", 0),
    ) * 1024
    if available_cap_bytes is None:
        limits = runtime_platform.linux_memory_limits(
            meminfo_path=meminfo_path
        )
        available_cap_bytes = limits.effective_available_bytes
    if available_cap_bytes is not None:
        # Host page-cache totals may include memory outside this process's
        # cgroup. Never treat more than our effective headroom as reclaimable.
        available_cap_bytes = max(0, available_cap_bytes)
        free_bytes = min(free_bytes, available_cap_bytes)
        file_bytes = min(
            file_bytes, max(0, available_cap_bytes - free_bytes)
        )
    out["free"] = free_bytes // PAGE
    out["external"] = file_bytes // PAGE
    out["swapouts"] = max(0, vm.get("pswpout", 0))
    # Linux has no direct analogue of the macOS compressor. Swapout movement is
    # the conservative, actionable signal; pressure_level() remains neutral.
    out["compressions"] = 0
    return out


def vm_snapshot():
    """Return compatible VM counters on Darwin and Linux."""
    if _IS_DARWIN:
        return _darwin_vm_snapshot()
    if _IS_LINUX:
        return _linux_vm_snapshot()
    return None


_PRESSURE_MIB = None


def pressure_level():
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical.
    This is the same signal DISPATCH_SOURCE_TYPE_MEMORYPRESSURE delivers, minus
    the dispatch queue -- and unlike a dispatch source it can be sampled from a
    worker thread with no runloop, which is where our admissions happen."""
    if not _IS_DARWIN:
        return 1
    global _PRESSURE_MIB
    lc = _lc()
    out = ctypes.c_int(0)
    sz = ctypes.c_size_t(4)
    if _PRESSURE_MIB is None:
        name = b"kern.memorystatus_vm_pressure_level"
        mib = (ctypes.c_int * 8)()
        n = ctypes.c_size_t(8)
        if lc.sysctlnametomib(name, mib, ctypes.byref(n)) != 0:
            _PRESSURE_MIB = False
        else:
            _PRESSURE_MIB = (mib, int(n.value))
    if _PRESSURE_MIB is False:
        return 1
    mib, n = _PRESSURE_MIB
    if lc.sysctl(mib, n, ctypes.byref(out), ctypes.byref(sz), None, 0) != 0:
        return 1
    return int(out.value) or 1


# ------------------------------------------------------------ watchdog ------
OK, WARN, CRIT = 0, 1, 2


class Watchdog:
    """Turns the kernel's counters into grow / hold / shrink.

    THE SIGNAL IS A RATE, NOT A LEVEL, AND THE WINDOW SLIDES.
    `compressions` and `swapouts` are lifetime counters; on a machine that has
    been up for weeks they are already in the hundreds of millions, so only
    movement matters.  Movement *since process start* is no good either: cross
    the tolerance once -- which any busy box does within seconds, for reasons
    that have nothing to do with us -- and the cache would be latched off
    forever.  So the reference point (`mark`) is re-armed after every successful
    admission and every clean poll, which makes the question "did the compressor
    run in the interval since we last touched anything?".  That is as close to
    attribution as this API allows.

    Deliberately NOT triggers:
      * low `free`.  With a big clean page cache, near-zero free memory is the
        healthy steady state of this program, not a warning.  Low free gates
        GROWTH (see room_for) but must never cause a shrink.
      * memorystatus level 2 (warn).  It sits at 2 for long stretches on a box
        with a full page cache.  Only level 4 (critical) is unambiguous."""

    def __init__(self, floor_bytes, tol_bytes, file_reserve_bytes=8e9):
        self.floor = floor_bytes
        self.tol = tol_bytes
        self.file_reserve = file_reserve_bytes
        self.base = vm_snapshot()
        self.mark = self.base
        self.last = self.base
        self.reason = ""
        self.level = 1
        self.worst = OK

    def check(self):
        s = vm_snapshot()
        if s is None or self.mark is None:
            return OK
        self.last = s
        self.level = lvl = pressure_level()
        d_comp = (s["compressions"] - self.mark["compressions"]) * PAGE
        d_swap = (s["swapouts"] - self.mark["swapouts"]) * PAGE
        st, why = OK, ""
        if lvl >= 4:
            st, why = CRIT, "memorystatus=critical"
        elif d_swap > self.tol:
            st, why = CRIT, f"swapouts +{d_swap/1e9:.2f} GB in window"
        elif d_comp > self.tol:
            st, why = WARN, f"compressions +{d_comp/1e9:.2f} GB in window"
        self.reason = why
        self.worst = max(self.worst, st)
        return st

    def rearm(self):
        """Slide the window forward: everything up to now is water under the
        bridge, judge the next interval on its own."""
        self.mark = self.last or vm_snapshot()

    def room_for(self, nbytes):
        """Honest headroom = pages the kernel can hand us WITHOUT COMPRESSING
        ANYTHING, minus a reserve for the file cache we still want:

            free + purgeable + speculative + max(0, file_backed - file_reserve)

        The old budget used `free + inactive`.  `inactive` is a mix: clean file
        pages (free to reclaim) AND anonymous pages belonging to this and other
        processes (reclaimable only by compressing them).  Counting the
        anonymous part as spare is precisely what walked the machine into the
        compressor and turned a 14 s token into 122 s.  `external` (file-backed,
        what vm_stat calls "File-backed pages") is the part that is genuinely
        cheap, and even that is not free: it is the page cache the expert blobs
        live in, hence file_reserve.

        Note this is only the GROWTH gate.  The real protection is check(): if
        the estimate is wrong, `compressions` moves and we give memory back."""
        s = self.last or vm_snapshot()
        if s is None:
            return False
        cheap = s["free"] + s["purgeable"] + s["speculative"]
        filed = max(0.0, s["external"] * PAGE - self.file_reserve)
        return cheap * PAGE + filed - self.floor >= nbytes


# --------------------------------------------------------------- cache ------
def _env_f(name, default):
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


class SpineCache:
    """Holds whole read_pack() results (already-packed host buffers) keyed by
    layer prefix.  A hit skips the read AND the repack, so a cached layer costs
    only its host->device copy.

    Eviction safety: an entry is dropped from the dict BEFORE it is unpinned, so
    a buffer can never be both reachable from the cache and eligible for the
    read pool's recycler.  A pack that the main thread is applying right now can
    be evicted harmlessly -- apply_pack() only returns buffers to the pool after
    its last copy, and by then nothing will read them again."""

    def __init__(self, spine_fast, n_layers=0, log=print):
        self.sf = spine_fast
        self.log = log
        self.n_layers = n_layers
        self.floor = _env_f("K3_SPINE_CACHE_FLOOR_GB", 8.0) * 1e9
        self.tol = _env_f("K3_SPINE_CACHE_TOL_GB", 0.5) * 1e9
        self.file_reserve = _env_f("K3_SPINE_FILE_RESERVE_GB", 8.0) * 1e9
        self.shrink_n = int(_env_f("K3_SPINE_CACHE_SHRINK", 4))
        gb = os.environ.get("K3_SPINE_CACHE_GB")
        self.auto = os.environ.get("K3_SPINE_CACHE_AUTO", "0") == "1"
        if gb is not None and gb != "":
            requested_ceiling = max(0.0, float(gb) * 1e9)
            if _IS_LINUX:
                memory = runtime_platform.linux_memory_limits()
                safe_ceiling = runtime_platform.safe_linux_host_budget(
                    memory, int(self.floor + self.file_reserve)
                )
                self.ceiling = min(requested_ceiling, safe_ceiling)
                self.capability_fingerprint = (
                    f"linux:{memory.effective_total_bytes}:"
                    f"{memory.cgroup_limit_bytes or 0}"
                )
                if self.ceiling < requested_ceiling:
                    self.log(
                        f"[spine-cache] clamped requested "
                        f"{requested_ceiling/1e9:.1f} GB to "
                        f"{self.ceiling/1e9:.1f} GB for host/cgroup headroom"
                    )
            else:
                self.ceiling = requested_ceiling
                self.capability_fingerprint = None
        else:
            # AUTO is an explicit experiment opt-in, but its size still comes
            # from capabilities rather than a chip/RAM product-name table.
            if self.auto and _IS_LINUX:
                memory = runtime_platform.linux_memory_limits()
                self.ceiling = runtime_platform.safe_linux_host_budget(
                    memory, int(self.floor + self.file_reserve)
                )
                self.capability_fingerprint = (
                    f"linux:{memory.effective_total_bytes}:"
                    f"{memory.cgroup_limit_bytes or 0}"
                )
            else:
                caps = apple_silicon.snapshot()
                self.ceiling = (
                    caps.safe_unified_budget(
                        host_reserve_bytes=int(self.floor + self.file_reserve),
                        device_reserve_bytes=int(self.floor),
                    )
                    if self.auto
                    else 0.0
                )
                self.capability_fingerprint = (
                    caps.fingerprint() if self.auto else None
                )
        self.wd = (Watchdog(self.floor, self.tol, self.file_reserve)
                   if self.enabled else None)
        self.d = {}
        self.order = []                    # admission order, for LIFO shrink
        self.bytes = 0
        self.frozen = False                # latched: never grow again
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.shrinks = 0
        self.dropped = 0
        self.skipped = 0
        self.peak = 0
        self._msg = ""
        self._hdr_msg = False
        self._poll_n = 0

    @property
    def enabled(self):
        return self.ceiling > 0

    # -- lookup ---------------------------------------------------------------
    def get(self, prefix):
        if not self.enabled:
            return None
        p = self.d.get(prefix)
        if p is None:
            self.misses += 1
        else:
            self.hits += 1
        return p

    def cached(self, prefix):
        return prefix in self.d

    # -- growth ---------------------------------------------------------------
    def admit(self, prefix, pack, nbytes):
        """Called from the preloader thread after a cold read.  Returns True if
        the pack is now owned by the cache (its buffers are pinned)."""
        if not self.enabled or self.frozen or not nbytes:
            return False
        st = self.wd.check()
        if st != OK:
            self._pressure(st)
            return False
        with self.lock:
            if self.bytes + nbytes > self.ceiling:
                if not self._msg:
                    self._msg = "ceiling"
                    self.log(f"[spine-cache] ceiling reached: {len(self.d)}"
                             f"/{self.n_layers or '?'} layers, "
                             f"{self.bytes/1e9:.1f} GB")
                self.frozen = True          # a static ceiling never reopens
                return False
            if (self.auto or _IS_LINUX) and not self.wd.room_for(nbytes):
                # NOT a latch. Headroom is a moving target -- `free` collapses
                # while the page cache warms up and recovers as clean pages age
                # out -- so a transient dip must not disable the cache for the
                # rest of the run. Only real pressure (check() != OK) latches.
                self.skipped += 1
                if not self._hdr_msg:
                    self._hdr_msg = True
                    self.log(f"[spine-cache] pausing growth at {len(self.d)}"
                             f"/{self.n_layers or '?'} layers, "
                             f"{self.bytes/1e9:.1f} GB: no reclaim-free headroom "
                             f"(floor {self.floor/1e9:.0f} GB, file reserve "
                             f"{self.file_reserve/1e9:.0f} GB)")
                return False
            self.sf.pin(pack.get("q"), pack.get("sc"), pack.get("other"))
            self.d[prefix] = pack
            self.order.append(prefix)
            self.bytes += nbytes
            self.peak = max(self.peak, self.bytes)
        self.wd.rearm()      # judge the next admission on the next interval
        return True

    # -- shrink ---------------------------------------------------------------
    def _pressure(self, st):
        self.frozen = True          # latched: real pressure means stop growing
        n = self.shrink_n * (4 if st == CRIT else 1)
        if not self.d:
            return
        self.shrinks += 1
        freed = self.evict(len(self.d) if st == CRIT else n)
        self.log(f"[spine-cache] memory pressure ({self.wd.reason}): dropped "
                 f"{freed/1e9:.1f} GB, {len(self.d)} layers / "
                 f"{self.bytes/1e9:.1f} GB left")

    def evict(self, n):
        """Drop the n most recently admitted layers.  LIFO keeps the surviving
        set the LOW-indexed prefix of the layer walk, which is also the part the
        preloader has the least time to hide (nothing computes ahead of it)."""
        freed = 0
        with self.lock:
            for _ in range(min(n, len(self.order))):
                prefix = self.order.pop()
                pack = self.d.pop(prefix, None)
                if pack is None:
                    continue
                bufs = (pack.get("q"), pack.get("sc"), pack.get("other"))
                freed += sum(len(b) for b in bufs if b is not None)
                # dict first, then unpin: never reachable-and-recyclable at once
                self.sf.unpin(*bufs)
                self.dropped += 1
            self.bytes = max(0, self.bytes - freed)
        return freed

    POLL_EVERY = 8

    def poll(self):
        """Per-layer heartbeat.  Only meaningful once something is cached; it
        catches pressure that arrives AFTER the cache stopped growing (another
        process on the box, a long context growing the KV cache, MPS taking
        more device memory).  Sampled every POLL_EVERY layers -- the syscall is
        ~2 us but there is no point sampling faster than we can react."""
        if not self.enabled or not self.d:
            return
        self._poll_n += 1
        if self._poll_n % self.POLL_EVERY:
            return
        st = self.wd.check()
        if st != OK:
            self._pressure(st)
        self.wd.rearm()

    # -- reporting ------------------------------------------------------------
    def report(self):
        if not self.enabled:
            return ""
        w = self.wd
        d = ""
        if w and w.base and w.last:
            dc = (w.last["compressions"] - w.base["compressions"]) * PAGE / 1e9
            ds = (w.last["swapouts"] - w.base["swapouts"]) * PAGE / 1e9
            d = (f" | since start: compressions +{dc:.2f} GB swapouts +{ds:.2f} GB "
                 f"free {w.last['free']*PAGE/1e9:.1f} GB")
        return (f"[spine-cache] {len(self.d)} layers / {self.bytes/1e9:.1f} GB "
                f"(peak {self.peak/1e9:.1f}) hits {self.hits} miss {self.misses} "
                f"shrinks {self.shrinks} dropped {self.dropped} "
                f"skipped {self.skipped}{d}")


# --------------------------------------------- page-cache resident tier -----
def layer_bytes(int8_dir, pfx, n_layers):
    """On-disk bytes per layer index, from one stat sweep of the blob dir
    (~2,500 stats, ~30 ms, once per process)."""
    sizes = [0] * n_layers
    try:
        names = os.listdir(int8_dir)
    except OSError:
        return sizes
    head = pfx + "layers."
    for f in names:
        if not f.startswith(head):
            continue
        try:
            i = int(f[len(head):].split(".", 1)[0])
        except ValueError:
            continue
        if 0 <= i < n_layers:
            try:
                sizes[i] += os.path.getsize(os.path.join(int8_dir, f))
            except OSError:
                pass
    return sizes


def resident_tier(int8_dir, pfx, n_layers, budget_bytes, skip=()):
    """Choose the fixed subset of layers to keep page-cache resident: the lowest
    indices that fit in `budget_bytes`, skipping any already held in the explicit
    RAM cache.  Lowest-first because the preloader has the least compute to hide
    behind at the start of a pass -- layer 0's read is never overlapped with
    anything.  Returns (set_of_prefixes, chosen_bytes)."""
    if budget_bytes <= 0:
        return set(), 0
    sizes = layer_bytes(int8_dir, pfx, n_layers)
    out, used = set(), 0
    for i, nb in enumerate(sizes):
        p = f"{pfx}layers.{i}."
        if p in skip:
            continue
        if not nb or used + nb > budget_bytes:
            continue
        out.add(p)
        used += nb
    return out, used


def describe_env():
    keys = ("K3_SPINE_CACHE_GB", "K3_SPINE_CACHE_AUTO", "K3_SPINE_CACHE_FLOOR_GB",
            "K3_SPINE_CACHE_TOL_GB", "K3_SPINE_CACHE_SHRINK")
    return " ".join(f"{k.replace('K3_SPINE_CACHE_', '').lower()}="
                    f"{os.environ.get(k, '-')}" for k in keys)
