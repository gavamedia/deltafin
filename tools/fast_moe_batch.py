"""Batched MoE expert path — drop-in replacement for tools/fast_moe.py.

fast_moe.py issues one ctypes call per GEMV: 48 per MoE layer (16 experts x w1/w3/w2),
and each of those spins up + tears down a fresh pthread pool (4 create + 4 join), i.e.
~192 thread create/destroy per layer per token.

This module talks to the platform's libmxfp4batch native library
(tools/fused_gemv_batch.c), which keeps one persistent worker pool alive and
work-steals rows across a whole batch of matrices. Per layer that is 2
dispatches (w1+w3 phase, then w2 phase) instead of 48, with zero thread
creation after the first call.

Bit-exact with fast_moe.moe_infer_fast: the same mxfp4_gemv_rows2() row loop runs on
the same 32-row-aligned partitions, the activation stays in numpy, and the per-token
combine accumulates in the same order.

Interface mirrors fast_moe: expert_ffn(raw, x), moe_infer_fast(x, ids, w, raw_experts).
"""
import ctypes
import os
import platform
import threading
import time

import numpy as np
import torch

try:
    import routing_record as routing_records
    from runtime_platform import (
        available_cpu_count,
        choose_gemv_autotune_width,
        configured_cpu_workers,
        default_gemv_threads,
        gemv_autotune_candidates,
        load_native_library,
    )
except ImportError:  # imported as tools.fast_moe_batch instead of top-level
    from . import routing_record as routing_records
    from .runtime_platform import (
        available_cpu_count,
        choose_gemv_autotune_width,
        configured_cpu_workers,
        default_gemv_threads,
        gemv_autotune_candidates,
        load_native_library,
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
_REQUIRED_SYMBOLS = (
    "mxfp4_gemv_batch",
    "mxfp4_moe_layer",
    "mxfp4_situ_batch",
    "mxfp4_moe_expert_set",
    "mxfp4_pool_init",
    "mxfp4_pool_threads",
    "mxfp4_pool_shutdown",
)
_LIB, _LIB_PATH = load_native_library(
    _HERE,
    "libmxfp4batch",
    env_var="K3_BATCH_LIB",
    required_symbols=_REQUIRED_SYMBOLS,
)

_u8p = np.ctypeslib.ndpointer(np.uintp, flags="C_CONTIGUOUS")   # array of raw pointers
_i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")

_LIB.mxfp4_gemv_batch.argtypes = [_u8p, _u8p, _u8p, _u8p, _i32, _i32,
                                  ctypes.c_int, ctypes.c_int]
_LIB.mxfp4_gemv_batch.restype = None
_LIB.mxfp4_moe_layer.argtypes = [_u8p, _u8p, _i32, _i32, ctypes.c_int,
                                 _f32, _f32, ctypes.c_int]
_LIB.mxfp4_moe_layer.restype = None
_LIB.mxfp4_situ_batch.argtypes = [_f32, _f32, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LIB.mxfp4_situ_batch.restype = None
_LIB.mxfp4_moe_expert_set.argtypes = [_u8p, _u8p, _u8p, _u8p, _f32, _f32,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, _f32, ctypes.c_int]
_LIB.mxfp4_moe_expert_set.restype = None
_LIB.mxfp4_pool_init.argtypes = [ctypes.c_int]
_LIB.mxfp4_pool_init.restype = ctypes.c_int
_LIB.mxfp4_pool_threads.argtypes = []
_LIB.mxfp4_pool_threads.restype = ctypes.c_int
_LIB.mxfp4_pool_shutdown.argtypes = []
_LIB.mxfp4_pool_shutdown.restype = None
_HAVE_AVX2 = getattr(_LIB, "mxfp4_have_avx2", None)
if _HAVE_AVX2 is not None:
    _HAVE_AVX2.argtypes = []
    _HAVE_AVX2.restype = ctypes.c_int

THREADS = configured_cpu_workers("K3_GEMV_THREADS", default_gemv_threads())
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0

_POOL_READY = False
_CALL_LOCK = threading.RLock()
_ACTIVE_THREADS = THREADS
_AUTOTUNE_STATE = "unconfigured"
_AUTOTUNE_REASON = "runner has not selected the CPU MoE backend"
_AUTOTUNE_RESULT = {}
_AUTOTUNE_LOAD_LIMIT = 0.40
_AUTOTUNE_DEADLINE_NS = 350_000_000
_AUTOTUNE_MIN_SPEEDUP = 1.05


def native_isa():
    """Describe the selected native CPU kernel without requiring a new-library ABI."""
    if _HAVE_AVX2 is not None and _HAVE_AVX2():
        return "avx2/fma"
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "neon"
    if machine in {"x86_64", "amd64"}:
        return "ssse3/fma"
    return machine or "unknown"


def _requested_autotune():
    value = os.environ.get("K3_GEMV_AUTOTUNE", "0").strip().lower()
    if value in ("0", "off", "false", "no", ""):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    raise ValueError(
        "K3_GEMV_AUTOTUNE must be 0/1, off/on, false/true, or no/yes"
    )


def configure_autotune(cpu_batch_backend):
    """Arm C23 only when the persistent CPU backend is the selected backend.

    The runner calls this after resolving CPU/Metal/CUDA. An emergency CPU
    replay after a CUDA failure deliberately keeps the conservative configured
    width rather than adding calibration work to an error-recovery path.
    """
    global _ACTIVE_THREADS, _AUTOTUNE_STATE, _AUTOTUNE_REASON
    global _AUTOTUNE_RESULT
    with _CALL_LOCK:
        try:
            observed_threads = int(pool_threads())
        except (TypeError, ValueError, RuntimeError):
            observed_threads = 0
        _ACTIVE_THREADS = (
            observed_threads if observed_threads > 0 else THREADS
        )
        _AUTOTUNE_RESULT = {}
        if "K3_GEMV_THREADS" in os.environ:
            _AUTOTUNE_STATE = "disabled"
            _AUTOTUNE_REASON = "K3_GEMV_THREADS is explicit"
        elif not cpu_batch_backend:
            _AUTOTUNE_STATE = "disabled"
            _AUTOTUNE_REASON = "persistent CPU MoE is not the selected backend"
        elif not _requested_autotune():
            _AUTOTUNE_STATE = "disabled"
            _AUTOTUNE_REASON = "K3_GEMV_AUTOTUNE is off"
        elif observed_threads > 0 and observed_threads != THREADS:
            _AUTOTUNE_STATE = "disabled"
            _AUTOTUNE_REASON = (
                f"configured pool requested {THREADS} workers but created "
                f"{observed_threads}"
            )
        else:
            _AUTOTUNE_STATE = "pending"
            _AUTOTUNE_REASON = "waiting for the first complete top-16 expert set"
        return autotune_status()


def autotune_status():
    with _CALL_LOCK:
        return {
            "state": _AUTOTUNE_STATE,
            "reason": _AUTOTUNE_REASON,
            "configured_threads": THREADS,
            "active_threads": _ACTIVE_THREADS,
            **_AUTOTUNE_RESULT,
        }


def _normalized_load():
    try:
        return os.getloadavg()[0] / available_cpu_count()
    except (AttributeError, OSError):
        return None


def pool_init(nthreads=None):
    """Create the persistent worker pool up front (otherwise done lazily in C)."""
    global _POOL_READY
    with _CALL_LOCK:
        n = _LIB.mxfp4_pool_init(
            _ACTIVE_THREADS if nthreads is None else nthreads
        )
        _POOL_READY = True
        return n


def pool_shutdown():
    global _POOL_READY
    with _CALL_LOCK:
        _LIB.mxfp4_pool_shutdown()
        _POOL_READY = False


def pool_threads():
    with _CALL_LOCK:
        return _LIB.mxfp4_pool_threads()


# --------------------------------------------------------------- scratch (reused)
class _Scratch:
    __slots__ = ("n", "pp", "sp", "xp", "yp", "rows", "cols")

    def __init__(self, n):
        self.n = 0
        self.grow(n)

    def grow(self, n):
        if n <= self.n:
            return
        self.pp = np.zeros(n, dtype=np.uintp)
        self.sp = np.zeros(n, dtype=np.uintp)
        self.xp = np.zeros(n, dtype=np.uintp)
        self.yp = np.zeros(n, dtype=np.uintp)
        self.rows = np.zeros(n, dtype=np.int32)
        self.cols = np.zeros(n, dtype=np.int32)
        self.n = n


_S = _Scratch(64)


def _addr(a):
    """Raw data pointer of a C-contiguous numpy array (cheaper than .ctypes.data)."""
    return a.__array_interface__["data"][0]


def gemv_batch(mats, nthreads=None):
    """mats: list of (packed, scale, x, y). Runs every GEMV in ONE dispatch.

    packed U8 [rows, cols/2], scale U8 [rows, cols/32], x f32 [cols], y f32 [rows];
    all C-contiguous. y is written in place.
    """
    n = len(mats)
    if n == 0:
        return
    # CDLL calls release the GIL. Keep the shared pointer/shape arena stable
    # until every native worker has finished consuming it.
    with _CALL_LOCK:
        _S.grow(n)
        pp, sp = _S.pp, _S.sp
        xp, yp, rows, cols = _S.xp, _S.yp, _S.rows, _S.cols
        for i, (p, s, x, y) in enumerate(mats):
            pp[i] = _addr(p)
            sp[i] = _addr(s)
            xp[i] = _addr(x)
            yp[i] = _addr(y)
            rows[i] = p.shape[0]
            cols[i] = p.shape[1] * 2
        _LIB.mxfp4_gemv_batch(
            pp, sp, xp, yp, rows, cols, n,
            _ACTIVE_THREADS if nthreads is None else nthreads,
        )


def _situ(gate, up):
    """Identical expression to fast_moe._situ (fp32 throughout)."""
    a = SITU_BETA * np.tanh(gate / SITU_BETA) / (1.0 + np.exp(-gate))
    return a * (SITU_LINEAR_BETA * np.tanh(up / SITU_LINEAR_BETA))


def _situ_inplace(gate, up, scratch):
    """Evaluate SiTU in place with the exact numpy expression order.

    ``gate`` and ``up`` are disposable GEMV outputs.  ``scratch`` is equally
    sized storage whose previous contents are disposable.  The returned array
    aliases ``gate``; this removes the activation's full-size result and
    transient arrays without changing the sequence of fp32 ufuncs.
    """
    # Form the numerator first, exactly like the historical expression, while
    # retaining the original gate values for the denominator.
    np.divide(gate, SITU_BETA, out=scratch)
    np.tanh(scratch, out=scratch)
    np.multiply(scratch, SITU_BETA, out=scratch)

    np.negative(gate, out=gate)
    np.exp(gate, out=gate)
    np.add(gate, 1.0, out=gate)
    np.divide(scratch, gate, out=gate)

    np.divide(up, SITU_LINEAR_BETA, out=up)
    np.tanh(up, out=up)
    np.multiply(up, SITU_LINEAR_BETA, out=up)
    np.multiply(gate, up, out=gate)
    return gate


# --------------------------------------------------------------- expert set (bit-exact)
def _phase_a(raws, x, gu, nthreads):
    mats = []
    for e, raw in enumerate(raws):
        packed1, scales1 = raw["w1"]
        packed3, scales3 = raw["w3"]
        mats.append((packed1, scales1, x, gu[0, e]))
        mats.append((packed3, scales3, x, gu[1, e]))
    gemv_batch(mats, nthreads)


def _phase_b(raws, h, yb, nthreads):
    mats = []
    for e, raw in enumerate(raws):
        packed2, scales2 = raw["w2"]
        mats.append((packed2, scales2, h[e], yb[e]))
    gemv_batch(mats, nthreads)


def _expert_set_ffn_fixed(raws, x, nthreads):
    """Established two-dispatch path at one explicit or active width."""
    ne = len(raws)
    if ne == 0:
        return np.zeros((0, x.shape[0]), dtype=np.float32)
    d_ff = raws[0]["w1"][0].shape[0]
    d_model = raws[0]["w2"][0].shape[0]

    # [2, ne, d_ff]: all gates contiguous, all ups contiguous, so the numpy activation
    # takes the same contiguous SIMD path fast_moe._situ does (bit-exactness).
    gu = np.empty((2, ne, d_ff), dtype=np.float32)
    _phase_a(raws, x, gu, nthreads)

    # Allocate the phase-B destination before activation.  K3's d_model is
    # wider than d_ff, so its unused prefix is a contiguous activation
    # scratch arena.  Keep a shape-generic fallback for other checkpoints.
    yb = np.empty((ne, d_model), dtype=np.float32)
    gate = gu[0]
    scratch_elems = gate.size
    if yb.size >= scratch_elems:
        scratch = yb.reshape(-1)[:scratch_elems].reshape(gate.shape)
    else:
        scratch = np.empty_like(gate)
    h = _situ_inplace(gate, gu[1], scratch)

    _phase_b(raws, h, yb, nthreads)
    return yb


def _autotune_eligible(raws, x):
    """Restrict calibration to the actual complete K3 top-16 CPU shape."""
    if (
        len(raws) != 16
        or not isinstance(x, np.ndarray)
        or x.dtype != np.float32
        or x.ndim != 1
        or not x.flags.c_contiguous
    ):
        return False
    d_model = x.shape[0]
    try:
        d_ff = raws[0]["w1"][0].shape[0]
        if (
            d_model != 3584
            or d_ff != 3072
            or raws[0]["w2"][0].shape[0] != d_model
        ):
            return False
        owners = set()
        for raw in raws:
            p1, _s1 = raw["w1"]
            p2, _s2 = raw["w2"]
            p3, _s3 = raw["w3"]
            if (
                p1.shape != (d_ff, d_model // 2)
                or p2.shape != (d_model, d_ff // 2)
                or p3.shape != (d_ff, d_model // 2)
            ):
                return False
            owners.add(_addr(p1))
        return len(owners) == len(raws)
    except (KeyError, TypeError, AttributeError):
        return False


def _report_autotune(message):
    print(f"[cpu-moe] GEMV autotune {message}", flush=True)


def _restore_configured_threads():
    global _ACTIVE_THREADS
    actual = None
    try:
        actual = int(pool_init(THREADS))
    except Exception:
        # Never adopt the last candidate merely because it still appears in
        # pool_threads(): it may be the width whose output just drifted. Tear
        # the ring down and qualify the safest possible request instead.
        try:
            pool_shutdown()
        except Exception:
            pass
        try:
            actual = int(pool_init(1))
        except Exception:
            actual = None
    # A short positive pool is still a valid exact backend. Retain what was
    # actually created instead of requesting the unattainable width on every
    # later dispatch and repeatedly tearing the pool down. If construction
    # failed or created no workers, requesting one later makes the C backend
    # retry and otherwise use its exact inline fallback.
    _ACTIVE_THREADS = actual if actual is not None and actual > 0 else 1
    return actual


def _disable_autotune(reason, *, result=None, report=True):
    global _AUTOTUNE_STATE, _AUTOTUNE_REASON, _AUTOTUNE_RESULT
    _AUTOTUNE_STATE = "disabled"
    _AUTOTUNE_REASON = reason
    _AUTOTUNE_RESULT = {} if result is None else result
    restored_threads = _restore_configured_threads()
    restored = restored_threads == THREADS
    if not restored:
        _AUTOTUNE_RESULT = {
            **_AUTOTUNE_RESULT,
            "configured_pool_restore_succeeded": False,
            "restored_pool_threads": restored_threads,
        }
    if report:
        if restored:
            suffix = f"using {THREADS} workers"
        elif restored_threads is not None and restored_threads > 0:
            suffix = (
                f"configured width unavailable; using the actual "
                f"{restored_threads}-worker pool"
            )
        else:
            suffix = (
                "worker creation unavailable; next dispatch will request the "
                "exact one-worker/inline fallback"
            )
        _report_autotune(f"skipped ({reason}); {suffix}")


def _maybe_autotune_expert_set(raws, x):
    """Run the bounded C23 selector once and return its exact first output."""
    global _ACTIVE_THREADS, _AUTOTUNE_STATE, _AUTOTUNE_REASON
    global _AUTOTUNE_RESULT
    if _AUTOTUNE_STATE != "pending":
        return None
    if not _autotune_eligible(raws, x):
        with _CALL_LOCK:
            if _AUTOTUNE_STATE == "pending":
                _disable_autotune(
                    "first CPU expert set is not the distinct top-16 K3 shape"
                )
        return None

    # Hold the same reentrant lock used by every dispatch and pool rebuild.
    # ctypes releases the GIL, so the outer lock is what makes the complete
    # multi-width transaction invisible to another inference caller.
    with _CALL_LOCK:
        if _AUTOTUNE_STATE != "pending":
            return None
        try:
            load_before = _normalized_load()
        except Exception as exc:
            _disable_autotune(f"{type(exc).__name__}: {exc}")
            return None
        if load_before is None or load_before > _AUTOTUNE_LOAD_LIMIT:
            _disable_autotune(
                "host load is unknown or above "
                f"{_AUTOTUNE_LOAD_LIMIT:.2f} per effective CPU",
                result={"normalized_load_before": load_before},
            )
            return None

        configured = THREADS
        try:
            effective = available_cpu_count()
            candidates = gemv_autotune_candidates(effective, configured)
        except Exception as exc:
            _disable_autotune(
                f"{type(exc).__name__}: {exc}",
                result={"normalized_load_before": load_before},
            )
            return None
        if configured not in candidates:
            _disable_autotune(
                "effective CPU ceiling changed after startup; restart the "
                "process or set K3_GEMV_THREADS explicitly",
                result={
                    "effective_cpus": effective,
                    "candidates": list(candidates),
                    "normalized_load_before": load_before,
                },
            )
            return None
        if len(candidates) < 2:
            _disable_autotune(
                "no alternative worker width is available",
                result={
                    "effective_cpus": effective,
                    "candidates": list(candidates),
                },
            )
            return None

        _AUTOTUNE_STATE = "running"
        _AUTOTUNE_REASON = "calibration in progress"
        started = time.perf_counter_ns()
        samples = {width: [] for width in candidates}
        phase_a_samples = {width: [] for width in candidates}
        phase_b_samples = {width: [] for width in candidates}
        reference_output = None

        try:
            ne = len(raws)
            d_ff = raws[0]["w1"][0].shape[0]
            d_model = raws[0]["w2"][0].shape[0]
            gu = np.empty((2, ne, d_ff), dtype=np.float32)
            yb = np.empty((ne, d_model), dtype=np.float32)
        except Exception as exc:
            _disable_autotune(
                f"{type(exc).__name__}: {exc}",
                result={
                    "effective_cpus": effective,
                    "candidates": list(candidates),
                    "normalized_load_before": load_before,
                },
            )
            return None

        def run_pair(width, reference_gu, reference_yb, h):
            actual = pool_init(width)
            if actual != width:
                raise RuntimeError(
                    f"requested {width} workers, native pool created {actual}"
                )

            # Rebuilding the persistent pool changes first-dispatch latency.
            # Prime it once outside the score so widths are compared in the
            # steady state in which the selected pool will actually run.
            _phase_a(raws, x, gu, width)
            if reference_gu is None:
                warm_gu = gu.view(np.uint32).copy()
                scratch = np.empty_like(gu[0])
                h = _situ_inplace(gu[0], gu[1], scratch).copy()
            elif not np.array_equal(gu.view(np.uint32), reference_gu):
                raise RuntimeError(
                    f"phase-A output differs at {width} workers during warm-up"
                )

            phase_started = time.perf_counter_ns()
            _phase_a(raws, x, gu, width)
            phase_a_ns = time.perf_counter_ns() - phase_started
            expected_gu = warm_gu if reference_gu is None else reference_gu
            if not np.array_equal(gu.view(np.uint32), expected_gu):
                raise RuntimeError(
                    f"phase-A output differs at {width} workers"
                )
            phase_a_bits = (
                gu.view(np.uint32).copy()
                if reference_gu is None else None
            )
            phase_started = time.perf_counter_ns()
            _phase_b(raws, h, yb, width)
            phase_b_ns = time.perf_counter_ns() - phase_started
            if reference_yb is not None and not np.array_equal(
                yb.view(np.uint32), reference_yb
            ):
                raise RuntimeError(
                    f"phase-B output differs at {width} workers"
                )
            return phase_a_ns, phase_b_ns, h, phase_a_bits

        try:
            # The configured-width call is both the exact fallback/reference
            # and trial zero's required default sample.
            phase_a_ns, phase_b_ns, h, reference_gu = run_pair(
                configured, None, None, None
            )
            reference_yb = yb.view(np.uint32).copy()
            reference_output = yb.copy()
            phase_a_samples[configured].append(phase_a_ns)
            phase_b_samples[configured].append(phase_b_ns)
            samples[configured].append(phase_a_ns + phase_b_ns)

            base = list(candidates)
            pivot = base.index(configured)
            base = base[pivot:] + base[:pivot]
            orders = (base[1:], list(reversed(base)))
            for order in orders:
                for width in order:
                    if time.perf_counter_ns() - started > _AUTOTUNE_DEADLINE_NS:
                        raise TimeoutError(
                            f"{_AUTOTUNE_DEADLINE_NS / 1e6:.0f} ms "
                            "cooperative calibration deadline expired"
                        )
                    phase_a_ns, phase_b_ns, _same_h, _phase_a_bits = run_pair(
                        width, reference_gu, reference_yb, h
                    )
                    phase_a_samples[width].append(phase_a_ns)
                    phase_b_samples[width].append(phase_b_ns)
                    samples[width].append(phase_a_ns + phase_b_ns)

            winner, reason = choose_gemv_autotune_width(
                samples,
                configured,
                minimum_speedup=_AUTOTUNE_MIN_SPEEDUP,
            )
            if winner is not None:
                actual = pool_init(winner)
                if actual != winner:
                    raise RuntimeError(
                        f"winning pool requested {winner}, created {actual}"
                    )
                # Installation can rebuild the ring once more. Prime and
                # qualify that exact physical pool before publishing it for
                # the next layer; this dispatch is charged to the deadline.
                _phase_a(raws, x, gu, winner)
                if not np.array_equal(gu.view(np.uint32), reference_gu):
                    raise RuntimeError(
                        "phase-A output differs after installing winning "
                        f"{winner}-worker pool"
                    )

            load_after = _normalized_load()
            elapsed_ns = time.perf_counter_ns() - started
            if elapsed_ns > _AUTOTUNE_DEADLINE_NS:
                raise TimeoutError(
                    f"{_AUTOTUNE_DEADLINE_NS / 1e6:.0f} ms cooperative "
                    "calibration deadline expired"
                )
            if load_after is None or load_after > _AUTOTUNE_LOAD_LIMIT:
                raise RuntimeError(
                    "post-calibration host load is unknown or above "
                    f"{_AUTOTUNE_LOAD_LIMIT:.2f} per effective CPU"
                )

            result = {
                "effective_cpus": effective,
                "candidates": list(candidates),
                "samples_ms": {
                    str(width): [value / 1e6 for value in values]
                    for width, values in samples.items()
                },
                "phase_a_samples_ms": {
                    str(width): [value / 1e6 for value in values]
                    for width, values in phase_a_samples.items()
                },
                "phase_b_samples_ms": {
                    str(width): [value / 1e6 for value in values]
                    for width, values in phase_b_samples.items()
                },
                "normalized_load_before": load_before,
                "normalized_load_after": load_after,
                "calibration_wall_ms": elapsed_ns / 1e6,
                "all_outputs_bit_exact": True,
                "winner_install_phase_a_bit_exact": winner is not None,
            }
            if winner is None:
                _disable_autotune(reason, result=result)
                return reference_output

            _ACTIVE_THREADS = winner
            _AUTOTUNE_STATE = "selected"
            _AUTOTUNE_REASON = reason
            _AUTOTUNE_RESULT = {
                **result,
                "selected_threads": winner,
                "trial_speedups_vs_configured": [
                    samples[configured][trial] / samples[winner][trial]
                    for trial in range(len(samples[winner]))
                ],
            }
            if winner == configured:
                _report_autotune(
                    f"kept {configured} workers ({reason}, "
                    f"{elapsed_ns / 1e6:.1f} ms)"
                )
            else:
                speedups = _AUTOTUNE_RESULT[
                    "trial_speedups_vs_configured"
                ]
                _report_autotune(
                    f"selected {winner} workers over {configured} "
                    f"({speedups[0]:.3f}x/{speedups[1]:.3f}x, "
                    f"{elapsed_ns / 1e6:.1f} ms)"
                )
            # The last phase-B trial is bit-identical to reference_yb, so it is
            # the real first-call result; no final GEMV replay is needed.
            return yb
        except Exception as exc:
            _disable_autotune(
                f"{type(exc).__name__}: {exc}",
                result={
                    "effective_cpus": effective,
                    "candidates": list(candidates),
                    "normalized_load_before": load_before,
                    "calibration_wall_ms": (
                        time.perf_counter_ns() - started
                    ) / 1e6,
                },
            )
            return reference_output


def expert_set_ffn(raws, x, nthreads=None):
    """raws: list of {w1|w2|w3: (packed, scale)}; x: fp32 [d_model] contiguous.
    Returns fp32 [n_ex, d_model] — row e is expert raws[e] applied to x.
    Two dispatches total, regardless of how many experts are in the set."""
    if nthreads is None:
        tuned = _maybe_autotune_expert_set(raws, x)
        if tuned is not None:
            return tuned
    return _expert_set_ffn_fixed(raws, x, nthreads)


def expert_ffn(raw, x, nthreads=None):
    """Single-expert compatibility shim (same signature as fast_moe.expert_ffn)."""
    return expert_set_ffn([raw], x, nthreads)[0]


def moe_infer_fast(
        x, topk_ids, topk_weight, raw_experts, nthreads=None,
        routing_record=None):
    """x: torch [N, d_model] fp32; returns torch [N, d_model] fp32.
    Same contract and same numerics as fast_moe.moe_infer_fast.
    ``routing_record`` is optional and follows ``nthreads`` so existing
    positional callers remain compatible."""
    xnp = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
    N = xnp.shape[0]
    out = np.zeros((N, xnp.shape[1]), dtype=np.float32)
    ids, ws = routing_records.resolve(
        topk_ids, topk_weight, routing_record)
    for t in range(N):
        xt = np.ascontiguousarray(xnp[t])
        sel = ids[t]
        yb = expert_set_ffn([raw_experts[e] for e in sel], xt, nthreads)
        for i, w in enumerate(ws[t]):
            # ``yb`` is dead after this combine. Scale each row in place so
            # top-k routing does not allocate one d_model temporary per expert.
            np.multiply(yb[i], np.float32(w), out=yb[i])
            np.add(out[t], yb[i], out=out[t])
    return torch.from_numpy(out).to(x.device, x.dtype)


# --------------------------------------------------------------- one-call variant (opt-in)
def moe_infer_fused(
        x, topk_ids, topk_weight, raw_experts, nthreads=None,
        routing_record=None):
    """Same result, but SiTU and the weighted combine run inside C: ONE ctypes call
    per token per layer instead of two. NOT bit-exact vs numpy — libm tanhf/expf differ
    from numpy's vectorized transcendentals by ~1 ulp. Kept behind its own name so the
    bit-exact path stays the default. The optional shared route follows
    ``nthreads`` to preserve legacy positional calls."""
    xnp = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
    N, d_model = xnp.shape
    out = np.zeros((N, d_model), dtype=np.float32)
    ids, ws = routing_records.resolve(
        topk_ids, topk_weight, routing_record)
    for t in range(N):
        xt = np.ascontiguousarray(xnp[t])
        sel = ids[t]
        ne = len(sel)
        d_ff = raw_experts[sel[0]]["w1"][0].shape[0]
        p13 = np.zeros(2 * ne, dtype=np.uintp)
        s13 = np.zeros(2 * ne, dtype=np.uintp)
        p2 = np.zeros(ne, dtype=np.uintp)
        s2 = np.zeros(ne, dtype=np.uintp)
        for i, e in enumerate(sel):
            r = raw_experts[e]
            p13[2 * i], s13[2 * i] = _addr(r["w1"][0]), _addr(r["w1"][1])
            p13[2 * i + 1], s13[2 * i + 1] = _addr(r["w3"][0]), _addr(r["w3"][1])
            p2[i], s2[i] = _addr(r["w2"][0]), _addr(r["w2"][1])
        wts = np.ascontiguousarray(ws[t], dtype=np.float32)
        yt = np.empty(d_model, dtype=np.float32)
        with _CALL_LOCK:
            nt = _ACTIVE_THREADS if nthreads is None else nthreads
            _LIB.mxfp4_moe_expert_set(
                p13, s13, p2, s2, xt, wts,
                ne, d_ff, d_model, None, yt, nt,
            )
        out[t] = yt
    return torch.from_numpy(out).to(x.device, x.dtype)
