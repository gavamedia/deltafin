"""Batched MoE expert path — drop-in replacement for tools/fast_moe.py.

fast_moe.py issues one ctypes call per GEMV: 48 per MoE layer (16 experts x w1/w3/w2),
and each of those spins up + tears down a fresh pthread pool (4 create + 4 join), i.e.
~192 thread create/destroy per layer per token.

This module talks to libmxfp4batch.dylib (tools/fused_gemv_batch.c), which keeps ONE
persistent QOS_CLASS_USER_INTERACTIVE worker pool alive and work-steals rows across a
whole batch of matrices. Per layer that is 2 dispatches (w1+w3 phase, then w2 phase)
instead of 48, with zero thread creation after the first call.

Bit-exact with fast_moe.moe_infer_fast: the same mxfp4_gemv_rows2() row loop runs on
the same 32-row-aligned partitions, the activation stays in numpy, and the per-token
combine accumulates in the same order.

Interface mirrors fast_moe: expert_ffn(raw, x), moe_infer_fast(x, ids, w, raw_experts).
"""
import ctypes, os, sys
import numpy as np
import torch

from mxfp4 import dequant_mxfp4

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBPATH = os.environ.get("K3_BATCH_LIB") or os.path.join(
    _HERE, "libmxfp4batch" + (".dylib" if sys.platform == "darwin" else ".so"))

_LIB = None
try:
    _LIB = ctypes.CDLL(_LIBPATH)
except OSError as e:
    print(f"[fast_moe_batch] SLOW PATH: batch kernel unavailable ({e}); using "
          f"the pure-numpy mxfp4 reference. Build it with the command in "
          f"tools/fused_gemv_batch.c.", file=sys.stderr, flush=True)

if _LIB is not None:
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
    print(f"[fast_moe_batch] batched MXFP4 kernel active: {_LIBPATH}",
          file=sys.stderr, flush=True)

KERNEL = "fused" if _LIB is not None else "numpy"
THREADS = int(os.environ.get("K3_GEMV_THREADS", "4"))
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0

_POOL_READY = False


def pool_init(nthreads=None):
    """Create the persistent worker pool up front (otherwise done lazily in C)."""
    global _POOL_READY
    if _LIB is None:
        _POOL_READY = True
        return 0
    n = _LIB.mxfp4_pool_init(THREADS if nthreads is None else nthreads)
    _POOL_READY = True
    return n


def pool_shutdown():
    global _POOL_READY
    if _LIB is not None:
        _LIB.mxfp4_pool_shutdown()
    _POOL_READY = False


def pool_threads():
    return _LIB.mxfp4_pool_threads() if _LIB is not None else 0


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
    if _LIB is None:                       # pure-numpy fallback (announced on import)
        for p, s, x, y in mats:
            y[:] = dequant_mxfp4(p, s) @ x
        return
    _S.grow(n)
    pp, sp, xp, yp, rows, cols = _S.pp, _S.sp, _S.xp, _S.yp, _S.rows, _S.cols
    for i, (p, s, x, y) in enumerate(mats):
        pp[i] = _addr(p)
        sp[i] = _addr(s)
        xp[i] = _addr(x)
        yp[i] = _addr(y)
        rows[i] = p.shape[0]
        cols[i] = p.shape[1] * 2
    _LIB.mxfp4_gemv_batch(pp, sp, xp, yp, rows, cols, n,
                          THREADS if nthreads is None else nthreads)


def _situ(gate, up):
    """Identical expression to fast_moe._situ (fp32 throughout)."""
    a = SITU_BETA * np.tanh(gate / SITU_BETA) / (1.0 + np.exp(-gate))
    return a * (SITU_LINEAR_BETA * np.tanh(up / SITU_LINEAR_BETA))


# --------------------------------------------------------------- expert set (bit-exact)
def expert_set_ffn(raws, x, nthreads=None):
    """raws: list of {w1|w2|w3: (packed, scale)}; x: fp32 [d_model] contiguous.
    Returns fp32 [n_ex, d_model] — row e is expert raws[e] applied to x.
    Two dispatches total, regardless of how many experts are in the set."""
    ne = len(raws)
    if ne == 0:
        return np.zeros((0, x.shape[0]), dtype=np.float32)
    d_ff = raws[0]["w1"][0].shape[0]
    d_model = raws[0]["w2"][0].shape[0]

    # [2, ne, d_ff]: all gates contiguous, all ups contiguous, so the numpy activation
    # takes the same contiguous SIMD path fast_moe._situ does (bit-exactness).
    gu = np.empty((2, ne, d_ff), dtype=np.float32)
    mats = []
    for e, r in enumerate(raws):
        p1, s1 = r["w1"]
        p3, s3 = r["w3"]
        mats.append((p1, s1, x, gu[0, e]))
        mats.append((p3, s3, x, gu[1, e]))
    gemv_batch(mats, nthreads)

    h = np.ascontiguousarray(_situ(gu[0], gu[1]), dtype=np.float32)

    yb = np.empty((ne, d_model), dtype=np.float32)
    mats = []
    for e, r in enumerate(raws):
        p2, s2 = r["w2"]
        mats.append((p2, s2, h[e], yb[e]))
    gemv_batch(mats, nthreads)
    return yb


def expert_ffn(raw, x, nthreads=None):
    """Single-expert compatibility shim (same signature as fast_moe.expert_ffn)."""
    return expert_set_ffn([raw], x, nthreads)[0]


def moe_infer_fast(x, topk_ids, topk_weight, raw_experts, nthreads=None):
    """x: torch [N, d_model] fp32; returns torch [N, d_model] fp32.
    Same contract and same numerics as fast_moe.moe_infer_fast."""
    xnp = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
    N = xnp.shape[0]
    out = np.zeros((N, xnp.shape[1]), dtype=np.float32)
    ids = topk_ids.tolist()
    ws = topk_weight.to(torch.float32).tolist()
    for t in range(N):
        xt = np.ascontiguousarray(xnp[t])
        sel = ids[t]
        yb = expert_set_ffn([raw_experts[e] for e in sel], xt, nthreads)
        for i, w in enumerate(ws[t]):
            out[t] += np.float32(w) * yb[i]
    return torch.from_numpy(out).to(x.device, x.dtype)


# --------------------------------------------------------------- one-call variant (opt-in)
def moe_infer_fused(x, topk_ids, topk_weight, raw_experts, nthreads=None):
    """Same result, but SiTU and the weighted combine run inside C: ONE ctypes call
    per token per layer instead of two. NOT bit-exact vs numpy — libm tanhf/expf differ
    from numpy's vectorized transcendentals by ~1 ulp. Kept behind its own name so the
    bit-exact path stays the default."""
    if _LIB is None:                       # no C SiTU without the lib
        return moe_infer_fast(x, topk_ids, topk_weight, raw_experts, nthreads)
    xnp = np.ascontiguousarray(x.detach().to("cpu", torch.float32).numpy())
    N, d_model = xnp.shape
    out = np.zeros((N, d_model), dtype=np.float32)
    ids = topk_ids.tolist()
    ws = topk_weight.to(torch.float32).tolist()
    nt = THREADS if nthreads is None else nthreads
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
        _LIB.mxfp4_moe_expert_set(p13, s13, p2, s2, xt, wts,
                                  ne, d_ff, d_model, None, yt, nt)
        out[t] = yt
    return torch.from_numpy(out).to(x.device, x.dtype)
