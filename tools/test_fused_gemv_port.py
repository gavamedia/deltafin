#!/usr/bin/env python3
"""Synthetic validation of the fused MXFP4 GEMV kernel against the pure-numpy
reference tools/mxfp4.py:dequant_mxfp4. No model weights required, so this is
the acceptance gate for the x86-64 port (and still passes on Apple Silicon).

  [1] DEQUANT EXACTNESS. One-hot GEMVs read every dequantized weight back out of
      the kernel and compare it to dequant_mxfp4 with `==` (exact fp32 equality;
      a -0.0 weight necessarily surfaces as +0.0 through the dot product). The
      sweep covers all 16 e2m1 codes (both zeros, all negatives) in both nibble
      positions x scale bytes over the trick's full validity domain s in [2,252]
      (result fp32 exponent 1..254), including the denormal-adjacent edge
      (0.5 * 2^-125 = 2^-126) and the overflow-adjacent edge (6 * 2^125).
  [2] GEMV ACCURACY. Randomized matrices vs float64 dequant@x, error asserted
      against an a-priori fp32 accumulation bound and reported as max abs/rel.
  [3] INTERNAL BIT-CONSISTENCY. mxfp4_gemv (1-row) vs mxfp4_gemv_mt (2-row,
      1/2/3/5 threads) vs libmxfp4batch's work-stolen gemv_batch must agree
      bit-for-bit, including odd row counts (rows2 tail) and cols=32 minimum.

Note on the domain: e8m0 scale bytes outside [2,252] can push the fp32 exponent
field out of [1,254], which the kernel's integer exponent-add documents as out
of contract (real checkpoint data sits in 112..122).
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mxfp4 import dequant_mxfp4  # noqa: E402

import fast_moe  # noqa: E402
import fast_moe_batch as fmb  # noqa: E402

if fast_moe._LIB is None or fmb._LIB is None:
    print("SKIP: compiled kernels not present (build commands in tools/fused_gemv*.c)")
    sys.exit(1)

_LIB = fast_moe._LIB
_u8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_LIB.mxfp4_gemv.argtypes = [_u8, _u8, _f32, _f32, ctypes.c_int, ctypes.c_int]
_LIB.mxfp4_gemv.restype = None

rng = np.random.default_rng(20260728)
fails = 0


def gemv1(p, s, x):
    y = np.empty(p.shape[0], dtype=np.float32)
    _LIB.mxfp4_gemv(p, s, x, y, p.shape[0], p.shape[1] * 2)
    return y


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


# ---- [1] dequant exactness via one-hot GEMVs ---------------------------------
SCALES = [2, 3, 20, 60, 100, 112, 117, 122, 126, 127, 128, 130, 200, 250, 252]
R, C = len(SCALES), 512                       # 256 bytes/row = every (lo,hi) code pair
p1 = np.tile(np.arange(256, dtype=np.uint8), (R, 1))
s1 = np.repeat(np.array(SCALES, dtype=np.uint8)[:, None], C // 32, axis=1)
s1 = np.ascontiguousarray(s1)
W_ref = dequant_mxfp4(p1, s1)                 # fp32 [R, C]
W_got = np.empty((R, C), dtype=np.float32)
x1 = np.zeros(C, dtype=np.float32)
for c in range(C):
    x1[c] = 1.0
    W_got[:, c] = gemv1(p1, s1, x1)
    x1[c] = 0.0
exact = np.all(W_got == W_ref)                # value-exact; -0.0 == 0.0 by design
nz = W_ref != 0
bit = np.array_equal(bits(W_got[nz]), bits(W_ref[nz]))
print(f"[1] dequant one-hot sweep ({R}x{C}, scales {SCALES[0]}..{SCALES[-1]}): "
      f"{'EXACT' if exact else 'DIFFERS'} "
      f"({'bit-exact on all nonzeros' if bit else 'nonzero bit mismatch'})")
fails += not (exact and bit)

# ---- [2] randomized GEMVs vs float64 reference -------------------------------
worst_rel = worst_abs = worst_ulp = 0.0
CASES = [(1, 32), (2, 32), (3, 64), (5, 96), (33, 320), (37, 512),
         (64, 3584), (3072, 3584), (3584, 3072)]
for rows, cols in CASES:
    p = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    s = rng.integers(100, 140, (rows, cols // 32), dtype=np.uint8)
    # sprinkle wide-dynamic-range scales into a few groups. Capped at 200
    # (2^73): a full ROW of scale-252 (2^125) weights is exact per element
    # (test [1]) but its fp32 row SUM overflows to inf, which is an fp32 range
    # limit shared by every fp32 GEMV, not a kernel property.
    s.flat[rng.integers(0, s.size, max(1, s.size // 16))] = rng.choice(
        [2, 20, 150, 200], max(1, s.size // 16))
    x = rng.standard_normal(cols).astype(np.float32)
    y = gemv1(p, s, x)
    W64 = dequant_mxfp4(p, s).astype(np.float64)
    ref64 = W64 @ x.astype(np.float64)
    err = np.abs(y - ref64)
    # a-priori bound: cols/32 sequential FMAs per lane + 9 reduction adds,
    # each rounding relative to the running magnitude <= sum |w||x|
    mag = np.abs(W64) @ np.abs(x).astype(np.float64)
    bound = np.finfo(np.float32).eps * (cols / 32 + 10) * np.maximum(mag, 1e-30)
    ok = bool(np.all(err <= bound))
    rel = float(np.max(err / np.maximum(np.abs(ref64), 1e-6)))
    with np.errstate(over="ignore", invalid="ignore"):
        ref32 = ref64.astype(np.float32)
        ulp = float(np.max(np.abs(y - ref32) / np.spacing(np.abs(ref32).astype(np.float32))))
    worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, float(err.max()))
    worst_ulp = max(worst_ulp, ulp)
    print(f"[2] {rows:5d}x{cols:<5d} vs fp64: max_abs={err.max():.3e} "
          f"max_rel={rel:.3e} max_ulp32={ulp:.1f} bound={'OK' if ok else 'EXCEEDED'}")
    fails += not ok
print(f"[2] worst over all shapes: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} "
      f"max_ulp32={worst_ulp:.1f}")

# ---- [3] internal bit-consistency (1r vs 2r/mt vs batch, odd tails) ----------
for rows, cols in [(1, 32), (5, 64), (37, 512), (127, 3584), (3072, 3584)]:
    p = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
    s = rng.integers(100, 140, (rows, cols // 32), dtype=np.uint8)
    x = rng.standard_normal(cols).astype(np.float32)
    ref = gemv1(p, s, x)
    ok = True
    for nt in (1, 2, 3, 5):
        y = np.empty(rows, dtype=np.float32)
        fast_moe._LIB.mxfp4_gemv_mt(p, s, x, y, rows, cols, nt)
        ok &= np.array_equal(bits(ref), bits(y))
    yb = np.empty(rows, dtype=np.float32)
    fmb.gemv_batch([(p, s, x, yb)])
    ok &= np.array_equal(bits(ref), bits(yb))
    print(f"[3] {rows:5d}x{cols:<5d} 1r vs mt(1,2,3,5) vs batch: "
          f"{'BIT-EXACT' if ok else 'DIFFERS'}")
    fails += not ok
fmb.pool_shutdown()

print("\nRESULT:", "PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
