#!/usr/bin/env python3
"""Synthetic, weight-free correctness gate for the native MXFP4 kernels.

The test builds the current local C sources into a temporary directory, then checks:

* every E2M1 code and every E8M0 scale byte through SIMD table expansion;
* the production scale lookup versus the original synthesis implementation;
* selected, explicit compatibility, and explicit AVX2 paths bit-for-bit;
* single-thread, multi-thread, and over-limit thread requests bit-for-bit;
* the persistent batch pool and its distinct-activation preparation count;
* the fixed-size expert-triple launcher above its supported thread limit; and
* the ABI version exported by both native libraries.

It needs no model download or expert cache:

    python tools/test_fused_gemv_portability.py
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mxfp4 import dequant_mxfp4


U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)


def _build_msvc(
    source: str,
    output: Path,
    *,
    extra_cflags: tuple[str, ...] = (),
) -> None:
    """Build one kernel with MSVC, matching what build_native.py installs.

    The GNU flags above have no cl.exe spelling, so the dialect is translated
    here rather than approximated: /arch:AVX is the -mavx baseline, and the
    -D defines the callers pass become /D.
    """
    import build_native as native

    environment = native.msvc_environment()
    compiler = shutil.which(
        "cl",
        path=native.environment_path(environment) if environment else None,
    )
    if compiler is None:
        raise RuntimeError("cl.exe not found; install the VS C++ build tools")
    scratch = output.parent
    defines = [f"/D{flag[2:]}" for flag in extra_cflags]
    if any(not flag.startswith("-D") for flag in extra_cflags):
        raise RuntimeError(f"cannot translate {extra_cflags!r} for MSVC")
    subprocess.run(
        [
            compiler,
            "/nologo",
            "/O2",
            "/std:c11",
            "/experimental:c11atomics",
            "/arch:AVX",
            *defines,
            "/DNO_MAIN",
            "/LD",
            str(HERE / source),
            f"/Fe:{output}",
            f"/Fo:{os.path.join(str(scratch), '')}",
            "/link",
            "/INCREMENTAL:NO",
            f"/IMPLIB:{scratch / (output.stem + '.lib')}",
        ],
        check=True,
        cwd=HERE,
        env=dict(environment) if environment else None,
    )


def _build(
    source: str,
    output: Path,
    *,
    extra_cflags: tuple[str, ...] = (),
) -> None:
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64", "x86_64", "amd64"}:
        raise RuntimeError(f"unsupported test architecture: {machine}")

    if os.name == "nt":
        _build_msvc(source, output, extra_cflags=extra_cflags)
        return

    cc = os.environ.get("CC", "cc")
    cmd = [
        cc,
        "-O3",
        "-std=gnu11",
        *extra_cflags,
        "-DNO_MAIN",
        "-shared",
        str(HERE / source),
        "-o",
        str(output),
        "-lpthread",
        "-lm",
    ]
    if machine in {"arm64", "aarch64"}:
        cmd[1:1] = ["-mcpu=native"]
    else:
        cmd[1:1] = [
            "-march=x86-64",
            "-mtune=x86-64" if sys.platform == "darwin" else "-mtune=native",
            "-msse3",
            "-mssse3",
            "-mavx",
            "-mfma",
        ]
    if sys.platform != "darwin":
        cmd[1:1] = ["-fPIC"]
    subprocess.run(cmd, check=True, cwd=HERE)


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.mxfp4_abi_version.argtypes = []
    lib.mxfp4_abi_version.restype = ctypes.c_uint32
    if lib.mxfp4_abi_version() != 1:
        raise AssertionError(f"{path.name}: expected ABI 1")
    return lib


def _bind_gemv(lib: ctypes.CDLL) -> None:
    single_args = [U8P, U8P, F32P, F32P, ctypes.c_int, ctypes.c_int]
    mt_args = [*single_args, ctypes.c_int]
    for name in ("mxfp4_gemv", "mxfp4_gemv_compat"):
        fn = getattr(lib, name)
        fn.argtypes = single_args
        fn.restype = None
    for name in ("mxfp4_gemv_mt", "mxfp4_gemv_mt_compat"):
        fn = getattr(lib, name)
        fn.argtypes = mt_args
        fn.restype = None
    lib.mxfp4_have_avx2.argtypes = []
    lib.mxfp4_have_avx2.restype = ctypes.c_int
    if lib.mxfp4_have_avx2():
        for name in ("mxfp4_gemv_avx2", "mxfp4_gemv_mt_avx2"):
            fn = getattr(lib, name)
            fn.argtypes = mt_args if name.endswith("_mt_avx2") else single_args
            fn.restype = None


def _gemv(lib: ctypes.CDLL, packed: np.ndarray, scales: np.ndarray, x: np.ndarray,
          nthreads: int | None = None, *, implementation: str = "selected",
          allow_nonfinite: bool = False) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.uint8)
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = packed.shape[0], packed.shape[1] * 2
    assert cols % 32 == 0
    assert scales.shape == (rows, cols // 32)
    assert x.shape == (cols,)
    out = np.full(rows, np.nan, dtype=np.float32)
    args = (
        packed.ctypes.data_as(U8P),
        scales.ctypes.data_as(U8P),
        x.ctypes.data_as(F32P),
        out.ctypes.data_as(F32P),
        rows,
        cols,
    )
    names = {
        ("selected", False): "mxfp4_gemv",
        ("selected", True): "mxfp4_gemv_mt",
        ("compat", False): "mxfp4_gemv_compat",
        ("compat", True): "mxfp4_gemv_mt_compat",
        ("avx2", False): "mxfp4_gemv_avx2",
        ("avx2", True): "mxfp4_gemv_mt_avx2",
    }
    name = names[(implementation, nthreads is not None)]
    fn = getattr(lib, name)
    if nthreads is None:
        fn(*args)
    else:
        fn(*args, nthreads)
    if not allow_nonfinite and np.isnan(out).any():
        raise AssertionError("GEMV left output rows unwritten")
    return out


def _bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def _test_single_and_mt(lib: ctypes.CDLL) -> None:
    e2m1 = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    codes = np.arange(16, dtype=np.uint8)
    packed = np.repeat((codes | (codes << 4))[:, None], 16, axis=1)
    scales = np.full((16, 1), 127, dtype=np.uint8)
    x = np.zeros(32, dtype=np.float32)
    x[0] = 1.0
    got = _gemv(lib, packed, scales, x)
    np.testing.assert_array_equal(got, e2m1)

    rng = np.random.default_rng(20260729)
    packed = rng.integers(0, 256, size=(19, 48), dtype=np.uint8)
    scales = rng.integers(120, 132, size=(19, 3), dtype=np.uint8)
    x = rng.standard_normal(96, dtype=np.float32)
    single = _gemv(lib, packed, scales, x)
    compat = _gemv(lib, packed, scales, x, implementation="compat")
    if not np.array_equal(_bits(single), _bits(compat)):
        raise AssertionError("selected single GEMV differs from compatibility path")
    if lib.mxfp4_have_avx2():
        avx2 = _gemv(lib, packed, scales, x, implementation="avx2")
        if not np.array_equal(_bits(avx2), _bits(compat)):
            raise AssertionError("explicit AVX2 GEMV differs from compatibility path")
    oracle = dequant_mxfp4(packed, scales).astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(single, oracle, rtol=2e-5, atol=2e-5)

    for nthreads in (1, 2, 3, 5, 16, 17, 64):
        threaded = _gemv(lib, packed, scales, x, nthreads=nthreads)
        if not np.array_equal(_bits(threaded), _bits(single)):
            raise AssertionError(f"nthreads={nthreads} differs from single GEMV")
        compat_mt = _gemv(
            lib, packed, scales, x,
            nthreads=nthreads, implementation="compat",
        )
        if not np.array_equal(_bits(compat_mt), _bits(single)):
            raise AssertionError(
                f"compat nthreads={nthreads} differs from single GEMV"
            )
        if lib.mxfp4_have_avx2():
            avx2_mt = _gemv(
                lib, packed, scales, x,
                nthreads=nthreads, implementation="avx2",
            )
            if not np.array_equal(_bits(avx2_mt), _bits(single)):
                raise AssertionError(
                    f"AVX2 nthreads={nthreads} differs from single GEMV"
                )


def _test_scale_lut_equivalence(
    lut: ctypes.CDLL,
    synthesis: ctypes.CDLL,
) -> None:
    """Exercise all 256 scale bytes, including wraparound edge cases."""
    rows, cols = 257, 96
    groups = cols // 32
    byte_codes = np.array(
        [low | ((15 - low) << 4) for low in range(16)],
        dtype=np.uint8,
    )
    packed = np.tile(byte_codes, rows * groups).reshape(rows, cols // 2)
    scales = np.arange(rows * groups, dtype=np.uint16)
    scales = np.ascontiguousarray((scales % 256).astype(np.uint8).reshape(rows, groups))
    rng = np.random.default_rng(0xE8A0)
    x = np.ascontiguousarray(rng.standard_normal(cols, dtype=np.float32))

    implementations = ["selected", "compat"]
    if lut.mxfp4_have_avx2() and synthesis.mxfp4_have_avx2():
        implementations.append("avx2")
    for implementation in implementations:
        for nthreads in (None, 1, 3, 16):
            expected = _gemv(
                synthesis,
                packed,
                scales,
                x,
                nthreads=nthreads,
                implementation=implementation,
                allow_nonfinite=True,
            )
            got = _gemv(
                lut,
                packed,
                scales,
                x,
                nthreads=nthreads,
                implementation=implementation,
                allow_nonfinite=True,
            )
            if not np.array_equal(_bits(got), _bits(expected)):
                mode = "single" if nthreads is None else f"mt({nthreads})"
                raise AssertionError(
                    f"scale LUT differs from synthesis in {implementation} {mode}"
                )


def _test_batch(lib: ctypes.CDLL, rng: np.random.Generator) -> None:
    _bind_gemv(lib)
    lib.mxfp4_pool_init.argtypes = [ctypes.c_int]
    lib.mxfp4_pool_init.restype = ctypes.c_int
    lib.mxfp4_pool_shutdown.argtypes = []
    lib.mxfp4_pool_shutdown.restype = None
    lib.mxfp4_gemv_batch.argtypes = [
        ctypes.POINTER(U8P),
        ctypes.POINTER(U8P),
        ctypes.POINTER(F32P),
        ctypes.POINTER(F32P),
        I32P,
        I32P,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mxfp4_gemv_batch.restype = None
    lib.mxfp4_batch_last_x_permutations.argtypes = []
    lib.mxfp4_batch_last_x_permutations.restype = ctypes.c_int

    specs = ((3, 96), (9, 96), (17, 96), (5, 64))
    packed = [
        np.ascontiguousarray(rng.integers(0, 256, size=(r, c // 2), dtype=np.uint8))
        for r, c in specs
    ]
    scales = [
        np.ascontiguousarray(rng.integers(121, 131, size=(r, c // 32), dtype=np.uint8))
        for r, c in specs
    ]
    shared_x = np.ascontiguousarray(rng.standard_normal(96, dtype=np.float32))
    xs = [
        shared_x,
        shared_x,
        np.ascontiguousarray(rng.standard_normal(96, dtype=np.float32)),
        np.ascontiguousarray(rng.standard_normal(64, dtype=np.float32)),
    ]
    refs = [_gemv(lib, p, s, x) for p, s, x in zip(packed, scales, xs)]
    outs = [np.full(r, np.nan, dtype=np.float32) for r, _ in specs]
    n = len(specs)
    pp = (U8P * n)(*(a.ctypes.data_as(U8P) for a in packed))
    sp = (U8P * n)(*(a.ctypes.data_as(U8P) for a in scales))
    xp = (F32P * n)(*(a.ctypes.data_as(F32P) for a in xs))
    yp = (F32P * n)(*(a.ctypes.data_as(F32P) for a in outs))
    rows = np.ascontiguousarray([r for r, _ in specs], dtype=np.int32)
    cols = np.ascontiguousarray([c for _, c in specs], dtype=np.int32)

    # 99 deliberately exceeds K3_MAX_THREADS; the public API must clamp it.
    for _ in range(2):
        for out in outs:
            out.fill(np.nan)
        lib.mxfp4_gemv_batch(
            pp, sp, xp, yp,
            rows.ctypes.data_as(I32P),
            cols.ctypes.data_as(I32P),
            n,
            99,
        )
        for i, (got, ref) in enumerate(zip(outs, refs)):
            if np.isnan(got).any() or not np.array_equal(_bits(got), _bits(ref)):
                raise AssertionError(f"batch matrix {i} differs from normal GEMV")
        expected_permutations = 3 if lib.mxfp4_have_avx2() else 0
        observed_permutations = lib.mxfp4_batch_last_x_permutations()
        if observed_permutations != expected_permutations:
            raise AssertionError(
                "batch prepared "
                f"{observed_permutations} activations; "
                f"expected {expected_permutations}"
            )
    lib.mxfp4_pool_shutdown()


def _batch_result(
    lib: ctypes.CDLL,
    packed: list[np.ndarray],
    scales: list[np.ndarray],
    xs: list[np.ndarray],
    nthreads: int,
) -> tuple[bytes, ...]:
    _bind_gemv(lib)
    lib.mxfp4_gemv_batch.argtypes = [
        ctypes.POINTER(U8P),
        ctypes.POINTER(U8P),
        ctypes.POINTER(F32P),
        ctypes.POINTER(F32P),
        I32P,
        I32P,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mxfp4_gemv_batch.restype = None
    lib.mxfp4_pool_shutdown.argtypes = []
    lib.mxfp4_pool_shutdown.restype = None

    n = len(packed)
    rows = np.ascontiguousarray([p.shape[0] for p in packed], dtype=np.int32)
    cols = np.ascontiguousarray([p.shape[1] * 2 for p in packed], dtype=np.int32)
    outs = [np.full(int(r), np.nan, dtype=np.float32) for r in rows]
    pp = (U8P * n)(*(a.ctypes.data_as(U8P) for a in packed))
    sp = (U8P * n)(*(a.ctypes.data_as(U8P) for a in scales))
    xp = (F32P * n)(*(a.ctypes.data_as(F32P) for a in xs))
    yp = (F32P * n)(*(a.ctypes.data_as(F32P) for a in outs))
    lib.mxfp4_gemv_batch(
        pp,
        sp,
        xp,
        yp,
        rows.ctypes.data_as(I32P),
        cols.ctypes.data_as(I32P),
        n,
        nthreads,
    )
    result = tuple(bytes(out) for out in outs)
    lib.mxfp4_pool_shutdown()
    return result


def _test_scale_lut_batch_equivalence(
    lut: ctypes.CDLL,
    synthesis: ctypes.CDLL,
) -> None:
    """Cover odd rows/groups and every scale byte in the persistent pool."""
    rng = np.random.default_rng(0x8A10)
    specs = ((257, 96), (33, 352), (19, 32))
    packed = [
        np.ascontiguousarray(
            rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        )
        for rows, cols in specs
    ]
    scales: list[np.ndarray] = []
    offset = 0
    for rows, cols in specs:
        count = rows * (cols // 32)
        values = (np.arange(offset, offset + count, dtype=np.uint32) % 256)
        scales.append(
            np.ascontiguousarray(values.astype(np.uint8).reshape(rows, cols // 32))
        )
        offset += count
    xs = [
        np.ascontiguousarray(rng.standard_normal(cols, dtype=np.float32))
        for _, cols in specs
    ]
    for nthreads in (1, 4, 32):
        expected = _batch_result(synthesis, packed, scales, xs, nthreads)
        got = _batch_result(lut, packed, scales, xs, nthreads)
        if got != expected:
            raise AssertionError(
                f"scale LUT persistent batch differs from synthesis at {nthreads} threads"
            )


def _test_expert_thread_bound(lib: ctypes.CDLL, rng: np.random.Generator) -> None:
    lib.mxfp4_expert_triple.argtypes = [
        U8P, U8P, U8P, U8P, U8P, U8P,
        F32P, F32P, F32P, F32P, F32P,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.mxfp4_expert_triple.restype = ctypes.c_double

    def matrix(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
        p = np.ascontiguousarray(
            rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        )
        s = np.ascontiguousarray(
            rng.integers(121, 130, size=(rows, cols // 32), dtype=np.uint8)
        )
        return p, s

    p1, s1 = matrix(3072, 3584)
    p3, s3 = matrix(3072, 3584)
    p2, s2 = matrix(3584, 3072)
    x = np.ascontiguousarray(rng.standard_normal(3584, dtype=np.float32))
    h = np.ascontiguousarray(rng.standard_normal(3072, dtype=np.float32))

    def run(
        nthreads: int,
        iters: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y1 = np.full(3072, np.nan, dtype=np.float32)
        y3 = np.full(3072, np.nan, dtype=np.float32)
        y2 = np.full(3584, np.nan, dtype=np.float32)
        lib.mxfp4_expert_triple(
            p1.ctypes.data_as(U8P), s1.ctypes.data_as(U8P),
            p3.ctypes.data_as(U8P), s3.ctypes.data_as(U8P),
            p2.ctypes.data_as(U8P), s2.ctypes.data_as(U8P),
            x.ctypes.data_as(F32P), h.ctypes.data_as(F32P),
            y1.ctypes.data_as(F32P), y3.ctypes.data_as(F32P), y2.ctypes.data_as(F32P),
            nthreads, iters,
        )
        if any(np.isnan(y).any() for y in (y1, y3, y2)):
            raise AssertionError("expert triple left output rows unwritten")
        return y1, y3, y2

    reference = run(4)
    over_limit = run(64)
    for name, got, expected in zip(("w1", "w3", "w2"), over_limit, reference):
        if not np.array_equal(_bits(got), _bits(expected)):
            raise AssertionError(f"expert triple {name} differs above thread limit")


def _test_partial_pthread_create(
    lib: ctypes.CDLL,
    rng: np.random.Generator,
) -> None:
    """Exercise both recovery protocols after exactly two workers start."""
    _bind_gemv(lib)
    reset = lib.mxfp4_test_reset_pthread_create
    reset.argtypes = []
    reset.restype = None

    packed = np.ascontiguousarray(
        rng.integers(0, 256, size=(19, 48), dtype=np.uint8))
    scales = np.ascontiguousarray(
        rng.integers(120, 132, size=(19, 3), dtype=np.uint8))
    x = np.ascontiguousarray(rng.standard_normal(96, dtype=np.float32))
    expected = _gemv(lib, packed, scales, x)
    reset()
    got = _gemv(lib, packed, scales, x, nthreads=5)
    if not np.array_equal(_bits(got), _bits(expected)):
        raise AssertionError("partial pthread_create GEMV recovery differs")

    # A fresh counter lets mxfp4_expert_triple start two of four requested
    # workers. Its start latch must reduce every barrier to that live width.
    reset()
    _test_expert_thread_bound(lib, rng)


def _release(*libraries: ctypes.CDLL) -> None:
    """Unload test libraries so the temporary directory can be removed.

    Windows keeps a loaded module's file locked; on POSIX this is a no-op.
    """
    import build_native as native

    for library in libraries:
        native.release_library(library)


def _run_portability_suite() -> None:
    if sys.platform == "darwin":
        suffix = ".dylib"
    elif os.name == "nt":
        suffix = ".dll"
    else:
        suffix = ".so"
    with tempfile.TemporaryDirectory(prefix="deltafin-kernel-test-") as td:
        tmp = Path(td)
        gemv_path = tmp / f"libmxfp4gemv{suffix}"
        batch_path = tmp / f"libmxfp4batch{suffix}"
        synth_gemv_path = tmp / f"libmxfp4gemv-synthesis{suffix}"
        synth_batch_path = tmp / f"libmxfp4batch-synthesis{suffix}"
        fail_path = tmp / f"libmxfp4gemv-pthread-fail{suffix}"
        _build("fused_gemv.c", gemv_path)
        _build("fused_gemv_batch.c", batch_path)
        _build(
            "fused_gemv.c",
            synth_gemv_path,
            extra_cflags=("-DMXFP4_USE_SCALE_LUT=0",),
        )
        _build(
            "fused_gemv_batch.c",
            synth_batch_path,
            extra_cflags=("-DMXFP4_USE_SCALE_LUT=0",),
        )
        _build(
            "fused_gemv.c",
            fail_path,
            extra_cflags=("-DMXFP4_TEST_PTHREAD_FAIL_AFTER=2",),
        )
        gemv = _load(gemv_path)
        batch = _load(batch_path)
        synth_gemv = _load(synth_gemv_path)
        synth_batch = _load(synth_batch_path)
        fail = _load(fail_path)
        try:
            _bind_gemv(gemv)
            _bind_gemv(synth_gemv)
            _test_single_and_mt(gemv)
            _test_scale_lut_equivalence(gemv, synth_gemv)
            rng = np.random.default_rng(0xD37AF1)
            _test_batch(batch, rng)
            _test_scale_lut_batch_equivalence(batch, synth_batch)
            _test_expert_thread_bound(gemv, rng)
            _test_partial_pthread_create(fail, rng)
        finally:
            # The pool owns live worker threads; stop them before unloading.
            batch.mxfp4_pool_shutdown()
            synth_batch.mxfp4_pool_shutdown()
            _release(gemv, batch, synth_gemv, synth_batch, fail)


class TestFusedGemvPortability(unittest.TestCase):
    def test_native_kernel_suite(self):
        _run_portability_suite()


def main() -> int:
    _run_portability_suite()

    print(
        f"PASS: {platform.system()} {platform.machine()} "
        "ABI=1, all E2M1/E8M0 codes, exact scale LUT, selected/compat GEMV, "
        "batch activation reuse, "
        "thread bounds, and pthread failure recovery"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
