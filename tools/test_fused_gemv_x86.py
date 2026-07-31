#!/usr/bin/env python3
"""Minimal NumPy-free exactness gate for Deltafin's x86-64 MXFP4 kernels.

This test is intentionally usable with macOS's system Python under Rosetta:

    arch -x86_64 /usr/bin/python3 tools/test_fused_gemv_x86.py

It also runs directly on Linux x86-64.  The established SSSE3/FMA compatibility
kernel is the bitwise oracle for the selected AVX2 path; the architecture-neutral
NumPy reference is covered by test_fused_gemv_portability.py.
"""

from __future__ import annotations

import ctypes
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
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

    /arch:AVX is the -mavx analogue used by the GNU command below: a VEX
    baseline that still leaves AVX2 to the kernel's runtime selection.
    """
    import build_native as native

    if any(not flag.startswith("-D") for flag in extra_cflags):
        raise RuntimeError(f"cannot translate {extra_cflags!r} for MSVC")
    environment = native.msvc_environment()
    compiler = shutil.which(
        "cl",
        path=native.environment_path(environment) if environment else None,
    )
    if compiler is None:
        raise RuntimeError("cl.exe not found; install the VS C++ build tools")
    scratch = output.parent
    subprocess.run(
        [
            compiler,
            "/nologo",
            "/O2",
            "/std:c11",
            "/experimental:c11atomics",
            "/arch:AVX",
            *(f"/D{flag[2:]}" for flag in extra_cflags),
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
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("this exactness gate must execute as x86-64")
    if os.name == "nt":
        _build_msvc(source, output, extra_cflags=extra_cflags)
        return
    cc = os.environ.get("CC", "cc")
    cmd = [
        cc,
        "-O3",
        "-std=gnu11",
        "-march=x86-64",
        "-mtune=x86-64" if sys.platform == "darwin" else "-mtune=native",
        "-msse3",
        "-mssse3",
        "-mavx",
        "-mfma",
        *extra_cflags,
        "-DNO_MAIN",
    ]
    if sys.platform == "darwin":
        cmd.extend(("-arch", "x86_64", "-dynamiclib"))
    else:
        cmd.extend(("-fPIC", "-shared"))
    cmd.extend((
        str(HERE / source),
        "-o",
        str(output),
        "-lpthread",
        "-lm",
    ))
    subprocess.run(cmd, cwd=HERE, check=True)


def _bind(
    lib: ctypes.CDLL, *, expected_avx2: int | None = None
) -> int:
    single = [U8P, U8P, F32P, F32P, ctypes.c_int, ctypes.c_int]
    threaded = [*single, ctypes.c_int]
    for name in (
        "mxfp4_gemv",
        "mxfp4_gemv_compat",
        "mxfp4_gemv_avx2",
    ):
        fn = getattr(lib, name)
        fn.argtypes = single
        fn.restype = None
    for name in (
        "mxfp4_gemv_mt",
        "mxfp4_gemv_mt_compat",
        "mxfp4_gemv_mt_avx2",
    ):
        fn = getattr(lib, name)
        fn.argtypes = threaded
        fn.restype = None
    lib.mxfp4_have_avx2.argtypes = []
    lib.mxfp4_have_avx2.restype = ctypes.c_int
    observed = lib.mxfp4_have_avx2()
    if expected_avx2 is not None and observed != expected_avx2:
        raise AssertionError(
            f"runtime dispatch reported AVX2={observed}, expected {expected_avx2}"
        )
    return observed


def _matrix(
    rng: random.Random,
    rows: int,
    cols: int,
) -> tuple[ctypes.Array, ctypes.Array]:
    packed = (ctypes.c_uint8 * (rows * cols // 2))(
        *(rng.randrange(256) for _ in range(rows * cols // 2))
    )
    scales = (ctypes.c_uint8 * (rows * cols // 32))(
        *(rng.randrange(119, 133) for _ in range(rows * cols // 32))
    )
    return packed, scales


def _activation(rng: random.Random, cols: int) -> ctypes.Array:
    return (ctypes.c_float * cols)(
        *(ctypes.c_float(rng.uniform(-2.0, 2.0)).value for _ in range(cols))
    )


def _run(
    lib: ctypes.CDLL,
    name: str,
    packed: ctypes.Array,
    scales: ctypes.Array,
    x: ctypes.Array,
    rows: int,
    cols: int,
    nthreads: int | None = None,
) -> bytes:
    out = (ctypes.c_float * rows)(*(float("nan") for _ in range(rows)))
    fn = getattr(lib, name)
    args = (
        ctypes.cast(packed, U8P),
        ctypes.cast(scales, U8P),
        ctypes.cast(x, F32P),
        ctypes.cast(out, F32P),
        rows,
        cols,
    )
    if nthreads is None:
        fn(*args)
    else:
        fn(*args, nthreads)
    return bytes(out)


def _test_gemv(lib: ctypes.CDLL, rng: random.Random) -> None:
    shapes = (
        (1, 32),
        (2, 64),
        (3, 96),
        (19, 128),
        (33, 352),
        (7, 3072),
        (5, 3584),
    )
    for rows, cols in shapes:
        packed, scales = _matrix(rng, rows, cols)
        x = _activation(rng, cols)
        expected = _run(
            lib, "mxfp4_gemv_compat", packed, scales, x, rows, cols
        )
        names = ["mxfp4_gemv"]
        if lib.mxfp4_have_avx2():
            names.insert(0, "mxfp4_gemv_avx2")
        for name in names:
            got = _run(lib, name, packed, scales, x, rows, cols)
            if got != expected:
                raise AssertionError(f"{name} differs at shape {rows}x{cols}")
        for nthreads in (1, 2, 3, 5, 16, 17, 64):
            names = ["mxfp4_gemv_mt_compat", "mxfp4_gemv_mt"]
            if lib.mxfp4_have_avx2():
                names.insert(1, "mxfp4_gemv_mt_avx2")
            for name in names:
                got = _run(
                    lib, name, packed, scales, x, rows, cols, nthreads
                )
                if got != expected:
                    raise AssertionError(
                        f"{name}({nthreads}) differs at shape {rows}x{cols}"
                    )


def _test_scale_lut_equivalence(
    lut: ctypes.CDLL,
    synthesis: ctypes.CDLL,
    rng: random.Random,
) -> None:
    """Compare against the old table synthesis over every E8M0 byte."""
    rows, cols = 257, 96
    packed_values = [
        low | ((15 - low) << 4)
        for _ in range(rows * (cols // 32))
        for low in range(16)
    ]
    packed = (ctypes.c_uint8 * len(packed_values))(*packed_values)
    scale_count = rows * (cols // 32)
    scales = (ctypes.c_uint8 * scale_count)(
        *(index % 256 for index in range(scale_count))
    )
    x = _activation(rng, cols)
    names = ["mxfp4_gemv", "mxfp4_gemv_compat"]
    if lut.mxfp4_have_avx2() and synthesis.mxfp4_have_avx2():
        names.append("mxfp4_gemv_avx2")
    for name in names:
        expected = _run(synthesis, name, packed, scales, x, rows, cols)
        got = _run(lut, name, packed, scales, x, rows, cols)
        if got != expected:
            raise AssertionError(f"scale LUT differs from synthesis in {name}")
    for nthreads in (1, 3, 16):
        names = ["mxfp4_gemv_mt", "mxfp4_gemv_mt_compat"]
        if lut.mxfp4_have_avx2() and synthesis.mxfp4_have_avx2():
            names.append("mxfp4_gemv_mt_avx2")
        for name in names:
            expected = _run(
                synthesis, name, packed, scales, x, rows, cols, nthreads
            )
            got = _run(lut, name, packed, scales, x, rows, cols, nthreads)
            if got != expected:
                raise AssertionError(
                    f"scale LUT differs from synthesis in {name}({nthreads})"
                )


def _test_batch(
    lib: ctypes.CDLL,
    oracle: ctypes.CDLL,
    rng: random.Random,
) -> None:
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
    lib.mxfp4_pool_shutdown.argtypes = []
    lib.mxfp4_pool_shutdown.restype = None

    specs = ((3, 96), (9, 96), (17, 96), (5, 64))
    matrices = [_matrix(rng, rows, cols) for rows, cols in specs]
    shared = _activation(rng, 96)
    xs = (shared, shared, _activation(rng, 96), _activation(rng, 64))
    refs = [
        _run(oracle, "mxfp4_gemv_compat", p, s, x, rows, cols)
        for (rows, cols), (p, s), x in zip(specs, matrices, xs)
    ]
    outs = [
        (ctypes.c_float * rows)(*(float("nan") for _ in range(rows)))
        for rows, _ in specs
    ]
    n = len(specs)
    pp = (U8P * n)(*(ctypes.cast(p, U8P) for p, _ in matrices))
    sp = (U8P * n)(*(ctypes.cast(s, U8P) for _, s in matrices))
    xp = (F32P * n)(*(ctypes.cast(x, F32P) for x in xs))
    yp = (F32P * n)(*(ctypes.cast(y, F32P) for y in outs))
    rows = (ctypes.c_int * n)(*(shape[0] for shape in specs))
    cols = (ctypes.c_int * n)(*(shape[1] for shape in specs))
    lib.mxfp4_gemv_batch(pp, sp, xp, yp, rows, cols, n, 64)
    for i, (got, expected) in enumerate(zip(outs, refs)):
        if bytes(got) != expected:
            raise AssertionError(f"batch matrix {i} differs from compatibility path")
    expected_permutations = 3 if lib.mxfp4_have_avx2() else 0
    if lib.mxfp4_batch_last_x_permutations() != expected_permutations:
        raise AssertionError(
            "batch activation preparation count did not match selected ISA"
        )
    lib.mxfp4_pool_shutdown()


def _release(*libraries: ctypes.CDLL) -> None:
    """Unload test libraries so the temporary directory can be removed.

    Windows keeps a loaded module's file locked; on POSIX this is a no-op.
    """
    import build_native as native

    for library in libraries:
        native.release_library(library)


def main() -> int:
    if sys.platform == "darwin":
        suffix = ".dylib"
    elif os.name == "nt":
        suffix = ".dll"
    else:
        suffix = ".so"
    with tempfile.TemporaryDirectory(prefix="deltafin-x86-kernel-test-") as td:
        tmp = Path(td)
        gemv_path = tmp / f"libmxfp4gemv{suffix}"
        batch_path = tmp / f"libmxfp4batch{suffix}"
        synth_path = tmp / f"libmxfp4gemv-synthesis{suffix}"
        forced_gemv_path = tmp / f"libmxfp4gemv-forced-compat{suffix}"
        forced_batch_path = tmp / f"libmxfp4batch-forced-compat{suffix}"
        _build("fused_gemv.c", gemv_path)
        _build("fused_gemv_batch.c", batch_path)
        _build(
            "fused_gemv.c",
            synth_path,
            extra_cflags=("-DMXFP4_USE_SCALE_LUT=0",),
        )
        _build("fused_gemv.c", forced_gemv_path)
        _build("fused_gemv_batch.c", forced_batch_path)
        gemv = ctypes.CDLL(str(gemv_path))
        batch = ctypes.CDLL(str(batch_path))
        synthesis = ctypes.CDLL(str(synth_path))
        forced_gemv = None
        forced_batch = None
        try:
            _bind(gemv)
            _bind(batch)
            _bind(synthesis)
            rng = random.Random(0xD37AF1)
            _test_gemv(gemv, rng)
            _test_scale_lut_equivalence(gemv, synthesis, rng)
            _test_batch(batch, synthesis, rng)

            # The override is disable-only: it deterministically exercises the
            # baseline through the stable entry points without ever permitting
            # an unsupported ISA to be forced on.
            old_disable = os.environ.get("K3_MXFP4_DISABLE_AVX2")
            os.environ["K3_MXFP4_DISABLE_AVX2"] = "1"
            try:
                forced_gemv = ctypes.CDLL(str(forced_gemv_path))
                forced_batch = ctypes.CDLL(str(forced_batch_path))
                _bind(forced_gemv, expected_avx2=0)
                _bind(forced_batch, expected_avx2=0)
                forced_rng = random.Random(0xD37AF1)
                _test_gemv(forced_gemv, forced_rng)
                _test_scale_lut_equivalence(forced_gemv, synthesis, forced_rng)
                _test_batch(forced_batch, synthesis, forced_rng)
            finally:
                if old_disable is None:
                    os.environ.pop("K3_MXFP4_DISABLE_AVX2", None)
                else:
                    os.environ["K3_MXFP4_DISABLE_AVX2"] = old_disable
        finally:
            loaded = [gemv, batch, synthesis]
            loaded += [item for item in (forced_gemv, forced_batch) if item]
            _release(*loaded)
    print(
        "PASS: x86-64 runtime ISA selection and forced compatibility dispatch, scale LUT "
        "versus synthesis, odd/even GEMV, MT, and deduplicated batch paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
