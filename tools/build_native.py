#!/usr/bin/env python3
"""Build and safely install Deltafin's native acceleration libraries.

The script is independent of the caller's working directory:

    python tools/build_native.py
    python tools/build_native.py --skip-metal

Every library is linked to a unique temporary path beside its destination.  All
requested outputs must compile and pass their ABI/symbol validation before any
existing library is replaced.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
NATIVE_ABI_VERSION = 1
METAL_SHAPES = (3584, 3072, 17_547_264)


def _find_nvcc() -> str | None:
    """Locate nvcc in PATH or common CUDA install directories."""
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return nvcc
    candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/lib/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None
X86_64_BASE_FEATURES = {
    "avx": frozenset(("avx",)),
    "fma": frozenset(("fma",)),
    "sse3": frozenset(("pni", "sse3")),
    "ssse3": frozenset(("ssse3",)),
}


class BuildError(RuntimeError):
    """A native build was rejected before installation."""


@dataclasses.dataclass(frozen=True)
class Target:
    platform: str
    machine: str
    suffix: str
    c_arch_flags: tuple[str, ...]
    supports_metal: bool


@dataclasses.dataclass(frozen=True)
class Artifact:
    label: str
    source: Path
    destination: Path
    language: str
    required_symbols: tuple[str, ...]
    abi_version: int | None = None
    expected_metal_shapes: tuple[int, int, int] | None = None


GEMV_SYMBOLS = (
    "mxfp4_gemv",
    "mxfp4_gemv_mt",
    "mxfp4_gemv_compat",
    "mxfp4_gemv_mt_compat",
    "mxfp4_have_avx2",
    "mxfp4_expert_triple",
)

X86_AVX2_SYMBOLS = (
    "mxfp4_gemv_avx2",
    "mxfp4_gemv_mt_avx2",
)

BATCH_SYMBOLS = (
    "mxfp4_gemv",
    "mxfp4_gemv_mt",
    "mxfp4_gemv_compat",
    "mxfp4_gemv_mt_compat",
    "mxfp4_have_avx2",
    "mxfp4_gemv_batch",
    "mxfp4_batch_last_x_permutations",
    "mxfp4_moe_layer",
    "mxfp4_situ_batch",
    "mxfp4_moe_expert_set",
    "mxfp4_pool_init",
    "mxfp4_pool_threads",
    "mxfp4_pool_shutdown",
)

METAL_SYMBOLS = (
    "k3_metal_init",
    "k3_metal_available",
    "k3_metal_last_error",
    "k3_metal_shapes",
    "k3_metal_stats",
    "k3_metal_mode",
    "k3_metal_set_mode",
    "k3_metal_drop",
    "k3_metal_flush",
    "k3_metal_moe_layer",
    "k3_metal_moe_positions",
)

CUDA_MOE_SYMBOLS = (
    "cuda_moe_available",
    "cuda_mxfp4_gemv",
    "cuda_mxfp4_moe_layer",
    "cuda_moe_zero_output",
    "cuda_int8_deq",
    "cuda_mxfp4_moe_positions",
    "cuda_moe_error",
)


def detect_target(
    platform_name: str | None = None,
    machine: str | None = None,
) -> Target:
    """Return the supported native build target or raise a useful error."""
    platform_name = sys.platform if platform_name is None else platform_name
    raw_machine = platform.machine() if machine is None else machine
    normalized = raw_machine.strip().lower()

    if platform_name == "darwin":
        if normalized not in {"arm64", "aarch64"}:
            raise BuildError(
                "unsupported macOS architecture "
                f"{raw_machine!r}; Deltafin's macOS native path requires Apple Silicon"
            )
        return Target("darwin", "arm64", ".dylib", ("-mcpu=native",), True)

    if platform_name.startswith("linux"):
        if normalized in {"x86_64", "amd64"}:
            return Target(
                "linux",
                "x86_64",
                ".so",
                (
                    "-march=x86-64",
                    "-mtune=native",
                    "-msse3",
                    "-mssse3",
                    "-mavx",
                    "-mfma",
                ),
                False,
            )
        if normalized in {"arm64", "aarch64"}:
            return Target("linux", "aarch64", ".so", ("-mcpu=native",), False)
        raise BuildError(
            "unsupported Linux architecture "
            f"{raw_machine!r}; supported architectures are x86-64 and aarch64"
        )

    raise BuildError(
        f"unsupported platform {platform_name!r}; native builds support "
        "Apple Silicon macOS and x86-64/aarch64 Linux"
    )


def read_linux_cpu_flags(cpuinfo: Path = Path("/proc/cpuinfo")) -> frozenset[str]:
    """Read feature names present on every logical CPU in /proc/cpuinfo."""
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise BuildError(
            f"could not verify x86-64 CPU features from {cpuinfo}: {exc}"
        ) from exc
    per_cpu: list[set[str]] = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"flags", "features"}:
            per_cpu.append(set(value.lower().split()))
    if not per_cpu:
        raise BuildError(f"no CPU feature flags found in {cpuinfo}")
    common = per_cpu[0]
    for flags in per_cpu[1:]:
        common.intersection_update(flags)
    return frozenset(common)


def preflight_target(
    target: Target,
    *,
    cpu_flags: frozenset[str] | set[str] | None = None,
) -> None:
    """Reject an x86 host that cannot execute the exact baseline kernel."""
    if target.platform != "linux" or target.machine != "x86_64":
        return
    observed = read_linux_cpu_flags() if cpu_flags is None else frozenset(cpu_flags)
    missing = sorted(
        name for name, aliases in X86_64_BASE_FEATURES.items()
        if observed.isdisjoint(aliases)
    )
    if missing:
        raise BuildError(
            "this Linux x86-64 CPU cannot run Deltafin's baseline "
            "AVX/FMA3/SSSE3 kernel; "
            f"missing CPU flags: {', '.join(missing)} "
            "(AVX2 is optional and selected at runtime)"
        )


def artifacts_for(
    target: Target,
    *,
    tools_dir: Path = TOOLS_DIR,
    skip_metal: bool = False,
) -> list[Artifact]:
    """Describe every output requested for one target."""
    tools_dir = tools_dir.resolve()
    x86_symbols = X86_AVX2_SYMBOLS if target.machine == "x86_64" else ()
    artifacts = [
        Artifact(
            "MXFP4 GEMV",
            tools_dir / "fused_gemv.c",
            tools_dir / f"libmxfp4gemv{target.suffix}",
            "c",
            (*GEMV_SYMBOLS, *x86_symbols),
            abi_version=NATIVE_ABI_VERSION,
        ),
        Artifact(
            "MXFP4 batch",
            tools_dir / "fused_gemv_batch.c",
            tools_dir / f"libmxfp4batch{target.suffix}",
            "c",
            (*BATCH_SYMBOLS, *x86_symbols),
            abi_version=NATIVE_ABI_VERSION,
        ),
    ]
    if target.supports_metal and not skip_metal:
        artifacts.append(
            Artifact(
                "Metal MoE",
                tools_dir / "metal_moe.mm",
                tools_dir / "libk3metalmoe.dylib",
                "cxx",
                METAL_SYMBOLS,
                expected_metal_shapes=METAL_SHAPES,
            )
        )
    # CUDA MoE library: built on Linux x86-64 or aarch64 when nvcc is found.
    if target.platform == "linux" and _find_nvcc() is not None:
        artifacts.append(
            Artifact(
                "CUDA MoE",
                tools_dir / "cuda_moe_kernels.cu",
                tools_dir / "libcudamoe.so",
                "cuda",
                CUDA_MOE_SYMBOLS,
            )
        )
    return artifacts


def _compiler_argv(
    language: str,
    target: Target,
    environ: Mapping[str, str],
) -> list[str]:
    variable = "CC" if language == "c" else "CXX"
    if target.platform == "darwin":
        fallback = "clang" if language == "c" else "clang++"
    else:
        fallback = "cc" if language == "c" else "c++"
    value = environ.get(variable, fallback).strip()
    try:
        argv = shlex.split(value)
    except ValueError as exc:
        raise BuildError(f"could not parse {variable}={value!r}: {exc}") from exc
    if not argv:
        raise BuildError(f"{variable} is empty; set it to a compiler executable")
    if shutil.which(argv[0]) is None:
        raise BuildError(
            f"{variable} compiler not found: {argv[0]!r}. "
            f"Install it or set {variable} to a compatible compiler"
        )
    return argv


def compile_command(
    artifact: Artifact,
    target: Target,
    output: Path,
    *,
    environ: Mapping[str, str] = os.environ,
) -> list[str]:
    """Construct the compiler invocation for one artifact."""
    compiler = _compiler_argv(artifact.language, target, environ)
    if artifact.language == "c":
        common = [
            "-O3",
            *target.c_arch_flags,
            "-std=gnu11",
            "-DNO_MAIN",
        ]
        linkage = (
            ["-dynamiclib"]
            if target.platform == "darwin"
            else ["-fPIC", "-shared"]
        )
        return [
            *compiler,
            *common,
            *linkage,
            str(artifact.source),
            "-o",
            str(output),
            "-lpthread",
            "-lm",
        ]

    if artifact.language == "cuda":
        nvcc = _find_nvcc()
        if nvcc is None:
            raise BuildError(
                "nvcc not found; install CUDA Toolkit or skip this artifact"
            )
        return [
            nvcc,
            "-O3",
            "-shared",
            "-gencode", "arch=compute_75,code=sm_75",
            "-gencode", "arch=compute_80,code=sm_80",
            "-gencode", "arch=compute_90,code=sm_90",
            "-gencode", "arch=compute_100,code=sm_100",
            "-gencode", "arch=compute_120,code=sm_120",
            "-t", "8",          # parallel compilation
            str(artifact.source),
            "-o",
            str(output),
        ]

    if artifact.language == "cxx" and target.platform == "darwin":
        return [
            *compiler,
            "-O2",
            "-std=c++17",
            "-fobjc-arc",
            "-dynamiclib",
            str(artifact.source),
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-o",
            str(output),
        ]
    raise BuildError(
        f"no {artifact.language} build recipe for "
        f"{target.platform}/{target.machine}"
    )


def run_command(command: Sequence[str], *, cwd: Path) -> None:
    """Run one compiler and turn its diagnostics into a concise BuildError."""
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise BuildError(f"could not start compiler {command[0]!r}: {exc}") from exc
    if result.returncode == 0:
        return
    rendered = shlex.join(command)
    diagnostics = (result.stderr or result.stdout).strip()
    if not diagnostics:
        diagnostics = "(compiler produced no diagnostics)"
    raise BuildError(
        f"compiler failed with exit code {result.returncode}\n"
        f"command: {rendered}\n{diagnostics}"
    )


def validate_artifact(
    path: Path,
    artifact: Artifact,
    *,
    cdll_factory: Callable[[str], object] = ctypes.CDLL,
) -> None:
    """Load a temporary library and validate its public contract."""
    if not path.is_file() or path.stat().st_size == 0:
        raise BuildError(f"{artifact.label} compiler produced no library at {path}")
    try:
        library = cdll_factory(str(path))
    except OSError as exc:
        raise BuildError(
            f"{artifact.label} built but could not be loaded: {exc}"
        ) from exc

    symbols = list(artifact.required_symbols)
    if artifact.abi_version is not None:
        symbols.insert(0, "mxfp4_abi_version")
    missing = [symbol for symbol in symbols if not hasattr(library, symbol)]
    if missing:
        raise BuildError(
            f"{artifact.label} is missing required symbols: {', '.join(missing)}"
        )

    if artifact.abi_version is not None:
        abi = getattr(library, "mxfp4_abi_version")
        try:
            abi.argtypes = []
            abi.restype = ctypes.c_uint32
            observed = int(abi())
        except Exception as exc:
            raise BuildError(
                f"{artifact.label} ABI query failed: {exc}"
            ) from exc
        if observed != artifact.abi_version:
            raise BuildError(
                f"{artifact.label} has ABI {observed}; expected "
                f"{artifact.abi_version}"
            )

    if artifact.expected_metal_shapes is not None:
        hidden = ctypes.c_int()
        intermediate = ctypes.c_int()
        span = ctypes.c_longlong()
        shapes = getattr(library, "k3_metal_shapes")
        try:
            shapes.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_longlong),
            ]
            shapes.restype = None
            shapes(
                ctypes.byref(hidden),
                ctypes.byref(intermediate),
                ctypes.byref(span),
            )
        except Exception as exc:
            raise BuildError(
                f"{artifact.label} shape handshake failed: {exc}"
            ) from exc
        observed_shapes = (hidden.value, intermediate.value, span.value)
        if observed_shapes != artifact.expected_metal_shapes:
            raise BuildError(
                f"{artifact.label} shape handshake returned {observed_shapes}; "
                f"expected {artifact.expected_metal_shapes}"
            )


def _temporary_output(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.build-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    return Path(name)


def _backup_path(destination: Path) -> Path:
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.backup-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    backup = Path(name)
    backup.unlink()
    return backup


def install_validated(
    prepared: Sequence[tuple[Artifact, Path]],
    *,
    replace: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None]
    = os.replace,
) -> None:
    """Install all prepared outputs, rolling back an interrupted replacement."""
    backups: dict[Path, Path | None] = {}
    installed: list[Path] = []
    preserve_backups: set[Path] = set()
    try:
        for artifact, _ in prepared:
            destination = artifact.destination
            if destination.exists():
                backup = _backup_path(destination)
                try:
                    os.link(destination, backup)
                except OSError:
                    shutil.copy2(destination, backup)
                backups[destination] = backup
            else:
                backups[destination] = None

        for artifact, temporary in prepared:
            replace(temporary, artifact.destination)
            installed.append(artifact.destination)
    except Exception as exc:
        rollback_errors: list[str] = []
        for destination in reversed(installed):
            backup = backups.get(destination)
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                elif backup.exists():
                    os.replace(backup, destination)
            except OSError as rollback_exc:
                rollback_errors.append(f"{destination}: {rollback_exc}")
                if backup is not None:
                    preserve_backups.add(backup)
        detail = (
            f"; rollback also failed for {'; '.join(rollback_errors)}"
            if rollback_errors
            else ""
        )
        raise BuildError(f"could not install native libraries: {exc}{detail}") from exc
    finally:
        for backup in backups.values():
            if backup is not None and backup not in preserve_backups:
                backup.unlink(missing_ok=True)


Validator = Callable[[Path, Artifact], None]


def build_native(
    *,
    target: Target | None = None,
    tools_dir: Path = TOOLS_DIR,
    skip_metal: bool = False,
    environ: Mapping[str, str] = os.environ,
    runner: Callable[[Sequence[str], Path], None] | None = None,
    validator: Validator | None = None,
    cpu_flags: frozenset[str] | set[str] | None = None,
    replace: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None]
    = os.replace,
) -> list[Path]:
    """Compile, validate, and atomically install all target libraries."""
    target = detect_target() if target is None else target
    preflight_target(target, cpu_flags=cpu_flags)
    tools_dir = tools_dir.resolve()
    artifacts = artifacts_for(target, tools_dir=tools_dir, skip_metal=skip_metal)
    for artifact in artifacts:
        if not artifact.source.is_file():
            raise BuildError(f"required source file not found: {artifact.source}")

    if runner is None:
        def invoke(command: Sequence[str], cwd: Path) -> None:
            run_command(command, cwd=cwd)
        runner = invoke
    if validator is None:
        def check(path: Path, artifact: Artifact) -> None:
            validate_artifact(path, artifact)
        validator = check

    prepared: list[tuple[Artifact, Path]] = []
    temporaries: list[Path] = []
    try:
        for artifact in artifacts:
            temporary = _temporary_output(artifact.destination)
            temporaries.append(temporary)
            command = compile_command(
                artifact, target, temporary, environ=environ
            )
            runner(command, tools_dir.parent)
            validator(temporary, artifact)
            temporary.chmod(0o755)
            prepared.append((artifact, temporary))

        # Nothing under a production filename changes until every output passed.
        install_validated(prepared, replace=replace)
        return [artifact.destination for artifact, _ in prepared]
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate Deltafin native libraries for this machine"
        )
    )
    parser.add_argument(
        "--skip-metal",
        action="store_true",
        help="on Apple Silicon, build only the CPU GEMV and batch libraries",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = detect_target()
        outputs = build_native(target=target, skip_metal=args.skip_metal)
    except BuildError as exc:
        print(f"native build failed: {exc}", file=sys.stderr)
        return 1

    print(f"native build complete for {target.platform}/{target.machine}:")
    for output in outputs:
        try:
            shown = output.relative_to(REPO_ROOT)
        except ValueError:
            shown = output
        print(f"  {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
