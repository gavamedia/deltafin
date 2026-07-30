#!/usr/bin/env python3
"""Build and safely install Deltafin's native acceleration libraries.

The script is independent of the caller's working directory:

    python tools/build_native.py
    python tools/build_native.py --skip-metal
    python tools/build_native.py --cuda=off

Every library is linked to a unique temporary path beside its destination.
Required CPU/Metal outputs remain one transaction.  On Linux, CUDA is a second,
optional transaction in the default ``auto`` mode, so an installed CUDA toolkit
that cannot build Deltafin's optional backend never breaks the CPU install.
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
CUDA_ABI_VERSION = 1
CUDA_POINTER_LAYOUT_VERSION = 1
CUDA_SHAPES = (3584, 3072, 17_547_264, CUDA_POINTER_LAYOUT_VERSION)
CUDA_MODES = frozenset(("auto", "on", "off"))
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
    abi_symbol: str | None = None
    expected_metal_shapes: tuple[int, int, int] | None = None
    expected_cuda_shapes: tuple[int, int, int, int] | None = None


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

CUDA_SYMBOLS = (
    "k3_cuda_moe_abi_version",
    "k3_cuda_moe_shapes",
    "k3_cuda_moe_available",
    "k3_cuda_moe_launch",
    "k3_cuda_mxfp4_gemv",
    "k3_cuda_int8_dequant",
    "k3_cuda_last_error",
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
    cuda_mode: str = "off",
    nvcc_available: bool = False,
) -> list[Artifact]:
    """Describe requested outputs without probing the host.

    ``nvcc_available`` is deliberately injected by the caller.  This keeps
    artifact selection deterministic in tests and ensures ``--cuda=off`` never
    performs an accidental compiler lookup.
    """
    if cuda_mode not in CUDA_MODES:
        raise BuildError(
            f"invalid CUDA mode {cuda_mode!r}; expected auto, on, or off"
        )
    if cuda_mode == "on" and target.platform != "linux":
        raise BuildError("the CUDA native backend is supported only on Linux")
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
            abi_symbol="mxfp4_abi_version",
        ),
        Artifact(
            "MXFP4 batch",
            tools_dir / "fused_gemv_batch.c",
            tools_dir / f"libmxfp4batch{target.suffix}",
            "c",
            (*BATCH_SYMBOLS, *x86_symbols),
            abi_version=NATIVE_ABI_VERSION,
            abi_symbol="mxfp4_abi_version",
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
    want_cuda = (
        target.platform == "linux"
        and cuda_mode != "off"
        and (cuda_mode == "on" or nvcc_available)
    )
    if want_cuda:
        artifacts.append(
            Artifact(
                "CUDA MoE",
                tools_dir / "cuda_moe_kernels.cu",
                tools_dir / "libcudamoe.so",
                "cuda",
                CUDA_SYMBOLS,
                abi_version=CUDA_ABI_VERSION,
                abi_symbol="k3_cuda_moe_abi_version",
                expected_cuda_shapes=CUDA_SHAPES,
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


def detect_cuda_arch(
    nvidia_smi: str = "nvidia-smi",
) -> str | None:
    """Detect the compute capability of the first CUDA-capable GPU.

    Returns "sm_XY" (e.g. "sm_89" for Ada Lovelace, "sm_120" for Blackwell)
    or None if no GPU is detected or nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        cap = result.stdout.strip()
        if not cap:
            return None
        major, _, minor = cap.partition(".")
        return f"sm_{major}{minor}"
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None


def cuda_compiler_argv(
    environ: Mapping[str, str],
    *,
    finder: Callable[[str], str | None] = shutil.which,
) -> list[str] | None:
    """Resolve ``NVCC`` safely, returning ``None`` when it is unavailable."""
    value = environ.get("NVCC", "nvcc").strip()
    try:
        argv = shlex.split(value)
    except ValueError as exc:
        raise BuildError(f"could not parse NVCC={value!r}: {exc}") from exc
    if not argv:
        raise BuildError(
            "NVCC is empty; set it to an nvcc executable or use --cuda=off"
        )
    try:
        resolved = finder(argv[0])
    except OSError as exc:
        raise BuildError(
            f"could not locate NVCC compiler {argv[0]!r}: {exc}"
        ) from exc
    if resolved is None:
        return None
    return [resolved, *argv[1:]]


def compile_command(
    artifact: Artifact,
    target: Target,
    output: Path,
    *,
    environ: Mapping[str, str] = os.environ,
    cuda_compiler: Sequence[str] | None = None,
) -> list[str]:
    """Construct the compiler invocation for one artifact."""
    if artifact.language == "cuda":
        if target.platform != "linux":
            raise BuildError("the CUDA native backend is supported only on Linux")
        compiler = (
            list(cuda_compiler)
            if cuda_compiler is not None
            else cuda_compiler_argv(environ)
        )
        if not compiler:
            raise BuildError(
                "NVCC compiler not found; install the CUDA toolkit, set NVCC, "
                "or use --cuda=off"
            )
        gpu_arch = environ.get("K3_CUDA_ARCH") or detect_cuda_arch() or "sm_75"
        compute = gpu_arch.removeprefix("sm_")
        return [
            *compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            f"-Xcompiler=-fPIC",
            f"-gencode=arch=compute_{compute},code={gpu_arch}",
            f"-gencode=arch=compute_{compute},code=compute_{compute}",
            str(artifact.source),
            "-o",
            str(output),
        ]

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
    abi_symbol = artifact.abi_symbol
    if artifact.abi_version is not None:
        abi_symbol = abi_symbol or "mxfp4_abi_version"
        if abi_symbol not in symbols:
            symbols.insert(0, abi_symbol)
    missing = [symbol for symbol in symbols if not hasattr(library, symbol)]
    if missing:
        raise BuildError(
            f"{artifact.label} is missing required symbols: {', '.join(missing)}"
        )

    if artifact.abi_version is not None:
        assert abi_symbol is not None
        abi = getattr(library, abi_symbol)
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

    if artifact.expected_cuda_shapes is not None:
        hidden = ctypes.c_int()
        intermediate = ctypes.c_int()
        span = ctypes.c_int64()
        pointer_layout = ctypes.c_uint32()
        shapes = getattr(library, "k3_cuda_moe_shapes")
        try:
            shapes.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            shapes.restype = None
            shapes(
                ctypes.byref(hidden),
                ctypes.byref(intermediate),
                ctypes.byref(span),
                ctypes.byref(pointer_layout),
            )
        except Exception as exc:
            raise BuildError(
                f"{artifact.label} shape/layout handshake failed: {exc}"
            ) from exc
        observed_shapes = (
            hidden.value,
            intermediate.value,
            span.value,
            pointer_layout.value,
        )
        if observed_shapes != artifact.expected_cuda_shapes:
            raise BuildError(
                f"{artifact.label} shape/layout handshake returned "
                f"{observed_shapes}; expected {artifact.expected_cuda_shapes}"
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
Reporter = Callable[[str], None]


def _default_reporter(message: str) -> None:
    print(message, file=sys.stderr)


def build_native(
    *,
    target: Target | None = None,
    tools_dir: Path = TOOLS_DIR,
    skip_metal: bool = False,
    cuda_mode: str = "auto",
    environ: Mapping[str, str] = os.environ,
    runner: Callable[[Sequence[str], Path], None] | None = None,
    validator: Validator | None = None,
    cpu_flags: frozenset[str] | set[str] | None = None,
    nvcc_finder: Callable[[str], str | None] = shutil.which,
    reporter: Reporter = _default_reporter,
    replace: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None]
    = os.replace,
) -> list[Path]:
    """Compile, validate, and atomically install target libraries.

    CPU and Metal artifacts are always one required transaction.  CUDA is part
    of that transaction only when explicitly required with ``cuda_mode="on"``.
    In the default ``auto`` mode its build and install are independent and any
    failure is reported as a warning after the required transaction succeeds.
    """
    target = detect_target() if target is None else target
    preflight_target(target, cpu_flags=cpu_flags)
    tools_dir = tools_dir.resolve()
    required_artifacts = artifacts_for(
        target,
        tools_dir=tools_dir,
        skip_metal=skip_metal,
        cuda_mode="off",
    )

    # Deliberately do not resolve NVCC on Darwin or in off mode.
    cuda_compiler: list[str] | None = None
    cuda_artifacts: list[Artifact] = []
    if cuda_mode not in CUDA_MODES:
        raise BuildError(
            f"invalid CUDA mode {cuda_mode!r}; expected auto, on, or off"
        )
    if cuda_mode == "on" and target.platform != "linux":
        raise BuildError("the CUDA native backend is supported only on Linux")
    if target.platform == "linux" and cuda_mode != "off":
        try:
            cuda_compiler = cuda_compiler_argv(
                environ, finder=nvcc_finder
            )
        except BuildError as exc:
            if cuda_mode == "on":
                raise
            reporter(f"optional CUDA build skipped: {exc}")
        if cuda_compiler is None and cuda_mode == "on":
            raise BuildError(
                "CUDA was required by --cuda=on, but NVCC was not found. "
                "Install the CUDA toolkit or set NVCC to its compiler"
            )
        if cuda_compiler is not None:
            cuda_artifacts = [
                artifact
                for artifact in artifacts_for(
                    target,
                    tools_dir=tools_dir,
                    skip_metal=skip_metal,
                    cuda_mode=cuda_mode,
                    nvcc_available=True,
                )
                if artifact.language == "cuda"
            ]

    if runner is None:
        def invoke(command: Sequence[str], cwd: Path) -> None:
            run_command(command, cwd=cwd)
        runner = invoke
    if validator is None:
        def check(path: Path, artifact: Artifact) -> None:
            validate_artifact(path, artifact)
        validator = check

    required_prepared: list[tuple[Artifact, Path]] = []
    cuda_prepared: list[tuple[Artifact, Path]] = []
    temporaries: list[Path] = []

    def prepare(
        artifacts: Sequence[Artifact],
        prepared: list[tuple[Artifact, Path]],
    ) -> None:
        for artifact in artifacts:
            if not artifact.source.is_file():
                qualifier = (
                    "optional CUDA" if artifact.language == "cuda" else "required"
                )
                raise BuildError(
                    f"{qualifier} source file not found: {artifact.source}"
                )
            temporary = _temporary_output(artifact.destination)
            temporaries.append(temporary)
            command = compile_command(
                artifact,
                target,
                temporary,
                environ=environ,
                cuda_compiler=cuda_compiler,
            )
            runner(command, tools_dir.parent)
            validator(temporary, artifact)
            temporary.chmod(0o755)
            prepared.append((artifact, temporary))

    try:
        prepare(required_artifacts, required_prepared)
        if cuda_artifacts:
            try:
                prepare(cuda_artifacts, cuda_prepared)
            except Exception as exc:
                if cuda_mode == "on":
                    raise BuildError(
                        "CUDA build required by --cuda=on failed: "
                        f"{exc}"
                    ) from exc
                reporter(
                    "optional CUDA build failed; continuing with required "
                    f"CPU libraries: {exc}"
                )
                cuda_prepared.clear()

        if cuda_mode == "on":
            # Explicitly requested outputs are one all-or-nothing transaction.
            prepared = [*required_prepared, *cuda_prepared]
            install_validated(prepared, replace=replace)
            return [artifact.destination for artifact, _ in prepared]

        # Required outputs retain their existing all-or-nothing transaction.
        install_validated(required_prepared, replace=replace)
        outputs = [
            artifact.destination for artifact, _ in required_prepared
        ]
        if cuda_prepared:
            try:
                install_validated(cuda_prepared, replace=replace)
            except BuildError as exc:
                reporter(
                    "optional CUDA library was not installed; CPU libraries "
                    f"remain available: {exc}"
                )
            else:
                outputs.extend(
                    artifact.destination for artifact, _ in cuda_prepared
                )
        return outputs
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
    parser.add_argument(
        "--cuda",
        choices=sorted(CUDA_MODES),
        default="auto",
        help=(
            "Linux CUDA backend: auto builds it when NVCC is available "
            "without blocking CPU installation; on requires it; off never probes "
            "(default: auto)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = detect_target()
        outputs = build_native(
            target=target,
            skip_metal=args.skip_metal,
            cuda_mode=args.cuda,
        )
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
