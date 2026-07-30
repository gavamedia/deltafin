#!/usr/bin/env python3
"""Build and safely install Deltafin's native acceleration libraries.

The script is independent of the caller's working directory:

    python tools/build_native.py
    python tools/build_native.py --skip-metal
    python tools/build_native.py --cuda=off

Every library is linked to a unique temporary path beside its destination.
Required CPU/Metal outputs remain one transaction.  On Linux and Windows, CUDA
is a second, optional transaction in the default ``auto`` mode, so an installed
CUDA toolkit that cannot build Deltafin's optional backend never breaks the CPU
install.

Windows builds with MSVC.  When ``cl`` is not already on PATH the script locates
a Visual Studio installation with ``vswhere`` and captures the environment
``vcvars64.bat`` sets, so the ordinary command above works from a plain shell
rather than only from a Developer Command Prompt.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import os
import platform
import re
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
CUDA_PLATFORMS = frozenset(("linux", "windows"))
CUDA_PORTABLE_ARCH = "75"
_CUDA_ARCH_TOKEN = re.compile(
    r"(?:(?:sm|compute)_)?(?:(\d{1,2})\.(\d)|(\d{2,3}))\Z",
    re.IGNORECASE,
)
_CUDA_LISTED_ARCH = re.compile(
    r"\b(?:sm|compute)_(\d{2,3})\b",
    re.IGNORECASE,
)
X86_64_BASE_FEATURES = {
    "avx": frozenset(("avx",)),
    "fma": frozenset(("fma",)),
    "sse3": frozenset(("pni", "sse3")),
    "ssse3": frozenset(("ssse3",)),
}

# IsProcessorFeaturePresent identifiers from winnt.h.  Windows has no query for
# FMA3, which is why windows_cpu_features() derives it from AVX2.
PF_SSE3_INSTRUCTIONS_AVAILABLE = 13
PF_SSSE3_INSTRUCTIONS_AVAILABLE = 36
PF_AVX_INSTRUCTIONS_AVAILABLE = 39
PF_AVX2_INSTRUCTIONS_AVAILABLE = 40

# Command-line dialect and library-locking rules follow the machine running the
# build, which is not necessarily the machine the target describes.
HOST_IS_WINDOWS = os.name == "nt"

MSVC_VSWHERE_RELATIVE = r"Microsoft Visual Studio\Installer\vswhere.exe"
MSVC_VCVARS_RELATIVE = r"VC\Auxiliary\Build\vcvars64.bat"
MSVC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"


class BuildError(RuntimeError):
    """A native build was rejected before installation."""


@dataclasses.dataclass(frozen=True)
class Target:
    platform: str
    machine: str
    suffix: str
    c_arch_flags: tuple[str, ...]
    supports_metal: bool
    # "gnu" spells options the GCC/Clang way, "msvc" the cl.exe way.  Keeping it
    # on the target rather than deriving it from the platform inside every
    # command builder keeps the two option dialects in one place.
    toolchain: str = "gnu"


@dataclasses.dataclass(frozen=True)
class CudaCodegen:
    """Portable PTX floor plus optional device-specific binary images."""

    portable: str = CUDA_PORTABLE_ARCH
    native: tuple[str, ...] = ()
    explicit: bool = False


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

    if platform_name == "win32":
        if normalized not in {"amd64", "x86_64"}:
            raise BuildError(
                "unsupported Windows architecture "
                f"{raw_machine!r}; Deltafin's Windows native path requires x86-64"
            )
        # /arch:AVX is the exact analogue of the Linux -mavx baseline: it keeps
        # the compatibility kernel VEX-encoded without promising AVX2, which the
        # kernel still selects at runtime through its own intrinsics.
        return Target(
            "windows", "x86_64", ".dll", ("/arch:AVX",), False, "msvc"
        )

    raise BuildError(
        f"unsupported platform {platform_name!r}; native builds support "
        "Apple Silicon macOS, x86-64/aarch64 Linux and x86-64 Windows"
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


def windows_cpu_features(
    *,
    feature_probe: Callable[[int], bool] | None = None,
    assume_fma3: bool = False,
) -> frozenset[str]:
    """Report baseline features through Windows' processor-feature API.

    Windows exposes no FMA3 query, so FMA is reported only when AVX2 is present,
    which every FMA3-and-AVX2 CPU satisfies.  The rare AVX+FMA3 parts that lack
    AVX2 (AMD Piledriver) are therefore refused unless the caller opts in; that
    is deliberate, since guessing wrong means an illegal instruction at runtime.
    """
    if feature_probe is None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        present = kernel32.IsProcessorFeaturePresent
        present.argtypes = [ctypes.c_uint32]
        present.restype = ctypes.c_int

        def feature_probe(identifier: int) -> bool:  # noqa: F811
            return bool(present(identifier))

    features: set[str] = set()
    if feature_probe(PF_SSE3_INSTRUCTIONS_AVAILABLE):
        features.add("sse3")
    if feature_probe(PF_SSSE3_INSTRUCTIONS_AVAILABLE):
        features.add("ssse3")
    if feature_probe(PF_AVX_INSTRUCTIONS_AVAILABLE):
        features.add("avx")
    if assume_fma3 or feature_probe(PF_AVX2_INSTRUCTIONS_AVAILABLE):
        features.add("fma")
    return frozenset(features)


def preflight_target(
    target: Target,
    *,
    cpu_flags: frozenset[str] | set[str] | None = None,
    environ: Mapping[str, str] = os.environ,
) -> None:
    """Reject an x86 host that cannot execute the exact baseline kernel."""
    if target.machine != "x86_64" or target.platform not in {"linux", "windows"}:
        return
    if cpu_flags is not None:
        observed = frozenset(cpu_flags)
    elif target.platform == "windows":
        observed = windows_cpu_features(
            assume_fma3=environ.get("K3_ASSUME_FMA3", "").strip() not in {"", "0"}
        )
    else:
        observed = read_linux_cpu_flags()
    missing = sorted(
        name for name, aliases in X86_64_BASE_FEATURES.items()
        if observed.isdisjoint(aliases)
    )
    if not missing:
        return
    if target.platform == "windows":
        raise BuildError(
            "this Windows x86-64 CPU cannot be confirmed to run Deltafin's "
            "baseline AVX/FMA3/SSSE3 kernel; "
            f"missing CPU features: {', '.join(missing)}. "
            "Windows cannot report FMA3 directly, so it is inferred from AVX2; "
            "set K3_ASSUME_FMA3=1 if this CPU has FMA3 without AVX2"
        )
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
    if cuda_mode == "on" and target.platform not in CUDA_PLATFORMS:
        raise BuildError(
            "the CUDA native backend is supported only on Linux and Windows"
        )
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
        target.platform in CUDA_PLATFORMS
        and cuda_mode != "off"
        and (cuda_mode == "on" or nvcc_available)
    )
    if want_cuda:
        artifacts.append(
            Artifact(
                "CUDA MoE",
                tools_dir / "cuda_moe_kernels.cu",
                tools_dir / f"libcudamoe{target.suffix}",
                "cuda",
                CUDA_SYMBOLS,
                abi_version=CUDA_ABI_VERSION,
                abi_symbol="k3_cuda_moe_abi_version",
                expected_cuda_shapes=CUDA_SHAPES,
            )
        )
    return artifacts


def split_command(value: str, *, windows: bool = HOST_IS_WINDOWS) -> list[str]:
    """Split a compiler override into argv without corrupting Windows paths.

    shlex's POSIX mode reads the separators in ``C:\\Tools\\cl.exe`` as escape
    characters, so Windows uses shlex's non-POSIX mode, which keeps backslashes
    intact and only needs the surrounding quotes stripped afterwards.

    The dialect follows the *host*, not the target: a CC or NVCC override names
    an executable this machine has to parse and run, whichever platform the
    resulting library is for.
    """
    if not windows:
        return shlex.split(value)
    lexer = shlex.shlex(value, posix=False)
    lexer.whitespace_split = True
    return [token.strip('"') for token in lexer if token.strip('"')]


def render_command(
    command: Sequence[str], *, windows: bool = HOST_IS_WINDOWS
) -> str:
    """Render argv the way the host shell would accept it, for diagnostics."""
    if windows:
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _compiler_argv(
    language: str,
    target: Target,
    environ: Mapping[str, str],
    *,
    search_path: str | None = None,
) -> list[str]:
    variable = "CC" if language == "c" else "CXX"
    if target.toolchain == "msvc":
        fallback = "cl"
    elif target.platform == "darwin":
        fallback = "clang" if language == "c" else "clang++"
    else:
        fallback = "cc" if language == "c" else "c++"
    value = environ.get(variable, fallback).strip()
    try:
        argv = split_command(value)
    except ValueError as exc:
        raise BuildError(f"could not parse {variable}={value!r}: {exc}") from exc
    if not argv:
        raise BuildError(f"{variable} is empty; set it to a compiler executable")
    resolved = shutil.which(argv[0], path=search_path)
    if resolved is None:
        hint = (
            "Install the Visual Studio C++ build tools, run this from a "
            "Developer Command Prompt, or set "
            if target.toolchain == "msvc"
            else "Install it or set "
        )
        raise BuildError(
            f"{variable} compiler not found: {argv[0]!r}. "
            f"{hint}{variable} to a compatible compiler"
        )
    # Return the resolved path, not the bare name.  Windows resolves a child
    # process against the *parent's* PATH, so a compiler found only through a
    # captured MSVC environment would otherwise fail to start.
    return [resolved, *argv[1:]]


def environment_path(environ: Mapping[str, str]) -> str | None:
    """Read PATH from an environment mapping regardless of its spelling.

    Windows reports its own variables as ``Path``, so a plain lookup would miss
    exactly the value that makes a captured MSVC environment usable.
    """
    for name, value in environ.items():
        if name.upper() == "PATH":
            return value
    return None


def find_vswhere(environ: Mapping[str, str]) -> Path | None:
    """Locate the fixed-path Visual Studio locator, if it is installed."""
    roots = [
        environ.get("ProgramFiles(x86)"),
        environ.get("ProgramFiles"),
    ]
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / MSVC_VSWHERE_RELATIVE
        if candidate.is_file():
            return candidate
    return None


def find_vcvars(
    environ: Mapping[str, str],
    *,
    runner: Callable[..., object] = subprocess.run,
) -> Path | None:
    """Return the newest installation's vcvars64.bat, or None if absent."""
    vswhere = find_vswhere(environ)
    if vswhere is None:
        return None
    output = _probe_output(
        [
            str(vswhere),
            "-latest",
            "-products", "*",
            "-requires", MSVC_TOOLS_COMPONENT,
            "-property", "installationPath",
        ],
        runner=runner,
        timeout=60.0,
    )
    if not output:
        return None
    for line in output.splitlines():
        installation = line.strip()
        if not installation:
            continue
        vcvars = Path(installation) / MSVC_VCVARS_RELATIVE
        if vcvars.is_file():
            return vcvars
    return None


def msvc_environment(
    environ: Mapping[str, str] = os.environ,
    *,
    runner: Callable[..., object] = subprocess.run,
    finder: Callable[..., str | None] = shutil.which,
) -> dict[str, str] | None:
    """Capture the environment vcvars64.bat exports, when one is needed.

    Returns None when the ambient environment already resolves ``cl``, which is
    the case inside a Developer Command Prompt, and raises only when no usable
    toolchain can be found at all.
    """
    if finder("cl", path=environ.get("PATH")) is not None:
        return None
    vcvars = find_vcvars(environ, runner=runner)
    if vcvars is None:
        raise BuildError(
            "no Visual Studio C++ toolchain found. Install the "
            '"Desktop development with C++" workload, or run this script from '
            "a Developer Command Prompt so cl.exe is already on PATH"
        )
    # subprocess passes a string command line to CreateProcess unchanged, which
    # is what lets cmd's own /s quoting rule handle a path containing spaces.
    command = f'cmd /d /s /c ""{vcvars}" x64 >nul && set"'
    try:
        result = runner(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=180.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildError(f"could not run {vcvars}: {exc}") from exc
    if int(getattr(result, "returncode", 1)) != 0:
        detail = (str(getattr(result, "stderr", "")) or "").strip()
        raise BuildError(
            f"{vcvars} failed with exit code {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    captured: dict[str, str] = {}
    for line in str(getattr(result, "stdout", "") or "").splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            captured[name] = value
    if environment_path(captured) is None:
        raise BuildError(f"{vcvars} produced no PATH; cannot locate cl.exe")
    return captured


def cuda_compiler_argv(
    environ: Mapping[str, str],
    *,
    finder: Callable[[str], str | None] = shutil.which,
    windows: bool = HOST_IS_WINDOWS,
) -> list[str] | None:
    """Resolve ``NVCC`` safely, returning ``None`` when it is unavailable."""
    value = environ.get("NVCC", "nvcc").strip()
    try:
        argv = split_command(value, windows=windows)
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


def normalize_cuda_arch(value: str) -> str:
    """Return NVCC's numeric architecture suffix after strict validation."""
    candidate = value.strip().lower()
    match = _CUDA_ARCH_TOKEN.fullmatch(candidate)
    if match is None:
        raise BuildError(
            "CUDA architectures must look like 8.9, sm_89, compute_89, "
            f"or sm_120 (got {value!r})"
        )
    major, minor, compact = match.groups()
    number = int(compact if compact is not None else f"{int(major)}{minor}")
    if number < int(CUDA_PORTABLE_ARCH):
        raise BuildError(
            f"CUDA architecture sm_{number} is below Deltafin's "
            f"sm_{CUDA_PORTABLE_ARCH} support floor"
        )
    return str(number)


def parse_cuda_arches(value: str) -> tuple[str, ...]:
    """Parse a comma/space-separated K3_CUDA_ARCH override."""
    stripped = value.strip()
    if not stripped:
        raise BuildError("K3_CUDA_ARCH is empty; use auto or e.g. sm_89")
    tokens = [token for token in re.split(r"[\s,;]+", stripped) if token]
    arches = {normalize_cuda_arch(token) for token in tokens}
    return tuple(sorted(arches, key=int))


def _probe_output(
    command: Sequence[str],
    *,
    runner: Callable[..., object] = subprocess.run,
    timeout: float = 10.0,
) -> str | None:
    """Capture a short, non-shell tool query; return None on any probe failure."""
    try:
        result = runner(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if int(getattr(result, "returncode", 1)) != 0:
        return None
    return str(getattr(result, "stdout", "") or "")


def detect_cuda_arches(
    *,
    nvidia_smi: str | None,
    runner: Callable[..., object] = subprocess.run,
) -> tuple[str, ...]:
    """Query every installed GPU capability, ignoring malformed rows safely."""
    if not nvidia_smi:
        return ()
    output = _probe_output(
        [
            nvidia_smi,
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
    )
    if output is None:
        return ()
    detected: set[str] = set()
    for line in output.splitlines()[:256]:
        candidate = line.strip()
        if not candidate or len(candidate) > 32:
            continue
        try:
            detected.add(normalize_cuda_arch(candidate))
        except BuildError:
            continue
    return tuple(sorted(detected, key=int))


def nvcc_supported_arches(
    cuda_compiler: Sequence[str],
    *,
    runner: Callable[..., object] = subprocess.run,
) -> frozenset[str] | None:
    """Return architectures accepted as both real code and virtual PTX."""
    real_output = _probe_output(
        [*cuda_compiler, "--list-gpu-code"], runner=runner
    )
    virtual_output = _probe_output(
        [*cuda_compiler, "--list-gpu-arch"], runner=runner
    )
    if real_output is None or virtual_output is None:
        return None

    def listed(output: str) -> set[str]:
        return {
            str(int(match.group(1)))
            for match in _CUDA_LISTED_ARCH.finditer(output)
            if int(match.group(1)) >= int(CUDA_PORTABLE_ARCH)
        }

    supported = listed(real_output).intersection(listed(virtual_output))
    return frozenset(supported)


def resolve_cuda_codegen(
    environ: Mapping[str, str],
    cuda_compiler: Sequence[str],
    *,
    runner: Callable[..., object] = subprocess.run,
    nvidia_smi_finder: Callable[[str], str | None] = shutil.which,
    reporter: Callable[[str], None] | None = None,
) -> CudaCodegen:
    """Select native images without sacrificing a forward-compatible PTX floor."""
    supported = nvcc_supported_arches(cuda_compiler, runner=runner)
    if supported is not None and not supported:
        raise BuildError(
            "NVCC reports no architecture at or above Deltafin's sm_75 floor"
        )
    portable = (
        CUDA_PORTABLE_ARCH
        if supported is None or CUDA_PORTABLE_ARCH in supported
        else min(supported, key=int)
    )

    override = environ.get("K3_CUDA_ARCH")
    explicit = override is not None and override.strip().lower() != "auto"
    if explicit:
        requested = parse_cuda_arches(override)
        if supported is not None:
            rejected = [arch for arch in requested if arch not in supported]
            if rejected:
                rendered = ", ".join(f"sm_{arch}" for arch in rejected)
                raise BuildError(
                    f"K3_CUDA_ARCH requests {rendered}, but this NVCC does "
                    "not list that architecture"
                )
        native = tuple(arch for arch in requested if arch != portable)
        return CudaCodegen(portable, native, True)

    try:
        nvidia_smi = nvidia_smi_finder("nvidia-smi")
    except OSError:
        nvidia_smi = None
    detected = detect_cuda_arches(
        nvidia_smi=nvidia_smi,
        runner=runner,
    )
    rejected = (
        tuple(arch for arch in detected if arch not in supported)
        if supported is not None else ()
    )
    native = tuple(
        arch for arch in detected
        if arch != portable and (supported is None or arch in supported)
    )
    if rejected and reporter is not None:
        reporter(
            "NVCC cannot target detected "
            + ", ".join(f"sm_{arch}" for arch in rejected)
            + f"; retaining portable compute_{portable} PTX"
        )
    return CudaCodegen(portable, native, False)


def cuda_gencode_flags(codegen: CudaCodegen) -> list[str]:
    """Render deterministic SASS targets plus one portable PTX image."""
    arches = sorted({codegen.portable, *codegen.native}, key=int)
    flags = [
        f"-gencode=arch=compute_{arch},code=sm_{arch}"
        for arch in arches
    ]
    flags.append(
        f"-gencode=arch=compute_{codegen.portable},"
        f"code=compute_{codegen.portable}"
    )
    return flags


def _scratch_dir(scratch: Path | None, output: Path) -> str:
    """Return a directory path with the trailing separator MSVC's /Fo needs."""
    base = scratch if scratch is not None else output.parent
    return os.path.join(str(base), "")


def _scratch_file(scratch: Path | None, output: Path, suffix: str) -> str:
    """Name one intermediate after its output, inside the scratch directory."""
    base = scratch if scratch is not None else output.parent
    return str(Path(base) / (output.stem + suffix))


def compile_command(
    artifact: Artifact,
    target: Target,
    output: Path,
    *,
    environ: Mapping[str, str] = os.environ,
    cuda_compiler: Sequence[str] | None = None,
    cuda_codegen: CudaCodegen | None = None,
    scratch: Path | None = None,
    search_path: str | None = None,
) -> list[str]:
    """Construct the compiler invocation for one artifact.

    ``scratch`` receives a toolchain's intermediate files.  MSVC writes object
    files and an import library beside its output unless told otherwise, and
    those must not land in the repository next to the installed libraries.
    """
    windows = target.platform == "windows"
    if artifact.language == "cuda":
        if target.platform not in CUDA_PLATFORMS:
            raise BuildError(
                "the CUDA native backend is supported only on Linux and Windows"
            )
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
        if cuda_codegen is None:
            override = environ.get("K3_CUDA_ARCH")
            cuda_codegen = CudaCodegen(
                CUDA_PORTABLE_ARCH,
                (
                    ()
                    if override is None or override.strip().lower() == "auto"
                    else tuple(
                        arch for arch in parse_cuda_arches(override)
                        if arch != CUDA_PORTABLE_ARCH
                    )
                ),
                override is not None and override.strip().lower() != "auto",
            )
        # Position-independent code is the Linux default requirement; on Windows
        # the equivalent host requirement is the shared UCRT the interpreter
        # itself uses, so the DLL and its caller share one heap.
        host_flag = "-Xcompiler=/MD" if windows else "-Xcompiler=-fPIC"
        link_flags = (
            ["-Xlinker", f"/IMPLIB:{_scratch_file(scratch, output, '.lib')}"]
            if windows
            else []
        )
        return [
            *compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            host_flag,
            *cuda_gencode_flags(cuda_codegen),
            *link_flags,
            str(artifact.source),
            "-o",
            str(output),
        ]

    compiler = _compiler_argv(
        artifact.language, target, environ, search_path=search_path
    )
    if artifact.language == "c" and target.toolchain == "msvc":
        return [
            *compiler,
            "/nologo",
            "/O2",
            "/std:c11",
            # MSVC still gates C11 atomics behind this switch; the kernel's
            # dispatch counters and worker latches are _Atomic.
            "/experimental:c11atomics",
            *target.c_arch_flags,
            "/DNO_MAIN",
            "/LD",
            str(artifact.source),
            f"/Fe:{output}",
            f"/Fo:{_scratch_dir(scratch, output)}",
            "/link",
            "/INCREMENTAL:NO",
            f"/IMPLIB:{_scratch_file(scratch, output, '.lib')}",
        ]

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


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run one compiler and turn its diagnostics into a concise BuildError."""
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise BuildError(f"could not start compiler {command[0]!r}: {exc}") from exc
    if result.returncode == 0:
        return
    rendered = render_command(command)
    diagnostics = (result.stderr or result.stdout).strip()
    if not diagnostics:
        diagnostics = "(compiler produced no diagnostics)"
    raise BuildError(
        f"compiler failed with exit code {result.returncode}\n"
        f"command: {rendered}\n{diagnostics}"
    )


def release_library(library: object) -> None:
    """Drop a validated library's handle.

    Windows keeps a loaded module's file locked, which would make the atomic
    replace and the temporary-file cleanup that follow validation fail.  On
    other platforms the loader has no such restriction and this is a no-op.
    """
    if not HOST_IS_WINDOWS:
        return
    handle = getattr(library, "_handle", None)
    if handle is None:
        return
    free_library = ctypes.WinDLL("kernel32", use_last_error=True).FreeLibrary
    free_library.argtypes = [ctypes.c_void_p]
    free_library.restype = ctypes.c_int
    free_library(ctypes.c_void_p(handle))


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
    try:
        _validate_loaded(library, artifact)
    finally:
        release_library(library)


def _validate_loaded(library: object, artifact: Artifact) -> None:
    """Check one loaded library's symbols, ABI and shape handshakes."""
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
    nvidia_smi_finder: Callable[[str], str | None] = shutil.which,
    cuda_probe_runner: Callable[..., object] = subprocess.run,
    cuda_codegen: CudaCodegen | None = None,
    reporter: Reporter = _default_reporter,
    replace: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None]
    = os.replace,
    toolchain_env: Mapping[str, str] | None = None,
) -> list[Path]:
    """Compile, validate, and atomically install target libraries.

    CPU and Metal artifacts are always one required transaction.  CUDA is part
    of that transaction only when explicitly required with ``cuda_mode="on"``.
    In the default ``auto`` mode its build and install are independent and any
    failure is reported as a warning after the required transaction succeeds.

    ``toolchain_env`` overrides the environment compilers run in.  It exists for
    MSVC, whose compiler is only usable once vcvars has populated PATH, INCLUDE
    and LIB; passing it explicitly keeps that discovery out of the build loop.
    """
    target = detect_target() if target is None else target
    preflight_target(target, cpu_flags=cpu_flags, environ=environ)
    tools_dir = tools_dir.resolve()

    build_env: Mapping[str, str] | None = toolchain_env
    if build_env is None and target.toolchain == "msvc":
        build_env = msvc_environment(environ)
    # Compiler lookups must search the same PATH the compiler will run with.
    search_path = environment_path(build_env) if build_env is not None else None
    required_artifacts = artifacts_for(
        target,
        tools_dir=tools_dir,
        skip_metal=skip_metal,
        cuda_mode="off",
    )

    # Deliberately do not resolve NVCC on Darwin or in off mode.
    cuda_compiler: list[str] | None = None
    cuda_artifacts: list[Artifact] = []
    resolved_cuda_codegen = cuda_codegen
    if cuda_mode not in CUDA_MODES:
        raise BuildError(
            f"invalid CUDA mode {cuda_mode!r}; expected auto, on, or off"
        )
    if cuda_mode == "on" and target.platform not in CUDA_PLATFORMS:
        raise BuildError(
            "the CUDA native backend is supported only on Linux and Windows"
        )
    if target.platform in CUDA_PLATFORMS and cuda_mode != "off":
        try:
            cuda_compiler = cuda_compiler_argv(environ, finder=nvcc_finder)
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
            if resolved_cuda_codegen is None:
                try:
                    resolved_cuda_codegen = resolve_cuda_codegen(
                        environ,
                        cuda_compiler,
                        runner=cuda_probe_runner,
                        nvidia_smi_finder=nvidia_smi_finder,
                        reporter=reporter,
                    )
                except BuildError as exc:
                    if cuda_mode == "on":
                        raise
                    reporter(f"optional CUDA build skipped: {exc}")
                    cuda_compiler = None
            if (
                resolved_cuda_codegen is not None
                and resolved_cuda_codegen.native
            ):
                reporter(
                    "CUDA codegen adds native "
                    + ", ".join(
                        f"sm_{arch}" for arch in resolved_cuda_codegen.native
                    )
                    + f" SASS while retaining compute_"
                    f"{resolved_cuda_codegen.portable} PTX"
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
            run_command(command, cwd=cwd, env=build_env)
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
        *,
        scratch: Path,
        codegen: CudaCodegen | None = None,
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
                cuda_codegen=codegen,
                scratch=scratch,
                search_path=search_path,
            )
            runner(command, tools_dir.parent)
            validator(temporary, artifact)
            temporary.chmod(0o755)
            prepared.append((artifact, temporary))

    # One scratch directory per build holds every toolchain intermediate, so
    # nothing lands in tools/ beside the libraries it installs.
    scratch = Path(tempfile.mkdtemp(prefix="deltafin-native-build-"))
    try:
        prepare(required_artifacts, required_prepared, scratch=scratch)
        if cuda_artifacts:
            try:
                prepare(
                    cuda_artifacts,
                    cuda_prepared,
                    scratch=scratch,
                    codegen=resolved_cuda_codegen,
                )
            except Exception as exc:
                retry_error = exc
                can_retry_portable = (
                    resolved_cuda_codegen is not None
                    and bool(resolved_cuda_codegen.native)
                    and not resolved_cuda_codegen.explicit
                )
                if can_retry_portable:
                    portable_codegen = CudaCodegen(
                        resolved_cuda_codegen.portable
                    )
                    # A future build may contain more than one CUDA artifact.
                    # Discard any candidates completed before the failure and
                    # rebuild the complete CUDA transaction at the safe floor.
                    cuda_prepared.clear()
                    reporter(
                        "native CUDA architecture build failed; retrying the "
                        f"portable sm_{portable_codegen.portable} + "
                        f"compute_{portable_codegen.portable} target: {exc}"
                    )
                    try:
                        prepare(
                            cuda_artifacts,
                            cuda_prepared,
                            scratch=scratch,
                            codegen=portable_codegen,
                        )
                    except Exception as portable_exc:
                        retry_error = portable_exc
                    else:
                        retry_error = None
                if retry_error is None:
                    pass
                elif cuda_mode == "on":
                    raise BuildError(
                        "CUDA build required by --cuda=on failed: "
                        f"{retry_error}"
                    ) from retry_error
                else:
                    reporter(
                        "optional CUDA build failed; continuing with required "
                        f"CPU libraries: {retry_error}"
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
        shutil.rmtree(scratch, ignore_errors=True)


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
            "Linux and Windows CUDA backend: auto builds it when NVCC is "
            "available without blocking CPU installation; on requires it; off "
            "never probes (default: auto)"
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
