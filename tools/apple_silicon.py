"""Capability-based Apple Silicon runtime dispatch and provenance.

The native probe asks ``MTLDevice`` directly. Device names are recorded for
benchmark provenance but are never used for decisions. Missing queries are
``unknown`` and therefore fail automatic experiment gates; ``force`` is the
explicit escape hatch for bring-up work.

The tiny native helper is compiled into a per-user temporary cache. A missing
compiler, non-Darwin host, unavailable Metal device, or older API leaves a
partial snapshot rather than breaking the baseline runtime.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import pathlib
import platform
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Iterable


HERE = pathlib.Path(__file__).resolve().parent
NATIVE_SOURCE = HERE / "apple_silicon_caps.mm"
FALSE_MODES = frozenset(("0", "false", "off", "no", "disabled"))
TRUE_MODES = frozenset(("1", "true", "on", "yes", "enabled", "auto"))
FORCE_MODES = frozenset(("force", "forced"))


def _command_output(argv: list[str], timeout: float = 3.0) -> str | None:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _windows_physical_memory_bytes() -> int | None:
    """Installed RAM from GlobalMemoryStatusEx.

    Callers size a RAM budget with this.  Returning None instead makes them
    pin nothing at all, which on a large-memory host silently forfeits the
    resident tier the layer cache exists for.
    """
    import ctypes
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return _positive_int(status.ullTotalPhys)


def _physical_memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        value = _command_output(["sysctl", "-n", "hw.memsize"])
        parsed = _positive_int(value)
        if parsed is not None:
            return parsed
    if os.name == "nt":
        try:
            return _windows_physical_memory_bytes()
        except OSError:
            return None
    try:
        # A POSIX host without this name raises; there is nothing else to try.
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return _positive_int(pages * page_size)


def _macos_version() -> dict[str, str]:
    out = {}
    for name, flag in (
        ("product_name", "-productName"),
        ("product_version", "-productVersion"),
        ("build_version", "-buildVersion"),
    ):
        value = _command_output(["sw_vers", flag])
        if value is not None:
            out[name] = value
    return out


def _helper_cache_path() -> pathlib.Path:
    source = NATIVE_SOURCE.read_bytes()
    key = hashlib.sha256(
        source + platform.machine().encode() + platform.release().encode()
    ).hexdigest()[:20]
    root = pathlib.Path(tempfile.gettempdir()) / f"deltafin-metal-caps-{os.getuid()}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root / f"probe-{key}"


def _safe_existing_helper(path: pathlib.Path) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.getuid()
        and bool(info.st_mode & stat.S_IXUSR)
    )


def _compile_native_helper() -> tuple[pathlib.Path | None, str | None]:
    override = os.environ.get("K3_METAL_CAPS_HELPER")
    if override:
        path = pathlib.Path(override).expanduser()
        if _safe_existing_helper(path):
            return path, None
        return None, f"K3_METAL_CAPS_HELPER is not a safe executable: {path}"
    if platform.system() != "Darwin":
        return None, "not Darwin"
    if not NATIVE_SOURCE.exists():
        return None, f"missing native source: {NATIVE_SOURCE}"

    target = _helper_cache_path()
    if _safe_existing_helper(target):
        return target, None
    compiler = shutil.which("xcrun")
    if compiler is None:
        return None, "xcrun is unavailable"

    fd, temporary = tempfile.mkstemp(prefix="probe-build-", dir=target.parent)
    os.close(fd)
    temp_path = pathlib.Path(temporary)
    try:
        temp_path.unlink()
    except OSError:
        pass
    command = [
        compiler,
        "clang++",
        "-O2",
        "-std=c++17",
        "-fobjc-arc",
        str(NATIVE_SOURCE),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-o",
        str(temp_path),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"native probe compile failed: {exc}"
    if proc.returncode != 0:
        try:
            temp_path.unlink()
        except OSError:
            pass
        tail = (proc.stderr or proc.stdout).strip()[-1000:]
        return None, f"native probe compile failed ({proc.returncode}): {tail}"
    try:
        temp_path.chmod(0o700)
        os.replace(temp_path, target)
    except OSError as exc:
        try:
            temp_path.unlink()
        except OSError:
            pass
        return None, f"could not publish native probe: {exc}"
    return target, None


def _metal_probe() -> tuple[dict[str, Any], str | None]:
    helper, error = _compile_native_helper()
    if helper is None:
        return {"available": False}, error
    try:
        proc = subprocess.run(
            [str(helper)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False}, f"native probe failed: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip()[-1000:]
        return (
            {"available": False},
            f"native probe exited {proc.returncode}: {tail}",
        )
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"available": False}, f"native probe returned invalid JSON: {exc}"
    if not isinstance(value, dict):
        return {"available": False}, "native probe JSON is not an object"
    value["available"] = bool(value.get("available"))
    return value, None


@dataclasses.dataclass(frozen=True)
class CapabilitySnapshot:
    system: str
    machine: str
    physical_memory_bytes: int | None
    macos: dict[str, str]
    metal: dict[str, Any]
    probe_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["fingerprint"] = self.fingerprint()
        return out

    def fingerprint(self) -> str:
        stable = {
            "system": self.system,
            "machine": self.machine,
            "physical_memory_bytes": self.physical_memory_bytes,
            "macos": self.macos,
            "metal": self.metal,
        }
        encoded = json.dumps(
            stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def supports_family(self, family: str) -> bool | None:
        if not self.metal.get("available"):
            return False
        families = self.metal.get("supported_families")
        if not isinstance(families, list):
            return None
        return family.lower() in {str(item).lower() for item in families}

    def feature(self, name: str) -> bool | None:
        if not self.metal.get("available"):
            return False
        features = self.metal.get("features")
        if not isinstance(features, dict) or name not in features:
            return None
        value = features[name]
        return value if isinstance(value, bool) else None

    def max_buffer_allows(self, nbytes: int) -> bool | None:
        limit = _positive_int(self.metal.get("max_buffer_length_bytes"))
        if limit is None:
            return None
        return int(nbytes) <= limit

    def safe_unified_budget(
        self,
        *,
        host_reserve_bytes: int = 0,
        device_reserve_bytes: int = 0,
        utilization: float = 1.0,
    ) -> int:
        """Return a conservative shared-memory budget.

        The budget is the smaller of physical RAM after the host reserve and
        ``recommendedMaxWorkingSetSize`` after the device reserve. Unknown
        limits are omitted; if every limit is unknown, the result is zero.
        """
        if not 0 < utilization <= 1:
            raise ValueError("utilization must be in (0, 1]")
        candidates = []
        physical = _positive_int(self.physical_memory_bytes)
        if physical is not None:
            candidates.append(max(0, physical - max(0, host_reserve_bytes)))
        recommended = _positive_int(
            self.metal.get("recommended_max_working_set_bytes")
        )
        if recommended is not None:
            candidates.append(
                max(0, recommended - max(0, device_reserve_bytes))
            )
        if not candidates:
            return 0
        return int(min(candidates) * utilization)


@dataclasses.dataclass(frozen=True)
class GateDecision:
    enabled: bool
    reason: str
    missing: tuple[str, ...] = ()


def gate_experiment(
    mode: str | int | bool,
    *,
    capabilities: CapabilitySnapshot | None = None,
    require_metal: bool = False,
    required_families: Iterable[str] = (),
    required_features: Iterable[str] = (),
    min_safe_unified_bytes: int = 0,
    host_reserve_bytes: int = 0,
    device_reserve_bytes: int = 0,
    min_max_buffer_bytes: int = 0,
) -> GateDecision:
    """Resolve an experiment without consulting a device marketing name."""
    normalized = str(mode).strip().lower()
    if isinstance(mode, bool):
        normalized = "on" if mode else "off"
    if normalized in FALSE_MODES:
        return GateDecision(False, "explicitly disabled")
    if normalized in FORCE_MODES:
        return GateDecision(True, "explicitly forced")
    if normalized not in TRUE_MODES:
        raise ValueError(f"unsupported experiment mode: {mode!r}")

    caps = capabilities or snapshot()
    missing = []
    if require_metal and not caps.metal.get("available"):
        missing.append("Metal device")
    for family in required_families:
        if caps.supports_family(family) is not True:
            missing.append(f"MTLGPUFamily {family}")
    for feature in required_features:
        if caps.feature(feature) is not True:
            missing.append(f"Metal feature {feature}")
    if min_safe_unified_bytes:
        safe = caps.safe_unified_budget(
            host_reserve_bytes=host_reserve_bytes,
            device_reserve_bytes=device_reserve_bytes,
        )
        if safe < min_safe_unified_bytes:
            missing.append(
                f"safe unified budget {safe} < {int(min_safe_unified_bytes)}"
            )
    if min_max_buffer_bytes:
        allows = caps.max_buffer_allows(min_max_buffer_bytes)
        if allows is not True:
            missing.append(
                f"maxBufferLength does not confirm {int(min_max_buffer_bytes)}"
            )
    if missing:
        return GateDecision(
            False,
            "capability gate failed: " + ", ".join(missing),
            tuple(missing),
        )
    return GateDecision(True, "capability gate passed")


@functools.lru_cache(maxsize=1)
def snapshot() -> CapabilitySnapshot:
    metal, error = _metal_probe()
    return CapabilitySnapshot(
        system=platform.system(),
        machine=platform.machine(),
        physical_memory_bytes=_physical_memory_bytes(),
        macos=_macos_version() if platform.system() == "Darwin" else {},
        metal=metal,
        probe_error=error,
    )


def refresh() -> CapabilitySnapshot:
    snapshot.cache_clear()
    return snapshot()


def main() -> int:
    print(json.dumps(snapshot().to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
