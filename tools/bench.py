#!/usr/bin/env python3
"""Auditable, interleaved end-to-end benchmark for Deltafin.

Every invocation creates a new result directory containing:

  campaign.json              immutable inputs and machine/repository state
  runs.jsonl                 one self-contained record per attempted run
  summary.json               aggregate statistics and correctness verdict
  run-NNN-<config>/
      stdout.log             complete child stdout
      stderr.log             complete child stderr
      events.jsonl           nanosecond events emitted by kimi_run.py
      result.json             parsed result, command, environment, and verdict

The first successful run of the first config is the exact-output oracle unless
--expect-token-ids and/or --expect-text is supplied. Configs are interleaved
(A B A B ...) so drift affects them similarly. A non-zero child exit, timeout,
missing structured result, or output mismatch makes the campaign exit non-zero.

Examples:

    python tools/bench.py
    python tools/bench.py --configs "" "K3_PILOT=0" --names default no-pilot
    python tools/bench.py --reps 5 --tokens 6 --output-dir bench-results/pilot
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import functools
import json
import math
import os
import pathlib
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

import apple_silicon


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = ROOT / "venv/bin/python"
DEFAULT_RUNNER = ROOT / "tools/kimi_run.py"
SCHEMA = "deltafin.benchmark.v1"
EVENT_SCHEMA = "deltafin.run_event.v1"
PERF_ENV_PREFIXES = (
    "K3_",
    "PYTORCH_",
    "OMP_",
    "MKL_",
    "VECLIB_",
    "METAL_",
    "MLX_",
)
DARWIN_SYSCTL_KEYS = (
    "hw.model",
    "hw.memsize",
    "hw.ncpu",
    "hw.physicalcpu",
    "hw.physicalcpu_max",
    "hw.logicalcpu",
    "hw.logicalcpu_max",
    "hw.packages",
    "hw.cpufamily",
    "hw.cputype",
    "hw.cpusubtype",
    "hw.cachelinesize",
    "hw.l1icachesize",
    "hw.l1dcachesize",
    "hw.l2cachesize",
    "hw.l3cachesize",
    "hw.perflevel0.name",
    "hw.perflevel0.physicalcpu",
    "hw.perflevel0.logicalcpu",
    "hw.perflevel0.l1icachesize",
    "hw.perflevel0.l1dcachesize",
    "hw.perflevel0.l2cachesize",
    "hw.perflevel1.name",
    "hw.perflevel1.physicalcpu",
    "hw.perflevel1.logicalcpu",
    "hw.perflevel1.l1icachesize",
    "hw.perflevel1.l1dcachesize",
    "hw.perflevel1.l2cachesize",
    "hw.optional.arm64",
    "hw.optional.neon",
    "hw.optional.armv8_1_atomics",
    "hw.optional.armv8_2_fhm",
    "hw.optional.arm.FEAT_DotProd",
    "hw.optional.arm.FEAT_FP16",
    "hw.optional.arm.FEAT_BF16",
    "hw.optional.arm.FEAT_I8MM",
    "hw.optional.arm.FEAT_SME",
    "hw.optional.arm.FEAT_SME2",
)
TORCH_PROBE_CODE = r"""
import importlib.metadata
import json
import sys

out = {"probe_python": sys.executable}
try:
    out["installed_version"] = importlib.metadata.version("torch")
except importlib.metadata.PackageNotFoundError:
    out["installed"] = False
else:
    out["installed"] = True

try:
    import torch
except Exception as exc:
    out["import_error"] = f"{type(exc).__name__}: {exc}"
else:
    out["version"] = torch.__version__
    out["debug_build"] = bool(getattr(torch.version, "debug", False))
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_state = {
        "backend_present": mps is not None,
        "is_built": bool(mps.is_built()) if mps is not None else False,
        "is_available": bool(mps.is_available()) if mps is not None else False,
    }
    dispatch_probe = getattr(
        getattr(torch, "_C", None),
        "_dispatch_has_kernel_for_dispatch_key",
        None,
    )
    if dispatch_probe is not None:
        try:
            mps_state["weight_int8pack_mm_has_mps_kernel"] = bool(
                dispatch_probe("aten::_weight_int8pack_mm", "MPS")
            )
        except Exception:
            pass
    out["mps"] = mps_state

print(json.dumps(out, sort_keys=True))
"""

DISPLAY_PROFILE_FIELDS = (
    "_name",
    "sppci_model",
    "sppci_device_type",
    "sppci_bus",
    "sppci_cores",
    "spdisplays_metal",
    "spdisplays_vram",
    "spdisplays_vram_shared",
)
NVME_CONTROLLER_FIELDS = (
    "_name",
    "spnvme_link_speed",
    "spnvme_link_width",
)
NVME_DEVICE_FIELDS = (
    "_name",
    "device_model",
    "device_revision",
    "detachable_drive",
    "spnvme_link_speed",
    "spnvme_link_width",
    "spnvme_trim_support",
)
STORAGE_PROFILE_FIELDS = (
    "_name",
    "bsd_name",
    "file_system",
    "free_space_in_bytes",
    "mount_point",
    "size_in_bytes",
    "writable",
)
PHYSICAL_DRIVE_FIELDS = (
    "device_name",
    "is_internal_disk",
    "medium_type",
    "partition_map_type",
    "protocol",
    "smart_status",
)

# Compatibility parser for old logs and alternate runners. The in-tree runner
# emits events.jsonl, which is authoritative whenever it is present.
LEGACY_TOKEN_RE = re.compile(
    r"\[token\s+(\d+):\s+([0-9]+(?:\.[0-9]+)?)s(?:[^\]]*)\]"
)
LEGACY_PREFILL_RE = re.compile(
    r"\[prefill done in\s+([0-9]+(?:\.[0-9]+)?)s\]"
)
LEGACY_TOTAL_RE = re.compile(r"\btotal\s+([0-9]+(?:\.[0-9]+)?)s\s+\|")
LEGACY_IDS_RE = re.compile(r"^token ids:\s*(\[.*\])\s*$", re.MULTILINE)
LEGACY_COMPLETION_RE = re.compile(r"^completion:\s?(.*)$", re.MULTILINE)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def write_json(path: pathlib.Path, value: Any) -> None:
    """Write deterministic, human-readable JSON and fsync it."""
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def command_output(argv: list[str], timeout: float = 3.0) -> str | None:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return text if text else None


def command_json(argv: list[str], timeout: float = 3.0) -> Any | None:
    text = command_output(argv, timeout=timeout)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A dependency may print a warning before its single-line JSON result.
        for line in reversed(text.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def selected_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in fields if key in value}


@functools.lru_cache(maxsize=1)
def macos_version_state() -> dict[str, str]:
    out = {}
    for field, flag in (
        ("product_name", "-productName"),
        ("product_version", "-productVersion"),
        ("build_version", "-buildVersion"),
    ):
        value = command_output(["sw_vers", flag])
        if value is not None:
            out[field] = value
    return out


@functools.lru_cache(maxsize=1)
def darwin_sysctl_state() -> dict[str, str]:
    """Read the allowlisted sysctls in one bounded process."""
    try:
        proc = subprocess.run(
            ["sysctl", *DARWIN_SYSCTL_KEYS],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=3.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {}
    allowed = set(DARWIN_SYSCTL_KEYS)
    out = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition(": ")
        if separator and key in allowed and value:
            out[key] = value
    return out


@functools.lru_cache(maxsize=1)
def system_profiler_state() -> dict[str, Any]:
    """Return bounded hardware facts, excluding serial numbers and raw profiles."""
    raw = command_json(
        [
            "system_profiler",
            "SPDisplaysDataType",
            "SPNVMeDataType",
            "SPStorageDataType",
            "-json",
            "-detailLevel",
            "mini",
        ],
        timeout=8.0,
    )
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    displays = raw.get("SPDisplaysDataType")
    if isinstance(displays, list):
        out["displays"] = [
            selected_fields(item, DISPLAY_PROFILE_FIELDS)
            for item in displays
            if isinstance(item, dict)
        ]

    controllers = raw.get("SPNVMeDataType")
    if isinstance(controllers, list):
        nvme = []
        for controller in controllers:
            if not isinstance(controller, dict):
                continue
            item = selected_fields(controller, NVME_CONTROLLER_FIELDS)
            devices = controller.get("_items")
            if isinstance(devices, list):
                item["devices"] = [
                    selected_fields(device, NVME_DEVICE_FIELDS)
                    for device in devices
                    if isinstance(device, dict)
                ]
            nvme.append(item)
        out["nvme_controllers"] = nvme

    volumes = raw.get("SPStorageDataType")
    if isinstance(volumes, list):
        storage = []
        for volume in volumes:
            if not isinstance(volume, dict):
                continue
            item = selected_fields(volume, STORAGE_PROFILE_FIELDS)
            physical = selected_fields(
                volume.get("physical_drive"), PHYSICAL_DRIVE_FIELDS
            )
            if physical:
                item["physical_drive"] = physical
            storage.append(item)
        out["storage"] = storage
    return out


@functools.lru_cache(maxsize=4)
def torch_state(python_executable: str) -> dict[str, Any]:
    """Probe Torch out of process so the benchmark parent never initializes MPS."""
    value = command_json(
        [python_executable, "-B", "-c", TORCH_PROBE_CODE],
        timeout=15.0,
    )
    return value if isinstance(value, dict) else {}


def git_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    rev = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if rev:
        state["commit"] = rev.splitlines()[0]
    branch = command_output(
        ["git", "-C", str(ROOT), "branch", "--show-current"]
    )
    if branch is not None:
        state["branch"] = branch
    status = command_output(
        ["git", "-C", str(ROOT), "status", "--short", "--untracked-files=normal"],
        timeout=10.0,
    )
    state["dirty"] = bool(status)
    if status:
        state["status_short"] = status.splitlines()
    return state


def lightweight_state(path: pathlib.Path = ROOT) -> dict[str, Any]:
    disk = shutil.disk_usage(path)
    out: dict[str, Any] = {
        "captured_at_utc": utc_now(),
        "time_ns": time.time_ns(),
        "cpu_count_logical": os.cpu_count(),
        "disk": {
            "path": str(path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    try:
        # Windows has no load average at all, which is an AttributeError rather
        # than the OSError a POSIX host raises when it cannot sample one.
        out["load_average_1m_5m_15m"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    # Per-run VM counters are cheap enough to sample around every child and
    # make memory experiments auditable.  In particular, a faster cache arm
    # must not be allowed to hide compressor or swap traffic between the
    # campaign-wide snapshots.
    if platform.system() == "Darwin":
        vm = command_output(["vm_stat"])
        if vm:
            out["vm_stat"] = vm
    return out


def system_state(
    torch_python: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    state = lightweight_state()
    probe_python = str(torch_python) if torch_python is not None else sys.executable
    state.update(
        {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": {
                "executable": sys.executable,
                "version": platform.python_version(),
            },
            "repository": git_state(),
            "apple_silicon": apple_silicon.snapshot().to_dict(),
        }
    )
    torch = torch_state(probe_python)
    if torch:
        state["torch"] = torch
    if platform.system() == "Darwin":
        state["sysctl"] = darwin_sysctl_state()
        macos = macos_version_state()
        if macos:
            state["macos"] = macos
        profiler = system_profiler_state()
        if profiler:
            state["system_profiler"] = profiler
        pmset = command_output(["pmset", "-g", "custom"])
        if pmset:
            state["pmset_custom"] = pmset
        vm = command_output(["vm_stat"])
        if vm:
            state["vm_stat"] = vm
    return state


def parse_env_delta(spec: str) -> dict[str, str]:
    """Parse shell-like KEY=VALUE words without invoking a shell."""
    delta: dict[str, str] = {}
    for word in shlex.split(spec):
        if "=" not in word:
            raise ValueError(
                f"invalid config word {word!r}; every word must be KEY=VALUE"
            )
        key, value = word.split("=", 1)
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment variable name {key!r}")
        delta[key] = value
    return delta


def relevant_env(env: dict[str, str]) -> dict[str, str]:
    """Record performance controls, never the process's unrelated secrets."""
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(PERF_ENV_PREFIXES)
    }


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return (value or "config")[:64]


def normalize_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def read_events(path: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return events, ["events.jsonl was not created"]
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"events.jsonl:{line_number}: {exc}")
                continue
            if not isinstance(event, dict):
                errors.append(
                    f"events.jsonl:{line_number}: event is not a JSON object"
                )
                continue
            if event.get("schema") != EVENT_SCHEMA:
                errors.append(
                    f"events.jsonl:{line_number}: unexpected schema "
                    f"{event.get('schema')!r}"
                )
            events.append(event)
    return events, errors


def parse_structured_events(
    events: list[dict[str, Any]], warmup_steps: int
) -> dict[str, Any]:
    starts = [e for e in events if e.get("event") == "run_start"]
    prefills = [e for e in events if e.get("event") == "prefill_done"]
    steps = [e for e in events if e.get("event") == "decode_step"]
    ends = [e for e in events if e.get("event") == "run_end"]
    errors: list[str] = []
    if len(starts) != 1:
        errors.append(f"expected one run_start event, found {len(starts)}")
    if len(prefills) != 1:
        errors.append(f"expected one prefill_done event, found {len(prefills)}")
    if len(ends) != 1:
        errors.append(f"expected one run_end event, found {len(ends)}")

    for i, step in enumerate(steps, 1):
        if not isinstance(step.get("duration_ns"), int) or step["duration_ns"] < 0:
            errors.append(f"decode_step {i} has invalid duration_ns")
        if not isinstance(step.get("emitted_token_ids"), list):
            errors.append(f"decode_step {i} has no emitted_token_ids list")

    end = ends[-1] if ends else {}
    prefill = prefills[-1] if prefills else {}
    all_decode_ns = sum(
        e.get("duration_ns", 0)
        for e in steps
        if isinstance(e.get("duration_ns"), int)
    )
    all_decode_tokens = sum(
        len(e.get("emitted_token_ids", []))
        for e in steps
        if isinstance(e.get("emitted_token_ids"), list)
    )
    steady = steps[warmup_steps:]
    steady_ns = sum(
        e.get("duration_ns", 0)
        for e in steady
        if isinstance(e.get("duration_ns"), int)
    )
    steady_tokens = sum(
        len(e.get("emitted_token_ids", []))
        for e in steady
        if isinstance(e.get("emitted_token_ids"), list)
    )

    return {
        "source": "structured_events",
        "parse_errors": errors,
        "runner_config": starts[-1].get("config") if starts else None,
        "runner_runtime": end.get("runtime"),
        "input_token_ids": starts[-1].get("input_token_ids") if starts else None,
        "prefill_ns": prefill.get("duration_ns"),
        "prefill_s": ns_to_s(prefill.get("duration_ns")),
        "decode_steps": steps,
        "decode_step_count": len(steps),
        "decode_ns": all_decode_ns,
        "decode_emitted_tokens": all_decode_tokens,
        "decode_tps": rate(all_decode_tokens, all_decode_ns),
        "steady_warmup_steps_dropped": min(warmup_steps, len(steps)),
        "steady_decode_ns": steady_ns,
        "steady_decode_tokens": steady_tokens,
        "steady_tps": rate(steady_tokens, steady_ns),
        "steady_s_per_token": reciprocal(rate(steady_tokens, steady_ns)),
        "emitted_token_ids": end.get("emitted_token_ids"),
        "completion_token_ids": end.get("completion_token_ids"),
        "completion_text": end.get("completion_text"),
        "runner_status": end.get("status"),
        "inference_ns": end.get("duration_ns"),
        "inference_s": ns_to_s(end.get("duration_ns")),
    }


def parse_legacy_stdout(stdout: str, warmup_steps: int) -> dict[str, Any]:
    """Best-effort parser for logs produced before structured events existed."""
    token_seconds = [float(m.group(2)) for m in LEGACY_TOKEN_RE.finditer(stdout)]
    steady = token_seconds[warmup_steps:]
    prefill_match = LEGACY_PREFILL_RE.search(stdout)
    total_match = LEGACY_TOTAL_RE.search(stdout)
    ids_match = LEGACY_IDS_RE.search(stdout)
    completion_match = LEGACY_COMPLETION_RE.search(stdout)
    ids = None
    errors = ["structured event stream unavailable; parsed legacy stdout"]
    if ids_match:
        try:
            candidate = ast.literal_eval(ids_match.group(1))
            if isinstance(candidate, list) and all(
                isinstance(x, int) for x in candidate
            ):
                ids = candidate
            else:
                errors.append("legacy token ids were not a list of integers")
        except (SyntaxError, ValueError):
            errors.append("could not parse legacy token ids")
    else:
        errors.append("legacy stdout had no token ids")
    steady_s = sum(steady)
    return {
        "source": "legacy_stdout",
        "parse_errors": errors,
        "runner_config": None,
        "input_token_ids": None,
        "prefill_ns": seconds_to_ns(
            float(prefill_match.group(1)) if prefill_match else None
        ),
        "prefill_s": (
            float(prefill_match.group(1)) if prefill_match else None
        ),
        "decode_steps": [
            {"step": i + 1, "duration_ns": seconds_to_ns(seconds)}
            for i, seconds in enumerate(token_seconds)
        ],
        "decode_step_count": len(token_seconds),
        "decode_ns": seconds_to_ns(sum(token_seconds)),
        "decode_emitted_tokens": len(token_seconds),
        "decode_tps": (
            len(token_seconds) / sum(token_seconds)
            if token_seconds and sum(token_seconds) > 0
            else None
        ),
        "steady_warmup_steps_dropped": min(warmup_steps, len(token_seconds)),
        "steady_decode_ns": seconds_to_ns(steady_s),
        "steady_decode_tokens": len(steady),
        "steady_tps": len(steady) / steady_s if steady_s > 0 else None,
        "steady_s_per_token": (
            steady_s / len(steady) if steady and steady_s > 0 else None
        ),
        "emitted_token_ids": ids,
        "completion_token_ids": ids,
        "completion_text": completion_match.group(1) if completion_match else None,
        "runner_status": None,
        "inference_ns": seconds_to_ns(
            float(total_match.group(1)) if total_match else None
        ),
        "inference_s": float(total_match.group(1)) if total_match else None,
    }


def seconds_to_ns(value: float | None) -> int | None:
    return round(value * 1_000_000_000) if value is not None else None


def ns_to_s(value: Any) -> float | None:
    return value / 1_000_000_000 if isinstance(value, int) else None


def rate(count: int, duration_ns: int) -> float | None:
    if count <= 0 or duration_ns <= 0:
        return None
    return count * 1_000_000_000 / duration_ns


def reciprocal(value: float | None) -> float | None:
    return 1.0 / value if value is not None and value > 0 else None


def numeric_stats(values: Iterable[float | int | None]) -> dict[str, Any] | None:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    med = statistics.median(xs)
    out: dict[str, Any] = {
        "count": len(xs),
        "median": med,
        "min": min(xs),
        "max": max(xs),
        "mean": statistics.fmean(xs),
        "mad": statistics.median(abs(x - med) for x in xs),
    }
    if len(xs) > 1:
        out["stdev"] = statistics.stdev(xs)
    return out


def parse_expected_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError("--expect-token-ids must be JSON [1,2] or comma-separated")
    return parsed


def compare_output(
    parsed: dict[str, Any],
    expected_ids: list[int] | None,
    expected_text: str | None,
) -> tuple[bool | None, list[str]]:
    reasons: list[str] = []
    checks = 0
    if expected_ids is not None:
        checks += 1
        if parsed.get("completion_token_ids") != expected_ids:
            reasons.append(
                "completion token IDs differ: "
                f"expected {expected_ids!r}, got "
                f"{parsed.get('completion_token_ids')!r}"
            )
    if expected_text is not None:
        checks += 1
        if parsed.get("completion_text") != expected_text:
            reasons.append(
                "completion text differs: "
                f"expected {expected_text!r}, got "
                f"{parsed.get('completion_text')!r}"
            )
    return ((not reasons) if checks else None), reasons


def performance_contract_errors(
    environment_delta: dict[str, str],
    parsed: dict[str, Any],
) -> list[str]:
    """Reject requested accelerators that silently executed their fallback.

    Exact output alone is insufficient for an optimization benchmark: a safe
    fallback can emit the right sequence while measuring two copies of the
    control. Keep this validation beside the generic runner parser so every
    future K3_INT8_KDA_QKV campaign fails closed. The historical environment
    name is retained for compatibility, but the optimized KDA unit now includes
    every eligible same-input first-stage projection, not only Q/K/V.
    """
    enabled_values = {"1", "true", "yes", "on", "force"}
    requested = (
        environment_delta.get("K3_INT8_KDA_QKV", "").strip().lower()
        in enabled_values
    )
    if not requested:
        return []
    runtime = parsed.get("runner_runtime")
    status = (
        runtime.get("int8_kda_qkv")
        if isinstance(runtime, dict) else None
    )
    if not isinstance(status, dict):
        return [
            "K3_INT8_KDA_QKV was requested but no runtime status was captured"
        ]
    errors = []
    if status.get("requested") is not True:
        errors.append("packed KDA bundle runtime did not record requested=true")
    if status.get("eligible") is not True:
        errors.append("packed KDA bundle runtime was not eligible")
    if type(status.get("controllers_installed")) is not int or (
        status["controllers_installed"] < 1
    ):
        errors.append("packed KDA bundle installed no controller")
    if status.get("enabled_at_end") is not True:
        errors.append(
            "packed KDA bundle was disabled or fell back before run end"
        )
    calls = status.get("packed_project_calls")
    if type(calls) is not int or calls <= 0:
        errors.append("packed KDA bundle executed no packed projection calls")
    controllers = status.get("controllers")
    if not isinstance(controllers, list) or not controllers:
        errors.append("packed KDA bundle reported no controller telemetry")
    else:
        required_roles = {"q", "k", "v", "f_a", "b"}
        for index, controller in enumerate(controllers):
            roles = (
                controller.get("packed_roles")
                if isinstance(controller, dict) else None
            )
            role_set = set(roles) if isinstance(roles, list) else set()
            missing = required_roles - role_set
            gate_count = len(role_set & {"g", "g_a"})
            if missing or gate_count != 1:
                details = []
                if missing:
                    details.append("missing " + ",".join(sorted(missing)))
                if gate_count != 1:
                    details.append("requires exactly one of g/g_a")
                errors.append(
                    f"packed KDA controller {index} is not the complete "
                    "same-input bundle (" + "; ".join(details) + ")"
                )
    if status.get("disable_reason") not in (None, ""):
        errors.append(
            "packed KDA bundle reported a disable reason: "
            f"{status.get('disable_reason')}"
        )
    storage_mode = environment_delta.get(
        "K3_INT8_KDA_STORAGE", "arena"
    ).strip().lower()
    stage_sync_mode = environment_delta.get(
        "K3_INT8_KDA_STAGE_SYNC", "event"
    ).strip().lower().replace("-", "_")
    if storage_mode == "stage":
        if status.get("storage_mode") != "stage":
            errors.append(
                "packed KDA stage run did not report storage_mode=stage"
            )
        if status.get("persistent_weight_bytes") != 0:
            errors.append(
                "packed KDA stage run retained a persistent int8 weight arena"
            )
        stage_bind_count = status.get("stage_bind_count")
        if type(stage_bind_count) is not int or stage_bind_count <= 0:
            errors.append("packed KDA stage run bound no upload-buffer generation")
        if status.get("stage_weight_copy_bytes") != 0:
            errors.append("packed KDA stage run copied int8 weights after upload")
        for field, description in (
            ("stage_bind_failures", "stage bind failures"),
            ("stage_stale_rejections", "stale stage generations"),
            ("stage_fence_failures", "stage fence failures"),
            ("stage_fence_sync_fallbacks", "blocking fence fallbacks"),
        ):
            if status.get(field) != 0:
                errors.append(
                    f"packed KDA stage run reported {description}"
                )
        probes = status.get("stage_full_shape_probes")
        passes = status.get("stage_full_shape_passes")
        if type(probes) is not int or probes <= 0 or passes != probes:
            errors.append(
                "packed KDA stage full-shape capability probe did not pass"
            )
        config = parsed.get("runner_config")
        device = (
            str(config.get("device", ""))
            if isinstance(config, dict) else ""
        )
        mps_fifo = (
            device == "mps" and stage_sync_mode == "mps_fifo"
        )
        for index, controller in enumerate(
            controllers if isinstance(controllers, list) else []
        ):
            if not isinstance(controller, dict):
                continue
            if controller.get("storage_mode") != "stage":
                errors.append(
                    f"packed KDA controller {index} did not use stage storage"
                )
            if controller.get("persistent_weight_bytes") != 0:
                errors.append(
                    f"packed KDA controller {index} retained int8 weights"
                )
            if controller.get("stage_bound") is not False:
                errors.append(
                    f"packed KDA controller {index} leaked a live stage binding"
                )
            expected_contract = (
                "mps_fifo"
                if (
                    stage_sync_mode == "mps_fifo"
                    and device == "mps"
                )
                else "event"
            )
            contract = controller.get("stage_sync_contract")
            if (
                (mps_fifo and contract != expected_contract)
                or (
                    not mps_fifo
                    and contract is not None
                    and contract != expected_contract
                )
            ):
                errors.append(
                    f"packed KDA controller {index} used "
                    f"stage_sync_contract={contract}, expected "
                    f"{expected_contract}"
                )
        if mps_fifo:
            if status.get("stage_sync_mode") != "mps_fifo":
                errors.append(
                    "packed KDA MPS FIFO run did not report "
                    "stage_sync_mode=mps_fifo"
                )
            records = status.get("stage_fifo_records")
            reuses = status.get("stage_fifo_reuses")
            if type(records) is not int or records <= 0:
                errors.append(
                    "packed KDA MPS FIFO run recorded no stream lease"
                )
            if type(reuses) is not int or reuses <= 0:
                errors.append(
                    "packed KDA MPS FIFO run reused no stream lease"
                )
            if status.get("stage_fence_records") != 0:
                errors.append(
                    "packed KDA MPS FIFO run unexpectedly recorded events"
                )
            if status.get("stage_fence_waits") != 0:
                errors.append(
                    "packed KDA MPS FIFO run unexpectedly waited on events"
                )
        elif device == "mps" or device.startswith("cuda"):
            records = status.get("stage_fence_records")
            waits = status.get("stage_fence_waits")
            if type(records) is not int or records <= 0:
                errors.append(
                    "packed KDA stage GPU run recorded no reuse fence"
                )
            if type(waits) is not int or waits <= 0:
                errors.append(
                    "packed KDA stage GPU run waited on no reuse fence"
                )
    return errors


def run_once(
    *,
    run_number: int,
    rep: int,
    name: str,
    env_spec: str,
    args: argparse.Namespace,
    output_dir: pathlib.Path,
    expected_ids: list[int] | None,
    expected_text: str | None,
) -> dict[str, Any]:
    run_dir = output_dir / f"run-{run_number:03d}-{slug(name)}"
    run_dir.mkdir()
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    events_path = run_dir / "events.jsonl"
    result_path = run_dir / "result.json"

    delta = parse_env_delta(env_spec)
    env = dict(os.environ)
    env.update(delta)
    command = [
        str(args.python),
        str(args.runner),
        "--prompt",
        args.prompt,
        "--max-new",
        str(args.tokens),
        "--events-jsonl",
        str(events_path),
    ]
    if args.chat:
        command.append("--chat")

    started_utc = utc_now()
    state_before = lightweight_state()
    started_ns = time.perf_counter_ns()
    timeout = False
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=args.timeout,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode: int | None = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timeout = True
        stdout = normalize_timeout_output(exc.stdout)
        stderr = normalize_timeout_output(exc.stderr)
        returncode = None
    wall_ns = time.perf_counter_ns() - started_ns
    state_after = lightweight_state()

    with stdout_path.open("x", encoding="utf-8") as f:
        f.write(stdout)
    with stderr_path.open("x", encoding="utf-8") as f:
        f.write(stderr)

    events, event_file_errors = read_events(events_path)
    if events:
        parsed = parse_structured_events(events, args.warmup_steps)
        parsed["parse_errors"] = event_file_errors + parsed["parse_errors"]
    else:
        parsed = parse_legacy_stdout(stdout, args.warmup_steps)
        parsed["parse_errors"] = event_file_errors + parsed["parse_errors"]

    output_match, mismatch_reasons = compare_output(
        parsed, expected_ids, expected_text
    )
    hard_errors: list[str] = []
    if timeout:
        hard_errors.append(f"runner timed out after {args.timeout:g}s")
    elif returncode != 0:
        hard_errors.append(f"runner exited with status {returncode}")
    if parsed["source"] != "structured_events":
        hard_errors.append("runner did not produce a structured event stream")
    if parsed.get("runner_status") not in (None, "ok"):
        hard_errors.append(f"runner status is {parsed.get('runner_status')!r}")
    if parsed.get("completion_token_ids") is None:
        hard_errors.append("no exact completion token IDs were captured")
    if parsed.get("completion_text") is None:
        hard_errors.append("no exact completion text was captured")
    if parsed.get("prefill_ns") is None:
        hard_errors.append("no prefill timing was captured")
    hard_errors.extend(parsed.get("parse_errors", []))
    if output_match is False:
        hard_errors.extend(mismatch_reasons)
    hard_errors.extend(performance_contract_errors(delta, parsed))

    record = {
        "schema": SCHEMA,
        "record_type": "run",
        "run_number": run_number,
        "repetition": rep,
        "config_name": name,
        "config_spec": env_spec,
        "environment_delta": delta,
        "effective_performance_environment": relevant_env(env),
        "command": command,
        "cwd": str(ROOT),
        "started_at_utc": started_utc,
        "finished_at_utc": utc_now(),
        "wall_ns": wall_ns,
        "wall_s": wall_ns / 1_000_000_000,
        "returncode": returncode,
        "timed_out": timeout,
        "state_before": state_before,
        "state_after": state_after,
        "artifacts": {
            "run_dir": str(run_dir),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "events": str(events_path),
            "result": str(result_path),
        },
        "parsed": parsed,
        "expected_completion_token_ids": expected_ids,
        "expected_completion_text": expected_text,
        "output_match": output_match,
        "valid": not hard_errors,
        "errors": hard_errors,
        "stderr_tail": stderr[-2000:] if stderr else "",
    }
    return record


def summarize(
    records: list[dict[str, Any]], names: list[str]
) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    baseline_tps = None
    for name in names:
        runs = [r for r in records if r["config_name"] == name]
        valid = [r for r in runs if r["valid"]]
        metrics = {
            "wall_s": numeric_stats(r["wall_s"] for r in valid),
            "prefill_s": numeric_stats(
                r["parsed"].get("prefill_s") for r in valid
            ),
            "inference_s": numeric_stats(
                r["parsed"].get("inference_s") for r in valid
            ),
            "decode_tps": numeric_stats(
                r["parsed"].get("decode_tps") for r in valid
            ),
            "steady_tps": numeric_stats(
                r["parsed"].get("steady_tps") for r in valid
            ),
            "steady_s_per_token": numeric_stats(
                r["parsed"].get("steady_s_per_token") for r in valid
            ),
        }
        med_tps = (
            metrics["steady_tps"]["median"] if metrics["steady_tps"] else None
        )
        if baseline_tps is None and med_tps is not None:
            baseline_tps = med_tps
        configs[name] = {
            "attempted_runs": len(runs),
            "valid_runs": len(valid),
            "invalid_runs": len(runs) - len(valid),
            "all_exact_output_match": bool(runs)
            and all(r["output_match"] is True for r in runs),
            "metrics": metrics,
            "relative_to_first_config": (
                med_tps / baseline_tps
                if med_tps is not None and baseline_tps
                else None
            ),
            "run_numbers": [r["run_number"] for r in runs],
        }

    # Conservative noise warning: min-max steady-TPS intervals overlap.
    if len(names) > 1:
        base_stats = configs[names[0]]["metrics"]["steady_tps"]
        for name in names[1:]:
            stats = configs[name]["metrics"]["steady_tps"]
            if base_stats and stats:
                configs[name]["steady_tps_range_overlaps_baseline"] = not (
                    stats["min"] > base_stats["max"]
                    or stats["max"] < base_stats["min"]
                )

    errors = [
        {
            "run_number": r["run_number"],
            "config_name": r["config_name"],
            "errors": r["errors"],
        }
        for r in records
        if not r["valid"]
    ]
    return {
        "schema": SCHEMA,
        "record_type": "summary",
        "generated_at_utc": utc_now(),
        "attempted_runs": len(records),
        "valid_runs": sum(r["valid"] for r in records),
        "all_runs_valid": bool(records) and all(r["valid"] for r in records),
        "all_outputs_exact": bool(records)
        and all(r["output_match"] is True for r in records),
        "configs": configs,
        "errors": errors,
    }


def print_run(record: dict[str, Any]) -> None:
    parsed = record["parsed"]
    prefill = parsed.get("prefill_s")
    tps = parsed.get("steady_tps")
    status = "ok" if record["valid"] else "INVALID"
    prefill_text = f"{prefill:.6f}s" if prefill is not None else "-"
    tps_text = f"{tps:.6f} tok/s" if tps is not None else "-"
    print(
        f"  rep{record['repetition']} {record['config_name']:<28} "
        f"prefill {prefill_text:>14} steady {tps_text:>18}  {status}",
        flush=True,
    )
    for error in record["errors"]:
        print(f"    ERROR: {error}", flush=True)
    if record["stderr_tail"] and not record["valid"]:
        tail = record["stderr_tail"].replace("\n", "\n      ")
        print(f"    stderr tail:\n      {tail}", flush=True)


def print_summary(summary: dict[str, Any], names: list[str]) -> None:
    print("\n" + "=" * 92)
    print(
        f"{'config':28} {'valid':>7} {'prefill s':>14} "
        f"{'steady tok/s':>16} {'range':>19} {'relative':>10}"
    )
    print("=" * 92)
    for name in names:
        cfg = summary["configs"][name]
        p = cfg["metrics"]["prefill_s"]
        t = cfg["metrics"]["steady_tps"]
        prefill = f"{p['median']:.6f}" if p else "-"
        tps = f"{t['median']:.6f}" if t else "-"
        spread = f"{t['min']:.6f}-{t['max']:.6f}" if t else "-"
        rel = cfg["relative_to_first_config"]
        relative = f"{rel:.4f}x" if rel is not None else "-"
        valid = f"{cfg['valid_runs']}/{cfg['attempted_runs']}"
        print(
            f"{name:28} {valid:>7} {prefill:>14} {tps:>16} "
            f"{spread:>19} {relative:>10}"
        )
    print("=" * 92)
    if summary["all_runs_valid"] and summary["all_outputs_exact"]:
        print("Every run succeeded and matched the exact token-ID/text oracle.")
    else:
        print("INVALID CAMPAIGN: failed runs or exact-output mismatches are present.")


def make_output_dir(requested: str | None) -> pathlib.Path:
    if requested:
        path = pathlib.Path(requested).expanduser()
        if not path.is_absolute():
            path = ROOT / path
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = ROOT / "bench-results" / f"{stamp}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[""],
        help='environment deltas, e.g. "" "K3_PILOT=0 K3_MOE=cpu"',
    )
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="decode passes omitted from steady-state TPS (default: 1)",
    )
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument(
        "--output-dir",
        help="new directory for evidence; default: bench-results/<UTC>-<pid>",
    )
    ap.add_argument(
        "--expect-token-ids",
        help='exact completion oracle as JSON ("[1,2]") or "1,2"',
    )
    ap.add_argument("--expect-text", help="exact completion text oracle")
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="attempt later runs after a child process/capture failure",
    )
    ap.add_argument(
        "--runner",
        type=pathlib.Path,
        default=DEFAULT_RUNNER,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--python",
        type=pathlib.Path,
        default=DEFAULT_PYTHON,
        help=argparse.SUPPRESS,
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.reps < 1:
        ap.error("--reps must be at least 1")
    if args.tokens < 1:
        ap.error("--tokens must be at least 1")
    if args.warmup_steps < 0:
        ap.error("--warmup-steps cannot be negative")
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    names = args.names or [c or "defaults" for c in args.configs]
    if len(names) != len(args.configs):
        ap.error("--names must have exactly one entry per --configs entry")
    if len(set(names)) != len(names):
        ap.error("--names must be unique")
    try:
        deltas = [parse_env_delta(spec) for spec in args.configs]
        expected_ids = parse_expected_ids(args.expect_token_ids)
        output_dir = make_output_dir(args.output_dir)
    except (ValueError, FileExistsError) as exc:
        ap.error(str(exc))
    if not args.python.exists():
        ap.error(f"Python executable does not exist: {args.python}")
    if not args.runner.exists():
        ap.error(f"runner does not exist: {args.runner}")

    campaign = {
        "schema": SCHEMA,
        "record_type": "campaign",
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "root": str(ROOT),
        "arguments": {
            "configs": args.configs,
            "names": names,
            "parsed_environment_deltas": deltas,
            "repetitions": args.reps,
            "max_new_tokens": args.tokens,
            "prompt": args.prompt,
            "chat": args.chat,
            "warmup_steps": args.warmup_steps,
            "timeout_s": args.timeout,
            "expected_completion_token_ids": expected_ids,
            "expected_completion_text": args.expect_text,
            "keep_going": args.keep_going,
            "runner": str(args.runner),
            "python": str(args.python),
        },
        "base_performance_environment": relevant_env(dict(os.environ)),
        "system_at_start": system_state(args.python),
    }
    write_json(output_dir / "campaign.json", campaign)
    print(f"evidence: {output_dir}", flush=True)

    records: list[dict[str, Any]] = []
    oracle_ids = expected_ids
    oracle_text = args.expect_text
    stopped_early = False
    run_number = 0
    for rep in range(1, args.reps + 1):
        for name, spec in zip(names, args.configs):
            run_number += 1
            record = run_once(
                run_number=run_number,
                rep=rep,
                name=name,
                env_spec=spec,
                args=args,
                output_dir=output_dir,
                expected_ids=oracle_ids,
                expected_text=oracle_text,
            )

            # The first valid reference-config run becomes the exact oracle.
            # Validate it against itself, then persist the amended result.
            if (
                oracle_ids is None
                and oracle_text is None
                and name == names[0]
                and record["valid"]
            ):
                oracle_ids = record["parsed"]["completion_token_ids"]
                oracle_text = record["parsed"]["completion_text"]
                record["expected_completion_token_ids"] = oracle_ids
                record["expected_completion_text"] = oracle_text
                record["output_match"] = True
            elif (
                oracle_ids is None
                and oracle_text is None
                and record["valid"]
            ):
                record["valid"] = False
                record["errors"].append(
                    "no exact-output oracle is available; a successful run of "
                    f"reference config {names[0]!r} must come first"
                )

            write_json(pathlib.Path(record["artifacts"]["result"]), record)
            records.append(record)
            append_jsonl(output_dir / "runs.jsonl", record)
            print_run(record)
            if not record["valid"] and not args.keep_going:
                stopped_early = True
                break
        if stopped_early:
            break

    summary = summarize(records, names)
    summary["stopped_early"] = stopped_early
    summary["exact_oracle"] = {
        "completion_token_ids": oracle_ids,
        "completion_text": oracle_text,
        "source": (
            "command_line"
            if expected_ids is not None or args.expect_text is not None
            else "first_valid_reference_run"
        ),
    }
    summary["system_at_end"] = system_state(args.python)
    write_json(output_dir / "summary.json", summary)
    print_summary(summary, names)
    print(f"raw evidence and summary: {output_dir}", flush=True)
    return 0 if summary["all_runs_valid"] and summary["all_outputs_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
