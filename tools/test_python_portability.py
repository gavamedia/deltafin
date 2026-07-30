#!/usr/bin/env python3
"""Weight-free portability gates for native loading, I/O hints, and devices."""

import ast
import contextlib
import importlib.util
import os
import pathlib
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runtime_platform as rp  # noqa: E402
import pilot  # noqa: E402
import spine_cache  # noqa: E402
import spine_io  # noqa: E402

BASE_FLAGS = {
    next(iter(aliases))
    for aliases in rp._X86_64_BASE_FEATURES.values()
}


class FakeFunction:
    def __init__(self, result=None):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.result


class FakeLibrary:
    def __init__(self, symbols=(), abi=rp.NATIVE_ABI_VERSION):
        self.mxfp4_abi_version = FakeFunction(abi)
        for name in symbols:
            setattr(self, name, FakeFunction())


def _native_loader_checks():
    assert rp.native_library_filename("libx", "darwin") == "libx.dylib"
    assert rp.native_library_filename("libx", "linux") == "libx.so"
    assert rp.native_library_filename("libx", "linux-musl") == "libx.so"
    assert rp.native_library_filename("libx", "win32") == "libx.dll"
    try:
        rp.native_library_filename("libx", "freebsd14")
    except rp.NativeLibraryError:
        pass
    else:
        raise AssertionError("unknown native platform was accepted")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "libdemo.so")
        pathlib.Path(path).touch()
        with mock.patch.dict(os.environ, {"TEST_NATIVE_LIB": path}):
            lib, got = rp.load_native_library(
                td,
                "libdemo",
                env_var="TEST_NATIVE_LIB",
                required_symbols=("run",),
                platform="linux",
                machine="aarch64",
                cdll_factory=lambda _path: FakeLibrary(("run",)),
            )
            assert got == path and hasattr(lib, "run")

            for fake, expected in (
                (FakeLibrary(()), "missing"),
                (FakeLibrary(("run",), abi=99), "found 99"),
            ):
                try:
                    rp.load_native_library(
                        td,
                        "libdemo",
                        env_var="TEST_NATIVE_LIB",
                        required_symbols=("run",),
                        platform="linux",
                        machine="aarch64",
                        cdll_factory=lambda _path, fake=fake: fake,
                    )
                except rp.NativeLibraryError as exc:
                    assert expected in str(exc)
                else:
                    raise AssertionError("incompatible native library was accepted")

        cpuinfo = os.path.join(td, "cpuinfo")
        pathlib.Path(cpuinfo).write_text(
            "processor: 0\nflags: " + " ".join(sorted(BASE_FLAGS)) + "\n"
            "processor: 1\nflags: " + " ".join(sorted(BASE_FLAGS | {"avx2"})) + "\n",
            encoding="utf-8",
        )
        assert rp.missing_native_cpu_features(
            platform="linux", machine="x86_64", cpuinfo_path=cpuinfo
        ) == ()
        pathlib.Path(cpuinfo).write_text(
            "processor: 0\nflags: "
            + " ".join(sorted(BASE_FLAGS - {"fma"})) + "\n",
            encoding="utf-8",
        )
        assert rp.missing_native_cpu_features(
            platform="linux", machine="x86_64", cpuinfo_path=cpuinfo
        ) == ("fma",)

        # Windows answers through IsProcessorFeaturePresent instead of
        # /proc/cpuinfo, and infers FMA3 from AVX2 because it cannot report it.
        full_windows = {
            rp._PF_SSE3_INSTRUCTIONS_AVAILABLE,
            rp._PF_SSSE3_INSTRUCTIONS_AVAILABLE,
            rp._PF_AVX_INSTRUCTIONS_AVAILABLE,
            rp._PF_AVX2_INSTRUCTIONS_AVAILABLE,
        }
        assert rp.missing_native_cpu_features(
            platform="win32",
            machine="AMD64",
            feature_probe=lambda item: item in full_windows,
            environ={},
        ) == ()
        no_avx2 = full_windows - {rp._PF_AVX2_INSTRUCTIONS_AVAILABLE}
        assert rp.missing_native_cpu_features(
            platform="win32",
            machine="AMD64",
            feature_probe=lambda item: item in no_avx2,
            environ={},
        ) == ("fma",)
        # The override exists for AVX+FMA3 parts that predate AVX2.
        assert rp.missing_native_cpu_features(
            platform="win32",
            machine="AMD64",
            feature_probe=lambda item: item in no_avx2,
            environ={"K3_ASSUME_FMA3": "1"},
        ) == ()
        assert rp.missing_native_cpu_features(
            platform="win32",
            machine="AMD64",
            feature_probe=lambda _item: False,
            environ={},
        ) == ("avx", "fma", "sse3", "ssse3")
        # A non-x86 host is never gated on x86 baseline features.
        assert rp.missing_native_cpu_features(
            platform="win32",
            machine="ARM64",
            feature_probe=lambda _item: False,
            environ={},
        ) == ()
        with mock.patch.dict(os.environ, {"TEST_NATIVE_LIB": path}):
            try:
                rp.load_native_library(
                    td,
                    "libdemo",
                    env_var="TEST_NATIVE_LIB",
                    required_symbols=("run",),
                    platform="linux",
                    machine="x86_64",
                    cpuinfo_path=cpuinfo,
                    cdll_factory=lambda _path: FakeLibrary(("run",)),
                )
            except rp.NativeLibraryError as exc:
                assert "AVX/FMA3/SSSE3 baseline" in str(exc) and "fma" in str(exc)
            else:
                raise AssertionError("unsafe x86 native library was loaded")


def _load_module_with_fake_native(filename):
    calls = []

    def fake_loader(directory, stem, **kwargs):
        calls.append((directory, stem, kwargs))
        return FakeLibrary(kwargs["required_symbols"]), f"/fake/{stem}"

    name = f"_portability_{filename.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(rp, "load_native_library", fake_loader):
        spec.loader.exec_module(module)
    return module, calls


def _native_module_manifest_checks():
    fast, calls = _load_module_with_fake_native("fast_moe.py")
    assert len(calls) == 1
    assert calls[0][1] == "libmxfp4gemv"
    assert calls[0][2]["env_var"] == "K3_GEMV_LIB"
    assert tuple(calls[0][2]["required_symbols"]) == ("mxfp4_gemv_mt",)
    assert fast._LIB_PATH == "/fake/libmxfp4gemv"

    batch, calls = _load_module_with_fake_native("fast_moe_batch.py")
    assert len(calls) == 1
    assert calls[0][1] == "libmxfp4batch"
    assert calls[0][2]["env_var"] == "K3_BATCH_LIB"
    assert {
        "mxfp4_gemv_batch",
        "mxfp4_moe_expert_set",
        "mxfp4_pool_init",
        "mxfp4_pool_shutdown",
    }.issubset(calls[0][2]["required_symbols"])
    assert batch._LIB_PATH == "/fake/libmxfp4batch"

    imported = []
    assert rp.import_when_enabled(
        False, "fast_moe", importer=lambda name: imported.append(name)
    ) is None
    assert imported == []

    # Wiring gate: the heavyweight runner is parsed, never imported.
    source = (HERE / "kimi_run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct = [
        node for node in tree.body
        if (isinstance(node, ast.Import)
            and any(alias.name == "fast_moe" for alias in node.names))
    ]
    assert not direct
    assert 'import_when_enabled(FAST_MOE, "fast_moe")' in source
    assert "torch.mps.synchronize" not in source
    assert 'if MOE_BACKEND == "cuda":' in source
    assert "cuda_moe.available(DEV)" in source
    assert "DT != torch.float32" in source
    assert "layer_index=li" in source
    assert "model_key=_CUDA_MODEL_KEY" in source
    assert "spine_fast.effective_dequant_backend(DEV)" in source
    assert 'DEV.type == "mps" and _SPINE_DEQ == "metal"' in source
    assert "spine_fast.describe(DEV)" in source
    # Definition + two CUDA-MoE profile gates + two whole-pass profile gates.
    # Normal decode must not add an unconditional accelerator synchronization.
    assert source.count("_device_synchronize()") == 5


def _cuda_dequant_checks():
    """Exercise the lazy CUDA seam without importing the heavyweight module."""
    source_path = HERE / "spine_fast.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_cuda_dequant_requested_for_config",
            "_effective_deq_for_config",
            "cuda_dequant_into",
            "cuda_dequant_error",
        }
    ]
    calls = []

    class FakeCuda:
        def available(self, device):
            calls.append(("available", device.type, device.index))
            return True

        def describe(self, device):
            return f"fake cuda:{device.index}"

        def dequant_int8_into(self, dst, q, sc):
            calls.append(("dequant", dst, q, sc))
            return True

        def disable(self, device, reason):
            calls.append(("disable", device.index, reason))

    namespace = {
        "_cuda_deq_module": FakeCuda(),
        "_cuda_deq_devices": set(),
        "_cuda_deq_errors": {},
        "_cuda_deq_lock": threading.Lock(),
        "_cuda_deq_requested": True,
    }
    isolated = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(source_path), "exec"), namespace)
    requested = namespace["_cuda_dequant_requested_for_config"]
    assert requested(False, "torch", False, "auto", False) is False
    assert requested(False, "torch", False, "1", True) is True
    assert requested(False, "cuda", True, "auto", False) is True
    assert requested(True, "torch", True, "1", True) is False
    assert requested(True, "cuda", True, "off", True) is False
    effective = namespace["_effective_deq_for_config"]
    assert effective("metal", False, True, True, "cuda") == "cuda"
    assert effective("metal", False, True, False, "cuda") == "torch"
    assert effective("metal", False, False, True, "cuda") == "torch"
    assert effective("metal", False, True, True, "mps") == "metal"
    assert effective("metal", False, True, True, "cpu") == "torch"
    assert effective("mulout", True, False, True, "cpu") == "mulout"
    dequant = namespace["cuda_dequant_into"]

    cpu = SimpleNamespace(device=SimpleNamespace(type="cpu", index=None))
    assert dequant(cpu, object(), object()) is False
    assert calls == []  # CPU/MPS never qualify or touch the CUDA bridge.

    cuda_device = SimpleNamespace(type="cuda", index=1)
    dst = SimpleNamespace(device=cuda_device)
    q, sc = object(), object()
    assert dequant(dst, q, sc) is True
    assert dequant(dst, q, sc) is True
    assert [call[0] for call in calls].count("available") == 1
    assert [call[0] for call in calls].count("dequant") == 2

    failure_calls = []

    def fail(_dst, _q, _sc):
        failure_calls.append("failure")
        raise RuntimeError("synthetic launch failure")

    namespace["_cuda_deq_module"].dequant_int8_into = fail
    assert dequant(dst, q, sc) is False
    assert "synthetic launch failure" in (
        namespace["cuda_dequant_error"](cuda_device)
    )
    prior = list(calls)
    assert dequant(dst, q, sc) is False
    assert calls == prior
    assert failure_calls == ["failure"]

    rejected_device = SimpleNamespace(type="cuda", index=2)
    rejected = SimpleNamespace(device=rejected_device)
    reject_calls = []

    def reject(_dst, _q, _sc):
        reject_calls.append("reject")
        return False

    namespace["_cuda_deq_module"].dequant_int8_into = reject
    assert dequant(rejected, q, sc) is False
    assert dequant(rejected, q, sc) is False
    assert reject_calls == ["reject"]  # one rejection opens the local circuit.
    assert not any(call[0] == "disable" for call in calls)


def _prefetch_residency_filter_checks():
    """A GPU-resident expert must not be read speculatively from disk."""
    saved_pref = dict(pilot._PREF)
    saved_bin, saved_npz = pilot.BIN, pilot.NPZ
    saved_counts = {
        key: pilot.STATS[key]
        for key in ("issued_layers", "issued_bin", "issued_npz")
    }
    issued = []
    fetch = SimpleNamespace(
        _has_bin=lambda _layer, _eid: True,
        prefetch_layer=lambda layer, ids: issued.append((layer, list(ids))),
    )
    try:
        pilot.BIN, pilot.NPZ = True, False
        pilot._PREF.clear()
        pilot._PREF[12] = [2, 4, 6, 8]
        pilot.issue_prefetch(
            12,
            fetch,
            pread=True,
            filter_ids=lambda layer, ids: (
                [expert for expert in ids if expert >= 6]
                if layer == 12 else ids
            ),
        )
        assert issued == [(12, [6, 8])]
        assert pilot.STATS["issued_bin"] == saved_counts["issued_bin"] + 2

        pilot._PREF[13] = [1, 3]
        before_layers = pilot.STATS["issued_layers"]
        pilot.issue_prefetch(
            13, fetch, pread=True, filter_ids=lambda _layer, _ids: [])
        assert pilot.STATS["issued_layers"] == before_layers
        assert issued == [(12, [6, 8])]
    finally:
        pilot._PREF.clear()
        pilot._PREF.update(saved_pref)
        pilot.BIN, pilot.NPZ = saved_bin, saved_npz
        for key, value in saved_counts.items():
            pilot.STATS[key] = value


def _device_checks():
    assert rp.default_gemv_threads("darwin", 10) == 4
    assert rp.default_gemv_threads("linux", 32) == 8
    assert rp.default_gemv_threads("linux", 6) == 6
    # Windows shares the x86-64 default rather than the Apple Silicon one.
    assert rp.default_gemv_threads("win32", 32) == 8
    assert rp.default_gemv_threads("win32", 3) == 3
    assert rp.default_gemv_threads("freebsd", 2) == 2
    assert rp.choose_device_spec(
        None, mps_available=True, cuda_available=True, cuda_device_count=2
    ) == "mps"
    assert rp.choose_device_spec(
        None, mps_available=False, cuda_available=True, cuda_device_count=2
    ) == "cuda"
    assert rp.choose_device_spec(
        None,
        mps_available=False,
        cuda_available=True,
        cuda_device_count=2,
        auto_cuda_index=1,
    ) == "cuda:1"
    assert rp.choose_device_spec(
        None, mps_available=False, cuda_available=False
    ) == "cpu"
    assert rp.choose_device_spec(
        "cuda:1", mps_available=False, cuda_available=True,
        cuda_device_count=2,
    ) == "cuda:1"
    for requested, error in (
        ("mps", RuntimeError),
        ("cuda", RuntimeError),
        ("cuda:2", ValueError),
        ("xpu", ValueError),
    ):
        try:
            rp.choose_device_spec(
                requested,
                mps_available=False,
                cuda_available=requested.startswith("cuda"),
                cuda_device_count=2 if requested == "cuda:2" else 0,
            )
        except error:
            pass
        else:
            raise AssertionError(f"invalid explicit device {requested} accepted")
    try:
        rp.choose_device_spec(
            None,
            mps_available=False,
            cuda_available=True,
            cuda_device_count=2,
            auto_cuda_index=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range automatic CUDA index was accepted")

    assert rp.normalize_requested_device(None) is None
    assert rp.normalize_requested_device("  ") is None
    assert rp.normalize_requested_device(" CUDA:1 ") == "cuda:1"
    for malformed in ("cudax", "cuda:", "cuda:-1", "cuda:1x", "metal"):
        try:
            rp.normalize_requested_device(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"malformed device request {malformed!r} was accepted"
            )

    calls = []
    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(synchronize=lambda: calls.append(("mps", None))),
        cuda=SimpleNamespace(
            synchronize=lambda device: calls.append(("cuda", device))
        ),
    )
    mps = SimpleNamespace(type="mps")
    cuda = SimpleNamespace(type="cuda")
    cpu = SimpleNamespace(type="cpu")
    rp.synchronize_device(fake_torch, mps)
    rp.synchronize_device(fake_torch, cuda)
    rp.synchronize_device(fake_torch, cpu)
    assert calls == [("mps", None), ("cuda", cuda)]
    assert rp.choose_moe_backend(None, "mps") == "metal"
    assert rp.choose_moe_backend(None, "cuda") == "cuda"
    assert rp.choose_moe_backend(None, "cpu") == "cpu"
    assert rp.choose_moe_backend("metal", "cuda") == "metal"
    assert rp.choose_moe_backend("cuda", "cuda") == "cuda"
    try:
        rp.choose_moe_backend("vulkan", "cuda")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported MoE backend was accepted")


def _cpu_availability_checks():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        proc = root / "proc"
        cgroup = root / "cgroup"
        (proc / "self").mkdir(parents=True)

        # cgroup v2: a fractional leaf quota wins over host count and affinity.
        leaf = cgroup / "team" / "job"
        leaf.mkdir(parents=True)
        cgroup_file = proc / "self" / "cgroup"
        cgroup_file.write_text("0::/team/job\n", encoding="utf-8")
        (leaf / "cpu.max").write_text("250000 100000\n", encoding="utf-8")
        (cgroup / "team" / "cpu.max").write_text(
            "600000 100000\n", encoding="utf-8"
        )
        assert rp.linux_cpu_quota_count(
            self_cgroup_path=str(cgroup_file),
            cgroup_root=str(cgroup),
        ) == 3
        assert rp.available_cpu_count(
            platform="linux",
            cpu_count=64,
            affinity_getter=lambda _pid: range(12),
            self_cgroup_path=str(cgroup_file),
            cgroup_root=str(cgroup),
        ) == 3

        # An unlimited leaf does not hide a tighter ancestor. The quota rounds
        # 1.5 CPUs up to two workers, then affinity can constrain it further.
        (leaf / "cpu.max").write_text("max 100000\n", encoding="utf-8")
        (cgroup / "team" / "cpu.max").write_text(
            "150000 100000\n", encoding="utf-8"
        )
        assert rp.linux_cpu_quota_count(
            self_cgroup_path=str(cgroup_file),
            cgroup_root=str(cgroup),
        ) == 2
        assert rp.available_cpu_count(
            platform="linux",
            cpu_count=64,
            affinity_getter=lambda _pid: range(1),
            self_cgroup_path=str(cgroup_file),
            cgroup_root=str(cgroup),
        ) == 1

        # v1 combined-controller declaration with the conventional cpu mount.
        v1_leaf = cgroup / "cpu" / "batch" / "job"
        v1_leaf.mkdir(parents=True)
        cgroup_file.write_text(
            "4:cpu,cpuacct:/batch/job\n", encoding="utf-8"
        )
        (v1_leaf / "cpu.cfs_quota_us").write_text(
            "-1\n", encoding="utf-8"
        )
        (v1_leaf / "cpu.cfs_period_us").write_text(
            "100000\n", encoding="utf-8"
        )
        v1_parent = cgroup / "cpu" / "batch"
        (v1_parent / "cpu.cfs_quota_us").write_text(
            "350000\n", encoding="utf-8"
        )
        (v1_parent / "cpu.cfs_period_us").write_text(
            "100000\n", encoding="utf-8"
        )
        assert rp.linux_cpu_quota_count(
            self_cgroup_path=str(cgroup_file),
            cgroup_root=str(cgroup),
        ) == 4

        # A declared path cannot escape the injected cgroup root.
        cgroup_file.write_text("0::/../../outside\n", encoding="utf-8")
        assert rp.linux_cpu_quota_count(
            self_cgroup_path=str(cgroup_file),
            cgroup_root=str(cgroup),
        ) is None

        # Non-Linux hosts still honor process affinity but never inspect Linux
        # quota files.
        assert rp.available_cpu_count(
            platform="darwin",
            cpu_count=12,
            affinity_getter=lambda _pid: range(6),
            self_cgroup_path=str(cgroup_file),
            cgroup_root=str(cgroup),
        ) == 6

    with mock.patch.object(rp, "available_cpu_count", return_value=3):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert rp.configured_cpu_workers("TEST_WORKERS", 8) == 3
            assert rp.configured_cpu_workers("TEST_WORKERS", 2) == 2
        with mock.patch.dict(
            os.environ, {"TEST_WORKERS": "17"}, clear=True
        ):
            assert rp.configured_cpu_workers("TEST_WORKERS", 8) == 17
        for invalid in ("0", "-2", "many"):
            with mock.patch.dict(
                os.environ, {"TEST_WORKERS": invalid}, clear=True
            ):
                try:
                    rp.configured_cpu_workers("TEST_WORKERS", 8)
                except ValueError as exc:
                    assert "positive integer" in str(exc)
                else:
                    raise AssertionError(
                        f"invalid worker override {invalid!r} was accepted"
                    )


def _memory_checks():
    gib = rp.GIB
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        proc = root / "proc"
        cg = root / "cgroup"
        (proc / "self").mkdir(parents=True)
        (cg / "demo").mkdir(parents=True)
        (proc / "meminfo").write_text(
            "MemTotal: 131072000 kB\nMemAvailable: 104857600 kB\n",
            encoding="utf-8",
        )
        (proc / "self" / "cgroup").write_text(
            "0::/demo\n", encoding="utf-8"
        )
        (cg / "demo" / "memory.max").write_text(
            str(64 * gib), encoding="utf-8"
        )
        (cg / "demo" / "memory.current").write_text(
            str(20 * gib), encoding="utf-8"
        )
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 64 * gib
        assert memory.effective_available_bytes == 44 * gib
        assert rp.safe_linux_host_budget(memory, 10 * gib) == 34 * gib

        # A constrained parent must win over an unlimited leaf.
        parent = cg / "parent"
        leaf = parent / "leaf"
        leaf.mkdir(parents=True)
        (proc / "self" / "cgroup").write_text(
            "0::/parent/leaf\n", encoding="utf-8"
        )
        (leaf / "memory.max").write_text("max", encoding="utf-8")
        (leaf / "memory.current").write_text(
            str(2 * gib), encoding="utf-8"
        )
        (parent / "memory.max").write_text(
            str(48 * gib), encoding="utf-8"
        )
        (parent / "memory.current").write_text(
            str(30 * gib), encoding="utf-8"
        )
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 48 * gib
        assert memory.effective_available_bytes == 18 * gib

        # Finite capacity with unreadable usage is not treated as free RAM.
        (parent / "memory.current").unlink()
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_available_bytes == 0

        (parent / "memory.max").write_text("0", encoding="utf-8")
        (parent / "memory.current").write_text("0", encoding="utf-8")
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 0
        assert memory.effective_available_bytes == 0

        # A declared but unreadable controller must not fall back to host RAM.
        (parent / "memory.max").unlink()
        (parent / "memory.max").mkdir()
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_total_bytes == 0
        assert memory.effective_available_bytes == 0

        # Host exhaustion is meaningful even when a cgroup has room.
        (parent / "memory.max").rmdir()
        (parent / "memory.max").write_text(
            str(48 * gib), encoding="utf-8"
        )
        (proc / "meminfo").write_text(
            "MemTotal: 131072000 kB\nMemAvailable: 0 kB\n",
            encoding="utf-8",
        )
        memory = rp.linux_memory_limits(
            meminfo_path=str(proc / "meminfo"),
            self_cgroup_path=str(proc / "self" / "cgroup"),
            cgroup_root=str(cg),
        )
        assert memory.effective_available_bytes == 0

    cap = rp.cuda_free_memory_budget(20 * gib, 24 * gib)
    assert 14 * gib < cap < 16 * gib


def _patchable_fcntl(module):
    """Give a fcntl-less host something to patch in place of the module.

    Windows has no fcntl.  The assertions below are worth running there anyway:
    what they check is that a non-Darwin host never issues a Darwin command
    number, which is exactly what a fcntl-less host has to get right.
    """
    if getattr(module, "fcntl", None) is not None:
        return contextlib.nullcontext()
    return mock.patch.object(module, "fcntl", SimpleNamespace(fcntl=None))


def _file_hint_checks():
    assert rp.darwin_file_hints_enabled(True, "darwin")
    assert not rp.darwin_file_hints_enabled(True, "linux")

    with _patchable_fcntl(spine_io):
        _spine_io_hint_checks()
    _fetch_v2_hint_checks()


def _spine_io_hint_checks():
    with mock.patch.object(spine_io, "_IS_DARWIN", False), \
            mock.patch.object(spine_io, "NOCACHE", True), \
            mock.patch.object(spine_io.fcntl, "fcntl") as call:
        assert spine_io._apply_nocache(7, True) is False
        call.assert_not_called()
    with mock.patch.object(spine_io, "_IS_DARWIN", True), \
            mock.patch.object(spine_io.fcntl, "fcntl", return_value=0) as call:
        assert spine_io._apply_nocache(7, True) is True
        call.assert_called_once_with(7, spine_io.F_NOCACHE, 1)
    with mock.patch.object(spine_io, "_IS_LINUX", True), \
            mock.patch.object(
                spine_io.os, "POSIX_FADV_DONTNEED", 4, create=True
            ), mock.patch.object(
                spine_io.os, "posix_fadvise", return_value=None, create=True
            ) as call:
        assert spine_io._drop_read_cache(7, 1024, 4096, True) is True
        call.assert_called_once_with(7, 1024, 4096, 4)
    with tempfile.TemporaryDirectory() as td, \
            mock.patch.object(spine_io, "_IS_DARWIN", False), \
            mock.patch.object(spine_io, "_IS_LINUX", True), \
            mock.patch.object(
                spine_io.os, "POSIX_FADV_WILLNEED", 3, create=True
            ), mock.patch.object(
                spine_io.os, "posix_fadvise", return_value=None, create=True
            ) as call:
        path = os.path.join(td, "layer.bin")
        pathlib.Path(path).write_bytes(b"x" * 32)
        # These hints are POSIX file-descriptor operations.  A Windows reader is
        # an overlapped handle with no descriptor, so rdadvise must decline
        # rather than invent one, even with the platform flags forced on.
        if os.name == "nt":
            assert spine_io.rdadvise(path, 4, 12) is False
            call.assert_not_called()
        else:
            assert spine_io.rdadvise(path, 4, 12) is True
            call.assert_called_once_with(mock.ANY, 4, 12, 3)


def _fetch_v2_hint_checks():
    with tempfile.TemporaryDirectory() as td, \
            mock.patch.dict(os.environ, {"DELTAFIN_ROOT": td}):
        import fetch_v2
    assert fetch_v2._pread_nocache_default("darwin") == "1"
    assert fetch_v2._pread_nocache_default("linux") == "0"
    with _patchable_fcntl(fetch_v2):
        _fetch_v2_nocache_checks(fetch_v2)
    _fetch_v2_read_order_checks(fetch_v2)


def _fetch_v2_nocache_checks(fetch_v2):
    with mock.patch.object(fetch_v2, "_IS_DARWIN", False), \
            mock.patch.object(fetch_v2, "PREAD_NOCACHE", True), \
            mock.patch.object(fetch_v2.fcntl, "fcntl") as call:
        assert fetch_v2._apply_pread_nocache(9) is False
        call.assert_not_called()
    with mock.patch.object(fetch_v2, "_IS_DARWIN", True), \
            mock.patch.object(fetch_v2, "PREAD_NOCACHE", True), \
            mock.patch.object(fetch_v2.fcntl, "fcntl", return_value=0) as call:
        assert fetch_v2._apply_pread_nocache(9) is True
        call.assert_called_once_with(9, fetch_v2.F_NOCACHE, 1)
    with mock.patch.object(fetch_v2, "_IS_DARWIN", False), \
            mock.patch.object(fetch_v2, "_IS_LINUX", True), \
            mock.patch.object(fetch_v2, "PREAD_NOCACHE", True), \
            mock.patch.object(
                fetch_v2.os, "POSIX_FADV_DONTNEED", 4, create=True
            ), mock.patch.object(
                fetch_v2.os, "posix_fadvise", return_value=None, create=True
            ) as call:
        assert fetch_v2._apply_pread_nocache(9) is False
        assert fetch_v2._drop_linux_pread_cache(9) is True
        call.assert_called_once_with(9, 0, 0, 4)


def _fetch_v2_read_order_checks(fetch_v2):
    # The Darwin hint must precede the read; Linux eviction must follow the last
    # successful read. Exercise the actual _Slot.read control flow with 4 bytes.
    events = []
    slot = object.__new__(fetch_v2._Slot)
    slot.mv = memoryview(bytearray(4))

    class FakeSource:
        def fileno(self):
            return 11

        def read_into(self, destination, _offset):
            events.append("read")
            return len(destination)

        def close(self):
            events.append("close")

    with mock.patch.object(fetch_v2, "EXPERT_SPAN", 4), \
            mock.patch.object(fetch_v2.positional_io, "open_positional",
                              return_value=FakeSource()), \
            mock.patch.object(
                fetch_v2, "_apply_pread_nocache",
                side_effect=lambda _fd: events.append("darwin-before"),
            ), mock.patch.object(
                fetch_v2, "_drop_linux_pread_cache",
                side_effect=lambda _fd: events.append("linux-after"),
            ):
        slot.read("unused")
    assert events == ["darwin-before", "read", "linux-after", "close"]


def _spine_cache_linux_checks():
    if hasattr(os, "sysconf"):
        assert spine_cache.PAGE == os.sysconf("SC_PAGE_SIZE")
    else:
        # Windows has no sysconf; the module must fall back, not raise.
        assert spine_cache.PAGE == 4096
    with tempfile.TemporaryDirectory() as td:
        meminfo = pathlib.Path(td) / "meminfo"
        vmstat = pathlib.Path(td) / "vmstat"
        meminfo.write_text(
            "MemFree: 1024 kB\nCached: 2048 kB\n"
            "SReclaimable: 512 kB\nShmem: 256 kB\n",
            encoding="utf-8",
        )
        vmstat.write_text("pswpout 7\n", encoding="utf-8")
        snap = spine_cache._linux_vm_snapshot(
            str(meminfo), str(vmstat), available_cap_bytes=1 << 40
        )
        assert snap["free"] * spine_cache.PAGE == 1024 * 1024
        assert snap["external"] * spine_cache.PAGE == 2304 * 1024
        assert snap["swapouts"] == 7
        assert snap["compressions"] == 0
        capped = spine_cache._linux_vm_snapshot(
            str(meminfo), str(vmstat), available_cap_bytes=0
        )
        assert capped["free"] == capped["external"] == 0

    with mock.patch.object(spine_cache, "_IS_DARWIN", False), \
            mock.patch.object(spine_cache, "_lc",
                              side_effect=AssertionError("Mach called")):
        assert spine_cache.pressure_level() == 1

    memory = rp.LinuxMemory(
        host_total_bytes=128 * rp.GIB,
        host_available_bytes=100 * rp.GIB,
        cgroup_limit_bytes=64 * rp.GIB,
        cgroup_current_bytes=10 * rp.GIB,
    )

    class FakeSpineFast:
        @staticmethod
        def pin(*_args):
            pass

        @staticmethod
        def unpin(*_args):
            pass

    fake_snapshot = dict.fromkeys(spine_cache._VM_KEYS, 0)
    fake_snapshot["free"] = 32 * rp.GIB // spine_cache.PAGE
    with mock.patch.dict(
        os.environ, {"K3_SPINE_CACHE_AUTO": "1"}, clear=True
    ), mock.patch.object(spine_cache, "_IS_LINUX", True), \
            mock.patch.object(
                spine_cache.runtime_platform,
                "linux_memory_limits",
                return_value=memory,
            ), mock.patch.object(
                spine_cache, "vm_snapshot", return_value=fake_snapshot
            ):
        cache = spine_cache.SpineCache(FakeSpineFast())
        expected = rp.safe_linux_host_budget(
            memory, int(cache.floor + cache.file_reserve)
        )
        assert cache.enabled and cache.ceiling == expected

    messages = []
    with mock.patch.dict(
        os.environ, {"K3_SPINE_CACHE_GB": "100"}, clear=True
    ), mock.patch.object(spine_cache, "_IS_LINUX", True), \
            mock.patch.object(
                spine_cache.runtime_platform,
                "linux_memory_limits",
                return_value=memory,
            ), mock.patch.object(
                spine_cache, "vm_snapshot", return_value=fake_snapshot
            ):
        cache = spine_cache.SpineCache(
            FakeSpineFast(), log=messages.append
        )
        expected = rp.safe_linux_host_budget(
            memory, int(cache.floor + cache.file_reserve)
        )
        assert cache.ceiling == expected < 100e9
        assert any("clamped requested" in message for message in messages)
        no_room = dict.fromkeys(spine_cache._VM_KEYS, 0)
        cache.wd.mark = cache.wd.last = no_room
        with mock.patch.object(
            spine_cache, "vm_snapshot", return_value=no_room
        ):
            assert not cache.admit(
                "layer.0.",
                {"q": bytearray(1), "sc": None, "other": None},
                1,
            )


def main():
    _native_loader_checks()
    _native_module_manifest_checks()
    _cuda_dequant_checks()
    _prefetch_residency_filter_checks()
    _device_checks()
    _cpu_availability_checks()
    _memory_checks()
    _file_hint_checks()
    _spine_cache_linux_checks()
    print("PASS Python portability (no model weights or native execution)")


if __name__ == "__main__":
    main()
