#!/usr/bin/env python3
"""Weight-free unit tests for tools/build_native.py."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_native as native

BASE_FLAGS = {
    next(iter(aliases))
    for aliases in native.X86_64_BASE_FEATURES.values()
}


class FakeFunction:
    def __init__(self, result=None, callback=None):
        self.result = result
        self.callback = callback
        self.argtypes = None
        self.restype = None
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if self.callback is not None:
            return self.callback(*args)
        return self.result


class FakeLibrary:
    def __init__(
        self,
        symbols: tuple[str, ...],
        *,
        abi: int | None = None,
        shapes: tuple[int, int, int] | None = None,
        cuda_abi: int | None = None,
        cuda_shapes: tuple[int, int, int, int] | None = None,
    ):
        for symbol in symbols:
            setattr(self, symbol, FakeFunction())
        if abi is not None:
            self.mxfp4_abi_version = FakeFunction(abi)
        if shapes is not None:
            def report_shapes(hidden, intermediate, span):
                ctypes.cast(hidden, ctypes.POINTER(ctypes.c_int)).contents.value = shapes[0]
                ctypes.cast(
                    intermediate, ctypes.POINTER(ctypes.c_int)
                ).contents.value = shapes[1]
                ctypes.cast(
                    span, ctypes.POINTER(ctypes.c_longlong)
                ).contents.value = shapes[2]
            self.k3_metal_shapes = FakeFunction(callback=report_shapes)
        if cuda_abi is not None:
            self.k3_cuda_moe_abi_version = FakeFunction(cuda_abi)
        if cuda_shapes is not None:
            def report_cuda_shapes(hidden, intermediate, span, pointer_layout):
                ctypes.cast(hidden, ctypes.POINTER(ctypes.c_int)).contents.value = (
                    cuda_shapes[0]
                )
                ctypes.cast(
                    intermediate, ctypes.POINTER(ctypes.c_int)
                ).contents.value = cuda_shapes[1]
                ctypes.cast(
                    span, ctypes.POINTER(ctypes.c_int64)
                ).contents.value = cuda_shapes[2]
                ctypes.cast(
                    pointer_layout, ctypes.POINTER(ctypes.c_uint32)
                ).contents.value = cuda_shapes[3]
            self.k3_cuda_moe_shapes = FakeFunction(
                callback=report_cuda_shapes
            )


class TargetTests(unittest.TestCase):
    def test_supported_target_flags(self):
        darwin = native.detect_target("darwin", "arm64")
        self.assertEqual(darwin.c_arch_flags, ("-mcpu=native",))
        self.assertTrue(darwin.supports_metal)

        x86 = native.detect_target("linux", "x86_64")
        self.assertEqual(
            x86.c_arch_flags,
            (
                "-march=x86-64",
                "-mtune=native",
                "-msse3",
                "-mssse3",
                "-mavx",
                "-mfma",
            ),
        )
        self.assertEqual(x86.suffix, ".so")

        arm = native.detect_target("linux", "aarch64")
        self.assertEqual(arm.c_arch_flags, ("-mcpu=native",))
        self.assertFalse(arm.supports_metal)

    def test_windows_target_uses_msvc_and_a_vex_baseline(self):
        windows = native.detect_target("win32", "AMD64")
        self.assertEqual(windows.platform, "windows")
        self.assertEqual(windows.machine, "x86_64")
        self.assertEqual(windows.suffix, ".dll")
        self.assertEqual(windows.toolchain, "msvc")
        self.assertFalse(windows.supports_metal)
        # /arch:AVX is the -mavx analogue: a VEX baseline that does not promise
        # AVX2, which the kernel still selects at runtime.
        self.assertEqual(windows.c_arch_flags, ("/arch:AVX",))
        self.assertNotIn("/arch:AVX2", windows.c_arch_flags)
        self.assertEqual(
            native.detect_target("win32", "x86_64").machine, "x86_64"
        )

    def test_unsupported_targets_are_clear(self):
        with self.assertRaisesRegex(native.BuildError, "Apple Silicon"):
            native.detect_target("darwin", "x86_64")
        with self.assertRaisesRegex(native.BuildError, "supported architectures"):
            native.detect_target("linux", "riscv64")
        with self.assertRaisesRegex(
            native.BuildError, "unsupported Windows architecture"
        ):
            native.detect_target("win32", "arm64")
        with self.assertRaisesRegex(native.BuildError, "unsupported platform"):
            native.detect_target("freebsd14", "amd64")

    def test_windows_artifacts_are_dlls_without_metal(self):
        windows = native.detect_target("win32", "AMD64")
        names = [
            artifact.destination.name
            for artifact in native.artifacts_for(windows)
        ]
        self.assertEqual(names, ["libmxfp4gemv.dll", "libmxfp4batch.dll"])
        for artifact in native.artifacts_for(windows):
            self.assertTrue(
                set(native.X86_AVX2_SYMBOLS).issubset(artifact.required_symbols)
            )
        cuda = [
            artifact
            for artifact in native.artifacts_for(
                windows, cuda_mode="on", nvcc_available=True
            )
            if artifact.language == "cuda"
        ]
        self.assertEqual([item.destination.name for item in cuda], ["libcudamoe.dll"])

    def test_windows_cpu_preflight_infers_fma_from_avx2(self):
        target = native.detect_target("win32", "AMD64")
        complete = {
            native.PF_SSE3_INSTRUCTIONS_AVAILABLE,
            native.PF_SSSE3_INSTRUCTIONS_AVAILABLE,
            native.PF_AVX_INSTRUCTIONS_AVAILABLE,
            native.PF_AVX2_INSTRUCTIONS_AVAILABLE,
        }
        self.assertEqual(
            native.windows_cpu_features(
                feature_probe=lambda item: item in complete
            ),
            frozenset({"sse3", "ssse3", "avx", "fma"}),
        )
        without_avx2 = complete - {native.PF_AVX2_INSTRUCTIONS_AVAILABLE}
        self.assertNotIn(
            "fma",
            native.windows_cpu_features(
                feature_probe=lambda item: item in without_avx2
            ),
        )
        self.assertIn(
            "fma",
            native.windows_cpu_features(
                feature_probe=lambda item: item in without_avx2,
                assume_fma3=True,
            ),
        )
        native.preflight_target(target, cpu_flags=BASE_FLAGS)
        with self.assertRaisesRegex(native.BuildError, "K3_ASSUME_FMA3"):
            native.preflight_target(target, cpu_flags=BASE_FLAGS - {"fma"})

    def test_x86_cpu_preflight(self):
        target = native.detect_target("linux", "x86_64")
        native.preflight_target(target, cpu_flags=BASE_FLAGS)
        # AVX2 is an optional runtime-selected island, not a load-time contract.
        native.preflight_target(target, cpu_flags=BASE_FLAGS | {"avx2"})
        with self.assertRaisesRegex(native.BuildError, "missing CPU flags: fma"):
            native.preflight_target(target, cpu_flags=BASE_FLAGS - {"fma"})

    def test_x86_artifacts_require_both_dispatch_islands(self):
        x86 = native.artifacts_for(
            native.detect_target("linux", "x86_64"), skip_metal=True
        )
        arm = native.artifacts_for(
            native.detect_target("linux", "aarch64"), skip_metal=True
        )
        for artifact in x86:
            self.assertTrue(
                set(native.X86_AVX2_SYMBOLS).issubset(
                    artifact.required_symbols
                )
            )
        for artifact in arm:
            self.assertTrue(
                set(native.X86_AVX2_SYMBOLS).isdisjoint(
                    artifact.required_symbols
                )
            )

    def test_cuda_artifact_selection_is_explicit_and_deterministic(self):
        linux = native.detect_target("linux", "x86_64")
        darwin = native.detect_target("darwin", "arm64")

        self.assertFalse(
            any(
                artifact.language == "cuda"
                for artifact in native.artifacts_for(
                    linux, cuda_mode="off", nvcc_available=True
                )
            )
        )
        self.assertFalse(
            any(
                artifact.language == "cuda"
                for artifact in native.artifacts_for(
                    linux, cuda_mode="auto", nvcc_available=False
                )
            )
        )
        auto = native.artifacts_for(
            linux, cuda_mode="auto", nvcc_available=True
        )
        required = native.artifacts_for(
            linux, cuda_mode="on", nvcc_available=False
        )
        for artifacts in (auto, required):
            cuda = [item for item in artifacts if item.language == "cuda"]
            self.assertEqual(len(cuda), 1)
            self.assertEqual(cuda[0].destination.name, "libcudamoe.so")
            self.assertEqual(cuda[0].expected_cuda_shapes, native.CUDA_SHAPES)

        self.assertFalse(
            any(
                artifact.language == "cuda"
                for artifact in native.artifacts_for(
                    darwin, cuda_mode="auto", nvcc_available=True
                )
            )
        )
        with self.assertRaisesRegex(native.BuildError, "only on Linux"):
            native.artifacts_for(darwin, cuda_mode="on")

    def test_cpuinfo_parser(self):
        with tempfile.TemporaryDirectory() as td:
            cpuinfo = Path(td) / "cpuinfo"
            cpuinfo.write_text(
                "processor : 0\nflags : sse2 ssse3 fma avx2\n"
                "processor : 1\nflags : sse2 ssse3 fma avx2 bmi2\n",
                encoding="utf-8",
            )
            flags = native.read_linux_cpu_flags(cpuinfo)
        self.assertEqual(flags, {"sse2", "ssse3", "fma", "avx2"})


class CommandTests(unittest.TestCase):
    def test_linux_commands_honor_cc_and_architecture(self):
        target = native.detect_target("linux", "x86_64")
        artifact = native.Artifact(
            "gemv",
            Path("/repo/tools/fused_gemv.c"),
            Path("/repo/tools/libmxfp4gemv.so"),
            "c",
            native.GEMV_SYMBOLS,
            abi_version=1,
        )
        command = native.compile_command(
            artifact,
            target,
            Path("/tmp/out.so"),
            environ={"CC": f"{sys.executable} --compiler-wrapper"},
        )
        self.assertEqual(command[:2], [sys.executable, "--compiler-wrapper"])
        self.assertIn("-march=x86-64", command)
        self.assertIn("-mtune=native", command)
        self.assertIn("-mavx", command)
        self.assertIn("-mfma", command)
        self.assertNotIn("-mavx2", command)
        self.assertNotIn("-march=x86-64-v3", command)
        self.assertIn("-fPIC", command)
        self.assertIn("-shared", command)
        self.assertNotIn("-dynamiclib", command)

    def test_darwin_metal_command_honors_cxx(self):
        target = native.detect_target("darwin", "arm64")
        artifact = native.artifacts_for(target)[2]
        command = native.compile_command(
            artifact,
            target,
            Path("/tmp/metal.dylib"),
            environ={"CXX": f"{sys.executable} --cxx-wrapper"},
        )
        self.assertEqual(command[:2], [sys.executable, "--cxx-wrapper"])
        self.assertIn("-fobjc-arc", command)
        self.assertIn("-framework", command)
        self.assertIn("Metal", command)

    def test_missing_compiler_is_clear(self):
        target = native.detect_target("linux", "aarch64")
        artifact = native.Artifact(
            "gemv",
            Path("/repo/fused_gemv.c"),
            Path("/repo/libmxfp4gemv.so"),
            "c",
            native.GEMV_SYMBOLS,
            abi_version=1,
        )
        with self.assertRaisesRegex(native.BuildError, "CC compiler not found"):
            native.compile_command(
                artifact,
                target,
                Path("/tmp/out.so"),
                environ={"CC": "definitely-not-a-real-deltafin-compiler"},
            )

    def test_cuda_command_uses_portable_floor_and_ptx_fallback(self):
        target = native.detect_target("linux", "x86_64")
        artifact = [
            item
            for item in native.artifacts_for(
                target, cuda_mode="on", nvcc_available=True
            )
            if item.language == "cuda"
        ][0]
        command = native.compile_command(
            artifact,
            target,
            Path("/tmp/libcudamoe.so"),
            cuda_compiler=[sys.executable, "--nvcc-wrapper"],
        )
        self.assertEqual(command[:2], [sys.executable, "--nvcc-wrapper"])
        self.assertIn("-Xcompiler=-fPIC", command)
        self.assertIn("-gencode=arch=compute_75,code=sm_75", command)
        self.assertIn("-gencode=arch=compute_75,code=compute_75", command)
        self.assertFalse(any("sm_100" in part for part in command))
        self.assertFalse(any("sm_120" in part for part in command))
        self.assertEqual(
            command[-2:], ["-o", str(Path("/tmp/libcudamoe.so"))]
        )

    def test_cuda_command_adds_native_sass_without_losing_portable_ptx(self):
        target = native.detect_target("linux", "x86_64")
        artifact = next(
            item
            for item in native.artifacts_for(
                target, cuda_mode="on", nvcc_available=True
            )
            if item.language == "cuda"
        )
        command = native.compile_command(
            artifact,
            target,
            Path("/tmp/libcudamoe.so"),
            cuda_compiler=[sys.executable, "--nvcc-wrapper"],
            cuda_codegen=native.CudaCodegen("75", ("89", "120")),
        )
        self.assertIn("-gencode=arch=compute_75,code=sm_75", command)
        self.assertIn("-gencode=arch=compute_89,code=sm_89", command)
        self.assertIn("-gencode=arch=compute_120,code=sm_120", command)
        self.assertIn("-gencode=arch=compute_75,code=compute_75", command)
        self.assertNotIn(
            "-gencode=arch=compute_120,code=compute_120", command
        )

    def test_nvcc_environment_is_parsed_and_resolved_safely(self):
        requested = []

        def find(executable):
            requested.append(executable)
            return "/opt/cuda/bin/nvcc"

        argv = native.cuda_compiler_argv(
            {"NVCC": "custom-nvcc --use_fast_math"}, finder=find
        )
        self.assertEqual(requested, ["custom-nvcc"])
        self.assertEqual(
            argv, ["/opt/cuda/bin/nvcc", "--use_fast_math"]
        )
        self.assertIsNone(
            native.cuda_compiler_argv({}, finder=lambda _name: None)
        )
        with self.assertRaisesRegex(native.BuildError, "NVCC is empty"):
            native.cuda_compiler_argv({"NVCC": "   "})

        def broken_path(_name):
            raise OSError("synthetic PATH failure")

        with self.assertRaisesRegex(native.BuildError, "could not locate NVCC"):
            native.cuda_compiler_argv({}, finder=broken_path)

    def test_windows_c_command_targets_msvc(self):
        target = native.detect_target("win32", "AMD64")
        artifact = native.artifacts_for(target)[0]
        command = native.compile_command(
            artifact,
            target,
            Path(r"C:\out\libmxfp4gemv.dll"),
            environ={"CC": sys.executable},
            scratch=Path(r"C:\scratch"),
        )
        self.assertEqual(command[0], sys.executable)
        self.assertIn("/LD", command)
        self.assertIn("/arch:AVX", command)
        self.assertIn("/DNO_MAIN", command)
        # MSVC gates C11 atomics, which the worker latches depend on.
        self.assertIn("/experimental:c11atomics", command)
        self.assertIn(r"/Fe:C:\out\libmxfp4gemv.dll", command)
        # Objects and the import library must not land beside the installed DLL.
        self.assertIn("/Fo:" + os.path.join(r"C:\scratch", ""), command)
        # Build the expected import-library path the same way, rather than
        # spelling the separator: this assertion has to hold when the suite
        # runs on macOS or Linux too, where joining produces a forward slash.
        self.assertIn(
            "/IMPLIB:" + str(Path(r"C:\scratch") / "libmxfp4gemv.lib"),
            command,
        )
        for gnu_only in ("-fPIC", "-shared", "-lpthread", "-lm", "-O3"):
            self.assertNotIn(gnu_only, command)

    def test_windows_cuda_command_uses_shared_crt_and_scratch_implib(self):
        target = native.detect_target("win32", "AMD64")
        artifact = next(
            item
            for item in native.artifacts_for(
                target, cuda_mode="on", nvcc_available=True
            )
            if item.language == "cuda"
        )
        command = native.compile_command(
            artifact,
            target,
            Path(r"C:\out\libcudamoe.dll"),
            cuda_compiler=[sys.executable, "--nvcc-wrapper"],
            cuda_codegen=native.CudaCodegen("75", ("120",)),
            scratch=Path(r"C:\scratch"),
        )
        # The DLL and its Python caller must share one CRT heap.
        self.assertIn("-Xcompiler=/MD", command)
        self.assertNotIn("-Xcompiler=-fPIC", command)
        self.assertIn(
            "/IMPLIB:" + str(Path(r"C:\scratch") / "libcudamoe.lib"), command
        )
        self.assertIn("-gencode=arch=compute_75,code=sm_75", command)
        self.assertIn("-gencode=arch=compute_120,code=sm_120", command)
        self.assertIn("-gencode=arch=compute_75,code=compute_75", command)

    def test_command_splitting_follows_the_host_not_the_target(self):
        # A CC override names an executable this machine runs, so Windows path
        # separators must survive even when building for another target.
        self.assertEqual(
            native.split_command(r"C:\Tools\cl.exe /nologo", windows=True),
            [r"C:\Tools\cl.exe", "/nologo"],
        )
        self.assertEqual(
            native.split_command(r'"C:\Program Files\cl.exe" /W3', windows=True),
            [r"C:\Program Files\cl.exe", "/W3"],
        )
        self.assertEqual(
            native.split_command("cc -pipe", windows=False), ["cc", "-pipe"]
        )
        self.assertEqual(
            native.render_command(["cl", r"C:\a b\x.c"], windows=True),
            'cl "C:\\a b\\x.c"',
        )

    def test_environment_path_lookup_is_case_insensitive(self):
        # Windows reports its own variable as "Path", not "PATH".
        self.assertEqual(
            native.environment_path({"Path": r"C:\bin", "OTHER": "x"}), r"C:\bin"
        )
        self.assertEqual(native.environment_path({"PATH": "/usr/bin"}), "/usr/bin")
        self.assertIsNone(native.environment_path({"HOME": "/root"}))


class CudaArchitectureTests(unittest.TestCase):
    @staticmethod
    def _result(stdout="", stderr="", returncode=0):
        return subprocess.CompletedProcess(
            (), returncode, stdout=stdout, stderr=stderr
        )

    def test_normalization_and_override_validation(self):
        expected = {
            "8.9": "89",
            "sm_89": "89",
            "compute_89": "89",
            "12.0": "120",
            "sm_120": "120",
        }
        for value, normalized in expected.items():
            with self.subTest(value=value):
                self.assertEqual(
                    native.normalize_cuda_arch(value), normalized
                )
        self.assertEqual(
            native.parse_cuda_arches("sm_120, 8.9;sm_89"),
            ("89", "120"),
        )
        for value in ("", "sm_7", "sm_74", "8.9\n8.6", "sm_89,-O0"):
            with self.subTest(value=value):
                with self.assertRaises(native.BuildError):
                    if value:
                        native.normalize_cuda_arch(value)
                    else:
                        native.parse_cuda_arches(value)

    def test_detector_handles_multiple_gpus_and_malformed_rows(self):
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            return self._result("8.9\n8.6\nN/A\n12.0\n8.9\n")

        arches = native.detect_cuda_arches(
            nvidia_smi="/usr/bin/nvidia-smi",
            runner=run,
        )
        self.assertEqual(arches, ("86", "89", "120"))
        self.assertEqual(
            commands,
            [[
                "/usr/bin/nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ]],
        )

    def test_detector_fails_closed_without_executing_a_shell(self):
        self.assertEqual(native.detect_cuda_arches(nvidia_smi=None), ())

        def timeout(_command, **_kwargs):
            raise subprocess.TimeoutExpired("nvidia-smi", 10)

        self.assertEqual(
            native.detect_cuda_arches(
                nvidia_smi="/usr/bin/nvidia-smi", runner=timeout
            ),
            (),
        )
        self.assertEqual(
            native.detect_cuda_arches(
                nvidia_smi="/usr/bin/nvidia-smi",
                runner=lambda _command, **_kwargs: self._result(
                    "8.9", returncode=9
                ),
            ),
            (),
        )

    def test_nvcc_support_intersects_real_and_virtual_targets(self):
        def run(command, **_kwargs):
            if "--list-gpu-code" in command:
                return self._result(
                    "sm_75\nsm_89\nsm_90a\nsm_100f\nsm_120\n"
                )
            if "--list-gpu-arch" in command:
                return self._result(
                    "compute_75\ncompute_89\ncompute_90a\ncompute_100f\n"
                )
            self.fail(f"unexpected command: {command}")

        self.assertEqual(
            native.nvcc_supported_arches(["/opt/cuda/bin/nvcc"], runner=run),
            {"75", "89"},
        )

    def test_auto_selection_filters_toolkit_mismatch_and_keeps_floor(self):
        reports = []

        def run(command, **_kwargs):
            if "--list-gpu-code" in command:
                return self._result("sm_75\nsm_89\n")
            if "--list-gpu-arch" in command:
                return self._result("compute_75\ncompute_89\n")
            if "--query-gpu=compute_cap" in command:
                return self._result("8.9\n12.0\n")
            self.fail(f"unexpected command: {command}")

        selected = native.resolve_cuda_codegen(
            {},
            ["/opt/cuda/bin/nvcc"],
            runner=run,
            nvidia_smi_finder=lambda _name: "/usr/bin/nvidia-smi",
            reporter=reports.append,
        )
        self.assertEqual(selected, native.CudaCodegen("75", ("89",)))
        self.assertIn("sm_120", reports[0])
        self.assertIn("compute_75 PTX", reports[0])

    def test_explicit_override_is_strict_and_compiler_checked(self):
        def run(command, **_kwargs):
            if "--list-gpu-code" in command:
                return self._result("sm_75\nsm_89\n")
            if "--list-gpu-arch" in command:
                return self._result("compute_75\ncompute_89\n")
            self.fail("an explicit override must not probe nvidia-smi")

        selected = native.resolve_cuda_codegen(
            {"K3_CUDA_ARCH": "sm_89"},
            ["/opt/cuda/bin/nvcc"],
            runner=run,
            nvidia_smi_finder=lambda _name: self.fail(
                "an explicit override must not locate nvidia-smi"
            ),
        )
        self.assertEqual(
            selected, native.CudaCodegen("75", ("89",), True)
        )
        with self.assertRaisesRegex(native.BuildError, "does not list"):
            native.resolve_cuda_codegen(
                {"K3_CUDA_ARCH": "sm_120"},
                ["/opt/cuda/bin/nvcc"],
                runner=run,
                nvidia_smi_finder=lambda _name: None,
            )


class ParserTests(unittest.TestCase):
    def test_cuda_mode_defaults_to_auto_and_accepts_all_modes(self):
        parser = native._parser()
        self.assertEqual(parser.parse_args([]).cuda, "auto")
        for mode in ("auto", "on", "off"):
            self.assertEqual(parser.parse_args([f"--cuda={mode}"]).cuda, mode)


class ValidationTests(unittest.TestCase):
    def _path(self, directory: str) -> Path:
        path = Path(directory) / "candidate.so"
        path.write_bytes(b"candidate")
        return path

    def test_abi_and_symbols(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._path(td)
            artifact = native.Artifact(
                "gemv", path, path, "c", native.GEMV_SYMBOLS, abi_version=1
            )
            library = FakeLibrary(native.GEMV_SYMBOLS, abi=1)
            native.validate_artifact(
                path, artifact, cdll_factory=lambda _: library
            )

            bad = FakeLibrary(native.GEMV_SYMBOLS, abi=9)
            with self.assertRaisesRegex(native.BuildError, "has ABI 9"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad
                )

            missing = FakeLibrary(("mxfp4_gemv",), abi=1)
            with self.assertRaisesRegex(native.BuildError, "missing required symbols"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: missing
                )

    def test_metal_symbol_and_shape_handshake(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._path(td)
            artifact = native.Artifact(
                "Metal",
                path,
                path,
                "cxx",
                native.METAL_SYMBOLS,
                expected_metal_shapes=native.METAL_SHAPES,
            )
            good = FakeLibrary(
                native.METAL_SYMBOLS, shapes=native.METAL_SHAPES
            )
            native.validate_artifact(
                path, artifact, cdll_factory=lambda _: good
            )
            bad = FakeLibrary(
                native.METAL_SYMBOLS, shapes=(3584, 3072, 1)
            )
            with self.assertRaisesRegex(native.BuildError, "shape handshake returned"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad
                )

    def test_cuda_symbols_abi_and_layout_without_device_probe(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._path(td)
            artifact = native.Artifact(
                "CUDA MoE",
                path,
                path,
                "cuda",
                native.CUDA_SYMBOLS,
                abi_version=native.CUDA_ABI_VERSION,
                abi_symbol="k3_cuda_moe_abi_version",
                expected_cuda_shapes=native.CUDA_SHAPES,
            )
            good = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=native.CUDA_SHAPES,
            )
            native.validate_artifact(
                path, artifact, cdll_factory=lambda _: good
            )
            self.assertEqual(good.k3_cuda_moe_abi_version.calls, 1)
            self.assertEqual(good.k3_cuda_moe_shapes.calls, 1)
            self.assertEqual(good.k3_cuda_moe_available.calls, 0)
            self.assertEqual(good.k3_cuda_moe_launch.calls, 0)

            bad_abi = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=99,
                cuda_shapes=native.CUDA_SHAPES,
            )
            with self.assertRaisesRegex(native.BuildError, "has ABI 99"):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad_abi
                )

            bad_span = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=(3584, 3072, 1, 1),
            )
            with self.assertRaisesRegex(
                native.BuildError, "shape/layout handshake returned"
            ):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad_span
                )

            bad_layout = FakeLibrary(
                native.CUDA_SYMBOLS,
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=(3584, 3072, 17_547_264, 99),
            )
            with self.assertRaisesRegex(
                native.BuildError, "shape/layout handshake returned"
            ):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: bad_layout
                )

            missing = FakeLibrary(
                tuple(
                    symbol
                    for symbol in native.CUDA_SYMBOLS
                    if symbol != "k3_cuda_last_error"
                ),
                cuda_abi=native.CUDA_ABI_VERSION,
                cuda_shapes=native.CUDA_SHAPES,
            )
            with self.assertRaisesRegex(
                native.BuildError, "k3_cuda_last_error"
            ):
                native.validate_artifact(
                    path, artifact, cdll_factory=lambda _: missing
                )


class AtomicBuildTests(unittest.TestCase):
    def _tree(
        self,
        root: Path,
        *,
        suffix: str = ".so",
        cuda: bool = False,
    ) -> Path:
        tools = root / "repo" / "tools"
        tools.mkdir(parents=True)
        (tools / "fused_gemv.c").write_text("gemv", encoding="utf-8")
        (tools / "fused_gemv_batch.c").write_text("batch", encoding="utf-8")
        (tools / f"libmxfp4gemv{suffix}").write_bytes(b"old-gemv")
        (tools / f"libmxfp4batch{suffix}").write_bytes(b"old-batch")
        if cuda:
            (tools / "cuda_moe_kernels.cu").write_text(
                "cuda", encoding="utf-8"
            )
            (tools / "libcudamoe.so").write_bytes(b"old-cuda")
        return tools

    @staticmethod
    def _write_output(command, _cwd):
        output = Path(command[command.index("-o") + 1])
        source = next(
            Path(part).name for part in command
            if str(part).endswith((".c", ".mm", ".cu"))
        )
        output.write_bytes(f"new:{source}".encode())

    def test_compile_failure_preserves_every_existing_library(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))
            calls = 0

            def fail_second(command, cwd):
                nonlocal calls
                calls += 1
                self._write_output(command, cwd)
                if calls == 2:
                    raise native.BuildError("synthetic compiler failure")

            with self.assertRaisesRegex(native.BuildError, "synthetic compiler"):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    environ={"CC": sys.executable},
                    runner=fail_second,
                    validator=lambda _path, _artifact: None,
                    cuda_mode="off",
                )
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
            )

    def test_validation_failure_preserves_every_existing_library(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))

            def reject_batch(_path, artifact):
                if artifact.label == "MXFP4 batch":
                    raise native.BuildError("synthetic ABI rejection")

            with self.assertRaisesRegex(native.BuildError, "synthetic ABI"):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    environ={"CC": sys.executable},
                    runner=self._write_output,
                    validator=reject_batch,
                    cuda_mode="off",
                )
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
            )
            self.assertFalse(list(tools.glob(".*.build-*")))

    def test_success_installs_only_after_all_validations(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))
            validated = []

            def accept(path, artifact):
                self.assertTrue(path.read_bytes().startswith(b"new:"))
                validated.append(artifact.label)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=accept,
                cuda_mode="off",
            )
            self.assertEqual(
                validated, ["MXFP4 GEMV", "MXFP4 batch"]
            )
            self.assertEqual(len(outputs), 2)
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(),
                b"new:fused_gemv_batch.c",
            )

    def test_darwin_auto_keeps_outputs_and_never_probes_nvcc(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), suffix=".dylib")

            def forbidden_probe(_name):
                self.fail("Darwin auto mode must not probe NVCC")

            outputs = native.build_native(
                target=native.detect_target("darwin", "arm64"),
                tools_dir=tools,
                skip_metal=True,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=lambda _path, _artifact: None,
                nvcc_finder=forbidden_probe,
                nvidia_smi_finder=forbidden_probe,
            )
            self.assertEqual(
                [path.name for path in outputs],
                ["libmxfp4gemv.dylib", "libmxfp4batch.dylib"],
            )

    def test_linux_auto_without_nvcc_builds_required_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td))
            probes = []

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=lambda _path, _artifact: None,
                nvcc_finder=lambda name: probes.append(name),
                nvidia_smi_finder=lambda _name: self.fail(
                    "nvidia-smi must not be probed without NVCC"
                ),
            )
            self.assertEqual(probes, ["nvcc"])
            self.assertEqual(len(outputs), 2)
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )

    def test_cuda_off_never_probes_or_builds_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            commands = []

            def forbidden_probe(_name):
                self.fail("CUDA off mode must not probe NVCC")

            def record(command, cwd):
                commands.append(command)
                self._write_output(command, cwd)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="off",
                environ={"CC": sys.executable},
                runner=record,
                validator=lambda _path, _artifact: None,
                nvcc_finder=forbidden_probe,
                nvidia_smi_finder=forbidden_probe,
            )
            self.assertEqual(len(outputs), 2)
            self.assertEqual(len(commands), 2)
            self.assertFalse(
                any(
                    any(str(part).endswith(".cu") for part in command)
                    for command in commands
                )
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_cuda_auto_compile_failure_does_not_block_cpu_install(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            reports = []

            def fail_cuda(command, cwd):
                self._write_output(command, cwd)
                if any(str(part).endswith(".cu") for part in command):
                    raise native.BuildError("synthetic nvcc rejection")

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=fail_cuda,
                validator=lambda _path, _artifact: None,
                nvcc_finder=lambda _name: sys.executable,
                cuda_codegen=native.CudaCodegen(),
                reporter=reports.append,
            )
            self.assertEqual(len(outputs), 2)
            self.assertIn("synthetic nvcc rejection", reports[0])
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(),
                b"new:fused_gemv_batch.c",
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_auto_detected_native_failure_retries_portable_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            reports = []
            cuda_commands = []

            def reject_native_only(command, cwd):
                if any(str(part).endswith(".cu") for part in command):
                    cuda_commands.append(list(command))
                    if any("sm_89" in str(part) for part in command):
                        raise native.BuildError("synthetic native rejection")
                self._write_output(command, cwd)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="on",
                environ={"CC": sys.executable},
                runner=reject_native_only,
                validator=lambda _path, _artifact: None,
                nvcc_finder=lambda _name: sys.executable,
                cuda_codegen=native.CudaCodegen("75", ("89",)),
                reporter=reports.append,
            )
            self.assertEqual(len(outputs), 3)
            self.assertEqual(len(cuda_commands), 2)
            self.assertTrue(
                any("sm_89" in str(part) for part in cuda_commands[0])
            )
            self.assertFalse(
                any("sm_89" in str(part) for part in cuda_commands[1])
            )
            self.assertTrue(
                any("retrying the portable" in report for report in reports)
            )

    def test_explicit_native_failure_does_not_silently_retry(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            cuda_calls = 0

            def reject_cuda(command, cwd):
                nonlocal cuda_calls
                if any(str(part).endswith(".cu") for part in command):
                    cuda_calls += 1
                    raise native.BuildError("synthetic explicit rejection")
                self._write_output(command, cwd)

            with self.assertRaisesRegex(
                native.BuildError, "synthetic explicit rejection"
            ):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    cuda_mode="on",
                    environ={"CC": sys.executable},
                    runner=reject_cuda,
                    validator=lambda _path, _artifact: None,
                    nvcc_finder=lambda _name: sys.executable,
                    cuda_codegen=native.CudaCodegen(
                        "75", ("89",), explicit=True
                    ),
                )
            self.assertEqual(cuda_calls, 1)

    def test_cuda_auto_validation_failure_does_not_block_cpu_install(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            reports = []

            def reject_cuda(_path, artifact):
                if artifact.language == "cuda":
                    raise ValueError("synthetic third-party validator failure")

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=reject_cuda,
                nvcc_finder=lambda _name: sys.executable,
                cuda_codegen=native.CudaCodegen(),
                reporter=reports.append,
            )
            self.assertEqual(len(outputs), 2)
            self.assertIn("third-party validator failure", reports[0])
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_cuda_on_failure_is_clear_and_preserves_all_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)

            def reject_cuda(_path, artifact):
                if artifact.language == "cuda":
                    raise native.BuildError("synthetic CUDA ABI rejection")

            with self.assertRaisesRegex(
                native.BuildError,
                "required by --cuda=on.*synthetic CUDA ABI rejection",
            ):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    cuda_mode="on",
                    environ={"CC": sys.executable},
                    runner=self._write_output,
                    validator=reject_cuda,
                    nvcc_finder=lambda _name: sys.executable,
                    cuda_codegen=native.CudaCodegen(),
                )
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )
            self.assertFalse(list(tools.glob(".*.build-*")))

    def test_cuda_on_success_installs_after_every_validation(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            validated = []

            def accept(path, artifact):
                self.assertTrue(path.read_bytes().startswith(b"new:"))
                self.assertEqual(
                    (tools / "libmxfp4gemv.so").read_bytes(), b"old-gemv"
                )
                self.assertEqual(
                    (tools / "libmxfp4batch.so").read_bytes(), b"old-batch"
                )
                self.assertEqual(
                    (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
                )
                validated.append(artifact.label)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="on",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=accept,
                nvcc_finder=lambda _name: sys.executable,
                cuda_codegen=native.CudaCodegen(),
            )
            self.assertEqual(
                validated, ["MXFP4 GEMV", "MXFP4 batch", "CUDA MoE"]
            )
            self.assertEqual(len(outputs), 3)
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(),
                b"new:cuda_moe_kernels.cu",
            )

    def test_cuda_auto_install_failure_does_not_roll_back_cpu(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            reports = []

            def reject_cuda_replace(source, destination):
                if Path(destination).name == "libcudamoe.so":
                    raise OSError("synthetic CUDA install rejection")
                os.replace(source, destination)

            outputs = native.build_native(
                target=native.detect_target("linux", "aarch64"),
                tools_dir=tools,
                cuda_mode="auto",
                environ={"CC": sys.executable},
                runner=self._write_output,
                validator=lambda _path, _artifact: None,
                nvcc_finder=lambda _name: sys.executable,
                cuda_codegen=native.CudaCodegen(),
                reporter=reports.append,
                replace=reject_cuda_replace,
            )
            self.assertEqual(len(outputs), 2)
            self.assertIn("synthetic CUDA install rejection", reports[0])
            self.assertEqual(
                (tools / "libmxfp4gemv.so").read_bytes(),
                b"new:fused_gemv.c",
            )
            self.assertEqual(
                (tools / "libmxfp4batch.so").read_bytes(),
                b"new:fused_gemv_batch.c",
            )
            self.assertEqual(
                (tools / "libcudamoe.so").read_bytes(), b"old-cuda"
            )

    def test_cuda_on_missing_nvcc_fails_before_required_build(self):
        with tempfile.TemporaryDirectory() as td:
            tools = self._tree(Path(td), cuda=True)
            commands = []

            with self.assertRaisesRegex(native.BuildError, "NVCC was not found"):
                native.build_native(
                    target=native.detect_target("linux", "aarch64"),
                    tools_dir=tools,
                    cuda_mode="on",
                    environ={"CC": sys.executable},
                    runner=lambda command, _cwd: commands.append(command),
                    validator=lambda _path, _artifact: None,
                    nvcc_finder=lambda _name: None,
                )
            self.assertEqual(commands, [])

    def test_install_error_rolls_back_prior_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            destination_a = root / "a.so"
            destination_b = root / "b.so"
            temporary_a = root / "new-a.so"
            temporary_b = root / "new-b.so"
            destination_a.write_bytes(b"old-a")
            destination_b.write_bytes(b"old-b")
            temporary_a.write_bytes(b"new-a")
            temporary_b.write_bytes(b"new-b")
            artifact_a = native.Artifact(
                "a", root / "a.c", destination_a, "c", ()
            )
            artifact_b = native.Artifact(
                "b", root / "b.c", destination_b, "c", ()
            )
            calls = 0

            def fail_second(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic replace failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(native.BuildError, "synthetic replace"):
                native.install_validated(
                    [(artifact_a, temporary_a), (artifact_b, temporary_b)],
                    replace=fail_second,
                )
            self.assertEqual(destination_a.read_bytes(), b"old-a")
            self.assertEqual(destination_b.read_bytes(), b"old-b")
            self.assertFalse(list(root.glob(".*.backup-*")))


if __name__ == "__main__":
    unittest.main()
