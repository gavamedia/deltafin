//! Single reviewed native build graph shared by Cargo production and xtask.

use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use deltafin_bootstrap::{InstallOptions, PlatformTarget};
use serde_json::Value;
use sha2::{Digest, Sha256};

mod msvc;

use msvc::CompilerFlavor;

const TORCH_CPP_API_HEADER: &str = "include/torch/csrc/api/include/torch/torch.h";
const TORCH_VERSION_HEADER: &str = "include/torch/headeronly/version.h";
const PYTHON_DENY_EXIT: i32 = 86;
/// The stack an isolated native test (and the product) runs on. Linux and
/// macOS give the main thread 8 MiB; a Windows executable defaults to 1 MiB,
/// which the provider's decode path is not written against.
const NATIVE_TEST_STACK_BYTES: u64 = 8 << 20;
/* No fast-math/TF32 contract for authoritative FP32 CUDA accumulation. */
const CUDA_IEEE_MATH_FLAGS: &[&str] = &[
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--fmad=true",
];
/// The HIPCC spellings of the same four decisions: keep denormals, keep
/// correctly rounded division and square root, keep approximate transcendental
/// substitution off, and keep explicit fmaf contraction on. Clang defaults
/// already match on a stock ROCm install; they are stated so a distribution
/// that changes a default cannot quietly change the arithmetic.
const HIP_IEEE_MATH_FLAGS: &[&str] = &[
    "-fno-fast-math",
    "-fno-gpu-flush-denormals-to-zero",
    "-fhip-fp32-correctly-rounded-divide-sqrt",
    "-fno-gpu-approx-transcendentals",
    "-ffp-contract=fast-honor-pragmas",
];
const PROVIDER_CPP_SOURCES: &[&str] = &[
    "provider_abi.cpp",
    "provider_bf16_cpu.cpp",
    "provider_bf16_device.cpp",
    "provider_dspark.cpp",
    "provider_dspark_model.cpp",
    "provider_eagle3.cpp",
    "provider_kda.cpp",
    "provider_kda_batch.cpp",
    "provider_mla.cpp",
    "provider_pilot.cpp",
    "provider_cuda_moe.cpp",
    "provider_spine_bf16_cuda.cpp",
    "provider_moe.cpp",
    "provider_qwen.cpp",
    "provider_target.cpp",
    "provider_target_sequence.cpp",
    "provider_target_tape.cpp",
    "provider_runtime.cpp",
];

/// Repository-relative native inputs that make up the in-process production
/// provider.  Cargo invalidation and the interpreter-boundary policy consume
/// this same inventory so a platform-specific translation unit, embedded Metal
/// source, or transitively included project header cannot escape review.
pub const PRODUCTION_PROVIDER_SOURCES: &[&str] = &[
    "native/provider_gate/provider_abi.h",
    "native/provider_gate/provider_abi.cpp",
    "native/provider_gate/provider_bf16_cpu.h",
    "native/provider_gate/provider_bf16_cpu.cpp",
    "native/provider_gate/provider_bf16_device.h",
    "native/provider_gate/provider_bf16_device.cpp",
    "native/provider_gate/provider_device.h",
    "native/provider_gate/provider_dspark.h",
    "native/provider_gate/provider_dspark.cpp",
    "native/provider_gate/provider_dspark_model.h",
    "native/provider_gate/provider_dspark_model.cpp",
    "native/provider_gate/provider_eagle3.h",
    "native/provider_gate/provider_eagle3.cpp",
    "native/provider_gate/provider_kda.h",
    "native/provider_gate/provider_kda.cpp",
    "native/provider_gate/provider_kda_batch.h",
    "native/provider_gate/provider_kda_batch.cpp",
    "native/provider_gate/provider_mla.h",
    "native/provider_gate/provider_mla.cpp",
    "native/provider_gate/provider_pilot.h",
    "native/provider_gate/provider_pilot.cpp",
    "native/provider_gate/provider_cuda_moe.h",
    "native/provider_gate/provider_cuda_moe.cpp",
    "native/provider_gate/provider_spine_bf16_cuda.h",
    "native/provider_gate/provider_spine_bf16_cuda.cpp",
    "native/provider_gate/provider_spine_bf16_cuda.cu",
    "native/provider_gate/provider_moe.h",
    "native/provider_gate/provider_moe.cpp",
    "native/provider_gate/provider_qwen.h",
    "native/provider_gate/provider_qwen.cpp",
    "native/provider_gate/provider_route_mailbox.h",
    "native/provider_gate/provider_route_mailbox.mm",
    "native/provider_gate/provider_route_mailbox.metal",
    "native/provider_gate/provider_spine_bf16_metal.h",
    "native/provider_gate/provider_spine_bf16_metal.mm",
    "native/provider_gate/provider_spine_bf16_metal.metal",
    "native/provider_gate/provider_spine_int8_metal.h",
    "native/provider_gate/provider_spine_int8_metal.mm",
    "native/provider_gate/provider_spine_int8_metal.metal",
    "native/provider_gate/provider_spine_debug.h",
    "native/provider_gate/provider_precision.h",
    "native/provider_gate/provider_target.h",
    "native/provider_gate/provider_target.cpp",
    "native/provider_gate/provider_target_sequence.h",
    "native/provider_gate/provider_target_sequence.cpp",
    "native/provider_gate/provider_target_tape.h",
    "native/provider_gate/provider_target_tape.cpp",
    "native/provider_gate/provider_runtime.cpp",
    "tools/fused_gemv.c",
    "tools/fused_gemv_batch.c",
    "tools/neon_compat_x86.h",
    "tools/win_compat.h",
    "tools/gpu_runtime_compat.h",
    "tools/metal_moe_abi.h",
    "tools/metal_moe.mm",
    "tools/metal/moe_mxfp4.metal",
    "tools/cuda_moe_kernels.cu",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeTestPlatform {
    Any,
    Macos,
    Linux,
    Windows,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderFlavor {
    None,
    Production,
    SyntheticMoe,
    CudaResidency,
    MetalSourceDevelopment,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeTestCase {
    pub name: &'static str,
    pub platform: NativeTestPlatform,
    pub arguments: &'static [&'static str],
    pub environment: &'static [(&'static str, &'static str)],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeTestSpec {
    pub name: &'static str,
    pub platform: NativeTestPlatform,
    pub main_source: &'static str,
    pub extra_sources: &'static [&'static str],
    pub provider: ProviderFlavor,
    pub cases: &'static [NativeTestCase],
    pub pass_marker: &'static str,
    pub timeout_seconds: u64,
}

const CPU_ISA_CASES: &[NativeTestCase] = &[
    NativeTestCase {
        name: "automatic-isa",
        platform: NativeTestPlatform::Any,
        arguments: &[],
        environment: &[],
    },
    NativeTestCase {
        name: "forced-baseline",
        platform: NativeTestPlatform::Any,
        arguments: &[],
        environment: &[("K3_MXFP4_DISABLE_AVX2", "1")],
    },
];

const DEFAULT_CASES: &[NativeTestCase] = &[NativeTestCase {
    name: "default",
    platform: NativeTestPlatform::Any,
    arguments: &[],
    environment: &[],
}];

const CPU_AND_MPS_CASES: &[NativeTestCase] = &[
    NativeTestCase {
        name: "cpu",
        platform: NativeTestPlatform::Any,
        arguments: &["--device", "cpu"],
        environment: &[],
    },
    NativeTestCase {
        name: "mps",
        platform: NativeTestPlatform::Macos,
        arguments: &["--device", "mps"],
        environment: &[],
    },
];

/// Decode-path parity across every device the provider can select.
///
/// `CPU_AND_MPS_CASES` left the accelerator decode path ungated wherever the
/// accelerator was not Apple's, so a CUDA host verified the T=1 branches on CPU
/// only. These specs drive real per-device parity from `--device`, so the CUDA
/// case belongs in the matrix rather than in a benchmark flag.
///
/// The case is `Any` rather than `Linux` because the harness matches on host OS
/// and has no runtime capability probe. Hosts with no visible CUDA device skip
/// through `cuda_case_should_skip` and still report their pass marker.
const CPU_MPS_AND_CUDA_CASES: &[NativeTestCase] = &[
    NativeTestCase {
        name: "cpu",
        platform: NativeTestPlatform::Any,
        arguments: &["--device", "cpu"],
        environment: &[],
    },
    NativeTestCase {
        name: "mps",
        platform: NativeTestPlatform::Macos,
        arguments: &["--device", "mps"],
        environment: &[],
    },
    NativeTestCase {
        name: "cuda",
        platform: NativeTestPlatform::Any,
        arguments: &["--device", "cuda"],
        environment: &[],
    },
];

const PROVIDER_GATE_CASES: &[NativeTestCase] = &[
    NativeTestCase {
        name: "cpu",
        platform: NativeTestPlatform::Any,
        arguments: &["--device", "cpu"],
        environment: &[],
    },
    NativeTestCase {
        name: "mps",
        platform: NativeTestPlatform::Macos,
        arguments: &["--device", "mps"],
        environment: &[],
    },
    NativeTestCase {
        name: "mps-packed-int8-real-shape",
        platform: NativeTestPlatform::Macos,
        arguments: &[
            "--device",
            "mps",
            "--require-packed-int8",
            "--packed-shape",
            "12288x7168",
        ],
        environment: &[],
    },
    NativeTestCase {
        name: "mps-split-boundary",
        platform: NativeTestPlatform::Macos,
        arguments: &["--device", "mps", "--split-boundary"],
        environment: &[],
    },
    NativeTestCase {
        name: "mps-spine-binding",
        platform: NativeTestPlatform::Macos,
        arguments: &["--device", "mps", "--spine-binding"],
        environment: &[],
    },
    NativeTestCase {
        name: "mps-kda-tape",
        platform: NativeTestPlatform::Macos,
        arguments: &["--device", "mps", "--kda-tape"],
        environment: &[],
    },
];

const METAL_SOURCE_CASES: &[NativeTestCase] = &[
    NativeTestCase {
        name: "embedded-source",
        platform: NativeTestPlatform::Macos,
        arguments: &[],
        environment: &[],
    },
    NativeTestCase {
        name: "explicit-source",
        platform: NativeTestPlatform::Macos,
        arguments: &["--source", "{REPOSITORY}/tools/metal/moe_mxfp4.metal"],
        environment: &[],
    },
];

pub const NATIVE_TEST_SPECS: &[NativeTestSpec] = &[
    NativeTestSpec {
        name: "provider-schedule-oracle",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_schedule_oracle_test.c",
        extra_sources: &[],
        provider: ProviderFlavor::None,
        cases: DEFAULT_CASES,
        pass_marker: "provider_schedule_oracle=PASS",
        timeout_seconds: 60,
    },
    NativeTestSpec {
        name: "cpu-isa",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_cpu_isa_test.c",
        extra_sources: &["tools/fused_gemv_batch.c"],
        provider: ProviderFlavor::None,
        cases: CPU_ISA_CASES,
        pass_marker: "provider_cpu_isa=PASS",
        timeout_seconds: 60,
    },
    NativeTestSpec {
        name: "gate",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_gate.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: PROVIDER_GATE_CASES,
        pass_marker: "result=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "mla",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_mla_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: CPU_MPS_AND_CUDA_CASES,
        pass_marker: "check.mla_decode=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "dspark",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_dspark_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_dspark.synthetic=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "dspark-model",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_dspark_model_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_dspark_model.synthetic=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "eagle3",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_eagle3_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_eagle3.synthetic=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "dspark-abi",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_dspark_abi_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_dspark_abi.synthetic=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "kda",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_kda_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: CPU_MPS_AND_CUDA_CASES,
        pass_marker: "check.kda_decode=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "kda-batch",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_kda_batch_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: CPU_MPS_AND_CUDA_CASES,
        pass_marker: "check.kda_batch_projection=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "moe",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_moe_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::SyntheticMoe,
        cases: DEFAULT_CASES,
        pass_marker: "provider_moe.parity=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "pilot",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_pilot_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: CPU_AND_MPS_CASES,
        pass_marker: "provider_pilot.result=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "cuda-moe",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_cuda_moe_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::CudaResidency,
        cases: DEFAULT_CASES,
        pass_marker: "provider_cuda_moe.result=PASS",
        timeout_seconds: 600,
    },
    NativeTestSpec {
        name: "cuda-plan-abi",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_cuda_plan_abi_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_cuda_plan_abi.portable=ok",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "provider-precision",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_precision_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_precision.cuda_ieee=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "bf16-cpu",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_bf16_cpu_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_bf16_cpu=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "bf16-cuda",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_spine_bf16_cuda_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_spine_bf16_cuda=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "metal-source",
        platform: NativeTestPlatform::Macos,
        main_source: "native/provider_gate/provider_metal_source_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::MetalSourceDevelopment,
        cases: METAL_SOURCE_CASES,
        pass_marker: "provider_metal_source=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "route-mailbox",
        platform: NativeTestPlatform::Macos,
        main_source: "native/provider_gate/provider_route_mailbox_test.mm",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_route_mailbox.mps=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "spine-bf16-metal",
        platform: NativeTestPlatform::Macos,
        main_source: "native/provider_gate/provider_spine_bf16_metal_test.mm",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_spine_bf16_metal.mps=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "spine-int8-metal",
        platform: NativeTestPlatform::Macos,
        main_source: "native/provider_gate/provider_spine_int8_metal_test.mm",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_spine_int8_metal.mps=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "spine-int8-arena",
        platform: NativeTestPlatform::Macos,
        main_source: "native/provider_gate/provider_spine_int8_arena_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_spine_int8_arena.mps=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "spine-bf16-arena",
        platform: NativeTestPlatform::Macos,
        main_source: "native/provider_gate/provider_spine_bf16_arena_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: DEFAULT_CASES,
        pass_marker: "provider_spine_bf16_arena=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "target",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_target_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::Production,
        cases: CPU_AND_MPS_CASES,
        pass_marker: "provider_target.residual=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "target-tape",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_target_tape_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::SyntheticMoe,
        cases: DEFAULT_CASES,
        pass_marker: "provider_target_tape.schedule=PASS",
        timeout_seconds: 300,
    },
    NativeTestSpec {
        name: "target-sequence",
        platform: NativeTestPlatform::Any,
        main_source: "native/provider_gate/provider_target_sequence_test.cpp",
        extra_sources: &[],
        provider: ProviderFlavor::SyntheticMoe,
        cases: DEFAULT_CASES,
        pass_marker: "provider target sequence: PASS",
        timeout_seconds: 600,
    },
];

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NativeBuildError {
    message: String,
}

impl NativeBuildError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for NativeBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for NativeBuildError {}

type NativeBuildResult<T> = Result<T, NativeBuildError>;

#[derive(Debug)]
pub struct NativeTestCaseReport {
    pub name: &'static str,
    pub stdout: String,
}

#[derive(Debug)]
pub struct NativeTestReport {
    pub name: &'static str,
    pub executable: PathBuf,
    pub cases: Vec<NativeTestCaseReport>,
}

#[derive(Debug)]
struct ProviderBuildArtifacts {
    archive: PathBuf,
    torch_root: PathBuf,
    torch_lib: PathBuf,
    native_build: PathBuf,
    generated_include: PathBuf,
    definitions: Vec<&'static str>,
    toolchain: NativeToolchain,
    guard: PythonGuard,
    cuda_provider: Option<CudaProviderBuild>,
    cuda: Option<CudaBuild>,
    gpu_runtime: Option<GpuRuntime>,
}

pub fn run_production_build() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RUNTIME");
    if env::var_os("CARGO_FEATURE_RUNTIME").is_none() {
        // Library-only tokenizer consumers must remain cheap to build and
        // load. In particular, do not discover LibTorch, invoke a native
        // compiler, or emit provider link directives for this feature set.
        return;
    }
    println!("cargo:rerun-if-env-changed=DELTAFIN_NATIVE_BUILD_CHECK_ONLY");
    if env::var_os("DELTAFIN_NATIVE_BUILD_CHECK_ONLY").is_some_and(|value| value == "1") {
        // Developer-only, for `cargo check` against a target this host cannot
        // build natively (the Windows target from macOS, say): type-check the
        // Rust without configuring LibTorch or compiling the provider. No
        // provider archive or link directive is produced, so nothing built
        // this way can link, let alone run.
        println!(
            "cargo:warning=DELTAFIN_NATIVE_BUILD_CHECK_ONLY=1 skipped the native provider build; this configuration can only be type-checked"
        );
        return;
    }

    let manifest_dir = PathBuf::from(required_env("CARGO_MANIFEST_DIR"));
    let repository = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("native/deltafin must remain inside the repository")
        .to_path_buf();
    println!("cargo:rerun-if-env-changed=DELTAFIN_TORCH_ROOT");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=DELTAFIN_CUDA_MOE");
    println!("cargo:rerun-if-env-changed=DELTAFIN_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=DELTAFIN_HIP_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=DELTAFIN_HIPCC");
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_COMPILER");
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=PATH");
    let explicit_torch_root = ["DELTAFIN_TORCH_ROOT", "LIBTORCH"]
        .into_iter()
        .any(|variable| env::var_os(variable).is_some());
    let output_dir = PathBuf::from(required_env("OUT_DIR"));
    let native_build = output_dir.join("provider-native");
    let artifacts = build_provider_artifacts(&repository, &native_build, true)
        .unwrap_or_else(|error| panic!("Rust-owned production provider build failed: {error}"));

    // Ordering is intentional: the archive's unresolved ATen symbols precede
    // its shared provider libraries on linkers that still honor left-to-right
    // resolution. Python libraries are never considered or emitted.
    println!(
        "cargo:rustc-link-search=native={}",
        artifacts.native_build.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        artifacts.torch_lib.display()
    );
    println!("cargo:rustc-link-lib=static=deltafin_provider_abi");
    link_required(&artifacts.torch_lib, "torch");
    link_required(&artifacts.torch_lib, "torch_cpu");
    let torch_cuda = library_file(&artifacts.torch_lib, "torch_cuda");
    let c10_cuda = library_file(&artifacts.torch_lib, "c10_cuda");
    if torch_cuda.is_some() != c10_cuda.is_some() {
        panic!(
            "selected native PyTorch root has an incomplete CUDA pair in {}: libtorch_cuda and libc10_cuda must either both exist or both be absent",
            artifacts.torch_lib.display()
        );
    }
    if torch_cuda.is_some() {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
        match artifacts
            .gpu_runtime
            .expect("GPU libraries require an identified runtime ABI")
        {
            GpuRuntime::Cuda(major) => {
                println!("cargo:rustc-env=DELTAFIN_LIBTORCH_CUDA_MAJOR={major}");
            }
            GpuRuntime::Rocm => println!("cargo:rustc-env=DELTAFIN_LIBTORCH_HIP=1"),
        }
    }
    link_optional(&artifacts.torch_lib, "torch_cuda_linalg");
    link_required(&artifacts.torch_lib, "c10");

    if let Some(provider) = &artifacts.cuda_provider {
        println!(
            "cargo:rustc-link-search=native={}",
            provider.runtime_directory.display()
        );
        println!("cargo:rustc-link-lib=dylib=cudart");
        emit_rpath(&provider.runtime_directory);
    }
    if let Some(cuda) = &artifacts.cuda {
        if cuda.runtime == KernelRuntime::Cuda
            && artifacts
                .cuda_provider
                .as_ref()
                .is_none_or(|provider| provider.runtime_directory != cuda.runtime_directory)
        {
            panic!(
                "CUDA provider and NVCC selected different runtimes: {} versus {}",
                artifacts.cuda_provider.as_ref().map_or_else(
                    || "none".to_owned(),
                    |provider| provider.runtime_directory.display().to_string()
                ),
                cuda.runtime_directory.display()
            );
        }
        if cuda.runtime == KernelRuntime::Hip {
            // A HIP object references the HIP runtime directly, and the linker
            // will not resolve those symbols through libtorch_cuda's own
            // DT_NEEDED entry, so libamdhip64 is named here the way libcudart
            // is named for NVIDIA.
            println!(
                "cargo:rustc-link-search=native={}",
                cuda.runtime_directory.display()
            );
            println!("cargo:rustc-link-lib=dylib=amdhip64");
            emit_rpath(&cuda.runtime_directory);
        }
        println!(
            "cargo:rustc-env=DELTAFIN_GPU_KERNEL_RUNTIME={}",
            cuda.runtime.label()
        );
        println!(
            "cargo:rustc-env=DELTAFIN_CUDA_TOOLKIT={}",
            cuda.toolkit_version
        );
        println!(
            "cargo:rustc-env=DELTAFIN_GPU_KERNEL_ARCHITECTURES={}",
            cuda.architectures
        );
        if cuda.runtime == KernelRuntime::Cuda {
            println!(
                "cargo:rustc-env=DELTAFIN_CUDA_ARCHITECTURES={}",
                cuda.architectures
            );
        }
    }
    // `deltafin upgrade` reproduces a build from NVCC-shaped variables and
    // validates the architecture list as numeric compute capabilities. A HIP
    // kernel build fits none of that, so it is deliberately absent from the
    // reproducible profile and `upgrade` refuses it outright rather than
    // replaying a different build.
    let effective_cuda = artifacts
        .cuda
        .as_ref()
        .filter(|cuda| cuda.runtime == KernelRuntime::Cuda)
        .map(|cuda| (cuda.architectures.clone(), cuda.compiler.clone()));
    emit_upgrade_build_profile(
        &artifacts.torch_root,
        explicit_torch_root,
        effective_cuda.as_ref(),
        artifacts.cuda_provider.is_some(),
    );

    match target_os().as_str() {
        "macos" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        "linux" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        // The MSVC C++ runtime comes in through the /MD objects' default-library
        // directives; there is no `stdc++` to name. A Windows executable's
        // main thread gets 1 MiB unless asked for more.
        "windows" => println!("cargo:rustc-link-arg=/STACK:{NATIVE_TEST_STACK_BYTES}"),
        target => panic!("native provider ABI does not support target OS {target}"),
    }
    if target_os() == "windows" {
        // Windows has no RPATH: the loader finds LibTorch's DLLs beside the
        // executable, so the authenticated ones are placed there. A CUDA build
        // additionally needs the toolkit's `cudart64_*` runtime beside it.
        let mut sources = vec![artifacts.torch_lib.clone()];
        if let Some(provider) = &artifacts.cuda_provider {
            sources.push(provider.runtime_directory.clone());
        }
        deploy_runtime_libraries(&sources, &PathBuf::from(required_env("OUT_DIR")));
    } else {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            artifacts.torch_lib.display()
        );
    }
}

/// `-Wl,-rpath` is a GNU-ld spelling; Windows `link.exe` rejects it and relies
/// on the deployed DLLs instead.
fn emit_rpath(directory: &Path) {
    if target_os() != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
    }
}

/// Place LibTorch's DLLs (and, for a CUDA build, the toolkit's `cudart64_*`
/// runtime) where a Windows process will find them: beside the executable Cargo
/// produces and beside the test executables in `deps/`. The files come from the
/// authenticated toolchain (or the operator's root) and are linked rather than
/// copied where the volume allows, so a 300 MB library is not duplicated per
/// profile.
fn deploy_runtime_libraries(sources: &[PathBuf], out_dir: &Path) {
    // OUT_DIR is <target>/<profile>/build/<package>-<hash>/out.
    let layout_ok = out_dir
        .ancestors()
        .nth(2)
        .and_then(Path::file_name)
        .is_some_and(|name| name == "build");
    let Some(profile_directory) = out_dir.ancestors().nth(3).filter(|_| layout_ok) else {
        panic!(
            "cannot locate the Cargo profile directory from OUT_DIR {}",
            out_dir.display()
        );
    };
    let mut libraries: Vec<PathBuf> = Vec::new();
    for source in sources {
        let entries = fs::read_dir(source).unwrap_or_else(|error| {
            panic!("list runtime libraries in {}: {error}", source.display())
        });
        for entry in entries {
            let path = entry
                .unwrap_or_else(|error| {
                    panic!(
                        "read runtime library directory {}: {error}",
                        source.display()
                    )
                })
                .path();
            if path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
                && !libraries
                    .iter()
                    .any(|existing| existing.file_name() == path.file_name())
            {
                libraries.push(path);
            }
        }
    }
    libraries.sort();
    for directory in [
        profile_directory.to_path_buf(),
        profile_directory.join("deps"),
    ] {
        fs::create_dir_all(&directory)
            .unwrap_or_else(|error| panic!("create {}: {error}", directory.display()));
        for library in &libraries {
            reject_python_dependency(library).unwrap_or_else(|error| panic!("{error}"));
            let name = library.file_name().expect("a listed file has a name");
            deploy_one_library(library, &directory.join(name));
        }
    }
    // A deleted copy must bring this script back.
    for library in &libraries {
        let name = library.file_name().expect("a listed file has a name");
        println!(
            "cargo:rerun-if-changed={}",
            profile_directory.join(name).display()
        );
    }
}

fn deploy_one_library(source: &Path, destination: &Path) {
    let source_metadata = fs::symlink_metadata(source)
        .unwrap_or_else(|error| panic!("inspect {}: {error}", source.display()));
    if source_metadata.file_type().is_symlink() || !source_metadata.is_file() {
        panic!(
            "LibTorch library must be a non-symlink regular file: {}",
            source.display()
        );
    }
    if let Ok(existing) = fs::symlink_metadata(destination) {
        if existing.file_type().is_symlink() || !existing.is_file() {
            panic!(
                "refusing to replace a non-regular deployed library: {}",
                destination.display()
            );
        }
        if existing.len() == source_metadata.len() && files_identical(source, destination) {
            return;
        }
        // A library the running product still has mapped cannot be replaced;
        // that error names the file rather than half-updating a deployment.
        fs::remove_file(destination).unwrap_or_else(|error| {
            panic!(
                "replace deployed library {} (is a Deltafin process still running?): {error}",
                destination.display()
            )
        });
    }
    if fs::hard_link(source, destination).is_err() {
        fs::copy(source, destination).unwrap_or_else(|error| {
            panic!(
                "deploy {} to {}: {error}",
                source.display(),
                destination.display()
            )
        });
    }
}

fn files_identical(left: &Path, right: &Path) -> bool {
    fn digest(path: &Path) -> Option<[u8; 32]> {
        let mut file = fs::File::open(path).ok()?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1 << 20];
        loop {
            let count = file.read(&mut buffer).ok()?;
            if count == 0 {
                return Some(hasher.finalize().into());
            }
            hasher.update(&buffer[..count]);
        }
    }
    match (digest(left), digest(right)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

/// Build and execute one or every isolated native provider test through the
/// same admitted compiler, archive, feature definitions and link closure used
/// by production. `all` builds the production archive once, then links each
/// independent test main (and only its explicit test-flavor overrides).
pub fn run_native_tests(
    repository: &Path,
    output_root: &Path,
    requested: &str,
) -> NativeBuildResult<Vec<NativeTestReport>> {
    let repository = deltafin_sys::fs::canonicalize(repository).map_err(|error| {
        NativeBuildError::new(format!(
            "resolve native-test repository {}: {error}",
            repository.display()
        ))
    })?;
    if !repository.join("Cargo.toml").is_file() || !repository.join("native/provider_gate").is_dir()
    {
        return Err(NativeBuildError::new(format!(
            "native-test root is not a Deltafin checkout: {}",
            repository.display()
        )));
    }
    let host_os = target_os();
    let selected: Vec<&NativeTestSpec> = if requested == "all" {
        NATIVE_TEST_SPECS
            .iter()
            .filter(|spec| platform_matches(spec.platform, &host_os))
            .collect()
    } else {
        let spec = NATIVE_TEST_SPECS
            .iter()
            .find(|spec| spec.name == requested)
            .ok_or_else(|| NativeBuildError::new(format!("unknown native test {requested:?}")))?;
        if !platform_matches(spec.platform, &host_os) {
            return Err(NativeBuildError::new(format!(
                "native test {} does not support host OS {host_os}",
                spec.name
            )));
        }
        vec![spec]
    };

    let target_root = repository.join("target");
    fs::create_dir_all(&target_root).map_err(|error| {
        NativeBuildError::new(format!(
            "create Cargo target directory {}: {error}",
            target_root.display()
        ))
    })?;
    let target_root = deltafin_sys::fs::canonicalize(&target_root).map_err(|error| {
        NativeBuildError::new(format!(
            "resolve Cargo target directory {}: {error}",
            target_root.display()
        ))
    })?;
    let requested_root = if output_root.is_absolute() {
        output_root.to_path_buf()
    } else {
        repository.join(output_root)
    };
    fs::create_dir_all(&requested_root).map_err(|error| {
        NativeBuildError::new(format!(
            "create native-test output directory {}: {error}",
            requested_root.display()
        ))
    })?;
    let output_root = deltafin_sys::fs::canonicalize(&requested_root).map_err(|error| {
        NativeBuildError::new(format!(
            "resolve native-test output directory {}: {error}",
            requested_root.display()
        ))
    })?;
    if !output_root.starts_with(&target_root) {
        return Err(NativeBuildError::new(format!(
            "native-test output must remain under canonical Cargo target root {}: {}",
            target_root.display(),
            output_root.display()
        )));
    }

    let needs_provider = selected
        .iter()
        .any(|spec| spec.provider != ProviderFlavor::None);
    let provider = if needs_provider {
        let provider_build = canonical_child_output(&output_root, "provider-production")?;
        Some(build_provider_artifacts(
            &repository,
            &provider_build,
            false,
        )?)
    } else {
        None
    };
    // Every selected spec runs before a failure is reported: a matrix that
    // stops at its first red spec costs one full run per spec to learn what
    // one run could have said, which on a slow or metered host is the
    // difference between a day and an hour of porting.
    let mut reports = Vec::with_capacity(selected.len());
    let mut failures: Vec<String> = Vec::new();
    for spec in selected {
        match build_and_run_native_test(&repository, &output_root, spec, provider.as_ref()) {
            Ok(report) => reports.push(report),
            Err(error) => {
                eprintln!("[xtask] {}: FAIL", spec.name);
                failures.push(format!("native test {} failed: {error}", spec.name));
            }
        }
    }
    if !failures.is_empty() {
        return Err(NativeBuildError::new(format!(
            "{} of {} native test(s) failed ({} passed: {}):\n\n{}",
            failures.len(),
            failures.len() + reports.len(),
            reports.len(),
            reports
                .iter()
                .map(|report| report.name)
                .collect::<Vec<_>>()
                .join(", "),
            failures.join("\n\n")
        )));
    }
    Ok(reports)
}

pub fn run_native_test(
    repository: &Path,
    output_root: &Path,
    requested: &str,
) -> NativeBuildResult<NativeTestReport> {
    let mut reports = run_native_tests(repository, output_root, requested)?;
    if reports.len() != 1 {
        return Err(NativeBuildError::new(
            "run_native_test accepts exactly one named test; use run_native_tests for all",
        ));
    }
    Ok(reports.remove(0))
}

fn platform_matches(platform: NativeTestPlatform, host_os: &str) -> bool {
    matches!(platform, NativeTestPlatform::Any)
        || matches!(
            (platform, host_os),
            (NativeTestPlatform::Macos, "macos")
                | (NativeTestPlatform::Linux, "linux")
                | (NativeTestPlatform::Windows, "windows")
        )
}

fn canonical_child_output(parent: &Path, child: &str) -> NativeBuildResult<PathBuf> {
    let requested = parent.join(child);
    fs::create_dir_all(&requested).map_err(|error| {
        NativeBuildError::new(format!(
            "create native output directory {}: {error}",
            requested.display()
        ))
    })?;
    let resolved = deltafin_sys::fs::canonicalize(&requested).map_err(|error| {
        NativeBuildError::new(format!(
            "resolve native output directory {}: {error}",
            requested.display()
        ))
    })?;
    if !resolved.starts_with(parent) {
        return Err(NativeBuildError::new(format!(
            "native output path escaped its canonical root {}: {}",
            parent.display(),
            resolved.display()
        )));
    }
    Ok(resolved)
}

/// One translation unit of the provider archive.
struct CompileJob<'a> {
    kind: CompileKind<'a>,
    source: PathBuf,
    object: PathBuf,
}

enum CompileKind<'a> {
    Cpp {
        generated_include: Option<&'a Path>,
        cuda_include: Option<&'a Path>,
        objective_cpp: bool,
    },
    Kernel,
}

/// Run every job, a few at a time, and return the objects in job order, or
/// the message of every job that failed.
fn run_compile_jobs<'a>(
    jobs: &[CompileJob<'a>],
    compile: impl Fn(&CompileJob<'a>) -> Result<(), String> + Sync,
) -> Result<Vec<PathBuf>, Vec<String>> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let workers = thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .clamp(1, 8)
        .min(jobs.len().max(1));
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<(), String>>>> =
        Mutex::new((0..jobs.len()).map(|_| None).collect());
    thread::scope(|scope| {
        for worker in 0..workers {
            thread::Builder::new()
                .name(format!("native-compile-{worker}"))
                .spawn_scoped(scope, || {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(index) else {
                            return;
                        };
                        let result = compile(job);
                        results.lock().expect("compile results lock")[index] = Some(result);
                    }
                })
                .expect("start a native compile worker thread");
        }
    });
    let results = results.into_inner().expect("compile results lock");
    let mut objects = Vec::with_capacity(jobs.len());
    let mut failures = Vec::new();
    for (job, result) in jobs.iter().zip(results) {
        match result.expect("every job ran") {
            Ok(()) => objects.push(job.object.clone()),
            Err(message) => failures.push(message),
        }
    }
    if failures.is_empty() {
        Ok(objects)
    } else {
        Err(failures)
    }
}

fn build_provider_artifacts(
    repository: &Path,
    native_build: &Path,
    emit_cargo_metadata: bool,
) -> NativeBuildResult<ProviderBuildArtifacts> {
    let provider_source = repository.join("native/provider_gate");
    let torch_root = find_torch_root(repository, emit_cargo_metadata);
    validate_torch_version(&torch_root, emit_cargo_metadata);
    let torch_lib = torch_root.join("lib");
    let libtorch_cxx11_abi = detect_libtorch_cxx11_abi(&torch_lib);
    let gpu_runtime = detect_gpu_runtime(&torch_lib);
    // Only NVIDIA needs the separate cudart toolkit discovery; a ROCm tree
    // resolves HIP through its own runtime and has no cudart to find.
    let cuda_provider = gpu_runtime
        .filter(|runtime| matches!(runtime, GpuRuntime::Cuda(_)))
        .map(|_| find_cuda_provider(&torch_root));
    fs::create_dir_all(native_build).unwrap_or_else(|error| {
        panic!(
            "create Rust-owned provider build directory {}: {error}",
            native_build.display()
        )
    });
    let native_build = deltafin_sys::fs::canonicalize(native_build).unwrap_or_else(|error| {
        panic!(
            "resolve Rust-owned provider build directory {}: {error}",
            native_build.display()
        )
    });

    if emit_cargo_metadata {
        println!(
            "cargo:rerun-if-changed={}",
            provider_source.join("deny_python.c.in").display()
        );
        for relative in PRODUCTION_PROVIDER_SOURCES {
            println!(
                "cargo:rerun-if-changed={}",
                repository.join(relative).display()
            );
        }
        // A new provider header or translation unit must invalidate Cargo even
        // before the reviewed source classification below is updated.
        println!("cargo:rerun-if-changed={}", provider_source.display());
    }

    let toolchain = NativeToolchain::discover();
    let mut guard = PythonGuard::build(&native_build, &toolchain, &provider_source);
    guard.loader_directories.push(torch_lib.clone());
    if let Some(provider) = &cuda_provider {
        guard
            .loader_directories
            .push(provider.runtime_directory.clone());
    }
    validate_compiler(&toolchain, &toolchain.cc, "C", &guard);
    validate_compiler(&toolchain, &toolchain.cxx, "C++", &guard);

    let generated_include = native_build.join("generated");
    let host_os = target_os();
    match host_os.as_str() {
        "macos" => build_embedded_metal_libraries(
            repository,
            &provider_source,
            &native_build,
            &generated_include,
            &guard,
            emit_cargo_metadata,
        ),
        "linux" | "windows" => {}
        target => panic!("native provider ABI does not support target OS {target}"),
    }

    let cuda = build_cuda_kernel(
        repository,
        &native_build,
        &toolchain,
        &guard,
        gpu_runtime,
        emit_cargo_metadata,
    );
    let mut definitions = vec!["DELTAFIN_HAVE_MXFP4_CPU_V1=1"];
    if cuda_provider.is_some() {
        definitions.push("DELTAFIN_HAVE_CUDA_PROVIDER_V1=1");
    }
    if cuda.is_some() {
        definitions.extend([
            "DELTAFIN_HAVE_CUDA_MOE_V1=1",
            "DELTAFIN_HAVE_CUDA_SPINE_BF16_V1=1",
        ]);
    }
    if host_os == "macos" {
        definitions.extend([
            "DELTAFIN_HAVE_METAL_MOE_V1=1",
            "DELTAFIN_HAVE_MPS_ROUTE_MAILBOX_V1=1",
            "DELTAFIN_HAVE_SPINE_BF16_METAL_V1=1",
            "DELTAFIN_HAVE_SPINE_INT8_METAL_V1=1",
            "DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1=1",
        ]);
    } else if host_os == "linux" {
        definitions.push(match libtorch_cxx11_abi {
            Some(0) => "_GLIBCXX_USE_CXX11_ABI=0",
            Some(1) => "_GLIBCXX_USE_CXX11_ABI=1",
            _ => unreachable!("Linux LibTorch C++ ABI was validated before compilation"),
        });
    }
    definitions.extend([
        "USE_C10D_GLOO",
        "USE_DISTRIBUTED",
        "USE_RPC",
        "USE_TENSORPIPE",
    ]);

    // Every translation unit is compiled before a failure is reported, so one
    // run shows all of what is wrong rather than the first of it (a port to a
    // new compiler would otherwise cost one full build per diagnostic), and
    // the independent units are compiled concurrently. Objects keep the order
    // of this list, so the archive is the same whatever the schedule.
    let mut jobs: Vec<CompileJob> = Vec::new();
    for source in PROVIDER_CPP_SOURCES {
        jobs.push(CompileJob {
            kind: CompileKind::Cpp {
                generated_include: (host_os == "macos").then_some(generated_include.as_path()),
                cuda_include: cuda_provider
                    .as_ref()
                    .map(|provider| provider.include_directory.as_path()),
                objective_cpp: false,
            },
            source: provider_source.join(source),
            object: native_build.join(object_file_name(
                source.strip_suffix(".cpp").expect("provider C++ suffix"),
            )),
        });
    }
    jobs.push(CompileJob {
        kind: CompileKind::Kernel,
        source: repository.join("tools/fused_gemv_batch.c"),
        object: native_build.join(object_file_name("fused_gemv_batch")),
    });
    if host_os == "macos" {
        for (source, object) in [
            (
                provider_source.join("provider_route_mailbox.mm"),
                native_build.join("provider_route_mailbox.o"),
            ),
            (
                provider_source.join("provider_spine_bf16_metal.mm"),
                native_build.join("provider_spine_bf16_metal.o"),
            ),
            (
                provider_source.join("provider_spine_int8_metal.mm"),
                native_build.join("provider_spine_int8_metal.o"),
            ),
            (
                repository.join("tools/metal_moe.mm"),
                native_build.join("metal_moe.o"),
            ),
        ] {
            jobs.push(CompileJob {
                kind: CompileKind::Cpp {
                    generated_include: Some(generated_include.as_path()),
                    cuda_include: None,
                    objective_cpp: true,
                },
                source,
                object,
            });
        }
    }
    let mut objects = run_compile_jobs(&jobs, |job| match job.kind {
        CompileKind::Cpp {
            generated_include,
            cuda_include,
            objective_cpp,
        } => compile_cpp(
            &toolchain,
            &job.source,
            &job.object,
            &torch_root,
            generated_include,
            cuda_include,
            &definitions,
            objective_cpp,
            &guard,
        ),
        CompileKind::Kernel => compile_gemv(&toolchain, &job.source, &job.object, &guard),
    })
    .unwrap_or_else(|failures| {
        panic!(
            "{} native translation unit(s) failed to compile:\n\n{}",
            failures.len(),
            failures.join("\n\n")
        )
    });
    if let Some(cuda) = &cuda {
        objects.push(cuda.object.clone());
        objects.push(cuda.spine_object.clone());
    }
    let archive = native_build.join(provider_archive_name());
    archive_objects(&toolchain, &archive, &objects, &guard);

    if let Some(cuda) = &cuda {
        if cuda.runtime == KernelRuntime::Cuda
            && cuda_provider
                .as_ref()
                .is_none_or(|provider| provider.runtime_directory != cuda.runtime_directory)
        {
            panic!(
                "CUDA provider and NVCC selected different runtimes: {} versus {}",
                cuda_provider.as_ref().map_or_else(
                    || "none".to_owned(),
                    |provider| provider.runtime_directory.display().to_string()
                ),
                cuda.runtime_directory.display()
            );
        }
    }

    Ok(ProviderBuildArtifacts {
        archive,
        torch_root,
        torch_lib,
        native_build,
        generated_include,
        definitions,
        toolchain,
        guard,
        cuda_provider,
        cuda,
        gpu_runtime,
    })
}

fn build_and_run_native_test(
    repository: &Path,
    output_root: &Path,
    spec: &NativeTestSpec,
    provider: Option<&ProviderBuildArtifacts>,
) -> NativeBuildResult<NativeTestReport> {
    let build = canonical_child_output(output_root, spec.name)?;

    if spec.provider == ProviderFlavor::None {
        return build_and_run_cpu_only_test(repository, &build, spec);
    }
    let provider = provider.ok_or_else(|| {
        NativeBuildError::new(format!(
            "native test {} requires the shared production provider archive",
            spec.name
        ))
    })?;

    let mut definitions = provider.definitions.clone();
    if spec.provider == ProviderFlavor::CudaResidency {
        definitions.push("DELTAFIN_PROVIDER_CUDA_MOE_TESTING=1");
    }
    let include = if spec.provider == ProviderFlavor::MetalSourceDevelopment {
        definitions
            .retain(|definition| *definition != "DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1=1");
        definitions.push("DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1=1");
        let include = build.join("generated-metal-source");
        write_embedded_metal_source_header(
            &repository.join("tools/metal/moe_mxfp4.metal"),
            &include.join("deltafin_embedded_moe_mxfp4_msl.h"),
        )?;
        include
    } else {
        provider.generated_include.clone()
    };

    let mut objects = Vec::new();
    let main = repository.join(spec.main_source);
    let main_object = build.join(object_file_name("test-main"));
    compile_test_source(&main, &main_object, &definitions, &include, provider)
        .map_err(NativeBuildError::new)?;
    objects.push(main_object);

    for (index, relative) in spec.extra_sources.iter().enumerate() {
        let source = repository.join(relative);
        let object = build.join(object_file_name(&format!("test-extra-{index}")));
        compile_test_source(&source, &object, &definitions, &include, provider)
            .map_err(NativeBuildError::new)?;
        objects.push(object);
    }

    match spec.provider {
        ProviderFlavor::SyntheticMoe => {
            let mut synthetic = definitions.clone();
            synthetic.push("DELTAFIN_PROVIDER_MOE_TESTING=1");
            let object = build.join(object_file_name("provider_moe_test_flavor"));
            compile_test_source(
                &repository.join("native/provider_gate/provider_moe.cpp"),
                &object,
                &synthetic,
                &include,
                provider,
            )
            .map_err(NativeBuildError::new)?;
            objects.push(object);
        }
        ProviderFlavor::CudaResidency => {
            let object = build.join(object_file_name("provider_cuda_moe_residency_test_flavor"));
            compile_test_source(
                &repository.join("native/provider_gate/provider_cuda_moe.cpp"),
                &object,
                &definitions,
                &include,
                provider,
            )
            .map_err(NativeBuildError::new)?;
            objects.push(object);
        }
        ProviderFlavor::MetalSourceDevelopment => {
            let object = build.join(object_file_name("metal_moe_source_development_flavor"));
            compile_test_source(
                &repository.join("tools/metal_moe.mm"),
                &object,
                &definitions,
                &include,
                provider,
            )
            .map_err(NativeBuildError::new)?;
            objects.push(object);
        }
        ProviderFlavor::Production => {}
        ProviderFlavor::None => unreachable!("CPU-only flavor returned above"),
    }

    let executable = build.join(native_test_executable_name(spec.name));
    link_provider_test(&objects, &executable, provider, spec)?;
    audit_transitive_dependencies(&executable, provider, &build)?;
    run_native_test_cases(repository, &build, spec, &executable, &provider.guard)
}

fn build_and_run_cpu_only_test(
    repository: &Path,
    build: &Path,
    spec: &NativeTestSpec,
) -> NativeBuildResult<NativeTestReport> {
    let provider_source = repository.join("native/provider_gate");
    let toolchain = NativeToolchain::discover();
    let guard = PythonGuard::build(build, &toolchain, &provider_source);
    validate_compiler(&toolchain, &toolchain.cc, "C", &guard);
    let mut objects = Vec::new();
    let main_object = build.join(object_file_name("test-main"));
    compile_c_test_main(
        &toolchain,
        &repository.join(spec.main_source),
        &main_object,
        &guard,
    )
    .map_err(NativeBuildError::new)?;
    objects.push(main_object);
    for (index, relative) in spec.extra_sources.iter().enumerate() {
        let object = build.join(object_file_name(&format!("test-extra-{index}")));
        compile_gemv(&toolchain, &repository.join(relative), &object, &guard)
            .map_err(NativeBuildError::new)?;
        objects.push(object);
    }
    let executable = build.join(native_test_executable_name(spec.name));
    prepare_generated_output(&executable, "native test executable");
    let linker_program: &Path = match toolchain.flavor {
        CompilerFlavor::Gnu => &toolchain.cc,
        CompilerFlavor::Msvc => toolchain
            .linker
            .as_deref()
            .expect("an MSVC toolchain always carries link.exe"),
    };
    let mut link = Command::new(linker_program);
    match toolchain.flavor {
        CompilerFlavor::Gnu => {
            link.args(&objects).arg("-o").arg(&executable);
            if target_os() == "linux" {
                link.args(["-pthread", "-lm"]);
            }
        }
        CompilerFlavor::Msvc => {
            // The kernels have no import libraries of their own to name.
            link.args(msvc::link_args(
                &objects,
                None,
                &executable,
                &[],
                &[],
                NATIVE_TEST_STACK_BYTES,
            ));
        }
    }
    run_guarded_try(
        &mut link,
        &format!("link isolated native test {}", spec.name),
        &guard,
    )
    .map_err(NativeBuildError::new)?;
    validate_native_executable(&executable, "isolated native test");
    audit_standalone_dependencies(&executable, build, &guard)?;
    run_native_test_cases(repository, build, spec, &executable, &guard)
}

fn compile_test_source(
    source: &Path,
    object: &Path,
    definitions: &[&str],
    generated_include: &Path,
    provider: &ProviderBuildArtifacts,
) -> Result<(), String> {
    match source.extension().and_then(OsStr::to_str) {
        Some("c") if source.file_name() == Some(OsStr::new("fused_gemv_batch.c")) => {
            compile_gemv(&provider.toolchain, source, object, &provider.guard)
        }
        Some("c") => compile_c_test_main(&provider.toolchain, source, object, &provider.guard),
        Some("cpp" | "mm") => compile_cpp(
            &provider.toolchain,
            source,
            object,
            &provider.torch_root,
            (target_os() == "macos").then_some(generated_include),
            provider
                .cuda_provider
                .as_ref()
                .map(|cuda| cuda.include_directory.as_path()),
            definitions,
            source.extension() == Some(OsStr::new("mm")),
            &provider.guard,
        ),
        extension => panic!(
            "native test source has an unclassified extension {extension:?}: {}",
            source.display()
        ),
    }
}

fn link_provider_test(
    objects: &[PathBuf],
    executable: &Path,
    provider: &ProviderBuildArtifacts,
    spec: &NativeTestSpec,
) -> NativeBuildResult<()> {
    prepare_generated_output(executable, "native test executable");
    if provider.toolchain.flavor == CompilerFlavor::Msvc {
        return link_provider_test_msvc(objects, executable, provider, spec);
    }
    let mut link = Command::new(&provider.toolchain.cxx);
    link.args(objects)
        .arg(&provider.archive)
        .arg("-L")
        .arg(&provider.torch_lib)
        .args(["-ltorch", "-ltorch_cpu"]);
    if library_file(&provider.torch_lib, "torch_cuda").is_some() {
        link.args(["-ltorch_cuda", "-lc10_cuda"]);
    }
    if library_file(&provider.torch_lib, "torch_cuda_linalg").is_some() {
        link.arg("-ltorch_cuda_linalg");
    }
    link.arg("-lc10");
    if let Some(cuda) = &provider.cuda_provider {
        link.arg("-L")
            .arg(&cuda.runtime_directory)
            .arg("-lcudart")
            .arg(format!("-Wl,-rpath,{}", cuda.runtime_directory.display()));
    }
    if let Some(cuda) = provider
        .cuda
        .as_ref()
        .filter(|cuda| cuda.runtime == KernelRuntime::Hip)
    {
        link.arg("-L")
            .arg(&cuda.runtime_directory)
            .arg("-lamdhip64")
            .arg(format!("-Wl,-rpath,{}", cuda.runtime_directory.display()));
    }
    match target_os().as_str() {
        "macos" => {
            // clang++ already supplies libc++; spelling it twice emits a
            // warning while producing the same dependency closure.
            link.args(["-framework", "Metal", "-framework", "Foundation"]);
        }
        "linux" => {
            link.args(["-pthread", "-lm"]);
        }
        target => {
            return Err(NativeBuildError::new(format!(
                "native provider tests do not support target OS {target}"
            )));
        }
    }
    link.arg(format!("-Wl,-rpath,{}", provider.torch_lib.display()))
        .arg("-o")
        .arg(executable);
    run_guarded_try(
        &mut link,
        &format!("link isolated native test {}", spec.name),
        &provider.guard,
    )
    .map_err(NativeBuildError::new)?;
    validate_native_executable(executable, "isolated native test");
    Ok(())
}

/// Link an isolated provider test with `link.exe`: the test's objects, the
/// provider archive, then LibTorch's import libraries.
fn link_provider_test_msvc(
    objects: &[PathBuf],
    executable: &Path,
    provider: &ProviderBuildArtifacts,
    spec: &NativeTestSpec,
) -> NativeBuildResult<()> {
    let linker = provider
        .toolchain
        .linker
        .as_ref()
        .expect("an MSVC toolchain always carries link.exe");
    let mut libraries = vec!["torch.lib", "torch_cpu.lib"];
    if library_file(&provider.torch_lib, "torch_cuda").is_some() {
        libraries.extend(["torch_cuda.lib", "c10_cuda.lib"]);
    }
    if library_file(&provider.torch_lib, "torch_cuda_linalg").is_some() {
        libraries.push("torch_cuda_linalg.lib");
    }
    libraries.push("c10.lib");
    let mut directories = vec![provider.torch_lib.as_path()];
    if let Some(cuda) = &provider.cuda_provider {
        directories.push(cuda.runtime_directory.as_path());
        libraries.push("cudart.lib");
    }
    let mut link = Command::new(linker);
    link.args(msvc::link_args(
        objects,
        Some(&provider.archive),
        executable,
        &directories,
        &libraries,
        NATIVE_TEST_STACK_BYTES,
    ));
    run_guarded_try(
        &mut link,
        &format!("link isolated native test {}", spec.name),
        &provider.guard,
    )
    .map_err(NativeBuildError::new)?;
    validate_native_executable(executable, "isolated native test");
    Ok(())
}

fn run_native_test_cases(
    repository: &Path,
    build: &Path,
    spec: &NativeTestSpec,
    executable: &Path,
    guard: &PythonGuard,
) -> NativeBuildResult<NativeTestReport> {
    let host_os = target_os();
    let mut reports = Vec::new();
    for case in spec
        .cases
        .iter()
        .filter(|case| platform_matches(case.platform, &host_os))
    {
        let mut command = Command::new(executable);
        for argument in case.arguments {
            command.arg(argument.replace("{REPOSITORY}", &repository.display().to_string()));
        }
        sanitize_test_environment(&mut command, guard)?;
        for (name, value) in case.environment {
            if is_loader_environment(name) {
                return Err(NativeBuildError::new(format!(
                    "native test {} case {} attempted to set loader environment {name}",
                    spec.name, case.name
                )));
            }
            command.env(name, value);
        }
        let output = run_bounded_child(
            &mut command,
            build,
            &format!("{}-{}", spec.name, case.name),
            Duration::from_secs(spec.timeout_seconds),
            8 * 1024 * 1024,
            guard,
        )?;
        if !output.status.success() {
            return Err(NativeBuildError::new(format!(
                "native test {} ({}) failed with {}\nstdout:\n{}\nstderr:\n{}",
                spec.name,
                case.name,
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let stdout = String::from_utf8(output.stdout).map_err(|error| {
            NativeBuildError::new(format!(
                "native test {} ({}) emitted non-UTF-8 stdout: {error}",
                spec.name, case.name
            ))
        })?;
        if !stdout.contains(spec.pass_marker) {
            return Err(NativeBuildError::new(format!(
                "native test {} ({}) omitted PASS marker {:?}: {stdout:?}",
                spec.name, case.name, spec.pass_marker
            )));
        }
        reports.push(NativeTestCaseReport {
            name: case.name,
            stdout,
        });
    }
    if reports.is_empty() {
        return Err(NativeBuildError::new(format!(
            "native test {} has no case for host OS {host_os}",
            spec.name
        )));
    }
    Ok(NativeTestReport {
        name: spec.name,
        executable: executable.to_path_buf(),
        cases: reports,
    })
}

#[derive(Debug)]
struct BoundedChildOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn sanitize_test_environment(command: &mut Command, guard: &PythonGuard) -> NativeBuildResult<()> {
    command.env_clear();
    let mut directories: Vec<PathBuf> = vec![guard.directory.clone()];
    if target_os() == "windows" {
        // The loader resolves implicit imports from the executable's own
        // directory, then the system's, then PATH: so PATH here is the system
        // directories (asked of the kernel, not read from the environment)
        // and then the one LibTorch directory this build audited.
        let system = deltafin_sys::fs::system_directory().map_err(|error| {
            NativeBuildError::new(format!("locate the Windows system directory: {error}"))
        })?;
        if let Some(windows) = system.parent() {
            directories.push(system.clone());
            directories.push(windows.to_path_buf());
        }
        directories.extend(guard.loader_directories.iter().cloned());
    } else {
        directories.extend(
            ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
                .iter()
                .map(PathBuf::from),
        );
    }
    let path = env::join_paths(directories)
        .map_err(|error| NativeBuildError::new(format!("construct native-test PATH: {error}")))?;
    command.env("PATH", path);
    for variable in [
        // Win32 programs (Winsock, the CRT's locale and time) read these.
        "SystemRoot",
        "windir",
        "SystemDrive",
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
    ] {
        if let Some(value) = env::var_os(variable) {
            command.env(variable, value);
        }
    }
    Ok(())
}

fn is_loader_environment(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    name.starts_with("LD_")
        || name.starts_with("DYLD_")
        || matches!(
            name.as_str(),
            "LIBPATH" | "SHLIB_PATH" | "_RLD_LIST" | "_RLD_ROOT"
        )
}

fn run_bounded_child(
    command: &mut Command,
    log_root: &Path,
    label: &str,
    timeout: Duration,
    output_limit: u64,
    guard: &PythonGuard,
) -> NativeBuildResult<BoundedChildOutput> {
    guard.assert_clean();
    let safe_label: String = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    let stdout_path = log_root.join(format!("{safe_label}.stdout"));
    let stderr_path = log_root.join(format!("{safe_label}.stderr"));
    for (path, kind) in [(&stdout_path, "stdout"), (&stderr_path, "stderr")] {
        prepare_generated_output(path, &format!("native-test {kind} log"));
    }
    let stdout_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stdout_path)
        .map_err(|error| {
            NativeBuildError::new(format!(
                "create bounded native-test stdout {}: {error}",
                stdout_path.display()
            ))
        })?;
    let stderr_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stderr_path)
        .map_err(|error| {
            NativeBuildError::new(format!(
                "create bounded native-test stderr {}: {error}",
                stderr_path.display()
            ))
        })?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    let rendered = format!("{command:?}");
    let mut child = command.spawn().map_err(|error| {
        NativeBuildError::new(format!("launch bounded native child {rendered}: {error}"))
    })?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|error| {
            NativeBuildError::new(format!("poll bounded native child {rendered}: {error}"))
        })? {
            break status;
        }
        let stdout_bytes = fs::metadata(&stdout_path)
            .map(|value| value.len())
            .unwrap_or(0);
        let stderr_bytes = fs::metadata(&stderr_path)
            .map(|value| value.len())
            .unwrap_or(0);
        if stdout_bytes > output_limit || stderr_bytes > output_limit {
            let _ = child.kill();
            let _ = child.wait();
            return Err(NativeBuildError::new(format!(
                "bounded native child exceeded {output_limit} bytes of output: {rendered}"
            )));
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(NativeBuildError::new(format!(
                "bounded native child exceeded {} seconds: {rendered}",
                timeout.as_secs()
            )));
        }
        thread::sleep(Duration::from_millis(20));
    };
    guard.assert_clean();
    let stdout = fs::read(&stdout_path).map_err(|error| {
        NativeBuildError::new(format!(
            "read bounded native-test stdout {}: {error}",
            stdout_path.display()
        ))
    })?;
    let stderr = fs::read(&stderr_path).map_err(|error| {
        NativeBuildError::new(format!(
            "read bounded native-test stderr {}: {error}",
            stderr_path.display()
        ))
    })?;
    if stdout.len() as u64 > output_limit || stderr.len() as u64 > output_limit {
        return Err(NativeBuildError::new(format!(
            "bounded native child crossed its output limit during exit: {rendered}"
        )));
    }
    Ok(BoundedChildOutput {
        status,
        stdout,
        stderr,
    })
}

fn audit_standalone_dependencies(
    executable: &Path,
    build: &Path,
    guard: &PythonGuard,
) -> NativeBuildResult<()> {
    audit_dependency_graph(executable, &[], build, guard)
}

fn audit_transitive_dependencies(
    executable: &Path,
    provider: &ProviderBuildArtifacts,
    build: &Path,
) -> NativeBuildResult<()> {
    let mut roots = vec![provider.torch_root.clone(), provider.torch_lib.clone()];
    if let Some(cuda) = &provider.cuda_provider {
        roots.push(cuda.runtime_directory.clone());
    }
    if let Some(cuda) = provider
        .cuda
        .as_ref()
        .filter(|cuda| cuda.runtime == KernelRuntime::Hip)
    {
        roots.push(cuda.runtime_directory.clone());
    }
    audit_dependency_graph(executable, &roots, build, &provider.guard)
}

fn audit_dependency_graph(
    executable: &Path,
    admitted_roots: &[PathBuf],
    log_root: &Path,
    guard: &PythonGuard,
) -> NativeBuildResult<()> {
    const DEPENDENCY_LIMIT: usize = 512;
    let executable = deltafin_sys::fs::canonicalize(executable).map_err(|error| {
        NativeBuildError::new(format!(
            "resolve native-test executable for dependency audit {}: {error}",
            executable.display()
        ))
    })?;
    let mut roots = Vec::new();
    for root in admitted_roots {
        if let Ok(root) = deltafin_sys::fs::canonicalize(root) {
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
    }
    if let Some(parent) = executable.parent() {
        roots.push(parent.to_path_buf());
    }
    if target_os() == "linux" {
        for directory in linux_loader_cache_directories(log_root, guard)? {
            if !roots.contains(&directory) {
                roots.push(directory);
            }
        }
    }
    let root_executable = executable.clone();
    let mut pending = VecDeque::from([executable]);
    let mut visited = BTreeSet::new();
    while let Some(binary) = pending.pop_front() {
        let binary = deltafin_sys::fs::canonicalize(&binary).map_err(|error| {
            NativeBuildError::new(format!(
                "resolve transitive native dependency {}: {error}",
                binary.display()
            ))
        })?;
        if !visited.insert(binary.clone()) {
            continue;
        }
        if visited.len() > DEPENDENCY_LIMIT {
            return Err(NativeBuildError::new(format!(
                "native dependency graph exceeded {DEPENDENCY_LIMIT} files"
            )));
        }
        reject_python_dependency(&binary)?;
        let discovered = match target_os().as_str() {
            "macos" => audit_macho_dependencies(
                &binary,
                &root_executable,
                &roots,
                log_root,
                visited.len(),
                guard,
            )?,
            "linux" => audit_elf_dependencies(&binary, &roots, log_root, visited.len(), guard)?,
            "windows" => audit_pe_dependencies(&binary, &roots)?,
            target => {
                return Err(NativeBuildError::new(format!(
                    "dependency audit does not support target OS {target}"
                )));
            }
        };
        pending.extend(discovered);
    }
    Ok(())
}

/// The DLLs a PE image needs, resolved as the loader will for a native test
/// that runs with the PATH this build gives it: the executable's own
/// directory, the Windows system directory, then the audited roots. Parsed in
/// process; nothing is executed and no tool (`dumpbin`, `ldd`) is consulted.
fn audit_pe_dependencies(binary: &Path, roots: &[PathBuf]) -> NativeBuildResult<Vec<PathBuf>> {
    let mut file = fs::File::open(binary).map_err(|error| {
        NativeBuildError::new(format!(
            "open PE image for dependency audit {}: {error}",
            binary.display()
        ))
    })?;
    let image = deltafin_sys::pe::parse_image(&mut file).map_err(|error| {
        NativeBuildError::new(format!("malformed PE image {}: {error}", binary.display()))
    })?;
    if image.headers.machine != deltafin_sys::pe::MACHINE_AMD64 || !image.headers.pe32_plus {
        return Err(NativeBuildError::new(format!(
            "native build produced a PE image for machine {:#06x}; only x86-64 is supported: {}",
            image.headers.machine,
            binary.display()
        )));
    }
    let system = deltafin_sys::fs::system_directory().map_err(|error| {
        NativeBuildError::new(format!("locate the Windows system directory: {error}"))
    })?;
    let own_directory = binary.parent().map(Path::to_path_buf);
    let mut found = Vec::new();
    for name in image.imports.all() {
        if name.contains(['/', '\\', ':']) {
            return Err(NativeBuildError::new(format!(
                "PE image {} imports {name:?}, a path rather than a DLL name",
                binary.display()
            )));
        }
        let lowercase = name.to_ascii_lowercase();
        // API-set contracts are virtual modules with no file to audit.
        if lowercase.starts_with("api-ms-win-") || lowercase.starts_with("ext-ms-win-") {
            continue;
        }
        if system.join(name).is_file()
            && !own_directory
                .as_ref()
                .is_some_and(|directory| directory.join(name).is_file())
        {
            continue;
        }
        let resolved = own_directory
            .iter()
            .chain(roots.iter())
            .map(|directory| directory.join(name))
            .find(|candidate| candidate.is_file());
        let Some(resolved) = resolved else {
            return Err(NativeBuildError::new(format!(
                "cannot prove the native dependency closure: {} imports {name:?}, found neither beside it, in the Windows system directory, nor in an audited root",
                binary.display()
            )));
        };
        let canonical = deltafin_sys::fs::canonicalize(&resolved).map_err(|error| {
            NativeBuildError::new(format!(
                "resolve native dependency {}: {error}",
                resolved.display()
            ))
        })?;
        if !roots.iter().chain(own_directory.iter()).any(|root| {
            deltafin_sys::fs::canonicalize(root).is_ok_and(|root| canonical.starts_with(root))
        }) {
            return Err(NativeBuildError::new(format!(
                "native dependency {} escapes every audited root",
                canonical.display()
            )));
        }
        found.push(canonical);
    }
    Ok(found)
}

fn reject_python_dependency(path: &Path) -> NativeBuildResult<()> {
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    if name.contains("python") || name.contains("torch_python") {
        return Err(NativeBuildError::new(format!(
            "native test transitively links a Python runtime: {}",
            path.display()
        )));
    }
    Ok(())
}

fn audit_macho_dependencies(
    binary: &Path,
    executable: &Path,
    roots: &[PathBuf],
    log_root: &Path,
    index: usize,
    guard: &PythonGuard,
) -> NativeBuildResult<Vec<PathBuf>> {
    let tool = PathBuf::from("/usr/bin/otool");
    validate_native_executable(&tool, "Mach-O dependency auditor");
    let install_name = macho_install_name(binary, log_root, index, guard, &tool)?;
    let mut command = Command::new(&tool);
    command.arg("-L").arg(binary);
    sanitize_test_environment(&mut command, guard)?;
    let output = run_bounded_child(
        &mut command,
        log_root,
        &format!("dependency-audit-{index}"),
        Duration::from_secs(30),
        4 * 1024 * 1024,
        guard,
    )?;
    if !output.status.success() {
        return Err(NativeBuildError::new(format!(
            "otool dependency audit failed for {}: {}",
            binary.display(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let text = std::str::from_utf8(&output.stdout).map_err(|error| {
        NativeBuildError::new(format!(
            "otool emitted non-UTF-8 dependency metadata for {}: {error}",
            binary.display()
        ))
    })?;
    let mut dependencies = Vec::new();
    for line in text.lines().skip(1) {
        let Some(name) = line.trim().split(" (").next() else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        // `otool -L` prints a dylib's LC_ID_DYLIB as its first entry even
        // though that install name is not a load dependency.  In particular,
        // the audited LibTorch bundle's libomp keeps its upstream
        // `/opt/llvm-openmp/...` identity while the executable loads the
        // bundle-local file.  Read LC_ID_DYLIB independently and omit only
        // that exact identity; every real LC_LOAD_* entry remains recursive.
        if install_name.as_deref() == Some(name) {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        if lower.contains("python") || lower.contains("torch_python") {
            return Err(NativeBuildError::new(format!(
                "{} names forbidden Python dependency {name}",
                binary.display()
            )));
        }
        if name.starts_with("/System/Library/") || name.starts_with("/usr/lib/") {
            continue;
        }
        let resolved = resolve_macho_dependency(name, binary, executable, roots)?;
        reject_python_dependency(&resolved)?;
        dependencies.push(resolved);
    }
    Ok(dependencies)
}

fn macho_install_name(
    binary: &Path,
    log_root: &Path,
    index: usize,
    guard: &PythonGuard,
    tool: &Path,
) -> NativeBuildResult<Option<String>> {
    let mut command = Command::new(tool);
    command.arg("-D").arg(binary);
    sanitize_test_environment(&mut command, guard)?;
    let output = run_bounded_child(
        &mut command,
        log_root,
        &format!("dependency-install-name-{index}"),
        Duration::from_secs(30),
        1024 * 1024,
        guard,
    )?;
    if !output.status.success() {
        return Err(NativeBuildError::new(format!(
            "otool install-name audit failed for {}: {}",
            binary.display(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let text = std::str::from_utf8(&output.stdout).map_err(|error| {
        NativeBuildError::new(format!(
            "otool emitted non-UTF-8 install-name metadata for {}: {error}",
            binary.display()
        ))
    })?;
    Ok(text
        .lines()
        .skip(1)
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned))
}

fn resolve_macho_dependency(
    name: &str,
    loader: &Path,
    executable: &Path,
    roots: &[PathBuf],
) -> NativeBuildResult<PathBuf> {
    let loader_directory = loader.parent().unwrap_or(Path::new("/"));
    let executable_directory = executable.parent().unwrap_or(Path::new("/"));
    let mut candidates = Vec::new();
    if let Some(relative) = name.strip_prefix("@loader_path/") {
        candidates.push(loader_directory.join(relative));
    } else if let Some(relative) = name.strip_prefix("@executable_path/") {
        candidates.push(executable_directory.join(relative));
    } else if let Some(relative) = name.strip_prefix("@rpath/") {
        candidates.push(loader_directory.join(relative));
        candidates.extend(roots.iter().map(|root| root.join(relative)));
        candidates.push(executable_directory.join(relative));
    } else if Path::new(name).is_absolute() {
        candidates.push(PathBuf::from(name));
    } else {
        return Err(NativeBuildError::new(format!(
            "Mach-O dependency uses an unsupported relative install name: {name}"
        )));
    }
    for candidate in candidates {
        if let Ok(candidate) = deltafin_sys::fs::canonicalize(&candidate) {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(NativeBuildError::new(format!(
        "could not resolve Mach-O dependency {name:?} from {}",
        loader.display()
    )))
}

fn audit_elf_dependencies(
    binary: &Path,
    roots: &[PathBuf],
    log_root: &Path,
    index: usize,
    guard: &PythonGuard,
) -> NativeBuildResult<Vec<PathBuf>> {
    let readelf = [
        Path::new("/usr/bin/readelf"),
        Path::new("/usr/bin/llvm-readelf"),
        Path::new("/bin/readelf"),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
    .ok_or_else(|| {
        NativeBuildError::new("Linux native tests require native readelf or llvm-readelf")
    })?;
    let readelf = resolve_tool_path(readelf, "ELF dependency auditor");
    let mut command = Command::new(&readelf);
    command.args(["-d"]).arg(binary);
    sanitize_test_environment(&mut command, guard)?;
    let output = run_bounded_child(
        &mut command,
        log_root,
        &format!("dependency-audit-{index}"),
        Duration::from_secs(30),
        4 * 1024 * 1024,
        guard,
    )?;
    if !output.status.success() {
        return Err(NativeBuildError::new(format!(
            "readelf dependency audit failed for {}: {}",
            binary.display(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let text = std::str::from_utf8(&output.stdout).map_err(|error| {
        NativeBuildError::new(format!(
            "readelf emitted non-UTF-8 dependency metadata for {}: {error}",
            binary.display()
        ))
    })?;
    let origin = binary.parent().unwrap_or(Path::new("/"));
    let mut search = vec![origin.to_path_buf()];
    search.extend_from_slice(roots);
    for value in elf_dynamic_values(text, "RPATH").chain(elf_dynamic_values(text, "RUNPATH")) {
        for component in value.split(':') {
            let expanded = component
                .replace("${ORIGIN}", &origin.display().to_string())
                .replace("$ORIGIN", &origin.display().to_string());
            let path = PathBuf::from(expanded);
            if path.is_absolute() && path.is_dir() {
                search.push(path);
            }
        }
    }
    search.extend(linux_system_library_directories());
    let mut dependencies = Vec::new();
    for name in elf_dynamic_values(text, "NEEDED") {
        let lower = name.to_ascii_lowercase();
        if lower.contains("python") || lower.contains("torch_python") {
            return Err(NativeBuildError::new(format!(
                "{} names forbidden Python dependency {name}",
                binary.display()
            )));
        }
        let resolved = search
            .iter()
            .map(|directory| directory.join(name))
            .find_map(|candidate| deltafin_sys::fs::canonicalize(candidate).ok())
            .filter(|candidate| candidate.is_file())
            .ok_or_else(|| {
                NativeBuildError::new(format!(
                    "could not resolve ELF dependency {name:?} from {}",
                    binary.display()
                ))
            })?;
        reject_python_dependency(&resolved)?;
        dependencies.push(resolved);
    }
    Ok(dependencies)
}

fn elf_dynamic_values<'a>(text: &'a str, tag: &'static str) -> impl Iterator<Item = &'a str> {
    text.lines().filter_map(move |line| {
        if !line.contains(&format!("({tag})")) {
            return None;
        }
        line.split_once('[')
            .and_then(|(_, tail)| tail.split_once(']'))
            .map(|(value, _)| value)
    })
}

fn linux_system_library_directories() -> Vec<PathBuf> {
    let multiarch = match target_arch().as_str() {
        "x86_64" => "x86_64-linux-gnu",
        "aarch64" => "aarch64-linux-gnu",
        _ => "",
    };
    let mut directories = vec![
        PathBuf::from("/lib"),
        PathBuf::from("/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
    ];
    if !multiarch.is_empty() {
        directories.push(PathBuf::from("/lib").join(multiarch));
        directories.push(PathBuf::from("/usr/lib").join(multiarch));
    }
    directories
        .into_iter()
        .filter(|directory| directory.is_dir())
        .collect()
}

fn linux_loader_cache_directories(
    log_root: &Path,
    guard: &PythonGuard,
) -> NativeBuildResult<Vec<PathBuf>> {
    // A candidate that is missing, or present but not a genuine native
    // executable (a wrapper script on some distributions, for example),
    // is treated identically: this enrichment is optional and already has
    // a real fallback, so there is nothing to validate-and-panic over here.
    let Some(tool) = [Path::new("/sbin/ldconfig"), Path::new("/usr/sbin/ldconfig")]
        .into_iter()
        .find(|candidate| candidate.is_file() && is_native_executable(candidate))
    else {
        return Ok(linux_system_library_directories());
    };
    let mut command = Command::new(tool);
    command.arg("-p");
    sanitize_test_environment(&mut command, guard)?;
    let output = run_bounded_child(
        &mut command,
        log_root,
        "loader-cache-audit",
        Duration::from_secs(30),
        4 * 1024 * 1024,
        guard,
    )?;
    if !output.status.success() {
        return Err(NativeBuildError::new(format!(
            "native ldconfig loader-cache audit failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let text = std::str::from_utf8(&output.stdout).map_err(|error| {
        NativeBuildError::new(format!(
            "native ldconfig emitted non-UTF-8 loader metadata: {error}"
        ))
    })?;
    let mut directories = BTreeSet::new();
    directories.extend(linux_system_library_directories());
    for line in text.lines() {
        let Some((_, path)) = line.rsplit_once(" => ") else {
            continue;
        };
        let path = Path::new(path.trim());
        if !path.is_absolute() {
            continue;
        }
        if let Some(parent) = path
            .parent()
            .and_then(|value| deltafin_sys::fs::canonicalize(value).ok())
        {
            directories.insert(parent);
        }
    }
    Ok(directories.into_iter().collect())
}

#[derive(Debug)]
struct NativeToolchain {
    flavor: CompilerFlavor,
    cc: PathBuf,
    cxx: PathBuf,
    /// `ar` on Unix, `lib.exe` on Windows.
    ar: PathBuf,
    /// `link.exe` (Windows only): links the isolated native tests, which on
    /// Unix go through the C++ driver.
    linker: Option<PathBuf>,
    /// What the tools need in their environment to run (`INCLUDE`, `LIB`, and
    /// a `PATH` that reaches their DLLs): empty for GCC and Clang.
    environment: Vec<(std::ffi::OsString, std::ffi::OsString)>,
}

impl NativeToolchain {
    fn discover() -> Self {
        let target_os = target_os();
        match target_os.as_str() {
            "macos" => Self::gnu(&["/usr/bin/clang"], &["/usr/bin/clang++"], &["/usr/bin/ar"]),
            "linux" => Self::gnu(&["cc", "clang", "gcc"], &["c++", "clang++", "g++"], &["ar"]),
            "windows" => Self::msvc(),
            target => panic!("native provider ABI does not support target OS {target}"),
        }
    }

    fn gnu(cc_defaults: &[&str], cxx_defaults: &[&str], ar_defaults: &[&str]) -> Self {
        Self {
            flavor: CompilerFlavor::Gnu,
            cc: resolve_required_native_tool("CC", cc_defaults),
            cxx: resolve_required_native_tool("CXX", cxx_defaults),
            ar: resolve_required_native_tool("AR", ar_defaults),
            linker: None,
            environment: Vec::new(),
        }
    }

    /// Microsoft's toolchain, found through Visual Studio's own setup
    /// configuration. An explicit `CC`/`CXX`/`AR` still wins, and its flavor
    /// follows its file name (`cl.exe` and `clang-cl.exe` speak MSVC options).
    fn msvc() -> Self {
        let target = format!("{}-pc-windows-msvc", target_arch());
        let discovered = msvc::discover(&target).unwrap_or_else(|error| panic!("{error}"));
        let explicit = |variable: &str, found: &Path| {
            if env::var_os(variable).is_some() {
                resolve_required_native_tool(variable, &[])
            } else {
                let path = resolve_tool_path(found, variable);
                validate_native_executable(&path, variable);
                path
            }
        };
        let cc = explicit("CC", &discovered.cl);
        let cxx = explicit("CXX", &discovered.cl);
        let flavor = msvc::flavor_of(&cc);
        if msvc::flavor_of(&cxx) != flavor {
            panic!(
                "CC ({}) and CXX ({}) must speak the same compiler dialect",
                cc.display(),
                cxx.display()
            );
        }
        if flavor == CompilerFlavor::Gnu {
            // A GNU-driver compiler on Windows (clang, gcc) is a different
            // graph from the one audited here; it is not silently half-supported.
            panic!(
                "the Windows native build admits MSVC-style compilers (cl.exe, clang-cl.exe); \
                 {} speaks GNU options",
                cc.display()
            );
        }
        Self {
            flavor,
            cc,
            cxx,
            ar: {
                let archiver = explicit("AR", &discovered.lib);
                let stem = archiver
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().to_ascii_lowercase())
                    .unwrap_or_default();
                if stem != "lib" && stem != "llvm-lib" {
                    panic!(
                        "the Windows native build archives with lib.exe or llvm-lib.exe, not {}",
                        archiver.display()
                    );
                }
                archiver
            },
            // `LINK` itself is link.exe's options variable, so the override
            // for the program is named for this project.
            linker: Some(explicit("DELTAFIN_LINK", &discovered.link)),
            environment: discovered.environment,
        }
    }
}

#[derive(Debug)]
struct PythonGuard {
    directory: PathBuf,
    marker: PathBuf,
    /// The selected toolchain's own environment, applied to every guarded
    /// command (`INCLUDE`, `LIB`, and the `PATH` that reaches its DLLs).
    tool_environment: Vec<(std::ffi::OsString, std::ffi::OsString)>,
    /// Directories a native test's loader may read DLLs from, in addition to
    /// the system's (Windows resolves implicit imports through `PATH`).
    loader_directories: Vec<PathBuf>,
}

impl PythonGuard {
    fn build(build: &Path, toolchain: &NativeToolchain, provider_source: &Path) -> Self {
        let cc = &toolchain.cc;
        let directory = build.join("python-deny");
        fs::create_dir_all(&directory).unwrap_or_else(|error| {
            panic!(
                "create native Python-denial directory {}: {error}",
                directory.display()
            )
        });
        let marker = directory.join("forbidden-python-attempted");
        if marker.exists() {
            fs::remove_file(&marker).unwrap_or_else(|error| {
                panic!(
                    "clear stale native Python-denial marker {}: {error}",
                    marker.display()
                )
            });
        }
        let template_path = provider_source.join("deny_python.c.in");
        let template = fs::read_to_string(&template_path).unwrap_or_else(|error| {
            panic!(
                "read native Python-denial source {}: {error}",
                template_path.display()
            )
        });
        if template.matches("@DELTAFIN_PYTHON_DENY_MARKER_C@").count() != 1 {
            panic!(
                "native Python-denial source has an ambiguous marker placeholder: {}",
                template_path.display()
            );
        }
        let source = template.replace(
            "@DELTAFIN_PYTHON_DENY_MARKER_C@",
            &c_string_literal_bytes(marker.as_os_str().as_encoded_bytes()),
        );
        let source_path = directory.join("deny_python.c");
        fs::write(&source_path, source).unwrap_or_else(|error| {
            panic!(
                "write native Python-denial source {}: {error}",
                source_path.display()
            )
        });
        let executable =
            directory.join(format!("deltafin-python-denied{}", env::consts::EXE_SUFFIX));
        let mut compile = Command::new(cc);
        match toolchain.flavor {
            CompilerFlavor::Gnu => {
                compile
                    .args(["-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror"])
                    .arg(&source_path)
                    .arg("-o")
                    .arg(&executable);
            }
            CompilerFlavor::Msvc => {
                compile.args(msvc::guard_compile_args(
                    &source_path,
                    &directory.join("deny_python.obj"),
                    &executable,
                ));
            }
        }
        sanitize_native_environment(&mut compile);
        apply_tool_environment(&mut compile, &toolchain.environment);
        run_checked(&mut compile, "compile native Python-denial guard");
        validate_native_executable(&executable, "Python-denial guard");

        let output = Command::new(&executable)
            .arg("--self-test")
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "execute native Python-denial self-test {}: {error}",
                    executable.display()
                )
            });
        if output.status.code() != Some(PYTHON_DENY_EXIT) || !marker.is_file() {
            panic!(
                "native Python-denial guard failed its self-test: {}",
                executable.display()
            );
        }
        let marker_text = fs::read_to_string(&marker).unwrap_or_else(|error| {
            panic!(
                "read native Python-denial self-test marker {}: {error}",
                marker.display()
            )
        });
        if marker_text != "forbidden-python-build-invocation\n" {
            panic!(
                "native Python-denial guard emitted malformed self-test marker: {}",
                marker.display()
            );
        }
        fs::remove_file(&marker).unwrap_or_else(|error| {
            panic!(
                "clear native Python-denial self-test marker {}: {error}",
                marker.display()
            )
        });
        for alias in ["python", "python2", "python3", "py"] {
            let alias = directory.join(format!("{alias}{}", env::consts::EXE_SUFFIX));
            if alias.exists() {
                let metadata = fs::symlink_metadata(&alias).unwrap_or_else(|error| {
                    panic!("inspect Python-denial alias {}: {error}", alias.display())
                });
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    panic!(
                        "Python-denial alias must be a regular file: {}",
                        alias.display()
                    );
                }
                fs::remove_file(&alias).unwrap_or_else(|error| {
                    panic!("replace Python-denial alias {}: {error}", alias.display())
                });
            }
            fs::copy(&executable, &alias).unwrap_or_else(|error| {
                panic!(
                    "install native Python-denial alias {}: {error}",
                    alias.display()
                )
            });
        }
        Self {
            directory,
            marker,
            tool_environment: toolchain.environment.clone(),
            loader_directories: Vec::new(),
        }
    }

    fn prepare(&self, command: &mut Command) {
        if self.marker.exists() {
            panic!(
                "a previous build dependency attempted to execute Python; marker: {}",
                self.marker.display()
            );
        }
        // The toolchain's environment first, so the PATH built below extends
        // the one that reaches its DLLs rather than the ambient one.
        apply_tool_environment(command, &self.tool_environment);
        let inherited = self
            .tool_environment
            .iter()
            .find(|(name, _)| msvc::same_variable(name, OsStr::new("PATH")))
            .map(|(_, value)| value.clone())
            .or_else(|| env::var_os("PATH"));
        let mut paths = vec![self.directory.clone()];
        if let Some(path) = inherited {
            paths.extend(env::split_paths(&path));
        }
        command.env(
            "PATH",
            env::join_paths(paths)
                .unwrap_or_else(|error| panic!("construct Python-denial PATH: {error}")),
        );
        sanitize_native_environment(command);
    }

    fn assert_clean(&self) {
        if self.marker.exists() {
            panic!(
                "a native build dependency attempted to execute Python; marker: {}",
                self.marker.display()
            );
        }
    }
}

/// Which vendor toolchain compiled the device kernel objects. A ROCm LibTorch
/// still reports `at::kCUDA` devices, so this is the only place the build
/// distinguishes them after detection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelRuntime {
    Cuda,
    Hip,
}

impl KernelRuntime {
    const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Hip => "HIP",
        }
    }
}

#[derive(Debug)]
struct CudaBuild {
    runtime: KernelRuntime,
    compiler: PathBuf,
    toolkit_version: String,
    architectures: String,
    runtime_directory: PathBuf,
    object: PathBuf,
    spine_object: PathBuf,
}

#[derive(Debug)]
struct CudaProviderBuild {
    include_directory: PathBuf,
    runtime_directory: PathBuf,
}

fn validate_torch_version(torch_root: &Path, emit_cargo_metadata: bool) {
    let version_path = torch_root.join(TORCH_VERSION_HEADER);
    let version = fs::read_to_string(&version_path).unwrap_or_else(|error| {
        panic!(
            "read selected LibTorch version header {}: {error}",
            version_path.display()
        )
    });
    for (name, expected) in [
        ("TORCH_VERSION_MAJOR", "2"),
        ("TORCH_VERSION_MINOR", "13"),
        ("TORCH_VERSION_PATCH", "0"),
        ("TORCH_VERSION_ABI_TAG", "0"),
    ] {
        let found = version.lines().find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some("#define") && fields.next() == Some(name))
                .then(|| fields.next())
                .flatten()
        });
        if found != Some(expected) {
            panic!(
                "selected native PyTorch root {} is not exact LibTorch 2.13.0 ABI tag 0: {name}={found:?}",
                torch_root.display()
            );
        }
    }
    if emit_cargo_metadata {
        println!("cargo:rerun-if-changed={}", version_path.display());
    }
}

#[derive(Debug)]
struct MetalToolchain {
    metal: PathBuf,
    metallib: PathBuf,
    identity: String,
}

fn build_embedded_metal_libraries(
    repository: &Path,
    provider_source: &Path,
    native_build: &Path,
    generated_include: &Path,
    guard: &PythonGuard,
    emit_cargo_metadata: bool,
) {
    let toolchain = discover_metal_toolchain(guard);
    let module_cache = native_build.join("metal-module-cache");
    fs::create_dir_all(&module_cache).unwrap_or_else(|error| {
        panic!(
            "create private Metal module cache {}: {error}",
            module_cache.display()
        )
    });
    fs::create_dir_all(generated_include).unwrap_or_else(|error| {
        panic!(
            "create generated native include directory {}: {error}",
            generated_include.display()
        )
    });

    for (label, source, stem, header, guard_name, symbol) in [
        (
            "MXFP4 MoE",
            repository.join("tools/metal/moe_mxfp4.metal"),
            "moe_mxfp4",
            "deltafin_embedded_moe_mxfp4_metallib.h",
            "DELTAFIN_EMBEDDED_MOE_MXFP4_METALLIB_H",
            "kDeltafinEmbeddedMoeMxfp4Metallib",
        ),
        (
            "route mailbox",
            provider_source.join("provider_route_mailbox.metal"),
            "provider_route_mailbox",
            "deltafin_embedded_route_mailbox_metallib.h",
            "DELTAFIN_EMBEDDED_ROUTE_MAILBOX_METALLIB_H",
            "kDeltafinEmbeddedRouteMailboxMetallib",
        ),
        (
            "BF16 spine GEMV",
            provider_source.join("provider_spine_bf16_metal.metal"),
            "provider_spine_bf16_metal",
            "deltafin_embedded_spine_bf16_metal_metallib.h",
            "DELTAFIN_EMBEDDED_SPINE_BF16_METAL_METALLIB_H",
            "kDeltafinEmbeddedSpineBf16MetalMetallib",
        ),
        (
            "int8 spine dequant",
            provider_source.join("provider_spine_int8_metal.metal"),
            "provider_spine_int8_metal",
            "deltafin_embedded_spine_int8_metal_metallib.h",
            "DELTAFIN_EMBEDDED_SPINE_INT8_METAL_METALLIB_H",
            "kDeltafinEmbeddedSpineInt8MetalMetallib",
        ),
    ] {
        let air = native_build.join(format!("{stem}.air"));
        let library = native_build.join(format!("{stem}.metallib"));
        prepare_generated_output(&air, "Metal AIR object");
        prepare_generated_output(&library, "Metal library");

        let mut compile = Command::new(&toolchain.metal);
        compile
            .current_dir(repository)
            .arg("-c")
            .arg("-O3")
            .arg("-mmacosx-version-min=14.0")
            .arg(format!("-fmodules-cache-path={}", module_cache.display()));
        // Original-BF16 target projections promise fp32 inputs, explicit
        // fp32 FMA/reduction, and fp32 outputs. Apple's offline Metal compiler
        // otherwise enables unsafe fast-math under -O3 (including finite-only
        // assumptions and reassociation), which is not a valid implementation
        // of that contract. Keep this source-specific: the established expert
        // and mailbox libraries retain their independently qualified flags.
        if stem == "provider_spine_bf16_metal" || stem == "provider_spine_int8_metal" {
            compile.arg("-fno-fast-math");
        }
        compile
            .arg(source.strip_prefix(repository).unwrap_or(&source))
            .arg("-o")
            .arg(&air);
        run_guarded_checked(
            &mut compile,
            &format!("compile reviewed {label} Metal source"),
            guard,
        );
        validate_object(&air);

        let mut link = Command::new(&toolchain.metallib);
        link.arg(&air).arg("-o").arg(&library);
        run_guarded_checked(
            &mut link,
            &format!("link reviewed {label} Metal library"),
            guard,
        );
        validate_metallib(&library, label);
        write_embedded_binary_header(
            &library,
            &generated_include.join(header),
            guard_name,
            symbol,
            &format!("{label} metallib built by {}", toolchain.identity),
        );
    }
    if emit_cargo_metadata {
        println!(
            "cargo:rustc-env=DELTAFIN_METAL_TOOLCHAIN={}",
            toolchain.identity
        );
    }
}

fn discover_metal_toolchain(guard: &PythonGuard) -> MetalToolchain {
    let xcode_select = PathBuf::from("/usr/bin/xcode-select");
    validate_apple_native_tool(&xcode_select, "xcode-select");
    let mut select = Command::new(&xcode_select);
    select.arg("-p");
    let selected = run_guarded_raw_output(&mut select, "query selected Xcode", guard);
    if selected.status.success() {
        let developer =
            parse_bounded_absolute_path(&selected.stdout, "xcode-select developer directory");
        if let Some(toolchain) = metal_toolchain_under(
            &developer.join("Toolchains/XcodeDefault.xctoolchain"),
            "selected Xcode",
            guard,
        ) {
            return toolchain;
        }
    }

    let xcodebuild = PathBuf::from("/usr/bin/xcodebuild");
    validate_apple_native_tool(&xcodebuild, "xcodebuild");
    let mut query = Command::new(&xcodebuild);
    query.args(["-showComponent", "MetalToolchain", "-json"]);
    let output = run_guarded_raw_output(
        &mut query,
        "query installed Apple MetalToolchain component",
        guard,
    );
    const METADATA_LIMIT: usize = 64 * 1024;
    if output.stdout.len() > METADATA_LIMIT || output.stderr.len() > METADATA_LIMIT {
        panic!("Apple MetalToolchain metadata exceeded {METADATA_LIMIT} bytes");
    }
    if !output.status.success() {
        panic!(
            "Deltafin's macOS source build requires full Xcode and its Metal toolchain component; run `xcodebuild -downloadComponent MetalToolchain` and retry. xcodebuild reported: {}",
            one_line(&output.stderr)
        );
    }
    let metadata: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!("parse bounded Apple MetalToolchain metadata JSON: {error}")
    });
    let object = metadata
        .as_object()
        .unwrap_or_else(|| panic!("Apple MetalToolchain metadata must be one JSON object"));
    if object.get("status").and_then(Value::as_str) != Some("installed") {
        panic!("Apple MetalToolchain component is not installed");
    }
    let identifier = object
        .get("toolchainIdentifier")
        .and_then(Value::as_str)
        .filter(|value| {
            value.starts_with("com.apple.dt.toolchain.Metal.")
                && value.len() <= 128
                && value.bytes().all(|byte| byte.is_ascii_graphic())
        })
        .unwrap_or_else(|| panic!("Apple MetalToolchain metadata has an invalid identifier"));
    let build_version = object
        .get("buildVersion")
        .and_then(Value::as_str)
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 64
                && value.bytes().all(|byte| byte.is_ascii_alphanumeric())
        })
        .unwrap_or_else(|| panic!("Apple MetalToolchain metadata has an invalid build version"));
    let search_path = object
        .get("toolchainSearchPath")
        .and_then(Value::as_str)
        .filter(|value| value.len() <= 4096)
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(|| panic!("Apple MetalToolchain metadata has an invalid search path"));
    let search_path = deltafin_sys::fs::canonicalize(&search_path).unwrap_or_else(|error| {
        panic!(
            "resolve Apple MetalToolchain search path {}: {error}",
            search_path.display()
        )
    });
    validate_root_owned_nonwritable(&search_path, "Apple MetalToolchain search root", true);
    let mut toolchain = metal_toolchain_under(
        &search_path.join("Metal.xctoolchain"),
        "downloaded Apple MetalToolchain component",
        guard,
    )
    .unwrap_or_else(|| {
        panic!(
            "installed Apple MetalToolchain has no usable native metal/metallib pair under {}",
            search_path.display()
        )
    });
    toolchain.identity = format!("{identifier}/{build_version}; {}", toolchain.identity);
    toolchain
}

fn metal_toolchain_under(root: &Path, label: &str, guard: &PythonGuard) -> Option<MetalToolchain> {
    let root = deltafin_sys::fs::canonicalize(root).ok()?;
    validate_root_owned_nonwritable(&root, &format!("{label} toolchain"), true);
    let bin = root.join("usr/bin");
    let metal = bin.join("metal");
    let metallib = bin.join("metallib");
    if !metal.exists() || !metallib.exists() {
        return None;
    }
    admit_apple_tool_preserving_name(&metal, &root, "Metal compiler");
    admit_apple_tool_preserving_name(&metallib, &root, "metallib linker");

    let mut metal_version = Command::new(&metal);
    metal_version.arg("--version");
    let metal_output = run_guarded_raw_output(&mut metal_version, "query Metal compiler", guard);
    let mut metallib_version = Command::new(&metallib);
    metallib_version.arg("--version");
    let metallib_output =
        run_guarded_raw_output(&mut metallib_version, "query metallib linker", guard);
    if !metal_output.status.success() || !metallib_output.status.success() {
        return None;
    }
    let metal_identity = bounded_tool_identity(&metal_output, "Metal compiler");
    let metallib_identity = bounded_tool_identity(&metallib_output, "metallib linker");
    if !metal_identity.contains("Apple metal version") || !metallib_identity.contains("AIR-LLD") {
        return None;
    }
    Some(MetalToolchain {
        metal,
        metallib,
        identity: format!(
            "{} / {}",
            one_line(metal_identity.as_bytes()),
            one_line(metallib_identity.as_bytes())
        ),
    })
}

fn bounded_tool_identity(output: &Output, label: &str) -> String {
    const IDENTITY_LIMIT: usize = 16 * 1024;
    if output.stdout.len() > IDENTITY_LIMIT || output.stderr.len() > IDENTITY_LIMIT {
        panic!("{label} identity exceeded {IDENTITY_LIMIT} bytes");
    }
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn parse_bounded_absolute_path(bytes: &[u8], label: &str) -> PathBuf {
    if bytes.is_empty() || bytes.len() > 4096 || bytes.contains(&0) {
        panic!("{label} is empty, oversized, or contains NUL");
    }
    let text = std::str::from_utf8(bytes)
        .unwrap_or_else(|_| panic!("{label} is not UTF-8"))
        .trim_end_matches(['\r', '\n']);
    if text.is_empty() || text.contains(['\r', '\n']) {
        panic!("{label} is not exactly one path");
    }
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        panic!("{label} is not absolute: {}", path.display());
    }
    deltafin_sys::fs::canonicalize(&path)
        .unwrap_or_else(|error| panic!("resolve {label} {}: {error}", path.display()))
}

fn validate_apple_native_tool(path: &Path, label: &str) {
    validate_root_owned_nonwritable(path, label, false);
    validate_native_executable(path, label);
}

#[cfg(not(unix))]
fn admit_apple_tool_preserving_name(path: &Path, _root: &Path, label: &str) {
    panic!("{label} admission is macOS-only: {}", path.display());
}

#[cfg(unix)]
fn admit_apple_tool_preserving_name(path: &Path, root: &Path, label: &str) {
    let named = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("inspect {label} {}: {error}", path.display()));
    if named.uid() != 0 {
        panic!("{label} path is not owned by root: {}", path.display());
    }
    if !named.file_type().is_symlink() && named.mode() & 0o022 != 0 {
        panic!("{label} path is group/world writable: {}", path.display());
    }
    let target = deltafin_sys::fs::canonicalize(path)
        .unwrap_or_else(|error| panic!("resolve {label} {}: {error}", path.display()));
    if !target.starts_with(root) {
        panic!(
            "{label} resolves outside its admitted Apple toolchain: {} -> {}",
            path.display(),
            target.display()
        );
    }
    validate_root_owned_nonwritable(&target, label, false);
    validate_native_executable(&target, label);
}

#[cfg(not(unix))]
fn validate_root_owned_nonwritable(path: &Path, label: &str, _directory: bool) {
    panic!("{label} admission is macOS-only: {}", path.display());
}

#[cfg(unix)]
fn validate_root_owned_nonwritable(path: &Path, label: &str, directory: bool) {
    let metadata = fs::metadata(path)
        .unwrap_or_else(|error| panic!("inspect {label} {}: {error}", path.display()));
    if metadata.uid() != 0 || metadata.mode() & 0o022 != 0 {
        panic!(
            "{label} must be root-owned and not group/world writable: {}",
            path.display()
        );
    }
    if (directory && !metadata.is_dir())
        || (!directory && (!metadata.is_file() || metadata.permissions().mode() & 0o111 == 0))
    {
        panic!(
            "{label} has the wrong file type or mode: {}",
            path.display()
        );
    }
}

fn prepare_generated_output(path: &Path, label: &str) {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            panic!("refusing unsafe prior {label}: {}", path.display());
        }
        fs::remove_file(path)
            .unwrap_or_else(|error| panic!("replace prior {label} {}: {error}", path.display()));
    }
}

fn validate_metallib(path: &Path, label: &str) {
    validate_object(path);
    let bytes = fs::read(path)
        .unwrap_or_else(|error| panic!("read linked {label} metallib {}: {error}", path.display()));
    if bytes.len() < 64 || !bytes.starts_with(b"MTLB") {
        panic!(
            "linked {label} artifact is not a bounded Metal library: {}",
            path.display()
        );
    }
}

fn write_embedded_binary_header(
    input: &Path,
    output: &Path,
    header_guard: &str,
    symbol: &str,
    provenance: &str,
) {
    let bytes = fs::read(input).unwrap_or_else(|error| {
        panic!("read generated Metal library {}: {error}", input.display())
    });
    if bytes.is_empty() || bytes.len() > 4 * 1024 * 1024 {
        panic!(
            "generated Metal library has an unsafe embedded size: {} bytes at {}",
            bytes.len(),
            input.display()
        );
    }
    let digest = Sha256::digest(&bytes);
    let mut header = format!(
        "// Generated {provenance}. Do not edit.\n#ifndef {header_guard}\n#define {header_guard}\n\n#include <cstddef>\n\nalignas(16) inline constexpr unsigned char {symbol}[] = {{\n"
    );
    for chunk in bytes.chunks(12) {
        header.push_str("    ");
        for byte in chunk {
            header.push_str(&format!("0x{byte:02x}, "));
        }
        header.push('\n');
    }
    header.push_str(&format!(
        "}};\ninline constexpr std::size_t {symbol}Bytes = sizeof({symbol});\ninline constexpr char {symbol}Sha256[] = \"{digest:x}\";\n\n#endif\n"
    ));
    if let Ok(metadata) = fs::symlink_metadata(output) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            panic!(
                "refusing unsafe generated Metal header: {}",
                output.display()
            );
        }
    }
    fs::write(output, header).unwrap_or_else(|error| {
        panic!(
            "write embedded Metal library header {}: {error}",
            output.display()
        )
    });
}

fn write_embedded_metal_source_header(input: &Path, output: &Path) -> NativeBuildResult<()> {
    const LIMIT: usize = 4 * 1024 * 1024;
    const DELIMITER: &str = "DF_MXFP4_MSL_V1";
    let source = fs::read(input).map_err(|error| {
        NativeBuildError::new(format!(
            "read reviewed Metal development source {}: {error}",
            input.display()
        ))
    })?;
    if source.is_empty() || source.len() > LIMIT {
        return Err(NativeBuildError::new(format!(
            "reviewed Metal development source has unsafe size {} at {}",
            source.len(),
            input.display()
        )));
    }
    let source_text = std::str::from_utf8(&source).map_err(|error| {
        NativeBuildError::new(format!(
            "reviewed Metal development source is not UTF-8 at {}: {error}",
            input.display()
        ))
    })?;
    let closing = format!("){DELIMITER}\"");
    if source_text.contains(&closing) {
        return Err(NativeBuildError::new(
            "reviewed Metal source collides with its bounded raw-string delimiter",
        ));
    }
    let digest = Sha256::digest(&source);
    let parent = output.parent().ok_or_else(|| {
        NativeBuildError::new(format!(
            "generated Metal development header has no parent: {}",
            output.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        NativeBuildError::new(format!(
            "create generated Metal development include {}: {error}",
            parent.display()
        ))
    })?;
    prepare_generated_output(output, "generated Metal development header");
    let header = format!(
        "// Generated from tools/metal/moe_mxfp4.metal for the isolated source-development gate. Do not edit.\n\
#ifndef DELTAFIN_EMBEDDED_MOE_MXFP4_MSL_H\n\
#define DELTAFIN_EMBEDDED_MOE_MXFP4_MSL_H\n\n\
#include <cstddef>\n\n\
inline constexpr char kDeltafinEmbeddedMoeMxfp4Msl[] = R\"{DELIMITER}({source_text}){DELIMITER}\";\n\
inline constexpr std::size_t kDeltafinEmbeddedMoeMxfp4MslBytes = sizeof(kDeltafinEmbeddedMoeMxfp4Msl) - 1;\n\
inline constexpr std::size_t kDeltafinEmbeddedMoeMxfp4MslInputBytes = {};\n\
inline constexpr char kDeltafinEmbeddedMoeMxfp4MslSha256[] = \"{digest:x}\";\n\
static_assert(kDeltafinEmbeddedMoeMxfp4MslBytes == kDeltafinEmbeddedMoeMxfp4MslInputBytes);\n\n\
#endif\n",
        source.len()
    );
    fs::write(output, header).map_err(|error| {
        NativeBuildError::new(format!(
            "write generated Metal development header {}: {error}",
            output.display()
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn compile_cpp(
    toolchain: &NativeToolchain,
    source: &Path,
    object: &Path,
    torch_root: &Path,
    generated_include: Option<&Path>,
    cuda_include: Option<&Path>,
    definitions: &[&str],
    objective_cpp: bool,
    guard: &PythonGuard,
) -> Result<(), String> {
    let mut command = Command::new(&toolchain.cxx);
    if toolchain.flavor == CompilerFlavor::Msvc {
        assert!(
            !objective_cpp,
            "Objective-C++ source {} cannot be compiled on Windows",
            source.display()
        );
        command.args(msvc::cpp_compile_args(
            source,
            object,
            torch_root,
            generated_include,
            cuda_include,
            definitions,
        ));
        run_guarded_try(
            &mut command,
            &format!("compile native provider source {}", source.display()),
            guard,
        )?;
        validate_object(object);
        return Ok(());
    }
    command.args([
        "-O3",
        "-DNDEBUG",
        "-std=gnu++20",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
    ]);
    if objective_cpp {
        command.arg("-fobjc-arc");
    }
    for definition in definitions {
        command.arg(format!("-D{definition}"));
    }
    if let Some(include) = generated_include {
        command.arg("-I").arg(include);
    }
    if let Some(include) = cuda_include {
        command.arg("-isystem").arg(include);
    }
    command
        .arg("-isystem")
        .arg(torch_root.join("include"))
        .arg("-isystem")
        .arg(torch_root.join("include/torch/csrc/api/include"))
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object);
    run_guarded_try(
        &mut command,
        &format!("compile native provider source {}", source.display()),
        guard,
    )?;
    validate_object(object);
    Ok(())
}

fn compile_gemv(
    toolchain: &NativeToolchain,
    source: &Path,
    object: &Path,
    guard: &PythonGuard,
) -> Result<(), String> {
    let mut command = Command::new(&toolchain.cc);
    if toolchain.flavor == CompilerFlavor::Msvc {
        command.args(msvc::c_kernel_compile_args(
            source,
            object,
            target_arch() == "x86_64",
        ));
        run_guarded_try(
            &mut command,
            &format!("compile exact MXFP4 CPU kernel {}", source.display()),
            guard,
        )?;
        validate_object(object);
        return Ok(());
    }
    command.args([
        "-O3",
        "-DNDEBUG",
        "-std=gnu11",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-Wno-unused-function",
    ]);
    if target_arch() == "x86_64" {
        command.args([
            "-march=x86-64",
            "-mtune=generic",
            "-mssse3",
            "-mavx",
            "-mfma",
        ]);
    }
    command.arg("-c").arg(source).arg("-o").arg(object);
    run_guarded_try(
        &mut command,
        &format!("compile exact MXFP4 CPU kernel {}", source.display()),
        guard,
    )?;
    validate_object(object);
    Ok(())
}

fn compile_c_test_main(
    toolchain: &NativeToolchain,
    source: &Path,
    object: &Path,
    guard: &PythonGuard,
) -> Result<(), String> {
    let mut command = Command::new(&toolchain.cc);
    if toolchain.flavor == CompilerFlavor::Msvc {
        command.args(msvc::c_test_compile_args(source, object));
        run_guarded_try(
            &mut command,
            &format!("compile isolated native test main {}", source.display()),
            guard,
        )?;
        validate_object(object);
        return Ok(());
    }
    command.args([
        "-O3",
        "-DNDEBUG",
        "-std=gnu11",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
    ]);
    command.arg("-c").arg(source).arg("-o").arg(object);
    run_guarded_try(
        &mut command,
        &format!("compile isolated native test main {}", source.display()),
        guard,
    )?;
    validate_object(object);
    Ok(())
}

fn archive_objects(
    toolchain: &NativeToolchain,
    archive: &Path,
    objects: &[PathBuf],
    guard: &PythonGuard,
) {
    if objects.is_empty() {
        panic!("native provider archive has no objects");
    }
    if let Ok(metadata) = fs::symlink_metadata(archive) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            panic!(
                "refusing unsafe previous native provider archive: {}",
                archive.display()
            );
        }
        fs::remove_file(archive).unwrap_or_else(|error| {
            panic!(
                "replace native provider archive {}: {error}",
                archive.display()
            )
        });
    }
    for object in objects {
        validate_object(object);
    }
    let mut command = Command::new(&toolchain.ar);
    match toolchain.flavor {
        CompilerFlavor::Gnu => {
            command.arg("rcs").arg(archive).args(objects);
        }
        CompilerFlavor::Msvc => {
            command.args(msvc::archive_args(archive, objects));
        }
    }
    run_guarded_checked(&mut command, "archive native provider ABI", guard);
    validate_object(archive);
}

fn validate_object(path: &Path) {
    let metadata = fs::symlink_metadata(path).unwrap_or_else(|error| {
        panic!("inspect native build artifact {}: {error}", path.display())
    });
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        panic!(
            "native build artifact must be a nonempty regular file: {}",
            path.display()
        );
    }
}

fn find_cuda_provider(torch_root: &Path) -> CudaProviderBuild {
    let target_arch = target_arch();
    let target_dir = match target_arch.as_str() {
        "x86_64" => "x86_64-linux",
        "aarch64" => "aarch64-linux",
        arch => panic!("CUDA provider does not support target architecture {arch}"),
    };
    let mut roots = vec![
        torch_root.to_path_buf(),
        torch_root
            .parent()
            .unwrap_or(torch_root)
            .join("nvidia/cuda_runtime"),
        PathBuf::from("/usr/local/cuda"),
        PathBuf::from("/usr"),
    ];
    for variable in ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(value) = env::var_os(variable) {
            let root = deltafin_sys::fs::canonicalize(&value).unwrap_or_else(|error| {
                panic!(
                    "resolve explicit CUDA toolkit {variable}={}: {error}",
                    PathBuf::from(&value).display()
                )
            });
            roots.push(root);
        }
    }
    if let Some(compiler) = discover_nvcc() {
        let root = compiler
            .parent()
            .and_then(Path::parent)
            .unwrap_or_else(|| panic!("NVCC path has no toolkit root: {}", compiler.display()));
        roots.push(root.to_path_buf());
    }
    for root in roots {
        for include in [
            root.join("include"),
            root.join("targets").join(target_dir).join("include"),
        ] {
            if !include.join("cuda_runtime_api.h").is_file() {
                continue;
            }
            if let Some(runtime_directory) = cuda_runtime_directory_optional(&root) {
                let include_directory =
                    deltafin_sys::fs::canonicalize(&include).unwrap_or_else(|error| {
                        panic!(
                            "resolve CUDA provider include directory {}: {error}",
                            include.display()
                        )
                    });
                return CudaProviderBuild {
                    include_directory,
                    runtime_directory,
                };
            }
        }
    }
    panic!(
        "CUDA-enabled LibTorch {} requires matching CUDA runtime headers and libcudart for native allocator accounting; set CUDAToolkit_ROOT, CUDA_HOME, or CUDA_PATH to its matching toolkit",
        torch_root.display()
    );
}

/// Toolchain selection for the device kernel translation units, resolved
/// before either source is compiled so the CUDA and HIP paths cannot drift
/// into two independent copies of the compile loop.
struct KernelBuildPlan {
    runtime: KernelRuntime,
    compiler: PathBuf,
    toolkit_version: String,
    architectures: String,
    runtime_directory: PathBuf,
}

fn build_cuda_kernel(
    repository: &Path,
    native_build: &Path,
    toolchain: &NativeToolchain,
    guard: &PythonGuard,
    gpu_runtime: Option<GpuRuntime>,
    emit_cargo_metadata: bool,
) -> Option<CudaBuild> {
    let mode = cuda_mode();
    let host_os = target_os();
    if mode == CudaMode::Off {
        return None;
    }
    if host_os != "linux" && host_os != "windows" {
        if mode == CudaMode::On {
            panic!("DELTAFIN_CUDA_MOE=ON is supported only for Linux and Windows targets");
        }
        return None;
    }
    let Some(runtime) = gpu_runtime else {
        if mode == CudaMode::On {
            panic!("DELTAFIN_CUDA_MOE=ON requires a GPU-enabled LibTorch root");
        }
        return None;
    };
    // The HIP port is Linux-only; a ROCm LibTorch on another host falls back to
    // the CPU MXFP4 path.
    if matches!(runtime, GpuRuntime::Rocm) && host_os != "linux" {
        if mode == CudaMode::On {
            panic!(
                "DELTAFIN_CUDA_MOE=ON selected a ROCm/HIP LibTorch on {host_os}; the HIP kernel port is Linux-only"
            );
        }
        return None;
    }
    let plan = match runtime {
        GpuRuntime::Cuda(major) => plan_cuda_kernel_build(major, mode, guard, emit_cargo_metadata)?,
        GpuRuntime::Rocm => plan_hip_kernel_build(mode, guard, emit_cargo_metadata)?,
    };
    let object = native_build.join(object_file_name("cuda_moe_kernels"));
    let spine_object = native_build.join(object_file_name("provider_spine_bf16_cuda_kernel"));
    let provider_source = repository.join("native/provider_gate");
    // Both translation units include tools/gpu_runtime_compat.h, which selects
    // the CUDA or HIP runtime headers. Only the MoE kernel finds it beside
    // itself; the spine kernel needs the search path. A separate `-I` argument
    // keeps a Windows path with spaces intact.
    let compat_include = repository.join("tools");
    for (source, output, label) in [
        (
            repository.join("tools/cuda_moe_kernels.cu"),
            &object,
            "exact device MXFP4 kernels",
        ),
        (
            provider_source.join("provider_spine_bf16_cuda.cu"),
            &spine_object,
            "exact device RAW_BF16 spine kernel",
        ),
    ] {
        let mut command = Command::new(&plan.compiler);
        match plan.runtime {
            KernelRuntime::Cuda => {
                command.args(["-std=c++17", "-O3", "-DNDEBUG"]);
                // The host compiler is `cl.exe` on Windows and the GNU C++
                // driver elsewhere. `cl.exe` has no `-fPIC`, needs LibTorch's
                // dynamic CRT (mixing CRTs across the DLL boundary corrupts the
                // heap) and the same exception/preprocessor contract as the
                // provider translation units, and `-ccbin` names its directory.
                let ccbin = if target_os() == "windows" {
                    command.args([
                        "-Xcompiler=/MD",
                        "-Xcompiler=/EHsc",
                        "-Xcompiler=/Zc:__cplusplus",
                        "-DNOMINMAX",
                        "-DWIN32_LEAN_AND_MEAN",
                        "-D_CRT_SECURE_NO_WARNINGS",
                    ]);
                    toolchain
                        .cxx
                        .parent()
                        .unwrap_or(toolchain.cxx.as_path())
                        .to_path_buf()
                } else {
                    command.arg("-Xcompiler=-fPIC");
                    toolchain.cxx.clone()
                };
                command.arg("-ccbin").arg(ccbin).args(CUDA_IEEE_MATH_FLAGS);
                for architecture in plan.architectures.split(';') {
                    command.arg(format!(
                        "--generate-code=arch=compute_{architecture},code=[compute_{architecture},sm_{architecture}]"
                    ));
                }
            }
            KernelRuntime::Hip => {
                // HIPCC is a Clang driver and compiles host and device code
                // itself, so there is no -ccbin equivalent. These translation
                // units export only extern "C" symbols and include no
                // libstdc++ container types across that boundary, so the host
                // compiler's dual ABI does not reach them.
                command
                    .args(["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-x", "hip"])
                    .args(HIP_IEEE_MATH_FLAGS);
                for architecture in plan.architectures.split(';') {
                    command.arg(format!("--offload-arch={architecture}"));
                }
            }
        }
        command
            .arg("-I")
            .arg(&compat_include)
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(output);
        run_guarded_checked(&mut command, &format!("compile {label}"), guard);
        validate_object(output);
    }
    Some(CudaBuild {
        runtime: plan.runtime,
        compiler: plan.compiler,
        toolkit_version: plan.toolkit_version,
        architectures: plan.architectures,
        runtime_directory: plan.runtime_directory,
        object,
        spine_object,
    })
}

fn plan_cuda_kernel_build(
    libtorch_major: u8,
    mode: CudaMode,
    guard: &PythonGuard,
    emit_cargo_metadata: bool,
) -> Option<KernelBuildPlan> {
    let Some(compiler) = discover_nvcc() else {
        if mode == CudaMode::On {
            panic!("DELTAFIN_CUDA_MOE=ON needs NVCC from CUDA 12.6 or any CUDA 13.x");
        }
        if emit_cargo_metadata {
            println!(
                "cargo:warning=CUDA-enabled LibTorch was found without NVCC; CUDA tensors and the int8 spine remain available with exact CPU MXFP4 fallback, but original-BF16 CUDA will fail early instead of expanding weights to FP32"
            );
        } else {
            eprintln!(
                "[xtask] CUDA-enabled LibTorch has no NVCC; testing CUDA tensors/int8 with CPU MXFP4 fallback; original-BF16 CUDA is intentionally unavailable"
            );
        }
        return None;
    };
    validate_cuda_driver(&compiler);
    let mut version_command = Command::new(&compiler);
    version_command.arg("--version");
    let version_output = run_guarded_output(&mut version_command, "query NVCC version", guard);
    let version_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&version_output.stdout),
        String::from_utf8_lossy(&version_output.stderr)
    );
    let (toolkit_version, toolkit_major, toolkit_minor) = parse_nvcc_version(&version_text);
    if toolkit_major != libtorch_major {
        panic!(
            "LibTorch requires CUDA {libtorch_major}.x, but NVCC {} belongs to CUDA {toolkit_version}; mixed CUDA major ABIs are forbidden",
            compiler.display()
        );
    }
    // CUDA minor releases are ABI-compatible within a major, so any 13.x
    // toolkit may build against the pinned cu130 LibTorch; 12.x is held to the
    // validated 12.6.
    if !matches!((toolkit_major, toolkit_minor), (12, 6) | (13, _)) {
        panic!(
            "Deltafin's PyTorch 2.13 CUDA gate accepts CUDA 12.6 or any CUDA 13.x; found {toolkit_version}"
        );
    }
    let toolkit_root = cuda_toolkit_root(&compiler);
    let runtime_directory = cuda_runtime_directory(&toolkit_root);
    let architectures = cuda_architectures(toolkit_major, toolkit_minor);
    Some(KernelBuildPlan {
        runtime: KernelRuntime::Cuda,
        compiler,
        toolkit_version,
        architectures,
        runtime_directory,
    })
}

fn plan_hip_kernel_build(
    mode: CudaMode,
    guard: &PythonGuard,
    emit_cargo_metadata: bool,
) -> Option<KernelBuildPlan> {
    // The device reduction is wavefront-agnostic as of the port, and the MXFP4
    // expert kernels reduce through shared memory and never read a lane at
    // all. That makes these kernels compilable for AMD; it does not make them
    // evidenced. No AMD device has executed them: there is no token-oracle
    // run, no timing arm, and no physical residency or fault evidence on ROCm.
    // Building them therefore stays opt-in, so a default build on an AMD host
    // keeps the exact CPU MXFP4 path instead of silently adopting an
    // accelerator no gate has ever confirmed.
    if mode != CudaMode::On {
        let note = "ROCm/HIP LibTorch detected: HIP tensors and the int8 spine are available with the exact CPU MXFP4 expert fallback. The device MXFP4 and original-BF16 kernels are wavefront-agnostic and can be built with DELTAFIN_CUDA_MOE=ON, but no AMD hardware has passed the token oracle, so they are not built by default";
        if emit_cargo_metadata {
            println!("cargo:warning={note}");
        } else {
            eprintln!("[xtask] {note}");
        }
        return None;
    }
    let Some(compiler) = discover_hipcc() else {
        panic!(
            "DELTAFIN_CUDA_MOE=ON selected a ROCm/HIP LibTorch root but found no hipcc; set DELTAFIN_HIPCC, HIP_PATH, or ROCM_PATH"
        );
    };
    validate_native_executable(&compiler, "HIPCC");
    let mut version_command = Command::new(&compiler);
    version_command.arg("--version");
    let version_output = run_guarded_output(&mut version_command, "query HIPCC version", guard);
    let version_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&version_output.stdout),
        String::from_utf8_lossy(&version_output.stderr)
    );
    let (toolkit_version, toolkit_major) = parse_hip_version(&version_text);
    // ROCm 6 is the first series where `warpSize` is the documented way to read
    // the wave width and `__AMDGCN_WAVEFRONT_SIZE__` is deprecated, which is
    // exactly the contract the reduction relies on. No specific ROCm release
    // is pinned the way CUDA 12.6/13.0 are, because no release has been
    // validated against the oracle; the floor is a compatibility bound, not
    // evidence.
    if toolkit_major < 6 {
        panic!(
            "Deltafin's HIP kernels require ROCm 6 or newer for the documented warpSize contract; found {toolkit_version}"
        );
    }
    let hip_root = hip_toolkit_root(&compiler);
    let runtime_directory = hip_runtime_directory(&hip_root);
    let architectures = hip_architectures();
    Some(KernelBuildPlan {
        runtime: KernelRuntime::Hip,
        compiler,
        toolkit_version,
        architectures,
        runtime_directory,
    })
}

/// NVCC's file name on this host: `nvcc.exe` on Windows, `nvcc` elsewhere.
fn nvcc_file_name() -> String {
    format!("nvcc{}", std::env::consts::EXE_SUFFIX)
}

fn discover_nvcc() -> Option<PathBuf> {
    for variable in ["CUDACXX", "CMAKE_CUDA_COMPILER"] {
        if let Some(value) = env::var_os(variable) {
            return Some(resolve_tool_value(variable, &value));
        }
    }
    for variable in ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(value) = env::var_os(variable) {
            let root = deltafin_sys::fs::canonicalize(&value).unwrap_or_else(|error| {
                panic!(
                    "resolve explicit CUDA toolkit {variable}={}: {error}",
                    PathBuf::from(&value).display()
                )
            });
            let candidate = root.join("bin").join(nvcc_file_name());
            if !candidate.is_file() {
                // A runtime/development-header package is sufficient for the
                // CUDA allocator provider. NVCC is independently required
                // only for the optional exact MXFP4 kernel.
                continue;
            }
            return Some(resolve_tool_path(&candidate, "NVCC"));
        }
    }
    find_on_path(OsStr::new(&nvcc_file_name())).map(|path| resolve_tool_path(&path, "NVCC"))
}

fn discover_hipcc() -> Option<PathBuf> {
    for variable in ["DELTAFIN_HIPCC", "HIPCXX"] {
        if let Some(value) = env::var_os(variable) {
            return Some(resolve_tool_value(variable, &value));
        }
    }
    for variable in ["HIP_PATH", "ROCM_PATH", "ROCM_HOME"] {
        if let Some(value) = env::var_os(variable) {
            let root = deltafin_sys::fs::canonicalize(&value).unwrap_or_else(|error| {
                panic!(
                    "resolve explicit ROCm root {variable}={}: {error}",
                    PathBuf::from(&value).display()
                )
            });
            let candidate = root.join("bin/hipcc");
            if !candidate.is_file() {
                continue;
            }
            return Some(resolve_tool_path(&candidate, "HIPCC"));
        }
    }
    find_on_path(OsStr::new("hipcc")).map(|path| resolve_tool_path(&path, "HIPCC"))
}

fn hip_toolkit_root(compiler: &Path) -> PathBuf {
    let inferred = compiler
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("HIPCC path has no ROCm root: {}", compiler.display()));
    let inferred = deltafin_sys::fs::canonicalize(inferred).unwrap_or_else(|error| {
        panic!("resolve inferred ROCm root {}: {error}", inferred.display())
    });
    let mut explicit_root: Option<PathBuf> = None;
    for variable in ["HIP_PATH", "ROCM_PATH", "ROCM_HOME"] {
        if let Some(value) = env::var_os(variable) {
            let root = deltafin_sys::fs::canonicalize(&value).unwrap_or_else(|error| {
                panic!(
                    "resolve explicit ROCm root {variable}={}: {error}",
                    PathBuf::from(&value).display()
                )
            });
            if let Some(previous) = &explicit_root {
                if previous != &root {
                    panic!(
                        "conflicting ROCm roots are forbidden: {} and {}",
                        previous.display(),
                        root.display()
                    );
                }
            }
            explicit_root = Some(root);
        }
    }
    // An explicit root that does not itself own the selected hipcc would give
    // a mixed toolchain the same way a mismatched CUDA_HOME would, so the two
    // must agree before either is trusted.
    let root = explicit_root.unwrap_or(inferred);
    let root_hipcc =
        deltafin_sys::fs::canonicalize(root.join("bin/hipcc")).unwrap_or_else(|error| {
            panic!(
                "selected ROCm root {} has no resolvable bin/hipcc: {error}",
                root.display()
            )
        });
    if root_hipcc != compiler {
        panic!(
            "mixed ROCm toolchains are forbidden: compiler {} is not {}/bin/hipcc",
            compiler.display(),
            root.display()
        );
    }
    root
}

fn hip_runtime_directory(root: &Path) -> PathBuf {
    for relative in ["lib", "lib64"] {
        let candidate = root.join(relative);
        if library_file(&candidate, "amdhip64").is_some() {
            return deltafin_sys::fs::canonicalize(&candidate).unwrap_or_else(|error| {
                panic!(
                    "resolve ROCm runtime directory {}: {error}",
                    candidate.display()
                )
            });
        }
    }
    panic!(
        "selected ROCm root {} has no libamdhip64 runtime directory",
        root.display()
    );
}

fn hip_architectures() -> String {
    if let Some(value) = env::var_os("DELTAFIN_HIP_ARCHITECTURES") {
        let value = value.to_string_lossy();
        if value.is_empty()
            || value.split(';').any(|part| {
                part.len() < 4
                    || !part.starts_with("gfx")
                    || !part[3..].bytes().all(|byte| byte.is_ascii_alphanumeric())
            })
        {
            panic!("DELTAFIN_HIP_ARCHITECTURES must be a semicolon-separated list of gfx targets");
        }
        return value.into_owned();
    }
    // gfx90a and gfx942 are the 64-lane CDNA parts the reduction rewrite
    // exists for; gfx1100 is a 32-lane RDNA3 control that exercises the same
    // source down the other wave width. None of the three has been executed.
    "gfx90a;gfx942;gfx1100".to_owned()
}

/// `hipcc --version` reports `HIP version: 6.2.41134-...`. Only the major and
/// the printable release are used; the build patch suffix is not a contract.
fn parse_hip_version(text: &str) -> (String, u8) {
    let release = text
        .split("HIP version:")
        .nth(1)
        .map(|tail| tail.trim_start())
        .and_then(|tail| tail.split_whitespace().next())
        .filter(|value| !value.is_empty() && value.starts_with(|byte: char| byte.is_ascii_digit()))
        .unwrap_or_else(|| {
            panic!(
                "could not parse the HIP version from HIPCC output: {}",
                one_line(text.as_bytes())
            )
        });
    let major = release
        .split(['.', '-'])
        .next()
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or_else(|| panic!("invalid HIP major version {release:?}"));
    (release.to_owned(), major)
}

fn cuda_toolkit_root(compiler: &Path) -> PathBuf {
    let inferred = compiler
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("NVCC path has no toolkit root: {}", compiler.display()));
    let inferred = deltafin_sys::fs::canonicalize(inferred).unwrap_or_else(|error| {
        panic!(
            "resolve inferred CUDA toolkit root {}: {error}",
            inferred.display()
        )
    });
    let mut explicit_root: Option<PathBuf> = None;
    for variable in ["CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(value) = env::var_os(variable) {
            let root = deltafin_sys::fs::canonicalize(&value).unwrap_or_else(|error| {
                panic!(
                    "resolve explicit CUDA toolkit {variable}={}: {error}",
                    PathBuf::from(&value).display()
                )
            });
            if let Some(previous) = &explicit_root {
                if previous != &root {
                    panic!(
                        "conflicting CUDA toolkit roots are forbidden: {} and {}",
                        previous.display(),
                        root.display()
                    );
                }
            }
            explicit_root = Some(root);
        }
    }
    let root = explicit_root.unwrap_or(inferred);
    let root_nvcc = deltafin_sys::fs::canonicalize(root.join("bin").join(nvcc_file_name()))
        .unwrap_or_else(|error| {
            panic!(
                "selected CUDA toolkit {} has no resolvable bin/nvcc: {error}",
                root.display()
            )
        });
    if root_nvcc != compiler {
        panic!(
            "mixed CUDA toolchains are forbidden: compiler {} is not {}/bin/nvcc",
            compiler.display(),
            root.display()
        );
    }
    root
}

fn cuda_runtime_directory(root: &Path) -> PathBuf {
    cuda_runtime_directory_optional(root).unwrap_or_else(|| {
        panic!(
            "selected CUDA toolkit {} has no libcudart runtime directory",
            root.display()
        )
    })
}

fn cuda_runtime_directory_optional(root: &Path) -> Option<PathBuf> {
    if target_os() == "windows" {
        return windows_cuda_runtime_directory(root);
    }
    let target_arch = target_arch();
    let (target_dir, multiarch_dir) = match target_arch.as_str() {
        "x86_64" => ("x86_64-linux", "x86_64-linux-gnu"),
        "aarch64" => ("aarch64-linux", "aarch64-linux-gnu"),
        arch => panic!("CUDA provider does not support target architecture {arch}"),
    };
    for candidate in [
        root.join("lib64"),
        root.join("lib"),
        root.join("targets").join(target_dir).join("lib"),
        root.join("lib").join(multiarch_dir),
    ] {
        let Ok(entries) = fs::read_dir(&candidate) else {
            continue;
        };
        let found = entries.filter_map(Result::ok).any(|entry| {
            entry
                .file_name()
                .as_encoded_bytes()
                .starts_with(b"libcudart.so")
                && entry.path().is_file()
        });
        if found {
            return Some(
                deltafin_sys::fs::canonicalize(&candidate).unwrap_or_else(|error| {
                    panic!(
                        "resolve CUDA runtime directory {}: {error}",
                        candidate.display()
                    )
                }),
            );
        }
    }
    None
}

/// The Windows CUDA toolkit names its runtime import library `cudart.lib` and
/// the DLL `cudart64_<major>.dll` (never `cudart.dll`), so the shared
/// `library_file` name check does not apply. Prefer a directory that carries
/// both, since the linker needs the import library and the loader the DLL.
fn windows_cuda_runtime_directory(root: &Path) -> Option<PathBuf> {
    let has_cudart_dll = |directory: &Path| {
        fs::read_dir(directory).ok().is_some_and(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                entry
                    .file_name()
                    .as_encoded_bytes()
                    .starts_with(b"cudart64_")
                    && entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
            })
        })
    };
    for relative in ["lib/x64", "lib", "lib/Win32"] {
        let candidate = root.join(relative);
        if candidate.join("cudart.lib").is_file() && has_cudart_dll(&candidate) {
            return Some(
                deltafin_sys::fs::canonicalize(&candidate).unwrap_or_else(|error| {
                    panic!(
                        "resolve CUDA runtime directory {}: {error}",
                        candidate.display()
                    )
                }),
            );
        }
    }
    // A DLL-only layout still lets the loader find cudart even when the import
    // library sits elsewhere; accept it as a last resort.
    for relative in ["bin", "lib/x64", "lib"] {
        let candidate = root.join(relative);
        if has_cudart_dll(&candidate) {
            return Some(
                deltafin_sys::fs::canonicalize(&candidate).unwrap_or_else(|error| {
                    panic!(
                        "resolve CUDA runtime directory {}: {error}",
                        candidate.display()
                    )
                }),
            );
        }
    }
    None
}

fn cuda_architectures(major: u8, minor: u8) -> String {
    if let Some(value) = env::var_os("DELTAFIN_CUDA_ARCHITECTURES") {
        let value = value.to_string_lossy();
        if value.is_empty()
            || value
                .split(';')
                .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            panic!(
                "DELTAFIN_CUDA_ARCHITECTURES must be a semicolon-separated numeric compute-capability list"
            );
        }
        return value.into_owned();
    }
    match (major, minor, target_arch().as_str()) {
        (13, _, "aarch64") => "80;90;100;110;120",
        (13, _, "x86_64") => "75;80;86;90;100;120",
        (12, 6, "aarch64") => "80;90",
        (12, 6, "x86_64") => "50;60;70;75;80;86;90",
        (_, _, arch) => panic!("unsupported CUDA toolkit/target combination for {arch}"),
    }
    .to_owned()
}

fn parse_nvcc_version(text: &str) -> (String, u8, u8) {
    let release = text
        .split("release ")
        .nth(1)
        .and_then(|tail| tail.split(',').next())
        .filter(|value| {
            !value.is_empty()
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b'.')
        })
        .unwrap_or_else(|| {
            panic!(
                "could not parse NVCC release from output: {}",
                one_line(text.as_bytes())
            )
        });
    let mut parts = release.split('.');
    let major = parts
        .next()
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or_else(|| panic!("invalid NVCC major version {release:?}"));
    let minor = parts
        .next()
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or_else(|| panic!("invalid NVCC minor version {release:?}"));
    if parts.next().is_some() {
        panic!("invalid NVCC release {release:?}");
    }
    let full = text
        .split(", V")
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .filter(|value| {
            !value.is_empty()
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b'.')
        })
        .unwrap_or(release)
        .to_owned();
    (full, major, minor)
}

fn validate_cuda_driver(path: &Path) {
    validate_native_executable(path, "NVCC");
}

fn validate_compiler(
    toolchain: &NativeToolchain,
    path: &Path,
    language: &str,
    guard: &PythonGuard,
) {
    let mut command = Command::new(path);
    command.arg(match toolchain.flavor {
        CompilerFlavor::Gnu => "--version",
        // `cl.exe` has no `--version`; its help is introduced by a banner
        // that names the compiler.
        CompilerFlavor::Msvc => "/?",
    });
    let output = run_guarded_output(
        &mut command,
        &format!("query selected {language} compiler"),
        guard,
    );
    let identity = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if toolchain.flavor == CompilerFlavor::Msvc {
        if !msvc::is_msvc_banner(&identity) {
            panic!(
                "Deltafin's Windows native provider requires Microsoft's compiler (or clang-cl) for {language}; selected {} reported {}",
                path.display(),
                one_line(identity.as_bytes())
            );
        }
        return;
    }
    let lower = identity.to_ascii_lowercase();
    if !(lower.contains("clang")
        || lower.contains("gcc")
        || lower.contains("g++")
        || lower.contains("free software foundation"))
    {
        panic!(
            "Deltafin's native provider requires GCC or Clang for {language}; selected {} reported {}",
            path.display(),
            one_line(identity.as_bytes())
        );
    }
}

fn resolve_required_native_tool(variable: &str, defaults: &[&str]) -> PathBuf {
    let path = if let Some(value) = env::var_os(variable) {
        resolve_tool_value(variable, &value)
    } else {
        defaults
            .iter()
            .find_map(|candidate| {
                let candidate = Path::new(candidate);
                if candidate.components().count() > 1 {
                    candidate.is_file().then(|| candidate.to_path_buf())
                } else {
                    find_on_path(candidate.as_os_str())
                }
            })
            .map(|path| resolve_tool_path(&path, variable))
            .unwrap_or_else(|| {
                panic!(
                    "could not locate required native build tool {variable}; tried {}",
                    defaults.join(", ")
                )
            })
    };
    validate_native_executable(&path, variable);
    path
}

fn resolve_tool_value(variable: &str, value: &OsStr) -> PathBuf {
    if value.is_empty() {
        panic!("explicit native build tool {variable} is empty");
    }
    let candidate = PathBuf::from(value);
    let path = if candidate.components().count() > 1 {
        candidate
    } else {
        find_on_path(value).unwrap_or_else(|| {
            panic!(
                "explicit native build tool {variable}={} was not found on PATH",
                candidate.display()
            )
        })
    };
    resolve_tool_path(&path, variable)
}

fn resolve_tool_path(path: &Path, label: &str) -> PathBuf {
    let path = deltafin_sys::fs::canonicalize(path).unwrap_or_else(|error| {
        panic!(
            "resolve native build tool {label}={}: {error}",
            path.display()
        )
    });
    let metadata = fs::metadata(&path)
        .unwrap_or_else(|error| panic!("inspect native build tool {}: {error}", path.display()));
    #[cfg(unix)]
    let executable = metadata.is_file() && metadata.permissions().mode() & 0o111 != 0;
    // Windows has no exec bit: a program is a `.exe` file, and the format
    // check that follows rejects scripts and shims renamed to one.
    #[cfg(windows)]
    let executable = metadata.is_file()
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"));
    if !executable {
        panic!(
            "native build tool must be an executable regular file: {}",
            path.display()
        );
    }
    // Every caller executes the returned path as a compiler, linker,
    // archiver, or binary-inspection tool.  Keep the admission rule here so a
    // fixed-path helper such as readelf cannot bypass the native-magic check
    // merely because it did not come through the normal CC/CXX/AR resolver.
    validate_native_executable(&path, label);
    path
}

fn find_on_path(name: &OsStr) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    let direct = env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file());
    if direct.is_some() {
        return direct;
    }
    // Windows resolves a bare name through `PATHEXT`; a name that already
    // carries an extension is exact.
    if target_os() != "windows" || Path::new(name).extension().is_some() {
        return None;
    }
    let extensions = env::var_os("PATHEXT")
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".to_owned());
    for extension in extensions.split(';').filter(|part| !part.is_empty()) {
        let candidate_name = format!("{}{}", name.to_string_lossy(), extension);
        if let Some(found) = env::split_paths(&path)
            .map(|directory| directory.join(&candidate_name))
            .find(|candidate| candidate.is_file())
        {
            return Some(found);
        }
    }
    None
}

/// Enough of a Windows program's headers to follow `e_lfanew` to its PE
/// signature and read the COFF header.
const PE_HEADER_BYTES: u64 = 4096;

fn read_pe_header(file: &mut fs::File) -> std::io::Result<Vec<u8>> {
    let mut header = Vec::new();
    file.take(PE_HEADER_BYTES).read_to_end(&mut header)?;
    Ok(header)
}

fn validate_native_executable(path: &Path, label: &str) {
    let mut file = fs::File::open(path).unwrap_or_else(|error| {
        panic!(
            "inspect native build tool {label} {}: {error}",
            path.display()
        )
    });
    if cfg!(windows) {
        let header = read_pe_header(&mut file).unwrap_or_else(|error| {
            panic!(
                "inspect native build tool {label} {}: {error}",
                path.display()
            )
        });
        if !deltafin_sys::pe::is_native_executable(&header) {
            panic!(
                "native build tool {label} is interpreted or is not a PE program: {}",
                path.display()
            );
        }
        return;
    }
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic).unwrap_or_else(|error| {
        panic!(
            "inspect native build tool {label} {}: {error}",
            path.display()
        )
    });
    if !is_native_magic(&magic) {
        panic!(
            "native build tool {label} is interpreted or has an unknown executable format: {}",
            path.display()
        );
    }
}

/// Non-panicking counterpart to `validate_native_executable`, for optional
/// tools with an established fallback. A candidate that cannot be opened,
/// cannot be read, or fails the magic check is simply not usable -- it is
/// never treated as "close enough" and never executed either way.
fn is_native_executable(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    if cfg!(windows) {
        return read_pe_header(&mut file)
            .is_ok_and(|header| deltafin_sys::pe::is_native_executable(&header));
    }
    let mut magic = [0_u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }
    is_native_magic(&magic)
}

fn is_native_magic(magic: &[u8; 4]) -> bool {
    matches!(
        *magic,
        [0x7f, b'E', b'L', b'F']
            | [0xfe, 0xed, 0xfa, 0xce]
            | [0xfe, 0xed, 0xfa, 0xcf]
            | [0xce, 0xfa, 0xed, 0xfe]
            | [0xcf, 0xfa, 0xed, 0xfe]
            | [0xca, 0xfe, 0xba, 0xbe]
            | [0xbe, 0xba, 0xfe, 0xca]
            | [0xca, 0xfe, 0xba, 0xbf]
            | [0xbf, 0xba, 0xfe, 0xca]
    )
}

fn c_string_literal_bytes(bytes: &[u8]) -> String {
    let mut escaped = String::with_capacity(bytes.len());
    for byte in bytes {
        match byte {
            b'\\' => escaped.push_str("\\\\"),
            b'\"' => escaped.push_str("\\\""),
            0x20..=0x7e => escaped.push(char::from(*byte)),
            byte => escaped.push_str(&format!("\\{byte:03o}")),
        }
    }
    escaped
}

fn apply_tool_environment(
    command: &mut Command,
    environment: &[(std::ffi::OsString, std::ffi::OsString)],
) {
    for (name, value) in environment {
        command.env(name, value);
    }
}

fn sanitize_native_environment(command: &mut Command) {
    for variable in [
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "BASH_ENV",
        "ENV",
        "ZDOTDIR",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "OBJC_INCLUDE_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
        "CCC_OVERRIDE_OPTIONS",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CUDAHOSTCXX",
        "CLANG_MODULE_CACHE_PATH",
        "DEVELOPER_DIR",
        "SDKROOT",
        "TOOLCHAINS",
    ] {
        command.env_remove(variable);
    }
    // Microsoft's tools read extra options from these; the audited command
    // line is the only source of options.
    for variable in msvc::INJECTION_VARIABLES {
        command.env_remove(variable);
    }
}

fn run_guarded_raw_output(command: &mut Command, operation: &str, guard: &PythonGuard) -> Output {
    guard.prepare(command);
    let rendered = format!("{command:?}");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to {operation} with {rendered}: {error}"));
    guard.assert_clean();
    output
}

fn run_guarded_output(command: &mut Command, operation: &str, guard: &PythonGuard) -> Output {
    let rendered = format!("{command:?}");
    let output = run_guarded_raw_output(command, operation, guard);
    if !output.status.success() {
        panic!(
            "failed to {operation} with {rendered}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output
}

fn run_guarded_checked(command: &mut Command, operation: &str, guard: &PythonGuard) {
    run_guarded_try(command, operation, guard).unwrap_or_else(|message| panic!("{message}"));
}

/// Like `run_guarded_checked`, but a failing command is a value rather than a
/// panic, so a caller compiling many translation units can report every
/// failure from one run instead of the first.
fn run_guarded_try(
    command: &mut Command,
    operation: &str,
    guard: &PythonGuard,
) -> Result<(), String> {
    let rendered = format!("{command:?}");
    let output = run_guarded_raw_output(command, operation, guard);
    if !output.status.success() {
        return Err(format!(
            "failed to {operation} with {rendered}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = without_compiler_echo(&output.stdout);
    if !stdout.is_empty() {
        println!("cargo:warning={operation}: {}", one_line(&stdout));
    }
    if !output.stderr.is_empty() {
        println!("cargo:warning={operation}: {}", one_line(&output.stderr));
    }
    Ok(())
}

/// `cl.exe` prints the name of every source file it compiles, banner or no
/// banner. That is not a diagnostic, and as a Cargo warning it would make every
/// translation unit of every Windows build look like a problem.
fn without_compiler_echo(stdout: &[u8]) -> Vec<u8> {
    if target_os() != "windows" {
        return stdout.to_vec();
    }
    let mut kept = Vec::with_capacity(stdout.len());
    for line in String::from_utf8_lossy(stdout).lines() {
        let name = line.trim();
        let is_echo = !name.is_empty()
            && !name.contains([' ', '(', ':'])
            && [".c", ".cpp", ".cc", ".cxx"]
                .iter()
                .any(|extension| name.to_ascii_lowercase().ends_with(extension));
        if !is_echo {
            kept.extend_from_slice(line.as_bytes());
            kept.push(b'\n');
        }
    }
    kept
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("Cargo did not set {name}"))
}

/// An object file's name on this target: `.obj` for MSVC, `.o` elsewhere.
fn object_file_name(stem: &str) -> String {
    if target_os() == "windows" {
        format!("{stem}.obj")
    } else {
        format!("{stem}.o")
    }
}

/// The provider archive's name: what rustc's `static=deltafin_provider_abi`
/// resolves to on this target.
fn provider_archive_name() -> String {
    if target_os() == "windows" {
        msvc::static_library_name("deltafin_provider_abi")
    } else {
        "libdeltafin_provider_abi.a".to_owned()
    }
}

fn native_test_executable_name(test: &str) -> String {
    format!("deltafin-native-test-{test}{}", env::consts::EXE_SUFFIX)
}

fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| env::consts::OS.to_owned())
}

fn target_arch() -> String {
    env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| env::consts::ARCH.to_owned())
}

/// `cl.exe` does not accept the Windows verbatim (`\\?\`) prefix that
/// `canonicalize` produces, so the LibTorch root must be spelled as an
/// ordinary drive path before it reaches the compiler's `/external:I`.
fn normalize_verbatim_path(path: PathBuf) -> PathBuf {
    if target_os() == "windows" {
        let text = path.to_string_lossy();
        if let Some(rest) = text.strip_prefix(r"\\?\") {
            return PathBuf::from(rest);
        }
    }
    path
}

fn find_torch_root(repository: &Path, emit_cargo_metadata: bool) -> PathBuf {
    let cuda_mode = cuda_mode();
    for variable in ["DELTAFIN_TORCH_ROOT", "LIBTORCH"] {
        if let Some(path) = env::var_os(variable).map(PathBuf::from) {
            return normalize_verbatim_path(validate_explicit_torch_root(
                variable,
                &path,
                emit_cargo_metadata,
            ));
        }
    }

    let target = bootstrap_target();
    if cuda_mode == CudaMode::On {
        panic!(
            "DELTAFIN_CUDA_MOE=ON explicitly requests CUDA, but Deltafin's automatic PyTorch {0} bootstrap currently has only an audited CPU artifact for {target}. Set DELTAFIN_TORCH_ROOT or LIBTORCH to a trusted CUDA-enabled LibTorch root. Deltafin will not install or label the CPU runtime as CUDA support",
            "2.13.0"
        );
    }
    let destination = deltafin_bootstrap::repository_toolchain_path(repository, target);
    if destination.exists() {
        let validated = deltafin_bootstrap::validate_repository_toolchain(repository, target)
            .unwrap_or_else(|error| {
                panic!(
                    "refusing invalid repo-local PyTorch toolchain {}: {error}",
                    destination.display()
                )
            });
        register_toolchain_inputs(&validated.tracked_paths, emit_cargo_metadata);
        return normalize_verbatim_path(validated.torch_root);
    }

    // Migration bridge: a Deltafin binary from before the standalone
    // bootstrap existed will fast-forward and immediately invoke this build
    // script. Install through the reviewed Rust library here so that upgrade
    // never needs an old venv or an impossible recursive Cargo invocation.
    if emit_cargo_metadata {
        println!("cargo:warning=installing pinned Python-free PyTorch C++ runtime for {target}");
    } else {
        eprintln!("[xtask] installing pinned Python-free PyTorch C++ runtime for {target}");
    }
    deltafin_bootstrap::install(&InstallOptions {
        repository_root: repository.to_path_buf(),
        target,
    })
    .unwrap_or_else(|error| {
        panic!(
            "could not bootstrap the pinned native PyTorch runtime into {}: {error}",
            destination.display()
        )
    });
    let validated = deltafin_bootstrap::validate_repository_toolchain(repository, target)
        .unwrap_or_else(|error| {
            panic!(
                "newly bootstrapped PyTorch toolchain failed validation in {}: {error}",
                destination.display()
            )
        });
    register_toolchain_inputs(&validated.tracked_paths, emit_cargo_metadata);
    normalize_verbatim_path(validated.torch_root)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaMode {
    Auto,
    On,
    Off,
}

/// Embed only inert build settings needed to reproduce this binary's native
/// provider profile during `deltafin upgrade`. Values are hex encoded so an
/// unusual Unix path cannot inject a Cargo build-script directive. The
/// repo-local authenticated CPU bootstrap is represented symbolically rather
/// than freezing its absolute checkout path.
fn emit_upgrade_build_profile(
    torch_root: &Path,
    explicit_torch_root: bool,
    effective_cuda: Option<&(String, PathBuf)>,
    cuda_provider_enabled: bool,
) {
    println!("cargo:rustc-env=DELTAFIN_BUILD_PROFILE_FORMAT=v2");
    println!(
        "cargo:rustc-env=DELTAFIN_BUILD_TORCH_SOURCE={}",
        if explicit_torch_root {
            "explicit"
        } else {
            "bootstrap"
        }
    );
    println!(
        "cargo:rustc-env=DELTAFIN_BUILD_TORCH_ROOT={}",
        if explicit_torch_root {
            encode_profile_bytes(torch_root.as_os_str().as_encoded_bytes())
        } else {
            "-".to_owned()
        }
    );
    println!(
        "cargo:rustc-env=DELTAFIN_BUILD_CUDA_MOE={}",
        if effective_cuda.is_some() {
            "ON"
        } else {
            "OFF"
        }
    );
    println!(
        "cargo:rustc-env=DELTAFIN_BUILD_CUDA_PROVIDER={}",
        if cuda_provider_enabled { "ON" } else { "OFF" }
    );
    for (variable, embedded) in [
        (
            "DELTAFIN_CUDA_ARCHITECTURES",
            "DELTAFIN_BUILD_CUDA_ARCHITECTURES",
        ),
        ("CUDACXX", "DELTAFIN_BUILD_CUDACXX"),
        ("CMAKE_CUDA_COMPILER", "DELTAFIN_BUILD_CMAKE_CUDA_COMPILER"),
        ("CUDAToolkit_ROOT", "DELTAFIN_BUILD_CUDA_TOOLKIT_ROOT"),
        ("CUDA_HOME", "DELTAFIN_BUILD_CUDA_HOME"),
        ("CUDA_PATH", "DELTAFIN_BUILD_CUDA_PATH"),
    ] {
        let effective = match (variable, effective_cuda) {
            ("DELTAFIN_CUDA_ARCHITECTURES", Some((architectures, _))) => {
                Some(encode_profile_bytes(architectures.as_bytes()))
            }
            ("CUDACXX" | "CMAKE_CUDA_COMPILER", Some((_, compiler))) => Some(encode_profile_bytes(
                compiler.as_os_str().as_encoded_bytes(),
            )),
            ("CUDAToolkit_ROOT" | "CUDA_HOME" | "CUDA_PATH", _) if cuda_provider_enabled => {
                env::var_os(variable).map(|value| {
                    let canonical =
                        deltafin_sys::fs::canonicalize(&value).unwrap_or_else(|error| {
                            panic!(
                                "resolve CUDA build-profile path {variable}={}: {error}",
                                PathBuf::from(&value).display()
                            )
                        });
                    encode_profile_bytes(canonical.as_os_str().as_encoded_bytes())
                })
            }
            _ => None,
        };
        let encoded = effective.unwrap_or_else(|| "-".to_owned());
        println!("cargo:rustc-env={embedded}={encoded}");
    }
}

fn encode_profile_bytes(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(4 + bytes.len() * 2);
    encoded.push_str("hex:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn cuda_mode() -> CudaMode {
    let Some(value) = env::var_os("DELTAFIN_CUDA_MOE") else {
        return CudaMode::Auto;
    };
    match value.to_string_lossy().to_ascii_uppercase().as_str() {
        "AUTO" => CudaMode::Auto,
        "ON" => CudaMode::On,
        "OFF" => CudaMode::Off,
        _ => panic!("DELTAFIN_CUDA_MOE must be AUTO, ON, or OFF"),
    }
}

fn bootstrap_target() -> PlatformTarget {
    match (target_os().as_str(), target_arch().as_str()) {
        ("macos", "aarch64") => PlatformTarget::MacosArm64,
        ("linux", "x86_64") => PlatformTarget::LinuxX86_64,
        ("linux", "aarch64") => PlatformTarget::LinuxAarch64,
        ("windows", "x86_64") => PlatformTarget::WindowsX86_64,
        (os, arch) => panic!(
            "native Deltafin has no pinned PyTorch {0} toolchain for {os}/{arch}",
            "2.13.0"
        ),
    }
}

fn register_toolchain_inputs(paths: &[PathBuf], emit_cargo_metadata: bool) {
    if !emit_cargo_metadata {
        return;
    }
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn validate_explicit_torch_root(variable: &str, path: &Path, emit_cargo_metadata: bool) -> PathBuf {
    let path = deltafin_sys::fs::canonicalize(path)
        .unwrap_or_else(|error| panic!("resolve explicit {variable}={}: {error}", path.display()));
    if !is_torch_root(&path) {
        panic!(
            "explicit {variable}={} does not contain the exact native PyTorch header/runtime layout",
            path.display()
        );
    }
    let mut required_files = vec![
        path.join(TORCH_CPP_API_HEADER),
        path.join(TORCH_VERSION_HEADER),
    ];
    for component in ["torch", "torch_cpu", "c10"] {
        required_files.push(path.join("lib").join(shared_library_name(component)));
        if target_os() == "windows" {
            // A DLL alone cannot be linked against.
            required_files.push(path.join("lib").join(format!("{component}.lib")));
        }
    }
    for required in required_files {
        let metadata = fs::symlink_metadata(&required).unwrap_or_else(|error| {
            panic!(
                "inspect explicit native PyTorch file {}: {error}",
                required.display()
            )
        });
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            panic!(
                "explicit native PyTorch file must be a non-symlink regular file: {}",
                required.display()
            );
        }
        if emit_cargo_metadata {
            println!("cargo:rerun-if-changed={}", required.display());
        }
    }
    let torch_cuda = library_file(&path.join("lib"), "torch_cuda");
    let c10_cuda = library_file(&path.join("lib"), "c10_cuda");
    if torch_cuda.is_some() != c10_cuda.is_some() {
        panic!(
            "explicit {variable}={} has an incomplete CUDA runtime: libtorch_cuda and libc10_cuda must either both exist or both be absent",
            path.display()
        );
    }
    for required in [torch_cuda.as_ref(), c10_cuda.as_ref()]
        .into_iter()
        .flatten()
    {
        let metadata = fs::symlink_metadata(required).unwrap_or_else(|error| {
            panic!(
                "inspect explicit CUDA library {}: {error}",
                required.display()
            )
        });
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            panic!(
                "explicit CUDA library must be a non-symlink regular file: {}",
                required.display()
            );
        }
        if emit_cargo_metadata {
            println!("cargo:rerun-if-changed={}", required.display());
        }
    }
    if cuda_mode() == CudaMode::On {
        if target_os() != "linux" && target_os() != "windows" {
            panic!("DELTAFIN_CUDA_MOE=ON is supported only for Linux and Windows targets");
        }
        for required in [torch_cuda, c10_cuda] {
            required.unwrap_or_else(|| {
                panic!(
                    "DELTAFIN_CUDA_MOE=ON requires a CUDA-enabled PyTorch root; {} contains only the CPU runtime",
                    path.display()
                )
            });
        }
    }
    path
}

fn is_torch_root(path: &Path) -> bool {
    path.is_dir()
        && path.join(TORCH_CPP_API_HEADER).is_file()
        && path.join(TORCH_VERSION_HEADER).is_file()
}

/// The file a linker names for LibTorch component `name` in `directory`: the
/// shared object on Unix, the import library on Windows (the DLL itself is
/// what the loader needs at run time, and is checked beside it).
fn library_file(directory: &Path, name: &str) -> Option<PathBuf> {
    let candidate = match target_os().as_str() {
        "macos" => directory.join(format!("lib{name}.dylib")),
        "windows" => {
            let import_library = directory.join(format!("{name}.lib"));
            if import_library.is_file() && !directory.join(format!("{name}.dll")).is_file() {
                panic!(
                    "LibTorch import library {} has no DLL beside it",
                    import_library.display()
                );
            }
            import_library
        }
        _ => directory.join(format!("lib{name}.so")),
    };
    candidate.is_file().then_some(candidate)
}

/// The shared library's own file name for `name` on this target.
fn shared_library_name(name: &str) -> String {
    match target_os().as_str() {
        "macos" => format!("lib{name}.dylib"),
        "windows" => format!("{name}.dll"),
        _ => format!("lib{name}.so"),
    }
}

/// Which GPU runtime backs a LibTorch tree's `torch_cuda`/`c10_cuda` pair.
///
/// A ROCm build of PyTorch keeps the CUDA library names and the `at::kCUDA`
/// device type, reaching AMD hardware through HIP, so the pair's presence
/// alone cannot distinguish the two. The dependency does: NVIDIA builds carry
/// `libcudart.so.N` / `cudart64_N.dll` and ROCm builds carry
/// `libamdhip64.so` / `amdhip64.dll`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuRuntime {
    Cuda(u8),
    Rocm,
}

/// The loadable module a runtime dependency scan must read: the shared object
/// on Unix, the DLL on Windows. `library_file` answers the linker's question
/// (the import library on Windows), which does not itself name the DLLs the
/// module imports.
fn runtime_module(directory: &Path, name: &str) -> Option<PathBuf> {
    if target_os() == "windows" {
        let module = directory.join(format!("{name}.dll"));
        module.is_file().then_some(module)
    } else {
        library_file(directory, name)
    }
}

fn detect_gpu_runtime(torch_lib: &Path) -> Option<GpuRuntime> {
    let host_os = target_os();
    if host_os != "linux" && host_os != "windows" {
        return None;
    }
    let cuda_libraries = [
        library_file(torch_lib, "c10_cuda"),
        library_file(torch_lib, "torch_cuda"),
    ];
    if cuda_libraries.iter().all(Option::is_none) {
        return None;
    }
    if cuda_libraries.iter().any(Option::is_none) {
        panic!(
            "selected native PyTorch root has an incomplete CUDA pair in {}",
            torch_lib.display()
        );
    }

    // libc10_cuda owns PyTorch's low-level device runtime calls and therefore
    // carries the direct cudart or HIP dependency: an ELF `DT_NEEDED` entry on
    // Unix, a PE import on Windows. Inspecting this small library avoids reading
    // the multi-gigabyte torch payload on every build.
    let c10_cuda = runtime_module(torch_lib, "c10_cuda").unwrap_or_else(|| {
        panic!(
            "selected native PyTorch root has no loadable c10_cuda module in {}",
            torch_lib.display()
        )
    });
    let (found_hip, found_12, found_13) = if host_os == "windows" {
        (
            file_contains(&c10_cuda, b"amdhip64.dll\0"),
            file_contains(&c10_cuda, b"cudart64_12.dll\0"),
            file_contains(&c10_cuda, b"cudart64_13.dll\0"),
        )
    } else {
        (
            file_contains(&c10_cuda, b"libamdhip64.so\0"),
            file_contains(&c10_cuda, b"libcudart.so.12\0"),
            file_contains(&c10_cuda, b"libcudart.so.13\0"),
        )
    };
    if found_hip && (found_12 || found_13) {
        panic!(
            "selected LibTorch root {} references both the HIP and CUDA runtimes",
            torch_lib.display()
        );
    }
    if found_hip {
        return Some(GpuRuntime::Rocm);
    }
    match (found_12, found_13) {
        (true, false) => Some(GpuRuntime::Cuda(12)),
        (false, true) => Some(GpuRuntime::Cuda(13)),
        (false, false) => panic!(
            "could not identify the GPU runtime ABI required by the selected LibTorch root {}; expected a dependency on libcudart.so.12/cudart64_12.dll, libcudart.so.13/cudart64_13.dll, or libamdhip64.so/amdhip64.dll",
            torch_lib.display()
        ),
        (true, true) => panic!(
            "selected LibTorch root {} references both CUDA 12 and CUDA 13 runtimes",
            torch_lib.display()
        ),
    }
}

fn detect_libtorch_cxx11_abi(torch_lib: &Path) -> Option<u8> {
    if target_os() != "linux" {
        return None;
    }
    let c10 = library_file(torch_lib, "c10").unwrap_or_else(|| {
        panic!(
            "selected native PyTorch root has no libc10 in {}",
            torch_lib.display()
        )
    });
    // Use a c10-owned exported constructor whose std::string parameter changes
    // mangling across libstdc++'s dual ABI. Looking for arbitrary __cxx11 text
    // is insufficient because an ABI-0 library may still contain unrelated
    // implementation symbols from ABI-1 dependencies.
    let abi_1 = file_contains(
        &c10,
        b"_ZN3c1010IndexErrorC1ENS_14SourceLocationENSt7__cxx1112basic_string",
    );
    let abi_0 = file_contains(&c10, b"_ZN3c1010IndexErrorC1ENS_14SourceLocationESs");
    match (abi_0, abi_1) {
        (false, true) => Some(1),
        (true, false) => Some(0),
        (false, false) => panic!(
            "could not identify _GLIBCXX_USE_CXX11_ABI from {}; the selected LibTorch ABI is not auditable",
            c10.display()
        ),
        (true, true) => panic!(
            "selected LibTorch {} exports both libstdc++ string ABIs",
            c10.display()
        ),
    }
}

fn file_contains(path: &Path, needle: &[u8]) -> bool {
    assert!(!needle.is_empty());
    let mut file = fs::File::open(path).unwrap_or_else(|error| {
        panic!("read CUDA dependency metadata {}: {error}", path.display())
    });
    let mut buffer = vec![0_u8; 1 << 20];
    let mut overlap = Vec::with_capacity(needle.len().saturating_sub(1));
    loop {
        let bytes = file.read(&mut buffer).unwrap_or_else(|error| {
            panic!("scan CUDA dependency metadata {}: {error}", path.display())
        });
        if bytes == 0 {
            return false;
        }
        let mut window = Vec::with_capacity(overlap.len() + bytes);
        window.extend_from_slice(&overlap);
        window.extend_from_slice(&buffer[..bytes]);
        if window
            .windows(needle.len())
            .any(|candidate| candidate == needle)
        {
            return true;
        }
        let retained = window.len().min(needle.len().saturating_sub(1));
        overlap.clear();
        overlap.extend_from_slice(&window[window.len() - retained..]);
    }
}

fn link_required(directory: &Path, name: &str) {
    if library_file(directory, name).is_none() {
        panic!(
            "required native provider library {} is absent from {}",
            shared_library_name(name),
            directory.display()
        );
    }
    println!("cargo:rustc-link-lib=dylib={name}");
}

fn link_optional(directory: &Path, name: &str) {
    if library_file(directory, name).is_some() {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
}

fn run_checked(command: &mut Command, operation: &str) {
    let rendered = format!("{command:?}");
    let Output {
        status,
        stdout,
        stderr,
    } = command
        .output()
        .unwrap_or_else(|error| panic!("failed to {operation} with {rendered}: {error}"));
    if !status.success() {
        panic!(
            "failed to {operation} with {rendered}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        );
    }
    if !stdout.is_empty() {
        println!("cargo:warning={operation}: {}", one_line(&stdout));
    }
    if !stderr.is_empty() {
        println!("cargo:warning={operation}: {}", one_line(&stderr));
    }
}

fn one_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

// These fixtures are fake ELF binaries and `#!/bin/sh` scripts; the Windows
// toolchain code has its own tests.
#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "deltafin-native-build-{label}-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create native-build test directory");
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn is_native_executable_accepts_elf_and_rejects_scripts_or_absent_files() {
        let root = TestDirectory::new("native-executable-check");
        let elf = root.0.join("elf-tool");
        fs::write(&elf, [0x7f, b'E', b'L', b'F', 0, 0, 0, 0]).unwrap();
        assert!(is_native_executable(&elf));

        let script = root.0.join("script-tool");
        fs::write(&script, "#!/bin/sh\necho not native\n").unwrap();
        assert!(!is_native_executable(&script));

        let missing = root.0.join("does-not-exist");
        assert!(!is_native_executable(&missing));
    }

    #[test]
    fn ldconfig_style_selection_falls_back_past_a_non_native_first_candidate() {
        // Mirrors linux_loader_cache_directories's own selection predicate:
        // a script at the first candidate path must not win over a genuine
        // native binary at the second, and neither should panic.
        let root = TestDirectory::new("ldconfig-candidate-selection");
        let first = root.0.join("sbin-ldconfig");
        fs::write(&first, "#!/bin/sh\necho wrapper\n").unwrap();
        let second = root.0.join("usr-sbin-ldconfig");
        fs::write(&second, [0x7f, b'E', b'L', b'F', 0, 0, 0, 0]).unwrap();

        let selected = [first.as_path(), second.as_path()]
            .into_iter()
            .find(|candidate| candidate.is_file() && is_native_executable(candidate));
        assert_eq!(selected, Some(second.as_path()));

        // Neither candidate being native must fall through cleanly, not panic.
        fs::write(&second, "#!/bin/sh\necho also a wrapper\n").unwrap();
        let selected = [first.as_path(), second.as_path()]
            .into_iter()
            .find(|candidate| candidate.is_file() && is_native_executable(candidate));
        assert_eq!(selected, None);
    }

    #[test]
    fn fixed_path_helper_rejects_shebang_before_invocation() {
        let root = TestDirectory::new("interpreted-helper");
        let marker = root.0.join("executed");
        let helper = root.0.join("readelf");
        fs::write(
            &helper,
            format!("#!/bin/sh\nprintf invoked > '{}'\n", marker.display()),
        )
        .expect("write inert interpreted helper fixture");
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o700))
            .expect("make interpreted helper fixture executable");

        let rejected =
            std::panic::catch_unwind(|| resolve_tool_path(&helper, "fixed-path fixture"));
        assert!(rejected.is_err(), "an executable shebang was admitted");
        assert!(
            !marker.exists(),
            "the rejected shebang fixture was invoked during admission"
        );
    }

    #[test]
    fn cuda_kernel_flags_preserve_fp32_math_contract() {
        assert_eq!(
            CUDA_IEEE_MATH_FLAGS,
            [
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
                "--fmad=true",
            ]
        );
        assert!(
            !CUDA_IEEE_MATH_FLAGS.contains(&"--use_fast_math"),
            "authoritative CUDA kernels must not enable fast math"
        );
    }
}

/// Tests of the Windows deployment and PE-audit logic. They exercise plain
/// file-system and byte-parsing code, so they run on every host.
#[cfg(test)]
mod windows_graph_tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static SERIAL: AtomicU64 = AtomicU64::new(0);

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let root = env::temp_dir().join(format!(
                "deltafin-native-build-{label}-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            Self(deltafin_sys::fs::canonicalize(root).unwrap())
        }

        fn write(&self, relative: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// `<root>/target/release/build/<package>-<hash>/out`, as Cargo lays it out.
    fn out_dir(scratch: &Scratch) -> PathBuf {
        let out = scratch
            .0
            .join("target/release/build/deltafin-0123456789abcdef/out");
        fs::create_dir_all(&out).unwrap();
        out
    }

    #[test]
    fn nvcc_discovery_uses_the_host_executable_suffix() {
        assert_eq!(
            nvcc_file_name(),
            format!("nvcc{}", std::env::consts::EXE_SUFFIX)
        );
    }

    #[test]
    fn windows_cudart_discovery_needs_both_the_import_library_and_a_dll() {
        let scratch = Scratch::new("cudart");
        scratch.write("lib/x64/cudart.lib", b"lib");
        scratch.write("lib/x64/cudart64_13.dll", b"dll");
        assert_eq!(
            windows_cuda_runtime_directory(&scratch.0).unwrap(),
            deltafin_sys::fs::canonicalize(scratch.0.join("lib/x64")).unwrap()
        );

        // A DLL-only layout is still accepted for the loader.
        let dll_only = Scratch::new("cudart-dll");
        dll_only.write("bin/cudart64_13.dll", b"dll");
        assert!(windows_cuda_runtime_directory(&dll_only.0).is_some());

        // The import library without any DLL cannot be loaded.
        let bare = Scratch::new("cudart-bare");
        bare.write("lib/cudart.lib", b"lib");
        assert!(windows_cuda_runtime_directory(&bare.0).is_none());
    }

    #[test]
    fn gpu_runtime_detection_reads_the_c10_cuda_dependency() {
        let scratch = Scratch::new("gpu-runtime");
        let lib = scratch.0.join("lib");
        if target_os() == "windows" {
            // The import library names the module; the module carries the
            // runtime import the scan reads.
            scratch.write("lib/c10_cuda.lib", b"import");
            scratch.write("lib/torch_cuda.lib", b"import");
            scratch.write("lib/c10_cuda.dll", b"cudart64_13.dll\0");
            scratch.write("lib/torch_cuda.dll", b"cudart64_13.dll\0");
        } else {
            scratch.write("lib/libc10_cuda.so", b"libcudart.so.13\0");
            scratch.write("lib/libtorch_cuda.so", b"libcudart.so.13\0");
        }
        assert_eq!(detect_gpu_runtime(&lib), Some(GpuRuntime::Cuda(13)));
    }

    #[test]
    fn one_library_is_deployed_and_an_identical_one_is_left_alone() {
        let scratch = Scratch::new("deploy-one");
        let source = scratch.write("torch/lib/c10.dll", b"library bytes");
        let destination = scratch.0.join("bin/c10.dll");
        fs::create_dir_all(destination.parent().unwrap()).unwrap();

        deploy_one_library(&source, &destination);
        assert_eq!(fs::read(&destination).unwrap(), b"library bytes");

        // Identical content: no replacement, so a running process that has the
        // file mapped does not make a no-op deployment fail.
        let before = fs::metadata(&destination).unwrap().modified().unwrap();
        deploy_one_library(&source, &destination);
        assert_eq!(
            fs::metadata(&destination).unwrap().modified().unwrap(),
            before
        );
    }

    #[test]
    fn a_changed_library_replaces_its_deployed_copy() {
        let scratch = Scratch::new("deploy-change");
        let source = scratch.write("torch/lib/torch_cpu.dll", b"version one");
        let destination = scratch.0.join("bin/torch_cpu.dll");
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        deploy_one_library(&source, &destination);

        // A new toolchain installs a new file (never an edit in place).
        fs::remove_file(&source).unwrap();
        fs::write(&source, b"version two!").unwrap();
        deploy_one_library(&source, &destination);

        assert_eq!(fs::read(&destination).unwrap(), b"version two!");
    }

    #[test]
    fn a_deployed_name_that_is_not_a_plain_file_is_refused() {
        let scratch = Scratch::new("deploy-refuse");
        let source = scratch.write("torch/lib/c10.dll", b"library bytes");
        let directory = scratch.0.join("bin/c10.dll");
        fs::create_dir_all(&directory).unwrap();
        let outcome = std::panic::catch_unwind(|| deploy_one_library(&source, &directory));
        assert!(outcome.is_err(), "a directory was replaced by a library");
        assert!(directory.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_source_or_destination_is_refused() {
        let scratch = Scratch::new("deploy-symlink");
        let real = scratch.write("elsewhere/real.dll", b"payload");
        let link = scratch.0.join("torch/lib/c10.dll");
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        deltafin_sys::fs::symlink(&real, &link).unwrap();
        let destination = scratch.0.join("bin/c10.dll");
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        assert!(std::panic::catch_unwind(|| deploy_one_library(&link, &destination)).is_err());
        assert!(!destination.exists());

        let source = scratch.write("torch/lib/torch.dll", b"payload");
        let planted = scratch.0.join("bin/torch.dll");
        deltafin_sys::fs::symlink(&real, &planted).unwrap();
        assert!(std::panic::catch_unwind(|| deploy_one_library(&source, &planted)).is_err());
    }

    #[test]
    fn deployment_reaches_the_profile_directory_and_deps_and_only_takes_dlls() {
        let scratch = Scratch::new("deploy-all");
        let torch_lib = scratch.0.join("torch/lib");
        for name in ["c10.dll", "torch.dll", "torch_cpu.dll", "libiomp5md.dll"] {
            scratch.write(&format!("torch/lib/{name}"), name.as_bytes());
        }
        for name in ["c10.lib", "torch.lib", "readme.txt"] {
            scratch.write(&format!("torch/lib/{name}"), b"not a runtime library");
        }
        let out = out_dir(&scratch);

        deploy_runtime_libraries(std::slice::from_ref(&torch_lib), &out);

        let profile = scratch.0.join("target/release");
        for directory in [profile.clone(), profile.join("deps")] {
            for name in ["c10.dll", "torch.dll", "torch_cpu.dll", "libiomp5md.dll"] {
                assert_eq!(
                    fs::read(directory.join(name)).unwrap(),
                    name.as_bytes(),
                    "{name} in {}",
                    directory.display()
                );
            }
            for name in ["c10.lib", "torch.lib", "readme.txt"] {
                assert!(!directory.join(name).exists(), "{name} was deployed");
            }
        }
    }

    #[test]
    fn deployment_refuses_an_unexpected_cargo_layout_and_python_libraries() {
        let scratch = Scratch::new("deploy-layout");
        let torch_lib = scratch.0.join("torch/lib");
        scratch.write("torch/lib/c10.dll", b"c10");
        // Not <profile>/build/<package>/out.
        let odd = scratch.0.join("somewhere/else/out");
        fs::create_dir_all(&odd).unwrap();
        assert!(
            std::panic::catch_unwind(|| deploy_runtime_libraries(
                std::slice::from_ref(&torch_lib),
                &odd
            ))
            .is_err()
        );

        // A Python binding library is refused by name, wherever it came from.
        scratch.write("torch/lib/torch_python.dll", b"never");
        let out = out_dir(&scratch);
        assert!(
            std::panic::catch_unwind(|| deploy_runtime_libraries(
                std::slice::from_ref(&torch_lib),
                &out
            ))
            .is_err()
        );
    }

    #[test]
    fn the_compilers_filename_echo_is_not_a_warning_but_diagnostics_are() {
        let echoed = b"provider_abi.cpp\r\n";
        let diagnostic = b"provider_abi.cpp\r\nD:\\src\\a.cpp(12): warning C4100: unused\r\n";
        if target_os() == "windows" {
            assert!(without_compiler_echo(echoed).is_empty());
            let kept = String::from_utf8(without_compiler_echo(diagnostic)).unwrap();
            assert!(kept.contains("warning C4100") && !kept.starts_with("provider_abi.cpp"));
        } else {
            // Other compilers print nothing of the kind; nothing is dropped.
            assert_eq!(without_compiler_echo(echoed), echoed);
        }
    }

    #[test]
    fn the_pe_audit_resolves_system_dlls_and_names_what_it_cannot() {
        use deltafin_sys::pe::testing::SyntheticImage;

        let scratch = Scratch::new("pe-audit");
        let torch_lib = scratch.0.join("torch/lib");
        scratch.write(
            "torch/lib/c10.dll",
            &SyntheticImage::dll()
                .importing(&["api-ms-win-core-synch-l1-2-0.dll"])
                .build(),
        );
        let test_dir = scratch.0.join("test");
        let executable = scratch.write(
            "test/test.exe",
            &SyntheticImage::executable()
                .importing(&[
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "c10.dll",
                    "no-such-library.dll",
                ])
                .build(),
        );
        let _ = &test_dir;

        // Without a Windows system directory (any non-Windows host) the
        // verdict is the same as on Windows for what is not a system DLL.
        if deltafin_sys::fs::system_directory().is_err() {
            let error = audit_pe_dependencies(&executable, std::slice::from_ref(&torch_lib))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("locate the Windows system directory"),
                "{error}"
            );
            return;
        }
        let error = audit_pe_dependencies(&executable, std::slice::from_ref(&torch_lib))
            .unwrap_err()
            .to_string();
        assert!(error.contains("no-such-library.dll"), "{error}");
    }
}
