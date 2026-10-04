//! Source-release policy for the compiled production boundary.
//!
//! This is intentionally an integration test rather than a Python linter: the
//! policy itself must remain runnable in the same native toolchain as the
//! product and must not need an interpreter to decide what is publishable.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const PUBLIC_DOCUMENTS: &[&str] = &[
    "README.md",
    "docs/NATIVE-RUNTIME.md",
    "docs/PERFORMANCE.md",
    "docs/STORAGE.md",
    "docs/OPTIMIZATIONS.md",
    "docs/COMPILED-RUNTIME.md",
    "docs/THIRD_PARTY_NOTICES.md",
    "docs/REQUIREMENTS.md",
    "native/deltafin-curl-sys-direct/README.md",
    "native/deltafin/TOKENIZER_AUDIT.md",
    "tools/README.md",
    "native/provider_gate/README.md",
];

const QUARANTINED_PYTHON_ROOTS: &[&str] = &[
    "convert_experts_scale4.py",
    "fetch_spine.py",
    "fetch_v2.py",
    "kimi_run.py",
    "serve_openai.py",
    "setup_dspark.py",
];

const RETIRED_PYTHON_ROOTS: &[&str] = &[
    "bench.py",
    "build_native.py",
    "convert_spine_int8.py",
    "convert_npz_cache.py",
    "fetch_experts_all.py",
    "selftest.py",
    "setup_draft.py",
    "setup_k3.py",
    "test_bench.py",
    "test_build_native.py",
    "test_model_source_pin.py",
    "test_native_build_guidance.py",
    "test_setup_draft.py",
    "test_setup_k3.py",
    "test_upgrade.py",
    "test_warm_expert_cache.py",
    "upgrade.py",
    "warm_expert_cache.py",
];

const RETIRED_NATIVE_HARNESSES: &[&str] = &[
    "native/provider_gate/CMakeLists.txt",
    "native/provider_gate/build_and_run.sh",
    "native/provider_gate/cmake/assert_no_python.cmake",
    "native/provider_gate/cmake/embed_metal_source.cmake",
    "native/deltafin/audit_binary.sh",
];

const NAMED_DEVELOPMENT_ENTRYPOINTS: &[&str] = &[
    "analyze_trace.py",
    "apple_silicon.py",
    "build_spine_layer_pack.py",
    "convert_spine_int4.py",
    "expert_locality.py",
    "idea_inventory.py",
    "int4_loader.py",
    "int4_quality_report.py",
    "metal_mps_bridge.py",
    "mixed_spine.py",
    "quantize_spine.py",
    "spine_codec.py",
    "spine_sensitivity.py",
];

const DEVELOPMENT_PREFIXES: &[&str] = &["bench_", "probe_", "test_", "validate_"];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("native/deltafin must remain two levels below the repository")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read release-policy input {}: {error}", path.display()))
}

fn tracked_tool_files(root: &Path) -> BTreeSet<String> {
    let output = Command::new("git")
        .args(["ls-files", "--cached", "--", "tools"])
        .current_dir(root)
        .output()
        .expect("execute native Git for retired-tool source audit");
    assert!(
        output.status.success(),
        "git ls-files failed during retired-tool source audit: {}",
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout)
        .expect("tracked tool paths must be UTF-8")
        .lines()
        .map(str::to_owned)
        .collect()
}

fn is_interpreted_command(line: &str) -> bool {
    let mut line = line.trim();
    for prompt in ["$ ", "> "] {
        if let Some(rest) = line.strip_prefix(prompt) {
            line = rest.trim_start();
        }
    }
    let lower = line.to_ascii_lowercase();
    [
        "python ",
        "python2 ",
        "python3 ",
        "py ",
        "pip ",
        "pip3 ",
        "./venv/bin/python",
        "sh ",
        "bash ",
        "zsh ",
        "/bin/sh ",
        "/bin/bash ",
        "/usr/bin/env python",
        "/usr/bin/env sh",
        "/usr/bin/env bash",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
        || lower.split_ascii_whitespace().next().is_some_and(|head| {
            head.starts_with("./")
                && [".py", ".sh", ".bash", ".zsh"]
                    .iter()
                    .any(|suffix| head.ends_with(suffix))
        })
}

fn is_python_entrypoint(source: &str) -> bool {
    source
        .lines()
        .next()
        .is_some_and(|line| line.starts_with("#!") && line.contains("python"))
        || source.contains("if __name__ == \"__main__\":")
}

fn is_named_development_entrypoint(name: &str) -> bool {
    NAMED_DEVELOPMENT_ENTRYPOINTS.contains(&name)
        || DEVELOPMENT_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

fn walk_regular_files(root: &Path, output: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(root)
        .unwrap_or_else(|error| panic!("scan release-policy root {}: {error}", root.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "read release-policy entry under {}: {error}",
                root.display()
            )
        });
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!("inspect release-policy path {}: {error}", path.display())
        });
        if file_type.is_dir() {
            walk_regular_files(&path, output);
        } else if file_type.is_file() {
            output.push(path);
        }
    }
}

#[test]
fn public_documentation_never_invokes_an_interpreted_entrypoint() {
    let root = repository_root();
    for relative in PUBLIC_DOCUMENTS {
        let path = root.join(relative);
        let source = read(&path);
        for (index, line) in source.lines().enumerate() {
            assert!(
                !is_interpreted_command(line),
                "{}:{} publishes an interpreted command: {line:?}",
                relative,
                index + 1,
            );
        }
    }
}

#[test]
fn production_source_graph_contains_only_compiled_languages() {
    let root = repository_root();
    let mut files = Vec::new();
    walk_regular_files(&root.join("native/deltafin/src"), &mut files);
    walk_regular_files(&root.join("native/deltafin-bootstrap/src"), &mut files);
    walk_regular_files(&root.join("native/deltafin-native-build/src"), &mut files);
    walk_regular_files(&root.join("native/deltafin-sys/src"), &mut files);
    walk_regular_files(&root.join("native/deltafin-xtask/src"), &mut files);
    for path in files {
        assert_eq!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs"),
            "interpreted or unclassified source entered a production Rust source tree: {}",
            path.display(),
        );
    }

    for relative in [
        "Cargo.toml",
        "native/deltafin/Cargo.toml",
        "native/deltafin-bootstrap/Cargo.toml",
        "native/deltafin-native-build/Cargo.toml",
        "native/deltafin-xtask/Cargo.toml",
    ] {
        let source = read(&root.join(relative));
        for forbidden in [".py\"", ".sh\"", ".bash\"", ".zsh\""] {
            assert!(
                !source.contains(forbidden),
                "{relative} selects an interpreted entrypoint containing {forbidden:?}",
            );
        }
    }

    let adapter = read(&root.join("native/deltafin/build.rs"));
    let build = read(&root.join("native/deltafin-native-build/src/lib.rs"));
    assert!(
        adapter.contains("deltafin_native_build::run_production_build()"),
        "Cargo build.rs stopped delegating to the single shared native build graph",
    );
    for forbidden in [
        "Command::new(\"python",
        "Command::new(\"python3",
        "Command::new(\"sh\")",
        "Command::new(\"bash\")",
        "Command::new(\"zsh\")",
        "Command::new(\"cmake\")",
        "Command::new(\"make\")",
        "CMakeLists.txt",
        ".py\"",
        ".sh\"",
    ] {
        assert!(
            !build.contains(forbidden),
            "Rust-owned production build reintroduced an interpreted/CMake edge: {forbidden:?}",
        );
    }
}

#[test]
fn obsolete_native_harnesses_and_bootstrap_cli_stay_retired() {
    let root = repository_root();
    for relative in RETIRED_NATIVE_HARNESSES {
        assert!(
            !root.join(relative).exists(),
            "obsolete second native build/test path was reintroduced: {relative}",
        );
    }

    let mut native_files = Vec::new();
    walk_regular_files(&root.join("native"), &mut native_files);
    for path in native_files {
        let extension = path.extension().and_then(|value| value.to_str());
        let name = path.file_name().and_then(|value| value.to_str());
        assert!(
            !matches!(extension, Some("sh" | "bash" | "zsh")) && name != Some("CMakeLists.txt"),
            "native tree reintroduced a shell/CMake build or test harness: {}",
            path.display(),
        );
    }

    let bootstrap_manifest = read(&root.join("native/deltafin-bootstrap/Cargo.toml"));
    assert!(
        !bootstrap_manifest.contains("[[bin]]")
            && !root.join("native/deltafin-bootstrap/src/main.rs").exists(),
        "the internal toolchain bootstrap must remain a library called by the shared Rust build graph, not a second public executable",
    );
}

#[test]
fn every_provider_translation_unit_is_classified_by_the_shared_rust_graph() {
    let root = repository_root();
    let graph = read(&root.join("native/deltafin-native-build/src/lib.rs"));
    let provider = root.join("native/provider_gate");
    let entries = fs::read_dir(&provider).expect("scan native provider translation units");
    for entry in entries {
        let entry = entry.expect("read native provider translation unit");
        if !entry
            .file_type()
            .expect("inspect provider source")
            .is_file()
        {
            continue;
        }
        let path = entry.path();
        let extension = path.extension().and_then(|value| value.to_str());
        if !matches!(extension, Some("c" | "cpp" | "mm" | "metal" | "cu")) {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let relative = path
            .strip_prefix(&root)
            .expect("provider source remains under repository root")
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let full_literal = format!("\"{relative}\"");
        let bare_literal = format!("\"{name}\"");
        let test_only = name.contains("_test.") || name.as_ref() == "provider_gate.cpp";
        let expected = if test_only {
            &full_literal
        } else {
            &bare_literal
        };
        assert!(
            graph.contains(expected),
            "provider translation unit is absent from the declarative production/test graph: {}",
            path.display(),
        );
    }
    for relative in [
        "tools/fused_gemv_batch.c",
        "tools/metal_moe.mm",
        "tools/metal/moe_mxfp4.metal",
        "tools/cuda_moe_kernels.cu",
    ] {
        assert!(
            graph.contains(relative),
            "external provider translation unit is absent from the shared Rust graph: {relative}",
        );
    }
}

#[test]
fn native_experiments_stay_quarantined_outside_the_shared_graph() {
    let root = repository_root();
    let ignore = read(&root.join(".gitignore"));
    let graph = read(&root.join("native/deltafin-native-build/src/lib.rs"));

    assert!(
        ignore
            .lines()
            .any(|line| line.trim() == "/native/experiments/"),
        "native experiments must remain explicitly excluded from source releases",
    );
    assert!(
        !ignore.lines().any(|line| {
            line.trim()
                .trim_start_matches('/')
                .starts_with("native/provider_gate/")
        }),
        "provider sources must never be hidden by .gitignore; move probes into native/experiments instead",
    );
    assert!(
        !graph.contains("native/experiments/"),
        "the shared production/test build graph must not depend on ignored native experiments",
    );
}

#[test]
fn legacy_python_roots_are_explicitly_quarantined_or_retired() {
    let root = repository_root();
    let tools = root.join("tools");
    let boundary = read(&tools.join("README.md"));
    let tracked = tracked_tool_files(&root);
    let ignore = read(&root.join(".gitignore"));

    assert!(
        ignore.lines().any(|line| line.trim() == "/tools/**/*.py"),
        "local historical Python must remain ignored after retirement from the source release",
    );

    for name in QUARANTINED_PYTHON_ROOTS {
        assert!(
            boundary.contains(&format!("`{name}`")),
            "tools/README.md does not classify quarantined root {name}",
        );
    }
    for name in RETIRED_PYTHON_ROOTS {
        assert!(
            !tracked.contains(&format!("tools/{name}")),
            "retired Python root was reintroduced into the source release: tools/{name}",
        );
    }

    let entries = fs::read_dir(&tools).expect("scan tools entrypoints");
    for entry in entries {
        let entry = entry.expect("read tools entrypoint");
        let path = entry.path();
        let file_type = entry.file_type().expect("inspect tools entrypoint");
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let extension = path.extension().and_then(|extension| extension.to_str());
        if matches!(extension, Some("sh" | "bash" | "zsh" | "ps1" | "js")) {
            panic!(
                "top-level interpreted tools entrypoint is forbidden; classify development material outside the product tools root: {}",
                path.display(),
            );
        }
        if extension != Some("py") || !is_python_entrypoint(&read(&path)) {
            continue;
        }
        assert!(
            QUARANTINED_PYTHON_ROOTS.contains(&name.as_ref())
                || is_named_development_entrypoint(&name),
            "unclassified top-level Python entrypoint could be mistaken for a supported command: {}",
            path.display(),
        );
    }
}

#[test]
fn public_cli_help_names_only_the_compiled_executable() {
    let root = repository_root();
    let source = read(&root.join("native/deltafin/src/cli.rs"));
    for forbidden in [
        "tools/kimi_run.py",
        "tools/serve_openai.py",
        "tools/setup_k3.py",
        "tools/upgrade.py",
        "python tools/",
        "./venv/bin/python",
    ] {
        assert!(
            !source.contains(forbidden),
            "native CLI source points users back to a legacy interpreter path: {forbidden}",
        );
    }
}

#[test]
fn macos_release_metal_is_compiled_and_embedded_at_build_time() {
    let root = repository_root();
    let adapter = read(&root.join("native/deltafin/build.rs"));
    let build = read(&root.join("native/deltafin-native-build/src/lib.rs"));

    for required in [
        "provider_route_mailbox.metal",
        "tools/metal/moe_mxfp4.metal",
        "-mmacosx-version-min=14.0",
        "write_embedded_binary_header(",
        "DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1=1",
    ] {
        assert!(
            build.contains(required),
            "production build stopped compiling and embedding reviewed Metal input {required:?}",
        );
    }
    assert!(
        !adapter.contains("DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1")
            && build
                .matches("DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1=1")
                .count()
                == 1
            && build.contains("ProviderFlavor::MetalSourceDevelopment"),
        "runtime Metal source compilation must exist only in the isolated xtask source-development flavor",
    );

    for relative in [
        "tools/metal_moe.mm",
        "native/provider_gate/provider_route_mailbox.mm",
    ] {
        let source = read(&root.join(relative));
        assert!(
            source.contains("DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1")
                && source.contains("newLibraryWithData"),
            "{relative} no longer loads the build-time metallib",
        );
    }

    let moe = read(&root.join("tools/metal_moe.mm"));
    assert!(
        moe.contains("#if defined(DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1)")
            && moe.contains("newLibraryWithSource"),
        "the development-only source compiler must stay explicitly gated",
    );
}

#[test]
fn dependency_features_preserve_the_audited_build_profile() {
    let root = repository_root();
    let runtime = read(&root.join("native/deltafin/Cargo.toml"));
    let bootstrap = read(&root.join("native/deltafin-bootstrap/Cargo.toml"));
    let lock = read(&root.join("Cargo.lock"));

    assert!(
        runtime.contains(
            "tokenizers = { version = \"=0.22.2\", default-features = false, features = [\"onig\"] }"
        ),
        "the measured Qwen tokenizer profile must retain Oniguruma without restoring tokenizers' unused default features",
    );
    for forbidden_feature in ["progressbar", "esaxx_fast"] {
        assert!(
            !runtime.contains(forbidden_feature),
            "the runtime manifest re-enabled unused tokenizers feature {forbidden_feature:?}",
        );
    }
    for forbidden_package in ["indicatif", "console"] {
        assert!(
            !lock.contains(&format!("name = \"{forbidden_package}\"")),
            "the locked graph restored the unused tokenizers progress dependency {forbidden_package:?}",
        );
    }
    for (manifest, label) in [(&runtime, "runtime"), (&bootstrap, "bootstrap")] {
        assert!(
            manifest.contains("features = [\"deflate-flate2-zlib-rs\"]"),
            "{label} ZIP support must use the pure-Rust zlib implementation",
        );
        assert!(
            !manifest.contains("features = [\"deflate-flate2-zlib\"]"),
            "{label} ZIP support reintroduced the system-zlib build probe",
        );
    }
}

/// The device-kernel compile workflow exists to prove that the two `.cu`
/// translation units build under the real vendor toolchains. It can only do
/// that if it passes the same arithmetic flags the shared Rust build graph
/// passes; a compile check running different flags would report success on a
/// configuration the product never ships, which is worse than no check at all.
#[test]
fn the_device_kernel_compile_workflow_matches_the_shared_build_flags() {
    let root = repository_root();
    let graph = read(&root.join("native/deltafin-native-build/src/lib.rs"));
    let workflow = read(&root.join(".github/workflows/device-kernel-compile.yml"));

    for constant in ["CUDA_IEEE_MATH_FLAGS", "HIP_IEEE_MATH_FLAGS"] {
        let block = graph
            .split(&format!("const {constant}: &[&str] = &["))
            .nth(1)
            .and_then(|tail| tail.split("];").next())
            .unwrap_or_else(|| panic!("{constant} is absent from the shared build graph"));
        let flags: Vec<&str> = block.split('"').skip(1).step_by(2).collect();
        assert!(
            !flags.is_empty(),
            "{constant} parsed as an empty flag list",
        );
        for flag in flags {
            assert!(
                workflow.contains(flag),
                "the device-kernel compile workflow omits {flag:?} from {constant}",
            );
        }
    }

    // The default HIP offload list is the reason the reduction rewrite exists:
    // two 64-lane CDNA parts and one 32-lane RDNA control. A workflow that
    // compiled only one width would not exercise the wave-dependent tree.
    let hip_defaults = graph
        .split("fn hip_architectures()")
        .nth(1)
        .and_then(|tail| tail.split("\nfn ").next())
        .expect("hip_architectures is absent from the shared build graph");
    // Anchored on the returned literal, not on the first "gfx" text: the
    // override validator in the same function also spells a bare "gfx" prefix,
    // and matching that would reduce every assertion below to a tautology.
    let defaults = hip_defaults
        .split(".to_owned()")
        .next()
        .and_then(|before_return| before_return.rsplit('"').nth(1))
        .filter(|value| value.starts_with("gfx") && value.contains(';'))
        .expect("hip_architectures does not return a semicolon-separated gfx literal");
    for architecture in defaults.split(';') {
        // Matched as the offload flag rather than as bare text: every default
        // target is also named by the code-object verification step, so a
        // substring search would still pass after the compile stopped
        // requesting it.
        assert!(
            workflow.contains(&format!("--offload-arch={architecture}")),
            "the device-kernel compile workflow does not offload to the default target {architecture:?}",
        );
    }

    // Both sources must be named, or the leg silently checks half the surface.
    for source in [
        "tools/cuda_moe_kernels.cu",
        "native/provider_gate/provider_spine_bf16_cuda.cu",
    ] {
        assert!(
            workflow.contains(source),
            "the device-kernel compile workflow does not compile {source}",
        );
    }
}
