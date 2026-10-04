//! Git source-publication policy for the native product.
//!
//! This test intentionally asks Git what will be published rather than
//! trusting the developer worktree.  Historical local Python experiments may
//! remain as ignored files, but neither they nor an untracked native build
//! input may pass a release checkout gate.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const WORKSPACE_INPUTS: &[&str] = &[
    // `.documentor-public-ignore` is deliberately absent: it is publishing control, not a build
    // input, and the publishing tool withholds it from the public mirror by design, so requiring
    // it here can never be satisfied in a published checkout.
    ".github/workflows/device-kernel-compile.yml",
    ".github/workflows/native-source-policy.yml",
    ".gitignore",
    "docs/COMPILED-RUNTIME.md",
    "Cargo.toml",
    "Cargo.lock",
    "docs/OPTIMIZATIONS.md",
    "README.md",
    "docs/REQUIREMENTS.md",
    "docs/THIRD_PARTY_NOTICES.md",
    "native/deltafin/Cargo.toml",
    "native/deltafin/TOKENIZER_AUDIT.md",
    "native/deltafin/build.rs",
    "native/deltafin-bootstrap/Cargo.toml",
    "native/deltafin-curl-sys-direct/Cargo.toml",
    "native/deltafin-curl-sys-direct/build.rs",
    "native/deltafin-curl-sys-direct/build_support.rs",
    "native/deltafin-curl-sys-direct/lib.rs",
    "native/deltafin-native-build/Cargo.toml",
    "native/deltafin-sys/Cargo.toml",
    "native/deltafin-xtask/Cargo.toml",
    "native/provider_gate/deny_python.c.in",
    "native/provider_gate/README.md",
    "native/provider_gate/qwen_hf_oracle_mps_f16.json",
    "tools/README.md",
];

const RUST_SOURCE_ROOTS: &[&str] = &[
    "native/deltafin/src",
    "native/deltafin/tests",
    "native/deltafin-bootstrap/src",
    "native/deltafin-curl-sys-direct/tests",
    "native/deltafin-native-build/src",
    "native/deltafin-sys/src",
    "native/deltafin-xtask/src",
];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("native/deltafin must remain two levels below the repository")
        .to_path_buf()
}

fn tracked_files(root: &Path) -> BTreeSet<PathBuf> {
    let output = Command::new("git")
        .args(["ls-files", "--cached", "-z", "--"])
        .current_dir(root)
        .output()
        .expect("execute native Git for publication audit");
    assert!(
        output.status.success(),
        "git ls-files failed during publication audit: {}",
        String::from_utf8_lossy(&output.stderr),
    );
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            // SAFETY: git emits these bytes as UTF-8 paths, which are valid
            // in this platform's OS-string encoding (bytes on Unix, WTF-8,
            // a superset of UTF-8, on Windows).
            PathBuf::from(unsafe { OsString::from_encoded_bytes_unchecked(path.to_vec()) })
        })
        .collect()
}

fn walk_rust_sources(root: &Path, relative: &Path, output: &mut BTreeSet<PathBuf>) {
    let directory = root.join(relative);
    let mut entries: Vec<_> = fs::read_dir(&directory)
        .unwrap_or_else(|error| {
            panic!(
                "scan publication source root {}: {error}",
                directory.display()
            )
        })
        .map(|entry| entry.expect("read publication source entry"))
        .collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type().unwrap_or_else(|error| {
            panic!("inspect publication input {}: {error}", path.display())
        });
        assert!(
            !file_type.is_symlink(),
            "publication source graph contains a symbolic link: {}",
            path.display(),
        );
        if file_type.is_dir() {
            let nested = path
                .strip_prefix(root)
                .expect("walked publication source remains under repository");
            walk_rust_sources(root, nested, output);
        } else if file_type.is_file()
            && path.extension().and_then(|extension| extension.to_str()) == Some("rs")
        {
            output.insert(
                path.strip_prefix(root)
                    .expect("publication Rust source remains under repository")
                    .to_path_buf(),
            );
        }
    }
}

fn canonical_inputs(root: &Path) -> BTreeSet<PathBuf> {
    let mut inputs: BTreeSet<PathBuf> = WORKSPACE_INPUTS.iter().map(PathBuf::from).collect();
    inputs.extend(
        deltafin_native_build::PRODUCTION_PROVIDER_SOURCES
            .iter()
            .map(PathBuf::from),
    );
    for spec in deltafin_native_build::NATIVE_TEST_SPECS {
        inputs.insert(PathBuf::from(spec.main_source));
        inputs.extend(spec.extra_sources.iter().map(PathBuf::from));
    }
    for relative in RUST_SOURCE_ROOTS {
        walk_rust_sources(root, Path::new(relative), &mut inputs);
    }
    inputs
}

fn interpreted_extension(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "py" | "pyw"
            | "pyi"
            | "sh"
            | "bash"
            | "zsh"
            | "fish"
            | "ksh"
            | "csh"
            | "js"
            | "mjs"
            | "cjs"
            | "ts"
            | "tsx"
            | "jsx"
            | "pl"
            | "pm"
            | "rb"
            | "lua"
            | "php"
            | "ps1"
            | "psm1"
            | "bat"
            | "cmd"
            | "tcl"
            | "r"
            | "jl"
            | "groovy"
            | "exs"
            | "clj"
            | "cljs"
            | "coffee"
            | "command"
    )
}

fn native_source_extension(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "c" | "cc"
            | "cpp"
            | "cxx"
            | "h"
            | "hh"
            | "hpp"
            | "hxx"
            | "m"
            | "mm"
            | "metal"
            | "cu"
            | "cuh"
            | "s"
            | "asm"
    )
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn native_python_extension_surface(bytes: &[u8]) -> bool {
    [
        b"PYBIND11_MODULE".as_slice(),
        b"#include <Python.h>".as_slice(),
        b"#include \"Python.h\"".as_slice(),
        b"#include <pybind11/".as_slice(),
        b"#include \"pybind11/".as_slice(),
        b"#include <torch/extension.h>".as_slice(),
        b"#include \"torch/extension.h\"".as_slice(),
    ]
    .iter()
    .any(|needle| contains_bytes(bytes, needle))
}

fn python_package_manifest(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "pyproject.toml"
            | "pipfile"
            | "pipfile.lock"
            | "poetry.lock"
            | "setup.py"
            | "setup.cfg"
            | "tox.ini"
            | "environment.yml"
            | "environment.yaml"
    ) || (lower.starts_with("requirements") && lower.ends_with(".txt"))
}

fn interpreted_shebang(bytes: &[u8]) -> bool {
    let first_line = bytes
        .split(|byte| *byte == b'\n')
        .next()
        .unwrap_or_default();
    if !first_line.starts_with(b"#!") {
        return false;
    }
    let Ok(line) = std::str::from_utf8(first_line) else {
        return true;
    };
    let mut words = line[2..].split_ascii_whitespace();
    let Some(mut program) = words.next() else {
        return true;
    };
    if program.rsplit('/').next() == Some("env") {
        let mut selected = None;
        for word in words {
            if word.starts_with('-') {
                continue;
            }
            selected = Some(word);
            break;
        }
        let Some(value) = selected else {
            return true;
        };
        program = value;
    }
    let basename = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .to_ascii_lowercase();
    basename.strip_prefix("python").is_some_and(|suffix| {
        suffix.is_empty()
            || suffix
                .chars()
                .all(|character| character.is_ascii_digit() || character == '.')
    }) || basename
        .strip_prefix("pypy")
        .is_some_and(|suffix| suffix.chars().all(|character| character.is_ascii_digit()))
        || matches!(
            basename.as_str(),
            "sh" | "bash"
                | "zsh"
                | "fish"
                | "ksh"
                | "csh"
                | "node"
                | "deno"
                | "perl"
                | "ruby"
                | "lua"
                | "php"
                | "pwsh"
                | "powershell"
                | "rscript"
                | "julia"
        )
}

#[test]
fn interpreter_classifier_covers_source_and_package_entrypoints() {
    for path in [
        "tools/kimi_run.py",
        "scripts/launch.sh",
        "web/task.mjs",
        "windows/setup.ps1",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
    ] {
        let path = Path::new(path);
        assert!(
            interpreted_extension(path) || python_package_manifest(path),
            "classifier missed {path:?}",
        );
    }
    for path in [
        "native/provider.cpp",
        "native/kernel.metal",
        "native/runtime.rs",
        ".github/workflows/native-source-policy.yml",
    ] {
        let path = Path::new(path);
        assert!(!interpreted_extension(path));
        assert!(!python_package_manifest(path));
    }
    assert!(interpreted_shebang(b"#!/usr/bin/env python3\n"));
    assert!(interpreted_shebang(b"#!/usr/bin/env sh\n"));
    assert!(interpreted_shebang(b"#!/bin/sh\n"));
    assert!(!interpreted_shebang(b"//! ordinary Rust source\n"));
    assert!(native_source_extension(Path::new("native/provider.mm")));
    assert!(native_source_extension(Path::new("native/kernel.metal")));
    assert!(!native_source_extension(Path::new("native/runtime.rs")));
    assert!(native_python_extension_surface(
        b"#include <torch/extension.h>\nPYBIND11_MODULE(x, m) {}\n"
    ));
}

#[test]
fn tracked_publication_is_compiled_only_and_contains_every_build_input() {
    let root = repository_root();
    let tracked = tracked_files(&root);
    let required = canonical_inputs(&root);
    assert!(!tracked.is_empty(), "publication audit found no Git index");

    let forbidden: Vec<_> = tracked
        .iter()
        .filter(|path| interpreted_extension(path) || python_package_manifest(path))
        .cloned()
        .collect();
    assert!(
        forbidden.is_empty(),
        "tracked source publication contains interpreted/Python inputs; keep local historical files ignored and remove them from the Git index:\n{}",
        forbidden
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    );

    let orphan_native_sources: Vec<_> = tracked
        .iter()
        .filter(|path| native_source_extension(path) && !required.contains(*path))
        .cloned()
        .collect();
    assert!(
        orphan_native_sources.is_empty(),
        "tracked native source is outside the canonical production/test graph; retire probes into ignored experiments or classify them explicitly:\n{}",
        orphan_native_sources
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    );

    for relative in &tracked {
        let path = root.join(relative);
        let metadata = fs::symlink_metadata(&path).unwrap_or_else(|error| {
            panic!(
                "inspect tracked publication file {}: {error}",
                path.display()
            )
        });
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            continue;
        }
        let mut bytes = [0_u8; 512];
        let length = fs::File::open(&path)
            .and_then(|mut file| file.read(&mut bytes))
            .unwrap_or_else(|error| {
                panic!("read tracked publication file {}: {error}", path.display())
            });
        assert!(
            !interpreted_shebang(&bytes[..length]),
            "tracked source publication contains an extensionless interpreted entrypoint: {}",
            relative.display(),
        );
        if native_source_extension(relative) {
            let source = fs::read(&path).unwrap_or_else(|error| {
                panic!("read tracked native source {}: {error}", path.display())
            });
            assert!(
                !native_python_extension_surface(&source),
                "tracked native source exposes a Python/PyBind extension boundary: {}",
                relative.display(),
            );
        }
    }

    let missing: Vec<_> = required.difference(&tracked).cloned().collect();
    assert!(
        missing.is_empty(),
        "canonical native build/test inputs are absent from the Git publication index:\n{}",
        missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    );
    for relative in required {
        let metadata = fs::symlink_metadata(root.join(&relative)).unwrap_or_else(|error| {
            panic!(
                "inspect canonical publication input {}: {error}",
                relative.display()
            )
        });
        assert!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "canonical publication input is not a regular non-symlink file: {}",
            relative.display(),
        );
    }
}
