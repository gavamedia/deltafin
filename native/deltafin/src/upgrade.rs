//! Conservative, Python-free updater for the compiled Deltafin binary.

use std::ffi::{OsStr, OsString};
use std::fs;
#[cfg(any(feature = "runtime", test))]
use std::io::Read;
#[cfg(all(unix, any(feature = "runtime", test)))]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
#[cfg(feature = "runtime")]
use std::process::{Command, Stdio};

use crate::error::{DeltafinError, Result};
use crate::sys::path::os_string_from_bytes;

const PRESERVED_ROOTS: &[&str] = &[
    ".cache",
    ".deltafin",
    ".venv",
    "checkpoints",
    "k3-cache",
    "k3-experts",
    "k3-experts-scale4",
    "k3-meta",
    "k3-model",
    "models",
    "venv",
    "weights",
];

const PRESERVED_SUFFIXES: &[&str] = &[
    ".bf16",
    ".bin",
    ".blob",
    ".ckpt",
    ".f16",
    ".f32",
    ".fp16",
    ".fp32",
    ".gguf",
    ".i4",
    ".i8",
    ".mlmodelc",
    ".mlmodel",
    ".mlpackage",
    ".mxfp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".s4",
    ".safetensors",
    ".sc",
    ".sc4",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Program {
    Git,
    Cargo,
}

const PROFILE_ENVIRONMENT: [&str; 9] = [
    "DELTAFIN_TORCH_ROOT",
    "LIBTORCH",
    "DELTAFIN_CUDA_MOE",
    "DELTAFIN_CUDA_ARCHITECTURES",
    "CUDACXX",
    "CMAKE_CUDA_COMPILER",
    "CUDAToolkit_ROOT",
    "CUDA_HOME",
    "CUDA_PATH",
];

const ISOLATED_CARGO_HOME: &str = ".deltafin/native-upgrade-cargo-home";

// These variables carry data rather than executable hooks. Everything else is
// removed from the child environment before Git or Cargo starts. In
// particular, this excludes Git/Cargo/Rust compiler wrappers, dynamic-loader
// injection, compiler-driver overrides, MAKEFLAGS, and Python environment
// state. PATH is retained because Cargo build scripts need the platform's
// native compiler tools; the actual git, cargo, rustc, and HTTPS Git helper are
// independently resolved and checked as native binaries below.
const SAFE_ENVIRONMENT_PASSTHROUGH: &[&str] = &[
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NO_COLOR",
    "PATH",
    // Rustup stores installed native toolchains here. Preserve its location,
    // but deliberately do not preserve RUSTUP_TOOLCHAIN or auto-download
    // controls that could select a different compiler for the upgrade.
    "RUSTUP_HOME",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
];

const GIT_CONFIGURATION: &[&str] = &[
    "core.hooksPath=/dev/null",
    "core.fsmonitor=false",
    "core.untrackedCache=false",
    "core.attributesFile=/dev/null",
    "credential.helper=",
    "credential.interactive=false",
    "protocol.file.allow=never",
    "protocol.ext.allow=never",
    "protocol.ssh.allow=never",
    "protocol.git.allow=never",
    "protocol.http.allow=never",
    "protocol.https.allow=always",
    "fetch.recurseSubmodules=false",
    "maintenance.auto=false",
    "gc.auto=0",
];

const CARGO_CONFIGURATION: &[&str] = &[
    "net.git-fetch-with-cli=false",
    "registry.global-credential-providers=[\"cargo:token\"]",
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct BuildProfile {
    explicit_torch_root: Option<PathBuf>,
    cuda_provider: bool,
    cuda_moe: &'static str,
    overrides: Vec<(&'static str, OsString)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CommandEnvironment {
    clear: bool,
    remove: Vec<OsString>,
    set: Vec<(OsString, OsString)>,
}

impl CommandEnvironment {
    fn isolated(program: Program, repo_root: &Path) -> Result<Self> {
        let mut environment = Self {
            clear: true,
            remove: Vec::new(),
            set: SAFE_ENVIRONMENT_PASSTHROUGH
                .iter()
                .filter_map(|name| {
                    std::env::var_os(name).map(|value| (OsString::from(name), value))
                })
                .collect(),
        };
        match program {
            Program::Git => {
                environment.set.extend([
                    (OsString::from("GIT_CONFIG_NOSYSTEM"), OsString::from("1")),
                    (
                        OsString::from("GIT_CONFIG_GLOBAL"),
                        OsString::from("/dev/null"),
                    ),
                    (OsString::from("GIT_ATTR_NOSYSTEM"), OsString::from("1")),
                    (OsString::from("GIT_TERMINAL_PROMPT"), OsString::from("0")),
                    (
                        OsString::from("GIT_ALLOW_PROTOCOL"),
                        OsString::from("https"),
                    ),
                    (OsString::from("GIT_OPTIONAL_LOCKS"), OsString::from("0")),
                ]);
            }
            Program::Cargo => {
                let cargo_home = require_isolated_cargo_home(repo_root)?;
                environment
                    .set
                    .push((OsString::from("CARGO_HOME"), cargo_home.into_os_string()));
                environment
                    .set
                    .push((OsString::from("RUSTUP_AUTO_INSTALL"), OsString::from("0")));
            }
        }
        Ok(environment)
    }

    fn overlay(&mut self, other: &Self) {
        self.clear |= other.clear;
        for name in &other.remove {
            self.set.retain(|(candidate, _)| candidate != name);
            if !self.remove.contains(name) {
                self.remove.push(name.clone());
            }
        }
        for (name, value) in &other.set {
            self.remove.retain(|candidate| candidate != name);
            self.set.retain(|(candidate, _)| candidate != name);
            self.set.push((name.clone(), value.clone()));
        }
    }
}

impl BuildProfile {
    #[cfg(feature = "runtime")]
    fn embedded() -> Result<Self> {
        Self::parse(
            option_env!("DELTAFIN_BUILD_PROFILE_FORMAT"),
            option_env!("DELTAFIN_BUILD_TORCH_SOURCE"),
            option_env!("DELTAFIN_BUILD_TORCH_ROOT"),
            option_env!("DELTAFIN_BUILD_CUDA_PROVIDER"),
            option_env!("DELTAFIN_BUILD_CUDA_MOE"),
            [
                (
                    "DELTAFIN_CUDA_ARCHITECTURES",
                    option_env!("DELTAFIN_BUILD_CUDA_ARCHITECTURES"),
                ),
                ("CUDACXX", option_env!("DELTAFIN_BUILD_CUDACXX")),
                (
                    "CMAKE_CUDA_COMPILER",
                    option_env!("DELTAFIN_BUILD_CMAKE_CUDA_COMPILER"),
                ),
                (
                    "CUDAToolkit_ROOT",
                    option_env!("DELTAFIN_BUILD_CUDA_TOOLKIT_ROOT"),
                ),
                ("CUDA_HOME", option_env!("DELTAFIN_BUILD_CUDA_HOME")),
                ("CUDA_PATH", option_env!("DELTAFIN_BUILD_CUDA_PATH")),
            ],
        )
    }

    fn parse(
        format: Option<&str>,
        torch_source: Option<&str>,
        torch_root: Option<&str>,
        cuda_provider: Option<&str>,
        cuda_moe: Option<&str>,
        overrides: [(&'static str, Option<&str>); 6],
    ) -> Result<Self> {
        let profile_version = match format {
            Some("v1") => 1,
            Some("v2") => 2,
            _ => {
                return Err(DeltafinError::new(
                    "running Deltafin binary lacks the versioned native build profile required for a safe upgrade; rebuild once with `cargo build --locked --release`",
                ));
            }
        };
        let decoded_root = decode_profile_value(
            torch_root
                .ok_or_else(|| DeltafinError::new("native build profile omits Torch root"))?,
            "Torch root",
        )?;
        let explicit_torch_root = match (torch_source, decoded_root) {
            (Some("bootstrap"), None) => None,
            (Some("explicit"), Some(value)) => {
                let path = PathBuf::from(value);
                if !path.is_absolute() {
                    return Err(DeltafinError::new(
                        "native build profile contains a non-absolute explicit Torch root",
                    ));
                }
                Some(path)
            }
            _ => {
                return Err(DeltafinError::new(
                    "native build profile has an inconsistent Torch source/root pair",
                ));
            }
        };
        let cuda_moe = match cuda_moe {
            Some("ON") => "ON",
            Some("OFF") => "OFF",
            _ => {
                return Err(DeltafinError::new(
                    "native build profile has an invalid CUDA-MoE mode",
                ));
            }
        };
        let cuda_provider = match (profile_version, cuda_provider) {
            (1, None) => cuda_moe == "ON",
            (2, Some("ON")) => true,
            (2, Some("OFF")) => false,
            _ => {
                return Err(DeltafinError::new(
                    "native build profile has an invalid CUDA-provider mode",
                ));
            }
        };
        if cuda_moe == "ON" && !cuda_provider {
            return Err(DeltafinError::new(
                "native build profile requests CUDA MoE without the CUDA provider",
            ));
        }
        if cuda_provider && explicit_torch_root.is_none() {
            return Err(DeltafinError::new(
                "native build profile requests CUDA without an explicit CUDA LibTorch root",
            ));
        }
        let mut decoded_overrides = Vec::new();
        for (name, encoded) in overrides {
            let encoded = encoded
                .ok_or_else(|| DeltafinError::new(format!("native build profile omits {name}")))?;
            if let Some(value) = decode_profile_value(encoded, name)? {
                decoded_overrides.push((name, value));
            }
        }
        if cuda_moe == "ON" {
            let required = |name| {
                decoded_overrides
                    .iter()
                    .find_map(|(candidate, value)| (*candidate == name).then_some(value))
                    .ok_or_else(|| {
                        DeltafinError::new(format!(
                            "CUDA-enabled native build profile omits effective {name}"
                        ))
                    })
            };
            let architectures = required("DELTAFIN_CUDA_ARCHITECTURES")?;
            let architectures = architectures.to_str().ok_or_else(|| {
                DeltafinError::new("CUDA build-profile architectures are not UTF-8")
            })?;
            if architectures.is_empty()
                || !architectures
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b';')
            {
                return Err(DeltafinError::new(
                    "CUDA build-profile architectures are malformed",
                ));
            }
            let cudacxx = required("CUDACXX")?;
            let cmake_compiler = required("CMAKE_CUDA_COMPILER")?;
            if cudacxx != cmake_compiler || !Path::new(cmake_compiler).is_absolute() {
                return Err(DeltafinError::new(
                    "CUDA build profile must retain one canonical absolute NVCC path",
                ));
            }
        } else {
            if decoded_overrides.iter().any(|(name, _)| {
                matches!(
                    *name,
                    "DELTAFIN_CUDA_ARCHITECTURES" | "CUDACXX" | "CMAKE_CUDA_COMPILER"
                )
            }) {
                return Err(DeltafinError::new(
                    "CUDA-provider-only build profile unexpectedly contains NVCC/MXFP4 overrides",
                ));
            }
            if !cuda_provider && !decoded_overrides.is_empty() {
                return Err(DeltafinError::new(
                    "CUDA-disabled native build profile unexpectedly contains CUDA overrides",
                ));
            }
        }
        Ok(Self {
            explicit_torch_root,
            cuda_provider,
            cuda_moe,
            overrides: decoded_overrides,
        })
    }

    fn command_environment(&self) -> CommandEnvironment {
        let mut set = vec![(
            OsString::from("DELTAFIN_CUDA_MOE"),
            OsString::from(self.cuda_moe),
        )];
        if let Some(root) = &self.explicit_torch_root {
            set.push((
                OsString::from("DELTAFIN_TORCH_ROOT"),
                root.as_os_str().to_owned(),
            ));
        }
        set.extend(
            self.overrides
                .iter()
                .map(|(name, value)| (OsString::from(name), value.clone())),
        );
        CommandEnvironment {
            clear: false,
            remove: PROFILE_ENVIRONMENT
                .into_iter()
                .map(OsString::from)
                .collect(),
            set,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Torch={} CUDA-provider={} CUDA-MoE={} explicit-CUDA-overrides={}",
            if self.explicit_torch_root.is_some() {
                "operator-supplied"
            } else {
                "authenticated automatic bootstrap"
            },
            if self.cuda_provider { "on" } else { "off" },
            self.cuda_moe.to_ascii_lowercase(),
            self.overrides.len(),
        )
    }

    fn loader_audit_policy(&self) -> Result<crate::loader_audit::LoaderAuditPolicy> {
        let Some(torch_root) = &self.explicit_torch_root else {
            return Ok(crate::loader_audit::LoaderAuditPolicy::bootstrap());
        };
        let mut roots = vec![torch_root.clone()];
        for (name, value) in &self.overrides {
            let path = PathBuf::from(value);
            match *name {
                "CUDAToolkit_ROOT" | "CUDA_HOME" | "CUDA_PATH" => roots.push(path),
                "CUDACXX" | "CMAKE_CUDA_COMPILER" => {
                    let toolkit = path
                        .parent()
                        .and_then(Path::parent)
                        .ok_or_else(|| {
                            DeltafinError::new(format!(
                                "recorded CUDA compiler path does not identify a toolkit root: {}",
                                path.display()
                            ))
                        })?
                        .to_path_buf();
                    roots.push(toolkit);
                }
                _ => {}
            }
        }
        roots.sort();
        roots.dedup();
        Ok(crate::loader_audit::LoaderAuditPolicy::operator_supplied(
            roots,
        ))
    }
}

fn decode_profile_value(encoded: &str, label: &str) -> Result<Option<OsString>> {
    if encoded == "-" {
        return Ok(None);
    }
    let hexadecimal = encoded.strip_prefix("hex:").ok_or_else(|| {
        DeltafinError::new(format!("native build profile {label} lacks hex encoding"))
    })?;
    if hexadecimal.len() % 2 != 0 || hexadecimal.len() > 32_768 {
        return Err(DeltafinError::new(format!(
            "native build profile {label} has an invalid bounded length"
        )));
    }
    let mut bytes = Vec::with_capacity(hexadecimal.len() / 2);
    for pair in hexadecimal.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0]).ok_or_else(|| {
            DeltafinError::new(format!("native build profile {label} is not lowercase hex"))
        })?;
        let low = hex_nibble(pair[1]).ok_or_else(|| {
            DeltafinError::new(format!("native build profile {label} is not lowercase hex"))
        })?;
        bytes.push((high << 4) | low);
    }
    if bytes.contains(&0) {
        return Err(DeltafinError::new(format!(
            "native build profile {label} contains a NUL byte"
        )));
    }
    Ok(Some(os_string_from_bytes(bytes)))
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

impl Program {
    fn executable(self) -> &'static str {
        match (self, cfg!(windows)) {
            (Self::Git, false) => "git",
            (Self::Git, true) => "git.exe",
            (Self::Cargo, false) => "cargo",
            (Self::Cargo, true) => "cargo.exe",
        }
    }
}

#[derive(Debug)]
struct CommandOutput {
    code: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

trait Runner {
    fn run(
        &self,
        program: Program,
        arguments: &[OsString],
        cwd: &Path,
        environment: &CommandEnvironment,
    ) -> Result<CommandOutput>;
}

#[cfg(feature = "runtime")]
struct ProcessRunner;

#[cfg(feature = "runtime")]
impl Runner for ProcessRunner {
    fn run(
        &self,
        program: Program,
        arguments: &[OsString],
        cwd: &Path,
        environment: &CommandEnvironment,
    ) -> Result<CommandOutput> {
        let executable = resolve_native_program(program.executable())?;
        let mut command = Command::new(&executable);
        command
            .args(arguments)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if environment.clear {
            command.env_clear();
        }
        for name in &environment.remove {
            command.env_remove(name);
        }
        for (name, value) in &environment.set {
            command.env(name, value);
        }
        if program == Program::Cargo {
            // Do not let Cargo rediscover a RUSTC command or wrapper through
            // PATH/config after the environment was isolated. Preserve the
            // unresolved path so rustup-style native multicall proxies retain
            // argv[0]="rustc" while their final target is still inspected.
            command.env("RUSTC", resolve_native_program("rustc")?);
        }
        let output = command.output().map_err(|error| {
            DeltafinError::new(format!(
                "could not start native {} at {}: {error}",
                program.executable(),
                executable.display(),
            ))
        })?;
        Ok(CommandOutput {
            code: output.status.code().unwrap_or(-1),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}

/// Resolve updater tools without admitting a shebang script or interpreter
/// shim. `Command` deliberately does not invoke a shell, but Unix would still
/// honor a script's `#!` line if an injected PATH entry named itself `git` or
/// `cargo`. The public native upgrade path permits only compiled ELF/Mach-O
/// executables; ordinary symlinks such as a Cargo rustup proxy are resolved to
/// and checked at their final target.
#[cfg(feature = "runtime")]
fn resolve_native_program(name: &str) -> Result<PathBuf> {
    let path = std::env::var_os("PATH")
        .ok_or_else(|| DeltafinError::new(format!("PATH is unset; cannot locate native {name}")))?;
    resolve_native_program_in_path(name, &path)
}

#[cfg(any(feature = "runtime", test))]
fn resolve_native_program_in_path(name: &str, path: &OsStr) -> Result<PathBuf> {
    let mut rejected = Vec::new();
    let resolution_root = std::env::current_dir().map_err(|error| {
        DeltafinError::new(format!(
            "cannot resolve relative PATH entries while locating native {name}: {error}"
        ))
    })?;
    for directory in std::env::split_paths(path) {
        let directory = if directory.is_absolute() {
            directory
        } else {
            resolution_root.join(directory)
        };
        let candidate = directory.join(name);
        let unresolved = match fs::symlink_metadata(&candidate) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                rejected.push(format!("{} ({error})", candidate.display()));
                continue;
            }
        };
        if !unresolved.file_type().is_file() && !unresolved.file_type().is_symlink() {
            rejected.push(format!("{} (not a file)", candidate.display()));
            continue;
        }
        // Windows has no uid; the executable-format check below is its gate.
        #[cfg(unix)]
        {
            // SAFETY: geteuid has no arguments, pointers, or memory preconditions.
            let effective_user = unsafe { libc::geteuid() };
            let owner = unresolved.uid();
            if owner != 0 && owner != effective_user {
                rejected.push(format!(
                    "{} (symlink/file is owned by uid {owner}, not root or the current uid {effective_user})",
                    candidate.display()
                ));
                continue;
            }
        }
        match inspect_native_executable(&candidate) {
            Ok(_) => {
                // Returning `candidate` rather than its canonical target is
                // intentional: rustup's cargo/rustc proxies are native
                // multicall binaries selected by the symlink's argv[0].
                return Ok(candidate);
            }
            Err(error) => {
                rejected.push(format!("{} ({error})", candidate.display()));
                continue;
            }
        }
    }
    let detail = if rejected.is_empty() {
        String::new()
    } else {
        format!("; rejected: {}", rejected.join(", "))
    };
    Err(DeltafinError::new(format!(
        "could not locate compiled native {name} on PATH{detail}"
    )))
}

#[cfg(any(feature = "runtime", test))]
fn inspect_native_executable(candidate: &Path) -> Result<PathBuf> {
    let canonical = crate::sys::fs::canonicalize(candidate).map_err(|error| {
        DeltafinError::new(format!("cannot resolve native executable: {error}"))
    })?;
    let metadata = fs::metadata(&canonical).map_err(|error| {
        DeltafinError::new(format!("cannot inspect {}: {error}", canonical.display()))
    })?;
    #[cfg(unix)]
    {
        if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
            return Err(DeltafinError::new(format!(
                "{} is not an executable file",
                canonical.display()
            )));
        }
        // SAFETY: geteuid has no arguments, pointers, or memory preconditions.
        let effective_user = unsafe { libc::geteuid() };
        let owner = metadata.uid();
        if owner != 0 && owner != effective_user {
            return Err(DeltafinError::new(format!(
                "{} is owned by uid {}, not root or the current uid {effective_user}",
                canonical.display(),
                owner
            )));
        }
        if metadata.mode() & 0o022 != 0 {
            return Err(DeltafinError::new(format!(
                "{} is group/world-writable and cannot be trusted as a native tool",
                canonical.display()
            )));
        }
    }
    #[cfg(windows)]
    {
        // Windows has no exec bit or uid. A program is a `.exe` file whose
        // headers say it is one; the format check below rejects `.cmd`/`.bat`
        // shims, scripts renamed to `.exe` and App Execution Alias stubs.
        let is_exe = canonical
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"));
        if !metadata.is_file() || !is_exe {
            return Err(DeltafinError::new(format!(
                "{} is not an executable file",
                canonical.display()
            )));
        }
    }
    let mut file = fs::File::open(&canonical).map_err(|error| {
        DeltafinError::new(format!("cannot open {}: {error}", canonical.display()))
    })?;
    let mut magic = [0u8; NATIVE_HEADER_BYTES];
    let count = file.read(&mut magic).map_err(|error| {
        DeltafinError::new(format!("cannot read {}: {error}", canonical.display()))
    })?;
    if !native_executable_magic(&magic[..count]) {
        return Err(DeltafinError::new(format!(
            "{} is a script or interpreter shim; only {} executables are permitted",
            canonical.display(),
            if cfg!(windows) { "PE" } else { "ELF/Mach-O" },
        )));
    }
    Ok(canonical)
}

/// The leading bytes needed to recognize a native executable: a four-byte
/// magic on Unix, enough of the headers to follow `e_lfanew` on Windows.
#[cfg(any(feature = "runtime", test))]
const NATIVE_HEADER_BYTES: usize = if cfg!(windows) { 4096 } else { 4 };

#[cfg(any(feature = "runtime", test))]
fn native_executable_magic(bytes: &[u8]) -> bool {
    #[cfg(windows)]
    {
        crate::sys::pe::is_native_executable(bytes)
    }
    #[cfg(not(windows))]
    {
        matches!(
            bytes,
            [0x7f, b'E', b'L', b'F']
                | [0xcf, 0xfa, 0xed, 0xfe]
                | [0xfe, 0xed, 0xfa, 0xcf]
                | [0xca, 0xfe, 0xba, 0xbe]
                | [0xbe, 0xba, 0xfe, 0xca]
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Relation {
    Equal,
    Behind,
    Ahead,
    Diverged,
}

#[derive(Debug)]
struct GitState {
    branch: String,
    upstream: String,
    upstream_head: String,
    relation: Relation,
}

#[cfg(feature = "runtime")]
pub fn run() -> Result<()> {
    let executable = std::env::current_exe()
        .map_err(|error| DeltafinError::new(format!("locate running Deltafin binary: {error}")))?;
    let current_directory = std::env::current_dir().map_err(|error| {
        DeltafinError::new(format!(
            "locate current directory for native upgrade: {error}"
        ))
    })?;
    let repository = locate_repository_from(&executable, &current_directory)?;
    // The reproducible profile carries NVCC-shaped variables and validates its
    // architecture list as numeric compute capabilities, so it cannot describe
    // a HIPCC build. Reproducing this binary would therefore silently drop the
    // device kernels back to the exact CPU MXFP4 path. Refuse instead, and say
    // exactly which command rebuilds the same thing.
    if option_env!("DELTAFIN_GPU_KERNEL_RUNTIME") == Some("HIP") {
        return Err(DeltafinError::new(
            "this binary's device kernels were compiled by HIPCC for ROCm, which the reproducible build profile does not describe; `deltafin upgrade` will not rebuild it as a different configuration. Update the checkout and rebuild on the ROCm host with DELTAFIN_CUDA_MOE=ON",
        ));
    }
    let profile = BuildProfile::embedded()?;
    run_with_profile(&repository, &ProcessRunner, &profile, &mut |message| {
        println!("{message}")
    })
}

/// Re-run the same bounded loader audit used for a freshly rebuilt upgrade on
/// the executable that is serving this process. `build.rs` cannot inspect the
/// final Rust link product (Cargo runs it before that artifact exists), so
/// `doctor` owns the installed-binary check while `upgrade` checks the newly
/// reported Cargo artifact before declaring success.
#[cfg(feature = "runtime")]
pub(crate) fn audit_running_artifact() -> Result<PathBuf> {
    let executable = std::env::current_exe()
        .map_err(|error| DeltafinError::new(format!("locate running Deltafin binary: {error}")))?;
    let profile = BuildProfile::embedded()?;
    verify_no_python_environment_path_with_profile(&executable, &profile)?;
    Ok(executable)
}

fn locate_repository_from(executable: &Path, current_directory: &Path) -> Result<PathBuf> {
    // A normal repository build lives at target/{debug,release}/deltafin.
    // Search that executable's bounded ancestor chain first, then support a
    // packaged binary deliberately invoked from the checkout root. Never use
    // CARGO_MANIFEST_DIR: embedding the builder's absolute source path would
    // make a distributed binary update the wrong machine-specific checkout.
    for candidate in executable
        .ancestors()
        .take(8)
        .chain(std::iter::once(current_directory))
    {
        let Ok(candidate) = crate::sys::fs::canonicalize(candidate) else {
            continue;
        };
        if candidate.join(".git").exists()
            && candidate.join("Cargo.toml").is_file()
            && candidate.join("Cargo.lock").is_file()
            && candidate.join("native/deltafin/Cargo.toml").is_file()
        {
            return Ok(candidate);
        }
    }
    Err(DeltafinError::new(
        "could not locate the Deltafin Git checkout from the running binary or current directory; run `deltafin upgrade` from the repository root",
    ))
}

fn run_with_profile<R: Runner>(
    repo_root: &Path,
    runner: &R,
    profile: &BuildProfile,
    printer: &mut dyn FnMut(&str),
) -> Result<()> {
    let repo_root = crate::sys::fs::canonicalize(repo_root).map_err(|error| {
        DeltafinError::new(format!(
            "resolve repository root {}: {error}",
            repo_root.display()
        ))
    })?;

    require_project_files(&repo_root)?;
    preflight_cargo(runner, &repo_root)?;
    printer("Deltafin safe native upgrade");
    printer(&format!("  repository: {}", repo_root.display()));
    printer("  model weights and caches are not touched");
    printer(&format!("  preserved build profile: {}", profile.summary()));

    let state = fetch_state(runner, &repo_root)?;
    match state.relation {
        Relation::Behind => {
            printer(&format!(
                "Fast-forwarding {} to {}...",
                state.branch,
                abbreviated(&state.upstream_head)
            ));
            apply_fast_forward(runner, &repo_root, &state)?;
            // The new revision owns both manifests. Validate it before asking
            // Cargo to compile anything from it.
            require_project_files(&repo_root)?;
            preflight_cargo(runner, &repo_root)?;
        }
        Relation::Ahead => printer(&format!(
            "{} is clean and locally ahead of {}; leaving commits unchanged.",
            state.branch, state.upstream
        )),
        Relation::Equal => printer(&format!(
            "{} is already current with {}.",
            state.branch, state.upstream
        )),
        Relation::Diverged => {
            return Err(DeltafinError::new(
                "internal upgrade state remained diverged; refusing to rebuild",
            ));
        }
    }

    printer(
        "Validating or installing the target revision's authenticated native runtime during the locked build...",
    );
    let artifact = rebuild_native(runner, &repo_root, profile)?;
    printer(&format!(
        "Upgrade complete: {}. Downloaded models and caches remain in place.",
        artifact.display()
    ));
    Ok(())
}

#[cfg(test)]
fn run_with<R: Runner>(repo_root: &Path, runner: &R, printer: &mut dyn FnMut(&str)) -> Result<()> {
    let profile = BuildProfile {
        explicit_torch_root: None,
        cuda_provider: false,
        cuda_moe: "OFF",
        overrides: Vec::new(),
    };
    run_with_profile(repo_root, runner, &profile, printer)
}

fn require_project_files(repo_root: &Path) -> Result<()> {
    let required = [
        repo_root.join("Cargo.toml"),
        repo_root.join("Cargo.lock"),
        repo_root.join("native/deltafin/Cargo.toml"),
        repo_root.join("native/deltafin-bootstrap/Cargo.toml"),
    ];
    let missing: Vec<_> = required
        .iter()
        .filter(|path| !path.is_file())
        .map(|path| {
            path.strip_prefix(repo_root)
                .unwrap_or(path)
                .display()
                .to_string()
        })
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(DeltafinError::new(format!(
            "the checkout is missing required native upgrade files: {}",
            missing.join(", ")
        )))
    }
}

fn preflight_cargo<R: Runner>(runner: &R, repo_root: &Path) -> Result<()> {
    require_no_ambient_toolchain_configuration(repo_root)?;
    checked(
        runner,
        Program::Cargo,
        strings(["--version"]),
        repo_root,
        "checking Cargo",
        &[0],
    )?;
    let output = checked(
        runner,
        Program::Cargo,
        strings(["metadata", "--locked", "--no-deps", "--format-version", "1"]),
        repo_root,
        "validating the locked Cargo workspace",
        &[0],
    )?;
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|error| {
        DeltafinError::new(format!(
            "Cargo returned invalid workspace metadata: {error}"
        ))
    })?;
    let workspace = metadata
        .get("workspace_root")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| DeltafinError::new("Cargo metadata omitted workspace_root"))?;
    let workspace = crate::sys::fs::canonicalize(workspace).map_err(|error| {
        DeltafinError::new(format!("resolve Cargo workspace root {workspace}: {error}"))
    })?;
    if workspace != repo_root {
        return Err(DeltafinError::new(format!(
            "compiled updater belongs to {}, but Cargo reported {}; refusing to build the wrong checkout",
            repo_root.display(),
            workspace.display()
        )));
    }

    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| DeltafinError::new("Cargo metadata omitted the workspace packages"))?;
    for (name, relative_manifest) in [
        ("deltafin", "native/deltafin/Cargo.toml"),
        ("deltafin-bootstrap", "native/deltafin-bootstrap/Cargo.toml"),
    ] {
        let expected_manifest = crate::sys::fs::canonicalize(repo_root.join(relative_manifest))
            .map_err(|error| DeltafinError::new(format!("resolve {name} manifest: {error}")))?;
        let package_matches = packages.iter().any(|package| {
            package.get("name").and_then(serde_json::Value::as_str) == Some(name)
                && package
                    .get("manifest_path")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|path| crate::sys::fs::canonicalize(path).ok())
                    .as_ref()
                    == Some(&expected_manifest)
        });
        if !package_matches {
            return Err(DeltafinError::new(format!(
                "the locked Cargo workspace does not contain {relative_manifest} as package {name}"
            )));
        }
    }
    Ok(())
}

fn require_no_ambient_toolchain_configuration(repo_root: &Path) -> Result<()> {
    let mut discovered = Vec::new();
    for ancestor in repo_root.ancestors() {
        for relative in [
            ".cargo/config",
            ".cargo/config.toml",
            "rust-toolchain",
            "rust-toolchain.toml",
        ] {
            let path = ancestor.join(relative);
            if fs::symlink_metadata(&path).is_ok() {
                discovered.push(path);
            }
        }
    }
    if discovered.is_empty() {
        return Ok(());
    }
    Err(DeltafinError::new(format!(
        "the native upgrader refuses hierarchical Cargo/rustup configuration because it can select compiler wrappers, linkers, credential processes, or an unreviewed toolchain:\n{}\nRemove or temporarily move these files, then run the upgrade again",
        discovered
            .iter()
            .map(|path| format!("  {}", path.display()))
            .collect::<Vec<_>>()
            .join("\n")
    )))
}

fn require_isolated_cargo_home(repo_root: &Path) -> Result<PathBuf> {
    let private_root = repo_root.join(".deltafin");
    let cargo_home = repo_root.join(ISOLATED_CARGO_HOME);
    for directory in [&private_root, &cargo_home] {
        match fs::symlink_metadata(directory) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(DeltafinError::new(format!(
                        "native-upgrade Cargo state must be a real directory, not a symlink or file: {}",
                        directory.display()
                    )));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(directory).map_err(|error| {
                    DeltafinError::new(format!(
                        "create isolated native-upgrade Cargo directory {}: {error}",
                        directory.display()
                    ))
                })?;
                crate::sys::fs::restrict_to_owner(directory).map_err(|error| {
                    DeltafinError::new(format!(
                        "protect isolated native-upgrade Cargo directory {}: {error}",
                        directory.display()
                    ))
                })?;
            }
            Err(error) => {
                return Err(DeltafinError::new(format!(
                    "inspect isolated native-upgrade Cargo directory {}: {error}",
                    directory.display()
                )));
            }
        }
    }
    crate::sys::fs::restrict_to_owner(&cargo_home).map_err(|error| {
        DeltafinError::new(format!(
            "protect isolated native-upgrade Cargo home {}: {error}",
            cargo_home.display()
        ))
    })?;
    let cargo_home = crate::sys::fs::canonicalize(&cargo_home).map_err(|error| {
        DeltafinError::new(format!(
            "resolve isolated native-upgrade Cargo directory {}: {error}",
            cargo_home.display()
        ))
    })?;
    if !cargo_home.starts_with(repo_root) {
        return Err(DeltafinError::new(format!(
            "isolated native-upgrade Cargo directory escaped the repository: {}",
            cargo_home.display()
        )));
    }
    for name in ["config", "config.toml"] {
        let path = cargo_home.join(name);
        if fs::symlink_metadata(&path).is_ok() {
            return Err(DeltafinError::new(format!(
                "isolated native-upgrade Cargo home unexpectedly contains executable configuration: {}",
                path.display()
            )));
        }
    }
    Ok(cargo_home)
}

fn fetch_state<R: Runner>(runner: &R, repo_root: &Path) -> Result<GitState> {
    let top_level = git_text(
        runner,
        repo_root,
        &["rev-parse", "--show-toplevel"],
        "locating the Git checkout",
    )?;
    let top_level = crate::sys::fs::canonicalize(&top_level).map_err(|error| {
        DeltafinError::new(format!("resolve Git checkout root {top_level}: {error}"))
    })?;
    if top_level != repo_root {
        return Err(DeltafinError::new(format!(
            "compiled updater belongs to {}, but Git reported {}; refusing to update the wrong checkout",
            repo_root.display(),
            top_level.display()
        )));
    }

    require_clean_worktree(runner, repo_root)?;
    require_no_executable_local_git_configuration(runner, repo_root)?;
    let branch_result = git_allowed(
        runner,
        repo_root,
        &["symbolic-ref", "--quiet", "--short", "HEAD"],
        "reading the current branch",
        &[0, 1],
    )?;
    let branch = utf8_trimmed(&branch_result.stdout, "current branch")?;
    if branch_result.code != 0 || branch.is_empty() {
        return Err(DeltafinError::new(
            "the checkout is in detached-HEAD state. Switch to a normal branch with an upstream before upgrading",
        ));
    }

    let upstream_result = git_allowed(
        runner,
        repo_root,
        &[
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ],
        "reading the branch upstream",
        &[0, 1, 128],
    )?;
    let upstream = utf8_trimmed(&upstream_result.stdout, "branch upstream")?;
    if upstream_result.code != 0 || upstream.is_empty() {
        return Err(DeltafinError::new(format!(
            "branch {branch:?} has no upstream. Configure one explicitly before upgrading"
        )));
    }

    let remote = git_text(
        runner,
        repo_root,
        &["config", "--get", &format!("branch.{branch}.remote")],
        "reading the upstream remote",
    )?;
    if remote.is_empty() || remote == "." {
        return Err(DeltafinError::new(format!(
            "branch {branch:?} does not track a normal remote upstream"
        )));
    }

    require_native_https_remote(runner, repo_root, &remote)?;

    git(
        runner,
        repo_root,
        &[
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            "--",
            &remote,
        ],
        &format!("fetching {remote}"),
    )?;
    // Fetch should not alter the worktree. This also narrows the race before
    // the fast-forward without ever stashing or discarding user data.
    require_clean_worktree(runner, repo_root)?;

    let head = git_text(
        runner,
        repo_root,
        &["rev-parse", "--verify", "HEAD"],
        "reading the local revision",
    )?;
    let upstream_head = git_text(
        runner,
        repo_root,
        &["rev-parse", "--verify", "@{upstream}"],
        "reading the fetched upstream revision",
    )?;
    let relation = relationship(runner, repo_root, &head, &upstream_head)?;
    if relation == Relation::Diverged {
        return Err(DeltafinError::new(format!(
            "{branch:?} and {upstream:?} have diverged. The updater only permits fast-forwards and will not merge, rebase, or reset"
        )));
    }
    if relation == Relation::Behind {
        require_update_preserves_data(runner, repo_root, &head, &upstream_head)?;
    }
    Ok(GitState {
        branch,
        upstream,
        upstream_head,
        relation,
    })
}

fn require_no_executable_local_git_configuration<R: Runner>(
    runner: &R,
    repo_root: &Path,
) -> Result<()> {
    // Command-line overrides disable each standard ambient execution path,
    // but arbitrary filter/merge-driver names cannot be wildcard-overridden.
    // Reject those entries in the repository-local config before any fetch or
    // checkout. Includes are rejected too, so later Git invocations cannot
    // smuggle equivalent commands in from another file.
    let forbidden = concat!(
        "^(include\\.path|includeif\\..*\\.path|extensions\\.worktreeconfig|",
        "filter\\..*\\.(clean|smudge|process|required)|",
        "diff\\.external|diff\\..*\\.command|",
        "core\\.(askpass|fsmonitor|hookspath|sshcommand|gitproxy|alternaterefscommand)|",
        "credential(\\..*)?\\.helper|",
        "merge\\..*\\.(driver|recursive)|branch\\..*\\.mergeoptions)$"
    );
    let output = git_allowed(
        runner,
        repo_root,
        &[
            "config",
            "--local",
            "--no-includes",
            "--name-only",
            "--null",
            "--get-regexp",
            forbidden,
        ],
        "auditing repository-local Git process configuration",
        &[0, 1],
    )?;
    if output.code == 1 || output.stdout.is_empty() {
        return Ok(());
    }
    let names = std::str::from_utf8(&output.stdout).map_err(|error| {
        DeltafinError::new(format!(
            "repository-local Git configuration names are not UTF-8: {error}"
        ))
    })?;
    let names: Vec<_> = names
        .split('\0')
        .filter(|name| !name.is_empty())
        .take(20)
        .collect();
    Err(DeltafinError::new(format!(
        "repository-local Git configuration can execute hooks, filters, helpers, or wrappers; the native upgrader will not run it:\n{}",
        names
            .iter()
            .map(|name| format!("  {name}"))
            .collect::<Vec<_>>()
            .join("\n")
    )))
}

fn require_native_https_remote<R: Runner>(
    runner: &R,
    repo_root: &Path,
    remote: &str,
) -> Result<()> {
    if remote.starts_with('-')
        || !remote
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(DeltafinError::new(format!(
            "upstream remote name {remote:?} is not a bounded ordinary Git remote name"
        )));
    }
    let urls = git_text(
        runner,
        repo_root,
        &["remote", "get-url", "--all", remote],
        "resolving the upstream remote URL",
    )?;
    let urls: Vec<_> = urls.lines().filter(|url| !url.is_empty()).collect();
    if urls.is_empty()
        || urls.iter().any(|url| {
            !url.starts_with("https://")
                || url["https://".len()..].is_empty()
                || url
                    .bytes()
                    .any(|byte| byte.is_ascii_control() || byte == b'\\')
        })
    {
        return Err(DeltafinError::new(
            "the native upgrader only fetches explicit HTTPS remote URLs; SSH, local/file, git, HTTP, and external remote-helper transports are intentionally disabled",
        ));
    }

    let exec_path = git_text(
        runner,
        repo_root,
        &["--exec-path"],
        "locating Git's HTTPS transport helper",
    )?;
    if exec_path.is_empty() {
        return Err(DeltafinError::new(
            "Git did not report its native helper directory",
        ));
    }
    let exec_path = PathBuf::from(exec_path);
    if !exec_path.is_absolute() {
        return Err(DeltafinError::new(
            "Git reported a relative helper directory; refusing an ambiguous HTTPS transport",
        ));
    }
    let helper = format!("git-remote-https{}", std::env::consts::EXE_SUFFIX);
    inspect_native_executable(&exec_path.join(helper)).map_err(|error| {
        DeltafinError::new(format!(
            "Git's HTTPS transport helper is not a compiled native executable: {error}"
        ))
    })?;
    Ok(())
}

fn relationship<R: Runner>(
    runner: &R,
    repo_root: &Path,
    head: &str,
    upstream_head: &str,
) -> Result<Relation> {
    if head == upstream_head {
        return Ok(Relation::Equal);
    }
    let head_is_ancestor = git_allowed(
        runner,
        repo_root,
        &["merge-base", "--is-ancestor", head, upstream_head],
        "comparing the local branch with its upstream",
        &[0, 1],
    )?
    .code
        == 0;
    let upstream_is_ancestor = git_allowed(
        runner,
        repo_root,
        &["merge-base", "--is-ancestor", upstream_head, head],
        "comparing the upstream with the local branch",
        &[0, 1],
    )?
    .code
        == 0;
    Ok(match (head_is_ancestor, upstream_is_ancestor) {
        (true, false) => Relation::Behind,
        (false, true) => Relation::Ahead,
        _ => Relation::Diverged,
    })
}

fn require_update_preserves_data<R: Runner>(
    runner: &R,
    repo_root: &Path,
    head: &str,
    upstream_head: &str,
) -> Result<()> {
    let output = git_allowed(
        runner,
        repo_root,
        &[
            "diff",
            "--no-ext-diff",
            "--name-only",
            "-z",
            "--no-renames",
            head,
            upstream_head,
            "--",
        ],
        "checking the update for model/cache path collisions",
        &[0],
    )?;
    let paths = std::str::from_utf8(&output.stdout).map_err(|error| {
        DeltafinError::new(format!(
            "incoming Git paths are not valid UTF-8; refusing the update: {error}"
        ))
    })?;
    let mut collisions: Vec<_> = paths
        .split('\0')
        .filter(|path| !path.is_empty() && is_preserved_data_path(path))
        .collect();
    collisions.sort_unstable();
    if collisions.is_empty() {
        Ok(())
    } else {
        Err(DeltafinError::new(format!(
            "the incoming update touches paths reserved for downloaded models or caches, so it was not applied:\n{}",
            collisions
                .iter()
                .map(|path| format!("  {path}"))
                .collect::<Vec<_>>()
                .join("\n")
        )))
    }
}

fn apply_fast_forward<R: Runner>(runner: &R, repo_root: &Path, state: &GitState) -> Result<()> {
    git(
        runner,
        repo_root,
        &[
            "merge",
            "--ff-only",
            "--no-autostash",
            "--no-gpg-sign",
            "--no-overwrite-ignore",
            "--no-rerere-autoupdate",
            "--no-stat",
            "--no-verify",
            "--no-verify-signatures",
            &state.upstream_head,
        ],
        &format!("fast-forwarding {}", state.branch),
    )?;
    let observed = git_text(
        runner,
        repo_root,
        &["rev-parse", "--verify", "HEAD"],
        "verifying the fast-forward",
    )?;
    if observed != state.upstream_head {
        return Err(DeltafinError::new(
            "Git reported success but HEAD does not match the fetched upstream; the native rebuild was not attempted",
        ));
    }
    require_clean_worktree(runner, repo_root)
}

fn require_clean_worktree<R: Runner>(runner: &R, repo_root: &Path) -> Result<()> {
    let changes = git_text(
        runner,
        repo_root,
        &["status", "--porcelain=v1", "--untracked-files=normal"],
        "checking the Git worktree",
    )?;
    if changes.is_empty() {
        return Ok(());
    }
    let mut shown: Vec<_> = changes.lines().take(20).collect();
    let remaining = changes.lines().count().saturating_sub(shown.len());
    if remaining > 0 {
        shown.push("...");
    }
    Err(DeltafinError::new(format!(
        "the checkout has tracked, staged, or non-ignored untracked changes. The updater will not stash, discard, or overwrite them:\n{}{}",
        shown
            .iter()
            .map(|line| format!("  {line}"))
            .collect::<Vec<_>>()
            .join("\n"),
        if remaining > 0 {
            format!("\n  ... and {remaining} more")
        } else {
            String::new()
        }
    )))
}

fn rebuild_native<R: Runner>(
    runner: &R,
    repo_root: &Path,
    profile: &BuildProfile,
) -> Result<PathBuf> {
    let target_root = repo_root.join("target");
    require_safe_target_root(repo_root, &target_root)?;
    let arguments = vec![
        OsString::from("build"),
        OsString::from("--locked"),
        OsString::from("--release"),
        OsString::from("--package"),
        OsString::from("deltafin"),
        OsString::from("--bin"),
        OsString::from("deltafin"),
        OsString::from("--target-dir"),
        target_root.as_os_str().to_owned(),
        OsString::from("--message-format=json-render-diagnostics"),
    ];
    let output = checked_with_environment(
        runner,
        Program::Cargo,
        arguments,
        repo_root,
        "building the locked release Deltafin binary",
        &[0],
        &profile.command_environment(),
    )?;
    let artifact = cargo_binary_artifact(&output.stdout)?;
    verify_artifact_with_profile(repo_root, &target_root, &artifact, profile)?;
    Ok(artifact)
}

fn require_safe_target_root(repo_root: &Path, target_root: &Path) -> Result<()> {
    match fs::symlink_metadata(target_root) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(DeltafinError::new(format!(
                    "refusing to build through non-directory Cargo target path {}",
                    target_root.display()
                )));
            }
            let resolved = crate::sys::fs::canonicalize(target_root).map_err(|error| {
                DeltafinError::new(format!(
                    "resolve Cargo target directory {}: {error}",
                    target_root.display()
                ))
            })?;
            if !resolved.starts_with(repo_root) {
                return Err(DeltafinError::new(format!(
                    "refusing to build outside the repository through Cargo target directory {}",
                    resolved.display()
                )));
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(DeltafinError::new(format!(
            "inspect Cargo target directory {}: {error}",
            target_root.display()
        ))),
    }
}

fn cargo_binary_artifact(stdout: &[u8]) -> Result<PathBuf> {
    let text = std::str::from_utf8(stdout).map_err(|error| {
        DeltafinError::new(format!("Cargo build output was not valid UTF-8: {error}"))
    })?;
    let mut artifact = None;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let message: serde_json::Value = serde_json::from_str(line).map_err(|error| {
            DeltafinError::new(format!("Cargo returned invalid build metadata: {error}"))
        })?;
        let is_deltafin_binary = message.get("reason").and_then(serde_json::Value::as_str)
            == Some("compiler-artifact")
            && message
                .pointer("/target/name")
                .and_then(serde_json::Value::as_str)
                == Some("deltafin")
            && message
                .pointer("/target/kind")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
        if is_deltafin_binary {
            if let Some(path) = message
                .get("executable")
                .and_then(serde_json::Value::as_str)
            {
                artifact = Some(PathBuf::from(path));
            }
        }
    }
    artifact.ok_or_else(|| {
        DeltafinError::new(
            "Cargo reported success but did not identify the compiled deltafin executable",
        )
    })
}

#[cfg(test)]
fn verify_artifact(repo_root: &Path, target_root: &Path, artifact: &Path) -> Result<()> {
    verify_artifact_with_profile(
        repo_root,
        target_root,
        artifact,
        &BuildProfile {
            explicit_torch_root: None,
            cuda_provider: false,
            cuda_moe: "OFF",
            overrides: Vec::new(),
        },
    )
}

fn verify_artifact_with_profile(
    repo_root: &Path,
    target_root: &Path,
    artifact: &Path,
    profile: &BuildProfile,
) -> Result<()> {
    let unresolved_metadata = fs::symlink_metadata(artifact).map_err(|error| {
        DeltafinError::new(format!(
            "inspect Cargo's Deltafin artifact {}: {error}",
            artifact.display()
        ))
    })?;
    if unresolved_metadata.file_type().is_symlink() {
        return Err(DeltafinError::new(format!(
            "Cargo's Deltafin artifact must not be a symbolic link: {}",
            artifact.display()
        )));
    }
    let artifact = crate::sys::fs::canonicalize(artifact).map_err(|error| {
        DeltafinError::new(format!(
            "Cargo's Deltafin artifact {} cannot be resolved: {error}",
            artifact.display()
        ))
    })?;
    let target_root = crate::sys::fs::canonicalize(target_root).map_err(|error| {
        DeltafinError::new(format!(
            "Cargo target directory {} cannot be resolved: {error}",
            target_root.display()
        ))
    })?;
    if !artifact.starts_with(&target_root) || !target_root.starts_with(repo_root) {
        return Err(DeltafinError::new(format!(
            "Cargo reported an artifact outside the repository target directory: {}",
            artifact.display()
        )));
    }
    let expected_name = if cfg!(windows) {
        "deltafin.exe"
    } else {
        "deltafin"
    };
    if artifact.file_name() != Some(OsStr::new(expected_name)) {
        return Err(DeltafinError::new(format!(
            "Cargo reported an unexpected Deltafin artifact name: {}",
            artifact.display()
        )));
    }
    let metadata = fs::metadata(&artifact).map_err(|error| {
        DeltafinError::new(format!(
            "inspect compiled artifact {}: {error}",
            artifact.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(DeltafinError::new(format!(
            "Cargo's Deltafin artifact is not a non-empty regular file: {}",
            artifact.display()
        )));
    }
    verify_no_python_environment_path_with_profile(&artifact, profile)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(DeltafinError::new(format!(
                "Cargo's Deltafin artifact is not executable: {}",
                artifact.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg_attr(all(windows, feature = "runtime"), allow(dead_code))]
fn verify_no_python_environment_path(artifact: &Path) -> Result<()> {
    crate::loader_audit::audit_loader_closure(
        artifact,
        &crate::loader_audit::LoaderAuditPolicy::bootstrap(),
    )
    .map(|_| ())
}

fn verify_no_python_environment_path_with_profile(
    artifact: &Path,
    profile: &BuildProfile,
) -> Result<()> {
    let policy = profile.loader_audit_policy()?;
    crate::loader_audit::audit_loader_closure(artifact, &policy).map(|_| ())
}

fn is_preserved_data_path(raw_path: &str) -> bool {
    let mut parts = raw_path.split('/');
    let Some(root) = parts.next().filter(|part| !part.is_empty()) else {
        return false;
    };
    if PRESERVED_ROOTS.contains(&root)
        || root.starts_with("k3-resident")
        || root.starts_with("k3-draft")
        || root.starts_with("k3-cache")
        || root.starts_with("k3-experts")
        || root.starts_with("k3-model")
    {
        return true;
    }
    if matches!(raw_path, "tiktoken.model" | "tools/expert_index.meta.json") {
        return true;
    }
    let components: Vec<_> = raw_path.split('/').collect();
    if components.len() >= 3
        && components[0] == "tools"
        && components[1] == "k3pkg"
        && components
            .last()
            .is_some_and(|name| name.starts_with("modeling_") || name.starts_with("configuration_"))
    {
        return true;
    }
    let name = components
        .last()
        .copied()
        .unwrap_or(raw_path)
        .to_ascii_lowercase();
    name.match_indices('.').any(|(index, _)| {
        let suffix = &name[index..];
        PRESERVED_SUFFIXES.contains(&suffix)
            || suffix.strip_prefix('.').is_some_and(|value| {
                value.len() >= 2
                    && matches!(value.as_bytes()[0], b'i' | b's')
                    && value.as_bytes()[1..].iter().all(u8::is_ascii_digit)
            })
    })
}

fn git<R: Runner>(runner: &R, root: &Path, args: &[&str], action: &str) -> Result<CommandOutput> {
    git_allowed(runner, root, args, action, &[0])
}

fn git_allowed<R: Runner>(
    runner: &R,
    root: &Path,
    args: &[&str],
    action: &str,
    allowed: &[i32],
) -> Result<CommandOutput> {
    checked(
        runner,
        Program::Git,
        strings(args.iter().copied()),
        root,
        action,
        allowed,
    )
}

fn git_text<R: Runner>(runner: &R, root: &Path, args: &[&str], action: &str) -> Result<String> {
    let output = git(runner, root, args, action)?;
    utf8_trimmed(&output.stdout, action)
}

fn checked<R: Runner>(
    runner: &R,
    program: Program,
    arguments: Vec<OsString>,
    cwd: &Path,
    action: &str,
    allowed: &[i32],
) -> Result<CommandOutput> {
    checked_with_environment(
        runner,
        program,
        arguments,
        cwd,
        action,
        allowed,
        &CommandEnvironment::default(),
    )
}

fn checked_with_environment<R: Runner>(
    runner: &R,
    program: Program,
    arguments: Vec<OsString>,
    cwd: &Path,
    action: &str,
    allowed: &[i32],
    environment: &CommandEnvironment,
) -> Result<CommandOutput> {
    let arguments = hardened_arguments(program, arguments);
    let mut isolated = CommandEnvironment::isolated(program, cwd)?;
    isolated.overlay(environment);
    let output = runner.run(program, &arguments, cwd, &isolated)?;
    if allowed.contains(&output.code) {
        return Ok(output);
    }
    let diagnostics = if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    };
    let diagnostics = String::from_utf8_lossy(diagnostics);
    let diagnostics = if diagnostics.trim().is_empty() {
        "(the command produced no diagnostics)"
    } else {
        diagnostics.trim()
    };
    Err(DeltafinError::new(format!(
        "{action} failed with exit code {}:\n{diagnostics}",
        output.code
    )))
}

fn hardened_arguments(program: Program, arguments: Vec<OsString>) -> Vec<OsString> {
    let mut hardened = Vec::new();
    match program {
        Program::Git => {
            hardened.push(OsString::from("--no-pager"));
            for setting in GIT_CONFIGURATION {
                hardened.push(OsString::from("-c"));
                hardened.push(OsString::from(setting));
            }
        }
        Program::Cargo => {
            for setting in CARGO_CONFIGURATION {
                hardened.push(OsString::from("--config"));
                hardened.push(OsString::from(setting));
            }
        }
    }
    hardened.extend(arguments);
    hardened
}

fn strings<I, S>(values: I) -> Vec<OsString>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    values
        .into_iter()
        .map(|value| value.as_ref().to_owned())
        .collect()
}

fn utf8_trimmed(bytes: &[u8], description: &str) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::trim)
        .map(str::to_owned)
        .map_err(|error| {
            DeltafinError::new(format!(
                "{description} returned non-UTF-8 output; refusing to continue: {error}"
            ))
        })
}

fn abbreviated(revision: &str) -> &str {
    revision.get(..12).unwrap_or(revision)
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    /// `deltafin` and `git-remote-https` as this host names a program.
    const EXE: &str = std::env::consts::EXE_SUFFIX;

    /// A program image this host's audit accepts and that loads nothing it
    /// could not resolve: Mach-O with a harmless loader path on Unix, a PE that
    /// imports only `kernel32.dll` on Windows.
    fn native_fixture() -> Vec<u8> {
        #[cfg(windows)]
        {
            pe_fixture(&["kernel32.dll"])
        }
        #[cfg(not(windows))]
        {
            macho_with_loader_path(b"@loader_path/native")
        }
    }

    fn pe_fixture(imports: &[&str]) -> Vec<u8> {
        deltafin_sys::pe::testing::SyntheticImage::executable()
            .importing(imports)
            .build()
    }

    /// An absolute path on this host for a Unix-style spelling: unchanged on
    /// Unix, on drive `C:` with backslashes on Windows.
    fn host_path(unix_style: &str) -> String {
        if cfg!(windows) {
            format!("C:{}", unix_style.replace('/', "\\"))
        } else {
            unix_style.to_owned()
        }
    }

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "deltafin-native-upgrade-test-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir_all(root.join("native/deltafin")).unwrap();
            fs::create_dir_all(root.join("native/deltafin-bootstrap")).unwrap();
            fs::create_dir(root.join(".git")).unwrap();
            fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
            fs::write(root.join("Cargo.lock"), "version = 4\n").unwrap();
            fs::write(
                root.join("native/deltafin/Cargo.toml"),
                "[package]\nname = \"deltafin\"\nversion = \"0.1.0\"\n",
            )
            .unwrap();
            fs::write(
                root.join("native/deltafin-bootstrap/Cargo.toml"),
                "[package]\nname = \"deltafin-bootstrap\"\nversion = \"0.1.0\"\n",
            )
            .unwrap();
            fs::create_dir(root.join("git-exec")).unwrap();
            let helper = root.join(format!("git-exec/git-remote-https{EXE}"));
            fs::write(&helper, native_fixture()).unwrap();
            #[cfg(unix)]
            fs::set_permissions(&helper, fs::Permissions::from_mode(0o755)).unwrap();
            Self {
                root: crate::sys::fs::canonicalize(root).unwrap(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[derive(Clone, Copy)]
    enum FakeRelation {
        Equal,
        Behind,
        Ahead,
        Diverged,
    }

    struct FakeRunner {
        root: PathBuf,
        reported_git_root: PathBuf,
        relation: FakeRelation,
        dirty: String,
        incoming: Vec<String>,
        forbidden_git_configuration: Vec<String>,
        remote_urls: Vec<String>,
        commands: RefCell<Vec<(Program, Vec<String>)>>,
        environments: RefCell<Vec<CommandEnvironment>>,
        merged: Cell<bool>,
    }

    impl FakeRunner {
        fn new(root: &Path, relation: FakeRelation) -> Self {
            Self {
                root: root.to_owned(),
                reported_git_root: root.to_owned(),
                relation,
                dirty: String::new(),
                incoming: Vec::new(),
                forbidden_git_configuration: Vec::new(),
                remote_urls: vec!["https://github.com/gavamedia/deltafin.git".into()],
                commands: RefCell::new(Vec::new()),
                environments: RefCell::new(Vec::new()),
                merged: Cell::new(false),
            }
        }

        fn output(stdout: impl Into<Vec<u8>>) -> Result<CommandOutput> {
            Ok(CommandOutput {
                code: 0,
                stdout: stdout.into(),
                stderr: Vec::new(),
            })
        }

        fn code(code: i32) -> Result<CommandOutput> {
            Ok(CommandOutput {
                code,
                stdout: Vec::new(),
                stderr: Vec::new(),
            })
        }

        fn command_lines(&self) -> Vec<String> {
            self.commands
                .borrow()
                .iter()
                .map(|(program, args)| {
                    format!(
                        "{} {}",
                        program_name(*program),
                        logical_arguments(*program, args).join(" ")
                    )
                })
                .collect()
        }

        fn raw_command_lines(&self) -> Vec<String> {
            self.commands
                .borrow()
                .iter()
                .map(|(program, args)| format!("{} {}", program_name(*program), args.join(" ")))
                .collect()
        }

        fn cargo_build_environment(&self) -> CommandEnvironment {
            let commands = self.commands.borrow();
            let index = commands
                .iter()
                .position(|(program, arguments)| {
                    *program == Program::Cargo
                        && logical_arguments(*program, arguments)
                            .first()
                            .is_some_and(|value| value == "build")
                })
                .expect("Cargo build command");
            self.environments.borrow()[index].clone()
        }
    }

    impl Runner for FakeRunner {
        fn run(
            &self,
            program: Program,
            arguments: &[OsString],
            cwd: &Path,
            environment: &CommandEnvironment,
        ) -> Result<CommandOutput> {
            assert_eq!(cwd, self.root);
            let args: Vec<_> = arguments
                .iter()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect();
            self.commands.borrow_mut().push((program, args.clone()));
            self.environments.borrow_mut().push(environment.clone());
            let args = logical_arguments(program, &args);
            match (program, args) {
                (Program::Cargo, [version]) if version == "--version" => {
                    Self::output("cargo 1.85.0\n")
                }
                (Program::Cargo, [metadata, ..]) if metadata == "metadata" => Self::output(
                    serde_json::json!({
                        "workspace_root": self.root,
                        "packages": [
                            {
                                "name": "deltafin",
                                "manifest_path": self.root.join("native/deltafin/Cargo.toml")
                            },
                            {
                                "name": "deltafin-bootstrap",
                                "manifest_path": self.root.join("native/deltafin-bootstrap/Cargo.toml")
                            }
                        ]
                    })
                    .to_string(),
                ),
                (Program::Cargo, [build, ..]) if build == "build" => {
                    let artifact = self.root.join(format!("target/release/deltafin{EXE}"));
                    fs::create_dir_all(artifact.parent().unwrap()).unwrap();
                    fs::write(&artifact, native_fixture()).unwrap();
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        fs::set_permissions(&artifact, fs::Permissions::from_mode(0o755)).unwrap();
                    }
                    Self::output(format!(
                        "{}\n",
                        serde_json::json!({
                            "reason": "compiler-artifact",
                            "target": {"name": "deltafin", "kind": ["bin"]},
                            "executable": artifact
                        })
                    ))
                }
                (Program::Git, [rev_parse, show])
                    if rev_parse == "rev-parse" && show == "--show-toplevel" =>
                {
                    Self::output(format!("{}\n", self.reported_git_root.display()))
                }
                (Program::Git, [status, ..]) if status == "status" => {
                    Self::output(self.dirty.clone())
                }
                (Program::Git, [symbolic_ref, ..]) if symbolic_ref == "symbolic-ref" => {
                    Self::output("main\n")
                }
                (Program::Git, [rev_parse, abbrev, ..])
                    if rev_parse == "rev-parse" && abbrev == "--abbrev-ref" =>
                {
                    Self::output("origin/main\n")
                }
                (Program::Git, [config, get, ..]) if config == "config" && get == "--get" => {
                    Self::output("origin\n")
                }
                (Program::Git, [config, local, ..])
                    if config == "config" && local == "--local" =>
                {
                    if self.forbidden_git_configuration.is_empty() {
                        Self::code(1)
                    } else {
                        let mut encoded = self.forbidden_git_configuration.join("\0");
                        encoded.push('\0');
                        Self::output(encoded)
                    }
                }
                (Program::Git, [remote, get_url, ..])
                    if remote == "remote" && get_url == "get-url" =>
                {
                    Self::output(format!("{}\n", self.remote_urls.join("\n")))
                }
                (Program::Git, [exec_path]) if exec_path == "--exec-path" => {
                    Self::output(format!("{}\n", self.root.join("git-exec").display()))
                }
                (Program::Git, [fetch, ..]) if fetch == "fetch" => Self::output(Vec::new()),
                (Program::Git, [rev_parse, verify, revision])
                    if rev_parse == "rev-parse" && verify == "--verify" =>
                {
                    let value = match revision.as_str() {
                        "HEAD" if self.merged.get() => "new",
                        "HEAD" => match self.relation {
                            FakeRelation::Equal | FakeRelation::Behind => "old",
                            FakeRelation::Ahead => "new",
                            FakeRelation::Diverged => "local",
                        },
                        "@{upstream}" => match self.relation {
                            FakeRelation::Equal | FakeRelation::Ahead => "old",
                            FakeRelation::Behind => "new",
                            FakeRelation::Diverged => "remote",
                        },
                        _ => panic!("unexpected revision {revision}"),
                    };
                    Self::output(format!("{value}\n"))
                }
                (Program::Git, [merge_base, ancestor, left, right])
                    if merge_base == "merge-base" && ancestor == "--is-ancestor" =>
                {
                    let is_ancestor = match self.relation {
                        FakeRelation::Behind => (left.as_str(), right.as_str()) == ("old", "new"),
                        FakeRelation::Ahead => (left.as_str(), right.as_str()) == ("old", "new"),
                        FakeRelation::Equal => true,
                        FakeRelation::Diverged => false,
                    };
                    Self::code(if is_ancestor { 0 } else { 1 })
                }
                (Program::Git, [diff, rest @ ..])
                    if diff == "diff" && rest.iter().any(|value| value == "--name-only") =>
                {
                    let mut encoded = self.incoming.join("\0");
                    if !encoded.is_empty() {
                        encoded.push('\0');
                    }
                    Self::output(encoded)
                }
                (Program::Git, [merge, rest @ ..])
                    if merge == "merge"
                        && rest.iter().any(|value| value == "--ff-only")
                        && rest.last().is_some_and(|revision| revision == "new") =>
                {
                    self.merged.set(true);
                    Self::output(Vec::new())
                }
                _ => panic!(
                    "unexpected command: {} {}",
                    program.executable(),
                    args.join(" ")
                ),
            }
        }
    }

    /// The program as these tests spell it: `git` and `cargo` on every host,
    /// not `git.exe` where Windows names it so.
    fn program_name(program: Program) -> &'static str {
        match program {
            Program::Git => "git",
            Program::Cargo => "cargo",
        }
    }

    fn logical_arguments<'a>(program: Program, arguments: &'a [String]) -> &'a [String] {
        let prefix = match program {
            Program::Git => 1 + GIT_CONFIGURATION.len() * 2,
            Program::Cargo => CARGO_CONFIGURATION.len() * 2,
        };
        &arguments[prefix..]
    }

    #[cfg_attr(windows, allow(dead_code))]
    fn macho_with_path_command(command: u32, path: &[u8]) -> Vec<u8> {
        const LC_RPATH: u32 = 0x8000_001c;
        let path_offset = if command == LC_RPATH { 12 } else { 24 };
        let unaligned = path_offset + path.len() + 1;
        let command_bytes = (unaligned + 7) & !7;
        let mut bytes = vec![0_u8; 32 + command_bytes];
        bytes[..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]);
        bytes[16..20].copy_from_slice(&1_u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&(command_bytes as u32).to_le_bytes());
        bytes[32..36].copy_from_slice(&command.to_le_bytes());
        bytes[36..40].copy_from_slice(&(command_bytes as u32).to_le_bytes());
        bytes[40..44].copy_from_slice(&(path_offset as u32).to_le_bytes());
        let start = 32 + path_offset;
        bytes[start..start + path.len()].copy_from_slice(path);
        bytes
    }

    #[cfg_attr(windows, allow(dead_code))]
    fn macho_with_loader_path(path: &[u8]) -> Vec<u8> {
        macho_with_path_command(0x8000_001c, path)
    }

    #[cfg_attr(windows, allow(dead_code))]
    fn elf_with_dynamic_path(tag: i64, path: &[u8]) -> Vec<u8> {
        const HEADER_BYTES: usize = 64;
        const PROGRAM_BYTES: usize = 56;
        const PROGRAMS: usize = 2;
        const DYNAMIC_ENTRIES: usize = 4;
        const BASE: u64 = 0x0040_0000;
        let dynamic_offset = HEADER_BYTES + PROGRAM_BYTES * PROGRAMS;
        let dynamic_bytes = DYNAMIC_ENTRIES * 16;
        let string_offset = dynamic_offset + dynamic_bytes;
        let mut strings = vec![0_u8];
        strings.extend_from_slice(path);
        strings.push(0);
        let total = string_offset + strings.len();
        let mut bytes = vec![0_u8; total];
        bytes[..6].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1]);
        bytes[32..40].copy_from_slice(&(HEADER_BYTES as u64).to_le_bytes());
        bytes[54..56].copy_from_slice(&(PROGRAM_BYTES as u16).to_le_bytes());
        bytes[56..58].copy_from_slice(&(PROGRAMS as u16).to_le_bytes());

        let load = HEADER_BYTES;
        bytes[load..load + 4].copy_from_slice(&1_u32.to_le_bytes());
        bytes[load + 16..load + 24].copy_from_slice(&BASE.to_le_bytes());
        bytes[load + 32..load + 40].copy_from_slice(&(total as u64).to_le_bytes());

        let dynamic = HEADER_BYTES + PROGRAM_BYTES;
        bytes[dynamic..dynamic + 4].copy_from_slice(&2_u32.to_le_bytes());
        bytes[dynamic + 8..dynamic + 16].copy_from_slice(&(dynamic_offset as u64).to_le_bytes());
        bytes[dynamic + 16..dynamic + 24]
            .copy_from_slice(&(BASE + dynamic_offset as u64).to_le_bytes());
        bytes[dynamic + 32..dynamic + 40].copy_from_slice(&(dynamic_bytes as u64).to_le_bytes());

        for (index, (tag, value)) in [
            (5_i64, BASE + string_offset as u64),
            (10_i64, strings.len() as u64),
            (tag, 1_u64),
            (0_i64, 0_u64),
        ]
        .into_iter()
        .enumerate()
        {
            let entry = dynamic_offset + index * 16;
            bytes[entry..entry + 8].copy_from_slice(&tag.to_le_bytes());
            bytes[entry + 8..entry + 16].copy_from_slice(&value.to_le_bytes());
        }
        bytes[string_offset..].copy_from_slice(&strings);
        bytes
    }

    fn profile_hex(bytes: &[u8]) -> String {
        let mut value = String::from("hex:");
        for byte in bytes {
            value.push_str(&format!("{byte:02x}"));
        }
        value
    }

    fn profile_overrides<'a>(
        architectures: Option<&'a str>,
        compiler: Option<&'a str>,
        cuda_home: Option<&'a str>,
    ) -> [(&'static str, Option<&'a str>); 6] {
        [
            ("DELTAFIN_CUDA_ARCHITECTURES", architectures),
            ("CUDACXX", compiler),
            ("CMAKE_CUDA_COMPILER", compiler),
            ("CUDAToolkit_ROOT", Some("-")),
            ("CUDA_HOME", cuda_home),
            ("CUDA_PATH", Some("-")),
        ]
    }

    #[test]
    fn repository_location_uses_runtime_paths_not_the_builders_embedded_path() {
        let fixture = Fixture::new();
        let executable = fixture.root.join("target/release/deltafin");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::write(&executable, b"fixture").unwrap();
        assert_eq!(
            locate_repository_from(&executable, Path::new("/")),
            Ok(fixture.root.clone())
        );

        let copied = std::env::temp_dir().join("deltafin-copied-binary");
        assert_eq!(
            locate_repository_from(&copied, &fixture.root),
            Ok(fixture.root.clone())
        );
    }

    #[test]
    fn cuda_build_profile_replays_exact_native_environment_and_clears_ambient_aliases() {
        let root = profile_hex(host_path("/opt/audited/libtorch").as_bytes());
        let architectures = profile_hex(b"90;100");
        let compiler = profile_hex(host_path("/opt/cuda-13.0/bin/nvcc").as_bytes());
        let cuda_home = profile_hex(host_path("/opt/cuda-13.0").as_bytes());
        let profile = BuildProfile::parse(
            Some("v2"),
            Some("explicit"),
            Some(&root),
            Some("ON"),
            Some("ON"),
            profile_overrides(Some(&architectures), Some(&compiler), Some(&cuda_home)),
        )
        .unwrap();

        let environment = profile.command_environment();
        assert_eq!(
            environment.remove,
            PROFILE_ENVIRONMENT.map(OsString::from).to_vec()
        );
        let set: std::collections::BTreeMap<_, _> = environment.set.into_iter().collect();
        assert_eq!(
            set.get(OsStr::new("DELTAFIN_TORCH_ROOT")),
            Some(&OsString::from(host_path("/opt/audited/libtorch")))
        );
        assert_eq!(
            set.get(OsStr::new("DELTAFIN_CUDA_MOE")),
            Some(&OsString::from("ON"))
        );
        assert_eq!(
            set.get(OsStr::new("DELTAFIN_CUDA_ARCHITECTURES")),
            Some(&OsString::from("90;100"))
        );
        assert_eq!(
            set.get(OsStr::new("CUDA_HOME")),
            Some(&OsString::from(host_path("/opt/cuda-13.0")))
        );
        assert!(!set.contains_key(OsStr::new("LIBTORCH")));
    }

    #[test]
    fn automatic_cpu_profile_does_not_freeze_a_checkout_specific_torch_path() {
        let profile = BuildProfile::parse(
            Some("v2"),
            Some("bootstrap"),
            Some("-"),
            Some("OFF"),
            Some("OFF"),
            profile_overrides(Some("-"), Some("-"), Some("-")),
        )
        .unwrap();

        let environment = profile.command_environment();
        assert!(environment.set.iter().all(|(name, _)| {
            name != OsStr::new("DELTAFIN_TORCH_ROOT") && name != OsStr::new("LIBTORCH")
        }));
        assert!(
            environment
                .set
                .contains(&(OsString::from("DELTAFIN_CUDA_MOE"), OsString::from("OFF")))
        );
    }

    #[test]
    fn cuda_provider_without_nvcc_replays_only_its_toolkit_root() {
        let root = profile_hex(host_path("/opt/audited/cuda-libtorch").as_bytes());
        let cuda_home = profile_hex(host_path("/opt/cuda-13.0-runtime").as_bytes());
        let profile = BuildProfile::parse(
            Some("v2"),
            Some("explicit"),
            Some(&root),
            Some("ON"),
            Some("OFF"),
            profile_overrides(Some("-"), Some("-"), Some(&cuda_home)),
        )
        .unwrap();

        assert!(profile.cuda_provider);
        assert_eq!(profile.cuda_moe, "OFF");
        let set: std::collections::BTreeMap<_, _> =
            profile.command_environment().set.into_iter().collect();
        assert_eq!(
            set.get(OsStr::new("DELTAFIN_CUDA_MOE")),
            Some(&OsString::from("OFF"))
        );
        assert_eq!(
            set.get(OsStr::new("CUDA_HOME")),
            Some(&OsString::from(host_path("/opt/cuda-13.0-runtime")))
        );
        assert!(!set.contains_key(OsStr::new("CUDACXX")));
        assert!(!set.contains_key(OsStr::new("DELTAFIN_CUDA_ARCHITECTURES")));
    }

    #[test]
    fn malformed_or_downgraded_build_profiles_fail_closed() {
        let error = BuildProfile::parse(
            Some("v0"),
            Some("bootstrap"),
            Some("-"),
            None,
            Some("OFF"),
            profile_overrides(Some("-"), Some("-"), Some("-")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("versioned native build profile"));

        let error = BuildProfile::parse(
            Some("v1"),
            Some("bootstrap"),
            Some("-"),
            None,
            Some("ON"),
            profile_overrides(Some("-"), Some("-"), Some("-")),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without an explicit CUDA LibTorch")
        );
    }

    #[test]
    fn upgrade_passes_the_preserved_cuda_profile_to_cargo_build() {
        let fixture = Fixture::new();
        let runner = FakeRunner::new(&fixture.root, FakeRelation::Equal);
        let torch = fixture.root.join("operator-libtorch");
        let cuda = fixture.root.join("cuda");
        fs::create_dir(&torch).unwrap();
        fs::create_dir_all(cuda.join("bin")).unwrap();
        let root = profile_hex(torch.to_str().unwrap().as_bytes());
        let architectures = profile_hex(b"90");
        let compiler_path = cuda.join("bin/nvcc");
        let compiler = profile_hex(compiler_path.to_str().unwrap().as_bytes());
        let profile = BuildProfile::parse(
            Some("v2"),
            Some("explicit"),
            Some(&root),
            Some("ON"),
            Some("ON"),
            profile_overrides(Some(&architectures), Some(&compiler), Some("-")),
        )
        .unwrap();

        run_with_profile(&fixture.root, &runner, &profile, &mut |_| {}).unwrap();

        let environment = runner.cargo_build_environment();
        assert!(environment.clear);
        let set: std::collections::BTreeMap<_, _> = environment.set.into_iter().collect();
        assert_eq!(
            set.get(OsStr::new("DELTAFIN_TORCH_ROOT")),
            Some(&torch.into_os_string())
        );
        assert_eq!(
            set.get(OsStr::new("DELTAFIN_CUDA_MOE")),
            Some(&OsString::from("ON"))
        );
        assert_eq!(
            set.get(OsStr::new("CUDACXX")),
            Some(&compiler_path.into_os_string())
        );
        assert!(set.contains_key(OsStr::new("CARGO_HOME")));
        for forbidden in [
            "PYTHONPATH",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC_WRAPPER",
            "DYLD_INSERT_LIBRARIES",
            "LD_PRELOAD",
        ] {
            assert!(!set.contains_key(OsStr::new(forbidden)), "{forbidden}");
        }
    }

    #[cfg(feature = "runtime")]
    #[test]
    fn running_test_artifact_contains_a_self_consistent_build_profile() {
        BuildProfile::embedded().unwrap();
    }

    #[cfg(feature = "runtime")]
    #[test]
    fn updater_program_gate_accepts_native_formats_and_rejects_shebangs() {
        for magic in [
            [0x7f, b'E', b'L', b'F'],
            [0xcf, 0xfa, 0xed, 0xfe],
            [0xfe, 0xed, 0xfa, 0xcf],
            [0xca, 0xfe, 0xba, 0xbe],
            [0xbe, 0xba, 0xfe, 0xca],
        ] {
            assert!(native_executable_magic(&magic));
        }
        for script in [b"#!/b".as_slice(), b"MZ\0\0", b"", b"git\n"] {
            assert!(!native_executable_magic(script));
        }
    }

    #[cfg(unix)]
    #[test]
    fn native_program_resolution_validates_target_but_preserves_multicall_symlink_name() {
        use crate::sys::fs::symlink;

        let fixture = Fixture::new();
        let tools = fixture.root.join("proxy-tools");
        fs::create_dir(&tools).unwrap();
        let proxy = tools.join("rustup-native");
        fs::write(&proxy, macho_with_loader_path(b"@loader_path/native")).unwrap();
        fs::set_permissions(&proxy, fs::Permissions::from_mode(0o755)).unwrap();
        let cargo = tools.join("cargo");
        symlink("rustup-native", &cargo).unwrap();

        let admitted = resolve_native_program_in_path("cargo", tools.as_os_str()).unwrap();

        assert_eq!(admitted, cargo);
        assert!(admitted.is_absolute());
        assert_ne!(admitted, crate::sys::fs::canonicalize(&admitted).unwrap());
        assert_eq!(crate::sys::fs::canonicalize(&admitted).unwrap(), proxy);
    }

    #[cfg(feature = "runtime")]
    #[cfg(unix)]
    #[test]
    fn isolated_cargo_home_still_launches_the_selected_native_cargo() {
        let fixture = Fixture::new();

        let output = checked(
            &ProcessRunner,
            Program::Cargo,
            strings(["--version"]),
            &fixture.root,
            "checking isolated native Cargo",
            &[0],
        )
        .unwrap();

        assert!(
            std::str::from_utf8(&output.stdout)
                .unwrap()
                .starts_with("cargo ")
        );
        let cargo_home = fixture.root.join(ISOLATED_CARGO_HOME);
        assert!(cargo_home.is_dir());
        assert_eq!(fs::metadata(cargo_home).unwrap().mode() & 0o077, 0);
    }

    #[test]
    fn preserved_path_classifier_matches_python_updater_contract() {
        for path in [
            ".deltafin/toolchains/pytorch-2.13.0-cpu-macos-arm64/torch/lib/libtorch.dylib",
            "k3-resident-mix46/tensors/embed.weight.i6",
            "k3-draft-qwen3-0.6b-base/model.safetensors",
            "k3-cache-raw/expert.npz",
            "k3-experts-scale4/L92.sc4",
            "k3-experts-scale4/scale4-manifest.json",
            "models/Kimi-K3/file",
            "nested/expert.safetensors",
            "nested/expert.sc4",
            "nested/expert.weight.i6",
            "tools/k3pkg/modeling_kimi_k3.py",
            "venv/bin/python",
            "tiktoken.model",
        ] {
            assert!(is_preserved_data_path(path), "{path}");
        }
        assert!(!is_preserved_data_path("tools/kimi_run.py"));
    }

    #[test]
    fn missing_target_source_bootstrap_stops_before_cargo_or_git() {
        let fixture = Fixture::new();
        fs::remove_file(fixture.root.join("native/deltafin-bootstrap/Cargo.toml")).unwrap();
        let runner = FakeRunner::new(&fixture.root, FakeRelation::Equal);

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("native/deltafin-bootstrap/Cargo.toml")
        );
        assert!(runner.command_lines().is_empty());
    }

    // Mach-O/ELF loader metadata: the PE spelling is tested below.
    #[cfg(unix)]
    #[test]
    fn compiled_artifact_rejects_embedded_python_environment_rpath() {
        let fixture = Fixture::new();
        let target = fixture.root.join("target/release");
        fs::create_dir_all(&target).unwrap();
        let artifact = target.join("deltafin");
        fs::write(
            &artifact,
            macho_with_loader_path(b"/Users/example/project/venv/lib/torch"),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o755)).unwrap();
        }

        let error =
            verify_artifact(&fixture.root, &fixture.root.join("target"), &artifact).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("forbidden venv Python-environment")
        );
    }

    // Mach-O/ELF loader metadata: the PE spelling is tested below.
    #[cfg(unix)]
    #[test]
    fn compiled_artifact_rejects_macho_libpython_load_dependency() {
        let fixture = Fixture::new();
        let target = fixture.root.join("target/release");
        fs::create_dir_all(&target).unwrap();
        let artifact = target.join("deltafin");
        fs::write(
            &artifact,
            macho_with_path_command(0x0000_000c, b"@rpath/libPython3.14.dylib"),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o755)).unwrap();
        }

        let error =
            verify_artifact(&fixture.root, &fixture.root.join("target"), &artifact).unwrap_err();

        assert!(error.to_string().contains("forbidden libpython dependency"));
        assert!(error.to_string().contains("libPython3.14.dylib"));
    }

    // Mach-O/ELF loader metadata: the PE spelling is tested below.
    #[cfg(unix)]
    #[test]
    fn compiled_artifact_rejects_elf_libtorch_python_needed_dependency() {
        let fixture = Fixture::new();
        let target = fixture.root.join("target/release");
        fs::create_dir_all(&target).unwrap();
        let artifact = target.join("deltafin");
        fs::write(
            &artifact,
            elf_with_dynamic_path(1, b"libtorch_python.so.2.13"),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o755)).unwrap();
        }

        let error =
            verify_artifact(&fixture.root, &fixture.root.join("target"), &artifact).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("forbidden libtorch_python dependency")
        );
        assert!(error.to_string().contains("libtorch_python.so.2.13"));
    }

    // Mach-O/ELF loader metadata: the PE spelling is tested below.
    #[cfg(unix)]
    #[test]
    fn python_named_search_path_is_not_mistaken_for_a_loaded_dependency() {
        let fixture = Fixture::new();
        let target = fixture.root.join("target/release");
        fs::create_dir_all(&target).unwrap();
        let artifact = target.join("deltafin");
        fs::write(
            &artifact,
            macho_with_loader_path(b"@loader_path/libpython-plugins"),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o755)).unwrap();
        }

        verify_artifact(&fixture.root, &fixture.root.join("target"), &artifact).unwrap();
    }

    fn write_release_artifact(fixture: &Fixture, image: Vec<u8>) -> PathBuf {
        let target = fixture.root.join("target/release");
        fs::create_dir_all(&target).unwrap();
        let artifact = target.join(format!("deltafin{EXE}"));
        fs::write(&artifact, image).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&artifact, fs::Permissions::from_mode(0o755)).unwrap();
        }
        artifact
    }

    #[test]
    fn compiled_artifact_rejects_a_python_dll_import() {
        for dll in ["python313.dll", "python3.dll", "torch_python.dll"] {
            let fixture = Fixture::new();
            // The interpreter's DLL is named first: the verdict must not
            // depend on whether this host has a Windows system directory.
            let artifact =
                write_release_artifact(&fixture, pe_fixture(&[dll, "kernel32.dll"]));

            let error = verify_artifact(&fixture.root, &fixture.root.join("target"), &artifact)
                .unwrap_err();

            assert!(error.to_string().contains("forbidden"), "{dll}: {error}");
            assert!(error.to_string().contains(dll), "{dll}: {error}");
        }
    }

    // Needs a real Windows system directory to resolve `kernel32.dll`.
    #[cfg(windows)]
    #[test]
    fn compiled_artifact_with_only_platform_imports_is_accepted() {
        let fixture = Fixture::new();
        let artifact = write_release_artifact(&fixture, native_fixture());
        verify_artifact(&fixture.root, &fixture.root.join("target"), &artifact).unwrap();
    }

    // The test executable of a runtime build imports LibTorch's DLLs, which
    // Windows resolves from beside the executable; those are the build's
    // concern, so only the dependency-free configuration is audited here.
    #[cfg(not(all(windows, feature = "runtime")))]
    #[test]
    fn loader_audit_accepts_the_real_platform_test_executable() {
        let executable = std::env::current_exe().unwrap();
        verify_no_python_environment_path(&executable).unwrap();
    }

    #[test]
    fn clean_behind_fast_forwards_then_builds_locked_release_binary() {
        let fixture = Fixture::new();
        let runner = FakeRunner::new(&fixture.root, FakeRelation::Behind);
        let mut output = Vec::new();

        run_with(&fixture.root, &runner, &mut |line| {
            output.push(line.to_owned())
        })
        .unwrap();

        assert!(runner.merged.get());
        let commands = runner.command_lines();
        let fetch = commands
            .iter()
            .position(|line| line.starts_with("git fetch "))
            .unwrap();
        let merge = commands
            .iter()
            .position(|line| line.starts_with("git merge --ff-only "))
            .unwrap();
        let build = commands
            .iter()
            .position(|line| line.starts_with("cargo build --locked --release"))
            .unwrap();
        assert!(fetch < merge && merge < build);
        assert!(commands[build].contains("--package deltafin --bin deltafin"));
        assert!(!commands.iter().any(|line| line.starts_with("cargo run ")));
        let flattened = commands.join("\n");
        for forbidden in ["python", " reset ", " clean ", " stash "] {
            assert!(!format!(" {flattened} ").contains(forbidden), "{forbidden}");
        }
        assert!(
            output
                .last()
                .unwrap()
                .contains("Downloaded models and caches remain in place")
        );
        assert!(
            output
                .iter()
                .any(|line| { line.contains("target revision's authenticated native runtime") })
        );
    }

    #[test]
    fn dirty_checkout_stops_before_fetch_merge_or_build() {
        let fixture = Fixture::new();
        let mut runner = FakeRunner::new(&fixture.root, FakeRelation::Behind);
        runner.dirty = " M native/deltafin/src/lib.rs\n".into();

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("will not stash, discard, or overwrite")
        );
        let commands = runner.command_lines().join("\n");
        assert!(!commands.contains("git fetch"));
        assert!(!commands.contains("git merge --ff-only"));
        assert!(!commands.contains("cargo build"));
    }

    #[test]
    fn mismatched_git_root_stops_before_fetch() {
        let fixture = Fixture::new();
        let mut runner = FakeRunner::new(&fixture.root, FakeRelation::Behind);
        runner.reported_git_root = fixture.root.parent().unwrap().to_owned();

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("refusing to update the wrong checkout")
        );
        assert!(!runner.command_lines().join("\n").contains("git fetch"));
    }

    #[test]
    fn diverged_checkout_never_merges_or_builds() {
        let fixture = Fixture::new();
        let runner = FakeRunner::new(&fixture.root, FakeRelation::Diverged);

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(error.to_string().contains("have diverged"));
        let commands = runner.command_lines().join("\n");
        assert!(!commands.contains("git merge --ff-only"));
        assert!(!commands.contains("cargo build"));
    }

    #[test]
    fn incoming_model_collision_stops_before_fast_forward() {
        let fixture = Fixture::new();
        let mut runner = FakeRunner::new(&fixture.root, FakeRelation::Behind);
        runner.incoming = vec![
            "native/deltafin/src/lib.rs".into(),
            "k3-resident-int8/weights.i8".into(),
        ];

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("reserved for downloaded models or caches")
        );
        assert!(!runner.merged.get());
        assert!(!runner.command_lines().join("\n").contains("cargo build"));
    }

    #[test]
    fn ahead_and_equal_rebuild_without_changing_git_state() {
        for relation in [FakeRelation::Ahead, FakeRelation::Equal] {
            let fixture = Fixture::new();
            let runner = FakeRunner::new(&fixture.root, relation);

            run_with(&fixture.root, &runner, &mut |_| {}).unwrap();

            assert!(!runner.merged.get());
            let commands = runner.command_lines().join("\n");
            assert!(!commands.contains("git merge --ff-only"));
            assert!(commands.contains("cargo build --locked --release"));
        }
    }

    #[test]
    fn every_git_and_cargo_child_is_environment_isolated_and_argument_hardened() {
        let fixture = Fixture::new();
        let runner = FakeRunner::new(&fixture.root, FakeRelation::Equal);

        run_with(&fixture.root, &runner, &mut |_| {}).unwrap();

        for environment in runner.environments.borrow().iter() {
            assert!(environment.clear);
            let names: Vec<_> = environment
                .set
                .iter()
                .map(|(name, _)| name.to_string_lossy())
                .collect();
            for forbidden in [
                "PYTHONPATH",
                "GIT_EXEC_PATH",
                "GIT_SSH_COMMAND",
                "GIT_ASKPASS",
                "RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER",
                "CARGO_BUILD_RUSTC_WRAPPER",
                "CARGO_TARGET_DIR",
                "DYLD_INSERT_LIBRARIES",
                "LD_PRELOAD",
                "CC",
                "CXX",
            ] {
                assert!(!names.iter().any(|name| name == forbidden), "{forbidden}");
            }
        }

        for line in runner.raw_command_lines() {
            if line.starts_with("git ") {
                assert!(line.starts_with("git --no-pager -c core.hooksPath=/dev/null "));
                for setting in GIT_CONFIGURATION {
                    assert!(line.contains(&format!("-c {setting}")), "{line}");
                }
            } else if line.starts_with("cargo ") {
                for setting in CARGO_CONFIGURATION {
                    assert!(line.contains(&format!("--config {setting}")), "{line}");
                }
            }
        }

        let environments = runner.environments.borrow();
        let git = environments
            .iter()
            .find(|environment| {
                environment
                    .set
                    .iter()
                    .any(|(name, _)| name == OsStr::new("GIT_ALLOW_PROTOCOL"))
            })
            .unwrap();
        assert!(git.set.contains(&(
            OsString::from("GIT_ALLOW_PROTOCOL"),
            OsString::from("https")
        )));
        assert!(git.set.contains(&(
            OsString::from("GIT_CONFIG_GLOBAL"),
            OsString::from("/dev/null")
        )));
    }

    #[test]
    fn executable_local_git_configuration_stops_before_fetch() {
        let fixture = Fixture::new();
        let mut runner = FakeRunner::new(&fixture.root, FakeRelation::Behind);
        runner.forbidden_git_configuration = vec!["filter.payload.process".into()];

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(error.to_string().contains("filter.payload.process"));
        assert!(!runner.command_lines().join("\n").contains("git fetch"));
    }

    #[test]
    fn non_https_remote_stops_before_fetch_or_helper_execution() {
        let fixture = Fixture::new();
        let mut runner = FakeRunner::new(&fixture.root, FakeRelation::Behind);
        runner.remote_urls = vec!["ext::sh -c payload".into()];

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(error.to_string().contains("only fetches explicit HTTPS"));
        let commands = runner.command_lines().join("\n");
        assert!(!commands.contains("git fetch"));
        assert!(!commands.contains("git --exec-path"));
    }

    #[test]
    fn interpreted_https_helper_stops_before_fetch() {
        let fixture = Fixture::new();
        let helper = fixture.root.join(format!("git-exec/git-remote-https{EXE}"));
        fs::write(&helper, b"#!/bin/sh\nexit 0\n").unwrap();

        let runner = FakeRunner::new(&fixture.root, FakeRelation::Behind);
        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("not a compiled native executable")
        );
        assert!(error.to_string().contains("script or interpreter shim"));
        assert!(!runner.command_lines().join("\n").contains("git fetch"));
    }

    #[test]
    fn hierarchical_cargo_configuration_stops_before_any_child_process() {
        let fixture = Fixture::new();
        fs::create_dir(fixture.root.join(".cargo")).unwrap();
        fs::write(
            fixture.root.join(".cargo/config.toml"),
            "[build]\nrustc-wrapper = '/tmp/payload'\n",
        )
        .unwrap();
        let runner = FakeRunner::new(&fixture.root, FakeRelation::Equal);

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(error.to_string().contains("compiler wrappers"));
        assert!(error.to_string().contains(".cargo/config.toml"));
        assert!(runner.command_lines().is_empty());
    }

    #[test]
    fn isolated_cargo_home_rejects_later_injected_configuration() {
        let fixture = Fixture::new();
        let cargo_home = fixture.root.join(ISOLATED_CARGO_HOME);
        fs::create_dir_all(&cargo_home).unwrap();
        fs::write(
            cargo_home.join("config.toml"),
            "[build]\nrustc-wrapper = '/tmp/payload'\n",
        )
        .unwrap();
        let runner = FakeRunner::new(&fixture.root, FakeRelation::Equal);

        let error = run_with(&fixture.root, &runner, &mut |_| {}).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unexpectedly contains executable configuration")
        );
        assert!(runner.command_lines().is_empty());
    }
}
