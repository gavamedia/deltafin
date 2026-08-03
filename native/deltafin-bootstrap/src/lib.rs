//! Python- and shell-free bootstrap for Deltafin's pinned C++ PyTorch runtime.
//!
//! PyTorch's official wheels carry the native headers and libraries consumed
//! directly by Deltafin. This crate admits only an exact,
//! size-and-SHA-256-pinned CPU wheel, extracts an authenticated native subset,
//! deliberately omits `libtorch_python`, and publishes the result atomically.
//! The version-1 subset retains upstream `share/cmake` metadata and native
//! helpers as inert authenticated files for install compatibility; Deltafin's
//! Rust-owned production build neither evaluates nor launches them.

#![cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]

use std::collections::BTreeSet;
use std::ffi::{CString, OsStr};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use curl::easy::{Easy, List};
use sha2::{Digest, Sha256};
use zip::{CompressionMethod, ZipArchive};

const PYTORCH_VERSION: &str = "2.13.0";
const TOOLCHAIN_FORMAT: &str = "deltafin-pytorch-toolchain-v1";
const MARKER_NAME: &str = ".deltafin-toolchain";
const FILES_MANIFEST_FORMAT: &str = "deltafin-native-files-v1";
const FILES_MANIFEST_NAME: &str = ".deltafin-files";
const TORCH_CPP_API_HEADER: &str = "torch/include/torch/csrc/api/include/torch/torch.h";
const MAX_FILES_MANIFEST_BYTES: u64 = 8 << 20;
const MAX_DOWNLOAD_HEADERS: u64 = 64 << 10;
const COPY_BUFFER: usize = 1 << 20;
const MINIMUM_LIBCURL_VERSION: u32 = 0x07_1c_00;

/// Bounds are deliberately above the official CPU wheels but far below an
/// unbounded archive expansion.  Tests use smaller values to exercise every
/// rejection without allocating large fixtures.
#[derive(Clone, Copy, Debug)]
struct ArchiveLimits {
    archive_bytes: u64,
    entries: usize,
    metadata_bytes: u64,
    expanded_bytes: u64,
    selected_bytes: u64,
    single_file_bytes: u64,
    path_bytes: usize,
    path_components: usize,
}

const PRODUCTION_LIMITS: ArchiveLimits = ArchiveLimits {
    archive_bytes: 1 << 30,
    entries: 50_000,
    metadata_bytes: 64 << 20,
    expanded_bytes: 4 << 30,
    selected_bytes: 2 << 30,
    single_file_bytes: 1 << 30,
    path_bytes: 4096,
    path_components: 64,
};

#[derive(Debug)]
pub struct BootstrapError(String);

impl BootstrapError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BootstrapError {}

pub type Result<T> = std::result::Result<T, BootstrapError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformTarget {
    MacosArm64,
    LinuxX86_64,
    LinuxAarch64,
}

impl PlatformTarget {
    pub fn detect() -> Result<Self> {
        match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => Ok(Self::MacosArm64),
            ("linux", "x86_64") => Ok(Self::LinuxX86_64),
            ("linux", "aarch64") => Ok(Self::LinuxAarch64),
            (os, arch) => Err(BootstrapError::new(format!(
                "PyTorch {PYTORCH_VERSION} native bootstrap does not have a pinned CPU artifact for {os}/{arch}"
            ))),
        }
    }

    fn library_suffix(self) -> &'static str {
        match self {
            Self::MacosArm64 => "dylib",
            Self::LinuxX86_64 | Self::LinuxAarch64 => "so",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::MacosArm64 => "macos-arm64",
            Self::LinuxX86_64 => "linux-x86_64",
            Self::LinuxAarch64 => "linux-aarch64",
        }
    }
}

impl fmt::Display for PlatformTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.id())
    }
}

#[derive(Clone, Copy, Debug)]
struct Artifact {
    target: PlatformTarget,
    url: &'static str,
    sha256: &'static str,
    size: u64,
    files_manifest_sha256: &'static str,
}

// These immutable pins come from the SHA-256 fragments and Content-Length
// values on PyTorch's official wheel index.  A Python ABI tag is merely the
// wheel container tag here: no Python file or Python library is extracted.
const MACOS_ARM64: Artifact = Artifact {
    target: PlatformTarget::MacosArm64,
    url: "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0-cp314-cp314-macosx_14_0_arm64.whl",
    sha256: "d849b390e07d8d333ce8ecaf91b273c656c598379a19c9acf1318a883f6b391c",
    size: 111_227_066,
    files_manifest_sha256: "a3f11e498fc1458304c0e2b678fd1d028ef764a7cae3d9654f057b197ca36aaf",
};
const LINUX_X86_64: Artifact = Artifact {
    target: PlatformTarget::LinuxX86_64,
    url: "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl",
    sha256: "d20fa53ee744502fa4c69818a720b05ca0d37abd055d4f6e66cae155114bc691",
    size: 191_822_516,
    files_manifest_sha256: "497eccc6a2d47f76fd5878ea7910e9305b2e7ee46ab535a6884becb9661165af",
};
const LINUX_AARCH64: Artifact = Artifact {
    target: PlatformTarget::LinuxAarch64,
    url: "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_aarch64.whl",
    sha256: "ca021f9eb2f8345c83fa03e3a04587308afb8df71bd472670b3ece00df58621c",
    size: 155_020_718,
    files_manifest_sha256: "bf0e29d0ea3dba35bc8e259d82f4d4b0e3b76e97501a5d467636f1685b510a1a",
};

fn artifact_for(target: PlatformTarget) -> &'static Artifact {
    match target {
        PlatformTarget::MacosArm64 => &MACOS_ARM64,
        PlatformTarget::LinuxX86_64 => &LINUX_X86_64,
        PlatformTarget::LinuxAarch64 => &LINUX_AARCH64,
    }
}

#[derive(Debug)]
pub struct InstallOptions {
    pub repository_root: PathBuf,
    pub target: PlatformTarget,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum InstallDisposition {
    Installed,
    AlreadyPresent,
}

#[derive(Debug)]
pub struct InstallReport {
    pub disposition: InstallDisposition,
    pub torch_root: PathBuf,
    pub artifact_url: &'static str,
    pub artifact_sha256: &'static str,
}

#[derive(Debug)]
pub struct ValidatedToolchain {
    pub torch_root: PathBuf,
    /// Every authenticated file and containing directory. Build scripts can
    /// register these as inputs so Cargo re-runs validation after local drift.
    pub tracked_paths: Vec<PathBuf>,
}

pub fn repository_toolchain_path(repository_root: &Path, target: PlatformTarget) -> PathBuf {
    repository_root.join(format!(
        ".deltafin/toolchains/pytorch-{PYTORCH_VERSION}-cpu-{}",
        target.id()
    ))
}

pub fn validate_repository_toolchain(
    repository_root: &Path,
    target: PlatformTarget,
) -> Result<ValidatedToolchain> {
    let repository_root = validate_repository_root(repository_root)?;
    let destination = repository_toolchain_path(&repository_root, target);
    let tracked_paths = validate_installed(&destination, artifact_for(target))?;
    Ok(ValidatedToolchain {
        torch_root: destination.join("torch"),
        tracked_paths,
    })
}

pub fn install(options: &InstallOptions) -> Result<InstallReport> {
    let repository_root = validate_repository_root(&options.repository_root)?;
    let artifact = artifact_for(options.target);
    validate_artifact(artifact)?;
    let toolchains = prepare_toolchain_directory(&repository_root)?;
    let destination = repository_toolchain_path(&repository_root, options.target);

    if destination.exists() {
        validate_installed(&destination, artifact)?;
        return Ok(report(
            InstallDisposition::AlreadyPresent,
            &destination,
            artifact,
        ));
    }

    let (archive, archive_file) = unique_file(&toolchains, "wheel", "whl")?;
    let mut cleanup = Cleanup::new();
    cleanup.files.push(archive.clone());
    download_exact(artifact, &archive, archive_file)?;

    let staging = unique_directory(&toolchains, "staging")?;
    cleanup.directories.push(staging.clone());
    let entries = extract_native_layout(&archive, &staging, artifact, PRODUCTION_LIMITS)?;
    let files_manifest_sha256 = write_files_manifest(&staging, &entries)?;
    write_marker(&staging, artifact, &files_manifest_sha256)?;
    finish_tree(&staging)?;

    fs::remove_file(&archive)
        .map_err(|error| io_error("remove verified wheel after extraction", &archive, error))?;
    cleanup.files.clear();
    fsync_directory(&toolchains)?;

    if let Err(error) = rename_noreplace(&staging, &destination) {
        if destination.exists() && validate_installed(&destination, artifact).is_ok() {
            return Ok(report(
                InstallDisposition::AlreadyPresent,
                &destination,
                artifact,
            ));
        }
        return Err(error);
    }
    cleanup.directories.clear();
    fsync_directory(&toolchains)?;
    validate_installed(&destination, artifact)?;
    Ok(report(
        InstallDisposition::Installed,
        &destination,
        artifact,
    ))
}

fn report(
    disposition: InstallDisposition,
    destination: &Path,
    artifact: &'static Artifact,
) -> InstallReport {
    InstallReport {
        disposition,
        torch_root: destination.join("torch"),
        artifact_url: artifact.url,
        artifact_sha256: artifact.sha256,
    }
}

fn validate_repository_root(path: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(path)
        .map_err(|error| io_error("canonicalize repository root", path, error))?;
    if !canonical.is_dir()
        || !canonical.join("Cargo.toml").is_file()
        || !canonical.join("native/deltafin/Cargo.toml").is_file()
    {
        return Err(BootstrapError::new(format!(
            "{} is not a Deltafin repository root",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn prepare_toolchain_directory(repository: &Path) -> Result<PathBuf> {
    let state = ensure_child_directory(repository, OsStr::new(".deltafin"), 0o700)?;
    ensure_child_directory(&state, OsStr::new("toolchains"), 0o700)
}

fn ensure_child_directory(parent: &Path, name: &OsStr, mode: u32) -> Result<PathBuf> {
    if name.as_bytes().is_empty() || name.as_bytes().contains(&b'/') {
        return Err(BootstrapError::new("unsafe toolchain directory component"));
    }
    let path = parent.join(name);
    match fs::create_dir(&path) {
        Ok(()) => {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))
                .map_err(|error| io_error("set toolchain directory permissions", &path, error))?;
            fsync_directory(parent)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io_error("create toolchain directory", &path, error)),
    }
    require_real_directory(&path)?;
    let permissions = fs::symlink_metadata(&path)
        .map_err(|error| io_error("inspect toolchain directory permissions", &path, error))?
        .permissions()
        .mode();
    if permissions & 0o022 != 0 {
        return Err(BootstrapError::new(format!(
            "toolchain directory {} is group/world writable; refusing a raceable install root",
            path.display()
        )));
    }
    Ok(path)
}

fn validate_artifact(artifact: &Artifact) -> Result<()> {
    let expected_prefix = "https://download-r2.pytorch.org/whl/";
    if !artifact.url.starts_with(expected_prefix)
        || artifact.url.contains(['#', '?', '\\', '@'])
        || artifact
            .url
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
    {
        return Err(BootstrapError::new("pinned PyTorch artifact URL is unsafe"));
    }
    for pin in [artifact.sha256, artifact.files_manifest_sha256] {
        let parsed = parse_sha256(pin)?;
        if format_digest(&parsed) != pin {
            return Err(BootstrapError::new(
                "pinned PyTorch digest is not canonical lowercase SHA-256",
            ));
        }
    }
    if artifact.size == 0 || artifact.size > (1 << 30) {
        return Err(BootstrapError::new(
            "pinned PyTorch artifact size is outside the admitted range",
        ));
    }
    Ok(())
}

fn download_exact(artifact: &Artifact, target: &Path, mut file: File) -> Result<()> {
    require_https_capable_libcurl()?;
    let mut easy = Easy::new();
    easy.url(artifact.url)
        .map_err(|error| curl_error("set pinned PyTorch URL", error))?;
    easy.useragent("deltafin-bootstrap/0.1")
        .map_err(|error| curl_error("set bootstrap user agent", error))?;
    easy.connect_timeout(Duration::from_secs(30))
        .map_err(|error| curl_error("set connect timeout", error))?;
    easy.low_speed_limit(1024)
        .map_err(|error| curl_error("set low-speed limit", error))?;
    easy.low_speed_time(Duration::from_secs(120))
        .map_err(|error| curl_error("set low-speed timeout", error))?;
    easy.follow_location(false)
        .map_err(|error| curl_error("disable redirects", error))?;
    easy.ssl_verify_peer(true)
        .map_err(|error| curl_error("require TLS peer verification", error))?;
    easy.ssl_verify_host(true)
        .map_err(|error| curl_error("require TLS host verification", error))?;
    let mut headers = List::new();
    headers
        .append("Accept-Encoding: identity")
        .map_err(|error| curl_error("set identity content encoding", error))?;
    easy.http_headers(headers)
        .map_err(|error| curl_error("install bootstrap request headers", error))?;

    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut header_bytes = 0_u64;
    let mut header_error: Option<BootstrapError> = None;
    let mut body_error: Option<BootstrapError> = None;
    let transfer_result = {
        let mut transfer = easy.transfer();
        transfer
            .header_function(|data| {
                header_bytes = match header_bytes.checked_add(data.len() as u64) {
                    Some(next) if next <= MAX_DOWNLOAD_HEADERS => next,
                    _ => {
                        header_error = Some(BootstrapError::new(
                            "PyTorch download headers exceeded 64 KiB",
                        ));
                        return false;
                    }
                };
                true
            })
            .map_err(|error| curl_error("install response-header bound", error))?;
        transfer
            .write_function(|data| {
                if body_error.is_some() {
                    return Ok(0);
                }
                let next = match bytes.checked_add(data.len() as u64) {
                    Some(next) if next <= artifact.size => next,
                    _ => {
                        body_error = Some(BootstrapError::new(format!(
                            "PyTorch response exceeded its exact {}-byte pin",
                            artifact.size
                        )));
                        return Ok(0);
                    }
                };
                if let Err(error) = file.write_all(data) {
                    body_error = Some(io_error("write pinned PyTorch wheel", target, error));
                    return Ok(0);
                }
                digest.update(data);
                bytes = next;
                Ok(data.len())
            })
            .map_err(|error| curl_error("install response-body verifier", error))?;
        transfer.perform()
    };
    if let Some(error) = header_error {
        return Err(error);
    }
    if let Some(error) = body_error {
        return Err(error);
    }
    transfer_result.map_err(|error| curl_error("download pinned PyTorch wheel", error))?;
    let status = easy
        .response_code()
        .map_err(|error| curl_error("read PyTorch response status", error))?;
    if status != 200 {
        return Err(BootstrapError::new(format!(
            "pinned PyTorch server returned HTTP {status}; redirects are intentionally not followed"
        )));
    }
    if bytes != artifact.size {
        return Err(BootstrapError::new(format!(
            "pinned PyTorch wheel was {bytes} bytes; expected exactly {}",
            artifact.size
        )));
    }
    let actual = format_digest(digest.finalize().as_slice());
    if actual != artifact.sha256 {
        return Err(BootstrapError::new(format!(
            "pinned PyTorch wheel SHA-256 mismatch: got {actual}, expected {}",
            artifact.sha256
        )));
    }
    file.sync_all()
        .map_err(|error| io_error("fsync pinned PyTorch wheel", target, error))?;
    Ok(())
}

/// Validate the linked system transfer library itself, once per process.  The
/// direct native build verifies the DSO/Mach-O selection; only libcurl can
/// authoritatively report whether that selected library supplies TLS and the
/// HTTPS protocol at runtime.
fn require_https_capable_libcurl() -> Result<()> {
    static CHECK: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    let result = CHECK.get_or_init(|| {
        let version = curl::Version::get();
        if version.version_num() < MINIMUM_LIBCURL_VERSION {
            return Err(format!(
                "linked libcurl {} is older than Deltafin's minimum 7.28.0",
                version.version()
            ));
        }
        if !version.feature_ssl() {
            return Err(format!(
                "linked libcurl {} has no TLS/SSL support",
                version.version()
            ));
        }
        if !version
            .protocols()
            .any(|protocol| protocol.eq_ignore_ascii_case("https"))
        {
            return Err(format!(
                "linked libcurl {} does not advertise the HTTPS protocol",
                version.version()
            ));
        }
        Ok(())
    });
    result
        .as_ref()
        .map(|_| ())
        .map_err(|message| BootstrapError::new(message.clone()))
}

#[derive(Debug)]
struct EntryPlan {
    index: usize,
    relative: PathBuf,
    size: u64,
    executable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ManifestEntry {
    relative: PathBuf,
    size: u64,
    sha256: [u8; 32],
}

fn extract_native_layout(
    archive_path: &Path,
    destination: &Path,
    artifact: &Artifact,
    limits: ArchiveLimits,
) -> Result<Vec<ManifestEntry>> {
    require_real_directory(destination)?;
    require_owned_nonwritable(archive_path, false)?;
    let mut file = secure_open_read(archive_path)?;
    preflight_zip_metadata(&mut file, archive_path, limits)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| io_error("rewind preflighted wheel", archive_path, error))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|error| BootstrapError::new(format!("open pinned wheel as ZIP: {error}")))?;
    if archive.is_empty() || archive.len() > limits.entries {
        return Err(BootstrapError::new(format!(
            "wheel has {} entries; admitted range is 1..={} ",
            archive.len(),
            limits.entries
        )));
    }

    let mut names = BTreeSet::new();
    let mut files = BTreeSet::new();
    let mut plans = Vec::new();
    let mut metadata_bytes = 0_u64;
    let mut expanded_bytes = 0_u64;
    let mut selected_bytes = 0_u64;
    let mut required = RequiredLayout::default();

    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|error| BootstrapError::new(format!("read wheel entry {index}: {error}")))?;
        if entry.encrypted() {
            return Err(BootstrapError::new(format!(
                "wheel entry {:?} is encrypted",
                entry.name()
            )));
        }
        if !matches!(
            entry.compression(),
            CompressionMethod::Stored | CompressionMethod::Deflated
        ) {
            return Err(BootstrapError::new(format!(
                "wheel entry {:?} uses an unadmitted compression method",
                entry.name()
            )));
        }
        if entry.is_symlink() {
            return Err(BootstrapError::new(format!(
                "wheel entry {:?} is a symbolic link",
                entry.name()
            )));
        }
        let is_directory = entry.is_dir();
        validate_unix_kind(entry.unix_mode(), is_directory, entry.name())?;
        let relative = validate_archive_path(
            entry.name_raw(),
            is_directory,
            limits.path_bytes,
            limits.path_components,
        )?;
        let portable = portable_path_key(&relative);
        if !names.insert(portable.clone()) {
            return Err(BootstrapError::new(format!(
                "wheel contains a duplicate or case-colliding path {}",
                relative.display()
            )));
        }
        for ancestor in relative.ancestors().skip(1) {
            if ancestor.as_os_str().is_empty() {
                break;
            }
            if files.contains(&portable_path_key(ancestor)) {
                return Err(BootstrapError::new(format!(
                    "wheel path {} descends through a file",
                    relative.display()
                )));
            }
        }
        if !is_directory {
            files.insert(portable);
        }

        metadata_bytes = checked_add_bound(
            metadata_bytes,
            entry.name_raw().len() as u64,
            limits.metadata_bytes,
            "wheel metadata",
        )?;
        metadata_bytes = checked_add_bound(
            metadata_bytes,
            entry.comment().len() as u64,
            limits.metadata_bytes,
            "wheel metadata",
        )?;
        metadata_bytes = checked_add_bound(
            metadata_bytes,
            entry.extra_data().map_or(0, |value| value.len()) as u64,
            limits.metadata_bytes,
            "wheel metadata",
        )?;
        if entry.size() > limits.single_file_bytes {
            return Err(BootstrapError::new(format!(
                "wheel entry {} expands beyond the single-file limit",
                relative.display()
            )));
        }
        expanded_bytes = checked_add_bound(
            expanded_bytes,
            entry.size(),
            limits.expanded_bytes,
            "wheel expansion",
        )?;

        if !is_directory && select_native_file(&relative) {
            selected_bytes = checked_add_bound(
                selected_bytes,
                entry.size(),
                limits.selected_bytes,
                "selected native runtime",
            )?;
            required.observe(&relative, artifact.target);
            plans.push(EntryPlan {
                index,
                relative: relative.clone(),
                size: entry.size(),
                executable: is_executable_native_file(&relative, entry.unix_mode()),
            });
        }
    }
    for file in &files {
        let prefix = format!("{file}/");
        if names
            .range(prefix.clone()..)
            .next()
            .is_some_and(|name| name.starts_with(&prefix))
        {
            return Err(BootstrapError::new(format!(
                "wheel contains entries below file path {file}"
            )));
        }
    }
    required.finish(artifact.target)?;

    let mut manifest_entries = Vec::with_capacity(plans.len());
    for plan in plans {
        let mut entry = archive.by_index(plan.index).map_err(|error| {
            BootstrapError::new(format!("reopen wheel entry {}: {error}", plan.index))
        })?;
        let target = destination.join(&plan.relative);
        let parent = target.parent().ok_or_else(|| {
            BootstrapError::new(format!(
                "wheel entry {} has no parent",
                plan.relative.display()
            ))
        })?;
        create_secure_directories(destination, parent)?;
        let mut output = secure_create_new(&target, 0o600)?;
        let sha256 = copy_exact(&mut entry, &mut output, plan.size, &plan.relative)?;
        output
            .sync_all()
            .map_err(|error| io_error("fsync extracted native runtime file", &target, error))?;
        fs::set_permissions(
            &target,
            fs::Permissions::from_mode(if plan.executable { 0o755 } else { 0o644 }),
        )
        .map_err(|error| io_error("set extracted native runtime permissions", &target, error))?;
        manifest_entries.push(ManifestEntry {
            relative: plan.relative,
            size: plan.size,
            sha256,
        });
    }
    ensure_no_python_library(destination)?;
    manifest_entries.sort_by_key(|entry| portable_path_key(&entry.relative));
    Ok(manifest_entries)
}

fn preflight_zip_metadata(file: &mut File, path: &Path, limits: ArchiveLimits) -> Result<()> {
    const EOCD_SIZE: usize = 22;
    const MAX_COMMENT: usize = u16::MAX as usize;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("stat wheel before ZIP preflight", path, error))?;
    if !metadata.is_file()
        || metadata.len() < EOCD_SIZE as u64
        || metadata.len() > limits.archive_bytes
    {
        return Err(BootstrapError::new(format!(
            "wheel size is outside {}..={} bytes",
            EOCD_SIZE, limits.archive_bytes
        )));
    }
    let tail_size = usize::try_from(metadata.len().min((EOCD_SIZE + MAX_COMMENT) as u64))
        .map_err(|_| BootstrapError::new("wheel tail length does not fit memory"))?;
    let tail_start = metadata.len() - tail_size as u64;
    file.seek(SeekFrom::Start(tail_start))
        .map_err(|error| io_error("seek to wheel ZIP footer", path, error))?;
    let mut tail = vec![0_u8; tail_size];
    file.read_exact(&mut tail)
        .map_err(|error| io_error("read wheel ZIP footer", path, error))?;
    let mut footer = None;
    for offset in (0..=tail.len() - EOCD_SIZE).rev() {
        if tail[offset..].starts_with(b"PK\x05\x06") {
            let comment = little_u16(&tail[offset + 20..offset + 22]) as usize;
            if offset + EOCD_SIZE + comment == tail.len() {
                footer = Some(offset);
                break;
            }
        }
    }
    let offset = footer.ok_or_else(|| {
        BootstrapError::new("wheel lacks one terminal, bounded ZIP central-directory footer")
    })?;
    let disk = little_u16(&tail[offset + 4..offset + 6]);
    let central_disk = little_u16(&tail[offset + 6..offset + 8]);
    let disk_entries = little_u16(&tail[offset + 8..offset + 10]);
    let entries = little_u16(&tail[offset + 10..offset + 12]);
    let central_size = little_u32(&tail[offset + 12..offset + 16]);
    let central_offset = little_u32(&tail[offset + 16..offset + 20]);
    if disk != 0
        || central_disk != 0
        || disk_entries != entries
        || entries == 0
        || entries == u16::MAX
        || entries as usize > limits.entries
        || central_size == u32::MAX
        || central_size as u64 > limits.metadata_bytes
        || central_offset == u32::MAX
    {
        return Err(BootstrapError::new(
            "wheel ZIP central-directory metadata is multi-disk, ZIP64, empty, or outside its bounds",
        ));
    }
    let footer_absolute = tail_start + offset as u64;
    let central_end = (central_offset as u64)
        .checked_add(central_size as u64)
        .ok_or_else(|| BootstrapError::new("wheel central-directory range overflowed"))?;
    if central_end != footer_absolute {
        return Err(BootstrapError::new(
            "wheel ZIP central directory is non-contiguous or outside the archive",
        ));
    }
    Ok(())
}

fn little_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn little_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn validate_unix_kind(mode: Option<u32>, directory: bool, name: &str) -> Result<()> {
    let Some(mode) = mode else {
        return Ok(());
    };
    match mode & 0o170000 {
        0 => Ok(()),
        0o040000 if directory => Ok(()),
        0o100000 if !directory => Ok(()),
        _ => Err(BootstrapError::new(format!(
            "wheel entry {name:?} is not a regular file/directory"
        ))),
    }
}

fn validate_archive_path(
    raw: &[u8],
    directory: bool,
    maximum_bytes: usize,
    maximum_components: usize,
) -> Result<PathBuf> {
    if raw.is_empty() || raw.len() > maximum_bytes || raw.contains(&0) || raw.contains(&b'\\') {
        return Err(BootstrapError::new("wheel contains an unsafe path"));
    }
    let value = std::str::from_utf8(raw)
        .map_err(|_| BootstrapError::new("wheel path is not strict UTF-8"))?;
    if !value.is_ascii() || value.starts_with('/') || value.starts_with('\\') {
        return Err(BootstrapError::new(format!(
            "wheel path {value:?} is not a portable relative ASCII path"
        )));
    }
    let value = if directory {
        value.strip_suffix('/').ok_or_else(|| {
            BootstrapError::new(format!("wheel directory {value:?} lacks a trailing slash"))
        })?
    } else {
        if value.ends_with('/') {
            return Err(BootstrapError::new(format!(
                "wheel file {value:?} has a trailing slash"
            )));
        }
        value
    };
    let components: Vec<_> = value.split('/').collect();
    if components.is_empty()
        || components.len() > maximum_components
        || components.iter().any(|component| {
            component.is_empty()
                || *component == "."
                || *component == ".."
                || component.len() > 255
                || component.contains(':')
        })
    {
        return Err(BootstrapError::new(format!(
            "wheel path {value:?} contains traversal or an unsafe component"
        )));
    }
    let path = PathBuf::from(value);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BootstrapError::new(format!(
            "wheel path {value:?} is not purely relative"
        )));
    }
    Ok(path)
}

fn portable_path_key(path: &Path) -> String {
    path.as_os_str().to_string_lossy().to_ascii_lowercase()
}

fn select_native_file(path: &Path) -> bool {
    let components: Vec<_> = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect();
    if components.len() >= 3 && components[0] == "torch" {
        if matches!(components[1], "include" | "share" | "bin") {
            return true;
        }
        if components[1] == "lib" {
            let filename = components.last().copied().unwrap_or_default();
            return !filename.to_ascii_lowercase().starts_with("libtorch_python");
        }
    }
    if components.len() >= 2
        && components[0].starts_with("torch-2.13.0")
        && components[0].ends_with(".dist-info")
    {
        return components[1] == "licenses"
            || (components.len() == 2
                && matches!(components[1], "LICENSE" | "METADATA" | "WHEEL"));
    }
    false
}

fn is_executable_native_file(path: &Path, unix_mode: Option<u32>) -> bool {
    let in_bin = path
        .components()
        .nth(1)
        .is_some_and(|value| value.as_os_str() == "bin");
    let archive_executable = unix_mode.is_some_and(|mode| mode & 0o111 != 0);
    in_bin
        || archive_executable
        || path
            .components()
            .nth(1)
            .is_some_and(|value| value.as_os_str() == "lib")
}

#[derive(Default)]
struct RequiredLayout {
    header: bool,
    config: bool,
    torch: bool,
    torch_cpu: bool,
    c10: bool,
}

impl RequiredLayout {
    fn observe(&mut self, path: &Path, target: PlatformTarget) {
        let value = path.to_string_lossy();
        self.header |= value == TORCH_CPP_API_HEADER;
        self.config |= value == "torch/share/cmake/Torch/TorchConfig.cmake";
        let suffix = target.library_suffix();
        self.torch |= value == format!("torch/lib/libtorch.{suffix}");
        self.torch_cpu |= value == format!("torch/lib/libtorch_cpu.{suffix}");
        self.c10 |= value == format!("torch/lib/libc10.{suffix}");
    }

    fn finish(&self, target: PlatformTarget) -> Result<()> {
        let mut missing = Vec::new();
        if !self.header {
            missing.push(TORCH_CPP_API_HEADER);
        }
        if !self.config {
            missing.push("torch/share/cmake/Torch/TorchConfig.cmake");
        }
        let suffix = target.library_suffix();
        if !self.torch {
            missing.push(if suffix == "dylib" {
                "torch/lib/libtorch.dylib"
            } else {
                "torch/lib/libtorch.so"
            });
        }
        if !self.torch_cpu {
            missing.push(if suffix == "dylib" {
                "torch/lib/libtorch_cpu.dylib"
            } else {
                "torch/lib/libtorch_cpu.so"
            });
        }
        if !self.c10 {
            missing.push(if suffix == "dylib" {
                "torch/lib/libc10.dylib"
            } else {
                "torch/lib/libc10.so"
            });
        }
        if !missing.is_empty() {
            return Err(BootstrapError::new(format!(
                "pinned wheel lacks required C++ runtime entries: {}",
                missing.join(", ")
            )));
        }
        Ok(())
    }
}

fn create_secure_directories(root: &Path, parent: &Path) -> Result<()> {
    let relative = parent.strip_prefix(root).map_err(|_| {
        BootstrapError::new(format!(
            "extraction parent {} escaped staging root {}",
            parent.display(),
            root.display()
        ))
    })?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            return Err(BootstrapError::new("unsafe extraction directory component"));
        };
        current.push(component);
        match fs::create_dir(&current) {
            Ok(()) => fs::set_permissions(&current, fs::Permissions::from_mode(0o700))
                .map_err(|error| io_error("set staging directory permissions", &current, error))?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                require_real_directory(&current)?;
            }
            Err(error) => return Err(io_error("create staging directory", &current, error)),
        }
    }
    Ok(())
}

fn copy_exact(
    reader: &mut dyn Read,
    writer: &mut dyn Write,
    expected: u64,
    path: &Path,
) -> Result<[u8; 32]> {
    let mut buffer = vec![0_u8; COPY_BUFFER];
    let mut total = 0_u64;
    let mut digest = Sha256::new();
    while total < expected {
        let remaining = (expected - total).min(buffer.len() as u64) as usize;
        let read = reader.read(&mut buffer[..remaining]).map_err(|error| {
            BootstrapError::new(format!("decompress {}: {error}", path.display()))
        })?;
        if read == 0 {
            return Err(BootstrapError::new(format!(
                "wheel entry {} ended at {total} bytes; expected {expected}",
                path.display()
            )));
        }
        writer
            .write_all(&buffer[..read])
            .map_err(|error| BootstrapError::new(format!("write {}: {error}", path.display())))?;
        digest.update(&buffer[..read]);
        total += read as u64;
    }
    let mut extra = [0_u8; 1];
    let trailing = reader
        .read(&mut extra)
        .map_err(|error| BootstrapError::new(format!("finish {}: {error}", path.display())))?;
    if trailing != 0 {
        return Err(BootstrapError::new(format!(
            "wheel entry {} exceeded its declared size",
            path.display()
        )));
    }
    Ok(digest.finalize().into())
}

fn checked_add_bound(current: u64, amount: u64, maximum: u64, label: &str) -> Result<u64> {
    let next = current
        .checked_add(amount)
        .ok_or_else(|| BootstrapError::new(format!("{label} byte count overflowed")))?;
    if next > maximum {
        return Err(BootstrapError::new(format!(
            "{label} exceeded its {maximum}-byte limit"
        )));
    }
    Ok(next)
}

fn ensure_no_python_library(root: &Path) -> Result<()> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .map_err(|error| io_error("inspect extracted runtime", &directory, error))?
        {
            let entry = entry.map_err(|error| {
                BootstrapError::new(format!("read extracted runtime directory: {error}"))
            })?;
            let file_type = entry.file_type().map_err(|error| {
                io_error("inspect extracted runtime entry", &entry.path(), error)
            })?;
            if file_type.is_symlink() {
                return Err(BootstrapError::new(format!(
                    "extracted runtime unexpectedly contains symlink {}",
                    entry.path().display()
                )));
            }
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if entry
                .file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .starts_with("libtorch_python")
            {
                return Err(BootstrapError::new(
                    "libtorch_python was selected despite the native-only policy",
                ));
            }
        }
    }
    Ok(())
}

fn files_manifest_text(entries: &[ManifestEntry]) -> Result<Vec<u8>> {
    if entries.is_empty() || entries.len() > PRODUCTION_LIMITS.entries {
        return Err(BootstrapError::new(
            "native files manifest has an invalid file count",
        ));
    }
    let mut total = 0_u64;
    let mut previous: Option<String> = None;
    let mut body = String::new();
    for entry in entries {
        let path = entry
            .relative
            .to_str()
            .ok_or_else(|| BootstrapError::new("native files manifest path is not strict UTF-8"))?;
        validate_archive_path(
            path.as_bytes(),
            false,
            PRODUCTION_LIMITS.path_bytes,
            PRODUCTION_LIMITS.path_components,
        )?;
        if !select_native_file(&entry.relative) {
            return Err(BootstrapError::new(format!(
                "native files manifest contains unselected path {path}"
            )));
        }
        let key = portable_path_key(&entry.relative);
        if previous.as_ref().is_some_and(|old| old >= &key) {
            return Err(BootstrapError::new(
                "native files manifest paths are not uniquely sorted",
            ));
        }
        previous = Some(key);
        total = checked_add_bound(
            total,
            entry.size,
            PRODUCTION_LIMITS.selected_bytes,
            "native files manifest",
        )?;
        body.push_str(&format!(
            "{}\t{}\t{}\n",
            format_digest(&entry.sha256),
            entry.size,
            path
        ));
    }
    let value = format!(
        "format={FILES_MANIFEST_FORMAT}\nfiles={}\nbytes={total}\n{body}",
        entries.len()
    )
    .into_bytes();
    if value.len() as u64 > MAX_FILES_MANIFEST_BYTES {
        return Err(BootstrapError::new(
            "native files manifest exceeded its encoded-size limit",
        ));
    }
    Ok(value)
}

fn write_files_manifest(root: &Path, entries: &[ManifestEntry]) -> Result<String> {
    let value = files_manifest_text(entries)?;
    let digest = format_digest(&Sha256::digest(&value));
    let path = root.join(FILES_MANIFEST_NAME);
    let mut file = secure_create_new(&path, 0o600)?;
    file.write_all(&value)
        .map_err(|error| io_error("write native files manifest", &path, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync native files manifest", &path, error))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644))
        .map_err(|error| io_error("set native files manifest permissions", &path, error))?;
    Ok(digest)
}

fn marker_text(artifact: &Artifact) -> String {
    format!(
        "format={TOOLCHAIN_FORMAT}\npytorch_version={PYTORCH_VERSION}\ntarget={}\nartifact_url={}\nartifact_sha256={}\nartifact_size={}\nfiles_manifest_sha256={}\ntorch_root=torch\npython_runtime=false\nlibtorch_python=false\n",
        artifact.target,
        artifact.url,
        artifact.sha256,
        artifact.size,
        artifact.files_manifest_sha256,
    )
}

fn write_marker(root: &Path, artifact: &Artifact, files_manifest_sha256: &str) -> Result<()> {
    let parsed = parse_sha256(files_manifest_sha256)?;
    if format_digest(&parsed) != files_manifest_sha256 {
        return Err(BootstrapError::new(
            "native files manifest SHA-256 is not canonical lowercase",
        ));
    }
    if files_manifest_sha256 != artifact.files_manifest_sha256 {
        return Err(BootstrapError::new(format!(
            "extracted native files manifest SHA-256 {files_manifest_sha256} does not match the exact artifact-derived pin {}",
            artifact.files_manifest_sha256
        )));
    }
    let path = root.join(MARKER_NAME);
    let mut file = secure_create_new(&path, 0o600)?;
    file.write_all(marker_text(artifact).as_bytes())
        .map_err(|error| io_error("write native toolchain marker", &path, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync native toolchain marker", &path, error))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644))
        .map_err(|error| io_error("set native toolchain marker permissions", &path, error))?;
    Ok(())
}

fn finish_tree(root: &Path) -> Result<()> {
    let mut directories = vec![root.to_path_buf()];
    let mut cursor = 0;
    while cursor < directories.len() {
        let directory = directories[cursor].clone();
        cursor += 1;
        for entry in fs::read_dir(&directory)
            .map_err(|error| io_error("read staging directory", &directory, error))?
        {
            let entry = entry.map_err(|error| {
                BootstrapError::new(format!("read staging directory entry: {error}"))
            })?;
            let kind = entry
                .file_type()
                .map_err(|error| io_error("inspect staging entry", &entry.path(), error))?;
            if kind.is_symlink() {
                return Err(BootstrapError::new("staging tree contains a symlink"));
            }
            if kind.is_dir() {
                directories.push(entry.path());
            }
        }
    }
    directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for directory in &directories {
        fs::set_permissions(directory, fs::Permissions::from_mode(0o755)).map_err(|error| {
            io_error(
                "set native toolchain directory permissions",
                directory,
                error,
            )
        })?;
        fsync_directory(directory)?;
    }
    Ok(())
}

fn validate_installed(destination: &Path, artifact: &Artifact) -> Result<Vec<PathBuf>> {
    require_real_directory(destination)?;
    require_owned_nonwritable(destination, true)?;
    let marker = destination.join(MARKER_NAME);
    let marker_value = read_regular_bounded(&marker, 4096)?;
    let manifest_digest = validate_marker(&marker_value, artifact, destination)?;
    let manifest_path = destination.join(FILES_MANIFEST_NAME);
    let manifest_value = read_regular_bounded(&manifest_path, MAX_FILES_MANIFEST_BYTES)?;
    let actual_manifest_digest = format_digest(&Sha256::digest(&manifest_value));
    if actual_manifest_digest != manifest_digest {
        return Err(BootstrapError::new(format!(
            "native files manifest in {} no longer matches its authenticated install marker",
            destination.display()
        )));
    }
    let entries = parse_files_manifest(&manifest_value, artifact.target)?;
    validate_complete_tree(destination, &entries)?;
    Ok(tracked_tree_paths(destination, &entries))
}

fn tracked_tree_paths(root: &Path, entries: &[ManifestEntry]) -> Vec<PathBuf> {
    let mut paths = BTreeSet::from([
        root.to_path_buf(),
        root.join(MARKER_NAME),
        root.join(FILES_MANIFEST_NAME),
    ]);
    for entry in entries {
        paths.insert(root.join(&entry.relative));
        for parent in entry.relative.ancestors().skip(1) {
            if parent.as_os_str().is_empty() {
                break;
            }
            paths.insert(root.join(parent));
        }
    }
    paths.into_iter().collect()
}

fn validate_marker(value: &[u8], artifact: &Artifact, destination: &Path) -> Result<String> {
    let text = std::str::from_utf8(value)
        .map_err(|_| BootstrapError::new("native toolchain marker is not strict UTF-8"))?;
    let text = text
        .strip_suffix('\n')
        .ok_or_else(|| BootstrapError::new("native toolchain marker is not newline terminated"))?;
    let lines: Vec<_> = text.split('\n').collect();
    if lines.len() != 10
        || lines[0] != format!("format={TOOLCHAIN_FORMAT}")
        || lines[1] != format!("pytorch_version={PYTORCH_VERSION}")
        || lines[2] != format!("target={}", artifact.target)
        || lines[3] != format!("artifact_url={}", artifact.url)
        || lines[4] != format!("artifact_sha256={}", artifact.sha256)
        || lines[5] != format!("artifact_size={}", artifact.size)
        || lines[6] != format!("files_manifest_sha256={}", artifact.files_manifest_sha256)
        || lines[7] != "torch_root=torch"
        || lines[8] != "python_runtime=false"
        || lines[9] != "libtorch_python=false"
    {
        return Err(BootstrapError::new(format!(
            "existing toolchain {} does not match the exact PyTorch artifact pin",
            destination.display()
        )));
    }
    let digest = lines[6]
        .strip_prefix("files_manifest_sha256=")
        .ok_or_else(|| BootstrapError::new("native toolchain marker lacks files manifest pin"))?;
    let parsed = parse_sha256(digest)?;
    if format_digest(&parsed) != digest {
        return Err(BootstrapError::new(
            "native files manifest pin is not canonical lowercase",
        ));
    }
    Ok(digest.to_owned())
}

fn parse_files_manifest(value: &[u8], target: PlatformTarget) -> Result<Vec<ManifestEntry>> {
    if value.is_empty() || value.len() as u64 > MAX_FILES_MANIFEST_BYTES || !value.ends_with(b"\n")
    {
        return Err(BootstrapError::new(
            "native files manifest is empty, oversized, or unterminated",
        ));
    }
    let text = std::str::from_utf8(value)
        .map_err(|_| BootstrapError::new("native files manifest is not strict UTF-8"))?;
    let mut lines = text.strip_suffix('\n').unwrap_or(text).split('\n');
    if lines.next() != Some(&format!("format={FILES_MANIFEST_FORMAT}")) {
        return Err(BootstrapError::new(
            "native files manifest format is invalid",
        ));
    }
    let expected_files = parse_manifest_decimal(lines.next(), "files")?;
    let expected_bytes = parse_manifest_decimal(lines.next(), "bytes")?;
    if expected_files == 0 || expected_files > PRODUCTION_LIMITS.entries as u64 {
        return Err(BootstrapError::new(
            "native files manifest file count is outside its limit",
        ));
    }
    if expected_bytes > PRODUCTION_LIMITS.selected_bytes {
        return Err(BootstrapError::new(
            "native files manifest byte count is outside its limit",
        ));
    }
    let mut entries: Vec<ManifestEntry> = Vec::with_capacity(expected_files as usize);
    let mut total = 0_u64;
    let mut names = BTreeSet::new();
    let mut required = RequiredLayout::default();
    for line in lines {
        let mut fields = line.split('\t');
        let digest = fields
            .next()
            .ok_or_else(|| BootstrapError::new("native files manifest digest is missing"))?;
        let size = fields
            .next()
            .ok_or_else(|| BootstrapError::new("native files manifest size is missing"))?;
        let path = fields
            .next()
            .ok_or_else(|| BootstrapError::new("native files manifest path is missing"))?;
        if fields.next().is_some() {
            return Err(BootstrapError::new(
                "native files manifest line has extra fields",
            ));
        }
        let sha256 = parse_sha256(digest)?;
        if format_digest(&sha256) != digest {
            return Err(BootstrapError::new(
                "native files manifest digest is not canonical lowercase",
            ));
        }
        let size = parse_canonical_decimal(size, "native file size")?;
        if size > PRODUCTION_LIMITS.single_file_bytes {
            return Err(BootstrapError::new(
                "native files manifest contains an oversized file",
            ));
        }
        let relative = validate_archive_path(
            path.as_bytes(),
            false,
            PRODUCTION_LIMITS.path_bytes,
            PRODUCTION_LIMITS.path_components,
        )?;
        if !select_native_file(&relative) || !names.insert(portable_path_key(&relative)) {
            return Err(BootstrapError::new(format!(
                "native files manifest contains an unselected or duplicate path {path}"
            )));
        }
        if let Some(previous) = entries.last() {
            if portable_path_key(&previous.relative) >= portable_path_key(&relative) {
                return Err(BootstrapError::new(
                    "native files manifest paths are not strictly sorted",
                ));
            }
        }
        total = checked_add_bound(
            total,
            size,
            PRODUCTION_LIMITS.selected_bytes,
            "native files manifest",
        )?;
        required.observe(&relative, target);
        entries.push(ManifestEntry {
            relative,
            size,
            sha256,
        });
    }
    if entries.len() as u64 != expected_files || total != expected_bytes {
        return Err(BootstrapError::new(
            "native files manifest totals do not match its entries",
        ));
    }
    required.finish(target)?;
    Ok(entries)
}

fn parse_manifest_decimal(line: Option<&str>, key: &str) -> Result<u64> {
    let line = line
        .ok_or_else(|| BootstrapError::new(format!("native files manifest lacks {key} total")))?;
    let value = line.strip_prefix(&format!("{key}=")).ok_or_else(|| {
        BootstrapError::new(format!("native files manifest {key} total is malformed"))
    })?;
    parse_canonical_decimal(value, key)
}

fn parse_canonical_decimal(value: &str, label: &str) -> Result<u64> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(BootstrapError::new(format!(
            "{label} is not a canonical decimal"
        )));
    }
    value
        .parse()
        .map_err(|_| BootstrapError::new(format!("{label} overflows u64")))
}

fn validate_complete_tree(root: &Path, entries: &[ManifestEntry]) -> Result<()> {
    let mut expected_files =
        BTreeSet::from([MARKER_NAME.to_owned(), FILES_MANIFEST_NAME.to_owned()]);
    let mut expected_directories = BTreeSet::new();
    for entry in entries {
        let key = portable_path_key(&entry.relative);
        expected_files.insert(key);
        for parent in entry.relative.ancestors().skip(1) {
            if parent.as_os_str().is_empty() {
                break;
            }
            expected_directories.insert(portable_path_key(parent));
        }
        verify_manifest_entry(root, entry)?;
    }
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        require_owned_nonwritable(&directory, true)?;
        let mut children: Vec<_> = fs::read_dir(&directory)
            .map_err(|error| io_error("read installed native toolchain", &directory, error))?
            .collect::<std::result::Result<_, _>>()
            .map_err(|error| {
                BootstrapError::new(format!("read native toolchain directory entry: {error}"))
            })?;
        children.sort_by_key(|entry| entry.file_name());
        for child in children {
            let path = child.path();
            let relative = path.strip_prefix(root).map_err(|_| {
                BootstrapError::new("installed native toolchain path escaped its root")
            })?;
            let key = portable_path_key(relative);
            let metadata = fs::symlink_metadata(&path).map_err(|error| {
                io_error("inspect installed native toolchain entry", &path, error)
            })?;
            if metadata.file_type().is_symlink() {
                return Err(BootstrapError::new(format!(
                    "installed native toolchain contains symlink {}",
                    relative.display()
                )));
            }
            if metadata.is_dir() {
                if !expected_directories.remove(&key) {
                    return Err(BootstrapError::new(format!(
                        "installed native toolchain contains unexpected directory {}",
                        relative.display()
                    )));
                }
                pending.push(path);
            } else if metadata.is_file() {
                if !expected_files.remove(&key) {
                    return Err(BootstrapError::new(format!(
                        "installed native toolchain contains unexpected file {}",
                        relative.display()
                    )));
                }
                require_owned_nonwritable(&path, false)?;
            } else {
                return Err(BootstrapError::new(format!(
                    "installed native toolchain contains special file {}",
                    relative.display()
                )));
            }
        }
    }
    if !expected_files.is_empty() || !expected_directories.is_empty() {
        return Err(BootstrapError::new(format!(
            "installed native toolchain is incomplete: {} files and {} directories are missing",
            expected_files.len(),
            expected_directories.len()
        )));
    }
    Ok(())
}

fn verify_manifest_entry(root: &Path, entry: &ManifestEntry) -> Result<()> {
    let path = root.join(&entry.relative);
    let before = fs::symlink_metadata(&path)
        .map_err(|error| io_error("inspect manifested native runtime file", &path, error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() != entry.size {
        return Err(BootstrapError::new(format!(
            "manifested native runtime file {} is missing, unsafe, or has changed size",
            entry.relative.display()
        )));
    }
    require_owned_nonwritable(&path, false)?;
    let mut file = secure_open_read(&path)?;
    let opened = file
        .metadata()
        .map_err(|error| io_error("stat opened native runtime file", &path, error))?;
    if (opened.dev(), opened.ino(), opened.len()) != (before.dev(), before.ino(), entry.size) {
        return Err(BootstrapError::new(format!(
            "native runtime file {} changed while opening",
            entry.relative.display()
        )));
    }
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; COPY_BUFFER];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| io_error("hash installed native runtime file", &path, error))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| BootstrapError::new("native runtime file byte count overflowed"))?;
        if total > entry.size {
            return Err(BootstrapError::new(format!(
                "native runtime file {} grew while hashing",
                entry.relative.display()
            )));
        }
        digest.update(&buffer[..read]);
    }
    let after = file
        .metadata()
        .map_err(|error| io_error("restat native runtime file", &path, error))?;
    if total != entry.size
        || (after.dev(), after.ino(), after.len()) != (opened.dev(), opened.ino(), entry.size)
        || <[u8; 32]>::from(digest.finalize()) != entry.sha256
    {
        return Err(BootstrapError::new(format!(
            "native runtime file {} no longer matches its authenticated extraction digest",
            entry.relative.display()
        )));
    }
    Ok(())
}

fn read_regular_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| io_error("inspect bounded native toolchain file", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > maximum {
        return Err(BootstrapError::new(format!(
            "{} is not a bounded regular native toolchain file",
            path.display()
        )));
    }
    require_owned_nonwritable(path, false)?;
    let file = secure_open_read(path)?;
    let opened = file
        .metadata()
        .map_err(|error| io_error("stat bounded native toolchain file", path, error))?;
    if (opened.dev(), opened.ino(), opened.len()) != (before.dev(), before.ino(), before.len()) {
        return Err(BootstrapError::new(format!(
            "bounded native toolchain file {} changed while opening",
            path.display()
        )));
    }
    let capacity = usize::try_from(before.len())
        .map_err(|_| BootstrapError::new("bounded native toolchain file does not fit memory"))?;
    let mut value = Vec::with_capacity(capacity);
    file.take(maximum + 1)
        .read_to_end(&mut value)
        .map_err(|error| io_error("read bounded native toolchain file", path, error))?;
    if value.len() as u64 != before.len() {
        return Err(BootstrapError::new(format!(
            "bounded native toolchain file {} changed while reading",
            path.display()
        )));
    }
    Ok(value)
}

fn require_owned_nonwritable(path: &Path, directory: bool) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| io_error("inspect native toolchain ownership", path, error))?;
    if metadata.file_type().is_symlink()
        || (directory && !metadata.is_dir())
        || (!directory && !metadata.is_file())
        || metadata.uid() != effective_uid()
        || metadata.permissions().mode() & 0o022 != 0
    {
        return Err(BootstrapError::new(format!(
            "{} is not an owner-controlled, non-group/world-writable {}",
            path.display(),
            if directory {
                "directory"
            } else {
                "regular file"
            }
        )));
    }
    Ok(())
}

fn require_real_directory(path: &Path) -> Result<()> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| io_error("inspect directory", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(BootstrapError::new(format!(
            "{} is not a real directory",
            path.display()
        )));
    }
    Ok(())
}

fn secure_create_new(path: &Path, mode: u32) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .custom_flags(open_nofollow_cloexec())
        .open(path)
        .map_err(|error| io_error("create file without following links", path, error))
}

fn secure_open_read(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(open_nofollow_cloexec())
        .open(path)
        .map_err(|error| io_error("open file without following links", path, error))
}

fn fsync_directory(path: &Path) -> Result<()> {
    require_real_directory(path)?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(open_directory_flags())
        .open(path)
        .map_err(|error| io_error("open directory for fsync", path, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync directory", path, error))
}

fn unique_file(parent: &Path, label: &str, extension: &str) -> Result<(PathBuf, File)> {
    let stem = unique_stem(label);
    for attempt in 0..128_u32 {
        let path = parent.join(format!(".{stem}-{attempt}.{extension}"));
        match secure_create_new(&path, 0o600) {
            Ok(file) => return Ok((path, file)),
            Err(_error) if path.exists() => continue,
            Err(error) => return Err(error),
        }
    }
    Err(BootstrapError::new(
        "could not allocate a unique wheel path",
    ))
}

fn unique_directory(parent: &Path, label: &str) -> Result<PathBuf> {
    let stem = unique_stem(label);
    for attempt in 0..128_u32 {
        let path = parent.join(format!(".{stem}-{attempt}"));
        match fs::create_dir(&path) {
            Ok(()) => {
                fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
                    .map_err(|error| io_error("set staging permissions", &path, error))?;
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error("create unique staging directory", &path, error)),
        }
    }
    Err(BootstrapError::new(
        "could not allocate a unique staging directory",
    ))
}

fn unique_stem(label: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "deltafin-bootstrap-{label}-{}-{timestamp}",
        std::process::id()
    )
}

struct Cleanup {
    files: Vec<PathBuf>,
    directories: Vec<PathBuf>,
}

impl Cleanup {
    fn new() -> Self {
        Self {
            files: Vec::new(),
            directories: Vec::new(),
        }
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        for path in &self.files {
            let _ = fs::remove_file(path);
        }
        for path in &self.directories {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn rename_noreplace(source: &Path, destination: &Path) -> Result<()> {
    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| BootstrapError::new("staging path contains a NUL byte"))?;
    let destination_c = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| BootstrapError::new("toolchain destination contains a NUL byte"))?;
    #[cfg(target_os = "macos")]
    // SAFETY: both retained CString values are NUL-terminated for the call.
    let status = unsafe { renamex_np(source.as_ptr(), destination_c.as_ptr(), RENAME_EXCL) };
    #[cfg(target_os = "linux")]
    // SAFETY: both retained CString values are NUL-terminated for the call.
    let status = unsafe {
        renameat2(
            AT_FDCWD,
            source.as_ptr(),
            AT_FDCWD,
            destination_c.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if status != 0 {
        return Err(io_error(
            "publish native toolchain without replacement",
            destination,
            std::io::Error::last_os_error(),
        ));
    }
    Ok(())
}

fn parse_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(BootstrapError::new("artifact SHA-256 pin is malformed"));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(BootstrapError::new("artifact SHA-256 pin is malformed")),
    }
}

fn format_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn curl_error(operation: &str, error: curl::Error) -> BootstrapError {
    BootstrapError::new(format!("{operation}: {error}"))
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> BootstrapError {
    BootstrapError::new(format!("{operation} {}: {error}", path.display()))
}

fn effective_uid() -> u32 {
    // SAFETY: geteuid has no arguments, memory effects, or failure return.
    unsafe { geteuid() }
}

unsafe extern "C" {
    fn geteuid() -> u32;
}

#[cfg(target_os = "macos")]
const fn open_nofollow_cloexec() -> i32 {
    0x0100_0100
}
#[cfg(target_os = "linux")]
const fn open_nofollow_cloexec() -> i32 {
    libc::O_NOFOLLOW | libc::O_CLOEXEC
}
#[cfg(target_os = "macos")]
const fn open_directory_flags() -> i32 {
    0x0110_0100
}
#[cfg(target_os = "linux")]
const fn open_directory_flags() -> i32 {
    libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn renamex_np(
        old: *const std::os::raw::c_char,
        new: *const std::os::raw::c_char,
        flags: u32,
    ) -> i32;
}
#[cfg(target_os = "macos")]
const RENAME_EXCL: u32 = 0x0000_0004;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn renameat2(
        olddirfd: i32,
        old: *const std::os::raw::c_char,
        newdirfd: i32,
        new: *const std::os::raw::c_char,
        flags: u32,
    ) -> i32;
}
#[cfg(target_os = "linux")]
const AT_FDCWD: i32 = -100;
#[cfg(target_os = "linux")]
const RENAME_NOREPLACE: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-bootstrap-test-{label}-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(entries: &[(&str, &[u8])], symlink: Option<(&str, &str)>) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = ZipWriter::new(&mut cursor);
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Deflated)
                .unix_permissions(0o644);
            for (name, value) in entries {
                writer.start_file(*name, options).unwrap();
                writer.write_all(value).unwrap();
            }
            if let Some((name, target)) = symlink {
                writer.add_symlink(name, target, options).unwrap();
            }
            writer.finish().unwrap();
        }
        cursor.into_inner()
    }

    fn required_entries() -> Vec<(&'static str, &'static [u8])> {
        vec![
            (TORCH_CPP_API_HEADER, b"header"),
            ("torch/share/cmake/Torch/TorchConfig.cmake", b"cmake config"),
            ("torch/lib/libtorch.dylib", b"torch"),
            ("torch/lib/libtorch_cpu.dylib", b"torch cpu"),
            ("torch/lib/libc10.dylib", b"c10"),
            ("torch/lib/libtorch_python.dylib", b"must not extract"),
            ("torch/bin/torch_shm_manager", b"native helper"),
            ("torch/__init__.py", b"must not extract"),
            (
                "torch-2.13.0.dist-info/licenses/LICENSE",
                b"upstream license",
            ),
        ]
    }

    fn duplicate_name_fixture() -> Vec<u8> {
        let mut bytes = fixture(&[("torch/lib/a", b"one"), ("torch/lib/A", b"two")], None);
        let needle = b"torch/lib/A";
        let replacement = b"torch/lib/a";
        let mut replacements = 0;
        for offset in 0..=bytes.len() - needle.len() {
            if &bytes[offset..offset + needle.len()] == needle {
                bytes[offset..offset + needle.len()].copy_from_slice(replacement);
                replacements += 1;
            }
        }
        assert!(
            replacements >= 2,
            "local and central ZIP names must be patched"
        );
        bytes
    }

    fn fake_artifact() -> Artifact {
        Artifact {
            target: PlatformTarget::MacosArm64,
            url: MACOS_ARM64.url,
            sha256: MACOS_ARM64.sha256,
            size: MACOS_ARM64.size,
            files_manifest_sha256: MACOS_ARM64.files_manifest_sha256,
        }
    }

    fn fake_artifact_with_manifest(files_manifest_sha256: String) -> Artifact {
        Artifact {
            files_manifest_sha256: Box::leak(files_manifest_sha256.into_boxed_str()),
            ..fake_artifact()
        }
    }

    #[test]
    fn official_cpu_artifacts_have_exact_safe_pins() {
        for artifact in [&MACOS_ARM64, &LINUX_X86_64, &LINUX_AARCH64] {
            validate_artifact(artifact).unwrap();
            assert_eq!(parse_sha256(artifact.sha256).unwrap().len(), 32);
            assert_eq!(
                parse_sha256(artifact.files_manifest_sha256).unwrap().len(),
                32
            );
            assert!(
                artifact
                    .url
                    .starts_with("https://download-r2.pytorch.org/whl/")
            );
        }
        assert_eq!(MACOS_ARM64.size, 111_227_066);
        assert_eq!(LINUX_X86_64.size, 191_822_516);
        assert_eq!(LINUX_AARCH64.size, 155_020_718);
    }

    #[test]
    fn linked_system_curl_meets_the_https_contract() {
        require_https_capable_libcurl().unwrap();
        require_https_capable_libcurl().unwrap();
    }

    #[test]
    fn extracts_only_native_layout_and_never_libtorch_python() {
        let temporary = TestDirectory::new("extract");
        let archive = temporary.0.join("fixture.whl");
        fs::write(&archive, fixture(&required_entries(), None)).unwrap();
        let output = temporary.0.join("output");
        fs::create_dir(&output).unwrap();
        extract_native_layout(&archive, &output, &fake_artifact(), PRODUCTION_LIMITS).unwrap();
        assert!(output.join(TORCH_CPP_API_HEADER).is_file());
        assert!(output.join("torch/lib/libtorch.dylib").is_file());
        assert!(output.join("torch/bin/torch_shm_manager").is_file());
        assert!(!output.join("torch/lib/libtorch_python.dylib").exists());
        assert!(!output.join("torch/__init__.py").exists());
        assert!(
            output
                .join("torch-2.13.0.dist-info/licenses/LICENSE")
                .is_file()
        );
    }

    #[test]
    fn requires_the_official_wheel_cpp_api_header_layout() {
        let temporary = TestDirectory::new("official-header-layout");
        let mut entries = required_entries();
        entries.retain(|(path, _)| *path != TORCH_CPP_API_HEADER);
        entries.push(("torch/include/torch/torch.h", b"non-official shortcut"));
        let archive = temporary.0.join("fixture.whl");
        fs::write(&archive, fixture(&entries, None)).unwrap();
        let output = temporary.0.join("output");
        fs::create_dir(&output).unwrap();

        let error = extract_native_layout(&archive, &output, &fake_artifact(), PRODUCTION_LIMITS)
            .unwrap_err();

        assert!(error.to_string().contains(TORCH_CPP_API_HEADER));
    }

    #[test]
    fn rejects_traversal_duplicate_case_collision_and_symlink() {
        for (label, bytes) in [
            (
                "traversal",
                fixture(&[("../torch/lib/libtorch.dylib", b"bad")], None),
            ),
            ("duplicate", duplicate_name_fixture()),
            (
                "case",
                fixture(
                    &[("torch/lib/Case", b"one"), ("torch/lib/case", b"two")],
                    None,
                ),
            ),
            (
                "symlink",
                fixture(&required_entries(), Some(("torch/lib/escape", "../../"))),
            ),
        ] {
            let temporary = TestDirectory::new(label);
            let archive = temporary.0.join("fixture.whl");
            fs::write(&archive, bytes).unwrap();
            let output = temporary.0.join("output");
            fs::create_dir(&output).unwrap();
            assert!(
                extract_native_layout(&archive, &output, &fake_artifact(), PRODUCTION_LIMITS,)
                    .is_err(),
                "fixture {label} should be rejected"
            );
        }
    }

    #[test]
    fn archive_limits_reject_metadata_and_expansion_bombs() {
        let temporary = TestDirectory::new("bounds");
        let archive = temporary.0.join("fixture.whl");
        fs::write(&archive, fixture(&required_entries(), None)).unwrap();
        for limits in [
            ArchiveLimits {
                archive_bytes: 8,
                ..PRODUCTION_LIMITS
            },
            ArchiveLimits {
                entries: 2,
                ..PRODUCTION_LIMITS
            },
            ArchiveLimits {
                metadata_bytes: 8,
                ..PRODUCTION_LIMITS
            },
            ArchiveLimits {
                expanded_bytes: 8,
                ..PRODUCTION_LIMITS
            },
            ArchiveLimits {
                selected_bytes: 8,
                ..PRODUCTION_LIMITS
            },
            ArchiveLimits {
                single_file_bytes: 4,
                ..PRODUCTION_LIMITS
            },
        ] {
            let output = temporary
                .0
                .join(format!("out-{}", limits.entries + limits.path_bytes));
            let _ = fs::remove_dir_all(&output);
            fs::create_dir(&output).unwrap();
            assert!(extract_native_layout(&archive, &output, &fake_artifact(), limits).is_err());
        }
    }

    #[test]
    fn marker_and_installed_validation_are_exact() {
        let temporary = TestDirectory::new("marker");
        let archive = temporary.0.join("fixture.whl");
        fs::write(&archive, fixture(&required_entries(), None)).unwrap();
        let output = temporary.0.join("toolchain");
        fs::create_dir(&output).unwrap();
        let entries =
            extract_native_layout(&archive, &output, &fake_artifact(), PRODUCTION_LIMITS).unwrap();
        let manifest_sha256 = write_files_manifest(&output, &entries).unwrap();
        assert!(write_marker(&output, &fake_artifact(), &manifest_sha256).is_err());
        let artifact = fake_artifact_with_manifest(manifest_sha256.clone());
        write_marker(&output, &artifact, &manifest_sha256).unwrap();
        finish_tree(&output).unwrap();
        validate_installed(&output, &artifact).unwrap();
        let forged_marker = marker_text(&artifact).replace(
            artifact.files_manifest_sha256,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        fs::write(output.join(MARKER_NAME), forged_marker).unwrap();
        assert!(validate_installed(&output, &artifact).is_err());
    }

    #[test]
    fn installed_validation_rejects_tamper_addition_and_deletion() {
        let temporary = TestDirectory::new("tree-tamper");
        let archive = temporary.0.join("fixture.whl");
        fs::write(&archive, fixture(&required_entries(), None)).unwrap();
        let output = temporary.0.join("toolchain");
        fs::create_dir(&output).unwrap();
        let entries =
            extract_native_layout(&archive, &output, &fake_artifact(), PRODUCTION_LIMITS).unwrap();
        let manifest_sha256 = write_files_manifest(&output, &entries).unwrap();
        let artifact = fake_artifact_with_manifest(manifest_sha256.clone());
        write_marker(&output, &artifact, &manifest_sha256).unwrap();
        finish_tree(&output).unwrap();
        validate_installed(&output, &artifact).unwrap();

        let library = output.join("torch/lib/libtorch.dylib");
        let original = fs::read(&library).unwrap();
        fs::write(&library, b"other").unwrap();
        assert!(validate_installed(&output, &artifact).is_err());
        fs::write(&library, &original).unwrap();
        validate_installed(&output, &artifact).unwrap();

        let unexpected = output.join("torch/lib/injected.dylib");
        fs::write(&unexpected, b"extra").unwrap();
        assert!(validate_installed(&output, &artifact).is_err());
        fs::remove_file(&unexpected).unwrap();
        validate_installed(&output, &artifact).unwrap();

        fs::remove_file(&library).unwrap();
        assert!(validate_installed(&output, &artifact).is_err());
    }

    #[test]
    fn publication_never_replaces_an_existing_directory() {
        let temporary = TestDirectory::new("publish");
        let source = temporary.0.join("source");
        let destination = temporary.0.join("destination");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        fs::write(source.join("ours"), b"ours").unwrap();
        fs::write(destination.join("theirs"), b"theirs").unwrap();
        assert!(rename_noreplace(&source, &destination).is_err());
        assert!(source.join("ours").is_file());
        assert_eq!(fs::read(destination.join("theirs")).unwrap(), b"theirs");
    }
}
