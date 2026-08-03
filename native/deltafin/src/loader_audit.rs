//! Bounded, non-executing audit of the native loader dependency closure.
//!
//! This deliberately parses Mach-O and ELF metadata itself. Calling `otool`,
//! `ldd`, or a shell would either reintroduce an interpreted production path
//! or, in `ldd`'s case, risk executing an untrusted object while inspecting it.

use std::collections::{HashSet, VecDeque};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use crate::error::{DeltafinError, Result};

const MAX_FILES: usize = 2_048;
const MAX_DEPTH: usize = 64;
const MAX_FILE_BYTES: u64 = 8 << 30;
const MAX_METADATA_BYTES: u64 = 256 << 20;
const MAX_PATHS_PER_IMAGE: usize = 65_536;
const MAX_SEARCH_DIRECTORIES: usize = 512;
const MAX_PROCESS_ENVIRONMENT_ENTRIES: usize = 8_192;
const MAX_PROCESS_ENVIRONMENT_NAME_BYTES: usize = 4_096;

/// Fail closed when the process environment asks the platform loader to add
/// or redirect native images. The executable has necessarily started by the
/// time Rust can inspect its environment, so this check complements rather
/// than replaces the on-disk transitive closure audit below. In particular,
/// benchmark children must not inherit a loader override that could make the
/// bytes executed differ from the pinned runner and its audited dependencies.
///
/// Values are deliberately neither inspected nor included in diagnostics.
pub(crate) fn reject_dynamic_loader_environment() -> Result<()> {
    validate_dynamic_loader_environment(std::env::vars_os())
}

fn validate_dynamic_loader_environment<I, K, V>(environment: I) -> Result<()>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<OsStr>,
{
    for (index, (name, _)) in environment.into_iter().enumerate() {
        if index >= MAX_PROCESS_ENVIRONMENT_ENTRIES {
            return Err(DeltafinError::new(format!(
                "process environment exceeds the bounded {MAX_PROCESS_ENVIRONMENT_ENTRIES}-entry native runtime audit"
            )));
        }
        let name = name.as_ref();
        let bytes = name.as_bytes();
        if bytes.len() > MAX_PROCESS_ENVIRONMENT_NAME_BYTES {
            return Err(DeltafinError::new(format!(
                "process environment contains a variable name longer than the bounded {MAX_PROCESS_ENVIRONMENT_NAME_BYTES}-byte native runtime limit"
            )));
        }
        if is_dynamic_loader_environment_name(name) {
            return Err(DeltafinError::new(format!(
                "dynamic-loader environment variable {:?} is forbidden for substantive native Deltafin commands; use the binary's audited embedded RPATH/RUNPATH dependencies instead",
                name,
            )));
        }
    }
    Ok(())
}

pub(crate) fn is_dynamic_loader_environment_name(name: &OsStr) -> bool {
    let name = name.as_bytes();
    name.starts_with(b"DYLD_")
        || name.starts_with(b"LD_")
        || matches!(
            name,
            b"_RLD_LIST" | b"LDR_PRELOAD" | b"LDR_LIBRARY_PATH" | b"LIBPATH" | b"SHLIB_PATH"
        )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ImageFormat {
    MachO,
    Elf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchKind {
    MachRpath,
    ElfRpath,
    ElfRunpath,
}

#[derive(Debug, Eq, PartialEq)]
struct Dependency {
    path: Vec<u8>,
    optional: bool,
}

#[derive(Debug, Eq, PartialEq)]
struct SearchPath {
    path: Vec<u8>,
    kind: SearchKind,
}

#[derive(Debug, Eq, PartialEq)]
struct LoaderImage {
    format: ImageFormat,
    dependencies: Vec<Dependency>,
    search_paths: Vec<SearchPath>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LoaderAuditPolicy {
    /// Canonical operator-controlled roots which may satisfy non-system
    /// dependencies. An empty set denotes the authenticated bootstrap build;
    /// its absolute executable rpaths become the bounded roots instead.
    controlled_roots: Vec<PathBuf>,
    operator_supplied: bool,
}

impl LoaderAuditPolicy {
    pub(crate) fn bootstrap() -> Self {
        Self::default()
    }

    pub(crate) fn operator_supplied(roots: Vec<PathBuf>) -> Self {
        Self {
            controlled_roots: roots,
            operator_supplied: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LoaderAuditReport {
    pub(crate) audited_files: usize,
    pub(crate) metadata_bytes: u64,
}

#[derive(Default)]
struct AuditBudget {
    files: usize,
    metadata_bytes: u64,
}

impl AuditBudget {
    fn start_file(&mut self, path: &Path, size: u64) -> Result<()> {
        self.files = self
            .files
            .checked_add(1)
            .ok_or_else(|| DeltafinError::new("native loader audit file counter overflowed"))?;
        if self.files > MAX_FILES {
            return Err(DeltafinError::new(format!(
                "native loader dependency closure exceeds its {MAX_FILES}-file bound at {}",
                path.display()
            )));
        }
        if size == 0 || size > MAX_FILE_BYTES {
            return Err(DeltafinError::new(format!(
                "native loader image {} has an invalid size {size}; the bounded maximum is {MAX_FILE_BYTES} bytes",
                path.display()
            )));
        }
        Ok(())
    }

    fn read(&mut self, bytes: usize) -> Result<()> {
        self.metadata_bytes = self
            .metadata_bytes
            .checked_add(bytes as u64)
            .ok_or_else(|| DeltafinError::new("native loader metadata byte count overflowed"))?;
        if self.metadata_bytes > MAX_METADATA_BYTES {
            return Err(DeltafinError::new(format!(
                "native loader dependency closure exceeds its {MAX_METADATA_BYTES}-byte metadata-read bound"
            )));
        }
        Ok(())
    }
}

struct QueueEntry {
    path: PathBuf,
    depth: usize,
    mach_rpaths: Vec<PathBuf>,
    elf_rpaths: Vec<PathBuf>,
}

/// Audit every non-system image reachable from `artifact` through the loader
/// metadata that Deltafin records on disk. System libraries are exempted by
/// platform-owned absolute roots or a narrow ELF soname allowlist; they are
/// not copied into or controlled by a LibTorch installation.
pub(crate) fn audit_loader_closure(
    artifact: &Path,
    policy: &LoaderAuditPolicy,
) -> Result<LoaderAuditReport> {
    let unresolved = fs::symlink_metadata(artifact).map_err(|error| {
        DeltafinError::new(format!(
            "inspect native runtime artifact {}: {error}",
            artifact.display()
        ))
    })?;
    if unresolved.file_type().is_symlink() || !unresolved.is_file() {
        return Err(DeltafinError::new(format!(
            "native runtime artifact must be a non-symlink regular file: {}",
            artifact.display()
        )));
    }
    let artifact = fs::canonicalize(artifact).map_err(|error| {
        DeltafinError::new(format!(
            "resolve native runtime artifact {}: {error}",
            artifact.display()
        ))
    })?;
    let executable_directory = artifact
        .parent()
        .ok_or_else(|| DeltafinError::new("native runtime artifact has no parent directory"))?
        .to_path_buf();

    let mut allowed_roots = vec![executable_directory.clone()];
    for root in &policy.controlled_roots {
        let canonical = fs::canonicalize(root).map_err(|error| {
            DeltafinError::new(format!(
                "resolve recorded native dependency root {}: {error}",
                root.display()
            ))
        })?;
        if !canonical.is_dir() {
            return Err(DeltafinError::new(format!(
                "recorded native dependency root is not a directory: {}",
                canonical.display()
            )));
        }
        push_unique(&mut allowed_roots, canonical, MAX_SEARCH_DIRECTORIES)?;
    }

    let mut budget = AuditBudget::default();
    let root_image = inspect_file(&artifact, &mut budget)?;
    audit_loader_strings(&artifact, &root_image)?;

    let root_searches = expanded_search_paths(&artifact, &artifact, &root_image, &[], &[])?;
    for directory in root_searches
        .mach
        .iter()
        .chain(root_searches.elf_rpath.iter())
        .chain(root_searches.elf_runpath.iter())
    {
        admit_root_search(directory, policy, &mut allowed_roots)?;
    }

    let mut queue = VecDeque::from([QueueEntry {
        path: artifact.clone(),
        depth: 0,
        mach_rpaths: Vec::new(),
        elf_rpaths: Vec::new(),
    }]);
    let mut root_image = Some(root_image);
    let mut visited = HashSet::new();

    while let Some(entry) = queue.pop_front() {
        if entry.depth > MAX_DEPTH {
            return Err(DeltafinError::new(format!(
                "native loader dependency closure exceeds its depth-{MAX_DEPTH} bound at {}",
                entry.path.display()
            )));
        }
        let canonical = fs::canonicalize(&entry.path).map_err(|error| {
            DeltafinError::new(format!(
                "resolve native loader image {}: {error}",
                entry.path.display()
            ))
        })?;
        if !visited.insert(canonical.clone()) {
            continue;
        }
        if !path_within_roots(&canonical, &allowed_roots) {
            return Err(DeltafinError::new(format!(
                "native loader dependency escapes every audited root: {}",
                canonical.display()
            )));
        }
        let image = if entry.depth == 0 {
            root_image
                .take()
                .expect("root loader image is consumed exactly once")
        } else {
            inspect_file(&canonical, &mut budget)?
        };
        audit_loader_strings(&canonical, &image)?;
        let searches = expanded_search_paths(
            &canonical,
            &artifact,
            &image,
            &entry.mach_rpaths,
            &entry.elf_rpaths,
        )?;
        validate_search_directories(&canonical, policy, &allowed_roots, &searches)?;

        let mut next_mach = searches.mach.clone();
        extend_unique(&mut next_mach, &entry.mach_rpaths, MAX_SEARCH_DIRECTORIES)?;
        let mut next_elf_rpath = entry.elf_rpaths.clone();
        if searches.elf_runpath.is_empty() {
            extend_unique(
                &mut next_elf_rpath,
                &searches.elf_rpath,
                MAX_SEARCH_DIRECTORIES,
            )?;
        }

        for dependency in &image.dependencies {
            if let Some(label) = forbidden_python_loader_basename(&dependency.path) {
                return Err(DeltafinError::new(format!(
                    "native loader image {} reaches forbidden {label} dependency {:?}; the native runtime must not link libpython or libtorch_python at any depth",
                    canonical.display(),
                    String::from_utf8_lossy(&dependency.path),
                )));
            }
            let Some(resolved) = resolve_dependency(
                &canonical,
                &artifact,
                image.format,
                dependency,
                &searches,
                &entry.mach_rpaths,
                &entry.elf_rpaths,
                &allowed_roots,
            )?
            else {
                continue;
            };
            queue.push_back(QueueEntry {
                path: resolved,
                depth: entry.depth + 1,
                mach_rpaths: next_mach.clone(),
                elf_rpaths: next_elf_rpath.clone(),
            });
        }
    }

    Ok(LoaderAuditReport {
        audited_files: budget.files,
        metadata_bytes: budget.metadata_bytes,
    })
}

struct ExpandedSearches {
    mach: Vec<PathBuf>,
    elf_rpath: Vec<PathBuf>,
    elf_runpath: Vec<PathBuf>,
}

fn expanded_search_paths(
    image_path: &Path,
    executable: &Path,
    image: &LoaderImage,
    inherited_mach: &[PathBuf],
    inherited_elf: &[PathBuf],
) -> Result<ExpandedSearches> {
    let mut result = ExpandedSearches {
        mach: Vec::new(),
        elf_rpath: Vec::new(),
        elf_runpath: Vec::new(),
    };
    for search in &image.search_paths {
        let pieces = if image.format == ImageFormat::Elf {
            split_elf_search_path(&search.path)?
        } else {
            vec![search.path.clone()]
        };
        for piece in pieces {
            let path = expand_loader_path(&piece, image_path, executable, image.format)?;
            let destination = match search.kind {
                SearchKind::MachRpath => &mut result.mach,
                SearchKind::ElfRpath => &mut result.elf_rpath,
                SearchKind::ElfRunpath => &mut result.elf_runpath,
            };
            push_unique(destination, path, MAX_SEARCH_DIRECTORIES)?;
        }
    }
    // These arguments are deliberately consumed by resolution rather than
    // merged here: ELF RUNPATH is not inherited, while Mach runpaths are a
    // loader stack and ELF RPATH is inherited only in the absence of RUNPATH.
    let _ = (inherited_mach, inherited_elf);
    Ok(result)
}

fn split_elf_search_path(path: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut pieces = Vec::new();
    for piece in path.split(|byte| *byte == b':') {
        if piece.is_empty() {
            return Err(DeltafinError::new(
                "ELF RPATH/RUNPATH contains an empty current-directory entry",
            ));
        }
        pieces.push(piece.to_vec());
    }
    Ok(pieces)
}

fn expand_loader_path(
    raw: &[u8],
    image_path: &Path,
    executable: &Path,
    format: ImageFormat,
) -> Result<PathBuf> {
    let loader = image_path
        .parent()
        .ok_or_else(|| DeltafinError::new("native loader image has no parent directory"))?;
    let executable_directory = executable
        .parent()
        .ok_or_else(|| DeltafinError::new("native executable has no parent directory"))?;
    let path = match format {
        ImageFormat::MachO => {
            if let Some(suffix) = raw.strip_prefix(b"@loader_path") {
                append_loader_suffix(loader, suffix)?
            } else if let Some(suffix) = raw.strip_prefix(b"@executable_path") {
                append_loader_suffix(executable_directory, suffix)?
            } else if raw.starts_with(b"@rpath") {
                return Err(DeltafinError::new(
                    "Mach-O LC_RPATH may not recursively contain @rpath",
                ));
            } else {
                PathBuf::from(OsString::from_vec(raw.to_vec()))
            }
        }
        ImageFormat::Elf => {
            let expanded = replace_origin(raw, loader.as_os_str().as_bytes())?;
            PathBuf::from(OsString::from_vec(expanded))
        }
    };
    if !path.is_absolute() {
        return Err(DeltafinError::new(format!(
            "native loader search path is relative to process state and cannot be audited safely: {:?}",
            String::from_utf8_lossy(raw)
        )));
    }
    Ok(path)
}

fn append_loader_suffix(base: &Path, suffix: &[u8]) -> Result<PathBuf> {
    if suffix.is_empty() {
        return Ok(base.to_path_buf());
    }
    let suffix = suffix.strip_prefix(b"/").ok_or_else(|| {
        DeltafinError::new("Mach-O loader token must be followed by '/' or terminate")
    })?;
    Ok(base.join(OsString::from_vec(suffix.to_vec())))
}

fn replace_origin(raw: &[u8], origin: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(raw.len().saturating_add(origin.len()));
    let mut cursor = 0;
    while cursor < raw.len() {
        if raw[cursor..].starts_with(b"${ORIGIN}") {
            output.extend_from_slice(origin);
            cursor += 9;
        } else if raw[cursor..].starts_with(b"$ORIGIN") {
            output.extend_from_slice(origin);
            cursor += 7;
        } else if raw[cursor] == b'$' {
            return Err(DeltafinError::new(format!(
                "ELF loader path contains unsupported dynamic token: {:?}",
                String::from_utf8_lossy(raw)
            )));
        } else {
            output.push(raw[cursor]);
            cursor += 1;
        }
    }
    Ok(output)
}

fn admit_root_search(
    directory: &Path,
    policy: &LoaderAuditPolicy,
    allowed_roots: &mut Vec<PathBuf>,
) -> Result<()> {
    if system_absolute_path(directory, ImageFormat::MachO)
        || system_absolute_path(directory, ImageFormat::Elf)
    {
        return Ok(());
    }
    let Ok(canonical) = fs::canonicalize(directory) else {
        // An unused rpath is harmless. A dependency that needs it will fail
        // resolution below rather than silently escaping the audit.
        return Ok(());
    };
    if !canonical.is_dir() {
        return Ok(());
    }
    if policy.operator_supplied && !path_within_roots(&canonical, allowed_roots) {
        return Err(DeltafinError::new(format!(
            "operator-supplied native build records a loader search directory outside its audited roots: {}",
            canonical.display()
        )));
    }
    push_unique(allowed_roots, canonical, MAX_SEARCH_DIRECTORIES)
}

fn validate_search_directories(
    image: &Path,
    policy: &LoaderAuditPolicy,
    allowed_roots: &[PathBuf],
    searches: &ExpandedSearches,
) -> Result<()> {
    for path in searches
        .mach
        .iter()
        .chain(searches.elf_rpath.iter())
        .chain(searches.elf_runpath.iter())
    {
        if system_absolute_path(path, ImageFormat::MachO)
            || system_absolute_path(path, ImageFormat::Elf)
        {
            continue;
        }
        if let Ok(canonical) = fs::canonicalize(path)
            && policy.operator_supplied
            && !path_within_roots(&canonical, allowed_roots)
        {
            return Err(DeltafinError::new(format!(
                "native loader image {} searches outside the operator-supplied audited roots: {}",
                image.display(),
                canonical.display()
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn resolve_dependency(
    image_path: &Path,
    executable: &Path,
    format: ImageFormat,
    dependency: &Dependency,
    searches: &ExpandedSearches,
    inherited_mach: &[PathBuf],
    inherited_elf: &[PathBuf],
    allowed_roots: &[PathBuf],
) -> Result<Option<PathBuf>> {
    let raw = &dependency.path;
    let raw_path = PathBuf::from(OsString::from_vec(raw.clone()));
    if raw_path.is_absolute() {
        if system_absolute_path(&raw_path, format) {
            return Ok(None);
        }
        return resolve_candidate(&raw_path, dependency, allowed_roots, image_path);
    }

    let mut candidates = Vec::new();
    match format {
        ImageFormat::MachO => {
            if let Some(suffix) = raw.strip_prefix(b"@loader_path") {
                candidates.push(append_loader_suffix(
                    image_path.parent().expect("audited image parent"),
                    suffix,
                )?);
            } else if let Some(suffix) = raw.strip_prefix(b"@executable_path") {
                candidates.push(append_loader_suffix(
                    executable.parent().expect("audited executable parent"),
                    suffix,
                )?);
            } else if let Some(suffix) = raw.strip_prefix(b"@rpath") {
                let suffix = suffix.strip_prefix(b"/").ok_or_else(|| {
                    DeltafinError::new("Mach-O @rpath dependency lacks a '/' suffix")
                })?;
                for directory in searches.mach.iter().chain(inherited_mach.iter()) {
                    candidates.push(directory.join(OsString::from_vec(suffix.to_vec())));
                }
            } else {
                return Err(DeltafinError::new(format!(
                    "Mach-O dependency uses an unauditable relative path in {}: {:?}",
                    image_path.display(),
                    String::from_utf8_lossy(raw)
                )));
            }
        }
        ImageFormat::Elf => {
            if raw.contains(&b'/') {
                let expanded = replace_origin(
                    raw,
                    image_path
                        .parent()
                        .expect("audited image parent")
                        .as_os_str()
                        .as_bytes(),
                )?;
                let expanded = PathBuf::from(OsString::from_vec(expanded));
                if !expanded.is_absolute() {
                    return Err(DeltafinError::new(format!(
                        "ELF dependency uses an unauditable relative path in {}: {:?}",
                        image_path.display(),
                        String::from_utf8_lossy(raw)
                    )));
                }
                candidates.push(expanded);
            } else {
                let active = if searches.elf_runpath.is_empty() {
                    searches
                        .elf_rpath
                        .iter()
                        .chain(inherited_elf.iter())
                        .collect::<Vec<_>>()
                } else {
                    searches.elf_runpath.iter().collect::<Vec<_>>()
                };
                for directory in active {
                    candidates.push(directory.join(OsString::from_vec(raw.clone())));
                }
            }
        }
    }

    for candidate in candidates {
        if fs::symlink_metadata(&candidate).is_ok() {
            return resolve_candidate(&candidate, dependency, allowed_roots, image_path);
        }
    }
    if dependency.optional {
        return Ok(None);
    }
    if format == ImageFormat::Elf && system_elf_soname(raw) {
        return Ok(None);
    }
    Err(DeltafinError::new(format!(
        "cannot prove the native dependency closure: {} references unresolved non-system dependency {:?}",
        image_path.display(),
        String::from_utf8_lossy(raw)
    )))
}

fn resolve_candidate(
    candidate: &Path,
    dependency: &Dependency,
    allowed_roots: &[PathBuf],
    image_path: &Path,
) -> Result<Option<PathBuf>> {
    let metadata = match fs::symlink_metadata(candidate) {
        Ok(metadata) => metadata,
        Err(error) if dependency.optional && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => {
            return Err(DeltafinError::new(format!(
                "inspect dependency {} referenced by {}: {error}",
                candidate.display(),
                image_path.display()
            )));
        }
    };
    if !metadata.is_file() && !metadata.file_type().is_symlink() {
        return Err(DeltafinError::new(format!(
            "native dependency is not a regular file: {}",
            candidate.display()
        )));
    }
    let canonical = fs::canonicalize(candidate).map_err(|error| {
        DeltafinError::new(format!(
            "resolve native dependency {}: {error}",
            candidate.display()
        ))
    })?;
    if !canonical.is_file() || !path_within_roots(&canonical, allowed_roots) {
        return Err(DeltafinError::new(format!(
            "native dependency {} escapes every audited root (resolved to {})",
            candidate.display(),
            canonical.display()
        )));
    }
    Ok(Some(canonical))
}

fn system_absolute_path(path: &Path, format: ImageFormat) -> bool {
    match format {
        ImageFormat::MachO => path.starts_with("/System/Library") || path.starts_with("/usr/lib"),
        ImageFormat::Elf => {
            path.starts_with("/lib")
                || path.starts_with("/lib64")
                || path.starts_with("/usr/lib")
                || path.starts_with("/usr/lib64")
        }
    }
}

fn system_elf_soname(path: &[u8]) -> bool {
    let basename = path.rsplit(|byte| *byte == b'/').next().unwrap_or(path);
    let normal = [
        b"libc.so".as_slice(),
        b"libdl.so".as_slice(),
        b"libgcc_s.so".as_slice(),
        b"libm.so".as_slice(),
        b"libpthread.so".as_slice(),
        b"librt.so".as_slice(),
        b"libstdc++.so".as_slice(),
        b"libutil.so".as_slice(),
        b"libresolv.so".as_slice(),
        b"libgomp.so".as_slice(),
        b"libnuma.so".as_slice(),
        // Authenticated downloads link the distribution's libcurl by design:
        // deltafin-curl-sys-direct selects the system library and requires the
        // exact libcurl.so.4 SONAME. Apple resolves it from an allowed system
        // root, so only the glibc closure reaches this allowlist.
        b"libcurl.so".as_slice(),
        b"libcuda.so".as_slice(),
        b"libnvidia-ml.so".as_slice(),
        b"libz.so".as_slice(),
    ]
    .iter()
    .any(|stem| versioned_soname(basename, stem));
    normal
        || basename == b"linux-vdso.so.1"
        || (basename.starts_with(b"ld-linux-")
            && basename.windows(4).any(|window| window == b".so."))
}

fn versioned_soname(name: &[u8], stem: &[u8]) -> bool {
    name == stem
        || name.strip_prefix(stem).is_some_and(|suffix| {
            suffix.len() >= 2
                && suffix[0] == b'.'
                && suffix[1..]
                    .iter()
                    .all(|byte| byte.is_ascii_digit() || *byte == b'.')
        })
}

fn audit_loader_strings(image: &Path, metadata: &LoaderImage) -> Result<()> {
    for path in metadata
        .dependencies
        .iter()
        .map(|dependency| dependency.path.as_slice())
        .chain(
            metadata
                .search_paths
                .iter()
                .map(|search| search.path.as_slice()),
        )
    {
        for (needle, label) in [
            (b"/venv/".as_slice(), "venv"),
            (b"/.venv/".as_slice(), "dot-venv"),
            (b"/site-packages/".as_slice(), "site-packages"),
        ] {
            if path.windows(needle.len()).any(|window| window == needle) {
                return Err(DeltafinError::new(format!(
                    "native loader image {} embeds a forbidden {label} Python-environment path",
                    image.display()
                )));
            }
        }
    }
    Ok(())
}

/// Match the final component as bytes so malformed UTF-8 cannot evade the
/// native-only policy. Backslash is also a separator for fail-closed handling
/// of accidentally cross-generated paths.
fn forbidden_python_loader_basename(path: &[u8]) -> Option<&'static str> {
    let basename = path
        .rsplit(|byte| matches!(byte, b'/' | b'\\'))
        .next()
        .unwrap_or(path);
    for (prefix, label) in [
        (b"libtorch_python".as_slice(), "libtorch_python"),
        (b"libpython".as_slice(), "libpython"),
    ] {
        if basename
            .get(..prefix.len())
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
        {
            return Some(label);
        }
    }
    None
}

fn path_within_roots(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| path.starts_with(root))
}

fn push_unique(values: &mut Vec<PathBuf>, value: PathBuf, bound: usize) -> Result<()> {
    if !values.contains(&value) {
        if values.len() >= bound {
            return Err(DeltafinError::new(format!(
                "native loader search set exceeds its {bound}-directory bound"
            )));
        }
        values.push(value);
    }
    Ok(())
}

fn extend_unique(values: &mut Vec<PathBuf>, added: &[PathBuf], bound: usize) -> Result<()> {
    for value in added {
        push_unique(values, value.clone(), bound)?;
    }
    Ok(())
}

trait RegionReader {
    fn len(&self) -> u64;
    fn read_region(&mut self, offset: u64, size: usize, label: &str) -> Result<Vec<u8>>;
}

struct FileRegions<'a> {
    file: File,
    len: u64,
    path: &'a Path,
    budget: &'a mut AuditBudget,
}

impl RegionReader for FileRegions<'_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_region(&mut self, offset: u64, size: usize, label: &str) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(size as u64)
            .filter(|end| *end <= self.len)
            .ok_or_else(|| {
                DeltafinError::new(format!(
                    "{label} exceeds native loader image {}",
                    self.path.display()
                ))
            })?;
        let _ = end;
        self.budget.read(size)?;
        self.file.seek(SeekFrom::Start(offset)).map_err(|error| {
            DeltafinError::new(format!(
                "seek to {label} in {}: {error}",
                self.path.display()
            ))
        })?;
        let mut bytes = vec![0; size];
        self.file.read_exact(&mut bytes).map_err(|error| {
            DeltafinError::new(format!(
                "read {label} from {}: {error}",
                self.path.display()
            ))
        })?;
        Ok(bytes)
    }
}

#[cfg(test)]
struct SliceRegions<'a> {
    bytes: &'a [u8],
}

#[cfg(test)]
impl RegionReader for SliceRegions<'_> {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_region(&mut self, offset: u64, size: usize, label: &str) -> Result<Vec<u8>> {
        let start = usize::try_from(offset)
            .map_err(|_| DeltafinError::new(format!("{label} offset exceeds usize")))?;
        let end = start
            .checked_add(size)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| DeltafinError::new(format!("{label} exceeds fixture bytes")))?;
        Ok(self.bytes[start..end].to_vec())
    }
}

fn inspect_file(path: &Path, budget: &mut AuditBudget) -> Result<LoaderImage> {
    let file = File::open(path).map_err(|error| {
        DeltafinError::new(format!(
            "open native loader image {}: {error}",
            path.display()
        ))
    })?;
    let metadata = file.metadata().map_err(|error| {
        DeltafinError::new(format!(
            "inspect native loader image {}: {error}",
            path.display()
        ))
    })?;
    budget.start_file(path, metadata.len())?;
    let mut regions = FileRegions {
        file,
        len: metadata.len(),
        path,
        budget,
    };
    parse_loader_image(&mut regions)
}

#[cfg(test)]
fn inspect_bytes(bytes: &[u8]) -> Result<LoaderImage> {
    parse_loader_image(&mut SliceRegions { bytes })
}

fn parse_loader_image(reader: &mut impl RegionReader) -> Result<LoaderImage> {
    if reader.len() < 4 {
        return Err(DeltafinError::new("native loader image is truncated"));
    }
    let magic = reader.read_region(0, 4, "native image magic")?;
    match magic.as_slice() {
        [0xcf, 0xfa, 0xed, 0xfe] => parse_macho(reader),
        [0x7f, b'E', b'L', b'F'] => parse_elf(reader),
        _ => Err(DeltafinError::new(
            "native loader image is neither supported little-endian 64-bit Mach-O nor ELF",
        )),
    }
}

fn parse_macho(reader: &mut impl RegionReader) -> Result<LoaderImage> {
    const HEADER_BYTES: usize = 32;
    const LC_RPATH: u32 = 0x8000_001c;
    const LC_LOAD_DYLIB: u32 = 0x0000_000c;
    const LC_LOAD_WEAK_DYLIB: u32 = 0x8000_0018;
    const LC_REEXPORT_DYLIB: u32 = 0x8000_001f;
    const LC_LAZY_LOAD_DYLIB: u32 = 0x0000_0020;
    const LC_LOAD_UPWARD_DYLIB: u32 = 0x8000_0023;
    const DEPENDENCIES: [u32; 5] = [
        LC_LOAD_DYLIB,
        LC_LOAD_WEAK_DYLIB,
        LC_REEXPORT_DYLIB,
        LC_LAZY_LOAD_DYLIB,
        LC_LOAD_UPWARD_DYLIB,
    ];
    let header = reader.read_region(0, HEADER_BYTES, "Mach-O header")?;
    let commands = read_u32(&header, 16, "Mach-O load-command count")? as usize;
    let command_bytes = read_u32(&header, 20, "Mach-O load-command bytes")? as usize;
    if commands > MAX_PATHS_PER_IMAGE || command_bytes > (16 << 20) {
        return Err(DeltafinError::new(
            "Mach-O load-command table exceeds its audit bound",
        ));
    }
    let table = reader.read_region(HEADER_BYTES as u64, command_bytes, "Mach-O commands")?;
    let mut cursor = 0;
    let mut dependencies = Vec::new();
    let mut search_paths = Vec::new();
    for _ in 0..commands {
        let command = read_u32(&table, cursor, "Mach-O load command")?;
        let size = read_u32(&table, cursor + 4, "Mach-O load-command size")? as usize;
        let end = cursor
            .checked_add(size)
            .filter(|end| size >= 8 && size % 8 == 0 && *end <= table.len())
            .ok_or_else(|| DeltafinError::new("Mach-O load command has an invalid size"))?;
        if command == LC_RPATH || DEPENDENCIES.contains(&command) {
            let offset = read_u32(&table, cursor + 8, "Mach-O loader-path offset")? as usize;
            if offset < 12 || offset >= size {
                return Err(DeltafinError::new(
                    "Mach-O loader-path command has an invalid string offset",
                ));
            }
            let path = read_nul_path(&table[cursor + offset..end], "Mach-O")?;
            if command == LC_RPATH {
                search_paths.push(SearchPath {
                    path,
                    kind: SearchKind::MachRpath,
                });
            } else {
                dependencies.push(Dependency {
                    path,
                    optional: matches!(command, LC_LOAD_WEAK_DYLIB | LC_LAZY_LOAD_DYLIB),
                });
            }
        }
        cursor = end;
    }
    if cursor != table.len() {
        return Err(DeltafinError::new(
            "Mach-O load-command count and byte size disagree",
        ));
    }
    Ok(LoaderImage {
        format: ImageFormat::MachO,
        dependencies,
        search_paths,
    })
}

#[derive(Clone, Copy)]
struct ElfSegment {
    kind: u32,
    offset: u64,
    virtual_address: u64,
    file_bytes: u64,
}

fn parse_elf(reader: &mut impl RegionReader) -> Result<LoaderImage> {
    const HEADER_BYTES: usize = 64;
    const PROGRAM_HEADER_BYTES: usize = 56;
    const PT_LOAD: u32 = 1;
    const PT_DYNAMIC: u32 = 2;
    const DT_NEEDED: i64 = 1;
    const DT_STRTAB: i64 = 5;
    const DT_STRSZ: i64 = 10;
    const DT_RPATH: i64 = 15;
    const DT_RUNPATH: i64 = 29;

    let header = reader.read_region(0, HEADER_BYTES, "ELF header")?;
    if header.get(4) != Some(&2) || header.get(5) != Some(&1) {
        return Err(DeltafinError::new("ELF image is not little-endian ELF64"));
    }
    let table_offset = read_u64(&header, 32, "ELF program-header offset")?;
    let entry_bytes = read_u16(&header, 54, "ELF program-header size")? as usize;
    let entries = read_u16(&header, 56, "ELF program-header count")? as usize;
    if !(PROGRAM_HEADER_BYTES..=4096).contains(&entry_bytes) || entries > 4096 {
        return Err(DeltafinError::new(
            "ELF program-header table exceeds its audit bounds",
        ));
    }
    let table_bytes = entry_bytes
        .checked_mul(entries)
        .ok_or_else(|| DeltafinError::new("ELF program-header table size overflowed"))?;
    let table = reader.read_region(table_offset, table_bytes, "ELF program headers")?;
    let mut segments = Vec::with_capacity(entries);
    for index in 0..entries {
        let start = index * entry_bytes;
        segments.push(ElfSegment {
            kind: read_u32(&table, start, "ELF segment type")?,
            offset: read_u64(&table, start + 8, "ELF segment offset")?,
            virtual_address: read_u64(&table, start + 16, "ELF segment address")?,
            file_bytes: read_u64(&table, start + 32, "ELF segment file size")?,
        });
    }
    let Some(dynamic) = segments.iter().find(|segment| segment.kind == PT_DYNAMIC) else {
        return Ok(LoaderImage {
            format: ImageFormat::Elf,
            dependencies: Vec::new(),
            search_paths: Vec::new(),
        });
    };
    let dynamic_size = usize::try_from(dynamic.file_bytes)
        .ok()
        .filter(|size| *size <= (16 << 20) && *size % 16 == 0)
        .ok_or_else(|| DeltafinError::new("ELF dynamic table has an invalid bounded size"))?;
    let dynamic_bytes = reader.read_region(dynamic.offset, dynamic_size, "ELF dynamic table")?;
    let mut string_address = None;
    let mut string_bytes = None;
    let mut offsets = Vec::new();
    let mut terminated = false;
    for entry in dynamic_bytes.chunks_exact(16) {
        let tag = i64::from_le_bytes(entry[..8].try_into().unwrap());
        let value = u64::from_le_bytes(entry[8..].try_into().unwrap());
        match tag {
            0 => {
                terminated = true;
                break;
            }
            DT_STRTAB => string_address = Some(value),
            DT_STRSZ => string_bytes = Some(value),
            DT_NEEDED => offsets.push((value, Some(false))),
            DT_RPATH => offsets.push((value, None)),
            DT_RUNPATH => offsets.push((value, Some(true))),
            _ => {}
        }
    }
    if !terminated {
        return Err(DeltafinError::new("ELF dynamic table is not terminated"));
    }
    let Some(string_address) = string_address else {
        return if offsets.is_empty() {
            Ok(LoaderImage {
                format: ImageFormat::Elf,
                dependencies: Vec::new(),
                search_paths: Vec::new(),
            })
        } else {
            Err(DeltafinError::new("ELF dynamic table lacks DT_STRTAB"))
        };
    };
    let string_bytes = string_bytes
        .and_then(|value| usize::try_from(value).ok())
        .filter(|size| *size <= (64 << 20))
        .ok_or_else(|| DeltafinError::new("ELF DT_STRSZ exceeds its audit bound"))?;
    let string_offset = segments
        .iter()
        .filter(|segment| segment.kind == PT_LOAD)
        .find_map(|segment| {
            let relative = string_address.checked_sub(segment.virtual_address)?;
            (relative.checked_add(string_bytes as u64)? <= segment.file_bytes)
                .then(|| segment.offset.checked_add(relative))
                .flatten()
        })
        .ok_or_else(|| DeltafinError::new("ELF DT_STRTAB is outside a file-backed segment"))?;
    let strings = reader.read_region(string_offset, string_bytes, "ELF dynamic strings")?;
    let mut dependencies = Vec::new();
    let mut search_paths = Vec::new();
    for (offset, kind) in offsets {
        let offset = usize::try_from(offset)
            .ok()
            .filter(|offset| *offset < strings.len())
            .ok_or_else(|| DeltafinError::new("ELF loader-path offset exceeds DT_STRSZ"))?;
        let path = read_nul_path(&strings[offset..], "ELF")?;
        match kind {
            Some(false) => dependencies.push(Dependency {
                path,
                optional: false,
            }),
            Some(true) => search_paths.push(SearchPath {
                path,
                kind: SearchKind::ElfRunpath,
            }),
            None => search_paths.push(SearchPath {
                path,
                kind: SearchKind::ElfRpath,
            }),
        }
    }
    Ok(LoaderImage {
        format: ImageFormat::Elf,
        dependencies,
        search_paths,
    })
}

fn read_nul_path(bytes: &[u8], format: &str) -> Result<Vec<u8>> {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| DeltafinError::new(format!("{format} loader path is not terminated")))?;
    if end == 0 || end > 4096 {
        return Err(DeltafinError::new(format!(
            "{format} loader path has an invalid bounded length"
        )));
    }
    Ok(bytes[..end].to_vec())
}

fn read_u16(bytes: &[u8], offset: usize, label: &str) -> Result<u16> {
    let value = read_integer(bytes, offset, 2, label)?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize, label: &str) -> Result<u32> {
    let value = read_integer(bytes, offset, 4, label)?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize, label: &str) -> Result<u64> {
    let value = read_integer(bytes, offset, 8, label)?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn read_integer<'a>(bytes: &'a [u8], offset: usize, size: usize, label: &str) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| DeltafinError::new(format!("{label} offset overflowed")))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| DeltafinError::new(format!("{label} is truncated")))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static SERIAL: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "deltafin-loader-audit-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            Self(fs::canonicalize(root).unwrap())
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn macho(commands: &[(u32, &[u8])]) -> Vec<u8> {
        let mut table = Vec::new();
        for (command, path) in commands {
            let path_offset = if *command == 0x8000_001c { 12 } else { 24 };
            let size = (path_offset + path.len() + 1 + 7) & !7;
            let start = table.len();
            table.resize(start + size, 0);
            table[start..start + 4].copy_from_slice(&command.to_le_bytes());
            table[start + 4..start + 8].copy_from_slice(&(size as u32).to_le_bytes());
            table[start + 8..start + 12].copy_from_slice(&(path_offset as u32).to_le_bytes());
            table[start + path_offset..start + path_offset + path.len()].copy_from_slice(path);
        }
        let mut bytes = vec![0; 32];
        bytes[..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]);
        bytes[16..20].copy_from_slice(&(commands.len() as u32).to_le_bytes());
        bytes[20..24].copy_from_slice(&(table.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&table);
        bytes
    }

    fn write_macho(path: &Path, commands: &[(u32, &[u8])]) {
        fs::write(path, macho(commands)).unwrap();
    }

    fn elf(dependencies: &[&[u8]], search: Option<(i64, &[u8])>) -> Vec<u8> {
        const HEADER_BYTES: usize = 64;
        const PROGRAM_BYTES: usize = 56;
        const PROGRAMS: usize = 2;
        const BASE: u64 = 0x0040_0000;

        let mut strings = vec![0_u8];
        let mut records = Vec::new();
        for dependency in dependencies {
            let offset = strings.len() as u64;
            strings.extend_from_slice(dependency);
            strings.push(0);
            records.push((1_i64, offset));
        }
        if let Some((tag, path)) = search {
            let offset = strings.len() as u64;
            strings.extend_from_slice(path);
            strings.push(0);
            records.push((tag, offset));
        }
        let dynamic_offset = HEADER_BYTES + PROGRAM_BYTES * PROGRAMS;
        let dynamic_entries = 2 + records.len() + 1;
        let dynamic_bytes = dynamic_entries * 16;
        let string_offset = dynamic_offset + dynamic_bytes;
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

        let mut dynamic_records = vec![
            (5_i64, BASE + string_offset as u64),
            (10_i64, strings.len() as u64),
        ];
        dynamic_records.extend(records);
        dynamic_records.push((0, 0));
        for (index, (tag, value)) in dynamic_records.into_iter().enumerate() {
            let entry = dynamic_offset + index * 16;
            bytes[entry..entry + 8].copy_from_slice(&tag.to_le_bytes());
            bytes[entry + 8..entry + 16].copy_from_slice(&value.to_le_bytes());
        }
        bytes[string_offset..].copy_from_slice(&strings);
        bytes
    }

    fn write_elf(path: &Path, dependencies: &[&[u8]], search: Option<(i64, &[u8])>) {
        fs::write(path, elf(dependencies, search)).unwrap();
    }

    #[test]
    fn follows_macho_rpath_transitively_and_rejects_python_grandchild() {
        let fixture = Fixture::new();
        let lib = fixture.0.join("lib");
        fs::create_dir(&lib).unwrap();
        let executable = fixture.0.join("deltafin");
        write_macho(
            &executable,
            &[
                (0x8000_001c, lib.as_os_str().as_bytes()),
                (0x0000_000c, b"@rpath/libtorch.dylib"),
            ],
        );
        write_macho(
            &lib.join("libtorch.dylib"),
            &[
                (0x8000_001c, b"@loader_path"),
                (0x0000_000c, b"@rpath/libmiddle.dylib"),
            ],
        );
        write_macho(
            &lib.join("libmiddle.dylib"),
            &[(0x0000_000c, b"@rpath/libtorch_python.dylib")],
        );

        let error = audit_loader_closure(&executable, &LoaderAuditPolicy::bootstrap()).unwrap_err();

        assert!(error.to_string().contains("at any depth"));
        assert!(error.to_string().contains("libtorch_python.dylib"));
    }

    #[test]
    fn follows_cycles_once_and_accepts_platform_system_libraries() {
        let fixture = Fixture::new();
        let lib = fixture.0.join("lib");
        fs::create_dir(&lib).unwrap();
        let executable = fixture.0.join("deltafin");
        write_macho(
            &executable,
            &[
                (0x8000_001c, lib.as_os_str().as_bytes()),
                (0x0000_000c, b"@rpath/liba.dylib"),
                (0x0000_000c, b"/usr/lib/libSystem.B.dylib"),
            ],
        );
        write_macho(
            &lib.join("liba.dylib"),
            &[
                (0x8000_001c, b"@loader_path"),
                (0x0000_000c, b"@rpath/libb.dylib"),
            ],
        );
        write_macho(
            &lib.join("libb.dylib"),
            &[
                (0x8000_001c, b"@loader_path"),
                (0x0000_000c, b"@rpath/liba.dylib"),
            ],
        );

        let report = audit_loader_closure(&executable, &LoaderAuditPolicy::bootstrap()).unwrap();

        assert_eq!(report.audited_files, 3);
        assert!(report.metadata_bytes < 1 << 20);
    }

    #[test]
    fn follows_elf_origin_runpath_and_rpath_to_python_grandchild() {
        let fixture = Fixture::new();
        let lib = fixture.0.join("lib");
        fs::create_dir(&lib).unwrap();
        let executable = fixture.0.join("deltafin");
        write_elf(
            &executable,
            &[b"libtorch.so", b"libc.so.6"],
            Some((29, b"$ORIGIN/lib")),
        );
        write_elf(
            &lib.join("libtorch.so"),
            &[b"libmiddle.so"],
            Some((15, b"${ORIGIN}")),
        );
        write_elf(
            &lib.join("libmiddle.so"),
            &[b"libtorch_python.so.2.13"],
            None,
        );

        let error = audit_loader_closure(&executable, &LoaderAuditPolicy::bootstrap()).unwrap_err();

        assert!(error.to_string().contains("at any depth"));
        assert!(error.to_string().contains("libtorch_python.so.2.13"));
    }

    #[test]
    fn unresolved_non_system_elf_dependency_fails_closed() {
        let fixture = Fixture::new();
        let executable = fixture.0.join("deltafin");
        write_elf(
            &executable,
            &[b"libnot-a-platform-library.so"],
            Some((29, b"$ORIGIN/lib")),
        );

        let error = audit_loader_closure(&executable, &LoaderAuditPolicy::bootstrap()).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unresolved non-system dependency")
        );
    }

    #[cfg(unix)]
    #[test]
    fn glibc_system_libcurl_satisfies_the_dependency_closure() {
        // deltafin-curl-sys-direct links the distribution's libcurl and
        // requires the exact libcurl.so.4 SONAME, so the ELF closure must
        // accept it as a platform library rather than an unaudited payload.
        let fixture = Fixture::new();
        let executable = fixture.0.join("deltafin");
        write_elf(&executable, &[b"libcurl.so.4"], None);

        audit_loader_closure(&executable, &LoaderAuditPolicy::bootstrap())
            .expect("system libcurl must satisfy the ELF dependency closure");
    }

    #[cfg(unix)]
    #[test]
    fn operator_root_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::new();
        let root = fixture.0.join("torch");
        let lib = root.join("lib");
        fs::create_dir_all(&lib).unwrap();
        let outside = fixture.0.join("outside.dylib");
        write_macho(&outside, &[]);
        symlink(&outside, lib.join("libtorch.dylib")).unwrap();
        let bin = fixture.0.join("bin");
        fs::create_dir(&bin).unwrap();
        let executable = bin.join("deltafin");
        write_macho(
            &executable,
            &[
                (0x8000_001c, lib.as_os_str().as_bytes()),
                (0x0000_000c, b"@rpath/libtorch.dylib"),
            ],
        );

        let error = audit_loader_closure(
            &executable,
            &LoaderAuditPolicy::operator_supplied(vec![root]),
        )
        .unwrap_err();

        assert!(error.to_string().contains("escapes every audited root"));
    }

    #[test]
    fn parser_rejects_relative_and_unknown_dynamic_search_tokens() {
        let fixture = Fixture::new();
        let executable = fixture.0.join("deltafin");
        write_macho(&executable, &[(0x8000_001c, b"relative/lib")]);
        assert!(
            audit_loader_closure(&executable, &LoaderAuditPolicy::bootstrap())
                .unwrap_err()
                .to_string()
                .contains("relative to process state")
        );

        assert!(replace_origin(b"${LIB}/thing", b"/tmp").is_err());
    }

    #[test]
    fn byte_parser_handles_real_test_executable_without_reading_whole_file() {
        let executable = std::env::current_exe().unwrap();
        let bytes = fs::read(executable).unwrap();
        let image = inspect_bytes(&bytes).unwrap();
        assert!(!image.dependencies.is_empty());
    }

    #[test]
    fn python_basename_is_case_insensitive_and_component_bounded() {
        assert_eq!(
            forbidden_python_loader_basename(b"@rpath/LIBPYTHON3.14.dylib"),
            Some("libpython")
        );
        assert_eq!(
            forbidden_python_loader_basename(b"C:\\native\\libtorch_python.dll"),
            Some("libtorch_python")
        );
        assert_eq!(
            forbidden_python_loader_basename(b"/libpython-dir/libtorch_cpu.so"),
            None
        );
    }

    #[test]
    fn runtime_environment_rejects_loader_injection_without_reading_values() {
        for name in [
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "LD_AUDIT",
            "_RLD_LIST",
            "LDR_PRELOAD",
            "LIBPATH",
            "SHLIB_PATH",
        ] {
            let error = validate_dynamic_loader_environment([(name, "never-print-this")])
                .unwrap_err()
                .to_string();
            assert!(error.contains(name), "missing rejected name in {error}");
            assert!(!error.contains("never-print-this"));
        }

        validate_dynamic_loader_environment([
            ("PATH", "/usr/bin"),
            ("CUDA_VISIBLE_DEVICES", "0"),
            ("DELTAFIN_TORCH_ROOT", "/native/libtorch"),
        ])
        .unwrap();
    }

    #[test]
    fn runtime_environment_audit_is_case_sensitive_and_bounded() {
        validate_dynamic_loader_environment([("ld_preload", "ignored-on-Unix")]).unwrap();

        let oversized = "X".repeat(MAX_PROCESS_ENVIRONMENT_NAME_BYTES + 1);
        let error = validate_dynamic_loader_environment([(oversized, String::new())])
            .unwrap_err()
            .to_string();
        assert!(error.contains("variable name longer"));
    }
}
