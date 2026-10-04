//! Native, fail-closed installer for the pinned, data-only K3 DSpark checkpoint.
//!
//! Network access is delegated to an in-process libcurl transport. Every byte
//! accepted from it is independently bounded, schema checked, length pinned,
//! and SHA-256 pinned here.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::cli::SetupDSparkArgs;
use crate::dspark_checkpoint::{
    CHECKPOINT_BASENAME, DSparkCheckpoint, DSparkConfig, OFFICIAL_CHECKPOINT_BYTES,
    OFFICIAL_CONFIG_BYTES, OFFICIAL_HEADER_BYTES, OFFICIAL_MODEL_ID, OFFICIAL_PARAMETER_COUNT,
    OFFICIAL_REVISION, OFFICIAL_TENSOR_COUNT, OFFICIAL_WEIGHTS_SHA256, digest_from_hex,
    strict_json, validate_official_config_bytes, validate_official_safetensors_prefix,
};
use crate::error::{DeltafinError, Result};
use crate::sys::fs::{self as sys_fs, Open};
use crate::trusted_download::{
    ByteRange, NativeHttpsTransport, Request, ResponseMeta, TimeoutPolicy, Transport,
    fsync_directory, publish_hard_link, read_bounded, rename_noreplace, secure_create_new,
    verify_regular_digest,
};

const BASE_URL: &str = "https://huggingface.co/Inferact/Kimi-K3-DSpark/resolve/cf6b8244620e7ea4b0651d214f28e89eac75bed6/";
const TREE_URL: &str = "https://huggingface.co/api/models/Inferact/Kimi-K3-DSpark/tree/cf6b8244620e7ea4b0651d214f28e89eac75bed6?recursive=true&expand=true";
const MANIFEST_NAME: &str = "deltafin-dspark-manifest.json";
const SPOTLIGHT_MARKER: &str = ".metadata_never_index";
const USER_AGENT: &str = "deltafin-dspark-setup/1";
const MAX_TREE_BYTES: u64 = 1 << 20;
const MAX_MANIFEST_BYTES: u64 = 64 << 10;

const DANGEROUS_SUFFIXES: &[&str] = &[
    ".py", ".pyc", ".pyo", ".pkl", ".pickle", ".pt", ".pth", ".bin", ".so", ".dylib", ".dll",
    ".zip", ".tar", ".tgz", ".gz", ".7z",
];

#[derive(Clone, Copy)]
struct FileSpec {
    name: &'static str,
    size: u64,
    sha256: &'static str,
}

const FILE_SPECS: [FileSpec; 5] = [
    FileSpec {
        name: ".gitattributes",
        size: 1_519,
        sha256: "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    },
    FileSpec {
        name: "LICENSE",
        size: 3_065,
        sha256: "20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887",
    },
    FileSpec {
        name: "README.md",
        size: 6_316,
        sha256: "4734eec1b55b5b40b4050512a029808aea117f24440dcaf27815efd1127a88dc",
    },
    FileSpec {
        name: "config.json",
        size: OFFICIAL_CONFIG_BYTES,
        sha256: "5a3c2f4f91c965ed93b14de5f12a4e9c17fd98d8c99916ed2deb26ce8702f970",
    },
    FileSpec {
        name: CHECKPOINT_BASENAME,
        size: OFFICIAL_CHECKPOINT_BYTES,
        sha256: OFFICIAL_WEIGHTS_SHA256,
    },
];

pub(crate) fn run(arguments: SetupDSparkArgs) -> Result<()> {
    let destination = arguments.destination.unwrap_or_else(default_destination);
    let mut transport = NativeHttpsTransport;
    if arguments.audit_only {
        audit_remote(&mut transport)?;
        println!(
            "DSpark remote audit passed: {OFFICIAL_MODEL_ID}@{OFFICIAL_REVISION}, {OFFICIAL_TENSOR_COUNT} BF16 tensors. No tensor payload downloaded."
        );
    } else if arguments.check {
        audit_local(&destination, true)?;
        println!(
            "DSpark local audit passed: {OFFICIAL_REVISION} at {}",
            destination.display()
        );
    } else {
        install(&destination, &mut transport)?;
    }
    Ok(())
}

/// Install DSpark below an explicit model root without consulting process
/// environment state.
pub(crate) fn install_at(root: &Path) -> Result<()> {
    let mut transport = NativeHttpsTransport;
    install(&root.join("k3-draft-dspark"), &mut transport)
}

/// Return the exact logical bytes in the pinned payload and generated
/// manifest. The zero-byte Spotlight marker needs no additional accounting.
pub(crate) fn exact_install_bytes() -> Result<u64> {
    let payload = FILE_SPECS.iter().map(|spec| spec.size).sum::<u64>();
    let mut manifest = serde_json::to_string_pretty(&expected_manifest()).map_err(|error| {
        DeltafinError::new(format!("serialize pinned DSpark manifest: {error}"))
    })?;
    manifest.push('\n');
    payload
        .checked_add(manifest.len() as u64)
        .ok_or_else(|| DeltafinError::new("DSpark install byte count overflowed"))
}

/// Credit an installation only after every pinned byte and semantic invariant
/// has been audited. An absent destination contributes no credit; interrupted
/// staging remains resumable but is conservatively ignored by capacity plans.
pub(crate) fn audited_installed_bytes(root: &Path) -> Result<u64> {
    let destination = root.join("k3-draft-dspark");
    match sys_fs::lstat(&destination) {
        Ok(_) => {
            audit_local(&destination, true)?;
            exact_install_bytes()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(io_error("inspect DSpark destination", &destination, error)),
    }
}

fn default_destination() -> PathBuf {
    std::env::var_os("DELTAFIN_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("k3-draft-dspark")
}

fn audit_remote(transport: &mut dyn Transport) -> Result<()> {
    let tree = fetch_bounded(
        transport,
        Request {
            url: TREE_URL.into(),
            range: None,
            user_agent: USER_AGENT,
            timeout: TimeoutPolicy::Metadata,
        },
        MAX_TREE_BYTES,
    )?;
    validate_tree(&strict_json(&tree, "Hugging Face repository tree")?)?;
    for spec in FILE_SPECS
        .iter()
        .filter(|spec| spec.name != CHECKPOINT_BASENAME)
    {
        let payload = fetch_pinned(transport, spec)?;
        if spec.name == "config.json" {
            validate_official_config_bytes(&payload)?;
        }
    }
    let length = fetch_range(transport, &FILE_SPECS[4], 0, 7)?;
    let header_length = u64::from_le_bytes(
        length
            .as_slice()
            .try_into()
            .map_err(|_| DeltafinError::new("remote Safetensors length prefix is not 8 bytes"))?,
    );
    if header_length != OFFICIAL_HEADER_BYTES {
        return Err(DeltafinError::new(format!(
            "remote Safetensors header is {header_length} bytes; expected {OFFICIAL_HEADER_BYTES}"
        )));
    }
    let header = fetch_range(transport, &FILE_SPECS[4], 8, 8 + header_length - 1)?;
    let mut prefix = length;
    prefix.extend_from_slice(&header);
    validate_official_safetensors_prefix(&prefix)
}

fn validate_tree(document: &Value) -> Result<()> {
    let entries = document
        .as_array()
        .ok_or_else(|| DeltafinError::new("Hugging Face tree response must be a list"))?;
    let expected: BTreeSet<_> = FILE_SPECS.iter().map(|spec| spec.name).collect();
    let mut actual = BTreeSet::new();
    for entry in entries {
        let entry = entry
            .as_object()
            .ok_or_else(|| DeltafinError::new("Hugging Face tree contains a non-object entry"))?;
        let path = entry
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| DeltafinError::new("repository entry path must be a string"))?;
        if path.contains('/') || matches!(path, "." | "..") {
            return Err(DeltafinError::new(format!(
                "unsafe repository path {path:?}"
            )));
        }
        reject_dangerous(path, "repository file")?;
        if entry.get("type").and_then(Value::as_str) != Some("file") {
            return Err(DeltafinError::new(format!(
                "repository entry {path:?} is not a file"
            )));
        }
        if !actual.insert(path) {
            return Err(DeltafinError::new(
                "repository tree contains duplicate paths",
            ));
        }
        let spec = file_spec(path).ok_or_else(|| {
            DeltafinError::new(format!(
                "repository allowlist contains unexpected file {path:?}"
            ))
        })?;
        if entry.get("size").and_then(Value::as_u64) != Some(spec.size) {
            return Err(DeltafinError::new(format!(
                "{path} remote size does not match the exact pin"
            )));
        }
        if path == CHECKPOINT_BASENAME {
            let lfs = entry
                .get("lfs")
                .and_then(Value::as_object)
                .ok_or_else(|| DeltafinError::new("model.safetensors lacks pinned LFS metadata"))?;
            if lfs.get("size").and_then(Value::as_u64) != Some(spec.size)
                || lfs.get("oid").and_then(Value::as_str) != Some(spec.sha256)
            {
                return Err(DeltafinError::new(
                    "model.safetensors LFS metadata does not match the pin",
                ));
            }
        }
    }
    if actual != expected {
        return Err(DeltafinError::new(format!(
            "repository allowlist mismatch; expected={expected:?}, actual={actual:?}"
        )));
    }
    Ok(())
}

fn fetch_pinned(transport: &mut dyn Transport, spec: &FileSpec) -> Result<Vec<u8>> {
    let payload = fetch_bounded(
        transport,
        Request {
            url: file_url(spec.name),
            range: None,
            user_agent: USER_AGENT,
            timeout: TimeoutPolicy::Metadata,
        },
        spec.size,
    )?;
    require_size_hash_bytes(spec, &payload)?;
    Ok(payload)
}

fn fetch_range(
    transport: &mut dyn Transport,
    spec: &FileSpec,
    start: u64,
    end: u64,
) -> Result<Vec<u8>> {
    if end < start {
        return Err(DeltafinError::new("invalid remote byte range"));
    }
    let expected = end - start + 1;
    let mut payload = Vec::with_capacity(expected as usize);
    let meta = transport.transfer(
        &Request {
            url: file_url(spec.name),
            range: Some(ByteRange {
                start,
                end: Some(end),
            }),
            user_agent: USER_AGENT,
            timeout: TimeoutPolicy::Metadata,
        },
        &mut payload,
        expected,
    )?;
    if meta.status != 206 {
        return Err(DeltafinError::new(format!(
            "{} server ignored the byte-range request",
            spec.name
        )));
    }
    require_content_range(&meta, start, end, spec.size)?;
    if meta.bytes != expected {
        return Err(DeltafinError::new(format!(
            "{} range returned {} bytes; expected {expected}",
            spec.name, meta.bytes
        )));
    }
    Ok(payload)
}

fn fetch_bounded(transport: &mut dyn Transport, request: Request, maximum: u64) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    let meta = transport.transfer(&request, &mut payload, maximum)?;
    if meta.status != 200 {
        return Err(DeltafinError::new(format!(
            "download returned HTTP status {}",
            meta.status
        )));
    }
    Ok(payload)
}

fn install(destination: &Path, transport: &mut dyn Transport) -> Result<()> {
    let destination = absolute_path(destination)?;
    let parent = destination
        .parent()
        .ok_or_else(|| DeltafinError::new("DSpark destination must have a parent directory"))?;
    require_real_directory(parent)?;
    let _lock = InstallationLock::acquire(&destination)?;
    match sys_fs::lstat(&destination) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            ensure_spotlight_marker(&destination)?;
            audit_local(&destination, true)?;
            println!(
                "DSpark is already installed and verified at {}",
                destination.display()
            );
            return Ok(());
        }
        Ok(_) => {
            return Err(DeltafinError::new(format!(
                "refusing to replace existing path {}",
                destination.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect DSpark destination", &destination, error)),
    }
    audit_remote(transport)?;
    let staging = suffix_path(&destination, ".installing")?;
    match sys_fs::lstat(&staging) {
        Ok(_) => require_real_directory(&staging)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(&staging)
                .map_err(|error| io_error("create DSpark staging directory", &staging, error))?;
            sys_fs::restrict_to_owner(&staging)
                .map_err(|error| io_error("secure DSpark staging directory", &staging, error))?;
            fsync_directory(parent)?;
        }
        Err(error) => {
            return Err(io_error(
                "inspect DSpark staging directory",
                &staging,
                error,
            ));
        }
    }
    ensure_spotlight_marker(&staging)?;
    validate_directory_names(&staging, true, false)?;
    println!(
        "Installing {OFFICIAL_MODEL_ID}@{OFFICIAL_REVISION} through {}",
        staging.display()
    );
    for spec in &FILE_SPECS[..4] {
        download_pinned_file(spec, &staging, transport)?;
    }
    download_pinned_file(&FILE_SPECS[4], &staging, transport)?;
    let manifest = staging.join(MANIFEST_NAME);
    if sys_fs::lstat(&manifest).is_ok() {
        audit_local(&staging, true)?;
    } else {
        audit_local(&staging, false)?;
        write_manifest(&staging)?;
    }
    audit_local(&staging, true)?;
    if sys_fs::lstat(&destination).is_ok() {
        return Err(DeltafinError::new(format!(
            "refusing to publish over path that appeared: {}",
            destination.display()
        )));
    }
    rename_noreplace(&staging, &destination)?;
    fsync_directory(parent)?;
    audit_local(&destination, true)?;
    println!("DSpark installed and verified at {}", destination.display());
    Ok(())
}

fn download_pinned_file(
    spec: &FileSpec,
    directory: &Path,
    transport: &mut dyn Transport,
) -> Result<()> {
    let final_path = directory.join(spec.name);
    match sys_fs::lstat(&final_path) {
        Ok(_) => {
            validate_file(&final_path, spec)?;
            if spec.name == CHECKPOINT_BASENAME {
                DSparkCheckpoint::open_official(directory)?;
            }
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect DSpark target file", &final_path, error)),
    }
    let part = suffix_path(&final_path, ".part")?;
    let start = existing_part_size(&part, spec.size)?;
    if start < spec.size {
        let (mut target, identity) = open_part_append(&part, start)?;
        let transfer = transport.transfer(
            &Request {
                url: file_url(spec.name),
                range: (start > 0).then_some(ByteRange { start, end: None }),
                user_agent: USER_AGENT,
                timeout: if spec.name == CHECKPOINT_BASENAME {
                    TimeoutPolicy::LargePayload
                } else {
                    TimeoutPolicy::Metadata
                },
            },
            &mut target,
            spec.size - start,
        );
        let meta = match transfer {
            Ok(meta) => meta,
            Err(error) => {
                rollback_partial(&target, &part, start)?;
                return Err(error);
            }
        };
        if let Err(error) = validate_resume_response(&meta, start, spec.size) {
            rollback_partial(&target, &part, start)?;
            return Err(error);
        }
        if meta.bytes != spec.size - start {
            rollback_partial(&target, &part, start)?;
            return Err(DeltafinError::new(format!(
                "{} download stopped at {} bytes; expected {}. Re-run to resume.",
                spec.name,
                start + meta.bytes,
                spec.size
            )));
        }
        target
            .flush()
            .map_err(|error| io_error("flush DSpark partial file", &part, error))?;
        target
            .sync_all()
            .map_err(|error| io_error("fsync DSpark partial file", &part, error))?;
        validate_open_identity(&target, identity, spec.size, &part)?;
    }
    validate_file(&part, spec)?;
    if spec.name == CHECKPOINT_BASENAME {
        validate_weight_path(&part)?;
    }
    publish_hard_link(&part, &final_path, directory)?;
    println!("  {}: verified", spec.name);
    Ok(())
}

fn rollback_partial(file: &File, path: &Path, length: u64) -> Result<()> {
    file.set_len(length)
        .map_err(|error| io_error("roll back rejected DSpark response", path, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync rolled-back DSpark partial", path, error))
}

fn audit_local(directory: &Path, require_manifest: bool) -> Result<()> {
    validate_directory_names(directory, false, require_manifest)?;
    for spec in &FILE_SPECS {
        validate_file(&directory.join(spec.name), spec)?;
    }
    DSparkConfig::load_official(directory)?;
    DSparkCheckpoint::open_official(directory)?;
    if require_manifest {
        let raw = read_regular_limited(&directory.join(MANIFEST_NAME), MAX_MANIFEST_BYTES)?;
        let actual = strict_json(&raw, MANIFEST_NAME)?;
        if actual != expected_manifest() {
            return Err(DeltafinError::new(
                "local DSpark manifest does not match the pin",
            ));
        }
    }
    Ok(())
}

fn validate_weight_path(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| DeltafinError::new("checkpoint path has no parent"))?;
    let temporary_name = path
        .file_name()
        .ok_or_else(|| DeltafinError::new("checkpoint path has no filename"))?;
    if temporary_name == CHECKPOINT_BASENAME {
        DSparkCheckpoint::open_official(parent)?;
        return Ok(());
    }
    let raw = read_regular_prefix(path, 8 + OFFICIAL_HEADER_BYTES)?;
    validate_official_safetensors_prefix(&raw)
}

fn validate_directory_names(directory: &Path, staging: bool, require_manifest: bool) -> Result<()> {
    require_real_directory(directory)?;
    let mut expected: BTreeSet<OsString> = FILE_SPECS
        .iter()
        .map(|spec| OsString::from(spec.name))
        .collect();
    if require_manifest {
        expected.insert(MANIFEST_NAME.into());
    }
    let mut allowed = expected.clone();
    allowed.insert(SPOTLIGHT_MARKER.into());
    if staging {
        for spec in &FILE_SPECS {
            allowed.insert(format!("{}.part", spec.name).into());
        }
        allowed.insert(format!("{MANIFEST_NAME}.part").into());
        allowed.insert(MANIFEST_NAME.into());
    }
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| io_error("read DSpark directory", directory, error))?
    {
        let entry =
            entry.map_err(|error| io_error("read DSpark directory entry", directory, error))?;
        let file_type = entry
            .file_type()
            .map_err(|error| io_error("inspect DSpark directory entry", &entry.path(), error))?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(DeltafinError::new(format!(
                "unexpected non-regular path {}",
                entry.path().display()
            )));
        }
        let name = entry.file_name();
        let text = name
            .to_str()
            .ok_or_else(|| DeltafinError::new("DSpark directory filename is not valid UTF-8"))?;
        reject_dangerous(text, "local file")?;
        actual.insert(name);
    }
    let extras: Vec<_> = actual.difference(&allowed).collect();
    if !extras.is_empty() {
        return Err(DeltafinError::new(format!(
            "unexpected files in {}: {extras:?}",
            directory.display()
        )));
    }
    if !staging {
        let missing: Vec<_> = expected.difference(&actual).collect();
        if !missing.is_empty() {
            return Err(DeltafinError::new(format!(
                "missing files in {}: {missing:?}",
                directory.display()
            )));
        }
    }
    Ok(())
}

fn validate_file(path: &Path, spec: &FileSpec) -> Result<()> {
    verify_regular_digest(path, spec.size, digest_from_hex(spec.sha256)?)
}

fn require_size_hash_bytes(spec: &FileSpec, payload: &[u8]) -> Result<()> {
    if payload.len() as u64 != spec.size {
        return Err(DeltafinError::new(format!(
            "{} returned {} bytes; expected {}",
            spec.name,
            payload.len(),
            spec.size
        )));
    }
    if crate::packfile::digest_bytes(payload) != digest_from_hex(spec.sha256)? {
        return Err(DeltafinError::new(format!(
            "{} SHA-256 does not match the exact pin",
            spec.name
        )));
    }
    Ok(())
}

fn expected_manifest() -> Value {
    let files: serde_json::Map<String, Value> = FILE_SPECS
        .iter()
        .map(|spec| {
            (
                spec.name.to_owned(),
                json!({"sha256": spec.sha256, "size": spec.size}),
            )
        })
        .collect();
    json!({
        "format": 1,
        "repository": OFFICIAL_MODEL_ID,
        "revision": OFFICIAL_REVISION,
        "trust_remote_code": false,
        "files": files,
        "safetensors": {
            "dtype": "BF16",
            "tensor_count": OFFICIAL_TENSOR_COUNT,
            "parameter_count": OFFICIAL_PARAMETER_COUNT,
            "header_size": OFFICIAL_HEADER_BYTES,
            "vllm_skips": ["confidence_head", "embed_tokens", "lm_head"]
        }
    })
}

fn write_manifest(directory: &Path) -> Result<()> {
    let final_path = directory.join(MANIFEST_NAME);
    let part = suffix_path(&final_path, ".part")?;
    match sys_fs::lstat(&part) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            fs::remove_file(&part)
                .map_err(|error| io_error("remove stale manifest partial", &part, error))?;
        }
        Ok(_) => return Err(DeltafinError::new("unsafe manifest partial path")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect manifest partial", &part, error)),
    }
    let mut payload = serde_json::to_string_pretty(&expected_manifest()).map_err(|error| {
        DeltafinError::new(format!("serialize pinned DSpark manifest: {error}"))
    })?;
    payload.push('\n');
    let mut file = secure_create_new(&part, 0o600)?;
    file.write_all(payload.as_bytes())
        .map_err(|error| io_error("write DSpark manifest", &part, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync DSpark manifest", &part, error))?;
    fs::rename(&part, &final_path)
        .map_err(|error| io_error("publish DSpark manifest", &final_path, error))?;
    fsync_directory(directory)
}

fn ensure_spotlight_marker(directory: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    match sys_fs::lstat(&marker) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(DeltafinError::new(format!(
            "unsafe Spotlight exclusion marker {}",
            marker.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            secure_create_new(&marker, 0o600)?
                .sync_all()
                .map_err(|error| io_error("fsync Spotlight marker", &marker, error))?;
            fsync_directory(directory)
        }
        Err(error) => Err(io_error("inspect Spotlight marker", &marker, error)),
    }
}

fn validate_resume_response(meta: &ResponseMeta, start: u64, expected_size: u64) -> Result<()> {
    if start == 0 {
        if meta.status != 200 && meta.status != 206 {
            return Err(DeltafinError::new(format!(
                "download returned HTTP status {}",
                meta.status
            )));
        }
        if meta.status == 206 {
            require_content_range(meta, 0, expected_size - 1, expected_size)?;
        }
    } else {
        if meta.status != 206 {
            return Err(DeltafinError::new(
                "server ignored the resume Range request; the verified .part file was preserved",
            ));
        }
        require_content_range(meta, start, expected_size - 1, expected_size)?;
    }
    Ok(())
}

fn require_content_range(meta: &ResponseMeta, start: u64, end: u64, total: u64) -> Result<()> {
    let value = meta
        .headers
        .get("content-range")
        .ok_or_else(|| DeltafinError::new("byte-range response lacks Content-Range"))?;
    let expected = format!("bytes {start}-{end}/{total}");
    if value != &expected {
        return Err(DeltafinError::new(format!(
            "unexpected Content-Range {value:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn file_url(name: &str) -> String {
    // The immutable allowlist contains only RFC 3986 unreserved filename
    // characters, so no general-purpose URL encoder is needed or permitted.
    debug_assert!(
        name.bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    );
    format!("{BASE_URL}{name}?download=true")
}

fn file_spec(name: &str) -> Option<&'static FileSpec> {
    FILE_SPECS.iter().find(|spec| spec.name == name)
}

fn reject_dangerous(name: &str, label: &str) -> Result<()> {
    let lower = name.to_ascii_lowercase();
    if DANGEROUS_SUFFIXES
        .iter()
        .any(|suffix| lower.ends_with(suffix))
    {
        return Err(DeltafinError::new(format!(
            "executable or unsafe {label} {name:?}"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct OpenIdentity {
    device: u64,
    inode: u64,
}

fn open_regular(path: &Path, expected_size: Option<u64>) -> Result<(File, OpenIdentity)> {
    let before = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect regular file", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(DeltafinError::new(format!(
            "{} is not a regular non-symlink file",
            path.display()
        )));
    }
    if expected_size.is_some_and(|size| before.len() != size) {
        return Err(DeltafinError::new(format!(
            "{} has {} bytes; expected {}",
            path.display(),
            before.len(),
            expected_size.unwrap_or_default()
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open regular file without following symlinks", path, error))?;
    let identity = OpenIdentity {
        device: before.dev(),
        inode: before.ino(),
    };
    validate_open_identity(&file, identity, before.len(), path)?;
    Ok((file, identity))
}

fn validate_open_identity(
    file: &File,
    identity: OpenIdentity,
    size: u64,
    path: &Path,
) -> Result<()> {
    let metadata = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat opened regular file", path, error))?;
    if !metadata.is_file()
        || metadata.dev() != identity.device
        || metadata.ino() != identity.inode
        || metadata.len() != size
    {
        return Err(DeltafinError::new(format!(
            "regular file changed while open: {}",
            path.display()
        )));
    }
    Ok(())
}

fn read_regular_limited(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let (mut file, identity) = open_regular(path, None)?;
    let raw = read_bounded(&mut file, maximum)?;
    validate_open_identity(&file, identity, raw.len() as u64, path)?;
    Ok(raw)
}

fn read_regular_prefix(path: &Path, length: u64) -> Result<Vec<u8>> {
    let (mut file, identity) = open_regular(path, None)?;
    if sys_fs::fstat(&file)
        .map_err(|error| io_error("stat file", path, error))?
        .len()
        < length
    {
        return Err(DeltafinError::new("file is shorter than its pinned header"));
    }
    let mut raw = vec![0; length as usize];
    file.read_exact(&mut raw)
        .map_err(|error| io_error("read pinned file prefix", path, error))?;
    let size = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat file", path, error))?
        .len();
    validate_open_identity(&file, identity, size, path)?;
    Ok(raw)
}

fn existing_part_size(path: &Path, maximum: u64) -> Result<u64> {
    match sys_fs::lstat(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(DeltafinError::new(format!(
                "refusing unsafe partial download {}",
                path.display()
            )))
        }
        Ok(metadata) if metadata.len() > maximum => Err(DeltafinError::new(format!(
            "partial download {} exceeds its pinned size",
            path.display()
        ))),
        Ok(metadata) => Ok(metadata.len()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(io_error("inspect partial download", path, error)),
    }
}

fn open_part_append(path: &Path, expected_size: u64) -> Result<(File, OpenIdentity)> {
    let file = Open::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open partial download safely", path, error))?;
    let metadata = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat partial download", path, error))?;
    if !metadata.is_file() || metadata.len() != expected_size {
        return Err(DeltafinError::new(format!(
            "partial download changed while opening: {}",
            path.display()
        )));
    }
    Ok((
        file,
        OpenIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        },
    ))
}

fn require_real_directory(path: &Path) -> Result<()> {
    let metadata =
        sys_fs::lstat(path).map_err(|error| io_error("inspect directory", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "{} is not a real directory",
            path.display()
        )));
    }
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .map_err(|error| DeltafinError::new(format!("resolve current directory: {error}")))
    }
}

fn suffix_path(path: &Path, suffix: &str) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| DeltafinError::new("path cannot carry a safe suffix"))?;
    let mut suffixed = name.to_os_string();
    suffixed.push(suffix);
    Ok(path.with_file_name(suffixed))
}

struct InstallationLock {
    file: File,
}

impl InstallationLock {
    fn acquire(destination: &Path) -> Result<Self> {
        let path = suffix_path(destination, ".install.lock")?;
        let file = Open::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .no_follow()
            .open(&path)
            .map_err(|error| io_error("open DSpark installation lock", &path, error))?;
        if !sys_fs::fstat(&file)
            .map_err(|error| io_error("stat DSpark installation lock", &path, error))?
            .is_file()
        {
            return Err(DeltafinError::new(
                "DSpark installation lock is not regular",
            ));
        }
        if !sys_fs::try_lock_exclusive(&file)
            .map_err(|error| io_error("acquire DSpark installation lock", &path, error))?
        {
            return Err(DeltafinError::new(format!(
                "another DSpark installer is active for {}",
                destination.display()
            )));
        }
        Ok(Self { file })
    }
}

impl Drop for InstallationLock {
    fn drop(&mut self) {
        sys_fs::unlock(&self.file);
    }
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::thread;

    type FakeResponse = (u16, Vec<(&'static str, String)>, Vec<u8>);

    struct FakeTransport {
        responses: VecDeque<FakeResponse>,
        requests: Vec<Request>,
    }

    impl Transport for FakeTransport {
        fn transfer(
            &mut self,
            request: &Request,
            target: &mut dyn Write,
            maximum: u64,
        ) -> Result<ResponseMeta> {
            self.requests.push(request.clone());
            let (status, headers, body) = self
                .responses
                .pop_front()
                .ok_or_else(|| DeltafinError::new("unexpected fake request"))?;
            if body.len() as u64 > maximum {
                return Err(DeltafinError::new("fake response exceeded safety bound"));
            }
            target
                .write_all(&body)
                .map_err(|error| DeltafinError::new(error.to_string()))?;
            Ok(ResponseMeta {
                status,
                headers: headers
                    .into_iter()
                    .map(|(name, value)| (name.into(), value))
                    .collect(),
                bytes: body.len() as u64,
            })
        }
    }

    fn valid_tree() -> Value {
        Value::Array(
            FILE_SPECS
                .iter()
                .map(|spec| {
                    let mut entry = json!({
                        "path": spec.name,
                        "type": "file",
                        "size": spec.size,
                    });
                    if spec.name == CHECKPOINT_BASENAME {
                        entry["lfs"] = json!({"size": spec.size, "oid": spec.sha256});
                    }
                    entry
                })
                .collect(),
        )
    }

    #[test]
    fn repository_tree_is_an_exact_data_only_allowlist() {
        validate_tree(&valid_tree()).unwrap();
        let mut bad = valid_tree().as_array().unwrap().clone();
        bad.push(json!({"path": "modeling.py", "type": "file", "size": 1}));
        assert!(validate_tree(&Value::Array(bad)).is_err());
    }

    #[test]
    fn fake_transport_stops_before_any_checkpoint_request_on_small_file_corruption() {
        let tree = serde_json::to_vec(&valid_tree()).unwrap();
        let mut transport = FakeTransport {
            responses: VecDeque::from([
                (200, vec![], tree),
                (200, vec![], vec![0; FILE_SPECS[0].size as usize]),
            ]),
            requests: Vec::new(),
        };
        assert!(audit_remote(&mut transport).is_err());
        assert_eq!(transport.requests.len(), 2);
        assert!(
            transport
                .requests
                .iter()
                .all(|request| request.range.is_none())
        );
        assert!(
            transport
                .requests
                .iter()
                .all(|request| !request.url.contains(CHECKPOINT_BASENAME))
        );
    }

    #[test]
    fn range_response_requires_exact_status_and_content_range() {
        let mut transport = FakeTransport {
            responses: VecDeque::from([(200, vec![], vec![0; 8])]),
            requests: Vec::new(),
        };
        assert!(fetch_range(&mut transport, &FILE_SPECS[4], 0, 7).is_err());
        assert_eq!(
            transport.requests[0].range,
            Some(ByteRange {
                start: 0,
                end: Some(7)
            })
        );
    }

    #[test]
    fn ignored_resume_response_restores_and_fsyncs_the_existing_prefix() {
        let root = std::env::temp_dir().join(format!(
            "deltafin-dspark-resume-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let part = root.join("fixture.part");
        fs::write(&part, b"keep").unwrap();
        let spec = FileSpec {
            name: "fixture",
            size: 8,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        };
        let mut transport = FakeTransport {
            responses: VecDeque::from([(200, vec![], b"evil".to_vec())]),
            requests: Vec::new(),
        };
        assert!(download_pinned_file(&spec, &root, &mut transport).is_err());
        assert_eq!(fs::read(&part).unwrap(), b"keep");
        assert!(!root.join("fixture").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn local_fixture_rejects_symlinks_and_dangerous_suffixes() {
        let root = std::env::temp_dir().join(format!(
            "deltafin-dspark-setup-test-{}-{}",
            std::process::id(),
            thread::current().name().unwrap_or("fixture")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        crate::sys::fs::symlink("missing", root.join("config.json")).unwrap();
        assert!(validate_directory_names(&root, true, false).is_err());
        fs::remove_file(root.join("config.json")).unwrap();
        File::create(root.join("payload.pkl")).unwrap();
        assert!(validate_directory_names(&root, true, false).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
