//! Native installer for the optional pinned Qwen raw-completion assistants.
//!
//! Only the eight immutable data/configuration files below are admitted as
//! runtime inputs. No repository code is requested, imported, or executed.
//! A former Hugging Face installer may have left a README, attributes file,
//! and bounded download metadata beside those inputs; native adoption audits
//! and ignores that inert material instead of deleting user data.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::cli::SetupQwenArgs;
use crate::dspark_checkpoint::strict_json;
use crate::error::{DeltafinError, Result};
use crate::packfile::Digest;
use crate::sys::fs::{self as sys_fs, Open};
use crate::trusted_download::{
    ByteRange, NativeHttpsTransport, Request, ResponseMeta, TimeoutPolicy, Transport,
    fsync_directory, publish_hard_link, rename_noreplace, secure_create_new, verify_regular_digest,
};

const USER_AGENT: &str = "deltafin-draft-setup";
const MANIFEST_NAME: &str = "deltafin-manifest.json";
const SPOTLIGHT_MARKER: &str = ".metadata_never_index";
const WEIGHT_NAME: &str = "model.safetensors";
const LEGACY_METADATA_MAX_FILE: u64 = 4 << 20;
const LEGACY_CACHE_MAX_FILE: u64 = 1 << 20;
const LEGACY_CACHE_MAX_BYTES: u64 = 16 << 20;
const LEGACY_CACHE_MAX_ENTRIES: usize = 512;
const LEGACY_CACHE_MAX_DEPTH: usize = 8;

#[derive(Debug, Clone, Copy)]
struct FilePin {
    name: &'static str,
    size: u64,
    sha256: Digest,
}

#[derive(Debug, Clone, Copy)]
struct ModelPin<'a> {
    destination: &'static str,
    repository: &'static str,
    revision: &'static str,
    base_url: &'static str,
    files: &'a [FilePin],
}

const WIDE_FILES: [FilePin; 8] = [
    pin(
        "LICENSE",
        11_343,
        "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    ),
    pin(
        "config.json",
        727,
        "1bb33a92c3548fbc68b889b490e810440435253598835bd71dff0396060c12db",
    ),
    pin(
        "generation_config.json",
        138,
        "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    ),
    pin(
        "merges.txt",
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    pin(
        WEIGHT_NAME,
        3_441_185_608,
        "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841",
    ),
    pin(
        "tokenizer.json",
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    pin(
        "tokenizer_config.json",
        9_678,
        "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    ),
    pin(
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
];

const PROBE_FILES: [FilePin; 8] = [
    pin(
        "LICENSE",
        11_343,
        "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    ),
    pin(
        "config.json",
        727,
        "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    ),
    pin(
        "generation_config.json",
        138,
        "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    ),
    pin(
        "merges.txt",
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    pin(
        WEIGHT_NAME,
        1_192_135_096,
        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    ),
    pin(
        "tokenizer.json",
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    pin(
        "tokenizer_config.json",
        9_678,
        "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    ),
    pin(
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
];

const WIDE: ModelPin<'static> = ModelPin {
    destination: "k3-draft-qwen3-1.7b-base",
    repository: "Qwen/Qwen3-1.7B-Base",
    revision: "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    base_url: "https://huggingface.co/Qwen/Qwen3-1.7B-Base/resolve/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/",
    files: &WIDE_FILES,
};

const PROBE: ModelPin<'static> = ModelPin {
    destination: "k3-draft-qwen3-0.6b-base",
    repository: "Qwen/Qwen3-0.6B-Base",
    revision: "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    base_url: "https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/",
    files: &PROBE_FILES,
};

pub(crate) fn run(arguments: SetupQwenArgs) -> Result<()> {
    let root = arguments
        .model_root
        .or_else(|| std::env::var_os("DELTAFIN_ROOT").map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if arguments.check {
        audit_model(&root.join(WIDE.destination), &WIDE)?;
        println!(
            "wide assistant check passed: {}@{}",
            WIDE.repository, WIDE.revision
        );
        audit_model(&root.join(PROBE.destination), &PROBE)?;
        println!(
            "probe assistant check passed: {}@{}",
            PROBE.repository, PROBE.revision
        );
        return Ok(());
    }
    let mut transport = NativeHttpsTransport;
    install_model(&root, &WIDE, &mut transport)?;
    install_model(&root, &PROBE, &mut transport)?;
    println!(
        "Optional Qwen raw-completion drafting installed. DSpark remains the default learned chat/server drafter."
    );
    Ok(())
}

/// Install both optional Qwen assistants below an explicit root. They remain
/// opt-in and are never part of the default full-K3 setup.
pub(crate) fn install_at(root: &Path) -> Result<()> {
    let mut transport = NativeHttpsTransport;
    install_model(root, &WIDE, &mut transport)?;
    install_model(root, &PROBE, &mut transport)
}

/// Exact logical bytes occupied by both pinned payloads and their manifests.
pub(crate) fn exact_install_bytes() -> Result<u64> {
    [WIDE, PROBE].into_iter().try_fold(0_u64, |total, model| {
        let files = model.files.iter().map(|pin| pin.size).sum::<u64>();
        let manifest = manifest_bytes(&model)?.len() as u64;
        total
            .checked_add(files)
            .and_then(|value| value.checked_add(manifest))
            .ok_or_else(|| DeltafinError::new("Qwen install byte count overflowed"))
    })
}

/// Return exact credit for fully authenticated published assistants. A model
/// that has not yet been published receives no credit; unsafe or invalid
/// published paths fail closed.
pub(crate) fn audited_installed_bytes(root: &Path) -> Result<u64> {
    [WIDE, PROBE].into_iter().try_fold(0_u64, |total, model| {
        let destination = root.join(model.destination);
        match sys_fs::lstat(&destination) {
            Ok(_) => {
                audit_model(&destination, &model)?;
                let actual_manifest =
                    read_regular_limited(&destination.join(MANIFEST_NAME), 64 << 10)?;
                let expected_manifest = manifest_bytes(&model)?;
                if actual_manifest != expected_manifest {
                    return Err(DeltafinError::new(format!(
                        "{} manifest does not match the exact pin",
                        model.repository
                    )));
                }
                let files = model.files.iter().map(|pin| pin.size).sum::<u64>();
                let manifest = expected_manifest.len() as u64;
                total
                    .checked_add(files)
                    .and_then(|value| value.checked_add(manifest))
                    .ok_or_else(|| DeltafinError::new("Qwen installed byte count overflowed"))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(total),
            Err(error) => Err(io_error("inspect Qwen destination", &destination, error)),
        }
    })
}

fn install_model(root: &Path, model: &ModelPin<'_>, transport: &mut dyn Transport) -> Result<()> {
    require_real_directory(root)?;
    let destination = root.join(model.destination);
    let _lock = InstallationLock::acquire(&destination)?;
    match sys_fs::lstat(&destination) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            ensure_spotlight_marker(&destination)?;
            // Installations created by the former Python setup path contain
            // the exact pinned payload but predate Deltafin's native receipt.
            // Authenticate every payload byte and its semantic config before
            // publishing only the small deterministic manifest.  This is an
            // adoption step, not a relaxed audit or a model redownload.
            audit_model(&destination, model)?;
            write_or_validate_manifest(&destination, model)?;
            audit_model(&destination, model)?;
            println!("{} is already installed and verified", model.repository);
            return Ok(());
        }
        Ok(_) => {
            return Err(DeltafinError::new(format!(
                "refusing to replace existing path {}",
                destination.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect Qwen destination", &destination, error)),
    }
    let staging = suffix_path(&destination, ".installing")?;
    match sys_fs::lstat(&staging) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
        Ok(_) => return Err(DeltafinError::new("Qwen staging path is unsafe")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(&staging)
                .map_err(|error| io_error("create Qwen staging directory", &staging, error))?;
            sys_fs::restrict_to_owner(&staging)
                .map_err(|error| io_error("secure Qwen staging directory", &staging, error))?;
            fsync_directory(root)?;
        }
        Err(error) => return Err(io_error("inspect Qwen staging directory", &staging, error)),
    }
    ensure_spotlight_marker(&staging)?;
    validate_staging_names(&staging, model)?;
    println!(
        "Downloading {} at pinned revision {} through {}",
        model.repository,
        model.revision,
        staging.display()
    );
    for file in model.files {
        download_file(model, file, &staging, transport)?;
    }
    validate_payload(&staging, model)?;
    write_or_validate_manifest(&staging, model)?;
    validate_payload(&staging, model)?;
    rename_noreplace(&staging, &destination)?;
    fsync_directory(root)?;
    audit_model(&destination, model)
}

fn audit_model(directory: &Path, model: &ModelPin<'_>) -> Result<()> {
    require_real_directory(directory)?;
    validate_published_names(directory, model)?;
    validate_spotlight_marker(directory)?;
    validate_payload(directory, model)?;
    // Semantic parsing reopens config.json; reauthenticate after it returns.
    for file in model.files {
        verify_regular_digest(&directory.join(file.name), file.size, file.sha256)?;
    }
    Ok(())
}

/// Keep the production input roster exact while permitting only the inert
/// artifacts left by the former Hugging Face downloader. Nothing in this
/// function makes those artifacts available to the model loader.
fn validate_published_names(directory: &Path, model: &ModelPin<'_>) -> Result<()> {
    let admitted: BTreeSet<&str> = model.files.iter().map(|pin| pin.name).collect();
    for entry in fs::read_dir(directory)
        .map_err(|error| io_error("read Qwen directory", directory, error))?
    {
        let entry = entry.map_err(|error| io_error("read Qwen entry", directory, error))?;
        let path = entry.path();
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| DeltafinError::new("Qwen filename is not UTF-8"))?;
        let kind = entry
            .file_type()
            .map_err(|error| io_error("inspect Qwen entry", &path, error))?;
        if admitted.contains(name.as_str()) || name == MANIFEST_NAME || name == SPOTLIGHT_MARKER {
            if kind.is_symlink() || !kind.is_file() {
                return Err(DeltafinError::new(format!(
                    "Qwen admitted input is not a regular non-symlink file: {}",
                    path.display()
                )));
            }
            continue;
        }
        match name.as_str() {
            "README.md" | ".gitattributes" => {
                // Open and snapshot the ignored file rather than trusting only
                // directory-entry metadata. Its bytes are deliberately unused.
                let _ = read_regular_limited(&path, LEGACY_METADATA_MAX_FILE)?;
            }
            ".cache" if kind.is_dir() && !kind.is_symlink() => {
                validate_ignored_legacy_cache(&path)?;
            }
            _ => {
                return Err(DeltafinError::new(format!(
                    "unexpected Qwen path {}; only pinned runtime inputs and bounded inert legacy metadata are permitted",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn validate_ignored_legacy_cache(root: &Path) -> Result<()> {
    let mut pending = vec![(root.to_path_buf(), 0_usize)];
    let mut entries = 0_usize;
    let mut bytes = 0_u64;
    while let Some((directory, depth)) = pending.pop() {
        if depth >= LEGACY_CACHE_MAX_DEPTH {
            return Err(DeltafinError::new(format!(
                "ignored Qwen cache exceeds the bounded nesting depth at {}",
                directory.display()
            )));
        }
        for entry in fs::read_dir(&directory)
            .map_err(|error| io_error("read ignored Qwen cache", &directory, error))?
        {
            let entry = entry
                .map_err(|error| io_error("read ignored Qwen cache entry", &directory, error))?;
            entries = entries
                .checked_add(1)
                .ok_or_else(|| DeltafinError::new("ignored Qwen cache entry count overflowed"))?;
            if entries > LEGACY_CACHE_MAX_ENTRIES {
                return Err(DeltafinError::new(
                    "ignored Qwen cache contains too many entries",
                ));
            }
            let path = entry.path();
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| DeltafinError::new("ignored Qwen cache filename is not UTF-8"))?;
            let kind = entry
                .file_type()
                .map_err(|error| io_error("inspect ignored Qwen cache entry", &path, error))?;
            if kind.is_symlink() || dangerous_name(&name) {
                return Err(DeltafinError::new(format!(
                    "ignored Qwen cache contains an unsafe path {}",
                    path.display()
                )));
            }
            if kind.is_dir() {
                pending.push((path, depth + 1));
                continue;
            }
            if !kind.is_file() {
                return Err(DeltafinError::new(format!(
                    "ignored Qwen cache contains a special file {}",
                    path.display()
                )));
            }
            let metadata = entry
                .metadata()
                .map_err(|error| io_error("stat ignored Qwen cache file", &path, error))?;
            if metadata.len() > LEGACY_CACHE_MAX_FILE {
                return Err(DeltafinError::new(format!(
                    "ignored Qwen cache file {} exceeds its bound",
                    path.display()
                )));
            }
            bytes = bytes
                .checked_add(metadata.len())
                .ok_or_else(|| DeltafinError::new("ignored Qwen cache size overflowed"))?;
            if bytes > LEGACY_CACHE_MAX_BYTES {
                return Err(DeltafinError::new(
                    "ignored Qwen cache exceeds its aggregate byte bound",
                ));
            }
        }
    }
    Ok(())
}

fn validate_payload(directory: &Path, model: &ModelPin<'_>) -> Result<()> {
    for file in model.files {
        verify_regular_digest(&directory.join(file.name), file.size, file.sha256)?;
    }
    let raw = read_regular_limited(&directory.join("config.json"), 64 << 10)?;
    let config = strict_json(&raw, "Qwen config.json")?;
    let config = config
        .as_object()
        .ok_or_else(|| DeltafinError::new("Qwen config.json must be an object"))?;
    if config.get("model_type").and_then(Value::as_str) != Some("qwen3") {
        return Err(DeltafinError::new(
            "Qwen config is not the built-in qwen3 model type",
        ));
    }
    if config.get("auto_map").is_some_and(|value| !value.is_null())
        || config.get("trust_remote_code").and_then(Value::as_bool) == Some(true)
    {
        return Err(DeltafinError::new(
            "Qwen config requests or describes remote model code",
        ));
    }
    Ok(())
}

fn download_file(
    model: &ModelPin<'_>,
    pin: &FilePin,
    directory: &Path,
    transport: &mut dyn Transport,
) -> Result<()> {
    let destination = directory.join(pin.name);
    match sys_fs::lstat(&destination) {
        Ok(_) => return verify_regular_digest(&destination, pin.size, pin.sha256),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect Qwen file", &destination, error)),
    }
    let part = suffix_path(&destination, ".part")?;
    let start = existing_part_size(&part, pin.size)?;
    if start < pin.size {
        let (mut target, identity) = open_part_append(&part, start)?;
        let transfer = transport.transfer(
            &Request {
                url: file_url(model, pin.name)?,
                range: (start > 0).then_some(ByteRange { start, end: None }),
                user_agent: USER_AGENT,
                timeout: if pin.name == WEIGHT_NAME {
                    TimeoutPolicy::LargePayload
                } else {
                    TimeoutPolicy::Metadata
                },
            },
            &mut target,
            pin.size - start,
        );
        let response = match transfer {
            Ok(response) => response,
            Err(error) => {
                rollback_partial(&target, &part, start)?;
                return Err(error);
            }
        };
        if let Err(error) = validate_download_response(&response, start, pin.size) {
            rollback_partial(&target, &part, start)?;
            return Err(error);
        }
        if response.bytes != pin.size - start {
            rollback_partial(&target, &part, start)?;
            return Err(DeltafinError::new(format!(
                "{} stopped at {} bytes; expected {}. Re-run to resume.",
                pin.name,
                start + response.bytes,
                pin.size
            )));
        }
        target
            .flush()
            .map_err(|error| io_error("flush Qwen partial", &part, error))?;
        target
            .sync_all()
            .map_err(|error| io_error("fsync Qwen partial", &part, error))?;
        validate_open_identity(&target, identity, pin.size, &part)?;
    }
    verify_regular_digest(&part, pin.size, pin.sha256)?;
    publish_hard_link(&part, &destination, directory)?;
    println!("  {}: verified ({:.1} MB)", pin.name, pin.size as f64 / 1e6);
    Ok(())
}

fn validate_download_response(response: &ResponseMeta, start: u64, size: u64) -> Result<()> {
    if start == 0 {
        if response.status == 200 {
            return Ok(());
        }
        if response.status != 206 {
            return Err(DeltafinError::new(format!(
                "Qwen download returned HTTP status {}",
                response.status
            )));
        }
    } else if response.status != 206 {
        return Err(DeltafinError::new(
            "Qwen server ignored the resume Range request; the .part prefix was preserved",
        ));
    }
    let expected = format!("bytes {start}-{}/{}", size - 1, size);
    if response.headers.get("content-range") != Some(&expected) {
        return Err(DeltafinError::new(format!(
            "Qwen resume returned an unexpected Content-Range; expected {expected:?}"
        )));
    }
    Ok(())
}

fn write_or_validate_manifest(directory: &Path, model: &ModelPin<'_>) -> Result<()> {
    let path = directory.join(MANIFEST_NAME);
    let expected = manifest_bytes(model)?;
    match sys_fs::lstat(&path) {
        Ok(_) => {
            let actual = read_regular_limited(&path, 64 << 10)?;
            if actual != expected {
                return Err(DeltafinError::new(
                    "existing Qwen manifest differs from the pin",
                ));
            }
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect Qwen manifest", &path, error)),
    }
    let part = suffix_path(&path, ".part")?;
    let mut file = secure_create_new(&part, 0o600)?;
    file.write_all(&expected)
        .map_err(|error| io_error("write Qwen manifest", &part, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync Qwen manifest", &part, error))?;
    rename_noreplace(&part, &path)?;
    fsync_directory(directory)
}

fn manifest_bytes(model: &ModelPin<'_>) -> Result<Vec<u8>> {
    let files: BTreeMap<_, _> = model
        .files
        .iter()
        .map(|pin| (pin.name, digest_hex(pin.sha256)))
        .collect();
    let sizes: BTreeMap<_, _> = model.files.iter().map(|pin| (pin.name, pin.size)).collect();
    let document = BTreeMap::from([
        ("files", json!(files)),
        ("repository", json!(model.repository)),
        ("revision", json!(model.revision)),
        ("sizes", json!(sizes)),
    ]);
    let mut raw = serde_json::to_string_pretty(&document)
        .map_err(|error| DeltafinError::new(format!("serialize Qwen manifest: {error}")))?
        .into_bytes();
    raw.push(b'\n');
    Ok(raw)
}

fn validate_staging_names(directory: &Path, model: &ModelPin<'_>) -> Result<()> {
    let mut allowed: BTreeSet<String> = model.files.iter().map(|pin| pin.name.into()).collect();
    for pin in model.files {
        allowed.insert(format!("{}.part", pin.name));
    }
    allowed.insert(MANIFEST_NAME.into());
    allowed.insert(format!("{MANIFEST_NAME}.part"));
    allowed.insert(SPOTLIGHT_MARKER.into());
    for entry in fs::read_dir(directory)
        .map_err(|error| io_error("read Qwen staging directory", directory, error))?
    {
        let entry = entry.map_err(|error| io_error("read Qwen staging entry", directory, error))?;
        let kind = entry
            .file_type()
            .map_err(|error| io_error("inspect Qwen staging entry", &entry.path(), error))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| DeltafinError::new("Qwen staging filename is not UTF-8"))?;
        if kind.is_symlink() || !kind.is_file() || !allowed.contains(&name) || dangerous_name(&name)
        {
            return Err(DeltafinError::new(format!(
                "unexpected or unsafe Qwen staging path {}",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

fn file_url(model: &ModelPin<'_>, name: &str) -> Result<String> {
    if !model.files.iter().any(|pin| pin.name == name) || dangerous_name(name) {
        return Err(DeltafinError::new(format!(
            "Qwen repository file {name:?} is outside the pinned data-only allowlist"
        )));
    }
    Ok(format!("{}{name}?download=true", model.base_url))
}

fn dangerous_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [
        ".py", ".pyc", ".pkl", ".pickle", ".pt", ".pth", ".bin", ".so", ".dylib", ".dll", ".exe",
        ".com", ".bat", ".cmd", ".ps1", ".sh", ".bash", ".zsh", ".fish", ".js", ".mjs", ".cjs",
        ".ts", ".rb", ".pl", ".php", ".lua", ".jar", ".class", ".wasm", ".zip", ".tar", ".tgz",
        ".gz", ".7z",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
}

#[derive(Clone, Copy)]
struct OpenIdentity {
    device: u64,
    inode: u64,
}

fn existing_part_size(path: &Path, maximum: u64) -> Result<u64> {
    match sys_fs::lstat(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => Err(
            DeltafinError::new(format!("refusing unsafe Qwen partial {}", path.display())),
        ),
        Ok(metadata) if metadata.len() > maximum => Err(DeltafinError::new(format!(
            "Qwen partial {} exceeds its pinned size",
            path.display()
        ))),
        Ok(metadata) => Ok(metadata.len()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(error) => Err(io_error("inspect Qwen partial", path, error)),
    }
}

fn open_part_append(path: &Path, expected_size: u64) -> Result<(File, OpenIdentity)> {
    let file = Open::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open Qwen partial safely", path, error))?;
    let metadata = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat Qwen partial", path, error))?;
    if !metadata.is_file() || metadata.len() != expected_size {
        return Err(DeltafinError::new("Qwen partial changed while opening"));
    }
    Ok((
        file,
        OpenIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
        },
    ))
}

fn validate_open_identity(
    file: &File,
    identity: OpenIdentity,
    size: u64,
    path: &Path,
) -> Result<()> {
    let metadata = sys_fs::fstat(&file)
        .map_err(|error| io_error("restat Qwen partial", path, error))?;
    if !metadata.is_file()
        || metadata.dev() != identity.device
        || metadata.ino() != identity.inode
        || metadata.len() != size
    {
        return Err(DeltafinError::new("Qwen partial changed while open"));
    }
    Ok(())
}

fn rollback_partial(file: &File, path: &Path, length: u64) -> Result<()> {
    file.set_len(length)
        .map_err(|error| io_error("roll back rejected Qwen response", path, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync rolled-back Qwen partial", path, error))
}

fn read_regular_limited(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let before = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect Qwen regular file", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > maximum {
        return Err(DeltafinError::new(format!(
            "{} is not a bounded regular non-symlink file",
            path.display()
        )));
    }
    let mut file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open Qwen regular file safely", path, error))?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat Qwen regular file", path, error))?;
    if (opened.dev(), opened.ino(), opened.len()) != (before.dev(), before.ino(), before.len()) {
        return Err(DeltafinError::new(
            "Qwen regular file changed while opening",
        ));
    }
    let mut raw = Vec::with_capacity(opened.len() as usize);
    file.read_to_end(&mut raw)
        .map_err(|error| io_error("read Qwen regular file", path, error))?;
    let after = sys_fs::fstat(&file)
        .map_err(|error| io_error("restat Qwen regular file", path, error))?;
    if (after.dev(), after.ino(), after.len()) != (opened.dev(), opened.ino(), raw.len() as u64) {
        return Err(DeltafinError::new(
            "Qwen regular file changed while reading",
        ));
    }
    Ok(raw)
}

fn ensure_spotlight_marker(directory: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    match sys_fs::lstat(&marker) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(DeltafinError::new("unsafe Qwen Spotlight marker")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            secure_create_new(&marker, 0o600)?
                .sync_all()
                .map_err(|error| io_error("fsync Qwen Spotlight marker", &marker, error))?;
            fsync_directory(directory)
        }
        Err(error) => Err(io_error("inspect Qwen Spotlight marker", &marker, error)),
    }
}

fn validate_spotlight_marker(directory: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    let metadata = sys_fs::lstat(&marker)
        .map_err(|error| io_error("inspect Qwen Spotlight marker", &marker, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(DeltafinError::new("Qwen Spotlight marker is unsafe"));
    }
    Ok(())
}

fn require_real_directory(path: &Path) -> Result<()> {
    let metadata = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect Qwen directory", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "{} is not a real directory",
            path.display()
        )));
    }
    Ok(())
}

fn suffix_path(path: &Path, suffix: &str) -> Result<PathBuf> {
    let mut name = path
        .file_name()
        .ok_or_else(|| DeltafinError::new("path cannot carry a safe suffix"))?
        .to_os_string();
    name.push(suffix);
    Ok(path.with_file_name(name))
}

struct InstallationLock(File);

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
            .map_err(|error| io_error("open Qwen installation lock", &path, error))?;
        if !sys_fs::fstat(&file)
            .map_err(|error| io_error("stat Qwen installation lock", &path, error))?
            .is_file()
        {
            return Err(DeltafinError::new("Qwen installation lock is not regular"));
        }
        if !sys_fs::try_lock_exclusive(&file)
            .map_err(|error| io_error("acquire Qwen installation lock", &path, error))?
        {
            return Err(DeltafinError::new(
                "another Qwen installer is active for this destination",
            ));
        }
        Ok(Self(file))
    }
}

impl Drop for InstallationLock {
    fn drop(&mut self) {
        sys_fs::unlock(&self.0);
    }
}

const fn pin(name: &'static str, size: u64, hex: &'static str) -> FilePin {
    FilePin {
        name,
        size,
        sha256: digest_from_hex(hex),
    }
}

const fn digest_from_hex(hex: &str) -> Digest {
    let bytes = hex.as_bytes();
    assert!(bytes.len() == 64, "pinned SHA-256 must have 64 hex digits");
    let mut digest = [0_u8; 32];
    let mut index = 0;
    while index < 32 {
        digest[index] = (hex_nibble(bytes[index * 2]) << 4) | hex_nibble(bytes[index * 2 + 1]);
        index += 1;
    }
    digest
}

const fn hex_nibble(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'a'..=b'f' => value - b'a' + 10,
        b'A'..=b'F' => value - b'A' + 10,
        _ => panic!("pinned SHA-256 contains a non-hex digit"),
    }
}

fn digest_hex(digest: Digest) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(text, "{byte:02x}").expect("writing to String cannot fail");
    }
    text
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    use crate::packfile::digest_bytes;

    struct FakeTransport {
        responses: VecDeque<(u16, BTreeMap<String, String>, Vec<u8>)>,
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
                .ok_or_else(|| DeltafinError::new("unexpected fake Qwen request"))?;
            if body.len() as u64 > maximum {
                return Err(DeltafinError::new("fake Qwen response exceeded bound"));
            }
            target
                .write_all(&body)
                .map_err(|error| DeltafinError::new(error.to_string()))?;
            Ok(ResponseMeta {
                status,
                headers,
                bytes: body.len() as u64,
            })
        }
    }

    fn fixture_files(config: &[u8], weights: &[u8]) -> Vec<FilePin> {
        vec![
            FilePin {
                name: "config.json",
                size: config.len() as u64,
                sha256: digest_bytes(config),
            },
            FilePin {
                name: WEIGHT_NAME,
                size: weights.len() as u64,
                sha256: digest_bytes(weights),
            },
        ]
    }

    #[test]
    fn production_variants_pin_the_exact_rosters_and_weight_sizes() {
        for model in [WIDE, PROBE] {
            assert_eq!(model.files.len(), 8);
            assert_eq!(
                model
                    .files
                    .iter()
                    .filter(|pin| pin.name.ends_with(".py"))
                    .count(),
                0
            );
            assert!(model.base_url.contains(model.revision));
        }
        assert_eq!(WIDE_FILES[4].size, 3_441_185_608);
        assert_eq!(PROBE_FILES[4].size, 1_192_135_096);
        assert_eq!(
            digest_hex(WIDE_FILES[4].sha256),
            "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"
        );
        assert_eq!(
            digest_hex(PROBE_FILES[4].sha256),
            "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
        );
    }

    #[test]
    fn fake_model_installs_transactionally_without_code() {
        let root = std::env::temp_dir().join(format!("deltafin-qwen-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let config = br#"{"model_type":"qwen3"}"#;
        let weights = b"safe tensor fixture";
        let files = fixture_files(config, weights);
        let model = ModelPin {
            destination: "fixture-qwen",
            repository: "fixture/qwen",
            revision: "0123456789abcdef",
            base_url: "https://fixture.invalid/pinned/",
            files: &files,
        };
        let mut transport = FakeTransport {
            responses: VecDeque::from([
                (200, BTreeMap::new(), config.to_vec()),
                (200, BTreeMap::new(), weights.to_vec()),
            ]),
            requests: Vec::new(),
        };
        install_model(&root, &model, &mut transport).unwrap();
        let destination = root.join(model.destination);
        audit_model(&destination, &model).unwrap();
        assert_eq!(fs::read(destination.join(WEIGHT_NAME)).unwrap(), weights);
        assert!(destination.join(MANIFEST_NAME).is_file());
        assert!(
            transport
                .requests
                .iter()
                .all(|request| !request.url.ends_with(".py?download=true"))
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_legacy_payload_is_adopted_without_network_or_redownload() {
        let root =
            std::env::temp_dir().join(format!("deltafin-qwen-adopt-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let config = br#"{"model_type":"qwen3"}"#;
        let weights = b"already downloaded safe tensor fixture";
        let files = fixture_files(config, weights);
        let model = ModelPin {
            destination: "fixture-qwen",
            repository: "fixture/qwen",
            revision: "0123456789abcdef",
            base_url: "https://fixture.invalid/pinned/",
            files: &files,
        };
        let destination = root.join(model.destination);
        fs::create_dir(&destination).unwrap();
        fs::write(destination.join("config.json"), config).unwrap();
        fs::write(destination.join(WEIGHT_NAME), weights).unwrap();
        fs::write(destination.join("README.md"), b"ignored model card").unwrap();
        fs::write(destination.join(".gitattributes"), b"*.json text\n").unwrap();
        let legacy_cache = destination.join(".cache/huggingface/download");
        fs::create_dir_all(&legacy_cache).unwrap();
        fs::write(
            legacy_cache.join("model.safetensors.metadata"),
            b"ignored download receipt",
        )
        .unwrap();
        let mut transport = FakeTransport {
            responses: VecDeque::new(),
            requests: Vec::new(),
        };

        install_model(&root, &model, &mut transport).unwrap();

        assert!(transport.requests.is_empty());
        assert_eq!(fs::read(destination.join(WEIGHT_NAME)).unwrap(), weights);
        assert_eq!(
            fs::read(destination.join(MANIFEST_NAME)).unwrap(),
            manifest_bytes(&model).unwrap()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_adoption_rejects_unpinned_executable_content() {
        let root = std::env::temp_dir().join(format!(
            "deltafin-qwen-unsafe-adopt-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let config = br#"{"model_type":"qwen3"}"#;
        let weights = b"already downloaded safe tensor fixture";
        let files = fixture_files(config, weights);
        let model = ModelPin {
            destination: "fixture-qwen",
            repository: "fixture/qwen",
            revision: "0123456789abcdef",
            base_url: "https://fixture.invalid/pinned/",
            files: &files,
        };
        let destination = root.join(model.destination);
        fs::create_dir(&destination).unwrap();
        fs::write(destination.join("config.json"), config).unwrap();
        fs::write(destination.join(WEIGHT_NAME), weights).unwrap();
        fs::write(destination.join("modeling_qwen.py"), b"raise SystemExit").unwrap();
        let mut transport = FakeTransport {
            responses: VecDeque::new(),
            requests: Vec::new(),
        };

        let error = install_model(&root, &model, &mut transport).unwrap_err();

        assert!(error.to_string().contains("unexpected Qwen path"));
        assert!(transport.requests.is_empty());
        assert!(!destination.join(MANIFEST_NAME).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_uses_exact_range_and_preserves_prefix_when_ignored() {
        let directory =
            std::env::temp_dir().join(format!("deltafin-qwen-resume-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let complete = b"prefix-suffix";
        let files = [FilePin {
            name: WEIGHT_NAME,
            size: complete.len() as u64,
            sha256: digest_bytes(complete),
        }];
        let model = ModelPin {
            destination: "fixture",
            repository: "fixture/qwen",
            revision: "pin",
            base_url: "https://fixture.invalid/pin/",
            files: &files,
        };
        fs::write(directory.join("model.safetensors.part"), b"prefix-").unwrap();
        let mut ignored = FakeTransport {
            responses: VecDeque::from([(200, BTreeMap::new(), b"suffix".to_vec())]),
            requests: Vec::new(),
        };
        assert!(download_file(&model, &files[0], &directory, &mut ignored).is_err());
        assert_eq!(
            fs::read(directory.join("model.safetensors.part")).unwrap(),
            b"prefix-"
        );

        let start = 7_u64;
        let mut resumed = FakeTransport {
            responses: VecDeque::from([(
                206,
                BTreeMap::from([(
                    "content-range".into(),
                    format!("bytes {start}-{}/{}", complete.len() - 1, complete.len()),
                )]),
                b"suffix".to_vec(),
            )]),
            requests: Vec::new(),
        };
        download_file(&model, &files[0], &directory, &mut resumed).unwrap();
        assert_eq!(fs::read(directory.join(WEIGHT_NAME)).unwrap(), complete);
        assert_eq!(
            resumed.requests[0].range,
            Some(ByteRange { start, end: None })
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn config_and_paths_reject_remote_code() {
        let directory =
            std::env::temp_dir().join(format!("deltafin-qwen-code-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let config = br#"{"model_type":"qwen3","auto_map":{"AutoModel":"evil.py"}}"#;
        let files = [FilePin {
            name: "config.json",
            size: config.len() as u64,
            sha256: digest_bytes(config),
        }];
        fs::write(directory.join("config.json"), config).unwrap();
        let model = ModelPin {
            destination: "fixture",
            repository: "fixture/qwen",
            revision: "pin",
            base_url: "https://fixture.invalid/pin/",
            files: &files,
        };
        assert!(validate_payload(&directory, &model).is_err());
        assert!(file_url(&model, "modeling_qwen.py").is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
