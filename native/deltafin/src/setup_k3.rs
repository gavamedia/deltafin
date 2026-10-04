//! Native metadata-only setup for the pinned Kimi K3 checkpoint.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::cli::SetupK3Args;
use crate::dspark_checkpoint::strict_json;
use crate::error::{DeltafinError, Result};
use crate::inventory::{
    INVENTORY_FILENAME, InventoryDocument, K3Inventory, MAX_HEADER_BYTES, PINNED_INVENTORY_BYTES,
    PINNED_INVENTORY_SHA256, PINNED_INVENTORY_TENSORS, SHARD_COUNT, TensorRecord,
    validate_document, validate_tensor_name, write_canonical_inventory,
};
use crate::k3_source::{self, NATIVE_METADATA_FILES, PinnedMetadataFile};
use crate::model::ModelSpec;
use crate::packfile::digest_bytes;
use crate::tokenizer::K3Tokenizer;
use crate::sys::fs::{self as sys_fs, Open};
use crate::trusted_download::{
    ByteRange, NativeHttpsTransport, Request, ResponseMeta, TimeoutPolicy, Transport,
    fsync_directory, publish_hard_link, rename_noreplace, secure_create_new, verify_regular_digest,
};

const USER_AGENT: &str = "deltafin-setup/1";
const SPOTLIGHT_MARKER: &str = ".metadata_never_index";

pub(crate) fn run(arguments: SetupK3Args) -> Result<()> {
    let root = arguments
        .model_root
        .or_else(|| std::env::var_os("DELTAFIN_ROOT").map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let metadata = root.join("k3-meta");
    if arguments.check {
        audit_local(&metadata)?;
        println!(
            "K3 metadata audit passed: {}@{} at {}",
            k3_source::MODEL_REPOSITORY,
            k3_source::MODEL_REVISION,
            metadata.display()
        );
        return Ok(());
    }
    debug_assert!(arguments.meta_only);
    let mut transport = NativeHttpsTransport;
    install(&metadata, &mut transport)
}

/// Install the pinned inert metadata at an explicit model root.
///
/// One-shot setup uses this boundary instead of changing `DELTAFIN_ROOT`,
/// which is process-global and therefore unsafe to use as an internal API.
pub(crate) fn install_at(root: &Path) -> Result<()> {
    let mut transport = NativeHttpsTransport;
    install(&root.join("k3-meta"), &mut transport)
}

/// Exact logical bytes in the immutable metadata payload and inventory.
pub(crate) fn exact_payload_bytes() -> u64 {
    NATIVE_METADATA_FILES
        .iter()
        .map(|spec| spec.size)
        .sum::<u64>()
        + PINNED_INVENTORY_BYTES
}

fn install(destination: &Path, transport: &mut dyn Transport) -> Result<()> {
    let destination = absolute_path(destination)?;
    let parent = destination
        .parent()
        .ok_or_else(|| DeltafinError::new("K3 metadata destination has no parent"))?;
    require_real_directory(parent)?;
    let _lock = InstallationLock::acquire(&destination)?;
    match sys_fs::lstat(&destination) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            ensure_spotlight_marker(&destination)?;
            audit_local(&destination)?;
            println!(
                "K3 metadata is already installed and verified at {}",
                destination.display()
            );
            return Ok(());
        }
        Ok(_) => {
            return Err(DeltafinError::new(format!(
                "refusing to replace existing K3 metadata path {}",
                destination.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(io_error(
                "inspect K3 metadata destination",
                &destination,
                error,
            ));
        }
    }

    let staging = suffix_path(&destination, ".installing")?;
    match sys_fs::lstat(&staging) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
        Ok(_) => {
            return Err(DeltafinError::new(format!(
                "K3 metadata staging path is unsafe: {}",
                staging.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(&staging)
                .map_err(|error| io_error("create K3 metadata staging", &staging, error))?;
            sys_fs::restrict_to_owner(&staging)
                .map_err(|error| io_error("secure K3 metadata staging", &staging, error))?;
            fsync_directory(parent)?;
        }
        Err(error) => return Err(io_error("inspect K3 metadata staging", &staging, error)),
    }
    ensure_spotlight_marker(&staging)?;
    validate_staging_names(&staging)?;
    let base = k3_source::base_url()?;
    for spec in &NATIVE_METADATA_FILES {
        fetch_metadata_file(&base, spec, &staging, transport)?;
    }
    ModelSpec::load(&staging.join("config.json"))?;
    K3Tokenizer::load_from_root(&staging)?;

    let inventory_path = staging.join(INVENTORY_FILENAME);
    if sys_fs::lstat(&inventory_path).is_ok() {
        K3Inventory::load(&inventory_path)?;
    } else {
        let (document, order) = build_inventory(transport, &base, PINNED_INVENTORY_TENSORS)?;
        publish_inventory(&staging, &document, &order)?;
    }
    audit_local(&staging)?;
    rename_noreplace(&staging, &destination)?;
    fsync_directory(parent)?;
    audit_local(&destination)?;
    println!(
        "K3 inert metadata and {}-tensor inventory installed at {}. Weights were not downloaded.",
        PINNED_INVENTORY_TENSORS,
        destination.display()
    );
    Ok(())
}

pub(crate) fn load_installed_inventory(root: &Path) -> Result<K3Inventory> {
    audit_local(&root.join("k3-meta")).map_err(|error| {
        DeltafinError::new(format!(
            "K3 metadata is not ready for weight installation; run `deltafin setup-k3 --meta-only` first: {error}"
        ))
    })
}

fn audit_local(directory: &Path) -> Result<K3Inventory> {
    require_real_directory(directory)?;
    validate_spotlight_marker(directory)?;
    for spec in &NATIVE_METADATA_FILES {
        verify_regular_digest(&directory.join(spec.name), spec.size, spec.sha256)?;
    }
    verify_regular_digest(
        &directory.join(INVENTORY_FILENAME),
        PINNED_INVENTORY_BYTES,
        PINNED_INVENTORY_SHA256,
    )?;
    ModelSpec::load(&directory.join("config.json"))?;
    K3Tokenizer::load_from_root(directory)?;
    let inventory = K3Inventory::load(&directory.join(INVENTORY_FILENAME))?;
    if inventory.len() != PINNED_INVENTORY_TENSORS {
        return Err(DeltafinError::new("K3 inventory tensor count changed"));
    }
    // The semantic validators reopen their inert inputs. Reauthenticate every
    // path after parsing so a cooperative concurrent replacement cannot turn
    // a successful check into validation of bytes other than the pinned set.
    for spec in &NATIVE_METADATA_FILES {
        verify_regular_digest(&directory.join(spec.name), spec.size, spec.sha256)?;
    }
    verify_regular_digest(
        &directory.join(INVENTORY_FILENAME),
        PINNED_INVENTORY_BYTES,
        PINNED_INVENTORY_SHA256,
    )?;
    Ok(inventory)
}

fn fetch_metadata_file(
    base: &str,
    spec: &PinnedMetadataFile,
    directory: &Path,
    transport: &mut dyn Transport,
) -> Result<()> {
    let destination = directory.join(spec.name);
    match sys_fs::lstat(&destination) {
        Ok(_) => {
            verify_regular_digest(&destination, spec.size, spec.sha256)?;
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect K3 metadata file", &destination, error)),
    }
    let mut payload = Vec::with_capacity(
        usize::try_from(spec.size)
            .map_err(|_| DeltafinError::new("K3 metadata file does not fit usize"))?,
    );
    let meta = transport.transfer(
        &Request {
            url: k3_source::metadata_url(base, spec.name)?,
            range: None,
            user_agent: USER_AGENT,
            timeout: TimeoutPolicy::Metadata,
        },
        &mut payload,
        spec.size,
    )?;
    if meta.status != 200 || meta.bytes != spec.size || digest_bytes(&payload) != spec.sha256 {
        return Err(DeltafinError::new(format!(
            "{} response does not match its exact size/SHA pin",
            spec.name
        )));
    }
    let part = suffix_path(&destination, ".part")?;
    let mut file = secure_create_new(&part, 0o600)?;
    file.write_all(&payload)
        .map_err(|error| io_error("write K3 metadata partial", &part, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync K3 metadata partial", &part, error))?;
    verify_regular_digest(&part, spec.size, spec.sha256)?;
    publish_hard_link(&part, &destination, directory)
}

fn build_inventory(
    transport: &mut dyn Transport,
    base: &str,
    expected_count: usize,
) -> Result<(InventoryDocument, Vec<String>)> {
    let mut document = BTreeMap::new();
    let mut order = Vec::with_capacity(expected_count);
    for shard_number in 1..=SHARD_COUNT {
        let (shard, header_length, header) = fetch_shard_header(transport, base, shard_number)?;
        let entries = parse_shard_header(&shard, header_length, &header)?;
        for (name, record) in entries {
            if document.insert(name.clone(), record).is_some() {
                return Err(DeltafinError::new(format!(
                    "duplicate tensor {name:?} found in {shard}"
                )));
            }
            order.push(name);
        }
    }
    validate_document(&document, expected_count)?;
    Ok((document, order))
}

fn fetch_shard_header(
    transport: &mut dyn Transport,
    base: &str,
    shard_number: usize,
) -> Result<(String, u64, Vec<u8>)> {
    let shard = k3_source::shard_name(shard_number)?;
    let url = k3_source::shard_url(base, shard_number)?;
    let (length_bytes, first_total) = fetch_range(transport, &url, 0, 7, None)?;
    let header_length = u64::from_le_bytes(
        length_bytes
            .as_slice()
            .try_into()
            .map_err(|_| DeltafinError::new("Safetensors header prefix is not eight bytes"))?,
    );
    if header_length == 0 || header_length > MAX_HEADER_BYTES {
        return Err(DeltafinError::new(format!(
            "{shard} has invalid header length {header_length}"
        )));
    }
    let header_end = 8_u64
        .checked_add(header_length)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| DeltafinError::new("K3 shard header range overflowed"))?;
    if header_end >= first_total {
        return Err(DeltafinError::new(format!(
            "{shard} header extends past its ranged file length"
        )));
    }
    let (header, _) = fetch_range(transport, &url, 8, header_end, Some(first_total))?;
    Ok((shard, header_length, header))
}

fn fetch_range(
    transport: &mut dyn Transport,
    url: &str,
    start: u64,
    end: u64,
    expected_total: Option<u64>,
) -> Result<(Vec<u8>, u64)> {
    let expected = end
        .checked_sub(start)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| DeltafinError::new("invalid K3 shard range"))?;
    let mut payload = Vec::with_capacity(
        usize::try_from(expected)
            .map_err(|_| DeltafinError::new("K3 shard range does not fit usize"))?,
    );
    let meta = transport.transfer(
        &Request {
            url: url.into(),
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
    if meta.status != 206 || meta.bytes != expected {
        return Err(DeltafinError::new(
            "K3 shard server ignored an exact byte range",
        ));
    }
    let (actual_start, actual_end, total) = parse_content_range(&meta)?;
    if (actual_start, actual_end) != (start, end) || expected_total.is_some_and(|pin| pin != total)
    {
        return Err(DeltafinError::new(
            "K3 shard Content-Range does not match the request",
        ));
    }
    Ok((payload, total))
}

fn parse_content_range(meta: &ResponseMeta) -> Result<(u64, u64, u64)> {
    let value = meta
        .headers
        .get("content-range")
        .ok_or_else(|| DeltafinError::new("ranged K3 shard response lacks Content-Range"))?;
    let value = value
        .strip_prefix("bytes ")
        .ok_or_else(|| DeltafinError::new("invalid K3 shard Content-Range unit"))?;
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| DeltafinError::new("malformed K3 shard Content-Range"))?;
    let (start, end) = range
        .split_once('-')
        .ok_or_else(|| DeltafinError::new("malformed K3 shard Content-Range bounds"))?;
    let parse = |value: &str| {
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(DeltafinError::new("non-decimal K3 shard Content-Range"));
        }
        value
            .parse::<u64>()
            .map_err(|_| DeltafinError::new("K3 shard Content-Range overflows u64"))
    };
    let parsed = (parse(start)?, parse(end)?, parse(total)?);
    if parsed.1 < parsed.0 || parsed.2 <= parsed.1 {
        return Err(DeltafinError::new(
            "invalid K3 shard Content-Range ordering",
        ));
    }
    Ok(parsed)
}

fn parse_shard_header(
    shard: &str,
    header_length: u64,
    raw: &[u8],
) -> Result<Vec<(String, TensorRecord)>> {
    if raw.len() as u64 != header_length {
        return Err(DeltafinError::new(format!(
            "{shard} header length changed while reading"
        )));
    }
    let value = strict_json(raw, &format!("{shard} Safetensors header"))?;
    let root = value
        .as_object()
        .ok_or_else(|| DeltafinError::new(format!("{shard} header must be a JSON object")))?;
    if let Some(metadata) = root.get("__metadata__") {
        let metadata = metadata.as_object().ok_or_else(|| {
            DeltafinError::new(format!("{shard} Safetensors metadata must be an object"))
        })?;
        if metadata.values().any(|value| !value.is_string()) {
            return Err(DeltafinError::new(format!(
                "{shard} Safetensors metadata values must be strings"
            )));
        }
    }
    let mut entries = Vec::with_capacity(root.len());
    for (name, descriptor) in root {
        if name == "__metadata__" {
            continue;
        }
        validate_tensor_name(name)?;
        let descriptor = descriptor.as_object().ok_or_else(|| {
            DeltafinError::new(format!("tensor {name:?} in {shard} has invalid metadata"))
        })?;
        let fields: BTreeSet<_> = descriptor.keys().map(String::as_str).collect();
        if fields != BTreeSet::from(["data_offsets", "dtype", "shape"]) {
            return Err(DeltafinError::new(format!(
                "tensor {name:?} in {shard} has unknown or missing descriptor fields"
            )));
        }
        let dtype = descriptor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| DeltafinError::new(format!("tensor {name:?} dtype is not a string")))?
            .to_owned();
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| DeltafinError::new(format!("tensor {name:?} shape is not an array")))?
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    DeltafinError::new(format!("tensor {name:?} shape contains a non-u64"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|values| values.len() == 2)
            .ok_or_else(|| {
                DeltafinError::new(format!("tensor {name:?} data_offsets is not a pair"))
            })?;
        let offsets = [
            offsets[0].as_u64().ok_or_else(|| {
                DeltafinError::new(format!("tensor {name:?} start offset is not a u64"))
            })?,
            offsets[1].as_u64().ok_or_else(|| {
                DeltafinError::new(format!("tensor {name:?} end offset is not a u64"))
            })?,
        ];
        entries.push((
            name.clone(),
            TensorRecord {
                dtype,
                shape,
                offsets,
                shard: shard.into(),
                hlen: header_length,
            },
        ));
    }
    if entries.is_empty() {
        return Err(DeltafinError::new(format!("{shard} contains no tensors")));
    }
    Ok(entries)
}

fn publish_inventory(
    directory: &Path,
    document: &InventoryDocument,
    order: &[String],
) -> Result<()> {
    validate_document(document, PINNED_INVENTORY_TENSORS)?;
    let destination = directory.join(INVENTORY_FILENAME);
    let part = suffix_path(&destination, ".part")?;
    let mut file = secure_create_new(&part, 0o600)?;
    write_canonical_inventory(&mut file, order, document, PINNED_INVENTORY_TENSORS)?;
    file.sync_all()
        .map_err(|error| io_error("fsync K3 inventory partial", &part, error))?;
    verify_regular_digest(&part, PINNED_INVENTORY_BYTES, PINNED_INVENTORY_SHA256)?;
    rename_noreplace(&part, &destination)?;
    fsync_directory(directory)?;
    K3Inventory::load(&destination)?;
    Ok(())
}

fn validate_staging_names(directory: &Path) -> Result<()> {
    let mut allowed: BTreeSet<_> = NATIVE_METADATA_FILES
        .iter()
        .map(|spec| spec.name.to_owned())
        .collect();
    allowed.insert(INVENTORY_FILENAME.into());
    allowed.insert(SPOTLIGHT_MARKER.into());
    for spec in &NATIVE_METADATA_FILES {
        allowed.insert(format!("{}.part", spec.name));
    }
    allowed.insert(format!("{INVENTORY_FILENAME}.part"));
    for entry in fs::read_dir(directory)
        .map_err(|error| io_error("read K3 metadata staging", directory, error))?
    {
        let entry =
            entry.map_err(|error| io_error("read K3 metadata staging entry", directory, error))?;
        let kind = entry
            .file_type()
            .map_err(|error| io_error("inspect K3 metadata staging entry", &entry.path(), error))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| DeltafinError::new("K3 metadata filename is not UTF-8"))?;
        if kind.is_symlink() || !kind.is_file() || !allowed.contains(&name) {
            return Err(DeltafinError::new(format!(
                "unexpected or unsafe K3 metadata staging path {}",
                entry.path().display()
            )));
        }
        if name.to_ascii_lowercase().ends_with(".py") {
            return Err(DeltafinError::new(
                "executable Python is forbidden from native K3 metadata staging",
            ));
        }
    }
    Ok(())
}

fn ensure_spotlight_marker(directory: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    match sys_fs::lstat(&marker) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(DeltafinError::new("unsafe K3 Spotlight marker")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            secure_create_new(&marker, 0o600)?
                .sync_all()
                .map_err(|error| io_error("fsync K3 Spotlight marker", &marker, error))?;
            fsync_directory(directory)
        }
        Err(error) => Err(io_error("inspect K3 Spotlight marker", &marker, error)),
    }
}

fn validate_spotlight_marker(directory: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    let metadata = sys_fs::lstat(&marker)
        .map_err(|error| io_error("inspect K3 Spotlight marker", &marker, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "K3 Spotlight marker is not a regular non-symlink file: {}",
            marker.display()
        )));
    }
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.into())
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .map_err(|error| DeltafinError::new(format!("resolve current directory: {error}")))
    }
}

fn suffix_path(path: &Path, suffix: &str) -> Result<PathBuf> {
    let mut name = path
        .file_name()
        .ok_or_else(|| DeltafinError::new("path cannot carry a safe suffix"))?
        .to_os_string();
    name.push(suffix);
    Ok(path.with_file_name(name))
}

fn require_real_directory(path: &Path) -> Result<()> {
    let metadata = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect K3 directory", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "{} is not a real directory",
            path.display()
        )));
    }
    Ok(())
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
            .map_err(|error| io_error("open K3 metadata lock", &path, error))?;
        if !sys_fs::fstat(&file)
            .map_err(|error| io_error("stat K3 metadata lock", &path, error))?
            .is_file()
        {
            return Err(DeltafinError::new("K3 metadata lock is not regular"));
        }
        if !sys_fs::try_lock_exclusive(&file)
            .map_err(|error| io_error("acquire K3 metadata installation lock", &path, error))?
        {
            return Err(DeltafinError::new(
                "another native K3 metadata installer is active",
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

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

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
                .ok_or_else(|| DeltafinError::new("unexpected fake K3 request"))?;
            if body.len() as u64 > maximum {
                return Err(DeltafinError::new("fake K3 response exceeded bound"));
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

    fn range_response(
        body: Vec<u8>,
        start: u64,
        end: u64,
        total: u64,
    ) -> (u16, BTreeMap<String, String>, Vec<u8>) {
        (
            206,
            BTreeMap::from([(
                "content-range".into(),
                format!("bytes {start}-{end}/{total}"),
            )]),
            body,
        )
    }

    #[test]
    fn model_free_fake_headers_build_in_shard_then_header_order() {
        let mut responses = VecDeque::new();
        for shard in 1..=SHARD_COUNT {
            let header = if shard == 1 {
                // Deliberately non-lexical: the pinned Python inventory keeps
                // each source header's insertion order.
                b"{\"z_1\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"a_1\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}".to_vec()
            } else {
                format!(
                    "{{\"z_{shard}\":{{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}}}"
                )
                .into_bytes()
            };
            let total = header.len() as u64 + 9;
            responses.push_back(range_response(
                (header.len() as u64).to_le_bytes().to_vec(),
                0,
                7,
                total,
            ));
            responses.push_back(range_response(
                header.clone(),
                8,
                7 + header.len() as u64,
                total,
            ));
        }
        let mut transport = FakeTransport {
            responses,
            requests: Vec::new(),
        };
        let (document, order) =
            build_inventory(&mut transport, "https://fixture.invalid/", SHARD_COUNT + 1).unwrap();
        assert_eq!(document.len(), SHARD_COUNT + 1);
        assert_eq!(order[0], "z_1");
        assert_eq!(order[1], "a_1");
        assert_eq!(order[96], "z_96");
        assert_eq!(transport.requests.len(), SHARD_COUNT * 2);
        assert!(
            transport
                .requests
                .iter()
                .all(|request| request.range.is_some())
        );
    }

    #[test]
    fn malformed_duplicate_and_unknown_descriptors_are_rejected() {
        let shard = "model-00001-of-000096.safetensors";
        for raw in [
            br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"evil":1}}"#.as_slice(),
            br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#.as_slice(),
            br#"{"a":{"dtype":"U8","shape":"bad","data_offsets":[0,1]}}"#.as_slice(),
        ] {
            assert!(parse_shard_header(shard, raw.len() as u64, raw).is_err());
        }
    }

    #[test]
    fn executable_upstream_files_are_outside_the_native_allowlist() {
        let base = k3_source::base_url_from("fixture.invalid", "/pinned/").unwrap();
        for name in [
            "modeling_kimi_k3.py",
            "tokenization_kimi.py",
            "encoding_k3.py",
        ] {
            assert!(k3_source::metadata_url(&base, name).is_err());
        }
    }

    #[test]
    fn fake_metadata_transfer_publishes_only_an_exact_pin() {
        let directory = std::env::temp_dir().join(format!(
            "deltafin-setup-k3-metadata-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let payload = b"inert metadata".to_vec();
        let spec = PinnedMetadataFile {
            name: "LICENSE",
            size: payload.len() as u64,
            sha256: digest_bytes(&payload),
        };
        let mut transport = FakeTransport {
            responses: VecDeque::from([(200, BTreeMap::new(), payload.clone())]),
            requests: Vec::new(),
        };
        fetch_metadata_file(
            "https://fixture.invalid/pinned/",
            &spec,
            &directory,
            &mut transport,
        )
        .unwrap();
        assert_eq!(fs::read(directory.join("LICENSE")).unwrap(), payload);
        assert!(!directory.join("LICENSE.part").exists());
        assert_eq!(transport.requests.len(), 1);
        fs::remove_dir_all(directory).unwrap();
    }
}
