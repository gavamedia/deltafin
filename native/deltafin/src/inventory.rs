//! Strict native validation of K3's generated Safetensors inventory.
//!
//! The inventory is not merely a convenient index.  Its exact bytes are tied
//! to Moonshot's pinned 96-shard checkpoint, and every tensor name later
//! becomes a local filename.  Load therefore authenticates the complete file
//! before parsing it, rejects symlinks and pathname races, and revalidates all
//! shapes, byte ranges, shard identities, and per-shard contiguity.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{DeltafinError, Result};
use crate::packfile::{Digest, digest_bytes};
use crate::sys::fs as sys_fs;

pub const INVENTORY_FILENAME: &str = "tensor_inventory_offsets.json";
pub const PINNED_INVENTORY_BYTES: u64 = 108_581_016;
pub const PINNED_INVENTORY_TENSORS: usize = 497_220;
pub const SHARD_COUNT: usize = 96;
pub const MAX_HEADER_BYTES: u64 = 128 << 20;
const MAX_TENSOR_RANK: usize = 16;
const MAX_DIMENSION: u64 = 1 << 31;
const MAX_SHARD_DATA_BYTES: u64 = 1 << 40;
pub const PINNED_INVENTORY_SHA256: Digest = [
    0xb2, 0x87, 0xe9, 0x65, 0x9a, 0xfb, 0xfd, 0x36, 0x1b, 0x14, 0x85, 0x72, 0x1a, 0x67, 0x03, 0xa5,
    0xbc, 0x55, 0xbb, 0x83, 0x99, 0xed, 0x07, 0x4a, 0x1e, 0x5a, 0x29, 0x80, 0x39, 0x58, 0xb4, 0x25,
];

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRecord {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub offsets: [u64; 2],
    pub shard: String,
    pub hlen: u64,
}

pub type InventoryDocument = BTreeMap<String, TensorRecord>;

/// Emit the exact bytes produced by Python's default `json.dump(document)`.
/// `order` is the deterministic shard-order/header-order roster captured while
/// parsing the 96 authenticated Safetensors headers.
pub(crate) fn write_canonical_inventory(
    mut target: impl Write,
    order: &[String],
    document: &InventoryDocument,
    expected_count: usize,
) -> Result<()> {
    validate_document(document, expected_count)?;
    if order.len() != document.len() {
        return Err(DeltafinError::new(
            "K3 inventory serialization order has the wrong length",
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    target
        .write_all(b"{")
        .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
    for (index, name) in order.iter().enumerate() {
        if !seen.insert(name.as_str()) {
            return Err(DeltafinError::new(format!(
                "duplicate tensor {name:?} in inventory serialization order"
            )));
        }
        let record = document.get(name).ok_or_else(|| {
            DeltafinError::new(format!(
                "inventory serialization order contains unknown tensor {name:?}"
            ))
        })?;
        if !name.is_ascii() {
            return Err(DeltafinError::new(
                "pinned K3 tensor names must be ASCII for canonical serialization",
            ));
        }
        if index != 0 {
            target
                .write_all(b", ")
                .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
        }
        write_json_string(&mut target, name)?;
        target
            .write_all(b": {\"dtype\": ")
            .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
        write_json_string(&mut target, &record.dtype)?;
        target
            .write_all(b", \"shape\": [")
            .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
        for (dimension_index, dimension) in record.shape.iter().enumerate() {
            if dimension_index != 0 {
                target
                    .write_all(b", ")
                    .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
            }
            write!(target, "{dimension}")
                .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
        }
        write!(
            target,
            "], \"offsets\": [{}, {}], \"shard\": ",
            record.offsets[0], record.offsets[1]
        )
        .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
        write_json_string(&mut target, &record.shard)?;
        write!(target, ", \"hlen\": {}}}", record.hlen)
            .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))?;
    }
    if seen.len() != document.len() {
        return Err(DeltafinError::new(
            "K3 inventory serialization order omits tensor records",
        ));
    }
    target
        .write_all(b"}")
        .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))
}

fn write_json_string(target: &mut impl Write, value: &str) -> Result<()> {
    if !value.is_ascii() {
        return Err(DeltafinError::new(
            "pinned K3 inventory strings must be ASCII",
        ));
    }
    let encoded = serde_json::to_string(value)
        .map_err(|error| DeltafinError::new(format!("encode K3 inventory string: {error}")))?;
    target
        .write_all(encoded.as_bytes())
        .map_err(|error| DeltafinError::new(format!("write K3 inventory: {error}")))
}

#[derive(Debug)]
pub struct K3Inventory {
    document: InventoryDocument,
}

impl K3Inventory {
    pub fn load_from_root(root: &Path) -> Result<Self> {
        Self::load(&root.join("k3-meta").join(INVENTORY_FILENAME))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let raw = read_authenticated_inventory(path)?;
        let document: InventoryDocument = serde_json::from_slice(&raw).map_err(|error| {
            DeltafinError::new(format!(
                "parse authenticated K3 tensor inventory {}: {error}",
                path.display()
            ))
        })?;
        validate_document(&document, PINNED_INVENTORY_TENSORS)?;
        Ok(Self { document })
    }

    pub fn from_authenticated_document(document: InventoryDocument) -> Result<Self> {
        validate_document(&document, PINNED_INVENTORY_TENSORS)?;
        Ok(Self { document })
    }

    pub fn len(&self) -> usize {
        self.document.len()
    }

    pub fn is_empty(&self) -> bool {
        self.document.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&TensorRecord> {
        self.document.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorRecord)> {
        self.document
            .iter()
            .map(|(name, record)| (name.as_str(), record))
    }

    pub fn document(&self) -> &InventoryDocument {
        &self.document
    }
}

pub fn validate_tensor_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains('\0') || name.contains(['/', '\\']) {
        return Err(DeltafinError::new(format!(
            "unsafe tensor filename {name:?}: expected one non-empty relative component"
        )));
    }
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(DeltafinError::new(format!(
            "unsafe tensor filename {name:?}: expected one relative component"
        )));
    }
    Ok(())
}

pub fn safe_tensor_path(root: &Path, name: &str) -> Result<PathBuf> {
    validate_tensor_name(name)?;
    Ok(root.join(name))
}

pub fn validate_document(document: &InventoryDocument, expected_count: usize) -> Result<()> {
    if document.len() != expected_count {
        return Err(DeltafinError::new(format!(
            "K3 tensor inventory has {} records; expected {expected_count}",
            document.len()
        )));
    }
    let mut ranges: Vec<Vec<(u64, u64)>> = (0..SHARD_COUNT).map(|_| Vec::new()).collect();
    let mut header_lengths = [None; SHARD_COUNT];
    for (name, record) in document {
        validate_tensor_name(name)?;
        let scalar_bytes = match record.dtype.as_str() {
            "BF16" => 2_u64,
            "F32" => 4_u64,
            "U8" => 1_u64,
            other => {
                return Err(DeltafinError::new(format!(
                    "tensor {name:?} has unsupported dtype {other:?}"
                )));
            }
        };
        if record.shape.is_empty() || record.shape.len() > MAX_TENSOR_RANK {
            return Err(DeltafinError::new(format!(
                "tensor {name:?} rank must be in 1..={MAX_TENSOR_RANK}"
            )));
        }
        let mut elements = 1_u64;
        for &dimension in &record.shape {
            if dimension == 0 || dimension > MAX_DIMENSION {
                return Err(DeltafinError::new(format!(
                    "tensor {name:?} has invalid shape dimension {dimension}"
                )));
            }
            elements = elements.checked_mul(dimension).ok_or_else(|| {
                DeltafinError::new(format!("tensor {name:?} element count overflows u64"))
            })?;
            if elements > MAX_SHARD_DATA_BYTES / scalar_bytes {
                return Err(DeltafinError::new(format!(
                    "tensor {name:?} exceeds the permitted byte bound"
                )));
            }
        }
        let expected_bytes = elements * scalar_bytes;
        let [start, end] = record.offsets;
        if end < start || end > MAX_SHARD_DATA_BYTES || end - start != expected_bytes {
            return Err(DeltafinError::new(format!(
                "tensor {name:?} byte range [{start},{end}) disagrees with its {}-byte shape",
                expected_bytes
            )));
        }
        let shard = parse_shard_name(&record.shard)?;
        if record.hlen == 0 || record.hlen > MAX_HEADER_BYTES {
            return Err(DeltafinError::new(format!(
                "tensor {name:?} has invalid Safetensors header length {}",
                record.hlen
            )));
        }
        match header_lengths[shard] {
            Some(previous) if previous != record.hlen => {
                return Err(DeltafinError::new(format!(
                    "shard {} has inconsistent header lengths {previous} and {}",
                    shard + 1,
                    record.hlen
                )));
            }
            Some(_) => {}
            None => header_lengths[shard] = Some(record.hlen),
        }
        ranges[shard].push((start, end));
    }
    for (index, shard_ranges) in ranges.iter_mut().enumerate() {
        if shard_ranges.is_empty() {
            return Err(DeltafinError::new(format!(
                "K3 tensor inventory has no records for shard {}",
                index + 1
            )));
        }
        shard_ranges.sort_unstable();
        let mut expected_start = 0_u64;
        for &(start, end) in shard_ranges.iter() {
            if start != expected_start {
                return Err(DeltafinError::new(format!(
                    "shard {} has a gap or overlap: expected offset {expected_start}, found {start}",
                    index + 1
                )));
            }
            expected_start = end;
        }
    }
    Ok(())
}

fn parse_shard_name(name: &str) -> Result<usize> {
    const PREFIX: &str = "model-";
    const SUFFIX: &str = "-of-000096.safetensors";
    let digits = name
        .strip_prefix(PREFIX)
        .and_then(|rest| rest.strip_suffix(SUFFIX))
        .filter(|digits| digits.len() == 5 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| DeltafinError::new(format!("invalid K3 shard filename {name:?}")))?;
    let one_based = digits
        .parse::<usize>()
        .map_err(|_| DeltafinError::new(format!("invalid K3 shard number in {name:?}")))?;
    if !(1..=SHARD_COUNT).contains(&one_based) {
        return Err(DeltafinError::new(format!(
            "K3 shard number {one_based} is outside 1..={SHARD_COUNT}"
        )));
    }
    Ok(one_based - 1)
}

fn read_authenticated_inventory(path: &Path) -> Result<Vec<u8>> {
    let before = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect K3 tensor inventory", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(DeltafinError::new(format!(
            "K3 tensor inventory is not a regular non-symlink file: {}",
            path.display()
        )));
    }
    if before.len() != PINNED_INVENTORY_BYTES {
        return Err(DeltafinError::new(format!(
            "K3 tensor inventory has {} bytes; pinned revision requires {PINNED_INVENTORY_BYTES}",
            before.len()
        )));
    }
    let mut file =
        File::open(path).map_err(|error| io_error("open K3 tensor inventory", path, error))?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat opened K3 tensor inventory", path, error))?;
    if !opened.is_file()
        || opened.dev() != before.dev()
        || opened.ino() != before.ino()
        || opened.len() != before.len()
    {
        return Err(DeltafinError::new(format!(
            "K3 tensor inventory changed while it was opened: {}",
            path.display()
        )));
    }
    let capacity = usize::try_from(PINNED_INVENTORY_BYTES)
        .map_err(|_| DeltafinError::new("K3 tensor inventory does not fit this host"))?;
    let mut raw = Vec::with_capacity(capacity);
    file.read_to_end(&mut raw)
        .map_err(|error| io_error("read K3 tensor inventory", path, error))?;
    let after = sys_fs::fstat(&file)
        .map_err(|error| io_error("restat opened K3 tensor inventory", path, error))?;
    if after.dev() != opened.dev()
        || after.ino() != opened.ino()
        || after.len() != opened.len()
        || raw.len() as u64 != PINNED_INVENTORY_BYTES
    {
        return Err(DeltafinError::new(format!(
            "K3 tensor inventory changed while it was read: {}",
            path.display()
        )));
    }
    if digest_bytes(&raw) != PINNED_INVENTORY_SHA256 {
        return Err(DeltafinError::new(format!(
            "K3 tensor inventory SHA-256 does not match the pinned checkpoint: {}",
            path.display()
        )));
    }
    Ok(raw)
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tiny_document() -> InventoryDocument {
        (1..=SHARD_COUNT)
            .map(|shard| {
                (
                    format!("tensor_{shard}"),
                    TensorRecord {
                        dtype: "U8".into(),
                        shape: vec![1],
                        offsets: [0, 1],
                        shard: format!("model-{shard:05}-of-000096.safetensors"),
                        hlen: 128,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn complete_bounded_inventory_is_accepted() {
        let document = tiny_document();
        validate_document(&document, SHARD_COUNT).unwrap();
    }

    #[test]
    fn tensor_names_are_single_safe_components() {
        for unsafe_name in ["", ".", "..", "a/b", "a\\b", "/absolute", "nul\0x"] {
            assert!(
                validate_tensor_name(unsafe_name).is_err(),
                "accepted unsafe tensor name {unsafe_name:?}"
            );
        }
        validate_tensor_name("language_model.model.layers.0.weight").unwrap();
    }

    #[test]
    fn range_shape_and_shard_gaps_fail_closed() {
        let mut document = tiny_document();
        document.get_mut("tensor_1").unwrap().offsets = [1, 2];
        assert!(validate_document(&document, SHARD_COUNT).is_err());

        let mut document = tiny_document();
        document.get_mut("tensor_2").unwrap().shape = vec![2];
        assert!(validate_document(&document, SHARD_COUNT).is_err());

        let mut document = tiny_document();
        document.get_mut("tensor_3").unwrap().shard = "model-00097-of-000096.safetensors".into();
        assert!(validate_document(&document, SHARD_COUNT).is_err());
    }

    #[test]
    fn serde_rejects_unknown_tensor_metadata_fields() {
        let raw = br#"{"dtype":"U8","shape":[1],"offsets":[0,1],"shard":"model-00001-of-000096.safetensors","hlen":8,"extra":1}"#;
        assert!(serde_json::from_slice::<TensorRecord>(raw).is_err());
    }

    #[test]
    fn canonical_writer_matches_python_default_json_separators_and_field_order() {
        let document = tiny_document();
        let order: Vec<_> = document.keys().cloned().collect();
        let mut raw = Vec::new();
        write_canonical_inventory(&mut raw, &order, &document, SHARD_COUNT).unwrap();
        let prefix = br#"{"tensor_1": {"dtype": "U8", "shape": [1], "offsets": [0, 1], "shard": "model-00001-of-000096.safetensors", "hlen": 128}"#;
        assert!(raw.starts_with(prefix));
        assert!(!raw.ends_with(b"\n"));
    }

    #[test]
    #[ignore = "reads and validates the installed 108 MB pinned inventory"]
    fn validates_the_installed_complete_inventory() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let inventory = K3Inventory::load_from_root(&root).unwrap();
        assert_eq!(inventory.len(), PINNED_INVENTORY_TENSORS);
    }

    #[test]
    #[ignore = "reads and reserializes the installed 108 MB pinned inventory"]
    fn canonical_writer_reproduces_the_installed_python_inventory_identity() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../k3-meta")
            .join(INVENTORY_FILENAME);
        let authenticated = K3Inventory::load(&path).unwrap();
        let raw = fs::read(&path).unwrap();
        let ordered: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&raw).unwrap();
        let order: Vec<_> = ordered.keys().cloned().collect();
        let mut canonical = Vec::with_capacity(raw.len());
        write_canonical_inventory(
            &mut canonical,
            &order,
            authenticated.document(),
            PINNED_INVENTORY_TENSORS,
        )
        .unwrap();
        assert_eq!(canonical.len() as u64, PINNED_INVENTORY_BYTES);
        assert_eq!(digest_bytes(&canonical), PINNED_INVENTORY_SHA256);
        assert_eq!(canonical, raw);
    }
}
