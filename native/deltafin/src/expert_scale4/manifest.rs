//! Strict activation manifest for a complete `K3SC4V2` corpus.
//!
//! A sidecar directory is inert until this exact document is published. The
//! loader admits no partial roster, path supplied by JSON, duplicate key, or
//! non-canonical digest. Individual record headers and hashes are still
//! checked lazily when selected for execution.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{
    FILE_BYTES, HEADER_BYTES, LAYOUT_ID, MANIFEST_NAME, MANIFEST_SCHEMA, MANIFEST_VERSION, VERSION,
    cache_neutral, drop_completed_cache, exponent_table, parse_header, record_digest,
    source_identity,
};
use crate::dspark_checkpoint::strict_json;
use crate::error::{DeltafinError, Result};
use crate::packfile::{Digest, DigestState, digest_bytes};
use crate::sys::fs::{self as sys_fs, Open};

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestRow {
    // Field declaration order is Python's sort_keys=True order. It is part of
    // entries_sha256 and therefore intentionally stable.
    pub bases: [u8; 3],
    pub expert: u16,
    pub layer: u32,
    pub record_sha256: String,
    pub source_sha256: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerFileDocument {
    // Stable lexicographic field order, matching Python sort_keys=True.
    file_bytes: u64,
    layer: u32,
    name: String,
    records: usize,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestDocument {
    // Stable lexicographic field order, matching Python sort_keys=True.
    complete: bool,
    entries: Vec<ManifestRow>,
    entries_sha256: String,
    expected_count: usize,
    expected_names_sha256: String,
    format: String,
    format_version: u32,
    layer_files: Vec<LayerFileDocument>,
    layout_id: u32,
    manifest_version: u32,
    schema: String,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ManifestEntry {
    pub layer: u32,
    pub expert: u16,
    pub record_offset: u64,
    pub source_sha256: Digest,
    pub record_sha256: Digest,
    pub bases: [u8; 3],
}

/// Canonical record count and byte extent for one activated layer sidecar.
///
/// This index is built once from the manifest's already-authenticated
/// `layer_files` roster. Hot expert plans can therefore resolve a layer with a
/// bounded binary search instead of rescanning every expert entry in the
/// complete corpus.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ManifestLayerExtent {
    pub layer: u32,
    pub records: usize,
    pub file_bytes: u64,
}

#[derive(Debug)]
pub struct Scale4Manifest {
    root: PathBuf,
    entries: Box<[ManifestEntry]>,
    layer_extents: Box<[ManifestLayerExtent]>,
    expected_names_sha256: Digest,
    entries_sha256: Digest,
}

impl Scale4Manifest {
    pub fn load_full(root: impl AsRef<Path>) -> Result<Self> {
        let names = full_raw_names();
        Self::load_for_raw_names(root, &names)
    }

    pub fn load_for_raw_names(root: impl AsRef<Path>, raw_names: &[String]) -> Result<Self> {
        let root = root.as_ref();
        let manifest_path = root.join(MANIFEST_NAME);
        let metadata = sys_fs::lstat(&manifest_path)
            .map_err(|error| io_error("inspect scale4 manifest", &manifest_path, error))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
            return Err(invalid(format!(
                "{} is not a non-empty regular manifest",
                manifest_path.display()
            )));
        }
        let raw = fs::read(&manifest_path)
            .map_err(|error| io_error("read scale4 manifest", &manifest_path, error))?;
        let value = strict_json(&raw, "scale4 manifest")?;
        let document: ManifestDocument = serde_json::from_value(value)
            .map_err(|error| invalid(format!("scale4 manifest schema: {error}")))?;
        let canonical = build_document(document.entries.clone(), raw_names)?;
        if document != canonical {
            return Err(invalid("scale4 activation manifest is not canonical"));
        }

        for layer_file in &document.layer_files {
            let expected_name = format!("L{}.sc4", layer_file.layer);
            if layer_file.name != expected_name {
                return Err(invalid("scale4 layer filename is not canonical"));
            }
            let path = root.join(&expected_name);
            if path.parent() != Some(root) {
                return Err(invalid("scale4 layer path escapes its corpus root"));
            }
            let metadata = sys_fs::lstat(&path)
                .map_err(|error| io_error("inspect scale4 layer", &path, error))?;
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || metadata.len() != layer_file.file_bytes
            {
                return Err(invalid(format!(
                    "{} is not a regular {}-byte scale4 layer",
                    path.display(),
                    layer_file.file_bytes
                )));
            }
        }

        let layer_extents = document
            .layer_files
            .iter()
            .map(|layer| ManifestLayerExtent {
                layer: layer.layer,
                records: layer.records,
                file_bytes: layer.file_bytes,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let entries = document
            .entries
            .into_iter()
            .map(|row| {
                let record_offset = u64::from(row.expert)
                    .checked_mul(FILE_BYTES as u64)
                    .ok_or_else(|| invalid("scale4 record offset overflowed"))?;
                Ok(ManifestEntry {
                    layer: row.layer,
                    expert: row.expert,
                    record_offset,
                    source_sha256: parse_digest(&row.source_sha256)?,
                    record_sha256: parse_digest(&row.record_sha256)?,
                    bases: row.bases,
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_boxed_slice();
        Ok(Self {
            root: root.to_path_buf(),
            entries,
            layer_extents,
            expected_names_sha256: parse_digest(&canonical.expected_names_sha256)?,
            entries_sha256: parse_digest(&canonical.entries_sha256)?,
        })
    }

    pub fn entry(&self, layer: u32, expert: u16) -> Option<&ManifestEntry> {
        self.entries
            .binary_search_by_key(&(layer, expert), |entry| (entry.layer, entry.expert))
            .ok()
            .map(|index| &self.entries[index])
    }

    pub fn entries(&self) -> &[ManifestEntry] {
        &self.entries
    }

    pub fn layer_extent(&self, layer: u32) -> Option<ManifestLayerExtent> {
        self.layer_extents
            .binary_search_by_key(&layer, |extent| extent.layer)
            .ok()
            .map(|index| self.layer_extents[index])
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn expected_names_sha256(&self) -> Digest {
        self.expected_names_sha256
    }

    pub const fn entries_sha256(&self) -> Digest {
        self.entries_sha256
    }

    /// Authenticate every fixed record in one sequential pass per layer.
    ///
    /// Normal inference verifies only selected experts lazily. Conversion
    /// resume is different: an already-published activation marker must never
    /// turn a truncated or modified corpus into a successful no-op. Grouping
    /// by layer avoids the Python converter's one-open-per-record behavior.
    pub fn verify_all_records(&self) -> Result<()> {
        let mut first = 0;
        while first < self.entries.len() {
            let layer = self.entries[first].layer;
            let mut end = first + 1;
            while end < self.entries.len() && self.entries[end].layer == layer {
                end += 1;
            }
            self.verify_layer_records(layer, &self.entries[first..end])?;
            first = end;
        }
        Ok(())
    }

    fn verify_layer_records(&self, layer: u32, entries: &[ManifestEntry]) -> Result<()> {
        if entries.is_empty()
            || entries.iter().enumerate().any(|(index, entry)| {
                entry.layer != layer
                    || usize::from(entry.expert) != index
                    || entry.record_offset != index as u64 * FILE_BYTES as u64
            })
        {
            return Err(invalid(format!(
                "scale4 layer {layer} records are not canonical and contiguous"
            )));
        }
        let path = self.root.join(format!("L{layer}.sc4"));
        let expected_bytes = (entries.len() as u64)
            .checked_mul(FILE_BYTES as u64)
            .ok_or_else(|| invalid("scale4 verification extent overflowed"))?;
        let observed = sys_fs::lstat(&path)
            .map_err(|error| io_error("inspect scale4 layer for verification", &path, error))?;
        if observed.file_type().is_symlink()
            || !observed.is_file()
            || observed.len() != expected_bytes
        {
            return Err(invalid(format!(
                "{} is not a regular {expected_bytes}-byte scale4 layer",
                path.display()
            )));
        }
        let observed_identity = source_identity(&observed);
        let mut file = Open::new()
            .read(true)
            .no_follow()
            .open(&path)
            .map_err(|error| io_error("open scale4 layer for verification", &path, error))?;
        let opened = sys_fs::fstat(&file)
            .map_err(|error| io_error("stat opened scale4 layer", &path, error))?;
        if source_identity(&opened) != observed_identity {
            return Err(invalid(format!(
                "scale4 layer changed while opening: {}",
                path.display()
            )));
        }
        cache_neutral(&file);
        let mut record = vec![0_u8; FILE_BYTES];
        for entry in entries {
            read_exact_retry(&mut file, &mut record, &path)?;
            let header = parse_header(&record[..HEADER_BYTES])?;
            if header.source_sha256 != entry.source_sha256
                || header.bases != entry.bases
                || record_digest(&record)? != entry.record_sha256
            {
                return Err(invalid(format!(
                    "scale4 record disagrees with the manifest for L{layer}-E{}",
                    entry.expert
                )));
            }
        }
        let after = sys_fs::fstat(&file)
            .map_err(|error| io_error("restat verified scale4 layer", &path, error))?;
        if source_identity(&after) != observed_identity {
            return Err(invalid(format!(
                "scale4 layer changed during verification: {}",
                path.display()
            )));
        }
        let cache_bytes = usize::try_from(expected_bytes)
            .map_err(|_| invalid("scale4 verification extent exceeds this platform"))?;
        drop_completed_cache(&file, cache_bytes);
        Ok(())
    }
}

fn read_exact_retry(file: &mut File, target: &mut [u8], path: &Path) -> Result<()> {
    let mut filled = 0;
    while filled < target.len() {
        match file.read(&mut target[filled..]) {
            Ok(0) => {
                return Err(invalid(format!(
                    "short scale4 read from {} at {filled} of {} bytes",
                    path.display(),
                    target.len()
                )));
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io_error("read scale4 record", path, error)),
        }
    }
    Ok(())
}

pub fn manifest_bytes(rows: Vec<ManifestRow>, raw_names: &[String]) -> Result<Box<[u8]>> {
    let document = build_document(rows, raw_names)?;
    let mut payload = serde_json::to_vec_pretty(&document)
        .map_err(|error| invalid(format!("serialize scale4 manifest: {error}")))?;
    payload.push(b'\n');
    Ok(payload.into_boxed_slice())
}

fn build_document(mut rows: Vec<ManifestRow>, raw_names: &[String]) -> Result<ManifestDocument> {
    let expected = canonical_expected(raw_names)?;
    rows.sort_by_key(|row| (row.layer, row.expert));
    if rows.len() != expected.len() {
        return Err(invalid(format!(
            "refusing partial scale4 activation: {} rows for {} expected experts",
            rows.len(),
            expected.len()
        )));
    }
    for (row, expected) in rows.iter_mut().zip(&expected) {
        if (row.layer, row.expert) != (expected.layer, expected.expert) {
            return Err(invalid(format!(
                "scale4 roster differs at L{}-E{}",
                expected.layer, expected.expert
            )));
        }
        for &base in &row.bases {
            exponent_table(base)?;
        }
        row.source_sha256 = canonical_digest_text(&row.source_sha256)?;
        row.record_sha256 = canonical_digest_text(&row.record_sha256)?;
    }

    let mut layers: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
    for expected in &expected {
        layers
            .entry(expected.layer)
            .or_default()
            .push(expected.expert);
    }
    let mut layer_files = Vec::with_capacity(layers.len());
    for (layer, experts) in layers {
        for (index, &expert) in experts.iter().enumerate() {
            if usize::from(expert) != index {
                return Err(invalid(format!(
                    "layer {layer} expected experts must be contiguous from zero"
                )));
            }
        }
        let file_bytes = (experts.len() as u64)
            .checked_mul(FILE_BYTES as u64)
            .ok_or_else(|| invalid("scale4 layer extent overflowed"))?;
        layer_files.push(LayerFileDocument {
            file_bytes,
            layer,
            name: format!("L{layer}.sc4"),
            records: experts.len(),
        });
    }

    let entries_payload = serde_json::to_vec(&rows)
        .map_err(|error| invalid(format!("serialize scale4 entries: {error}")))?;
    let entries_sha256 = hex_digest(digest_bytes(&entries_payload));
    let mut names_hasher = DigestState::new();
    for expected in &expected {
        names_hasher.update(expected.scale4_name.as_bytes());
        names_hasher.update(b"\n");
    }
    Ok(ManifestDocument {
        complete: true,
        entries: rows,
        entries_sha256,
        expected_count: expected.len(),
        expected_names_sha256: hex_digest(names_hasher.finalize()),
        format: "K3SC4V2".into(),
        format_version: VERSION,
        layer_files,
        layout_id: LAYOUT_ID,
        manifest_version: MANIFEST_VERSION,
        schema: MANIFEST_SCHEMA.into(),
    })
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ExpectedEntry {
    layer: u32,
    expert: u16,
    scale4_name: String,
}

fn canonical_expected(raw_names: &[String]) -> Result<Vec<ExpectedEntry>> {
    let mut expected = Vec::with_capacity(raw_names.len());
    for name in raw_names {
        let (layer, expert, extension) = parse_expert_name(name)?;
        if extension != "bin" {
            return Err(invalid(format!("expected a raw .bin name, got {name:?}")));
        }
        expected.push(ExpectedEntry {
            layer,
            expert,
            scale4_name: format!("L{layer}-E{expert}.sc4"),
        });
    }
    expected.sort_by_key(|entry| (entry.layer, entry.expert));
    if expected
        .windows(2)
        .any(|pair| (pair[0].layer, pair[0].expert) == (pair[1].layer, pair[1].expert))
    {
        return Err(invalid("duplicate expected scale4 expert"));
    }
    Ok(expected)
}

fn parse_expert_name(name: &str) -> Result<(u32, u16, &'static str)> {
    let body = name
        .strip_prefix('L')
        .ok_or_else(|| invalid(format!("invalid expert filename {name:?}")))?;
    let (body, extension) = if let Some(body) = body.strip_suffix(".bin") {
        (body, "bin")
    } else if let Some(body) = body.strip_suffix(".sc4") {
        (body, "sc4")
    } else {
        return Err(invalid(format!("invalid expert filename {name:?}")));
    };
    let (layer, expert) = body
        .split_once("-E")
        .ok_or_else(|| invalid(format!("invalid expert filename {name:?}")))?;
    if !canonical_positive_decimal(layer) || !canonical_nonnegative_decimal(expert) {
        return Err(invalid(format!("invalid expert filename {name:?}")));
    }
    let layer = layer
        .parse::<u32>()
        .map_err(|_| invalid(format!("expert layer is too large in {name:?}")))?;
    let expert = expert
        .parse::<u16>()
        .map_err(|_| invalid(format!("expert ID is too large in {name:?}")))?;
    Ok((layer, expert, extension))
}

fn canonical_positive_decimal(text: &str) -> bool {
    !text.is_empty() && !text.starts_with('0') && text.as_bytes().iter().all(u8::is_ascii_digit)
}

fn canonical_nonnegative_decimal(text: &str) -> bool {
    text == "0"
        || (!text.is_empty()
            && !text.starts_with('0')
            && text.as_bytes().iter().all(u8::is_ascii_digit))
}

fn full_raw_names() -> Vec<String> {
    let mut names = Vec::with_capacity(92 * 896);
    for layer in 1..=92 {
        for expert in 0..896 {
            names.push(format!("L{layer}-E{expert}.bin"));
        }
    }
    names
}

fn canonical_digest_text(text: &str) -> Result<String> {
    Ok(hex_digest(parse_digest(text)?))
}

fn parse_digest(text: &str) -> Result<Digest> {
    if text.len() != 64 {
        return Err(invalid(
            "scale4 digest must contain 64 lowercase hex digits",
        ));
    }
    let mut digest = [0_u8; 32];
    for (slot, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        digest[slot] = high << 4 | low;
    }
    if hex_digest(digest) != text {
        return Err(invalid("scale4 digest must use canonical lowercase hex"));
    }
    Ok(digest)
}

fn hex_nibble(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(invalid("scale4 digest contains a non-lowercase-hex digit")),
    }
}

fn hex_digest(digest: Digest) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").unwrap();
    }
    output
}

fn invalid(message: impl Into<String>) -> DeltafinError {
    DeltafinError::new(format!("invalid K3SC4V2 manifest: {}", message.into()))
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-scale4-manifest-{nonce}-{}-{sequence}",
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

    fn fixture() -> (Vec<String>, Vec<ManifestRow>) {
        let names = vec!["L1-E0.bin".into(), "L1-E1.bin".into()];
        let rows = vec![
            ManifestRow {
                bases: [7, 8, 9],
                expert: 0,
                layer: 1,
                record_sha256: "22".repeat(32),
                source_sha256: "11".repeat(32),
            },
            ManifestRow {
                bases: [10, 11, 12],
                expert: 1,
                layer: 1,
                record_sha256: "44".repeat(32),
                source_sha256: "33".repeat(32),
            },
        ];
        (names, rows)
    }

    #[test]
    fn manifest_bytes_match_python_sort_keys_contract() {
        let (names, rows) = fixture();
        let bytes = manifest_bytes(rows, &names).unwrap();
        assert_eq!(
            hex_digest(digest_bytes(&bytes)),
            "ed11c1c4270b0306d1bde35a7336c73a8d70c032d10b14c5d294ee350b0a6306"
        );
    }

    #[test]
    fn strict_loader_admits_only_the_complete_canonical_roster() {
        let directory = TestDirectory::new();
        let (names, rows) = fixture();
        let bytes = manifest_bytes(rows, &names).unwrap();
        fs::write(directory.0.join(MANIFEST_NAME), bytes).unwrap();
        let layer = File::create(directory.0.join("L1.sc4")).unwrap();
        layer.set_len(2 * FILE_BYTES as u64).unwrap();
        layer.sync_all().unwrap();

        let manifest = Scale4Manifest::load_for_raw_names(&directory.0, &names).unwrap();
        assert_eq!(manifest.entries().len(), 2);
        assert_eq!(
            manifest.entry(1, 1).unwrap().record_offset,
            FILE_BYTES as u64
        );
        assert!(manifest.entry(1, 2).is_none());
        assert_eq!(
            manifest.layer_extent(1),
            Some(ManifestLayerExtent {
                layer: 1,
                records: 2,
                file_bytes: 2 * FILE_BYTES as u64,
            })
        );
        assert_eq!(manifest.layer_extent(2), None);
    }

    #[test]
    fn layer_extent_index_preserves_sparse_partial_test_corpora() {
        let directory = TestDirectory::new();
        let (mut names, mut rows) = fixture();
        names.push("L3-E0.bin".into());
        rows.push(ManifestRow {
            bases: [13, 14, 15],
            expert: 0,
            layer: 3,
            record_sha256: "66".repeat(32),
            source_sha256: "55".repeat(32),
        });
        fs::write(
            directory.0.join(MANIFEST_NAME),
            manifest_bytes(rows, &names).unwrap(),
        )
        .unwrap();
        File::create(directory.0.join("L1.sc4"))
            .unwrap()
            .set_len(2 * FILE_BYTES as u64)
            .unwrap();
        File::create(directory.0.join("L3.sc4"))
            .unwrap()
            .set_len(FILE_BYTES as u64)
            .unwrap();

        let manifest = Scale4Manifest::load_for_raw_names(&directory.0, &names).unwrap();
        assert_eq!(
            manifest.layer_extent(1),
            Some(ManifestLayerExtent {
                layer: 1,
                records: 2,
                file_bytes: 2 * FILE_BYTES as u64,
            })
        );
        assert_eq!(manifest.layer_extent(2), None);
        assert_eq!(
            manifest.layer_extent(3),
            Some(ManifestLayerExtent {
                layer: 3,
                records: 1,
                file_bytes: FILE_BYTES as u64,
            })
        );
    }

    #[test]
    fn strict_loader_rejects_duplicate_json_keys_and_partial_rosters() {
        let directory = TestDirectory::new();
        fs::write(
            directory.0.join(MANIFEST_NAME),
            br#"{"complete":true,"complete":true}"#,
        )
        .unwrap();
        let (names, _) = fixture();
        assert!(Scale4Manifest::load_for_raw_names(&directory.0, &names).is_err());

        let (_, mut rows) = fixture();
        rows.pop();
        assert!(manifest_bytes(rows, &names).is_err());
    }

    #[test]
    fn full_record_verification_authenticates_every_manifest_entry() {
        use crate::expert_scale4::{SCALE_BYTES, build_record, pack_scale4};

        let directory = TestDirectory::new();
        let names = vec!["L1-E0.bin".into(), "L1-E1.bin".into()];
        let mut rows = Vec::with_capacity(names.len());
        let mut layer = Vec::with_capacity(names.len() * FILE_BYTES);
        for expert in 0..names.len() {
            let bases = [32 + expert as u8, 64 + expert as u8, 96 + expert as u8];
            let w1_values = vec![bases[0]; SCALE_BYTES];
            let w2_values = vec![bases[1]; SCALE_BYTES];
            let w3_values = vec![bases[2]; SCALE_BYTES];
            let w1 = pack_scale4(&w1_values, Some(bases[0])).unwrap();
            let w2 = pack_scale4(&w2_values, Some(bases[1])).unwrap();
            let w3 = pack_scale4(&w3_values, Some(bases[2])).unwrap();
            let source_sha256 = [expert as u8 + 1; 32];
            let record = build_record(source_sha256, [&w1, &w2, &w3]).unwrap();
            rows.push(ManifestRow {
                bases,
                expert: expert as u16,
                layer: 1,
                record_sha256: hex_digest(record_digest(&record).unwrap()),
                source_sha256: hex_digest(source_sha256),
            });
            layer.extend_from_slice(&record);
        }
        fs::write(
            directory.0.join(MANIFEST_NAME),
            manifest_bytes(rows, &names).unwrap(),
        )
        .unwrap();
        fs::write(directory.0.join("L1.sc4"), &layer).unwrap();

        let manifest = Scale4Manifest::load_for_raw_names(&directory.0, &names).unwrap();
        manifest.verify_all_records().unwrap();

        layer[FILE_BYTES + HEADER_BYTES] ^= 1;
        fs::write(directory.0.join("L1.sc4"), &layer).unwrap();
        let error = manifest.verify_all_records().unwrap_err();
        assert!(error.to_string().contains("disagrees with the manifest"));
    }

    #[test]
    fn expert_names_are_strict_and_canonical() {
        assert_eq!(parse_expert_name("L92-E895.bin").unwrap(), (92, 895, "bin"));
        for bad in [
            "L0-E0.bin",
            "L01-E0.bin",
            "L1-E00.bin",
            "L1-E0.bin/../x",
            "L1-E0.py",
        ] {
            assert!(parse_expert_name(bad).is_err(), "{bad}");
        }
    }
}
