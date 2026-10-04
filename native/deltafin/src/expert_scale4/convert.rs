//! Resumable native builder for a lossless `K3SC4V2` expert corpus.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::manifest::{ManifestRow, Scale4Manifest, manifest_bytes};
use super::{
    FILE_BYTES, HEADER_BYTES, SOURCE_BYTES, SourceIdentity, cache_neutral, drop_completed_cache,
    encode_raw_expert, parse_header, record_digest, source_identity,
};
use crate::error::{DeltafinError, Result};
use crate::trusted_download::{fsync_directory, publish_hard_link, secure_create_new};
use crate::sys::fs::{self as sys_fs, Open};

const FIRST_LAYER: u32 = 1;
const LAST_LAYER: u32 = 92;
const EXPERTS_PER_LAYER: u16 = 896;
const SPOTLIGHT_MARKER: &str = ".metadata_never_index";
pub const DEFAULT_CONVERSION_WORKERS: usize = 4;
pub const MAX_CONVERSION_WORKERS: usize = 16;

static NEXT_PART: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub workers: usize,
    pub resume: bool,
}

impl ConvertOptions {
    pub fn under(model_root: impl AsRef<Path>) -> Self {
        let root = model_root.as_ref();
        Self {
            source_root: root.join("k3-experts"),
            output_root: root.join("k3-experts-scale4"),
            workers: DEFAULT_CONVERSION_WORKERS,
            resume: true,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConvertReport {
    pub already_complete: bool,
    pub records: usize,
    pub converted_records: usize,
    pub resumed_records: usize,
    pub sidecar_bytes: u64,
    pub manifest: PathBuf,
}

#[derive(Debug, Clone)]
struct SourceWitness {
    path: PathBuf,
    identity: SourceIdentity,
    row: ManifestRow,
    resumed: bool,
}

pub fn convert_full(options: &ConvertOptions) -> Result<ConvertReport> {
    let names = full_raw_names();
    convert_for_raw_names(options, &names)
}

pub fn convert_for_raw_names(
    options: &ConvertOptions,
    raw_names: &[String],
) -> Result<ConvertReport> {
    if options.workers == 0 || options.workers > MAX_CONVERSION_WORKERS {
        return Err(invalid(format!(
            "converter workers must be in 1..={MAX_CONVERSION_WORKERS}"
        )));
    }
    validate_real_directory(&options.source_root, "raw expert root")?;
    validate_source_set(&options.source_root, raw_names)?;
    create_real_directory(&options.output_root)?;
    let manifest_path = options.output_root.join(super::MANIFEST_NAME);
    if sys_fs::lstat(&manifest_path).is_ok() {
        let manifest = Scale4Manifest::load_for_raw_names(&options.output_root, raw_names)?;
        manifest.verify_all_records()?;
        return Ok(ConvertReport {
            already_complete: true,
            records: manifest.entries().len(),
            converted_records: 0,
            resumed_records: manifest.entries().len(),
            sidecar_bytes: (manifest.entries().len() as u64)
                .checked_mul(FILE_BYTES as u64)
                .ok_or_else(|| invalid("sidecar corpus extent overflowed"))?,
            manifest: manifest_path,
        });
    }

    let jobs = layer_jobs(raw_names)?;
    let slots = (0..jobs.len())
        .map(|_| Mutex::new(None))
        .collect::<Vec<Mutex<Option<Result<Vec<SourceWitness>>>>>>();
    let next = AtomicUsize::new(0);
    let workers = options.workers.min(jobs.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some((layer, experts)) = jobs.get(index) else {
                        break;
                    };
                    let result = convert_layer(
                        &options.source_root,
                        &options.output_root,
                        *layer,
                        experts,
                        options.resume,
                    );
                    *slots[index].lock().unwrap() = Some(result);
                }
            });
        }
    });

    let mut witnesses = Vec::with_capacity(raw_names.len());
    for slot in slots {
        let result = slot
            .into_inner()
            .map_err(|_| invalid("scale4 converter result lock was poisoned"))?
            .ok_or_else(|| invalid("scale4 converter worker produced no result"))??;
        witnesses.extend(result);
    }
    for witness in &witnesses {
        validate_witness(witness)?;
    }
    let rows = witnesses
        .iter()
        .map(|witness| witness.row.clone())
        .collect::<Vec<_>>();
    let payload = manifest_bytes(rows, raw_names)?;
    publish_manifest(&options.output_root, &manifest_path, &payload)?;
    let activated = Scale4Manifest::load_for_raw_names(&options.output_root, raw_names)?;
    let converted_records = witnesses.iter().filter(|row| !row.resumed).count();
    let resumed_records = witnesses.len() - converted_records;
    Ok(ConvertReport {
        already_complete: false,
        records: activated.entries().len(),
        converted_records,
        resumed_records,
        sidecar_bytes: (activated.entries().len() as u64)
            .checked_mul(FILE_BYTES as u64)
            .ok_or_else(|| invalid("sidecar corpus extent overflowed"))?,
        manifest: manifest_path,
    })
}

fn convert_layer(
    source_root: &Path,
    output_root: &Path,
    layer: u32,
    experts: &[u16],
    resume: bool,
) -> Result<Vec<SourceWitness>> {
    if experts.is_empty()
        || experts
            .iter()
            .enumerate()
            .any(|(index, &expert)| usize::from(expert) != index)
    {
        return Err(invalid(format!(
            "layer {layer} experts must be contiguous from zero"
        )));
    }
    let destination = output_root.join(format!("L{layer}.sc4"));
    match sys_fs::lstat(&destination) {
        Ok(_) if !resume => {
            return Err(invalid(format!(
                "sidecar already exists and resume is disabled: {}",
                destination.display()
            )));
        }
        Ok(_) => return verify_existing_layer(source_root, &destination, layer, experts),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect sidecar layer", &destination, error)),
    }

    let part = unique_part_path(output_root, &format!("L{layer}.sc4"));
    let mut output = secure_create_new(&part, 0o600)?;
    cache_neutral(&output);
    let mut witnesses = Vec::with_capacity(experts.len());
    let result = (|| -> Result<()> {
        for &expert in experts {
            let source = source_root.join(format!("L{layer}-E{expert}.bin"));
            let encoded = encode_raw_expert(&source)?;
            output
                .write_all(&encoded.record)
                .map_err(|error| io_error("write scale4 layer", &part, error))?;
            witnesses.push(SourceWitness {
                path: source,
                identity: encoded.source_identity,
                row: ManifestRow {
                    bases: encoded.bases,
                    expert,
                    layer,
                    record_sha256: hex_digest(encoded.record_sha256),
                    source_sha256: hex_digest(encoded.source_sha256),
                },
                resumed: false,
            });
        }
        let expected = (experts.len() as u64)
            .checked_mul(FILE_BYTES as u64)
            .ok_or_else(|| invalid("scale4 layer extent overflowed"))?;
        if output
            .metadata()
            .map_err(|error| io_error("stat scale4 layer partial", &part, error))?
            .len()
            != expected
        {
            return Err(invalid("scale4 layer partial has the wrong extent"));
        }
        output
            .sync_all()
            .map_err(|error| io_error("fsync scale4 layer", &part, error))?;
        let cache_bytes = usize::try_from(expected)
            .map_err(|_| invalid("scale4 layer extent exceeds this platform"))?;
        drop_completed_cache(&output, cache_bytes);
        drop(output);
        verify_layer_records(&part, &witnesses)?;
        for witness in &witnesses {
            validate_witness(witness)?;
        }
        publish_hard_link(&part, &destination, output_root)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result?;
    Ok(witnesses)
}

fn verify_existing_layer(
    source_root: &Path,
    destination: &Path,
    layer: u32,
    experts: &[u16],
) -> Result<Vec<SourceWitness>> {
    let expected = (experts.len() as u64)
        .checked_mul(FILE_BYTES as u64)
        .ok_or_else(|| invalid("scale4 layer extent overflowed"))?;
    let metadata = sys_fs::lstat(destination)
        .map_err(|error| io_error("inspect resumed scale4 layer", destination, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != expected {
        return Err(invalid(format!(
            "resumed scale4 layer {} is not a regular {expected}-byte file",
            destination.display()
        )));
    }
    let mut sidecar = Open::new()
        .read(true)
        .no_follow()
        .open(destination)
        .map_err(|error| io_error("open resumed scale4 layer", destination, error))?;
    cache_neutral(&sidecar);
    let mut record = vec![0_u8; FILE_BYTES];
    let mut witnesses = Vec::with_capacity(experts.len());
    for &expert in experts {
        read_exact_retry(&mut sidecar, &mut record, destination)?;
        let source = source_root.join(format!("L{layer}-E{expert}.bin"));
        let encoded = encode_raw_expert(&source)?;
        if encoded.record.as_ref() != record {
            return Err(invalid(format!(
                "resumed sidecar differs from exact raw expert L{layer}-E{expert}"
            )));
        }
        witnesses.push(SourceWitness {
            path: source,
            identity: encoded.source_identity,
            row: ManifestRow {
                bases: encoded.bases,
                expert,
                layer,
                record_sha256: hex_digest(encoded.record_sha256),
                source_sha256: hex_digest(encoded.source_sha256),
            },
            resumed: true,
        });
    }
    let cache_bytes = usize::try_from(expected)
        .map_err(|_| invalid("scale4 layer extent exceeds this platform"))?;
    drop_completed_cache(&sidecar, cache_bytes);
    Ok(witnesses)
}

fn verify_layer_records(path: &Path, witnesses: &[SourceWitness]) -> Result<()> {
    let mut file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("reopen durable scale4 layer", path, error))?;
    cache_neutral(&file);
    let mut record = vec![0_u8; FILE_BYTES];
    for witness in witnesses {
        read_exact_retry(&mut file, &mut record, path)?;
        let header = parse_header(&record[..HEADER_BYTES])?;
        if header.bases != witness.row.bases
            || hex_digest(header.source_sha256) != witness.row.source_sha256
            || hex_digest(record_digest(&record)?) != witness.row.record_sha256
        {
            return Err(invalid(format!(
                "durable sidecar changed for L{}-E{}",
                witness.row.layer, witness.row.expert
            )));
        }
    }
    Ok(())
}

fn validate_source_set(source_root: &Path, raw_names: &[String]) -> Result<()> {
    if raw_names.is_empty() {
        return Err(invalid("refusing to activate an empty scale4 corpus"));
    }
    let jobs = layer_jobs(raw_names)?;
    for (layer, experts) in jobs {
        for expert in experts {
            let path = source_root.join(format!("L{layer}-E{expert}.bin"));
            let metadata = sys_fs::lstat(&path)
                .map_err(|error| io_error("inspect raw expert", &path, error))?;
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || metadata.len() != SOURCE_BYTES as u64
            {
                return Err(invalid(format!(
                    "{} must be a regular {SOURCE_BYTES}-byte raw expert",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

fn layer_jobs(raw_names: &[String]) -> Result<Vec<(u32, Vec<u16>)>> {
    let mut layers: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
    for name in raw_names {
        let (layer, expert) = parse_raw_name(name)?;
        layers.entry(layer).or_default().push(expert);
    }
    let mut jobs = Vec::with_capacity(layers.len());
    for (layer, mut experts) in layers {
        experts.sort_unstable();
        if experts.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(format!("duplicate raw expert in layer {layer}")));
        }
        if experts
            .iter()
            .enumerate()
            .any(|(index, &expert)| usize::from(expert) != index)
        {
            return Err(invalid(format!(
                "layer {layer} raw experts must be contiguous from zero"
            )));
        }
        jobs.push((layer, experts));
    }
    Ok(jobs)
}

fn parse_raw_name(name: &str) -> Result<(u32, u16)> {
    let body = name
        .strip_prefix('L')
        .and_then(|body| body.strip_suffix(".bin"))
        .ok_or_else(|| invalid(format!("non-canonical raw expert name {name:?}")))?;
    let (layer, expert) = body
        .split_once("-E")
        .ok_or_else(|| invalid(format!("non-canonical raw expert name {name:?}")))?;
    if layer.is_empty()
        || layer.starts_with('0')
        || !layer.as_bytes().iter().all(u8::is_ascii_digit)
        || expert.is_empty()
        || (expert.len() > 1 && expert.starts_with('0'))
        || !expert.as_bytes().iter().all(u8::is_ascii_digit)
    {
        return Err(invalid(format!("non-canonical raw expert name {name:?}")));
    }
    Ok((
        layer
            .parse()
            .map_err(|_| invalid(format!("raw expert layer is too large in {name:?}")))?,
        expert
            .parse()
            .map_err(|_| invalid(format!("raw expert ID is too large in {name:?}")))?,
    ))
}

fn validate_witness(witness: &SourceWitness) -> Result<()> {
    let metadata = sys_fs::lstat(&witness.path)
        .map_err(|error| io_error("reinspect raw expert", &witness.path, error))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || source_identity(&metadata) != witness.identity
    {
        return Err(invalid(format!(
            "raw source changed before scale4 activation: {}",
            witness.path.display()
        )));
    }
    Ok(())
}

fn publish_manifest(root: &Path, destination: &Path, payload: &[u8]) -> Result<()> {
    let part = unique_part_path(root, super::MANIFEST_NAME);
    let result = (|| -> Result<()> {
        let mut file = secure_create_new(&part, 0o600)?;
        file.write_all(payload)
            .map_err(|error| io_error("write scale4 manifest", &part, error))?;
        file.sync_all()
            .map_err(|error| io_error("fsync scale4 manifest", &part, error))?;
        drop(file);
        publish_hard_link(&part, destination, root)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result
}

fn create_real_directory(path: &Path) -> Result<()> {
    ensure_directory_without_links(path)?;
    validate_real_directory(path, "scale4 output root")?;
    ensure_spotlight_marker(path)?;
    fsync_directory(path)
}

fn ensure_directory_without_links(path: &Path) -> Result<()> {
    let selected = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    match sys_fs::lstat(selected) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => return Ok(()),
        Ok(_) => {
            return Err(invalid(format!(
                "scale4 directory component is not a real directory: {}",
                selected.display()
            )));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect scale4 directory", selected, error)),
    }
    let parent = selected
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent == selected {
        return Err(invalid(format!(
            "cannot establish scale4 directory ancestor for {}",
            selected.display()
        )));
    }
    ensure_directory_without_links(parent)?;
    match fs::create_dir(selected) {
        Ok(()) => fsync_directory(parent)?,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io_error("create scale4 directory", selected, error)),
    }
    validate_real_directory(selected, "scale4 directory component")
}

fn ensure_spotlight_marker(directory: &Path) -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }
    let marker = directory.join(SPOTLIGHT_MARKER);
    match sys_fs::lstat(&marker) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(invalid(format!(
            "unsafe Spotlight marker {}",
            marker.display()
        ))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            secure_create_new(&marker, 0o600)?
                .sync_all()
                .map_err(|error| io_error("fsync scale4 Spotlight marker", &marker, error))?;
            fsync_directory(directory)
        }
        Err(error) => Err(io_error("inspect scale4 Spotlight marker", &marker, error)),
    }
}

fn validate_real_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = sys_fs::lstat(path).map_err(|error| io_error("inspect", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!("{label} is not a real directory")));
    }
    Ok(())
}

fn unique_part_path(root: &Path, name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let sequence = NEXT_PART.fetch_add(1, Ordering::Relaxed);
    root.join(format!(
        ".{name}.tmp-{nonce}-{}-{sequence}",
        std::process::id()
    ))
}

fn read_exact_retry(file: &mut File, target: &mut [u8], path: &Path) -> Result<()> {
    let mut filled = 0;
    while filled < target.len() {
        match file.read(&mut target[filled..]) {
            Ok(0) => {
                return Err(invalid(format!(
                    "short read from {} at {filled} of {} bytes",
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

fn full_raw_names() -> Vec<String> {
    let mut names = Vec::with_capacity(
        (LAST_LAYER - FIRST_LAYER + 1) as usize * usize::from(EXPERTS_PER_LAYER),
    );
    for layer in FIRST_LAYER..=LAST_LAYER {
        for expert in 0..EXPERTS_PER_LAYER {
            names.push(format!("L{layer}-E{expert}.bin"));
        }
    }
    names
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").unwrap();
    }
    output
}

fn invalid(message: impl Into<String>) -> DeltafinError {
    DeltafinError::new(format!(
        "native scale4 conversion failed: {}",
        message.into()
    ))
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "deltafin-scale4-convert-{}-{}",
                std::process::id(),
                NEXT_TEST.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_raw(path: &Path, salt: usize) {
        let mut file = File::create(path).unwrap();
        for matrix in 0..3 {
            let packed = (0..super::super::PACKED_BYTES)
                .map(|index| ((index + matrix * 31 + salt) & 255) as u8)
                .collect::<Vec<_>>();
            file.write_all(&packed).unwrap();
            let base = [50_u8, 100, 200][matrix];
            let scales = (0..super::super::SCALE_BYTES)
                .map(|index| base + ((index + salt) % 16) as u8)
                .collect::<Vec<_>>();
            file.write_all(&scales).unwrap();
        }
        file.sync_all().unwrap();
    }

    #[test]
    fn converts_resumes_and_activates_a_small_exact_corpus() {
        let directory = TestDirectory::new();
        let source = directory.0.join("raw");
        let output = directory.0.join("scale4");
        fs::create_dir(&source).unwrap();
        write_raw(&source.join("L1-E0.bin"), 0);
        write_raw(&source.join("L1-E1.bin"), 7);
        let names = vec!["L1-E0.bin".into(), "L1-E1.bin".into()];
        let options = ConvertOptions {
            source_root: source,
            output_root: output.clone(),
            workers: 2,
            resume: true,
        };

        let first = convert_for_raw_names(&options, &names).unwrap();
        assert!(!first.already_complete);
        assert_eq!(first.records, 2);
        assert_eq!(first.converted_records, 2);
        assert_eq!(
            fs::metadata(output.join("L1.sc4")).unwrap().len(),
            2 * FILE_BYTES as u64
        );

        let second = convert_for_raw_names(&options, &names).unwrap();
        assert!(second.already_complete);
        assert_eq!(second.records, 2);
        assert_eq!(second.resumed_records, 2);

        let layer_path = output.join("L1.sc4");
        let mut layer = OpenOptions::new().write(true).open(&layer_path).unwrap();
        use std::io::{Seek, SeekFrom};
        layer.seek(SeekFrom::End(-1)).unwrap();
        layer.write_all(&[0xff]).unwrap();
        layer.sync_all().unwrap();
        assert!(convert_for_raw_names(&options, &names).is_err());
    }

    #[test]
    fn preflight_rejects_partial_sources_before_creating_output() {
        let directory = TestDirectory::new();
        let source = directory.0.join("raw");
        let output = directory.0.join("scale4");
        fs::create_dir(&source).unwrap();
        write_raw(&source.join("L1-E0.bin"), 0);
        let options = ConvertOptions {
            source_root: source,
            output_root: output.clone(),
            workers: 1,
            resume: true,
        };
        let names = vec!["L1-E0.bin".into(), "L1-E1.bin".into()];
        assert!(convert_for_raw_names(&options, &names).is_err());
        assert!(!output.exists());
    }
}
