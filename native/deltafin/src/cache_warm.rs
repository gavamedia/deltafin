//! Native idle-time planner and fetcher for trace-observed K3 experts.
//!
//! Planning is read-only. Network and disk mutation occur only through the
//! explicit [`fetch_planned`] call, which reuses the authenticated native K3
//! inventory and the same transactional range fetcher as ordinary inference.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::dspark_checkpoint::strict_json;
use crate::error::{DeltafinError, Result};
use crate::experts::{
    K3_EXPERT_SOURCE_BYTES, K3_EXPERTS_PER_LAYER, K3_MOE_LAYER_FIRST, K3_MOE_LAYER_LAST,
};
use crate::inventory::K3Inventory;
use crate::weight_fetch::{ExpertFetchCatalog, FetchLimits, ProgressSink, WeightFetchProgress};
use crate::sys::fs::{self as sys_fs, Open};

const MAX_TRACE_BYTES: u64 = 8 << 30;
const MAX_TRACE_LINE_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ObservedExpert {
    pub layer: u32,
    pub expert: u16,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct WarmCandidate {
    pub frequency: u64,
    pub layer: u32,
    pub expert: u16,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct WarmPlan {
    pub trace_files: Vec<PathBuf>,
    pub observed_routes: u64,
    pub unique_observed_experts: usize,
    pub missing_observed_experts: usize,
    pub missing_observed_bytes: u64,
    pub candidates: Vec<WarmCandidate>,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct WarmFetchReport {
    pub selected_experts: usize,
    pub files_completed: usize,
    pub files_reused: usize,
    pub requests_completed: usize,
    pub bytes_transferred: u64,
}

/// Parse bounded route records and rank absent canonical experts by descending
/// observation count, then layer and expert ID for deterministic ties.
pub fn plan(cache: &Path, traces: &[PathBuf]) -> Result<WarmPlan> {
    inspect_optional_directory(cache, "expert cache")?;
    let mut counts = BTreeMap::<ObservedExpert, u64>::new();
    let mut observed_routes = 0_u64;
    for trace in traces {
        parse_trace(trace, &mut counts, &mut observed_routes)?;
    }
    let mut candidates = Vec::new();
    for (key, frequency) in &counts {
        if !exact_cached_expert(cache, *key)? {
            candidates.push(WarmCandidate {
                frequency: *frequency,
                layer: key.layer,
                expert: key.expert,
            });
        }
    }
    candidates.sort_unstable_by(|left, right| {
        right
            .frequency
            .cmp(&left.frequency)
            .then_with(|| left.layer.cmp(&right.layer))
            .then_with(|| left.expert.cmp(&right.expert))
    });
    let missing_observed_bytes = (candidates.len() as u64)
        .checked_mul(K3_EXPERT_SOURCE_BYTES as u64)
        .ok_or_else(|| DeltafinError::new("warm-cache missing byte count overflowed"))?;
    Ok(WarmPlan {
        trace_files: traces.to_vec(),
        observed_routes,
        unique_observed_experts: counts.len(),
        missing_observed_experts: candidates.len(),
        missing_observed_bytes,
        candidates,
    })
}

/// Explicitly fetch at most `limit` highest-ranked candidates. A zero limit is
/// a read-only no-op. The catalog authenticates all tensor offsets before the
/// first network request and publishes every raw expert without replacement.
pub fn fetch_planned(
    model_root: &Path,
    plan: &WarmPlan,
    limit: usize,
    limits: FetchLimits,
    progress: &dyn ProgressSink,
) -> Result<WarmFetchReport> {
    if limit == 0 || plan.candidates.is_empty() {
        return Ok(WarmFetchReport::default());
    }
    let inventory = K3Inventory::load_from_root(model_root)?;
    let catalog = ExpertFetchCatalog::open(model_root, &inventory, limits)?;
    let selected = &plan.candidates[..limit.min(plan.candidates.len())];
    let mut by_layer = BTreeMap::<u32, Vec<u16>>::new();
    for candidate in selected {
        by_layer
            .entry(candidate.layer)
            .or_default()
            .push(candidate.expert);
    }
    let mut report = WarmFetchReport {
        selected_experts: selected.len(),
        ..WarmFetchReport::default()
    };
    for (layer, experts) in by_layer {
        let completed = catalog.fetch_layer(layer, &experts, progress)?;
        absorb_progress(&mut report, completed)?;
    }
    Ok(report)
}

fn absorb_progress(report: &mut WarmFetchReport, progress: WeightFetchProgress) -> Result<()> {
    report.files_completed = report
        .files_completed
        .checked_add(progress.files_completed)
        .ok_or_else(|| DeltafinError::new("warm-cache completed-file count overflowed"))?;
    report.files_reused = report
        .files_reused
        .checked_add(progress.files_reused)
        .ok_or_else(|| DeltafinError::new("warm-cache reused-file count overflowed"))?;
    report.requests_completed = report
        .requests_completed
        .checked_add(progress.requests_completed)
        .ok_or_else(|| DeltafinError::new("warm-cache request count overflowed"))?;
    report.bytes_transferred = report
        .bytes_transferred
        .checked_add(progress.bytes_transferred)
        .ok_or_else(|| DeltafinError::new("warm-cache transferred byte count overflowed"))?;
    Ok(())
}

fn parse_trace(
    path: &Path,
    counts: &mut BTreeMap<ObservedExpert, u64>,
    observed_routes: &mut u64,
) -> Result<()> {
    let before = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect router trace", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > MAX_TRACE_BYTES {
        return Err(DeltafinError::new(format!(
            "router trace must be a regular non-symlink file no larger than {MAX_TRACE_BYTES} bytes: {}",
            path.display()
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open router trace without following links", path, error))?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat opened router trace", path, error))?;
    if !same_file(&before, &opened) {
        return Err(DeltafinError::new(format!(
            "router trace changed while opening: {}",
            path.display()
        )));
    }
    let mut reader = BufReader::with_capacity(256 << 10, file);
    let mut line = Vec::new();
    let mut line_number = 0_u64;
    while read_bounded_line(&mut reader, &mut line)? {
        line_number = line_number
            .checked_add(1)
            .ok_or_else(|| DeltafinError::new("router trace line count overflowed"))?;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let value = strict_json(&line, "router trace row").map_err(|error| {
            DeltafinError::new(format!("{}:{line_number}: {error}", path.display()))
        })?;
        count_trace_row(path, line_number, &value, counts, observed_routes)?;
    }
    let after = sys_fs::fstat(reader.get_ref())
        .map_err(|error| io_error("restat router trace", path, error))?;
    if !same_file(&opened, &after) {
        return Err(DeltafinError::new(format!(
            "router trace changed while reading: {}",
            path.display()
        )));
    }
    Ok(())
}

fn count_trace_row(
    path: &Path,
    line_number: u64,
    value: &Value,
    counts: &mut BTreeMap<ObservedExpert, u64>,
    observed_routes: &mut u64,
) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| trace_error(path, line_number))?;
    let layer = object
        .get("layer")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| (K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(value))
        .ok_or_else(|| trace_error(path, line_number))?;
    let ids = object
        .get("ids")
        .and_then(Value::as_array)
        .filter(|ids| !ids.is_empty() && ids.len() <= K3_EXPERTS_PER_LAYER)
        .ok_or_else(|| trace_error(path, line_number))?;
    for value in ids {
        let expert = value
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .filter(|value| (*value as usize) < K3_EXPERTS_PER_LAYER)
            .ok_or_else(|| trace_error(path, line_number))?;
        let entry = counts.entry(ObservedExpert { layer, expert }).or_default();
        *entry = entry
            .checked_add(1)
            .ok_or_else(|| DeltafinError::new("router frequency count overflowed"))?;
        *observed_routes = observed_routes
            .checked_add(1)
            .ok_or_else(|| DeltafinError::new("observed route count overflowed"))?;
    }
    Ok(())
}

fn trace_error(path: &Path, line_number: u64) -> DeltafinError {
    DeltafinError::new(format!(
        "{}:{line_number}: invalid router trace row",
        path.display()
    ))
}

fn read_bounded_line(reader: &mut impl BufRead, output: &mut Vec<u8>) -> Result<bool> {
    output.clear();
    loop {
        let available = reader
            .fill_buf()
            .map_err(|error| DeltafinError::new(format!("read router trace: {error}")))?;
        if available.is_empty() {
            return Ok(!output.is_empty());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |position| position + 1);
        if output.len().saturating_add(consumed) > MAX_TRACE_LINE_BYTES {
            return Err(DeltafinError::new(format!(
                "router trace line exceeds {MAX_TRACE_LINE_BYTES} bytes"
            )));
        }
        output.extend_from_slice(&available[..consumed]);
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(true);
        }
    }
}

fn exact_cached_expert(cache: &Path, key: ObservedExpert) -> Result<bool> {
    let path = cache.join(format!("L{}-E{}.bin", key.layer, key.expert));
    match sys_fs::lstat(&path) {
        Ok(metadata)
            if metadata.is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() == K3_EXPERT_SOURCE_BYTES as u64 =>
        {
            let file = Open::new()
                .read(true)
                .no_follow()
                .open(&path)
                .map_err(|error| io_error("open cached expert", &path, error))?;
            let opened = sys_fs::fstat(&file)
                .map_err(|error| io_error("stat cached expert", &path, error))?;
            if !same_file(&metadata, &opened) {
                return Err(DeltafinError::new(format!(
                    "cached expert changed while opening: {}",
                    path.display()
                )));
            }
            Ok(true)
        }
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => Err(
            DeltafinError::new(format!("unsafe cached expert: {}", path.display())),
        ),
        Ok(_) => Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io_error("inspect cached expert", &path, error)),
    }
}

fn inspect_optional_directory(path: &Path, label: &str) -> Result<()> {
    match sys_fs::lstat(path) {
        Ok(metadata) if !metadata.file_type().is_symlink() && metadata.is_dir() => Ok(()),
        Ok(_) => Err(DeltafinError::new(format!(
            "{label} is not a real directory: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error(&format!("inspect {label}"), path, error)),
    }
}

fn same_file(left: &sys_fs::Stat, right: &sys_fs::Stat) -> bool {
    left.is_file()
        && right.is_file()
        && left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "deltafin-cache-warm-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
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

    #[test]
    fn ranks_missing_experts_without_network_or_full_file_allocation() {
        let root = TestDirectory::new();
        let cache = root.0.join("cache");
        fs::create_dir(&cache).unwrap();
        let trace = root.0.join("trace.jsonl");
        fs::write(
            &trace,
            b"{\"layer\":1,\"ids\":[7,2]}\n{\"layer\":1,\"ids\":[7]}\n{\"layer\":2,\"ids\":[3]}\n",
        )
        .unwrap();
        File::create(cache.join("L2-E3.bin"))
            .unwrap()
            .set_len(K3_EXPERT_SOURCE_BYTES as u64)
            .unwrap();

        let plan = plan(&cache, std::slice::from_ref(&trace)).unwrap();
        assert_eq!(plan.observed_routes, 4);
        assert_eq!(plan.unique_observed_experts, 3);
        assert_eq!(plan.missing_observed_experts, 2);
        assert_eq!(
            plan.candidates,
            vec![
                WarmCandidate {
                    frequency: 2,
                    layer: 1,
                    expert: 7,
                },
                WarmCandidate {
                    frequency: 1,
                    layer: 1,
                    expert: 2,
                },
            ]
        );
    }

    #[test]
    fn malformed_oversized_and_unsafe_inputs_fail_closed() {
        let root = TestDirectory::new();
        let cache = root.0.join("cache");
        fs::create_dir(&cache).unwrap();
        let invalid = root.0.join("invalid.jsonl");
        fs::write(&invalid, b"{\"layer\":0,\"ids\":[1]}\n").unwrap();
        assert!(plan(&cache, &[invalid]).is_err());

        let long = root.0.join("long.jsonl");
        let mut file = File::create(&long).unwrap();
        file.write_all(&vec![b'x'; MAX_TRACE_LINE_BYTES + 1])
            .unwrap();
        drop(file);
        assert!(plan(&cache, &[long]).is_err());

        #[cfg(unix)]
        {
            crate::sys::fs::symlink("missing", cache.join("L1-E1.bin")).unwrap();
            let trace = root.0.join("unsafe.jsonl");
            fs::write(&trace, b"{\"layer\":1,\"ids\":[1]}\n").unwrap();
            assert!(plan(&cache, &[trace]).is_err());
        }
    }

    #[test]
    fn zero_fetch_limit_is_a_strict_read_only_noop() {
        let plan = WarmPlan {
            trace_files: Vec::new(),
            observed_routes: 1,
            unique_observed_experts: 1,
            missing_observed_experts: 1,
            missing_observed_bytes: K3_EXPERT_SOURCE_BYTES as u64,
            candidates: vec![WarmCandidate {
                frequency: 1,
                layer: 1,
                expert: 0,
            }],
        };
        let report = fetch_planned(
            Path::new("/does/not/exist"),
            &plan,
            0,
            FetchLimits::default(),
            &|_| {},
        )
        .unwrap();
        assert_eq!(report, WarmFetchReport::default());
    }
}
