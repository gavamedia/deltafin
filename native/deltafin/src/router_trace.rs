//! Native, bounded router-trace production for expert-cache planning.
//!
//! Tracing is disabled by default and the disabled path owns no file or
//! buffer. Buffered mode publishes one complete 93-layer pass at a time;
//! dropping an uncommitted pass discards it. Sync mode deliberately flushes
//! each row for crash-oriented debugging and may therefore retain a partial
//! pass, matching its explicit durability tradeoff.

use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::{DeltafinError, Result};
use crate::experts::{K3_EXPERTS_PER_LAYER, K3_MOE_LAYER_FIRST, K3_MOE_LAYER_LAST};
use crate::provider::TargetSequenceMailbox;
use crate::sys::fs::{self as sys_fs, Open};

pub const MAX_ROUTER_TRACE_BYTES: u64 = 8 << 30;
pub const ROUTER_TRACE_HOST_RESERVE_BYTES: u64 = 5 * 1024 * 1024;
const MAX_BUFFERED_PASS_BYTES: usize = 4 * 1024 * 1024;
const ROUTE_TOP_K: usize = 16;
// The native warm-cache parser intentionally bounds one record to the expert
// universe. Split wide prefills before this point rather than producing a
// trace that the native consumer must reject.
const MAX_ROUTE_ROWS_PER_RECORD: usize = K3_EXPERTS_PER_LAYER / ROUTE_TOP_K;

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum RouterTraceMode {
    #[default]
    Off,
    Buffered,
    Sync,
}

impl RouterTraceMode {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("off").trim().to_ascii_lowercase().as_str() {
            "" | "off" | "0" | "false" | "no" => Ok(Self::Off),
            "buffered" | "1" | "true" | "yes" | "on" => Ok(Self::Buffered),
            "sync" => Ok(Self::Sync),
            _ => Err(DeltafinError::new(
                "router trace mode must be off, buffered, or sync",
            )),
        }
    }
}

pub struct RouterTrace {
    state: RouterTraceState,
}

enum RouterTraceState {
    Off,
    Active(TraceFile),
}

struct TraceFile {
    mode: RouterTraceMode,
    path: PathBuf,
    file: File,
    bytes_written: u64,
    next_step: u64,
    pending: Vec<u8>,
    scratch: Vec<u8>,
}

#[derive(Serialize)]
struct TraceRecord<'a> {
    step: u64,
    layer: u32,
    ids: &'a [u16],
    w: &'a [f32],
}

impl RouterTrace {
    pub fn open(mode: RouterTraceMode, path: Option<&Path>) -> Result<Self> {
        if mode == RouterTraceMode::Off {
            if path.is_some() {
                return Err(DeltafinError::new(
                    "a router-trace path cannot be combined with trace mode off",
                ));
            }
            return Ok(Self {
                state: RouterTraceState::Off,
            });
        }
        let path = path.ok_or_else(|| {
            DeltafinError::new("buffered/sync router tracing requires an explicit path")
        })?;
        if path.as_os_str().is_empty() {
            return Err(DeltafinError::new("router-trace path must not be empty"));
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent_meta = sys_fs::lstat(parent)
            .map_err(|error| io_error("inspect router-trace parent", parent, error))?;
        if parent_meta.file_type().is_symlink() || !parent_meta.is_dir() {
            return Err(DeltafinError::new(format!(
                "router-trace parent must be a real directory: {}",
                parent.display()
            )));
        }
        let before = match sys_fs::lstat(path) {
            Ok(metadata) => {
                validate_existing(path, &metadata)?;
                Some(metadata)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(io_error("inspect router trace", path, error)),
        };
        let file = Open::new()
            .append(true)
            .create(true)
            .mode(0o600)
            .no_follow()
            .open(path)
            .map_err(|error| io_error("open router trace without following links", path, error))?;
        let opened = sys_fs::fstat(&file)
            .map_err(|error| io_error("stat opened router trace", path, error))?;
        if !opened.is_file() || opened.len() > MAX_ROUTER_TRACE_BYTES {
            return Err(DeltafinError::new(format!(
                "router trace must be a regular file no larger than {MAX_ROUTER_TRACE_BYTES} bytes: {}",
                path.display()
            )));
        }
        if before
            .as_ref()
            .is_some_and(|before| !same_file(before, &opened))
        {
            return Err(DeltafinError::new(format!(
                "router trace changed while opening: {}",
                path.display()
            )));
        }
        Ok(Self {
            state: RouterTraceState::Active(TraceFile {
                mode,
                path: path.to_path_buf(),
                file,
                bytes_written: opened.len(),
                next_step: 0,
                pending: Vec::with_capacity(256 << 10),
                scratch: Vec::with_capacity(16 << 10),
            }),
        })
    }

    pub const fn enabled(&self) -> bool {
        matches!(self.state, RouterTraceState::Active(_))
    }

    pub fn begin_pass(&mut self) -> RouterTracePass<'_> {
        if let RouterTraceState::Active(trace) = &mut self.state {
            trace.pending.clear();
        }
        RouterTracePass {
            trace: self,
            committed: false,
        }
    }
}

pub struct RouterTracePass<'a> {
    trace: &'a mut RouterTrace,
    committed: bool,
}

impl RouterTracePass<'_> {
    /// Copy the already-materialized fixed mailbox into the optional trace.
    /// The disabled case returns before allocating or traversing route rows.
    pub fn record_mailbox(&mut self, mailbox: &TargetSequenceMailbox) -> Result<()> {
        if !matches!(&self.trace.state, RouterTraceState::Active(_)) {
            return Ok(());
        }
        let edges = mailbox
            .position_count()
            .checked_mul(ROUTE_TOP_K)
            .ok_or_else(|| DeltafinError::new("router-trace mailbox edge count overflowed"))?;
        let mut ids = Vec::with_capacity(edges);
        let mut weights = Vec::with_capacity(edges);
        for route in mailbox.routes() {
            if route.layer_index() != mailbox.layer_index()
                || route.spine_generation() != mailbox.spine_generation()
            {
                return Err(DeltafinError::new(
                    "router trace received a stale route mailbox",
                ));
            }
            ids.extend_from_slice(route.ordered_experts());
            weights.extend_from_slice(route.ordered_weight_bits());
        }
        self.record_routes(mailbox.layer_index(), &ids, &weights)
    }

    /// Record complete authoritative route rows. IDs and fp32 weight bits are
    /// never reordered or used as model input; this is a diagnostic copy of
    /// the mailbox K3 already published for expert execution.
    pub fn record_routes(
        &mut self,
        layer: u32,
        ordered_experts: &[u16],
        ordered_weight_bits: &[u32],
    ) -> Result<()> {
        let RouterTraceState::Active(trace) = &mut self.trace.state else {
            return Ok(());
        };
        if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer)
            || ordered_experts.is_empty()
            || ordered_experts.len() != ordered_weight_bits.len()
            || ordered_experts.len() % ROUTE_TOP_K != 0
        {
            return Err(DeltafinError::new(
                "router trace received an invalid authoritative route shape",
            ));
        }
        let rows = ordered_experts.len() / ROUTE_TOP_K;
        for first_row in (0..rows).step_by(MAX_ROUTE_ROWS_PER_RECORD) {
            let row_count = (rows - first_row).min(MAX_ROUTE_ROWS_PER_RECORD);
            let first = first_row * ROUTE_TOP_K;
            let end = first + row_count * ROUTE_TOP_K;
            let ids = &ordered_experts[first..end];
            if ids
                .iter()
                .any(|id| usize::from(*id) >= K3_EXPERTS_PER_LAYER)
            {
                return Err(DeltafinError::new(
                    "router trace received an expert outside K3's universe",
                ));
            }
            let mut weights = Vec::with_capacity(ids.len());
            for bits in &ordered_weight_bits[first..end] {
                let weight = f32::from_bits(*bits);
                if !weight.is_finite() || weight < 0.0 {
                    return Err(DeltafinError::new(
                        "router trace received an invalid authoritative weight",
                    ));
                }
                weights.push((weight * 100_000.0).round() / 100_000.0);
            }
            trace.append_record(layer, ids, &weights)?;
        }
        Ok(())
    }

    pub fn commit(mut self) -> Result<()> {
        let result = match &mut self.trace.state {
            RouterTraceState::Off => Ok(()),
            RouterTraceState::Active(trace) => trace.commit_pass(),
        };
        if result.is_ok() {
            self.committed = true;
        }
        result
    }
}

impl Drop for RouterTracePass<'_> {
    fn drop(&mut self) {
        if !self.committed
            && let RouterTraceState::Active(trace) = &mut self.trace.state
        {
            trace.pending.clear();
        }
    }
}

impl TraceFile {
    fn append_record(&mut self, layer: u32, ids: &[u16], weights: &[f32]) -> Result<()> {
        self.scratch.clear();
        serde_json::to_writer(
            &mut self.scratch,
            &TraceRecord {
                step: self.next_step,
                layer,
                ids,
                w: weights,
            },
        )
        .map_err(|error| DeltafinError::new(format!("serialize router trace: {error}")))?;
        self.scratch.push(b'\n');
        match self.mode {
            RouterTraceMode::Off => unreachable!("active router trace cannot have off mode"),
            RouterTraceMode::Buffered => {
                if self.pending.len().saturating_add(self.scratch.len()) > MAX_BUFFERED_PASS_BYTES {
                    return Err(DeltafinError::new(format!(
                        "one buffered router pass exceeds {MAX_BUFFERED_PASS_BYTES} bytes"
                    )));
                }
                let prospective = self
                    .bytes_written
                    .checked_add(self.pending.len() as u64)
                    .and_then(|bytes| bytes.checked_add(self.scratch.len() as u64))
                    .ok_or_else(|| DeltafinError::new("router-trace size overflowed u64"))?;
                ensure_below_limit(prospective)?;
                self.pending.extend_from_slice(&self.scratch);
            }
            RouterTraceMode::Sync => {
                let prospective = self
                    .bytes_written
                    .checked_add(self.scratch.len() as u64)
                    .ok_or_else(|| DeltafinError::new("router-trace size overflowed u64"))?;
                ensure_below_limit(prospective)?;
                self.file
                    .write_all(&self.scratch)
                    .map_err(|error| io_error("write router trace", &self.path, error))?;
                self.file
                    .flush()
                    .map_err(|error| io_error("flush router trace", &self.path, error))?;
                self.bytes_written = prospective;
            }
        }
        Ok(())
    }

    fn commit_pass(&mut self) -> Result<()> {
        if self.mode == RouterTraceMode::Buffered && !self.pending.is_empty() {
            let prospective = self
                .bytes_written
                .checked_add(self.pending.len() as u64)
                .ok_or_else(|| DeltafinError::new("router-trace size overflowed u64"))?;
            ensure_below_limit(prospective)?;
            self.file
                .write_all(&self.pending)
                .map_err(|error| io_error("write router trace", &self.path, error))?;
            self.file
                .flush()
                .map_err(|error| io_error("flush router trace", &self.path, error))?;
            self.bytes_written = prospective;
            self.pending.clear();
        }
        self.next_step = self
            .next_step
            .checked_add(1)
            .ok_or_else(|| DeltafinError::new("router-trace step overflowed u64"))?;
        Ok(())
    }
}

fn ensure_below_limit(bytes: u64) -> Result<()> {
    if bytes >= MAX_ROUTER_TRACE_BYTES {
        return Err(DeltafinError::new(format!(
            "router trace would reach or exceed its {MAX_ROUTER_TRACE_BYTES}-byte hard limit"
        )));
    }
    Ok(())
}

fn validate_existing(path: &Path, metadata: &sys_fs::Stat) -> Result<()> {
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_ROUTER_TRACE_BYTES
    {
        return Err(DeltafinError::new(format!(
            "router trace must be a regular non-symlink file no larger than {MAX_ROUTER_TRACE_BYTES} bytes: {}",
            path.display()
        )));
    }
    Ok(())
}

fn same_file(left: &sys_fs::Stat, right: &sys_fs::Stat) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use serde_json::Value;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    struct TempDir(PathBuf);

    impl TempDir {
        fn create() -> Self {
            let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-router-trace-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn routes(rows: usize) -> (Vec<u16>, Vec<u32>) {
        let ids = (0..rows * ROUTE_TOP_K)
            .map(|edge| (edge % K3_EXPERTS_PER_LAYER) as u16)
            .collect::<Vec<_>>();
        let weights = vec![0.125_f32.to_bits(); ids.len()];
        (ids, weights)
    }

    #[test]
    fn disabled_path_owns_no_file_and_rejects_a_misleading_path() {
        let mut trace = RouterTrace::open(RouterTraceMode::Off, None).unwrap();
        assert!(!trace.enabled());
        let (ids, weights) = routes(1);
        let mut pass = trace.begin_pass();
        pass.record_routes(1, &ids, &weights).unwrap();
        pass.commit().unwrap();
        assert!(RouterTrace::open(RouterTraceMode::Off, Some(Path::new("x"))).is_err());
    }

    #[test]
    fn buffered_pass_is_atomic_and_wide_mailboxes_remain_parseable() {
        let root = TempDir::create();
        let path = root.0.join("router.jsonl");
        let mut trace = RouterTrace::open(RouterTraceMode::Buffered, Some(&path)).unwrap();
        let (ids, weights) = routes(64);
        {
            let mut abandoned = trace.begin_pass();
            abandoned.record_routes(1, &ids, &weights).unwrap();
        }
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
        let mut pass = trace.begin_pass();
        pass.record_routes(1, &ids, &weights).unwrap();
        pass.commit().unwrap();

        let text = fs::read_to_string(&path).unwrap();
        let rows = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["ids"].as_array().unwrap().len(), 896);
        assert_eq!(rows[1]["ids"].as_array().unwrap().len(), 128);
        assert_eq!(rows[0]["step"], 0);

        let cache = root.0.join("cache");
        fs::create_dir(&cache).unwrap();
        let plan = crate::cache_warm::plan(&cache, std::slice::from_ref(&path)).unwrap();
        assert_eq!(plan.observed_routes, 1_024);
        assert_eq!(plan.unique_observed_experts, K3_EXPERTS_PER_LAYER);
    }

    #[test]
    fn synchronous_mode_flushes_each_record_before_pass_commit() {
        let root = TempDir::create();
        let path = root.0.join("router.jsonl");
        let mut trace = RouterTrace::open(RouterTraceMode::Sync, Some(&path)).unwrap();
        let (ids, weights) = routes(1);
        let mut pass = trace.begin_pass();
        pass.record_routes(92, &ids, &weights).unwrap();
        assert!(fs::metadata(&path).unwrap().len() > 0);
        pass.commit().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn trace_open_never_follows_a_symlink() {
        use crate::sys::fs::symlink;

        let root = TempDir::create();
        let target = root.0.join("target");
        let link = root.0.join("trace");
        fs::write(&target, b"safe").unwrap();
        symlink(&target, &link).unwrap();
        assert!(RouterTrace::open(RouterTraceMode::Buffered, Some(&link)).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"safe");
    }
}
