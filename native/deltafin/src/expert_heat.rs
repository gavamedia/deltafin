//! Persistent cross-run expert-heat histogram and the opt-in RAM pin tier.
//!
//! Every committed pass already publishes its authoritative routes through the
//! target mailbox. This module counts that free signal — presence of each
//! (layer, expert) per mailbox, never per row — into a small decayed histogram
//! persisted at `k3-meta/expert_heat.v1.bin`, and can freeze a bounded
//! candidate set of hot experts whose spans are promoted into a permanent
//! RAM tier the first time generation naturally reads them.
//!
//! Learned storage placement, never learned inference: nothing here selects
//! experts, supplies weights, or reorders reductions. A tier hit changes only
//! where authoritative bytes are copied from.
//!
//! Unlike the router trace (explicitly requested, fail-closed on open), the
//! histogram is default-on and advisory, so this module deliberately fails
//! open: a missing, truncated, corrupt, or symlinked heat file means "nothing
//! learned yet", flush errors are swallowed after one warning, and the
//! observation path is infallible. Only per-run counting honesty is kept
//! strict — presence-per-mailbox blunts the prefill-row and re-run inflation
//! that overstated the naive frequency study (BRAINSTORM-SPEED.md: 51.9%
//! in-sample versus 41.1% held-out at a 40 GB pin tier).

use std::alloc::Layout;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use sha2::{Digest, Sha256};

use crate::experts::{K3_MOE_LAYER_FIRST, K3_MOE_LAYER_LAST};
use crate::provider::ROUTE_TOP_K;
use crate::routing::{K3_EXPERTS, ROUTED_EXPERTS};
use crate::storage::BUFFER_ALIGNMENT;
use crate::trusted_download::{fsync_directory, secure_create_new};
use crate::sys::fs::{self as sys_fs, Open};

pub(crate) const EXPERT_HEAT_FILE_NAME: &str = "expert_heat.v1.bin";
const EXPERT_HEAT_LOCK_NAME: &str = "expert_heat.v1.lock";
const EXPERT_HEAT_TEMP_NAME: &str = "expert_heat.v1.bin.tmp";

/// Flush the accumulated deltas every this many committed passes. A SIGKILL
/// loses at most one window; Ctrl-C unwinds through the engine's Drop flush.
pub(crate) const FLUSH_EVERY_PASSES: u64 = 64;
/// Host bytes charged for the recorder's two counter arrays (delta atomics
/// plus the loaded snapshot) before residency selection.
pub(crate) const EXPERT_HEAT_HOST_RESERVE_BYTES: u64 = 1 << 20;
/// Exponential half-life of stored heat, in committed passes. Old workloads
/// age out instead of pinning yesterday's experts forever.
const HALF_LIFE_PASSES: f64 = 8192.0;
/// No pinning until the histogram has seen this many committed passes. The
/// held-out study showed thin samples flatter frequency ranking badly.
const MIN_WEIGHT_PASSES: f64 = 256.0;
/// A candidate must be routed at least this multiple of the uniform-random
/// expectation (16 of 896 per layer per pass). Filters the non-generalizing
/// tail that widened the in-sample/held-out gap.
const FLOOR_MULT: f64 = 2.0;

const MOE_LAYERS: usize = (K3_MOE_LAYER_LAST - K3_MOE_LAYER_FIRST + 1) as usize;
const HEAT_SLOTS: usize = MOE_LAYERS * K3_EXPERTS;
const HEAT_BITSET_WORDS: usize = K3_EXPERTS / 64;

const HEAT_MAGIC: [u8; 8] = *b"K3HEATv1";
const HEAT_VERSION: u32 = 1;
const HEAT_HEADER_BYTES: usize = 48;
const HEAT_PAYLOAD_BYTES: usize = HEAT_SLOTS * 4;
const HEAT_DIGEST_BYTES: usize = 32;
const HEAT_FILE_BYTES: usize = HEAT_HEADER_BYTES + HEAT_PAYLOAD_BYTES + HEAT_DIGEST_BYTES;

/// One immutable decoded histogram: per-slot heat plus the decayed pass mass
/// it was accumulated over. Slots are layer-major: `(layer - 1) * 896 + expert`.
pub(crate) struct ExpertHeatSnapshot {
    pub(crate) heats: Box<[f32]>,
    pub(crate) total_weight: f64,
}

impl ExpertHeatSnapshot {
    fn empty() -> Self {
        Self {
            heats: vec![0.0; HEAT_SLOTS].into_boxed_slice(),
            total_weight: 0.0,
        }
    }
}

/// Cross-run route-frequency recorder. Observation is lock-free and
/// infallible; persistence is best-effort behind a sidecar flock.
pub(crate) struct ExpertHeat {
    data_path: PathBuf,
    lock_path: PathBuf,
    temp_path: PathBuf,
    recording: bool,
    deltas: Box<[AtomicU32]>,
    pass_delta: AtomicU64,
    snapshot: ExpertHeatSnapshot,
    warned: AtomicBool,
}

impl ExpertHeat {
    /// Open the histogram under `<model_root>/k3-meta/`. Never fails: an
    /// unreadable or invalid file is reported once and treated as empty.
    pub(crate) fn open(model_root: &Path, recording: bool) -> Self {
        let meta = model_root.join("k3-meta");
        let data_path = meta.join(EXPERT_HEAT_FILE_NAME);
        let snapshot = match load_snapshot(&data_path) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                eprintln!(
                    "deltafin: ignoring expert-heat history {}: {error}",
                    data_path.display()
                );
                ExpertHeatSnapshot::empty()
            }
        };
        let mut deltas = Vec::with_capacity(HEAT_SLOTS);
        deltas.resize_with(HEAT_SLOTS, || AtomicU32::new(0));
        Self {
            lock_path: meta.join(EXPERT_HEAT_LOCK_NAME),
            temp_path: meta.join(EXPERT_HEAT_TEMP_NAME),
            data_path,
            recording,
            deltas: deltas.into_boxed_slice(),
            pass_delta: AtomicU64::new(0),
            snapshot,
            warned: AtomicBool::new(false),
        }
    }

    pub(crate) const fn recording(&self) -> bool {
        self.recording
    }

    /// The histogram loaded at startup, before this run contributed anything.
    /// Candidate selection uses exactly this view.
    pub(crate) const fn snapshot(&self) -> &ExpertHeatSnapshot {
        &self.snapshot
    }

    /// Approximate live pass mass for status reporting.
    pub(crate) fn live_weight(&self) -> f64 {
        self.snapshot.total_weight + self.pass_delta.load(Ordering::Relaxed) as f64
    }

    /// Record one layer mailbox: each expert present in any row counts once,
    /// regardless of how many rows routed to it. Out-of-range indices are
    /// skipped; this path must never fail or allocate per call.
    pub(crate) fn observe_layer_routes<'a, I>(&self, layer_index: u32, rows: I)
    where
        I: IntoIterator<Item = &'a [u16; ROUTE_TOP_K]>,
    {
        if !self.recording {
            return;
        }
        if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer_index) {
            return;
        }
        let mut present = [0_u64; HEAT_BITSET_WORDS];
        let mut any = false;
        for row in rows {
            for &expert in row {
                let expert = expert as usize;
                if expert < K3_EXPERTS {
                    present[expert / 64] |= 1_u64 << (expert % 64);
                    any = true;
                }
            }
        }
        if !any {
            return;
        }
        let base = (layer_index - K3_MOE_LAYER_FIRST) as usize * K3_EXPERTS;
        for (word_index, mut word) in present.into_iter().enumerate() {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                self.deltas[base + word_index * 64 + bit].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Count one committed pass. Returns true when a flush is due.
    pub(crate) fn note_pass_committed(&self) -> bool {
        if !self.recording {
            return false;
        }
        self.pass_delta.fetch_add(1, Ordering::Relaxed) + 1 >= FLUSH_EVERY_PASSES
    }

    /// Merge this run's deltas into the shared file. Best-effort: lock
    /// contention or any I/O failure restores the harvested deltas so a later
    /// flush retries, and at most one warning is printed per process.
    pub(crate) fn flush_best_effort(&self) {
        if !self.recording {
            return;
        }
        let harvested_weight = self.pass_delta.swap(0, Ordering::Relaxed);
        let mut harvested = Vec::new();
        for (slot, delta) in self.deltas.iter().enumerate() {
            let value = delta.swap(0, Ordering::Relaxed);
            if value != 0 {
                harvested.push((slot, value));
            }
        }
        if harvested_weight == 0 && harvested.is_empty() {
            return;
        }
        if let Err(error) = self.merge_locked(harvested_weight, &harvested) {
            self.pass_delta.fetch_add(harvested_weight, Ordering::Relaxed);
            for &(slot, value) in &harvested {
                self.deltas[slot].fetch_add(value, Ordering::Relaxed);
            }
            if !self.warned.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "deltafin: expert-heat flush deferred for {}: {error}",
                    self.data_path.display()
                );
            }
        }
    }

    fn merge_locked(&self, weight: u64, deltas: &[(usize, u32)]) -> crate::error::Result<()> {
        let _lock = HeatFileLock::acquire(&self.lock_path)?;
        // Re-read under the lock: another process may have merged since this
        // one loaded its startup snapshot (or since its previous flush).
        let mut current = load_snapshot(&self.data_path).unwrap_or_else(|_| {
            // A file that appeared but cannot be trusted is rebuilt from this
            // run's deltas rather than propagated.
            ExpertHeatSnapshot::empty()
        });
        let factor = 0.5_f64.powf(weight as f64 / HALF_LIFE_PASSES);
        for heat in current.heats.iter_mut() {
            *heat = (f64::from(*heat) * factor) as f32;
        }
        current.total_weight = current.total_weight * factor + weight as f64;
        for &(slot, value) in deltas {
            current.heats[slot] += value as f32;
        }
        fs::remove_file(&self.temp_path)
            .or_else(ignore_not_found)
            .map_err(|error| {
                crate::error::DeltafinError::new(format!(
                    "clear stale expert-heat temp file: {error}"
                ))
            })?;
        let file = secure_create_new(&self.temp_path, 0o600)?;
        let result = write_snapshot(&file, &current).and_then(|()| {
            fs::rename(&self.temp_path, &self.data_path).map_err(|error| {
                crate::error::DeltafinError::new(format!(
                    "publish expert-heat file {}: {error}",
                    self.data_path.display()
                ))
            })
        });
        if result.is_err() {
            let _ = fs::remove_file(&self.temp_path);
            return result;
        }
        fsync_directory(
            self.data_path
                .parent()
                .ok_or_else(|| crate::error::DeltafinError::new("expert-heat file has no parent"))?,
        )
    }
}

fn ignore_not_found(error: std::io::Error) -> std::io::Result<()> {
    if error.kind() == std::io::ErrorKind::NotFound {
        Ok(())
    } else {
        Err(error)
    }
}

fn write_snapshot(mut file: &File, snapshot: &ExpertHeatSnapshot) -> crate::error::Result<()> {
    use std::io::Write as _;
    let bytes = encode_snapshot(snapshot);
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| {
            crate::error::DeltafinError::new(format!("write expert-heat file: {error}"))
        })
}

fn encode_snapshot(snapshot: &ExpertHeatSnapshot) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEAT_FILE_BYTES);
    bytes.extend_from_slice(&HEAT_MAGIC);
    bytes.extend_from_slice(&HEAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&K3_MOE_LAYER_FIRST.to_le_bytes());
    bytes.extend_from_slice(&K3_MOE_LAYER_LAST.to_le_bytes());
    bytes.extend_from_slice(&(K3_EXPERTS as u32).to_le_bytes());
    bytes.extend_from_slice(&snapshot.total_weight.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    debug_assert_eq!(bytes.len(), HEAT_HEADER_BYTES);
    for heat in &snapshot.heats {
        bytes.extend_from_slice(&heat.to_le_bytes());
    }
    let digest = Sha256::digest(&bytes);
    bytes.extend_from_slice(&digest);
    debug_assert_eq!(bytes.len(), HEAT_FILE_BYTES);
    bytes
}

/// Decode and fully validate one heat file. Any deviation — length, magic,
/// version, dimensions, checksum, or non-finite/negative values — is an error
/// the caller treats as "no history".
fn load_snapshot(path: &Path) -> crate::error::Result<ExpertHeatSnapshot> {
    let mut file = match Open::new()
        .read(true)
        .no_follow()
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ExpertHeatSnapshot::empty());
        }
        Err(error) => {
            return Err(crate::error::DeltafinError::new(format!(
                "open expert-heat file: {error}"
            )));
        }
    };
    let metadata = sys_fs::fstat(&file)
        .map_err(|error| crate::error::DeltafinError::new(format!("stat expert-heat: {error}")))?;
    if !metadata.is_file() || metadata.len() != HEAT_FILE_BYTES as u64 {
        return Err(crate::error::DeltafinError::new(
            "expert-heat file is not a regular file of the expected length",
        ));
    }
    let mut bytes = vec![0_u8; HEAT_FILE_BYTES];
    file.read_exact(&mut bytes)
        .map_err(|error| crate::error::DeltafinError::new(format!("read expert-heat: {error}")))?;
    decode_snapshot(&bytes)
}

fn decode_snapshot(bytes: &[u8]) -> crate::error::Result<ExpertHeatSnapshot> {
    let invalid = |what: &str| crate::error::DeltafinError::new(format!("expert-heat {what}"));
    if bytes.len() != HEAT_FILE_BYTES {
        return Err(invalid("length is wrong"));
    }
    let (body, digest) = bytes.split_at(HEAT_FILE_BYTES - HEAT_DIGEST_BYTES);
    if Sha256::digest(body).as_slice() != digest {
        return Err(invalid("checksum does not match"));
    }
    if body[..8] != HEAT_MAGIC {
        return Err(invalid("magic is unknown"));
    }
    let u32_at = |offset: usize| u32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
    if u32_at(8) != HEAT_VERSION
        || u32_at(12) != K3_MOE_LAYER_FIRST
        || u32_at(16) != K3_MOE_LAYER_LAST
        || u32_at(20) != K3_EXPERTS as u32
    {
        return Err(invalid("dimensions do not match this model"));
    }
    let total_weight = f64::from_le_bytes(body[24..32].try_into().unwrap());
    if !total_weight.is_finite() || total_weight < 0.0 {
        return Err(invalid("weight is not a finite non-negative value"));
    }
    let mut heats = Vec::with_capacity(HEAT_SLOTS);
    for chunk in body[HEAT_HEADER_BYTES..].chunks_exact(4) {
        let heat = f32::from_le_bytes(chunk.try_into().unwrap());
        if !heat.is_finite() || heat < 0.0 {
            return Err(invalid("histogram contains a non-finite or negative value"));
        }
        heats.push(heat);
    }
    Ok(ExpertHeatSnapshot {
        heats: heats.into_boxed_slice(),
        total_weight,
    })
}

struct HeatFileLock {
    file: File,
}

impl HeatFileLock {
    fn acquire(path: &Path) -> crate::error::Result<Self> {
        let file = Open::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .no_follow()
            .open(path)
            .map_err(|error| {
                crate::error::DeltafinError::new(format!("open expert-heat lock: {error}"))
            })?;
        if !sys_fs::fstat(&file)
            .map_err(|error| {
                crate::error::DeltafinError::new(format!("stat expert-heat lock: {error}"))
            })?
            .is_file()
        {
            return Err(crate::error::DeltafinError::new(
                "expert-heat lock is not regular",
            ));
        }
        if !sys_fs::try_lock_exclusive(&file)
            .map_err(|error| crate::error::DeltafinError::new(format!("acquire expert-heat lock: {error}")))?
        {
            return Err(crate::error::DeltafinError::new(
                "another process is flushing expert heat",
            ));
        }
        Ok(Self { file })
    }
}

impl Drop for HeatFileLock {
    fn drop(&mut self) {
        sys_fs::unlock(&self.file);
    }
}

/// One expert span in a heap allocation aligned to the reader-arena boundary.
/// Alignment matters on Metal: the no-copy `MTLBuffer` wrapper cache only
/// admits page-aligned, page-multiple spans; anything else silently degrades
/// to a per-use scratch copy of the whole span.
pub(crate) struct PageAlignedSpan {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl PageAlignedSpan {
    pub(crate) fn copy_from(bytes: &[u8]) -> Option<Arc<Self>> {
        let layout = Layout::from_size_align(bytes.len(), BUFFER_ALIGNMENT).ok()?;
        if layout.size() == 0 {
            return None;
        }
        // SAFETY: the layout has a non-zero size.
        let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) })?;
        // SAFETY: the fresh allocation is `bytes.len()` writable bytes and
        // cannot overlap the borrowed source.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len()) };
        Some(Arc::new(Self { ptr, layout }))
    }
}

impl std::ops::Deref for PageAlignedSpan {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the allocation is live for `self`'s lifetime and exactly
        // `layout.size()` bytes were initialized at construction.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for PageAlignedSpan {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated with exactly this layout and is owned.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

// SAFETY: the span is an exclusively owned, never-mutated heap block.
unsafe impl Send for PageAlignedSpan {}
// SAFETY: shared access is read-only.
unsafe impl Sync for PageAlignedSpan {}

/// Every (layer, expert) slot that clears the confidence ramp and frequency
/// floor, ranked hottest-first with deterministic ties. Empty when history is
/// too thin to trust. Shared by explicit-budget planning and auto sizing so
/// the two can never disagree about what qualifies.
fn qualifying_candidates(snapshot: &ExpertHeatSnapshot) -> Vec<(f32, u32)> {
    if snapshot.heats.len() != HEAT_SLOTS || snapshot.total_weight < MIN_WEIGHT_PASSES {
        return Vec::new();
    }
    let floor = FLOOR_MULT * snapshot.total_weight * (ROUTED_EXPERTS as f64 / K3_EXPERTS as f64);
    let mut ranked: Vec<(f32, u32)> = snapshot
        .heats
        .iter()
        .enumerate()
        .filter(|&(_, &heat)| f64::from(heat) >= floor)
        .map(|(slot, &heat)| (heat, slot as u32))
        .collect();
    ranked.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    ranked
}

/// Free host memory the automatic tier budget must leave untouched. The tier
/// takes only what remains above this line, so a squeezed host resolves to a
/// trimmed roster or to zero instead of competing with the spine, the page
/// cache, or context growth. The 2 GB explicit budget this floor typically
/// reproduces on the 64 GiB reference host is the configuration the 60-run
/// five-prompt campaign measured: +15–19% on expert-read-bound natural
/// language, neutral elsewhere, including its residency cost.
pub(crate) const AUTO_PIN_HEADROOM_BYTES: u64 = 8 << 30;

/// Resolve `K3_EXPERT_PIN_GB=auto` (the default) into a byte budget: exactly
/// what the qualifying roster needs, never more than free memory minus the
/// reserved headroom. Zero — tier off — when history is thin, nothing
/// qualifies, or the host is too tight. Unknown free memory fails closed.
pub(crate) fn resolve_auto_pin_budget(
    snapshot: &ExpertHeatSnapshot,
    span_bytes: usize,
    host_available_bytes: Option<u64>,
) -> u64 {
    let roster_bytes = qualifying_candidates(snapshot).len() as u64 * span_bytes as u64;
    let spare = host_available_bytes
        .unwrap_or(0)
        .saturating_sub(AUTO_PIN_HEADROOM_BYTES);
    roster_bytes.min(spare)
}

/// The permanent RAM tier. The candidate roster is frozen at construction, so
/// residency can only grow toward `candidate_count × span_bytes ≤ budget` and
/// no eviction logic exists. Promotion is sticky: an expert enters only when
/// an authoritative read already produced its authenticated bytes.
pub(crate) struct ExpertPinTier {
    span_bytes: usize,
    candidate_count: usize,
    candidates: Box<[[u64; HEAT_BITSET_WORDS]]>,
    resident: RwLock<HashMap<(u32, u16), Arc<PageAlignedSpan>>>,
    resident_count: AtomicU64,
    hits: AtomicU64,
}

impl ExpertPinTier {
    /// Freeze the candidate set from a startup snapshot. Returns None when
    /// the budget is zero, the confidence ramp is unmet, or nothing clears
    /// the frequency floor — callers treat None as "tier disabled".
    pub(crate) fn plan(
        snapshot: &ExpertHeatSnapshot,
        span_bytes: usize,
        budget_bytes: u64,
    ) -> Option<Self> {
        if budget_bytes == 0 || span_bytes == 0 {
            return None;
        }
        let mut ranked = qualifying_candidates(snapshot);
        let capacity = usize::try_from(budget_bytes / span_bytes as u64).unwrap_or(usize::MAX);
        ranked.truncate(capacity);
        if ranked.is_empty() {
            return None;
        }
        let mut candidates = vec![[0_u64; HEAT_BITSET_WORDS]; MOE_LAYERS].into_boxed_slice();
        for &(_, slot) in &ranked {
            let layer = slot as usize / K3_EXPERTS;
            let expert = slot as usize % K3_EXPERTS;
            candidates[layer][expert / 64] |= 1_u64 << (expert % 64);
        }
        Some(Self {
            span_bytes,
            candidate_count: ranked.len(),
            candidates,
            resident: RwLock::new(HashMap::with_capacity(ranked.len())),
            resident_count: AtomicU64::new(0),
            hits: AtomicU64::new(0),
        })
    }

    fn is_candidate(&self, layer: u32, expert: u16) -> bool {
        if !(K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST).contains(&layer)
            || (expert as usize) >= K3_EXPERTS
        {
            return false;
        }
        let words = &self.candidates[(layer - K3_MOE_LAYER_FIRST) as usize];
        words[expert as usize / 64] & (1_u64 << (expert as usize % 64)) != 0
    }

    /// A resident span for this expert, if promotion already happened. The
    /// candidate bitset screens non-candidates without touching the lock.
    pub(crate) fn lookup(&self, layer: u32, expert: u16) -> Option<Arc<PageAlignedSpan>> {
        if !self.is_candidate(layer, expert) {
            return None;
        }
        let resident = self.resident.read().ok()?;
        let span = resident.get(&(layer, expert)).cloned()?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(span)
    }

    /// Sticky promotion from an authenticated expert-major slab whose spans
    /// follow `canonical_expert_ids` order. Non-candidates and already
    /// resident experts are skipped; a shape mismatch is ignored outright
    /// because this path must never affect the authoritative read it trails.
    pub(crate) fn maybe_promote(
        &self,
        layer: u32,
        canonical_expert_ids: &[u16],
        expert_major_bytes: &[u8],
    ) {
        if canonical_expert_ids
            .len()
            .checked_mul(self.span_bytes)
            .is_none_or(|expected| expected != expert_major_bytes.len())
        {
            return;
        }
        for (index, &expert) in canonical_expert_ids.iter().enumerate() {
            if !self.is_candidate(layer, expert) {
                continue;
            }
            if let Ok(resident) = self.resident.read() {
                if resident.contains_key(&(layer, expert)) {
                    continue;
                }
            }
            let start = index * self.span_bytes;
            let Some(span) =
                PageAlignedSpan::copy_from(&expert_major_bytes[start..start + self.span_bytes])
            else {
                continue;
            };
            if let Ok(mut resident) = self.resident.write() {
                if resident.len() < self.candidate_count
                    && resident.insert((layer, expert), span).is_none()
                {
                    self.resident_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// The exact ceiling this tier can occupy; charged as a fixed host cost.
    pub(crate) fn charged_host_bytes(&self) -> u64 {
        self.candidate_count as u64 * self.span_bytes as u64
    }

    pub(crate) fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    pub(crate) fn resident_experts(&self) -> u64 {
        self.resident_count.load(Ordering::Relaxed)
    }

    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident_experts() * self.span_bytes as u64
    }

    pub(crate) fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub(crate) const fn span_bytes(&self) -> usize {
        self.span_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "deltafin-expert-heat-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(path.join("k3-meta")).unwrap();
            Self(path)
        }

        fn heat_path(&self) -> PathBuf {
            self.0.join("k3-meta").join(EXPERT_HEAT_FILE_NAME)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn route(ids: [u16; ROUTE_TOP_K]) -> [u16; ROUTE_TOP_K] {
        ids
    }

    fn one_route(first: u16) -> [u16; ROUTE_TOP_K] {
        let mut ids = [0_u16; ROUTE_TOP_K];
        for (offset, id) in ids.iter_mut().enumerate() {
            *id = first + offset as u16;
        }
        ids
    }

    fn snapshot_with(slots: &[(u32, u16, f32)], total_weight: f64) -> ExpertHeatSnapshot {
        let mut heats = vec![0.0_f32; HEAT_SLOTS].into_boxed_slice();
        for &(layer, expert, heat) in slots {
            heats[(layer - 1) as usize * K3_EXPERTS + expert as usize] = heat;
        }
        ExpertHeatSnapshot {
            heats,
            total_weight,
        }
    }

    #[test]
    fn round_trips_and_merges_with_decay() {
        let root = TestDirectory::new();
        let heat = ExpertHeat::open(&root.0, true);
        assert_eq!(heat.snapshot().total_weight, 0.0);

        // Three rows in one mailbox: presence counts once per expert.
        heat.observe_layer_routes(1, [&one_route(0), &one_route(0), &one_route(8)].into_iter());
        assert!(!heat.note_pass_committed());
        heat.flush_best_effort();

        let reloaded = load_snapshot(&root.heat_path()).unwrap();
        assert_eq!(reloaded.total_weight, 1.0);
        // Experts 0..=7 shared by both duplicate rows still count 1.0.
        assert_eq!(reloaded.heats[0], 1.0);
        assert_eq!(reloaded.heats[8], 1.0);
        assert_eq!(reloaded.heats[23], 1.0);
        assert_eq!(reloaded.heats[24], 0.0);

        // A second run merges on top with one pass of decay.
        let second = ExpertHeat::open(&root.0, true);
        second.observe_layer_routes(1, [&one_route(0)].into_iter());
        second.note_pass_committed();
        second.flush_best_effort();
        let merged = load_snapshot(&root.heat_path()).unwrap();
        let factor = 0.5_f64.powf(1.0 / HALF_LIFE_PASSES);
        let expected_weight = factor + 1.0;
        assert!((merged.total_weight - expected_weight).abs() < 1e-9);
        let expected_rerouted = (factor + 1.0) as f32;
        assert!((merged.heats[0] - expected_rerouted).abs() < 1e-6);
        assert!((merged.heats[8] - expected_rerouted).abs() < 1e-6);
        // Expert 20 appeared only in the first run: pure decay.
        assert!((merged.heats[20] - factor as f32).abs() < 1e-6);
    }

    #[test]
    fn invalid_files_are_ignored_and_rebuilt() {
        let root = TestDirectory::new();
        let path = root.heat_path();

        for corrupt in [
            b"garbage".to_vec(),
            vec![0_u8; HEAT_FILE_BYTES - 1],
            {
                let mut bytes = encode_snapshot(&snapshot_with(&[(1, 0, 4.0)], 4.0));
                bytes[HEAT_HEADER_BYTES] ^= 0x40;
                bytes
            },
            {
                let mut snapshot = snapshot_with(&[], 4.0);
                snapshot.total_weight = f64::NAN;
                let heats = std::mem::take(&mut snapshot.heats);
                let mut bytes = encode_snapshot(&ExpertHeatSnapshot {
                    heats,
                    total_weight: f64::NAN,
                });
                // Re-seal so only the NaN weight is wrong.
                let digest = Sha256::digest(&bytes[..HEAT_FILE_BYTES - HEAT_DIGEST_BYTES]);
                let offset = HEAT_FILE_BYTES - HEAT_DIGEST_BYTES;
                bytes[offset..].copy_from_slice(&digest);
                bytes
            },
        ] {
            fs::write(&path, &corrupt).unwrap();
            assert!(load_snapshot(&path).is_err());
            let heat = ExpertHeat::open(&root.0, true);
            assert_eq!(heat.snapshot().total_weight, 0.0);
            heat.observe_layer_routes(1, [&one_route(0)].into_iter());
            heat.note_pass_committed();
            heat.flush_best_effort();
            let rebuilt = load_snapshot(&path).unwrap();
            assert_eq!(rebuilt.total_weight, 1.0);
            fs::remove_file(&path).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_history_is_never_followed() {
        let root = TestDirectory::new();
        let target = root.0.join("target.bin");
        let original = encode_snapshot(&snapshot_with(&[(1, 3, 9.0)], 9.0));
        fs::write(&target, &original).unwrap();
        crate::sys::fs::symlink(&target, root.heat_path()).unwrap();

        // Loading refuses the symlink.
        assert!(load_snapshot(&root.heat_path()).is_err());

        // Flushing replaces the symlink entry itself; the target is untouched.
        let heat = ExpertHeat::open(&root.0, true);
        heat.observe_layer_routes(1, [&one_route(0)].into_iter());
        heat.note_pass_committed();
        heat.flush_best_effort();
        assert!(root.heat_path().symlink_metadata().unwrap().is_file());
        assert_eq!(fs::read(&target).unwrap(), original);
        assert_eq!(load_snapshot(&root.heat_path()).unwrap().total_weight, 1.0);
    }

    #[test]
    fn concurrent_flushes_serialize_and_conserve_mass() {
        let root = TestDirectory::new();
        let flushes_per_thread = 8_usize;
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    let heat = ExpertHeat::open(&root.0, true);
                    for _ in 0..flushes_per_thread {
                        heat.observe_layer_routes(2, [&one_route(0)].into_iter());
                        heat.note_pass_committed();
                        // Contention restores deltas, so retry until merged.
                        // The lock holder keeps the lock across a file and a
                        // directory fsync, which under a loaded parallel test
                        // run can outlast any fixed number of yields; bound
                        // the retry by time and let the loser actually sleep.
                        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                        let mut merged = false;
                        while std::time::Instant::now() < deadline {
                            heat.flush_best_effort();
                            if heat.pass_delta.load(Ordering::Relaxed) == 0 {
                                merged = true;
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                        assert!(merged, "flush never acquired the heat lock within 30 s");
                    }
                });
            }
        });
        let merged = load_snapshot(&root.heat_path()).unwrap();
        // Sixteen single-pass merges regardless of interleaving order:
        // weight_n = weight_{n-1} * λ + 1.
        let factor = 0.5_f64.powf(1.0 / HALF_LIFE_PASSES);
        let mut expected = 0.0_f64;
        for _ in 0..(2 * flushes_per_thread) {
            expected = expected * factor + 1.0;
        }
        assert!((merged.total_weight - expected).abs() < 1e-6);
        let slot = K3_EXPERTS + 5;
        assert!((f64::from(merged.heats[slot]) - expected).abs() < 1e-3);
    }

    #[test]
    fn presence_ignores_row_multiplicity_and_bad_indices() {
        let root = TestDirectory::new();
        let heat = ExpertHeat::open(&root.0, true);
        let mut bad = one_route(0);
        bad[15] = (K3_EXPERTS as u16) + 7;
        heat.observe_layer_routes(5, [&bad, &bad, &bad].into_iter());
        heat.observe_layer_routes(0, [&route(one_route(0))].into_iter());
        heat.observe_layer_routes(93, [&route(one_route(0))].into_iter());
        heat.note_pass_committed();
        heat.flush_best_effort();
        let merged = load_snapshot(&root.heat_path()).unwrap();
        let base = 4 * K3_EXPERTS;
        assert_eq!(merged.heats[base], 1.0);
        assert_eq!(merged.heats[base + 14], 1.0);
        let total: f64 = merged.heats.iter().map(|&h| f64::from(h)).sum();
        assert_eq!(total, 15.0);
    }

    #[test]
    fn disabled_recording_observes_and_flushes_nothing() {
        let root = TestDirectory::new();
        let heat = ExpertHeat::open(&root.0, false);
        heat.observe_layer_routes(1, [&one_route(0)].into_iter());
        assert!(!heat.note_pass_committed());
        heat.flush_best_effort();
        assert!(!root.heat_path().exists());
    }

    #[test]
    fn plan_requires_budget_ramp_and_floor() {
        let span = 16_384_usize;
        let hot = snapshot_with(&[(1, 7, 300.0), (2, 11, 200.0)], 512.0);
        assert!(ExpertPinTier::plan(&hot, span, 0).is_none());
        let thin = snapshot_with(&[(1, 7, 100.0)], MIN_WEIGHT_PASSES - 1.0);
        assert!(ExpertPinTier::plan(&thin, span, u64::MAX).is_none());
        // Floor: 2 × 512 × 16/896 ≈ 18.29 — a uniform-ish expert stays out.
        let mixed = snapshot_with(&[(1, 7, 300.0), (3, 40, 18.0)], 512.0);
        let tier = ExpertPinTier::plan(&mixed, span, u64::MAX).unwrap();
        assert_eq!(tier.candidate_count(), 1);
        assert!(tier.is_candidate(1, 7));
        assert!(!tier.is_candidate(3, 40));
        // Everything under the floor: no tier at all.
        let cold = snapshot_with(&[(1, 7, 17.0)], 512.0);
        assert!(ExpertPinTier::plan(&cold, span, u64::MAX).is_none());
    }

    #[test]
    fn plan_ranks_globally_and_truncates_to_budget() {
        let span = 16_384_usize;
        let snapshot = snapshot_with(
            &[
                (5, 100, 400.0),
                (1, 7, 300.0),
                (2, 11, 300.0),
                (92, 895, 250.0),
            ],
            512.0,
        );
        let tier = ExpertPinTier::plan(&snapshot, span, 3 * span as u64).unwrap();
        assert_eq!(tier.candidate_count(), 3);
        assert_eq!(tier.charged_host_bytes(), 3 * span as u64);
        assert!(tier.is_candidate(5, 100));
        // The 300.0 tie breaks toward the lower (layer, expert) slot.
        assert!(tier.is_candidate(1, 7));
        assert!(tier.is_candidate(2, 11));
        assert!(!tier.is_candidate(92, 895));
    }

    #[test]
    fn promotion_is_candidates_only_sticky_and_shape_checked() {
        let span = 4_096_usize;
        let snapshot = snapshot_with(&[(4, 2, 300.0), (4, 9, 280.0)], 512.0);
        let tier = ExpertPinTier::plan(&snapshot, span, 2 * span as u64).unwrap();

        let slab: Vec<u8> = (0..3 * span).map(|byte| (byte / span + 1) as u8).collect();
        // Shape mismatch: silently ignored.
        tier.maybe_promote(4, &[2, 5, 9], &slab[..span]);
        assert_eq!(tier.resident_experts(), 0);

        tier.maybe_promote(4, &[2, 5, 9], &slab);
        assert_eq!(tier.resident_experts(), 2);
        assert_eq!(tier.resident_bytes(), 2 * span as u64);
        assert!(tier.lookup(4, 5).is_none());
        let two = tier.lookup(4, 2).unwrap();
        assert_eq!(&two[..], &slab[..span]);
        assert_eq!(two.as_ptr() as usize % BUFFER_ALIGNMENT, 0);
        let nine = tier.lookup(4, 9).unwrap();
        assert_eq!(&nine[..], &slab[2 * span..]);
        assert_eq!(tier.hits(), 2);

        // Sticky: a later slab with different bytes cannot replace residents.
        let other: Vec<u8> = vec![0xEE; span];
        tier.maybe_promote(4, &[2], &other);
        assert_eq!(&tier.lookup(4, 2).unwrap()[..], &slab[..span]);
        assert_eq!(tier.resident_experts(), 2);
    }

    #[test]
    fn auto_budget_is_roster_sized_and_headroom_capped() {
        let span = 16_384_usize;
        let hot = snapshot_with(&[(1, 7, 300.0), (2, 11, 280.0), (5, 100, 260.0)], 512.0);
        let roster_bytes = 3 * span as u64;

        // Spacious host: exactly what the roster needs, nothing more.
        let spacious = Some(AUTO_PIN_HEADROOM_BYTES + 10 * span as u64);
        assert_eq!(resolve_auto_pin_budget(&hot, span, spacious), roster_bytes);
        // The resolved budget admits the complete roster through plan().
        let tier = ExpertPinTier::plan(&hot, span, roster_bytes).unwrap();
        assert_eq!(tier.candidate_count(), 3);

        // Tight host: trimmed to whatever clears the headroom floor.
        let tight = Some(AUTO_PIN_HEADROOM_BYTES + 2 * span as u64);
        assert_eq!(resolve_auto_pin_budget(&hot, span, tight), 2 * span as u64);

        // At or below the floor, or unknown: fail closed to off.
        assert_eq!(resolve_auto_pin_budget(&hot, span, Some(AUTO_PIN_HEADROOM_BYTES)), 0);
        assert_eq!(resolve_auto_pin_budget(&hot, span, Some(1)), 0);
        assert_eq!(resolve_auto_pin_budget(&hot, span, None), 0);

        // Thin history: zero regardless of how much memory is free.
        let thin = snapshot_with(&[(1, 7, 300.0)], MIN_WEIGHT_PASSES - 1.0);
        assert_eq!(resolve_auto_pin_budget(&thin, span, spacious), 0);
        // Nothing clears the floor: likewise zero.
        let cold = snapshot_with(&[(1, 7, 1.0)], 512.0);
        assert_eq!(resolve_auto_pin_budget(&cold, span, spacious), 0);
    }

    #[test]
    fn lookup_misses_before_promotion_and_off_candidates() {
        let span = 1_024_usize;
        let snapshot = snapshot_with(&[(7, 0, 300.0)], 512.0);
        let tier = ExpertPinTier::plan(&snapshot, span, span as u64).unwrap();
        assert!(tier.lookup(7, 0).is_none());
        assert!(tier.lookup(7, 1).is_none());
        assert!(tier.lookup(0, 0).is_none());
        assert!(tier.lookup(93, 0).is_none());
        assert_eq!(tier.hits(), 0);
    }
}
