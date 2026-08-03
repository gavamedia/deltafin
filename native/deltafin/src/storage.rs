//! Persistent, Python-free positional I/O for streamed model weights.
//!
//! A `ReadPlan` opens every immutable source once, validates complete source
//! and destination coverage, and splits large extents before the hot path. A
//! `Reader` owns fixed worker threads and a bounded reusable buffer arena.
//! Submitted reads return tickets immediately after admission; workers pull
//! short quanta from demand/prefetch queues instead of constructing a Future,
//! dictionary, or memoryview per chunk.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::FileExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::error::{DeltafinError, Result};
use crate::packfile::DigestState;

#[cfg(target_os = "macos")]
const BUFFER_ALIGNMENT: usize = 16 * 1024;
#[cfg(not(target_os = "macos"))]
// Linux/aarch64 deployments can use 64 KiB base pages. A conservative 64 KiB
// alignment also remains valid on 4 KiB x86-64 hosts and avoids rebuilding
// the arena when direct/pinned provider uploads are enabled later.
const BUFFER_ALIGNMENT: usize = 64 * 1024;
const MAX_PLAN_JOBS: usize = 1_000_000;
const DEFAULT_ARENA_SLOTS: usize = 2;
const WORK_QUANTUM: usize = 4;
const MAX_DEMAND_STREAK: usize = 8;
// Keep storage from consuming the process-wide descriptor table. Packed
// layers retain their descriptors across reads, but only inside this budget.
// A complete loose K3 int8 spine has several thousand immutable components.
// The high ceiling is not an allocation target: every admission is still
// proven against the live process soft limit and leaves explicit headroom.
const MAX_PERSISTENT_STORAGE_DESCRIPTORS: usize = 8_192;
const MIN_DESCRIPTOR_HEADROOM: usize = 32;
const MAX_DESCRIPTOR_HEADROOM: usize = 1_024;
const FALLBACK_SOFT_DESCRIPTOR_LIMIT: usize = 256;
pub const LOOSE_SPINE_DESCRIPTOR_RESERVE: usize = 1_024;
// Routed decode always selects exactly 16 experts. Keeping those whole-file
// jobs inside the already-required Batch allocation removes a second jobs
// allocation and its Arc without inflating every generic spine batch.
const MAX_INLINE_DEFERRED_FILES: usize = 16;
// A loose K3 layer currently has at most 42 source components. Keep modest
// format headroom, while proving a batch can never recreate the old unbounded
// descriptor cache. With the default two arena slots, at most 128 manifest
// descriptors can coexist even if both reads reach their widest point.
const MAX_DEFERRED_MANIFEST_SOURCES: usize = 64;
// The exact full-commit Scale4 verifier is bounded to nine rows × 16 routed
// experts. Its widest union therefore names 144 raw files plus one shared
// layer sidecar and at most two authenticated ranges per expert. Keep exact
// headroom for that compile-time ceiling without admitting unbounded
// path/range metadata. The independent loose-spine manifest limit above stays
// at 64 sources.
const MAX_DEFERRED_AUTHENTICATED_SOURCES: usize = 145;
const MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS: usize = 288;
// Keep vectored positional reads bounded on the stack and comfortably below
// the minimum POSIX IOV_MAX. Scale4 uses four destinations: its header and
// three exact compressed-scale planes.
const MAX_VECTORED_DESTINATIONS: usize = 16;
// K3's longest canonical expert name is 12 bytes. Thirty-one leaves ample
// format headroom while keeping all 82,432 catalog entries near 2.7 MiB rather
// than doubling that cold but permanently resident metadata.
const MAX_DEFERRED_SOURCE_NAME_BYTES: usize = 31;

#[derive(Debug)]
struct DescriptorBudget {
    capacity: usize,
    in_use: Mutex<usize>,
    soft_limit: usize,
    observed_open: Option<usize>,
    headroom: usize,
}

impl DescriptorBudget {
    fn for_process() -> Arc<Self> {
        let soft_limit = soft_descriptor_limit().unwrap_or(FALLBACK_SOFT_DESCRIPTOR_LIMIT);
        let observed_open = count_open_descriptors();
        let proportional_headroom = (soft_limit / 8).clamp(
            MIN_DESCRIPTOR_HEADROOM.min(soft_limit),
            MAX_DESCRIPTOR_HEADROOM.min(soft_limit),
        );
        let (capacity, headroom) = if let Some(open) = observed_open {
            (
                soft_limit
                    .saturating_sub(open)
                    .saturating_sub(proportional_headroom)
                    .min(MAX_PERSISTENT_STORAGE_DESCRIPTORS),
                proportional_headroom,
            )
        } else {
            // Both supported hosts expose /dev/fd or /proc/self/fd. If that
            // inspection is unavailable, consume at most one quarter of the
            // soft limit and leave the remainder to the runtime and libraries.
            let capacity = (soft_limit / 4)
                .min(MAX_PERSISTENT_STORAGE_DESCRIPTORS)
                .min(64);
            (capacity, soft_limit.saturating_sub(capacity))
        };
        Arc::new(Self {
            capacity,
            in_use: Mutex::new(0),
            soft_limit,
            observed_open,
            headroom,
        })
    }

    #[cfg(test)]
    fn fixed(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            in_use: Mutex::new(0),
            soft_limit: capacity,
            observed_open: Some(0),
            headroom: 0,
        })
    }

    fn reserve(self: &Arc<Self>, count: usize) -> Result<DescriptorReservation> {
        let mut in_use = self.in_use.lock().unwrap();
        let available = self.capacity.saturating_sub(*in_use);
        if count > available {
            return Err(DeltafinError::new(format!(
                "read plan needs {count} persistent source descriptors, but only {available} remain in Deltafin's {}-descriptor storage budget (soft limit {}, observed open {}, reserved headroom {}); use packed layer files, release inactive read plans, or raise the descriptor limit",
                self.capacity,
                self.soft_limit,
                self.observed_open
                    .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                self.headroom,
            )));
        }
        *in_use += count;
        drop(in_use);
        Ok(DescriptorReservation {
            budget: Arc::clone(self),
            count,
        })
    }
}

#[derive(Debug)]
struct DescriptorReservation {
    budget: Arc<DescriptorBudget>,
    count: usize,
}

impl Drop for DescriptorReservation {
    fn drop(&mut self) {
        let mut in_use = self.budget.in_use.lock().unwrap();
        debug_assert!(*in_use >= self.count);
        *in_use -= self.count;
    }
}

fn process_descriptor_budget() -> Arc<DescriptorBudget> {
    static BUDGET: OnceLock<Arc<DescriptorBudget>> = OnceLock::new();
    Arc::clone(BUDGET.get_or_init(DescriptorBudget::for_process))
}

/// Prove that an all-or-nothing persistent loose-source roster can coexist
/// with the process and provider descriptor working set.
///
/// The soft limit is raised only for this process, only when the hard limit
/// already permits it, and before the singleton storage budget is frozen.
/// Failure never admits a partial cache; automatic callers can fall back to
/// ordinary per-batch descriptors, while explicit callers surface the error.
pub fn prepare_persistent_descriptor_capacity(required: usize, reserve: usize) -> Result<()> {
    if required == 0 {
        return Ok(());
    }
    if required > MAX_PERSISTENT_STORAGE_DESCRIPTORS {
        return Err(DeltafinError::new(format!(
            "persistent loose-spine roster needs {required} descriptors, exceeding Deltafin's {MAX_PERSISTENT_STORAGE_DESCRIPTORS}-descriptor safety ceiling"
        )));
    }
    let open = count_open_descriptors().ok_or_else(|| {
        DeltafinError::new(
            "cannot inspect the live process descriptor count; persistent loose-spine cache is not safe",
        )
    })?;
    let needed_soft = open
        .checked_add(required)
        .and_then(|value| value.checked_add(reserve))
        .ok_or_else(|| DeltafinError::new("persistent descriptor requirement overflows usize"))?;
    let (mut soft, hard) = descriptor_limits().ok_or_else(|| {
        DeltafinError::new(
            "cannot inspect the process descriptor limits; persistent loose-spine cache is not safe",
        )
    })?;
    if soft < needed_soft {
        let target = needed_soft
            .checked_next_power_of_two()
            .unwrap_or(needed_soft)
            .max(4_096)
            .min(hard);
        if target < needed_soft || !set_soft_descriptor_limit(target) {
            return Err(DeltafinError::new(format!(
                "persistent loose-spine cache needs {required} descriptors plus {reserve} reserved (currently {open} open), but the process soft/hard limits are {soft}/{hard}"
            )));
        }
        soft = descriptor_limits().map_or(soft, |limits| limits.0);
    }
    if soft.saturating_sub(open).saturating_sub(reserve) < required {
        return Err(DeltafinError::new(format!(
            "persistent loose-spine cache needs {required} descriptors plus {reserve} reserved (currently {open} open), but only {soft} are admitted"
        )));
    }
    let budget = process_descriptor_budget();
    let available = budget
        .capacity
        .saturating_sub(*budget.in_use.lock().unwrap());
    if available < required {
        return Err(DeltafinError::new(format!(
            "persistent loose-spine cache needs {required} descriptors, but the initialized storage budget has only {available} available"
        )));
    }
    Ok(())
}

#[repr(C)]
struct NativeRlimit {
    current: u64,
    maximum: u64,
}

fn soft_descriptor_limit() -> Option<usize> {
    descriptor_limits().map(|limits| limits.0)
}

fn descriptor_limits() -> Option<(usize, usize)> {
    #[cfg(target_os = "linux")]
    const RLIMIT_NOFILE: i32 = 7;
    #[cfg(target_os = "macos")]
    const RLIMIT_NOFILE: i32 = 8;
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return None;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    unsafe extern "C" {
        fn getrlimit(resource: i32, limits: *mut NativeRlimit) -> i32;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut limits = NativeRlimit {
            current: 0,
            maximum: 0,
        };
        // SAFETY: `limits` points to writable storage with the platform's
        // two-rlim_t layout; Linux and Darwin use 64-bit rlim_t on supported
        // x86-64/aarch64 targets.
        if unsafe { getrlimit(RLIMIT_NOFILE, &mut limits) } != 0 {
            return None;
        }
        Some((
            usize::try_from(limits.current).ok()?,
            usize::try_from(limits.maximum).ok()?,
        ))
    }
}

fn set_soft_descriptor_limit(soft: usize) -> bool {
    #[cfg(target_os = "linux")]
    const RLIMIT_NOFILE: i32 = 7;
    #[cfg(target_os = "macos")]
    const RLIMIT_NOFILE: i32 = 8;
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return false;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    unsafe extern "C" {
        fn setrlimit(resource: i32, limits: *const NativeRlimit) -> i32;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let Some((_, hard)) = descriptor_limits() else {
            return false;
        };
        if soft > hard {
            return false;
        }
        let limits = NativeRlimit {
            current: soft as u64,
            maximum: hard as u64,
        };
        // SAFETY: `limits` has the supported platform's two-rlim_t layout and
        // remains live for the duration of this process-local syscall.
        unsafe { setrlimit(RLIMIT_NOFILE, &limits) == 0 }
    }
}

fn count_open_descriptors() -> Option<usize> {
    #[cfg(target_os = "linux")]
    const FD_DIRECTORIES: [&str; 2] = ["/proc/self/fd", "/dev/fd"];
    #[cfg(not(target_os = "linux"))]
    const FD_DIRECTORIES: [&str; 2] = ["/dev/fd", "/proc/self/fd"];

    for directory in FD_DIRECTORIES {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        let mut count = 0_usize;
        let mut complete = true;
        for entry in entries {
            if entry.is_err() {
                complete = false;
                break;
            }
            let Some(next) = count.checked_add(1) else {
                complete = false;
                break;
            };
            count = next;
        }
        if complete {
            return Some(count);
        }
    }
    None
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BufferKind {
    Quantized,
    Scales,
    Other,
}

impl BufferKind {
    const COUNT: usize = 3;
    const ALL: [Self; Self::COUNT] = [Self::Quantized, Self::Scales, Self::Other];

    const fn index(self) -> usize {
        match self {
            Self::Quantized => 0,
            Self::Scales => 1,
            Self::Other => 2,
        }
    }
}

/// Exact logical sizes of the three packed destination buffers.
///
/// These are supplied by the audited tensor manifest rather than inferred
/// from the last extent. That makes a missing trailing extent an error instead
/// of silently producing a shorter tensor.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct BufferLengths {
    pub quantized: usize,
    pub scales: usize,
    pub other: usize,
}

impl BufferLengths {
    pub const fn new(quantized: usize, scales: usize, other: usize) -> Self {
        Self {
            quantized,
            scales,
            other,
        }
    }

    const fn get(self, kind: BufferKind) -> usize {
        match kind {
            BufferKind::Quantized => self.quantized,
            BufferKind::Scales => self.scales,
            BufferKind::Other => self.other,
        }
    }

    const fn as_array(self) -> [usize; BufferKind::COUNT] {
        [self.quantized, self.scales, self.other]
    }

    fn max(self, other: Self) -> Self {
        Self::new(
            self.quantized.max(other.quantized),
            self.scales.max(other.scales),
            self.other.max(other.other),
        )
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CachePolicy {
    Resident,
    Streaming,
}

/// One direct child of a deferred-read directory, stored inline and already
/// NUL terminated for `openat(2)`. Catalog construction is cold-path work;
/// selecting a source later copies only its integer index and never creates a
/// `PathBuf`, `String`, `CString`, or hash-table entry.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct DeferredSourceName {
    bytes: [u8; MAX_DEFERRED_SOURCE_NAME_BYTES + 1],
    len: u8,
}

impl DeferredSourceName {
    pub fn new(name: &str) -> Result<Self> {
        let source = name.as_bytes();
        if source.is_empty() {
            return Err(DeltafinError::new("a deferred source name cannot be empty"));
        }
        if source.len() > MAX_DEFERRED_SOURCE_NAME_BYTES {
            return Err(DeltafinError::new(format!(
                "deferred source name is {} bytes; maximum is {MAX_DEFERRED_SOURCE_NAME_BYTES}",
                source.len()
            )));
        }
        if source == b"." || source == b".." || source.contains(&b'/') || source.contains(&0) {
            return Err(DeltafinError::new(
                "deferred source must be one non-special directory entry",
            ));
        }
        let mut bytes = [0_u8; MAX_DEFERRED_SOURCE_NAME_BYTES + 1];
        bytes[..source.len()].copy_from_slice(source);
        Ok(Self {
            bytes,
            len: source.len() as u8,
        })
    }

    fn as_c_str(&self) -> &CStr {
        // Construction rejects interior NULs and the fixed trailing byte is
        // zero, so this prefix is always exactly one valid C string.
        CStr::from_bytes_with_nul(&self.bytes[..usize::from(self.len) + 1])
            .expect("validated deferred source name must remain NUL terminated")
    }

    pub(crate) fn as_str(&self) -> &str {
        // Construction accepts UTF-8 `str` input and never mutates the prefix.
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("validated deferred source name must remain UTF-8")
    }
}

impl std::fmt::Debug for DeferredSourceName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("DeferredSourceName")
            .field(&self.as_str())
            .finish()
    }
}

#[derive(Debug)]
struct DeferredExactCatalogInner {
    directory: File,
    directory_path: PathBuf,
    sources: Box<[DeferredSourceName]>,
    exact_source_length: u64,
    cache_policy: CachePolicy,
}

/// Immutable directory-relative source catalog for repeated whole-file reads.
///
/// The directory itself is opened once with `O_NOFOLLOW`. Individual files
/// remain ephemeral: workers use `openat` with `O_NOFOLLOW|O_CLOEXEC`, validate
/// regular-file type and exact size on that live descriptor, read it, and
/// close it. This keeps routed experts inside a bounded FD budget while moving
/// path assembly, hashing, source deduplication, and plan validation entirely
/// out of the per-layer path.
#[derive(Clone, Debug)]
pub struct DeferredExactCatalog {
    inner: Arc<DeferredExactCatalogInner>,
}

impl DeferredExactCatalog {
    pub fn open(
        directory: &Path,
        sources: impl IntoIterator<Item = DeferredSourceName>,
        exact_source_length: u64,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        if exact_source_length == 0 {
            return Err(DeltafinError::new(
                "a deferred source catalog needs a positive exact length",
            ));
        }
        let sources: Vec<_> = sources.into_iter().collect();
        if sources.is_empty() {
            return Err(DeltafinError::new(
                "a deferred source catalog needs at least one source",
            ));
        }
        let directory_file = OpenOptions::new()
            .read(true)
            .custom_flags(open_cloexec_nofollow())
            .open(directory)
            .map_err(|error| {
                io_error(
                    "open deferred source directory without following symlinks",
                    directory,
                    error,
                )
            })?;
        let metadata = directory_file
            .metadata()
            .map_err(|error| io_error("stat deferred source directory", directory, error))?;
        if !metadata.is_dir() {
            return Err(DeltafinError::new(format!(
                "deferred source catalog root is not a directory: {}",
                directory.display()
            )));
        }
        Ok(Self {
            inner: Arc::new(DeferredExactCatalogInner {
                directory: directory_file,
                directory_path: directory.to_path_buf(),
                sources: sources.into_boxed_slice(),
                exact_source_length,
                cache_policy,
            }),
        })
    }

    pub fn source_count(&self) -> usize {
        self.inner.sources.len()
    }

    pub fn exact_source_length(&self) -> u64 {
        self.inner.exact_source_length
    }

    #[cfg(test)]
    fn source_name(&self, index: usize) -> Option<&str> {
        self.inner
            .sources
            .get(index)
            .map(DeferredSourceName::as_str)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Extent {
    Read {
        path: PathBuf,
        source_offset: u64,
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
        /// Optional digest of this complete extent. Verified extents are never
        /// split into smaller jobs, and their successful first-read result is
        /// cached inside the immutable `ReadPlan` that owns the opened file.
        expected_digest: Option<[u8; 32]>,
    },
    /// One contiguous source range scattered, in source order, into several
    /// disjoint destination ranges by a single positional vectored read.
    ///
    /// This is deliberately narrower than an arbitrary gather: source bytes
    /// may not contain gaps or change order. That exact contract maps to
    /// `preadv(2)` on both macOS and Linux without temporary buffers or an
    /// additional copy.
    ReadVectored {
        path: PathBuf,
        source_offset: u64,
        destinations: Box<[VectoredDestination]>,
        /// Optional digest over the contiguous source range, reconstructed by
        /// hashing the ordered destination vectors after `preadv` completes.
        expected_digest: Option<[u8; 32]>,
    },
    /// An explicit zero/padding range. Implicit holes are rejected.
    Zero {
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
    },
}

impl Extent {
    pub fn new(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
    ) -> Self {
        Self::Read {
            path: path.into(),
            source_offset,
            destination,
            destination_offset,
            length,
            expected_digest: None,
        }
    }

    /// Describe one independently authenticated file range. This is used by
    /// DFSP pack chunks: hashing the bytes already in the destination slab
    /// avoids a second disk pass, while the verification bit remains tied to
    /// this read plan's exact, already-open descriptor generation.
    pub fn verified(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
        expected_digest: [u8; 32],
    ) -> Self {
        Self::Read {
            path: path.into(),
            source_offset,
            destination,
            destination_offset,
            length,
            expected_digest: Some(expected_digest),
        }
    }

    /// Describe a contiguous source range whose bytes are scattered into
    /// disjoint destination ranges in the supplied order.
    pub fn vectored(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destinations: impl IntoIterator<Item = VectoredDestination>,
    ) -> Result<Self> {
        Self::vectored_with_digest(path, source_offset, destinations, None)
    }

    /// Describe a vectored positional read whose complete contiguous source
    /// range must match `expected_digest` before its private batch can publish.
    pub fn vectored_verified(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destinations: impl IntoIterator<Item = VectoredDestination>,
        expected_digest: [u8; 32],
    ) -> Result<Self> {
        Self::vectored_with_digest(path, source_offset, destinations, Some(expected_digest))
    }

    fn vectored_with_digest(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destinations: impl IntoIterator<Item = VectoredDestination>,
        expected_digest: Option<[u8; 32]>,
    ) -> Result<Self> {
        let destinations: Vec<_> = destinations.into_iter().collect();
        if destinations.is_empty() {
            return Err(DeltafinError::new(
                "a vectored read needs at least one destination",
            ));
        }
        if destinations.len() > MAX_VECTORED_DESTINATIONS {
            return Err(DeltafinError::new(format!(
                "a vectored read has {} destinations; bounded maximum is {MAX_VECTORED_DESTINATIONS}",
                destinations.len()
            )));
        }
        let mut total = 0_usize;
        for destination in &destinations {
            if destination.length == 0 {
                return Err(DeltafinError::new(
                    "a vectored read destination may not be empty",
                ));
            }
            total = total
                .checked_add(destination.length)
                .ok_or_else(|| DeltafinError::new("vectored read length overflows usize"))?;
        }
        source_offset
            .checked_add(total as u64)
            .ok_or_else(|| DeltafinError::new("vectored source range overflows u64"))?;
        Ok(Self::ReadVectored {
            path: path.into(),
            source_offset,
            destinations: destinations.into_boxed_slice(),
            expected_digest,
        })
    }

    pub const fn zero(destination: BufferKind, destination_offset: usize, length: usize) -> Self {
        Self::Zero {
            destination,
            destination_offset,
            length,
        }
    }
}

/// One output range in a contiguous positional vectored read.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct VectoredDestination {
    destination: BufferKind,
    destination_offset: usize,
    length: usize,
}

impl VectoredDestination {
    pub const fn new(destination: BufferKind, destination_offset: usize, length: usize) -> Self {
        Self {
            destination,
            destination_offset,
            length,
        }
    }
}

/// One exact source range that must authenticate before any gathered bytes
/// from the same live descriptor may be published.
///
/// Unlike [`Extent::verified`], this digest is over source bytes rather than
/// one contiguous destination extent. It therefore supports strict gathers
/// where an authenticated source record is scattered into disjoint output
/// ranges. The source is opened lazily by a reader worker with `O_NOFOLLOW`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeferredSourceVerification {
    path: PathBuf,
    exact_source_length: u64,
    source_offset: u64,
    length: usize,
    expected_digest: [u8; 32],
}

/// Exact live-descriptor length for a deferred partial-range source.
///
/// This is the non-hashing counterpart to [`DeferredSourceVerification`]. It
/// exists for formats whose retained bytes are authenticated after assembly
/// (for example, a Scale4 sidecar record) while their canonical raw packed
/// planes use the same exact-length/no-follow trust boundary as raw-v1 expert
/// execution. Workers still open every source with `O_NOFOLLOW|O_CLOEXEC` and
/// validate the regular file and exact length on the descriptor they read.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeferredSourceLength {
    path: PathBuf,
    exact_source_length: u64,
    expected_identity: Option<DeferredSourceIdentity>,
}

impl DeferredSourceLength {
    pub fn new(path: impl Into<PathBuf>, exact_source_length: u64) -> Self {
        Self {
            path: path.into(),
            exact_source_length,
            expected_identity: None,
        }
    }

    /// Pin the current no-follow descriptor identity as part of this range
    /// contract. The reader rechecks that identity on the descriptor used for
    /// every positional read and once more after the complete batch finishes.
    /// This is intended for immutable files whose successful record digests may
    /// be cached by identity across batches.
    pub fn new_with_live_identity(
        path: impl Into<PathBuf>,
        exact_source_length: u64,
    ) -> Result<Self> {
        let path = path.into();
        let expected_identity = capture_deferred_source_identity(&path, exact_source_length)?;
        Ok(Self {
            path,
            exact_source_length,
            expected_identity: Some(expected_identity),
        })
    }

    /// Reuse an identity previously captured by
    /// [`Self::new_with_live_identity`] without synchronously reopening the
    /// source. This crate-private constructor is reserved for immutable session
    /// caches: workers still open with `O_NOFOLLOW`, require the exact length,
    /// compare this identity before reading, and compare it again after the
    /// batch completes.
    pub(crate) fn new_with_captured_identity(
        path: impl Into<PathBuf>,
        exact_source_length: u64,
        expected_identity: DeferredSourceIdentity,
    ) -> Result<Self> {
        if expected_identity.bytes != exact_source_length {
            return Err(DeltafinError::new(format!(
                "captured deferred source identity is {} bytes; expected exact length {exact_source_length}",
                expected_identity.bytes,
            )));
        }
        Ok(Self {
            path: path.into(),
            exact_source_length,
            expected_identity: Some(expected_identity),
        })
    }

    pub const fn identity(&self) -> Option<DeferredSourceIdentity> {
        self.expected_identity
    }
}

impl DeferredSourceVerification {
    pub fn new(
        path: impl Into<PathBuf>,
        exact_source_length: u64,
        source_offset: u64,
        length: usize,
        expected_digest: [u8; 32],
    ) -> Self {
        Self {
            path: path.into(),
            exact_source_length,
            source_offset,
            length,
            expected_digest,
        }
    }
}

#[derive(Debug, Clone)]
struct SourceVerification {
    source_offset: u64,
    length: usize,
    expected_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy)]
struct SourceScatter {
    source_offset: u64,
    destination: BufferKind,
    destination_offset: usize,
    length: usize,
    verification_index: usize,
}

#[derive(Debug, Clone)]
struct AuthenticatedSourceContract {
    exact_length: u64,
    verifications: Vec<SourceVerification>,
}

#[derive(Debug, Clone, Copy)]
struct DeferredRangeContract {
    exact_length: u64,
    expected_identity: Option<DeferredSourceIdentity>,
}

#[derive(Debug)]
struct Source {
    path: PathBuf,
    file: Option<File>,
    /// A deferred manifest can retain its first validated descriptor for the
    /// entire immutable plan lifetime. `OnceLock<Result<_>>` also pins a
    /// first-use failure, so a path replacement can never turn a rejected
    /// source into a different inode inside the published plan.
    persistent_file: Option<OnceLock<Result<File>>>,
    length: u64,
    expected_identity: Option<DeferredSourceIdentity>,
    verifications: Box<[SourceVerification]>,
    /// Disjoint authenticated source ranges copied directly into the private
    /// arena while their enclosing verification range is hashed.  Keeping
    /// these on the source lets one worker make one physical pass over every
    /// authenticated byte instead of hashing the source and then rereading
    /// the retained ranges.
    scatter_extents: Vec<SourceScatter>,
    cache_policy: CachePolicy,
}

#[derive(Debug)]
struct Sources {
    values: Vec<Source>,
    persistent_count: usize,
    // The reservation must outlive every Batch clone of this source set, not
    // merely the ReadPlan that admitted it.
    _descriptor_reservation: DescriptorReservation,
}

#[derive(Debug, Clone, Copy)]
enum JobSource {
    File { source: usize, source_offset: u64 },
    Vectored { source: usize, scatter: usize },
    AuthenticatedScatter { source: usize },
    DeferredCatalog { source: u32 },
    Zero,
}

#[derive(Debug)]
struct VectoredRead {
    source_offset: u64,
    destinations: Box<[VectoredDestination]>,
}

#[derive(Debug, Clone, Copy)]
struct ReadJob {
    source: JobSource,
    destination: BufferKind,
    destination_offset: usize,
    length: usize,
    expected_digest: Option<[u8; 32]>,
    verification_index: Option<usize>,
}

struct InlineReadJobs {
    sources: [u32; MAX_INLINE_DEFERRED_FILES],
    len: usize,
    destination: BufferKind,
    source_length: usize,
}

enum BatchJobs {
    Shared(Arc<Vec<ReadJob>>),
    Inline(InlineReadJobs),
}

impl BatchJobs {
    fn get(&self, index: usize) -> Option<ReadJob> {
        match self {
            Self::Shared(jobs) => jobs.get(index).copied(),
            Self::Inline(jobs) if index < jobs.len => Some(ReadJob {
                source: JobSource::DeferredCatalog {
                    source: jobs.sources[index],
                },
                destination: jobs.destination,
                destination_offset: index * jobs.source_length,
                length: jobs.source_length,
                expected_digest: None,
                verification_index: None,
            }),
            Self::Inline(_) => None,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Shared(jobs) => jobs.len(),
            Self::Inline(jobs) => jobs.len,
        }
    }
}

enum BatchSources {
    Plan {
        sources: Arc<Sources>,
        /// One live descriptor per deferred source for the lifetime of this
        /// batch. Large whole-file tensors are split into several parallel
        /// positional jobs; sharing the validated descriptor removes repeated
        /// open/stat/cache-policy calls and guarantees every chunk observes
        /// the same inode. Persistent plans leave this absent.
        deferred_files: Option<Box<[OnceLock<Result<File>>]>>,
    },
    DeferredCatalog(Arc<DeferredExactCatalogInner>),
}

#[derive(Debug)]
pub struct ReadPlan {
    sources: Arc<Sources>,
    jobs: Arc<Vec<ReadJob>>,
    vectored_reads: Arc<Vec<VectoredRead>>,
    verified_extents: Arc<Vec<AtomicBool>>,
    buffer_lengths: BufferLengths,
    logical_bytes: u64,
}

#[derive(Debug, Clone)]
enum SourceAdmission {
    Persistent,
    DeferredExact(u64),
    /// Every read extent names one complete source file.  Its declared extent
    /// length becomes that source's exact-size contract, but the file is not
    /// opened until a reader worker executes the job.  This is the loose-spine
    /// compatibility path: thousands of immutable tensor components can be
    /// compiled once without retaining thousands of descriptors or serially
    /// opening/statting them on the inference thread.
    DeferredManifest,
    /// Same exact whole-file contract as `DeferredManifest`, with a bounded
    /// all-or-nothing descriptor reservation made at plan construction. Files
    /// are still opened lazily on reader workers and validated with
    /// O_NOFOLLOW against the descriptor actually used for every pread.
    DeferredManifestPersistent,
    /// Partial ranges from heterogeneous exact-length files. Unlike the
    /// manifest variants, extents need not cover a complete source. Unlike an
    /// authenticated gather, no source bytes are reread solely to hash gaps;
    /// callers must validate their assembled format before publication.
    DeferredRanges(Arc<HashMap<PathBuf, DeferredRangeContract>>),
    /// Sources are opened on workers; every declared verification range is
    /// authenticated on its live descriptor while its disjoint retained
    /// extents are scattered into the private destination arena.
    DeferredAuthenticated(Arc<HashMap<PathBuf, AuthenticatedSourceContract>>),
}

impl ReadPlan {
    pub fn open(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::Persistent,
        )
    }

    /// Compile a plan without opening its source files on the caller thread.
    /// Each worker opens one source with `O_NOFOLLOW`, validates the exact
    /// regular-file length on that descriptor, reads it positionally, applies
    /// cache policy, and closes it. This is intended for ephemeral routed
    /// expert unions where retaining hundreds of descriptors would exceed the
    /// process budget and serial open/stat calls would extend token latency.
    pub fn open_deferred_exact(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
        exact_source_length: u64,
    ) -> Result<Self> {
        if exact_source_length == 0 {
            return Err(DeltafinError::new(
                "deferred read sources need a positive exact length",
            ));
        }
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredExact(exact_source_length),
        )
    }

    /// Compile a heterogeneous whole-file manifest without opening sources.
    ///
    /// Unlike [`Self::open_deferred_exact`], each source may have a different
    /// exact length. Every non-empty read must cover its complete source from
    /// offset zero. Workers later open with `O_NOFOLLOW|O_CLOEXEC`, validate a
    /// live regular-file descriptor against the recorded length, read it, and
    /// close it. This makes the unpacked model layout safe under low descriptor
    /// limits while keeping file validation on the descriptor actually read.
    pub fn open_deferred_manifest(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredManifest,
        )
    }

    pub fn open_persistent_deferred_manifest(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredManifestPersistent,
        )
    }

    /// Compile a heterogeneous partial-range gather without opening sources.
    ///
    /// Every source must have exactly one explicit length contract. The
    /// worker-owned descriptor is shared by all ranges from that source in one
    /// batch, then closed after completion. This keeps hot gathers free of
    /// redundant whole-file authentication reads while retaining the same
    /// no-follow, regular-file and exact-length boundary as deferred raw-v1.
    pub fn open_deferred_ranges(
        extents: impl IntoIterator<Item = Extent>,
        sources: impl IntoIterator<Item = DeferredSourceLength>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let mut contracts = HashMap::new();
        for source in sources {
            if source.exact_source_length == 0 {
                return Err(DeltafinError::new(
                    "deferred range source needs a positive exact length",
                ));
            }
            if contracts.len() == MAX_DEFERRED_AUTHENTICATED_SOURCES
                && !contracts.contains_key(&source.path)
            {
                return Err(DeltafinError::new(format!(
                    "deferred range plan exceeds the {MAX_DEFERRED_AUTHENTICATED_SOURCES}-source safety limit"
                )));
            }
            let contract = DeferredRangeContract {
                exact_length: source.exact_source_length,
                expected_identity: source.expected_identity,
            };
            if contracts.insert(source.path, contract).is_some() {
                return Err(DeltafinError::new(
                    "deferred range source length is declared more than once",
                ));
            }
        }
        if contracts.is_empty() {
            return Err(DeltafinError::new(
                "deferred range plan needs at least one source contract",
            ));
        }
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredRanges(Arc::new(contracts)),
        )
    }

    /// Compile a deferred, exact-length gather whose source identity is
    /// authenticated on the same live descriptor used for every copied byte.
    ///
    /// Each source named by an extent must have at least one verification.
    /// Multiple verifications may qualify disjoint records in one larger
    /// source file. Authentication is performed once per admitted batch
    /// descriptor, and an identity recheck after the one-pass authenticated
    /// scatter fails closed if the source changes while that descriptor is in
    /// use.
    pub fn open_deferred_authenticated(
        extents: impl IntoIterator<Item = Extent>,
        verifications: impl IntoIterator<Item = DeferredSourceVerification>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let mut contracts: HashMap<PathBuf, AuthenticatedSourceContract> = HashMap::new();
        let mut verification_count = 0_usize;
        for verification in verifications {
            verification_count = verification_count
                .checked_add(1)
                .ok_or_else(|| DeltafinError::new("authenticated range count overflows usize"))?;
            if verification_count > MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS {
                return Err(DeltafinError::new(format!(
                    "authenticated deferred plan exceeds the {MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS}-range safety limit"
                )));
            }
            if verification.exact_source_length == 0 || verification.length == 0 {
                return Err(DeltafinError::new(
                    "authenticated deferred sources and ranges must be non-empty",
                ));
            }
            let end = verification
                .source_offset
                .checked_add(verification.length as u64)
                .ok_or_else(|| DeltafinError::new("authenticated source range overflows u64"))?;
            if end > verification.exact_source_length {
                return Err(DeltafinError::new(format!(
                    "authenticated range {}..{} exceeds exact source length {} for {}",
                    verification.source_offset,
                    end,
                    verification.exact_source_length,
                    verification.path.display(),
                )));
            }
            if !contracts.contains_key(&verification.path)
                && contracts.len() == MAX_DEFERRED_AUTHENTICATED_SOURCES
            {
                return Err(DeltafinError::new(format!(
                    "authenticated deferred plan exceeds the {MAX_DEFERRED_AUTHENTICATED_SOURCES}-source safety limit"
                )));
            }
            let contract =
                contracts
                    .entry(verification.path)
                    .or_insert_with(|| AuthenticatedSourceContract {
                        exact_length: verification.exact_source_length,
                        verifications: Vec::new(),
                    });
            if contract.exact_length != verification.exact_source_length {
                return Err(DeltafinError::new(
                    "authenticated source has conflicting exact-length contracts",
                ));
            }
            if contract.verifications.iter().any(|existing| {
                existing.source_offset == verification.source_offset
                    && existing.length == verification.length
            }) {
                return Err(DeltafinError::new(
                    "authenticated source range is declared more than once",
                ));
            }
            contract.verifications.push(SourceVerification {
                source_offset: verification.source_offset,
                length: verification.length,
                expected_digest: verification.expected_digest,
            });
        }
        if contracts.is_empty() {
            return Err(DeltafinError::new(
                "authenticated deferred read needs at least one source contract",
            ));
        }
        for contract in contracts.values_mut() {
            contract
                .verifications
                .sort_by_key(|verification| verification.source_offset);
            for adjacent in contract.verifications.windows(2) {
                let previous = &adjacent[0];
                let next = &adjacent[1];
                let previous_end = previous
                    .source_offset
                    .checked_add(previous.length as u64)
                    .ok_or_else(|| {
                        DeltafinError::new("authenticated source range overflows u64")
                    })?;
                if next.source_offset < previous_end {
                    return Err(DeltafinError::new(
                        "authenticated source ranges overlap; every gathered byte must have one unambiguous digest contract",
                    ));
                }
            }
        }
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredAuthenticated(Arc::new(contracts)),
        )
    }

    fn open_with_admission(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
        admission: SourceAdmission,
    ) -> Result<Self> {
        let extents: Vec<Extent> = extents.into_iter().collect();
        validate_destinations(&extents, buffer_lengths)?;

        let unique_source_count = extents
            .iter()
            .filter_map(|extent| match extent {
                Extent::Read { path, length, .. } if *length != 0 => Some(path),
                Extent::ReadVectored {
                    path, destinations, ..
                } if !destinations.is_empty() => Some(path),
                _ => None,
            })
            .collect::<HashSet<_>>()
            .len();
        if matches!(
            &admission,
            SourceAdmission::DeferredManifest | SourceAdmission::DeferredManifestPersistent
        ) && unique_source_count > MAX_DEFERRED_MANIFEST_SOURCES
        {
            return Err(DeltafinError::new(format!(
                "deferred manifest has {unique_source_count} sources; bounded maximum is {MAX_DEFERRED_MANIFEST_SOURCES}"
            )));
        }
        let persistent_count = match &admission {
            SourceAdmission::Persistent | SourceAdmission::DeferredManifestPersistent => {
                unique_source_count
            }
            SourceAdmission::DeferredExact(_)
            | SourceAdmission::DeferredManifest
            | SourceAdmission::DeferredRanges(_)
            | SourceAdmission::DeferredAuthenticated(_) => 0,
        };
        let descriptor_reservation = process_descriptor_budget().reserve(persistent_count)?;

        let mut sources = Vec::with_capacity(unique_source_count);
        let mut source_indices = HashMap::with_capacity(unique_source_count);
        let mut jobs = Vec::new();
        let mut vectored_reads = Vec::new();
        let mut logical_bytes = 0_u64;

        for extent in extents {
            if let Extent::ReadVectored {
                path,
                source_offset,
                destinations,
                expected_digest,
            } = extent
            {
                if destinations.is_empty()
                    || destinations
                        .iter()
                        .any(|destination| destination.length == 0)
                {
                    return Err(DeltafinError::new(
                        "a vectored read needs only non-empty destinations",
                    ));
                }
                let SourceAdmission::DeferredRanges(contracts) = &admission else {
                    return Err(DeltafinError::new(
                        "vectored reads require deferred-range source contracts",
                    ));
                };
                let extent_length =
                    destinations
                        .iter()
                        .try_fold(0_usize, |total, destination| {
                            total.checked_add(destination.length).ok_or_else(|| {
                                DeltafinError::new("vectored read length overflows usize")
                            })
                        })?;
                if extent_length == 0 {
                    continue;
                }
                let source_end = source_offset
                    .checked_add(extent_length as u64)
                    .ok_or_else(|| DeltafinError::new("vectored source range overflows u64"))?;
                let contract = contracts.get(&path).ok_or_else(|| {
                    DeltafinError::new(format!(
                        "deferred range source {} has no exact-length contract",
                        path.display()
                    ))
                })?;
                if source_end > contract.exact_length {
                    return Err(DeltafinError::new(format!(
                        "vectored source extent {}..{} exceeds {}-byte file {}",
                        source_offset,
                        source_end,
                        contract.exact_length,
                        path.display()
                    )));
                }
                logical_bytes = logical_bytes
                    .checked_add(extent_length as u64)
                    .ok_or_else(|| DeltafinError::new("read-plan byte count overflows u64"))?;
                let source = if let Some(source) = source_indices.get(&path) {
                    *source
                } else {
                    let index = sources.len();
                    sources.push(Source {
                        path: path.clone(),
                        file: None,
                        persistent_file: None,
                        length: contract.exact_length,
                        expected_identity: contract.expected_identity,
                        verifications: Vec::new().into_boxed_slice(),
                        scatter_extents: Vec::new(),
                        cache_policy,
                    });
                    source_indices.insert(path, index);
                    index
                };
                let first = *destinations
                    .first()
                    .expect("vectored constructor and plan validation reject empty targets");
                let scatter = vectored_reads.len();
                vectored_reads.push(VectoredRead {
                    source_offset,
                    destinations,
                });
                jobs.push(ReadJob {
                    source: JobSource::Vectored { source, scatter },
                    destination: first.destination,
                    destination_offset: first.destination_offset,
                    length: extent_length,
                    expected_digest,
                    verification_index: None,
                });
                if jobs.len() > MAX_PLAN_JOBS {
                    return Err(DeltafinError::new(format!(
                        "read plan exceeds the {MAX_PLAN_JOBS}-job safety limit"
                    )));
                }
                continue;
            }
            if matches!(
                &extent,
                Extent::Read { length: 0, .. } | Extent::Zero { length: 0, .. }
            ) {
                continue;
            }
            let (source, destination, destination_offset, extent_length, expected_digest) =
                match extent {
                    Extent::Read {
                        path,
                        source_offset,
                        destination,
                        destination_offset,
                        length,
                        expected_digest,
                    } => {
                        let source_end = source_offset
                            .checked_add(length as u64)
                            .ok_or_else(|| DeltafinError::new("source extent overflows u64"))?;
                        if let SourceAdmission::DeferredAuthenticated(contracts) = &admission {
                            if expected_digest.is_some() {
                                return Err(DeltafinError::new(
                                    "authenticated deferred gathers use source-range digests and may not also declare a per-extent digest",
                                ));
                            }
                            let contract = contracts.get(&path).ok_or_else(|| {
                                DeltafinError::new(format!(
                                    "deferred gather source {} has no authentication contract",
                                    path.display()
                                ))
                            })?;
                            let covered = contract.verifications.iter().any(|verification| {
                                let verification_end =
                                    verification.source_offset + verification.length as u64;
                                verification.source_offset <= source_offset
                                    && source_end <= verification_end
                            });
                            if !covered {
                                return Err(DeltafinError::new(format!(
                                    "deferred gather extent {}..{} from {} is outside every authenticated range",
                                    source_offset,
                                    source_end,
                                    path.display()
                                )));
                            }
                        }
                        if let SourceAdmission::DeferredRanges(contracts) = &admission
                            && !contracts.contains_key(&path)
                        {
                            return Err(DeltafinError::new(format!(
                                "deferred range source {} has no exact-length contract",
                                path.display()
                            )));
                        }
                        if matches!(
                            &admission,
                            SourceAdmission::DeferredManifest
                                | SourceAdmission::DeferredManifestPersistent
                        ) && source_offset != 0
                        {
                            return Err(DeltafinError::new(format!(
                                "deferred-manifest source {} must be read whole from offset zero",
                                path.display()
                            )));
                        }
                        logical_bytes =
                            logical_bytes.checked_add(length as u64).ok_or_else(|| {
                                DeltafinError::new("read-plan byte count overflows u64")
                            })?;
                        let source = if let Some(source) = source_indices.get(&path) {
                            *source
                        } else {
                            let (file, file_size, expected_identity, verifications) =
                                match &admission {
                                    SourceAdmission::Persistent => {
                                        let file = File::open(&path).map_err(|error| {
                                    if matches!(error.raw_os_error(), Some(23 | 24)) {
                                        DeltafinError::new(format!(
                                            "descriptor headroom was consumed by another subsystem while opening {}; close unrelated files or raise the descriptor limit",
                                            path.display()
                                        ))
                                    } else {
                                        io_error("open", &path, error)
                                    }
                                })?;
                                        let file_size = file
                                            .metadata()
                                            .map_err(|error| io_error("stat", &path, error))?
                                            .len();
                                        configure_cache_policy(&file, &path, cache_policy)?;
                                        (Some(file), file_size, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredExact(length) => {
                                        (None, *length, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredManifest => {
                                        (None, source_end, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredManifestPersistent => {
                                        (None, source_end, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredRanges(contracts) => {
                                        let contract = contracts.get(&path).ok_or_else(|| {
                                        DeltafinError::new(format!(
                                            "deferred range source {} has no exact-length contract",
                                            path.display()
                                        ))
                                    })?;
                                        (
                                            None,
                                            contract.exact_length,
                                            contract.expected_identity,
                                            Vec::new().into_boxed_slice(),
                                        )
                                    }
                                    SourceAdmission::DeferredAuthenticated(contracts) => {
                                        let contract = contracts.get(&path).ok_or_else(|| {
                                        DeltafinError::new(format!(
                                            "deferred gather source {} has no authentication contract",
                                            path.display()
                                        ))
                                    })?;
                                        (
                                            None,
                                            contract.exact_length,
                                            None,
                                            contract.verifications.clone().into_boxed_slice(),
                                        )
                                    }
                                };
                            let index = sources.len();
                            sources.push(Source {
                                path: path.clone(),
                                file,
                                persistent_file: matches!(
                                    &admission,
                                    SourceAdmission::DeferredManifestPersistent
                                )
                                .then(OnceLock::new),
                                length: file_size,
                                expected_identity,
                                verifications,
                                scatter_extents: Vec::new(),
                                cache_policy,
                            });
                            source_indices.insert(path.clone(), index);
                            index
                        };
                        let file_size = sources[source].length;
                        if matches!(
                            &admission,
                            SourceAdmission::DeferredManifest
                                | SourceAdmission::DeferredManifestPersistent
                        ) && file_size != source_end
                        {
                            return Err(DeltafinError::new(format!(
                                "deferred-manifest source {} has conflicting whole-file lengths {} and {}",
                                path.display(),
                                file_size,
                                source_end
                            )));
                        }
                        if source_end > file_size {
                            return Err(DeltafinError::new(format!(
                                "source extent {}..{} exceeds {}-byte file {}",
                                source_offset,
                                source_end,
                                file_size,
                                path.display()
                            )));
                        }
                        if matches!(&admission, SourceAdmission::DeferredAuthenticated(_)) {
                            let verification_index = sources[source]
                                .verifications
                                .iter()
                                .position(|verification| {
                                    let verification_end = verification.source_offset
                                        + verification.length as u64;
                                    verification.source_offset <= source_offset
                                        && source_end <= verification_end
                                })
                                .ok_or_else(|| {
                                    DeltafinError::new(format!(
                                        "deferred gather extent {}..{} from {} lost its authentication contract",
                                        source_offset,
                                        source_end,
                                        path.display()
                                    ))
                                })?;
                            sources[source].scatter_extents.push(SourceScatter {
                                source_offset,
                                destination,
                                destination_offset,
                                length,
                                verification_index,
                            });
                            // Authenticated gathers become one source-owned
                            // job below.  Emitting ordinary extent jobs here
                            // would authenticate the source and then reread
                            // every retained byte a second time.
                            continue;
                        }
                        (
                            JobSource::File {
                                source,
                                source_offset,
                            },
                            destination,
                            destination_offset,
                            length,
                            expected_digest,
                        )
                    }
                    Extent::Zero {
                        destination,
                        destination_offset,
                        length,
                    } => (
                        JobSource::Zero,
                        destination,
                        destination_offset,
                        length,
                        None,
                    ),
                    Extent::ReadVectored { .. } => {
                        unreachable!("vectored extents are compiled before ordinary read jobs")
                    }
                };

            let chunk = if expected_digest.is_some() || chunk_bytes == 0 {
                extent_length
            } else {
                chunk_bytes
            };
            let mut consumed = 0_usize;
            while consumed < extent_length {
                let length = chunk.min(extent_length - consumed);
                let source = match source {
                    JobSource::File {
                        source,
                        source_offset,
                    } => JobSource::File {
                        source,
                        source_offset: source_offset + consumed as u64,
                    },
                    JobSource::Vectored { source, scatter } => {
                        JobSource::Vectored { source, scatter }
                    }
                    JobSource::AuthenticatedScatter { source } => {
                        JobSource::AuthenticatedScatter { source }
                    }
                    JobSource::DeferredCatalog { source } => JobSource::DeferredCatalog { source },
                    JobSource::Zero => JobSource::Zero,
                };
                jobs.push(ReadJob {
                    source,
                    destination,
                    destination_offset: destination_offset + consumed,
                    length,
                    expected_digest,
                    verification_index: None,
                });
                if jobs.len() > MAX_PLAN_JOBS {
                    return Err(DeltafinError::new(format!(
                        "read plan exceeds the {MAX_PLAN_JOBS}-job safety limit; increase chunk size"
                    )));
                }
                consumed += length;
            }
        }

        if let SourceAdmission::DeferredRanges(contracts) = &admission
            && sources.len() != contracts.len()
        {
            return Err(DeltafinError::new(
                "deferred range plan contains an unused source contract",
            ));
        }

        if let SourceAdmission::DeferredAuthenticated(contracts) = &admission {
            if sources.len() != contracts.len() {
                return Err(DeltafinError::new(
                    "authenticated deferred plan contains an unused source contract",
                ));
            }
            for (source_index, source) in sources.iter_mut().enumerate() {
                source
                    .scatter_extents
                    .sort_by_key(|extent| extent.source_offset);
                for adjacent in source.scatter_extents.windows(2) {
                    let previous_end = adjacent[0]
                        .source_offset
                        .checked_add(adjacent[0].length as u64)
                        .ok_or_else(|| {
                            DeltafinError::new("authenticated gather range overflows u64")
                        })?;
                    if adjacent[1].source_offset < previous_end {
                        return Err(DeltafinError::new(format!(
                            "authenticated gather ranges overlap in source {}",
                            source.path.display()
                        )));
                    }
                }
                let physical_bytes =
                    source
                        .verifications
                        .iter()
                        .try_fold(0_usize, |total, verification| {
                            total.checked_add(verification.length).ok_or_else(|| {
                                DeltafinError::new(
                                    "authenticated source verification byte count overflows usize",
                                )
                            })
                        })?;
                let first = source.scatter_extents.first().ok_or_else(|| {
                    DeltafinError::new(format!(
                        "authenticated source {} has no gathered extent",
                        source.path.display()
                    ))
                })?;
                jobs.push(ReadJob {
                    source: JobSource::AuthenticatedScatter {
                        source: source_index,
                    },
                    // The authenticated-scatter worker uses the source-owned
                    // destination list. These fields retain ReadJob's compact
                    // common scheduling shape and provide a deterministic
                    // diagnostic anchor only.
                    destination: first.destination,
                    destination_offset: first.destination_offset,
                    length: physical_bytes,
                    expected_digest: None,
                    verification_index: None,
                });
            }
            if jobs.len() > MAX_PLAN_JOBS {
                return Err(DeltafinError::new(format!(
                    "read plan exceeds the {MAX_PLAN_JOBS}-job safety limit"
                )));
            }
        }

        // Match the established longest-processing-time-first policy. Fixed
        // workers pull through one atomic index, so the short chunks naturally
        // collect at the tail instead of stranding one worker on a large read.
        jobs.sort_by_key(|job| Reverse(job.length));
        let mut verified_extents = Vec::new();
        for job in &mut jobs {
            if job.expected_digest.is_some() && !matches!(job.source, JobSource::Vectored { .. }) {
                job.verification_index = Some(verified_extents.len());
                verified_extents.push(AtomicBool::new(false));
            }
        }
        Ok(Self {
            sources: Arc::new(Sources {
                values: sources,
                persistent_count,
                _descriptor_reservation: descriptor_reservation,
            }),
            jobs: Arc::new(jobs),
            vectored_reads: Arc::new(vectored_reads),
            verified_extents: Arc::new(verified_extents),
            buffer_lengths,
            logical_bytes,
        })
    }

    pub fn jobs(&self) -> usize {
        self.jobs.len()
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn source_count(&self) -> usize {
        self.sources.values.len()
    }

    pub fn persistent_source_count(&self) -> usize {
        self.sources.persistent_count
    }

    /// Number of lazy persistent descriptors already opened by this plan.
    /// This is observation only; it never opens or stats a source.
    pub fn opened_persistent_source_count(&self) -> usize {
        self.sources
            .values
            .iter()
            .filter(|source| {
                source
                    .persistent_file
                    .as_ref()
                    .is_some_and(|slot| matches!(slot.get(), Some(Ok(_))))
            })
            .count()
    }

    /// Uniform OS cache-admission policy carried by this immutable plan.
    /// Plans built only from zero-fill extents have no source policy.
    pub fn cache_policy(&self) -> Option<CachePolicy> {
        let first = self.sources.values.first()?.cache_policy;
        self.sources
            .values
            .iter()
            .all(|source| source.cache_policy == first)
            .then_some(first)
    }

    pub fn buffer_len(&self, kind: BufferKind) -> usize {
        self.buffer_lengths.get(kind)
    }

    /// Require every source contract to have one canonical length.
    ///
    /// Some raw cache formats use the file itself as their versioned storage
    /// envelope. For those formats, accepting an oversized object just because
    /// all requested extents fit would weaken the format contract and differ
    /// from the established loader's exact-size admission rule.
    pub fn require_all_sources_exact_length(&self, expected: u64) -> Result<()> {
        for source in &self.sources.values {
            if source.length != expected {
                return Err(DeltafinError::new(format!(
                    "source {} is {} bytes; canonical length is {expected}",
                    source.path.display(),
                    source.length,
                )));
            }
        }
        Ok(())
    }
}

fn validate_destinations(extents: &[Extent], declared: BufferLengths) -> Result<()> {
    let mut ranges: Vec<(BufferKind, usize, usize)> = Vec::new();
    for extent in extents {
        let destinations: Box<dyn Iterator<Item = VectoredDestination> + '_> = match extent {
            Extent::Read {
                destination,
                destination_offset,
                length,
                ..
            }
            | Extent::Zero {
                destination,
                destination_offset,
                length,
            } => Box::new(std::iter::once(VectoredDestination::new(
                *destination,
                *destination_offset,
                *length,
            ))),
            Extent::ReadVectored { destinations, .. } => Box::new(destinations.iter().copied()),
        };
        for range in destinations.filter(|range| range.length != 0) {
            let end = range
                .destination_offset
                .checked_add(range.length)
                .ok_or_else(|| DeltafinError::new("destination extent overflows usize"))?;
            if end > declared.get(range.destination) {
                return Err(DeltafinError::new(format!(
                    "{:?} destination extent {}..{} exceeds declared length {}",
                    range.destination,
                    range.destination_offset,
                    end,
                    declared.get(range.destination)
                )));
            }
            ranges.push((range.destination, range.destination_offset, end));
        }
    }
    ranges.sort_unstable();
    for kind in BufferKind::ALL {
        let mut covered = 0_usize;
        for &(_, start, end) in ranges
            .iter()
            .filter(|(range_kind, _, _)| *range_kind == kind)
        {
            if start < covered {
                return Err(DeltafinError::new(format!(
                    "overlapping {:?} destination ranges at {}",
                    kind, start
                )));
            }
            if start > covered {
                return Err(DeltafinError::new(format!(
                    "uncovered {:?} destination range {}..{}; declare padding with Extent::zero",
                    kind, covered, start
                )));
            }
            covered = end;
        }
        let expected = declared.get(kind);
        if covered != expected {
            return Err(DeltafinError::new(format!(
                "uncovered {:?} destination range {}..{}; declared length is {}",
                kind, covered, expected, expected
            )));
        }
    }
    Ok(())
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

struct AlignedBuffer {
    pointer: NonNull<u8>,
    capacity: usize,
    layout: Layout,
}

impl AlignedBuffer {
    fn new(capacity: usize) -> Result<Self> {
        let allocated_len = aligned_allocation_len(capacity)?;
        let layout = Layout::from_size_align(allocated_len, BUFFER_ALIGNMENT)
            .map_err(|_| DeltafinError::new("invalid aligned-buffer layout"))?;
        // SAFETY: `layout` is non-zero and valid. Ownership is retained by
        // this value and released exactly once in `Drop` with the same layout.
        let pointer = unsafe { alloc_zeroed(layout) };
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| DeltafinError::new("aligned-buffer allocation failed"))?;
        Ok(Self {
            pointer,
            capacity,
            layout,
        })
    }

    fn as_slice(&self, logical_len: usize) -> &[u8] {
        debug_assert!(logical_len <= self.capacity);
        // SAFETY: the allocation is live for `self`; immutable access is only
        // exposed after a batch reaches completion.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), logical_len) }
    }

    const fn allocation_len(&self) -> usize {
        self.layout.size()
    }

    fn pointer_at(&self, offset: usize, length: usize) -> *mut u8 {
        debug_assert!(offset <= self.capacity);
        debug_assert!(length <= self.capacity - offset);
        // The caller may turn this into a mutable slice only for a prevalidated
        // destination range while the batch is unpublished.
        self.pointer.as_ptr().wrapping_add(offset)
    }
}

fn aligned_allocation_len(capacity: usize) -> Result<usize> {
    capacity
        .max(1)
        .checked_add(BUFFER_ALIGNMENT - 1)
        .map(|length| length / BUFFER_ALIGNMENT * BUFFER_ALIGNMENT)
        .ok_or_else(|| DeltafinError::new("aligned-buffer size overflows usize"))
}

fn shared_allocation_len(capacities: BufferLengths) -> Result<u64> {
    BufferKind::ALL.iter().try_fold(0_u64, |total, &kind| {
        let bytes = u64::try_from(aligned_allocation_len(capacities.get(kind))?)
            .map_err(|_| DeltafinError::new("aligned-buffer allocation exceeds u64"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| DeltafinError::new("shared-buffer allocation overflows u64"))
    })
}

// SAFETY: mutation is private to `Batch::run_job`, all mutable ranges are
// proven disjoint before publication, and readers only receive immutable views
// after every job has completed.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `pointer` came from `alloc_zeroed(self.layout)` and has not
        // been deallocated or transferred.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

struct SharedBuffers {
    values: [AlignedBuffer; BufferKind::COUNT],
}

impl SharedBuffers {
    fn new(capacities: BufferLengths) -> Result<Self> {
        let capacities = capacities.as_array();
        Ok(Self {
            values: [
                AlignedBuffer::new(capacities[0])?,
                AlignedBuffer::new(capacities[1])?,
                AlignedBuffer::new(capacities[2])?,
            ],
        })
    }

    fn get(&self, kind: BufferKind) -> &AlignedBuffer {
        &self.values[kind.index()]
    }
}

struct ArenaSlot {
    buffers: Option<Arc<SharedBuffers>>,
    capacities: BufferLengths,
    in_use: bool,
}

pub(crate) type BufferRetireHook = Arc<dyn Fn() -> Result<()> + Send + Sync>;

fn invoke_buffer_retire_hook(hook: &BufferRetireHook) -> Result<()> {
    match catch_unwind(AssertUnwindSafe(|| hook())) {
        Ok(result) => result,
        Err(_) => Err(DeltafinError::new(
            "storage arena retirement hook panicked before releasing an externally aliased allocation",
        )),
    }
}

struct BufferArena {
    inner: Mutex<Vec<ArenaSlot>>,
    available: Condvar,
    retire_hook: Option<BufferRetireHook>,
}

impl BufferArena {
    fn new(slots: usize) -> Result<Arc<Self>> {
        Self::new_with_retire_hook(slots, None)
    }

    fn new_with_retire_hook(
        slots: usize,
        retire_hook: Option<BufferRetireHook>,
    ) -> Result<Arc<Self>> {
        if slots == 0 {
            return Err(DeltafinError::new(
                "storage buffer arena needs at least one slot",
            ));
        }
        Ok(Arc::new(Self {
            inner: Mutex::new(
                (0..slots)
                    .map(|_| ArenaSlot {
                        buffers: None,
                        capacities: BufferLengths::default(),
                        in_use: false,
                    })
                    .collect(),
            ),
            available: Condvar::new(),
            retire_hook,
        }))
    }

    fn acquire(
        self: &Arc<Self>,
        lengths: BufferLengths,
        wait: bool,
        priority: ReadPriority,
    ) -> Result<Option<Arc<BufferLeaseInner>>> {
        let (slot_index, target_capacities, needs_allocation, retired_capacities, retired_buffers) = {
            let mut slots = self.inner.lock().unwrap();
            loop {
                let free_slots = slots.iter().filter(|slot| !slot.in_use).count();
                // With more than one slot, prefetch may not consume the final
                // slot: a newly routed demand read must remain admissible even
                // if a speculative result is waiting for its CPU consumer.
                let can_admit =
                    priority == ReadPriority::Demand || slots.len() == 1 || free_slots > 1;
                if can_admit {
                    if let Some(index) = slots
                        .iter()
                        .enumerate()
                        .filter(|(_, slot)| !slot.in_use)
                        .min_by_key(|(index, slot)| arena_slot_cost(slot, lengths, *index))
                        .map(|(index, _)| index)
                    {
                        let slot = &mut slots[index];
                        slot.in_use = true;
                        let old_capacities = slot.capacities;
                        let needs_allocation = slot.buffers.is_none()
                            || BufferKind::ALL
                                .iter()
                                .any(|&kind| old_capacities.get(kind) < lengths.get(kind));
                        let target_capacities = old_capacities.max(lengths);
                        // A free slot has no live CPU lease. Retire its old slab
                        // now so a growth allocation does not temporarily hold
                        // old+new multi-gigabyte buffers at once.
                        let retired_buffers = if needs_allocation {
                            slot.capacities = BufferLengths::default();
                            slot.buffers.take()
                        } else {
                            None
                        };
                        break (
                            index,
                            target_capacities,
                            needs_allocation,
                            old_capacities,
                            retired_buffers,
                        );
                    }
                }
                if !wait {
                    return Ok(None);
                }
                slots = self.available.wait(slots).unwrap();
            }
        };

        let mut retired_buffers = retired_buffers;
        if retired_buffers.is_some()
            && let Some(retire_hook) = self.retire_hook.as_ref()
            && let Err(error) = invoke_buffer_retire_hook(retire_hook)
        {
            // The old allocation remains owned until the external cache has
            // proven that no no-copy wrapper aliases it. A failed flush must
            // therefore restore the exact free slot and reject this growth.
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            debug_assert!(slot.in_use);
            debug_assert!(slot.buffers.is_none());
            slot.buffers = retired_buffers.take();
            slot.capacities = retired_capacities;
            slot.in_use = false;
            self.available.notify_all();
            return Err(DeltafinError::new(format!(
                "storage arena refused to retire an externally aliased allocation: {error}"
            )));
        }
        // Cache retirement has completed before the allocation's final arena
        // Arc can be released. Stable, fitting slabs never invoke the hook.
        drop(retired_buffers);
        let replacement = if needs_allocation {
            match SharedBuffers::new(target_capacities) {
                Ok(buffers) => Some((Arc::new(buffers), target_capacities)),
                Err(error) => {
                    self.release(slot_index);
                    return Err(error);
                }
            }
        } else {
            None
        };

        let buffers = {
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            if let Some((buffers, capacities)) = replacement {
                slot.buffers = Some(buffers);
                slot.capacities = capacities;
            }
            Arc::clone(
                slot.buffers
                    .as_ref()
                    .expect("reserved arena slot must contain buffers"),
            )
        };

        Ok(Some(Arc::new(BufferLeaseInner {
            arena: Arc::downgrade(self),
            slot_index,
            buffers: Some(buffers),
            lengths,
        })))
    }

    fn release(&self, slot_index: usize) {
        let mut slots = self.inner.lock().unwrap();
        let slot = slots
            .get_mut(slot_index)
            .expect("buffer lease refers to an unknown arena slot");
        debug_assert!(slot.in_use);
        slot.in_use = false;
        // Demand and prefetch use different admission predicates; wake all so
        // an ineligible prefetch waiter cannot strand an eligible demand one.
        self.available.notify_all();
    }

    /// Complete allocation which a new request may add at its next arena
    /// admission boundary. A fitting free slot needs no allocation. Growth
    /// charges the complete replacement slab, not merely its delta, because a
    /// platform allocator may temporarily retain pages from the retired slab.
    /// When every slot is busy its future capacity is deliberately treated as
    /// unknown and the complete requested slab is charged conservatively.
    fn replacement_admission_bytes(&self, lengths: BufferLengths) -> Result<u64> {
        let slots = self.inner.lock().unwrap();
        if slots.iter().any(|slot| {
            !slot.in_use
                && slot.buffers.is_some()
                && BufferKind::ALL
                    .iter()
                    .all(|&kind| slot.capacities.get(kind) >= lengths.get(kind))
        }) {
            return Ok(0);
        }
        let target = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.in_use)
            .min_by_key(|(index, slot)| arena_slot_cost(slot, lengths, *index))
            .map_or(lengths, |(_, slot)| slot.capacities.max(lengths));
        shared_allocation_len(target)
    }

    /// Grow a free arena slot without publishing a read. The retired slab is
    /// released before allocation, matching ordinary low-peak arena growth.
    /// An allocation failure leaves a valid empty slot for the caller's
    /// smaller exact fallback.
    fn reserve_capacity(&self, lengths: BufferLengths) -> Result<()> {
        let (slot_index, target, old_capacities, old_buffers) = {
            let mut slots = self.inner.lock().unwrap();
            if slots.iter().any(|slot| {
                !slot.in_use
                    && slot.buffers.is_some()
                    && BufferKind::ALL
                        .iter()
                        .all(|&kind| slot.capacities.get(kind) >= lengths.get(kind))
            }) {
                return Ok(());
            }
            let index = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| !slot.in_use)
                .min_by_key(|(index, slot)| arena_slot_cost(slot, lengths, *index))
                .map(|(index, _)| index)
                .ok_or_else(|| DeltafinError::new("storage arena has no free slot to reserve"))?;
            let slot = &mut slots[index];
            slot.in_use = true;
            let old_capacities = slot.capacities;
            let target = old_capacities.max(lengths);
            let old_buffers = slot.buffers.take();
            slot.capacities = BufferLengths::default();
            (index, target, old_capacities, old_buffers)
        };

        if old_buffers.is_some()
            && let Some(retire_hook) = self.retire_hook.as_ref()
            && let Err(error) = invoke_buffer_retire_hook(retire_hook)
        {
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            slot.buffers = old_buffers;
            slot.capacities = old_capacities;
            slot.in_use = false;
            self.available.notify_all();
            return Err(error);
        }
        drop(old_buffers);
        let replacement = match SharedBuffers::new(target) {
            Ok(buffers) => Arc::new(buffers),
            Err(error) => {
                let mut slots = self.inner.lock().unwrap();
                let slot = &mut slots[slot_index];
                slot.in_use = false;
                self.available.notify_all();
                return Err(error);
            }
        };
        {
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            slot.buffers = Some(replacement);
            slot.capacities = target;
            slot.in_use = false;
            self.available.notify_all();
        }
        Ok(())
    }
}

impl Drop for BufferArena {
    fn drop(&mut self) {
        let Some(retire_hook) = self.retire_hook.as_ref() else {
            return;
        };
        let slots = self
            .inner
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !slots.iter().any(|slot| slot.buffers.is_some()) {
            return;
        }
        if invoke_buffer_retire_hook(retire_hook).is_err() {
            // Teardown cannot return an error. Leaking these bounded slabs is
            // safer than freeing pages still named by a no-copy device cache;
            // the hook's captured provider lease remains live through this
            // drop attempt, so ordinary successful teardown releases both.
            for slot in slots {
                if let Some(buffers) = slot.buffers.take() {
                    std::mem::forget(buffers);
                }
            }
        }
    }
}

fn arena_slot_cost(
    slot: &ArenaSlot,
    lengths: BufferLengths,
    index: usize,
) -> (u8, u128, u128, usize) {
    let fits = slot.buffers.is_some()
        && BufferKind::ALL
            .iter()
            .all(|&kind| slot.capacities.get(kind) >= lengths.get(kind));
    let mut growth = 0_u128;
    let mut waste_or_total = 0_u128;
    for kind in BufferKind::ALL {
        let old = if slot.buffers.is_some() {
            slot.capacities.get(kind)
        } else {
            0
        };
        let requested = lengths.get(kind);
        if fits {
            waste_or_total += (old - requested) as u128;
        } else {
            let target = old.max(requested);
            growth += (target - old) as u128;
            waste_or_total += target as u128;
        }
    }
    // Reuse a fitting slab first. Otherwise minimize growth in total retained
    // memory, then the resulting slab size; the index is a deterministic tie.
    (
        !fits as u8,
        if fits { waste_or_total } else { growth },
        waste_or_total,
        index,
    )
}

struct BufferLeaseInner {
    arena: Weak<BufferArena>,
    slot_index: usize,
    buffers: Option<Arc<SharedBuffers>>,
    lengths: BufferLengths,
}

impl BufferLeaseInner {
    fn buffers(&self) -> &SharedBuffers {
        self.buffers
            .as_deref()
            .expect("live buffer lease must contain its arena buffers")
    }
}

impl Drop for BufferLeaseInner {
    fn drop(&mut self) {
        // Drop the lease's buffer Arc before advertising the arena slot as
        // free. A waiter growing this slot can then retire the slot's final Arc
        // before allocating its replacement, even under concurrent teardown.
        drop(self.buffers.take());
        if let Some(arena) = self.arena.upgrade() {
            arena.release(self.slot_index);
        }
    }
}

/// A CPU-complete arena lease. Its storage returns to the bounded pool when
/// this value drops. A future asynchronous Metal/CUDA bridge must retain this
/// lease until its device completion event fires; this type deliberately does
/// not pretend that submitting a GPU command is completion.
pub struct LayerBuffers {
    lease: Arc<BufferLeaseInner>,
}

impl LayerBuffers {
    pub fn quantized(&self) -> &[u8] {
        self.lease
            .buffers()
            .get(BufferKind::Quantized)
            .as_slice(self.lease.lengths.quantized)
    }

    pub fn scales(&self) -> &[u8] {
        self.lease
            .buffers()
            .get(BufferKind::Scales)
            .as_slice(self.lease.lengths.scales)
    }

    pub fn other(&self) -> &[u8] {
        self.lease
            .buffers()
            .get(BufferKind::Other)
            .as_slice(self.lease.lengths.other)
    }

    /// Borrow the stable host pointer while this lease remains alive.
    ///
    /// This is not yet a GPU lifetime contract. Callers must not drop the
    /// `LayerBuffers` while any native CPU consumer can still access it.
    pub fn pointer(&self, kind: BufferKind) -> *const u8 {
        self.lease.buffers().get(kind).pointer.as_ptr()
    }

    /// Return the allocator's complete rounded backing lengths, not the
    /// manifest's logical tensor lengths.
    ///
    /// A device bridge that borrows these pages (Metal shared storage, CUDA
    /// host registration, or a later Windows equivalent) must receive the
    /// actual allocation envelope. Inferring it from the logical slice would
    /// make the final aligned page invisible to the lifetime contract.
    pub(crate) fn allocation_lengths(&self) -> BufferLengths {
        BufferLengths::new(
            self.lease
                .buffers()
                .get(BufferKind::Quantized)
                .allocation_len(),
            self.lease
                .buffers()
                .get(BufferKind::Scales)
                .allocation_len(),
            self.lease.buffers().get(BufferKind::Other).allocation_len(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReadStats {
    /// Logical bytes published in the compact destination buffers. An
    /// authenticated gather may physically read more enclosing source bytes
    /// while proving their digest; `elapsed` includes that work.
    pub bytes: u64,
    pub jobs: usize,
    pub workers: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ReadPriority {
    Demand,
    Prefetch,
}

// Split out from Batch so a worker can hand off its own strong reference to
// the lease-holding Batch before publishing completion. Batch::run_quantum
// used to notify from inside its own `&self` call, while the caller
// (worker_main) kept its Arc<Batch> -- and therefore the batch's arena lease
// -- alive until the *next* loop iteration. A waiter woken by that notify
// could observe the lease still held. Cloning this handle costs nothing (it
// never touches the lease) and lets worker_main drop its Arc<Batch> first.
struct BatchCompletion {
    remaining: AtomicUsize,
    cancelled: AtomicBool,
    first_error: Mutex<Option<DeltafinError>>,
    lock: Mutex<()>,
    condvar: Condvar,
}

impl BatchCompletion {
    fn new(jobs: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(jobs),
            cancelled: AtomicBool::new(false),
            first_error: Mutex::new(None),
            lock: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }
}

struct Batch {
    sources: BatchSources,
    jobs: BatchJobs,
    vectored_reads: Option<Arc<Vec<VectoredRead>>>,
    verified_extents: Option<Arc<Vec<AtomicBool>>>,
    lease: Arc<BufferLeaseInner>,
    priority: ReadPriority,
    next_job: AtomicUsize,
    completion: Arc<BatchCompletion>,
}

/// Requeue: unclaimed jobs remain, push this Arc<Batch> back onto the queue.
/// Idle: nothing left to claim right now (someone else may still be finishing
/// it). Finished: this call's last job made `remaining` hit zero; the caller
/// must drop its own Arc<Batch> before notifying the enclosed handle.
enum QuantumOutcome {
    Requeue,
    Idle,
    Finished(Arc<BatchCompletion>),
}

impl Batch {
    fn new(plan: &ReadPlan, lease: Arc<BufferLeaseInner>, priority: ReadPriority) -> Self {
        let deferred_files = plan
            .sources
            .values
            .iter()
            .any(|source| source.file.is_none() && source.persistent_file.is_none())
            .then(|| {
                (0..plan.sources.values.len())
                    .map(|_| OnceLock::new())
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            });
        Self {
            sources: BatchSources::Plan {
                sources: Arc::clone(&plan.sources),
                deferred_files,
            },
            jobs: BatchJobs::Shared(Arc::clone(&plan.jobs)),
            vectored_reads: Some(Arc::clone(&plan.vectored_reads)),
            verified_extents: Some(Arc::clone(&plan.verified_extents)),
            lease,
            priority,
            next_job: AtomicUsize::new(0),
            completion: Arc::new(BatchCompletion::new(plan.jobs.len())),
        }
    }

    fn new_deferred_exact_validated(
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        source_length: usize,
        lease: Arc<BufferLeaseInner>,
        priority: ReadPriority,
    ) -> Self {
        debug_assert!(!source_indices.is_empty());
        debug_assert!(source_indices.len() <= MAX_INLINE_DEFERRED_FILES);
        debug_assert!(
            source_indices
                .iter()
                .all(|&source| (source as usize) < catalog.inner.sources.len())
        );
        debug_assert_eq!(source_length as u64, catalog.inner.exact_source_length);
        let mut sources = [0_u32; MAX_INLINE_DEFERRED_FILES];
        sources[..source_indices.len()].copy_from_slice(source_indices);
        Self {
            sources: BatchSources::DeferredCatalog(Arc::clone(&catalog.inner)),
            jobs: BatchJobs::Inline(InlineReadJobs {
                sources,
                len: source_indices.len(),
                destination,
                source_length,
            }),
            vectored_reads: None,
            // Raw-v1 is guarded by exact length/type/name contracts rather
            // than an authenticated pack digest. Scale4-v2 must use its
            // separate manifest/hash path and never enters this constructor.
            verified_extents: None,
            lease,
            priority,
            next_job: AtomicUsize::new(0),
            completion: Arc::new(BatchCompletion::new(source_indices.len())),
        }
    }

    /// Run at most one scheduling quantum. `Finished` carries the completion
    /// handle the caller must notify -- but only after it has dropped its own
    /// Arc<Batch>, so the notified waiter never observes this batch's arena
    /// lease still held by the worker that just finished it.
    fn run_quantum(&self) -> QuantumOutcome {
        for _ in 0..WORK_QUANTUM {
            let index = self.next_job.fetch_add(1, Ordering::Relaxed);
            let Some(job) = self.jobs.get(index) else {
                return QuantumOutcome::Idle;
            };
            if let Err(error) = self.run_job(job) {
                let mut first_error = self.completion.first_error.lock().unwrap();
                if first_error.is_none() {
                    *first_error = Some(error);
                }
            }
            if self.completion.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                return QuantumOutcome::Finished(Arc::clone(&self.completion));
            }
        }
        if self.next_job.load(Ordering::Relaxed) < self.jobs.len() {
            QuantumOutcome::Requeue
        } else {
            QuantumOutcome::Idle
        }
    }

    /// Atomically claim every job that no worker has started. Jobs already in
    /// `pread` finish normally; their arena lease remains owned by this batch
    /// until the last active worker publishes completion. This is the exact
    /// cancellation primitive needed by speculative I/O: no recycled buffer
    /// can race an in-flight kernel or read.
    fn cancel_unclaimed(&self) {
        self.completion.cancelled.store(true, Ordering::Release);
        let jobs = self.jobs.len();
        loop {
            let next = self.next_job.load(Ordering::Acquire);
            if next >= jobs {
                break;
            }
            if self
                .next_job
                .compare_exchange(next, jobs, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let cancelled = jobs - next;
                if self.completion.remaining.fetch_sub(cancelled, Ordering::AcqRel) == cancelled {
                    let _guard = self.completion.lock.lock().unwrap();
                    self.completion.condvar.notify_all();
                }
                break;
            }
        }
    }

    fn run_job(&self, job: ReadJob) -> Result<()> {
        if let JobSource::Vectored { source, scatter } = job.source {
            return self.run_vectored_read(source, scatter, job.expected_digest);
        }
        if let JobSource::AuthenticatedScatter { source } = job.source {
            return self.run_authenticated_scatter(source);
        }
        // SAFETY: `ReadPlan::open` rejected every overlapping destination and
        // bounded every range. The batch does not publish immutable access
        // until `remaining` reaches zero.
        let pointer = self
            .lease
            .buffers()
            .get(job.destination)
            .pointer_at(job.destination_offset, job.length);
        // SAFETY: `ReadPlan::open` rejected overlapping destinations and
        // bounded the range. No immutable view is exposed until completion.
        let destination = unsafe { std::slice::from_raw_parts_mut(pointer, job.length) };
        // Snapshot qualification before starting I/O. If two first-use
        // batches race, both authenticate the bytes they individually read;
        // one completed batch cannot retroactively qualify the other's
        // already-in-flight read.
        let verify_after_read = match job.verification_index {
            Some(index) => {
                let verified = self
                    .verified_extents
                    .as_ref()
                    .and_then(|extents| extents.get(index))
                    .ok_or_else(|| {
                        DeltafinError::new("read job refers to an unknown verification slot")
                    })?;
                !verified.load(Ordering::Acquire)
            }
            None => false,
        };
        match job.source {
            JobSource::Zero => {
                destination.fill(0);
            }
            JobSource::File {
                source,
                source_offset,
            } => {
                let BatchSources::Plan {
                    sources,
                    deferred_files,
                } = &self.sources
                else {
                    return Err(DeltafinError::new(
                        "file read job is attached to the wrong source set",
                    ));
                };
                let source_index = source;
                let source = sources
                    .values
                    .get(source_index)
                    .ok_or_else(|| DeltafinError::new("read job refers to an unknown source"))?;
                if !source.verifications.is_empty() {
                    return Err(DeltafinError::new(
                        "authenticated source was incorrectly routed through a double-read file job",
                    ));
                }
                let file = if let Some(file) = source.file.as_ref() {
                    file
                } else if let Some(slot) = source.persistent_file.as_ref() {
                    match slot.get_or_init(|| open_deferred_source(source)) {
                        Ok(file) => file,
                        Err(error) => return Err(error.clone()),
                    }
                } else {
                    let slot = deferred_files
                        .as_ref()
                        .and_then(|files| files.get(source_index))
                        .ok_or_else(|| {
                            DeltafinError::new("deferred read source has no batch descriptor slot")
                        })?;
                    match slot.get_or_init(|| open_deferred_source(source)) {
                        Ok(file) => file,
                        Err(error) => return Err(error.clone()),
                    }
                };
                let mut completed = 0_usize;
                while completed < destination.len() {
                    let count = match file.read_at(
                        &mut destination[completed..],
                        source_offset + completed as u64,
                    ) {
                        Ok(count) => count,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        Err(error) => return Err(io_error("pread", &source.path, error)),
                    };
                    if count == 0 {
                        return Err(DeltafinError::new(format!(
                            "short pread {}/{} from {} at {}",
                            completed,
                            destination.len(),
                            source.path.display(),
                            source_offset
                        )));
                    }
                    completed += count;
                }
                if let (true, Some(expected), Some(index)) = (
                    verify_after_read,
                    job.expected_digest,
                    job.verification_index,
                ) {
                    let verified = self
                        .verified_extents
                        .as_ref()
                        .and_then(|extents| extents.get(index))
                        .ok_or_else(|| {
                            DeltafinError::new("read job refers to an unknown verification slot")
                        })?;
                    let actual = crate::packfile::digest_bytes(destination);
                    if actual != expected {
                        return Err(DeltafinError::new(format!(
                            "authenticated read from {} at {} failed SHA-256 verification",
                            source.path.display(),
                            source_offset,
                        )));
                    }
                    verified.store(true, Ordering::Release);
                }
                drop_completed_cache(file, source.cache_policy, source_offset, job.length);
            }
            JobSource::AuthenticatedScatter { .. } => unreachable!(
                "authenticated scatter jobs are handled before creating one destination slice"
            ),
            JobSource::Vectored { .. } => unreachable!(
                "vectored jobs are handled before creating one contiguous destination slice"
            ),
            JobSource::DeferredCatalog { source } => {
                let BatchSources::DeferredCatalog(catalog) = &self.sources else {
                    return Err(DeltafinError::new(
                        "catalog read job is attached to the wrong source set",
                    ));
                };
                let source_name = catalog.sources.get(source as usize).ok_or_else(|| {
                    DeltafinError::new("catalog read job refers to an unknown source")
                })?;
                let file = open_deferred_catalog_source(catalog, source_name)?;
                let mut completed = 0_usize;
                while completed < destination.len() {
                    let count = match file.read_at(&mut destination[completed..], completed as u64)
                    {
                        Ok(count) => count,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        Err(error) => {
                            return Err(catalog_io_error("pread", catalog, source_name, error));
                        }
                    };
                    if count == 0 {
                        return Err(DeltafinError::new(format!(
                            "short pread {}/{} from {}/{} at 0",
                            completed,
                            destination.len(),
                            catalog.directory_path.display(),
                            source_name.as_str(),
                        )));
                    }
                    completed += count;
                }
                drop_completed_cache(&file, catalog.cache_policy, 0, job.length);
            }
        }
        Ok(())
    }

    fn run_vectored_read(
        &self,
        source_index: usize,
        scatter_index: usize,
        expected_digest: Option<[u8; 32]>,
    ) -> Result<()> {
        let BatchSources::Plan {
            sources,
            deferred_files,
        } = &self.sources
        else {
            return Err(DeltafinError::new(
                "vectored read job is attached to the wrong source set",
            ));
        };
        let source = sources
            .values
            .get(source_index)
            .ok_or_else(|| DeltafinError::new("vectored job refers to an unknown source"))?;
        if !source.verifications.is_empty() {
            return Err(DeltafinError::new(
                "authenticated source was incorrectly routed through a vectored read job",
            ));
        }
        let scatter = self
            .vectored_reads
            .as_ref()
            .and_then(|reads| reads.get(scatter_index))
            .ok_or_else(|| DeltafinError::new("vectored job refers to an unknown scatter"))?;
        if scatter.destinations.is_empty() || scatter.destinations.len() > MAX_VECTORED_DESTINATIONS
        {
            return Err(DeltafinError::new(
                "vectored job has an invalid destination count",
            ));
        }
        let file = if let Some(file) = source.file.as_ref() {
            file
        } else if let Some(slot) = source.persistent_file.as_ref() {
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        } else {
            let slot = deferred_files
                .as_ref()
                .and_then(|files| files.get(source_index))
                .ok_or_else(|| {
                    DeltafinError::new("vectored read source has no batch descriptor slot")
                })?;
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        };

        // A fixed stack table avoids an allocation for every routed expert.
        // validate_destinations proved these arena regions are bounded and
        // globally disjoint; the batch remains private until every job ends.
        let mut vectors: [libc::iovec; MAX_VECTORED_DESTINATIONS] = unsafe { std::mem::zeroed() };
        let mut total = 0_usize;
        for (vector, destination) in vectors.iter_mut().zip(scatter.destinations.iter()) {
            let pointer = self
                .lease
                .buffers()
                .get(destination.destination)
                .pointer_at(destination.destination_offset, destination.length);
            vector.iov_base = pointer.cast();
            vector.iov_len = destination.length;
            total = total
                .checked_add(destination.length)
                .ok_or_else(|| DeltafinError::new("vectored read length overflows usize"))?;
        }
        if total > isize::MAX as usize {
            return Err(DeltafinError::new(
                "vectored read exceeds the platform syscall length",
            ));
        }
        let mut first = 0_usize;
        let mut remaining = total;
        let mut file_offset = i64::try_from(scatter.source_offset)
            .map_err(|_| DeltafinError::new("vectored source offset exceeds off_t"))?;
        while remaining != 0 {
            let count = i32::try_from(scatter.destinations.len() - first)
                .map_err(|_| DeltafinError::new("vectored destination count exceeds c_int"))?;
            // SAFETY: every iovec points into a live, private, prevalidated
            // arena range. `file` remains open for the syscall, `count` is
            // bounded, and preadv does not retain either pointer or descriptor.
            let read = unsafe {
                libc::preadv(
                    file.as_raw_fd(),
                    vectors[first..].as_ptr(),
                    count,
                    file_offset,
                )
            };
            if read == -1 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(io_error("preadv", &source.path, error));
            }
            if read == 0 {
                return Err(DeltafinError::new(format!(
                    "short preadv {}/{} from {} at {}",
                    total - remaining,
                    total,
                    source.path.display(),
                    scatter.source_offset,
                )));
            }
            let read = usize::try_from(read)
                .map_err(|_| DeltafinError::new("preadv returned a negative byte count"))?;
            if read > remaining {
                return Err(DeltafinError::new("preadv exceeded its destination length"));
            }
            remaining -= read;
            file_offset = file_offset
                .checked_add(read as i64)
                .ok_or_else(|| DeltafinError::new("vectored source offset overflows off_t"))?;

            let mut consumed = read;
            while consumed != 0 {
                let length = vectors[first].iov_len;
                if consumed >= length {
                    consumed -= length;
                    first += 1;
                } else {
                    // SAFETY: `consumed < iov_len`, so advancing the pointer
                    // remains inside the same validated destination range.
                    vectors[first].iov_base =
                        unsafe { vectors[first].iov_base.cast::<u8>().add(consumed).cast() };
                    vectors[first].iov_len -= consumed;
                    consumed = 0;
                }
            }
        }
        if let Some(expected) = expected_digest {
            // Unlike persistent ordinary extents, deferred vectored
            // sources can reopen on every batch. Hash every verified job;
            // higher-level immutable identity caches omit the digest from
            // later plans instead of storing a plan-local qualification
            // that might cross descriptor generations.
            let mut digest = DigestState::new();
            for destination in &scatter.destinations {
                let pointer = self
                    .lease
                    .buffers()
                    .get(destination.destination)
                    .pointer_at(destination.destination_offset, destination.length);
                // SAFETY: destination validation proved this complete
                // range is bounded. Its read job has finished filling it,
                // and every other in-flight job owns a disjoint range.
                let bytes =
                    unsafe { std::slice::from_raw_parts(pointer.cast_const(), destination.length) };
                digest.update(bytes);
            }
            if digest.finalize() != expected {
                return Err(DeltafinError::new(format!(
                    "authenticated vectored read from {} at {} failed SHA-256 verification",
                    source.path.display(),
                    scatter.source_offset,
                )));
            }
        }
        drop_completed_cache(file, source.cache_policy, scatter.source_offset, total);
        Ok(())
    }

    fn run_authenticated_scatter(&self, source_index: usize) -> Result<()> {
        let BatchSources::Plan {
            sources,
            deferred_files,
            ..
        } = &self.sources
        else {
            return Err(DeltafinError::new(
                "authenticated scatter job is attached to the wrong source set",
            ));
        };
        let source = sources
            .values
            .get(source_index)
            .ok_or_else(|| DeltafinError::new("scatter job refers to an unknown source"))?;
        if source.verifications.is_empty() || source.scatter_extents.is_empty() {
            return Err(DeltafinError::new(
                "authenticated scatter source has an incomplete plan",
            ));
        }
        let file = if let Some(file) = source.file.as_ref() {
            file
        } else if let Some(slot) = source.persistent_file.as_ref() {
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        } else {
            let slot = deferred_files
                .as_ref()
                .and_then(|files| files.get(source_index))
                .ok_or_else(|| {
                    DeltafinError::new("authenticated scatter source has no batch descriptor slot")
                })?;
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        };
        authenticate_and_scatter_source(file, source, &self.lease)
    }

    fn wait_for_completion(&self) {
        let mut guard = self.completion.lock.lock().unwrap();
        while self.completion.remaining.load(Ordering::Acquire) != 0 {
            guard = self.completion.condvar.wait(guard).unwrap();
        }
        drop(guard);
    }

    fn validate_deferred_source_identities(&self) -> Result<()> {
        let BatchSources::Plan {
            sources,
            deferred_files,
        } = &self.sources
        else {
            return Ok(());
        };
        for (source_index, source) in sources.values.iter().enumerate() {
            let Some(expected) = source.expected_identity else {
                continue;
            };
            let actual = if let Some(file) = source.file.as_ref() {
                descriptor_identity(file, &source.path)?
            } else if let Some(slot) = source.persistent_file.as_ref() {
                match slot.get() {
                    Some(Ok(file)) => descriptor_identity(file, &source.path)?,
                    Some(Err(error)) => return Err(error.clone()),
                    None => {
                        return Err(DeltafinError::new(
                            "identity-pinned persistent source was never opened",
                        ));
                    }
                }
            } else {
                let slot = deferred_files
                    .as_ref()
                    .and_then(|files| files.get(source_index))
                    .ok_or_else(|| {
                        DeltafinError::new("identity-pinned deferred source has no descriptor slot")
                    })?;
                match slot.get() {
                    Some(Ok(file)) => descriptor_identity(file, &source.path)?,
                    Some(Err(error)) => return Err(error.clone()),
                    None => {
                        return Err(DeltafinError::new(
                            "identity-pinned deferred source was never opened",
                        ));
                    }
                }
            };
            if actual != expected {
                return Err(DeltafinError::new(format!(
                    "deferred source identity changed during range gather: {}",
                    source.path.display(),
                )));
            }
        }
        Ok(())
    }

    fn wait(&self) -> Result<()> {
        self.wait_for_completion();
        if self.completion.cancelled.load(Ordering::Acquire) {
            return Err(DeltafinError::new("storage read ticket was cancelled"));
        }
        if let Some(error) = self.completion.first_error.lock().unwrap().clone() {
            return Err(error);
        }
        self.validate_deferred_source_identities()
    }
}

pub struct ReadTicket {
    batch: Arc<Batch>,
    started: Instant,
    bytes: u64,
    jobs: usize,
    workers: usize,
}

impl ReadTicket {
    pub fn is_ready(&self) -> bool {
        self.batch.completion.remaining.load(Ordering::Acquire) == 0
    }

    pub fn wait(self) -> Result<(LayerBuffers, ReadStats)> {
        self.batch.wait()?;
        let buffers = LayerBuffers {
            lease: Arc::clone(&self.batch.lease),
        };
        Ok((
            buffers,
            ReadStats {
                bytes: self.bytes,
                jobs: self.jobs,
                workers: self.workers,
                elapsed: self.started.elapsed(),
            },
        ))
    }

    /// Cancel every unclaimed job and wait only for work already inside an I/O
    /// syscall. The ticket publishes no bytes; dropping it then releases its
    /// arena slot. Optional callers deliberately ignore read errors here
    /// because authoritative demand I/O will retry any selected expert.
    pub fn cancel_and_wait(self) {
        self.batch.cancel_unclaimed();
        self.batch.wait_for_completion();
    }

    pub fn cancel_unclaimed(&self) {
        self.batch.cancel_unclaimed();
    }

    pub fn drain_cancelled(self) {
        self.batch.wait_for_completion();
    }
}

impl Drop for ReadTicket {
    fn drop(&mut self) {
        self.batch.cancel_unclaimed();
    }
}

struct PriorityQueues<T> {
    demand: VecDeque<T>,
    prefetch: VecDeque<T>,
    demand_streak: usize,
}

impl<T> PriorityQueues<T> {
    fn new() -> Self {
        Self {
            demand: VecDeque::new(),
            prefetch: VecDeque::new(),
            demand_streak: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.demand.is_empty() && self.prefetch.is_empty()
    }

    fn push(&mut self, priority: ReadPriority, value: T) {
        match priority {
            ReadPriority::Demand => self.demand.push_back(value),
            ReadPriority::Prefetch => self.prefetch.push_back(value),
        }
    }

    fn pop(&mut self) -> Option<T> {
        if !self.demand.is_empty()
            && (self.prefetch.is_empty() || self.demand_streak < MAX_DEMAND_STREAK)
        {
            self.demand_streak = self.demand_streak.saturating_add(1);
            return self.demand.pop_front();
        }
        if let Some(value) = self.prefetch.pop_front() {
            self.demand_streak = 0;
            return Some(value);
        }
        self.demand_streak = self.demand_streak.saturating_add(1);
        self.demand.pop_front()
    }
}

struct PoolState {
    inner: Mutex<PoolInner>,
    available: Condvar,
}

struct PoolInner {
    queues: PriorityQueues<Arc<Batch>>,
    closed: bool,
}

pub struct Reader {
    state: Arc<PoolState>,
    arena: Arc<BufferArena>,
    threads: Vec<JoinHandle<()>>,
}

impl Reader {
    pub fn new(workers: usize) -> Result<Self> {
        Self::with_arena_capacity(workers, DEFAULT_ARENA_SLOTS)
    }

    pub fn with_arena_capacity(workers: usize, arena_slots: usize) -> Result<Self> {
        Self::with_arena_capacity_and_retire_hook(workers, arena_slots, None)
    }

    pub(crate) fn with_arena_capacity_and_retire_hook(
        workers: usize,
        arena_slots: usize,
        retire_hook: Option<BufferRetireHook>,
    ) -> Result<Self> {
        if workers == 0 {
            return Err(DeltafinError::new(
                "storage reader needs at least one worker",
            ));
        }
        crate::io_priority::configure_process_for_model_io();
        let state = Arc::new(PoolState {
            inner: Mutex::new(PoolInner {
                queues: PriorityQueues::new(),
                closed: false,
            }),
            available: Condvar::new(),
        });
        let arena = BufferArena::new_with_retire_hook(arena_slots, retire_hook)?;
        let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(workers);
        for index in 0..workers {
            let worker_state = Arc::clone(&state);
            let handle = match thread::Builder::new()
                .name(format!("deltafin-io-{index}"))
                .spawn(move || worker_main(worker_state))
            {
                Ok(handle) => handle,
                Err(error) => {
                    state.inner.lock().unwrap().closed = true;
                    state.available.notify_all();
                    for thread in threads {
                        let _ = thread.join();
                    }
                    return Err(DeltafinError::new(format!("start I/O worker: {error}")));
                }
            };
            threads.push(handle);
        }
        Ok(Self {
            state,
            arena,
            threads,
        })
    }

    pub fn workers(&self) -> usize {
        self.threads.len()
    }

    pub(crate) fn replacement_admission_bytes(&self, lengths: BufferLengths) -> Result<u64> {
        self.arena.replacement_admission_bytes(lengths)
    }

    pub(crate) fn reserve_capacity(&self, lengths: BufferLengths) -> Result<()> {
        self.arena.reserve_capacity(lengths)
    }

    pub fn read(&self, plan: &ReadPlan) -> Result<(LayerBuffers, ReadStats)> {
        self.submit(plan, ReadPriority::Demand)?.wait()
    }

    /// Admit a read into the bounded arena and return while its I/O is in
    /// flight. Admission waits when every slot is still leased by a caller.
    pub fn submit(&self, plan: &ReadPlan, priority: ReadPriority) -> Result<ReadTicket> {
        self.submit_inner(plan, priority, true)?.ok_or_else(|| {
            DeltafinError::new("blocking storage submission unexpectedly found no arena slot")
        })
    }

    /// Non-blocking admission for callers that need to apply their own
    /// backpressure. `None` means every bounded arena slot is currently leased.
    pub fn try_submit(
        &self,
        plan: &ReadPlan,
        priority: ReadPriority,
    ) -> Result<Option<ReadTicket>> {
        self.submit_inner(plan, priority, false)
    }

    /// Submit up to sixteen catalogued whole files into adjacent slots of one
    /// destination slab without compiling a `ReadPlan`.
    ///
    /// This is the routed-decode fast path: catalog paths and directory
    /// ownership are immutable session state, jobs live inline in `Batch`, and
    /// source selection is a fixed array of integer indices. The only heap
    /// ownership admitted per read is the shared batch/ticket required by the
    /// persistent worker pool and the already-bounded arena lease.
    pub fn submit_deferred_exact(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        priority: ReadPriority,
    ) -> Result<ReadTicket> {
        let (lengths, source_length) =
            deferred_batch_lengths(catalog, source_indices, destination)?;
        let logical_bytes = lengths.get(destination) as u64;
        let started = Instant::now();
        {
            let inner = self.state.inner.lock().unwrap();
            if inner.closed {
                return Err(DeltafinError::new("storage reader is closed"));
            }
        }
        let lease = self
            .arena
            .acquire(lengths, true, priority)?
            .ok_or_else(|| DeltafinError::new("blocking catalog submission found no arena slot"))?;
        let batch = Arc::new(Batch::new_deferred_exact_validated(
            catalog,
            source_indices,
            destination,
            source_length,
            lease,
            priority,
        ));
        let participating = self.workers().min(source_indices.len());
        let mut inner = self.state.inner.lock().unwrap();
        if inner.closed {
            return Err(DeltafinError::new("storage reader is closed"));
        }
        for _ in 0..participating {
            inner.queues.push(priority, Arc::clone(&batch));
        }
        self.state.available.notify_all();
        drop(inner);
        Ok(ReadTicket {
            batch,
            started,
            bytes: logical_bytes,
            jobs: source_indices.len(),
            workers: participating,
        })
    }

    pub fn read_deferred_exact(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
    ) -> Result<(LayerBuffers, ReadStats)> {
        self.submit_deferred_exact(catalog, source_indices, destination, ReadPriority::Demand)?
            .wait()
    }

    fn submit_inner(
        &self,
        plan: &ReadPlan,
        priority: ReadPriority,
        wait_for_slot: bool,
    ) -> Result<Option<ReadTicket>> {
        let started = Instant::now();
        {
            let inner = self.state.inner.lock().unwrap();
            if inner.closed {
                return Err(DeltafinError::new("storage reader is closed"));
            }
        }
        let Some(lease) = self
            .arena
            .acquire(plan.buffer_lengths, wait_for_slot, priority)?
        else {
            return Ok(None);
        };
        let batch = Arc::new(Batch::new(plan, lease, priority));
        let participating = self.workers().min(plan.jobs.len());
        if participating != 0 {
            let mut inner = self.state.inner.lock().unwrap();
            if inner.closed {
                return Err(DeltafinError::new("storage reader is closed"));
            }
            for _ in 0..participating {
                inner.queues.push(priority, Arc::clone(&batch));
            }
            self.state.available.notify_all();
        }
        Ok(Some(ReadTicket {
            batch,
            started,
            bytes: plan.logical_bytes,
            jobs: plan.jobs.len(),
            workers: participating,
        }))
    }
}

fn deferred_batch_lengths(
    catalog: &DeferredExactCatalog,
    source_indices: &[u32],
    destination: BufferKind,
) -> Result<(BufferLengths, usize)> {
    if source_indices.is_empty() || source_indices.len() > MAX_INLINE_DEFERRED_FILES {
        return Err(DeltafinError::new(format!(
            "an inline deferred batch needs 1..={MAX_INLINE_DEFERRED_FILES} sources; got {}",
            source_indices.len()
        )));
    }
    for &source in source_indices {
        if source as usize >= catalog.inner.sources.len() {
            return Err(DeltafinError::new(format!(
                "deferred source index {source} is outside catalog length {}",
                catalog.inner.sources.len()
            )));
        }
    }
    let source_length = usize::try_from(catalog.inner.exact_source_length).map_err(|_| {
        DeltafinError::new("deferred source length does not fit this platform's usize")
    })?;
    let batch_length = source_length
        .checked_mul(source_indices.len())
        .ok_or_else(|| DeltafinError::new("deferred batch length overflows usize"))?;
    let lengths = match destination {
        BufferKind::Quantized => BufferLengths::new(batch_length, 0, 0),
        BufferKind::Scales => BufferLengths::new(0, batch_length, 0),
        BufferKind::Other => BufferLengths::new(0, 0, batch_length),
    };
    Ok((lengths, source_length))
}

fn worker_main(state: Arc<PoolState>) {
    crate::io_priority::configure_model_io_thread();
    loop {
        let batch = {
            let mut inner = state.inner.lock().unwrap();
            while inner.queues.is_empty() && !inner.closed {
                inner = state.available.wait(inner).unwrap();
            }
            let Some(batch) = inner.queues.pop() else {
                return;
            };
            batch
        };
        match batch.run_quantum() {
            QuantumOutcome::Requeue => {
                let priority = batch.priority;
                let mut inner = state.inner.lock().unwrap();
                // Internal requeues remain valid during shutdown: Reader::drop
                // drains admitted work before joining the workers.
                inner.queues.push(priority, batch);
                state.available.notify_one();
            }
            QuantumOutcome::Idle => drop(batch),
            QuantumOutcome::Finished(completion) => {
                // Drop this worker's Arc<Batch> -- and with it, if this was
                // the last reference, the batch's arena lease -- before
                // publishing completion. A waiter woken by the notify below
                // must never be able to observe this lease still held.
                drop(batch);
                let _guard = completion.lock.lock().unwrap();
                completion.condvar.notify_one();
            }
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.state.inner.lock().unwrap().closed = true;
        self.state.available.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DeferredSourceIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

fn metadata_identity(metadata: &std::fs::Metadata) -> DeferredSourceIdentity {
    DeferredSourceIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

fn descriptor_identity(file: &File, path: &Path) -> Result<DeferredSourceIdentity> {
    let metadata = file
        .metadata()
        .map_err(|error| io_error("stat authenticated source", path, error))?;
    Ok(metadata_identity(&metadata))
}

fn capture_deferred_source_identity(
    path: &Path,
    exact_source_length: u64,
) -> Result<DeferredSourceIdentity> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(open_cloexec_nofollow())
        .open(path)
        .map_err(|error| io_error("open deferred identity source", path, error))?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("stat deferred identity source", path, error))?;
    if !metadata.is_file() || metadata.len() != exact_source_length {
        return Err(DeltafinError::new(format!(
            "deferred identity source {} is not an exact regular {exact_source_length}-byte file",
            path.display(),
        )));
    }
    Ok(metadata_identity(&metadata))
}

/// Authenticate every byte in each declared verification range and scatter
/// retained subranges in the same positional-read pass.
///
/// The arena remains batch-private until all source jobs succeed, so a digest
/// or identity failure can leave partially written private pages but can never
/// publish them. Destination ranges were globally proven disjoint when the
/// plan was opened, which also makes concurrent source jobs safe.
fn authenticate_and_scatter_source(
    file: &File,
    source: &Source,
    lease: &BufferLeaseInner,
) -> Result<()> {
    let before = descriptor_identity(file, &source.path)?;
    if before.bytes != source.length {
        return Err(DeltafinError::new(format!(
            "authenticated source {} changed length before verification",
            source.path.display()
        )));
    }

    let mut scratch = vec![0_u8; 256 * 1024];
    for (verification_index, verification) in source.verifications.iter().enumerate() {
        let verification_end = verification
            .source_offset
            .checked_add(verification.length as u64)
            .ok_or_else(|| DeltafinError::new("authenticated source range overflows u64"))?;
        let mut cursor = verification.source_offset;
        let mut digest = DigestState::new();

        for scatter in source
            .scatter_extents
            .iter()
            .filter(|scatter| scatter.verification_index == verification_index)
        {
            if scatter.source_offset < cursor {
                return Err(DeltafinError::new(format!(
                    "authenticated gather order overlaps in {}",
                    source.path.display()
                )));
            }
            while cursor < scatter.source_offset {
                let remaining = usize::try_from(scatter.source_offset - cursor)
                    .map_err(|_| DeltafinError::new("authenticated gap exceeds usize"))?;
                let request = scratch.len().min(remaining);
                let count = match file.read_at(&mut scratch[..request], cursor) {
                    Ok(count) => count,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        return Err(io_error(
                            "pread authenticated source gap",
                            &source.path,
                            error,
                        ));
                    }
                };
                if count == 0 {
                    return Err(DeltafinError::new(format!(
                        "short authenticated pread from {} at {}",
                        source.path.display(),
                        cursor
                    )));
                }
                digest.update(&scratch[..count]);
                cursor += count as u64;
            }

            // SAFETY: validate_destinations proved all output extents are
            // bounded and globally disjoint. The lease is not exposed as an
            // immutable LayerBuffers value until every batch job completes.
            let pointer = lease
                .buffers()
                .get(scatter.destination)
                .pointer_at(scatter.destination_offset, scatter.length);
            let destination = unsafe { std::slice::from_raw_parts_mut(pointer, scatter.length) };
            let mut completed = 0_usize;
            while completed < destination.len() {
                let count = match file.read_at(
                    &mut destination[completed..],
                    scatter.source_offset + completed as u64,
                ) {
                    Ok(count) => count,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        return Err(io_error("pread authenticated gather", &source.path, error));
                    }
                };
                if count == 0 {
                    return Err(DeltafinError::new(format!(
                        "short authenticated gather pread {}/{} from {} at {}",
                        completed,
                        destination.len(),
                        source.path.display(),
                        scatter.source_offset
                    )));
                }
                digest.update(&destination[completed..completed + count]);
                completed += count;
            }
            cursor = scatter.source_offset + scatter.length as u64;
        }

        while cursor < verification_end {
            let remaining = usize::try_from(verification_end - cursor)
                .map_err(|_| DeltafinError::new("authenticated tail exceeds usize"))?;
            let request = scratch.len().min(remaining);
            let count = match file.read_at(&mut scratch[..request], cursor) {
                Ok(count) => count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(io_error(
                        "pread authenticated source tail",
                        &source.path,
                        error,
                    ));
                }
            };
            if count == 0 {
                return Err(DeltafinError::new(format!(
                    "short authenticated pread from {} at {}",
                    source.path.display(),
                    cursor
                )));
            }
            digest.update(&scratch[..count]);
            cursor += count as u64;
        }

        if digest.finalize() != verification.expected_digest {
            return Err(DeltafinError::new(format!(
                "source authentication failed SHA-256 verification for {} at {}..{}",
                source.path.display(),
                verification.source_offset,
                verification_end,
            )));
        }
        drop_completed_cache(
            file,
            source.cache_policy,
            verification.source_offset,
            verification.length,
        );
    }

    let after = descriptor_identity(file, &source.path)?;
    if after != before {
        return Err(DeltafinError::new(format!(
            "authenticated source identity changed during verification and gather: {}",
            source.path.display()
        )));
    }
    Ok(())
}

fn open_deferred_source(source: &Source) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(open_cloexec_nofollow())
        .open(&source.path)
        .map_err(|error| {
            if matches!(error.raw_os_error(), Some(23 | 24)) {
                DeltafinError::new(format!(
                    "open deferred source {} without following symlinks: descriptor limit exhausted",
                    source.path.display()
                ))
            } else {
                io_error("open deferred non-symlink source", &source.path, error)
            }
        })?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("stat deferred source", &source.path, error))?;
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "deferred source is not a regular file: {}",
            source.path.display()
        )));
    }
    if metadata.len() != source.length {
        return Err(DeltafinError::new(format!(
            "deferred source {} is {} bytes; expected exact length {}",
            source.path.display(),
            metadata.len(),
            source.length,
        )));
    }
    if let Some(expected) = source.expected_identity {
        let actual = metadata_identity(&metadata);
        if actual != expected {
            return Err(DeltafinError::new(format!(
                "deferred source identity changed before range gather: {}",
                source.path.display(),
            )));
        }
    }
    configure_cache_policy(&file, &source.path, source.cache_policy)?;
    Ok(file)
}

#[cfg(target_os = "macos")]
const fn open_cloexec_nofollow() -> i32 {
    // Darwin O_CLOEXEC | O_NOFOLLOW.
    0x0100_0100
}

#[cfg(target_os = "linux")]
const fn open_cloexec_nofollow() -> i32 {
    // Linux O_CLOEXEC | O_NOFOLLOW.
    0x000a_0000
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("Deltafin native storage currently supports macOS and Linux");

fn open_deferred_catalog_source(
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
) -> Result<File> {
    unsafe extern "C" {
        fn openat(
            directory: libc::c_int,
            path: *const libc::c_char,
            flags: libc::c_int,
            ...
        ) -> libc::c_int;
    }
    // SAFETY: the catalog retains a live directory descriptor, `source` is a
    // validated NUL-terminated direct child name, and no mode argument is
    // required because these flags never create a file.
    let descriptor = unsafe {
        openat(
            catalog.directory.as_raw_fd(),
            source.as_c_str().as_ptr(),
            open_cloexec_nofollow(),
        )
    };
    if descriptor < 0 {
        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(23 | 24)) {
            return Err(DeltafinError::new(format!(
                "open deferred source {}/{} without following symlinks: descriptor limit exhausted",
                catalog.directory_path.display(),
                source.as_str(),
            )));
        }
        return Err(catalog_io_error(
            "open deferred non-symlink source",
            catalog,
            source,
            error,
        ));
    }
    // SAFETY: `openat` returned a new owned descriptor. This is its unique
    // owner and `File` closes it on every subsequent success/error path.
    let file = unsafe { File::from_raw_fd(descriptor) };
    let metadata = file
        .metadata()
        .map_err(|error| catalog_io_error("stat deferred source", catalog, source, error))?;
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "deferred source is not a regular file: {}/{}",
            catalog.directory_path.display(),
            source.as_str(),
        )));
    }
    if metadata.len() != catalog.exact_source_length {
        return Err(DeltafinError::new(format!(
            "deferred source {}/{} is {} bytes; expected exact length {}",
            catalog.directory_path.display(),
            source.as_str(),
            metadata.len(),
            catalog.exact_source_length,
        )));
    }
    configure_catalog_cache_policy(&file, catalog, source)?;
    Ok(file)
}

fn catalog_io_error(
    operation: &str,
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
    error: io::Error,
) -> DeltafinError {
    DeltafinError::new(format!(
        "{operation} {}/{}: {error}",
        catalog.directory_path.display(),
        source.as_str(),
    ))
}

#[cfg(target_os = "macos")]
fn configure_catalog_cache_policy(
    file: &File,
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
) -> Result<()> {
    if catalog.cache_policy != CachePolicy::Streaming {
        return Ok(());
    }
    const F_NOCACHE: i32 = 48;
    unsafe extern "C" {
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }
    // SAFETY: the descriptor is live and F_NOCACHE accepts an integer.
    if unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) } == -1 {
        return Err(catalog_io_error(
            "enable F_NOCACHE",
            catalog,
            source,
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn configure_catalog_cache_policy(
    _file: &File,
    _catalog: &DeferredExactCatalogInner,
    _source: &DeferredSourceName,
) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn configure_cache_policy(file: &File, path: &Path, policy: CachePolicy) -> Result<()> {
    if policy != CachePolicy::Streaming {
        return Ok(());
    }
    const F_NOCACHE: i32 = 48;
    unsafe extern "C" {
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }
    // SAFETY: the descriptor is live, F_NOCACHE accepts an integer argument,
    // and this call does not transfer descriptor ownership.
    let result = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
    if result == -1 {
        return Err(io_error(
            "enable F_NOCACHE",
            path,
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn configure_cache_policy(_file: &File, _path: &Path, _policy: CachePolicy) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn drop_completed_cache(file: &File, policy: CachePolicy, offset: u64, length: usize) {
    if policy != CachePolicy::Streaming {
        return;
    }
    const POSIX_FADV_DONTNEED: i32 = 4;
    unsafe extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, length: i64, advice: i32) -> i32;
    }
    if let (Ok(offset), Ok(length)) = (i64::try_from(offset), i64::try_from(length)) {
        // Best effort, matching the existing Linux behavior. Some filesystems
        // legitimately reject advisory cache control.
        let _ = unsafe { posix_fadvise(file.as_raw_fd(), offset, length, POSIX_FADV_DONTNEED) };
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_completed_cache(_file: &File, _policy: CachePolicy, _offset: u64, _length: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEST_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let serial = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-storage-{}-{nonce}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            let mut file = File::create(&path).unwrap();
            file.write_all(bytes).unwrap();
            path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn assembles_three_buffers_with_chunked_parallel_reads() {
        let directory = TestDirectory::new();
        let first = directory.write("first", &(0_u8..64).collect::<Vec<_>>());
        let second = directory.write("second", &(100_u8..164).collect::<Vec<_>>());
        let plan = ReadPlan::open(
            [
                Extent::new(&first, 4, BufferKind::Quantized, 0, 20),
                Extent::new(&second, 8, BufferKind::Quantized, 20, 16),
                Extent::new(&second, 30, BufferKind::Scales, 0, 12),
                Extent::zero(BufferKind::Other, 0, 3),
                Extent::new(&first, 40, BufferKind::Other, 3, 8),
            ],
            BufferLengths::new(36, 12, 11),
            7,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(4).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(&buffers.quantized()[..20], &(4_u8..24).collect::<Vec<_>>());
        assert_eq!(
            &buffers.quantized()[20..],
            &(108_u8..124).collect::<Vec<_>>()
        );
        assert_eq!(buffers.scales(), &(130_u8..142).collect::<Vec<_>>());
        assert_eq!(&buffers.other()[..3], &[0, 0, 0]);
        assert_eq!(&buffers.other()[3..], &(40_u8..48).collect::<Vec<_>>());
        assert_eq!(stats.bytes, 56);
        assert_eq!(stats.jobs, 11);
        assert_eq!(stats.workers, 4);
        assert_eq!(
            buffers.pointer(BufferKind::Quantized) as usize % BUFFER_ALIGNMENT,
            0
        );
        assert_eq!(
            buffers.allocation_lengths(),
            BufferLengths::new(BUFFER_ALIGNMENT, BUFFER_ALIGNMENT, BUFFER_ALIGNMENT)
        );
    }

    #[test]
    fn replacement_admission_is_zero_only_for_a_free_fitting_slot() {
        let small = BufferLengths::new(0, 0, 100);
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        assert_eq!(
            reader.replacement_admission_bytes(small).unwrap(),
            (3 * BUFFER_ALIGNMENT) as u64
        );
        let plan = ReadPlan::open(
            [Extent::zero(BufferKind::Other, 0, 100)],
            small,
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let (buffers, _) = reader.read(&plan).unwrap();
        // A busy slot is unknown to a future waiter, even when its observed
        // capacity happens to fit, so admission remains conservative.
        assert_eq!(
            reader.replacement_admission_bytes(small).unwrap(),
            (3 * BUFFER_ALIGNMENT) as u64
        );
        drop(buffers);
        assert_eq!(reader.replacement_admission_bytes(small).unwrap(), 0);

        let growth = BufferLengths::new(0, 0, BUFFER_ALIGNMENT + 1);
        assert_eq!(
            reader.replacement_admission_bytes(growth).unwrap(),
            (4 * BUFFER_ALIGNMENT) as u64
        );
    }

    #[test]
    fn explicit_capacity_reservation_grows_once_and_failure_leaves_a_free_slot() {
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        let small = BufferLengths::new(0, 0, 100);
        reader.reserve_capacity(small).unwrap();
        assert_eq!(reader.replacement_admission_bytes(small).unwrap(), 0);

        let large = BufferLengths::new(0, 0, BUFFER_ALIGNMENT + 1);
        reader.reserve_capacity(large).unwrap();
        assert_eq!(reader.replacement_admission_bytes(large).unwrap(), 0);
        reader.reserve_capacity(large).unwrap();
        assert_eq!(reader.replacement_admission_bytes(large).unwrap(), 0);

        assert!(
            reader
                .reserve_capacity(BufferLengths::new(0, 0, usize::MAX))
                .is_err()
        );
        assert!(reader.replacement_admission_bytes(small).is_ok());
    }

    #[test]
    fn wide_scale4_deferred_bounds_match_the_exact_nine_row_ceiling() {
        const ROUTED_EXPERTS_PER_ROW: usize = 16;
        const MAXIMUM_ROWS: usize = 9;
        let experts = ROUTED_EXPERTS_PER_ROW * MAXIMUM_ROWS;
        assert_eq!(MAX_DEFERRED_AUTHENTICATED_SOURCES, experts + 1);
        assert_eq!(MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS, experts * 2);
        assert_eq!(MAX_DEFERRED_MANIFEST_SOURCES, 64);
        assert!(MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS < MAX_PLAN_JOBS);
        assert!(MAX_VECTORED_DESTINATIONS >= 4);
    }

    #[test]
    fn authenticated_gather_requalifies_each_deferred_descriptor_identity() {
        let directory = TestDirectory::new();
        let original = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let path = directory.write("authenticated", &original);
        let plan = ReadPlan::open_deferred_authenticated(
            [
                Extent::new(&path, 2, BufferKind::Other, 0, 2),
                Extent::new(&path, 6, BufferKind::Other, 2, 2),
            ],
            [DeferredSourceVerification::new(
                &path,
                original.len() as u64,
                0,
                original.len(),
                crate::packfile::digest_bytes(&original),
            )],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.persistent_source_count(), 0);
        assert_eq!(plan.jobs(), 1);
        let reader = Reader::with_arena_capacity(2, 1).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), &[3, 4, 7, 8]);
        assert_eq!(stats.jobs, 1);
        drop(buffers);

        // A second admission reopens and reauthenticates the path. It may not
        // inherit the first descriptor's successful digest bit. Corrupt a gap
        // that is authenticated but not copied to prove the one-pass gather
        // still hashes every byte in the enclosing contract.
        fs::write(&path, [1_u8, 2, 3, 4, 0xff, 6, 7, 8]).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("replacement authenticated source was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("failed SHA-256 verification"));
    }

    #[test]
    fn authenticated_gather_rejects_an_extent_outside_its_digest_range() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("partially-authenticated", &bytes);
        let error = ReadPlan::open_deferred_authenticated(
            [Extent::new(&path, 2, BufferKind::Other, 0, 2)],
            [DeferredSourceVerification::new(
                &path,
                bytes.len() as u64,
                0,
                2,
                crate::packfile::digest_bytes(&bytes[..2]),
            )],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("outside every authenticated range")
        );
    }

    #[test]
    fn authenticated_gather_rejects_ambiguous_overlapping_digest_ranges() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4, 5, 6];
        let path = directory.write("overlapping-authentication", &bytes);
        let error = ReadPlan::open_deferred_authenticated(
            [Extent::new(&path, 2, BufferKind::Other, 0, 2)],
            [
                DeferredSourceVerification::new(
                    &path,
                    bytes.len() as u64,
                    0,
                    4,
                    crate::packfile::digest_bytes(&bytes[..4]),
                ),
                DeferredSourceVerification::new(
                    &path,
                    bytes.len() as u64,
                    2,
                    4,
                    crate::packfile::digest_bytes(&bytes[2..]),
                ),
            ],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(error.to_string().contains("ranges overlap"));
    }

    #[test]
    fn authenticated_gather_rejects_a_second_per_extent_digest_contract() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("double-authentication", &bytes);
        let error = ReadPlan::open_deferred_authenticated(
            [Extent::verified(
                &path,
                1,
                BufferKind::Other,
                0,
                2,
                [0xff; 32],
            )],
            [DeferredSourceVerification::new(
                &path,
                bytes.len() as u64,
                0,
                bytes.len(),
                crate::packfile::digest_bytes(&bytes),
            )],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(error.to_string().contains("per-extent digest"));
    }

    #[test]
    fn rejects_overlapping_or_out_of_file_extents_before_workers_start() {
        let directory = TestDirectory::new();
        let path = directory.write("weights", &[1, 2, 3, 4]);
        assert!(
            ReadPlan::open(
                [
                    Extent::new(&path, 0, BufferKind::Other, 0, 3),
                    Extent::new(&path, 1, BufferKind::Other, 2, 2),
                ],
                BufferLengths::new(0, 0, 4),
                0,
                CachePolicy::Resident,
            )
            .unwrap_err()
            .to_string()
            .contains("overlapping")
        );
        assert!(
            ReadPlan::open(
                [Extent::new(&path, 3, BufferKind::Other, 0, 2)],
                BufferLengths::new(0, 0, 2),
                0,
                CachePolicy::Resident,
            )
            .unwrap_err()
            .to_string()
            .contains("exceeds")
        );
    }

    #[test]
    fn exact_length_contract_rejects_an_oversized_source() {
        let directory = TestDirectory::new();
        let path = directory.write("weights", &[1, 2, 3, 4, 5]);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        assert!(plan.require_all_sources_exact_length(5).is_ok());
        assert!(
            plan.require_all_sources_exact_length(4)
                .unwrap_err()
                .to_string()
                .contains("canonical length is 4")
        );
    }

    #[test]
    fn deferred_exact_sources_open_in_workers_without_persistent_descriptors() {
        let directory = TestDirectory::new();
        let first = directory.write("first-deferred", &[1, 2, 3, 4]);
        let second = directory.write("second-deferred", &[5, 6, 7, 8]);
        let plan = ReadPlan::open_deferred_exact(
            [
                Extent::new(&first, 0, BufferKind::Other, 0, 4),
                Extent::new(&second, 0, BufferKind::Other, 4, 4),
            ],
            BufferLengths::new(0, 0, 8),
            0,
            CachePolicy::Resident,
            4,
        )
        .unwrap();
        assert_eq!(plan.source_count(), 2);
        assert_eq!(plan.persistent_source_count(), 0);

        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(stats.jobs, 2);
    }

    #[test]
    fn deferred_exact_source_validates_length_on_the_open_descriptor() {
        let directory = TestDirectory::new();
        let path = directory.write("wrong-length", &[1, 2, 3, 4, 5]);
        let plan = ReadPlan::open_deferred_exact(
            [Extent::new(&path, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
            4,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("wrong-sized deferred source was unexpectedly accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 4"));
    }

    #[test]
    fn deferred_exact_source_never_follows_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("target", &[1, 2, 3, 4]);
        let link = directory.0.join("link");
        symlink(target, &link).unwrap();
        let plan = ReadPlan::open_deferred_exact(
            [Extent::new(&link, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
            4,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("deferred source symlink was unexpectedly followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
    }

    #[test]
    fn deferred_manifest_reads_variable_whole_file_lengths_without_retained_descriptors() {
        let directory = TestDirectory::new();
        let first = directory.write("first-manifest", &[1, 2, 3]);
        let second = directory.write("second-manifest", &[4, 5, 6, 7, 8]);
        let plan = ReadPlan::open_deferred_manifest(
            [
                Extent::new(&first, 0, BufferKind::Other, 0, 3),
                Extent::new(&second, 0, BufferKind::Other, 3, 5),
            ],
            BufferLengths::new(0, 0, 8),
            2,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.source_count(), 2);
        assert_eq!(plan.persistent_source_count(), 0);
        // Do not sample the process-wide descriptor table here: Rust runs
        // unrelated storage/server tests concurrently, so their transient
        // sockets and files make that observation inherently racy. The plan's
        // zero persistent-source count is the local invariant; the ignored
        // single-test audit below proves worker descriptors close in practice.

        let reader = Reader::new(2).unwrap();
        let ticket = reader.submit(&plan, ReadPriority::Demand).unwrap();
        let completed_batch = Arc::clone(&ticket.batch);
        let (buffers, stats) = ticket.wait().unwrap();
        assert_eq!(buffers.other(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(stats.bytes, 8);
        assert_eq!(stats.jobs, 5);
        let BatchSources::Plan {
            deferred_files: Some(files),
            ..
        } = &completed_batch.sources
        else {
            panic!("manifest batch did not own deferred descriptor slots")
        };
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|slot| matches!(slot.get(), Some(Ok(_)))));
    }

    #[test]
    fn deferred_ranges_gather_partial_heterogeneous_sources_without_retained_descriptors() {
        let directory = TestDirectory::new();
        let first = directory.write("first-ranges", &[1, 2, 3, 4, 5, 6]);
        let second = directory.write("second-ranges", &[7, 8, 9, 10]);
        let plan = ReadPlan::open_deferred_ranges(
            [
                Extent::new(&first, 1, BufferKind::Other, 0, 2),
                Extent::new(&first, 4, BufferKind::Other, 2, 2),
                Extent::new(&second, 0, BufferKind::Other, 4, 1),
                Extent::new(&second, 2, BufferKind::Other, 5, 2),
            ],
            [
                DeferredSourceLength::new(&first, 6),
                DeferredSourceLength::new(&second, 4),
            ],
            BufferLengths::new(0, 0, 7),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.source_count(), 2);
        assert_eq!(plan.persistent_source_count(), 0);
        assert_eq!(plan.jobs(), 4);
        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), &[2, 3, 5, 6, 7, 9, 10]);
        assert_eq!(stats.jobs, 4);
    }

    #[test]
    fn identity_pinned_deferred_range_rejects_same_length_path_replacement() {
        let directory = TestDirectory::new();
        let original = [1_u8, 2, 3, 4];
        let replacement = [9_u8, 8, 7, 6];
        let path = directory.write("identity-pinned-range", &original);
        let source =
            DeferredSourceLength::new_with_live_identity(&path, original.len() as u64).unwrap();
        let original_identity = source.identity().unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [Extent::new(&path, 0, BufferKind::Other, 0, original.len())],
            [source],
            BufferLengths::new(0, 0, original.len()),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();

        let displaced = directory.0.join("identity-pinned-range-original");
        fs::rename(&path, displaced).unwrap();
        fs::write(&path, replacement).unwrap();
        let replacement_source =
            DeferredSourceLength::new_with_live_identity(&path, replacement.len() as u64).unwrap();
        assert_ne!(replacement_source.identity().unwrap(), original_identity);

        let error = match Reader::new(1).unwrap().read(&plan) {
            Ok(_) => panic!("same-length replacement satisfied an identity-pinned range"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("identity changed before range gather")
        );

        // Identity pinning is opt-in: the original exact-length contract still
        // admits the replacement and reads from the descriptor opened by the
        // worker for this batch.
        let unpinned = ReadPlan::open_deferred_ranges(
            [Extent::new(
                &path,
                0,
                BufferKind::Other,
                0,
                replacement.len(),
            )],
            [DeferredSourceLength::new(&path, replacement.len() as u64)],
            BufferLengths::new(0, 0, replacement.len()),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let (buffers, _) = Reader::new(1).unwrap().read(&unpinned).unwrap();
        assert_eq!(buffers.other(), replacement);
    }

    #[test]
    fn identity_pinned_deferred_range_rechecks_descriptor_after_read_completion() {
        use std::os::unix::fs::FileExt;

        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("identity-pinned-tail-check", &bytes);
        let source =
            DeferredSourceLength::new_with_live_identity(&path, bytes.len() as u64).unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [Extent::new(&path, 0, BufferKind::Other, 0, bytes.len())],
            [source],
            BufferLengths::new(0, 0, bytes.len()),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let ticket = reader.submit(&plan, ReadPriority::Demand).unwrap();
        while !ticket.is_ready() {
            std::thread::yield_now();
        }

        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        assert_eq!(file.write_at(&[9], 0).unwrap(), 1);
        file.sync_all().unwrap();
        let error = match ticket.wait() {
            Ok(_) => panic!("post-read mutation escaped the descriptor identity tail check"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("identity changed during range gather")
        );
    }

    #[test]
    fn deferred_vectored_read_scatters_one_contiguous_source_range_in_one_job() {
        let directory = TestDirectory::new();
        let path = directory.write("vectored-range", &(0_u8..16).collect::<Vec<_>>());
        let extent = Extent::vectored(
            &path,
            2,
            [
                VectoredDestination::new(BufferKind::Other, 0, 3),
                VectoredDestination::new(BufferKind::Scales, 0, 2),
                VectoredDestination::new(BufferKind::Other, 3, 2),
                VectoredDestination::new(BufferKind::Quantized, 0, 1),
            ],
        )
        .unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [extent],
            [DeferredSourceLength::new(&path, 16)],
            BufferLengths::new(1, 2, 5),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.jobs(), 1);
        assert_eq!(plan.logical_bytes(), 8);

        let (buffers, stats) = Reader::new(2).unwrap().read(&plan).unwrap();
        assert_eq!(buffers.quantized(), &[9]);
        assert_eq!(buffers.scales(), &[5, 6]);
        assert_eq!(buffers.other(), &[2, 3, 4, 7, 8]);
        assert_eq!(stats.jobs, 1);
        assert_eq!(stats.bytes, 8);
        assert_eq!(stats.workers, 1);
    }

    #[test]
    fn verified_vectored_read_hashes_destinations_in_contiguous_source_order() {
        let directory = TestDirectory::new();
        let bytes: Vec<_> = (0_u8..16).collect();
        let path = directory.write("verified-vectored-range", &bytes);
        let extent = Extent::vectored_verified(
            &path,
            2,
            [
                VectoredDestination::new(BufferKind::Other, 0, 3),
                VectoredDestination::new(BufferKind::Scales, 0, 2),
                VectoredDestination::new(BufferKind::Other, 3, 2),
                VectoredDestination::new(BufferKind::Quantized, 0, 1),
            ],
            crate::packfile::digest_bytes(&bytes[2..10]),
        )
        .unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [extent],
            [DeferredSourceLength::new(&path, bytes.len() as u64)],
            BufferLengths::new(1, 2, 5),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();

        let (buffers, stats) = Reader::new(2).unwrap().read(&plan).unwrap();
        assert_eq!(buffers.quantized(), &[9]);
        assert_eq!(buffers.scales(), &[5, 6]);
        assert_eq!(buffers.other(), &[2, 3, 4, 7, 8]);
        assert_eq!(stats.jobs, 1);
        drop(buffers);

        // The same deferred plan reopens its source on each batch. A previous
        // successful digest must never qualify a later descriptor generation.
        let mut changed = bytes;
        changed[6] ^= 0xff;
        fs::write(&path, changed).unwrap();
        let error = match Reader::new(2).unwrap().read(&plan) {
            Ok(_) => panic!("same-plan vectored reread reused a stale digest result"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("failed SHA-256 verification"));
    }

    #[test]
    fn verified_vectored_read_rejects_corruption_before_publication() {
        let directory = TestDirectory::new();
        let original: Vec<_> = (0_u8..16).collect();
        let path = directory.write("corrupt-verified-vectored", &original);
        let extent = Extent::vectored_verified(
            &path,
            2,
            [
                VectoredDestination::new(BufferKind::Other, 0, 3),
                VectoredDestination::new(BufferKind::Other, 3, 5),
            ],
            crate::packfile::digest_bytes(&original[2..10]),
        )
        .unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [extent],
            [DeferredSourceLength::new(&path, original.len() as u64)],
            BufferLengths::new(0, 0, 8),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let mut corrupted = original;
        corrupted[6] ^= 0xff;
        fs::write(&path, corrupted).unwrap();

        let error = match Reader::new(2).unwrap().read(&plan) {
            Ok(_) => panic!("corrupted verified vectored source was published"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("failed SHA-256 verification"));
    }

    #[test]
    fn deferred_vectored_read_retains_live_exact_length_and_no_follow_contracts() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("vectored-target", &[1, 2, 3, 4, 5]);
        let link = directory.0.join("vectored-link");
        symlink(&target, &link).unwrap();
        let make_extent = |path: &Path| {
            Extent::vectored(
                path,
                1,
                [
                    VectoredDestination::new(BufferKind::Other, 0, 1),
                    VectoredDestination::new(BufferKind::Other, 1, 2),
                ],
            )
            .unwrap()
        };

        let wrong_length = ReadPlan::open_deferred_ranges(
            [make_extent(&target)],
            [DeferredSourceLength::new(&target, 4)],
            BufferLengths::new(0, 0, 3),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let error = match Reader::new(1).unwrap().read(&wrong_length) {
            Ok(_) => panic!("wrong-sized vectored source was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 4"));

        let symlinked = ReadPlan::open_deferred_ranges(
            [make_extent(&link)],
            [DeferredSourceLength::new(&link, 5)],
            BufferLengths::new(0, 0, 3),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let error = match Reader::new(1).unwrap().read(&symlinked) {
            Ok(_) => panic!("vectored source symlink was followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
    }

    #[test]
    fn deferred_range_plan_rejects_manually_constructed_empty_vectored_extents() {
        let directory = TestDirectory::new();
        let path = directory.write("malformed-vectored", &[1, 2, 3, 4]);
        for destinations in [
            Vec::new().into_boxed_slice(),
            vec![VectoredDestination::new(BufferKind::Other, 0, 0)].into_boxed_slice(),
        ] {
            let error = ReadPlan::open_deferred_ranges(
                [Extent::ReadVectored {
                    path: path.clone(),
                    source_offset: 0,
                    destinations,
                    expected_digest: Some(crate::packfile::digest_bytes(&[])),
                }],
                [DeferredSourceLength::new(&path, 4)],
                BufferLengths::new(0, 0, 0),
                0,
                CachePolicy::Streaming,
            )
            .unwrap_err();
            assert!(error.to_string().contains("non-empty destinations"));
        }
    }

    #[test]
    fn deferred_ranges_reject_missing_duplicate_unused_and_wrong_length_contracts() {
        let directory = TestDirectory::new();
        let path = directory.write("range-contract", &[1, 2, 3, 4]);
        let other = directory.write("unused-range-contract", &[5, 6]);
        let extent = || Extent::new(&path, 1, BufferKind::Other, 0, 2);

        let missing = ReadPlan::open_deferred_ranges(
            [extent()],
            [DeferredSourceLength::new(&other, 2)],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(missing.to_string().contains("no exact-length contract"));

        let duplicate = ReadPlan::open_deferred_ranges(
            [extent()],
            [
                DeferredSourceLength::new(&path, 4),
                DeferredSourceLength::new(&path, 4),
            ],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("declared more than once"));

        let unused = ReadPlan::open_deferred_ranges(
            [extent()],
            [
                DeferredSourceLength::new(&path, 4),
                DeferredSourceLength::new(&other, 2),
            ],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(unused.to_string().contains("unused source contract"));

        let plan = ReadPlan::open_deferred_ranges(
            [extent()],
            [DeferredSourceLength::new(&path, 5)],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let error = match Reader::new(1).unwrap().read(&plan) {
            Ok(_) => panic!("wrong-sized deferred range source was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 5"));
    }

    #[test]
    fn persistent_manifest_opens_once_and_pins_the_validated_inode() {
        let directory = TestDirectory::new();
        let original = [1_u8, 2, 3, 4];
        let path = directory.write("persistent-manifest", &original);
        let plan = ReadPlan::open_persistent_deferred_manifest(
            [Extent::new(&path, 0, BufferKind::Other, 0, original.len())],
            BufferLengths::new(0, 0, original.len()),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        assert_eq!(plan.persistent_source_count(), 1);
        assert_eq!(plan.opened_persistent_source_count(), 0);

        let reader = Reader::new(1).unwrap();
        let (first, _) = reader.read(&plan).unwrap();
        assert_eq!(first.other(), original);
        drop(first);
        assert_eq!(plan.opened_persistent_source_count(), 1);

        let displaced = directory.0.join("persistent-manifest-original");
        fs::rename(&path, &displaced).unwrap();
        fs::write(&path, [9_u8, 9, 9, 9]).unwrap();
        let (second, _) = reader.read(&plan).unwrap();
        assert_eq!(second.other(), original);
        drop(second);
        assert_eq!(plan.opened_persistent_source_count(), 1);
    }

    #[test]
    fn persistent_manifest_first_open_never_follows_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("persistent-target", &[1, 2, 3, 4]);
        let link = directory.0.join("persistent-link");
        symlink(target, &link).unwrap();
        let plan = ReadPlan::open_persistent_deferred_manifest(
            [Extent::new(&link, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("persistent manifest unexpectedly followed a symlink"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
        assert_eq!(plan.opened_persistent_source_count(), 0);
    }

    #[test]
    fn deferred_manifest_validates_each_live_source_length() {
        let directory = TestDirectory::new();
        let oversized = directory.write("oversized-manifest", &[1, 2, 3, 4, 5]);
        let short = directory.write("short-manifest", &[1, 2, 3]);
        let reader = Reader::new(2).unwrap();
        for path in [&oversized, &short] {
            let plan = ReadPlan::open_deferred_manifest(
                [Extent::new(path, 0, BufferKind::Other, 0, 4)],
                BufferLengths::new(0, 0, 4),
                0,
                CachePolicy::Resident,
            )
            .unwrap();
            let error = match reader.read(&plan) {
                Ok(_) => panic!("wrong-sized manifest source was unexpectedly accepted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("expected exact length 4"));
        }
    }

    #[test]
    fn deferred_manifest_rejects_partial_or_conflicting_source_contracts() {
        let directory = TestDirectory::new();
        let path = directory.0.join("need-not-exist");
        let partial = ReadPlan::open_deferred_manifest(
            [Extent::new(&path, 1, BufferKind::Other, 0, 3)],
            BufferLengths::new(0, 0, 3),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(partial.to_string().contains("offset zero"));

        let conflicting = ReadPlan::open_deferred_manifest(
            [
                Extent::new(&path, 0, BufferKind::Other, 0, 3),
                Extent::new(&path, 0, BufferKind::Other, 3, 4),
            ],
            BufferLengths::new(0, 0, 7),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(
            conflicting
                .to_string()
                .contains("conflicting whole-file lengths")
        );
    }

    #[test]
    fn deferred_manifest_has_a_hard_per_batch_descriptor_bound() {
        let directory = TestDirectory::new();
        let extents = (0..=MAX_DEFERRED_MANIFEST_SOURCES)
            .map(|index| {
                Extent::new(
                    directory.0.join(format!("source-{index}")),
                    0,
                    BufferKind::Other,
                    index,
                    1,
                )
            })
            .collect::<Vec<_>>();
        let error = ReadPlan::open_deferred_manifest(
            extents,
            BufferLengths::new(0, 0, MAX_DEFERRED_MANIFEST_SOURCES + 1),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(error.to_string().contains("bounded maximum is 64"));
    }

    #[test]
    fn deferred_manifest_never_follows_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("manifest-target", &[1, 2, 3, 4]);
        let link = directory.0.join("manifest-link");
        symlink(target, &link).unwrap();
        let plan = ReadPlan::open_deferred_manifest(
            [Extent::new(&link, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("deferred manifest symlink was unexpectedly followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
    }

    #[test]
    fn deferred_catalog_reads_integer_selected_sources_in_adjacent_order() {
        let directory = TestDirectory::new();
        directory.write("first.bin", &[1, 2, 3, 4]);
        directory.write("second.bin", &[5, 6, 7, 8]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [
                DeferredSourceName::new("first.bin").unwrap(),
                DeferredSourceName::new("second.bin").unwrap(),
            ],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        assert_eq!(catalog.source_count(), 2);
        assert_eq!(catalog.exact_source_length(), 4);
        assert_eq!(catalog.source_name(0), Some("first.bin"));

        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader
            .read_deferred_exact(&catalog, &[1, 0], BufferKind::Other)
            .unwrap();
        assert_eq!(buffers.other(), &[5, 6, 7, 8, 1, 2, 3, 4]);
        assert_eq!(stats.bytes, 8);
        assert_eq!(stats.jobs, 2);
        assert_eq!(stats.workers, 2);
    }

    #[test]
    fn deferred_catalog_rejects_unsafe_names_and_bounds_requests() {
        assert!(DeferredSourceName::new("").is_err());
        assert!(DeferredSourceName::new(".").is_err());
        assert!(DeferredSourceName::new("..").is_err());
        assert!(DeferredSourceName::new("nested/file").is_err());
        assert!(DeferredSourceName::new(&"x".repeat(32)).is_err());

        let directory = TestDirectory::new();
        directory.write("only.bin", &[1, 2, 3, 4]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("only.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        assert!(
            reader
                .submit_deferred_exact(&catalog, &[], BufferKind::Other, ReadPriority::Demand)
                .is_err()
        );
        assert!(
            reader
                .submit_deferred_exact(&catalog, &[1], BufferKind::Other, ReadPriority::Demand)
                .is_err()
        );
        assert!(
            reader
                .submit_deferred_exact(
                    &catalog,
                    &[0; MAX_INLINE_DEFERRED_FILES + 1],
                    BufferKind::Other,
                    ReadPriority::Demand,
                )
                .is_err()
        );
    }

    #[test]
    fn deferred_catalog_openat_never_follows_source_or_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        directory.write("target.bin", &[1, 2, 3, 4]);
        symlink(directory.0.join("target.bin"), directory.0.join("link.bin")).unwrap();
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("link.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read_deferred_exact(&catalog, &[0], BufferKind::Other) {
            Ok(_) => panic!("catalog source symlink was unexpectedly followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));

        let parent = TestDirectory::new();
        let directory_link = parent.0.join("directory-link");
        symlink(&directory.0, &directory_link).unwrap();
        assert!(
            DeferredExactCatalog::open(
                &directory_link,
                [DeferredSourceName::new("target.bin").unwrap()],
                4,
                CachePolicy::Resident,
            )
            .is_err()
        );
    }

    #[test]
    fn deferred_catalog_validates_exact_length_on_the_openat_descriptor() {
        let directory = TestDirectory::new();
        directory.write("wrong.bin", &[1, 2, 3, 4, 5]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("wrong.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read_deferred_exact(&catalog, &[0], BufferKind::Other) {
            Ok(_) => panic!("wrong-sized catalog source was unexpectedly accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 4"));
    }

    #[test]
    #[ignore = "global descriptor-count audit; run this test alone"]
    fn deferred_catalog_closes_every_ephemeral_source_descriptor() {
        let directory = TestDirectory::new();
        directory.write("source.bin", &[1, 2, 3, 4]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("source.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(2).unwrap();
        let before = count_open_descriptors().expect("supported host exposes descriptor count");
        for _ in 0..64 {
            let (buffers, _) = reader
                .read_deferred_exact(&catalog, &[0], BufferKind::Other)
                .unwrap();
            assert_eq!(buffers.other(), &[1, 2, 3, 4]);
            drop(buffers);
        }
        let after = count_open_descriptors().expect("supported host exposes descriptor count");
        assert_eq!(after, before, "catalog source descriptor leaked");
    }

    #[test]
    fn authenticated_extent_hashes_the_first_read_without_a_second_io_pass() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(32 * 1024).collect();
        let path = directory.write("authenticated", &bytes);
        let plan = ReadPlan::open(
            [Extent::verified(
                &path,
                0,
                BufferKind::Other,
                0,
                bytes.len(),
                crate::packfile::digest_bytes(&bytes),
            )],
            BufferLengths::new(0, 0, bytes.len()),
            // A verified range is already one canonical integrity unit and
            // must not be silently split by the ordinary scheduling chunk.
            1,
            CachePolicy::Resident,
        )
        .unwrap();
        assert_eq!(plan.jobs(), 1);
        assert!(!plan.verified_extents[0].load(Ordering::Acquire));

        let reader = Reader::new(2).unwrap();
        let (buffers, _) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), bytes);
        assert!(plan.verified_extents[0].load(Ordering::Acquire));

        // A later read through the same immutable plan and opened descriptor
        // reuses the successful first-read qualification.
        let (again, _) = reader.read(&plan).unwrap();
        assert_eq!(again.other(), bytes);
    }

    #[test]
    fn authenticated_extent_never_publishes_bytes_with_the_wrong_digest() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("corrupt", &bytes);
        let plan = ReadPlan::open(
            [Extent::verified(
                &path,
                0,
                BufferKind::Other,
                0,
                bytes.len(),
                [0; 32],
            )],
            BufferLengths::new(0, 0, bytes.len()),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("wrong authenticated bytes were unexpectedly published"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("SHA-256"));
        assert!(!plan.verified_extents[0].load(Ordering::Acquire));
    }

    #[test]
    fn rejects_implicit_gaps_and_missing_trailing_bytes() {
        let directory = TestDirectory::new();
        let path = directory.write("weights", &[1, 2, 3, 4]);
        let gap = ReadPlan::open(
            [
                Extent::new(&path, 0, BufferKind::Other, 0, 2),
                Extent::new(&path, 3, BufferKind::Other, 3, 1),
            ],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(gap.to_string().contains("Extent::zero"));

        let trailing = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Other, 0, 3)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(trailing.to_string().contains("3..4"));
    }

    #[test]
    fn reuses_fixed_workers_across_many_batches() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(32 * 1024).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            1024,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(3, 1).unwrap();
        let mut arena_pointer = None;
        for _ in 0..20 {
            let (result, stats) = reader.read(&plan).unwrap();
            assert_eq!(result.quantized(), bytes);
            assert_eq!(stats.workers, 3);
            let pointer = result.pointer(BufferKind::Quantized);
            assert_eq!(*arena_pointer.get_or_insert(pointer), pointer);
        }
    }

    #[test]
    fn empty_plan_returns_without_scheduling_workers() {
        let plan =
            ReadPlan::open([], BufferLengths::default(), 1024, CachePolicy::Resident).unwrap();
        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert!(buffers.quantized().is_empty());
        assert_eq!(stats.jobs, 0);
        assert_eq!(stats.workers, 0);
    }

    #[test]
    fn asynchronous_tickets_preserve_exact_bytes() {
        let directory = TestDirectory::new();
        let first_bytes: Vec<u8> = (0..=127).cycle().take(16 * 1024).collect();
        let second_bytes: Vec<u8> = (128..=255).cycle().take(16 * 1024).collect();
        let first = directory.write("first", &first_bytes);
        let second = directory.write("second", &second_bytes);
        let first_plan = ReadPlan::open(
            [Extent::new(
                &first,
                0,
                BufferKind::Quantized,
                0,
                first_bytes.len(),
            )],
            BufferLengths::new(first_bytes.len(), 0, 0),
            1024,
            CachePolicy::Resident,
        )
        .unwrap();
        let second_plan = ReadPlan::open(
            [Extent::new(
                &second,
                0,
                BufferKind::Quantized,
                0,
                second_bytes.len(),
            )],
            BufferLengths::new(second_bytes.len(), 0, 0),
            1024,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(2, 2).unwrap();
        let prefetch = reader.submit(&first_plan, ReadPriority::Prefetch).unwrap();
        let demand = reader.submit(&second_plan, ReadPriority::Demand).unwrap();
        let (demand_buffers, _) = demand.wait().unwrap();
        let (prefetch_buffers, _) = prefetch.wait().unwrap();
        assert_eq!(demand_buffers.quantized(), second_bytes);
        assert_eq!(prefetch_buffers.quantized(), first_bytes);
    }

    #[test]
    fn arena_is_bounded_and_returns_a_slot_when_the_cpu_lease_drops() {
        let arena = BufferArena::new(1).unwrap();
        let lengths = BufferLengths::new(64, 8, 0);
        let first = arena
            .acquire(lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let first_pointer = first.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        assert!(
            arena
                .acquire(lengths, false, ReadPriority::Demand)
                .unwrap()
                .is_none()
        );
        drop(first);
        let second = arena
            .acquire(lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            second.buffers().get(BufferKind::Quantized).pointer.as_ptr(),
            first_pointer
        );
    }

    #[test]
    fn arena_retire_hook_skips_initial_and_fitting_allocations() {
        let calls = Arc::new(AtomicUsize::new(0));
        let hook_calls = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            hook_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let first = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let pointer = first.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        drop(first);

        let fitting = arena
            .acquire(BufferLengths::new(32, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            fitting
                .buffers()
                .get(BufferKind::Quantized)
                .pointer
                .as_ptr(),
            pointer
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        drop(fitting);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn arena_retire_hook_runs_before_the_old_slab_final_arc_drops() {
        let retired = Arc::new(Mutex::new(None::<Weak<SharedBuffers>>));
        let retired_for_hook = Arc::clone(&retired);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_hook = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            calls_for_hook.fetch_add(1, Ordering::SeqCst);
            assert!(
                retired_for_hook
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(Weak::upgrade)
                    .is_some(),
                "retirement hook must run while the old slab is still owned"
            );
            Ok(())
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let old = Arc::downgrade(initial.buffers.as_ref().unwrap());
        *retired.lock().unwrap() = Some(old.clone());
        drop(initial);

        let grown = arena
            .acquire(BufferLengths::new(128, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(old.upgrade().is_none());
        // Do not make final arena teardown inspect the already-retired weak
        // pointer; this test is solely about the growth boundary.
        *retired.lock().unwrap() = Some(Arc::downgrade(grown.buffers.as_ref().unwrap()));
        drop(grown);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn arena_retire_hook_error_restores_the_exact_slab_and_slot() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_hook = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            if calls_for_hook.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(DeltafinError::new("injected cache flush failure"))
            } else {
                Ok(())
            }
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let pointer = initial
            .buffers()
            .get(BufferKind::Quantized)
            .pointer
            .as_ptr();
        let old = Arc::downgrade(initial.buffers.as_ref().unwrap());
        drop(initial);

        let error = match arena.acquire(BufferLengths::new(128, 0, 0), false, ReadPriority::Demand)
        {
            Err(error) => error,
            Ok(_) => panic!("injected retirement-hook error should reject arena growth"),
        };
        assert!(error.to_string().contains("injected cache flush failure"));
        assert!(old.upgrade().is_some());
        let restored = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            restored
                .buffers()
                .get(BufferKind::Quantized)
                .pointer
                .as_ptr(),
            pointer
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(restored);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn arena_retire_hook_panic_is_caught_and_restores_the_slot() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_hook = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            if calls_for_hook.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("injected cache flush panic");
            }
            Ok(())
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let pointer = initial
            .buffers()
            .get(BufferKind::Quantized)
            .pointer
            .as_ptr();
        drop(initial);

        let error = match arena.acquire(BufferLengths::new(128, 0, 0), false, ReadPriority::Demand)
        {
            Err(error) => error,
            Ok(_) => panic!("injected retirement-hook panic should reject arena growth"),
        };
        assert!(error.to_string().contains("hook panicked"));
        let restored = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            restored
                .buffers()
                .get(BufferKind::Quantized)
                .pointer
                .as_ptr(),
            pointer
        );
        drop(restored);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn descriptor_budget_rejects_before_overcommit_and_releases_exactly() {
        let budget = DescriptorBudget::fixed(2);
        let reservation = budget.reserve(2).unwrap();
        let error = budget.reserve(1).unwrap_err();
        assert!(error.to_string().contains("only 0 remain"));
        assert_eq!(*budget.in_use.lock().unwrap(), 2);
        drop(reservation);
        assert_eq!(*budget.in_use.lock().unwrap(), 0);
        let replacement = budget.reserve(2).unwrap();
        assert_eq!(*budget.in_use.lock().unwrap(), 2);
        drop(replacement);
    }

    #[test]
    fn arena_reuses_the_smallest_fitting_free_slot() {
        let arena = BufferArena::new(2).unwrap();
        let small_lengths = BufferLengths::new(64, 0, 0);
        let large_lengths = BufferLengths::new(4096, 0, 0);

        let small = arena
            .acquire(small_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let small_pointer = small.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        // Slot zero remains occupied, forcing this allocation into slot one.
        let large = arena
            .acquire(large_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let large_pointer = large.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        assert_ne!(small_pointer, large_pointer);
        drop((small, large));

        // Both slots are now free. First-free selection would grow slot zero;
        // best-fit selection must reuse slot one's already sufficient slab.
        let reused = arena
            .acquire(large_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            reused.buffers().get(BufferKind::Quantized).pointer.as_ptr(),
            large_pointer
        );
    }

    #[test]
    fn arena_retires_an_undersized_slab_before_growth_allocation() {
        let arena = BufferArena::new(1).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let retired = Arc::downgrade(initial.buffers.as_ref().unwrap());
        drop(initial);

        // This fails before attempting a real allocation, after the reserved
        // slot has retired its old buffers. Retaining the old slab until a new
        // one succeeded would leave this Weak reference live.
        assert!(
            arena
                .acquire(
                    BufferLengths::new(usize::MAX, 0, 0),
                    false,
                    ReadPriority::Demand,
                )
                .is_err()
        );
        assert!(retired.upgrade().is_none());

        let retry = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(retry.lengths.quantized, 64);
    }

    #[test]
    fn prefetch_cannot_consume_the_last_demand_arena_slot() {
        let arena = BufferArena::new(2).unwrap();
        let lengths = BufferLengths::new(64, 0, 0);
        let prefetch = arena
            .acquire(lengths, false, ReadPriority::Prefetch)
            .unwrap()
            .unwrap();
        assert!(
            arena
                .acquire(lengths, false, ReadPriority::Prefetch)
                .unwrap()
                .is_none()
        );
        let demand = arena
            .acquire(lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        drop((demand, prefetch));
    }

    #[test]
    fn demand_preempts_prefetch_but_prefetch_cannot_starve() {
        let mut queues = PriorityQueues::new();
        queues.push(ReadPriority::Prefetch, 10_000);
        for demand in 0..(MAX_DEMAND_STREAK + 2) {
            queues.push(ReadPriority::Demand, demand);
        }
        for expected in 0..MAX_DEMAND_STREAK {
            assert_eq!(queues.pop(), Some(expected));
        }
        assert_eq!(queues.pop(), Some(10_000));
        assert_eq!(queues.pop(), Some(MAX_DEMAND_STREAK));

        let mut preemption = PriorityQueues::new();
        preemption.push(ReadPriority::Prefetch, 1);
        preemption.push(ReadPriority::Prefetch, 2);
        preemption.push(ReadPriority::Demand, 3);
        assert_eq!(preemption.pop(), Some(3));
        assert_eq!(preemption.pop(), Some(1));
    }

    #[test]
    fn worker_executes_only_a_bounded_quantum_before_requeue() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..(WORK_QUANTUM * 2 + 1) as u8).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            1,
            CachePolicy::Resident,
        )
        .unwrap();
        let arena = BufferArena::new(1).unwrap();
        let lease = arena
            .acquire(plan.buffer_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let batch = Batch::new(&plan, Arc::clone(&lease), ReadPriority::Prefetch);
        assert!(matches!(batch.run_quantum(), QuantumOutcome::Requeue));
        assert_eq!(batch.next_job.load(Ordering::Relaxed), WORK_QUANTUM);
        assert_eq!(
            batch.completion.remaining.load(Ordering::Relaxed),
            bytes.len() - WORK_QUANTUM
        );
        while matches!(batch.run_quantum(), QuantumOutcome::Requeue) {}
        batch.wait().unwrap();
        assert_eq!(
            lease
                .buffers()
                .get(BufferKind::Quantized)
                .as_slice(bytes.len()),
            bytes
        );
    }

    #[test]
    fn cancellation_claims_every_unstarted_job_before_arena_reuse() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (1..=(WORK_QUANTUM * 2 + 1) as u8).collect();
        let path = directory.write("cancel-weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            1,
            CachePolicy::Resident,
        )
        .unwrap();
        let arena = BufferArena::new(1).unwrap();
        let lease = arena
            .acquire(plan.buffer_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let batch = Batch::new(&plan, Arc::clone(&lease), ReadPriority::Prefetch);
        assert!(matches!(batch.run_quantum(), QuantumOutcome::Requeue));
        batch.cancel_unclaimed();
        assert_eq!(batch.completion.remaining.load(Ordering::Acquire), 0);
        assert!(matches!(batch.run_quantum(), QuantumOutcome::Idle));
        assert!(batch.wait().unwrap_err().to_string().contains("cancelled"));
        let stored = lease
            .buffers()
            .get(BufferKind::Quantized)
            .as_slice(bytes.len());
        assert_eq!(&stored[..WORK_QUANTUM], &bytes[..WORK_QUANTUM]);
        assert!(stored[WORK_QUANTUM..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn shutdown_drains_admitted_work_and_leaves_ticket_readable() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(64 * 1024).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            64,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(2, 1).unwrap();
        let ticket = reader.submit(&plan, ReadPriority::Prefetch).unwrap();
        drop(reader);
        let (buffers, _) = ticket.wait().unwrap();
        assert_eq!(buffers.quantized(), bytes);

        for _ in 0..32 {
            drop(Reader::new(2).unwrap());
        }
    }

    #[test]
    fn dropping_a_ticket_does_not_reuse_its_slot_before_io_finishes() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(128 * 1024).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            64,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(2, 1).unwrap();
        let abandoned = reader.submit(&plan, ReadPriority::Demand).unwrap();
        drop(abandoned);
        // Admission may wait for an active syscall, but it cannot observe or
        // overwrite the abandoned slot until active I/O drains and every
        // unclaimed job has been atomically cancelled.
        let (buffers, _) = reader.read(&plan).unwrap();
        assert_eq!(buffers.quantized(), bytes);
    }

    #[test]
    fn failed_read_releases_its_arena_slot() {
        let directory = TestDirectory::new();
        let broken_path = directory.write("broken", &[1, 2, 3, 4]);
        let broken = ReadPlan::open(
            [Extent::new(&broken_path, 0, BufferKind::Quantized, 0, 4)],
            BufferLengths::new(4, 0, 0),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        File::options()
            .write(true)
            .open(&broken_path)
            .unwrap()
            .set_len(0)
            .unwrap();

        let good_bytes = [9, 8, 7, 6];
        let good_path = directory.write("good", &good_bytes);
        let good = ReadPlan::open(
            [Extent::new(
                &good_path,
                0,
                BufferKind::Quantized,
                0,
                good_bytes.len(),
            )],
            BufferLengths::new(good_bytes.len(), 0, 0),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        assert!(reader.read(&broken).is_err());
        let (buffers, _) = reader.read(&good).unwrap();
        assert_eq!(buffers.quantized(), good_bytes);
    }
}
