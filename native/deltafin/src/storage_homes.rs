//! Extra drives ("homes") holding byte-identical copies of model files, and
//! the per-read drive choice that spreads storage traffic over them.
//!
//! Decode streams the whole int8 spine and every routed expert from storage
//! each pass, so one SSD's bandwidth sets the pace. A home is a directory that
//! mirrors the model root -- `k3-experts/` and/or `k3-resident-int8/tensors/`,
//! any subset of either. The model root's own drive is always device 0.
//!
//! The drive is chosen for each read job as a worker starts it, not when its
//! batch is submitted: every reader (spine, expert demand, expert prefetch)
//! charges the bytes it is reading to one shared per-drive counter, and each
//! drive's throughput is measured from the reads it completes. A job goes to
//! the holder with the earliest expected finish -- (bytes in flight there +
//! this job) / measured rate -- so the chunks of one large file spread over
//! every drive, spine load is visible to expert placement, a throttling drive
//! sheds work as it slows, and no speed has to be configured.
//! A read that fails on one drive is retried on another copy and the failing
//! drive sits out with an increasing back-off.
//!
//! Selection only changes *where* bytes come from, never which bytes: a copy
//! is admitted at startup only as a regular, non-symlink file with exactly the
//! primary's length and a modification time no older than the primary's (a
//! primary rewritten after the copy was made marks that copy stale), and every
//! open re-validates type and length on the live descriptor.
//! `deltafin populate-storage-home` makes copies and byte-compares each one.

use std::collections::HashSet;
#[cfg(target_os = "macos")]
use std::ffi::CStr;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::{DeltafinError, Result};
use crate::sys::fs::Open;

/// Device indices travel in a `u32` "already tried" mask.
pub const MAX_STORAGE_DEVICES: usize = 8;

/// Model-root-relative directories a home may mirror. Scale4 sidecars are
/// deliberately absent: their reads pin one file identity per session.
pub const MIRRORED_TREES: [&str; 2] = ["k3-experts", "k3-resident-int8/tensors"];

/// Assumed rate before a drive has been measured. Only ratios matter, and
/// the first ~64 MB a drive serves replaces it.
const DEFAULT_RATE: f64 = 2.5e9;
/// A measurement window closes after this much busy time and data.
const WINDOW_BUSY: Duration = Duration::from_millis(40);
const WINDOW_BYTES: u64 = 48 << 20;
/// Weight of the newest window in the smoothed rate.
const RATE_SMOOTHING: f64 = 0.35;
/// A low estimate that has not been re-measured relaxes back toward the
/// drive's starting rate with this time constant, so one slow spell (a
/// thermal step, a bus hiccup) cannot sideline a good drive for good: it is
/// tried again, and a drive that really is slow is simply measured again.
const STALE_RATE_RELAX: Duration = Duration::from_secs(60);
const QUARANTINE_BASE: Duration = Duration::from_secs(5);
const QUARANTINE_MAX: Duration = Duration::from_secs(300);

/// One operator-declared home from `K3_STORAGE_HOMES`. `path == None` names
/// the model root's own drive.
#[derive(Debug, Clone, PartialEq)]
pub struct StorageHomeSpec {
    pub path: Option<PathBuf>,
    /// Starting read rate in GB/s; `None` starts from a default and measures.
    pub gbps: Option<f64>,
}

/// Parse `K3_STORAGE_HOMES`: comma-separated `PATH[@GBPS]` entries, each a
/// directory laid out like the model root. `primary@GBPS` seeds the model
/// root drive's rate. Rates are starting points only; reads measure them.
pub fn parse_storage_homes(raw: &str) -> Result<Vec<StorageHomeSpec>> {
    let mut specs: Vec<StorageHomeSpec> = Vec::new();
    for entry in raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (path, gbps) = match entry.rsplit_once('@') {
            Some((path, speed)) => {
                let gbps = speed
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| {
                        DeltafinError::new(format!(
                            "K3_STORAGE_HOMES entry `{entry}` needs a positive finite GB/s after `@`"
                        ))
                    })?;
                (path.trim(), Some(gbps))
            }
            None => (entry, None),
        };
        let path = if path == "primary" {
            None
        } else {
            let path = PathBuf::from(path);
            if !path.is_absolute() {
                return Err(DeltafinError::new(format!(
                    "K3_STORAGE_HOMES entry `{entry}` must be an absolute directory"
                )));
            }
            Some(path)
        };
        if specs.iter().any(|spec| spec.path == path) {
            return Err(DeltafinError::new(format!(
                "K3_STORAGE_HOMES names `{entry}` more than once"
            )));
        }
        specs.push(StorageHomeSpec { path, gbps });
    }
    if specs.len() > MAX_STORAGE_DEVICES {
        return Err(DeltafinError::new(format!(
            "K3_STORAGE_HOMES admits at most {MAX_STORAGE_DEVICES} drives including primary"
        )));
    }
    Ok(specs)
}

#[derive(Debug)]
struct Device {
    root: PathBuf,
    outstanding: AtomicU64,
    served: AtomicU64,
    reads: AtomicU64,
    failures: AtomicU64,
    state: Mutex<DeviceState>,
}

#[derive(Debug)]
struct DeviceState {
    active: u32,
    busy_since: Option<Instant>,
    window_bytes: u64,
    window_busy: Duration,
    rate: f64,
    prior: f64,
    measured: bool,
    last_sample: Option<Instant>,
    strikes: u32,
    quarantined_until: Option<Instant>,
}

#[derive(Debug)]
struct Mirror {
    device: u8,
    directory_path: PathBuf,
    directory: File,
    present: HashSet<Box<str>>,
}

#[derive(Debug)]
struct Tree {
    /// `<model root>/<relative>` exactly as plans spell it.
    primary: PathBuf,
    canonical: Option<PathBuf>,
    mirrors: Box<[Mirror]>,
}

/// Per-drive counters for `--stats`.
#[derive(Debug, Clone)]
pub struct StorageDeviceStats {
    pub root: PathBuf,
    pub served_bytes: u64,
    pub reads: u64,
    pub failures: u64,
    pub rate_gbps: f64,
    pub measured: bool,
    pub quarantined: bool,
    pub mirrored_files: usize,
}

/// The admitted drives and which files each holds.
#[derive(Debug)]
pub struct StorageHomes {
    devices: Box<[Device]>,
    trees: Box<[Tree]>,
}

/// Bytes of one read charged to its drive until [`DeviceRead::finish`].
#[derive(Debug)]
pub(crate) struct DeviceRead<'a> {
    homes: &'a StorageHomes,
    device: u8,
    bytes: u64,
    finished: bool,
}

impl DeviceRead<'_> {
    /// Release the bytes. A successful read feeds the drive's measured rate;
    /// a failed one puts the drive in quarantine with an increasing back-off.
    pub(crate) fn finish(mut self, ok: bool) {
        self.finished = true;
        self.homes.release(self.device, self.bytes, Some(ok));
    }
}

impl Drop for DeviceRead<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.homes.release(self.device, self.bytes, None);
        }
    }
}

impl StorageHomes {
    /// Admit the declared homes under `model_root`. Returns `None` when no
    /// extra drive holds anything, so the default path carries no state. A
    /// missing, empty or unreadable home is reported and skipped -- the copies
    /// are an optimization, never a requirement.
    pub fn open(model_root: &Path, specs: &[StorageHomeSpec]) -> Result<Option<Arc<Self>>> {
        let primary_rate = specs
            .iter()
            .find(|spec| spec.path.is_none())
            .and_then(|spec| spec.gbps)
            .map_or(DEFAULT_RATE, |gbps| gbps * 1e9);
        let mut devices = vec![Device::new(model_root.to_path_buf(), primary_rate)];
        let mut trees: Vec<Tree> = MIRRORED_TREES
            .iter()
            .map(|relative| {
                let primary = model_root.join(relative);
                let canonical = crate::sys::fs::canonicalize(&primary).ok();
                Tree {
                    primary,
                    canonical,
                    mirrors: Box::new([]),
                }
            })
            .collect();
        let mut mirrors: Vec<Vec<Mirror>> = trees.iter().map(|_| Vec::new()).collect();
        let primary_canonical = crate::sys::fs::canonicalize(model_root).ok();
        let mut seen_roots = Vec::new();
        for spec in specs {
            let Some(root) = &spec.path else {
                continue;
            };
            let canonical = match crate::sys::fs::canonicalize(root) {
                Ok(canonical) => canonical,
                Err(error) => {
                    eprintln!("[storage-homes] skipping {}: {error}", root.display());
                    continue;
                }
            };
            if Some(&canonical) == primary_canonical.as_ref() || seen_roots.contains(&canonical) {
                eprintln!(
                    "[storage-homes] skipping {}: it is the model root or a duplicate",
                    root.display()
                );
                continue;
            }
            let device = devices.len() as u8;
            let mut held = 0_usize;
            for (tree_index, relative) in MIRRORED_TREES.iter().enumerate() {
                let directory_path = root.join(relative);
                if !directory_path.is_dir() {
                    continue;
                }
                match census(&trees[tree_index].primary, &directory_path) {
                    Ok((present, excluded)) => {
                        if excluded != 0 {
                            eprintln!(
                                "[storage-homes] {}: ignoring {excluded} copies that are not exact, current regular files",
                                directory_path.display()
                            );
                        }
                        if present.is_empty() {
                            continue;
                        }
                        let directory = Open::new()
                            .read(true)
                            .directory()
                            .open(&directory_path)
                            .map_err(|error| {
                            DeltafinError::new(format!(
                                "open storage home {}: {error}",
                                directory_path.display()
                            ))
                        })?;
                        held += present.len();
                        mirrors[tree_index].push(Mirror {
                            device,
                            directory_path,
                            directory,
                            present,
                        });
                    }
                    Err(error) => eprintln!(
                        "[storage-homes] skipping {}: {error}",
                        directory_path.display()
                    ),
                }
            }
            if held == 0 {
                eprintln!(
                    "[storage-homes] skipping {}: it holds no usable copies of {}",
                    root.display(),
                    MIRRORED_TREES.join(" or ")
                );
                continue;
            }
            warn_about_filesystem(root, model_root);
            seen_roots.push(canonical);
            devices.push(Device::new(
                root.clone(),
                spec.gbps.map_or(DEFAULT_RATE, |gbps| gbps * 1e9),
            ));
        }
        if devices.len() == 1 {
            return Ok(None);
        }
        for (tree, mirrors) in trees.iter_mut().zip(mirrors) {
            tree.mirrors = mirrors.into_boxed_slice();
        }
        Ok(Some(Arc::new(Self {
            devices: devices.into_boxed_slice(),
            trees: trees.into_boxed_slice(),
        })))
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn stats(&self) -> Vec<StorageDeviceStats> {
        let now = Instant::now();
        self.devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let state = device.state.lock().unwrap();
                StorageDeviceStats {
                    root: device.root.clone(),
                    served_bytes: device.served.load(Ordering::Relaxed),
                    reads: device.reads.load(Ordering::Relaxed),
                    failures: device.failures.load(Ordering::Relaxed),
                    rate_gbps: state.rate / 1e9,
                    measured: state.measured,
                    quarantined: state.quarantined_until.is_some_and(|until| until > now),
                    mirrored_files: self
                        .trees
                        .iter()
                        .flat_map(|tree| tree.mirrors.iter())
                        .filter(|mirror| usize::from(mirror.device) == index)
                        .map(|mirror| mirror.present.len())
                        .sum(),
                }
            })
            .collect()
    }

    /// Bytes currently charged to each drive's in-flight reads.
    #[cfg(test)]
    pub(crate) fn outstanding_bytes(&self) -> Vec<u64> {
        self.devices
            .iter()
            .map(|device| device.outstanding.load(Ordering::Acquire))
            .collect()
    }

    /// The mirrored tree and file name of a primary path, if any drive may
    /// hold a copy of it.
    pub(crate) fn locate<'p>(&self, path: &'p Path) -> Option<(usize, &'p str)> {
        let name = path.file_name()?.to_str()?;
        let parent = path.parent()?;
        self.locate_in(parent, name)
    }

    pub(crate) fn locate_in<'n>(
        &self,
        directory: &Path,
        name: &'n str,
    ) -> Option<(usize, &'n str)> {
        let tree = self.trees.iter().position(|tree| {
            !tree.mirrors.is_empty()
                && (tree.primary == directory || tree.canonical.as_deref() == Some(directory))
        })?;
        self.trees[tree]
            .mirrors
            .iter()
            .any(|mirror| mirror.present.contains(name))
            .then_some((tree, name))
    }

    /// The holder of `name` with the earliest expected finish for `bytes`,
    /// skipping drives in `tried` and drives in quarantine (unless nothing
    /// else holds the file). `None` once every holder has been tried.
    pub(crate) fn choose(&self, tree: usize, name: &str, bytes: u64, tried: u32) -> Option<u8> {
        let now = Instant::now();
        let mut best: Option<(u8, f64, bool)> = None;
        let candidates = std::iter::once(0_u8).chain(
            self.trees[tree]
                .mirrors
                .iter()
                .filter(|mirror| mirror.present.contains(name))
                .map(|mirror| mirror.device),
        );
        for device in candidates {
            if tried & (1_u32 << device) != 0 {
                continue;
            }
            let entry = &self.devices[usize::from(device)];
            let (rate, quarantined) = {
                let state = entry.state.lock().unwrap();
                (
                    state.effective_rate(now),
                    state.quarantined_until.is_some_and(|until| until > now),
                )
            };
            let eta = (entry.outstanding.load(Ordering::Acquire) + bytes) as f64 / rate;
            let better = match best {
                None => true,
                // A healthy drive always beats a quarantined one.
                Some((_, best_eta, best_quarantined)) => {
                    (best_quarantined && !quarantined)
                        || (best_quarantined == quarantined && eta < best_eta)
                }
            };
            if better {
                best = Some((device, eta, quarantined));
            }
        }
        best.map(|(device, _, _)| device)
    }

    /// Where `name` lives on `device` (the primary path for device 0).
    pub(crate) fn file_path(&self, tree: usize, device: u8, name: &str, primary: &Path) -> PathBuf {
        if device == 0 {
            return primary.to_path_buf();
        }
        self.mirror(tree, device).map_or_else(
            || primary.to_path_buf(),
            |mirror| mirror.directory_path.join(name),
        )
    }

    /// The directory descriptor holding `tree` on `device`, for `openat`.
    pub(crate) fn mirror_directory(&self, tree: usize, device: u8) -> Option<(&File, &Path)> {
        self.mirror(tree, device)
            .map(|mirror| (&mirror.directory, mirror.directory_path.as_path()))
    }

    fn mirror(&self, tree: usize, device: u8) -> Option<&Mirror> {
        self.trees
            .get(tree)?
            .mirrors
            .iter()
            .find(|mirror| mirror.device == device)
    }

    /// Charge `bytes` to `device` for the duration of one read.
    pub(crate) fn begin(&self, device: u8, bytes: u64) -> DeviceRead<'_> {
        let entry = &self.devices[usize::from(device).min(self.devices.len() - 1)];
        entry.outstanding.fetch_add(bytes, Ordering::AcqRel);
        let mut state = entry.state.lock().unwrap();
        if state.active == 0 {
            state.busy_since = Some(Instant::now());
        }
        state.active += 1;
        DeviceRead {
            homes: self,
            device,
            bytes,
            finished: false,
        }
    }

    /// Note a failure that happened outside a charged read (an open).
    pub(crate) fn strike(&self, device: u8) {
        let entry = &self.devices[usize::from(device).min(self.devices.len() - 1)];
        entry.failures.fetch_add(1, Ordering::Relaxed);
        let mut state = entry.state.lock().unwrap();
        quarantine(&mut state, &entry.root);
    }

    fn release(&self, device: u8, bytes: u64, outcome: Option<bool>) {
        let entry = &self.devices[usize::from(device).min(self.devices.len() - 1)];
        entry.outstanding.fetch_sub(bytes, Ordering::AcqRel);
        let now = Instant::now();
        let mut state = entry.state.lock().unwrap();
        state.active = state.active.saturating_sub(1);
        match outcome {
            Some(true) => {
                entry.served.fetch_add(bytes, Ordering::Relaxed);
                entry.reads.fetch_add(1, Ordering::Relaxed);
                state.window_bytes += bytes;
                state.strikes = 0;
            }
            Some(false) => {
                entry.failures.fetch_add(1, Ordering::Relaxed);
                quarantine(&mut state, &entry.root);
            }
            None => {}
        }
        if let Some(since) = state.busy_since {
            if state.active == 0 {
                state.window_busy += now.saturating_duration_since(since);
                state.busy_since = None;
            }
        }
        let ongoing = state
            .busy_since
            .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
        let busy = state.window_busy + ongoing;
        if busy >= WINDOW_BUSY && state.window_bytes >= WINDOW_BYTES {
            let sample = state.window_bytes as f64 / busy.as_secs_f64();
            state.rate = if state.measured {
                state.rate + RATE_SMOOTHING * (sample - state.rate)
            } else {
                sample
            };
            state.measured = true;
            state.last_sample = Some(now);
            state.window_bytes = 0;
            state.window_busy = Duration::ZERO;
            if state.busy_since.is_some() {
                state.busy_since = Some(now);
            }
        }
    }
}

impl DeviceState {
    /// The measured rate, except that an estimate below the starting rate
    /// drifts back up while no new measurement arrives.
    fn effective_rate(&self, now: Instant) -> f64 {
        match self.last_sample {
            Some(sampled) if self.rate < self.prior => {
                let idle = now.saturating_duration_since(sampled).as_secs_f64();
                let kept = (-idle / STALE_RATE_RELAX.as_secs_f64()).exp();
                self.prior + (self.rate - self.prior) * kept
            }
            _ => self.rate,
        }
    }
}

impl Device {
    fn new(root: PathBuf, rate: f64) -> Self {
        Self {
            root,
            outstanding: AtomicU64::new(0),
            served: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            state: Mutex::new(DeviceState {
                active: 0,
                busy_since: None,
                window_bytes: 0,
                window_busy: Duration::ZERO,
                rate,
                prior: rate,
                measured: false,
                last_sample: None,
                strikes: 0,
                quarantined_until: None,
            }),
        }
    }
}

fn quarantine(state: &mut DeviceState, root: &Path) {
    state.strikes = state.strikes.saturating_add(1);
    let backoff = QUARANTINE_BASE
        .saturating_mul(1_u32 << (state.strikes - 1).min(10))
        .min(QUARANTINE_MAX);
    state.quarantined_until = Some(Instant::now() + backoff);
    if state.strikes == 1 || state.strikes.is_power_of_two() {
        eprintln!(
            "[storage-homes] {} failed a read; using other copies for {:.0}s",
            root.display(),
            backoff.as_secs_f64()
        );
    }
}

/// Admit one mirror directory's copies: a direct, regular, non-symlink file
/// whose primary exists with the same length and is not newer than the copy.
fn census(primary: &Path, directory: &Path) -> Result<(HashSet<Box<str>>, usize)> {
    let mut present = HashSet::new();
    let mut excluded = 0;
    let entries = fs::read_dir(directory)
        .map_err(|error| DeltafinError::new(format!("scan {}: {error}", directory.display())))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            DeltafinError::new(format!("scan {}: {error}", directory.display()))
        })?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let (Ok(file_type), Ok(copy)) = (entry.file_type(), entry.metadata()) else {
            excluded += 1;
            continue;
        };
        let original = fs::symlink_metadata(primary.join(name));
        let admitted = file_type.is_file()
            && !file_type.is_symlink()
            && original.as_ref().is_ok_and(|original| {
                original.is_file()
                    && original.len() == copy.len()
                    && matches!(
                        (original.modified(), copy.modified()),
                        (Ok(original), Ok(copy)) if original <= copy
                    )
            });
        if admitted {
            present.insert(Box::from(name));
        } else {
            excluded += 1;
        }
    }
    Ok((present, excluded))
}

/// Warn about homes that cannot add bandwidth or that make the per-file
/// opens slow: network shares, FAT-family volumes (linear directory scans
/// over tens of thousands of entries), and copies on the primary's own disk.
fn warn_about_filesystem(root: &Path, model_root: &Path) {
    let Some((kind, from)) = filesystem(root) else {
        return;
    };
    match kind.as_str() {
        "apfs" | "hfs" => {}
        "smbfs" | "nfs" | "afpfs" | "webdav" | "ftp" => eprintln!(
            "[storage-homes] {} is a network share ({kind}); it will only be used when it is measurably fast",
            root.display()
        ),
        other => eprintln!(
            "[storage-homes] {} is {other}; APFS is recommended (large directories are slow to search on FAT-family volumes)",
            root.display()
        ),
    }
    if let Some((_, primary_from)) = filesystem(model_root) {
        if whole_disk(&from) == whole_disk(&primary_from) {
            eprintln!(
                "[storage-homes] {} is on the same disk as the model root ({from}); it adds no bandwidth",
                root.display()
            );
        }
    }
}

/// The filesystem type and backing device of `path`. Only macOS reports them
/// (it is where mirrors on exotic or networked volumes are likely and where
/// the disk naming below is meaningful); everywhere else there is no answer.
#[cfg(target_os = "macos")]
fn filesystem(path: &Path) -> Option<(String, String)> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `statfs` fills the zeroed structure on success; the path is a
    // live NUL-terminated string for the duration of the call.
    let info = unsafe {
        let mut info: libc::statfs = std::mem::zeroed();
        if libc::statfs(path.as_ptr(), &mut info) != 0 {
            return None;
        }
        info
    };
    // SAFETY: the kernel NUL-terminates both fixed-size name fields.
    let kind = unsafe { CStr::from_ptr(info.f_fstypename.as_ptr()) };
    let from = unsafe { CStr::from_ptr(info.f_mntfromname.as_ptr()) };
    Some((
        kind.to_string_lossy().into_owned(),
        from.to_string_lossy().into_owned(),
    ))
}

#[cfg(not(target_os = "macos"))]
fn filesystem(_path: &Path) -> Option<(String, String)> {
    None
}

/// `/dev/disk3s5` and `/dev/disk3s1` are volumes of one physical disk.
fn whole_disk(device: &str) -> String {
    match device.strip_prefix("/dev/disk") {
        Some(rest) => {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            format!("/dev/disk{digits}")
        }
        None => device.to_string(),
    }
}

/// What `deltafin populate-storage-home` should copy.
#[derive(Debug, Clone)]
pub struct PopulateOptions {
    pub model_root: PathBuf,
    pub destination: PathBuf,
    /// Byte ceiling for the home; `None` copies the whole spine and every
    /// expert that fits.
    pub budget_bytes: Option<u64>,
    /// Byte-compare copies that already exist instead of trusting them.
    pub verify_existing: bool,
    pub workers: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PopulateReport {
    pub copied: usize,
    pub already_present: usize,
    pub missing_on_primary: usize,
    pub copied_bytes: u64,
}

/// Leave this much free on the destination volume: a full APFS container
/// stalls the whole machine.
const POPULATE_FREE_RESERVE_BYTES: u64 = 8 << 30;
const POPULATE_CHUNK_BYTES: usize = 32 << 20;

/// Fill a home: the whole int8 spine first (every pass reads all of it),
/// then experts hottest-first by the persistent heat histogram, until the
/// budget or the destination's free-space reserve is reached. Every copy is
/// written under a private partial name, synced, read back without the page
/// cache and compared by SHA-256 against the bytes read from the primary,
/// and only then renamed into place.
pub fn populate(options: &PopulateOptions) -> Result<PopulateReport> {
    let canonical_root = crate::sys::fs::canonicalize(&options.model_root).ok();
    fs::create_dir_all(&options.destination).map_err(|error| {
        DeltafinError::new(format!(
            "create storage home {}: {error}",
            options.destination.display()
        ))
    })?;
    if canonical_root.is_some() && crate::sys::fs::canonicalize(&options.destination).ok() == canonical_root {
        return Err(DeltafinError::new(
            "the storage home must differ from the model root",
        ));
    }
    // Keep Spotlight from indexing tens of thousands of weight files.
    let _ = File::create(options.destination.join(".metadata_never_index"));

    let mut roster: Vec<(PathBuf, PathBuf, u64)> = Vec::new();
    let spine = options.model_root.join(MIRRORED_TREES[1]);
    if let Ok(entries) = fs::read_dir(&spine) {
        let mut files: Vec<_> = entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .filter_map(|entry| {
                let length = entry.metadata().ok()?.len();
                Some((entry.file_name(), length))
            })
            .collect();
        files.sort();
        for (name, length) in files {
            roster.push((
                spine.join(&name),
                options.destination.join(MIRRORED_TREES[1]).join(&name),
                length,
            ));
        }
    }
    let experts = options.model_root.join(MIRRORED_TREES[0]);
    let heat = crate::expert_heat::ExpertHeat::open(&options.model_root, false);
    let heats = &heat.snapshot().heats;
    let mut order: Vec<usize> = (0..crate::experts::K3_EXPERT_RAW_FILES).collect();
    order.sort_by(|&left, &right| {
        heats[right]
            .partial_cmp(&heats[left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(&right))
    });
    for source in order {
        let layer = crate::experts::K3_MOE_LAYER_FIRST
            + (source / crate::experts::K3_EXPERTS_PER_LAYER) as u32;
        let expert = source % crate::experts::K3_EXPERTS_PER_LAYER;
        let name = format!("L{layer}-E{expert}.bin");
        roster.push((
            experts.join(&name),
            options.destination.join(MIRRORED_TREES[0]).join(&name),
            crate::experts::K3_EXPERT_SOURCE_BYTES as u64,
        ));
    }

    let free = crate::one_shot_setup::available_disk_bytes(&options.destination)?;
    let mut space = free.saturating_sub(POPULATE_FREE_RESERVE_BYTES);
    if let Some(budget) = options.budget_bytes {
        space = space.min(budget);
    }
    let mut report = PopulateReport::default();
    let mut work = Vec::new();
    let mut used = 0_u64;
    let mut planned = 0_u64;
    for (source, destination, length) in roster {
        if let Some(budget) = options.budget_bytes {
            if used + length > budget {
                break;
            }
        }
        let present = fs::symlink_metadata(&destination)
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == length);
        used += length;
        if present && !options.verify_existing {
            report.already_present += 1;
            continue;
        }
        if !present {
            if planned + length > space {
                break;
            }
            planned += length;
        }
        work.push((source, destination, present));
    }
    for tree in MIRRORED_TREES {
        fs::create_dir_all(options.destination.join(tree)).map_err(|error| {
            DeltafinError::new(format!(
                "create {}: {error}",
                options.destination.join(tree).display()
            ))
        })?;
    }
    eprintln!(
        "[storage-home] {}: {} files to {} ({:.1} GB new), {} already present",
        options.destination.display(),
        work.len(),
        if options.verify_existing {
            "copy or verify"
        } else {
            "copy"
        },
        planned as f64 / 1e9,
        report.already_present,
    );
    let next = std::sync::atomic::AtomicUsize::new(0);
    let shared = Mutex::new((report, None::<DeltafinError>));
    std::thread::scope(|scope| {
        for _ in 0..options.workers.clamp(1, 16) {
            scope.spawn(|| {
                let mut buffer = vec![0_u8; POPULATE_CHUNK_BYTES];
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some((source, destination, present)) = work.get(index) else {
                        return;
                    };
                    if shared.lock().unwrap().1.is_some() {
                        return;
                    }
                    let outcome = copy_verified(source, destination, *present, &mut buffer);
                    let mut state = shared.lock().unwrap();
                    match outcome {
                        Ok(Copied::Copied(bytes)) => {
                            state.0.copied += 1;
                            state.0.copied_bytes += bytes;
                            if state.0.copied % 500 == 0 {
                                eprintln!(
                                    "[storage-home] {} copied ({:.1} GB)",
                                    state.0.copied,
                                    state.0.copied_bytes as f64 / 1e9
                                );
                            }
                        }
                        Ok(Copied::Verified) => state.0.already_present += 1,
                        Ok(Copied::MissingOnPrimary) => state.0.missing_on_primary += 1,
                        Err(error) => {
                            if state.1.is_none() {
                                state.1 = Some(error);
                            }
                        }
                    }
                }
            });
        }
    });
    let (report, error) = shared.into_inner().unwrap();
    match error {
        Some(error) => Err(error),
        None => Ok(report),
    }
}

enum Copied {
    Copied(u64),
    Verified,
    MissingOnPrimary,
}

fn copy_verified(
    source: &Path,
    destination: &Path,
    present: bool,
    buffer: &mut [u8],
) -> Result<Copied> {
    use std::io::{Read, Write};
    let error = |action: &str, path: &Path, error: std::io::Error| {
        DeltafinError::new(format!("{action} {}: {error}", path.display()))
    };
    let mut input = match open_uncached(source) {
        Ok(file) => file,
        Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Copied::MissingOnPrimary);
        }
        Err(failure) => return Err(error("open", source, failure)),
    };
    let metadata = input
        .metadata()
        .map_err(|failure| error("stat", source, failure))?;
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "primary {} is not a regular file",
            source.display()
        )));
    }
    let length = metadata.len();
    if present {
        let original = digest_file(&mut input, length, buffer)
            .map_err(|failure| error("read", source, failure))?;
        let mut copy =
            open_uncached(destination).map_err(|failure| error("open", destination, failure))?;
        let copied = digest_file(&mut copy, length, buffer)
            .map_err(|failure| error("read", destination, failure))?;
        if original == copied {
            return Ok(Copied::Verified);
        }
        eprintln!(
            "[storage-home] {} differs from the primary; replacing it",
            destination.display()
        );
        input = open_uncached(source).map_err(|failure| error("reopen", source, failure))?;
    }
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("copy");
    let partial = destination.with_file_name(format!(".{name}.partial"));
    let mut digest = crate::packfile::DigestState::new();
    {
        let mut output =
            File::create(&partial).map_err(|failure| error("create", &partial, failure))?;
        let mut remaining = length;
        while remaining != 0 {
            let take = remaining.min(buffer.len() as u64) as usize;
            input
                .read_exact(&mut buffer[..take])
                .map_err(|failure| error("read", source, failure))?;
            digest.update(&buffer[..take]);
            output
                .write_all(&buffer[..take])
                .map_err(|failure| error("write", &partial, failure))?;
            remaining -= take as u64;
        }
        output
            .sync_all()
            .map_err(|failure| error("sync", &partial, failure))?;
    }
    let expected = digest.finalize();
    let mut check =
        open_uncached(&partial).map_err(|failure| error("reopen", &partial, failure))?;
    let actual = digest_file(&mut check, length, buffer)
        .map_err(|failure| error("read back", &partial, failure))?;
    if actual != expected {
        let _ = fs::remove_file(&partial);
        return Err(DeltafinError::new(format!(
            "read-back of {} does not match the primary; the destination drive is not storing bytes faithfully",
            partial.display()
        )));
    }
    fs::rename(&partial, destination).map_err(|failure| error("publish", destination, failure))?;
    Ok(Copied::Copied(length))
}

fn digest_file(
    file: &mut File,
    length: u64,
    buffer: &mut [u8],
) -> std::io::Result<crate::packfile::Digest> {
    use std::io::Read;
    let mut digest = crate::packfile::DigestState::new();
    let mut remaining = length;
    while remaining != 0 {
        let take = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..take])?;
        digest.update(&buffer[..take]);
        remaining -= take as u64;
    }
    Ok(digest.finalize())
}

fn open_uncached(path: &Path) -> std::io::Result<File> {
    let file = Open::new().read(true).no_follow().open(path)?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: the descriptor is live and F_NOCACHE takes an integer.
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn temp_directory(tag: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "deltafin-storage-homes-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn write(path: &Path, bytes: &[u8]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    /// A model root with two experts and one spine tensor, plus a home that
    /// mirrors all three.
    fn fixture(tag: &str) -> (PathBuf, PathBuf) {
        let root = temp_directory(&format!("{tag}-root"));
        write(&root.join("k3-experts/L1-E0.bin"), b"expert-zero");
        write(&root.join("k3-experts/L1-E1.bin"), b"expert-one!");
        write(&root.join("k3-resident-int8/tensors/a.i8"), b"spine");
        let home = temp_directory(&format!("{tag}-home"));
        // Copies are written after the primaries, as populate does.
        std::thread::sleep(Duration::from_millis(5));
        write(&home.join("k3-experts/L1-E0.bin"), b"expert-zero");
        write(&home.join("k3-experts/L1-E1.bin"), b"expert-one!");
        write(&home.join("k3-resident-int8/tensors/a.i8"), b"spine");
        (root, home)
    }

    #[test]
    fn parses_specs() {
        let specs = parse_storage_homes("/a@3, primary@6.5 ,/b").unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(
            specs[1],
            StorageHomeSpec {
                path: None,
                gbps: Some(6.5)
            }
        );
        assert_eq!(specs[2].gbps, None);
        assert!(parse_storage_homes("relative").is_err());
        assert!(parse_storage_homes("/a@0").is_err());
        assert!(parse_storage_homes("/a,/a").is_err());
        assert!(parse_storage_homes("").unwrap().is_empty());
    }

    #[test]
    fn open_admits_exact_current_copies_and_skips_unusable_homes() {
        let (root, home) = fixture("admit");
        let stale = temp_directory("admit-stale");
        // Wrong length, a symlink, and a copy older than its primary.
        write(&stale.join("k3-experts/L1-E0.bin"), b"short");
        crate::sys::fs::symlink(
            root.join("k3-experts/L1-E1.bin"),
            stale.join("k3-experts/L1-E1.bin"),
        )
        .unwrap();
        write(&stale.join("k3-resident-int8/tensors/a.i8"), b"spine");
        let old = std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        File::options()
            .write(true)
            .open(stale.join("k3-resident-int8/tensors/a.i8"))
            .unwrap()
            .set_modified(old)
            .unwrap();
        let specs = parse_storage_homes(&format!(
            "{},{},/nonexistent/home",
            home.display(),
            stale.display()
        ))
        .unwrap();
        let homes = StorageHomes::open(&root, &specs).unwrap().unwrap();
        // The stale home holds nothing usable and the missing one is skipped.
        assert_eq!(homes.device_count(), 2);
        assert_eq!(homes.stats()[1].mirrored_files, 3);
        let expert = root.join("k3-experts/L1-E0.bin");
        assert_eq!(homes.locate(&expert), Some((0, "L1-E0.bin")));
        assert_eq!(homes.locate(&root.join("k3-experts-scale4/L1.sc4")), None);
        assert!(StorageHomes::open(&root, &[]).unwrap().is_none());
        for directory in [root, home, stale] {
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn choice_follows_load_and_measured_rate_and_avoids_failed_drives() {
        let (root, home) = fixture("choose");
        let specs = parse_storage_homes(&format!("primary@1,{}@1", home.display())).unwrap();
        let homes = StorageHomes::open(&root, &specs).unwrap().unwrap();
        let tree = 0;
        // Idle and equal: the primary wins ties.
        assert_eq!(homes.choose(tree, "L1-E0.bin", 100, 0), Some(0));
        // Load on the primary sends the next read to the copy.
        let busy = homes.begin(0, 1_000);
        assert_eq!(homes.choose(tree, "L1-E0.bin", 100, 0), Some(1));
        // Already tried the copy: back to the primary; nothing left after both.
        assert_eq!(homes.choose(tree, "L1-E0.bin", 100, 0b10), Some(0));
        assert_eq!(homes.choose(tree, "L1-E0.bin", 100, 0b11), None);
        drop(busy);
        // A failed read quarantines the copy even though it is idle.
        homes.begin(1, 10).finish(false);
        let _load = homes.begin(0, 1_000_000);
        assert_eq!(homes.choose(tree, "L1-E0.bin", 100, 0), Some(0));
        assert!(homes.stats()[1].quarantined);
        assert_eq!(homes.stats()[1].failures, 1);
        for directory in [root, home] {
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn completed_reads_measure_each_drives_rate() {
        let (root, home) = fixture("rate");
        let specs = parse_storage_homes(&format!("{}", home.display())).unwrap();
        let homes = StorageHomes::open(&root, &specs).unwrap().unwrap();
        // 64 MiB over at least 50 ms of busy time. A loaded host may oversleep,
        // so the bounds come from the elapsed time the test itself observed:
        // the busy window lies inside it (rate >= bytes/elapsed) and lasts at
        // least the sleep (rate <= bytes/50 ms).
        let bytes = 64_u64 << 20;
        let started = Instant::now();
        let read = homes.begin(1, bytes);
        std::thread::sleep(Duration::from_millis(50));
        read.finish(true);
        let elapsed = started.elapsed().as_secs_f64();
        let stats = homes.stats();
        assert!(stats[1].measured);
        let measured = stats[1].rate_gbps * 1e9;
        assert!(
            measured <= bytes as f64 / 0.050 * 1.01 && measured >= bytes as f64 / elapsed * 0.99,
            "{measured} B/s outside [{}, {}]",
            bytes as f64 / elapsed,
            bytes as f64 / 0.050,
        );
        assert!(!stats[0].measured);
        assert_eq!(stats[1].served_bytes, 64 << 20);
        for directory in [root, home] {
            let _ = fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn a_stale_low_estimate_relaxes_back_toward_the_starting_rate() {
        let now = Instant::now();
        let mut state = DeviceState {
            active: 0,
            busy_since: None,
            window_bytes: 0,
            window_busy: Duration::ZERO,
            rate: 1e8,
            prior: 2e9,
            measured: true,
            last_sample: Some(now),
            strikes: 0,
            quarantined_until: None,
        };
        assert!((state.effective_rate(now) - 1e8).abs() < 1.0);
        let later = now + STALE_RATE_RELAX * 3;
        assert!(state.effective_rate(later) > 1.85e9);
        // A measured rate above the starting point is trusted as it is.
        state.rate = 5e9;
        assert_eq!(state.effective_rate(later), 5e9);
    }

    #[test]
    fn whole_disk_groups_volumes() {
        assert_eq!(whole_disk("/dev/disk3s5"), "/dev/disk3");
        assert_eq!(whole_disk("/dev/disk12s1"), "/dev/disk12");
        assert_eq!(whole_disk("//jason@host/share"), "//jason@host/share");
    }

    #[test]
    fn populate_copies_spine_then_hottest_experts_and_verifies() {
        let root = temp_directory("populate-root");
        let experts = root.join("k3-experts");
        fs::create_dir_all(root.join("k3-meta")).unwrap();
        for expert in 0..3_u16 {
            let mut bytes = vec![0_u8; crate::experts::K3_EXPERT_SOURCE_BYTES];
            bytes[0] = expert as u8 + 1;
            write(&experts.join(format!("L1-E{expert}.bin")), &bytes);
        }
        write(&root.join("k3-resident-int8/tensors/a.i8"), &[7_u8; 1000]);
        let destination = temp_directory("populate-home");
        let options = PopulateOptions {
            model_root: root.clone(),
            destination: destination.clone(),
            budget_bytes: Some(1000 + 2 * crate::experts::K3_EXPERT_SOURCE_BYTES as u64),
            verify_existing: false,
            workers: 2,
        };
        let first = populate(&options).unwrap();
        assert_eq!(first.copied, 3, "spine tensor plus two experts");
        assert_eq!(
            fs::read(destination.join("k3-resident-int8/tensors/a.i8")).unwrap(),
            vec![7_u8; 1000]
        );
        assert!(destination.join("k3-experts/L1-E1.bin").exists());
        assert!(!destination.join("k3-experts/L1-E2.bin").exists());
        assert!(destination.join(".metadata_never_index").exists());
        let again = populate(&options).unwrap();
        assert_eq!((again.copied, again.already_present), (0, 3));
        let mut corrupt = fs::read(destination.join("k3-experts/L1-E0.bin")).unwrap();
        corrupt[9] ^= 0xFF;
        fs::write(destination.join("k3-experts/L1-E0.bin"), &corrupt).unwrap();
        let verified = populate(&PopulateOptions {
            verify_existing: true,
            ..options.clone()
        })
        .unwrap();
        assert_eq!((verified.copied, verified.already_present), (1, 2));
        assert_eq!(
            fs::read(destination.join("k3-experts/L1-E0.bin")).unwrap(),
            fs::read(experts.join("L1-E0.bin")).unwrap()
        );
        let specs = parse_storage_homes(&format!("{}", destination.display())).unwrap();
        let homes = StorageHomes::open(&root, &specs).unwrap().unwrap();
        assert_eq!(homes.stats()[1].mirrored_files, 3);
        for directory in [root, destination] {
            let _ = fs::remove_dir_all(directory);
        }
    }
}
