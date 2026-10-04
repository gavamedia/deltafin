//! Conservative resident-layer sizing without Python or subprocesses.
//!
//! The probe and the policy are intentionally separate.  Probing is a small,
//! platform-specific snapshot; selection is a pure function over that snapshot
//! and explicit byte costs.  This keeps the policy deterministic in tests and
//! lets the target engine refresh the snapshot before a large allocation.
//!
//! `FixedCosts` must describe allocations which will happen *after* the memory
//! snapshot.  Already-live allocations are already reflected in available
//! memory and must not be counted a second time.

#[cfg(any(target_os = "linux", test))]
use std::fs;
#[cfg(any(target_os = "linux", test))]
use std::path::{Component, Path, PathBuf};

pub const GIB: u64 = 1 << 30;

/// Host memory visible to this process at one instant.
///
/// A `false` `constraints_readable` value means Linux declared a cgroup memory
/// hierarchy but its controlling files could not be read.  That is an unknown
/// limit, not evidence that all host RAM is available.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct HostMemory {
    pub physical_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
    pub cgroup_available_bytes: Option<u64>,
    pub constraints_readable: bool,
}

impl HostMemory {
    pub const fn unknown() -> Self {
        Self {
            physical_bytes: None,
            available_bytes: None,
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: false,
        }
    }

    pub fn effective_total_bytes(self) -> Option<u64> {
        if !self.constraints_readable {
            return None;
        }
        let physical = self.physical_bytes?;
        Some(
            self.cgroup_limit_bytes
                .map_or(physical, |limit| physical.min(limit)),
        )
    }

    pub fn effective_available_bytes(self) -> Option<u64> {
        if !self.constraints_readable {
            return None;
        }
        let host_available = self.available_bytes?;
        let available = self
            .cgroup_available_bytes
            .map_or(host_available, |value| host_available.min(value));
        self.effective_total_bytes()
            .map(|total| available.min(total))
            .or(Some(available))
    }
}

/// Where provider-owned tensors consume memory.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ProviderMemory {
    /// CPU tensors consume ordinary host memory.
    Host,
    /// MPS-style unified memory consumes host RAM and is additionally bounded
    /// by the provider's recommended working-set ceiling when one is known.
    Unified {
        recommended_working_set_bytes: Option<u64>,
        /// Additional bytes available before reaching the provider working
        /// set. A live snapshot derives this from recommendation minus the
        /// complete driver reservation, including inactive allocator blocks.
        available_working_set_bytes: Option<u64>,
    },
    /// CUDA-style device memory is distinct from host RAM.  Both values are
    /// required; an unavailable query fails closed instead of guessing.
    Discrete {
        total_bytes: Option<u64>,
        available_bytes: Option<u64>,
    },
}

/// Planned non-layer allocations which are not yet reflected in the probe.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct FixedCosts {
    /// Host-only I/O buffers, protected file cache, and other host allocations.
    pub host_bytes: u64,
    /// Embeddings/head, KV state, and transient allocations in provider memory.
    pub provider_bytes: u64,
}

/// Explicit requests are requests, never permission to exceed safe limits.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ResidencyOverride {
    pub requested_layers: Option<usize>,
    pub requested_provider_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ResidencyPolicy {
    pub host_reserve_floor_bytes: u64,
    pub host_reserve_permille: u16,
    pub device_reserve_floor_bytes: u64,
    pub device_reserve_permille: u16,
    pub discrete_utilization_permille: u16,
}

impl Default for ResidencyPolicy {
    fn default() -> Self {
        Self {
            // Match the mature Python policy: keep enough room for the OS,
            // applications, expert I/O, and memory-pressure variation.
            host_reserve_floor_bytes: 10 * GIB,
            host_reserve_permille: 180,
            device_reserve_floor_bytes: 2 * GIB,
            device_reserve_permille: 100,
            discrete_utilization_permille: 850,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResidencyStop {
    AllLayersFit,
    ExplicitLayerLimit,
    ExplicitByteLimit,
    InvalidPolicy,
    HostBudgetUnknown,
    DeviceBudgetUnknown,
    FixedCostsExceedBudget,
    LayerCostUnknown,
    NextLayerWouldExceedBudget,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencySelection {
    /// Always the ordered prefix `0..resident_layers`.
    pub resident_layers: usize,
    pub resident_provider_bytes: u64,
    /// Maximum new host allocation after reserves, before fixed costs.
    pub host_envelope_bytes: Option<u64>,
    /// Additional provider ceiling for unified/discrete accelerators.
    pub provider_envelope_bytes: Option<u64>,
    pub stop: ResidencyStop,
    /// True only when an explicit request exceeded a safety-derived limit.
    pub override_clamped_by_safety: bool,
}

/// Select the largest safe ordered layer prefix.
///
/// `provider_layer_bytes[i]` is the complete provider-owned cost of retaining
/// layer `i` (including expanded scales/storage, not merely its file size).
/// Zero is treated as unknown and stops the prefix rather than granting a free
/// layer.  All arithmetic saturates toward fewer resident layers.
pub fn select_resident_prefix(
    host: HostMemory,
    provider: ProviderMemory,
    provider_layer_bytes: &[u64],
    fixed: FixedCosts,
    request: ResidencyOverride,
    policy: ResidencyPolicy,
) -> ResidencySelection {
    if policy.host_reserve_permille > 1_000
        || policy.device_reserve_permille > 1_000
        || policy.discrete_utilization_permille > 1_000
    {
        return empty_selection(
            None,
            None,
            ResidencyStop::InvalidPolicy,
            request_requests_residency(request),
        );
    }
    let host_envelope = safe_host_envelope(host, policy);
    let provider_envelope = safe_provider_envelope(provider, policy);

    let Some(host_limit) = host_envelope else {
        return empty_selection(
            host_envelope,
            provider_envelope,
            ResidencyStop::HostBudgetUnknown,
            request_requests_residency(request),
        );
    };
    if provider_requires_known_budget(provider) && provider_envelope.is_none() {
        return empty_selection(
            host_envelope,
            provider_envelope,
            ResidencyStop::DeviceBudgetUnknown,
            request_requests_residency(request),
        );
    }

    let (base_host, base_provider) = match provider {
        ProviderMemory::Host | ProviderMemory::Unified { .. } => {
            let Some(base_host) = fixed.host_bytes.checked_add(fixed.provider_bytes) else {
                return empty_selection(
                    host_envelope,
                    provider_envelope,
                    ResidencyStop::ArithmeticOverflow,
                    request_requests_residency(request),
                );
            };
            (base_host, fixed.provider_bytes)
        }
        ProviderMemory::Discrete { .. } => (fixed.host_bytes, fixed.provider_bytes),
    };
    if base_host > host_limit || provider_envelope.is_some_and(|limit| base_provider > limit) {
        return empty_selection(
            host_envelope,
            provider_envelope,
            ResidencyStop::FixedCostsExceedBudget,
            request_requests_residency(request),
        );
    }

    let unrestricted = fit_prefix(
        provider,
        provider_layer_bytes,
        base_host,
        base_provider,
        host_limit,
        provider_envelope,
        provider_layer_bytes.len(),
        u64::MAX,
    );
    let layer_limit = request
        .requested_layers
        .unwrap_or(provider_layer_bytes.len())
        .min(provider_layer_bytes.len());
    let byte_limit = request.requested_provider_bytes.unwrap_or(u64::MAX);
    let selected = fit_prefix(
        provider,
        provider_layer_bytes,
        base_host,
        base_provider,
        host_limit,
        provider_envelope,
        layer_limit,
        byte_limit,
    );

    let explicit_safety_clamp = request
        .requested_layers
        .is_some_and(|wanted| wanted > unrestricted.layers)
        || request.requested_provider_bytes.is_some_and(|wanted| {
            wanted > unrestricted.provider_layer_bytes
                && unrestricted.layers < provider_layer_bytes.len()
        });

    let stop = if selected.layers == provider_layer_bytes.len() {
        ResidencyStop::AllLayersFit
    } else if selected.layers == layer_limit
        && request.requested_layers.is_some()
        && layer_limit <= unrestricted.layers
    {
        ResidencyStop::ExplicitLayerLimit
    } else if selected.hit_byte_limit {
        ResidencyStop::ExplicitByteLimit
    } else {
        selected.stop
    };

    ResidencySelection {
        resident_layers: selected.layers,
        resident_provider_bytes: selected.provider_layer_bytes,
        host_envelope_bytes: host_envelope,
        provider_envelope_bytes: provider_envelope,
        stop,
        override_clamped_by_safety: explicit_safety_clamp,
    }
}

#[derive(Debug, Clone, Copy)]
struct PrefixFit {
    layers: usize,
    provider_layer_bytes: u64,
    stop: ResidencyStop,
    hit_byte_limit: bool,
}

#[allow(clippy::too_many_arguments)]
fn fit_prefix(
    provider: ProviderMemory,
    layer_bytes: &[u64],
    mut used_host: u64,
    mut used_provider: u64,
    host_limit: u64,
    provider_limit: Option<u64>,
    layer_limit: usize,
    byte_limit: u64,
) -> PrefixFit {
    let mut resident_provider_bytes = 0_u64;
    for (index, &cost) in layer_bytes.iter().take(layer_limit).enumerate() {
        if cost == 0 {
            return PrefixFit {
                layers: index,
                provider_layer_bytes: resident_provider_bytes,
                stop: ResidencyStop::LayerCostUnknown,
                hit_byte_limit: false,
            };
        }
        let Some(next_resident_bytes) = resident_provider_bytes.checked_add(cost) else {
            return PrefixFit {
                layers: index,
                provider_layer_bytes: resident_provider_bytes,
                stop: ResidencyStop::ArithmeticOverflow,
                hit_byte_limit: false,
            };
        };
        if next_resident_bytes > byte_limit {
            return PrefixFit {
                layers: index,
                provider_layer_bytes: resident_provider_bytes,
                stop: ResidencyStop::ExplicitByteLimit,
                hit_byte_limit: true,
            };
        }
        let host_increment = match provider {
            ProviderMemory::Host | ProviderMemory::Unified { .. } => cost,
            ProviderMemory::Discrete { .. } => 0,
        };
        let Some(next_host) = used_host.checked_add(host_increment) else {
            return PrefixFit {
                layers: index,
                provider_layer_bytes: resident_provider_bytes,
                stop: ResidencyStop::ArithmeticOverflow,
                hit_byte_limit: false,
            };
        };
        let Some(next_provider) = used_provider.checked_add(cost) else {
            return PrefixFit {
                layers: index,
                provider_layer_bytes: resident_provider_bytes,
                stop: ResidencyStop::ArithmeticOverflow,
                hit_byte_limit: false,
            };
        };
        if next_host > host_limit || provider_limit.is_some_and(|limit| next_provider > limit) {
            return PrefixFit {
                layers: index,
                provider_layer_bytes: resident_provider_bytes,
                stop: ResidencyStop::NextLayerWouldExceedBudget,
                hit_byte_limit: false,
            };
        }
        used_host = next_host;
        used_provider = next_provider;
        resident_provider_bytes = next_resident_bytes;
    }
    PrefixFit {
        layers: layer_limit.min(layer_bytes.len()),
        provider_layer_bytes: resident_provider_bytes,
        stop: ResidencyStop::AllLayersFit,
        hit_byte_limit: false,
    }
}

fn empty_selection(
    host_envelope_bytes: Option<u64>,
    provider_envelope_bytes: Option<u64>,
    stop: ResidencyStop,
    override_clamped_by_safety: bool,
) -> ResidencySelection {
    ResidencySelection {
        resident_layers: 0,
        resident_provider_bytes: 0,
        host_envelope_bytes,
        provider_envelope_bytes,
        stop,
        override_clamped_by_safety,
    }
}

fn request_requests_residency(request: ResidencyOverride) -> bool {
    request.requested_layers.is_some_and(|value| value > 0)
        || request
            .requested_provider_bytes
            .is_some_and(|value| value > 0)
}

fn safe_host_envelope(host: HostMemory, policy: ResidencyPolicy) -> Option<u64> {
    let total = host.effective_total_bytes()?;
    let available = host.effective_available_bytes()?;
    let reserve = reserve(
        total,
        policy.host_reserve_floor_bytes,
        policy.host_reserve_permille,
    )?;
    Some(
        total
            .saturating_sub(reserve)
            .min(available.saturating_sub(reserve)),
    )
}

fn safe_provider_envelope(provider: ProviderMemory, policy: ResidencyPolicy) -> Option<u64> {
    match provider {
        ProviderMemory::Host => None,
        ProviderMemory::Unified {
            recommended_working_set_bytes: None,
            available_working_set_bytes: _,
        } => None,
        ProviderMemory::Unified {
            recommended_working_set_bytes: Some(total),
            available_working_set_bytes,
        } => {
            let reserve = reserve(
                total,
                policy.device_reserve_floor_bytes,
                policy.device_reserve_permille,
            )?;
            Some(
                total.saturating_sub(reserve).min(
                    available_working_set_bytes
                        .unwrap_or(total)
                        .min(total)
                        .saturating_sub(reserve),
                ),
            )
        }
        ProviderMemory::Discrete {
            total_bytes: Some(total),
            available_bytes: Some(available),
        } => {
            let reserve = reserve(
                total,
                policy.device_reserve_floor_bytes,
                policy.device_reserve_permille,
            )?;
            let raw = total
                .saturating_sub(reserve)
                .min(available.min(total).saturating_sub(reserve));
            raw.checked_mul(u64::from(policy.discrete_utilization_permille))
                .map(|bytes| bytes / 1_000)
        }
        ProviderMemory::Discrete { .. } => None,
    }
}

fn provider_requires_known_budget(provider: ProviderMemory) -> bool {
    matches!(provider, ProviderMemory::Discrete { .. })
}

fn reserve(total: u64, floor: u64, permille: u16) -> Option<u64> {
    if permille > 1_000 {
        return None;
    }
    let proportional = total.checked_mul(u64::from(permille))? / 1_000;
    Some(floor.max(proportional).min(total))
}

/// Query memory using native kernel interfaces only.
pub fn probe_host_memory() -> HostMemory {
    #[cfg(target_os = "macos")]
    {
        probe_macos_memory()
    }
    #[cfg(target_os = "linux")]
    {
        probe_linux_memory(
            Path::new("/proc/meminfo"),
            Path::new("/proc/self/cgroup"),
            Path::new("/sys/fs/cgroup"),
        )
    }
    #[cfg(windows)]
    {
        probe_windows_memory()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        HostMemory::unknown()
    }
}

/// Windows: `GlobalMemoryStatusEx` for physical and available memory, and a
/// job-object memory limit (how a container caps a process) as the analogue
/// of a Linux cgroup limit. An unreadable kernel answer stays unknown, which
/// the planner treats as "no resident layers", never as "all of RAM".
#[cfg(windows)]
fn probe_windows_memory() -> HostMemory {
    let Some(memory) = deltafin_sys::memory::system_memory() else {
        return HostMemory::unknown();
    };
    HostMemory {
        physical_bytes: Some(memory.physical_bytes),
        available_bytes: Some(memory.available_bytes),
        cgroup_limit_bytes: memory.job_limit_bytes,
        cgroup_available_bytes: None,
        constraints_readable: true,
    }
}

// Keep the same file-cache reserve as the mature Python watchdog. The model's
// enormous file-backed population is overwhelmingly read-only, but Mach does
// not distinguish clean from dirty external pages here, so this remains a
// buffered estimate rather than a proof that every page is immediately free.
// Anonymous inactive pages are deliberately excluded: claiming those can
// start compression or swap and turn a small allocation into a severe cliff.
#[cfg(any(target_os = "macos", test))]
const MACOS_FILE_CACHE_RESERVE_BYTES: u64 = 8 * GIB;

#[cfg(any(target_os = "macos", test))]
fn macos_reclaim_free_bytes(
    words: &[u32],
    returned_count: u32,
    page_size: u64,
    file_cache_reserve_bytes: u64,
) -> Option<u64> {
    // vm_statistics64 revision 1, expressed as Mach integer_t words:
    // free_count=0, speculative_count=23, and external_page_count=34. Apple's
    // header says speculative pages are already in free_count; the file versus
    // anonymous classification also includes them. Subtracting that overlap
    // from external avoids counting the same pages twice. Purgeable pages are
    // likewise a classification of resident pages, not a separate pool, so
    // this deliberately does not add word 22.
    const FREE_COUNT: usize = 0;
    const SPECULATIVE_COUNT: usize = 23;
    const EXTERNAL_PAGE_COUNT: usize = 34;
    if page_size == 0 || returned_count == 0 || words.is_empty() {
        return None;
    }

    let free_bytes = u64::from(words[FREE_COUNT]).checked_mul(page_size)?;
    let required = EXTERNAL_PAGE_COUNT + 1;
    if returned_count < required as u32 || words.len() < required {
        // A shorter successful revision still provides a conservative free
        // value; it simply cannot recover clean file-cache headroom.
        return Some(free_bytes);
    }
    let external_non_speculative_pages =
        u64::from(words[EXTERNAL_PAGE_COUNT]).saturating_sub(u64::from(words[SPECULATIVE_COUNT]));
    let external_non_speculative_bytes = external_non_speculative_pages.checked_mul(page_size)?;
    free_bytes.checked_add(external_non_speculative_bytes.saturating_sub(file_cache_reserve_bytes))
}

#[cfg(any(target_os = "macos", test))]
fn cap_available_to_physical(
    available_bytes: Option<u64>,
    physical_bytes: Option<u64>,
) -> Option<u64> {
    // Missing dynamic availability must remain unknown. Substituting physical
    // RAM here would fail open precisely when host_statistics64 failed.
    available_bytes
        .map(|available| physical_bytes.map_or(available, |physical| available.min(physical)))
}

#[cfg(target_os = "macos")]
#[repr(C, align(8))]
struct AlignedMachVmWords([u32; 64]);

#[cfg(target_os = "macos")]
fn probe_macos_memory() -> HostMemory {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old_value: *mut c_void,
            old_len: *mut usize,
            new_value: *mut c_void,
            new_len: usize,
        ) -> c_int;
        fn mach_host_self() -> u32;
        fn host_statistics64(host: u32, flavor: c_int, info: *mut c_int, count: *mut u32) -> c_int;
        fn mach_port_deallocate(task: u32, name: u32) -> c_int;
        fn getpagesize() -> c_int;
        static mach_task_self_: u32;
    }

    let physical_bytes = {
        let mut bytes = 0_u64;
        let mut length = std::mem::size_of::<u64>();
        // SAFETY: `hw.memsize` is a static C string and the output pointer/length
        // describe a live `u64`; this is a read-only sysctl query.
        let result = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut bytes as *mut u64).cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        (result == 0 && length == std::mem::size_of::<u64>() && bytes > 0).then_some(bytes)
    };

    let available_bytes = {
        const HOST_VM_INFO64: c_int = 4;
        // Revision 1 ends after `total_uncompressed_pages_in_compressor` and
        // is stable across the supported arm64 macOS releases.
        const HOST_VM_INFO64_REV1_COUNT: u32 = 38;
        // vm_statistics64 revision 4 requires 64-bit alignment even though
        // Mach exposes the transfer buffer as integer_t words.
        let mut words = AlignedMachVmWords([0_u32; 64]);
        let mut count = HOST_VM_INFO64_REV1_COUNT;
        // SAFETY: the aligned 64-word buffer is larger than the requested
        // 38-word revision, and count is an initialized in/out parameter.
        // `mach_host_self` creates a send-right user reference. Keep the name
        // long enough for the query, then release it on every result path so
        // repeated context-growth probes cannot leak port references.
        let host = unsafe { mach_host_self() };
        let result = unsafe {
            host_statistics64(
                host,
                HOST_VM_INFO64,
                words.0.as_mut_ptr().cast(),
                &mut count,
            )
        };
        // SAFETY: both names are task-local Mach port names. Deallocation only
        // drops the user reference obtained above; it does not destroy host.
        let _ = unsafe { mach_port_deallocate(mach_task_self_, host) };
        // SAFETY: `getpagesize` has no pointer arguments or preconditions.
        let page_size = unsafe { getpagesize() };
        if result == 0 && page_size > 0 {
            macos_reclaim_free_bytes(
                &words.0,
                count,
                page_size as u64,
                MACOS_FILE_CACHE_RESERVE_BYTES,
            )
        } else {
            None
        }
    };

    HostMemory {
        physical_bytes,
        available_bytes: cap_available_to_physical(available_bytes, physical_bytes),
        cgroup_limit_bytes: None,
        cgroup_available_bytes: None,
        constraints_readable: true,
    }
}

#[cfg(any(target_os = "linux", test))]
fn probe_linux_memory(meminfo: &Path, self_cgroup: &Path, cgroup_root: &Path) -> HostMemory {
    let meminfo_text = fs::read_to_string(meminfo).ok();
    let (physical_bytes, available_bytes) = meminfo_text
        .as_deref()
        .map(parse_meminfo)
        .unwrap_or((None, None));

    let cgroup_text = match fs::read_to_string(self_cgroup) {
        Ok(text) => text,
        Err(_) => {
            return HostMemory {
                physical_bytes,
                available_bytes,
                cgroup_limit_bytes: None,
                cgroup_available_bytes: None,
                constraints_readable: false,
            };
        }
    };
    let discovery = cgroup_hierarchies(&cgroup_text, cgroup_root);
    if discovery.malformed || (discovery.memory_declared && discovery.hierarchies.is_empty()) {
        return HostMemory {
            physical_bytes,
            available_bytes,
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: false,
        };
    }
    if !discovery.memory_declared {
        return HostMemory {
            physical_bytes,
            available_bytes,
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        };
    }

    let mut constraints_readable = true;
    let mut finite: Vec<(u64, u64)> = Vec::new();
    for hierarchy in discovery.hierarchies {
        let mut saw_leaf_limit = false;
        for (depth, directory) in ancestors_within(hierarchy.leaf, hierarchy.root)
            .into_iter()
            .enumerate()
        {
            let limit_path = directory.join(hierarchy.limit_name);
            let current_path = directory.join(hierarchy.current_name);
            let raw_limit = match fs::read_to_string(&limit_path) {
                Ok(value) => {
                    if depth == 0 {
                        saw_leaf_limit = true;
                    }
                    value
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => {
                    constraints_readable = false;
                    continue;
                }
            };
            let raw_limit = raw_limit.trim();
            if raw_limit == "max" {
                continue;
            }
            let Ok(limit) = raw_limit.parse::<u64>() else {
                constraints_readable = false;
                continue;
            };
            if cgroup_limit_is_sentinel(limit, physical_bytes) {
                continue;
            }
            let current = match fs::read_to_string(&current_path)
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok())
            {
                Some(value) => value,
                None => {
                    constraints_readable = false;
                    continue;
                }
            };
            finite.push((limit, limit.saturating_sub(current)));
        }
        if !saw_leaf_limit {
            constraints_readable = false;
        }
    }

    HostMemory {
        physical_bytes,
        available_bytes,
        cgroup_limit_bytes: finite.iter().map(|pair| pair.0).min(),
        cgroup_available_bytes: finite.iter().map(|pair| pair.1).min(),
        constraints_readable,
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_meminfo(text: &str) -> (Option<u64>, Option<u64>) {
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(token) = rest.split_whitespace().next() else {
            continue;
        };
        let Ok(kib) = token.parse::<u64>() else {
            continue;
        };
        let Some(bytes) = kib.checked_mul(1_024) else {
            continue;
        };
        match name {
            "MemTotal" if bytes > 0 => total = Some(bytes),
            "MemAvailable" => available = Some(bytes),
            _ => {}
        }
    }
    (total, available)
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug)]
struct CgroupHierarchy {
    root: PathBuf,
    leaf: PathBuf,
    limit_name: &'static str,
    current_name: &'static str,
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug)]
struct CgroupDiscovery {
    hierarchies: Vec<CgroupHierarchy>,
    memory_declared: bool,
    malformed: bool,
}

#[cfg(any(target_os = "linux", test))]
fn cgroup_hierarchies(text: &str, cgroup_root: &Path) -> CgroupDiscovery {
    let mut out = Vec::new();
    let mut memory_declared = false;
    let mut malformed = false;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, ':');
        let (Some(hierarchy), Some(controllers), Some(group)) =
            (parts.next(), parts.next(), parts.next())
        else {
            malformed = true;
            continue;
        };
        if hierarchy == "0" && controllers.is_empty() {
            memory_declared = true;
            if let Some(leaf) = safe_cgroup_join(cgroup_root, group) {
                out.push(CgroupHierarchy {
                    root: cgroup_root.to_path_buf(),
                    leaf,
                    limit_name: "memory.max",
                    current_name: "memory.current",
                });
            }
        } else if controllers.split(',').any(|value| value == "memory") {
            memory_declared = true;
            let root = cgroup_root.join("memory");
            if let Some(leaf) = safe_cgroup_join(&root, group) {
                out.push(CgroupHierarchy {
                    root,
                    leaf,
                    limit_name: "memory.limit_in_bytes",
                    current_name: "memory.usage_in_bytes",
                });
            }
        }
    }
    CgroupDiscovery {
        hierarchies: out,
        memory_declared,
        malformed,
    }
}

#[cfg(any(target_os = "linux", test))]
fn safe_cgroup_join(root: &Path, group: &str) -> Option<PathBuf> {
    let mut out = root.to_path_buf();
    for component in Path::new(group).components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(value) => out.push(value),
            Component::ParentDir | Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

#[cfg(any(target_os = "linux", test))]
fn ancestors_within(mut leaf: PathBuf, root: PathBuf) -> Vec<PathBuf> {
    if leaf != root && !leaf.starts_with(&root) {
        return Vec::new();
    }
    let mut out = Vec::new();
    loop {
        out.push(leaf.clone());
        if leaf == root {
            break;
        }
        if !leaf.pop() || (leaf != root && !leaf.starts_with(&root)) {
            break;
        }
    }
    out
}

#[cfg(any(target_os = "linux", test))]
fn cgroup_limit_is_sentinel(limit: u64, host_total: Option<u64>) -> bool {
    limit >= (1_u64 << 60)
        || host_total.is_some_and(|total| {
            total
                .checked_mul(4)
                .is_some_and(|threshold| limit >= threshold)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_reserve_policy() -> ResidencyPolicy {
        ResidencyPolicy {
            host_reserve_floor_bytes: 0,
            host_reserve_permille: 0,
            device_reserve_floor_bytes: 0,
            device_reserve_permille: 0,
            discrete_utilization_permille: 1_000,
        }
    }

    fn host(total: u64, available: u64) -> HostMemory {
        HostMemory {
            physical_bytes: Some(total),
            available_bytes: Some(available),
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
            constraints_readable: true,
        }
    }

    #[test]
    fn chooses_an_ordered_prefix_instead_of_skipping_an_expensive_layer() {
        let selection = select_resident_prefix(
            host(100, 30),
            ProviderMemory::Host,
            &[8, 15, 4],
            FixedCosts {
                host_bytes: 2,
                provider_bytes: 3,
            },
            ResidencyOverride::default(),
            no_reserve_policy(),
        );
        assert_eq!(selection.resident_layers, 2);
        assert_eq!(selection.resident_provider_bytes, 23);
        assert_eq!(selection.stop, ResidencyStop::NextLayerWouldExceedBudget);
    }

    #[test]
    fn a_large_override_is_clamped_to_the_safe_envelope() {
        let selection = select_resident_prefix(
            host(100, 20),
            ProviderMemory::Host,
            &[10, 10, 10],
            FixedCosts::default(),
            ResidencyOverride {
                requested_layers: Some(3),
                requested_provider_bytes: Some(1_000),
            },
            no_reserve_policy(),
        );
        assert_eq!(selection.resident_layers, 2);
        assert!(selection.override_clamped_by_safety);
    }

    #[test]
    fn a_smaller_explicit_limit_remains_authoritative() {
        let selection = select_resident_prefix(
            host(100, 100),
            ProviderMemory::Host,
            &[10, 10, 10],
            FixedCosts::default(),
            ResidencyOverride {
                requested_layers: Some(1),
                requested_provider_bytes: None,
            },
            no_reserve_policy(),
        );
        assert_eq!(selection.resident_layers, 1);
        assert_eq!(selection.stop, ResidencyStop::ExplicitLayerLimit);
        assert!(!selection.override_clamped_by_safety);
    }

    #[test]
    fn discrete_device_query_failure_fails_closed() {
        let selection = select_resident_prefix(
            host(100, 100),
            ProviderMemory::Discrete {
                total_bytes: Some(80),
                available_bytes: None,
            },
            &[10],
            FixedCosts::default(),
            ResidencyOverride::default(),
            no_reserve_policy(),
        );
        assert_eq!(selection.resident_layers, 0);
        assert_eq!(selection.stop, ResidencyStop::DeviceBudgetUnknown);
    }

    #[test]
    fn discrete_device_and_host_budgets_are_accounted_separately() {
        let selection = select_resident_prefix(
            host(100, 12),
            ProviderMemory::Discrete {
                total_bytes: Some(100),
                available_bytes: Some(26),
            },
            &[10, 10, 10],
            FixedCosts {
                host_bytes: 12,
                provider_bytes: 5,
            },
            ResidencyOverride::default(),
            no_reserve_policy(),
        );
        assert_eq!(selection.resident_layers, 2);
    }

    #[test]
    fn unified_memory_must_fit_both_host_and_working_set_envelopes() {
        let selection = select_resident_prefix(
            host(100, 100),
            ProviderMemory::Unified {
                recommended_working_set_bytes: Some(25),
                available_working_set_bytes: Some(25),
            },
            &[10, 10, 10],
            FixedCosts {
                host_bytes: 0,
                provider_bytes: 5,
            },
            ResidencyOverride::default(),
            no_reserve_policy(),
        );
        assert_eq!(selection.resident_layers, 2);
    }

    #[test]
    fn unified_live_working_set_limits_only_new_provider_bytes() {
        let selection = select_resident_prefix(
            host(100, 100),
            ProviderMemory::Unified {
                recommended_working_set_bytes: Some(100),
                available_working_set_bytes: Some(20),
            },
            &[10, 10, 10],
            FixedCosts {
                host_bytes: 0,
                provider_bytes: 5,
            },
            ResidencyOverride::default(),
            no_reserve_policy(),
        );
        assert_eq!(selection.resident_layers, 1);
        assert_eq!(selection.stop, ResidencyStop::NextLayerWouldExceedBudget);
    }

    #[test]
    fn unknown_or_overflowing_costs_never_grant_layers() {
        let unknown = select_resident_prefix(
            host(100, 100),
            ProviderMemory::Host,
            &[0, 10],
            FixedCosts::default(),
            ResidencyOverride::default(),
            no_reserve_policy(),
        );
        assert_eq!(unknown.stop, ResidencyStop::LayerCostUnknown);
        assert_eq!(unknown.resident_layers, 0);

        let overflow = select_resident_prefix(
            host(u64::MAX, u64::MAX),
            ProviderMemory::Host,
            &[2],
            FixedCosts {
                host_bytes: 0,
                provider_bytes: u64::MAX,
            },
            ResidencyOverride::default(),
            no_reserve_policy(),
        );
        assert_eq!(overflow.stop, ResidencyStop::ArithmeticOverflow);
        assert_eq!(overflow.resident_layers, 0);
    }

    #[test]
    fn malformed_policy_fails_closed() {
        let mut policy = no_reserve_policy();
        policy.discrete_utilization_permille = 1_001;
        let selection = select_resident_prefix(
            host(100, 100),
            ProviderMemory::Host,
            &[10],
            FixedCosts::default(),
            ResidencyOverride::default(),
            policy,
        );
        assert_eq!(selection.resident_layers, 0);
        assert_eq!(selection.stop, ResidencyStop::InvalidPolicy);
    }

    #[test]
    fn a_partial_host_probe_is_not_treated_as_capacity() {
        for incomplete in [
            HostMemory {
                physical_bytes: None,
                available_bytes: Some(100),
                cgroup_limit_bytes: Some(100),
                cgroup_available_bytes: Some(100),
                constraints_readable: true,
            },
            HostMemory {
                physical_bytes: Some(100),
                available_bytes: None,
                cgroup_limit_bytes: Some(100),
                cgroup_available_bytes: Some(100),
                constraints_readable: true,
            },
        ] {
            let selection = select_resident_prefix(
                incomplete,
                ProviderMemory::Host,
                &[1],
                FixedCosts::default(),
                ResidencyOverride::default(),
                no_reserve_policy(),
            );
            assert_eq!(selection.stop, ResidencyStop::HostBudgetUnknown);
            assert_eq!(selection.resident_layers, 0);
        }
    }

    #[test]
    fn default_policy_leaves_the_documented_host_reserve() {
        let memory = host(64 * GIB, 40 * GIB);
        let envelope = safe_host_envelope(memory, ResidencyPolicy::default()).unwrap();
        // 18% of 64 GiB is larger than the 10 GiB floor.
        assert_eq!(envelope, 40 * GIB - (64 * GIB * 180 / 1_000));
    }

    #[test]
    fn macos_probe_counts_only_reclaim_free_pages_and_retains_file_cache() {
        let page = 16 * 1_024_u64;
        let pages = |bytes: u64| u32::try_from(bytes / page).unwrap();
        let mut words = [0_u32; 38];
        words[0] = pages(4 * GIB); // free (already includes speculative)
        words[22] = pages(GIB); // purgeable: resident classification
        words[23] = pages(2 * GIB); // speculative: overlaps free/external
        words[34] = pages(20 * GIB); // clean/file-backed population

        assert_eq!(
            macos_reclaim_free_bytes(&words, 38, page, MACOS_FILE_CACHE_RESERVE_BYTES,),
            Some(14 * GIB)
        );
        words[22] = pages(3 * GIB);
        assert_eq!(
            macos_reclaim_free_bytes(&words, 38, page, MACOS_FILE_CACHE_RESERVE_BYTES,),
            Some(14 * GIB),
            "purgeable_count must not double-count resident pages"
        );
    }

    #[test]
    fn macos_probe_rejects_truncated_or_invalid_snapshots() {
        let words = [0_u32; 38];
        assert_eq!(
            macos_reclaim_free_bytes(&words, 34, 16 * 1_024, 8 * GIB),
            Some(0)
        );
        assert_eq!(macos_reclaim_free_bytes(&words, 38, 0, 8 * GIB), None);
        assert_eq!(
            macos_reclaim_free_bytes(&words, 0, 16 * 1_024, 8 * GIB),
            None
        );
    }

    #[test]
    fn macos_probe_never_substitutes_total_ram_for_a_failed_vm_snapshot() {
        assert_eq!(cap_available_to_physical(None, Some(64 * GIB)), None);
        assert_eq!(
            cap_available_to_physical(Some(80 * GIB), Some(64 * GIB)),
            Some(64 * GIB)
        );
    }

    #[test]
    fn reclaimable_macos_file_cache_admits_small_unified_growth() {
        let page = 16 * 1_024_u64;
        let pages = |bytes: u64| u32::try_from(bytes / page).unwrap();
        let mut words = [0_u32; 38];
        words[0] = pages(8 * GIB);
        words[23] = pages(GIB);
        words[34] = pages(24 * GIB);
        let available =
            macos_reclaim_free_bytes(&words, 38, page, MACOS_FILE_CACHE_RESERVE_BYTES).unwrap();
        assert_eq!(available, 23 * GIB);

        let selection = select_resident_prefix(
            host(64 * GIB, available),
            ProviderMemory::Unified {
                recommended_working_set_bytes: None,
                available_working_set_bytes: None,
            },
            &[],
            FixedCosts {
                host_bytes: 0,
                provider_bytes: 256 << 20,
            },
            ResidencyOverride::default(),
            ResidencyPolicy::default(),
        );
        assert_eq!(selection.stop, ResidencyStop::AllLayersFit);

        let old_free_only = select_resident_prefix(
            host(64 * GIB, 8 * GIB),
            ProviderMemory::Unified {
                recommended_working_set_bytes: None,
                available_working_set_bytes: None,
            },
            &[],
            FixedCosts {
                host_bytes: 0,
                provider_bytes: 256 << 20,
            },
            ResidencyOverride::default(),
            ResidencyPolicy::default(),
        );
        assert_eq!(old_free_only.stop, ResidencyStop::FixedCostsExceedBudget);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_native_probe_returns_bounded_kernel_values() {
        let observed = probe_host_memory();
        let physical = observed.physical_bytes.expect("hw.memsize");
        let available = observed.available_bytes.expect("host_statistics64");
        assert!(physical > 0);
        assert!(available <= physical);
        assert!(observed.constraints_readable);
    }

    mod linux {
        use super::*;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

        struct Fixture(PathBuf);

        impl Fixture {
            fn new() -> Self {
                let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "deltafin-residency-{}-{serial}",
                    std::process::id()
                ));
                fs::create_dir_all(&path).unwrap();
                Self(path)
            }

            fn write(&self, relative: &str, value: &str) -> PathBuf {
                let path = self.0.join(relative);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).unwrap();
                }
                fs::write(&path, value).unwrap();
                path
            }
        }

        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        #[test]
        fn v2_uses_the_tightest_leaf_or_parent_constraint() {
            let fixture = Fixture::new();
            let meminfo = fixture.write("meminfo", "MemTotal: 100000 kB\nMemAvailable: 70000 kB\n");
            let self_cgroup = fixture.write("self.cgroup", "0::/tenant/job\n");
            let root = fixture.0.join("cg");
            fixture.write("cg/memory.max", "max\n");
            fixture.write("cg/memory.current", "0\n");
            fixture.write("cg/tenant/memory.max", "90000000\n");
            fixture.write("cg/tenant/memory.current", "50000000\n");
            fixture.write("cg/tenant/job/memory.max", "80000000\n");
            fixture.write("cg/tenant/job/memory.current", "30000000\n");

            let observed = probe_linux_memory(&meminfo, &self_cgroup, &root);
            assert_eq!(observed.cgroup_limit_bytes, Some(80_000_000));
            assert_eq!(observed.cgroup_available_bytes, Some(40_000_000));
            assert_eq!(observed.effective_total_bytes(), Some(80_000_000));
            assert_eq!(observed.effective_available_bytes(), Some(40_000_000));
        }

        #[test]
        fn v1_memory_controller_and_unlimited_sentinel_are_understood() {
            let fixture = Fixture::new();
            let meminfo = fixture.write("meminfo", "MemTotal: 65536 kB\nMemAvailable: 32768 kB\n");
            let self_cgroup = fixture.write("self.cgroup", "7:cpu:/x\n5:memory:/job\n");
            let root = fixture.0.join("cg");
            fixture.write(
                "cg/memory/job/memory.limit_in_bytes",
                "9223372036854771712\n",
            );
            fixture.write("cg/memory/job/memory.usage_in_bytes", "1\n");
            fixture.write("cg/memory/memory.limit_in_bytes", "9223372036854771712\n");
            fixture.write("cg/memory/memory.usage_in_bytes", "1\n");

            let observed = probe_linux_memory(&meminfo, &self_cgroup, &root);
            assert!(observed.constraints_readable);
            assert_eq!(observed.cgroup_limit_bytes, None);
            assert_eq!(observed.effective_total_bytes(), Some(65_536 * 1_024));
            assert_eq!(observed.effective_available_bytes(), Some(32_768 * 1_024));
        }

        #[test]
        fn declared_but_unreadable_cgroup_never_falls_back_to_host_ram() {
            let fixture = Fixture::new();
            let meminfo = fixture.write("meminfo", "MemTotal: 100000 kB\nMemAvailable: 90000 kB\n");
            let self_cgroup = fixture.write("self.cgroup", "0::/missing\n");
            let observed = probe_linux_memory(&meminfo, &self_cgroup, &fixture.0.join("cg"));
            assert!(!observed.constraints_readable);
            assert_eq!(observed.effective_total_bytes(), None);

            let selection = select_resident_prefix(
                observed,
                ProviderMemory::Host,
                &[1],
                FixedCosts::default(),
                ResidencyOverride::default(),
                no_reserve_policy(),
            );
            assert_eq!(selection.stop, ResidencyStop::HostBudgetUnknown);
        }

        #[test]
        fn cgroup_path_cannot_escape_the_supplied_root() {
            let root = Path::new("/safe/cgroup");
            assert!(safe_cgroup_join(root, "/tenant/job").is_some());
            assert!(safe_cgroup_join(root, "/../../outside").is_none());
        }

        #[test]
        fn malformed_or_escaping_declared_hierarchy_fails_closed() {
            let fixture = Fixture::new();
            let meminfo = fixture.write("meminfo", "MemTotal: 100000 kB\nMemAvailable: 90000 kB\n");
            for (serial, contents) in ["malformed", "0::/../../escape"].into_iter().enumerate() {
                let self_cgroup = fixture.write(&format!("self-{serial}.cgroup"), contents);
                let observed = probe_linux_memory(&meminfo, &self_cgroup, &fixture.0.join("cg"));
                assert!(!observed.constraints_readable);
                assert_eq!(observed.effective_total_bytes(), None);
            }
        }
    }
}
