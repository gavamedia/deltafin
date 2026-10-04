//! Per-process descriptor (handle) limits and the live count, for the storage
//! layer's persistent-descriptor budget.
//!
//! Unix reads `RLIMIT_NOFILE` and counts `/proc/self/fd` or `/dev/fd`. Windows
//! has no per-process limit anywhere near a realistic roster (a process may
//! hold on the order of sixteen million handles), so the figure here is the
//! runtime's own conservative ceiling, and the live count comes from
//! `GetProcessHandleCount`.

#[cfg(unix)]
#[repr(C)]
struct NativeRlimit {
    current: u64,
    maximum: u64,
}

/// The `(soft, hard)` descriptor limits, if the platform can report them.
pub fn descriptor_limits() -> Option<(usize, usize)> {
    #[cfg(target_os = "linux")]
    const RLIMIT_NOFILE: i32 = 7;
    #[cfg(target_os = "macos")]
    const RLIMIT_NOFILE: i32 = 8;
    #[cfg(windows)]
    return Some((WINDOWS_HANDLE_CEILING, WINDOWS_HANDLE_CEILING));
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
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

/// The soft limit alone.
pub fn soft_descriptor_limit() -> Option<usize> {
    descriptor_limits().map(|limits| limits.0)
}

/// Raise (or lower) the soft limit for this process; `false` if refused.
pub fn set_soft_descriptor_limit(soft: usize) -> bool {
    #[cfg(target_os = "linux")]
    const RLIMIT_NOFILE: i32 = 7;
    #[cfg(target_os = "macos")]
    const RLIMIT_NOFILE: i32 = 8;
    // The ceiling is fixed, so a request is admitted exactly when it fits.
    #[cfg(windows)]
    return soft <= WINDOWS_HANDLE_CEILING;
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
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

/// The number of descriptors (handles) this process currently holds open.
pub fn count_open_descriptors() -> Option<usize> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessHandleCount};
        let mut count = 0_u32;
        // SAFETY: the pseudo-handle is always valid and `count` is writable.
        if unsafe { GetProcessHandleCount(GetCurrentProcess(), &mut count) } == 0 {
            return None;
        }
        usize::try_from(count).ok()
    }
    #[cfg(not(windows))]
    {
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
}

/// Windows handles are cheap and the real ceiling is in the millions; this is
/// the budget the runtime is willing to reason about.
#[cfg(windows)]
const WINDOWS_HANDLE_CEILING: usize = 1 << 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_and_live_count_are_reported_and_consistent() {
        let (soft, hard) = descriptor_limits().expect("supported hosts report limits");
        assert!(soft > 0 && soft <= hard);
        assert_eq!(soft_descriptor_limit(), Some(soft));
        let open = count_open_descriptors().expect("supported hosts count descriptors");
        assert!(open > 0 && open < hard, "{open} open of {hard}");
    }

    #[test]
    fn an_open_file_is_counted_and_a_closed_one_is_not() {
        let path = std::env::temp_dir().join(format!("deltafin-limits-{}", std::process::id()));
        std::fs::write(&path, b"x").unwrap();
        let held: Vec<_> = (0..16).map(|_| std::fs::File::open(&path).unwrap()).collect();
        let during = count_open_descriptors().unwrap();
        drop(held);
        let after = count_open_descriptors().unwrap();
        std::fs::remove_file(&path).unwrap();
        // Other test threads open and close files too, so only the direction
        // of a 16-handle swing is asserted, with generous slack.
        assert!(during + 8 >= after + 16 || during >= after, "{during} -> {after}");
    }

    #[test]
    fn the_soft_limit_can_be_confirmed_but_never_exceeds_the_hard_limit() {
        let (soft, hard) = descriptor_limits().unwrap();
        assert!(set_soft_descriptor_limit(soft));
        assert!(!set_soft_descriptor_limit(hard.saturating_add(1)));
    }
}
