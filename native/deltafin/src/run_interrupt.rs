//! Cooperative cancellation sources for native target generation.
//!
//! The CLI's SIGINT handler performs exactly one lock-free atomic store, and
//! the OpenAI server adapts its client-liveness probe to the same
//! [`InterruptSource`] trait.  Target code observes cancellation only at
//! transaction boundaries, so neither a signal nor a client disconnect can
//! ever unwind Rust, C++, Metal, or CUDA while provider-owned state is being
//! mutated.  The server never constructs the SIGINT guard and retains the
//! operating system's normal signal disposition.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{DeltafinError, Result};

#[cfg(all(
    any(target_os = "macos", target_os = "linux"),
    not(target_has_atomic = "8")
))]
compile_error!("cooperative SIGINT requires a lock-free 8-bit atomic target");

static RUN_INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);
#[cfg(any(target_os = "macos", target_os = "linux", windows))]
static RUN_HANDLER_ARMED: AtomicBool = AtomicBool::new(false);

/// Read-only cancellation source used by the target transaction loop.
pub(crate) trait InterruptSource {
    fn requested(&self) -> bool;
}

/// A sticky atomic source. Tests use a local atomic; the CLI guard exposes the
/// process-global flag written by the async-signal-safe handler.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AtomicInterrupt<'a> {
    requested: &'a AtomicBool,
}

impl<'a> AtomicInterrupt<'a> {
    const fn new(requested: &'a AtomicBool) -> Self {
        Self { requested }
    }
}

impl InterruptSource for AtomicInterrupt<'_> {
    #[inline(always)]
    fn requested(&self) -> bool {
        self.requested.load(Ordering::Relaxed)
    }
}

/// Restores the exact prior SIGINT disposition when the one CLI run ends.
/// Construction is intentionally absent from every server and library path.
pub(crate) struct RunInterruptGuard {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    previous: libc::sigaction,
    #[cfg(windows)]
    _ctrl_c: deltafin_sys::console::CtrlCGuard,
}

impl RunInterruptGuard {
    pub(crate) fn arm() -> Result<Self> {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            if RUN_HANDLER_ARMED
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return Err(DeltafinError::new(
                    "native run SIGINT handler is already armed",
                ));
            }
            RUN_INTERRUPT_REQUESTED.store(false, Ordering::Relaxed);

            // SAFETY: zero is the platform-defined empty baseline for sigaction;
            // sigemptyset initializes the mask before sigaction observes it.
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction = record_sigint as *const () as libc::sighandler_t;
            action.sa_flags = libc::SA_RESTART;
            // SAFETY: `action.sa_mask` is valid writable storage owned here.
            if unsafe { libc::sigemptyset(&mut action.sa_mask) } != 0 {
                RUN_HANDLER_ARMED.store(false, Ordering::Release);
                return Err(last_signal_error("initialize native run SIGINT mask"));
            }
            // SAFETY: both sigaction pointers remain valid for the synchronous
            // call; `previous` is initialized by libc on success.
            let mut previous: libc::sigaction = unsafe { std::mem::zeroed() };
            if unsafe { libc::sigaction(libc::SIGINT, &action, &mut previous) } != 0 {
                RUN_HANDLER_ARMED.store(false, Ordering::Release);
                return Err(last_signal_error("arm native run SIGINT handler"));
            }
            Ok(Self { previous })
        }

        #[cfg(windows)]
        {
            if RUN_HANDLER_ARMED
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return Err(DeltafinError::new(
                    "native run Ctrl-C handler is already armed",
                ));
            }
            RUN_INTERRUPT_REQUESTED.store(false, Ordering::Relaxed);
            match deltafin_sys::console::interrupt_on_ctrl_c(&RUN_INTERRUPT_REQUESTED) {
                Ok(guard) => Ok(Self { _ctrl_c: guard }),
                Err(error) => {
                    RUN_HANDLER_ARMED.store(false, Ordering::Release);
                    Err(DeltafinError::new(format!(
                        "arm native run Ctrl-C handler: {error}"
                    )))
                }
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
        {
            Err(DeltafinError::new(
                "cooperative native-run Ctrl-C is supported on macOS, Linux and Windows",
            ))
        }
    }

    pub(crate) const fn source(&self) -> AtomicInterrupt<'_> {
        AtomicInterrupt::new(&RUN_INTERRUPT_REQUESTED)
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
extern "C" fn record_sigint(_signal: libc::c_int) {
    // A lock-free atomic store is the handler's only operation. In particular,
    // it performs no allocation, I/O, locking, unwinding, or provider call.
    RUN_INTERRUPT_REQUESTED.store(true, Ordering::Relaxed);
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn last_signal_error(operation: &str) -> DeltafinError {
    DeltafinError::new(format!("{operation}: {}", std::io::Error::last_os_error()))
}

impl Drop for RunInterruptGuard {
    fn drop(&mut self) {
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            // SAFETY: `previous` was filled by the successful arm operation
            // and remains live for this synchronous restoration call.
            let _ = unsafe { libc::sigaction(libc::SIGINT, &self.previous, std::ptr::null_mut()) };
            RUN_INTERRUPT_REQUESTED.store(false, Ordering::Relaxed);
            RUN_HANDLER_ARMED.store(false, Ordering::Release);
        }
        // The console guard disarms the handler when its field drops; the
        // flags are cleared here, after, for the next run.
        #[cfg(windows)]
        {
            RUN_INTERRUPT_REQUESTED.store(false, Ordering::Relaxed);
            RUN_HANDLER_ARMED.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_atomic_source_is_sticky_without_signalling_the_process() {
        let requested = AtomicBool::new(false);
        let source = AtomicInterrupt::new(&requested);
        assert!(!source.requested());
        requested.store(true, Ordering::Relaxed);
        assert!(source.requested());
        assert!(source.requested());
    }

}
