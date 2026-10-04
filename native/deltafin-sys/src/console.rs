//! Cooperative Ctrl-C for a console process (Windows).
//!
//! Unix delivers SIGINT to a signal handler; Windows runs a console control
//! handler on a thread the system creates for the purpose. Either way the
//! handler must do exactly one lock-free atomic store and nothing else, so the
//! generation loop observes the request only at a transaction boundary.

use std::io;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler};

/// The flag the handler sets; null while no guard is armed.
static TARGET: AtomicPtr<AtomicBool> = AtomicPtr::new(null_mut());

unsafe extern "system" fn handler(event: u32) -> i32 {
    if event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT {
        let target = TARGET.load(Ordering::Acquire);
        if !target.is_null() {
            // SAFETY: the pointer came from a `&'static AtomicBool` stored by
            // `interrupt_on_ctrl_c` and is cleared only after the handler is
            // removed.
            unsafe { (*target).store(true, Ordering::Relaxed) };
            // Handled: do not run the default handler, which ends the process.
            return 1;
        }
    }
    // Close, logoff and shutdown events keep their default handling.
    0
}

/// Disarms the handler when dropped, restoring the default Ctrl-C behavior.
#[derive(Debug)]
pub struct CtrlCGuard(());

/// Route Ctrl-C and Ctrl-Break to `flag` until the returned guard is dropped.
/// Only one guard may be armed at a time.
pub fn interrupt_on_ctrl_c(flag: &'static AtomicBool) -> io::Result<CtrlCGuard> {
    let pointer = std::ptr::from_ref(flag).cast_mut();
    if TARGET
        .compare_exchange(null_mut(), pointer, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(io::Error::other("a Ctrl-C handler is already armed"));
    }
    // SAFETY: `handler` is a valid console control routine for the life of
    // the process.
    if unsafe { SetConsoleCtrlHandler(Some(handler), 1) } == 0 {
        TARGET.store(null_mut(), Ordering::Release);
        return Err(io::Error::last_os_error());
    }
    Ok(CtrlCGuard(()))
}

impl Drop for CtrlCGuard {
    fn drop(&mut self) {
        // SAFETY: removes the routine added by `interrupt_on_ctrl_c`.
        unsafe { SetConsoleCtrlHandler(Some(handler), 0) };
        TARGET.store(null_mut(), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard};

    use super::*;

    static FLAG: AtomicBool = AtomicBool::new(false);

    /// The handler is process-global, so tests that arm it take turns: running
    /// them on parallel test threads would make one test's guard the "already
    /// armed" error of another.
    static ARMING: Mutex<()> = Mutex::new(());

    fn one_at_a_time() -> MutexGuard<'static, ()> {
        ARMING.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn one_guard_at_a_time_and_disarming_allows_the_next() {
        let _turn = one_at_a_time();
        let first = interrupt_on_ctrl_c(&FLAG).expect("arming must work");
        assert!(interrupt_on_ctrl_c(&FLAG).is_err(), "a second guard was admitted");
        drop(first);
        let again = interrupt_on_ctrl_c(&FLAG).expect("re-arming after a drop must work");
        drop(again);
        assert!(!FLAG.load(Ordering::Relaxed), "arming must not itself request an interrupt");
    }

    #[test]
    fn the_handler_sets_the_flag_for_ctrl_c_and_ignores_other_events() {
        static SEEN: AtomicBool = AtomicBool::new(false);
        let _turn = one_at_a_time();
        let guard = interrupt_on_ctrl_c(&SEEN).expect("arming must work");
        // SAFETY: the handler is a plain function; call it as the system would.
        assert_eq!(unsafe { handler(CTRL_C_EVENT) }, 1);
        assert!(SEEN.load(Ordering::Relaxed));
        SEEN.store(false, Ordering::Relaxed);
        assert_eq!(unsafe { handler(CTRL_BREAK_EVENT) }, 1);
        assert!(SEEN.load(Ordering::Relaxed));
        SEEN.store(false, Ordering::Relaxed);
        // CTRL_CLOSE_EVENT (2) is not ours to absorb.
        assert_eq!(unsafe { handler(2) }, 0);
        assert!(!SEEN.load(Ordering::Relaxed));
        drop(guard);
    }
}
