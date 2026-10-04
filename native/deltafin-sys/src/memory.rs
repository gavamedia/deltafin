//! Physical memory and memory-limit probing (Windows).
//!
//! Linux and macOS read `/proc` and Mach statistics directly in the residency
//! planner; Windows asks the kernel through `GlobalMemoryStatusEx`, and treats
//! a job-object memory limit (how a container or a managed launcher caps a
//! process) as the analogue of a Linux cgroup limit.

use std::ffi::c_void;
use std::mem::{size_of, zeroed};
use std::ptr::null_mut;

use windows_sys::Win32::System::JobObjects::{
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectExtendedLimitInformation, QueryInformationJobObject,
};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

/// A snapshot of host memory.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SystemMemory {
    /// Installed physical memory visible to the OS.
    pub physical_bytes: u64,
    /// Physical memory that can be given to a process without paging out
    /// active memory (free plus reclaimable standby pages).
    pub available_bytes: u64,
    /// The smallest memory limit of the job this process runs in, if any.
    pub job_limit_bytes: Option<u64>,
}

/// Snapshot the machine's memory, or `None` if the kernel will not say.
pub fn system_memory() -> Option<SystemMemory> {
    // SAFETY: an all-zero MEMORYSTATUSEX is a valid out value once its length
    // field is set, which is the documented precondition of the call.
    let mut status: MEMORYSTATUSEX = unsafe { zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is writable storage of the exact type the call fills.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return None;
    }
    Some(SystemMemory {
        physical_bytes: status.ullTotalPhys,
        available_bytes: status.ullAvailPhys,
        job_limit_bytes: job_memory_limit(),
    })
}

/// The tightest memory limit of the calling process's job. A process that is
/// not in a job (the ordinary desktop case) has none: the query fails and that
/// is reported as no limit, not as an unknown one.
fn job_memory_limit() -> Option<u64> {
    // SAFETY: plain data; zero means "no limits set".
    let mut information: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { zeroed() };
    // SAFETY: a null job handle means the calling process's own job; the
    // buffer is exactly the size of the structure the class returns.
    let queried = unsafe {
        QueryInformationJobObject(
            null_mut(),
            JobObjectExtendedLimitInformation,
            std::ptr::from_mut(&mut information).cast::<c_void>(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            null_mut(),
        )
    };
    if queried == 0 {
        return None;
    }
    let flags = information.BasicLimitInformation.LimitFlags;
    let job = (flags & JOB_OBJECT_LIMIT_JOB_MEMORY != 0).then_some(information.JobMemoryLimit as u64);
    let process =
        (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY != 0).then_some(information.ProcessMemoryLimit as u64);
    match (job, process) {
        (Some(job), Some(process)) => Some(job.min(process)),
        (limit, None) | (None, limit) => limit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_machine_reports_plausible_memory() {
        let memory = system_memory().expect("GlobalMemoryStatusEx must work");
        assert!(memory.physical_bytes >= 1 << 30, "{memory:?}");
        assert!(memory.available_bytes > 0, "{memory:?}");
        assert!(memory.available_bytes <= memory.physical_bytes, "{memory:?}");
        if let Some(limit) = memory.job_limit_bytes {
            assert!(limit > 0, "{memory:?}");
        }
    }
}
