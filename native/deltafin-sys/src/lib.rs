//! Operating-system primitives shared by Deltafin's native crates.
//!
//! The runtime's integrity story rests on a handful of narrow OS operations:
//! open a path without following a final symlink, pin the identity of what was
//! opened, read at an offset from many threads at once, publish a file without
//! replacing a racer, and hold an advisory lock. On Unix these are the POSIX
//! calls the runtime has always made, collected here unchanged. On Windows
//! each one is built from its Win32 equivalent, so the callers above this
//! crate carry no `cfg` of their own.
//!
//! This crate is deliberately a leaf: it depends on nothing but `libc` (Unix)
//! or `windows-sys` (Windows), so the build-time installer, the build graph
//! and the runtime can all share one implementation, and its tests run on
//! their own on every host.

// Each platform implements the whole surface; an item that only one platform's
// callers happen to use is still part of the contract.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(windows)]
pub mod console;
pub mod fs;
#[cfg(windows)]
pub mod https;
pub mod limits;
#[cfg(windows)]
pub mod memory;
pub mod path;
pub mod pe;
