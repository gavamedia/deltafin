//! File-system primitives with one portable shape.
//!
//! * [`Open`] opens a path with the safety properties the runtime relies on
//!   (no following of a final symlink, a directory handle, concurrent
//!   positional reads) without the caller spelling any platform flag.
//! * [`Stat`] / [`FileId`] snapshot a file's type, length, identity and times
//!   from a path (`lstat`) or an open handle (`fstat`), so the
//!   "stat, open, re-stat, compare identity" pattern is the same everywhere.
//! * [`FileExt`] supplies `read_at`/`write_at` on every platform.
//!
//! The Unix implementation is the code the runtime ran before this module
//! existed; the Windows one is documented where it differs.

use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
use unix as imp;
#[cfg(windows)]
use windows as imp;

#[allow(unused_imports)]
pub use imp::{
    FileExt, ReadOnlyMap, Stat, available_space, canonicalize, create_private_directory, fstat,
    is_descriptor_exhaustion, lstat, publish_without_replace, rename_noreplace, restrict_to_owner,
    set_mode, stat, symlink, sync_directory, system_directory, try_lock_exclusive, unlock,
};

/// The identity of a file object: its volume plus its index on that volume
/// (`st_dev`/`st_ino` on Unix, volume serial/file id on Windows). Two paths
/// name the same file exactly when their ids are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId {
    pub device: u64,
    pub inode: u64,
}

/// What a path or handle is, with links classified the way `lstat` would.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileKind {
    kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    File,
    Directory,
    Symlink,
    Other,
}

impl FileKind {
    pub const fn is_file(self) -> bool {
        matches!(self.kind, Kind::File)
    }

    pub const fn is_dir(self) -> bool {
        matches!(self.kind, Kind::Directory)
    }

    pub const fn is_symlink(self) -> bool {
        matches!(self.kind, Kind::Symlink)
    }

    #[cfg(any(windows, test))]
    const fn of(kind: Kind) -> Self {
        Self { kind }
    }
}

/// A file-open request. It replaces `OpenOptions` plus a per-platform pile of
/// `custom_flags`: the safety properties are named, and each platform maps
/// them to what it actually has.
#[derive(Debug, Clone, Default)]
pub struct Open {
    read: bool,
    write: bool,
    append: bool,
    create: bool,
    create_new: bool,
    truncate: bool,
    mode: Option<u32>,
    no_follow: bool,
    directory: bool,
    concurrent_reads: bool,
    hold_in_place: bool,
}

impl Open {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read(&mut self, value: bool) -> &mut Self {
        self.read = value;
        self
    }

    pub fn write(&mut self, value: bool) -> &mut Self {
        self.write = value;
        self
    }

    pub fn append(&mut self, value: bool) -> &mut Self {
        self.append = value;
        self
    }

    pub fn create(&mut self, value: bool) -> &mut Self {
        self.create = value;
        self
    }

    pub fn create_new(&mut self, value: bool) -> &mut Self {
        self.create_new = value;
        self
    }

    pub fn truncate(&mut self, value: bool) -> &mut Self {
        self.truncate = value;
        self
    }

    /// Permission bits for a file this call creates. Windows files take the
    /// ACL inherited from their directory, so the mode is ignored there.
    pub fn mode(&mut self, mode: u32) -> &mut Self {
        self.mode = Some(mode);
        self
    }

    /// Refuse a final path component that is a symbolic link (and, on
    /// Windows, a junction). The handle is also closed on `exec`.
    pub fn no_follow(&mut self) -> &mut Self {
        self.no_follow = true;
        self
    }

    /// Open a directory (rather than a file) so it can be inspected or synced.
    pub fn directory(&mut self) -> &mut Self {
        self.directory = true;
        self
    }

    /// The handle will serve `read_at` from many threads at once. Windows then
    /// opens it for overlapped I/O so those reads do not queue behind one
    /// another; Unix `pread` always runs concurrently.
    pub fn concurrent_reads(&mut self) -> &mut Self {
        self.concurrent_reads = true;
        self
    }

    /// Keep the opened object from being renamed or deleted while the handle
    /// lives, so the path it was opened by stays valid for later relative
    /// work. Windows withholds `FILE_SHARE_DELETE`; Unix has no equivalent and
    /// uses directory-relative opens (`openat`) for the same guarantee.
    pub fn hold_in_place(&mut self) -> &mut Self {
        self.hold_in_place = true;
        self
    }

    pub fn open(&self, path: &Path) -> io::Result<File> {
        imp::open(self, path)
    }
}

#[cfg(test)]
mod tests;
