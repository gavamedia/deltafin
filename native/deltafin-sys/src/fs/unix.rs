//! POSIX implementation of the portable file-system surface.
//!
//! Everything here is what the runtime did before this module existed: the
//! same flags, the same calls, collected in one place. One deliberate
//! difference: the no-follow flags come from `libc` for the target ABI instead
//! of a per-file literal. `0x000a_0000` is `O_NOFOLLOW | O_CLOEXEC` only on
//! x86-64 Linux; on aarch64 Linux it is `O_LARGEFILE | O_CLOEXEC`, so a
//! literal silently dropped the symlink guard there.

use std::ffi::CString;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::{FileId, FileKind, Kind, Open};

pub use std::os::unix::fs::FileExt;

pub(super) fn open(spec: &Open, path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options
        .read(spec.read)
        .write(spec.write)
        .append(spec.append)
        .create(spec.create)
        .create_new(spec.create_new)
        .truncate(spec.truncate);
    if let Some(mode) = spec.mode {
        options.mode(mode);
    }
    let mut flags = 0;
    if spec.no_follow {
        flags |= libc::O_NOFOLLOW | libc::O_CLOEXEC;
    }
    if spec.directory {
        flags |= libc::O_DIRECTORY;
    }
    if flags != 0 {
        options.custom_flags(flags);
    }
    options.open(path)
}

/// A snapshot of one file's type, size, identity, ownership and times.
#[derive(Debug, Clone)]
pub struct Stat(Metadata);

impl Stat {
    pub fn len(&self) -> u64 {
        self.0.len()
    }

    pub fn file_type(&self) -> FileKind {
        let kind = self.0.file_type();
        FileKind {
            kind: if kind.is_symlink() {
                Kind::Symlink
            } else if kind.is_dir() {
                Kind::Directory
            } else if kind.is_file() {
                Kind::File
            } else {
                Kind::Other
            },
        }
    }

    pub fn is_file(&self) -> bool {
        self.0.is_file()
    }

    pub fn is_symlink(&self) -> bool {
        self.0.file_type().is_symlink()
    }

    pub fn is_dir(&self) -> bool {
        self.0.is_dir()
    }

    pub fn id(&self) -> FileId {
        FileId {
            device: self.0.dev(),
            inode: self.0.ino(),
        }
    }

    pub fn dev(&self) -> u64 {
        self.0.dev()
    }

    pub fn ino(&self) -> u64 {
        self.0.ino()
    }

    pub fn mode(&self) -> u32 {
        self.0.mode()
    }

    pub fn uid(&self) -> u32 {
        self.0.uid()
    }

    /// Whether the effective user owns the object and neither group nor others
    /// can write it: the "owner-controlled" test for a tool or toolchain file
    /// the runtime is about to trust.
    pub fn is_owner_controlled(&self) -> bool {
        // SAFETY: geteuid has no arguments, memory effects, or failure return.
        self.0.uid() == unsafe { libc::geteuid() } && !self.is_shared_writable()
    }

    /// Whether group or others can write the object.
    pub fn is_shared_writable(&self) -> bool {
        self.0.mode() & 0o022 != 0
    }

    /// Whether group and others have no access: the owner-only privacy the
    /// runtime requires of lock files, receipts and evidence.
    pub fn is_owner_only(&self) -> bool {
        self.0.mode() & 0o077 == 0
    }

    pub fn nlink(&self) -> u64 {
        self.0.nlink()
    }

    pub fn mtime(&self) -> i64 {
        self.0.mtime()
    }

    pub fn mtime_nsec(&self) -> i64 {
        self.0.mtime_nsec()
    }

    pub fn ctime(&self) -> i64 {
        self.0.ctime()
    }

    pub fn ctime_nsec(&self) -> i64 {
        self.0.ctime_nsec()
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        self.0.modified()
    }
}

/// A private, read-only mapping of a whole admitted file. The file must stay
/// open (and unmodified) for as long as the mapping is read.
pub struct ReadOnlyMap {
    address: *mut libc::c_void,
    length: usize,
}

impl ReadOnlyMap {
    pub fn map(file: &File, length: usize) -> io::Result<Self> {
        // SAFETY: the admitted regular file is live, and this is a private
        // read-only mapping at offset zero (page aligned).
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { address, length })
    }

    pub fn address(&self) -> *const u8 {
        self.address.cast::<u8>().cast_const()
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

impl Drop for ReadOnlyMap {
    fn drop(&mut self) {
        // SAFETY: this object owns exactly this successful mapping.
        let _ = unsafe { libc::munmap(self.address, self.length) };
    }
}

/// Create a directory only its owner can enter (`0o700`).
pub fn create_private_directory(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(path)
}

/// Create a symbolic link at `link` pointing at `target`.
pub fn symlink<T: AsRef<Path>, L: AsRef<Path>>(target: T, link: L) -> io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

/// Apply Unix permission bits to a path.
pub fn set_mode(path: &Path, mode: u32) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
}

/// Restrict an existing directory to its owner (`0o700`).
pub fn restrict_to_owner(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

/// Whether an error means the process or system ran out of descriptors
/// (`EMFILE`/`ENFILE`), as opposed to a problem with the file itself.
pub fn is_descriptor_exhaustion(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(libc::EMFILE | libc::ENFILE))
}

/// Stat a path without following a final symlink.
pub fn lstat(path: &Path) -> io::Result<Stat> {
    fs::symlink_metadata(path).map(Stat)
}

/// Stat a path, following symlinks.
pub fn stat(path: &Path) -> io::Result<Stat> {
    fs::metadata(path).map(Stat)
}

/// Stat an open handle: what was actually opened, whatever the path now says.
pub fn fstat(file: &File) -> io::Result<Stat> {
    file.metadata().map(Stat)
}

/// Resolve a path to its canonical absolute form.
pub fn canonicalize<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    fs::canonicalize(path)
}

/// The operating system's own program directory (`System32` on Windows).
/// Unix has no such single directory: its loader consults a configured set,
/// which the ELF and Mach-O audits model themselves.
pub fn system_directory() -> io::Result<PathBuf> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Flush a directory's entries to stable storage through an open directory
/// handle, so a rename or unlink inside it survives a crash.
pub fn sync_directory(directory: &File) -> io::Result<()> {
    directory.sync_all()
}

/// Take an exclusive, non-blocking advisory lock. `Ok(false)` means another
/// holder has it; the lock lasts until [`unlock`] or the handle closes.
pub fn try_lock_exclusive(file: &File) -> io::Result<bool> {
    // SAFETY: `file` owns a live descriptor for the call and flock does not
    // retain it.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(error)
    }
}

/// Release a lock taken by [`try_lock_exclusive`].
pub fn unlock(file: &File) {
    // SAFETY: the descriptor remains live for the duration of this call.
    let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
}

/// Bytes available to this user on the filesystem holding an open directory.
/// `f_bavail`, not `f_bfree`: quotas and blocks reserved from the current user
/// must not make a capacity gate optimistic.
pub fn available_space(directory: &File, _path: &Path) -> io::Result<u64> {
    // SAFETY: `fstatvfs` initializes the entire output on success, the file
    // descriptor remains live for the call, and the output points to writable
    // storage of the exact libc type for this target.
    let filesystem = unsafe {
        let mut value = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        if libc::fstatvfs(directory.as_raw_fd(), value.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error());
        }
        value.assume_init()
    };
    let fragment_bytes = if filesystem.f_frsize == 0 {
        filesystem.f_bsize
    } else {
        filesystem.f_frsize
    };
    u64::from(filesystem.f_bavail)
        .checked_mul(fragment_bytes)
        .ok_or_else(|| io::Error::other("filesystem available byte count overflowed"))
}

/// Atomically rename without ever replacing a path that raced into place.
pub fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    let source_c = CString::new(source.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "rename source contains a NUL byte")
    })?;
    let destination_c = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "rename destination contains a NUL byte",
        )
    })?;
    #[cfg(target_os = "macos")]
    // SAFETY: both C strings are live, NUL terminated, and retained for the call.
    let status = unsafe { renamex_np(source_c.as_ptr(), destination_c.as_ptr(), RENAME_EXCL) };
    #[cfg(target_os = "linux")]
    // SAFETY: both C strings are live, NUL terminated, and retained for the call.
    let status = unsafe {
        renameat2(
            AT_FDCWD,
            source_c.as_ptr(),
            AT_FDCWD,
            destination_c.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Make `destination` hold `part`'s content without ever replacing an existing
/// `destination`, leaving no `part` behind: a hard link, then an unlink.
pub fn publish_without_replace(part: &Path, destination: &Path) -> io::Result<()> {
    fs::hard_link(part, destination)?;
    fs::remove_file(part).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("remove published partial link {}: {error}", part.display()),
        )
    })
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn renamex_np(
        old: *const std::os::raw::c_char,
        new: *const std::os::raw::c_char,
        flags: u32,
    ) -> i32;
}
#[cfg(target_os = "macos")]
const RENAME_EXCL: u32 = 0x0000_0004;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn renameat2(
        olddirfd: i32,
        old: *const std::os::raw::c_char,
        newdirfd: i32,
        new: *const std::os::raw::c_char,
        flags: u32,
    ) -> i32;
}
#[cfg(target_os = "linux")]
const AT_FDCWD: i32 = -100;
#[cfg(target_os = "linux")]
const RENAME_NOREPLACE: u32 = 1;
