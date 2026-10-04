//! Win32 implementation of the portable file-system surface.
//!
//! How each Unix guarantee is delivered here:
//!
//! * **No-follow open.** The path is opened with `FILE_FLAG_OPEN_REPARSE_POINT`
//!   so a link is opened *as* a link, and the handle is then inspected: a
//!   symbolic link or junction (any "name surrogate" reparse point) is
//!   refused. A reparse point that is not a link (a cloud placeholder, a
//!   compressed-file filter, a deduplicated file) must still be read through
//!   its filter, so it is reopened normally and the two handles' identities
//!   are compared to prove nothing was swapped in between.
//! * **Identity.** Volume serial plus file id from
//!   `GetFileInformationByHandleEx(FileIdInfo)`, falling back to the 64-bit
//!   index where a filesystem has no 128-bit id. The std accessors for these
//!   are unstable, so they are read directly.
//! * **Positional reads.** A synchronous Windows handle serializes every read
//!   on it, which would turn the worker pools into a queue. `ReadFile` with an
//!   `OVERLAPPED` offset on a handle opened for overlapped I/O lets reads from
//!   many threads be in flight together; each read still blocks its own
//!   caller, on a per-thread event.
//! * **Locks.** `LockFileEx` on one byte far beyond any data. Windows locks are
//!   mandatory for I/O on the locked range, so the range must not be one
//!   anyone reads.

use std::ffi::c_void;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::mem::{size_of, zeroed};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::ptr::null;
use std::time::{SystemTime, UNIX_EPOCH};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, ERROR_LOCK_VIOLATION, GetLastError, HANDLE,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_READONLY, FILE_ATTRIBUTE_REPARSE_POINT,
    FILE_BASIC_INFO, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_FLAG_OVERLAPPED, FILE_ID_INFO, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE,
    FileBasicInfo, FileIdInfo,
    GetDiskFreeSpaceExW, GetFileInformationByHandle, GetFileInformationByHandleEx,
    LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx, MOVEFILE_WRITE_THROUGH,
    MoveFileExW, ReadFile, UnlockFileEx, WriteFile,
};
use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY,
    UnmapViewOfFile,
};
use windows_sys::Win32::System::Threading::CreateEventW;

use super::{FileId, FileKind, Kind, Open};

/// Largest single `ReadFile`/`WriteFile` request; the length argument is a
/// `u32`, and callers already loop on short counts.
const MAX_TRANSFER: usize = 1 << 30;

/// FILETIME ticks (100 ns) between 1601-01-01 and the Unix epoch.
const UNIX_EPOCH_AS_FILETIME: i64 = 116_444_736_000_000_000;

fn open_once(spec: &Open, path: &Path, reparse: bool, creating: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options
        .read(spec.read)
        .write(spec.write)
        .append(spec.append);
    if creating {
        options
            .create(spec.create)
            .create_new(spec.create_new)
            .truncate(spec.truncate);
    }
    let mut flags = 0;
    if reparse {
        flags |= FILE_FLAG_OPEN_REPARSE_POINT;
    }
    if spec.directory {
        flags |= FILE_FLAG_BACKUP_SEMANTICS;
    }
    if spec.concurrent_reads {
        flags |= FILE_FLAG_OVERLAPPED;
    }
    if flags != 0 {
        options.custom_flags(flags);
    }
    if spec.hold_in_place {
        options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
    }
    options.open(path)
}

pub(super) fn open(spec: &Open, path: &Path) -> io::Result<File> {
    let file = open_once(spec, path, spec.no_follow, true)?;
    if !spec.no_follow {
        return Ok(file);
    }
    let metadata = file.metadata()?;
    if metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} is a symbolic link or junction", path.display()),
        ));
    }
    if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0 {
        return Ok(file);
    }
    // A reparse point that is not a link: reopen it normally so its filter
    // runs, and prove the path still names the object just inspected.
    let inspected = identify(&file)?.0;
    drop(file);
    let followed = open_once(spec, path, false, false)?;
    if identify(&followed)?.0 != inspected {
        return Err(io::Error::other(format!(
            "{} changed while it was being opened",
            path.display()
        )));
    }
    Ok(followed)
}

/// Volume and file identity plus the link count of an open handle.
fn identify(file: &File) -> io::Result<(FileId, u64)> {
    let handle = file.as_raw_handle() as HANDLE;
    // SAFETY: an all-zero BY_HANDLE_FILE_INFORMATION is a valid out value and
    // the handle is live for the call.
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { zeroed() };
    // SAFETY: `info` is writable storage of the exact type the call fills.
    if unsafe { GetFileInformationByHandle(handle, &mut info) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: as above; FILE_ID_INFO is plain data.
    let mut wide: FILE_ID_INFO = unsafe { zeroed() };
    // SAFETY: the buffer is exactly `size_of::<FILE_ID_INFO>()` writable bytes.
    let have_wide = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            std::ptr::from_mut(&mut wide).cast::<c_void>(),
            size_of::<FILE_ID_INFO>() as u32,
        )
    } != 0;
    let id = if have_wide {
        let bytes = wide.FileId.Identifier;
        let low = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"));
        let high = u64::from_le_bytes(bytes[8..].try_into().expect("8 bytes"));
        FileId {
            device: wide.VolumeSerialNumber,
            // NTFS ids are 64 bits zero-extended; only ReFS fills the rest.
            inode: if high == 0 {
                low
            } else {
                low ^ high.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            },
        }
    } else {
        FileId {
            device: u64::from(info.dwVolumeSerialNumber),
            inode: (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
        }
    };
    Ok((id, u64::from(info.nNumberOfLinks)))
}

/// The NTFS change time (the closest thing Windows has to `ctime`).
fn change_time(file: &File) -> Option<(i64, i64)> {
    let handle = file.as_raw_handle() as HANDLE;
    // SAFETY: FILE_BASIC_INFO is plain data and the buffer is exactly its size.
    let mut basic: FILE_BASIC_INFO = unsafe { zeroed() };
    let ok = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileBasicInfo,
            std::ptr::from_mut(&mut basic).cast::<c_void>(),
            size_of::<FILE_BASIC_INFO>() as u32,
        )
    };
    (ok != 0).then(|| filetime_parts(basic.ChangeTime))
}

fn filetime_parts(filetime: i64) -> (i64, i64) {
    let since_unix = filetime - UNIX_EPOCH_AS_FILETIME;
    (
        since_unix.div_euclid(10_000_000),
        since_unix.rem_euclid(10_000_000) * 100,
    )
}

fn system_time_parts(time: SystemTime) -> (i64, i64) {
    match time.duration_since(UNIX_EPOCH) {
        Ok(after) => (after.as_secs() as i64, i64::from(after.subsec_nanos())),
        Err(error) => {
            let before = error.duration();
            let nanos = i64::from(before.subsec_nanos());
            if nanos == 0 {
                (-(before.as_secs() as i64), 0)
            } else {
                (-(before.as_secs() as i64) - 1, 1_000_000_000 - nanos)
            }
        }
    }
}

/// A snapshot of one file's type, size, identity and times. Windows has no
/// ownership bits: `mode` is synthesized (read-only files 0o444, other files
/// 0o644, directories 0o755) and `uid` is always zero.
#[derive(Debug, Clone)]
pub struct Stat {
    kind: Kind,
    len: u64,
    id: FileId,
    nlink: u64,
    mode: u32,
    modified: SystemTime,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

impl Stat {
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn file_type(&self) -> FileKind {
        FileKind::of(self.kind)
    }

    pub fn is_file(&self) -> bool {
        self.kind == Kind::File
    }

    pub fn is_symlink(&self) -> bool {
        self.kind == Kind::Symlink
    }

    pub fn is_dir(&self) -> bool {
        self.kind == Kind::Directory
    }

    pub fn id(&self) -> FileId {
        self.id
    }

    pub fn dev(&self) -> u64 {
        self.id.device
    }

    pub fn ino(&self) -> u64 {
        self.id.inode
    }

    pub fn mode(&self) -> u32 {
        self.mode
    }

    pub fn uid(&self) -> u32 {
        0
    }

    /// Windows has no uid: the object is taken to be owner-controlled, since
    /// the runtime does not inspect ACLs (the install root is under the user's
    /// profile, whose inherited ACL excludes other standard users).
    pub fn is_owner_controlled(&self) -> bool {
        true
    }

    /// Windows has no group/other write bits; the ACL is not inspected.
    pub fn is_shared_writable(&self) -> bool {
        false
    }

    /// Windows has no group/other mode bits; access follows the inherited ACL,
    /// which the runtime does not inspect, so owner-only is taken as given.
    pub fn is_owner_only(&self) -> bool {
        true
    }

    pub fn nlink(&self) -> u64 {
        self.nlink
    }

    pub fn mtime(&self) -> i64 {
        self.mtime.0
    }

    pub fn mtime_nsec(&self) -> i64 {
        self.mtime.1
    }

    pub fn ctime(&self) -> i64 {
        self.ctime.0
    }

    pub fn ctime_nsec(&self) -> i64 {
        self.ctime.1
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        Ok(self.modified)
    }
}

fn stat_of_handle(file: &File) -> io::Result<Stat> {
    let metadata = file.metadata()?;
    let file_type = metadata.file_type();
    let kind = if file_type.is_symlink() {
        Kind::Symlink
    } else if file_type.is_dir() {
        Kind::Directory
    } else if file_type.is_file() {
        Kind::File
    } else {
        Kind::Other
    };
    let (id, nlink) = identify(file)?;
    let modified = metadata.modified()?;
    let mtime = system_time_parts(modified);
    let mode = match kind {
        Kind::Directory => 0o755,
        _ if metadata.file_attributes() & FILE_ATTRIBUTE_READONLY != 0 => 0o444,
        _ => 0o644,
    };
    Ok(Stat {
        kind,
        len: metadata.len(),
        id,
        nlink,
        mode,
        modified,
        mtime,
        ctime: change_time(file).unwrap_or(mtime),
    })
}

/// A private, read-only mapping of a whole admitted file. The file must stay
/// open (and unmodified) for as long as the mapping is read; while a view
/// exists Windows also refuses to truncate or delete the file.
pub struct ReadOnlyMap {
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    length: usize,
}

impl ReadOnlyMap {
    pub fn map(file: &File, length: usize) -> io::Result<Self> {
        if length == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot map an empty file",
            ));
        }
        // SAFETY: the handle is live; no security attributes, whole file,
        // unnamed mapping object.
        let mapping = unsafe {
            CreateFileMappingW(
                file.as_raw_handle() as HANDLE,
                null(),
                PAGE_READONLY,
                0,
                0,
                null(),
            )
        };
        if mapping.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mapping` is a live mapping object; offset zero, whole file.
        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };
        let failure = view.Value.is_null().then(io::Error::last_os_error);
        // The view keeps the mapping object alive; the handle is not needed.
        // SAFETY: `mapping` was created above and is closed exactly once.
        unsafe { CloseHandle(mapping) };
        match failure {
            Some(error) => Err(error),
            None => Ok(Self { view, length }),
        }
    }

    pub fn address(&self) -> *const u8 {
        self.view.Value.cast::<u8>().cast_const()
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

impl Drop for ReadOnlyMap {
    fn drop(&mut self) {
        // SAFETY: this object owns exactly this successful view.
        let _ = unsafe { UnmapViewOfFile(self.view) };
    }
}

/// Create a directory. A new Windows directory takes the ACL inherited from
/// its parent; it is owner-only wherever that parent is (a user profile is),
/// and the runtime does not rewrite ACLs.
pub fn create_private_directory(path: &Path) -> io::Result<()> {
    fs::create_dir(path)
}

/// Create a symbolic link at `link` pointing at `target`. Windows has separate
/// file and directory links, so the kind of the (possibly relative) target is
/// looked up; a dangling target is linked as a file. Creating any symbolic
/// link needs the symlink privilege (administrators have it; other accounts
/// need Developer Mode).
pub fn symlink<T: AsRef<Path>, L: AsRef<Path>>(target: T, link: L) -> io::Result<()> {
    let (target, link) = (target.as_ref(), link.as_ref());
    let resolved = if target.is_absolute() {
        target.to_path_buf()
    } else {
        link.parent().unwrap_or_else(|| Path::new(".")).join(target)
    };
    if resolved.is_dir() {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    }
}

/// Unix permission bits have no Windows equivalent; access follows the ACL the
/// file inherited, which the runtime does not rewrite.
pub fn set_mode(_path: &Path, _mode: u32) -> io::Result<()> {
    Ok(())
}

/// There is no mode to tighten: Windows access is governed by inherited ACLs.
pub fn restrict_to_owner(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// Whether an error means the process ran out of handles
/// (`ERROR_TOO_MANY_OPEN_FILES`), as opposed to a problem with the file itself.
pub fn is_descriptor_exhaustion(error: &io::Error) -> bool {
    error.raw_os_error() == Some(4)
}

/// Stat a path without following a final link: the object is opened for
/// attribute access only, which never conflicts with another process's share
/// mode.
pub fn lstat(path: &Path) -> io::Result<Stat> {
    let file = OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    stat_of_handle(&file)
}

/// Stat a path, following links.
pub fn stat(path: &Path) -> io::Result<Stat> {
    let file = OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    stat_of_handle(&file)
}

/// Stat an open handle: what was actually opened, whatever the path now says.
pub fn fstat(file: &File) -> io::Result<Stat> {
    stat_of_handle(file)
}

/// Resolve a path to its canonical absolute form, without the `\\?\` verbatim
/// prefix `fs::canonicalize` adds. Std re-adds the prefix itself when a path
/// is too long for the legacy limit, so the plain spelling is safe to keep,
/// compare and hand to other tools.
pub fn canonicalize<P: AsRef<Path>>(path: P) -> io::Result<PathBuf> {
    fs::canonicalize(path).map(strip_verbatim_prefix)
}

pub(super) fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path;
    };
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = text.strip_prefix(r"\\?\") {
        let bytes = rest.as_bytes();
        let drive = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
        if drive && (bytes.len() == 2 || bytes[2] == b'\\') {
            return PathBuf::from(rest);
        }
    }
    path
}

/// The Windows system directory (`C:\Windows\System32`), asked of the
/// kernel rather than read from an environment variable a caller controls.
pub fn system_directory() -> io::Result<PathBuf> {
    use std::os::windows::ffi::OsStringExt;

    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    let mut buffer = vec![0_u16; 260];
    loop {
        // SAFETY: the buffer is writable for its stated length.
        let written = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) } as usize;
        if written == 0 {
            return Err(io::Error::last_os_error());
        }
        if written < buffer.len() {
            buffer.truncate(written);
            return Ok(strip_verbatim_prefix(PathBuf::from(std::ffi::OsString::from_wide(&buffer))));
        }
        // The call reports the size it needed, terminator included.
        buffer.resize(written + 1, 0);
    }
}

/// Directory entries are journaled by NTFS and a directory handle cannot be
/// flushed (`FlushFileBuffers` needs write access to a volume-level object),
/// so there is nothing further to do once the file itself was flushed.
pub fn sync_directory(_directory: &File) -> io::Result<()> {
    Ok(())
}

fn wide(path: &Path) -> io::Result<Vec<u16>> {
    let mut units: Vec<u16> = path.as_os_str().encode_wide().collect();
    if units.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path contains a NUL character",
        ));
    }
    units.push(0);
    Ok(units)
}

/// Bytes available to this user on the volume holding `path` (a directory).
/// The caller-available figure honors per-user quotas, as `f_bavail` does.
pub fn available_space(_directory: &File, path: &Path) -> io::Result<u64> {
    let path = wide(path)?;
    let mut available = 0_u64;
    let mut total = 0_u64;
    let mut free = 0_u64;
    // SAFETY: `path` is NUL terminated and the three outputs are valid u64s.
    if unsafe { GetDiskFreeSpaceExW(path.as_ptr(), &mut available, &mut total, &mut free) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(available)
}

/// Rename without ever replacing a path that raced into place. Without
/// `MOVEFILE_REPLACE_EXISTING`, `MoveFileExW` fails when the destination
/// exists, on any filesystem, for files and directories alike.
pub fn rename_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
    let source = wide(source)?;
    let destination = wide(destination)?;
    // SAFETY: both strings are NUL terminated and outlive the call.
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), MOVEFILE_WRITE_THROUGH) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Make `destination` hold `part`'s content without ever replacing an existing
/// `destination`, leaving no `part` behind. One no-replace rename does both,
/// atomically, without needing hard-link support from the filesystem.
pub fn publish_without_replace(part: &Path, destination: &Path) -> io::Result<()> {
    rename_noreplace(part, destination)
}

/// The byte whose lock stands for "the file is locked": far beyond any data,
/// so the mandatory range lock never blocks anyone's reads or writes.
const LOCK_BYTE: u64 = 0x7fff_ffff_ffff_fff0;

fn lock_range() -> OVERLAPPED {
    // SAFETY: an all-zero OVERLAPPED is the documented initial state.
    let mut overlapped: OVERLAPPED = unsafe { zeroed() };
    // Assigning a plain-integer union field is safe; only reads are not.
    overlapped.Anonymous.Anonymous.Offset = LOCK_BYTE as u32;
    overlapped.Anonymous.Anonymous.OffsetHigh = (LOCK_BYTE >> 32) as u32;
    overlapped
}

/// Take an exclusive, non-blocking lock. `Ok(false)` means another handle
/// holds it; the lock lasts until [`unlock`] or the handle closes.
pub fn try_lock_exclusive(file: &File) -> io::Result<bool> {
    let mut overlapped = lock_range();
    // SAFETY: the handle is live and `overlapped` outlives the call.
    let locked = unsafe {
        LockFileEx(
            file.as_raw_handle() as HANDLE,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            &mut overlapped,
        )
    };
    if locked != 0 {
        return Ok(true);
    }
    // SAFETY: reads the calling thread's last-error value.
    let code = unsafe { GetLastError() };
    if code == ERROR_LOCK_VIOLATION {
        Ok(false)
    } else {
        Err(io::Error::from_raw_os_error(code as i32))
    }
}

/// Release a lock taken by [`try_lock_exclusive`].
pub fn unlock(file: &File) {
    let mut overlapped = lock_range();
    // SAFETY: the handle is live and `overlapped` outlives the call.
    let _ = unsafe { UnlockFileEx(file.as_raw_handle() as HANDLE, 0, 1, 0, &mut overlapped) };
}

/// A manual-reset event owned by one thread, reused for every positional read
/// that thread performs.
struct ThreadEvent(HANDLE);

impl ThreadEvent {
    fn create() -> Self {
        // SAFETY: no security attributes, manual reset, initially clear, unnamed.
        Self(unsafe { CreateEventW(null(), 1, 0, null()) })
    }
}

impl Drop for ThreadEvent {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the handle was created by this value and is closed once.
            unsafe { CloseHandle(self.0) };
        }
    }
}

thread_local! {
    static READ_EVENT: ThreadEvent = ThreadEvent::create();
}

fn offset_overlapped(offset: u64, event: HANDLE) -> OVERLAPPED {
    // SAFETY: an all-zero OVERLAPPED is the documented initial state.
    let mut overlapped: OVERLAPPED = unsafe { zeroed() };
    // Assigning a plain-integer union field is safe; only reads are not.
    overlapped.Anonymous.Anonymous.Offset = offset as u32;
    overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
    overlapped.hEvent = event;
    overlapped
}

/// Run one offset transfer to completion on the calling thread. `submit`
/// issues the `ReadFile`/`WriteFile` and reports whether it succeeded outright.
fn transfer(
    file: &File,
    offset: u64,
    submit: impl FnOnce(HANDLE, *mut OVERLAPPED) -> bool,
) -> io::Result<usize> {
    let event = READ_EVENT.with(|event| event.0);
    if event.is_null() {
        return Err(io::Error::other("cannot create the per-thread I/O event"));
    }
    let handle = file.as_raw_handle() as HANDLE;
    let mut overlapped = offset_overlapped(offset, event);
    if !submit(handle, &mut overlapped) {
        // SAFETY: reads the calling thread's last-error value.
        let code = unsafe { GetLastError() };
        if code == ERROR_HANDLE_EOF {
            return Ok(0);
        }
        if code != ERROR_IO_PENDING {
            return Err(io::Error::from_raw_os_error(code as i32));
        }
    }
    let mut transferred = 0_u32;
    // SAFETY: the handle is live and `overlapped` is the structure the request
    // used; waiting keeps both alive until the kernel is done with them.
    if unsafe { GetOverlappedResult(handle, &overlapped, &mut transferred, 1) } == 0 {
        // SAFETY: reads the calling thread's last-error value.
        let code = unsafe { GetLastError() };
        if code == ERROR_HANDLE_EOF {
            return Ok(0);
        }
        return Err(io::Error::from_raw_os_error(code as i32));
    }
    Ok(transferred as usize)
}

/// Positional I/O with the Unix `FileExt` shape. Reads from several threads on
/// one handle overlap when it was opened with `Open::concurrent_reads`; on an
/// ordinary handle they are correct but take turns.
pub trait FileExt {
    fn read_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<usize>;
    fn write_at(&self, buffer: &[u8], offset: u64) -> io::Result<usize>;

    fn read_exact_at(&self, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
        while !buffer.is_empty() {
            match self.read_at(buffer, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                Ok(count) => {
                    buffer = &mut buffer[count..];
                    offset += count as u64;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn write_all_at(&self, mut buffer: &[u8], mut offset: u64) -> io::Result<()> {
        while !buffer.is_empty() {
            match self.write_at(buffer, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    ));
                }
                Ok(count) => {
                    buffer = &buffer[count..];
                    offset += count as u64;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

impl FileExt for File {
    fn read_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let length = buffer.len().min(MAX_TRANSFER) as u32;
        let target = buffer.as_mut_ptr();
        transfer(self, offset, |handle, overlapped| {
            // SAFETY: `target` addresses `length` writable bytes that outlive
            // the call, which `transfer` waits out before returning.
            unsafe { ReadFile(handle, target, length, std::ptr::null_mut(), overlapped) != 0 }
        })
    }

    fn write_at(&self, buffer: &[u8], offset: u64) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let length = buffer.len().min(MAX_TRANSFER) as u32;
        let source = buffer.as_ptr();
        transfer(self, offset, |handle, overlapped| {
            // SAFETY: `source` addresses `length` readable bytes that outlive
            // the call, which `transfer` waits out before returning.
            unsafe { WriteFile(handle, source, length, std::ptr::null_mut(), overlapped) != 0 }
        })
    }
}
