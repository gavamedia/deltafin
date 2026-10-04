//! The portable file-system surface, exercised against the real OS.
//!
//! Every test here runs unchanged on Unix and Windows; that is the point of
//! the module. A test that needs a symbolic link skips itself where the host
//! refuses to create one (an unprivileged Windows user).

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::*;

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "deltafin-sys-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn file_with(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Create a symlink, or report that this host will not let the test do so.
fn try_symlink(target: &Path, link: &Path, directory: bool) -> bool {
    #[cfg(unix)]
    let result = {
        let _ = directory;
        std::os::unix::fs::symlink(target, link)
    };
    #[cfg(windows)]
    let result = if directory {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    };
    match result {
        Ok(()) => true,
        Err(error) => {
            eprintln!("skipping: cannot create a symlink here: {error}");
            false
        }
    }
}

#[test]
fn no_follow_opens_regular_files_and_refuses_links() {
    let scratch = Scratch::new("nofollow");
    let real = scratch.file_with("real", b"payload");
    let link = scratch.path("link");
    let mut file = Open::new().read(true).no_follow().open(&real).unwrap();
    assert_eq!(fstat(&file).unwrap().len(), 7);
    let mut text = String::new();
    io::Read::read_to_string(&mut file, &mut text).unwrap();
    assert_eq!(text, "payload");
    if try_symlink(&real, &link, false) {
        assert!(
            Open::new().read(true).no_follow().open(&link).is_err(),
            "a final symlink must be refused"
        );
        // Following is still possible when asked for.
        assert!(Open::new().read(true).open(&link).is_ok());
    }
}

#[test]
fn no_follow_directory_open_refuses_linked_directories() {
    let scratch = Scratch::new("nofollow-dir");
    let real = scratch.path("real");
    fs::create_dir(&real).unwrap();
    let link = scratch.path("link");
    assert!(
        Open::new()
            .read(true)
            .directory()
            .no_follow()
            .open(&real)
            .is_ok()
    );
    if try_symlink(&real, &link, true) {
        assert!(
            Open::new()
                .read(true)
                .directory()
                .no_follow()
                .open(&link)
                .is_err()
        );
    }
}

#[test]
fn create_new_never_overwrites_an_existing_file() {
    let scratch = Scratch::new("create-new");
    let path = scratch.path("fresh");
    let mut file = Open::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .no_follow()
        .open(&path)
        .unwrap();
    file.write_all(b"first").unwrap();
    drop(file);
    let error = Open::new()
        .write(true)
        .create_new(true)
        .no_follow()
        .open(&path)
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    assert_eq!(fs::read(&path).unwrap(), b"first");
    #[cfg(unix)]
    assert_eq!(lstat(&path).unwrap().mode() & 0o777, 0o600);
}

#[test]
fn stat_identity_tells_files_apart_and_matches_hard_links() {
    let scratch = Scratch::new("identity");
    let first = scratch.file_with("first", b"one");
    let second = scratch.file_with("second", b"two");
    let linked = scratch.path("linked");
    fs::hard_link(&first, &linked).unwrap();

    let first_id = lstat(&first).unwrap().id();
    assert_ne!(first_id, lstat(&second).unwrap().id());
    assert_eq!(first_id, lstat(&linked).unwrap().id());
    assert_eq!(first_id, stat(&first).unwrap().id());
    let handle = Open::new().read(true).open(&first).unwrap();
    assert_eq!(first_id, fstat(&handle).unwrap().id());
    assert_eq!(first_id.device, fstat(&handle).unwrap().dev());
    assert!(lstat(&first).unwrap().nlink() >= 2);
    assert!(lstat(&first).unwrap().file_type().is_file());
    assert!(lstat(&scratch.0).unwrap().file_type().is_dir());
    assert!(lstat(&scratch.path("absent")).is_err());
}

#[test]
fn stat_classifies_links_without_following_them() {
    let scratch = Scratch::new("kinds");
    let real = scratch.file_with("real", b"x");
    let link = scratch.path("link");
    if try_symlink(&real, &link, false) {
        let seen = lstat(&link).unwrap();
        assert!(seen.file_type().is_symlink());
        assert!(!seen.is_file());
        assert!(stat(&link).unwrap().file_type().is_file());
        assert_eq!(stat(&link).unwrap().id(), lstat(&real).unwrap().id());
    }
}

#[test]
fn stat_reports_length_and_plausible_times() {
    let scratch = Scratch::new("times");
    let path = scratch.file_with("data", &[7_u8; 1234]);
    let seen = lstat(&path).unwrap();
    assert_eq!(seen.len(), 1234);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    for stamp in [seen.mtime(), seen.ctime()] {
        assert!((now - 600..=now + 600).contains(&stamp), "{stamp} vs {now}");
    }
    assert!((0..1_000_000_000).contains(&seen.mtime_nsec()));
    assert!((0..1_000_000_000).contains(&seen.ctime_nsec()));
    assert!(seen.modified().is_ok());
}

#[test]
fn positional_reads_are_correct_from_many_threads_on_one_handle() {
    let scratch = Scratch::new("pread");
    let pattern: Vec<u8> = (0..(4 << 20)).map(|index: u32| (index.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
    let path = scratch.file_with("data", &pattern);
    let handle = Arc::new(
        Open::new()
            .read(true)
            .no_follow()
            .concurrent_reads()
            .open(&path)
            .unwrap(),
    );
    let pattern = Arc::new(pattern);
    thread::scope(|scope| {
        for worker in 0..8_u64 {
            let handle = Arc::clone(&handle);
            let pattern = Arc::clone(&pattern);
            scope.spawn(move || {
                let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ worker;
                let mut buffer = vec![0_u8; 70_000];
                for _ in 0..64 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let offset = (state % (pattern.len() as u64 - 70_000)) as usize;
                    handle.read_exact_at(&mut buffer, offset as u64).unwrap();
                    assert_eq!(buffer, pattern[offset..offset + 70_000], "offset {offset}");
                }
            });
        }
    });
    // Past the end reads nothing; a read straddling the end reads the rest.
    let mut buffer = [0_u8; 16];
    assert_eq!(handle.read_at(&mut buffer, pattern.len() as u64).unwrap(), 0);
    assert_eq!(handle.read_at(&mut buffer, pattern.len() as u64 + 99).unwrap(), 0);
    let tail = pattern.len() as u64 - 5;
    assert_eq!(handle.read_at(&mut buffer, tail).unwrap(), 5);
    assert_eq!(buffer[..5], pattern[pattern.len() - 5..]);
    assert_eq!(handle.read_at(&mut [], 0).unwrap(), 0);
    let error = handle.read_exact_at(&mut buffer, tail).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn positional_reads_also_work_on_an_ordinary_handle() {
    let scratch = Scratch::new("pread-plain");
    let path = scratch.file_with("data", b"0123456789");
    let handle = Open::new().read(true).open(&path).unwrap();
    let mut buffer = [0_u8; 4];
    handle.read_exact_at(&mut buffer, 3).unwrap();
    assert_eq!(&buffer, b"3456");
    handle.read_exact_at(&mut buffer, 0).unwrap();
    assert_eq!(&buffer, b"0123");
}

#[test]
fn positional_writes_extend_and_overwrite() {
    let scratch = Scratch::new("pwrite");
    let path = scratch.path("data");
    let handle = Open::new().read(true).write(true).create(true).open(&path).unwrap();
    handle.write_all_at(b"abcdef", 0).unwrap();
    handle.write_all_at(b"XY", 2).unwrap();
    handle.write_all_at(b"Z", 10).unwrap();
    drop(handle);
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..6], b"abXYef");
    assert_eq!(bytes.len(), 11);
    assert_eq!(bytes[10], b'Z');
}

#[test]
fn rename_noreplace_never_replaces_a_racer() {
    let scratch = Scratch::new("rename");
    let source = scratch.file_with("source", b"new");
    let destination = scratch.path("destination");
    rename_noreplace(&source, &destination).unwrap();
    assert!(!source.exists());
    assert_eq!(fs::read(&destination).unwrap(), b"new");

    let second = scratch.file_with("second", b"newer");
    let error = rename_noreplace(&second, &destination).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::AlreadyExists, "{error}");
    assert_eq!(fs::read(&destination).unwrap(), b"new", "racer was replaced");
    assert_eq!(fs::read(&second).unwrap(), b"newer", "source was consumed");

    let directory = scratch.path("directory");
    fs::create_dir(&directory).unwrap();
    let renamed = scratch.path("renamed");
    rename_noreplace(&directory, &renamed).unwrap();
    assert!(renamed.is_dir() && !directory.exists());
    fs::create_dir(&directory).unwrap();
    assert!(rename_noreplace(&directory, &renamed).is_err());
}

#[test]
fn publish_without_replace_leaves_no_partial_and_no_overwrite() {
    let scratch = Scratch::new("publish");
    let part = scratch.file_with("weights.part", b"verified");
    let published = scratch.path("weights");
    publish_without_replace(&part, &published).unwrap();
    assert!(!part.exists());
    assert_eq!(fs::read(&published).unwrap(), b"verified");

    let next = scratch.file_with("next.part", b"other");
    assert!(publish_without_replace(&next, &published).is_err());
    assert_eq!(fs::read(&published).unwrap(), b"verified");
    assert!(next.exists(), "a refused publish must not consume its source");
}

#[test]
fn an_exclusive_lock_excludes_a_second_handle_until_released() {
    let scratch = Scratch::new("lock");
    let path = scratch.path("lock");
    let open = || {
        Open::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .no_follow()
            .open(&path)
            .unwrap()
    };
    let first = open();
    let second = open();
    assert!(try_lock_exclusive(&first).unwrap());
    assert!(!try_lock_exclusive(&second).unwrap(), "second handle must not lock");
    unlock(&first);
    assert!(try_lock_exclusive(&second).unwrap());
    assert!(!try_lock_exclusive(&first).unwrap());
    // Closing the holder releases it.
    drop(second);
    assert!(try_lock_exclusive(&first).unwrap());
    // The lock never gets in the way of the file's own bytes.
    let mut peer = open();
    peer.write_all(b"still writable").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"still writable");
}

#[test]
fn available_space_is_positive_for_a_real_directory() {
    let scratch = Scratch::new("space");
    let directory = Open::new().read(true).directory().no_follow().open(&scratch.0).unwrap();
    assert!(available_space(&directory, &scratch.0).unwrap() > 0);
    sync_directory(&directory).unwrap();
}

#[test]
fn canonicalize_returns_a_plain_absolute_path() {
    let scratch = Scratch::new("canonical");
    let nested = scratch.path("a");
    fs::create_dir(&nested).unwrap();
    let resolved = canonicalize(&nested.join("..").join("a")).unwrap();
    assert!(resolved.is_absolute());
    assert!(resolved.ends_with("a"));
    #[cfg(windows)]
    assert!(
        !resolved.to_string_lossy().starts_with(r"\\?\"),
        "{}",
        resolved.display()
    );
    assert_eq!(
        canonicalize(&resolved).unwrap(),
        resolved,
        "canonical form is a fixed point"
    );
}

#[cfg(windows)]
#[test]
fn verbatim_prefixes_are_stripped_only_where_it_is_safe() {
    use super::windows::strip_verbatim_prefix as strip;
    let cases = [
        (r"\\?\C:\models\k3", r"C:\models\k3"),
        (r"\\?\c:", "c:"),
        (r"\\?\UNC\server\share\k3", r"\\server\share\k3"),
        (r"\\?\Volume{1234}\k3", r"\\?\Volume{1234}\k3"),
        (r"\\?\GLOBALROOT\Device\x", r"\\?\GLOBALROOT\Device\x"),
        (r"C:\plain", r"C:\plain"),
    ];
    for (input, expected) in cases {
        assert_eq!(strip(PathBuf::from(input)), PathBuf::from(expected), "{input}");
    }
}

#[cfg(windows)]
#[test]
fn the_system_directory_is_a_real_directory_holding_the_core_dlls() {
    let directory = system_directory().unwrap();
    assert!(directory.is_absolute(), "{}", directory.display());
    assert!(directory.is_dir(), "{}", directory.display());
    for dll in ["kernel32.dll", "ntdll.dll"] {
        assert!(directory.join(dll).is_file(), "{dll} missing from {}", directory.display());
    }
    assert!(
        !directory.to_string_lossy().starts_with(r"\\?\"),
        "{}",
        directory.display()
    );
}

#[cfg(unix)]
#[test]
fn unix_has_no_single_system_directory_to_report() {
    assert_eq!(
        system_directory().unwrap_err().kind(),
        std::io::ErrorKind::Unsupported
    );
}
