"""Positional file reads that work on POSIX and Windows alike.

Deltafin's readers deliberately use positional reads rather than seek-then-read:
one cached descriptor is shared by a whole worker pool, and a positional read
does not touch a file pointer, so those threads never serialise against each
other.  CPython exposes that as ``os.preadv`` on POSIX and not at all on
Windows.

Windows does have the primitive: ``ReadFile`` takes the offset in an
``OVERLAPPED`` structure.  On a handle opened ``FILE_FLAG_OVERLAPPED`` the call
neither uses nor advances the shared file pointer, and several reads may be in
flight at once, which is exactly the property the POSIX path relies on.  Each
read here still blocks its own caller, so the reader keeps its simple
synchronous shape; only the serialisation disappears.

This module is intentionally narrow, like win_compat.h: it implements the one
operation the readers need, on top of one file object they can open and close.
"""

from __future__ import annotations

import os
import sys
import threading

IS_WINDOWS = os.name == "nt"

# ReadFile counts bytes in a DWORD.  Deltafin never asks for anywhere near this
# much in one call; the read loop below makes the bound irrelevant regardless.
MAX_SINGLE_READ = 1 << 30


class PositionalFile:
    """A read-only file supporting concurrent positional reads.

    Instances are safe to share across threads.  ``read_into`` fills the whole
    destination or raises; a caller that wants to tolerate a short file must
    size the destination itself.
    """

    __slots__ = ("_path",)

    def __init__(self, path: str):
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def fileno(self) -> int | None:
        """The POSIX descriptor, or None when there is no such thing.

        Callers use this only to apply platform file hints, all of which are
        already guarded by a platform check.
        """
        raise NotImplementedError

    def read_into(self, destination, offset: int) -> int:
        raise NotImplementedError

    def read(self, count: int, offset: int) -> bytes:
        """Read exactly ``count`` bytes, allocating the buffer."""
        buffer = bytearray(count)
        self.read_into(memoryview(buffer), offset)
        return bytes(buffer)

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "PositionalFile":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


class _PosixPositionalFile(PositionalFile):
    __slots__ = ("_fd",)

    def __init__(self, path: str):
        super().__init__(path)
        self._fd = os.open(path, os.O_RDONLY)

    def fileno(self) -> int | None:
        return self._fd

    def read_into(self, destination, offset: int) -> int:
        view = memoryview(destination)
        total = len(view)
        got = 0
        while got < total:
            read = os.preadv(self._fd, [view[got:]], offset + got)
            if read <= 0:
                raise OSError(
                    f"short read: {got}/{total} @{offset} from {self._path}"
                )
            got += read
        return got

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _OPEN_EXISTING = 3
    _FILE_FLAG_OVERLAPPED = 0x40000000
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _ERROR_IO_PENDING = 997
    _ERROR_HANDLE_EOF = 38

    class _Overlapped(ctypes.Structure):
        # Offset and OffsetHigh occupy the union the documentation describes as
        # a pointer; supplying them is how a read is made positional.
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    _kernel32.CreateFileW.restype = wintypes.HANDLE

    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_Overlapped),
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL

    _kernel32.GetOverlappedResult.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_Overlapped),
        ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
    ]
    _kernel32.GetOverlappedResult.restype = wintypes.BOOL

    _kernel32.CreateEventW.argtypes = [
        ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR,
    ]
    _kernel32.CreateEventW.restype = wintypes.HANDLE

    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    # Every in-flight read needs its own event, and a thread performs one read
    # at a time, so one event per thread is both necessary and sufficient.
    # Worker pools here are long-lived, so these are created once per worker.
    _events = threading.local()

    def _thread_event():
        event = getattr(_events, "handle", None)
        if event is None:
            # Manual-reset: ReadFile clears it when the operation starts.
            event = _kernel32.CreateEventW(None, True, False, None)
            if not event:
                raise ctypes.WinError(ctypes.get_last_error())
            _events.handle = event
        return event

    def _windows_error(operation: str, path: str, code: int) -> OSError:
        error = ctypes.WinError(code)
        return OSError(f"{operation} failed for {path}: {error}")

    class _WindowsPositionalFile(PositionalFile):
        __slots__ = ("_handle",)

        def __init__(self, path: str):
            super().__init__(path)
            handle = _kernel32.CreateFileW(
                os.fspath(path),
                _GENERIC_READ,
                _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
                None,
                _OPEN_EXISTING,
                _FILE_FLAG_OVERLAPPED,
                None,
            )
            if handle == _INVALID_HANDLE_VALUE or not handle:
                code = ctypes.get_last_error()
                # Report the errors callers already branch on as themselves.
                raise ctypes.WinError(code, f"cannot open {path}")
            self._handle = handle

        def fileno(self) -> int | None:
            # There is no descriptor: the handle was never registered with the
            # CRT, deliberately, because FILE_FLAG_OVERLAPPED has no fd analogue.
            return None

        def _read_once(self, buffer, count: int, offset: int) -> int:
            overlapped = _Overlapped()
            overlapped.Offset = offset & 0xFFFFFFFF
            overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
            overlapped.hEvent = _thread_event()
            transferred = wintypes.DWORD(0)
            ok = _kernel32.ReadFile(
                self._handle, buffer, count,
                ctypes.byref(transferred), ctypes.byref(overlapped),
            )
            if not ok:
                code = ctypes.get_last_error()
                if code == _ERROR_HANDLE_EOF:
                    return 0
                if code != _ERROR_IO_PENDING:
                    raise _windows_error("ReadFile", self._path, code)
                completed = _kernel32.GetOverlappedResult(
                    self._handle, ctypes.byref(overlapped),
                    ctypes.byref(transferred), True,
                )
                if not completed:
                    code = ctypes.get_last_error()
                    if code == _ERROR_HANDLE_EOF:
                        return 0
                    raise _windows_error(
                        "GetOverlappedResult", self._path, code
                    )
            return transferred.value

        def read_into(self, destination, offset: int) -> int:
            view = memoryview(destination)
            if view.readonly:
                raise ValueError("read_into needs a writable destination")
            total = view.nbytes
            if total == 0:
                return 0
            # One ctypes view over the destination; byref(buffer, got) then
            # advances within it without re-exporting the buffer each pass.
            buffer = (ctypes.c_char * total).from_buffer(view.cast("B"))
            try:
                got = 0
                while got < total:
                    count = min(total - got, MAX_SINGLE_READ)
                    read = self._read_once(
                        ctypes.byref(buffer, got), count, offset + got
                    )
                    if read <= 0:
                        raise OSError(
                            f"short read: {got}/{total} @{offset} "
                            f"from {self._path}"
                        )
                    got += read
                return got
            finally:
                # Release the buffer export before the caller's view goes away.
                del buffer

        def close(self) -> None:
            if self._handle is not None:
                _kernel32.CloseHandle(self._handle)
                self._handle = None


def open_positional(path: str) -> PositionalFile:
    """Open a file for concurrent positional reads."""
    if IS_WINDOWS:
        return _WindowsPositionalFile(path)
    return _PosixPositionalFile(path)


def supported() -> bool:
    """Whether this interpreter can perform positional reads at all."""
    return IS_WINDOWS or hasattr(os, "preadv")
