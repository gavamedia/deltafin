//! Byte views of operating-system strings.
//!
//! Several places treat a path or variable name as bytes: loader metadata
//! stores them that way, and identities and receipts hex-encode them. On Unix
//! an `OsStr` *is* bytes. On Windows it is WTF-8, which `as_encoded_bytes`
//! exposes; going the other way, bytes that came from a Windows binary's
//! metadata are ASCII/UTF-8 names, so a lossy decode is exact for every real
//! input and merely safe for a hostile one.

use std::ffi::{OsStr, OsString};

/// The bytes of an operating-system string, as the platform encodes it.
pub fn os_str_bytes(value: &OsStr) -> &[u8] {
    value.as_encoded_bytes()
}

/// Build an operating-system string from bytes read from binary metadata.
pub fn os_string_from_bytes(bytes: Vec<u8>) -> OsString {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        OsString::from_vec(bytes)
    }
    #[cfg(windows)]
    {
        OsString::from(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_round_trip_for_ordinary_names() {
        let name = OsString::from("libtorch_cpu.dll");
        assert_eq!(os_str_bytes(&name), b"libtorch_cpu.dll");
        assert_eq!(os_string_from_bytes(b"libtorch_cpu.dll".to_vec()), name);
        assert_eq!(os_string_from_bytes(Vec::new()), OsString::new());
    }

    #[cfg(unix)]
    #[test]
    fn unix_names_keep_arbitrary_bytes() {
        let raw = vec![b'a', 0xff, b'/', 0xfe];
        assert_eq!(os_str_bytes(&os_string_from_bytes(raw.clone())), raw);
    }

    #[cfg(windows)]
    #[test]
    fn windows_decode_never_panics_on_hostile_bytes() {
        let name = os_string_from_bytes(vec![b'a', 0xff, 0xfe, b'.', b'd', b'l', b'l']);
        assert!(name.to_string_lossy().ends_with(".dll"));
    }
}
