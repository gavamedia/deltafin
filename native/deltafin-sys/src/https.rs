//! One bounded, synchronous HTTPS GET over the operating system's own
//! WinHTTP/Schannel stack.
//!
//! Windows ships no libcurl, and bundling one would put an unaudited native
//! library on the path of every pinned download. WinHTTP is part of the OS: it
//! validates certificates against the system trust store, follows the system
//! and corporate proxy configuration, and needs no DLL beside the executable.
//!
//! The contract mirrors what the libcurl transport enforces elsewhere:
//! * HTTPS only, with certificate and host validation always on (the one
//!   exception is `http://` to a loopback address, for the tests of this very
//!   module; production callers reject `http` before they get here);
//! * redirects are never followed: the caller sees the 3xx and decides;
//! * identity transfer: no content decoding happens in this layer;
//! * the caller bounds everything it receives. Both callbacks can abort the
//!   transfer, and a stalled connection or an overlong transfer fails rather
//!   than hanging.

use std::ffi::c_void;
use std::io;
use std::ptr::{null, null_mut};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, GetLastError};
use windows_sys::Win32::Networking::WinHttp::{
    WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_QUERY_STATUS_CODE, WinHttpAddRequestHeaders,
    WinHttpCloseHandle, WinHttpConnect, WinHttpOpen, WinHttpOpenRequest, WinHttpQueryDataAvailable,
    WinHttpQueryHeaders, WinHttpReadData, WinHttpReceiveResponse, WinHttpSendRequest, WinHttpSetOption,
    WinHttpSetTimeouts, WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_ADDREQ_FLAG_ADD,
    WINHTTP_FLAG_SECURE, WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2, WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3,
    WINHTTP_OPTION_REDIRECT_POLICY, WINHTTP_OPTION_REDIRECT_POLICY_NEVER, WINHTTP_OPTION_SECURE_PROTOCOLS,
};

/// How long name resolution and the send phase may take; connection setup and
/// stalls use the request's own timeouts.
const PHASE_TIMEOUT_MS: i32 = 30_000;
/// Body read buffer: large enough to keep a fast link busy, small enough that
/// the abort and total-timeout checks run often.
const BODY_CHUNK: usize = 256 << 10;

/// One GET request.
#[derive(Debug, Clone)]
pub struct Get<'a> {
    pub url: &'a str,
    pub user_agent: &'a str,
    /// Extra request header lines, each `Name: value` without a line ending.
    pub headers: &'a [&'a str],
    pub connect_timeout: Duration,
    /// The longest the connection may stay silent before the transfer fails.
    pub stall_timeout: Duration,
    /// A wall-clock cap on the whole transfer, if the caller wants one.
    pub total_timeout: Option<Duration>,
}

/// The status line and headers of a response, as the server sent them.
#[derive(Debug, Clone)]
pub struct Head {
    pub status: u16,
    /// Status line, header lines and the terminating blank line, CRLF-ended.
    pub raw: Vec<u8>,
}

/// How a transfer that did not fail ended.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Outcome {
    /// The whole response body was delivered.
    Complete,
    /// A callback asked to stop; the caller holds the reason.
    Aborted,
}

/// A WinHTTP handle that closes itself.
struct Handle(*mut c_void);

impl Handle {
    fn check(pointer: *mut c_void, what: &str) -> io::Result<Self> {
        if pointer.is_null() {
            Err(winhttp_error(what))
        } else {
            Ok(Self(pointer))
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: this value owns the handle and closes it exactly once.
        unsafe { WinHttpCloseHandle(self.0) };
    }
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A WinHTTP failure with a readable message: the 12xxx codes are not in the
/// system message table, so `io::Error`'s own formatting would be useless.
fn winhttp_error(operation: &str) -> io::Error {
    // SAFETY: reads the calling thread's last-error value.
    let code = unsafe { GetLastError() };
    let (kind, meaning) = match code {
        12002 => (io::ErrorKind::TimedOut, "the request timed out"),
        12007 => (io::ErrorKind::NotFound, "the host name could not be resolved"),
        12029 => (io::ErrorKind::ConnectionRefused, "cannot connect to the server"),
        12030 => (io::ErrorKind::ConnectionAborted, "the connection with the server was terminated"),
        12031 => (io::ErrorKind::ConnectionReset, "the connection was reset"),
        12044 | 12045 | 12037 | 12038 | 12157 | 12175 => (
            io::ErrorKind::InvalidData,
            "TLS negotiation or certificate validation failed",
        ),
        12152 => (io::ErrorKind::InvalidData, "the server sent an invalid response"),
        _ => (io::ErrorKind::Other, "WinHTTP reported an error"),
    };
    io::Error::new(kind, format!("{operation}: {meaning} (WinHTTP error {code})"))
}

/// Split `https://host[:port]/path?query` into `(secure, host, port, path)`,
/// where `path` is everything from the first `/` or `?` (possibly empty).
fn split_url(url: &str) -> io::Result<(bool, &str, u16, &str)> {
    let invalid = |why: &str| io::Error::new(io::ErrorKind::InvalidInput, format!("{why}: {url}"));
    let (secure, rest) = if let Some(rest) = url.strip_prefix("https://") {
        (true, rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        (false, rest)
    } else {
        return Err(invalid("not an absolute http(s) URL"));
    };
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(end);
    if authority.contains('@') || tail.contains('#') {
        return Err(invalid("unsupported URL userinfo or fragment"));
    }
    let default_port = if secure { 443 } else { 80 };
    let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        // An IPv6 literal: `[::1]` or `[::1]:8443`.
        let (host, after) = bracketed
            .split_once(']')
            .ok_or_else(|| invalid("unterminated IPv6 literal"))?;
        match after {
            "" => (host, default_port),
            _ => match after.strip_prefix(':') {
                Some(port) => (host, port.parse::<u16>().map_err(|_| invalid("invalid port"))?),
                None => return Err(invalid("invalid URL authority")),
            },
        }
    } else {
        match authority.split_once(':') {
            None => (authority, default_port),
            Some((host, port)) if !port.contains(':') => {
                (host, port.parse::<u16>().map_err(|_| invalid("invalid port"))?)
            }
            Some(_) => return Err(invalid("invalid URL authority")),
        }
    };
    if host.is_empty() {
        return Err(invalid("empty host"));
    }
    Ok((secure, host, port, tail))
}

fn is_loopback(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

/// Perform one GET. `on_head` sees the status and headers before any body;
/// `on_body` sees each chunk as it arrives. Either returns `false` to stop
/// the transfer, which ends it with [`Outcome::Aborted`].
pub fn get(
    request: &Get<'_>,
    on_head: &mut dyn FnMut(&Head) -> bool,
    on_body: &mut dyn FnMut(&[u8]) -> bool,
) -> io::Result<Outcome> {
    let (secure, host, port, tail) = split_url(request.url)?;
    if !secure && !is_loopback(host) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("refusing to fetch a non-loopback URL over plain HTTP: {}", request.url),
        ));
    }
    let started = Instant::now();

    // SAFETY: every pointer argument is either null or a live NUL-terminated
    // UTF-16 string for the duration of the call.
    let session = Handle::check(
        unsafe {
            WinHttpOpen(
                wide(request.user_agent).as_ptr(),
                WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                null(),
                null(),
                0,
            )
        },
        "open a WinHTTP session",
    )?;
    if secure {
        let both = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3;
        let only_12 = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
        // Older Windows 10 builds reject the TLS 1.3 flag; fall back to 1.2.
        for protocols in [both, only_12] {
            // SAFETY: the option value is a live u32 of the size passed.
            let applied = unsafe {
                WinHttpSetOption(
                    session.0,
                    WINHTTP_OPTION_SECURE_PROTOCOLS,
                    std::ptr::from_ref(&protocols).cast::<c_void>(),
                    std::mem::size_of::<u32>() as u32,
                )
            };
            if applied != 0 {
                break;
            }
            if protocols == only_12 {
                return Err(winhttp_error("require TLS 1.2 or newer"));
            }
        }
    }
    let millis = |duration: Duration| i32::try_from(duration.as_millis()).unwrap_or(i32::MAX);
    // SAFETY: plain integers for a live session handle.
    if unsafe {
        WinHttpSetTimeouts(
            session.0,
            PHASE_TIMEOUT_MS,
            millis(request.connect_timeout),
            PHASE_TIMEOUT_MS,
            millis(request.stall_timeout),
        )
    } == 0
    {
        return Err(winhttp_error("set WinHTTP timeouts"));
    }

    // SAFETY: as above.
    let connection = Handle::check(
        unsafe { WinHttpConnect(session.0, wide(host).as_ptr(), port, 0) },
        "connect to the server",
    )?;
    let path = if tail.is_empty() || tail.starts_with('?') {
        format!("/{tail}")
    } else {
        tail.to_owned()
    };
    // SAFETY: as above; no referrer, no accept types, default HTTP version.
    let transfer = Handle::check(
        unsafe {
            WinHttpOpenRequest(
                connection.0,
                wide("GET").as_ptr(),
                wide(&path).as_ptr(),
                null(),
                null(),
                null(),
                if secure { WINHTTP_FLAG_SECURE } else { 0 },
            )
        },
        "open the request",
    )?;
    let never_redirect = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
    // SAFETY: the option value is a live u32 of the size passed.
    if unsafe {
        WinHttpSetOption(
            transfer.0,
            WINHTTP_OPTION_REDIRECT_POLICY,
            std::ptr::from_ref(&never_redirect).cast::<c_void>(),
            std::mem::size_of::<u32>() as u32,
        )
    } == 0
    {
        return Err(winhttp_error("disable automatic redirects"));
    }
    if !request.headers.is_empty() {
        let mut block = request.headers.join("\r\n");
        block.push_str("\r\n");
        let block = wide(&block);
        // SAFETY: a NUL-terminated header block; the length excludes the NUL.
        if unsafe {
            WinHttpAddRequestHeaders(
                transfer.0,
                block.as_ptr(),
                (block.len() - 1) as u32,
                WINHTTP_ADDREQ_FLAG_ADD,
            )
        } == 0
        {
            return Err(winhttp_error("add request headers"));
        }
    }

    // SAFETY: no optional data, no context.
    if unsafe { WinHttpSendRequest(transfer.0, null(), 0, null(), 0, 0, 0) } == 0 {
        return Err(winhttp_error("send the request"));
    }
    // SAFETY: a live request handle.
    if unsafe { WinHttpReceiveResponse(transfer.0, null_mut()) } == 0 {
        return Err(winhttp_error("receive the response"));
    }

    let mut status = 0_u32;
    let mut size = std::mem::size_of::<u32>() as u32;
    // SAFETY: the buffer is a live u32 and `size` is its byte length.
    if unsafe {
        WinHttpQueryHeaders(
            transfer.0,
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            null(),
            std::ptr::from_mut(&mut status).cast::<c_void>(),
            &mut size,
            null_mut(),
        )
    } == 0
    {
        return Err(winhttp_error("read the response status"));
    }
    let head = Head {
        status: u16::try_from(status).map_err(|_| io::Error::other("implausible HTTP status"))?,
        raw: raw_headers(&transfer)?,
    };
    if !on_head(&head) {
        return Ok(Outcome::Aborted);
    }

    let mut buffer = vec![0_u8; BODY_CHUNK];
    loop {
        if request
            .total_timeout
            .is_some_and(|limit| started.elapsed() > limit)
        {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the transfer exceeded its overall time limit",
            ));
        }
        let mut available = 0_u32;
        // SAFETY: `available` is a live u32.
        if unsafe { WinHttpQueryDataAvailable(transfer.0, &mut available) } == 0 {
            return Err(winhttp_error("wait for response data"));
        }
        if available == 0 {
            return Ok(Outcome::Complete);
        }
        let want = (available as usize).min(buffer.len());
        let mut read = 0_u32;
        // SAFETY: the buffer holds at least `want` writable bytes.
        if unsafe {
            WinHttpReadData(transfer.0, buffer.as_mut_ptr().cast::<c_void>(), want as u32, &mut read)
        } == 0
        {
            return Err(winhttp_error("read response data"));
        }
        if read == 0 {
            return Ok(Outcome::Complete);
        }
        if !on_body(&buffer[..read as usize]) {
            return Ok(Outcome::Aborted);
        }
    }
}

/// The raw header block as CRLF-separated UTF-8 text.
fn raw_headers(request: &Handle) -> io::Result<Vec<u8>> {
    let mut size = 0_u32;
    // The first call reports the needed size (in bytes of UTF-16) and fails
    // with ERROR_INSUFFICIENT_BUFFER.
    // SAFETY: a null buffer with a zero size is the documented size probe.
    unsafe {
        WinHttpQueryHeaders(
            request.0,
            WINHTTP_QUERY_RAW_HEADERS_CRLF,
            null(),
            null_mut(),
            &mut size,
            null_mut(),
        );
    }
    // SAFETY: reads the calling thread's last-error value.
    if size == 0 || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER {
        return Err(winhttp_error("size the response headers"));
    }
    let mut units = vec![0_u16; (size as usize).div_ceil(2) + 1];
    let mut capacity = (units.len() * 2) as u32;
    // SAFETY: the buffer holds `capacity` writable bytes.
    if unsafe {
        WinHttpQueryHeaders(
            request.0,
            WINHTTP_QUERY_RAW_HEADERS_CRLF,
            null(),
            units.as_mut_ptr().cast::<c_void>(),
            &mut capacity,
            null_mut(),
        )
    } == 0
    {
        return Err(winhttp_error("read the response headers"));
    }
    units.truncate(capacity as usize / 2);
    let text = String::from_utf16_lossy(&units);
    // WinHTTP returns the header lines NUL-separated or CRLF-separated
    // depending on the query; normalize to CRLF lines and a blank terminator.
    let mut normalized = String::with_capacity(text.len() + 4);
    for line in text.split(['\0', '\n']) {
        let line = line.trim_end_matches('\r');
        if !line.is_empty() {
            normalized.push_str(line);
            normalized.push_str("\r\n");
        }
    }
    normalized.push_str("\r\n");
    Ok(normalized.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    /// A one-shot HTTP/1.1 server on an ephemeral loopback port: it reads one
    /// request, records it, and answers with `response`.
    fn serve(response: Vec<u8>) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0_u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).unwrap() == 0 {
                    break;
                }
                request.push(byte[0]);
            }
            stream.write_all(&response).unwrap();
            let _ = stream.flush();
            String::from_utf8_lossy(&request).into_owned()
        });
        (format!("http://127.0.0.1:{port}"), handle)
    }

    fn request<'a>(url: &'a str, headers: &'a [&'a str]) -> Get<'a> {
        Get {
            url,
            user_agent: "deltafin-test/1",
            headers,
            connect_timeout: Duration::from_secs(5),
            stall_timeout: Duration::from_secs(5),
            total_timeout: Some(Duration::from_secs(30)),
        }
    }

    #[test]
    fn urls_split_into_origin_and_path() {
        assert_eq!(split_url("https://example.com/a/b?c=d").unwrap(), (true, "example.com", 443, "/a/b?c=d"));
        assert_eq!(split_url("https://example.com:8443/x").unwrap(), (true, "example.com", 8443, "/x"));
        assert_eq!(split_url("http://127.0.0.1:9/").unwrap(), (false, "127.0.0.1", 9, "/"));
        assert_eq!(split_url("http://[::1]:8080/x").unwrap(), (false, "::1", 8080, "/x"));
        assert_eq!(split_url("https://[2001:db8::1]/").unwrap(), (true, "2001:db8::1", 443, "/"));
        assert_eq!(split_url("https://example.com").unwrap(), (true, "example.com", 443, ""));
        for bad in ["ftp://x/", "https://", "https://user@host/", "https://host:99999/", "host/path", "https://host/p#f", "https://[::1/", "https://a:b:c/", "https://[::1]x/"] {
            assert!(split_url(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_response_arrives_with_headers_and_body_and_the_request_is_as_asked() {
        let body = b"hello from the loopback".to_vec();
        let response = [
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nX-Probe: yes\r\nConnection: close\r\n\r\n", body.len()).into_bytes(),
            body.clone(),
        ]
        .concat();
        let (url, server) = serve(response);
        let mut head = None;
        let mut received = Vec::new();
        let outcome = get(
            &request(&format!("{url}/path?q=1"), &["Accept-Encoding: identity", "Range: bytes=0-"]),
            &mut |value| {
                head = Some(value.clone());
                true
            },
            &mut |chunk| {
                received.extend_from_slice(chunk);
                true
            },
        )
        .unwrap();
        assert_eq!(outcome, Outcome::Complete);
        assert_eq!(received, body);
        let head = head.unwrap();
        assert_eq!(head.status, 200);
        let text = String::from_utf8(head.raw).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text:?}");
        assert!(text.to_ascii_lowercase().contains("x-probe: yes\r\n"), "{text:?}");
        assert!(text.ends_with("\r\n\r\n"), "{text:?}");
        let seen = server.join().unwrap();
        assert!(seen.starts_with("GET /path?q=1 HTTP/1.1\r\n"), "{seen:?}");
        assert!(seen.to_ascii_lowercase().contains("user-agent: deltafin-test/1\r\n"), "{seen:?}");
        assert!(seen.to_ascii_lowercase().contains("accept-encoding: identity\r\n"), "{seen:?}");
        assert!(seen.to_ascii_lowercase().contains("range: bytes=0-\r\n"), "{seen:?}");
    }

    #[test]
    fn a_redirect_is_reported_and_never_followed() {
        let response = b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let (url, server) = serve(response);
        let mut status = 0;
        let mut text = String::new();
        let outcome = get(
            &request(&url, &[]),
            &mut |head| {
                status = head.status;
                text = String::from_utf8_lossy(&head.raw).into_owned();
                true
            },
            &mut |_| true,
        )
        .unwrap();
        assert_eq!(outcome, Outcome::Complete);
        assert_eq!(status, 302);
        assert!(text.to_ascii_lowercase().contains("location: http://127.0.0.1:1/elsewhere"), "{text:?}");
        server.join().unwrap();
    }

    #[test]
    fn either_callback_can_abort_the_transfer() {
        let body = vec![7_u8; 200_000];
        let response = [
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes(),
            body,
        ]
        .concat();
        let (url, server) = serve(response.clone());
        let outcome = get(&request(&url, &[]), &mut |_| false, &mut |_| panic!("no body after a refused head")).unwrap();
        assert_eq!(outcome, Outcome::Aborted);
        server.join().unwrap();

        let (url, server) = serve(response);
        let mut delivered = 0_usize;
        let outcome = get(
            &request(&url, &[]),
            &mut |_| true,
            &mut |chunk| {
                delivered += chunk.len();
                false
            },
        )
        .unwrap();
        assert_eq!(outcome, Outcome::Aborted);
        assert!(delivered > 0 && delivered < 200_000, "{delivered}");
        server.join().unwrap();
    }

    #[test]
    fn plain_http_to_a_remote_host_is_refused_before_any_connection() {
        let error = get(&request("http://example.com/", &[]), &mut |_| true, &mut |_| true).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn a_closed_port_is_a_clean_error_not_a_hang() {
        let port = {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let started = Instant::now();
        let error = get(&request(&format!("http://127.0.0.1:{port}/"), &[]), &mut |_| true, &mut |_| true).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(20), "{:?}", started.elapsed());
        assert!(!error.to_string().is_empty());
    }
}
