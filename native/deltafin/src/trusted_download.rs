//! Reusable shell-free HTTPS transfer substrate for pinned native installers.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
#[cfg(not(windows))]
use std::sync::OnceLock;
use std::time::Duration;

#[cfg(not(windows))]
use curl::easy::{Easy, List};

use crate::error::{DeltafinError, Result};
use crate::packfile::{Digest, digest_open_file};
use crate::sys::fs::{self as sysfs, Open};

const MAX_HEADER_BYTES: u64 = 64 << 10;
const CHUNK: usize = 8 << 20;
#[cfg(not(windows))]
const MINIMUM_LIBCURL_VERSION: u32 = 0x07_1c_00;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ByteRange {
    pub start: u64,
    pub end: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct Request {
    pub url: String,
    pub range: Option<ByteRange>,
    pub user_agent: &'static str,
    pub timeout: TimeoutPolicy,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum TimeoutPolicy {
    /// Bounded repository metadata, manifests, and byte-range probes.
    Metadata,
    /// Multi-gigabyte payload: no unrealistic wall-clock cap, but abort a
    /// connection that sustains less than 1 KiB/s for two minutes.
    LargePayload,
}

#[derive(Debug, Default)]
pub(crate) struct ResponseMeta {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    pub bytes: u64,
}

pub(crate) trait Transport {
    fn transfer(
        &mut self,
        request: &Request,
        target: &mut dyn Write,
        maximum: u64,
    ) -> Result<ResponseMeta>;
}

pub(crate) struct NativeHttpsTransport;

/// Admit and hash an exact regular file through one no-follow descriptor.
pub(crate) fn verify_regular_digest(path: &Path, size: u64, digest: Digest) -> Result<()> {
    let before =
        sysfs::lstat(path).map_err(|error| io_error("inspect pinned file", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() != size {
        return Err(DeltafinError::new(format!(
            "{} is not a regular, non-symlink file of pinned size {size}",
            path.display()
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open pinned file without following symlinks", path, error))?;
    let opened =
        sysfs::fstat(&file).map_err(|error| io_error("stat pinned file", path, error))?;
    if !opened.is_file() || opened.len() != size || opened.id() != before.id() {
        return Err(DeltafinError::new(format!(
            "pinned file changed while opening: {}",
            path.display()
        )));
    }
    let actual =
        digest_open_file(&file, path).map_err(|error| DeltafinError::new(error.to_string()))?;
    let after =
        sysfs::fstat(&file).map_err(|error| io_error("restat pinned file", path, error))?;
    if (after.id(), after.len()) != (opened.id(), size) {
        return Err(DeltafinError::new(format!(
            "pinned file changed while hashing: {}",
            path.display()
        )));
    }
    if actual != digest {
        return Err(DeltafinError::new(format!(
            "{} SHA-256 does not match its pin",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn secure_create_new(path: &Path, mode: u32) -> Result<File> {
    Open::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("create regular file safely", path, error))
}

pub(crate) fn fsync_directory(path: &Path) -> Result<()> {
    let metadata =
        sysfs::lstat(path).map_err(|error| io_error("inspect directory", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "{} is not a real directory",
            path.display()
        )));
    }
    let directory = Open::new()
        .read(true)
        .directory()
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open directory for fsync", path, error))?;
    sysfs::sync_directory(&directory).map_err(|error| io_error("fsync directory", path, error))
}

/// Publish a verified partial without overwrite, then durably unlink its old name.
pub(crate) fn publish_hard_link(part: &Path, final_path: &Path, directory: &Path) -> Result<()> {
    sysfs::publish_without_replace(part, final_path)
        .map_err(|error| io_error("publish verified file without overwrite", final_path, error))?;
    fsync_directory(directory)
}

/// Atomically rename without ever replacing a path that raced into place.
pub(crate) fn rename_noreplace(source: &Path, destination: &Path) -> Result<()> {
    sysfs::rename_noreplace(source, destination)
        .map_err(|error| io_error("atomically publish without replacement", destination, error))
}

impl Transport for NativeHttpsTransport {
    fn transfer(
        &mut self,
        request: &Request,
        target: &mut dyn Write,
        maximum: u64,
    ) -> Result<ResponseMeta> {
        require_absolute_https(&request.url)?;
        let mut url = request.url.clone();
        let mut header_budget = MAX_HEADER_BYTES;
        for redirects_followed in 0..=MAX_REDIRECTS {
            let response = perform_once(request, &url, target, maximum, header_budget)?;
            header_budget = header_budget
                .checked_sub(response.header_bytes)
                .ok_or_else(|| {
                    DeltafinError::new("HTTP response headers exceeded the safety limit")
                })?;
            let Some(next) =
                redirect_destination(&url, response.status, &response.headers, redirects_followed)?
            else {
                return Ok(ResponseMeta {
                    status: response.status,
                    headers: response.headers,
                    bytes: response.bytes,
                });
            };
            url = next;
        }
        unreachable!("redirect_destination rejects a redirect beyond the limit")
    }
}

const MAX_REDIRECTS: usize = 5;

#[derive(Debug)]
struct SingleResponse {
    status: u16,
    headers: BTreeMap<String, String>,
    bytes: u64,
    header_bytes: u64,
}

/// Windows speaks HTTPS through the operating system's own WinHTTP/Schannel
/// stack (via `deltafin_sys::https`), not through a bundled or system libcurl.
/// The header and body state machines are the ones the libcurl path uses, fed
/// the same bytes in the same order, so every bound and every refusal is
/// identical on both transports.
#[cfg(windows)]
fn perform_once(
    request: &Request,
    url: &str,
    target: &mut dyn Write,
    maximum: u64,
    header_budget: u64,
) -> Result<SingleResponse> {
    use deltafin_sys::https;
    require_absolute_https(url)?;
    let mut request_headers = vec!["Accept-Encoding: identity".to_owned()];
    if let Some(range) = &request.range {
        request_headers.push(match range.end {
            Some(end) => format!("Range: bytes={}-{end}", range.start),
            None => format!("Range: bytes={}-", range.start),
        });
    }
    let request_headers: Vec<&str> = request_headers.iter().map(String::as_str).collect();
    let header_state = RefCell::new(HeaderState::new(header_budget));
    let mut body_error = None;
    let mut body_bytes = 0_u64;
    let outcome = https::get(
        &https::Get {
            url,
            user_agent: request.user_agent,
            headers: &request_headers,
            connect_timeout: Duration::from_secs(30),
            stall_timeout: Duration::from_secs(120),
            total_timeout: (request.timeout == TimeoutPolicy::Metadata)
                .then_some(Duration::from_secs(120)),
        },
        &mut |head| {
            // Hand the header block to the shared parser one line at a time,
            // exactly as libcurl's header callback would.
            let mut state = header_state.borrow_mut();
            head.raw
                .split_inclusive(|byte| *byte == b'\n')
                .all(|line| state.receive(line))
        },
        &mut |data| {
            match receive_body_chunk(
                &header_state.borrow(),
                target,
                data,
                maximum,
                &mut body_bytes,
            ) {
                Ok(_) => true,
                Err(error) => {
                    body_error = Some(error);
                    false
                }
            }
        },
    );
    let state = header_state.into_inner();
    if let Some(error) = state.error {
        return Err(error);
    }
    if let Some(error) = body_error {
        return Err(error);
    }
    outcome.map_err(|error| {
        DeltafinError::new(format!(
            "in-process HTTPS transfer failed for {url}: {error}"
        ))
    })?;
    let (status, headers) = state.parsed.ok_or_else(|| {
        DeltafinError::new("HTTPS transport returned no complete response headers")
    })?;
    Ok(SingleResponse {
        status,
        headers,
        bytes: body_bytes,
        header_bytes: state.raw.len() as u64,
    })
}

#[cfg(not(windows))]
fn perform_once(
    request: &Request,
    url: &str,
    target: &mut dyn Write,
    maximum: u64,
    header_budget: u64,
) -> Result<SingleResponse> {
    require_absolute_https(url)?;
    require_https_capable_libcurl()?;
    let mut easy = Easy::new();
    easy.url(url)
        .map_err(|error| curl_option_error("set HTTPS URL", error))?;
    easy.useragent(request.user_agent)
        .map_err(|error| curl_option_error("set HTTPS user agent", error))?;
    easy.connect_timeout(Duration::from_secs(30))
        .map_err(|error| curl_option_error("set HTTPS connect timeout", error))?;
    easy.low_speed_limit(1024)
        .map_err(|error| curl_option_error("set HTTPS low-speed limit", error))?;
    easy.low_speed_time(Duration::from_secs(120))
        .map_err(|error| curl_option_error("set HTTPS low-speed timeout", error))?;
    easy.follow_location(false)
        .map_err(|error| curl_option_error("disable automatic redirects", error))?;
    if request.timeout == TimeoutPolicy::Metadata {
        easy.timeout(Duration::from_secs(120))
            .map_err(|error| curl_option_error("set HTTPS metadata timeout", error))?;
    }
    if let Some(range) = &request.range {
        let value = match range.end {
            Some(end) => format!("{}-{end}", range.start),
            None => format!("{}-", range.start),
        };
        easy.range(&value)
            .map_err(|error| curl_option_error("set HTTPS byte range", error))?;
    }
    let mut request_headers = List::new();
    request_headers
        .append("Accept-Encoding: identity")
        .map_err(|error| curl_option_error("set identity content encoding", error))?;
    easy.http_headers(request_headers)
        .map_err(|error| curl_option_error("install HTTPS request headers", error))?;

    let header_state = RefCell::new(HeaderState::new(header_budget));
    let mut body_error = None;
    let mut body_bytes = 0_u64;
    let transfer_result = {
        let mut transfer = easy.transfer();
        transfer
            .header_function(|data| header_state.borrow_mut().receive(data))
            .map_err(|error| curl_option_error("install HTTPS header callback", error))?;
        transfer
            .write_function(|data| {
                match receive_body_chunk(
                    &header_state.borrow(),
                    target,
                    data,
                    maximum,
                    &mut body_bytes,
                ) {
                    Ok(consumed) => Ok(consumed),
                    Err(error) => {
                        body_error = Some(error);
                        Ok(0)
                    }
                }
            })
            .map_err(|error| curl_option_error("install HTTPS body callback", error))?;
        transfer.perform()
    };

    let state = header_state.into_inner();
    if let Some(error) = state.error {
        return Err(error);
    }
    if let Some(error) = body_error {
        return Err(error);
    }
    transfer_result.map_err(|error| {
        DeltafinError::new(format!(
            "in-process HTTPS transfer failed for {url}: {error}"
        ))
    })?;
    let (status, headers) = state.parsed.ok_or_else(|| {
        DeltafinError::new("HTTPS transport returned no complete response headers")
    })?;
    Ok(SingleResponse {
        status,
        headers,
        bytes: body_bytes,
        header_bytes: state.raw.len() as u64,
    })
}

/// Prove once that the system library selected by the reviewed direct
/// `curl-sys` build can implement Deltafin's only admitted transport.  The
/// build-time ELF/Mach-O checks establish library identity; these runtime
/// capability bits establish version, TLS, and HTTPS support without launching
/// `curl`, `curl-config`, a shell, or any other helper process.
#[cfg(not(windows))]
fn require_https_capable_libcurl() -> Result<()> {
    static CHECK: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    let result = CHECK.get_or_init(|| {
        let version = curl::Version::get();
        if version.version_num() < MINIMUM_LIBCURL_VERSION {
            return Err(format!(
                "linked libcurl {} is older than Deltafin's minimum 7.28.0",
                version.version()
            ));
        }
        if !version.feature_ssl() {
            return Err(format!(
                "linked libcurl {} has no TLS/SSL support",
                version.version()
            ));
        }
        if !version
            .protocols()
            .any(|protocol| protocol.eq_ignore_ascii_case("https"))
        {
            return Err(format!(
                "linked libcurl {} does not advertise the HTTPS protocol",
                version.version()
            ));
        }
        Ok(())
    });
    result
        .as_ref()
        .map(|_| ())
        .map_err(|message| DeltafinError::new(message.clone()))
}

#[derive(Debug)]
struct HeaderState {
    raw: Vec<u8>,
    maximum: u64,
    parsed: Option<(u16, BTreeMap<String, String>)>,
    error: Option<DeltafinError>,
}

impl HeaderState {
    fn new(maximum: u64) -> Self {
        Self {
            raw: Vec::new(),
            maximum,
            parsed: None,
            error: None,
        }
    }

    fn receive(&mut self, data: &[u8]) -> bool {
        if self.error.is_some() {
            return false;
        }
        let next = self.raw.len() as u64 + data.len() as u64;
        if next > self.maximum {
            self.error = Some(DeltafinError::new(
                "HTTP response headers exceeded the safety limit",
            ));
            return false;
        }
        self.raw.extend_from_slice(data);
        if data == b"\r\n" {
            match parse_final_headers(&self.raw) {
                Ok(parsed) => self.parsed = Some(parsed),
                Err(error) => {
                    self.error = Some(error);
                    return false;
                }
            }
        }
        true
    }

    fn is_followable_redirect(&self) -> bool {
        self.parsed.as_ref().is_some_and(|(status, headers)| {
            is_redirect_status(*status)
                && headers
                    .get("location")
                    .is_some_and(|value| !value.is_empty())
        })
    }
}

fn receive_body_chunk(
    headers: &HeaderState,
    target: &mut dyn Write,
    data: &[u8],
    maximum: u64,
    total: &mut u64,
) -> Result<usize> {
    if headers.error.is_some() || headers.parsed.is_none() {
        return Err(DeltafinError::new(
            "HTTPS response body arrived before valid complete headers",
        ));
    }
    if headers.is_followable_redirect() {
        return Ok(data.len());
    }
    write_response_chunk(target, data, maximum, total)?;
    Ok(data.len())
}

fn write_response_chunk(
    target: &mut dyn Write,
    data: &[u8],
    maximum: u64,
    total: &mut u64,
) -> Result<()> {
    let next = total
        .checked_add(data.len() as u64)
        .ok_or_else(|| DeltafinError::new("HTTPS response byte count overflowed"))?;
    if next > maximum {
        return Err(DeltafinError::new(format!(
            "remote response exceeded {maximum} bytes"
        )));
    }
    target
        .write_all(data)
        .map_err(|error| DeltafinError::new(format!("write HTTPS response body: {error}")))?;
    *total = next;
    Ok(())
}

fn redirect_destination(
    current: &str,
    status: u16,
    headers: &BTreeMap<String, String>,
    redirects_followed: usize,
) -> Result<Option<String>> {
    if !is_redirect_status(status) {
        return Ok(None);
    }
    let Some(location) = headers.get("location").filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if redirects_followed >= MAX_REDIRECTS {
        return Err(DeltafinError::new(format!(
            "HTTPS redirect limit of {MAX_REDIRECTS} was exceeded"
        )));
    }
    resolve_https_redirect(current, location).map(Some)
}

fn is_redirect_status(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

fn require_absolute_https(url: &str) -> Result<()> {
    let bytes = url.as_bytes();
    let authority_end = bytes
        .get(8..)
        .and_then(|rest| {
            rest.iter()
                .position(|byte| matches!(byte, b'/' | b'?' | b'#'))
                .map(|offset| offset + 8)
        })
        .unwrap_or(bytes.len());
    let authority = bytes.get(8..authority_end).unwrap_or_default();
    if bytes.len() < 9
        || !bytes[..8].eq_ignore_ascii_case(b"https://")
        || authority.is_empty()
        || authority[0] == b':'
        || authority.iter().any(|byte| matches!(byte, b'@' | b'\\'))
        || bytes
            .iter()
            .any(|byte| byte.is_ascii_control() || matches!(byte, b' ' | b'\\'))
    {
        return Err(DeltafinError::new(
            "trusted download transport accepts only absolute HTTPS URLs",
        ));
    }
    Ok(())
}

fn resolve_https_redirect(current: &str, location: &str) -> Result<String> {
    require_absolute_https(current)?;
    if location.is_empty()
        || location
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
    {
        return Err(DeltafinError::new("HTTPS redirect location is unsafe"));
    }
    if has_uri_scheme(location) {
        require_absolute_https(location)?;
        return Ok(location.to_owned());
    }
    let scheme_end = current
        .find("://")
        .ok_or_else(|| DeltafinError::new("current HTTPS URL has no authority"))?;
    let authority_start = scheme_end + 3;
    let authority_end = current[authority_start..]
        .find(['/', '?', '#'])
        .map_or(current.len(), |offset| authority_start + offset);
    let origin = &current[..authority_end];
    if let Some(rest) = location.strip_prefix("//") {
        let next = format!("https://{rest}");
        require_absolute_https(&next)?;
        return Ok(next);
    }
    if location.starts_with('/') {
        return Ok(format!("{origin}{location}"));
    }
    let current_without_fragment = current.split('#').next().unwrap_or(current);
    if location.starts_with('#') {
        return Ok(format!("{}{location}", current_without_fragment));
    }
    let current_without_query = current_without_fragment
        .split('?')
        .next()
        .unwrap_or(current_without_fragment);
    if location.starts_with('?') {
        return Ok(format!("{current_without_query}{location}"));
    }
    let path_start = current_without_query[authority_end.min(current_without_query.len())..]
        .find('/')
        .map(|offset| authority_end + offset);
    let directory_end = path_start
        .and_then(|start| {
            current_without_query[start..]
                .rfind('/')
                .map(|offset| start + offset + 1)
        })
        .unwrap_or_else(|| {
            current_without_query.len() + usize::from(!current_without_query.ends_with('/'))
        });
    let base = if path_start.is_none() {
        format!("{origin}/")
    } else {
        current_without_query[..directory_end].to_owned()
    };
    Ok(format!("{base}{location}"))
}

fn has_uri_scheme(value: &str) -> bool {
    let Some(colon) = value.find(':') else {
        return false;
    };
    let prefix = &value[..colon];
    !prefix.is_empty()
        && prefix.as_bytes()[0].is_ascii_alphabetic()
        && prefix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
}

#[cfg(not(windows))]
fn curl_option_error(operation: &str, error: curl::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation}: {error}"))
}

pub(crate) fn read_bounded(mut reader: impl Read, maximum: u64) -> Result<Vec<u8>> {
    let mut value = Vec::new();
    copy_bounded(&mut reader, &mut value, maximum)?;
    Ok(value)
}

fn copy_bounded(reader: &mut dyn Read, target: &mut dyn Write, maximum: u64) -> Result<u64> {
    let mut buffer = vec![0_u8; CHUNK];
    let mut total = 0_u64;
    loop {
        let remaining = maximum.saturating_sub(total);
        let capacity = if remaining >= buffer.len() as u64 {
            buffer.len()
        } else {
            usize::try_from(remaining + 1)
                .map_err(|_| DeltafinError::new("response bound does not fit usize"))?
        };
        let read = reader
            .read(&mut buffer[..capacity])
            .map_err(|error| DeltafinError::new(format!("read bounded response body: {error}")))?;
        if read == 0 {
            return Ok(total);
        }
        if read as u64 > remaining {
            return Err(DeltafinError::new(format!(
                "remote response exceeded {maximum} bytes"
            )));
        }
        target
            .write_all(&buffer[..read])
            .map_err(|error| DeltafinError::new(format!("write bounded response body: {error}")))?;
        total += read as u64;
    }
}

fn parse_final_headers(raw: &[u8]) -> Result<(u16, BTreeMap<String, String>)> {
    if raw.len() as u64 > MAX_HEADER_BYTES {
        return Err(DeltafinError::new(
            "HTTP response headers exceeded the safety limit",
        ));
    }
    let text = std::str::from_utf8(raw)
        .map_err(|_| DeltafinError::new("HTTP response headers are not UTF-8"))?;
    let block = text
        .split("\r\n\r\n")
        .filter(|block| block.starts_with("HTTP/"))
        .last()
        .ok_or_else(|| DeltafinError::new("no complete HTTP response headers were returned"))?;
    let mut lines = block.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| DeltafinError::new("HTTP status line is missing"))?;
    let mut fields = status_line.split_ascii_whitespace();
    let protocol = fields.next().unwrap_or_default();
    let status_text = fields.next().unwrap_or_default();
    if !protocol.starts_with("HTTP/") || status_text.len() != 3 {
        return Err(DeltafinError::new("invalid HTTP status line"));
    }
    let status = status_text
        .parse::<u16>()
        .map_err(|_| DeltafinError::new("invalid HTTP status code"))?;
    let mut headers = BTreeMap::new();
    for line in lines {
        if line.starts_with([' ', '\t']) || line.contains('\0') {
            return Err(DeltafinError::new("unsafe folded HTTP response header"));
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| DeltafinError::new("malformed HTTP response header"))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(DeltafinError::new("invalid HTTP response header name"));
        }
        let name = name.to_ascii_lowercase();
        if headers
            .insert(name.clone(), value.trim().to_owned())
            .is_some()
        {
            return Err(DeltafinError::new(format!(
                "duplicate HTTP response header {name:?}"
            )));
        }
    }
    Ok((status, headers))
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[cfg(not(windows))]
    #[test]
    fn linked_system_curl_meets_the_https_contract() {
        require_https_capable_libcurl().unwrap();
        // The cached result must remain reusable without touching the loader
        // or constructing a network handle again.
        require_https_capable_libcurl().unwrap();
    }

    #[test]
    fn final_header_block_is_strict_and_bounded() {
        let raw = b"HTTP/2 302\r\nlocation: https://example.invalid\r\n\r\nHTTP/2 206\r\ncontent-range: bytes 0-7/10\r\n\r\n";
        let (status, headers) = parse_final_headers(raw).unwrap();
        assert_eq!(status, 206);
        assert_eq!(headers["content-range"], "bytes 0-7/10");
        assert!(parse_final_headers(b"HTTP/1.1 200 OK\r\nx-a: 1\r\nx-a: 2\r\n\r\n").is_err());

        let mut callback = HeaderState::new(raw.len() as u64);
        for line in [
            &b"HTTP/2 302\r\n"[..],
            &b"location: https://example.invalid\r\n"[..],
            &b"\r\n"[..],
            &b"HTTP/2 206\r\n"[..],
            &b"content-range: bytes 0-7/10\r\n"[..],
            &b"\r\n"[..],
        ] {
            assert!(callback.receive(line));
        }
        assert_eq!(callback.parsed.as_ref().unwrap().0, 206);
        assert_eq!(callback.raw, raw);
        assert!(!HeaderState::new(3).receive(b"HTTP/2 200\r\n"));
    }

    #[test]
    fn body_callback_discards_redirects_and_propagates_exact_bounds_and_writes() {
        let mut redirect = HeaderState::new(256);
        assert!(redirect.receive(b"HTTP/2 302\r\n"));
        assert!(redirect.receive(b"location: /next\r\n"));
        assert!(redirect.receive(b"\r\n"));
        let mut output = Vec::new();
        let mut total = 0;
        assert_eq!(
            receive_body_chunk(&redirect, &mut output, b"discard", 1, &mut total).unwrap(),
            7
        );
        assert!(output.is_empty());
        assert_eq!(total, 0);

        let mut final_headers = HeaderState::new(256);
        assert!(final_headers.receive(b"HTTP/2 206\r\n"));
        assert!(final_headers.receive(b"content-range: bytes 0-3/4\r\n"));
        assert!(final_headers.receive(b"\r\n"));
        assert_eq!(
            receive_body_chunk(&final_headers, &mut output, b"1234", 4, &mut total).unwrap(),
            4
        );
        assert_eq!(output, b"1234");
        assert!(receive_body_chunk(&final_headers, &mut output, b"5", 4, &mut total).is_err());
        let mut early_total = 0;
        assert!(
            receive_body_chunk(
                &HeaderState::new(256),
                &mut Vec::new(),
                b"early",
                8,
                &mut early_total,
            )
            .is_err()
        );

        struct RefuseWrites;
        impl Write for RefuseWrites {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "refused"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut refused_total = 0;
        let error = receive_body_chunk(
            &final_headers,
            &mut RefuseWrites,
            b"x",
            8,
            &mut refused_total,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("write HTTPS response body: refused")
        );

        let mut output = Vec::new();
        assert_eq!(copy_bounded(&mut &b"1234"[..], &mut output, 4).unwrap(), 4);
        assert!(copy_bounded(&mut &b"12345"[..], &mut Vec::new(), 4).is_err());
    }

    #[test]
    fn redirect_resolution_never_crosses_out_of_https() {
        let current = "https://example.invalid/a/b?old=1#fragment";
        assert_eq!(
            resolve_https_redirect(current, "https://cdn.invalid/model").unwrap(),
            "https://cdn.invalid/model"
        );
        assert_eq!(
            resolve_https_redirect(current, "//cdn.invalid/model").unwrap(),
            "https://cdn.invalid/model"
        );
        assert_eq!(
            resolve_https_redirect(current, "/model").unwrap(),
            "https://example.invalid/model"
        );
        assert_eq!(
            resolve_https_redirect(current, "next").unwrap(),
            "https://example.invalid/a/next"
        );
        assert_eq!(
            resolve_https_redirect(current, "?new=2").unwrap(),
            "https://example.invalid/a/b?new=2"
        );
        assert!(resolve_https_redirect(current, "http://example.invalid/model").is_err());
        assert!(resolve_https_redirect(current, "ftp://example.invalid/model").is_err());
        assert!(require_absolute_https("https:///missing-host").is_err());
        assert!(require_absolute_https("https://user@example.invalid/model").is_err());
        assert!(require_absolute_https("https://:443/model").is_err());
        assert!(require_absolute_https("https://example.invalid\\@evil.invalid/model").is_err());

        let headers = BTreeMap::from([("location".to_owned(), "/next".to_owned())]);
        assert_eq!(
            redirect_destination(current, 302, &headers, 4).unwrap(),
            Some("https://example.invalid/next".to_owned())
        );
        assert!(redirect_destination(current, 302, &headers, 5).is_err());
        assert!(
            redirect_destination(current, 304, &headers, 0)
                .unwrap()
                .is_none()
        );
    }
}
