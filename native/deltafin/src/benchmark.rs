//! Auditable, Python-free end-to-end benchmark campaigns.
//!
//! A campaign always executes the current compiled Deltafin binary.  Human
//! stdout is retained as an artifact, but it is never parsed as benchmark
//! evidence: [`crate::run_events`] JSONL is the only production result API.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
#[cfg(unix)]
use std::ffi::{CStr, CString};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::dspark_checkpoint::strict_json;
use crate::error::{DeltafinError, Result};
use crate::packfile::DigestState;
use crate::run_events::{EVENT_SCHEMA, MAX_EVENT_STREAM_BYTES};
use crate::trusted_download::{fsync_directory, secure_create_new};
use crate::sys::fs::{self as sys_fs, Open};

pub const BENCHMARK_SCHEMA: &str = "deltafin.benchmark.v1";
const MAX_EVENT_LINE_BYTES: usize = 16 * 1024 * 1024;
const MAX_EVENT_FILE_BYTES: u64 = MAX_EVENT_STREAM_BYTES;
const MAX_STDOUT_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_STDERR_FILE_BYTES: u64 = 16 * 1024 * 1024;
const POLL_QUANTUM: Duration = Duration::from_millis(10);
const MAX_GIT_POINTER_BYTES: u64 = 4 * 1024;
const MAX_GIT_PACKED_REFS_BYTES: u64 = 64 * 1024 * 1024;
const MAX_GIT_INDEX_BYTES: u64 = 256 * 1024 * 1024;

const PERFORMANCE_ENV_PREFIXES: &[&str] = &[
    "K3_",
    "DELTAFIN_",
    "PYTORCH_",
    "TORCH_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "BLIS_",
    "VECLIB_",
    "METAL_",
    "MLX_",
    "CUDA_",
    "HIP_",
    "NCCL_",
    "RAYON_",
];

const SECRET_KEY_WORDS: &[&str] = &[
    "AUTH",
    "AUTHORIZATION",
    "BEARER",
    "COOKIE",
    "CREDENTIAL",
    "CREDENTIALS",
    "PASSWD",
    "PASSWORD",
    "PAT",
    "SECRET",
    "TOKEN",
];

const SECRET_KEY_COMPOUNDS: &[&str] = &[
    "ACCESSKEY",
    "ACCESSTOKEN",
    "APIKEY",
    "AUTHTOKEN",
    "BEARERTOKEN",
    "CLIENTSECRET",
    "COOKIEJAR",
    "PRIVATEKEY",
    "REFRESHTOKEN",
    "SECRETACCESSKEY",
    "SECRETKEY",
    "SESSIONTOKEN",
    "SIGNINGKEY",
];

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BenchmarkArm {
    pub name: String,
    pub environment_spec: String,
}

impl BenchmarkArm {
    pub fn new(name: impl Into<String>, environment_spec: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            environment_spec: environment_spec.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    /// Repository and default model root used as the child's working directory.
    pub repository_root: PathBuf,
    pub prompt: String,
    pub chat: bool,
    pub max_new_tokens: u64,
    pub repetitions: usize,
    pub warmup_steps: usize,
    pub timeout: Duration,
    pub output_dir: Option<PathBuf>,
    pub arms: Vec<BenchmarkArm>,
    pub expected_completion_token_ids: Option<Vec<u64>>,
    pub expected_completion_text: Option<String>,
    pub keep_going: bool,
    runner: PathBuf,
}

impl BenchmarkOptions {
    /// Construct production options whose runner is this exact executable.
    pub fn for_current_executable(repository_root: impl Into<PathBuf>) -> Result<Self> {
        let runner = env::current_exe().map_err(|error| {
            DeltafinError::new(format!("resolve current Deltafin executable: {error}"))
        })?;
        Ok(Self {
            repository_root: repository_root.into(),
            prompt: "The capital of France is".into(),
            chat: false,
            max_new_tokens: 4,
            repetitions: 3,
            warmup_steps: 0,
            timeout: Duration::from_secs(3600),
            output_dir: None,
            arms: vec![BenchmarkArm::new("defaults", "")],
            expected_completion_token_ids: None,
            expected_completion_text: None,
            keep_going: false,
            runner,
        })
    }

    #[cfg(test)]
    fn with_test_runner(mut self, runner: PathBuf) -> Self {
        self.runner = runner;
        self
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub output_dir: PathBuf,
    pub summary: Value,
}

impl BenchmarkReport {
    pub fn succeeded(&self) -> bool {
        self.summary.get("all_runs_valid").and_then(Value::as_bool) == Some(true)
            && self
                .summary
                .get("all_outputs_exact")
                .and_then(Value::as_bool)
                == Some(true)
    }
}

#[derive(Debug, Clone, Serialize)]
struct ParsedRun {
    source: &'static str,
    parse_errors: Vec<String>,
    runner_config: Option<Value>,
    runner_runtime: Option<Value>,
    eos_token_id: Option<u64>,
    input_token_ids: Option<Vec<u64>>,
    prompt_token_ids_at_end: Option<Vec<u64>>,
    prefill_ns: Option<u64>,
    prefill_s: Option<f64>,
    decode_steps: Vec<Value>,
    decode_step_count: usize,
    decode_ns: u64,
    decode_emitted_tokens: usize,
    decode_tps: Option<f64>,
    steady_warmup_steps_dropped: usize,
    steady_decode_ns: u64,
    steady_decode_tokens: usize,
    steady_tps: Option<f64>,
    steady_s_per_token: Option<f64>,
    emitted_token_ids: Option<Vec<u64>>,
    completion_token_ids: Option<Vec<u64>>,
    completion_text: Option<String>,
    runner_status: Option<String>,
    stop_reason: Option<String>,
    proposed_draft_tokens: u64,
    accepted_draft_tokens: u64,
    draft_acceptance_rate: Option<f64>,
    steady_proposed_draft_tokens: u64,
    steady_accepted_draft_tokens: u64,
    steady_draft_acceptance_rate: Option<f64>,
    inference_ns: Option<u64>,
    inference_s: Option<f64>,
}

#[derive(Debug)]
struct ChildOutcome {
    status: Option<ExitStatus>,
    timed_out: bool,
    spawn_error: Option<String>,
    capture_errors: Vec<String>,
    stdout_bytes: u64,
    stderr_bytes: u64,
    wall_ns: u64,
}

#[derive(Debug)]
struct CapturedPipe {
    bytes_written: u64,
    overflowed: bool,
    io_error: Option<String>,
}

#[derive(Debug)]
struct RunnerWait {
    status: ExitStatus,
    timed_out: bool,
    live_limit_error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct CaptureLimits {
    stdout_bytes: u64,
    stderr_bytes: u64,
    event_bytes: u64,
}

const PRODUCTION_CAPTURE_LIMITS: CaptureLimits = CaptureLimits {
    stdout_bytes: MAX_STDOUT_FILE_BYTES,
    stderr_bytes: MAX_STDERR_FILE_BYTES,
    event_bytes: MAX_EVENT_FILE_BYTES,
};

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
struct NativeFileIdentity {
    device: u64,
    inode: u64,
    size_bytes: u64,
    mode: u32,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
struct PinnedRunner {
    original_path: PathBuf,
    executable_path: PathBuf,
    sha256: String,
    size_bytes: u64,
    source_identity: NativeFileIdentity,
}

#[derive(Debug)]
struct RunRecord {
    value: Value,
    valid: bool,
    output_match: Option<bool>,
    completion_ids: Option<Vec<u64>>,
    completion_text: Option<String>,
    result_path: PathBuf,
}

/// Parse `KEY=VALUE` words with POSIX-style quoting, without invoking a shell.
/// Expansions, substitutions, comments, and command execution do not exist.
pub fn parse_environment_delta(spec: &str) -> Result<BTreeMap<String, String>> {
    let words = split_shell_words(spec)?;
    let mut delta = BTreeMap::new();
    for word in words {
        let Some((key, value)) = word.split_once('=') else {
            return Err(DeltafinError::new(format!(
                "invalid benchmark config word {word:?}; every word must be KEY=VALUE"
            )));
        };
        if !valid_environment_name(key) {
            return Err(DeltafinError::new(format!(
                "invalid benchmark environment variable name {key:?}"
            )));
        }
        if crate::loader_audit::is_dynamic_loader_environment_name(OsStr::new(key)) {
            return Err(DeltafinError::new(format!(
                "dynamic-loader environment variable {key:?} is forbidden in benchmark arms"
            )));
        }
        if key == "K3_METAL_SRC" && !cfg!(debug_assertions) {
            return Err(DeltafinError::new(
                "K3_METAL_SRC is disabled in production/default release benchmark builds; the embedded reviewed precompiled Metal library is mandatory",
            ));
        }
        if sensitive_environment_name(key) {
            return Err(DeltafinError::new(format!(
                "secret-bearing environment variable {key:?} is not allowed in a persisted benchmark config; inherit credentials from the parent instead"
            )));
        }
        delta.insert(key.to_owned(), value.to_owned());
    }
    Ok(delta)
}

/// Parse an exact completion-token oracle from JSON (`[1,2]`) or a compact
/// comma-separated form (`1,2`).
pub fn parse_expected_token_ids(spec: &str) -> Result<Vec<u64>> {
    if let Ok(value) = strict_json(spec.as_bytes(), "expected completion token IDs") {
        return token_ids(
            Some(&value),
            "expected completion token IDs",
            &mut Vec::new(),
        )
        .ok_or_else(|| {
            DeltafinError::new(
                "expected completion token IDs must be a JSON array of nonnegative integers",
            )
        });
    }
    let mut ids = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        ids.push(part.parse::<u64>().map_err(|_| {
            DeltafinError::new(
                "expected completion token IDs must be JSON [1,2] or comma-separated nonnegative integers",
            )
        })?);
    }
    if ids.is_empty() && !spec.trim().is_empty() {
        return Err(DeltafinError::new(
            "expected completion token IDs contain no integers",
        ));
    }
    Ok(ids)
}

fn split_shell_words(spec: &str) -> Result<Vec<String>> {
    #[derive(Clone, Copy, Eq, PartialEq)]
    enum Quote {
        None,
        Single,
        Double,
    }

    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote = Quote::None;
    let mut escaped = false;
    let mut started = false;
    for character in spec.chars() {
        if escaped {
            // POSIX double quotes only make these characters escapable.  For
            // compatibility with shlex, retain a backslash before others.
            if quote == Quote::Double && !matches!(character, '"' | '\\' | '$' | '`' | '\n') {
                word.push('\\');
            }
            if character != '\n' {
                word.push(character);
            }
            escaped = false;
            started = true;
            continue;
        }
        match quote {
            Quote::Single => {
                if character == '\'' {
                    quote = Quote::None;
                } else {
                    word.push(character);
                }
                started = true;
            }
            Quote::Double => match character {
                '"' => {
                    quote = Quote::None;
                    started = true;
                }
                '\\' => {
                    escaped = true;
                    started = true;
                }
                _ => {
                    word.push(character);
                    started = true;
                }
            },
            Quote::None => match character {
                '\'' => {
                    quote = Quote::Single;
                    started = true;
                }
                '"' => {
                    quote = Quote::Double;
                    started = true;
                }
                '\\' => {
                    escaped = true;
                    started = true;
                }
                value if value.is_whitespace() => {
                    if started {
                        words.push(std::mem::take(&mut word));
                        started = false;
                    }
                }
                _ => {
                    word.push(character);
                    started = true;
                }
            },
        }
    }
    if escaped {
        return Err(DeltafinError::new(
            "benchmark environment specification ends with an escape",
        ));
    }
    if quote != Quote::None {
        return Err(DeltafinError::new(
            "benchmark environment specification has an unterminated quote",
        ));
    }
    if started {
        words.push(word);
    }
    Ok(words)
}

fn valid_environment_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'a'..=b'z' | b'_'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn sensitive_environment_name(name: &str) -> bool {
    let uppercase = name.to_ascii_uppercase();
    let words = uppercase.split('_').collect::<Vec<_>>();
    if SECRET_KEY_WORDS.iter().any(|secret| words.contains(secret)) {
        return true;
    }
    if words.iter().any(|word| {
        word.contains("PASSWORD")
            || word.contains("PASSWD")
            || word.contains("SECRET")
            || word.contains("CREDENTIAL")
            || word.contains("BEARER")
            || word.contains("COOKIE")
            || word.ends_with("TOKEN")
            || word.starts_with("AUTH")
    }) {
        return true;
    }
    let normalized = uppercase
        .bytes()
        .filter(u8::is_ascii_alphanumeric)
        .map(char::from)
        .collect::<String>();
    SECRET_KEY_COMPOUNDS
        .iter()
        .any(|compound| normalized.contains(compound))
}

fn relevant_environment<I, K, V>(environment: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
{
    environment
        .into_iter()
        .filter_map(|(key, value)| {
            let key = key.as_ref().to_str()?;
            if sensitive_environment_name(key)
                || !PERFORMANCE_ENV_PREFIXES
                    .iter()
                    .any(|prefix| key.starts_with(prefix))
            {
                return None;
            }
            Some((
                key.to_owned(),
                value.as_ref().to_string_lossy().into_owned(),
            ))
        })
        .collect()
}

fn read_events(path: &Path) -> (Vec<Value>, Vec<String>) {
    let mut errors = Vec::new();
    let metadata = match sys_fs::lstat(path) {
        Ok(metadata) => metadata,
        Err(error) => {
            return (
                Vec::new(),
                vec![format!(
                    "events.jsonl was not created or cannot be inspected: {error}"
                )],
            );
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return (
            Vec::new(),
            vec!["events.jsonl is not a real regular file".into()],
        );
    }
    if metadata.len() >= MAX_EVENT_FILE_BYTES {
        return (
            Vec::new(),
            vec![format!(
                "events.jsonl reaches or exceeds the {}-byte evidence limit",
                MAX_EVENT_FILE_BYTES
            )],
        );
    }
    let file = match Open::new()
        .read(true)
        .no_follow()
        .open(path)
    {
        Ok(file) => file,
        Err(error) => {
            return (
                Vec::new(),
                vec![format!("cannot securely open events.jsonl: {error}")],
            );
        }
    };
    let opened = match sys_fs::fstat(&file) {
        Ok(opened) => opened,
        Err(error) => {
            return (
                Vec::new(),
                vec![format!("cannot inspect opened events.jsonl: {error}")],
            );
        }
    };
    if (opened.dev(), opened.ino(), opened.len())
        != (metadata.dev(), metadata.ino(), metadata.len())
    {
        return (
            Vec::new(),
            vec!["events.jsonl changed identity before it could be parsed".into()],
        );
    }

    let mut events = Vec::new();
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    let mut line_number = 0usize;
    loop {
        line.clear();
        line_number += 1;
        let bytes = match reader
            .by_ref()
            .take((MAX_EVENT_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
        {
            Ok(bytes) => bytes,
            Err(error) => {
                errors.push(format!("events.jsonl:{line_number}: read error: {error}"));
                break;
            }
        };
        if bytes == 0 {
            break;
        }
        if line.len() > MAX_EVENT_LINE_BYTES {
            errors.push(format!(
                "events.jsonl:{line_number}: line exceeds {MAX_EVENT_LINE_BYTES} bytes"
            ));
            break;
        }
        if !line.ends_with(b"\n") {
            errors.push(format!(
                "events.jsonl:{line_number}: event record is not newline-terminated"
            ));
        }
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let event: Value = match strict_json(&line, "native run event") {
            Ok(event) => event,
            Err(error) => {
                errors.push(format!("events.jsonl:{line_number}: {error}"));
                continue;
            }
        };
        let Some(object) = event.as_object() else {
            errors.push(format!(
                "events.jsonl:{line_number}: event is not a JSON object"
            ));
            continue;
        };
        if object.get("schema").and_then(Value::as_str) != Some(EVENT_SCHEMA) {
            errors.push(format!(
                "events.jsonl:{line_number}: unexpected schema {:?}",
                object.get("schema")
            ));
        }
        if object.get("event").and_then(Value::as_str).is_none() {
            errors.push(format!(
                "events.jsonl:{line_number}: event has no string event name"
            ));
        }
        for field in ["wall_time_ns", "monotonic_ns"] {
            if object.get(field).and_then(Value::as_u64).is_none() {
                errors.push(format!(
                    "events.jsonl:{line_number}: event has no nonnegative integer {field}"
                ));
            }
        }
        events.push(event);
    }
    let final_metadata = match sys_fs::fstat(&reader.into_inner()) {
        Ok(metadata) => metadata,
        Err(error) => {
            errors.push(format!("cannot restat events.jsonl after parsing: {error}"));
            return (events, errors);
        }
    };
    if (
        final_metadata.dev(),
        final_metadata.ino(),
        final_metadata.len(),
    ) != (opened.dev(), opened.ino(), opened.len())
    {
        errors.push("events.jsonl changed while it was being parsed".into());
    }
    (events, errors)
}

fn token_ids(value: Option<&Value>, label: &str, errors: &mut Vec<String>) -> Option<Vec<u64>> {
    let Some(value) = value else {
        errors.push(format!("{label} is missing"));
        return None;
    };
    let Some(values) = value.as_array() else {
        errors.push(format!("{label} is not an array"));
        return None;
    };
    let mut ids = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let Some(id) = value.as_u64() else {
            errors.push(format!("{label}[{index}] is not a nonnegative integer"));
            return None;
        };
        if id > u32::MAX as u64 {
            errors.push(format!(
                "{label}[{index}] exceeds the native u32 token-ID range"
            ));
            return None;
        }
        ids.push(id);
    }
    Some(ids)
}

fn nonnegative_duration(
    value: Option<&Value>,
    label: &str,
    errors: &mut Vec<String>,
) -> Option<u64> {
    match value.and_then(Value::as_u64) {
        Some(value) => Some(value),
        None => {
            errors.push(format!("{label} has no nonnegative integer duration_ns"));
            None
        }
    }
}

fn checked_metric_add(total: u64, value: u64, label: &str, errors: &mut Vec<String>) -> u64 {
    match total.checked_add(value) {
        Some(total) => total,
        None => {
            errors.push(format!("{label} overflowed u64"));
            total
        }
    }
}

fn checked_token_count_add(
    total: usize,
    value: usize,
    label: &str,
    errors: &mut Vec<String>,
) -> usize {
    match total.checked_add(value) {
        Some(total) => total,
        None => {
            errors.push(format!("{label} overflowed usize"));
            total
        }
    }
}

fn parse_structured_events(
    events: &[Value],
    warmup_steps: usize,
    mut errors: Vec<String>,
) -> ParsedRun {
    let positions = |name: &str| {
        events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| {
                (event.get("event").and_then(Value::as_str) == Some(name)).then_some(index)
            })
            .collect::<Vec<_>>()
    };
    let starts = positions("run_start");
    let prefills = positions("prefill_done");
    let steps = positions("decode_step");
    let ends = positions("run_end");
    let run_errors = positions("run_error");
    if starts.len() != 1 {
        errors.push(format!(
            "expected one run_start event, found {}",
            starts.len()
        ));
    }
    if prefills.len() != 1 {
        errors.push(format!(
            "expected one prefill_done event, found {}",
            prefills.len()
        ));
    }
    if ends.len() != 1 {
        errors.push(format!("expected one run_end event, found {}", ends.len()));
    }
    if !run_errors.is_empty() {
        errors.push(format!(
            "runner emitted {} run_error event(s)",
            run_errors.len()
        ));
    }
    if let (Some(&start), Some(&prefill), Some(&end)) =
        (starts.first(), prefills.first(), ends.first())
    {
        if start != 0 {
            errors.push("run_start is not the first structured event".into());
        }
        if !(start < prefill && prefill < end) {
            errors.push("run_start, prefill_done, and run_end are out of order".into());
        }
        if steps.iter().any(|step| *step <= prefill || *step >= end) {
            errors.push("decode_step event occurred outside the decode interval".into());
        }
        if end + 1 != events.len() {
            errors.push("run_end is not the final structured event".into());
        }
    }
    let monotonic = events
        .iter()
        .filter_map(|event| event.get("monotonic_ns").and_then(Value::as_u64))
        .collect::<Vec<_>>();
    if monotonic.windows(2).any(|pair| pair[1] < pair[0]) {
        errors.push("structured event monotonic_ns values moved backwards".into());
    }

    let start = starts.first().and_then(|index| events.get(*index));
    let prefill = prefills.first().and_then(|index| events.get(*index));
    let end = ends.first().and_then(|index| events.get(*index));
    let runner_config = start.and_then(|event| event.get("config")).cloned();
    if runner_config.as_ref().and_then(Value::as_object).is_none() {
        errors.push("run_start.config is not an object".into());
    }
    let eos_token_id = runner_config
        .as_ref()
        .and_then(|config| config.get("eos_token_id"))
        .and_then(Value::as_u64);
    match eos_token_id {
        Some(token_id) if token_id <= u32::MAX as u64 => {}
        Some(token_id) => errors.push(format!(
            "run_start.config.eos_token_id {token_id} exceeds the native u32 token-ID range"
        )),
        None => errors.push(
            "run_start.config.eos_token_id is missing or is not a nonnegative integer".into(),
        ),
    }
    if end
        .and_then(|event| event.get("runtime"))
        .and_then(Value::as_object)
        .is_none()
    {
        errors.push("run_end.runtime is not an object".into());
    }
    let input_token_ids = start.and_then(|event| {
        token_ids(
            event.get("input_token_ids"),
            "run_start.input_token_ids",
            &mut errors,
        )
    });
    let prefill_ns = prefill.and_then(|event| {
        nonnegative_duration(event.get("duration_ns"), "prefill_done", &mut errors)
    });
    let prefill_ids = prefill.and_then(|event| {
        token_ids(
            event.get("emitted_token_ids"),
            "prefill_done.emitted_token_ids",
            &mut errors,
        )
    });
    let mut decode_ns = 0u64;
    let mut decode_emitted_tokens = 0usize;
    let mut proposed_draft_tokens = 0u64;
    let mut accepted_draft_tokens = 0u64;
    let mut step_ids = Vec::with_capacity(steps.len());
    let mut decoded_steps = Vec::with_capacity(steps.len());
    for (ordinal, index) in steps.iter().copied().enumerate() {
        let event = events[index].clone();
        let expected_step = u64::try_from(ordinal).unwrap_or(u64::MAX);
        match event.get("step").and_then(Value::as_u64) {
            Some(step) if step == expected_step => {}
            Some(step) => errors.push(format!(
                "decode_step event {ordinal} declared step {step}, expected contiguous zero-based step {expected_step}"
            )),
            None => errors.push(format!(
                "decode_step event {ordinal} lacks a nonnegative integer step"
            )),
        }
        let duration = nonnegative_duration(
            event.get("duration_ns"),
            &format!("decode_step {}", ordinal + 1),
            &mut errors,
        )
        .unwrap_or(0);
        let ids = token_ids(
            event.get("emitted_token_ids"),
            &format!("decode_step {}.emitted_token_ids", ordinal + 1),
            &mut errors,
        )
        .unwrap_or_default();
        if ids.is_empty() {
            errors.push(format!(
                "decode_step {} accounted for no emitted token",
                ordinal + 1
            ));
        }
        let proposed = event.get("proposed_token_count").and_then(Value::as_u64);
        let accepted = event.get("accepted_token_count").and_then(Value::as_u64);
        match (proposed, accepted) {
            (Some(proposed), Some(accepted)) => {
                if accepted > proposed {
                    errors.push(format!(
                        "decode_step {} accepted {accepted} drafts after proposing only {proposed}",
                        ordinal + 1
                    ));
                }
                if accepted > ids.len() as u64 {
                    errors.push(format!(
                        "decode_step {} accepted {accepted} drafts but emitted only {} authoritative tokens",
                        ordinal + 1,
                        ids.len()
                    ));
                }
                match accepted.checked_add(1) {
                    Some(maximum_emitted) if ids.len() as u64 > maximum_emitted => {
                        errors.push(format!(
                            "decode_step {} emitted {} tokens but accounted for only {accepted} accepted drafts plus one target tail",
                            ordinal + 1,
                            ids.len()
                        ));
                    }
                    None => errors.push(format!(
                        "decode_step {} accepted draft count cannot represent its target tail",
                        ordinal + 1
                    )),
                    _ => {}
                }
                proposed_draft_tokens = checked_metric_add(
                    proposed_draft_tokens,
                    proposed,
                    "proposed draft token total",
                    &mut errors,
                );
                accepted_draft_tokens = checked_metric_add(
                    accepted_draft_tokens,
                    accepted,
                    "accepted draft token total",
                    &mut errors,
                );
            }
            _ => errors.push(format!(
                "decode_step {} lacks nonnegative proposed_token_count/accepted_token_count accounting",
                ordinal + 1
            )),
        }
        decode_ns = checked_metric_add(decode_ns, duration, "decode duration total", &mut errors);
        decode_emitted_tokens = checked_token_count_add(
            decode_emitted_tokens,
            ids.len(),
            "decode token count",
            &mut errors,
        );
        step_ids.push(ids);
        decoded_steps.push(event);
    }
    let mut steady_decode_tokens = 0usize;
    let mut steady_decode_ns = 0u64;
    let mut steady_proposed_draft_tokens = 0u64;
    let mut steady_accepted_draft_tokens = 0u64;
    for (ids, event) in step_ids.iter().zip(&decoded_steps).skip(warmup_steps) {
        steady_decode_tokens = checked_token_count_add(
            steady_decode_tokens,
            ids.len(),
            "steady decode token count",
            &mut errors,
        );
        if let Some(duration) = event.get("duration_ns").and_then(Value::as_u64) {
            steady_decode_ns = checked_metric_add(
                steady_decode_ns,
                duration,
                "steady decode duration total",
                &mut errors,
            );
        }
        if let Some(proposed) = event.get("proposed_token_count").and_then(Value::as_u64) {
            steady_proposed_draft_tokens = checked_metric_add(
                steady_proposed_draft_tokens,
                proposed,
                "steady proposed draft token total",
                &mut errors,
            );
        }
        if let Some(accepted) = event.get("accepted_token_count").and_then(Value::as_u64) {
            steady_accepted_draft_tokens = checked_metric_add(
                steady_accepted_draft_tokens,
                accepted,
                "steady accepted draft token total",
                &mut errors,
            );
        }
    }

    let emitted_token_ids = end.and_then(|event| {
        token_ids(
            event.get("emitted_token_ids"),
            "run_end.emitted_token_ids",
            &mut errors,
        )
    });
    let completion_token_ids = end.and_then(|event| {
        token_ids(
            event.get("completion_token_ids"),
            "run_end.completion_token_ids",
            &mut errors,
        )
    });
    let prompt_token_ids_at_end = end.and_then(|event| {
        token_ids(
            event.get("prompt_token_ids"),
            "run_end.prompt_token_ids",
            &mut errors,
        )
    });
    let completion_text = end
        .and_then(|event| event.get("completion_text"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if end.is_some() && completion_text.is_none() {
        errors.push("run_end.completion_text is not a string".into());
    }
    let runner_status = end
        .and_then(|event| event.get("status"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if end.is_some() && !matches!(runner_status.as_deref(), Some("ok" | "interrupted")) {
        errors.push(format!(
            "run_end.status is missing or invalid: {runner_status:?}"
        ));
    }
    let stop_reason = end
        .and_then(|event| event.get("stop_reason"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if end.is_some()
        && !matches!(
            stop_reason.as_deref(),
            Some("eos" | "max_new" | "context_full" | "interrupted")
        )
    {
        errors.push(format!(
            "run_end.stop_reason is missing or invalid: {stop_reason:?}"
        ));
    }
    match (runner_status.as_deref(), stop_reason.as_deref()) {
        (Some("interrupted"), Some("interrupted"))
        | (Some("ok"), Some("eos" | "max_new" | "context_full")) => {}
        (Some(status), Some(stop)) => errors.push(format!(
            "run_end status {status:?} is inconsistent with stop_reason {stop:?}"
        )),
        _ => {}
    }
    if let Some(ids) = &prefill_ids {
        match runner_status.as_deref() {
            Some("interrupted") if ids.len() <= 1 => {}
            _ if ids.len() == 1 => {}
            _ => errors.push(format!(
                "prefill_done accounted for {} emitted tokens; expected exactly one, or zero for an interrupted prefill",
                ids.len()
            )),
        }
        if ids.is_empty() && !steps.is_empty() {
            errors.push(
                "interrupted prefill without an authoritative first token may not have decode_step events"
                    .into(),
            );
        }
    }
    let inference_ns = end
        .and_then(|event| nonnegative_duration(event.get("duration_ns"), "run_end", &mut errors));

    let mut accounted = prefill_ids.unwrap_or_default();
    for ids in &step_ids {
        accounted.extend_from_slice(ids);
    }
    if let Some(emitted) = &emitted_token_ids {
        if &accounted != emitted {
            errors.push(format!(
                "transaction token accounting differs from run_end.emitted_token_ids: accounted {accounted:?}, end {emitted:?}"
            ));
        }
    }
    if let (Some(completion), Some(emitted)) = (&completion_token_ids, &emitted_token_ids) {
        if completion.len() > emitted.len() || emitted[..completion.len()] != completion[..] {
            errors.push("run_end.completion_token_ids is not a prefix of emitted_token_ids".into());
        } else {
            let omitted = emitted.len() - completion.len();
            match stop_reason.as_deref() {
                Some("eos") => {
                    if omitted != 1 {
                        errors.push(format!(
                            "EOS stop must omit exactly one terminal token from completion_token_ids, omitted {omitted}"
                        ));
                    } else if emitted.last().copied() != eos_token_id {
                        errors.push(format!(
                            "EOS stop omitted token {:?}, but run_start.config.eos_token_id is {eos_token_id:?}",
                            emitted.last()
                        ));
                    }
                }
                Some("interrupted") if omitted == 1 && emitted.last().copied() == eos_token_id => {}
                Some("interrupted") if omitted != 0 => errors.push(format!(
                    "interrupted stop omitted {omitted} tokens that are not one terminal EOS"
                )),
                Some("max_new" | "context_full") if omitted != 0 => errors.push(format!(
                    "{stop_reason:?} stop may not omit emitted tokens from completion_token_ids"
                )),
                _ => {}
            }
        }
    }
    if let (Some(start_ids), Some(end_ids)) = (&input_token_ids, &prompt_token_ids_at_end)
        && start_ids != end_ids
    {
        errors.push(format!(
            "run_start and run_end prompt token accounting differs: start {start_ids:?}, end {end_ids:?}"
        ));
    }

    ParsedRun {
        source: "structured_events",
        parse_errors: errors,
        runner_config,
        runner_runtime: end.and_then(|event| event.get("runtime")).cloned(),
        eos_token_id,
        input_token_ids,
        prompt_token_ids_at_end,
        prefill_ns,
        prefill_s: prefill_ns.map(ns_to_seconds),
        decode_steps: decoded_steps,
        decode_step_count: steps.len(),
        decode_ns,
        decode_emitted_tokens,
        decode_tps: rate(decode_emitted_tokens, decode_ns),
        steady_warmup_steps_dropped: warmup_steps.min(steps.len()),
        steady_decode_ns,
        steady_decode_tokens,
        steady_tps: rate(steady_decode_tokens, steady_decode_ns),
        steady_s_per_token: reciprocal(rate(steady_decode_tokens, steady_decode_ns)),
        emitted_token_ids,
        completion_token_ids,
        completion_text,
        runner_status,
        stop_reason,
        proposed_draft_tokens,
        accepted_draft_tokens,
        draft_acceptance_rate: rate_u64(accepted_draft_tokens, proposed_draft_tokens),
        steady_proposed_draft_tokens,
        steady_accepted_draft_tokens,
        steady_draft_acceptance_rate: rate_u64(
            steady_accepted_draft_tokens,
            steady_proposed_draft_tokens,
        ),
        inference_ns,
        inference_s: inference_ns.map(ns_to_seconds),
    }
}

fn ns_to_seconds(value: u64) -> f64 {
    value as f64 / 1_000_000_000.0
}

fn rate(count: usize, duration_ns: u64) -> Option<f64> {
    (count > 0 && duration_ns > 0).then_some(count as f64 * 1_000_000_000.0 / duration_ns as f64)
}

fn rate_u64(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn reciprocal(value: Option<f64>) -> Option<f64> {
    value.filter(|value| *value > 0.0).map(|value| 1.0 / value)
}

fn validate_options(options: &BenchmarkOptions) -> Result<(PathBuf, PathBuf)> {
    if options.repetitions == 0 {
        return Err(DeltafinError::new(
            "benchmark repetitions must be at least one",
        ));
    }
    if options.max_new_tokens == 0 {
        return Err(DeltafinError::new(
            "benchmark max-new token count must be at least one",
        ));
    }
    if options.timeout.is_zero() {
        return Err(DeltafinError::new("benchmark timeout must be positive"));
    }
    if options.arms.is_empty() {
        return Err(DeltafinError::new(
            "benchmark requires at least one configuration arm",
        ));
    }
    let mut names = BTreeSet::new();
    for arm in &options.arms {
        if arm.name.trim().is_empty() {
            return Err(DeltafinError::new(
                "benchmark configuration names may not be empty",
            ));
        }
        if !names.insert(arm.name.clone()) {
            return Err(DeltafinError::new(format!(
                "benchmark configuration name {:?} is duplicated",
                arm.name
            )));
        }
        parse_environment_delta(&arm.environment_spec)?;
    }
    let repository_root = crate::sys::fs::canonicalize(&options.repository_root).map_err(|error| {
        DeltafinError::new(format!(
            "resolve benchmark repository/model root {}: {error}",
            options.repository_root.display()
        ))
    })?;
    let root_metadata = sys_fs::lstat(&repository_root).map_err(|error| {
        DeltafinError::new(format!(
            "inspect benchmark repository/model root {}: {error}",
            repository_root.display()
        ))
    })?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(DeltafinError::new(format!(
            "benchmark repository/model root is not a real directory: {}",
            repository_root.display()
        )));
    }
    let runner = verify_native_executable(&options.runner)?;
    Ok((repository_root, runner))
}

fn verify_native_executable(path: &Path) -> Result<PathBuf> {
    let (path, _, _) = open_verified_native_executable(path)?;
    Ok(path)
}

fn open_verified_native_executable(path: &Path) -> Result<(PathBuf, File, NativeFileIdentity)> {
    let supplied_metadata = sys_fs::lstat(path).map_err(|error| {
        DeltafinError::new(format!(
            "inspect compiled Deltafin benchmark runner {}: {error}",
            path.display()
        ))
    })?;
    if supplied_metadata.file_type().is_symlink() {
        return Err(DeltafinError::new(format!(
            "benchmark runner path may not be a symbolic link: {}",
            path.display()
        )));
    }
    let path = crate::sys::fs::canonicalize(path).map_err(|error| {
        DeltafinError::new(format!(
            "resolve compiled Deltafin benchmark runner {}: {error}",
            path.display()
        ))
    })?;
    let metadata = sys_fs::lstat(&path).map_err(|error| {
        DeltafinError::new(format!(
            "inspect compiled Deltafin benchmark runner {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(DeltafinError::new(format!(
            "benchmark runner is not a real regular file: {}",
            path.display()
        )));
    }
    if metadata.mode() & 0o111 == 0 {
        return Err(DeltafinError::new(format!(
            "benchmark runner is not executable: {}",
            path.display()
        )));
    }
    let mut file = Open::new()
        .read(true)
        .no_follow()
        .open(&path)
        .map_err(|error| {
            DeltafinError::new(format!(
                "open compiled Deltafin benchmark runner {}: {error}",
                path.display()
            ))
        })?;
    let opened_metadata = sys_fs::fstat(&file).map_err(|error| {
        DeltafinError::new(format!(
            "inspect opened Deltafin benchmark runner {}: {error}",
            path.display()
        ))
    })?;
    let path_identity = native_file_identity(&metadata);
    let opened_identity = native_file_identity(&opened_metadata);
    if path_identity != opened_identity {
        return Err(DeltafinError::new(format!(
            "benchmark runner changed identity while it was being opened: {}",
            path.display()
        )));
    }
    let mut magic = [0u8; 8];
    let count = file.read(&mut magic).map_err(|error| {
        DeltafinError::new(format!(
            "read compiled Deltafin benchmark runner {}: {error}",
            path.display()
        ))
    })?;
    if !native_binary_magic(&magic[..count]) {
        return Err(DeltafinError::new(format!(
            "benchmark runner is not an ELF, Mach-O, or PE compiled executable (scripts and Python runners are forbidden): {}",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        DeltafinError::new(format!(
            "rewind compiled Deltafin benchmark runner {}: {error}",
            path.display()
        ))
    })?;
    Ok((path, file, opened_identity))
}

fn native_file_identity(metadata: &sys_fs::Stat) -> NativeFileIdentity {
    NativeFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        size_bytes: metadata.len(),
        mode: metadata.mode(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

fn pin_native_executable(runner: &Path, output_dir: &Path) -> Result<PinnedRunner> {
    let (original_path, mut source, source_identity) = open_verified_native_executable(runner)?;
    let executable_path = output_dir.join("runner.pinned");
    let mut destination = secure_create_new(&executable_path, 0o500)?;
    let mut digest = DigestState::new();
    let mut copied = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = match source.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(DeltafinError::new(format!(
                    "read benchmark runner while pinning {}: {error}",
                    original_path.display()
                )));
            }
        };
        destination.write_all(&buffer[..count]).map_err(|error| {
            DeltafinError::new(format!(
                "write pinned benchmark runner {}: {error}",
                executable_path.display()
            ))
        })?;
        digest.update(&buffer[..count]);
        copied = copied.checked_add(count as u64).ok_or_else(|| {
            DeltafinError::new("pinned benchmark runner byte count overflowed u64")
        })?;
    }
    let final_source_identity = native_file_identity(&sys_fs::fstat(&source).map_err(|error| {
        DeltafinError::new(format!(
            "reinspect benchmark runner after pinning {}: {error}",
            original_path.display()
        ))
    })?);
    if final_source_identity != source_identity || copied != source_identity.size_bytes {
        return Err(DeltafinError::new(format!(
            "benchmark runner changed while its private copy was being pinned: {}",
            original_path.display()
        )));
    }
    #[cfg(unix)]
    destination
        .set_permissions(fs::Permissions::from_mode(0o500))
        .map_err(|error| {
            DeltafinError::new(format!(
                "set pinned benchmark runner permissions {}: {error}",
                executable_path.display()
            ))
        })?;
    destination.sync_all().map_err(|error| {
        DeltafinError::new(format!(
            "fsync pinned benchmark runner {}: {error}",
            executable_path.display()
        ))
    })?;
    fsync_directory(output_dir)?;

    let sha256 = hex_digest(&digest.finalize());
    let (_, mut pinned_file, pinned_identity) = open_verified_native_executable(&executable_path)?;
    if pinned_identity.size_bytes != copied {
        return Err(DeltafinError::new(format!(
            "pinned benchmark runner size changed before verification: {}",
            executable_path.display()
        )));
    }
    let (verified_sha256, verified_size) = digest_open_file(&mut pinned_file, &executable_path)?;
    if verified_size != copied || verified_sha256 != sha256 {
        return Err(DeltafinError::new(format!(
            "pinned benchmark runner digest changed before campaign start: {}",
            executable_path.display()
        )));
    }
    Ok(PinnedRunner {
        original_path,
        executable_path,
        sha256,
        size_bytes: copied,
        source_identity,
    })
}

fn verify_pinned_runner(runner: &PinnedRunner) -> Result<()> {
    let (path, mut file, identity) = open_verified_native_executable(&runner.executable_path)?;
    if path != runner.executable_path || identity.size_bytes != runner.size_bytes {
        return Err(DeltafinError::new(format!(
            "pinned benchmark runner identity changed: {}",
            runner.executable_path.display()
        )));
    }
    let (sha256, size_bytes) = digest_open_file(&mut file, &path)?;
    if size_bytes != runner.size_bytes || sha256 != runner.sha256 {
        return Err(DeltafinError::new(format!(
            "pinned benchmark runner digest changed: {}",
            runner.executable_path.display()
        )));
    }
    audit_benchmark_executable(&path)
}

fn audit_benchmark_executable(path: &Path) -> Result<()> {
    crate::loader_audit::audit_loader_closure(
        path,
        &crate::loader_audit::LoaderAuditPolicy::bootstrap(),
    )
    .map(|_| ())
}

fn digest_open_file(file: &mut File, path: &Path) -> Result<(String, u64)> {
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        DeltafinError::new(format!(
            "rewind native executable {}: {error}",
            path.display()
        ))
    })?;
    let mut digest = DigestState::new();
    let mut total = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(DeltafinError::new(format!(
                    "hash native executable {}: {error}",
                    path.display()
                )));
            }
        };
        digest.update(&buffer[..count]);
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| DeltafinError::new("native executable size overflowed u64"))?;
    }
    Ok((hex_digest(&digest.finalize()), total))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}

fn native_binary_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x7fELF")
        || bytes.starts_with(b"MZ")
        || matches!(
            bytes.get(..4),
            Some(
                [0xfe, 0xed, 0xfa, 0xce]
                    | [0xfe, 0xed, 0xfa, 0xcf]
                    | [0xce, 0xfa, 0xed, 0xfe]
                    | [0xcf, 0xfa, 0xed, 0xfe]
                    | [0xca, 0xfe, 0xba, 0xbe]
                    | [0xbe, 0xba, 0xfe, 0xca]
                    | [0xca, 0xfe, 0xba, 0xbf]
                    | [0xbf, 0xba, 0xfe, 0xca]
            )
        )
}

fn make_output_dir(repository_root: &Path, requested: Option<&Path>) -> Result<PathBuf> {
    let requested_path = match requested {
        Some(requested) if requested.is_absolute() => requested.to_path_buf(),
        Some(requested) => repository_root.join(requested),
        None => repository_root.join("bench-results").join(format!(
            "{}-{}",
            compact_utc_now()?,
            std::process::id()
        )),
    };
    let path = normalize_output_path(&requested_path)?;
    let parent = path.parent().ok_or_else(|| {
        DeltafinError::new(format!(
            "benchmark output path has no parent: {}",
            path.display()
        ))
    })?;
    create_real_directory_tree(parent)?;
    sys_fs::create_private_directory(&path)
        .map_err(|error| {
            DeltafinError::new(format!(
                "exclusively create benchmark evidence directory {}: {error}",
                path.display()
            ))
        })?;
    fsync_directory(parent)?;
    crate::sys::fs::canonicalize(&path).map_err(|error| {
        DeltafinError::new(format!(
            "resolve benchmark evidence directory {}: {error}",
            path.display()
        ))
    })
}

fn normalize_output_path(path: &Path) -> Result<PathBuf> {
    let mut cursor = path;
    let mut missing = Vec::new();
    while !cursor.exists() {
        if sys_fs::lstat(cursor).is_ok() {
            return Err(DeltafinError::new(format!(
                "benchmark output path contains a dangling symlink: {}",
                cursor.display()
            )));
        }
        let name = cursor.file_name().ok_or_else(|| {
            DeltafinError::new(format!(
                "benchmark output path has no existing ancestor: {}",
                path.display()
            ))
        })?;
        missing.push(name.to_owned());
        cursor = cursor.parent().ok_or_else(|| {
            DeltafinError::new(format!(
                "benchmark output path has no parent: {}",
                path.display()
            ))
        })?;
    }
    let mut normalized = crate::sys::fs::canonicalize(cursor).map_err(|error| {
        DeltafinError::new(format!(
            "resolve existing benchmark output ancestor {}: {error}",
            cursor.display()
        ))
    })?;
    for component in missing.into_iter().rev() {
        normalized.push(component);
    }
    Ok(normalized)
}

fn create_real_directory_tree(path: &Path) -> Result<()> {
    if path.exists() {
        return validate_real_directory_chain(path);
    }
    let parent = path.parent().ok_or_else(|| {
        DeltafinError::new(format!("directory has no parent: {}", path.display()))
    })?;
    create_real_directory_tree(parent)?;
    sys_fs::create_private_directory(path)
        .map_err(|error| {
            DeltafinError::new(format!("create directory {}: {error}", path.display()))
        })?;
    fsync_directory(parent)?;
    validate_real_directory_chain(path)
}

fn validate_real_directory_chain(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        let metadata = sys_fs::lstat(ancestor).map_err(|error| {
            DeltafinError::new(format!("inspect directory {}: {error}", ancestor.display()))
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(DeltafinError::new(format!(
                "benchmark output path contains a non-directory or symlink component: {}",
                ancestor.display()
            )));
        }
    }
    Ok(())
}

fn create_run_directory(path: &Path) -> Result<()> {
    sys_fs::create_private_directory(path)
        .map_err(|error| {
            DeltafinError::new(format!(
                "exclusively create benchmark run directory {}: {error}",
                path.display()
            ))
        })?;
    if let Some(parent) = path.parent() {
        fsync_directory(parent)?;
    }
    Ok(())
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonicalize_json).collect()),
        Value::Object(object) => {
            let sorted = object.iter().collect::<BTreeMap<_, _>>();
            let mut canonical = Map::with_capacity(sorted.len());
            for (key, value) in sorted {
                canonical.insert(key.clone(), canonicalize_json(value));
            }
            Value::Object(canonical)
        }
        _ => value.clone(),
    }
}

fn write_json_exclusive(path: &Path, value: &Value) -> Result<()> {
    let mut file = secure_create_new(path, 0o600)?;
    serde_json::to_writer_pretty(&mut file, &canonicalize_json(value)).map_err(|error| {
        DeltafinError::new(format!(
            "serialize benchmark JSON {}: {error}",
            path.display()
        ))
    })?;
    file.write_all(b"\n").map_err(|error| {
        DeltafinError::new(format!("write benchmark JSON {}: {error}", path.display()))
    })?;
    file.sync_all().map_err(|error| {
        DeltafinError::new(format!("fsync benchmark JSON {}: {error}", path.display()))
    })?;
    if let Some(parent) = path.parent() {
        fsync_directory(parent)?;
    }
    Ok(())
}

fn append_jsonl(file: &mut File, path: &Path, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *file, &canonicalize_json(value)).map_err(|error| {
        DeltafinError::new(format!(
            "serialize benchmark JSONL record {}: {error}",
            path.display()
        ))
    })?;
    file.write_all(b"\n").map_err(|error| {
        DeltafinError::new(format!("write benchmark JSONL {}: {error}", path.display()))
    })?;
    file.sync_all().map_err(|error| {
        DeltafinError::new(format!("fsync benchmark JSONL {}: {error}", path.display()))
    })
}

fn utc_now() -> Result<String> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DeltafinError::new("system clock precedes the Unix epoch"))?;
    Ok(format_utc(
        duration.as_secs(),
        duration.subsec_nanos(),
        false,
    ))
}

fn compact_utc_now() -> Result<String> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DeltafinError::new("system clock precedes the Unix epoch"))?;
    Ok(format_utc(
        duration.as_secs(),
        duration.subsec_nanos(),
        true,
    ))
}

fn format_utc(seconds: u64, nanoseconds: u32, compact: bool) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = seconds_of_day % 3_600 / 60;
    let second = seconds_of_day % 60;
    let micros = nanoseconds / 1_000;
    if compact {
        format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}.{micros:06}Z")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{micros:06}Z")
    }
}

// Howard Hinnant's public-domain civil calendar conversion, with day zero at
// 1970-01-01.
fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn wait_for_runner(
    child: &mut Child,
    timeout: Duration,
    events_path: &Path,
    event_limit: u64,
    pipe_failure: &AtomicBool,
) -> io::Result<RunnerWait> {
    let started = Instant::now();
    loop {
        if pipe_failure.load(Ordering::Acquire) {
            terminate_process_group(child);
            return child.wait().map(|status| {
                RunnerWait {
                status,
                timed_out: false,
                live_limit_error: Some(
                    "runner stdout or stderr exceeded its bounded capture or could not be persisted"
                        .into(),
                ),
            }
            });
        }
        if let Some(error) = live_event_stream_error(events_path, event_limit)? {
            terminate_process_group(child);
            return child.wait().map(|status| RunnerWait {
                status,
                timed_out: false,
                live_limit_error: Some(error),
            });
        }
        if let Some(status) = child.try_wait()? {
            return Ok(RunnerWait {
                status,
                timed_out: false,
                live_limit_error: live_event_stream_error(events_path, event_limit)?,
            });
        }
        let elapsed = started.elapsed();
        if elapsed >= timeout {
            terminate_process_group(child);
            return child.wait().map(|status| RunnerWait {
                status,
                timed_out: true,
                live_limit_error: None,
            });
        }
        thread::sleep(POLL_QUANTUM.min(timeout.saturating_sub(elapsed)));
    }
}

fn live_event_stream_error(path: &Path, limit: u64) -> io::Result<Option<String>> {
    let metadata = match sys_fs::lstat(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Ok(Some(format!(
            "runner event stream became a symlink or non-regular file: {}",
            path.display()
        )));
    }
    if metadata.len() >= limit {
        return Ok(Some(format!(
            "runner event stream reached the {limit}-byte live capture limit"
        )));
    }
    Ok(None)
}

#[cfg(windows)]
fn terminate_process_group(child: &mut Child) {
    let _ = child.kill();
}

/// The signal that ended a child, where the platform has signals.
#[cfg(unix)]
fn exit_signal(status: &ExitStatus) -> Option<i32> {
    status.signal()
}

#[cfg(windows)]
fn exit_signal(_status: &ExitStatus) -> Option<i32> {
    None
}

#[cfg(unix)]
fn terminate_process_group(child: &mut Child) {
    let process_id = child.id();
    if process_id <= i32::MAX as u32 {
        // The child is placed in its own process group before exec. Killing
        // the group prevents a timed-out runner from leaving CUDA helpers or
        // other descendants alive after the benchmark declares failure.
        // SAFETY: kill takes an integer process-group ID and no pointers.
        let _ = unsafe { kill(-(process_id as i32), SIGKILL) };
    }
    let _ = child.kill();
}

#[cfg(unix)]
const SIGKILL: i32 = 9;

#[cfg(unix)]
unsafe extern "C" {
    fn kill(process_or_group: i32, signal: i32) -> i32;
}

#[allow(clippy::too_many_arguments)]
fn execute_runner(
    runner: &Path,
    repository_root: &Path,
    arguments: &[OsString],
    environment_delta: &BTreeMap<String, String>,
    stdout_path: &Path,
    stderr_path: &Path,
    events_path: &Path,
    timeout: Duration,
) -> Result<ChildOutcome> {
    execute_runner_with_limits(
        runner,
        repository_root,
        arguments,
        environment_delta,
        stdout_path,
        stderr_path,
        events_path,
        timeout,
        PRODUCTION_CAPTURE_LIMITS,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_runner_with_limits(
    runner: &Path,
    repository_root: &Path,
    arguments: &[OsString],
    environment_delta: &BTreeMap<String, String>,
    stdout_path: &Path,
    stderr_path: &Path,
    events_path: &Path,
    timeout: Duration,
    limits: CaptureLimits,
) -> Result<ChildOutcome> {
    let stdout_file = secure_create_new(stdout_path, 0o600)?;
    let stderr_file = secure_create_new(stderr_path, 0o600)?;
    let started = Instant::now();
    let mut command = Command::new(runner);
    command
        .args(arguments)
        .current_dir(repository_root)
        .envs(environment_delta)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    command.process_group(0);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            sync_capture_file(stdout_file, stdout_path, "stdout")?;
            sync_capture_file(stderr_file, stderr_path, "stderr")?;
            if let Some(parent) = stdout_path.parent() {
                fsync_directory(parent)?;
            }
            return Ok(ChildOutcome {
                status: None,
                timed_out: false,
                spawn_error: Some(format!("spawn compiled Deltafin benchmark runner: {error}")),
                capture_errors: Vec::new(),
                stdout_bytes: 0,
                stderr_bytes: 0,
                wall_ns: elapsed_ns_saturating(started),
            });
        }
    };
    let stdout_pipe = child.stdout.take().ok_or_else(|| {
        terminate_process_group(&mut child);
        let _ = child.wait();
        DeltafinError::new("compiled benchmark runner did not expose its stdout pipe")
    })?;
    let stderr_pipe = child.stderr.take().ok_or_else(|| {
        terminate_process_group(&mut child);
        let _ = child.wait();
        DeltafinError::new("compiled benchmark runner did not expose its stderr pipe")
    })?;
    let pipe_failure = Arc::new(AtomicBool::new(false));
    let stdout_failure = Arc::clone(&pipe_failure);
    let stderr_failure = Arc::clone(&pipe_failure);
    let stdout_display = stdout_path.to_path_buf();
    let stderr_display = stderr_path.to_path_buf();
    let stdout_thread = thread::spawn(move || {
        capture_pipe(
            stdout_pipe,
            stdout_file,
            &stdout_display,
            "stdout",
            limits.stdout_bytes,
            &stdout_failure,
        )
    });
    let stderr_thread = thread::spawn(move || {
        capture_pipe(
            stderr_pipe,
            stderr_file,
            &stderr_display,
            "stderr",
            limits.stderr_bytes,
            &stderr_failure,
        )
    });
    let (status, timed_out, spawn_error, live_limit_error) = match wait_for_runner(
        &mut child,
        timeout,
        events_path,
        limits.event_bytes,
        &pipe_failure,
    ) {
        Ok(wait) => (
            Some(wait.status),
            wait.timed_out,
            None,
            wait.live_limit_error,
        ),
        Err(error) => {
            terminate_process_group(&mut child);
            let _ = child.wait();
            (
                None,
                false,
                Some(format!("wait for benchmark runner: {error}")),
                None,
            )
        }
    };
    let stdout_capture = join_capture(stdout_thread, "stdout");
    let stderr_capture = join_capture(stderr_thread, "stderr");
    let mut capture_errors = Vec::new();
    if let Some(error) = live_limit_error {
        capture_errors.push(error);
    }
    for (label, capture, limit) in [
        ("stdout", &stdout_capture, limits.stdout_bytes),
        ("stderr", &stderr_capture, limits.stderr_bytes),
    ] {
        if capture.overflowed {
            capture_errors.push(format!(
                "runner {label} exceeded the {limit}-byte capture limit"
            ));
        }
        if let Some(error) = &capture.io_error {
            capture_errors.push(error.clone());
        }
    }
    capture_errors.sort();
    capture_errors.dedup();
    if let Some(parent) = stdout_path.parent() {
        fsync_directory(parent)?;
    }
    Ok(ChildOutcome {
        status,
        timed_out,
        spawn_error,
        capture_errors,
        stdout_bytes: stdout_capture.bytes_written,
        stderr_bytes: stderr_capture.bytes_written,
        wall_ns: elapsed_ns_saturating(started),
    })
}

fn capture_pipe<R: Read>(
    mut source: R,
    mut destination: File,
    path: &Path,
    label: &str,
    limit: u64,
    failure: &AtomicBool,
) -> CapturedPipe {
    let mut bytes_written = 0u64;
    let mut overflowed = false;
    let mut io_error = None;
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let count = match source.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                io_error = Some(format!("read runner {label} pipe: {error}"));
                failure.store(true, Ordering::Release);
                break;
            }
        };
        let remaining = limit - bytes_written;
        let keep = usize::try_from(remaining.min(count as u64)).unwrap_or(count);
        if keep > 0
            && let Err(error) = destination.write_all(&buffer[..keep])
        {
            io_error = Some(format!(
                "write captured benchmark {label} {}: {error}",
                path.display()
            ));
            failure.store(true, Ordering::Release);
        }
        bytes_written += keep as u64;
        if keep < count {
            overflowed = true;
            failure.store(true, Ordering::Release);
        }
    }
    if let Err(error) = destination.sync_all()
        && io_error.is_none()
    {
        io_error = Some(format!(
            "fsync captured benchmark {label} {}: {error}",
            path.display()
        ));
        failure.store(true, Ordering::Release);
    }
    CapturedPipe {
        bytes_written,
        overflowed,
        io_error,
    }
}

fn join_capture(handle: thread::JoinHandle<CapturedPipe>, label: &str) -> CapturedPipe {
    handle.join().unwrap_or_else(|_| CapturedPipe {
        bytes_written: 0,
        overflowed: false,
        io_error: Some(format!("runner {label} capture thread panicked")),
    })
}

fn sync_capture_file(file: File, path: &Path, label: &str) -> Result<()> {
    file.sync_all().map_err(|error| {
        DeltafinError::new(format!(
            "fsync captured benchmark {label} {}: {error}",
            path.display()
        ))
    })
}

fn elapsed_ns_saturating(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn stderr_tail(path: &Path, maximum_bytes: u64) -> String {
    let Ok(mut file) = File::open(path) else {
        return String::new();
    };
    let Ok(length) = sys_fs::fstat(&file).map(|metadata| metadata.len()) else {
        return String::new();
    };
    let start = length.saturating_sub(maximum_bytes);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return String::new();
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return String::new();
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn lightweight_state(path: &Path) -> Value {
    let captured_at_utc = utc_now().unwrap_or_else(|_| "unavailable".into());
    let wall_time_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
    let logical_cpus = thread::available_parallelism()
        .ok()
        .map(|value| value.get());
    let disk = filesystem_state(path);
    let load_average = load_average();
    let mut state = json!({
        "captured_at_utc": captured_at_utc,
        "time_ns": wall_time_ns,
        "cpu_count_logical": logical_cpus,
        "disk": disk,
        "load_average_1m_5m_15m": load_average,
    });
    if cfg!(target_os = "linux") {
        state["memory"] = linux_memory_state();
    }
    state
}

#[cfg(windows)]
fn filesystem_state(path: &Path) -> Option<Value> {
    let canonical = crate::sys::fs::canonicalize(path).ok()?;
    let directory = sys_fs::Open::new().read(true).directory().open(&canonical).ok()?;
    let free = sys_fs::available_space(&directory, &canonical).ok()?;
    Some(json!({ "free_bytes": free }))
}

#[cfg(unix)]
fn filesystem_state(path: &Path) -> Option<Value> {
    let canonical = crate::sys::fs::canonicalize(path).ok()?;
    let path = CString::new(canonical.as_os_str().as_bytes()).ok()?;
    // SAFETY: `path` is a live NUL-terminated string and `statvfs` initializes
    // the complete output structure before returning success.
    let filesystem = unsafe {
        let mut value = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        if libc::statvfs(path.as_ptr(), value.as_mut_ptr()) != 0 {
            return None;
        }
        value.assume_init()
    };
    let fragment_bytes = if filesystem.f_frsize == 0 {
        filesystem.f_bsize
    } else {
        filesystem.f_frsize
    } as u64;
    let total = (filesystem.f_blocks as u64).checked_mul(fragment_bytes)?;
    let free = (filesystem.f_bfree as u64).checked_mul(fragment_bytes)?;
    let available = (filesystem.f_bavail as u64).checked_mul(fragment_bytes)?;
    Some(json!({
        "block_bytes": fragment_bytes,
        "total_bytes": total,
        "used_bytes": total.saturating_sub(free),
        "free_bytes": available,
    }))
}

fn load_average() -> Option<Value> {
    #[cfg(target_os = "linux")]
    {
        let text = fs::read_to_string("/proc/loadavg").ok()?;
        let values = text
            .split_whitespace()
            .take(3)
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()?;
        return Some(json!(values));
    }
    #[cfg(target_os = "macos")]
    {
        let mut values = [0.0_f64; 3];
        // SAFETY: the output buffer contains three writable doubles and
        // `getloadavg` writes no more than the supplied element count.
        let count = unsafe { libc::getloadavg(values.as_mut_ptr(), values.len() as libc::c_int) };
        (count == values.len() as libc::c_int).then(|| json!(values))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn linux_memory_state() -> Value {
    let allowlist = [
        "MemTotal",
        "MemAvailable",
        "SwapTotal",
        "SwapFree",
        "HugePages_Total",
        "HugePages_Free",
    ];
    let mut values = BTreeMap::new();
    if let Ok(text) = fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            let Some((key, raw)) = line.split_once(':') else {
                continue;
            };
            if !allowlist.contains(&key) {
                continue;
            }
            let fields = raw.split_whitespace().collect::<Vec<_>>();
            if let Some(value) = fields.first().and_then(|value| value.parse::<u64>().ok()) {
                let bytes = if fields.get(1) == Some(&"kB") {
                    value.saturating_mul(1024)
                } else {
                    value
                };
                values.insert(format!("{}_bytes", key.to_ascii_lowercase()), bytes);
            }
        }
    }
    serde_json::to_value(values).unwrap_or(Value::Null)
}

fn direct_repository_state(repository_root: &Path) -> Value {
    match read_direct_repository_state(repository_root) {
        Ok(state) => state,
        Err(error) => json!({
            "source": "bounded_direct_git_control_files_v1",
            "available": false,
            "error": error.to_string(),
            "commit": Value::Null,
            "dirty": Value::Null,
            "changed_entry_count": Value::Null,
        }),
    }
}

fn read_direct_repository_state(repository_root: &Path) -> Result<Value> {
    let Some(git_directory) = resolve_git_directory(repository_root)? else {
        return Ok(json!({
            "source": "bounded_direct_git_control_files_v1",
            "available": false,
            "reason": "no .git control directory at benchmark root",
            "commit": Value::Null,
            "dirty": Value::Null,
            "changed_entry_count": Value::Null,
        }));
    };
    let common_directory = resolve_git_common_directory(&git_directory)?;
    let head_bytes = read_bounded_git_file(
        &git_directory.join("HEAD"),
        MAX_GIT_POINTER_BYTES,
        "Git HEAD",
    )?
    .ok_or_else(|| DeltafinError::new("Git control directory has no HEAD file"))?;
    let head = bounded_git_line(&head_bytes, "Git HEAD")?;
    let (commit, symbolic_head) = if let Some(reference) = head.strip_prefix("ref: ") {
        validate_git_reference(reference)?;
        (
            resolve_git_reference(&git_directory, &common_directory, reference)?,
            true,
        )
    } else {
        (Some(validate_git_object_id(head)?.to_owned()), false)
    };
    let index = bounded_git_file_digest(
        &git_directory.join("index"),
        MAX_GIT_INDEX_BYTES,
        "Git index",
    )?;

    // A faithful `git status` requires interpreting the complete index and
    // comparing every tracked and untracked worktree path. Pretending an
    // index timestamp is a dirty bit would make benchmark evidence weaker.
    // The exact pinned executable digest below is the authoritative build
    // identity; these direct control-file fields are bounded context only.
    Ok(json!({
        "source": "bounded_direct_git_control_files_v1",
        "available": true,
        "commit": commit,
        "symbolic_head": symbolic_head,
        "index": index,
        "dirty": Value::Null,
        "changed_entry_count": Value::Null,
        "worktree_comparison": "not_performed_without_invoking_or_emulating_git_status",
    }))
}

fn resolve_git_directory(repository_root: &Path) -> Result<Option<PathBuf>> {
    let dot_git = repository_root.join(".git");
    let metadata = match sys_fs::lstat(&dot_git) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(DeltafinError::new(format!(
                "inspect Git control entry {}: {error}",
                dot_git.display()
            )));
        }
    };
    if metadata.file_type().is_symlink() {
        return Err(DeltafinError::new(format!(
            "Git control entry may not be a symbolic link: {}",
            dot_git.display()
        )));
    }
    if metadata.is_dir() {
        return canonical_git_directory(&dot_git, "Git control directory").map(Some);
    }
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "Git control entry is neither a directory nor a worktree pointer: {}",
            dot_git.display()
        )));
    }
    let bytes = read_bounded_git_file(&dot_git, MAX_GIT_POINTER_BYTES, "Git worktree pointer")?
        .ok_or_else(|| DeltafinError::new("Git worktree pointer disappeared"))?;
    let line = bounded_git_line(&bytes, "Git worktree pointer")?;
    let raw = line
        .strip_prefix("gitdir: ")
        .ok_or_else(|| DeltafinError::new("Git worktree pointer does not start with `gitdir: `"))?;
    if raw.is_empty() || raw.as_bytes().contains(&0) {
        return Err(DeltafinError::new(
            "Git worktree pointer is empty or malformed",
        ));
    }
    let path = PathBuf::from(raw);
    let path = if path.is_absolute() {
        path
    } else {
        repository_root.join(path)
    };
    canonical_git_directory(&path, "Git worktree directory").map(Some)
}

fn resolve_git_common_directory(git_directory: &Path) -> Result<PathBuf> {
    let Some(bytes) = read_bounded_git_file(
        &git_directory.join("commondir"),
        MAX_GIT_POINTER_BYTES,
        "Git common-directory pointer",
    )?
    else {
        return Ok(git_directory.to_owned());
    };
    let raw = bounded_git_line(&bytes, "Git common-directory pointer")?;
    if raw.is_empty() {
        return Err(DeltafinError::new(
            "Git common-directory pointer may not be empty",
        ));
    }
    let path = PathBuf::from(raw);
    let path = if path.is_absolute() {
        path
    } else {
        git_directory.join(path)
    };
    canonical_git_directory(&path, "Git common directory")
}

fn canonical_git_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let metadata = sys_fs::lstat(path).map_err(|error| {
        DeltafinError::new(format!("inspect {label} {}: {error}", path.display()))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DeltafinError::new(format!(
            "{label} is not a real directory: {}",
            path.display()
        )));
    }
    crate::sys::fs::canonicalize(path)
        .map_err(|error| DeltafinError::new(format!("resolve {label} {}: {error}", path.display())))
}

fn bounded_git_line<'a>(bytes: &'a [u8], label: &str) -> Result<&'a str> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| DeltafinError::new(format!("{label} is not valid UTF-8")))?;
    let text = text.trim_end_matches(['\r', '\n']);
    if text.is_empty() || text.contains(['\r', '\n']) {
        return Err(DeltafinError::new(format!(
            "{label} must contain exactly one non-empty line"
        )));
    }
    Ok(text)
}

fn validate_git_reference(reference: &str) -> Result<()> {
    let path = Path::new(reference);
    if !reference.starts_with("refs/")
        || reference.len() > MAX_GIT_POINTER_BYTES as usize
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
        || reference.contains("..")
        || reference.contains("@{")
        || reference.bytes().any(|byte| {
            byte.is_ascii_control()
                || matches!(byte, b' ' | b'~' | b'^' | b':' | b'?' | b'*' | b'[' | b'\\')
        })
    {
        return Err(DeltafinError::new(
            "Git HEAD contains an unsafe or malformed symbolic reference",
        ));
    }
    Ok(())
}

fn validate_git_object_id(value: &str) -> Result<&str> {
    if !matches!(value.len(), 40 | 64) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DeltafinError::new(
            "Git object identity is not a bounded SHA-1 or SHA-256 hexadecimal value",
        ));
    }
    Ok(value)
}

fn resolve_git_reference(
    git_directory: &Path,
    common_directory: &Path,
    reference: &str,
) -> Result<Option<String>> {
    for root in [git_directory, common_directory] {
        if let Some(bytes) = read_bounded_git_file(
            &root.join(reference),
            MAX_GIT_POINTER_BYTES,
            "Git loose reference",
        )? {
            return Ok(Some(
                validate_git_object_id(bounded_git_line(&bytes, "Git loose reference")?)?
                    .to_owned(),
            ));
        }
    }
    for root in [common_directory, git_directory] {
        let Some(bytes) = read_bounded_git_file(
            &root.join("packed-refs"),
            MAX_GIT_PACKED_REFS_BYTES,
            "Git packed references",
        )?
        else {
            continue;
        };
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| DeltafinError::new("Git packed references are not valid UTF-8"))?;
        for line in text.lines() {
            if line.is_empty() || line.starts_with(['#', '^']) {
                continue;
            }
            let Some((object, candidate)) = line.split_once(' ') else {
                return Err(DeltafinError::new(
                    "Git packed references contain a malformed entry",
                ));
            };
            if candidate == reference {
                return Ok(Some(validate_git_object_id(object)?.to_owned()));
            }
        }
    }
    // An unborn branch has a valid symbolic HEAD but no object identity yet.
    Ok(None)
}

fn read_bounded_git_file(path: &Path, maximum_bytes: u64, label: &str) -> Result<Option<Vec<u8>>> {
    let metadata = match sys_fs::lstat(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(DeltafinError::new(format!(
                "inspect {label} {}: {error}",
                path.display()
            )));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum_bytes {
        return Err(DeltafinError::new(format!(
            "{label} is not a real regular file within its {maximum_bytes}-byte bound: {}",
            path.display()
        )));
    }
    let expected = native_file_identity(&metadata);
    let mut file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| DeltafinError::new(format!("open {label} {}: {error}", path.display())))?;
    if native_file_identity(&sys_fs::fstat(&file).map_err(|error| {
        DeltafinError::new(format!(
            "inspect opened {label} {}: {error}",
            path.display()
        ))
    })?) != expected
    {
        return Err(DeltafinError::new(format!(
            "{label} changed identity while it was opened: {}",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    (&mut file)
        .take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| DeltafinError::new(format!("read {label} {}: {error}", path.display())))?;
    if bytes.len() as u64 > maximum_bytes {
        return Err(DeltafinError::new(format!(
            "{label} grew beyond its {maximum_bytes}-byte bound while being read: {}",
            path.display()
        )));
    }
    let final_identity = native_file_identity(&sys_fs::fstat(&file).map_err(|error| {
        DeltafinError::new(format!(
            "reinspect opened {label} {}: {error}",
            path.display()
        ))
    })?);
    if final_identity != expected {
        return Err(DeltafinError::new(format!(
            "{label} changed while it was read: {}",
            path.display()
        )));
    }
    Ok(Some(bytes))
}

fn bounded_git_file_digest(path: &Path, maximum_bytes: u64, label: &str) -> Result<Option<Value>> {
    let Some(bytes) = read_bounded_git_file(path, maximum_bytes, label)? else {
        return Ok(None);
    };
    Ok(Some(json!({
        "size_bytes": bytes.len(),
        "sha256": hex_digest(&crate::packfile::digest_bytes(&bytes)),
    })))
}

fn system_state(repository_root: &Path, runner: &PinnedRunner) -> Value {
    let mut state = lightweight_state(repository_root);
    state["platform"] = json!({
        "os": env::consts::OS,
        "architecture": env::consts::ARCH,
        "family": env::consts::FAMILY,
        "uname": uname_state(),
    });
    state["repository"] = direct_repository_state(repository_root);
    state["build_identity"] = json!({
        "source": "pinned_native_executable_sha256_v1",
        "sha256": runner.sha256,
        "size_bytes": runner.size_bytes,
    });
    state["runner"] = runner_state(runner);

    #[cfg(target_os = "macos")]
    {
        state["macos"] = macos_state();
        state["sysctl"] = darwin_sysctl_state();
    }
    #[cfg(target_os = "linux")]
    {
        state["linux_cpu"] = linux_cpu_state();
    }
    state
}

#[cfg(windows)]
fn uname_state() -> Option<String> {
    None
}

#[cfg(unix)]
fn uname_state() -> Option<String> {
    // SAFETY: `uname` initializes the complete `utsname` structure before
    // returning success.
    let name = unsafe {
        let mut value = std::mem::MaybeUninit::<libc::utsname>::uninit();
        if libc::uname(value.as_mut_ptr()) != 0 {
            return None;
        }
        value.assume_init()
    };
    let fields = [
        name.sysname.as_ptr(),
        name.release.as_ptr(),
        name.version.as_ptr(),
        name.machine.as_ptr(),
    ];
    fields
        .into_iter()
        .map(|field| {
            // SAFETY: every `utsname` member is a NUL-terminated C string in
            // the successfully initialized structure above.
            unsafe { CStr::from_ptr(field) }
                .to_str()
                .ok()
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .collect::<Option<Vec<_>>>()
        .map(|values| values.join(" "))
}

fn runner_state(runner: &PinnedRunner) -> Value {
    let metadata = fs::metadata(&runner.executable_path).ok();
    let modified_ns = metadata
        .as_ref()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
    json!({
        "original_path": runner.original_path,
        "pinned_path": runner.executable_path,
        "sha256": runner.sha256,
        "pinned_size_bytes": runner.size_bytes,
        "source_identity": runner.source_identity,
        "size_bytes": metadata.as_ref().map(fs::Metadata::len),
        "modified_ns": modified_ns,
        "native_magic_verified": true,
    })
}

#[cfg(target_os = "macos")]
fn macos_state() -> Value {
    let mut state = BTreeMap::new();
    state.insert("product_name", "macOS".to_owned());
    for (key, sysctl) in [
        ("product_version", "kern.osproductversion"),
        ("build_version", "kern.osversion"),
    ] {
        if let Some(value) = darwin_sysctl_string(sysctl) {
            state.insert(key, value);
        }
    }
    serde_json::to_value(state).unwrap_or(Value::Null)
}

#[cfg(target_os = "macos")]
fn darwin_sysctl_state() -> Value {
    const STRING_KEYS: &[&str] = &["hw.model", "hw.perflevel0.name", "hw.perflevel1.name"];
    const INTEGER_KEYS: &[&str] = &[
        "hw.memsize",
        "hw.ncpu",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "hw.packages",
        "hw.cpufamily",
        "hw.cachelinesize",
        "hw.l1icachesize",
        "hw.l1dcachesize",
        "hw.l2cachesize",
        "hw.l3cachesize",
        "hw.perflevel0.physicalcpu",
        "hw.perflevel0.logicalcpu",
        "hw.perflevel0.l2cachesize",
        "hw.perflevel1.physicalcpu",
        "hw.perflevel1.logicalcpu",
        "hw.perflevel1.l2cachesize",
        "hw.optional.arm64",
        "hw.optional.neon",
        "hw.optional.armv8_1_atomics",
        "hw.optional.arm.FEAT_DotProd",
        "hw.optional.arm.FEAT_FP16",
        "hw.optional.arm.FEAT_BF16",
        "hw.optional.arm.FEAT_I8MM",
        "hw.optional.arm.FEAT_SME",
        "hw.optional.arm.FEAT_SME2",
    ];
    let mut values = BTreeMap::new();
    for key in STRING_KEYS {
        if let Some(value) = darwin_sysctl_string(key) {
            values.insert((*key).to_owned(), value);
        }
    }
    for key in INTEGER_KEYS {
        if let Some(value) = darwin_sysctl_integer(key) {
            values.insert((*key).to_owned(), value.to_string());
        }
    }
    serde_json::to_value(values).unwrap_or(Value::Null)
}

#[cfg(target_os = "macos")]
fn darwin_sysctl_bytes(name: &str) -> Option<Vec<u8>> {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old_value: *mut c_void,
            old_len: *mut usize,
            new_value: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }

    let name = CString::new(name).ok()?;
    let mut length = 0_usize;
    // SAFETY: the name is a live C string and the null output pointer requests
    // only the required buffer size.
    if unsafe {
        sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    } != 0
        || length == 0
        || length > 1024 * 1024
    {
        return None;
    }
    let mut bytes = vec![0_u8; length];
    // SAFETY: `bytes` has exactly the capacity advertised by the first query;
    // sysctl receives that length as an in/out bound and no write input.
    if unsafe {
        sysctlbyname(
            name.as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    } != 0
        || length > bytes.len()
    {
        return None;
    }
    bytes.truncate(length);
    Some(bytes)
}

#[cfg(target_os = "macos")]
fn darwin_sysctl_string(name: &str) -> Option<String> {
    let mut bytes = darwin_sysctl_bytes(name)?;
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8(bytes)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(target_os = "macos")]
fn darwin_sysctl_integer(name: &str) -> Option<u64> {
    let bytes = darwin_sysctl_bytes(name)?;
    match bytes.as_slice() {
        [a, b, c, d] => Some(u32::from_ne_bytes([*a, *b, *c, *d]).into()),
        [a, b, c, d, e, f, g, h] => Some(u64::from_ne_bytes([*a, *b, *c, *d, *e, *f, *g, *h])),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn linux_cpu_state() -> Value {
    let allowlist = [
        "model name",
        "hardware",
        "cpu implementer",
        "cpu part",
        "flags",
        "features",
    ];
    let mut values = BTreeMap::new();
    if let Ok(text) = fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if line.is_empty() && !values.is_empty() {
                break;
            }
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let key = key.trim().to_ascii_lowercase();
            if allowlist.contains(&key.as_str()) {
                values.insert(key.replace(' ', "_"), value.trim().to_owned());
            }
        }
    }
    serde_json::to_value(values).unwrap_or(Value::Null)
}

fn invocation_contract_errors(
    events: &[Value],
    parsed: &ParsedRun,
    prompt: &str,
    chat: bool,
    max_new_tokens: u64,
) -> Vec<String> {
    let mut errors = Vec::new();
    let starts = events
        .iter()
        .filter(|event| event.get("event").and_then(Value::as_str) == Some("run_start"))
        .collect::<Vec<_>>();
    if let Some(start) = starts.first() {
        if start.get("prompt").and_then(Value::as_str) != Some(prompt) {
            errors.push("run_start.prompt does not exactly match the requested prompt".into());
        }
        if start.get("chat").and_then(Value::as_bool) != Some(chat) {
            errors.push("run_start.chat does not match the requested chat mode".into());
        }
        if start.get("max_new").and_then(Value::as_u64) != Some(max_new_tokens) {
            errors.push(format!(
                "run_start.max_new does not match requested value {max_new_tokens}"
            ));
        }
    }
    if let Some(ids) = &parsed.emitted_token_ids {
        if ids.len() as u64 > max_new_tokens {
            errors.push(format!(
                "runner emitted {} tokens despite --max-new {max_new_tokens}",
                ids.len()
            ));
        }
        if parsed.stop_reason.as_deref() == Some("max_new") && ids.len() as u64 != max_new_tokens {
            errors.push(format!(
                "max_new stop accounted for {} emitted tokens, expected exactly {max_new_tokens}",
                ids.len()
            ));
        }
    }
    if parsed.input_token_ids.is_none() {
        errors.push("runner did not account for exact input token IDs".into());
    }
    if let Some(inference_ns) = parsed.inference_ns {
        match parsed.prefill_ns.unwrap_or(0).checked_add(parsed.decode_ns) {
            Some(measured) if measured > inference_ns => errors.push(format!(
                "prefill plus decode timing ({measured}ns) exceeds run_end duration ({inference_ns}ns)"
            )),
            None => errors.push("prefill plus decode timing overflowed u64".into()),
            _ => {}
        }
    }
    errors
}

fn benchmark_runner_status_error(status: Option<&str>) -> Option<String> {
    match status {
        Some("ok") => None,
        Some("interrupted") => Some(
            "runner was interrupted; its partial structured evidence is not benchmark-valid".into(),
        ),
        status => Some(format!("runner status is {status:?}, expected \"ok\"")),
    }
}

fn requested_on(delta: &BTreeMap<String, String>, name: &str) -> bool {
    delta
        .get(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "force"
            )
        })
        .unwrap_or(false)
}

fn performance_contract_errors(
    environment_delta: &BTreeMap<String, String>,
    parsed: &ParsedRun,
) -> Vec<String> {
    if !requested_on(environment_delta, "K3_INT8_KDA_QKV") {
        return Vec::new();
    }
    let Some(status) = parsed
        .runner_runtime
        .as_ref()
        .and_then(|runtime| runtime.get("int8_kda_qkv"))
        .and_then(Value::as_object)
    else {
        return vec!["K3_INT8_KDA_QKV was requested but no runtime status was captured".into()];
    };
    let mut errors = Vec::new();
    if status.get("requested").and_then(Value::as_bool) != Some(true) {
        errors.push("packed KDA bundle runtime did not record requested=true".into());
    }
    if status.get("eligible").and_then(Value::as_bool) != Some(true) {
        errors.push("packed KDA bundle runtime was not eligible".into());
    }
    if status
        .get("controllers_installed")
        .and_then(Value::as_u64)
        .is_none_or(|value| value < 1)
    {
        errors.push("packed KDA bundle installed no controller".into());
    }
    if status.get("enabled_at_end").and_then(Value::as_bool) != Some(true) {
        errors.push("packed KDA bundle was disabled or fell back before run end".into());
    }
    if status
        .get("packed_project_calls")
        .and_then(Value::as_u64)
        .is_none_or(|value| value == 0)
    {
        errors.push("packed KDA bundle executed no packed projection calls".into());
    }
    let controllers = status.get("controllers").and_then(Value::as_array);
    if controllers.is_none_or(Vec::is_empty) {
        errors.push("packed KDA bundle reported no controller telemetry".into());
    } else if let Some(controllers) = controllers {
        let required = ["q", "k", "v", "f_a", "b"];
        for (index, controller) in controllers.iter().enumerate() {
            let roles = controller
                .get("packed_roles")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<BTreeSet<_>>()
                })
                .unwrap_or_default();
            let missing = required
                .iter()
                .filter(|role| !roles.contains(**role))
                .copied()
                .collect::<Vec<_>>();
            let gate_count = usize::from(roles.contains("g")) + usize::from(roles.contains("g_a"));
            if !missing.is_empty() || gate_count != 1 {
                errors.push(format!(
                    "packed KDA controller {index} is not the complete same-input bundle (missing {}; requires exactly one of g/g_a)",
                    missing.join(",")
                ));
            }
        }
    }
    if status
        .get("disable_reason")
        .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
    {
        errors.push(format!(
            "packed KDA bundle reported a disable reason: {:?}",
            status.get("disable_reason")
        ));
    }

    let storage_mode = environment_delta
        .get("K3_INT8_KDA_STORAGE")
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "arena".into());
    if storage_mode != "stage" {
        return errors;
    }
    if status.get("storage_mode").and_then(Value::as_str) != Some("stage") {
        errors.push("packed KDA stage run did not report storage_mode=stage".into());
    }
    if status
        .get("persistent_weight_bytes")
        .and_then(Value::as_u64)
        != Some(0)
    {
        errors.push("packed KDA stage run retained a persistent int8 weight arena".into());
    }
    if status
        .get("stage_bind_count")
        .and_then(Value::as_u64)
        .is_none_or(|value| value == 0)
    {
        errors.push("packed KDA stage run bound no upload-buffer generation".into());
    }
    if status
        .get("stage_weight_copy_bytes")
        .and_then(Value::as_u64)
        != Some(0)
    {
        errors.push("packed KDA stage run copied int8 weights after upload".into());
    }
    for (field, description) in [
        ("stage_bind_failures", "stage bind failures"),
        ("stage_stale_rejections", "stale stage generations"),
        ("stage_fence_failures", "stage fence failures"),
        ("stage_fence_sync_fallbacks", "blocking fence fallbacks"),
    ] {
        if status.get(field).and_then(Value::as_u64) != Some(0) {
            errors.push(format!("packed KDA stage run reported {description}"));
        }
    }
    let probes = status
        .get("stage_full_shape_probes")
        .and_then(Value::as_u64);
    let passes = status
        .get("stage_full_shape_passes")
        .and_then(Value::as_u64);
    if probes.is_none_or(|value| value == 0) || passes != probes {
        errors.push("packed KDA stage full-shape capability probe did not pass".into());
    }

    let device = parsed
        .runner_config
        .as_ref()
        .and_then(|config| config.get("device"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let sync_mode = environment_delta
        .get("K3_INT8_KDA_STAGE_SYNC")
        .map(|value| value.trim().to_ascii_lowercase().replace('-', "_"))
        .unwrap_or_else(|| "event".into());
    let mps_fifo = device == "mps" && sync_mode == "mps_fifo";
    if let Some(controllers) = controllers {
        for (index, controller) in controllers.iter().enumerate() {
            if controller.get("storage_mode").and_then(Value::as_str) != Some("stage") {
                errors.push(format!(
                    "packed KDA controller {index} did not use stage storage"
                ));
            }
            if controller
                .get("persistent_weight_bytes")
                .and_then(Value::as_u64)
                != Some(0)
            {
                errors.push(format!(
                    "packed KDA controller {index} retained int8 weights"
                ));
            }
            if controller.get("stage_bound").and_then(Value::as_bool) != Some(false) {
                errors.push(format!(
                    "packed KDA controller {index} leaked a live stage binding"
                ));
            }
            let expected = if mps_fifo { "mps_fifo" } else { "event" };
            let actual = controller
                .get("stage_sync_contract")
                .and_then(Value::as_str);
            if (mps_fifo && actual != Some(expected))
                || (!mps_fifo && actual.is_some() && actual != Some(expected))
            {
                errors.push(format!(
                    "packed KDA controller {index} used stage_sync_contract={actual:?}, expected {expected}"
                ));
            }
        }
    }
    if mps_fifo {
        if status.get("stage_sync_mode").and_then(Value::as_str) != Some("mps_fifo") {
            errors.push("packed KDA MPS FIFO run did not report stage_sync_mode=mps_fifo".into());
        }
        for (field, description) in [
            ("stage_fifo_records", "recorded no stream lease"),
            ("stage_fifo_reuses", "reused no stream lease"),
        ] {
            if status
                .get(field)
                .and_then(Value::as_u64)
                .is_none_or(|value| value == 0)
            {
                errors.push(format!("packed KDA MPS FIFO run {description}"));
            }
        }
        if status.get("stage_fence_records").and_then(Value::as_u64) != Some(0) {
            errors.push("packed KDA MPS FIFO run unexpectedly recorded events".into());
        }
        if status.get("stage_fence_waits").and_then(Value::as_u64) != Some(0) {
            errors.push("packed KDA MPS FIFO run unexpectedly waited on events".into());
        }
    } else if device == "mps" || device.starts_with("cuda") {
        if status
            .get("stage_fence_records")
            .and_then(Value::as_u64)
            .is_none_or(|value| value == 0)
        {
            errors.push("packed KDA stage GPU run recorded no reuse fence".into());
        }
        if status
            .get("stage_fence_waits")
            .and_then(Value::as_u64)
            .is_none_or(|value| value == 0)
        {
            errors.push("packed KDA stage GPU run waited on no reuse fence".into());
        }
    }
    errors
}

fn drafter_contract_errors(
    environment_delta: &BTreeMap<String, String>,
    parsed: &ParsedRun,
) -> Vec<String> {
    let dspark_on = requested_on(environment_delta, "K3_DSPARK");
    let qwen_on = requested_on(environment_delta, "K3_UAG_DRAFT");
    if !dspark_on && !qwen_on {
        return Vec::new();
    }
    let (Some(config), Some(runtime)) = (
        parsed.runner_config.as_ref().and_then(Value::as_object),
        parsed.runner_runtime.as_ref().and_then(Value::as_object),
    ) else {
        return vec!["requested drafter has no structured runtime evidence".into()];
    };
    let mut errors = Vec::new();
    let should_propose = parsed
        .completion_token_ids
        .as_ref()
        .is_some_and(|ids| ids.len() >= 3);
    if dspark_on {
        if config.get("dspark_loaded").and_then(Value::as_bool) != Some(true) {
            errors.push("DSpark was requested on but did not load".into());
        }
        let Some(status) = runtime.get("dspark").and_then(Value::as_object) else {
            errors.push("DSpark was requested on but has no runtime status".into());
            return errors;
        };
        if status.get("available").and_then(Value::as_bool) != Some(true) {
            errors.push("DSpark runtime did not report available=true".into());
        }
        if should_propose
            && status
                .get("proposals")
                .and_then(Value::as_u64)
                .is_none_or(|value| value == 0)
        {
            errors.push("DSpark executed no proposal".into());
        }
        for field in ["proposal_failures", "state_failures"] {
            if status.get(field).and_then(Value::as_u64) != Some(0) {
                errors.push(format!("DSpark reported {field}={:?}", status.get(field)));
            }
        }
        return errors;
    }
    if config
        .get("universal_draft_loaded")
        .and_then(Value::as_bool)
        != Some(true)
    {
        errors.push("Qwen drafting was requested on but did not load".into());
    }
    let Some(status) = runtime.get("universal_draft").and_then(Value::as_object) else {
        errors.push("Qwen drafting was requested on but has no runtime status".into());
        return errors;
    };
    if should_propose
        && status
            .get("proposals")
            .and_then(Value::as_u64)
            .is_none_or(|value| value == 0)
    {
        errors.push("Qwen executed no proposal".into());
    }
    if status.get("failures").and_then(Value::as_u64) != Some(0) {
        errors.push(format!(
            "Qwen reported failures={:?}",
            status.get("failures")
        ));
    }
    errors
}

fn compare_output(
    parsed: &ParsedRun,
    expected_ids: Option<&[u64]>,
    expected_text: Option<&str>,
) -> (Option<bool>, Vec<String>) {
    let mut checks = 0usize;
    let mut errors = Vec::new();
    if let Some(expected_ids) = expected_ids {
        checks += 1;
        if parsed.completion_token_ids.as_deref() != Some(expected_ids) {
            errors.push(format!(
                "completion token IDs differ: expected {expected_ids:?}, got {:?}",
                parsed.completion_token_ids
            ));
        }
    }
    if let Some(expected_text) = expected_text {
        checks += 1;
        if parsed.completion_text.as_deref() != Some(expected_text) {
            errors.push(format!(
                "completion text differs: expected {expected_text:?}, got {:?}",
                parsed.completion_text
            ));
        }
    }
    ((checks > 0).then_some(errors.is_empty()), errors)
}

#[allow(clippy::too_many_arguments)]
fn run_once(
    run_number: usize,
    repetition: usize,
    arm: &BenchmarkArm,
    options: &BenchmarkOptions,
    runner: &PinnedRunner,
    repository_root: &Path,
    output_dir: &Path,
    expected_ids: Option<&[u64]>,
    expected_text: Option<&str>,
) -> Result<RunRecord> {
    verify_pinned_runner(runner)?;
    let run_dir = output_dir.join(format!("run-{run_number:03}-{}", slug(&arm.name)));
    create_run_directory(&run_dir)?;
    let stdout_path = run_dir.join("stdout.log");
    let stderr_path = run_dir.join("stderr.log");
    let events_path = run_dir.join("events.jsonl");
    let result_path = run_dir.join("result.json");
    let delta = parse_environment_delta(&arm.environment_spec)?;

    let mut arguments = vec![
        OsString::from("run"),
        OsString::from("--prompt"),
        OsString::from(&options.prompt),
        OsString::from("--max-new"),
        OsString::from(options.max_new_tokens.to_string()),
        OsString::from("--events-jsonl"),
        events_path.as_os_str().to_owned(),
        OsString::from("--model-root"),
        repository_root.as_os_str().to_owned(),
    ];
    if options.chat {
        arguments.push(OsString::from("--chat"));
    }
    let command_display = std::iter::once(
        runner
            .executable_path
            .as_os_str()
            .to_string_lossy()
            .into_owned(),
    )
    .chain(
        arguments
            .iter()
            .map(|argument| argument.to_string_lossy().into_owned()),
    )
    .collect::<Vec<_>>();
    let started_at_utc = utc_now()?;
    let state_before = lightweight_state(repository_root);
    let outcome = execute_runner(
        &runner.executable_path,
        repository_root,
        &arguments,
        &delta,
        &stdout_path,
        &stderr_path,
        &events_path,
        options.timeout,
    )?;
    let runner_integrity_error = verify_pinned_runner(runner)
        .err()
        .map(|error| error.to_string());
    let state_after = lightweight_state(repository_root);
    if events_path.exists() {
        let event_file = Open::new()
            .read(true)
            .write(true)
            .no_follow()
            .open(&events_path)
            .map_err(|error| {
                DeltafinError::new(format!(
                    "open completed native event stream {} for fsync: {error}",
                    events_path.display()
                ))
            })?;
        event_file.sync_all().map_err(|error| {
            DeltafinError::new(format!(
                "fsync completed native event stream {}: {error}",
                events_path.display()
            ))
        })?;
    }
    let (events, event_errors) = read_events(&events_path);
    let parsed = parse_structured_events(&events, options.warmup_steps, event_errors);
    let (output_match, mismatch_errors) = compare_output(&parsed, expected_ids, expected_text);
    let mut errors = Vec::new();
    if let Some(spawn_error) = &outcome.spawn_error {
        errors.push(spawn_error.clone());
    }
    if let Some(error) = runner_integrity_error {
        errors.push(error);
    }
    errors.extend(outcome.capture_errors.iter().cloned());
    if outcome.timed_out {
        errors.push(format!(
            "runner timed out after {:.6}s",
            options.timeout.as_secs_f64()
        ));
    } else if let Some(status) = outcome.status {
        if !status.success() {
            match (status.code(), exit_signal(&status)) {
                (Some(code), _) => errors.push(format!("runner exited with status {code}")),
                (_, Some(signal)) => errors.push(format!("runner exited after signal {signal}")),
                _ => errors.push("runner exited unsuccessfully".into()),
            }
        }
    } else if outcome.spawn_error.is_none() {
        errors.push("runner exit status was not captured".into());
    }
    if events.is_empty() {
        errors.push("runner did not produce a structured native event stream".into());
    }
    if let Some(error) = benchmark_runner_status_error(parsed.runner_status.as_deref()) {
        errors.push(error);
    }
    errors.extend(parsed.parse_errors.iter().cloned());
    errors.extend(invocation_contract_errors(
        &events,
        &parsed,
        &options.prompt,
        options.chat,
        options.max_new_tokens,
    ));
    if output_match == Some(false) {
        errors.extend(mismatch_errors);
    }
    errors.extend(performance_contract_errors(&delta, &parsed));
    errors.extend(drafter_contract_errors(&delta, &parsed));

    let effective_environment = relevant_environment(
        env::vars_os().chain(delta.iter().map(|(key, value)| (key.into(), value.into()))),
    );
    let parsed_value = serde_json::to_value(&parsed).map_err(|error| {
        DeltafinError::new(format!(
            "serialize parsed native benchmark evidence: {error}"
        ))
    })?;
    let return_code = outcome.status.and_then(|status| status.code());
    let exit_signal = outcome.status.and_then(|status| exit_signal(&status));
    let valid = errors.is_empty();
    let value = json!({
        "schema": BENCHMARK_SCHEMA,
        "record_type": "run",
        "run_number": run_number,
        "repetition": repetition,
        "config_name": arm.name,
        "config_spec": arm.environment_spec,
        "environment_delta": delta,
        "effective_performance_environment": effective_environment,
        "command": command_display,
        "runner": runner_state(runner),
        "cwd": repository_root,
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now()?,
        "wall_ns": outcome.wall_ns,
        "wall_s": ns_to_seconds(outcome.wall_ns),
        "returncode": return_code,
        "exit_signal": exit_signal,
        "timed_out": outcome.timed_out,
        "capture": {
            "stdout_bytes": outcome.stdout_bytes,
            "stdout_limit_bytes": MAX_STDOUT_FILE_BYTES,
            "stderr_bytes": outcome.stderr_bytes,
            "stderr_limit_bytes": MAX_STDERR_FILE_BYTES,
            "event_limit_bytes": MAX_EVENT_FILE_BYTES,
        },
        "state_before": state_before,
        "state_after": state_after,
        "artifacts": {
            "run_dir": run_dir,
            "stdout": stdout_path,
            "stderr": stderr_path,
            "events": events_path,
            "result": result_path,
        },
        "parsed": parsed_value,
        "expected_completion_token_ids": expected_ids,
        "expected_completion_text": expected_text,
        "output_match": output_match,
        "valid": valid,
        "errors": errors,
        "stderr_tail": stderr_tail(&stderr_path, 2_000),
    });
    Ok(RunRecord {
        value,
        valid,
        output_match,
        completion_ids: parsed.completion_token_ids,
        completion_text: parsed.completion_text,
        result_path,
    })
}

fn slug(text: &str) -> String {
    let mut output = String::new();
    let mut separator = false;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            if separator && !output.is_empty() {
                output.push('-');
            }
            separator = false;
            output.push(character);
        } else {
            separator = true;
        }
        if output.len() >= 64 {
            break;
        }
    }
    let output = output.trim_matches('-');
    if output.is_empty() {
        "config".into()
    } else {
        output.to_owned()
    }
}

fn numeric_stats(values: impl IntoIterator<Item = Option<f64>>) -> Option<Value> {
    let mut values = values
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let median = median_sorted(&values);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let mad = median_sorted(&deviations);
    let stdev = (values.len() > 1).then(|| {
        let squared = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (squared / (values.len() - 1) as f64).sqrt()
    });
    Some(json!({
        "count": values.len(),
        "median": median,
        "min": values.first(),
        "max": values.last(),
        "mean": mean,
        "mad": mad,
        "stdev": stdev,
    }))
}

fn median_sorted(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn metric(records: &[&RunRecord], pointer: &str) -> Option<Value> {
    numeric_stats(
        records
            .iter()
            .map(|record| record.value.pointer(pointer).and_then(Value::as_f64)),
    )
}

fn summarize(records: &[RunRecord], arms: &[BenchmarkArm]) -> Result<Value> {
    let mut summaries = Vec::with_capacity(arms.len());
    for arm in arms {
        let runs = records
            .iter()
            .filter(|record| {
                record.value.get("config_name").and_then(Value::as_str) == Some(&arm.name)
            })
            .collect::<Vec<_>>();
        let valid = runs
            .iter()
            .copied()
            .filter(|record| record.valid)
            .collect::<Vec<_>>();
        let metrics = json!({
            "wall_s": metric(&valid, "/wall_s"),
            "prefill_s": metric(&valid, "/parsed/prefill_s"),
            "inference_s": metric(&valid, "/parsed/inference_s"),
            "decode_tps": metric(&valid, "/parsed/decode_tps"),
            "steady_tps": metric(&valid, "/parsed/steady_tps"),
            "steady_s_per_token": metric(&valid, "/parsed/steady_s_per_token"),
            "draft_acceptance_rate": metric(&valid, "/parsed/draft_acceptance_rate"),
            "steady_draft_acceptance_rate": metric(&valid, "/parsed/steady_draft_acceptance_rate"),
        });
        let median_tps = metrics
            .pointer("/steady_tps/median")
            .and_then(Value::as_f64);
        summaries.push((
            arm.name.clone(),
            median_tps,
            json!({
                "attempted_runs": runs.len(),
                "valid_runs": valid.len(),
                "invalid_runs": runs.len().saturating_sub(valid.len()),
                "all_exact_output_match": !runs.is_empty()
                    && runs.iter().all(|record| record.output_match == Some(true)),
                "metrics": metrics,
                "relative_to_first_config": Value::Null,
                "run_numbers": runs.iter().filter_map(|record| {
                    record.value.get("run_number").and_then(Value::as_u64)
                }).collect::<Vec<_>>(),
            }),
        ));
    }
    let baseline_tps = summaries.first().and_then(|(_, median, _)| *median);
    let baseline_range = summaries.first().and_then(|(_, _, summary)| {
        Some((
            summary.pointer("/metrics/steady_tps/min")?.as_f64()?,
            summary.pointer("/metrics/steady_tps/max")?.as_f64()?,
        ))
    });
    let mut configurations = Map::new();
    for (index, (name, median_tps, mut summary)) in summaries.into_iter().enumerate() {
        summary["relative_to_first_config"] = match (median_tps, baseline_tps) {
            (Some(value), Some(baseline)) if baseline > 0.0 => json!(value / baseline),
            _ => Value::Null,
        };
        if index > 0 {
            if let (Some((base_min, base_max)), Some(min), Some(max)) = (
                baseline_range,
                summary
                    .pointer("/metrics/steady_tps/min")
                    .and_then(Value::as_f64),
                summary
                    .pointer("/metrics/steady_tps/max")
                    .and_then(Value::as_f64),
            ) {
                summary["steady_tps_range_overlaps_baseline"] =
                    Value::Bool(!(min > base_max || max < base_min));
            }
        }
        configurations.insert(name, summary);
    }
    let errors = records
        .iter()
        .filter(|record| !record.valid)
        .map(|record| {
            json!({
                "run_number": record.value.get("run_number"),
                "config_name": record.value.get("config_name"),
                "errors": record.value.get("errors"),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": BENCHMARK_SCHEMA,
        "record_type": "summary",
        "generated_at_utc": utc_now()?,
        "attempted_runs": records.len(),
        "valid_runs": records.iter().filter(|record| record.valid).count(),
        "all_runs_valid": !records.is_empty() && records.iter().all(|record| record.valid),
        "all_outputs_exact": !records.is_empty()
            && records.iter().all(|record| record.output_match == Some(true)),
        "configs": configurations,
        "errors": errors,
    }))
}

fn amend_oracle(record: &mut RunRecord, oracle_ids: &[u64], oracle_text: &str) {
    record.value["expected_completion_token_ids"] = json!(oracle_ids);
    record.value["expected_completion_text"] = json!(oracle_text);
    record.value["output_match"] = Value::Bool(true);
    record.output_match = Some(true);
}

fn invalidate_without_oracle(record: &mut RunRecord, reference_name: &str) {
    let message = format!(
        "no complete exact-output oracle is available; a successful run of reference config {reference_name:?} must come first"
    );
    record.value["valid"] = Value::Bool(false);
    if let Some(errors) = record.value.get_mut("errors").and_then(Value::as_array_mut) {
        errors.push(Value::String(message));
    }
    record.valid = false;
}

fn print_summary(summary: &Value, arms: &[BenchmarkArm]) {
    println!();
    println!("{}", "=".repeat(92));
    println!(
        "{:<28} {:>7} {:>14} {:>16} {:>19} {:>10}",
        "config", "valid", "prefill s", "steady tok/s", "range", "relative"
    );
    println!("{}", "=".repeat(92));
    for arm in arms {
        let config = &summary["configs"][&arm.name];
        let attempted = config["attempted_runs"].as_u64().unwrap_or(0);
        let valid = config["valid_runs"].as_u64().unwrap_or(0);
        let prefill = config["metrics"]["prefill_s"]["median"]
            .as_f64()
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".into());
        let steady = config["metrics"]["steady_tps"]["median"]
            .as_f64()
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".into());
        let range = match (
            config["metrics"]["steady_tps"]["min"].as_f64(),
            config["metrics"]["steady_tps"]["max"].as_f64(),
        ) {
            (Some(minimum), Some(maximum)) => format!("{minimum:.6}-{maximum:.6}"),
            _ => "-".into(),
        };
        let relative = config["relative_to_first_config"]
            .as_f64()
            .map(|value| format!("{value:.4}x"))
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<28} {:>7} {:>14} {:>16} {:>19} {:>10}",
            arm.name,
            format!("{valid}/{attempted}"),
            prefill,
            steady,
            range,
            relative
        );
    }
    println!("{}", "=".repeat(92));
    if summary.get("all_runs_valid").and_then(Value::as_bool) == Some(true)
        && summary.get("all_outputs_exact").and_then(Value::as_bool) == Some(true)
    {
        println!("Every run succeeded and matched the exact token-ID/text oracle.");
    } else {
        println!("INVALID CAMPAIGN: failed runs or exact-output mismatches are present.");
    }
}

/// Execute an interleaved native benchmark campaign and durably persist all
/// evidence. A successful return describes both valid and invalid campaigns;
/// use [`BenchmarkReport::succeeded`] for the CLI exit-status decision.
#[cfg_attr(windows, allow(unreachable_code, unused_variables))]
pub fn run_campaign(options: &BenchmarkOptions) -> Result<BenchmarkReport> {
    // The campaign pins an exec-bit-protected private copy of the runner and
    // supervises it as a Unix process group, and a Windows runner cannot find
    // its DLLs from a copy elsewhere. Refuse rather than measure something
    // different from what the evidence claims.
    #[cfg(windows)]
    return Err(DeltafinError::new(
        "`deltafin benchmark` is not supported on Windows yet; run the compiled binary directly with --stats",
    ));
    // Cargo injects a DYLD fallback path into this crate's macOS test harness
    // so the test executable can find LibTorch. Production binaries never
    // receive this cfg exemption; the pure loader-audit tests cover the exact
    // rejection policy without mutating the process-global test environment.
    #[cfg(not(test))]
    crate::loader_audit::reject_dynamic_loader_environment()?;
    crate::engine::reject_product_metal_source_override()?;
    let (repository_root, original_runner) = validate_options(options)?;
    audit_benchmark_executable(&original_runner)?;
    let output_dir = make_output_dir(&repository_root, options.output_dir.as_deref())?;
    let runner = pin_native_executable(&original_runner, &output_dir)?;
    verify_pinned_runner(&runner)?;
    let parsed_deltas = options
        .arms
        .iter()
        .map(|arm| parse_environment_delta(&arm.environment_spec))
        .collect::<Result<Vec<_>>>()?;
    let campaign = json!({
        "schema": BENCHMARK_SCHEMA,
        "record_type": "campaign",
        "created_at_utc": utc_now()?,
        "output_dir": output_dir,
        "root": repository_root,
        "arguments": {
            "configs": options.arms.iter().map(|arm| &arm.environment_spec).collect::<Vec<_>>(),
            "names": options.arms.iter().map(|arm| &arm.name).collect::<Vec<_>>(),
            "parsed_environment_deltas": parsed_deltas,
            "repetitions": options.repetitions,
            "max_new_tokens": options.max_new_tokens,
            "prompt": options.prompt,
            "chat": options.chat,
            "warmup_steps": options.warmup_steps,
            "timeout_s": options.timeout.as_secs_f64(),
            "expected_completion_token_ids": options.expected_completion_token_ids,
            "expected_completion_text": options.expected_completion_text,
            "keep_going": options.keep_going,
            "runner": runner_state(&runner),
            "runner_kind": "compiled_native_executable",
        },
        "base_performance_environment": relevant_environment(env::vars_os()),
        "system_at_start": system_state(&repository_root, &runner),
    });
    write_json_exclusive(&output_dir.join("campaign.json"), &campaign)?;
    let runs_path = output_dir.join("runs.jsonl");
    let mut runs_file = secure_create_new(&runs_path, 0o600)?;
    runs_file
        .sync_all()
        .map_err(|error| DeltafinError::new(format!("fsync new benchmark runs stream: {error}")))?;
    fsync_directory(&output_dir)?;
    println!("evidence: {}", output_dir.display());

    let supplied_ids = options.expected_completion_token_ids.is_some();
    let supplied_text = options.expected_completion_text.is_some();
    let mut oracle_ids = options.expected_completion_token_ids.clone();
    let mut oracle_text = options.expected_completion_text.clone();
    let mut records = Vec::new();
    let mut stopped_early = false;
    let mut run_number = 0usize;
    'campaign: for repetition in 1..=options.repetitions {
        for (arm_index, arm) in options.arms.iter().enumerate() {
            run_number += 1;
            let mut record = run_once(
                run_number,
                repetition,
                arm,
                options,
                &runner,
                &repository_root,
                &output_dir,
                oracle_ids.as_deref(),
                oracle_text.as_deref(),
            )?;

            // Fill either or both absent oracle components only from a valid
            // reference-arm result. Explicitly supplied components were
            // already checked by run_once before this point.
            if arm_index == 0 && record.valid && (oracle_ids.is_none() || oracle_text.is_none()) {
                if oracle_ids.is_none() {
                    oracle_ids.clone_from(&record.completion_ids);
                }
                if oracle_text.is_none() {
                    oracle_text.clone_from(&record.completion_text);
                }
                if let (Some(ids), Some(text)) = (&oracle_ids, &oracle_text) {
                    amend_oracle(&mut record, ids, text);
                }
            } else if (oracle_ids.is_none() || oracle_text.is_none()) && record.valid {
                invalidate_without_oracle(&mut record, &options.arms[0].name);
            }

            write_json_exclusive(&record.result_path, &record.value)?;
            append_jsonl(&mut runs_file, &runs_path, &record.value)?;
            let prefill = record
                .value
                .pointer("/parsed/prefill_s")
                .and_then(Value::as_f64)
                .map(|value| format!("{value:.6}s"))
                .unwrap_or_else(|| "-".into());
            let steady = record
                .value
                .pointer("/parsed/steady_tps")
                .and_then(Value::as_f64)
                .map(|value| format!("{value:.6} tok/s"))
                .unwrap_or_else(|| "-".into());
            println!(
                "  rep{repetition} {:<28} prefill {:>14} steady {:>18}  {}",
                arm.name,
                prefill,
                steady,
                if record.valid { "ok" } else { "INVALID" }
            );
            if !record.valid {
                if let Some(errors) = record.value.get("errors").and_then(Value::as_array) {
                    for error in errors.iter().filter_map(Value::as_str) {
                        println!("    ERROR: {error}");
                    }
                }
                if let Some(tail) = record
                    .value
                    .get("stderr_tail")
                    .and_then(Value::as_str)
                    .filter(|tail| !tail.is_empty())
                {
                    println!("    stderr tail:\n      {}", tail.replace('\n', "\n      "));
                }
            }
            records.push(record);
            if !records.last().is_some_and(|record| record.valid) && !options.keep_going {
                stopped_early = true;
                break 'campaign;
            }
        }
    }

    let mut summary = summarize(&records, &options.arms)?;
    summary["stopped_early"] = Value::Bool(stopped_early);
    summary["exact_oracle"] = json!({
        "completion_token_ids": oracle_ids,
        "completion_text": oracle_text,
        "source": match (supplied_ids, supplied_text) {
            (true, true) => "command_line",
            (false, false) => "first_valid_reference_run",
            _ => "command_line_plus_first_valid_reference_run",
        },
    });
    summary["system_at_end"] = system_state(&repository_root, &runner);
    write_json_exclusive(&output_dir.join("summary.json"), &summary)?;
    print_summary(&summary, &options.arms);
    println!("raw evidence and summary: {}", output_dir.display());
    Ok(BenchmarkReport {
        output_dir,
        summary,
    })
}

// The harness pins a private copy of the runner (exec-bit and process-group
// semantics) and so does not run on Windows yet; its tests are Unix-only.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let path = env::temp_dir().join(format!(
                "deltafin-benchmark-{label}-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn envelope(event: &str, fields: Value, monotonic_ns: u64) -> Value {
        let mut object = fields.as_object().unwrap().clone();
        object.insert("schema".into(), Value::String(EVENT_SCHEMA.into()));
        object.insert("event".into(), Value::String(event.into()));
        object.insert("wall_time_ns".into(), json!(1_000_000 + monotonic_ns));
        object.insert("monotonic_ns".into(), json!(monotonic_ns));
        Value::Object(object)
    }

    fn valid_events() -> Vec<Value> {
        vec![
            envelope(
                "run_start",
                json!({
                    "prompt": "native prompt",
                    "chat": false,
                    "max_new": 4,
                    "input_token_ids": [7, 8],
                    "config": {"device": "cpu", "eos_token_id": 99},
                }),
                1,
            ),
            envelope(
                "prefill_done",
                json!({"duration_ns": 1_250_000_000u64, "emitted_token_ids": [10]}),
                2,
            ),
            envelope(
                "decode_step",
                json!({
                    "step": 0,
                    "duration_ns": 2_000_000_000u64,
                    "emitted_token_ids": [11],
                    "proposed_token_count": 0,
                    "accepted_token_count": 0,
                }),
                3,
            ),
            envelope(
                "decode_step",
                json!({
                    "step": 1,
                    "duration_ns": 500_000_000u64,
                    "emitted_token_ids": [12, 13],
                    "proposed_token_count": 1,
                    "accepted_token_count": 1,
                }),
                4,
            ),
            envelope(
                "run_end",
                json!({
                    "status": "ok",
                    "duration_ns": 3_750_000_000u64,
                    "prompt_token_ids": [7, 8],
                    "emitted_token_ids": [10, 11, 12, 13],
                    "completion_token_ids": [10, 11, 12, 13],
                    "completion_text": "fake completion",
                    "stop_reason": "max_new",
                    "runtime": {},
                }),
                5,
            ),
        ]
    }

    #[test]
    fn shell_like_environment_parser_never_executes_or_expands() {
        let parsed = parse_environment_delta(
            r#"K3_A=1 K3_LABEL="two words" K3_EMPTY= K3_LITERAL='$(touch /tmp/nope)'"#,
        )
        .unwrap();
        assert_eq!(parsed["K3_A"], "1");
        assert_eq!(parsed["K3_LABEL"], "two words");
        assert_eq!(parsed["K3_EMPTY"], "");
        assert_eq!(parsed["K3_LITERAL"], "$(touch /tmp/nope)");
        assert!(parse_environment_delta("K3_OK=1 bare-word").is_err());
        assert!(parse_environment_delta("K3_BAD='unterminated").is_err());
        assert!(parse_environment_delta("K3_API_TOKEN=secret").is_err());
        assert!(parse_environment_delta("LD_PRELOAD=/tmp/injected.so").is_err());
        assert!(parse_environment_delta("DYLD_LIBRARY_PATH=/tmp/native").is_err());
        for secret in [
            "K3_APIKEY=secret",
            "K3_PAT=secret",
            "DELTAFIN_BEARER=secret",
            "K3_AUTHTOKEN=secret",
            "K3_ACCESSKEYID=secret",
            "K3_CLIENTSECRET=secret",
            "K3_COOKIEJAR=secret",
            "K3_PASSWORDHASH=secret",
            "K3_SECRETFILE=secret",
            "DELTAFIN_AUTHHEADER=secret",
        ] {
            assert!(
                parse_environment_delta(secret).is_err(),
                "secret variant was persisted: {secret}"
            );
        }
        assert!(parse_environment_delta("K3_CACHE_KEY_MODE=exact").is_ok());
        assert!(parse_environment_delta("K3_MAX_NEW_TOKENS=4").is_ok());
        assert!(parse_environment_delta("K3_TOKENIZER_MODE=exact").is_ok());
        assert_eq!(parse_expected_token_ids("[1, 2]").unwrap(), [1, 2]);
        assert_eq!(parse_expected_token_ids("1,2").unwrap(), [1, 2]);
        assert!(parse_expected_token_ids("[-1]").is_err());
        assert!(parse_expected_token_ids("[4294967296]").is_err());
    }

    #[test]
    fn structured_metrics_count_transactions_and_enforce_accounting() {
        let events = valid_events();
        let parsed = parse_structured_events(&events, 1, Vec::new());
        assert!(parsed.parse_errors.is_empty(), "{:?}", parsed.parse_errors);
        assert_eq!(parsed.prefill_s, Some(1.25));
        assert_eq!(parsed.decode_emitted_tokens, 3);
        assert_eq!(parsed.steady_decode_tokens, 2);
        assert_eq!(parsed.steady_decode_ns, 500_000_000);
        assert_eq!(parsed.steady_tps, Some(4.0));
        assert_eq!(parsed.draft_acceptance_rate, Some(1.0));
        assert_eq!(parsed.steady_draft_acceptance_rate, Some(1.0));
        assert_eq!(parsed.completion_token_ids, Some(vec![10, 11, 12, 13]));
        assert!(invocation_contract_errors(&events, &parsed, "native prompt", false, 4).is_empty());

        let mut corrupt = valid_events();
        corrupt[3]["emitted_token_ids"] = json!([99]);
        let parsed = parse_structured_events(&corrupt, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("token accounting differs"))
        );
    }

    #[test]
    fn interrupted_events_remain_parseable_but_are_never_benchmark_valid() {
        let mut after_transaction = valid_events();
        after_transaction[4]["status"] = json!("interrupted");
        after_transaction[4]["stop_reason"] = json!("interrupted");
        let parsed = parse_structured_events(&after_transaction, 0, Vec::new());
        assert!(parsed.parse_errors.is_empty(), "{:?}", parsed.parse_errors);
        assert_eq!(parsed.runner_status.as_deref(), Some("interrupted"));
        assert!(
            benchmark_runner_status_error(parsed.runner_status.as_deref())
                .is_some_and(|error| error.contains("not benchmark-valid"))
        );

        let mut interrupted_at_eos = after_transaction.clone();
        interrupted_at_eos[3]["emitted_token_ids"] = json!([12, 99]);
        interrupted_at_eos[4]["emitted_token_ids"] = json!([10, 11, 12, 99]);
        interrupted_at_eos[4]["completion_token_ids"] = json!([10, 11, 12]);
        let parsed = parse_structured_events(&interrupted_at_eos, 0, Vec::new());
        assert!(parsed.parse_errors.is_empty(), "{:?}", parsed.parse_errors);

        let before_completion = vec![
            envelope(
                "run_start",
                json!({
                    "prompt": "native prompt",
                    "chat": false,
                    "max_new": 4,
                    "input_token_ids": [7, 8],
                    "config": {"device": "cpu", "eos_token_id": 99},
                }),
                1,
            ),
            envelope(
                "prefill_done",
                json!({"duration_ns": 1_250_000_000u64, "emitted_token_ids": []}),
                2,
            ),
            envelope(
                "run_end",
                json!({
                    "status": "interrupted",
                    "duration_ns": 1_250_000_000u64,
                    "prompt_token_ids": [7, 8],
                    "emitted_token_ids": [],
                    "completion_token_ids": [],
                    "completion_text": "",
                    "stop_reason": "interrupted",
                    "runtime": {},
                }),
                3,
            ),
        ];
        let parsed = parse_structured_events(&before_completion, 0, Vec::new());
        assert!(parsed.parse_errors.is_empty(), "{:?}", parsed.parse_errors);
        assert_eq!(parsed.decode_step_count, 0);
        assert_eq!(parsed.emitted_token_ids, Some(Vec::new()));
        assert!(
            invocation_contract_errors(&before_completion, &parsed, "native prompt", false, 4)
                .is_empty()
        );
    }

    #[test]
    fn structured_events_reject_noncontiguous_steps_and_impossible_draft_totals() {
        for invalid_step in [Value::Null, json!(1), json!("0")] {
            let mut events = valid_events();
            events[2]["step"] = invalid_step;
            let parsed = parse_structured_events(&events, 0, Vec::new());
            assert!(
                parsed
                    .parse_errors
                    .iter()
                    .any(|error| error.contains("contiguous zero-based")
                        || error.contains("lacks a nonnegative integer step")),
                "{:?}",
                parsed.parse_errors
            );
        }
        let mut duplicate = valid_events();
        duplicate[3]["step"] = json!(0);
        let parsed = parse_structured_events(&duplicate, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("expected contiguous zero-based step 1"))
        );

        let mut events = valid_events();
        events[3]["accepted_token_count"] = json!(2);
        events[3]["proposed_token_count"] = json!(2);
        events[3]["emitted_token_ids"] = json!([12]);
        let parsed = parse_structured_events(&events, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("accepted 2 drafts but emitted only 1"))
        );

        let mut events = valid_events();
        events[2]["proposed_token_count"] = json!(u64::MAX);
        events[3]["proposed_token_count"] = json!(u64::MAX);
        let parsed = parse_structured_events(&events, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("proposed draft token total overflowed u64"))
        );
    }

    #[test]
    fn structured_events_require_a_bounded_eos_identity() {
        let mut missing = valid_events();
        missing[0]["config"]
            .as_object_mut()
            .unwrap()
            .remove("eos_token_id");
        let parsed = parse_structured_events(&missing, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("config.eos_token_id is missing"))
        );

        let mut oversized = valid_events();
        oversized[0]["config"]["eos_token_id"] = json!(u32::MAX as u64 + 1);
        let parsed = parse_structured_events(&oversized, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("exceeds the native u32 token-ID range"))
        );
    }

    #[test]
    fn eos_may_be_excluded_once_but_not_arbitrary_tokens() {
        let mut events = valid_events();
        events[4]["stop_reason"] = json!("eos");
        events[3]["emitted_token_ids"] = json!([12, 99]);
        events[4]["emitted_token_ids"] = json!([10, 11, 12, 99]);
        events[4]["completion_token_ids"] = json!([10, 11, 12]);
        let parsed = parse_structured_events(&events, 0, Vec::new());
        assert!(parsed.parse_errors.is_empty(), "{:?}", parsed.parse_errors);

        events[3]["emitted_token_ids"] = json!([12, 13]);
        events[4]["emitted_token_ids"] = json!([10, 11, 12, 13]);
        let parsed = parse_structured_events(&events, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("config.eos_token_id"))
        );

        events[4]["completion_token_ids"] = json!([10, 11]);
        let parsed = parse_structured_events(&events, 0, Vec::new());
        assert!(
            parsed
                .parse_errors
                .iter()
                .any(|error| error.contains("EOS stop must omit exactly one"))
        );
    }

    #[test]
    fn native_executable_check_rejects_scripts_even_when_executable() {
        let root = TestDirectory::new("reject-script");
        let script = root.0.join("runner");
        fs::write(&script, b"#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let error = verify_native_executable(&script).unwrap_err().to_string();
        assert!(error.contains("compiled executable"), "{error}");
    }

    #[test]
    fn state_capture_filters_credentials_and_calendar_is_exact() {
        let visible = relevant_environment([
            ("K3_FAST", "1"),
            ("K3_HF_TOKEN", "do-not-record"),
            ("DELTAFIN_API_KEY", "do-not-record"),
            ("K3_APIKEY", "do-not-record"),
            ("K3_PAT", "do-not-record"),
            ("DELTAFIN_BEARER", "do-not-record"),
            ("K3_AUTHTOKEN", "do-not-record"),
            ("UNRELATED", "do-not-record"),
        ]);
        assert_eq!(visible.get("K3_FAST").map(String::as_str), Some("1"));
        assert_eq!(visible.len(), 1);
        assert_eq!(
            format_utc(0, 123_456_789, false),
            "1970-01-01T00:00:00.123456Z"
        );
        assert_eq!(
            format_utc(1_704_067_199, 999_999_999, false),
            "2023-12-31T23:59:59.999999Z"
        );
    }

    #[test]
    fn repository_identity_reads_head_and_index_without_git() {
        let root = TestDirectory::new("direct-repository-state");
        let git = root.0.join(".git");
        fs::create_dir_all(git.join("refs/heads")).unwrap();
        let commit = "0123456789abcdef0123456789abcdef01234567";
        fs::write(git.join("HEAD"), b"ref: refs/heads/main\n").unwrap();
        fs::write(git.join("refs/heads/main"), format!("{commit}\n")).unwrap();
        fs::write(git.join("index"), b"DIRC\0bounded-index-fixture").unwrap();

        let state = read_direct_repository_state(&root.0).unwrap();
        assert_eq!(state.get("commit").and_then(Value::as_str), Some(commit));
        assert_eq!(
            state.get("symbolic_head").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(state.get("dirty"), Some(&Value::Null));
        assert_eq!(
            state
                .pointer("/index/sha256")
                .and_then(Value::as_str)
                .map(str::len),
            Some(64)
        );
        assert_eq!(
            state.get("worktree_comparison").and_then(Value::as_str),
            Some("not_performed_without_invoking_or_emulating_git_status")
        );
    }

    #[test]
    fn repository_identity_supports_worktree_pointer_and_packed_ref() {
        let root = TestDirectory::new("direct-worktree-state");
        let common = root.0.join("control");
        let worktree = common.join("worktrees/current");
        fs::create_dir_all(&worktree).unwrap();
        fs::write(
            root.0.join(".git"),
            format!("gitdir: {}\n", worktree.display()),
        )
        .unwrap();
        fs::write(worktree.join("commondir"), "../..\n").unwrap();
        fs::write(worktree.join("HEAD"), "ref: refs/heads/topic\n").unwrap();
        fs::write(worktree.join("index"), b"DIRC\0worktree").unwrap();
        let commit = "fedcba9876543210fedcba9876543210fedcba98";
        fs::write(
            common.join("packed-refs"),
            format!("# pack-refs with: peeled\n{commit} refs/heads/topic\n"),
        )
        .unwrap();

        let state = read_direct_repository_state(&root.0).unwrap();
        assert_eq!(state.get("commit").and_then(Value::as_str), Some(commit));
        assert_eq!(state.get("available").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn explicitly_requested_drafters_require_runtime_proof() {
        let mut parsed = parse_structured_events(&valid_events(), 0, Vec::new());
        let dspark = BTreeMap::from([("K3_DSPARK".into(), "on".into())]);
        assert!(!drafter_contract_errors(&dspark, &parsed).is_empty());
        parsed.runner_config = Some(json!({"dspark_loaded": true}));
        parsed.runner_runtime = Some(json!({
            "dspark": {
                "available": true,
                "proposals": 2,
                "proposal_failures": 0,
                "state_failures": 0,
            }
        }));
        assert!(drafter_contract_errors(&dspark, &parsed).is_empty());

        let qwen = BTreeMap::from([("K3_UAG_DRAFT".into(), "on".into())]);
        parsed.runner_config = Some(json!({"universal_draft_loaded": true}));
        parsed.runner_runtime = Some(json!({
            "universal_draft": {"proposals": 2, "failures": 0}
        }));
        assert!(drafter_contract_errors(&qwen, &parsed).is_empty());
    }

    #[test]
    fn event_reader_rejects_duplicate_json_keys() {
        let root = TestDirectory::new("duplicate-event-key");
        let path = root.0.join("events.jsonl");
        fs::write(
            &path,
            br#"{"schema":"deltafin.run_event.v1","event":"run_start","event":"run_end","wall_time_ns":1,"monotonic_ns":1}
"#,
        )
        .unwrap();
        let (_, errors) = read_events(&path);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate JSON key"))
        );
    }

    fn compile_fake_runner(root: &Path) -> PathBuf {
        const SOURCE: &str = r#"
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::thread;
use std::time::Duration;

fn escape(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn emit(path: &str, kind: &str, fields: &str, clock: u64) {
    let mut file = OpenOptions::new().create(true).append(true).open(path).unwrap();
    writeln!(file, "{{\"event\":\"{}\",\"monotonic_ns\":{},\"schema\":\"deltafin.run_event.v1\",\"wall_time_ns\":{},{} }}", kind, clock, 1000 + clock, fields).unwrap();
    file.sync_all().unwrap();
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    assert_eq!(args.get(1).map(String::as_str), Some("run"));
    let value = |name: &str| {
        let index = args.iter().position(|arg| arg == name).unwrap();
        args[index + 1].clone()
    };
    let prompt = value("--prompt");
    let max_new: u64 = value("--max-new").parse().unwrap();
    let events = value("--events-jsonl");
    let chat = args.iter().any(|arg| arg == "--chat");
    emit(&events, "run_start", &format!("\"chat\":{},\"config\":{{\"device\":\"cpu\",\"eos_token_id\":99}},\"input_token_ids\":[7,8],\"max_new\":{},\"prompt\":\"{}\"", chat, max_new, escape(&prompt)), 1);
    if let Ok(bytes) = env::var("K3_FAKE_STDOUT_BYTES") {
        std::io::stdout().write_all(&vec![b'o'; bytes.parse().unwrap()]).unwrap();
        std::io::stdout().flush().unwrap();
    }
    if let Ok(bytes) = env::var("K3_FAKE_STDERR_BYTES") {
        std::io::stderr().write_all(&vec![b'e'; bytes.parse().unwrap()]).unwrap();
        std::io::stderr().flush().unwrap();
    }
    if let Ok(bytes) = env::var("K3_FAKE_EVENT_BYTES") {
        let mut file = OpenOptions::new().append(true).open(&events).unwrap();
        file.write_all(&vec![b'x'; bytes.parse().unwrap()]).unwrap();
        file.sync_all().unwrap();
    }
    if let Ok(milliseconds) = env::var("K3_FAKE_SLEEP_MS") {
        thread::sleep(Duration::from_millis(milliseconds.parse().unwrap()));
    }
    if env::var("K3_FAKE_FAIL").as_deref() == Ok("1") {
        eprintln!("intentional native failure");
        std::process::exit(7);
    }
    if env::var("K3_FAKE_NO_EVENTS").as_deref() == Ok("1") {
        std::fs::remove_file(&events).unwrap();
        println!("completion: fake completion 13");
        println!("token ids: [10, 11, 12, 13]");
        return;
    }
    let last = if env::var("K3_FAKE_MISMATCH").as_deref() == Ok("1") { 99 } else { 13 };
    emit(&events, "prefill_done", "\"duration_ns\":1250000000,\"emitted_token_ids\":[10]", 2);
    emit(&events, "decode_step", "\"accepted_token_count\":0,\"duration_ns\":2000000000,\"emitted_token_ids\":[11],\"proposed_token_count\":0,\"step\":0", 3);
    emit(&events, "decode_step", &format!("\"accepted_token_count\":1,\"duration_ns\":500000000,\"emitted_token_ids\":[12,{}],\"proposed_token_count\":1,\"step\":1", last), 4);
    emit(&events, "run_end", &format!("\"completion_text\":\"fake completion {}\",\"completion_token_ids\":[10,11,12,{}],\"duration_ns\":3750000000,\"emitted_token_ids\":[10,11,12,{}],\"prompt_token_ids\":[7,8],\"runtime\":{{}},\"status\":\"ok\",\"stop_reason\":\"max_new\"", last, last, last), 5);
    println!("native fake completion");
}
"#;
        let source = root.join("fake_runner.rs");
        let binary = root.join("fake-deltafin");
        fs::write(&source, SOURCE).unwrap();
        let status = Command::new("rustc")
            .args(["--edition=2021", "-O"])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .status()
            .unwrap();
        assert!(status.success());
        binary
    }

    fn fake_options(root: &Path, runner: PathBuf, output_name: &str) -> BenchmarkOptions {
        let mut options = BenchmarkOptions::for_current_executable(root).unwrap();
        options = options.with_test_runner(runner);
        options.prompt = "native prompt".into();
        options.repetitions = 2;
        options.warmup_steps = 1;
        options.output_dir = Some(root.join(output_name));
        options.timeout = Duration::from_secs(5);
        options
    }

    fn fake_arguments(root: &Path, events: &Path) -> Vec<OsString> {
        vec![
            OsString::from("run"),
            OsString::from("--prompt"),
            OsString::from("native prompt"),
            OsString::from("--max-new"),
            OsString::from("4"),
            OsString::from("--events-jsonl"),
            events.as_os_str().to_owned(),
            OsString::from("--model-root"),
            root.as_os_str().to_owned(),
        ]
    }

    #[test]
    fn pinned_runner_survives_replacement_of_the_original_path() {
        let root = TestDirectory::new("pinned-runner");
        let original = compile_fake_runner(&root.0);
        let evidence = root.0.join("evidence");
        fs::DirBuilder::new().mode(0o700).create(&evidence).unwrap();
        let pinned = pin_native_executable(&original, &evidence).unwrap();
        assert_eq!(pinned.sha256.len(), 64);
        assert_eq!(
            fs::metadata(&pinned.executable_path).unwrap().len(),
            pinned.size_bytes
        );

        fs::write(&original, b"#!/bin/sh\nexit 91\n").unwrap();
        fs::set_permissions(&original, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(verify_native_executable(&original).is_err());

        let run_dir = evidence.join("run");
        fs::DirBuilder::new().mode(0o700).create(&run_dir).unwrap();
        let events = run_dir.join("events.jsonl");
        let outcome = execute_runner_with_limits(
            &pinned.executable_path,
            &root.0,
            &fake_arguments(&root.0, &events),
            &BTreeMap::new(),
            &run_dir.join("stdout.log"),
            &run_dir.join("stderr.log"),
            &events,
            Duration::from_secs(5),
            CaptureLimits {
                stdout_bytes: 1024 * 1024,
                stderr_bytes: 1024 * 1024,
                event_bytes: 1024 * 1024,
            },
        )
        .unwrap();
        assert!(outcome.status.is_some_and(|status| status.success()));
        assert!(
            outcome.capture_errors.is_empty(),
            "{:?}",
            outcome.capture_errors
        );
        let (events, errors) = read_events(&events);
        assert!(errors.is_empty(), "{errors:?}");
        assert!(
            parse_structured_events(&events, 0, errors)
                .parse_errors
                .is_empty()
        );

        fs::remove_file(&pinned.executable_path).unwrap();
        fs::write(&pinned.executable_path, b"#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(&pinned.executable_path, fs::Permissions::from_mode(0o500)).unwrap();
        assert!(verify_pinned_runner(&pinned).is_err());
    }

    #[test]
    fn child_capture_limits_stop_stdout_stderr_and_event_growth() {
        let root = TestDirectory::new("capture-limits");
        let runner = compile_fake_runner(&root.0);
        for (label, variable, expected) in [
            ("stdout", "K3_FAKE_STDOUT_BYTES", "runner stdout exceeded"),
            ("stderr", "K3_FAKE_STDERR_BYTES", "runner stderr exceeded"),
            (
                "events",
                "K3_FAKE_EVENT_BYTES",
                "runner event stream reached",
            ),
        ] {
            let run_dir = root.0.join(label);
            fs::DirBuilder::new().mode(0o700).create(&run_dir).unwrap();
            let events = run_dir.join("events.jsonl");
            let outcome = execute_runner_with_limits(
                &runner,
                &root.0,
                &fake_arguments(&root.0, &events),
                &BTreeMap::from([(variable.to_owned(), "8192".to_owned())]),
                &run_dir.join("stdout.log"),
                &run_dir.join("stderr.log"),
                &events,
                Duration::from_secs(5),
                CaptureLimits {
                    stdout_bytes: 1024,
                    stderr_bytes: 1024,
                    event_bytes: 1024,
                },
            )
            .unwrap();
            assert!(
                outcome
                    .capture_errors
                    .iter()
                    .any(|error| error.contains(expected)),
                "{label}: {:?}",
                outcome.capture_errors
            );
            assert!(fs::metadata(run_dir.join("stdout.log")).unwrap().len() <= 1024);
            assert!(fs::metadata(run_dir.join("stderr.log")).unwrap().len() <= 1024);
        }
    }

    #[test]
    fn fake_native_campaign_is_interleaved_exact_and_durable() {
        let root = TestDirectory::new("campaign");
        let runner = compile_fake_runner(&root.0);
        let mut options = fake_options(&root.0, runner, "evidence");
        options.arms = vec![
            BenchmarkArm::new("base", ""),
            BenchmarkArm::new("alternate", "K3_FAKE_LABEL='no shell'"),
        ];
        let report = run_campaign(&options).unwrap();
        assert!(report.succeeded(), "{}", report.summary);
        assert_eq!(report.summary["attempted_runs"], 4);
        assert_eq!(
            report.summary["configs"]["base"]["metrics"]["steady_tps"]["median"],
            4.0
        );
        let runs = fs::read_to_string(report.output_dir.join("runs.jsonl")).unwrap();
        let names = runs
            .lines()
            .map(|line| {
                serde_json::from_str::<Value>(line).unwrap()["config_name"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect::<Vec<_>>();
        assert_eq!(names, ["base", "alternate", "base", "alternate"]);
        assert!(!runs.contains(".py"));
        assert!(report.output_dir.join("campaign.json").is_file());
        assert!(report.output_dir.join("summary.json").is_file());
        assert!(report.output_dir.join("runner.pinned").is_file());
        let campaign: Value = serde_json::from_str(
            &fs::read_to_string(report.output_dir.join("campaign.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            campaign["arguments"]["runner"]["sha256"]
                .as_str()
                .map(str::len),
            Some(64)
        );
        assert!(
            report
                .output_dir
                .join("run-001-base/events.jsonl")
                .is_file()
        );
    }

    #[test]
    fn exact_mismatch_invalidates_campaign_and_stops() {
        let root = TestDirectory::new("mismatch");
        let runner = compile_fake_runner(&root.0);
        let mut options = fake_options(&root.0, runner, "evidence");
        options.arms = vec![
            BenchmarkArm::new("base", ""),
            BenchmarkArm::new("changed", "K3_FAKE_MISMATCH=1"),
        ];
        let report = run_campaign(&options).unwrap();
        assert!(!report.succeeded());
        assert_eq!(report.summary["attempted_runs"], 2);
        assert_eq!(report.summary["stopped_early"], true);
        let result =
            fs::read_to_string(report.output_dir.join("run-002-changed/result.json")).unwrap();
        assert!(result.contains("completion token IDs differ"));
        assert!(result.contains("completion text differs"));
    }

    #[test]
    fn timeout_kills_native_runner_and_preserves_invalid_evidence() {
        let root = TestDirectory::new("timeout");
        let runner = compile_fake_runner(&root.0);
        let mut options = fake_options(&root.0, runner, "evidence");
        options.repetitions = 1;
        options.timeout = Duration::from_millis(25);
        options.arms = vec![BenchmarkArm::new("timeout", "K3_FAKE_SLEEP_MS=500")];
        let report = run_campaign(&options).unwrap();
        assert!(!report.succeeded());
        let result =
            fs::read_to_string(report.output_dir.join("run-001-timeout/result.json")).unwrap();
        assert!(result.contains("runner timed out"));
        assert!(result.contains("\"timed_out\": true"));
    }

    #[test]
    fn nonzero_child_and_missing_events_fail_without_stdout_fallback() {
        let root = TestDirectory::new("failures");
        let runner = compile_fake_runner(&root.0);
        let mut options = fake_options(&root.0, runner.clone(), "failed-evidence");
        options.repetitions = 1;
        options.arms = vec![BenchmarkArm::new("failure", "K3_FAKE_FAIL=1")];
        let report = run_campaign(&options).unwrap();
        assert!(!report.succeeded());
        let result =
            fs::read_to_string(report.output_dir.join("run-001-failure/result.json")).unwrap();
        assert!(result.contains("runner exited with status 7"));
        assert!(result.contains("intentional native failure"));

        let mut options = fake_options(&root.0, runner, "missing-events-evidence");
        options.repetitions = 1;
        options.arms = vec![BenchmarkArm::new("missing-events", "K3_FAKE_NO_EVENTS=1")];
        let report = run_campaign(&options).unwrap();
        assert!(!report.succeeded());
        let result =
            fs::read_to_string(report.output_dir.join("run-001-missing-events/result.json"))
                .unwrap();
        assert!(result.contains("did not produce a structured native event stream"));
        assert!(!result.contains("legacy_stdout"));
        let stdout =
            fs::read_to_string(report.output_dir.join("run-001-missing-events/stdout.log"))
                .unwrap();
        assert!(stdout.contains("token ids: [10, 11, 12, 13]"));
    }
}
