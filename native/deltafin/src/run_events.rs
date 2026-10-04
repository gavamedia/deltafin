//! Exclusive, flushed JSONL evidence for native Deltafin runs.
//!
//! The schema deliberately matches the mature benchmark reader. Human output
//! is not an evidence API: it rounds timings and cannot represent a verified
//! speculative transaction that emits more than one authoritative token.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::error::{DeltafinError, Result};
use crate::sys::fs::{self as sys_fs, Open};

pub const EVENT_SCHEMA: &str = "deltafin.run_event.v1";
pub(crate) const MAX_EVENT_STREAM_BYTES: u64 = 128 * 1024 * 1024;

static MONOTONIC_EPOCH: OnceLock<Instant> = OnceLock::new();

pub struct RunEventLog {
    path: Option<PathBuf>,
    file: Option<File>,
    byte_limit: u64,
    bytes_written: u64,
    buffer: Vec<u8>,
}

impl RunEventLog {
    pub fn open(path: Option<&Path>) -> Result<Self> {
        Self::open_with_limit(path, MAX_EVENT_STREAM_BYTES)
    }

    fn open_with_limit(path: Option<&Path>, byte_limit: u64) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self {
                path: None,
                file: None,
                byte_limit,
                bytes_written: 0,
                buffer: Vec::new(),
            });
        };
        if byte_limit == 0 {
            return Err(DeltafinError::new(
                "native run event byte limit must be positive",
            ));
        }
        if path.as_os_str().is_empty() {
            return Err(DeltafinError::new(
                "--events-jsonl requires a non-empty file path",
            ));
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|error| io_error("create event directory", parent, error))?;
            let metadata = sys_fs::lstat(parent)
                .map_err(|error| io_error("inspect event directory", parent, error))?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(DeltafinError::new(format!(
                    "event output parent is not a real directory: {}",
                    parent.display()
                )));
            }
        }
        let file = Open::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .no_follow()
            .open(path)
            .map_err(|error| io_error("exclusively create event stream", path, error))?;
        Ok(Self {
            path: Some(path.to_path_buf()),
            file: Some(file),
            byte_limit,
            bytes_written: 0,
            buffer: Vec::new(),
        })
    }

    #[cfg(test)]
    pub const fn enabled(&self) -> bool {
        self.file.is_some()
    }

    pub fn emit_run_start(
        &mut self,
        prompt: &str,
        chat: bool,
        maximum_new: u64,
        input_token_ids: &[u32],
        config: Value,
    ) -> Result<()> {
        self.emit(
            "run_start",
            [
                ("prompt", Value::String(prompt.to_owned())),
                ("chat", Value::Bool(chat)),
                ("max_new", Value::from(maximum_new)),
                ("input_token_ids", token_ids(input_token_ids)),
                ("config", config),
            ],
        )
    }

    #[cfg(test)]
    pub fn emit_prefill_done(&mut self, duration_ns: u64, emitted_token_ids: &[u32]) -> Result<()> {
        self.emit_prefill_done_with_profile(duration_ns, emitted_token_ids, None)
    }

    /// Emit one buffered first-token profile rather than flushing one event
    /// per layer. The profile is prepared after model execution, so JSON
    /// serialization and file I/O cannot perturb any measured phase.
    pub fn emit_prefill_done_with_profile(
        &mut self,
        duration_ns: u64,
        emitted_token_ids: &[u32],
        target_phase_profile: Option<Value>,
    ) -> Result<()> {
        if target_phase_profile
            .as_ref()
            .is_some_and(|profile| !profile.is_object())
        {
            return Err(DeltafinError::new(
                "native target phase profile must be a JSON object",
            ));
        }
        let mut fields = vec![
            ("duration_ns", Value::from(duration_ns)),
            ("emitted_token_ids", token_ids(emitted_token_ids)),
        ];
        if let Some(profile) = target_phase_profile {
            fields.push(("target_phase_profile", profile));
        }
        self.emit("prefill_done", fields)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_decode_step(
        &mut self,
        step: u64,
        duration_ns: u64,
        emitted_token_ids: &[u32],
        proposal_candidate_count: u64,
        proposed_token_count: u64,
        accepted_token_count: u64,
        proposal_memory_rejected: bool,
    ) -> Result<()> {
        if emitted_token_ids.is_empty() {
            return Err(DeltafinError::new(
                "native decode_step must contain an authoritative token",
            ));
        }
        if accepted_token_count > proposed_token_count {
            return Err(DeltafinError::new(
                "native decode_step accepted more draft tokens than were proposed",
            ));
        }
        if proposed_token_count > proposal_candidate_count {
            return Err(DeltafinError::new(
                "native decode_step submitted more drafts than its proposal sources produced",
            ));
        }
        if proposal_memory_rejected
            && (proposal_candidate_count == 0
                || proposed_token_count != 0
                || accepted_token_count != 0)
        {
            return Err(DeltafinError::new(
                "native decode_step memory rejection fields are inconsistent",
            ));
        }
        if accepted_token_count > emitted_token_ids.len() as u64 {
            return Err(DeltafinError::new(
                "native decode_step accepted more draft tokens than it emitted",
            ));
        }
        let maximum_emitted = accepted_token_count.checked_add(1).ok_or_else(|| {
            DeltafinError::new("native decode_step accepted draft count overflowed its tail")
        })?;
        if emitted_token_ids.len() as u64 > maximum_emitted {
            return Err(DeltafinError::new(
                "native decode_step emitted more than its accepted draft prefix plus target tail",
            ));
        }
        self.emit(
            "decode_step",
            [
                ("step", Value::from(step)),
                ("duration_ns", Value::from(duration_ns)),
                ("emitted_token_ids", token_ids(emitted_token_ids)),
                (
                    "proposal_candidate_count",
                    Value::from(proposal_candidate_count),
                ),
                ("proposed_token_count", Value::from(proposed_token_count)),
                ("accepted_token_count", Value::from(accepted_token_count)),
                (
                    "proposal_memory_rejected",
                    Value::Bool(proposal_memory_rejected),
                ),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_run_end(
        &mut self,
        status: &str,
        duration_ns: u64,
        prompt_token_ids: &[u32],
        emitted_token_ids: &[u32],
        completion_token_ids: &[u32],
        stop_reason: &str,
        completion_text: String,
        runtime: Value,
    ) -> Result<()> {
        self.emit(
            "run_end",
            [
                ("status", Value::String(status.to_owned())),
                ("duration_ns", Value::from(duration_ns)),
                ("prompt_token_ids", token_ids(prompt_token_ids)),
                ("emitted_token_ids", token_ids(emitted_token_ids)),
                ("completion_token_ids", token_ids(completion_token_ids)),
                ("stop_reason", Value::String(stop_reason.to_owned())),
                ("completion_text", Value::String(completion_text)),
                ("runtime", runtime),
            ],
        )
    }

    pub fn emit_run_error(
        &mut self,
        phase: &str,
        error_type: &str,
        message: &str,
        duration_ns: u64,
    ) -> Result<()> {
        self.emit(
            "run_error",
            [
                ("phase", Value::String(phase.to_owned())),
                ("error_type", Value::String(error_type.to_owned())),
                ("message", Value::String(message.to_owned())),
                ("duration_ns", Value::from(duration_ns)),
            ],
        )
    }

    pub fn emit<I, K>(&mut self, event: &str, fields: I) -> Result<()>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        let event_path = self.path.clone();
        let Some(file) = self.file.as_mut() else {
            return Ok(());
        };
        if event.is_empty() {
            return Err(DeltafinError::new("native run event name may not be empty"));
        }
        let mut record = BTreeMap::new();
        for (key, value) in fields {
            let key = key.into();
            if matches!(
                key.as_str(),
                "schema" | "event" | "wall_time_ns" | "monotonic_ns"
            ) {
                return Err(DeltafinError::new(format!(
                    "native run event field {key:?} is reserved"
                )));
            }
            if record.insert(key.clone(), value).is_some() {
                return Err(DeltafinError::new(format!(
                    "native run event field {key:?} was supplied twice"
                )));
            }
        }
        let wall_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| DeltafinError::new("system clock precedes the Unix epoch"))?
            .as_nanos();
        let monotonic_ns = MONOTONIC_EPOCH
            .get_or_init(Instant::now)
            .elapsed()
            .as_nanos();
        let wall_time_ns = u64::try_from(wall_time_ns)
            .map_err(|_| DeltafinError::new("wall clock nanoseconds exceed JSON integer range"))?;
        let monotonic_ns = u64::try_from(monotonic_ns).map_err(|_| {
            DeltafinError::new("monotonic clock nanoseconds exceed JSON integer range")
        })?;
        record.insert("event".into(), Value::String(event.into()));
        record.insert("monotonic_ns".into(), Value::from(monotonic_ns));
        record.insert("schema".into(), Value::String(EVENT_SCHEMA.into()));
        record.insert("wall_time_ns".into(), Value::from(wall_time_ns));
        self.buffer.clear();
        serde_json::to_writer(&mut self.buffer, &record).map_err(|error| {
            DeltafinError::new(format!("serialize native run event {event:?}: {error}"))
        })?;
        let record_bytes = u64::try_from(self.buffer.len())
            .ok()
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| DeltafinError::new("native run event size overflowed u64"))?;
        let prospective_bytes = self
            .bytes_written
            .checked_add(record_bytes)
            .ok_or_else(|| DeltafinError::new("native run event stream size overflowed u64"))?;
        if prospective_bytes >= self.byte_limit {
            return Err(DeltafinError::new(format!(
                "native run event stream would reach or exceed the {}-byte hard limit",
                self.byte_limit
            )));
        }
        file.write_all(&self.buffer).map_err(|error| {
            event_io_error("write native run event", event_path.as_deref(), error)
        })?;
        file.write_all(b"\n").map_err(|error| {
            event_io_error("write native run event", event_path.as_deref(), error)
        })?;
        file.flush().map_err(|error| {
            event_io_error("flush native run event", event_path.as_deref(), error)
        })?;
        self.bytes_written = prospective_bytes;
        Ok(())
    }
}

fn token_ids(ids: &[u32]) -> Value {
    Value::Array(ids.iter().copied().map(Value::from).collect())
}

impl Drop for RunEventLog {
    fn drop(&mut self) {
        if let Some(file) = self.file.as_mut() {
            let _ = file.flush();
        }
    }
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

fn event_io_error(operation: &str, path: Option<&Path>, error: io::Error) -> DeltafinError {
    match path {
        Some(path) => io_error(operation, path, error),
        None => DeltafinError::new(format!("{operation}: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "deltafin-run-events-{}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn exclusive_flushed_events_use_the_established_schema() {
        let root = TestDirectory::new();
        let path = root.0.join("events.jsonl");
        let mut events = RunEventLog::open(Some(&path)).unwrap();
        events
            .emit(
                "run_start",
                [("input_token_ids", serde_json::json!([1, 2, 3]))],
            )
            .unwrap();
        let raw = fs::read_to_string(&path).unwrap();
        let row: Value = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(row["schema"], EVENT_SCHEMA);
        assert_eq!(row["event"], "run_start");
        assert_eq!(row["input_token_ids"], serde_json::json!([1, 2, 3]));
        assert!(row["wall_time_ns"].as_u64().is_some());
        assert!(row["monotonic_ns"].as_u64().is_some());
        assert!(RunEventLog::open(Some(&path)).is_err());
    }

    #[test]
    fn disabled_log_is_allocation_free_and_reserved_fields_fail_closed() {
        let mut disabled = RunEventLog::open(None).unwrap();
        assert!(!disabled.enabled());
        disabled
            .emit("ignored", std::iter::empty::<(&str, Value)>())
            .unwrap();

        let root = TestDirectory::new();
        let path = root.0.join("events.jsonl");
        let mut enabled = RunEventLog::open(Some(&path)).unwrap();
        assert!(enabled.emit("bad", [("schema", Value::Null)]).is_err());
    }

    #[test]
    fn prefill_profile_is_one_flushed_object_and_rejects_non_objects() {
        let root = TestDirectory::new();
        let path = root.0.join("profile-events.jsonl");
        let mut events = RunEventLog::open(Some(&path)).unwrap();
        let profile = serde_json::json!({
            "schema": "deltafin.target_phase_profile.v1",
            "totals": {"sequence_total_ns": 123},
            "layers": [{"layer": 0, "layer_total_ns": 100}],
        });
        events
            .emit_prefill_done_with_profile(150, &[17], Some(profile.clone()))
            .unwrap();
        let raw = fs::read_to_string(&path).unwrap();
        assert_eq!(raw.lines().count(), 1);
        let event: Value = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(event["event"], "prefill_done");
        assert_eq!(event["target_phase_profile"], profile);

        assert!(
            events
                .emit_prefill_done_with_profile(200, &[18], Some(serde_json::json!([])))
                .unwrap_err()
                .to_string()
                .contains("must be a JSON object")
        );
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 1);
    }

    #[test]
    fn typed_run_contract_is_ordered_complete_and_flushed() {
        let root = TestDirectory::new();
        let path = root.0.join("typed-events.jsonl");
        let mut events = RunEventLog::open(Some(&path)).unwrap();
        events
            .emit_run_start(
                "hello",
                false,
                4,
                &[10, 11],
                serde_json::json!({"device": "cpu", "eos_token_id": 99}),
            )
            .unwrap();
        events.emit_prefill_done(20, &[12]).unwrap();
        events
            .emit_decode_step(0, 30, &[13, 14], 2, 2, 1, false)
            .unwrap();
        events
            .emit_run_end(
                "ok",
                60,
                &[10, 11],
                &[12, 13, 14],
                &[12, 13, 14],
                "max_new",
                "answer".to_owned(),
                serde_json::json!({"qwen_state": "off"}),
            )
            .unwrap();

        let rows = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0]["event"], "run_start");
        assert_eq!(rows[1]["event"], "prefill_done");
        assert_eq!(rows[2]["event"], "decode_step");
        assert_eq!(rows[3]["event"], "run_end");
        assert_eq!(rows[0]["prompt"], "hello");
        assert_eq!(rows[0]["config"]["eos_token_id"], 99);
        assert_eq!(rows[1]["emitted_token_ids"], serde_json::json!([12]));
        assert_eq!(rows[2]["proposed_token_count"], 2);
        assert_eq!(rows[2]["proposal_candidate_count"], 2);
        assert_eq!(rows[2]["accepted_token_count"], 1);
        assert_eq!(rows[2]["proposal_memory_rejected"], false);
        assert_eq!(
            rows[3]["emitted_token_ids"],
            serde_json::json!([12, 13, 14])
        );
        assert_eq!(rows[3]["completion_text"], "answer");
        assert!(rows.iter().all(|row| row["schema"] == EVENT_SCHEMA));
    }

    #[test]
    fn decode_step_rejects_empty_or_impossible_authority_counts() {
        let root = TestDirectory::new();
        let path = root.0.join("invalid-events.jsonl");
        let mut events = RunEventLog::open(Some(&path)).unwrap();
        assert!(events.emit_decode_step(0, 1, &[], 0, 0, 0, false).is_err());
        assert!(events.emit_decode_step(0, 1, &[1], 0, 0, 1, false).is_err());
        assert!(events.emit_decode_step(0, 1, &[1], 2, 2, 2, false).is_err());
        assert!(
            events
                .emit_decode_step(0, 1, &[1, 2], 0, 0, 0, false)
                .is_err()
        );
        assert!(events.emit_decode_step(0, 1, &[1], 0, 0, 0, true).is_err());
        assert!(events.emit_decode_step(0, 1, &[1], 2, 1, 0, true).is_err());
        assert_eq!(fs::read_to_string(path).unwrap(), "");
    }

    #[test]
    fn decode_step_exposes_a_candidate_rejected_by_live_memory_admission() {
        let root = TestDirectory::new();
        let path = root.0.join("memory-rejected-events.jsonl");
        let mut events = RunEventLog::open(Some(&path)).unwrap();
        events.emit_decode_step(0, 1, &[1], 2, 0, 0, true).unwrap();
        let raw = fs::read_to_string(path).unwrap();
        let row: Value = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(row["proposal_candidate_count"], 2);
        assert_eq!(row["proposed_token_count"], 0);
        assert_eq!(row["proposal_memory_rejected"], true);
    }

    #[test]
    fn event_stream_limit_rejects_before_writing_a_partial_record() {
        let root = TestDirectory::new();
        let path = root.0.join("bounded-events.jsonl");
        let mut events = RunEventLog::open_with_limit(Some(&path), 1).unwrap();
        let error = events
            .emit("run_start", [("input_token_ids", serde_json::json!([1]))])
            .unwrap_err()
            .to_string();
        assert!(error.contains("hard limit"), "{error}");
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    }
}
