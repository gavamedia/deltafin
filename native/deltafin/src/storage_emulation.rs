//! Diagnostic storage-device emulation (`K3_STORAGE_EMULATE`).
//!
//! Multi-drive behavior cannot be measured on a host that has one SSD, so
//! this layer lets every positional read be charged to a *virtual* device
//! chosen by path prefix. Each device is a FIFO server with a bandwidth, a
//! per-request latency, an optional thermal-throttle step and an optional
//! failure point (an unplugged enclosure). The real read still happens and
//! still supplies the bytes; the worker then waits until the virtual device
//! would have finished. Emulated devices are therefore never *faster* than
//! the real one, so keep their aggregate below what the host SSD sustains and
//! check the `late` column of the report: late jobs mean the real SSD, not the
//! emulated topology, set the pace.
//!
//! Spec: `NAME=GBPS[,lat=MS][,throttle=GB:FACTOR][,fail=GB]@PREFIX[|PREFIX...]`
//! entries separated by `;`. A read belongs to the device with the longest
//! matching path prefix; reads under no prefix are not emulated. This is a
//! measurement tool only: it never changes which bytes are read.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::error::{DeltafinError, Result};

#[derive(Debug)]
struct EmulatedDevice {
    name: String,
    prefixes: Box<[PathBuf]>,
    bytes_per_second: f64,
    latency: Duration,
    throttle_after_bytes: u64,
    throttle_factor: f64,
    fail_after_bytes: u64,
    state: Mutex<DeviceState>,
}

#[derive(Debug, Default)]
struct DeviceState {
    busy_until: Option<Instant>,
    reserved_bytes: u64,
    jobs: u64,
    late_jobs: u64,
    late: Duration,
    failed_jobs: u64,
}

/// A device slot reserved for one read; [`DeviceEmulator::finish`] waits
/// until the virtual device would have delivered it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reservation {
    device: usize,
    finish: Instant,
}

#[derive(Debug)]
pub(crate) struct DeviceEmulator {
    devices: Box<[EmulatedDevice]>,
}

#[derive(Debug, Clone)]
pub(crate) struct EmulatedDeviceReport {
    pub name: String,
    pub bytes: u64,
    pub jobs: u64,
    pub late_jobs: u64,
    pub late: Duration,
    pub failed_jobs: u64,
}

static GLOBAL: OnceLock<Option<DeviceEmulator>> = OnceLock::new();

impl DeviceEmulator {
    /// The process-wide emulator from `K3_STORAGE_EMULATE`, parsed once. An
    /// invalid spec is reported and disables emulation rather than silently
    /// emulating something other than what was asked.
    pub(crate) fn global() -> Option<&'static Self> {
        GLOBAL
            .get_or_init(|| {
                let spec = std::env::var("K3_STORAGE_EMULATE").ok()?;
                match Self::parse(&spec) {
                    Ok(emulator) => {
                        eprintln!("[storage-emulate] {}", emulator.describe());
                        Some(emulator)
                    }
                    Err(error) => {
                        eprintln!("[storage-emulate] ignoring K3_STORAGE_EMULATE: {error}");
                        None
                    }
                }
            })
            .as_ref()
    }

    pub(crate) fn parse(spec: &str) -> Result<Self> {
        let mut devices = Vec::new();
        for entry in spec
            .split(';')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            let invalid = |why: &str| DeltafinError::new(format!("device `{entry}`: {why}"));
            let (head, prefixes) = entry
                .split_once('@')
                .ok_or_else(|| invalid("expected NAME=GBPS[,...]@PREFIX"))?;
            let (name, params) = head
                .split_once('=')
                .ok_or_else(|| invalid("expected NAME=GBPS"))?;
            let mut params = params.split(',').map(str::trim);
            let gbps = params
                .next()
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| invalid("bandwidth must be a positive GB/s value"))?;
            let mut latency = Duration::ZERO;
            let mut throttle_after_bytes = u64::MAX;
            let mut throttle_factor = 1.0;
            let mut fail_after_bytes = u64::MAX;
            for param in params {
                let (key, value) = param
                    .split_once('=')
                    .ok_or_else(|| invalid("parameters are key=value"))?;
                let number = |text: &str| {
                    text.parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite() && *value >= 0.0)
                        .ok_or_else(|| invalid("parameter values must be non-negative numbers"))
                };
                match key {
                    "lat" => latency = Duration::from_secs_f64(number(value)? / 1e3),
                    "throttle" => {
                        let (after, factor) = value
                            .split_once(':')
                            .ok_or_else(|| invalid("throttle=GB:FACTOR"))?;
                        throttle_after_bytes = (number(after)? * 1e9) as u64;
                        throttle_factor = number(factor)?;
                        if throttle_factor <= 0.0 {
                            return Err(invalid("throttle factor must be positive"));
                        }
                    }
                    "fail" => fail_after_bytes = (number(value)? * 1e9) as u64,
                    _ => return Err(invalid("unknown parameter")),
                }
            }
            let prefixes: Box<[PathBuf]> = prefixes
                .split('|')
                .map(str::trim)
                .filter(|prefix| !prefix.is_empty())
                .map(PathBuf::from)
                .collect();
            if prefixes.is_empty() || prefixes.iter().any(|prefix| !prefix.is_absolute()) {
                return Err(invalid("needs at least one absolute path prefix"));
            }
            devices.push(EmulatedDevice {
                name: name.trim().to_string(),
                prefixes,
                bytes_per_second: gbps * 1e9,
                latency,
                throttle_after_bytes,
                throttle_factor,
                fail_after_bytes,
                state: Mutex::new(DeviceState::default()),
            });
        }
        if devices.is_empty() {
            return Err(DeltafinError::new("no emulated devices"));
        }
        Ok(Self {
            devices: devices.into_boxed_slice(),
        })
    }

    fn describe(&self) -> String {
        self.devices
            .iter()
            .map(|device| {
                format!(
                    "{}={:.2}GB/s lat={:.2}ms{}{} [{}]",
                    device.name,
                    device.bytes_per_second / 1e9,
                    device.latency.as_secs_f64() * 1e3,
                    if device.throttle_after_bytes == u64::MAX {
                        String::new()
                    } else {
                        format!(
                            " throttle@{:.0}GB x{}",
                            device.throttle_after_bytes as f64 / 1e9,
                            device.throttle_factor
                        )
                    },
                    if device.fail_after_bytes == u64::MAX {
                        String::new()
                    } else {
                        format!(" fail@{:.0}GB", device.fail_after_bytes as f64 / 1e9)
                    },
                    device
                        .prefixes
                        .iter()
                        .map(|prefix| prefix.display().to_string())
                        .collect::<Vec<_>>()
                        .join("|"),
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Longest matching prefix wins, so a mirror nested under the primary
    /// tree can still be its own device.
    pub(crate) fn device_for(&self, path: &Path) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (index, device) in self.devices.iter().enumerate() {
            for prefix in device.prefixes.iter() {
                if path.starts_with(prefix) {
                    let depth = prefix.components().count();
                    if best.is_none_or(|(_, best_depth)| depth > best_depth) {
                        best = Some((index, depth));
                    }
                }
            }
        }
        best.map(|(index, _)| index)
    }

    /// Reserve `bytes` on the device owning `path`, starting now. Returns
    /// `Ok(None)` for unemulated paths and an I/O error once the device has
    /// passed its failure point.
    pub(crate) fn reserve(&self, path: &Path, bytes: u64) -> io::Result<Option<Reservation>> {
        let Some(index) = self.device_for(path) else {
            return Ok(None);
        };
        let device = &self.devices[index];
        let now = Instant::now();
        let mut state = device.state.lock().unwrap();
        if state.reserved_bytes >= device.fail_after_bytes {
            state.failed_jobs += 1;
            return Err(io::Error::from_raw_os_error(libc::EIO));
        }
        let rate = if state.reserved_bytes >= device.throttle_after_bytes {
            device.bytes_per_second * device.throttle_factor
        } else {
            device.bytes_per_second
        };
        let start = state.busy_until.map_or(now, |busy| busy.max(now));
        let busy_until = start + Duration::from_secs_f64(bytes as f64 / rate);
        state.busy_until = Some(busy_until);
        state.reserved_bytes += bytes;
        state.jobs += 1;
        Ok(Some(Reservation {
            device: index,
            finish: busy_until + device.latency,
        }))
    }

    /// Hold the calling worker until the reserved virtual completion.
    pub(crate) fn finish(&self, reservation: Reservation) {
        let now = Instant::now();
        if now < reservation.finish {
            std::thread::sleep(reservation.finish - now);
        } else {
            let mut state = self.devices[reservation.device].state.lock().unwrap();
            state.late_jobs += 1;
            state.late += now - reservation.finish;
        }
    }

    pub(crate) fn report(&self) -> Vec<EmulatedDeviceReport> {
        self.devices
            .iter()
            .map(|device| {
                let state = device.state.lock().unwrap();
                EmulatedDeviceReport {
                    name: device.name.clone(),
                    bytes: state.reserved_bytes,
                    jobs: state.jobs,
                    late_jobs: state.late_jobs,
                    late: state.late,
                    failed_jobs: state.failed_jobs,
                }
            })
            .collect()
    }

    pub(crate) fn print_report(&self) {
        for device in self.report() {
            eprintln!(
                "[storage-emulate] {}: {:.2} GB in {} reads, late {} ({:.2}s), failed {}",
                device.name,
                device.bytes as f64 / 1e9,
                device.jobs,
                device.late_jobs,
                device.late.as_secs_f64(),
                device.failed_jobs,
            );
        }
    }
}

/// Reserve on the process-wide emulator, if one is configured.
pub(crate) fn begin(path: &Path, bytes: usize) -> io::Result<Option<Reservation>> {
    match DeviceEmulator::global() {
        Some(emulator) => emulator.reserve(path, bytes as u64),
        None => Ok(None),
    }
}

pub(crate) fn end(reservation: Option<Reservation>) {
    if let (Some(reservation), Some(emulator)) = (reservation, DeviceEmulator::global()) {
        emulator.finish(reservation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_devices_and_matches_the_longest_prefix() {
        let emulator = DeviceEmulator::parse(
            "internal=2,lat=0.1@/model; ext=1,throttle=10:0.5,fail=20@/model/mirror|/Volumes/x",
        )
        .unwrap();
        assert_eq!(
            emulator.device_for(Path::new("/model/k3-experts/L1-E0.bin")),
            Some(0)
        );
        assert_eq!(
            emulator.device_for(Path::new("/model/mirror/L1-E0.bin")),
            Some(1)
        );
        assert_eq!(emulator.device_for(Path::new("/Volumes/x/a")), Some(1));
        assert_eq!(emulator.device_for(Path::new("/elsewhere")), None);
        assert_eq!(emulator.devices[1].throttle_after_bytes, 10_000_000_000);
        assert!(DeviceEmulator::parse("x=0@/a").is_err());
        assert!(DeviceEmulator::parse("x=1@relative").is_err());
        assert!(DeviceEmulator::parse("x=1,bogus=2@/a").is_err());
        assert!(DeviceEmulator::parse("").is_err());
    }

    #[test]
    fn a_device_serves_reservations_in_fifo_order_at_its_bandwidth() {
        let emulator = DeviceEmulator::parse("d=1@/d").unwrap();
        let started = Instant::now();
        let first = emulator
            .reserve(Path::new("/d/a"), 20_000_000)
            .unwrap()
            .unwrap();
        let second = emulator
            .reserve(Path::new("/d/b"), 20_000_000)
            .unwrap()
            .unwrap();
        // 20 MB at 1 GB/s is 20 ms each, back to back.
        let gap = second.finish.duration_since(first.finish);
        assert!((gap.as_secs_f64() - 0.020).abs() < 0.002, "{gap:?}");
        emulator.finish(second);
        assert!(started.elapsed() >= Duration::from_millis(39));
        assert!(emulator.reserve(Path::new("/other"), 1).unwrap().is_none());
    }

    #[test]
    fn throttle_and_failure_points_apply_after_their_byte_counts() {
        let emulator = DeviceEmulator::parse("d=1,throttle=0.01:0.5,fail=0.03@/d").unwrap();
        let first = emulator
            .reserve(Path::new("/d/a"), 10_000_000)
            .unwrap()
            .unwrap();
        let second = emulator
            .reserve(Path::new("/d/a"), 10_000_000)
            .unwrap()
            .unwrap();
        let gap = second.finish.duration_since(first.finish).as_secs_f64();
        assert!(
            (gap - 0.020).abs() < 0.002,
            "throttled to half speed: {gap}"
        );
        let _ = emulator.reserve(Path::new("/d/a"), 10_000_000).unwrap();
        let error = emulator.reserve(Path::new("/d/a"), 1).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EIO));
        assert_eq!(emulator.report()[0].failed_jobs, 1);
    }
}
