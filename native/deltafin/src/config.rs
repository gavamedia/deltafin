use std::path::PathBuf;

use crate::cli::{RunArgs, ServeArgs};
use crate::dspark_runtime::Mode as DSparkRuntimeMode;
use crate::error::{DeltafinError, Result};
use crate::platform::DeviceRequest;
use crate::program::K3_LAYER_COUNT;
use crate::quality::{QualityPolicy, ResidentWeightAuthority};
use crate::router_trace::RouterTraceMode;

pub const MAX_SPINE_READ_THREADS: usize = 16;
pub const MAX_EXPERT_READ_THREADS: usize = 16;

pub fn parse_spine_read_threads(raw: &str) -> Result<usize> {
    let workers = raw
        .parse::<usize>()
        .map_err(|_| DeltafinError::new("spine read threads must be an integer in 1..=16"))?;
    if !(1..=MAX_SPINE_READ_THREADS).contains(&workers) {
        return Err(DeltafinError::new(
            "spine read threads must be an integer in 1..=16",
        ));
    }
    Ok(workers)
}

pub fn parse_expert_read_threads(raw: &str) -> Result<usize> {
    let workers = raw
        .parse::<usize>()
        .map_err(|_| DeltafinError::new("K3_EXPERT_READ_THREADS must be an integer in 1..=16"))?;
    if !(1..=MAX_EXPERT_READ_THREADS).contains(&workers) {
        return Err(DeltafinError::new(
            "K3_EXPERT_READ_THREADS must be an integer in 1..=16",
        ));
    }
    Ok(workers)
}

fn parse_spine_fd_cache(raw: &str) -> Result<Option<bool>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "1" | "true" | "on" | "yes" | "enabled" => Ok(Some(true)),
        "0" | "false" | "off" | "no" | "disabled" => Ok(Some(false)),
        _ => Err(DeltafinError::new(
            "K3_SPINE_FDCACHE must be auto, 0/1, false/true, or off/on",
        )),
    }
}

fn parse_spine_stream_nocache(raw: &str) -> Result<Option<bool>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "1" | "true" | "on" | "yes" | "enabled" => Ok(Some(true)),
        "0" | "false" | "off" | "no" | "disabled" => Ok(Some(false)),
        _ => Err(DeltafinError::new(
            "K3_SPINE_STREAM_NOCACHE must be auto, 0/1, false/true, or off/on",
        )),
    }
}

fn parse_expert_stream_nocache(raw: &str) -> Result<Option<bool>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "1" | "true" | "on" | "yes" | "enabled" => Ok(Some(true)),
        "0" | "false" | "off" | "no" | "disabled" => Ok(Some(false)),
        _ => Err(DeltafinError::new(
            "K3_EXPERT_STREAM_NOCACHE must be auto, 0/1, false/true, or off/on",
        )),
    }
}

fn parse_spine_resident_gb(raw: &str) -> Result<u64> {
    let gigabytes = if raw.trim().is_empty() {
        0.0
    } else {
        raw.trim().parse::<f64>().map_err(|_| {
            DeltafinError::new("K3_SPINE_RESIDENT_GB must be a finite non-negative number")
        })?
    };
    let bytes = gigabytes * 1_000_000_000.0;
    if !gigabytes.is_finite() || gigabytes < 0.0 || bytes > u64::MAX as f64 {
        return Err(DeltafinError::new(
            "K3_SPINE_RESIDENT_GB must be a finite non-negative number",
        ));
    }
    Ok(bytes as u64)
}

fn parse_expert_pin_gb(raw: &str) -> Result<Option<u64>> {
    if raw.trim().eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    let gigabytes = if raw.trim().is_empty() {
        0.0
    } else {
        raw.trim().parse::<f64>().map_err(|_| {
            DeltafinError::new("K3_EXPERT_PIN_GB must be auto or a finite non-negative number")
        })?
    };
    let bytes = gigabytes * 1_000_000_000.0;
    if !gigabytes.is_finite() || gigabytes < 0.0 || bytes > u64::MAX as f64 {
        return Err(DeltafinError::new(
            "K3_EXPERT_PIN_GB must be auto or a finite non-negative number",
        ));
    }
    Ok(Some(bytes as u64))
}

fn parse_expert_heat(raw: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "enabled" => Ok(true),
        "0" | "false" | "off" | "no" | "disabled" => Ok(false),
        _ => Err(DeltafinError::new(
            "K3_EXPERT_HEAT must be 0/1, false/true, or off/on",
        )),
    }
}

fn parse_expert_early_drain(raw: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "enabled" => Ok(true),
        "0" | "false" | "off" | "no" | "disabled" => Ok(false),
        _ => Err(DeltafinError::new(
            "K3_EXPERT_EARLY_DRAIN must be 0/1, false/true, or off/on",
        )),
    }
}

fn parse_pilot_gate_threshold(raw: &str) -> Result<f64> {
    let threshold = raw.trim().parse::<f64>().map_err(|_| {
        DeltafinError::new("K3_PILOT_GATE_THRESHOLD must be a finite number in [0,1)")
    })?;
    if !threshold.is_finite() || !(0.0..1.0).contains(&threshold) {
        return Err(DeltafinError::new(
            "K3_PILOT_GATE_THRESHOLD must be a finite number in [0,1)",
        ));
    }
    Ok(threshold)
}

pub const MAX_PILOT_GATE_WARMUP: u32 = 100_000;

fn parse_pilot_gate_warmup(raw: &str) -> Result<u32> {
    let warmup = raw
        .trim()
        .parse::<u32>()
        .map_err(|_| DeltafinError::new("K3_PILOT_GATE_WARMUP must be an integer in 1..=100000"))?;
    if !(1..=MAX_PILOT_GATE_WARMUP).contains(&warmup) {
        return Err(DeltafinError::new(
            "K3_PILOT_GATE_WARMUP must be an integer in 1..=100000",
        ));
    }
    Ok(warmup)
}

fn parse_reasoning_effort(raw: &str) -> Result<String> {
    let effort = raw.trim().to_ascii_lowercase();
    match effort.as_str() {
        "low" | "high" | "max" => Ok(effort),
        _ => Err(DeltafinError::new(
            "K3_REASONING_EFFORT must be low, high, or max",
        )),
    }
}

fn parse_provider_resident_layers(raw: &str) -> Result<usize> {
    let layers = raw.parse::<usize>().map_err(|_| {
        DeltafinError::new(format!(
            "K3_PROVIDER_RESIDENT_LAYERS must be an integer in 0..={K3_LAYER_COUNT}"
        ))
    })?;
    if layers > K3_LAYER_COUNT {
        return Err(DeltafinError::new(format!(
            "K3_PROVIDER_RESIDENT_LAYERS must be an integer in 0..={K3_LAYER_COUNT}"
        )));
    }
    Ok(layers)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpineRequest {
    Auto,
    Bf16,
    Int8,
}

impl SpineRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let value = value.unwrap_or("auto").trim().to_ascii_lowercase();
        match value.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "bf16" => Ok(Self::Bf16),
            "int8" => Ok(Self::Int8),
            "mixed" => Err(DeltafinError::new(
                "K3_SPINE=mixed is unsupported because its research codecs change target weights",
            )),
            _ => Err(DeltafinError::new(
                "spine must be auto/int8 (quantized row-int8 default, non-weight-exact) or explicit bf16 (original weights)",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertBackendRequest {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertScale4Request {
    Off,
    Auto,
    Require,
}

impl ExpertScale4Request {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "" | "auto" => Ok(Self::Auto),
            "require" => Ok(Self::Require),
            _ => Err(DeltafinError::new(
                "K3_EXPERT_SCALE4 must be auto, off, or require",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DSparkRequest {
    Off,
    Auto,
    On,
}

/// EAGLE-3.1 hidden-state chain drafter (`k3-draft-eagle3/`). When it loads it
/// takes the proposal slot DSpark would otherwise hold, under the same
/// `K3_DSPARK` mode; `off` leaves the slot to DSpark.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Eagle3Request {
    Off,
    Auto,
    On,
}

impl Eagle3Request {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" | "0" => Ok(Self::Off),
            "" | "auto" => Ok(Self::Auto),
            "on" | "1" => Ok(Self::On),
            _ => Err(DeltafinError::new("K3_EAGLE3 must be off, auto, or on")),
        }
    }
}

/// Admission policy for PILOT speculative expert reads. `On` scores every
/// prediction and gates each layer's disk reads on its trailing measured
/// recall; `Measure` keeps the scoring and reporting but never suppresses or
/// redirects a read (the A/B baseline); `Off` restores the ungoverned legacy
/// scheduler exactly. All three are scheduling-only: the authoritative router
/// still selects every executed expert.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PilotGateRequest {
    Off,
    Measure,
    On,
}

impl PilotGateRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("off").trim().to_ascii_lowercase().as_str() {
            "" | "off" | "0" => Ok(Self::Off),
            "measure" => Ok(Self::Measure),
            "on" | "1" => Ok(Self::On),
            _ => Err(DeltafinError::new(
                "K3_PILOT_GATE must be on, measure, or off",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QwenRequest {
    Off,
    Auto,
    On,
}

/// Product surface which owns the native engine.
///
/// A direct run has one immutable request shape, so automatic draft models
/// that cannot serve that shape must not consume unified memory. The server
/// accepts both chat and raw-completion requests over its lifetime and must
/// retain both automatic proposal paths when they are installed.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RuntimeSurface {
    DirectRun,
    Server,
}

impl QwenRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" => Ok(Self::Off),
            "" | "auto" => Ok(Self::Auto),
            "on" | "1" | "true" | "yes" => Ok(Self::On),
            _ => Err(DeltafinError::new("K3_UAG_DRAFT must be off, auto, or on")),
        }
    }
}

impl DSparkRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" | "0" => Ok(Self::Off),
            "" | "auto" => Ok(Self::Auto),
            "on" | "1" => Ok(Self::On),
            _ => Err(DeltafinError::new("K3_DSPARK must be off, auto, or on")),
        }
    }

    pub const fn runtime_mode(self) -> DSparkRuntimeMode {
        match self {
            Self::Off => DSparkRuntimeMode::Off,
            Self::Auto => DSparkRuntimeMode::Auto,
            Self::On => DSparkRuntimeMode::On,
        }
    }
}

impl ExpertBackendRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let value = value.unwrap_or("auto").trim().to_ascii_lowercase();
        match value.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(DeltafinError::new(
                "expert backend must be auto, cpu, metal, or cuda",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    pub surface: RuntimeSurface,
    pub prompt: String,
    pub max_new: Option<u64>,
    pub chat: bool,
    pub stats: bool,
    /// Verbose per-layer, per-chunk phase breakdown. Independent of `stats`:
    /// the cheap cumulative summary line needs neither this nor per-layer
    /// profile collection.
    pub layer_profile: bool,
    pub events_jsonl: Option<PathBuf>,
    pub router_trace_mode: RouterTraceMode,
    pub router_trace_path: Option<PathBuf>,
    pub model_root: PathBuf,
    pub device: DeviceRequest,
    pub spine: SpineRequest,
    pub spine_read_threads: Option<usize>,
    /// `None` selects the capability-qualified automatic loose-spine policy.
    /// Explicit enablement remains all-or-nothing and fails if the complete
    /// immutable source roster cannot fit with descriptor headroom.
    pub spine_fd_cache: Option<bool>,
    /// `None` is the capability-qualified automatic policy. `Some` preserves
    /// the established K3_SPINE_STREAM_NOCACHE explicit override.
    pub spine_stream_nocache: Option<bool>,
    /// Page-cache policy for streaming expert reads. `None` is automatic:
    /// purge on the memory-tight macOS reference host, keep the kernel file
    /// cache warm elsewhere. `Some` is the K3_EXPERT_STREAM_NOCACHE override.
    pub expert_stream_nocache: Option<bool>,
    pub spine_resident_bytes: Option<u64>,
    /// Diagnostic ceiling for provider-owned layer storage. `None` preserves
    /// automatic safe-prefix selection; every explicit value remains subject
    /// to the same host/device safety envelope.
    pub provider_resident_layers: Option<usize>,
    pub expert_read_threads: Option<usize>,
    /// Extra drives holding byte-identical copies of model files
    /// (`K3_STORAGE_HOMES`). Empty keeps every read on the model root's drive.
    pub storage_homes: Vec<crate::storage_homes::StorageHomeSpec>,
    pub expert_backend: ExpertBackendRequest,
    pub expert_scale4: ExpertScale4Request,
    /// Byte budget for the permanent learned-expert RAM tier. `None` is the
    /// automatic default: sized at startup to exactly the qualifying
    /// histogram roster, capped by live free memory minus a reserved
    /// headroom, and zero (off) while history is thin or memory is tight.
    /// `Some(0)` forces the tier off; `Some(n)` is an explicit budget. Any
    /// budget only marks histogram candidates, which are still promoted
    /// lazily on their first authoritative read.
    pub expert_pin_bytes: Option<u64>,
    /// Whether ordinary runs accumulate the persistent expert-heat histogram.
    /// Recording is advisory and default-on; it never affects routing.
    pub expert_heat: bool,
    /// Whether a decode layer starts each missing expert's own matmul as soon
    /// as that expert's bytes land, instead of waiting for the layer's whole
    /// miss set. Scheduling only: the router still fixes which experts run,
    /// their fp32 weights, and the order they are reduced in, so this cannot
    /// change output. Default off: it measured as a clean null on the
    /// reference host (docs/OPTIMIZATIONS.md), so it stays an explicit option
    /// rather than an unmeasured default in the hot path.
    pub expert_early_drain: bool,
    /// Adaptive admission for PILOT speculative expert reads. Threshold and
    /// warmup are resolved even when the gate is off so a bad value never
    /// silently rides along with a disabled feature.
    pub pilot_gate: PilotGateRequest,
    pub pilot_gate_threshold: f64,
    pub pilot_gate_warmup: u32,
    pub quality: QualityPolicy,
    pub dspark: DSparkRequest,
    pub dspark_max_context: Option<usize>,
    pub dspark_min_auto_speedup: f64,
    pub eagle3: Eagle3Request,
    pub qwen: QwenRequest,
    /// Chat thinking depth (`low`, `high`, or `max`), normalized. `None`
    /// defers to the chat template's own default of `max`; the server's
    /// per-request `reasoning_effort` field overrides this per request.
    pub reasoning_effort: Option<String>,
}

impl RuntimeConfig {
    pub fn resolve<F>(arguments: RunArgs, mut environment: F) -> Result<Self>
    where
        F: FnMut(&str) -> Option<String>,
    {
        let device_value = arguments.device.or_else(|| environment("K3_DEV"));
        let spine_value = arguments.spine.or_else(|| environment("K3_SPINE"));
        let spine_read_threads = arguments
            .spine_read_threads
            .map(Ok)
            .or_else(|| {
                environment("K3_SPINE_READ_THREADS").map(|raw| parse_spine_read_threads(&raw))
            })
            .transpose()?;
        let spine_fd_cache = environment("K3_SPINE_FDCACHE")
            .as_deref()
            .map(parse_spine_fd_cache)
            .transpose()?
            .flatten();
        let spine_stream_nocache = environment("K3_SPINE_STREAM_NOCACHE")
            .as_deref()
            .map(parse_spine_stream_nocache)
            .transpose()?
            .flatten();
        let expert_stream_nocache = environment("K3_EXPERT_STREAM_NOCACHE")
            .as_deref()
            .map(parse_expert_stream_nocache)
            .transpose()?
            .flatten();
        let spine_resident_bytes = environment("K3_SPINE_RESIDENT_GB")
            .as_deref()
            .map(parse_spine_resident_gb)
            .transpose()?;
        let provider_resident_layers = environment("K3_PROVIDER_RESIDENT_LAYERS")
            .as_deref()
            .map(parse_provider_resident_layers)
            .transpose()?;
        let expert_read_threads = environment("K3_EXPERT_READ_THREADS")
            .as_deref()
            .map(parse_expert_read_threads)
            .transpose()?;
        let storage_homes = environment("K3_STORAGE_HOMES")
            .as_deref()
            .map(crate::storage_homes::parse_storage_homes)
            .transpose()?
            .unwrap_or_default();
        let expert_pin_bytes = environment("K3_EXPERT_PIN_GB")
            .as_deref()
            .map(parse_expert_pin_gb)
            .transpose()?
            .flatten();
        let expert_heat = environment("K3_EXPERT_HEAT")
            .as_deref()
            .map(parse_expert_heat)
            .transpose()?
            .unwrap_or(true);
        let expert_early_drain = environment("K3_EXPERT_EARLY_DRAIN")
            .as_deref()
            .map(parse_expert_early_drain)
            .transpose()?
            .unwrap_or(false);
        let backend_value = arguments.expert_backend.or_else(|| environment("K3_MOE"));
        let router_trace_path = arguments
            .router_trace
            .or_else(|| environment("K3_TRACE_PATH").map(PathBuf::from));
        let router_trace_mode_value = arguments
            .router_trace_mode
            .or_else(|| environment("K3_TRACE"));
        let router_trace_mode = if router_trace_mode_value.is_none() && router_trace_path.is_some()
        {
            RouterTraceMode::Buffered
        } else {
            RouterTraceMode::parse(router_trace_mode_value.as_deref())?
        };
        if router_trace_mode == RouterTraceMode::Off && router_trace_path.is_some() {
            return Err(DeltafinError::new(
                "router-trace path was supplied while tracing is explicitly off",
            ));
        }
        let spine = SpineRequest::parse(spine_value.as_deref())?;
        let resident_weights = match spine {
            // The default resident spine is the measured row-int8
            // representation; setup prepares it automatically. The original
            // BF16 checkpoint remains on disk as the conversion source and
            // verification authority, and stays selectable with --spine bf16.
            SpineRequest::Auto | SpineRequest::Int8 => ResidentWeightAuthority::QuantizedInt8,
            SpineRequest::Bf16 => ResidentWeightAuthority::OriginalBf16,
        };
        let quality = QualityPolicy::from_legacy_environment(
            environment("K3_APPROX").as_deref(),
            environment("K3_DTYPE").as_deref(),
            environment("K3_MOE_TOP_K").as_deref(),
        )?
        .with_resident_weights(resident_weights);
        let dspark = DSparkRequest::parse(environment("K3_DSPARK").as_deref())?;
        let eagle3 = Eagle3Request::parse(environment("K3_EAGLE3").as_deref())?;
        let qwen = QwenRequest::parse(environment("K3_UAG_DRAFT").as_deref())?;
        let reasoning_effort = arguments
            .reasoning_effort
            .or_else(|| environment("K3_REASONING_EFFORT"))
            .as_deref()
            .map(parse_reasoning_effort)
            .transpose()?;
        let expert_scale4 = ExpertScale4Request::parse(environment("K3_EXPERT_SCALE4").as_deref())?;
        let pilot_gate = PilotGateRequest::parse(environment("K3_PILOT_GATE").as_deref())?;
        let pilot_gate_threshold = environment("K3_PILOT_GATE_THRESHOLD")
            .as_deref()
            .map(parse_pilot_gate_threshold)
            .transpose()?
            .unwrap_or(0.10);
        let pilot_gate_warmup = environment("K3_PILOT_GATE_WARMUP")
            .as_deref()
            .map(parse_pilot_gate_warmup)
            .transpose()?
            .unwrap_or(16);
        let dspark_max_context = parse_optional_positive_usize(
            environment("K3_DSPARK_MAX_CONTEXT").as_deref(),
            8_192,
            "K3_DSPARK_MAX_CONTEXT",
        )?;
        let dspark_min_auto_speedup = environment("K3_DSPARK_AUTO_MIN_SPEEDUP")
            .as_deref()
            .unwrap_or("0.03")
            .parse::<f64>()
            .map_err(|_| {
                DeltafinError::new("K3_DSPARK_AUTO_MIN_SPEEDUP must be a finite number in [0,1)")
            })?;
        if !dspark_min_auto_speedup.is_finite() || !(0.0..1.0).contains(&dspark_min_auto_speedup) {
            return Err(DeltafinError::new(
                "K3_DSPARK_AUTO_MIN_SPEEDUP must be a finite number in [0,1)",
            ));
        }

        Ok(Self {
            surface: RuntimeSurface::DirectRun,
            prompt: arguments.prompt,
            max_new: arguments.max_new,
            chat: arguments.chat,
            stats: arguments.stats,
            layer_profile: arguments.layer_profile,
            events_jsonl: arguments.events_jsonl,
            router_trace_mode,
            router_trace_path,
            model_root: arguments.model_root,
            device: DeviceRequest::parse(device_value.as_deref())?,
            spine,
            spine_read_threads,
            spine_fd_cache,
            spine_stream_nocache,
            expert_stream_nocache,
            spine_resident_bytes,
            provider_resident_layers,
            expert_read_threads,
            storage_homes,
            expert_backend: ExpertBackendRequest::parse(backend_value.as_deref())?,
            expert_scale4,
            expert_pin_bytes,
            expert_heat,
            expert_early_drain,
            pilot_gate,
            pilot_gate_threshold,
            pilot_gate_warmup,
            quality,
            dspark,
            dspark_max_context,
            dspark_min_auto_speedup,
            eagle3,
            qwen,
            reasoning_effort,
        })
    }

    pub fn from_process(arguments: RunArgs) -> Result<Self> {
        Self::resolve(arguments, |name| std::env::var(name).ok())
    }

    pub fn from_server(arguments: &ServeArgs) -> Result<Self> {
        let mut config = Self::resolve(
            RunArgs {
                prompt: String::new(),
                max_new: None,
                chat: false,
                stats: false,
                layer_profile: false,
                events_jsonl: None,
                router_trace: arguments.router_trace.clone(),
                router_trace_mode: arguments.router_trace_mode.clone(),
                model_root: arguments.model_root.clone(),
                device: arguments.device.clone(),
                spine: arguments.spine.clone(),
                spine_read_threads: arguments.spine_read_threads,
                expert_backend: arguments.expert_backend.clone(),
                reasoning_effort: None,
            },
            |name| std::env::var(name).ok(),
        )?;
        config.surface = RuntimeSurface::Server;
        Ok(config)
    }
}

/// Prints every resolved runtime knob on one line, independent of which of
/// CLI flag, environment variable, or built-in default supplied it — a
/// contributed log or an A/B pair should be auditable without access to the
/// process that produced it.
impl std::fmt::Display for RuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "surface={:?} device={:?} spine={:?} spine_read_threads={} spine_fd_cache={} \
             spine_stream_nocache={} expert_stream_nocache={} spine_resident_bytes={} \
             provider_resident_layers={} \
             expert_read_threads={} storage_homes={} expert_backend={:?} expert_scale4={:?} \
             expert_pin_bytes={} expert_heat={} expert_early_drain={} pilot_gate={:?} \
             pilot_gate_threshold={} pilot_gate_warmup={} quality={:?} \
             dspark={:?} dspark_max_context={} dspark_min_auto_speedup={} eagle3={:?} qwen={:?} \
             reasoning_effort={} router_trace_mode={:?} router_trace_path={} chat={} \
             stats={} layer_profile={} max_new={}",
            self.surface,
            self.device,
            self.spine,
            describe_usize(self.spine_read_threads),
            describe_bool(self.spine_fd_cache),
            describe_bool(self.spine_stream_nocache),
            describe_bool(self.expert_stream_nocache),
            describe_u64(self.spine_resident_bytes),
            describe_usize(self.provider_resident_layers),
            describe_usize(self.expert_read_threads),
            describe_storage_homes(&self.storage_homes),
            self.expert_backend,
            self.expert_scale4,
            describe_u64(self.expert_pin_bytes),
            self.expert_heat,
            self.expert_early_drain,
            self.pilot_gate,
            self.pilot_gate_threshold,
            self.pilot_gate_warmup,
            self.quality,
            self.dspark,
            describe_usize(self.dspark_max_context),
            self.dspark_min_auto_speedup,
            self.eagle3,
            self.qwen,
            self.reasoning_effort.as_deref().unwrap_or("default"),
            self.router_trace_mode,
            describe_path(&self.router_trace_path),
            self.chat,
            self.stats,
            self.layer_profile,
            describe_u64(self.max_new),
        )
    }
}

fn describe_storage_homes(homes: &[crate::storage_homes::StorageHomeSpec]) -> String {
    if homes.iter().all(|home| home.path.is_none()) {
        return "none".to_string();
    }
    homes
        .iter()
        .map(|home| {
            let path = home
                .path
                .as_ref()
                .map_or_else(|| "primary".to_string(), |path| path.display().to_string());
            match home.gbps {
                Some(gbps) => format!("{path}@{gbps}"),
                None => path,
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn describe_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "auto".to_string(), |v| v.to_string())
}

fn describe_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "auto".to_string(), |v| v.to_string())
}

fn describe_bool(value: Option<bool>) -> String {
    value.map_or_else(|| "auto".to_string(), |v| v.to_string())
}

fn describe_path(value: &Option<PathBuf>) -> String {
    value
        .as_ref()
        .map_or_else(|| "none".to_string(), |p| p.display().to_string())
}

fn parse_optional_positive_usize(
    value: Option<&str>,
    default: usize,
    name: &str,
) -> Result<Option<usize>> {
    let parsed = value
        .unwrap_or("")
        .trim()
        .parse::<usize>()
        .or_else(|_| value.is_none().then_some(default).ok_or(()))
        .map_err(|_| DeltafinError::new(format!("{name} must be a non-negative integer")))?;
    Ok((parsed != 0).then_some(parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> RunArgs {
        RunArgs {
            prompt: "hello".into(),
            max_new: Some(4),
            chat: true,
            stats: false,
            layer_profile: false,
            events_jsonl: None,
            router_trace: None,
            router_trace_mode: None,
            model_root: PathBuf::from("."),
            device: None,
            spine: None,
            spine_read_threads: None,
            expert_backend: None,
            reasoning_effort: None,
        }
    }

    #[test]
    fn reasoning_effort_is_normalized_validated_and_cli_wins() {
        let mut explicit = arguments();
        explicit.reasoning_effort = Some(" MAX ".into());
        let config = RuntimeConfig::resolve(explicit, |name| {
            (name == "K3_REASONING_EFFORT").then(|| "low".into())
        })
        .unwrap();
        assert_eq!(config.reasoning_effort.as_deref(), Some("max"));

        let from_environment = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_REASONING_EFFORT").then(|| "high".into())
        })
        .unwrap();
        assert_eq!(from_environment.reasoning_effort.as_deref(), Some("high"));

        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_REASONING_EFFORT").then(|| "medium".into())
            })
            .is_err()
        );

        let unset = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(unset.reasoning_effort, None);
    }

    #[test]
    fn expert_pin_and_heat_knobs_parse_strictly_and_default_safely() {
        let defaults = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(defaults.expert_pin_bytes, None, "unset means automatic sizing");
        assert!(defaults.expert_heat);

        let explicit_auto = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_PIN_GB").then(|| " AUTO ".into())
        })
        .unwrap();
        assert_eq!(explicit_auto.expert_pin_bytes, None);

        let enabled = RuntimeConfig::resolve(arguments(), |name| match name {
            "K3_EXPERT_PIN_GB" => Some("2.5".into()),
            "K3_EXPERT_HEAT" => Some("off".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(enabled.expert_pin_bytes, Some(2_500_000_000));
        assert!(!enabled.expert_heat);

        let zero = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_PIN_GB").then(|| "0".into())
        })
        .unwrap();
        assert_eq!(zero.expert_pin_bytes, Some(0), "explicit zero forces the tier off");

        // A bad value fails closed even though it would disable the feature.
        for bad in ["-1", "nan", "gigabytes", "inf"] {
            assert!(
                RuntimeConfig::resolve(arguments(), |name| {
                    (name == "K3_EXPERT_PIN_GB").then(|| bad.into())
                })
                .is_err(),
                "K3_EXPERT_PIN_GB={bad} should be rejected"
            );
        }
        for bad in ["auto", "2", "maybe", ""] {
            assert!(
                RuntimeConfig::resolve(arguments(), |name| {
                    (name == "K3_EXPERT_HEAT").then(|| bad.into())
                })
                .is_err(),
                "K3_EXPERT_HEAT={bad} should be rejected"
            );
        }
    }

    #[test]
    fn resolved_configuration_display_names_every_knob() {
        let config = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        let line = config.to_string();
        for key in [
            "surface=",
            "device=",
            "spine=",
            "spine_read_threads=",
            "spine_fd_cache=",
            "spine_stream_nocache=",
            "spine_resident_bytes=",
            "provider_resident_layers=",
            "expert_read_threads=",
            "storage_homes=",
            "expert_backend=",
            "expert_scale4=",
            "expert_pin_bytes=",
            "expert_heat=",
            "expert_early_drain=",
            "pilot_gate=",
            "pilot_gate_threshold=",
            "pilot_gate_warmup=",
            "quality=",
            "dspark=",
            "dspark_max_context=",
            "dspark_min_auto_speedup=",
            "eagle3=",
            "qwen=",
            "reasoning_effort=",
            "router_trace_mode=",
            "router_trace_path=",
            "chat=",
            "stats=",
            "layer_profile=",
            "max_new=",
        ] {
            assert!(
                line.contains(key),
                "resolved-config line is missing {key}: {line}"
            );
        }
    }

    #[test]
    fn cli_overrides_environment_without_mutating_process_state() {
        let mut arguments = arguments();
        arguments.device = Some("cpu".into());
        arguments.spine = Some("int8".into());
        let config = RuntimeConfig::resolve(arguments, |name| match name {
            "K3_DEV" => Some("cuda:2".into()),
            "K3_SPINE" => Some("bf16".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.device, DeviceRequest::Cpu);
        assert_eq!(config.spine, SpineRequest::Int8);
        assert_eq!(
            config.quality.resident_weights,
            ResidentWeightAuthority::QuantizedInt8
        );
    }

    #[test]
    fn auto_resolves_the_default_int8_spine_and_baseline_settings() {
        let auto = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(auto.surface, RuntimeSurface::DirectRun);
        assert_eq!(auto.spine, SpineRequest::Auto);
        assert_eq!(auto.spine_fd_cache, None);
        assert_eq!(auto.provider_resident_layers, None);
        assert_eq!(auto.expert_read_threads, None);
        // The default resident spine is the measured row-int8 conversion;
        // setup prepares it automatically, and it is not weight-exact.
        assert!(!auto.quality.is_weight_exact());
        assert_eq!(
            auto.quality.resident_weights,
            ResidentWeightAuthority::QuantizedInt8
        );
        assert_eq!(auto.dspark, DSparkRequest::Auto);
        assert_eq!(auto.dspark_max_context, Some(8_192));
        assert_eq!(auto.dspark_min_auto_speedup, 0.03);
        assert_eq!(auto.eagle3, Eagle3Request::Auto);
        assert_eq!(auto.router_trace_mode, RouterTraceMode::Off);
        assert!(auto.router_trace_path.is_none());
    }

    #[test]
    fn explicit_bf16_preserves_original_checkpoint_authority() {
        let mut explicit = arguments();
        explicit.spine = Some("bf16".into());
        let bf16 = RuntimeConfig::resolve(explicit, |_| None).unwrap();
        assert_eq!(bf16.spine, SpineRequest::Bf16);
        assert!(bf16.quality.is_weight_exact());
        assert_eq!(
            bf16.quality.resident_weights,
            ResidentWeightAuthority::OriginalBf16
        );
    }

    #[test]
    fn spine_reader_cli_override_wins_and_environment_is_bounded() {
        let from_environment = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_SPINE_READ_THREADS").then(|| "6".into())
        })
        .unwrap();
        assert_eq!(from_environment.spine_read_threads, Some(6));

        let mut explicit = arguments();
        explicit.spine_read_threads = Some(3);
        let from_cli = RuntimeConfig::resolve(explicit, |name| {
            (name == "K3_SPINE_READ_THREADS").then(|| "6".into())
        })
        .unwrap();
        assert_eq!(from_cli.spine_read_threads, Some(3));

        for invalid in ["0", "17", "many"] {
            assert!(
                RuntimeConfig::resolve(arguments(), |name| {
                    (name == "K3_SPINE_READ_THREADS").then(|| invalid.into())
                })
                .is_err()
            );
        }
    }

    #[test]
    fn established_spine_cache_environment_overrides_are_preserved() {
        let config = RuntimeConfig::resolve(arguments(), |name| match name {
            "K3_SPINE_STREAM_NOCACHE" => Some("on".into()),
            "K3_SPINE_RESIDENT_GB" => Some("2.1".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.spine_stream_nocache, Some(true));
        assert_eq!(config.spine_resident_bytes, Some(2_100_000_000));

        let auto = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_SPINE_STREAM_NOCACHE").then(|| "auto".into())
        })
        .unwrap();
        assert_eq!(auto.spine_stream_nocache, None);

        for (name, value) in [
            ("K3_SPINE_STREAM_NOCACHE", ""),
            ("K3_SPINE_STREAM_NOCACHE", "sometimes"),
            ("K3_SPINE_RESIDENT_GB", "-1"),
            ("K3_SPINE_RESIDENT_GB", "nan"),
        ] {
            assert!(
                RuntimeConfig::resolve(arguments(), |candidate| {
                    (candidate == name).then(|| value.into())
                })
                .is_err()
            );
        }
    }

    #[test]
    fn expert_stream_cache_policy_is_explicit_or_auto() {
        for (raw, expected) in [
            ("auto", None),
            ("1", Some(true)),
            ("on", Some(true)),
            ("0", Some(false)),
            ("off", Some(false)),
        ] {
            let config = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_EXPERT_STREAM_NOCACHE").then(|| raw.into())
            })
            .unwrap();
            assert_eq!(config.expert_stream_nocache, expected);
        }
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_EXPERT_STREAM_NOCACHE").then(|| "perhaps".into())
            })
            .is_err()
        );
        let unset = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(unset.expert_stream_nocache, None);
    }

    #[test]
    fn loose_spine_descriptor_cache_is_explicit_or_auto() {
        for (raw, expected) in [
            ("auto", None),
            ("1", Some(true)),
            ("on", Some(true)),
            ("0", Some(false)),
            ("off", Some(false)),
        ] {
            let config = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_SPINE_FDCACHE").then(|| raw.into())
            })
            .unwrap();
            assert_eq!(config.spine_fd_cache, expected);
        }
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_SPINE_FDCACHE").then(|| "perhaps".into())
            })
            .is_err()
        );
    }

    #[test]
    fn expert_reader_environment_override_is_bounded() {
        let config = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_READ_THREADS").then(|| "8".into())
        })
        .unwrap();
        assert_eq!(config.expert_read_threads, Some(8));

        for invalid in ["0", "17", "many"] {
            assert!(
                RuntimeConfig::resolve(arguments(), |name| {
                    (name == "K3_EXPERT_READ_THREADS").then(|| invalid.into())
                })
                .is_err()
            );
        }
    }

    #[test]
    fn provider_resident_layer_control_is_env_only_and_bounded() {
        for (raw, expected) in [("0", 0), ("13", 13), ("93", 93)] {
            let config = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_PROVIDER_RESIDENT_LAYERS").then(|| raw.into())
            })
            .unwrap();
            assert_eq!(config.provider_resident_layers, Some(expected));
        }

        for invalid in ["", "-1", "94", "many"] {
            let error = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_PROVIDER_RESIDENT_LAYERS").then(|| invalid.into())
            })
            .unwrap_err();
            assert!(error.to_string().contains("K3_PROVIDER_RESIDENT_LAYERS"));
        }
    }

    #[test]
    fn router_trace_path_enables_buffering_and_explicit_off_fails_closed() {
        let mut explicit_path = arguments();
        explicit_path.router_trace = Some(PathBuf::from("k3-meta/native-routes.jsonl"));
        let config = RuntimeConfig::resolve(explicit_path, |_| None).unwrap();
        assert_eq!(config.router_trace_mode, RouterTraceMode::Buffered);
        assert_eq!(
            config.router_trace_path,
            Some(PathBuf::from("k3-meta/native-routes.jsonl"))
        );

        let mut conflict = arguments();
        conflict.router_trace = Some(PathBuf::from("routes.jsonl"));
        conflict.router_trace_mode = Some("off".into());
        assert!(RuntimeConfig::resolve(conflict, |_| None).is_err());
    }

    #[test]
    fn legacy_quality_environment_is_still_fail_closed() {
        let error = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_MOE_TOP_K").then(|| "15".into())
        })
        .unwrap_err();
        assert!(error.to_string().contains("all 16 routed experts"));
    }

    #[test]
    fn dspark_environment_is_bounded_and_fail_closed() {
        let disabled = RuntimeConfig::resolve(arguments(), |name| match name {
            "K3_DSPARK" => Some("off".into()),
            "K3_DSPARK_MAX_CONTEXT" => Some("0".into()),
            "K3_DSPARK_AUTO_MIN_SPEEDUP" => Some("0.2".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(disabled.dspark, DSparkRequest::Off);
        assert_eq!(disabled.dspark_max_context, None);
        assert_eq!(disabled.dspark_min_auto_speedup, 0.2);

        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_DSPARK_AUTO_MIN_SPEEDUP").then(|| "1".into())
            })
            .unwrap_err()
            .to_string()
            .contains("[0,1)")
        );
    }

    #[test]
    fn eagle3_environment_is_tristate_and_fail_closed() {
        for (raw, expected) in [
            ("off", Eagle3Request::Off),
            ("0", Eagle3Request::Off),
            ("auto", Eagle3Request::Auto),
            ("", Eagle3Request::Auto),
            ("ON", Eagle3Request::On),
            ("1", Eagle3Request::On),
        ] {
            let resolved = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_EAGLE3").then(|| raw.into())
            })
            .unwrap();
            assert_eq!(resolved.eagle3, expected, "{raw:?}");
        }
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_EAGLE3").then(|| "maybe".into())
            })
            .unwrap_err()
            .to_string()
            .contains("K3_EAGLE3")
        );
    }

    #[test]
    fn qwen_environment_preserves_auto_and_explicit_modes() {
        let automatic = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(automatic.qwen, QwenRequest::Auto);
        let enabled = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_UAG_DRAFT").then(|| "true".into())
        })
        .unwrap();
        assert_eq!(enabled.qwen, QwenRequest::On);
        let disabled = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_UAG_DRAFT").then(|| "off".into())
        })
        .unwrap();
        assert_eq!(disabled.qwen, QwenRequest::Off);
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_UAG_DRAFT").then(|| "maybe".into())
            })
            .is_err()
        );
    }

    #[test]
    fn pilot_gate_environment_is_bounded_and_fail_closed() {
        let default = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(default.pilot_gate, PilotGateRequest::Off);
        assert_eq!(default.pilot_gate_threshold, 0.10);
        assert_eq!(default.pilot_gate_warmup, 16);

        for (raw, expected) in [
            ("on", PilotGateRequest::On),
            ("1", PilotGateRequest::On),
            ("", PilotGateRequest::Off),
            ("measure", PilotGateRequest::Measure),
            ("off", PilotGateRequest::Off),
            ("0", PilotGateRequest::Off),
            (" MEASURE ", PilotGateRequest::Measure),
        ] {
            let config = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_PILOT_GATE").then(|| raw.into())
            })
            .unwrap();
            assert_eq!(config.pilot_gate, expected);
        }
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_PILOT_GATE").then(|| "auto".into())
            })
            .unwrap_err()
            .to_string()
            .contains("on, measure, or off")
        );

        let tuned = RuntimeConfig::resolve(arguments(), |name| match name {
            "K3_PILOT_GATE_THRESHOLD" => Some("0.25".into()),
            "K3_PILOT_GATE_WARMUP" => Some("100000".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(tuned.pilot_gate_threshold, 0.25);
        assert_eq!(tuned.pilot_gate_warmup, 100_000);

        // A zero threshold means "never suppress" and stays legal; the knobs
        // fail closed even while the gate itself is off.
        let never = RuntimeConfig::resolve(arguments(), |name| match name {
            "K3_PILOT_GATE" => Some("off".into()),
            "K3_PILOT_GATE_THRESHOLD" => Some("0".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(never.pilot_gate_threshold, 0.0);
        for (name, value) in [
            ("K3_PILOT_GATE_THRESHOLD", "1"),
            ("K3_PILOT_GATE_THRESHOLD", "-0.1"),
            ("K3_PILOT_GATE_THRESHOLD", "nan"),
            ("K3_PILOT_GATE_THRESHOLD", "many"),
            ("K3_PILOT_GATE_WARMUP", "0"),
            ("K3_PILOT_GATE_WARMUP", "100001"),
            ("K3_PILOT_GATE_WARMUP", "-1"),
            ("K3_PILOT_GATE_WARMUP", "many"),
        ] {
            assert!(
                RuntimeConfig::resolve(arguments(), |candidate| {
                    let off = (candidate == "K3_PILOT_GATE").then(|| "off".to_string());
                    off.or_else(|| (candidate == name).then(|| value.into()))
                })
                .is_err(),
                "{name}={value} must fail closed"
            );
        }
    }

    #[test]
    fn scale4_environment_is_explicit_and_fail_closed() {
        let automatic = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(automatic.expert_scale4, ExpertScale4Request::Auto);
        let raw = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_SCALE4").then(|| "off".into())
        })
        .unwrap();
        assert_eq!(raw.expert_scale4, ExpertScale4Request::Off);
        let required = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_SCALE4").then(|| "require".into())
        })
        .unwrap();
        assert_eq!(required.expert_scale4, ExpertScale4Request::Require);
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_EXPERT_SCALE4").then(|| "on".into())
            })
            .unwrap_err()
            .to_string()
            .contains("auto, off, or require")
        );
    }
}
