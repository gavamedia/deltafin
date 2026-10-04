use std::ffi::OsString;
use std::path::PathBuf;

use crate::error::{DeltafinError, Result};

pub const HELP: &str = "\
Deltafin native runtime\n\
\n\
Usage:\n\
  deltafin [run] [OPTIONS]\n\
  deltafin serve [OPTIONS]\n\
  deltafin benchmark [OPTIONS]\n\
  deltafin doctor [--runtime-only | --model-root PATH]\n\
  deltafin upgrade\n\
  deltafin setup [--dry-run | --check] [--full | --stream] [--include-qwen] [--workers N] [--model-root PATH]\n\
  deltafin setup-dspark [--check | --audit-only] [--destination PATH]\n\
  deltafin setup-k3 (--meta-only | --check) [--model-root PATH]\n\
  deltafin setup-qwen [--check] [--model-root PATH]\n\
  deltafin fetch-weights [OPTIONS]\n\
  deltafin warm-expert-cache [OPTIONS]\n\
  deltafin populate-storage-home --dir PATH [--budget-gb GB] [--verify-existing] [--workers N] [--model-root PATH]\n\
  deltafin convert-spine-int8 [OPTIONS]\n\
  deltafin convert-experts-scale4 [OPTIONS]\n\
  deltafin pack-spine [OPTIONS]\n\
\n\
Run options:\n\
  --prompt TEXT          Text to continue (default: The capital of France is)\n\
  --max-new N            Stop after at most N generated tokens\n\
  --chat                 Apply the K3 chat template\n\
  --reasoning-effort L   Chat thinking depth: low, high, or max (K3_REASONING_EFFORT fallback; template default max)\n\
  --stats                Show live cumulative performance statistics\n\
  --layer-profile        Also print a per-layer, per-chunk phase breakdown (verbose: ~93 lines/token; independent of --stats)\n\
  --events-jsonl PATH    Write the benchmark event stream\n\
  --router-trace PATH    Append native expert-route JSONL (relative to model root)\n\
  --router-trace-mode M  off, buffered, or sync (path alone implies buffered)\n\
  --model-root PATH      Deltafin install/model root (default: current directory)\n\
  --device DEVICE        auto, cpu, mps, cuda, or cuda:N (K3_DEV fallback)\n\
  --spine FORMAT         auto/int8 = quantized row-int8 default; bf16 = original weights, explicit\n\
  --spine-read-threads N Override bounded spine readers (1..=16; K3_SPINE_READ_THREADS fallback)\n\
  --expert-backend NAME  auto, cpu, metal, or cuda (K3_MOE fallback)\n\
  -h, --help             Show this help\n\
  -V, --version          Show the binary version\n\
\n\
Serve options:\n\
  --host ADDRESS         Listen address (default: 127.0.0.1)\n\
  --port N               Listen port (default: 8000)\n\
  --max-tokens N         Per-response server ceiling (default: 1000000)\n\
  --queue N              Concurrent requests allowed to wait for the generation slot\n\
                         (default: 0 = immediate 429; max: 64; K3_SERVER_QUEUE fallback)\n\
  --max-request-bytes N  Bounded JSON request size (default: 134217728; max: 1073741824)\n\
  --response-memo-entries N  Exact-response cache entries (default: 32; 0 disables)\n\
  --response-memo-bytes N    Exact-response cache bytes (default: 67108864; max: 1073741824; 0 disables)\n\
  --router-trace PATH    Append native expert-route JSONL (relative to model root)\n\
  --router-trace-mode M  off, buffered, or sync (path alone implies buffered)\n\
  --model-root PATH      Deltafin install/model root (default: current directory)\n\
  --device DEVICE        auto, cpu, mps, cuda, or cuda:N (K3_DEV fallback)\n\
  --spine FORMAT         auto/int8 = quantized row-int8 default; bf16 = original weights, explicit\n\
  --spine-read-threads N Override bounded spine readers (1..=16; K3_SPINE_READ_THREADS fallback)\n\
  --expert-backend NAME  auto, cpu, metal, or cuda (K3_MOE fallback)\n\
\n\
Benchmark options:\n\
  --prompt TEXT          Text to continue (default: The capital of France is)\n\
  --chat                 Apply the K3 chat template\n\
  --max-new N            Generated-token bound per run (default: 4)\n\
  --reps N               Interleaved repetitions per arm (default: 3)\n\
  --warmup-steps N       Exclude this many decode transactions from steady state\n\
  --timeout SECONDS      Kill a stalled native run after this duration (default: 3600)\n\
  --configs ENV [...]    Environment deltas; quote each KEY=VALUE list\n\
  --names NAME [...]     Optional unique names matching --configs\n\
  --arm NAME=ENV         Add a named environment delta; repeat for A/B arms\n\
  --expect-token-ids IDS Exact JSON or comma-separated completion-token oracle\n\
  --expect-text TEXT     Exact completion-text oracle\n\
  --output-dir PATH      New evidence directory (default: bench-results/native-...)\n\
  --keep-going           Continue after an invalid run\n\
  --model-root PATH      Deltafin install/model root (default: current directory)\n\
\n\
Doctor options:\n\
  --runtime-only         Validate the host, binary, providers and native canaries without opening model data\n\
  --model-root PATH      Deltafin install/model root (default: DELTAFIN_ROOT or current directory)\n\
\n\
Pack-spine options:\n\
  --model-root PATH      Deltafin install/model root (default: current directory)\n\
  --spine FORMAT         auto/int8 = quantized row-int8 default; bf16 = original weights, explicit\n\
  --output PATH          Pack directory (default: ...-packs-bf16; int8: ...-packs)\n\
  --layer N              Build or verify only zero-based layer N\n\
  --verify-only          Verify existing packs without creating missing ones\n\
\n\
Setup-dspark options:\n\
  --check                Verify an installed checkpoint without network access\n\
  --audit-only           Audit pinned remote metadata/header without tensor payload\n\
  --destination PATH     Install/check directory (default: DELTAFIN_ROOT/k3-draft-dspark)\n\
\n\
Setup options:\n\
  --full                 Require the complete local expert pool (fastest inference)\n\
  --stream               Install the resident spine; fetch exact experts on demand\n\
                         With neither flag, choose full when disk allows, else stream\n\
  --dry-run              Plan exact downloads and peak disk use without network or writes\n\
  --check                Audit that the capacity-selected installation is complete\n\
  --include-qwen         Also install/check the optional raw-completion Qwen assistants\n\
  --workers N            Parallel HTTPS transfers (default: 8, maximum: 16)\n\
  --model-root PATH      Install root (default: DELTAFIN_ROOT or current directory)\n\
\n\
Setup-k3 options:\n\
  --meta-only            Install only pinned inert K3 metadata and tensor inventory\n\
  --check                Verify installed K3 metadata without network access\n\
  --model-root PATH      Install root (default: DELTAFIN_ROOT or current directory)\n\
\n\
Setup-qwen options:\n\
  --check                Verify both optional Qwen assistants without network access\n\
  --model-root PATH      Install root (default: DELTAFIN_ROOT or current directory)\n\
\n\
Fetch-weights options:\n\
  --dry-run              Authenticate metadata and print work without writing or downloading\n\
  --spine-only           Install only the canonical non-expert resident tensors\n\
  --experts-only         Install only all 92 x 896 canonical routed experts\n\
  --layers SPEC          Install selected expert layers, e.g. 1-40,45,92 (implies experts-only)\n\
  --workers N            Parallel HTTPS transfers (default: 8, maximum: 16)\n\
  --model-root PATH      Install root (default: DELTAFIN_ROOT or current directory)\n\
\n\
Warm-expert-cache options:\n\
  --trace PATH           Router JSONL trace; repeat, or auto-discover k3-meta traces\n\
  --fetch N              Explicitly fetch at most N highest-ranked missing experts\n\
  --convert-npz          Losslessly migrate legacy six-member NPZ experts to raw .bin\n\
  --convert-workers N    Bounded parallel NPZ converters (default: 4, maximum: 16)\n\
  --convert-limit N      Convert only the first N numerically sorted NPZ experts\n\
  --convert-throttle-ms N  Pause N milliseconds after each converted/reused expert\n\
  --keep-npz             Keep exact legacy NPZ sources after durable .bin publication\n\
  --show N               Print at most N ranked candidates (default: 20)\n\
  --json                 Print the bounded plan as JSON\n\
  --workers N            Parallel HTTPS fetch transfers (default: 8, maximum: 16)\n\
  --cache PATH           Inspect/convert a noncanonical expert cache\n\
  --model-root PATH      Install root (default: DELTAFIN_ROOT or current directory)\n\
\n\
Convert-spine-int8 options:\n\
  --model-root PATH      Root containing k3-resident (default: current directory)\n\
  --checkpoint-rows N    Durable resume interval in rows (default: 128)\n\
  --no-resume            Refuse existing output state instead of authenticating it\n\
                         This conversion is quantized and not weight-exact\n\
\n\
Convert-experts-scale4 options:\n\
  --model-root PATH      Root containing k3-experts (default: current directory)\n\
  --source-root PATH     Override the canonical raw-expert directory\n\
  --output-root PATH     Override the lossless scale-sidecar directory\n\
  --workers N            Parallel layer converters (default: 4, maximum: 16)\n\
  --no-resume            Refuse existing layer sidecars instead of authenticating them\n";

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct DoctorArgs {
    pub model_root: Option<PathBuf>,
    pub runtime_only: bool,
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct SetupDSparkArgs {
    pub check: bool,
    pub audit_only: bool,
    pub destination: Option<PathBuf>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SetupArgs {
    pub dry_run: bool,
    pub check: bool,
    pub mode: crate::one_shot_setup::SetupMode,
    pub include_qwen: bool,
    pub workers: usize,
    pub model_root: Option<PathBuf>,
}

impl Default for SetupArgs {
    fn default() -> Self {
        Self {
            dry_run: false,
            check: false,
            mode: crate::one_shot_setup::SetupMode::Auto,
            include_qwen: false,
            workers: crate::weight_fetch::DEFAULT_PARALLEL_TRANSFERS,
            model_root: None,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SetupK3Args {
    pub meta_only: bool,
    pub check: bool,
    pub model_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SetupQwenArgs {
    pub check: bool,
    pub model_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BenchmarkArmArgs {
    pub name: String,
    pub environment_spec: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BenchmarkArgs {
    pub prompt: String,
    pub chat: bool,
    pub max_new_tokens: u64,
    pub repetitions: usize,
    pub warmup_steps: usize,
    pub timeout_ns: u64,
    pub output: Option<PathBuf>,
    pub arms: Vec<BenchmarkArmArgs>,
    pub expected_token_ids: Option<String>,
    pub expected_text: Option<String>,
    pub keep_going: bool,
    pub model_root: PathBuf,
}

impl Default for BenchmarkArgs {
    fn default() -> Self {
        Self {
            prompt: "The capital of France is".into(),
            chat: false,
            max_new_tokens: 4,
            repetitions: 3,
            warmup_steps: 0,
            timeout_ns: 3_600_000_000_000,
            output: None,
            arms: Vec::new(),
            expected_token_ids: None,
            expected_text: None,
            keep_going: false,
            model_root: PathBuf::from("."),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FetchWeightsArgs {
    pub dry_run: bool,
    pub spine_only: bool,
    pub experts_only: bool,
    pub layers: Option<Vec<u32>>,
    pub workers: usize,
    pub model_root: Option<PathBuf>,
}

/// Copy the int8 spine and the hottest experts onto another drive for
/// `K3_STORAGE_HOMES`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PopulateStorageHomeArgs {
    pub model_root: Option<PathBuf>,
    pub dir: PathBuf,
    pub budget_bytes: Option<u64>,
    pub verify_existing: bool,
    pub workers: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WarmExpertCacheArgs {
    pub model_root: Option<PathBuf>,
    pub cache: Option<PathBuf>,
    pub traces: Vec<PathBuf>,
    pub fetch: usize,
    pub show: usize,
    pub json: bool,
    pub workers: usize,
    pub convert_npz: bool,
    pub convert_workers: usize,
    pub convert_limit: Option<usize>,
    pub convert_throttle_ms: u64,
    pub keep_npz: bool,
}

impl Default for WarmExpertCacheArgs {
    fn default() -> Self {
        Self {
            model_root: None,
            cache: None,
            traces: Vec::new(),
            fetch: 0,
            show: 20,
            json: false,
            workers: crate::weight_fetch::DEFAULT_PARALLEL_TRANSFERS,
            convert_npz: false,
            convert_workers: crate::legacy_npz::DEFAULT_CONVERSION_WORKERS,
            convert_limit: None,
            convert_throttle_ms: 0,
            keep_npz: false,
        }
    }
}

impl Default for FetchWeightsArgs {
    fn default() -> Self {
        Self {
            dry_run: false,
            spine_only: false,
            experts_only: false,
            layers: None,
            workers: crate::weight_fetch::DEFAULT_PARALLEL_TRANSFERS,
            model_root: None,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConvertExpertsScale4Args {
    pub model_root: PathBuf,
    pub source_root: Option<PathBuf>,
    pub output_root: Option<PathBuf>,
    pub workers: usize,
    pub resume: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConvertSpineInt8Args {
    pub model_root: PathBuf,
    pub checkpoint_rows: usize,
    pub resume: bool,
}

impl Default for ConvertSpineInt8Args {
    fn default() -> Self {
        Self {
            model_root: PathBuf::from("."),
            checkpoint_rows: crate::spine_int8::DEFAULT_CHECKPOINT_ROWS,
            resume: true,
        }
    }
}

impl Default for ConvertExpertsScale4Args {
    fn default() -> Self {
        Self {
            model_root: PathBuf::from("."),
            source_root: None,
            output_root: None,
            workers: crate::expert_scale4::convert::DEFAULT_CONVERSION_WORKERS,
            resume: true,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RunArgs {
    pub prompt: String,
    pub max_new: Option<u64>,
    pub chat: bool,
    pub reasoning_effort: Option<String>,
    pub stats: bool,
    /// Per-layer, per-chunk phase breakdown (93 lines/token). Independent of
    /// `stats`: the cheap cumulative summary line needs neither this nor
    /// per-layer profile collection at all.
    pub layer_profile: bool,
    pub events_jsonl: Option<PathBuf>,
    pub router_trace: Option<PathBuf>,
    pub router_trace_mode: Option<String>,
    pub model_root: PathBuf,
    pub device: Option<String>,
    pub spine: Option<String>,
    pub spine_read_threads: Option<usize>,
    pub expert_backend: Option<String>,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            prompt: "The capital of France is".into(),
            max_new: None,
            chat: false,
            reasoning_effort: None,
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
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackSpineArgs {
    pub model_root: PathBuf,
    pub spine: Option<String>,
    pub output: Option<PathBuf>,
    pub layer: Option<u32>,
    pub verify_only: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ServeArgs {
    pub host: String,
    pub port: u16,
    pub max_tokens: usize,
    /// `--queue`; `None` defers to the `K3_SERVER_QUEUE` environment fallback
    /// resolved at server startup.
    pub queue_slots: Option<usize>,
    pub max_request_bytes: usize,
    pub response_memo_entries: usize,
    pub response_memo_bytes: usize,
    pub router_trace: Option<PathBuf>,
    pub router_trace_mode: Option<String>,
    pub model_root: PathBuf,
    pub device: Option<String>,
    pub spine: Option<String>,
    pub spine_read_threads: Option<usize>,
    pub expert_backend: Option<String>,
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8000,
            max_tokens: 1_000_000,
            queue_slots: None,
            max_request_bytes: crate::openai::DEFAULT_MAX_REQUEST_BODY_BYTES,
            response_memo_entries: crate::openai::DEFAULT_RESPONSE_MEMO_ENTRIES,
            response_memo_bytes: crate::openai::DEFAULT_RESPONSE_MEMO_BYTES,
            router_trace: None,
            router_trace_mode: None,
            model_root: PathBuf::from("."),
            device: None,
            spine: None,
            spine_read_threads: None,
            expert_backend: None,
        }
    }
}

impl Default for PackSpineArgs {
    fn default() -> Self {
        Self {
            model_root: PathBuf::from("."),
            spine: None,
            output: None,
            layer: None,
            verify_only: false,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Command {
    Run(RunArgs),
    Serve(ServeArgs),
    Benchmark(BenchmarkArgs),
    Setup(SetupArgs),
    PackSpine(PackSpineArgs),
    SetupDSpark(SetupDSparkArgs),
    SetupK3(SetupK3Args),
    SetupQwen(SetupQwenArgs),
    FetchWeights(FetchWeightsArgs),
    WarmExpertCache(WarmExpertCacheArgs),
    PopulateStorageHome(PopulateStorageHomeArgs),
    ConvertSpineInt8(ConvertSpineInt8Args),
    ConvertExpertsScale4(ConvertExpertsScale4Args),
    Doctor(DoctorArgs),
    Upgrade,
    Help,
    Version,
}

pub fn parse<I, S>(arguments: I) -> Result<Command>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut arguments = arguments
        .into_iter()
        .map(Into::into)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| DeltafinError::new("command-line arguments must be valid UTF-8"))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter();
    let _program = arguments.next();
    let mut values: Vec<String> = arguments.collect();

    // Every public subcommand owns the same complete help document for now.
    // Resolve the conventional `deltafin <command> --help` shape before the
    // command-specific parser so help is always successful, including for
    // commands whose ordinary argument contract requires another option.
    if matches!(
        values.as_slice(),
        [subcommand, help]
            if is_subcommand(subcommand) && matches!(help.as_str(), "-h" | "--help")
    ) {
        return Ok(Command::Help);
    }

    if values.first().is_some_and(|value| value == "run") {
        values.remove(0);
    } else if values.first().is_some_and(|value| value == "doctor") {
        values.remove(0);
        return parse_doctor(&values).map(Command::Doctor);
    } else if values.first().is_some_and(|value| value == "upgrade") {
        if values.len() != 1 {
            return Err(DeltafinError::new("upgrade accepts no arguments"));
        }
        return Ok(Command::Upgrade);
    } else if values.first().is_some_and(|value| value == "serve") {
        values.remove(0);
        return parse_serve(&values).map(Command::Serve);
    } else if values.first().is_some_and(|value| value == "benchmark") {
        values.remove(0);
        return parse_benchmark(&values).map(Command::Benchmark);
    } else if values.first().is_some_and(|value| value == "setup") {
        values.remove(0);
        return parse_setup(&values).map(Command::Setup);
    } else if values.first().is_some_and(|value| value == "pack-spine") {
        values.remove(0);
        return parse_pack_spine(&values).map(Command::PackSpine);
    } else if values.first().is_some_and(|value| value == "setup-dspark") {
        values.remove(0);
        return parse_setup_dspark(&values).map(Command::SetupDSpark);
    } else if values.first().is_some_and(|value| value == "setup-k3") {
        values.remove(0);
        return parse_setup_k3(&values).map(Command::SetupK3);
    } else if values.first().is_some_and(|value| value == "setup-qwen") {
        values.remove(0);
        return parse_setup_qwen(&values).map(Command::SetupQwen);
    } else if values.first().is_some_and(|value| value == "fetch-weights") {
        values.remove(0);
        return parse_fetch_weights(&values).map(Command::FetchWeights);
    } else if values
        .first()
        .is_some_and(|value| value == "warm-expert-cache")
    {
        values.remove(0);
        return parse_warm_expert_cache(&values).map(Command::WarmExpertCache);
    } else if values
        .first()
        .is_some_and(|value| value == "populate-storage-home")
    {
        values.remove(0);
        return parse_populate_storage_home(&values).map(Command::PopulateStorageHome);
    } else if values
        .first()
        .is_some_and(|value| value == "convert-spine-int8")
    {
        values.remove(0);
        return parse_convert_spine_int8(&values).map(Command::ConvertSpineInt8);
    } else if values
        .first()
        .is_some_and(|value| value == "convert-experts-scale4")
    {
        values.remove(0);
        return parse_convert_experts_scale4(&values).map(Command::ConvertExpertsScale4);
    }

    if values.len() == 1 && matches!(values[0].as_str(), "-h" | "--help") {
        return Ok(Command::Help);
    }
    if values.len() == 1 && matches!(values[0].as_str(), "-V" | "--version") {
        return Ok(Command::Version);
    }

    let mut run = RunArgs::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--chat" => run.chat = true,
            "--reasoning-effort" => {
                run.reasoning_effort = Some(take_value(&values, &mut index, option)?);
            }
            "--stats" => run.stats = true,
            "--layer-profile" => run.layer_profile = true,
            "--prompt" => run.prompt = take_value(&values, &mut index, option)?,
            "--max-new" => {
                let raw = take_value(&values, &mut index, option)?;
                run.max_new =
                    Some(raw.parse::<u64>().map_err(|_| {
                        DeltafinError::new("--max-new must be a non-negative integer")
                    })?);
            }
            "--events-jsonl" => {
                run.events_jsonl = Some(take_value(&values, &mut index, option)?.into());
            }
            "--router-trace" => {
                run.router_trace = Some(take_value(&values, &mut index, option)?.into());
            }
            "--router-trace-mode" => {
                run.router_trace_mode = Some(take_value(&values, &mut index, option)?);
            }
            "--model-root" => {
                run.model_root = take_value(&values, &mut index, option)?.into();
            }
            "--device" => run.device = Some(take_value(&values, &mut index, option)?),
            "--spine" => run.spine = Some(take_value(&values, &mut index, option)?),
            "--spine-read-threads" => {
                run.spine_read_threads = Some(crate::config::parse_spine_read_threads(
                    &take_value(&values, &mut index, option)?,
                )?);
            }
            "--expert-backend" => {
                run.expert_backend = Some(take_value(&values, &mut index, option)?);
            }
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    require_continuable_prompt("--prompt", &run.prompt, run.chat)?;
    Ok(Command::Run(run))
}

/// Refuse a raw completion with no text to continue, here at argument
/// parsing, instead of after the engine has spent minutes loading the model
/// and then rejected it (the HTTP server refuses the same input with a 400).
/// A chat turn is rendered through the template first, so it is never empty.
fn require_continuable_prompt(option: &str, prompt: &str, chat: bool) -> Result<()> {
    if prompt.is_empty() && !chat {
        return Err(DeltafinError::new(format!(
            "{option} must not be empty: a raw completion needs text to continue (use --chat for a chat turn)"
        )));
    }
    Ok(())
}

fn is_subcommand(value: &str) -> bool {
    matches!(
        value,
        "run"
            | "serve"
            | "benchmark"
            | "doctor"
            | "upgrade"
            | "setup"
            | "setup-dspark"
            | "setup-k3"
            | "setup-qwen"
            | "fetch-weights"
            | "warm-expert-cache"
            | "populate-storage-home"
            | "convert-spine-int8"
            | "convert-experts-scale4"
            | "pack-spine"
    )
}

fn parse_doctor(values: &[String]) -> Result<DoctorArgs> {
    let mut doctor = DoctorArgs::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--runtime-only" => doctor.runtime_only = true,
            "--model-root" => {
                doctor.model_root = Some(take_value(values, &mut index, option)?.into());
            }
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown doctor argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if doctor.runtime_only && doctor.model_root.is_some() {
        return Err(DeltafinError::new(
            "--runtime-only and --model-root are mutually exclusive",
        ));
    }
    Ok(doctor)
}

fn parse_benchmark(values: &[String]) -> Result<BenchmarkArgs> {
    let mut benchmark = BenchmarkArgs::default();
    let mut configs = Vec::new();
    let mut names = Vec::new();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--prompt" => benchmark.prompt = take_value(values, &mut index, option)?,
            "--chat" => benchmark.chat = true,
            "--max-new" | "--tokens" => {
                benchmark.max_new_tokens =
                    positive_u64(&take_value(values, &mut index, option)?, "--max-new")?;
            }
            "--repetitions" | "--reps" => {
                let value =
                    positive_u64(&take_value(values, &mut index, option)?, "--repetitions")?;
                benchmark.repetitions = usize::try_from(value)
                    .map_err(|_| DeltafinError::new("--repetitions does not fit this host"))?;
            }
            "--warmup-steps" => {
                let raw = take_value(values, &mut index, option)?;
                benchmark.warmup_steps = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new("--warmup-steps must be a non-negative integer")
                })?;
            }
            "--timeout-seconds" | "--timeout" => {
                benchmark.timeout_ns =
                    positive_duration_ns(&take_value(values, &mut index, option)?, "--timeout")?;
            }
            "--output" | "--output-dir" => {
                benchmark.output = Some(take_value(values, &mut index, option)?.into());
            }
            "--config" => configs.push(take_value(values, &mut index, option)?),
            "--name" => names.push(take_value(values, &mut index, option)?),
            "--configs" => {
                collect_benchmark_values(values, &mut index, option, &mut configs)?;
                continue;
            }
            "--names" => {
                collect_benchmark_values(values, &mut index, option, &mut names)?;
                continue;
            }
            "--arm" => {
                let raw = take_value(values, &mut index, option)?;
                let (name, environment_spec) = raw.split_once('=').ok_or_else(|| {
                    DeltafinError::new("--arm must be NAME=ENV; ENV may be empty")
                })?;
                if name.is_empty()
                    || !name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                {
                    return Err(DeltafinError::new(
                        "benchmark arm names may contain only ASCII letters, digits, '_' and '-'",
                    ));
                }
                if benchmark.arms.iter().any(|arm| arm.name == name) {
                    return Err(DeltafinError::new(format!(
                        "benchmark arm {name:?} was supplied twice"
                    )));
                }
                crate::benchmark::parse_environment_delta(environment_spec)?;
                benchmark.arms.push(BenchmarkArmArgs {
                    name: name.to_owned(),
                    environment_spec: environment_spec.to_owned(),
                });
            }
            "--expect-token-ids" => {
                benchmark.expected_token_ids = Some(take_value(values, &mut index, option)?);
            }
            "--expect-text" => {
                benchmark.expected_text = Some(take_value(values, &mut index, option)?);
            }
            "--keep-going" => benchmark.keep_going = true,
            "--model-root" => {
                benchmark.model_root = take_value(values, &mut index, option)?.into();
            }
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown benchmark argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if !benchmark.arms.is_empty() && (!configs.is_empty() || !names.is_empty()) {
        return Err(DeltafinError::new(
            "--arm cannot be combined with --configs/--names",
        ));
    }
    if !configs.is_empty() || !names.is_empty() {
        if configs.is_empty() {
            return Err(DeltafinError::new("--names requires --configs"));
        }
        if names.is_empty() {
            names = configs
                .iter()
                .map(|config| {
                    if config.is_empty() {
                        "defaults".to_owned()
                    } else {
                        config.clone()
                    }
                })
                .collect();
        }
        if names.len() != configs.len() {
            return Err(DeltafinError::new(
                "--names must have exactly one entry per --configs entry",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        benchmark.arms = names
            .into_iter()
            .zip(configs)
            .map(|(name, environment_spec)| {
                if name.trim().is_empty() || !seen.insert(name.clone()) {
                    return Err(DeltafinError::new(
                        "benchmark configuration names must be non-empty and unique",
                    ));
                }
                crate::benchmark::parse_environment_delta(&environment_spec)?;
                Ok(BenchmarkArmArgs {
                    name,
                    environment_spec,
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }
    require_continuable_prompt("--prompt", &benchmark.prompt, benchmark.chat)?;
    Ok(benchmark)
}

fn collect_benchmark_values(
    values: &[String],
    index: &mut usize,
    option: &str,
    output: &mut Vec<String>,
) -> Result<()> {
    *index += 1;
    let start = *index;
    while *index < values.len() && !values[*index].starts_with("--") {
        output.push(values[*index].clone());
        *index += 1;
    }
    if *index == start {
        return Err(DeltafinError::new(format!(
            "{option} requires at least one value"
        )));
    }
    Ok(())
}

fn positive_u64(raw: &str, option: &str) -> Result<u64> {
    let value = raw
        .parse::<u64>()
        .map_err(|_| DeltafinError::new(format!("{option} must be a positive integer")))?;
    if value == 0 {
        return Err(DeltafinError::new(format!(
            "{option} must be a positive integer"
        )));
    }
    Ok(value)
}

fn positive_duration_ns(raw: &str, option: &str) -> Result<u64> {
    let seconds = raw
        .parse::<f64>()
        .map_err(|_| DeltafinError::new(format!("{option} must be a positive finite number")))?;
    let duration = std::time::Duration::try_from_secs_f64(seconds)
        .map_err(|_| DeltafinError::new(format!("{option} must be a positive finite number")))?;
    if duration.is_zero() {
        return Err(DeltafinError::new(format!(
            "{option} must be a positive finite number"
        )));
    }
    u64::try_from(duration.as_nanos())
        .map_err(|_| DeltafinError::new(format!("{option} is too large")))
}

fn parse_setup(values: &[String]) -> Result<SetupArgs> {
    let mut setup = SetupArgs::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--dry-run" => setup.dry_run = true,
            "--check" => setup.check = true,
            "--full" => {
                if setup.mode == crate::one_shot_setup::SetupMode::Stream {
                    return Err(DeltafinError::new(
                        "--full and --stream are mutually exclusive",
                    ));
                }
                setup.mode = crate::one_shot_setup::SetupMode::Full;
            }
            "--stream" => {
                if setup.mode == crate::one_shot_setup::SetupMode::Full {
                    return Err(DeltafinError::new(
                        "--full and --stream are mutually exclusive",
                    ));
                }
                setup.mode = crate::one_shot_setup::SetupMode::Stream;
            }
            "--include-qwen" => setup.include_qwen = true,
            "--workers" => {
                let raw = take_value(values, &mut index, option)?;
                setup.workers = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::weight_fetch::MAX_PARALLEL_TRANSFERS
                    ))
                })?;
                if !(1..=crate::weight_fetch::MAX_PARALLEL_TRANSFERS).contains(&setup.workers) {
                    return Err(DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::weight_fetch::MAX_PARALLEL_TRANSFERS
                    )));
                }
            }
            "--model-root" => {
                setup.model_root = Some(take_value(values, &mut index, option)?.into());
            }
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown setup argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if setup.dry_run && setup.check {
        return Err(DeltafinError::new(
            "--dry-run and --check are mutually exclusive",
        ));
    }
    Ok(setup)
}

fn parse_convert_experts_scale4(values: &[String]) -> Result<ConvertExpertsScale4Args> {
    let mut convert = ConvertExpertsScale4Args::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--model-root" => {
                convert.model_root = take_value(values, &mut index, option)?.into();
            }
            "--source-root" => {
                convert.source_root = Some(take_value(values, &mut index, option)?.into());
            }
            "--output-root" => {
                convert.output_root = Some(take_value(values, &mut index, option)?.into());
            }
            "--workers" => {
                let raw = take_value(values, &mut index, option)?;
                convert.workers = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::expert_scale4::convert::MAX_CONVERSION_WORKERS
                    ))
                })?;
                if !(1..=crate::expert_scale4::convert::MAX_CONVERSION_WORKERS)
                    .contains(&convert.workers)
                {
                    return Err(DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::expert_scale4::convert::MAX_CONVERSION_WORKERS
                    )));
                }
            }
            "--no-resume" => convert.resume = false,
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown convert-experts-scale4 argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    Ok(convert)
}

fn parse_convert_spine_int8(values: &[String]) -> Result<ConvertSpineInt8Args> {
    let mut convert = ConvertSpineInt8Args::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--model-root" => {
                convert.model_root = take_value(values, &mut index, option)?.into();
            }
            "--checkpoint-rows" => {
                let raw = take_value(values, &mut index, option)?;
                convert.checkpoint_rows = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--checkpoint-rows must be an integer in 1..={}",
                        crate::spine_int8::MAX_CHECKPOINT_ROWS
                    ))
                })?;
                if !(1..=crate::spine_int8::MAX_CHECKPOINT_ROWS).contains(&convert.checkpoint_rows)
                {
                    return Err(DeltafinError::new(format!(
                        "--checkpoint-rows must be an integer in 1..={}",
                        crate::spine_int8::MAX_CHECKPOINT_ROWS
                    )));
                }
            }
            "--no-resume" => convert.resume = false,
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown convert-spine-int8 argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    Ok(convert)
}

fn parse_fetch_weights(values: &[String]) -> Result<FetchWeightsArgs> {
    let mut fetch = FetchWeightsArgs::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--dry-run" => fetch.dry_run = true,
            "--spine-only" => fetch.spine_only = true,
            "--experts-only" => fetch.experts_only = true,
            "--layers" => {
                if fetch.layers.is_some() {
                    return Err(DeltafinError::new("--layers may be supplied only once"));
                }
                let raw = take_value(values, &mut index, option)?;
                fetch.layers = Some(parse_expert_layers(&raw)?);
            }
            "--workers" => {
                let raw = take_value(values, &mut index, option)?;
                fetch.workers = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::weight_fetch::MAX_PARALLEL_TRANSFERS
                    ))
                })?;
                if !(1..=crate::weight_fetch::MAX_PARALLEL_TRANSFERS).contains(&fetch.workers) {
                    return Err(DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::weight_fetch::MAX_PARALLEL_TRANSFERS
                    )));
                }
            }
            "--model-root" => {
                fetch.model_root = Some(take_value(values, &mut index, option)?.into());
            }
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown fetch-weights argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if fetch.spine_only && fetch.experts_only {
        return Err(DeltafinError::new(
            "--spine-only and --experts-only are mutually exclusive",
        ));
    }
    if fetch.spine_only && fetch.layers.is_some() {
        return Err(DeltafinError::new(
            "--layers selects routed experts and cannot be combined with --spine-only",
        ));
    }
    Ok(fetch)
}

fn parse_expert_layers(raw: &str) -> Result<Vec<u32>> {
    use std::collections::BTreeSet;

    let mut selected = BTreeSet::new();
    if raw.is_empty() {
        return Err(DeltafinError::new(
            "--layers must be a comma-separated layer/range specification",
        ));
    }
    for item in raw.split(',') {
        if item.is_empty() || item.trim() != item {
            return Err(DeltafinError::new(
                "--layers must use canonical form such as 1-40,45,92 (no spaces)",
            ));
        }
        let (first, last) = match item.split_once('-') {
            Some((first, last)) if !first.is_empty() && !last.is_empty() && !last.contains('-') => {
                (parse_expert_layer(first)?, parse_expert_layer(last)?)
            }
            Some(_) => {
                return Err(DeltafinError::new(format!(
                    "invalid --layers item {item:?}; expected N or A-B"
                )));
            }
            None => {
                let layer = parse_expert_layer(item)?;
                (layer, layer)
            }
        };
        if first > last {
            return Err(DeltafinError::new(format!(
                "invalid descending --layers range {item:?}"
            )));
        }
        selected.extend(first..=last);
    }
    if selected.is_empty() {
        return Err(DeltafinError::new("--layers selected no routed layers"));
    }
    Ok(selected.into_iter().collect())
}

fn parse_expert_layer(raw: &str) -> Result<u32> {
    let layer = raw.parse::<u32>().map_err(|_| {
        DeltafinError::new(format!(
            "expert layer {raw:?} must be an integer in {}..={}",
            crate::experts::K3_MOE_LAYER_FIRST,
            crate::experts::K3_MOE_LAYER_LAST,
        ))
    })?;
    if !(crate::experts::K3_MOE_LAYER_FIRST..=crate::experts::K3_MOE_LAYER_LAST).contains(&layer) {
        return Err(DeltafinError::new(format!(
            "expert layer {layer} is outside {}..={}",
            crate::experts::K3_MOE_LAYER_FIRST,
            crate::experts::K3_MOE_LAYER_LAST,
        )));
    }
    Ok(layer)
}

fn parse_warm_expert_cache(values: &[String]) -> Result<WarmExpertCacheArgs> {
    let mut warm = WarmExpertCacheArgs::default();
    let mut conversion_option_seen = false;
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--model-root" => {
                warm.model_root = Some(take_value(values, &mut index, option)?.into());
            }
            "--cache" => warm.cache = Some(take_value(values, &mut index, option)?.into()),
            "--trace" => warm
                .traces
                .push(take_value(values, &mut index, option)?.into()),
            "--fetch" => {
                let raw = take_value(values, &mut index, option)?;
                warm.fetch = raw
                    .parse::<usize>()
                    .map_err(|_| DeltafinError::new("--fetch must be a non-negative integer"))?;
            }
            "--show" => {
                let raw = take_value(values, &mut index, option)?;
                warm.show = raw
                    .parse::<usize>()
                    .map_err(|_| DeltafinError::new("--show must be a non-negative integer"))?;
            }
            "--json" => warm.json = true,
            "--convert-npz" => warm.convert_npz = true,
            "--keep-npz" => {
                conversion_option_seen = true;
                warm.keep_npz = true;
            }
            "--convert-limit" => {
                conversion_option_seen = true;
                let raw = take_value(values, &mut index, option)?;
                let limit = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new("--convert-limit must be a positive integer")
                })?;
                if limit == 0 {
                    return Err(DeltafinError::new(
                        "--convert-limit must be a positive integer",
                    ));
                }
                warm.convert_limit = Some(limit);
            }
            "--convert-throttle-ms" => {
                conversion_option_seen = true;
                let raw = take_value(values, &mut index, option)?;
                warm.convert_throttle_ms = raw.parse::<u64>().map_err(|_| {
                    DeltafinError::new("--convert-throttle-ms must be a non-negative integer")
                })?;
            }
            "--convert-workers" => {
                conversion_option_seen = true;
                let raw = take_value(values, &mut index, option)?;
                warm.convert_workers = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--convert-workers must be an integer in 1..={}",
                        crate::legacy_npz::MAX_CONVERSION_WORKERS
                    ))
                })?;
                if !(1..=crate::legacy_npz::MAX_CONVERSION_WORKERS).contains(&warm.convert_workers)
                {
                    return Err(DeltafinError::new(format!(
                        "--convert-workers must be an integer in 1..={}",
                        crate::legacy_npz::MAX_CONVERSION_WORKERS
                    )));
                }
            }
            "--workers" => {
                let raw = take_value(values, &mut index, option)?;
                warm.workers = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::weight_fetch::MAX_PARALLEL_TRANSFERS
                    ))
                })?;
                if !(1..=crate::weight_fetch::MAX_PARALLEL_TRANSFERS).contains(&warm.workers) {
                    return Err(DeltafinError::new(format!(
                        "--workers must be an integer in 1..={}",
                        crate::weight_fetch::MAX_PARALLEL_TRANSFERS
                    )));
                }
            }
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown warm-expert-cache argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if warm.fetch > 0 && warm.cache.is_some() {
        return Err(DeltafinError::new(
            "--cache cannot be combined with --fetch; explicit fetch always uses model-root/k3-experts",
        ));
    }
    if !warm.convert_npz && conversion_option_seen {
        return Err(DeltafinError::new(
            "--keep-npz and --convert-* options require --convert-npz",
        ));
    }
    Ok(warm)
}

fn parse_populate_storage_home(values: &[String]) -> Result<PopulateStorageHomeArgs> {
    let mut dir = None;
    let mut model_root = None;
    let mut budget_bytes = None;
    let mut verify_existing = false;
    let mut workers = 4;
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--dir" => dir = Some(PathBuf::from(take_value(values, &mut index, option)?)),
            "--model-root" => {
                model_root = Some(PathBuf::from(take_value(values, &mut index, option)?));
            }
            "--budget-gb" => {
                let raw = take_value(values, &mut index, option)?;
                let gigabytes = raw
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| DeltafinError::new("--budget-gb must be a positive number"))?;
                budget_bytes = Some((gigabytes * 1e9) as u64);
            }
            "--verify-existing" => verify_existing = true,
            "--workers" => {
                let raw = take_value(values, &mut index, option)?;
                workers = raw
                    .parse::<usize>()
                    .ok()
                    .filter(|value| (1..=16).contains(value))
                    .ok_or_else(|| DeltafinError::new("--workers must be an integer in 1..=16"))?;
            }
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown populate-storage-home argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    let dir = dir.ok_or_else(|| DeltafinError::new("populate-storage-home needs --dir PATH"))?;
    if !dir.is_absolute() {
        return Err(DeltafinError::new("--dir must be an absolute path"));
    }
    Ok(PopulateStorageHomeArgs {
        model_root,
        dir,
        budget_bytes,
        verify_existing,
        workers,
    })
}

fn parse_setup_qwen(values: &[String]) -> Result<SetupQwenArgs> {
    let mut setup = SetupQwenArgs {
        check: false,
        model_root: None,
    };
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--check" => setup.check = true,
            "--model-root" => {
                setup.model_root = Some(take_value(values, &mut index, option)?.into());
            }
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown setup-qwen argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    Ok(setup)
}

fn parse_setup_k3(values: &[String]) -> Result<SetupK3Args> {
    let mut setup = SetupK3Args {
        meta_only: false,
        check: false,
        model_root: None,
    };
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--meta-only" => setup.meta_only = true,
            "--check" => setup.check = true,
            "--model-root" => {
                setup.model_root = Some(take_value(values, &mut index, option)?.into());
            }
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown setup-k3 argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if setup.meta_only == setup.check {
        return Err(DeltafinError::new(
            "setup-k3 requires exactly one of --meta-only or --check",
        ));
    }
    Ok(setup)
}

fn parse_setup_dspark(values: &[String]) -> Result<SetupDSparkArgs> {
    let mut setup = SetupDSparkArgs::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--check" => setup.check = true,
            "--audit-only" => setup.audit_only = true,
            "--destination" => {
                setup.destination = Some(take_value(values, &mut index, option)?.into());
            }
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown setup-dspark argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if setup.check && setup.audit_only {
        return Err(DeltafinError::new(
            "--check and --audit-only are mutually exclusive",
        ));
    }
    Ok(setup)
}

fn parse_serve(values: &[String]) -> Result<ServeArgs> {
    let mut serve = ServeArgs::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--host" => serve.host = take_value(values, &mut index, option)?,
            "--port" => {
                let raw = take_value(values, &mut index, option)?;
                serve.port = raw
                    .parse::<u16>()
                    .map_err(|_| DeltafinError::new("--port must be an integer in 0..=65535"))?;
            }
            "--max-tokens" => {
                let raw = take_value(values, &mut index, option)?;
                serve.max_tokens = raw
                    .parse::<usize>()
                    .map_err(|_| DeltafinError::new("--max-tokens must be a positive integer"))?;
                if serve.max_tokens == 0 {
                    return Err(DeltafinError::new(
                        "--max-tokens must be a positive integer",
                    ));
                }
            }
            "--queue" => {
                let raw = take_value(values, &mut index, option)?;
                serve.queue_slots = Some(parse_generation_queue_slots(&raw)?);
            }
            "--max-request-bytes" => {
                let raw = take_value(values, &mut index, option)?;
                serve.max_request_bytes = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--max-request-bytes must be an integer in 1..={}",
                        crate::openai::MAX_REQUEST_BODY_BYTES
                    ))
                })?;
                if !(1..=crate::openai::MAX_REQUEST_BODY_BYTES).contains(&serve.max_request_bytes) {
                    return Err(DeltafinError::new(format!(
                        "--max-request-bytes must be an integer in 1..={}",
                        crate::openai::MAX_REQUEST_BODY_BYTES
                    )));
                }
            }
            "--response-memo-entries" => {
                let raw = take_value(values, &mut index, option)?;
                serve.response_memo_entries = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new("--response-memo-entries must be a non-negative integer")
                })?;
            }
            "--response-memo-bytes" => {
                let raw = take_value(values, &mut index, option)?;
                serve.response_memo_bytes = raw.parse::<usize>().map_err(|_| {
                    DeltafinError::new(format!(
                        "--response-memo-bytes must be an integer in 0..={}",
                        crate::openai::MAX_RESPONSE_MEMO_BYTES
                    ))
                })?;
                if serve.response_memo_bytes > crate::openai::MAX_RESPONSE_MEMO_BYTES {
                    return Err(DeltafinError::new(format!(
                        "--response-memo-bytes must be an integer in 0..={}",
                        crate::openai::MAX_RESPONSE_MEMO_BYTES
                    )));
                }
            }
            "--router-trace" => {
                serve.router_trace = Some(take_value(values, &mut index, option)?.into());
            }
            "--router-trace-mode" => {
                serve.router_trace_mode = Some(take_value(values, &mut index, option)?);
            }
            "--model-root" => serve.model_root = take_value(values, &mut index, option)?.into(),
            "--device" => serve.device = Some(take_value(values, &mut index, option)?),
            "--spine" => serve.spine = Some(take_value(values, &mut index, option)?),
            "--spine-read-threads" => {
                serve.spine_read_threads = Some(crate::config::parse_spine_read_threads(
                    &take_value(values, &mut index, option)?,
                )?);
            }
            "--expert-backend" => {
                serve.expert_backend = Some(take_value(values, &mut index, option)?);
            }
            "-h" | "--help" => return Err(DeltafinError::new(HELP)),
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown serve argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    if serve.host.trim().is_empty() || serve.host.contains(['/', '\\', '\0']) {
        return Err(DeltafinError::new(
            "--host must be a non-empty socket host name or IP address",
        ));
    }
    Ok(serve)
}

fn parse_generation_queue_slots(raw: &str) -> Result<usize> {
    let bounds_message = || {
        DeltafinError::new(format!(
            "the server generation queue must be an integer in 0..={} waiting slots",
            crate::openai::MAX_GENERATION_QUEUE_SLOTS
        ))
    };
    let slots = raw.parse::<usize>().map_err(|_| bounds_message())?;
    if slots > crate::openai::MAX_GENERATION_QUEUE_SLOTS {
        return Err(bounds_message());
    }
    Ok(slots)
}

/// Resolve the serve queue size: an explicit `--queue` wins, the
/// `K3_SERVER_QUEUE` environment value is the fallback, and its absence means
/// the documented immediate-429 behavior. A malformed environment value is an
/// error rather than a silently ignored setting.
pub(crate) fn resolve_generation_queue_slots(
    flag: Option<usize>,
    environment: Option<&str>,
) -> Result<usize> {
    if let Some(slots) = flag {
        return Ok(slots);
    }
    match environment {
        None => Ok(0),
        Some(raw) => parse_generation_queue_slots(raw)
            .map_err(|error| DeltafinError::new(format!("K3_SERVER_QUEUE: {error}"))),
    }
}

fn parse_pack_spine(values: &[String]) -> Result<PackSpineArgs> {
    let mut pack = PackSpineArgs::default();
    let mut index = 0;
    while index < values.len() {
        let option = values[index].as_str();
        match option {
            "--model-root" => pack.model_root = take_value(values, &mut index, option)?.into(),
            "--spine" => pack.spine = Some(take_value(values, &mut index, option)?),
            "--output" => pack.output = Some(take_value(values, &mut index, option)?.into()),
            "--layer" => {
                let raw = take_value(values, &mut index, option)?;
                pack.layer =
                    Some(raw.parse::<u32>().map_err(|_| {
                        DeltafinError::new("--layer must be a non-negative integer")
                    })?);
            }
            "--verify-only" => pack.verify_only = true,
            _ => {
                return Err(DeltafinError::new(format!(
                    "unknown pack-spine argument {option:?}; run deltafin --help"
                )));
            }
        }
        index += 1;
    }
    Ok(pack)
}

fn take_value(values: &[String], index: &mut usize, option: &str) -> Result<String> {
    *index += 1;
    values
        .get(*index)
        .cloned()
        .ok_or_else(|| DeltafinError::new(format!("{option} requires a value")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_the_existing_direct_run_shape() {
        let command = parse([
            "deltafin",
            "--prompt",
            "hello",
            "--max-new",
            "17",
            "--chat",
            "--stats",
        ])
        .unwrap();
        let Command::Run(run) = command else {
            panic!("expected run command")
        };
        assert_eq!(run.prompt, "hello");
        assert_eq!(run.max_new, Some(17));
        assert!(run.chat);
        assert!(run.stats);
        assert!(!run.layer_profile);
    }

    #[test]
    fn layer_profile_is_independent_of_stats() {
        let Command::Run(neither) = parse(["deltafin"]).unwrap() else {
            panic!("expected run command")
        };
        assert!(!neither.stats && !neither.layer_profile);

        let Command::Run(profile_only) = parse(["deltafin", "--layer-profile"]).unwrap() else {
            panic!("expected run command")
        };
        assert!(!profile_only.stats && profile_only.layer_profile);

        let Command::Run(both) = parse(["deltafin", "--stats", "--layer-profile"]).unwrap() else {
            panic!("expected run command")
        };
        assert!(both.stats && both.layer_profile);
    }

    #[test]
    fn supports_an_explicit_run_subcommand() {
        assert!(matches!(
            parse(["deltafin", "run", "--device", "cuda:1"]).unwrap(),
            Command::Run(RunArgs {
                device: Some(device),
                ..
            }) if device == "cuda:1"
        ));
    }

    #[test]
    fn parses_bounded_spine_reader_overrides_for_run_and_server() {
        let Command::Run(run) = parse(["deltafin", "run", "--spine-read-threads", "6"]).unwrap()
        else {
            panic!("expected run command")
        };
        assert_eq!(run.spine_read_threads, Some(6));

        let Command::Serve(serve) =
            parse(["deltafin", "serve", "--spine-read-threads", "3"]).unwrap()
        else {
            panic!("expected serve command")
        };
        assert_eq!(serve.spine_read_threads, Some(3));

        for invalid in ["0", "17", "many"] {
            assert!(parse(["deltafin", "run", "--spine-read-threads", invalid]).is_err());
            assert!(parse(["deltafin", "serve", "--spine-read-threads", invalid]).is_err());
        }
    }

    #[test]
    fn parses_and_bounds_the_serve_generation_queue() {
        let Command::Serve(serve) = parse(["deltafin", "serve", "--queue", "3"]).unwrap() else {
            panic!("expected serve command")
        };
        assert_eq!(serve.queue_slots, Some(3));

        let Command::Serve(explicit_zero) = parse(["deltafin", "serve", "--queue", "0"]).unwrap()
        else {
            panic!("expected serve command")
        };
        assert_eq!(explicit_zero.queue_slots, Some(0));

        let Command::Serve(unset) = parse(["deltafin", "serve"]).unwrap() else {
            panic!("expected serve command")
        };
        assert_eq!(unset.queue_slots, None);

        for invalid in ["-1", "many", "65"] {
            assert!(parse(["deltafin", "serve", "--queue", invalid]).is_err());
        }
    }

    #[test]
    fn queue_slots_resolve_from_flag_then_environment_then_immediate_429_default() {
        assert_eq!(resolve_generation_queue_slots(Some(2), Some("9")).unwrap(), 2);
        assert_eq!(resolve_generation_queue_slots(None, Some("9")).unwrap(), 9);
        assert_eq!(resolve_generation_queue_slots(None, None).unwrap(), 0);
        assert!(resolve_generation_queue_slots(None, Some("many")).is_err());
        assert!(resolve_generation_queue_slots(None, Some("65")).is_err());
        // The explicit flag also wins over a malformed environment value:
        // the operator asked for a specific admission policy on this run.
        assert_eq!(
            resolve_generation_queue_slots(Some(1), Some("many")).unwrap(),
            1
        );
    }

    #[test]
    fn parses_native_router_tracing_for_run_and_server() {
        let Command::Run(run) = parse([
            "deltafin",
            "run",
            "--router-trace",
            "k3-meta/native-routes.jsonl",
            "--router-trace-mode",
            "sync",
        ])
        .unwrap() else {
            panic!("expected run command")
        };
        assert_eq!(
            run.router_trace,
            Some(PathBuf::from("k3-meta/native-routes.jsonl"))
        );
        assert_eq!(run.router_trace_mode.as_deref(), Some("sync"));

        let Command::Serve(serve) =
            parse(["deltafin", "serve", "--router-trace", "routes.jsonl"]).unwrap()
        else {
            panic!("expected serve command")
        };
        assert_eq!(serve.router_trace, Some(PathBuf::from("routes.jsonl")));
        assert!(serve.router_trace_mode.is_none());
    }

    #[test]
    fn doctor_accepts_the_same_explicit_model_root_as_native_run() {
        assert_eq!(
            parse(["deltafin", "doctor", "--model-root", "/model"]).unwrap(),
            Command::Doctor(DoctorArgs {
                model_root: Some(PathBuf::from("/model")),
                runtime_only: false,
            })
        );
        assert_eq!(
            parse(["deltafin", "doctor", "--runtime-only"]).unwrap(),
            Command::Doctor(DoctorArgs {
                model_root: None,
                runtime_only: true,
            })
        );
        assert!(
            parse([
                "deltafin",
                "doctor",
                "--runtime-only",
                "--model-root",
                "/model",
            ])
            .is_err()
        );
        assert!(parse(["deltafin", "doctor", "--device", "cpu"]).is_err());
    }

    #[test]
    fn refuses_missing_and_invalid_values() {
        assert!(parse(["deltafin", "--prompt"]).is_err());
        assert!(parse(["deltafin", "--max-new", "-1"]).is_err());
        assert!(parse(["deltafin", "--unknown"]).is_err());
    }

    #[test]
    fn an_empty_raw_prompt_is_refused_before_the_engine_loads() {
        // The engine rejects it too, but only after minutes of model loading;
        // the HTTP server refuses the same input up front with a 400.
        for arguments in [
            &["deltafin", "--prompt", ""][..],
            &["deltafin", "run", "--prompt", ""][..],
            &["deltafin", "benchmark", "--prompt", ""][..],
        ] {
            let error = parse(arguments.iter().copied())
                .err()
                .unwrap_or_else(|| panic!("{arguments:?} was accepted"))
                .to_string();
            assert!(error.contains("--prompt must not be empty"), "{error}");
        }
        // A chat turn is rendered through the template first, so an empty
        // message is a legitimate (if unusual) turn; whitespace is text.
        assert!(parse(["deltafin", "--chat", "--prompt", ""]).is_ok());
        assert!(parse(["deltafin", "--prompt", "", "--chat"]).is_ok());
        assert!(parse(["deltafin", "benchmark", "--chat", "--prompt", ""]).is_ok());
        assert!(parse(["deltafin", "--prompt", " "]).is_ok());
        assert!(parse(["deltafin"]).is_ok(), "the default prompt is non-empty");
    }

    #[test]
    fn parses_the_single_binary_pack_subcommand() {
        assert_eq!(
            parse([
                "deltafin",
                "pack-spine",
                "--model-root",
                "/model",
                "--output",
                "/packs",
                "--layer",
                "17",
                "--verify-only",
            ])
            .unwrap(),
            Command::PackSpine(PackSpineArgs {
                model_root: PathBuf::from("/model"),
                spine: None,
                output: Some(PathBuf::from("/packs")),
                layer: Some(17),
                verify_only: true,
            })
        );
    }

    #[test]
    fn parses_an_explicit_non_weight_exact_int8_pack_request() {
        let Command::PackSpine(pack) =
            parse(["deltafin", "pack-spine", "--spine", "int8"]).unwrap()
        else {
            panic!("expected pack-spine command")
        };
        assert_eq!(pack.spine.as_deref(), Some("int8"));
        assert!(pack.output.is_none());
    }

    #[test]
    fn parses_the_native_openai_server_without_python() {
        assert_eq!(
            parse([
                "deltafin",
                "serve",
                "--host",
                "::1",
                "--port",
                "9000",
                "--max-tokens",
                "4096",
                "--max-request-bytes",
                "67108864",
                "--response-memo-entries",
                "17",
                "--response-memo-bytes",
                "33554432",
                "--device",
                "mps",
            ])
            .unwrap(),
            Command::Serve(ServeArgs {
                host: "::1".into(),
                port: 9000,
                max_tokens: 4096,
                max_request_bytes: 67_108_864,
                response_memo_entries: 17,
                response_memo_bytes: 33_554_432,
                device: Some("mps".into()),
                ..ServeArgs::default()
            })
        );
        assert!(parse(["deltafin", "serve", "--max-tokens", "0"]).is_err());
        assert!(parse(["deltafin", "serve", "--max-request-bytes", "0"]).is_err());
        assert!(parse(["deltafin", "serve", "--max-request-bytes", "1073741825",]).is_err());
        let Command::Serve(disabled_memo) = parse([
            "deltafin",
            "serve",
            "--response-memo-entries",
            "0",
            "--response-memo-bytes",
            "0",
        ])
        .unwrap() else {
            panic!("expected serve command")
        };
        assert_eq!(disabled_memo.response_memo_entries, 0);
        assert_eq!(disabled_memo.response_memo_bytes, 0);
        assert!(parse(["deltafin", "serve", "--response-memo-bytes", "1073741825",]).is_err());
        assert!(parse(["deltafin", "serve", "--response-memo-entries", "-1"]).is_err());
        assert!(parse(["deltafin", "serve", "--port", "70000"]).is_err());
    }

    #[test]
    fn parses_a_native_only_benchmark_campaign() {
        assert_eq!(
            parse([
                "deltafin",
                "benchmark",
                "--prompt",
                "native prompt",
                "--chat",
                "--max-new",
                "17",
                "--repetitions",
                "2",
                "--warmup-steps",
                "1",
                "--timeout-seconds",
                "900",
                "--arm",
                "baseline=",
                "--arm",
                "candidate=K3_TEST_MODE=fast K3_OTHER='two words'",
                "--expect-token-ids",
                "1,2,3",
                "--expect-text",
                "answer",
                "--output",
                "bench-results/native-test",
                "--keep-going",
                "--model-root",
                "/model",
            ])
            .unwrap(),
            Command::Benchmark(BenchmarkArgs {
                prompt: "native prompt".into(),
                chat: true,
                max_new_tokens: 17,
                repetitions: 2,
                warmup_steps: 1,
                timeout_ns: 900_000_000_000,
                output: Some(PathBuf::from("bench-results/native-test")),
                arms: vec![
                    BenchmarkArmArgs {
                        name: "baseline".into(),
                        environment_spec: String::new(),
                    },
                    BenchmarkArmArgs {
                        name: "candidate".into(),
                        environment_spec: "K3_TEST_MODE=fast K3_OTHER='two words'".into(),
                    },
                ],
                expected_token_ids: Some("1,2,3".into()),
                expected_text: Some("answer".into()),
                keep_going: true,
                model_root: PathBuf::from("/model"),
            })
        );
        assert!(parse(["deltafin", "benchmark", "--max-new", "0"]).is_err());
        assert!(parse(["deltafin", "benchmark", "--repetitions", "0"]).is_err());
        assert!(parse(["deltafin", "benchmark", "--arm", "bad name="]).is_err());
        let Command::Benchmark(compatible) = parse([
            "deltafin",
            "benchmark",
            "--configs",
            "",
            "K3_PILOT=0 K3_MOE=cpu",
            "--names",
            "defaults",
            "cpu",
            "--reps",
            "2",
            "--tokens",
            "8",
            "--timeout",
            "1.5",
            "--output-dir",
            "evidence",
        ])
        .unwrap() else {
            panic!("expected benchmark command")
        };
        assert_eq!(compatible.arms.len(), 2);
        assert_eq!(compatible.arms[1].name, "cpu");
        assert_eq!(compatible.repetitions, 2);
        assert_eq!(compatible.max_new_tokens, 8);
        assert_eq!(compatible.timeout_ns, 1_500_000_000);
        assert_eq!(compatible.output, Some(PathBuf::from("evidence")));
        assert!(
            parse([
                "deltafin",
                "benchmark",
                "--configs",
                "",
                "K3_X=1",
                "--names",
                "only-one",
            ])
            .is_err()
        );
        assert!(parse(["deltafin", "benchmark", "--timeout", "NaN"]).is_err());
        assert!(
            parse([
                "deltafin",
                "benchmark",
                "--arm",
                "same=",
                "--arm",
                "same=K3_X=1",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_the_native_upgrade_without_options() {
        assert_eq!(parse(["deltafin", "upgrade"]).unwrap(), Command::Upgrade);
        assert!(parse(["deltafin", "upgrade", "--force"]).is_err());
    }

    #[test]
    fn every_supported_subcommand_has_successful_consistent_help() {
        for subcommand in [
            "run",
            "serve",
            "benchmark",
            "doctor",
            "upgrade",
            "setup",
            "setup-dspark",
            "setup-k3",
            "setup-qwen",
            "fetch-weights",
            "warm-expert-cache",
            "convert-spine-int8",
            "convert-experts-scale4",
            "pack-spine",
        ] {
            for help in ["-h", "--help"] {
                assert_eq!(
                    parse(["deltafin", subcommand, help]).unwrap(),
                    Command::Help,
                    "{subcommand} {help}",
                );
            }
        }
    }

    #[test]
    fn parses_native_dspark_setup_modes() {
        assert_eq!(
            parse([
                "deltafin",
                "setup-dspark",
                "--check",
                "--destination",
                "/model/dspark",
            ])
            .unwrap(),
            Command::SetupDSpark(SetupDSparkArgs {
                check: true,
                audit_only: false,
                destination: Some(PathBuf::from("/model/dspark")),
            })
        );
        assert!(parse(["deltafin", "setup-dspark", "--check", "--audit-only",]).is_err());
    }

    #[test]
    fn parses_full_native_one_shot_setup() {
        assert_eq!(
            parse([
                "deltafin",
                "setup",
                "--dry-run",
                "--include-qwen",
                "--workers",
                "4",
                "--model-root",
                "/model",
            ])
            .unwrap(),
            Command::Setup(SetupArgs {
                dry_run: true,
                check: false,
                mode: crate::one_shot_setup::SetupMode::Auto,
                include_qwen: true,
                workers: 4,
                model_root: Some(PathBuf::from("/model")),
            })
        );
        assert!(parse(["deltafin", "setup", "--dry-run", "--check"]).is_err());
        assert!(parse(["deltafin", "setup", "--workers", "0"]).is_err());
        assert!(parse(["deltafin", "setup", "--workers", "17"]).is_err());
        let Command::Setup(full) = parse(["deltafin", "setup", "--full"]).unwrap() else {
            panic!("expected setup command")
        };
        assert_eq!(full.mode, crate::one_shot_setup::SetupMode::Full);
        let Command::Setup(stream) = parse(["deltafin", "setup", "--stream"]).unwrap() else {
            panic!("expected setup command")
        };
        assert_eq!(stream.mode, crate::one_shot_setup::SetupMode::Stream);
        assert!(parse(["deltafin", "setup", "--full", "--stream"]).is_err());
    }

    #[test]
    fn parses_native_k3_metadata_setup_modes() {
        assert_eq!(
            parse([
                "deltafin",
                "setup-k3",
                "--meta-only",
                "--model-root",
                "/model",
            ])
            .unwrap(),
            Command::SetupK3(SetupK3Args {
                meta_only: true,
                check: false,
                model_root: Some(PathBuf::from("/model")),
            })
        );
        assert_eq!(
            parse(["deltafin", "setup-k3", "--check"]).unwrap(),
            Command::SetupK3(SetupK3Args {
                meta_only: false,
                check: true,
                model_root: None,
            })
        );
        assert!(parse(["deltafin", "setup-k3"]).is_err());
        assert!(parse(["deltafin", "setup-k3", "--meta-only", "--check"]).is_err());
    }

    #[test]
    fn parses_native_optional_qwen_setup() {
        assert_eq!(
            parse(["deltafin", "setup-qwen"]).unwrap(),
            Command::SetupQwen(SetupQwenArgs {
                check: false,
                model_root: None,
            })
        );
        assert_eq!(
            parse([
                "deltafin",
                "setup-qwen",
                "--check",
                "--model-root",
                "/model",
            ])
            .unwrap(),
            Command::SetupQwen(SetupQwenArgs {
                check: true,
                model_root: Some(PathBuf::from("/model")),
            })
        );
        assert!(parse(["deltafin", "setup-qwen", "--model-root"]).is_err());
        assert!(parse(["deltafin", "setup-qwen", "--probe-only"]).is_err());
    }

    #[test]
    fn parses_native_weight_fetch_defaults_and_explicit_subsets() {
        assert_eq!(
            parse(["deltafin", "fetch-weights"]).unwrap(),
            Command::FetchWeights(FetchWeightsArgs::default())
        );
        assert_eq!(
            parse([
                "deltafin",
                "fetch-weights",
                "--dry-run",
                "--experts-only",
                "--workers",
                "4",
                "--model-root",
                "/model",
            ])
            .unwrap(),
            Command::FetchWeights(FetchWeightsArgs {
                dry_run: true,
                spine_only: false,
                experts_only: true,
                layers: None,
                workers: 4,
                model_root: Some(PathBuf::from("/model")),
            })
        );
        assert!(
            parse([
                "deltafin",
                "fetch-weights",
                "--spine-only",
                "--experts-only",
            ])
            .is_err()
        );
        assert!(parse(["deltafin", "fetch-weights", "--workers", "0"]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--workers", "17"]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--workers"]).is_err());

        let Command::FetchWeights(layers) =
            parse(["deltafin", "fetch-weights", "--layers", "1-3,2,45,92"]).unwrap()
        else {
            panic!("expected fetch-weights command")
        };
        assert_eq!(layers.layers, Some(vec![1, 2, 3, 45, 92]));
        assert!(!layers.experts_only);
        assert!(parse(["deltafin", "fetch-weights", "--layers", ""]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--layers", "0"]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--layers", "93"]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--layers", "3-1"]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--layers", "1--2"]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--layers", "1, 2"]).is_err());
        assert!(parse(["deltafin", "fetch-weights", "--spine-only", "--layers", "1",]).is_err());
    }

    #[test]
    fn parses_native_read_only_and_explicit_expert_warming() {
        assert_eq!(
            parse([
                "deltafin",
                "warm-expert-cache",
                "--trace",
                "k3-meta/router_trace-a.jsonl",
                "--trace",
                "k3-meta/router_trace-b.jsonl",
                "--fetch",
                "128",
                "--show",
                "5",
                "--json",
                "--workers",
                "4",
                "--model-root",
                "/model",
            ])
            .unwrap(),
            Command::WarmExpertCache(WarmExpertCacheArgs {
                model_root: Some(PathBuf::from("/model")),
                cache: None,
                traces: vec![
                    PathBuf::from("k3-meta/router_trace-a.jsonl"),
                    PathBuf::from("k3-meta/router_trace-b.jsonl"),
                ],
                fetch: 128,
                show: 5,
                json: true,
                workers: 4,
                convert_npz: false,
                convert_workers: crate::legacy_npz::DEFAULT_CONVERSION_WORKERS,
                convert_limit: None,
                convert_throttle_ms: 0,
                keep_npz: false,
            })
        );
        assert_eq!(
            parse([
                "deltafin",
                "warm-expert-cache",
                "--cache",
                "/read-only-cache",
            ])
            .unwrap(),
            Command::WarmExpertCache(WarmExpertCacheArgs {
                cache: Some(PathBuf::from("/read-only-cache")),
                ..WarmExpertCacheArgs::default()
            })
        );
        assert!(
            parse([
                "deltafin",
                "warm-expert-cache",
                "--cache",
                "/other",
                "--fetch",
                "1",
            ])
            .is_err()
        );
        assert!(parse(["deltafin", "warm-expert-cache", "--workers", "0"]).is_err());

        assert_eq!(
            parse([
                "deltafin",
                "warm-expert-cache",
                "--convert-npz",
                "--convert-workers",
                "3",
                "--convert-limit",
                "17",
                "--convert-throttle-ms",
                "25",
                "--keep-npz",
                "--cache",
                "/legacy-cache",
            ])
            .unwrap(),
            Command::WarmExpertCache(WarmExpertCacheArgs {
                cache: Some(PathBuf::from("/legacy-cache")),
                convert_npz: true,
                convert_workers: 3,
                convert_limit: Some(17),
                convert_throttle_ms: 25,
                keep_npz: true,
                ..WarmExpertCacheArgs::default()
            })
        );
        assert!(parse(["deltafin", "warm-expert-cache", "--convert-workers", "2",]).is_err());
        assert!(
            parse([
                "deltafin",
                "warm-expert-cache",
                "--convert-npz",
                "--convert-workers",
                "0",
            ])
            .is_err()
        );
        assert!(
            parse([
                "deltafin",
                "warm-expert-cache",
                "--convert-npz",
                "--convert-limit",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_native_lossless_expert_conversion() {
        assert_eq!(
            parse([
                "deltafin",
                "convert-experts-scale4",
                "--model-root",
                "/model",
                "--source-root",
                "/raw",
                "--output-root",
                "/compact",
                "--workers",
                "8",
                "--no-resume",
            ])
            .unwrap(),
            Command::ConvertExpertsScale4(ConvertExpertsScale4Args {
                model_root: PathBuf::from("/model"),
                source_root: Some(PathBuf::from("/raw")),
                output_root: Some(PathBuf::from("/compact")),
                workers: 8,
                resume: false,
            })
        );
        assert!(parse(["deltafin", "convert-experts-scale4", "--workers", "0"]).is_err());
        assert!(parse(["deltafin", "convert-experts-scale4", "--workers", "17"]).is_err());
    }

    #[test]
    fn parses_native_explicit_non_exact_spine_conversion() {
        assert_eq!(
            parse([
                "deltafin",
                "convert-spine-int8",
                "--model-root",
                "/model",
                "--checkpoint-rows",
                "512",
                "--no-resume",
            ])
            .unwrap(),
            Command::ConvertSpineInt8(ConvertSpineInt8Args {
                model_root: PathBuf::from("/model"),
                checkpoint_rows: 512,
                resume: false,
            })
        );
        assert!(parse(["deltafin", "convert-spine-int8", "--checkpoint-rows", "0"]).is_err());
        assert!(
            parse([
                "deltafin",
                "convert-spine-int8",
                "--checkpoint-rows",
                "1048577",
            ])
            .is_err()
        );
    }
}
