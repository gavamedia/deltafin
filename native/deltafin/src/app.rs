//! Production command-line application built on the reusable Deltafin core.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::Duration;

use crate::cli::{
    BenchmarkArgs, Command, ConvertExpertsScale4Args, ConvertSpineInt8Args, DoctorArgs,
    FetchWeightsArgs, PopulateStorageHomeArgs, SetupArgs, WarmExpertCacheArgs,
};
use crate::config::RuntimeConfig;
use crate::engine::NativeTargetEngine;
use crate::error::{DeltafinError, Result};
use crate::openai::{OpenAiHttpServer, ServerConfig};
use crate::platform::Host;
use crate::provider::NativeProvider;
use crate::run_interrupt::RunInterruptGuard;
use crate::sys::fs as sys_fs;

fn run() -> Result<()> {
    let command = crate::cli::parse(std::env::args_os())?;
    let audited_executable = command_requires_native_runtime_preflight(&command)
        .then(|| require_audited_native_runtime(command_uses_target_runtime(&command)))
        .transpose()?;
    match command {
        Command::Help => {
            print!("{}", crate::cli::HELP);
            Ok(())
        }
        Command::Version => {
            println!("deltafin {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Doctor(arguments) => {
            let host = Host::compiled();
            println!("host: {host}");
            host.validate()?;
            let payload = match audit_doctor_payload(&arguments)? {
                Some((model_root, payload)) => {
                    println!("model root: {}", model_root.display());
                    println!(
                        "installed payload: complete canonical {} K3 weight roster present (regular-file identity and exact size checked against the pinned inventory); pinned metadata and required default DSpark digest audits passed{}",
                        payload.mode.as_str(),
                        if payload.qwen_installed {
                            "; optional Qwen pair digest audit passed"
                        } else {
                            "; optional Qwen absent"
                        },
                    );
                    Some(payload)
                }
                None => {
                    println!("model payload: skipped (--runtime-only)");
                    None
                }
            };
            let executable = audited_executable.as_ref().ok_or_else(|| {
                DeltafinError::new(
                    "doctor command reached payload audit without native-runtime preflight",
                )
            })?;
            report_native_runtime(executable)?;
            if let Some(payload) = payload {
                let spec = payload.spec;
                println!(
                    "model contract: Kimi K3 {} layers ({} KDA, {} MLA), {} experts/top-{}, context {}",
                    spec.num_hidden_layers,
                    spec.kda_layers(),
                    spec.mla_layers(),
                    spec.num_experts,
                    spec.num_experts_per_token,
                    spec.max_position_embeddings,
                );
            }
            println!(
                "target execution: native multi-position sequence tape structurally linked; original-BF16/raw-v1 Metal completed exact 17-token full-target parity on physical MPS (other backend/representation combinations remain separately identified)"
            );
            Ok(())
        }
        Command::Upgrade => crate::upgrade::run(),
        Command::Benchmark(arguments) => run_native_benchmark(arguments),
        Command::Setup(arguments) => {
            // Only a completed installation reports the runtime. `--dry-run`
            // promises it created nothing, and `--check` is an offline audit;
            // neither should execute the provider canaries.
            let installed = !arguments.dry_run && !arguments.check;
            let model_root = arguments.model_root.clone();
            run_one_shot_setup(arguments)?;
            if installed {
                // The default resident spine is row-int8; prepare it from the
                // just-installed BF16 source so a fresh install runs the
                // default without a separate conversion step. The converter
                // is resumable and recognizes an already-complete output.
                let root = fetch_model_root(model_root.as_deref())?;
                // Belt-and-suspenders on top of the per-subdirectory markers
                // setup already places under k3-experts/k3-resident-*: one
                // marker at the model root covers the whole install tree,
                // including anything future code adds here without its own
                // call. A marker is a best-effort precaution, not a proof
                // that indexing was ever the cause of any specific slowdown;
                // it costs nothing and is safe to place regardless.
                crate::weight_fetch::ensure_spotlight_marker_here(&root)?;
                println!();
                println!(
                    "preparing the default row-int8 resident spine (original BF16 remains on disk; --spine bf16 selects it)"
                );
                let mut progress = CliSpineInt8Progress;
                crate::spine_int8::convert_full(
                    &crate::spine_int8::ConvertOptions::under(&root),
                    &mut progress,
                )?;
                let executable = audited_executable.as_ref().ok_or_else(|| {
                    DeltafinError::new("setup command completed without native-runtime preflight")
                })?;
                println!();
                report_native_runtime(executable)?;
                if cfg!(target_os = "macos") {
                    println!();
                    println!("macOS tip: this model store is large, so setup just excluded it");
                    println!("from Spotlight search indexing automatically (a hidden marker file).");
                    println!("For extra assurance, you can also exclude it yourself:");
                    println!("  1. Open System Settings");
                    println!("  2. Go to Spotlight > Search Privacy");
                    println!("  3. Add this folder: {}", root.display());
                    println!("This is optional, but a large folder that Spotlight tries to index");
                    println!("can compete with Deltafin for disk and memory while it runs.");
                }
            }
            Ok(())
        }
        Command::SetupDSpark(arguments) => crate::dspark_setup::run(arguments),
        Command::SetupK3(arguments) => crate::setup_k3::run(arguments),
        Command::SetupQwen(arguments) => crate::setup_qwen::run(arguments),
        Command::FetchWeights(arguments) => run_fetch_weights(arguments),
        Command::WarmExpertCache(arguments) => run_warm_expert_cache(arguments),
        Command::PopulateStorageHome(arguments) => run_populate_storage_home(arguments),
        Command::ConvertSpineInt8(arguments) => run_convert_spine_int8(arguments),
        Command::ConvertExpertsScale4(arguments) => run_convert_experts_scale4(arguments),
        Command::PackSpine(arguments) => crate::pack_command::run(arguments),
        Command::Serve(arguments) => {
            let host = Host::compiled().validate()?;
            let config = RuntimeConfig::from_server(&arguments)?;
            let engine = NativeTargetEngine::bootstrap(&config)?;
            let status = engine.status();
            eprintln!(
                "[native-server] host={host} device={} experts={}/{} root={} representation={} resident={}/{} drafts={} router-trace={} readiness={}",
                status.device,
                status.expert_backend,
                status.expert_storage,
                status.model_root.display(),
                status.spine_representation,
                status.resident_prefix_layers,
                status.spine_layers,
                status.speculative_max_drafts,
                status.router_trace_enabled,
                status.readiness,
            );
            if let Some(note) = status.device_selection_policy.startup_note() {
                eprintln!("[native-server] {note}");
            }
            let queue_slots = crate::cli::resolve_generation_queue_slots(
                arguments.queue_slots,
                environment_text("K3_SERVER_QUEUE")?.as_deref(),
            )?;
            let address = socket_address(&arguments.host, arguments.port);
            let server_config = ServerConfig {
                max_request_body_bytes: arguments.max_request_bytes,
                max_new_tokens: arguments.max_tokens,
                default_completion_tokens: 256.min(arguments.max_tokens),
                default_chat_tokens: arguments.max_tokens,
                response_memo_entries: arguments.response_memo_entries,
                response_memo_bytes: arguments.response_memo_bytes,
                generation_queue_slots: queue_slots,
                ..ServerConfig::default()
            };
            let server =
                OpenAiHttpServer::bind(&address, engine, server_config).map_err(|error| {
                    DeltafinError::new(format!("bind native OpenAI server at {address}: {error}"))
                })?;
            eprintln!("[native-server] OpenAI-compatible API listening at http://{address}/v1");
            if queue_slots == 0 {
                eprintln!(
                    "[native-server] admission: one generation at a time; a concurrent request gets an immediate 429 (--queue N allows waiting)"
                );
            } else {
                eprintln!(
                    "[native-server] admission: one generation at a time; up to {queue_slots} request(s) wait in arrival order, the rest get 429"
                );
            }
            server
                .serve_forever()
                .map_err(|error| DeltafinError::new(format!("native OpenAI server: {error}")))
        }
        Command::Run(arguments) => {
            let host = Host::compiled().validate()?;
            let config = RuntimeConfig::from_process(arguments)?;
            eprintln!("[config] resolved: {config}");
            let mut engine = NativeTargetEngine::bootstrap(&config)?;
            let status = engine.status();
            eprintln!(
                "[native] host={host} device={} experts={}/{} root={} spine={} representation={} layers={} globals={} resident={}/{} ({:.2} GiB) transient={:.2} GiB readers=spine:{}t/expert:{}t/prefetch:{}t spine-fdcache={}/{}fd cpu-expert:{}t router-trace={} lazy-missing={} LibTorch={}",
                status.device,
                status.expert_backend,
                status.expert_storage,
                status.model_root.display(),
                status.spine_source,
                status.spine_representation,
                status.spine_layers,
                status.global_transfer_groups,
                status.resident_prefix_layers,
                status.spine_layers,
                status.resident_prefix_bytes as f64 / (1_u64 << 30) as f64,
                status.transient_layer_bytes as f64 / (1_u64 << 30) as f64,
                status.spine_reader_workers,
                status.expert_reader_workers,
                status.expert_prefetch_reader_workers,
                status.spine_fd_cache_opened,
                status.spine_fd_cache_descriptors,
                status.expert_cpu_threads,
                status.router_trace_enabled,
                status.lazy_expert_files_missing,
                status.libtorch_version,
            );
            if let Some(note) = status.device_selection_policy.startup_note() {
                eprintln!("[native] {note}");
            }
            eprintln!(
                "[native] optional Qwen: state={} model+KV-reserve={:.2} GiB verify-reserve={:.2} GiB/{} positions admitted-context={} tokens startup-error={}",
                status.qwen_state,
                status.qwen_reserved_provider_bytes as f64 / (1_u64 << 30) as f64,
                status.qwen_reserved_verify_bytes as f64 / (1_u64 << 30) as f64,
                status.qwen_reserved_verify_positions,
                status.qwen_context_capacity,
                status.qwen_last_error.unwrap_or("none"),
            );
            let (startup_reserved_tokens, startup_reserve_bytes) =
                status.context_growth.startup_growth_reserve()?;
            eprintln!(
                "[native] context cache: initial={} tokens/{:.2} MiB, growth={:.2} MiB per capacity token, startup reserve={} tokens/{:.2} MiB, exact-expanded cap={} / model contract={} tokens (live peak admission required beyond reserve)",
                status.context_growth.initial_capacity_tokens,
                status.context_growth.initial_provider_bytes as f64 / (1_u64 << 20) as f64,
                status.context_growth.bytes_per_capacity_token as f64 / (1_u64 << 20) as f64,
                startup_reserved_tokens,
                startup_reserve_bytes as f64 / (1_u64 << 20) as f64,
                status.context_growth.admitted_expanded_context_tokens,
                status.context_growth.model_max_context_tokens,
            );
            eprintln!(
                "[native] verify: one {:.2} MiB KDA generation plus a {:.2} MiB prefix record per row, width 1..={} (fresh peak admission; otherwise ordinary exact decode)",
                status.verify_snapshots.bytes_per_kda_generation as f64 / (1_u64 << 20) as f64,
                status.verify_snapshots.replay_bytes_per_position as f64 / (1_u64 << 20) as f64,
                status.verify_snapshots.max_positions,
            );
            eprintln!("[native] {}", status.readiness);
            if status.expert_heat_recording || status.expert_pin_candidates != 0 {
                eprintln!(
                    "[native] expert heat: weight={:.0} passes recording={}; pin tier: candidates={} charged={:.2} GiB",
                    status.expert_heat_weight,
                    status.expert_heat_recording,
                    status.expert_pin_candidates,
                    status.expert_pin_charged_bytes as f64 / (1_u64 << 30) as f64,
                );
            }
            let result = {
                // Arm only around interactive CLI inference. Bootstrap keeps
                // the operating system's default SIGINT behavior, and the
                // OpenAI server never installs this process-global handler.
                let interrupt_guard = RunInterruptGuard::arm()?;
                let interrupt = interrupt_guard.source();
                engine.prefill_and_decode_interruptible(&config, &interrupt)
            };
            if let Some(report) = engine.pilot_gate_report()
                && report.telemetry.passes_scored != 0
            {
                let suppressed_layers: Vec<u32> = report
                    .layers
                    .iter()
                    .filter(|layer| layer.reads_suppressed)
                    .map(|layer| layer.layer_index)
                    .collect();
                eprintln!(
                    "[pilot-gate] mode={} threshold={:.0}% warmup={} passes={} experts issued={} suppressed={} prev-token-plans={} suppressed-layers={:?}",
                    if report.measure_only { "measure" } else { "on" },
                    report.threshold * 100.0,
                    report.warmup,
                    report.telemetry.passes_scored,
                    report.telemetry.experts_issued,
                    report.telemetry.experts_suppressed,
                    report.telemetry.prev_token_plans,
                    suppressed_layers,
                );
                if config.stats {
                    for layer in &report.layers {
                        eprintln!(
                            "[pilot-gate] layer {:>2}: pilot {:>5.1}% (n={}) prev-token {:>5.1}% (n={}) preferred={} reads={}",
                            layer.layer_index,
                            layer.pilot_ema * 100.0,
                            layer.pilot_samples,
                            layer.prev_token_ema * 100.0,
                            layer.prev_token_samples,
                            layer.preferred.as_str(),
                            if layer.reads_suppressed {
                                "suppressed"
                            } else {
                                "active"
                            },
                        );
                    }
                }
            }
            let pin_status = engine.status();
            if pin_status.expert_pin_candidates != 0 {
                eprintln!(
                    "[expert-pin] candidates={} resident={} ({:.2} GiB) hits={} served-from-ram={:.2} GiB",
                    pin_status.expert_pin_candidates,
                    pin_status.expert_pin_resident_experts,
                    pin_status.expert_pin_resident_bytes as f64 / (1_u64 << 30) as f64,
                    pin_status.expert_pin_hits,
                    pin_status.expert_pin_served_bytes as f64 / (1_u64 << 30) as f64,
                );
            }
            let qwen = engine.status().qwen_telemetry;
            if qwen.proposals != 0
                || qwen.proposed_tokens != 0
                || qwen.accepted_tokens != 0
                || qwen.fallbacks != 0
                || qwen.verify_memory_rejections != 0
                || qwen.wide_loads != 0
                || qwen.wide_invocations != 0
                || qwen.wide_skips != 0
                || qwen.wide_failures != 0
                || qwen.raw_override_selections != 0
                || qwen.raw_override_failures != 0
            {
                eprintln!(
                    "[native] Qwen proposals={} tokens={}/{} fallbacks={} verify-memory-rejections={} wide-loads={} wide-invocations={} wide-skips={} wide-failures={} raw-overrides={} raw-override-failures={}",
                    qwen.proposals,
                    qwen.accepted_tokens,
                    qwen.proposed_tokens,
                    qwen.fallbacks,
                    qwen.verify_memory_rejections,
                    qwen.wide_loads,
                    qwen.wide_invocations,
                    qwen.wide_skips,
                    qwen.wide_failures,
                    qwen.raw_override_selections,
                    qwen.raw_override_failures,
                );
            }
            result
        }
    }
}

fn command_requires_native_runtime_preflight(command: &Command) -> bool {
    match command {
        Command::Help | Command::Version => false,
        Command::Run(_)
        | Command::Serve(_)
        | Command::Benchmark(_)
        | Command::Setup(_)
        | Command::PackSpine(_)
        | Command::SetupDSpark(_)
        | Command::SetupK3(_)
        | Command::SetupQwen(_)
        | Command::FetchWeights(_)
        | Command::WarmExpertCache(_)
        | Command::PopulateStorageHome(_)
        | Command::ConvertSpineInt8(_)
        | Command::ConvertExpertsScale4(_)
        | Command::Doctor(_)
        | Command::Upgrade => true,
    }
}

fn command_uses_target_runtime(command: &Command) -> bool {
    matches!(
        command,
        Command::Run(_) | Command::Serve(_) | Command::Benchmark(_)
    )
}

fn require_audited_native_runtime(target_runtime: bool) -> Result<PathBuf> {
    crate::loader_audit::reject_dynamic_loader_environment()?;
    if target_runtime {
        crate::engine::reject_product_metal_source_override()?;
    }
    crate::upgrade::audit_running_artifact()
}

fn run_native_benchmark(arguments: BenchmarkArgs) -> Result<()> {
    let root = fetch_model_root(Some(&arguments.model_root))?;
    let mut options = crate::benchmark::BenchmarkOptions::for_current_executable(root)?;
    options.prompt = arguments.prompt;
    options.chat = arguments.chat;
    options.max_new_tokens = arguments.max_new_tokens;
    options.repetitions = arguments.repetitions;
    options.warmup_steps = arguments.warmup_steps;
    options.timeout = Duration::from_nanos(arguments.timeout_ns);
    options.output_dir = arguments.output;
    options.expected_completion_token_ids = arguments
        .expected_token_ids
        .as_deref()
        .map(crate::benchmark::parse_expected_token_ids)
        .transpose()?;
    options.expected_completion_text = arguments.expected_text;
    options.keep_going = arguments.keep_going;
    if !arguments.arms.is_empty() {
        options.arms = arguments
            .arms
            .into_iter()
            .map(|arm| crate::benchmark::BenchmarkArm::new(arm.name, arm.environment_spec))
            .collect();
    }
    let report = crate::benchmark::run_campaign(&options)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report.summary).map_err(|error| {
            DeltafinError::new(format!("serialize native benchmark summary: {error}"))
        })?
    );
    require_successful_benchmark(&report)
}

fn run_populate_storage_home(arguments: PopulateStorageHomeArgs) -> Result<()> {
    let model_root = fetch_model_root(arguments.model_root.as_deref())?;
    let started = std::time::Instant::now();
    let report = crate::storage_homes::populate(&crate::storage_homes::PopulateOptions {
        model_root,
        destination: arguments.dir.clone(),
        budget_bytes: arguments.budget_bytes,
        verify_existing: arguments.verify_existing,
        workers: arguments.workers,
    })?;
    println!(
        "storage home {}: copied {} files ({:.1} GB) in {:.0}s, {} already present{}, {} missing on primary",
        arguments.dir.display(),
        report.copied,
        report.copied_bytes as f64 / 1e9,
        started.elapsed().as_secs_f64(),
        report.already_present,
        if arguments.verify_existing { " and verified" } else { "" },
        report.missing_on_primary,
    );
    println!(
        "use it with: K3_STORAGE_HOMES={}",
        arguments.dir.display()
    );
    Ok(())
}

fn run_warm_expert_cache(arguments: WarmExpertCacheArgs) -> Result<()> {
    let root = fetch_model_root(arguments.model_root.as_deref())?;
    let cache = absolute_under(
        &root,
        arguments
            .cache
            .as_deref()
            .unwrap_or(Path::new("k3-experts")),
    );
    let conversion = if arguments.convert_npz {
        let started = std::time::Instant::now();
        let report = crate::legacy_npz::convert_all(
            &cache,
            crate::legacy_npz::ConversionOptions {
                workers: arguments.convert_workers,
                limit: arguments.convert_limit,
                keep_npz: arguments.keep_npz,
                throttle: Duration::from_millis(arguments.convert_throttle_ms),
            },
            &|progress| {
                if progress.completed == 1
                    || progress.completed == progress.total
                    || progress.completed % 100 == 0
                {
                    eprintln!(
                        "[native-npz] {}/{} complete, converted={}, reused={}, failed={}, {} written",
                        progress.completed,
                        progress.total,
                        progress.converted,
                        progress.reused,
                        progress.failed,
                        format_bytes(progress.bytes_converted),
                    );
                }
            },
        )?;
        eprintln!(
            "[native-npz] durable migration complete: selected={}/{}, converted={}, exact-reused={}, deleted={}, {} written in {:.1}s",
            report.selected_npz,
            report.discovered_npz,
            report.converted_npz,
            report.reused_raw,
            report.deleted_npz,
            format_bytes(report.bytes_converted),
            started.elapsed().as_secs_f64(),
        );
        report
    } else {
        crate::legacy_npz::ConversionReport::default()
    };
    let traces = if arguments.traces.is_empty() {
        discover_router_traces(&root.join("k3-meta"))?
    } else {
        arguments
            .traces
            .iter()
            .map(|path| absolute_under(&root, path))
            .collect()
    };
    let plan = crate::cache_warm::plan(&cache, &traces)?;
    let shown = &plan.candidates[..arguments.show.min(plan.candidates.len())];
    if arguments.json {
        let payload = serde_json::json!({
            "trace_files": &plan.trace_files,
            "observed_routes": plan.observed_routes,
            "unique_observed_experts": plan.unique_observed_experts,
            "missing_observed_experts": plan.missing_observed_experts,
            "missing_observed_bytes": plan.missing_observed_bytes,
            "legacy_npz_discovered": conversion.discovered_npz,
            "legacy_npz_selected": conversion.selected_npz,
            "converted_npz": conversion.converted_npz,
            "already_raw": conversion.reused_raw,
            "deleted_npz": conversion.deleted_npz,
            "converted_npz_bytes": conversion.bytes_converted,
            "top_missing": shown,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).map_err(|error| {
                DeltafinError::new(format!("serialize native warm-cache plan: {error}"))
            })?
        );
    } else {
        println!(
            "routes={} unique={} missing={} ({}) converted={} exact-reused={}",
            plan.observed_routes,
            plan.unique_observed_experts,
            plan.missing_observed_experts,
            format_bytes(plan.missing_observed_bytes),
            conversion.converted_npz,
            conversion.reused_raw,
        );
        for candidate in shown {
            println!(
                "  L{}-E{}: {} observations",
                candidate.layer, candidate.expert, candidate.frequency
            );
        }
    }
    if arguments.fetch > 0 {
        let progress = CliWeightProgress::default();
        let report = crate::cache_warm::fetch_planned(
            &root,
            &plan,
            arguments.fetch,
            crate::weight_fetch::FetchLimits {
                workers: arguments.workers,
                ..crate::weight_fetch::FetchLimits::default()
            },
            &progress,
        )?;
        println!(
            "native expert warming complete: selected={}, completed={}, reused={}, transferred={} in {} requests",
            report.selected_experts,
            report.files_completed,
            report.files_reused,
            format_bytes(report.bytes_transferred),
            report.requests_completed,
        );
    }
    Ok(())
}

fn absolute_under(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn discover_router_traces(metadata_root: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(metadata_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(DeltafinError::new(format!(
                "inspect router-trace directory {}: {error}",
                metadata_root.display()
            )));
        }
    };
    let mut traces = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            DeltafinError::new(format!(
                "read router-trace directory {}: {error}",
                metadata_root.display()
            ))
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("router_trace") && name.ends_with(".jsonl") {
            traces.push(entry.path());
        }
    }
    traces.sort();
    Ok(traces)
}

fn require_successful_benchmark(report: &crate::benchmark::BenchmarkReport) -> Result<()> {
    if report.succeeded() {
        Ok(())
    } else {
        Err(DeltafinError::new(format!(
            "native benchmark did not satisfy its validity/output oracles; inspect {}",
            report.output_dir.display()
        )))
    }
}

fn run_one_shot_setup(arguments: SetupArgs) -> Result<()> {
    let root = fetch_model_root(arguments.model_root.as_deref())?;
    if !arguments.dry_run && !arguments.check {
        ensure_model_root(&root)?;
    }
    let options = crate::one_shot_setup::SetupOptions {
        root,
        workers: arguments.workers,
        include_qwen: arguments.include_qwen,
        mode: arguments.mode,
    };
    if arguments.dry_run {
        let plan = crate::one_shot_setup::plan(&options)?;
        print_setup_plan(&options.root, plan);
        println!("dry run: no directories, files, or network requests were created");
        return Ok(());
    }
    if arguments.check {
        let plan = crate::one_shot_setup::check(&options)?;
        print_setup_plan(&options.root, plan);
        println!("{} native K3 installation audit passed", plan.mode.as_str());
        return Ok(());
    }

    let progress = CliWeightProgress::default();
    let report = crate::one_shot_setup::run(&options, &progress)?;
    print_setup_plan(&options.root, report.final_plan);
    println!(
        "{} native K3 setup complete: {} canonical weight files, {} transferred in {} authenticated HTTPS requests{}",
        report.final_plan.mode.as_str(),
        report.weight_progress.total_files,
        format_bytes(report.weight_progress.bytes_transferred),
        report.weight_progress.requests_completed,
        if options.include_qwen {
            "; optional Qwen assistants included"
        } else {
            ""
        },
    );
    Ok(())
}

fn ensure_model_root(root: &Path) -> Result<()> {
    match sys_fs::lstat(root) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
        Ok(_) => Err(DeltafinError::new(format!(
            "model root is not a real directory: {}",
            root.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = root.parent().ok_or_else(|| {
                DeltafinError::new(format!("model root has no parent: {}", root.display()))
            })?;
            let parent_metadata = sys_fs::lstat(parent).map_err(|error| {
                DeltafinError::new(format!(
                    "inspect model-root parent {}: {error}",
                    parent.display()
                ))
            })?;
            if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
                return Err(DeltafinError::new(format!(
                    "model-root parent is not a real directory: {}",
                    parent.display()
                )));
            }
            crate::sys::fs::create_private_directory(root).map_err(|error| {
                DeltafinError::new(format!("create model root {}: {error}", root.display()))
            })?;
            crate::trusted_download::fsync_directory(parent)
        }
        Err(error) => Err(DeltafinError::new(format!(
            "inspect model root {}: {error}",
            root.display()
        ))),
    }
}

fn print_setup_plan(root: &Path, plan: crate::one_shot_setup::SetupPlan) {
    println!(
        "{} native setup at {}: payload missing={}, peak additional={}, retained headroom={}, available={}, shortfall={}, metadata-authenticated={}, optional-qwen={}",
        plan.mode.as_str(),
        root.display(),
        format_bytes(plan.payload_bytes_missing),
        format_bytes(plan.peak_additional_bytes),
        format_bytes(plan.headroom_bytes),
        format_bytes(plan.available_bytes),
        format_bytes(plan.shortfall_bytes),
        plan.authenticated_inventory,
        plan.includes_qwen,
    );
    if let Some(shortfall) = plan.auto_full_shortfall_bytes {
        println!(
            "auto setup selected streaming because the complete local expert pool needs {} more free space; output quality is unchanged, but uncached experts are fetched on demand",
            format_bytes(shortfall),
        );
    }
}

fn run_convert_experts_scale4(arguments: ConvertExpertsScale4Args) -> Result<()> {
    let mut options = crate::expert_scale4::convert::ConvertOptions::under(&arguments.model_root);
    if let Some(source_root) = arguments.source_root {
        options.source_root = source_root;
    }
    if let Some(output_root) = arguments.output_root {
        options.output_root = output_root;
    }
    options.workers = arguments.workers;
    options.resume = arguments.resume;
    eprintln!(
        "[native-scale4] lossless conversion source={} output={} workers={} resume={}",
        options.source_root.display(),
        options.output_root.display(),
        options.workers,
        options.resume,
    );
    let report = crate::expert_scale4::convert::convert_full(&options)?;
    println!(
        "lossless K3SC4V2 corpus {}: {} records, {} converted, {} resumed, {} sidecars; manifest={}",
        if report.already_complete {
            "already complete and fully authenticated"
        } else {
            "activated"
        },
        report.records,
        report.converted_records,
        report.resumed_records,
        format_bytes(report.sidecar_bytes),
        report.manifest.display(),
    );
    Ok(())
}

fn run_convert_spine_int8(arguments: ConvertSpineInt8Args) -> Result<()> {
    let options = crate::spine_int8::ConvertOptions {
        model_root: arguments.model_root,
        resume: arguments.resume,
        checkpoint_rows: arguments.checkpoint_rows,
    };
    eprintln!(
        "[native-spine-int8] explicit quantized conversion; output is not weight-exact; root={} checkpoint_rows={} resume={}",
        options.model_root.display(),
        options.checkpoint_rows,
        options.resume,
    );
    let mut progress = CliSpineInt8Progress;
    let report = crate::spine_int8::convert_full(&options, &mut progress)?;
    if report.weight_exact {
        return Err(DeltafinError::new(
            "row-int8 converter incorrectly reported a weight-exact representation",
        ));
    }
    println!(
        "explicit row-int8 spine {}: {} targets, {} converted, {} resumed; source={} output={} + {} scales; not weight-exact",
        if report.already_complete {
            "already complete and authenticated"
        } else {
            "activated"
        },
        report.target_tensors,
        report.converted_tensors,
        report.resumed_tensors,
        format_bytes(report.source_bytes),
        format_bytes(report.quantized_bytes),
        format_bytes(report.scale_bytes),
    );
    Ok(())
}

struct CliSpineInt8Progress;

impl crate::spine_int8::ProgressSink for CliSpineInt8Progress {
    fn record(&mut self, event: crate::spine_int8::ProgressEvent<'_>) -> Result<()> {
        use crate::spine_int8::ProgressEvent;
        match event {
            ProgressEvent::Inventory {
                tensors,
                source_bytes,
            } => eprintln!(
                "[native-spine-int8] {tensors} tensors selected from {} of BF16 source",
                format_bytes(source_bytes),
            ),
            ProgressEvent::TensorStarted {
                index,
                tensors,
                name,
                rows,
                resumed_rows,
            } => eprintln!(
                "[native-spine-int8] {index}/{tensors} {name}: rows={rows} resume={resumed_rows}"
            ),
            ProgressEvent::RowsCommitted {
                index,
                tensors,
                name,
                completed_rows,
                rows,
            } if completed_rows == rows => eprintln!(
                "[native-spine-int8] {index}/{tensors} {name}: {completed_rows}/{rows} rows durable"
            ),
            ProgressEvent::ScalesPublished { .. }
            | ProgressEvent::TensorPublished { .. }
            | ProgressEvent::Complete { .. }
            | ProgressEvent::RowsCommitted { .. } => {}
        }
        Ok(())
    }
}

fn run_fetch_weights(arguments: FetchWeightsArgs) -> Result<()> {
    let root = fetch_model_root(arguments.model_root.as_deref())?;
    let selection = fetch_selection(&arguments);
    let inventory = crate::setup_k3::load_installed_inventory(&root)?;
    let limits = crate::weight_fetch::FetchLimits {
        workers: arguments.workers,
        ..crate::weight_fetch::FetchLimits::default()
    };
    let plan = make_weight_plan(&root, &inventory, &arguments, selection, limits)?;
    print_weight_plan(&root, &plan);
    if arguments.dry_run {
        println!("dry run: no directories, files, or network requests were created");
        return Ok(());
    }

    let progress = CliWeightProgress::default();
    let completed = crate::weight_fetch::execute(&plan, &progress)?;
    let verification = make_weight_plan(&root, &inventory, &arguments, selection, limits)?;
    if verification.dry_run.resident_files_missing != 0
        || verification.dry_run.expert_files_missing != 0
        || verification.dry_run.bytes_missing != 0
    {
        return Err(DeltafinError::new(format!(
            "weight installation ended without a complete selected corpus: {} files / {} remain",
            verification.dry_run.resident_files_missing + verification.dry_run.expert_files_missing,
            format_bytes(verification.dry_run.bytes_missing)
        )));
    }
    println!(
        "weight installation complete and locally revalidated: {} files, {} transferred in {} requests",
        completed.total_files,
        format_bytes(completed.bytes_transferred),
        completed.requests_completed,
    );
    Ok(())
}

fn fetch_selection(arguments: &FetchWeightsArgs) -> crate::weight_fetch::WeightSelection {
    if arguments.spine_only {
        crate::weight_fetch::WeightSelection::ResidentSpine
    } else if arguments.experts_only || arguments.layers.is_some() {
        crate::weight_fetch::WeightSelection::ExpertPool
    } else {
        crate::weight_fetch::WeightSelection::Full
    }
}

fn make_weight_plan(
    root: &Path,
    inventory: &crate::inventory::K3Inventory,
    arguments: &FetchWeightsArgs,
    selection: crate::weight_fetch::WeightSelection,
    limits: crate::weight_fetch::FetchLimits,
) -> Result<crate::weight_fetch::WeightFetchPlan> {
    match arguments.layers.as_deref() {
        Some(layers) if selection == crate::weight_fetch::WeightSelection::ExpertPool => {
            crate::weight_fetch::plan_expert_layers(root, inventory, layers, limits)
        }
        Some(_) => Err(DeltafinError::new(
            "routed expert layers cannot be combined with a resident-spine selection",
        )),
        None => crate::weight_fetch::plan(root, inventory, selection, limits),
    }
}

#[derive(Debug)]
struct InstalledPayloadAudit {
    spec: crate::model::ModelSpec,
    mode: crate::one_shot_setup::SetupMode,
    qwen_installed: bool,
}

/// Keep runtime-only diagnosis completely outside the model-root boundary. In
/// particular, do not resolve, stat, or scan a default root before deciding to
/// skip payload validation.
fn audit_doctor_payload(
    arguments: &DoctorArgs,
) -> Result<Option<(PathBuf, InstalledPayloadAudit)>> {
    if arguments.runtime_only {
        return Ok(None);
    }
    let model_root = fetch_model_root(arguments.model_root.as_deref())?;
    let payload = audit_installed_payload(&model_root)?;
    Ok(Some((model_root, payload)))
}

/// Validate the complete local payload selected by the installation without
/// writes or network access. K3 metadata/inventory and draft checkpoints are
/// digest checked by their existing auditors; the multi-terabyte K3 payload is
/// checked as a complete canonical roster of stable regular files with exact
/// inventory-derived sizes rather than misleadingly claiming a 1.7 TB rehash.
/// Prefer a complete full installation; if the expert corpus is intentionally
/// absent, require the complete streaming payload instead.
fn audit_installed_payload(root: &Path) -> Result<InstalledPayloadAudit> {
    let spec = crate::model::ModelSpec::load_from_root(root).map_err(|error| {
        DeltafinError::new(format!(
            "K3 model contract at {} is missing or invalid: {error}",
            root.display()
        ))
    })?;

    // The setup auditor authenticates the pinned nested metadata before it
    // inspects every selected weight. This prevents a valid-looking unpinned
    // root config from standing in for an installation.
    let options = crate::one_shot_setup::SetupOptions {
        root: root.to_owned(),
        workers: crate::weight_fetch::DEFAULT_PARALLEL_TRANSFERS,
        include_qwen: false,
        mode: crate::one_shot_setup::SetupMode::Auto,
    };
    let installation = crate::one_shot_setup::audit_installed(&options)?;

    // Qwen is optional, but a published partial pair is not a valid optional
    // installation. Existing native auditors authenticate every present byte.
    let qwen_installed_bytes = crate::setup_qwen::audited_installed_bytes(root)?;
    let qwen_total_bytes = crate::setup_qwen::exact_install_bytes()?;
    let qwen_installed = match qwen_installed_bytes {
        0 => false,
        installed if installed == qwen_total_bytes => true,
        installed => {
            return Err(DeltafinError::new(format!(
                "optional Qwen installation is incomplete: authenticated {installed} of {qwen_total_bytes} pinned bytes; run `deltafin setup-qwen` to resume it or remove the incomplete add-on"
            )));
        }
    };

    Ok(InstalledPayloadAudit {
        spec,
        mode: installation.mode,
        qwen_installed,
    })
}

/// Read one environment variable as text, failing on a value that exists but
/// is not UTF-8 instead of silently treating it as unset.
fn environment_text(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(DeltafinError::new(format!(
            "{name} must be valid UTF-8 text"
        ))),
    }
}

fn fetch_model_root(explicit: Option<&Path>) -> Result<PathBuf> {
    let selected = explicit
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("DELTAFIN_ROOT").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    if selected.is_absolute() {
        Ok(selected)
    } else {
        std::env::current_dir()
            .map(|current| current.join(selected))
            .map_err(|error| DeltafinError::new(format!("resolve weight model root: {error}")))
    }
}

fn print_weight_plan(root: &Path, plan: &crate::weight_fetch::WeightFetchPlan) {
    let dry = plan.dry_run;
    println!(
        "authenticated K3 weight plan at {}: resident missing/reused={}/{}, experts missing/reused={}/{}, download={}, reuse={}, HTTPS requests={}",
        root.display(),
        dry.resident_files_missing,
        dry.resident_files_reused,
        dry.expert_files_missing,
        dry.expert_files_reused,
        format_bytes(dry.bytes_missing),
        format_bytes(dry.bytes_reused),
        dry.transfer_requests,
    );
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000_000 {
        format!("{:.2} TB", bytes as f64 / 1_000_000_000_000_f64)
    } else if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000_f64)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000_f64)
    } else {
        format!("{bytes} bytes")
    }
}

#[derive(Default)]
struct CliWeightProgress {
    last_files: Mutex<usize>,
}

impl crate::weight_fetch::ProgressSink for CliWeightProgress {
    fn update(&self, progress: crate::weight_fetch::WeightFetchProgress) {
        let files = progress.files_completed + progress.files_reused;
        let mut last = self
            .last_files
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if files == progress.total_files || files >= last.saturating_add(256) {
            *last = files;
            eprintln!(
                "weights: {files}/{} files, {} transferred, {} validated requests",
                progress.total_files,
                format_bytes(progress.bytes_transferred),
                progress.requests_completed,
            );
        }
    }
}

fn socket_address(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn print_split_canary(report: crate::provider::SplitLayerCanaryReport) {
    println!(
        "split-layer ABI {}: PASS positions={} route_edges={} cache_version={} (synthetic transaction; not a full-model parity result)",
        report.device, report.positions, report.route_edges, report.committed_cache_version,
    );
}

fn print_spine_binding_canary(report: crate::provider::SpineBindingCanaryReport) {
    println!(
        "spine-binding ABI {}: PASS tensors={} q8={} raw={} (synthetic ownership/storage canary; not a full-model parity result)",
        report.device, report.tensors, report.quantized_tensors, report.raw_tensors,
    );
}

fn print_kda_canary(report: crate::provider::KdaTransactionCanaryReport) {
    println!(
        "KDA decode ABI {}: PASS cancel_version={} commit_version={} conv_elements={} recurrent_elements={} (synthetic zero-weight transaction; production remains gated)",
        report.device,
        report.canceled_version,
        report.committed_version,
        report.convolution_elements,
        report.recurrent_elements,
    );
}

fn print_mla_canary(report: crate::provider::MlaTransactionCanaryReport) {
    let input_path = if report.input_bundle_rows == 0 {
        "three-call fallback".to_owned()
    } else {
        format!(
            "same-input bundle synthetic_rows={} production_rows={}",
            report.input_bundle_rows, report.production_bundle_rows
        )
    };
    println!(
        "MLA decode ABI {}: PASS cancel_version={} commit_version={} length={} capacity={} input_path={} (synthetic zero-weight transaction; production remains gated)",
        report.device,
        report.canceled_version,
        report.committed_version,
        report.committed_length,
        report.capacity,
        input_path,
    );
}

/// Report the compiled runtime, its linked providers and their exact canaries.
///
/// `doctor` and the end of `setup` share this. It covers the executable and
/// provider ABI rather than installed model data: setup already fails when the
/// payload is incomplete, so repeating the roster audit there would re-stat the
/// whole corpus to prove what has just passed.
fn report_native_runtime(executable: &Path) -> Result<()> {
    println!(
        "runtime artifact: {} (bounded transitive loader-closure audit passed; no Python runtime dependency)",
        executable.display()
    );
    println!("native control plane: ready");
    let inventory = NativeProvider::inventory()?;
    println!("provider ABI: linked without Python");
    println!("LibTorch: {}", inventory.libtorch_version);
    println!(
        "providers: cpu=available mps={} cuda_devices={} cuda_moe_compiled={} cuda_exact_bf16_compiled={}",
        availability(inventory.providers.mps),
        inventory.providers.cuda_devices,
        yes_no(inventory.cuda_moe_compiled),
        yes_no(inventory.cuda_exact_bf16_compiled),
    );
    if let Some(cuda_major) = option_env!("DELTAFIN_LIBTORCH_CUDA_MAJOR") {
        println!("LibTorch CUDA runtime ABI: {cuda_major}.x");
    }
    if option_env!("DELTAFIN_LIBTORCH_HIP").is_some() {
        let experts = if option_env!("DELTAFIN_GPU_KERNEL_RUNTIME") == Some("HIP") {
            "HIP MXFP4 expert kernels are linked but have no token-oracle evidence on AMD hardware"
        } else {
            "MXFP4 experts use the exact CPU path"
        };
        println!("LibTorch GPU runtime: ROCm/HIP (reported as CUDA devices; {experts})");
    }
    if let (Some(runtime), Some(toolkit), Some(architectures)) = (
        option_env!("DELTAFIN_GPU_KERNEL_RUNTIME"),
        option_env!("DELTAFIN_CUDA_TOOLKIT"),
        option_env!("DELTAFIN_GPU_KERNEL_ARCHITECTURES"),
    ) {
        println!(
            "device expert build: {runtime} toolkit {toolkit}, architectures {architectures}"
        );
    }
    if inventory.providers.cuda_devices != 0 && !inventory.cuda_moe_compiled {
        println!(
            "CUDA diagnostic: target tensors are available, but the exact MXFP4 expert kernel was not compiled; automatic expert execution will use the exact CPU fallback"
        );
    } else if inventory.providers.cuda_devices == 0 && inventory.cuda_moe_compiled {
        println!(
            "CUDA diagnostic: the exact MXFP4 expert kernel is linked, but no usable CUDA device is visible to LibTorch"
        );
    }
    if inventory.providers.cuda_devices != 0 && !inventory.cuda_exact_bf16_compiled {
        println!(
            "CUDA original-BF16 diagnostic: the exact RAW_BF16 projection kernel was not compiled; Auto will ignore CUDA only for original-BF16 K3, explicit CUDA will fail early with NVCC rebuild guidance, and --spine int8 remains eligible"
        );
    } else if inventory.providers.cuda_devices == 0 && inventory.cuda_exact_bf16_compiled {
        println!(
            "CUDA original-BF16 diagnostic: the exact RAW_BF16 projection kernel is linked, but no usable CUDA device is visible to LibTorch"
        );
    }
    print_canary(NativeProvider::canary(
        crate::platform::Device::Cpu,
        (32, 32),
        false,
    )?);
    print_split_canary(NativeProvider::split_layer_canary(
        crate::platform::Device::Cpu,
    )?);
    print_spine_binding_canary(NativeProvider::spine_binding_canary(
        crate::platform::Device::Cpu,
    )?);
    print_kda_canary(NativeProvider::kda_transaction_canary(
        crate::platform::Device::Cpu,
    )?);
    print_mla_canary(NativeProvider::mla_transaction_canary(
        crate::platform::Device::Cpu,
    )?);
    if inventory.providers.mps {
        // These are K3's real KDA QKV and output-projection matrix
        // shapes. Packed int8 remains disabled unless both pass with
        // the production fp32 activation/scale contract.
        print_canary(NativeProvider::canary(
            crate::platform::Device::Mps,
            (12_288, 7_168),
            true,
        )?);
        print_canary(NativeProvider::canary(
            crate::platform::Device::Mps,
            (7_168, 12_288),
            true,
        )?);
        print_split_canary(NativeProvider::split_layer_canary(
            crate::platform::Device::Mps,
        )?);
        print_spine_binding_canary(NativeProvider::spine_binding_canary(
            crate::platform::Device::Mps,
        )?);
        print_kda_canary(NativeProvider::kda_transaction_canary(
            crate::platform::Device::Mps,
        )?);
        print_mla_canary(NativeProvider::mla_transaction_canary(
            crate::platform::Device::Mps,
        )?);
    }
    for index in 0..inventory.providers.cuda_devices {
        let device = crate::platform::Device::Cuda(index);
        print_canary(NativeProvider::canary(device, (32, 32), false)?);
        print_split_canary(NativeProvider::split_layer_canary(device)?);
        print_spine_binding_canary(NativeProvider::spine_binding_canary(device)?);
        print_kda_canary(NativeProvider::kda_transaction_canary(device)?);
        print_mla_canary(NativeProvider::mla_transaction_canary(device)?);
    }
    Ok(())
}

fn availability(available: bool) -> &'static str {
    if available {
        "available"
    } else {
        "unavailable"
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn print_canary(report: crate::provider::CanaryReport) {
    println!(
        "canary {}: core={} rms_fp32={} matmul_fp32={} softmax_fp32={} packed_int8_fp32={} packed_shape={}x{}",
        report.device,
        pass_fail(report.core_passed()),
        pass_fail(report.rms_fp32),
        pass_fail(report.matmul_fp32),
        pass_fail(report.softmax_fp32),
        pass_fail(report.packed_int8_fp32),
        report.packed_shape.0,
        report.packed_shape.1,
    );
}

fn pass_fail(passed: bool) -> &'static str {
    if passed { "PASS" } else { "unavailable" }
}

/// Run the production command-line application from this process's arguments.
pub fn run_from_env() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("deltafin: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static DOCTOR_FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct DoctorFixture(PathBuf);

    impl DoctorFixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "deltafin-doctor-test-{}-{}",
                std::process::id(),
                DOCTOR_FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
    }

    impl Drop for DoctorFixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn weight_fetch_defaults_to_the_complete_canonical_corpus() {
        let default = FetchWeightsArgs::default();
        assert_eq!(
            fetch_selection(&default),
            crate::weight_fetch::WeightSelection::Full
        );
        assert_eq!(
            fetch_selection(&FetchWeightsArgs {
                spine_only: true,
                ..default.clone()
            }),
            crate::weight_fetch::WeightSelection::ResidentSpine
        );
        assert_eq!(
            fetch_selection(&FetchWeightsArgs {
                experts_only: true,
                ..default
            }),
            crate::weight_fetch::WeightSelection::ExpertPool
        );
        assert_eq!(
            fetch_selection(&FetchWeightsArgs {
                layers: Some(vec![1, 2]),
                ..FetchWeightsArgs::default()
            }),
            crate::weight_fetch::WeightSelection::ExpertPool
        );
    }

    #[test]
    fn invalid_native_benchmark_report_becomes_a_nonzero_cli_result() {
        let report = crate::benchmark::BenchmarkReport {
            output_dir: PathBuf::from("/tmp/deltafin-invalid-benchmark"),
            summary: serde_json::json!({
                "all_runs_valid": true,
                "all_outputs_exact": false,
            }),
        };
        let error = require_successful_benchmark(&report).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("/tmp/deltafin-invalid-benchmark")
        );
    }

    #[test]
    fn every_substantive_command_requires_native_runtime_preflight() {
        let substantive = [
            vec!["deltafin"],
            vec!["deltafin", "serve"],
            vec!["deltafin", "benchmark"],
            vec!["deltafin", "setup", "--dry-run"],
            vec!["deltafin", "pack-spine", "--verify-only"],
            vec!["deltafin", "setup-dspark", "--check"],
            vec!["deltafin", "setup-k3", "--check"],
            vec!["deltafin", "setup-qwen", "--check"],
            vec!["deltafin", "fetch-weights", "--dry-run"],
            vec!["deltafin", "warm-expert-cache"],
            vec!["deltafin", "convert-spine-int8"],
            vec!["deltafin", "convert-experts-scale4"],
            vec!["deltafin", "doctor"],
            vec!["deltafin", "upgrade"],
        ];
        for (index, arguments) in substantive.into_iter().enumerate() {
            let command = crate::cli::parse(arguments.clone()).unwrap();
            assert!(
                command_requires_native_runtime_preflight(&command),
                "substantive command bypassed preflight: {arguments:?}"
            );
            assert_eq!(
                command_uses_target_runtime(&command),
                index < 3,
                "target-runtime classification changed for {arguments:?}"
            );
        }

        for arguments in [vec!["deltafin", "--help"], vec!["deltafin", "--version"]] {
            let command = crate::cli::parse(arguments.clone()).unwrap();
            assert!(
                !command_requires_native_runtime_preflight(&command),
                "informational command unexpectedly requires preflight: {arguments:?}"
            );
            assert!(!command_uses_target_runtime(&command));
        }
    }

    #[test]
    fn doctor_fails_closed_before_payload_scan_when_model_spec_is_missing() {
        let fixture = DoctorFixture::new();

        let error = audit_installed_payload(&fixture.0).unwrap_err().to_string();

        assert!(error.contains("K3 model contract"));
        assert!(error.contains("missing or invalid"));
        assert!(error.contains("k3-meta/config.json"));
    }

    #[test]
    fn runtime_only_doctor_never_enters_the_model_root_boundary() {
        let fixture = DoctorFixture::new();
        let missing = fixture.0.join("must-not-be-opened");
        let arguments = DoctorArgs {
            model_root: Some(missing.clone()),
            runtime_only: true,
        };

        assert!(audit_doctor_payload(&arguments).unwrap().is_none());
        assert!(!missing.exists());
    }

    #[test]
    fn doctor_fails_closed_on_an_invalid_direct_model_spec() {
        let fixture = DoctorFixture::new();
        fs::write(fixture.0.join("config.json"), b"{}\n").unwrap();

        let error = audit_installed_payload(&fixture.0).unwrap_err().to_string();

        assert!(error.contains("K3 model contract"));
        assert!(error.contains("missing or invalid"));
        assert!(error.contains("model_type"));
    }
}
