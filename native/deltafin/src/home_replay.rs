//! Replay harness for multi-drive storage experiments.
//!
//! Replays real K3 routes (`k3-meta/router_trace.jsonl`) through the real
//! storage stack in the engine's per-layer order: the layer's spine is
//! streamed one layer ahead on its own reader, the routed expert union is a
//! demand read, and optional per-layer compute time is slept on the critical
//! path. Combined with `K3_STORAGE_EMULATE` (virtual devices by path prefix)
//! this predicts how a multi-drive topology behaves on a one-SSD host.
//!
//! Everything is configured through environment variables so one compiled
//! test binary can sweep many topologies:
//!
//! - `DELTAFIN_REPLAY_PASSES` (6): decode passes replayed.
//! - `DELTAFIN_REPLAY_LAYER_STRIDE` (4): replay every Nth MoE layer.
//! - `DELTAFIN_REPLAY_ROWS` (1): verifier width; each pass reads the union of
//!   this many consecutive decode steps' routes.
//! - `DELTAFIN_REPLAY_COMPUTE_MS` (0): critical-path compute per layer.
//! - `DELTAFIN_REPLAY_SPINE` (1): stream each layer's int8 spine.
//! - `DELTAFIN_REPLAY_STORAGE_HOMES`: `K3_STORAGE_HOMES`-style extra drives.
//! - `DELTAFIN_REPLAY_SPINE_HOMES` (1): let spine reads use the drives too;
//!   `0` keeps the spine on the model root's drive (experts-only homes).
//! - `DELTAFIN_REPLAY_EXPERT_WORKERS` / `DELTAFIN_REPLAY_SPINE_WORKERS`.
//! - `DELTAFIN_REPLAY_LAYOUT` (`scale4`): `scale4` or `raw`.

#![cfg(test)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::experts::{ExpertStorageLayout, K3_MOE_LAYER_FIRST, K3_MOE_LAYER_LAST, RawExpertCorpus};
use crate::storage::{
    BufferKind, BufferLengths, CachePolicy, Extent, ReadPlan, ReadPriority, Reader,
};
use crate::storage_homes::{StorageHomes, parse_storage_homes};

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Decode windows from the JSONL router trace: each element is `rows`
/// consecutive single-row decode steps of one run, as per-layer routes. The
/// trace concatenates many runs whose step counter restarts, so a run ends
/// whenever (step, layer) stops increasing.
fn load_decode_windows(path: &Path, rows: usize) -> Vec<Vec<BTreeMap<u32, Vec<u16>>>> {
    let text = fs::read_to_string(path).unwrap();
    let mut runs: Vec<BTreeMap<u32, BTreeMap<u32, Vec<u16>>>> = Vec::new();
    let mut previous: Option<(u64, u64)> = None;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let (Some(step), Some(layer), Some(ids)) = (
            value.get("step").and_then(serde_json::Value::as_u64),
            value.get("layer").and_then(serde_json::Value::as_u64),
            value.get("ids").and_then(serde_json::Value::as_array),
        ) else {
            continue;
        };
        if previous.is_none_or(|previous| (step, layer) <= previous) {
            runs.push(BTreeMap::new());
        }
        previous = Some((step, layer));
        let ids = ids
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|id| id as u16)
            .collect();
        runs.last_mut()
            .unwrap()
            .entry(step as u32)
            .or_default()
            .insert(layer as u32, ids);
    }
    let mut windows = Vec::new();
    for run in runs {
        let decode: Vec<_> = run
            .into_iter()
            .filter(|(step, layers)| {
                *step > 0
                    && layers.len() == K3_MOE_LAYER_LAST as usize
                    && layers.values().all(|ids| ids.len() == 16)
            })
            .map(|(_, layers)| layers)
            .collect();
        for chunk in decode.chunks_exact(rows) {
            windows.push(chunk.to_vec());
        }
    }
    windows
}

/// One whole-file deferred read plan per transformer layer's spine tensors.
fn spine_plans(root: &Path) -> BTreeMap<u32, (ReadPlan, u64)> {
    let directory = root.join("k3-resident-int8/tensors");
    let mut files: BTreeMap<u32, Vec<(PathBuf, u64)>> = BTreeMap::new();
    for entry in fs::read_dir(&directory).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("language_model.model.layers.") else {
            continue;
        };
        let Some(layer) = rest
            .split('.')
            .next()
            .and_then(|text| text.parse::<u32>().ok())
        else {
            continue;
        };
        files
            .entry(layer)
            .or_default()
            .push((entry.path(), entry.metadata().unwrap().len()));
    }
    files
        .into_iter()
        .map(|(layer, mut tensors)| {
            tensors.sort();
            let mut cursor = 0_usize;
            let mut extents = Vec::with_capacity(tensors.len());
            let mut bytes = 0_u64;
            for (path, length) in tensors {
                extents.push(Extent::new(
                    path,
                    0,
                    BufferKind::Other,
                    cursor,
                    length as usize,
                ));
                cursor += length as usize;
                bytes += length;
            }
            let plan = ReadPlan::open_deferred_manifest(
                extents,
                BufferLengths::new(0, 0, cursor),
                crate::program::DEFAULT_SPINE_CHUNK_BYTES,
                CachePolicy::Streaming,
            )
            .unwrap();
            (layer, (plan, bytes))
        })
        .collect()
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

#[test]
#[ignore = "replays real routes over the installed corpus; configure with DELTAFIN_REPLAY_*"]
fn replay_routes_through_storage() {
    let root = crate::sys::fs::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
    let passes: usize = env_or("DELTAFIN_REPLAY_PASSES", 6);
    let stride: u32 = env_or("DELTAFIN_REPLAY_LAYER_STRIDE", 4);
    let rows: usize = env_or("DELTAFIN_REPLAY_ROWS", 1);
    let compute = Duration::from_secs_f64(env_or("DELTAFIN_REPLAY_COMPUTE_MS", 0.0) / 1e3);
    let stream_spine = env_or("DELTAFIN_REPLAY_SPINE", 1) != 0;
    let homes = std::env::var("DELTAFIN_REPLAY_STORAGE_HOMES").unwrap_or_default();
    let spine_homes = env_or("DELTAFIN_REPLAY_SPINE_HOMES", 1) != 0;
    let layout = match std::env::var("DELTAFIN_REPLAY_LAYOUT").as_deref() {
        Ok("raw") => ExpertStorageLayout::RawV1,
        _ => ExpertStorageLayout::Scale4V2,
    };
    let storage = StorageHomes::open(&root, &parse_storage_homes(&homes).unwrap()).unwrap();
    let drives = storage.as_ref().map_or(1, |homes| homes.device_count());
    let corpus = RawExpertCorpus::open(&root, layout).unwrap();
    let expert_workers: usize = env_or("DELTAFIN_REPLAY_EXPERT_WORKERS", 4 * drives);
    let spine_workers: usize = env_or("DELTAFIN_REPLAY_SPINE_WORKERS", 6 + 2 * (drives - 1));
    let mut expert_reader = Reader::with_arena_capacity(expert_workers.min(16), 1).unwrap();
    expert_reader.set_storage_homes(storage.clone());
    let mut spine_reader = Reader::with_arena_capacity(spine_workers.min(16), 2).unwrap();
    if spine_homes {
        spine_reader.set_storage_homes(storage.clone());
    }

    let windows = load_decode_windows(&root.join("k3-meta/router_trace.jsonl"), rows);
    assert!(
        windows.len() >= passes,
        "trace has only {} decode windows of {rows} rows",
        windows.len()
    );
    let layers: Vec<u32> = (K3_MOE_LAYER_FIRST..=K3_MOE_LAYER_LAST)
        .step_by(stride as usize)
        .collect();
    let spine = if stream_spine {
        spine_plans(&root)
    } else {
        BTreeMap::new()
    };

    eprintln!(
        "[replay] layout={layout:?} drives={drives} spine_homes={spine_homes} passes={passes} layers={}/{} rows={rows} compute={:.1}ms spine={} expert_workers={} spine_workers={}",
        layers.len(),
        K3_MOE_LAYER_LAST,
        compute.as_secs_f64() * 1e3,
        stream_spine,
        expert_reader.workers(),
        spine_reader.workers(),
    );
    let mut expert_latencies = Vec::new();
    let mut spine_waits = Vec::new();
    let mut pass_seconds = Vec::new();
    let mut expert_bytes = 0_u64;
    let mut spine_bytes = 0_u64;
    let started = Instant::now();
    for pass in 0..passes {
        let pass_started = Instant::now();
        // Spread passes over the whole trace rather than one run.
        let window = &windows[pass * windows.len() / passes];
        let mut pending_spine = layers
            .first()
            .and_then(|layer| spine.get(layer))
            .map(|(plan, _)| spine_reader.submit(plan, ReadPriority::Demand).unwrap());
        for (position, &layer) in layers.iter().enumerate() {
            // The layer's spine must be bound before attention can run.
            if let Some(ticket) = pending_spine.take() {
                let waited = Instant::now();
                drop(ticket.wait().unwrap());
                spine_waits.push(waited.elapsed().as_secs_f64());
                spine_bytes += spine.get(&layer).map_or(0, |(_, bytes)| *bytes);
            }
            if let Some((plan, _)) = layers.get(position + 1).and_then(|next| spine.get(next)) {
                pending_spine = Some(spine_reader.submit(plan, ReadPriority::Demand).unwrap());
            }
            std::thread::sleep(compute * 2 / 3);
            let union: Vec<u16> = window
                .iter()
                .flat_map(|layers| layers[&layer].iter().copied())
                .collect::<BTreeSet<u16>>()
                .into_iter()
                .collect();
            let read_started = Instant::now();
            let batch = corpus.read_union(&expert_reader, layer, &union).unwrap();
            expert_latencies.push(read_started.elapsed().as_secs_f64());
            expert_bytes += batch.buffers().other().len() as u64;
            drop(batch);
            std::thread::sleep(compute / 3);
        }
        if let Some(ticket) = pending_spine.take() {
            drop(ticket.wait().unwrap());
        }
        pass_seconds.push(pass_started.elapsed().as_secs_f64());
    }
    let elapsed = started.elapsed().as_secs_f64();
    expert_latencies.sort_by(f64::total_cmp);
    spine_waits.sort_by(f64::total_cmp);
    let mean_pass = pass_seconds.iter().sum::<f64>() / pass_seconds.len() as f64;
    eprintln!(
        "[replay] mean pass {:.3}s (x{} layers -> {:.2}s full-model equivalent); passes {:?}",
        mean_pass,
        f64::from(K3_MOE_LAYER_LAST) / layers.len() as f64,
        mean_pass * f64::from(K3_MOE_LAYER_LAST) / layers.len() as f64,
        pass_seconds
            .iter()
            .map(|value| (value * 1e3).round() / 1e3)
            .collect::<Vec<_>>(),
    );
    eprintln!(
        "[replay] expert read ms p50={:.1} p90={:.1} max={:.1}; spine wait ms p50={:.1} p90={:.1} sum={:.2}s",
        percentile(&expert_latencies, 0.5) * 1e3,
        percentile(&expert_latencies, 0.9) * 1e3,
        percentile(&expert_latencies, 1.0) * 1e3,
        percentile(&spine_waits, 0.5) * 1e3,
        percentile(&spine_waits, 0.9) * 1e3,
        spine_waits.iter().sum::<f64>(),
    );
    eprintln!(
        "[replay] moved experts {:.2} GB + spine {:.2} GB in {:.2}s = {:.2} GB/s",
        expert_bytes as f64 / 1e9,
        spine_bytes as f64 / 1e9,
        elapsed,
        (expert_bytes + spine_bytes) as f64 / 1e9 / elapsed,
    );
    if let Some(homes) = &storage {
        for (index, drive) in homes.stats().iter().enumerate() {
            eprintln!(
                "[replay] drive {index}: {:.2} GB served, {} {:.2} GB/s, {} failed, {}",
                drive.served_bytes as f64 / 1e9,
                if drive.measured {
                    "measured"
                } else {
                    "assumed"
                },
                drive.rate_gbps,
                drive.failures,
                drive.root.display()
            );
        }
    }
    if let Some(emulator) = crate::storage_emulation::DeviceEmulator::global() {
        emulator.print_report();
    }
}
