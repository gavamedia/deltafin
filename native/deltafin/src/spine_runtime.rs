//! Rust-owned resident-spine I/O/upload pipeline.
//!
//! The Python runtime waits for one layer's files, walks a parameter mapping,
//! launches many conversions, then asks Python to prefetch the next layer.
//! This state machine replaces that control path with one persistent native
//! reader and one coarse provider bind per layer.  The next layer's bounded
//! read starts before the current provider upload, so its I/O remains in flight
//! throughout upload and the later layer computation owned by the engine.

use crate::error::{DeltafinError, Result};
use crate::program::LayerSpinePlan;
use crate::provider::{BoundSpineLayerReport, NativeProviderSession, SpineLayerRetention};
use crate::spine_source_use::{ReclaimAdmission, SpineSourceUseController};
use crate::storage::{LayerBuffers, ReadPriority, ReadStats, ReadTicket, Reader};
use std::time::{Duration, Instant};

const K3_LAYER_COUNT: u32 = 93;

#[derive(Debug, Clone, Copy)]
pub struct SpineBindStats {
    pub layer: u32,
    pub generation: u64,
    pub read: ReadStats,
    pub binding: BoundSpineLayerReport,
    pub next_prefetch_started: bool,
    pub reused_resident: bool,
    pub resident_prefix_layers: u32,
    pub resident_storage_bytes: u64,
    /// Caller-observed time spent blocked on the current read ticket. This is
    /// deliberately distinct from `read.elapsed`, which begins when the read
    /// was submitted and therefore includes useful overlap with the preceding
    /// layer. Zero when phase profiling is disabled or the layer is resident.
    pub read_wait: Duration,
    /// Time spent admitting the already-selected successor read. This is
    /// control-plane work, not current-layer I/O or provider upload.
    pub next_prefetch_submit: Duration,
    /// Duration of the existing synchronous provider bind. Depending on the
    /// device this includes conversion/upload, but never adds a new wait.
    pub bind_upload: Duration,
    pub profiled: bool,
}

struct PendingRead {
    layer: u32,
    ticket: ReadTicket,
}

#[derive(Debug, Clone, Copy)]
struct ResidentBinding {
    layer: u32,
    binding: BoundSpineLayerReport,
}

/// A bounded, allocation-reusing layer stream owned by the one Rust process.
///
/// This type performs no Python calls, creates no per-layer worker pools, and
/// accepts either detached V1-compatible payloads or explicit V2 borrows. CPU
/// T=1 production already uses the borrow path; Metal/CUDA preparations use the
/// same ownership protocol. A borrowed arena lease stays inside `source_uses`
/// until an explicit device-fence reclaim or abort.
pub struct SpinePipeline {
    reader: Reader,
    pending: Option<PendingRead>,
    generation: u64,
    resident_prefix_target: u32,
    resident: Vec<ResidentBinding>,
    resident_storage_bytes: u64,
    provider_identity: Option<u64>,
    source_uses: SpineSourceUseController<LayerBuffers>,
    /// Sticky fail-closed state. Once a provider can no longer prove that a
    /// borrowed slab is unused, no read, bind, or source-use transition may
    /// recycle this pipeline's arenas in the current process.
    poisoned: bool,
}

impl SpinePipeline {
    pub fn new(read_workers: usize, arena_slots: usize) -> Result<Self> {
        Self::with_resident_prefix(read_workers, arena_slots, 0)
    }

    pub fn with_resident_prefix(
        read_workers: usize,
        arena_slots: usize,
        resident_prefix_layers: u32,
    ) -> Result<Self> {
        if arena_slots < 2 {
            return Err(DeltafinError::new(
                "spine pipeline needs at least two arena slots for safe read/upload overlap",
            ));
        }
        if resident_prefix_layers > K3_LAYER_COUNT {
            return Err(DeltafinError::new(format!(
                "resident spine prefix cannot exceed {K3_LAYER_COUNT} layers",
            )));
        }
        Ok(Self {
            reader: Reader::with_arena_capacity(read_workers, arena_slots)?,
            pending: None,
            generation: 0,
            resident_prefix_target: resident_prefix_layers,
            resident: Vec::with_capacity(resident_prefix_layers as usize),
            resident_storage_bytes: 0,
            provider_identity: None,
            source_uses: SpineSourceUseController::default(),
            poisoned: false,
        })
    }

    fn require_healthy(&self) -> Result<()> {
        if self.poisoned {
            return Err(DeltafinError::new(
                "spine pipeline is poisoned after an unproven borrowed-source transition",
            ));
        }
        Ok(())
    }

    fn poison(&mut self) {
        self.poisoned = true;
    }

    pub fn workers(&self) -> usize {
        self.reader.workers()
    }

    /// Bytes this pipeline has read from storage since it was created.
    pub fn bytes_read(&self) -> u64 {
        self.reader.bytes_read()
    }

    /// Let spine reads use, and be charged to, the drives in `homes`.
    pub fn set_storage_homes(&mut self, homes: Option<std::sync::Arc<crate::storage_homes::StorageHomes>>) {
        self.reader.set_storage_homes(homes);
    }

    pub fn resident_prefix_target(&self) -> u32 {
        self.resident_prefix_target
    }

    pub fn resident_prefix_layers(&self) -> u32 {
        self.resident.len() as u32
    }

    pub fn resident_storage_bytes(&self) -> u64 {
        self.resident_storage_bytes
    }

    fn retained(&self, layer: u32) -> Option<ResidentBinding> {
        self.resident
            .get(layer as usize)
            .copied()
            .filter(|binding| binding.layer == layer)
    }

    fn next_pending(&self, next: Option<&LayerSpinePlan>) -> Result<Option<PendingRead>> {
        next.filter(|plan| self.retained(plan.layer()).is_none())
            .map(|plan| {
                self.reader
                    .submit(plan.read_plan(), ReadPriority::Demand)
                    .map(|ticket| PendingRead {
                        layer: plan.layer(),
                        ticket,
                    })
            })
            .transpose()
    }

    /// Start the first layer read. A pipeline may have only one unpublished
    /// layer because retaining additional 0.5--1.2 GiB slabs creates severe
    /// memory pressure on the reference 64 GiB host.
    pub fn prime(&mut self, plan: &LayerSpinePlan) -> Result<()> {
        self.require_healthy()?;
        if self.pending.is_some() {
            return Err(DeltafinError::new(
                "spine pipeline already has a pending layer read",
            ));
        }
        if self.retained(plan.layer()).is_some() {
            return Ok(());
        }
        self.pending = Some(PendingRead {
            layer: plan.layer(),
            ticket: self.reader.submit(plan.read_plan(), ReadPriority::Demand)?,
        });
        Ok(())
    }

    /// Finish the current read, launch the next prefetch, then bind every
    /// current-layer tensor through one C ABI call.
    ///
    /// The caller should execute the bound layer immediately after this method
    /// returns. The stored next-layer ticket then overlaps that compute. A
    /// prefetch admission error aborts before provider state changes; a bind
    /// error discards the speculative ticket and leaves the provider's prior
    /// generation intact.
    pub fn bind_current(
        &mut self,
        provider: &NativeProviderSession,
        current: &LayerSpinePlan,
        next: Option<&LayerSpinePlan>,
    ) -> Result<SpineBindStats> {
        self.bind_current_with_timing(provider, current, next, false)
    }

    /// Identical to [`Self::bind_current`] except that it records host-side
    /// durations around waits/calls that already exist. It does not query a
    /// device clock, synchronize a stream, or change source ownership.
    pub fn bind_current_profiled(
        &mut self,
        provider: &NativeProviderSession,
        current: &LayerSpinePlan,
        next: Option<&LayerSpinePlan>,
    ) -> Result<SpineBindStats> {
        self.bind_current_with_timing(provider, current, next, true)
    }

    fn bind_current_with_timing(
        &mut self,
        provider: &NativeProviderSession,
        current: &LayerSpinePlan,
        next: Option<&LayerSpinePlan>,
        profile: bool,
    ) -> Result<SpineBindStats> {
        self.require_healthy()?;
        if let Some(next) = next
            && next.layer() <= current.layer()
        {
            return Err(DeltafinError::new(format!(
                "spine prefetch must advance beyond layer {}; got {}",
                current.layer(),
                next.layer(),
            )));
        }
        if let Some(identity) = self.provider_identity
            && identity != provider.identity()
        {
            return Err(DeltafinError::new(
                "spine pipeline cannot move retained or generation state to a different provider session",
            ));
        }

        if let Some(resident) = self.retained(current.layer()) {
            if self.provider_identity.is_none() {
                return Err(DeltafinError::new(
                    "spine pipeline lost the provider identity for a retained layer",
                ));
            }
            if let Some(pending) = &self.pending {
                return Err(DeltafinError::new(format!(
                    "spine pipeline has an unexpected pending layer {} while reusing resident layer {}",
                    pending.layer,
                    current.layer(),
                )));
            }
            let prefetch_started = profile.then(Instant::now);
            let next_pending = self.next_pending(next)?;
            let next_prefetch_submit = profiled_elapsed(prefetch_started);
            let next_prefetch_started = next_pending.is_some();
            self.pending = next_pending;
            return Ok(SpineBindStats {
                layer: current.layer(),
                generation: resident.binding.generation,
                read: ReadStats {
                    bytes: 0,
                    jobs: 0,
                    workers: self.reader.workers(),
                    elapsed: Duration::ZERO,
                },
                binding: resident.binding,
                next_prefetch_started,
                reused_resident: true,
                resident_prefix_layers: self.resident_prefix_layers(),
                resident_storage_bytes: self.resident_storage_bytes,
                read_wait: Duration::ZERO,
                next_prefetch_submit,
                bind_upload: Duration::ZERO,
                profiled: profile,
            });
        }

        let retention = if current.layer() < self.resident_prefix_target {
            if current.layer() != self.resident_prefix_layers() {
                return Err(DeltafinError::new(format!(
                    "resident spine binding must append layer {}; got layer {}",
                    self.resident_prefix_layers(),
                    current.layer(),
                )));
            }
            SpineLayerRetention::Retained
        } else {
            SpineLayerRetention::Transient
        };
        if let Some(pending) = &self.pending
            && pending.layer != current.layer()
        {
            return Err(DeltafinError::new(format!(
                "spine pipeline expected layer {}, but caller requested layer {}",
                pending.layer,
                current.layer(),
            )));
        }
        let pending = match self.pending.take() {
            Some(pending) => pending,
            None => PendingRead {
                layer: current.layer(),
                ticket: self
                    .reader
                    .submit(current.read_plan(), ReadPriority::Demand)?,
            },
        };

        let read_wait_started = profile.then(Instant::now);
        let (buffers, read) = pending.ticket.wait()?;
        let read_wait = profiled_elapsed(read_wait_started);
        // This is the already-selected next layer in a serialized target
        // pass, not speculative work. Demand admission lets a two-slot arena
        // hold current upload + next read without deadlocking behind the
        // unrelated-prefetch reserve. Already-retained successors need no I/O.
        let prefetch_started = profile.then(Instant::now);
        let next_pending = self.next_pending(next)?;
        let next_prefetch_submit = profiled_elapsed(prefetch_started);
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| DeltafinError::new("spine binding generation is exhausted"))?;
        let bind_started = profile.then(Instant::now);
        let owned_binding = match provider.bind_spine_layer_owned(
            current.layer(),
            generation,
            current.descriptors(),
            buffers,
            retention,
        ) {
            Ok(binding) => binding,
            Err(error) => {
                drop(next_pending);
                return Err(error);
            }
        };
        let bind_upload = profiled_elapsed(bind_started);
        let binding = owned_binding.binding;
        if let Err(error) = self.source_uses.admit(
            provider,
            generation,
            retention,
            binding.source_use,
            owned_binding.source_lease,
        ) {
            // Admission may have failed while attempting to abort an
            // unpublished borrowed token. The controller retains/leaks the
            // lease when cancellation is not provable, so this pipeline must
            // never admit another arena slot.
            self.poison();
            drop(next_pending);
            return Err(error);
        }
        self.provider_identity = Some(provider.identity());
        self.generation = generation;
        if retention == SpineLayerRetention::Retained {
            // Provider inputs are individually bounded to 16 GiB and there
            // are only 93 layers, so this sum is far below u64::MAX.
            self.resident_storage_bytes += binding.resident_storage_bytes;
            self.resident.push(ResidentBinding {
                layer: current.layer(),
                binding,
            });
        }
        let next_prefetch_started = next_pending.is_some();
        self.pending = next_pending;
        Ok(SpineBindStats {
            layer: current.layer(),
            generation,
            read,
            binding,
            next_prefetch_started,
            reused_resident: false,
            resident_prefix_layers: self.resident_prefix_layers(),
            resident_storage_bytes: self.resident_storage_bytes,
            read_wait,
            next_prefetch_submit,
            bind_upload,
            profiled: profile,
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn pending_layer(&self) -> Option<u32> {
        self.pending.as_ref().map(|pending| pending.layer)
    }

    /// True when the selected CPU, Metal, or CUDA provider explicitly borrowed
    /// the current reader-arena slab instead of publishing a detached copy.
    pub fn has_borrowed_source_use(&self) -> bool {
        self.source_uses.has_active_borrow()
    }

    /// Seal a borrowed source after the enclosing layer has submitted every
    /// device operation that can access it. Sealing is consume-once.
    pub fn seal_borrowed_source_use(
        &mut self,
        provider: &NativeProviderSession,
        generation: u64,
    ) -> Result<()> {
        self.require_healthy()?;
        self.source_uses.seal(provider, generation)
    }

    /// Poll a sealed borrowed source without blocking. A completed fence drops
    /// its arena lease before returning, making the freed slot available to
    /// the next `try_submit`/bind admission.
    pub fn try_reclaim_borrowed_source_use(
        &mut self,
        provider: &NativeProviderSession,
        generation: u64,
    ) -> Result<bool> {
        self.require_healthy()?;
        let reclaimed = self
            .source_uses
            .try_reclaim_then(provider, generation, || Ok(None::<()>));
        let reclaimed = match reclaimed {
            Ok(reclaimed) => reclaimed,
            Err(error) => {
                self.poison();
                return Err(error);
            }
        };
        match reclaimed {
            ReclaimAdmission::Pending => Ok(false),
            ReclaimAdmission::Reclaimed(None) => Ok(true),
            ReclaimAdmission::Reclaimed(Some(())) => {
                self.poison();
                Err(DeltafinError::new(
                    "spine source-use reclaim unexpectedly admitted a value",
                ))
            }
        }
    }

    /// Synchronously cancel a borrowed source. If the provider cannot prove
    /// cancellation, the controller retains (and on teardown leaks) the slab
    /// rather than recycling pages that a device might still access.
    pub fn abort_borrowed_source_use(
        &mut self,
        provider: &NativeProviderSession,
        generation: u64,
    ) -> Result<()> {
        match self.source_uses.abort(provider, generation) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.poison();
                Err(error)
            }
        }
    }

    /// Abort the controller's actual active token. This is the only safe
    /// cleanup entry point when a returned binding report is itself stale or
    /// malformed and its generation therefore cannot be trusted.
    pub fn abort_active_borrowed_source_use(
        &mut self,
        provider: &NativeProviderSession,
    ) -> Result<()> {
        let generation = self.source_uses.active_generation().ok_or_else(|| {
            DeltafinError::new("spine pipeline has no active borrowed source to abort")
        })?;
        self.abort_borrowed_source_use(provider, generation)
    }

    /// Terminal teardown while the provider session is still alive. Pending
    /// reads are unpublished and may be discarded directly; a borrowed source
    /// must be synchronously aborted. Failure leaves the controller's lease
    /// live and tells the engine to retain the native provider session too.
    pub(crate) fn teardown(&mut self, provider: &NativeProviderSession) -> Result<()> {
        self.poison();
        self.pending.take();
        if self.source_uses.has_active_borrow() {
            let generation = self.source_uses.active_generation().ok_or_else(|| {
                DeltafinError::new("borrowed spine source lost its active generation at teardown")
            })?;
            self.source_uses
                .abort(provider, generation)
                .map_err(|error| {
                    DeltafinError::new(format!(
                        "abort borrowed spine source during engine teardown: {error}"
                    ))
                })?;
        }
        if self.source_uses.has_unproven_untracked_source() {
            return Err(DeltafinError::new(
                "spine teardown retained an untracked borrowed source after failed admission",
            ));
        }
        Ok(())
    }

    /// Discard an unpublished next-layer ticket after the enclosing target
    /// transaction fails. Dropping the ticket never publishes bytes or provider
    /// state; an already-running read drains inside the bounded reader and then
    /// releases its arena lease. Retained provider bindings remain immutable.
    pub fn discard_pending(&mut self) -> Option<u32> {
        self.pending.take().map(|pending| pending.layer)
    }
}

fn profiled_elapsed(started: Option<Instant>) -> Duration {
    started.map_or(Duration::ZERO, |started| started.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LayerKind;
    use crate::program::{
        SPINE_BUFFER_NONE, SPINE_BUFFER_OTHER, SPINE_ENCODING_RAW_BF16, SpineTensorDescriptorV1,
    };
    use crate::provider::{SpineComponent, SpineStoredScalar};
    use crate::storage::{BufferKind, BufferLengths, CachePolicy, Extent, ReadPlan};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_SOURCE: AtomicU64 = AtomicU64::new(1);

    struct TempSource(PathBuf);

    impl TempSource {
        fn new(bytes: &[u8]) -> Self {
            let serial = NEXT_SOURCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-spine-pipeline-{}-{serial}.bin",
                std::process::id()
            ));
            let mut file = File::create(&path).unwrap();
            file.write_all(bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for TempSource {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn tiny_plan(layer: u32, values: [u8; 4]) -> (TempSource, LayerSpinePlan) {
        let mut slab = vec![0_u8; 256];
        slab[..4].copy_from_slice(&values);
        let source = TempSource::new(&slab);
        let read_plan = ReadPlan::open(
            [Extent::new(&source.0, 0, BufferKind::Other, 0, slab.len())],
            BufferLengths::new(0, 0, slab.len()),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let descriptor = SpineTensorDescriptorV1 {
            slot: 1,
            encoding: SPINE_ENCODING_RAW_BF16,
            rank: 1,
            data_buffer: SPINE_BUFFER_OTHER,
            auxiliary_buffer: SPINE_BUFFER_NONE,
            reserved0: 0,
            shape: [2, 0, 0, 0, 0, 0, 0, 0],
            data_offset: 0,
            data_length: 4,
            auxiliary_offset: 0,
            auxiliary_length: 0,
            reserved: [0; 4],
        };
        (
            source,
            LayerSpinePlan {
                layer,
                kind: LayerKind::Kda,
                descriptors: vec![descriptor].into_boxed_slice(),
                buffer_lengths: BufferLengths::new(0, 0, slab.len()),
                read_plan,
            },
        )
    }

    #[test]
    fn prefetches_next_layer_and_binds_each_layer_once_without_python() {
        let (_first_source, first) = tiny_plan(0, [0x80, 0x3f, 0x00, 0x40]);
        let (_second_source, second) = tiny_plan(1, [0x40, 0x40, 0x80, 0x40]);
        let provider = NativeProviderSession::target(crate::platform::Device::Cpu).unwrap();
        let mut pipeline = SpinePipeline::new(2, 2).unwrap();

        pipeline.prime(&first).unwrap();
        let first_report = pipeline
            .bind_current(&provider, &first, Some(&second))
            .unwrap();
        assert_eq!(first_report.generation, 1);
        assert!(!first_report.profiled);
        assert_eq!(first_report.read_wait, Duration::ZERO);
        assert_eq!(first_report.next_prefetch_submit, Duration::ZERO);
        assert_eq!(first_report.bind_upload, Duration::ZERO);
        assert!(first_report.next_prefetch_started);
        assert_eq!(pipeline.pending_layer(), Some(1));
        let readback = provider
            .read_spine_tensor_f32(0, 1, 1, SpineComponent::Data, 2)
            .unwrap();
        assert_eq!(readback.stored_scalar, SpineStoredScalar::F32);
        assert_eq!(&*readback.values, &[1.0, 2.0]);

        let second_report = pipeline.bind_current(&provider, &second, None).unwrap();
        assert_eq!(second_report.generation, 2);
        assert!(!second_report.next_prefetch_started);
        assert_eq!(pipeline.pending_layer(), None);
        let readback = provider
            .read_spine_tensor_f32(1, 2, 1, SpineComponent::Data, 2)
            .unwrap();
        assert_eq!(&*readback.values, &[3.0, 4.0]);
    }

    #[test]
    fn profiled_bind_observes_existing_boundaries_without_changing_results() {
        let (_first_source, first) = tiny_plan(0, [0x80, 0x3f, 0x00, 0x40]);
        let (_second_source, second) = tiny_plan(1, [0x40, 0x40, 0x80, 0x40]);
        let provider = NativeProviderSession::target(crate::platform::Device::Cpu).unwrap();
        let mut pipeline = SpinePipeline::new(2, 2).unwrap();

        let window = std::time::Instant::now();
        pipeline.prime(&first).unwrap();
        let report = pipeline
            .bind_current_profiled(&provider, &first, Some(&second))
            .unwrap();
        let window = window.elapsed();
        assert!(report.profiled);
        assert_eq!(report.layer, 0);
        assert_eq!(report.read.bytes, 256);
        // The read's lifetime (submit to completion) and the bind's wait for
        // it are two different intervals that each lie inside this window;
        // comparing them with each other races on a loaded host.
        assert!(report.read.elapsed <= window);
        assert!(report.read_wait <= window);
        assert!(report.next_prefetch_started);
        assert_eq!(
            &*provider
                .read_spine_tensor_f32(0, report.generation, 1, SpineComponent::Data, 2)
                .unwrap()
                .values,
            &[1.0, 2.0]
        );
    }

    #[test]
    fn failed_pass_discards_its_unpublished_read_before_reprime() {
        let (_source, first) = tiny_plan(0, [0x80, 0x3f, 0x00, 0x40]);
        let mut pipeline = SpinePipeline::new(1, 2).unwrap();
        pipeline.prime(&first).unwrap();
        assert_eq!(pipeline.pending_layer(), Some(0));
        assert_eq!(pipeline.discard_pending(), Some(0));
        assert_eq!(pipeline.pending_layer(), None);

        // The abandoned ticket may still be draining on the sole worker, but
        // it owns no published state and the second bounded arena slot keeps a
        // deterministic re-prime admissible.
        pipeline.prime(&first).unwrap();
        assert_eq!(pipeline.pending_layer(), Some(0));
        assert_eq!(pipeline.discard_pending(), Some(0));
    }

    #[test]
    fn poisoned_pipeline_never_recycles_or_reuses_its_reader_arena() {
        let (_source, first) = tiny_plan(0, [0x80, 0x3f, 0x00, 0x40]);
        let provider = NativeProviderSession::target(crate::platform::Device::Cpu).unwrap();
        let mut pipeline = SpinePipeline::new(1, 2).unwrap();
        pipeline.prime(&first).unwrap();
        pipeline.poison();

        // Cleanup remains possible, but no subsequent read or bind may enter
        // the arena after an unproven source-use transition.
        assert_eq!(pipeline.discard_pending(), Some(0));
        let prime_error = pipeline.prime(&first).unwrap_err();
        assert!(prime_error.to_string().contains("poisoned"));
        let bind_error = pipeline.bind_current(&provider, &first, None).unwrap_err();
        assert!(bind_error.to_string().contains("poisoned"));
    }

    #[test]
    fn explicit_teardown_runs_while_provider_is_live_and_is_terminal() {
        let (_source, first) = tiny_plan(0, [0x80, 0x3f, 0x00, 0x40]);
        let provider = NativeProviderSession::target(crate::platform::Device::Cpu).unwrap();
        let mut pipeline = SpinePipeline::new(1, 2).unwrap();
        pipeline.prime(&first).unwrap();
        pipeline.teardown(&provider).unwrap();
        assert_eq!(pipeline.pending_layer(), None);
        assert!(
            pipeline
                .prime(&first)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
    }

    #[test]
    fn retained_prefix_skips_second_pass_io_and_survives_transient_churn() {
        let (_first_source, first) = tiny_plan(0, [0x80, 0x3f, 0x00, 0x40]);
        let (_second_source, second) = tiny_plan(1, [0x40, 0x40, 0x80, 0x40]);
        let (_third_source, third) = tiny_plan(2, [0xa0, 0x40, 0xc0, 0x40]);
        let provider = NativeProviderSession::target(crate::platform::Device::Cpu).unwrap();
        let mut pipeline = SpinePipeline::with_resident_prefix(2, 2, 2).unwrap();

        pipeline.prime(&first).unwrap();
        let first_bound = pipeline
            .bind_current(&provider, &first, Some(&second))
            .unwrap();
        assert!(!first_bound.reused_resident);
        assert_eq!(first_bound.binding.retention, SpineLayerRetention::Retained);
        assert_eq!(first_bound.resident_prefix_layers, 1);
        assert_eq!(first_bound.resident_storage_bytes, 8);
        let second_bound = pipeline
            .bind_current(&provider, &second, Some(&third))
            .unwrap();
        assert_eq!(
            second_bound.binding.retention,
            SpineLayerRetention::Retained
        );
        assert_eq!(second_bound.resident_prefix_layers, 2);
        assert_eq!(second_bound.resident_storage_bytes, 16);
        let third_bound = pipeline.bind_current(&provider, &third, None).unwrap();
        assert_eq!(
            third_bound.binding.retention,
            SpineLayerRetention::Transient
        );
        assert_eq!(pipeline.generation(), 3);
        assert_eq!(pipeline.resident_prefix_target(), 2);
        assert_eq!(pipeline.resident_prefix_layers(), 2);
        assert_eq!(pipeline.resident_storage_bytes(), 16);

        // A new serialized pass does no I/O and issues no provider bind for
        // the two retained layers. The first streamed successor begins while
        // the second resident layer is being executed.
        pipeline.prime(&first).unwrap();
        let first_reused = pipeline
            .bind_current(&provider, &first, Some(&second))
            .unwrap();
        assert!(first_reused.reused_resident);
        assert_eq!(first_reused.generation, 1);
        assert_eq!(first_reused.read.bytes, 0);
        assert_eq!(first_reused.read.jobs, 0);
        assert_eq!(first_reused.read.elapsed, Duration::ZERO);
        assert!(!first_reused.next_prefetch_started);
        let second_reused = pipeline
            .bind_current(&provider, &second, Some(&third))
            .unwrap();
        assert!(second_reused.reused_resident);
        assert_eq!(second_reused.generation, 2);
        assert!(second_reused.next_prefetch_started);
        assert_eq!(pipeline.pending_layer(), Some(2));
        let third_rebound = pipeline.bind_current(&provider, &third, None).unwrap();
        assert!(!third_rebound.reused_resident);
        assert_eq!(third_rebound.generation, 4);

        assert_eq!(
            &*provider
                .read_spine_tensor_f32(0, 1, 1, SpineComponent::Data, 2)
                .unwrap()
                .values,
            &[1.0, 2.0]
        );
        assert_eq!(
            &*provider
                .read_spine_tensor_f32(1, 2, 1, SpineComponent::Data, 2)
                .unwrap()
                .values,
            &[3.0, 4.0]
        );
        assert_eq!(
            &*provider
                .read_spine_tensor_f32(2, 4, 1, SpineComponent::Data, 2)
                .unwrap()
                .values,
            &[5.0, 6.0]
        );
    }

    #[test]
    fn resident_prefix_is_bounded_and_cannot_cross_provider_sessions() {
        assert!(SpinePipeline::with_resident_prefix(1, 2, 94).is_err());
        let (_source, first) = tiny_plan(0, [0x80, 0x3f, 0x00, 0x40]);
        let first_provider = NativeProviderSession::target(crate::platform::Device::Cpu).unwrap();
        let second_provider = NativeProviderSession::target(crate::platform::Device::Cpu).unwrap();
        let mut pipeline = SpinePipeline::with_resident_prefix(1, 2, 1).unwrap();
        pipeline.prime(&first).unwrap();
        pipeline
            .bind_current(&first_provider, &first, None)
            .unwrap();
        let error = pipeline
            .bind_current(&second_provider, &first, None)
            .unwrap_err();
        assert!(error.to_string().contains("different provider session"));
    }
}
