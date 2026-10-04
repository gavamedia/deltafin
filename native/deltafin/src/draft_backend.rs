//! The engine's one optional proposal slot. DSpark and EAGLE-3.1 share the
//! transactional controller in [`crate::dspark_runtime`]; whichever loaded
//! decides which K3 layers a target sequence captures for it.

use crate::dspark_provider::{NativeDSparkBackend, NativeDSparkSnapshot};
use crate::dspark_runtime::{BackendFailure, BackendProposal, DraftBackend, ModelIdentity};
use crate::eagle3_provider::{NativeEagle3Backend, NativeEagle3Snapshot};
use crate::provider::{ProposalCapture, ProviderTensor};

pub(crate) enum NativeDraftBackend {
    DSpark(NativeDSparkBackend),
    Eagle3(NativeEagle3Backend),
}

#[derive(Clone)]
pub(crate) enum NativeDraftSnapshot {
    DSpark(NativeDSparkSnapshot),
    Eagle3(NativeEagle3Snapshot),
}

impl NativeDraftBackend {
    pub(crate) const fn capture(&self) -> ProposalCapture {
        match self {
            Self::DSpark(_) => ProposalCapture::DSpark,
            Self::Eagle3(_) => ProposalCapture::Eagle3,
        }
    }

    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::DSpark(_) => "dspark",
            Self::Eagle3(_) => "eagle3",
        }
    }
}

impl DraftBackend for NativeDraftBackend {
    type Snapshot = NativeDraftSnapshot;
    type TargetContext = ProviderTensor;

    fn reset_state(&mut self) -> std::result::Result<(), BackendFailure> {
        match self {
            Self::DSpark(backend) => backend.reset_state(),
            Self::Eagle3(backend) => backend.reset_state(),
        }
    }

    fn snapshot_state(&mut self) -> std::result::Result<Self::Snapshot, BackendFailure> {
        match self {
            Self::DSpark(backend) => backend.snapshot_state().map(NativeDraftSnapshot::DSpark),
            Self::Eagle3(backend) => backend.snapshot_state().map(NativeDraftSnapshot::Eagle3),
        }
    }

    fn restore_state(
        &mut self,
        snapshot: &Self::Snapshot,
    ) -> std::result::Result<(), BackendFailure> {
        match (self, snapshot) {
            (Self::DSpark(backend), NativeDraftSnapshot::DSpark(snapshot)) => {
                backend.restore_state(snapshot)
            }
            (Self::Eagle3(backend), NativeDraftSnapshot::Eagle3(snapshot)) => {
                backend.restore_state(snapshot)
            }
            _ => Err(BackendFailure::new(
                "proposal snapshot belongs to another draft model",
            )),
        }
    }

    fn state_token_count(&mut self) -> std::result::Result<usize, BackendFailure> {
        match self {
            Self::DSpark(backend) => backend.state_token_count(),
            Self::Eagle3(backend) => backend.state_token_count(),
        }
    }

    fn model_identity(&mut self) -> std::result::Result<ModelIdentity, BackendFailure> {
        match self {
            Self::DSpark(backend) => backend.model_identity(),
            Self::Eagle3(backend) => backend.model_identity(),
        }
    }

    fn propose(
        &mut self,
        pending_token_id: u32,
        max_drafts: u8,
    ) -> std::result::Result<BackendProposal, BackendFailure> {
        match self {
            Self::DSpark(backend) => backend.propose(pending_token_id, max_drafts),
            Self::Eagle3(backend) => backend.propose(pending_token_id, max_drafts),
        }
    }

    fn advance_target_state(
        &mut self,
        target_context: &Self::TargetContext,
        committed_input_ids: &[u32],
    ) -> std::result::Result<(), BackendFailure> {
        match self {
            Self::DSpark(backend) => {
                backend.advance_target_state(target_context, committed_input_ids)
            }
            Self::Eagle3(backend) => {
                backend.advance_target_state(target_context, committed_input_ids)
            }
        }
    }
}
