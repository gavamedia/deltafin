//! Reusable native Deltafin core.
//!
//! The production executable enables the `runtime` feature. Lightweight
//! in-process consumers can disable it and reuse the exact tokenizer and
//! streaming decoder without discovering or initializing LibTorch, the spine,
//! expert storage, or an accelerator provider.

// Native provider and positional-I/O boundaries necessarily use small audited
// unsafe blocks. Keep operations inside unsafe functions explicit instead of
// banning the FFI and disjoint-buffer primitives the runtime requires.
#![deny(unsafe_op_in_unsafe_fn)]
#![recursion_limit = "256"]

pub mod chat;
pub mod decode;
pub mod draft;
pub mod error;
#[cfg(any(feature = "runtime", test))]
mod loader_audit;
pub mod openai;
pub mod output;
// Operating-system primitives live in the shared `deltafin-sys` crate so the
// build-time installer, the build graph and the runtime use one
// implementation. The runtime keeps finding them under `crate::sys`.
#[cfg(any(feature = "runtime", test))]
#[allow(unused_imports)]
mod sys {
    pub(crate) use deltafin_sys::{fs, limits, path, pe};
}
pub mod tokenizer;
#[cfg(any(feature = "runtime", test))]
mod upgrade;

#[cfg(feature = "runtime")]
mod app;
#[cfg(feature = "runtime")]
pub mod benchmark;
#[cfg(feature = "runtime")]
pub mod cache_warm;
#[cfg(feature = "runtime")]
mod cli;
#[cfg(feature = "runtime")]
mod config;
#[cfg(feature = "runtime")]
pub mod dspark_checkpoint;
#[cfg(feature = "runtime")]
pub mod dspark_provider;
#[cfg(feature = "runtime")]
pub mod dspark_runtime;
#[cfg(feature = "runtime")]
mod dspark_setup;
#[cfg(feature = "runtime")]
mod draft_backend;
#[cfg(feature = "runtime")]
pub mod eagle3_provider;
#[cfg(feature = "runtime")]
pub mod embedding;
#[cfg(feature = "runtime")]
mod engine;
#[cfg(feature = "runtime")]
mod expert_heat;
#[cfg(feature = "runtime")]
pub mod expert_scale4;
#[cfg(feature = "runtime")]
pub mod experts;
#[cfg(feature = "runtime")]
pub mod inventory;
#[cfg(feature = "runtime")]
mod io_priority;
#[cfg(feature = "runtime")]
pub mod k3_source;
#[cfg(feature = "runtime")]
pub mod legacy_npz;
#[cfg(feature = "runtime")]
mod model;
#[cfg(feature = "runtime")]
pub(crate) mod one_shot_setup;
#[cfg(feature = "runtime")]
mod pack_command;
#[cfg(feature = "runtime")]
pub mod packfile;
#[cfg(feature = "runtime")]
mod pilot_gate;
#[cfg(feature = "runtime")]
mod platform;
#[cfg(feature = "runtime")]
mod program;
#[cfg(feature = "runtime")]
mod provider;
#[cfg(feature = "runtime")]
mod quality;
#[cfg(feature = "runtime")]
pub mod qwen_checkpoint;
#[cfg(feature = "runtime")]
pub mod qwen_draft;
#[cfg(feature = "runtime")]
pub mod qwen_provider;
#[cfg(feature = "runtime")]
pub mod residency;
#[cfg(feature = "runtime")]
mod router_trace;
#[cfg(feature = "runtime")]
pub mod routing;
#[cfg(feature = "runtime")]
mod run_events;
#[cfg(feature = "runtime")]
mod run_interrupt;
#[cfg(feature = "runtime")]
mod setup_k3;
#[cfg(feature = "runtime")]
mod setup_qwen;
#[cfg(feature = "runtime")]
pub mod spine_int8;
#[cfg(feature = "runtime")]
pub mod spine_runtime;
#[cfg(feature = "runtime")]
mod spine_source_use;
#[cfg(feature = "runtime")]
pub mod storage;
#[cfg(feature = "runtime")]
mod storage_emulation;
#[cfg(feature = "runtime")]
pub mod storage_homes;
#[cfg(all(feature = "runtime", test))]
mod home_replay;
#[cfg(feature = "runtime")]
mod trusted_download;
#[cfg(feature = "runtime")]
pub mod weight_fetch;

#[cfg(feature = "runtime")]
pub use app::run_from_env;
