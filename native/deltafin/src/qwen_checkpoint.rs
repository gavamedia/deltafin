//! Strict, Python-free admission for Deltafin's two pinned Qwen3 assistants.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::dspark_checkpoint::{digest_from_hex, strict_json};
use crate::error::{DeltafinError, Result};
use crate::packfile::{digest_bytes, digest_open_file};
use crate::sys::fs::{self as sys_fs, Open};

const MAX_HEADER_BYTES: u64 = 1 << 20;
const CONFIG_BYTES: u64 = 727;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QwenVariant {
    Probe06B,
    Wide17B,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct QwenArchitecture {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub layers: u64,
    pub attention_heads: u64,
    pub key_value_heads: u64,
    pub head_dim: u64,
    pub vocabulary_size: u64,
    pub maximum_position: u64,
    pub eos_token_id: u32,
    pub rope_theta: u64,
}

impl QwenVariant {
    pub const fn directory(self) -> &'static str {
        match self {
            Self::Probe06B => "k3-draft-qwen3-0.6b-base",
            Self::Wide17B => "k3-draft-qwen3-1.7b-base",
        }
    }

    pub const fn architecture(self) -> QwenArchitecture {
        let (hidden_size, intermediate_size) = match self {
            Self::Probe06B => (1_024, 3_072),
            Self::Wide17B => (2_048, 6_144),
        };
        QwenArchitecture {
            hidden_size,
            intermediate_size,
            layers: 28,
            attention_heads: 16,
            key_value_heads: 8,
            head_dim: 128,
            vocabulary_size: 151_936,
            maximum_position: 32_768,
            eos_token_id: 151_643,
            rope_theta: 1_000_000,
        }
    }

    const fn checkpoint_bytes(self) -> u64 {
        match self {
            Self::Probe06B => 1_192_135_096,
            Self::Wide17B => 3_441_185_608,
        }
    }

    pub const fn parameter_count(self) -> u64 {
        match self {
            Self::Probe06B => 596_049_920,
            Self::Wide17B => 1_720_574_976,
        }
    }

    const fn header_bytes(self) -> u64 {
        match self {
            Self::Probe06B => 35_248,
            Self::Wide17B => 35_648,
        }
    }

    const fn config_digest(self) -> &'static str {
        match self {
            Self::Probe06B => "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
            Self::Wide17B => "1bb33a92c3548fbc68b889b490e810440435253598835bd71dff0396060c12db",
        }
    }

    const fn checkpoint_digest(self) -> &'static str {
        match self {
            Self::Probe06B => "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
            Self::Wide17B => "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct QwenTensor {
    pub name: String,
    pub shape: Box<[u64]>,
    pub bytes: Range<u64>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
    fn from(metadata: &sys_fs::Stat) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    fn validate(self, file: &File, path: &Path) -> Result<()> {
        let metadata = sys_fs::fstat(&file)
            .map_err(|error| io_error("restat Qwen file", path, error))?;
        if !metadata.is_file() || Self::from(&metadata) != self {
            return Err(DeltafinError::new(format!(
                "Qwen file identity changed while admitted: {}",
                path.display()
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct QwenCheckpoint {
    variant: QwenVariant,
    root: PathBuf,
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    tensors: Box<[QwenTensor]>,
}

impl QwenCheckpoint {
    pub fn open(model_root: &Path, variant: QwenVariant) -> Result<Self> {
        let root = model_root.join(variant.directory());
        validate_config(&root.join("config.json"), variant)?;
        let path = root.join("model.safetensors");
        let (mut file, identity) = open_regular(&path, variant.checkpoint_bytes())?;
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|error| io_error("read Qwen safetensors prefix", &path, error))?;
        let header_bytes = u64::from_le_bytes(prefix);
        if header_bytes != variant.header_bytes() || header_bytes > MAX_HEADER_BYTES {
            return Err(DeltafinError::new(format!(
                "Qwen safetensors header length {header_bytes} differs from its pin"
            )));
        }
        let mut header = vec![0_u8; header_bytes as usize];
        file.read_exact(&mut header)
            .map_err(|error| io_error("read Qwen safetensors header", &path, error))?;
        identity.validate(&file, &path)?;
        let tensors = validate_header(&header, variant, 8 + header_bytes)?;
        Ok(Self {
            variant,
            root,
            path,
            file,
            identity,
            tensors: tensors.into_boxed_slice(),
        })
    }

    pub const fn variant(&self) -> QwenVariant {
        self.variant
    }

    pub const fn architecture(&self) -> QwenArchitecture {
        self.variant.architecture()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn tensors(&self) -> &[QwenTensor] {
        &self.tensors
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    pub(crate) fn validate_live_identity(&self) -> Result<()> {
        self.identity.validate(&self.file, &self.path)
    }

    pub fn verify_full_digest(&self) -> Result<()> {
        let actual = digest_open_file(&self.file, &self.path)
            .map_err(|error| DeltafinError::new(error.to_string()))?;
        self.validate_live_identity()?;
        if actual != digest_from_hex(self.variant.checkpoint_digest())? {
            return Err(DeltafinError::new(
                "Qwen checkpoint SHA-256 does not match the pinned inert artifact",
            ));
        }
        Ok(())
    }
}

fn validate_config(path: &Path, variant: QwenVariant) -> Result<()> {
    let (mut file, identity) = open_regular(path, CONFIG_BYTES)?;
    let mut raw = Vec::with_capacity(CONFIG_BYTES as usize);
    file.read_to_end(&mut raw)
        .map_err(|error| io_error("read Qwen config", path, error))?;
    identity.validate(&file, path)?;
    if digest_bytes(&raw) != digest_from_hex(variant.config_digest())? {
        return Err(DeltafinError::new("Qwen config does not match its pin"));
    }
    let document = strict_json(&raw, "Qwen config")?;
    let architecture = variant.architecture();
    for (name, expected) in [
        ("hidden_size", architecture.hidden_size),
        ("intermediate_size", architecture.intermediate_size),
        ("num_hidden_layers", architecture.layers),
        ("num_attention_heads", architecture.attention_heads),
        ("num_key_value_heads", architecture.key_value_heads),
        ("head_dim", architecture.head_dim),
        ("vocab_size", architecture.vocabulary_size),
        ("max_position_embeddings", architecture.maximum_position),
        ("rope_theta", architecture.rope_theta),
    ] {
        if document.get(name).and_then(Value::as_u64) != Some(expected) {
            return Err(DeltafinError::new(format!(
                "Qwen config field {name} differs from the pinned architecture"
            )));
        }
    }
    if document.get("model_type").and_then(Value::as_str) != Some("qwen3")
        || document.get("hidden_act").and_then(Value::as_str) != Some("silu")
        || document.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true)
        || document.get("attention_bias").and_then(Value::as_bool) != Some(false)
        || document
            .get("auto_map")
            .is_some_and(|value| !value.is_null())
        || document
            .get("trust_remote_code")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return Err(DeltafinError::new(
            "Qwen config is not the built-in, tied, bias-free Qwen3 architecture",
        ));
    }
    Ok(())
}

fn validate_header(raw: &[u8], variant: QwenVariant, data_start: u64) -> Result<Vec<QwenTensor>> {
    let root = strict_json(raw, "Qwen safetensors header")?;
    let root = root
        .as_object()
        .ok_or_else(|| DeltafinError::new("Qwen safetensors header is not an object"))?;
    if root.get("__metadata__") != Some(&serde_json::json!({"format": "pt"})) {
        return Err(DeltafinError::new(
            "Qwen safetensors metadata differs from its pin",
        ));
    }
    let expected = expected_roster(variant.architecture());
    let actual: BTreeSet<_> = root
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect();
    if actual != expected.keys().cloned().collect() {
        return Err(DeltafinError::new(
            "Qwen safetensors tensor roster is not exact",
        ));
    }
    let mut tensors = Vec::with_capacity(expected.len());
    for (name, shape) in expected {
        let descriptor = root[&name]
            .as_object()
            .ok_or_else(|| DeltafinError::new(format!("Qwen tensor {name} is not an object")))?;
        if descriptor.len() != 3 || descriptor.get("dtype").and_then(Value::as_str) != Some("BF16")
        {
            return Err(DeltafinError::new(format!(
                "Qwen tensor {name} is not exact BF16 safetensors data"
            )));
        }
        let actual_shape = parse_u64_array(descriptor.get("shape"), &name)?;
        if actual_shape != shape {
            return Err(DeltafinError::new(format!(
                "Qwen tensor {name} shape differs"
            )));
        }
        let offsets = parse_u64_array(descriptor.get("data_offsets"), &name)?;
        if offsets.len() != 2 || offsets[0] >= offsets[1] {
            return Err(DeltafinError::new(format!(
                "Qwen tensor {name} offsets are invalid"
            )));
        }
        let elements = shape.iter().try_fold(1_u64, |total, value| {
            total
                .checked_mul(*value)
                .ok_or_else(|| DeltafinError::new("Qwen shape overflows"))
        })?;
        let expected_bytes = elements
            .checked_mul(2)
            .ok_or_else(|| DeltafinError::new("Qwen tensor byte length overflows"))?;
        if offsets[1] - offsets[0] != expected_bytes {
            return Err(DeltafinError::new(format!(
                "Qwen tensor {name} byte extent differs"
            )));
        }
        tensors.push(QwenTensor {
            name,
            shape: shape.into_boxed_slice(),
            bytes: data_start + offsets[0]..data_start + offsets[1],
        });
    }
    tensors.sort_by_key(|tensor| tensor.bytes.start);
    let mut cursor = data_start;
    for tensor in &tensors {
        if tensor.bytes.start != cursor {
            return Err(DeltafinError::new(
                "Qwen tensor payload has a gap or overlap",
            ));
        }
        cursor = tensor.bytes.end;
    }
    if cursor != variant.checkpoint_bytes() {
        return Err(DeltafinError::new(
            "Qwen tensor payload does not cover the pinned file",
        ));
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(tensors)
}

fn expected_roster(architecture: QwenArchitecture) -> BTreeMap<String, Vec<u64>> {
    let h = architecture.hidden_size;
    let i = architecture.intermediate_size;
    let q = architecture.attention_heads * architecture.head_dim;
    let kv = architecture.key_value_heads * architecture.head_dim;
    let mut expected = BTreeMap::new();
    expected.insert(
        "model.embed_tokens.weight".into(),
        vec![architecture.vocabulary_size, h],
    );
    expected.insert("model.norm.weight".into(), vec![h]);
    for layer in 0..architecture.layers {
        let prefix = format!("model.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_norm.weight", vec![architecture.head_dim]),
            ("self_attn.k_norm.weight", vec![architecture.head_dim]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("mlp.gate_proj.weight", vec![i, h]),
            ("mlp.up_proj.weight", vec![i, h]),
            ("mlp.down_proj.weight", vec![h, i]),
        ] {
            expected.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
    expected
}

fn parse_u64_array(value: Option<&Value>, name: &str) -> Result<Vec<u64>> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| DeltafinError::new(format!("Qwen tensor {name} has no integer array")))?
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                DeltafinError::new(format!("Qwen tensor {name} array is not unsigned"))
            })
        })
        .collect()
}

fn open_regular(path: &Path, expected_bytes: u64) -> Result<(File, FileIdentity)> {
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| io_error("open pinned Qwen file", path, error))?;
    let metadata = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat pinned Qwen file", path, error))?;
    if !metadata.is_file() || metadata.len() != expected_bytes {
        return Err(DeltafinError::new(format!(
            "pinned Qwen file {} is not a regular {expected_bytes}-byte file",
            path.display()
        )));
    }
    Ok((file, FileIdentity::from(&metadata)))
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_pinned_architectures_have_the_exact_qwen3_roster() {
        for variant in [QwenVariant::Probe06B, QwenVariant::Wide17B] {
            let architecture = variant.architecture();
            let roster = expected_roster(architecture);
            assert_eq!(roster.len(), 310);
            assert_eq!(
                roster["model.embed_tokens.weight"],
                vec![151_936, architecture.hidden_size]
            );
            assert_eq!(
                roster["model.layers.27.self_attn.k_proj.weight"],
                vec![1_024, architecture.hidden_size]
            );
            assert!(!roster.contains_key("lm_head.weight"));
        }
    }

    #[test]
    fn installed_headers_are_admitted_without_materializing_weights() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for variant in [QwenVariant::Probe06B, QwenVariant::Wide17B] {
            if !root.join(variant.directory()).is_dir() {
                continue;
            }
            let checkpoint = QwenCheckpoint::open(&root, variant).unwrap();
            assert_eq!(checkpoint.tensors().len(), 310);
            checkpoint.validate_live_identity().unwrap();
        }
    }
}
