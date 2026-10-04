//! Strict admission for the pinned K3 DSpark proposal checkpoint.
//!
//! This parser intentionally performs no dynamic model import and depends on
//! no Python/safetensors runtime. It opens the installed files with
//! `O_NOFOLLOW`, pins their live inode/length, rejects duplicate JSON keys,
//! and validates the complete 68-tensor BF16 allowlist before any provider is
//! permitted to materialize proposal weights.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fmt::{self, Formatter};
use std::fs::File;
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};

use crate::error::{DeltafinError, Result};
use crate::packfile::{Digest, digest_bytes, digest_open_file};
use crate::sys::fs::{self as sys_fs, Open};

pub const OFFICIAL_MODEL_ID: &str = "Inferact/Kimi-K3-DSpark";
pub const OFFICIAL_REVISION: &str = "cf6b8244620e7ea4b0651d214f28e89eac75bed6";
pub const TRAINED_TARGET_REVISION: &str = "cdd2e49a";
pub const CHECKPOINT_BASENAME: &str = "model.safetensors";
pub const CONFIG_BASENAME: &str = "config.json";
pub const OFFICIAL_CHECKPOINT_BYTES: u64 = 7_124_633_450;
pub const OFFICIAL_HEADER_BYTES: u64 = 7_520;
pub const OFFICIAL_PARAMETER_COUNT: u64 = 3_562_312_961;
pub const OFFICIAL_TENSOR_COUNT: usize = 68;
pub const OWNED_TENSOR_COUNT: usize = 67;
pub const MAX_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const OFFICIAL_CONFIG_BYTES: u64 = 1_251;
const MAX_CONFIG_BYTES: u64 = 64 * 1024;
pub(crate) const OFFICIAL_WEIGHTS_SHA256: &str =
    "f9972a636d92a11994cdcfc88fd4c5b5d50d6eb2a89af016031593b8c65c2053";
pub(crate) const OFFICIAL_CONFIG_SHA256: &str =
    "5a3c2f4f91c965ed93b14de5f12a4e9c17fd98d8c99916ed2deb26ce8702f970";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DSparkConfig {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub layers: u64,
    pub attention_heads: u64,
    pub q_lora_rank: u64,
    pub kv_lora_rank: u64,
    pub qk_nope_head_dim: u64,
    pub qk_rope_head_dim: u64,
    pub value_head_dim: u64,
    pub vocab_size: u64,
    pub target_layers: [u32; 5],
    pub mask_token_id: u32,
    pub eos_token_id: u32,
    pub markov_rank: u64,
    pub maximum_drafts: usize,
    pub maximum_context: u64,
}

impl DSparkConfig {
    pub const OFFICIAL: Self = Self {
        hidden_size: 7_168,
        intermediate_size: 14_336,
        layers: 5,
        attention_heads: 64,
        q_lora_rank: 1_536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        value_head_dim: 128,
        vocab_size: 163_840,
        target_layers: [2, 23, 47, 71, 89],
        mask_token_id: 163_837,
        eos_token_id: 163_586,
        markov_rank: 256,
        maximum_drafts: 7,
        maximum_context: 1_048_576,
    };

    pub const fn target_context_width(self) -> u64 {
        self.hidden_size * self.target_layers.len() as u64
    }

    pub const fn latent_cache_width(self) -> u64 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    pub fn load_official(directory: &Path) -> Result<Self> {
        let path = directory.join(CONFIG_BASENAME);
        let (mut file, identity) = open_regular(&path, Some(OFFICIAL_CONFIG_BYTES))?;
        let mut raw = Vec::with_capacity(OFFICIAL_CONFIG_BYTES as usize);
        file.by_ref()
            .take(MAX_CONFIG_BYTES.saturating_add(1))
            .read_to_end(&mut raw)
            .map_err(|error| io_error("read DSpark config", &path, error))?;
        if raw.len() as u64 != OFFICIAL_CONFIG_BYTES {
            return Err(DeltafinError::new(format!(
                "{} changed length while reading",
                path.display()
            )));
        }
        identity.validate(&file, &path)?;
        if digest_bytes(&raw) != digest_from_hex(OFFICIAL_CONFIG_SHA256)? {
            return Err(DeltafinError::new(
                "DSpark config SHA-256 does not match the pinned artifact",
            ));
        }
        let actual = strict_json(&raw, "DSpark config")?;
        let expected = expected_config_document();
        exact_json_object(&actual, &expected, "DSpark config")?;
        Ok(Self::OFFICIAL)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DSparkTensor {
    pub name: String,
    pub shape: Box<[u64]>,
    /// Absolute byte range inside the admitted checkpoint descriptor.
    pub bytes: Range<u64>,
    /// The target embedding is present in the public artifact but must be
    /// borrowed from K3 rather than materialized as a duplicate DSpark weight.
    pub owned_by_dspark: bool,
}

#[derive(Debug)]
pub struct DSparkCheckpoint {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
    tensors: Box<[DSparkTensor]>,
    data_start: u64,
}

impl DSparkCheckpoint {
    pub fn open_official(directory: &Path) -> Result<Self> {
        let path = directory.join(CHECKPOINT_BASENAME);
        if path.file_name() != Some(OsStr::new(CHECKPOINT_BASENAME)) {
            return Err(DeltafinError::new(
                "DSpark checkpoint basename is not allowlisted",
            ));
        }
        let (mut file, identity) = open_regular(&path, Some(OFFICIAL_CHECKPOINT_BYTES))?;
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|error| io_error("read DSpark safetensors prefix", &path, error))?;
        let header_length = u64::from_le_bytes(prefix);
        if !(2..=MAX_SAFETENSORS_HEADER_BYTES).contains(&header_length)
            || header_length != OFFICIAL_HEADER_BYTES
        {
            return Err(DeltafinError::new(format!(
                "DSpark safetensors header is {header_length} bytes; expected pinned {OFFICIAL_HEADER_BYTES}"
            )));
        }
        let mut raw_header = vec![
            0_u8;
            usize::try_from(header_length).map_err(|_| {
                DeltafinError::new("DSpark safetensors header does not fit usize")
            })?
        ];
        file.read_exact(&mut raw_header)
            .map_err(|error| io_error("read DSpark safetensors header", &path, error))?;
        identity.validate(&file, &path)?;

        let (mut tensors, data_start) = validate_official_header(&raw_header)?;
        identity.validate(&file, &path)?;
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(Self {
            path,
            file,
            identity,
            tensors: tensors.into_boxed_slice(),
            data_start,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn tensors(&self) -> &[DSparkTensor] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&DSparkTensor> {
        self.tensors
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }

    pub fn owned_storage_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|tensor| tensor.owned_by_dspark)
            .map(|tensor| tensor.bytes.end - tensor.bytes.start)
            .sum()
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    pub(crate) fn validate_live_identity(&self) -> Result<()> {
        self.identity.validate(&self.file, &self.path)
    }

    /// Full digest verification is deliberately explicit so the future
    /// provider loader can fuse it with its one streaming materialization
    /// pass instead of rereading 7.1 GB at every startup.
    pub fn verify_full_digest(&self) -> Result<()> {
        let digest = digest_open_file(&self.file, &self.path)
            .map_err(|error| DeltafinError::new(error.to_string()))?;
        self.identity.validate(&self.file, &self.path)?;
        if digest != digest_from_hex(OFFICIAL_WEIGHTS_SHA256)? {
            return Err(DeltafinError::new(
                "DSpark checkpoint SHA-256 does not match the pinned artifact",
            ));
        }
        Ok(())
    }
}

pub(crate) fn validate_official_config_bytes(raw: &[u8]) -> Result<()> {
    if raw.len() as u64 != OFFICIAL_CONFIG_BYTES {
        return Err(DeltafinError::new(format!(
            "DSpark config is {} bytes; expected {OFFICIAL_CONFIG_BYTES}",
            raw.len()
        )));
    }
    if digest_bytes(raw) != digest_from_hex(OFFICIAL_CONFIG_SHA256)? {
        return Err(DeltafinError::new(
            "DSpark config SHA-256 does not match the pinned artifact",
        ));
    }
    let actual = strict_json(raw, "DSpark config")?;
    exact_json_object(&actual, &expected_config_document(), "DSpark config")
}

pub(crate) fn validate_official_safetensors_prefix(prefix: &[u8]) -> Result<()> {
    let expected = usize::try_from(8 + OFFICIAL_HEADER_BYTES)
        .map_err(|_| DeltafinError::new("DSpark prefix length does not fit usize"))?;
    if prefix.len() != expected {
        return Err(DeltafinError::new(format!(
            "DSpark safetensors prefix is {} bytes; expected {expected}",
            prefix.len()
        )));
    }
    let header_length = u64::from_le_bytes(prefix[..8].try_into().expect("eight-byte prefix"));
    if header_length != OFFICIAL_HEADER_BYTES {
        return Err(DeltafinError::new(format!(
            "DSpark safetensors header is {header_length} bytes; expected pinned {OFFICIAL_HEADER_BYTES}"
        )));
    }
    validate_official_header(&prefix[8..]).map(|_| ())
}

fn validate_official_header(raw_header: &[u8]) -> Result<(Vec<DSparkTensor>, u64)> {
    if raw_header.len() as u64 != OFFICIAL_HEADER_BYTES {
        return Err(DeltafinError::new(
            "DSpark safetensors header has the wrong length",
        ));
    }
    let root = strict_json(raw_header, "DSpark safetensors header")?;
    let root = root.as_object().ok_or_else(|| {
        DeltafinError::new("DSpark safetensors header root must be a JSON object")
    })?;
    if root.get("__metadata__") != Some(&json!({"torchspec_version": "0.1.0"})) {
        return Err(DeltafinError::new(
            "DSpark safetensors metadata does not match the pin",
        ));
    }
    let expected = checkpoint_schema();
    let actual_count = root.len() - usize::from(root.contains_key("__metadata__"));
    if actual_count != OFFICIAL_TENSOR_COUNT {
        return Err(DeltafinError::new(format!(
            "DSpark checkpoint has {actual_count} tensors; expected {OFFICIAL_TENSOR_COUNT}"
        )));
    }
    let unexpected: Vec<_> = root
        .keys()
        .filter(|name| name.as_str() != "__metadata__" && !expected.contains_key(*name))
        .cloned()
        .collect();
    let missing: Vec<_> = expected
        .keys()
        .filter(|name| !root.contains_key(*name))
        .cloned()
        .collect();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(DeltafinError::new(format!(
            "DSpark checkpoint tensor allowlist differs; missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    let data_start = 8 + OFFICIAL_HEADER_BYTES;
    let data_bytes = OFFICIAL_CHECKPOINT_BYTES - data_start;
    let mut tensors = Vec::with_capacity(expected.len());
    for (name, expected_shape) in expected {
        let descriptor = root
            .get(&name)
            .and_then(Value::as_object)
            .ok_or_else(|| DeltafinError::new(format!("{name}: descriptor is not an object")))?;
        let fields: std::collections::BTreeSet<_> = descriptor.keys().map(String::as_str).collect();
        let required = std::collections::BTreeSet::from(["data_offsets", "dtype", "shape"]);
        if fields != required {
            return Err(DeltafinError::new(format!(
                "{name}: descriptor fields must be exactly dtype, shape, data_offsets"
            )));
        }
        if descriptor.get("dtype").and_then(Value::as_str) != Some("BF16") {
            return Err(DeltafinError::new(format!(
                "{name}: only the pinned BF16 proposal tensor is allowed"
            )));
        }
        let shape = u64_array(descriptor.get("shape"), &format!("{name}.shape"))?;
        if shape.as_slice() != expected_shape.as_slice() {
            return Err(DeltafinError::new(format!(
                "{name}: shape {shape:?} differs from {expected_shape:?}"
            )));
        }
        let offsets = u64_array(
            descriptor.get("data_offsets"),
            &format!("{name}.data_offsets"),
        )?;
        if offsets.len() != 2 || offsets[1] < offsets[0] {
            return Err(DeltafinError::new(format!(
                "{name}: data_offsets must be an increasing pair"
            )));
        }
        let elements = expected_shape
            .iter()
            .try_fold(1_u64, |product, &dimension| {
                product.checked_mul(dimension).ok_or_else(|| {
                    DeltafinError::new(format!("{name}: shape product overflows u64"))
                })
            })?;
        let expected_bytes = elements
            .checked_mul(2)
            .ok_or_else(|| DeltafinError::new(format!("{name}: BF16 length overflows u64")))?;
        if offsets[1] - offsets[0] != expected_bytes || offsets[1] > data_bytes {
            return Err(DeltafinError::new(format!(
                "{name}: data range does not match its exact BF16 shape"
            )));
        }
        tensors.push(DSparkTensor {
            owned_by_dspark: name != "embed_tokens.weight",
            name,
            shape: shape.into_boxed_slice(),
            bytes: data_start + offsets[0]..data_start + offsets[1],
        });
    }
    let mut by_offset: Vec<_> = tensors.iter().collect();
    by_offset.sort_by_key(|tensor| tensor.bytes.start);
    let mut cursor = data_start;
    for tensor in by_offset {
        if tensor.bytes.start != cursor {
            return Err(DeltafinError::new(format!(
                "{}: checkpoint data ranges overlap or leave a gap before byte {}",
                tensor.name, tensor.bytes.start
            )));
        }
        cursor = tensor.bytes.end;
    }
    if cursor != OFFICIAL_CHECKPOINT_BYTES {
        return Err(DeltafinError::new(format!(
            "DSpark checkpoint has trailing or missing payload bytes: covered {cursor}, expected {OFFICIAL_CHECKPOINT_BYTES}"
        )));
    }
    Ok((tensors, data_start))
}

/// Identity of an admitted proposal-model file, checked again around every
/// read so a swapped or rewritten file is refused. Shared with EAGLE-3.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct FileIdentity {
    device: u64,
    inode: u64,
    length: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
}

impl FileIdentity {
    fn from(metadata: &sys_fs::Stat) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            length: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
        }
    }

    pub(crate) fn validate(self, file: &File, path: &Path) -> Result<()> {
        let metadata = sys_fs::fstat(&file)
            .map_err(|error| io_error("stat admitted proposal-model file", path, error))?;
        if !metadata.is_file() || Self::from(&metadata) != self {
            return Err(DeltafinError::new(format!(
                "admitted proposal-model file changed while open: {}",
                path.display()
            )));
        }
        Ok(())
    }
}

pub(crate) fn open_regular(
    path: &Path,
    expected_length: Option<u64>,
) -> Result<(File, FileIdentity)> {
    let before = sys_fs::lstat(path)
        .map_err(|error| io_error("inspect proposal-model file", path, error))?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(DeltafinError::new(format!(
            "proposal-model input must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    if expected_length.is_some_and(|length| before.len() != length) {
        return Err(DeltafinError::new(format!(
            "{} is {} bytes; expected exact pinned length {}",
            path.display(),
            before.len(),
            expected_length.unwrap_or_default(),
        )));
    }
    let file = Open::new()
        .read(true)
        .no_follow()
        .open(path)
        .map_err(|error| {
            io_error("open proposal-model file without following symlinks", path, error)
        })?;
    let opened = sys_fs::fstat(&file)
        .map_err(|error| io_error("stat opened proposal-model file", path, error))?;
    if !opened.is_file()
        || (before.dev(), before.ino()) != (opened.dev(), opened.ino())
        || expected_length.is_some_and(|length| opened.len() != length)
    {
        return Err(DeltafinError::new(format!(
            "proposal-model file changed identity while opening: {}",
            path.display()
        )));
    }
    Ok((file, FileIdentity::from(&opened)))
}

fn checkpoint_schema() -> BTreeMap<String, Vec<u64>> {
    let config = DSparkConfig::OFFICIAL;
    let mut schema = BTreeMap::new();
    schema.insert("confidence_head.proj.bias".into(), vec![1]);
    schema.insert(
        "confidence_head.proj.weight".into(),
        vec![1, config.hidden_size + config.markov_rank],
    );
    schema.insert("context_norm.weight".into(), vec![config.hidden_size]);
    schema.insert(
        "context_proj.weight".into(),
        vec![config.hidden_size, config.target_context_width()],
    );
    schema.insert(
        "embed_tokens.weight".into(),
        vec![config.vocab_size, config.hidden_size],
    );
    schema.insert("final_norm.weight".into(), vec![config.hidden_size]);
    for layer in 0..config.layers {
        let prefix = format!("layers.{layer}");
        schema.insert(
            format!("{prefix}.input_layernorm.weight"),
            vec![config.hidden_size],
        );
        schema.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![config.hidden_size, config.intermediate_size],
        );
        schema.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![config.intermediate_size, config.hidden_size],
        );
        schema.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![config.intermediate_size, config.hidden_size],
        );
        schema.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![config.hidden_size],
        );
        schema.insert(
            format!("{prefix}.self_attn.kv_a_layernorm.weight"),
            vec![config.kv_lora_rank],
        );
        schema.insert(
            format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
            vec![config.latent_cache_width(), config.hidden_size],
        );
        schema.insert(
            format!("{prefix}.self_attn.kv_b_proj.weight"),
            vec![
                config.attention_heads * (config.qk_nope_head_dim + config.value_head_dim),
                config.kv_lora_rank,
            ],
        );
        schema.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![
                config.hidden_size,
                config.attention_heads * config.value_head_dim,
            ],
        );
        schema.insert(
            format!("{prefix}.self_attn.q_a_layernorm.weight"),
            vec![config.q_lora_rank],
        );
        schema.insert(
            format!("{prefix}.self_attn.q_a_proj.weight"),
            vec![config.q_lora_rank, config.hidden_size],
        );
        schema.insert(
            format!("{prefix}.self_attn.q_b_proj.weight"),
            vec![
                config.attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.q_lora_rank,
            ],
        );
    }
    schema.insert(
        "markov_head.markov_w1.weight".into(),
        vec![config.vocab_size, config.markov_rank],
    );
    schema.insert(
        "markov_head.markov_w2.weight".into(),
        vec![config.vocab_size, config.markov_rank],
    );
    schema
}

fn expected_config_document() -> Value {
    json!({
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": false,
        "mla_use_output_gate": false,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 1048576,
        "rope_theta": 50000.0,
        "num_target_layers": 5,
        "target_hidden_size": 7168,
        "target_num_hidden_layers": 93,
        "target_layer_ids": [2, 23, 47, 71, 89],
        "mask_token_id": 163837,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": true,
        "confidence_head_with_markov": true,
        "tie_word_embeddings": false,
        "draft_vocab_size": 163840,
        "_torchspec_version": "0.1.0",
        "torch_dtype": "bfloat16",
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0
        }
    })
}

fn exact_json_object(actual: &Value, expected: &Value, label: &str) -> Result<()> {
    let actual = actual
        .as_object()
        .ok_or_else(|| DeltafinError::new(format!("{label} root must be a JSON object")))?;
    let expected = expected
        .as_object()
        .expect("owned expected config is an object");
    let missing: Vec<_> = expected
        .keys()
        .filter(|key| !actual.contains_key(*key))
        .cloned()
        .collect();
    let extra: Vec<_> = actual
        .keys()
        .filter(|key| !expected.contains_key(*key))
        .cloned()
        .collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(DeltafinError::new(format!(
            "{label} fields differ; missing={missing:?}, extra={extra:?}"
        )));
    }
    for (key, expected_value) in expected {
        if actual.get(key) != Some(expected_value) {
            return Err(DeltafinError::new(format!(
                "{label}.{key} differs from the pinned model contract"
            )));
        }
    }
    Ok(())
}

fn u64_array(value: Option<&Value>, label: &str) -> Result<Vec<u64>> {
    let values = value
        .and_then(Value::as_array)
        .ok_or_else(|| DeltafinError::new(format!("{label} must be an array")))?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| DeltafinError::new(format!("{label} contains a non-u64 value")))
        })
        .collect()
}

pub(crate) fn digest_from_hex(text: &str) -> Result<Digest> {
    if text.len() != 64 {
        return Err(DeltafinError::new(
            "SHA-256 text must contain 64 hex digits",
        ));
    }
    let mut digest = [0_u8; 32];
    for (index, output) in digest.iter_mut().enumerate() {
        let offset = index * 2;
        *output = u8::from_str_radix(&text[offset..offset + 2], 16)
            .map_err(|_| DeltafinError::new("SHA-256 text contains a non-hex digit"))?;
    }
    Ok(digest)
}

#[derive(Debug)]
struct StrictJson(Value);

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("a duplicate-free JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(StrictJson)
            .ok_or_else(|| E::custom("JSON number is not finite"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::String(value)))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictJson>()? {
            values.push(value.0);
        }
        Ok(StrictJson(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON key is forbidden: {key}"
                )));
            }
            let value = object.next_value::<StrictJson>()?;
            values.insert(key, value.0);
        }
        Ok(StrictJson(Value::Object(values)))
    }
}

pub(crate) fn strict_json(raw: &[u8], label: &str) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let value = StrictJson::deserialize(&mut deserializer)
        .map_err(|error| DeltafinError::new(format!("invalid {label}: {error}")))?;
    deserializer
        .end()
        .map_err(|error| DeltafinError::new(format!("invalid trailing {label} data: {error}")))?;
    Ok(value.0)
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_the_exact_public_parameter_roster() {
        let schema = checkpoint_schema();
        assert_eq!(schema.len(), OFFICIAL_TENSOR_COUNT);
        let elements = schema
            .values()
            .fold(0_u64, |sum, shape| sum + shape.iter().product::<u64>());
        assert_eq!(elements, OFFICIAL_PARAMETER_COUNT);
        assert_eq!(
            elements * 2 + 8 + OFFICIAL_HEADER_BYTES,
            OFFICIAL_CHECKPOINT_BYTES
        );
        assert_eq!(
            schema
                .keys()
                .filter(|name| name.as_str() != "embed_tokens.weight")
                .count(),
            OWNED_TENSOR_COUNT
        );
    }

    #[test]
    fn duplicate_json_keys_are_rejected_at_every_depth() {
        assert!(strict_json(br#"{"a":1,"a":2}"#, "test JSON").is_err());
        assert!(strict_json(br#"{"a":{"b":1,"b":2}}"#, "test JSON").is_err());
        assert!(strict_json(br#"{"a":[{"b":1,"b":2}]}"#, "test JSON").is_err());
        assert_eq!(
            strict_json(br#"{"a":[1,true,null]}"#, "test JSON").unwrap()["a"][0],
            1
        );
    }

    #[test]
    fn installed_official_config_and_header_are_admitted_without_tensor_runtime() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap();
        let directory = repository.join("k3-draft-dspark");
        if !directory.is_dir() {
            return;
        }
        assert_eq!(
            DSparkConfig::load_official(&directory).unwrap(),
            DSparkConfig::OFFICIAL
        );
        let checkpoint = DSparkCheckpoint::open_official(&directory).unwrap();
        assert_eq!(checkpoint.tensors().len(), OFFICIAL_TENSOR_COUNT);
        assert_eq!(checkpoint.data_start(), 8 + OFFICIAL_HEADER_BYTES);
        assert_eq!(checkpoint.owned_storage_bytes(), 4_775_815_682);
        assert!(
            !checkpoint
                .tensor("embed_tokens.weight")
                .unwrap()
                .owned_by_dspark
        );
        assert!(
            checkpoint
                .tensor("layers.4.self_attn.q_b_proj.weight")
                .is_some()
        );
    }

    #[test]
    fn pinned_digests_decode_to_exact_sha256_width() {
        assert_eq!(digest_from_hex(OFFICIAL_WEIGHTS_SHA256).unwrap().len(), 32);
        assert!(digest_from_hex("00").is_err());
        assert!(digest_from_hex(&"z".repeat(64)).is_err());
    }
}
