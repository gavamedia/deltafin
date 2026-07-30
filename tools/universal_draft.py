"""Stateless cross-tokenizer draft model for exact speculative decoding.

The assistant is never trusted to emit a token.  It writes a short text
continuation, that text is re-tokenized with K3's tokenizer, and the ordinary
K3 forward pass verifies every candidate before it can be committed.  A bad
assistant therefore costs time but cannot change greedy output.

The first implementation deliberately discards the assistant KV cache after
each proposal.  Rebuilding a sub-billion-parameter assistant is cheap compared
with one K3 spine sweep and avoids subtle cache-cropping errors when the two
tokenizers disagree about boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

import torch


FALSE_MODES = {"", "0", "false", "off", "no", "disabled"}
TRUE_MODES = {"1", "true", "on", "yes", "enabled"}


@dataclass(frozen=True)
class Proposal:
    token_ids: tuple[int, ...]
    assistant_tokens: int
    seconds: float
    confidence_stopped: bool = False
    raw_token_ids: tuple[int, ...] = ()
    minimum_confidence: float | None = None


class UniversalDraftError(RuntimeError):
    pass


class UniversalTextDrafter:
    """Greedy local assistant whose text is translated into target token IDs."""

    def __init__(
        self,
        model,
        assistant_tokenizer,
        target_tokenizer,
        device,
        *,
        max_assistant_tokens: int = 20,
        confidence_threshold: float = 0.0,
    ):
        if max_assistant_tokens < 1:
            raise ValueError("max_assistant_tokens must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model = model
        self.assistant_tokenizer = assistant_tokenizer
        self.target_tokenizer = target_tokenizer
        self.device = torch.device(device)
        self.max_assistant_tokens = int(max_assistant_tokens)
        self.confidence_threshold = float(confidence_threshold)
        self.proposals = 0
        self.failures = 0
        self.assistant_tokens = 0
        self.target_drafts = 0
        self.accepted_drafts = 0
        self.emitted_tokens = 0
        self.seconds = 0.0
        self.confidence_truncations = 0
        self.confidence_skips = 0

    def _target_decode(self, token_ids) -> str:
        # K3's tokenizer has a direct tiktoken path when no generic
        # Transformers kwargs are supplied.  Besides being faster, it preserves
        # structural special tokens exactly for chat histories.
        return self.target_tokenizer.decode(list(token_ids))

    def _target_encode(self, text: str) -> list[int]:
        return [
            int(token)
            for token in self.target_tokenizer.encode(
                text, allow_special_tokens=True
            )
        ]

    @torch.inference_mode()
    def propose(self, target_history, max_target_drafts: int) -> Proposal:
        if max_target_drafts < 1:
            return Proposal((), 0, 0.0)
        started = time.perf_counter()
        history = [int(token) for token in target_history]
        canonical = self._target_decode(history)
        if self._target_encode(canonical) != history:
            raise UniversalDraftError(
                "target history does not round-trip through canonical text"
            )

        encoded = self.assistant_tokenizer(
            canonical,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        def generate_and_translate(new_tokens):
            # Admission probes are deliberately only two drafts wide. Verify
            # those directly: a confidence false-negative there can prevent a
            # good assistant from ever reaching the fast lane, while avoiding
            # a bad T=3 pass saves little. Confidence gating earns its keep on
            # the expensive wider verifier shapes.
            scored = (
                self.confidence_threshold > 0
                and max_target_drafts > 2
            )
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.assistant_tokenizer.eos_token_id,
                return_dict_in_generate=scored,
                output_scores=scored,
            )
            sequences = output.sequences if scored else output
            generated = int(sequences.shape[1] - input_ids.shape[1])
            keep = generated
            confidence_stopped = False
            minimum_confidence = None
            if scored and generated:
                selected = sequences[
                    0, input_ids.shape[1]:input_ids.shape[1] + generated
                ]
                # Compute all selected-token probabilities on the assistant
                # device, then make one small host transfer. This is display/
                # policy metadata only; target verification remains exact.
                selected_logits = torch.stack([
                    score[0, token]
                    for score, token in zip(output.scores, selected)
                ]).to(torch.float32)
                normalizers = torch.stack([
                    torch.logsumexp(score[0].to(torch.float32), dim=-1)
                    for score in output.scores
                ])
                probabilities = torch.exp(
                    selected_logits - normalizers
                ).detach().cpu().tolist()
                minimum_confidence = min(probabilities)
                for index, probability in enumerate(probabilities):
                    if probability < self.confidence_threshold:
                        keep = index
                        confidence_stopped = True
                        break
            raw_text = self.assistant_tokenizer.decode(
                sequences[0].detach().cpu().tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            full_text = self.assistant_tokenizer.decode(
                sequences[
                    0, :input_ids.shape[1] + keep
                ].detach().cpu().tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            raw_translated_ids = self._target_encode(raw_text)
            translated_ids = self._target_encode(full_text)
            prefix_ok = (
                raw_translated_ids[:len(history)] == history
                and translated_ids[:len(history)] == history
            )
            return (
                translated_ids,
                raw_translated_ids,
                generated,
                prefix_ok,
                confidence_stopped,
                minimum_confidence,
            )

        # Similar BPE tokenizers normally need only a little more than one
        # assistant token per requested target token. Avoid generating 20
        # tokens for a two-draft probe. If that conservative first budget does
        # not yield enough translated IDs, retry once at the configured ceiling.
        proposal_budget = min(
            self.max_assistant_tokens,
            max(4, int(max_target_drafts) + 2),
        )
        (
            translated,
            raw_translated,
            generated_count,
            prefix_ok,
            confidence_stopped,
            minimum_confidence,
        ) = generate_and_translate(proposal_budget)
        drafts = tuple(
            translated[len(history):len(history) + max_target_drafts]
        ) if prefix_ok else ()
        raw_drafts = tuple(
            raw_translated[len(history):len(history) + max_target_drafts]
        ) if prefix_ok else ()
        if (
            (not prefix_ok or len(drafts) < max_target_drafts)
            and proposal_budget < self.max_assistant_tokens
            and generated_count >= proposal_budget
            and not confidence_stopped
        ):
            (
                translated,
                raw_translated,
                retry_count,
                prefix_ok,
                confidence_stopped,
                minimum_confidence,
            ) = generate_and_translate(self.max_assistant_tokens)
            generated_count += retry_count
            drafts = tuple(
                translated[len(history):len(history) + max_target_drafts]
            ) if prefix_ok else ()
            raw_drafts = tuple(
                raw_translated[
                    len(history):len(history) + max_target_drafts
                ]
            ) if prefix_ok else ()
        if not prefix_ok:
            raise UniversalDraftError(
                "assistant text did not preserve the exact target-token prefix"
            )
        elapsed = time.perf_counter() - started
        self.proposals += 1
        self.assistant_tokens += generated_count
        self.target_drafts += len(drafts)
        self.seconds += elapsed
        if confidence_stopped:
            self.confidence_truncations += 1
            if not drafts:
                self.confidence_skips += 1
        return Proposal(
            drafts,
            generated_count,
            elapsed,
            confidence_stopped=confidence_stopped,
            raw_token_ids=raw_drafts,
            minimum_confidence=minimum_confidence,
        )

    def record_verified(self, accepted_drafts: int, emitted_tokens: int) -> None:
        self.accepted_drafts += int(accepted_drafts)
        self.emitted_tokens += int(emitted_tokens)

    def status(self) -> dict[str, Any]:
        return {
            "proposals": self.proposals,
            "failures": self.failures,
            "assistant_tokens": self.assistant_tokens,
            "target_drafts": self.target_drafts,
            "accepted_drafts": self.accepted_drafts,
            "emitted_tokens": self.emitted_tokens,
            "seconds": self.seconds,
            "max_assistant_tokens": self.max_assistant_tokens,
            "confidence_threshold": self.confidence_threshold,
            "confidence_truncations": self.confidence_truncations,
            "confidence_skips": self.confidence_skips,
            "device": str(self.device),
        }


class HybridTextDrafter:
    """Use a small probe model and confidence-select wider proposals.

    When both assistants independently produce the same complete target-token
    draft, verify that consensus without confidence truncation. On divergence,
    prefer the larger model only when its weakest generated-token probability
    beats the probe by a real margin; otherwise retain the cheaper probe.
    Assistant output is still only a proposal and remains target-verified.
    """

    WIDE_CONFIDENCE_MARGIN = 0.02

    def __init__(self, probe, wide):
        self.probe = probe
        self.wide = wide
        self.device = wide.device
        self.max_assistant_tokens = max(
            probe.max_assistant_tokens, wide.max_assistant_tokens
        )
        self.confidence_threshold = wide.confidence_threshold
        self.proposals = 0
        self.failures = 0
        self.assistant_tokens = 0
        self.target_drafts = 0
        self.accepted_drafts = 0
        self.emitted_tokens = 0
        self.seconds = 0.0
        self.confidence_truncations = 0
        self.confidence_skips = 0
        self.probe_selections = 0
        self.wide_selections = 0
        self.consensus_selections = 0

    @staticmethod
    def _raw(proposal):
        return proposal.raw_token_ids or proposal.token_ids

    @staticmethod
    def _score(proposal):
        value = proposal.minimum_confidence
        return -1.0 if value is None else float(value)

    def propose(self, target_history, max_target_drafts: int) -> Proposal:
        if max_target_drafts <= 2:
            result = self.probe.propose(
                target_history, max_target_drafts
            )
            self.probe_selections += 1
        else:
            probe = self.probe.propose(
                target_history, max_target_drafts
            )
            wide = self.wide.propose(
                target_history, max_target_drafts
            )
            probe_raw = self._raw(probe)
            wide_raw = self._raw(wide)
            total_tokens = (
                probe.assistant_tokens + wide.assistant_tokens
            )
            total_seconds = probe.seconds + wide.seconds
            if (
                probe_raw
                and len(probe_raw) == max_target_drafts
                and probe_raw == wide_raw
            ):
                # Independent agreement is stronger evidence than either
                # model's isolated low-confidence calibration.
                result = Proposal(
                    probe_raw,
                    total_tokens,
                    total_seconds,
                    raw_token_ids=probe_raw,
                    minimum_confidence=max(
                        self._score(probe), self._score(wide)
                    ),
                )
                self.consensus_selections += 1
            elif (
                self._score(wide)
                > self._score(probe) + self.WIDE_CONFIDENCE_MARGIN
            ):
                result = Proposal(
                    wide.token_ids,
                    total_tokens,
                    total_seconds,
                    confidence_stopped=wide.confidence_stopped,
                    raw_token_ids=wide_raw,
                    minimum_confidence=wide.minimum_confidence,
                )
                self.wide_selections += 1
            else:
                result = Proposal(
                    probe.token_ids,
                    total_tokens,
                    total_seconds,
                    confidence_stopped=probe.confidence_stopped,
                    raw_token_ids=probe_raw,
                    minimum_confidence=probe.minimum_confidence,
                )
                self.probe_selections += 1

        self.proposals += 1
        self.assistant_tokens += result.assistant_tokens
        self.target_drafts += len(result.token_ids)
        self.seconds += result.seconds
        if result.confidence_stopped:
            self.confidence_truncations += 1
            if not result.token_ids:
                self.confidence_skips += 1
        return result

    def record_verified(self, accepted_drafts: int, emitted_tokens: int) -> None:
        self.accepted_drafts += int(accepted_drafts)
        self.emitted_tokens += int(emitted_tokens)

    def status(self) -> dict[str, Any]:
        return {
            "proposals": self.proposals,
            "failures": self.failures,
            "assistant_tokens": self.assistant_tokens,
            "target_drafts": self.target_drafts,
            "accepted_drafts": self.accepted_drafts,
            "emitted_tokens": self.emitted_tokens,
            "seconds": self.seconds,
            "max_assistant_tokens": self.max_assistant_tokens,
            "confidence_threshold": self.confidence_threshold,
            "confidence_truncations": self.confidence_truncations,
            "confidence_skips": self.confidence_skips,
            "probe_selections": self.probe_selections,
            "wide_selections": self.wide_selections,
            "consensus_selections": self.consensus_selections,
            "device": str(self.device),
        }


def mode() -> str:
    value = os.environ.get("K3_UAG_DRAFT", "auto").strip().lower()
    if value in FALSE_MODES:
        return "off"
    if value in TRUE_MODES:
        return "on"
    if value == "auto":
        return "auto"
    raise ValueError(
        "K3_UAG_DRAFT must be on/1/true/auto or off/0/false"
    )


def requested() -> bool:
    return mode() != "off"


def _default_model_path(root: str) -> str:
    """Prefer the stronger drafter while preserving existing 0.6B installs."""
    candidates = (
        os.path.join(root, "k3-draft-qwen3-1.7b-base"),
        os.path.join(root, "k3-draft-qwen3-0.6b-base"),
    )
    required = ("config.json", "model.safetensors", "tokenizer.json")
    for candidate in candidates:
        if all(os.path.isfile(os.path.join(candidate, name))
               for name in required):
            return candidate
    return candidates[0]


def _is_hybrid_wide_config(config) -> bool:
    """Identify the pinned 1.7B architecture without relying on its folder."""
    return (
        getattr(config, "model_type", None) == "qwen3"
        and int(getattr(config, "hidden_size", 0)) == 2_048
        and int(getattr(config, "intermediate_size", 0)) == 6_144
        and int(getattr(config, "num_hidden_layers", 0)) == 28
    )


def load_local_drafter(
    root,
    target_tokenizer,
    device,
    *,
    model_dir=None,
    max_assistant_tokens=None,
    _allow_hybrid=True,
):
    """Load a built-in Transformers architecture from local safetensors only."""
    selected_mode = mode()
    if selected_mode == "off":
        return None
    # K3_SPEC is the master parity control for every speculative path.
    if os.environ.get("K3_SPEC", "1") != "1":
        return None
    dev = torch.device(device)
    # Automatic selection is limited to the physically measured accelerator
    # path. CUDA/CPU remain available with explicit ``on`` until their
    # end-to-end crossover has equally strong evidence.
    if selected_mode == "auto" and dev.type != "mps":
        return None

    path = os.path.abspath(
        model_dir
        or os.environ.get(
            "K3_UAG_MODEL",
            _default_model_path(root),
        )
    )
    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not os.path.isfile(
        os.path.join(path, name)
    )]
    if missing:
        error = UniversalDraftError(
            f"local assistant is incomplete at {path}: missing "
            + ", ".join(missing)
        )
        if selected_mode == "on":
            raise error
        print(
            "[uag] optional assistant is not installed; using target-only "
            "drafting (run tools/setup_draft.py to enable it)",
            flush=True,
        )
        return None

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    try:
        config = AutoConfig.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
        if (
            getattr(config, "model_type", None) != "qwen3"
            or getattr(config, "auto_map", None)
        ):
            raise UniversalDraftError(
                "assistant must be a built-in qwen3 config without remote code"
            )
        assistant_tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
        dtype = (
            torch.float16
            if dev.type in {"mps", "cuda"}
            else torch.float32
        )
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=dtype,
        ).to(dev).eval()
    except Exception as exc:
        if selected_mode == "on":
            raise
        # A failed accelerator move can leave allocator-held blocks even after
        # Python drops the partial model. Release those before target-only
        # generation continues on a memory-constrained host.
        try:
            if dev.type == "mps":
                torch.mps.empty_cache()
            elif dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(
            f"[uag] local assistant failed its load gates "
            f"({type(exc).__name__}: {exc}); using target-only drafting",
            flush=True,
        )
        return None
    load_seconds = time.perf_counter() - started
    resident_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))
    )
    drafter = UniversalTextDrafter(
        model,
        assistant_tokenizer,
        target_tokenizer,
        dev,
        max_assistant_tokens=int(
            max_assistant_tokens
            or os.environ.get("K3_UAG_ASSISTANT_TOKENS", "20")
        ),
        confidence_threshold=float(
            os.environ.get("K3_UAG_CONFIDENCE", "0.3")
        ),
    )
    print(
        f"[uag] local qwen3 assistant ready: {resident_bytes/1e9:.2f} GB "
        f"{str(dtype).removeprefix('torch.')} on {dev}, "
        f"{load_seconds:.2f}s one-time load",
        flush=True,
    )
    probe_spec = os.environ.get(
        "K3_UAG_PROBE_MODEL", "auto"
    ).strip()
    if (
        _allow_hybrid
        and _is_hybrid_wide_config(config)
        and probe_spec.lower() not in FALSE_MODES
    ):
        probe_path = os.path.abspath(
            os.path.join(root, "k3-draft-qwen3-0.6b-base")
            if probe_spec.lower() == "auto" else probe_spec
        )
        if probe_path != path and all(
            os.path.isfile(os.path.join(probe_path, name))
            for name in required
        ):
            probe = load_local_drafter(
                root,
                target_tokenizer,
                device,
                model_dir=probe_path,
                max_assistant_tokens=max_assistant_tokens,
                _allow_hybrid=False,
            )
            if probe is not None:
                print(
                    "[uag] hybrid drafting ready: 0.6B admission + "
                    "confidence-selected 0.6B/1.7B wide proposals",
                    flush=True,
                )
                return HybridTextDrafter(probe, drafter)
    return drafter


__all__ = [
    "Proposal",
    "HybridTextDrafter",
    "UniversalDraftError",
    "UniversalTextDrafter",
    "load_local_drafter",
    "mode",
    "requested",
]
