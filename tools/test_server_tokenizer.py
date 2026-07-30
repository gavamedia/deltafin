#!/usr/bin/env python3
"""Correctness and lifecycle tests for server-only chat tokenization."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server_tokenizer as st


@dataclass(frozen=True)
class _Segment:
    text: str
    allow_special: bool = False


class _FakeModel:
    n_vocab = 1024

    @staticmethod
    def decode_single_token_bytes(token):
        return int(token).to_bytes(2, "little")


class _FakeTokenizer:
    def __init__(self):
        self.special_tokens = {
            f"<S{i}>": 512 + i for i in range(256)
        }
        self.model = _FakeModel()

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(text, maximum):
        current = 0
        is_space = text[0].isspace() if text else False
        start = 0
        for index, char in enumerate(text):
            now_space = char.isspace()
            if is_space ^ now_space:
                current = 1
                is_space = now_space
            else:
                current += 1
                if current > maximum:
                    yield text[start:index]
                    start = index
                    current = 1
        yield text[start:]

    def _encode_text_piece(self, text, allow_special_tokens=True):
        if allow_special_tokens and text in self.special_tokens:
            return [self.special_tokens[text]]
        return [ord(char) % 251 for char in text]

    def _encode_chat_segments(self, segments):
        output = []
        for segment in segments:
            output.extend(
                self._encode_text_piece(
                    segment.text,
                    allow_special_tokens=segment.allow_special,
                )
            )
        return output

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize=False,
        add_generation_prompt=True,
        **_kwargs,
    ):
        segments = list(conversation)
        if add_generation_prompt:
            segments.append(_Segment("<S0>", True))
        if not tokenize:
            return "".join(segment.text for segment in segments)
        return self._encode_chat_segments(segments)


class _FakeBackend:
    n_vocab = 1024

    def __init__(self, tokenizer, *, fail=None, delay=0.0):
        self.tokenizer = tokenizer
        self.fail = fail
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._activity_lock = threading.Lock()

    def encode_batch(
        self,
        texts,
        num_threads=1,
        *,
        allowed_special=(),
        disallowed_special="all",
    ):
        texts = list(texts)
        self.calls.append(
            (texts, allowed_special, disallowed_special, num_threads)
        )
        with self._activity_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail is not None:
                raise self.fail
            structural = allowed_special == "all"
            if not structural and any(
                special in text
                for text in texts
                for special in self.tokenizer.special_tokens
            ):
                raise NotImplementedError("ordinary special")
            return [
                self.tokenizer._encode_text_piece(
                    text, allow_special_tokens=structural
                )
                for text in texts
            ]
        finally:
            with self._activity_lock:
                self.active -= 1

    @staticmethod
    def decode_single_token_bytes(token):
        return int(token).to_bytes(2, "little")


def _controller(
    *,
    mode="auto",
    backend=None,
    factory=None,
    init_wait_seconds=1.0,
    log=None,
):
    tokenizer = _FakeTokenizer()
    if backend is None:
        backend = _FakeBackend(tokenizer)
    if factory is None:
        factory = lambda _tokenizer: backend
    controller = st.ServerChatTokenizer(
        tokenizer,
        ".",
        mode=mode,
        threads=1,
        init_wait_seconds=init_wait_seconds,
        log=log,
        backend_factory=factory,
    )
    # Lifecycle tests exercise publication and dispatch independently of the
    # expensive real-vocabulary qualification covered by the integration test.
    controller._validate_backend = lambda _backend: None
    return controller, tokenizer, backend


def _wait_state(controller, expected, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = controller.status()["state"]
        if state == expected:
            return
        time.sleep(0.005)
    raise AssertionError(
        f"state remained {controller.status()['state']!r}; "
        f"expected {expected!r}"
    )


class ServerTokenizerUnitTests(unittest.TestCase):
    def test_modes_are_strict(self):
        for mode in ("auto", "on", "off", " AUTO "):
            self.assertIn(st.parse_mode(mode), st.MODES)
        with self.assertRaisesRegex(ValueError, "auto, on, or off"):
            st.parse_mode("sometimes")

    def test_off_never_constructs_backend(self):
        calls = []
        controller, _, _ = _controller(
            mode="off",
            factory=lambda tokenizer: calls.append(tokenizer),
        )
        result = controller.encode_chat([_Segment("abc")])
        self.assertEqual(result.backend, "tiktoken")
        self.assertEqual(calls, [])
        self.assertFalse(controller.after_response(1_000_000))

    def test_mixed_policies_are_batched_and_reassembled_without_merges(self):
        controller, tokenizer, backend = _controller(mode="on")
        self.assertTrue(controller.initialize(required=True))
        segments = [
            _Segment("ab"),
            _Segment("<S7>", True),
            _Segment("cd"),
        ]
        expected = tokenizer.apply_chat_template(
            segments, tokenize=True, add_generation_prompt=True)
        result = controller.encode_chat(segments)
        self.assertEqual(result.backend, "gigatoken")
        self.assertEqual(result.token_ids, expected)
        ordinary = [
            call for call in backend.calls
            if call[1] != "all"
        ][0]
        self.assertEqual(ordinary[0], ["ab", "cd"])
        self.assertNotIn("abcd", ordinary[0])

    def test_ordinary_literal_special_restarts_whole_call(self):
        controller, tokenizer, _ = _controller(mode="on")
        controller.initialize(required=True)
        segments = [_Segment("left<S3>right"), _Segment("<S0>", True)]
        expected = tokenizer.apply_chat_template(
            segments, tokenize=True, add_generation_prompt=True)
        result = controller.encode_chat(segments)
        self.assertEqual(result.backend, "tiktoken-special-fallback")
        self.assertEqual(result.token_ids, expected)
        self.assertEqual(
            controller.status()["expected_fallbacks"], 1)
        self.assertEqual(controller.status()["state"], "ready")

    def test_unexpected_native_error_is_sticky_and_exact(self):
        controller, tokenizer, backend = _controller(mode="on")
        controller.initialize(required=True)
        backend.fail = RuntimeError("poison")
        segments = [_Segment("ordinary")]
        expected = tokenizer.apply_chat_template(
            segments, tokenize=True, add_generation_prompt=True)
        first = controller.encode_chat(segments)
        calls = len(backend.calls)
        second = controller.encode_chat(segments)
        self.assertEqual(first.backend, "tiktoken-error-fallback")
        self.assertEqual(second.backend, "tiktoken")
        self.assertEqual(first.token_ids, expected)
        self.assertEqual(second.token_ids, expected)
        self.assertEqual(len(backend.calls), calls)
        self.assertEqual(controller.status()["state"], "failed")

    def test_next_request_waits_for_post_response_initialization(self):
        entered = threading.Event()
        release = threading.Event()
        tokenizer = _FakeTokenizer()
        backend = _FakeBackend(tokenizer)

        def factory(_tokenizer):
            entered.set()
            release.wait(2)
            return backend

        controller, _, _ = _controller(
            mode="auto", backend=backend, factory=factory)
        before = controller.encode_chat([_Segment("first")])
        self.assertEqual(before.backend, "tiktoken")
        qualified = []
        qualifier = threading.Thread(
            target=lambda: qualified.append(
                controller.after_response(0)
            )
        )
        qualifier.start()
        self.assertTrue(entered.wait(1))
        result = []
        request = threading.Thread(
            target=lambda: result.append(
                controller.encode_chat([_Segment("next")])
            )
        )
        request.start()
        time.sleep(0.02)
        self.assertTrue(request.is_alive())
        release.set()
        qualifier.join(1)
        request.join(1)
        self.assertFalse(qualifier.is_alive())
        self.assertFalse(request.is_alive())
        self.assertEqual(qualified, [True])
        _wait_state(controller, "ready")
        self.assertEqual(result[0].backend, "gigatoken")

    def test_initialization_wait_timeout_uses_exact_baseline(self):
        entered = threading.Event()
        release = threading.Event()
        tokenizer = _FakeTokenizer()
        backend = _FakeBackend(tokenizer)

        def factory(_tokenizer):
            entered.set()
            release.wait(2)
            return backend

        controller, expected_tokenizer, _ = _controller(
            mode="auto",
            backend=backend,
            factory=factory,
            init_wait_seconds=0.02,
        )
        qualifier = threading.Thread(
            target=lambda: controller.after_response(0)
        )
        qualifier.start()
        self.assertTrue(entered.wait(1))
        segments = [_Segment("next")]
        result = controller.encode_chat(segments)
        self.assertEqual(result.backend, "tiktoken")
        self.assertEqual(
            result.token_ids,
            expected_tokenizer.apply_chat_template(
                segments, tokenize=True, add_generation_prompt=True
            ),
        )
        release.set()
        qualifier.join(1)
        _wait_state(controller, "ready")

    def test_ready_backend_calls_are_serialized(self):
        controller, _, backend = _controller(mode="on")
        controller.initialize(required=True)
        backend.delay = 0.02
        errors = []

        def run(index):
            try:
                result = controller.encode_chat(
                    [_Segment(f"request {index}")])
                if result.backend != "gigatoken":
                    errors.append(result.backend)
            except Exception as exc:  # pragma: no cover - diagnostic
                errors.append(repr(exc))

        threads = [
            threading.Thread(target=run, args=(index,))
            for index in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(backend.max_active, 1)

    def test_queued_request_never_uses_backend_after_poison(self):
        entered = threading.Event()
        release = threading.Event()
        tokenizer = _FakeTokenizer()

        class PoisonBackend(_FakeBackend):
            def __init__(self):
                super().__init__(tokenizer)
                self.attempts = 0
                self.attempt_lock = threading.Lock()

            def encode_batch(self, texts, **kwargs):
                with self.attempt_lock:
                    self.attempts += 1
                    attempt = self.attempts
                if attempt == 1:
                    entered.set()
                    release.wait(2)
                    raise RuntimeError("poison")
                return super().encode_batch(texts, **kwargs)

        backend = PoisonBackend()
        controller, _, _ = _controller(
            mode="on", backend=backend)
        controller.initialize(required=True)
        results = []

        first = threading.Thread(
            target=lambda: results.append(
                controller.encode_chat([_Segment("first")])
            )
        )
        second = threading.Thread(
            target=lambda: results.append(
                controller.encode_chat([_Segment("second")])
            )
        )
        first.start()
        self.assertTrue(entered.wait(1))
        second.start()
        time.sleep(0.02)
        release.set()
        first.join(1)
        second.join(1)
        self.assertEqual(
            sorted(result.backend for result in results),
            ["tiktoken", "tiktoken-error-fallback"],
        )
        self.assertEqual(backend.attempts, 1)
        self.assertEqual(controller.status()["state"], "failed")

    def test_close_during_initialization_prevents_publication(self):
        entered = threading.Event()
        release = threading.Event()

        def factory(_tokenizer):
            entered.set()
            release.wait(2)
            return _FakeBackend(_tokenizer)

        controller, _, _ = _controller(
            mode="auto", factory=factory)
        qualifier = threading.Thread(
            target=lambda: controller.after_response(0)
        )
        qualifier.start()
        self.assertTrue(entered.wait(1))
        closed = threading.Event()
        closer = threading.Thread(
            target=lambda: (controller.close(), closed.set())
        )
        closer.start()
        time.sleep(0.02)
        self.assertFalse(closed.is_set())
        release.set()
        qualifier.join(1)
        closer.join(1)
        self.assertTrue(closed.is_set())
        _wait_state(controller, "closed")
        self.assertEqual(controller.status()["state"], "closed")

    def test_native_baseexception_disables_backend_and_falls_back(self):
        class NativePanic(BaseException):
            pass

        controller, tokenizer, backend = _controller(mode="on")
        controller.initialize(required=True)
        backend.fail = NativePanic("rust panic")
        segments = [_Segment("ordinary")]
        result = controller.encode_chat(segments)
        self.assertEqual(result.backend, "tiktoken-error-fallback")
        self.assertEqual(
            result.token_ids,
            tokenizer.apply_chat_template(
                segments, tokenize=True, add_generation_prompt=True
            ),
        )
        self.assertEqual(controller.status()["state"], "failed")

    def test_raising_logger_cannot_block_exact_native_fallback(self):
        controller, tokenizer, backend = _controller(
            mode="on",
            log=lambda _message: (_ for _ in ()).throw(
                BrokenPipeError("log closed")
            ),
        )
        self.assertTrue(controller.initialize(required=True))
        backend.fail = RuntimeError("poison")
        segments = [_Segment("ordinary")]
        result = controller.encode_chat(segments)
        self.assertEqual(result.backend, "tiktoken-error-fallback")
        self.assertEqual(
            result.token_ids,
            tokenizer.apply_chat_template(
                segments, tokenize=True, add_generation_prompt=True
            ),
        )

    def test_initialization_baseexception_wakes_next_request(self):
        class NativePanic(BaseException):
            pass

        controller, _, _ = _controller(
            mode="auto",
            factory=lambda _tokenizer: (_ for _ in ()).throw(
                NativePanic("rust init panic")
            ),
        )
        self.assertFalse(controller.after_response(0))
        _wait_state(controller, "failed")
        self.assertEqual(
            controller.encode_chat([_Segment("safe")]).backend,
            "tiktoken",
        )

    def test_explicit_on_reports_initialization_failure(self):
        controller, _, _ = _controller(
            mode="on",
            factory=lambda _tokenizer: (_ for _ in ()).throw(
                RuntimeError("bad backend")
            ),
        )
        with self.assertRaisesRegex(
            st.ServerTokenizerError, "explicit GigaToken startup failed"
        ):
            controller.initialize(required=True)
        self.assertEqual(controller.status()["state"], "failed")


@unittest.skipUnless(
    importlib.util.find_spec("gigatoken") is not None,
    "reviewed GigaToken wheel is not installed",
)
class RealK3TokenizerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.meta = Path(__file__).resolve().parents[1] / "k3-meta"
        if not cls.meta.is_dir():
            raise unittest.SkipTest("local K3 tokenizer metadata unavailable")
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.meta, trust_remote_code=True)
        cls.controller = st.ServerChatTokenizer(
            cls.tokenizer,
            cls.meta,
            mode="on",
            threads=1,
        )
        cls.controller.initialize(required=True)

    @classmethod
    def tearDownClass(cls):
        cls.controller.close()

    def _assert_chat_exact(self, messages):
        expected = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        actual = self.controller.encode_chat(messages)
        self.assertEqual(actual.token_ids, expected)
        return actual

    def test_multilingual_growing_chat_is_exact(self):
        messages = []
        for index in range(40):
            messages.extend([
                {
                    "role": "user",
                    "content": (
                        f"turn {index}: 世界 café re\u0301sume\u0301 "
                        "👨🏽‍💻 العربية\n"
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"answer {index}: 123 / path...",
                },
            ])
        result = self._assert_chat_exact(messages)
        self.assertEqual(result.backend, "gigatoken")

    def test_user_special_spelling_uses_exact_whole_call_fallback(self):
        result = self._assert_chat_exact([
            {
                "role": "user",
                "content": "literal <|open|> and [BOS] are ordinary text",
            }
        ])
        self.assertEqual(result.backend, "tiktoken-special-fallback")

    def test_400k_and_25k_boundaries_are_exact(self):
        content = (
            "x" * 25_001
            + " "
            + ("a " * 190_000)
            + "界" * 10_001
        )
        result = self._assert_chat_exact([
            {"role": "user", "content": content}
        ])
        self.assertEqual(result.backend, "gigatoken")


if __name__ == "__main__":
    unittest.main()
