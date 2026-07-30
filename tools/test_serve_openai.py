#!/usr/bin/env python3
"""Focused generation-boundary tests for the OpenAI-compatible server."""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_openai as server


class _Tokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return ",".join(str(token) for token in ids)


class _ChatTokenizer:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.encodes = []
        self.responses = []

    def encode_chat(self, messages, add_generation_prompt=True):
        self.encodes.append((messages, add_generation_prompt))
        return server.server_tokenizer.ChatEncoding(
            [1, 2, 3], 12_345, "tiktoken")

    def after_response(self, rendered_chars):
        # Qualification runs after the response flush while both admission
        # locks still exclude a next chat encode or target generation.
        generation_free = server._lock.acquire(blocking=False)
        if generation_free:
            server._lock.release()
        tokenizer_free = server._chat_tokenizer_gate.acquire(
            blocking=False
        )
        if tokenizer_free:
            server._chat_tokenizer_gate.release()
        self.responses.append(
            (rendered_chars, generation_free, tokenizer_free)
        )
        self.events.append("after_response")
        return True


def _handler(path, body):
    handler = object.__new__(server.Handler)
    payload = json.dumps(body).encode()
    handler.path = path
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)
    handler.wfile = io.BytesIO()
    handler.send_response = mock.Mock()
    handler.send_header = mock.Mock()
    handler.end_headers = mock.Mock()
    handler.log_request = mock.Mock()
    return handler


class ServerGenerationBoundaryTests(unittest.TestCase):
    def test_nonstreaming_request_installs_no_token_callback(self):
        observed = {}

        def generate(
            layers,
            cache,
            embed,
            ids,
            max_new,
            on_token=None,
            universal_drafter=None,
        ):
            observed["on_token"] = on_token
            observed["universal_drafter"] = universal_drafter
            return [7, 8, server.kr.EOS_ID]

        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "_layers", object()),
            mock.patch.object(server, "_embed", object()),
            mock.patch.object(
                server.kr.ml, "KimiDynamicCache", return_value=object()
            ),
            mock.patch.object(server.kr, "generate", side_effect=generate),
            mock.patch.object(
                server.kr, "IncrementalTokenDecoder"
            ) as decoder,
        ):
            out, text, finish = server._gen([1, 2], 3)
        self.assertIsNone(observed["on_token"])
        self.assertIsNone(observed["universal_drafter"])
        decoder.assert_not_called()
        self.assertEqual(out, [7, 8])
        self.assertEqual(text, "7,8")
        self.assertEqual(finish, "stop")

    def test_streaming_preserves_delta_order_tail_and_eos_filter(self):
        deltas = []

        class Decoder:
            def append(self, token):
                return {7: "", 8: "eight"}[token]

            def finish(self):
                return "tail"

        def generate(
            layers,
            cache,
            embed,
            ids,
            max_new,
            on_token=None,
            universal_drafter=None,
        ):
            self.assertIsNotNone(on_token)
            self.assertIsNone(universal_drafter)
            on_token(7)
            on_token(8)
            on_token(server.kr.EOS_ID)
            return [7, 8, server.kr.EOS_ID]

        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "_layers", object()),
            mock.patch.object(server, "_embed", object()),
            mock.patch.object(
                server.kr.ml, "KimiDynamicCache", return_value=object()
            ),
            mock.patch.object(server.kr, "generate", side_effect=generate),
            mock.patch.object(
                server.kr, "IncrementalTokenDecoder", return_value=Decoder()
            ),
        ):
            out, text, finish = server._gen(
                [1, 2], 3, on_delta=deltas.append
            )
        self.assertEqual(deltas, ["eight", "tail"])
        self.assertEqual(out, [7, 8])
        self.assertEqual(text, "7,8")
        self.assertEqual(finish, "stop")

    def test_length_finish_keeps_last_non_eos_token(self):
        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "_layers", object()),
            mock.patch.object(server, "_embed", object()),
            mock.patch.object(
                server.kr.ml, "KimiDynamicCache", return_value=object()
            ),
            mock.patch.object(
                server.kr, "generate", return_value=[7, 8]
            ),
        ):
            out, text, finish = server._gen([1, 2], 2)
        self.assertEqual(out, [7, 8])
        self.assertEqual(text, "7,8")
        self.assertEqual(finish, "length")


class ServerConfigurationTests(unittest.TestCase):
    def test_gigatoken_defaults_to_auto_and_cli_overrides_environment(self):
        with mock.patch.dict(
            os.environ, {"K3_SERVER_GIGATOKEN": "off"}, clear=False
        ):
            parser = server._build_parser()
            self.assertEqual(
                parser.parse_args([]).gigatoken, "off")
            self.assertEqual(
                parser.parse_args(
                    ["--gigatoken", "on"]
                ).gigatoken,
                "on",
            )
        with mock.patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(
                server._build_parser().parse_args([]).gigatoken,
                "auto",
            )

    def test_encode_chat_uses_controller_without_replacing_decode_tokenizer(self):
        controller = _ChatTokenizer()
        original = _Tokenizer()
        with (
            mock.patch.object(server, "_tok", original),
            mock.patch.object(server, "_chat_tokenizer", controller),
        ):
            ids, chars, backend = server._encode_chat(
                [{"role": "user", "content": "hello"}])
            self.assertIs(server._tok, original)
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(chars, 12_345)
        self.assertEqual(backend, "tiktoken")


class ServerHandlerTokenizerLifecycleTests(unittest.TestCase):
    def setUp(self):
        # A failed test must not leave the process-global admission lock held.
        if server._lock.acquire(blocking=False):
            server._lock.release()
        if server._chat_tokenizer_gate.acquire(blocking=False):
            server._chat_tokenizer_gate.release()

    def test_nonstream_chat_qualifies_after_response_under_admission(self):
        events = []
        controller = _ChatTokenizer(events)
        handler = _handler(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        )

        def response(code, obj):
            events.append(("json", code, obj["usage"]))

        handler._json = response
        handler._err = mock.Mock()
        with (
            mock.patch.object(server, "_chat_tokenizer", controller),
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(
                server, "_gen", return_value=([7], "answer", "stop")
            ),
            mock.patch.object(
                server, "_memo",
                server.DeterministicResponseMemo(0),
            ),
        ):
            handler.do_POST()

        self.assertEqual(
            controller.responses, [(12_345, False, False)])
        self.assertEqual(events[-1], "after_response")
        self.assertEqual(events[0][0], "json")
        handler._err.assert_not_called()

    def test_stream_chat_qualifies_only_after_done_is_flushed(self):
        events = []
        controller = _ChatTokenizer(events)

        class Writer(io.BytesIO):
            def flush(self):
                if self.getvalue().endswith(b"data: [DONE]\n\n"):
                    events.append("done_flushed")

        handler = _handler(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 1,
            },
        )
        handler.wfile = Writer()
        handler._err = mock.Mock()

        def generate(ids, max_new, on_delta=None):
            on_delta("answer")
            return [7], "answer", "stop"

        with (
            mock.patch.object(server, "_chat_tokenizer", controller),
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "_gen", side_effect=generate),
            mock.patch.object(
                server, "_memo",
                server.DeterministicResponseMemo(0),
            ),
        ):
            handler.do_POST()

        self.assertIn("done_flushed", events)
        self.assertEqual(events[-1], "after_response")
        self.assertEqual(
            controller.responses, [(12_345, False, False)])
        handler._err.assert_not_called()

    def test_failed_chat_and_raw_completion_never_arm(self):
        controller = _ChatTokenizer()
        failed = _handler(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        )
        failed._err = mock.Mock()
        failed._json = mock.Mock()
        with (
            mock.patch.object(server, "_chat_tokenizer", controller),
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(
                server, "_gen", side_effect=RuntimeError("generation failed")
            ),
            mock.patch.object(
                server, "_memo",
                server.DeterministicResponseMemo(0),
            ),
        ):
            failed.do_POST()
        self.assertEqual(controller.responses, [])
        failed._err.assert_called_once()

        raw = _handler(
            "/v1/completions",
            {"prompt": "hello", "max_tokens": 1},
        )
        raw._json = mock.Mock()
        raw._err = mock.Mock()
        with (
            mock.patch.object(server, "_chat_tokenizer", controller),
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(
                server, "_gen", return_value=([7], "answer", "stop")
            ),
            mock.patch.object(
                server, "_memo",
                server.DeterministicResponseMemo(0),
            ),
        ):
            raw.do_POST()
        self.assertEqual(controller.responses, [])
        self.assertEqual(len(controller.encodes), 1)


if __name__ == "__main__":
    unittest.main()
