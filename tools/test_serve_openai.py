#!/usr/bin/env python3
"""Focused generation-boundary tests for the OpenAI-compatible server."""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import threading
import time
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

    def test_queue_defaults_to_env_and_cli_overrides_environment(self):
        with mock.patch.dict(
            os.environ, {"K3_SERVER_QUEUE": "3"}, clear=False
        ):
            parser = server._build_parser()
            self.assertEqual(parser.parse_args([]).queue, 3)
            self.assertEqual(
                parser.parse_args(["--queue", "5"]).queue, 5)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                server._build_parser().parse_args([]).queue, 0)

    def test_negative_queue_is_rejected(self):
        parser = server._build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--queue", "-1"])

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

        def generate(ids, max_new, on_delta=None, check_abort=None):
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


class ServerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self._leave_lock_free()
        server._active = 0

    def tearDown(self):
        self._leave_lock_free()
        server._active = 0

    @staticmethod
    def _leave_lock_free():
        # Unlike the acquire-if-free probe, this also releases a lock a
        # failed test left held (threading.Lock has no owner).
        server._lock.acquire(blocking=False)
        server._lock.release()

    def _completion_handler(self):
        handler = _handler(
            "/v1/completions", {"prompt": "hi", "max_tokens": 1})
        handler._json = mock.Mock()
        handler._err = mock.Mock()
        return handler

    def test_busy_server_rejects_instantly_with_retry_after(self):
        server._active = 1        # a generation is running
        server._lock.acquire()
        handler = self._completion_handler()
        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "QUEUE_SIZE", 0),
        ):
            started = time.monotonic()
            handler.do_POST()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.0)
        (code, _message), kwargs = handler._err.call_args
        self.assertEqual(code, 429)
        self.assertEqual(kwargs["err_type"], "rate_limit_error")
        self.assertGreaterEqual(kwargs["retry_after"], 30)
        handler._json.assert_not_called()
        self.assertEqual(server._active, 1)   # the 429 path never joined

    def test_queue_lets_one_request_wait_and_rejects_the_next(self):
        server._active = 1        # a generation is running
        server._lock.acquire()    # ...and holds the generation slot
        waiter = self._completion_handler()
        rejected = self._completion_handler()
        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "QUEUE_SIZE", 1),
            mock.patch.object(
                server, "_gen", return_value=([7], "answer", "stop")
            ),
            mock.patch.object(
                server, "_memo", server.DeterministicResponseMemo(0)
            ),
        ):
            thread = threading.Thread(target=waiter.do_POST, daemon=True)
            thread.start()
            for _ in range(500):
                if server._active == 2:
                    break
                time.sleep(0.01)
            self.assertEqual(server._active, 2)   # waiter occupies the queue
            rejected.do_POST()
            server._lock.release()                # the running generation ends
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        (code, _message), kwargs = rejected._err.call_args
        self.assertEqual(code, 429)
        self.assertEqual(kwargs["err_type"], "rate_limit_error")
        waiter._err.assert_not_called()
        (code, obj), _ = waiter._json.call_args
        self.assertEqual(code, 200)
        self.assertEqual(obj["choices"][0]["text"], "answer")
        self.assertEqual(server._active, 1)   # only the simulated runner left

    def test_active_count_and_lock_recover_after_generation_error(self):
        handler = self._completion_handler()
        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "QUEUE_SIZE", 0),
            mock.patch.object(
                server, "_gen", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(
                server, "_memo", server.DeterministicResponseMemo(0)
            ),
        ):
            handler.do_POST()
        (code, _message), kwargs = handler._err.call_args
        self.assertEqual(code, 500)
        self.assertEqual(kwargs["err_type"], "server_error")
        self.assertEqual(server._active, 0)
        self.assertTrue(server._lock.acquire(blocking=False))
        server._lock.release()


class ServerErrorResponseTests(unittest.TestCase):
    def setUp(self):
        if server._lock.acquire(blocking=False):
            server._lock.release()
        server._active = 0

    tearDown = setUp

    def test_non_integer_max_tokens_gets_400(self):
        for bad in ("abc", 1.5, -1, True):
            handler = _handler(
                "/v1/completions", {"prompt": "x", "max_tokens": bad})
            handler._err = mock.Mock()
            with mock.patch.object(server, "_tok", _Tokenizer()):
                handler.do_POST()
            (code, message), _ = handler._err.call_args
            self.assertEqual(code, 400, msg=repr(bad))
            self.assertIn("max_tokens", message)

    def test_bad_content_length_gets_400_not_a_dropped_connection(self):
        handler = _handler("/v1/completions", {"prompt": "x"})
        handler.headers = {"Content-Length": "not-a-number"}
        handler._err = mock.Mock()
        handler.do_POST()
        (code, _message), _ = handler._err.call_args
        self.assertEqual(code, 400)

    def test_encode_failure_returns_json_500(self):
        controller = mock.Mock()
        controller.encode_chat.side_effect = RuntimeError("template broke")
        handler = _handler(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        handler._err = mock.Mock()
        with (
            mock.patch.object(server, "_chat_tokenizer", controller),
            mock.patch.object(server, "_tok", _Tokenizer()),
        ):
            handler.do_POST()
        (code, message), kwargs = handler._err.call_args
        self.assertEqual(code, 500)
        self.assertEqual(kwargs["err_type"], "server_error")
        self.assertIn("template broke", message)
        self.assertEqual(server._active, 0)

    def test_stream_failure_emits_sse_error_event_not_second_headers(self):
        handler = _handler(
            "/v1/completions",
            {"prompt": "x", "stream": True, "max_tokens": 1},
        )
        with (
            mock.patch.object(server, "_tok", _Tokenizer()),
            mock.patch.object(server, "QUEUE_SIZE", 0),
            mock.patch.object(
                server, "_gen", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(
                server, "_memo", server.DeterministicResponseMemo(0)
            ),
        ):
            handler.do_POST()
        handler.send_response.assert_called_once_with(200)
        payload = handler.wfile.getvalue()
        self.assertIn(b'"type": "server_error"', payload)
        self.assertIn(b'"boom"', payload)
        self.assertTrue(payload.endswith(b"data: [DONE]\n\n"))
        self.assertEqual(server._active, 0)
        self.assertTrue(server._lock.acquire(blocking=False))
        server._lock.release()


class ServerDisconnectTests(unittest.TestCase):
    def setUp(self):
        ServerAdmissionTests._leave_lock_free()
        server._active = 0

    tearDown = setUp

    def test_client_disconnected_peeks_without_consuming(self):
        handler = _handler("/v1/completions", {"prompt": "x"})
        self.assertFalse(handler._client_disconnected())  # no socket at all
        local, remote = socket.socketpair()
        try:
            handler.connection = local
            self.assertFalse(handler._client_disconnected())  # open, idle
            remote.sendall(b"p")           # pipelined bytes are not an EOF...
            time.sleep(0.05)
            self.assertFalse(handler._client_disconnected())
            self.assertEqual(local.recv(1), b"p")   # ...and are not consumed
            remote.close()
            time.sleep(0.05)
            self.assertTrue(handler._client_disconnected())
        finally:
            local.close()

    def test_queued_client_disconnect_frees_its_slot(self):
        server._active = 1        # a generation is running
        server._lock.acquire()    # ...and never finishes during this test
        local, remote = socket.socketpair()
        handler = _handler(
            "/v1/completions", {"prompt": "hi", "max_tokens": 1})
        handler.connection = local
        handler._json = mock.Mock()
        handler._err = mock.Mock()
        try:
            with (
                mock.patch.object(server, "_tok", _Tokenizer()),
                mock.patch.object(server, "QUEUE_SIZE", 1),
                mock.patch.object(server, "_gen") as gen,
            ):
                thread = threading.Thread(
                    target=handler.do_POST, daemon=True)
                thread.start()
                for _ in range(500):
                    if server._active == 2:
                        break
                    time.sleep(0.01)
                self.assertEqual(server._active, 2)   # waiter is queued
                remote.close()            # the queued client goes away
                thread.join(timeout=5)    # slot frees with _lock still held
                self.assertFalse(thread.is_alive())
                gen.assert_not_called()
            self.assertEqual(server._active, 1)
            handler._json.assert_not_called()
            handler._err.assert_not_called()
        finally:
            local.close()
            server._lock.release()

    def test_abort_check_hook_installs_and_clears(self):
        def hook():
            pass

        self.assertIsNone(server.kr._ABORT_CHECK)
        with server.kr.abort_check(hook):
            self.assertIs(server.kr._ABORT_CHECK, hook)
        self.assertIsNone(server.kr._ABORT_CHECK)
        with self.assertRaises(RuntimeError):
            with server.kr.abort_check(hook):
                raise RuntimeError("abandoned pass")
        self.assertIsNone(server.kr._ABORT_CHECK)

    def test_generation_aborts_when_client_vanishes(self):
        local, remote = socket.socketpair()
        handler = _handler(
            "/v1/completions", {"prompt": "zz", "max_tokens": 5})
        handler.connection = local
        handler._json = mock.Mock()
        handler._err = mock.Mock()
        emitted = []

        def generate(layers, cache, embed, ids, max_new,
                     on_token=None, universal_drafter=None):
            # Do what forward_pass does before each decoder layer. The hook
            # covers prefill this way: no token needs to be emitted first.
            hook = server.kr._ABORT_CHECK
            self.assertIsNotNone(hook)
            self.assertIsNone(on_token)     # non-streaming stays callback-free
            hook()                          # client still connected
            emitted.append(7)
            remote.close()                  # client vanishes mid-prefill
            time.sleep(0.05)
            hook()                          # must raise BrokenPipeError
            emitted.append(8)
            return [7, 8]

        try:
            with (
                mock.patch.object(server, "_tok", _Tokenizer()),
                mock.patch.object(server, "QUEUE_SIZE", 0),
                mock.patch.object(
                    server.kr.ml, "KimiDynamicCache", return_value=object()
                ),
                mock.patch.object(
                    server.kr, "generate", side_effect=generate
                ),
                mock.patch.object(
                    server, "_memo", server.DeterministicResponseMemo(0)
                ),
            ):
                handler.do_POST()
        finally:
            local.close()
        self.assertEqual(emitted, [7])      # aborted before the second token
        handler._json.assert_not_called()   # nothing sent to the dead client
        handler._err.assert_not_called()
        self.assertEqual(server._active, 0)
        self.assertTrue(server._lock.acquire(blocking=False))
        server._lock.release()


if __name__ == "__main__":
    unittest.main()
