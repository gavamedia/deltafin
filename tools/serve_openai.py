#!/usr/bin/env python3
"""Deltafin OpenAI-compatible API server.

Exposes /v1/chat/completions, /v1/completions, and /v1/models over HTTP so any
OpenAI-SDK client, chat UI, or coding agent can talk to Kimi K3 running locally:

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1  OPENAI_API_KEY=none  <your tool>

Design notes, honestly stated:
  * Decoding is greedy and reproducible; temperature/top_p are accepted and
    ignored. Omitted max_tokens means "until the model finishes" for chat
    (K3 emits an end token) and 256 for raw completions (which never end on
    their own). Explicit max_tokens is honored as-is; operators can set a
    ceiling with K3_SERVER_MAX_TOKENS.
  * One generation at a time. Concurrency would be meaningless at this speed,
    so by default a concurrent request is rejected immediately with 429 plus a
    Retry-After estimate. K3_SERVER_QUEUE=N (or --queue N) instead lets up to
    N requests wait for the generation slot; wakeup is arrival-order in
    practice, though CPython's lock does not guarantee it. A client that
    disconnects is detected by peeking its socket: every 0.5 s while queued
    (freeing the slot) and before every decoder layer while generating —
    prefill included — abandoning the generation. Worst case is one layer's
    remaining work (e.g. a single cold expert fetch), not a whole pass.
  * Internal failures always produce a JSON error response: 400 for malformed
    requests, 500 (type "server_error") for everything else. A failure after
    streaming has begun is reported as a final SSE error event instead.
  * Chat mode renders K3's template, which includes a thinking section; the
    response splits it into `reasoning_content` and `content` (DeepSeek-style).
  * Each request uses a fresh KV/state cache; nothing is shared across calls.
  * A chat request's prompt is ~60+ tokens; on a COLD expert cache the prefill
    can take hours because most experts get fetched. Warm up with short
    completions first, or let the cache grow across sessions.

Usage:  python tools/serve_openai.py [--port 8000]
(device and spine format are auto-detected; see K3_DEV / K3_SPINE to override)

On supported hosts, the optional native chat-tokenizer accelerator comes from
the pinned wheel in requirements.txt; ``--gigatoken off`` never imports it.
"""
import argparse
import json
import math
import os
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr  # noqa: E402
import server_tokenizer  # noqa: E402
import universal_draft  # noqa: E402
from response_memo import DeterministicResponseMemo  # noqa: E402

MODEL_ID = "deltafin-kimi-k3"
MAX_TOKENS_CAP = int(os.environ.get("K3_SERVER_MAX_TOKENS", "0"))   # 0 = no cap
QUEUE_SIZE = int(os.environ.get("K3_SERVER_QUEUE", "0"))  # waiting slots; 0 = reject when busy
RESPONSE_MEMO_ENTRIES = int(
    os.environ.get("K3_RESPONSE_MEMO_ENTRIES", "32"))
RESPONSE_MARKER = "<|open|>response<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"

_lock = threading.Lock()
_active_lock = threading.Lock()
_active = 0                 # requests holding or waiting for _lock
_last_gen_seconds = None    # wall time of the last completed generation
_chat_tokenizer_gate = threading.Lock()
_tok = None
_layers = None
_embed = None
_universal_drafter = None
_chat_tokenizer = None
_memo = DeterministicResponseMemo(RESPONSE_MEMO_ENTRIES)


def _boot(gigatoken_mode="auto"):
    global _tok, _layers, _embed, _universal_drafter, _chat_tokenizer
    from transformers import AutoTokenizer
    print("[serve] loading tokenizer + layer skeletons...", flush=True)
    _tok = AutoTokenizer.from_pretrained(
        os.path.join(kr.ROOT, "k3-meta"), trust_remote_code=True)
    _chat_tokenizer = server_tokenizer.ServerChatTokenizer(
        _tok,
        os.path.join(kr.ROOT, "k3-meta"),
        mode=gigatoken_mode,
        log=lambda message: print(message, flush=True),
    )
    if gigatoken_mode == "on":
        _chat_tokenizer.initialize(required=True)
    kr.check_expert_pool()
    _universal_drafter = universal_draft.load_local_drafter(
        kr.ROOT, _tok, kr.DEV
    )
    _layers = kr.build_layers()
    _embed = kr.LazyEmbed()
    print("[serve] ready", flush=True)


def _encode_chat(messages):
    with _chat_tokenizer_gate:
        if _chat_tokenizer is None:
            ids = _tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            return list(ids), 0, "tiktoken"
        result = _chat_tokenizer.encode_chat(
            messages, add_generation_prompt=True)
        return result.token_ids, result.rendered_chars, result.backend


def _split_reasoning(text):
    """K3 chat output: <think...><|close|>think<|sep|><|open|>response<|sep|><answer>."""
    if RESPONSE_MARKER in text:
        pre, ans = text.rsplit(RESPONSE_MARKER, 1)
        reasoning = pre.replace(THINK_CLOSE, "").replace("<|open|>think<|sep|>", "").strip()
        return reasoning or None, ans
    return None, text


def _retry_after_estimate():
    """Seconds to suggest in Retry-After; the caller holds _active_lock.

    Based on the last completed generation times the number of requests
    already ahead — an honest guess, not a promise.
    """
    per_gen = _last_gen_seconds if _last_gen_seconds else 60.0
    return max(30, math.ceil(per_gen * max(1, _active)))


def _gen(ids, max_new, on_delta=None, check_abort=None):
    """Run one generation under the global lock; stream decoded-text deltas.

    check_abort, when given, runs before every decoder layer of every pass —
    prefill included — and may raise to abandon the generation (used to stop
    working for a vanished client).
    """
    global _last_gen_seconds
    started = time.monotonic()
    cache = kr.ml.KimiDynamicCache(kr.config)
    decoder = kr.IncrementalTokenDecoder(_tok) if on_delta else None

    def on_token(t):
        if t == kr.EOS_ID:
            return
        delta = decoder.append(t)
        if delta:
            on_delta(delta)

    with kr.abort_check(check_abort):
        out = kr.generate(
            _layers,
            cache,
            _embed,
            ids,
            max_new,
            on_token=on_token if on_delta else None,
            universal_drafter=_universal_drafter,
        )
    if decoder is not None:
        tail = decoder.finish()
        if tail:
            on_delta(tail)
    # generate() guarantees that EOS, when present, is the final emitted ID.
    if out and out[-1] == kr.EOS_ID:
        out.pop()
        finish = "stop"
    else:
        finish = "length"
    _last_gen_seconds = time.monotonic() - started
    return out, _tok.decode(out), finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[serve] {self.address_string()} {fmt % a}", flush=True)

    def _json(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self._headers_sent = True
        self.wfile.write(body)
        self.wfile.flush()

    def _err(self, code, msg, err_type="invalid_request_error", retry_after=None):
        headers = None
        if retry_after is not None:
            headers = {"Retry-After": str(retry_after)}
        self._json(code, {"error": {"message": msg, "type": err_type}}, headers)

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "deltafin"}]})
        else:
            self._err(404, f"no route {self.path}")

    def do_POST(self):
        self._headers_sent = False
        self._streaming = False
        try:
            self._handle_post()
        except BrokenPipeError:
            print("[serve] client disconnected", flush=True)
        except Exception as e:
            print(f"[serve] error: {e!r}", flush=True)
            self._report_failure(str(e))

    def _report_failure(self, message):
        try:
            if self._streaming:
                # The 200 status and SSE headers are already on the wire; a
                # status line cannot be un-sent, so report through the stream.
                self.wfile.write(b"data: " + json.dumps(
                    {"error": {"message": message, "type": "server_error"}}
                ).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            elif not self._headers_sent:
                self._err(500, message, err_type="server_error")
        except Exception:
            pass

    def _client_disconnected(self):
        """True once the client socket reports EOF or reset (peek, no consume).

        Readable-with-data means a pipelined request, not a disconnect.
        Fabricated handlers without a socket (unit tests) count as connected.
        """
        conn = getattr(self, "connection", None)
        if conn is None:
            return False
        try:
            readable, _, _ = select.select([conn], [], [], 0)
            if not readable:
                return False
            return conn.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _abort_if_disconnected(self):
        if self._client_disconnected():
            raise BrokenPipeError("client disconnected")

    def _acquire_generation_slot(self):
        """Admit this request and wait for the generation lock.

        Returns True with the lock held, or False after replying 429 (no free
        slot) or silently dropping a client that disconnected while queued.
        """
        global _active
        with _active_lock:
            if _active >= 1 + QUEUE_SIZE:
                if QUEUE_SIZE:
                    msg = (f"a generation is running and the queue is full "
                           f"({QUEUE_SIZE} waiting); retry later")
                else:
                    msg = ("a generation is already running (Deltafin serves "
                           "one at a time; K3_SERVER_QUEUE allows waiting)")
                self._err(429, msg, err_type="rate_limit_error",
                          retry_after=_retry_after_estimate())
                return False
            _active += 1
        while not _lock.acquire(timeout=0.5):
            if self._client_disconnected():
                with _active_lock:
                    _active -= 1
                print("[serve] queued client disconnected; slot freed",
                      flush=True)
                return False
        if self._client_disconnected():
            self._release_generation_slot()
            print("[serve] client disconnected before its turn", flush=True)
            return False
        return True

    def _release_generation_slot(self):
        global _active
        _lock.release()
        with _active_lock:
            _active -= 1

    def _handle_post(self):
        rendered_chars = 0
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:  # bad Content-Length or bad JSON (JSONDecodeError)
            return self._err(400, "invalid JSON body")
        chat = self.path in ("/v1/chat/completions", "/chat/completions")
        comp = self.path in ("/v1/completions", "/completions")
        if not (chat or comp):
            return self._err(404, f"no route {self.path}")

        if chat:
            messages = body.get("messages")
            if not messages:
                return self._err(400, "messages required")
            ids, rendered_chars, _tokenizer_backend = _encode_chat(messages)
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                return self._err(400, "prompt (string) required")
            ids = _tok.encode(prompt)
        # OpenAI semantics: omitted max_tokens means the model decides. Chat ends
        # naturally at EOS; raw completions have no terminator, so only THEY get
        # a default cap (256) — an explicit max_tokens is always honored.
        req_max = body.get("max_tokens")
        if req_max is not None and (
                not isinstance(req_max, int) or isinstance(req_max, bool)
                or req_max < 0):
            return self._err(400, "max_tokens must be a non-negative integer")
        if req_max:
            max_new = req_max
        else:
            max_new = 1_000_000 if chat else 256
        if MAX_TOKENS_CAP:
            max_new = min(max_new, MAX_TOKENS_CAP)
        stream = bool(body.get("stream"))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:20]
        created = int(time.time())

        if not self._acquire_generation_slot():
            return
        try:
            mode = "chat" if chat else "completion"
            cached = _memo.get(mode, ids, max_new)
            if cached is not None:
                print(f"[serve] deterministic response memo hit "
                      f"({len(cached.token_ids)} tokens)", flush=True)
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self._headers_sent = True
                self._streaming = True

                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                def on_delta(delta):
                    if chat:
                        sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                             "model": MODEL_ID,
                             "choices": [{"index": 0, "delta": {"content": delta},
                                          "finish_reason": None}]})
                    else:
                        sse({"id": rid, "object": "text_completion", "created": created,
                             "model": MODEL_ID,
                             "choices": [{"index": 0, "text": delta, "finish_reason": None}]})

                if cached is None:
                    out, text, finish = _gen(
                        ids, max_new, on_delta=on_delta,
                        check_abort=self._abort_if_disconnected)
                    _memo.put(mode, ids, max_new, out, text, finish)
                else:
                    out = list(cached.token_ids)
                    text = cached.text
                    finish = cached.finish_reason
                    if text:
                        on_delta(text)
                key = "delta" if chat else "text"

                def finish_stream():
                    sse({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                         "created": created, "model": MODEL_ID,
                         "choices": [{"index": 0, key: {} if chat else "", "finish_reason": finish}]})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

                if chat:
                    # Hold the chat-tokenizer gate across the final flush and
                    # post-response qualification.  A client cannot squeeze
                    # its next chat encode into the tiny interval between them.
                    with _chat_tokenizer_gate:
                        finish_stream()
                        if _chat_tokenizer is not None:
                            _chat_tokenizer.after_response(rendered_chars)
                else:
                    finish_stream()
                return

            if cached is None:
                out, text, finish = _gen(
                    ids, max_new,
                    check_abort=self._abort_if_disconnected)
                _memo.put(mode, ids, max_new, out, text, finish)
            else:
                out = list(cached.token_ids)
                text = cached.text
                finish = cached.finish_reason
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(out),
                     "total_tokens": len(ids) + len(out)}
            if chat:
                reasoning, content = _split_reasoning(text)
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                with _chat_tokenizer_gate:
                    self._json(200, {"id": rid, "object": "chat.completion", "created": created,
                                     "model": MODEL_ID, "usage": usage,
                                     "choices": [{"index": 0, "message": msg,
                                                  "finish_reason": finish}]})
                    if _chat_tokenizer is not None:
                        _chat_tokenizer.after_response(rendered_chars)
            else:
                self._json(200, {"id": rid, "object": "text_completion", "created": created,
                                 "model": MODEL_ID, "usage": usage,
                                 "choices": [{"index": 0, "text": text,
                                              "finish_reason": finish}]})
        finally:
            self._release_generation_slot()


def _queue_size(value):
    queue = int(value)
    if queue < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return queue


def _build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--gigatoken",
        choices=sorted(server_tokenizer.MODES),
        default=os.environ.get("K3_SERVER_GIGATOKEN", "auto"),
        help=(
            "server chat-tokenizer acceleration: auto warms after the first "
            "completed chat response (default), on initializes at startup, and "
            "off never imports GigaToken"
        ),
    )
    ap.add_argument(
        "--queue",
        type=_queue_size,
        default=os.environ.get("K3_SERVER_QUEUE", "0"),
        help=(
            "requests allowed to wait for the single generation slot; 0 "
            "(default) rejects a concurrent request immediately with 429 + "
            "Retry-After; the K3_SERVER_QUEUE environment variable sets the "
            "default"
        ),
    )
    return ap


def main():
    global QUEUE_SIZE
    ap = _build_parser()
    args = ap.parse_args()
    try:
        gigatoken_mode = server_tokenizer.parse_mode(args.gigatoken)
    except ValueError as exc:
        ap.error(str(exc))
    QUEUE_SIZE = args.queue
    _boot(gigatoken_mode)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] Deltafin OpenAI-compatible API on http://{args.host}:{args.port}/v1",
          flush=True)
    print(
        "[serve] note: speed depends strongly on draft acceptance and cache "
        "state; the first chat request on a cold expert cache can be very slow "
        "(large prefill fetch). Warm up with short completions.",
        flush=True,
    )
    status = _chat_tokenizer.status()
    if status["mode"] == "auto":
        tokenizer_message = (
            "auto (tiktoken for the first chat; GigaToken prepares only "
            "after that response and accelerates later chats)"
        )
    elif status["mode"] == "on":
        tokenizer_message = "on (GigaToken initialized at startup)"
    else:
        tokenizer_message = "off (tiktoken only; GigaToken is not imported)"
    print(
        f"[serve] chat tokenizer: {tokenizer_message}",
        flush=True,
    )
    if QUEUE_SIZE:
        admission_message = (
            f"one generation at a time; up to {QUEUE_SIZE} request(s) wait "
            "for the slot (roughly arrival order), the rest get 429"
        )
    else:
        admission_message = (
            "one generation at a time; a concurrent request gets an "
            "immediate 429 + Retry-After (--queue N allows waiting)"
        )
    print(f"[serve] admission: {admission_message}", flush=True)
    try:
        srv.serve_forever()
    finally:
        try:
            if _chat_tokenizer is not None:
                _chat_tokenizer.close()
        finally:
            srv.server_close()


if __name__ == "__main__":
    main()
