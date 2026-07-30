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
  * One generation at a time (a global lock). Concurrency would be meaningless
    at this speed.
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
import os
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
RESPONSE_MEMO_ENTRIES = int(
    os.environ.get("K3_RESPONSE_MEMO_ENTRIES", "32"))
RESPONSE_MARKER = "<|open|>response<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"

_lock = threading.Lock()
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


def _gen(ids, max_new, on_delta=None):
    """Run one generation under the global lock; stream decoded-text deltas."""
    cache = kr.ml.KimiDynamicCache(kr.config)
    decoder = kr.IncrementalTokenDecoder(_tok) if on_delta else None

    def on_token(t):
        if t == kr.EOS_ID:
            return
        delta = decoder.append(t)
        if delta:
            on_delta(delta)

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
    return out, _tok.decode(out), finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[serve] {self.address_string()} {fmt % a}", flush=True)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _err(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "deltafin"}]})
        else:
            self._err(404, f"no route {self.path}")

    def do_POST(self):
        rendered_chars = 0
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        except json.JSONDecodeError:
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
        if req_max:
            max_new = int(req_max)
        else:
            max_new = 1_000_000 if chat else 256
        if MAX_TOKENS_CAP:
            max_new = min(max_new, MAX_TOKENS_CAP)
        stream = bool(body.get("stream"))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:20]
        created = int(time.time())

        if not _lock.acquire(timeout=5):
            return self._err(429, "a generation is already running (Deltafin serves one at a time)")
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
                        ids, max_new, on_delta=on_delta)
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
                out, text, finish = _gen(ids, max_new)
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
        except BrokenPipeError:
            print("[serve] client disconnected mid-generation", flush=True)
        except Exception as e:
            print(f"[serve] error: {e!r}", flush=True)
            try:
                self._err(500, str(e))
            except Exception:
                pass
        finally:
            _lock.release()


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
    return ap


def main():
    ap = _build_parser()
    args = ap.parse_args()
    try:
        gigatoken_mode = server_tokenizer.parse_mode(args.gigatoken)
    except ValueError as exc:
        ap.error(str(exc))
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
