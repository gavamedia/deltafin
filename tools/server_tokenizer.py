"""Exact optional chat-tokenization acceleration for the persistent server.

The authoritative K3 tokenizer remains untouched.  This module makes a shallow
read-only proxy whose chat-segment hook can use a separately constructed
GigaToken encoder after it has passed exact compatibility checks.  Any
unsupported input or runtime failure reruns the complete chat encode through
the original tiktoken path.

``auto`` deliberately initializes only after the first successful response has
finished.  The server keeps generation admission reserved during this short
post-response qualification, so it never overlaps a target-model pass.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import threading
from types import MethodType
from typing import Any, Callable, Iterable


MODES = frozenset(("auto", "on", "off"))
EXPECTED_GIGATOKEN_VERSION = "0.10.0"
DEFAULT_THREADS = 1
DEFAULT_INIT_WAIT_SECONDS = 5.0
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000
_FATAL_EXCEPTIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)

# The wheel archives are SHA-256-pinned in requirements.txt.  These inner
# digests add a runtime binding: an identically versioned, shadowed, locally
# modified, or differently built package is rejected before any of its code is
# imported.  The Linux wheels were unpacked and statically audited separately;
# physical runtime performance has only been measured on macOS arm64.
_REVIEWED_ARTIFACTS = {
    ("darwin", "arm64"): {
        "package": (
            "a0f664720a9230d293a6cb49b57ecc3fbcd97c25d70ecf28f"
            "c7d12a51aef6d8f"
        ),
        "native": (
            "0a1870479e202bdedd60d1ce04e2877e727622f60f3303493"
            "ca3264b1078e874"
        ),
    },
    ("linux", "x86_64"): {
        "package": (
            "5f90bea3fb5fcd35d82796ec15152c1c95fb739b76fde355d"
            "7b3bf409255cf78"
        ),
        "native": (
            "2a91c21099ffc836b0d601138ccb745734e67e02d47a497eb"
            "dbc807b49b97e99"
        ),
    },
    ("linux", "aarch64"): {
        "package": (
            "8f20cda0c131d6ec913e2c63d02214af49f11b522b6c1139"
            "5af057f46c53ac22"
        ),
        "native": (
            "72733b2ba055a0da64c45ce835622fb6314f8e04bf6ea214"
            "ef2d0467c1a8436a"
        ),
    },
}


class ServerTokenizerError(RuntimeError):
    """The explicitly requested accelerated tokenizer could not be enabled."""


@dataclass(frozen=True)
class ChatEncoding:
    token_ids: list[int]
    rendered_chars: int
    backend: str


@dataclass
class _CallContext:
    rendered_chars: int = 0
    backend: str = "tiktoken"
    visited: bool = False


def parse_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in MODES:
        raise ValueError(
            "GigaToken mode must be auto, on, or off; "
            f"got {value!r}"
        )
    return mode


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reviewed_artifact() -> dict[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        libc_name = platform.libc_ver()[0].lower()
        try:
            gnu_libc = os.confstr("CS_GNU_LIBC_VERSION") or ""
        except (AttributeError, OSError, ValueError):
            gnu_libc = ""
        if libc_name not in {"glibc", "gnu libc"} and not gnu_libc.startswith(
            "glibc "
        ):
            raise RuntimeError(
                "the reviewed GigaToken Linux wheels require glibc"
            )
    artifact = _REVIEWED_ARTIFACTS.get((system, machine))
    if artifact is None:
        raise RuntimeError(
            "no reviewed GigaToken wheel for "
            f"{system or 'unknown'}/{machine or 'unknown'}"
        )
    return artifact


def _hash_package_tree(package_root: Path) -> str:
    items: list[tuple[str, Path]] = []
    for path in package_root.rglob("*"):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        if path.is_symlink():
            raise RuntimeError(
                f"reviewed GigaToken package contains a symlink: {path}"
            )
        items.append((path.relative_to(package_root).as_posix(), path))
    digest = hashlib.sha256()
    for relative, path in sorted(items):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class ServerChatTokenizer:
    """Server-scoped exact chat encoder with lazy native initialization."""

    def __init__(
        self,
        tokenizer,
        model_root: str | os.PathLike[str],
        *,
        mode: str = "auto",
        threads: int | None = None,
        init_wait_seconds: float = DEFAULT_INIT_WAIT_SECONDS,
        log: Callable[[str], object] | None = None,
        backend_factory: Callable[[Any], Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.model_root = Path(model_root).resolve()
        self.mode = parse_mode(mode)
        self.threads = (
            _int_env(
                "K3_SERVER_GIGATOKEN_THREADS", DEFAULT_THREADS, minimum=1
            )
            if threads is None
            else int(threads)
        )
        if self.threads < 1:
            raise ValueError("GigaToken threads must be positive")
        self.init_wait_seconds = float(init_wait_seconds)
        if self.init_wait_seconds < 0:
            raise ValueError(
                "GigaToken initialization wait cannot be negative"
            )

        self._log = log if log is not None else (lambda _message: None)
        self._backend_factory = backend_factory
        self._baseline_segments = tokenizer._encode_chat_segments
        self._baseline_piece = tokenizer._encode_text_piece
        self._state_lock = threading.Lock()
        # Serializes every call into the foreign native library, including its
        # construction/qualification.  close() also crosses this barrier.
        self._native_lock = threading.Lock()
        self._init_done = threading.Event()
        self._init_done.set()
        self._local = threading.local()
        self._backend = None
        self._error: str | None = None
        self._state = "off" if self.mode == "off" else "cold"
        self._uses = 0
        self._expected_fallbacks = 0
        self._unexpected_fallbacks = 0
        self._max_seen_chars = 0

        # apply_chat_template builds authoritative EncodeSegment objects before
        # invoking this hook.  Only this private proxy is patched; the live K3
        # tokenizer used by decoding and universal drafting is never mutated.
        self._proxy = copy.copy(tokenizer)

        def encode_segments(_proxy, segments):
            return self._dispatch_segments(segments)

        self._proxy._encode_chat_segments = MethodType(
            encode_segments, self._proxy
        )

    def _safe_log(self, message: str) -> None:
        try:
            self._log(message)
        except BaseException as exc:
            # Diagnostics must never prevent the authoritative fallback.
            if isinstance(exc, _FATAL_EXCEPTIONS):
                raise

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "mode": self.mode,
                "state": self._state,
                "ready": self._state == "ready",
                "threads": self.threads,
                "init_wait_seconds": self.init_wait_seconds,
                "max_seen_chars": self._max_seen_chars,
                "uses": self._uses,
                "expected_fallbacks": self._expected_fallbacks,
                "unexpected_fallbacks": self._unexpected_fallbacks,
                "error": self._error,
            }

    def encode_chat(
        self,
        conversation,
        *,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> ChatEncoding:
        context = _CallContext()
        if getattr(self._local, "context", None) is not None:
            raise RuntimeError("nested server chat tokenization is unsupported")
        self._local.context = context
        try:
            token_ids = self._proxy.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        finally:
            self._local.context = None
        if not context.visited:
            raise RuntimeError(
                "K3 chat tokenizer did not invoke its segment encoder"
            )
        return ChatEncoding(
            [int(token) for token in token_ids],
            context.rendered_chars,
            context.backend,
        )

    def initialize(self, *, required: bool = False) -> bool:
        """Initialize synchronously; used by explicit ``on`` at server boot."""
        with self._state_lock:
            if self._state == "ready":
                return True
            if self._state in {"closing", "closed"}:
                if required:
                    raise ServerTokenizerError(
                        "GigaToken controller is closing"
                    )
                return False
            if self._state == "starting":
                if required:
                    raise ServerTokenizerError(
                        "GigaToken initialization is already running"
                    )
                return False
            if self.mode == "off":
                if required:
                    raise ServerTokenizerError(
                        "GigaToken is explicitly disabled"
                    )
                return False
            self._state = "starting"
            self._init_done.clear()
        return self._finish_initialization(required=required)

    def after_response(
        self,
        rendered_chars: int,
    ) -> bool:
        """Qualify auto mode synchronously after a completed response."""
        rendered_chars = max(0, int(rendered_chars))
        with self._state_lock:
            self._max_seen_chars = max(
                self._max_seen_chars, rendered_chars
            )
            if (
                self.mode != "auto"
                or self._state != "cold"
            ):
                return False
            self._state = "starting"
            self._init_done.clear()
        return self._finish_initialization(required=False)

    def close(self) -> None:
        with self._state_lock:
            if self._state == "closed":
                return
            self._state = "closing"
            self._init_done.set()

        # Wait for any native construction or encode already in progress.
        # New calls see "closing" and take the authoritative baseline instead.
        with self._native_lock:
            with self._state_lock:
                self._backend = None
                self._state = "closed"
                self._init_done.set()

    def _finish_initialization(self, *, required: bool) -> bool:
        ready = False
        failure: BaseException | None = None
        with self._native_lock:
            with self._state_lock:
                if self._state != "starting":
                    self._init_done.set()
                    return False
            try:
                backend = self._construct_backend()
                self._validate_backend(backend)
            except BaseException as exc:
                failure = exc
                message = f"{type(exc).__name__}: {exc}"
                with self._state_lock:
                    if self._state == "starting":
                        self._state = "failed"
                        self._error = message
                    self._init_done.set()
            else:
                with self._state_lock:
                    if self._state == "starting":
                        self._backend = backend
                        self._state = "ready"
                        self._error = None
                        ready = True
                    self._init_done.set()

        if failure is not None:
            message = f"{type(failure).__name__}: {failure}"
            self._safe_log(
                "[tokenizer] GigaToken unavailable; retaining exact "
                f"tiktoken path ({message})"
            )
            if isinstance(failure, _FATAL_EXCEPTIONS):
                raise failure
            if required:
                raise ServerTokenizerError(
                    "explicit GigaToken startup failed: " + message
                ) from failure
            return False

        if ready:
            self._safe_log(
                "[tokenizer] GigaToken ready for subsequent server chats"
            )
        return ready

    def _construct_backend(self):
        if self._backend_factory is not None:
            return self._backend_factory(self.tokenizer)
        try:
            distribution = importlib.metadata.distribution("gigatoken")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "gigatoken is not installed; install requirements.txt "
                "or select --gigatoken off"
            ) from exc
        version = distribution.version
        if version != EXPECTED_GIGATOKEN_VERSION:
            raise RuntimeError(
                "expected gigatoken "
                f"{EXPECTED_GIGATOKEN_VERSION}, found {version}"
            )
        artifact = _reviewed_artifact()
        package_root = distribution.locate_file("gigatoken").resolve()
        if not package_root.is_dir():
            raise RuntimeError(
                f"missing installed GigaToken package: {package_root}"
            )
        package_digest = _hash_package_tree(package_root)
        if package_digest != artifact["package"]:
            raise RuntimeError(
                "installed GigaToken files do not match the reviewed wheel"
            )

        module = importlib.import_module("gigatoken")
        module_path = Path(module.__file__).resolve()
        if module_path.parent != package_root:
            raise RuntimeError(
                "imported GigaToken is shadowing the reviewed distribution"
            )
        native_module = importlib.import_module(
            "gigatoken.gigatoken_rs"
        )
        native_path = Path(native_module.__file__).resolve()
        if (
            native_path.parent != package_root
            or _sha256_file(native_path) != artifact["native"]
        ):
            raise RuntimeError(
                "loaded GigaToken native module differs from the reviewed wheel"
            )
        rank_path = (self.model_root / "tiktoken.model").resolve()
        if not rank_path.is_file():
            raise RuntimeError(f"missing local K3 rank file: {rank_path}")
        native = module.Tokenizer.from_tiktoken(
            rank_path,
            pretokenizer="kimi",
            special_tokens=dict(self.tokenizer.special_tokens),
        )
        return native.as_tiktoken()

    def _validate_backend(self, backend) -> None:
        original = self.tokenizer.model
        special_tokens = dict(self.tokenizer.special_tokens)
        if len(special_tokens) != 256:
            raise RuntimeError(
                f"expected 256 K3 special tokens, found {len(special_tokens)}"
            )
        if int(backend.n_vocab) != int(original.n_vocab):
            raise RuntimeError(
                f"vocabulary size {backend.n_vocab} != {original.n_vocab}"
            )
        observed_specials = getattr(backend, "_special_tokens", None)
        if observed_specials != special_tokens:
            raise RuntimeError("GigaToken special-token map differs from K3")

        special_rows = backend.encode_batch(
            list(special_tokens),
            num_threads=self.threads,
            allowed_special="all",
        )
        for (text, token_id), row in zip(
            special_tokens.items(), special_rows
        ):
            if list(row) != [token_id]:
                raise RuntimeError(
                    f"GigaToken special mapping differs for {text!r}"
                )

        canaries = (
            "",
            "Hello, world! I'll test 123456789 and contractions.",
            "人工智能正在改变世界。한글 العربية नमस्ते",
            "emoji 👨🏽‍💻🏳️‍🌈 combining re\u0301sume\u0301",
            "spaces \t\r\n and punctuation /path/to/file...",
        )
        for allow_special in (False, True):
            expected = [
                self._baseline_piece(
                    text, allow_special_tokens=allow_special
                )
                for text in canaries
            ]
            kwargs = (
                {"allowed_special": "all"}
                if allow_special
                else {
                    "allowed_special": (),
                    "disallowed_special": (),
                }
            )
            actual = backend.encode_batch(
                list(canaries),
                num_threads=self.threads,
                **kwargs,
            )
            if [list(row) for row in actual] != expected:
                raise RuntimeError(
                    "GigaToken Kimi pretokenizer canary differs from K3"
                )

        for token_id in range(int(original.n_vocab)):
            if (
                backend.decode_single_token_bytes(token_id)
                != original.decode_single_token_bytes(token_id)
            ):
                raise RuntimeError(
                    f"GigaToken token bytes differ at ID {token_id}"
                )

    def _dispatch_segments(self, segments: Iterable[Any]) -> list[int]:
        segment_list = list(segments)
        rendered_chars = sum(
            len(str(segment.text)) for segment in segment_list
        )
        context = getattr(self._local, "context", None)
        if context is None:
            # The proxy hook is private to encode_chat(), but fail safely if a
            # caller bypasses that public boundary.
            return self._baseline_segments(segment_list)
        context.visited = True
        context.rendered_chars = rendered_chars
        with self._state_lock:
            self._max_seen_chars = max(
                self._max_seen_chars, rendered_chars
            )
            starting = self.mode == "auto" and self._state == "starting"
            backend = self._backend if self._state == "ready" else None

        # Auto qualification begins only after the preceding response.  If an
        # API client submits the next turn immediately, wait briefly for that
        # already-running qualification so the next turn receives the warm
        # path.  Failure or timeout remains an exact baseline encode.
        if starting:
            self._init_done.wait(self.init_wait_seconds)
            with self._state_lock:
                backend = (
                    self._backend if self._state == "ready" else None
                )

        if backend is None:
            context.backend = "tiktoken"
            return self._baseline_segments(segment_list)

        failure: BaseException | None = None
        expected_fallback = False
        stale = False
        result: list[int] | None = None
        with self._native_lock:
            # A queued request must not call a backend that another request
            # disabled while this one waited at the native boundary.
            with self._state_lock:
                stale = (
                    self._state != "ready"
                    or self._backend is not backend
                )
            if not stale:
                try:
                    result = self._encode_segments_native(
                        segment_list, backend
                    )
                except BaseException as exc:
                    if isinstance(exc, NotImplementedError):
                        expected_fallback = True
                    else:
                        failure = exc

            # close() marks the state before crossing _native_lock.  Discard a
            # native result if shutdown or another failure began meanwhile.
            with self._state_lock:
                stale = stale or (
                    self._state != "ready"
                    or self._backend is not backend
                )

                if expected_fallback and not stale:
                    self._expected_fallbacks += 1
                elif failure is not None and not stale:
                    message = f"{type(failure).__name__}: {failure}"
                    self._backend = None
                    self._state = "failed"
                    self._error = message
                    self._unexpected_fallbacks += 1
                elif result is not None and not stale:
                    self._uses += 1

        if failure is not None and isinstance(
            failure, _FATAL_EXCEPTIONS
        ):
            raise failure

        if stale:
            context.backend = "tiktoken"
            return self._baseline_segments(segment_list)

        if expected_fallback:
            context.backend = "tiktoken-special-fallback"
            return self._baseline_segments(segment_list)

        if failure is not None:
            message = f"{type(failure).__name__}: {failure}"
            self._safe_log(
                "[tokenizer] disabling GigaToken after an encoding error; "
                "the complete chat tokenization was rerun with tiktoken "
                f"({message})"
            )
            context.backend = "tiktoken-error-fallback"
            return self._baseline_segments(segment_list)

        if result is None:
            # A native backend returning no rows is an unexpected contract
            # violation even if it somehow raised no Python exception.
            with self._state_lock:
                if self._state == "ready" and self._backend is backend:
                    self._backend = None
                    self._state = "failed"
                    self._error = "GigaToken returned no result"
                    self._unexpected_fallbacks += 1
            context.backend = "tiktoken-error-fallback"
            return self._baseline_segments(segment_list)

        context.backend = "gigatoken"
        return result

    def _split_pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for offset in range(
            0, len(text), TIKTOKEN_MAX_ENCODE_CHARS
        ):
            pieces.extend(
                self.tokenizer._split_whitespaces_or_nonwhitespaces(
                    text[
                        offset:offset + TIKTOKEN_MAX_ENCODE_CHARS
                    ],
                    MAX_NO_WHITESPACES_CHARS,
                )
            )
        return pieces or [""]

    def _encode_segments_native(self, segments, backend) -> list[int]:
        jobs: dict[bool, list[str]] = {False: [], True: []}
        tape: list[tuple[bool, int]] = []
        for segment in segments:
            allow_special = bool(segment.allow_special)
            pieces = self._split_pieces(str(segment.text))
            jobs[allow_special].extend(pieces)
            tape.append((allow_special, len(pieces)))

        rows: dict[bool, list[list[int]]] = {False: [], True: []}
        if jobs[False]:
            encoded = backend.encode_batch(
                jobs[False],
                num_threads=self.threads,
                allowed_special=(),
                disallowed_special=(),
            )
            rows[False] = [list(row) for row in encoded]
        if jobs[True]:
            encoded = backend.encode_batch(
                jobs[True],
                num_threads=self.threads,
                allowed_special="all",
            )
            rows[True] = [list(row) for row in encoded]
        for policy in (False, True):
            if len(rows[policy]) != len(jobs[policy]):
                raise RuntimeError(
                    "GigaToken returned the wrong batch length"
                )

        cursors = {False: 0, True: 0}
        output: list[int] = []
        n_vocab = int(self.tokenizer.model.n_vocab)
        for policy, count in tape:
            start = cursors[policy]
            for row in rows[policy][start:start + count]:
                for token in row:
                    if (
                        type(token) is not int
                        or token < 0
                        or token >= n_vocab
                    ):
                        raise RuntimeError(
                            f"GigaToken returned invalid token {token!r}"
                        )
                    output.append(token)
            cursors[policy] += count
        return output


def default_mode() -> str:
    return parse_mode(os.environ.get("K3_SERVER_GIGATOKEN", "auto"))
