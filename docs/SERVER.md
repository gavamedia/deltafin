# OpenAI-compatible server reference

This is the detailed behavior of `deltafin serve` — which OpenAI API features it accepts, which it refuses, and the automatic speedups it applies. For starting the server, see [the README](README.md).

## What requests can include

The server supports a deliberately small slice of the OpenAI API, and it is strict about it: if a request asks for something the server cannot do exactly, the request is refused up front ("fail-closed") instead of being quietly ignored. On both `/v1/chat/completions` and `/v1/completions`, a request may use:

- One generated answer per request (`n` and `best_of`, if present, must be `1`).
- `max_tokens` or `max_completion_tokens` to cap the answer's length.
- A normal single response, or streaming via server-sent events, including `stream_options.include_usage`.
- A `model` name — when present, it must be the model this server advertises (`deltafin-kimi-k3`).
- `user` and object-valued `metadata`, which are accepted but never change generation.

Many client libraries also send harmless default values without being asked. Those are accepted only when they change nothing: zero penalties, an empty stop or tool list, `store: false`, text-only modalities, and `response_format: {"type": "text"}`.

Output is always greedy and reproducible: the same prompt produces the same tokens every time. `temperature` and `top_p` are accepted for compatibility and checked for valid ranges, but they intentionally do not change the output.

## What gets refused

Chat requests accept `reasoning_effort` (`low`, `high`, or `max`) to control K3's thinking depth; when absent, the `K3_REASONING_EFFORT` environment variable and then the chat template's own default of `max` apply.

These OpenAI features are not implemented: non-default `stop` sequences, penalties, logit bias, logprobs, seeds, suffix/echo, tools and tool choice, structured response formats, prediction, and service tiers. Sending any of them — or any field the server does not recognize — returns an ordinary OpenAI-shaped HTTP 400 error before the model is ever entered, so a client can never believe an option worked when it didn't.

A request with nothing to continue is refused the same way. `/v1/completions` needs a non-empty string `prompt` and `/v1/chat/completions` a non-empty `messages` array; an empty, missing or non-string `prompt` (streaming or not) gets an HTTP 400 `invalid_request_error` rather than a retryable 5xx, because retrying cannot help. Whitespace is text and is accepted, and a chat message with empty `content` is still a rendered chat turn. The `deltafin` command line applies the same rule to a raw `--prompt ""` before it loads the model.

The server also generates one answer at a time, on purpose: a second generation request arriving mid-generation receives an OpenAI-shaped HTTP 429 ("busy"), while `/v1/models` and other non-generation requests keep responding normally.

## Text only, for now

The production runtime is text-only. K3's pinned model files do include the original vision tower and multimodal projector, but that execution path is not connected yet. Chat parts containing images, audio or other non-text content are rejected before inference begins, rather than being silently reduced to meaningless placeholder tokens.

## Speedups that happen automatically

Chat requests get the two optimizations that matter most as a conversation grows. Neither has a switch, because each one either proves it is safe for a given request or quietly steps aside:

- **Conversation state reuse.** After a response completes successfully, the server may keep one snapshot of the model's internal state (the provider-owned KDA/MLA boundary). If the next request is a strict continuation — the same conversation with new content only appended — generation can branch from that snapshot and process just the new part instead of the whole history. Anything else (an edited, branched or truncated history, a changed model or configuration, a memory rejection, a failed response) discards the snapshot and safely performs an ordinary full pass.
- **Automatic DSpark drafting.** The included DSpark draft model proposes up to seven tokens ahead when hardware, memory, context and measured request economics all qualify. Full K3 verifies every proposed token, so the output is identical to running K3 alone — drafting only changes how fast tokens arrive. Draft state is paired with the exact K3 snapshot above, and losing draft state never invalidates usable K3 state.

## The exact-response memo

The server keeps a small memory of recent answers, and a new request that is byte-for-byte identical to a remembered one can be answered from it instantly. It defaults to 32 entries and 64 MiB; disable it with `--response-memo-entries 0 --response-memo-bytes 0`. This is exact matching, not semantic caching — change even one token of the prompt and it is a miss.
