use std::collections::VecDeque;
use std::error::Error;
use std::io::{self, Read, Write};
use std::net::ToSocketAddrs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use super::response_memo::{DeterministicResponseMemo, ResponseMode};
use super::types::{
    AuthoritativeTarget, ChatMessage, ClientPresence, StreamGenerationError, StreamPublication,
    TargetDelta, TargetDeltaChannel, TargetDeltaSink, TargetOutput, TargetPrompt, TargetRequest,
    TargetStreamSummary, TokenUsage,
};

/// A million-token chat can readily exceed one MiB once JSON escaping and
/// message metadata are included. Keep the admission finite, but large enough
/// for realistic long-context requests without a hidden early ceiling.
pub const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_REQUEST_BODY_BYTES: usize = 1024 * 1024 * 1024;
pub const DEFAULT_RESPONSE_MEMO_ENTRIES: usize = 32;
pub const DEFAULT_RESPONSE_MEMO_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_RESPONSE_MEMO_BYTES: usize = 1024 * 1024 * 1024;
/// Each waiting request holds one worker thread and one open connection for
/// up to an entire generation, so the queue stays deliberately small.
pub const MAX_GENERATION_QUEUE_SLOTS: usize = 64;
const DEFAULT_MAX_NEW_TOKENS: usize = 1_000_000;
const DEFAULT_COMPLETION_TOKENS: usize = 256;
const DEFAULT_CHAT_TOKENS: usize = 1_000_000;
const OWNER: &str = "deltafin";
static RESPONSE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ServerConfig {
    pub model_id: String,
    pub max_request_body_bytes: usize,
    pub max_new_tokens: usize,
    pub default_completion_tokens: usize,
    pub default_chat_tokens: usize,
    pub response_memo_entries: usize,
    pub response_memo_bytes: usize,
    /// Concurrent generation requests allowed to wait, in arrival order, for
    /// the single generation slot. Zero — the default — preserves the
    /// documented immediate 429 for every concurrent request.
    pub generation_queue_slots: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_id: "deltafin-kimi-k3".to_owned(),
            max_request_body_bytes: DEFAULT_MAX_REQUEST_BODY_BYTES,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            default_completion_tokens: DEFAULT_COMPLETION_TOKENS,
            default_chat_tokens: DEFAULT_CHAT_TOKENS,
            response_memo_entries: DEFAULT_RESPONSE_MEMO_ENTRIES,
            response_memo_bytes: DEFAULT_RESPONSE_MEMO_BYTES,
            generation_queue_slots: 0,
        }
    }
}

impl ServerConfig {
    fn validate(&self) -> crate::error::Result<()> {
        if self.model_id.trim().is_empty() {
            return Err("OpenAI server model ID cannot be empty".into());
        }
        if self.max_request_body_bytes == 0 || self.max_request_body_bytes > MAX_REQUEST_BODY_BYTES
        {
            return Err(format!(
                "OpenAI server request-body limit must be in 1..={MAX_REQUEST_BODY_BYTES} bytes"
            )
            .into());
        }
        if self.max_new_tokens == 0 {
            return Err("OpenAI server token limit must be positive".into());
        }
        if self.default_completion_tokens == 0
            || self.default_completion_tokens > self.max_new_tokens
        {
            return Err("completion default must be within the server token limit".into());
        }
        if self.default_chat_tokens == 0 || self.default_chat_tokens > self.max_new_tokens {
            return Err("chat default must be within the server token limit".into());
        }
        if self.response_memo_bytes > MAX_RESPONSE_MEMO_BYTES {
            return Err(format!(
                "OpenAI server response-memo limit must be in 0..={MAX_RESPONSE_MEMO_BYTES} bytes"
            )
            .into());
        }
        if self.generation_queue_slots > MAX_GENERATION_QUEUE_SLOTS {
            return Err(format!(
                "OpenAI server generation queue must be in 0..={MAX_GENERATION_QUEUE_SLOTS} waiting slots"
            )
            .into());
        }
        Ok(())
    }
}

/// Thread-safe OpenAI request dispatcher.
///
/// One owned generation permit protects the target and its response-boundary
/// transaction. A concurrent completion is rejected immediately — or, when
/// the operator configures waiting slots, admitted to a bounded
/// arrival-ordered queue — instead of waiting indefinitely behind a
/// generation that may run for hours; model discovery and other
/// non-generation routes do not need the permit. A client that provably
/// disconnects, whether waiting or generating, is abandoned so its turn costs
/// no target compute.
pub struct OpenAiService<T> {
    shared: Arc<ServiceInner<T>>,
}

struct ServiceInner<T> {
    target: Mutex<T>,
    generation_gate: Arc<GenerationGate>,
    response_memo: Mutex<DeterministicResponseMemo>,
    config: ServerConfig,
    created: u64,
}

/// How often a queued request re-checks its client's socket while waiting for
/// the generation slot.
const QUEUED_DISCONNECT_PROBE_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Debug, Default)]
struct GenerationGate {
    state: Mutex<GateState>,
    turned: Condvar,
}

#[derive(Debug, Default)]
struct GateState {
    generating: bool,
    /// Tickets of admitted waiters, front first. Strict arrival order: a
    /// waiter claims the slot only while its ticket is the queue front, so a
    /// condvar wake can never reorder admissions.
    waiting: VecDeque<u64>,
    next_ticket: u64,
}

/// Admission decision for one generation-capable request.
enum Admission {
    /// The caller owns the sole generation slot immediately.
    Permit(GenerationPermit),
    /// The slot is taken but a waiting slot was free; the ticket holds the
    /// caller's place in arrival order.
    Queued(QueueTicket),
    /// The slot is taken and every waiting slot is occupied.
    Busy,
}

impl GenerationGate {
    fn lock_state(&self) -> std::sync::MutexGuard<'_, GateState> {
        // No caller code runs while this lock is held, so poisoning can only
        // follow a panic elsewhere; admission must keep working regardless.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn admit(self: &Arc<Self>, waiting_slots: usize) -> Admission {
        let mut state = self.lock_state();
        if !state.generating && state.waiting.is_empty() {
            state.generating = true;
            return Admission::Permit(GenerationPermit {
                gate: Arc::clone(self),
            });
        }
        if state.waiting.len() < waiting_slots {
            let number = state.next_ticket;
            state.next_ticket += 1;
            state.waiting.push_back(number);
            return Admission::Queued(QueueTicket {
                gate: Arc::clone(self),
                number,
                claimed: false,
            });
        }
        Admission::Busy
    }
}

/// Owned so the listener can reserve the sole generation slot before moving
/// an accepted completion request to its worker. Dropping this on every path
/// reopens admission — and wakes the queue front — only after the previous
/// transaction is over.
struct GenerationPermit {
    gate: Arc<GenerationGate>,
}

impl Drop for GenerationPermit {
    fn drop(&mut self) {
        let mut state = self.gate.lock_state();
        state.generating = false;
        drop(state);
        self.gate.turned.notify_all();
    }
}

/// One place in the arrival-ordered wait queue. Dropping an unclaimed ticket
/// (client disconnect, body-read failure, worker panic) removes it and wakes
/// the queue, so a departed waiter can never wedge admission.
struct QueueTicket {
    gate: Arc<GenerationGate>,
    number: u64,
    claimed: bool,
}

impl QueueTicket {
    /// Block until this ticket reaches the queue front and the generation
    /// slot frees, probing `client` between waits. Returns `None` — freeing
    /// the waiting slot — once the client has provably disconnected.
    fn wait_for_permit(mut self, client: &ClientPresence) -> Option<GenerationPermit> {
        loop {
            {
                let mut state = self.gate.lock_state();
                loop {
                    if !state.generating && state.waiting.front() == Some(&self.number) {
                        state.waiting.pop_front();
                        state.generating = true;
                        self.claimed = true;
                        return Some(GenerationPermit {
                            gate: Arc::clone(&self.gate),
                        });
                    }
                    let (reacquired, timeout) = self
                        .gate
                        .turned
                        .wait_timeout(state, QUEUED_DISCONNECT_PROBE_INTERVAL)
                        .unwrap_or_else(PoisonError::into_inner);
                    state = reacquired;
                    if timeout.timed_out() {
                        break;
                    }
                }
            }
            // The probe is a syscall; run it outside the admission lock.
            if client.disconnected() {
                return None;
            }
        }
    }
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        if self.claimed {
            return;
        }
        let mut state = self.gate.lock_state();
        state.waiting.retain(|number| *number != self.number);
        drop(state);
        // Removing the queue front can make the next waiter claimable.
        self.gate.turned.notify_all();
    }
}

impl<T> Clone for OpenAiService<T> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<T: AuthoritativeTarget> OpenAiService<T> {
    pub fn new(target: T, config: ServerConfig) -> crate::error::Result<Self> {
        config.validate()?;
        let response_memo = DeterministicResponseMemo::new(
            config.response_memo_entries,
            config.response_memo_bytes,
        );
        Ok(Self {
            shared: Arc::new(ServiceInner {
                target: Mutex::new(target),
                generation_gate: Arc::new(GenerationGate::default()),
                response_memo: Mutex::new(response_memo),
                config,
                created: unix_timestamp(),
            }),
        })
    }

    pub fn config(&self) -> &ServerConfig {
        &self.shared.config
    }

    /// Consume and answer one `tiny_http` request.
    ///
    /// This synchronous entry point has no socket probe for its client, so a
    /// queued caller waits as long as its turn takes; `serve_forever` is the
    /// path that observes disconnects.
    pub fn handle_request(&self, request: Request) -> io::Result<()> {
        let permit = if request_requires_generation_permit(&request) {
            let queue_slots = self.shared.config.generation_queue_slots;
            match self.shared.generation_gate.admit(queue_slots) {
                Admission::Permit(permit) => Some(permit),
                Admission::Queued(ticket) => Some(
                    ticket
                        .wait_for_permit(&ClientPresence::assumed_present())
                        .expect("an assumed-present client cannot disconnect"),
                ),
                Admission::Busy => return respond_busy(request),
            }
        } else {
            None
        };
        self.handle_admitted_request(request, permit)
    }

    fn handle_admitted_request(
        &self,
        mut request: Request,
        permit: Option<GenerationPermit>,
    ) -> io::Result<()> {
        let method = HttpMethod::from(request.method());
        if method == HttpMethod::Post {
            let body = read_bounded_body(&mut request, self.shared.config.max_request_body_bytes);
            return self.respond_to_post(request, permit, body, &ClientPresence::assumed_present());
        }
        let path = request_path(&request);
        let response = self.dispatch(method, &path, &[]);
        request.respond(response.into_tiny_http())
    }

    /// Read the body and answer while already holding the generation permit,
    /// exactly as a single uncontended request always has.
    fn respond_generation(
        &self,
        mut request: Request,
        permit: GenerationPermit,
        client: &ClientPresence,
    ) -> io::Result<()> {
        let body = read_bounded_body(&mut request, self.shared.config.max_request_body_bytes);
        self.respond_to_post(request, Some(permit), body, client)
    }

    /// Wait for the generation slot from a queue ticket, dropping the request
    /// silently if its client disconnects first.
    fn respond_queued_generation(
        &self,
        mut request: Request,
        ticket: QueueTicket,
        client: &ClientPresence,
    ) -> io::Result<()> {
        // Drain the request body before waiting: a malformed or oversized
        // request is answered right away without ever consuming a generation
        // turn, and the client's send side is not left blocked on a full
        // socket buffer for however long the queue takes.
        let body = match read_bounded_body(&mut request, self.shared.config.max_request_body_bytes)
        {
            Ok(body) => body,
            Err(error) => {
                // The unclaimed ticket drops here, freeing its waiting slot.
                return request.respond(error.into_response().into_tiny_http());
            }
        };
        let Some(permit) = ticket.wait_for_permit(client) else {
            eprintln!("[serve] queued client disconnected; its waiting slot was freed");
            abandon_unanswered(request);
            return Ok(());
        };
        self.respond_to_post(request, Some(permit), Ok(body), client)
    }

    /// Answer one POST whose body has already been read (or failed to read).
    ///
    /// Every generation-capable request funnels through here: the
    /// immediate-permit flow, the queued flow, and fabricated test requests
    /// (which probe as assumed-present).
    fn respond_to_post(
        &self,
        request: Request,
        _permit: Option<GenerationPermit>,
        body: Result<Vec<u8>, ApiFailure>,
        client: &ClientPresence,
    ) -> io::Result<()> {
        if client.disconnected() {
            // The last pre-generation moment to notice a vanished client:
            // skip the whole response instead of aborting one layer into
            // prefill. The permit drops with this frame, reopening admission.
            eprintln!("[serve] client disconnected before generation; request abandoned");
            abandon_unanswered(request);
            return Ok(());
        }
        let path = request_path(&request);
        let response = match body {
            Ok(body) => {
                if let Some(stream_kind) = StreamKind::for_path(&path)
                    && body_requests_stream(&body)
                {
                    return self.handle_stream_request(request, stream_kind, &body, client);
                }
                self.dispatch_for_client(HttpMethod::Post, &path, &body, client)
            }
            Err(error) => error.into_response(),
        };

        let response_succeeded = (200..300).contains(&response.status);
        let settle_target = response.settle_target;
        let result = request.respond(response.into_tiny_http());
        if settle_target && let Ok(mut target) = self.shared.target.lock() {
            target.finish_target_response(if response_succeeded && result.is_ok() {
                StreamPublication::Complete
            } else {
                StreamPublication::Aborted
            });
        }
        result
    }

    /// Route a request whose client has no liveness probe, mirroring the
    /// interrupt-source pattern: tests and embedders route here, and the HTTP
    /// paths that can observe a socket route through [`Self::dispatch_for_client`].
    fn dispatch(&self, method: HttpMethod, path: &str, body: &[u8]) -> WireResponse {
        self.dispatch_for_client(method, path, body, &ClientPresence::assumed_present())
    }

    fn dispatch_for_client(
        &self,
        method: HttpMethod,
        path: &str,
        body: &[u8],
        client: &ClientPresence,
    ) -> WireResponse {
        match (method, path) {
            (HttpMethod::Get, "/v1/models" | "/models") => self.models(),
            (HttpMethod::Post, "/v1/completions" | "/completions") => self.completion(body, client),
            (HttpMethod::Post, "/v1/chat/completions" | "/chat/completions") => {
                self.chat_completion(body, client)
            }
            (
                _,
                "/v1/models"
                | "/models"
                | "/v1/completions"
                | "/completions"
                | "/v1/chat/completions"
                | "/chat/completions",
            ) => ApiFailure::new(
                405,
                "method_not_allowed",
                "method is not allowed for this endpoint",
            )
            .into_response(),
            _ => ApiFailure::new(404, "not_found", "endpoint not found").into_response(),
        }
    }

    fn models(&self) -> WireResponse {
        WireResponse::json(
            200,
            json!({
                "object": "list",
                "data": [{
                    "id": self.shared.config.model_id,
                    "object": "model",
                    "created": self.shared.created,
                    "owned_by": OWNER,
                }],
            }),
        )
    }

    fn completion(&self, body: &[u8], client: &ClientPresence) -> WireResponse {
        let request = match self.parse_completion(body) {
            Ok(request) => request,
            Err(error) => return error.into_response(),
        };
        let generated = match self.generate(&request, ResponseMode::Completion, client) {
            Ok(output) => output,
            Err(response) => return response,
        };
        let output = &generated.output;
        let created = unix_timestamp();
        let mut response = json!({
            "id": response_id("cmpl-", created),
            "object": "text_completion",
            "created": created,
            "model": self.shared.config.model_id,
            "choices": [{
                "index": 0,
                "text": output.text(),
                "logprobs": Value::Null,
                "finish_reason": output.finish_reason().as_openai_str(),
            }],
        });
        insert_usage(&mut response, output);
        WireResponse::target_json(200, response, generated.target_invoked)
    }

    fn chat_completion(&self, body: &[u8], client: &ClientPresence) -> WireResponse {
        let request = match self.parse_chat(body) {
            Ok(request) => request,
            Err(error) => return error.into_response(),
        };
        let generated = match self.generate(&request, ResponseMode::Chat, client) {
            Ok(output) => output,
            Err(response) => return response,
        };
        let output = &generated.output;
        let created = unix_timestamp();
        let mut message = Map::new();
        message.insert("role".to_owned(), Value::String("assistant".to_owned()));
        message.insert(
            "content".to_owned(),
            Value::String(output.text().to_owned()),
        );
        if let Some(reasoning) = output.reasoning_content() {
            message.insert(
                "reasoning_content".to_owned(),
                Value::String(reasoning.to_owned()),
            );
        }
        let mut response = json!({
            "id": response_id("chatcmpl-", created),
            "object": "chat.completion",
            "created": created,
            "model": self.shared.config.model_id,
            "choices": [{
                "index": 0,
                "message": Value::Object(message),
                "finish_reason": output.finish_reason().as_openai_str(),
            }],
        });
        insert_usage(&mut response, output);
        WireResponse::target_json(200, response, generated.target_invoked)
    }

    fn generate(
        &self,
        request: &TargetRequest,
        mode: ResponseMode,
        client: &ClientPresence,
    ) -> Result<GeneratedOutput, WireResponse> {
        if let Ok(mut memo) = self.shared.response_memo.lock()
            && let Some(output) = memo.get(mode, request)
        {
            return Ok(GeneratedOutput {
                output,
                target_invoked: false,
            });
        }
        let mut target = self.shared.target.lock().map_err(|_| {
            ApiFailure::new(
                500,
                "server_error",
                "authoritative target owner is unavailable",
            )
            .into_response()
        })?;
        let output = match target.generate_target(request, client) {
            Ok(output) => output,
            Err(error) => {
                let mut response =
                    ApiFailure::new(500, "server_error", error.to_string()).into_response();
                // The target was entered and may have staged a partial reuse
                // branch before failing. It must receive Aborted after the
                // error body is published, even though no output is memoized.
                response.settle_target = true;
                return Err(response);
            }
        };
        if let Ok(mut memo) = self.shared.response_memo.lock() {
            memo.put(mode, request, &output);
        }
        Ok(GeneratedOutput {
            output,
            target_invoked: true,
        })
    }

    fn parse_completion(&self, body: &[u8]) -> Result<TargetRequest, ApiFailure> {
        self.parse_completion_mode(body, false)
            .map(|parsed| parsed.request)
    }

    fn parse_completion_mode(
        &self,
        body: &[u8],
        streaming: bool,
    ) -> Result<ParsedTargetRequest, ApiFailure> {
        let parsed = ParsedObject::parse(body)?;
        let object = &parsed.0;
        validate_request_contract(
            object,
            RequestEndpoint::Completion,
            &self.shared.config.model_id,
        )?;
        validate_stream_mode(object, streaming)?;
        validate_single_choice(object)?;
        // An empty prompt can never be continued; refuse it here as a client
        // error, like an empty chat `messages` array, rather than letting it
        // fail inside the engine as a retryable server error.
        let prompt = object
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|prompt| !prompt.is_empty())
            .ok_or_else(|| {
                ApiFailure::invalid(
                    "`prompt` must be a non-empty JSON string for native completions",
                )
            })?;
        let max_new_tokens = parse_max_tokens(
            object,
            self.shared.config.default_completion_tokens,
            self.shared.config.max_new_tokens,
        )?;
        Ok(ParsedTargetRequest {
            request: TargetRequest {
                prompt: TargetPrompt::Completion(prompt.to_owned()),
                max_new_tokens,
                reasoning_effort: None,
            },
            include_usage: parse_stream_usage(object, streaming)?,
        })
    }

    fn parse_chat(&self, body: &[u8]) -> Result<TargetRequest, ApiFailure> {
        self.parse_chat_mode(body, false)
            .map(|parsed| parsed.request)
    }

    fn parse_chat_mode(
        &self,
        body: &[u8],
        streaming: bool,
    ) -> Result<ParsedTargetRequest, ApiFailure> {
        let parsed = ParsedObject::parse(body)?;
        let object = &parsed.0;
        validate_request_contract(object, RequestEndpoint::Chat, &self.shared.config.model_id)?;
        validate_stream_mode(object, streaming)?;
        validate_single_choice(object)?;
        let values = object
            .get("messages")
            .and_then(Value::as_array)
            .filter(|messages| !messages.is_empty())
            .ok_or_else(|| ApiFailure::invalid("`messages` must be a non-empty JSON array"))?;
        let mut messages = Vec::with_capacity(values.len());
        for value in values {
            let source = value
                .as_object()
                .ok_or_else(|| ApiFailure::invalid("every chat message must be an object"))?;
            let role = source
                .get("role")
                .and_then(Value::as_str)
                .filter(|role| !role.is_empty())
                .ok_or_else(|| {
                    ApiFailure::invalid("every chat message must have a non-empty string `role`")
                })?;
            validate_chat_message_contract(source, role)?;
            let content = source
                .get("content")
                .cloned()
                .ok_or_else(|| ApiFailure::invalid("every chat message must include `content`"))?;
            if !matches!(content, Value::String(_) | Value::Array(_) | Value::Null) {
                return Err(ApiFailure::invalid(
                    "chat message `content` must be text, an array, or null",
                ));
            }
            validate_text_only_chat_content(&content)?;
            let mut additional_fields = source.clone();
            additional_fields.remove("role");
            additional_fields.remove("content");
            messages.push(ChatMessage {
                role: role.to_owned(),
                content,
                additional_fields,
            });
        }
        let max_new_tokens = parse_max_tokens(
            object,
            self.shared.config.default_chat_tokens,
            self.shared.config.max_new_tokens,
        )?;
        Ok(ParsedTargetRequest {
            request: TargetRequest {
                prompt: TargetPrompt::Chat(messages),
                max_new_tokens,
                reasoning_effort: parse_reasoning_effort(object.get("reasoning_effort"))?,
            },
            include_usage: parse_stream_usage(object, streaming)?,
        })
    }

    fn handle_stream_request(
        &self,
        request: Request,
        kind: StreamKind,
        body: &[u8],
        client: &ClientPresence,
    ) -> io::Result<()> {
        let parsed = match kind {
            StreamKind::Completion => self.parse_completion_mode(body, true),
            StreamKind::Chat => self.parse_chat_mode(body, true),
        };
        let parsed = match parsed {
            Ok(parsed) => parsed,
            Err(error) => return request.respond(error.into_response().into_tiny_http()),
        };
        let mode = kind.response_mode();
        let cached = self
            .shared
            .response_memo
            .lock()
            .ok()
            .and_then(|mut memo| memo.get(mode, &parsed.request));
        if let Some(output) = cached {
            let version = request.http_version().to_string();
            let created = unix_timestamp();
            let prefix = match kind {
                StreamKind::Completion => "cmpl-",
                StreamKind::Chat => "chatcmpl-",
            };
            let id = response_id(prefix, created);
            let identity = StreamIdentity {
                id: &id,
                created,
                model: &self.shared.config.model_id,
            };
            let mut writer = request.into_writer();
            return stream_cached_output_to_writer(
                &output,
                kind,
                parsed.include_usage,
                &identity,
                &version,
                &mut writer,
            );
        }
        let mut target = match self.shared.target.lock() {
            Ok(target) => target,
            Err(_) => {
                return request.respond(
                    ApiFailure::new(
                        500,
                        "server_error",
                        "authoritative target owner is unavailable",
                    )
                    .into_response()
                    .into_tiny_http(),
                );
            }
        };
        let version = request.http_version().to_string();
        let created = unix_timestamp();
        let prefix = match kind {
            StreamKind::Completion => "cmpl-",
            StreamKind::Chat => "chatcmpl-",
        };
        let id = response_id(prefix, created);
        let identity = StreamIdentity {
            id: &id,
            created,
            model: &self.shared.config.model_id,
        };
        let mut writer = request.into_writer();
        let generated = stream_target_to_writer(
            &mut *target,
            &parsed.request,
            kind,
            parsed.include_usage,
            &identity,
            &version,
            &mut writer,
            client,
        );
        if let Ok(output) = &generated
            && let Ok(mut memo) = self.shared.response_memo.lock()
        {
            memo.put(mode, &parsed.request, output);
        }
        generated.map(|_| ())
    }
}

/// Bound `tiny_http` listener plus the model-independent OpenAI service.
pub struct OpenAiHttpServer<T> {
    listener: Server,
    service: OpenAiService<T>,
}

impl<T: AuthoritativeTarget> OpenAiHttpServer<T> {
    pub fn bind<A: ToSocketAddrs>(
        address: A,
        target: T,
        config: ServerConfig,
    ) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        let service = OpenAiService::new(target, config)
            .map_err(|error| -> Box<dyn Error + Send + Sync> { Box::new(error) })?;
        let listener = Server::http(address)?;
        Ok(Self { listener, service })
    }

    pub fn from_listener(listener: Server, service: OpenAiService<T>) -> Self {
        Self { listener, service }
    }

    pub fn listener(&self) -> &Server {
        &self.listener
    }

    pub fn service(&self) -> &OpenAiService<T> {
        &self.service
    }

    pub fn serve_forever(&self) -> io::Result<()>
    where
        T: 'static,
    {
        let listener_port = listener_tcp_port(&self.listener);
        for request in self.listener.incoming_requests() {
            if !request_requires_generation_permit(&request) {
                if let Err(error) = self.service.handle_admitted_request(request, None) {
                    eprintln!("[serve] response aborted safely: {error}");
                }
                continue;
            }

            let queue_slots = self.service.shared.config.generation_queue_slots;
            match self.service.shared.generation_gate.admit(queue_slots) {
                Admission::Busy => {
                    if let Err(error) = respond_busy(request) {
                        eprintln!("[serve] busy response aborted safely: {error}");
                    }
                }
                // There can be at most one live generation worker because the
                // permit is reserved in the listener thread before spawning,
                // and at most `queue_slots` waiting workers because tickets
                // are too. Keeping the listener free is what lets the next
                // client receive its 429 or waiting slot instead of sitting
                // in the socket queue for the duration of generation.
                Admission::Permit(permit) => {
                    let service = self.service.clone();
                    let client = client_presence_for(&request, listener_port);
                    let _worker = thread::spawn(move || {
                        if let Err(error) = service.respond_generation(request, permit, &client) {
                            // A disconnected client or one failed target
                            // request must not take down the listener. The
                            // streaming transaction has already received
                            // `Aborted` before this error escapes.
                            eprintln!("[serve] response aborted safely: {error}");
                        }
                    });
                }
                Admission::Queued(ticket) => {
                    let service = self.service.clone();
                    let client = client_presence_for(&request, listener_port);
                    let _worker = thread::spawn(move || {
                        let outcome = service.respond_queued_generation(request, ticket, &client);
                        if let Err(error) = outcome {
                            eprintln!("[serve] queued response aborted safely: {error}");
                        }
                    });
                }
            }
        }
        Ok(())
    }

    pub fn serve_one(&self) -> io::Result<()> {
        self.service.handle_request(self.listener.recv()?)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum HttpMethod {
    Get,
    Post,
    Other,
}

impl From<&Method> for HttpMethod {
    fn from(method: &Method) -> Self {
        match method {
            Method::Get => Self::Get,
            Method::Post => Self::Post,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum StreamKind {
    Completion,
    Chat,
}

impl StreamKind {
    fn for_path(path: &str) -> Option<Self> {
        match path {
            "/v1/completions" | "/completions" => Some(Self::Completion),
            "/v1/chat/completions" | "/chat/completions" => Some(Self::Chat),
            _ => None,
        }
    }

    fn response_mode(self) -> ResponseMode {
        match self {
            Self::Completion => ResponseMode::Completion,
            Self::Chat => ResponseMode::Chat,
        }
    }
}

fn request_path(request: &Request) -> String {
    request
        .url()
        .split_once('?')
        .map_or(request.url(), |(path, _)| path)
        .to_owned()
}

fn request_requires_generation_permit(request: &Request) -> bool {
    request.method() == &Method::Post && StreamKind::for_path(&request_path(request)).is_some()
}

/// Consume a request whose client has provably disconnected without writing
/// tiny_http's automatic 500: extracting the response writer marks the
/// request answered, and dropping that writer closes the connection quietly.
fn abandon_unanswered(request: Request) {
    drop(request.into_writer());
}

fn listener_tcp_port(listener: &Server) -> Option<u16> {
    listener.server_addr().to_ip().map(|address| address.port())
}

/// Build the liveness probe for one accepted generation request.
///
/// A request whose connection cannot be located — a fabricated test request,
/// a Unix-socket listener, or a platform without a readable descriptor
/// table — degrades to an assumed-present client, restoring the previous
/// generate-to-completion behavior rather than risking a false disconnect.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn client_presence_for(request: &Request, listener_port: Option<u16>) -> ClientPresence {
    let (Some(port), Some(peer)) = (listener_port, request.remote_addr().copied()) else {
        return ClientPresence::assumed_present();
    };
    match client_socket::capture(peer, port) {
        Some(socket) => ClientPresence::from_probe(Box::new(move || socket.provably_disconnected())),
        None => ClientPresence::assumed_present(),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn client_presence_for(_request: &Request, _listener_port: Option<u16>) -> ClientPresence {
    ClientPresence::assumed_present()
}

/// Peer-socket liveness observation for `tiny_http` connections.
///
/// `tiny_http` deliberately hides its `TcpStream`, so the server locates the
/// accepted connection's descriptor in this process's own descriptor table by
/// matching the request's exact TCP 4-tuple — `getpeername` equal to the
/// request's remote address and `getsockname` on the listener's port — and
/// duplicates it. Matching both endpoints means a live connection can never
/// select the wrong descriptor: one live 4-tuple maps to one socket. A scan
/// miss degrades to an assumed-present client, never to a false disconnect.
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod client_socket {
    use std::fs;
    use std::net::{IpAddr, SocketAddr};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

    #[cfg(target_os = "macos")]
    const DESCRIPTOR_TABLE: &str = "/dev/fd";
    #[cfg(target_os = "linux")]
    const DESCRIPTOR_TABLE: &str = "/proc/self/fd";

    pub(super) struct ClientSocket {
        connection: OwnedFd,
    }

    impl ClientSocket {
        /// True only on positive evidence that the client is gone.
        ///
        /// `recv(MSG_PEEK)` cannot be the signal here: `tiny_http` shuts down
        /// its read half once it knows a connection carries no further
        /// requests, and a local read shutdown makes `recv` report EOF while
        /// the peer is still waiting for its response. The kernel's TCP
        /// state ignores local API-level shutdown: an alive, waiting client
        /// stays ESTABLISHED, while a departed one has advanced to
        /// CLOSE-WAIT (its FIN arrived) or CLOSED (reset). Nothing is ever
        /// read from the stream, so pipelined bytes stay untouched, and an
        /// unanswerable state query degrades to "present" rather than
        /// abandoning a live client. A client that half-closes its send side
        /// yet keeps reading is indistinguishable from a departed one and is
        /// treated as gone; HTTP clients do not do this mid-response.
        pub(super) fn provably_disconnected(&self) -> bool {
            match tcp_state(self.connection.as_raw_fd()) {
                Some(state) => state != TCP_STATE_ESTABLISHED,
                None => false,
            }
        }
    }

    /// `TCPS_ESTABLISHED` from xnu's `netinet/tcp_fsm.h`.
    #[cfg(target_os = "macos")]
    const TCP_STATE_ESTABLISHED: u8 = 4;
    /// `TCP_ESTABLISHED` from the Linux kernel's `include/net/tcp_states.h`.
    #[cfg(target_os = "linux")]
    const TCP_STATE_ESTABLISHED: u8 = 1;

    #[cfg(target_os = "macos")]
    fn tcp_state(fd: RawFd) -> Option<u8> {
        // SAFETY: an all-zero tcp_connection_info is a valid out-parameter
        // for this plain-data kernel report.
        let mut info: libc::tcp_connection_info = unsafe { std::mem::zeroed() };
        let mut length = std::mem::size_of::<libc::tcp_connection_info>() as libc::socklen_t;
        // SAFETY: `info` and `length` describe one writable, properly sized
        // report buffer owned by this frame.
        let outcome = unsafe {
            libc::getsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_CONNECTION_INFO,
                std::ptr::from_mut(&mut info).cast(),
                &mut length,
            )
        };
        (outcome == 0).then_some(info.tcpi_state)
    }

    #[cfg(target_os = "linux")]
    fn tcp_state(fd: RawFd) -> Option<u8> {
        // SAFETY: an all-zero tcp_info is a valid out-parameter for this
        // plain-data kernel report.
        let mut info: libc::tcp_info = unsafe { std::mem::zeroed() };
        let mut length = std::mem::size_of::<libc::tcp_info>() as libc::socklen_t;
        // SAFETY: `info` and `length` describe one writable, properly sized
        // report buffer owned by this frame.
        let outcome = unsafe {
            libc::getsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_INFO,
                std::ptr::from_mut(&mut info).cast(),
                &mut length,
            )
        };
        (outcome == 0).then_some(info.tcpi_state)
    }

    pub(super) fn capture(peer: SocketAddr, listener_port: u16) -> Option<ClientSocket> {
        let peer = canonical(peer);
        for entry in fs::read_dir(DESCRIPTOR_TABLE).ok()?.flatten() {
            let Some(fd) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<RawFd>().ok())
            else {
                continue;
            };
            if socket_peer(fd).map(canonical) != Some(peer)
                || socket_local_port(fd) != Some(listener_port)
            {
                continue;
            }
            let duplicate = duplicate_descriptor(fd)?;
            // Re-verify on the duplicate: the matched descriptor number could
            // have been closed and reused between the scan and the dup. The
            // duplicate's identity can no longer change underneath us.
            if socket_peer(duplicate.as_raw_fd()).map(canonical) == Some(peer)
                && socket_local_port(duplicate.as_raw_fd()) == Some(listener_port)
            {
                return Some(ClientSocket {
                    connection: duplicate,
                });
            }
        }
        None
    }

    fn duplicate_descriptor(fd: RawFd) -> Option<OwnedFd> {
        // SAFETY: fcntl reads only its arguments; a non-negative return is a
        // fresh descriptor this process owns exclusively.
        let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
        if duplicate < 0 {
            return None;
        }
        // SAFETY: `duplicate` was just returned open by fcntl and is owned by
        // nobody else; OwnedFd takes over its close.
        Some(unsafe { OwnedFd::from_raw_fd(duplicate) })
    }

    /// Dual-stack listeners report IPv4 peers as v4-mapped IPv6 addresses;
    /// compare both sides through the canonical form.
    fn canonical(address: SocketAddr) -> (IpAddr, u16) {
        (address.ip().to_canonical(), address.port())
    }

    fn socket_peer(fd: RawFd) -> Option<SocketAddr> {
        // SAFETY: an all-zero sockaddr_storage is a valid initial value for a
        // kernel-filled out-parameter of this plain-data type.
        let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut length = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        // SAFETY: `storage` and `length` describe one writable, properly
        // sized address buffer owned by this frame.
        let outcome =
            unsafe { libc::getpeername(fd, std::ptr::from_mut(&mut storage).cast(), &mut length) };
        if outcome != 0 {
            return None;
        }
        decode_socket_address(&storage)
    }

    fn socket_local_port(fd: RawFd) -> Option<u16> {
        // SAFETY: as in `socket_peer`; zeroed storage is a valid sockaddr
        // out-parameter.
        let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
        let mut length = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        // SAFETY: `storage` and `length` describe one writable, properly
        // sized address buffer owned by this frame.
        let outcome =
            unsafe { libc::getsockname(fd, std::ptr::from_mut(&mut storage).cast(), &mut length) };
        if outcome != 0 {
            return None;
        }
        decode_socket_address(&storage).map(|address| address.port())
    }

    fn decode_socket_address(storage: &libc::sockaddr_storage) -> Option<SocketAddr> {
        match libc::c_int::from(storage.ss_family) {
            libc::AF_INET => {
                // SAFETY: the kernel reported AF_INET, so the storage prefix
                // is an initialized sockaddr_in, a plain-data type.
                let source = unsafe { &*std::ptr::from_ref(storage).cast::<libc::sockaddr_in>() };
                let ip = std::net::Ipv4Addr::from(u32::from_be(source.sin_addr.s_addr));
                Some(SocketAddr::from((ip, u16::from_be(source.sin_port))))
            }
            libc::AF_INET6 => {
                // SAFETY: the kernel reported AF_INET6, so the storage prefix
                // is an initialized sockaddr_in6, a plain-data type.
                let source = unsafe { &*std::ptr::from_ref(storage).cast::<libc::sockaddr_in6>() };
                let ip = std::net::Ipv6Addr::from(source.sin6_addr.s6_addr);
                Some(SocketAddr::from((ip, u16::from_be(source.sin6_port))))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct StreamIdentity<'a> {
    id: &'a str,
    created: u64,
    model: &'a str,
}

struct SsePublisher<'a, W> {
    writer: &'a mut W,
    kind: StreamKind,
    identity: StreamIdentity<'a>,
    include_usage: bool,
}

impl<W: Write> SsePublisher<'_, W> {
    fn begin(&mut self) -> io::Result<()> {
        if self.kind != StreamKind::Chat {
            return Ok(());
        }
        let mut frame = json!({
            "id": self.identity.id,
            "object": "chat.completion.chunk",
            "created": self.identity.created,
            "model": self.identity.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "",
                },
                "finish_reason": Value::Null,
            }],
        });
        insert_stream_usage_placeholder(&mut frame, self.include_usage);
        write_sse_json(self.writer, &frame)
    }

    fn finish(&mut self, summary: TargetStreamSummary) -> io::Result<()> {
        let mut frame = match self.kind {
            StreamKind::Completion => json!({
                "id": self.identity.id,
                "object": "text_completion",
                "created": self.identity.created,
                "model": self.identity.model,
                "choices": [{
                    "index": 0,
                    "text": "",
                    "finish_reason": summary.finish_reason().as_openai_str(),
                }],
            }),
            StreamKind::Chat => json!({
                "id": self.identity.id,
                "object": "chat.completion.chunk",
                "created": self.identity.created,
                "model": self.identity.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": summary.finish_reason().as_openai_str(),
                }],
            }),
        };
        insert_stream_usage_placeholder(&mut frame, self.include_usage);
        write_sse_json(self.writer, &frame)?;

        if self.include_usage
            && let Some(usage) = summary.usage()
        {
            write_sse_json(
                self.writer,
                &json!({
                    "id": self.identity.id,
                    "object": match self.kind {
                        StreamKind::Completion => "text_completion",
                        StreamKind::Chat => "chat.completion.chunk",
                    },
                    "created": self.identity.created,
                    "model": self.identity.model,
                    "choices": [],
                    "usage": usage_json(usage),
                }),
            )?;
        }
        self.writer.write_all(b"data: [DONE]\n\n")?;
        self.writer.flush()
    }

    fn write_error(&mut self, message: &str) -> io::Result<()> {
        write_sse_json(
            self.writer,
            &json!({
                "error": {
                    "message": message,
                    "type": "server_error",
                    "param": Value::Null,
                    "code": "server_error",
                }
            }),
        )
    }
}

impl<W: Write> TargetDeltaSink for SsePublisher<'_, W> {
    fn publish_target_delta(&mut self, delta: TargetDelta) -> io::Result<()> {
        if delta.text().is_empty() {
            return Ok(());
        }
        let mut frame = match self.kind {
            StreamKind::Completion => {
                // Raw completions have one text channel. A well-formed raw
                // target adapter will not produce structured reasoning; if a
                // shared adapter does, keep it private rather than silently
                // changing raw completion text relative to non-streaming.
                if delta.channel() == TargetDeltaChannel::ReasoningContent {
                    return Ok(());
                }
                json!({
                    "id": self.identity.id,
                    "object": "text_completion",
                    "created": self.identity.created,
                    "model": self.identity.model,
                    "choices": [{
                        "index": 0,
                        "text": delta.text(),
                        "finish_reason": Value::Null,
                    }],
                })
            }
            StreamKind::Chat => {
                let channel = match delta.channel() {
                    TargetDeltaChannel::Content => "content",
                    TargetDeltaChannel::ReasoningContent => "reasoning_content",
                };
                let mut fields = Map::new();
                fields.insert(channel.to_owned(), Value::String(delta.text().to_owned()));
                json!({
                    "id": self.identity.id,
                    "object": "chat.completion.chunk",
                    "created": self.identity.created,
                    "model": self.identity.model,
                    "choices": [{
                        "index": 0,
                        "delta": Value::Object(fields),
                        "finish_reason": Value::Null,
                    }],
                })
            }
        };
        insert_stream_usage_placeholder(&mut frame, self.include_usage);
        write_sse_json(self.writer, &frame)
    }
}

struct CapturingSseSink<'publisher, 'writer, W> {
    publisher: &'publisher mut SsePublisher<'writer, W>,
    content: String,
    reasoning: String,
}

impl<'publisher, 'writer, W> CapturingSseSink<'publisher, 'writer, W> {
    fn new(publisher: &'publisher mut SsePublisher<'writer, W>) -> Self {
        Self {
            publisher,
            content: String::new(),
            reasoning: String::new(),
        }
    }

    fn into_output(self, summary: TargetStreamSummary) -> TargetOutput {
        let mut output = TargetOutput::target_verified(self.content, summary.finish_reason());
        if !self.reasoning.is_empty() {
            output = output.with_reasoning_content(self.reasoning);
        }
        if let Some(usage) = summary.usage() {
            output = output.with_usage(usage);
        }
        output
    }
}

impl<W: Write> TargetDeltaSink for CapturingSseSink<'_, '_, W> {
    fn publish_target_delta(&mut self, delta: TargetDelta) -> io::Result<()> {
        let channel = delta.channel();
        let text = delta.text().to_owned();
        self.publisher.publish_target_delta(delta)?;
        match channel {
            TargetDeltaChannel::Content => self.content.push_str(&text),
            TargetDeltaChannel::ReasoningContent if self.publisher.kind == StreamKind::Chat => {
                self.reasoning.push_str(&text);
            }
            TargetDeltaChannel::ReasoningContent => {}
        }
        Ok(())
    }
}

struct StreamFinalizer<'a, T: AuthoritativeTarget> {
    target: &'a mut T,
    settled: bool,
}

impl<'a, T: AuthoritativeTarget> StreamFinalizer<'a, T> {
    fn new(target: &'a mut T) -> Self {
        Self {
            target,
            settled: false,
        }
    }

    fn target(&mut self) -> &mut T {
        self.target
    }

    fn complete(mut self) {
        self.settled = true;
        self.target
            .finish_target_stream(StreamPublication::Complete);
    }
}

impl<T: AuthoritativeTarget> Drop for StreamFinalizer<'_, T> {
    fn drop(&mut self) {
        if !self.settled {
            self.target.finish_target_stream(StreamPublication::Aborted);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the SSE boundary keeps identity, transport, and client-liveness controls explicit"
)]
fn stream_target_to_writer<T: AuthoritativeTarget, W: Write>(
    target: &mut T,
    request: &TargetRequest,
    kind: StreamKind,
    include_usage: bool,
    identity: &StreamIdentity<'_>,
    http_version: &str,
    writer: &mut W,
    client: &ClientPresence,
) -> io::Result<TargetOutput> {
    write_stream_headers(writer, http_version)?;
    let mut publisher = SsePublisher {
        writer,
        kind,
        identity: *identity,
        include_usage,
    };
    publisher.begin()?;
    let mut finalizer = StreamFinalizer::new(target);
    let mut capture = CapturingSseSink::new(&mut publisher);
    let summary = match finalizer
        .target()
        .generate_target_stream(request, &mut capture, client)
    {
        Ok(summary) => summary,
        Err(StreamGenerationError::Publication(error)) => return Err(error),
        Err(StreamGenerationError::Target(error)) => {
            capture.publisher.write_error(&error.to_string())?;
            return Err(io::Error::other(error));
        }
    };
    let output = capture.into_output(summary);
    publisher.finish(summary)?;
    finalizer.complete();
    Ok(output)
}

fn stream_cached_output_to_writer<W: Write>(
    output: &TargetOutput,
    kind: StreamKind,
    include_usage: bool,
    identity: &StreamIdentity<'_>,
    http_version: &str,
    writer: &mut W,
) -> io::Result<()> {
    write_stream_headers(writer, http_version)?;
    let mut publisher = SsePublisher {
        writer,
        kind,
        identity: *identity,
        include_usage,
    };
    publisher.begin()?;
    if kind == StreamKind::Chat
        && let Some(reasoning) = output.reasoning_content().filter(|text| !text.is_empty())
    {
        publisher.publish_target_delta(TargetDelta::target_verified_reasoning(reasoning))?;
    }
    if !output.text().is_empty() {
        publisher.publish_target_delta(TargetDelta::target_verified_content(output.text()))?;
    }
    let mut summary = TargetStreamSummary::target_verified(output.finish_reason());
    if let Some(usage) = output.usage() {
        summary = summary.with_usage(usage);
    }
    publisher.finish(summary)
}

fn write_stream_headers(writer: &mut impl Write, http_version: &str) -> io::Result<()> {
    write!(
        writer,
        "HTTP/{http_version} 200 OK\r\n\
         Content-Type: text/event-stream; charset=utf-8\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n\
         X-Accel-Buffering: no\r\n\
         X-Content-Type-Options: nosniff\r\n\r\n"
    )?;
    writer.flush()
}

fn write_sse_json(writer: &mut impl Write, value: &Value) -> io::Result<()> {
    writer.write_all(b"data: ")?;
    serde_json::to_writer(&mut *writer, value).map_err(io::Error::other)?;
    writer.write_all(b"\n\n")?;
    writer.flush()
}

fn insert_stream_usage_placeholder(frame: &mut Value, include_usage: bool) {
    if include_usage {
        frame
            .as_object_mut()
            .expect("stream frame object")
            .insert("usage".to_owned(), Value::Null);
    }
}

fn usage_json(usage: TokenUsage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens.saturating_add(usage.completion_tokens),
    })
}

struct WireResponse {
    status: u16,
    body: Vec<u8>,
    close_connection: bool,
    retry_after_seconds: Option<u64>,
    settle_target: bool,
}

impl WireResponse {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            // Serializing a serde_json value to a vector is infallible in
            // practice; a failure can only be an allocator/write failure that
            // Rust cannot recover from here.
            body: serde_json::to_vec(&value).expect("JSON value serialization"),
            close_connection: false,
            retry_after_seconds: None,
            settle_target: false,
        }
    }

    fn target_json(status: u16, value: Value, settle_target: bool) -> Self {
        let mut response = Self::json(status, value);
        response.settle_target = settle_target;
        response
    }

    fn into_tiny_http(self) -> Response<std::io::Cursor<Vec<u8>>> {
        let mut response = Response::from_data(self.body)
            .with_status_code(StatusCode(self.status))
            .with_header(
                Header::from_bytes("Content-Type", "application/json; charset=utf-8")
                    .expect("static HTTP header"),
            )
            .with_header(
                Header::from_bytes("X-Content-Type-Options", "nosniff")
                    .expect("static HTTP header"),
            );
        if self.close_connection {
            // A rejected oversized request can still have unread bytes.  Do
            // not let them become the apparent start of another request on a
            // persistent connection.
            response
                .add_header(Header::from_bytes("Connection", "close").expect("static HTTP header"));
        }
        if let Some(seconds) = self.retry_after_seconds {
            response.add_header(
                Header::from_bytes("Retry-After", seconds.to_string())
                    .expect("decimal retry delay is a valid HTTP header"),
            );
        }
        response
    }
}

#[derive(Debug)]
struct ApiFailure {
    status: u16,
    kind: &'static str,
    message: String,
}

impl ApiFailure {
    fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request_error", message)
    }

    fn busy() -> Self {
        Self::new(
            429,
            "server_busy",
            "Deltafin is already serving one authoritative generation; retry this request later",
        )
    }

    fn into_response(self) -> WireResponse {
        let mut response = WireResponse::json(
            self.status,
            json!({
                "error": {
                    "message": self.message,
                    "type": self.kind,
                    "param": Value::Null,
                    "code": self.kind,
                }
            }),
        );
        response.close_connection = self.status == 413;
        response
    }
}

fn busy_response() -> WireResponse {
    let mut response = ApiFailure::busy().into_response();
    // Admission occurs before reading a potentially huge request body. Close
    // this connection so unread bytes cannot be mistaken for another request.
    response.close_connection = true;
    response.retry_after_seconds = Some(1);
    response
}

fn respond_busy(request: Request) -> io::Result<()> {
    let http_version = request.http_version().to_string();
    let mut writer = request.into_writer();
    write_busy_response(&mut writer, &http_version)
}

fn write_busy_response(writer: &mut impl Write, http_version: &str) -> io::Result<()> {
    let response = busy_response();
    write!(
        writer,
        "HTTP/{http_version} 429 Too Many Requests\r\n\
         Content-Type: application/json; charset=utf-8\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Retry-After: 1\r\n\
         Connection: close\r\n\
         Content-Length: {}\r\n\r\n",
        response.body.len()
    )?;
    writer.write_all(&response.body)?;
    writer.flush()
}

fn read_bounded_body(request: &mut Request, limit: usize) -> Result<Vec<u8>, ApiFailure> {
    if request.body_length().is_some_and(|length| length > limit) {
        return Err(ApiFailure::new(
            413,
            "request_too_large",
            format!("request body exceeds the {limit}-byte limit"),
        ));
    }
    let maximum_read = limit.saturating_add(1) as u64;
    let mut body = Vec::with_capacity(request.body_length().unwrap_or(0).min(limit));
    request
        .as_reader()
        .take(maximum_read)
        .read_to_end(&mut body)
        .map_err(|_| ApiFailure::invalid("could not read request body"))?;
    if body.len() > limit {
        return Err(ApiFailure::new(
            413,
            "request_too_large",
            format!("request body exceeds the {limit}-byte limit"),
        ));
    }
    Ok(body)
}

struct ParsedObject(Map<String, Value>);

struct ParsedTargetRequest {
    request: TargetRequest,
    include_usage: bool,
}

struct GeneratedOutput {
    output: TargetOutput,
    target_invoked: bool,
}

impl ParsedObject {
    fn parse(body: &[u8]) -> Result<Self, ApiFailure> {
        let value: Value = serde_json::from_slice(body)
            .map_err(|_| ApiFailure::invalid("request body must be valid JSON"))?;
        value
            .as_object()
            .cloned()
            .map(Self)
            .ok_or_else(|| ApiFailure::invalid("request body must be a JSON object"))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RequestEndpoint {
    Completion,
    Chat,
}

impl RequestEndpoint {
    fn label(self) -> &'static str {
        match self {
            Self::Completion => "completions",
            Self::Chat => "chat/completions",
        }
    }

    fn admits(self, field: &str) -> bool {
        const COMMON: &[&str] = &[
            "model",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "n",
            "best_of",
            "temperature",
            "top_p",
            "user",
            "metadata",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "seed",
            "stop",
        ];
        if COMMON.contains(&field) {
            return true;
        }
        match self {
            Self::Completion => ["prompt", "echo", "logprobs", "suffix"].contains(&field),
            Self::Chat => [
                "messages",
                "logprobs",
                "top_logprobs",
                "response_format",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "modalities",
                "audio",
                "reasoning_effort",
                "prediction",
                "service_tier",
                "store",
            ]
            .contains(&field),
        }
    }
}

/// Validate every admitted top-level field before an `AuthoritativeTarget`
/// can be entered. The small number of ignored fields below are either request
/// metadata, exact no-ops, or the deliberately documented greedy-compatibility
/// aliases (`temperature`/`top_p`). Everything else fails closed.
fn validate_request_contract(
    object: &Map<String, Value>,
    endpoint: RequestEndpoint,
    configured_model: &str,
) -> Result<(), ApiFailure> {
    if let Some(field) = object.keys().find(|field| !endpoint.admits(field)) {
        return Err(ApiFailure::invalid(format!(
            "`{field}` is not supported by Deltafin's native /v1/{} endpoint; refusing to silently ignore an unknown or generation-semantic field",
            endpoint.label()
        )));
    }

    validate_model(object.get("model"), configured_model)?;
    validate_max_token_aliases(object)?;
    validate_greedy_compatibility_field(object.get("temperature"), "temperature", 0.0, 2.0)?;
    validate_greedy_compatibility_field(object.get("top_p"), "top_p", 0.0, 1.0)?;
    validate_optional_string(object.get("user"), "user")?;
    validate_optional_object(object.get("metadata"), "metadata")?;
    validate_exact_zero(object.get("frequency_penalty"), "frequency_penalty")?;
    validate_exact_zero(object.get("presence_penalty"), "presence_penalty")?;
    validate_empty_object(object.get("logit_bias"), "logit_bias")?;
    validate_null_or_empty_array(object.get("stop"), "stop")?;
    validate_null_only(object.get("seed"), "seed")?;

    match endpoint {
        RequestEndpoint::Completion => {
            validate_false(object.get("echo"), "echo")?;
            validate_null_only(object.get("logprobs"), "logprobs")?;
            validate_null_or_empty_string(object.get("suffix"), "suffix")?;
        }
        RequestEndpoint::Chat => {
            validate_false(object.get("logprobs"), "logprobs")?;
            validate_null_only(object.get("top_logprobs"), "top_logprobs")?;
            validate_text_response_format(object.get("response_format"))?;
            validate_null_or_empty_array(object.get("tools"), "tools")?;
            validate_null_only(object.get("tool_choice"), "tool_choice")?;
            validate_optional_bool(object.get("parallel_tool_calls"), "parallel_tool_calls")?;
            validate_text_modality(object.get("modalities"))?;
            validate_null_only(object.get("audio"), "audio")?;
            parse_reasoning_effort(object.get("reasoning_effort"))?;
            validate_null_only(object.get("prediction"), "prediction")?;
            validate_null_only(object.get("service_tier"), "service_tier")?;
            validate_false(object.get("store"), "store")?;
        }
    }
    Ok(())
}

fn validate_model(value: Option<&Value>, configured_model: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(model)) if model == configured_model => Ok(()),
        Some(Value::String(model)) => Err(ApiFailure::invalid(format!(
            "requested model `{model}` is unavailable; this server exposes only `{configured_model}`"
        ))),
        Some(_) => Err(ApiFailure::invalid("`model` must be a JSON string")),
    }
}

fn validate_max_token_aliases(object: &Map<String, Value>) -> Result<(), ApiFailure> {
    if object
        .get("max_tokens")
        .is_some_and(|value| !value.is_null())
        && object
            .get("max_completion_tokens")
            .is_some_and(|value| !value.is_null())
    {
        return Err(ApiFailure::invalid(
            "send only one of `max_tokens` and `max_completion_tokens`",
        ));
    }
    Ok(())
}

/// Deltafin intentionally remains greedy. These two fields are parsed and
/// range-checked solely because many OpenAI clients always serialize them;
/// tests pin that they do not alter the target request or memo identity.
fn validate_greedy_compatibility_field(
    value: Option<&Value>,
    field: &str,
    minimum: f64,
    maximum: f64,
) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(value) => match value.as_f64() {
            Some(number) if (minimum..=maximum).contains(&number) => Ok(()),
            _ => Err(ApiFailure::invalid(format!(
                "`{field}` must be a finite JSON number in {minimum}..={maximum}; Deltafin validates it for client compatibility but decodes greedily"
            ))),
        },
    }
}

fn validate_optional_string(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null | Value::String(_)) => Ok(()),
        Some(_) => Err(ApiFailure::invalid(format!(
            "`{field}` must be a JSON string"
        ))),
    }
}

fn validate_optional_object(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null | Value::Object(_)) => Ok(()),
        Some(_) => Err(ApiFailure::invalid(format!(
            "`{field}` must be a JSON object"
        ))),
    }
}

fn validate_optional_bool(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null | Value::Bool(_)) => Ok(()),
        Some(_) => Err(ApiFailure::invalid(format!("`{field}` must be a boolean"))),
    }
}

fn validate_exact_zero(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(value) if value.as_f64() == Some(0.0) => Ok(()),
        Some(_) => Err(unsupported_non_default(field)),
    }
}

fn validate_empty_object(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(object)) if object.is_empty() => Ok(()),
        Some(_) => Err(unsupported_non_default(field)),
    }
}

fn validate_null_or_empty_array(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(values)) if values.is_empty() => Ok(()),
        Some(_) => Err(unsupported_non_default(field)),
    }
}

fn validate_null_or_empty_string(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(text)) if text.is_empty() => Ok(()),
        Some(_) => Err(unsupported_non_default(field)),
    }
}

fn validate_null_only(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(_) => Err(unsupported_non_default(field)),
    }
}

fn validate_false(value: Option<&Value>, field: &str) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null | Value::Bool(false)) => Ok(()),
        Some(_) => Err(unsupported_non_default(field)),
    }
}

/// Chat-only `reasoning_effort`: absent/null defers to the engine default,
/// and a string must normalize to one of the chat template's accepted levels.
fn parse_reasoning_effort(value: Option<&Value>) -> Result<Option<String>, ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(raw)) => {
            let effort = raw.trim().to_ascii_lowercase();
            if matches!(effort.as_str(), "low" | "high" | "max") {
                Ok(Some(effort))
            } else {
                Err(ApiFailure::invalid(
                    "`reasoning_effort` must be `low`, `high`, or `max`",
                ))
            }
        }
        Some(_) => Err(ApiFailure::invalid(
            "`reasoning_effort` must be a JSON string: `low`, `high`, or `max`",
        )),
    }
}

fn unsupported_non_default(field: &str) -> ApiFailure {
    ApiFailure::invalid(format!(
        "non-default `{field}` semantics are not implemented by native Deltafin; refusing to ignore the request"
    ))
}

fn validate_text_response_format(value: Option<&Value>) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(object))
            if object.len() == 1 && object.get("type").and_then(Value::as_str) == Some("text") =>
        {
            Ok(())
        }
        Some(_) => Err(unsupported_non_default("response_format")),
    }
}

fn validate_text_modality(value: Option<&Value>) -> Result<(), ApiFailure> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(modalities))
            if modalities.len() == 1 && modalities[0].as_str() == Some("text") =>
        {
            Ok(())
        }
        Some(_) => Err(unsupported_non_default("modalities")),
    }
}

fn validate_chat_message_contract(
    source: &Map<String, Value>,
    role: &str,
) -> Result<(), ApiFailure> {
    let allowed: &[&str] = match role {
        "system" | "user" => &["role", "content", "name"],
        "assistant" => &["role", "content", "name", "reasoning_content", "tool_calls"],
        "tool" => &["role", "content", "name", "tool", "tool_call_id", "id"],
        _ => {
            return Err(ApiFailure::invalid(format!(
                "unsupported chat message role `{role}`; native Deltafin supports system, user, assistant, and tool history"
            )));
        }
    };
    if let Some(field) = source
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(ApiFailure::invalid(format!(
            "chat message field `{field}` is not implemented for role `{role}`; refusing to silently ignore it"
        )));
    }
    validate_optional_string(source.get("name"), "message.name")?;
    if role != "assistant" && source.get("content").is_some_and(Value::is_null) {
        return Err(ApiFailure::invalid(format!(
            "`content` cannot be null for a `{role}` chat message"
        )));
    }
    if role == "assistant" {
        validate_optional_string(source.get("reasoning_content"), "message.reasoning_content")?;
        validate_historical_tool_calls(source.get("tool_calls"))?;
    }
    if role == "tool" {
        for field in ["tool", "tool_call_id", "id"] {
            validate_optional_string(source.get(field), &format!("message.{field}"))?;
        }
    }
    Ok(())
}

fn validate_historical_tool_calls(value: Option<&Value>) -> Result<(), ApiFailure> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let calls = value
        .as_array()
        .ok_or_else(|| ApiFailure::invalid("`message.tool_calls` must be an array"))?;
    for call in calls {
        let call = call
            .as_object()
            .ok_or_else(|| ApiFailure::invalid("every historical tool call must be an object"))?;
        if let Some(field) = call
            .keys()
            .find(|field| !["id", "type", "function", "index"].contains(&field.as_str()))
        {
            return Err(ApiFailure::invalid(format!(
                "historical tool-call field `{field}` is not implemented"
            )));
        }
        validate_optional_string(call.get("id"), "tool_call.id")?;
        match call.get("type") {
            None | Some(Value::Null) => {}
            Some(Value::String(kind)) if kind == "function" => {}
            Some(_) => {
                return Err(ApiFailure::invalid(
                    "historical `tool_call.type` must be `function`",
                ));
            }
        }
        if let Some(index) = call.get("index")
            && !index.is_null()
            && index.as_u64().is_none()
        {
            return Err(ApiFailure::invalid(
                "historical `tool_call.index` must be a non-negative integer",
            ));
        }
        let function = call
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ApiFailure::invalid("every historical tool call must contain a function object")
            })?;
        if let Some(field) = function
            .keys()
            .find(|field| !["name", "arguments"].contains(&field.as_str()))
        {
            return Err(ApiFailure::invalid(format!(
                "historical tool-call function field `{field}` is not implemented"
            )));
        }
        if function
            .get("name")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(ApiFailure::invalid(
                "every historical tool-call function needs a non-empty string `name`",
            ));
        }
        match function.get("arguments") {
            None | Some(Value::Null | Value::String(_) | Value::Object(_)) => {}
            Some(_) => {
                return Err(ApiFailure::invalid(
                    "historical tool-call `arguments` must be a JSON object or JSON object string",
                ));
            }
        }
    }
    Ok(())
}

fn body_requests_stream(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| value.get("stream").and_then(Value::as_bool))
        == Some(true)
}

fn validate_stream_mode(object: &Map<String, Value>, streaming: bool) -> Result<(), ApiFailure> {
    match (streaming, object.get("stream")) {
        (false, None | Some(Value::Null | Value::Bool(false))) => Ok(()),
        (false, Some(Value::Bool(true))) => Err(ApiFailure::invalid(
            "streaming requests must use the SSE response path",
        )),
        (true, Some(Value::Bool(true))) => Ok(()),
        (true, _) => Err(ApiFailure::invalid(
            "the SSE response path requires `stream: true`",
        )),
        (_, Some(_)) => Err(ApiFailure::invalid("`stream` must be a boolean")),
    }
}

fn parse_stream_usage(object: &Map<String, Value>, streaming: bool) -> Result<bool, ApiFailure> {
    if !streaming {
        return match object.get("stream_options") {
            None | Some(Value::Null) => Ok(false),
            Some(_) => Err(ApiFailure::invalid(
                "`stream_options` requires `stream: true`",
            )),
        };
    }
    let Some(options) = object.get("stream_options") else {
        return Ok(false);
    };
    if options.is_null() {
        return Ok(false);
    }
    let options = options
        .as_object()
        .ok_or_else(|| ApiFailure::invalid("`stream_options` must be a JSON object"))?;
    if let Some(field) = options
        .keys()
        .find(|field| field.as_str() != "include_usage")
    {
        return Err(ApiFailure::invalid(format!(
            "`stream_options.{field}` is not implemented; refusing to silently ignore it"
        )));
    }
    match options.get("include_usage") {
        None | Some(Value::Null | Value::Bool(false)) => Ok(false),
        Some(Value::Bool(true)) => Ok(true),
        Some(_) => Err(ApiFailure::invalid(
            "`stream_options.include_usage` must be a boolean",
        )),
    }
}

fn validate_single_choice(object: &Map<String, Value>) -> Result<(), ApiFailure> {
    for field in ["n", "best_of"] {
        match object.get(field) {
            None | Some(Value::Null) => {}
            Some(value) if value.as_u64() == Some(1) => {}
            Some(_) => {
                return Err(ApiFailure::invalid(format!(
                    "`{field}` must be 1; Deltafin has one authoritative target stream"
                )));
            }
        }
    }
    Ok(())
}

fn validate_text_only_chat_content(content: &Value) -> Result<(), ApiFailure> {
    let Value::Array(parts) = content else {
        return Ok(());
    };
    for part in parts {
        let object = part.as_object().ok_or_else(|| {
            ApiFailure::invalid("every chat content-array entry must be an object")
        })?;
        let part_type = object.get("type").and_then(Value::as_str);
        if part_type != Some("text") {
            return Err(ApiFailure::invalid(
                "native Deltafin production inference is text-only; image and other multimodal chat content require the K3 vision/projector path, which is not implemented",
            ));
        }
        if !object.get("text").is_some_and(Value::is_string) {
            return Err(ApiFailure::invalid(
                "a text chat content part must contain a string `text` field",
            ));
        }
        if let Some(field) = object
            .keys()
            .find(|field| !["type", "text"].contains(&field.as_str()))
        {
            return Err(ApiFailure::invalid(format!(
                "text chat content field `{field}` is not implemented; refusing to silently ignore it"
            )));
        }
    }
    Ok(())
}

fn parse_max_tokens(
    object: &Map<String, Value>,
    default: usize,
    limit: usize,
) -> Result<usize, ApiFailure> {
    let value = object
        .get("max_tokens")
        .filter(|value| !value.is_null())
        .or_else(|| {
            object
                .get("max_completion_tokens")
                .filter(|value| !value.is_null())
        });
    let requested = match value {
        None | Some(Value::Null) => default,
        Some(value) => {
            let count = value.as_u64().ok_or_else(|| {
                ApiFailure::invalid("`max_tokens` must be a non-negative integer")
            })?;
            if count == 0 {
                // Preserve the mature server's compatibility behavior: zero
                // is treated like an omitted maximum rather than generating
                // an empty response.
                default
            } else {
                usize::try_from(count).unwrap_or(usize::MAX)
            }
        }
    };
    Ok(requested.min(limit))
}

fn insert_usage(response: &mut Value, output: &TargetOutput) {
    let Some(usage) = output.usage() else {
        return;
    };
    response.as_object_mut().expect("response object").insert(
        "usage".to_owned(),
        json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens.saturating_add(usage.completion_tokens),
        }),
    );
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn response_id(prefix: &str, timestamp: u64) -> String {
    let sequence = RESPONSE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}{timestamp:08x}{sequence:012x}")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::error::DeltafinError;
    use crate::openai::{FinishReason, TokenUsage};

    /// Poll `condition` until it holds, panicking with `subject` on timeout.
    /// Generation admission and socket teardown are asynchronous by nature;
    /// bounded polling keeps these tests deterministic without fixed sleeps.
    fn wait_until(subject: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {subject}");
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[derive(Clone)]
    struct RecordingTarget {
        requests: Arc<Mutex<Vec<TargetRequest>>>,
        active: Arc<AtomicUsize>,
        maximum_active: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl RecordingTarget {
        fn new() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                active: Arc::new(AtomicUsize::new(0)),
                maximum_active: Arc::new(AtomicUsize::new(0)),
                delay: Duration::ZERO,
            }
        }
    }

    impl AuthoritativeTarget for RecordingTarget {
        fn generate_target(
            &mut self,
            request: &TargetRequest,
            _client: &ClientPresence,
        ) -> crate::error::Result<TargetOutput> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.maximum_active.fetch_max(active, Ordering::SeqCst);
            thread::sleep(self.delay);
            self.requests.lock().unwrap().push(request.clone());
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(
                TargetOutput::target_verified("K3 output", FinishReason::Stop).with_usage(
                    TokenUsage {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                    },
                ),
            )
        }
    }

    struct StreamingTarget {
        deltas: Vec<TargetDelta>,
        publications: Vec<StreamPublication>,
        fail_target: bool,
    }

    impl StreamingTarget {
        fn successful(deltas: Vec<TargetDelta>) -> Self {
            Self {
                deltas,
                publications: Vec::new(),
                fail_target: false,
            }
        }
    }

    impl AuthoritativeTarget for StreamingTarget {
        fn generate_target(
            &mut self,
            _request: &TargetRequest,
            _client: &ClientPresence,
        ) -> crate::error::Result<TargetOutput> {
            Ok(TargetOutput::target_verified(
                "non-streaming fallback",
                FinishReason::Stop,
            ))
        }

        fn generate_target_stream(
            &mut self,
            _request: &TargetRequest,
            sink: &mut dyn TargetDeltaSink,
            _client: &ClientPresence,
        ) -> std::result::Result<TargetStreamSummary, StreamGenerationError> {
            if self.fail_target {
                return Err(DeltafinError::new("target stream failed").into());
            }
            for delta in self.deltas.clone() {
                sink.publish_target_delta(delta)?;
            }
            Ok(
                TargetStreamSummary::target_verified(FinishReason::Stop).with_usage(TokenUsage {
                    prompt_tokens: 7,
                    completion_tokens: 2,
                }),
            )
        }

        fn finish_target_stream(&mut self, publication: StreamPublication) {
            self.publications.push(publication);
        }
    }

    fn completion_request() -> TargetRequest {
        TargetRequest {
            prompt: TargetPrompt::Completion("hello".to_owned()),
            max_new_tokens: 8,
            reasoning_effort: None,
        }
    }

    fn fixed_identity() -> StreamIdentity<'static> {
        StreamIdentity {
            id: "cmpl-test",
            created: 123,
            model: "deltafin-kimi-k3",
        }
    }

    fn sse_data_lines(wire: &[u8]) -> Vec<&str> {
        let wire = std::str::from_utf8(wire).unwrap();
        let (_, body) = wire.split_once("\r\n\r\n").unwrap();
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect()
    }

    fn service(target: RecordingTarget) -> OpenAiService<RecordingTarget> {
        OpenAiService::new(target, ServerConfig::default()).unwrap()
    }

    fn body(response: &WireResponse) -> Value {
        serde_json::from_slice(&response.body).unwrap()
    }

    #[test]
    fn identical_target_request_is_generated_once_and_replayed_from_native_memo() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let service = service(target);
        let exact = br#"{"prompt":"repeat","max_tokens":2}"#;
        assert_eq!(
            service
                .dispatch(HttpMethod::Post, "/v1/completions", exact)
                .status,
            200
        );
        assert_eq!(
            service
                .dispatch(HttpMethod::Post, "/v1/completions", exact)
                .status,
            200
        );
        assert_eq!(requests.lock().unwrap().len(), 1);

        // Wire formatting does not affect the validated target request.
        assert_eq!(
            service
                .dispatch(
                    HttpMethod::Post,
                    "/v1/completions",
                    br#"{"prompt": "repeat", "max_tokens": 2}"#,
                )
                .status,
            200
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn memo_hit_does_not_settle_a_target_transaction_that_never_ran() {
        struct SettlementTarget {
            generated: Arc<AtomicUsize>,
            settled: Arc<AtomicUsize>,
        }

        impl AuthoritativeTarget for SettlementTarget {
            fn generate_target(
                &mut self,
                _request: &TargetRequest,
                _client: &ClientPresence,
            ) -> crate::error::Result<TargetOutput> {
                self.generated.fetch_add(1, Ordering::SeqCst);
                Ok(TargetOutput::target_verified("exact", FinishReason::Stop))
            }

            fn finish_target_response(&mut self, _publication: StreamPublication) {
                self.settled.fetch_add(1, Ordering::SeqCst);
            }
        }

        let generated = Arc::new(AtomicUsize::new(0));
        let settled = Arc::new(AtomicUsize::new(0));
        let service = OpenAiService::new(
            SettlementTarget {
                generated: Arc::clone(&generated),
                settled: Arc::clone(&settled),
            },
            ServerConfig::default(),
        )
        .unwrap();
        for _ in 0..2 {
            let request: Request = tiny_http::TestRequest::new()
                .with_method(Method::Post)
                .with_path("/v1/completions")
                .with_body(r#"{"prompt":"repeat","max_tokens":2}"#)
                .into();
            service.handle_request(request).unwrap();
        }
        assert_eq!(generated.load(Ordering::SeqCst), 1);
        assert_eq!(settled.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn exact_stream_request_is_generated_once_and_replayed_as_sse() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let service = service(target);
        let wire_body = r#"{"prompt":"repeat","max_tokens":2,"stream":true}"#;
        for _ in 0..2 {
            let request: Request = tiny_http::TestRequest::new()
                .with_method(Method::Post)
                .with_path("/v1/completions")
                .with_body(wire_body)
                .into();
            service.handle_request(request).unwrap();
        }
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn memo_reuses_the_same_target_request_across_json_and_sse_transports() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let service = service(target);
        let ordinary: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/v1/completions")
            .with_body(r#"{"prompt":"repeat","max_tokens":2}"#)
            .into();
        service.handle_request(ordinary).unwrap();
        let streaming: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/v1/completions")
            .with_body(r#"{"stream":true,"max_tokens":2,"prompt":"repeat"}"#)
            .into();
        service.handle_request(streaming).unwrap();
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn models_route_does_not_enter_the_target() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(HttpMethod::Get, "/v1/models", b"");
        assert_eq!(response.status, 200);
        assert_eq!(body(&response)["data"][0]["id"], "deltafin-kimi-k3");
        assert!(requests.lock().unwrap().is_empty());
    }

    #[test]
    fn completion_is_validated_and_returned_as_target_text() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/completions",
            br#"{"prompt":"hello","max_tokens":12}"#,
        );
        assert_eq!(response.status, 200);
        assert_eq!(body(&response)["choices"][0]["text"], "K3 output");
        assert_eq!(body(&response)["usage"]["total_tokens"], 5);
        assert_eq!(
            requests.lock().unwrap().as_slice(),
            &[TargetRequest {
                prompt: TargetPrompt::Completion("hello".to_owned()),
                max_new_tokens: 12,
                reasoning_effort: None,
            }]
        );
    }

    #[test]
    fn an_empty_completion_prompt_is_a_client_error_before_the_target() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let service = service(target);
        let response = service.dispatch(
            HttpMethod::Post,
            "/v1/completions",
            br#"{"prompt":"","max_tokens":4}"#,
        );
        assert_eq!(response.status, 400);
        assert_eq!(body(&response)["error"]["type"], "invalid_request_error");
        assert!(
            body(&response)["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("non-empty"))
        );
        // The SSE path parses through the same boundary.
        assert!(
            service
                .parse_completion_mode(br#"{"prompt":"","max_tokens":4,"stream":true}"#, true)
                .is_err()
        );
        assert!(requests.lock().unwrap().is_empty());
    }

    #[test]
    fn the_completion_prompt_boundary_refuses_emptiness_and_nothing_else() {
        // Refuse exactly what can never be continued; whitespace is text, and
        // a chat message with empty content is still a rendered chat turn.
        for (request_body, status) in [
            (br#"{"prompt":" ","max_tokens":2}"#.as_slice(), 200),
            (br#"{"prompt":"\n","max_tokens":2}"#.as_slice(), 200),
            (br#"{"prompt":"","max_tokens":2}"#.as_slice(), 400),
            (br#"{"max_tokens":2}"#.as_slice(), 400),
            (br#"{"prompt":null,"max_tokens":2}"#.as_slice(), 400),
            (br#"{"prompt":["a"],"max_tokens":2}"#.as_slice(), 400),
            (br#"{"prompt":7,"max_tokens":2}"#.as_slice(), 400),
        ] {
            let target = RecordingTarget::new();
            let requests = Arc::clone(&target.requests);
            let response =
                service(target).dispatch(HttpMethod::Post, "/v1/completions", request_body);
            let shown = String::from_utf8_lossy(request_body);
            assert_eq!(response.status, status, "{shown}");
            assert_eq!(
                requests.lock().unwrap().len(),
                usize::from(status == 200),
                "{shown}"
            );
            if status == 400 {
                assert_eq!(
                    body(&response)["error"]["type"],
                    "invalid_request_error",
                    "{shown}"
                );
            }
        }
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/chat/completions",
            br#"{"messages":[{"role":"user","content":""}],"max_tokens":2}"#,
        );
        assert_eq!(response.status, 200);
        assert_eq!(requests.lock().unwrap().len(), 1);
        // An empty `messages` array stays refused, as before.
        let target = RecordingTarget::new();
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/chat/completions",
            br#"{"messages":[],"max_tokens":2}"#,
        );
        assert_eq!(response.status, 400);
    }

    #[test]
    fn a_streamed_empty_completion_prompt_is_a_400_over_the_wire() {
        // The SSE handler has its own entry; prove over a real socket that an
        // empty prompt is refused before any event stream starts and before
        // the target is entered, with and without `stream`.
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let server =
            OpenAiHttpServer::bind("127.0.0.1:0", target, ServerConfig::default()).unwrap();
        let port = listener_tcp_port(server.listener()).expect("TCP listener has a port");
        thread::spawn(move || {
            let _ = server.serve_forever();
        });
        for stream in ["false", "true"] {
            let body = format!(r#"{{"prompt":"","max_tokens":2,"stream":{stream}}}"#);
            // The server answers, then releases its one generation permit; a
            // client that is quick enough to send its next request in between
            // is told to come back (429 with Retry-After), exactly as a real
            // client would be. Come back, as that response says to.
            let mut wire = String::new();
            for _ in 0..200 {
                let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
                write!(
                    socket,
                    "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len(),
                )
                .unwrap();
                socket.flush().unwrap();
                wire = read_http_response(socket);
                if !wire.starts_with("HTTP/1.1 429") {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            assert!(wire.starts_with("HTTP/1.1 400"), "stream={stream}: {wire}");
            assert!(wire.contains("invalid_request_error"), "stream={stream}: {wire}");
            assert!(!wire.contains("text/event-stream"), "stream={stream}: {wire}");
        }
        assert!(requests.lock().unwrap().is_empty());
    }

    #[test]
    fn chat_preserves_additional_message_fields() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/chat/completions",
            br#"{"messages":[{"role":"user","content":"hi","name":"Chris"}],"max_tokens":4}"#,
        );
        assert_eq!(response.status, 200);
        assert_eq!(
            body(&response)["choices"][0]["message"]["content"],
            "K3 output"
        );
        let recorded = requests.lock().unwrap();
        let TargetPrompt::Chat(messages) = &recorded[0].prompt else {
            panic!("expected chat request")
        };
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hi");
        assert_eq!(messages[0].additional_fields["name"], "Chris");
    }

    #[test]
    fn chat_reasoning_effort_is_normalized_validated_and_reaches_the_target() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/chat/completions",
            br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"reasoning_effort":" LOW "}"#,
        );
        assert_eq!(response.status, 200);
        assert_eq!(
            requests.lock().unwrap()[0].reasoning_effort.as_deref(),
            Some("low")
        );

        let absent = RecordingTarget::new();
        let absent_requests = Arc::clone(&absent.requests);
        let response = service(absent).dispatch(
            HttpMethod::Post,
            "/v1/chat/completions",
            br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
        );
        assert_eq!(response.status, 200);
        assert_eq!(absent_requests.lock().unwrap()[0].reasoning_effort, None);

        for wire in [
            br#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"medium"}"#.as_slice(),
            br#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":3}"#.as_slice(),
        ] {
            let target = RecordingTarget::new();
            let requests = Arc::clone(&target.requests);
            let response =
                service(target).dispatch(HttpMethod::Post, "/v1/chat/completions", wire);
            assert_eq!(response.status, 400);
            assert!(
                body(&response)["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("reasoning_effort")
            );
            assert!(requests.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn text_part_arrays_remain_valid_native_chat_input() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/chat/completions",
            br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":4}"#,
        );
        assert_eq!(response.status, 200);
        let recorded = requests.lock().unwrap();
        let TargetPrompt::Chat(messages) = &recorded[0].prompt else {
            panic!("expected chat request")
        };
        assert_eq!(messages[0].content[0]["type"], "text");
        assert_eq!(messages[0].content[0]["text"], "hi");
    }

    #[test]
    fn multimodal_parts_fail_before_the_authoritative_target_is_entered() {
        for content in [
            r#"[{"type":"image_url","image_url":{"url":"file:///tmp/image.png"}}]"#,
            r#"[{"type":"image","image":"opaque"}]"#,
            r#"[{"type":"input_audio","input_audio":{"data":"opaque"}}]"#,
        ] {
            let target = RecordingTarget::new();
            let requests = Arc::clone(&target.requests);
            let wire =
                format!(r#"{{"messages":[{{"role":"user","content":{content}}}],"max_tokens":4}}"#);
            let response =
                service(target).dispatch(HttpMethod::Post, "/v1/chat/completions", wire.as_bytes());
            assert_eq!(response.status, 400);
            assert!(
                body(&response)["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("text-only"))
            );
            assert!(requests.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn unsupported_top_level_semantics_fail_before_target_entry() {
        let cases: &[(&str, &[u8], &str)] = &[
            (
                "/v1/completions",
                br#"{"prompt":"x","stop":"END"}"#,
                "stop",
            ),
            (
                "/v1/completions",
                br#"{"prompt":"x","logprobs":1}"#,
                "logprobs",
            ),
            (
                "/v1/completions",
                br#"{"prompt":"x","echo":true}"#,
                "echo",
            ),
            (
                "/v1/completions",
                br#"{"prompt":"x","unknown_future_option":true}"#,
                "unknown_future_option",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f"}}]}"#,
                "tools",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"user","content":"x"}],"tool_choice":"required"}"#,
                "tool_choice",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}"#,
                "response_format",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"user","content":"x"}],"frequency_penalty":0.2}"#,
                "frequency_penalty",
            ),
        ];
        for &(path, wire, field) in cases {
            let target = RecordingTarget::new();
            let requests = Arc::clone(&target.requests);
            let response = service(target).dispatch(HttpMethod::Post, path, wire);
            assert_eq!(response.status, 400, "field {field}");
            assert!(
                body(&response)["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(field)),
                "field {field} did not receive a clear error"
            );
            assert!(requests.lock().unwrap().is_empty(), "field {field}");
        }
    }

    #[test]
    fn documented_greedy_compatibility_fields_are_validated_but_not_semantic() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let service = service(target);
        let first = service.dispatch(
            HttpMethod::Post,
            "/v1/completions",
            br#"{
                "model":"deltafin-kimi-k3","prompt":"same","max_tokens":3,
                "temperature":0.7,"top_p":0.25,"user":"one","metadata":{"trace":"a"},
                "frequency_penalty":0,"presence_penalty":0,"logit_bias":{},
                "stop":[],"seed":null,"echo":false,"suffix":"","logprobs":null
            }"#,
        );
        assert_eq!(first.status, 200);
        let second = service.dispatch(
            HttpMethod::Post,
            "/v1/completions",
            br#"{
                "model":"deltafin-kimi-k3","prompt":"same","max_tokens":3,
                "temperature":2,"top_p":1,"user":"two","metadata":{"trace":"b"}
            }"#,
        );
        assert_eq!(second.status, 200);
        assert_eq!(requests.lock().unwrap().len(), 1);

        for wire in [
            br#"{"prompt":"x","temperature":2.01}"#.as_slice(),
            br#"{"prompt":"x","temperature":"hot"}"#.as_slice(),
            br#"{"prompt":"x","top_p":-0.01}"#.as_slice(),
            br#"{"prompt":"x","top_p":1.01}"#.as_slice(),
        ] {
            let response = service.dispatch(HttpMethod::Post, "/v1/completions", wire);
            assert_eq!(response.status, 400);
        }
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn exact_noop_chat_defaults_and_safe_metadata_remain_compatible() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/chat/completions",
            br#"{
                "model":"deltafin-kimi-k3",
                "messages":[{"role":"user","content":"x","name":"Chris"}],
                "max_completion_tokens":2,"temperature":0,"top_p":1,
                "user":"local","metadata":{"request":"42"},"store":false,
                "frequency_penalty":0,"presence_penalty":0,"logit_bias":{},
                "stop":[],"seed":null,"logprobs":false,"top_logprobs":null,
                "response_format":{"type":"text"},"tools":[],"tool_choice":null,
                "parallel_tool_calls":true,"modalities":["text"],"audio":null
            }"#,
        );
        assert_eq!(response.status, 200);
        let recorded = requests.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].max_new_tokens, 2);
    }

    #[test]
    fn model_identity_alias_conflicts_and_message_semantics_fail_closed() {
        let cases: &[(&str, &[u8], &str)] = &[
            (
                "/v1/completions",
                br#"{"model":"some-other-model","prompt":"x"}"#,
                "unavailable",
            ),
            (
                "/v1/completions",
                br#"{"prompt":"x","max_tokens":2,"max_completion_tokens":3}"#,
                "only one",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"developer","content":"x"}]}"#,
                "developer",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"user","content":"x","refusal":"hidden"}]}"#,
                "refusal",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"user","content":[{"type":"text","text":"x","future_semantic":1}]}]}"#,
                "future_semantic",
            ),
            (
                "/v1/chat/completions",
                br#"{"messages":[{"role":"assistant","content":null,"tool_calls":[{"type":"function","function":{"name":"f","arguments":[]}}]}]}"#,
                "arguments",
            ),
        ];
        for &(path, wire, message_fragment) in cases {
            let target = RecordingTarget::new();
            let requests = Arc::clone(&target.requests);
            let response = service(target).dispatch(HttpMethod::Post, path, wire);
            assert_eq!(response.status, 400, "{message_fragment}");
            assert!(
                body(&response)["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(message_fragment))
            );
            assert!(requests.lock().unwrap().is_empty());
        }

        let service = service(RecordingTarget::new());
        assert_eq!(
            service
                .parse_completion(br#"{"prompt":"x","max_tokens":null,"max_completion_tokens":7}"#)
                .unwrap()
                .max_new_tokens,
            7
        );
    }

    #[test]
    fn unsupported_streaming_fails_before_target_entry() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let response = service(target).dispatch(
            HttpMethod::Post,
            "/v1/completions",
            br#"{"prompt":"hello","stream":true}"#,
        );
        assert_eq!(response.status, 400);
        assert!(requests.lock().unwrap().is_empty());
    }

    #[test]
    fn token_requests_are_clamped_and_zero_uses_endpoint_default() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let config = ServerConfig {
            max_new_tokens: 100,
            default_completion_tokens: 20,
            default_chat_tokens: 80,
            ..ServerConfig::default()
        };
        let service = OpenAiService::new(target, config).unwrap();
        assert_eq!(
            service
                .parse_completion(br#"{"prompt":"x","max_tokens":500}"#)
                .unwrap()
                .max_new_tokens,
            100
        );
        assert_eq!(
            service
                .parse_completion(br#"{"prompt":"x","max_tokens":0}"#)
                .unwrap()
                .max_new_tokens,
            20
        );
        assert!(requests.lock().unwrap().is_empty());
    }

    #[test]
    fn long_context_body_default_is_bounded_and_override_is_validated() {
        let default = ServerConfig::default();
        assert_eq!(
            default.max_request_body_bytes,
            DEFAULT_MAX_REQUEST_BODY_BYTES
        );
        assert!(default.max_request_body_bytes > 1024 * 1024);
        assert!(OpenAiService::new(RecordingTarget::new(), default).is_ok());

        let too_large = ServerConfig {
            max_request_body_bytes: MAX_REQUEST_BODY_BYTES + 1,
            ..ServerConfig::default()
        };
        assert!(OpenAiService::new(RecordingTarget::new(), too_large).is_err());
    }

    #[test]
    fn concurrent_dispatch_still_has_one_target_owner() {
        let mut target = RecordingTarget::new();
        target.delay = Duration::from_millis(20);
        let maximum_active = Arc::clone(&target.maximum_active);
        let service = service(target);
        let one = service.clone();
        let two = service.clone();
        let first = thread::spawn(move || {
            one.dispatch(HttpMethod::Post, "/v1/completions", br#"{"prompt":"one"}"#)
        });
        let second = thread::spawn(move || {
            two.dispatch(HttpMethod::Post, "/v1/completions", br#"{"prompt":"two"}"#)
        });
        assert_eq!(first.join().unwrap().status, 200);
        assert_eq!(second.join().unwrap().status, 200);
        assert_eq!(maximum_active.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn busy_response_is_openai_shaped_429_with_retry_hint() {
        let response = busy_response();
        assert_eq!(response.status, 429);
        assert!(response.close_connection);
        assert_eq!(response.retry_after_seconds, Some(1));
        assert_eq!(body(&response)["error"]["type"], "server_busy");
        assert_eq!(body(&response)["error"]["code"], "server_busy");

        let response = response.into_tiny_http();
        assert_eq!(response.status_code(), StatusCode(429));
        assert!(
            response.headers().iter().any(|header| {
                header.field.equiv("Retry-After") && header.value.as_str() == "1"
            })
        );

        let mut wire = Vec::new();
        write_busy_response(&mut wire, "1.1").unwrap();
        let wire = std::str::from_utf8(&wire).unwrap();
        let (headers, body) = wire.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("HTTP/1.1 429 Too Many Requests\r\n"));
        assert!(headers.contains("Retry-After: 1\r\n"));
        assert!(headers.contains("Connection: close\r\n"));
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["error"]["code"], "server_busy");
    }

    #[test]
    fn only_completion_posts_need_the_generation_permit() {
        let completion: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/v1/completions?client=test")
            .into();
        let chat: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/chat/completions")
            .into();
        let models: Request = tiny_http::TestRequest::new()
            .with_method(Method::Get)
            .with_path("/v1/models")
            .into();
        let wrong_method: Request = tiny_http::TestRequest::new()
            .with_method(Method::Get)
            .with_path("/v1/completions")
            .into();
        assert!(request_requires_generation_permit(&completion));
        assert!(request_requires_generation_permit(&chat));
        assert!(!request_requires_generation_permit(&models));
        assert!(!request_requires_generation_permit(&wrong_method));
    }

    #[test]
    fn concurrent_http_request_fails_fast_without_entering_or_settling_target() {
        struct BlockingStreamTarget {
            entered: mpsc::SyncSender<()>,
            release: mpsc::Receiver<()>,
            calls: Arc<AtomicUsize>,
            publications: Arc<Mutex<Vec<StreamPublication>>>,
        }

        impl AuthoritativeTarget for BlockingStreamTarget {
            fn generate_target(
                &mut self,
                _request: &TargetRequest,
                _client: &ClientPresence,
            ) -> crate::error::Result<TargetOutput> {
                panic!("stream request entered the non-streaming target method")
            }

            fn generate_target_stream(
                &mut self,
                _request: &TargetRequest,
                sink: &mut dyn TargetDeltaSink,
                _client: &ClientPresence,
            ) -> std::result::Result<TargetStreamSummary, StreamGenerationError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                sink.publish_target_delta(TargetDelta::target_verified_content("certified"))?;
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
                Ok(TargetStreamSummary::target_verified(FinishReason::Stop))
            }

            fn finish_target_stream(&mut self, publication: StreamPublication) {
                self.publications.lock().unwrap().push(publication);
            }
        }

        let (entered_tx, entered_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::sync_channel(0);
        let calls = Arc::new(AtomicUsize::new(0));
        let publications = Arc::new(Mutex::new(Vec::new()));
        let service = OpenAiService::new(
            BlockingStreamTarget {
                entered: entered_tx,
                release: release_rx,
                calls: Arc::clone(&calls),
                publications: Arc::clone(&publications),
            },
            ServerConfig::default(),
        )
        .unwrap();

        let first_service = service.clone();
        let first = thread::spawn(move || {
            let request: Request = tiny_http::TestRequest::new()
                .with_method(Method::Post)
                .with_path("/v1/completions")
                .with_body(r#"{"prompt":"first","stream":true}"#)
                .into();
            first_service.handle_request(request).unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        // Health/model discovery remains responsive because it cannot touch
        // the mutable target transaction.
        let models: Request = tiny_http::TestRequest::new()
            .with_method(Method::Get)
            .with_path("/v1/models")
            .into();
        service.handle_request(models).unwrap();

        let second_service = service.clone();
        let (second_done_tx, second_done_rx) = mpsc::sync_channel(0);
        let second = thread::spawn(move || {
            let request: Request = tiny_http::TestRequest::new()
                .with_method(Method::Post)
                .with_path("/v1/completions")
                .with_body(r#"{"prompt":"must not run","stream":true}"#)
                .into();
            second_service.handle_request(request).unwrap();
            second_done_tx.send(()).unwrap();
        });

        // This deadline is not a performance benchmark. It proves the second
        // handler returned while the first target remained deliberately
        // blocked, rather than serializing behind its mutex.
        second_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("busy request must return before active generation ends");
        second.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(publications.lock().unwrap().is_empty());

        release_tx.send(()).unwrap();
        first.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            publications.lock().unwrap().as_slice(),
            &[StreamPublication::Complete]
        );
        assert!(matches!(
            service.shared.generation_gate.admit(0),
            Admission::Permit(_)
        ));
    }

    #[test]
    fn malformed_json_and_multiple_choices_are_rejected() {
        let service = service(RecordingTarget::new());
        assert_eq!(
            service
                .dispatch(HttpMethod::Post, "/v1/completions", b"{")
                .status,
            400
        );
        assert_eq!(
            service
                .dispatch(
                    HttpMethod::Post,
                    "/v1/completions",
                    br#"{"prompt":"x","n":2}"#,
                )
                .status,
            400
        );
    }

    #[test]
    fn tiny_http_adapter_enforces_the_body_limit_before_generation() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let config = ServerConfig {
            max_request_body_bytes: 8,
            ..ServerConfig::default()
        };
        let service = OpenAiService::new(target, config).unwrap();
        let request: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/v1/completions")
            .with_body(r#"{"prompt":"too large"}"#)
            .into();
        service.handle_request(request).unwrap();
        assert!(requests.lock().unwrap().is_empty());
    }

    #[test]
    fn tiny_http_adapter_reaches_the_authoritative_target() {
        let target = RecordingTarget::new();
        let requests = Arc::clone(&target.requests);
        let service = service(target);
        let request: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/v1/completions?client=test")
            .with_body(r#"{"prompt":"through HTTP","max_tokens":3}"#)
            .into();
        service.handle_request(request).unwrap();
        assert_eq!(
            requests.lock().unwrap().as_slice(),
            &[TargetRequest {
                prompt: TargetPrompt::Completion("through HTTP".to_owned()),
                max_new_tokens: 3,
                reasoning_effort: None,
            }]
        );
    }

    #[test]
    fn backend_error_becomes_openai_server_error() {
        struct FailingTarget;
        impl AuthoritativeTarget for FailingTarget {
            fn generate_target(
                &mut self,
                _request: &TargetRequest,
                _client: &ClientPresence,
            ) -> crate::error::Result<TargetOutput> {
                Err(DeltafinError::new("target failed"))
            }
        }
        let service = OpenAiService::new(FailingTarget, ServerConfig::default()).unwrap();
        let response = service.dispatch(HttpMethod::Post, "/v1/completions", br#"{"prompt":"x"}"#);
        assert_eq!(response.status, 500);
        assert_eq!(body(&response)["error"]["type"], "server_error");
    }

    #[test]
    fn backend_error_aborts_nonstream_target_publication() {
        struct FailingTarget {
            publications: Arc<Mutex<Vec<StreamPublication>>>,
        }
        impl AuthoritativeTarget for FailingTarget {
            fn generate_target(
                &mut self,
                _request: &TargetRequest,
                _client: &ClientPresence,
            ) -> crate::error::Result<TargetOutput> {
                Err(DeltafinError::new("target failed"))
            }

            fn finish_target_response(&mut self, publication: StreamPublication) {
                self.publications.lock().unwrap().push(publication);
            }
        }

        let publications = Arc::new(Mutex::new(Vec::new()));
        let service = OpenAiService::new(
            FailingTarget {
                publications: Arc::clone(&publications),
            },
            ServerConfig::default(),
        )
        .unwrap();
        let request: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/v1/completions")
            .with_body(r#"{"prompt":"x"}"#)
            .into();

        service.handle_request(request).unwrap();
        assert_eq!(
            publications.lock().unwrap().as_slice(),
            &[StreamPublication::Aborted]
        );
    }

    #[test]
    fn completion_sse_wire_has_deltas_finish_usage_and_done() {
        let mut target = StreamingTarget::successful(vec![
            TargetDelta::target_verified_content("A"),
            TargetDelta::target_verified_reasoning("private"),
            TargetDelta::target_verified_content("B"),
        ]);
        let mut wire = Vec::new();
        stream_target_to_writer(
            &mut target,
            &completion_request(),
            StreamKind::Completion,
            true,
            &fixed_identity(),
            "1.1",
            &mut wire,
            &ClientPresence::assumed_present(),
        )
        .unwrap();

        let header = std::str::from_utf8(&wire).unwrap();
        assert!(header.starts_with("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream"));
        let lines = sse_data_lines(&wire);
        assert_eq!(lines.len(), 5);
        let first: Value = serde_json::from_str(lines[0]).unwrap();
        let second: Value = serde_json::from_str(lines[1]).unwrap();
        let finish: Value = serde_json::from_str(lines[2]).unwrap();
        let usage: Value = serde_json::from_str(lines[3]).unwrap();
        assert_eq!(first["choices"][0]["text"], "A");
        assert_eq!(second["choices"][0]["text"], "B");
        assert_eq!(first["usage"], Value::Null);
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["total_tokens"], 9);
        assert_eq!(lines[4], "[DONE]");
        assert_eq!(target.publications, [StreamPublication::Complete]);
    }

    #[test]
    fn chat_sse_keeps_reasoning_and_content_channels_separate() {
        let mut target = StreamingTarget::successful(vec![
            TargetDelta::target_verified_reasoning("think"),
            TargetDelta::target_verified_content("answer"),
        ]);
        let request = TargetRequest {
            prompt: TargetPrompt::Chat(vec![ChatMessage {
                role: "user".to_owned(),
                content: Value::String("question".to_owned()),
                additional_fields: Map::new(),
            }]),
            max_new_tokens: 8,
            reasoning_effort: None,
        };
        let identity = StreamIdentity {
            id: "chatcmpl-test",
            created: 456,
            model: "deltafin-kimi-k3",
        };
        let mut wire = Vec::new();
        stream_target_to_writer(
            &mut target,
            &request,
            StreamKind::Chat,
            false,
            &identity,
            "1.1",
            &mut wire,
            &ClientPresence::assumed_present(),
        )
        .unwrap();

        let lines = sse_data_lines(&wire);
        assert_eq!(lines.len(), 5);
        let role: Value = serde_json::from_str(lines[0]).unwrap();
        let reasoning: Value = serde_json::from_str(lines[1]).unwrap();
        let content: Value = serde_json::from_str(lines[2]).unwrap();
        let finish: Value = serde_json::from_str(lines[3]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            reasoning["choices"][0]["delta"]["reasoning_content"],
            "think"
        );
        assert_eq!(content["choices"][0]["delta"]["content"], "answer");
        assert_eq!(finish["object"], "chat.completion.chunk");
        assert_eq!(finish["choices"][0]["delta"], json!({}));
        assert_eq!(lines[4], "[DONE]");
        assert_eq!(target.publications, [StreamPublication::Complete]);
    }

    #[test]
    fn terminal_flush_failure_aborts_response_boundary_publication() {
        struct FailFinalFlush {
            bytes: Vec<u8>,
            saw_done: bool,
        }

        impl Write for FailFinalFlush {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if bytes.windows(6).any(|window| window == b"[DONE]") {
                    self.saw_done = true;
                }
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                if self.saw_done {
                    Err(io::Error::new(io::ErrorKind::BrokenPipe, "client left"))
                } else {
                    Ok(())
                }
            }
        }

        let mut target =
            StreamingTarget::successful(vec![TargetDelta::target_verified_content("certified")]);
        let mut writer = FailFinalFlush {
            bytes: Vec::new(),
            saw_done: false,
        };
        let error = stream_target_to_writer(
            &mut target,
            &completion_request(),
            StreamKind::Completion,
            false,
            &fixed_identity(),
            "1.1",
            &mut writer,
            &ClientPresence::assumed_present(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(target.publications, [StreamPublication::Aborted]);
        assert!(String::from_utf8_lossy(&writer.bytes).contains("[DONE]"));
    }

    #[test]
    fn target_stream_failure_emits_error_without_done_and_aborts() {
        let mut target = StreamingTarget {
            deltas: Vec::new(),
            publications: Vec::new(),
            fail_target: true,
        };
        let mut wire = Vec::new();
        let error = stream_target_to_writer(
            &mut target,
            &completion_request(),
            StreamKind::Completion,
            false,
            &fixed_identity(),
            "1.1",
            &mut wire,
            &ClientPresence::assumed_present(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        let lines = sse_data_lines(&wire);
        assert_eq!(lines.len(), 1);
        let frame: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(frame["error"]["message"], "target stream failed");
        assert_eq!(target.publications, [StreamPublication::Aborted]);
    }

    #[test]
    fn stream_options_are_validated_before_target_entry() {
        let target = RecordingTarget::new();
        let service = service(target);
        let parsed = service
            .parse_completion_mode(
                br#"{"prompt":"x","stream":true,"stream_options":{"include_usage":true}}"#,
                true,
            )
            .unwrap();
        assert!(parsed.include_usage);
        assert!(
            service
                .parse_completion_mode(
                    br#"{"prompt":"x","stream":true,"stream_options":{"include_usage":7}}"#,
                    true,
                )
                .is_err()
        );
        assert!(
            service
                .parse_completion_mode(
                    br#"{"prompt":"x","stream":true,"stream_options":{"future":true}}"#,
                    true,
                )
                .is_err()
        );
        assert!(
            service
                .parse_completion_mode(
                    br#"{"prompt":"x","stream_options":{"include_usage":true}}"#,
                    false,
                )
                .is_err()
        );
    }

    #[test]
    fn tiny_http_stream_route_uses_streaming_target_and_completes() {
        struct ObservedStreamTarget {
            calls: Arc<AtomicUsize>,
            publications: Arc<Mutex<Vec<StreamPublication>>>,
        }

        impl AuthoritativeTarget for ObservedStreamTarget {
            fn generate_target(
                &mut self,
                _request: &TargetRequest,
                _client: &ClientPresence,
            ) -> crate::error::Result<TargetOutput> {
                panic!("stream request entered the non-streaming target method")
            }

            fn generate_target_stream(
                &mut self,
                _request: &TargetRequest,
                sink: &mut dyn TargetDeltaSink,
                _client: &ClientPresence,
            ) -> std::result::Result<TargetStreamSummary, StreamGenerationError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                sink.publish_target_delta(TargetDelta::target_verified_content("native"))?;
                Ok(TargetStreamSummary::target_verified(FinishReason::Stop))
            }

            fn finish_target_stream(&mut self, publication: StreamPublication) {
                self.publications.lock().unwrap().push(publication);
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let publications = Arc::new(Mutex::new(Vec::new()));
        let target = ObservedStreamTarget {
            calls: Arc::clone(&calls),
            publications: Arc::clone(&publications),
        };
        let service = OpenAiService::new(target, ServerConfig::default()).unwrap();
        let request: Request = tiny_http::TestRequest::new()
            .with_method(Method::Post)
            .with_path("/v1/completions")
            .with_body(r#"{"prompt":"hello","stream":true}"#)
            .into();
        service.handle_request(request).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            publications.lock().unwrap().as_slice(),
            &[StreamPublication::Complete]
        );
    }

    #[test]
    fn assumed_present_client_never_disconnects_and_probes_reflect_their_socket() {
        assert!(!ClientPresence::assumed_present().disconnected());
        let departed = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&departed);
        let client =
            ClientPresence::from_probe(Box::new(move || observed.load(Ordering::Relaxed)));
        assert!(!client.disconnected());
        departed.store(true, Ordering::Relaxed);
        assert!(client.disconnected());
    }

    #[test]
    fn generation_queue_beyond_the_cap_fails_validation() {
        let beyond = ServerConfig {
            generation_queue_slots: MAX_GENERATION_QUEUE_SLOTS + 1,
            ..ServerConfig::default()
        };
        assert!(OpenAiService::new(RecordingTarget::new(), beyond).is_err());
        let at_cap = ServerConfig {
            generation_queue_slots: MAX_GENERATION_QUEUE_SLOTS,
            ..ServerConfig::default()
        };
        assert!(OpenAiService::new(RecordingTarget::new(), at_cap).is_ok());
    }

    fn expect_permit(admission: Admission) -> GenerationPermit {
        match admission {
            Admission::Permit(permit) => permit,
            Admission::Queued(_) => panic!("expected an immediate permit, got a queue ticket"),
            Admission::Busy => panic!("expected an immediate permit, got busy"),
        }
    }

    fn expect_ticket(admission: Admission) -> QueueTicket {
        match admission {
            Admission::Queued(ticket) => ticket,
            Admission::Permit(_) => panic!("expected a queue ticket, got an immediate permit"),
            Admission::Busy => panic!("expected a queue ticket, got busy"),
        }
    }

    #[test]
    fn gate_serves_waiters_in_arrival_order_and_rejects_beyond_capacity() {
        let gate = Arc::new(GenerationGate::default());
        let permit = expect_permit(gate.admit(2));
        let first = expect_ticket(gate.admit(2));
        let second = expect_ticket(gate.admit(2));
        assert!(matches!(gate.admit(2), Admission::Busy));

        let (second_claimed_tx, second_claimed_rx) = mpsc::channel();
        let second_waiter = thread::spawn(move || {
            let permit = second
                .wait_for_permit(&ClientPresence::assumed_present())
                .expect("an assumed-present waiter cannot be dropped");
            second_claimed_tx.send(()).unwrap();
            drop(permit);
        });

        drop(permit);
        // The freed slot must go to the queue front even though the younger
        // waiter has been runnable for longer.
        let first_permit = first
            .wait_for_permit(&ClientPresence::assumed_present())
            .expect("an assumed-present waiter cannot be dropped");
        assert!(
            second_claimed_rx
                .recv_timeout(Duration::from_millis(200))
                .is_err(),
            "the younger waiter claimed the slot ahead of the queue front"
        );
        drop(first_permit);
        second_claimed_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the second waiter must claim the slot after the first releases");
        second_waiter.join().unwrap();
        drop(expect_permit(gate.admit(2)));
    }

    #[test]
    fn dropping_an_unclaimed_ticket_frees_its_waiting_slot() {
        let gate = Arc::new(GenerationGate::default());
        let _permit = expect_permit(gate.admit(1));
        let ticket = expect_ticket(gate.admit(1));
        assert!(matches!(gate.admit(1), Admission::Busy));
        drop(ticket);
        drop(expect_ticket(gate.admit(1)));
    }

    #[test]
    fn queued_wait_ends_and_frees_the_slot_when_the_client_disconnects() {
        let gate = Arc::new(GenerationGate::default());
        let _permit = expect_permit(gate.admit(1));
        let ticket = expect_ticket(gate.admit(1));
        let departed = Arc::new(AtomicBool::new(true));
        let observed = Arc::clone(&departed);
        let client =
            ClientPresence::from_probe(Box::new(move || observed.load(Ordering::Relaxed)));
        assert!(
            ticket.wait_for_permit(&client).is_none(),
            "a provably disconnected waiter must not receive the generation slot"
        );
        assert_eq!(gate.lock_state().waiting.len(), 0);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn socket_probe_reports_only_peer_departure_and_never_consumes_bytes() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut client = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let (mut accepted, peer) = listener.accept().unwrap();
        let socket = client_socket::capture(peer, port)
            .expect("the accepted connection must be locatable by its exact 4-tuple");

        assert!(!socket.provably_disconnected(), "open idle socket");
        client.write_all(b"p").unwrap();
        assert!(
            !socket.provably_disconnected(),
            "a pipelined byte is not a disconnect"
        );
        let mut pipelined = [0_u8; 1];
        accepted.read_exact(&mut pipelined).unwrap();
        assert_eq!(&pipelined, b"p", "the probe must never consume stream bytes");

        // tiny_http shuts down its read half once a connection carries no
        // further requests; a local read shutdown must not read as the
        // client leaving.
        accepted.shutdown(std::net::Shutdown::Read).unwrap();
        assert!(
            !socket.provably_disconnected(),
            "a local read shutdown is not a client departure"
        );

        drop(client);
        wait_until("the probe to observe the client's departure", || {
            socket.provably_disconnected()
        });
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn probe_capture_misses_fabricated_peers_instead_of_guessing() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        // No connection exists from this fabricated peer, so capture must
        // decline rather than duplicate an unrelated descriptor.
        let fabricated: std::net::SocketAddr = "127.0.0.1:23456".parse().unwrap();
        assert!(client_socket::capture(fabricated, port).is_none());
    }

    fn waiting_count<T>(service: &OpenAiService<T>) -> usize {
        service.shared.generation_gate.lock_state().waiting.len()
    }

    fn http_post_completion(port: u16, prompt: &str) -> std::net::TcpStream {
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let body = format!(r#"{{"prompt":"{prompt}","max_tokens":2}}"#);
        write!(
            stream,
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len(),
        )
        .unwrap();
        stream.flush().unwrap();
        stream
    }

    fn read_http_response(mut stream: std::net::TcpStream) -> String {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut wire = String::new();
        let _ = stream.read_to_string(&mut wire);
        wire
    }

    /// A target that reports entry, then blocks until released, so tests can
    /// hold the generation slot at a precise moment.
    struct GatedTarget {
        entered: mpsc::SyncSender<()>,
        release: mpsc::Receiver<()>,
        calls: Arc<AtomicUsize>,
    }

    impl AuthoritativeTarget for GatedTarget {
        fn generate_target(
            &mut self,
            _request: &TargetRequest,
            _client: &ClientPresence,
        ) -> crate::error::Result<TargetOutput> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            Ok(TargetOutput::target_verified(
                "K3 output",
                FinishReason::Stop,
            ))
        }
    }

    fn spawn_gated_server(
        queue_slots: usize,
    ) -> (
        u16,
        OpenAiService<GatedTarget>,
        mpsc::Receiver<()>,
        mpsc::Sender<()>,
        Arc<AtomicUsize>,
    ) {
        let (entered_tx, entered_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let target = GatedTarget {
            entered: entered_tx,
            release: release_rx,
            calls: Arc::clone(&calls),
        };
        let config = ServerConfig {
            generation_queue_slots: queue_slots,
            ..ServerConfig::default()
        };
        let server = OpenAiHttpServer::bind("127.0.0.1:0", target, config).unwrap();
        let port = listener_tcp_port(server.listener()).expect("TCP listener has a port");
        let service = server.service().clone();
        thread::spawn(move || {
            let _ = server.serve_forever();
        });
        (port, service, entered_rx, release_tx, calls)
    }

    #[test]
    fn served_queue_admits_one_waiter_in_order_and_rejects_the_next() {
        let (port, service, entered_rx, release_tx, calls) = spawn_gated_server(1);

        let running = http_post_completion(port, "running");
        entered_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the first request must enter the target");

        let queued = http_post_completion(port, "queued");
        wait_until("the second request to join the wait queue", || {
            waiting_count(&service) == 1
        });

        let rejected = http_post_completion(port, "rejected");
        let rejection = read_http_response(rejected);
        assert!(rejection.starts_with("HTTP/1.1 429"), "{rejection}");
        assert!(rejection.contains("server_busy"), "{rejection}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        release_tx.send(()).unwrap();
        let first = read_http_response(running);
        assert!(first.contains("\"K3 output\""), "{first}");

        entered_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the queued request must enter the target after the slot frees");
        release_tx.send(()).unwrap();
        let second = read_http_response(queued);
        assert!(second.contains("\"K3 output\""), "{second}");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(waiting_count(&service), 0);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn served_queue_frees_the_slot_of_a_client_that_disconnects_while_waiting() {
        let (port, service, entered_rx, release_tx, calls) = spawn_gated_server(1);

        let running = http_post_completion(port, "running");
        entered_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the first request must enter the target");

        let abandoned = http_post_completion(port, "abandoned");
        wait_until("the second request to join the wait queue", || {
            waiting_count(&service) == 1
        });
        drop(abandoned);
        wait_until("the departed waiter's slot to free", || {
            waiting_count(&service) == 0
        });

        release_tx.send(()).unwrap();
        let first = read_http_response(running);
        assert!(first.contains("\"K3 output\""), "{first}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a waiter that disconnected must never enter the target"
        );
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn served_generation_observes_its_vanished_client_through_the_probe() {
        struct DepartureObservingTarget {
            entered: mpsc::SyncSender<()>,
            observed_departure: Arc<AtomicBool>,
        }

        impl AuthoritativeTarget for DepartureObservingTarget {
            fn generate_target(
                &mut self,
                _request: &TargetRequest,
                client: &ClientPresence,
            ) -> crate::error::Result<TargetOutput> {
                self.entered.send(()).unwrap();
                // Poll exactly as the engine's transaction boundaries do.
                let deadline = Instant::now() + Duration::from_secs(10);
                while Instant::now() < deadline {
                    if client.disconnected() {
                        self.observed_departure.store(true, Ordering::SeqCst);
                        return Err(DeltafinError::new("client disconnected"));
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Ok(TargetOutput::target_verified(
                    "nobody is listening",
                    FinishReason::Stop,
                ))
            }
        }

        let (entered_tx, entered_rx) = mpsc::sync_channel(0);
        let observed_departure = Arc::new(AtomicBool::new(false));
        let target = DepartureObservingTarget {
            entered: entered_tx,
            observed_departure: Arc::clone(&observed_departure),
        };
        let server =
            OpenAiHttpServer::bind("127.0.0.1:0", target, ServerConfig::default()).unwrap();
        let port = listener_tcp_port(server.listener()).expect("TCP listener has a port");
        thread::spawn(move || {
            let _ = server.serve_forever();
        });

        let vanishing = http_post_completion(port, "vanishing");
        entered_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the request must enter the target while its client is connected");
        drop(vanishing);
        wait_until("the target to observe the disconnect", || {
            observed_departure.load(Ordering::SeqCst)
        });
    }
}
