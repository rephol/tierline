//! tierline: a pass-through proxy in front of the Anthropic API that logs cost per request.
//! With `TIERLINE_ROUTE=1` it also picks the model per request; otherwise it is pure pass-through.

pub mod classify;
pub mod models;
pub mod pricing;
pub mod route;
pub mod summary;
pub mod tokens;
pub mod usage;

use axum::{
    body::{to_bytes, Body},
    extract::State,
    http::{header, HeaderMap, HeaderName, Request, Response, StatusCode},
    Router,
};
use futures_util::StreamExt;
use pricing::Pricing;
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{io::AsyncWriteExt, sync::mpsc};
use usage::{Usage, UsageParser};

const MAX_REQUEST_BODY: usize = 64 * 1024 * 1024;

pub struct AppState {
    pub upstream: String,
    pub client: reqwest::Client,
    pub pricing: Pricing,
    /// Serialized log lines go to a single writer task; see `spawn_log_writer`.
    pub log: mpsc::UnboundedSender<String>,
    /// `None` is pure pass-through; `Some` rewrites the `model` of POST /v1/messages.
    pub routing: Option<route::Routing>,
}

/// Starts the single task that appends log lines to `file`. It exits once every sender is
/// dropped and the queue is drained, so awaiting the handle after shutdown flushes the log.
pub fn spawn_log_writer(
    file: std::fs::File,
) -> (mpsc::UnboundedSender<String>, tokio::task::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let handle = tokio::spawn(async move {
        let mut f = tokio::fs::File::from_std(file);
        while let Some(mut line) = rx.recv().await {
            line.push('\n');
            if f.write_all(line.as_bytes()).await.is_err() || f.flush().await.is_err() {
                eprintln!("tierline: log write failed");
            }
        }
    });
    (tx, handle)
}

/// One line of the JSONL cost log.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LogRecord {
    pub ts_ms: u128,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub requested_model: Option<String>,
    pub model: Option<String>,
    pub stream: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
    pub cost_usd: Option<f64>,
    /// Set when no upstream response was obtained (status is then 502).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Hash of system prompt + first user message (routable requests only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_key: Option<String>,
    /// The routing decision, present only when routing is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<route::RouteLog>,
    /// Microseconds from the request body being fully read to the upstream send being
    /// issued. Includes JSON parsing and routing, which is the overhead the proxy adds.
    pub pre_upstream_us: u128,
    pub ttfb_ms: u128,
    pub total_ms: u128,
}

pub fn app(state: Arc<AppState>) -> Router {
    Router::new().fallback(proxy).with_state(state)
}

/// Hop-by-hop headers, plus ones the proxy must recompute. `accept-encoding` is dropped so
/// the upstream answers uncompressed and usage can be read from the body.
fn skip_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "host"
            | "connection"
            | "keep-alive"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "content-length"
            | "accept-encoding"
    )
}

fn copy_headers(src: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in src {
        if !skip_header(k) {
            out.append(k.clone(), v.clone());
        }
    }
    out
}

fn error(status: StatusCode, msg: String) -> Response<Body> {
    let body =
        serde_json::json!({"type":"error","error":{"type":"tierline_proxy_error","message":msg}});
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn emit(state: &AppState, record: &LogRecord) {
    if let Ok(line) = serde_json::to_string(record) {
        let _ = state.log.send(line);
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Queues the log record when the response stream is dropped (completed or aborted).
struct Tracker {
    state: Arc<AppState>,
    parser: Option<UsageParser>,
    started: Instant,
    record: LogRecord,
}

impl Drop for Tracker {
    fn drop(&mut self) {
        let u: Usage = self
            .parser
            .take()
            .map(UsageParser::finish)
            .unwrap_or_default();
        let r = &mut self.record;
        r.total_ms = self.started.elapsed().as_millis();
        r.cost_usd = self.state.pricing.cost(&u);
        r.model = u
            .model
            .or_else(|| r.route.as_ref().and_then(|x| x.chosen_model.clone()));
        r.input_tokens = u.input_tokens;
        r.output_tokens = u.output_tokens;
        r.cache_creation_tokens = u.cache_creation_tokens;
        r.cache_read_tokens = u.cache_read_tokens;
        emit(&self.state, r);
    }
}

/// Only message creation is routed; count_tokens and everything else pass through.
fn is_routable(method: &axum::http::Method, path: &str) -> bool {
    method == axum::http::Method::POST && path.split('?').next() == Some("/v1/messages")
}

fn server_failure(status: StatusCode) -> bool {
    status.is_server_error() // includes Anthropic's 529 overloaded
}

/// Debug aid: with `TIERLINE_DUMP_BODIES=<dir>`, writes every request body to that
/// directory. Bodies contain prompts and file contents; keep the directory private.
fn dump_body(bytes: &[u8]) {
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let Some(dir) = DIR.get_or_init(|| std::env::var_os("TIERLINE_DUMP_BODIES").map(Into::into))
    else {
        return;
    };
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(format!("{}-{n:04}.json", now_ms())), bytes);
}

async fn proxy(State(st): State<Arc<AppState>>, req: Request<Body>) -> Response<Body> {
    let started = Instant::now();
    let (parts, body) = req.into_parts();
    let bytes = match to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(b) => b,
        Err(e) => return error(StatusCode::PAYLOAD_TOO_LARGE, e.to_string()),
    };
    let read_done = Instant::now();
    dump_body(&bytes);

    let path = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/")
        .to_string();
    let url = format!("{}{}", st.upstream.trim_end_matches('/'), path);
    let req_json: Option<serde_json::Value> = serde_json::from_slice(&bytes).ok();
    let requested_model = req_json
        .as_ref()
        .and_then(|v| v.get("model"))
        .and_then(|m| m.as_str())
        .map(String::from);
    let is_stream_req = req_json.as_ref().is_some_and(|v| v["stream"] == true);

    let routable = is_routable(&parts.method, &path) && requested_model.is_some();
    let conversation_key = req_json
        .as_ref()
        .filter(|_| routable)
        .map(route::conversation_key);
    // Models to try, best first. `None` means send the client's body untouched.
    let mut plan_log: Option<route::RouteLog> = None;
    let mut candidates: Vec<Option<String>> = vec![None];
    if let (true, Some(routing), Some(v)) = (routable, st.routing.as_ref(), req_json.as_ref()) {
        let plan = routing.plan(v, &st.pricing);
        plan_log = Some(plan.to_log());
        if !plan.ranked.is_empty() {
            candidates = plan.ranked.iter().map(|c| Some(c.model.clone())).collect();
        }
    }
    let routed = candidates[0].is_some();

    let mut record = LogRecord {
        ts_ms: now_ms(),
        method: parts.method.to_string(),
        path: path.clone(),
        requested_model,
        stream: is_stream_req,
        conversation_key: conversation_key.clone(),
        route: plan_log,
        ..Default::default()
    };

    let mut attempt = 0;
    let mut pre_upstream_us = 0;
    let mut fallback_from: Option<String> = None;
    let resp = loop {
        let chosen = candidates[attempt].clone();
        let body = match (&chosen, &req_json) {
            (Some(m), Some(v)) => {
                let mut v = v.clone();
                v["model"] = serde_json::Value::String(m.clone());
                bytes::Bytes::from(serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec()))
            }
            _ => bytes.clone(),
        };
        let upstream_req = st
            .client
            .request(parts.method.clone(), &url)
            .headers(copy_headers(&parts.headers))
            .body(body);
        if attempt == 0 {
            pre_upstream_us = read_done.elapsed().as_micros();
        }
        let result = upstream_req.send().await;
        let failed = match &result {
            Ok(r) => server_failure(r.status()),
            Err(_) => true,
        };
        if failed && routed {
            if let (Some(rt), Some(m)) = (st.routing.as_ref(), &chosen) {
                rt.mark_unhealthy(m);
            }
        }
        if failed && routed && attempt == 0 && candidates.len() > 1 {
            // Nothing has been streamed to the client yet: retry the next candidate once.
            fallback_from = chosen;
            attempt = 1;
            continue;
        }
        match result {
            Ok(r) => break (r, chosen),
            Err(e) => {
                let msg = format!("upstream: {e}");
                record.status = 502;
                record.error = Some(msg.clone());
                record.pre_upstream_us = pre_upstream_us;
                record.total_ms = started.elapsed().as_millis();
                if let Some(rl) = record.route.as_mut() {
                    rl.chosen_model = chosen;
                    rl.fallback_from = fallback_from;
                }
                emit(&st, &record);
                return error(StatusCode::BAD_GATEWAY, msg);
            }
        }
    };
    let (resp, chosen) = resp;
    let ttfb_ms = started.elapsed().as_millis();

    let status = resp.status();
    if let (Some(rt), Some(m), Some(k)) = (st.routing.as_ref(), &chosen, &conversation_key) {
        if !server_failure(status) {
            rt.remember(k, m);
        }
    }
    let stream = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));

    let mut builder = Response::builder().status(status);
    if let Some(h) = builder.headers_mut() {
        *h = copy_headers(resp.headers());
    }

    if let Some(rl) = record.route.as_mut() {
        rl.chosen_model = chosen;
        rl.fallback_from = fallback_from;
    }
    record.status = status.as_u16();
    record.stream = stream;
    record.pre_upstream_us = pre_upstream_us;
    record.ttfb_ms = ttfb_ms;
    let mut tracker = Tracker {
        state: st.clone(),
        parser: Some(UsageParser::new(stream)),
        started,
        record,
    };

    let body = resp.bytes_stream().map(move |chunk| {
        if let (Ok(b), Some(p)) = (&chunk, tracker.parser.as_mut()) {
            p.feed(b);
        }
        chunk
    });
    builder.body(Body::from_stream(body)).unwrap()
}
