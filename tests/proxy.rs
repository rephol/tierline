//! End to end: client -> tierline -> mock upstream, checking pass-through and the cost log.

use axum::{
    body::Body,
    http::{header, HeaderMap, Response},
    routing::post,
    Router,
};
use std::sync::Arc;
use tierline::{app, pricing::Pricing, AppState, LogRecord};

const SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-test\",\"usage\":{\"input_tokens\":1000,\"output_tokens\":1}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":500}}\n\n";

async fn serve(router: Router) -> String {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    format!("http://{addr}")
}

async fn mock_messages(headers: HeaderMap, body: String) -> Response<Body> {
    // The proxy must forward auth and strip accept-encoding.
    assert_eq!(headers.get("x-api-key").unwrap(), "sk-test");
    assert!(headers.get("accept-encoding").is_none());
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    if v["stream"] == true {
        let chunks: Vec<Result<String, std::io::Error>> = SSE
            .as_bytes()
            .chunks(5)
            .map(|c| Ok(String::from_utf8_lossy(c).into_owned()))
            .collect();
        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(futures_util::stream::iter(chunks)))
            .unwrap()
    } else {
        Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"model":"claude-opus-test","usage":{"input_tokens":10,"output_tokens":20}}"#,
            ))
            .unwrap()
    }
}

#[tokio::test]
async fn streams_through_and_logs_cost() {
    let upstream = serve(Router::new().route("/v1/messages", post(mock_messages))).await;
    let log = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(AppState {
        upstream,
        client: reqwest::Client::new(),
        pricing: Pricing::from_json(r#"{"claude-opus":{"input":10,"output":50}}"#).unwrap(),
        log: tierline::spawn_log_writer(log.reopen().unwrap()).0,
        routing: None,
    });
    let proxy = serve(app(state)).await;
    let client = reqwest::Client::new();

    for stream in [true, false] {
        let body = client
            .post(format!("{proxy}/v1/messages?beta=true"))
            .header("x-api-key", "sk-test")
            .header("accept-encoding", "gzip")
            .json(&serde_json::json!({"model":"claude-opus-test","stream":stream,"messages":[]}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        if stream {
            assert_eq!(body, SSE, "SSE must pass through byte for byte");
        }
    }

    // The record is written when the response stream is dropped; give it a moment.
    let mut lines = Vec::new();
    for _ in 0..50 {
        lines = std::fs::read_to_string(log.path())
            .unwrap()
            .lines()
            .map(String::from)
            .collect();
        if lines.len() == 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(lines.len(), 2, "log: {lines:?}");

    let s: LogRecord = serde_json::from_str(&lines[0]).unwrap();
    assert!(s.stream);
    assert_eq!(s.path, "/v1/messages?beta=true");
    assert_eq!(s.requested_model.as_deref(), Some("claude-opus-test"));
    assert_eq!((s.input_tokens, s.output_tokens), (1000, 500));
    // 1000 * 10/1M + 500 * 50/1M
    assert!((s.cost_usd.unwrap() - 0.035).abs() < 1e-12);

    let j: LogRecord = serde_json::from_str(&lines[1]).unwrap();
    assert!(!j.stream);
    assert_eq!((j.input_tokens, j.output_tokens), (10, 20));
}

async fn wait_lines(path: &std::path::Path, n: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for _ in 0..100 {
        lines = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(String::from)
            .collect();
        if lines.len() >= n {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    lines
}

#[tokio::test]
async fn failed_upstream_send_is_logged_as_502() {
    // Bind then drop a listener so the port is closed.
    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let log = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(AppState {
        upstream: dead,
        client: reqwest::Client::new(),
        pricing: Pricing::default(),
        log: tierline::spawn_log_writer(log.reopen().unwrap()).0,
        routing: None,
    });
    let proxy = serve(app(state)).await;
    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/messages"))
        .json(&serde_json::json!({"model":"claude-opus-test","messages":[]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);

    let lines = wait_lines(log.path(), 1).await;
    assert_eq!(lines.len(), 1, "log: {lines:?}");
    let r: LogRecord = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(r.status, 502);
    assert!(r.error.as_deref().unwrap().starts_with("upstream:"));
    assert_eq!(r.requested_model.as_deref(), Some("claude-opus-test"));
}

// ---- routing ----

use axum::extract::State;
use tierline::{
    classify::{Category, RuleClassifier},
    models::ModelInfo,
    route::{RouteConfig, Routing},
};

type Seen = Arc<std::sync::Mutex<Vec<String>>>;

/// Records the model of every request; models whose id contains "bad" answer 529.
async fn recording_upstream(State(seen): State<Seen>, body: String) -> Response<Body> {
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let model = v["model"].as_str().unwrap().to_string();
    seen.lock().unwrap().push(model.clone());
    if model.contains("bad") {
        return Response::builder()
            .status(529)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"type":"error","error":{"type":"overloaded_error"}}"#,
            ))
            .unwrap();
    }
    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(format!(
            r#"{{"model":"{model}","usage":{{"input_tokens":10,"output_tokens":20}}}}"#
        )))
        .unwrap()
}

fn model(id: &str, q: f64, tools: bool) -> ModelInfo {
    ModelInfo {
        id: id.into(),
        context_window: 100_000,
        tools,
        images: true,
        thinking: true,
        quality: Category::ALL
            .iter()
            .map(|c| (c.name().to_string(), q))
            .collect(),
    }
}

struct Rig {
    proxy: String,
    seen: Seen,
    log: tempfile::NamedTempFile,
}

async fn rig(models: Vec<ModelInfo>, routing: bool) -> Rig {
    let seen: Seen = Arc::default();
    let upstream = serve(
        Router::new()
            .route("/v1/messages", post(recording_upstream))
            .route("/v1/messages/count_tokens", post(recording_upstream))
            .with_state(seen.clone()),
    )
    .await;
    let log = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(AppState {
        upstream,
        client: reqwest::Client::new(),
        pricing: Pricing::default(),
        log: tierline::spawn_log_writer(log.reopen().unwrap()).0,
        routing: routing
            .then(|| Routing::new(models, RouteConfig::default(), Box::new(RuleClassifier))),
    });
    Rig {
        proxy: serve(app(state)).await,
        seen,
        log,
    }
}

async fn post_json(rig: &Rig, path: &str, v: serde_json::Value) -> u16 {
    reqwest::Client::new()
        .post(format!("{}{path}", rig.proxy))
        .json(&v)
        .send()
        .await
        .unwrap()
        .status()
        .as_u16()
}

fn hello() -> serde_json::Value {
    serde_json::json!({"model":"client-asked","system":"s",
        "messages":[{"role":"user","content":"hello"}]})
}

#[tokio::test]
async fn routing_off_leaves_model_untouched() {
    let r = rig(vec![model("m-a", 0.9, true)], false).await;
    assert_eq!(post_json(&r, "/v1/messages", hello()).await, 200);
    assert_eq!(*r.seen.lock().unwrap(), ["client-asked"]);
    let lines = wait_lines(r.log.path(), 1).await;
    let rec: LogRecord = serde_json::from_str(&lines[0]).unwrap();
    // The conversation key is logged even without routing, for `summary`.
    assert!(rec.route.is_none() && rec.conversation_key.is_some());
}

#[tokio::test]
async fn routing_rewrites_model_and_logs_the_decision() {
    let r = rig(
        vec![model("m-low", 0.3, true), model("m-high", 0.9, true)],
        true,
    )
    .await;
    assert_eq!(post_json(&r, "/v1/messages", hello()).await, 200);
    assert_eq!(*r.seen.lock().unwrap(), ["m-high"]);

    let lines = wait_lines(r.log.path(), 1).await;
    let rec: LogRecord = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(rec.requested_model.as_deref(), Some("client-asked"));
    assert_eq!(rec.model.as_deref(), Some("m-high"));
    assert_eq!(rec.conversation_key.as_ref().unwrap().len(), 16);
    let route = rec.route.unwrap();
    assert_eq!(route.chosen_model.as_deref(), Some("m-high"));
    assert_eq!(route.candidates.len(), 2);
    assert_eq!(route.candidates[0].model, "m-high");
    assert_eq!(route.top_categories.len(), 3);
    let ratings = route.ratings.unwrap();
    assert!((1..=5).contains(&ratings.complexity));
    assert!(route.fallback_from.is_none());
}

#[tokio::test]
async fn filtered_candidates_and_empty_filter_fail_open() {
    // The best model cannot do tools, so the request must go to the other one.
    let r = rig(
        vec![model("m-notools", 0.9, false), model("m-tools", 0.5, true)],
        true,
    )
    .await;
    let mut v = hello();
    v["tools"] = serde_json::json!([{"name":"t","input_schema":{}}]);
    assert_eq!(post_json(&r, "/v1/messages", v).await, 200);
    assert_eq!(r.seen.lock().unwrap().last().unwrap(), "m-tools");

    // Nothing can serve images: the client's own model is used untouched.
    let r = rig(
        vec![ModelInfo {
            images: false,
            ..model("m-x", 0.9, true)
        }],
        true,
    )
    .await;
    let mut v = hello();
    v["messages"] = serde_json::json!([{"role":"user","content":[
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":""}}]}]);
    assert_eq!(post_json(&r, "/v1/messages", v).await, 200);
    assert_eq!(*r.seen.lock().unwrap(), ["client-asked"]);
}

#[tokio::test]
async fn non_message_paths_are_not_rewritten() {
    let r = rig(vec![model("m-a", 0.9, true)], true).await;
    assert_eq!(
        post_json(&r, "/v1/messages/count_tokens", hello()).await,
        200
    );
    assert_eq!(*r.seen.lock().unwrap(), ["client-asked"]);
}

#[tokio::test]
async fn falls_back_once_on_529_and_marks_the_model_unhealthy() {
    let r = rig(
        vec![model("m-bad", 0.9, true), model("m-good", 0.5, true)],
        true,
    )
    .await;
    assert_eq!(post_json(&r, "/v1/messages", hello()).await, 200);
    assert_eq!(*r.seen.lock().unwrap(), ["m-bad", "m-good"]);
    // m-bad is now unhealthy, so the next request skips it entirely.
    assert_eq!(post_json(&r, "/v1/messages", hello()).await, 200);
    assert_eq!(*r.seen.lock().unwrap(), ["m-bad", "m-good", "m-good"]);

    let lines = wait_lines(r.log.path(), 2).await;
    let first: LogRecord = serde_json::from_str(&lines[0]).unwrap();
    let route = first.route.unwrap();
    assert_eq!(first.status, 200);
    assert_eq!(route.chosen_model.as_deref(), Some("m-good"));
    assert_eq!(route.fallback_from.as_deref(), Some("m-bad"));
    let second: LogRecord = serde_json::from_str(&lines[1]).unwrap();
    let route = second.route.unwrap();
    assert!(route.fallback_from.is_none());
    assert_eq!(route.candidates.len(), 1);
}

#[tokio::test]
async fn retries_only_once_then_returns_the_failure() {
    let r = rig(
        vec![
            model("m-bad1", 0.9, true),
            model("m-bad2", 0.8, true),
            model("m-bad3", 0.7, true),
        ],
        true,
    )
    .await;
    assert_eq!(post_json(&r, "/v1/messages", hello()).await, 529);
    assert_eq!(r.seen.lock().unwrap().len(), 2, "one retry, not more");
    let lines = wait_lines(r.log.path(), 1).await;
    let rec: LogRecord = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(rec.status, 529);
}

#[tokio::test]
async fn mid_turn_tool_result_sticks_to_the_previous_model_end_to_end() {
    let r = rig(
        vec![model("m-low", 0.3, true), model("m-high", 0.9, true)],
        true,
    )
    .await;
    let first = hello();
    assert_eq!(post_json(&r, "/v1/messages", first.clone()).await, 200);
    // Same conversation, but now the last message is a tool_result (mid-turn).
    let mut second = first.clone();
    second["messages"] = serde_json::json!([
        {"role":"user","content":"hello"},
        {"role":"assistant","content":[{"type":"tool_use","id":"1","name":"a","input":{}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"ok"}]}]);
    assert_eq!(post_json(&r, "/v1/messages", second).await, 200);
    let lines = wait_lines(r.log.path(), 2).await;
    let a: LogRecord = serde_json::from_str(&lines[0]).unwrap();
    let b: LogRecord = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(a.conversation_key, b.conversation_key);
    let rb = b.route.unwrap();
    assert!(rb.sticky);
    assert_eq!(rb.chosen_model, a.route.unwrap().chosen_model);
}
