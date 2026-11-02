use std::{env, fs::OpenOptions, sync::Arc};
use tierline::{
    app,
    classify::RuleClassifier,
    pricing::Pricing,
    route::{RouteConfig, Routing},
    summary, AppState,
};

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("summary") => {
            let path = args
                .get(2)
                .cloned()
                .unwrap_or_else(|| env_or("TIERLINE_LOG", "tierline.jsonl"));
            match std::fs::read_to_string(&path) {
                Ok(s) => println!("{}", summary::render(&s)),
                Err(e) => {
                    eprintln!("tierline: cannot read {path}: {e}");
                    std::process::exit(1);
                }
            }
            return;
        }
        Some("-h" | "--help") => {
            println!(
                "tierline: pass-through proxy that logs cost per request\n\n\
                 usage:\n  tierline            run the proxy\n  tierline summary [log.jsonl]\n\n\
                 env:\n  TIERLINE_LISTEN    default 127.0.0.1:8787\n  \
                 TIERLINE_UPSTREAM  default https://api.anthropic.com\n  \
                 TIERLINE_LOG       default tierline.jsonl\n  \
                 TIERLINE_PRICING   default pricing.json (optional; costs are null without it)\n  \
                 TIERLINE_ROUTE     set to 1 to route (needs TIERLINE_MODELS)\n  \
                 TIERLINE_MODELS    default models.json\n  \
                 TIERLINE_UNHEALTHY_SECS     default 60\n  \
                 TIERLINE_LAMBDA             5 comma-separated weights, complexity 1..5\n  \
                 TIERLINE_EST_OUTPUT_TOKENS  default 1000\n  \
                 TIERLINE_CHARS_PER_TOKEN    default 3.5\n  \
                 TIERLINE_IMAGE_TOKENS       default 1600\n  \
                 TIERLINE_CONTEXT_MARGIN     default 1.15"
            );
            return;
        }
        _ => {}
    }

    let listen = env_or("TIERLINE_LISTEN", "127.0.0.1:8787");
    let upstream = env_or("TIERLINE_UPSTREAM", "https://api.anthropic.com");
    let log_path = env_or("TIERLINE_LOG", "tierline.jsonl");
    let pricing_path = env_or("TIERLINE_PRICING", "pricing.json");

    let pricing = match std::fs::read_to_string(&pricing_path) {
        Ok(s) => Pricing::from_json(&s).unwrap_or_else(|e| {
            eprintln!("tierline: bad {pricing_path}: {e}");
            std::process::exit(1);
        }),
        Err(_) => {
            eprintln!("tierline: no {pricing_path}; logging tokens without cost");
            Pricing::default()
        }
    };
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .unwrap_or_else(|e| {
            eprintln!("tierline: cannot open {log_path}: {e}");
            std::process::exit(1);
        });

    let routing = (env_or("TIERLINE_ROUTE", "0") == "1").then(|| build_routing(&pricing));
    let (log_tx, log_task) = tierline::spawn_log_writer(log);
    let state = Arc::new(AppState {
        upstream: upstream.clone(),
        client: reqwest::Client::builder().build().expect("http client"),
        pricing,
        log: log_tx,
        routing,
    });

    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .unwrap_or_else(|e| {
            eprintln!("tierline: cannot bind {listen}: {e}");
            std::process::exit(1);
        });
    eprintln!("tierline: http://{listen} -> {upstream}, logging to {log_path}");
    axum::serve(listener, app(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("server");
    // The server (and its senders) is gone; wait for queued records to reach the file.
    let _ = log_task.await;
}

fn die(msg: String) -> ! {
    eprintln!("tierline: {msg}");
    std::process::exit(1);
}

fn build_routing(pricing: &Pricing) -> Routing {
    let path = env_or("TIERLINE_MODELS", "models.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| die(format!("TIERLINE_ROUTE=1 but cannot read {path}: {e}")));
    let models =
        tierline::models::from_json(&text).unwrap_or_else(|e| die(format!("bad {path}: {e}")));
    if models.is_empty() {
        die(format!("{path} lists no models"));
    }
    let mut cfg = RouteConfig::default();
    if let Ok(v) = env::var("TIERLINE_UNHEALTHY_SECS") {
        let n = v
            .parse()
            .unwrap_or_else(|_| die("bad TIERLINE_UNHEALTHY_SECS".into()));
        cfg.unhealthy_for = std::time::Duration::from_secs(n);
    }
    if let Ok(v) = env::var("TIERLINE_EST_OUTPUT_TOKENS") {
        cfg.est_output_tokens = v
            .parse()
            .unwrap_or_else(|_| die("bad TIERLINE_EST_OUTPUT_TOKENS".into()));
    }
    if let Ok(v) = env::var("TIERLINE_CHARS_PER_TOKEN") {
        cfg.tokens.chars_per_token = v
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite() && *x > 0.0)
            .unwrap_or_else(|| die("bad TIERLINE_CHARS_PER_TOKEN".into()));
    }
    if let Ok(v) = env::var("TIERLINE_IMAGE_TOKENS") {
        cfg.tokens.image_tokens = v
            .parse()
            .unwrap_or_else(|_| die("bad TIERLINE_IMAGE_TOKENS".into()));
    }
    if let Ok(v) = env::var("TIERLINE_CONTEXT_MARGIN") {
        cfg.tokens.context_margin = v
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite() && *x >= 1.0)
            .unwrap_or_else(|| die("bad TIERLINE_CONTEXT_MARGIN (must be >= 1)".into()));
    }
    if let Ok(v) = env::var("TIERLINE_LAMBDA") {
        cfg.lambda = tierline::route::parse_lambda(&v).unwrap_or_else(|e| die(e));
    }
    for m in &models {
        if pricing.lookup(&m.id).is_none() {
            eprintln!(
                "tierline: warning: no price for {}; its cost is treated as 0 when scoring",
                m.id
            );
        }
    }
    eprintln!("tierline: routing over {} models", models.len());
    Routing::new(models, cfg, Box::new(RuleClassifier))
}
