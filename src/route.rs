//! Candidate filtering, scoring and ranking, plus the state routing needs across requests
//! (model health and the last model used per conversation).

use crate::{
    classify::{Category, Classification, Classifier, Features, Ratings},
    models::ModelInfo,
    pricing::Pricing,
    tokens::TokenConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Duration, Instant},
};

/// Cap on remembered conversations; the map is cleared when it is exceeded.
const MAX_CONVERSATIONS: usize = 10_000;

#[derive(Debug, Clone)]
pub struct RouteConfig {
    /// How long a model that returned 5xx/529 is skipped.
    pub unhealthy_for: Duration,
    /// Cost weight per complexity rating 1..=5, in utility points per USD. Large for easy
    /// requests, shrinking as complexity rises. Untuned placeholder defaults.
    pub lambda: [f64; 5],
    /// Output tokens assumed for the cost estimate.
    pub est_output_tokens: u64,
    /// Input-token estimate settings, including the context-window margin.
    pub tokens: TokenConfig,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            unhealthy_for: Duration::from_secs(60),
            lambda: [8.0, 4.0, 2.0, 1.0, 0.5],
            est_output_tokens: 1000,
            tokens: TokenConfig::default(),
        }
    }
}

impl RouteConfig {
    pub fn lambda_for(&self, complexity: u8) -> f64 {
        self.lambda[(complexity.clamp(1, 5) - 1) as usize]
    }
}

/// Parses `"8,4,2,1,0.5"` into a lambda curve; values must be finite and non-negative.
pub fn parse_lambda(s: &str) -> Result<[f64; 5], String> {
    let v: Vec<f64> = s
        .split(',')
        .map(|x| {
            x.trim()
                .parse::<f64>()
                .map_err(|e| format!("TIERLINE_LAMBDA: {e}"))
        })
        .collect::<Result<_, _>>()?;
    let arr: [f64; 5] = v
        .try_into()
        .map_err(|_| "TIERLINE_LAMBDA needs exactly 5 values".to_string())?;
    if arr.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("TIERLINE_LAMBDA values must be finite and >= 0".into());
    }
    Ok(arr)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CandidateLog {
    pub model: String,
    pub utility: f64,
    /// False when pricing.json has no entry, so its cost was treated as 0.
    pub priced: bool,
}

/// The routing decision as written to the log.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouteLog {
    pub top_categories: Vec<(String, f64)>,
    pub ratings: Option<Ratings>,
    pub chosen_model: Option<String>,
    pub candidates: Vec<CandidateLog>,
    /// True when the mid-turn rule kept the previous model.
    pub sticky: bool,
    /// Set when the chosen model was reached by falling back from this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_from: Option<String>,
}

pub struct Plan {
    pub conversation_key: String,
    pub classification: Classification,
    /// Best first.
    pub ranked: Vec<CandidateLog>,
    pub sticky: bool,
}

impl Plan {
    pub fn to_log(&self) -> RouteLog {
        RouteLog {
            top_categories: self
                .classification
                .top(3)
                .into_iter()
                .map(|(c, p)| (c.name().to_string(), p))
                .collect(),
            ratings: Some(self.classification.ratings),
            chosen_model: self.ranked.first().map(|c| c.model.clone()),
            candidates: self.ranked.clone(),
            sticky: self.sticky,
            fallback_from: None,
        }
    }
}

fn fnv1a(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for p in parts {
        for b in p.bytes().chain(std::iter::once(0xff)) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Stable key for a conversation: hash of the system prompt and the first user message.
pub fn conversation_key(req: &Value) -> String {
    let system = req.get("system").map(Value::to_string).unwrap_or_default();
    let first_user = req
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|m| {
            m.iter()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        })
        .and_then(|m| m.get("content"))
        .map(Value::to_string)
        .unwrap_or_default();
    format!("{:016x}", fnv1a(&[&system, &first_user]))
}

pub struct Routing {
    pub models: Vec<ModelInfo>,
    pub cfg: RouteConfig,
    classifier: Box<dyn Classifier>,
    health: Mutex<HashMap<String, Instant>>,
    last_model: Mutex<HashMap<String, String>>,
}

impl Routing {
    pub fn new(models: Vec<ModelInfo>, cfg: RouteConfig, classifier: Box<dyn Classifier>) -> Self {
        Self {
            models,
            cfg,
            classifier,
            health: Mutex::default(),
            last_model: Mutex::default(),
        }
    }

    pub fn mark_unhealthy(&self, model: &str) {
        if let Ok(mut h) = self.health.lock() {
            h.insert(model.to_string(), Instant::now());
        }
    }

    pub fn is_unhealthy(&self, model: &str) -> bool {
        self.health
            .lock()
            .ok()
            .and_then(|h| h.get(model).map(|t| t.elapsed() < self.cfg.unhealthy_for))
            .unwrap_or(false)
    }

    pub fn remember(&self, key: &str, model: &str) {
        if let Ok(mut m) = self.last_model.lock() {
            if m.len() >= MAX_CONVERSATIONS {
                m.clear();
            }
            m.insert(key.to_string(), model.to_string());
        }
    }

    pub fn last_model(&self, key: &str) -> Option<String> {
        self.last_model.lock().ok()?.get(key).cloned()
    }

    /// Candidates able to serve the request: capabilities, context window and health.
    pub fn filter(&self, f: &Features, input_tokens: u64) -> Vec<&ModelInfo> {
        self.models
            .iter()
            .filter(|m| {
                (!f.has_images || m.images)
                    && (f.tool_count == 0 || m.tools)
                    && (!f.wants_thinking || m.thinking)
                    && (input_tokens as f64)
                        <= m.context_window as f64 * self.cfg.tokens.context_margin
                    && !self.is_unhealthy(&m.id)
            })
            .collect()
    }

    /// Filters, scores and ranks. `ranked` is empty when no model can serve the request.
    pub fn plan(&self, req: &Value, pricing: &Pricing) -> Plan {
        let f = Features::from_request(req);
        let key = conversation_key(req);
        let input_tokens = self.cfg.tokens.estimate(req);
        let classification = self.classifier.classify(&f);
        let r = classification.ratings;
        let last = self.last_model(&key);

        let new_tokens = self
            .cfg
            .tokens
            .text_len(f.last_user_text.len() + f.last_tool_result_len);
        let ctx_tokens = input_tokens.saturating_sub(new_tokens);
        let lambda = self.cfg.lambda_for(r.complexity);
        let m = 1_000_000.0;

        let mut ranked: Vec<CandidateLog> = self
            .filter(&f, input_tokens)
            .into_iter()
            .map(|model| {
                let raw: f64 = Category::ALL
                    .iter()
                    .map(|c| classification.prob(*c) * model.quality_for(*c))
                    .sum();
                // Weak models lose more as complexity rises.
                let quality = raw.powf(1.0 + 0.25 * (r.complexity as f64 - 1.0));
                let price = pricing.lookup(&model.id);
                let cost = price.map_or(0.0, |p| {
                    let base = new_tokens as f64 * p.input
                        + self.cfg.est_output_tokens as f64 * p.output
                        + ctx_tokens as f64 * p.cache_read;
                    let switch = match &last {
                        Some(l) if *l != model.id => {
                            ctx_tokens as f64 * (p.cache_write - p.cache_read).max(0.0)
                        }
                        _ => 0.0,
                    };
                    (base + switch) / m
                });
                CandidateLog {
                    model: model.id.clone(),
                    utility: quality - lambda * cost,
                    priced: price.is_some(),
                }
            })
            .collect();
        ranked.sort_by(|a, b| b.utility.total_cmp(&a.utility).then(a.model.cmp(&b.model)));

        let mut sticky = false;
        if f.last_is_tool_result {
            if let Some(pos) = last
                .as_ref()
                .and_then(|l| ranked.iter().position(|c| c.model == *l))
            {
                let keep = ranked.remove(pos);
                ranked.insert(0, keep);
                sticky = true;
            }
        }
        Plan {
            conversation_key: key,
            classification,
            ranked,
            sticky,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::RuleClassifier;
    use serde_json::json;

    fn model(id: &str, ctx: u64, q: f64) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            context_window: ctx,
            tools: true,
            images: true,
            thinking: true,
            quality: Category::ALL
                .iter()
                .map(|c| (c.name().to_string(), q))
                .collect(),
        }
    }

    fn pricing() -> Pricing {
        Pricing::from_json(
            r#"{"cheap":{"input":1,"output":5,"cache_write":1.25,"cache_read":0.1},
                "pricey":{"input":10,"output":50,"cache_write":12.5,"cache_read":1}}"#,
        )
        .unwrap()
    }

    fn routing(models: Vec<ModelInfo>) -> Routing {
        Routing::new(models, RouteConfig::default(), Box::new(RuleClassifier))
    }

    fn req(text: &str) -> Value {
        json!({"model":"x","messages":[{"role":"user","content":text}]})
    }

    fn plan_for(r: &Routing, v: &Value) -> Plan {
        r.plan(v, &pricing())
    }

    #[test]
    fn filter_drops_incapable_and_small_context_models() {
        let mut no_tools = model("a", 1000, 0.5);
        no_tools.tools = false;
        let mut no_img = model("b", 1000, 0.5);
        no_img.images = false;
        let mut no_think = model("c", 1000, 0.5);
        no_think.thinking = false;
        let small = model("d", 10, 0.5);
        let ok = model("e", 1000, 0.5);
        let r = routing(vec![no_tools, no_img, no_think, small, ok]);
        let ids = |f: &Features, t| -> Vec<String> {
            r.filter(f, t).iter().map(|m| m.id.clone()).collect()
        };
        assert_eq!(ids(&Features::default(), 100), ["a", "b", "c", "e"]);
        let tools = Features {
            tool_count: 1,
            ..Default::default()
        };
        assert_eq!(ids(&tools, 100), ["b", "c", "e"]);
        let img = Features {
            has_images: true,
            ..Default::default()
        };
        assert_eq!(ids(&img, 100), ["a", "c", "e"]);
        let think = Features {
            wants_thinking: true,
            ..Default::default()
        };
        assert_eq!(ids(&think, 100), ["a", "b", "e"]);
        assert_eq!(ids(&Features::default(), 5000), Vec::<String>::new());
    }

    #[test]
    fn unhealthy_models_are_filtered_until_they_recover() {
        let mut r = routing(vec![model("a", 1000, 0.5), model("b", 1000, 0.5)]);
        r.mark_unhealthy("a");
        assert!(r.is_unhealthy("a"));
        assert_eq!(r.filter(&Features::default(), 1).len(), 1);
        r.cfg.unhealthy_for = Duration::ZERO;
        assert!(!r.is_unhealthy("a"));
        assert_eq!(r.filter(&Features::default(), 1).len(), 2);
    }

    #[test]
    fn conversation_key_depends_on_system_and_first_user_only() {
        let a = json!({"system":"s","messages":[{"role":"user","content":"hi"}]});
        let b = json!({"system":"s","messages":[{"role":"user","content":"hi"},
            {"role":"assistant","content":"yo"},{"role":"user","content":"more"}]});
        let c = json!({"system":"t","messages":[{"role":"user","content":"hi"}]});
        let d = json!({"system":"s","messages":[{"role":"user","content":"other"}]});
        assert_eq!(conversation_key(&a), conversation_key(&b));
        assert_ne!(conversation_key(&a), conversation_key(&c));
        assert_ne!(conversation_key(&a), conversation_key(&d));
        assert_eq!(conversation_key(&a).len(), 16);
    }

    #[test]
    fn easy_requests_prefer_cheap_hard_ones_prefer_quality() {
        // pricey is a bit better but 10x the price.
        let r = routing(vec![
            model("cheap", 100_000, 0.6),
            model("pricey", 100_000, 0.8),
        ]);
        let easy = plan_for(&r, &req("hello"));
        assert_eq!(easy.ranked[0].model, "cheap");
        let long = "x ".repeat(2000);
        let hard = plan_for(&r, &req(&format!("fix the production auth bug {long}")));
        assert!(hard.classification.ratings.complexity >= 4);
        assert_eq!(hard.ranked[0].model, "pricey");
    }

    #[test]
    fn unknown_price_counts_as_zero_cost_and_is_flagged() {
        let r = routing(vec![
            model("mystery", 100_000, 0.5),
            model("cheap", 100_000, 0.5),
        ]);
        let p = plan_for(&r, &req("hello"));
        let mystery = p.ranked.iter().find(|c| c.model == "mystery").unwrap();
        assert!(!mystery.priced);
        assert!(
            (mystery.utility
                - p.ranked
                    .iter()
                    .find(|c| c.model == "cheap")
                    .unwrap()
                    .utility)
                .abs()
                > 0.0
        );
        assert_eq!(p.ranked[0].model, "mystery");
    }

    #[test]
    fn switching_penalty_keeps_the_previous_model() {
        let r = routing(vec![
            model("cheap", 1_000_000, 0.6),
            model("pricey", 1_000_000, 0.9),
        ]);
        let big = "word ".repeat(100_000);
        let v = json!({"model":"x","system":"s","messages":[
            {"role":"user","content":big},
            {"role":"assistant","content":"read it"},
            {"role":"user","content":"plan the architecture"}]});
        // With no cache history the much better model wins this hard, large request.
        let fresh = plan_for(&r, &v);
        assert_eq!(fresh.ranked[0].model, "pricey");
        // Once the conversation is cached on the cheap model, a rewrite of ~125k context
        // tokens at the cache-write price outweighs the quality gain.
        r.remember(&fresh.conversation_key, "cheap");
        let again = plan_for(&r, &v);
        assert_eq!(again.ranked[0].model, "cheap");
        assert!(!again.sticky);
    }

    #[test]
    fn mid_turn_tool_result_sticks_to_previous_model() {
        let r = routing(vec![
            model("cheap", 100_000, 0.1),
            model("pricey", 100_000, 0.9),
        ]);
        let v = json!({"model":"x","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":[{"type":"tool_use","id":"1","name":"a","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"ok"}]}]});
        let key = conversation_key(&v);
        let free = plan_for(&r, &v);
        let free_best = free.ranked[0].model.clone();
        let other = if free_best == "cheap" {
            "pricey"
        } else {
            "cheap"
        };
        r.remember(&key, other);
        let p = plan_for(&r, &v);
        assert!(p.sticky);
        assert_eq!(p.ranked[0].model, other);
        // Unless it was filtered out.
        r.mark_unhealthy(other);
        let p = plan_for(&r, &v);
        assert!(!p.sticky);
        assert!(p.ranked.iter().all(|c| c.model != other));
    }

    #[test]
    fn plan_log_lists_every_candidate_and_ratings() {
        let r = routing(vec![
            model("cheap", 100_000, 0.5),
            model("pricey", 100_000, 0.6),
        ]);
        let l = plan_for(&r, &req("fix this bug")).to_log();
        assert_eq!(l.candidates.len(), 2);
        assert_eq!(l.top_categories.len(), 3);
        assert!(l.ratings.is_some());
        assert_eq!(
            l.chosen_model.as_deref(),
            Some(l.candidates[0].model.as_str())
        );
    }

    #[test]
    fn lambda_parsing() {
        assert_eq!(
            parse_lambda("8, 4,2,1,0.5").unwrap(),
            [8.0, 4.0, 2.0, 1.0, 0.5]
        );
        assert!(parse_lambda("1,2,3").is_err());
        assert!(parse_lambda("1,2,3,4,x").is_err());
        assert!(parse_lambda("1,2,3,4,-1").is_err());
    }

    #[test]
    fn lambda_shrinks_with_complexity_by_default() {
        let c = RouteConfig::default();
        for i in 1..5 {
            assert!(c.lambda_for(i) > c.lambda_for(i + 1));
        }
    }
}
