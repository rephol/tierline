//! `tierline summary <log.jsonl>`: totals per model and per conversation.

use crate::LogRecord;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
struct Row {
    requests: u64,
    input: u64,
    output: u64,
    cache_write: u64,
    cache_read: u64,
    cost: f64,
    unpriced: u64,
    models: BTreeSet<String>,
}

pub fn render(log: &str) -> String {
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    let mut convs: BTreeMap<String, Row> = BTreeMap::new();
    let mut bad = 0;
    for line in log.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(r) = serde_json::from_str::<LogRecord>(line) else {
            bad += 1;
            continue;
        };
        let key = r
            .model
            .or(r.requested_model)
            .unwrap_or_else(|| "(none)".into());
        let add = |row: &mut Row| {
            row.requests += 1;
            row.input += r.input_tokens;
            row.output += r.output_tokens;
            row.cache_write += r.cache_creation_tokens;
            row.cache_read += r.cache_read_tokens;
            match r.cost_usd {
                Some(c) => row.cost += c,
                None => row.unpriced += 1,
            }
            row.models.insert(key.clone());
        };
        add(rows.entry(key.clone()).or_default());
        if let Some(ck) = &r.conversation_key {
            add(convs.entry(ck.clone()).or_default());
        }
    }

    let mut out = format!(
        "{:<32} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10}\n",
        "model", "requests", "input", "output", "cache_write", "cache_read", "cost_usd"
    );
    let mut total = Row::default();
    for (model, r) in &rows {
        out += &format!(
            "{:<32} {:>8} {:>12} {:>12} {:>12} {:>12} {:>10.4}{}\n",
            model,
            r.requests,
            r.input,
            r.output,
            r.cache_write,
            r.cache_read,
            r.cost,
            if r.unpriced > 0 {
                format!("  ({} unpriced)", r.unpriced)
            } else {
                String::new()
            }
        );
        total.requests += r.requests;
        total.cost += r.cost;
        total.unpriced += r.unpriced;
    }
    out += &format!("total: {} requests, ${:.4}", total.requests, total.cost);
    if total.unpriced > 0 {
        out += &format!(
            " ({} requests had no price; add the model to pricing.json)",
            total.unpriced
        );
    }
    if !convs.is_empty() {
        out += &format!(
            "\n\n{:<18} {:>8} {:>12} {:>12} {:>10}  {}\n",
            "conversation", "requests", "input", "output", "cost_usd", "models"
        );
        for (k, r) in &convs {
            out += &format!(
                "{:<18} {:>8} {:>12} {:>12} {:>10.4}  {}{}\n",
                k,
                r.requests,
                r.input,
                r.output,
                r.cost,
                r.models.iter().cloned().collect::<Vec<_>>().join(","),
                if r.unpriced > 0 {
                    format!("  ({} unpriced)", r.unpriced)
                } else {
                    String::new()
                }
            );
        }
        out = out.trim_end().to_string();
    }
    if bad > 0 {
        out += &format!("\n{bad} unreadable log lines skipped");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(key: Option<&str>, model: &str, cost: Option<f64>) -> String {
        serde_json::to_string(&LogRecord {
            model: Some(model.into()),
            conversation_key: key.map(String::from),
            input_tokens: 100,
            output_tokens: 10,
            cost_usd: cost,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn groups_by_model_and_by_conversation() {
        let log = [
            line(Some("aaaa"), "m-1", Some(1.0)),
            line(Some("aaaa"), "m-2", Some(0.5)),
            line(Some("bbbb"), "m-1", None),
            line(None, "m-1", Some(0.25)),
            "garbage".into(),
        ]
        .join("\n");
        let out = render(&log);
        let m1 = out.lines().find(|l| l.starts_with("m-1")).unwrap();
        assert!(m1.contains(" 3 "), "{m1}"); // three requests on m-1
        assert!(out.contains("total: 4 requests, $1.7500"), "{out}");
        let a = out.lines().find(|l| l.starts_with("aaaa")).unwrap();
        assert!(
            a.contains(" 2 ") && a.contains("1.5000") && a.contains("m-1,m-2"),
            "{a}"
        );
        let b = out.lines().find(|l| l.starts_with("bbbb")).unwrap();
        assert!(b.contains("(1 unpriced)"), "{b}");
        // The request without a key appears in the model table only.
        assert_eq!(
            out.lines()
                .filter(|l| l.starts_with("aaaa") || l.starts_with("bbbb"))
                .count(),
            2
        );
        assert!(out.contains("1 unreadable log lines skipped"));
    }

    #[test]
    fn no_conversation_table_without_keys() {
        let out = render(&line(None, "m-1", Some(1.0)));
        assert!(!out.contains("conversation"));
    }
}
