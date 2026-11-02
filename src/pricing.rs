//! Per-model prices, loaded from a user-supplied JSON file. No prices are built in.

use crate::usage::Usage;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Price {
    /// USD per million tokens.
    pub input: f64,
    pub output: f64,
    #[serde(default)]
    pub cache_write: f64,
    #[serde(default)]
    pub cache_read: f64,
}

#[derive(Debug, Default)]
pub struct Pricing(HashMap<String, Price>);

impl Pricing {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(s)?;
        let mut map = HashMap::new();
        for (k, v) in raw {
            if k.starts_with('_') {
                continue;
            }
            map.insert(k, serde_json::from_value(v)?);
        }
        Ok(Self(map))
    }

    /// Longest key that prefixes the model name.
    pub fn lookup(&self, model: &str) -> Option<Price> {
        self.0
            .iter()
            .filter(|(k, _)| model.starts_with(k.as_str()))
            .max_by_key(|(k, _)| k.len())
            .map(|(_, p)| *p)
    }

    pub fn cost(&self, u: &Usage) -> Option<f64> {
        let p = self.lookup(u.model.as_deref()?)?;
        let m = 1_000_000.0;
        Some(
            u.input_tokens as f64 * p.input / m
                + u.output_tokens as f64 * p.output / m
                + u.cache_creation_tokens as f64 * p.cache_write / m
                + u.cache_read_tokens as f64 * p.cache_read / m,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_prefix_wins_and_cost_adds_up() {
        let p = Pricing::from_json(
            r#"{"_comment":"x","claude":{"input":1,"output":1},
                "claude-opus":{"input":10,"output":20,"cache_write":5,"cache_read":1}}"#,
        )
        .unwrap();
        let u = Usage {
            model: Some("claude-opus-9".into()),
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            cache_creation_tokens: 200_000,
            cache_read_tokens: 1_000_000,
        };
        assert!((p.cost(&u).unwrap() - (10.0 + 10.0 + 1.0 + 1.0)).abs() < 1e-9);
        assert!(p
            .cost(&Usage {
                model: Some("gpt".into()),
                ..u.clone()
            })
            .is_none());
    }
}
