//! Candidate models, loaded from a user-supplied JSON file. Nothing is built in.

use crate::classify::Category;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub context_window: u64,
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub images: bool,
    #[serde(default)]
    pub thinking: bool,
    /// 0-1 quality per task category, keyed by `Category::name`. Missing means 0.
    #[serde(default)]
    pub quality: HashMap<String, f64>,
}

impl ModelInfo {
    pub fn quality_for(&self, c: Category) -> f64 {
        self.quality.get(c.name()).copied().unwrap_or(0.0)
    }
}

#[derive(Debug, Default, Deserialize)]
struct File {
    models: Vec<ModelInfo>,
}

pub fn from_json(s: &str) -> Result<Vec<ModelInfo>, String> {
    let f: File = serde_json::from_str(s).map_err(|e| e.to_string())?;
    for m in &f.models {
        for (k, v) in &m.quality {
            if Category::from_name(k).is_none() {
                return Err(format!("{}: unknown category {k:?}", m.id));
            }
            if !(0.0..=1.0).contains(v) {
                return Err(format!("{}: quality for {k} must be in 0..=1", m.id));
            }
        }
    }
    Ok(f.models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates() {
        let ms = from_json(
            r#"{"_c":"x","models":[{"id":"a","context_window":10,"tools":true,
                "quality":{"planning":0.7}}]}"#,
        )
        .unwrap();
        assert_eq!(ms[0].quality_for(Category::Planning), 0.7);
        assert_eq!(ms[0].quality_for(Category::Writing), 0.0);
        assert!(!ms[0].images);
        assert!(
            from_json(r#"{"models":[{"id":"a","context_window":1,"quality":{"nope":1}}]}"#)
                .is_err()
        );
        assert!(from_json(
            r#"{"models":[{"id":"a","context_window":1,"quality":{"planning":2}}]}"#
        )
        .is_err());
    }

    #[test]
    fn example_file_parses() {
        let ms = from_json(include_str!("../models.example.json")).unwrap();
        assert_eq!(ms.len(), 3);
    }
}
