//! Input-token estimate from the parsed request, with no tokenizer download.
//!
//! Only text is counted: system prompt, message text, tool_use input, tool_result content
//! and tool definitions. JSON syntax is ignored and base64 payloads never reach the text
//! counter; each image (or non-text document) costs a fixed, configurable amount instead.
//! Text is `ascii_chars / chars_per_token` plus one token per non-ASCII char (CJK, emoji
//! and the like tokenize at roughly one token or more per char). It is a heuristic: expect
//! tens of percent of error, which is why the context-window filter has a margin.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct TokenConfig {
    /// ASCII characters per token. 3.5 sits between prose (~4) and code (~3).
    pub chars_per_token: f64,
    /// Tokens charged per image or binary document block.
    pub image_tokens: u64,
    /// The window filter drops a model only when the estimate exceeds
    /// `context_window * context_margin`. Above 1 on purpose: wrongly dropping a model
    /// that could serve is worse than keeping one that cannot (see README).
    pub context_margin: f64,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            chars_per_token: 3.5,
            image_tokens: 1600,
            context_margin: 1.15,
        }
    }
}

/// Fixed cost per message for role and framing.
const MESSAGE_OVERHEAD: f64 = 4.0;

impl TokenConfig {
    /// Tokens for `ascii` ASCII chars plus `other` non-ASCII chars.
    fn count(&self, ascii: usize, other: usize) -> f64 {
        ascii as f64 / self.chars_per_token + other as f64
    }

    /// Tokens for a plain string.
    pub fn text(&self, s: &str) -> f64 {
        let ascii = s.bytes().filter(u8::is_ascii).count();
        let other = s.chars().count() - s.chars().filter(char::is_ascii).count();
        self.count(ascii, other)
    }

    /// Tokens for `len` bytes of text, when only the length is known.
    pub fn text_len(&self, len: usize) -> u64 {
        (len as f64 / self.chars_per_token).ceil() as u64
    }

    /// Text inside arbitrary JSON: strings and object keys count, syntax does not, and
    /// scalars cost one token each.
    fn json(&self, v: &Value) -> f64 {
        match v {
            Value::String(s) => self.text(s),
            Value::Array(a) => a.iter().map(|x| self.json(x)).sum(),
            Value::Object(o) => o.iter().map(|(k, x)| self.text(k) + self.json(x)).sum(),
            Value::Null => 0.0,
            _ => 1.0,
        }
    }

    /// `content` fields: a string, or an array of blocks.
    fn content(&self, v: &Value) -> f64 {
        match v {
            Value::String(s) => self.text(s),
            Value::Array(a) => a.iter().map(|b| self.block(b)).sum(),
            _ => 0.0,
        }
    }

    fn block(&self, b: &Value) -> f64 {
        let text = |k: &str| {
            b.get(k)
                .and_then(Value::as_str)
                .map_or(0.0, |s| self.text(s))
        };
        match b.get("type").and_then(Value::as_str) {
            Some("text") => text("text"),
            Some("thinking") => text("thinking"),
            Some("image") => self.image_tokens as f64,
            Some("document") => match b.get("source") {
                Some(s) if s.get("type").and_then(Value::as_str) == Some("text") => s
                    .get("data")
                    .and_then(Value::as_str)
                    .map_or(0.0, |d| self.text(d)),
                _ => self.image_tokens as f64,
            },
            Some("tool_use") => text("name") + b.get("input").map_or(0.0, |i| self.json(i)),
            Some("tool_result") => b.get("content").map_or(0.0, |c| self.content(c)),
            _ => 0.0,
        }
    }

    /// Estimated input tokens of a Messages API request.
    pub fn estimate(&self, req: &Value) -> u64 {
        let mut t = req.get("system").map_or(0.0, |s| self.content(s));
        if let Some(msgs) = req.get("messages").and_then(Value::as_array) {
            for m in msgs {
                t += MESSAGE_OVERHEAD + m.get("content").map_or(0.0, |c| self.content(c));
            }
        }
        if let Some(tools) = req.get("tools").and_then(Value::as_array) {
            for tool in tools {
                t += MESSAGE_OVERHEAD
                    + ["name", "description"]
                        .iter()
                        .filter_map(|k| tool.get(*k).and_then(Value::as_str))
                        .map(|s| self.text(s))
                        .sum::<f64>()
                    + tool.get("input_schema").map_or(0.0, |s| self.json(s));
            }
        }
        t.ceil() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> TokenConfig {
        TokenConfig::default()
    }

    #[test]
    fn plain_text_uses_chars_per_token() {
        let c = TokenConfig {
            chars_per_token: 4.0,
            ..cfg()
        };
        assert_eq!(c.text(&"a".repeat(400)), 100.0);
        assert_eq!(c.text_len(401), 101);
    }

    #[test]
    fn non_ascii_counts_one_token_per_char() {
        let c = TokenConfig {
            chars_per_token: 4.0,
            ..cfg()
        };
        // 4 ASCII chars = 1 token, 3 CJK chars = 3 tokens (not 9 bytes / 4).
        assert_eq!(c.text("abcd日本語"), 4.0);
    }

    #[test]
    fn counts_system_messages_and_tools() {
        let c = TokenConfig {
            chars_per_token: 1.0,
            ..cfg()
        };
        let req = json!({
            "system": [{"type":"text","text":"sys"}],
            "messages": [
                {"role":"user","content":"hello"},
                {"role":"assistant","content":[{"type":"text","text":"ok"}]}],
            "tools": [{"name":"ab","description":"cd","input_schema":{"type":"object"}}]
        });
        // system 3 + (4+5) + (4+2) + tool (4 + 2 + 2 + key "type" 4 + value "object" 6)
        assert_eq!(c.estimate(&req), 3 + 9 + 6 + 18);
    }

    #[test]
    fn tool_use_input_and_tool_result_are_counted_without_json_syntax() {
        let c = TokenConfig {
            chars_per_token: 1.0,
            ..cfg()
        };
        let req = json!({"messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_xxxxxxxx","name":"rd",
                "input":{"path":"a.rs","n":3,"flag":true}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_xxxxxxxx",
                "content":[{"type":"text","text":"abcde"}]}]}]});
        // msg1: 4 + name 2 + keys (4+1+4) + "a.rs" 4 + 3 -> 1 + true -> 1 = 4+2+9+4+1+1
        // msg2: 4 + 5
        assert_eq!(c.estimate(&req), 21 + 9);
    }

    #[test]
    fn images_cost_a_fixed_amount_and_base64_is_never_counted() {
        let c = TokenConfig {
            image_tokens: 777,
            ..cfg()
        };
        let small = json!({"messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]});
        let huge = json!({"messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png",
             "data":"A".repeat(2_000_000)}}]}]});
        assert_eq!(c.estimate(&small), 777 + 4);
        assert_eq!(c.estimate(&huge), c.estimate(&small));
    }

    #[test]
    fn realistic_body_is_far_below_the_bytes_over_four_guess_when_padded_with_base64() {
        let c = cfg();
        let req = json!({"messages":[{"role":"user","content":[
            {"type":"text","text":"What is in this screenshot?"},
            {"type":"image","source":{"type":"base64","media_type":"image/png",
             "data":"A".repeat(400_000)}}]}]});
        let old = (req.to_string().len() / 4) as u64;
        let new = c.estimate(&req);
        assert!(old > 100_000, "old estimate {old}");
        assert!(new < 2_000, "new estimate {new}");
    }

    #[test]
    fn code_sized_body_lands_near_a_plausible_token_count() {
        // ~3500 chars of code-ish text is about 1000 tokens at 3.5 chars/token.
        let req = json!({"messages":[{"role":"user","content":"fn main() {}\n".repeat(270)}]});
        let n = cfg().estimate(&req);
        assert!((900..=1100).contains(&n), "got {n}");
    }

    #[test]
    fn unknown_or_malformed_parts_count_as_zero() {
        let req = json!({"messages":[{"role":"user","content":42}], "system": null});
        assert_eq!(cfg().estimate(&req), 4);
        assert_eq!(cfg().estimate(&json!({})), 0);
    }
}
