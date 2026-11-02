//! Extracts model and token usage from an upstream response body as it streams past.

use serde_json::Value;

/// Cap on how much of a non-streaming body is kept for parsing.
const MAX_JSON_BUF: usize = 8 * 1024 * 1024;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Usage {
    pub model: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
}

impl Usage {
    fn absorb(&mut self, u: &Value) {
        let get = |k: &str| u.get(k).and_then(Value::as_u64);
        // Values in a later event are cumulative, so take the max rather than summing.
        if let Some(v) = get("input_tokens") {
            self.input_tokens = self.input_tokens.max(v);
        }
        if let Some(v) = get("output_tokens") {
            self.output_tokens = self.output_tokens.max(v);
        }
        if let Some(v) = get("cache_creation_input_tokens") {
            self.cache_creation_tokens = self.cache_creation_tokens.max(v);
        }
        if let Some(v) = get("cache_read_input_tokens") {
            self.cache_read_tokens = self.cache_read_tokens.max(v);
        }
    }
}

/// Feeds on response chunks; understands Anthropic SSE streams and plain JSON bodies.
#[derive(Debug)]
pub struct UsageParser {
    sse: bool,
    buf: Vec<u8>,
    overflowed: bool,
    usage: Usage,
}

impl UsageParser {
    pub fn new(sse: bool) -> Self {
        Self {
            sse,
            buf: Vec::new(),
            overflowed: false,
            usage: Usage::default(),
        }
    }

    pub fn feed(&mut self, chunk: &[u8]) {
        if self.sse {
            self.buf.extend_from_slice(chunk);
            while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=pos).collect();
                self.sse_line(&line);
            }
        } else if !self.overflowed {
            if self.buf.len() + chunk.len() > MAX_JSON_BUF {
                self.overflowed = true;
                self.buf.clear();
            } else {
                self.buf.extend_from_slice(chunk);
            }
        }
    }

    fn sse_line(&mut self, line: &[u8]) {
        let Ok(line) = std::str::from_utf8(line) else {
            return;
        };
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            return;
        };
        let Ok(ev) = serde_json::from_str::<Value>(data.trim()) else {
            return;
        };
        match ev.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(msg) = ev.get("message") {
                    self.take_model(msg);
                    if let Some(u) = msg.get("usage") {
                        self.usage.absorb(u);
                    }
                }
            }
            Some("message_delta") => {
                if let Some(u) = ev.get("usage") {
                    self.usage.absorb(u);
                }
            }
            _ => {}
        }
    }

    fn take_model(&mut self, v: &Value) {
        if let Some(m) = v.get("model").and_then(Value::as_str) {
            self.usage.model = Some(m.to_string());
        }
    }

    pub fn finish(mut self) -> Usage {
        if self.sse {
            if !self.buf.is_empty() {
                let rest = std::mem::take(&mut self.buf);
                self.sse_line(&rest);
            }
        } else if !self.overflowed {
            if let Ok(v) = serde_json::from_slice::<Value>(&self.buf) {
                self.take_model(&v);
                if let Some(u) = v.get("usage") {
                    self.usage.absorb(u);
                }
            }
        }
        self.usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-sonnet-x\",\"usage\":{\"input_tokens\":120,\"cache_read_input_tokens\":900,\"cache_creation_input_tokens\":30,\"output_tokens\":1}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n";

    #[test]
    fn parses_sse_split_across_arbitrary_chunks() {
        for size in [1, 3, 7, 64, SSE.len()] {
            let mut p = UsageParser::new(true);
            for c in SSE.as_bytes().chunks(size) {
                p.feed(c);
            }
            let u = p.finish();
            assert_eq!(
                u.model.as_deref(),
                Some("claude-sonnet-x"),
                "chunk size {size}"
            );
            assert_eq!(u.input_tokens, 120);
            assert_eq!(u.output_tokens, 42);
            assert_eq!(u.cache_read_tokens, 900);
            assert_eq!(u.cache_creation_tokens, 30);
        }
    }

    #[test]
    fn parses_plain_json() {
        let body = br#"{"model":"claude-haiku-y","usage":{"input_tokens":5,"output_tokens":7}}"#;
        let mut p = UsageParser::new(false);
        p.feed(&body[..10]);
        p.feed(&body[10..]);
        let u = p.finish();
        assert_eq!(u.model.as_deref(), Some("claude-haiku-y"));
        assert_eq!((u.input_tokens, u.output_tokens), (5, 7));
    }

    #[test]
    fn garbage_yields_empty_usage() {
        let mut p = UsageParser::new(false);
        p.feed(b"not json");
        assert_eq!(p.finish(), Usage::default());
    }
}
