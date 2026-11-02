//! Request classification: task-category probabilities plus 1-5 difficulty ratings.
//!
//! `Classifier` is a trait so a trained model can replace the rule-based baseline.

use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    CodingEdit,
    CodingExplore,
    Planning,
    Debugging,
    Refactoring,
    Testing,
    CodeReview,
    Research,
    Writing,
    DataAnalysis,
    Devops,
    ChitchatToolPlumbing,
}

impl Category {
    pub const ALL: [Category; 12] = [
        Category::CodingEdit,
        Category::CodingExplore,
        Category::Planning,
        Category::Debugging,
        Category::Refactoring,
        Category::Testing,
        Category::CodeReview,
        Category::Research,
        Category::Writing,
        Category::DataAnalysis,
        Category::Devops,
        Category::ChitchatToolPlumbing,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Category::CodingEdit => "coding_edit",
            Category::CodingExplore => "coding_explore",
            Category::Planning => "planning",
            Category::Debugging => "debugging",
            Category::Refactoring => "refactoring",
            Category::Testing => "testing",
            Category::CodeReview => "code_review",
            Category::Research => "research",
            Category::Writing => "writing",
            Category::DataAnalysis => "data_analysis",
            Category::Devops => "devops",
            Category::ChitchatToolPlumbing => "chitchat_tool_plumbing",
        }
    }

    pub fn from_name(s: &str) -> Option<Category> {
        Category::ALL.into_iter().find(|c| c.name() == s)
    }
}

/// 1-5 ratings (1 = low, 5 = high).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Ratings {
    pub complexity: u8,
    pub ambiguity: u8,
    pub stakes: u8,
    pub context_dependence: u8,
}

#[derive(Debug, Clone)]
pub struct Classification {
    /// One probability per `Category::ALL`, summing to 1.
    pub probs: [f64; 12],
    pub ratings: Ratings,
}

impl Classification {
    pub fn prob(&self, c: Category) -> f64 {
        self.probs[Category::ALL.iter().position(|x| *x == c).unwrap()]
    }

    /// The `n` most likely categories, most likely first.
    pub fn top(&self, n: usize) -> Vec<(Category, f64)> {
        let mut v: Vec<_> = Category::ALL.iter().copied().zip(self.probs).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.truncate(n);
        v
    }
}

/// What a classifier gets to look at, extracted once from the request body.
#[derive(Debug, Default, Clone)]
pub struct Features {
    pub turns: usize,
    pub last_is_tool_result: bool,
    /// Text of the newest user message, excluding tool results.
    pub last_user_text: String,
    /// Total characters of the newest tool_result content, if the last message has one.
    pub last_tool_result_len: usize,
    pub tool_count: usize,
    pub has_images: bool,
    pub wants_thinking: bool,
}

fn block_text(b: &Value) -> Option<&str> {
    match b.get("type").and_then(Value::as_str) {
        Some("text") => b.get("text").and_then(Value::as_str),
        _ => None,
    }
}

fn content_len(v: &Value) -> usize {
    match v {
        Value::String(s) => s.len(),
        Value::Array(a) => a
            .iter()
            .map(|b| block_text(b).map(str::len).unwrap_or(0))
            .sum(),
        _ => 0,
    }
}

fn is_injected(text: &str) -> bool {
    text.trim_start().starts_with("<system-reminder>")
}

fn user_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) if !is_injected(s) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(block_text)
            .filter(|t| !is_injected(t))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

impl Features {
    pub fn from_request(req: &Value) -> Features {
        let msgs = req
            .get("messages")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let mut f = Features {
            turns: msgs.len(),
            tool_count: req
                .get("tools")
                .and_then(Value::as_array)
                .map_or(0, Vec::len),
            wants_thinking: req
                .get("thinking")
                .and_then(|t| t.get("type"))
                .and_then(Value::as_str)
                .is_some_and(|t| t != "disabled"),
            ..Default::default()
        };
        let blocks = |m: &Value| -> Vec<Value> {
            match m.get("content") {
                Some(Value::Array(a)) => a.clone(),
                _ => Vec::new(),
            }
        };
        for m in msgs {
            if blocks(m)
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("image"))
            {
                f.has_images = true;
            }
        }
        if let Some(last) = msgs.last() {
            for b in blocks(last) {
                if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                    f.last_is_tool_result = true;
                    f.last_tool_result_len += b.get("content").map_or(0, content_len);
                }
            }
        }
        // Newest user message that carries real text. Tool-result-only messages and
        // harness-injected `<system-reminder>` blocks are not what the user asked for.
        f.last_user_text = msgs
            .iter()
            .rev()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .map(|m| user_text(m.get("content")))
            .find(|t| !t.trim().is_empty())
            .unwrap_or_default();
        f
    }
}

pub trait Classifier: Send + Sync {
    fn classify(&self, f: &Features) -> Classification;
}

/// Hand-written rules. Crude on purpose: the baseline a trained classifier must beat.
#[derive(Debug, Default)]
pub struct RuleClassifier;

fn idx(c: Category) -> usize {
    Category::ALL.iter().position(|x| *x == c).unwrap()
}

struct Text {
    lc: String,
    words: HashSet<String>,
}

impl Text {
    fn new(s: &str) -> Self {
        let lc = s.to_lowercase();
        let words = lc
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect();
        Text { lc, words }
    }

    /// Single words match whole words; entries containing a space match as substrings.
    fn has(&self, kws: &[&str]) -> bool {
        kws.iter().any(|k| {
            if k.contains(' ') {
                self.lc.contains(k)
            } else {
                self.words.contains(*k)
            }
        })
    }
}

impl Classifier for RuleClassifier {
    fn classify(&self, f: &Features) -> Classification {
        use Category::*;
        let t = Text::new(&f.last_user_text);
        let len = f.last_user_text.chars().count();
        let code_blocks = f.last_user_text.matches("```").count() / 2;

        // Small prior so every category keeps some probability.
        let mut s = [0.2f64; 12];
        let mut add = |c: Category, w: f64| s[idx(c)] += w;

        if f.last_is_tool_result {
            add(ChitchatToolPlumbing, 3.0);
            add(CodingExplore, 1.0);
            if f.last_tool_result_len < 200 {
                add(ChitchatToolPlumbing, 1.0);
            }
        }
        if t.has(&[
            "fix",
            "bug",
            "error",
            "errors",
            "fails",
            "failing",
            "panic",
            "traceback",
            "exception",
            "crash",
            "broken",
            "stack trace",
        ]) {
            add(Debugging, 2.5);
        }
        if t.has(&[
            "implement",
            "add",
            "change",
            "update",
            "edit",
            "modify",
            "create",
            "write a function",
        ]) {
            add(CodingEdit, 2.0);
        }
        if t.has(&[
            "refactor", "rename", "extract", "clean up", "cleanup", "simplify",
        ]) {
            add(Refactoring, 2.5);
        }
        if t.has(&["test", "tests", "pytest", "coverage", "unit test"]) {
            add(Testing, 2.5);
        }
        if t.has(&[
            "plan",
            "design",
            "architecture",
            "approach",
            "proposal",
            "roadmap",
            "strategy",
        ]) {
            add(Planning, 2.5);
        }
        if t.has(&["review", "pr", "diff", "critique"]) {
            add(CodeReview, 2.0);
        }
        if t.has(&[
            "explain",
            "where is",
            "how does",
            "find",
            "read",
            "look at",
            "show me",
            "what does",
            "list",
        ]) {
            add(CodingExplore, 2.0);
        }
        if t.has(&[
            "research",
            "compare",
            "survey",
            "latest",
            "docs",
            "documentation",
            "options",
        ]) {
            add(Research, 2.0);
        }
        if t.has(&[
            "draft",
            "email",
            "readme",
            "blog",
            "essay",
            "summarize",
            "rewrite",
            "changelog",
        ]) {
            add(Writing, 2.0);
        }
        if t.has(&[
            "analyze",
            "analyse",
            "csv",
            "dataset",
            "chart",
            "sql",
            "statistics",
            "query",
            "dataframe",
        ]) {
            add(DataAnalysis, 2.5);
        }
        if t.has(&[
            "deploy",
            "docker",
            "ci",
            "kubernetes",
            "config",
            "install",
            "build",
            "pipeline",
            "terraform",
        ]) {
            add(Devops, 2.0);
        }
        if !f.last_is_tool_result
            && len < 40
            && t.has(&[
                "thanks",
                "thank you",
                "ok",
                "okay",
                "hi",
                "hello",
                "yes",
                "no",
                "lgtm",
                "continue",
                "go ahead",
            ])
        {
            add(ChitchatToolPlumbing, 3.0);
        }
        if code_blocks > 0 {
            add(CodingEdit, 1.0);
            add(Debugging, 0.5);
        }

        let total: f64 = s.iter().sum();
        let probs = s.map(|x| x / total);
        let top = {
            let mut v: Vec<_> = Category::ALL.iter().copied().zip(probs).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            v[0].0
        };

        let clamp = |x: i32| x.clamp(1, 5) as u8;

        let mut complexity = 2;
        if len > 800 {
            complexity += 1;
        }
        if len > 3000 {
            complexity += 1;
        }
        if matches!(top, Debugging | Planning | Refactoring | CodeReview) {
            complexity += 1;
        }
        if f.tool_count > 15 && !f.last_is_tool_result {
            complexity += 1;
        }
        if f.wants_thinking {
            complexity += 1;
        }
        if f.last_is_tool_result && f.last_tool_result_len < 2000 {
            complexity -= 1;
        }
        if top == ChitchatToolPlumbing {
            complexity -= 1;
        }

        let mut ambiguity = 2;
        if f.last_is_tool_result {
            ambiguity -= 1;
        } else if len < 40 {
            ambiguity += 1;
        }
        if t.has(&[
            "something",
            "somehow",
            "maybe",
            "not sure",
            "any idea",
            "whatever",
            "best way",
        ]) || f.last_user_text.matches('?').count() > 1
        {
            ambiguity += 1;
        }
        if len > 800 || code_blocks > 0 {
            ambiguity -= 1;
        }

        let mut stakes = 2;
        if t.has(&[
            "production",
            "prod",
            "deploy",
            "migration",
            "delete",
            "drop",
            "security",
            "auth",
            "payment",
            "payments",
            "billing",
            "secret",
            "secrets",
            "credentials",
            "customer",
            "customers",
        ]) {
            stakes += 2;
        }
        if f.last_is_tool_result {
            stakes -= 1;
        }

        let mut ctx = 2;
        if f.turns > 10 {
            ctx += 1;
        }
        if f.turns > 30 {
            ctx += 1;
        }
        if len < 80
            && t.has(&[
                "this", "that", "it", "above", "earlier", "previous", "again", "same",
            ])
        {
            ctx += 1;
        }
        if f.last_is_tool_result {
            ctx += 1;
        }

        Classification {
            probs,
            ratings: Ratings {
                complexity: clamp(complexity),
                ambiguity: clamp(ambiguity),
                stakes: clamp(stakes),
                context_dependence: clamp(ctx),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn classify(req: Value) -> Classification {
        RuleClassifier.classify(&Features::from_request(&req))
    }

    fn top(c: &Classification) -> Category {
        c.top(1)[0].0
    }

    #[test]
    fn category_names_round_trip() {
        for c in Category::ALL {
            assert_eq!(Category::from_name(c.name()), Some(c));
        }
    }

    #[test]
    fn probabilities_sum_to_one() {
        let c = classify(json!({"messages":[{"role":"user","content":"hello"}]}));
        assert!((c.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn features_read_newest_turns() {
        let f = Features::from_request(&json!({
        "tools":[{"name":"a"},{"name":"b"}],
        "thinking":{"type":"enabled","budget_tokens":1000},
        "messages":[
            {"role":"user","content":[{"type":"image","source":{}},{"type":"text","text":"look"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"1","name":"a","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"ok"}]}
        ]}));
        assert_eq!(f.turns, 3);
        assert!(f.last_is_tool_result && f.has_images && f.wants_thinking);
        assert_eq!(f.tool_count, 2);
        assert_eq!(f.last_tool_result_len, 2);
        // The newest message is only a tool_result, so the text comes from the newest
        // user message that has some.
        assert_eq!(f.last_user_text, "look");
    }

    // Synthetic body shaped like Claude Code's: injected reminder blocks ahead of the
    // real prompt, and a tool loop afterwards.
    fn cc_body(with_tool_turn: bool) -> Value {
        let mut msgs = vec![json!({"role":"user","content":[
            {"type":"text","text":"<system-reminder>\nProject notes. Plan the roadmap, design the architecture, review the PR, deploy to production.\n</system-reminder>"},
            {"type":"text","text":"<system-reminder>\nMore harness context about tests and config.\n</system-reminder>"},
            {"type":"text","text":"hello there"}]})];
        if with_tool_turn {
            msgs.push(json!({"role":"assistant","content":[{"type":"tool_use","id":"1","name":"Read","input":{}}]}));
            msgs.push(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"ok"}]}));
        }
        json!({"system":"You are a coding agent. Plan carefully, design well.","messages":msgs})
    }

    #[test]
    fn system_reminders_and_system_prompt_are_ignored() {
        let f = Features::from_request(&cc_body(false));
        assert_eq!(f.last_user_text, "hello there");
        let c = RuleClassifier.classify(&f);
        assert!(
            c.prob(Category::Planning) < 0.15,
            "planning {}",
            c.prob(Category::Planning)
        );
        assert!(c.ratings.complexity <= 2);
    }

    #[test]
    fn tool_loop_turns_still_see_the_real_prompt_not_reminders() {
        let f = Features::from_request(&cc_body(true));
        assert!(f.last_is_tool_result);
        assert_eq!(f.last_user_text, "hello there");
    }

    #[test]
    fn tool_result_turn_is_plumbing_and_easy() {
        let c = classify(json!({"messages":[
            {"role":"user","content":"fix the failing test"},
            {"role":"assistant","content":[{"type":"tool_use","id":"1","name":"a","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"ok"}]}]}));
        assert_eq!(top(&c), Category::ChitchatToolPlumbing);
        assert_eq!(c.ratings.complexity, 1);
    }

    #[test]
    fn keywords_pick_categories() {
        let p = |t: &str| top(&classify(json!({"messages":[{"role":"user","content":t}]})));
        assert_eq!(
            p("the build panics with this stack trace, please fix"),
            Category::Debugging
        );
        assert_eq!(
            p("write a design and plan for the new cache"),
            Category::Planning
        );
        assert_eq!(p("add unit tests with pytest coverage"), Category::Testing);
        assert_eq!(p("thanks"), Category::ChitchatToolPlumbing);
        assert_eq!(p("analyze this csv dataset"), Category::DataAnalysis);
    }

    #[test]
    fn ratings_respond_to_text() {
        let long = "x ".repeat(2000);
        let hard = classify(json!({"messages":[{"role":"user","content":
            format!("fix the production auth bug. {long}")}]}));
        let easy = classify(json!({"messages":[{"role":"user","content":"hello"}]}));
        assert!(hard.ratings.complexity > easy.ratings.complexity);
        assert!(hard.ratings.stakes > easy.ratings.stakes);
        assert!(easy.ratings.ambiguity > hard.ratings.ambiguity);
        for r in [hard.ratings, easy.ratings] {
            for v in [r.complexity, r.ambiguity, r.stakes, r.context_dependence] {
                assert!((1..=5).contains(&v));
            }
        }
    }

    #[test]
    fn long_conversations_raise_context_dependence() {
        let msgs: Vec<Value> = (0..40)
            .map(|i| json!({"role": if i % 2 == 0 {"user"} else {"assistant"}, "content":"x"}))
            .collect();
        let long = classify(json!({"messages":msgs}));
        let short = classify(json!({"messages":[{"role":"user","content":"x"}]}));
        assert!(long.ratings.context_dependence > short.ratings.context_dependence);
    }
}
