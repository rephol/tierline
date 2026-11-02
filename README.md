# tierline

[![CI](https://github.com/rephol/tierline/actions/workflows/ci.yml/badge.svg)](https://github.com/rephol/tierline/actions/workflows/ci.yml)

> **Status: pre-alpha.** One small pilot benchmark has been run (see [Metrics](#the-metrics-how-this-will-be-judged)).
> Its numbers come from 20 attempts per arm on self-written tasks. Treat them as early
> signal, not proof.

A model-routing proxy for coding agents, written in Rust. Point Claude Code (or any
Anthropic- or OpenAI-compatible agent) at it, and it sends each request to the cheapest
model tier likely to handle it: Haiku, Sonnet or Opus.

```bash
export ANTHROPIC_BASE_URL=http://localhost:8787
claude   # unchanged workflow; tierline picks the model per request
```

## The problem

A coding agent sends every turn to one model. Many turns don't need the strongest one,
for example reading a file, running `ls`, a one-line edit or summarizing tool output.
Paying top-tier prices for those wastes money, but downgrading blindly breaks the hard
turns.

## What it does

1. **Intercepts** each `POST /v1/messages` request, streaming included.
2. **Filters** the candidate models down to ones that can serve it: enough context window,
   tool / image / thinking support, and not recently failing (5xx / 529).
3. **Classifies** the newest turns into task-category probabilities plus 1-5 ratings for
   complexity, ambiguity, stakes and context-dependence.
4. **Scores** every remaining candidate: expected quality for that mix of tasks, minus a
   cost penalty that matters a lot for easy requests and little for hard ones, minus a
   penalty for switching models mid-conversation (it throws away the prompt cache).
5. **Routes** to the best-scoring model, rewriting the `model` field. If that model
   fails with a 5xx / 529 before any bytes are streamed, it retries the next-ranked
   candidate once and marks the failed one unhealthy for a while.
6. **Records** cost, latency and the routing decision of every request.

Routing is opt-in (`TIERLINE_ROUTE=1`); without it the proxy is a pure pass-through.

## Inspiration

The design is a reimplementation, not a fork, of the method described in Cloudflare's
post [Auto Router](https://blog.cloudflare.com/auto-router/): classify the request, filter
the candidate models, score them and pick. The code here is written from scratch for
coding agents on the Anthropic API, and adds cache-aware switching and fallback.

## The metrics (how this will be judged)

The headline metric is **cost per successful task**: total spend on a fixed task set
divided by the number of tasks that passed, set against the same tasks run with every
request sent to Opus. Cost per session alone would reward a router that is cheap because
it fails.

Pilot run on 2026-11-02 with [agentbench](https://github.com/rephol/agentbench#results):
the `hard-v2` set, 10 tasks × 2 samples, Claude Code headless. The dollar figures are
API-equivalent estimates.

| Metric | Pilot result |
|---|---|
| Cost per successful task, compared with all-Opus | $0.204 vs $0.323 (**−37%**) |
| Task success rate, compared with all-Opus | 18/20 vs 18/20 (95% CI 70–100% for both) |
| Overhead the proxy adds, `pre_upstream_us` (routing on) | p50 348 µs, p99 674 µs (94 requests) |
| Requests that needed a fallback retry | 0 of 94 |

**Read with care:**
- **Same success rate:** the only failures (one task, in every arm) look like a flaw in the
  task itself. The run can't show that routing keeps quality, only that it didn't lose any here.
- **Sonnet alone did better:** in the same run it cost $0.174 per success, cheaper than
  routing, with the same success rate. On these tasks a single cheaper model was enough. See
  the [agentbench results](https://github.com/rephol/agentbench#results) for the full
  comparison.
- **Model mix:** the router sent coding-edit and writing requests (71) to Sonnet and kept every
  debugging request (23) on Opus.

**`pre_upstream_us`** is the time from the request body being fully read to the upstream
send being issued. It includes JSON parsing, classification, filtering and scoring, i.e.
everything the proxy adds before the model starts working. It excludes reading the body
from the client and everything after the send (time to first byte, streaming).

**How it's measured:** a separate benchmark repo runs a fixed task set headless
(`claude -p`) through the proxy and directly against the API, on the same tasks and repo
snapshots, and reports cost per successful task and success rate. The numbers are
reproduced there, not in this repo. No result has been measured yet.

## Design

```
agent --> tierline (axum) --> filter --> classify --> score --> rank --> upstream API
              |                                          ^                 |
              |                  health + last-model per conversation      |
              +-- JSONL log (conversation, categories, ratings, utilities) <-- retry next
                                                                     on 5xx / 529
```

- **Rust (axum + tokio + reqwest)** keeps the proxy's own overhead low enough not to matter.
- **Streaming pass-through:** SSE is forwarded chunk by chunk and never buffered.
- **Fails open:** if no candidate survives filtering, or the body is not parseable, the
  request goes to the model the client asked for, untouched.
- **Candidate filter:** `models.json` lists each model's context window, whether it
  supports tools / images / thinking, and a 0-1 quality score per task category. Input tokens
  are estimated from the parsed request, not the body size: text in the system prompt,
  messages, `tool_use` input, `tool_result` content and tool definitions at a configurable
  chars-per-token ratio (JSON syntax and base64 are not counted), plus a fixed cost per
  image. It is a heuristic with no tokenizer download. The window filter drops a model
  only when the estimate exceeds `context_window * 1.15`: the estimate can be off by tens
  of percent, and wrongly dropping a model that could have served the request is worse
  than keeping one that cannot (that case is not retried, since only 5xx / 529 trigger
  a fallback, so the client sees the upstream's error for that request).
- **Classifier:** a `Classifier` trait. The baseline is rule-based and reads mostly the
  newest turns (is the last message a tool result, length of the last user text, keywords,
  code blocks, turn count, tool count). It is a trait so a fine-tuned model can replace
  it later; the rules are the baseline that model must beat.
- **Scoring:** `expected_quality = sum(category_prob * model.quality[category])`, sharpened
  by complexity (low-quality models lose more as complexity rises).
  `utility = expected_quality - lambda(complexity) * (candidate cost / max candidate cost)`,
  where candidate cost includes the cache-switch penalty; with no priced candidate the penalty is 0.
  `lambda` is the quality you would give up for a full saving: 0.30 at complexity 1, then
  0.15, 0.08, 0.04, 0.02. Reasoning: on a trivial request nearly any model is right, so
  trading up to 0.3 quality points for the cheapest option is fine; on a hard one a 2-point
  quality loss is the most a full saving should buy. Chosen from first principles, not fitted to
  benchmark results; the curve is
  configurable.
- **Cache-aware switching:** a conversation key (hash of the system prompt and first user
  message) remembers the last model used. Switching adds a penalty proportional to the
  tokens already in context, priced at the cache-write minus cache-read difference. If the
  last message is a tool result (mid-turn), the previous model is kept unless it was
  filtered out.
- **Fallback:** the scoring produces a ranked list. A 5xx / 529 (or a failed send) before
  any response bytes are streamed retries the next candidate once and marks the failed
  model unhealthy for `TIERLINE_UNHEALTHY_SECS`.

## Trade-offs (to be written up as they're decided)

- A rule-based classifier is cheap and inspectable but crude. How much does a trained one add?
- Prompt caching is per model, so switching loses cache hits. The switch penalty prices
  this in; whether the price is right is an empirical question for the benchmark.
- The quality scores in `models.json` are user-supplied, not measured. Routing is only as
  good as those numbers, and calibrating them from benchmark results is the intended loop.
- The `lambda` curve and the complexity sharpening are untuned defaults.
- Only requests whose models appear in `models.json` are ever chosen; unknown prices count
  as zero cost, which biases toward unpriced models (a warning is logged).

## Running it

The proxy, cost log and a rule-based router exist. Routing is off unless you opt in, and
you supply the model list and prices yourself; no model ids, prices or quality scores are built in.

```bash
cp pricing.example.json pricing.json   # then fill in prices (USD per million tokens)
cp models.example.json models.json     # only for routing; replace every placeholder
cargo run --release                    # listens on 127.0.0.1:8787
```

In another shell:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude
```

Configuration is by environment variable:

| Variable | Default | |
|---|---|---|
| `TIERLINE_LISTEN` | `127.0.0.1:8787` | address to bind |
| `TIERLINE_UPSTREAM` | `https://api.anthropic.com` | upstream API base URL |
| `TIERLINE_LOG` | `tierline.jsonl` | JSONL cost log, one line per request |
| `TIERLINE_PRICING` | `pricing.json` | optional; `cost_usd` is null without it |
| `TIERLINE_ROUTE` | off | `1` turns routing on (needs `models.json`) |
| `TIERLINE_MODELS` | `models.json` | candidates: context window, capabilities, quality per category |
| `TIERLINE_UNHEALTHY_SECS` | `60` | how long a model that returned 5xx/529 is skipped |
| `TIERLINE_LAMBDA` | `0.3,0.15,0.08,0.04,0.02` | quality given up for a full saving, complexity 1..5; untuned |
| `TIERLINE_DUMP_BODIES` | off | debug: directory to write every request body to (contains prompts; keep private) |
| `TIERLINE_EST_OUTPUT_TOKENS` | `1000` | output tokens assumed in the cost estimate |
| `TIERLINE_CHARS_PER_TOKEN` | `3.5` | ASCII chars per token in the input estimate; non-ASCII chars count 1 token each |
| `TIERLINE_IMAGE_TOKENS` | `1600` | tokens charged per image / binary document |
| `TIERLINE_CONTEXT_MARGIN` | `1.15` | filter drops a model only above `context_window * margin` (>= 1) |

Each log line records the model, input/output/cache-creation/cache-read tokens, cost,
`pre_upstream_us`, time to first byte and total time. A failed upstream send is logged
with status 502 and an `error` field. With routing on, a line also carries the
conversation key, the top predicted categories, the four 1-5 ratings, the chosen model,
every candidate's utility (and whether it had a price) and, after a retry, `fallback_from`.
A model missing from `pricing.json` is costed at 0 when scoring and flagged `priced: false`.
Records are written by a single background task, never on the request path.

Summarize a log (totals per model and per conversation key) with:

```bash
cargo run --release -- summary [tierline.jsonl]
```

Run the tests with `cargo test` (end-to-end, against a mock upstream; no API calls).

## Roadmap

- [x] Pass-through proxy with streaming and a cost log
- [x] Rule-based router as the baseline (filter, score, cache-aware switching, fallback)
- [ ] Fine-tuned classifier and a comparison against the rules
- [x] Pilot benchmark run (agentbench, 2026-11-02)
- [ ] Larger benchmark with tasks that separate model quality

## License

MIT
