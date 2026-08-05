# general-agent

A generic [LangChain 1.0](https://docs.langchain.com/oss/python/langchain/overview) agent
(`create_agent`) with two things a bare quickstart doesn't have:

- **Rate-limit failover** — a custom middleware
  ([`bug-ss/ratelimit-fallback`](https://github.com/bug-ss/ratelimit-fallback)) that
  switches to the next model and retries whenever a provider answers HTTP 429.
- **Langfuse tracing** — every run is one trace, with the failover visible as a
  first-class step rather than an invisible retry.

```bash
pip install -e ".[all]"
cp .env.example .env        # fill in your keys
general-agent "What is 17 * 23?"
```

## What it does

`general-agent` is a starting point, not a finished product: a tool-using agent
with a small, safe default tool set that you extend in `src/general_agent/tools.py`.

| Tool | Purpose |
| --- | --- |
| `calculator` | Arithmetic via a restricted AST evaluator — no `eval`, no names, no calls |
| `current_datetime` | The real date/time in any IANA timezone, since the model's own is stale |
| `tavily_search` | Web search, added only when `langchain-tavily` and `TAVILY_API_KEY` are both present |

Two entry points:

```bash
general-agent "What is 17 * 23?"   # one question, then exit
general-agent                      # interactive chat, with :up / :down feedback
```

```python
from general_agent import AgentRunner

with AgentRunner() as runner:                     # flushes traces on exit
    reply = runner.ask("What is 17 * 23?")
    print(reply.text, reply.model, reply.trace_id)
    runner.feedback(reply, positive=True)
```

## Install

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # both providers + web search
# or pick what you need:
pip install -e ".[openrouter]"
pip install -e ".[google]"
```

The middleware is pulled straight from GitHub, so `pip` needs `git` available.
It is **pinned to a commit** rather than tracking a branch: the project publishes
no tags or releases yet and its version has stayed `0.1.0` across changes, so an
unpinned URL would install whatever `HEAD` happens to be — two people running the
same command on different days would get different code, with nothing in the
package metadata to tell them apart.

The cost of a pin is that an upstream fix never arrives on its own, so CI
watches for one — see [Keeping the pin current](#keeping-the-pin-current).

## Configuration

Everything is environment-driven; `Settings.from_env()` reads it once at startup.
Copy `.env.example` to `.env` — it is gitignored, and `load_dotenv()` runs before
anything reads the environment.

### Models

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_MODELS` | `openrouter:openai/gpt-4o-mini`, `openrouter:meta-llama/llama-3.3-70b-instruct`, `google:gemini-flash-latest` | Comma-separated failover chain, best model first |
| `AGENT_SYSTEM_PROMPT` | a generic assistant prompt | Agent instructions |
| `AGENT_NAME` | `general-agent` | Graph name |

The **first entry is the agent's own model**; the rest are what a 429 falls back
to. Crossing providers matters: rate limits are usually metered per *account*, so
a chain of three models on one key fails into the same wall three times.

Provider keys: `OPENROUTER_API_KEY`, `GOOGLE_API_KEY` (see
[the middleware's README](https://github.com/bug-ss/ratelimit-fallback#configuration)
for the full list, including OpenRouter attribution headers).

### Failover behaviour

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_COOLDOWN_SECONDS` | `60` | How long a 429'd model is deprioritised, so later steps in the same run skip it |
| `AGENT_ACCOUNT_COOLDOWN` | `false` | Cool down every route sharing a credential, not just the model that failed — set this when limits are account-wide |
| `AGENT_RESPECT_RETRY_AFTER` | `false` | Sleep for the provider's `Retry-After` hint before the next attempt |
| `AGENT_POOL_OPENROUTER_ACCOUNTS` | `false` | Route each OpenRouter model over several credentials |
| `AGENT_OPENROUTER_KEY_VARS` | `OPENROUTER_API_KEY` | Which env vars hold those credentials |

Account pooling only engages when at least two credentials are actually found —
one key is not a pool, and the extra indirection would just obscure the chain in
logs and traces.

### Observability

| Variable | Default | Meaning |
| --- | --- | --- |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | **Both** required; tracing stays off with only one |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Must match your project's region, or be your self-hosted URL |
| `AGENT_TRACING_ENABLED` | `true` | Turn tracing off without removing credentials |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | Fraction of traces to send; lower under load |
| `APP_ENV` | `development` | Langfuse environment — keeps test traces out of production dashboards |
| `APP_RELEASE` | — | Version on every trace, so a regression ties to a deploy |
| `AGENT_TAGS` | — | Comma-separated trace tags to break metrics down by |
| `USER_ID` | `anonymous` | Default end user for cost and quality attribution |

### Tools

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_ENABLE_WEB_SEARCH` | `true` | Include Tavily search when available |
| `AGENT_MAX_SEARCH_RESULTS` | `5` | Results per search call |
| `TAVILY_API_KEY` | — | Without it, search is silently left out of the tool set |

## How the failover middleware fits in

The middleware hooks `wrap_model_call`, not `after_model`. That is the whole
trick: a 429 raises *inside* the model call, so no message is produced and
`after_model` never runs. `wrap_model_call` wraps the call, sees the exception,
and can invoke the handler again against a different model.

```python
RateLimitFallbackMiddleware(
    models=build_model_chain(settings),
    try_request_model_first=False,   # the chain already starts with the agent's model
    cooldown_seconds=60.0,
    on_rate_limit=observability.on_rate_limit,   # record the failover on the trace
    generation_name="generate-response",         # stable observation name
)
```

Any non-429 error propagates untouched. When the whole chain is exhausted the
last 429 is re-raised, annotated with every model that was tried.

## Tracing

Tracing follows the Langfuse [instrumentation
guidance](https://langfuse.com/docs/observability/overview) and uses the
[LangChain integration](https://langfuse.com/integrations/frameworks/langchain)
rather than hand-rolled spans — the integration captures model names, token
usage, tool calls and their nesting for free, and gets more context than manual
instrumentation would.

A run whose primary model is rate limited, and which then calls a tool, emits
exactly this (captured from a real run against an in-memory exporter):

```
answer-question                (agent)       input: the question, output: the answer
└─ general-agent               (agent)       the LangGraph agent
   ├─ model                    (chain)
   │  ├─ generate-response     (generation)  model=openai/gpt-4o-mini  level=ERROR
   │  ├─ model-rate-limited    (event)       level=WARNING  metadata: next_model, retry_after…
   │  └─ generate-response     (generation)  model=gemini-flash-latest → tool call
   ├─ tools                    (chain)
   │  └─ calculator            (tool)        {"expression": "17 * 23"} → 391
   └─ model                    (chain)
      └─ generate-response     (generation)  model=gemini-flash-latest → the answer
```

What each piece buys you:

- **Root `agent` observation.** Left to itself, the LangChain handler makes the
  raw LangGraph state dict the trace's input and output — a JSON blob no
  evaluator can read. `AgentRunner.ask` sets the input to the user's question and
  the output to the answer, and types the root as `agent` so it renders as a node
  in the Agent Graph.
- **Stable observation names.** By default a generation is named after the model
  that served it, which under failover changes from trace to trace and breaks
  every filter, dashboard and evaluator targeting it. Names here are fixed
  (`answer-question`, `generate-response`); the model is recorded as the
  generation's `model` attribute, where cost and token accounting expect it.

  The middleware delivers this via `generation_name`. It is worth pinning here
  too, because this app is what breaks if it regresses — and it did once: a
  tool-using agent binds tools to the model, which used to discard the run name
  and send every generation back to being named after its model.
  `tests/test_agent.py` asserts the name over both `tools=[]` and
  `tools=[...]`, reading it from the same callback field Langfuse names
  observations from.
- **The failover is a step, not a silence.** Without the `on_rate_limit` hook, a
  trace shows one successful generation and no hint that the primary model was
  ever limited.
- **Sessions and users.** The Langfuse `session_id` and the checkpointer
  `thread_id` are the same id on purpose: a conversation in the Sessions view maps
  exactly onto the memory the agent had when it answered.
- **Secret masking.** Provider and Langfuse keys, and email addresses, are
  redacted from span attributes at export time — including spans emitted by the
  LangChain instrumentation, not just ones this code creates.
- **Never breaks the app.** Missing credentials, a failed setup or a rejected key
  degrade to a no-op backend with the same interface. The agent runs identically;
  it just isn't recorded.
- **Flush on exit.** `AgentRunner` is a context manager, and closing it flushes.
  A short-lived process that skips this loses every batched span.

### Feedback

Interactive sessions can score the last answer, stored as a Langfuse score named
`user-thumbs` — named after the *signal*, not what we hope it measures:

```
you> :up   that was exactly right
you> :down wrong, it ignored the timezone
```

Programmatically: `runner.feedback(reply, positive=False, comment="…")`.

### Verifying traces reach Langfuse

The middleware ships a self-check that forces a 429, flushes, reads the trace
back from the Langfuse API and audits it — using fake models, so no provider key
is needed:

```bash
python -m ratelimit_fallback  # not a module entry point; run the script instead:
git clone https://github.com/bug-ss/ratelimit-fallback && \
  python ratelimit-fallback/scripts/verify_langfuse_tracing.py
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

CI (`.github/workflows/ci.yml`) runs lint and tests on every push and pull
request, against Python 3.11 and 3.13.

### Keeping the pin current

A pinned dependency doesn't update itself, so a second CI job — weekly, plus
`workflow_dispatch` — compares the pinned SHA against upstream `main` and opens
(or refreshes) a `dependencies` issue when it falls behind. One open issue at a
time, rather than a fresh one every week.

Run it yourself any time:

```bash
python scripts/check_middleware_pin.py           # report only
python scripts/check_middleware_pin.py --bump    # rewrite the pin to upstream HEAD
```

Exit codes are `0` up to date, `1` behind, `2` couldn't reach the remote — so it
composes into other checks. After a `--bump`, reinstall and run the tests: the
trace-shape assertions in `tests/test_agent.py` are what tell you whether the new
middleware still produces the observation names this project's Langfuse
dashboards depend on.

The suite uses a fake chat model, synthetic 429s and a recording tracing backend,
so it runs offline and makes no network calls.

## Layout

```
src/general_agent/
├── config.py          Settings.from_env() — all environment reading lives here
├── tools.py           the agent's tools; add yours here
├── agent.py           model chain, middleware stack, create_agent
├── observability.py   Langfuse client, callbacks, scores — and the no-op fallback
├── runner.py          AgentRunner: one question → one trace
└── cli.py             one-shot and interactive entry points

scripts/
└── check_middleware_pin.py   reports (or bumps) a stale middleware pin
```

## Agent skills

`.claude/skills/` carries the skill packs this project was built against, so a
coding agent working in this repo gets the same guidance:

- [`langchain-ai/langchain-skills`](https://github.com/langchain-ai/langchain-skills)
  — LangChain/LangGraph fundamentals, middleware, dependencies, quickstarts.
- [`langfuse/skills`](https://github.com/langfuse/skills) — the Langfuse skill,
  covering instrumentation, the CLI, error analysis and evaluation.
