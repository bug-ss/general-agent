# general-agent

A generic [LangChain 1.0](https://docs.langchain.com/oss/python/langchain/overview) agent
(`create_agent`) with four things a bare quickstart doesn't have:

- **Rate-limit failover** — a custom middleware
  ([`bug-ss/ratelimit-fallback`](https://github.com/bug-ss/ratelimit-fallback)) that
  switches to the next model and retries whenever a provider answers HTTP 429.
- **Langfuse tracing** — every run is one trace, with the failover visible as a
  first-class step rather than an invisible retry.
- **MCP onboarding** — point it at any [MCP](https://modelcontextprotocol.io)
  server and that server's tools become the agent's tools. No code, no restart
  of anything but this process.
- **A2A** — other agents can call this one over the
  [Agent2Agent](https://a2a-protocol.org) protocol, discovering what it can do
  from its agent card.

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

Plus whatever the MCP servers you configure expose — see
[MCP servers](#mcp-servers).

Four entry points:

```bash
general-agent "What is 17 * 23?"   # one question, then exit
general-agent                      # interactive chat, with :up / :down feedback
general-agent --serve              # answer other agents over A2A
general-agent --list-tools         # what the agent can do, MCP included
```

```python
from general_agent import AgentRunner

with AgentRunner() as runner:                     # flushes traces on exit
    reply = runner.ask("What is 17 * 23?")
    print(reply.text, reply.model, reply.trace_id)
    runner.feedback(reply, positive=True)
```

`ask` blocks; `await runner.aask(...)` is the same thing from async code, and
the one to use inside an event loop. The run path is async underneath because
MCP tools have no synchronous implementation.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # providers, web search, MCP and A2A
# or pick what you need:
pip install -e ".[openrouter]"
pip install -e ".[google]"
pip install -e ".[mcp]"          # onboard MCP servers
pip install -e ".[a2a]"          # serve other agents
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

### MCP

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_MCP_CONFIG` | `./mcp.json` if it exists | Path to an `mcpServers` JSON file |
| `AGENT_MCP_SERVERS` | — | Inline server definitions; overrides the file on a name clash |
| `AGENT_MCP_STRICT` | `false` | Fail startup when a configured server is unreachable |
| `AGENT_MCP_TOOL_PREFIX` | `true` | Namespace MCP tool names by server |
| `AGENT_MCP_STARTUP_TIMEOUT` | `30` | Seconds to wait for one server's tool list |

### A2A

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_A2A_HOST` | `127.0.0.1` | Bind address |
| `AGENT_A2A_PORT` | `8080` | Port |
| `AGENT_A2A_URL` | — | The URL to advertise, when it differs from the bind address |

## MCP servers

Any MCP server's tools can become this agent's tools. Copy
`mcp.json.example` to `mcp.json`, or point `AGENT_MCP_CONFIG` anywhere:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"]
    },
    "internal-api": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer ${INTERNAL_API_TOKEN}" }
    }
  }
}
```

```bash
general-agent --list-tools
```

The format is the one Claude Desktop, VS Code and Cursor already use, so a
server someone has running elsewhere is onboarded by copying its stanza across.
`stdio`, `http` (streamable HTTP), `sse` and `websocket` all work; the transport
is inferred from `command` or `url` when you don't say.

The decisions worth knowing about:

- **`${VAR}` is expanded from the environment**, and an unset variable is an
  error rather than an empty string — `Authorization: Bearer ` fails as a 401
  three layers from the cause. Tokens stay in `.env`; the config file stays
  committable.
- **One unreachable server costs you its tools, not the agent.** Servers are
  contacted one at a time, and a failure is a warning. `AGENT_MCP_STRICT=true`
  inverts that, which is the right setting when the deployment's whole job
  depends on one of them.
- **Tool names are namespaced by server** (`filesystem_read_file`). Two servers
  exposing `search` is entirely normal, and an unprefixed collision silently
  shadows one of them. A collision with a built-in tool is resolved in the
  built-in's favour and logged.
- **Unknown config keys are dropped with a warning.** They would otherwise be
  forwarded as keyword arguments and blow up at the first tool call instead of
  at startup.
- **The whole run path is async.** The adapter builds MCP tools with a coroutine
  and no sync implementation, so a synchronous `invoke` anywhere in the run
  would raise the moment the model called one. `AgentRunner.ask` wraps
  `aask`; the failover middleware implements `awrap_model_call` alongside its
  sync hook, so 429 handling is unchanged.
- **A session is opened per tool call.** That is the adapter's stateless
  default, and the right trade here: a long-lived stdio subprocess owned by a
  CLI that may sit idle for an hour costs more than it saves.

## A2A: other agents calling this one

```bash
general-agent --serve --port 8080
curl http://127.0.0.1:8080/.well-known/agent-card.json
```

The card advertises JSON-RPC and REST interfaces, and lists the agent's skills —
**one per tool, MCP-onboarded ones included**. That is what makes the card worth
fetching: onboard a server that talks to your internal API, and a remote agent
can see this one can now do that, without anyone editing a description.

```
POST /a2a/jsonrpc     JSON-RPC   (A2A 1.0, with 0.3 compatibility)
POST /a2a/rest/...    HTTP+JSON
GET  /.well-known/agent-card.json
```

A request becomes a task, and the task reports where the work is: `submitted` →
`working` → an artifact carrying the answer → `completed`, or `failed` with the
error. The answer's artifact metadata carries the model that served it and the
Langfuse trace id — the two things a caller debugging a bad answer cannot see
from its own side.

Two mappings make multi-turn work: the A2A `context_id` becomes both the
LangGraph checkpointer thread and the Langfuse session id, so a second message
on the same context reaches an agent that remembers the first, and the whole
exchange reads as one session in the traces.

Cancelling a task cancels the asyncio task running it, so an in-flight model
call actually stops rather than continuing to bill against work nobody is
waiting for.

Note what this is not: serving A2A does not make this agent an A2A *client*.
Nothing here calls out to other agents.

> **The endpoint is unauthenticated and the card is public.** That is why the
> default bind is loopback. Exposing it means putting a proxy in front that
> terminates TLS and authenticates callers, and setting `AGENT_A2A_URL` to the
> address that proxy answers on — otherwise the card advertises a bind address
> nobody can reach. Tasks are stored in memory, so they are lost on restart and
> invisible to a second replica; swap in a database-backed store before running
> more than one.

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

The suite runs offline and makes no network calls: a fake chat model, synthetic
429s and a recording tracing backend. Two exceptions are local, not remote — the
MCP tests spawn a real MCP server as a subprocess and talk to it over stdio, and
the A2A tests drive the real ASGI app through Starlette's test client. Both are
deliberate: what those features promise is that an arbitrary server's tools work
and that another agent can reach this one, and a mock proves neither.

## Layout

```
src/general_agent/
├── config.py          Settings.from_env() — all environment reading lives here
├── tools.py           the agent's built-in tools; add yours here
├── mcp.py             onboarding MCP servers as tools
├── agent.py           model chain, middleware stack, create_agent
├── a2a_server.py      agent card, executor, ASGI app — serving other agents
├── observability.py   Langfuse client, callbacks, scores — and the no-op fallback
├── runner.py          AgentRunner: one question → one trace
└── cli.py             one-shot, interactive, --serve and --list-tools

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

There is no official skill pack for MCP onboarding or A2A. `langchain-skills`
publishes neither, and the only MCP-related skill in the Anthropic library —
`mcp-builder` — is about *writing* MCP servers, which is the opposite direction
from consuming them. The A2A SDK ships a `.agents/skills/mistake-reflection`
skill, but that is a contributor workflow for people working on the SDK itself.
Both features here were built against the SDKs' own source instead.
