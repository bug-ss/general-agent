"""A generic LangChain agent with rate-limit failover and Langfuse tracing.

```python
from general_agent import AgentRunner

with AgentRunner() as runner:
    reply = runner.ask("What is 17 * 23?")
    print(reply.text)
```
"""

from general_agent.agent import build_agent, build_middleware, build_model_chain
from general_agent.config import Settings
from general_agent.observability import (
    NullObservability,
    Observability,
    build_observability,
)
from general_agent.runner import AgentReply, AgentRunner
from general_agent.tools import build_tools

__all__ = [
    "AgentReply",
    "AgentRunner",
    "NullObservability",
    "Observability",
    "Settings",
    "build_agent",
    "build_middleware",
    "build_model_chain",
    "build_observability",
    "build_tools",
]

__version__ = "0.1.0"
