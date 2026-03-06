# akgentic-agent

LLM-driven collaborative agent patterns for the Akgentic framework. Agents
communicate through a single flat `AgentMessage` type; routing decisions are
made by the LLM via structured output, and dynamic team composition is managed
at runtime through role-based hiring.

> **Note**: `akgentic-team` is deprecated. `akgentic-agent` is its replacement.
> See [Migration from akgentic-team](#migration-from-akgentic-team) for guidance.

## Installation

### Workspace Installation (Recommended)

This package is designed for use within the Akgentic monorepo workspace:

```bash
# From workspace root
git clone git@github.com:b12consulting/akgentic-quick-start.git
cd akgentic-quick-start

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install all workspace packages
uv sync --all-packages --all-extras
```

All dependencies (`akgentic-core`, `akgentic-llm`, `akgentic-tool`) are automatically
resolved via workspace configuration.

## Architecture

`akgentic-agent` moves routing intelligence from the framework into the LLM.

In `akgentic-team` (deprecated), the framework tracked a continuation call stack
(`HelpRequestMessage` / `HelpAnswerMessage`) and automatically routed answers back.
In `akgentic-agent`, the LLM itself decides who to message next: each invocation
returns a `StructuredOutput` — a list of `{message, recipient}` pairs. The framework
delivers them; the LLM navigates the conversation graph.

```
┌─────────────────────────────────────────────────────────────┐
│                       BaseAgent                             │
│                                                             │
│  AgentMessage ──► receiveMsg_AgentMessage                   │
│                          │                                  │
│                          ▼                                  │
│                   process_message()                         │
│                          │                                  │
│                          ▼                                  │
│                     act() → ReactAgent.run_sync()           │
│                          │                                  │
│                          ▼                                  │
│                   StructuredOutput                          │
│              list[{message, recipient}]                     │
│                          │                                  │
│          ┌───────────────┼───────────────┐                  │
│          ▼               ▼               ▼                  │
│   @MemberName      RoleName (hire)   sender (reply)         │
│   direct send      auto-hire then    reply to caller        │
│                    send                                     │
└─────────────────────────────────────────────────────────────┘
```

See [docs/agent-collaboration.md](docs/agent-collaboration.md) for a full walkthrough
of the collaboration model.

## Features

- **LLM-driven routing** — The LLM decides which team members to address next
  via `StructuredOutput`; the framework delivers the messages
- **Single message type** — `AgentMessage` carries all inter-agent communication;
  no separate request/answer types
- **Dynamic hire-by-role** — Recipient expressed as a role name triggers automatic
  actor creation via `TeamTool` before message delivery
- **`BaseAgent` with REACT** — LLM-powered agents via `akgentic-llm`'s `ReactAgent`
- **Tool integration** — `TeamTool` (hire/fire), `PlanningTool`, and custom `ToolCard`
  instances exposed to the LLM
- **`HumanProxy`** — Human-in-the-loop agent receiving `AgentMessage` for oversight
- **Recursion protection** — Bad-routing retries limited to `MAX_RECURSION_DEPTH=3`
  with automatic human notification on failure

## Dependencies

- `pydantic` — Data validation and configuration models
- `akgentic` (`akgentic-core`) — Actor system, messaging, orchestration
- `akgentic-llm` — LLM provider abstraction, REACT agent, prompt management
- `akgentic-tool` — Tool abstractions: `TeamTool`, `PlanningTool`, `ToolCard`

## Quick Example

```python
from akgentic.agent import AgentConfig, AgentMessage, BaseAgent, HumanProxy
from akgentic.core import ActorSystem, AgentCard, BaseConfig, Orchestrator
from akgentic.llm import ModelConfig, PromptTemplate

# 1. Create the actor system and orchestrator
actor_system = ActorSystem()
orchestrator_addr = actor_system.createActor(
    Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
)
orchestrator_proxy = actor_system.proxy_ask(orchestrator_addr, Orchestrator)

# 2. Define roles via AgentCards and register them
manager_card = AgentCard(
    role="Manager",
    description="Project manager who coordinates specialists",
    skills=["coordination", "delegation"],
    agent_class="akgentic.agent.BaseAgent",
    config=AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(
            template="You are a project manager. Delegate tasks to specialists."
        ),
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
    ),
    routes_to=["Developer", "QA"],  # roles this agent can hire
)
orchestrator_proxy.register_agent_profiles([manager_card])

# 3. Create a HumanProxy (human-in-the-loop entry point)
human_addr = orchestrator_proxy.createActor(
    HumanProxy, config=BaseConfig(name="@Human", role="Human")
)
human_proxy = actor_system.proxy_tell(human_addr, HumanProxy)

# 4. Instantiate the Manager agent
manager_addr = orchestrator_proxy.createActor(
    BaseAgent, config=manager_card.get_config_copy()
)

# 5. Send the first message from the human
human_proxy.send(manager_addr, AgentMessage(content="Plan the next sprint."))
```

## Migration from akgentic-team

| akgentic-team                                | akgentic-agent                                            |
| -------------------------------------------- | --------------------------------------------------------- |
| `HelpRequestMessage(content=…, owner=agent)` | `AgentMessage(content=…, recipient=agent)`                |
| `HelpAnswerMessage(content=…, request_id=…)` | `AgentMessage(content=…, recipient=sender)` — LLM decides |
| `Continuation` call stack                    | LLM context via `ReactAgent.ContextManager`               |
| Framework-routed answer propagation          | LLM-directed routing via `StructuredOutput`               |
| `from akgentic.team import BaseAgent`        | `from akgentic.agent import BaseAgent`                    |

Key steps:

1. Replace all `HelpRequestMessage` / `HelpAnswerMessage` usages with `AgentMessage`
2. Remove explicit `Continuation` construction — routing is LLM's responsibility
3. Update imports: `akgentic.team` → `akgentic.agent`
4. Review agent prompts — the LLM now needs to reason about who to address

## Documentation

- [Agent Collaboration System](docs/agent-collaboration.md) — How agents collaborate,
  route messages, and delegate tasks in the LLM-driven model

## Examples

See the [examples/](examples/) directory for usage patterns (coming soon).
