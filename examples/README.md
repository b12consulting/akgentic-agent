# akgentic-agent Examples

Examples demonstrating the akgentic-agent module capabilities.

## Running the Examples

From the **project root**:

```bash
uv run python packages/akgentic-agent/examples/00_example.py
```

From the **akgentic-agent directory**:

```bash
cd packages/akgentic-agent
uv run python examples/00_example.py
```

## Available Examples

### [00_example.py](00_example.py) - Simple BaseAgent Setup ✅

**What it demonstrates:**

- Creating an ActorSystem
- Creating an Orchestrator to manage the team
- Creating a BaseAgent with LLM configuration
- Sending a UserMessage to the agent
- Basic agent configuration with model, runtime, and usage limits

**Key concepts:**

- `AgentConfig` — Agent configuration with prompt, model, runtime settings
- `BaseAgent` — LLM-powered agent using ReactAgent from akgentic-llm
- `Orchestrator` — Team management and message telemetry
- `UserMessage` — User input message type
- `ResultMessage` — Agent response message type

**Requirements:**

- OpenAI API key in environment (`OPENAI_API_KEY`)

### [04_simple_team.py](04_simple_team.py) - V1 Migration Example (Interactive Team Chat) ✅

**What it demonstrates:**

- Creating a team with Manager, Assistant, and Expert roles using AgentCard
- Interactive chat loop with @ mention routing (e.g., `@Expert help me`)
- HumanProxy for human-to-manager communication
- EventSubscriber for real-time message flow visibility
- Dynamic team composition (Manager can hire Assistant/Expert on demand)

**Team Structure:**

- **Manager**: Coordinates team, can hire Assistant and Expert roles
- **Assistant**: Provides support and research
- **Expert**: Provides specialized knowledge
- **HumanProxy**: Routes human input to Manager

**Interactive Features:**

- Type messages directly to communicate via Manager
- Use `@AgentName message` to route to specific agents
- Type `exit` to quit the chat loop
- Real-time message flow printed via event subscriber

**Key concepts:**

- `AgentCard` — Defines agent roles with skills, prompts, and routing rules
- `AgentCard.config` — Embedded AgentConfig for each role
- `routes_to` — Defines which roles an agent can hire
- `register_agent_profiles()` — Registers multiple AgentCards with orchestrator
- `EventSubscriber.on_message()` — Event-driven message monitoring
- `HumanProxy.send()` — Sends UserMessages from human to agents
- `orchestrator.get_team()` — Retrieves current team roster

**Requirements:**

- OpenAI API key in environment (`OPENAI_API_KEY`)

**Run:**

```bash
uv run packages/akgentic-agent/examples/04_simple_team.py
```

**Example interaction:**

```
Team members:
  - @Orchestrator (Orchestrator)
  - @Human (Human)
  - @Manager (Manager)
  - @Assistant (Assistant)
  - @Expert (Expert)

Type your message (use @Name to route to specific agent, 'exit' to quit):
----------------------------------------------------------------------------------------------------
> Hello, I need help with Python
[Manager] -> ResultMessage (@Human):
Hello! I'd be happy to help you with Python. What specific aspect of Python do you need assistance with?
----------------------------------------------------------------------------------------------------
> @Expert What are Python's best practices for error handling?
[Expert] -> ResultMessage (@Human):
Python's best practices for error handling include using try-except blocks, handling specific exceptions...
----------------------------------------------------------------------------------------------------
> exit
Exiting chat loop.
```

## Planned Examples

1. **01_team_basics.py** - Basic team creation and configuration
2. **02_continuation_messaging.py** - Multi-hop request chains with automatic answer routing
3. **03_dynamic_composition.py** - Hire/fire agents at runtime
4. **05_human_proxy.py** - Additional continuation-aware human-in-the-loop patterns

## Prerequisites

All examples require:

- Python 3.12+
- `akgentic` (akgentic-core)
- `akgentic-llm`
- Valid LLM API credentials (OpenAI, Anthropic, etc.)

Install with:

```bash
uv sync
```

## Environment Variables

Set your LLM provider API key:

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

## Common Patterns

### Creating an Agent

```python
from akgentic.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.llm.config import ModelConfig, RuntimeConfig, UsageLimits
from akgentic.llm.prompts import PromptTemplate

config = AgentConfig(
    name="MyAgent",
    role="Assistant",
    prompt=PromptTemplate(template="You are a helpful assistant"),
    model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
    runtime_cfg=RuntimeConfig(temperature=0.7),
    usage_limits=UsageLimits(total_tokens_limit=10000),
)

agent_addr = system.createActor(BaseAgent, config=config)
```

### Sending Messages

```python
from akgentic.core.messages import UserMessage

message = UserMessage(content="What is 2 + 2?")
system.ask(agent_addr, message, timeout=30)
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'akgentic.agent'"**

- Run from project root with `uv run python packages/akgentic-agent/examples/00_example.py`
- Or ensure you're in the akgentic-agent directory: `cd packages/akgentic-agent && uv run python examples/00_example.py`

**"API key not found"**

- Set your environment variable: `export OPENAI_API_KEY="your-key"`

Examples will be implemented during later stories of the akgentic-agent development cycle.
