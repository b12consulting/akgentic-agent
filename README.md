# akgentic-agent

Collaborative agent patterns for the Akgentic framework — continuation-based
request/answer flows, dynamic team composition (hire/fire), hierarchical agent
management, and structured output patterns for multi-agent coordination.

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

All dependencies (akgentic-core, akgentic-llm) are automatically resolved via workspace configuration.

## Features

- **Continuation-based messaging** — Multi-hop request chains with automatic answer routing
- **Dynamic team composition** — Hire/fire agents at runtime
- **BaseAgent with REACT** — LLM-powered agents via akgentic-llm's ReactAgent
- **Structured output** — Provider-aware output types for agent communication
- **TeamFactory** — Create teams from configuration dicts
- **HumanProxy** — Continuation-aware human-in-the-loop interactions

## Dependencies

- `pydantic` — Data validation and configuration models
- `akgentic` (akgentic-core) — Actor system, messaging, orchestration
- `akgentic-llm` — LLM provider abstraction, REACT agent, prompt management

## Examples

See the [examples/](examples/) directory for usage patterns (coming soon).
