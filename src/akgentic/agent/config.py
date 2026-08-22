"""Configuration models for team agents.

This module provides configuration for BaseAgent instances:
- AgentConfig: Per-agent configuration (prompt, LLM settings, limits)
- AgentState: Runtime state (backstory from prompt)

Team-level configuration for hiring is managed by Orchestrator and passed
dynamically during agent creation.
"""

import warnings
from typing import Any

from pydantic import Field, model_validator

from akgentic.core import BaseConfig, BaseState
from akgentic.llm.config import (
    AgentUsageLimits,
    CompactionConfig,
    ModelConfig,
    RuntimeConfig,
    RunUsageLimits,
    UsageLimits,
)
from akgentic.llm.prompts import PromptTemplate
from akgentic.tool.core import ToolCard, ToolState

_SHIM_REMOVAL_RELEASE = "akgentic-agent 2.0.0"


class AgentConfig(BaseConfig):
    """Configuration for a team agent instance.

    Each agent gets its own configuration with:
    - prompt: Agent's backstory/system prompt
    - model_cfg: LLM model provider and settings
    - runtime_cfg: Execution parameters (retries, tool-call strategy, HTTP client)
    - run_usage_limits: Budget for ONE run; enforced by pydantic-ai, resets per run
    - agent_usage_limits: Budget for the agent's WHOLE lifetime; enforced pre-flight
      by ReactAgent, which reseeds it from replayed usage events on restore
    - compaction_cfg: Context-compaction strategy and auto-trigger settings

    Neither tier is enforced here: both are carried into the ReactAgentConfig that
    BaseAgent builds, and enforced inside ReactAgent (akgentic-llm).

    Team-level config for hiring members is provided by Orchestrator,
    not stored in individual agents.

    Attributes:
        prompt: Agent backstory or system prompt, as a PromptTemplate.
        model_cfg: LLM model configuration (provider, model name, API settings).
        runtime_cfg: Runtime execution settings (retries, end strategy, HTTP client).
            Sampling settings such as temperature live on model_cfg, not here.
        run_usage_limits: Per-run token/request budget; the tier pydantic-ai enforces.
        agent_usage_limits: Agent-lifetime token/run budget. Defaults to an all-None
            budget, which never blocks — "unlimited" without a nullable field.
        compaction_cfg: Context-compaction strategy and auto-trigger config (opt-in;
            off unless model_cfg.context_length is set).

    Deprecated:
        ``usage_limits`` survives as a constructor keyword and a read accessor for
        ``run_usage_limits``. Both warn, and both are removed in akgentic-agent 2.0.0.
        Passing ``usage_limits`` and ``run_usage_limits`` together raises ValueError.

    Example:
        >>> from akgentic.llm.config import (
        ...     AgentUsageLimits, ModelConfig, RunUsageLimits, RuntimeConfig
        ... )
        >>> from akgentic.llm.prompts import PromptTemplate
        >>> config = AgentConfig(
        ...     prompt=PromptTemplate(template="You are a helpful software developer."),
        ...     model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        ...     runtime_cfg=RuntimeConfig(retries=3),
        ...     run_usage_limits=RunUsageLimits(
        ...         run_request_limit=50, total_tokens_limit=100000
        ...     ),
        ...     agent_usage_limits=AgentUsageLimits(agent_request_limit=200),
        ... )
    """

    prompt: PromptTemplate = Field(
        default_factory=PromptTemplate,
        description="Agent backstory or system prompt defining behavior and personality",
    )
    model_cfg: ModelConfig = Field(
        default_factory=ModelConfig,
        description="LLM model configuration including provider, model name, and API settings",
    )
    runtime_cfg: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Runtime execution settings: retries, tool-call end strategy, HTTP client",
    )
    run_usage_limits: RunUsageLimits = Field(
        default_factory=RunUsageLimits,
        description="Per-run token and request limits, enforced by pydantic-ai on every run",
    )
    agent_usage_limits: AgentUsageLimits = Field(
        default_factory=AgentUsageLimits,
        description="Agent-lifetime token and run limits, enforced pre-flight on every run",
    )
    compaction_cfg: CompactionConfig = Field(
        default_factory=CompactionConfig,
        description="Context-compaction strategy and auto-trigger config (opt-in, off by default)",
    )
    tools: list[ToolCard] = Field(
        default_factory=list,
        description="List of tool cards defining the tools this agent can use, with parameters",
    )

    @model_validator(mode="before")
    @classmethod
    def _map_pre_split_usage_limits(cls, data: Any) -> Any:
        """Warn on the deprecated ``usage_limits=`` keyword and route it to the run tier.

        Runs before field validation, so ``usage_limits`` never reaches Pydantic as an
        unexpected keyword — which is why the shim is a validator and not a property
        alone. The value must actually land on ``run_usage_limits``: accepting it and
        discarding it would leave the agent on a budget nobody chose.

        A mapping carrying the pre-split ``request_limit`` key goes through the
        deprecated ``UsageLimits`` alias, whose own shim folds that key onto
        ``run_request_limit``. Validated directly as ``RunUsageLimits`` the key would be
        unknown and Pydantic would drop it in silence — the same accepted-and-discarded
        failure with one more layer of indirection. Configs arrive as mappings whenever
        they are rebuilt from a persisted or declarative source, so this is a live path.
        """
        if not isinstance(data, dict) or "usage_limits" not in data:
            return data
        if "run_usage_limits" in data:
            raise ValueError(
                "AgentConfig received both usage_limits (deprecated) and "
                "run_usage_limits; which one wins would depend on argument order — "
                "pass only run_usage_limits"
            )
        mapped = dict(data)
        value = mapped.pop("usage_limits")
        if isinstance(value, dict) and "request_limit" in value:
            # Raises on the both-spellings case before warning, so the error is not
            # pre-empted under -W error::DeprecationWarning.
            value = UsageLimits.model_validate(value)
        warnings.warn(
            f"AgentConfig(usage_limits=...) is deprecated and will be removed in "
            f"{_SHIM_REMOVAL_RELEASE}; use run_usage_limits=... instead",
            DeprecationWarning,
            stacklevel=3,
        )
        mapped["run_usage_limits"] = value
        return mapped

    @property
    def usage_limits(self) -> RunUsageLimits:
        """DEPRECATED read accessor for ``run_usage_limits``.

        Removed in akgentic-agent 2.0.0. Returns the run tier itself, not a copy.
        """
        warnings.warn(
            f"AgentConfig.usage_limits is deprecated and will be removed in "
            f"{_SHIM_REMOVAL_RELEASE}; read run_usage_limits instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_usage_limits


class AgentState(BaseState):
    """Runtime state for team agents.

    Stores the agent's backstory (resolved from config prompt) and
    maintains state change notifications for observers.

    Carrying ``tool_state`` is also what makes this model satisfy the tool
    layer's ``ToolStateCarrier`` protocol — structurally, with no method, no
    property and no import edge beyond the field's own type.

    Attributes:
        backstory: Resolved backstory/system prompt from config
        tool_state: The tool layer's persistent per-agent slot — context-update
            baselines and the block counter. It is a **cache, never a record**:
            the message history is the record, so a lost or stale slot can only
            cost a full-snapshot re-send, never a lost update. Mutated in place
            by ``akgentic.tool.core.ContextUpdater``; the existing
            ``notify_if_changed()`` checkpoints pick that up, because change
            detection compares serializations rather than reading a dirty flag.
            Never hold a reference to it across ``init_state()`` — restore
            replaces the whole state object, so a cached handle goes silently
            stale. Read it through ``self.state`` every time.
    """

    backstory: str
    tool_state: ToolState = Field(default_factory=ToolState)
