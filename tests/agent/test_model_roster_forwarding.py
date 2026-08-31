"""Tests that ``BaseAgent.on_start()`` carries the declared roster into its ReactAgent.

``ReactAgentConfig`` supplies a default empty ``model_roster``, so omitting the keyword
constructs perfectly well and every other agent test still passes — with the operator's
declared roster dropped on the floor and switching silently unavailable. The only
assertion that separates the fixed code from the broken code inspects the config object
actually handed to the ``ReactAgent``.

The real ``ReactAgent`` is swapped for a capturing fake, so no LLM is built and no
network is touched.
"""

import time

from akgentic.core import ActorSystem, BaseConfig, Orchestrator
from akgentic.llm import (
    AgentUsageLimits,
    CompactionConfig,
    ModelConfig,
    PromptTemplate,
    ReactAgentConfig,
    RuntimeConfig,
    RunUsageLimits,
)

import akgentic.agent.agent as agent_module
from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig

# Three distinguishable entries: a transposed or truncated roster cannot pass green.
GPT = ModelConfig(provider="openai", model="gpt-4o")
CLAUDE = ModelConfig(provider="anthropic", model="claude-sonnet-4-5")
GEMINI = ModelConfig(provider="google-gla", model="gemini-2.0-flash")


class _CapturingReactAgent:
    """Stands in for ReactAgent and records the kwargs on_start passes to it."""

    captured: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).captured.append(kwargs)

    def system_prompt(self, fn: object) -> object:
        return fn

    def close(self) -> None:
        pass


def _start_agent_and_capture_config(config: AgentConfig) -> ReactAgentConfig:
    """Start a BaseAgent through the real actor system; return its built LLM config."""
    _CapturingReactAgent.captured = []
    system = ActorSystem()
    original = agent_module.ReactAgent
    agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[misc, assignment]
    try:
        orch_addr = system.createActor(
            Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
        )
        orchestrator = system.proxy_ask(orch_addr, Orchestrator)
        orchestrator.createActor(BaseAgent, config=config)
        time.sleep(0.5)

        assert _CapturingReactAgent.captured, "ReactAgent was never constructed"
        built = _CapturingReactAgent.captured[-1]["config"]
        assert isinstance(built, ReactAgentConfig)
        return built
    finally:
        agent_module.ReactAgent = original  # type: ignore[misc]
        try:
            system.shutdown(timeout=5)
        except Exception:
            pass


def _agent_config(**overrides: object) -> AgentConfig:
    return AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(template="You are a manager."),
        **overrides,  # type: ignore[arg-type]
    )


class TestRosterReachesTheReactAgent:
    """AC #6: the declared roster arrives on the config the ReactAgent is built with."""

    def test_roster_arrives_intact_and_in_order(self) -> None:
        """This is the assertion the story exists for.

        Without the forwarded keyword the roster is accepted on AgentConfig and
        silently discarded here, leaving an agent that declares three models and can
        switch to none of them.
        """
        built = _start_agent_and_capture_config(_agent_config(model_cfg=[GPT, CLAUDE, GEMINI]))
        assert built.model_roster == [GPT, CLAUDE, GEMINI]

    def test_the_active_model_is_element_zero(self) -> None:
        built = _start_agent_and_capture_config(_agent_config(model_cfg=[GPT, CLAUDE, GEMINI]))
        assert built.model_cfg == GPT

    def test_the_roster_is_not_truncated_to_the_active_entry(self) -> None:
        """Pin the shape of the most plausible partial fix."""
        built = _start_agent_and_capture_config(_agent_config(model_cfg=[GPT, CLAUDE, GEMINI]))
        assert len(built.model_roster) == 3
        assert [entry.provider for entry in built.model_roster] == [
            "openai",
            "anthropic",
            "google-gla",
        ]

    def test_a_single_model_agent_forwards_an_empty_roster(self) -> None:
        built = _start_agent_and_capture_config(_agent_config(model_cfg=GPT))
        assert built.model_roster == []
        assert built.model_cfg == GPT

    def test_a_default_agent_forwards_an_empty_roster(self) -> None:
        built = _start_agent_and_capture_config(_agent_config())
        assert built.model_roster == []

    def test_the_built_config_survives_its_own_membership_rule(self) -> None:
        """ReactAgentConfig requires the active model to be in a non-empty roster.

        Forwarding one half without the other would construct a config that fails
        that rule inside the actor — where the traceback names no operator line.
        """
        built = _start_agent_and_capture_config(_agent_config(model_cfg=[CLAUDE, GPT]))
        assert built.model_cfg == CLAUDE
        assert built.model_cfg in built.model_roster
        ReactAgentConfig.model_validate(built.model_dump())


class TestEverySharedFieldIsForwarded:
    """The general form of the defect this story fixed by hand.

    ``model_roster`` was added to ReactAgentConfig and then had to be carried across as
    a separate manual keyword. Nothing forced that: the roster was accepted on
    AgentConfig and silently dropped on the way to the ReactAgent, and every other test
    in the package stayed green. The next field the two models come to share will be
    dropped in exactly the same way, for exactly as long as nobody notices.

    So the pairing is derived from ``model_fields`` at runtime rather than from a list
    written out here — a hand-written list is correct on the day it is written and stale
    the day a field is added, which is the failure mode itself. Every shared field is
    given a non-default value, so a dropped keyword surfaces as that field's default
    rather than as a coincidental match.
    """

    def test_every_field_shared_with_react_agent_config_arrives(self) -> None:
        agent_cfg = _agent_config(
            model_cfg=[GPT, CLAUDE, GEMINI],
            runtime_cfg=RuntimeConfig(retries=7),
            run_usage_limits=RunUsageLimits(run_request_limit=13),
            agent_usage_limits=AgentUsageLimits(agent_request_limit=17),
            compaction_cfg=CompactionConfig(keep_recent_messages=9),
        )
        built = _start_agent_and_capture_config(agent_cfg)

        shared = set(AgentConfig.model_fields) & set(ReactAgentConfig.model_fields)
        assert "model_roster" in shared, "the guard is vacuous if the field it guards is gone"

        dropped = {
            name: (getattr(agent_cfg, name), getattr(built, name))
            for name in sorted(shared)
            if getattr(built, name) != getattr(agent_cfg, name)
        }
        assert not dropped, (
            f"declared on AgentConfig but not carried onto ReactAgentConfig: {dropped}"
        )
