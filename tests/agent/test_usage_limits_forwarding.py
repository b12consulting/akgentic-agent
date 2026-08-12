"""Tests that BaseAgent.on_start() forwards BOTH usage-limit tiers to its ReactAgent.

`ReactAgentConfig` supplies a default `agent_usage_limits`, so omitting the keyword
constructs perfectly well and every other agent test still passes — with the operator's
lifetime budget dropped on the floor. The only assertion that separates the fixed code
from the broken code inspects the config object actually handed to the `ReactAgent`.
A test that merely starts an agent passes against the bug.

The real `ReactAgent` is swapped for a capturing fake, so no LLM is built and no network
is touched.
"""

import time
import warnings

import akgentic.agent.agent as agent_module
from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.core import ActorSystem, BaseConfig, Orchestrator
from akgentic.llm import (
    AgentUsageLimits,
    ModelConfig,
    PromptTemplate,
    ReactAgentConfig,
    RunUsageLimits,
)

# Two distinguishable sentinels. Identical values would let a transposed keyword pair
# pass green, which is the most likely way to implement this story wrongly.
RUN_TIER = RunUsageLimits(run_request_limit=13, total_tokens_limit=9999)
AGENT_TIER = AgentUsageLimits(agent_request_limit=17, total_tokens_limit=8888)


class _CapturingReactAgent:
    """Stands in for ReactAgent and records the kwargs on_start passes to it."""

    captured: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).captured.append(kwargs)

    def system_prompt(self, fn: object) -> object:
        return fn

    def close(self) -> None:
        pass


def _start_agent_and_capture_config(
    config: AgentConfig,
) -> tuple[ReactAgentConfig, list[warnings.WarningMessage]]:
    """Start a BaseAgent through the real actor system; return its built LLM config.

    Also returns every warning raised across the start, so a caller can assert the
    absence of deprecation noise — the observable symptom this epic removes.
    """
    _CapturingReactAgent.captured = []
    system = ActorSystem()
    original = agent_module.ReactAgent
    agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[misc, assignment]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            orch_addr = system.createActor(
                Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
            )
            orchestrator = system.proxy_ask(orch_addr, Orchestrator)
            orchestrator.createActor(BaseAgent, config=config)
            time.sleep(0.5)

        assert _CapturingReactAgent.captured, "ReactAgent was never constructed"
        built = _CapturingReactAgent.captured[-1]["config"]
        assert isinstance(built, ReactAgentConfig)
        return built, list(caught)
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
        model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
        **overrides,  # type: ignore[arg-type]
    )


class TestBothTiersReachTheReactAgent:
    """AC #2/#3/#4: both tiers arrive on the built config, and do not cross."""

    def test_agent_tier_arrives(self) -> None:
        """The lifetime budget reaches the ReactAgent that enforces it.

        This is the assertion the story exists for: before the fix the field was
        accepted on AgentConfig and silently discarded here.
        """
        built, _ = _start_agent_and_capture_config(
            _agent_config(run_usage_limits=RUN_TIER, agent_usage_limits=AGENT_TIER)
        )
        assert built.agent_usage_limits == AGENT_TIER
        assert built.agent_usage_limits.agent_request_limit == 17
        assert built.agent_usage_limits.total_tokens_limit == 8888

    def test_agent_tier_is_not_the_default_budget(self) -> None:
        """Pin the exact pre-fix symptom: a default, all-None, never-blocking budget."""
        built, _ = _start_agent_and_capture_config(
            _agent_config(agent_usage_limits=AGENT_TIER)
        )
        assert built.agent_usage_limits != AgentUsageLimits()
        assert built.agent_usage_limits.agent_request_limit is not None

    def test_run_tier_arrives(self) -> None:
        """The run tier survives the rename to its new keyword."""
        built, _ = _start_agent_and_capture_config(
            _agent_config(run_usage_limits=RUN_TIER, agent_usage_limits=AGENT_TIER)
        )
        assert built.run_usage_limits == RUN_TIER
        assert built.run_usage_limits.run_request_limit == 13
        assert built.run_usage_limits.total_tokens_limit == 9999

    def test_the_tiers_do_not_cross(self) -> None:
        """A transposed keyword pair must not pass — the sentinels differ on purpose."""
        built, _ = _start_agent_and_capture_config(
            _agent_config(run_usage_limits=RUN_TIER, agent_usage_limits=AGENT_TIER)
        )
        assert built.run_usage_limits.total_tokens_limit == 9999
        assert built.agent_usage_limits.total_tokens_limit == 8888
        assert built.run_usage_limits.run_request_limit == 13
        assert built.agent_usage_limits.agent_request_limit == 17

    def test_defaults_still_forward_cleanly(self) -> None:
        """An unconfigured agent gets the run tier's safety brake and no lifetime cap."""
        built, _ = _start_agent_and_capture_config(_agent_config())
        assert built.run_usage_limits.run_request_limit == 50
        assert built.agent_usage_limits.agent_request_limit is None


class TestOnStartIsDeprecationFree:
    """AC #5: starting an agent emits no DeprecationWarning.

    That warning is the observable symptom of this epic's problem — the agent surface
    depending on a deprecated name on every single agent construction — so its absence
    is the cleanest proof the adoption is complete.
    """

    def test_no_deprecation_warning_during_on_start(self) -> None:
        _, caught = _start_agent_and_capture_config(
            _agent_config(run_usage_limits=RUN_TIER, agent_usage_limits=AGENT_TIER)
        )
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations == [], f"on_start emitted: {[str(w.message) for w in deprecations]}"

    def test_no_deprecation_warning_for_a_default_agent(self) -> None:
        """The default path is the one every team agent takes."""
        _, caught = _start_agent_and_capture_config(_agent_config())
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations == [], f"on_start emitted: {[str(w.message) for w in deprecations]}"
