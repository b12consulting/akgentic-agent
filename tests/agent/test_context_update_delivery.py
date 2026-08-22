"""Tests for per-turn context-update block delivery (Epic 19, story 19-1).

BaseAgent collects context-state providers at ``on_start`` and, at the top of
every ``act()`` turn, appends at most one **Context update** block through
``ContextManager.record_operator_action``: full renderings on first sight,
deltas afterwards, nothing at all on an unchanged turn. Providers and
renderers degrade to "no section" on any failure — never to a failed turn.

Most specs use the bare-agent + stubbed-ReactAgent pattern shared by the other
unit tests in this package; the first-run fold spec uses the real akgentic-llm
``ContextManager`` because the buffer-vs-append decision is the property under
test; the on_start spec goes through the real actor system.
"""

import logging
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import MagicMock

import pytest
from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.llm import ContextManager, ReactAgent

from akgentic.tool.core import ContextState

# =============================================================================
# HELPERS
# =============================================================================


class _RosterState(ContextState):
    """Roster-shaped test state: full lists members, delta names only joins."""

    members: tuple[str, ...] = ()

    def render_full(self) -> str:
        if not self.members:
            return ""
        return "**Team roster:**\n" + "\n".join(self.members)

    def render_delta(self, previous: Self) -> str | None:
        joined = [m for m in self.members if m not in previous.members]
        if not joined:
            return None
        return ", ".join(joined) + " joined the team."


class _PlanningState(ContextState):
    """Planning-shaped test state: sentences joined with spaces, by design."""

    tasks: tuple[str, ...] = ()

    def render_full(self) -> str:
        if not self.tasks:
            return ""
        return "**Planning:** " + " ".join(self.tasks)

    def render_delta(self, previous: Self) -> str | None:
        added = [t for t in self.tasks if t not in previous.tasks]
        if not added:
            return None
        return "New tasks: " + " ".join(added)


class _OtherTypeState(ContextState):
    """A reconfigured-card stand-in: diffing across types is forbidden."""

    text: str = ""

    def render_full(self) -> str:
        return self.text

    def render_delta(self, previous: Self) -> str | None:
        raise AssertionError("render_delta must never be asked to diff across types")


class _ExplodingDeltaState(ContextState):
    """Full render works; the delta renderer raises (NFR3 degradation path)."""

    text: str = ""

    def render_full(self) -> str:
        return self.text

    def render_delta(self, previous: Self) -> str | None:
        raise RuntimeError("renderer boom")


def _provider(name: str, holder: dict[str, ContextState | None]) -> Callable[[], Any]:
    """A named provider reading its current state from a mutable holder."""

    def provider() -> ContextState | None:
        return holder["state"]

    provider.__name__ = name
    return provider


def _raising_provider(name: str) -> Callable[[], Any]:
    def provider() -> ContextState | None:
        raise RuntimeError("provider boom")

    provider.__name__ = name
    return provider


def _make_agent(providers: list[Callable[[], Any]] | None = None) -> BaseAgent:
    """Bare BaseAgent (no Pykka) with a stubbed ReactAgent and given providers."""
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = MagicMock()  # type: ignore[attr-defined]

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent._context_state_providers = providers or []  # type: ignore[attr-defined]
    agent._context_baselines = {}  # type: ignore[attr-defined]
    agent._context_update_seq = 0  # type: ignore[attr-defined]
    return agent


def _recorded_blocks(agent: BaseAgent) -> list[str]:
    record = agent._react_agent.context.record_operator_action  # type: ignore[attr-defined]
    return [call.args[0] for call in record.call_args_list]


# =============================================================================
# AC 3 — first block: full renderings, marker, counter
# =============================================================================


class TestFirstBlock:
    def test_first_turn_appends_one_block_with_every_full_rendering(self) -> None:
        roster: dict[str, ContextState | None] = {
            "state": _RosterState(members=("@Manager", "@Analyst"))
        }
        planning: dict[str, ContextState | None] = {"state": _PlanningState(tasks=("ID 1 [t]",))}
        agent = _make_agent(
            [_provider("team_roster_state", roster), _provider("planning_state", planning)]
        )

        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 1
        expected = (
            "**Context update 1** — current state.\n\n"
            "**Team roster:**\n@Manager\n@Analyst\n\n"
            "**Planning:** ID 1 [t]"
        )
        assert blocks[0] == expected
        assert agent._context_update_seq == 1  # type: ignore[attr-defined]

    def test_empty_full_rendering_contributes_nothing_and_counter_stays(self) -> None:
        roster: dict[str, ContextState | None] = {"state": _RosterState(members=())}
        agent = _make_agent([_provider("team_roster_state", roster)])

        agent._deliver_context_update()

        assert _recorded_blocks(agent) == []
        assert agent._context_update_seq == 0  # type: ignore[attr-defined]
        # No baseline advance either: an empty full render is a skip, not a delivery.
        assert agent._context_baselines == {}  # type: ignore[attr-defined]


# =============================================================================
# AC 3 / AC 4 — unchanged turn appends nothing; deltas afterwards
# =============================================================================


class TestDeltaTurns:
    def test_unchanged_second_turn_appends_nothing(self) -> None:
        roster: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", roster)])

        agent._deliver_context_update()
        roster["state"] = _RosterState(members=("@Manager",))
        agent._deliver_context_update()

        assert len(_recorded_blocks(agent)) == 1
        assert agent._context_update_seq == 1  # type: ignore[attr-defined]

    def test_hire_between_turns_appends_delta_not_relisted_roster(self) -> None:
        roster: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", roster)])

        agent._deliver_context_update()
        roster["state"] = _RosterState(members=("@Manager", "@Bob"))
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        assert blocks[1] == (
            "**Context update 2** — state has changed since the last update.\n\n"
            "@Bob joined the team."
        )
        assert "**Team roster:**" not in blocks[1]

    def test_two_changed_providers_produce_one_block_in_factory_order(self) -> None:
        roster: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        planning: dict[str, ContextState | None] = {"state": _PlanningState(tasks=("ID 1 [t]",))}
        agent = _make_agent(
            [_provider("team_roster_state", roster), _provider("planning_state", planning)]
        )

        agent._deliver_context_update()
        roster["state"] = _RosterState(members=("@Manager", "@Bob"))
        planning["state"] = _PlanningState(tasks=("ID 1 [t]", "ID 2 [u]"))
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        assert blocks[1] == (
            "**Context update 2** — state has changed since the last update.\n\n"
            "@Bob joined the team.\n\n"
            "New tasks: ID 2 [u]"
        )


# =============================================================================
# AC 4 — same-type rule: a type change renders full, never a cross-type diff
# =============================================================================


class TestTypeChange:
    def test_state_of_different_type_renders_full_not_delta(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        agent._deliver_context_update()
        # Card reconfigured mid-life: same provider name, different concrete type.
        # _OtherTypeState.render_delta raises AssertionError if it is ever consulted.
        holder["state"] = _OtherTypeState(text="Reconfigured snapshot.")
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        assert "Reconfigured snapshot." in blocks[1]
        baselines = agent._context_baselines  # type: ignore[attr-defined]
        assert isinstance(baselines["team_roster_state"], _OtherTypeState)


# =============================================================================
# AC 7 — degradation: None states, raising providers, raising renderers
# =============================================================================


class TestDegradation:
    def test_none_state_is_skipped_and_others_still_render(self) -> None:
        gone: dict[str, ContextState | None] = {"state": None}
        planning: dict[str, ContextState | None] = {"state": _PlanningState(tasks=("ID 1 [t]",))}
        agent = _make_agent(
            [_provider("team_roster_state", gone), _provider("planning_state", planning)]
        )

        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 1
        assert "**Planning:** ID 1 [t]" in blocks[0]
        baselines = agent._context_baselines  # type: ignore[attr-defined]
        assert "team_roster_state" not in baselines

    def test_raising_provider_is_skipped_logged_and_turn_completes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        planning: dict[str, ContextState | None] = {"state": _PlanningState(tasks=("ID 1 [t]",))}
        agent = _make_agent(
            [_raising_provider("team_roster_state"), _provider("planning_state", planning)]
        )

        with caplog.at_level(logging.ERROR, logger="akgentic.agent.agent"):
            agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 1
        assert "**Planning:** ID 1 [t]" in blocks[0]
        assert any("team_roster_state" in r.message for r in caplog.records)
        baselines = agent._context_baselines  # type: ignore[attr-defined]
        assert "team_roster_state" not in baselines

    def test_raising_renderer_is_skipped_and_baseline_does_not_advance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = _ExplodingDeltaState(text="First snapshot.")
        holder: dict[str, ContextState | None] = {"state": first}
        agent = _make_agent([_provider("planning_state", holder)])

        agent._deliver_context_update()  # full render — fine
        holder["state"] = _ExplodingDeltaState(text="Second snapshot.")
        with caplog.at_level(logging.ERROR, logger="akgentic.agent.agent"):
            agent._deliver_context_update()  # delta raises — skipped

        assert len(_recorded_blocks(agent)) == 1
        assert any("planning_state" in r.message for r in caplog.records)
        baselines = agent._context_baselines  # type: ignore[attr-defined]
        assert baselines["planning_state"] is first


# =============================================================================
# AC 2 — single delivery site: top of act(), before run_sync
# =============================================================================


class TestActDeliverySite:
    def test_act_delivers_before_run_sync(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        order: list[str] = []
        record = agent._react_agent.context.record_operator_action  # type: ignore[attr-defined]
        record.side_effect = lambda entry: order.append("record")
        agent._react_agent.run_sync.side_effect = (  # type: ignore[attr-defined]
            lambda *a, **k: order.append("run_sync") or "response"
        )

        agent.act("hello", output_type=str)

        assert order == ["record", "run_sync"]


# =============================================================================
# AC 5 / AC 6 — delivery primitive and the pre-first-run fold (real manager)
# =============================================================================


class TestFirstRunFold:
    def test_fresh_agent_block_is_buffered_and_folded_not_a_standalone_message(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        context = ContextManager()
        agent._react_agent.context = context  # type: ignore[attr-defined]

        agent._deliver_context_update()

        # No standalone message: the message_history handed to the first iter()
        # is context.messages, and it must still be empty so pydantic-ai's
        # system-prompt injection is not suppressed.
        assert context.messages == []

        # The block reaches the model folded into the run's user prompt, by the
        # real ReactAgent fold over the real buffer.
        folded = ReactAgent._fold_pending_operator_actions(
            SimpleNamespace(_context=context),  # type: ignore[arg-type]
            "hello",
        )
        assert isinstance(folded, str)
        assert folded.startswith("**Context update 1** — current state.")
        assert folded.endswith("hello")
        assert context.messages == []  # folding never materializes history

    def test_after_first_run_block_is_appended_as_user_role_message(self) -> None:
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        context = ContextManager()
        agent._react_agent.context = context  # type: ignore[attr-defined]

        # Simulate a completed first run: history is non-empty now.
        context.add_message(ModelRequest(parts=[UserPromptPart(content="earlier turn")]))

        agent._deliver_context_update()

        assert len(context.messages) == 2
        appended = context.messages[-1]
        assert isinstance(appended, ModelRequest)
        part = appended.parts[0]
        assert isinstance(part, UserPromptPart)
        assert str(part.content).startswith("**Context update 1** — current state.")


# =============================================================================
# AC 1 — on_start collects the factory's providers (real actor system)
# =============================================================================


class TestOnStartCollection:
    """on_start holds tool_factory.get_context_states() for the agent's lifetime.

    Exercised through the real actor system (on_start runs on createActor) with
    a capturing fake ReactAgent, and asserted through behavior: a card exposing
    a provider gets its rendering delivered in the first turn's block.
    """

    def test_card_provider_is_collected_and_delivered_on_first_turn(self) -> None:
        import akgentic.agent.agent as agent_module
        from akgentic.agent.messages import AgentMessage
        from akgentic.agent.output_models import StructuredOutput
        from akgentic.core import ActorSystem, BaseConfig, Orchestrator
        from akgentic.llm import ModelConfig, PromptTemplate

        from akgentic.tool.core import ToolCard

        class _StateCard(ToolCard):
            def get_tools(self) -> list[Callable[..., Any]]:
                return []

            def get_context_states(self) -> list[Callable[[], ContextState | None]]:
                def demo_state() -> ContextState | None:
                    return _PlanningState(tasks=("ID 9 [demo]",))

                return [demo_state]

        recorded: list[str] = []

        class _CapturingReactAgent:
            def __init__(self, **kwargs: object) -> None:
                self.context = SimpleNamespace(record_operator_action=recorded.append)

            def system_prompt(self, fn: object) -> object:
                return fn

            def run_sync(self, prompt: object, **kwargs: object) -> StructuredOutput:
                return StructuredOutput(messages=[])

            def close(self) -> None:
                pass

        system = ActorSystem()
        original = agent_module.ReactAgent
        agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[misc, assignment]
        try:
            orch_addr = system.createActor(
                Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
            )
            orchestrator = system.proxy_ask(orch_addr, Orchestrator)

            config = AgentConfig(
                name="@Manager",
                role="Manager",
                prompt=PromptTemplate(template="You are a manager."),
                model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
                tools=[_StateCard()],
            )
            manager_addr = orchestrator.createActor(BaseAgent, config=config)
            time.sleep(0.3)

            system.tell(manager_addr, AgentMessage(content="hello"))

            deadline = time.time() + 10
            while not recorded and time.time() < deadline:
                time.sleep(0.2)

            assert len(recorded) == 1, "first turn must deliver exactly one block"
            assert recorded[0].startswith("**Context update 1** — current state.")
            assert "**Planning:** ID 9 [demo]" in recorded[0]
        finally:
            agent_module.ReactAgent = original  # type: ignore[misc]
            try:
                system.shutdown(timeout=5)
            except Exception:
                pass
