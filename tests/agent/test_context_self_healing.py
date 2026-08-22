"""Tests for the two self-healing properties the agent still owns (Epics 19, 21).

``clear()`` is the one legitimate zeroing of the context-update state: it wipes
the history, so it resets the persisted slot through
``ContextUpdater.reset()``. ``compact()`` deliberately gets no reset — the
updater's own reconciliation against the visible history catches a folded-away
block, including the automatic compaction a ``compact()`` hook would miss.

The prefix-stability property (NFR1) is the other: two consecutive no-change
turns must leave the ``SystemPromptPart``s of ``messages[0]`` byte-identical and
add no message beyond the model's own. It runs a real ``ReactAgent`` over a
token-free pydantic-ai ``FunctionModel``, because the property under guard is
pydantic-ai's dynamic system-prompt re-evaluation.

Every other eviction and restore scenario the old story-19-2 suite covered is
engine behaviour now, tested in ``akgentic-tool``; the restore path through a
real ``init_state()`` is covered in ``test_context_restore.py``.
"""

from collections.abc import Callable
from typing import Any, Self
from unittest.mock import MagicMock

from akgentic.llm import ModelConfig, ReactAgent, ReactAgentConfig
from akgentic.tool.core import ContextState, ContextUpdater
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from akgentic.agent.agent import BaseAgent, MailboxCancelCapability
from akgentic.agent.config import AgentConfig, AgentState

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


def _provider(name: str, holder: dict[str, ContextState | None]) -> Callable[[], Any]:
    """A named provider reading its current state from a mutable holder."""

    def provider() -> ContextState | None:
        return holder["state"]

    provider.__name__ = name
    return provider


def _make_agent(providers: list[Callable[[], Any]] | None = None) -> BaseAgent:
    """Bare BaseAgent (no Pykka) with a stubbed ReactAgent and a real updater.

    ``record_operator_action`` appends each block as a user-role message to a
    real ``messages`` list (the post-first-run shape), so the updater's
    reconciliation scans the same history it would against the real
    ``ContextManager``. The updater holds the agent weakly, so the caller must
    keep the returned agent alive.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = MagicMock()  # type: ignore[attr-defined]
    messages: list[Any] = []
    agent._react_agent.context.messages = messages  # type: ignore[attr-defined]
    agent._react_agent.context.record_operator_action.side_effect = (  # type: ignore[attr-defined]
        lambda entry: messages.append(ModelRequest(parts=[UserPromptPart(content=entry)]))
    )

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]
    agent._context_updater = ContextUpdater(  # type: ignore[attr-defined]
        agent, providers or []
    )

    # Cancel capability normally built in _build_react_agent (Epic 20).
    agent._cancel_capability = MailboxCancelCapability(observer=agent)  # type: ignore[arg-type]
    return agent


def _recorded_blocks(agent: BaseAgent) -> list[str]:
    record = agent._react_agent.context.record_operator_action  # type: ignore[attr-defined]
    return [call.args[0] for call in record.call_args_list]


def _messages(agent: BaseAgent) -> list[Any]:
    return agent._react_agent.context.messages  # type: ignore[attr-defined, no-any-return]


# =============================================================================
# AC 5 — clear() resets through the updater; compact() gains no reset call
# =============================================================================


class TestClearReset:
    def test_clear_zeroes_the_persisted_slot(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()
        assert agent.state.tool_state.context_update_seq == 1

        history = _messages(agent)
        agent._react_agent.clear_context.side_effect = (  # type: ignore[attr-defined]
            lambda: history.clear() or "Cleared."
        )
        result = agent.clear()

        assert result == "Cleared."
        assert agent.state.tool_state.context_baselines == {}
        assert agent.state.tool_state.context_update_seq == 0

    def test_clear_resets_through_the_updater_not_by_touching_the_slot(self) -> None:
        """The agent owns no baseline field of its own — reset() is the path."""
        agent = _make_agent()
        agent._context_updater = MagicMock()  # type: ignore[attr-defined]
        agent._react_agent.clear_context.return_value = "Cleared."  # type: ignore[attr-defined]

        assert agent.clear() == "Cleared."

        agent._context_updater.reset.assert_called_once_with()  # type: ignore[attr-defined]

    def test_next_block_after_clear_is_update_1_current_state(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()

        history = _messages(agent)
        agent._react_agent.clear_context.side_effect = (  # type: ignore[attr-defined]
            lambda: history.clear() or "Cleared."
        )
        agent.clear()
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        assert blocks[1] == (
            "**Context update 1** — current state.\n\n**Team roster:**\n@Manager"
        )

    def test_compact_gains_no_reset_call(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()
        baselines_before = dict(agent.state.tool_state.context_baselines)

        agent.compact()

        # compact() delegates only; the updater's reconciliation owns the repair.
        assert agent.state.tool_state.context_baselines == baselines_before
        assert agent.state.tool_state.context_update_seq == 1


# =============================================================================
# NFR1 — prefix stability: real ReactAgent, token-free FunctionModel
# =============================================================================


def _stub_model(messages: list[Any], info: AgentInfo) -> ModelResponse:
    """Token-free FunctionModel target: a fixed one-part text response."""
    return ModelResponse(parts=[TextPart(content="ok")])


def _real_react_agent_pair() -> tuple[ReactAgent, BaseAgent, dict[str, ContextState | None]]:
    """Bare BaseAgent over a REAL ReactAgent, with one roster provider.

    The real agent keeps pydantic-ai's dynamic system-prompt re-evaluation
    path intact — the property NFR1 guards, which the stubbed ``_make_agent``
    cannot exercise. The caller registers any dynamic system prompts before
    the first run and owns ``react_agent.close()``, and must keep the returned
    agent alive: the updater holds it weakly.
    """
    react_agent = ReactAgent(
        config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
        deps_type=BaseAgent,
    )
    holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = react_agent  # type: ignore[attr-defined]
    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]
    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]
    agent._context_updater = ContextUpdater(  # type: ignore[attr-defined]
        agent, [_provider("team_roster_state", holder)]
    )

    # Cancel capability normally built in _build_react_agent (Epic 20).
    agent._cancel_capability = MailboxCancelCapability(observer=agent)  # type: ignore[arg-type]
    return react_agent, agent, holder


class TestPrefixStability:
    def test_no_change_turns_leave_system_parts_byte_identical(self) -> None:
        """Two no-change turns: messages[0]'s system parts stay byte-identical
        and each turn adds exactly one ModelRequest + one ModelResponse.

        The real dynamic-re-evaluation path is the property under guard:
        ``ReactAgent.system_prompt`` registers with ``dynamic=True``, so
        pydantic-ai re-runs every registered closure on each turn and rewrites
        ``messages[0]`` in place — the spec proves the re-evaluated content is
        byte-identical because nothing volatile is registered any more.
        """
        react_agent, agent, holder = _real_react_agent_pair()
        try:
            @react_agent.system_prompt
            def agent_backstory(ctx: RunContext[BaseAgent]) -> str:
                return "You are a test agent with a fixed backstory."

            @react_agent.system_prompt
            def current_date(ctx: RunContext[BaseAgent]) -> str:
                return "The current date is 2026-08-22."

            with react_agent.pydantic_agent.override(model=FunctionModel(_stub_model)):
                agent.act("turn one", output_type=str)
                first_system = [
                    p.content
                    for p in react_agent.context.messages[0].parts
                    if isinstance(p, SystemPromptPart)
                ]
                count_after_first = len(react_agent.context.messages)
                holder["state"] = _RosterState(members=("@Manager",))
                agent.act("turn two", output_type=str)

            second_system = [
                p.content
                for p in react_agent.context.messages[0].parts
                if isinstance(p, SystemPromptPart)
            ]
            assert first_system, "the first request must carry system prompt parts"
            assert second_system == first_system

            # No message beyond the model's own: one ModelRequest + one
            # ModelResponse per turn — no context-update message on turn two.
            assert count_after_first == 2
            assert len(react_agent.context.messages) == 4
            assert isinstance(react_agent.context.messages[2], ModelRequest)
            assert isinstance(react_agent.context.messages[3], ModelResponse)
        finally:
            react_agent.close()

    def test_first_turn_block_is_folded_and_marker_found_next_turn(self) -> None:
        """The fresh-agent fold lands the marker in a UserPromptPart; the
        updater's reconciliation finds it there with no fold-specific branch."""
        react_agent, agent, holder = _real_react_agent_pair()
        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(_stub_model)):
                agent.act("turn one", output_type=str)
                # The block was folded into turn one's user prompt.
                user_texts = [
                    p.content
                    for p in react_agent.context.messages[0].parts
                    if isinstance(p, UserPromptPart)
                ]
                assert any("**Context update 1**" in str(t) for t in user_texts)
                # Unchanged second turn: reconciliation finds the folded marker,
                # so the surviving baseline appends nothing.
                holder["state"] = _RosterState(members=("@Manager",))
                agent.act("turn two", output_type=str)

            assert agent.state.tool_state.context_update_seq == 1
            assert len(react_agent.context.messages) == 4
        finally:
            react_agent.close()
