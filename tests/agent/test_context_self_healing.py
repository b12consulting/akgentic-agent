"""Tests for context-update self-healing (Epic 19, story 19-2).

Before reading any provider, ``_deliver_context_update`` verifies that its last
delivered **Context update** marker is still visible in the user-role prompt
parts of the context history; a miss drops every baseline so the next block is
a full snapshot — one rule covering manual ``compact()``, automatic compaction,
sliding-window trimming, restore, and any out-of-band wipe. ``clear()`` resets
baselines and counter directly, and the frozen system block stays byte-identical
across no-change turns (the prefix-stability guard runs a real ``ReactAgent``
over a token-free pydantic-ai ``FunctionModel``).

Most specs use the bare-agent + stubbed-ReactAgent pattern from story 19-1's
suite; the stub context carries a real ``messages`` list whose delivered blocks
land as user-role messages, so the presence check sees exactly the post-run
shape the real ``ContextManager`` would give it.
"""

from collections.abc import Callable
from typing import Any, Self
from unittest.mock import MagicMock

from akgentic.llm import ModelConfig, ReactAgent, ReactAgentConfig
from akgentic.tool.core import ContextState
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig

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
    """Bare BaseAgent (no Pykka) with a stubbed ReactAgent and given providers.

    ``record_operator_action`` appends each block as a user-role message to a
    real ``messages`` list (the post-first-run shape), so the presence check
    scans the same history it would against the real ``ContextManager``.
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

    agent._context_state_providers = providers or []  # type: ignore[attr-defined]
    agent._context_baselines = {}  # type: ignore[attr-defined]
    agent._context_update_seq = 0  # type: ignore[attr-defined]
    return agent


def _recorded_blocks(agent: BaseAgent) -> list[str]:
    record = agent._react_agent.context.record_operator_action  # type: ignore[attr-defined]
    return [call.args[0] for call in record.call_args_list]


def _messages(agent: BaseAgent) -> list[Any]:
    return agent._react_agent.context.messages  # type: ignore[attr-defined, no-any-return]


FULL_ROSTER_BLOCK_2 = "**Context update 2** — current state.\n\n**Team roster:**\n@Manager"


# =============================================================================
# AC 3 — clear() resets baselines and counter; compact() gains no reset call
# =============================================================================


class TestClearReset:
    def test_clear_drops_baselines_and_resets_counter(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()
        assert agent._context_update_seq == 1  # type: ignore[attr-defined]

        history = _messages(agent)
        agent._react_agent.clear_context.side_effect = (  # type: ignore[attr-defined]
            lambda: history.clear() or "Cleared."
        )
        result = agent.clear()

        assert result == "Cleared."
        assert agent._context_baselines == {}  # type: ignore[attr-defined]
        assert agent._context_update_seq == 0  # type: ignore[attr-defined]

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
        baselines_before = dict(agent._context_baselines)  # type: ignore[attr-defined]

        agent.compact()

        # compact() delegates only; FR5's presence check owns the repair.
        assert agent._context_baselines == baselines_before  # type: ignore[attr-defined]
        assert agent._context_update_seq == 1  # type: ignore[attr-defined]


# =============================================================================
# AC 1 — presence miss drops baselines; counter stays monotonic
# =============================================================================


class TestEvictionSelfHealing:
    def test_manual_compact_that_folds_block_away_yields_full_snapshot_n_plus_1(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()

        # A compaction summary replaced the history; the marker is gone.
        _messages(agent)[:] = [
            ModelRequest(parts=[UserPromptPart(content="Summary of the conversation so far.")])
        ]
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        # Full snapshot, numbered N+1 (never reset to 1), worded as current state.
        assert blocks[1] == FULL_ROSTER_BLOCK_2

    def test_automatic_compaction_between_act_turns_self_heals(self) -> None:
        """No agent-visible call at all — the case a compact() hook would miss."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._react_agent.run_sync.return_value = "response"  # type: ignore[attr-defined]

        agent.act("turn one", output_type=str)
        # Out-of-band eviction between turns: nothing on the agent was called.
        _messages(agent)[:] = [
            ModelRequest(parts=[UserPromptPart(content="Summary of the conversation so far.")])
        ]
        agent.act("turn two", output_type=str)

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        assert blocks[1] == FULL_ROSTER_BLOCK_2

    def test_sliding_window_trim_dropping_the_block_yields_full_snapshot(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()
        holder["state"] = _RosterState(members=("@Manager", "@Bob"))
        agent._deliver_context_update()  # delta block, marker 2

        # The window trims the newest marker away; marker 1 survives behind it.
        del _messages(agent)[-1]
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 3
        # Marker 2 absent → baselines dropped → full snapshot; the counter is
        # NOT reset by the miss (a partially trimmed history still shows 1).
        assert blocks[2] == (
            "**Context update 3** — current state.\n\n**Team roster:**\n@Manager\n@Bob"
        )

    def test_marker_present_and_no_change_still_appends_nothing(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        agent._deliver_context_update()
        holder["state"] = _RosterState(members=("@Manager",))
        agent._deliver_context_update()

        assert len(_recorded_blocks(agent)) == 1
        assert agent._context_update_seq == 1  # type: ignore[attr-defined]
        assert "team_roster_state" in agent._context_baselines  # type: ignore[attr-defined]


# =============================================================================
# AC 2 — restored agent: numbering continuity, full snapshot, never adoption
# =============================================================================


class TestRestoredAgent:
    def test_restored_history_with_markers_continues_numbering_with_full_snapshot(self) -> None:
        agent = _make_agent(
            [_provider("team_roster_state", {"state": _RosterState(members=("@Manager",))})]
        )
        # A restored life: baselines empty, counter 0, history carries old blocks.
        _messages(agent)[:] = [
            ModelRequest(
                parts=[UserPromptPart(content="**Context update 1** — current state.\n\nold")]
            ),
            ModelResponse(parts=[TextPart(content="ok")]),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=(
                            "**Context update 3** — state has changed since the last update."
                            "\n\n@Bob joined the team."
                        )
                    )
                ]
            ),
        ]

        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        # A full snapshot IS delivered (never a silent baseline adoption), and it
        # continues the sequence past the highest restored marker.
        assert len(blocks) == 1
        assert blocks[0] == (
            "**Context update 4** — current state.\n\n**Team roster:**\n@Manager"
        )
        assert agent._context_update_seq == 4  # type: ignore[attr-defined]

    def test_restored_history_with_no_marker_starts_at_1(self) -> None:
        agent = _make_agent(
            [_provider("team_roster_state", {"state": _RosterState(members=("@Manager",))})]
        )
        _messages(agent)[:] = [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="hi")]),
        ]

        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 1
        assert blocks[0].startswith("**Context update 1** — current state.")


# =============================================================================
# AC 6 / NFR2 — scan scope: user prompt parts only, on ModelRequest only
# =============================================================================


class TestScanScope:
    def test_marker_only_in_tool_return_part_counts_as_absent(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()
        delivered = _recorded_blocks(agent)[0]

        # The only occurrence of the marker now sits in a tool return.
        _messages(agent)[:] = [
            ModelRequest(
                parts=[ToolReturnPart(tool_name="echo", content=delivered, tool_call_id="c1")]
            )
        ]
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        assert blocks[1] == FULL_ROSTER_BLOCK_2

    def test_marker_only_in_model_response_counts_as_absent(self) -> None:
        """A model echoing the marker verbatim must not satisfy the check."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()
        delivered = _recorded_blocks(agent)[0]

        _messages(agent)[:] = [ModelResponse(parts=[TextPart(content=delivered)])]
        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 2
        assert blocks[1] == FULL_ROSTER_BLOCK_2

    def test_marker_in_multimodal_list_content_counts_as_present(self) -> None:
        """A block folded into a multimodal prompt is found in the str items."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        agent._deliver_context_update()
        delivered = _recorded_blocks(agent)[0]

        _messages(agent)[:] = [
            ModelRequest(parts=[UserPromptPart(content=[delivered, "and an image ref"])])
        ]
        holder["state"] = _RosterState(members=("@Manager",))
        agent._deliver_context_update()

        # Marker present + no change → nothing appended, baselines intact.
        assert len(_recorded_blocks(agent)) == 1
        assert agent._context_update_seq == 1  # type: ignore[attr-defined]


# =============================================================================
# AC 5 / NFR1 — prefix stability: real ReactAgent, token-free FunctionModel
# =============================================================================


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
        react_agent = ReactAgent(
            config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
            deps_type=BaseAgent,
        )
        try:
            @react_agent.system_prompt
            def agent_backstory(ctx: RunContext[BaseAgent]) -> str:
                return "You are a test agent with a fixed backstory."

            @react_agent.system_prompt
            def current_date(ctx: RunContext[BaseAgent]) -> str:
                return "The current date is 2026-08-22."

            holder: dict[str, ContextState | None] = {
                "state": _RosterState(members=("@Manager",))
            }
            agent: BaseAgent = object.__new__(BaseAgent)
            agent._react_agent = react_agent  # type: ignore[attr-defined]
            registry = MagicMock()
            registry.has.return_value = False
            agent._command_registry = registry  # type: ignore[attr-defined]
            mock_config = MagicMock(spec=AgentConfig)
            mock_config.name = "@TestAgent"
            agent.config = mock_config  # type: ignore[attr-defined]
            agent._context_state_providers = [  # type: ignore[attr-defined]
                _provider("team_roster_state", holder)
            ]
            agent._context_baselines = {}  # type: ignore[attr-defined]
            agent._context_update_seq = 0  # type: ignore[attr-defined]

            def stub_model(messages: list[Any], info: AgentInfo) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content="ok")])

            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
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
        """The fresh-agent fold lands the marker in a UserPromptPart; the scan
        finds it there with no fold-specific branch (ADR-037 §7)."""
        react_agent = ReactAgent(
            config=ReactAgentConfig(model_cfg=ModelConfig(provider="openai", model="gpt-4o")),
            deps_type=BaseAgent,
        )
        try:
            holder: dict[str, ContextState | None] = {
                "state": _RosterState(members=("@Manager",))
            }
            agent: BaseAgent = object.__new__(BaseAgent)
            agent._react_agent = react_agent  # type: ignore[attr-defined]
            registry = MagicMock()
            registry.has.return_value = False
            agent._command_registry = registry  # type: ignore[attr-defined]
            mock_config = MagicMock(spec=AgentConfig)
            mock_config.name = "@TestAgent"
            agent.config = mock_config  # type: ignore[attr-defined]
            agent._context_state_providers = [  # type: ignore[attr-defined]
                _provider("team_roster_state", holder)
            ]
            agent._context_baselines = {}  # type: ignore[attr-defined]
            agent._context_update_seq = 0  # type: ignore[attr-defined]

            def stub_model(messages: list[Any], info: AgentInfo) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content="ok")])

            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                agent.act("turn one", output_type=str)
                # The block was folded into turn one's user prompt.
                user_texts = [
                    p.content
                    for p in react_agent.context.messages[0].parts
                    if isinstance(p, UserPromptPart)
                ]
                assert any("**Context update 1**" in str(t) for t in user_texts)
                # Unchanged second turn: the presence check finds the folded
                # marker, so the surviving baseline appends nothing.
                holder["state"] = _RosterState(members=("@Manager",))
                agent.act("turn two", output_type=str)

            assert agent._context_update_seq == 1  # type: ignore[attr-defined]
            assert len(react_agent.context.messages) == 4
        finally:
            react_agent.close()
