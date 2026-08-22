"""The restore path: a persisted slot arriving through a real ``init_state()``.

This is the payoff of persisting the baselines. Before, a restored agent held no
baselines and re-appended the whole roster, plan and graph summary on its first
turn — content the model could already see one page up. Now the slot rides
``AgentState`` through the restore, and the updater's reconciliation decides
whether to trust it by looking for its marker in the history that came back.

Three outcomes, one per scenario the trust rules allow:

- marker present, nothing changed  → nothing appended at all
- marker gone (compacted/cleared)  → a full snapshot, worded as current state
- persisted counter behind the history → numbering catches up, baselines kept

A fourth spec guards the property all three rest on: the slot is dereferenced on
every call, never captured, so a delivery after ``init_state()`` follows the
state that just arrived rather than the one it replaced.

The agent under test is assembled the way ``test_context_self_healing.py``
assembles one: no Pykka, a stubbed ``ReactAgent`` whose ``record_operator_action``
appends a user-role message to a real history list, and a real ``ContextUpdater``
so the reconciliation being exercised is the shipped one.
"""

from collections.abc import Callable
from typing import Any, Self
from unittest.mock import MagicMock

from akgentic.tool.core import ContextState, ContextUpdater
from pydantic_ai.messages import ModelRequest, UserPromptPart

from akgentic.agent.agent import BaseAgent
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


def _make_agent(providers: list[Callable[[], Any]], history: list[Any]) -> BaseAgent:
    """A bare agent whose context history is ``history``, seeded by the caller.

    The seeded history stands in for what ``init_llm_context()`` replays after a
    restore; what matters to the updater is only what the messages contain.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = MagicMock()  # type: ignore[attr-defined]
    agent._react_agent.context.messages = history  # type: ignore[attr-defined]
    agent._react_agent.context.record_operator_action.side_effect = (  # type: ignore[attr-defined]
        lambda entry: history.append(ModelRequest(parts=[UserPromptPart(content=entry)]))
    )

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]
    agent._context_updater = ContextUpdater(agent, providers)  # type: ignore[attr-defined]
    return agent


def _delivered_block(agent: BaseAgent) -> str:
    """The single block appended by this turn, or ``""`` when none was."""
    calls = agent._react_agent.context.record_operator_action.call_args_list  # type: ignore[attr-defined]
    assert len(calls) <= 1, f"expected at most one block, got {len(calls)}"
    return str(calls[0].args[0]) if calls else ""


def _persisted_state(members: tuple[str, ...], seq: int) -> AgentState:
    """The state as it would come back off the event store, mid-restore.

    Round-tripped through ``model_dump`` / ``model_validate`` deliberately: a
    restored slot has been through serialization, and the baseline must still be
    the concrete subclass for the diff to work at all.
    """
    saved = AgentState(backstory="You are a test agent.")
    saved.tool_state.context_baselines["team_roster_state"] = _RosterState(members=members)
    saved.tool_state.context_update_seq = seq
    return AgentState.model_validate(saved.model_dump())


def _marker_message(number: int) -> ModelRequest:
    """A previously delivered block, as it sits in a restored history."""
    return ModelRequest(
        parts=[
            UserPromptPart(
                content=(
                    f"**Context update {number}** — current state.\n\n**Team roster:**\n@Manager"
                )
            )
        ]
    )


# =============================================================================
# AC 7 — a restore whose marker survived delivers a delta, never a snapshot
# =============================================================================


class TestRestoreWithSurvivingMarker:
    def test_unchanged_state_appends_nothing_at_all(self) -> None:
        """The headline: a restart no longer costs every agent a full snapshot."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)], [_marker_message(4)])
        agent.init_state(_persisted_state(("@Manager",), seq=4))

        agent._deliver_context_update()

        assert _delivered_block(agent) == ""
        assert agent.state.tool_state.context_update_seq == 4

    def test_a_changed_provider_delivers_a_delta_worded_block(self) -> None:
        holder: dict[str, ContextState | None] = {
            "state": _RosterState(members=("@Manager", "@Coder"))
        }
        agent = _make_agent([_provider("team_roster_state", holder)], [_marker_message(4)])
        agent.init_state(_persisted_state(("@Manager",), seq=4))

        agent._deliver_context_update()

        assert _delivered_block(agent) == (
            "**Context update 5** — state has changed since the last update.\n\n"
            "@Coder joined the team."
        )
        assert agent.state.tool_state.context_update_seq == 5


# =============================================================================
# AC 8 — a history that lost its blocks re-snapshots
# =============================================================================


class TestRestoreWithEvictedMarker:
    def test_missing_marker_drops_the_baselines_and_re_snapshots(self) -> None:
        """Compacted, cleared or trimmed away: the repair is a full snapshot."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        summarised = ModelRequest(
            parts=[UserPromptPart(content="Summary of the conversation so far: the team met.")]
        )
        agent = _make_agent([_provider("team_roster_state", holder)], [summarised])
        agent.init_state(_persisted_state(("@Manager",), seq=4))

        agent._deliver_context_update()

        assert _delivered_block(agent) == (
            "**Context update 5** — current state.\n\n**Team roster:**\n@Manager"
        )


# =============================================================================
# The slot is read live, never captured — the trap all of the above rests on
# =============================================================================


class TestStateReplacedMidLife:
    def test_delivery_follows_the_new_slot_after_init_state(self) -> None:
        """The slot is read live, never captured — the decision's stated first trap.

        ``init_state()`` replaces the state object wholesale, so an engine that
        held on to the old ``ToolState`` would keep advancing a slot nobody
        persists any more. Two deliveries either side of a replacement is the
        only shape that catches it: a single post-restore delivery passes just
        as well against a captured reference.
        """
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        history: list[Any] = []
        agent = _make_agent([_provider("team_roster_state", holder)], history)

        agent._deliver_context_update()
        assert agent.state.tool_state.context_update_seq == 1

        # A fresh, empty slot arrives with a history that carries no marker.
        history.clear()
        agent.init_state(AgentState(backstory="You are a test agent."))
        agent._deliver_context_update()

        # Read live: an empty slot against an empty history is a first block.
        # Held captive: the old slot's seq of 1 would number this one 2.
        blocks = [
            call.args[0]
            for call in agent._react_agent.context.record_operator_action.call_args_list  # type: ignore[attr-defined]
        ]
        assert blocks[1] == ("**Context update 1** — current state.\n\n**Team roster:**\n@Manager")
        assert agent.state.tool_state.context_update_seq == 1


# =============================================================================
# AC 9 — a stale persisted counter catches up to the history
# =============================================================================


class TestStalePersistedCounter:
    def test_counter_catches_up_and_baselines_are_kept(self) -> None:
        """A crash after the append but before the checkpoint: repeat, never omit.

        The history proves block 6 was delivered while the save still says 4. The
        counter follows the history, and the baselines are kept — so the next
        delta is computed against the older baseline and re-states what the
        missed blocks said. The failure mode is a repeat, which is the whole
        reason the baselines are allowed to be a cache.
        """
        holder: dict[str, ContextState | None] = {
            "state": _RosterState(members=("@Manager", "@Coder"))
        }
        agent = _make_agent(
            [_provider("team_roster_state", holder)],
            [_marker_message(5), _marker_message(6)],
        )
        agent.init_state(_persisted_state(("@Manager",), seq=4))

        agent._deliver_context_update()

        assert _delivered_block(agent) == (
            "**Context update 7** — state has changed since the last update.\n\n"
            "@Coder joined the team."
        )
        assert agent.state.tool_state.context_update_seq == 7
