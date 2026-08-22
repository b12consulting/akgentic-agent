"""``AgentState.tool_state``: the slot exists, defaults empty, and survives serialization.

The slot is what carries the context-update baselines and the block counter across a
restore. Its one hard requirement beyond existing is polymorphic round-tripping: the
baselines are keyed by provider name and hold *concrete* ``ContextState`` subclasses,
so a round-trip that returns the abstract base — or the wrong subclass — would silently
turn every restored delta into a full snapshot.
"""

from typing import Self

from akgentic.tool.core import ContextState, ToolState

from akgentic.agent.config import AgentState


class RosterState(ContextState):
    """A concrete context state — one of two deliberately different types."""

    members: list[str] = []

    def render_full(self) -> str:
        return f"Roster: {', '.join(self.members)}"

    def render_delta(self, previous: Self) -> str | None:
        if previous.members == self.members:
            return None
        return f"Roster now: {', '.join(self.members)}"


class PlanState(ContextState):
    """A second concrete context state, structurally unlike ``RosterState``."""

    step: int = 0
    note: str = ""

    def render_full(self) -> str:
        return f"Plan step {self.step}: {self.note}"

    def render_delta(self, previous: Self) -> str | None:
        if previous.step == self.step:
            return None
        return f"Plan advanced to step {self.step}"


def _seeded_state() -> AgentState:
    """An ``AgentState`` whose slot holds baselines of two different concrete types."""
    state = AgentState(backstory="You are a coordinator.")
    state.tool_state.context_baselines["roster"] = RosterState(members=["alice", "bob"])
    state.tool_state.context_baselines["plan"] = PlanState(step=3, note="draft the report")
    state.tool_state.context_update_seq = 7
    return state


def test_default_agent_state_carries_an_empty_slot() -> None:
    """AC 1: the field exists, defaults empty, and no other field changed."""
    state = AgentState(backstory="You are a coordinator.")

    assert isinstance(state.tool_state, ToolState)
    assert state.tool_state.context_baselines == {}
    assert state.tool_state.context_update_seq == 0
    assert state.backstory == "You are a coordinator."


def test_serializable_copy_preserves_both_concrete_baseline_types() -> None:
    """AC 2: ``serializable_copy()`` keeps both baselines and their concrete classes."""
    copied = _seeded_state().serializable_copy()

    assert isinstance(copied, AgentState)
    assert copied.tool_state.context_update_seq == 7
    roster = copied.tool_state.context_baselines["roster"]
    plan = copied.tool_state.context_baselines["plan"]
    assert type(roster) is RosterState
    assert type(plan) is PlanState
    assert roster.members == ["alice", "bob"]
    assert plan.step == 3
    assert plan.note == "draft the report"


def test_model_dump_validate_round_trip_preserves_both_concrete_baseline_types() -> None:
    """AC 2: the ``__model__`` stamp does the polymorphic work — proven, not assumed."""
    dumped = _seeded_state().model_dump()

    # The stamp is what makes the reconstruction polymorphic; without it both
    # baselines would come back as the abstract base (or fail to construct).
    assert dumped["tool_state"]["context_baselines"]["roster"]["__model__"].endswith("RosterState")
    assert dumped["tool_state"]["context_baselines"]["plan"]["__model__"].endswith("PlanState")

    restored = AgentState.model_validate(dumped)

    assert restored.tool_state.context_update_seq == 7
    assert type(restored.tool_state.context_baselines["roster"]) is RosterState
    assert type(restored.tool_state.context_baselines["plan"]) is PlanState
    assert restored.tool_state.context_baselines["roster"].members == ["alice", "bob"]


def test_restored_baselines_still_diff_against_a_live_state() -> None:
    """A round-tripped baseline is usable as a baseline, not just structurally equal."""
    restored = AgentState.model_validate(_seeded_state().model_dump())

    baseline = restored.tool_state.context_baselines["roster"]
    assert RosterState(members=["alice", "bob"]).render_delta(baseline) is None
    assert RosterState(members=["alice", "bob", "carol"]).render_delta(baseline) == (
        "Roster now: alice, bob, carol"
    )
