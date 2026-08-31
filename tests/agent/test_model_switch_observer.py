"""``BaseAgent`` as a ``ModelSwitchToolObserver``: the roster projection and the switch.

The two methods ``ModelTool`` calls, and the one mapping in the framework allowed
to see both a ``ModelConfig`` and a ``ModelRow`` — ``akgentic-agent`` is the only
package that may import ``akgentic-llm`` and ``akgentic-tool`` at once.

Three properties carry the file:

- the projection is rebuilt per call and keyed by the **imported**
  ``model_roster_key``, so a re-spelled grammar cannot pass;
- an active model no roster entry matches is tolerated — all rows come back
  ``active=False`` and nothing raises. Hand-setting ``model_roster`` is legal, so
  this is a reachable configuration, not a hypothetical one;
- a refused switch **raises**. It must never come back as a friendly string:
  ``ModelTool._switch_model_factory`` writes ``ToolState.active_model`` right
  after the observer call returns normally, so a returned message would persist a
  key the llm layer had just refused. The AC-6 spec drives the real card and
  asserts the slot is untouched.

The agent is assembled the way ``test_context_restore.py`` assembles one: no
Pykka, a stubbed ``ReactAgent`` — here one with real roster semantics.
"""

import logging
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.llm import ModelSwitchError, model_roster_key
from akgentic.llm.config import ModelConfig
from akgentic.tool.errors import RetriableError
from akgentic.tool.model import ModelRow, ModelSwitchToolObserver, ModelTool

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig, AgentState

# =============================================================================
# HELPERS
# =============================================================================

# Three distinguishable entries: distinct providers, distinct model names, and
# distinct context lengths, so a transposed or truncated projection cannot pass.
FAST = ModelConfig(provider="openai", model="gpt-5-mini", context_length=128_000)
DEEP = ModelConfig(provider="anthropic", model="claude-sonnet-4", context_length=200_000)
CHEAP = ModelConfig(provider="mistral", model="mistral-small", context_length=32_000)

ROSTER = [FAST, DEEP, CHEAP]


class _RosterReactAgent:
    """A ``ReactAgent`` stand-in with the roster semantics the real one has.

    ``switch_model`` resolves against the roster, installs the entry it found and
    returns it, and raises ``ModelSwitchError`` — and only that — on a key no
    entry carries. Every call is recorded, because "how many times was the switch
    attempted" is itself an acceptance criterion.
    """

    def __init__(self, active: ModelConfig, roster: list[ModelConfig]) -> None:
        self._active = active
        self._roster = list(roster)
        self.switch_calls: list[str] = []

    def active_model(self) -> ModelConfig:
        return self._active

    def model_roster(self) -> list[ModelConfig]:
        return list(self._roster)

    def switch_model(self, key: str) -> ModelConfig:
        self.switch_calls.append(key)
        for entry in self._roster:
            if model_roster_key(entry) == key:
                self._active = entry
                return entry
        available = ", ".join(model_roster_key(entry) for entry in self._roster)
        raise ModelSwitchError(f"cannot switch to '{key}': available keys: {available}")


def _make_agent(react_agent: Any) -> BaseAgent:
    """A bare ``BaseAgent`` over *react_agent*, with a real ``AgentState``.

    The actor ref, the orchestrator address and the team id are the three things
    a real construction supplies that ``object.__new__`` does not; the protocol
    reads them through ``ActorToolObserver``, so a bare instance would fail an
    ``isinstance`` check for reasons that have nothing to do with this story.
    They are supplied here rather than asserted around. The same conformance is
    re-checked against a genuinely constructed agent in
    ``test_model_tool_wiring.py``, where nothing is hand-populated.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = react_agent  # type: ignore[attr-defined]
    agent._actor_ref = MagicMock()  # type: ignore[attr-defined]
    agent._orchestrator = None  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]
    return agent


def _rows_of(agent: BaseAgent) -> list[ModelRow]:
    return agent.list_model_rows()


def _card_switch(agent: BaseAgent) -> Any:
    """The **real** ``ModelTool`` switch closure, bound to *agent* as its observer.

    Not a hand-written slot assignment: the production write to
    ``ToolState.active_model`` lives inside this closure, so a spec that assigns
    the slot itself skips the one line that makes the feature persist anything.
    """
    card = ModelTool().observer(agent)
    tools = card.get_tools()
    switch = next(tool for tool in tools if tool.__name__ == "switch_model")
    return switch


# =============================================================================
# AC 1 — BaseAgent satisfies ModelSwitchToolObserver, and the methods answer
# =============================================================================


class TestProtocolConformance:
    def test_isinstance_reports_conformance(self) -> None:
        agent = _make_agent(_RosterReactAgent(FAST, ROSTER))

        assert isinstance(agent, ModelSwitchToolObserver)

    def test_both_methods_answer_through_a_protocol_typed_reference(self) -> None:
        """``isinstance`` alone proves almost nothing — the protocol is method-presence.

        So the companion: call both through a reference annotated as the
        protocol, exactly as ``ModelTool`` holds it, and assert on the results.
        """
        agent = _make_agent(_RosterReactAgent(FAST, ROSTER))
        observer: ModelSwitchToolObserver = agent

        rows = observer.list_model_rows()
        outcome = observer.switch_model("mistral:mistral-small")

        assert [row.key for row in rows] == [
            "openai:gpt-5-mini",
            "anthropic:claude-sonnet-4",
            "mistral:mistral-small",
        ]
        assert "mistral:mistral-small" in outcome


# =============================================================================
# AC 2 — one row per entry, declaration order, keyed by the imported grammar
# =============================================================================


class TestRosterProjection:
    def test_one_row_per_entry_in_declaration_order(self) -> None:
        rows = _rows_of(_make_agent(_RosterReactAgent(FAST, ROSTER)))

        assert len(rows) == 3
        assert [(row.provider, row.model) for row in rows] == [
            ("openai", "gpt-5-mini"),
            ("anthropic", "claude-sonnet-4"),
            ("mistral", "mistral-small"),
        ]

    def test_keys_are_spelled_by_the_imported_model_roster_key(self) -> None:
        """MUTATION — spell the key ``f"{cfg.model}:{cfg.provider}"`` and this goes red.

        The expectation is computed through the imported function rather than
        written out, so the spec cannot drift from the grammar it pins.
        """
        rows = _rows_of(_make_agent(_RosterReactAgent(FAST, ROSTER)))

        assert [row.key for row in rows] == [model_roster_key(entry) for entry in ROSTER]

    def test_context_length_comes_from_the_matching_entry(self) -> None:
        """Distinct lengths, so a projection that mixes up entries cannot pass."""
        rows = _rows_of(_make_agent(_RosterReactAgent(FAST, ROSTER)))

        assert [row.context_length for row in rows] == [128_000, 200_000, 32_000]

    def test_an_entry_declaring_no_context_length_projects_none(self) -> None:
        undeclared = ModelConfig(provider="azure", model="gpt-4o-mini")
        rows = _rows_of(_make_agent(_RosterReactAgent(undeclared, [undeclared])))

        assert rows[0].context_length is None


# =============================================================================
# AC 3 — exactly one row is active, and it follows a switch
# =============================================================================


class TestActiveFlag:
    def test_exactly_one_row_is_active_and_it_is_the_reported_entry(self) -> None:
        """MUTATION — set ``active=True`` on every row and this goes red."""
        agent = _make_agent(_RosterReactAgent(DEEP, ROSTER))

        rows = _rows_of(agent)

        assert [row.active for row in rows] == [False, True, False]
        assert sum(row.active for row in rows) == 1

    def test_a_fresh_call_after_a_switch_marks_the_new_entry_and_only_it(self) -> None:
        """MUTATION — capture ``model_roster()``/the active key once and reuse it: red.

        Rows are rebuilt per call and nothing is cached, so the second listing
        must disagree with the first.
        """
        react_agent = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react_agent)

        before = _rows_of(agent)
        agent.switch_model("mistral:mistral-small")
        after = _rows_of(agent)

        assert [row.active for row in before] == [True, False, False]
        assert [row.active for row in after] == [False, False, True]

    def test_an_equal_but_distinct_entry_still_lights_up(self) -> None:
        """Identity is the KEY, never the object.

        A hand-set roster can hold an entry that is equal to the active model
        without being the same object. Comparing with ``is`` would light nothing
        up; comparing by key lights the right row.
        """
        twin = ModelConfig(provider="openai", model="gpt-5-mini", context_length=128_000)
        assert twin is not FAST
        agent = _make_agent(_RosterReactAgent(twin, ROSTER))

        assert [row.active for row in _rows_of(agent)] == [True, False, False]


# =============================================================================
# AC 4 — an active model absent from the roster, and the empty roster
# =============================================================================


class TestActiveModelOutsideTheRoster:
    def test_an_absent_active_model_yields_no_active_row_and_no_exception(self) -> None:
        """MUTATION — compute ``active`` with a defaultless ``next(...)`` and this goes red.

        Hand-setting ``model_roster`` is legal (story 22-1 took the duplicate-key
        guard, not the membership guard), so an agent can reach here with an
        active model no entry matches. That is the designed degradation:
        ``ModelTool.active_model_state()`` then composes no block at all.
        """
        stranger = ModelConfig(provider="google-gla", model="gemini-2.5-pro")
        agent = _make_agent(_RosterReactAgent(stranger, ROSTER))

        rows = _rows_of(agent)

        assert len(rows) == 3
        assert [row.active for row in rows] == [False, False, False]

    def test_no_row_is_synthesised_for_the_absent_active_model(self) -> None:
        stranger = ModelConfig(provider="google-gla", model="gemini-2.5-pro")
        agent = _make_agent(_RosterReactAgent(stranger, ROSTER))

        assert "google-gla:gemini-2.5-pro" not in [row.key for row in _rows_of(agent)]

    def test_an_empty_roster_returns_no_rows_at_all(self) -> None:
        """MUTATION — synthesise one row for the active model and this goes red.

        A single-model agent declares no roster, and ``ModelTool`` renders its own
        "no roster" line off exactly this empty list.
        """
        agent = _make_agent(_RosterReactAgent(FAST, []))

        assert _rows_of(agent) == []


# =============================================================================
# AC 5 — the switch delegates and confirms
# =============================================================================


class TestSwitchDelegates:
    def test_the_delegate_is_called_with_the_key(self) -> None:
        react_agent = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react_agent)

        agent.switch_model("anthropic:claude-sonnet-4")

        assert react_agent.switch_calls == ["anthropic:claude-sonnet-4"]
        assert react_agent.active_model() is DEEP

    def test_the_confirmation_names_the_new_key_and_the_next_turn_boundary(self) -> None:
        """Two facts, not a sentence — the wording is free to improve."""
        agent = _make_agent(_RosterReactAgent(FAST, ROSTER))

        outcome = agent.switch_model("anthropic:claude-sonnet-4")

        assert "anthropic:claude-sonnet-4" in outcome
        assert "next turn" in outcome.lower()

    def test_the_key_is_the_delegate_s_own_outcome_not_a_re_derivation(self) -> None:
        """The confirmation names the entry the delegate returned.

        The delegate is free to install an entry whose key is not spelled the way
        the caller spelled it; what the confirmation reports is what came back.
        """
        react_agent = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react_agent)

        outcome = agent.switch_model("mistral:mistral-small")

        assert model_roster_key(react_agent.active_model()) in outcome


# =============================================================================
# AC 6 — a refused switch RAISES; the durable slot stays untouched
# =============================================================================


class TestRefusalRaises:
    def test_a_refusal_raises_retriable_error_carrying_the_refusal_text(self) -> None:
        agent = _make_agent(_RosterReactAgent(FAST, ROSTER))

        with pytest.raises(RetriableError) as caught:
            agent.switch_model("openai:not-in-the-roster")

        assert "openai:not-in-the-roster" in str(caught.value)
        assert "available keys" in str(caught.value)

    def test_the_refusal_is_chained_as_the_cause(self) -> None:
        agent = _make_agent(_RosterReactAgent(FAST, ROSTER))

        with pytest.raises(RetriableError) as caught:
            agent.switch_model("openai:not-in-the-roster")

        assert isinstance(caught.value.__cause__, ModelSwitchError)

    def test_the_real_card_leaves_the_durable_slot_untouched_on_a_refusal(self) -> None:
        """The whole reason a refusal must raise rather than return a message.

        ``ModelTool._switch_model_factory`` writes
        ``observer.state.tool_state.active_model = model`` immediately after the
        observer call returns normally. An observer that answered "cannot switch:
        ..." would be read as a success and persist a key the llm layer had just
        refused — a lie in durable state, with no error anywhere.

        MUTATION — ``return str(exc)`` instead of raising, and this goes red: the
        slot comes back holding the refused key.
        """
        react_agent = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react_agent)
        switch = _card_switch(agent)

        with pytest.raises(RetriableError):
            switch("openai:not-in-the-roster")

        # Non-vacuous: the delegate really was consulted, so the raise came from
        # the refusal and not from the observer method being missing entirely.
        assert react_agent.switch_calls == ["openai:not-in-the-roster"]
        assert agent.state.tool_state.active_model is None

    def test_the_real_card_records_the_key_on_a_successful_switch(self) -> None:
        """The other half: the write does happen when the switch is accepted.

        Without it the refusal spec above would pass against a card that never
        writes at all.
        """
        agent = _make_agent(_RosterReactAgent(FAST, ROSTER))
        switch = _card_switch(agent)

        switch("mistral:mistral-small")

        assert agent.state.tool_state.active_model == "mistral:mistral-small"


# =============================================================================
# AC 7 — only ModelSwitchError is caught, and the rule is tested behaviourally
# =============================================================================


class _UnrelatedSwitchError(RuntimeError):
    """Distinctive, and deliberately NOT a ``ModelSwitchError``."""


class _ExplodingReactAgent(_RosterReactAgent):
    def switch_model(self, key: str) -> ModelConfig:
        raise _UnrelatedSwitchError("the http client is closed")


class TestOnlyModelSwitchErrorIsCaught:
    def test_a_non_model_switch_error_propagates_unchanged(self) -> None:
        """MUTATION — widen the ``except`` to ``Exception`` and this goes red.

        Behavioural, not a source-text assertion: same class, same message, and
        not wrapped in ``RetriableError``.
        """
        agent = _make_agent(_ExplodingReactAgent(FAST, ROSTER))

        with pytest.raises(_UnrelatedSwitchError) as caught:
            agent.switch_model("mistral:mistral-small")

        assert str(caught.value) == "the http client is closed"
        assert not isinstance(caught.value, RetriableError)

    def test_the_card_still_reports_an_unrelated_failure_as_retriable(self) -> None:
        """The card's own broad catch is what turns it into a retry, one layer up.

        Named here so the agent-side narrowness is not mistaken for a hole: the
        model still gets a correctable message, it simply is not this package
        that decided so.
        """
        agent = _make_agent(_ExplodingReactAgent(FAST, ROSTER))
        switch = _card_switch(agent)

        with pytest.raises(RetriableError) as caught:
            switch("mistral:mistral-small")

        assert isinstance(caught.value.__cause__, _UnrelatedSwitchError)
        assert agent.state.tool_state.active_model is None


# =============================================================================
# The refusal reaches a caller as a diagnosis, not a bare class name
# =============================================================================


class TestRefusalDiagnostics:
    def test_nothing_is_logged_at_warning_by_the_switch_itself(self, caplog: Any) -> None:
        """A refused switch is the caller's business — it is raised, not logged.

        The restore path is the one that logs (it swallows), and keeping the two
        apart is what lets the restore spec count records.
        """
        agent = _make_agent(_RosterReactAgent(FAST, ROSTER))

        with caplog.at_level(logging.WARNING, logger="akgentic.agent.agent"):
            with pytest.raises(RetriableError):
                agent.switch_model("openai:not-in-the-roster")

        assert caplog.records == []
