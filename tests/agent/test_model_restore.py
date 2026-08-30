"""The restore: a switched model that survives a restart, and never strands a team.

This is the payoff of the whole three-package feature. ``ModelTool`` records the
key it switched to in ``ToolState.active_model``; ``AgentState`` carries that slot
through persistence; and ``BaseAgent`` re-applies it at the top of ``act()``, so a
resumed agent answers on the model it was switched to rather than the one its
config declares.

Four properties, and the order of the first two is the point:

- the re-application runs **before** ``run_sync``, so the turn really is answered
  by the restored model;
- it runs **before** ``_deliver_context_update()``, so the turn's ``LLM_CONTEXT``
  block cannot advertise a model that is not the one answering;
- it is **never fatal** — a key the delegate refuses costs one warning and the
  declared entry answers. A restore that raised over a remembered preference
  would strand the whole team on a restart;
- it happens **once**. A switch is a model rebuild plus a compaction-strategy
  rebuild, deliberately not short-circuited on the already-active key, so
  re-applying per turn would be a per-turn cost and a per-turn failure surface.

AC 8 and AC 9 drive the **real** ``ModelTool`` card rather than assigning the slot
by hand: the production write lives inside the card's own closure, and a spec that
sets ``state.tool_state.active_model`` itself skips the one line that makes the
feature persist anything.
"""

import logging
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from akgentic.llm import ModelSwitchError, model_roster_key
from akgentic.llm.config import ModelConfig
from akgentic.tool.core import ContextUpdater
from akgentic.tool.model import ModelTool
from pydantic_ai.messages import ModelRequest, UserPromptPart

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

AGENT_LOGGER = "akgentic.agent.agent"

FAST = ModelConfig(provider="openai", model="gpt-5-mini", context_length=128_000)
DEEP = ModelConfig(provider="anthropic", model="claude-sonnet-4", context_length=200_000)
CHEAP = ModelConfig(provider="mistral", model="mistral-small", context_length=32_000)

ROSTER = [FAST, DEEP, CHEAP]

FAST_KEY = "openai:gpt-5-mini"
CHEAP_KEY = "mistral:mistral-small"

# =============================================================================
# HELPERS
# =============================================================================


class _RosterReactAgent:
    """A ``ReactAgent`` stand-in with roster semantics and a recording ``run_sync``.

    ``runs`` records the model **in force at the moment the run started**, which
    is how "the switch happened before ``run_sync``" is asserted without reaching
    into ordering machinery.
    """

    def __init__(self, active: ModelConfig, roster: list[ModelConfig]) -> None:
        self._active = active
        self._roster = list(roster)
        self.switch_calls: list[str] = []
        self.runs: list[str] = []
        self.blocks: list[str] = []
        history: list[Any] = []
        self.context = SimpleNamespace(
            messages=history,
            append_user_prompt=self._append_user_prompt,
        )

    def _append_user_prompt(self, entry: str) -> None:
        self.blocks.append(entry)
        self.context.messages.append(ModelRequest(parts=[UserPromptPart(content=entry)]))

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

    def run_sync(self, prompt: Any, deps: Any = None, output_type: Any = None) -> Any:
        self.runs.append(model_roster_key(self._active))
        return output_type()


class _AlwaysRefusingReactAgent(_RosterReactAgent):
    """A delegate whose switch always refuses — the mock's shipped behaviour.

    ``MockReactAgent.switch_model`` raises ``ModelSwitchError`` on every key, by
    construction: it smuggles its scenario path through ``model_cfg.model`` and
    builds no model at all. So a mock-mode agent carrying a persisted key
    exercises the drop-and-log path in production, with no stubbing — if AC 10
    breaks, a mock-mode agent stops starting.
    """

    def switch_model(self, key: str) -> ModelConfig:
        self.switch_calls.append(key)
        raise ModelSwitchError(
            f"cannot switch to '{key}': this agent replays a scenario bound at "
            "construction and builds no model, so switching is unavailable"
        )


def _make_agent(react_agent: _RosterReactAgent) -> BaseAgent:
    """A bare ``BaseAgent`` wired to the **real** ``ModelTool`` card.

    The card contributes its real ``active_model_state`` provider to a real
    ``ContextUpdater``, so the ``LLM_CONTEXT`` block AC 9 reads is the shipped one
    and not a test rendering. The card is held on the agent because it keeps only
    a *weak* reference back — nothing else in this harness would keep it alive.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = react_agent  # type: ignore[attr-defined, assignment]
    agent._actor_ref = MagicMock()  # type: ignore[attr-defined]
    agent._orchestrator = None  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()  # type: ignore[attr-defined]

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]

    card = ModelTool().observer(agent)
    agent._model_card = card  # type: ignore[attr-defined]
    agent._context_updater = ContextUpdater(  # type: ignore[attr-defined]
        agent, card.get_context_states()
    )
    return agent


def _card_switch(agent: BaseAgent) -> Any:
    """The real card's ``switch_model`` closure — the one that writes the slot."""
    card: ModelTool = agent._model_card  # type: ignore[attr-defined]
    return next(tool for tool in card.get_tools() if tool.__name__ == "switch_model")


def _persisted(state: AgentState) -> AgentState:
    """*state* as it comes back off the event store: through a real round trip."""
    return AgentState.model_validate(state.model_dump())


def _turn(agent: BaseAgent, content: str = "hello") -> StructuredOutput:
    return agent.act(AgentMessage(content=content), StructuredOutput)


def _warnings(caplog: Any) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == AGENT_LOGGER
    ]


# =============================================================================
# AC 8 — the persisted selection is re-applied before the first turn
# =============================================================================


class TestTheSelectionSurvivesARestart:
    """The acceptance criterion of the whole three-epic set."""

    def test_a_switch_made_through_the_card_answers_the_next_agent_s_first_turn(self) -> None:
        """Switch on one agent, persist, restore into another, run one turn.

        MUTATION — remove the ``_restore_active_model()`` call from ``act()`` and
        this goes red: the fresh agent answers on its declared model.
        """
        before = _RosterReactAgent(FAST, ROSTER)
        first = _make_agent(before)
        _card_switch(first)(CHEAP_KEY)

        # It is the card that wrote this, not the test.
        persisted = _persisted(first.state)
        assert persisted.tool_state.active_model == CHEAP_KEY

        after = _RosterReactAgent(FAST, ROSTER)
        resumed = _make_agent(after)
        resumed.init_state(persisted)

        _turn(resumed)

        assert model_roster_key(after.active_model()) == CHEAP_KEY
        # The switch preceded the run: the run recorded the model in force when
        # it started, and it is the restored one.
        assert after.runs == [CHEAP_KEY]

    def test_the_declared_model_answers_when_nothing_was_ever_switched(self) -> None:
        """The other half — without it the spec above would pass on a hardcode."""
        react = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react)

        _turn(agent)

        assert react.switch_calls == []
        assert react.runs == [FAST_KEY]


# =============================================================================
# AC 9 — the re-application precedes the turn's context-update block
# =============================================================================


class TestTheContextBlockNamesTheRestoredModel:
    def test_the_first_block_after_a_restore_names_the_restored_key(self) -> None:
        """MUTATION — move ``_restore_active_model()`` **after**
        ``_deliver_context_update()`` and this goes red: the block advertises the
        declared model while the restored one answers.

        The block is composed by the real ``ModelTool`` provider through a real
        ``ContextUpdater``; nothing here renders it.
        """
        before = _RosterReactAgent(FAST, ROSTER)
        first = _make_agent(before)
        _card_switch(first)(CHEAP_KEY)

        after = _RosterReactAgent(FAST, ROSTER)
        resumed = _make_agent(after)
        resumed.init_state(_persisted(first.state))

        _turn(resumed)

        assert len(after.blocks) == 1, "the first turn must deliver exactly one block"
        assert CHEAP_KEY in after.blocks[0]
        assert FAST_KEY not in after.blocks[0]


# =============================================================================
# AC 10 — a stale key is dropped with a log line, and the agent starts
# =============================================================================


class TestAStaleKeyIsDropped:
    def test_a_refused_key_warns_once_and_the_turn_completes(self, caplog: Any) -> None:
        """MUTATION — remove the restore's ``try``/``except`` and this goes red:
        the ``ModelSwitchError`` leaves ``act()`` and the turn never runs.

        The assertions name the stale key **and** an available roster key, so the
        spec cannot pass on some other module's warning or on a bare
        ``assert caplog.records``.
        """
        react = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react)
        stale = AgentState(backstory="You are a test agent.")
        stale.tool_state.active_model = "openai:retired-model"
        agent.init_state(_persisted(stale))

        with caplog.at_level(logging.WARNING, logger=AGENT_LOGGER):
            _turn(agent)

        records = _warnings(caplog)
        assert len(records) == 1
        message = records[0].getMessage()
        assert "openai:retired-model" in message
        assert FAST_KEY in message

        # Nothing raised, nothing re-raised: the declared entry answered.
        assert model_roster_key(react.active_model()) == FAST_KEY
        assert react.runs == [FAST_KEY]

    def test_a_delegate_that_refuses_every_key_still_starts(self, caplog: Any) -> None:
        """The mock's own behaviour, and therefore a free production case.

        Under the load-test flag every switch is refused by construction. An agent
        carrying a persisted key must still take its turn.
        """
        react = _AlwaysRefusingReactAgent(FAST, ROSTER)
        agent = _make_agent(react)
        remembered = AgentState(backstory="You are a test agent.")
        remembered.tool_state.active_model = CHEAP_KEY
        agent.init_state(_persisted(remembered))

        with caplog.at_level(logging.WARNING, logger=AGENT_LOGGER):
            _turn(agent)

        assert react.switch_calls == [CHEAP_KEY]
        assert react.runs == [FAST_KEY]
        assert len(_warnings(caplog)) == 1

    def test_the_available_keys_are_reported_even_with_no_roster_at_all(
        self, caplog: Any
    ) -> None:
        """A single-model agent that somehow carries a key gets a usable diagnosis.

        Joining an empty roster yields an empty string, which would read as a
        truncated sentence; the message says so in words instead.
        """
        react = _RosterReactAgent(FAST, [])
        agent = _make_agent(react)
        remembered = AgentState(backstory="You are a test agent.")
        remembered.tool_state.active_model = CHEAP_KEY
        agent.init_state(_persisted(remembered))

        with caplog.at_level(logging.WARNING, logger=AGENT_LOGGER):
            _turn(agent)

        message = _warnings(caplog)[0].getMessage()
        assert CHEAP_KEY in message
        assert "no roster" in message
        assert react.runs == [FAST_KEY]


# =============================================================================
# AC 11 — an empty slot is a no-op
# =============================================================================


class TestNoPersistedSelection:
    def test_no_switch_is_attempted_and_nothing_is_logged(self, caplog: Any) -> None:
        react = _RosterReactAgent(DEEP, ROSTER)
        agent = _make_agent(react)
        assert agent.state.tool_state.active_model is None

        with caplog.at_level(logging.WARNING, logger=AGENT_LOGGER):
            _turn(agent)

        assert react.switch_calls == []
        assert _warnings(caplog) == []
        assert react.runs == ["anthropic:claude-sonnet-4"]


# =============================================================================
# AC 12 — the re-application is idempotent and does not repeat every turn
# =============================================================================


class TestIdempotence:
    def test_three_turns_re_apply_a_restored_key_exactly_once(self) -> None:
        """MUTATION — drop the ``_restored_model_key`` check and this goes red:
        three turns become three switches, each a model rebuild.
        """
        react = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react)
        remembered = AgentState(backstory="You are a test agent.")
        remembered.tool_state.active_model = CHEAP_KEY
        agent.init_state(_persisted(remembered))

        _turn(agent, "one")
        _turn(agent, "two")
        _turn(agent, "three")

        assert react.switch_calls == [CHEAP_KEY]
        assert react.runs == [CHEAP_KEY, CHEAP_KEY, CHEAP_KEY]

    def test_a_switch_made_this_session_is_not_re_applied_next_turn(self) -> None:
        """The latch is set by the switch itself, not only by the restore.

        The agent wrote that key into the slot a moment ago; re-applying it at the
        top of the next turn would rebuild the model for nothing.
        """
        react = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react)

        _turn(agent, "before the switch")
        _card_switch(agent)(CHEAP_KEY)
        _turn(agent, "after the switch")

        assert react.switch_calls == [CHEAP_KEY]  # the card's own call, and no other
        assert react.runs == [FAST_KEY, CHEAP_KEY]


# =============================================================================
# AC 13 — the slot is read live, never captured
# =============================================================================


class TestTheSlotIsReadLive:
    def test_a_state_replaced_after_the_first_turn_is_the_one_restored_from(self) -> None:
        """MUTATION — cache ``self.state.tool_state`` on first use (or bind it in
        ``on_start``) and read the slot off the field: this goes red.

        Two turns either side of the replacement is the only shape that catches
        it. A single post-restore turn passes just as well against a captured
        carrier, because the capture would happen after the replacement.
        """
        react = _RosterReactAgent(FAST, ROSTER)
        agent = _make_agent(react)

        _turn(agent, "first, with an empty slot")
        assert react.switch_calls == []

        remembered = AgentState(backstory="You are a test agent.")
        remembered.tool_state.active_model = CHEAP_KEY
        agent.init_state(_persisted(remembered))

        _turn(agent, "second, after the slot arrived")

        assert react.switch_calls == [CHEAP_KEY]
        assert react.runs == [FAST_KEY, CHEAP_KEY]
