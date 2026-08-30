"""Usage-limit handling in ``receiveMsg_AgentMessage``, on the real BaseAgent.

**This file used to be ``test_usage_limit_tier_branch.py``, and there is no tier
branch left to test.** ``akgentic-llm``'s ``LimitRecoveryCapability`` owns
degradation: it decides whether a run-tier breach concludes, drives the
conclusion, and re-raises the ORIGINAL breach when that declines or fails. What
reaches this package is therefore never a choice — it is an error that has
already exhausted its second chance, and the only response left is to tell the
human.

So the two tiers are handled identically here, and the file keeps its place for
the three things the fake-agent suite in ``test_usage_limit_guard.py`` cannot
show:

- the guard is really on the real ``BaseAgent.act``, and pages through the real
  ``notify_human`` / ``config.name`` wiring;
- **no handler carries it** — every test below goes through an undecorated
  ``receiveMsg_AgentMessage`` and still escalates, which is the whole point of
  moving the decorator onto the LLM call;
- a turn ``akgentic-llm`` concluded is **indistinguishable** from an ordinary one
  at this layer — it comes back out through the guard as a ``StructuredOutput``
  and routes through ``_route_output`` with nothing knowing it was a rescue.

Every test plants its outcome on ``_react_agent.run_sync`` rather than on
``act``, because mocking ``act`` away would take the guard with it.

``test_limit_recovery_parity.py`` covers the same ground against a *live*
ReactAgent and a real breach; this file is the fast, mocked view.
"""

import uuid
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import ActorAddress
from akgentic.core.agent import WarningError
from akgentic.llm import AgentUsageLimitError, ReactAgent, RunUsageLimitError, UsageLimitError
from akgentic.tool.errors import CommandNotRecognized

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import Request, StructuredOutput

REQUESTER = "@Human"

# =============================================================================
# HELPERS (same _make_minimal_agent pattern as test_agent_coverage.py)
# =============================================================================


def _make_address(name: str) -> MagicMock:
    """Return a mock ActorAddress that passes isinstance checks."""
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _make_registry(callables: dict[str, Callable] | None = None) -> MagicMock:
    table = callables or {}
    registry = MagicMock()
    registry.has.side_effect = lambda name: name in table

    def _callable(name: str) -> Callable:
        try:
            return table[name]
        except KeyError:
            raise CommandNotRecognized(name) from None

    registry.callable.side_effect = _callable
    return registry


def _make_agent() -> BaseAgent:
    """Construct a BaseAgent without Pykka, with a *spec'd* ReactAgent mock.

    ``spec=ReactAgent`` matters: on a bare MagicMock a misspelt conclusion method
    would silently return a truthy mock, so "no conclusion was attempted" would
    read as satisfied when nothing had been checked at all.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    # The agent's own state, normally assigned in on_start. act() reads
    # state.tool_state.active_model to re-apply a persisted model selection;
    # an empty slot makes that a no-op, which is what these specs want.
    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]

    agent._command_registry = _make_registry()  # type: ignore[attr-defined]
    agent._react_agent = MagicMock(spec=ReactAgent)  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    # The context updater normally built in on_start. These specs are not about
    # context delivery, so a stub that composes nothing keeps act() alive.
    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    agent.get_team_member = MagicMock(return_value=None)  # type: ignore[method-assign]
    agent.notify_human = MagicMock()  # type: ignore[method-assign]

    return agent


def _make_message() -> AgentMessage:
    message = AgentMessage(content="what is the status?", type="request")
    message.sender = _make_address(REQUESTER)
    return message


def _agent_whose_turn(outcome: Exception | StructuredOutput) -> tuple[BaseAgent, MagicMock]:
    """An agent whose LLM call raises or returns ``outcome``, plus the requester.

    ``get_team_member`` is keyed **by name**: it resolves the requester and nobody
    else. A blanket ``return_value`` would hand the requester's address back for
    whatever name the model happened to choose, so "the requester received the
    answer" would hold even for an answer addressed to a third agent.

    **The outcome is planted on ``run_sync``, not on ``act``.** The guard lives on
    ``act()`` now, so replacing ``act`` with a mock would remove the very thing
    under test and every escalation assertion below would fail for the wrong
    reason. Driving the real ``act()`` is also what makes the rescued-turn class
    honest: the conclusion has to travel back out through the guard.
    """
    agent = _make_agent()
    requester = _make_address(REQUESTER)
    agent.get_team_member = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda name: requester if name == REQUESTER else None
    )
    agent.hire_member = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("handling a usage breach must not hire anyone")
    )
    if isinstance(outcome, Exception):
        agent._react_agent.run_sync.side_effect = outcome  # type: ignore[attr-defined]
    else:
        agent._react_agent.run_sync.return_value = outcome  # type: ignore[attr-defined]
    return agent, requester


def _conclusion(
    recipient: str = REQUESTER, message: str = "Here is what I found."
) -> StructuredOutput:
    """A StructuredOutput carrying one deliverable Request."""
    return StructuredOutput(
        messages=[Request(recipient=recipient, message=message, message_type="response")]
    )


# =============================================================================
# Every tier ends the same way
# =============================================================================


class TestEveryTierNotifiesAndStops:
    """Run tier, agent tier, base class: one notification, then WarningError."""

    @pytest.mark.parametrize(
        "error",
        [
            RunUsageLimitError("run request limit"),
            AgentUsageLimitError("lifetime budget spent"),
            UsageLimitError("token limit"),
        ],
        ids=["run-tier", "agent-tier", "base"],
    )
    @patch("akgentic.agent.agent.sleep")
    def test_the_human_is_paged_and_nothing_is_sent(
        self, mock_sleep: MagicMock, error: UsageLimitError
    ) -> None:
        agent, _ = _agent_whose_turn(error)

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        notice = agent.notify_human.call_args[0][0]  # type: ignore[attr-defined]
        assert "@TestAgent" in notice
        assert str(error) in notice
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent._react_agent.context.append_user_prompt.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_no_tier_makes_this_package_attempt_a_conclusion(
        self, mock_sleep: MagicMock
    ) -> None:
        """The retirement, seen from the real handler.

        ``conclude_without_tools_sync`` is ``akgentic-llm``'s to call from inside
        its own recovery. Nothing in this package calls it any more, on any tier.
        """
        for error in (RunUsageLimitError("run"), AgentUsageLimitError("lifetime")):
            agent, _ = _agent_whose_turn(error)

            with pytest.raises(WarningError):
                agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

            agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_the_breach_reported_is_the_one_that_arrived(self, mock_sleep: MagicMock) -> None:
        """This package never rewrites the error, so llm's choice of it stands.

        On a run-tier breach whose conclusion failed, llm surfaces the ORIGINAL
        breach rather than the secondary failure. That guarantee lives in
        ``akgentic-llm``; what is pinned here is that nothing downstream disturbs
        it.
        """
        agent, _ = _agent_whose_turn(RunUsageLimitError("original run breach"))

        with pytest.raises(WarningError) as excinfo:
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        assert "original run breach" in str(excinfo.value)
        assert "original run breach" in agent.notify_human.call_args[0][0]  # type: ignore[attr-defined]


# =============================================================================
# A rescued turn looks like any other turn
# =============================================================================


class TestARescuedTurnIsIndistinguishable:
    """What llm's recovery returns comes back through ``act()`` and just routes."""

    @patch("akgentic.agent.agent.sleep")
    def test_the_concluded_answer_reaches_the_requester_with_nobody_paged(
        self, mock_sleep: MagicMock
    ) -> None:
        """The rescue is invisible here — no branch, no flag, no notification.

        This is the whole reason the tier branch could be retired. ``act()``
        returns the conclusion's ``StructuredOutput``; ``_route_output`` sends it;
        the handler returns normally.
        """
        agent, requester = _agent_whose_turn(
            _conclusion(message="Partial answer: two of three sources checked.")
        )

        agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        agent.get_team_member.assert_called_once_with(REQUESTER)  # type: ignore[attr-defined]
        target, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert target is requester
        assert sent.content == "Partial answer: two of three sources checked."
        assert sent.type == "response"
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_a_conclusion_that_routes_nothing_is_silent(self, mock_sleep: MagicMock) -> None:
        """The known gap, pinned so it is a decision rather than a surprise.

        A ``StructuredOutput`` with no requests is an ordinary success: nothing
        raises, nothing is sent, and no human hears about it. The retired helper
        checked this via a ``delivered`` flag. ``akgentic-llm`` cannot check it —
        it sees the output as ``Any`` — and the decorator never sees the output at
        all, so closing it needs a seam on the capability. Open question ``§Q2`` on
        the degradation-boundary decision (ADR-021); this test records today's
        behaviour so the day it changes is visible in the diff.
        """
        agent, _ = _agent_whose_turn(StructuredOutput(messages=[]))

        agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]


# =============================================================================
# Everything else is somebody else's
# =============================================================================


class TestNonUsageErrorsAreUntouched:
    """The decorator owns usage-limit errors and nothing more."""

    @patch("akgentic.agent.agent.sleep")
    def test_an_unrelated_failure_propagates(self, mock_sleep: MagicMock) -> None:
        agent, _ = _agent_whose_turn(RuntimeError("ReactAgent is closed"))

        with pytest.raises(RuntimeError, match="ReactAgent is closed"):
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]
