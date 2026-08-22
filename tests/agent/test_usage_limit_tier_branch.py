"""Tier-branch behaviour of receiveMsg_AgentMessage's usage-limit handling.

A run-tier breach means the *turn* ran out of budget while the agent may still
have lifetime budget: the agent is asked to conclude without tools and the
requester gets an answer. An agent-tier breach is terminal: a human is paged.
The base class stays a backstop.

The policy itself now lives in the ``guard_usage_limits`` decorator rather than
inside the handler, so every test here drives the **decorated handler** and makes
the turn breach by raising out of ``act()`` — the one call the handler makes that
can reach the LLM. The guard is covered directly, on its own, in
``test_usage_limit_guard.py``; this file stays the BaseAgent-shaped view of the
same policy.

Every test here asserts the **outcome** — what the requester received, whether a
human was notified, whether WarningError escaped — not merely that a mock was
called. The tiers are told apart by exception class; no test reads message text
to decide which tier fired. The three base-class tests in test_agent_coverage.py
are the backstop-clause coverage and stay unmodified.
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
from akgentic.agent.config import AgentConfig
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
    would silently return a truthy mock whose ``.messages`` is also a mock —
    neither empty nor a real list — and the empty-conclusion case would look
    satisfied when nothing was checked at all.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    agent._command_registry = _make_registry()  # type: ignore[attr-defined]
    agent._react_agent = MagicMock(spec=ReactAgent)  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    # Context-state delivery attributes normally set by on_start (Epic 19).
    agent._context_state_providers = []  # type: ignore[attr-defined]
    agent._context_baselines = {}  # type: ignore[attr-defined]
    agent._context_update_seq = 0  # type: ignore[attr-defined]

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


def _breaching_agent(error: Exception) -> tuple[BaseAgent, MagicMock]:
    """An agent whose turn raises ``error``, plus the requester's address.

    ``get_team_member`` is keyed **by name**: it resolves the requester and nobody
    else. A blanket ``return_value`` would hand the requester's address back for
    whatever name the model happened to choose, so "the requester received the
    conclusion" would hold even for an answer addressed to a third agent — the
    exact failure this suite exists to catch. ``hire_member`` is stubbed to fail
    loudly, because a conclusion should never hire anyone.

    The breach is raised out of ``act()`` — the handler's own LLM call — so the
    decorated handler is driven end to end rather than through a seam.
    """
    agent = _make_agent()
    requester = _make_address(REQUESTER)
    agent.get_team_member = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda name: requester if name == REQUESTER else None
    )
    agent.hire_member = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("a tool-free conclusion must not hire anyone")
    )
    agent.act = MagicMock(side_effect=error)  # type: ignore[method-assign]
    return agent, requester


def _conclusion(
    recipient: str = REQUESTER, message: str = "Here is what I found."
) -> StructuredOutput:
    """A StructuredOutput carrying one deliverable Request."""
    return StructuredOutput(
        messages=[Request(recipient=recipient, message=message, message_type="response")]
    )


# =============================================================================
# RUN TIER — the turn is out of budget, the agent is not
# =============================================================================


class TestRunTierConcludes:
    """A run-tier breach answers the requester instead of paging a human."""

    @patch("akgentic.agent.agent.sleep")
    def test_requester_receives_the_conclusion_and_no_human_is_paged(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC2/AC8: the outcome is a delivered reply — not merely a call that happened."""
        agent, requester = _breaching_agent(RunUsageLimitError("run request limit"))
        agent._react_agent.conclude_without_tools_sync.return_value = _conclusion(  # type: ignore[attr-defined]
            message="Partial answer: two of three sources checked."
        )

        agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        # The requester got a real AgentMessage at their own address — and the
        # address was reached by looking THEM up, not whoever the model named.
        agent.get_team_member.assert_called_once_with(REQUESTER)  # type: ignore[attr-defined]
        agent.send.assert_called_once()  # type: ignore[attr-defined]
        target, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert target is requester
        assert isinstance(sent, AgentMessage)
        assert sent.content == "Partial answer: two of three sources checked."
        assert sent.type == "response"

        # No human was involved, and nothing escaped the handler.
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
        agent._react_agent.conclude_without_tools_sync.assert_called_once()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_a_successful_conclusion_writes_nothing_to_its_own_context(
        self, mock_sleep: MagicMock
    ) -> None:
        """The early-conclusion operator-action record is gone, on purpose.

        Story 18.1 appended a synthetic entry saying the turn had concluded early.
        The reason handed to the conclusion already carries that fact, and the
        conclusion's own exchange lands in the context like any other turn, so the
        entry restated what the history already said. ``_record_operator_action``
        is now reached only from ``_dispatch_command`` — a human's slash command,
        which is a genuinely out-of-band event.
        """
        agent, _ = _breaching_agent(RunUsageLimitError("run request limit"))
        agent._react_agent.conclude_without_tools_sync.return_value = _conclusion()  # type: ignore[attr-defined]

        agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        agent._react_agent.context.record_operator_action.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_the_reason_prompt_names_the_requester(self, mock_sleep: MagicMock) -> None:
        """AC2: the recipient is model-chosen, so the prompt must name who to answer."""
        agent, _ = _breaching_agent(RunUsageLimitError("run request limit"))
        agent._react_agent.conclude_without_tools_sync.return_value = _conclusion()  # type: ignore[attr-defined]

        agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        call = agent._react_agent.conclude_without_tools_sync.call_args  # type: ignore[attr-defined]
        reason = call[0][0]
        assert REQUESTER in reason
        assert call.kwargs["deps"] is agent
        assert call.kwargs["output_type"] is StructuredOutput

    @patch("akgentic.agent.agent.sleep")
    def test_attempts_the_conclusion_exactly_once(self, mock_sleep: MagicMock) -> None:
        """AC7: one attempt, no retry loop and no counter."""
        agent, _ = _breaching_agent(RunUsageLimitError("run request limit"))
        agent._react_agent.conclude_without_tools_sync.return_value = _conclusion()  # type: ignore[attr-defined]

        agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        assert agent._react_agent.conclude_without_tools_sync.call_count == 1  # type: ignore[attr-defined]


# =============================================================================
# RUN TIER — every way the attempt can fail falls through to escalation
# =============================================================================


class TestRunTierFallsThrough:
    """A failed attempt is today's behaviour, reporting the original breach."""

    @patch("akgentic.agent.agent.sleep")
    def test_empty_conclusion_delivers_nothing_so_it_escalates(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC4: a successful call that delivers no Request is a failure."""
        agent, _ = _breaching_agent(RunUsageLimitError("run request limit"))
        agent._react_agent.conclude_without_tools_sync.return_value = StructuredOutput(  # type: ignore[attr-defined]
            messages=[]
        )

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent._react_agent.context.record_operator_action.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_agent_tier_raised_by_the_conclusion_reports_the_original_error(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC5: the human hears about the breach that started this, not the secondary one."""
        agent, _ = _breaching_agent(RunUsageLimitError("original run breach"))
        agent._react_agent.conclude_without_tools_sync.side_effect = AgentUsageLimitError(  # type: ignore[attr-defined]
            "lifetime budget spent"
        )

        with pytest.raises(WarningError) as excinfo:
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        assert "original run breach" in str(excinfo.value)
        assert "lifetime budget spent" not in str(excinfo.value)
        assert "original run breach" in agent.notify_human.call_args[0][0]  # type: ignore[attr-defined]
        agent._react_agent.context.record_operator_action.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_a_second_run_tier_breach_does_not_recurse(self, mock_sleep: MagicMock) -> None:
        """AC5/AC7: the conclusion is attempted once, then escalated — never retried."""
        agent, _ = _breaching_agent(RunUsageLimitError("original run breach"))
        agent._react_agent.conclude_without_tools_sync.side_effect = RunUsageLimitError(  # type: ignore[attr-defined]
            "second run breach"
        )

        with pytest.raises(WarningError) as excinfo:
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        assert agent._react_agent.conclude_without_tools_sync.call_count == 1  # type: ignore[attr-defined]
        assert "original run breach" in str(excinfo.value)
        agent.send.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_a_message_with_no_sender_is_not_concluded_to_at_all(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC5: ``Message.sender`` is optional, and "unknown" is not a recipient.

        Naming a placeholder in the reason would get it echoed back as the Request's
        recipient, and a recipient without a leading ``@`` is a *role to hire* — so
        the teardown of a breached turn would hire a member called "unknown". With no
        requester there is nobody to answer, so the attempt is skipped entirely.
        """
        agent, _ = _breaching_agent(RunUsageLimitError("original run breach"))
        senderless = AgentMessage(content="what is the status?", type="request")
        assert senderless.sender is None

        with pytest.raises(WarningError) as excinfo:
            agent.receiveMsg_AgentMessage(senderless, _make_address(REQUESTER))

        assert "original run breach" in str(excinfo.value)
        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent.hire_member.assert_not_called()  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent._react_agent.context.record_operator_action.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_an_unexpected_failure_of_the_attempt_still_escalates(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC5: 'any other exception' means any — a broken conclusion is not a broken turn."""
        agent, _ = _breaching_agent(RunUsageLimitError("original run breach"))
        agent._react_agent.conclude_without_tools_sync.side_effect = RuntimeError(  # type: ignore[attr-defined]
            "ReactAgent is closed"
        )

        with pytest.raises(WarningError) as excinfo:
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        assert "original run breach" in str(excinfo.value)
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent._react_agent.context.record_operator_action.assert_not_called()  # type: ignore[attr-defined]


# =============================================================================
# AGENT TIER + BACKSTOP — unchanged behaviour
# =============================================================================


class TestTerminalTiers:
    """The agent tier is terminal, and the base class remains handled."""

    @patch("akgentic.agent.agent.sleep")
    def test_agent_tier_escalates_without_attempting_a_conclusion(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC6: no attempt, no record — the budget that would pay for it is spent."""
        agent, _ = _breaching_agent(AgentUsageLimitError("lifetime budget spent"))

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent._react_agent.context.record_operator_action.assert_not_called()  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        notice = agent.notify_human.call_args[0][0]  # type: ignore[attr-defined]
        assert "@TestAgent" in notice
        assert "lifetime budget spent" in notice

    @patch("akgentic.agent.agent.sleep")
    def test_base_class_is_still_caught_by_the_backstop_clause(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC1: an akgentic-llm raising the base directly must still be handled."""
        agent, _ = _breaching_agent(UsageLimitError("token limit"))

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))

        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_the_two_tiers_are_told_apart_by_class_not_by_text(
        self, mock_sleep: MagicMock
    ) -> None:
        """AC1: identical message text, opposite outcomes."""
        text = "usage limit exceeded"

        run_tier, _ = _breaching_agent(RunUsageLimitError(text))
        run_tier._react_agent.conclude_without_tools_sync.return_value = _conclusion()  # type: ignore[attr-defined]
        run_tier.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))
        run_tier.notify_human.assert_not_called()  # type: ignore[attr-defined]

        agent_tier, _ = _breaching_agent(AgentUsageLimitError(text))
        with pytest.raises(WarningError):
            agent_tier.receiveMsg_AgentMessage(_make_message(), _make_address(REQUESTER))
        agent_tier.notify_human.assert_called_once()  # type: ignore[attr-defined]
