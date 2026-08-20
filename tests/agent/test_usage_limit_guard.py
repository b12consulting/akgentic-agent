"""Direct coverage of the usage-limit policy in ``usage_limits.py``.

The 18.1 tests reach this code only through ``BaseAgent.receiveMsg_AgentMessage``,
which proves the policy works *for that handler with that schema*. The whole point
of extracting it is that a new agent class gets it without inheriting anything, so
everything here is exercised against a fake that satisfies ``AgentLike`` and
nothing else — no BaseAgent, no actor system, no ReactAgent.

The clause ordering inside ``guard_usage_limits`` is the load-bearing part.
``RunUsageLimitError`` and ``AgentUsageLimitError`` both subclass
``UsageLimitError``, so a base clause placed first catches both and the tier
branch never runs — silently, since a wrong order still compiles and an
agent-tier test still passes. ``TestClauseOrderIsLoadBearing`` below is the test
that goes red for it; its docstring records the mutation that was run.
"""

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core import ActorAddress, BaseConfig
from akgentic.core.agent import WarningError
from akgentic.core.messages import Message
from akgentic.llm import AgentUsageLimitError, ReactAgent, RunUsageLimitError
from akgentic.llm import UsageLimitError as LLMUsageLimitError
from pydantic import BaseModel

from akgentic.agent.usage_limits import (
    escalate_usage_limit,
    guard_usage_limits,
    try_conclude_without_tools,
)

REQUESTER = "@Requester"


# =============================================================================
# A second "agent class" that owes the policy nothing but the Protocol
# =============================================================================


class Answer(BaseModel):
    """A structured output that is deliberately NOT StructuredOutput.

    ``try_conclude_without_tools`` must never inspect the schema — it cannot, since
    the schema belongs to the caller. Using a type with no ``.messages`` at all is
    what makes that provable rather than asserted.
    """

    text: str = ""


class _FakeAgent:
    """The smallest thing that satisfies ``AgentLike``."""

    def __init__(self, name: str = "@Guarded") -> None:
        self._react_agent = MagicMock(spec=ReactAgent)
        self._config = BaseConfig(name=name, role="Guarded")
        self.notified: list[str] = []

    @property
    def config(self) -> BaseConfig:
        """Read-only, exactly as the Protocol declares it."""
        return self._config

    def notify_human(self, message: str) -> None:
        self.notified.append(message)


class _GuardedAgent(_FakeAgent):
    """A fake agent whose one handler carries the decorator.

    Nothing here overrides any policy: the schema and the router are decorator
    arguments, so this class supplies only its own work.
    """

    def __init__(self, name: str = "@Guarded") -> None:
        super().__init__(name)
        self.turns: list[str] = []
        self.senders: list[Any] = []
        self.routed: list[Answer] = []
        self.turn_raises: Exception | None = None
        self.delivers = True

    def _route_answer(self, output: Answer) -> bool:
        """Deliver an Answer; report whether anything actually went out."""
        self.routed.append(output)
        return self.delivers

    @guard_usage_limits(output_type=Answer, route=_route_answer)
    def receiveMsg_Ask(  # noqa: N802
        self, message: Message, sender: ActorAddress | None = None, *, note: str = ""
    ) -> None:
        """One turn of work, and no error handling of its own."""
        self.turns.append(note)
        self.senders.append(sender)
        if self.turn_raises is not None:
            raise self.turn_raises


def _address(name: str) -> MagicMock:
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _message(sender: str | None = REQUESTER) -> Message:
    message = Message()
    if sender is not None:
        message.sender = _address(sender)
    return message


def _breaching(error: Exception, *, delivers: bool = True) -> _GuardedAgent:
    agent = _GuardedAgent()
    agent.turn_raises = error
    agent.delivers = delivers
    agent._react_agent.conclude_without_tools_sync.return_value = Answer(text="partial")
    return agent


# =============================================================================
# guard_usage_limits — the clause order
# =============================================================================


class TestClauseOrderIsLoadBearing:
    """The one property a hand-copied ladder loses silently.

    **Verified by mutation.** Moving the ``LLMUsageLimitError`` clause to the front
    of the ladder in ``guard_usage_limits`` turns
    ``test_a_run_tier_breach_is_concluded_before_anyone_is_paged`` red — the base
    clause swallows the run tier and the conclusion is never attempted. The
    agent-tier and backstop tests below stay green under that mutation, which is
    exactly why the run-tier case has to be asserted here rather than assumed from
    them.
    """

    def test_a_run_tier_breach_is_concluded_before_anyone_is_paged(self) -> None:
        agent = _breaching(RunUsageLimitError("run request limit"))

        agent.receiveMsg_Ask(_message())

        agent._react_agent.conclude_without_tools_sync.assert_called_once()
        assert [answer.text for answer in agent.routed] == ["partial"]
        assert agent.notified == []

    def test_an_agent_tier_breach_escalates_with_no_attempt(self) -> None:
        agent = _breaching(AgentUsageLimitError("lifetime budget spent"))

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_Ask(_message())

        agent._react_agent.conclude_without_tools_sync.assert_not_called()
        assert agent.routed == []
        assert len(agent.notified) == 1

    def test_the_base_class_is_caught_by_the_backstop_clause(self) -> None:
        """An akgentic-llm raising the base directly must still be handled."""
        agent = _breaching(LLMUsageLimitError("token limit"))

        with pytest.raises(WarningError, match="token limit"):
            agent.receiveMsg_Ask(_message())

        agent._react_agent.conclude_without_tools_sync.assert_not_called()
        assert len(agent.notified) == 1

    def test_the_two_tiers_are_told_apart_by_class_never_by_text(self) -> None:
        """Identical message text, opposite outcomes."""
        text = "usage limit exceeded"

        run_tier = _breaching(RunUsageLimitError(text))
        run_tier.receiveMsg_Ask(_message())
        assert run_tier.notified == []

        agent_tier = _breaching(AgentUsageLimitError(text))
        with pytest.raises(WarningError):
            agent_tier.receiveMsg_Ask(_message())
        assert len(agent_tier.notified) == 1


# =============================================================================
# guard_usage_limits — what it does and does not touch
# =============================================================================


class TestGuardLeavesTheHandlerAlone:
    """The decorator adds the policy and changes nothing else about the handler."""

    def test_an_uneventful_turn_passes_straight_through(self) -> None:
        agent = _GuardedAgent()

        agent.receiveMsg_Ask(_message(), _address("@Someone"), note="work")

        assert agent.turns == ["work"]
        agent._react_agent.conclude_without_tools_sync.assert_not_called()
        assert agent.notified == []

    def test_positional_and_keyword_arguments_reach_the_handler(self) -> None:
        """The wrapper forwards everything after the message untouched.

        Both halves are asserted: the ``sender`` the wrapper passes on positionally
        through ``*args``, and the keyword-only ``note`` through ``**kwargs``. The
        handler has to record the sender for that first half to be provable —
        asserting only on ``note`` would leave the positional path untested while
        reading as though it were covered.
        """
        agent = _GuardedAgent()
        sender = _address("@Someone")

        agent.receiveMsg_Ask(_message(), sender, note="kw")

        assert agent.turns == ["kw"]
        assert agent.senders == [sender]

    def test_the_handler_keeps_its_identity(self) -> None:
        """``@wraps`` — dispatch finds handlers by name, so the name must survive."""
        assert _GuardedAgent.receiveMsg_Ask.__name__ == "receiveMsg_Ask"
        assert _GuardedAgent.receiveMsg_Ask.__doc__ is not None

    def test_the_signature_actor_dispatch_reads_is_the_handlers_own(self) -> None:
        """``sender`` must stay visible through the wrapper, or it stops arriving.

        ``Akgent._receiveMessage`` decides how to call a handler by inspecting its
        signature: ``"sender" in inspect.signature(method).parameters`` selects
        ``method(self, message, sender)`` over ``method(self, message)``. The
        wrapper declares ``(self, message, /, *args, **kwargs)`` and has no
        ``sender`` of its own — dispatch keeps working only because ``@wraps`` sets
        ``__wrapped__`` and ``inspect.signature`` follows it.

        Swap ``@wraps`` for a hand-assigned ``__name__`` and every other test in
        this file stays green while every decorated handler in the package loses
        its sender argument the first time a live actor delivers a message. Same
        shape of failure as the clause order, so it is pinned the same way.
        """
        parameters = inspect.signature(_GuardedAgent.receiveMsg_Ask).parameters

        assert "message" in parameters
        assert "sender" in parameters

    def test_a_non_usage_error_propagates_untouched(self) -> None:
        """Usage-limit errors are the only ones the guard is for."""
        agent = _breaching(RuntimeError("something else entirely"))

        with pytest.raises(RuntimeError, match="something else entirely"):
            agent.receiveMsg_Ask(_message())

        agent._react_agent.conclude_without_tools_sync.assert_not_called()
        assert agent.notified == []

    def test_a_conclusion_that_routes_nothing_escalates(self) -> None:
        """The router's answer, not the schema, decides whether anything went out."""
        agent = _breaching(RunUsageLimitError("original run breach"), delivers=False)

        with pytest.raises(WarningError, match="original run breach"):
            agent.receiveMsg_Ask(_message())

        assert len(agent.routed) == 1
        assert len(agent.notified) == 1

    def test_the_requester_is_read_off_the_handlers_own_message(self) -> None:
        """The guard lifts it from the message argument — handlers thread nothing."""
        agent = _breaching(RunUsageLimitError("run request limit"))

        agent.receiveMsg_Ask(_message("@Ada"))

        reason = agent._react_agent.conclude_without_tools_sync.call_args[0][0]
        assert "@Ada" in reason


# =============================================================================
# try_conclude_without_tools — against a schema that is not StructuredOutput
# =============================================================================


class TestConcludeWithoutToolsOnACustomSchema:
    """The claim the extraction rests on: the policy never knows the schema."""

    def _route(self, delivered: bool) -> Any:
        route = MagicMock(return_value=delivered)
        return route

    def test_it_asks_for_the_output_type_it_was_given(self) -> None:
        agent = _FakeAgent()
        agent._react_agent.conclude_without_tools_sync.return_value = Answer(text="ok")
        route = self._route(True)

        # A delivered conclusion ends the turn quietly: the requester has their
        # answer, so there is nothing left to escalate.
        try_conclude_without_tools(
            agent,
            RunUsageLimitError("run request limit"),
            REQUESTER,
            output_type=Answer,
            route=route,
        )

        call = agent._react_agent.conclude_without_tools_sync.call_args
        assert call.kwargs["output_type"] is Answer
        assert call.kwargs["deps"] is agent
        assert REQUESTER in call[0][0]
        route.assert_called_once()
        assert route.call_args[0][0] is agent
        assert isinstance(route.call_args[0][1], Answer)

    def test_a_route_reporting_nothing_delivered_is_a_failure(self) -> None:
        agent = _FakeAgent()
        agent._react_agent.conclude_without_tools_sync.return_value = Answer()

        with pytest.raises(WarningError):
            try_conclude_without_tools(
                agent,
                RunUsageLimitError("run request limit"),
                REQUESTER,
                output_type=Answer,
                route=self._route(False),
            )

        assert agent.notified, "a breach nobody could answer must still page the human"

    def test_no_requester_means_no_attempt_at_all(self) -> None:
        """A placeholder would be echoed back as a recipient, and hired as a role."""
        agent = _FakeAgent()
        route = self._route(True)

        with pytest.raises(WarningError):
            try_conclude_without_tools(
                agent,
                RunUsageLimitError("run request limit"),
                None,
                output_type=Answer,
                route=route,
            )

        agent._react_agent.conclude_without_tools_sync.assert_not_called()
        route.assert_not_called()

    def test_an_attempt_that_raises_is_a_failure_whatever_it_raised(self) -> None:
        agent = _FakeAgent()
        agent._react_agent.conclude_without_tools_sync.side_effect = RuntimeError("closed")

        with pytest.raises(WarningError) as caught:
            try_conclude_without_tools(
                agent,
                RunUsageLimitError("run request limit"),
                REQUESTER,
                output_type=Answer,
                route=self._route(True),
            )

        # The ORIGINAL breach is reported, never the secondary failure: "closed"
        # would send whoever reads the log chasing the wrong thing.
        assert "run request limit" in str(caught.value)
        assert "closed" not in str(caught.value)

    def test_a_route_that_raises_is_a_failure_too(self) -> None:
        agent = _FakeAgent()
        agent._react_agent.conclude_without_tools_sync.return_value = Answer(text="ok")
        route = MagicMock(side_effect=RuntimeError("delivery blew up"))

        with pytest.raises(WarningError):
            try_conclude_without_tools(
                agent,
                RunUsageLimitError("run request limit"),
                REQUESTER,
                output_type=Answer,
                route=route,
            )

    def test_the_reason_forbids_further_tools_and_promises_no_follow_up(self) -> None:
        """The prompt is the whole mechanism — there is no tool-free flag to set."""
        agent = _FakeAgent()
        agent._react_agent.conclude_without_tools_sync.return_value = Answer(text="ok")

        try_conclude_without_tools(
            agent,
            RunUsageLimitError("run request limit"),
            REQUESTER,
            output_type=Answer,
            route=self._route(True),
        )

        reason = agent._react_agent.conclude_without_tools_sync.call_args[0][0]
        assert "cannot" in reason and "tool" in reason
        assert "do not promise follow-up work" in reason


# =============================================================================
# escalate_usage_limit
# =============================================================================


class TestEscalateUsageLimit:
    """Page the human, then end the turn — in that order."""

    def test_it_names_the_agent_and_the_breach_then_raises(self) -> None:
        agent = _FakeAgent(name="@Breacher")

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            escalate_usage_limit(agent, LLMUsageLimitError("token limit"))

        assert len(agent.notified) == 1
        assert "@Breacher" in agent.notified[0]
        assert "token limit" in agent.notified[0]

    def test_the_original_breach_is_what_the_run_tier_reports(self) -> None:
        """A failed conclusion must not overwrite the breach that started the turn."""
        agent = _breaching(RunUsageLimitError("original run breach"))
        agent._react_agent.conclude_without_tools_sync.side_effect = AgentUsageLimitError(
            "lifetime budget spent"
        )

        with pytest.raises(WarningError) as excinfo:
            agent.receiveMsg_Ask(_message())

        assert "original run breach" in str(excinfo.value)
        assert "lifetime budget spent" not in str(excinfo.value)
        assert "original run breach" in agent.notified[0]
