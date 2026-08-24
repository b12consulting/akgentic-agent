"""Direct coverage of the usage-limit policy in ``usage_limits.py``.

The policy this package applies is now one sentence — notify the team's human and
end the turn — for every tier. Degradation belongs to ``akgentic-llm``: by the
time an error arrives here its ``LimitRecoveryCapability`` has already decided
whether the turn concludes, and a conclusion that succeeded never raises at all.

Everything here is exercised against a fake that satisfies ``AgentLike`` and
nothing else — no BaseAgent, no actor system, no live ReactAgent — because the
whole point of extracting the policy is that a new agent class gets it without
inheriting anything.

**The guard wraps the LLM call, not a message handler.** The fake below therefore
decorates a method shaped like ``BaseAgent.act`` — arguments the policy never
reads, and a return value it must hand back untouched. Three properties are
pinned by mutation rather than by assertion:

- **the single ``except`` is on the base class**, so a tier nobody enumerated is
  still handled (``TestOneClauseCoversEveryTier``);
- **nothing in this package attempts a conclusion any more**
  (``TestTheDecoratorNeverReachesTheLlm``);
- **the wrapper returns the value** — a guard written for a ``None``-returning
  handler swallows every ordinary turn's output while every escalation test
  stays green (``test_the_output_of_an_uneventful_call_is_returned``).
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

from akgentic.agent.usage_limits import escalate_usage_limit, guard_usage_limits

REQUESTER = "@Requester"


class _FutureTierError(LLMUsageLimitError):
    """A usage-limit tier ``usage_limits.py`` has never heard of.

    Nothing in the package refers to this class. It exists so the ``except`` can
    be proved to be on the base rather than on a list of today's subclasses — the
    same reason Golden Rule #12's guard uses a subclass carrying an unknown field.
    """


# =============================================================================
# A second "agent class" that owes the policy nothing but the Protocol
# =============================================================================


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
    """A fake agent whose LLM call carries the decorator.

    ``act`` is shaped like ``BaseAgent.act``: a positional argument, a keyword
    one, and a real return value. None of the three matters to the policy, which
    is exactly what has to be provable — the guard reads no argument and must
    hand the return value straight back.
    """

    def __init__(self, name: str = "@Guarded") -> None:
        super().__init__(name)
        self.turns: list[str] = []
        self.prompts: list[Any] = []
        self.turn_raises: Exception | None = None

    @guard_usage_limits()
    def act(self, message: Message, output_type: type[str] = str, *, note: str = "") -> str:
        """One turn of work, and no error handling of its own."""
        self.turns.append(note)
        self.prompts.append(message)
        if self.turn_raises is not None:
            raise self.turn_raises
        return "the answer"


def _address(name: str) -> MagicMock:
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _message(sender: str | None = REQUESTER) -> Message:
    message = Message()
    if sender is not None:
        message.sender = _address(sender)
    return message


def _breaching(error: Exception) -> _GuardedAgent:
    agent = _GuardedAgent()
    agent.turn_raises = error
    return agent


# =============================================================================
# One clause, every tier
# =============================================================================


class TestOneClauseCoversEveryTier:
    """Both tiers, and any tier added later, end the same way.

    **Verified by mutation.** Replacing the single ``except LLMUsageLimitError``
    with the per-tier ladder this module used to carry — ``except
    RunUsageLimitError`` then ``except AgentUsageLimitError``, both escalating —
    turns ``test_a_tier_this_module_never_heard_of_is_still_caught`` red, along
    with every spec in the suite that raises the base class itself: the ``[base]``
    parametrisation here, its twin in ``test_usage_limit_handling.py``, and the
    three ``LLMUsageLimitError`` specs in ``test_agent_coverage.py`` — six in all.
    Nothing that raises a *subclass* moves, which is the point: the ladder is
    correct on the day it is written and silently incomplete afterwards, and only
    the unenumerated tier exposes it.
    """

    @pytest.mark.parametrize(
        "error",
        [
            RunUsageLimitError("run request limit"),
            AgentUsageLimitError("lifetime budget spent"),
            LLMUsageLimitError("token limit"),
        ],
        ids=["run-tier", "agent-tier", "base"],
    )
    def test_every_tier_notifies_once_and_raises(self, error: LLMUsageLimitError) -> None:
        agent = _breaching(error)

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.act(_message())

        assert len(agent.notified) == 1
        assert str(error) in agent.notified[0]

    def test_a_tier_this_module_never_heard_of_is_still_caught(self) -> None:
        """The clause is on the base, so a future subclass needs no edit here."""
        agent = _breaching(_FutureTierError("a limit invented after this code was written"))

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.act(_message())

        assert len(agent.notified) == 1

    def test_the_tiers_are_no_longer_told_apart_at_all(self) -> None:
        """Identical text, identical outcome — the distinction was spent inside the LLM.

        Its predecessor asserted the opposite (same text, opposite outcomes) and
        was the reason the ladder had to stay ordered. Recording the reversal here
        keeps the next reader from restoring a branch this package cannot act on.
        """
        text = "usage limit exceeded"

        run_tier = _breaching(RunUsageLimitError(text))
        with pytest.raises(WarningError):
            run_tier.act(_message())

        agent_tier = _breaching(AgentUsageLimitError(text))
        with pytest.raises(WarningError):
            agent_tier.act(_message())

        assert run_tier.notified == agent_tier.notified


# =============================================================================
# The conclusion is gone from this package
# =============================================================================


class TestTheDecoratorNeverReachesTheLlm:
    """No breach, of any tier, makes this package call the model.

    **Verified by mutation.** Restoring any call to
    ``_react_agent.conclude_without_tools_sync`` in the ``except`` turns both
    tests below red — and seven more that assert the same thing from the real
    ``BaseAgent`` and ``CustomAgent``, and against a live ReactAgent in
    ``test_limit_recovery_parity.py``. Without them the retirement is invisible: a
    reinstated conclusion would satisfy every escalation assertion above, because
    it also escalates whenever it fails.
    """

    @pytest.mark.parametrize(
        "error",
        [RunUsageLimitError("run request limit"), AgentUsageLimitError("spent")],
        ids=["run-tier", "agent-tier"],
    )
    def test_no_conclusion_is_attempted(self, error: LLMUsageLimitError) -> None:
        agent = _breaching(error)

        with pytest.raises(WarningError):
            agent.act(_message())

        agent._react_agent.conclude_without_tools_sync.assert_not_called()

    def test_the_module_does_not_touch_the_react_agent_at_all(self) -> None:
        """``AgentLike`` no longer declares ``_react_agent``, and nothing reads it.

        The Protocol shrank with the helper it existed for. Asserting on the
        Protocol rather than on a call keeps this true for a policy that grows a
        new branch later.
        """
        from akgentic.agent.usage_limits import AgentLike

        assert "_react_agent" not in AgentLike.__annotations__


# =============================================================================
# What the decorator does not touch
# =============================================================================


class TestGuardLeavesTheCallAlone:
    """The decorator adds the policy and changes nothing else about the method."""

    def test_the_output_of_an_uneventful_call_is_returned(self) -> None:
        """The property a handler-shaped guard did not have to have.

        **Verified by mutation.** Dropping the ``return`` from the wrapper —
        ``method(self, *args, **kwargs)`` on its own, which is exactly what the
        previous handler-shaped guard did — turns this test red, and 34 others
        with it across nine files: every ordinary turn in the package silently
        returns ``None``, so ``_route_output`` and ``_route_triage`` blow up on
        ``NoneType`` and ``compact()`` hands back nothing. Every *breach* test
        stays green, which is the trap this spec exists for — the escalation path
        never returns, so it cannot notice a missing ``return``. This is the one
        spec that fails on the value rather than on a downstream ``AttributeError``,
        and it is the one that says why.
        """
        agent = _GuardedAgent()

        assert agent.act(_message(), note="work") == "the answer"
        assert agent.turns == ["work"]
        assert agent.notified == []

    def test_positional_and_keyword_arguments_reach_the_method(self) -> None:
        """The wrapper forwards everything after ``self`` untouched.

        Both halves are asserted: the message the wrapper passes on positionally
        through ``*args``, and the keyword-only ``note`` through ``**kwargs``.
        Asserting only on ``note`` would leave the positional path untested while
        reading as though it were covered — and the positional path is the one
        that changed, because the guard no longer names ``message`` at all.
        """
        agent = _GuardedAgent()
        message = _message()

        agent.act(message, str, note="kw")

        assert agent.turns == ["kw"]
        assert agent.prompts == [message]

    def test_the_method_keeps_its_identity(self) -> None:
        """``@wraps`` — the guarded method must still look like itself."""
        assert _GuardedAgent.act.__name__ == "act"
        assert _GuardedAgent.act.__doc__ is not None

    def test_the_signature_is_the_methods_own(self) -> None:
        """The wrapper declares only ``(self, /, *args, **kwargs)``.

        Anything that introspects a guarded method — and ``Akgent._receiveMessage``
        does exactly this to decide whether to pass ``sender`` — would see that
        useless signature were it not for ``@wraps`` setting ``__wrapped__``, which
        ``inspect.signature`` follows. No handler is decorated today, so nothing
        depends on it right now; it is pinned because the cost of losing it is a
        failure that appears only in a live actor, and only for whoever decorates
        a handler next.
        """
        parameters = inspect.signature(_GuardedAgent.act).parameters

        assert "message" in parameters
        assert "output_type" in parameters
        assert "note" in parameters

    def test_a_non_usage_error_propagates_untouched(self) -> None:
        """Usage-limit errors are the only ones the guard is for."""
        agent = _breaching(RuntimeError("something else entirely"))

        with pytest.raises(RuntimeError, match="something else entirely"):
            agent.act(_message())

        assert agent.notified == []

    def test_nothing_is_read_off_the_arguments(self) -> None:
        """A sender-less message is ordinary, because no argument is inspected.

        Its predecessor refused to act without a requester to name, because the
        conclusion prompt had to address someone. There is no prompt here now, and
        no argument the policy reads — which is what let the guard move off the
        handler and onto ``act()`` in the first place.
        """
        agent = _breaching(RunUsageLimitError("run request limit"))

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.act(_message(sender=None))

        assert len(agent.notified) == 1


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

    def test_the_breach_reported_is_the_one_that_arrived(self) -> None:
        """Whatever ``akgentic-llm`` re-raised is what the human is told, verbatim.

        On a run-tier breach that is the ORIGINAL breach — the capability's
        conclusion may have failed with something else entirely, and llm surfaces
        the original rather than the secondary. This package simply does not
        rewrite it.
        """
        agent = _breaching(RunUsageLimitError("original run breach"))

        with pytest.raises(WarningError) as excinfo:
            agent.act(_message())

        assert "original run breach" in str(excinfo.value)
        assert "original run breach" in agent.notified[0]
