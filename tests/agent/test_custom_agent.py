"""``CustomAgent`` — the claim the whole extraction rests on, asserted.

A second agent class, with its own structured output and its own message type,
must get the usage-limit policy **without overriding anything, and without
declaring anything**. It now declares even less than when this file was written:
the decorator asks for no schema and no router, because concluding a breached
turn is ``akgentic-llm``'s and it reuses the ``output_type`` the handler already
passes to ``act()``.

So a rescued turn reaches ``_route_triage`` as an ordinary ``TriageOutput``, and
a breach that escapes the LLM pages the human — the same two outcomes
``BaseAgent`` gets, from the same one decorator.

Nothing here touches ``BaseAgent``'s own handler or ``StructuredOutput``. If the
policy had stayed a clause ladder inside ``receiveMsg_AgentMessage``, every test
in this file would have to be written by copying that ladder into ``CustomAgent``
first — which is precisely the copying this story removed.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from akgentic.core import ActorAddress
from akgentic.core.agent import WarningError
from akgentic.llm import AgentUsageLimitError, ReactAgent, RunUsageLimitError
from pydantic import BaseModel

from akgentic.agent.agent import RunInterruptedError
from akgentic.agent.config import AgentConfig
from akgentic.agent.custom_agent import CustomAgent, Handoff, TriageMessage, TriageOutput
from akgentic.agent.messages import AgentMessage

REQUESTER = "@Ops"


def _address(name: str) -> MagicMock:
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _make_custom_agent() -> CustomAgent:
    """Build a CustomAgent outside Pykka, with a spec'd ReactAgent mock.

    ``spec=ReactAgent`` matters: on a bare MagicMock a misspelt conclusion method
    would return a truthy mock and the run-tier path would look exercised when
    nothing had been called.
    """
    agent: CustomAgent = object.__new__(CustomAgent)

    agent._react_agent = MagicMock(spec=ReactAgent)  # type: ignore[attr-defined]
    agent._command_registry = MagicMock()  # type: ignore[attr-defined]
    # Needed only by the specs that drive the REAL act(): a bare MagicMock
    # answers has("_expand_media_refs") truthily, sending act() into media
    # expansion over a MagicMock; and act()'s first statement resets the
    # mailbox capability's run-local tracking.
    agent._command_registry.has.return_value = False  # type: ignore[attr-defined]
    agent._mailbox_capability = MagicMock()  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    # The context updater normally built in on_start. These specs are not about
    # context delivery, so a stub that composes nothing keeps act() alive.
    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    config = MagicMock(spec=AgentConfig)
    config.name = "@Triage"
    agent.config = config  # type: ignore[attr-defined]

    requester = _address(REQUESTER)
    agent.send = MagicMock()  # type: ignore[method-assign]
    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.get_team_member = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda name: requester if name == REQUESTER else None
    )
    agent.hire_member = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("a tool-free conclusion must not hire anyone")
    )
    agent.notify_human = MagicMock()  # type: ignore[method-assign]
    return agent


def _incident(sender: str | None = REQUESTER) -> TriageMessage:
    message = TriageMessage(incident="disk full on node 3", reported_by="monitoring")
    if sender is not None:
        message.sender = _address(sender)
    return message


def _triage(recipient: str = REQUESTER, task: str = "Free 20GB on node 3") -> TriageOutput:
    return TriageOutput(
        severity="high",
        summary="node 3 out of disk",
        handoffs=[Handoff(recipient=recipient, task=task)],
    )


class TestCustomAgentNormalTurn:
    """The uneventful path, so the breached one is a contrast and not the only case."""

    def test_a_turn_reasons_against_this_agents_own_schema(self) -> None:
        agent = _make_custom_agent()
        agent.act = MagicMock(return_value=_triage())  # type: ignore[method-assign]

        agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        # act() receives the MESSAGE now; the framing is the message's own.
        message, output_type = agent.act.call_args[0]  # type: ignore[attr-defined]
        assert output_type is TriageOutput
        assert isinstance(message, TriageMessage)
        assert "disk full on node 3" in message.render_for_llm()
        assert "monitoring" in message.render_for_llm()

        agent.send.assert_called_once()  # type: ignore[attr-defined]
        _, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert isinstance(sent, AgentMessage)
        assert sent.content == "Free 20GB on node 3"

    def test_a_handoff_to_a_name_the_team_does_not_have_is_skipped(self) -> None:
        """An ``@name`` matching nobody costs a delivery, not an exception.

        The rule ``_route_output`` applies, applied by ``_route_triage``. The bool
        it returns is now read by nobody — the guard that consumed it is retired —
        but the skip itself is the behaviour under test.
        """
        agent = _make_custom_agent()
        agent.act = MagicMock(return_value=_triage(recipient="@Ghost"))  # type: ignore[method-assign]

        agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent.get_team_member.assert_called_once_with("@Ghost")  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent.hire_member.assert_not_called()  # type: ignore[attr-defined]


class TestCustomAgentUsageBreach:
    """AC-5, restated: a breach that reaches this subclass pages the human.

    Concluding is ``akgentic-llm``'s, and it needs nothing declared here — it
    reuses the ``output_type`` the handler already asked ``act()`` for, so a
    rescued turn arrives back as an ordinary ``TriageOutput``. The class the
    subclass used to need was the *decorator's* schema argument, and that is gone.
    """

    def test_a_rescued_turn_arrives_as_an_ordinary_triage_output(self) -> None:
        """No subclass declaration, no branch: the conclusion just routes.

        Standing in for what ``akgentic-llm`` returns after it degrades a run-tier
        breach — the handler cannot tell this from a turn that never breached, and
        that indistinguishability is what let the decorator's schema argument go.
        """
        agent = _make_custom_agent()
        agent.act = MagicMock(  # type: ignore[method-assign]
            return_value=_triage(
                task="Partial triage: node 3 is out of disk, cause not yet identified."
            )
        )

        agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent.get_team_member.assert_called_once_with(REQUESTER)  # type: ignore[attr-defined]
        target, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert target.name == REQUESTER
        assert isinstance(sent, AgentMessage)
        assert sent.content.startswith("Partial triage:")
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "error",
        [RunUsageLimitError("run request limit"), AgentUsageLimitError("lifetime budget spent")],
        ids=["run-tier", "agent-tier"],
    )
    def test_a_breach_that_escapes_the_llm_pages_the_human(
        self, error: Exception
    ) -> None:
        """Both tiers, one outcome — and no conclusion attempted from this package.

        The breach is planted on ``run_sync``, not on ``act``: the guard lives on
        ``act()`` now, so mocking ``act`` away would remove the thing under test.
        """
        agent = _make_custom_agent()
        agent._react_agent.run_sync.side_effect = error  # type: ignore[attr-defined]

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]

    def test_a_triage_with_no_handoffs_is_silent(self) -> None:
        """The gap the retired helper used to close, pinned as today's behaviour.

        ``TriageOutput`` has no ``.messages``, so ``akgentic-llm`` could not judge
        this even if it wanted to, and the decorator never sees the output. A
        rescued turn that hands off to nobody therefore ends quietly. Open question
        ``§Q2`` on the degradation-boundary decision (ADR-021); recorded here so the
        day it changes shows up in the diff.
        """
        agent = _make_custom_agent()
        agent.act = MagicMock(  # type: ignore[method-assign]
            return_value=TriageOutput(severity="low", summary="nothing conclusive")
        )

        agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]

    def test_an_incident_with_no_sender_is_handled_like_any_other(self) -> None:
        """Nothing is read off the message any more, so a sender-less one is ordinary."""
        agent = _make_custom_agent()
        agent._react_agent.run_sync.side_effect = RunUsageLimitError(  # type: ignore[attr-defined]
            "original run breach"
        )
        senderless = _incident(sender=None)
        assert senderless.sender is None

        with pytest.raises(WarningError, match="original run breach"):
            agent.receiveMsg_TriageMessage(senderless, _address(REQUESTER))

        agent.hire_member.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]


class TestCustomAgentRunInterruption:
    """Epic 20 invariant 2 holds in the exemplar: no escape into the failure path.

    The mailbox capability is built unconditionally by ``_build_react_agent``,
    so every subclass run is interruptible — and the subclass supplies
    **nothing** for it: ``act()`` absorbs the ``RunInterruptedError``, notifies
    the human once and returns a default ``TriageOutput``.

    Both specs drive the **real** ``act()``, with ``run_sync`` raising as the
    mailbox capability would. Mocking ``act`` away would test the handler's own
    catch, and there is none left to test.
    """

    def test_an_interrupted_run_notifies_and_routes_nothing(self) -> None:
        agent = _make_custom_agent()
        agent._react_agent.run_sync.side_effect = RunInterruptedError("cancelled")  # type: ignore[attr-defined]

        agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent.notify_human.assert_called_once_with("Run interrupted.")  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]

    def test_act_hands_the_handler_a_default_triage_output(self) -> None:
        """The caller is not told anything went wrong — it gets an empty triage.

        ``TriageOutput`` constructs with no arguments because every field has a
        default, which is exactly the condition ``act()`` relies on.
        """
        agent = _make_custom_agent()
        agent._react_agent.run_sync.side_effect = RunInterruptedError("cancelled")  # type: ignore[attr-defined]

        output = agent.act(_incident(), TriageOutput)

        assert isinstance(output, TriageOutput)
        assert output.handoffs == []

    def test_an_output_type_that_cannot_be_defaulted_re_raises_the_interruption(self) -> None:
        """AC-2's honest bound: the caller sees the interruption, not a ValidationError."""

        class _Mandatory(BaseModel):
            verdict: str  # no default — TriageOutput's opposite

        agent = _make_custom_agent()
        interruption = RunInterruptedError("cancelled")
        agent._react_agent.run_sync.side_effect = interruption  # type: ignore[attr-defined]

        with pytest.raises(RunInterruptedError) as raised:
            agent.act(_incident(), _Mandatory)

        assert raised.value is interruption
        agent.notify_human.assert_called_once_with("Run interrupted.")  # type: ignore[attr-defined]


class TestCustomAgentOverridesNothing:
    """The structural half of AC-5: the policy is applied, never re-implemented."""

    def test_it_defines_only_its_own_router_handler_and_capability_hook(self) -> None:
        """Its own work and one supported hook — no framework method re-implemented.

        ``extra_capabilities`` is the third name because the exemplar contributes
        a capability of its own; it is an *override point the framework offers*,
        not a policy this class took over. The set stays exact rather than a
        superset check, so a ``CustomAgent`` that started overriding ``act``,
        ``_route_output`` or ``_build_react_agent`` still turns this red — which
        is the whole claim.
        """
        own = {
            name
            for name, value in vars(CustomAgent).items()
            if callable(value) and not name.startswith("__")
        }
        assert own == {"_route_triage", "receiveMsg_TriageMessage", "extra_capabilities"}

    def test_the_handler_carries_no_error_handling_and_no_decorator(self) -> None:
        """The handler is only the work — bypass ``act()`` and the breach escapes raw.

        The guard used to sit on this handler, and this spec unwrapped it through
        ``__wrapped__`` to show the two halves were separable. It sits on ``act()``
        now, so the handler is undecorated outright and the separation is shown by
        stubbing ``act`` away instead: nothing between ``run_sync`` and the caller
        catches anything, so the same breach comes straight out.
        """
        agent = _make_custom_agent()
        agent.act = MagicMock(side_effect=RunUsageLimitError("run request limit"))  # type: ignore[method-assign]

        assert not hasattr(CustomAgent.receiveMsg_TriageMessage, "__wrapped__")

        with pytest.raises(RunUsageLimitError):
            agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
