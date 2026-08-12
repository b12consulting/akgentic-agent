"""Tests that init_llm_context accepts core's EventMessage envelopes unchanged.

`BaseAgent.init_llm_context` is typed `list[EventMessage]` from `akgentic.core.messages`
— the concrete envelope `akgentic-core` declares on `Akgent.init_llm_context` and the one
`akgentic-team`'s restorer actually passes. Not `akgentic.llm.event.EventMessage`, which
is a structural stand-in existing only because `akgentic-llm` may not import core.

This is a typing correction with no behavioural change: the method remains a pure
pass-through, forwarding the list **by identity** to `ReactAgent.restore_context`.
"""

from unittest.mock import MagicMock

from akgentic.core.messages import EventMessage

from akgentic.agent.agent import BaseAgent


def _bare_agent_with_stubbed_llm() -> BaseAgent:
    """Bare BaseAgent (no Pykka actor system) with a stubbed ReactAgent."""
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = MagicMock()  # type: ignore[attr-defined]
    return agent


def _envelopes(count: int) -> list[EventMessage]:
    """Genuine core EventMessage envelopes carrying arbitrary payloads.

    The payload type is deliberately not an LLM event: restore_context is handed a
    team's whole event stream and ignores what it does not recognise.
    """
    return [EventMessage(event={"seq": i}) for i in range(count)]


class TestCoreEnvelopesAreAccepted:
    """A list of core EventMessages reaches ReactAgent.restore_context unchanged."""

    def test_forwards_core_envelopes_by_identity(self) -> None:
        agent = _bare_agent_with_stubbed_llm()
        events = _envelopes(3)

        agent.init_llm_context(events)

        restore = agent._react_agent.restore_context  # type: ignore[attr-defined]
        restore.assert_called_once_with(events)
        # Identity, not equality — the pass-through must not copy or rebuild the list.
        assert restore.call_args[0][0] is events

    def test_elements_arrive_in_order_and_unmodified(self) -> None:
        agent = _bare_agent_with_stubbed_llm()
        events = _envelopes(3)

        agent.init_llm_context(events)

        passed = agent._react_agent.restore_context.call_args[0][0]  # type: ignore[attr-defined]
        assert all(actual is expected for actual, expected in zip(passed, events, strict=True))
        assert [envelope.event for envelope in passed] == [{"seq": 0}, {"seq": 1}, {"seq": 2}]

    def test_every_forwarded_element_is_a_core_event_message(self) -> None:
        """Pins the element type the signature names, on the elements that ARRIVE.

        Asserting the helper's own output would assert isinstance on a class the
        helper just constructed — a test that cannot fail. The claim worth pinning
        is that core envelopes are what reaches restore_context.
        """
        agent = _bare_agent_with_stubbed_llm()

        agent.init_llm_context(_envelopes(2))

        passed = agent._react_agent.restore_context.call_args[0][0]  # type: ignore[attr-defined]
        assert passed and all(isinstance(envelope, EventMessage) for envelope in passed)

    def test_empty_list_forwarded(self) -> None:
        agent = _bare_agent_with_stubbed_llm()
        events: list[EventMessage] = []

        agent.init_llm_context(events)

        agent._react_agent.restore_context.assert_called_once_with(events)  # type: ignore[attr-defined]
