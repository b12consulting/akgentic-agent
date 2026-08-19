"""``CustomAgent`` — the claim the whole extraction rests on, asserted.

A second agent class, with its own structured output and its own message type,
must get the usage-limit tier policy **without overriding anything**: the schema
and the router are decorator arguments, so a run-tier breach in ``CustomAgent``
concludes in ``TriageOutput`` and is delivered by ``_route_triage``.

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
    agent.team_id = uuid.uuid4()

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

        prompt, output_type = agent.act.call_args[0]  # type: ignore[attr-defined]
        assert output_type is TriageOutput
        assert "disk full on node 3" in prompt
        assert "monitoring" in prompt

        agent.send.assert_called_once()  # type: ignore[attr-defined]
        _, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert isinstance(sent, AgentMessage)
        assert sent.content == "Free 20GB on node 3"


class TestCustomAgentRunTierBreach:
    """AC-5: a run-tier breach concludes in TriageOutput, routed by _route_triage."""

    def test_the_conclusion_is_asked_for_this_agents_schema(self) -> None:
        agent = _make_custom_agent()
        agent.act = MagicMock(side_effect=RunUsageLimitError("run request limit"))  # type: ignore[method-assign]
        agent._react_agent.conclude_without_tools_sync.return_value = _triage()  # type: ignore[attr-defined]

        agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        call = agent._react_agent.conclude_without_tools_sync.call_args  # type: ignore[attr-defined]
        assert call.kwargs["output_type"] is TriageOutput
        assert call.kwargs["deps"] is agent
        # The recipient is model-chosen, so the reason has to name who to answer.
        assert REQUESTER in call[0][0]

    def test_the_requester_receives_the_handoff_and_no_human_is_paged(self) -> None:
        agent = _make_custom_agent()
        agent.act = MagicMock(side_effect=RunUsageLimitError("run request limit"))  # type: ignore[method-assign]
        agent._react_agent.conclude_without_tools_sync.return_value = _triage(  # type: ignore[attr-defined]
            task="Partial triage: node 3 is out of disk, cause not yet identified."
        )

        agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent.get_team_member.assert_called_once_with(REQUESTER)  # type: ignore[attr-defined]
        agent.send.assert_called_once()  # type: ignore[attr-defined]
        target, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert target.name == REQUESTER
        assert isinstance(sent, AgentMessage)
        assert sent.content.startswith("Partial triage:")

        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]

    def test_a_conclusion_with_no_handoffs_escalates(self) -> None:
        """_route_triage's bool is the only thing that can tell the guard.

        ``TriageOutput`` has no ``.messages``, so the guard cannot inspect it —
        "did anything go out?" is the router's answer to give, and an empty triage
        is a failure exactly as an empty StructuredOutput is.
        """
        agent = _make_custom_agent()
        agent.act = MagicMock(side_effect=RunUsageLimitError("original run breach"))  # type: ignore[method-assign]
        agent._react_agent.conclude_without_tools_sync.return_value = TriageOutput(  # type: ignore[attr-defined]
            severity="low", summary="nothing conclusive"
        )

        with pytest.raises(WarningError, match="original run breach"):
            agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]

    def test_an_agent_tier_breach_is_terminal_here_too(self) -> None:
        agent = _make_custom_agent()
        agent.act = MagicMock(side_effect=AgentUsageLimitError("lifetime budget spent"))  # type: ignore[method-assign]

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_TriageMessage(_incident(), _address(REQUESTER))

        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]

    def test_an_incident_with_no_sender_is_not_concluded_to(self) -> None:
        agent = _make_custom_agent()
        agent.act = MagicMock(side_effect=RunUsageLimitError("original run breach"))  # type: ignore[method-assign]
        senderless = _incident(sender=None)
        assert senderless.sender is None

        with pytest.raises(WarningError, match="original run breach"):
            agent.receiveMsg_TriageMessage(senderless, _address(REQUESTER))

        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent.hire_member.assert_not_called()  # type: ignore[attr-defined]


class TestCustomAgentOverridesNothing:
    """The structural half of AC-5: the policy is applied, never re-implemented."""

    def test_it_defines_only_its_own_router_and_handler(self) -> None:
        own = {
            name
            for name, value in vars(CustomAgent).items()
            if callable(value) and not name.startswith("__")
        }
        assert own == {"_route_triage", "receiveMsg_TriageMessage"}

    def test_the_handler_itself_carries_no_error_handling(self) -> None:
        """Undecorated, the same breach escapes — so the guard is what catches it.

        ``@wraps`` keeps the original reachable, which makes the two halves
        separable: the handler is only the work, and the policy is only the
        decorator. A handler that had kept its own ``except`` would swallow this.
        """
        agent = _make_custom_agent()
        agent.act = MagicMock(side_effect=RunUsageLimitError("run request limit"))  # type: ignore[method-assign]
        undecorated = CustomAgent.receiveMsg_TriageMessage.__wrapped__  # type: ignore[attr-defined]

        with pytest.raises(RunUsageLimitError):
            undecorated(agent, _incident(), _address(REQUESTER))

        agent._react_agent.conclude_without_tools_sync.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
