"""Real-chain durability of the mid-run arrival notice (Epic 20, FR4c).

Drives a **real** pydantic-ai run graph — the ``test_retry_wins_exhaustive``
pattern — with the ``MailboxCancelCapability`` wired in as a capability, so the
delivery path under test is the shipped one: the hook enqueues the notice via
``ctx.enqueue(notice, priority="asap")`` and pydantic-ai's auto-injected,
outermost ``PendingMessageDrainCapability`` drains it into the next model
request, into the durable history, and — through ``ContextManager.add_message``
— into the ``LlmMessageEvent`` stream. A double that faked the drain would
assert what the author *believes* pydantic-ai does rather than what it does.

Two shapes are pinned:

- the ordinary next-step-boundary delivery (mail pending from the start, a
  plain tool call creates the boundary the drain delivers into), and
- the end-of-run redirect (the notice is enqueued at the run's last step
  boundary; the drain's ``after_node_run`` redirects through one extra model
  request so the notice is delivered rather than lost).

Both assert the same durability contract: the notice text lands in the durable
history (``react_agent.context.messages``) exactly once, and exactly one
``LlmMessageEvent`` carrying it reaches the observer.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from akgentic.core import ActorAddress
from akgentic.llm import LlmMessageEvent, ModelConfig, ReactAgent, ReactAgentConfig
from akgentic.tool.mailbox import render_arrival_notice
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from akgentic.agent.agent import BaseAgent, MailboxCancelCapability
from akgentic.agent.config import AgentConfig
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

# =============================================================================
# HELPERS
# =============================================================================


class _MailboxDouble:
    """Exposes ``get_mailbox()`` over a mutable pending list (no actor)."""

    def __init__(self, pending: list[AgentMessage] | None = None) -> None:
        self.pending: list[AgentMessage] = list(pending or [])

    def get_mailbox(self) -> list[AgentMessage]:
        return list(self.pending)


class _EventRecorder:
    """ContextObserver double: records every notified domain event."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify_event(self, event: object) -> None:
        self.events.append(event)


def _pending_message(content: str = "please review", sender_name: str = "@Alice") -> AgentMessage:
    message = AgentMessage(content=content, type="request")
    sender = MagicMock(spec=ActorAddress)
    sender.name = sender_name
    message.sender = sender
    return message


def _make_minimal_agent(mailbox: _MailboxDouble) -> BaseAgent:
    """Construct a BaseAgent without the Pykka actor system.

    Same shape as ``test_retry_wins_exhaustive._make_minimal_agent``, except the
    cancel capability observes a mailbox double instead of the agent itself, so
    the test controls what mail is pending during the run.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    agent._context_state_providers = []  # type: ignore[attr-defined]
    agent._context_baselines = {}  # type: ignore[attr-defined]
    agent._context_update_seq = 0  # type: ignore[attr-defined]

    agent._cancel_capability = MailboxCancelCapability(observer=mailbox)

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    return agent


def _build_react_agent(agent: BaseAgent, observer: _EventRecorder) -> ReactAgent:
    """A real ReactAgent carrying the agent's cancel capability and the recorder.

    The provider is ``google-gla`` on purpose (see ``test_retry_wins_exhaustive``):
    only the non-native path makes the output a discrete output tool call, which
    the stub model needs to finalise a turn. The API key is never dereferenced —
    ``FunctionModel`` replaces the model before any run happens.
    """
    react_config = ReactAgentConfig(
        model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
    )
    react_agent = ReactAgent(
        config=react_config,
        deps_type=BaseAgent,
        observer=observer,
        capabilities=[agent._cancel_capability],
    )
    agent._react_agent = react_agent  # type: ignore[attr-defined]
    return react_agent


def _empty_output_args() -> dict[str, list[object]]:
    """Valid ``StructuredOutput`` tool args routing nothing."""
    return {"messages": []}


def _notice_count_in_history(messages: list[ModelMessage], notice: str) -> int:
    """How many user-prompt parts of the durable history are exactly the notice."""
    count = 0
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and part.content == notice:
                count += 1
    return count


def _notice_events(events: list[object], notice: str) -> list[LlmMessageEvent]:
    """The recorded ``LlmMessageEvent``s whose message carries the notice."""
    return [
        event
        for event in events
        if isinstance(event, LlmMessageEvent)
        and _notice_count_in_history([event.message], notice) > 0
    ]


# =============================================================================
# FR4c — the notice is durable: history and event stream, exactly once
# =============================================================================


class TestArrivalNoticeDurability:
    def test_notice_lands_in_durable_history_and_event_stream_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Next-step-boundary delivery: enqueued at firing 1, drained into request 2.

        Mail is pending from the run's start, so the hook's first firing
        enqueues the notice. The stub model's first response calls a plain tool
        — creating the step boundary whose model request the drain delivers
        into — and the second finalises via the output tool.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        arrived = _pending_message("news", "@Alice")
        notice = render_arrival_notice([arrived])
        agent = _make_minimal_agent(_MailboxDouble([arrived]))
        recorder = _EventRecorder()
        react_agent = _build_react_agent(agent, recorder)

        @react_agent.pydantic_agent.tool_plain
        def check_status() -> str:
            return "all good"

        model_call_count = 0

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_call_count
            model_call_count += 1
            output_tool_name = info.output_tools[0].name
            if model_call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name="check_status", args={}, tool_call_id="fn-1")
                    ]
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=output_tool_name,
                        args=_empty_output_args(),
                        tool_call_id="out-1",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                agent.act("carry on", StructuredOutput)
        finally:
            react_agent.close()

        # The boundary existed: turn 1 called the tool, turn 2 finalised.
        assert model_call_count == 2
        # Durable history holds the notice exactly once.
        assert _notice_count_in_history(react_agent.context.messages, notice) == 1
        # Exactly one LlmMessageEvent carried it to the observer.
        assert len(_notice_events(recorder.events, notice)) == 1

    def test_end_of_run_redirect_delivers_a_leftover_notice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A notice enqueued at the run's last boundary costs one extra model turn.

        With no tool call, the only firing is before request 1 — the drain has
        already run for that request, so the notice is still queued when the
        model finalises. The drain's ``after_node_run`` then redirects through
        one more model request instead of losing it: the model sees the notice
        and finalises again. Same durability contract — history and event
        stream, exactly once.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        arrived = _pending_message("late news", "@Bob")
        notice = render_arrival_notice([arrived])
        agent = _make_minimal_agent(_MailboxDouble([arrived]))
        recorder = _EventRecorder()
        react_agent = _build_react_agent(agent, recorder)

        model_call_count = 0

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_call_count
            model_call_count += 1
            output_tool_name = info.output_tools[0].name
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=output_tool_name,
                        args=_empty_output_args(),
                        tool_call_id=f"out-{model_call_count}",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                agent.act("wrap up", StructuredOutput)
        finally:
            react_agent.close()

        # The redirect turn is the documented occasional extra model call.
        assert model_call_count == 2
        assert _notice_count_in_history(react_agent.context.messages, notice) == 1
        assert len(_notice_events(recorder.events, notice)) == 1
