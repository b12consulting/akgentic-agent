"""Real-chain durability of the mid-run arrival notice (Epic 20, FR4c).

Drives a **real** pydantic-ai run graph — the ``test_retry_wins_exhaustive``
pattern — with the ``MailboxCapability`` wired in as a capability, so the
delivery path under test is the shipped one: the hook enqueues the notice via
``ctx.enqueue(notice, priority="asap")`` and pydantic-ai delivers it into the
model request, into the durable history, and — through
``ContextManager.add_message`` — into the ``LlmMessageEvent`` stream. A double
that faked the delivery would assert what the author *believes* pydantic-ai
does rather than what it does.

Since pydantic-ai 2.58 (this package's floor), ``'asap'`` content a
``before_model_request`` hook enqueues goes out with **that same request**, not
the next one. Three shapes are pinned:

- mail pending from the start — the model sees the notice on its first
  request, the notice lands in durable history (``react_agent.context.messages``)
  exactly once, exactly one ``LlmMessageEvent`` carries it, and a later request
  in the same run does not announce it again;
- a run that finalises on that first request — the notice already went out
  with it, so nothing is queued when the run reaches its ``End(FinalResult)``:
  no redirect turn, and the answer the model produced is what ``act()``
  returns;
- the same shape with a **second producer** also enqueueing at ``'asap'`` — its
  entry goes out with the same request too, so again no redirect.

Before 2.58 the notice arrived one request late, and a notice raised on the
final request was still queued at the run's end. pydantic-ai's drain then
discarded the ``End(FinalResult)`` and redirected through one more model
request, so the answer the agent had already written was never returned —
issue #123. ``MailboxCapability.after_node_run`` withdraws such a notice; the
same-request delivery now means a real chain no longer leaves one behind, and
the withdrawal itself is pinned at hook level in
``test_run_cancellation.py::TestRunEndWithdrawal``. The queued message was
never at risk: it is still in the actor mailbox and gets its own turn, exactly
as ADR-010 §5 specifies.
"""

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core import ActorAddress
from akgentic.core.messages import CancelMessage, Message
from akgentic.llm import LlmMessageEvent, ModelConfig, ReactAgent, ReactAgentConfig
from akgentic.tool.mailbox import MailboxTool, render_arrival_notice
from pydantic_ai import AgentCapability, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

from akgentic.agent.agent import BaseAgent, MailboxCapability, RunInterruptedError
from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

# =============================================================================
# HELPERS
# =============================================================================


class _MailboxDouble:
    """The ``MailboxAccess`` surface over a mutable pending list (no actor).

    Swept by hand when the Protocol widened: ``@runtime_checkable`` checks
    member presence, not signatures, so a fake that misses a method fails at
    the call rather than at construction.
    """

    def __init__(
        self,
        pending: list[AgentMessage] | None = None,
        current: AgentMessage | None = None,
    ) -> None:
        self.pending: list[AgentMessage] = list(pending or [])
        self.current = current
        self.consumed: list[uuid.UUID] = []

    def get_mailbox(self) -> list[AgentMessage]:
        return list(self.pending)

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[AgentMessage]:
        taken = [m for m in self.pending if m.id in message_ids]
        self.pending = [m for m in self.pending if m.id not in message_ids]
        self.consumed.extend(message_ids)
        return taken

    def current_message(self) -> AgentMessage | None:
        return self.current


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
    mailbox capability observes a mailbox double instead of the agent itself, so
    the test controls what mail is pending during the run.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    # The agent's own state, normally assigned in on_start. act() reads
    # state.tool_state.active_model to re-apply a persisted model selection;
    # an empty slot makes that a no-op, which is what these specs want.
    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    # No context state to deliver: this test is about the arrival notice, and a
    # Context update block would add a message the assertions would have to
    # discount.
    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    agent._mailbox_capability = MailboxCapability(observer=mailbox, card=MailboxTool())

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    return agent


class _ForeignEnqueue(AbstractCapability[Any]):
    """Any other producer of queued content — a stand-in for one, at least.

    It enqueues once, at ``'asap'``, from ``before_model_request``: the same
    position the arrival notice occupies, so its entry goes out with the same
    request the notice does.
    """

    CONTENT = "a note from somewhere else entirely"

    def __init__(self) -> None:
        self.enqueued = False

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if not self.enqueued:
            self.enqueued = True
            ctx.enqueue(self.CONTENT, priority="asap")
        return request_context


def _build_react_agent(
    agent: BaseAgent,
    observer: _EventRecorder,
    extra: list[AgentCapability[Any]] | None = None,
) -> ReactAgent:
    """A real ReactAgent carrying the agent's mailbox capability and the recorder.

    The provider is ``google-gla`` on purpose (see ``test_retry_wins_exhaustive``):
    only the non-native path makes the output a discrete output tool call, which
    the stub model needs to finalise a turn. The API key is never dereferenced —
    ``FunctionModel`` replaces the model before any run happens.

    ``extra`` goes *after* the mailbox capability, mirroring what
    ``_assemble_capabilities`` builds for a subclass's ``extra_capabilities()``.
    """
    react_config = ReactAgentConfig(
        model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
    )
    react_agent = ReactAgent(
        config=react_config,
        deps_type=BaseAgent,
        observer=observer,
        capabilities=[agent._mailbox_capability, *(extra or [])],
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
        """Same-request delivery: enqueued before request 1, sent with request 1.

        Mail is pending from the run's start, so the hook's first firing
        enqueues the notice and the model sees it on its first call. The stub
        model's first response calls a plain tool, so a second request follows —
        and it must not announce the same mail again.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        arrived = _pending_message("news", "@Alice")
        handled = _pending_message("carry on", "@Human")
        notice = render_arrival_notice([arrived], {arrived.id})
        agent = _make_minimal_agent(_MailboxDouble([arrived], current=handled))
        recorder = _EventRecorder()
        react_agent = _build_react_agent(agent, recorder)

        @react_agent.pydantic_agent.tool_plain
        def check_status() -> str:
            return "all good"

        model_call_count = 0
        # Per model call: how many copies of the notice the model was sent.
        seen: list[int] = []

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_call_count
            model_call_count += 1
            seen.append(_notice_count_in_history(messages, notice))
            output_tool_name = info.output_tools[0].name
            if model_call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="check_status", args={}, tool_call_id="fn-1")]
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
                agent.act(handled, StructuredOutput)
        finally:
            react_agent.close()

        # Turn 1 called the tool, turn 2 finalised.
        assert model_call_count == 2
        # Sent with the first request; the second carries that same copy, no new one.
        assert seen == [1, 1]
        # Durable history holds the notice exactly once.
        assert _notice_count_in_history(react_agent.context.messages, notice) == 1
        # Exactly one LlmMessageEvent carried it to the observer.
        assert len(_notice_events(recorder.events, notice)) == 1

    def test_a_notice_raised_on_the_final_request_goes_out_with_it_and_the_answer_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run keeps the output it produced (#123).

        With no tool call, the hook's only firing is before request 1, and the
        model finalises on that request. The notice goes out with it, so nothing
        is queued when the run reaches its ``End(FinalResult)`` and there is no
        redirect into a second model request — the one that used to make
        ``run_sync`` return the *second* output and leave the first answer, the
        one the user was waiting for, in durable history and nowhere else.

        So the assertion that matters is on the returned value, not on a queue
        length: the stub model answers differently each turn, and ``act()`` must
        return the **first** answer.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        arrived = _pending_message("late news", "@Bob")
        handled = _pending_message("wrap up", "@Human")
        notice = render_arrival_notice([arrived], {arrived.id})
        mailbox = _MailboxDouble([arrived], current=handled)
        agent = _make_minimal_agent(mailbox)
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
                        args={
                            "messages": [
                                {
                                    "message_type": "response",
                                    "message": f"answer from turn {model_call_count}",
                                    "recipient": "@Human",
                                }
                            ]
                        },
                        tool_call_id=f"out-{model_call_count}",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                output = agent.act(handled, StructuredOutput)
        finally:
            react_agent.close()

        # The run ended where it said it would — no redirect turn.
        assert model_call_count == 1
        # ...and the answer it produced is the one the caller got.
        assert [m.message for m in output.messages] == ["answer from turn 1"]

        # The notice went out with that one request: once in history, once in the stream.
        assert _notice_count_in_history(react_agent.context.messages, notice) == 1
        assert len(_notice_events(recorder.events, notice)) == 1

        # The message itself was never at risk: still queued, never consumed,
        # so it gets its own turn (ADR-010 §5).
        assert mailbox.pending == [arrived]
        assert mailbox.consumed == []

    def test_a_foreign_producers_entry_goes_out_with_the_same_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second producer's entry is delivered alongside the notice (AC-5).

        Same shape as the spec above, plus a second capability enqueueing its own
        ``'asap'`` content from the same hook. Both entries go out with request 1,
        so neither is queued when the model finalises: no redirect, and ``act()``
        returns the first answer. Each lands in history exactly once — the mailbox
        capability neither drops nor duplicates another producer's content.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        arrived = _pending_message("late news", "@Bob")
        handled = _pending_message("wrap up", "@Human")
        notice = render_arrival_notice([arrived], {arrived.id})
        mailbox = _MailboxDouble([arrived], current=handled)
        agent = _make_minimal_agent(mailbox)
        recorder = _EventRecorder()
        foreign = _ForeignEnqueue()
        react_agent = _build_react_agent(agent, recorder, extra=[foreign])

        model_call_count = 0

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_call_count
            model_call_count += 1
            output_tool_name = info.output_tools[0].name
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=output_tool_name,
                        args={
                            "messages": [
                                {
                                    "message_type": "response",
                                    "message": f"answer from turn {model_call_count}",
                                    "recipient": "@Human",
                                }
                            ]
                        },
                        tool_call_id=f"out-{model_call_count}",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                output = agent.act(handled, StructuredOutput)
        finally:
            react_agent.close()

        # Nothing was left queued at the end: no redirect, the first answer stands.
        assert model_call_count == 1
        assert [m.message for m in output.messages] == ["answer from turn 1"]

        # Both entries went out with that request, each exactly once.
        assert _notice_count_in_history(react_agent.context.messages, _ForeignEnqueue.CONTENT) == 1
        assert _notice_count_in_history(react_agent.context.messages, notice) == 1
        # The mail itself is still untouched.
        assert mailbox.pending == [arrived]


class _CancelDouble(AgentMessage):
    """An ordinary message whose content is a ``/stop`` — the everyday cancel."""


class TestArrivalNoticeIsGatedOnTheReadTool:
    """Story 26-2: no doorbell when the model has no way to answer it.

    Two halves, and the second is the one that matters. Suppressing the notice
    is the feature; **still being cancellable while suppressed** is the
    invariant, because the gate sits one line away from the purge-and-raise that
    enforces it.
    """

    def test_a_suppressed_run_announces_nothing_and_still_returns_its_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the notice off, mail is neither rendered nor enqueued.

        The run also keeps its own output — trivially, since nothing was queued
        for the drain to redirect on. Asserted anyway: it stops a later refactor
        from re-coupling suppression and withdrawal.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        arrived = _pending_message("late news", "@Bob")
        handled = _pending_message("wrap up", "@Human")
        notice = render_arrival_notice([arrived], {arrived.id})
        mailbox = _MailboxDouble([arrived], current=handled)

        agent = _make_minimal_agent(mailbox)
        agent._mailbox_capability = MailboxCapability(
            observer=mailbox, card=MailboxTool(read_mailbox=False)
        )
        recorder = _EventRecorder()
        react_agent = _build_react_agent(agent, recorder)

        model_call_count = 0

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_call_count
            model_call_count += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={
                            "messages": [
                                {
                                    "message_type": "response",
                                    "message": f"answer from turn {model_call_count}",
                                    "recipient": "@Human",
                                }
                            ]
                        },
                        tool_call_id=f"out-{model_call_count}",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                output = agent.act(handled, StructuredOutput)
        finally:
            react_agent.close()

        assert model_call_count == 1
        assert [m.message for m in output.messages] == ["answer from turn 1"]
        assert _notice_count_in_history(react_agent.context.messages, notice) == 0
        assert _notice_events(recorder.events, notice) == []
        # Untouched: suppressing the doorbell must not consume anyone's mail.
        assert mailbox.pending == [arrived]
        assert mailbox.consumed == []

    @pytest.mark.parametrize(
        "cancel",
        [
            pytest.param(CancelMessage(), id="CancelMessage"),
            pytest.param(_CancelDouble(content="/stop", type="request"), id="slash-stop"),
        ],
    )
    def test_a_suppressed_run_is_still_cancellable(
        self, cancel: Message, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate silences the doorbell; it must not disarm the cancel.

        MUTATION — move the ``if not self._arrival_notice`` gate ABOVE the cancel
        block in ``before_model_request`` and both parameters go red: the run
        completes normally instead of raising. That is the mutation a reader
        would actually make, because the gate reads like a cheap early-out for
        the whole method, and it is why this spec exists.

        ``is_cancel`` recognises two forms and only one is a ``CancelMessage``,
        so both are driven — a gate that happened to special-case the class
        would still be caught by the ``/stop`` parameter.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        handled = _pending_message("wrap up", "@Human")
        mailbox = _MailboxDouble(current=handled)
        mailbox.pending = [cancel]  # type: ignore[list-item]

        agent = _make_minimal_agent(mailbox)
        agent._mailbox_capability = MailboxCapability(
            observer=mailbox, card=MailboxTool(read_mailbox=False)
        )
        react_agent = _build_react_agent(agent, _EventRecorder())

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise AssertionError("the cancel must fire before any model request")

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                with pytest.raises(RunInterruptedError):
                    react_agent.run_sync("anything", deps=agent, output_type=StructuredOutput)
        finally:
            react_agent.close()

        # Recognising the cancel and consuming it are one act.
        assert mailbox.consumed == [cancel.id]
