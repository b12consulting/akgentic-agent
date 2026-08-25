"""Real-chain durability of the mid-run arrival notice (Epic 20, FR4c).

Drives a **real** pydantic-ai run graph — the ``test_retry_wins_exhaustive``
pattern — with the ``MailboxCapability`` wired in as a capability, so the
delivery path under test is the shipped one: the hook enqueues the notice via
``ctx.enqueue(notice, priority="asap")`` and pydantic-ai's auto-injected,
outermost ``PendingMessageDrainCapability`` drains it into the next model
request, into the durable history, and — through ``ContextManager.add_message``
— into the ``LlmMessageEvent`` stream. A double that faked the drain would
assert what the author *believes* pydantic-ai does rather than what it does.

Three shapes are pinned, and the first two are opposites:

- the ordinary next-step-boundary delivery (mail pending from the start, a
  plain tool call creates the boundary the drain delivers into) — the notice
  text lands in the durable history (``react_agent.context.messages``) exactly
  once, and exactly one ``LlmMessageEvent`` carrying it reaches the observer;
- the **run-end withdrawal** (the notice is enqueued at the run's last step
  boundary, so the run would otherwise terminate with it still queued) — the
  notice is withdrawn, the run ends on its own ``End(FinalResult)``, and the
  answer the model already produced is what ``act()`` returns;
- the same run-end shape with a **second producer** also holding an ``'asap'``
  entry — the notice is withdrawn and the foreign entry is not, so the drain
  redirects for it exactly as it always did. The withdrawal is this
  capability's own, keyed on the ids it recorded; it is not a queue flush.

The second shape used to be the opposite claim: that the drain's
``after_node_run`` redirects through one extra model request so the notice is
"delivered rather than lost". It does — and the price is the run's own
``End(FinalResult)``, which that redirect **discards**. The answer the agent had
already written was then never returned by ``run_sync`` and reached nobody,
which is what issue #123 reported from a live process. The queued message was
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
from akgentic.agent.capabilities import render_arrival_notice
from akgentic.agent.config import AgentConfig
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

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    # No context state to deliver: this test is about the arrival notice, and a
    # Context update block would add a message the assertions would have to
    # discount.
    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    agent._mailbox_capability = MailboxCapability(observer=mailbox)

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    return agent


class _ForeignEnqueue(AbstractCapability[Any]):
    """Any other producer of queued content — a stand-in for one, at least.

    It enqueues once, at ``'asap'``, from ``before_model_request``: the same
    one-step-late position the arrival notice occupies, since the outermost
    drain has already run for that request by the time any other capability's
    hook fires. Its entry is therefore still queued when the run reaches its
    ``End``, which is the whole point — nothing withdraws it, so the drain's
    redirect fires for it exactly as it always did.
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
        """Next-step-boundary delivery: enqueued at firing 1, drained into request 2.

        Mail is pending from the run's start, so the hook's first firing
        enqueues the notice. The stub model's first response calls a plain tool
        — creating the step boundary whose model request the drain delivers
        into — and the second finalises via the output tool.
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
                agent.act(handled, StructuredOutput)
        finally:
            react_agent.close()

        # The boundary existed: turn 1 called the tool, turn 2 finalised.
        assert model_call_count == 2
        # Durable history holds the notice exactly once.
        assert _notice_count_in_history(react_agent.context.messages, notice) == 1
        # Exactly one LlmMessageEvent carried it to the observer.
        assert len(_notice_events(recorder.events, notice)) == 1

    def test_a_notice_left_at_the_run_end_is_withdrawn_and_the_answer_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run keeps the output it produced; the notice is withdrawn (#123).

        With no tool call, the hook's only firing is before request 1 — the
        drain has already run for that request, so the notice is still queued
        when the model finalises. Without the withdrawal, the drain's
        ``after_node_run`` **discards** that ``End(FinalResult)`` and redirects
        into a second model request; the run then finalises again and
        ``run_sync`` returns the *second* output. The first answer — the one the
        user was waiting for — is in durable history and nowhere else.

        So the assertion that matters is on the returned value, not on a queue
        length: the stub model answers differently each turn, and ``act()`` must
        return the **first** answer.

        MUTATION — delete ``MailboxCapability.after_node_run`` (the pre-#123
        state) and this spec goes red on ``model_call_count == 1`` first, then on
        the returned message. It is **the** regression guard for the bug, and the
        only one that fails on the returned output rather than on a queue.

        Measured against the full suite: that mutation reddens 5 of 447, and all
        five are withdrawal specs —
        ``test_a_foreign_producers_entry_still_redirects_at_the_run_end`` here,
        plus ``test_the_notice_is_withdrawn_when_the_run_has_ended``,
        ``test_another_producers_entry_is_left_in_the_queue`` and
        ``test_an_absorbed_messages_rendering_is_never_withdrawn`` in
        ``test_run_cancellation.py::TestRunEndWithdrawal``. What stays green is
        the point: ``test_notice_lands_in_durable_history_and_event_stream_exactly_once``
        above, and every cancel and offer spec — the mid-run doorbell is untouched.
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

        # The notice was withdrawn, so it reached neither history nor the stream.
        assert _notice_count_in_history(react_agent.context.messages, notice) == 0
        assert _notice_events(recorder.events, notice) == []

        # The message itself was never at risk: still queued, never consumed,
        # so it gets its own turn (ADR-010 §5).
        assert mailbox.pending == [arrived]
        assert mailbox.consumed == []

    def test_a_foreign_producers_entry_still_redirects_at_the_run_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The withdrawal is ours alone — everyone else keeps the redirect (AC-5).

        Same shape as the spec above, plus a second capability enqueueing its own
        ``'asap'`` content. Both entries are queued when the model finalises. The
        notice is withdrawn; the foreign entry is not, so pydantic-ai's drain
        discards the ``End`` and redirects through one more model request exactly
        as it does today, and ``act()`` returns the *second* answer.

        That is the correct outcome, not a residual bug: the drain's redirect is
        right for content with no other delivery path, and this capability knows
        of only one such exception — the arrival notice, whose message is still
        sitting in the actor mailbox. Narrowing the withdrawal to the ids this
        capability recorded (rather than clearing the queue) is what keeps that
        true, and this is the spec that says so.
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

        # The drain redirected for the foreign entry: a second turn happened.
        assert model_call_count == 2
        assert [m.message for m in output.messages] == ["answer from turn 2"]

        # The foreign content was delivered — once, into that redirect turn.
        assert _notice_count_in_history(react_agent.context.messages, _ForeignEnqueue.CONTENT) == 1
        # Our own notice was still withdrawn, and the mail still untouched.
        assert _notice_count_in_history(react_agent.context.messages, notice) == 0
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
        agent._mailbox_capability = MailboxCapability(observer=mailbox, arrival_notice=False)
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
        agent._mailbox_capability = MailboxCapability(observer=mailbox, arrival_notice=False)
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
