"""Tests for run cancellation and the mid-run arrival notice (Epic 20).

Two layers, matching the design's two halves:

The capability list itself is pinned in ``TestCapabilityWiring`` (assembly and
order, both build sites) and in ``TestExtraCapabilityFires`` (a subclass's own
capability actually running inside a real run).

- Hook level: ``MailboxCapability.before_model_request`` is invoked
  directly against a double exposing ``get_mailbox()``, a recording ``ctx``
  double exposing ``enqueue``, and a fabricated request context — no LLM, no
  actor. This is where the cancel-before-notice ordering, the
  announce-once/growth-only tracking, and the enqueue contract (one
  ``ctx.enqueue(notice, priority="asap")`` per growth, nothing appended by
  the hook itself) are pinned. Durable delivery through pydantic-ai's drain
  is pinned by the real-chain test in ``test_arrival_notice_durability.py``.
- Actor level: the catch site in ``receiveMsg_AgentMessage`` and the
  ``receiveMsg_CancelMessage`` handler run through the real actor system with
  a ReactAgent double whose ``run_sync`` raises ``RunInterruptedError`` on
  demand — simulating the capability raising mid-run.
"""

import asyncio
import logging
import sys
import time
import types
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import ActorAddress, ActorSystem, BaseConfig, Orchestrator
from akgentic.core.messages import CancelMessage, HandledMessage
from akgentic.llm import ModelConfig, PromptTemplate, ReactAgent, ReactAgentConfig
from akgentic.tool import MailboxTool
from akgentic.tool.mailbox import is_cancel, render_arrival_notice
from akgentic.tool.mailbox.capability import (
    _CLOSING_WITH_IDS,
    _CLOSING_WITHOUT_IDS,
    ABSORBED_PREFIX,
    MESSAGE_ID_ARG,
    READ_MAILBOX_TOOL,
)
from pydantic_ai import Agent, AgentCapability
from pydantic_ai._enqueue import PendingMessage
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.result import FinalResult
from pydantic_graph import End

import akgentic.agent
import akgentic.agent.agent as agent_module
from akgentic.agent import RunInterruptedError
from akgentic.agent.agent import BaseAgent, MailboxCapability
from akgentic.agent.config import AgentConfig
from akgentic.agent.custom_agent import CustomAgent, TriageMessage, TriageOutput
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

AGENT_LOGGER = "akgentic.agent.agent"

# =============================================================================
# HELPERS — hook level
# =============================================================================


class _MailboxDouble:
    """A mutable pending list behind the two methods ``MailboxAccess`` names.

    ``consume_mailbox`` is real, not a spy: it removes the named ids from
    ``pending``, so a spec can assert what is *left* in the mailbox rather than
    only what the hook asked for. It records its calls too, because "exactly the
    cancels' ids, and nothing else" is a separate claim from "the cancel is
    gone". It emits no telemetry — that is the core primitive's job, and a
    double that emitted any would hide the hook emitting one as well.
    """

    def __init__(self, pending: list[Any] | None = None, current: Any = None) -> None:
        self.pending: list[Any] = list(pending or [])
        self.consume_calls: list[list[uuid.UUID]] = []
        self.current = current

    def get_mailbox(self) -> list[Any]:
        return list(self.pending)

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Any]:
        self.consume_calls.append(list(message_ids))
        wanted = set(message_ids)
        removed = [message for message in self.pending if message.id in wanted]
        self.pending = [message for message in self.pending if message.id not in wanted]
        return removed

    def current_message(self) -> Any:
        """The handler's message — what the offer rule matches a pending class against.

        Added by hand when ``MailboxAccess`` widened: the Protocol checks member
        presence, not signatures, so a fake missing this fails at the call site
        rather than at construction.
        """
        return self.current


def _pending_message(content: str = "please review", sender_name: str = "@Alice") -> AgentMessage:
    message = AgentMessage(content=content, type="request")
    sender = MagicMock(spec=ActorAddress)
    sender.name = sender_name
    message.sender = sender
    return message


def _context(messages: list[Any] | None = None) -> Any:
    """Fabricated ModelRequestContext double: ``messages`` is a plain list."""
    return SimpleNamespace(messages=messages if messages is not None else [])


class _CtxDouble:
    """Recording RunContext double: exposes ``enqueue`` and records every call."""

    def __init__(self) -> None:
        self.enqueue_calls: list[tuple[tuple[Any, ...], Any]] = []

    def enqueue(self, *content: Any, priority: Any = "asap") -> str:
        self.enqueue_calls.append((content, priority))
        return f"enqueue-{len(self.enqueue_calls)}"


class _QueueCtxDouble:
    """RunContext double carrying a real ``pending_messages`` queue.

    ``enqueue`` builds its entry through pydantic-ai's own
    ``PendingMessage.from_content`` and returns *that entry's* ``enqueue_id``,
    which is what makes the withdrawal specs mean anything: a double handing out
    ids of its own invention would let a withdrawal keyed on the wrong thing
    pass, because the id the capability recorded and the id sitting on the queue
    entry would agree only by construction of the double.
    """

    def __init__(self) -> None:
        self.pending_messages: list[PendingMessage] = []

    def enqueue(self, *content: Any, priority: Any = "asap") -> str | None:
        pending = PendingMessage.from_content(*content, priority=priority)
        if pending is None:
            return None
        self.pending_messages.append(pending)
        return pending.enqueue_id


def _queued_text(pending: PendingMessage) -> str:
    """The user-prompt text one queue entry would deliver, concatenated."""
    return "".join(
        part.content
        for message in pending.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def _end() -> End[FinalResult[str]]:
    """The graph result that ends a run — what the withdrawal keys on."""
    return End(FinalResult(output="the answer the run already produced"))


_MID_RUN_NODE = object()
"""Stand-in for a node result that is not an ``End``, and for the node itself.

The hook reads nothing off either — ``isinstance(result, End)`` is the whole
test — so anything that is not an ``End`` exercises the mid-run branch exactly,
and the same sentinel serves as the ``node`` argument every spec must pass.
"""


# =============================================================================
# FR3 — RunInterruptedError declaration and export
# =============================================================================


class TestRunInterruptedError:
    def test_is_a_plain_exception_with_message_only(self) -> None:
        error = RunInterruptedError("cancelled")
        assert isinstance(error, Exception)
        assert str(error) == "cancelled"

    def test_exported_from_public_api(self) -> None:
        assert "RunInterruptedError" in akgentic.agent.__all__
        assert akgentic.agent.RunInterruptedError is RunInterruptedError


# =============================================================================
# FR4 — the cancel check
# =============================================================================


class TestCancelCheck:
    async def test_pending_stop_raises(self) -> None:
        capability = MailboxCapability(
            observer=_MailboxDouble([_pending_message("/stop")]), card=MailboxTool()
        )

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

    async def test_pending_cancel_message_raises(self) -> None:
        capability = MailboxCapability(
            observer=_MailboxDouble([CancelMessage()]), card=MailboxTool()
        )

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

    async def test_cancel_buried_behind_other_mail_still_raises(self) -> None:
        pending = [_pending_message("hello"), _pending_message("/stop now", "@Bob")]
        capability = MailboxCapability(observer=_MailboxDouble(pending), card=MailboxTool())

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

    async def test_cancel_check_runs_before_the_notice(self) -> None:
        """A pending /stop raises — the new mail beside it is never announced."""
        pending = [_pending_message("hello"), _pending_message("/stop")]
        capability = MailboxCapability(observer=_MailboxDouble(pending), card=MailboxTool())
        ctx = _CtxDouble()

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(ctx, _context())

        assert ctx.enqueue_calls == []
        assert capability._announced_ids == set()

    async def test_empty_mailbox_neither_raises_nor_enqueues(self) -> None:
        capability = MailboxCapability(observer=_MailboxDouble(), card=MailboxTool())
        ctx = _CtxDouble()
        context = _context()

        result = await capability.before_model_request(ctx, context)

        assert result is context
        assert ctx.enqueue_calls == []


# =============================================================================
# FR4c — the mid-run arrival notice, delivered via ctx.enqueue
# =============================================================================


class TestArrivalNotice:
    async def test_growth_enqueues_one_notice_at_asap_priority(self) -> None:
        """One growth, one ``ctx.enqueue`` call — the hook itself appends nothing."""
        arrived = _pending_message("news", "@Alice")
        handled = _pending_message("the turn prompt", "@Human")
        capability = MailboxCapability(
            observer=_MailboxDouble([arrived], current=handled), card=MailboxTool()
        )
        existing = ModelRequest(parts=[UserPromptPart(content="the turn prompt")])
        existing_parts = existing.parts
        ctx = _CtxDouble()
        context = _context([existing])

        result = await capability.before_model_request(ctx, context)

        assert result is context
        assert context.messages == [existing]  # delivery is the drain's job, not the hook's
        assert existing.parts is existing_parts  # no part-level mutation
        assert ctx.enqueue_calls == [((render_arrival_notice([arrived], {arrived.id}),), "asap")]

    async def test_same_message_is_announced_once_across_firings(self) -> None:
        arrived = _pending_message()
        capability = MailboxCapability(observer=_MailboxDouble([arrived]), card=MailboxTool())
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())
        await capability.before_model_request(ctx, _context())

        assert len(ctx.enqueue_calls) == 1

    async def test_second_arrival_announces_only_the_growth(self) -> None:
        first = _pending_message("one", "@Alice")
        second = _pending_message("two", "@Bob")
        mailbox = _MailboxDouble([first], current=_pending_message("handled", "@Human"))
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())
        mailbox.pending.append(second)
        await capability.before_model_request(ctx, _context())

        assert len(ctx.enqueue_calls) == 2
        growth_content, growth_priority = ctx.enqueue_calls[-1]
        assert growth_content == (render_arrival_notice([second], {second.id}),)
        assert growth_priority == "asap"

    async def test_before_run_forgets_the_announced_backlog(self) -> None:
        """The run-start hook is what clears the set — no caller has to remember."""
        arrived = _pending_message()
        capability = MailboxCapability(observer=_MailboxDouble([arrived]), card=MailboxTool())
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())
        await capability.before_run(ctx)
        await capability.before_model_request(ctx, _context())

        assert len(ctx.enqueue_calls) == 2  # the backlog re-announced after reset

    async def test_without_before_run_the_backlog_stays_announced(self) -> None:
        """The complement: nothing else clears the set, so the hook is load-bearing."""
        arrived = _pending_message()
        capability = MailboxCapability(observer=_MailboxDouble([arrived]), card=MailboxTool())
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())
        await capability.before_model_request(ctx, _context())

        assert len(ctx.enqueue_calls) == 1

    async def test_a_real_run_calls_before_run(self) -> None:
        """pydantic-ai drives the hook — the two specs above only prove it works.

        Calling ``before_run`` by hand says nothing about whether anything ever
        calls it, and that gap is the entire risk of moving the reset off
        ``act()``: a hook the framework does not recognise fails silently and
        permanently, leaking announced ids across every run for the agent's life
        with no error anywhere. So this drives a real ``Agent.run`` and asserts
        the set came back empty.
        """
        capability = MailboxCapability(observer=_MailboxDouble([]), card=MailboxTool())
        capability._announced_ids.add(uuid.uuid4())

        def _reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="done")])

        await Agent(FunctionModel(_reply), capabilities=[capability]).run("anything")

        assert capability._announced_ids == set()

    async def test_no_growth_enqueues_nothing(self) -> None:
        arrived = _pending_message()
        mailbox = _MailboxDouble([arrived])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())
        await capability.before_model_request(_CtxDouble(), _context())

        ctx = _CtxDouble()
        await capability.before_model_request(ctx, _context())

        assert ctx.enqueue_calls == []


# =============================================================================
# Epic 27 — the injected prompt text is the capability's, not the module's
# =============================================================================


class TestTheInjectedTextIsWhatTheCapabilityWasBuiltWith:
    """A custom string in, the same string out — for both injected strings.

    The invariant, never the phrasing: story 26-4 retired an assertion on a
    clause of the prefix for exactly that reason, and re-coupling these specs to
    wording would make the next tuning pass a test failure for no behavioural
    reason. What matters is that the value the capability was constructed with is
    the value that reaches the run.
    """

    _CLOSING = "SENTINEL CLOSING — configured on the card."

    async def test_the_capabilitys_closing_line_closes_the_notice(self) -> None:
        """AC 3 — MUTATION: pass ``_CLOSING_WITH_IDS`` instead of
        ``self._arrival_closing`` at the ``render_arrival_notice`` call in
        ``before_model_request`` and this goes red alone.
        """
        arrived = _pending_message("news", "@Alice")
        handled = _pending_message("the turn prompt", "@Human")
        capability = MailboxCapability(
            observer=_MailboxDouble([arrived], current=handled),
            card=MailboxTool(arrival_closing=self._CLOSING),
        )
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())

        (notice,), _priority = ctx.enqueue_calls[0]
        assert str(arrived.id) in notice, "the listing must have offered an id at all"
        assert notice.endswith(self._CLOSING)

    async def test_an_id_less_listing_keeps_the_unconfigurable_closing(self) -> None:
        """AC 5 — no id on offer, so no configured closing either.

        ``_MailboxDouble`` with no ``current`` is the idle case: nothing can be
        offered, so the notice carries no id and must not promise a read.
        """
        arrived = _pending_message()
        capability = MailboxCapability(
            observer=_MailboxDouble([arrived]), card=MailboxTool(arrival_closing=self._CLOSING)
        )
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())

        (notice,), _priority = ctx.enqueue_calls[0]
        assert notice.endswith(_CLOSING_WITHOUT_IDS)
        assert self._CLOSING not in notice

    async def test_a_capability_built_with_neither_string_behaves_as_it_always_has(self) -> None:
        """Both parameters are optional; the module constants are the defaults."""
        capability = MailboxCapability(observer=_MailboxDouble(), card=MailboxTool())

        assert capability._absorbed_prefix == ABSORBED_PREFIX
        assert capability._arrival_closing == _CLOSING_WITH_IDS


class TestCancellationConsultsNeitherString:
    """AC 8 — a capability built with no usable prompt text is still interruptible.

    The purge-and-raise runs *above* the notice gate and reads no configured
    value. No notice-shaped spec would catch a regression here: strip the text
    and every notice spec above simply stops asserting anything, while a run
    that can no longer be stopped is invisible.
    """

    @pytest.mark.parametrize(
        ("prefix", "closing"),
        [
            pytest.param("SENTINEL PREFIX", "SENTINEL CLOSING", id="sentinel"),
            pytest.param("", "", id="empty"),
        ],
    )
    async def test_a_pending_cancel_is_purged_and_raised_whatever_the_text(
        self, prefix: str, closing: str
    ) -> None:
        cancel = CancelMessage()
        mailbox = _MailboxDouble([_pending_message("hello"), cancel])
        capability = MailboxCapability(
            observer=mailbox, card=MailboxTool(absorbed_prefix=prefix, arrival_closing=closing)
        )

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

        assert mailbox.consume_calls == [[cancel.id]]
        assert cancel not in mailbox.pending


# =============================================================================
# FR4d — the run-end withdrawal (#123)
# =============================================================================


class TestRunEndWithdrawal:
    """``after_node_run`` — the notice is withdrawn once the run has ended.

    Hook arithmetic only: which entries leave the queue, and which stay. That
    the withdrawal actually **defeats** pydantic-ai's end-of-run redirect is a
    separate claim, and one these specs cannot make — they drive the hook by
    hand, with no drain in the picture. It is pinned by the real-chain specs in
    ``test_arrival_notice_durability.py``, which assert on the value ``act()``
    returns.
    """

    async def test_the_notice_is_withdrawn_when_the_run_has_ended(self) -> None:
        """AC-1, AC-3 — the entry the hook enqueued leaves; the result does not change."""
        arrived = _pending_message("news", "@Alice")
        capability = MailboxCapability(observer=_MailboxDouble([arrived]), card=MailboxTool())
        ctx = _QueueCtxDouble()
        await capability.before_model_request(ctx, _context())
        assert len(ctx.pending_messages) == 1

        result = _end()
        returned = await capability.after_node_run(
            ctx,  # type: ignore[arg-type]
            node=_MID_RUN_NODE,  # type: ignore[arg-type]
            result=result,
        )

        # Never converted, never created, never redirected.
        assert returned is result
        assert ctx.pending_messages == []

    async def test_a_result_that_is_not_an_end_leaves_the_queue_untouched(self) -> None:
        """AC-2 — the doorbell still rings mid-run; only the run's end withdraws.

        MUTATION — drop the ``isinstance(result, End)`` condition and the notice
        is withdrawn at every node boundary, so this spec goes red on a queue
        that has been emptied one step after it was filled. The real-chain
        ``test_notice_lands_in_durable_history_and_event_stream_exactly_once``
        goes red with it; nothing else in the suite does.
        """
        arrived = _pending_message("news", "@Alice")
        capability = MailboxCapability(observer=_MailboxDouble([arrived]), card=MailboxTool())
        ctx = _QueueCtxDouble()
        await capability.before_model_request(ctx, _context())
        queued = list(ctx.pending_messages)

        await capability.after_node_run(
            ctx,  # type: ignore[arg-type]
            node=_MID_RUN_NODE,  # type: ignore[arg-type]
            result=_MID_RUN_NODE,  # type: ignore[arg-type]
        )

        assert ctx.pending_messages == queued

    async def test_another_producers_entry_is_left_in_the_queue(self) -> None:
        """AC-5 — withdrawal is ours alone, so the drain still redirects for the rest.

        Anything this capability did not enqueue is none of its business: the
        redirect is right for content with no other delivery path, and only the
        arrival notice has one.
        """
        arrived = _pending_message("news", "@Alice")
        capability = MailboxCapability(observer=_MailboxDouble([arrived]), card=MailboxTool())
        ctx = _QueueCtxDouble()
        await capability.before_model_request(ctx, _context())
        ctx.enqueue("a note from somewhere else entirely", priority="asap")
        foreign = ctx.pending_messages[-1]

        await capability.after_node_run(
            ctx,  # type: ignore[arg-type]
            node=_MID_RUN_NODE,  # type: ignore[arg-type]
            result=_end(),
        )

        assert ctx.pending_messages == [foreign]

    async def test_an_absorbed_messages_rendering_is_never_withdrawn(self) -> None:
        """AC-4 — an **invariant**, not a path the graph can reach today.

        A step that called a tool returns a ``ModelRequestNode`` and never an
        ``End``, so content ``after_tool_execute`` enqueued is always drained
        normally and this arrangement — an absorbed rendering still queued at an
        ``End`` — is one no run produces. It is pinned anyway, because the code
        that guarantees it is not the code that will be edited next, and the cost
        of losing it is total: the message was **already consumed** from the
        mailbox by that hook, so the queue is the only thing still holding it.
        Withdrawing it would lose it outright, with nothing left to deliver it as
        its own turn — where a withdrawn *notice* costs only an announcement of
        mail that is still sitting in the mailbox.

        MUTATION — record ``after_tool_execute``'s ``ctx.enqueue`` return into
        ``_notice_enqueue_ids`` (the one-line widening a future edit would most
        plausibly make) and this spec goes red on an empty queue. It is the only
        spec in the suite that fails on that mutation.
        """
        absorbed = _pending_message("the body of the absorbed message", "@Alice")
        handled = _pending_message("the turn prompt", "@Human")
        capability = MailboxCapability(
            observer=_MailboxDouble([absorbed], current=handled), card=MailboxTool()
        )
        ctx = _QueueCtxDouble()

        # The notice goes out first, exactly as a run would produce it...
        await capability.before_model_request(ctx, _context())
        # ...then the model names the id, and this hook absorbs and enqueues it.
        await capability.after_tool_execute(
            ctx,  # type: ignore[arg-type]
            call=ToolCallPart(
                tool_name=READ_MAILBOX_TOOL,
                args={MESSAGE_ID_ARG: str(absorbed.id)},
                tool_call_id="read-1",
            ),
            tool_def=MagicMock(),
            args={MESSAGE_ID_ARG: str(absorbed.id)},
            result="Acknowledged.",
        )
        assert len(ctx.pending_messages) == 2

        await capability.after_node_run(
            ctx,  # type: ignore[arg-type]
            node=_MID_RUN_NODE,  # type: ignore[arg-type]
            result=_end(),
        )

        # The absorbed entry survives, carrying the message's own rendering
        # inside the added-work framing the injection wraps it in.
        remaining = [_queued_text(pending) for pending in ctx.pending_messages]
        assert len(remaining) == 1
        assert absorbed.rendering() in remaining[0]

    async def test_before_run_forgets_which_notices_it_could_withdraw(self) -> None:
        """AC-6 — the tracking set is run-local, so a stale id withdraws nothing.

        Contrived on purpose: a queue does not really outlive the run that filled
        it. The claim is about the *set*, and this is the only way to state it
        behaviourally — an id recorded in one run must be unable to reach into
        another, whatever ends up in front of it.
        """
        arrived = _pending_message("news", "@Alice")
        capability = MailboxCapability(observer=_MailboxDouble([arrived]), card=MailboxTool())
        ctx = _QueueCtxDouble()
        await capability.before_model_request(ctx, _context())
        from_the_previous_run = ctx.pending_messages[0]

        await capability.before_run(ctx)  # type: ignore[arg-type]
        assert capability._notice_enqueue_ids == set()

        await capability.after_node_run(
            ctx,  # type: ignore[arg-type]
            node=_MID_RUN_NODE,  # type: ignore[arg-type]
            result=_end(),
        )

        assert ctx.pending_messages == [from_the_previous_run]


# =============================================================================
# HELPERS — actor level
# =============================================================================


class _InterruptibleReactAgent:
    """ReactAgent double: records wiring; ``run_sync`` raises on demand.

    ``interrupts_remaining`` simulates the mailbox capability raising
    ``RunInterruptedError`` out of ``run_sync`` mid-run — the doubles never
    drive pydantic-ai's capability chain, so the raise is injected here.
    """

    captured: ClassVar[list[dict[str, Any]]] = []
    recorded_blocks: ClassVar[list[str]] = []
    run_calls: ClassVar[int] = 0
    interrupts_remaining: ClassVar[int] = 0

    def __init__(self, **kwargs: Any) -> None:
        # Kept on the instance too: the real-hook subclass below needs *its own*
        # observer and capability, not whichever agent was constructed last.
        self.kwargs = kwargs
        _InterruptibleReactAgent.captured.append(kwargs)
        self.context = SimpleNamespace(
            append_user_prompt=_InterruptibleReactAgent.recorded_blocks.append, messages=[]
        )

    def system_prompt(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    def run_sync(self, prompt: object, **kwargs: object) -> StructuredOutput:
        # The counters live on this class by name, never on ``type(self)``: a
        # subclass assigning through ``type(self)`` would shadow them and the
        # specs would read a stale base value.
        _InterruptibleReactAgent.run_calls += 1
        if _InterruptibleReactAgent.interrupts_remaining > 0:
            _InterruptibleReactAgent.interrupts_remaining -= 1
            raise RunInterruptedError("cancelled by a queued /stop or CancelMessage")
        return StructuredOutput(messages=[])

    def close(self) -> None:
        pass


class _RealHookReactAgent(_InterruptibleReactAgent):
    """Same double, except the raise comes from the REAL hook and REAL inbox.

    ``_InterruptibleReactAgent`` *simulates* the interruption, so the capability
    never runs and nothing is ever purged — useless for proving a purge. Here
    the first run instead waits for the cancel to land in the actor's own inbox
    and then fires ``before_model_request`` exactly as a step boundary would.
    The hook purges through core's real ``consume_mailbox`` against the real
    inbox and raises out of ``run_sync``, which ``act()`` absorbs unchanged.

    Everything it needs comes off the wiring ``BaseAgent`` already passes —
    ``observer=`` is the agent, ``capabilities=`` holds its ``MailboxCapability``
    — so no actor internal is touched.
    """

    def run_sync(self, prompt: object, **kwargs: object) -> StructuredOutput:
        _InterruptibleReactAgent.run_calls += 1
        if _InterruptibleReactAgent.interrupts_remaining > 0:
            _InterruptibleReactAgent.interrupts_remaining -= 1
            observer = self.kwargs["observer"]
            capability = self.kwargs["capabilities"][0]
            assert _wait_until(
                lambda: any(is_cancel(message) for message in observer.get_mailbox())
            ), "the cancel never reached the inbox — this is not the mid-run case"
            asyncio.run(capability.before_model_request(_CtxDouble(), _context()))
            raise AssertionError("the real hook did not raise on a pending cancel")
        return StructuredOutput(messages=[])


def _reset_captures(interrupts: int = 0) -> None:
    _InterruptibleReactAgent.captured = []
    _InterruptibleReactAgent.recorded_blocks = []
    _InterruptibleReactAgent.run_calls = 0
    _InterruptibleReactAgent.interrupts_remaining = interrupts


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(template="You are a manager."),
        model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
    )


@contextmanager
def _running_agent(
    interrupts: int = 0,
    agent_class: type[BaseAgent] = BaseAgent,
    react_agent_class: type[_InterruptibleReactAgent] = _InterruptibleReactAgent,
) -> Iterator[tuple[ActorSystem, ActorAddress, ActorAddress]]:
    """Run an agent through the real actor system with the interruptible double.

    ``agent_class`` defaults to ``BaseAgent``; ``CustomAgent`` is passed in to
    exercise a *subclass* handler, which is the shape the exemplar has.
    ``react_agent_class`` defaults to the simulated raise; ``_RealHookReactAgent``
    is passed in when the raise must come from the real hook and real inbox.
    """
    _reset_captures(interrupts)
    system = ActorSystem()
    original_react = agent_module.ReactAgent
    agent_module.ReactAgent = react_agent_class  # type: ignore[misc, assignment]
    try:
        orch_addr = system.createActor(
            Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
        )
        orchestrator = system.proxy_ask(orch_addr, Orchestrator)
        agent_addr = orchestrator.createActor(agent_class, config=_agent_config())
        time.sleep(0.5)
        assert _InterruptibleReactAgent.captured, "ReactAgent was never constructed"
        yield system, agent_addr, orch_addr
    finally:
        agent_module.ReactAgent = original_react  # type: ignore[misc]
        try:
            system.shutdown(timeout=5)
        except Exception:
            pass


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


# =============================================================================
# FR4 wiring — self._capabilities is the one list both branches receive
# =============================================================================


class _RecordingCapability(AbstractCapability[Any]):
    """A custom capability that records every firing and changes nothing.

    Stands in for what a consumer would actually contribute — an observability
    wrapper, a domain guard — reduced to the part the specs can observe. It
    returns ``request_context`` unchanged, so it cannot perturb the run it is
    added to.
    """

    def __init__(self) -> None:
        self.firings = 0

    async def before_model_request(
        self, ctx: Any, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.firings += 1
        return request_context


class _AgentWithOneExtra(BaseAgent):
    """A subclass contributing exactly one capability of its own.

    ``extra_capabilities`` reads nothing off ``self`` — the wiring specs build
    the agent with ``object.__new__``, so ``config``, ``state`` and
    ``_command_registry`` do not exist when ``_build_react_agent`` calls it.
    """

    extra: ClassVar[_RecordingCapability] = _RecordingCapability()

    def extra_capabilities(self) -> list[AgentCapability[Any]]:
        return [self.extra]


class TestCapabilityWiring:
    def test_bare_base_agent_yields_exactly_the_mailbox_capability(self) -> None:
        """A subclass that overrides nothing gets ``[mailbox]`` and nothing else."""
        with _running_agent():
            capabilities = _InterruptibleReactAgent.captured[-1]["capabilities"]
            assert len(capabilities) == 1
            assert isinstance(capabilities[0], MailboxCapability)

    def test_base_agent_extra_capabilities_defaults_to_empty(self) -> None:
        """The hook is safe on a half-built agent — it reads nothing off self."""
        agent: BaseAgent = object.__new__(BaseAgent)
        assert agent.extra_capabilities() == []

    def test_assembly_is_mailbox_first_then_the_subclass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order is mailbox-first: the cancel check precedes any custom work.

        Asserted at ``_assemble_capabilities``, which is where the order is
        decided. ``_build_react_agent`` only forwards what it is handed, so
        asserting order there would pin the wrong method.
        """
        extra = _RecordingCapability()
        monkeypatch.setattr(_AgentWithOneExtra, "extra", extra)
        agent: _AgentWithOneExtra = object.__new__(_AgentWithOneExtra)

        capabilities = agent._assemble_capabilities(MailboxTool())

        assert capabilities == [agent._mailbox_capability, extra]
        assert isinstance(agent._mailbox_capability, MailboxCapability)

    def test_real_branch_forwards_a_copy_of_the_list_it_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AKGENTIC_MOCK_SCENARIO", raising=False)
        captured: dict[str, object] = {}

        class _FakeReactAgent:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(agent_module, "ReactAgent", _FakeReactAgent)
        agent: BaseAgent = object.__new__(BaseAgent)
        given: list[AgentCapability[Any]] = [_RecordingCapability()]

        agent._build_react_agent(ReactAgentConfig(), given, [], [])

        capabilities = captured["capabilities"]
        # A copy, not the caller's own list: pydantic-ai injects its
        # auto-capabilities into whatever it is handed, and only its own
        # (undocumented) copy keeps that off the agent's list today.
        assert capabilities == given
        assert capabilities is not given

    def test_mock_branch_forwards_a_copy_of_the_list_it_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKGENTIC_MOCK_SCENARIO", "/tmp/sandpile-research.yaml")
        captured: dict[str, object] = {}

        class _FakeMockReactAgent:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        fake_loadtest = types.ModuleType("akgentic.llm.loadtest")
        fake_loadtest.MockReactAgent = _FakeMockReactAgent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "akgentic.llm.loadtest", fake_loadtest)
        agent: BaseAgent = object.__new__(BaseAgent)
        given: list[AgentCapability[Any]] = [_RecordingCapability()]

        agent._build_react_agent(ReactAgentConfig(), given, [], [])

        capabilities = captured["capabilities"]
        # A copy here too — drop-in parity extends to the copy.
        assert capabilities == given
        assert capabilities is not given


# =============================================================================
# FR5 — the catch site: the run dies, the agent survives
# =============================================================================


class TestCatchSite:
    def test_interrupted_turn_notifies_human_and_routes_nothing(self) -> None:
        """The human is told once, and nothing is delivered.

        Since ``act()`` owns the interruption, ``_route_output`` **is** called —
        with the default, empty ``StructuredOutput`` ``act()`` hands back. So
        "routes nothing" is measured as *nothing delivered*, never as the router
        being skipped: an assertion that it was skipped would pin the old design.
        """
        with (
            patch.object(BaseAgent, "notify_human") as notify,
            patch.object(BaseAgent, "_route_output") as route,
            _running_agent(interrupts=1) as (system, agent_addr, _),
        ):
            system.tell(agent_addr, AgentMessage(content="do the thing", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 1)
            assert _wait_until(lambda: notify.call_count >= 1)
            notify.assert_called_once_with("Run interrupted.")

            assert _wait_until(lambda: route.call_count >= 1)
            (routed,), _ = route.call_args
            assert isinstance(routed, StructuredOutput)
            assert routed.messages == []

    def test_agent_survives_and_processes_the_next_queued_message(self) -> None:
        """The actor-death guard — NFR2's mutation target.

        Without the catch, ``RunInterruptedError`` escapes the handler into
        the actor failure path (``Akgent._handle_failure`` — an ErrorMessage
        to the orchestrator; actor death under stock pykka, whose
        ``_handle_failure`` stops the actor). The spec pins both halves of
        surviving *cleanly*: the next queued message is processed normally,
        and the failure path never ran.
        """
        with (
            patch.object(BaseAgent, "notify_human"),
            patch.object(BaseAgent, "_handle_failure") as failure,
            _running_agent(interrupts=1) as (system, agent_addr, _),
        ):
            system.tell(agent_addr, AgentMessage(content="first — interrupted", type="request"))
            system.tell(agent_addr, AgentMessage(content="second — normal", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 2), (
                "the agent did not survive the interruption: the second queued "
                "message was never processed"
            )
            failure.assert_not_called()

    def test_no_user_proxy_branch_logs_instead_of_delivering(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch.object(BaseAgent, "get_team", return_value=[]),
            _running_agent(interrupts=1) as (system, agent_addr, _),
            caplog.at_level(logging.WARNING, logger=AGENT_LOGGER),
        ):
            system.tell(agent_addr, AgentMessage(content="do the thing", type="request"))

            assert _wait_until(
                lambda: any("not delivered" in record.getMessage() for record in caplog.records)
            )


# =============================================================================
# ADR-019 §3 — the recognised cancel is purged at recognition
# =============================================================================


class TestPurgeAtRecognition:
    """A cancel the hook acts on leaves the mailbox in the same act.

    Two layers. The hook-level specs pin what the hook asks for and what
    survives it; the actor-level one drives the real chain — real inbox, real
    ``consume_mailbox``, real orchestrator — and pins the consequence the whole
    story exists for: a mid-run cancel never gets a turn.
    """

    async def test_the_recognised_cancel_is_gone_from_the_mailbox(self) -> None:
        stop = _pending_message("/stop")
        mailbox = _MailboxDouble([stop])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

        assert mailbox.get_mailbox() == []
        assert mailbox.consume_calls == [[stop.id]]

    async def test_non_cancel_mail_beside_it_survives_the_purge(self) -> None:
        """Only cancels go — the rest is still owed its own turn."""
        keep = _pending_message("hello", "@Alice")
        stop = _pending_message("/stop now", "@Bob")
        mailbox = _MailboxDouble([keep, stop])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

        assert mailbox.get_mailbox() == [keep]
        assert mailbox.consume_calls == [[stop.id]]

    async def test_every_pending_cancel_is_purged_not_only_the_first(self) -> None:
        first = _pending_message("/stop", "@Alice")
        keep = _pending_message("hello", "@Bob")
        second = CancelMessage(reason="and again")
        mailbox = _MailboxDouble([first, keep, second])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

        assert mailbox.consume_calls == [[first.id, second.id]]
        assert mailbox.get_mailbox() == [keep]

    async def test_the_run_still_dies_when_the_purge_removed_nothing(self) -> None:
        """An empty return means "already gone", never "no cancel".

        ``consume_mailbox`` ignores ids that are no longer queued, so a hook
        that branched on its return would let a recognised cancel through
        whenever the actor happened to win the race.
        """

        class _AlreadyGoneMailbox(_MailboxDouble):
            def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Any]:
                self.consume_calls.append(list(message_ids))
                return []

        mailbox = _AlreadyGoneMailbox([_pending_message("/stop")])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

        assert mailbox.consume_calls != []

    def test_a_mid_run_stop_never_gets_a_turn(self) -> None:
        """The whole story, end to end, through the real chain.

        Three messages are queued back to back. The agent is inside the first
        turn when the ``/stop`` lands, which makes it the mid-run case by
        construction. The hook purges it through core's real ``consume_mailbox``
        against the real inbox, so it is never dequeued: no command dispatch, no
        operator-action entry, no run of its own. One ``HandledMessage`` names
        it in the orchestrator's log, and the third message runs normally.
        """
        with (
            patch.object(BaseAgent, "notify_human"),
            patch.object(BaseAgent, "_handle_failure") as failure,
            _running_agent(interrupts=1, react_agent_class=_RealHookReactAgent) as (
                system,
                agent_addr,
                orch_addr,
            ),
        ):
            stop = AgentMessage(content="/stop", type="request")
            system.tell(agent_addr, AgentMessage(content="first — interrupted", type="request"))
            system.tell(agent_addr, stop)
            system.tell(agent_addr, AgentMessage(content="second — normal", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 2), (
                "the queued message after the purged cancel was never processed"
            )
            orchestrator = system.proxy_ask(orch_addr, Orchestrator)

            def _handled() -> list[Any]:
                return list(orchestrator.get_messages(message_type=HandledMessage))

            assert _wait_until(lambda: bool(_handled()))

            # It never reached command dispatch: a dispatched /stop would have
            # recorded an operator action, and it never reached the LLM either —
            # two runs for three messages is the missing turn.
            assert not any(
                'The human ran "/stop"' in block
                for block in _InterruptibleReactAgent.recorded_blocks
            )
            assert _InterruptibleReactAgent.run_calls == 2
            failure.assert_not_called()

            # Exactly one HandledMessage, naming the stop and nothing else.
            assert [message.message_id for message in _handled()] == [stop.id]


# =============================================================================
# A subclass handler that writes NO catch survives identically
# =============================================================================


class TestNoCatchSubclassSurvives:
    """The regression the shipped exemplar failed, pinned at actor level.

    ``CustomAgent.receiveMsg_TriageMessage`` carries no ``try``/``except`` — it
    writes nothing at all for cancellation, because ``act()`` owns it. Driven
    through the real ``act()`` with a ``run_sync`` that raises, the turn must
    still end the designed way: the human told once, nothing delivered, the
    handler returning normally, and ``Akgent._handle_failure`` never entered.

    This is the mutation target for the catch in ``act()``: let the
    interruption propagate out of ``act()`` instead, and this spec goes red
    because the subclass has nothing left to catch it.
    """

    def test_interrupted_subclass_turn_ends_cleanly_with_no_handler_catch(self) -> None:
        with (
            patch.object(BaseAgent, "notify_human") as notify,
            patch.object(BaseAgent, "_handle_failure") as failure,
            patch.object(CustomAgent, "_route_triage") as route,
            _running_agent(interrupts=1, agent_class=CustomAgent) as (system, agent_addr, _),
        ):
            system.tell(agent_addr, TriageMessage(incident="disk full on node 3"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 1)
            assert _wait_until(lambda: notify.call_count >= 1)
            notify.assert_called_once_with("Run interrupted.")

            # The handler ran to its end and routed the default output act()
            # returned — an empty triage, so nothing goes out.
            assert _wait_until(lambda: route.call_count >= 1)
            (routed,), _ = route.call_args
            assert isinstance(routed, TriageOutput)
            assert routed.handoffs == []

            failure.assert_not_called()


# =============================================================================
# FR6 — receiveMsg_CancelMessage: an idle cancel is a logged no-op
# =============================================================================


class TestIdleCancel:
    def test_idle_cancel_message_is_a_logged_noop(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            _running_agent() as (system, agent_addr, _),
            caplog.at_level(logging.INFO, logger=AGENT_LOGGER),
        ):
            system.tell(agent_addr, CancelMessage(reason="operator changed their mind"))

            assert _wait_until(
                lambda: any("nothing to cancel" in record.getMessage() for record in caplog.records)
            )
            assert _InterruptibleReactAgent.run_calls == 0

    def test_idle_cancel_does_not_poison_the_next_run(self) -> None:
        with _running_agent() as (system, agent_addr, _):
            system.tell(agent_addr, CancelMessage())
            system.tell(agent_addr, AgentMessage(content="carry on", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 1)
            assert _InterruptibleReactAgent.interrupts_remaining == 0


# =============================================================================
# FR8 — an agent with NO MailboxTool is still interruptible
# =============================================================================


def _make_cardless_agent(pending: list[Any]) -> BaseAgent:
    """A BaseAgent assembled with no tool cards at all.

    ``object.__new__`` means ``on_start`` never runs, so nothing is auto-added —
    this really is an agent whose config carries no ``MailboxTool``. The cancel
    capability observes the agent itself, exactly as ``_build_react_agent``
    wires it in production; ``get_mailbox`` is core's own method, which is why
    it survives the absence of the card. The pending list stands in for the
    actor inbox, which ``object.__new__`` leaves unbuilt.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@CardlessAgent"
    # Explicit: a bare MagicMock(spec=AgentConfig) would answer *any* attribute,
    # so the card-less precondition has to be set, not assumed.
    mock_config.tools = []
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_mailbox = MagicMock(return_value=pending)  # type: ignore[method-assign]
    # ``object.__new__`` leaves no ``_actor_ref``, so core's real
    # ``consume_mailbox`` has no inbox to reach into. Stubbed rather than
    # dropped: the hook purges before it raises, so the cancel path calls it.
    agent.consume_mailbox = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent._mailbox_capability = MailboxCapability(observer=agent, card=MailboxTool())

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    return agent


class TestCardlessAgentStillCancels:
    """The requirement that forced the vocabulary out of the card.

    ``MailboxCapability`` is built unconditionally so that an agent
    configured without ``MailboxTool`` is still interruptible — and a
    ``CancelMessage`` from the frontend to such an agent must still kill the
    run. A predicate that shipped with the card could not serve this agent:
    there is no card here to import one from.
    """

    def test_pending_cancel_kills_the_run_of_an_agent_with_no_mailbox_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        agent = _make_cardless_agent([CancelMessage(reason="frontend Esc")])

        # The card-less precondition, asserted rather than relied upon.
        assert agent.config.tools == []
        assert agent._command_registry.has("stop") is False

        react_agent = ReactAgent(
            config=ReactAgentConfig(
                model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
            ),
            deps_type=BaseAgent,
            capabilities=[agent._mailbox_capability],
        )
        agent._react_agent = react_agent  # type: ignore[attr-defined]

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise AssertionError("the model was reached: the cancel never fired")

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                # The cancel still kills the run — the stub proves the model was
                # never reached — but act() absorbs the interruption and hands
                # back the default output instead of raising it at the caller.
                output = agent.act(AgentMessage(content="do the long thing"), StructuredOutput)
        finally:
            react_agent.close()

        assert isinstance(output, StructuredOutput)
        assert output.messages == []


# =============================================================================
# The extension point works — a subclass's own capability actually fires
# =============================================================================


def _make_extension_point_agent() -> _AgentWithOneExtra:
    """An ``_AgentWithOneExtra`` assembled far enough to survive one real run.

    Same ``object.__new__`` shape as ``_make_cardless_agent``, but the
    capability list is **not** written by hand: ``_assemble_capabilities`` builds
    it, which is the whole point — the spec must exercise the framework's own
    assembly, not a list the test wrote.
    """
    agent: _AgentWithOneExtra = object.__new__(_AgentWithOneExtra)

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@ExtensionPointAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    # No mail: the run must reach the model, so the cancel check must not fire.
    agent.get_mailbox = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    agent.notify_event = MagicMock()  # type: ignore[method-assign]
    return agent


class TestExtraCapabilityFires:
    """The AC that proves the extension point works rather than merely assembling.

    Everything else pins the *list*. This drives a real ``ReactAgent`` — built
    by ``_build_react_agent`` itself, over a ``FunctionModel`` so no network is
    touched — and observes the subclass's own hook having run.
    """

    def test_a_subclass_capability_runs_during_a_real_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.delenv("AKGENTIC_MOCK_SCENARIO", raising=False)
        extra = _RecordingCapability()
        monkeypatch.setattr(_AgentWithOneExtra, "extra", extra)

        agent = _make_extension_point_agent()
        agent._capabilities = agent._assemble_capabilities(MailboxTool())
        react_agent = agent._build_react_agent(
            ReactAgentConfig(
                model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
            ),
            agent._capabilities,
            [],
            [],
        )
        agent._react_agent = react_agent  # type: ignore[attr-defined]

        # The framework assembled it, mailbox first, with the subclass's own second.
        assert agent._capabilities == [agent._mailbox_capability, extra]
        assert extra.firings == 0

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={"messages": []},
                        tool_call_id="out-1",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                output = agent.act(AgentMessage(content="do the thing"), StructuredOutput)
        finally:
            react_agent.close()

        assert isinstance(output, StructuredOutput)
        assert extra.firings >= 1
