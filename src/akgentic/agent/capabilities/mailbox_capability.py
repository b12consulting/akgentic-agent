"""Mailbox-driven run control: the vocabulary and the hook that acts on it.

The hook has more than one duty. It purges a pending cancel from the mailbox
and raises on it, and it renders and enqueues the mid-run arrival notice for
mail that landed while the run was in flight. The mailbox is the single input
to all of them, which is what makes them one capability rather than three.

This is the first member of ``akgentic.agent.capabilities`` — the home for
pydantic-ai capabilities the agent builds for itself. More are coming; a
capability that lives here is one the agent wires unconditionally, not one a
tool card contributes.

**Why the vocabulary lives here and not on the card.** ``is_cancel`` and
``render_arrival_notice`` used to be imported from ``akgentic.tool.mailbox``.
They are defined here instead, because ``BaseAgent`` builds
``MailboxCapability`` unconditionally — precisely so that an agent
configured with no ``MailboxTool`` at all is still interruptible, and a
``CancelMessage`` sent to such an agent still stops its run. A predicate that
ships with the card cannot serve that case: on a card-less agent there is no
card to import it from. Enforcement and vocabulary therefore sit together, and
this module imports nothing from ``akgentic.tool.mailbox`` at all — its mailbox
contract, ``MailboxAccess`` below, is the agent's own too, for the same reason.

What the card still owns is its own surface: the ``MailboxState`` provider,
the ``read_mailbox`` tool, and the ``/stop`` command registration — a string
surface ``is_cancel`` recognises without importing anything from the card.
"""

import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

from akgentic.core.messages import CancelMessage, Message

PREVIEW_LIMIT = 120
"""How much of a message's content the arrival notice previews, in characters."""


@runtime_checkable
class MailboxAccess(Protocol):
    """The mailbox contract this capability needs from the agent it observes.

    Declared here rather than borrowed from a card's observer protocol, for the
    same reason the vocabulary is: the capability is built unconditionally, so
    an agent carrying no ``MailboxTool`` must still satisfy it. ``Akgent``
    satisfies it structurally, through the two methods below, exactly as it
    satisfied the narrower card-side protocol this replaced.
    """

    def get_mailbox(self) -> list[Message]:
        """Peek at the pending messages, removing none of them."""
        ...

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
        """Remove the named messages from the mailbox, recording each removal."""
        ...


class RunInterruptedError(Exception):
    """The current run was cancelled by a /stop or CancelMessage.

    Raised by ``MailboxCapability.before_model_request`` at the next step
    boundary once a cancel is pending in the mailbox (ADR-040 §5). Carries a
    message only. It must never escape a message handler: an escape ends the
    turn through the actor failure path (``Akgent._handle_failure`` — an
    ErrorMessage to the orchestrator; actor death under stock pykka) instead
    of the designed clean end. ``act()`` owns the catch — it absorbs this
    error, notifies the human once and returns a default instance of the
    caller's output type — so the *run* dies while the *agent* carries on
    cleanly, and no handler needs a catch of its own.
    """


def is_cancel(msg: Message) -> bool:
    """Whether ``msg`` asks the recipient to abandon its current run.

    ``True`` for a ``CancelMessage`` instance, or for a message whose content
    strips to a string whose first whitespace-delimited token is exactly
    ``/stop`` — so ``"  /stop now"`` cancels and ``"/stopwatch"`` does not.
    A message without usable string content is simply ``False``.
    """
    if isinstance(msg, CancelMessage):
        return True
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        return False
    tokens = content.split(maxsplit=1)
    return bool(tokens) and tokens[0] == "/stop"


def render_arrival_notice(new_messages: list[Message]) -> str:
    """Doorbell for messages that arrived mid-run (ADR-040 §5, ADR-019 §4b).

    A count line, then one line per message carrying its sender, its type and a
    preview of its content cut at ``PREVIEW_LIMIT`` characters, then the pointer
    to ``read_mailbox``. Enough for the model to judge whether any of it is
    worth interrupting itself for; the whole of it stays one ``read_mailbox``
    call away, and arrives as its own turn regardless.

    Returns ``""`` for an empty list. Defensive on message shapes: the base
    ``Message`` declares neither ``content`` nor ``type`` — ``CancelMessage``
    carries ``reason``, ``AgentMessage`` carries both — so a message missing
    either, or carrying a non-string value in it, still renders.
    """
    if not new_messages:
        return ""
    count = len(new_messages)
    noun = "message" if count == 1 else "messages"
    lines = [f"{count} new {noun} arrived:"]
    lines.extend(_message_line(message) for message in new_messages)
    lines.append(
        "Call `read_mailbox` to see them entirely, or finish your current work "
        "first — you will get them just after."
    )
    return "\n".join(lines)


def _message_line(message: Message) -> str:
    """``- @Sender (type): preview`` for one message; no preview, no colon."""
    head = f"- {_sender_name(message)} ({_message_type(message)})"
    preview = _content_preview(message)
    return f"{head}: {preview}" if preview else head


def _sender_name(message: Message) -> str:
    """The sender's display name, or ``"unknown"`` when the message has none."""
    sender = getattr(message, "sender", None)
    name = getattr(sender, "name", None)
    return name if isinstance(name, str) and name else "unknown"


def _message_type(message: Message) -> str:
    """The message's declared type, or the bare ``"message"`` when it has none."""
    message_type = getattr(message, "type", None)
    return message_type if isinstance(message_type, str) and message_type else "message"


def _content_preview(message: Message) -> str:
    """The first ``PREVIEW_LIMIT`` characters of the content, ellipsised if cut.

    ``""`` when the message carries no string content at all — the line then
    renders as sender and type alone rather than as an empty quotation.
    """
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return ""
    content = " ".join(content.split())
    if len(content) <= PREVIEW_LIMIT:
        return content
    return f"{content[:PREVIEW_LIMIT]}…"


class MailboxCapability(AbstractCapability[Any]):
    """Mailbox-driven run cancellation and mid-run arrival notice (ADR-040 §5).

    One instance per agent, built unconditionally by ``BaseAgent`` — never
    contributed by a card, so cancellation cannot be de-configured by omitting
    ``MailboxTool``. The agent owns both the *vocabulary* (``is_cancel``,
    ``render_arrival_notice``, defined in this module) and the *enforcement*:
    an agent with no ``MailboxTool`` has no card to borrow a predicate from,
    and must still be interruptible.

    ``before_model_request`` fires before EVERY model request inside the REACT
    loop, bracketing every tool call and reasoning step — exactly the
    granularity cancellation needs. On each firing, in order:

    1. Cancel check — every pending message matching ``is_cancel`` is purged
       from the mailbox through ``consume_mailbox``, and then
       ``RunInterruptedError`` is raised. The mailbox is the cancellation's
       single source of truth: no flag, no consumed marker — recognising the
       cancel and consuming it are one act, performed here, at recognition.
       So a cancel never gets a turn of its own after interrupting a run, and
       the human hears about it once, through the interruption. A cancel that
       reaches a handler is by construction the *idle* case: nothing was
       running for this hook to have seen it.
    2. Arrival notice — pending messages not yet announced in this run are
       announced through one ``ctx.enqueue(notice, priority="asap")`` call,
       pydantic-ai's supported injection path. The auto-injected, outermost
       ``PendingMessageDrainCapability`` drains the queue at the *next* step
       boundary: the notice lands in that model request, in the durable
       history and in the ``LlmMessageEvent`` stream by design — the event
       store is the audit trail that the doorbell rang. When the run would
       otherwise end first, the drain redirects through one final model
       request so an already-enqueued notice is delivered rather than lost.
       The hook constructs no message of its own and never mutates an
       existing message's parts (they are shared with durable history).

    Announced-id tracking is run-local: the instance lives for the agent's
    lifetime, so ``act()`` resets the set at each run start. A backlog
    re-announced next run is acceptable; a leak of announced ids across runs
    is not.
    """

    def __init__(self, observer: MailboxAccess) -> None:
        self._observer = observer
        self._announced_ids: set[uuid.UUID] = set()

    def reset_run_tracking(self) -> None:
        """Forget which arrivals this run announced (called at each run start)."""
        self._announced_ids.clear()

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Purge and raise on a pending cancel, else announce mail that arrived.

        The purge runs *before* the raise, and its result is deliberately not
        branched on. ``consume_mailbox`` ignores ids that are no longer queued,
        so an empty return means "already gone", never "no cancel" — the
        recognition has already happened and the run must die either way. It
        also emits the ``HandledMessage`` per removal itself, which is why
        nothing is emitted here: the telemetry belongs to the primitive so that
        no call site can forget it, and none can double it.
        """
        pending = self._observer.get_mailbox()
        cancels = [message for message in pending if is_cancel(message)]
        if cancels:
            self._observer.consume_mailbox([message.id for message in cancels])
            raise RunInterruptedError(
                "The current run was cancelled by a queued /stop or CancelMessage."
            )
        new_messages = [m for m in pending if m.id not in self._announced_ids]
        if new_messages:
            ctx.enqueue(render_arrival_notice(new_messages), priority="asap")
            self._announced_ids.update(message.id for message in new_messages)
        return request_context
