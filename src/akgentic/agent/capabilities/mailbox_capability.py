"""Mailbox-driven run control: the vocabulary and the hook that acts on it.

The hook has more than one duty. It raises on a pending cancel, and it renders
and enqueues the mid-run arrival notice for mail that landed while the run was
in flight; under ADR-019 §3 it will also purge the recognised cancel from the
mailbox. The mailbox is the single input to all of them, which is what makes
them one capability rather than three.

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
this module imports nothing from ``akgentic.tool.mailbox`` for either of them.

What the card still owns is its own surface: the ``MailboxState`` provider,
the ``read_mailbox`` tool, and the ``/stop`` command registration — a string
surface ``is_cancel`` recognises without importing anything from the card.
"""

import uuid
from collections.abc import Iterable
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

from akgentic.core.messages import CancelMessage, Message

# The one symbol still taken from the tool package, and deliberately so: a
# structural Protocol for the ``observer`` type hint, satisfied by core's own
# ``Akgent.get_mailbox``. It carries no card dependency and does not weaken the
# ownership above — the vocabulary below is defined here, not imported.
from akgentic.tool.mailbox import MailboxToolObserver


class RunInterruptedError(Exception):
    """The current run was cancelled by a /stop or CancelMessage.

    Raised by ``MailboxCapability.before_model_request`` at the next step
    boundary once a cancel is pending in the mailbox (ADR-040 §5). Carries a
    message only. It must never escape a message handler: an escape ends the
    turn through the actor failure path (``Akgent._handle_failure`` — an
    ErrorMessage to the orchestrator; actor death under stock pykka) instead
    of the designed clean end. ``act()`` owns the catch — it absorbs this
    error, notifies the human once and returns a neutral instance of the
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
    """One-line doorbell for messages that arrived mid-run (ADR-040 §5).

    Returns ``""`` for an empty list. Defensive on message shapes: a message
    without a usable sender or content still renders (as ``unknown``).
    """
    if not new_messages:
        return ""
    count = len(new_messages)
    noun = "message" if count == 1 else "messages"
    senders = ", ".join(_unique_ordered(_sender_name(message) for message in new_messages))
    return (
        f"{count} new {noun} arrived (from {senders}) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )


def _sender_name(message: Message) -> str:
    """The sender's display name, or ``"unknown"`` when the message has none."""
    sender = getattr(message, "sender", None)
    name = getattr(sender, "name", None)
    return name if isinstance(name, str) and name else "unknown"


def _unique_ordered(values: Iterable[str]) -> list[str]:
    """Deduplicate while preserving first-seen order."""
    return list(dict.fromkeys(values))


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

    1. Cancel check — any pending message matching ``is_cancel`` raises
       ``RunInterruptedError``. The mailbox is the cancellation's single
       source of truth: no flag, no consumed marker — recognising the cancel
       and consuming it are the same dequeue, performed later by the actor
       loop.
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

    def __init__(self, observer: MailboxToolObserver) -> None:
        self._observer = observer
        self._announced_ids: set[uuid.UUID] = set()

    def reset_run_tracking(self) -> None:
        """Forget which arrivals this run announced (called at each run start)."""
        self._announced_ids.clear()

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Raise on a pending cancel, else enqueue an arrival notice for new mail."""
        pending = self._observer.get_mailbox()
        if any(is_cancel(message) for message in pending):
            raise RunInterruptedError(
                "The current run was cancelled by a queued /stop or CancelMessage."
            )
        new_messages = [m for m in pending if m.id not in self._announced_ids]
        if new_messages:
            ctx.enqueue(render_arrival_notice(new_messages), priority="asap")
            self._announced_ids.update(message.id for message in new_messages)
        return request_context
