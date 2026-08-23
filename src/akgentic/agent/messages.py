"""Team message types, and the two contracts a message may satisfy.

``AgentMessage`` is the single message type used for agent-to-agent
communication. It is also the reference implementation of both Protocols
declared here.

**Rendering a message is the message's job.** ``act()`` needs a prompt, and the
object that knows how a message should read to a model is the message — not the
handler that happens to be running, and not a tool. The contract is therefore a
*method*, ``render_for_llm()``, and never a ``content`` field: keying it on a
field re-encodes the exact assumption that produced the defect this design
removes, where a mailbox read rendered every message as
``getattr(message, "content", "")`` and silently emptied every class that
declares its own fields instead.

Both Protocols live here, in ``akgentic-agent``, rather than in
``akgentic-core``. Core is the actor framework — messaging, orchestrator,
lifecycle — and knows nothing about models; ``render_for_llm()`` is an LLM
concern and would leak prompt composition into the actor layer. This package is
the glue that already depends on core, llm and tool, so the contract sits where
it is conceptually correct.

There is deliberately **no intermediate base class**: a class implements the
method directly, and previewability is a second, separate opt-in.
"""

from typing import Literal, Protocol, runtime_checkable

from akgentic.agent.output_models import REPLY_PROTOCOLS
from akgentic.core.messages import Message

PREVIEW_LIMIT = 120
"""How much of a message's content the arrival notice previews, in characters."""


@runtime_checkable
class LlmRenderable(Protocol):
    """A message that can render itself as a prompt for the model.

    Exactly one member, and it is a method. ``act()`` accepts this and nothing
    else — there is no string path, so framing cannot be bypassed by a caller
    that composes its own prompt.

    ``@runtime_checkable`` checks member *presence*, never signatures, so
    conformance is guarded by a real test rather than assumed.
    """

    def render_for_llm(self) -> str:
        """The message as the model should read it, framing included."""
        ...


@runtime_checkable
class MailboxPreviewable(Protocol):
    """A message that can be summarised in a mid-run arrival notice.

    Separate from :class:`LlmRenderable` on purpose: rendering a message once it
    has been taken on is a different question from advertising it while another
    run is in flight. Previewability is the discriminator — a class that offers
    no preview is a class that cannot be read mid-run, and the arrival notice
    lists it without an id rather than offering an affordance that would fail.
    """

    def mailbox_preview(self) -> str:
        """One line summarising this message for the arrival notice."""
        ...


class AgentMessage(Message):
    """Base message type for team communication.

    Implements both Protocols directly. The reply-protocol prefix is a property
    of a *typed* message, so it belongs where ``type`` is declared; the table it
    reads keeps its single home in ``akgentic.agent.output_models``.

    Attributes:
        type: The message's intent, which selects its reply protocol.
        content: The message text content.
    """

    type: Literal[
        "request",
        "response",
        "notification",
        "instruction",
        "acknowledgment",
    ] = "request"
    content: str

    def render_for_llm(self) -> str:
        """The reply-protocol prefix for this message's type, then its body.

        The prefix names who wrote, what kind of message it is, and whether a
        reply is expected — message mechanics only, never team policy, which
        belongs in the agents' prompts where it can differ per role.
        """
        sender_name = self.sender.name if self.sender else "unknown"
        article = "an" if self.type[0] in "aeiou" else "a"
        return (
            f"You received {article} {self.type} from {sender_name}. "
            f"{REPLY_PROTOCOLS.get(self.type, '').format(sender=sender_name)}"
            f"\n\n{self.content}"
        )

    def mailbox_preview(self) -> str:
        """``sender (type): content`` — content collapsed and cut at the limit.

        No trailing colon when the content is empty: the line then carries the
        sender and type alone rather than an empty quotation.
        """
        sender_name = self.sender.name if self.sender else ""
        head = f"{sender_name or 'unknown'} ({self.type})"
        content = " ".join(self.content.split())
        if len(content) > PREVIEW_LIMIT:
            content = f"{content[:PREVIEW_LIMIT]}…"
        return f"{head}: {content}" if content else head
