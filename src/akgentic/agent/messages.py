r"""Team message types.

``AgentMessage`` is the single message type used for agent-to-agent
communication, and the reference implementation of both rendering contracts
declared on ``akgentic.core.messages.Message``.

**Both renderings come from ``MailboxMessage``**, the base ``AgentMessage``
extends, which is why this module declares neither contract. ``rendering()`` is
the message as prose; ``rendering_preview()`` is the message as one summary
line. Both default to ``None`` there, and ``AgentMessage`` overrides both. In
this package the reader is a model, so what ``rendering()`` returns is a prompt
— a fact about the *consumer*, not about the method.

They were a pair of one-method ``Protocol``\ s first, which forced the same
contract to be declared **twice**: once here for ``act()``, once in
``akgentic-tool`` for the mailbox capability, because neither package may import
the other. A base class in the package they both already depend on is declared
once and inherited, and it makes the ``isinstance`` gymnastics redundant.
``akgentic-core`` was the other candidate and was rejected: prose rendering is
not the actor framework's business.

There is deliberately **no intermediate base class and no marker Protocol**: a
class overrides the method directly, and previewability is a second, separate
opt-in from rendering.
"""

from typing import Literal, Protocol, runtime_checkable

from akgentic.agent.output_models import REPLY_PROTOCOLS
from akgentic.tool.mailbox import PREVIEW_LIMIT, MailboxMessage

__all__ = ["AgentMessage", "LlmRenderable"]


@runtime_checkable
class LlmRenderable(Protocol):
    """Anything ``act()`` can turn into a prompt: one method, ``rendering()``.

    **Structural, and a strict subset of ``MailboxMessage``.** ``act()`` needs a
    prompt and nothing else — not a mailbox, not a preview, not an id — so this
    is what it accepts. Every ``MailboxMessage`` satisfies it for free, and so
    does a class that simply declares ``rendering()`` without joining the mailbox
    at all: ``TriageMessage`` in ``custom_agent.py`` is the worked example. An
    agent configured with no ``MailboxTool`` still runs, because nothing here
    reaches for one.

    This is **not** a second declaration of the mailbox contract. That one is
    nominal and lives on ``MailboxMessage`` in ``akgentic-tool``, where the
    capability tests it with ``isinstance`` to decide what may be absorbed
    mid-run. This is a view of one method of it, held by the package that owns
    ``act()``. The two agree because the method name is the same — pinned by a
    spec, not by care.

    **Keyed on the method, never on a ``content`` field.** A field-keyed contract
    is what let one consumer render every message as
    ``getattr(message, "content", "")``: correct for the single class it was
    written against, silently empty for every class that declares its own fields.
    Accepting ``content`` here as a fallback would reintroduce exactly that.
    """

    def rendering(self) -> str:
        """This message as the prompt a model should read."""
        ...


class AgentMessage(MailboxMessage):
    """Base message type for team communication.

    Overrides both of ``Message``'s rendering methods. The reply-protocol prefix
    is a property
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

    def rendering(self) -> str:
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

    def rendering_preview(self) -> str:
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
