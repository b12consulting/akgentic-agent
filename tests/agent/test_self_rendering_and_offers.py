"""Messages render themselves, and the mailbox offers ids (Epic 23).

Four subjects, in the order the design builds them:

1. **The contracts.** ``rendering()`` and ``rendering_preview()``, both
   ``@runtime_checkable``, both keyed on a *method*. The negative case — a class
   carrying ``content`` but no ``rendering`` — is the one that pins the
   method-not-field decision, because a field-based contract is exactly what
   produced the data loss this epic removes.
2. **The renderings.** ``AgentMessage.rendering()`` reproduces the prefix
   its handler used to build inline, byte-for-byte, for **every**
   ``REPLY_PROTOCOLS`` key — the proof that moving the framing carried no
   behaviour with it.
3. **The offer rule.** Four conditions, each pinned on its own, plus the
   internal-invariant guard that fires only when the filter itself is wrong.
4. **The injection.** A completed ``read_mailbox`` call is turned back into the
   named message's own rendering; anything it cannot resolve is a silent no-op.

``@runtime_checkable`` checks member *presence*, not signatures, so the
conformance specs here are what make the Protocols real. This codebase has a
recorded case of a Protocol whose conformance tests were inert while its fakes
drifted for months.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import akgentic.core.messages as core_messages
import pytest
from akgentic.core import ActorAddress
from akgentic.core.messages import CancelMessage, Message, UserMessage
from akgentic.tool.mailbox import (
    PREVIEW_LIMIT,
    MailboxMessage,
    MailboxRenderError,
    MailboxTool,
    render_arrival_notice,
)
from akgentic.tool.mailbox.capability import (
    ABSORBED_PREFIX,
    MESSAGE_ID_ARG,
    READ_MAILBOX_TOOL,
    UNOFFERABLE_LINE,
    MailboxCapability,
)

from akgentic.agent.custom_agent import TriageMessage
from akgentic.agent.messages import AgentMessage, LlmRenderable
from akgentic.agent.output_models import REPLY_PROTOCOLS

# =============================================================================
# HELPERS
# =============================================================================


def _address(name: str) -> ActorAddress:
    sender = MagicMock(spec=ActorAddress)
    sender.name = name
    return sender


def _agent_message(
    content: str = "hello", sender: str = "@Alice", type_: Any = "request"
) -> AgentMessage:
    message = AgentMessage(content=content, type=type_)
    message.sender = _address(sender)
    return message


class _MailboxDouble:
    """The whole ``MailboxAccess`` surface over a mutable pending list."""

    def __init__(
        self, pending: list[Message] | None = None, current: Message | None = None
    ) -> None:
        self.pending: list[Message] = list(pending or [])
        self.current = current
        self.consume_calls: list[list[uuid.UUID]] = []

    def get_mailbox(self) -> list[Message]:
        return list(self.pending)

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
        self.consume_calls.append(list(message_ids))
        wanted = set(message_ids)
        taken = [m for m in self.pending if m.id in wanted]
        self.pending = [m for m in self.pending if m.id not in wanted]
        return taken

    def current_message(self) -> Message | None:
        return self.current


class _CtxDouble:
    """Recording RunContext double: exposes ``enqueue`` and records every call."""

    def __init__(self) -> None:
        self.enqueue_calls: list[tuple[tuple[Any, ...], Any]] = []

    def enqueue(self, *content: Any, priority: Any = "asap") -> str:
        self.enqueue_calls.append((content, priority))
        return f"enqueue-{len(self.enqueue_calls)}"


def _tool_call(name: str) -> Any:
    """A completed tool call carrying only the name the hook filters on."""
    call = MagicMock()
    call.tool_name = name
    return call


async def _after_read(
    capability: MailboxCapability,
    ctx: _CtxDouble,
    args: dict[str, Any],
    tool: str = READ_MAILBOX_TOOL,
) -> Any:
    """Drive ``after_tool_execute`` for one completed call."""
    return await capability.after_tool_execute(
        ctx,  # type: ignore[arg-type]
        call=_tool_call(tool),
        tool_def=MagicMock(),
        args=args,
        result="Acknowledged.",
    )


# =============================================================================
# A. The two contracts (AC 1-4)
# =============================================================================


class TestRenderingIsKeyedOnTheMethod:
    """The contract is ``rendering()``. It is never a ``content`` field."""

    def test_the_base_promises_both_renderings(self) -> None:
        """Extending ``MailboxMessage`` is a declaration, not an offer.

        A subclass that forgets either method fails loudly here rather than
        silently. For ``rendering()`` that is data loss averted: the capability
        consumes a message it takes on, so a quiet ``None`` would delete it from
        the mailbox and deliver nothing. For ``rendering_preview()`` it keeps one
        rule with one mechanism — a message opts out of mid-run reads by not
        extending this class, never by declining a method.
        """

        class _SaysNothing(MailboxMessage):
            pass

        message = _SaysNothing()
        with pytest.raises(NotImplementedError):
            message.rendering()
        with pytest.raises(NotImplementedError):
            message.rendering_preview()

    def test_a_subclass_overriding_only_the_method_renders(self) -> None:
        class _RendersOnly(MailboxMessage):
            def rendering(self) -> str:
                return "I know how I should read."

        assert _RendersOnly().rendering() == "I know how I should read."

    def test_a_content_field_alone_does_not_render(self) -> None:
        """The whole decision, in one assertion.

        A field-keyed contract is what let a mailbox read render every message
        as ``getattr(message, "content", "")`` — correct for one class, silently
        empty for every other. Keying on the method is what makes a class that
        declares its own fields a first-class citizen rather than a casualty.
        """

        class _ContentOnly(MailboxMessage):
            content: str = "I have a body but no idea how to present it."

        with pytest.raises(NotImplementedError):
            _ContentOnly().rendering()

    def test_agent_message_extends_the_mailbox_base(self) -> None:
        assert AgentMessage.__mro__[1] is MailboxMessage
        assert issubclass(MailboxMessage, Message)


class TestThePreviewIsSeparate:
    """Previewability is a second opt-in, and it is the offer discriminator."""

    def test_agent_message_answers_both(self) -> None:
        message = _agent_message()
        assert message.rendering() is not None
        assert message.rendering_preview() is not None

    def test_triage_message_renders_without_joining_the_mailbox(self) -> None:
        """``act()`` needs a rendering; it does not need a mailbox.

        ``TriageMessage`` declares ``rendering()`` and nothing else, so it is a
        valid ``act()`` argument while never being absorbable mid-run — the
        arrival notice lists it with no id.
        """
        incident = TriageMessage(incident="disk full", reported_by="monitoring")
        assert incident.rendering() is not None
        assert isinstance(incident, LlmRenderable)
        assert not isinstance(incident, MailboxMessage)

    def test_messages_outside_the_mailbox_base_render_nothing(self) -> None:
        for message in (Message(), CancelMessage(reason="stop"), UserMessage(content="hi")):
            assert not isinstance(message, MailboxMessage), type(message).__name__


class TestTheContractLivesWithBothItsConsumers:
    """NFR3 still holds — ``akgentic-core`` knows nothing about readers.

    The two renderings are declared once, on ``MailboxMessage`` in
    ``akgentic-tool``: the package ``akgentic-agent`` (for ``act()``) and the
    mailbox capability (for the mid-run absorb) both already depend on. Declaring
    them as Protocols forced the same contract to be written twice, once per
    package, because neither may import the other.
    """

    def test_the_base_is_declared_by_the_mailbox_package(self) -> None:
        assert MailboxMessage.__module__ == "akgentic.tool.mailbox.message"

    def test_core_declares_neither_rendering(self) -> None:
        # The other candidate home, rejected: prose rendering is not the actor
        # framework's business, and a base Message that rendered would make it so.
        assert not hasattr(core_messages.Message, "rendering")
        assert not hasattr(core_messages.Message, "rendering_preview")

    def test_the_act_protocol_and_the_mailbox_base_name_the_same_method(self) -> None:
        # LlmRenderable is a structural view of one MailboxMessage method, held by
        # the package that owns act(). They agree only while the name matches.
        assert set(LlmRenderable.__protocol_attrs__) == {"rendering"}  # type: ignore[attr-defined]
        assert isinstance(_agent_message(), LlmRenderable)

    def test_the_duplicated_preview_protocols_are_gone(self) -> None:
        import akgentic.tool.mailbox as mailbox_package

        for name in ("MailboxPreviewable", "SelfRendering"):
            assert getattr(mailbox_package, name, None) is None, name


# =============================================================================
# B. AgentMessage renders itself, byte-for-byte (AC 5, 6)
# =============================================================================


class TestAgentMessageRendersTheOldPrefix:
    """NFR2: the prefix moved onto the message; its text did not change."""

    @pytest.mark.parametrize("message_type", sorted(REPLY_PROTOCOLS))
    def test_every_reply_protocol_renders_byte_for_byte(self, message_type: str) -> None:
        """The composition the handler used to build inline, for every key.

        Reproduced here as the literal it was, rather than by calling the
        renderer twice — a spec that re-derived the string through the code
        under test would agree with any rewrite of it.
        """
        message = _agent_message("the body", "@Manager", message_type)

        article = "an" if message_type[0] in "aeiou" else "a"
        protocol = REPLY_PROTOCOLS[message_type].format(sender="@Manager")
        expected = f"You received {article} {message_type} from @Manager. {protocol}\n\nthe body"

        assert message.rendering() == expected

    def test_a_senderless_message_renders_unknown(self) -> None:
        message = AgentMessage(content="orphan", type="request")
        assert message.sender is None
        assert "from unknown. " in message.rendering()

    def test_an_unknown_type_still_renders_the_body(self) -> None:
        """``REPLY_PROTOCOLS.get`` returns "" — the framing degrades, the body survives."""
        message = _agent_message("the body", "@Manager")
        object.__setattr__(message, "type", "telegram")
        rendered = message.rendering()
        assert rendered.startswith("You received a telegram from @Manager. ")
        assert rendered.endswith("\n\nthe body")


class TestAgentMessagePreview:
    """AC 6: the arrival-notice line body, minus the bullet."""

    def test_it_is_sender_type_and_content(self) -> None:
        assert _agent_message("ship it", "@Bob").rendering_preview() == "@Bob (request): ship it"

    def test_content_whitespace_is_collapsed(self) -> None:
        preview = _agent_message("two\n\n  words", "@Bob").rendering_preview()
        assert preview == "@Bob (request): two words"

    def test_a_long_body_is_cut_at_the_limit_with_an_ellipsis(self) -> None:
        preview = _agent_message("x" * 400, "@Bob").rendering_preview()
        assert preview == f"@Bob (request): {'x' * PREVIEW_LIMIT}…"

    def test_an_empty_body_leaves_no_dangling_colon(self) -> None:
        assert _agent_message("", "@Bob").rendering_preview() == "@Bob (request)"

    def test_a_senderless_message_previews_as_unknown(self) -> None:
        assert AgentMessage(content="hi").rendering_preview() == "unknown (request): hi"


class TestTriageMessageRendersItself:
    """AC 8: the prompt its handler used to build inline, byte-for-byte."""

    def test_it_reproduces_the_inline_triage_prompt(self) -> None:
        incident = TriageMessage(incident="disk full on node 3", reported_by="monitoring")
        assert incident.rendering() == (
            "Incident reported by monitoring:\n\ndisk full on node 3\n\n"
            "Assess severity, summarise in one line, and hand off whatever you "
            "cannot resolve yourself."
        )


# =============================================================================
# C. The offer rule (AC 14-20, 25)
# =============================================================================


class TestOfferRule:
    """All four conditions must hold; each is pinned on its own below."""

    def test_a_same_class_previewable_non_cancel_is_offered(self) -> None:
        pending = _agent_message("please review", "@Alice")
        capability = MailboxCapability(
            observer=_MailboxDouble([pending], current=_agent_message("handled", "@Human")),  # type: ignore[arg-type]
            card=MailboxTool(),
        )

        assert capability.offerable_ids([pending]) == {pending.id}

    def test_a_message_of_another_class_is_never_offered(self) -> None:
        """Rule 15c — same class means same handler means same output type.

        Without it a pending ``AgentMessage`` read during a ``TriageMessage``
        run would be answered as ``TriageOutput``: correctly framed, wrongly
        routed.
        """
        pending = _agent_message("please review", "@Alice")
        during_triage = TriageMessage(incident="disk full")
        capability = MailboxCapability(
            observer=_MailboxDouble([pending], current=during_triage),  # type: ignore[arg-type]
            card=MailboxTool(),
        )

        assert capability.offerable_ids([pending]) == set()

    def test_a_triage_message_pending_during_an_agent_run_is_never_offered(self) -> None:
        """The mirror of the above, which is the direction the defect ran."""
        pending = TriageMessage(incident="disk full")
        capability = MailboxCapability(
            observer=_MailboxDouble([pending], current=_agent_message()),  # type: ignore[arg-type]
            card=MailboxTool(),
        )

        assert capability.offerable_ids([pending]) == set()

    def test_a_stop_bearing_agent_message_is_never_offered(self) -> None:
        """Rule 15d — offering its id would let the model read its way out of a cancel."""
        stop = _agent_message("/stop", "@Human")
        capability = MailboxCapability(
            observer=_MailboxDouble([stop], current=_agent_message("handled", "@Human")),  # type: ignore[arg-type]
            card=MailboxTool(),
        )

        assert capability.offerable_ids([stop]) == set()

    def test_a_cancel_message_is_never_offered(self) -> None:
        cancel = CancelMessage(reason="user pressed Esc")
        capability = MailboxCapability(
            observer=_MailboxDouble([cancel], current=cancel),  # type: ignore[arg-type]
            card=MailboxTool(),
        )

        assert capability.offerable_ids([cancel]) == set()

    def test_a_non_previewable_message_of_the_handlers_own_class_is_not_offered(self) -> None:
        """Rule 15a — the condition whose absence was a production crash.

        A ``TriageMessage`` pending during a ``TriageMessage`` run passes every
        other rule on the default configuration. Without previewability in the
        filter it would be marked offerable and the render would raise.
        """
        pending = TriageMessage(incident="another disk")
        capability = MailboxCapability(
            observer=_MailboxDouble([pending], current=TriageMessage(incident="first disk")),  # type: ignore[arg-type]
            card=MailboxTool(),
        )

        assert capability.offerable_ids([pending]) == set()

    def test_nothing_is_offered_while_the_agent_is_idle(self) -> None:
        pending = _agent_message()
        capability = MailboxCapability(
            observer=_MailboxDouble([pending], current=None),  # type: ignore[arg-type]
            card=MailboxTool(),
        )

        assert capability.offerable_ids([pending]) == set()


class TestNoticeIntegration:
    """The filter and the render, driven together through the hook."""

    async def test_the_notice_offers_an_id_for_what_this_run_can_handle(self) -> None:
        pending = _agent_message("please review", "@Alice")
        capability = MailboxCapability(
            observer=_MailboxDouble([pending], current=_agent_message("handled", "@Human")),  # type: ignore[arg-type]
            card=MailboxTool(),
        )
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, MagicMock(messages=[]))  # type: ignore[arg-type]

        (content,), priority = ctx.enqueue_calls[0]
        assert priority == "asap"
        assert f"(id: {pending.id})" in content
        assert "please review" in content

    async def test_the_notice_withholds_the_id_for_what_it_cannot(self) -> None:
        """The everyday case the guard must NOT turn into an exception."""
        pending = TriageMessage(incident="disk full")
        capability = MailboxCapability(
            observer=_MailboxDouble([pending], current=TriageMessage(incident="other")),  # type: ignore[arg-type]
            card=MailboxTool(),
        )
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, MagicMock(messages=[]))  # type: ignore[arg-type]

        (content,), _ = ctx.enqueue_calls[0]
        assert UNOFFERABLE_LINE in content
        assert str(pending.id) not in content
        assert "disk full" not in content


class TestMailboxRenderErrorGuardsTheFilterOnly:
    """AC 25 — the guard fires on a broken filter, never on a non-offer."""

    def test_it_raises_when_an_unpreviewable_message_is_marked_offerable(self) -> None:
        pending = TriageMessage(incident="disk full")

        with pytest.raises(MailboxRenderError) as raised:
            render_arrival_notice([pending], {pending.id})

        assert "TriageMessage" in str(raised.value)

    def test_it_does_not_raise_for_the_same_class_when_no_id_is_offered(self) -> None:
        pending = TriageMessage(incident="disk full")

        notice = render_arrival_notice([pending], set())

        assert UNOFFERABLE_LINE in notice


# =============================================================================
# D. Id-based injection (AC 21-24)
# =============================================================================


class TestAfterToolExecuteInjection:
    async def test_a_read_mailbox_call_consumes_that_id_and_injects_its_rendering(self) -> None:
        """AC 22 — exactly one consume, exactly one enqueue, result untouched."""
        named = _agent_message("the real content", "@Alice")
        other = _agent_message("not this one", "@Bob")
        mailbox = _MailboxDouble([named, other])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())  # type: ignore[arg-type]
        ctx = _CtxDouble()

        result = await _after_read(capability, ctx, {MESSAGE_ID_ARG: str(named.id)})

        assert mailbox.consume_calls == [[named.id]]
        assert mailbox.pending == [other]
        # The rendering is wrapped in the added-work framing, and carried whole:
        # the hook delivers, and framing a delivery is part of delivering it.
        (enqueued,), priority = ctx.enqueue_calls[0]
        assert len(ctx.enqueue_calls) == 1
        assert priority == "asap"
        assert named.rendering() in enqueued
        assert "does NOT replace" in enqueued
        assert result == "Acknowledged."

    async def test_another_tool_is_left_entirely_alone(self) -> None:
        """AC 21 — the hook does nothing at all for any other tool."""
        pending = _agent_message()
        mailbox = _MailboxDouble([pending])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())  # type: ignore[arg-type]
        ctx = _CtxDouble()

        result = await _after_read(
            capability, ctx, {MESSAGE_ID_ARG: str(pending.id)}, tool="workspace_read"
        )

        assert mailbox.consume_calls == []
        assert ctx.enqueue_calls == []
        assert result == "Acknowledged."

    @pytest.mark.parametrize(
        "args",
        [
            pytest.param({}, id="absent"),
            pytest.param({MESSAGE_ID_ARG: ""}, id="empty"),
            pytest.param({MESSAGE_ID_ARG: "not-a-uuid"}, id="malformed"),
            pytest.param({MESSAGE_ID_ARG: None}, id="null"),
        ],
    )
    async def test_an_unusable_id_is_a_silent_no_op(self, args: dict[str, Any]) -> None:
        """AC 23 — including the empty ``args`` an older ``read_mailbox()`` produces.

        This is not a hypothetical branch. A deployment pinned to a published
        tool whose ``read_mailbox`` takes no arguments reaches exactly here, and
        the correct behaviour is to do nothing rather than raise inside a run.
        """
        pending = _agent_message()
        mailbox = _MailboxDouble([pending])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())  # type: ignore[arg-type]
        ctx = _CtxDouble()

        result = await _after_read(capability, ctx, args)

        assert mailbox.consume_calls == []
        assert mailbox.pending == [pending]
        assert ctx.enqueue_calls == []
        assert result == "Acknowledged."

    async def test_an_id_that_is_no_longer_queued_injects_nothing(self) -> None:
        mailbox = _MailboxDouble([])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())  # type: ignore[arg-type]
        ctx = _CtxDouble()

        await _after_read(capability, ctx, {MESSAGE_ID_ARG: str(uuid.uuid4())})

        assert ctx.enqueue_calls == []

    async def test_a_message_that_cannot_render_is_left_in_the_mailbox(self) -> None:
        """Not consumed, so not lost — it still arrives as its own turn.

        The hook peeks and renders BEFORE it consumes. Consuming first would
        delete a message that turns out to render nothing: gone from the queue,
        never enqueued, with nothing left holding it. That is silent data loss,
        and it is what this order exists to prevent.

        MUTATION — move ``consume_mailbox`` above the render check and this goes
        red on its own.
        """
        unrenderable = UserMessage(content="I have a body but no rendering")
        mailbox = _MailboxDouble([unrenderable])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())  # type: ignore[arg-type]
        ctx = _CtxDouble()

        await _after_read(capability, ctx, {MESSAGE_ID_ARG: str(unrenderable.id)})

        assert mailbox.consume_calls == []
        assert mailbox.pending == [unrenderable]
        assert ctx.enqueue_calls == []


class TestTheToolContractIsReadByName:
    """The cross-repository half: ``akgentic-tool`` owns the signature.

    These two specs are the ones expected to fail in remote CI until the tool
    release carrying the id-based ``read_mailbox`` is on PyPI — the documented
    merge-order non-blocker, not a defect here.
    """

    def test_the_card_serves_a_read_mailbox_taking_the_argument_we_read(self) -> None:
        import inspect

        from akgentic.tool.mailbox import MailboxTool

        card = MailboxTool()
        card._observer = lambda: None  # type: ignore[assignment, method-assign]
        read_mailbox = next(t for t in card.get_tools() if t.__name__ == READ_MAILBOX_TOOL)

        parameters = list(inspect.signature(read_mailbox).parameters)
        assert parameters == [MESSAGE_ID_ARG]

    async def test_the_id_the_tool_names_is_the_id_the_hook_absorbs(self) -> None:
        """End to end across the boundary, without a live model."""
        import inspect

        from akgentic.tool.mailbox import MailboxTool

        named = _agent_message("absorb me", "@Alice")
        mailbox = _MailboxDouble([named])
        capability = MailboxCapability(observer=mailbox, card=MailboxTool())  # type: ignore[arg-type]
        ctx = _CtxDouble()

        card = MailboxTool()
        card._observer = lambda: None  # type: ignore[assignment, method-assign]
        read_mailbox = next(t for t in card.get_tools() if t.__name__ == READ_MAILBOX_TOOL)
        argument_name = next(iter(inspect.signature(read_mailbox).parameters))

        await _after_read(capability, ctx, {argument_name: str(named.id)})

        assert mailbox.consume_calls == [[named.id]]
        # The rendering is wrapped in the added-work framing, and carried whole:
        # the hook delivers, and framing a delivery is part of delivering it.
        (enqueued,), priority = ctx.enqueue_calls[0]
        assert len(ctx.enqueue_calls) == 1
        assert priority == "asap"
        assert named.rendering() in enqueued
        assert "does NOT replace" in enqueued


class TestAnAbsorbedMessageIsFramedAsAddedWork:
    """An absorbed message must not read as a replacement for the current one.

    ``rendering()`` renders a message the way its own handler receives it —
    imperative and self-contained. Injected mid-run that reads as a fresh
    assignment, and the model answers it *instead of* what it was already doing.
    Observed in the field: an agent that had just written a report answered only
    the newer question, and the report answer reached nobody.

    What is pinned here is the **shape of the delivery**, not the sentences the
    prefix happens to use: the prefix arrives ahead of the message's own
    rendering, and that rendering is carried whole. The one clause pinned by its
    text is *"does NOT replace"* — the clause the field failure was opened for.
    Everything else in the string is prompt wording, free to be re-tuned without
    reddening a suite; an assert on a phrase would make the next wording pass a
    test failure for no behavioural reason.
    """

    async def test_the_injection_says_additional_and_carries_the_rendering_whole(self) -> None:
        """MUTATION — enqueue ``message.rendering()`` bare, as it was before,
        and the first two assertions go red, along with the two sibling specs
        above that pin the same ``"does NOT replace"`` clause: three in this
        file, nothing outside it. Restoring the prefix's earlier *wording*
        reddens none of the three — these pin the delivery's shape, not its
        sentences.
        """
        absorbed = _agent_message("what is the colour of the sky?", "@Human")
        capability = MailboxCapability(observer=_MailboxDouble([absorbed]), card=MailboxTool())
        ctx = _CtxDouble()

        await _after_read(capability, ctx, {MESSAGE_ID_ARG: str(absorbed.id)})

        (enqueued,), priority = ctx.enqueue_calls[0]
        assert "does NOT replace" in enqueued
        assert enqueued == f"{ABSORBED_PREFIX}\n\n{absorbed.rendering()}"
        assert priority == "asap"

    async def test_the_prefix_is_the_one_the_capability_was_built_with(self) -> None:
        """Epic 27 — a custom prefix in, the same prefix out.

        The framing text is the capability's, taken from whoever constructed it,
        which at the wiring site is the ``MailboxTool`` card. What is pinned is
        the invariant, not the sentences: the constructed value frames the
        delivery, and the message's own rendering is still carried whole.

        MUTATION — restore ``ABSORBED_PREFIX`` at the ``ctx.enqueue`` call in
        ``after_tool_execute`` and this goes red on its own; the sibling spec
        above stays green, because it constructs no prefix of its own.
        """
        sentinel = "SENTINEL PREFIX — configured on the card."
        absorbed = _agent_message("what is the colour of the sky?", "@Human")
        capability = MailboxCapability(
            observer=_MailboxDouble([absorbed]),
            card=MailboxTool(absorbed_prefix=sentinel),
        )
        ctx = _CtxDouble()

        await _after_read(capability, ctx, {MESSAGE_ID_ARG: str(absorbed.id)})

        (enqueued,), _priority = ctx.enqueue_calls[0]
        assert enqueued == f"{sentinel}\n\n{absorbed.rendering()}"
