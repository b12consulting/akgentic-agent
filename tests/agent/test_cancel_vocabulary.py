"""Tests for the run-cancellation vocabulary owned by the agent (Epic 20, FR8).

``is_cancel`` and ``render_arrival_notice`` are *defined* in
``akgentic.agent.capabilities.mailbox_capability`` — not imported from
``akgentic.tool.mailbox``. The behavioural specs below are the tool suite's,
ported verbatim so a byte-for-byte identical vocabulary is proven rather than
assumed; the two ownership specs at the bottom are what keeps it here.

This module deliberately imports nothing from ``akgentic.tool.mailbox``: the
absence is part of the proof that the vocabulary carries no card dependency.
"""

from __future__ import annotations

import uuid

import pytest
from akgentic.core import ActorAddressProxy
from akgentic.core.messages import CancelMessage, Message, UserMessage

from akgentic.agent.capabilities import is_cancel, render_arrival_notice
from akgentic.agent.capabilities.mailbox_capability import UNOFFERABLE_LINE
from akgentic.agent.messages import PREVIEW_LIMIT, AgentMessage

# The two closing lines, once each: every exact-string spec below ends with one.
# Which one is not decoration — the notice may only point at ``read_mailbox``
# when the listing actually carries an id to name.
CLOSING_WITH_IDS = (
    "Call `read_mailbox` with one of the ids above to take that message on now, "
    "or finish your current work first — you will get them just after."
)
CLOSING_WITHOUT_IDS = "Finish your current work first — you will get them just after."

# =============================================================================
# HELPERS — local by design, no reach across the package boundary
# =============================================================================


def _address(name: str, role: str = "Agent") -> ActorAddressProxy:
    """A mock ActorAddress carrying a display name."""
    return ActorAddressProxy(
        {
            "__actor_address__": True,
            "__actor_type__": "test.Agent",
            "agent_id": str(uuid.uuid4()),
            "name": name,
            "role": role,
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "is_user_proxy": False,
        }
    )


def _agent_message(sender: str, content: str) -> AgentMessage:
    """An AgentMessage carrying a mock sender address.

    The notice specs use this rather than ``UserMessage`` because previewability
    is now the discriminator: ``AgentMessage`` declares ``mailbox_preview()``
    and ``UserMessage`` does not, so only the former can ever carry an id.
    """
    message = AgentMessage(content=content, type="request")
    message.sender = _address(sender)
    return message


# =============================================================================
# is_cancel — both spellings of one intent
# =============================================================================


def test_cancel_message_instance_is_cancel() -> None:
    # The typed spelling (programmatic senders).
    assert is_cancel(CancelMessage(reason="user pressed Esc")) is True


def test_stop_content_is_cancel() -> None:
    # The string spelling (human / frontend Esc).
    assert is_cancel(UserMessage(content="/stop")) is True


def test_stop_with_leading_space_and_trailing_words_is_cancel() -> None:
    assert is_cancel(UserMessage(content="  /stop now")) is True


def test_ordinary_content_is_not_cancel() -> None:
    assert is_cancel(UserMessage(content="please summarize the thread")) is False


def test_stopwatch_is_not_cancel() -> None:
    # Exact-token rule: /stop followed by end or whitespace only.
    assert is_cancel(UserMessage(content="/stopwatch")) is False


def test_message_without_content_is_not_cancel() -> None:
    # A content-less message is simply False, never an error.
    assert is_cancel(Message()) is False


class _PayloadMessage(Message):
    """A message whose content is not a string."""

    content: int


def test_non_string_content_is_not_cancel() -> None:
    # The non-str content guard: simply False, never an error.
    assert is_cancel(_PayloadMessage(content=5)) is False


def test_empty_content_is_not_cancel() -> None:
    assert is_cancel(UserMessage(content="")) is False


def test_stop_mid_sentence_is_not_cancel() -> None:
    # The first token must be /stop — mentioning it later is not a cancel.
    assert is_cancel(UserMessage(content="please /stop")) is False


# =============================================================================
# render_arrival_notice — the mid-run doorbell wording
# =============================================================================


def _offered(*messages: Message) -> set[uuid.UUID]:
    """The offerable-id set naming every message given."""
    return {message.id for message in messages}


def test_arrival_notice_empty_list_says_nothing() -> None:
    assert render_arrival_notice([], set()) == ""


def test_arrival_notice_renders_count_one_line_per_offered_message_and_pointer() -> None:
    # A count line, a line per message carrying sender/type/preview AND its id,
    # then the pointer naming the ids.
    first = _agent_message("@Alice", "hello")
    second = _agent_message("@Bob", "ping")
    assert render_arrival_notice([first, second], _offered(first, second)) == (
        "2 new messages arrived:\n"
        f"- @Alice (request): hello (id: {first.id})\n"
        f"- @Bob (request): ping (id: {second.id})\n" + CLOSING_WITH_IDS
    )


def test_arrival_notice_singular_message() -> None:
    only = _agent_message("@Alice", "hello")
    assert render_arrival_notice([only], _offered(only)) == (
        f"1 new message arrived:\n- @Alice (request): hello (id: {only.id})\n" + CLOSING_WITH_IDS
    )


def test_a_message_not_offered_is_listed_without_an_id_or_its_content() -> None:
    """The missing id IS the guard — visible, but unaskable.

    Not an error and not a refusal: the affordance simply is not offered. The
    body is withheld too, so nothing about a message this run cannot handle
    leaks into the prompt.
    """
    unofferable = _agent_message("@Alice", "a secret nobody asked for")

    notice = render_arrival_notice([unofferable], set())

    assert notice == "1 new message arrived:\n" + UNOFFERABLE_LINE + "\n" + CLOSING_WITHOUT_IDS
    assert "secret" not in notice
    assert str(unofferable.id) not in notice


def test_a_listing_with_no_ids_does_not_promise_a_read() -> None:
    """The closing line may not point at a tool call the model cannot make."""
    notice = render_arrival_notice([Message(), Message()], set())

    assert "read_mailbox" not in notice
    assert notice.endswith(CLOSING_WITHOUT_IDS)


def test_a_bare_message_is_never_offered_even_when_the_filter_says_so() -> None:
    """The internal-invariant guard: our own filter admitted the inadmissible.

    A bare ``Message`` declares no ``mailbox_preview``, so an id for it can only
    come from a broken filter. It raises rather than rendering a blank line —
    and it raises **only** on that path, which the sibling assertion pins.
    """
    from akgentic.agent.capabilities import MailboxRenderError

    unpreviewable = Message()

    with pytest.raises(MailboxRenderError):
        render_arrival_notice([unpreviewable], {unpreviewable.id})

    # The same class, the same helper, no id: a silent non-offer, never a raise.
    assert render_arrival_notice([unpreviewable], set()) == (
        "1 new message arrived:\n" + UNOFFERABLE_LINE + "\n" + CLOSING_WITHOUT_IDS
    )


def test_arrival_notice_names_the_declared_message_type() -> None:
    # ``AgentMessage`` carries a type; it is what the parenthesis shows.
    message = AgentMessage(content="deploy finished", type="notification")
    message.sender = _address("@Bob")
    notice = render_arrival_notice([message], _offered(message))
    assert f"- @Bob (notification): deploy finished (id: {message.id})\n" in notice


def test_arrival_notice_gives_every_message_its_own_line_in_reception_order() -> None:
    # Senders are no longer deduplicated: a sender who wrote twice gets two
    # lines, and the order is the order the mailbox returned.
    first = _agent_message("@Bob", "one")
    second = _agent_message("@Alice", "two")
    third = _agent_message("@Bob", "three")

    notice = render_arrival_notice([first, second, third], _offered(first, second, third))

    assert notice.splitlines()[1:4] == [
        f"- @Bob (request): one (id: {first.id})",
        f"- @Alice (request): two (id: {second.id})",
        f"- @Bob (request): three (id: {third.id})",
    ]


def test_an_unofferable_message_keeps_its_place_in_the_order() -> None:
    """Nothing is dropped from the listing — only the id varies."""
    offered = _agent_message("@Alice", "readable")
    withheld = _agent_message("@Bob", "not for this run")

    notice = render_arrival_notice([withheld, offered], _offered(offered))

    assert notice.splitlines()[1:3] == [
        UNOFFERABLE_LINE,
        f"- @Alice (request): readable (id: {offered.id})",
    ]


def test_arrival_notice_truncates_a_long_preview_with_an_ellipsis() -> None:
    long_message = _agent_message("@Alice", "x" * 400)
    notice = render_arrival_notice([long_message], _offered(long_message))

    assert f"- @Alice (request): {'x' * PREVIEW_LIMIT}…" in notice
    assert "x" * (PREVIEW_LIMIT + 1) not in notice


def test_arrival_notice_leaves_a_short_preview_whole() -> None:
    short = _agent_message("@Alice", "short enough")
    assert "…" not in render_arrival_notice([short], _offered(short))


# =============================================================================
# FR8 — the vocabulary is the agent's, structurally
# =============================================================================


class TestVocabularyOwnership:
    """The predicate and the wording are defined here, not borrowed from a card.

    ``MailboxCapability`` is built unconditionally, so an agent carrying
    no ``MailboxTool`` must still cancel — a vocabulary that shipped with the
    card could not serve that agent. ``__module__`` rather than an identity
    comparison against the tool's copy: the tool's copies are on their way out,
    and this guard must outlive their removal.
    """

    def test_is_cancel_is_defined_by_the_agent(self) -> None:
        assert is_cancel.__module__ == "akgentic.agent.capabilities.mailbox_capability"

    def test_render_arrival_notice_is_defined_by_the_agent(self) -> None:
        assert (
            render_arrival_notice.__module__ == "akgentic.agent.capabilities.mailbox_capability"
        )
