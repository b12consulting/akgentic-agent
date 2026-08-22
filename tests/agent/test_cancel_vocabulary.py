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

from akgentic.core import ActorAddressProxy
from akgentic.core.messages import CancelMessage, Message, UserMessage

from akgentic.agent.capabilities import is_cancel, render_arrival_notice

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


def _user_message(sender: str, content: str) -> UserMessage:
    """A UserMessage carrying a mock sender address."""
    message = UserMessage(content=content)
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


def test_arrival_notice_empty_list_says_nothing() -> None:
    assert render_arrival_notice([]) == ""


def test_arrival_notice_renders_count_senders_and_pointer() -> None:
    # One line — count, senders, and the read_mailbox pointer.
    messages: list[Message] = [
        _user_message("@Alice", "hello"),
        _user_message("@Bob", "ping"),
    ]
    assert render_arrival_notice(messages) == (
        "2 new messages arrived (from @Alice, @Bob) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )


def test_arrival_notice_singular_message() -> None:
    assert render_arrival_notice([_user_message("@Alice", "hello")]) == (
        "1 new message arrived (from @Alice) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )


def test_arrival_notice_defends_senderless_message() -> None:
    # A message without a usable sender/content still renders.
    assert render_arrival_notice([Message()]) == (
        "1 new message arrived (from unknown) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )


def test_arrival_notice_deduplicates_senders_in_first_seen_order() -> None:
    messages: list[Message] = [
        _user_message("@Bob", "one"),
        _user_message("@Alice", "two"),
        _user_message("@Bob", "three"),
    ]
    assert "(from @Bob, @Alice)" in render_arrival_notice(messages)


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
