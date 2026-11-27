"""Structured output models for agent response patterns.

This module defines Pydantic models that enable agents to return structured responses
in three distinct patterns:
"""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, Field, ValidationError

from akgentic.llm import MergeableOutput
from akgentic.llm.output import error_lines

# ---------------------------------------------------------------------------
# Reply protocols
#
# Prepended to the front of every incoming message, ahead of the sender's
# content (see BaseAgent.receiveMsg_AgentMessage) — the most salient position in
# the prompt.
#
# Because of that salience, these strings must describe MESSAGE MECHANICS only:
# what kind of message arrived and whether a reply is expected. They must not
# say anything about who should do the work, or whether to delegate — that is
# team policy and belongs in the agents' prompts, where it can differ per role.
#
# The stock text said "Carry out the task, then respond to {sender}. You may
# also delegate to others", which is policy, applied to every agent in every
# team, from the highest-priority position in the prompt. It is what made
# coordinators do their specialists' work. Stating the reverse ("delegate")
# is the same mistake with the opposite sign: it made specialists fan out to
# each other. Both are removed here.
# ---------------------------------------------------------------------------

REPLY_PROTOCOLS: dict[str, str] = {
    "request": "A reply is expected: respond to {sender} with the result.",
    "response": "This is a reply to something you asked. Take it into account and continue.",
    "instruction": "Carry it out; acknowledge to {sender} only if asked to.",
    "notification": "Informational message. No reply is expected.",
    "acknowledgment": "Receipt confirmed. No further action needed.",
}


class Request(BaseModel):
    """A message directed to a specific team member or role."""

    message_type: Literal[
        "request",
        "response",
        "notification",
        "instruction",
        "acknowledgment",
    ] = Field(
        ...,
        description="Choose based on intent: "
        "'request' = ask recipient to perform a task and reply to you with the result; "
        "'response' = respond to a previous request; "
        "'instruction' = direct recipient to perform a task, you may ask for acknowledgement; "
        "'notification' = send information to the recipient, no reply is expected; "
        "'acknowledgment' = confirm receipt of an instruction, no reply is expected.",
    )
    message: str = Field(..., description="The message content to send")
    recipient: str = Field(
        ...,
        description=(
            "The exact '@Name' of a team member, copied from the team roster "
            "(e.g. '@Developer'). A value that is not an '@Name' is treated as a role "
            "and HIRES A NEW AGENT for that role, if the team allows hiring it — use it "
            "only when you intend to add a member to the team, never to reach a teammate "
            "you already have."
        ),
    )


class StructuredOutput(MergeableOutput):
    """Your whole reply for this turn: exactly one JSON object holding a "messages" array."""

    messages: list[Request] = Field(
        default_factory=list,
        description=(
            "All messages you send this turn, as items of this one array. This list is your only "
            "channel: content that is not in a message here reaches nobody, including the human. "
            "To send several messages, add several items to this array — they are dispatched in "
            "parallel. Leave it empty only when you are waiting on someone and have nothing to "
            'send. Your reply is exactly ONE JSON object with ONE "messages" array: never write '
            "a second JSON object after it, and never one JSON object per message."
        ),
    )

    @classmethod
    def merge(cls, outputs: Sequence[Self]) -> Self:
        """Every output's messages, in the order written, in one output; a repeat is sent once.

        A split output restates its turn, so a ``(message_type, message, recipient)`` already
        seen earlier in the merge is dropped, and its first occurrence keeps its place. Every
        other field a subclass adds comes from the last output, the one the model finished on.
        """
        unique: dict[tuple[str, str, str], Request] = {}
        for output in outputs:
            for message in output.messages:
                key = (message.message_type, message.message, message.recipient)
                unique.setdefault(key, message)
        return outputs[-1].model_copy(update={"messages": list(unique.values())})

    @classmethod
    def unreadable_output_error(cls, error: ValidationError) -> str:
        """Say that nothing was sent, so the model resends every message, not only the rest."""
        lines = [
            "Nothing was sent: your reply could not be read, so none of its messages reached "
            "anyone."
        ]
        lines += error_lines(error)
        lines.append(
            'Resend every message of this turn as items of ONE "messages" array, in exactly ONE '
            "JSON object."
        )
        return "\n".join(lines)
