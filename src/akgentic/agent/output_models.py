"""Structured output models for agent response patterns.

This module defines Pydantic models that enable agents to return structured responses
in three distinct patterns:
"""

from typing import Literal

from pydantic import BaseModel, Field

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
            "and HIRES A NEW AGENT for that role — use it only when you intend to add a "
            "member to the team, never to reach a teammate you already have."
        ),
    )


class StructuredOutput(BaseModel):
    messages: list[Request] = Field(
        default_factory=list,
        description=(
            "Every message you send this turn. This list is your only channel: content "
            "that is not in a message here reaches nobody, including the human. You may "
            "send several messages in one turn — they are dispatched in parallel. Leave "
            "empty only when you are waiting on someone and have nothing to send."
        ),
    )
