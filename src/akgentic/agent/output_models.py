"""Structured output models for agent response patterns.

This module defines Pydantic models that enable agents to return structured responses
in three distinct patterns:
"""

from typing import Literal

from pydantic import BaseModel, Field

structured_output = """This thread was triggered by a {message_type} from ({sender}).
{reply_protocol}

You CANNOT wait, sleep, poll, or loop. Return an empty list instead.
You process ONE message at a time. After you conclude, your turn ends.

Team members: {team}.
Available roles: {roles}.
"""

REPLY_PROTOCOLS: dict[str, str] = {
    "request": "You MUST respond to {sender}. You may also delegate to others.",
    "response": "This is a reply to your earlier request. Continue or end the exchange.",
    "notification": "Informational only. Do NOT reply to {sender}. Return an empty list.",
    "instruction": "Acknowledge to {sender} that you understood. You may also delegate to others.",
    "acknowledgment": "Receipt confirmed. No further action needed. Return an empty list.",
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
        "'request' = ask recipient to act, a reply to this message is expected; "
        "'response' = reply to a previous request; "
        "'notification' = notification the recipient, no reply to this message is expected; "
        "'instruction' = give instruction to the recipient, an acknowledgment is expected; "
        "'acknowledgment' = confirm receipt of an instruction.",
    )
    message: str = Field(..., description="The message content to send")
    recipient: str = Field(
        ...,
        description="The recipient by name (e.g. '@Developer') or by role (e.g. 'Developer')",
    )


class StructuredOutput(BaseModel):
    messages: list[Request] = Field(
        default_factory=list,
        description="Requests to send to team members; empty if no delegation needed",
    )
