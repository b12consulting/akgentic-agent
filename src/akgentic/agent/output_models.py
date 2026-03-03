"""Structured output models for agent response patterns.

This module defines Pydantic models that enable agents to return structured responses
in three distinct patterns:
"""

from pydantic import BaseModel, Field

structured_output = """This thread was triggered by ({sender}).

Execution model:
You process ONE message at a time. After you respond, your turn ends.
If other members send you messages later, you will be called again with full context history.
You CANNOT wait, sleep, poll, or loop. Do NOT use tools as a stalling mechanism.

How to respond - choose ONE of these actions:
1. You can complete your task: respond to {sender} and/or other members,
   or return an empty list if no message is needed.
2. You need input from team members: send request(s) to existing members by name ({team})
   or to new members by role ({roles}).
3. You are waiting for messages that have not arrived yet: return an EMPTY messages list.
   You will be called again when the next message arrives. Your context is preserved.

CRITICAL: Only contact members when there is a clear request. Each thread has a cost!
Do NOT call tools just to stay active or wait. Return an empty list instead.
"""


class Request(BaseModel):
    """A message directed to a specific team member or role."""

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
