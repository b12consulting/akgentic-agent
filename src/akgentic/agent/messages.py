"""Team message types.

This module provides ``AgentMessage``, the single message type used for
agent-to-agent communication.
"""

from typing import Literal

from akgentic.core.messages import Message


class AgentMessage(Message):
    """Base message type for team communication.

    Attributes:
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
