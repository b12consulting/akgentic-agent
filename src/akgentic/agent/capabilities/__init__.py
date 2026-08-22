"""Capabilities the agent builds for itself, independent of any tool card.

A capability here is wired unconditionally by ``BaseAgent`` — it is not
contributed by a ``ToolCard`` and cannot be de-configured by omitting one.
That is why the mailbox capability defines its own cancel vocabulary rather
than importing it from ``akgentic.tool.mailbox``: an agent configured with no
``MailboxTool`` must still be interruptible.
"""

from akgentic.agent.capabilities.mailbox_capability import (
    MailboxCancelCapability,
    RunInterruptedError,
    is_cancel,
    render_arrival_notice,
)

__all__ = [
    "MailboxCancelCapability",
    "RunInterruptedError",
    "is_cancel",
    "render_arrival_notice",
]
