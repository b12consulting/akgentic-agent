"""akgentic-agent: Collaborative agent patterns for the Akgentic framework.

Dynamic team composition, hierarchical agent management, and a single static
structured-output schema for routing messages between team members.
"""

from importlib import metadata as _metadata

from akgentic.agent.agent import BaseAgent
from akgentic.agent.capabilities import MailboxRenderError, RunInterruptedError
from akgentic.agent.config import AgentConfig
from akgentic.agent.human_proxy import HumanProxy
from akgentic.agent.messages import AgentMessage, LlmRenderable, MailboxPreviewable

__all__ = [
    # Version
    "__version__",
    "AgentConfig",
    # Human-in-the-loop
    "HumanProxy",
    # BaseAgent
    "BaseAgent",
    # Run cancellation
    "RunInterruptedError",
    # Mailbox offer-rule invariant
    "MailboxRenderError",
    # Team messages
    "AgentMessage",
    # Rendering contracts
    "LlmRenderable",
    "MailboxPreviewable",
]

try:
    __version__ = _metadata.version("akgentic-agent")
except _metadata.PackageNotFoundError:  # pragma: no cover - source tree, never installed
    # Importing from a source tree that was never installed must not fail over a
    # version string. A hardcoded literal here is what drifted from pyproject.toml
    # across several releases; the sentinel is unmistakably "not a real version".
    __version__ = "0.0.0+unknown"
