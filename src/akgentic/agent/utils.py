"""Shared helpers for agent classes — currently team addressing.

Free of any import from ``agent.py`` on purpose: these are pieces a *new* agent
class needs, so a dependency in that direction would make them unusable from the
module that defines the base class. What each needs from an agent is stated
structurally, as a Protocol, never by naming a class.

The usage-limit policy lives in ``usage_limits.py``, not here.
"""

import logging
from typing import Protocol

from pydantic_ai import ModelRetry

from akgentic.core import ActorAddress

logger = logging.getLogger(__name__)


class TeamResolver(Protocol):
    """What ``resolve_recipient`` needs from an agent: the two lookup paths."""

    def get_team_member(self, name: str) -> ActorAddress | None:
        """Return an existing member by ``@name``, or None when absent."""
        ...

    def hire_member(self, role: str) -> ActorAddress:
        """Hire a member for ``role`` and return its address."""
        ...


def resolve_recipient(agent: TeamResolver, recipient: str) -> ActorAddress | None:
    """Resolve a recipient the LLM named to an actor address.

    The team's addressing convention in one place: a leading ``@`` means an
    existing member, anything else is a role to hire on demand. It belongs to the
    team rather than to any output schema, so every agent class routing an
    LLM-chosen recipient needs exactly this.

    Args:
        agent: The agent performing the lookup.
        recipient: An ``@member`` name, or a role name to hire.

    A role that cannot be hired is treated the same way. This runs after the LLM
    run has ended, where nothing can honour a retry, so a refused hire
    (``ModelRetry``) or a missing hire command (``RuntimeError``) is logged and
    skipped rather than raised out of the actor's handler — which would lose the
    rest of the turn's messages along with it.

    Returns:
        The member's address, or ``None`` when an ``@name`` matches nobody or a role
        cannot be hired — so a recipient nobody answers to costs a delivery, not an
        exception.
    """
    if recipient.startswith("@"):
        return agent.get_team_member(recipient)
    try:
        return agent.hire_member(recipient)
    except (ModelRetry, RuntimeError) as error:
        logger.warning(
            "No member answers to %r and it cannot be hired (%s); message not delivered",
            recipient,
            error,
        )
        return None
