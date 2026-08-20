"""Shared helpers for agent classes — currently team addressing.

Free of any import from ``agent.py`` on purpose: these are pieces a *new* agent
class needs, so a dependency in that direction would make them unusable from the
module that defines the base class. What each needs from an agent is stated
structurally, as a Protocol, never by naming a class.

The usage-limit tier policy lives in ``usage_limits.py``, not here.
"""

from typing import Protocol

from akgentic.core import ActorAddress


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

    Returns:
        The member's address, or ``None`` when an ``@name`` matches nobody — so the
        model naming someone who does not exist costs a delivery, not an exception.
    """
    if recipient.startswith("@"):
        return agent.get_team_member(recipient)
    return agent.hire_member(recipient)
