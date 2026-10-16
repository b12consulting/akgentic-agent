"""Recipient validation inside the LLM run.

A recipient the model writes is resolved by ``resolve_recipient`` only after the run
has ended: an ``@name`` must match a member, and anything else is a role to hire.
At that point a recipient nobody answers to can only be skipped. This module checks
the same thing while the run is still open, so the model is told and can correct it.

- :class:`TeamDirectory` — who a recipient can name right now: the members, and the
  roles a hire would accept. Built by ``BaseAgent.team_directory()``, which reads
  those roles from the orchestrator's agent catalog.
- :class:`RecipientCheckCapability` — a pydantic-ai capability. On every validated
  output it asks the directory about each recipient and raises ``ModelRetry`` while
  one is unreachable. ``BaseAgent`` mounts one for ``StructuredOutput``; an agent
  routing another shape mounts its own with a different extractor.

The check stands down whenever the run could not honour a retry, and routing then
skips what is still unreachable. It never fails a run that would otherwise succeed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.output import OutputContext

from akgentic.agent.output_models import StructuredOutput

RecipientExtractor = Callable[[object], list[str] | None]
"""Returns the recipients an output names, or ``None`` when it is not a shape to check."""


class TeamDirectory(BaseModel):
    """Who a recipient can name: the team's members and the roles a hire would accept.

    Attributes:
        members: ``@name`` of every member, tool actors (``#``-prefixed) excluded.
        hireable_roles: Roles ``hire_member`` would accept: the roles of the
            orchestrator's agent catalog whose cards ``can_be_hired``, sorted.
            Empty when the agent cannot hire.
        can_hire: Whether this agent can hire through routing at all — whether
            its command registry has ``hire_member``. Kept apart from
            ``hireable_roles`` so the model is told "you cannot hire" rather than
            "this role is not hireable" when the answer is the former.
        self_name: The asking agent's own name, left out when the directory is
            shown to the model.
    """

    members: list[str]
    hireable_roles: list[str]
    can_hire: bool
    self_name: str

    def reaches(self, recipient: str) -> bool:
        """Whether a message to ``recipient`` would be delivered.

        Args:
            recipient: An ``@name``, or a role to hire.

        Returns:
            ``True`` for a current member's ``@name`` or a hireable role.
        """
        if recipient.startswith("@"):
            return recipient in self.members
        return recipient in self.hireable_roles

    def correction(self, unreachable: list[str]) -> str:
        """The retry prompt: why each recipient failed, then what to address instead.

        One line per recipient, because the three failures call for different
        corrections: an ``@name`` nobody has, a bare role when this agent cannot
        hire, and a bare role that is not hireable. The closing instruction
        depends only on whether the agent can hire, so a model that cannot is
        never invited to try another role.

        Args:
            unreachable: The recipients nobody answers to.

        Returns:
            The message handed to the model with ``ModelRetry``.
        """
        lines = ["Nothing was sent."]
        lines += [f"- {self._why(recipient)}" for recipient in unreachable]
        others = ", ".join(name for name in self.members if name != self.self_name)
        if self.can_hire and self.hireable_roles:
            lines.append(
                f"Address a member by exact @name: {others or '(none)'} — or write one of "
                f"these roles without '@' to hire it: {', '.join(self.hireable_roles)}."
            )
        else:
            lines.append(
                f"Address every message to a current member by exact @name: {others or '(none)'}."
            )
        return "\n".join(lines)

    def _why(self, recipient: str) -> str:
        if recipient.startswith("@"):
            return f"No team member is called '{recipient}'."
        if not self.can_hire:
            return (
                f"'{recipient}' has no '@', so it was read as a role to hire, and you cannot "
                f"hire in this team."
            )
        return (
            f"'{recipient}' has no '@', so it was read as a role to hire, and it is not a "
            f"role you can hire."
        )


@runtime_checkable
class DirectorySource(Protocol):
    """What the check needs from the run's ``deps``: a current team directory."""

    def team_directory(self) -> TeamDirectory:
        """Return who a recipient can name right now."""
        ...


def structured_output_recipients(output: object) -> list[str] | None:
    """The recipients of a ``StructuredOutput``; ``None`` for any other shape.

    Args:
        output: The validated output of a run.

    Returns:
        One recipient per ``Request``, or ``None`` when ``output`` is not a
        ``StructuredOutput``.
    """
    if isinstance(output, StructuredOutput):
        return [request.recipient for request in output.messages]
    return None


@dataclass
class RecipientCheckCapability(AbstractCapability[Any]):
    """Send an output back, inside the run, while it names a recipient nobody answers to.

    Runs on ``after_output_validate``. The run's ``deps`` must be a
    :class:`DirectorySource` — ``BaseAgent`` is one; with any other ``deps`` the
    check does nothing.

    Attributes:
        recipients: Reads the recipients off an output. Defaults to
            :func:`structured_output_recipients`. An agent routing its own shape
            mounts a second instance with its own extractor.
    """

    recipients: RecipientExtractor = structured_output_recipients

    # The Any below is pydantic-ai's own hook signature: deps are narrowed at
    # runtime to a DirectorySource, and the output is any shape an extractor reads.
    async def after_output_validate(
        self,
        ctx: RunContext[Any],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Raise ``ModelRetry`` while a recipient is unreachable and a retry is possible.

        Args:
            ctx: The run context; ``ctx.deps`` is the agent.
            output_context: Unused.
            output: The validated output.

        Returns:
            ``output`` unchanged when every recipient is reachable, when the output
            is not a shape this check reads, or when the run could not honour a
            retry — routing then skips what is unreachable.

        Raises:
            ModelRetry: Naming the unreachable recipients, the members and the
                hireable roles.
        """
        if ctx.partial_output or not can_retry(ctx) or not isinstance(ctx.deps, DirectorySource):
            return output
        named = self.recipients(output)
        if not named:
            return output
        directory = ctx.deps.team_directory()
        unreachable = [recipient for recipient in named if not directory.reaches(recipient)]
        if unreachable:
            raise ModelRetry(directory.correction(unreachable))
        return output


def can_retry(ctx: RunContext[Any]) -> bool:
    """Whether a ``ModelRetry`` raised now would get another model request.

    Two ways it would not. With the output retries spent, the run ends in
    ``UnexpectedModelBehavior``. With the run's request limit reached — always the
    case in akgentic-llm's tool-free conclusion, which allows one request — it ends
    in a usage-limit error. Either would turn an answer that can still be delivered
    in part into a failed turn.

    Args:
        ctx: The run context.

    Returns:
        ``True`` when both the retry budget and the request budget allow one more.
    """
    if ctx.retry >= ctx.max_retries:
        return False
    limit = ctx.usage_limits.request_limit if ctx.usage_limits else None
    return limit is None or ctx.usage.requests < limit
