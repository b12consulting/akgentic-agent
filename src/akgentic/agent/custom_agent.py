"""CustomAgent: a worked example of subclassing BaseAgent with your own schema.

Shows what a subclass gets for free when it reasons against a structured output
of its own and handles a message type of its own, and what it has to supply.

Reused unchanged from BaseAgent:

- ``act(user_content, output_type)`` — forwards the type you name to the REACT
  loop, so a custom output model needs no plumbing. This is the whole reason a
  subclass can have its own schema at all.
- ``@guard_usage_limits(output_type=..., route=...)`` — the usage-limit tier
  policy. Decorate every ``receiveMsg_*`` that can reach the LLM; never copy the
  clause ladder into a new handler, because the ordering is load-bearing and a
  wrong copy compiles, passes an ordinary test, and silently stops concluding.
  Because the schema and the routing are arguments, a run-tier breach concludes
  in THIS agent's schema with nothing overridden.
- ``notify_human``, ``send``, ``get_team_member``, ``hire_member`` — no schema in
  their signatures.
- ``MailboxCancelCapability`` — built unconditionally by ``_build_react_agent``,
  so every subclass run is interruptible by a queued ``/stop`` or
  ``CancelMessage``. What the subclass has to supply is the catch: every
  ``receiveMsg_*`` that calls ``act()`` must catch ``RunInterruptedError``
  around that call (notify the human, route nothing, return) — an escape ends
  the turn through the actor failure path instead of the designed clean end.

Supplied here:

- ``TriageOutput`` — the structured output this agent reasons against.
- ``TriageMessage`` — the message type, with its own ``receiveMsg_`` handler.
- ``_route_triage`` — how a TriageOutput is delivered. Passed to the decorator,
  so it serves the normal turn and the interrupted one alike.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from akgentic.agent.agent import BaseAgent, RunInterruptedError
from akgentic.agent.messages import AgentMessage
from akgentic.agent.usage_limits import guard_usage_limits
from akgentic.agent.utils import resolve_recipient
from akgentic.core import ActorAddress
from akgentic.core.messages import Message

logger = logging.getLogger(__name__)


# ============================================================================
# The agent's own structured output
# ============================================================================


class Handoff(BaseModel):
    """One piece of work this agent wants someone else to pick up."""

    recipient: str = Field(description="An @member name, or a role to hire")
    task: str = Field(description="What that recipient should do")


class TriageOutput(BaseModel):
    """What the LLM returns for a triage turn.

    Nothing here is imposed by the framework: the shape is yours, and ``act()``
    hands it to the REACT loop as given.
    """

    severity: Literal["low", "medium", "high"] = "low"
    summary: str = Field(default="", description="One-line assessment of the incident")
    handoffs: list[Handoff] = Field(default_factory=list)


# ============================================================================
# The agent's own message type
# ============================================================================


class TriageMessage(Message):
    """An incident handed to this agent for triage.

    Subclasses ``Message`` rather than ``AgentMessage`` so it carries its own
    fields and its own protocol. Dispatch walks the message class MRO looking
    for ``receiveMsg_<Type>``, so this lands on ``receiveMsg_TriageMessage``
    below with no registration step.
    """

    incident: str
    reported_by: str = "unknown"


# ============================================================================
# The agent
# ============================================================================


class CustomAgent(BaseAgent):
    """A BaseAgent that triages incidents against its own schema."""

    def _route_triage(self, output: TriageOutput) -> bool:
        """Act on a TriageOutput: log the assessment, deliver the handoffs.

        Defined before the handler because the decorator names it as an argument,
        which is evaluated while the class body runs.

        Args:
            output: The triage the LLM produced.

        Returns:
            Whether at least one handoff was delivered — what the usage-limit
            guard needs to tell a real conclusion from an empty one.
        """
        logger.info(
            "[%s] triage: severity=%s — %s", self.config.name, output.severity, output.summary
        )

        delivered = False
        for handoff in output.handoffs:
            member = resolve_recipient(self, handoff.recipient)
            if member is None:
                continue

            self.send(
                member,
                AgentMessage(content=handoff.task, type="request", recipient=member),
            )
            delivered = True

        return delivered

    @guard_usage_limits(output_type=TriageOutput, route=_route_triage)
    def receiveMsg_TriageMessage(  # noqa: N802
        self, message: TriageMessage, sender: ActorAddress
    ) -> None:
        """Handle one incident.

        The decorator owns the usage-limit policy, reads the requester off
        ``message``, and — because it was given this agent's schema and routing —
        concludes a breached turn in TriageOutput without this class overriding
        anything. The one ``except`` this body carries is the
        ``RunInterruptedError`` catch around ``act()`` — the piece every
        ``receiveMsg_*`` that reaches the LLM must supply itself, exactly as
        ``receiveMsg_AgentMessage`` does: a queued cancel ends the run, the
        human is told, nothing is routed, and the handler returns normally.

        Args:
            message: The incident to triage.
            sender: Who sent it.
        """
        prompt = (
            f"Incident reported by {message.reported_by}:\n\n{message.incident}\n\n"
            "Assess severity, summarise in one line, and hand off whatever you "
            "cannot resolve yourself."
        )
        try:
            output = self.act(prompt, TriageOutput)
        except RunInterruptedError:
            logger.info(
                "[%s] run interrupted by a queued cancel; turn abandoned, nothing routed",
                self.config.name,
            )
            self.notify_human("Run interrupted.")
            return

        self._route_triage(output)
