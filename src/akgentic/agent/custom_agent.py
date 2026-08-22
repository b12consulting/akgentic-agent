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
- ``MailboxCapability`` (``akgentic.agent.capabilities``) — built
  unconditionally by ``_build_react_agent``, so every subclass gets all of its
  duties without asking: a queued ``/stop`` or ``CancelMessage`` interrupts the
  run, and mail that arrives mid-run is announced to the model once. Under
  ADR-019 §3 it will also purge the recognised cancel from the mailbox. None of
  it depends on the config carrying a ``MailboxTool``. The interruption is absorbed
  by ``act()`` itself, which notifies the human once and returns a neutral
  instance of the output type you named — so a subclass handler writes nothing
  for it.

Supplied here:

- ``TriageOutput`` — the structured output this agent reasons against.
- ``TriageMessage`` — the message type, with its own ``receiveMsg_`` handler.
- ``_route_triage`` — how a TriageOutput is delivered. Passed to the decorator,
  so it serves the normal turn and the interrupted one alike.
- ``extra_capabilities`` — one pydantic-ai capability of this agent's own,
  ``TriageAuditCapability``. The framework prepends its own, so the list the
  ReactAgent receives is ``[mailbox, audit]``: the cancel check still runs
  first, and this agent never returns the mailbox capability itself.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import AgentCapability, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

from akgentic.agent.agent import BaseAgent
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
# The agent's own capability
# ============================================================================


class TriageAuditCapability(AbstractCapability[Any]):
    """Log one line before every model request — a capability of the agent's own.

    Deliberately the smallest thing a capability can usefully be: it observes
    and returns ``request_context`` unchanged. It constructs no message, mutates
    no part (parts are shared with the durable history) and enqueues nothing, so
    it cannot perturb the run it watches. An observability wrapper, a domain
    guard or a tenant-resolution hook all start from this shape.
    """

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Record that a model request is about to go out, then pass it through."""
        logger.info("[%s] triage audit: model request about to be sent", self._agent_name)
        return request_context


# ============================================================================
# The agent
# ============================================================================


class CustomAgent(BaseAgent):
    """A BaseAgent that triages incidents against its own schema."""

    def extra_capabilities(self) -> list[AgentCapability[Any]]:
        """Contribute this agent's audit capability to the ReactAgent.

        The framework prepends ``MailboxCapability``, so the ReactAgent is
        handed ``[mailbox, audit]`` — cancellation stays unconditional and its
        check still runs first. This override deliberately does not call
        ``super()`` and does not return the mailbox capability: doing either
        would duplicate a capability the framework already supplies.

        Runs from ``_build_react_agent`` during ``on_start``, before
        ``self._react_agent`` exists, so it reads only ``self.config`` — which
        is assigned before ``on_start`` — and nothing built later.

        Returns:
            One capability, appended after the framework's own.
        """
        return [TriageAuditCapability(agent_name=self.config.name)]

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
        anything. This body carries no ``try``/``except``: a queued cancel is
        absorbed by ``act()``, which tells the human and hands back an empty
        ``TriageOutput``, so ``_route_triage`` delivers nothing and the handler
        returns normally — exactly as ``receiveMsg_AgentMessage`` does.

        Args:
            message: The incident to triage.
            sender: Who sent it.
        """
        prompt = (
            f"Incident reported by {message.reported_by}:\n\n{message.incident}\n\n"
            "Assess severity, summarise in one line, and hand off whatever you "
            "cannot resolve yourself."
        )
        output = self.act(prompt, TriageOutput)

        self._route_triage(output)
