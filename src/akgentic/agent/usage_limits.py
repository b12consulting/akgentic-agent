"""The usage-limit tier policy, and the pieces agent classes share to apply it.

Deliberately free of any import from ``agent.py``: these are the pieces a *new*
agent class needs, so a dependency in that direction would make them unusable
from the module that defines the base class. What each piece needs from an agent
is stated structurally — as a Protocol, or as an explicit argument — never by
naming a class.

The team's addressing convention lives in ``util.py``: a breached turn must never
be handed a placeholder recipient, because anything without a leading ``@`` is a
role to hire.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, NoReturn, ParamSpec, Protocol, TypeVar

from akgentic.core import BaseConfig
from akgentic.core.agent import WarningError
from akgentic.core.messages import Message
from akgentic.llm import AgentUsageLimitError, ReactAgent, RunUsageLimitError
from akgentic.llm import UsageLimitError as LLMUsageLimitError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
MessageT = TypeVar("MessageT", bound=Message)
OutputT = TypeVar("OutputT")


class AgentLike(Protocol):
    """What the usage-limit policy needs from an agent.

    ``config`` is a read-only property rather than an attribute on purpose: a
    mutable protocol attribute is invariant, so declaring ``config: BaseConfig``
    would reject an agent whose config is a *subclass* of BaseConfig — which is
    every agent in this package.
    """

    _react_agent: ReactAgent

    @property
    def config(self) -> BaseConfig:
        """The agent's public configuration."""
        ...

    def notify_human(self, message: str) -> None:
        """Notify the team's user-proxy member."""
        ...


AgentT = TypeVar("AgentT", bound=AgentLike)
Handler = Callable[Concatenate[AgentT, MessageT, P], None]

# Delivers one LLM output and reports whether anything actually went out.
RouteFn = Callable[[Any, OutputT], bool]


def escalate_usage_limit(agent: AgentLike, error: LLMUsageLimitError) -> NoReturn:
    """Notify the team's human about a usage breach and end the turn.

    Args:
        agent: The agent that breached.
        error: The usage-limit error to report. On a run-tier breach whose
            conclusion failed this is the **original** breach, not whatever the
            failed conclusion raised.

    Raises:
        WarningError: Always.
    """
    agent.notify_human(
        f"The agent {agent.config.name} has exceeded its usage limits ({error}). \n"
        + "Please review the agent's activity and give your instruction."
    )
    raise WarningError(f"LLM usage limit exceeded: {error}")


def try_conclude_without_tools(
    agent: AgentLike,
    error: RunUsageLimitError,
    requester: str | None,
    *,
    output_type: type[OutputT],
    route: RouteFn[OutputT],
) -> None:
    """Turn a run-tier breach into one delivered answer, or report failure.

    Runs exactly one tool-free conclusion through the ReactAgent sync bridge — on
    the actor's own thread, like every other LLM call — asking for ``output_type``
    and delivering it through ``route``, the same routing the normal turn uses.
    There is deliberately **no retry and no counter**: akgentic-llm's agent-tier
    pre-flight consumes lifetime budget before each call, so repeated run-tier
    breaches walk the agent into its terminal tier by construction.

    The reason names the requester because the returned output is LLM-authored:
    the model chooses each recipient, so an answer that does not name them can be
    routed perfectly and still leave the requester with nothing.

    A conclusion that routes nothing is a **failure**, not a quiet success — the
    requester received nothing, exactly as if the call had raised. Whether anything
    went out is ``route``'s answer to give, because only it knows the shape of
    ``output_type``.

    A turn with no identifiable requester is not attempted at all. The reason has
    nobody to name, and a placeholder such as "unknown" would not stay prose: the
    model would echo it as a recipient, and the team's addressing convention treats
    any recipient without a leading ``@`` as a role to **hire** — so the teardown of
    a breached turn would spin up a member called "unknown". Escalation is the
    honest outcome when there is no one to answer.

    Args:
        agent: The agent whose turn was interrupted.
        error: The run-tier breach that interrupted it. This is what every
            escalation below reports, so the human always hears about the
            ORIGINAL breach rather than any secondary failure.
        requester: Name of the requester being answered, or ``None`` when the
            incoming message carried no sender.
        output_type: The schema to ask the conclusion for.
        route: Delivers that output; returns whether anything was actually sent.

    Returns:
        Nothing. A delivered conclusion simply ends the turn — the requester has
        their answer, so there is nothing left to report. Nothing is written to the
        agent's own context either: the reason above already says the turn was cut
        short, and the conclusion's exchange lands in the history like any other
        turn.

    Raises:
        WarningError: When there was no requester, or the attempt raised or
            delivered nothing. Escalation happens here rather than being signalled
            back to the caller, so a caller cannot forget to check a result.
    """
    if requester is None:
        logger.warning(
            "[%s] run-tier usage breach on a message with no sender; "
            "there is no requester to conclude to, escalating instead",
            agent.config.name,
        )
        escalate_usage_limit(agent, error)

    reason = (
        f"This turn has run out of its tool-call budget, so you cannot "
        "call any further tool and this is your last chance to answer.\n"
        f"Answer {requester} now with what you have already gathered. State your "
        "conclusion plainly, say explicitly which parts you could not check or "
        "finish, and do not promise follow-up work — the turn ends with this answer."
    )
    try:
        output = agent._react_agent.conclude_without_tools_sync(
            reason, deps=agent, output_type=output_type
        )
        delivered = route(agent, output)
    except Exception:
        logger.exception(
            "[%s] tool-free conclusion failed after a run-tier usage breach",
            agent.config.name,
        )
        escalate_usage_limit(agent, error)

    if not delivered:
        logger.warning(
            "[%s] tool-free conclusion produced no message; escalating instead",
            agent.config.name,
        )
        escalate_usage_limit(agent, error)


def guard_usage_limits(
    *, output_type: type[OutputT], route: RouteFn[OutputT]
) -> Callable[[Handler[AgentT, MessageT, P]], Handler[AgentT, MessageT, P]]:
    """Run a message handler under the usage-limit tier policy.

    Apply to every ``receiveMsg_*`` handler that can reach the LLM. The handler
    then reads as just the work: the tier policy lives here, written once, and the
    requester is lifted off the handler's own message argument rather than threaded
    through it.

    A run-tier breach — the turn ran out of its own budget while the agent may
    still have lifetime budget — first attempts one tool-free conclusion so the
    requester gets what was already gathered; only if that delivers nothing does it
    escalate, reporting the ORIGINAL breach rather than any secondary error. An
    agent-tier breach is terminal and escalates immediately.

    The clause order below is load-bearing, and is why this is a decorator rather
    than a paragraph each handler copies: both tier errors subclass
    ``LLMUsageLimitError``, so a base clause placed first catches both and the
    branch never runs — a copy that gets it wrong still compiles and still passes
    an ordinary test.

    Args:
        output_type: The schema a tool-free conclusion should be asked for.
        route: Delivers that output; returns whether anything was actually sent.

    Returns:
        A decorator for a handler whose first argument after ``self`` is the
        incoming message. That argument is what the requester is read from.
    """

    def decorate(handler: Handler[AgentT, MessageT, P]) -> Handler[AgentT, MessageT, P]:
        @wraps(handler)
        def wrapper(
            self: AgentT, message: MessageT, /, *args: P.args, **kwargs: P.kwargs
        ) -> None:
            # A message may legitimately carry no sender (Message.sender is
            # optional), in which case there is no requester to answer — kept as
            # None rather than an "unknown" placeholder, which is prose in a prompt
            # but would become a routing target in the tool-free conclusion.
            requester = message.sender.name if message.sender else None

            # ── Usage-limit tiers, most specific FIRST ────────────────────────
            # Both subclasses are listed before LLMUsageLimitError: they inherit
            # from it, so a base clause placed first swallows both tiers silently.
            try:
                handler(self, message, *args, **kwargs)

            except RunUsageLimitError as e:
                # Recoverable: the turn is out of budget, the agent may not be.
                try_conclude_without_tools(
                    self, e, requester, output_type=output_type, route=route
                )

            except AgentUsageLimitError as e:
                # Terminal: the lifetime budget that would pay for a conclusion is
                # exactly the one that is spent. No attempt.
                escalate_usage_limit(self, e)

            except LLMUsageLimitError as e:
                # Backstop, LAST: an akgentic-llm that raises the base class
                # directly must still be handled.
                escalate_usage_limit(self, e)

        return wrapper

    return decorate
