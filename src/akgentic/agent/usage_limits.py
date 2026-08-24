"""The usage-limit policy this package applies: notify the team's human, end the turn.

Degradation itself is not here. ``akgentic-llm``'s ``LimitRecoveryCapability``
owns the whole of it — whether a breached turn concludes, with what prompt, and
whether the result was worth returning. By the time a usage-limit error reaches
this module that policy has already run and declined, or run and failed. There is
nothing left to attempt, so this module does the one thing it is placed to do:
tell the human who can act on it.

Deliberately free of any import from ``agent.py``: these are the pieces a *new*
agent class needs, so a dependency in that direction would make them unusable
from the module that defines the base class. What each piece needs from an agent
is stated structurally — as a Protocol — never by naming a class.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, NoReturn, ParamSpec, Protocol, TypeVar

from akgentic.core import BaseConfig
from akgentic.core.agent import WarningError
from akgentic.llm import UsageLimitError as LLMUsageLimitError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class AgentLike(Protocol):
    """What the usage-limit policy needs from an agent.

    ``config`` is a read-only property rather than an attribute on purpose: a
    mutable protocol attribute is invariant, so declaring ``config: BaseConfig``
    would reject an agent whose config is a *subclass* of BaseConfig — which is
    every agent in this package.
    """

    @property
    def config(self) -> BaseConfig:
        """The agent's public configuration."""
        ...

    def notify_human(self, message: str) -> None:
        """Notify the team's user-proxy member."""
        ...


AgentT = TypeVar("AgentT", bound=AgentLike)

LlmCall = Callable[Concatenate[AgentT, P], T]
"""A method that reaches the model: any signature, any return value.

The policy needs neither, which is why one decorator sits on ``act()`` and
``compact()`` alike without either declaring anything for it.
"""


def escalate_usage_limit(agent: AgentLike, error: LLMUsageLimitError) -> NoReturn:
    """Notify the team's human about a usage breach and end the turn.

    Args:
        agent: The agent that breached.
        error: The usage-limit error to report. On a run-tier breach this is the
            **original** breach — ``akgentic-llm`` re-raises it verbatim when its
            own recovery declined or failed, never the secondary error.

    Raises:
        WarningError: Always.
    """
    logger.warning("[%s] usage limit exceeded, notifying the human: %s", agent.config.name, error)
    agent.notify_human(
        f"The agent {agent.config.name} has exceeded its usage limits ({error}). \n"
        + "Please review the agent's activity and give your instruction."
    )
    raise WarningError(f"LLM usage limit exceeded: {error}")


def guard_usage_limits() -> Callable[[LlmCall[AgentT, P, T]], LlmCall[AgentT, P, T]]:
    """Turn any usage-limit breach into a human notification, at the LLM call itself.

    **Apply it to the methods that reach the model, not to message handlers.**
    ``BaseAgent.act`` and ``BaseAgent.compact`` carry it; no ``receiveMsg_*``
    does, and no subclass declares anything at all. That placement is what makes
    the policy unskippable: a new agent class gets it by calling ``act()``, which
    is the only way it can reach the LLM in the first place — rather than by
    remembering to decorate every handler it writes, an obligation invisible from
    this module and silently dropped by the one handler that forgets.

    It also puts the two ways a turn can end early in the same place. ``act()``
    already absorbs ``RunInterruptedError`` itself and hands back a default
    output; a usage breach is the other one, and a caller now writes
    ``try``/``except`` for neither.

    **One clause, both tiers, deliberately.** ``RunUsageLimitError`` and
    ``AgentUsageLimitError`` both subclass ``UsageLimitError``, and this package
    now responds to them identically: notify and stop. Catching the base is
    therefore not a shortcut but the accurate statement — and it removes a real
    trap, because a per-tier version only works while its clauses stay ordered
    most-specific-first, an ordering a copy can get wrong while still compiling
    and still passing an ordinary test.

    **A run-tier breach reaching here has already lost its second chance.**
    ``akgentic-llm`` consults ``LimitRecoveryCapability`` on every run-tier
    breach and, by default, drives one tool-free conclusion through the caller's
    own ``output_type``. Only a declined or failed conclusion re-raises; a
    successful one returns *through* the guarded ``act()`` as an ordinary output
    and routes like any other turn. So the two tiers differ where the difference
    is actionable — inside the LLM — and by the time they arrive here it is spent.

    **Where this is currently too coarse** is a conclusion that succeeds emptily:
    a ``StructuredOutput`` carrying no requests is an ordinary success, so nothing
    raises, nothing is routed, and nobody is notified. ``akgentic-llm`` cannot
    judge that — it sees the output as ``Any`` — and this decorator never sees the
    output at all. Closing it needs a seam on the capability; it is an open
    question on the degradation-boundary decision, not something a wider
    ``except`` can reach.

    Returns:
        A decorator preserving the method's own signature and return value. A
        guarded method returns exactly what it always did; only a usage-limit
        error is intercepted, and that path never returns.
    """

    def decorate(method: LlmCall[AgentT, P, T]) -> LlmCall[AgentT, P, T]:
        @wraps(method)
        def wrapper(self: AgentT, /, *args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return method(self, *args, **kwargs)
            except LLMUsageLimitError as e:
                escalate_usage_limit(self, e)

        return wrapper

    return decorate
