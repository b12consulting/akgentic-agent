"""CustomAgent: a worked example of subclassing BaseAgent with your own schema.

Shows what a subclass gets for free when it reasons against a structured output
of its own and handles a message type of its own, and what it has to supply.

Reused unchanged from BaseAgent:

- ``act(message, output_type)`` — forwards the type you name to the REACT loop,
  so a custom output model needs no plumbing. This is the whole reason a
  subclass can have its own schema at all. It takes the *message*, not a string:
  framing is the message's own ``rendering()``, so a handler composes no
  prompt and there is no second way in that could bypass it.

  **Call it; never override it.** It carries framework mechanism that grows with
  the framework, so an override silently stops performing whatever is threaded
  through it next. To add something around a turn, write a method that *wraps*
  the call — ``act_with_source_tracking`` below is this file's worked example of
  exactly that, and the reason it exists here rather than in a test.
- The usage-limit policy — **nothing to declare, and nothing to remember**. It
  is applied inside ``act()`` itself, so it arrives with the LLM call. A breach
  that akgentic-llm could not degrade notifies the human and ends the turn; one
  it could comes back from ``act()`` as an ordinary ``TriageOutput``, concluded
  in THIS agent's schema because ``act()`` was asked for that schema.
- ``notify_human``, ``send``, ``get_team_member``, ``hire_member`` — no schema in
  their signatures.
- ``MailboxCapability`` (``akgentic.tool.mailbox``) — built
  unconditionally by ``_build_react_agent``, so every subclass gets all of its
  duties without asking: a queued ``/stop`` or ``CancelMessage`` is purged from
  the mailbox and interrupts the run, and mail that arrives mid-run is announced
  to the model once. None of it depends on the config carrying a
  ``MailboxTool``. The interruption is absorbed by ``act()`` itself, which
  notifies the human once and returns a default instance of the output type you
  named — so a subclass handler writes nothing for it.

Supplied here:

- ``TriageOutput`` — the structured output this agent reasons against.
- ``TriageMessage`` — the message type, with its own ``receiveMsg_`` handler and
  its own ``rendering()``. It declares no ``rendering_preview``, which is
  what keeps it out of mid-run mailbox reads.
- ``_route_triage`` — how a TriageOutput is delivered. Called from the handler
  body, and that one call serves the normal turn, the interrupted one, and the
  turn ``akgentic-llm`` concluded after a run-tier breach: all three arrive back
  from ``act()`` as an ordinary ``TriageOutput``.
- ``extra_capabilities`` — one pydantic-ai capability of this agent's own,
  ``TriageAuditCapability``. The framework prepends its own, so the list the
  ReactAgent receives is ``[mailbox, audit]``: the cancel check still runs
  first, and this agent never returns the mailbox capability itself.
- ``act_with_source_tracking`` — the **wrap-don't-override** pattern. It calls
  ``act()`` and does its own work around the result, so it inherits every future
  change to that method instead of freezing today's version of it.
- ``CustomConfig`` and ``CustomState`` — this agent's own configuration and its
  own runtime state, plus the two things a subclass must do to make them real:
  **re-declare the attribute types** (``BaseAgent`` fixes the generics, so
  without that the extra fields are invisible to the type checker), and
  **upgrade the state in ``on_start``** through ``init_state`` (``BaseAgent``
  constructs a plain ``AgentState`` itself, so an annotation alone would be a
  lie). See *Config, state and metadata* below for which of the three a given
  value belongs in.
- ``CustomMetaData`` and ``get_metadata`` — this agent's per-team metadata class,
  and the one place it is narrowed from the opaque type the orchestrator holds.
  See *Team metadata* below.

Config, state and metadata
--------------------------

Three places to put a value, and picking the wrong one is the usual mistake:

==================  ===============================  =========================
Where               Chosen by                        Changing it means
==================  ===============================  =========================
``CustomConfig``    the deployment, at wiring time   redeploying
``CustomState``     the agent itself, during a turn  nothing — it just changes
``CustomMetaData``  whoever creates the team, in a form  a different team
==================  ===============================  =========================

``escalation_threshold`` is the same for every team this class serves, so it is
config. ``triaged_count`` is discovered as the agent works, so it is state — and
it is published to observers on every checkpoint. ``case_id`` differs per team
and is typed into a form at creation, so it is metadata.

Team metadata
-------------

Per-team values — a case id, a tenant — supplied by whoever created the team rather
than written into the code, so one agent class serves many teams.

**There is no framework accessor yet, so a subclass writes its own.**
``Orchestrator.get_metadata()`` exists in ``akgentic-core`` and returns
``SerializableBaseModel | None`` — deliberately opaque, since core cannot know the
class a given team declared. Narrowing it is the subclass's job:
:meth:`CustomAgent.get_metadata` below is the worked example, reaching the value
through the orchestrator proxy and narrowing once so that every handler downstream
reads a concrete type. ADR-27 (``_bmad-output/akgentic-team/decisions/``) replaces
that with ``Akgent.get_metadata()`` and ``Orchestrator.get_metadata_type()``; when it
lands, the example collapses to a call and the loop below is unchanged.

**The TeamCard is what switches the feature on.** Declaring a ``TeamMetadata``
subclass does nothing on its own; the card must name it::

    # in code
    TeamCard(..., metadata_type=SupportCaseMetadata)

    # or in catalog YAML, where a type serialises as a dotted path
    metadata_type:
      __type__: acme.support.SupportCaseMetadata

With no declaration there is no form to fill and nothing is persisted. That is the
whole switch.

The loop, once the card declares it:

1. **The catalog projects the class into a form contract.** Each field becomes a
   descriptor carrying its key, description, whether it is mandatory, its regex
   pattern, and whether it is indexed. The frontend reads this from the namespace
   listing — there is no separate metadata endpoint.
2. **The frontend asks, at creation, and only when there is something to ask.**
   A modal opens before the team is created; it is skipped when the card declares
   no metadata class, *and also* when the projected contract has no fields — so a
   declared-but-empty class is silently no-ask, not an empty dialog. Every field
   renders as free text: the descriptor deliberately carries no type, so the
   server's validation is the one that counts.
3. **The team is created through the infra server**, which validates the submitted
   values against the declared class before the team is placed. A payload for a
   card that declares nothing is rejected; an absent one is fine even for a card
   that declares.
4. **The values are persisted with the team**, alongside a derived array of
   ``"key|value"`` entries — one per *set* indexed field. That array is what each
   backend indexes, and what a metadata filter queries; derivation and query build
   the entry through the same function, so the two sides cannot drift.
5. **The orchestrator is handed the metadata after the write**, and holds it for the
   running team to read.

Two things worth knowing before relying on it:

- **The orchestrator's copy is a cache, not the record.** The persisted team entry
  is authoritative; the in-memory copy is pushed to it and can lag.
- **It is handed out by reference, as the declared subclass.** Do not mutate what you
  read — every reader in the process shares that object, and a change made there is
  written nowhere.
"""

import logging
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field
from pydantic_ai import AgentCapability, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage, LlmRenderable
from akgentic.agent.utils import resolve_recipient
from akgentic.core import ActorAddress
from akgentic.core.messages import Message
from akgentic.team import TeamMetadata

logger = logging.getLogger(__name__)



class CustomMetaData(TeamMetadata):
    """A worked example of a per-team metadata class.

    The fields are arbitrary, and the class is never read by the framework. It is
    a convention: the catalog declares it on the card, the frontend projects it
    into a form, and the agent reads it through the orchestrator proxy.
    """

    case_id: str = Field(
        description="The support case this team is triaging",
        pattern=r"^[a-z][a-z0-9-]{3,31}$",
        json_schema_extra={"indexed": True},
    )
    tenant_id: str = Field(
        description="The tenant this team is triaging for",
    )


# ============================================================================
# The agent's own configuration and runtime state
# ============================================================================


class CustomConfig(AgentConfig):
    """This agent's own configuration — declared per agent, at wiring time.

    **Config is what the deployment chooses; metadata is what the team's creator
    supplies.** Both end up on the running agent and the distinction is easy to
    lose: ``escalation_threshold`` is a property of *this role* and is the same
    for every team the class serves, while :class:`CustomMetaData`'s ``case_id``
    differs per team and is typed into a form. Put a value here when changing it
    means redeploying, and in metadata when changing it means creating a
    different team.

    Nothing needs to be registered. ``Akgent.__init__`` assigns whatever
    ``config`` it is handed straight to ``self.config``, so passing a subclass
    through the ``AgentCard`` is the whole mechanism::

        AgentCard(
            role="Triage",
            agent_class=CustomAgent,
            config=CustomConfig(prompt=..., escalation_threshold=5),
        )

    The one thing to get right is the **declared type** on the subclass, below —
    without it every read is typed ``AgentConfig`` and the new fields are
    invisible to the type checker even though they are there at runtime.
    """

    escalation_threshold: int = Field(
        default=3,
        ge=1,
        description="How many handoffs a turn may request before this agent escalates instead",
    )
    triage_queue: str = Field(
        default="general",
        description="Which queue this agent's handoffs are filed against",
    )


class CustomState(AgentState):
    """This agent's own runtime state — what survives between turns.

    ``AgentState`` already carries ``backstory`` and ``tool_state``; a subclass
    adds what its own handlers need to remember. State is **published**: every
    ``notify_if_changed()`` checkpoint sends it to the orchestrator, so a field
    added here becomes visible to observers and to a restored team without any
    further wiring.

    Two rules, both learned the hard way:

    - **Never enumerate the inherited fields when building one of these.** See
      ``CustomAgent.on_start`` — the upgrade spreads ``model_dump()`` rather than
      naming ``backstory=`` and ``tool_state=``, so a field added to
      ``AgentState`` later is carried across instead of being silently dropped.
    - **Never hold a reference to ``self.state`` across turns.** Restore replaces
      the whole object, so a cached handle goes stale in silence. Read it through
      ``self.state`` every time.
    """

    triaged_count: int = Field(
        default=0,
        description="How many triage turns this agent has completed",
    )
    last_handoff_to: str | None = Field(
        default=None,
        description="The recipient of the most recent handoff, or None if there has been none",
    )


# ============================================================================
# The agent's own structured output
# ============================================================================


class TrackingOutput(BaseModel):
    """A base output schema that asks the model to name what it drew on.

    Nothing in the framework knows about this class. It is one deployment's own
    convention: put a common field on a base model, have every output schema
    inherit it, and a wrapper around ``act()`` can then read that field off any
    of them without knowing which one it got — which is what
    :meth:`CustomAgent.act_with_source_tracking` does.

    The pattern generalises past this example. Whenever several output types need
    the same post-turn treatment, giving them a shared base is what lets one
    wrapper serve all of them, and it is why the wrapper can be typed
    ``type[T]`` bound to this class rather than to one concrete schema.
    """

    sources: list[str] = Field(
        default_factory=list,
        description="Ids of the messages this answer drew on",
    )


T = TypeVar("T", bound=TrackingOutput)
"""Any output schema carrying the tracking field.

Bound to ``TrackingOutput`` rather than to ``BaseModel`` on purpose: the wrapper
below READS ``output.sources``, so the bound is what makes that read type-safe.
With a ``BaseModel`` bound the read is unchecked, and the only way to defend it
is a runtime ``assert issubclass(...)`` — a check mypy cannot see, that fires at
call time rather than at the call site, and that is stripped entirely under
``python -O``.
"""


class Handoff(BaseModel):
    """One piece of work this agent wants someone else to pick up."""

    recipient: str = Field(description="An @member name, or a role to hire")
    task: str = Field(description="What that recipient should do")


class TriageOutput(TrackingOutput):
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

    It declares ``rendering()`` below and so satisfies ``LlmRenderable`` — the whole
    point of keying that contract on a method: this class has no ``content``
    field and was never going to grow one. It declares **no**
    ``rendering_preview``, so it is never offered for a mid-run read: a triage run
    is not a place to absorb unrelated mail, and a class that cannot render a
    preview must not be handed an id it cannot honour.
    """

    incident: str
    reported_by: str = "unknown"

    def rendering(self) -> str:
        """The incident, framed as the triage prompt this agent reasons against."""
        return (
            f"Incident reported by {self.reported_by}:\n\n{self.incident}\n\n"
            "Assess severity, summarise in one line, and hand off whatever you "
            "cannot resolve yourself."
        )


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

    config: CustomConfig
    """Narrow the declared type, or the extra fields are invisible to mypy.

    ``BaseAgent`` fixes the generics — ``Akgent[AgentConfig, AgentState]`` — so
    without this line ``self.config`` reads as ``AgentConfig`` and
    ``self.config.escalation_threshold`` is an error, even though the object
    handed in through the ``AgentCard`` really is a ``CustomConfig``. The
    re-declaration is annotation-only: it assigns nothing and changes no
    behaviour, it just tells the checker what ``__init__`` already stored.

    It is a promise, not a guarantee. Nothing stops a caller passing a plain
    ``AgentConfig``; the card is where that is got right.
    """

    state: CustomState
    """Same narrowing, for the same reason — but state needs a real upgrade too.

    ``BaseAgent.on_start`` constructs an ``AgentState`` unconditionally, so this
    annotation alone would be a lie. :meth:`on_start` below replaces it.
    """

    def get_metadata(self) -> CustomMetaData:
        """Read the per-team metadata this agent was handed at team creation.

        The framework never reads this class; the orchestrator holds it as an
        opaque ``SerializableBaseModel | None``. Narrowing it is therefore the
        subclass's job, and doing it **here, once** is the point of the example:
        every handler downstream reads ``.case_id`` off a concrete type, with no
        runtime check and no ``getattr`` of its own.

        Raising rather than returning ``None`` is a deliberate choice for a
        worked example. A team whose card declares the wrong metadata class — or
        none — is misconfigured, and the useful moment to learn that is the first
        read, naming both the expected and the actual type. Returning ``None``
        would push an ``if`` into every caller and defer the diagnosis to
        wherever the missing value first mattered.

        Returns:
            This team's metadata, narrowed to ``CustomMetaData``.

        Raises:
            TypeError: If the orchestrator holds no metadata, or metadata of a
                different class than this agent expects.
        """
        metadata = self.orchestrator_proxy_ask.get_metadata()
        if isinstance(metadata, CustomMetaData):
            return metadata

        raise TypeError(
            f"team_id ${self.team_id} - "
            f"get_metadata expects {CustomMetaData.__name__}, got {type(metadata).__name__}"
        )

    def on_start(self) -> None:
        """Boot as ``BaseAgent`` does, then upgrade the state to this agent's own.

        The order is forced. ``BaseAgent.on_start`` assigns
        ``AgentState(backstory=...)`` itself, so the upgrade cannot come first —
        it would be overwritten. It happens **after** ``super()`` and goes
        through :meth:`~akgentic.core.agent.Akgent.init_state`, which re-attaches
        the outgoing state's observer and stamps the change-detection baseline.
        Assigning ``self.state`` directly would drop the observer and the agent
        would stop publishing state changes — silently, and only visibly on a
        restore.

        **The spread is the point, and it is not stylistic.** Naming
        ``backstory=self.state.backstory, tool_state=self.state.tool_state``
        would be correct today and would silently destroy any field added to
        ``AgentState`` tomorrow — no error, no failing test, the value simply
        gone on whichever path ran. Spreading carries every inherited field
        across by construction, so the only names written here are the ones this
        subclass is actually adding.

        **``dict(state)``, never ``state.model_dump()``.** These models are
        ``SerializableBaseModel``s, whose serializer injects a ``__model__`` key
        naming the class it dumped. Spread into a *different* class, that key
        makes the before-validator rebuild the **old** class and validation then
        rejects it outright — so the ``model_dump()`` form does not merely lose
        something subtle, it raises. ``dict(state)`` yields the raw field values,
        which also preserves ``tool_state``'s object identity instead of
        round-tripping it through a serialize/deserialize pair.

        Safe with respect to the context-update engine, which ``super()`` builds
        just after assigning state: it reads ``state.tool_state`` live through
        this agent on every call rather than caching a handle, so swapping the
        state object underneath it is fine.
        """
        super().on_start()
        self.init_state(
            CustomState(
                **dict(self.state),
                triaged_count=0,
                last_handoff_to=None,
            )
        )

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

    def act_with_source_tracking(self, message: LlmRenderable, output_type: type[T]) -> T:
        """Run one turn, then log which messages its answer drew on.

        **This is the pattern to copy when you need to extend a turn: wrap
        ``act()``, do not override it.** ``act()`` carries framework mechanism
        that grows with the framework — context-update delivery, the message's
        own framing, media expansion, the usage-limit guard, the interruption
        absorption — and an override freezes today's version of that list. A
        wrapper inherits every future addition for free, and the only thing it
        has to get right is its own work.

        Two things it deliberately does NOT do, both of which an override would
        have had to reimplement: it does not touch the ReactAgent, and it writes
        no ``try``/``except``. A queued cancel and a usage breach are both
        handled inside ``act()``; a cancelled turn simply returns a default
        ``output_type()`` here, whose empty ``sources`` logs as empty.

        Args:
            message: The message to reason about, passed straight through.
            output_type: The schema to reason against. Bound to ``TrackingOutput``
                so ``output.sources`` is guaranteed to exist — the wrapper reads a
                field, so the bound is what makes reading it safe rather than a
                hopeful ``getattr``.

        Returns:
            The turn's output, unchanged. A wrapper that swallowed or rewrote it
            would be an override in disguise.
        """
        output = self.act(message, output_type)

        logger.info(
            "[%s] answered from sources: %s",
            self.config.name,
            output.sources,
        )

        return output

    def _route_triage(self, output: TriageOutput) -> bool:
        """Act on a TriageOutput: log the assessment, deliver the handoffs.

        Split out from the handler rather than inlined so the three ways a turn
        can end — a normal one, an interrupted one, and one ``akgentic-llm``
        concluded after a usage breach — all route through a single path. All
        three arrive back from ``act()`` as an ordinary ``TriageOutput``.

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

    def receiveMsg_TriageMessage(  # noqa: N802
        self, message: TriageMessage, sender: ActorAddress
    ) -> None:
        """Handle one incident.

        ``act()`` owns the usage-limit policy — notify the human, end the turn —
        and needs nothing from this agent to do it. Concluding a breached turn is
        ``akgentic-llm``'s, and it too needs nothing declared here: it reuses the
        ``output_type`` this body already asks for, so the conclusion comes back
        as a ``TriageOutput`` and routes below like any other.

        This body carries no ``try``/``except``: a queued cancel is absorbed by
        ``act()``, which tells the human and hands back an empty ``TriageOutput``,
        so ``_route_triage`` delivers nothing and the handler returns normally —
        exactly as ``receiveMsg_AgentMessage`` does.

        The turn goes through :meth:`act_with_source_tracking` rather than
        ``act()`` directly, which is what makes the sources log happen, and the
        case id comes from :meth:`get_metadata` — so a misconfigured team fails
        here, on the first read, rather than somewhere further downstream.

        Args:
            message: The incident to triage.
            sender: Who sent it.
        """
        output = self.act_with_source_tracking(message, TriageOutput)

        case_id = self.get_metadata().case_id
        print(
            f"[{self.config.name}] triage for case {case_id} concluded: "
            f"severity={output.severity}, summary={output.summary}, "
            f"sources={output.sources}, handoffs={len(output.handoffs)}"
        )

        self._route_triage(output)
