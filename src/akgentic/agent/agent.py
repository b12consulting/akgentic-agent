"""BaseAgent: LLM-powered team agent with delegation and collaboration.
BaseAgent integrates with akgentic-llm's ReactAgent for all LLM management,
with team-specific message handling and a static structured-output schema.

Architecture:
- Extends Akgent[AgentConfig, AgentState] from akgentic-core (pykka actor model)
- Composes ReactAgent (akgentic-llm) for model, http client, context, usage limits
- Composes ToolFactory (akgentic-tool) aggregating ToolCard[] into 4 channels:
  · TOOL_CALL — LLM-callable tools (hire_members, fire_members, + config.tools)
  · SYSTEM_PROMPT — static prompts rendered once into the frozen system block
  · LLM_CONTEXT — volatile team state, one per-turn **Context update** block,
    composed by akgentic-tool's ContextUpdater against baselines persisted on
    AgentState.tool_state; this class only decides when to deliver and how to append
  · COMMAND — commands via a CommandRegistry, dispatched from /-prefixed messages
- TeamTool and MailboxTool auto-injected if not already in config.tools
- ReactAgent.run_sync(output_type=...) — act() forwards the caller's type;
  receiveMsg_AgentMessage asks for StructuredOutput, so the team path stays
  schema-driven
- get_output_type() applied inside ReactAgent.run() — no leakage into BaseAgent
- Implements TeamManagementToolObserver and ModelSwitchToolObserver protocols
  (structural typing). The latter is what makes ModelTool's roster listing and
  runtime switch reach a real roster, and what re-applies a persisted selection
  at the top of act() so a resumed agent answers on the model it was switched to
- Two message handlers of its own: receiveMsg_AgentMessage carries all team
  traffic (every message is an AgentMessage), and receiveMsg_CancelMessage
  acknowledges a cancel that lands while the agent is idle. Akgent still
  contributes the lifecycle handler receiveMsg_StopRecursively
- Mailbox-driven run control (ADR-040): MailboxCapability, from
  akgentic.tool.mailbox, peeks the mailbox before every model request and
  does two things with what it finds — it purges a pending /stop or
  CancelMessage and raises RunInterruptedError on it, and it renders the mid-run
  arrival notice for mail not yet announced this run and enqueues it for the
  next request. act() absorbs the interruption, notifies the human and returns a
  default output_type() — an empty StructuredOutput on the team path, which
  routes nothing, so no handler carries a catch. The run dies, the agent
  survives. The agent owns both the cancel vocabulary and its enforcement: the
  capability is built unconditionally, so it must work on an agent configured
  without MailboxTool
- Custom capabilities: extra_capabilities() is the subclass override. The
  framework prepends its own capability, so self._capabilities is always
  [mailbox, *extra_capabilities()] and cancellation cannot be de-configured
- The usage-limit policy is applied, not written here, and not by any handler:
  @guard_usage_limits() from usage_limits.py sits on act() and compact() — the
  two methods that reach the model — so a subclass gets it by calling act()
  rather than by declaring anything. See custom_agent.py, which declares nothing
- Delegation is a plain send per Request in the LLM's StructuredOutput. Each hop
  is an independent turn — no call stack, no automatic return path
"""

import logging  # noqa: I001
import os
import random
from datetime import datetime, timezone
from time import sleep
from typing import Any, TypeVar, cast

from pydantic_ai import AgentCapability, BinaryContent, ModelRetry, RunContext

from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage, LlmRenderable
from akgentic.agent.output_models import StructuredOutput
from akgentic.agent.usage_limits import guard_usage_limits
from akgentic.agent.utils import resolve_recipient
from akgentic.core import ActorAddress, Akgent, Orchestrator
from akgentic.core.messages import CancelMessage, EventMessage, Message
from akgentic.llm import (
    AgentUsageSummary,
    LlmUsageEvent,
    ModelSwitchError,
    ReactAgent,
    ReactAgentConfig,
    UserPrompt,
    aggregate_usage,
    model_roster_key,
)
from akgentic.tool.core import CommandRegistry, ContextUpdater, ToolFactory
from akgentic.tool.errors import CommandNotRecognized, RetriableError
from akgentic.tool.core.event import CommandsAnnouncedEvent
from akgentic.tool.mailbox import MailboxCapability, MailboxTool, RunInterruptedError
from akgentic.tool.model import ModelRow
from akgentic.tool.team import TeamTool
from akgentic.tool.workspace.readers import MediaContent

logger = logging.getLogger(__name__)


T = TypeVar("T")


class BaseAgent(Akgent[AgentConfig, AgentState]):
    """LLM-powered team agent with delegation and collaboration capabilities.

    Composition:
    - ReactAgent (akgentic-llm): model instantiation, HTTP retry, usage limits,
      context history, REACT loop. Reasoning turns go through run_sync();
      compact() and clear() bypass it — compact() is ReactAgent's own synchronous
      bridge onto the agent loop, clear() a plain wrapper over the context.
    - ToolFactory (akgentic-tool): aggregates ToolCard[] into tools, prompts, context
      states and commands — TOOL_CALL, SYSTEM_PROMPT, LLM_CONTEXT, COMMAND.
    - TeamTool: auto-injected if absent from config.tools; provides hire/fire
      capabilities and team-awareness context state.
    - MailboxTool: auto-injected if absent from config.tools; a two-channel
      card serving read_mailbox on TOOL_CALL and /stop on COMMAND, and no
      LLM_CONTEXT at all. The read *consumes*: it absorbs the mail it shows,
      which is therefore not delivered again as its own turn, while anything
      left unread stays queued and arrives as its own turn after the run. A
      pending cancel is never consumed by it. A card supplied in config.tools
      wins over the default.
    - MailboxCapability: built unconditionally in _build_react_agent (no
      card involvement, no MailboxTool presence check) and handed to the
      ReactAgent via capabilities= — the cancel check and the mid-run arrival
      notice before every model request. act() resets its run-local tracking.
    - extra_capabilities(): the subclass override for pydantic-ai capabilities
      of your own. The framework prepends its own, so the list handed to the
      ReactAgent is always [mailbox, *extra_capabilities()].

    Observer Protocol:
    - Implements TeamManagementToolObserver (structural typing via @runtime_checkable)
    - Provides createActor(), on_hire(), on_fire(), proxy_ask(), notify_event()
      to ToolFactory/TeamTool without explicit interface inheritance.

    Execution in act():
    - ReactAgent.run_sync(output_type=<the caller's type>). act() forwards its
      own output_type argument unchanged; ReactAgent.run() wraps it with
      get_output_type() internally and manages context, limits, and the REACT
      loop. receiveMsg_AgentMessage passes StructuredOutput, which is why the
      team delegation path is schema-driven.

    Message Flow:
    - This class defines two handlers: receiveMsg_AgentMessage for all team
      traffic, and receiveMsg_CancelMessage acknowledging a cancel that lands
      while idle (Akgent contributes receiveMsg_StopRecursively). /-prefixed
      content is offered to the CommandRegistry first; everything else —
      including a /-prefixed token the registry does not recognise — is run as
      one act() turn, framed by the message's own rendering(), which is
      where the reply protocol for its type lives.
    - A turn interrupted by a queued cancel never reaches a handler:
      act() absorbs the RunInterruptedError itself, calls
      notify_human("Run interrupted.") once and returns an empty
      StructuredOutput, which _route_output delivers as nothing. No
      receiveMsg_* — here or in a subclass — writes a catch. The context
      arrives already healed (akgentic-llm repairs dangling tool calls before
      re-raising), so nobody performs context surgery, and the agent survives
      to process the next queued message — which is ordinary mail, the cancel
      itself having been purged at recognition rather than left to dequeue.
    - That turn's StructuredOutput goes to _route_output(), which sends one
      AgentMessage per Request. A recipient starting with "@" resolves to an
      existing member; anything else is hired by role. A recipient that resolves
      to None is skipped.
    - A usage breach that reaches this class has already exhausted its second
      chance: akgentic-llm's LimitRecoveryCapability drives one tool-free
      conclusion on a run-tier breach, and that conclusion — when it succeeds —
      returns through act() as an ordinary StructuredOutput and routes through
      that same _route_output(). Only a declined or failed conclusion, and every
      agent-tier breach, surfaces as an error; the @guard_usage_limits decorator
      then notifies the human and raises WarningError. No tier branch is left in
      this package.

    Structured Output:
    - One type: StructuredOutput (output_models.py), a list of Request, each
      carrying message_type, message and recipient. An empty list = no delegation.

    Tools exposed to LLM (via ToolFactory.get_tools()):
    - hire_members(roles: list[str]) → str
    - fire_members(names: list[str]) → str
    - Additional tools from config.tools ToolCards

    System Prompts (all registered in on_start; ReactAgent registers none):
    - agent_backstory (from AgentState.backstory)
    - current_date
    - whatever ToolFactory.get_system_prompts() yields — nothing on a default
      card set: the roster, role profiles, planning and knowledge-graph
      capabilities declare LLM_CONTEXT and arrive as context-update blocks.
      MailboxTool declares no LLM_CONTEXT at all; mailbox awareness reaches
      the model through the mid-run arrival notice alone

    Commands (programmatic, via CommandRegistry built in on_start):
    - A single generic CommandRegistry holds every COMMAND-channel callable keyed
      by its canonical name (e.g. "hire_member", "fire_member", "_expand_media_refs").
    - Humans reach commands through registry.dispatch("/<name> ...") (string surface);
      /-prefixed messages are intercepted in receiveMsg_AgentMessage. Unknown leading
      tokens raise CommandNotRecognized → fall back to the normal LLM path.
    - In-agent code reaches commands through registry.callable("<name>")(...) (typed
      surface, native return) — see hire_member() and act()'s media expansion.
    - on_start emits exactly one CommandsAnnouncedEvent so services can discover the
      command set without per-command coupling.

    Context updates (LLM_CONTEXT channel):
    - One ContextUpdater, obtained from the factory in on_start and held for the
      agent's lifetime, composes at most one block per turn. Its baselines and
      block counter live on AgentState.tool_state, so they are persisted and
      restored with the rest of the state — a restored agent whose history is
      intact resumes delta delivery instead of re-snapshotting.
    - This class holds no baseline state of its own and composes no block text.
      It owns only the delivery: _deliver_context_update() at the top of act(),
      appending through append_user_prompt.

    Internal method (used by _route_output()):
    - hire_member(role) → ActorAddress. A failed hire raises ModelRetry; see
      hire_member() for where that retry is, and is not, honoured.

    Model switching (ModelSwitchToolObserver):
    - list_model_rows() and switch_model() are the two methods ModelTool calls.
      The ModelConfig → ModelRow mapping lives here and only here: this is the
      one package that may import both akgentic-llm and akgentic-tool, so
      neither of them gains an import edge to the other.
    - _restore_active_model() re-applies the persisted selection at the top of
      act(), before the turn's context-update block. ModelTool is NOT
      auto-injected — an agent gets the card only because its card list says so.
    """

    _restored_model_key: str | None = None
    """Which persisted preference has already been re-applied — never what model is in force.

    A class attribute with an immutable default, so ``on_start`` gains no line
    and every construction path (fresh, resumed, subclassed) starts from ``None``.
    See :meth:`_restore_active_model` for the distinction this stores.
    """

    def on_start(self) -> None:
        """Initialize BaseAgent using ReactAgent from akgentic-llm.

        ReactAgent internally handles:
        - create_model() / create_model_settings() / create_http_client()
          — that client is an ``httpx2.AsyncClient``, not encode's ``httpx``
        - ContextManager for conversation history
        - Usage limits conversion

        Every dynamic system prompt is registered here, after construction, via
        @self._react_agent.system_prompt: agent_backstory, current_date, and
        whatever ToolFactory.get_system_prompts() yields (nothing on a default
        card set — the volatile capabilities declare LLM_CONTEXT). ReactAgent
        registers none of its own.

        Tools (hire_members, fire_members) come from TeamTool.get_tools() as
        closures over the orchestrator proxy — not bound methods of this agent —
        and take no RunContext, so pydantic-ai treats them as plain tools.
        Commands are aggregated into a single generic CommandRegistry, held for the
        agent's lifetime, and announced once via a CommandsAnnouncedEvent.

        The context-update engine is built the same way and for the same
        lifetime, but strictly after self.state is assigned: it is handed this
        agent as an ActorToolObserver and reads state.tool_state — where the
        baselines and the block counter persist — live on every call.
        """
        assert self._orchestrator is not None, "Orchestrator address must be provided in config"
        self.orchestrator_proxy_ask = self.proxy_ask(self._orchestrator, Orchestrator)

        # ── State ───────────────────────────────────────────────────────────────
        self.state = AgentState(backstory=self.config.prompt.render()).observer(self)

        # ── Add TeamTool and MailboxTool automatically (without mutating config) ──
        # Both intrinsic cards are hardcoded in akgentic-agent package; a card
        # already present in config.tools wins over the prepended default.
        tool_cards = list(self.config.tools)
        mailbox_card = next((t for t in tool_cards if isinstance(t, MailboxTool)), None)
        if mailbox_card is None:
            mailbox_card = MailboxTool()
            tool_cards.insert(0, mailbox_card)
        if not any(isinstance(t, TeamTool) for t in tool_cards):
            tool_cards.insert(0, TeamTool())

        # ── ReactAgent: wraps model, http client, context, usage limits ──────
        # Tools come from ToolFactory (includes TeamTool hire/fire via factory pattern)
        # result_type=str is the default; structured calls use pydantic_agent.iter()
        # with a per-call output_type override.
        react_agent_config = ReactAgentConfig(
            model_cfg=self.config.model_cfg,
            model_roster=self.config.model_roster,
            runtime_cfg=self.config.runtime_cfg,
            run_usage_limits=self.config.run_usage_limits,
            agent_usage_limits=self.config.agent_usage_limits,
            compaction_cfg=self.config.compaction_cfg,
        )

        tool_factory = ToolFactory(
            tool_cards=tool_cards,
            observer=self,
            retry_exception=ModelRetry,
        )
        tools = tool_factory.get_tools()
        toolsets = tool_factory.get_toolsets()

        # ── Build the generic command registry and announce it once ────────
        # /compact and /clear join as command-only built-ins (never TOOL_CALL).
        # Bound methods are captured here but invoked only at dispatch time, by
        # which point self._react_agent (built below) exists.
        self._command_registry: CommandRegistry = tool_factory.get_command_registry(
            extra_commands=[self.compact, self.clear]
        )
        self.notify_event(
            CommandsAnnouncedEvent(
                agent=self.myAddress,
                commands=self._command_registry.descriptors(),
            )
        )

        # ── Context-update engine (LLM_CONTEXT channel) ────────────────────────
        # One updater for the agent's lifetime, like the command registry. It is
        # built after self.state is assigned: the factory isinstance-checks this
        # agent against ActorToolObserver, and the engine reads
        # observer.state.tool_state live on every call — the baselines and the
        # block counter persist in that slot, so a restored agent resumes delta
        # delivery instead of re-snapshotting. The engine itself lives in
        # akgentic-tool, which owns the semantics it encodes.
        self._context_updater: ContextUpdater = tool_factory.get_context_updater()

        self._capabilities = self._assemble_capabilities(mailbox_card)

        self._react_agent = self._build_react_agent(
            react_agent_config, self._capabilities, tools, toolsets
        )

        # ── Dynamic system prompts ────────────────────────────────────────────
        # ReactAgent registers none of its own: its system_prompt is a bare
        # decorator wrapper over pydantic-ai, and ReactAgentConfig has no
        # system_prompts field. Everything the model sees is registered below.
        @self._react_agent.system_prompt
        def agent_backstory(ctx: RunContext[BaseAgent]) -> str:
            return ctx.deps.state.backstory

        @self._react_agent.system_prompt
        def current_date(ctx: RunContext[BaseAgent]) -> str:
            now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            return f"The current date is {now}."

        for system_prompt in tool_factory.get_system_prompts():
            self._react_agent.system_prompt(system_prompt)

    def on_stop(self) -> None:
        """Release LLM resources on stop, then run the base teardown.

        Delegates teardown to the ReactAgent's synchronous, idempotent
        ``close()`` — the agent owns and closes its own loop now, so BaseAgent
        no longer drives ``aclose()`` on an actor loop. ``super().on_stop()``
        always runs last so the core StopMessage telemetry fires.
        """
        try:
            self._react_agent.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.exception("[%s] ReactAgent.close() failed on stop", self.config.name)
        super().on_stop()

    def extra_capabilities(self) -> list[AgentCapability[Any]]:
        """Contribute pydantic-ai capabilities of your own. Override point.

        Returns an empty list here. A subclass returning capabilities gets them
        appended to the framework's, in the order returned, at both build sites:
        ``self._capabilities = [self._mailbox_capability, *self.extra_capabilities()]``.

        **Never return the mailbox capability from this hook.** The framework
        prepends it, for two reasons that are the whole point of the split:

        - Cancellation is unconditional (ADR-040 §5). A subclass that forgot to
          call ``super()`` would otherwise silently lose the ability to be
          stopped.
        - The cancel check runs before any custom capability's work. Hook order
          is registration order, so a run that is about to be cancelled does not
          first pay for a third party's ``before_model_request``.

        Called from ``on_start``, before ``self._react_agent`` exists.
        ``self.config`` is assigned before ``on_start`` and is safe to read; an
        override must not touch the ReactAgent, and must not assume anything
        built later in ``on_start``.

        Returns:
            Capabilities to append after the framework's own. Both an
            ``AbstractCapability`` instance and a plain capability function are
            accepted — ``AgentCapability`` is the union of the two.
        """
        return []

    def _assemble_capabilities(self, mailbox_card: MailboxTool) -> list[AgentCapability[Any]]:
        """Build the run's capability stack: the framework's own, then the subclass's.

        Separate from ``_build_react_agent`` because the two need different
        things. This needs the ``MailboxTool`` card; the builder needs only the
        finished list. Keeping them apart is what lets the builder be a pure
        function of its arguments.

        **The card is handed over whole, and nothing here inspects it.** Which
        fields the mailbox capability reads, what each falls back to, and how it
        tolerates a card predating a field are all that capability's business —
        this method's is to know that the mailbox needs its card. An earlier
        shape unpacked four values here instead, and every field added to the
        card would have meant editing this method again.

        The mailbox capability is held on ``self`` as well as returned:
        ``after_tool_execute`` and the cancel check must share one instance for
        the agent's life, and ``act()``'s interruption handling looks it up by
        name.

        Args:
            mailbox_card: The agent's card — either the one the config supplied
                or the auto-inserted default.

        Returns:
            ``[mailbox, *extra_capabilities()]``. Mailbox first because hook
            order is registration order, so a run about to be cancelled does not
            first pay for a third party's ``before_model_request``.
        """
        self._mailbox_capability = MailboxCapability(observer=self, card=mailbox_card)
        return [self._mailbox_capability, *self.extra_capabilities()]

    def _build_react_agent(
        self,
        config: ReactAgentConfig,
        capabilities: list[AgentCapability[Any]],
        tools: list[Any],
        toolsets: list[Any],
    ) -> ReactAgent:
        """Build the LLM agent for this BaseAgent.

        When ``AKGENTIC_MOCK_SCENARIO`` names a scenario YAML, swap in the
        token-free ``MockReactAgent`` for load testing; otherwise build the
        real ``ReactAgent``. The deferred import keeps the optional ``loadtest``
        extra off the normal runtime path.

        Both branches receive the same ``capabilities`` list, assembled by
        ``on_start`` as ``[mailbox, *extra_capabilities()]``. It arrives as an
        argument rather than being built here so this method only *builds*: the
        mailbox capability needs the tool-card list to read its whitelist from,
        and that list is on_start's, not this method's. The framework's own
        capability is built unconditionally, on the agent and never on a card,
        so cancellation works even when the config carries no ``MailboxTool``
        (ADR-040 §5), and it is first because hook order is registration order —
        a run about to be cancelled should not first pay for a third party's
        ``before_model_request``. The mock accepts and ignores
        ``capabilities=``; drop-in parity keeps this wiring identical.

        Each branch is handed a **copy** of that list. pydantic-ai happens to
        copy before injecting its own auto-capabilities today, but that is
        undocumented upstream behaviour; copying here makes the caller's list
        an agent-side guarantee rather than an assumption to re-verify on every
        bump.

        Args:
            config: The ReactAgent configuration to build against.
            capabilities: ``[mailbox, *extra_capabilities()]``, in hook order.
            tools: Plain tool callables from the ToolFactory.
            toolsets: Toolsets from the ToolFactory.

        Returns:
            The built ``ReactAgent``, or a ``MockReactAgent`` under
            ``AKGENTIC_MOCK_SCENARIO``.
        """
        # Env var name mirrors akgentic.llm.loadtest.SCENARIO_ENV_VAR.
        scenario = os.environ.get("AKGENTIC_MOCK_SCENARIO")
        if scenario:
            from akgentic.llm.loadtest import MockReactAgent  # noqa: PLC0415

            # Carry the scenario path in a config copy's model field (the mock
            # reads model_cfg.model first); self.config is left untouched.
            # The roster goes with it: model_copy skips validation, so keeping a
            # roster the rewritten active model is no longer part of would leave the
            # copy internally inconsistent, raising only on some later
            # re-validation. A mock serves exactly one scenario file anyway, so a
            # roster it could switch away from is meaningless.
            mock_cfg = config.model_copy(
                update={
                    "model_cfg": config.model_cfg.model_copy(update={"model": scenario}),
                    "model_roster": [],
                }
            )
            return cast(
                ReactAgent,
                MockReactAgent(
                    config=mock_cfg,
                    deps_type=BaseAgent,
                    tools=tools,
                    toolsets=toolsets,
                    observer=self,
                    capabilities=list(capabilities),
                ),
            )
        return ReactAgent(
            config=config,
            deps_type=BaseAgent,
            tools=tools,
            toolsets=toolsets,
            observer=self,
            capabilities=list(capabilities),
        )

    def init_llm_context(self, context: list[EventMessage]) -> None:
        """Restore LLM conversation context from persisted events.

        Pure pass-through: forwards events to ReactAgent which owns
        the filtering and extraction logic (LlmMessageEvent -> ModelMessage).
        Part of the 4-layer restoration chain defined in ADR-009 (Layer 3).

        Args:
            context: List of EventMessage objects from the restorer.
        """
        self._react_agent.restore_context(context)

    # ============================================================================
    # MODEL SWITCHING (ModelSwitchToolObserver)
    # ============================================================================

    def list_model_rows(self) -> list[ModelRow]:
        """Project the declared roster onto ``ModelRow``, one row per entry.

        The mapping between ``akgentic-llm``'s ``ModelConfig`` and
        ``akgentic-tool``'s ``ModelRow`` lives here and nowhere else: this is the
        one package allowed to see both types, which is why the observer's
        implementation belongs on this side of the boundary at all.

        Roster and active model are both read **live**, at call time — a switch
        moves them, and nothing here is cached. ``active`` is decided by **key
        equality**, never by identity: a hand-set roster may hold an entry that is
        equal to the active model without being the same object, and that entry
        must still light up.

        An active model that no roster entry matches is a legal configuration —
        the membership rule is deliberately absent from ``AgentConfig`` — so it is
        tolerated rather than repaired: every row comes back ``active=False``,
        nothing is synthesized and nothing raises. ``ModelTool`` then composes no
        ``LLM_CONTEXT`` block that turn, which is the designed degradation.

        Returns:
            One row per roster entry, in declaration order; ``[]`` for an agent
            that declares no roster, for which switching is unavailable. No row is
            ever synthesized for an active model the roster does not carry.
        """
        active_key = model_roster_key(self._react_agent.active_model())
        return [
            ModelRow(
                key=model_roster_key(entry),
                provider=entry.provider,
                model=entry.model,
                active=model_roster_key(entry) == active_key,
                context_length=entry.context_length,
            )
            for entry in self._react_agent.model_roster()
        ]

    def switch_model(self, key: str) -> str:
        """Make the roster entry named by *key* the model in force, from the next turn.

        A refusal **raises**; it is never returned as a message.
        ``ModelTool._switch_model_factory`` records ``ToolState.active_model``
        immediately after this call returns normally, so an error string would be
        read as a success and persist a key the llm layer has just refused.

        Only ``ModelSwitchError`` is caught. ``akgentic-llm`` already translates a
        provider constructor's own failure — pydantic-ai raises ``UserError``, a
        ``RuntimeError``, for a missing API key — into that one class precisely so
        this caller needs one ``except`` and never ``except Exception``. Anything
        else is a defect and propagates untouched.

        The latch is set here as well as in :meth:`_restore_active_model`: a
        preference this agent just wrote into the slot itself must not be
        re-applied at the top of the next turn.

        Args:
            key: Roster key of the target entry, ``f"{provider}:{model}"``.

        Returns:
            A confirmation naming the entry now active and the turn from which it
            answers.

        Raises:
            RetriableError: When the switch was refused. Carries the refusal's own
                text — the only diagnosis a tool-driven caller gets — and chains
                the refusal as ``__cause__``. ``ToolFactory`` converts it into the
                injected retry exception, so the model sees a correctable retry.
        """
        try:
            entry = self._react_agent.switch_model(key)
        except ModelSwitchError as exc:
            raise RetriableError(str(exc)) from exc

        activated = model_roster_key(entry)
        self._restored_model_key = activated
        return (
            f"Switched to {activated}. The model is bound once per run, so this "
            f"takes effect from the next turn — the current one finishes on the "
            f"model that started it."
        )

    def _restore_active_model(self) -> None:
        """Re-apply the persisted model selection, once, before the turn runs.

        Called as the first statement of :meth:`act` — before
        ``_deliver_context_update()``, so the turn's ``LLM_CONTEXT`` block cannot
        advertise a model that is not the one answering, and before ``run_sync``,
        so the turn actually runs on the restored entry. ``act()`` is the single
        run entry point of the class, which makes this the one placement that is
        correct on every construction path regardless of when ``init_state()``
        lands relative to ``on_start``.

        The slot is read through the full ``self.state.tool_state`` chain at the
        moment of use: ``init_state()`` replaces the state object wholesale, so a
        reference captured at ``on_start`` would read a carrier nobody persists
        any more.

        ``_restored_model_key`` records **which persisted preference has been
        applied**, never what model is in force — every live answer still comes
        from ``self._react_agent``. It is what keeps a switch from being redone on
        every turn: a switch is a model rebuild plus a compaction-strategy
        rebuild, deliberately not short-circuited on the already-active key.

        **Never fatal.** A key the delegate refuses — stale after a roster edit,
        or an entry that will not build — leaves the declared active entry in
        force, costs one warning, and the turn proceeds. Raising over a remembered
        preference would strand the whole team on a restart.
        """
        key = self.state.tool_state.active_model
        if key is None or key == self._restored_model_key:
            return

        try:
            self._react_agent.switch_model(key)
        except ModelSwitchError as exc:
            available = ", ".join(
                model_roster_key(entry) for entry in self._react_agent.model_roster()
            )
            logger.warning(
                "[%s] persisted model %r was not restored (%s); continuing on the declared "
                "model. Available roster keys: %s",
                self.config.name,
                key,
                exc,
                available or "none — this agent declares no roster",
            )
            return

        self._restored_model_key = key

    # ============================================================================
    # USAGE TRACKING
    # ============================================================================

    def get_usage_summary(self, by_run: bool = False) -> AgentUsageSummary:
        """Query LLM usage events and return an aggregated cost summary.

        Queries the orchestrator for all LlmUsageEvent events emitted by this
        agent, extracts the event payloads, and delegates to aggregate_usage()
        for hierarchical cost aggregation.

        Callable via Pykka proxy:
            proxy_ask(agent_addr, BaseAgent).get_usage_summary().get()

        Args:
            by_run: When True, include per-run breakdown in the summary.

        Returns:
            AgentUsageSummary with totals, by-model, and optionally by-run detail.
        """
        events = self.orchestrator_proxy_ask.get_events(
            agent_id=str(self.agent_id),
            event_class=LlmUsageEvent,
        ).get()  # type: ignore[attr-defined]
        return aggregate_usage([e.event for e in events], by_run=by_run)

    # ============================================================================
    # CORE LLM INTERACTION
    # ============================================================================

    def current_message(self) -> Message | None:
        """The message whose handler is running, or ``None`` when idle.

        The public reader for core's ``_current_message``, which core sets for
        the whole of a handler and clears after it — so this is live during
        ``before_model_request`` and is what the mailbox capability matches a
        pending message's class against. Part of ``MailboxAccess``.
        """
        return self._current_message

    @guard_usage_limits()
    def act(self, message: LlmRenderable, output_type: type[T]) -> T:
        """Execute one LLM REACT loop against the output type the caller names.

        Takes the **message**, not a prompt. Framing is
        ``message.rendering()`` — one definition per message class, living
        in the class — so a handler composes no prompt and there is no second
        way in that could bypass the framing. There is deliberately no string
        overload: passing a bare ``str`` is a type error, not a supported path.

        Delegates entirely to ReactAgent.run_sync(), which:
        - Manages context history via ContextManager
        - Enforces usage limits
        - Wraps output_type with get_output_type() for provider-aware structured
          output (NativeOutput for OpenAI/Anthropic, raw type otherwise)
        - Runs the full REACT loop (tools, retries, system prompts)

        Recipient validity is NOT constrained in the schema — it is enforced at
        routing time in _route_output(). Reply-protocol guidance is carried by
        the message itself (see ``AgentMessage.renderer``), not the
        output-schema docstring.

        Two things happen before the model is reached, in this order and for this
        reason: ``_restore_active_model()`` re-applies a persisted model selection,
        then ``_deliver_context_update()`` composes the turn's context block. The
        reverse order would let the first block after a restart advertise the
        declared model while the restored one answers.

        Args:
            message: The message to reason about. Rendered exactly once, at the
                top of this method; media expansion then runs on the rendered
                string, so a ``!!glob`` written anywhere in a message's own
                framing expands the same way it always did.
            output_type: The type the REACT loop reasons against. Forwarded to
                ReactAgent.run_sync(), which wraps it with get_output_type().
                receiveMsg_AgentMessage passes StructuredOutput.

        Returns:
            An instance of output_type, as produced by the REACT loop.

        Raises:
            RunInterruptedError: **Absorbed here, not propagated** — a queued
                /stop or CancelMessage cancelled the run at a step boundary, so
                this method logs it, notifies the human once, and returns a
                default ``output_type()`` instead. Callers therefore need no
                try/except: a StructuredOutput with an empty request list routes
                nothing and the handler completes normally. The context arrives
                already healed (akgentic-llm repairs dangling tool calls before
                re-raising), so this method performs no context surgery.
                The one case a caller can still see it: an ``output_type`` that
                cannot be default-constructed — a model with at least one
                required field. The original interruption is then re-raised
                unchanged, never a ValidationError and never a wrapped exception.
            WarningError: **Raised here, in place of any usage-limit error** —
                @guard_usage_limits wraps this method, so a RunUsageLimitError,
                an AgentUsageLimitError or the base UsageLimitError never leaves
                it. The human is notified first, with the breach that arrived.
                Callers therefore need no try/except for a budget any more than
                for a cancel: both end the turn here.

                A run-tier breach only gets this far when akgentic-llm's own
                recovery declined or failed. A conclusion that *succeeded* is
                returned below as an ordinary output, indistinguishable from a
                turn that never breached, which is why nothing in this package
                tells the tiers apart any more.
        """
        self._restore_active_model()
        self._deliver_context_update()
        rendered_message = message.rendering()
        prompt = self._build_prompt_expanding_media_refs(rendered_message)
        try:
            output = self._react_agent.run_sync(prompt, deps=self, output_type=output_type)
        except RunInterruptedError as interruption:
            logger.info(
                "[%s] run interrupted by a queued cancel; turn abandoned, routing nothing",
                self.config.name,
            )
            self.notify_human("Run interrupted.")
            # A default instance is only available when every field of the
            # caller's type has a default. When it is not, the caller sees the
            # interruption itself — never the construction error behind it, so
            # the exception is named rather than bare-re-raised.
            try:
                default = output_type()
            except Exception:
                raise interruption from None
            return default

        return cast(T, output)

    def _deliver_context_update(self) -> None:
        """Append at most one **Context update** block for this turn.

        What the agent still owns is the whole of this method: the *when* —
        this is the single delivery site, called at the top of ``act()`` before
        ``run_sync`` — and the *how* — the append goes through
        ``ContextManager.append_user_prompt``, never a bare
        ``ModelRequest``, so the buffer-vs-append decision stays with the
        context and a fresh agent's first block is folded into the first run's
        user prompt instead of suppressing system-prompt injection.

        Everything else is the engine's: reading the providers, diffing against
        the persisted baselines, reconciling them against the visible history,
        composing the block and advancing the counter. See
        ``akgentic.tool.core.ContextUpdater``, which owns those semantics along
        with the cards that produce them. It never raises and returns ``None``
        when there is nothing to say.
        """
        block = self._context_updater.compose_update(self._react_agent.context.messages)
        if block is not None:
            self._react_agent.context.append_user_prompt(block)

    def _build_prompt_expanding_media_refs(self, rendered: str) -> UserPrompt:
        """Build the run's ``UserPrompt``, expanding any ``!!glob`` media references.

        Expansion is a ``COMMAND``-channel capability of the workspace card,
        reached through the command registry rather than imported. An agent
        configured without that card simply has no ``_expand_media_refs``
        command registered, so the prompt passes straight through — the absence
        is the off switch, and no branch here has to know which cards exist.

        The unchanged prompt is returned as the **plain string** it arrived as,
        not as a single-element list. Both satisfy ``UserPrompt``, but wrapping
        would make every prompt multipart for the benefit of the rare one that
        actually carries media.

        Args:
            rendered: The prompt text, already produced by the message's own
                ``rendering()``.

        Returns:
            ``rendered`` unchanged when no reference expanded; otherwise the
            mixed list of text and ``BinaryContent`` parts the command produced.
        """
        prompt: UserPrompt = rendered
        if self._command_registry.has("_expand_media_refs"):
            expand = self._command_registry.callable("_expand_media_refs")
            parts = expand(rendered)
            if parts != [rendered]:
                prompt = [
                    BinaryContent(data=p.data, media_type=p.media_type)
                    if isinstance(p, MediaContent)
                    else p
                    for p in parts
                ]
        return prompt

    def _route_output(self, output: StructuredOutput) -> bool:
        """Send one AgentMessage per Request — the class's single routed send path.

        A recipient starting with ``@`` resolves to an existing member via
        ``get_team_member``; anything else is hired by role. A recipient that
        resolves to ``None`` is skipped, so the model naming someone who does not
        exist costs a delivery, not an exception.

        Extracted so every turn delivers through exactly the same code, and there
        is now only one caller: ``receiveMsg_AgentMessage``. An interrupted turn
        and a turn ``akgentic-llm`` concluded after a run-tier breach both come
        back from ``act()`` as an ordinary ``StructuredOutput``, so neither needs
        a second delivery path — the empty one routes nothing, the concluded one
        routes its requests.

        Args:
            output: The StructuredOutput whose Requests are to be delivered.

        Returns:
            Whether anything was actually delivered. The usage-limit guard asks
            this to tell a real conclusion from one that routed nothing — it
            cannot inspect the output itself, since the schema is the caller's.
        """
        delivered = False

        for request in output.messages:
            member = resolve_recipient(self, request.recipient)

            if member is not None:
                self.send(
                    member,
                    AgentMessage(
                        content=request.message,
                        type=request.message_type,
                        recipient=member,
                    ),
                )
                delivered = True

        return delivered

    def receiveMsg_AgentMessage(self, message: AgentMessage, sender: ActorAddress) -> None:  # noqa: N802
        """Handle an incoming AgentMessage — the agent's only message handler.

        Content starting with ``/`` is offered to the command registry first; if a
        command handles it, the method returns without involving the LLM. Otherwise
        the message itself is run as one act() turn — it frames itself through
        ``AgentMessage.rendering()``, which is where the reply protocol for
        ``message.type`` now lives — whose StructuredOutput goes to
        _route_output().

        This body carries no ``try``/``except`` at all. A queued ``/stop`` or
        ``CancelMessage`` is absorbed inside ``act()``, which notifies the human
        and returns an empty ``StructuredOutput``; ``_route_output`` then
        delivers nothing and the handler returns normally — the run dies, the
        agent survives. A usage breach is likewise the decorator's (see
        ``usage_limits.guard_usage_limits``), and it owns usage-limit errors only.

        A run-tier breach that ``akgentic-llm`` recovered never reaches the
        decorator: the tool-free conclusion returns from ``act()`` as an ordinary
        ``StructuredOutput`` and routes below like any other turn.

        Args:
            message: The AgentMessage instance containing the message content and recipient.
            sender: The ActorAddress of the sender of the message.

        Raises:
            WarningError: Raised by the decorator when a usage-limit error escapes
                the LLM — either tier, since both are now handled identically.
                notify_human() runs first — a no-op with a log line when the team
                has no user-proxy member. Usage-limit errors are the only ones the
                decorator catches, so anything else propagates out untouched.
        """

        logger.info(
            f"[{self.config.name}-{self.team_id}] Received '{message.type}' AgentMessage "
            f"from {sender} ({len(message.content)} chars)"
        )

        sleep(random.uniform(0.25, 0.5))  # Simulate processing delay

        # Slash-command interception runs on the RAW content (before the
        # message's own framing) so dispatch sees the leading "/<command>".
        if message.content.startswith("/") and self._dispatch_command(message, sender):
            return

        output = self.act(message, StructuredOutput)

        self._route_output(output)

    def receiveMsg_CancelMessage(  # noqa: N802
        self, message: CancelMessage, sender: ActorAddress
    ) -> None:
        """Acknowledge a CancelMessage delivered as its own turn — a logged no-op.

        This is the **idle** cancel handler, and by construction the only one it
        could be. Cancellation is enforced by the mailbox peek inside the run
        (see ``MailboxCapability``), which purges what it recognises at the
        moment it recognises it — so a cancel that arrived mid-run is gone from
        the mailbox before the actor could ever dequeue it, and can never reach
        here. Arriving here therefore *is* the proof that nothing was running:
        there is nothing to cancel, no state to set, no run-state check to make,
        and the next run is unaffected.

        Args:
            message: The CancelMessage being acknowledged.
            sender: The ActorAddress of the sender of the message.
        """
        logger.info(
            "[%s] CancelMessage received while idle (reason: %r) — nothing to cancel",
            self.config.name,
            message.reason,
        )

    def _dispatch_command(self, message: AgentMessage, sender: ActorAddress) -> bool:
        """Dispatch a ``/``-prefixed message through the command registry.

        A dispatch has three outcomes:

        - **A string result** — sent back to ``sender`` as a ``notification`` (a
          non-``request`` type, so it does not trigger a reply loop), recorded as
          one operator action, and reported as ``True``.
        - **``None``** — the command handled itself and has decided it has
          nothing to report. Returns ``True`` at once: no notification, no
          operator-action entry. A command whose whole effect is elsewhere —
          it writes a file, or sends a message of its own — would otherwise be
          double-reported, once by its own effect and once by an empty reply.
          This is a general primitive, not a carve-out for any one command.
        - **:class:`CommandNotRecognized`** — the leading token is not a
          registered command. Swallowed; returns ``False`` so the caller falls
          back to the normal LLM path with the original content, and nothing is
          recorded because the command never ran.

        Post-identification failures (bad/missing args) are caught inside
        ``dispatch`` and returned as a result string — handled here exactly like
        a success, never falling back to the LLM.

        The entry is synthetic and human-attributed: it is composed here and
        appended to the ReactAgent context via
        :meth:`_record_user_action`, so the agent reasons about the human's
        action (and its result) on its next turn without mistaking it for its
        own tool call.

        Args:
            message: The incoming AgentMessage whose raw content starts with ``/``.
            sender: The ActorAddress to send the command result back to.

        Returns:
            ``True`` if the content was dispatched as a command — whether the
            result was sent back (string) or deliberately silent (``None``);
            ``False`` if the leading token was not a known command.
        """
        # ``CommandRegistry.dispatch`` is declared ``-> str | None``: a command
        # may have nothing to say, and the None branch below is the general
        # answer to that, not a special case for any one command.
        result: str | None
        try:
            result = self._command_registry.dispatch(message.content)
        except CommandNotRecognized:
            return False

        if result is None:
            return True

        self.send(
            sender,
            AgentMessage(content=result, type="notification", recipient=sender),
        )

        self._record_user_action(
            f'**User action** - The human ran "{message.content}". \nResult:\n{result}'
        )
        return True

    def _record_user_action(self, entry: str) -> None:
        """Hand one out-of-band, user-role entry to the LLM ContextManager.

        One of two points where this class writes non-agent content into its own
        history — the sibling is the context-update delivery at the top of
        :meth:`act`, which calls the context primitive directly. Its one caller
        today is :meth:`_dispatch_command`, for a human's slash command; the
        wording of the entry belongs to the caller, and the buffer-vs-append
        decision belongs to the context (ADR-007 §3), not reimplemented here.

        Args:
            entry: The pre-composed entry text.
        """
        self._react_agent.context.append_user_prompt(entry)

    def notify_human(self, message: str) -> None:
        """Notify the team's user-proxy member; log and return if there is none."""
        human = next((member for member in self.get_team() if member.is_user_proxy), None)
        if human is None:
            logger.warning(
                "No user-proxy team member found; notice not delivered: %s",
                message,
            )
            return
        self.send(human, AgentMessage(content=message, recipient=human, type="notification"))

    # ============================================================================
    # TEAM AWARENESS
    # ============================================================================

    def on_hire(self, address: ActorAddress) -> None:
        pass

    def on_fire(self, address: ActorAddress) -> None:
        pass

    def hire_member(self, role: str) -> ActorAddress:
        """Hire a single team member by role via the registry's hire_member command.

        Resolves the typed ``hire_member`` callable from the command registry and
        invokes it with the native ``role`` (native ``ActorAddress`` return — no
        ``/hire …`` string round-trip). Reached from ``_route_output`` via
        ``resolve_recipient``.

        A failed hire raises ``ModelRetry``: the registry retry-wraps every command,
        converting the tool layer's ``RetriableError``. **On this path nothing
        honours that retry.** ``_route_output`` runs after ``act()`` has already
        returned, so the REACT loop is over and the routing is outside the
        usage-limit guard as well — that guard is on ``act()`` and catches
        usage-limit errors only, which this is not. The exception therefore leaves
        the actor message handler, deliberately unswallowed. Retry *is* honoured on
        the other path: when the model calls the ``hire_members`` tool
        mid-reasoning, pydantic-ai is still inside the loop and retries there.

        Args:
            role: Role to hire (must exist in agent catalog)

        Returns:
            ActorAddress: Address of the newly hired member.

        Raises:
            RuntimeError: If the hire_member command is not registered
                (TeamTool not configured).
            ModelRetry: If role is invalid or the hire fails.
        """
        if not self._command_registry.has("hire_member"):
            raise RuntimeError("hire_member command not available — TeamTool not configured")

        hire = self._command_registry.callable("hire_member")
        return cast(ActorAddress, hire(role))

    @guard_usage_limits()
    def compact(self) -> str:
        """Compact this agent's conversation history into a summary, preserving system prompts."""
        return self._react_agent.compact()

    def clear(self) -> str:
        """Clear this agent's conversation; the system prompt regenerates on the next run.

        Resetting the updater zeroes the persisted slot — baselines and block
        counter both — because the emptied history has no markers left to
        continue, so the next block is ``**Context update 1**``, a full
        snapshot of current state. This is the one legitimate zeroing of the
        counter. ``compact()`` deliberately gets no reset: the updater's own
        reconciliation against the visible history catches a folded-away
        block, including the automatic compaction a ``compact()`` hook would
        miss entirely.
        """
        result = self._react_agent.clear_context()
        self._context_updater.reset()
        return result
