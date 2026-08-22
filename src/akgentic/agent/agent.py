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
- Implements TeamManagementToolObserver protocol (structural typing)
- One message handler of its own, receiveMsg_AgentMessage: all team traffic is
  AgentMessage, so there is no per-message-type handler set here. Akgent still
  contributes the lifecycle handler receiveMsg_StopRecursively
- The usage-limit tier policy is applied, not written here: the handler carries
  @guard_usage_limits(output_type=..., route=...) from usage_limits.py. See
  custom_agent.py for a second agent class doing the same with its own schema
- Delegation is a plain send per Request in the LLM's StructuredOutput. Each hop
  is an independent turn — no call stack, no automatic return path
"""

import logging  # noqa: I001
import os
import random
from datetime import datetime, timezone
from time import sleep
from typing import Any, TypeVar, cast

from pydantic_ai import BinaryContent, ModelRetry, RunContext

from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import REPLY_PROTOCOLS, StructuredOutput
from akgentic.agent.usage_limits import guard_usage_limits
from akgentic.agent.utils import resolve_recipient
from akgentic.core import ActorAddress, Akgent, Orchestrator
from akgentic.core.messages import EventMessage
from akgentic.llm import (
    AgentUsageSummary,
    LlmUsageEvent,
    ReactAgent,
    ReactAgentConfig,
    UserPrompt,
    aggregate_usage,
)
from akgentic.tool.core import CommandRegistry, ContextUpdater, ToolFactory
from akgentic.tool.errors import CommandNotRecognized
from akgentic.tool.core.event import CommandsAnnouncedEvent
from akgentic.tool.mailbox import MailboxTool
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
    - MailboxTool: auto-injected if absent from config.tools; provides mailbox
      status as LLM_CONTEXT state, the read_mailbox peek tool, and the /stop
      command. A card supplied in config.tools wins over the default.

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
    - receiveMsg_AgentMessage is the only handler this class defines (Akgent
      contributes receiveMsg_StopRecursively). /-prefixed content is offered to
      the CommandRegistry first; everything else — including a /-prefixed token
      the registry does not recognise — is prefixed with the reply protocol for
      its message type and run as one act() turn.
    - That turn's StructuredOutput goes to _route_output(), which sends one
      AgentMessage per Request. A recipient starting with "@" resolves to an
      existing member; anything else is hired by role. A recipient that resolves
      to None is skipped.
    - A usage breach is branched on by tier (never on message text), by the
      @guard_usage_limits decorator rather than by this class: a run-tier breach
      gets one tool-free conclusion, delivered through that same _route_output();
      an agent-tier breach, or a conclusion that delivers nothing, notifies the
      human and raises WarningError.

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
      card set: the roster, role profiles, planning, knowledge-graph and
      mailbox capabilities declare LLM_CONTEXT and arrive as context-update
      blocks

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
      appending through record_operator_action.

    Internal method (used by _route_output()):
    - hire_member(role) → ActorAddress. A failed hire raises ModelRetry; see
      hire_member() for where that retry is, and is not, honoured.
    """

    def on_start(self) -> None:
        """Initialize BaseAgent using ReactAgent from akgentic-llm.

        ReactAgent internally handles:
        - create_model() / create_model_settings() / create_http_client()
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

        self._current_message: AgentMessage | None = None

        # ── State ───────────────────────────────────────────────────────────────
        self.state = AgentState(backstory=self.config.prompt.render()).observer(self)

        # ── Add TeamTool and MailboxTool automatically (without mutating config) ──
        # Both intrinsic cards are hardcoded in akgentic-agent package; a card
        # already present in config.tools wins over the prepended default.
        tool_cards = list(self.config.tools)
        if not any(isinstance(t, MailboxTool) for t in tool_cards):
            tool_cards.insert(0, MailboxTool())
        if not any(isinstance(t, TeamTool) for t in tool_cards):
            tool_cards.insert(0, TeamTool())

        # ── ReactAgent: wraps model, http client, context, usage limits ──────
        # Tools come from ToolFactory (includes TeamTool hire/fire via factory pattern)
        # result_type=str is the default; structured calls use pydantic_agent.iter()
        # with a per-call output_type override.
        react_agent_config = ReactAgentConfig(
            model_cfg=self.config.model_cfg,
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

        self._react_agent = self._build_react_agent(react_agent_config, tools, toolsets)

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

    def _build_react_agent(
        self, config: ReactAgentConfig, tools: list[Any], toolsets: list[Any]
    ) -> ReactAgent:
        """Build the LLM agent for this BaseAgent.

        When ``AKGENTIC_MOCK_SCENARIO`` names a scenario YAML, swap in the
        token-free ``MockReactAgent`` for load testing; otherwise build the
        real ``ReactAgent``. The deferred import keeps the optional ``loadtest``
        extra off the normal runtime path.
        """
        # Env var name mirrors akgentic.llm.loadtest.SCENARIO_ENV_VAR.
        scenario = os.environ.get("AKGENTIC_MOCK_SCENARIO")
        if scenario:
            from akgentic.llm.loadtest import MockReactAgent  # noqa: PLC0415

            # Carry the scenario path in a config copy's model field (the mock
            # reads model_cfg.model first); self.config is left untouched.
            mock_cfg = config.model_copy(
                update={"model_cfg": config.model_cfg.model_copy(update={"model": scenario})}
            )
            return cast(
                ReactAgent,
                MockReactAgent(
                    config=mock_cfg,
                    deps_type=BaseAgent,
                    tools=tools,
                    toolsets=toolsets,
                    observer=self,
                ),
            )
        return ReactAgent(
            config=config,
            deps_type=BaseAgent,
            tools=tools,
            toolsets=toolsets,
            observer=self,
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

    def act(self, user_content: str, output_type: type[T]) -> T:
        """Execute one LLM REACT loop against the output type the caller names.

        Delegates entirely to ReactAgent.run_sync(), which:
        - Manages context history via ContextManager
        - Enforces usage limits
        - Wraps output_type with get_output_type() for provider-aware structured
          output (NativeOutput for OpenAI/Anthropic, raw type otherwise)
        - Runs the full REACT loop (tools, retries, system prompts)

        Recipient validity is NOT constrained in the schema — it is enforced at
        routing time in _route_output(). Reply-protocol guidance is carried in
        the prompt (see receiveMsg_AgentMessage), not the output-schema docstring.

        Args:
            user_content: User message to process.
            output_type: The type the REACT loop reasons against. Forwarded to
                ReactAgent.run_sync(), which wraps it with get_output_type().
                receiveMsg_AgentMessage passes StructuredOutput.

        Returns:
            An instance of output_type, as produced by the REACT loop.

        Raises:
            RunUsageLimitError: The turn exhausted its own budget. Recoverable —
                the @guard_usage_limits decorator on the calling handler answers
                it with a tool-free conclusion.
            AgentUsageLimitError: The agent's lifetime budget is spent. Terminal.
            LLMUsageLimitError: Base of both, if akgentic-llm raises it directly.
                All three are propagated unchanged from ReactAgent.run_sync():
                this method neither notifies anyone nor wraps them, and it does
                not tell the tiers apart — the decorator does all of it.
        """
        self._deliver_context_update()
        # ── Media expansion (!!glob_pattern → BinaryContent) ────────────────────
        prompt: UserPrompt = user_content
        if self._command_registry.has("_expand_media_refs"):
            expand = self._command_registry.callable("_expand_media_refs")
            parts = expand(user_content)
            if parts != [user_content]:
                prompt = [
                    BinaryContent(data=p.data, media_type=p.media_type)
                    if isinstance(p, MediaContent)
                    else p
                    for p in parts
                ]
        # ── End media expansion ─────────────────────────────────────────────────
        output = self._react_agent.run_sync(prompt, deps=self, output_type=output_type)

        return cast(T, output)

    def _deliver_context_update(self) -> None:
        """Append at most one **Context update** block for this turn.

        What the agent still owns is the whole of this method: the *when* —
        this is the single delivery site, called at the top of ``act()`` before
        ``run_sync`` — and the *how* — the append goes through
        ``ContextManager.record_operator_action``, never a bare
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
            self._react_agent.context.record_operator_action(block)

    def _route_output(self, output: StructuredOutput) -> bool:
        """Send one AgentMessage per Request — the class's single routed send path.

        A recipient starting with ``@`` resolves to an existing member via
        ``get_team_member``; anything else is hired by role. A recipient that
        resolves to ``None`` is skipped, so the model naming someone who does not
        exist costs a delivery, not an exception.

        Extracted so the normal turn and the tool-free conclusion of an interrupted
        turn deliver through exactly the same code: ``receiveMsg_AgentMessage``
        calls it directly, and hands it to ``@guard_usage_limits`` as the ``route``
        argument so a breached turn is delivered the same way. The name matches
        ADR-008 §1 so the dev-overridable usage-limit capability merges with this
        extraction rather than renaming it.

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

    @guard_usage_limits(output_type=StructuredOutput, route=_route_output)
    def receiveMsg_AgentMessage(self, message: AgentMessage, sender: ActorAddress) -> None:  # noqa: N802
        """Handle an incoming AgentMessage — the agent's only message handler.

        Content starting with ``/`` is offered to the command registry first; if a
        command handles it, the method returns without involving the LLM. Otherwise
        the raw content is prefixed with the reply protocol for ``message.type``
        (see REPLY_PROTOCOLS) and run as one act() turn, whose StructuredOutput
        goes to _route_output().

        This body carries no ``try``/``except`` of its own. The usage-limit tier
        policy is the decorator's (see ``usage_limits.guard_usage_limits``), which
        is handed this handler's schema and routing so an interrupted turn is
        concluded and delivered exactly as a normal one would be. Written once
        there rather than copied here, because the ``except`` ordering it depends on
        is load-bearing and a wrong copy fails silently.

        Args:
            message: The AgentMessage instance containing the message content and recipient.
            sender: The ActorAddress of the sender of the message.

        Raises:
            WarningError: Raised by the decorator when the turn exceeds a usage
                limit and no conclusion was delivered. notify_human() runs first —
                a no-op with a log line when the team has no user-proxy member. A
                run-tier breach that concluded successfully raises nothing.
                Usage-limit errors are the only ones the decorator catches, so
                anything else propagates out of the handler untouched.
        """

        logger.info(
            f"[{self.config.name}-{self.team_id}] Received '{message.type}' AgentMessage "
            f"from {sender} ({len(message.content)} chars)"
        )

        # "unknown" is prose for the prompt. The guard keeps its own requester as
        # None, because a placeholder there would become a routing target in the
        # tool-free conclusion — it would answer a member called "unknown".
        sender_name = message.sender.name if message.sender else "unknown"

        sleep(random.uniform(0.25, 0.5))  # Simulate processing delay

        # Slash-command interception runs on the RAW content (before the
        # typed-protocol prefix) so dispatch sees the leading "/<command>".
        if message.content.startswith("/") and self._dispatch_command(message, sender):
            return

        article = "an" if message.type[0] in "aeiou" else "a"
        prefixed_content = (
            f"You received {article} {message.type} from {sender_name}. "
            f"{REPLY_PROTOCOLS.get(message.type, '').format(sender=sender_name)}"
            f"\n\n{message.content}"
        )
        output = self.act(prefixed_content, StructuredOutput)

        self._route_output(output)

    def _dispatch_command(self, message: AgentMessage, sender: ActorAddress) -> bool:
        """Dispatch a ``/``-prefixed message through the command registry.

        Sends the dispatch result back to ``sender`` as a ``notification`` (a
        non-``request`` type, so it does not trigger a reply loop) and returns
        ``True`` to signal the message was handled as a command.

        When the leading token is not a registered command, ``dispatch`` raises
        :class:`CommandNotRecognized`; this method swallows that and returns
        ``False`` so the caller falls back to the normal LLM path with the
        original content. Post-identification failures (bad/missing args) are
        caught inside ``dispatch`` and returned as a result string — they are
        handled here exactly like a success and never fall back to the LLM.

        On any dispatched (non-``CommandNotRecognized``) outcome, exactly one
        synthetic, human-attributed operator-action entry is composed here and
        appended to the ReactAgent context via :meth:`_record_operator_action`, so
        the agent reasons about the human's action (and its result) on its next
        turn without mistaking it for its own tool call. The fallback branch
        records nothing — the command never ran.

        Args:
            message: The incoming AgentMessage whose raw content starts with ``/``.
            sender: The ActorAddress to send the command result back to.

        Returns:
            ``True`` if the content was dispatched as a command (result sent),
            ``False`` if the leading token was not a known command.
        """
        try:
            result = self._command_registry.dispatch(message.content)
        except CommandNotRecognized:
            return False

        self.send(
            sender,
            AgentMessage(content=result, type="notification", recipient=sender),
        )

        self._record_operator_action(
            f'[Operator action] The human ran "{message.content}". \nResult:\n{result}'
        )
        return True

    def _record_operator_action(self, entry: str) -> None:
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
        self._react_agent.context.record_operator_action(entry)

    def notify_human(self, message: str) -> None:
        """Notify the team's user-proxy member; log and return if there is none."""
        human = next((member for member in self.get_team() if member.is_user_proxy), None)
        if human is None:
            logger.warning(
                "No user-proxy team member found; usage-limit notice not delivered: %s",
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
        returned, so the REACT loop is over, and the usage-limit guard around the
        handler catches only usage-limit errors — so the exception leaves the actor
        message handler. It is deliberately not swallowed. Retry *is* honoured on
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
