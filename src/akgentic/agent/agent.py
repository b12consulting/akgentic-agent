"""BaseAgent: LLM-powered team agent with delegation and collaboration.
BaseAgent integrates with akgentic-llm's ReactAgent for all LLM management,
with team-specific message handling and a static structured-output schema.

Architecture:
- Extends Akgent[AgentConfig, AgentState] from akgentic-core (pykka actor model)
- Composes ReactAgent (akgentic-llm) for model, http client, context, usage limits
- Composes ToolFactory (akgentic-tool) aggregating ToolCard[] into 3 channels:
  · TOOL_CALL — LLM-callable tools (hire_members, fire_members, + config.tools)
  · SYSTEM_PROMPT — dynamic prompts (TeamTool yields team_roster, role_profiles)
  · COMMAND — programmatic commands exposed via a generic CommandRegistry
    (built once in on_start; dispatched by name from /-prefixed messages)
- TeamTool auto-injected if not already in config.tools
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
import re
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from time import sleep
from typing import Any, TypeVar, cast

from pydantic_ai import BinaryContent, ModelRetry, RunContext
from pydantic_ai.messages import ModelRequest, UserPromptPart

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
from akgentic.tool.core import CommandRegistry, ContextState, ToolFactory
from akgentic.tool.errors import CommandNotRecognized
from akgentic.tool.core.event import CommandsAnnouncedEvent
from akgentic.tool.team import TeamTool
from akgentic.tool.workspace.readers import MediaContent

logger = logging.getLogger(__name__)


T = TypeVar("T")

# The Context update marker line (ADR-037 §6). The numbered pattern recovers the
# block counter from a restored history; the presence check uses the exact
# per-sequence substring instead.
_CONTEXT_UPDATE_MARKER = re.compile(r"\*\*Context update (\d+)\*\*")


class BaseAgent(Akgent[AgentConfig, AgentState]):
    """LLM-powered team agent with delegation and collaboration capabilities.

    Composition:
    - ReactAgent (akgentic-llm): model instantiation, HTTP retry, usage limits,
      context history, REACT loop. Reasoning turns go through run_sync();
      compact() and clear() bypass it — compact() is ReactAgent's own synchronous
      bridge onto the agent loop, clear() a plain wrapper over the context.
    - ToolFactory (akgentic-tool): aggregates ToolCard[] into tools, system prompts,
      and commands via 3-channel architecture (TOOL_CALL, SYSTEM_PROMPT, COMMAND).
    - TeamTool: auto-injected if absent from config.tools; provides hire/fire
      capabilities and team awareness prompts.

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
    - whatever ToolFactory.get_system_prompts() yields — from TeamTool, a team
      roster and/or a role-profiles prompt, each only if SYSTEM_PROMPT is in that
      capability's expose set
    - mailbox_notifications, only when the mailbox is non-empty at on_start

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
        @self._react_agent.system_prompt: agent_backstory, current_date, whatever
        ToolFactory.get_system_prompts() yields (from TeamTool: a team roster
        and/or a role-profiles prompt), and mailbox_notifications when the mailbox
        is non-empty at start. ReactAgent contributes none of its own.

        Tools (hire_members, fire_members) come from TeamTool.get_tools() as
        closures over the orchestrator proxy — not bound methods of this agent —
        and take no RunContext, so pydantic-ai treats them as plain tools.
        Commands are aggregated into a single generic CommandRegistry, held for the
        agent's lifetime, and announced once via a CommandsAnnouncedEvent.
        """
        assert self._orchestrator is not None, "Orchestrator address must be provided in config"
        self.orchestrator_proxy_ask = self.proxy_ask(self._orchestrator, Orchestrator)

        self._current_message: AgentMessage | None = None

        # ── State ───────────────────────────────────────────────────────────────
        self.state = AgentState(backstory=self.config.prompt.render()).observer(self)

        # ── Add TeamTool automatically (without mutating config) ──────────────
        # TeamTool is hardcoded in akgentic-agent package
        has_team_tool = any(isinstance(t, TeamTool) for t in self.config.tools)
        tool_cards = self.config.tools if has_team_tool else [TeamTool(), *self.config.tools]

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

        # ── Context-state providers (LLM_CONTEXT channel) ──────────────────────
        # Collected once for the agent's lifetime, like the command registry.
        # Baselines are an in-memory cache keyed by provider __name__ — never
        # persisted: the message history is the record, and a lost baseline can
        # only cause a full-snapshot re-send, never a lost update.
        self._context_state_providers: list[Callable[[], ContextState | None]] = (
            tool_factory.get_context_states()
        )
        self._context_baselines: dict[str, ContextState] = {}
        self._context_update_seq: int = 0

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

        if inbox := self.get_mailbox():
            @self._react_agent.system_prompt
            def mailbox_notifications(ctx: RunContext[BaseAgent]) -> str | None:
                    senders = {msg.sender.name for msg in inbox if msg.sender}
                    return (
                        f"NOTICE: {len(inbox)} new message(s) arrived in your mailbox "
                        f"from team member(s): {', '.join(senders)}."
                        "\nConsider wrapping up the current thread to process them."
                    )

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

        The single delivery site (called at the top of ``act()``, before
        ``run_sync``): reads every context-state provider in factory order,
        diffs against the in-memory baselines, composes at most one block, and
        appends it through ``ContextManager.record_operator_action`` — the
        buffer-vs-append decision stays with the context, so on a fresh agent
        the first block is folded into the first run's user prompt instead of
        suppressing system-prompt injection.

        The baselines of contributing providers and the block counter advance
        only when a block is actually appended; a turn with nothing to say
        appends nothing, so an unchanged turn stays byte-identical for the
        prompt cache. Context construction never raises (see
        ``_render_context_section`` for the per-provider degradation).

        Before any provider is read, ``_verify_context_baselines`` checks that
        the last delivered block is still visible to the model and drops the
        baselines when it is not, so every eviction path (compaction, window
        trimming, restore, out-of-band wipe) self-heals with a full snapshot.
        ``full_snapshot`` is captured right after that check — a no-change
        delta advances a baseline mid-loop, so the wording flag must be taken
        before the providers are read.
        """
        self._verify_context_baselines()
        full_snapshot = not self._context_baselines
        sections: list[str] = []
        advanced: dict[str, ContextState] = {}
        for provider in self._context_state_providers:
            contribution = self._render_context_section(provider)
            if contribution is None:
                continue
            rendering, state = contribution
            sections.append(rendering)
            advanced[provider.__name__] = state
        if not sections:
            return
        block = self._compose_context_update(sections, full_snapshot)
        self._react_agent.context.record_operator_action(block)
        self._context_baselines.update(advanced)
        self._context_update_seq += 1

    def _verify_context_baselines(self) -> None:
        """Trust the baselines only while the last delivered marker is visible.

        A delta is only correct relative to a baseline the model can still see
        (ADR-037 §7). When a block has been delivered (``seq > 0``), its exact
        marker substring must appear in the user-role prompt parts of the
        context history; a miss — compaction (manual or automatic), sliding-
        window trimming, any out-of-band wipe — drops **every** baseline so the
        next block is a full snapshot. The counter is never reset on a miss:
        ``N`` stays monotonic, because a partially trimmed history may still
        show older numbers.

        On a fresh life (``seq == 0`` — baselines are never persisted), the
        counter is instead recovered as the highest marker number found in the
        restored history, so the next block continues the sequence as ``N+1``
        rather than re-emitting a number the model may already see. Baselines
        stay empty either way — a restored agent re-delivers a full snapshot,
        never silently adopts current state as a baseline (a lost baseline may
        only cause a re-send, never a lost update).
        """
        if self._context_update_seq > 0:
            marker = f"**Context update {self._context_update_seq}**"
            if not any(marker in text for text in self._iter_user_prompt_texts()):
                self._context_baselines.clear()
            return
        highest = 0
        for text in self._iter_user_prompt_texts():
            for match in _CONTEXT_UPDATE_MARKER.finditer(text):
                highest = max(highest, int(match.group(1)))
        self._context_update_seq = highest

    def _iter_user_prompt_texts(self) -> Iterator[str]:
        """Yield the user-prompt text of the context history, one lazy pass.

        Scan scope is deliberately narrow: only ``UserPromptPart`` content on
        ``ModelRequest`` messages — a ``str`` content directly, the ``str``
        items of a multimodal ``list`` content otherwise. Tool returns,
        retry prompts, system prompts and ``ModelResponse`` messages are never
        inspected (a model echoing the marker must not count), and nothing is
        concatenated. Never construct a ``ModelRequest`` here or anywhere
        outside ``record_operator_action`` — appending one is the retired
        ADR-007 defect; these imports exist for ``isinstance`` checks only.
        """
        for message in self._react_agent.context.messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if not isinstance(part, UserPromptPart):
                    continue
                if isinstance(part.content, str):
                    yield part.content
                elif isinstance(part.content, list):
                    yield from (item for item in part.content if isinstance(item, str))

    def _render_context_section(
        self, provider: Callable[[], ContextState | None]
    ) -> tuple[str, ContextState] | None:
        """Compute one provider's section for this turn, or ``None`` for no section.

        First-seen states — and states whose concrete type differs from their
        baseline's (a card reconfigured mid-life) — render ``render_full()``;
        otherwise ``render_delta(baseline)``. The rendering is used verbatim.

        Degradation, never failure: a provider or renderer that raises is
        logged and skipped without advancing its baseline; a ``None`` state or
        an empty full rendering contributes nothing. A ``None`` delta means no
        change, so the baseline may advance to the current (equal) state even
        though no section is produced.

        Returns:
            The ``(rendering, state)`` pair to contribute, or ``None``.
        """
        name = provider.__name__
        try:
            state = provider()
        except Exception:
            logger.exception(
                "[%s] context-state provider '%s' raised; skipped", self.config.name, name
            )
            return None
        if state is None:
            return None
        baseline = self._context_baselines.get(name)
        try:
            if baseline is None or type(state) is not type(baseline):
                rendering: str | None = state.render_full()
            else:
                rendering = state.render_delta(baseline)
        except Exception:
            logger.exception(
                "[%s] context-state renderer '%s' raised; skipped", self.config.name, name
            )
            return None
        if rendering is None:
            self._context_baselines[name] = state
            return None
        if not rendering:
            return None
        return rendering, state

    def _compose_context_update(self, sections: list[str], full_snapshot: bool) -> str:
        """Compose one Context update block: marker line + verbatim sections.

        The marker is ``**Context update N**`` with a fixed suffix — worded as
        current state when the baselines were empty as delivery began (first
        block of a life, post-``/clear``, post-eviction, post-restore: every
        section renders full then), as change only when the block was diffed
        against surviving baselines. Both suffixes are fixed strings — nothing
        turn-varying beyond ``N`` may appear: a timestamp or "as of" line would
        defeat the cache property this epic exists for. Sections are joined to
        the marker and to each other with blank lines, never re-wrapped — the
        renderers own their internal join style.
        """
        number = self._context_update_seq + 1
        suffix = (
            "current state."
            if full_snapshot
            else "state has changed since the last update."
        )
        marker = f"**Context update {number}** — {suffix}"
        return "\n\n".join([marker, *sections])

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

        The single point where this class writes something the agent did not say
        itself into its own history. Its one caller today is
        :meth:`_dispatch_command`, for a human's slash command. The wording of the
        entry belongs to the caller, so a second kind of out-of-band event can
        never be framed as the first; the buffer-vs-append decision belongs to the
        context (ADR-007 §3) and is not reimplemented here.

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

        Also drops the context-update baselines and resets the block counter:
        the emptied history has no markers left to continue, so the next block
        is ``**Context update 1**`` — a full snapshot of current state.
        ``compact()`` deliberately gets no such reset — the presence check in
        ``_verify_context_baselines`` catches a folded-away block, including
        the automatic compaction a ``compact()`` hook would miss entirely.
        """
        result = self._react_agent.clear_context()
        self._context_baselines.clear()
        self._context_update_seq = 0
        return result
