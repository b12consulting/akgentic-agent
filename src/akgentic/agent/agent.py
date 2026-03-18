
"""BaseAgent: LLM-powered team agent with delegation and collaboration.
BaseAgent integrates with akgentic-llm's ReactAgent for all LLM management,
with team-specific message handling and structured output patterns.

Architecture:
- Extends Akgent[AgentConfig, AgentState] from akgentic-core (pykka actor model)
- Composes ReactAgent (akgentic-llm) for model, http client, context, usage limits
- Composes ToolFactory (akgentic-tool) aggregating ToolCard[] into 3 channels:
  · TOOL_CALL — LLM-callable tools (hire_members, fire_members, + config.tools)
  · SYSTEM_PROMPT — dynamic prompts (team_roster, role_profiles, backstory)
  · COMMAND — programmatic commands (cmd_hire_member, cmd_fire_member, etc.)
- TeamTool auto-injected if not already in config.tools
- ReactAgent.run_sync(output_type=T) for both str and structured output
- get_output_type() applied inside ReactAgent.run() — no leakage into BaseAgent
- Implements TeamManagementToolObserver protocol (structural typing)
- Continuation-based routing for multi-hop delegation chains
- Message handlers: UserMessage → ResultMessage, HelpRequest ↔ HelpAnswer
"""

import logging
import random
from datetime import datetime, timezone
from time import sleep
from typing import TypeVar, cast

from pydantic_ai import BinaryContent, ModelRetry, RunContext

from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import REPLY_PROTOCOLS, StructuredOutput, structured_output
from akgentic.core import ActorAddress, Akgent, Orchestrator
from akgentic.core.agent import WarningError
from akgentic.llm import ReactAgent, ReactAgentConfig, UserPrompt
from akgentic.llm import UsageLimitError as LLMUsageLimitError
from akgentic.tool.core import ToolFactory
from akgentic.tool.planning import GetPlanning, GetPlanningTask
from akgentic.tool.team import (
    FireTeamMember,
    GetRoleProfiles,
    GetTeamRoster,
    HireTeamMember,
    TeamTool,
)
from akgentic.tool.workspace import ExpandMediaRefs
from akgentic.tool.workspace.readers import MediaContent

logger = logging.getLogger(__name__)


class RecusionError(Exception):
    pass


MAX_RECURSION_DEPTH = 3

T = TypeVar("T")


class BaseAgent(Akgent[AgentConfig, AgentState]):
    """LLM-powered team agent with delegation and collaboration capabilities.

    Composition:
    - ReactAgent (akgentic-llm): model instantiation, HTTP retry, usage limits,
      context history, REACT loop. All LLM calls delegate to run_sync().
    - ToolFactory (akgentic-tool): aggregates ToolCard[] into tools, system prompts,
      and commands via 3-channel architecture (TOOL_CALL, SYSTEM_PROMPT, COMMAND).
    - TeamTool: auto-injected if absent from config.tools; provides hire/fire
      capabilities and team awareness prompts.

    Observer Protocol:
    - Implements TeamManagementToolObserver (structural typing via @runtime_checkable)
    - Provides createActor(), on_hire(), on_fire(), proxy_ask(), notify_event()
      to ToolFactory/TeamTool without explicit interface inheritance.

    Execution in act():
    - All paths → ReactAgent.run_sync(output_type=T) which wraps T with
      get_output_type() internally and manages context, limits, and REACT loop

    Message Flow (continuation-based routing):
    - UserMessage → act_with_output_and_optional_requests → ResultMessage
    - HelpRequestMessage → act_with_output_or_requests → HelpAnswerMessage
    - HelpAnswerMessage → route via Continuation chain → HelpAnswerMessage/ResultMessage
    - Continuation implements distributed call stack for multi-hop delegation:
      e.g. Human → Manager → Dev → Tester → Dev → Manager → Human

    Structured Output Patterns:
    - MessageOrRequests (either/or): LLM returns EITHER answer OR help requests
    - MessageAndOptionalRequests (both): LLM returns answer AND optional requests

    Tools exposed to LLM (via ToolFactory.get_tools()):
    - hire_members(roles) → list[tuple[str, str]]
    - fire_members(names) → list[str]
    - Additional tools from config.tools ToolCards

    System Prompts (via ToolFactory.get_system_prompts() + @react_agent.system_prompt):
    - current_datetime (from ReactAgent)
    - agent_backstory (from AgentState)
    - GetTeamRoster dynamic prompt (from TeamTool)
    - GetRoleProfiles dynamic prompt (from TeamTool)

    Commands (programmatic, via ToolFactory.get_commands(), prefixed cmd_):
    - cmd_hire_member(role) → tuple[str, ActorAddress] | str
    - cmd_fire_member(name) → str
    - cmd_get_planning() → str
    - cmd_get_team_roster() → str
    - cmd_get_role_profiles() → str
    - cmd_get_planning_task(task_id) → Task | str

    Internal method (used by request()):
    - hire_member(role) → tuple[str, ActorAddress]  (raises ModelRetry on failure)
    """

    def on_start(self) -> None:
        """Initialize BaseAgent using ReactAgent from akgentic-llm.

        ReactAgent internally handles:
        - create_model() / create_model_settings() / create_http_client()
        - ContextManager for conversation history
        - Usage limits conversion
        - current_datetime_prompt system prompt

        Additional dynamic system prompts (backstory, team context, request context)
        are registered via @self._react_agent.system_prompt after construction.

        Tools (hire_members, fire_members) are registered at construction time
        as bound methods — pydantic-ai treats them as plain tools without RunContext.
        Commands (hire_member) are extracted from ToolFactory for programmatic use.
        """
        assert self._orchestrator is not None, "Orchestrator address must be provided in config"
        self.orchestrator_proxy_ask = self.proxy_ask(self._orchestrator, Orchestrator)

        self._current_message: AgentMessage | None = None

        # ── State ───────────────────────────────────────────────────────────────
        self.state = AgentState(backstory=self.config.prompt.render()).observer(self)

        # Private attributes (not serialized)
        self._max_help_requests: int = self.config.max_help_requests

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
            usage_limits=self.config.usage_limits,
        )

        tool_factory = ToolFactory(
            tool_cards=tool_cards,
            observer=self,
            retry_exception=ModelRetry,
        )
        tools = tool_factory.get_tools()
        toolsets = tool_factory.get_toolsets()

        # ── Extract commands for programmatic use ─────────────────────────
        commands = tool_factory.get_commands()
        self._hire_member_command = commands.get(HireTeamMember)
        self._fire_member_command = commands.get(FireTeamMember)
        self._get_planning_command = commands.get(GetPlanning)
        self._get_planning_task_command = commands.get(GetPlanningTask)
        self._get_team_roster_command = commands.get(GetTeamRoster)
        self._get_role_profiles_command = commands.get(GetRoleProfiles)
        self._expand_media_refs_command = commands.get(ExpandMediaRefs)

        self._react_agent = ReactAgent(
            config=react_agent_config,
            deps_type=BaseAgent,
            tools=tools,
            toolsets=toolsets,
            event_loop=self._event_loop,
            observer=self,
        )

        # ── Additional dynamic system prompts ─────────────────────────────────
        # ReactAgent already registers: current_datetime_prompt + config.system_prompts
        @self._react_agent.system_prompt
        def agent_backstory(ctx: RunContext[BaseAgent]) -> str:
            return ctx.deps.state.backstory

        @self._react_agent.system_prompt
        def current_date(ctx: RunContext[BaseAgent]) -> str:
            now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            return f"The current date is {now}."

        for system_prompt in tool_factory.get_system_prompts():
            self._react_agent.system_prompt(system_prompt)

        @self._react_agent.system_prompt
        def mailbox_notifications(ctx: RunContext[BaseAgent]) -> str | None:
            if inbox := ctx.deps.get_mailbox():
                senders = {msg.sender.name for msg in inbox if msg.sender}
                return (
                    f"NOTICE: {len(inbox)} new message(s) arrived in your mailbox "
                    f"from team member(s): {', '.join(senders)}."
                    "\nConsider wrapping up the current thread to process them."
                )

    # ============================================================================
    # CORE LLM INTERACTION
    # ============================================================================

    def act(self, user_content: str, output_type: type[T]) -> T:
        """Execute LLM reasoning with optional structured output.

        Delegates entirely to ReactAgent.run_sync(), which:
        - Manages context history via ContextManager
        - Enforces usage limits
        - Wraps output_type with get_output_type() for provider-aware structured
          output (NativeOutput for OpenAI/Anthropic, raw type for others)
        - Runs the full REACT loop (tools, retries, system prompts)

        Args:
            user_content: User message to process.
            output_type: Optional Pydantic model for structured output.
                        When None, returns str (ReactAgent's default result_type).

        Returns:
            output_type

        Raises:
            LLMUsageLimitError: When usage limits exceeded; sends help request
                to Human and wraps the LLMUsageLimitError.
        """
        # ── Media expansion (!!glob_pattern → BinaryContent) ────────────────────
        prompt: UserPrompt = user_content
        if self._expand_media_refs_command is not None:
            parts = self._expand_media_refs_command(user_content)
            if any(isinstance(p, MediaContent) for p in parts):
                prompt = [
                    BinaryContent(data=p.data, media_type=p.media_type)
                    if isinstance(p, MediaContent)
                    else p
                    for p in parts
                ]
        # ── End media expansion ─────────────────────────────────────────────────

        assert self._current_message is not None
        assert self._current_message.sender is not None

        sender = self._current_message.sender.name
        team = [
            agent.name
            for agent in self.get_team()
            if agent.name != sender and not agent.name.startswith("#")
        ]
        roles = self.get_available_roles()

        # Create a per-call subclass to avoid mutating the shared class __doc__
        # (thread-safety: multiple agents run act() concurrently on separate pykka threads)
        local_output_type = type(
            output_type.__name__,
            (output_type,),
            {
                "__doc__": structured_output.format(
                    sender=sender,
                    message_type=self._current_message.type,
                    reply_protocol=REPLY_PROTOCOLS.get(
                        self._current_message.type, ""
                    ).format(sender=sender),
                    team=", ".join(team) or "no other members",
                    roles=", ".join(roles) or "no roles available",
                )
            },
        )

        output = self._react_agent.run_sync(prompt, deps=self, output_type=local_output_type)

        return cast(T, output)

    def process_message(self, message_content: str, sender: ActorAddress, count: int = 0):
        assert self._current_message is not None
        assert self._current_message.sender is not None

        if count > MAX_RECURSION_DEPTH:
            raise RecusionError()

        output = self.act(message_content, StructuredOutput)

        roles = self.get_available_roles()
        result = []
        member_err = []
        role_err = []

        for request in output.messages:
            recipient = request.recipient

            member = None
            if recipient.startswith("@"):
                if not (member := self.get_team_member(recipient)):
                    logger.error(f"Invalid recipient {recipient}: not found in team members.")
                    member_err.append(recipient)

            elif recipient in roles:
                member = self.hire_member(recipient)

            else:
                logger.error(f"Invalid recipient {recipient}: not found in team roles")
                role_err.append(recipient)

            if member is not None:
                article = "an" if request.message_type[0] in "aeiou" else "a"
                content = (
                    f"You received {article} {request.message_type}"
                    f" from {self.config.name}:\n\n"
                    + request.message
                )
                self.send(
                    member,
                    AgentMessage(
                        content=content,
                        type=request.message_type,
                        recipient=member,
                    ),
                )
                result.append(member.name)

        if member_err or role_err:
            content = f"Message sent successfully to: {'; '.join(result)}.\n"
            content += "Errors sending message to:\n"
            content += (f"- {', '.join(role_err)}, role(s) not found.\n") if role_err else ""
            content += (f"- {', '.join(member_err)}, member(s) not found.") if member_err else ""
            self.process_message(content, sender, count + 1)

    def receiveMsg_AgentMessage(self, message: AgentMessage, sender: ActorAddress) -> None:  # noqa: N802
        """Handle incoming AgentMessage.

        This is a generic handler for messages of type AgentMessage. Depending on the
        content and recipient, this method can be extended to route messages to specific
        handlers or process them directly.

        Args:
            message: The AgentMessage instance containing the message content and recipient.
            sender: The ActorAddress of the sender of the message.

        Returns:
            None: This method can be extended to send responses or route messages as needed.
        """

        logger.info(f"Received AgentMessage from {sender}: {message.content}")

        try:
            sleep(random.uniform(0.25, 0.5))  # Simulate processing delay
            self.process_message(message.content, sender)

        except LLMUsageLimitError as e:
            self.notify_human(
                f"The agent {self.config.name} has exceeded its usage limits ({e}). \n"
                + "Please review the agent's activity and give your instruction."
            )
            raise WarningError(f"LLM usage limit exceeded: {e}")

        except RecusionError as e:
            self.notify_human(
                f"The agent {self.config.name} fails the process the message 3 times: \n"
                + f"{message.content}\n\n"
                + "Please review the agent's activity and give your instruction."
            )
            raise WarningError(f"Recursion error: {e}")

    def notify_human(self, message: str) -> None:
        # Sent request to the first Human agent
        human = next((agent for agent in self.get_team() if agent.role == "human"), None)
        self.send(human, AgentMessage(content=message, recipient=human, type="notification"))

    # ============================================================================
    # TEAM AWARENESS
    # ============================================================================

    def on_hire(self, address: ActorAddress) -> None:
        pass

    def on_fire(self, address: ActorAddress) -> None:
        pass

    def hire_member(self, role: str) -> ActorAddress:
        """Hire a single team member by role via the TeamTool's hire_member command.

        Used internally by request() — raises ModelRetry on failure so the LLM
        can retry with a different role.

        Args:
            role: Role to hire (must exist in agent catalog)

        Returns:
            tuple[str, ActorAddress]: (member_name, member_address) on success.

        Raises:
            RuntimeError: If hire_member command is not available.
            ModelRetry: If role is invalid or hire fails (propagated for LLM retry).
        """
        if self._hire_member_command is None:
            raise RuntimeError("hire_member command not available — TeamTool not configured")

        return self._hire_member_command(role)

    # ============================================================================
    # CMD_ COMMANDS (for programmatic / slash-command use, e.g. agent_team.py)
    # ============================================================================

    def cmd_hire_member(self, role: str) -> ActorAddress | str:
        """Hire a single team member by role (command version).

        Catches ModelRetry and returns the error as a string instead of raising.
        Used by agent_team.py slash commands.

        Args:
            role: Role to hire (must exist in agent catalog)

        Returns:
            tuple[str, ActorAddress]: (member_name, member_address) on success.
            str: Error message on failure.

        Raises:
            RuntimeError: If hire_member command is not available.
        """
        try:
            return self.hire_member(role)
        except ModelRetry as e:
            return str(e)

    def cmd_fire_member(self, name: str) -> str:
        """Fire a team member by name (command version).

        Catches ModelRetry and returns the error as a string instead of raising.
        Used by agent_team.py slash commands.

        Args:
            name: Name of the member to fire (e.g., '@Developer123')

        Returns:
            str: Confirmation message on success, error message on failure.

        Raises:
            RuntimeError: If fire_member command is not available.
        """
        if self._fire_member_command is None:
            raise RuntimeError("fire_member command not available — TeamTool not configured")

        try:
            return self._fire_member_command(name)
        except ModelRetry as e:
            return str(e)

    def cmd_get_planning(self) -> str:
        """Get the full team planning (command version).

        Returns:
            str: Formatted planning text.

        Raises:
            RuntimeError: If get_planning command is not available.
        """
        if self._get_planning_command is None:
            raise RuntimeError("get_planning command not available — PlanningTool not configured")

        return self._get_planning_command()

    def cmd_get_team_roster(self) -> str:
        """Get the current team roster as a formatted string (command version).

        Returns:
            str: Formatted team roster.

        Raises:
            RuntimeError: If get_team_roster command is not available.
        """
        if self._get_team_roster_command is None:
            raise RuntimeError("get_team_roster command not available — TeamTool not configured")

        return self._get_team_roster_command()

    def cmd_get_role_profiles(self) -> str:
        """Get available team roles and their descriptions (command version).

        Returns:
            str: Formatted role profiles.

        Raises:
            RuntimeError: If get_role_profiles command is not available.
        """
        if self._get_role_profiles_command is None:
            raise RuntimeError("get_role_profiles command not available — TeamTool not configured")

        return self._get_role_profiles_command()

    def cmd_get_planning_task(self, task_id: int):
        """Get a single planning task by ID (command version).

        Args:
            task_id: The task ID to retrieve.

        Returns:
            Task or str if not found.

        Raises:
            RuntimeError: If get_planning_task command is not available.
        """
        if self._get_planning_task_command is None:
            raise RuntimeError(
                "get_planning_task command not available — PlanningTool not configured"
            )

        return self._get_planning_task_command(task_id)
