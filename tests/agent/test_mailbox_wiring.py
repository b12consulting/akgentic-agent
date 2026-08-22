"""Tests for mailbox wiring in ``BaseAgent.on_start`` (Epic 20, story 20-1).

The stale ``mailbox_notifications`` system prompt is deleted — under any
mailbox condition, no such prompt is registered and nothing renders the old
"NOTICE: N new message(s)" text. In its place ``MailboxTool`` is auto-added to
the card list handed to ``ToolFactory`` exactly as ``TeamTool`` is: a default
instance is prepended when the config carries none, a config-supplied instance
wins, and ``config.tools`` itself is never mutated.

All specs go through the real actor system (``on_start`` runs on
``createActor``) with the capturing-ReactAgent swap shared by the other
on_start tests; the factory is swapped for a recording subclass so the card
list, the collected providers, and the built command registry — the exact
objects ``on_start`` wires into the agent — are observable without reaching
into actor internals. Mailbox state delivery is live through the Epic 19
delivery loop, so the delivery spec asserts the mailbox rendering reaches the
first turn's context-update block.
"""

import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

from akgentic.core import ActorAddress, ActorSystem, BaseConfig, Orchestrator
from akgentic.llm import ModelConfig, PromptTemplate
from akgentic.tool.core import CommandRegistry, ContextState, ToolCard, ToolFactory
from akgentic.tool.mailbox import MailboxTool
from akgentic.tool.team import TeamTool

import akgentic.agent.agent as agent_module
from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

# =============================================================================
# HELPERS
# =============================================================================


class _CapturingReactAgent:
    """Stands in for ReactAgent; records init kwargs and registered prompts."""

    captured: ClassVar[list[dict[str, object]]] = []
    prompts: ClassVar[list[Callable[..., Any]]] = []
    recorded_blocks: ClassVar[list[str]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).captured.append(kwargs)
        self.context = SimpleNamespace(
            record_operator_action=type(self).recorded_blocks.append, messages=[]
        )

    def system_prompt(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        type(self).prompts.append(fn)
        return fn

    def run_sync(self, prompt: object, **kwargs: object) -> StructuredOutput:
        return StructuredOutput(messages=[])

    def close(self) -> None:
        pass


class _RecordingToolFactory(ToolFactory):
    """Real ToolFactory that records what on_start hands it and takes from it."""

    captured_cards: ClassVar[list[list[ToolCard]]] = []
    captured_providers: ClassVar[list[list[Callable[[], ContextState | None]]]] = []
    captured_registries: ClassVar[list[CommandRegistry]] = []

    def __init__(
        self,
        tool_cards: list[ToolCard],
        observer: Any = None,
        retry_exception: type[Exception] | None = None,
    ) -> None:
        type(self).captured_cards.append(list(tool_cards))
        super().__init__(tool_cards=tool_cards, observer=observer, retry_exception=retry_exception)

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        providers = super().get_context_states()
        type(self).captured_providers.append(providers)
        return providers

    def get_command_registry(
        self, extra_commands: list[Callable[..., Any]] | None = None
    ) -> CommandRegistry:
        registry = super().get_command_registry(extra_commands=extra_commands)
        type(self).captured_registries.append(registry)
        return registry


def _reset_captures() -> None:
    _CapturingReactAgent.captured = []
    _CapturingReactAgent.prompts = []
    _CapturingReactAgent.recorded_blocks = []
    _RecordingToolFactory.captured_cards = []
    _RecordingToolFactory.captured_providers = []
    _RecordingToolFactory.captured_registries = []


def _make_pending_message(sender_name: str = "@Alice") -> AgentMessage:
    """A pending mailbox message with a named sender, as get_mailbox yields it."""
    message = AgentMessage(content="please review the draft", type="request")
    sender = MagicMock(spec=ActorAddress)
    sender.name = sender_name
    message.sender = sender
    return message


def _agent_config(**overrides: object) -> AgentConfig:
    return AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(template="You are a manager."),
        model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
        **overrides,  # type: ignore[arg-type]
    )


def _start_agent(config: AgentConfig, mailbox: list[AgentMessage] | None = None) -> None:
    """Start a BaseAgent through the real actor system with the capturing doubles.

    ``mailbox`` non-``None`` patches ``BaseAgent.get_mailbox`` for the whole
    start, simulating messages already queued in the actor inbox when
    ``on_start`` runs — the stale-closure repro condition.
    """
    _reset_captures()
    system = ActorSystem()
    original_react = agent_module.ReactAgent
    original_factory = agent_module.ToolFactory
    agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[misc, assignment]
    agent_module.ToolFactory = _RecordingToolFactory  # type: ignore[misc]
    patcher = (
        patch.object(BaseAgent, "get_mailbox", return_value=mailbox)
        if mailbox is not None
        else None
    )
    try:
        if patcher is not None:
            patcher.start()
        orch_addr = system.createActor(
            Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
        )
        orchestrator = system.proxy_ask(orch_addr, Orchestrator)
        orchestrator.createActor(BaseAgent, config=config)
        time.sleep(0.5)
        assert _CapturingReactAgent.captured, "ReactAgent was never constructed"
    finally:
        if patcher is not None:
            patcher.stop()
        agent_module.ReactAgent = original_react  # type: ignore[misc]
        agent_module.ToolFactory = original_factory  # type: ignore[misc]
        try:
            system.shutdown(timeout=5)
        except Exception:
            pass


def _mailbox_cards(cards: list[ToolCard]) -> list[MailboxTool]:
    return [card for card in cards if isinstance(card, MailboxTool)]


# =============================================================================
# FR1 — the mailbox_notifications system prompt is gone (stale-closure repro)
# =============================================================================


class TestNoticeDeleted:
    """AC 1: a non-empty mailbox at start registers no mailbox_notifications."""

    def test_non_empty_mailbox_registers_no_mailbox_notifications_prompt(self) -> None:
        _start_agent(_agent_config(), mailbox=[_make_pending_message("@Alice")])

        names = [fn.__name__ for fn in _CapturingReactAgent.prompts]
        assert "mailbox_notifications" not in names

    def test_no_registered_prompt_renders_the_old_notice_text(self) -> None:
        """None of the surviving prompts renders 'NOTICE: N new message(s)'."""
        _start_agent(
            _agent_config(),
            mailbox=[_make_pending_message("@Alice"), _make_pending_message("@Bob")],
        )

        ctx = SimpleNamespace(
            deps=SimpleNamespace(state=SimpleNamespace(backstory="You are a manager."))
        )
        renderings = [fn(ctx) for fn in _CapturingReactAgent.prompts]
        assert all("NOTICE:" not in (text or "") for text in renderings)

    def test_empty_mailbox_registers_no_mailbox_notifications_prompt(self) -> None:
        """Under any mailbox condition — empty included — no such prompt exists."""
        _start_agent(_agent_config(), mailbox=[])

        names = [fn.__name__ for fn in _CapturingReactAgent.prompts]
        assert "mailbox_notifications" not in names


# =============================================================================
# FR2 — MailboxTool auto-add, absent and present cases
# =============================================================================


class TestMailboxToolAutoAdd:
    """AC 2/3: default prepended when absent; a config-supplied card wins."""

    def test_absent_config_gets_exactly_one_default_mailbox_tool(self) -> None:
        config = _agent_config()

        _start_agent(config)

        cards = _RecordingToolFactory.captured_cards[-1]
        mailbox_cards = _mailbox_cards(cards)
        assert len(mailbox_cards) == 1
        assert mailbox_cards[0].read_mailbox is True  # the default instance
        # TeamTool auto-add is untouched by the extension.
        assert sum(isinstance(card, TeamTool) for card in cards) == 1

    def test_absent_config_tools_is_not_mutated(self) -> None:
        config = _agent_config()

        _start_agent(config)

        assert config.tools == []
        assert _RecordingToolFactory.captured_cards[-1] is not config.tools

    def test_present_config_card_wins_and_no_default_is_added(self) -> None:
        customised = MailboxTool(read_mailbox=False)
        config = _agent_config(tools=[customised])

        _start_agent(config)

        cards = _RecordingToolFactory.captured_cards[-1]
        mailbox_cards = _mailbox_cards(cards)
        assert len(mailbox_cards) == 1
        assert mailbox_cards[0].read_mailbox is False  # the user's card, not a default
        assert config.tools == [customised]


# =============================================================================
# AC 4 — the card's capabilities reach the factory outputs wired into the agent
# =============================================================================


class TestMailboxWiring:
    """AC 4: provider collected, read_mailbox tool wired, stop command registered."""

    def test_mailbox_provider_read_tool_and_stop_command_are_wired(self) -> None:
        _start_agent(_agent_config())

        # The provider list on_start assigned to _context_state_providers.
        providers = _RecordingToolFactory.captured_providers[-1]
        assert "mailbox_state" in [provider.__name__ for provider in providers]

        # The tools handed to the ReactAgent include the mailbox peek.
        tools = _CapturingReactAgent.captured[-1]["tools"]
        assert isinstance(tools, list)
        assert "read_mailbox" in [tool.__name__ for tool in tools]

        # The registry on_start assigned to _command_registry carries /stop.
        registry = _RecordingToolFactory.captured_registries[-1]
        assert registry.has("stop")

    def test_mailbox_state_reaches_the_first_context_update_block(self) -> None:
        """Delivery is LIVE: the collected provider feeds the Epic 19 loop.

        With messages pending, the first turn's **Context update** block carries
        the mailbox rendering — not a system prompt, and not nothing.
        """
        _reset_captures()
        system = ActorSystem()
        original_react = agent_module.ReactAgent
        original_factory = agent_module.ToolFactory
        agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[misc, assignment]
        agent_module.ToolFactory = _RecordingToolFactory  # type: ignore[misc]
        patcher = patch.object(
            BaseAgent, "get_mailbox", return_value=[_make_pending_message("@Alice")]
        )
        try:
            patcher.start()
            orch_addr = system.createActor(
                Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
            )
            orchestrator = system.proxy_ask(orch_addr, Orchestrator)
            manager_addr = orchestrator.createActor(BaseAgent, config=_agent_config())
            time.sleep(0.3)

            system.tell(manager_addr, AgentMessage(content="hello"))

            deadline = time.time() + 10
            while not _CapturingReactAgent.recorded_blocks and time.time() < deadline:
                time.sleep(0.2)

            blocks = _CapturingReactAgent.recorded_blocks
            assert len(blocks) == 1, "first turn must deliver exactly one block"
            assert blocks[0].startswith("**Context update 1** — current state.")
            assert "pending from @Alice" in blocks[0]
        finally:
            patcher.stop()
            agent_module.ReactAgent = original_react  # type: ignore[misc]
            agent_module.ToolFactory = original_factory  # type: ignore[misc]
            try:
                system.shutdown(timeout=5)
            except Exception:
                pass
