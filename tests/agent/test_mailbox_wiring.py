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
into actor internals. ``MailboxTool`` contributes **no** context-state
provider: it is a two-channel card, and mailbox awareness reaches the model
through the agent's mid-run arrival notice alone. The delivery spec asserts
that absence against a live Epic 19 loop — with mail pending, the first turn's
context-update block carries the roster and no mailbox rendering.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import ActorAddress, ActorSystem, BaseConfig, Orchestrator
from akgentic.llm import ModelConfig, PromptTemplate
from akgentic.tool.core import CommandRegistry, ContextState, ToolCard, ToolFactory
from akgentic.tool.mailbox import MailboxTool, ReadMailbox
from akgentic.tool.team import TeamTool

import akgentic.agent.agent as agent_module
from akgentic.agent.agent import BaseAgent
from akgentic.agent.capabilities import MailboxCapability
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
            append_user_prompt=type(self).recorded_blocks.append, messages=[]
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


@contextmanager
def _running_agent(
    config: AgentConfig, mailbox: list[AgentMessage] | None = None
) -> Iterator[tuple[ActorSystem, ActorAddress]]:
    """Run a BaseAgent through the real actor system with the capturing doubles.

    Swaps ReactAgent/ToolFactory for the recording doubles, starts an
    orchestrator and one BaseAgent, waits for ``on_start`` to complete, and
    yields the system with the agent's address for specs that message it. The
    doubles stay swapped and the mailbox patch stays active until the with
    block exits, so a turn run inside it sees the same environment ``on_start``
    saw.

    ``mailbox`` non-``None`` patches ``BaseAgent.get_mailbox`` for the whole
    lifetime, simulating messages already queued in the actor inbox when
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
        agent_addr = orchestrator.createActor(BaseAgent, config=config)
        time.sleep(0.5)
        assert _CapturingReactAgent.captured, "ReactAgent was never constructed"
        yield system, agent_addr
    finally:
        if patcher is not None:
            patcher.stop()
        agent_module.ReactAgent = original_react  # type: ignore[misc]
        agent_module.ToolFactory = original_factory  # type: ignore[misc]
        try:
            system.shutdown(timeout=5)
        except Exception:
            pass


def _start_agent(config: AgentConfig, mailbox: list[AgentMessage] | None = None) -> None:
    """Start and stop an agent, leaving the captures behind for assertion."""
    with _running_agent(config, mailbox):
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
# AC 4 — the card's two channels reach the factory outputs wired into the agent
# =============================================================================


class TestMailboxWiring:
    """AC 4: read_mailbox tool wired, stop command registered — and no more."""

    def test_read_tool_and_stop_command_are_wired(self) -> None:
        _start_agent(_agent_config())

        # The tools handed to the ReactAgent include the mailbox read.
        tools = _CapturingReactAgent.captured[-1]["tools"]
        assert isinstance(tools, list)
        assert "read_mailbox" in [tool.__name__ for tool in tools]

        # The registry on_start assigned to _command_registry carries /stop.
        registry = _RecordingToolFactory.captured_registries[-1]
        assert registry.has("stop")

    def test_first_context_update_block_carries_the_roster_and_no_mailbox(self) -> None:
        """Delivery is LIVE, and the mailbox is deliberately absent from it.

        With a message pending, the first turn's **Context update** block still
        arrives and still carries the roster — so the Epic 19 loop is running and
        this is not a vacuous assertion — but nothing in it names the pending
        mail. ``MailboxTool`` serves no ``LLM_CONTEXT``; awareness of that
        message reaches the model through the agent's mid-run arrival notice.
        """
        mailbox = [_make_pending_message("@Alice")]
        with _running_agent(_agent_config(), mailbox) as (system, agent_addr):
            system.tell(agent_addr, AgentMessage(content="hello"))

            deadline = time.time() + 10
            while not _CapturingReactAgent.recorded_blocks and time.time() < deadline:
                time.sleep(0.2)

            blocks = _CapturingReactAgent.recorded_blocks
            assert len(blocks) == 1, "first turn must deliver exactly one block"
            assert blocks[0].startswith("**Context update 1** — current state.")

            # The loop is live: the roster provider's rendering is there.
            assert "team member list" in blocks[0]
            assert "@Manager" in blocks[0]

            # The mailbox is not: no sender, no content, no count wording.
            assert "@Alice" not in blocks[0]
            assert "please review the draft" not in blocks[0]
            assert "pending" not in blocks[0].lower()
            assert "mailbox" not in blocks[0].lower()


# =============================================================================
# Epic 23 — the preview whitelist travels from the card to the capability
# =============================================================================


def _wired_capability() -> MailboxCapability:
    """The MailboxCapability on_start handed to the ReactAgent it built."""
    capabilities = _CapturingReactAgent.captured[-1]["capabilities"]
    assert isinstance(capabilities, list)
    capability = capabilities[0]
    assert isinstance(capability, MailboxCapability)
    return capability


class TestPreviewWhitelistReachesTheCapability:
    """The whitelist is configured on a card and enforced in a capability.

    Nothing else crosses that gap: every offer-rule spec constructs the
    capability with ``preview_handlers=`` directly. A wrong attribute name, a
    card filter that misses, or a dropped constructor argument would leave all
    of them green while every deployment fell back to admitting every handler —
    the permissive direction, and a silent one.

    These specs configure a stock ``MailboxTool`` on purpose. Reading the field
    off the real card is what makes them a guard on the seam rather than on this
    package alone: a rename on the ``akgentic-tool`` side turns them red here,
    which is the only place the mismatch is visible.
    """

    _HANDLER = "akgentic.agent.messages.AgentMessage"

    def test_the_cards_whitelist_is_what_the_capability_enforces(self) -> None:
        card = MailboxTool(mailbox_preview_handlers=[self._HANDLER])

        _start_agent(_agent_config(tools=[card]))

        assert _wired_capability()._preview_handlers == [self._HANDLER]

    def test_an_empty_whitelist_survives_the_trip_as_itself(self) -> None:
        """``[]`` admits no handler and is never coerced to ``None`` on the way."""
        card = MailboxTool(mailbox_preview_handlers=[])

        _start_agent(_agent_config(tools=[card]))

        assert _wired_capability()._preview_handlers == []

    def test_a_card_declaring_no_whitelist_admits_every_handler(self) -> None:
        """The card's own default, reached through the auto-inserted mailbox card."""
        _start_agent(_agent_config())

        assert _wired_capability()._preview_handlers is None


class TestTheNoticeIsGatedOnTheReadTool:
    """Story 26-2: `read_mailbox=False` turns the doorbell off.

    The notice tells the model to call `read_mailbox`. With that capability
    removed the notice is an instruction the model cannot follow, offered on
    every step boundary of every run — so `BaseAgent` reads the card and hands
    `MailboxCapability` the answer.

    These specs go through the **real** `on_start` and read the capability off
    the `capabilities=` kwarg the ReactAgent was built with. An earlier version
    asserted `bool(card.read_mailbox)` instead, which tested Python rather than
    the wiring: hardcoding `arrival_notice=True` at the call site left the whole
    suite green.
    """

    @pytest.mark.parametrize(
        ("card", "announces"),
        [
            pytest.param(MailboxTool(), True, id="default"),
            pytest.param(MailboxTool(read_mailbox=True), True, id="true"),
            pytest.param(MailboxTool(read_mailbox=ReadMailbox()), True, id="param-instance"),
            pytest.param(MailboxTool(read_mailbox=False), False, id="false"),
        ],
    )
    def test_the_card_decides_whether_a_run_announces_mail(
        self, card: MailboxTool, announces: bool
    ) -> None:
        """MUTATION — hardcode `arrival_notice=True` at the wiring site in
        `_assemble_capabilities` and the `false` parameter goes red on its own.
        Every other parameter stays green, because they all expect the doorbell
        to ring, and no spec outside this class touches the flag.
        """
        with _running_agent(_agent_config(tools=[card])):
            capability = _CapturingReactAgent.captured[-1]["capabilities"][0]  # type: ignore[index]

        assert isinstance(capability, MailboxCapability)
        assert capability._arrival_notice is announces

    def test_a_card_without_reads_still_registers_stop(self) -> None:
        """Turning the doorbell off must not take the cancel surface with it.

        They are separate capabilities on separate channels, and a deployment
        that wants no mid-run reads still wants `/stop`. The capability is built
        unconditionally either way, so the run stays interruptible even for an
        agent carrying no `MailboxTool` at all.
        """
        card = MailboxTool(read_mailbox=False, stop=True)

        with _running_agent(_agent_config(tools=[card])):
            capability = _CapturingReactAgent.captured[-1]["capabilities"][0]  # type: ignore[index]

        assert isinstance(capability, MailboxCapability)
        assert capability._arrival_notice is False
        assert card.get_commands(), "the /stop command must survive read_mailbox=False"
