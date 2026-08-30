"""``ModelTool`` on a real agent: the card reaches both surfaces, and only when asked.

Two claims, and the second is the one with teeth.

**It works when configured.** An agent whose ``config.tools`` carries a
``ModelTool`` gets ``list_models`` and ``switch_model`` in its ``CommandRegistry``
and in the descriptors of the single ``CommandsAnnouncedEvent`` ``on_start``
emits, and its turns carry an ``LLM_CONTEXT`` block naming the model in force. No
source change was needed for any of that — it is a proof that the existing factory
path carries the new card, not a feature this story added.

**It changes nothing when it is not.** ``ModelTool`` is opt-in by decision:
``BaseAgent`` auto-adds ``TeamTool`` and ``MailboxTool``, and granting every agent
the standing power to change its own model is a cost and governance decision that
belongs to whoever writes the card list. So an agent without the card announces
neither command, composes no block naming a model, and attempts no switch.

The harness is ``test_mailbox_wiring.py``'s: the real actor system, ``on_start``
running on ``createActor``, a capturing ``ReactAgent`` and a recording
``ToolFactory`` so the exact objects ``on_start`` wired are observable without
reaching into actor internals. The factory also captures the **observer**
``on_start`` handed it — a fully constructed ``BaseAgent``, which is where the
protocol-conformance claim is re-checked with nothing hand-populated.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, ClassVar

from akgentic.core import ActorAddress, ActorSystem, BaseConfig, EventSubscriber, Orchestrator
from akgentic.core.messages import EventMessage, Message
from akgentic.llm import ModelConfig, PromptTemplate
from akgentic.tool.core import CommandRegistry, ToolCard, ToolFactory
from akgentic.tool.core.event import CommandsAnnouncedEvent
from akgentic.tool.mailbox import MailboxTool
from akgentic.tool.model import ModelSwitchToolObserver, ModelTool
from akgentic.tool.team import TeamTool

import akgentic.agent.agent as agent_module
from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

FAST = ModelConfig(provider="openai", model="gpt-5-mini", context_length=128_000)
CHEAP = ModelConfig(provider="mistral", model="mistral-small", context_length=32_000)

FAST_KEY = "openai:gpt-5-mini"

# =============================================================================
# HELPERS
# =============================================================================


class _CapturingReactAgent:
    """Stands in for ReactAgent, answering the roster questions off its config.

    The roster answers are read from the ``ReactAgentConfig`` ``on_start`` built,
    so the block a turn composes reflects the agent's real configuration rather
    than a value the test chose.
    """

    captured: ClassVar[list[dict[str, object]]] = []
    recorded_blocks: ClassVar[list[str]] = []
    switch_calls: ClassVar[list[str]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).captured.append(kwargs)
        config: Any = kwargs["config"]
        self._active: ModelConfig = config.model_cfg
        self._roster: list[ModelConfig] = list(config.model_roster)
        self.context = SimpleNamespace(
            append_user_prompt=type(self).recorded_blocks.append, messages=[]
        )

    def active_model(self) -> ModelConfig:
        return self._active

    def model_roster(self) -> list[ModelConfig]:
        return list(self._roster)

    def switch_model(self, key: str) -> ModelConfig:
        type(self).switch_calls.append(key)
        return self._active

    def system_prompt(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    def run_sync(self, prompt: object, **kwargs: object) -> StructuredOutput:
        return StructuredOutput(messages=[])

    def close(self) -> None:
        pass


class _RecordingToolFactory(ToolFactory):
    """Real ToolFactory recording what on_start hands it and takes from it."""

    captured_cards: ClassVar[list[list[ToolCard]]] = []
    captured_observers: ClassVar[list[Any]] = []
    captured_registries: ClassVar[list[CommandRegistry]] = []

    def __init__(
        self,
        tool_cards: list[ToolCard],
        observer: Any = None,
        retry_exception: type[Exception] | None = None,
    ) -> None:
        type(self).captured_cards.append(list(tool_cards))
        type(self).captured_observers.append(observer)
        super().__init__(tool_cards=tool_cards, observer=observer, retry_exception=retry_exception)

    def get_command_registry(
        self, extra_commands: list[Callable[..., Any]] | None = None
    ) -> CommandRegistry:
        registry = super().get_command_registry(extra_commands=extra_commands)
        type(self).captured_registries.append(registry)
        return registry


class _AnnouncementSubscriber(EventSubscriber):
    """Collects every ``CommandsAnnouncedEvent`` the orchestrator routes."""

    def __init__(self, sink: list[CommandsAnnouncedEvent]) -> None:
        self._sink = sink

    def on_stop(self) -> None:
        pass

    def on_message(self, message: Message) -> None:
        if isinstance(message, EventMessage) and isinstance(
            message.event, CommandsAnnouncedEvent
        ):
            self._sink.append(message.event)


def _reset_captures() -> None:
    _CapturingReactAgent.captured = []
    _CapturingReactAgent.recorded_blocks = []
    _CapturingReactAgent.switch_calls = []
    _RecordingToolFactory.captured_cards = []
    _RecordingToolFactory.captured_observers = []
    _RecordingToolFactory.captured_registries = []


def _agent_config(**overrides: object) -> AgentConfig:
    fields: dict[str, object] = {
        "name": "@Manager",
        "role": "Manager",
        "prompt": PromptTemplate(template="You are a manager."),
        "model_cfg": FAST,
        "model_roster": [FAST, CHEAP],
    }
    fields.update(overrides)
    return AgentConfig(**fields)  # type: ignore[arg-type]


@contextmanager
def _running_agent(
    config: AgentConfig,
) -> Iterator[tuple[ActorSystem, ActorAddress, list[CommandsAnnouncedEvent]]]:
    """Run a BaseAgent through the real actor system with the capturing doubles."""
    _reset_captures()
    announced: list[CommandsAnnouncedEvent] = []
    system = ActorSystem()
    original_react = agent_module.ReactAgent
    original_factory = agent_module.ToolFactory
    agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[misc, assignment]
    agent_module.ToolFactory = _RecordingToolFactory  # type: ignore[misc]
    try:
        orch_addr = system.createActor(
            Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
        )
        orchestrator = system.proxy_ask(orch_addr, Orchestrator)
        orchestrator.subscribe(_AnnouncementSubscriber(announced))
        agent_addr = orchestrator.createActor(BaseAgent, config=config)
        time.sleep(0.5)
        assert _CapturingReactAgent.captured, "ReactAgent was never constructed"
        yield system, agent_addr, announced
    finally:
        agent_module.ReactAgent = original_react  # type: ignore[misc]
        agent_module.ToolFactory = original_factory  # type: ignore[misc]
        try:
            system.shutdown(timeout=5)
        except Exception:
            pass


def _announced_command_names(announced: list[CommandsAnnouncedEvent]) -> set[str]:
    assert len(announced) == 1, f"expected exactly one announcement, got {len(announced)}"
    return {descriptor.name for descriptor in announced[0].commands}


def _run_one_turn(system: ActorSystem, agent_addr: ActorAddress) -> list[str]:
    """Send one message and return the context-update blocks the turn delivered."""
    system.tell(agent_addr, AgentMessage(content="hello"))
    deadline = time.time() + 10
    while not _CapturingReactAgent.recorded_blocks and time.time() < deadline:
        time.sleep(0.2)
    return list(_CapturingReactAgent.recorded_blocks)


# =============================================================================
# AC 1 (companion) — the shipped agent conforms, with nothing hand-populated
# =============================================================================


class TestConformanceOnARealAgent:
    def test_the_observer_on_start_hands_the_factory_satisfies_the_protocol(self) -> None:
        """The instance ``ModelTool`` will actually hold, checked as it holds it.

        ``test_model_switch_observer.py`` makes the same claim about a bare
        instance whose three actor attributes are supplied by hand; this one
        makes it about an agent the actor system really built.
        """
        with _running_agent(_agent_config(tools=[ModelTool()])):
            observer = _RecordingToolFactory.captured_observers[-1]

        assert isinstance(observer, BaseAgent)
        assert isinstance(observer, ModelSwitchToolObserver)


# =============================================================================
# AC 14 — the card reaches the registry and the announcement
# =============================================================================


class TestTheCardReachesBothSurfaces:
    def test_both_commands_are_registered_and_announced(self) -> None:
        """No source change should be needed for this — it proves the path.

        If it goes red, the defect is in ``ModelTool.get_commands()``'s wiring on
        the tool side, not something to patch from the agent side.
        """
        with _running_agent(_agent_config(tools=[ModelTool()])) as (_, _, announced):
            registry = _RecordingToolFactory.captured_registries[-1]

            assert registry.has("list_models")
            assert registry.has("switch_model")
            assert {"list_models", "switch_model"} <= _announced_command_names(announced)

    def test_a_turn_carries_a_context_block_naming_the_model_in_force(self) -> None:
        """The third capability, on the third channel, through the real card."""
        with _running_agent(_agent_config(tools=[ModelTool()])) as (system, addr, _):
            blocks = _run_one_turn(system, addr)

        assert blocks, "the first turn must deliver a context-update block"
        assert FAST_KEY in blocks[0]


# =============================================================================
# AC 15 — an agent without the card is unchanged
# =============================================================================


class TestWithoutTheCard:
    def test_neither_command_is_registered_or_announced(self) -> None:
        with _running_agent(_agent_config()) as (_, _, announced):
            registry = _RecordingToolFactory.captured_registries[-1]

            assert not registry.has("list_models")
            assert not registry.has("switch_model")
            names = _announced_command_names(announced)
            assert "list_models" not in names
            assert "switch_model" not in names
            # Non-vacuous: the announcement itself is live and carries the
            # auto-injected cards' own commands.
            assert "hire_member" in names
            assert "stop" in names

    def test_no_context_block_names_a_model(self) -> None:
        """The same roster, the same turn — and no model block, because no card.

        Paired with the AC-14 spec above, which delivers exactly such a block
        from the same configuration plus the card.
        """
        with _running_agent(_agent_config()) as (system, addr, _):
            blocks = _run_one_turn(system, addr)

        assert blocks, "the first turn must still deliver a context-update block"
        assert "@Manager" in blocks[0]  # the team roster block is live
        assert FAST_KEY not in blocks[0]
        assert "Active model" not in blocks[0]

    def test_a_single_model_agent_attempts_no_switch(self) -> None:
        """Empty roster, empty slot: the restore hook does nothing at all."""
        with _running_agent(_agent_config(model_roster=[])) as (system, addr, _):
            _run_one_turn(system, addr)

        assert _CapturingReactAgent.switch_calls == []

    def test_model_tool_is_in_no_auto_injection_list(self) -> None:
        """The auto-added set is exactly ``MailboxTool`` and ``TeamTool``.

        ``ModelTool`` is opt-in by decision: every agent holding the standing
        power to change its own model is a cost and governance choice, not a
        default.
        """
        config = _agent_config()

        with _running_agent(config):
            cards = _RecordingToolFactory.captured_cards[-1]

        assert {type(card) for card in cards} == {TeamTool, MailboxTool}
        assert not any(isinstance(card, ModelTool) for card in cards)
        assert config.tools == []  # and config.tools itself was never mutated
