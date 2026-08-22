"""Tests for the agent's half of context-update delivery (Epics 19 and 21).

What ``BaseAgent`` still owns after Epic 21 is the *when* and the *how*: one
``ContextUpdater``, obtained from the factory at ``on_start`` and held for the
agent's lifetime; ``_deliver_context_update`` as the single delivery site at the
top of every ``act()`` turn; and the append through
``ContextManager.record_operator_action`` rather than a bare ``ModelRequest``,
so a fresh agent's first block is folded into the first run's prompt instead of
suppressing system-prompt injection.

The composition itself — reading providers, diffing against baselines,
reconciling against the visible history, the block grammar, the per-provider
degradation — moved to ``akgentic.tool.core.ContextUpdater`` and is tested
there. The specs below deliberately drive a *real* updater rather than a stub,
so a wiring change that breaks the pairing fails here.

Most specs use the bare-agent + stubbed-ReactAgent pattern shared by the other
unit tests in this package; the first-run fold specs use the real akgentic-llm
``ContextManager`` because the buffer-vs-append decision is the property under
test; the on_start spec goes through the real actor system.
"""

import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import MagicMock

from akgentic.llm import ContextManager, ReactAgent
from akgentic.tool.core import ContextState, ContextUpdater
from pydantic_ai.messages import ModelRequest, UserPromptPart

import akgentic.agent.agent as agent_module
from akgentic.agent.agent import BaseAgent, MailboxCapability
from akgentic.agent.config import AgentConfig, AgentState

# =============================================================================
# HELPERS
# =============================================================================


class _RosterState(ContextState):
    """Roster-shaped test state: full lists members, delta names only joins."""

    members: tuple[str, ...] = ()

    def render_full(self) -> str:
        if not self.members:
            return ""
        return "**Team roster:**\n" + "\n".join(self.members)

    def render_delta(self, previous: Self) -> str | None:
        joined = [m for m in self.members if m not in previous.members]
        if not joined:
            return None
        return ", ".join(joined) + " joined the team."


class _PlanningState(ContextState):
    """Planning-shaped test state: sentences joined with spaces, by design."""

    tasks: tuple[str, ...] = ()

    def render_full(self) -> str:
        if not self.tasks:
            return ""
        return "**Planning:** " + " ".join(self.tasks)

    def render_delta(self, previous: Self) -> str | None:
        added = [t for t in self.tasks if t not in previous.tasks]
        if not added:
            return None
        return "New tasks: " + " ".join(added)


def _provider(name: str, holder: dict[str, ContextState | None]) -> Callable[[], Any]:
    """A named provider reading its current state from a mutable holder."""

    def provider() -> ContextState | None:
        return holder["state"]

    provider.__name__ = name
    return provider


def _make_agent(providers: list[Callable[[], Any]] | None = None) -> BaseAgent:
    """Bare BaseAgent (no Pykka) with a stubbed ReactAgent and a real updater.

    The stubbed context carries a real ``messages`` list, and its
    ``record_operator_action`` appends each delivered block as a user-role
    message (the post-first-run shape) — so the updater's reconciliation finds
    the marker on later turns exactly as it would against the real
    ContextManager.

    The agent gets a real ``AgentState``, because the updater dereferences
    ``observer.state.tool_state`` on every call, and a real ``ContextUpdater``
    built over this agent — which the updater holds *weakly*, so the caller
    must keep the returned agent alive for the duration of the test.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._react_agent = MagicMock()  # type: ignore[attr-defined]
    messages: list[Any] = []
    agent._react_agent.context.messages = messages  # type: ignore[attr-defined]
    agent._react_agent.context.record_operator_action.side_effect = (  # type: ignore[attr-defined]
        lambda entry: messages.append(ModelRequest(parts=[UserPromptPart(content=entry)]))
    )

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]
    agent._context_updater = ContextUpdater(  # type: ignore[attr-defined]
        agent, providers or []
    )

    # Mailbox capability normally built in _build_react_agent (Epic 20).
    agent._mailbox_capability = MailboxCapability(observer=agent)  # type: ignore[arg-type]
    return agent


def _recorded_blocks(agent: BaseAgent) -> list[str]:
    record = agent._react_agent.context.record_operator_action  # type: ignore[attr-defined]
    return [call.args[0] for call in record.call_args_list]


# =============================================================================
# AC 3 / AC 6 — no baseline state on the agent; one updater holds it all
# =============================================================================


class TestNoAgentSideBaselineState:
    def test_the_deleted_engine_members_are_gone(self) -> None:
        """The deleted engine members, asserted where they would actually live.

        The methods belong to the class and the marker pattern to the module,
        so checking them there is what makes this spec falsifiable: restore the
        engine and it goes red. The three deleted *instance* fields cannot be
        checked on the bare fixture below — ``on_start`` is what used to create
        them, and a fixture that never runs ``on_start`` would report them
        absent either way. They are covered behaviourally instead: by the next
        spec, and by the restore specs, which only pass while the counter lives
        on the persisted slot.
        """
        for gone in (
            "_verify_context_baselines",
            "_iter_user_prompt_texts",
            "_render_context_section",
            "_compose_context_update",
        ):
            assert not hasattr(BaseAgent, gone), f"BaseAgent still carries {gone}"
        assert not hasattr(agent_module, "_CONTEXT_UPDATE_MARKER"), (
            "the marker pattern belongs to the engine's package now"
        )

    def test_delivery_caches_nothing_of_its_own_on_the_agent(self) -> None:
        """A delivered turn leaves the updater as the agent's only context state."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        agent._deliver_context_update()

        assert {name for name in vars(agent) if name.startswith("_context")} == {
            "_context_updater"
        }

    def test_delivery_advances_the_persisted_slot_not_an_agent_field(self) -> None:
        """The counter and baselines advance on ``state.tool_state``."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        agent._deliver_context_update()

        assert agent.state.tool_state.context_update_seq == 1
        assert "team_roster_state" in agent.state.tool_state.context_baselines


# =============================================================================
# AC 4 — thin delivery: compose, then append when there is something to append
# =============================================================================


class TestThinDelivery:
    def test_a_composed_block_is_appended_through_record_operator_action(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        agent._deliver_context_update()

        blocks = _recorded_blocks(agent)
        assert len(blocks) == 1
        assert blocks[0] == (
            "**Context update 1** — current state.\n\n**Team roster:**\n@Manager"
        )

    def test_nothing_to_say_appends_nothing(self) -> None:
        """``compose_update`` returning ``None`` must not reach the context."""
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        agent._deliver_context_update()
        holder["state"] = _RosterState(members=("@Manager",))
        agent._deliver_context_update()

        assert len(_recorded_blocks(agent)) == 1

    def test_the_agent_composes_no_text_of_its_own(self) -> None:
        """Whatever the updater returns is appended verbatim, unwrapped."""
        agent = _make_agent()
        agent._context_updater = MagicMock()  # type: ignore[attr-defined]
        agent._context_updater.compose_update.return_value = "SENTINEL BLOCK"  # type: ignore[attr-defined]

        agent._deliver_context_update()

        assert _recorded_blocks(agent) == ["SENTINEL BLOCK"]

    def test_the_updater_is_handed_the_live_context_messages(self) -> None:
        """Reconciliation needs the history, so the messages must be passed."""
        agent = _make_agent()
        agent._context_updater = MagicMock()  # type: ignore[attr-defined]
        agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

        agent._deliver_context_update()

        passed = agent._context_updater.compose_update.call_args.args[0]  # type: ignore[attr-defined]
        assert passed is agent._react_agent.context.messages  # type: ignore[attr-defined]
        assert _recorded_blocks(agent) == []


# =============================================================================
# AC 4 — single delivery site: top of act(), before run_sync
# =============================================================================


class TestActDeliverySite:
    def test_act_delivers_before_run_sync(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])

        order: list[str] = []
        record = agent._react_agent.context.record_operator_action  # type: ignore[attr-defined]
        record.side_effect = lambda entry: order.append("record")
        agent._react_agent.run_sync.side_effect = (  # type: ignore[attr-defined]
            lambda *a, **k: order.append("run_sync") or "response"
        )

        agent.act("hello", output_type=str)

        assert order == ["record", "run_sync"]


# =============================================================================
# AC 10 — delivery primitive and the pre-first-run fold (real manager)
# =============================================================================


class TestFirstRunFold:
    def test_fresh_agent_block_is_buffered_and_folded_not_a_standalone_message(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        context = ContextManager()
        agent._react_agent.context = context  # type: ignore[attr-defined]

        agent._deliver_context_update()

        # No standalone message: the message_history handed to the first iter()
        # is context.messages, and it must still be empty so pydantic-ai's
        # system-prompt injection is not suppressed.
        assert context.messages == []

        # The block reaches the model folded into the run's user prompt, by the
        # real ReactAgent fold over the real buffer.
        folded = ReactAgent._fold_pending_operator_actions(
            SimpleNamespace(_context=context),  # type: ignore[arg-type]
            "hello",
        )
        assert isinstance(folded, str)
        assert folded.startswith("**Context update 1** — current state.")
        assert folded.endswith("hello")
        assert context.messages == []  # folding never materializes history

    def test_after_first_run_block_is_appended_as_user_role_message(self) -> None:
        holder: dict[str, ContextState | None] = {"state": _RosterState(members=("@Manager",))}
        agent = _make_agent([_provider("team_roster_state", holder)])
        context = ContextManager()
        agent._react_agent.context = context  # type: ignore[attr-defined]

        # Simulate a completed first run: history is non-empty now.
        context.add_message(ModelRequest(parts=[UserPromptPart(content="earlier turn")]))

        agent._deliver_context_update()

        assert len(context.messages) == 2
        appended = context.messages[-1]
        assert isinstance(appended, ModelRequest)
        part = appended.parts[0]
        assert isinstance(part, UserPromptPart)
        assert str(part.content).startswith("**Context update 1** — current state.")


# =============================================================================
# AC 3 — on_start builds one updater from the factory (real actor system)
# =============================================================================


class TestOnStartWiring:
    """on_start holds one ``tool_factory.get_context_updater()`` for the lifetime.

    Exercised through the real actor system (on_start runs on createActor) with
    a capturing fake ReactAgent, and asserted through behavior: a card exposing
    a provider gets its rendering delivered in the first turn's block. That
    round trip also proves the updater is built *after* ``self.state`` is
    assigned — it dereferences ``state.tool_state`` on the very first call, and
    the default ``BaseState`` an agent carries before ``on_start`` has no such
    slot.
    """

    def test_card_provider_is_collected_and_delivered_on_first_turn(self) -> None:
        from akgentic.core import ActorSystem, BaseConfig, Orchestrator
        from akgentic.llm import ModelConfig, PromptTemplate
        from akgentic.tool.core import ToolCard

        import akgentic.agent.agent as agent_module
        from akgentic.agent.messages import AgentMessage
        from akgentic.agent.output_models import StructuredOutput

        class _StateCard(ToolCard):
            def get_tools(self) -> list[Callable[..., Any]]:
                return []

            def get_context_states(self) -> list[Callable[[], ContextState | None]]:
                def demo_state() -> ContextState | None:
                    return _PlanningState(tasks=("ID 9 [demo]",))

                return [demo_state]

        recorded: list[str] = []

        class _CapturingReactAgent:
            def __init__(self, **kwargs: object) -> None:
                self.context = SimpleNamespace(
                    record_operator_action=recorded.append, messages=[]
                )

            def system_prompt(self, fn: object) -> object:
                return fn

            def run_sync(self, prompt: object, **kwargs: object) -> StructuredOutput:
                return StructuredOutput(messages=[])

            def close(self) -> None:
                pass

        system = ActorSystem()
        original = agent_module.ReactAgent
        agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[misc, assignment]
        try:
            orch_addr = system.createActor(
                Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
            )
            orchestrator = system.proxy_ask(orch_addr, Orchestrator)

            config = AgentConfig(
                name="@Manager",
                role="Manager",
                prompt=PromptTemplate(template="You are a manager."),
                model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
                tools=[_StateCard()],
            )
            manager_addr = orchestrator.createActor(BaseAgent, config=config)
            time.sleep(0.3)

            system.tell(manager_addr, AgentMessage(content="hello"))

            deadline = time.time() + 10
            while not recorded and time.time() < deadline:
                time.sleep(0.2)

            assert len(recorded) == 1, "first turn must deliver exactly one block"
            assert recorded[0].startswith("**Context update 1** — current state.")
            assert "**Planning:** ID 9 [demo]" in recorded[0]
        finally:
            agent_module.ReactAgent = original  # type: ignore[misc]
            try:
                system.shutdown(timeout=5)
            except Exception:
                pass
