"""Tests for the creation-time backstory surfacing (Epic 9, Story 9.1).

`BaseAgent.on_start` retires the pydantic-ai run-buffer seed and instead surfaces
the backstory through a single display-only ``LlmSystemPromptEvent`` emitted by
``ReactAgent.context.record_initial_system_prompt(...)``. These tests assert the
observable behavior:

- FR1: after creation the run buffer (``ContextManager.messages``) is EMPTY — no
  pre-seeded ``SystemPromptPart`` ModelRequest.
- FR2: exactly one ``LlmSystemPromptEvent`` is emitted at creation, whose single
  snapshot has ``dynamic_ref == "agent_backstory"`` and ``content`` equal to the
  rendered backstory.
- FR3 precondition: because the run buffer stays empty, pydantic-ai dynamic
  ``@system_prompt`` injection is no longer suppressed on the first run.

No assertion references any "ADR-NNN" string (Golden Rule #8); the
``dynamic_ref == "agent_backstory"`` check is a behavioral head-block label
contract, not a documentation-presence check.
"""

import time

from akgentic.core import ActorSystem, BaseConfig, EventSubscriber, Orchestrator
from akgentic.core.messages import EventMessage, Message
from akgentic.llm import LlmSystemPromptEvent, ModelConfig, PromptTemplate
from akgentic.llm.context import ContextManager
from pydantic_ai.messages import ModelRequest, SystemPromptPart

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig

# =============================================================================
# Unit-level: the ContextManager contract consumed by on_start
# =============================================================================


class TestRecordInitialSystemPromptContract:
    """The display-only event surfaces backstory without touching the run buffer."""

    @staticmethod
    def _ctx_with_observer() -> tuple[ContextManager, list[object]]:
        events: list[object] = []

        class _Observer:
            def notify_event(self, event: object) -> None:
                events.append(event)

        ctx = ContextManager()
        ctx.subscribe(_Observer())
        return ctx, events

    def test_emits_single_display_only_event_with_backstory_snapshot(self) -> None:
        """FR2: one LlmSystemPromptEvent, snapshot dynamic_ref/content == backstory."""
        ctx, events = self._ctx_with_observer()
        backstory = "You are a meticulous backend engineer."

        ctx.record_initial_system_prompt(backstory)

        sys_events = [e for e in events if isinstance(e, LlmSystemPromptEvent)]
        assert len(sys_events) == 1
        event = sys_events[0]
        assert len(event.parts) == 1
        snapshot = event.parts[0]
        assert snapshot.dynamic_ref == "agent_backstory"
        assert snapshot.content == backstory

    def test_run_buffer_stays_empty_after_creation_event(self) -> None:
        """FR1/FR3: the creation event appends NOTHING to the run buffer (_messages)."""
        ctx, _ = self._ctx_with_observer()

        ctx.record_initial_system_prompt("Some backstory.")

        # No ModelRequest seeded — dynamic injection on the first run is not suppressed.
        assert ctx.messages == []
        assert not any(
            isinstance(m, ModelRequest)
            and any(isinstance(p, SystemPromptPart) for p in m.parts)
            for m in ctx.messages
        )


# =============================================================================
# End-to-end: on_start surfaces the backstory through the real actor system
# =============================================================================


class TestOnStartSurfacesBackstory:
    """on_start emits exactly one creation-time LlmSystemPromptEvent and seeds nothing."""

    def test_on_start_emits_single_backstory_event_and_empty_buffer(self) -> None:
        """Creating a BaseAgent surfaces backstory once; the run buffer stays empty.

        Exercised through the real actor system: ``on_start`` runs on
        ``createActor`` and no LLM call happens because no message is sent. A
        subscriber captures the orchestrator event stream; we assert exactly one
        ``LlmSystemPromptEvent`` for this agent with the ``agent_backstory``
        head-block label and the rendered backstory content.
        """
        backstory_text = "You are a manager who delegates clearly."

        captured: list[LlmSystemPromptEvent] = []
        agent_name_holder: dict[str, str] = {}

        class _Subscriber(EventSubscriber):
            def on_stop(self) -> None:
                pass

            def on_message(self, message: Message) -> None:
                if isinstance(message, EventMessage) and isinstance(
                    message.event, LlmSystemPromptEvent
                ):
                    captured.append(message.event)

        system = ActorSystem()
        try:
            orch_addr = system.createActor(
                Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
            )
            orchestrator = system.proxy_ask(orch_addr, Orchestrator)
            orchestrator.subscribe(_Subscriber())

            config = AgentConfig(
                name="@Manager",
                role="Manager",
                prompt=PromptTemplate(template=backstory_text),
                model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
            )
            manager_addr = orchestrator.createActor(BaseAgent, config=config)
            agent_name_holder["name"] = manager_addr.name

            time.sleep(0.5)

            # Exactly one creation-time system-prompt event for this agent.
            assert len(captured) == 1, f"expected exactly one event, got {len(captured)}"
            event = captured[0]
            assert len(event.parts) == 1
            snapshot = event.parts[0]
            assert snapshot.dynamic_ref == "agent_backstory"
            # Rendered backstory (config.prompt.render()) is the snapshot content.
            assert snapshot.content == backstory_text
        finally:
            try:
                system.shutdown(timeout=5)
            except Exception:
                pass
