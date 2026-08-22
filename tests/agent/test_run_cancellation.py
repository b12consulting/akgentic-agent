"""Tests for run cancellation and the mid-run arrival notice (Epic 20).

Two layers, matching the design's two halves:

The capability list itself is pinned in ``TestCapabilityWiring`` (assembly and
order, both build sites) and in ``TestExtraCapabilityFires`` (a subclass's own
capability actually running inside a real run).

- Hook level: ``MailboxCapability.before_model_request`` is invoked
  directly against a double exposing ``get_mailbox()``, a recording ``ctx``
  double exposing ``enqueue``, and a fabricated request context — no LLM, no
  actor. This is where the cancel-before-notice ordering, the
  announce-once/growth-only tracking, and the enqueue contract (one
  ``ctx.enqueue(notice, priority="asap")`` per growth, nothing appended by
  the hook itself) are pinned. Durable delivery through pydantic-ai's drain
  is pinned by the real-chain test in ``test_arrival_notice_durability.py``.
- Actor level: the catch site in ``receiveMsg_AgentMessage`` and the
  ``receiveMsg_CancelMessage`` handler run through the real actor system with
  a ReactAgent double whose ``run_sync`` raises ``RunInterruptedError`` on
  demand — simulating the capability raising mid-run.
"""

import logging
import sys
import time
import types
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import ActorAddress, ActorSystem, BaseConfig, Orchestrator
from akgentic.core.messages import CancelMessage
from akgentic.llm import ModelConfig, PromptTemplate, ReactAgent, ReactAgentConfig
from pydantic_ai import AgentCapability
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

import akgentic.agent
import akgentic.agent.agent as agent_module
from akgentic.agent import RunInterruptedError
from akgentic.agent.agent import BaseAgent, MailboxCapability
from akgentic.agent.capabilities import render_arrival_notice
from akgentic.agent.config import AgentConfig
from akgentic.agent.custom_agent import CustomAgent, TriageMessage, TriageOutput
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

AGENT_LOGGER = "akgentic.agent.agent"

# =============================================================================
# HELPERS — hook level
# =============================================================================


class _MailboxDouble:
    """Exposes ``get_mailbox()`` over a mutable pending list (no actor)."""

    def __init__(self, pending: list[Any] | None = None) -> None:
        self.pending: list[Any] = list(pending or [])

    def get_mailbox(self) -> list[Any]:
        return list(self.pending)


def _pending_message(content: str = "please review", sender_name: str = "@Alice") -> AgentMessage:
    message = AgentMessage(content=content, type="request")
    sender = MagicMock(spec=ActorAddress)
    sender.name = sender_name
    message.sender = sender
    return message


def _context(messages: list[Any] | None = None) -> Any:
    """Fabricated ModelRequestContext double: ``messages`` is a plain list."""
    return SimpleNamespace(messages=messages if messages is not None else [])


class _CtxDouble:
    """Recording RunContext double: exposes ``enqueue`` and records every call."""

    def __init__(self) -> None:
        self.enqueue_calls: list[tuple[tuple[Any, ...], Any]] = []

    def enqueue(self, *content: Any, priority: Any = "asap") -> str:
        self.enqueue_calls.append((content, priority))
        return f"enqueue-{len(self.enqueue_calls)}"


# =============================================================================
# FR3 — RunInterruptedError declaration and export
# =============================================================================


class TestRunInterruptedError:
    def test_is_a_plain_exception_with_message_only(self) -> None:
        error = RunInterruptedError("cancelled")
        assert isinstance(error, Exception)
        assert str(error) == "cancelled"

    def test_exported_from_public_api(self) -> None:
        assert "RunInterruptedError" in akgentic.agent.__all__
        assert akgentic.agent.RunInterruptedError is RunInterruptedError


# =============================================================================
# FR4 — the cancel check
# =============================================================================


class TestCancelCheck:
    async def test_pending_stop_raises(self) -> None:
        capability = MailboxCapability(observer=_MailboxDouble([_pending_message("/stop")]))

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

    async def test_pending_cancel_message_raises(self) -> None:
        capability = MailboxCapability(observer=_MailboxDouble([CancelMessage()]))

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

    async def test_cancel_buried_behind_other_mail_still_raises(self) -> None:
        pending = [_pending_message("hello"), _pending_message("/stop now", "@Bob")]
        capability = MailboxCapability(observer=_MailboxDouble(pending))

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(_CtxDouble(), _context())

    async def test_cancel_check_runs_before_the_notice(self) -> None:
        """A pending /stop raises — the new mail beside it is never announced."""
        pending = [_pending_message("hello"), _pending_message("/stop")]
        capability = MailboxCapability(observer=_MailboxDouble(pending))
        ctx = _CtxDouble()

        with pytest.raises(RunInterruptedError):
            await capability.before_model_request(ctx, _context())

        assert ctx.enqueue_calls == []
        assert capability._announced_ids == set()

    async def test_empty_mailbox_neither_raises_nor_enqueues(self) -> None:
        capability = MailboxCapability(observer=_MailboxDouble())
        ctx = _CtxDouble()
        context = _context()

        result = await capability.before_model_request(ctx, context)

        assert result is context
        assert ctx.enqueue_calls == []


# =============================================================================
# FR4c — the mid-run arrival notice, delivered via ctx.enqueue
# =============================================================================


class TestArrivalNotice:
    async def test_growth_enqueues_one_notice_at_asap_priority(self) -> None:
        """One growth, one ``ctx.enqueue`` call — the hook itself appends nothing."""
        arrived = _pending_message("news", "@Alice")
        capability = MailboxCapability(observer=_MailboxDouble([arrived]))
        existing = ModelRequest(parts=[UserPromptPart(content="the turn prompt")])
        existing_parts = existing.parts
        ctx = _CtxDouble()
        context = _context([existing])

        result = await capability.before_model_request(ctx, context)

        assert result is context
        assert context.messages == [existing]  # delivery is the drain's job, not the hook's
        assert existing.parts is existing_parts  # no part-level mutation
        assert ctx.enqueue_calls == [((render_arrival_notice([arrived]),), "asap")]

    async def test_same_message_is_announced_once_across_firings(self) -> None:
        arrived = _pending_message()
        capability = MailboxCapability(observer=_MailboxDouble([arrived]))
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())
        await capability.before_model_request(ctx, _context())

        assert len(ctx.enqueue_calls) == 1

    async def test_second_arrival_announces_only_the_growth(self) -> None:
        first = _pending_message("one", "@Alice")
        second = _pending_message("two", "@Bob")
        mailbox = _MailboxDouble([first])
        capability = MailboxCapability(observer=mailbox)
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())
        mailbox.pending.append(second)
        await capability.before_model_request(ctx, _context())

        assert len(ctx.enqueue_calls) == 2
        growth_content, growth_priority = ctx.enqueue_calls[-1]
        assert growth_content == (render_arrival_notice([second]),)
        assert growth_priority == "asap"

    async def test_reset_run_tracking_forgets_the_announced_backlog(self) -> None:
        arrived = _pending_message()
        capability = MailboxCapability(observer=_MailboxDouble([arrived]))
        ctx = _CtxDouble()

        await capability.before_model_request(ctx, _context())
        capability.reset_run_tracking()
        await capability.before_model_request(ctx, _context())

        assert len(ctx.enqueue_calls) == 2  # the backlog re-announced after reset

    async def test_no_growth_enqueues_nothing(self) -> None:
        arrived = _pending_message()
        mailbox = _MailboxDouble([arrived])
        capability = MailboxCapability(observer=mailbox)
        await capability.before_model_request(_CtxDouble(), _context())

        ctx = _CtxDouble()
        await capability.before_model_request(ctx, _context())

        assert ctx.enqueue_calls == []


# =============================================================================
# HELPERS — actor level
# =============================================================================


class _InterruptibleReactAgent:
    """ReactAgent double: records wiring; ``run_sync`` raises on demand.

    ``interrupts_remaining`` simulates the mailbox capability raising
    ``RunInterruptedError`` out of ``run_sync`` mid-run — the doubles never
    drive pydantic-ai's capability chain, so the raise is injected here.
    """

    captured: ClassVar[list[dict[str, Any]]] = []
    recorded_blocks: ClassVar[list[str]] = []
    run_calls: ClassVar[int] = 0
    interrupts_remaining: ClassVar[int] = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self).captured.append(kwargs)
        self.context = SimpleNamespace(
            record_operator_action=type(self).recorded_blocks.append, messages=[]
        )

    def system_prompt(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    def run_sync(self, prompt: object, **kwargs: object) -> StructuredOutput:
        cls = type(self)
        cls.run_calls += 1
        if cls.interrupts_remaining > 0:
            cls.interrupts_remaining -= 1
            raise RunInterruptedError("cancelled by a queued /stop or CancelMessage")
        return StructuredOutput(messages=[])

    def close(self) -> None:
        pass


def _reset_captures(interrupts: int = 0) -> None:
    _InterruptibleReactAgent.captured = []
    _InterruptibleReactAgent.recorded_blocks = []
    _InterruptibleReactAgent.run_calls = 0
    _InterruptibleReactAgent.interrupts_remaining = interrupts


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="@Manager",
        role="Manager",
        prompt=PromptTemplate(template="You are a manager."),
        model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
    )


@contextmanager
def _running_agent(
    interrupts: int = 0, agent_class: type[BaseAgent] = BaseAgent
) -> Iterator[tuple[ActorSystem, ActorAddress, ActorAddress]]:
    """Run an agent through the real actor system with the interruptible double.

    ``agent_class`` defaults to ``BaseAgent``; ``CustomAgent`` is passed in to
    exercise a *subclass* handler, which is the shape the exemplar has.
    """
    _reset_captures(interrupts)
    system = ActorSystem()
    original_react = agent_module.ReactAgent
    agent_module.ReactAgent = _InterruptibleReactAgent  # type: ignore[misc, assignment]
    try:
        orch_addr = system.createActor(
            Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
        )
        orchestrator = system.proxy_ask(orch_addr, Orchestrator)
        agent_addr = orchestrator.createActor(agent_class, config=_agent_config())
        time.sleep(0.5)
        assert _InterruptibleReactAgent.captured, "ReactAgent was never constructed"
        yield system, agent_addr, orch_addr
    finally:
        agent_module.ReactAgent = original_react  # type: ignore[misc]
        try:
            system.shutdown(timeout=5)
        except Exception:
            pass


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


# =============================================================================
# FR4 wiring — self._capabilities is the one list both branches receive
# =============================================================================


class _RecordingCapability(AbstractCapability[Any]):
    """A custom capability that records every firing and changes nothing.

    Stands in for what a consumer would actually contribute — an observability
    wrapper, a domain guard — reduced to the part the specs can observe. It
    returns ``request_context`` unchanged, so it cannot perturb the run it is
    added to.
    """

    def __init__(self) -> None:
        self.firings = 0

    async def before_model_request(
        self, ctx: Any, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.firings += 1
        return request_context


class _AgentWithOneExtra(BaseAgent):
    """A subclass contributing exactly one capability of its own.

    ``extra_capabilities`` reads nothing off ``self`` — the wiring specs build
    the agent with ``object.__new__``, so ``config``, ``state`` and
    ``_command_registry`` do not exist when ``_build_react_agent`` calls it.
    """

    extra: ClassVar[_RecordingCapability] = _RecordingCapability()

    def extra_capabilities(self) -> list[AgentCapability[Any]]:
        return [self.extra]


class TestCapabilityWiring:
    def test_bare_base_agent_yields_exactly_the_mailbox_capability(self) -> None:
        """A subclass that overrides nothing gets ``[mailbox]`` and nothing else."""
        with _running_agent():
            capabilities = _InterruptibleReactAgent.captured[-1]["capabilities"]
            assert len(capabilities) == 1
            assert isinstance(capabilities[0], MailboxCapability)

    def test_base_agent_extra_capabilities_defaults_to_empty(self) -> None:
        """The hook is safe on a half-built agent — it reads nothing off self."""
        agent: BaseAgent = object.__new__(BaseAgent)
        assert agent.extra_capabilities() == []

    def test_real_branch_builds_and_stores_the_capability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AKGENTIC_MOCK_SCENARIO", raising=False)
        captured: dict[str, object] = {}

        class _FakeReactAgent:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(agent_module, "ReactAgent", _FakeReactAgent)
        agent: BaseAgent = object.__new__(BaseAgent)

        agent._build_react_agent(ReactAgentConfig(), [], [])

        capabilities = captured["capabilities"]
        assert capabilities is agent._capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) == 1
        assert capabilities[0] is agent._mailbox_capability
        assert isinstance(agent._mailbox_capability, MailboxCapability)

    def test_mock_branch_receives_the_same_capability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKGENTIC_MOCK_SCENARIO", "/tmp/sandpile-research.yaml")
        captured: dict[str, object] = {}

        class _FakeMockReactAgent:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        fake_loadtest = types.ModuleType("akgentic.llm.loadtest")
        fake_loadtest.MockReactAgent = _FakeMockReactAgent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "akgentic.llm.loadtest", fake_loadtest)
        agent: BaseAgent = object.__new__(BaseAgent)

        agent._build_react_agent(ReactAgentConfig(), [], [])

        capabilities = captured["capabilities"]
        assert capabilities is agent._capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) == 1
        assert capabilities[0] is agent._mailbox_capability

    def test_real_branch_puts_the_mailbox_capability_before_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order is mailbox-first: the cancel check precedes any custom work."""
        monkeypatch.delenv("AKGENTIC_MOCK_SCENARIO", raising=False)
        captured: dict[str, object] = {}

        class _FakeReactAgent:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(agent_module, "ReactAgent", _FakeReactAgent)
        extra = _RecordingCapability()
        monkeypatch.setattr(_AgentWithOneExtra, "extra", extra)
        agent: _AgentWithOneExtra = object.__new__(_AgentWithOneExtra)

        agent._build_react_agent(ReactAgentConfig(), [], [])

        assert captured["capabilities"] == [agent._mailbox_capability, extra]

    def test_mock_branch_puts_the_mailbox_capability_before_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same order in the load-test branch — drop-in parity."""
        monkeypatch.setenv("AKGENTIC_MOCK_SCENARIO", "/tmp/sandpile-research.yaml")
        captured: dict[str, object] = {}

        class _FakeMockReactAgent:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        fake_loadtest = types.ModuleType("akgentic.llm.loadtest")
        fake_loadtest.MockReactAgent = _FakeMockReactAgent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "akgentic.llm.loadtest", fake_loadtest)
        extra = _RecordingCapability()
        monkeypatch.setattr(_AgentWithOneExtra, "extra", extra)
        agent: _AgentWithOneExtra = object.__new__(_AgentWithOneExtra)

        agent._build_react_agent(ReactAgentConfig(), [], [])

        assert captured["capabilities"] == [agent._mailbox_capability, extra]

    def test_act_resets_the_run_local_tracking_at_run_start(self) -> None:
        with _running_agent() as (system, agent_addr, _):
            capability = _InterruptibleReactAgent.captured[-1]["capabilities"][0]
            capability._announced_ids.add(uuid.uuid4())

            system.tell(agent_addr, AgentMessage(content="hello", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 1)
            assert _wait_until(lambda: capability._announced_ids == set())


# =============================================================================
# FR5 — the catch site: the run dies, the agent survives
# =============================================================================


class TestCatchSite:
    def test_interrupted_turn_notifies_human_and_routes_nothing(self) -> None:
        """The human is told once, and nothing is delivered.

        Since ``act()`` owns the interruption, ``_route_output`` **is** called —
        with the neutral, empty ``StructuredOutput`` ``act()`` hands back. So
        "routes nothing" is measured as *nothing delivered*, never as the router
        being skipped: an assertion that it was skipped would pin the old design.
        """
        with (
            patch.object(BaseAgent, "notify_human") as notify,
            patch.object(BaseAgent, "_route_output") as route,
            _running_agent(interrupts=1) as (system, agent_addr, _),
        ):
            system.tell(agent_addr, AgentMessage(content="do the thing", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 1)
            assert _wait_until(lambda: notify.call_count >= 1)
            notify.assert_called_once_with("Run interrupted.")

            assert _wait_until(lambda: route.call_count >= 1)
            (routed,), _ = route.call_args
            assert isinstance(routed, StructuredOutput)
            assert routed.messages == []

    def test_agent_survives_and_processes_the_next_queued_message(self) -> None:
        """The actor-death guard — NFR2's mutation target.

        Without the catch, ``RunInterruptedError`` escapes the handler into
        the actor failure path (``Akgent._handle_failure`` — an ErrorMessage
        to the orchestrator; actor death under stock pykka, whose
        ``_handle_failure`` stops the actor). The spec pins both halves of
        surviving *cleanly*: the next queued message is processed normally,
        and the failure path never ran.
        """
        with (
            patch.object(BaseAgent, "notify_human"),
            patch.object(BaseAgent, "_handle_failure") as failure,
            _running_agent(interrupts=1) as (system, agent_addr, _),
        ):
            system.tell(agent_addr, AgentMessage(content="first — interrupted", type="request"))
            system.tell(agent_addr, AgentMessage(content="second — normal", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 2), (
                "the agent did not survive the interruption: the second queued "
                "message was never processed"
            )
            failure.assert_not_called()

    def test_no_user_proxy_branch_logs_instead_of_delivering(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch.object(BaseAgent, "get_team", return_value=[]),
            _running_agent(interrupts=1) as (system, agent_addr, _),
            caplog.at_level(logging.WARNING, logger=AGENT_LOGGER),
        ):
            system.tell(agent_addr, AgentMessage(content="do the thing", type="request"))

            assert _wait_until(
                lambda: any(
                    "not delivered" in record.getMessage() for record in caplog.records
                )
            )

    def test_dequeued_stop_answers_through_command_dispatch(self) -> None:
        with _running_agent() as (system, agent_addr, orch_addr):
            stop = AgentMessage(content="/stop", type="request")
            stop.sender = orch_addr
            system.tell(agent_addr, stop)

            assert _wait_until(
                lambda: any(
                    'The human ran "/stop"' in block
                    for block in _InterruptibleReactAgent.recorded_blocks
                )
            )
            assert any(
                "nothing is running" in block
                for block in _InterruptibleReactAgent.recorded_blocks
            )
            assert _InterruptibleReactAgent.run_calls == 0  # never reached the LLM


# =============================================================================
# A subclass handler that writes NO catch survives identically
# =============================================================================


class TestNoCatchSubclassSurvives:
    """The regression the shipped exemplar failed, pinned at actor level.

    ``CustomAgent.receiveMsg_TriageMessage`` carries no ``try``/``except`` — it
    writes nothing at all for cancellation, because ``act()`` owns it. Driven
    through the real ``act()`` with a ``run_sync`` that raises, the turn must
    still end the designed way: the human told once, nothing delivered, the
    handler returning normally, and ``Akgent._handle_failure`` never entered.

    This is the mutation target for the catch in ``act()``: let the
    interruption propagate out of ``act()`` instead, and this spec goes red
    because the subclass has nothing left to catch it.
    """

    def test_interrupted_subclass_turn_ends_cleanly_with_no_handler_catch(self) -> None:
        with (
            patch.object(BaseAgent, "notify_human") as notify,
            patch.object(BaseAgent, "_handle_failure") as failure,
            patch.object(CustomAgent, "_route_triage") as route,
            _running_agent(interrupts=1, agent_class=CustomAgent) as (system, agent_addr, _),
        ):
            system.tell(agent_addr, TriageMessage(incident="disk full on node 3"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 1)
            assert _wait_until(lambda: notify.call_count >= 1)
            notify.assert_called_once_with("Run interrupted.")

            # The handler ran to its end and routed the neutral output act()
            # returned — an empty triage, so nothing goes out.
            assert _wait_until(lambda: route.call_count >= 1)
            (routed,), _ = route.call_args
            assert isinstance(routed, TriageOutput)
            assert routed.handoffs == []

            failure.assert_not_called()


# =============================================================================
# FR6 — receiveMsg_CancelMessage: an idle cancel is a logged no-op
# =============================================================================


class TestIdleCancel:
    def test_idle_cancel_message_is_a_logged_noop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            _running_agent() as (system, agent_addr, _),
            caplog.at_level(logging.INFO, logger=AGENT_LOGGER),
        ):
            system.tell(agent_addr, CancelMessage(reason="operator changed their mind"))

            assert _wait_until(
                lambda: any(
                    "nothing to cancel" in record.getMessage() for record in caplog.records
                )
            )
            assert _InterruptibleReactAgent.run_calls == 0

    def test_idle_cancel_does_not_poison_the_next_run(self) -> None:
        with _running_agent() as (system, agent_addr, _):
            system.tell(agent_addr, CancelMessage())
            system.tell(agent_addr, AgentMessage(content="carry on", type="request"))

            assert _wait_until(lambda: _InterruptibleReactAgent.run_calls >= 1)
            assert _InterruptibleReactAgent.interrupts_remaining == 0


# =============================================================================
# FR8 — an agent with NO MailboxTool is still interruptible
# =============================================================================


def _make_cardless_agent(pending: list[Any]) -> BaseAgent:
    """A BaseAgent assembled with no tool cards at all.

    ``object.__new__`` means ``on_start`` never runs, so nothing is auto-added —
    this really is an agent whose config carries no ``MailboxTool``. The cancel
    capability observes the agent itself, exactly as ``_build_react_agent``
    wires it in production; ``get_mailbox`` is core's own method, which is why
    it survives the absence of the card. The pending list stands in for the
    actor inbox, which ``object.__new__`` leaves unbuilt.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@CardlessAgent"
    # Explicit: a bare MagicMock(spec=AgentConfig) would answer *any* attribute,
    # so the card-less precondition has to be set, not assumed.
    mock_config.tools = []
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_mailbox = MagicMock(return_value=pending)  # type: ignore[method-assign]
    agent._mailbox_capability = MailboxCapability(observer=agent)

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    return agent


class TestCardlessAgentStillCancels:
    """The requirement that forced the vocabulary out of the card.

    ``MailboxCapability`` is built unconditionally so that an agent
    configured without ``MailboxTool`` is still interruptible — and a
    ``CancelMessage`` from the frontend to such an agent must still kill the
    run. A predicate that shipped with the card could not serve this agent:
    there is no card here to import one from.
    """

    def test_pending_cancel_kills_the_run_of_an_agent_with_no_mailbox_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        agent = _make_cardless_agent([CancelMessage(reason="frontend Esc")])

        # The card-less precondition, asserted rather than relied upon.
        assert agent.config.tools == []
        assert agent._command_registry.has("stop") is False

        react_agent = ReactAgent(
            config=ReactAgentConfig(
                model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
            ),
            deps_type=BaseAgent,
            capabilities=[agent._mailbox_capability],
        )
        agent._react_agent = react_agent  # type: ignore[attr-defined]

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise AssertionError("the model was reached: the cancel never fired")

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                # The cancel still kills the run — the stub proves the model was
                # never reached — but act() absorbs the interruption and hands
                # back the neutral output instead of raising it at the caller.
                output = agent.act("do the long thing", StructuredOutput)
        finally:
            react_agent.close()

        assert isinstance(output, StructuredOutput)
        assert output.messages == []


# =============================================================================
# The extension point works — a subclass's own capability actually fires
# =============================================================================


def _make_extension_point_agent() -> _AgentWithOneExtra:
    """An ``_AgentWithOneExtra`` assembled far enough to survive one real run.

    Same ``object.__new__`` shape as ``_make_cardless_agent``, but the
    capability list is **not** set by hand: ``_build_react_agent`` assembles it,
    which is the whole point — the spec must exercise the framework's own
    assembly, not a list the test wrote.
    """
    agent: _AgentWithOneExtra = object.__new__(_AgentWithOneExtra)

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()

    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@ExtensionPointAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    # No mail: the run must reach the model, so the cancel check must not fire.
    agent.get_mailbox = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    agent.notify_event = MagicMock()  # type: ignore[method-assign]
    return agent


class TestExtraCapabilityFires:
    """The AC that proves the extension point works rather than merely assembling.

    Everything else pins the *list*. This drives a real ``ReactAgent`` — built
    by ``_build_react_agent`` itself, over a ``FunctionModel`` so no network is
    touched — and observes the subclass's own hook having run.
    """

    def test_a_subclass_capability_runs_during_a_real_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.delenv("AKGENTIC_MOCK_SCENARIO", raising=False)
        extra = _RecordingCapability()
        monkeypatch.setattr(_AgentWithOneExtra, "extra", extra)

        agent = _make_extension_point_agent()
        react_agent = agent._build_react_agent(
            ReactAgentConfig(
                model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
            ),
            [],
            [],
        )
        agent._react_agent = react_agent  # type: ignore[attr-defined]

        # The framework assembled it, mailbox first, with the subclass's own second.
        assert agent._capabilities == [agent._mailbox_capability, extra]
        assert extra.firings == 0

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={"messages": []},
                        tool_call_id="out-1",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                output = agent.act("do the thing", StructuredOutput)
        finally:
            react_agent.close()

        assert isinstance(output, StructuredOutput)
        assert extra.firings >= 1
