"""Parity of ``akgentic-llm``'s ``LimitRecoveryCapability`` against ``@guard_usage_limits``.

Story 24-1 exists so that story 24-2 deletes the decorator on **evidence** rather than on the
assumption that the replacement is equivalent. Nothing here deletes anything:
``@guard_usage_limits`` stays mounted on both handlers throughout, and every spec below describes
the **capability's** behaviour with the decorator still in place. That is what makes them valid
before and after the removal — with the capability live, the breach no longer escapes
``ReactAgent.run()``, so the decorator's ``except RunUsageLimitError`` never fires and the
decorator is inert on the recovered path.

**The breach is real.** The subject of this file is the seam between two packages, so a mocked
breach would only test the mock. Every spec drives a real ``ReactAgent`` — a real capability
stack, a real ``UsageLimitExceeded`` from pydantic-ai, a real recovery — with
``pydantic_ai.models.function.FunctionModel`` standing in for the provider through the public
``react_agent.pydantic_agent.override(model=...)`` context manager. No network, no API key in
use, no ``MockReactAgent`` (which does not enforce the agent-lifetime budget, so anything
counting runs against it passes while the real class fails).

**Why ``google-gla``.** ``get_output_type`` (``akgentic/llm/providers.py``) wraps a structured
output in ``NativeOutput`` for OpenAI/Anthropic and leaves the type raw for everyone else. On the
raw path pydantic-ai asks for the output through an **output tool**, which a ``FunctionModel``
stub can emit honestly as a ``ToolCallPart`` naming ``info.output_tools[0]``. The native path
would require the stub to fabricate a provider-specific structured response, which is a fiction
about the provider rather than a fact about the recovery. ``override(tools=[], toolsets=[])``
replaces only the *function* tools, so the conclusion still has its output tool — which is why a
structured conclusion is expressible at all.

**Three behaviours are lapsing**, and each has its own spec here pinning what actually happens
now rather than what ought to. They are what story 26-2's review found
``try_conclude_without_tools`` doing beyond catching the error, and they are tracked as
``b12consulting/akgentic-agent#119``:
naming the requester in the conclusion prompt, refusing a sender-less message, and escalating when
routing delivers nothing. The third fails **silently**. See the epic's ``## Deferred findings``.

**Two of the three have since been accepted as losses; the third has not.** The silent one is
being restored rather than dropped — ``akgentic-llm`` grows an ``is_conclusion_usable`` seam that
``akgentic-agent`` overrides to report structured emptiness, and the human escalation comes back
through it. So story 24-2 **inverts** that spec instead of deleting it, and cannot land until the
llm half of that seam has shipped. Detail in the epic's ``## Deferred findings``.

Assertions are **outcomes** — what the requester received, whether a human was notified, whether
``WarningError`` escaped — never merely that a mock was called, and never on the decorator's own
call counts, which would not survive 24-2. The tiers are told apart by exception class; no spec
reads message text to decide which tier fired.
"""

import uuid
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import ActorAddress
from akgentic.core.agent import WarningError
from akgentic.llm import AgentUsageLimits, ModelConfig, ReactAgent, ReactAgentConfig, RunUsageLimits
from akgentic.llm.capabilities import DEFAULT_CONCLUSION_REASON
from akgentic.tool.errors import CommandNotRecognized
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.agent.custom_agent import CustomAgent, TriageMessage, TriageOutput
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput

REQUESTER = "@Human"
THIRD_PARTY = "@Someone-Else"

# What the conclusion is asked to produce, as the output tool's arguments. Shaped like the
# schemas the two decorated handlers reason against, so a stub never has to know more than the
# recipient and the words.
ANSWER = "Partial answer: two of three sources checked."


# =============================================================================
# MODEL STUBS — a REAL breach, and a REAL conclusion, without a provider
# =============================================================================


def weather_lookup(city: str) -> str:
    """Look up the weather for a city.

    Args:
        city: The city to look up.

    Returns:
        A canned forecast string.
    """
    return f"sunny in {city}"


def _structured_output_args(recipient: str = REQUESTER, message: str = ANSWER) -> dict[str, Any]:
    """Output-tool arguments for a ``StructuredOutput`` carrying one deliverable Request."""
    return {
        "messages": [
            {"recipient": recipient, "message": message, "message_type": "response"},
        ]
    }


def _triage_output_args(recipient: str = REQUESTER, task: str = ANSWER) -> dict[str, Any]:
    """Output-tool arguments for a ``TriageOutput`` carrying one deliverable handoff."""
    return {
        "severity": "high",
        "summary": "escalating what was gathered before the budget ran out",
        "handoffs": [{"recipient": recipient, "task": task}],
    }


def _answer(info: AgentInfo, payload: dict[str, Any]) -> ModelResponse:
    """Return ``payload`` through the output tool a non-native provider uses."""
    return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=payload)])


def _spend_the_turn() -> ModelResponse:
    """Call the one registered tool, which is what walks the turn into its breach.

    Against ``RunUsageLimits(run_request_limit=1)`` the first request is allowed and spends the
    budget; the tool runs; the second request is refused by pydantic-ai before it is made. The
    breach therefore lands on a turn that has genuinely done some work, which is the situation
    a conclusion exists for.
    """
    return ModelResponse(parts=[ToolCallPart(tool_name="weather_lookup", args={"city": "Paris"})])


def _breaching_then_concluding_model(
    payload: dict[str, Any], prompts: list[str] | None = None
) -> FunctionModel:
    """Breach the outer run; answer plainly once the tools are gone.

    ``AgentInfo.function_tools`` is empty exactly when ``conclude_without_tools`` has overridden
    the toolset away, which is the honest condition to branch on: a model that always calls a
    tool would breach the conclusion too, because the tool it names no longer exists.

    Args:
        payload: Output-tool arguments the conclusion answers with.
        prompts: When given, each conclusion's user prompt is appended to it.
    """

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.function_tools:
            return _spend_the_turn()
        if prompts is not None:
            prompts.append(_last_user_prompt(messages))
        return _answer(info, payload)

    return FunctionModel(stub)


def _breaching_then_failing_model(error: Exception) -> FunctionModel:
    """Breach the outer run, then make every conclusion raise ``error``."""

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.function_tools:
            return _spend_the_turn()
        raise error

    return FunctionModel(stub)


def _breaching_once_then_answering_model(payload: dict[str, Any]) -> FunctionModel:
    """Breach the FIRST turn only; answer directly on every later one.

    The discriminator for "a rescued turn costs two units of the lifetime run budget": a second
    turn that needs exactly one run either fits in what is left or does not, and which of those
    happens is the whole claim.
    """
    outer_runs = 0

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal outer_runs
        if not info.function_tools:
            return _answer(info, payload)
        outer_runs += 1
        if outer_runs == 1:
            return _spend_the_turn()
        return _answer(info, payload)

    return FunctionModel(stub)


def _answering_model(payload: dict[str, Any], requests: list[str] | None = None) -> FunctionModel:
    """Answer every request directly — a turn that never breaches, costing one run."""

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if requests is not None:
            requests.append("request")
        return _answer(info, payload)

    return FunctionModel(stub)


def _last_user_prompt(messages: list[ModelMessage]) -> str:
    """The most recent user prompt in a history — the conclusion's ``reason`` when it is one."""
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in reversed(message.parts):
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


# =============================================================================
# HARNESS — a real ReactAgent, attached to a hand-built agent
# =============================================================================


def _make_address(name: str) -> MagicMock:
    """Return a mock ActorAddress that passes isinstance checks."""
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _make_registry() -> MagicMock:
    """A command registry that recognises nothing, so every message reaches the LLM."""
    registry = MagicMock()
    registry.has.side_effect = lambda name: False

    def _callable(name: str) -> Callable[..., Any]:
        raise CommandNotRecognized(name)

    registry.callable.side_effect = _callable
    return registry


def _react_config(agent_request_limit: int | None = None) -> ReactAgentConfig:
    """A tight run-tier budget on a provider whose structured output is an output tool.

    ``run_request_limit=1`` makes a breach cheap and real: one model request is paid for, the
    tool runs, and the second request is refused. The agent tier is left unset unless a spec is
    about it.
    """
    return ReactAgentConfig(
        model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
        run_usage_limits=RunUsageLimits(run_request_limit=1),
        agent_usage_limits=AgentUsageLimits(agent_request_limit=agent_request_limit),
    )


@pytest.fixture
def build_agent(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Any]]:
    """Build agents around a REAL ``ReactAgent``, and close every loop afterwards.

    ``get_team_member`` is keyed **by name**: it resolves the requester and nobody else. A
    blanket ``return_value`` would hand the requester's address back for whatever name the model
    happened to choose, so "the requester received the conclusion" would hold even for an answer
    addressed to a third agent — the exact failure this suite exists to catch. ``hire_member`` is
    stubbed to fail loudly, because a conclusion must never hire anyone.

    ``ReactAgent`` owns an event loop per instance, so each one built here is closed on teardown.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-never-used")
    built: list[ReactAgent] = []

    def _build(
        *,
        agent_cls: type[BaseAgent] = BaseAgent,
        agent_request_limit: int | None = None,
    ) -> tuple[BaseAgent, MagicMock]:
        react = ReactAgent(
            config=_react_config(agent_request_limit),
            deps_type=BaseAgent,
            tools=[weather_lookup],
        )
        built.append(react)

        agent: BaseAgent = object.__new__(agent_cls)
        agent._react_agent = react  # type: ignore[attr-defined]
        agent._command_registry = _make_registry()  # type: ignore[attr-defined]
        agent.team_id = uuid.uuid4()

        # The context updater is normally built in on_start. These specs are not about context
        # delivery, so a stub that composes nothing keeps act() alive.
        agent._context_updater = MagicMock()  # type: ignore[attr-defined]
        agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

        mock_config = MagicMock(spec=AgentConfig)
        mock_config.name = "@TestAgent"
        agent.config = mock_config  # type: ignore[attr-defined]

        requester = _make_address(REQUESTER)
        agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
        agent.send = MagicMock()  # type: ignore[method-assign]
        agent.notify_human = MagicMock()  # type: ignore[method-assign]
        agent.get_team_member = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda name: requester if name == REQUESTER else None
        )
        agent.hire_member = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("a tool-free conclusion must not hire anyone")
        )
        return agent, requester

    yield _build

    for react in built:
        react.close()


def _message(sender: str | None = REQUESTER) -> AgentMessage:
    """An incoming request, optionally without a sender (``Message.sender`` is optional)."""
    message = AgentMessage(content="what is the status?", type="request")
    if sender is not None:
        message.sender = _make_address(sender)
    return message


def _no_decorator_conclusion(agent: BaseAgent) -> MagicMock:
    """Spy on the DECORATOR's entry point, so a spec can prove it was never taken.

    ``try_conclude_without_tools`` reaches the LLM through
    ``ReactAgent.conclude_without_tools_sync``; the capability's own recovery drives the async
    ``conclude_without_tools`` from inside ``run()`` and never touches the sync bridge. The two
    paths are therefore distinguishable at this one method.
    """
    spy = MagicMock(side_effect=AssertionError("the decorator's conclusion path must not run"))
    agent._react_agent.conclude_without_tools_sync = spy  # type: ignore[attr-defined, method-assign]
    return spy


# =============================================================================
# FR1 — the requester gets the concluded answer, routed the ordinary way
# =============================================================================


class TestTheRequesterGetsTheConcludedAnswer:
    """A real run-tier breach ends with the requester holding a real answer."""

    @patch("akgentic.agent.agent.sleep")
    def test_the_requester_receives_the_conclusion_at_their_own_address(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC2: the concluded answer, at the address the requester was resolved to BY NAME.

        "Something was delivered" is not the claim — an empty answer delivered to whoever the
        model happened to name would satisfy that, and is precisely the silent failure pinned
        below in ``test_a_conclusion_that_routes_nothing_is_silent``.
        """
        agent, requester = build_agent()

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model(_structured_output_args())
        ):
            agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        agent.get_team_member.assert_called_once_with(REQUESTER)  # type: ignore[attr-defined]
        agent.send.assert_called_once()  # type: ignore[attr-defined]
        target, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert target is requester
        assert isinstance(sent, AgentMessage)
        assert sent.content == ANSWER
        assert sent.type == "response"

    @patch("akgentic.agent.agent.sleep")
    def test_an_answer_addressed_elsewhere_never_reaches_the_requester(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC2: the by-name lookup is load-bearing, proven by making the model miss.

        The mutation guard for the spec above: with ``get_team_member`` keyed by name, a
        conclusion addressed to somebody else resolves to ``None`` and costs a delivery. A
        blanket ``return_value`` would hand back the requester's address here and the previous
        spec would pass for an answer the requester never asked for.
        """
        agent, _ = build_agent()

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model(_structured_output_args(THIRD_PARTY))
        ):
            agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        agent.get_team_member.assert_called_once_with(THIRD_PARTY)  # type: ignore[attr-defined]
        agent.send.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_the_recovery_came_from_the_capability_not_the_decorator(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC3: the answer arrives through the handler's ORDINARY routing path.

        Nothing raises out of ``act()`` any more, so ``_route_output`` is reached from the
        handler body — the normal turn's path — and not from the decorator's ``route=``
        callback. Two negatives say so: the decorator's entry point
        (``conclude_without_tools_sync``) is never called, and no human is notified. This is the
        spec that will still be true, unchanged, once story 24-2 removes the decorator.
        """
        agent, _ = build_agent()
        decorator_path = _no_decorator_conclusion(agent)

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model(_structured_output_args())
        ):
            agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        decorator_path.assert_not_called()
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
        agent.send.assert_called_once()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_a_rescued_turn_costs_two_units_of_the_lifetime_run_budget(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC2: the outer run AND the conclusion each pay the agent-tier pre-flight.

        Inherent to the sibling-run design: the conclusion is a run like any other. Asserted as
        an outcome rather than by reading a counter — with a lifetime budget of two, a rescued
        turn spends the lot, so the very next turn is refused before it starts even though it
        would need only one run. Had the rescue cost one, that second turn would have answered.

        ``MockReactAgent`` does not enforce this budget at all, which is why the spec needs the
        real class.
        """
        agent, _ = build_agent(agent_request_limit=2)

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_once_then_answering_model(_structured_output_args())
        ):
            agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))
            agent.send.assert_called_once()  # type: ignore[attr-defined]

            with pytest.raises(WarningError, match="LLM usage limit exceeded"):
                agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        # Still one — the second turn never produced anything to deliver.
        agent.send.assert_called_once()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]


# =============================================================================
# FR3 — the agent tier is terminal, and stays terminal from inside a conclusion
# =============================================================================


class TestTheAgentTierStaysTerminal:
    """A spent lifetime budget is never answered with a conclusion, wherever it is noticed."""

    @patch("akgentic.agent.agent.sleep")
    def test_a_lifetime_refusal_escalates_without_attempting_a_conclusion(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC4: the pre-flight refusal is terminal — no conclusion, human paged, WarningError.

        ``AgentUsageLimitError`` is this package's own class and not a ``UsageLimitExceeded``,
        so ``LimitRecoveryCapability.on_run_error`` passes it straight through and the seam is
        never consulted. The model is not reached at all on the refused turn, which is the
        strongest available form of "no conclusion was attempted".
        """
        agent, _ = build_agent(agent_request_limit=1)
        decorator_path = _no_decorator_conclusion(agent)
        requests: list[str] = []

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_answering_model(_structured_output_args(), requests)
        ):
            # Turn one spends the single unit of lifetime budget and answers normally.
            agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))
            assert requests == ["request"]

            with pytest.raises(WarningError, match="LLM usage limit exceeded"):
                agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        assert requests == ["request"], "the refused turn reached the model"
        decorator_path.assert_not_called()
        agent.send.assert_called_once()  # type: ignore[attr-defined]
        notice = agent.notify_human.call_args[0][0]  # type: ignore[attr-defined]
        assert "@TestAgent" in notice
        assert "agent_request_limit" in notice

    @patch("akgentic.agent.agent.sleep")
    def test_an_agent_tier_refusal_inside_a_conclusion_never_becomes_an_answer(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC4: a conclusion refused by the lifetime pre-flight ends the turn escalated.

        With a lifetime budget of one, the outer run spends it and then breaches at the run
        tier; the conclusion the seam asks for is refused pre-flight. That refusal must not
        surface as the caller's error and must not produce an answer — the turn ends reporting
        the ORIGINAL run-tier breach, with nobody having received anything.
        """
        agent, _ = build_agent(agent_request_limit=1)

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model(_structured_output_args())
        ):
            with pytest.raises(WarningError) as excinfo:
                agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        assert "request_limit" in str(excinfo.value), "the original run-tier breach"
        assert "agent_request_limit" not in str(excinfo.value), "the secondary refusal leaked"
        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_called_once()  # type: ignore[attr-defined]


# =============================================================================
# FR2 — the original breach is what surfaces, never the secondary failure
# =============================================================================


class TestTheOriginalBreachIsWhatSurfaces:
    """When the conclusion fails, the human hears about the breach that started the turn."""

    @patch("akgentic.agent.agent.sleep")
    def test_a_failed_conclusion_reports_the_breach_and_not_its_own_error(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC5: both sides asserted — the original present, the secondary absent.

        A secondary failure replacing the breach would turn "this turn ran out of budget" into
        an unrelated error message, and the human would be told to investigate the wrong thing.
        Asserted on the ``WarningError`` and on the ``notify_human`` argument alike, because
        those are the two places a person actually reads it.
        """
        agent, _ = build_agent()

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_failing_model(RuntimeError("secondary-conclusion-failure"))
        ):
            with pytest.raises(WarningError) as excinfo:
                agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        assert "request_limit" in str(excinfo.value)
        assert "secondary-conclusion-failure" not in str(excinfo.value)

        notice = agent.notify_human.call_args[0][0]  # type: ignore[attr-defined]
        assert "request_limit" in notice
        assert "secondary-conclusion-failure" not in notice
        agent.send.assert_not_called()  # type: ignore[attr-defined]


# =============================================================================
# THE THREE LAPSING BEHAVIOURS — one spec each, pinning what actually happens
# =============================================================================


class TestTheLapsingBehaviours:
    """What ``try_conclude_without_tools`` did beyond catching the error, and what is left.

    Each of the three is defensible to drop, but dropping one by accident is not the same as
    deciding to. These specs pin the live behaviour so 24-2 removes the decorator against a
    written record rather than an assumption. Tracked as ``b12consulting/akgentic-agent#119``.
    """

    @patch("akgentic.agent.agent.sleep")
    def test_the_conclusion_prompt_no_longer_names_the_requester(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC6a — DIVERGENCE. The decorator named the requester; the capability names nobody.

        ``try_conclude_without_tools`` wrote "Answer {requester} now …" because the returned
        output is LLM-authored: the model chooses each recipient, so an answer that does not
        name them can be routed perfectly and still leave the requester with nothing.
        ``DEFAULT_CONCLUSION_REASON`` drops that clause deliberately — ``akgentic-llm`` has no
        notion of a requester, and a placeholder there would be echoed back as a recipient,
        where an unprefixed name is a role to HIRE. A deployment that wants the requester named
        returns its own ``ConclusionDecision(reason=…)`` from the seam.

        Compared against the constant rather than against re-typed prompt text, so a reword of
        the prompt does not turn this red for no behavioural reason.
        """
        agent, _ = build_agent()
        prompts: list[str] = []

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model(_structured_output_args(), prompts)
        ):
            agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        assert prompts == [DEFAULT_CONCLUSION_REASON]
        assert REQUESTER not in prompts[0]

    @patch("akgentic.agent.agent.sleep")
    def test_a_sender_less_message_is_concluded_to_anyway(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC6b — DIVERGENCE. The decorator refused; the capability concludes regardless.

        ``Message.sender`` is legitimately optional. The decorator skipped the attempt entirely
        for a sender-less turn and escalated instead, on the reasoning that a reason with nobody
        to name would get a placeholder echoed back as a recipient. The capability has no notion
        of a requester at all, so nothing distinguishes this turn: it concludes, the model picks
        the recipient unaided, and the answer is routed.

        The turn completes normally — no ``WarningError``, no human paged.
        """
        agent, requester = build_agent()
        senderless = _message(sender=None)
        assert senderless.sender is None

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model(_structured_output_args())
        ):
            agent.receiveMsg_AgentMessage(senderless, _make_address(REQUESTER))

        agent.send.assert_called_once()  # type: ignore[attr-defined]
        target, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert target is requester
        assert sent.content == ANSWER
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
        agent.hire_member.assert_not_called()  # type: ignore[attr-defined]

    @patch("akgentic.agent.agent.sleep")
    def test_a_conclusion_that_routes_nothing_is_silent(
        self, mock_sleep: MagicMock, build_agent: Callable[..., Any]
    ) -> None:
        """AC6c — DIVERGENCE, and the silent one. Nobody is told the requester got nothing.

        The decorator treated a conclusion that delivered no message as a FAILURE and escalated
        to a human, because the requester received nothing — exactly as if the call had raised.
        "Nothing usable" is deliberately narrow in ``akgentic-llm``: ``None``, or a ``str`` that
        is empty or whitespace-only. A ``StructuredOutput`` with an empty ``messages`` list is a
        *successful* output there, so it is returned from ``run()``, routed by the handler body,
        and delivers nothing.

        The full silence is the assertion: nothing sent, nothing raised, nobody notified.

        **This spec is 24-2's to invert, not to delete.** The silence was the finding this story
        was written to surface, and it has since been ruled a defect rather than an accepted
        loss: ``akgentic-llm`` grows an ``is_conclusion_usable`` seam, this package overrides it
        to report that a ``StructuredOutput`` carrying no ``Request`` reached nobody, and the
        original breach is raised so the existing escalation pages a human exactly as the
        decorator did. ``escalate_usage_limit`` therefore survives 24-2. When that lands, the
        three assertions below flip to: nothing sent, ``WarningError`` raised, human notified.
        """
        agent, _ = build_agent()

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model({"messages": []})
        ):
            agent.receiveMsg_AgentMessage(_message(), _make_address(REQUESTER))

        agent.send.assert_not_called()  # type: ignore[attr-defined]
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
        agent.get_team_member.assert_not_called()  # type: ignore[attr-defined]


# =============================================================================
# FR1 — the conclusion arrives in the CALLER's schema, on the second handler
# =============================================================================


class TestTheConclusionKeepsTheCallersSchema:
    """``_conclude_after_breach`` threads ``output_type`` verbatim, so the schema stays theirs."""

    def test_a_breached_triage_turn_recovers_as_a_triage_output(
        self, build_agent: Callable[..., Any]
    ) -> None:
        """AC11: the cheapest strong proof, on the handler most likely to break silently.

        ``CustomAgent``'s own docstring credits the decorator's ``output_type=`` argument for
        concluding "in THIS agent's schema with nothing overridden". That property now comes
        from the capability instead: the conclusion is a sibling run carrying the breached
        call's own ``output_type``, so ``act(message, TriageOutput)`` returns a ``TriageOutput``
        and the handler body routes it through ``_route_triage`` like any other turn.

        ``receiveMsg_TriageMessage`` has no processing-delay sleep, so nothing is patched here.
        """
        agent, requester = build_agent(agent_cls=CustomAgent)
        decorator_path = _no_decorator_conclusion(agent)

        routed: list[Any] = []
        deliver = agent._route_triage  # type: ignore[attr-defined]

        def _recording_route(output: TriageOutput) -> bool:
            routed.append(output)
            return deliver(output)

        agent._route_triage = _recording_route  # type: ignore[attr-defined, method-assign]

        message = TriageMessage(incident="the pager is on fire", reported_by="ops")
        message.sender = _make_address(REQUESTER)

        with agent._react_agent.pydantic_agent.override(  # type: ignore[attr-defined]
            model=_breaching_then_concluding_model(_triage_output_args())
        ):
            agent.receiveMsg_TriageMessage(message, _make_address(REQUESTER))

        # The recovered output is the CALLER's type, not the base class's StructuredOutput.
        assert len(routed) == 1
        assert isinstance(routed[0], TriageOutput)
        assert not isinstance(routed[0], StructuredOutput)
        assert routed[0].severity == "high"

        # …and it was delivered by this agent's own router, to a member resolved by name.
        agent.get_team_member.assert_called_once_with(REQUESTER)  # type: ignore[attr-defined]
        agent.send.assert_called_once()  # type: ignore[attr-defined]
        target, sent = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert target is requester
        assert sent.content == ANSWER
        assert sent.type == "request"

        decorator_path.assert_not_called()
        agent.notify_human.assert_not_called()  # type: ignore[attr-defined]
