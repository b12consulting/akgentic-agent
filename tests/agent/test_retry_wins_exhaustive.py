"""Pins pydantic-ai v2's *retry-wins* invariant on this package's own operating shape.

Story 16-2 (Epic 16, ADR-009). This is the only test in `akgentic-agent` that drives a
real pydantic-ai run graph, which is deliberate: the behaviour under test belongs to
pydantic-ai's graph, so a double that fakes the outcome would assert what the author
*believes* v2 does rather than what it actually does.

The exposure
------------
v2 added a retry-wins invariant to ``end_strategy="exhaustive"``. When a function tool
produces a ``RetryPromptPart`` (from ``ModelRetry`` or arg-validation failure) in the
same round as an *already-successful* output tool call, the output is suppressed and the
run stays open for a further model turn. The pinned v1.107.0 baseline
(``pydantic_ai/_agent_graph.py::process_tool_calls``) had no code path from a function
tool's ``RetryPromptPart`` to an already-set ``final_result``; the output won immediately
and the run ended on that turn. v2 moved this into ``pydantic_ai/_tool_execution.py``
(``_ExhaustiveProcessor`` / ``_apply_retry_wins`` / ``_is_retry_wins_trigger``).

Those module and symbol names are navigation aids for a human reader only — nothing here
asserts on them. The test asserts observable behaviour: how many times the model was
called, and which `StructuredOutput` reached `_route_output`.

Why this is `akgentic-agent`'s normal mode, not an edge case
-----------------------------------------------------------
All three ingredients are defaults on every routing turn:

* tools that raise ``ModelRetry`` by design — ``ToolFactory(..., retry_exception=ModelRetry)``
  in ``BaseAgent.on_start``; ``hire_member`` raises it on an unknown role;
* ``end_strategy="exhaustive"`` — the ``RuntimeConfig`` default, forwarded verbatim by
  ``AgentConfig`` into ``ReactAgentConfig`` (asserted below as an explicit precondition);
* a ``StructuredOutput`` output tool — ``act()`` always runs against that static schema.

Cost of the extra turn
----------------------
The retry-wins turn is a **second model call**. It is billed against *both* Epic 13 usage
tiers: the run tier (``RunUsageLimits`` — requests and tokens within one ``run()``) and
the agent-lifetime tier (``AgentUsageLimits``). An agent close to either budget can now
trip a limit on a turn that completed under v1.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from akgentic.agent.agent import BaseAgent, MailboxCancelCapability
from akgentic.agent.config import AgentConfig
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import StructuredOutput
from akgentic.core import ActorAddress
from akgentic.llm import ModelConfig, ReactAgent, ReactAgentConfig
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

# The two turns emit *different* routing decisions. That is what makes the routing
# assertion discriminating on its own: under v1 semantics the first turn's output would
# have won and "@FirstTurn" would be routed.
FIRST_TURN_RECIPIENT = "@FirstTurn"
SECOND_TURN_RECIPIENT = "@Assistant"
SECOND_TURN_MESSAGE = "second-turn output"
SECOND_TURN_TYPE = "instruction"


def _make_mock_address(name: str) -> MagicMock:
    """Return a mock ActorAddress that passes isinstance checks."""
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _make_minimal_agent() -> BaseAgent:
    """Construct a BaseAgent without the Pykka actor system.

    Same shape as `test_agent_coverage._make_minimal_agent`, except `_react_agent` is
    left unset — the caller installs a *real* ReactAgent. The registry answers
    ``has(...) == False`` for everything, so `act()` skips media expansion.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    registry = MagicMock()
    registry.has.return_value = False
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.team_id = uuid.uuid4()  # normally injected by Akgent.__init__ / createActor

    # The context updater normally built in on_start. These specs are not about
    # context delivery, so a stub that composes nothing keeps act() alive.
    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    # Cancel capability normally built in _build_react_agent (Epic 20).
    agent._cancel_capability = MailboxCancelCapability(observer=agent)  # type: ignore[arg-type]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = "@TestAgent"
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    agent.get_team_member = MagicMock(  # type: ignore[method-assign]
        return_value=_make_mock_address(SECOND_TURN_RECIPIENT)
    )
    return agent


def _structured_output_args(
    recipient: str, message: str, message_type: str
) -> dict[str, list[dict[str, str]]]:
    """Build valid `StructuredOutput` tool args. All three Request fields are required."""
    return {
        "messages": [
            {"message_type": message_type, "message": message, "recipient": recipient}
        ]
    }


class TestRetryWinsUnderExhaustiveStrategy:
    """The `ModelRetry` x `end_strategy='exhaustive'` interaction, end-to-end."""

    def test_function_tool_retry_forces_a_second_turn_and_that_output_is_routed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retrying function tool suppresses a concurrent output; turn 2's output routes.

        The provider is `google-gla` **on purpose**. `get_output_type()` returns
        `NativeOutput[T]` for OpenAI/Azure/Anthropic and the raw type otherwise; only the
        raw-type path makes the output a discrete output *tool call*, which is what this
        scenario needs. Under a native provider there is no output `ToolCallPart` to
        suppress, the scenario cannot be constructed at all, and the test would silently
        degrade into one that proves nothing. The API key is never dereferenced —
        `FunctionModel` replaces the model before any run happens.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Precondition, asserted rather than assumed: if this default ever moves, the
        # exposure changes shape and this test must be revisited, not silently pass.
        assert AgentConfig().runtime_cfg.end_strategy == "exhaustive"

        agent = _make_minimal_agent()

        react_config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
        )
        assert react_config.runtime_cfg.end_strategy == "exhaustive"

        react_agent = ReactAgent(config=react_config, deps_type=BaseAgent)
        agent._react_agent = react_agent  # type: ignore[attr-defined]

        flaky_tool_calls = 0

        # A plain tool that raises ModelRetry. Chosen over a full
        # ToolFactory(retry_exception=ModelRetry) wiring because the mechanism under test
        # is pydantic-ai's *graph* reaction to a RetryPromptPart, not the tool's
        # provenance — and TeamTool's hire path would drag in orchestrator plumbing that
        # has nothing to do with the invariant being pinned.
        @react_agent.pydantic_agent.tool_plain
        def flaky_tool(value: str) -> str:
            nonlocal flaky_tool_calls
            flaky_tool_calls += 1
            raise ModelRetry("needs correction")

        model_call_count = 0

        def stub_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_call_count
            model_call_count += 1
            # Never hardcode the output tool name — take it from the run's own schema.
            output_tool_name = info.output_tools[0].name

            if model_call_count == 1:
                # One round, two tool calls: an already-valid output AND a function tool
                # call that will retry.
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=output_tool_name,
                            args=_structured_output_args(
                                FIRST_TURN_RECIPIENT, "first-turn output", "request"
                            ),
                            tool_call_id="out-1",
                        ),
                        ToolCallPart(
                            tool_name="flaky_tool",
                            args={"value": "x"},
                            tool_call_id="fn-1",
                        ),
                    ]
                )

            # Second turn: no function tool call, so nothing can retry-win — this output
            # finalises the run.
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=output_tool_name,
                        args=_structured_output_args(
                            SECOND_TURN_RECIPIENT, SECOND_TURN_MESSAGE, SECOND_TURN_TYPE
                        ),
                        tool_call_id="out-2",
                    )
                ]
            )

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(stub_model)):
                # One routed turn, as receiveMsg_AgentMessage runs it: act() into
                # _route_output(). Driven directly so the run graph — not a message
                # handler's prompt assembly — is the only thing under test.
                agent._route_output(agent.act("route this", StructuredOutput))
        finally:
            # Close on the way out so no event loop or httpx pool leaks into later tests.
            react_agent.close()

        # (0) The retry came from `flaky_tool` actually running and raising `ModelRetry`,
        # not from the tool call failing to resolve. Verified by mutation: pointing turn 1
        # at a name no tool answers to collapses the run back to a single turn, so without
        # this line that regression would surface as a confusing `assert 1 == 2` on the
        # count below — reading as "upstream reverted retry-wins" rather than "the tool
        # stopped resolving".
        assert flaky_tool_calls == 1

        # (1) The extra model turn v2's retry-wins invariant forces. Under v1 the first
        # turn's output would have won immediately and this would be 1.
        assert model_call_count == 2

        # (2) This package still produces one coherent StructuredOutput across that extra
        # turn, and it is the *second* turn's decision that routes. Turn count alone would
        # pin upstream behaviour without proving akgentic-agent survives it.
        agent.get_team_member.assert_called_once_with(SECOND_TURN_RECIPIENT)  # type: ignore[attr-defined]
        agent.send.assert_called_once()  # type: ignore[attr-defined]

        routed_to, routed_msg = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert isinstance(routed_msg, AgentMessage)
        assert routed_msg.content == SECOND_TURN_MESSAGE
        assert routed_msg.type == SECOND_TURN_TYPE
        assert routed_msg.recipient is routed_to
        assert routed_to.name == SECOND_TURN_RECIPIENT
