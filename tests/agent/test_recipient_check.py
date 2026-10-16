"""The recipient check: an unreachable recipient is sent back inside the run.

Unit specs pin the directory, the extractor and every stand-down rule. The
end-to-end specs drive a real pydantic-ai run graph with a scripted model, because
what is under test — a ``ModelRetry`` from ``after_output_validate`` reaching the
model, and the run surviving when no retry is left — belongs to that graph. The
graph keeps two retry budgets, one per output path (an output tool, or native
text), and akgentic-llm's tool-free conclusion allows a single request, so each of
the three gets a run of its own.
"""

import asyncio
import json
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core import ActorAddress, AgentCard
from akgentic.llm import ModelConfig, ReactAgent, ReactAgentConfig, RuntimeConfig
from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig, AgentState
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import Request, StructuredOutput
from akgentic.agent.recipients import (
    RecipientCheckCapability,
    TeamDirectory,
    can_retry,
    structured_output_recipients,
)


def _request(recipient: str) -> Request:
    return Request(message_type="request", message="what is x?", recipient=recipient)


def _directory(hireable: tuple[str, ...] = (), *, can_hire: bool | None = None) -> TeamDirectory:
    return TeamDirectory(
        members=["@Manager", "@Expert", "@Human"],
        hireable_roles=list(hireable),
        can_hire=bool(hireable) if can_hire is None else can_hire,
        self_name="@Manager",
    )


def _ctx(
    deps: object,
    *,
    retry: int = 0,
    max_retries: int = 3,
    requests: int = 1,
    request_limit: int | None = None,
    partial: bool = False,
) -> Any:
    return SimpleNamespace(
        deps=deps,
        retry=retry,
        max_retries=max_retries,
        partial_output=partial,
        usage=SimpleNamespace(requests=requests),
        usage_limits=None if request_limit is None else UsageLimits(request_limit=request_limit),
    )


def _check(ctx: Any, output: object, capability: RecipientCheckCapability | None = None) -> Any:
    check = capability or RecipientCheckCapability()
    return asyncio.run(check.after_output_validate(ctx, output_context=MagicMock(), output=output))


class _Source:
    """A ``DirectorySource`` serving a fixed directory."""

    def __init__(self, directory: TeamDirectory) -> None:
        self.directory = directory

    def team_directory(self) -> TeamDirectory:
        return self.directory


class TestTheDirectory:
    def test_an_at_name_reaches_a_member_only(self) -> None:
        directory = _directory()

        assert directory.reaches("@Expert")
        assert not directory.reaches("@Ghost")

    def test_a_bare_role_reaches_a_hireable_role_only(self) -> None:
        directory = _directory(hireable=("Developer",))

        assert directory.reaches("Developer")
        assert not directory.reaches("Expert")  # a member's name is not its role

    def test_the_correction_names_the_members_but_not_the_asker(self) -> None:
        text = _directory().correction(["Expert"])

        assert text.startswith("Nothing was sent.")
        assert text.endswith("@Expert, @Human.")
        assert "@Manager" not in text

    def test_an_agent_that_cannot_hire_is_told_so_and_offered_no_role(self) -> None:
        text = _directory(can_hire=False).correction(["Expert"])

        assert "'Expert' has no '@', so it was read as a role to hire" in text
        assert "you cannot hire in this team" in text
        assert "hire it" not in text

    def test_a_role_that_is_not_hireable_is_told_apart_and_the_roles_are_offered(self) -> None:
        text = _directory(("Developer", "Tester")).correction(["Expert"])

        assert "'Expert' has no '@'" in text
        assert "it is not a role you can hire" in text
        assert "cannot hire in this team" not in text
        assert "to hire it: Developer, Tester." in text

    def test_an_unknown_at_name_is_named_as_such(self) -> None:
        text = _directory(can_hire=False).correction(["@Ghost"])

        assert "No team member is called '@Ghost'." in text
        assert "has no '@'" not in text

    def test_each_unreachable_recipient_gets_its_own_line_and_one_instruction(self) -> None:
        lines = _directory(can_hire=False).correction(["@Ghost", "Expert"]).splitlines()

        assert lines[0] == "Nothing was sent."
        assert lines[1] == "- No team member is called '@Ghost'."
        assert lines[2].startswith("- 'Expert' has no '@'")
        assert lines[3].startswith("Address every message to a current member")
        assert len(lines) == 4

    def test_an_agent_that_can_hire_with_no_hireable_role_is_offered_none(self) -> None:
        text = _directory(can_hire=True).correction(["Expert"])

        assert "it is not a role you can hire" in text
        assert "hire it:" not in text


class TestTheExtractor:
    def test_a_structured_output_gives_one_recipient_per_request(self) -> None:
        output = StructuredOutput(messages=[_request("@A"), _request("B")])

        assert structured_output_recipients(output) == ["@A", "B"]

    def test_any_other_shape_is_not_read(self) -> None:
        assert structured_output_recipients("plain text") is None


class TestTheCheck:
    def test_reachable_recipients_pass_untouched(self) -> None:
        output = StructuredOutput(messages=[_request("@Expert"), _request("Developer")])

        assert _check(_ctx(_Source(_directory(("Developer",)))), output) is output

    def test_an_unreachable_recipient_is_sent_back(self) -> None:
        output = StructuredOutput(messages=[_request("@Human"), _request("Expert")])

        with pytest.raises(ModelRetry, match="'Expert' has no '@'"):
            _check(_ctx(_Source(_directory())), output)

    def test_an_output_naming_nobody_passes(self) -> None:
        output = StructuredOutput()

        assert _check(_ctx(_Source(_directory())), output) is output

    def test_another_shape_is_not_checked(self) -> None:
        assert _check(_ctx(_Source(_directory())), "plain text") == "plain text"

    def test_deps_without_a_directory_are_not_checked(self) -> None:
        output = StructuredOutput(messages=[_request("Expert")])

        assert _check(_ctx(object()), output) is output

    def test_a_custom_extractor_reads_its_own_shape(self) -> None:
        check = RecipientCheckCapability(recipients=lambda out: [out["to"]])

        with pytest.raises(ModelRetry):
            _check(_ctx(_Source(_directory())), {"to": "Expert"}, check)

    @pytest.mark.parametrize(
        "ctx_kwargs",
        [
            {"retry": 3, "max_retries": 3},
            {"requests": 1, "request_limit": 1},
            {"partial": True},
        ],
        ids=["retries spent", "no request left", "partial output"],
    )
    def test_it_stands_down_when_a_retry_cannot_be_honoured(
        self, ctx_kwargs: dict[str, Any]
    ) -> None:
        output = StructuredOutput(messages=[_request("Expert")])

        assert _check(_ctx(_Source(_directory()), **ctx_kwargs), output) is output


class TestCanRetry:
    def test_a_fresh_run_can_retry(self) -> None:
        assert can_retry(_ctx(None))

    def test_a_request_left_under_the_limit_allows_a_retry(self) -> None:
        assert can_retry(_ctx(None, requests=1, request_limit=2))


def _named(name: str) -> MagicMock:
    address = MagicMock(spec=ActorAddress)
    address.name = name
    return address


def _card(role: str, *, can_be_hired: bool = True) -> AgentCard:
    return AgentCard(
        description=f"The {role}",
        skills=[],
        agent_class="akgentic.agent.BaseAgent",
        config=AgentConfig(name=f"@{role}", role=role),
        can_be_hired=can_be_hired,
    )


def _make_agent(catalog: Sequence[AgentCard] = (), *, hiring: bool = True) -> BaseAgent:
    """A BaseAgent without the actor system, with a team of @Manager, @Expert, @Human.

    ``catalog`` is what the orchestrator's agent catalog returns. With ``hiring``
    the registry has ``hire_member`` and nothing else — the shape of the released
    akgentic-tool; without it the registry has no command at all.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    agent.state = AgentState(backstory="You are a test agent.")  # type: ignore[attr-defined]

    commands = {"hire_member": MagicMock()} if hiring else {}
    registry = MagicMock()
    registry.has.side_effect = lambda name: name in commands
    registry.callable.side_effect = lambda name: commands[name]
    agent._command_registry = registry  # type: ignore[attr-defined]
    agent.discover_catalog = MagicMock(return_value=list(catalog))  # type: ignore[method-assign]
    agent.team_id = uuid.uuid4()

    agent._context_updater = MagicMock()  # type: ignore[attr-defined]
    agent._context_updater.compose_update.return_value = None  # type: ignore[attr-defined]

    config = MagicMock(spec=AgentConfig)
    config.name = "@Manager"
    agent.config = config  # type: ignore[attr-defined]

    team = [_named("@Manager"), _named("@Expert"), _named("@Human"), _named("#Tool")]
    agent.get_team = MagicMock(return_value=team)  # type: ignore[method-assign]
    agent.get_team_member = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda name: next((m for m in team if m.name == name), None)
    )
    agent.send = MagicMock()  # type: ignore[method-assign]
    return agent


class TestTheAgentsDirectory:
    def test_it_lists_the_members_without_tool_actors(self) -> None:
        directory = _make_agent([_card("Developer")]).team_directory()

        assert directory.members == ["@Manager", "@Expert", "@Human"]
        assert directory.hireable_roles == ["Developer"]
        assert directory.can_hire is True
        assert directory.self_name == "@Manager"

    def test_the_hireable_roles_are_the_catalog_roles_that_can_be_hired_sorted(self) -> None:
        catalog = [_card("Tester"), _card("Architect", can_be_hired=False), _card("Developer")]

        agent = _make_agent(catalog)

        directory = agent.team_directory()

        assert directory.hireable_roles == ["Developer", "Tester"]
        agent.discover_catalog.assert_called_once_with()  # type: ignore[attr-defined]

    def test_with_hire_member_alone_a_hireable_role_passes_and_another_is_sent_back(
        self,
    ) -> None:
        """The released akgentic-tool registers ``hire_member`` and no role-listing command."""
        agent = _make_agent([_card("Developer"), _card("Architect", can_be_hired=False)])
        hireable = StructuredOutput(messages=[_request("Developer")])
        not_hireable = StructuredOutput(messages=[_request("Architect")])

        assert _check(_ctx(agent), hireable) is hireable
        with pytest.raises(ModelRetry, match="'Architect' has no '@'.*not a role you can hire"):
            _check(_ctx(agent), not_hireable)

    def test_without_hire_member_the_agent_cannot_hire_and_the_catalog_is_not_read(
        self,
    ) -> None:
        agent = _make_agent([_card("Developer")], hiring=False)

        directory = agent.team_directory()

        assert directory.can_hire is False
        assert directory.hireable_roles == []
        agent.discover_catalog.assert_not_called()  # type: ignore[attr-defined]
        assert "you cannot hire in this team" in directory.correction(["Developer"])


def _output_call(info: AgentInfo, recipient: str, call_id: str) -> ModelResponse:
    args = {"messages": [{"message_type": "request", "message": "hi", "recipient": recipient}]}
    return ModelResponse(
        parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args, tool_call_id=call_id)]
    )


def _react_agent(retries: int = 3) -> ReactAgent:
    """A ReactAgent with only the recipient check mounted.

    ``google-gla`` makes the output a discrete output tool call, which is what the
    scripted model answers with; the key is never used, ``FunctionModel`` replaces
    the model before any request.
    """
    config = ReactAgentConfig(
        model_cfg=ModelConfig(provider="google-gla", model="gemini-2.0-flash"),
        runtime_cfg=RuntimeConfig(retries=retries),
    )
    return ReactAgent(config=config, deps_type=BaseAgent, capabilities=[RecipientCheckCapability()])


class TestInARealRun:
    def test_the_model_is_told_and_its_correction_is_routed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        agent = _make_agent()
        react_agent = _react_agent()
        agent._react_agent = react_agent  # type: ignore[attr-defined]
        retry_prompts: list[str] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            retry_prompts.extend(
                str(part.content)
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            )
            if not retry_prompts:
                return _output_call(info, "Expert", "out-1")
            return _output_call(info, "@Expert", "out-2")

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(model)):
                agent._route_output(agent.act(AgentMessage(content="go"), StructuredOutput))
        finally:
            react_agent.close()

        assert len(retry_prompts) == 1
        assert "'Expert' has no '@'" in retry_prompts[0]
        assert "it is not a role you can hire" in retry_prompts[0]
        assert "@Expert, @Human" in retry_prompts[0]
        routed_to, _ = agent.send.call_args[0]  # type: ignore[attr-defined]
        assert routed_to.name == "@Expert"

    def test_with_no_retry_left_the_run_ends_and_routing_skips(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model that never corrects itself costs that message, never the turn."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        agent = _make_agent()
        react_agent = _react_agent(retries=1)
        agent._react_agent = react_agent  # type: ignore[attr-defined]
        calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return _output_call(info, "Expert", f"out-{calls}")

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(model)):
                output = agent.act(AgentMessage(content="go"), StructuredOutput)
        finally:
            react_agent.close()

        assert calls == 2  # the first answer, and one retry
        assert [r.recipient for r in output.messages] == ["Expert"]

    def test_on_the_native_text_path_no_retry_left_ends_the_run_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The text path counts the run-wide output budget, not an output tool's."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        agent = _make_agent()
        config = ReactAgentConfig(
            model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
            runtime_cfg=RuntimeConfig(retries=1),
        )
        react_agent = ReactAgent(
            config=config, deps_type=BaseAgent, capabilities=[RecipientCheckCapability()]
        )
        agent._react_agent = react_agent  # type: ignore[attr-defined]
        calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            assert not info.output_tools  # native output: the answer is text
            args = {"messages": [{"message_type": "request", "message": "hi", "recipient": "X"}]}
            return ModelResponse(parts=[TextPart(content=json.dumps(args))])

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(model)):
                output = agent.act(AgentMessage(content="go"), StructuredOutput)
        finally:
            react_agent.close()

        assert calls == 2  # the first answer, and one retry
        assert [r.recipient for r in output.messages] == ["X"]

    def test_the_tool_free_conclusion_is_never_sent_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """akgentic-llm's conclusion allows one request; a retry there would fail the turn."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        agent = _make_agent()
        react_agent = _react_agent()
        calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return _output_call(info, "Expert", f"out-{calls}")

        try:
            with react_agent.pydantic_agent.override(model=FunctionModel(model)):
                output = react_agent.conclude_without_tools_sync(
                    "Conclude now.", deps=agent, output_type=StructuredOutput
                )
        finally:
            react_agent.close()

        assert calls == 1
        assert [r.recipient for r in output.messages] == ["Expert"]
