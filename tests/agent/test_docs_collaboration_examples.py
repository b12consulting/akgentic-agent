"""Executable copies of every code sample in README.md and docs/agent-collaboration.md.

Both documents spent several releases describing a `cmd_*` programmatic API that no
longer exists anywhere in `src/`, and shipping `Request` snippets that raise
`ValidationError` when pasted. Nothing in the suite could catch either, because the
claims lived only in prose.

Each test below is a documented snippet transcribed as runnable Python, plus the
assertions that pin what the surrounding prose claims — so a future rename breaks the
example *here* rather than in a reader's editor.

Deliberately NOT tests that read or grep a markdown file: that would check
documentation instead of behaviour. These run the code the documentation shows.

`ReactAgent` is stubbed throughout — no live model, no network.
"""

import time
import uuid
from typing import Any, Callable, cast
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import (
    ActorAddress,
    ActorSystem,
    AgentCard,
    BaseConfig,
    EventSubscriber,
    Orchestrator,
)
from akgentic.core.messages import Message
from akgentic.core.messages.orchestrator import EventMessage, SentMessage
from akgentic.llm import (
    AgentUsageLimits,
    CompactionConfig,
    ModelConfig,
    PromptTemplate,
    RuntimeConfig,
    RunUsageLimits,
    ToolCallEvent,
)
from akgentic.tool.core import ToolCard, ToolFactory
from akgentic.tool.errors import CommandNotRecognized
from akgentic.tool.event import CommandsAnnouncedEvent
from akgentic.tool.planning import GetPlanning, PlanningTool, UpdatePlanning
from akgentic.tool.team import TeamTool
from pydantic import BaseModel, ValidationError
from pydantic_ai import AgentCapability, ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

import akgentic.agent.agent as agent_module
from akgentic.agent.agent import BaseAgent, MailboxCapability
from akgentic.agent.config import AgentConfig
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import REPLY_PROTOCOLS, Request, StructuredOutput

# =============================================================================
# HELPERS
# =============================================================================


def _make_address(name: str = "@Human") -> MagicMock:
    """Return a mock ActorAddress that passes isinstance checks."""
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _stub(agent: object, attribute: str, **kwargs: Any) -> MagicMock:
    """Replace a *method* of *agent* with a MagicMock and hand it back.

    Going through ``setattr`` keeps the call sites free of ``method-assign``
    suppressions and gives each test a properly typed handle to assert on.

    The class-level check is not defensive noise — it closes the way four tests in
    this file went vacuous. ``setattr`` happily invents a name that does not exist,
    so when a method was renamed out of ``BaseAgent`` every ``_stub(agent, "<old
    name>")`` kept working and every ``assert_not_called()`` on it passed
    trivially: green tests asserting nothing about the code that actually runs.
    Renaming a method now breaks its tests loudly, at the stub.

    Deliberately checks the **class**, not the instance: a name only ``on_start``
    installs would pass an instance check for the wrong reason once some earlier
    line had set it. Runtime attributes are assigned directly instead.
    """
    if not hasattr(type(agent), attribute):
        raise AttributeError(
            f"{type(agent).__name__} defines no {attribute!r} to stub — stubbing a "
            "name the class does not have makes every assertion on it vacuous"
        )
    stub = MagicMock(**kwargs)
    setattr(agent, attribute, stub)
    return stub


def _react_agent_of(agent: BaseAgent) -> MagicMock:
    """Return the stubbed ReactAgent that ``_make_agent`` installed."""
    return cast(MagicMock, agent._react_agent)


def _make_agent(
    name: str = "@Manager", commands: dict[str, Callable[..., Any]] | None = None
) -> BaseAgent:
    """Build a BaseAgent outside the actor system, with a real CommandRegistry.

    ``commands`` are registered as agent-owned extra callables, so the registry is
    the production class rather than a mock — dispatch, coercion and the
    ``CommandNotRecognized`` fallback are all exercised for real.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    # Installed, not stubbed: on_start creates this one, so there is no class-level
    # name for _stub to check against.
    agent._react_agent = MagicMock()  # type: ignore[assignment]
    agent._current_message = None
    agent.team_id = uuid.uuid4()

    # The context updater normally built in on_start. These specs are not about
    # context delivery, so a stub that composes nothing keeps act() alive.
    agent._context_updater = MagicMock()
    agent._context_updater.compose_update.return_value = None

    # Mailbox capability normally built in _build_react_agent (Epic 20).
    agent._mailbox_capability = MailboxCapability(observer=agent)  # type: ignore[arg-type]

    config = MagicMock(spec=AgentConfig)
    config.name = name
    setattr(agent, "config", config)

    agent._command_registry = ToolFactory(tool_cards=[]).get_command_registry(
        extra_commands=list((commands or {}).values())
    )

    _stub(agent, "send")
    _stub(agent, "get_team", return_value=[])
    _stub(agent, "get_team_member", return_value=None)
    return agent


def _stub_act(agent: BaseAgent) -> MagicMock:
    """Stub ``act`` — the handler's only LLM call — returning an empty output.

    ``receiveMsg_AgentMessage`` calls ``act`` and hands the result straight to
    ``_route_output``; there is no method between the two to patch, so a test
    about what the handler *asks for* stubs the LLM call itself.
    """
    return _stub(agent, "act", return_value=StructuredOutput())


def _run_turn(agent: BaseAgent, prompt: str = "prompt") -> bool:
    """Drive one routed turn the way ``receiveMsg_AgentMessage`` does."""
    return agent._route_output(agent.act(prompt, StructuredOutput))


# =============================================================================
# The data model: README `StructuredOutput and Request` + collaboration `Data Model`
# =============================================================================


class TestRequestSnippet:
    """`Request` as both documents now show it — message_type first, and required."""

    def test_documented_request_constructs(self) -> None:
        request = Request(
            message_type="request",
            message="Estimate feature X",
            recipient="@Developer",
        )

        assert request.message_type == "request"
        assert request.message == "Estimate feature X"
        assert request.recipient == "@Developer"

    def test_message_type_is_required(self) -> None:
        """The pre-fix snippet omitted it; a reader pasting that got this error."""
        with pytest.raises(ValidationError) as excinfo:
            Request(message="Estimate feature X", recipient="@Developer")  # type: ignore[call-arg]

        assert "message_type" in str(excinfo.value)

    def test_every_field_is_required(self) -> None:
        with pytest.raises(ValidationError):
            Request(message_type="request", message="x")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Request(message_type="request", recipient="@Dev")  # type: ignore[call-arg]

    def test_recipient_is_an_unconstrained_plain_string(self) -> None:
        """Documented as validated at ROUTING time, not in the schema."""
        nonsense = Request(
            message_type="request", message="hi", recipient="@NobodyHasThisName"
        )
        assert nonsense.recipient == "@NobodyHasThisName"

        schema = Request.model_json_schema()["properties"]["recipient"]
        assert schema["type"] == "string"
        assert "enum" not in schema

    def test_the_five_documented_intents_are_exactly_the_accepted_set(self) -> None:
        documented = {
            "request",
            "response",
            "notification",
            "instruction",
            "acknowledgment",
        }
        for intent in documented:
            assert (
                Request(message_type=intent, message="m", recipient="@X").message_type  # type: ignore[arg-type]
                == intent
            )
        with pytest.raises(ValidationError):
            Request(message_type="answer", message="m", recipient="@X")  # type: ignore[arg-type]


class TestAgentMessageSnippet:
    """`AgentMessage` as both documents now show it."""

    def test_type_defaults_to_request_and_content_is_required(self) -> None:
        message = AgentMessage(content="Plan the next sprint.")
        assert message.type == "request"

        with pytest.raises(ValidationError):
            AgentMessage()  # type: ignore[call-arg]

    def test_declares_only_type_and_content_of_its_own(self) -> None:
        """The API Reference claims exactly two own fields; the rest are inherited."""
        assert set(AgentMessage.__annotations__) == {"type", "content"}

    def test_inherits_the_documented_message_fields(self) -> None:
        for inherited in (
            "id",
            "parent_id",
            "team_id",
            "timestamp",
            "sender",
            "recipient",
            "display_type",
        ):
            assert inherited in AgentMessage.model_fields

    def test_carries_a_recipient_field_it_can_hold_an_address_in(self) -> None:
        """The old text claimed it did not; _route_output sets it on every send."""
        address = _make_address("@Developer456")
        message = AgentMessage(content="x", recipient=address)
        assert message.recipient is address


class TestStructuredOutputSnippet:
    """`StructuredOutput` — Example 3's empty list, and the populated case."""

    def test_defaults_to_an_empty_message_list(self) -> None:
        assert StructuredOutput().messages == []
        assert StructuredOutput(messages=[]).messages == []

    def test_example_2_snippet_constructs(self) -> None:
        output = StructuredOutput(
            messages=[
                Request(
                    message_type="response",
                    message="I'll coordinate...",
                    recipient="@Human",
                ),
                Request(
                    message_type="instruction",
                    message="Implement OAuth",
                    recipient="@Developer456",
                ),
                Request(
                    message_type="request",
                    message="Audit auth flow",
                    recipient="SecurityEngineer",
                ),
            ]
        )

        assert [r.recipient for r in output.messages] == [
            "@Human",
            "@Developer456",
            "SecurityEngineer",
        ]

    def test_example_4_snippets_construct(self) -> None:
        first = StructuredOutput(
            messages=[
                Request(
                    message_type="instruction",
                    message="Start with Task 1: design the auth flow",
                    recipient="@Developer456",
                )
            ]
        )
        second = StructuredOutput(
            messages=[
                Request(
                    message_type="response",
                    message="Design complete — JWT + OAuth2. Ready for Task 2.",
                    recipient="@Manager",
                )
            ]
        )

        assert first.messages[0].message_type == "instruction"
        assert second.messages[0].message_type == "response"


class TestReplyProtocolsTable:
    """The README's `Message Protocol` table, keyed on the same five intents."""

    def test_table_matches_the_documented_text_verbatim(self) -> None:
        assert REPLY_PROTOCOLS == {
            "request": "A reply is expected: respond to {sender} with the result.",
            "response": (
                "This is a reply to something you asked. Take it into account and continue."
            ),
            "instruction": "Carry it out; acknowledge to {sender} only if asked to.",
            "notification": "Informational message. No reply is expected.",
            "acknowledgment": "Receipt confirmed. No further action needed.",
        }

    def test_the_protocols_state_mechanics_and_never_team_policy(self) -> None:
        """The constraint the wording serves, which a verbatim copy cannot express.

        These lines sit at the most salient position in the prompt, so anything
        they say about *who does the work* is applied to every agent in every
        team. Saying "you may also delegate" made coordinators do their
        specialists' work; saying "delegate" made specialists fan out to each
        other. Both directions are policy and belong in the agents' own prompts.

        This is the invariant a future wording change must not break — the
        verbatim table above only pins today's text.
        """
        for intent, protocol in REPLY_PROTOCOLS.items():
            lowered = protocol.lower()
            assert "delegate" not in lowered, f"{intent} states delegation policy"
            assert "carry out the task" not in lowered, f"{intent} assigns the work itself"

    def test_every_intent_a_request_can_carry_has_a_protocol_line(self) -> None:
        """A missing key would silently degrade the receiver prefix to no guidance."""
        intents = Request.model_fields["message_type"].annotation
        assert set(getattr(intents, "__args__", ())) == set(REPLY_PROTOCOLS)


# =============================================================================
# Routing and delivery — the `_route_output` snippet in both documents
# =============================================================================


class TestRoutedTurnSnippet:
    """`Key Implementation §1` and the README `Routing and Delivery` prose."""

    def _agent_returning(self, output: StructuredOutput) -> BaseAgent:
        agent = _make_agent()
        _react_agent_of(agent).run_sync.return_value = output
        return agent

    def test_at_prefix_resolves_an_existing_member(self) -> None:
        agent = self._agent_returning(
            StructuredOutput(
                messages=[
                    Request(
                        message_type="request", message="do this", recipient="@Assistant"
                    )
                ]
            )
        )
        member = _make_address("@Assistant")
        get_team_member = _stub(agent, "get_team_member", return_value=member)
        hire = _stub(agent, "hire_member")

        assert _run_turn(agent) is True

        get_team_member.assert_called_once_with("@Assistant")
        hire.assert_not_called()

    def test_bare_role_name_triggers_hire(self) -> None:
        agent = self._agent_returning(
            StructuredOutput(
                messages=[
                    Request(
                        message_type="request",
                        message="Audit auth flow",
                        recipient="SecurityEngineer",
                    )
                ]
            )
        )
        hired = _make_address("@SecurityEngineer456")
        hire = _stub(agent, "hire_member", return_value=hired)

        _run_turn(agent)

        hire.assert_called_once_with("SecurityEngineer")
        assert cast(MagicMock, agent.send).call_args[0][0] is hired

    def test_the_sender_transmits_raw_content_and_copies_the_intent(self) -> None:
        """Guards the corrected attribution: no prefix is baked in by the SENDER."""
        agent = self._agent_returning(
            StructuredOutput(
                messages=[
                    Request(
                        message_type="instruction",
                        message="Implement OAuth",
                        recipient="@Developer456",
                    )
                ]
            )
        )
        member = _make_address("@Developer456")
        _stub(agent, "get_team_member", return_value=member)

        _run_turn(agent)

        sent = cast(MagicMock, agent.send).call_args[0][1]
        assert sent.content == "Implement OAuth"
        assert "You received" not in sent.content
        assert sent.type == "instruction"
        assert sent.recipient is member

    def test_unresolvable_at_member_is_skipped_not_raised(self) -> None:
        agent = self._agent_returning(
            StructuredOutput(
                messages=[
                    Request(
                        message_type="request", message="hi", recipient="@Designer"
                    )
                ]
            )
        )
        _stub(agent, "get_team_member", return_value=None)

        assert _run_turn(agent) is False

        cast(MagicMock, agent.send).assert_not_called()

    def test_an_empty_output_list_sends_nothing(self) -> None:
        """Example 3: the agent's turn simply ends.

        ``_route_output`` reports ``False`` — the answer the usage-limit guard asks
        for when it has to tell a real conclusion from one that routed nothing.
        """
        agent = self._agent_returning(StructuredOutput(messages=[]))

        assert _run_turn(agent) is False

        cast(MagicMock, agent.send).assert_not_called()


class TestReceiverSidePrefixSnippet:
    """`Key Implementation §2` — the receiver builds the protocol line."""

    @patch("akgentic.agent.agent.sleep", MagicMock())
    def test_receiver_prepends_the_protocol_for_the_intent_it_received(self) -> None:
        agent = _make_agent()
        act = _stub_act(agent)

        message = AgentMessage(content="Estimate feature X", type="request")
        message.sender = _make_address("@Manager")

        agent.receiveMsg_AgentMessage(message, _make_address("@Manager"))

        prompt = act.call_args[0][0]
        # The subject here is the COMPOSITION — framing, article, protocol, blank
        # line, raw content — not the protocol's wording, which is pinned verbatim
        # by TestReplyProtocolsTable. Deriving that one span keeps this test on its
        # own subject and stops a deliberate rewording from failing it.
        protocol = REPLY_PROTOCOLS["request"].format(sender="@Manager")
        assert prompt == (
            f"You received a request from @Manager. {protocol}\n\nEstimate feature X"
        )

    @patch("akgentic.agent.agent.sleep", MagicMock())
    def test_the_article_agrees_with_the_intent(self) -> None:
        """The prefix says `an instruction` / `an acknowledgment`, but `a request`."""
        agent = _make_agent()
        act = _stub_act(agent)

        message = AgentMessage(content="Do X", type="instruction")
        message.sender = _make_address("@Manager")
        agent.receiveMsg_AgentMessage(message, _make_address("@Manager"))

        assert act.call_args[0][0].startswith("You received an instruction from @Manager.")

    @patch("akgentic.agent.agent.sleep", MagicMock())
    def test_raw_content_is_preserved_after_the_prefix(self) -> None:
        agent = _make_agent()
        act = _stub_act(agent)

        message = AgentMessage(content="3 days", type="response")
        message.sender = _make_address("@Developer456")
        agent.receiveMsg_AgentMessage(message, _make_address("@Developer456"))

        assert act.call_args[0][0].endswith("\n\n3 days")


# =============================================================================
# The command registry — README `The Command Registry`, collaboration §4, DO #4
# =============================================================================


# These two are registry stand-ins, so their __name__ IS the command name — the
# registry keys by it. They must therefore be spelled exactly as the docs print them.
def hire_member(role: str, name: str | None = None) -> str:
    """Stand-in for TeamTool's hire_member command."""
    return f"hired {role}" + (f" as {name}" if name else "")


def team_members() -> str:
    """Stand-in for TeamTool's team_members command."""
    return "**Here is the team member list by name (and role):**\n@Manager (role: Manager)"


class TestCommandRegistrySurfaces:
    """The two documented surfaces, on a real CommandRegistry."""

    def _registry(self) -> Any:
        return ToolFactory(tool_cards=[]).get_command_registry(
            extra_commands=[hire_member, team_members]
        )

    def test_has_reports_availability(self) -> None:
        registry = self._registry()
        assert registry.has("hire_member") is True
        assert registry.has("cmd_hire_member") is False

    def test_typed_surface_returns_the_native_value(self) -> None:
        registry = self._registry()
        assert registry.callable("hire_member")("Developer") == "hired Developer"

    def test_typed_surface_raises_for_an_unknown_name(self) -> None:
        with pytest.raises(CommandNotRecognized):
            self._registry().callable("cmd_get_planning")

    def test_text_surface_dispatches_a_slash_command(self) -> None:
        assert self._registry().dispatch("/hire_member Developer") == "hired Developer"

    def test_text_surface_binds_documented_keyword_arguments(self) -> None:
        """`/hire_member Developer name=@Ada` binds both, as the README states."""
        result = self._registry().dispatch("/hire_member Developer name=@Ada")
        assert result == "hired Developer as @Ada"

    def test_unknown_leading_token_raises_command_not_recognized(self) -> None:
        with pytest.raises(CommandNotRecognized):
            self._registry().dispatch("/definitely_not_a_command")

    def test_bad_arguments_come_back_as_a_result_string_not_an_exception(self) -> None:
        """Documented: post-identification failures never fall back to the LLM."""
        result = self._registry().dispatch("/hire_member")
        assert result.startswith("Command 'hire_member' failed:")

    def test_descriptors_expose_name_description_and_args(self) -> None:
        by_name = {d.name: d for d in self._registry().descriptors()}

        assert set(by_name) == {"hire_member", "team_members"}
        hire = by_name["hire_member"]
        assert hire.description.startswith("Stand-in for TeamTool's hire_member")
        assert [(a.name, a.required) for a in hire.args] == [
            ("role", True),
            ("name", False),
        ]

    def test_commands_are_keyed_by_callable_name_with_no_cmd_prefix(self) -> None:
        names = {d.name for d in self._registry().descriptors()}
        assert not any(name.startswith("cmd_") for name in names)


class TestSlashDispatchFallback:
    """`_dispatch_command` — the behaviour the fallback paragraph describes."""

    def test_a_known_command_is_handled_and_replies_as_a_notification(self) -> None:
        agent = _make_agent(commands={"team_members": team_members})

        message = AgentMessage(content="/team_members", type="request")
        message.sender = _make_address("@Human")

        handled = agent._dispatch_command(message, _make_address("@Human"))

        assert handled is True
        reply = cast(MagicMock, agent.send).call_args[0][1]
        assert reply.type == "notification"
        assert reply.content.startswith("**Here is the team member list")

    def test_a_known_command_records_exactly_one_operator_action(self) -> None:
        agent = _make_agent(commands={"team_members": team_members})

        message = AgentMessage(content="/team_members", type="request")
        message.sender = _make_address("@Human")
        agent._dispatch_command(message, _make_address("@Human"))

        record = _react_agent_of(agent).context.record_operator_action
        record.assert_called_once()
        assert '"/team_members"' in record.call_args[0][0]

    def test_an_unknown_token_falls_through_and_records_nothing(self) -> None:
        agent = _make_agent(commands={"team_members": team_members})

        message = AgentMessage(content="/not/a/path we know", type="request")
        message.sender = _make_address("@Human")

        handled = agent._dispatch_command(message, _make_address("@Human"))

        assert handled is False
        cast(MagicMock, agent.send).assert_not_called()
        _react_agent_of(agent).context.record_operator_action.assert_not_called()

    @patch("akgentic.agent.agent.sleep", MagicMock())
    def test_an_unrecognised_slash_message_reaches_the_normal_llm_path(self) -> None:
        """Documented: a sentence that happens to start with a slash is never lost."""
        agent = _make_agent(commands={"team_members": team_members})
        act = _stub_act(agent)

        message = AgentMessage(content="/usr/local is full — please fix", type="request")
        message.sender = _make_address("@Human")

        agent.receiveMsg_AgentMessage(message, _make_address("@Human"))

        act.assert_called_once()
        assert act.call_args[0][0].endswith("\n\n/usr/local is full — please fix")

    @patch("akgentic.agent.agent.sleep", MagicMock())
    def test_a_recognised_slash_message_never_reaches_the_llm_path(self) -> None:
        agent = _make_agent(commands={"team_members": team_members})
        act = _stub_act(agent)

        message = AgentMessage(content="/team_members", type="request")
        message.sender = _make_address("@Human")

        agent.receiveMsg_AgentMessage(message, _make_address("@Human"))

        act.assert_not_called()

    def test_a_none_result_is_handled_with_no_reply_and_no_operator_action(self) -> None:
        """The third fallback bullet: `None` means *handled, say nothing*.

        Dispatch is stubbed rather than driven through a real command, and that
        is not a shortcut: ``CommandRegistry._invoke`` currently returns
        ``str(fn(...))``, so a ``None``-returning command reaches this method as
        the string ``"None"``. The `None` outcome is a contract this agent
        honours ahead of the tool package widening that signature — the double
        is the only way to exercise it today.
        """
        agent = _make_agent(commands={"team_members": team_members})
        setattr(agent._command_registry, "dispatch", MagicMock(return_value=None))

        message = AgentMessage(content="/team_members", type="request")
        message.sender = _make_address("@Human")

        handled = agent._dispatch_command(message, _make_address("@Human"))

        assert handled is True
        cast(MagicMock, agent.send).assert_not_called()
        _react_agent_of(agent).context.record_operator_action.assert_not_called()


class TestHireMemberSnippet:
    """`Key Implementation §3`, corrected to the registry form."""

    def test_raises_runtime_error_when_the_command_is_not_registered(self) -> None:
        agent = _make_agent()

        with pytest.raises(RuntimeError, match="TeamTool not configured"):
            agent.hire_member("Developer")

    def test_resolves_and_invokes_the_typed_callable(self) -> None:
        address = _make_address("@Developer456")

        def hire_member(role: str) -> Any:
            """Registry stand-in returning a native ActorAddress."""
            return address

        agent = _make_agent(commands={"hire_member": hire_member})

        assert agent.hire_member("Developer") is address

    def test_model_retry_from_the_command_propagates_for_llm_retry(self) -> None:
        def hire_member(role: str) -> Any:
            """Registry stand-in that fails the way an invalid role does."""
            raise ModelRetry(f"no agent card for role '{role}'")

        agent = _make_agent(commands={"hire_member": hire_member})

        with pytest.raises(ModelRetry):
            agent.hire_member("Nonexistent")


class TestNoCmdApiRemains:
    """The headline defect, asserted against the class rather than the prose."""

    def test_base_agent_exposes_no_cmd_prefixed_attribute(self) -> None:
        assert [name for name in dir(BaseAgent) if name.startswith("cmd_")] == []

    def test_the_documented_replacements_are_real_methods(self) -> None:
        for method in (
            "act",
            "_route_output",
            "receiveMsg_AgentMessage",
            "hire_member",
            "notify_human",
            "get_usage_summary",
            "compact",
            "clear",
            "init_llm_context",
            "on_start",
            "on_stop",
        ):
            assert callable(getattr(BaseAgent, method))


class TestSimpleTeamAliasSnippet:
    """DO #4's alias map must point at names TeamTool/PlanningTool really register.

    The alias map is transcribed from ``examples/simple_team.py``; the names it maps
    *onto* are read from a live registry. Renaming a command in ``akgentic-tool``
    therefore breaks this test rather than silently staling the snippet.
    """

    def test_every_alias_target_is_a_real_registered_command(self) -> None:
        documented_aliases = {
            "team": "team_members",
            "roles": "team_roles",
            "planning": "planning_summary",
            "task": "get_planning_task",
            "hire": "hire_member",
            "fire": "fire_member",
        }

        registered = _announced_command_names(
            [TeamTool(), PlanningTool(vector_store=False)]
        )

        missing = {
            alias: real
            for alias, real in documented_aliases.items()
            if real not in registered
        }
        assert not missing, f"alias targets that no command registers: {missing}"


# =============================================================================
# act() — the media-expansion paragraph in the README
# =============================================================================


class TestActMediaExpansion:
    """`act()` only swaps in a parts prompt when expansion changed something."""

    def test_plain_string_is_sent_when_no_expansion_command_exists(self) -> None:
        agent = _make_agent()
        _react_agent_of(agent).run_sync.return_value = StructuredOutput()

        agent.act("just text", StructuredOutput)

        assert _react_agent_of(agent).run_sync.call_args[0][0] == "just text"

    def test_unchanged_expansion_leaves_the_prompt_a_plain_string(self) -> None:
        def _expand_media_refs(prompt: str) -> Any:
            """Registry stand-in returning the prompt untouched."""
            return [prompt]

        agent = _make_agent(commands={"_expand_media_refs": _expand_media_refs})
        _react_agent_of(agent).run_sync.return_value = StructuredOutput()

        agent.act("just text", StructuredOutput)

        assert _react_agent_of(agent).run_sync.call_args[0][0] == "just text"

    def test_changed_expansion_switches_to_a_parts_prompt(self) -> None:
        def _expand_media_refs(prompt: str) -> Any:
            """Registry stand-in that splits the prompt."""
            return ["look at ", "", "the shot"]

        agent = _make_agent(commands={"_expand_media_refs": _expand_media_refs})
        _react_agent_of(agent).run_sync.return_value = StructuredOutput()

        agent.act("look at !!shot.png the shot", StructuredOutput)

        assert _react_agent_of(agent).run_sync.call_args[0][0] == [
            "look at ",
            "",
            "the shot",
        ]


# =============================================================================
# act() — README `Static Schema + Prompt-Carried Reply Protocol`
#         + collaboration doc §2 (the run_sync snippet both documents show)
# =============================================================================


class TestActOutputTypePassThrough:
    """Both documents show `run_sync(..., output_type=output_type)`.

    The snippet is only honest if the caller's type really reaches the loop, and the
    delegation path is only schema-driven because ``receiveMsg_AgentMessage`` names
    ``StructuredOutput`` itself. One test per half of that claim.
    """

    def test_the_callers_output_type_reaches_the_react_loop(self) -> None:
        """Documented: act() reasons against the type it is handed, not a hardcoded one."""

        class Summary(BaseModel):
            headline: str

        agent = _make_agent()
        _react_agent_of(agent).run_sync.return_value = Summary(headline="ok")

        agent.act("anything", Summary)

        kwargs = _react_agent_of(agent).run_sync.call_args[1]
        assert kwargs["output_type"] is Summary

    @patch("akgentic.agent.agent.sleep", MagicMock())
    def test_the_handler_asks_the_loop_for_structured_output(self) -> None:
        """The routing path names StructuredOutput itself — pass-through keeps it schema-driven.

        Driven through ``receiveMsg_AgentMessage`` rather than ``act()`` directly: the
        claim is about what the *routing* entry point asks for, so re-pointing it at
        another type must turn this red. The handler is also what the guard decorator
        was given ``StructuredOutput`` for, so the normal turn and an interrupted one
        reason against the same schema by construction.
        """
        agent = _make_agent()
        _react_agent_of(agent).run_sync.return_value = StructuredOutput()

        message = AgentMessage(content="anything", type="request")
        message.sender = _make_address("@Human")
        agent.receiveMsg_AgentMessage(message, _make_address("@Human"))

        kwargs = _react_agent_of(agent).run_sync.call_args[1]
        assert kwargs["output_type"] is StructuredOutput


# =============================================================================
# extra_capabilities() — README `Adding a capability of your own`
#                        + collaboration doc §6 (`Supplied here`)
# =============================================================================


class AuditCapability(AbstractCapability[Any]):
    """The capability the README's `AuditedAgent` snippet returns."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    async def before_model_request(
        self, ctx: Any, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return request_context


class AuditedAgent(BaseAgent):
    """Transcribed verbatim from the README's `Adding a capability of your own`."""

    def extra_capabilities(self) -> list[AgentCapability[Any]]:
        return [AuditCapability(self.config.name)]


class TestExtraCapabilitiesSnippet:
    """Both documents claim the list is `[mailbox, *extra_capabilities()]`.

    The snippet is only honest if a subclass overriding nothing else really gets
    its capability into the ReactAgent, in that order, without touching
    ``_build_react_agent``. ``ReactAgent`` is replaced with a recorder so the
    assembly is observed at the build site the documentation names.
    """

    @staticmethod
    def _build(agent: BaseAgent, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        monkeypatch.delenv("AKGENTIC_MOCK_SCENARIO", raising=False)
        captured: dict[str, Any] = {}

        class _RecordingReactAgent:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(agent_module, "ReactAgent", _RecordingReactAgent)
        agent._build_react_agent(MagicMock(), [], [])
        return cast(list[Any], captured["capabilities"])

    def test_the_documented_subclass_gets_its_capability_wired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent: AuditedAgent = object.__new__(AuditedAgent)
        config = MagicMock(spec=AgentConfig)
        config.name = "@Audited"
        agent.config = config  # type: ignore[attr-defined]

        capabilities = self._build(agent, monkeypatch)

        # Documented order: the framework's own first, the subclass's second —
        # handed over as a copy, so pydantic-ai's auto-capabilities cannot land
        # back on the agent's own list.
        assert capabilities == agent._capabilities
        assert capabilities is not agent._capabilities
        assert len(capabilities) == 2
        assert capabilities[0] is agent._mailbox_capability
        assert isinstance(capabilities[0], MailboxCapability)
        assert isinstance(capabilities[1], AuditCapability)
        # Documented: an override may read self.config.
        assert capabilities[1].agent_name == "@Audited"

    def test_a_subclass_overriding_nothing_gets_only_the_framework_capability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Documented: `extra_capabilities()` returns `[]` on BaseAgent."""
        agent: BaseAgent = object.__new__(BaseAgent)

        assert agent.extra_capabilities() == []
        capabilities = self._build(agent, monkeypatch)

        assert len(capabilities) == 1
        assert capabilities[0] is agent._mailbox_capability


# =============================================================================
# Configuration — README `AgentConfig` table + collaboration API Reference
# =============================================================================


class TestAgentConfigDocumentedShape:
    """The field table and the API-Reference docstring, made checkable."""

    def test_every_field_has_a_default(self) -> None:
        config = AgentConfig()

        assert isinstance(config.prompt, PromptTemplate)
        assert isinstance(config.model_cfg, ModelConfig)
        assert isinstance(config.runtime_cfg, RuntimeConfig)
        assert isinstance(config.run_usage_limits, RunUsageLimits)
        assert isinstance(config.agent_usage_limits, AgentUsageLimits)
        assert isinstance(config.compaction_cfg, CompactionConfig)
        assert config.tools == []

    def test_prompt_is_a_prompt_template_not_a_bare_string(self) -> None:
        """The API Reference used to say "string or PromptTemplate"."""
        with pytest.raises(ValidationError):
            AgentConfig(prompt="You are a project manager.")  # type: ignore[arg-type]

    def test_temperature_lives_on_model_cfg_not_runtime_cfg(self) -> None:
        assert "temperature" in ModelConfig.model_fields
        assert "temperature" not in RuntimeConfig.model_fields

    def test_runtime_cfg_carries_the_documented_fields(self) -> None:
        assert {
            "retries",
            "end_strategy",
            "parallel_tool_calls",
            "http_client_config",
        } <= set(RuntimeConfig.model_fields)


class TestAgentCardSnippet:
    """`role` is a derived property, not a constructor keyword.

    Both documents printed `AgentCard(role="Manager", ...)` for several releases.
    Pydantic ignores the unknown keyword, so the card only worked because the
    snippets also set `config.role`. A reader who dropped the config field — the
    natural reading, since `role=` looked authoritative — got a card with no role.
    """

    def _card(self, config_role: str) -> AgentCard:
        return AgentCard(
            description="Writes and reviews code",
            skills=["python", "testing"],
            agent_class="akgentic.agent.BaseAgent",
            config=AgentConfig(name="@Developer", role=config_role),
            routes_to=["Reviewer", "Tester"],
        )

    def test_role_is_not_a_declared_field(self) -> None:
        assert "role" not in AgentCard.model_fields

    def test_role_reads_through_to_config_role(self) -> None:
        assert self._card("Developer").role == "Developer"

    def test_a_role_keyword_is_silently_ignored(self) -> None:
        """Constructed with a `role=` that disagrees, `config.role` still wins."""
        card = AgentCard(
            role="LooksAuthoritative",  # type: ignore[call-arg]
            description="Writes and reviews code",
            agent_class="akgentic.agent.BaseAgent",
            config=AgentConfig(name="@Developer", role="Developer"),
        )
        assert card.role == "Developer"

    def test_get_config_copy_returns_a_fresh_independent_config(self) -> None:
        card = self._card("Developer")
        copy = card.get_config_copy()

        assert isinstance(copy, AgentConfig)
        assert copy is not card.config
        assert copy.role == "Developer"


class TestUsageLimitsShimClaims:
    """The corrected `Migrating from usage_limits` paragraph."""

    def test_the_agent_config_shims_still_work_and_warn(self) -> None:
        with pytest.warns(DeprecationWarning):
            config = AgentConfig(
                usage_limits=RunUsageLimits(run_request_limit=50)  # type: ignore[call-arg]
            )
        assert config.run_usage_limits.run_request_limit == 50

        with pytest.warns(DeprecationWarning):
            assert config.usage_limits is config.run_usage_limits

    def test_both_spellings_together_raise(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(
                usage_limits=RunUsageLimits(run_request_limit=50),  # type: ignore[call-arg]
                run_usage_limits=RunUsageLimits(run_request_limit=10),
            )

    def test_the_llm_owned_usage_limits_alias_still_ships(self) -> None:
        """The README used to claim it was removed in akgentic-llm 2.0.0."""
        from akgentic.llm.config import UsageLimits

        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=50)  # type: ignore[call-arg]

        assert isinstance(limits, RunUsageLimits)
        assert limits.run_request_limit == 50


class TestUpdatePlanningInstructionsSnippet:
    """The `Custom Instructions via UpdatePlanning` block."""

    def test_instructions_are_appended_under_the_documented_header(self) -> None:
        params = UpdatePlanning(instructions="CRITICAL: Always keep the plan updated.")

        formatted = params.format_docstring("Update team tasks (create, update, delete).")

        assert formatted == (
            "Update team tasks (create, update, delete)."
            "\n\nAdditional Instructions:\n"
            "CRITICAL: Always keep the plan updated."
        )

    def test_no_instructions_leaves_the_docstring_untouched(self) -> None:
        original = "Update team tasks (create, update, delete)."
        assert UpdatePlanning().format_docstring(original) == original

    def test_planning_is_filtered_to_the_reading_agent_by_default(self) -> None:
        """The doc used to show the whole board as the default view; it is the opt-out."""
        assert GetPlanning().filter_by_agent is True
        assert GetPlanning(filter_by_agent=False).filter_by_agent is False


# =============================================================================
# The README EventSubscriber snippet
# =============================================================================


class TestEventSubscriberSnippet:
    """The README `EventSubscriber` example, run against real telemetry messages."""

    def test_snippet_handles_both_documented_branches(self) -> None:
        printed: list[str] = []

        class MessagePrinter(EventSubscriber):
            def on_message(self, message: Message) -> None:
                if isinstance(message, SentMessage):
                    sender = cast(ActorAddress, message.sender)
                    body = cast(AgentMessage, message.message)
                    printed.append(
                        f"[{sender.name}] → {message.recipient.name}: {body.content}"
                    )
                elif isinstance(message, EventMessage) and isinstance(
                    message.event, ToolCallEvent
                ):
                    printed.append(f"TOOL: {message.event.tool_name}")

        # EventSubscriber is a Protocol whose hooks all have no-op bodies, so a
        # subscriber implements only what it cares about — as the README states.
        subscriber = MessagePrinter()  # type: ignore[abstract]

        sent = SentMessage(
            message=AgentMessage(content="Implement OAuth"),
            recipient=_make_address("@Developer456"),
        )
        sent.sender = _make_address("@Manager")
        subscriber.on_message(sent)

        event = EventMessage(
            event=ToolCallEvent(
                run_id="run-1",
                tool_name="hire_members",
                tool_call_id="call-1",
                arguments='{"roles": ["Developer"]}',
            )
        )
        subscriber.on_message(event)

        assert printed == [
            "[@Manager] → @Developer456: Implement OAuth",
            "TOOL: hire_members",
        ]


# =============================================================================
# The documented command table, read from the live registry (never transcribed)
# =============================================================================


class _CapturingReactAgent:
    """Stand-in that records what on_start builds, without touching a model."""

    captured: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        _CapturingReactAgent.captured.append(kwargs)

    def system_prompt(self, fn: Any) -> Any:
        return fn

    def close(self) -> None:
        pass


def _announced_command_names(tool_cards: list[ToolCard]) -> set[str]:
    """Start a real BaseAgent carrying *tool_cards*; return the names it announces.

    Goes through the production ``on_start`` path inside a live ActorSystem, so the
    result is whatever the tools actually register — never a list transcribed from
    the documentation.
    """
    announced: list[CommandsAnnouncedEvent] = []

    class _Subscriber(EventSubscriber):
        def on_message(self, message: Message) -> None:
            if isinstance(message, EventMessage) and isinstance(
                message.event, CommandsAnnouncedEvent
            ):
                announced.append(message.event)

    system = ActorSystem()
    original = agent_module.ReactAgent  # type: ignore[attr-defined]
    agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[assignment, attr-defined]
    try:
        orch_addr = system.createActor(
            Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
        )
        orchestrator = system.proxy_ask(orch_addr, Orchestrator)
        orchestrator.subscribe(_Subscriber())  # type: ignore[abstract]

        card = AgentCard(
            description="Coordinates the team",
            skills=["coordination"],
            agent_class="akgentic.agent.BaseAgent",
            config=AgentConfig(
                name="@Manager",
                role="Manager",
                prompt=PromptTemplate(template="You are a manager."),
                model_cfg=ModelConfig(provider="openai", model="gpt-4.1"),
                tools=tool_cards,
            ),
        )
        orchestrator.register_agent_profiles([card])
        manager_addr = orchestrator.createActor(BaseAgent, config=card.get_config_copy())

        deadline = time.monotonic() + 5.0
        while not announced and time.monotonic() < deadline:
            time.sleep(0.05)

        for_manager = [e for e in announced if e.agent.name == manager_addr.name]
        assert len(for_manager) == 1, "expected exactly one announcement"
        return {descriptor.name for descriptor in for_manager[0].commands}
    finally:
        agent_module.ReactAgent = original  # type: ignore[attr-defined]
        try:
            system.shutdown(timeout=5)
        except Exception:  # noqa: BLE001 — teardown must not mask a failure
            pass


class TestAnnouncedCommandSet:
    """`descriptors()` is the source of truth for the documented command table.

    Rather than transcribe command names into the docs and hope they stay put,
    this pins the real registry a live `BaseAgent` builds from a real `TeamTool`.
    """

    def test_teamtool_commands_carry_the_documented_canonical_names(self) -> None:
        _CapturingReactAgent.captured = []
        announced: list[CommandsAnnouncedEvent] = []

        class _Subscriber(EventSubscriber):
            def on_message(self, message: Message) -> None:
                if isinstance(message, EventMessage) and isinstance(
                    message.event, CommandsAnnouncedEvent
                ):
                    announced.append(message.event)

        system = ActorSystem()
        original = agent_module.ReactAgent  # type: ignore[attr-defined]
        agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[assignment, attr-defined]
        try:
            orch_addr = system.createActor(
                Orchestrator,
                config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
            )
            orchestrator = system.proxy_ask(orch_addr, Orchestrator)
            orchestrator.subscribe(_Subscriber())  # type: ignore[abstract]

            card = AgentCard(
                description="Coordinates the team",
                skills=["coordination"],
                agent_class="akgentic.agent.BaseAgent",
                config=AgentConfig(
                    name="@Manager",
                    role="Manager",
                    prompt=PromptTemplate(template="You are a manager."),
                    model_cfg=ModelConfig(provider="openai", model="gpt-4.1"),
                    tools=[TeamTool()],
                ),
            )
            orchestrator.register_agent_profiles([card])
            manager_addr = orchestrator.createActor(
                BaseAgent, config=card.get_config_copy()
            )

            deadline = time.monotonic() + 5.0
            while not announced and time.monotonic() < deadline:
                time.sleep(0.05)

            for_manager = [e for e in announced if e.agent.name == manager_addr.name]
            assert len(for_manager) == 1, "expected exactly one announcement"

            names = {descriptor.name for descriptor in for_manager[0].commands}

            # The names the documents print for TeamTool + the BaseAgent built-ins.
            assert {
                "hire_member",
                "fire_member",
                "team_members",
                "team_roles",
                "compact",
                "clear",
            } <= names

            # And the API the documents used to print does not exist.
            assert not any(name.startswith("cmd_") for name in names)

            # Documented as command-only built-ins: neither reaches the TOOL_CALL channel.
            assert _CapturingReactAgent.captured, "on_start never built a ReactAgent"
            tool_names = {
                getattr(tool, "__name__", None)
                for tool in _CapturingReactAgent.captured[-1]["tools"]
            }
            assert "compact" not in tool_names
            assert "clear" not in tool_names
        finally:
            agent_module.ReactAgent = original  # type: ignore[attr-defined]
            try:
                system.shutdown(timeout=5)
            except Exception:  # noqa: BLE001 — teardown must not mask a failure
                pass


# =============================================================================
# The README Quick Start
# =============================================================================


class TestQuickStartSnippet:
    """The README Quick Start, run end to end with a stubbed ReactAgent."""

    def test_snippet_wires_a_team_and_delivers_the_first_message(self) -> None:
        _CapturingReactAgent.captured = []
        system = ActorSystem()
        original = agent_module.ReactAgent  # type: ignore[attr-defined]
        agent_module.ReactAgent = _CapturingReactAgent  # type: ignore[assignment, attr-defined]
        try:
            orchestrator_addr = system.createActor(
                Orchestrator,
                config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
            )
            orchestrator_proxy = system.proxy_ask(orchestrator_addr, Orchestrator)

            manager_card = AgentCard(
                description="Project manager who coordinates specialists",
                skills=["coordination", "delegation"],
                agent_class="akgentic.agent.BaseAgent",
                config=AgentConfig(
                    name="@Manager",
                    role="Manager",
                    prompt=PromptTemplate(
                        template="You are a project manager. Delegate to specialists."
                    ),
                    model_cfg=ModelConfig(provider="openai", model="gpt-4.1"),
                ),
                routes_to=["Developer", "QA"],
            )
            orchestrator_proxy.register_agent_profiles([manager_card])

            from akgentic.agent import HumanProxy

            human_addr = orchestrator_proxy.createActor(
                HumanProxy, config=BaseConfig(name="@Human", role="Human")
            )
            human_proxy = system.proxy_tell(human_addr, HumanProxy)

            manager_addr = orchestrator_proxy.createActor(
                BaseAgent, config=manager_card.get_config_copy()
            )

            time.sleep(0.3)

            human_proxy.send(manager_addr, AgentMessage(content="Plan the next sprint."))

            # The blueprint really is registered and instantiable by role.
            assert orchestrator_proxy.get_available_roles() == ["Manager"]
            assert orchestrator_proxy.get_team_member("@Manager") is not None

            # get_config_copy() hands back a fresh, independent AgentConfig.
            copy = manager_card.get_config_copy()
            assert isinstance(copy, AgentConfig)
            assert copy is not manager_card.config
        finally:
            agent_module.ReactAgent = original  # type: ignore[attr-defined]
            try:
                system.shutdown(timeout=5)
            except Exception:  # noqa: BLE001 — teardown must not mask a failure
                pass


class TestHumanProxySnippet:
    """The README/API-Reference claims about HumanProxy's two roles."""

    def test_process_human_input_routes_back_to_the_original_sender(self) -> None:
        from akgentic.agent import HumanProxy

        proxy: HumanProxy = object.__new__(HumanProxy)
        send = _stub(proxy, "send")
        proxy._current_message = None

        asking_agent = _make_address("@Manager")
        original = AgentMessage(content="Should we proceed?")
        original.sender = asking_agent

        proxy.process_human_input("Yes, proceed", original)

        target, reply = send.call_args[0]
        assert target is asking_agent
        assert reply.content == "Yes, proceed"
        assert reply.type == "response"
        assert proxy._current_message is None  # reset in the finally block
