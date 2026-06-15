"""Tests for BaseAgent methods not covered by test_agent.py or test_simple_team.py.

Covers: process_message, receiveMsg_AgentMessage (incl. slash-command dispatch),
notify_human, on_hire, on_fire, hire_member, and the on_start discovery event.
All tests use _make_minimal_agent pattern — no live actors or LLM calls. The
command surface is exercised through a mocked CommandRegistry.
"""

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core import ActorAddress
from akgentic.tool.errors import CommandNotRecognized
from pydantic_ai import ModelRetry

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.agent.messages import AgentMessage
from akgentic.agent.output_models import Request, StructuredOutput

# =============================================================================
# HELPERS (same pattern as test_agent.py)
# =============================================================================


def _make_mock_sender(name: str = "@Human") -> MagicMock:
    """Return a mock ActorAddress that passes isinstance checks."""
    addr = MagicMock(spec=ActorAddress)
    addr.name = name
    return addr


def _make_mock_message(sender_name: str = "@Human") -> MagicMock:
    """Return a mock AgentMessage with sender set."""
    msg = MagicMock(spec=AgentMessage)
    msg.sender = _make_mock_sender(sender_name)
    msg.type = "request"
    return msg


def _make_registry(callables: dict[str, Callable] | None = None) -> MagicMock:
    """Return a mock CommandRegistry backed by ``callables``.

    - ``has(name)`` → name in callables
    - ``callable(name)`` → callables[name], else raises CommandNotRecognized
    - ``dispatch`` / ``descriptors`` are plain MagicMocks the caller configures.
    """
    table = callables or {}
    registry = MagicMock()
    registry.has.side_effect = lambda name: name in table

    def _callable(name: str) -> Callable:
        try:
            return table[name]
        except KeyError:
            raise CommandNotRecognized(name) from None

    registry.callable.side_effect = _callable
    return registry


def _make_minimal_agent(
    name: str = "@TestAgent", callables: dict[str, Callable] | None = None
) -> BaseAgent:
    """Construct a BaseAgent without Pykka actor system."""
    agent: BaseAgent = object.__new__(BaseAgent)

    agent._command_registry = _make_registry(callables)  # type: ignore[attr-defined]
    agent._react_agent = MagicMock()  # type: ignore[attr-defined]
    # __init__ is bypassed (object.__new__), so set the runtime attrs the
    # operator-action buffering path depends on. Default to an empty, iterable
    # context so _context_has_system_prompt() returns False (→ buffer).
    agent._pending_operator_actions = []  # type: ignore[attr-defined]
    agent._react_agent.context.messages = []  # type: ignore[attr-defined]
    agent._current_message = _make_mock_message()  # type: ignore[attr-defined]

    mock_config = MagicMock(spec=AgentConfig)
    mock_config.name = name
    agent.config = mock_config  # type: ignore[attr-defined]

    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.get_available_roles = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.send = MagicMock()  # type: ignore[method-assign]
    agent.get_team_member = MagicMock(return_value=None)  # type: ignore[method-assign]

    return agent


# =============================================================================
# process_message
# =============================================================================


class TestProcessMessage:
    """Test process_message routing logic."""

    def test_routes_to_existing_member_via_at_prefix(self) -> None:
        """recipient starting with '@' → get_team_member; content is raw (no prefix)."""
        agent = _make_minimal_agent()

        member_addr = _make_mock_sender("@Assistant")
        agent.get_team_member = MagicMock(return_value=member_addr)  # type: ignore[method-assign]

        output = StructuredOutput(
            messages=[
                Request(
                    recipient="@Assistant",
                    message="do this",
                    message_type="request",
                )
            ]
        )
        agent._react_agent.run_sync.return_value = output  # type: ignore[attr-defined]

        sender = _make_mock_sender("@Human")
        agent.process_message("test content", sender)

        agent.get_team_member.assert_called_once_with("@Assistant")
        agent.send.assert_called_once()
        sent_msg = agent.send.call_args[0][1]
        assert isinstance(sent_msg, AgentMessage)
        assert sent_msg.type == "request"
        assert sent_msg.content == "do this"

    def test_routes_to_hire_member_without_at_prefix(self) -> None:
        """recipient without '@' → hire via registry callable; content is raw."""
        hired_addr = _make_mock_sender("@Developer")
        hire_cmd = MagicMock(return_value=hired_addr)
        agent = _make_minimal_agent(callables={"hire_member": hire_cmd})

        output = StructuredOutput(
            messages=[
                Request(
                    recipient="Developer",
                    message="build this",
                    message_type="instruction",
                )
            ]
        )
        agent._react_agent.run_sync.return_value = output  # type: ignore[attr-defined]

        sender = _make_mock_sender("@Human")
        agent.process_message("test", sender)

        hire_cmd.assert_called_once_with("Developer")
        agent.send.assert_called_once()
        sent_msg = agent.send.call_args[0][1]
        assert sent_msg.type == "instruction"
        assert sent_msg.content == "build this"

    def test_no_send_when_member_not_found(self) -> None:
        """When get_team_member returns None, no message is sent."""
        agent = _make_minimal_agent()
        agent.get_team_member = MagicMock(return_value=None)  # type: ignore[method-assign]

        output = StructuredOutput(
            messages=[
                Request(
                    recipient="@Ghost",
                    message="hello",
                    message_type="notification",
                )
            ]
        )
        agent._react_agent.run_sync.return_value = output  # type: ignore[attr-defined]

        agent.process_message("test", _make_mock_sender())
        agent.send.assert_not_called()

    def test_empty_messages_no_routing(self) -> None:
        """No requests → no routing happens."""
        agent = _make_minimal_agent()
        output = StructuredOutput(messages=[])
        agent._react_agent.run_sync.return_value = output  # type: ignore[attr-defined]

        agent.process_message("test", _make_mock_sender())
        agent.send.assert_not_called()

    def test_sends_raw_content_without_prefix(self) -> None:
        """process_message sends AgentMessage with raw content (no prefix baked in)."""
        agent = _make_minimal_agent()
        member_addr = _make_mock_sender("@Assistant")
        agent.get_team_member = MagicMock(return_value=member_addr)  # type: ignore[method-assign]

        output = StructuredOutput(
            messages=[
                Request(
                    recipient="@Human",
                    message="ack",
                    message_type="acknowledgment",
                )
            ]
        )
        agent._react_agent.run_sync.return_value = output  # type: ignore[attr-defined]

        agent.process_message("test", _make_mock_sender())
        sent_msg = agent.send.call_args[0][1]
        assert sent_msg.content == "ack"
        assert "You received" not in sent_msg.content


# =============================================================================
# receiveMsg_AgentMessage
# =============================================================================


class TestReceiveAgentMessage:
    """Test receiveMsg_AgentMessage handler."""

    @patch("akgentic.agent.agent.sleep")
    def test_calls_process_message_with_reconstructed_prefix(self, mock_sleep: MagicMock) -> None:
        """Normal path: calls process_message with reconstructed prefix from metadata."""
        agent = _make_minimal_agent()
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        msg_sender = _make_mock_sender("@Alice")
        message = AgentMessage(content="hello world", type="request")
        message.sender = msg_sender

        sender = _make_mock_sender("@Alice")

        agent.receiveMsg_AgentMessage(message, sender)

        expected = "You received a request from @Alice:\n\nhello world"
        agent.process_message.assert_called_once_with(expected, sender)

    @patch("akgentic.agent.agent.sleep")
    def test_usage_limit_error_notifies_human(self, mock_sleep: MagicMock) -> None:
        """LLMUsageLimitError → notify_human + WarningError raised."""
        from akgentic.core.agent import WarningError
        from akgentic.llm import UsageLimitError as LLMUsageLimitError

        agent = _make_minimal_agent()
        agent.process_message = MagicMock(  # type: ignore[method-assign]
            side_effect=LLMUsageLimitError("token limit")
        )
        agent.notify_human = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="do something")
        sender = _make_mock_sender("@Human")

        with pytest.raises(WarningError, match="LLM usage limit exceeded"):
            agent.receiveMsg_AgentMessage(message, sender)

        agent.notify_human.assert_called_once()
        assert "exceeded" in agent.notify_human.call_args[0][0]

    @patch("akgentic.agent.agent.sleep")
    def test_reconstructs_prefix_with_a_for_consonant_type(self, mock_sleep: MagicMock) -> None:
        """type='request' (consonant) → prefix uses 'a request'."""
        agent = _make_minimal_agent()
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="do this", type="request")
        message.sender = _make_mock_sender("@Bob")

        sender = _make_mock_sender("@Bob")
        agent.receiveMsg_AgentMessage(message, sender)

        expected = "You received a request from @Bob:\n\ndo this"
        agent.process_message.assert_called_once_with(expected, sender)

    @patch("akgentic.agent.agent.sleep")
    def test_reconstructs_prefix_with_an_for_acknowledgment(self, mock_sleep: MagicMock) -> None:
        """type='acknowledgment' (vowel) → prefix uses 'an acknowledgment'."""
        agent = _make_minimal_agent()
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="ack", type="acknowledgment")
        message.sender = _make_mock_sender("@Carol")

        agent.receiveMsg_AgentMessage(message, _make_mock_sender("@Carol"))

        called_content = agent.process_message.call_args[0][0]
        assert called_content.startswith("You received an acknowledgment from @Carol:")
        assert called_content.endswith("ack")

    @patch("akgentic.agent.agent.sleep")
    def test_reconstructs_prefix_with_an_for_instruction(self, mock_sleep: MagicMock) -> None:
        """type='instruction' (vowel) → prefix uses 'an instruction'."""
        agent = _make_minimal_agent()
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="step 1", type="instruction")
        message.sender = _make_mock_sender("@Manager")

        agent.receiveMsg_AgentMessage(message, _make_mock_sender("@Manager"))

        called_content = agent.process_message.call_args[0][0]
        assert called_content.startswith("You received an instruction from @Manager:")
        assert called_content.endswith("step 1")

    @patch("akgentic.agent.agent.sleep")
    def test_reconstructs_prefix_with_unknown_when_sender_is_none(
        self, mock_sleep: MagicMock
    ) -> None:
        """message.sender is None → fallback to 'unknown'."""
        agent = _make_minimal_agent()
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="hello", type="request")
        message.sender = None

        agent.receiveMsg_AgentMessage(message, _make_mock_sender("@Somewhere"))

        called_content = agent.process_message.call_args[0][0]
        assert "from unknown:" in called_content
        assert called_content.endswith("hello")


# =============================================================================
# notify_human
# =============================================================================


class TestNotifyHuman:
    """Test notify_human sends notification to first Human agent."""

    def test_sends_notification_to_human(self) -> None:
        agent = _make_minimal_agent()
        human_member = MagicMock(spec=ActorAddress)
        human_member.name = "@Human"
        human_member.role = "human"
        agent.get_team = MagicMock(return_value=[human_member])  # type: ignore[method-assign]

        agent.notify_human("usage limit exceeded")

        agent.send.assert_called_once()
        sent_msg = agent.send.call_args[0][1]
        assert isinstance(sent_msg, AgentMessage)
        assert sent_msg.type == "notification"
        assert "usage limit exceeded" in sent_msg.content


# =============================================================================
# on_hire / on_fire
# =============================================================================


class TestOnHireOnFire:
    """Test on_hire/on_fire callbacks (no-op but must be callable)."""

    def test_on_hire_is_noop(self) -> None:
        agent = _make_minimal_agent()
        agent.on_hire(MagicMock())  # should not raise

    def test_on_fire_is_noop(self) -> None:
        agent = _make_minimal_agent()
        agent.on_fire(MagicMock())  # should not raise


# =============================================================================
# hire_member
# =============================================================================


class TestHireMember:
    """Test hire_member resolves and invokes the registry callable (AC-8)."""

    def test_delegates_to_registry_callable(self) -> None:
        mock_addr = MagicMock()
        hire_cmd = MagicMock(return_value=mock_addr)
        agent = _make_minimal_agent(callables={"hire_member": hire_cmd})

        result = agent.hire_member("Developer")

        hire_cmd.assert_called_once_with("Developer")
        assert result == mock_addr

    def test_raises_when_command_not_registered(self) -> None:
        agent = _make_minimal_agent()  # registry has no hire_member

        with pytest.raises(RuntimeError, match="hire_member command not available"):
            agent.hire_member("Developer")

    def test_propagates_model_retry(self) -> None:
        """ModelRetry from the typed callable is propagated (not swallowed)."""
        hire_cmd = MagicMock(side_effect=ModelRetry("role not found"))
        agent = _make_minimal_agent(callables={"hire_member": hire_cmd})

        with pytest.raises(ModelRetry, match="role not found"):
            agent.hire_member("Unknown")


# =============================================================================
# Slash-command dispatch (receiveMsg_AgentMessage → _dispatch_command)
# =============================================================================


class TestSlashCommandDispatch:
    """AC-3/4/5: /-prefixed interception, fallback, and post-id failure."""

    @patch("akgentic.agent.agent.sleep")
    def test_dispatch_success_sends_result_no_llm(self, mock_sleep: MagicMock) -> None:
        """AC-3: known /command → dispatch result sent to sender, LLM not called."""
        agent = _make_minimal_agent()
        agent._command_registry.dispatch.return_value = "Member @Dev123 hired."  # type: ignore[attr-defined]
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="/hire Developer", type="request")
        message.sender = _make_mock_sender("@Human")
        sender = _make_mock_sender("@Human")

        agent.receiveMsg_AgentMessage(message, sender)

        agent._command_registry.dispatch.assert_called_once_with("/hire Developer")  # type: ignore[attr-defined]
        agent.process_message.assert_not_called()
        agent.send.assert_called_once()
        sent_to, sent_msg = agent.send.call_args[0]
        assert sent_to is sender
        assert isinstance(sent_msg, AgentMessage)
        assert sent_msg.type == "notification"
        assert sent_msg.content == "Member @Dev123 hired."

    @patch("akgentic.agent.agent.sleep")
    def test_command_not_recognized_falls_back_to_llm(self, mock_sleep: MagicMock) -> None:
        """AC-4: CommandNotRecognized → original content goes to the LLM path."""
        agent = _make_minimal_agent()
        agent._command_registry.dispatch.side_effect = CommandNotRecognized("etc")  # type: ignore[attr-defined]
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="/etc/passwd", type="request")
        message.sender = _make_mock_sender("@Human")
        sender = _make_mock_sender("@Human")

        agent.receiveMsg_AgentMessage(message, sender)

        # No result string sent for the dispatch attempt
        agent.send.assert_not_called()
        # Fell back to the normal path with the prefixed ORIGINAL content
        agent.process_message.assert_called_once()
        prefixed = agent.process_message.call_args[0][0]
        assert prefixed.endswith("/etc/passwd")
        assert "You received a request from @Human:" in prefixed

    @patch("akgentic.agent.agent.sleep")
    def test_post_identification_failure_returns_string_no_llm(self, mock_sleep: MagicMock) -> None:
        """AC-5: known command with bad args → dispatch returns a string, no LLM."""
        agent = _make_minimal_agent()
        agent._command_registry.dispatch.return_value = (  # type: ignore[attr-defined]
            "Command 'hire_member' failed: requires at least 1 argument(s), got 0"
        )
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="/hire_member", type="request")
        message.sender = _make_mock_sender("@Human")
        sender = _make_mock_sender("@Human")

        agent.receiveMsg_AgentMessage(message, sender)

        agent.process_message.assert_not_called()
        agent.send.assert_called_once()
        sent_msg = agent.send.call_args[0][1]
        assert sent_msg.type == "notification"
        assert "failed" in sent_msg.content

    @patch("akgentic.agent.agent.sleep")
    def test_non_slash_content_never_dispatches(self, mock_sleep: MagicMock) -> None:
        """Plain (non-/) content bypasses dispatch entirely and goes to the LLM."""
        agent = _make_minimal_agent()
        agent.process_message = MagicMock()  # type: ignore[method-assign]

        message = AgentMessage(content="hello there", type="request")
        message.sender = _make_mock_sender("@Human")

        agent.receiveMsg_AgentMessage(message, _make_mock_sender("@Human"))

        agent._command_registry.dispatch.assert_not_called()  # type: ignore[attr-defined]
        agent.process_message.assert_called_once()


# =============================================================================
# on_start discovery event (CommandsAnnouncedEvent) — actor-system level
# =============================================================================


class TestCommandsAnnouncedEventEmission:
    """AC-2: on_start emits exactly one CommandsAnnouncedEvent for the agent."""

    def test_on_start_emits_single_discovery_event(self) -> None:
        """Creating a BaseAgent emits one CommandsAnnouncedEvent(agent==myAddress).

        Exercised through the real actor system (on_start runs on createActor);
        no LLM call happens because no message is sent. A subscriber captures the
        orchestrator event stream and we assert exactly one CommandsAnnouncedEvent
        is announced for this agent, carrying the registry descriptors.
        """
        import time

        from akgentic.core import ActorSystem, BaseConfig, EventSubscriber, Orchestrator
        from akgentic.core.messages import EventMessage, Message
        from akgentic.llm import ModelConfig, PromptTemplate
        from akgentic.tool.event import CommandsAnnouncedEvent

        announced: list[CommandsAnnouncedEvent] = []

        class _Subscriber(EventSubscriber):
            def on_stop(self) -> None:
                pass

            def on_message(self, message: Message) -> None:
                if isinstance(message, EventMessage) and isinstance(
                    message.event, CommandsAnnouncedEvent
                ):
                    announced.append(message.event)

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
                prompt=PromptTemplate(template="You are a manager."),
                model_cfg=ModelConfig(provider="openai", model="gpt-5-mini"),
            )
            manager_addr = orchestrator.createActor(BaseAgent, config=config)

            time.sleep(0.5)

            for_agent = [e for e in announced if e.agent.name == manager_addr.name]
            assert len(for_agent) == 1, f"expected exactly one event, got {len(for_agent)}"
            event = for_agent[0]
            # TeamTool is auto-injected, so the registry exposes hire/fire commands.
            command_names = {desc.name for desc in event.commands}
            assert "hire_member" in command_names
            assert "fire_member" in command_names
        finally:
            try:
                system.shutdown(timeout=5)
            except Exception:
                pass


# =============================================================================
# _dispatch_command helper (unit level)
# =============================================================================


class TestDispatchCommandHelper:
    """Unit coverage of the _dispatch_command helper return contract."""

    def test_dispatch_command_helper_emits_nothing_extra(self) -> None:
        """_dispatch_command returns False on CommandNotRecognized (fallback signal)."""
        agent = _make_minimal_agent()
        agent._command_registry.dispatch.side_effect = CommandNotRecognized("x")  # type: ignore[attr-defined]

        message = AgentMessage(content="/x", type="request")
        message.sender = _make_mock_sender("@Human")

        handled = agent._dispatch_command(message, _make_mock_sender("@Human"))

        assert handled is False
        agent.send.assert_not_called()

    def test_dispatch_command_helper_sends_and_returns_true(self) -> None:
        """_dispatch_command sends the result and returns True on success."""
        agent = _make_minimal_agent()
        agent._command_registry.dispatch.return_value = "ok"  # type: ignore[attr-defined]

        message = AgentMessage(content="/roster", type="request")
        message.sender = _make_mock_sender("@Human")
        sender = _make_mock_sender("@Human")

        handled = agent._dispatch_command(message, sender)

        assert handled is True
        agent.send.assert_called_once()
        sent_msg = agent.send.call_args[0][1]
        assert sent_msg.content == "ok"
        assert sent_msg.type == "notification"


# =============================================================================
# Operator-action LLM-context injection (Story 8.3)
# =============================================================================


class TestOperatorActionInjection:
    """AC 1-7: post-dispatch human-attributed operator-action context entry."""

    @staticmethod
    def _real_react_agent(agent: BaseAgent) -> tuple[list[Any], list[Any]]:
        """Swap in a real ContextManager + observer; return (events, messages).

        ``events`` collects every event emitted by the context manager (so we can
        assert exactly one ``LlmMessageEvent``); ``messages`` is the live message
        list inside the ContextManager.
        """
        from akgentic.llm.context import ContextManager

        events: list[Any] = []

        class _Observer:
            def notify_event(self, event: object) -> None:
                events.append(event)

        ctx = ContextManager()
        ctx.subscribe(_Observer())

        react = MagicMock()
        react.context = ctx
        agent._react_agent = react  # type: ignore[attr-defined]
        return events, ctx.messages  # initial snapshot (empty)

    def _seed_system_request(self, agent: BaseAgent) -> None:
        """Simulate a completed first run: append a system-bearing ModelRequest
        so subsequent operator actions append directly instead of buffering."""
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

        agent._react_agent.context.add_message(  # type: ignore[attr-defined]
            ModelRequest(
                parts=[
                    SystemPromptPart(content="You are X.", dynamic_ref="agent_backstory"),
                    UserPromptPart(content="prior"),
                ]
            )
        )

    def test_pre_run_operator_action_is_buffered_not_appended(self) -> None:
        """AC-1/2/3: before the first run, the action is BUFFERED (not appended).

        Appending a system-less ModelRequest before the first run would make
        message_history non-empty and suppress pydantic-ai system-prompt
        injection. The entry is buffered for act() to fold into the first run's
        prompt, so no LlmMessageEvent is emitted yet.
        """
        from akgentic.llm import LlmMessageEvent

        agent = _make_minimal_agent()
        events, _ = self._real_react_agent(agent)
        ctx = agent._react_agent.context  # type: ignore[attr-defined]
        agent._command_registry.dispatch.return_value = "hired @Developer42"  # type: ignore[attr-defined]

        message = AgentMessage(content="/hire Developer", type="request")
        message.sender = _make_mock_sender("@Human")

        handled = agent._dispatch_command(message, _make_mock_sender("@Human"))

        assert handled is True
        # Buffered, not appended; nothing in context; no LlmMessageEvent yet.
        assert ctx.messages == []
        assert [e for e in events if isinstance(e, LlmMessageEvent)] == []
        buffered = agent._pending_operator_actions  # type: ignore[attr-defined]
        assert len(buffered) == 1
        # AC-2/3: operator marker + command text + result reproduced verbatim.
        assert buffered[0].startswith("[Operator action]")
        assert "/hire Developer" in buffered[0]
        assert "hired @Developer42" in buffered[0]
        # The result is still sent back to the human.
        agent.send.assert_called_once()

    def test_buffered_entry_is_human_subject_never_agent_first_person(self) -> None:
        """AC-2: 'The human ran' phrasing; no agent first-person framing."""
        agent = _make_minimal_agent()
        self._real_react_agent(agent)
        agent._command_registry.dispatch.return_value = "hired @Dev9"  # type: ignore[attr-defined]

        message = AgentMessage(content="/hire Developer", type="request")
        message.sender = _make_mock_sender("@Human")

        agent._dispatch_command(message, _make_mock_sender("@Human"))

        content = agent._pending_operator_actions[0]  # type: ignore[attr-defined]
        assert "The human ran" in content
        assert "I ran" not in content
        assert "I hired" not in content

    def test_post_run_operator_action_appended_and_emits_event(self) -> None:
        """AC-4: after the first run (system-bearing context), the action is
        appended directly and emits exactly one user-role LlmMessageEvent."""
        from akgentic.llm import LlmMessageEvent
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        agent = _make_minimal_agent()
        events, _ = self._real_react_agent(agent)
        ctx = agent._react_agent.context  # type: ignore[attr-defined]
        self._seed_system_request(agent)
        events.clear()  # drop the seed's event; count only the operator action
        agent._command_registry.dispatch.return_value = "hired @Dev1"  # type: ignore[attr-defined]

        message = AgentMessage(content="/hire Developer", type="request")
        message.sender = _make_mock_sender("@Human")

        agent._dispatch_command(message, _make_mock_sender("@Human"))

        # Nothing buffered; appended as the latest message.
        assert agent._pending_operator_actions == []  # type: ignore[attr-defined]
        appended = ctx.messages[-1]
        assert isinstance(appended, ModelRequest)
        assert isinstance(appended.parts[0], UserPromptPart)
        content = appended.parts[0].content
        assert content.startswith("[Operator action]")
        assert "/hire Developer" in content
        assert "hired @Dev1" in content
        llm_events = [e for e in events if isinstance(e, LlmMessageEvent)]
        assert len(llm_events) == 1

    def test_command_not_recognized_injects_nothing(self) -> None:
        """AC-5: CommandNotRecognized → nothing appended or buffered, returns False."""
        agent = _make_minimal_agent()
        self._real_react_agent(agent)
        ctx = agent._react_agent.context  # type: ignore[attr-defined]
        agent._command_registry.dispatch.side_effect = CommandNotRecognized("frob")  # type: ignore[attr-defined]

        message = AgentMessage(content="/frobnicate", type="request")
        message.sender = _make_mock_sender("@Human")

        handled = agent._dispatch_command(message, _make_mock_sender("@Human"))

        assert handled is False
        assert ctx.messages == []
        assert agent._pending_operator_actions == []  # type: ignore[attr-defined]
        agent.send.assert_not_called()

    def test_post_identification_failure_buffers_one_entry_with_error_result(self) -> None:
        """AC-6: bad-args dispatch returns a usage string → one buffered entry."""
        agent = _make_minimal_agent()
        self._real_react_agent(agent)
        agent._command_registry.dispatch.return_value = "usage: /hire <role>"  # type: ignore[attr-defined]

        message = AgentMessage(content="/hire", type="request")
        message.sender = _make_mock_sender("@Human")

        handled = agent._dispatch_command(message, _make_mock_sender("@Human"))

        assert handled is True
        buffered = agent._pending_operator_actions  # type: ignore[attr-defined]
        assert len(buffered) == 1
        assert buffered[0].startswith("[Operator action]")
        assert "/hire" in buffered[0]
        assert "usage: /hire <role>" in buffered[0]
        agent.send.assert_called_once()  # exactly one send to sender

    def test_at_most_one_entry_and_one_send_per_dispatch(self) -> None:
        """AC-7: one dispatched message → at most one buffered entry and one send."""
        agent = _make_minimal_agent()
        self._real_react_agent(agent)
        agent._command_registry.dispatch.return_value = "ok"  # type: ignore[attr-defined]

        message = AgentMessage(content="/roster", type="request")
        message.sender = _make_mock_sender("@Human")

        agent._dispatch_command(message, _make_mock_sender("@Human"))

        assert len(agent._pending_operator_actions) == 1  # type: ignore[attr-defined]
        agent.send.assert_called_once()

    def test_inject_helper_builds_canonical_entry(self) -> None:
        """_inject_operator_action builds the canonical human-attributed entry
        (buffered before the first run)."""
        agent = _make_minimal_agent()
        self._real_react_agent(agent)

        agent._inject_operator_action("/hire Developer", "hired @Developer42")

        assert agent._pending_operator_actions[0] == (  # type: ignore[attr-defined]
            '[Operator action] The human ran "/hire Developer". \nResult:\nhired @Developer42'
        )

    def test_buffered_actions_flushed_into_first_run_prompt(self) -> None:
        """act() prepends buffered operator actions to the first run's prompt and
        clears the buffer, so message_history stays empty for system-prompt injection."""
        agent = _make_minimal_agent()
        agent._pending_operator_actions = [  # type: ignore[attr-defined]
            '[Operator action] The human ran "/get_planning_task 1".'
            " \nResult:\nNo task with that ID."
        ]
        agent._command_registry.has = MagicMock(return_value=False)  # type: ignore[attr-defined]
        agent._react_agent.run_sync.return_value = StructuredOutput(messages=[])  # type: ignore[attr-defined]

        agent.act("You received a request from @Human:\n\nHello", StructuredOutput)

        prompt = agent._react_agent.run_sync.call_args[0][0]  # type: ignore[attr-defined]
        assert prompt.startswith('[Operator action] The human ran "/get_planning_task 1".')
        assert "You received a request from @Human:\n\nHello" in prompt
        assert agent._pending_operator_actions == []  # type: ignore[attr-defined]


# =============================================================================
# init_llm_context (ADR-009 Layer 3 pass-through)
# =============================================================================


class TestInitLlmContext:
    """Test init_llm_context pass-through to ReactAgent.restore_context."""

    def test_passes_events_through_to_react_agent(self) -> None:
        """AC-1/AC-2: init_llm_context delegates to _react_agent.restore_context
        with the exact same list object (no filtering, no transformation)."""
        agent = _make_minimal_agent()
        events: list[Any] = [MagicMock(), MagicMock(), MagicMock()]

        agent.init_llm_context(events)

        agent._react_agent.restore_context.assert_called_once_with(events)  # type: ignore[attr-defined]
        # Verify identity — same object, not a copy
        passed_arg = agent._react_agent.restore_context.call_args[0][0]  # type: ignore[attr-defined]
        assert passed_arg is events

    def test_passes_empty_list(self) -> None:
        """Edge case: empty event list is forwarded unchanged."""
        agent = _make_minimal_agent()
        events: list[Any] = []

        agent.init_llm_context(events)

        agent._react_agent.restore_context.assert_called_once_with(events)  # type: ignore[attr-defined]
