"""Tests for BaseAgent media expansion wiring (Story 3-3).

Tests the _expand_media_refs_command wiring in on_start() and the
media expansion pre-step in act() without requiring a live actor system
or real LLM calls.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import BinaryContent

from akgentic.agent.agent import BaseAgent
from akgentic.agent.config import AgentConfig
from akgentic.agent.messages import AgentMessage
from akgentic.tool.core import ToolFactory
from akgentic.tool.team import TeamTool
from akgentic.tool.workspace import ExpandMediaRefs
from akgentic.tool.workspace.readers import MediaContent


# =============================================================================
# HELPERS
# =============================================================================


def _make_mock_sender(name: str = "@Human") -> MagicMock:
    """Return a mock ActorAddress with a .name attribute."""
    addr = MagicMock()
    addr.name = name
    return addr


def _make_mock_message(sender_name: str = "@Human") -> MagicMock:
    """Return a mock AgentMessage with sender set."""
    msg = MagicMock(spec=AgentMessage)
    msg.sender = _make_mock_sender(sender_name)
    msg.type = "request"
    return msg


def _make_minimal_agent() -> BaseAgent:
    """Construct a BaseAgent-like object without the Pykka actor system.

    Creates a bare BaseAgent instance, bypassing the actor __init__, and
    sets up the minimal attributes needed to call act() in unit tests:
    - _expand_media_refs_command: None (default — no WorkspaceReadTool)
    - _react_agent: MagicMock
    - _current_message: mock AgentMessage
    - get_team / get_available_roles: return empty lists

    All test-specific overrides (e.g. setting _expand_media_refs_command)
    are applied AFTER calling this helper.
    """
    agent: BaseAgent = object.__new__(BaseAgent)

    # Minimal attributes expected by act()
    agent._expand_media_refs_command = None  # type: ignore[attr-defined]
    agent._react_agent = MagicMock()  # type: ignore[attr-defined]
    agent._current_message = _make_mock_message()  # type: ignore[attr-defined]

    # get_team() and get_available_roles() are called inside act()
    agent.get_team = MagicMock(return_value=[])  # type: ignore[method-assign]
    agent.get_available_roles = MagicMock(return_value=[])  # type: ignore[method-assign]

    return agent


# =============================================================================
# AC-1 / AC-2: _expand_media_refs_command wired in on_start()
# =============================================================================


class TestExpandMediaRefsCommandWiring:
    """AC-1, AC-2: _expand_media_refs_command extracted from ToolFactory commands."""

    def test_command_present_when_workspace_read_tool_in_cards(self) -> None:
        """AC-1: WorkspaceReadTool in tool cards → _expand_media_refs_command is not None."""
        from akgentic.tool.workspace import WorkspaceReadTool

        mock_cmd = MagicMock()

        # Build a mock WorkspaceReadTool that exposes ExpandMediaRefs in get_commands()
        mock_workspace_tool = MagicMock(spec=WorkspaceReadTool)
        mock_workspace_tool.get_tools.return_value = []
        mock_workspace_tool.get_toolsets.return_value = []
        mock_workspace_tool.get_system_prompts.return_value = []
        mock_workspace_tool.get_commands.return_value = {ExpandMediaRefs: mock_cmd}

        # Build ToolFactory with TeamTool + mocked WorkspaceReadTool
        tool_factory = ToolFactory(
            tool_cards=[TeamTool(), mock_workspace_tool],
            observer=MagicMock(),
            retry_exception=None,
        )
        commands = tool_factory.get_commands()

        assert commands.get(ExpandMediaRefs) is not None

    def test_command_absent_when_no_workspace_read_tool(self) -> None:
        """AC-2: No WorkspaceReadTool → ExpandMediaRefs not in commands."""
        tool_factory = ToolFactory(
            tool_cards=[TeamTool()],
            observer=MagicMock(),
            retry_exception=None,
        )
        commands = tool_factory.get_commands()

        assert commands.get(ExpandMediaRefs) is None

    def test_expand_media_refs_command_is_none_on_minimal_agent(self) -> None:
        """AC-2: _make_minimal_agent() produces agent with command = None."""
        agent = _make_minimal_agent()
        assert agent._expand_media_refs_command is None  # type: ignore[attr-defined]


# =============================================================================
# AC-3 to AC-6: act() media expansion pre-step
# =============================================================================


class TestBaseAgentMediaExpansion:
    """AC-3 to AC-8: act() expansion pre-step — mock-based, no LLM required."""

    # ------------------------------------------------------------------
    # AC-3: image token → BinaryContent in run_sync call
    # ------------------------------------------------------------------

    def test_image_token_converted_to_binary_content(self) -> None:
        """AC-3: MediaContent in expansion result → BinaryContent passed to run_sync."""
        agent = _make_minimal_agent()

        mock_cmd = MagicMock(
            return_value=[
                "describe ",
                MediaContent(data=b"png-bytes", media_type="image/png"),
                " please",
            ]
        )
        agent._expand_media_refs_command = mock_cmd  # type: ignore[attr-defined]

        # Capture what run_sync receives
        captured_prompts: list[Any] = []

        def capture_run_sync(prompt: Any, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "mocked response"

        agent._react_agent.run_sync.side_effect = capture_run_sync  # type: ignore[attr-defined]

        agent.act("describe !!photo.png please", output_type=str)

        assert len(captured_prompts) == 1
        result_prompt = captured_prompts[0]
        assert isinstance(result_prompt, list)
        assert result_prompt[0] == "describe "
        assert isinstance(result_prompt[1], BinaryContent)
        assert result_prompt[1].data == b"png-bytes"
        assert result_prompt[1].media_type == "image/png"
        assert result_prompt[2] == " please"

    # ------------------------------------------------------------------
    # AC-4: pure text → str passed unchanged
    # ------------------------------------------------------------------

    def test_pure_text_passed_as_str_unchanged(self) -> None:
        """AC-4: No MediaContent in expansion result → original str passed to run_sync."""
        agent = _make_minimal_agent()

        mock_cmd = MagicMock(return_value=["no special tokens"])
        agent._expand_media_refs_command = mock_cmd  # type: ignore[attr-defined]

        captured_prompts: list[Any] = []

        def capture_run_sync(prompt: Any, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "mocked response"

        agent._react_agent.run_sync.side_effect = capture_run_sync  # type: ignore[attr-defined]

        agent.act("no special tokens", output_type=str)

        assert len(captured_prompts) == 1
        result_prompt = captured_prompts[0]
        # Must be the original string, NOT a list
        assert isinstance(result_prompt, str)
        assert result_prompt == "no special tokens"

    # ------------------------------------------------------------------
    # AC-5: document token → hint string list, no BinaryContent
    # ------------------------------------------------------------------

    def test_document_token_passes_str_list_unchanged(self) -> None:
        """AC-5: Doc hint strings (no MediaContent) → str list passed to run_sync."""
        agent = _make_minimal_agent()

        hint_list = ["check ", "!!report.pdf[=> Use workspace_read tool]"]
        mock_cmd = MagicMock(return_value=hint_list)
        agent._expand_media_refs_command = mock_cmd  # type: ignore[attr-defined]

        captured_prompts: list[Any] = []

        def capture_run_sync(prompt: Any, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "mocked response"

        agent._react_agent.run_sync.side_effect = capture_run_sync  # type: ignore[attr-defined]

        agent.act("check !!report.pdf", output_type=str)

        assert len(captured_prompts) == 1
        result_prompt = captured_prompts[0]
        # Pure string list — no BinaryContent construction
        assert isinstance(result_prompt, str)
        assert result_prompt == "check !!report.pdf"

    # ------------------------------------------------------------------
    # AC-6: _expand_media_refs_command is None → user_content passed as-is
    # ------------------------------------------------------------------

    def test_command_none_passes_user_content_as_is(self) -> None:
        """AC-6: _expand_media_refs_command is None → user_content str unchanged."""
        agent = _make_minimal_agent()
        agent._expand_media_refs_command = None  # type: ignore[attr-defined]

        captured_prompts: list[Any] = []

        def capture_run_sync(prompt: Any, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "mocked response"

        agent._react_agent.run_sync.side_effect = capture_run_sync  # type: ignore[attr-defined]

        agent.act("look at !!photo.png", output_type=str)

        assert len(captured_prompts) == 1
        result_prompt = captured_prompts[0]
        assert isinstance(result_prompt, str)
        assert result_prompt == "look at !!photo.png"

    # ------------------------------------------------------------------
    # AC-7: act() signature unchanged
    # ------------------------------------------------------------------

    def test_act_signature_unchanged(self) -> None:
        """AC-7: act(user_content: str, output_type: type[T]) -> T signature intact."""
        import inspect

        sig = inspect.signature(BaseAgent.act)
        params = list(sig.parameters.keys())
        assert params == ["self", "user_content", "output_type"]
        assert sig.parameters["user_content"].annotation is str

    # ------------------------------------------------------------------
    # AC-3 branch: both True/False paths of any(isinstance(p, MediaContent)...)
    # ------------------------------------------------------------------

    def test_any_media_content_true_branch(self) -> None:
        """Branch coverage: any(isinstance(p, MediaContent)) == True → list built."""
        agent = _make_minimal_agent()

        mixed = [
            "prefix ",
            MediaContent(data=b"data", media_type="image/jpeg"),
        ]
        mock_cmd = MagicMock(return_value=mixed)
        agent._expand_media_refs_command = mock_cmd  # type: ignore[attr-defined]

        captured_prompts: list[Any] = []

        def capture_run_sync(prompt: Any, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "ok"

        agent._react_agent.run_sync.side_effect = capture_run_sync  # type: ignore[attr-defined]

        agent.act("prefix !!img.jpg", output_type=str)

        result_prompt = captured_prompts[0]
        assert isinstance(result_prompt, list)
        assert any(isinstance(p, BinaryContent) for p in result_prompt)

    def test_any_media_content_false_branch(self) -> None:
        """Branch coverage: any(isinstance(p, MediaContent)) == False → str returned."""
        agent = _make_minimal_agent()

        mock_cmd = MagicMock(return_value=["only strings", " here"])
        agent._expand_media_refs_command = mock_cmd  # type: ignore[attr-defined]

        captured_prompts: list[Any] = []

        def capture_run_sync(prompt: Any, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "ok"

        agent._react_agent.run_sync.side_effect = capture_run_sync  # type: ignore[attr-defined]

        agent.act("only strings here", output_type=str)

        result_prompt = captured_prompts[0]
        assert isinstance(result_prompt, str)
