"""``resolve_recipient`` turns a hire that cannot happen into a skipped delivery.

Routing runs after the LLM run has ended, so an exception there cannot be retried;
it would only leave the actor's handler and take the turn's other messages with it.
"""

import logging
from unittest.mock import MagicMock

import pytest
from akgentic.core import ActorAddress
from pydantic_ai import ModelRetry

from akgentic.agent.utils import resolve_recipient


def _agent(hire: object) -> MagicMock:
    agent = MagicMock()
    agent.hire_member.side_effect = hire
    return agent


class TestAHireThatCannotHappenIsSkipped:
    def test_a_refused_role_resolves_to_none(self) -> None:
        agent = _agent(ModelRetry("Hire error - role 'Manager' cannot be hired."))

        assert resolve_recipient(agent, "Manager") is None

    def test_a_missing_hire_command_resolves_to_none(self) -> None:
        agent = _agent(RuntimeError("hire_member command not available"))

        assert resolve_recipient(agent, "Manager") is None

    def test_the_skip_is_logged_with_the_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = _agent(RuntimeError("hire_member command not available"))

        with caplog.at_level(logging.WARNING, logger="akgentic.agent.utils"):
            resolve_recipient(agent, "Manager")

        assert "'Manager'" in caplog.text
        assert "hire_member command not available" in caplog.text

    def test_any_other_failure_still_propagates(self) -> None:
        """A configuration error is a bug to surface, not a recipient to skip."""
        agent = _agent(ValueError("agent class for role Manager is a string"))

        with pytest.raises(ValueError, match="is a string"):
            resolve_recipient(agent, "Manager")


class TestTheOtherPathsAreUnchanged:
    def test_a_hireable_role_is_hired(self) -> None:
        hired = MagicMock(spec=ActorAddress)
        agent = _agent(None)
        agent.hire_member.side_effect = None
        agent.hire_member.return_value = hired

        assert resolve_recipient(agent, "Developer") is hired

    def test_an_at_name_never_hires(self) -> None:
        agent = _agent(AssertionError("must not hire"))
        agent.get_team_member.return_value = None

        assert resolve_recipient(agent, "@Ghost") is None
        agent.hire_member.assert_not_called()
