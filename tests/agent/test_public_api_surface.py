"""The public import surface of ``akgentic.agent``, frozen (Epic 20, FR8 AC 7).

Moving ``MailboxCapability`` and ``RunInterruptedError`` into
``akgentic.agent.capabilities`` is a *relocation*, not an API change. Four
things have to stay true, and each is a way the move or the rename could have
gone wrong silently:

- ``__all__`` is exactly what it was — the move adds no export and drops none.
  ``MailboxCapability`` was never in it and is not added by the rename.
- Every name in ``__all__`` actually resolves off the package.
- The classes are **one** object each, however you reach them. Seven existing
  test modules import them from ``akgentic.agent.agent``; the public docs show
  ``from akgentic.agent import RunInterruptedError``; the canonical home is
  ``akgentic.agent.capabilities``. A duplicate definition on any of those paths
  would break ``except RunInterruptedError`` for whoever imported the other one.
- The old name ``MailboxCancelCapability`` resolves from **nowhere**. It was
  never released — it shipped on the epic branch and was renamed before the
  umbrella PR merged — so there is deliberately no deprecation alias, and a
  leftover one would be a shadow the rename was supposed to remove.
"""

from __future__ import annotations

import akgentic.agent
import akgentic.agent.agent as agent_module
import akgentic.agent.capabilities as capabilities_module
import akgentic.agent.capabilities.mailbox_capability as mailbox_capability_module
from akgentic.agent.capabilities.mailbox_capability import (
    MailboxCapability,
    RunInterruptedError,
)

EXPECTED_PUBLIC_API = {
    "__version__",
    "AgentConfig",
    "HumanProxy",
    "BaseAgent",
    "RunInterruptedError",
    "AgentMessage",
}


class TestPublicSurface:
    def test_all_is_exactly_the_frozen_set(self) -> None:
        assert set(akgentic.agent.__all__) == EXPECTED_PUBLIC_API

    def test_every_exported_name_resolves(self) -> None:
        for name in akgentic.agent.__all__:
            assert getattr(akgentic.agent, name, None) is not None, name


class TestOneClassNotTwo:
    """The same object down every import path that ships or is already in use."""

    def test_run_interrupted_error_is_the_capabilities_class(self) -> None:
        assert akgentic.agent.RunInterruptedError is RunInterruptedError
        assert capabilities_module.RunInterruptedError is RunInterruptedError
        assert agent_module.RunInterruptedError is RunInterruptedError

    def test_mailbox_capability_is_the_capabilities_class(self) -> None:
        assert capabilities_module.MailboxCapability is MailboxCapability
        assert agent_module.MailboxCapability is MailboxCapability

    def test_capabilities_package_re_exports_all_four_symbols(self) -> None:
        assert set(capabilities_module.__all__) == {
            "MailboxCapability",
            "RunInterruptedError",
            "is_cancel",
            "render_arrival_notice",
        }
        for name in capabilities_module.__all__:
            assert getattr(capabilities_module, name, None) is not None, name


class TestOldNameIsGone:
    """``MailboxCancelCapability`` resolves from nowhere — no deprecation alias.

    An import-surface fact, asserted with ``getattr``: reading the source for
    the absence of a string would pass on a file that still defines the alias
    under a different spelling, and fail on a comment that merely names it.
    """

    def test_old_name_is_absent_from_every_module_that_could_carry_it(self) -> None:
        for module in (
            akgentic.agent,
            agent_module,
            capabilities_module,
            mailbox_capability_module,
        ):
            assert getattr(module, "MailboxCancelCapability", None) is None, module.__name__

    def test_old_name_is_absent_from_the_capabilities_all(self) -> None:
        assert "MailboxCancelCapability" not in capabilities_module.__all__
