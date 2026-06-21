"""Behavior tests for ``BaseAgent.on_stop()`` releasing the ReactAgent http pool.

Covers Story 10-1 (Epic 10): ``on_stop()`` must run ``ReactAgent.aclose()`` on the
actor's persistent event loop (``self._event_loop``) BEFORE ``super().on_stop()``
closes that loop, must never propagate an exception from ``aclose()``, and must
skip ``aclose()`` when the loop is absent/closed — always delegating to
``super().on_stop()``.

Test-boundary note (per story): the FR1/FR2 guarantees are white-box ordering
invariants of ``on_stop()`` itself — "the loop is still open at ``aclose()`` call
time" and "``aclose()`` runs before the base teardown closes the loop" cannot be
observed deterministically from outside a Pykka ``ActorSystem`` stop. So these
tests call ``on_stop()`` directly on a bare ``BaseAgent`` (the established
``object.__new__`` pattern used by the other unit tests in this package),
wiring a REAL ``asyncio`` loop so the close path is exercised for real and the
spy on ``aclose`` can capture the loop's open/closed state at call time. No real
network egress occurs — ``_react_agent`` is a stub with an async ``aclose``.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

from akgentic.agent.agent import BaseAgent
from akgentic.core import Akgent

# =============================================================================
# HELPERS
# =============================================================================


def _make_agent(
    loop: asyncio.AbstractEventLoop | None,
    aclose: Any,
) -> BaseAgent:
    """Bare ``BaseAgent`` (no Pykka actor system) wired for ``on_stop()`` tests.

    Sets only the attributes ``on_stop()`` (and the real ``super().on_stop()``)
    read: ``_event_loop``, ``_react_agent`` (with the given ``aclose``),
    ``config.name`` (for log/telemetry formatting) and ``_orchestrator = None``
    so the base teardown's ``_notify_orchestrator`` is a no-op.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    agent._event_loop = loop  # type: ignore[assignment]

    react_agent = MagicMock()
    react_agent.aclose = aclose
    agent._react_agent = react_agent  # type: ignore[attr-defined]

    config = MagicMock()
    config.name = "@TestAgent"
    agent.config = config  # type: ignore[assignment]

    agent._orchestrator = None  # type: ignore[assignment]
    return agent


# =============================================================================
# AC #1 — FR1: aclose() runs on the persistent loop, BEFORE the loop is closed
# =============================================================================


def test_aclose_runs_on_persistent_loop_before_loop_closed() -> None:
    """FR1: ``on_stop()`` awaits ``aclose()`` on ``self._event_loop`` while it is
    still open, then the real ``super().on_stop()`` closes that loop."""
    loop = asyncio.new_event_loop()

    observed: dict[str, Any] = {}

    async def spy_aclose() -> None:
        # Captured at the moment aclose() is awaited: the actor loop is the one
        # running us and it is NOT yet closed (base teardown has not run).
        observed["closed_at_call"] = loop.is_closed()
        observed["running_loop"] = asyncio.get_running_loop()
        observed["calls"] = observed.get("calls", 0) + 1

    agent = _make_agent(loop, spy_aclose)

    agent.on_stop()

    # aclose() was awaited exactly once...
    assert observed["calls"] == 1
    # ...on the actor's persistent loop...
    assert observed["running_loop"] is loop
    # ...while that loop was still open (BEFORE base teardown closed it)...
    assert observed["closed_at_call"] is False
    # ...and the real super().on_stop() has since closed the loop.
    assert loop.is_closed() is True


def test_aclose_called_before_super_on_stop() -> None:
    """FR1 ordering: ``aclose()`` is awaited strictly BEFORE ``super().on_stop()``.

    Records the call order; if the order were inverted, ``aclose()`` would run on
    an already-closed loop and leak the pool.
    """
    loop = asyncio.new_event_loop()
    order: list[str] = []

    async def spy_aclose() -> None:
        order.append("aclose")

    agent = _make_agent(loop, spy_aclose)

    def fake_super_on_stop(_self: Akgent[Any, Any]) -> None:
        order.append("super")
        loop.close()

    with patch.object(Akgent, "on_stop", fake_super_on_stop):
        agent.on_stop()

    assert order == ["aclose", "super"]
    assert loop.is_closed() is True


# =============================================================================
# AC #2 — FR2: teardown never raises when aclose() fails; super() still runs
# =============================================================================


def test_on_stop_swallows_aclose_exception_and_calls_super() -> None:
    """FR2: when ``aclose()`` raises, ``on_stop()`` does not propagate the error,
    logs it, and still completes ``super().on_stop()``."""
    loop = asyncio.new_event_loop()
    try:
        boom = RuntimeError("pool release failed")

        async def failing_aclose() -> None:
            raise boom

        agent = _make_agent(loop, failing_aclose)

        with patch.object(Akgent, "on_stop") as super_on_stop:
            # Must not raise despite aclose() blowing up.
            agent.on_stop()

        super_on_stop.assert_called_once()
    finally:
        loop.close()


def test_on_stop_logs_when_aclose_raises(caplog: Any) -> None:
    """FR2: the swallowed ``aclose()`` failure is logged (not silent)."""
    loop = asyncio.new_event_loop()
    try:

        async def failing_aclose() -> None:
            raise RuntimeError("pool release failed")

        agent = _make_agent(loop, failing_aclose)

        with patch.object(Akgent, "on_stop"):
            with caplog.at_level("WARNING", logger="akgentic.agent.agent"):
                agent.on_stop()

        assert any("aclose" in record.getMessage() for record in caplog.records), (
            "expected a log record mentioning the failed aclose()"
        )
    finally:
        loop.close()


# =============================================================================
# AC #3 — FR2 guard: missing/closed loop skips aclose(), still calls super()
# =============================================================================


def test_on_stop_skips_aclose_when_loop_is_none() -> None:
    """FR2 guard: ``self._event_loop is None`` → ``aclose()`` is never awaited,
    but ``super().on_stop()`` is still called."""
    aclose = MagicMock(name="aclose")  # would fail the test if awaited (not a coroutine)
    agent = _make_agent(None, aclose)

    with patch.object(Akgent, "on_stop") as super_on_stop:
        agent.on_stop()

    aclose.assert_not_called()
    super_on_stop.assert_called_once()


def test_on_stop_skips_aclose_when_loop_closed() -> None:
    """FR2 guard: an already-closed ``self._event_loop`` → ``aclose()`` is never
    awaited, but ``super().on_stop()`` is still called."""
    loop = asyncio.new_event_loop()
    loop.close()
    assert loop.is_closed() is True

    aclose = MagicMock(name="aclose")
    agent = _make_agent(loop, aclose)

    with patch.object(Akgent, "on_stop") as super_on_stop:
        agent.on_stop()

    aclose.assert_not_called()
    super_on_stop.assert_called_once()
