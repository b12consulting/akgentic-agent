"""Tests for the env-gated LLM-agent selection in ``BaseAgent._build_react_agent``.

When ``AKGENTIC_MOCK_SCENARIO`` names a scenario YAML, ``on_start`` swaps the
real ``ReactAgent`` for the token-free ``MockReactAgent`` (akgentic-llm loadtest),
carrying the scenario path in a config copy's ``model_cfg.model`` field so the
agent's own ``self.config`` is left intact. Unset -> the real ``ReactAgent``.

The mock branch injects a fake ``akgentic.llm.loadtest`` module so the test is
self-contained and does not require the optional ``loadtest`` extra to be
installed.
"""

import sys
import types

import akgentic.agent.agent as agent_module
import pytest
from akgentic.agent.agent import BaseAgent
from akgentic.llm import ModelConfig, ReactAgentConfig

ENV_VAR = "AKGENTIC_MOCK_SCENARIO"


def _make_agent() -> BaseAgent:
    """Bare BaseAgent (no Pykka actor system) for ``_build_react_agent`` tests.

    No ``_event_loop`` is wired — ``_build_react_agent`` no longer reads it.
    """
    agent: BaseAgent = object.__new__(BaseAgent)
    return agent


def test_real_react_agent_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env -> real ReactAgent built with the config passed straight through."""
    monkeypatch.delenv(ENV_VAR, raising=False)

    captured: dict[str, object] = {}

    class _FakeReactAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(agent_module, "ReactAgent", _FakeReactAgent)

    agent = _make_agent()
    config = ReactAgentConfig()
    result = agent._build_react_agent(config, [], [], [])

    assert isinstance(result, _FakeReactAgent)
    assert captured["config"] is config  # not copied/mutated
    assert captured["observer"] is agent
    assert captured["deps_type"] is BaseAgent


def test_mock_react_agent_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env set -> MockReactAgent built; scenario path smuggled into a config COPY."""
    monkeypatch.setenv(ENV_VAR, "/tmp/sandpile-research.yaml")

    captured: dict[str, object] = {}

    class _FakeMockReactAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_loadtest = types.ModuleType("akgentic.llm.loadtest")
    fake_loadtest.MockReactAgent = _FakeMockReactAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "akgentic.llm.loadtest", fake_loadtest)

    agent = _make_agent()
    config = ReactAgentConfig()  # model_cfg.model defaults to "gpt-5.2"
    result = agent._build_react_agent(config, [], [], [])

    assert isinstance(result, _FakeMockReactAgent)
    assert captured["observer"] is agent
    # Scenario path lands in the COPY's model field...
    mock_cfg = captured["config"]
    assert isinstance(mock_cfg, ReactAgentConfig)
    assert mock_cfg.model_cfg.model == "/tmp/sandpile-research.yaml"
    # ...while the original config is untouched.
    assert config.model_cfg.model == "gpt-5.2"


def test_blank_env_falls_back_to_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty AKGENTIC_MOCK_SCENARIO is treated as unset (real ReactAgent)."""
    monkeypatch.setenv(ENV_VAR, "")

    built: list[str] = []

    class _FakeReactAgent:
        def __init__(self, **kwargs: object) -> None:
            built.append("real")

    monkeypatch.setattr(agent_module, "ReactAgent", _FakeReactAgent)

    agent = _make_agent()
    result = agent._build_react_agent(ReactAgentConfig(), [], [], [])

    assert isinstance(result, _FakeReactAgent)
    assert built == ["real"]


def test_real_path_omits_event_loop_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC #5 (real path): ReactAgent is built with NO ``event_loop`` keyword."""
    monkeypatch.delenv(ENV_VAR, raising=False)

    captured: dict[str, object] = {}

    class _FakeReactAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(agent_module, "ReactAgent", _FakeReactAgent)

    agent = _make_agent()
    agent._build_react_agent(ReactAgentConfig(), [], [], [])

    assert "event_loop" not in captured


def test_mock_path_omits_event_loop_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC #5 (mock path): MockReactAgent is built with NO ``event_loop`` keyword."""
    monkeypatch.setenv(ENV_VAR, "/tmp/sandpile-research.yaml")

    captured: dict[str, object] = {}

    class _FakeMockReactAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_loadtest = types.ModuleType("akgentic.llm.loadtest")
    fake_loadtest.MockReactAgent = _FakeMockReactAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "akgentic.llm.loadtest", fake_loadtest)

    agent = _make_agent()
    agent._build_react_agent(ReactAgentConfig(), [], [], [])

    assert "event_loop" not in captured


GPT = ModelConfig(provider="openai", model="gpt-4o")
CLAUDE = ModelConfig(provider="anthropic", model="claude-sonnet-4-5")


class TestMockScenarioCopyStaysConsistent:
    """The scenario copy replaces the active model, so it must drop the roster with it.

    ``model_copy(update=...)`` skips validation, so rewriting ``model_cfg.model`` to the
    scenario path while leaving the roster alone produces a config whose active model is
    absent from its own roster — internally inconsistent, and raising only later, on the
    next re-validation, where nothing points back at this line. A mock serves exactly one
    scenario file, so a roster it could switch away from is meaningless anyway.
    """

    @staticmethod
    def _build_mock_config(
        monkeypatch: pytest.MonkeyPatch, config: ReactAgentConfig
    ) -> ReactAgentConfig:
        monkeypatch.setenv(ENV_VAR, "/tmp/sandpile-research.yaml")

        captured: dict[str, object] = {}

        class _FakeMockReactAgent:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        fake_loadtest = types.ModuleType("akgentic.llm.loadtest")
        fake_loadtest.MockReactAgent = _FakeMockReactAgent  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "akgentic.llm.loadtest", fake_loadtest)

        _make_agent()._build_react_agent(config, [], [], [])

        mock_cfg = captured["config"]
        assert isinstance(mock_cfg, ReactAgentConfig)
        return mock_cfg

    def test_copy_survives_revalidation_of_its_own_dump(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cfg = self._build_mock_config(
            monkeypatch, ReactAgentConfig(model_cfg=[GPT, CLAUDE])
        )
        ReactAgentConfig.model_validate(mock_cfg.model_dump())

    def test_copy_carries_the_scenario_path_and_no_roster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cfg = self._build_mock_config(
            monkeypatch, ReactAgentConfig(model_cfg=[GPT, CLAUDE])
        )
        assert mock_cfg.model_cfg.model == "/tmp/sandpile-research.yaml"
        assert mock_cfg.model_roster == []

    def test_the_original_config_keeps_its_roster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``self.config`` is left untouched — the copy is the only thing rewritten."""
        config = ReactAgentConfig(model_cfg=[GPT, CLAUDE])
        self._build_mock_config(monkeypatch, config)
        assert config.model_roster == [GPT, CLAUDE]
        assert config.model_cfg == GPT

    def test_a_single_model_mock_config_is_unaffected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cfg = self._build_mock_config(monkeypatch, ReactAgentConfig(model_cfg=GPT))
        assert mock_cfg.model_roster == []
        ReactAgentConfig.model_validate(mock_cfg.model_dump())
