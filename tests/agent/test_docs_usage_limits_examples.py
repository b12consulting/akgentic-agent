"""Executable copies of the usage-limits examples in README.md.

A migration guide that does not run is worse than none: it is trusted, pasted, and
fails in the reader's code rather than in this suite. Each test below is the README
snippet verbatim plus the assertions that pin what the surrounding prose claims —
so a future rename breaks the example here first.

Deliberately NOT a test that greps the README for strings: that would check
documentation instead of behaviour. These run the code the documentation shows.
"""

import warnings

import pytest
from akgentic.agent.config import AgentConfig
from akgentic.llm import AgentUsageLimits, ModelConfig, RunUsageLimits, UsageLimits


class TestTwoTierExample:
    """The `Usage limits: two tiers` snippet."""

    def test_snippet_constructs_and_keeps_both_budgets(self) -> None:
        config = AgentConfig(
            name="@Manager",
            role="Manager",
            model_cfg=ModelConfig(provider="openai", model="gpt-4.1"),
            run_usage_limits=RunUsageLimits(run_request_limit=50, total_tokens_limit=100_000),
            agent_usage_limits=AgentUsageLimits(
                agent_request_limit=200, total_tokens_limit=2_000_000
            ),
        )

        assert config.run_usage_limits.run_request_limit == 50
        assert config.run_usage_limits.total_tokens_limit == 100_000
        assert config.agent_usage_limits.agent_request_limit == 200
        assert config.agent_usage_limits.total_tokens_limit == 2_000_000

    def test_snippet_emits_no_deprecation_warning(self) -> None:
        """The documented call is the migration target, so it must be silent."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig(
                name="@Manager",
                role="Manager",
                model_cfg=ModelConfig(provider="openai", model="gpt-4.1"),
                run_usage_limits=RunUsageLimits(run_request_limit=50),
                agent_usage_limits=AgentUsageLimits(agent_request_limit=200),
            )
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


class TestDefaultsClaim:
    """`Both defaults are safe to leave alone` — the prose, made checkable."""

    def test_run_tier_default_keeps_the_fifty_request_brake(self) -> None:
        assert AgentConfig().run_usage_limits.run_request_limit == 50

    def test_agent_tier_default_is_all_none(self) -> None:
        """An all-None budget never blocks, which is why the field is never None."""
        agent_tier = AgentConfig().agent_usage_limits
        assert all(value is None for value in agent_tier.model_dump().values())


class TestMigrationExample:
    """The `Migrating from usage_limits` before/after pair."""

    def test_the_before_line_still_works_and_warns(self) -> None:
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=50, total_tokens_limit=100_000)
        with pytest.warns(DeprecationWarning):
            before = AgentConfig(usage_limits=limits)
        assert before.run_usage_limits.run_request_limit == 50
        assert before.run_usage_limits.total_tokens_limit == 100_000

    def test_the_after_line_is_equivalent_and_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            after = AgentConfig(
                run_usage_limits=RunUsageLimits(
                    run_request_limit=50, total_tokens_limit=100_000
                )
            )
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []
        assert after.run_usage_limits.run_request_limit == 50
        assert after.run_usage_limits.total_tokens_limit == 100_000

    def test_both_spellings_together_raise_as_documented(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(
                usage_limits=RunUsageLimits(run_request_limit=50),
                run_usage_limits=RunUsageLimits(run_request_limit=10),
            )

    def test_reading_the_old_name_returns_the_run_tier_as_documented(self) -> None:
        config = AgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=50))
        with pytest.warns(DeprecationWarning):
            assert config.usage_limits is config.run_usage_limits
