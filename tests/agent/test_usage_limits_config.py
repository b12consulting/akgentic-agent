"""Tests for AgentConfig's two usage-limit tiers and the pre-split deprecation shim.

`akgentic-llm` split one usage budget into a run tier (`RunUsageLimits`, bounding a
single `run()`) and an agent tier (`AgentUsageLimits`, bounding the agent's whole
lifetime). `AgentConfig` now carries both under their new names, and keeps the
pre-split `usage_limits` spelling alive for one release cycle.

The load-bearing assertion throughout is that a deprecated value **arrives at its
destination**. A shim that accepts the old keyword and drops the value is worse than
no shim: the agent silently runs on a budget nobody chose, and nothing fails until
the bill does.
"""

import warnings

import pytest
from akgentic.llm import (
    AgentUsageLimits,
    CompactionConfig,
    ModelConfig,
    RuntimeConfig,
    RunUsageLimits,
    UsageLimits,
)
from akgentic.llm.prompts import PromptTemplate

from akgentic.agent.config import AgentConfig

# The release that deletes the shim. Every warning and docstring must name it, so
# removing the shim is a scheduled task rather than an archaeological dig.
REMOVAL_RELEASE = "akgentic-agent 2.0.0"


def _deprecations(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


# =============================================================================
# FR1 — the two tiers as real fields (AC #1, #2)
# =============================================================================


class TestRunTierField:
    """AC #1: usage_limits became run_usage_limits, typed as the run tier."""

    def test_default_is_run_usage_limits_with_the_safety_brake(self) -> None:
        """The default keeps the run tier's 50-request brake."""
        cfg = AgentConfig()
        assert isinstance(cfg.run_usage_limits, RunUsageLimits)
        assert cfg.run_usage_limits.run_request_limit == 50

    def test_explicit_run_tier_roundtrips_unchanged(self) -> None:
        """An explicit RunUsageLimits is stored as given."""
        limits = RunUsageLimits(run_request_limit=7, total_tokens_limit=1234)
        cfg = AgentConfig(run_usage_limits=limits)
        assert cfg.run_usage_limits is limits
        assert cfg.run_usage_limits.run_request_limit == 7
        assert cfg.run_usage_limits.total_tokens_limit == 1234

    def test_run_usage_limits_is_a_real_field(self) -> None:
        """The run tier is storage, not a computed view."""
        assert "run_usage_limits" in AgentConfig.model_fields
        assert AgentConfig.model_fields["run_usage_limits"].annotation is RunUsageLimits

    def test_dump_emits_the_new_field_name(self) -> None:
        """The wire format follows the rename."""
        cfg = AgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        data = cfg.model_dump()
        assert data["run_usage_limits"]["run_request_limit"] == 10
        assert "usage_limits" not in data


class TestAgentTierField:
    """AC #2: the agent-lifetime tier is reachable, and never None."""

    def test_default_is_an_all_none_budget(self) -> None:
        """Default-constructed means "unlimited" — every field is None."""
        cfg = AgentConfig()
        assert isinstance(cfg.agent_usage_limits, AgentUsageLimits)
        assert cfg.agent_usage_limits.agent_request_limit is None
        assert all(value is None for value in cfg.agent_usage_limits.model_dump().values())

    def test_field_is_not_optional(self) -> None:
        """An all-None budget already expresses "unlimited"; None would be a second way.

        Making the field ``AgentUsageLimits | None`` would force every reader to unwrap
        an optional to reach a state the model can hold on its own.
        """
        field = AgentConfig.model_fields["agent_usage_limits"]
        assert field.annotation is AgentUsageLimits
        assert field.is_required() is False
        assert isinstance(AgentConfig().agent_usage_limits, AgentUsageLimits)

    def test_explicit_agent_tier_roundtrips_unchanged(self) -> None:
        """An explicit AgentUsageLimits is stored as given."""
        limits = AgentUsageLimits(agent_request_limit=100, total_tokens_limit=1_000_000)
        cfg = AgentConfig(agent_usage_limits=limits)
        assert cfg.agent_usage_limits is limits
        assert cfg.agent_usage_limits.agent_request_limit == 100
        assert cfg.agent_usage_limits.total_tokens_limit == 1_000_000

    def test_agent_tier_is_independent_of_the_run_tier(self) -> None:
        """Setting one tier does not disturb the other."""
        cfg = AgentConfig(agent_usage_limits=AgentUsageLimits(agent_request_limit=3))
        assert cfg.agent_usage_limits.agent_request_limit == 3
        assert cfg.run_usage_limits.run_request_limit == 50


# =============================================================================
# FR2 — the deprecation shim (AC #3, #4, #5, #6, #7)
# =============================================================================


class TestDeprecatedKeyword:
    """AC #3: AgentConfig(usage_limits=...) survives as a deprecated keyword."""

    def test_keyword_accepted_and_value_reaches_run_usage_limits(self) -> None:
        """The destination is the assertion — acceptance alone proves nothing."""
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        assert cfg.run_usage_limits.run_request_limit == 10

    def test_construction_emits_exactly_one_warning(self) -> None:
        """One deprecated construction, one warning — not zero, not a storm."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        assert len(_deprecations(caught)) == 1

    def test_warning_names_replacement_and_release(self) -> None:
        """The warning must tell the caller what to write instead, and by when."""
        with pytest.warns(DeprecationWarning) as record:
            AgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        message = str(record[0].message)
        assert "run_usage_limits" in message
        assert REMOVAL_RELEASE in message

    def test_warning_points_at_the_caller(self) -> None:
        """stacklevel must blame the caller's line, not config.py.

        A deprecation warning whose traceback lands inside the library tells the
        user nothing about which of their own lines to fix.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        assert _deprecations(caught)[0].filename == __file__

    def test_deprecated_class_instance_stored_as_is(self) -> None:
        """The pre-split UsageLimits alias satisfies the run-tier annotation."""
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=10)
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(usage_limits=limits)
        assert cfg.run_usage_limits is limits
        assert cfg.run_usage_limits.run_request_limit == 10

    def test_agent_tier_still_settable_alongside_the_deprecated_keyword(self) -> None:
        """A half-migrated caller can reach the new tier without migrating both."""
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(
                usage_limits=RunUsageLimits(run_request_limit=10),
                agent_usage_limits=AgentUsageLimits(agent_request_limit=3),
            )
        assert cfg.run_usage_limits.run_request_limit == 10
        assert cfg.agent_usage_limits.agent_request_limit == 3


class TestDeprecatedReadAccessor:
    """AC #4: config.usage_limits reads back the run tier."""

    def test_read_returns_the_run_tier_itself(self) -> None:
        """A view over the one real field, not a copy."""
        cfg = AgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with pytest.warns(DeprecationWarning) as record:
            value = cfg.usage_limits
        assert value is cfg.run_usage_limits
        message = str(record[0].message)
        assert "run_usage_limits" in message
        assert REMOVAL_RELEASE in message

    def test_read_reflects_whichever_spelling_set_it(self) -> None:
        """The read path follows the field."""
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(usage_limits=RunUsageLimits(run_request_limit=7))
        with pytest.warns(DeprecationWarning):
            assert cfg.usage_limits.run_request_limit == 7

    def test_usage_limits_is_not_a_field(self) -> None:
        """A second storage slot would reintroduce the split-brain the rename removed."""
        assert "usage_limits" not in AgentConfig.model_fields

    def test_assignment_through_the_deprecated_name_is_not_shimmed(self) -> None:
        """A write fails loudly instead of parking the value on a shadow attribute."""
        cfg = AgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with pytest.raises((AttributeError, ValueError)):
            cfg.usage_limits = RunUsageLimits(run_request_limit=99)  # type: ignore[misc]
        assert cfg.run_usage_limits.run_request_limit == 10

    def test_accessor_docstring_names_the_removal_release(self) -> None:
        """The accessor schedules its own deletion."""
        doc = AgentConfig.usage_limits.__doc__
        assert doc is not None
        assert REMOVAL_RELEASE in doc


class TestBothNamesRejected:
    """AC #5: ambiguity is an error, not a silent winner decided by argument order."""

    def test_rejected_old_first(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(
                usage_limits=RunUsageLimits(run_request_limit=10),
                run_usage_limits=RunUsageLimits(run_request_limit=20),
            )

    def test_rejected_new_first(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(
                run_usage_limits=RunUsageLimits(run_request_limit=20),
                usage_limits=RunUsageLimits(run_request_limit=10),
            )

    def test_rejected_when_values_are_equal(self) -> None:
        """Equal values do not resolve the ambiguity in either name's favour."""
        limits = RunUsageLimits(run_request_limit=10)
        with pytest.raises(ValueError):
            AgentConfig(usage_limits=limits, run_usage_limits=limits)

    def test_raises_before_the_deprecation_warning(self) -> None:
        """The ValueError must survive -W error::DeprecationWarning.

        Downstream projects routinely run with that filter to hunt deprecated usage.
        If the warning fires first it becomes the raised exception, and the caller
        never sees the message telling them what is actually wrong with their call.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(ValueError):
                AgentConfig(
                    usage_limits=RunUsageLimits(run_request_limit=10),
                    run_usage_limits=RunUsageLimits(run_request_limit=20),
                )


class TestMappingValues:
    """AC #6: a dict under the old keyword keeps its budget too.

    ``AgentConfig`` is built from persisted and declarative sources, so the value
    arriving as a mapping rather than a model instance is a real path. Validated
    straight as ``RunUsageLimits``, the pre-split ``request_limit`` key is unknown
    and Pydantic drops it in silence — the caller is told the shim handled their
    input, and it did not.
    """

    def test_inner_pre_split_spelling_is_folded(self) -> None:
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(usage_limits={"request_limit": 10})
        assert cfg.run_usage_limits.run_request_limit == 10

    def test_inner_pre_split_spelling_keeps_sibling_keys(self) -> None:
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(usage_limits={"request_limit": 10, "total_tokens_limit": 5000})
        assert cfg.run_usage_limits.run_request_limit == 10
        assert cfg.run_usage_limits.total_tokens_limit == 5000

    def test_inner_new_spelling_still_lands(self) -> None:
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(usage_limits={"run_request_limit": 10})
        assert cfg.run_usage_limits.run_request_limit == 10

    def test_inner_both_spellings_rejected(self) -> None:
        """Ambiguity one level down is still ambiguity."""
        with pytest.raises(ValueError):
            AgentConfig(usage_limits={"request_limit": 10, "run_request_limit": 20})


class TestShimShape:
    """AC #7: the shim intercepts construction, which a property alone cannot do."""

    def test_deprecated_keyword_is_not_stored_as_an_extra(self) -> None:
        """The before-validator removes the keyword; nothing shadows the real field."""
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        assert "usage_limits" not in cfg.model_dump()
        assert cfg.__pydantic_extra__ in (None, {})

    def test_validate_from_instance_skips_the_shim_entirely(self) -> None:
        """Revalidating an existing config neither re-warns nor re-maps."""
        cfg = AgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = AgentConfig.model_validate(cfg)
        assert restored.run_usage_limits.run_request_limit == 10
        assert _deprecations(caught) == []


# =============================================================================
# AC #8 — the migration target is silent
# =============================================================================


class TestNewSpellingsAreWarningFree:
    """A migration target that warns is noise."""

    def test_bare_construction_silent(self) -> None:
        """Defaults alone must not warn — that would fire on every agent."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig()
        assert _deprecations(caught) == []

    def test_both_new_keywords_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig(
                run_usage_limits=RunUsageLimits(run_request_limit=10),
                agent_usage_limits=AgentUsageLimits(agent_request_limit=3),
            )
        assert _deprecations(caught) == []

    def test_reading_new_fields_silent(self) -> None:
        """Only the deprecated accessor warns; the real fields are silent."""
        cfg = AgentConfig(
            run_usage_limits=RunUsageLimits(run_request_limit=10),
            agent_usage_limits=AgentUsageLimits(agent_request_limit=3),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert cfg.run_usage_limits.run_request_limit == 10
            assert cfg.agent_usage_limits.agent_request_limit == 3
        assert _deprecations(caught) == []

    def test_full_new_style_config_silent(self) -> None:
        """The whole migrated call shape, end to end."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig(
                name="@Manager",
                role="Manager",
                prompt=PromptTemplate(template="You are a manager."),
                model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
                runtime_cfg=RuntimeConfig(),
                run_usage_limits=RunUsageLimits(run_request_limit=7),
                agent_usage_limits=AgentUsageLimits(agent_request_limit=100),
                compaction_cfg=CompactionConfig(),
            )
        assert _deprecations(caught) == []


# =============================================================================
# AC #9 — vacuity guard
# =============================================================================


class TestWarningsAreObservable:
    """The shim tests must not pass on a silenced warning.

    Deliberately does NOT call simplefilter("always"). It asserts the warning is
    visible under the suite's *ambient* filter state, so adding a global
    ``ignore::DeprecationWarning`` to the pytest configuration turns this test red
    instead of quietly hollowing out every warning assertion above.
    """

    def test_construction_warning_observable_under_default_filters(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            AgentConfig(usage_limits=RunUsageLimits(run_request_limit=10))
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_read_warning_observable_under_default_filters(self) -> None:
        cfg = AgentConfig(run_usage_limits=RunUsageLimits(run_request_limit=10))
        with warnings.catch_warnings(record=True) as caught:
            _ = cfg.usage_limits
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# =============================================================================
# The real consumer's call shape, reproduced verbatim
# =============================================================================


class TestRealConsumerCallShape:
    """The shape a pre-split caller writes today, so the shim is pinned to a real
    migration path rather than a hypothetical one."""

    def test_pre_split_call_still_works_and_keeps_its_budget(self) -> None:
        with pytest.warns(DeprecationWarning):
            limits = UsageLimits(request_limit=50, total_tokens_limit=100000)
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(
                prompt=PromptTemplate(template="You are a helpful software developer."),
                model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
                runtime_cfg=RuntimeConfig(retries=5),
                usage_limits=limits,
            )
        assert cfg.run_usage_limits.run_request_limit == 50
        assert cfg.run_usage_limits.total_tokens_limit == 100000
        # The agent tier the old caller never named still exists, unlimited.
        assert cfg.agent_usage_limits.agent_request_limit is None


# =============================================================================
# max_help_requests is gone for good
# =============================================================================


class TestMaxHelpRequestsRemoved:
    """`max_help_requests` advertised a delegation-depth cap that no code path ever
    applied: it was declared, documented, copied into a private attribute, and read
    by nobody. It is deleted rather than enforced or annotated as inert.

    The guard is on `model_fields`, not on constructor behaviour, and that is the
    point. `AgentConfig` inherits Pydantic's `extra="ignore"`, so
    `AgentConfig(max_help_requests=5)` neither raises before the removal nor after
    it — a construction-based test would pass green with the field reinstated.
    Only the field registry distinguishes the two states.
    """

    def test_field_is_not_declared(self) -> None:
        assert "max_help_requests" not in AgentConfig.model_fields

    def test_instance_carries_no_such_attribute(self) -> None:
        assert not hasattr(AgentConfig(), "max_help_requests")

    def test_passing_it_is_ignored_rather_than_stored(self) -> None:
        """Unchanged observable behaviour: the kwarg was inert before and is inert now.

        Deliberately NOT `extra="forbid"` — that would turn every existing caller
        into a hard ValidationError, and it is a decision about the whole config
        surface rather than this one field.
        """
        cfg = AgentConfig(max_help_requests=5)
        assert not hasattr(cfg, "max_help_requests")
        assert "max_help_requests" not in cfg.model_dump()
