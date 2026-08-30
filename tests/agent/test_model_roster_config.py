"""Tests for the model roster on ``AgentConfig``.

``AgentConfig.model_cfg`` accepts a **list** of model configs at the input boundary:
element 0 becomes the active model and the whole list — the active entry included, in
declaration order — becomes ``model_roster``. Everything downstream keeps seeing a
single ``ModelConfig``.

The grammar itself is not written here. ``akgentic-llm`` owns it, and this package
*imports* it. That is the load-bearing property of the whole story: a re-spelled key
grammar would not error, it would simply switch to a model that matches nothing. The
identity and behaviour specs below are the guard against a second copy appearing.
"""

import warnings

import pytest
from akgentic.llm import ModelConfig, RunUsageLimits
from akgentic.llm.config import normalize_model_roster, validate_unique_roster_keys

import akgentic.agent.config as agent_config_module
from akgentic.agent.config import AgentConfig

# Three distinguishable entries. Identical or near-identical models would let a
# truncated or transposed roster pass green.
GPT = ModelConfig(provider="openai", model="gpt-4o")
CLAUDE = ModelConfig(provider="anthropic", model="claude-sonnet-4-5")
GEMINI = ModelConfig(provider="google-gla", model="gemini-2.0-flash")


def _deprecations(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


# =============================================================================
# AC #1 — the single-model path is untouched (NFR1)
# =============================================================================


class TestSingleModelUnchanged:
    """Every existing agent, catalog entry and example must behave exactly as before."""

    def test_single_model_config_is_stored_as_given(self) -> None:
        cfg = AgentConfig(model_cfg=GPT)
        assert cfg.model_cfg == GPT
        assert cfg.model_cfg.provider == "openai"
        assert cfg.model_cfg.model == "gpt-4o"

    def test_single_model_yields_an_empty_roster(self) -> None:
        """Empty roster is what "switching unavailable" looks like — not None."""
        cfg = AgentConfig(model_cfg=GPT)
        assert cfg.model_roster == []

    def test_no_model_cfg_at_all_yields_an_empty_roster(self) -> None:
        cfg = AgentConfig()
        assert cfg.model_roster == []
        assert isinstance(cfg.model_cfg, ModelConfig)

    def test_model_cfg_annotation_stays_a_single_model_config(self) -> None:
        """The union lives at the input boundary only.

        A polymorphic annotation would turn every downstream ``config.model_cfg.<attr>``
        read into an isinstance branch — the exact cost this design refuses to pay.
        """
        assert AgentConfig.model_fields["model_cfg"].annotation is ModelConfig

    def test_model_roster_is_a_real_field_of_model_configs(self) -> None:
        assert "model_roster" in AgentConfig.model_fields
        assert AgentConfig.model_fields["model_roster"].annotation == list[ModelConfig]


# =============================================================================
# AC #2 — a list folds into active + roster
# =============================================================================


class TestListFoldsIntoActivePlusRoster:
    def test_element_zero_becomes_the_active_model(self) -> None:
        cfg = AgentConfig(model_cfg=[GPT, CLAUDE, GEMINI])
        assert cfg.model_cfg == GPT

    def test_the_whole_list_becomes_the_roster_in_declaration_order(self) -> None:
        """Order is the operator's declaration order, and the active entry is included."""
        cfg = AgentConfig(model_cfg=[GPT, CLAUDE, GEMINI])
        assert cfg.model_roster == [GPT, CLAUDE, GEMINI]

    def test_a_one_entry_list_is_a_one_entry_roster(self) -> None:
        """Not folded away to empty: the operator declared a roster, however short."""
        cfg = AgentConfig(model_cfg=[CLAUDE])
        assert cfg.model_cfg == CLAUDE
        assert cfg.model_roster == [CLAUDE]

    def test_stored_active_model_is_a_single_model_config_not_a_list(self) -> None:
        cfg = AgentConfig(model_cfg=[GPT, CLAUDE])
        assert isinstance(cfg.model_cfg, ModelConfig)

    def test_the_declaration_list_is_not_aliased_into_the_config(self) -> None:
        """Mutating the caller's list afterwards must not rewrite the roster."""
        declared = [GPT, CLAUDE]
        cfg = AgentConfig(model_cfg=declared)
        declared.append(GEMINI)
        assert cfg.model_roster == [GPT, CLAUDE]


# =============================================================================
# AC #3 — the inherited rejections fire, naming AgentConfig
# =============================================================================


class TestInheritedRejections:
    """The owner argument is what makes an operator's error message actionable.

    ``AgentConfig`` and ``ReactAgentConfig`` are two different declaration surfaces;
    a message naming the wrong one sends the reader to a file they did not write.
    """

    def test_empty_list_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            AgentConfig(model_cfg=[])
        message = str(excinfo.value)
        # The "empty" fragment keeps this from passing on the generic "not a
        # ModelConfig" type error a list produced before the normalizer existed.
        assert "empty" in message
        assert "ReactAgentConfig" not in message
        assert "AgentConfig" in message

    def test_list_plus_explicit_roster_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            AgentConfig(model_cfg=[GPT, CLAUDE], model_roster=[GPT, CLAUDE])
        message = str(excinfo.value)
        assert "model_roster" in message
        assert "ReactAgentConfig" not in message
        assert "AgentConfig" in message

    def test_an_explicit_roster_alone_is_still_allowed(self) -> None:
        """Only the *combination* is ambiguous; hand-setting the roster is legal."""
        cfg = AgentConfig(model_cfg=GPT, model_roster=[GPT, CLAUDE])
        assert cfg.model_cfg == GPT
        assert cfg.model_roster == [GPT, CLAUDE]


# =============================================================================
# AC #4 — duplicate roster keys are rejected
# =============================================================================


class TestDuplicateRosterKeys:
    """Two entries with one ``provider:model`` key make a switch request ambiguous."""

    def test_two_identical_entries_rejected(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(model_cfg=[GPT, CLAUDE, GPT])

    def test_implicit_and_explicit_provider_spellings_collide(self) -> None:
        """``{"model": "gpt-4o"}`` and ``{"provider": "openai", ...}`` are one model.

        This is why the guard must run *after* field validation: before it, the
        implicit spelling has no ``provider`` at all and the two look distinct.
        """
        with pytest.raises(ValueError):
            AgentConfig(
                model_cfg=[
                    {"model": "gpt-4o"},
                    {"provider": "openai", "model": "gpt-4o"},
                ]
            )

    def test_duplicates_in_a_hand_set_roster_are_rejected_too(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(model_cfg=GPT, model_roster=[GPT, GPT])

    def test_same_model_name_under_two_providers_is_not_a_duplicate(self) -> None:
        """The key is ``provider:model`` — the provider half carries real weight."""
        azure_gpt = ModelConfig(provider="azure", model="gpt-4o")
        cfg = AgentConfig(model_cfg=[GPT, azure_gpt])
        assert cfg.model_roster == [GPT, azure_gpt]

    def test_an_empty_roster_never_trips_the_guard(self) -> None:
        assert AgentConfig(model_cfg=GPT).model_roster == []


# =============================================================================
# AC #5 — one implementation, imported (FR1's whole point)
# =============================================================================


class TestOneImplementationImported:
    """A second copy of the key grammar would not error — it would silently match nothing.

    Two specs, because either alone is passable by a wrong implementation: identity
    can be satisfied by an unused import, and behaviour by a coincidentally-identical
    local copy. Together they pin that the validator *calls the shared symbol*.
    """

    def test_normalizer_is_the_llm_symbol_itself(self) -> None:
        assert agent_config_module.normalize_model_roster is normalize_model_roster

    def test_uniqueness_guard_is_the_llm_symbol_itself(self) -> None:
        assert agent_config_module.validate_unique_roster_keys is validate_unique_roster_keys

    def test_replacing_the_normalizer_changes_what_agent_config_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behavioural half: the before-validator dispatches through the module binding.

        An inlined re-spelling of the grammar would ignore this patch and construct
        happily — which is exactly the failure this spec exists to catch.
        """

        def _sentinel(data: object, owner: str) -> object:
            raise RuntimeError(f"sentinel normalizer reached for {owner}")

        monkeypatch.setattr(agent_config_module, "normalize_model_roster", _sentinel)
        with pytest.raises(RuntimeError, match="sentinel normalizer"):
            AgentConfig(model_cfg=[GPT, CLAUDE])

    def test_replacing_the_uniqueness_guard_changes_what_agent_config_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behavioural half for the after-validator, on a roster that is perfectly valid."""

        def _sentinel(roster: object, owner: str) -> None:
            raise RuntimeError(f"sentinel guard reached for {owner}")

        monkeypatch.setattr(agent_config_module, "validate_unique_roster_keys", _sentinel)
        with pytest.raises(RuntimeError, match="sentinel guard"):
            AgentConfig(model_cfg=[GPT, CLAUDE])

    def test_the_guard_is_skipped_entirely_on_an_empty_roster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-model agents must not pay for a rule that cannot apply to them."""

        def _sentinel(roster: object, owner: str) -> None:
            raise RuntimeError("guard must not run on an empty roster")

        monkeypatch.setattr(agent_config_module, "validate_unique_roster_keys", _sentinel)
        assert AgentConfig(model_cfg=GPT).model_roster == []


# =============================================================================
# AC #7 — the two before-validators coexist in either key order (NFR2)
# =============================================================================


class TestValidatorsCoexist:
    """Each before-validator owns one key and returns its input unchanged otherwise.

    That is what makes their evaluation order irrelevant — and the pair is the only
    place in this config where two before-validators see the same mapping.
    """

    def test_deprecated_keyword_first_then_roster(self) -> None:
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(
                usage_limits=RunUsageLimits(run_request_limit=11),
                model_cfg=[GPT, CLAUDE, GEMINI],
            )
        assert cfg.run_usage_limits.run_request_limit == 11
        assert cfg.model_cfg == GPT
        assert cfg.model_roster == [GPT, CLAUDE, GEMINI]

    def test_roster_first_then_deprecated_keyword(self) -> None:
        with pytest.warns(DeprecationWarning):
            cfg = AgentConfig(
                model_cfg=[GPT, CLAUDE, GEMINI],
                usage_limits=RunUsageLimits(run_request_limit=11),
            )
        assert cfg.run_usage_limits.run_request_limit == 11
        assert cfg.model_cfg == GPT
        assert cfg.model_roster == [GPT, CLAUDE, GEMINI]

    def test_the_deprecation_warning_still_fires_exactly_once(self) -> None:
        """The roster fold must neither swallow the warning nor duplicate it."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig(
                model_cfg=[GPT, CLAUDE],
                usage_limits=RunUsageLimits(run_request_limit=11),
            )
        assert len(_deprecations(caught)) == 1

    def test_a_roster_alone_emits_no_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AgentConfig(model_cfg=[GPT, CLAUDE])
        assert _deprecations(caught) == []

    def test_the_usage_limits_rejection_survives_a_roster(self) -> None:
        """Neither validator may mask the other's error."""
        with pytest.raises(ValueError):
            AgentConfig(
                model_cfg=[GPT, CLAUDE],
                usage_limits=RunUsageLimits(run_request_limit=10),
                run_usage_limits=RunUsageLimits(run_request_limit=20),
            )


# =============================================================================
# AC #8 — a list of dicts normalizes identically (FR5, reachable half)
# =============================================================================


class TestListOfDicts:
    """The shape a catalog hands over.

    The resolver inlines every ``{"__ref__": ...}`` marker before the owning model
    validates, so a ``model_cfg:`` node listing three refs reaches ``AgentConfig`` as a
    list of plain dicts. The end-to-end fixture belongs to ``akgentic-catalog``; this
    is the half reachable from inside the module boundary.
    """

    def test_dicts_produce_the_same_active_model_and_roster_as_instances(self) -> None:
        cfg = AgentConfig.model_validate(
            {
                "name": "@Manager",
                "role": "Manager",
                "model_cfg": [
                    {"provider": "openai", "model": "gpt-4o"},
                    {"provider": "anthropic", "model": "claude-sonnet-4-5"},
                    {"provider": "google-gla", "model": "gemini-2.0-flash"},
                ],
            }
        )
        assert cfg.model_cfg == GPT
        assert cfg.model_roster == [GPT, CLAUDE, GEMINI]

    def test_entries_are_validated_into_model_configs(self) -> None:
        cfg = AgentConfig.model_validate(
            {"model_cfg": [{"provider": "openai", "model": "gpt-4o"}]}
        )
        assert all(isinstance(entry, ModelConfig) for entry in cfg.model_roster)

    def test_an_empty_dict_list_is_rejected_the_same_way(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig.model_validate({"model_cfg": []})


# =============================================================================
# AC #9 — the round trip preserves the roster
# =============================================================================


class TestRoundTrip:
    """The normalizer's documented trap, and the only spec that reaches it.

    On ``model_validate(model_dump())`` the active model arrives as a single dict while
    ``model_roster`` is *already populated*. A normalizer that fell through to an
    ``else`` clearing the roster would destroy it here — on a path no construction test
    exercises, and silently.
    """

    def test_roster_survives_a_dump_and_reload(self) -> None:
        cfg = AgentConfig(model_cfg=[GPT, CLAUDE, GEMINI])
        restored = AgentConfig.model_validate(cfg.model_dump())
        assert restored.model_cfg == GPT
        assert restored.model_roster == [GPT, CLAUDE, GEMINI]

    def test_the_dump_carries_both_the_active_model_and_the_roster(self) -> None:
        data = AgentConfig(model_cfg=[GPT, CLAUDE]).model_dump()
        assert data["model_cfg"]["model"] == "gpt-4o"
        assert [entry["model"] for entry in data["model_roster"]] == [
            "gpt-4o",
            "claude-sonnet-4-5",
        ]

    def test_a_single_model_config_round_trips_to_an_empty_roster(self) -> None:
        cfg = AgentConfig(model_cfg=GPT)
        restored = AgentConfig.model_validate(cfg.model_dump())
        assert restored.model_cfg == GPT
        assert restored.model_roster == []

    def test_the_round_trip_is_stable_across_two_generations(self) -> None:
        """A roster that survives one reload but not the next is still lost data."""
        first = AgentConfig(model_cfg=[GPT, CLAUDE, GEMINI])
        second = AgentConfig.model_validate(first.model_dump())
        third = AgentConfig.model_validate(second.model_dump())
        assert third.model_roster == [GPT, CLAUDE, GEMINI]
        assert third.model_cfg == GPT
