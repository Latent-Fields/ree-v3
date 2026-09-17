"""Contracts for environment identity in the Experiment Pack `environment` block.

Surfaces under test:
  (1) ree_core.environment.causal_grid_world.CausalGridWorld.environment_identity()
      -- the producer of the seven-field block, and its hash contract.
  (2) experiments.pack_writer.DEFAULT_ENVIRONMENT -- the honest all-"unknown"
      fallback, and specifically that it ASSERTS NOTHING.
  (3) experiments.pack_writer.environment_for() -- the sanctioned thread-through,
      and that it RAISES rather than falling back.

WHY THIS GATE EXISTS. The `environment` block exists to answer "did these two runs
execute the same environment?". Until 2026-09-17 nothing produced it: both writers
fell back to a hardcoded literal asserting env_id "ree.causal_grid_world_v3" with
all four content hashes "unknown". Measured across REE_assembly/evidence/experiments
on 2026-09-17: 1752 of the 1826 packs carrying an environment block asserted exactly
that, including runs whose config was in fact CausalGridWorldV2 -- and there is no
CausalGridWorldV3, in this module or anywhere else. The V3-EXQ-1036 failure autopsy
surfaced it (evidence/planning/failure_autopsy_V3-EXQ-1036_2026-09-17.md).

An asserted-but-wrong env_id is worse than an absent one: it reads as recorded
provenance and no consumer can tell it from a real one. The two invariants that
keep it from coming back are therefore (a) the default asserts nothing, and (b)
the only way to get a real block is to ask the environment itself.

ASCII-only output (CLAUDE.md).
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ree_core.environment.causal_grid_world import (  # noqa: E402
    ENV_ID,
    ENV_VERSION,
    CausalGridWorld,
    CausalGridWorldV2,
)
from experiments.pack_writer import (  # noqa: E402
    DEFAULT_ENVIRONMENT,
    REQUIRED_ENVIRONMENT_FIELDS,
    environment_for,
    write_flat_manifest,
)

# The exact literal both writers emitted before 2026-09-17. Named here so the
# regression is pinned by VALUE, not by a paraphrase that could drift.
FABRICATED_DEFAULT = {
    "env_id": "ree.causal_grid_world_v3",
    "env_version": "3.0.0",
    "tier": "causal_grid_world_v3",
}


class TestHonestDefault:
    def test_default_environment_asserts_nothing(self):
        """Every field of the no-driver-input default is "unknown".

        This is the load-bearing invariant. A default that names a specific
        environment is a fabrication for every run that did not use it, and
        because the default applies when the driver supplied NOTHING, that is
        every run whose driver did not opt in.
        """
        for field in REQUIRED_ENVIRONMENT_FIELDS:
            assert DEFAULT_ENVIRONMENT[field] == "unknown", (
                "DEFAULT_ENVIRONMENT[%r] is %r, expected 'unknown'. A default "
                "that asserts a value is a fabrication for every run that did "
                "not have it -- see the note at DEFAULT_ENVIRONMENT."
                % (field, DEFAULT_ENVIRONMENT[field])
            )

    def test_default_covers_exactly_the_required_fields(self):
        assert set(DEFAULT_ENVIRONMENT) == set(REQUIRED_ENVIRONMENT_FIELDS)

    def test_the_fabricated_literal_is_gone(self):
        for field, bad in FABRICATED_DEFAULT.items():
            assert DEFAULT_ENVIRONMENT[field] != bad

    def test_no_causal_grid_world_v3_class_exists(self):
        """The fabricated env_id named a class that has never existed.

        Pinned so that if a CausalGridWorldV3 is ever genuinely added, this test
        fails and forces a deliberate decision about the 1752 legacy packs that
        already claim to be it -- rather than those packs silently becoming
        "correct" by coincidence.
        """
        import ree_core.environment.causal_grid_world as mod
        assert not hasattr(mod, "CausalGridWorldV3")


class TestEnvironmentIdentity:
    def test_returns_every_required_field_non_blank(self):
        identity = CausalGridWorldV2(seed=1).environment_identity()
        for field in REQUIRED_ENVIRONMENT_FIELDS:
            assert str(identity.get(field, "")).strip(), field

    def test_env_id_names_the_real_class(self):
        identity = CausalGridWorld(seed=1).environment_identity()
        assert identity["env_id"] == ENV_ID
        assert identity["env_version"] == ENV_VERSION
        assert identity["env_id"] != FABRICATED_DEFAULT["env_id"]

    def test_no_field_is_unknown(self):
        """An environment that can identify itself never reports "unknown".

        Otherwise the honest-absence signal and a real-but-incomplete block
        would be indistinguishable to the one consumer that reads them
        (REE_assembly scripts/generate_experiment_profile.py).
        """
        identity = CausalGridWorld(seed=3).environment_identity()
        assert "unknown" not in identity.values()

    def test_config_hash_is_deterministic(self):
        a = CausalGridWorld(seed=1, size=9).environment_identity()
        b = CausalGridWorld(seed=1, size=9).environment_identity()
        assert a == b

    def test_config_hash_ignores_seed(self):
        """Two seeds of one configuration are the SAME environment.

        config_hash answers "is this the same environment configuration?" --
        which is what makes it usable for deciding whether two arms or two seeds
        are comparable. A per-seed hash would answer "is this the same run?",
        which run_id already answers.
        """
        a = CausalGridWorld(seed=1).environment_identity()
        b = CausalGridWorld(seed=99).environment_identity()
        assert a["config_hash"] == b["config_hash"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"size": 12},
            {"num_hazards": 6},
            {"num_resources": 1},
            {"use_proxy_fields": True},
            {"contamination_spread": 0.25},
            {"max_episode_steps": 123},
        ],
    )
    def test_config_hash_separates_different_configurations(self, kwargs):
        """EQUAL HASH => SAME CONFIGURATION. A collision here would license
        comparing runs that did not execute the same environment."""
        base = CausalGridWorld(seed=1).environment_identity()
        other = CausalGridWorld(seed=1, **kwargs).environment_identity()
        assert base["config_hash"] != other["config_hash"], kwargs

    def test_v2_alias_is_distinguished_from_base(self):
        """CausalGridWorldV2 is an alias factory (use_proxy_fields=True), not a
        separate class -- so the difference must show up in config_hash and tier,
        which is the whole reason env_id does not encode the mode."""
        base = CausalGridWorld(seed=1).environment_identity()
        v2 = CausalGridWorldV2(seed=1).environment_identity()
        assert base["env_id"] == v2["env_id"]
        assert base["config_hash"] != v2["config_hash"]
        assert base["tier"] == "base"
        assert v2["tier"] == "proxy_fields"

    def test_content_hashes_are_stable_across_instances(self):
        """The three source hashes describe CODE, so they must not vary with
        instance configuration -- otherwise they could not be used to ask
        whether two differently-configured runs ran the same substrate."""
        a = CausalGridWorld(seed=1, size=9).environment_identity()
        b = CausalGridWorldV2(seed=7, size=14, num_hazards=6).environment_identity()
        for field in ("dynamics_hash", "reward_hash", "observation_hash"):
            assert a[field] == b[field], field

    def test_content_hashes_are_distinct_from_each_other(self):
        identity = CausalGridWorld(seed=1).environment_identity()
        hashes = {identity[f] for f in
                  ("dynamics_hash", "reward_hash", "observation_hash", "config_hash")}
        assert len(hashes) == 4

    def test_content_hashes_track_the_source(self):
        """A change to a hashed member must move its hash.

        Exercised through the group hasher rather than by editing the module, so
        the test does not depend on a specific method's current text.
        """
        import ree_core.environment.causal_grid_world as mod
        before = mod._group_hash(CausalGridWorld, mod._DYNAMICS_MEMBERS)
        after = mod._group_hash(CausalGridWorld, mod._DYNAMICS_MEMBERS + ("reset",))
        assert before != after

    def test_absent_member_is_hashed_not_skipped(self):
        """Removing a method is a code change and must move the hash. A skip
        would make deletion invisible."""
        import ree_core.environment.causal_grid_world as mod
        assert mod._member_source(CausalGridWorld, "no_such_method_xyz") == "<absent>"

    def test_step_is_in_both_dynamics_and_reward(self):
        """Deliberate, documented over-inclusion: harm/benefit/energy accrue
        INLINE in step(), so reward is not separable from dynamics in this
        module. Over-inclusion preserves the EQUAL HASH => SAME CODE contract;
        dropping step() from reward would break it."""
        import ree_core.environment.causal_grid_world as mod
        assert "step" in mod._DYNAMICS_MEMBERS
        assert "step" in mod._REWARD_MEMBERS

    def test_every_hashed_member_name_resolves(self):
        """A typo in a member list would silently hash "<absent>" forever,
        quietly narrowing what the hash covers."""
        import ree_core.environment.causal_grid_world as mod
        groups = (mod._DYNAMICS_MEMBERS, mod._REWARD_MEMBERS, mod._OBSERVATION_MEMBERS)
        unresolved = [
            name
            for group in groups
            for name in group
            if not hasattr(CausalGridWorld, name)
        ]
        assert unresolved == [], (
            "hashed member name(s) do not resolve on CausalGridWorld: %s" % unresolved
        )


class TestStableRepr:
    """config_hash digests constructor arguments via _stable_repr.

    Constructor arguments here are not all scalars -- 5 are tuples and several
    take lists or dicts from drivers (resource_type_names,
    resource_type_distribution, resource_type_benefit_amplitudes ...). repr()
    alone is not a safe identity for dicts and sets, whose iteration order is not
    guaranteed across constructions: two equal configurations could hash
    differently, which would report a spurious environment difference between
    arms that are in fact identical.
    """

    @staticmethod
    def _repr(value):
        import ree_core.environment.causal_grid_world as mod
        return mod._stable_repr(value)

    def test_dict_key_order_does_not_matter(self):
        assert self._repr({"b": 2, "a": 1}) == self._repr({"a": 1, "b": 2})

    def test_set_order_does_not_matter(self):
        assert self._repr({"b", "a"}) == self._repr({"a", "b"})

    def test_nested_dict_order_does_not_matter(self):
        assert self._repr({"k": {"b": 2, "a": 1}}) == self._repr({"k": {"a": 1, "b": 2}})

    def test_list_and_tuple_are_distinguished(self):
        assert self._repr([1, 2]) != self._repr((1, 2))

    def test_value_change_is_visible(self):
        assert self._repr({"a": 1}) != self._repr({"a": 2})

    def test_container_valued_params_hash_stably(self):
        """End-to-end through a real construction, on the container-valued
        parameters drivers actually pass."""
        kwargs = dict(
            resource_type_names=("a", "b"),
            resource_type_distribution=[0.5, 0.5],
            resource_type_benefit_amplitudes=(1.0, 2.0),
            resource_type_drive_axes=("x", "y"),
            resource_type_benefit_curves=("linear", "linear"),
        )
        first = CausalGridWorld(seed=1, **kwargs).environment_identity()["config_hash"]
        second = CausalGridWorld(seed=2, **kwargs).environment_identity()["config_hash"]
        assert first == second
        assert first != CausalGridWorld(seed=1).environment_identity()["config_hash"]

        changed = dict(kwargs, resource_type_distribution=[0.6, 0.4])
        assert CausalGridWorld(
            seed=1, **changed).environment_identity()["config_hash"] != first


class TestSubclassIdentity:
    """Several drivers subclass CausalGridWorld (SmallViewEnv, which narrows the
    observation window). A subclass that reported the base env_id would assert
    that a narrowed-view run and a default run used the same environment -- the
    same class of claim this whole file exists to stop, one level down."""

    def test_subclass_does_not_claim_the_base_env_id(self):
        class NarrowedEnv(CausalGridWorld):
            def _place_random_landmarks(self, *a, **k):
                return super()._place_random_landmarks(*a, **k)

        base = CausalGridWorld(seed=1).environment_identity()
        sub = NarrowedEnv(seed=1).environment_identity()
        assert sub["env_id"] != base["env_id"]
        assert sub["env_id"].startswith(ENV_ID)
        assert "NarrowedEnv" in sub["env_id"]

    def test_overridden_method_moves_the_content_hash(self):
        """The hashes resolve type(self), not CausalGridWorld, so an override is
        visible. Without this a subclass could differ in code and still report
        the base's hashes."""
        class NarrowedEnv(CausalGridWorld):
            def _place_random_landmarks(self, *a, **k):
                return super()._place_random_landmarks(*a, **k)

        base = CausalGridWorld(seed=1).environment_identity()
        sub = NarrowedEnv(seed=1).environment_identity()
        assert sub["dynamics_hash"] != base["dynamics_hash"]

    def test_subclass_skipping_super_init_raises_clearly(self):
        """An AttributeError on a private attribute would be an unhelpful way to
        learn that the configuration was never captured."""
        class NoSuper(CausalGridWorld):
            def __init__(self):  # deliberately does not call super()
                pass

        with pytest.raises(AttributeError, match="_ctor_params"):
            NoSuper().environment_identity()


class TestEnvironmentFor:
    def test_returns_the_environments_own_identity(self):
        env = CausalGridWorldV2(seed=5, size=9)
        assert environment_for(env) == dict(env.environment_identity())

    def test_result_is_a_complete_block(self):
        block = environment_for(CausalGridWorld(seed=5))
        assert set(block) >= set(REQUIRED_ENVIRONMENT_FIELDS)
        assert "unknown" not in block.values()

    def test_raises_rather_than_falling_back(self):
        """The regression this whole file guards is a SILENT FALLBACK. A caller
        with an unidentifiable environment must be told, not handed a guess."""
        with pytest.raises(TypeError):
            environment_for(object())

    def test_raises_on_incomplete_identity(self):
        class Partial:
            def environment_identity(self):
                return {"env_id": "ree.demo"}

        with pytest.raises(ValueError):
            environment_for(Partial())

    def test_raises_on_non_mapping_identity(self):
        class Wrong:
            def environment_identity(self):
                return "ree.demo"

        with pytest.raises(TypeError):
            environment_for(Wrong())


class TestFlatManifestChokepoint:
    """write_flat_manifest(env=...) is the harness half.

    The flat manifest -- not emit_pack -- is what ~all packs are actually built
    from: the REE_assembly converter projects the flat and cannot construct the
    environment to ask it anything. So the flat is where identity has to be
    recorded, and it has to be recorded at ONE chokepoint rather than by each
    driver hand-rolling a block.
    """

    @staticmethod
    def _write(tmp_path, monkeypatch, **kwargs):
        # The always-core provenance gate is a separate contract (it fails on a
        # synthetic manifest that never went through stamp_recording_core) and is
        # not what this class is testing.
        monkeypatch.setenv("REE_ALLOW_INCOMPLETE_PROVENANCE", "1")
        path = write_flat_manifest(
            {"run_id": "demo_env_probe_v3", "status": "PASS"},
            tmp_path, stamp=False, **kwargs)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_env_stamps_the_identity_block(self, tmp_path, monkeypatch):
        env = CausalGridWorldV2(seed=1, size=9)
        written = self._write(tmp_path, monkeypatch, env=env)
        assert written["environment"] == dict(env.environment_identity())

    def test_no_env_records_no_environment(self, tmp_path, monkeypatch):
        """Honest absence. The converter supplies the all-"unknown" block, so the
        flat does not need to carry a placeholder -- and a placeholder in the flat
        would be indistinguishable from a driver that tried and failed."""
        written = self._write(tmp_path, monkeypatch)
        assert "environment" not in written

    def test_explicit_manifest_value_is_not_overridden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REE_ALLOW_INCOMPLETE_PROVENANCE", "1")
        deliberate = {f: "deliberate" for f in REQUIRED_ENVIRONMENT_FIELDS}
        path = write_flat_manifest(
            {"run_id": "demo_env_probe_v3", "status": "PASS",
             "environment": dict(deliberate)},
            tmp_path, stamp=False, env=CausalGridWorldV2(seed=1))
        assert json.loads(Path(path).read_text(encoding="utf-8"))["environment"] == deliberate

    def test_unidentifiable_env_raises_rather_than_degrading(self, tmp_path, monkeypatch):
        """A driver that passed env= asserted "I know my environment". Swallowing
        the failure would silently put the run back to an unrecorded environment,
        which is the bug this whole file guards -- so this path deliberately does
        NOT share the always-core stamp's except-and-continue behaviour."""
        with pytest.raises(TypeError):
            self._write(tmp_path, monkeypatch, env=object())
