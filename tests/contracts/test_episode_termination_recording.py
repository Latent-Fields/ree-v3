"""Contracts for the SD-094 episode-termination recording block.

Substrate: experiments/_lib/episode_termination.py, wired into
experiments/_lib/manifest_core.stamp_recording_core and
experiments/pack_writer.write_flat_manifest.

The gap this closes: a flat manifest records a CONFIGURED step budget and an
aggregate outcome and nothing in between, so a run whose episodes died on step
19 of a configured 400 is indistinguishable on the page from one that spent the
whole budget -- same config block, same metric shape, merely worse numbers.
Confirmed against V3-EXQ-884 (32/19/90 of 400 steps on seeds 42/43/44, all via
self-inflicted contamination); establishing that took a live re-run inside a
failure autopsy, for a fact the run knew at every step.

The contracts that matter here:
  (a) the block is OMITTED, never zero-filled, when a driver collected nothing --
      "unmeasured" and "ran the full budget" must not look alike, which is the
      exact confusion the block exists to remove;
  (b) full_budget_frac actually separates a full-budget run from a truncated one,
      and causes says WHY;
  (c) it is NOT in ALWAYS_CORE_KEYS -- the pre-2026-08-03 corpus cannot carry it
      and making it core would turn every legacy manifest into a WARN;
  (d) stamping never raises, whatever it is handed: provenance must not be able
      to crash an experiment at manifest-write time, when the compute is spent;
  (e) it round-trips from the env's own info fields, which is the path a real
      driver uses.
"""

from __future__ import annotations

import numpy as np

from experiments._lib import manifest_core
from experiments._lib.episode_termination import (
    MANIFEST_KEY,
    EpisodeTerminationAccumulator,
    stamp_episode_termination,
    stats_from_episodes,
)
from ree_core.environment.causal_grid_world import CausalGridWorld as Env


# --- (a) absence means unmeasured --------------------------------------------


def test_block_is_omitted_when_nothing_was_collected():
    m = {}
    stamp_episode_termination(m, EpisodeTerminationAccumulator(steps_configured=400))
    assert MANIFEST_KEY not in m, "an empty accumulator must stamp nothing"


def test_block_is_omitted_when_no_source_is_passed():
    m = {}
    stamp_episode_termination(m, None)
    assert MANIFEST_KEY not in m


def test_stamp_recording_core_omits_the_block_by_default():
    """The overwhelming majority of existing drivers pass nothing. They must keep
    producing a manifest with no episode_termination key at all."""
    m = manifest_core.stamp_recording_core({}, config={"n_steps": 400}, seeds=[1])
    assert MANIFEST_KEY not in m


# --- (b) the measurement itself ----------------------------------------------


def test_full_budget_run_reports_frac_one():
    acc = EpisodeTerminationAccumulator(steps_configured=400)
    for _ in range(3):
        acc.record(400, "")
    block = acc.stats()
    assert block["n_episodes"] == 3
    assert block["full_budget_frac"] == 1.0
    assert block["truncated_frac"] == 0.0
    assert block["steps_min"] == block["steps_max"] == 400
    # "" is a driver-ended episode, not a cause -- it must not be counted as one.
    assert block["causes"] == {"driver_budget": 3}


def test_exq884_shaped_run_reports_the_truncation_and_its_cause():
    """The reported failure, as it would now appear in a manifest."""
    acc = EpisodeTerminationAccumulator(steps_configured=400)
    for steps in (32, 19, 90):
        acc.record(steps, "health_depleted")
    block = acc.stats()
    assert block["full_budget_frac"] == 0.0
    assert block["truncated_frac"] == 1.0
    assert block["steps_configured"] == 400
    assert block["steps_min"] == 19
    assert block["steps_max"] == 90
    assert block["causes"] == {"health_depleted": 3}


def test_mixed_run_reports_a_partial_fraction_and_both_causes():
    acc = EpisodeTerminationAccumulator(steps_configured=100)
    acc.record(100, "step_limit")
    acc.record(100, "step_limit")
    acc.record(12, "health_depleted")
    acc.record(40, "health_depleted")
    block = acc.stats()
    assert block["full_budget_frac"] == 0.5
    assert block["truncated_frac"] == 0.5
    assert block["causes"] == {"health_depleted": 2, "step_limit": 2}
    assert block["steps_mean"] == 63.0


def test_no_declared_budget_reports_null_rather_than_a_fabricated_denominator():
    """Using max(steps) as the budget would make every run look complete."""
    block = stats_from_episodes([(10, "health_depleted"), (40, "step_limit")])
    assert block["steps_configured"] is None
    assert block["full_budget_frac"] is None
    assert block["truncated_frac"] is None
    assert block["steps_max"] == 40


# --- (c) not always-core ------------------------------------------------------


def test_block_is_not_in_always_core_keys():
    """Making it core would WARN on the entire pre-2026-08-03 corpus."""
    assert MANIFEST_KEY not in manifest_core.ALWAYS_CORE_KEYS


def test_legacy_manifest_without_the_block_is_not_missing_core():
    m = manifest_core.stamp_recording_core(
        {}, config={"n_steps": 5}, seeds=[1], elapsed_seconds=1.0
    )
    assert MANIFEST_KEY not in manifest_core.missing_core_fields(m)


# --- (d) stamping is unkillable and non-destructive ---------------------------

def test_stamping_never_raises_on_junk_input():
    for junk in (object(), 17, "nope", [("a", "b")], [(1,)], {}):
        m = {}
        stamp_episode_termination(m, junk)  # must not raise
    assert stamp_episode_termination(None, None) is None


def test_an_author_set_block_wins_unless_overwrite():
    mine = {"n_episodes": 99}
    acc = EpisodeTerminationAccumulator(steps_configured=10).record(10, "")
    m = {MANIFEST_KEY: mine}
    stamp_episode_termination(m, acc)
    assert m[MANIFEST_KEY] == mine
    stamp_episode_termination(m, acc, overwrite=True)
    assert m[MANIFEST_KEY]["n_episodes"] == 1


def test_record_from_info_is_a_noop_on_a_pre_sd094_info_dict():
    """An env that does not emit the fields must omit the block, not stamp zeros."""
    acc = EpisodeTerminationAccumulator(steps_configured=400)
    acc.record_from_info({"transition_type": "none"})
    acc.record_from_info(None)
    assert acc.stats() is None


# --- (e) end-to-end from the env's own info fields ----------------------------


def _drive(seed, n_steps, acc, **kw):
    env = Env(
        size=10,
        num_hazards=0,
        num_resources=0,
        subgoal_mode=True,
        num_waypoints=3,
        seed=seed,
        **kw,
    )
    env.reset()
    rng = np.random.RandomState(seed)
    info = None
    for _ in range(n_steps):
        _, _, done, info, _ = env.step(int(rng.randint(0, env.action_dim)))
        if done:
            break
    acc.record_from_info(info)


def test_end_to_end_records_the_v3_exq_884_early_death():
    acc = EpisodeTerminationAccumulator(steps_configured=400)
    for seed in (42, 43, 44):
        _drive(seed, 400, acc)
    block = acc.stats()
    assert block["n_episodes"] == 3
    assert block["causes"] == {"health_depleted": 3}
    assert block["full_budget_frac"] == 0.0
    assert block["steps_max"] < 400


def test_end_to_end_records_a_full_budget_run_with_the_gate_on():
    acc = EpisodeTerminationAccumulator(steps_configured=400)
    for seed in (42, 43, 44):
        _drive(seed, 400, acc, hazard_free_contamination_gate=True)
    block = acc.stats()
    assert block["full_budget_frac"] == 1.0
    assert block["steps_min"] == 400
    assert block["causes"] == {"driver_budget": 3}


def test_stamp_recording_core_carries_an_accumulator_through():
    acc = EpisodeTerminationAccumulator(steps_configured=400)
    acc.record(19, "health_depleted")
    m = manifest_core.stamp_recording_core(
        {}, config={"n_steps": 400}, seeds=[42], episode_termination=acc
    )
    assert m[MANIFEST_KEY]["full_budget_frac"] == 0.0
    assert m[MANIFEST_KEY]["causes"] == {"health_depleted": 1}
