"""Contract tests for experiments/_lib/sample_driven_rollout.

Guards the fix routed by
`REE_assembly/evidence/planning/failure_autopsy_MECH-063-777-779-cluster_2026-07-18.md`
section 7(c): the shared 2x2 read-only telemetry-probe template fixed its
sampling budget in EPISODES while its readiness gates were denominated in
SELECTIONS / EVENT TICKS. With an untrained agent in a hazard-terminating env,
episode survival varied 40x across one seed set (seed 23: ~21 of 900 steps;
seeds 29/37: 900 of 900), so the budget did not determine the sample and a
`min_seeds = 4 of 5` bar became arithmetically unreachable.

The load-bearing assertions here:
  * stopping is denominated in SAMPLES, and the rollout auto-resets across
    episodes to reach them (test_starved_env_still_reaches_floor -- this is the
    exact 777/779 defect, and it is the test that would have failed before);
  * a hard step cap bounds a pathological cell, and the shortfall is reported
    EXPLICITLY rather than silently under-sampling;
  * realised counts (n_env_steps especially) come back for the manifest;
  * starvation self-routes to `sample_starvation_requeue` and NAMES the cell.

Fakes only -- no substrate, no torch, no env. ASCII-only.
Run: pytest tests/contracts/test_sample_driven_rollout.py -q
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

# conftest puts ree-v3 root on sys.path -> `experiments._lib.*` importable.
from experiments._lib.sample_driven_rollout import (
    SELF_ROUTE_SAMPLE_STARVATION,
    RolloutBudget,
    RolloutOutcome,
    TickContext,
    run_cell_until_samples,
    starvation_selfroute,
)


# --------------------------------------------------------------------------- #
# Fakes: an env that terminates after a fixed number of steps, an agent whose
# e3 hands out a NEW probs object on selection ticks (identity-based freshness),
# and a harness that just forwards to the env.
# --------------------------------------------------------------------------- #

class _FakeStepResult:
    def __init__(self, done: bool, next_obs_dict: Dict[str, Any]):
        self.done = done
        self.next_obs_dict = next_obs_dict


class _FakeEnv:
    """Terminates `episode_length` steps after each reset."""

    def __init__(self, episode_length: int):
        self.episode_length = int(episode_length)
        self.n_resets = 0
        self._t = 0

    def reset(self):
        self.n_resets += 1
        self._t = 0
        return None, {"obs": 0}

    def step(self):
        self._t += 1
        return _FakeStepResult(
            done=self._t >= self.episode_length, next_obs_dict={"obs": self._t}
        )


class _FakeE3:
    def __init__(self):
        self.last_precommit_probs: Optional[List[float]] = None


class _FakeAgent:
    """Emits a NEW probs object every `select_every` steps; else leaves it put.

    Mirrors the real freshness signature: E3 does not necessarily run a fresh
    selection on every env tick, and both 777 and 779 detected that by tensor
    IDENTITY (`id(probs) != prev_probs_id`), not by value.
    """

    def __init__(self, select_every: int = 1):
        self.e3 = _FakeE3()
        self.select_every = int(select_every)
        self.n_resets = 0
        self._t = 0

    def reset(self):
        self.n_resets += 1
        self._t = 0

    def tick(self):
        self._t += 1
        if self.e3 is not None and self._t % self.select_every == 0:
            # New object each time -> fresh.
            self.e3.last_precommit_probs = [0.5, 0.5]


class _FakeHarness:
    def __init__(self, env: _FakeEnv, agent: _FakeAgent):
        self.env = env
        self.agent = agent
        self.n_resets = 0

    def reset(self):
        self.n_resets += 1

    def step(self, obs_dict):
        self.agent.tick()
        return self.env.step()


def _mk(episode_length: int, select_every: int = 1):
    env = _FakeEnv(episode_length)
    agent = _FakeAgent(select_every)
    return env, agent, _FakeHarness(env, agent)


def _count_fresh(ctx: TickContext) -> Dict[str, int]:
    return {"selections": 1} if ctx.fresh else {}


def _run(env, agent, harness, budget, observe=_count_fresh, **kw):
    lines: List[str] = []
    outcome = run_cell_until_samples(
        env=env,
        agent=agent,
        harness=harness,
        budget=budget,
        observe=observe,
        progress_fn=lines.append,
        **kw,
    )
    return outcome, lines


# --------------------------------------------------------------------------- #
# THE core regression: sample-driven stopping survives a hazard-terminating env.
# --------------------------------------------------------------------------- #

def test_starved_env_still_reaches_floor():
    """A seed-23-shaped env (21-step episodes) must still reach 20 selections.

    Under the OLD episode-denominated template (3 episodes x 300 steps, break on
    done) this cell yielded ~21 env steps total and 20 selections -- exactly ON
    the MIN_SELECTS bar, and below it in V3-EXQ-779 (19). Sample-driven stopping
    auto-resets and continues until the floor is genuinely met.
    """
    env, agent, harness = _mk(episode_length=7)
    budget = RolloutBudget(
        sample_floors={"selections": 20}, max_env_steps=900, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.floors_met
    assert outcome.stop_reason == "floors_met"
    assert outcome.counts["selections"] == 20
    # 7-step episodes -> needed multiple resets to get there.
    assert outcome.n_episodes == 3
    assert env.n_resets == 3
    assert outcome.n_env_steps == 20
    assert outcome.starved_readouts == []


def test_all_three_are_reset_at_each_episode_boundary():
    """env / agent / harness must all be reset per episode, in that order."""
    env, agent, harness = _mk(episode_length=5)
    budget = RolloutBudget(
        sample_floors={"selections": 12}, max_env_steps=900, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.n_episodes == env.n_resets == agent.n_resets == harness.n_resets


def test_healthy_env_stops_as_soon_as_floor_is_met():
    """A seed-29-shaped env (never terminates) must not burn the whole cap."""
    env, agent, harness = _mk(episode_length=10_000)
    budget = RolloutBudget(
        sample_floors={"selections": 20}, max_env_steps=900, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.stop_reason == "floors_met"
    assert outcome.n_env_steps == 20  # not 900
    assert outcome.n_episodes == 1


# --------------------------------------------------------------------------- #
# The cap binds -> report it explicitly, do not silently under-sample.
# --------------------------------------------------------------------------- #

def test_step_cap_binds_and_shortfall_is_explicit():
    env, agent, harness = _mk(episode_length=10_000, select_every=10)
    budget = RolloutBudget(
        sample_floors={"selections": 20}, max_env_steps=50, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.stop_reason == "step_cap"
    assert not outcome.floors_met
    assert outcome.starved_readouts == ["selections"]
    assert outcome.counts["selections"] == 5
    assert outcome.n_env_steps == 50  # cap respected exactly, never exceeded


def test_step_cap_is_never_exceeded_across_episode_boundaries():
    env, agent, harness = _mk(episode_length=7)
    budget = RolloutBudget(
        sample_floors={"selections": 10_000}, max_env_steps=30, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.n_env_steps == 30
    assert outcome.stop_reason == "step_cap"


def test_episode_cap_backstops_a_degenerate_env():
    """An env that terminates on step 1 must not spin forever on resets."""
    env, agent, harness = _mk(episode_length=1)
    budget = RolloutBudget(
        sample_floors={"selections": 10_000},
        max_env_steps=10_000,
        steps_per_episode=300,
        max_episodes=25,
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.stop_reason == "episode_cap"
    assert outcome.n_episodes == 25
    assert not outcome.floors_met


# --------------------------------------------------------------------------- #
# Freshness: identity-based, matching the inline 777/779 logic.
# --------------------------------------------------------------------------- #

def test_stale_probs_do_not_count_as_samples():
    """Only NEW probs objects count -- a left-in-place distribution does not."""
    env, agent, harness = _mk(episode_length=10_000, select_every=4)
    budget = RolloutBudget(
        sample_floors={"selections": 10}, max_env_steps=900, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.counts["selections"] == 10
    assert outcome.n_env_steps == 40  # 4 env steps per fresh selection


def test_freshness_survives_an_episode_boundary():
    """The first tick of a new episode is fresh even if the object is reused."""
    env, agent, harness = _mk(episode_length=3)
    seen: List[bool] = []

    def observe(ctx: TickContext) -> Dict[str, int]:
        seen.append(ctx.fresh)
        return {"selections": 1} if ctx.fresh else {}

    budget = RolloutBudget(
        sample_floors={"selections": 9}, max_env_steps=900, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget, observe=observe)

    assert all(seen)  # select_every=1 -> every tick fresh
    assert outcome.counts["selections"] == 9


def test_absent_e3_yields_no_fresh_samples():
    env, agent, harness = _mk(episode_length=10_000)
    agent.e3 = None
    budget = RolloutBudget(
        sample_floors={"selections": 5}, max_env_steps=20, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.counts["selections"] == 0
    assert outcome.stop_reason == "step_cap"
    assert not outcome.floors_met


# --------------------------------------------------------------------------- #
# Multi-readout floors -- the V3-EXQ-779 event-tick case.
# --------------------------------------------------------------------------- #

def test_multiple_floors_all_must_be_met():
    """Stopping waits for the SLOWEST readout, not the first one satisfied."""
    env, agent, harness = _mk(episode_length=10_000)

    def observe(ctx: TickContext) -> Dict[str, int]:
        if not ctx.fresh:
            return {}
        inc = {"selections": 1}
        if ctx.n_env_steps % 5 == 0:  # sparse "event tick"
            inc["event_ticks"] = 1
        return inc

    budget = RolloutBudget(
        sample_floors={"selections": 20, "event_ticks": 10},
        max_env_steps=900,
        steps_per_episode=300,
    )
    outcome, _ = _run(env, agent, harness, budget, observe=observe)

    assert outcome.floors_met
    assert outcome.counts["event_ticks"] == 10
    assert outcome.counts["selections"] == 50  # kept going past its own floor


def test_readout_with_no_floor_is_counted_but_does_not_gate():
    """Arms where a readout cannot fire (e.g. PHASIC-OFF) must not starve.

    Declaring a floor for a structurally-zero readout would burn the whole step
    cap and report false starvation -- the module docstring warns about this and
    this test pins the escape hatch.
    """
    env, agent, harness = _mk(episode_length=10_000)

    def observe(ctx: TickContext) -> Dict[str, int]:
        return {"selections": 1, "event_ticks": 0} if ctx.fresh else {}

    budget = RolloutBudget(
        sample_floors={"selections": 10}, max_env_steps=900, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget, observe=observe)

    assert outcome.floors_met
    assert outcome.stop_reason == "floors_met"
    assert outcome.counts["event_ticks"] == 0
    assert "event_ticks" not in outcome.starved_readouts


def test_no_floors_runs_to_the_step_cap():
    env, agent, harness = _mk(episode_length=10_000)
    budget = RolloutBudget(
        sample_floors={}, max_env_steps=25, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)

    assert outcome.n_env_steps == 25
    assert outcome.stop_reason == "step_cap"
    assert outcome.floors_met  # vacuously -- nothing was demanded


# --------------------------------------------------------------------------- #
# Manifest recording -- what made the original diagnosis possible.
# --------------------------------------------------------------------------- #

def test_manifest_fields_preserve_realised_counts():
    env, agent, harness = _mk(episode_length=7)
    budget = RolloutBudget(
        sample_floors={"selections": 20}, max_env_steps=900, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)
    fields = outcome.as_manifest_fields()

    # n_env_steps per cell is the field the autopsy singled out (learning 7).
    assert fields["n_env_steps"] == 20
    assert fields["n_episodes"] == 3
    assert fields["rollout_stop_reason"] == "floors_met"
    assert fields["rollout_floors_met"] is True
    assert fields["rollout_sample_counts"]["selections"] == 20
    assert fields["rollout_sample_floors"]["selections"] == 20
    assert "rollout_starved_readouts" not in fields


def test_manifest_fields_name_the_starved_readout():
    env, agent, harness = _mk(episode_length=10_000, select_every=10)
    budget = RolloutBudget(
        sample_floors={"selections": 20}, max_env_steps=50, steps_per_episode=300
    )
    outcome, _ = _run(env, agent, harness, budget)
    fields = outcome.as_manifest_fields()

    assert fields["rollout_floors_met"] is False
    assert fields["rollout_starved_readouts"] == ["selections"]


# --------------------------------------------------------------------------- #
# Self-routing -- autopsy learning 4.
# --------------------------------------------------------------------------- #

def test_selfroute_is_none_when_every_cell_is_healthy():
    healthy = RolloutOutcome(
        n_env_steps=20,
        n_episodes=1,
        counts={"selections": 20},
        floors={"selections": 20},
        stop_reason="floors_met",
    )
    assert starvation_selfroute([{"arm": "T0S0", "seed": 11, "outcome": healthy}]) is None


def test_selfroute_names_the_offending_cell():
    """779's readiness said 'measured 5, threshold 10' and named no cell.

    One starved cell (seed 23 / T1P1, 19 env steps) vetoed a run in which the
    substrate was firing everywhere else, under a label that pointed at a
    substrate capability. The record must name the cell and its shortfall.
    """
    starved = RolloutOutcome(
        n_env_steps=900,
        n_episodes=42,
        counts={"selections": 20, "event_ticks": 5},
        floors={"selections": 20, "event_ticks": 10},
        stop_reason="step_cap",
    )
    healthy = RolloutOutcome(
        n_env_steps=20,
        n_episodes=1,
        counts={"selections": 20, "event_ticks": 10},
        floors={"selections": 20, "event_ticks": 10},
        stop_reason="floors_met",
    )
    record = starvation_selfroute(
        [
            {"arm": "T0P0", "seed": 11, "outcome": healthy},
            {"arm": "T1P1", "seed": 23, "outcome": starved},
        ]
    )

    assert record is not None
    assert record["self_route"] == SELF_ROUTE_SAMPLE_STARVATION
    assert record["self_route"] != "substrate_not_ready_requeue"
    assert record["n_starved_cells"] == 1

    cell = record["starved_cells"][0]
    assert cell["arm"] == "T1P1" and cell["seed"] == 23
    assert cell["n_env_steps"] == 900
    assert cell["stop_reason"] == "step_cap"
    assert cell["shortfall"] == {"event_ticks": {"measured": 5, "threshold": 10}}
    # The readout that WAS satisfied must not appear as a shortfall.
    assert "selections" not in cell["shortfall"]


def test_selfroute_ignores_non_outcome_entries():
    assert starvation_selfroute([{"arm": "T0P0", "seed": 11, "outcome": None}]) is None


# --------------------------------------------------------------------------- #
# Budget validation + output hygiene.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_env_steps": 0},
        {"max_env_steps": -1},
        {"max_env_steps": 100, "steps_per_episode": 0},
        {"max_env_steps": 100, "sample_floors": {"selections": -1}},
    ],
)
def test_budget_rejects_unbounded_or_nonsense_configs(kwargs):
    with pytest.raises(ValueError):
        RolloutBudget(**kwargs)


def test_max_episodes_defaults_to_a_finite_bound():
    budget = RolloutBudget(sample_floors={"selections": 5}, max_env_steps=900)
    assert budget.max_episodes > 0


def test_progress_output_is_ascii_only():
    env, agent, harness = _mk(episode_length=7)
    budget = RolloutBudget(
        sample_floors={"selections": 20}, max_env_steps=900, steps_per_episode=300
    )
    outcome, lines = _run(env, agent, harness, budget, progress_label="T1S0 seed=11")

    assert lines
    for line in lines:
        line.encode("ascii")  # raises on any non-ASCII character
    assert "T1S0 seed=11" in lines[0]
    outcome.summary_line().encode("ascii")
