"""Contracts pinning that the SD-070 z_world encoder warmup reaches EVERY driver.

CONTEXT -- what went wrong, and what these contracts are actually defending against.
The SD-070 adoption landed as ree-v3 b523b9c (2026-07-20T19:04Z), wiring `run_zworld_p0`
into both `_train_all_on_agent` definition sites. The defect it fixed was silent by
construction: an unfixed driver trains PPO on a FROZEN RANDOM PROJECTION with no error and
no warning, at ~17h of cloud compute per run.

The standing hazard is NOT that the fix was wrong -- it is that the fix has to reach a
FAMILY of drivers, two of which define their own `_train_all_on_agent` copy rather than
importing the shared one. A future encoder-warmup change can therefore land on one
definition site and not the other, and nothing about that failure is visible at runtime.
These are the contracts that make it visible at commit time instead.

WHY THE STRUCTURAL CHECKS CARRY THE WEIGHT. A behavioural test ("did the encoder move?")
needs a real warmup per driver, which is seconds-to-minutes each. The structural checks
(C1/C2) run in milliseconds and pin the property that actually regresses -- a driver
silently ceasing to opt in. C3 pays for ONE real warmup as the behavioural anchor, so the
structural checks cannot all pass against a `run_zworld_p0` that has stopped working.

C4/C5 pin the 2026-07-22 fail-fast, whose absence cost the V3-EXQ-734 run 16.8 hours: the
guard refused `ree_trained_allon` on all 4 rungs x 4 seeds and the driver trained and
evaluated every one of them regardless.

ASCII-only (repo rule).
"""

import sys
from pathlib import Path

import pytest
import torch

import experiments._lib.zworld_encoder_guard as guard
import experiments._lib.zworld_p0_warmup as zp0
import experiments.v3_exq_728b_trained_allon_capability_point as x728b
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734

# The two INDEPENDENT `_train_all_on_agent` definition sites. Every other driver in the
# family (737, 742, mech457_fanout, the exq742 bias-head baseline and its 742m mint) reaches
# the warmup by importing x734's, so pinning these two pins the family.
DEFINITION_SITES = [
    ("v3_exq_734_env_difficulty_competence_recovery_sweep", x734),
    ("v3_exq_728b_trained_allon_capability_point", x728b),
]


def _train_all_on_agent_source_module(mod):
    """The module `_train_all_on_agent` is actually DEFINED in for this driver.

    As of 2026-07-23, x734 no longer defines its own copy -- it re-exports the shared
    `experiments._lib.allon_training._train_all_on_agent` (the substrate-hash
    under-inclusion fix; see that module's docstring). x728b still defines its own in-file
    (deliberate baseline insulation, re-decided DO-NOT-COLLAPSE 2026-07-22). Resolving via
    `__module__` rather than reading `mod`'s own file is what keeps C1 correct for BOTH
    shapes, and still catches a regression if either site grows its own copy again: the
    grown copy's `__module__` reverts to the driver itself, and this then reads that
    driver's source exactly as before.
    """
    fn_module_name = mod._train_all_on_agent.__module__
    return sys.modules[fn_module_name]


# --------------------------------------------------------------------------- C1
@pytest.mark.parametrize("name,mod", DEFINITION_SITES, ids=[n for n, _ in DEFINITION_SITES])
def test_c1_every_definition_site_routes_through_the_shared_warmup(name, mod):
    """Wherever `_train_all_on_agent` is actually DEFINED must call the SHARED `run_zworld_p0`.

    This is the anti-fork contract. If a site grows its own inlined copy of the recipe, this
    fails -- which is the whole point, because a per-copy recipe is exactly how the original
    defect reached some drivers and not others.
    """
    src_mod = _train_all_on_agent_source_module(mod)
    src = Path(src_mod.__file__).read_text(encoding="utf-8")
    assert "from experiments._lib.zworld_p0_warmup import run_zworld_p0" in src, (
        f"{name}'s _train_all_on_agent (defined in {src_mod.__name__}) does not import the "
        "shared run_zworld_p0; a local re-implementation of the SD-070 recipe would drift "
        "from the other definition site silently"
    )
    assert "run_zworld_p0(" in src, (
        f"{name}'s _train_all_on_agent (defined in {src_mod.__name__}) imports run_zworld_p0 "
        "but never calls it"
    )


# --------------------------------------------------------------------------- C2
@pytest.mark.parametrize("name,mod", DEFINITION_SITES, ids=[n for n, _ in DEFINITION_SITES])
def test_c2_every_driver_actually_opts_in(name, mod):
    """Wiring the warmup in is necessary but NOT sufficient -- the driver must also pass a
    non-zero episode count. `zworld_p0_episodes` defaults to 0 (bit-identical prior
    behaviour), so a driver that wires it and then never opts in is exactly as broken as one
    that never wired it, and looks identical in a grep for `run_zworld_p0`.

    ZWORLD_P0_EPISODES is a module constant rather than a CLI arg precisely so that opting in
    is not something a queue entry can forget.
    """
    episodes = getattr(mod, "ZWORLD_P0_EPISODES", None)
    assert episodes is not None, f"{name} defines no ZWORLD_P0_EPISODES constant"
    assert int(episodes) > 0, (
        f"{name} sets ZWORLD_P0_EPISODES={episodes}; with 0 the run trains PPO on a frozen "
        "random projection, which is the V3-EXQ-737/728 defect"
    )


# --------------------------------------------------------------------------- C3
def test_c3_standard_warmup_actually_moves_the_world_encoder():
    """THE BEHAVIOURAL ANCHOR: after the standard P0a warmup, world_encoder_max_abs_delta > 0.

    Deliberately a STRICT comparison against 0.0, matching the guard: the signature being
    detected is bit-identity, so `delta == 0.0` is the failure case and an inclusive floor
    would recompute a frozen random projection as passing.

    Run on ONE definition site only -- C1 pins that both route through the same shared
    `run_zworld_p0`, so a second real warmup would re-measure the same code for real compute.
    """
    env = x734._make_env(42, x734._env_kwargs_for_rung(x734.DIFFICULTY_RUNGS[-1]))
    agent = x734._make_all_on_agent(env)
    before = guard.latent_stack_snapshot(agent)
    assert before, "agent exposes no latent_stack parameters; the guard would be vacuous"

    # A DEDICATED env instance, same seed/kwargs: the warmup rollout consumes env RNG.
    zp0.run_zworld_p0(
        agent,
        x734._make_env(42, x734._env_kwargs_for_rung(x734.DIFFICULTY_RUNGS[-1])),
        42, 2, 8,
        policy=x734.RandomPolicy(42),
        label="contract:c3",
        dry_run=True,
    )

    report = guard.latent_stack_weight_delta(agent, before)
    assert report["world_encoder_max_abs_delta"] > 0.0, (
        "the standard P0a warmup left every split_encoder.world_encoder tensor "
        "bit-identical -- z_world is a frozen random projection. This is the "
        "V3-EXQ-780/737/728 defect reappearing. "
        f"report={report}"
    )
    assert report["zworld_encoder_trained"] is True
    # SD-070 contract C5: the warmup trains the WORLD path and must leave z_self alone.
    assert report["n_world_encoder_changed"] > 0


# --------------------------------------------------------------------------- C4
@pytest.mark.parametrize("name,mod", DEFINITION_SITES, ids=[n for n, _ in DEFINITION_SITES])
def test_c4_refused_cells_are_not_evaluated(name, mod):
    """A guard-refused cell must produce an UNEVALUATED row, not a measurement.

    V3-EXQ-734 evaluated all 16 refused cells and emitted ordinary-looking competence
    numbers for every one. Those numbers are uninterpretable as evidence about z_world, but
    nothing downstream that fails to also read `zworld_guard_ok` can tell. Emitting None
    rather than a number makes the absence of a measurement structural instead of advisory.

    None rather than 0.0 is load-bearing: 0.0 is a REAL possible foraging_competence on these
    rungs, so a zero-filled refusal would be indistinguishable from a genuine floor result.
    """
    row = mod._unevaluated_row("ree_trained_allon")
    for metric in (
        "foraging_competence", "survival_horizon", "death_rate",
        "goal_reach_rate", "planning_depth",
    ):
        assert metric in row, f"{name}._unevaluated_row drops {metric}, so printing it KeyErrors"
        assert row[metric] is None, (
            f"{name}._unevaluated_row sets {metric}={row[metric]!r}; a refused cell has no "
            "measurement and a numeric placeholder would enter summarize_arm as if real"
        )
    assert row["n_episodes"] == 0


# --------------------------------------------------------------------------- C5
def test_c5_failfast_latch_elides_later_cells():
    """Once the guard refuses, later cells of the guarded arm must be skipped BEFORE training.

    The frozen-encoder condition is a property of the code path, not of a seed, so every
    remaining cell reproduces it. Skipping before the agent is built is what elides the
    P0a/P0/P1 warmup -- the bulk of the cost, and the reason V3-EXQ-734 burned 16.8h.
    """
    state = {"tripped": True, "first_context": "rung=D0_baseline_724 seed=42"}
    row = x734._run_cell(
        x734.DIFFICULTY_RUNGS[0], "ree_trained_allon", 43,
        x734._env_kwargs_for_rung(x734.DIFFICULTY_RUNGS[0]),
        p0_episodes=2, p1_episodes=2, p1_ppo_episodes=2,
        eval_episodes=1, steps_per_episode=8, rollout_episodes=1,
        zworld_p0_episodes=0, zworld_p0_dry_run=True,
        guard_state=state,
    )
    assert row["zworld_guard_refused"] is True
    assert row["zworld_guard_skipped"] is True
    assert row["zworld_guard_ok"] is False
    assert row["foraging_competence"] is None
    # Never trained: a skipped cell has no guard report because no warmup was run.
    assert row["zworld_guard"] is None
    assert row["train_stats"] == {}


# --------------------------------------------------------------------------- C6
def test_c6_sources_are_ascii():
    for _name, mod in DEFINITION_SITES + [("zworld_p0_warmup", zp0)]:
        src = Path(mod.__file__).read_text(encoding="utf-8")
        bad = [(i, ch) for i, ch in enumerate(src) if ord(ch) > 127]
        assert not bad, f"non-ASCII in {mod.__name__}: {bad[:5]}"
