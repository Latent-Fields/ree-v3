"""Q-042 contract: dynamic-precision running-variance update.

Background: 5 experiment scripts (V3-EXQ-514d, 514e, 524, 530, 536) skipped
``agent.update_residue()`` in their step loops. ``update_residue()`` is the
only call site of ``E3Selector.post_action_update()``, which is the only
call site of ``E3Selector.update_running_variance()``. With that path
dormant, ``_running_variance`` stays at ``precision_init=0.5`` for the
whole run; the BG commit gate (commit when rv < threshold) never fires;
``current_precision = 1/(0.5 + 1e-6) = 1.999996`` constant. EXQ-530's
"ARC-016 weakens" classification was not actually a test of ARC-016
because the precision signal was never a variable.

Q-042 lit-pull (docs/architecture/precision_update_callsite.md) confirmed
that the post-outcome update site is correct on both biological and ML
grounds (Behrens 2007 ACC volatility at outcome-monitoring, Nassar 2012
LC-NE post-outcome PE, Yu & Dayan 2005 NE-as-unexpected-uncertainty,
Friston active-inference sensory attenuation at action time; Kalman
predict-then-update, PPO running stats over rollout buffer, DreamerV3
posterior in world-model training step). The robustness problem is a
software-contract issue, not an architectural one.

Guarantees enforced here:
  R1. Over a 100-step random-action episode that calls update_residue() each
      step, e3._running_variance moves more than ``1e-6`` from
      ``config.precision_init``. This is the positive contract: the
      precision update path is wired and live.
  R2. Over a 100-step random-action episode that omits update_residue()
      (the EXQ-514d/530/536 failure mode), e3._running_variance stays
      EXACTLY at ``config.precision_init``. This locks update_residue()
      as the sole call site so a future refactor that adds a parallel
      update path will be visible to this test.
  R3. The StepHarness canonical-loop path (experiments/_harness.py) keeps
      rv live -- it must not silently regress to the failure mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# experiments/ is not on sys.path by default in pytest; add it.
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from _harness import StepHarness  # noqa: E402

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import step_once


N_STEPS = 100
RV_MOVE_TOL = 1e-6


def _build_agent_env(seed: int = 0):
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)
    return agent, env, cfg


# -- R1: rv moves under the canonical loop --------------------------------

def test_running_variance_moves_when_update_residue_called():
    """Positive contract: when the loop calls update_residue() each step,
    running_variance must move from precision_init within a representative
    episode. If this fails, the precision update wiring is broken."""
    agent, env, cfg = _build_agent_env(seed=0)

    rv_init = agent.e3._running_variance
    assert rv_init == cfg.e3.precision_init, (
        f"rv at construction is {rv_init}; expected precision_init="
        f"{cfg.e3.precision_init}. Init contract regression."
    )

    agent.reset()
    _flat, obs_dict = env.reset()
    for _ in range(N_STEPS):
        _action, _idx, _ticks, obs_dict = step_once(
            agent, env, obs_dict, update_residue=True
        )

    rv_final = agent.e3._running_variance
    assert abs(rv_final - cfg.e3.precision_init) > RV_MOVE_TOL, (
        f"running_variance pinned at {rv_final} after {N_STEPS} steps with "
        f"update_residue() called every step. precision_init was "
        f"{cfg.e3.precision_init}. The Q-042 EXQ-514d/530/536 failure mode "
        f"has resurfaced: update_residue() -> e3.post_action_update() -> "
        f"update_running_variance() chain is broken."
    )
    # Sanity: current_precision must therefore differ from the init constant.
    init_precision = 1.0 / (cfg.e3.precision_init + 1e-6)
    assert abs(agent.e3.current_precision - init_precision) > RV_MOVE_TOL, (
        f"current_precision still pinned at {agent.e3.current_precision} "
        f"(init {init_precision}); rv changed but precision did not -- "
        f"property wiring is broken."
    )


# -- R2: failure-mode lock -------------------------------------------------

def test_running_variance_pinned_when_update_residue_omitted():
    """Negative contract: with update_residue() never called, rv stays at
    precision_init exactly. This locks update_residue() as the sole call
    site of the rv update path. If a future refactor adds a parallel update
    site (e.g. hoisting into select_action() per Q-042 Option B), this test
    fails and the architectural decision must be revisited explicitly --
    not silently. See docs/architecture/precision_update_callsite.md."""
    agent, env, cfg = _build_agent_env(seed=1)

    agent.reset()
    _flat, obs_dict = env.reset()
    for _ in range(N_STEPS):
        _action, _idx, _ticks, obs_dict = step_once(
            agent, env, obs_dict, update_residue=False
        )

    rv_final = agent.e3._running_variance
    assert rv_final == cfg.e3.precision_init, (
        f"running_variance moved to {rv_final} (init {cfg.e3.precision_init}) "
        f"after {N_STEPS} steps WITHOUT update_residue() being called. The "
        f"Q-042 architectural commitment (post-outcome single-site update) "
        f"has been broken: rv now updates from somewhere other than "
        f"update_residue() -> e3.post_action_update(). If this is "
        f"intentional, see docs/architecture/precision_update_callsite.md "
        f"and update Q-042 in claims.yaml before changing this test."
    )


# -- R3: StepHarness canonical loop keeps rv live --------------------------

def test_step_harness_canonical_loop_keeps_running_variance_live():
    """StepHarness is the canonical experiment-loop pattern. Confirm it
    drives the rv update across a representative episode -- if it regressed
    to a no-update_residue path, the entire post-cohort-fix retest plan
    would silently re-contaminate."""
    set_all_seeds(2)
    env = make_tiny_env(seed=2)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)

    rv_init = agent.e3._running_variance
    assert rv_init == cfg.e3.precision_init

    harness = StepHarness(agent, env, train_mode=False, seed=2)
    _flat, obs_dict = env.reset()
    agent.reset()
    harness.reset()

    for _ in range(N_STEPS):
        result = harness.step(obs_dict)
        obs_dict = result.next_obs_dict
        if result.done:
            break

    rv_final = agent.e3._running_variance
    assert abs(rv_final - cfg.e3.precision_init) > RV_MOVE_TOL, (
        f"StepHarness ran {N_STEPS} steps and running_variance is still "
        f"pinned at {rv_final} (init {cfg.e3.precision_init}). The harness "
        f"is no longer driving the rv update -- it must call "
        f"agent.update_residue() each step (see "
        f"experiments/_harness.py::StepHarness.step)."
    )
