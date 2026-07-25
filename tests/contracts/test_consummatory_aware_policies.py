"""Contract tests for consummatory-aware reference/demonstrator policies (2026-07-25).

mech457_consummatory_act (2026-07-25) made contact AFFORD rather than EFFECT consumption in the
consummatory env: a competent forager must select the distinct CONSUME action while standing on a
resource cell, not merely arrive. The hand-coded greedy policies in capability_eval.py
(OraclePolicy, LocalViewGreedyPolicy) otherwise return "stay" on the target cell and so never
forage in the consummatory env -- which would leave the readiness anchors reading the env as
unsolvable and a BC install (cloned from the LocalViewGreedyPolicy demonstrator) unable to take.
This is the consumer that lets the retention framing of MECH-457 H-consummation-binding (leg 4)
run in the consummatory env.

Contracts:
  CG1 DEFAULT NO-OP. In a NON-consummatory env, neither greedy policy ever emits the CONSUME
      index and the helper returns None -- the new branch is inert, so behaviour is byte-identical
      to before the change. Both still forage >= the competence floor there.
  CG2 CONSUMMATORY CAPABILITY. In the consummatory env both greedy policies forage >= the
      competence floor -- only possible if they select CONSUME on contact (arrival alone yields 0
      benefit). RandomPolicy stays at the floor (not made consummatory-aware).
  CG3 HELPER SEMANTICS. _consummatory_consume_action returns None off / when not on a resource,
      and the CONSUME index when consummatory AND standing on a resource cell.
  CG4 Module source is ASCII-only (repo runtime-string rule).
"""

from pathlib import Path

import pytest

import experiments._lib.capability_eval as cap
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    OraclePolicy,
    RandomPolicy,
    _consummatory_consume_action,
    evaluate_seed,
)
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734


EVAL_EPS = 8
STEPS = 40
_EK = x734._env_kwargs_for_rung(fan.RUNG)


def _env(seed: int, consummatory: bool):
    kw = dict(_EK)
    if consummatory:
        kw["consummatory_act_enabled"] = True
    return x734._make_env(seed, kw)


class _DummyEnv:
    def __init__(self, consummatory, on_resource, consume_idx=5):
        self.consummatory_act_enabled = consummatory
        self._on_consumable_resource = on_resource
        self.CONSUME_ACTION = consume_idx


# --------------------------------------------------------------------------- CG1
def test_cg1_non_consummatory_is_noop():
    """Off-path: the helper returns None (so act() falls through to the pre-change branch), and
    both greedy policies still forage >= the floor in the 5-action env. The env exposes no CONSUME
    action (action_dim == 5), so the off-path is byte-identical to before this change."""
    env = _env(0, consummatory=False)
    assert env.action_dim == 5
    assert _consummatory_consume_action(env) is None
    lv = evaluate_seed(LocalViewGreedyPolicy(seed=0), _env(0, False), EVAL_EPS, STEPS)
    orc = evaluate_seed(OraclePolicy(), _env(0, False), EVAL_EPS, STEPS)
    assert lv["foraging_competence"] >= COMPETENCE_RESOURCE_FLOOR
    assert orc["foraging_competence"] >= COMPETENCE_RESOURCE_FLOOR


# --------------------------------------------------------------------------- CG2
def test_cg2_consummatory_greedy_policies_forage():
    """In the consummatory env, arrival yields 0 benefit -- foraging >= floor is only possible if
    the policy selects CONSUME on contact. RandomPolicy is NOT consummatory-aware -> stays low."""
    env = _env(0, consummatory=True)
    assert env.action_dim == 6
    lv = evaluate_seed(LocalViewGreedyPolicy(seed=0), _env(0, True), EVAL_EPS, STEPS)
    orc = evaluate_seed(OraclePolicy(), _env(0, True), EVAL_EPS, STEPS)
    assert lv["foraging_competence"] >= COMPETENCE_RESOURCE_FLOOR, (
        f"local_view_greedy did not forage in the consummatory env "
        f"({lv['foraging_competence']}) -- CONSUME hand-off not firing"
    )
    assert orc["foraging_competence"] >= COMPETENCE_RESOURCE_FLOOR, (
        f"greedy_oracle did not forage in the consummatory env ({orc['foraging_competence']})"
    )
    rw = evaluate_seed(RandomPolicy(0), _env(0, True), EVAL_EPS, STEPS)
    assert rw["foraging_competence"] < orc["foraging_competence"], (
        "random_walk should not have been made consummatory-aware"
    )


# --------------------------------------------------------------------------- CG3
def test_cg3_helper_semantics():
    assert _consummatory_consume_action(_DummyEnv(False, True)) is None      # off -> None
    assert _consummatory_consume_action(_DummyEnv(True, False)) is None      # not on resource
    assert _consummatory_consume_action(_DummyEnv(True, True)) == 5          # consume now
    assert _consummatory_consume_action(_DummyEnv(True, True, consume_idx=7)) == 7


# --------------------------------------------------------------------------- CG4
def test_cg4_module_ascii_only():
    src = Path(cap.__file__).read_text(encoding="utf-8")
    bad = [(i + 1, ln) for i, ln in enumerate(src.splitlines())
           if any(ord(c) > 127 for c in ln)]
    assert not bad, f"{cap.__file__} has non-ASCII lines: {bad[:3]}"
