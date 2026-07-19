"""Contract tests for SD-075 phasic_ema_episode_continuity.

Repairs a MEASUREMENT defect in SD-069 phasic_surprise_burst (MECH-063 sub-
claim ii): reset() cleared the surprise-EMA cold at every episode boundary,
and the first waking tick of an episode cannot fire (it seeds the baseline).
With surprise_ema_decay 0.1 the baseline needs ~10 ticks, so a seed whose
episodes are shorter than that never runs against a converged baseline and
n_event_ticks measures episode LENGTH rather than surprise. V3-EXQ-779b seed
23 ran ~6.9-step episodes against seeds 29/37 at 300 steps; raising its budget
835 -> 2400 env steps delivered 345 MORE short episodes and the
phasic_fires_real_events precondition did not move (6 vs threshold 10).

Contracts (D1-D9):
  D1: defaults are no-op. baseline_continuity "reset" + warmup_ticks 0 ->
      SD-069 behaviour is bit-identical, including reset() still clearing the
      baseline.
  D2: "carry" preserves the surprise-EMA across reset() while STILL clearing
      the envelope, the cached temperature delta, and per-episode diagnostics.
  D3: THE DEFECT ITSELF. On a short-episode schedule, "reset" structurally
      under-reports events versus "carry" on the SAME surprise stream.
  D4: warmup_ticks derives as ceil(3 / surprise_ema_decay) when -1, is used
      verbatim when positive, and means OFF when 0.
  D5: the gate is ACCOUNTING ONLY -- burst_level, temperature_delta, and the
      action stream are unchanged by warmup_ticks; only the counts split.
  D6: n_events_converged + n_events_prewarmup == n_events, and the converged/
      prewarmup tick counts partition lifetime_ticks.
  D7: MECH-094 simulation gate does not advance the SD-075 lifetime counters.
  D8: input validation. Bad baseline_continuity / warmup_ticks raise.
  D9: agent-level wiring. The two REEConfig fields reach the live regulator.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.regulators import PhasicSurpriseBurst, PhasicSurpriseBurstConfig
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments._harness import StepHarness


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _mk_env(seed):
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, seed=seed)


def _dims(env):
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )


def _run(cfg, steps=12, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = REEAgent(cfg)
    res = StepHarness(agent, _mk_env(seed), train_mode=True, seed=seed).run_episode(
        max_steps=steps
    )
    actions = [int(r.action.argmax().item()) for r in res]
    return agent, actions


def _mk(**kw):
    return PhasicSurpriseBurst(PhasicSurpriseBurstConfig(**kw))


# Two episode shapes. The distinction matters and is easy to get wrong.
#
# LATE SPIKE: five quiet ticks then a spike. A cold-reset baseline has already
# converged by tick 6, so "reset" recovers the event and this shape does NOT
# discriminate the modes. It is used where a CONSTANT non-zero per-episode
# yield is wanted.
_EPISODE_LATE_SPIKE = [1.0, 1.0, 1.0, 1.0, 1.0, 4.0]

# EARLY SPIKE: the salient moment lands at the START of a short episode -- the
# 779b shape, where the episode is over before a ~10-tick baseline converges.
# Under "reset" the opening tick only SEEDS the baseline (and seeds it HIGH,
# from the spike itself), so the event is structurally invisible and stays
# invisible no matter how many episodes are added. Under "carry" the baseline
# reflects the agent's quiet lifetime, so the same spike reads as a real
# relative excess.
_EPISODE_EARLY_SPIKE = [4.0, 0.5, 0.5, 0.5]


def _run_episodes(pb, n_episodes, episode=_EPISODE_LATE_SPIKE):
    """Drive n_episodes of `episode`, calling reset() at each boundary."""
    for _ in range(n_episodes):
        for s in episode:
            pb.tick(s)
        pb.reset()
    return pb.get_state()


# ----------------------------------------------------------------------
# D1 defaults are no-op
# ----------------------------------------------------------------------
def test_d1_defaults_are_sd069_no_op():
    c = PhasicSurpriseBurstConfig()
    assert c.baseline_continuity == "reset"
    assert c.warmup_ticks == 0

    pb = _mk()
    pb.tick(1.0)
    pb.tick(5.0)
    assert pb.get_state()["n_events"] == 1
    pb.reset()
    st = pb.get_state()
    # SD-069 C7 behaviour must survive verbatim.
    assert st["surprise_ema"] == 0.0
    assert st["n_events"] == 0
    assert st["n_waking_ticks"] == 0
    assert st["burst_level"] == 0.0
    # warmup_ticks 0 -> every tick is converged, so the split is degenerate
    # and n_events_converged tracks n_events exactly.
    assert st["warmup_ticks"] == 0
    assert st["n_prewarmup_ticks"] == 0
    assert st["baseline_converged"] is True


# ----------------------------------------------------------------------
# D2 "carry" preserves the baseline but not the envelope
# ----------------------------------------------------------------------
def test_d2_carry_preserves_baseline_clears_envelope():
    pb = _mk(baseline_continuity="carry")
    pb.tick(1.0)
    pb.tick(1.0)
    pb.tick(6.0)
    ema_before = pb.get_state()["surprise_ema"]
    assert ema_before > 0.0
    assert pb.burst_level > 0.0

    pb.reset()
    st = pb.get_state()
    # The slow baseline persists ...
    assert st["surprise_ema"] == pytest.approx(ema_before)
    # ... but nothing episode-scoped leaks across the boundary.
    assert st["burst_level"] == 0.0
    assert st["temperature_delta"] == 0.0
    assert st["n_events"] == 0
    assert st["n_waking_ticks"] == 0
    assert st["last_event_fired"] is False


def test_d2b_carry_first_tick_of_later_episode_can_fire():
    """Under "reset" the first tick of an episode can NEVER fire (it seeds the
    baseline). That is the proximate defect; "carry" must remove it."""
    pb_reset = _mk(baseline_continuity="reset")
    pb_carry = _mk(baseline_continuity="carry")
    for pb in (pb_reset, pb_carry):
        for s in (1.0, 1.0, 1.0):
            pb.tick(s)
        pb.reset()

    # A spike as the very FIRST tick of episode 2.
    pb_reset.tick(9.0)
    pb_carry.tick(9.0)
    assert pb_reset.get_state()["n_events"] == 0, "reset mode seeds, cannot fire"
    assert pb_carry.get_state()["n_events"] == 1, "carry mode has a live baseline"


# ----------------------------------------------------------------------
# D3 the defect: short episodes under-report under "reset"
# ----------------------------------------------------------------------
def test_d3_short_episodes_under_report_under_reset():
    """Same surprise stream, same total ticks, only the continuity differs."""
    n_eps = 20
    ep = _EPISODE_EARLY_SPIKE
    st_reset = _run_episodes(_mk(baseline_continuity="reset"), n_eps, ep)
    st_carry = _run_episodes(_mk(baseline_continuity="carry"), n_eps, ep)

    assert st_reset["lifetime_ticks"] == st_carry["lifetime_ticks"] == n_eps * len(ep)
    assert st_reset["lifetime_episodes"] == st_carry["lifetime_episodes"] == n_eps
    # Identical exposure. "reset" cannot see the spike AT ALL -- every episode's
    # salient tick is the one that seeds the baseline.
    assert st_reset["n_events_converged"] == 0
    assert st_carry["n_events_converged"] > 0


def test_d3b_more_short_episodes_do_not_rescue_reset_mode():
    """779b's finding in miniature: adding budget as MORE SHORT EPISODES does
    not raise the per-episode event yield under "reset". The binding axis is
    episode length, so no step-budget increase can reach it.

    Checked on BOTH shapes, because the two failure modes differ: the early
    spike is invisible (yield 0), the late spike is visible but its yield is
    capped at one per episode. Neither improves with more episodes."""
    for ep, expected in ((_EPISODE_EARLY_SPIKE, 0.0), (_EPISODE_LATE_SPIKE, 1.0)):
        st_20 = _run_episodes(_mk(baseline_continuity="reset"), 20, ep)
        st_60 = _run_episodes(_mk(baseline_continuity="reset"), 60, ep)
        per_ep_20 = st_20["n_events_converged"] / st_20["lifetime_episodes"]
        per_ep_60 = st_60["n_events_converged"] / st_60["lifetime_episodes"]
        assert per_ep_20 == pytest.approx(expected)
        assert per_ep_60 == pytest.approx(per_ep_20), (
            "tripling the budget as more short episodes must not change the "
            "per-episode yield -- if it does, this test no longer models 779b"
        )


# ----------------------------------------------------------------------
# D4 warmup_ticks resolution
# ----------------------------------------------------------------------
def test_d4_warmup_ticks_resolution():
    # -1 derives three EMA time constants.
    pb = _mk(warmup_ticks=-1, surprise_ema_decay=0.1)
    st = pb.get_state()
    assert st["warmup_ticks"] == int(math.ceil(3.0 / 0.1)) == 30
    assert st["warmup_ticks_derived"] is True

    # The derive tracks the decay rather than hardcoding 30.
    assert _mk(warmup_ticks=-1, surprise_ema_decay=0.25).get_state()["warmup_ticks"] == 12

    # Positive is verbatim; 0 is OFF.
    assert _mk(warmup_ticks=7).get_state()["warmup_ticks"] == 7
    assert _mk(warmup_ticks=7).get_state()["warmup_ticks_derived"] is False
    assert _mk(warmup_ticks=0).get_state()["warmup_ticks"] == 0


def test_d4b_convergence_flips_at_the_boundary():
    pb = _mk(baseline_continuity="carry", warmup_ticks=5)
    assert pb.get_state()["baseline_converged"] is False
    for _ in range(4):
        pb.tick(1.0)
    assert pb.get_state()["baseline_converged"] is False
    assert pb.get_state()["n_converged_ticks"] == 0
    pb.tick(1.0)
    st = pb.get_state()
    assert st["baseline_converged"] is True
    assert st["n_prewarmup_ticks"] == 5
    assert st["n_converged_ticks"] == 0
    pb.tick(1.0)
    assert pb.get_state()["n_converged_ticks"] == 1


# ----------------------------------------------------------------------
# D5 the gate is ACCOUNTING ONLY
# ----------------------------------------------------------------------
def test_d5_warmup_does_not_suppress_the_burst():
    """warmup_ticks must not change dynamics -- only bookkeeping. If this ever
    fails, the SD-075 gate has become a second mechanism change and would
    confound the MECH-063 (ii) retest."""
    stream = [1.0, 1.0, 6.0, 1.0, 1.0, 8.0, 1.0]
    pb_off = _mk(baseline_continuity="carry", warmup_ticks=0)
    pb_gate = _mk(baseline_continuity="carry", warmup_ticks=4)
    for s in stream:
        lvl_off = pb_off.tick(s)
        lvl_gate = pb_gate.tick(s)
        assert lvl_gate == lvl_off
        assert pb_gate.temperature_delta == pb_off.temperature_delta
    # Identical dynamics, identical TOTAL events ...
    assert pb_gate.get_state()["n_events"] == pb_off.get_state()["n_events"]
    # ... but the gated instance attributes some of them to warmup.
    assert pb_gate.get_state()["n_events_prewarmup"] > 0
    assert pb_off.get_state()["n_events_prewarmup"] == 0


def test_d5b_agent_action_stream_unchanged_by_warmup_ticks():
    env = _mk_env(0)
    base = _dims(env)
    cfg_a = REEConfig.from_dims(use_phasic_burst=True, **base)
    cfg_b = REEConfig.from_dims(
        use_phasic_burst=True, phasic_burst_warmup_ticks=5, **base
    )
    _, act_a = _run(cfg_a, seed=13)
    _, act_b = _run(cfg_b, seed=13)
    assert act_a == act_b, "warmup_ticks must be behaviourally inert"


# ----------------------------------------------------------------------
# D6 the split partitions the totals
# ----------------------------------------------------------------------
@pytest.mark.parametrize("warmup", [0, 3, 10, -1])
@pytest.mark.parametrize("continuity", ["reset", "carry"])
def test_d6_split_partitions_totals(warmup, continuity):
    pb = _mk(baseline_continuity=continuity, warmup_ticks=warmup)
    st = _run_episodes(pb, 12)
    assert st["n_converged_ticks"] + st["n_prewarmup_ticks"] == st["lifetime_ticks"]
    assert st["lifetime_ticks"] == 12 * len(_EPISODE_LATE_SPIKE)
    # n_events is per-EPISODE (cleared by reset), so it cannot be compared to
    # the lifetime split directly; drive one uninterrupted episode for that.
    pb2 = _mk(baseline_continuity=continuity, warmup_ticks=warmup)
    for s in _EPISODE_LATE_SPIKE * 6:
        pb2.tick(s)
    st2 = pb2.get_state()
    assert st2["n_events_converged"] + st2["n_events_prewarmup"] == st2["n_events"]


# ----------------------------------------------------------------------
# D7 MECH-094 simulation gate
# ----------------------------------------------------------------------
def test_d7_simulation_mode_does_not_advance_lifetime_counters():
    pb = _mk(baseline_continuity="carry", warmup_ticks=5)
    pb.tick(1.0)
    before = pb.get_state()
    for _ in range(10):
        pb.tick(99.0, simulation_mode=True)
    after = pb.get_state()
    assert after["lifetime_ticks"] == before["lifetime_ticks"]
    assert after["n_events_converged"] == before["n_events_converged"]
    assert after["n_events_prewarmup"] == before["n_events_prewarmup"]
    assert after["baseline_converged"] == before["baseline_converged"]
    assert after["surprise_ema"] == before["surprise_ema"]
    assert after["n_simulation_skips"] == 10


# ----------------------------------------------------------------------
# D8 input validation
# ----------------------------------------------------------------------
def test_d8_rejects_bad_continuity():
    with pytest.raises(ValueError, match="baseline_continuity"):
        _mk(baseline_continuity="warm")
    with pytest.raises(ValueError, match="baseline_continuity"):
        _mk(baseline_continuity="Carry")


def test_d8b_rejects_bad_warmup_ticks():
    # -1 is DERIVE; anything below it is a typo and must not silently mean OFF.
    with pytest.raises(ValueError, match="warmup_ticks"):
        _mk(warmup_ticks=-2)
    with pytest.raises(ValueError, match="warmup_ticks"):
        _mk(warmup_ticks=-30)


# ----------------------------------------------------------------------
# D9 agent-level wiring
# ----------------------------------------------------------------------
def test_d9_config_fields_reach_the_regulator():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(
        use_phasic_burst=True,
        phasic_burst_baseline_continuity="carry",
        phasic_burst_warmup_ticks=-1,
        **_dims(env),
    )
    agent, _ = _run(cfg, seed=7)
    c = agent.phasic_burst.config
    assert c.baseline_continuity == "carry"
    assert c.warmup_ticks == -1
    assert agent.phasic_burst.get_state()["warmup_ticks"] == 30


def test_d9b_agent_reset_respects_carry():
    """The agent's episode-boundary reset must not clear a carried baseline."""
    env = _mk_env(0)
    cfg = REEConfig.from_dims(
        use_phasic_burst=True,
        phasic_burst_baseline_continuity="carry",
        phasic_burst_signal_source="instantaneous_pe",
        **_dims(env),
    )
    agent, _ = _run(cfg, seed=7)
    pb = agent.phasic_burst
    for s in (1.0, 1.0, 3.0):
        pb.tick(s)
    ema = pb.get_state()["surprise_ema"]
    assert ema > 0.0
    agent.reset()
    assert pb.get_state()["surprise_ema"] == pytest.approx(ema)
