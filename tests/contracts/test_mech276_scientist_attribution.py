"""Contract tests for MECH-276 scientist-agent counterfactual-backed attribution.

Interface-level guarantees that should hold regardless of tuning:
  C1  default OFF -> agent.scientist_attribution_buffer is None;
      bit-identical action stream (no RNG perturbation, the buffer reads
      latents but emits no bias).
  C2  precondition: use_scientist_attribution=True with NO comparator
      (use_e2_world_forward / use_e2_harm_s_forward both off) raises (loud).
  C3  buffer records counterfactual-backed attributions; SKIPS correlational
      (contrast < cf_margin) under only_counterfactual_backed; MECH-094 sim no-op.
  C4  evidence_snapshot is region-merged across domains + carries the
      GLOBAL_REGION sentinel; lookup falls back to the global mean.
  C5  correlational-control arm (only_counterfactual_backed=False) buffers
      everything -- the predicted noise-fit input to the aggregator.
  C6  ScientistAttributionConfig validation (loud on bad values).
  C7  agent activation (e2_world @ dim128) records over waking steps; the
      buffer PERSISTS across agent.reset() (cross-episode aggregation) while the
      prev-latent caches clear.
  C8  sleep-loop integration: _build_evidence_snapshot prefers the buffer when
      present (REPLACES staleness); _lookup_evidence honors the global sentinel;
      legacy staleness path bit-identical when the buffer is absent.
  C9  decay_cycle applies the per-cycle multiplicative decay to region EMAs.
"""

import torch

from ree_core.attribution.scientist_attribution_buffer import (
    ScientistAttributionBuffer,
    ScientistAttributionConfig,
)
from ree_core.sleep.phase_manager import SleepLoopManager
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _build(env, **kw):
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        **kw,
    )


# The comparator is held FIXED across both arms of the C1 control. The no-op
# claim under test is about `use_scientist_attribution` alone, so the OFF arm
# must already carry the comparator -- otherwise the contrast flips TWO flags
# at once and measures the comparator's effect, not attribution's.
#
# That confound made C1 a cross-machine-class failure: adding the comparator
# module changes how much of the global RNG stream the forward pass consumes,
# and the committed action is drawn with torch.multinomial, which is NOT
# reproducible across machine classes (linux-x86_64 vs darwin-arm64 return
# different categories from a bit-identical probability tensor at the same
# seed). The Mac absorbed the shift and the fleet did not. Holding the
# comparator fixed makes the arms differ only in attribution, and the streams
# then match bit-for-bit on BOTH classes (verified 2026-07-19).
_COMPARATOR = {"use_e2_world_forward": False, "use_e2_harm_s_forward": True}


def _action_stream(use_sci, n=20):
    torch.manual_seed(321)
    env = CausalGridWorldV2(size=8, seed=11)
    kw = dict(_COMPARATOR)
    if use_sci:
        # ON with a comparator, but at default world_dim=32 the e2_world
        # comparator is NOT attribution_ready, so the buffer records nothing and
        # emits no bias -- the action stream must stay bit-identical.
        kw["use_scientist_attribution"] = True
    ag = REEAgent(_build(env, **kw))
    _, od = env.reset()
    acts = []
    for _ in range(n):
        a = ag.act_with_split_obs(od["body_state"], od["world_state"])
        acts.append(int(a.argmax()))
        _, h, d, inf, od = env.step(a)
        if d:
            _, od = env.reset()
            ag.reset()
    return acts


# ---- a routed-event stub for the phase_manager lookup helpers ----
class _Ev:
    def __init__(self, key):
        self.key = key


class _Routed:
    def __init__(self, key):
        self.event = _Ev(key)


# ---------------------------------------------------------------- C1
def test_c1_default_off_no_op():
    env = CausalGridWorldV2(size=8, seed=0)
    ag = REEAgent(_build(env))
    _, od = env.reset()
    ag.sense(od["body_state"], od["world_state"])
    assert ag.scientist_attribution_buffer is None
    # ON-with-comparator-but-no-attribution-ready must not perturb the stream.
    assert _action_stream(False) == _action_stream(True)


# ---------------------------------------------------------------- C2
def test_c2_precondition_raises_without_comparator():
    env = CausalGridWorldV2(size=8, seed=0)
    raised = False
    try:
        REEAgent(_build(env, use_scientist_attribution=True))
    except ValueError as e:
        raised = "comparator" in str(e)
    assert raised, "use_scientist_attribution without a comparator must raise"


# ---------------------------------------------------------------- C3
def test_c3_record_counterfactual_backed_skips_correlational_sim_noop():
    b = ScientistAttributionBuffer(ScientistAttributionConfig(cf_margin=0.05))
    assert b.record("place", ("fast", "0.1"), 2.0, 0.5) is True   # cf-backed
    assert b.record("self", ("fast", "0.1"), 1.0, 0.9) is True    # cf-backed
    assert b.record("place", ("fast", "0.2"), 9.0, 0.0) is False  # correlational skip
    assert b.record("place", ("fast", "0.1"), 5.0, 0.5, simulation_mode=True) is False
    assert b.n_records == 2
    assert b.n_counterfactual_backed == 2
    assert b.n_correlational_skipped == 1
    m = b.get_metrics()
    assert m["mech276_n_simulation_skipped"] == 1.0
    assert m["mech276_counterfactual_backed_fraction"] == 1.0


# ---------------------------------------------------------------- C4
def test_c4_evidence_snapshot_region_merged_global_sentinel():
    b = ScientistAttributionBuffer()
    b.record("place", ("fast", "0.1"), 2.0, 0.9)
    b.record("self", ("fast", "0.1"), 4.0, 0.9)   # same region, other domain
    b.record("place", ("fast", "0.2"), 6.0, 0.9)
    snap = b.evidence_snapshot()
    # region (fast,0.1) merges the two domain EMAs (mean 3.0).
    assert abs(snap[("fast", "0.1")] - 3.0) < 1e-9
    assert ("__global__", "") in snap
    # lookup falls back to the global mean for an absent region.
    miss = b.lookup(("fast", "9.9"), snap)
    assert abs(miss - snap[("__global__", "")]) < 1e-9
    # empty buffer -> empty snapshot, lookup 0.0.
    assert ScientistAttributionBuffer().evidence_snapshot() == {}
    assert ScientistAttributionBuffer().lookup(("a", "0")) == 0.0


# ---------------------------------------------------------------- C5
def test_c5_correlational_control_arm_buffers_everything():
    b = ScientistAttributionBuffer(
        ScientistAttributionConfig(only_counterfactual_backed=False, cf_margin=0.05)
    )
    assert b.record("place", ("a", "0"), 1.0, 0.0) is True  # correlational, kept
    assert b.record("place", ("a", "1"), 1.0, 0.9) is True  # cf-backed, kept
    assert b.n_records == 2
    assert b.n_counterfactual_backed == 1  # only one cleared the margin
    assert b.n_correlational_skipped == 0


# ---------------------------------------------------------------- C6
def test_c6_config_validation():
    for bad in (dict(ema_alpha=0.0), dict(ema_alpha=1.5), dict(decay=0.0),
                dict(cf_margin=-0.1)):
        raised = False
        try:
            ScientistAttributionConfig(**bad)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad}"


# ---------------------------------------------------------------- C7
def test_c7_agent_activation_and_cross_episode_persistence():
    env = CausalGridWorldV2(size=8, seed=7)
    _, od = env.reset()
    ag = REEAgent(_build(env, world_dim=128, use_e2_world_forward=True,
                         use_scientist_attribution=True))
    assert ag.scientist_attribution_buffer is not None
    assert ag.e2_world is not None and ag.e2_world.attribution_ready
    torch.manual_seed(5)
    for _ in range(15):
        a = ag.act_with_split_obs(od["body_state"], od["world_state"])
        _, h, d, inf, od = env.step(a)
        if d:
            _, od = env.reset()
            ag.reset()
    buf = ag.scientist_attribution_buffer
    assert buf.n_records > 0
    n_before = buf.n_records
    # reset() clears the prev-latent caches but the buffer persists cross-episode.
    ag.reset()
    assert ag._sci_prev_z_world is None and ag._sci_prev_z_harm_s is None
    assert ag.scientist_attribution_buffer.n_records == n_before


# ---------------------------------------------------------------- C8
def test_c8_sleep_loop_evidence_source_select_and_legacy_bit_identical():
    # Buffer present -> _build_evidence_snapshot sources the MECH-276 feedstock.
    class _AgentWithBuf:
        def __init__(self, buf):
            self.scientist_attribution_buffer = buf
            self.hippocampal = None

    b = ScientistAttributionBuffer()
    b.record("place", ("fast", "0.1"), 3.0, 0.9)
    snap = SleepLoopManager._build_evidence_snapshot(_AgentWithBuf(b))
    assert ("fast", "0.1") in snap and ("__global__", "") in snap
    # Buffer absent + no staleness -> legacy empty (bit-identical pre-MECH-276).
    assert SleepLoopManager._build_evidence_snapshot(_AgentWithBuf(None)) == {}

    # _lookup_evidence: sentinel fallback for a routed region missing from the
    # MECH-276 snapshot; staleness snapshot (no sentinel) falls back to 0.0.
    sci_snap = {("fast", "0.1"): 2.0, ("__global__", ""): 1.5}
    assert SleepLoopManager._lookup_evidence(_Routed(("fast", "0.1")), sci_snap) == 2.0
    assert SleepLoopManager._lookup_evidence(_Routed(("fast", "9.9")), sci_snap) == 1.5
    stale_snap = {("fast", "0.1"): 2.0}
    assert SleepLoopManager._lookup_evidence(_Routed(("fast", "9.9")), stale_snap) == 0.0


# ---------------------------------------------------------------- C9
def test_c9_decay_cycle():
    b = ScientistAttributionBuffer(ScientistAttributionConfig(decay=0.5))
    b.record("place", ("fast", "0.1"), 4.0, 0.9)
    before = b.evidence_snapshot()[("fast", "0.1")]
    b.decay_cycle()
    after = b.evidence_snapshot()[("fast", "0.1")]
    assert abs(after - 0.5 * before) < 1e-9
    # decay=1.0 is a no-op.
    b2 = ScientistAttributionBuffer(ScientistAttributionConfig(decay=1.0))
    b2.record("place", ("fast", "0.1"), 4.0, 0.9)
    v = b2.evidence_snapshot()[("fast", "0.1")]
    b2.decay_cycle()
    assert abs(b2.evidence_snapshot()[("fast", "0.1")] - v) < 1e-9
