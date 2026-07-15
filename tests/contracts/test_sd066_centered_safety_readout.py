"""
Contract tests for SD-066 common-mode-invariant (centered) ConditionedSafetyStore
readout (MECH-304 behavioural promote-to-active gate, 2026-07-15).

The raw-cosine store gate saturates under z_world under-differentiation (SD-008):
every z_world sits at cosine ~0.99, so sigmoid(gain*cos) > 0.5 fires for any
non-empty prototype -> the behavioural release gate cannot resolve the cue. The
centered readout subtracts a slow EMA of z_world (the common-mode) before
accumulating and querying the prototype, so the cue-carrying residual dominates.

Five contracts:
  C1  Off by default -- bit-identical to the pre-SD-066 raw-cosine store (a
      reference raw computation reproduces update()'s returns exactly).
  C2  Baseline lifecycle: centered mode advances _baseline over ticks; reset()
      clears both prototype and baseline; raw mode never touches _baseline.
  C3  Common-mode separation: with a large shared common-mode + a small cue on
      one dim present at events, the CENTERED query separates cue-present from
      cue-absent (release-able) while the RAW query saturates (both ~1 -> can't).
  C4  sim_mode gate preserved: update(sim_mode=True) returns 0.0 and advances
      neither prototype nor baseline (MECH-094).
  C5  Config default: REEConfig.safety_store_centered defaults False.
"""

import math

import pytest

from ree_core.safety import ConditionedSafetyStore


def _raw_update_reference(proto, vec, event_fired, ema_alpha, decay_rate, min_norm, gain):
    """Independent re-implementation of the pre-SD-066 raw update() arithmetic."""
    proto = [p * (1.0 - decay_rate) for p in proto]
    if event_fired:
        norm = math.sqrt(sum(x * x for x in vec)) + 1e-8
        normed = [x / norm for x in vec]
        proto = [(1.0 - ema_alpha) * p + ema_alpha * n for p, n in zip(proto, normed)]
    pn = math.sqrt(sum(p * p for p in proto)) + 1e-8
    if pn < min_norm:
        return proto, 0.0
    vn = math.sqrt(sum(x * x for x in vec)) + 1e-8
    dot = sum(p * v for p, v in zip(proto, vec))
    cos = dot / (pn * vn)
    return proto, 1.0 / (1.0 + math.exp(-gain * cos))


# ---------------------------------------------------------------------------
# C1  Off by default -- bit-identical to the raw reference
# ---------------------------------------------------------------------------

def test_c1_off_bit_identical_to_raw_reference():
    store = ConditionedSafetyStore(world_dim=4)   # centered defaults False
    assert store.centered is False
    ref_proto = [0.0] * 4
    seq = [([1.0, 0.0, 0.0, 0.0], True), ([0.0, 1.0, 0.0, 0.0], False),
           ([1.0, 1.0, 0.0, 0.0], True), ([0.5, 0.5, 0.5, 0.0], True)]
    for vec, ev in seq:
        got = store.update(vec, ev)
        ref_proto, exp = _raw_update_reference(
            ref_proto, vec, ev, store.ema_alpha, store.decay_rate, store.min_norm, store.gain)
        assert abs(got - exp) < 1e-12, f"raw path drifted: {got} vs {exp}"


def test_c1_off_never_touches_baseline():
    store = ConditionedSafetyStore(world_dim=4)
    store.update([1.0, 2.0, 3.0, 4.0], True)
    assert store._baseline == [0.0] * 4
    assert store._baseline_seen is False


# ---------------------------------------------------------------------------
# C2  Baseline lifecycle in centered mode
# ---------------------------------------------------------------------------

def test_c2_baseline_advances_and_resets():
    store = ConditionedSafetyStore(world_dim=3, centered=True, baseline_alpha=0.5)
    store.update([2.0, 0.0, 0.0], False)     # first tick -> baseline = vec
    assert store._baseline_seen is True
    assert store._baseline == [2.0, 0.0, 0.0]
    store.update([0.0, 0.0, 0.0], False)     # EMA toward 0 at alpha 0.5
    assert abs(store._baseline[0] - 1.0) < 1e-9
    store.reset()
    assert store._baseline == [0.0] * 3
    assert store._baseline_seen is False
    assert store._prototype == [0.0] * 3


# ---------------------------------------------------------------------------
# C3  Centered readout resolves a cue the raw readout cannot
# ---------------------------------------------------------------------------

def test_c3_centered_separates_common_mode_cue():
    dim = 6
    C = [5.0] * dim                 # large shared common-mode
    cue = [1.0] + [0.0] * (dim - 1)  # small cue on dim 0
    z_event = [c + q for c, q in zip(C, cue)]
    z_cue = list(z_event)
    z_nocue = list(C)

    raw = ConditionedSafetyStore(world_dim=dim, min_norm=0.05)
    ctr = ConditionedSafetyStore(world_dim=dim, min_norm=0.05, centered=True, baseline_alpha=0.3)
    # Teach: many non-event common-mode ticks (build baseline) + event ticks (cue paired).
    for _ in range(40):
        raw.update(C, False)
        ctr.update(C, False)
    for _ in range(20):
        raw.update(z_event, True)
        ctr.update(z_event, True)

    raw_cue = raw.predict(z_cue); raw_nocue = raw.predict(z_nocue)
    ctr_cue = ctr.predict(z_cue); ctr_nocue = ctr.predict(z_nocue)

    # Raw: saturated by the common-mode -> cannot separate (both high, gap tiny).
    assert (raw_cue - raw_nocue) < 0.05
    # Centered: cue-present clearly separates from cue-absent, above a 0.5 gate.
    assert ctr_cue > 0.5
    assert ctr_cue - ctr_nocue > 0.2


# ---------------------------------------------------------------------------
# C4  sim_mode gate preserved (MECH-094)
# ---------------------------------------------------------------------------

def test_c4_sim_mode_gate_preserved():
    store = ConditionedSafetyStore(world_dim=3, centered=True)
    assert store.update([1.0, 1.0, 1.0], True, sim_mode=True) == 0.0
    assert store._prototype == [0.0] * 3
    assert store._baseline_seen is False   # baseline not advanced in sim mode


# ---------------------------------------------------------------------------
# C5  Config default
# ---------------------------------------------------------------------------

def test_c5_config_default_raw():
    from ree_core.utils.config import REEConfig
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=5,
                              use_conditioned_safety_store=True)
    assert cfg.safety_store_centered is False
    assert cfg.safety_store_baseline_alpha == 0.02
