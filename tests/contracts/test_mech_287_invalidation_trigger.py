"""Contract tests for MECH-287 broadcast invalidation trigger.

Phase 2 iv of the V_s invalidation runtime. The trigger is a BoundaryEvent
subscriber (verdict-3 architectural commitment; no independent comparator)
that re-emits MECH-288 BoundaryEvents as graded BroadcastEvent objects with
strength = posterior * gain.

Guarantees enforced here:
  C1. Backward compat: default HippocampalConfig has
      use_invalidation_trigger=False; with the flag off,
      HippocampalModule.invalidation_trigger is None, the broadcast queue
      stays empty, and drain is idempotent.
  C2. A BoundaryEvent arriving at the trigger fires a BroadcastEvent with
      strength == posterior * gain. Source payload (scale,
      segment_id_old/new, sources) is preserved.
  C3. Graded input -> graded output: a sweep of posterior values in
      [0, 1] produces broadcast_strength values equal to posterior * gain
      across the full range. No binary thresholding of strength.
  C4. Tonic guardrail: a synthetic high-tonic-noise period (boundaries
      fired densely enough that the rolling mean exceeds tonic_threshold)
      suppresses the NEXT phasic broadcast. A subsequent quiet period
      lets the tonic estimate decay and broadcast resumes.
  C5. Verdict-3 dissociation: with the event segmenter lesioned (no
      BoundaryEvents ever queued), the trigger never fires a broadcast
      regardless of internal state. This is the falsifiable prediction
      tertiary in MECH-288's claim entry -- the trigger has no independent
      mismatch detector, so silencing the segmenter silences the trigger.
"""

from __future__ import annotations

import pytest


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_boundary_event(
    *,
    posterior: float = 1.0,
    scale: str = "fast",
    segment_id_old: str = "0.0",
    segment_id_new: str = "0.1",
    sources=None,
    t: int = 0,
):
    """Construct a BoundaryEvent without invoking the segmenter."""
    from ree_core.hippocampal.event_segmenter import BoundaryEvent
    return BoundaryEvent(
        segment_id_old=segment_id_old,
        segment_id_new=segment_id_new,
        scale=scale,
        posterior=float(posterior),
        sources=list(sources or ["z_world"]),
        t=int(t),
    )


def _make_trigger(**overrides):
    """Construct an InvalidationTrigger with an overridable default config."""
    from ree_core.regulators.invalidation_trigger import InvalidationTrigger
    from ree_core.utils.config import InvalidationTriggerConfig
    cfg = InvalidationTriggerConfig(**overrides) if overrides else InvalidationTriggerConfig()
    return InvalidationTrigger(cfg)


def _make_module(
    use_event_segmenter: bool = False,
    use_invalidation_trigger: bool = False,
    **trig_overrides,
):
    """Minimal HippocampalModule for agent-level tests."""
    from ree_core.utils.config import (
        HippocampalConfig,
        E2Config,
        ResidueConfig,
        EventSegmenterConfig,
        InvalidationTriggerConfig,
    )
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.residue.field import ResidueField
    from ree_core.hippocampal.module import HippocampalModule

    trig_cfg = (
        InvalidationTriggerConfig(**trig_overrides)
        if trig_overrides
        else InvalidationTriggerConfig()
    )
    hcfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=4,
        hidden_dim=16, horizon=3, num_candidates=4,
        num_cem_iterations=1, elite_fraction=0.5,
        use_event_segmenter=use_event_segmenter,
        event_segmenter=EventSegmenterConfig(),
        use_invalidation_trigger=use_invalidation_trigger,
        invalidation_trigger=trig_cfg,
    )
    e2cfg = E2Config(self_dim=8, world_dim=8, action_dim=4, action_object_dim=4)
    rcfg = ResidueConfig(world_dim=8, num_basis_functions=4)
    e2 = E2FastPredictor(e2cfg)
    residue = ResidueField(rcfg)
    return HippocampalModule(hcfg, e2, residue)


# ------------------------------------------------------------------ #
# C1                                                                 #
# ------------------------------------------------------------------ #

def test_c1_default_config_backward_compatible():
    """C1: flag off by default; module surface is inert."""
    from ree_core.utils.config import HippocampalConfig
    cfg = HippocampalConfig()
    assert getattr(cfg, "use_invalidation_trigger", False) is False

    mod = _make_module(use_invalidation_trigger=False)
    assert mod.invalidation_trigger is None
    assert mod.drain_broadcast_events() == []
    # drain is idempotent and stays empty
    assert mod.drain_broadcast_events() == []

    # Even with the segmenter ON, a trigger-OFF module has no broadcast
    # queue writes and no trigger attribute.
    mod2 = _make_module(use_event_segmenter=True, use_invalidation_trigger=False)
    assert mod2.invalidation_trigger is None
    assert mod2.drain_broadcast_events() == []


# ------------------------------------------------------------------ #
# C2                                                                 #
# ------------------------------------------------------------------ #

def test_c2_boundary_event_fires_trigger_strength_equals_posterior_times_gain():
    """C2: a BoundaryEvent causes a BroadcastEvent with strength = posterior * gain."""
    trig = _make_trigger(gain=2.0, tonic_threshold=10.0)  # tonic gate disabled

    ev = _make_boundary_event(posterior=0.7, scale="fast",
                              segment_id_old="0.0", segment_id_new="0.1",
                              sources=["z_world", "z_self"], t=5)
    out = trig.step(boundary_events=[ev], t=5)

    assert len(out) == 1, "expected exactly one BroadcastEvent"
    bcast = out[0]
    assert bcast.strength == pytest.approx(0.7 * 2.0)
    assert bcast.posterior == pytest.approx(0.7)
    assert bcast.t == 5
    assert bcast.source_scale == "fast"
    assert bcast.source_segment_id_old == "0.0"
    assert bcast.source_segment_id_new == "0.1"
    assert bcast.source_sources == ["z_world", "z_self"]
    # Default targets from config.
    assert bcast.targets == ["mech_269_anchor_set"]

    stats = trig.get_stats()
    assert stats["n_broadcast"] == 1
    assert stats["n_suppressed"] == 0


# ------------------------------------------------------------------ #
# C3                                                                 #
# ------------------------------------------------------------------ #

def test_c3_graded_posterior_to_graded_broadcast_no_binary_threshold():
    """C3: sweep posterior in [0, 1]; strength = posterior * gain for every value.

    Binary thresholding (e.g. fire-iff-posterior>0.5) would produce a step
    function or would drop the below-threshold cases. This test verifies
    the strength is linear in posterior across the full range.
    """
    gain = 1.5
    # Run each posterior through its OWN trigger instance with the tonic
    # gate disabled, so per-value state does not leak across the sweep.
    posteriors = [0.01, 0.1, 0.25, 0.49, 0.5, 0.51, 0.75, 0.99, 1.0]
    for i, p in enumerate(posteriors):
        trig = _make_trigger(gain=gain, tonic_threshold=10.0, tonic_window=4)
        ev = _make_boundary_event(posterior=p, t=i)
        out = trig.step(boundary_events=[ev], t=i)
        assert len(out) == 1, f"posterior={p} produced no broadcast (binary threshold?)"
        assert out[0].strength == pytest.approx(p * gain), (
            f"posterior={p}: expected {p * gain}, got {out[0].strength}"
        )


# ------------------------------------------------------------------ #
# C4                                                                 #
# ------------------------------------------------------------------ #

def test_c4_tonic_guardrail_suppresses_next_phasic():
    """C4: high-tonic period suppresses next phasic; quiet period reopens the gate.

    Aston-Jones & Cohen 2005 phasic/tonic dissociation: sustained elevated
    activity signatures a tonic regime in which additional phasic bursts
    are inverted-U suppressed. Our implementation: when the rolling-mean
    of past boundary posteriors exceeds tonic_threshold, the whole next
    tick is suppressed.
    """
    # Window 4, threshold 0.5: four consecutive ticks at posterior 1.0 give
    # a rolling mean of 1.0 (well above 0.5) by tick 5, suppressing it.
    trig = _make_trigger(gain=1.0, tonic_threshold=0.5, tonic_window=4)

    # Burst: 4 ticks each with a posterior-1.0 boundary. Every one fires
    # because the tonic estimate (from empty history) is below threshold
    # at tick 0, and the rolling mean only reaches threshold after the
    # fourth tick has been appended.
    for t in range(4):
        ev = _make_boundary_event(posterior=1.0, t=t)
        out = trig.step(boundary_events=[ev], t=t)
        # First tick: history empty, tonic=0 -> fire.
        # Subsequent ticks: the rolling mean is computed from history
        # BEFORE the current tick is appended. History at tick 1 = [1.0];
        # at tick 2 = [1.0, 1.0]; both under 0.5? No -- mean of a
        # single 1.0 is 1.0, so ticks 1+ are already under the gate. The
        # expectation is that tick 0 fires AND the very next tick (tick 1)
        # is the first to see the high tonic.
        pass

    # By tick 1, tonic estimate >= 1.0 (mean of [1.0] from tick 0's
    # appended activity). Test that tick 5 is suppressed.
    ev_next = _make_boundary_event(posterior=1.0, t=5)
    out = trig.step(boundary_events=[ev_next], t=5)
    assert out == [], "high-tonic next phasic must be suppressed"
    assert trig.get_stats()["n_suppressed"] >= 1
    # Counters: tonic was already high well before this tick, so most of
    # the burst ticks 1..3 should also have been suppressed. n_broadcast
    # should reflect only the first tick (tick 0, tonic was zero).
    assert trig.get_stats()["n_broadcast"] == 1

    # Quiet period: tick empty boundaries so the rolling window fills
    # with zeros. tonic_window=4 plus the starting tick-5 activity means
    # we need tonic_window+1 quiet ticks for the history to be fully
    # zeroed by the next step()'s mean computation.
    for t in range(6, 6 + trig.config.tonic_window + 1):
        trig.step(boundary_events=[], t=t)
    assert trig.tonic_estimate == pytest.approx(0.0), (
        "after tonic_window+1 quiet ticks, tonic estimate must decay to 0"
    )

    # A subsequent boundary now fires.
    t_after = 6 + trig.config.tonic_window + 1
    ev_after = _make_boundary_event(posterior=1.0, t=t_after)
    out = trig.step(boundary_events=[ev_after], t=t_after)
    assert len(out) == 1, "after tonic decay, broadcast must resume"
    assert out[0].strength == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# C5                                                                 #
# ------------------------------------------------------------------ #

def test_c5_verdict_3_dissociation_no_events_no_broadcast():
    """C5: event_segmenter lesioned -> trigger silent regardless of state.

    Verdict 3 (option c) of the V_s foundation lit-pull: MECH-287 is
    implemented as a BoundaryEvent subscriber with no independent
    comparator stage. This is the falsifiable prediction tertiary in
    MECH-288's claim entry -- if lesioning MECH-288 does NOT silence
    MECH-287, the implementation has an independent mismatch detector
    inconsistent with the architectural commitment.

    Test protocol:
      - unit arm: ticks the trigger directly with an empty boundary list
        for many ticks; manipulates internal state (tonic history filled
        with synthetic activity) to probe whether any internal trigger
        can fire a broadcast in the absence of boundary input.
      - agent arm: constructs a HippocampalModule with the segmenter OFF
        and the trigger ON, confirms the broadcast queue stays empty
        across multiple simulated ticks.
    """
    # --- Unit arm ---
    trig = _make_trigger(gain=1.0, tonic_threshold=0.5, tonic_window=4)

    # Empty boundary list for 100 ticks -- never fires.
    for t in range(100):
        out = trig.step(boundary_events=[], t=t)
        assert out == [], f"tick {t}: trigger fired with no boundary events"
    assert trig.get_stats()["n_broadcast"] == 0
    assert trig.get_stats()["n_suppressed"] == 0

    # Manually seed the tonic history to a synthetic high-tonic state.
    # Even with the tonic estimate above threshold, an empty boundary
    # list must produce no broadcasts (the gate is a suppressor, not an
    # independent fire source).
    for _ in range(trig.config.tonic_window):
        trig._tonic_history.append(10.0)
    for t in range(100, 110):
        out = trig.step(boundary_events=[], t=t)
        assert out == [], "trigger with high tonic state MUST stay silent with no boundaries"
    assert trig.get_stats()["n_broadcast"] == 0

    # --- Agent arm ---
    # Segmenter OFF, trigger ON -- HippocampalModule construction still
    # succeeds; drain_broadcast_events stays empty across simulated
    # agent ticks (we simulate by calling step on the trigger via the
    # module's reference with empty events, matching the path agent.sense
    # takes in the elif branch).
    mod = _make_module(
        use_event_segmenter=False,
        use_invalidation_trigger=True,
        gain=1.0,
        tonic_threshold=0.5,
        tonic_window=4,
    )
    assert mod.event_segmenter is None
    assert mod.invalidation_trigger is not None

    for t in range(50):
        # Mirrors agent.sense elif branch: no segmenter -> empty list.
        mod.invalidation_trigger.step(boundary_events=[], t=t)
    assert mod.drain_broadcast_events() == [], (
        "with segmenter lesioned, broadcast queue MUST stay empty"
    )
    assert mod.invalidation_trigger.get_stats()["n_broadcast"] == 0
