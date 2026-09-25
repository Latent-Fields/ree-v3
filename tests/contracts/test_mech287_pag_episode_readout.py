"""Contract tests for the MECH-287 phase-scoped PAG re-commit readout.

THE DEFECT THIS INSTRUMENT REPAIRS (analysis:
REE_assembly/evidence/planning/mech287_dv_instrument_confound_20260924.md):

  PAGFreezeGate.reset() clears the per-episode freeze state but NOT
  _n_ticks / _n_commits / _n_releases, and REEAgent.reset() calls it once per
  episode. The counters are therefore cumulative across warmup+eval. An
  episode that ends while frozen consumes a commit with no matching release,
  so n_commits - n_releases <= n_episodes and the naive ratio
  n_commits / n_releases is dominated by EPISODE COUNT, not by re-commit
  behaviour after a release. V3-EXQ-475's "~12.9 re-commits per release"
  (71/6, 70/5, 64/5 over 65 episodes) is that artifact.

  Separately, agent.reset() calls reset_invalidation_trigger() and
  reset_staleness_accumulator(), so a post-loop get_stats() read sees only the
  LAST episode.

Guarantees enforced here:
  C1.  PAGEpisodeRecord importable and exported.
  C2.  Per-episode readout equals a HAND COUNT on a scripted sequence.
  C3.  An episode ending frozen is counted ended_frozen, NOT as a release.
  C4.  BLIND SPOT: the naive cumulative ratio scales with episode count while
       the repaired DV does not. Fails against the old cumulative read.
  C5.  Backward compat: the cumulative counters keep their exact semantics.
  C6.  Phase scoping isolates eval from warmup.
  C7.  Negative instrument: dv_measurable is False when no release occurred.
  C8.  The final, never-reset episode is still counted.
  C9.  reset_diagnostics() zeroes the cumulative counters too.
  C10. Agent snapshots are captured BEFORE reset erases the source counters.
  C11. Agent snapshot is exactly-once per episode.
  C12. Agent readout reports instrument_present=False when nothing is built.
  C13. Reading the readout does not mutate gate state (read-only instrument).
"""

from __future__ import annotations

import pytest


# theta_freeze=2.0, duration_input_threshold=0.4, gaba_tone=1.0
#   -> exit_threshold = 2.0
#   -> z=2.5 commits on its FIRST tick (2.5 * 1 > 2.0) and does NOT release
#      (2.5 is not < 2.0); z=0.0 releases on the next tick.
HIGH = 2.5
LOW = 0.0

# One commit, one release, one RE-commit, one release.
EP_TWO_CYCLES = [HIGH, LOW, HIGH, LOW]
# One commit, never released -- the episode ends still frozen.
EP_ENDS_FROZEN = [HIGH, HIGH, HIGH]


def _gate(**overrides):
    from ree_core.pag import PAGFreezeGate, PAGFreezeGateConfig
    return PAGFreezeGate(PAGFreezeGateConfig(**overrides))


def _drive(gate, seq):
    for z in seq:
        gate.tick(z_harm_a_norm=z, gaba_tone=1.0)


def _run_episodes(gate, seq, n):
    for _ in range(n):
        _drive(gate, seq)
        gate.reset()


def _naive_ratio(gate):
    """The OLD, confounded read: cumulative commits over cumulative releases."""
    d = gate.diagnostics
    return d["n_commits"] / max(d["n_releases"], 1)


# ------------------------------------------------------------------ #
# C1                                                                 #
# ------------------------------------------------------------------ #

def test_c1_episode_record_exported():
    from ree_core.pag import PAGEpisodeRecord
    rec = PAGEpisodeRecord(index=1, phase="eval", ticks=4, commits=2,
                           releases=1, recommits=1, ended_frozen=True)
    assert rec.as_dict()["recommits"] == 1
    assert rec.as_dict()["ended_frozen"] is True


# ------------------------------------------------------------------ #
# C2 -- hand count                                                   #
# ------------------------------------------------------------------ #

def test_c2_per_episode_readout_equals_hand_count():
    """Scripted freeze/release sequence, counted by hand.

    EP_TWO_CYCLES = [HIGH, LOW, HIGH, LOW]:
      t1 HIGH -> duration=1, 2.5*1 > 2.0        -> COMMIT   (not a re-commit)
      t2 LOW  -> 0.0 < 2.0                      -> RELEASE
      t3 HIGH -> duration=1 again               -> COMMIT   (IS a re-commit)
      t4 LOW                                    -> RELEASE
    Hand count per episode: commits=2, releases=2, recommits=1, ended_frozen=False.
    """
    gate = _gate()
    _run_episodes(gate, EP_TWO_CYCLES, 3)

    ep = gate.episode_diagnostics()
    assert ep["n_episodes"] == 3
    assert ep["commits"] == 6          # 2 per episode
    assert ep["releases"] == 6         # 2 per episode
    assert ep["recommits"] == 3        # exactly 1 per episode
    assert ep["episodes_ending_frozen"] == 0
    assert ep["recommits_per_release"] == pytest.approx(3.0 / 6.0)

    rows = gate.episode_records()
    assert len(rows) == 3
    for r in rows:
        assert (r["commits"], r["releases"], r["recommits"]) == (2, 2, 1)
        assert r["ended_frozen"] is False


# ------------------------------------------------------------------ #
# C3 -- ends-frozen is not a release                                 #
# ------------------------------------------------------------------ #

def test_c3_episode_ending_frozen_is_not_counted_as_a_release():
    gate = _gate()
    _run_episodes(gate, EP_ENDS_FROZEN, 4)

    ep = gate.episode_diagnostics()
    assert ep["n_episodes"] == 4
    assert ep["commits"] == 4
    assert ep["releases"] == 0, "an episode ending frozen must not fabricate a release"
    assert ep["recommits"] == 0
    assert ep["episodes_ending_frozen"] == 4
    assert ep["frac_episodes_ending_frozen"] == pytest.approx(1.0)
    assert all(r["ended_frozen"] is True for r in gate.episode_records())

    # And the cumulative counters show exactly the confound signature:
    # every commit is unmatched, so the difference IS the episode count.
    d = gate.diagnostics
    assert d["n_commits"] - d["n_releases"] == ep["n_episodes"]


# ------------------------------------------------------------------ #
# C4 -- THE BLIND SPOT                                               #
# ------------------------------------------------------------------ #

def test_c4_blind_spot_naive_ratio_scales_with_episode_count_repaired_does_not():
    """The measurement that FAILS against the old cumulative read.

    Same per-episode freeze behaviour, three different episode counts. The
    naive cumulative ratio tracks the episode count exactly; the repaired DV
    is invariant. Computing `recommits_per_release` the old way (n_commits /
    n_releases) makes the second assertion fail.
    """
    naive_by_n = {}
    repaired_by_n = {}
    for n in (5, 13, 65):
        gate = _gate()
        _run_episodes(gate, EP_ENDS_FROZEN, n)
        naive_by_n[n] = _naive_ratio(gate)
        repaired_by_n[n] = gate.episode_diagnostics()["recommits_per_release"]

    # The defect, pinned so it cannot be quietly "fixed" out of the cumulative
    # counters without this test noticing: the naive ratio IS the episode count.
    assert naive_by_n == {5: 5.0, 13: 13.0, 65: 65.0}
    assert len(set(naive_by_n.values())) == 3, "naive read must move with episode count"

    # The repaired DV does not move with episode count.
    assert len(set(repaired_by_n.values())) == 1, (
        "repaired DV moved with episode count -- it is still reading the "
        "cumulative counters: %r" % (repaired_by_n,)
    )


def test_c4b_blind_spot_with_real_recommits_present():
    """Same, where genuine re-commits DO occur, so the DV is non-zero.

    Per episode: COMMIT, RELEASE, RE-COMMIT, then stay frozen to the end.
    Hand count: commits=2, releases=1, recommits=1, ended_frozen=True.
    Repaired recommits_per_release is 1.0 at every episode count; the naive
    ratio climbs from 2.0 toward 2.0 only because commits and the unmatched
    tail scale together -- it reports a re-commit rate that no episode had.
    """
    ep_seq = [HIGH, LOW, HIGH, HIGH]
    repaired = set()
    for n in (5, 13, 65):
        gate = _gate()
        _run_episodes(gate, ep_seq, n)
        d = gate.diagnostics
        ep = gate.episode_diagnostics()
        assert ep["commits"] == 2 * n
        assert ep["releases"] == n
        assert ep["recommits"] == n
        assert ep["episodes_ending_frozen"] == n
        # The confound signature again.
        assert d["n_commits"] - d["n_releases"] == n
        repaired.add(ep["recommits_per_release"])
    assert repaired == {1.0}


# ------------------------------------------------------------------ #
# C5 -- backward compatibility                                       #
# ------------------------------------------------------------------ #

def test_c5_cumulative_counters_semantics_unchanged():
    """reset() must still NOT clear the cumulative counters.

    Prior experiments (V3-EXQ-475/483a/603i/603p/603t) read pag_n_commits /
    pag_n_releases / pag_n_ticks with exactly this meaning. The readout is
    additive; it must not silently re-point those three numbers.
    """
    gate = _gate()
    _run_episodes(gate, EP_TWO_CYCLES, 3)
    d = gate.diagnostics
    assert d["n_ticks"] == 12          # 4 ticks x 3 episodes, never zeroed
    assert d["n_commits"] == 6
    assert d["n_releases"] == 6
    # Per-episode state IS cleared, exactly as before.
    assert d["freeze_active"] is False
    assert d["duration_above_threshold"] == 0
    assert d["ticks_in_freeze"] == 0
    # And the legacy keys are all still present.
    for k in ("n_ticks", "n_commits", "n_releases", "freeze_active",
              "duration_above_threshold", "ticks_in_freeze"):
        assert k in d


def test_c5b_readout_keys_surfaced_on_diagnostics():
    """Manifest writers dump `diagnostics` wholesale -- the repaired DV rides along."""
    gate = _gate()
    _run_episodes(gate, EP_TWO_CYCLES, 2)
    d = gate.diagnostics
    assert d["episode_n_episodes"] == 2
    assert d["episode_recommits"] == 2
    assert d["episode_recommits_per_release"] == pytest.approx(0.5)
    assert d["episode_episodes_ending_frozen"] == 0


# ------------------------------------------------------------------ #
# C6 -- phase scoping                                                #
# ------------------------------------------------------------------ #

def test_c6_phase_scoping_isolates_eval_from_warmup():
    """The V3-EXQ-475 shape: 60 warmup episodes then 5 eval episodes."""
    gate = _gate()
    gate.set_phase("warmup")
    _run_episodes(gate, EP_ENDS_FROZEN, 60)
    gate.set_phase("eval")
    _run_episodes(gate, EP_TWO_CYCLES, 5)

    assert gate.episode_phases == ["warmup", "eval"]

    ev = gate.episode_diagnostics(phase="eval")
    assert ev["n_episodes"] == 5
    assert ev["recommits"] == 5
    assert ev["releases"] == 10
    assert ev["recommits_per_release"] == pytest.approx(0.5)
    assert ev["episodes_ending_frozen"] == 0

    wu = gate.episode_diagnostics(phase="warmup")
    assert wu["n_episodes"] == 60
    assert wu["releases"] == 0
    assert wu["dv_measurable"] is False

    both = gate.episode_diagnostics()
    assert both["n_episodes"] == 65
    # The all-phase read is NOT the eval read -- which is the whole point.
    assert both["commits"] != ev["commits"]


# ------------------------------------------------------------------ #
# C7 -- negative instrument                                          #
# ------------------------------------------------------------------ #

def test_c7_dv_measurable_false_when_no_release_ever_occurred():
    """"0.0 re-commits per release" must not be confusable with "never measured"."""
    gate = _gate()
    _run_episodes(gate, EP_ENDS_FROZEN, 10)
    ep = gate.episode_diagnostics()
    assert ep["releases"] == 0
    assert ep["recommits_per_release"] == 0.0
    assert ep["dv_measurable"] is False, (
        "no release ever occurred, so the DV had no occasion to be measured; "
        "reporting 0.0 without this flag reads as a measured null"
    )

    gate2 = _gate()
    _run_episodes(gate2, EP_TWO_CYCLES, 2)
    assert gate2.episode_diagnostics()["dv_measurable"] is True


# ------------------------------------------------------------------ #
# C8 -- the last episode                                             #
# ------------------------------------------------------------------ #

def test_c8_final_never_reset_episode_is_still_counted():
    gate = _gate()
    _run_episodes(gate, EP_TWO_CYCLES, 2)
    _drive(gate, EP_TWO_CYCLES)          # third episode, deliberately not reset

    assert gate.episode_diagnostics()["n_episodes"] == 3
    assert gate.episode_diagnostics(include_current=False)["n_episodes"] == 2
    assert gate.episode_diagnostics()["recommits"] == 3


def test_c8b_reset_before_any_tick_records_no_phantom_episode():
    gate = _gate()
    gate.reset()
    gate.reset()
    assert gate.episode_diagnostics()["n_episodes"] == 0
    _drive(gate, EP_TWO_CYCLES)
    gate.reset()
    gate.reset()
    assert gate.episode_diagnostics()["n_episodes"] == 1


# ------------------------------------------------------------------ #
# C9 -- explicit eval-phase reset                                    #
# ------------------------------------------------------------------ #

def test_c9_reset_diagnostics_zeroes_cumulative_counters():
    gate = _gate()
    _run_episodes(gate, EP_TWO_CYCLES, 3)
    assert gate.diagnostics["n_commits"] == 6

    gate.reset_diagnostics()
    d = gate.diagnostics
    assert (d["n_ticks"], d["n_commits"], d["n_releases"]) == (0, 0, 0)
    assert gate.episode_diagnostics()["n_episodes"] == 0

    # Post-reset accumulation describes the new phase alone.
    _run_episodes(gate, EP_TWO_CYCLES, 2)
    assert gate.diagnostics["n_commits"] == 4
    assert gate.episode_diagnostics()["n_episodes"] == 2


# ------------------------------------------------------------------ #
# C10-C12 -- agent-level snapshots                                   #
# ------------------------------------------------------------------ #

def _agent_with_instruments():
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_staleness_accumulator=True,
        use_anchor_sets=True,
    )
    agent = REEAgent(cfg)
    # Guard against from_dims silently swallowing an unknown kwarg (MECH-307):
    # if these are None the test below would pass vacuously.
    assert agent.hippocampal is not None
    assert agent.hippocampal.invalidation_trigger is not None, (
        "use_invalidation_trigger did not reach HippocampalModule -- the "
        "snapshot test would be vacuous"
    )
    return agent


def test_c10_snapshot_is_captured_before_reset_erases_the_source():
    """The ordering contract. Fails if the capture moves below the resets."""
    agent = _agent_with_instruments()
    trig = agent.hippocampal.invalidation_trigger

    trig._n_broadcast = 7
    trig._n_suppressed = 2
    agent._step_count = 11               # an episode ran

    agent.reset()

    # The source really was erased by reset ...
    assert trig.get_stats()["n_broadcast"] == 0
    # ... and the pre-erase value survived in the snapshot.
    rows = agent.get_mech287_episode_records()
    assert len(rows) == 1
    assert rows[0]["n_broadcast"] == 7
    assert rows[0]["n_suppressed"] == 2

    ro = agent.get_mech287_episode_readout()
    assert ro["n_episodes"] == 1
    assert ro["n_broadcast_total"] == 7
    assert ro["episodes_with_broadcast"] == 1
    assert ro["instrument_present"] is True


def test_c10b_peak_across_episodes_survives_where_a_post_loop_read_cannot():
    """A post-loop get_stats() sees only the last episode. The readout does not."""
    agent = _agent_with_instruments()
    trig = agent.hippocampal.invalidation_trigger

    for n in (5, 9, 0):                  # the loud episodes come first
        trig._n_broadcast = n
        agent._step_count = 4
        agent.reset()

    assert trig.get_stats()["n_broadcast"] == 0          # the post-loop read
    ro = agent.get_mech287_episode_readout()
    assert ro["n_episodes"] == 3
    assert ro["n_broadcast_total"] == 14
    assert ro["broadcast_peak_per_episode"] == 9
    assert ro["episodes_with_broadcast"] == 2


def test_c11_snapshot_is_exactly_once_per_episode():
    agent = _agent_with_instruments()
    agent.hippocampal.invalidation_trigger._n_broadcast = 3

    agent._step_count = 6
    agent.capture_mech287_episode_snapshot()     # explicit end-of-loop capture
    agent._step_count = 6
    agent.reset()                                # must NOT record it again
    assert agent.get_mech287_episode_readout()["n_episodes"] == 1

    # A reset with no ticks since the last one records no phantom episode.
    agent._step_count = 0
    agent.reset()
    assert agent.get_mech287_episode_readout()["n_episodes"] == 1

    # The next real episode is recorded normally.
    agent._step_count = 3
    agent.reset()
    assert agent.get_mech287_episode_readout()["n_episodes"] == 2


def test_c12_instrument_absent_is_distinguishable_from_a_measured_zero():
    """With the flags off nothing is constructed, so 0 is BY CONSTRUCTION."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    agent = REEAgent(REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4))
    agent._step_count = 5
    agent.reset()

    ro = agent.get_mech287_episode_readout()
    assert ro["instrument_present"] is False
    assert ro["n_episodes"] == 0
    assert ro["n_broadcast_total"] == 0


def test_c12b_agent_phase_label_forwards_to_the_freeze_gate():
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_pag_freeze_gate=True,
    )
    agent = REEAgent(cfg)
    assert agent.pag_freeze_gate is not None
    agent.set_mech287_phase("eval")
    assert agent._mech287_phase == "eval"
    assert agent.pag_freeze_gate._phase == "eval"


# ------------------------------------------------------------------ #
# C13 -- the instrument is read-only                                 #
# ------------------------------------------------------------------ #

def test_c13_reading_the_readout_does_not_mutate_state():
    gate = _gate()
    _run_episodes(gate, EP_TWO_CYCLES, 2)
    _drive(gate, EP_TWO_CYCLES)          # leave an episode in progress

    before = gate.episode_diagnostics()
    for _ in range(5):
        gate.episode_diagnostics()
        gate.episode_records()
        _ = gate.diagnostics
    after = gate.episode_diagnostics()
    assert before == after
    assert gate.diagnostics["n_commits"] == 6
