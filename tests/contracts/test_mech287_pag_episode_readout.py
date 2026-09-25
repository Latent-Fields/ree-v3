"""Contract tests for the MECH-287 phase-scoped PAG re-commit readout.

THE DEFECT THIS INSTRUMENT REPAIRS (analysis:
REE_assembly/evidence/planning/mech287_dv_instrument_confound_20260924.md):

  PAGFreezeGate.reset() clears the per-episode freeze state but NOT
  _n_ticks / _n_commits / _n_releases, and REEAgent.reset() calls it once per
  episode. The counters are therefore cumulative across warmup+eval. An
  episode that ends while frozen consumes a commit with no matching release,
  so n_commits - n_releases <= n_episodes and the naive ratio
  n_commits / n_releases is dominated by EPISODE COUNT. V3-EXQ-475's
  "~12.9 re-commits per release" (71/6, 70/5, 64/5 over 65 episodes) is that
  artifact. Separately, agent.reset() zeroes the invalidation-trigger and
  staleness counters, so a post-loop get_stats() read sees only the LAST
  episode.

THE DV (ratified 2026-09-25, replacing the registered per-release wording --
findings: REE_assembly/evidence/planning/mech287_readout_redteam_findings_20260925.md):

  `recommits_per_episode`. A "per PAG release" ratio is NOT instrumentable: a
  commit fires only from the inactive state and a release only from the active
  state, so recommits <= releases ALWAYS, the ratio is bounded in [0, 1], and
  it is PINNED at exactly 1.0 with zero variance whenever every episode ends
  frozen -- V3-EXQ-475's phenotype and the comparator regime MECH-287's own
  non-degeneracy precondition requires. `recommits_per_release` is retained as
  a secondary diagnostic and must not be used as a criterion.

Guarantees enforced here:
  C1.  PAGEpisodeRecord importable and exported.
  C2.  Per-episode readout equals a HAND COUNT on a scripted sequence.
  C3.  An episode ending frozen is counted ended_frozen, NOT as a release.
  C4.  BLIND SPOT: at V3-EXQ-475's real shape the naive cumulative ratio moves
       with episode count while the DV does not. Fails against the old read.
  C4b. CEILING: the per-release ratio is pinned at 1.0 where the DV moves.
  C5.  Backward compat: the cumulative counters keep their exact semantics.
  C6.  Phase scoping isolates eval from warmup under BOTH driver conventions.
  C7.  Negative instrument: dv_measurable / dv_name are explicit.
  C8.  The final, never-reset episode is still counted, and records agree.
  C9.  reset_diagnostics() zeroes the cumulative counters AND the freeze.
  C10. Agent snapshots are captured BEFORE reset erases the source counters.
  C11. Agent snapshot is exactly-once per episode.
  C12. Agent readout reports instrument_present=False when nothing is built.
  C13. Reading the readout does not mutate gate state (read-only instrument).
  C14. F9: max_freeze_duration cap-forced releases are reported separately.
  C15. F2: the staleness fields are named as end-of-episode reads, not peaks.
  C16. F6: agent totals are exact past the bounded record list.
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
    """The OLD, confounded read: cumulative commits over cumulative releases.

    F7 FIX: no `max(..., 1)` fudge. Where the run produced no releases at all
    the old read is genuinely UNDEFINED (0/0) and this returns None -- a test
    must not manufacture a divisor the production code never had.
    """
    d = gate.diagnostics
    if d["n_releases"] == 0:
        return None
    return d["n_commits"] / d["n_releases"]


# ------------------------------------------------------------------ #
# C1                                                                 #
# ------------------------------------------------------------------ #

def test_c1_episode_record_exported():
    from ree_core.pag import PAGEpisodeRecord
    rec = PAGEpisodeRecord(index=1, phase="eval", e3_ticks=4, commits=2,
                           releases=1, recommits=1, ended_frozen=True)
    d = rec.as_dict()
    assert d["recommits"] == 1
    assert d["ended_frozen"] is True
    assert d["releases_forced_by_cap"] == 0
    # F5c: the field is named e3_ticks because select_action returns early on
    # non-E3 ticks -- it is NOT an environment-step count.
    assert d["e3_ticks"] == 4
    assert "ticks" not in d


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
    # THE DV.
    assert ep["dv_name"] == "recommits_per_episode"
    assert ep["recommits_per_episode"] == pytest.approx(1.0)
    assert ep["dv_value"] == ep["recommits_per_episode"]
    # Secondary, bounded diagnostic.
    assert ep["recommits_per_release"] == pytest.approx(3.0 / 6.0)

    rows = gate.episode_records()
    assert len(rows) == 3
    for r in rows:
        assert (r["commits"], r["releases"], r["recommits"]) == (2, 2, 1)
        assert r["ended_frozen"] is False
        assert r["releases_forced_by_cap"] == 0


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

def test_c4_blind_spot_naive_read_moves_with_episode_count_the_dv_does_not():
    """The measurement that FAILS against the old cumulative read.

    F7 FIX: built at V3-EXQ-475's REAL shape -- nonzero releases AND nonzero
    ends-frozen -- so the naive read is well defined (no 0/0, no invented
    divisor) and the comparison is honest.

    Per episode [HIGH, LOW, HIGH, HIGH]: COMMIT, RELEASE, RE-COMMIT, then stay
    frozen to the end. Hand count: commits=2, releases=1, recommits=1,
    ended_frozen=True -- exactly 475's signature, n_commits - n_releases == n.

    The naive cumulative read counts the unmatched ends-frozen commit once per
    episode, so it is 2n/n = 2.0 -- but that 2.0 is (1 re-commit + 1 episode
    tail) / (1 release), i.e. an episode-boundary term sitting in the
    numerator. Drive the SAME per-episode behaviour with two releases instead
    of one and the naive read moves while the DV stays put.
    """
    ep_one_release = [HIGH, LOW, HIGH, HIGH]           # 1 release, ends frozen
    ep_two_releases = [HIGH, LOW, HIGH, LOW, HIGH, HIGH]  # 2 releases, ends frozen

    naive, dv = {}, {}
    for label, seq, recommits_per_ep in (
        ("one", ep_one_release, 1),
        ("two", ep_two_releases, 2),
    ):
        for n in (5, 13, 65):
            gate = _gate()
            _run_episodes(gate, seq, n)
            d = gate.diagnostics
            e = gate.episode_diagnostics()
            # 475's confound signature, at every episode count.
            assert d["n_commits"] - d["n_releases"] == n
            assert e["episodes_ending_frozen"] == n
            assert e["recommits"] == recommits_per_ep * n
            naive[(label, n)] = _naive_ratio(gate)
            dv[(label, n)] = e["recommits_per_episode"]

    # The DV is the per-episode re-commit rate and reads it exactly, at every
    # episode count.
    assert {k: v for k, v in dv.items() if k[0] == "one"} == {
        ("one", 5): 1.0, ("one", 13): 1.0, ("one", 65): 1.0}
    assert {k: v for k, v in dv.items() if k[0] == "two"} == {
        ("two", 5): 2.0, ("two", 13): 2.0, ("two", 65): 2.0}

    # The naive read carries the episode tail in its numerator, so it does NOT
    # report the re-commit rate: it says 2.0 and 1.5 for true rates of 1 and 2,
    # and it moves in the WRONG DIRECTION as re-commits increase.
    assert naive[("one", 65)] == pytest.approx(2.0)
    assert naive[("two", 65)] == pytest.approx(1.5)
    assert naive[("one", 65)] > naive[("two", 65)], (
        "the naive read must fall as genuine re-commits RISE -- that is the "
        "confound; if it tracks the DV here the test is not measuring it")
    for n in (5, 13, 65):
        assert naive[("one", n)] != dv[("one", n)]
        assert naive[("two", n)] != dv[("two", n)]


def test_c4b_per_release_ratio_is_pinned_at_1_where_the_dv_moves():
    """The CEILING finding: why the registered per-release DV was replaced.

    A commit fires only from the inactive state and a release only from the
    active state, so within an episode every re-commit follows exactly one
    release: recommits <= releases, always. In the ends-frozen regime the two
    are EQUAL, so the ratio is exactly 1.0 no matter how many re-commits
    happened -- while the DV reads the rate correctly.
    """
    ratio, dv = set(), []
    for cycles in (1, 2, 5, 10):
        seq = ([HIGH, LOW] * cycles) + [HIGH, HIGH]   # ends frozen
        gate = _gate()
        _run_episodes(gate, seq, 11)
        e = gate.episode_diagnostics()
        assert e["episodes_ending_frozen"] == 11
        assert e["recommits"] == e["releases"]
        ratio.add(e["recommits_per_release"])
        dv.append(e["recommits_per_episode"])

    assert ratio == {1.0}, (
        "the per-release ratio must be pinned at 1.0 in the ends-frozen "
        "regime -- that is why it cannot be MECH-287's DV")
    assert dv == [1.0, 2.0, 5.0, 10.0], (
        "the DV must move with the re-commit rate where the ratio cannot: %r" % dv)


def test_c4c_per_release_ratio_can_never_exceed_one():
    """recommits <= releases is structural, so the ratio is bounded in [0, 1]."""
    import random
    rng = random.Random(0)
    worst = 0.0
    for _ in range(200):
        gate = _gate()
        for _ in range(rng.randint(1, 8)):
            _drive(gate, [rng.choice([HIGH, LOW])
                          for _ in range(rng.randint(2, 24))])
            gate.reset()
        e = gate.episode_diagnostics()
        assert e["recommits"] <= e["releases"]
        worst = max(worst, e["recommits_per_release"])
    assert worst <= 1.0


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
    # F4: named episode_allphase_* because that is what they are -- NO phase
    # filter. A driver must not mistake them for a phase-scoped read.
    assert d["episode_allphase_n_episodes"] == 2
    assert d["episode_allphase_recommits"] == 2
    assert d["episode_allphase_recommits_per_episode"] == pytest.approx(1.0)
    assert d["episode_allphase_recommits_per_release"] == pytest.approx(0.5)
    assert d["episode_allphase_episodes_ending_frozen"] == 0
    # F4: the cannot-determine category rides along too.
    assert d["episode_allphase_dv_measurable"] is True
    assert d["episode_allphase_dv_name"] == "recommits_per_episode"
    assert not any(k.startswith("episode_n_") for k in d), (
        "the unscoped keys must not be named as if they were phase-scoped")


def test_c5c_allphase_ride_along_is_warmup_dominated_not_a_phase_read():
    """F4: at a 60/5 shape the pooled keys are nothing like the eval read."""
    gate = _gate()
    gate.set_phase("warmup")
    _run_episodes(gate, EP_ENDS_FROZEN, 60)
    gate.set_phase("eval")
    _run_episodes(gate, EP_TWO_CYCLES, 5)

    d = gate.diagnostics
    ev = gate.episode_diagnostics(phase="eval")
    assert d["episode_allphase_n_episodes"] == 65
    assert ev["n_episodes"] == 5
    assert d["episode_allphase_recommits_per_episode"] != ev["recommits_per_episode"]


# ------------------------------------------------------------------ #
# C6 -- phase scoping                                                #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("reset_at_start", [False, True])
def test_c6_phase_scoping_isolates_eval_from_warmup(reset_at_start):
    """The V3-EXQ-475 shape: 60 warmup episodes then 5 eval episodes.

    F3: parametrised over BOTH driver conventions. Before the fix the label
    was stamped at finalize, so RESET-AT-START gave warmup 59 / eval 6 -- a
    20% contamination of a 5-episode eval denominator, by an ends-frozen
    episode, which is the worst kind for this DV. The old test only ever used
    RESET-AT-END, the one convention under which the bug is invisible.
    """
    gate = _gate()

    def run(seq, n):
        for _ in range(n):
            if reset_at_start:
                gate.reset()
                _drive(gate, seq)
            else:
                _drive(gate, seq)
                gate.reset()

    gate.set_phase("warmup")
    run(EP_ENDS_FROZEN, 60)
    gate.set_phase("eval")
    run(EP_TWO_CYCLES, 5)
    if reset_at_start:
        gate.reset()

    assert gate.episode_phases == ["warmup", "eval"]

    ev = gate.episode_diagnostics(phase="eval")
    assert ev["n_episodes"] == 5
    assert ev["recommits"] == 5
    assert ev["releases"] == 10
    assert ev["recommits_per_episode"] == pytest.approx(1.0)
    assert ev["episodes_ending_frozen"] == 0, (
        "a warmup ends-frozen episode leaked into the eval scope")

    wu = gate.episode_diagnostics(phase="warmup")
    assert wu["n_episodes"] == 60
    assert wu["releases"] == 0
    assert wu["episodes_ending_frozen"] == 60

    both = gate.episode_diagnostics()
    assert both["n_episodes"] == 65
    assert both["commits"] != ev["commits"]


# ------------------------------------------------------------------ #
# C7 -- negative instrument                                          #
# ------------------------------------------------------------------ #

def test_c7_negative_instrument_categories_are_explicit():
    """"0 re-commits" must not be confusable with "never measured"."""
    gate = _gate()
    _run_episodes(gate, EP_ENDS_FROZEN, 10)
    ep = gate.episode_diagnostics()
    # Episodes WERE observed, and the DV is a per-episode rate, so this is a
    # genuine measured zero: 10 episodes, each one commit, no re-commits.
    assert ep["n_episodes"] == 10
    assert ep["recommits_per_episode"] == 0.0
    assert ep["dv_measurable"] is True
    # The per-release ratio, by contrast, had no denominator at all here.
    assert ep["releases"] == 0
    assert ep["recommits_per_release"] == 0.0

    # No episodes at all -> the DV genuinely could not be computed.
    empty = _gate()
    e0 = empty.episode_diagnostics()
    assert e0["n_episodes"] == 0
    assert e0["dv_measurable"] is False, (
        "with no episode observed the DV had no occasion to be measured; "
        "reporting 0.0 without this flag reads as a measured null")


def test_c7b_records_truncated_flag_marks_a_short_record_list():
    """F12/F6: the record list is bounded; the aggregates are not."""
    gate = _gate()
    _run_episodes(gate, [HIGH, LOW], 8300)
    ep = gate.episode_diagnostics()
    assert ep["n_episodes"] == 8300, "aggregates must stay exact past the bound"
    assert len(gate.episode_records(include_current=False)) == 8192
    assert ep["records_truncated"] is True


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

    # F5a: records and diagnostics must agree on the denominator, both ways.
    assert len(gate.episode_records()) == 3
    assert len(gate.episode_records(include_current=False)) == 2


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


def test_c9b_reset_diagnostics_clears_the_freeze_so_phases_cannot_bleed():
    """F8: a warmup freeze must not be RELEASED inside the eval scope."""
    gate = _gate()
    gate.set_phase("warmup")
    _drive(gate, [HIGH, HIGH])            # committed, still frozen
    assert gate.is_active is True

    gate.reset_diagnostics()
    assert gate.is_active is False, (
        "reset_diagnostics left the freeze set -- its release would be scored "
        "in the next phase against a commit belonging to the previous one")

    gate.set_phase("eval")
    _drive(gate, [LOW, HIGH, LOW])
    ev = gate.episode_diagnostics(phase="eval")
    assert ev["releases"] <= ev["commits"]
    assert ev["recommits"] == 0


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
    # F2: the staleness fields say what they are.
    assert "staleness_at_episode_end_max" in ro
    assert ro["staleness_is_end_of_episode_not_peak"] is True
    assert "staleness_peak_over_episodes" not in ro, (
        "the old name claimed a within-episode peak the field is not")
    assert "mean_staleness_peak" not in ro


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


# ------------------------------------------------------------------ #
# C14-C16                                                            #
# ------------------------------------------------------------------ #

def test_c14_cap_forced_releases_are_reported_separately():
    """F9: max_freeze_duration turns the release rate into the cap's period.

    Held at z=HIGH throughout, so z never falls below exit_threshold and NOT
    ONE release is a genuine threshold exit -- every one is the debug cap
    firing. The config comment recommends the cap to prevent permanent locks,
    which is exactly the regime MECH-287 studies.
    """
    gate = _gate(max_freeze_duration=2)
    for _ in range(3):
        _drive(gate, [HIGH] * 12)
        gate.reset()

    ep = gate.episode_diagnostics()
    assert ep["releases"] > 0
    assert ep["releases_forced_by_cap"] == ep["releases"]
    assert ep["releases_all_forced_by_cap"] is True

    # An uncapped gate on genuine threshold exits reports none.
    clean = _gate()
    _run_episodes(clean, EP_TWO_CYCLES, 3)
    c = clean.episode_diagnostics()
    assert c["releases"] > 0
    assert c["releases_forced_by_cap"] == 0
    assert c["releases_all_forced_by_cap"] is False


def test_c15_in_progress_episode_is_not_counted_as_having_ended_frozen():
    """F10: an episode that has not ended cannot be reported as ended-frozen."""
    gate = _gate()
    gate.tick(z_harm_a_norm=HIGH, gaba_tone=1.0)     # commits, still frozen
    ep = gate.episode_diagnostics()
    assert ep["episodes_ending_frozen"] == 0
    assert ep["frac_episodes_ending_frozen"] == 0.0
    assert ep["current_episode_frozen"] is True

    gate.reset()                                      # NOW it has ended frozen
    assert gate.episode_diagnostics()["episodes_ending_frozen"] == 1


def test_c16_agent_totals_are_exact_past_the_bounded_record_list():
    """F6: the readout must not aggregate FROM the bounded deque."""
    agent = _agent_with_instruments()
    trig = agent.hippocampal.invalidation_trigger

    n = 8300
    for _ in range(n):
        trig._n_broadcast = 1
        agent._step_count = 2
        agent.reset()

    ro = agent.get_mech287_episode_readout()
    assert ro["n_episodes"] == n, (
        "agent totals were truncated by the record deque -- a wrong "
        "denominator that reads as a measurement")
    assert ro["n_broadcast_total"] == n
    assert ro["records_truncated"] is True
    assert len(agent.get_mech287_episode_records()) == 8192


def test_c16b_agent_phase_is_stamped_at_episode_start():
    """F3, agent side: a RESET-AT-START driver must not mislabel an episode."""
    agent = _agent_with_instruments()
    trig = agent.hippocampal.invalidation_trigger

    agent.set_mech287_phase("warmup")
    for _ in range(4):
        agent.reset()                 # reset FIRST, then the episode runs
        trig._n_broadcast = 1
        agent._step_count = 3
    agent.set_mech287_phase("eval")
    for _ in range(2):
        agent.reset()
        trig._n_broadcast = 5
        agent._step_count = 3
    agent.reset()

    assert agent.get_mech287_episode_readout(phase="warmup")["n_episodes"] == 4
    assert agent.get_mech287_episode_readout(phase="eval")["n_episodes"] == 2
    assert agent.get_mech287_episode_readout(phase="eval")["n_broadcast_total"] == 10
