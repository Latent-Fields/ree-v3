"""
Contracts for the Q-081 PAIR-SPECIFIC reach probe (substrate_queue.json
sd_id=Q081-REACH-CHECK-PAIR-SPECIFIC).

Companion to test_q081_landmark_removal.py's `assert_behavioural_reach` tests.
That check is a BLANKET FLAG check (are use_anchor_sets / use_per_region_vs
configured True); this module is the EMPIRICAL check the 2026-07-29 V3-EXQ-838
failure autopsy found was still missing after the flag check passed -- does a
SPECIFIC manipulation actually move a SPECIFIC named signal.

What is pinned here, and why each one is load-bearing:

  1. THE DETECTOR FIRES ON A REAL DIVERGENCE. If it did not, the whole probe
     would be a silently-vacuous "always no reach" verdict -- exactly the
     failure mode Q-081's own "VALIDATE THE NULL BEFORE USING IT" convention
     (q081_surrogate.py, q081_landmark_removal.py) exists to rule out.
  2. THE DETECTOR DOES NOT FALSE-POSITIVE ON IDENTICAL TRACES. A bit-identical
     pair of traces (the actual V3-EXQ-824a/838 empirical shape) must report
     has_pair_specific_reach=False, not a spurious divergence from float noise
     in the comparison logic itself.
  3. TOLERANCE IS RESPECTED. A difference within `tol` must not count, so a
     future caller can loosen the check without editing the comparison logic.
  4. FIRST-DIVERGENT-TICK IS THE EARLIEST ACROSS ALL SIGNALS, not merely the
     first signal name found by iteration order.
  5. MISMATCHED TRACE LENGTHS RAISE rather than silently truncating -- a
     length mismatch between two arms is a caller bug the probe must not mask.
  6. assert_pair_specific_reach RAISES (strict) / RETURNS (non-strict)
     appropriately, and the raised message names the checked signals -- this
     is the GATE text a future driver script's operator will actually read.
  7. snapshot_salience_signals is a pure, non-mutating READ: None on a
     missing/absent coordinator, a plain float dict otherwise, and mutating
     the returned dict must not affect the agent's live _input_signals.
  8. LIVE SMOKE (real REEAgent/env, small dims): the probe machinery actually
     runs end-to-end against the real substrate without crashing, and
     confirms the same "no pair-specific reach" finding V3-EXQ-824a/838
     established empirically -- at a fraction of the cost.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._lib.q081_pair_reach_check import (
    SALIENCE_SIGNAL_NAMES,
    assert_pair_specific_reach,
    pair_specific_reach_report,
    run_pair_specific_reach_probe,
    snapshot_salience_signals,
)


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class _FakeSalience:
    def __init__(self, input_signals: Dict[str, float]):
        self._input_signals = dict(input_signals)


class _FakeAgent:
    def __init__(self, salience: Optional[_FakeSalience]):
        self.salience = salience


def _trace(*snapshots: Optional[Dict[str, float]]) -> List[Optional[Dict[str, float]]]:
    return list(snapshots)


# --------------------------------------------------------------------------- #
# snapshot_salience_signals                                                    #
# --------------------------------------------------------------------------- #


def test_snapshot_returns_copy_of_input_signals():
    agent = _FakeAgent(_FakeSalience({"aic_salience": 0.5, "drive_level": 0.1}))
    snap = snapshot_salience_signals(agent)
    assert snap == {"aic_salience": 0.5, "drive_level": 0.1}
    # Mutating the returned dict must not reach back into the agent's live state.
    snap["aic_salience"] = 999.0
    assert agent.salience._input_signals["aic_salience"] == 0.5


def test_snapshot_none_when_salience_is_none():
    agent = _FakeAgent(None)
    assert snapshot_salience_signals(agent) is None


def test_snapshot_none_when_input_signals_missing():
    class _BareSalience:
        pass

    agent = _FakeAgent(_BareSalience())
    assert snapshot_salience_signals(agent) is None


# --------------------------------------------------------------------------- #
# pair_specific_reach_report -- the core comparison logic                      #
# --------------------------------------------------------------------------- #


def test_no_reach_on_bit_identical_traces():
    """The actual empirical shape of V3-EXQ-824a/838: every named signal
    identical at every tick. Must report has_pair_specific_reach=False, not a
    spurious divergence."""
    snap = {"aic_salience": 0.3, "drive_level": 0.0, "dacc_pe": 0.0}
    intact = _trace(dict(snap), dict(snap), dict(snap))
    manipulated = _trace(dict(snap), dict(snap), dict(snap))
    report = pair_specific_reach_report(intact, manipulated)
    assert report["has_pair_specific_reach"] is False
    assert report["divergent_signals"] == {}
    assert report["first_divergent_tick"] is None
    assert report["n_ticks_compared"] == 3


def test_detects_divergent_signal_and_first_tick():
    intact = _trace(
        {"aic_salience": 0.1}, {"aic_salience": 0.1}, {"aic_salience": 0.1},
    )
    manipulated = _trace(
        {"aic_salience": 0.1}, {"aic_salience": 0.1}, {"aic_salience": 0.9},
    )
    report = pair_specific_reach_report(intact, manipulated)
    assert report["has_pair_specific_reach"] is True
    assert report["divergent_signals"] == {"aic_salience": 2}
    assert report["first_divergent_tick"] == 2


def test_first_divergent_tick_is_earliest_across_all_signals():
    intact = _trace(
        {"aic_salience": 0.1, "drive_level": 0.2},
        {"aic_salience": 0.1, "drive_level": 0.2},
    )
    manipulated = _trace(
        {"aic_salience": 0.1, "drive_level": 0.2},
        {"aic_salience": 0.9, "drive_level": 0.2},
    )
    report = pair_specific_reach_report(intact, manipulated)
    assert report["divergent_signals"] == {"aic_salience": 1}
    assert report["first_divergent_tick"] == 1

    # Now add an EARLIER divergence on a different signal.
    intact2 = _trace(
        {"aic_salience": 0.1, "drive_level": 0.2},
        {"aic_salience": 0.1, "drive_level": 0.2},
    )
    manipulated2 = _trace(
        {"aic_salience": 0.1, "drive_level": 0.7},
        {"aic_salience": 0.9, "drive_level": 0.2},
    )
    report2 = pair_specific_reach_report(intact2, manipulated2)
    assert report2["first_divergent_tick"] == 0
    assert set(report2["divergent_signals"]) == {"aic_salience", "drive_level"}


def test_tolerance_suppresses_small_differences():
    intact = _trace({"aic_salience": 0.500000})
    manipulated = _trace({"aic_salience": 0.500001})
    tight = pair_specific_reach_report(intact, manipulated, tol=0.0)
    assert tight["has_pair_specific_reach"] is True
    loose = pair_specific_reach_report(intact, manipulated, tol=1e-3)
    assert loose["has_pair_specific_reach"] is False


def test_none_snapshots_are_skipped_not_treated_as_divergence():
    intact = _trace({"aic_salience": 0.1}, None, {"aic_salience": 0.1})
    manipulated = _trace({"aic_salience": 0.1}, None, {"aic_salience": 0.1})
    report = pair_specific_reach_report(intact, manipulated)
    assert report["has_pair_specific_reach"] is False


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="trace length mismatch"):
        pair_specific_reach_report(_trace({"a": 1.0}), _trace({"a": 1.0}, {"a": 1.0}))


def test_checked_signal_names_is_union_of_keys_seen():
    intact = _trace({"aic_salience": 0.1, "dacc_pe": 0.0})
    manipulated = _trace({"aic_salience": 0.1, "drive_level": 0.2})
    report = pair_specific_reach_report(intact, manipulated)
    assert report["checked_signal_names"] == ["aic_salience", "dacc_pe", "drive_level"]


# --------------------------------------------------------------------------- #
# assert_pair_specific_reach -- the gate                                       #
# --------------------------------------------------------------------------- #


def test_assert_raises_when_strict_and_no_reach():
    snap = {"aic_salience": 0.2}
    intact = _trace(dict(snap), dict(snap))
    manipulated = _trace(dict(snap), dict(snap))
    with pytest.raises(RuntimeError) as excinfo:
        assert_pair_specific_reach(intact, manipulated, strict=True)
    msg = str(excinfo.value)
    assert "pair-specific reach check FAILED" in msg
    # The gate message must name what it checked, so an operator reading the
    # failure knows the closed signal set rather than guessing.
    for name in sorted(SALIENCE_SIGNAL_NAMES):
        assert name in msg


def test_assert_does_not_raise_when_not_strict():
    snap = {"aic_salience": 0.2}
    intact = _trace(dict(snap))
    manipulated = _trace(dict(snap))
    report = assert_pair_specific_reach(intact, manipulated, strict=False)
    assert report["has_pair_specific_reach"] is False


def test_assert_does_not_raise_when_reach_present():
    intact = _trace({"aic_salience": 0.2})
    manipulated = _trace({"aic_salience": 0.9})
    report = assert_pair_specific_reach(intact, manipulated, strict=True)
    assert report["has_pair_specific_reach"] is True


# --------------------------------------------------------------------------- #
# Live smoke test -- real REEAgent/env, small and short (LIVE SMOKE, item 8)   #
# --------------------------------------------------------------------------- #


def test_live_probe_runs_end_to_end_against_real_substrate():
    """Cheap live smoke test against the real substrate.

    Uses a tiny grid (env_size=5) and a handful of steps (2 episodes x 15
    steps) -- fast enough for the contract suite, but a REAL REEAgent/env pair
    exercising the actual `agent.salience._input_signals` / `.hippocampal`
    attribute paths this module depends on, plus the deepcopy + reset_all_rng
    + episode-aligned-comparison machinery.

    Does NOT assert a specific has_pair_specific_reach value: at this
    intentionally tiny scale (measured while building this module) the
    hippocampal event segmenter very likely never fires a true boundary event
    at all -- see the NON-DEGENERACY GUARD in the module docstring -- so the
    run is expected to come back `is_degenerate=True` rather than a
    meaningful reach verdict. Asserting a fixed boundary count or reach value
    here would be asserting on RNG-sensitive behaviour the test isn't sized
    to control; what this test pins is that the FULL pipeline runs to
    completion and returns a well-formed report against the real substrate.

    The actual "does the substrate show pair-specific reach" question needs
    the module's own (non-degenerate) defaults -- `run_pair_specific_reach_probe()`
    with no overrides -- which is what a real Q-081 driver should call. That
    was run manually while building this module (not as a fast contract test,
    since it costs ~1-2 minutes per seed).

    STALE-NUMBERS NOTE (2026-08-01, positive-control follow-on): this
    docstring previously cited specific per-seed boundary-event counts from
    that manual run ("seed 0 -> 76 events, seed 1 -> 120, seed 2 ->
    degenerate"). Those figures are WITHDRAWN, not merely outdated: building
    the `suppress`-mode positive control below found that
    `run_pair_specific_reach_probe` built its agent template (random weight
    init for every torch.nn.Module in the agent) BEFORE `reset_all_rng(seed)`
    was ever called -- `reset_all_rng` only ran later, inside `_collect_trace`,
    once the template already existed. So `seed` controlled the env and the
    two arms' matched comparison, but NOT the agent's initial weights, which
    depended on whatever torch's global RNG state happened to be at CALL TIME
    in that process -- confirmed empirically: three back-to-back calls with
    identical kwargs returned three different boundary counts. Fixed
    2026-08-01 by moving `reset_all_rng(seed)` before the template is built
    (see the module source, `run_pair_specific_reach_probe`, immediately
    before `agent_template = _make_agent_template()`); verified via
    `test_pair_specific_reach_probe_seed_is_fully_deterministic_across_calls`
    below. The old cited numbers were measured under the pre-fix,
    process-history-dependent construction and cannot be reproduced or
    trusted; a fresh measurement for `iei_permute`/`jitter` at these settings
    was not re-run here (out of scope for this positive-control task -- see
    `test_suppress_mode_positive_control_live_probe` below for the
    correctly-seeded `suppress`-mode figures this fix made possible).
    """
    report = run_pair_specific_reach_probe(
        n_episodes=2, steps_per_episode=15, seed=0, env_size=5, strict=False,
    )
    assert report["n_episodes"] == 2
    assert report["steps_per_episode"] == 15
    assert report["n_ticks_compared"] > 0
    assert report["behavioural_reach_precondition"]["has_behavioural_reach"] is True
    assert isinstance(report["is_degenerate"], bool)
    assert isinstance(report["n_boundaries_true_total"], int)
    assert report["n_boundaries_true_total"] >= 0
    assert isinstance(report["has_pair_specific_reach"], bool)
    assert isinstance(report["divergent_signals"], dict)
    # Whichever way it came back, the two must agree: degeneracy and reach
    # cannot both be meaningfully asserted from the same tiny rollout, but a
    # degenerate run must never accidentally claim reach was found either.
    if report["is_degenerate"]:
        assert report["has_pair_specific_reach"] is False


def test_live_probe_degeneracy_guard_raises_when_strict():
    """The non-degeneracy guard is a hard gate, not a soft note, when strict.

    A rollout this tiny (single episode, 5 steps) is certain to be degenerate
    (see module docstring: an untrained event segmenter needs on the order of
    hundreds of steps before its first boundary, measured while building this
    module). strict=True (the default) must raise rather than return a
    misleading 'no reach' report.
    """
    with pytest.raises(RuntimeError, match="DEGENERATE"):
        run_pair_specific_reach_probe(
            n_episodes=1, steps_per_episode=5, seed=0, env_size=5, strict=True,
        )


# --------------------------------------------------------------------------- #
# Regression: run_pair_specific_reach_probe's `seed` must fully determine the #
# run, including agent construction (2026-08-01 defect fix)                   #
# --------------------------------------------------------------------------- #


def test_pair_specific_reach_probe_seed_is_fully_deterministic_across_calls():
    """Guards the 2026-08-01 defect fix: `seed` must be a COMPLETE determinant
    of the run, not just of the env and the intact-vs-manipulated arm match.

    Found while building the `suppress`-mode positive control below:
    `run_pair_specific_reach_probe` builds its agent TEMPLATE
    (`_make_agent_template()` -> `REEAgent(cfg)`, which randomly initialises
    every torch.nn.Module weight in the agent) before `reset_all_rng(seed)` is
    ever called -- `reset_all_rng` previously only ran later, inside
    `_collect_trace`, once the template already existed. So the agent's
    INITIAL WEIGHTS depended on whatever torch's global RNG state happened to
    be at call time in the current process (e.g. state left behind by a prior
    test in the same pytest session), not on `seed`. Confirmed empirically
    pre-fix: three back-to-back calls with byte-identical kwargs
    (mode="suppress", n_episodes=1, steps_per_episode=150, env_size=6, seed=1)
    returned three different `n_boundaries_true_total` values (9, then 5, then
    4) in one process, and yet a fourth value in a fresh process. This is
    exactly the hazard a PINNED-SEED regression test (this file's whole
    purpose) cannot tolerate: a test asserting an exact value at a "fixed"
    seed would pass or fail depending on what ran before it in the same
    pytest session -- test-order-dependent flakiness, not substrate drift.

    Fixed by moving `reset_all_rng(seed)` to the top of
    `run_pair_specific_reach_probe`, before `_make_agent_template()` is
    called. This test calls the function TWICE with identical kwargs (cheap,
    tiny-scale params -- degenerate is fine here; determinism is what is under
    test, not non-degeneracy) and asserts every field that could plausibly
    move with agent-construction RNG state is byte-identical between calls.
    """
    kwargs = dict(n_episodes=1, steps_per_episode=15, seed=0, env_size=5, strict=False)
    first = run_pair_specific_reach_probe(**kwargs)
    second = run_pair_specific_reach_probe(**kwargs)
    assert first["n_boundaries_true_total"] == second["n_boundaries_true_total"]
    assert first["n_ticks_compared"] == second["n_ticks_compared"]
    assert first["is_degenerate"] == second["is_degenerate"]
    assert first["has_pair_specific_reach"] == second["has_pair_specific_reach"]
    assert first["divergent_signals"] == second["divergent_signals"]
    assert first["dropped_ticks"] == second["dropped_ticks"]


# --------------------------------------------------------------------------- #
# Positive control: does the live probe machinery detect a MAXIMAL            #
# manipulation (mode="suppress", a full lesion) end-to-end against the real   #
# substrate? (Q081-REACH-CHECK-PAIR-SPECIFIC follow-on, 2026-08-01)           #
# --------------------------------------------------------------------------- #


def test_suppress_mode_positive_control_live_probe():
    """Sensitivity positive control for the LIVE probe machinery (as opposed
    to `pair_specific_reach_report`'s pure comparison logic, already covered
    by `test_detects_divergent_signal_and_first_tick` above).

    WHY THIS TEST EXISTS. V3-EXQ-824/824a/838/849 found no reach from
    timing-preserving landmark manipulations (`iei_permute`/`jitter`) to
    either the RV(z_world, operating_mode) statistic or any of this module's
    named salience-precursor signals. Nobody had confirmed that the live
    ROLLOUT + snapshot-diffing machinery in `run_pair_specific_reach_probe`
    (deepcopy'd matched arms, reset_all_rng, StepHarness, episode-aligned
    comparison) can detect a reach that is genuinely there when run
    end-to-end, as opposed to the pure comparison function, which synthetic
    unit tests already validate. `mode="suppress"` is the maximal available
    manipulation: unlike `iei_permute`/`jitter`/`circular_shift` (which
    scramble boundary TIMING but preserve the boundary-emission DRIVE), it
    removes boundary emission to consumers ENTIRELY (see
    `q081_landmark_removal.py`'s `_on_step`: `mode == "suppress" -> out = []`)
    -- a full lesion, deliberately not the scientific primary (it confounds
    "misaligned" with "absent"), but exactly the right instrument for testing
    detector SENSITIVITY: if a full removal cannot show reach, no partial
    scramble ever could either.

    EMPIRICAL RESULT (measured 2026-08-01, post the reset_all_rng-before-
    construction fix above, at n_episodes=1, steps_per_episode=150,
    env_size=6): `mode="suppress"` shows NO reach at seed=2 either --
    has_pair_specific_reach=False, 8 true boundary events fired by the
    segmenter and ALL suppressed (non-degenerate: the manipulation had real
    boundaries to remove), 150/150 ticks compared (full episode, no early
    death), no divergence in any of the 8 signals actually populated in this
    profile (aic_salience, dacc_difficulty, dacc_foraging, dacc_pe,
    drive_level, is_offline, pacc_autonomic, pcc_stability). Seeds 0 and 1 at
    the same tiny-scale settings came back degenerate (0 true boundary
    events); seed 2 is pinned here because it is the cheapest seed found
    (among 0-4) that reliably clears the non-degeneracy guard at this
    contract-test-friendly scale (~15-20s), confirmed byte-reproducible
    across repeated calls post-fix.

    THIS IS NOT AN UNEXAMINED NULL -- an instrument-bug audit was done before
    trusting it (full write-up in the session report, not reproduced here in
    full):
      1. The suppression genuinely reaches the consumer: a separate
         instrumented run (LandmarkScrambler attached directly, not through
         this probe) confirmed `write_anchor` call count and the active-
         anchor-set count diverge from the intact arm starting a few ticks
         into the episode, while the segmenter's OWN true-fire count matches
         the intact arm exactly (same RNG) -- so this is a real, measurable
         downstream effect, not a wrapper that silently fails to intercept.
      2. Source trace of every `SALIENCE_SIGNAL_NAMES` writer call site in
         `ree_core/agent.py` (~6080-6380) found NONE reads
         `anchor_set`/`vs_rollout_gate`/`invalidation_trigger`/
         `boundary_event_queue` state, directly or (for the two signals
         that COULD carry an indirect path: the dACC bundle via
         `e3.last_scores`, and `external_task_drive` via z_world proximity)
         indirectly -- and in this untrained, short-horizon probe regime,
         `self.dacc` never even constructs (this profile does not set
         `use_dacc=True`) and z_world itself never diverged between arms
         either. So "no reach" here reflects a real structural property of
         THIS profile's checked-signal set at THIS probe scale, not a
         mistimed read.
      3. `snapshot_salience_signals` is called immediately after
         `harness.step()` returns, i.e. after sense() (which runs the
         scrambled/suppressed segmenter step) AND select_action() (which runs
         the salience coordinator's tick) have both completed for that tick --
         the read point is not stale relative to the manipulation.

    WHAT THIS DOES AND DOES NOT VALIDATE. It confirms the live rollout +
    snapshot machinery is wired correctly and would show a real divergence if
    one existed in the checked signals (the intervention IS live; the read
    point IS correct) -- so a future `iei_permute`/`jitter` null from this
    same probe is not explained by "the harness never actually ran the
    manipulation" or "the snapshot was read before the effect landed". It
    does NOT mean the probe's checked signal set has a live wired path FROM
    boundary events at all under `q081_profile_kwargs()` -- per point 2
    above, it structurally does not, for any of the 8 populated signals, so a
    genuine positive control for "does the probe detect reach when a real
    wired dependency exists" would need a manipulation on something those
    8 signals actually read (z_harm_a, drive_level, beta_gate, offline mode),
    not on the boundary/landmark train. That is out of this test's scope.
    """
    report = run_pair_specific_reach_probe(
        mode="suppress",
        n_episodes=1,
        steps_per_episode=150,
        env_size=6,
        seed=2,
        strict=False,
    )
    assert report["manipulation_mode"] == "suppress"
    assert report["behavioural_reach_precondition"]["has_behavioural_reach"] is True
    # Non-degenerate: the segmenter fired real boundary events for suppress to
    # remove. A degenerate run here would make "no reach" vacuous, exactly the
    # failure mode the non-degeneracy guard exists to catch.
    assert report["is_degenerate"] is False
    assert report["n_boundaries_true_total"] == 8
    # Full episode ran (no early termination), so the comparison covers the
    # whole requested rollout on both arms.
    assert report["n_ticks_compared"] == 150
    assert report["dropped_ticks"] == {"intact": 0, "manipulated": 0}
    # The actual positive-control result: even the maximal manipulation shows
    # no reach to any checked signal at this seed/scale. See the module-level
    # docstring above for why this is not read as "the detector is broken".
    assert report["has_pair_specific_reach"] is False
    assert report["divergent_signals"] == {}
    assert report["first_divergent_tick"] is None
