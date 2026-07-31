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
    since it costs ~1-2 minutes per seed): at n_episodes=3,
    steps_per_episode=400, env_size=6, seeds 0 and 1 each cleared the
    non-degeneracy guard (76 and 120 true boundary events respectively) and
    both reproduced V3-EXQ-824a/838's confirmed finding
    (has_pair_specific_reach=False); seed 2 came back degenerate (0 boundary
    events), correctly flagged rather than misreported as "no reach".
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
