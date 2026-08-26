"""Contracts for `experiments/_lib/gate_dv.py` -- the corrected gate-level DV instrument.

The load-bearing test here is `test_repairs_the_709_hold_weighted_defect`: it
runs ONE simulated tick stream through both the old 709 reading (read
`last_score_diagnostics` every env step, count ticks) and the new recorder, and
asserts they DISAGREE in the documented direction. If that test ever passes
trivially -- i.e. the two agree -- the instrument is not repairing anything.

Roughly half of these are negative controls: a healthy gate must NOT trip the
saturation guard, a latched tick must NOT contribute a sample, and a
segregation-off selection must NOT dilute the means. A guard that fires on
ordinary work gets disabled, which is worse than no guard.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _lib.gate_dv import (  # noqa: E402
    DEFAULT_M_CROSS_RANGE_CEILING,
    DEFAULT_W_EFF_RATIO_CEILING,
    GATE_DIAGNOSTIC_KEYS,
    GateDVRecorder,
    _entropy_from_int_counts,
)

E3_SELECTOR = REPO_ROOT / "ree_core" / "predictors" / "e3_selector.py"
GATE_DV_SRC = REPO_ROOT / "experiments" / "_lib" / "gate_dv.py"
EXQ_707C = (
    REPO_ROOT / "experiments"
    / "v3_exq_707c_arc110_loop_segregation_c2_release_repair.py"
)


# ---------------------------------------------------------------------------
# Fakes: reproduce the selector's wholesale-reassign semantics exactly.
# ---------------------------------------------------------------------------
class _FakeE3:
    def __init__(self) -> None:
        self.last_score_diagnostics = {}


class FakeAgent:
    """Minimal stand-in. `select()` reassigns the diagnostics dict WHOLESALE,
    which is the invariant the fresh_select sentinel depends on
    (e3_selector.py:2452); a latched tick leaves the dict untouched."""

    def __init__(self) -> None:
        self.e3 = _FakeE3()

    def select(self, **gate):
        payload = {"loop_segregation_active": True}
        payload.update(gate)
        self.e3.last_score_diagnostics = payload

    def latch(self):
        pass


def _healthy_gate(limbic_wins: bool):
    return {
        "loop_cross_loop_w_motor_eff": 1.0,
        "loop_cross_loop_w_assoc_eff": 0.4,
        "loop_cross_loop_w_limbic_eff": 1.2 if limbic_wins else 0.5,
        "loop_cross_loop_limbic_ge_motor": bool(limbic_wins),
        "loop_cross_loop_m_range": 0.11,
        "loop_cross_loop_limbic_to_motor": 0.03,
        "loop_cross_loop_n_updates": 7,
        "loop_assoc_pref_range": 0.9,
        "loop_limbic_pref_range": 1.414,
        "loop_limbic_routed_max_range": 1.414,
        "loop_d1_d2_active": True,
        "loop_d1_d2_conflict_signal": 0.2,
        "loop_committed_neq_motor_winner": bool(limbic_wins),
        "loop_cross_loop_winner_disagreement": bool(limbic_wins),
    }


# ---------------------------------------------------------------------------
# THE load-bearing differential test
# ---------------------------------------------------------------------------
def test_repairs_the_709_hold_weighted_defect():
    """Old reading and new reading must DISAGREE on a class-dependent hold.

    Stream: two genuine selections. The limbic-winning one is held for 2 env
    steps, the non-winning one for 8 -- exactly the class-dependent hold that
    makes the distortion fail to cancel. Fresh-gated truth is 1/2 = 0.50; the
    709 tick-count reading is 2/10 = 0.20.
    """
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    old_ticks_limbic_ge = 0
    old_ticks_total = 0

    plan = [(True, 2), (False, 8)]  # (limbic_wins, hold_duration)
    rec.begin_episode()
    for limbic_wins, hold in plan:
        for i in range(hold):
            if i == 0:
                with rec.watch(agent) as sel:
                    agent.select(**_healthy_gate(limbic_wins))
            else:
                with rec.watch(agent) as sel:
                    agent.latch()
            rec.record(agent, sel, committed_class=1 if limbic_wins else 2)
            # the 709 reading: every env step, no freshness guard
            diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
            old_ticks_total += 1
            if bool(diag.get("loop_cross_loop_limbic_ge_motor", False)):
                old_ticks_limbic_ge += 1
    rec.end_episode()

    new_frac = rec.as_dict()["gate_limbic_ge_motor_frac"]
    old_frac = old_ticks_limbic_ge / old_ticks_total

    assert new_frac == pytest.approx(0.5), new_frac
    assert old_frac == pytest.approx(0.2), old_frac
    assert new_frac != pytest.approx(old_frac), (
        "instrument agrees with the defective reading -- it repairs nothing"
    )
    assert rec.as_dict()["gate_n_gate_samples"] == 2


def test_fresh_select_counts_are_selections_not_ticks():
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    rec.begin_episode()
    for i in range(10):
        with rec.watch(agent) as sel:
            agent.select(**_healthy_gate(False)) if i == 0 else agent.latch()
        rec.record(agent, sel, committed_class=3)
    rec.end_episode()
    d = rec.as_dict()
    assert d["gate_n_dv_selections"] == 1
    assert d["gate_n_gate_samples"] == 1


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------
def test_latched_tick_contributes_no_gate_sample():
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    rec.begin_episode()
    with rec.watch(agent) as sel:
        agent.latch()
    counted = rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    assert counted is False
    assert rec.as_dict()["gate_n_gate_samples"] == 0
    assert rec.as_dict()["gate_n_dv_selections"] == 0


def test_segregation_inactive_selection_does_not_dilute_means():
    """A selection where arbitration did not run must not push means toward 0."""
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    rec.begin_episode()
    with rec.watch(agent) as sel:
        agent.select(**_healthy_gate(True))
    rec.record(agent, sel, committed_class=1)
    with rec.watch(agent) as sel:
        agent.e3.last_score_diagnostics = {"loop_segregation_active": False}
    rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    d = rec.as_dict()
    assert d["gate_n_gate_samples"] == 1
    assert d["gate_w_limbic_eff_mean"] == pytest.approx(1.2)
    assert d["gate_limbic_ge_motor_frac"] == pytest.approx(1.0)


def test_healthy_gate_does_not_trip_saturation():
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    rec.begin_episode()
    for _ in range(5):
        with rec.watch(agent) as sel:
            agent.select(**_healthy_gate(True))
        rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    assert rec.saturated is False
    assert rec.gate_readiness()["gate_parity_not_saturated"] is True


def test_fallback_selection_excluded_from_dv_but_counted_fresh():
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    rec.begin_episode()
    with rec.watch(agent) as sel:
        agent.select(**_healthy_gate(True))
    rec.record(agent, sel, committed_class=1, fallback=True)
    rec.end_episode()
    d = rec.as_dict()
    assert d["gate_n_dv_selections"] == 0
    assert d["gate_n_fallback_skipped"] == 1
    assert d["gate_n_gate_samples"] == 1


# ---------------------------------------------------------------------------
# Saturation guard (the 711 lesson)
# ---------------------------------------------------------------------------
def test_saturation_guard_fires_on_711_values():
    """V3-EXQ-711 measured M_cross range 4897.8 and w_eff ratio up to 2274x."""
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    gate = _healthy_gate(True)
    gate["loop_cross_loop_m_range"] = 4897.8
    gate["loop_cross_loop_w_limbic_eff"] = 2274.0
    gate["loop_cross_loop_w_motor_eff"] = 1.0
    rec.begin_episode()
    with rec.watch(agent) as sel:
        agent.select(**gate)
    rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    assert rec.saturated is True
    r = rec.gate_readiness()
    assert r["gate_limbic_can_win"] is True, "711 did meet limbic_can_win"
    assert r["gate_ready"] is False, (
        "a saturated cell must self-route substrate_not_ready_requeue, "
        "never be scored as a parity win"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("loop_cross_loop_m_range", DEFAULT_M_CROSS_RANGE_CEILING + 1.0),
        ("loop_cross_loop_w_limbic_eff", DEFAULT_W_EFF_RATIO_CEILING + 1.0),
    ],
)
def test_either_ceiling_alone_trips_saturation(field, value):
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    gate = _healthy_gate(True)
    gate["loop_cross_loop_w_motor_eff"] = 1.0
    gate[field] = value
    rec.begin_episode()
    with rec.watch(agent) as sel:
        agent.select(**gate)
    rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    assert rec.saturated is True


def test_readiness_refuses_when_learning_never_engaged():
    """M_cross pinned at its zero init -> vacuous cell, not a weakens."""
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    gate = _healthy_gate(False)
    gate["loop_cross_loop_m_range"] = 0.0
    gate["loop_cross_loop_n_updates"] = 0
    rec.begin_episode()
    for _ in range(40):
        with rec.watch(agent) as sel:
            agent.select(**gate)
        rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    r = rec.gate_readiness()
    assert r["gate_learning_engaged"] is False
    assert r["gate_ready"] is False


def test_readiness_passes_on_a_healthy_engaged_gate():
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    rec.begin_episode()
    for i in range(40):
        with rec.watch(agent) as sel:
            agent.select(**_healthy_gate(i % 2 == 0))
        rec.record(agent, sel, committed_class=i % 3)
    rec.end_episode()
    r = rec.gate_readiness()
    assert r["gate_fresh_selects_sufficient"] is True
    assert r["gate_arbitration_engaged"] is True
    assert r["gate_learning_engaged"] is True
    assert r["gate_parity_not_saturated"] is True
    assert r["gate_ready"] is True


# ---------------------------------------------------------------------------
# Cross-instrument agreement + rename guards
# ---------------------------------------------------------------------------
def test_entropy_matches_the_707c_repaired_reference():
    """Byte-for-byte agreement with 707c's own entropy, extracted from source."""
    tree = ast.parse(EXQ_707C.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_entropy_from_int_counts"
    )
    ns = {"math": math, "Dict": dict}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<707c>", "exec"), ns)
    ref = ns["_entropy_from_int_counts"]
    for counts in ({}, {1: 5}, {1: 3, 2: 3}, {1: 1, 2: 2, 3: 7}, {4: 0}):
        assert _entropy_from_int_counts(counts) == ref(counts), counts


def test_every_gate_key_still_exists_in_the_selector():
    """Rename guard: a renamed diagnostic must fail here, not read zeros forever."""
    src = E3_SELECTOR.read_text()
    missing = [k for k in GATE_DIAGNOSTIC_KEYS if '"%s"' % k not in src]
    assert not missing, "diagnostics missing from e3_selector.py: %s" % missing


def test_module_output_is_ascii_only():
    raw = GATE_DV_SRC.read_bytes()
    assert all(b < 128 for b in raw), "non-ASCII byte in gate_dv.py"


# ---------------------------------------------------------------------------
# MECH-464: straddle fraction + da=0 shadow argmin
# ---------------------------------------------------------------------------
def test_straddle_and_reorder_confound_denominated_on_d1d2_active_only():
    """A d1d2-OFF sample in the same run must not dilute the MECH-464 fields
    (the same dilution bug loop_d1_d2_conflict_signal already carries, which
    these must not repeat)."""
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    rec.begin_episode()
    gate_on = _healthy_gate(True)
    gate_on.update(
        {
            "loop_assoc_straddle_frac": 0.6,
            "loop_limbic_straddle_frac": 0.8,
            "loop_d1_d2_reorder_vs_da0": True,
            "loop_d1_d2_d2_gain_zero": True,
        }
    )
    with rec.watch(agent) as sel:
        agent.select(**gate_on)
    rec.record(agent, sel, committed_class=1)

    gate_off = _healthy_gate(False)
    gate_off["loop_d1_d2_active"] = False
    with rec.watch(agent) as sel:
        agent.select(**gate_off)
    rec.record(agent, sel, committed_class=2)
    rec.end_episode()

    d = rec.as_dict()
    assert d["gate_assoc_straddle_frac_mean"] == pytest.approx(0.6)
    assert d["gate_limbic_straddle_frac_mean"] == pytest.approx(0.8)
    assert d["gate_d1_d2_reorder_vs_da0_frac"] == pytest.approx(1.0)
    assert d["gate_d1_d2_d2_gain_zero_frac"] == pytest.approx(1.0)


def test_straddle_nonvacuous_gate_refuses_near_zero_straddle():
    """MECH-464 MANDATORY non-vacuity gate: a ~0 straddle fraction must
    refuse, not pass, so the falsifier scores precondition_unmet rather than
    a spurious null."""
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    gate = _healthy_gate(True)
    gate["loop_assoc_straddle_frac"] = 0.0
    gate["loop_limbic_straddle_frac"] = 0.0
    rec.begin_episode()
    for _ in range(5):
        with rec.watch(agent) as sel:
            agent.select(**gate)
        rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    r = rec.gate_readiness()
    assert r["gate_d1_d2_ran"] is True
    assert r["gate_straddle_nonvacuous"] is False


def test_straddle_nonvacuous_gate_passes_above_floor():
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    gate = _healthy_gate(True)
    gate["loop_assoc_straddle_frac"] = 0.5
    gate["loop_limbic_straddle_frac"] = 0.0
    rec.begin_episode()
    for _ in range(5):
        with rec.watch(agent) as sel:
            agent.select(**gate)
        rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    assert rec.gate_readiness()["gate_straddle_nonvacuous"] is True


def test_straddle_nonvacuous_gate_false_when_d1d2_never_ran():
    agent = FakeAgent()
    rec = GateDVRecorder("testns")
    gate = _healthy_gate(True)
    gate["loop_d1_d2_active"] = False
    rec.begin_episode()
    with rec.watch(agent) as sel:
        agent.select(**gate)
    rec.record(agent, sel, committed_class=1)
    rec.end_episode()
    r = rec.gate_readiness()
    assert r["gate_d1_d2_ran"] is False
    assert r["gate_straddle_nonvacuous"] is False
