"""Gate-level DV instrument: fresh-select-gated readings of the ARC-110 arbitration gate.

WHY THIS EXISTS -- the two half-instruments that were never combined
-------------------------------------------------------------------
The cross-loop arbitration lineage has two instrument halves that live in
DIFFERENT drivers and have never been used together:

  * `v3_exq_707c_...` carries the fresh-select repair (the 699 / 689d
    instrument fix, now factored into `_lib/fresh_select.py`) but reads NO gate
    telemetry at all -- its DV is behavioural, three stages downstream of the
    gate.
  * `v3_exq_709 / 711 / 713` read the gate telemetry richly
    (`loop_cross_loop_w_*_eff`, `loop_cross_loop_m_range`,
    `loop_cross_loop_limbic_ge_motor`, ...) but predate the fresh-select repair,
    so every one of those reads is taken on EVERY env step:

        diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}   # 709:801

    E3 cadence defaults to 10 (`heartbeat.e3_steps_per_tick`), so a quantity
    sampled once per genuine selection is counted ~10x -- and UNEQUALLY across
    arms, because an arm that changes commitment dynamics changes hold
    duration. `clg_limbic_ge_motor_ticks` is the sharp case: a COUNT whose
    denominator is env steps, i.e. exactly the disqualifying class named in
    `fresh_select.py`.

That hold-weighted C1 is why the 709/711/713 exhaustion verdict was WITHDRAWN
(2026-07-20) and why `substrate_queue.v4_loop_segregation` records
`713x_re_letter_STILL_REFUSED_corrected_DV_instrument_required`. This module is
that corrected instrument: the gate DV, read at the gate, once per selection.

WHAT IS AND IS NOT REPLICATION-SENSITIVE (read before adding a field)
--------------------------------------------------------------------
Not every hold-weighted read was wrong, and pretending otherwise would
overstate the 709/711/713 defect:

  * MAX-reductions (`clg_m_range_peak`, `clg_w_limbic_eff_peak`) are
    replication-INVARIANT -- sampling the same value ten times does not move a
    max. Those 709 readings were already sound.
  * COUNTS, FRACTIONS, MEANS and ENTROPIES are replication-SENSITIVE. These are
    the ones the hold silently reweights.

This recorder fresh-gates EVERYTHING anyway (one uniform rule is far harder to
get wrong than a per-field judgement call), and emits both the mean and the
peak for each gate quantity so a consumer can see the two agree.

THE SATURATION GUARD (the 711 lesson)
-------------------------------------
V3-EXQ-711 met `limbic_loop_can_win` on 3/3 divergent seeds -- by SATURATION.
`M_cross` range peaked at 4897.8 (vs the healthy un-gained ~0.02-0.12) and
`w_eff[limbic]` reached 10-2274x `w_eff[motor]`: a limbic MONOPOLY replacing
the motor pinning, not a fair arbitration. Committed-class entropy FELL. So
"the limbic loop won" is NOT on its own a readiness signal; it has to be a
parity win inside a bounded band. `saturated` reports that, and
`gate_readiness()` refuses the cell when it fires -- the `substrate_not_ready_
requeue` self-route, never a false weakens.

USAGE
-----
    from _lib.gate_dv import GateDVRecorder

    rec = GateDVRecorder("exq713x")

    for ep in range(n_episodes):
        rec.begin_episode()                  # no hold may span an episode
        for step in range(n_steps):
            with rec.watch(agent) as sel:
                action = agent.select_action(candidates, ticks)
            if is_p2:
                rec.record(
                    agent, sel,
                    committed_class=committed_class,
                    fallback=action_was_fallback,
                )
        rec.end_episode()

    row.update(rec.as_dict())
    row.update(rec.gate_readiness())

`record()` is deliberately outside `watch()`: the caller owns the phase gate
(`is_p2`) and the fallback gate, exactly as in `fresh_select.FreshSelectCounter`.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from _lib.fresh_select import (
    FRESH_SELECT_RATIONALE,
    FreshSelectCounter,
    FreshSelectProbe,
    FreshSelectResult,
)

__all__ = [
    "GateDVRecorder",
    "GATE_DIAGNOSTIC_KEYS",
    "GATE_DV_RATIONALE",
    "DEFAULT_FRESH_SELECT_FLOOR",
    "DEFAULT_W_EFF_RATIO_CEILING",
    "DEFAULT_M_CROSS_RANGE_CEILING",
    "DEFAULT_M_CROSS_MOVED_FLOOR",
    "DEFAULT_STRADDLE_FRAC_FLOOR",
]

# The e3_selector.last_score_diagnostics keys this instrument reads. Kept as a
# module constant so a contract test can assert they still exist in the
# selector rather than silently reading zeros forever if one is renamed.
GATE_DIAGNOSTIC_KEYS = (
    "loop_segregation_active",
    "loop_cross_loop_w_motor_eff",
    "loop_cross_loop_w_assoc_eff",
    "loop_cross_loop_w_limbic_eff",
    "loop_cross_loop_limbic_ge_motor",
    "loop_cross_loop_m_range",
    "loop_cross_loop_limbic_to_motor",
    "loop_cross_loop_n_updates",
    "loop_assoc_pref_range",
    "loop_limbic_pref_range",
    "loop_limbic_routed_max_range",
    "loop_d1_d2_active",
    "loop_d1_d2_conflict_signal",
    "loop_committed_neq_motor_winner",
    "loop_cross_loop_winner_disagreement",
    # MECH-464: the straddle-fraction non-vacuity gate + da=0 shadow argmin.
    "loop_assoc_straddle_frac",
    "loop_limbic_straddle_frac",
    "loop_d1_d2_reorder_vs_da0",
    "loop_d1_d2_d2_gain_zero",
)

# Readiness defaults.
#   floor 30      -- matches 707c's fresh_selects_sufficient (worst cell 922 >> 30).
#   ratio 5.0     -- 711 measured 10-2274x; healthy parity sits near 1.0.
#   range 50.0    -- 711 measured M_cross range 4897.8; healthy is ~0.02-0.12.
#   moved 1e-6    -- non-vacuity: M_cross must leave its exact zero init.
DEFAULT_FRESH_SELECT_FLOOR = 30
DEFAULT_W_EFF_RATIO_CEILING = 5.0
DEFAULT_M_CROSS_RANGE_CEILING = 50.0
DEFAULT_M_CROSS_MOVED_FLOOR = 1e-6
#   straddle 0.01 -- MECH-464's MANDATORY non-vacuity gate: "if [the straddle
#   fraction] is ~0 the run is vacuous and MUST be scored precondition_unmet,
#   not as a null". 1% is comfortably above float noise, well below "healthy".
DEFAULT_STRADDLE_FRAC_FLOOR = 0.01

GATE_DV_RATIONALE = (
    "Gate-level DVs are read via experiments/_lib/gate_dv.py, which takes every "
    "loop_* / loop_cross_loop_* reading ONCE PER GENUINE E3 SELECTION using the "
    "shared fresh_select sentinel, never per env step. This repairs the "
    "hold-weighted gate readout of V3-EXQ-709/711/713 (whose "
    "clg_limbic_ge_motor_ticks was a count with an env-step denominator, "
    "replicated ~10x by e3_steps_per_tick and unequally across arms). "
    "Committed-class entropy is accumulated on fresh, non-fallback selections "
    "only (the 707c repair). A limbic win is additionally checked against a "
    "saturation guard so the V3-EXQ-711 monopoly regime (M_cross range 4897.8, "
    "w_eff ratio up to 2274x) self-routes substrate_not_ready_requeue instead "
    "of being scored as a parity win. " + FRESH_SELECT_RATIONALE
)


def _f(diag: Dict[str, Any], key: str) -> Optional[float]:
    """Float read that distinguishes 'absent' from 'present and zero'."""
    if key not in diag:
        return None
    v = diag.get(key)
    if v is None or isinstance(v, bool):
        return float(v) if isinstance(v, bool) else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _peak(xs: List[float]) -> float:
    return float(max(xs)) if xs else 0.0


def _entropy_from_int_counts(counts: Dict[int, int]) -> float:
    """Shannon entropy in nats over an integer-class histogram.

    Byte-identical to `v3_exq_707c._entropy_from_int_counts` (the repaired
    reference) so a re-letter's DV is comparable to 707c's without a
    reimplementation drifting from it.
    """
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


class GateDVRecorder:
    """Fresh-select-gated recorder for the ARC-110 cross-loop arbitration gate."""

    def __init__(
        self,
        namespace: str,
        fresh_select_floor: int = DEFAULT_FRESH_SELECT_FLOOR,
        w_eff_ratio_ceiling: float = DEFAULT_W_EFF_RATIO_CEILING,
        m_cross_range_ceiling: float = DEFAULT_M_CROSS_RANGE_CEILING,
        m_cross_moved_floor: float = DEFAULT_M_CROSS_MOVED_FLOOR,
        straddle_frac_floor: float = DEFAULT_STRADDLE_FRAC_FLOOR,
    ) -> None:
        self.probe = FreshSelectProbe(namespace)
        self.counter = FreshSelectCounter()
        self.fresh_select_floor = int(fresh_select_floor)
        self.straddle_frac_floor = float(straddle_frac_floor)
        self.w_eff_ratio_ceiling = float(w_eff_ratio_ceiling)
        self.m_cross_range_ceiling = float(m_cross_range_ceiling)
        self.m_cross_moved_floor = float(m_cross_moved_floor)

        # DV accumulators -- fresh, non-fallback selections only.
        self.committed_class_counts: Dict[int, int] = {}
        self.n_dv_selections = 0
        self.n_fallback_skipped = 0

        # Per-fresh-selection gate samples.
        self._w_motor: List[float] = []
        self._w_assoc: List[float] = []
        self._w_limbic: List[float] = []
        self._m_range: List[float] = []
        self._limbic_to_motor: List[float] = []
        self._assoc_pref_range: List[float] = []
        self._limbic_pref_range: List[float] = []
        self._limbic_routed_range: List[float] = []
        self._d1d2_conflict: List[float] = []
        self._w_eff_ratio: List[float] = []
        # MECH-464: straddle fraction / reorder confound, sampled only on
        # d1d2-active gate samples (see _n_d1d2_active denominator below) so an
        # arm that runs with d1d2 off does not dilute these means/fractions
        # toward the vacuous zero -- the same dilution _d1d2_conflict above
        # already carries and which these MUST NOT repeat, since MECH-464's
        # falsifier reads the straddle fraction as its non-vacuity gate.
        self._assoc_straddle: List[float] = []
        self._limbic_straddle: List[float] = []
        self._n_d1d2_reorder_vs_da0 = 0
        self._n_d1d2_d2_gain_zero = 0

        self._n_gate_samples = 0
        self._n_limbic_ge_motor = 0
        self._n_d1d2_active = 0
        self._n_committed_neq_motor = 0
        self._n_winner_disagreement = 0
        self._n_segregation_active = 0
        self._n_updates_last = 0

    # -- lifecycle ---------------------------------------------------------
    def begin_episode(self) -> None:
        """Close any hold left open by the previous episode."""
        self.counter.flush()

    def end_episode(self) -> None:
        """Close a hold left open at the end of an episode."""
        self.counter.flush()

    @contextmanager
    def watch(self, agent: Any) -> Iterator[FreshSelectResult]:
        """Delegate to the shared sentinel probe."""
        with self.probe.watch(agent) as res:
            yield res

    # -- accumulation ------------------------------------------------------
    def record(
        self,
        agent: Any,
        sel: FreshSelectResult,
        committed_class: Optional[int] = None,
        fallback: bool = False,
    ) -> bool:
        """Record one measured tick. Returns True iff it counted as a selection.

        `sel` must come from a CLOSED `watch()` block. Latched ticks update the
        hold bookkeeping and nothing else; fallback selections update the
        freshness counters (they really were fresh selections) but are excluded
        from the DV histogram, matching the 707c repair.
        """
        fresh = bool(sel)
        self.counter.record(fresh)
        if not fresh:
            return False

        if fallback:
            self.n_fallback_skipped += 1
        elif committed_class is not None:
            self.n_dv_selections += 1
            key = int(committed_class)
            self.committed_class_counts[key] = (
                self.committed_class_counts.get(key, 0) + 1
            )

        # `diagnostics()` returns {} on a latched tick, so every read below is
        # fresh-gated by construction.
        diag = self.probe.diagnostics(agent, fresh)
        if not diag:
            return True
        if not bool(diag.get("loop_segregation_active", False)):
            # Arbitration did not run this selection (segregation OFF, or a
            # single-eligible shortcut). Counting zeros here would dilute every
            # mean toward 0 and manufacture a false "limbic never wins".
            return True

        self._n_segregation_active += 1
        self._n_gate_samples += 1

        wm = _f(diag, "loop_cross_loop_w_motor_eff")
        wa = _f(diag, "loop_cross_loop_w_assoc_eff")
        wl = _f(diag, "loop_cross_loop_w_limbic_eff")
        if wm is not None:
            self._w_motor.append(wm)
        if wa is not None:
            self._w_assoc.append(wa)
        if wl is not None:
            self._w_limbic.append(wl)
        if wm is not None and wl is not None and abs(wm) > 1e-12:
            self._w_eff_ratio.append(abs(wl) / abs(wm))

        for key, sink in (
            ("loop_cross_loop_m_range", self._m_range),
            ("loop_cross_loop_limbic_to_motor", self._limbic_to_motor),
            ("loop_assoc_pref_range", self._assoc_pref_range),
            ("loop_limbic_pref_range", self._limbic_pref_range),
            ("loop_limbic_routed_max_range", self._limbic_routed_range),
            ("loop_d1_d2_conflict_signal", self._d1d2_conflict),
        ):
            v = _f(diag, key)
            if v is not None:
                sink.append(v)

        if bool(diag.get("loop_cross_loop_limbic_ge_motor", False)):
            self._n_limbic_ge_motor += 1
        d1d2_active = bool(diag.get("loop_d1_d2_active", False))
        if d1d2_active:
            self._n_d1d2_active += 1
            # MECH-464: only sampled on d1d2-active ticks (see the __init__
            # comment) -- the straddle fraction and reorder confound are
            # undefined, not zero, when the split never ran.
            assoc_s = _f(diag, "loop_assoc_straddle_frac")
            if assoc_s is not None:
                self._assoc_straddle.append(assoc_s)
            limbic_s = _f(diag, "loop_limbic_straddle_frac")
            if limbic_s is not None:
                self._limbic_straddle.append(limbic_s)
            if bool(diag.get("loop_d1_d2_reorder_vs_da0", False)):
                self._n_d1d2_reorder_vs_da0 += 1
            if bool(diag.get("loop_d1_d2_d2_gain_zero", False)):
                self._n_d1d2_d2_gain_zero += 1
        if bool(diag.get("loop_committed_neq_motor_winner", False)):
            self._n_committed_neq_motor += 1
        if bool(diag.get("loop_cross_loop_winner_disagreement", False)):
            self._n_winner_disagreement += 1

        nu = _f(diag, "loop_cross_loop_n_updates")
        if nu is not None:
            self._n_updates_last = int(nu)
        return True

    # -- reporting ---------------------------------------------------------
    def _frac(self, n: int) -> float:
        return float(n / self._n_gate_samples) if self._n_gate_samples else 0.0

    def _frac_of(self, n: int, denom: int) -> float:
        return float(n / denom) if denom else 0.0

    @property
    def saturated(self) -> bool:
        """True iff the gate is in the V3-EXQ-711 blow-up regime."""
        return bool(
            _peak(self._w_eff_ratio) > self.w_eff_ratio_ceiling
            or _peak(self._m_range) > self.m_cross_range_ceiling
        )

    def as_dict(self, n_ticks: Optional[int] = None) -> Dict[str, Any]:
        """Manifest row fragment: fresh-select telemetry + gate DVs."""
        total_ticks = (
            int(n_ticks)
            if n_ticks is not None
            else (self.counter.n_fresh_select + self.counter.n_latched)
        )
        out: Dict[str, Any] = dict(self.counter.as_dict(total_ticks))

        out.update(
            {
                # Primary DV -- fresh, non-fallback selections only (707c repair).
                "gate_committed_class_entropy_nats": round(
                    _entropy_from_int_counts(self.committed_class_counts), 6
                ),
                "gate_n_dv_selections": int(self.n_dv_selections),
                "gate_n_unique_committed_classes": int(
                    len(self.committed_class_counts)
                ),
                "gate_n_fallback_skipped": int(self.n_fallback_skipped),
                # Exposure -- the +97.6% arm spread that drove the 707b
                # distortion stays on the record. Gates nothing.
                "gate_n_gate_samples": int(self._n_gate_samples),
                "gate_segregation_active_frac": self._frac(
                    self._n_segregation_active
                ),
                # Per-loop effective authority (mean AND peak; peak is the
                # replication-invariant reading 709 already had right).
                "gate_w_motor_eff_mean": round(_mean(self._w_motor), 6),
                "gate_w_motor_eff_peak": round(_peak(self._w_motor), 6),
                "gate_w_assoc_eff_mean": round(_mean(self._w_assoc), 6),
                "gate_w_assoc_eff_peak": round(_peak(self._w_assoc), 6),
                "gate_w_limbic_eff_mean": round(_mean(self._w_limbic), 6),
                "gate_w_limbic_eff_peak": round(_peak(self._w_limbic), 6),
                # THE repaired reading: a FRACTION OF SELECTIONS, not a tick
                # count with an env-step denominator (the 709 defect).
                "gate_limbic_ge_motor_frac": self._frac(self._n_limbic_ge_motor),
                "gate_w_eff_ratio_mean": round(_mean(self._w_eff_ratio), 6),
                "gate_w_eff_ratio_peak": round(_peak(self._w_eff_ratio), 6),
                # Learning engagement.
                "gate_m_cross_range_mean": round(_mean(self._m_range), 6),
                "gate_m_cross_range_peak": round(_peak(self._m_range), 6),
                "gate_limbic_to_motor_mean": round(_mean(self._limbic_to_motor), 6),
                "gate_limbic_to_motor_peak": round(_peak(self._limbic_to_motor), 6),
                "gate_n_updates": int(self._n_updates_last),
                # Loop signal liveness.
                "gate_assoc_pref_range_mean": round(_mean(self._assoc_pref_range), 6),
                "gate_limbic_pref_range_mean": round(
                    _mean(self._limbic_pref_range), 6
                ),
                "gate_limbic_routed_max_range_peak": round(
                    _peak(self._limbic_routed_range), 6
                ),
                # D1/D2 opponent structure (ARC-109).
                "gate_d1_d2_active_frac": self._frac(self._n_d1d2_active),
                "gate_d1_d2_conflict_mean": round(_mean(self._d1d2_conflict), 6),
                # MECH-464 straddle-fraction non-vacuity gate + da=0 shadow argmin
                # reorder confound -- all denominated on d1d2-ACTIVE samples
                # (_n_d1d2_active), not _n_gate_samples, so an arm run with the
                # split off never dilutes these toward a spurious "vacuous" 0.
                "gate_assoc_straddle_frac_mean": round(_mean(self._assoc_straddle), 6),
                "gate_assoc_straddle_frac_peak": round(_peak(self._assoc_straddle), 6),
                "gate_limbic_straddle_frac_mean": round(
                    _mean(self._limbic_straddle), 6
                ),
                "gate_limbic_straddle_frac_peak": round(
                    _peak(self._limbic_straddle), 6
                ),
                "gate_d1_d2_reorder_vs_da0_frac": self._frac_of(
                    self._n_d1d2_reorder_vs_da0, self._n_d1d2_active
                ),
                "gate_d1_d2_d2_gain_zero_frac": self._frac_of(
                    self._n_d1d2_d2_gain_zero, self._n_d1d2_active
                ),
                # Arbitration outcome.
                "gate_committed_neq_motor_winner_frac": self._frac(
                    self._n_committed_neq_motor
                ),
                "gate_winner_disagreement_frac": self._frac(
                    self._n_winner_disagreement
                ),
                # Saturation guard (711).
                "gate_saturated": bool(self.saturated),
            }
        )
        return out

    def gate_readiness(self) -> Dict[str, Any]:
        """Non-vacuity / readiness booleans for the falsifier's precondition block.

        `gate_ready` is deliberately AND-of-all: a cell that is saturated, or
        whose arbitration never engaged, cannot validly answer the conversion
        question, and must self-route `substrate_not_ready_requeue` rather than
        be scored as a weakens.
        """
        fresh_ok = self.counter.n_fresh_select >= self.fresh_select_floor
        engaged = _peak(self._m_range) > self.m_cross_moved_floor
        learning = self._n_updates_last > 0
        arbitration_ran = self._n_gate_samples > 0
        limbic_can_win = self._n_limbic_ge_motor > 0
        not_saturated = not self.saturated
        # MECH-464 MANDATORY non-vacuity gate: independent of the
        # learned-cross-loop `gate_ready` below (a D1/D2 sweep need not enable
        # use_learned_cross_loop_arbitration at all) -- "if [the straddle
        # fraction] is ~0 the run is vacuous and MUST be scored
        # precondition_unmet, not as a null". Reported False when d1d2 never
        # ran at all, distinct from a run where it ran but straddled nothing.
        d1d2_ran = self._n_d1d2_active > 0
        straddle_mean_max = max(_mean(self._assoc_straddle), _mean(self._limbic_straddle))
        straddle_nonvacuous = bool(d1d2_ran and straddle_mean_max >= self.straddle_frac_floor)
        return {
            "gate_fresh_selects_sufficient": bool(fresh_ok),
            "gate_arbitration_engaged": bool(arbitration_ran),
            "gate_learning_engaged": bool(engaged and learning),
            "gate_limbic_can_win": bool(limbic_can_win),
            "gate_parity_not_saturated": bool(not_saturated),
            "gate_d1_d2_ran": bool(d1d2_ran),
            "gate_straddle_nonvacuous": straddle_nonvacuous,
            "gate_ready": bool(
                fresh_ok
                and arbitration_ran
                and engaged
                and learning
                and not_saturated
            ),
        }
