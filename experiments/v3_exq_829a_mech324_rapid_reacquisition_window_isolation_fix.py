"""V3-EXQ-829a -- MECH-324 relapse falsifier (a) RE-RUN: reacquisition-window
isolation fix (supersedes V3-EXQ-829).

BUG FIX, NOT A REDESIGN. The scientific question is UNCHANGED from V3-EXQ-829:
dissolve a crystallised chunk, then re-present the same consistent, above-baseline
regime -- re-formation must occur in FAR FEWER than R_min repetitions, predicted
R_reacq = R_min * f_reacq. V3-EXQ-829 (ree-v3 77e3ddc, run
v3_exq_829_mech324_rapid_reacquisition_falsifier_20260727T170539Z_v3) FALSIFIED
this: median_r_reacq was flat at 90 (W=100) / 28 (W=30) across every tested
f_reacq, and r_reacq/window_trials measured 0.908 +/- 0.029 -- the DV tracked the
WINDOW, not the repetition bar. That FAIL's own "LIKELY READING" (ree-v3/CLAUDE.md
MECH-324 entry) diagnosed the implementation, not the claim, as broken: the
revival gate's variance/mean readout came from the accumulator's whole-lifetime
tally, saturated at the moment of dissolution with the very high-variance stream
that caused it, so clearing the variance gate took ~window_trials trials
regardless of the reacquisition_repetition_factor bar.

THE FIX under test (IGW-20260731-196; design doc
REE_assembly/docs/architecture/mech324_reacquisition_window_isolation.md; ree-v3
commit implementing it precedes this queue entry). New sub-switch
use_reacquisition_window_isolation (default False, nested under
use_chunk_dissolution_retention) routes the revival gate through a dedicated
per-chunk window (ChunkedPrimitive.reacquisition_outcomes) populated only from
real executions SINCE the most recent dissolution, instead of the contaminated
whole-lifetime tally. This run adds that flag as a THIRD ablation axis (ON vs
OFF), crossed with the EXACT SAME f_reacq sweep and window_trials sweep V3-EXQ-829
used, so the isolation-OFF arms are the direct apples-to-apples reproduction of
829's own (confirmed-buggy) measurement -- an in-run negative control, not a
separate claim.

WHAT CARRIES OVER FROM V3-EXQ-829 UNCHANGED (see that script's docstring for the
full derivation; not re-litigated here): the OPERATOR-LEVEL driver rationale
(V3-EXQ-810 found the accumulator silent under agent control), the OUTCOME_SPAN
pre-registration (claim's own variance thresholds are otherwise arithmetically
unreachable), the DV-symmetry audit (r_reacq is an ordered stopping time; none of
retention/f_reacq/window_trials/isolation is invariant under it -- isolation
toggles WHICH window the same two unchanged gates read, so it is exactly the same
symmetry class as window_trials itself), the readiness precondition set, and the
CLAIM_IDS (MECH-324 primary, MECH-323 for the joint formation gate).

WHAT CHANGES. The isolation axis makes C1/C2 (the load-bearing criteria)
measured on the ISOLATION-ON default arm rather than the isolation-agnostic
default arm 829 used (829 predates the flag; every one of its ON arms is what
this run calls ISOOFF). A NEW reported (non-gating) criterion, C6, confirms the
ISOOFF arms reproduce 829's flat signature at THIS run's seeds/config -- if C6
did not hold, the two runs would not be measuring the same substrate defect and
the "fix" framing would be unsupported. The MILD-dissolution robustness arm
(829's C5) is DROPPED to bound this run's grid size at 2x the arms rather than
2x the arms plus a robustness check that isolation does not need re-litigating
(the mechanism MILD checked -- is the flat signature an artifact of the specific
dissolution stream -- is orthogonal to which window is read, and 829 already
confirmed it was not: med_mild within 25% of med_severe). A design-doc-cited
successor experiment MAY re-add it if isolation x dissolution-stream interaction
becomes a live question.

ARM REUSE. V3-EXQ-829's OFF-retention and ON-retention cells ARE NOT
mechanically reusable here: 829 fingerprinted every cell with the DEFAULT
include_driver_script_in_hash=True (its own docstring states this is a
deliberate refuse-by-default choice), so a different driver script -- this one
-- can never produce a matching fingerprint regardless of config. This was
checked (GOV-REUSE-1) before authoring: the decisive readout (median_r_reacq
under the corrected substrate) is NOT recorded anywhere, and 829's flat-arm
cells are not recoverable via try_reuse_cell for the reason above, so a fresh
run is the only path. This run follows 829's own precedent and likewise
fingerprints with the default include_driver_script_in_hash=True: the cells
here are pure-python operator trials costing milliseconds, so the reuse surface
has no value and a canonical baseline module would be pure overhead.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.policy.policy_chunking import (  # noqa: E402
    ChunkState,
    PolicyChunking,
    PolicyChunkingConfig,
)
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_829a_mech324_rapid_reacquisition_window_isolation_fix"
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "v3_exq_829_mech324_rapid_reacquisition_falsifier"

# Same claim tagging as 829 -- the scientific question is unchanged; see module
# docstring "BUG FIX, NOT A REDESIGN".
CLAIM_IDS = ["MECH-324", "MECH-323"]
PARENT_CLAIM_ID = "ARC-071"

# Identical seeds to 829 for direct comparability of the ISOOFF arms.
SEEDS = [11, 23, 37, 43, 59, 71]

# ---------------------------------------------------------------------------
# Pre-registered configuration. IDENTICAL to V3-EXQ-829 -- nothing about the
# claim's own thresholds changes; only which window the revival gate reads.
# ---------------------------------------------------------------------------
R_MIN = 20
VARIANCE_LOW = 0.15
VARIANCE_HIGH = 0.45
EVALUATIVE_MARGIN = 0.05
CRYSTALLISATION_MIN = 5
DISSOLVE_TRIALS = 50

MAX_LIBRARY_SIZE = 256
MAX_TRACKED_SEQUENCES = 512

OUTCOME_SPAN = 2.0
OUTCOME_MAX_VARIANCE = (OUTCOME_SPAN * OUTCOME_SPAN) / 4.0
TARGET_OUTCOME = 1.6
FILLER_OUTCOME = 0.4
CONSISTENT_NOISE_SD = 0.02

DISSOLVE_SEVERE = (2.0, 0.0)

TARGET_SEQUENCE = (1, 2, 3)
FILLER_SEQUENCE = (0, 4)

P0_MAX_TRIALS = 400
P1_MAX_TRIALS = 400
P2_CAP_TRIALS = 250

TRIALS_PER_RUN = P0_MAX_TRIALS + P1_MAX_TRIALS + P2_CAP_TRIALS

# ---------------------------------------------------------------------------
# Pre-registered acceptance thresholds -- IDENTICAL values to 829, now applied
# to the isolation-ON arms as the corrected substrate's primary test.
# ---------------------------------------------------------------------------
SCALING_RHO_FLOOR = 0.8
F_REACQ_SWEEP = [1.0, 0.5, 0.25, 0.1]
F_REACQ_DEFAULT = 0.25
WINDOW_DEFAULT = 100
WINDOW_SHORT = 30

BOOL_FLOOR = 0.5


def _arm_specs() -> List[Dict[str, Any]]:
    """The pre-registered arm grid: 829's grid with `window_isolation` crossed
    into every ON-retention arm. OFF-retention arms are unaffected by the flag
    (PolicyChunkingConfig.validate() requires retention=True to set isolation=
    True at all) and are kept exactly as 829 had them, at ISOOFF-equivalent
    config, to preserve the C3 erasure-discrimination test unchanged.
    """
    arms: List[Dict[str, Any]] = []
    for w in (WINDOW_SHORT, WINDOW_DEFAULT):
        arms.append({
            "arm_id": f"ARM_OFF_W{w}",
            "retention": False,
            "window_isolation": False,
            "f_reacq": F_REACQ_DEFAULT,
            "window_trials": w,
        })
    for iso in (False, True):
        iso_tag = "ISOON" if iso else "ISOOFF"
        for w in (WINDOW_SHORT, WINDOW_DEFAULT):
            for f in F_REACQ_SWEEP:
                arms.append({
                    "arm_id": f"ARM_ON_F{int(round(f * 100)):03d}_W{w}_{iso_tag}",
                    "retention": True,
                    "window_isolation": iso,
                    "f_reacq": f,
                    "window_trials": w,
                })
    return arms


ARMS = _arm_specs()

# ---------------------------------------------------------------------------
# Readiness preconditions -- IDENTICAL to V3-EXQ-829. The isolation flag
# changes which window the revival gate reads; it does not change any of what
# these preconditions certify (the driver reached a reachable variance regime,
# crystallised, dissolved, stayed under the FIFO/library caps).
# ---------------------------------------------------------------------------
PRECONDITIONS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="dissolution_variance_reachable",
        description=(
            "Max attainable population variance on the pre-registered outcome "
            "span (span^2/4) must exceed F_high, else CRYSTALLISED -> "
            "DISSOLVING can never fire."
        ),
        control="closed-form bound on the pre-registered outcome span",
        threshold=VARIANCE_HIGH,
        direction="lower",
        kind="readiness",
        structural_max=lambda ctx: OUTCOME_MAX_VARIANCE,
    ),
    PreconditionSpec(
        name="p0_chunk_crystallised",
        description="The target sequence reached CRYSTALLISED in P0.",
        control="target sequence driven at a consistent above-baseline outcome",
        threshold=BOOL_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="p1_chunk_dissolved",
        description="The target chunk reached DISSOLVED in P1.",
        control="target driven at a supra-F_high variance stream",
        threshold=BOOL_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="evaluative_separation_supra_margin",
        description=(
            "Target outcome mean minus the running baseline must exceed the "
            "evaluative margin, else MECH-323 gate (3) refuses re-formation "
            "for a reason unrelated to the repetition bar."
        ),
        control="measured on the re-presented regime at end of P2",
        threshold=EVALUATIVE_MARGIN,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="tally_below_fifo_cap",
        description="Distinct tracked sub-sequences must stay below max_tracked_sequences.",
        control="bounded action alphabet",
        threshold=float(MAX_TRACKED_SEQUENCES),
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="library_below_eviction_cap",
        description="Library size must stay below max_library_size.",
        control="bounded action alphabet plus a deliberately oversized cap",
        threshold=float(MAX_LIBRARY_SIZE),
        direction="upper",
        kind="readiness",
    ),
]


def _build_config(arm: Dict[str, Any]) -> PolicyChunkingConfig:
    cfg = PolicyChunkingConfig(
        use_policy_chunking=True,
        use_chunk_maintenance=True,
        use_chunk_dissolution_retention=bool(arm["retention"]),
        use_reacquisition_window_isolation=bool(arm["window_isolation"]),
        reacquisition_repetition_factor=float(arm["f_reacq"]),
        min_repetitions=R_MIN,
        window_trials=int(arm["window_trials"]),
        variance_low=VARIANCE_LOW,
        variance_high=VARIANCE_HIGH,
        evaluative_margin=EVALUATIVE_MARGIN,
        crystallisation_min=CRYSTALLISATION_MIN,
        dissolve_trials=DISSOLVE_TRIALS,
        max_library_size=MAX_LIBRARY_SIZE,
        max_tracked_sequences=MAX_TRACKED_SEQUENCES,
        use_chunk_replay_origin_path=False,
    )
    cfg.validate()
    return cfg


def _config_slice(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": arm["arm_id"],
        "retention": bool(arm["retention"]),
        "window_isolation": bool(arm["window_isolation"]),
        "f_reacq": float(arm["f_reacq"]),
        "window_trials": int(arm["window_trials"]),
        "min_repetitions": R_MIN,
        "variance_low": VARIANCE_LOW,
        "variance_high": VARIANCE_HIGH,
        "evaluative_margin": EVALUATIVE_MARGIN,
        "crystallisation_min": CRYSTALLISATION_MIN,
        "dissolve_trials": DISSOLVE_TRIALS,
        "max_library_size": MAX_LIBRARY_SIZE,
        "max_tracked_sequences": MAX_TRACKED_SEQUENCES,
        "outcome_span": OUTCOME_SPAN,
        "target_outcome": TARGET_OUTCOME,
        "filler_outcome": FILLER_OUTCOME,
        "consistent_noise_sd": CONSISTENT_NOISE_SD,
        "target_sequence": list(TARGET_SEQUENCE),
        "filler_sequence": list(FILLER_SEQUENCE),
        "schedule": {
            "p0_max": P0_MAX_TRIALS,
            "p1_max": P1_MAX_TRIALS,
            "p2_cap": P2_CAP_TRIALS,
        },
    }


class _Driver:
    """Identical driver to V3-EXQ-829 -- see that script for full rationale."""

    def __init__(self, cfg: PolicyChunkingConfig, seed: int) -> None:
        self.pc = PolicyChunking(cfg)
        self.rng = random.Random(seed)
        self.trials = 0

    def _emit(self, sequence: Tuple[int, ...], outcome: float) -> None:
        for action in sequence:
            self.pc.record_step(int(action))
        self.pc.note_outcome(float(outcome))

    def trial(self, target_outcome: float) -> None:
        self._emit(TARGET_SEQUENCE, target_outcome)
        self._emit(FILLER_SEQUENCE,
                   FILLER_OUTCOME + self.rng.gauss(0.0, CONSISTENT_NOISE_SD))
        self.trials += 1

    def consistent_outcome(self) -> float:
        return TARGET_OUTCOME + self.rng.gauss(0.0, CONSISTENT_NOISE_SD)

    def target(self):
        return self.pc.library.get(TARGET_SEQUENCE)

    def target_state(self) -> Optional[ChunkState]:
        chunk = self.target()
        return chunk.state if chunk is not None else None

    def baseline(self) -> float:
        hist = self.pc.accumulator._outcome_history
        return statistics.fmean(hist) if hist else 0.0

    def target_mean(self) -> float:
        bucket = self.pc.accumulator._tally.get(TARGET_SEQUENCE, ())
        return statistics.fmean(bucket) if bucket else 0.0

    def n_tracked(self) -> int:
        return len(self.pc.accumulator._tally)

    def lib_state(self) -> Dict[str, Any]:
        return self.pc.library.get_state()


def _run_cell(arm: Dict[str, Any], seed: int, progress_prefix: str) -> Dict[str, Any]:
    """P0 acquire -> P1 dissolve -> P2 reacquire, for one (arm, seed) cell.

    IDENTICAL to V3-EXQ-829's cell logic -- the only difference between arms is
    the config the isolation flag selects inside PolicyChunking itself.
    """
    cfg = _build_config(arm)
    drv = _Driver(cfg, seed)
    hi, lo = DISSOLVE_SEVERE

    def _tick(done: int) -> None:
        if done % 100 == 0:
            print(f"  [train] {progress_prefix} ep {done}/{TRIALS_PER_RUN}",
                  flush=True)

    r_acq_form: Optional[int] = None
    r_acq_crystallise: Optional[int] = None
    for t in range(P0_MAX_TRIALS):
        drv.trial(drv.consistent_outcome())
        _tick(drv.trials)
        chunk = drv.target()
        if chunk is not None and r_acq_form is None:
            r_acq_form = t + 1
        if chunk is not None and chunk.state is ChunkState.CRYSTALLISED:
            r_acq_crystallise = t + 1
            break
    p0_trials = drv.trials

    dissolved_at: Optional[int] = None
    if r_acq_crystallise is not None:
        for t in range(P1_MAX_TRIALS):
            drv.trial(hi if t % 2 == 0 else lo)
            _tick(drv.trials)
            if drv.target_state() is ChunkState.DISSOLVED:
                dissolved_at = t + 1
                break
    p1_trials = drv.trials - p0_trials

    r_reacq: Optional[int] = None
    if dissolved_at is not None:
        for t in range(P2_CAP_TRIALS):
            drv.trial(drv.consistent_outcome())
            _tick(drv.trials)
            chunk = drv.target()
            if chunk is not None and chunk.n_reacquisitions > 0:
                r_reacq = t + 1
                break
    p2_trials = drv.trials - p0_trials - p1_trials

    lib = drv.lib_state()
    chunk = drv.target()
    forced_bar = cfg.reacquisition_min_repetitions
    row: Dict[str, Any] = {
        "arm_id": arm["arm_id"],
        "seed": seed,
        "retention": bool(arm["retention"]),
        "window_isolation": bool(arm["window_isolation"]),
        "f_reacq": float(arm["f_reacq"]),
        "window_trials": int(arm["window_trials"]),
        "r_acq_form": r_acq_form,
        "r_acq_crystallise": r_acq_crystallise,
        "r_reacq": r_reacq,
        "r_reacq_censored": r_reacq is None,
        "r_reacq_cap": P2_CAP_TRIALS,
        "forced_bar": forced_bar,
        "r_reacq_minus_forced_bar": (None if r_reacq is None
                                     else r_reacq - forced_bar),
        "r_reacq_over_window": (None if r_reacq is None
                                else r_reacq / float(arm["window_trials"])),
        "dissolved_at": dissolved_at,
        "final_state": (chunk.state.value if chunk is not None else None),
        "n_reacquisitions": (chunk.n_reacquisitions if chunk is not None else 0),
        "n_dissolutions": (chunk.n_dissolutions if chunk is not None else 0),
        "lib_state": lib,
        "trials_p0": p0_trials,
        "trials_p1": p1_trials,
        "trials_p2": p2_trials,
        "measured_dissolution_variance_reachable": OUTCOME_MAX_VARIANCE,
        "measured_p0_chunk_crystallised": 1.0 if r_acq_crystallise is not None else 0.0,
        "measured_p1_chunk_dissolved": 1.0 if dissolved_at is not None else 0.0,
        "measured_evaluative_separation": drv.target_mean() - drv.baseline(),
        "measured_n_tracked": float(drv.n_tracked()),
        "measured_lib_size": float(lib.get("chunk_lib_size", 0)),
    }
    return row


def _gate_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    arm_ctx = {
        "arm_id": row["arm_id"],
        "retention": row["retention"],
        "window_isolation": row["window_isolation"],
        "f_reacq": row["f_reacq"],
        "window_trials": row["window_trials"],
        "seed": row["seed"],
    }
    measured = {
        "dissolution_variance_reachable": row["measured_dissolution_variance_reachable"],
        "p0_chunk_crystallised": row["measured_p0_chunk_crystallised"],
        "p1_chunk_dissolved": row["measured_p1_chunk_dissolved"],
        "evaluative_separation_supra_margin": row["measured_evaluative_separation"],
        "tally_below_fifo_cap": row["measured_n_tracked"],
        "library_below_eviction_cap": row["measured_lib_size"],
    }
    return evaluate_arm_gate(
        arm_id=f"{row['arm_id']}/seed{row['seed']}",
        arm_ctx=arm_ctx,
        specs=PRECONDITIONS,
        measured=measured,
    )


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rho. Returns None when either side has no rank variation
    (a flat DV does not scale; that must fail C2, not read as rho=0)."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def _ranks(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["arm_id"] == arm_id]


def _uncensored_r_reacq(rows: List[Dict[str, Any]]) -> List[float]:
    return [float(r["r_reacq"]) for r in rows if r["r_reacq"] is not None]


def _sweep(rows: List[Dict[str, Any]], iso_tag: str, w: int) -> Tuple[
        List[float], List[float], Dict[str, Any]]:
    xs: List[float] = []
    ys: List[float] = []
    detail: Dict[str, Any] = {}
    for f in F_REACQ_SWEEP:
        arm_id = f"ARM_ON_F{int(round(f * 100)):03d}_W{w}_{iso_tag}"
        vals = _uncensored_r_reacq(_arm_rows(rows, arm_id))
        med = _median(vals)
        detail[arm_id] = {"f_reacq": f, "median_r_reacq": med,
                          "n_uncensored": len(vals),
                          "forced_bar": max(1, math.ceil(R_MIN * f))}
        if med is not None:
            xs.append(f)
            ys.append(med)
    return xs, ys, detail


def _analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the pre-registered criteria. No threshold is derived from rows."""
    default_on = f"ARM_ON_F{int(round(F_REACQ_DEFAULT * 100)):03d}_W{WINDOW_DEFAULT}_ISOON"
    default_off = f"ARM_ON_F{int(round(F_REACQ_DEFAULT * 100)):03d}_W{WINDOW_DEFAULT}_ISOOFF"

    # -- C1: rapid reacquisition at the claim defaults, under the FIX (ISOON) --
    d_rows = _arm_rows(rows, default_on)
    d_reacq = _uncensored_r_reacq(d_rows)
    r_acq_all = [float(r["r_acq_form"]) for r in rows if r["r_acq_form"] is not None]
    med_reacq_default = _median(d_reacq)
    med_r_acq = _median(r_acq_all)
    c1_pass = (
        med_reacq_default is not None
        and med_r_acq is not None
        and med_reacq_default < med_r_acq
        and med_reacq_default < float(R_MIN)
    )
    c1_non_degenerate = bool(d_reacq) and med_r_acq is not None

    # -- C2: does reacquisition scale with f_reacq at W=100, ISOLATION ON? -----
    sweep_x, sweep_y, sweep_detail = _sweep(rows, "ISOON", WINDOW_DEFAULT)
    rho_iso_on = _spearman(sweep_x, sweep_y)
    c2_pass = rho_iso_on is not None and rho_iso_on >= SCALING_RHO_FLOOR
    c2_non_degenerate = len(sweep_y) >= 3

    # -- C3: retention vs erasure. OFF must never revive; ON must sometimes. --
    off_rows = [r for r in rows if not r["retention"]]
    on_rows = [r for r in rows if r["retention"]]
    off_any_revived = any(r["n_reacquisitions"] > 0 for r in off_rows)
    on_any_revived = any(r["n_reacquisitions"] > 0 for r in on_rows)
    c3_pass = (not off_any_revived) and on_any_revived
    off_dissolved = [r for r in off_rows if r["dissolved_at"] is not None]
    c3_non_degenerate = len(off_dissolved) >= max(1, len(off_rows) // 2)

    # -- C4 (reported): short-window scaling discriminator, ISOLATION ON ------
    sweep_x_s, sweep_y_s, sweep_detail_short = _sweep(rows, "ISOON", WINDOW_SHORT)
    rho_iso_on_short = _spearman(sweep_x_s, sweep_y_s)
    c4_pass = rho_iso_on_short is not None and rho_iso_on_short >= SCALING_RHO_FLOOR
    c4_non_degenerate = len(sweep_y_s) >= 3

    # -- C5 (reported): does the ON arm actually change the DV vs OFF? --------
    # Direct paired comparison at the claim-default cell, same seeds, same
    # config apart from the flag -- this is the single number that answers
    # "does the fix matter" without going through the sweep at all.
    off_default_rows = _arm_rows(rows, default_off)
    off_default_reacq = _uncensored_r_reacq(off_default_rows)
    med_reacq_off_default = _median(off_default_reacq)
    c5_pass = (
        med_reacq_default is not None
        and med_reacq_off_default is not None
        and med_reacq_default < med_reacq_off_default
    )
    c5_non_degenerate = bool(d_reacq) and bool(off_default_reacq)

    # -- C6 (reported): does ISOLATION OFF reproduce 829's flat signature at
    #    THIS run's seeds/config? If not, the two runs are not comparable and
    #    the "fix" framing is unsupported.
    sweep_x_off, sweep_y_off, sweep_detail_off = _sweep(rows, "ISOOFF", WINDOW_DEFAULT)
    rho_iso_off = _spearman(sweep_x_off, sweep_y_off)
    c6_pass = rho_iso_off is None or rho_iso_off < SCALING_RHO_FLOOR
    c6_non_degenerate = len(sweep_y_off) >= 3

    on_uncensored_iso_on = [r for r in on_rows
                            if r["window_isolation"] and r["r_reacq"] is not None]
    all_iso_on_forced_bar = bool(on_uncensored_iso_on) and all(
        r["r_reacq_minus_forced_bar"] == 0 for r in on_uncensored_iso_on)
    if all_iso_on_forced_bar:
        c2_non_degenerate = False

    w_ratios_off = [r["r_reacq_over_window"] for r in on_rows
                    if not r["window_isolation"] and r["r_reacq_over_window"] is not None]

    overall_pass = c1_pass and c2_pass and c3_pass

    metrics: Dict[str, Any] = {
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
        "c4_pass": c4_pass, "c5_pass": c5_pass, "c6_pass": c6_pass,
        "median_r_acq_form": med_r_acq,
        "median_r_reacq_default_arm_iso_on": med_reacq_default,
        "median_r_reacq_default_arm_iso_off": med_reacq_off_default,
        "r_min": R_MIN,
        "forced_bar_default_arm": max(1, math.ceil(R_MIN * F_REACQ_DEFAULT)),
        "scaling_rho_iso_on_w100": rho_iso_on,
        "scaling_rho_iso_on_w30": rho_iso_on_short,
        "scaling_rho_iso_off_w100": rho_iso_off,
        "scaling_rho_floor": SCALING_RHO_FLOOR,
        "f_reacq_sweep_iso_on_w100": sweep_detail,
        "f_reacq_sweep_iso_on_w30": sweep_detail_short,
        "f_reacq_sweep_iso_off_w100": sweep_detail_off,
        "off_arm_any_revived": off_any_revived,
        "on_arm_any_revived": on_any_revived,
        "n_off_cells_dissolved": len(off_dissolved),
        "n_off_cells": len(off_rows),
        "r_reacq_over_window_mean_iso_off": (
            statistics.fmean(w_ratios_off) if w_ratios_off else None),
        "r_reacq_over_window_stdev_iso_off": (
            statistics.pstdev(w_ratios_off) if len(w_ratios_off) > 1 else None),
        "all_iso_on_cells_sit_on_forced_bar": all_iso_on_forced_bar,
        "n_censored_cells": sum(1 for r in rows if r["r_reacq_censored"]),
        "n_cells": len(rows),
    }

    criteria = [
        {"name": "C1_reacquisition_faster_than_acquisition_isolation_on",
         "load_bearing": True, "passed": c1_pass},
        {"name": "C2_reacquisition_scales_with_f_reacq_isolation_on",
         "load_bearing": True, "passed": c2_pass},
        {"name": "C3_retention_discriminates_erasure",
         "load_bearing": True, "passed": c3_pass},
        {"name": "C4_short_window_scaling_isolation_on",
         "load_bearing": False, "passed": c4_pass},
        {"name": "C5_isolation_on_faster_than_isolation_off_paired",
         "load_bearing": False, "passed": c5_pass},
        {"name": "C6_isolation_off_reproduces_829_flat_signature",
         "load_bearing": False, "passed": c6_pass},
    ]
    non_degenerate_map = {
        "C1": c1_non_degenerate, "C2": c2_non_degenerate, "C3": c3_non_degenerate,
        "C4": c4_non_degenerate, "C5": c5_non_degenerate, "C6": c6_non_degenerate,
    }
    return {
        "overall_pass": overall_pass,
        "metrics": metrics,
        "criteria": criteria,
        "criteria_non_degenerate": non_degenerate_map,
    }


def _criterion_owning_cells(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    default_on = f"ARM_ON_F{int(round(F_REACQ_DEFAULT * 100)):03d}_W{WINDOW_DEFAULT}_ISOON"
    default_off = f"ARM_ON_F{int(round(F_REACQ_DEFAULT * 100)):03d}_W{WINDOW_DEFAULT}_ISOOFF"
    sweep_on_w100 = {f"ARM_ON_F{int(round(f * 100)):03d}_W{WINDOW_DEFAULT}_ISOON"
                     for f in F_REACQ_SWEEP}
    sweep_on_w30 = {f"ARM_ON_F{int(round(f * 100)):03d}_W{WINDOW_SHORT}_ISOON"
                    for f in F_REACQ_SWEEP}
    sweep_off_w100 = {f"ARM_ON_F{int(round(f * 100)):03d}_W{WINDOW_DEFAULT}_ISOOFF"
                      for f in F_REACQ_SWEEP}

    def _cells(pred) -> List[str]:
        return [f"{r['arm_id']}/seed{r['seed']}" for r in rows if pred(r)]

    return {
        "C1": _cells(lambda r: r["arm_id"] == default_on),
        "C2": _cells(lambda r: r["arm_id"] in sweep_on_w100),
        "C3": _cells(lambda r: not r["retention"] or r["arm_id"] == default_on),
        "C4": _cells(lambda r: r["arm_id"] in sweep_on_w30),
        "C5": _cells(lambda r: r["arm_id"] in (default_on, default_off)),
        "C6": _cells(lambda r: r["arm_id"] in sweep_off_w100),
    }


def _label_for(analysis: Dict[str, Any]) -> str:
    m = analysis["metrics"]
    if not analysis["criteria_non_degenerate"]["C1"]:
        return "reacquisition_dv_unmeasured"
    if analysis["overall_pass"]:
        return "reacquisition_window_isolation_fix_confirmed"
    if m["c3_pass"] and not m["c1_pass"] and not m["c2_pass"]:
        return "retention_real_but_isolation_fix_did_not_resolve_falsifier"
    if not m["c3_pass"]:
        return "retention_did_not_discriminate_erasure"
    return "isolation_fix_partially_resolved_falsifier"


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    arms = ARMS
    rows: List[Dict[str, Any]] = []
    arm_gates: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []

    for arm in arms:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            prefix = f"{arm['arm_id']} seed={seed}"
            with arm_cell(
                seed,
                config_slice=_config_slice(arm),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_cell(arm, seed, prefix)
                cell.stamp(row)
            rows.append(row)
            arm_gates.append(_gate_for_row(row))
            arm_results.append(row)
            rr = row["r_reacq"]
            print(f"  r_acq={row['r_acq_form']} r_reacq="
                  f"{'CENSORED' if rr is None else rr} "
                  f"bar={row['forced_bar']} iso={row['window_isolation']} "
                  f"state={row['final_state']}", flush=True)
            print(f"verdict: {'PASS' if rr is not None else 'FAIL'}", flush=True)

    analysis = _analyse(rows)
    per_arm = aggregate_arm_gates(arm_gates)

    green_cells = set(per_arm.get("green_arms", []))
    owning_cells = _criterion_owning_cells(rows)
    for name, cells in owning_cells.items():
        gate_ok = bool(cells) and all(c in green_cells for c in cells)
        analysis["criteria_non_degenerate"][name] = (
            analysis["criteria_non_degenerate"][name] and gate_ok)

    outcome = "PASS" if analysis["overall_pass"] else "FAIL"
    m = analysis["metrics"]

    if analysis["overall_pass"]:
        mech324_dir = "supports"
    elif not analysis["criteria_non_degenerate"]["C1"]:
        mech324_dir = "unknown"
    elif m["c3_pass"]:
        mech324_dir = "mixed"
    else:
        mech324_dir = "weakens"

    med_acq = m["median_r_acq_form"]
    mech323_dir = ("supports" if med_acq is not None and med_acq <= float(R_MIN)
                   else ("mixed" if med_acq is not None else "unknown"))

    non_degenerate = bool(analysis["criteria_non_degenerate"]["C1"])
    degeneracy_reason = (
        "" if non_degenerate else
        "The isolation-ON default arm produced no uncensored r_reacq, so the "
        "load-bearing DV was never measured under the fix."
    )

    return {
        "outcome": outcome,
        "metrics": m,
        "per_seed_rows": rows,
        "arm_results": arm_results,
        "evidence_direction_per_claim": {
            "MECH-324": mech324_dir,
            "MECH-323": mech323_dir,
        },
        "interpretation": {
            "label": _label_for(analysis),
            "criteria": analysis["criteria"],
            "criteria_non_degenerate": analysis["criteria_non_degenerate"],
            "preconditions": per_arm.get("adjudication_preconditions",
                                         per_arm.get("preconditions", [])),
            "per_arm_gate": per_arm,
            "criterion_owning_cells": _criterion_owning_cells(rows),
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Single seed, manifest relocated out of evidence/.")
    args = parser.parse_args()

    t0 = time.perf_counter()

    arm_contexts = [
        {"arm_id": a["arm_id"], "retention": a["retention"],
         "window_isolation": a["window_isolation"],
         "f_reacq": a["f_reacq"], "window_trials": a["window_trials"]}
        for a in ARMS
    ]
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, arm_contexts)

    result = run_experiment(dry_run=args.dry_run)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    full_config = {
        "r_min": R_MIN,
        "variance_low": VARIANCE_LOW,
        "variance_high": VARIANCE_HIGH,
        "evaluative_margin": EVALUATIVE_MARGIN,
        "crystallisation_min": CRYSTALLISATION_MIN,
        "dissolve_trials": DISSOLVE_TRIALS,
        "max_library_size": MAX_LIBRARY_SIZE,
        "max_tracked_sequences": MAX_TRACKED_SEQUENCES,
        "outcome_span": OUTCOME_SPAN,
        "outcome_max_variance": OUTCOME_MAX_VARIANCE,
        "target_outcome": TARGET_OUTCOME,
        "filler_outcome": FILLER_OUTCOME,
        "consistent_noise_sd": CONSISTENT_NOISE_SD,
        "dissolve_severe": list(DISSOLVE_SEVERE),
        "f_reacq_sweep": F_REACQ_SWEEP,
        "f_reacq_default": F_REACQ_DEFAULT,
        "window_default": WINDOW_DEFAULT,
        "window_short": WINDOW_SHORT,
        "scaling_rho_floor": SCALING_RHO_FLOOR,
        "p0_max_trials": P0_MAX_TRIALS,
        "p1_max_trials": P1_MAX_TRIALS,
        "p2_cap_trials": P2_CAP_TRIALS,
        "trials_per_run": TRIALS_PER_RUN,
        "arm_config_slices": {a["arm_id"]: _config_slice(a) for a in ARMS},
    }

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "parent_claim_id": PARENT_CLAIM_ID,
        "supersedes": SUPERSEDES,
        "evidence_direction": (
            "supports" if result["outcome"] == "PASS" else
            ("mixed" if result["evidence_direction_per_claim"]["MECH-324"] == "mixed"
             else "weakens")
        ),
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "custom_information": {
            "bug_fix_context": (
                "V3-EXQ-829 (ree-v3 77e3ddc) confirmed median_r_reacq flat "
                "across every tested f_reacq, r_reacq/window_trials = "
                "0.908 +/- 0.029. Root cause: the revival gate read the "
                "accumulator's whole-lifetime tally, saturated at dissolution "
                "with the contaminating stream. This run adds "
                "use_reacquisition_window_isolation as a THIRD ablation axis "
                "to test the fix against the SAME f_reacq/window_trials grid."
            ),
            "gov_reuse_1_check": (
                "829's cells fingerprinted with the DEFAULT "
                "include_driver_script_in_hash=True (829's own docstring: "
                "deliberate refuse-by-default), so a different driver script "
                "can never match regardless of config. Checked before "
                "authoring: not recoverable via try_reuse_cell. This run "
                "fingerprints the same way for the same reason (pure-python "
                "cells, no reuse value)."
            ),
            "transfer_limit": (
                "Unchanged from 829: Barnes 2005 / Bouton 2012 extinguish an "
                "object far simpler than a multi-element composed chunked "
                "primitive; f_reacq = 0.25 remains an uncalibrated "
                "engineering default. This run tests direction and scaling, "
                "never the value of f_reacq."
            ),
            "driver_rationale": (
                "Operator-level driver, unchanged from 829: V3-EXQ-810 found "
                "the accumulator silent under agent control."
            ),
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )

    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"median r_acq={m0(result,'median_r_acq_form')} "
          f"median r_reacq(ISOON default)={m0(result,'median_r_reacq_default_arm_iso_on')} "
          f"median r_reacq(ISOOFF default)={m0(result,'median_r_reacq_default_arm_iso_off')} "
          f"R_min={R_MIN} forced_bar={m0(result,'forced_bar_default_arm')}",
          flush=True)
    print(f"C1={result['metrics']['c1_pass']} C2={result['metrics']['c2_pass']} "
          f"C3={result['metrics']['c3_pass']} C4={result['metrics']['c4_pass']} "
          f"C5={result['metrics']['c5_pass']} C6={result['metrics']['c6_pass']} "
          f"rho_ISOON_W{WINDOW_DEFAULT}={m0(result,'scaling_rho_iso_on_w100')} "
          f"rho_ISOON_W{WINDOW_SHORT}={m0(result,'scaling_rho_iso_on_w30')} "
          f"rho_ISOOFF_W{WINDOW_DEFAULT}={m0(result,'scaling_rho_iso_off_w100')}",
          flush=True)
    print(f"censored={result['metrics']['n_censored_cells']}/"
          f"{result['metrics']['n_cells']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


def m0(result: Dict[str, Any], key: str) -> str:
    """Format a possibly-None metric for an ASCII-only print."""
    val = result["metrics"].get(key)
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
