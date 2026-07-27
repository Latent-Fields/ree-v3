"""V3-EXQ-829 -- MECH-324 relapse falsifier (a): RAPID REACQUISITION.

THE FALSIFIER (claims.yaml MECH-324, "RELAPSE FALSIFIERS", registered 2026-07-27).
Dissolve a crystallised chunk, then re-present the same consistent, above-baseline
regime. Re-formation must occur in FAR FEWER than R_min repetitions -- predicted
R_reacq = R_min * f_reacq. It is FALSIFIED IF re-formation takes >= R_min
repetitions, OR if the reacquisition count does not scale with f_reacq. The null it
discriminates against is f_reacq = 1.0 (reacquisition no faster than acquisition),
which is exactly what an ERASURE operator predicts.

Substrate under test: ree-v3 6c3e67e5d3 added dissolution-with-retention behind
use_chunk_dissolution_retention (default OFF) plus chunk_reacquisition_repetition_factor
(default 0.25). Before that commit DISSOLVED was an ABSORBING TOMBSTONE that also
permanently BLOCKED re-formation (measured on the contract fixture: 200 further
perfect trials -> 0 re-formations), so the OFF arm here is not merely "erasure", it
is strictly worse than erasure -- which makes it a clean floor for the contrast.

WHY THIS DRIVES THE OPERATOR DIRECTLY AND NOT THE WHOLE AGENT. V3-EXQ-810 (the
ARC-071 readiness probe) found the accumulator SILENT under agent control:
c1_form_seed_frac = 0.333, i.e. only 1 of 3 seeds minted any chunk at all, and its
load-bearing C1 FAILED with label chunk_accumulator_silent. An agent-level
reacquisition run would therefore spend most of its seeds failing to crystallise
anything to dissolve, and would measure THAT READINESS GAP rather than MECH-324.
Falsifier (a) is a claim about the FORMATION-THRESHOLD OPERATOR -- "re-forming a
dormant chunk costs fewer repetitions than forming a new one" -- so the operator is
the correct unit of analysis, and a synthetic action/outcome stream driven straight
into PolicyChunking is the instrument that isolates it. Nothing here touches E3,
the env, or the proposal pool; no behavioural claim is made or could be.

WHAT THIS DOES NOT CLAIM. Not tagged ARC-071: that claim is about policy composition
changing BEHAVIOUR (latency, rollout cost) via chunks in the proposal pool, and
proposal injection is absent from this run entirely. ARC-071 is the parent cluster,
recorded as parent_claim_id, not as evidence. Falsifiers (b) renewal and (c)
resurgence are NOT tested: (b) is structurally blocked (REE_assembly fb29f0bdf8
REFUSED the context-scoping build -- initiation_set is declared-but-never-populated,
both mint() call sites omit it, zero reads repo-wide) and (c) needs library-level
competition that does not exist. (a) is precisely the one testable against the
operator as it stands.

--------------------------------------------------------------------------------
DESIGN AUDIT, done BEFORE queuing (skill Step 3.5). Three things had to be settled.

(1) OUTCOME SCALE -- a structural reachability finding, recorded as a live
    precondition rather than silently worked around. MECH-323 specifies outcome
    variance "on a 0-1 normalised scale", and MECH-324 sets F_high = 0.45. The
    maximum attainable POPULATION variance of values confined to a span S is
    S^2 / 4, so on a unit scale it is 0.25 -- BELOW F_high. On the claim's own
    specified scale, var > F_high is ARITHMETICALLY UNREACHABLE, CRYSTALLISED ->
    DISSOLVING can never fire, and the entire relapse-falsifier family (a)/(b)/(c),
    all of which need a dissolved chunk, has no reachable entry point at the
    pre-registered defaults. policy_chunking.py does not itself clamp the outcome,
    so the scale is a free harness parameter; this run pre-registers OUTCOME_SPAN
    = 2.0 (max attainable variance 1.0), under which BOTH claim thresholds are
    reachable and NEITHER is altered -- F_low stays 0.15 and F_high stays 0.45.
    That is choosing the harness's unit so the claim's own thresholds are testable,
    NOT lowering a threshold to make a gate satisfiable (which the skill forbids
    and which would convert a detected artifact into a citable result). The
    arithmetic is emitted as the precondition dissolution_variance_reachable so a
    reader can recompute it, and the unit-scale finding is carried in
    custom_information for governance regardless of this run's outcome.

(2) IS THE DV FIXED BEFORE THE RUN? (DV-symmetry invariance, per arm.) DV is
    r_reacq, an ORDERED STOPPING TIME: the count of post-dissolution executions
    until the chunk returns to FORMING. Its symmetry group is trivial for the
    manipulations used -- none of the three is a broadcast additive constant under
    an argmax/softmax DV, a monotone rescaling under a rank DV, or a permutation of
    interchangeable units under a set-aggregate DV.
      * retention ON/OFF  -- toggles whether the revival code path exists at all;
        an absorbing tombstone yields a censored DV. Not invariant.
      * f_reacq           -- enters as bar = ceil(R_min * f_reacq), a threshold on
        the very counter the DV measures. Not invariant.
      * window_trials W   -- sets the length of the sliding window the variance
        gate reads. Not invariant.
    The one shape that WOULD have been an arithmetic identity is r_reacq ==
    ceil(R_min * f_reacq) exactly, which is what a naive reading predicts. It is
    NOT what the operator does, because the two unchanged MECH-323 gates are
    evaluated over a sliding window contaminated by the dissolution episode. The
    run therefore records r_reacq_minus_forced_bar per cell and marks C2 degenerate
    if every cell sits exactly on its forced bar -- the identity is TESTED, not
    assumed away.

(3) VERDICT ALIASING -- the reason the W axis exists. Without it, a FAIL cannot
    distinguish "MECH-324's quantitative prediction is wrong" from "the prediction
    is masked by an interaction with the window length". Dissolution is structurally
    slow (T_dissolve = 50 trials of supra-F_high variance are REQUIRED to reach
    DISSOLVED), so at the default W = 100 at least half the variance window is
    necessarily contaminated at the moment of dissolution, whatever stream drove
    it. Crossing f_reacq with W in {30, 100} separates the two: if r_reacq tracks W
    and ignores f_reacq, the reduced bar is inert BY CONSTRUCTION rather than
    merely mis-valued. A MILD-dissolution arm additionally checks the finding is
    not an artifact of this driver's dissolution stream.

AUTHORING PROBE (disclosed for the audit trail; thresholds were NOT tuned to it).
Every threshold below is taken verbatim from the claim's own falsifier text
(r_reacq < R_min; reacquisition count scales with f_reacq), fixed before the probe
and unchanged after. A scratch probe at authoring time showed r_reacq = 28 / 46 / 90
for W = 30 / 50 / 100, FLAT across f_reacq in {1.0, 0.5, 0.25, 0.1}, with
retention-OFF never reviving and r_acq = 20 = R_min exactly. If that reproduces,
BOTH legs of the FALSIFIED condition fire. That expected direction is not a reason
to redesign until it passes: a falsifier that falsifies is the instrument working.
--------------------------------------------------------------------------------

TRANSFER LIMIT (registered on the claim, load-bearing, NOT boilerplate). The
grounding is Barnes et al. 2005 (Nature, 10.1038/nature04053; rodent T-maze
procedural learning) and Bouton et al. 2012 (Behav Processes,
10.1016/j.beproc.2012.03.004; single-response operant contingencies). In both, the
extinguished object is FAR SIMPLER than a multi-element composed chunked primitive,
and REE's outcome dimension is an E3-score-receipt rather than a food reward.
Whether a SEQUENCE extinguishes with the same asymmetry is EXTRAPOLATION, NOT
EVIDENCE. Neither source quantifies a threshold: f_reacq = 0.25 is an UNCALIBRATED
ENGINEERING DEFAULT on exactly the same footing as F_high = 0.45. So this run tests
the DIRECTION and the SCALING of the prediction, never the value of f_reacq.

ARM REUSE. All cells are fingerprinted with the DEFAULT
include_driver_script_in_hash=True, i.e. deliberately NOT minted for cross-driver
reuse. A cell here is a few thousand pure-python trials costing milliseconds and
trains nothing, so the reuse surface has no value, while an excluded-driver hash
without a canonical baseline module would carry real false-HIT risk from an
under-declared config slice. Refuse-by-default is the correct trade at zero cost;
this is a stated judgement, not an omission.
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

EXPERIMENT_TYPE = "v3_exq_829_mech324_rapid_reacquisition_falsifier"
EXPERIMENT_PURPOSE = "evidence"

# MECH-324 is the maintenance/dissolution operator whose rapid-reacquisition
# prediction is under test. MECH-323 is tagged because P0 measures repetitions-to-
# FORMATION against R_min under the joint gate, which is MECH-323's own trigger
# condition -- not inherited from a prior iteration. ARC-071 is deliberately NOT
# tagged: see "WHAT THIS DOES NOT CLAIM" above.
CLAIM_IDS = ["MECH-324", "MECH-323"]
PARENT_CLAIM_ID = "ARC-071"

SEEDS = [11, 23, 37, 43, 59, 71]

# ---------------------------------------------------------------------------
# Pre-registered configuration. Every claim-specified threshold is at its
# registered default and is NOT varied: R_min, F_low, F_high, C_min, T_dissolve.
# ---------------------------------------------------------------------------
R_MIN = 20                 # MECH-323 (1) repetition count
VARIANCE_LOW = 0.15        # F_low, MECH-323 (2)
VARIANCE_HIGH = 0.45       # F_high, MECH-324 (2a)
EVALUATIVE_MARGIN = 0.05   # MECH-323 (3)
CRYSTALLISATION_MIN = 5    # C_min, MECH-324 (1)
DISSOLVE_TRIALS = 50       # T_dissolve, MECH-324 (3)

# Sized ABOVE anything this run can produce. policy_chunking's _evict_one docstring
# is explicit that DISSOLVED is also the dormant pool, so max_library_size eviction
# is the one remaining path by which a dormant trace is genuinely ERASED -- "a
# reacquisition experiment must size max_library_size above the number of chunks it
# expects to dissolve, or it will measure the eviction policy instead of the
# mechanism". Both caps are asserted as ceiling preconditions below.
MAX_LIBRARY_SIZE = 256
MAX_TRACKED_SEQUENCES = 512

# Outcome scale -- see design audit (1). Max attainable population variance is
# OUTCOME_SPAN^2 / 4 = 1.0, so F_high = 0.45 is reachable WITHOUT altering F_high.
OUTCOME_SPAN = 2.0
OUTCOME_MAX_VARIANCE = (OUTCOME_SPAN * OUTCOME_SPAN) / 4.0
TARGET_OUTCOME = 1.6       # consistent, above-baseline regime
FILLER_OUTCOME = 0.4       # keeps the RELATIVE evaluative gate meaningful
CONSISTENT_NOISE_SD = 0.02

# Dissolution streams. SEVERE spans the full scale; MILD is the gentlest stream
# that still clears F_high (population variance 0.49 vs the 0.45 gate), and exists
# only to show the finding is not an artifact of the severe stream.
DISSOLVE_SEVERE = (2.0, 0.0)
DISSOLVE_MILD = (1.6, 0.2)

# Action alphabet kept small ON PURPOSE. note_outcome credits every contiguous
# sub-sequence of length 2..5 ending at the current position, so a wide alphabet
# would push the tally past max_tracked_sequences and FIFO-evict the TARGET's own
# bucket mid-run -- which would destroy the DV silently. Bounded here and asserted.
TARGET_SEQUENCE = (1, 2, 3)
FILLER_SEQUENCE = (0, 4)

# Per-phase trial budgets (target repetitions).
P0_MAX_TRIALS = 400
P1_MAX_TRIALS = 400
P2_CAP_TRIALS = 250        # r_reacq censoring cap; probe max was 90 at W=100
TRIALS_PER_RUN = P0_MAX_TRIALS + P1_MAX_TRIALS + P2_CAP_TRIALS

# ---------------------------------------------------------------------------
# Pre-registered acceptance thresholds. Taken verbatim from the claim's falsifier
# text, defined HERE, never inferred from this run's own statistics.
# ---------------------------------------------------------------------------
SCALING_RHO_FLOOR = 0.8    # C2: reacquisition count must scale with f_reacq
F_REACQ_SWEEP = [1.0, 0.5, 0.25, 0.1]
F_REACQ_DEFAULT = 0.25     # the substrate default under test
WINDOW_DEFAULT = 100       # W, MECH-324 W_maint default -- the claim-default regime
WINDOW_SHORT = 30          # discriminator only; >= R_MIN so the config stays legal
MILD_TOLERANCE_FRAC = 0.25 # C5: MILD within 25% of the matched SEVERE arm

# Precondition thresholds.
BOOL_FLOOR = 0.5           # a 1/0 indicator clears a 0.5 floor iff it is 1


def _arm_specs() -> List[Dict[str, Any]]:
    """The pre-registered arm grid. Order is the manifest's arm order."""
    arms: List[Dict[str, Any]] = []
    for w in (WINDOW_SHORT, WINDOW_DEFAULT):
        arms.append({
            "arm_id": f"ARM_OFF_W{w}",
            "retention": False,
            # Inert when retention is off; recorded so the slice is complete.
            "f_reacq": F_REACQ_DEFAULT,
            "window_trials": w,
            "dissolve_stream": "SEVERE",
        })
    for w in (WINDOW_SHORT, WINDOW_DEFAULT):
        for f in F_REACQ_SWEEP:
            arms.append({
                "arm_id": f"ARM_ON_F{int(round(f * 100)):03d}_W{w}",
                "retention": True,
                "f_reacq": f,
                "window_trials": w,
                "dissolve_stream": "SEVERE",
            })
    arms.append({
        "arm_id": f"ARM_ON_F{int(round(F_REACQ_DEFAULT * 100)):03d}_W{WINDOW_DEFAULT}_MILD",
        "retention": True,
        "f_reacq": F_REACQ_DEFAULT,
        "window_trials": WINDOW_DEFAULT,
        "dissolve_stream": "MILD",
    })
    return arms


ARMS = _arm_specs()

# ---------------------------------------------------------------------------
# Readiness preconditions. Every one applies to EVERY arm -- there is no regime for
# which any of them is not meaningful, so none carries applies_to. They are still
# evaluated PER ARM via precondition_gate so that one arm failing readiness cannot
# vacate a clean arm's result (the V3-EXQ-785 defect).
# ---------------------------------------------------------------------------
PRECONDITIONS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="dissolution_variance_reachable",
        description=(
            "Max attainable population variance on the pre-registered outcome span "
            "(span^2/4) must exceed F_high, else CRYSTALLISED -> DISSOLVING can "
            "never fire and falsifier (a) has no entry point. On the claim's own "
            "0-1 normalised scale this is 0.25 vs F_high 0.45 and is UNREACHABLE."
        ),
        control="closed-form bound on the pre-registered outcome span",
        threshold=VARIANCE_HIGH,
        direction="lower",
        kind="readiness",
        # Design-time proof: the bound is fixed by pre-registered config alone.
        structural_max=lambda ctx: OUTCOME_MAX_VARIANCE,
    ),
    PreconditionSpec(
        name="p0_chunk_crystallised",
        description=(
            "The target sequence reached CRYSTALLISED in P0. Nothing can be "
            "dissolved, and so nothing reacquired, without this."
        ),
        control="target sequence driven at a consistent above-baseline outcome",
        threshold=BOOL_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="p1_chunk_dissolved",
        description=(
            "The target chunk reached DISSOLVED in P1. Without this the P2 measure "
            "is not a REacquisition and 'never revived' would be vacuous."
        ),
        control="target driven at a supra-F_high variance stream",
        threshold=BOOL_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="evaluative_separation_supra_margin",
        description=(
            "Target outcome mean minus the running baseline must exceed the "
            "evaluative margin, else MECH-323 gate (3) refuses re-formation for a "
            "reason that has nothing to do with the repetition bar."
        ),
        control="measured on the re-presented regime at end of P2",
        threshold=EVALUATIVE_MARGIN,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="tally_below_fifo_cap",
        description=(
            "Distinct tracked sub-sequences must stay below max_tracked_sequences, "
            "else the accumulator FIFO-evicts the TARGET's own tally bucket and the "
            "DV is destroyed silently."
        ),
        control="bounded action alphabet",
        threshold=float(MAX_TRACKED_SEQUENCES),
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="library_below_eviction_cap",
        description=(
            "Library size must stay below max_library_size. DISSOLVED IS the dormant "
            "pool, so hitting the cap erases dormant traces and would measure the "
            "eviction policy instead of the mechanism (policy_chunking _evict_one)."
        ),
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
        # MECH-322 carve-out OFF: replay-origin chunks are never revivable by
        # design (revive() fails closed on them), so leaving it on could only
        # confound the measurement.
        use_chunk_replay_origin_path=False,
    )
    cfg.validate()
    return cfg


def _config_slice(arm: Dict[str, Any]) -> Dict[str, Any]:
    """Exactly what this cell's computation reads. Nothing more."""
    return {
        "arm_id": arm["arm_id"],
        "retention": bool(arm["retention"]),
        "f_reacq": float(arm["f_reacq"]),
        "window_trials": int(arm["window_trials"]),
        "dissolve_stream": arm["dissolve_stream"],
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
    """Drives PolicyChunking with a synthetic action/outcome stream.

    One "trial" is: emit the target sequence, report its outcome, then emit a
    filler sequence at a LOWER outcome. The filler is what makes the MECH-323
    evaluative gate (mean > running baseline + margin) a real gate rather than a
    tautology -- with a single sequence the running baseline IS that sequence's
    own mean and the gate could never discriminate. Filler outcomes sit below
    baseline so filler can never itself form a chunk.
    """

    def __init__(self, cfg: PolicyChunkingConfig, seed: int) -> None:
        self.pc = PolicyChunking(cfg)
        self.rng = random.Random(seed)
        self.trials = 0

    def _emit(self, sequence: Tuple[int, ...], outcome: float) -> None:
        for action in sequence:
            self.pc.record_step(int(action))
        self.pc.note_outcome(float(outcome))

    def trial(self, target_outcome: float) -> None:
        """One target repetition plus one filler repetition."""
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

    # -- measurements -------------------------------------------------
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
    """P0 acquire -> P1 dissolve -> P2 reacquire, for one (arm, seed) cell."""
    cfg = _build_config(arm)
    drv = _Driver(cfg, seed)
    severe = arm["dissolve_stream"] == "SEVERE"
    hi, lo = DISSOLVE_SEVERE if severe else DISSOLVE_MILD

    def _tick(done: int) -> None:
        if done % 100 == 0:
            print(f"  [train] {progress_prefix} ep {done}/{TRIALS_PER_RUN}",
                  flush=True)

    # -- P0 ACQUISITION. r_acq is MEASURED, not assumed to be R_min. -------
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

    # -- P1 DISSOLUTION. Drive outcome variance above F_high, then let the
    #    T_dissolve timer run. Recovery back to CRYSTALLISED is possible by
    #    design, so this loop keeps driving until DISSOLVED or the budget ends.
    dissolved_at: Optional[int] = None
    if r_acq_crystallise is not None:
        for t in range(P1_MAX_TRIALS):
            drv.trial(hi if t % 2 == 0 else lo)
            _tick(drv.trials)
            if drv.target_state() is ChunkState.DISSOLVED:
                dissolved_at = t + 1
                break
    p1_trials = drv.trials - p0_trials

    # -- P2 REACQUISITION. Re-present the SAME consistent, above-baseline regime
    #    as P0 and count target repetitions until the chunk returns to FORMING.
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
        "f_reacq": float(arm["f_reacq"]),
        "window_trials": int(arm["window_trials"]),
        "dissolve_stream": arm["dissolve_stream"],
        # -- the DV and its denominators --
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
        # -- state + diagnostics --
        "final_state": (chunk.state.value if chunk is not None else None),
        "n_reacquisitions": (chunk.n_reacquisitions if chunk is not None else 0),
        "n_dissolutions": (chunk.n_dissolutions if chunk is not None else 0),
        "lib_state": lib,
        "trials_p0": p0_trials,
        "trials_p1": p1_trials,
        "trials_p2": p2_trials,
        # -- precondition measurements --
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
    """Spearman rho. Returns None when either side has no rank variation.

    A tie-free-rank assumption would be wrong here: the whole point of C2 is that
    the DV may be FLAT in f_reacq, and a flat y has zero rank variance, for which
    rho is undefined rather than 0. Reporting None and letting C2 fail on the
    undefined case is the honest reading -- a constant DV does not scale.
    """
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


def _analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the pre-registered criteria. No threshold is derived from rows."""
    default_arm = f"ARM_ON_F{int(round(F_REACQ_DEFAULT * 100)):03d}_W{WINDOW_DEFAULT}"
    mild_arm = f"{default_arm}_MILD"

    # -- C1: rapid reacquisition at the claim defaults ---------------------
    d_rows = _arm_rows(rows, default_arm)
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
    # Degenerate if the arm produced no DV at all -- then C1 did not discriminate,
    # it simply had nothing to read.
    c1_non_degenerate = bool(d_reacq) and med_r_acq is not None

    # -- C2: does the reacquisition count scale with f_reacq (at claim-default W)?
    sweep_x: List[float] = []
    sweep_y: List[float] = []
    sweep_detail: Dict[str, Any] = {}
    for f in F_REACQ_SWEEP:
        arm_id = f"ARM_ON_F{int(round(f * 100)):03d}_W{WINDOW_DEFAULT}"
        vals = _uncensored_r_reacq(_arm_rows(rows, arm_id))
        med = _median(vals)
        sweep_detail[arm_id] = {"f_reacq": f, "median_r_reacq": med,
                                "n_uncensored": len(vals),
                                "forced_bar": max(1, math.ceil(R_MIN * f))}
        if med is not None:
            sweep_x.append(f)
            sweep_y.append(med)
    rho_default = _spearman(sweep_x, sweep_y)
    c2_pass = rho_default is not None and rho_default >= SCALING_RHO_FLOOR
    # C2 discriminated only if at least 3 of the 4 sweep cells produced a DV.
    c2_non_degenerate = len(sweep_y) >= 3

    # -- C3: retention vs erasure. OFF must never revive; ON must sometimes. --
    off_rows = [r for r in rows if not r["retention"]]
    on_rows = [r for r in rows if r["retention"]]
    off_any_revived = any(r["n_reacquisitions"] > 0 for r in off_rows)
    on_any_revived = any(r["n_reacquisitions"] > 0 for r in on_rows)
    c3_pass = (not off_any_revived) and on_any_revived
    # Vacuous unless the OFF arms actually reached DISSOLVED -- "never revived" is
    # meaningless for a chunk that never dissolved.
    off_dissolved = [r for r in off_rows if r["dissolved_at"] is not None]
    c3_non_degenerate = len(off_dissolved) >= max(1, len(off_rows) // 2)

    # -- C4 (reported, NOT part of overall PASS): short-window discriminator ---
    sweep_x_s: List[float] = []
    sweep_y_s: List[float] = []
    sweep_detail_short: Dict[str, Any] = {}
    for f in F_REACQ_SWEEP:
        arm_id = f"ARM_ON_F{int(round(f * 100)):03d}_W{WINDOW_SHORT}"
        vals = _uncensored_r_reacq(_arm_rows(rows, arm_id))
        med = _median(vals)
        sweep_detail_short[arm_id] = {"f_reacq": f, "median_r_reacq": med,
                                      "n_uncensored": len(vals),
                                      "forced_bar": max(1, math.ceil(R_MIN * f))}
        if med is not None:
            sweep_x_s.append(f)
            sweep_y_s.append(med)
    rho_short = _spearman(sweep_x_s, sweep_y_s)
    c4_pass = rho_short is not None and rho_short >= SCALING_RHO_FLOOR
    c4_non_degenerate = len(sweep_y_s) >= 3

    # -- C5 (reported): mild-dissolution robustness ---------------------------
    mild_vals = _uncensored_r_reacq(_arm_rows(rows, mild_arm))
    med_mild = _median(mild_vals)
    if med_mild is not None and med_reacq_default:
        mild_rel_delta = abs(med_mild - med_reacq_default) / med_reacq_default
        c5_pass = mild_rel_delta <= MILD_TOLERANCE_FRAC
    else:
        mild_rel_delta = None
        c5_pass = False
    c5_non_degenerate = bool(mild_vals) and bool(d_reacq)

    # -- Is the DV a pure function of W rather than of f_reacq? ---------------
    # Reported as a diagnostic, not a criterion: it is the mechanism a FAIL on
    # C1/C2 would implicate, and recording it saves a successor the re-derivation.
    w_ratios = [r["r_reacq_over_window"] for r in rows
                if r["r_reacq_over_window"] is not None]
    # Do every sweep cell sit EXACTLY on its forced bar? If so C2 would have been
    # an arithmetic identity and must not be read as a measurement.
    on_uncensored = [r for r in on_rows if r["r_reacq"] is not None]
    all_on_forced_bar = bool(on_uncensored) and all(
        r["r_reacq_minus_forced_bar"] == 0 for r in on_uncensored)
    if all_on_forced_bar:
        c2_non_degenerate = False

    overall_pass = c1_pass and c2_pass and c3_pass

    metrics: Dict[str, Any] = {
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass,
        "c4_pass": c4_pass, "c5_pass": c5_pass,
        "median_r_acq_form": med_r_acq,
        "median_r_reacq_default_arm": med_reacq_default,
        "r_min": R_MIN,
        "forced_bar_default_arm": max(1, math.ceil(R_MIN * F_REACQ_DEFAULT)),
        "scaling_rho_default_window": rho_default,
        "scaling_rho_short_window": rho_short,
        "scaling_rho_floor": SCALING_RHO_FLOOR,
        "f_reacq_sweep_default_window": sweep_detail,
        "f_reacq_sweep_short_window": sweep_detail_short,
        "off_arm_any_revived": off_any_revived,
        "on_arm_any_revived": on_any_revived,
        "n_off_cells_dissolved": len(off_dissolved),
        "n_off_cells": len(off_rows),
        "median_r_reacq_mild_arm": med_mild,
        "mild_vs_severe_rel_delta": mild_rel_delta,
        "r_reacq_over_window_mean": (statistics.fmean(w_ratios) if w_ratios else None),
        "r_reacq_over_window_stdev": (statistics.pstdev(w_ratios)
                                      if len(w_ratios) > 1 else None),
        "all_on_cells_sit_on_forced_bar": all_on_forced_bar,
        "n_censored_cells": sum(1 for r in rows if r["r_reacq_censored"]),
        "n_cells": len(rows),
    }

    criteria = [
        {"name": "C1_reacquisition_faster_than_acquisition",
         "load_bearing": True, "passed": c1_pass},
        {"name": "C2_reacquisition_scales_with_f_reacq",
         "load_bearing": False, "passed": c2_pass},
        {"name": "C3_retention_discriminates_erasure",
         "load_bearing": False, "passed": c3_pass},
        {"name": "C4_short_window_scaling_discriminator",
         "load_bearing": False, "passed": c4_pass},
        {"name": "C5_mild_dissolution_robustness",
         "load_bearing": False, "passed": c5_pass},
    ]
    non_degenerate_map = {
        "C1": c1_non_degenerate, "C2": c2_non_degenerate, "C3": c3_non_degenerate,
        "C4": c4_non_degenerate, "C5": c5_non_degenerate,
    }
    return {
        "overall_pass": overall_pass,
        "metrics": metrics,
        "criteria": criteria,
        "criteria_non_degenerate": non_degenerate_map,
    }


def _criterion_owning_cells(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Which (arm, seed) cells each criterion actually reads.

    Cell ids match the arm_id passed to evaluate_arm_gate ("<arm_id>/seed<N>"), so
    a criterion can be gated on the readiness of exactly the cells it consumes.
    """
    default_arm = f"ARM_ON_F{int(round(F_REACQ_DEFAULT * 100)):03d}_W{WINDOW_DEFAULT}"
    sweep_default = {f"ARM_ON_F{int(round(f * 100)):03d}_W{WINDOW_DEFAULT}"
                     for f in F_REACQ_SWEEP}
    sweep_short = {f"ARM_ON_F{int(round(f * 100)):03d}_W{WINDOW_SHORT}"
                   for f in F_REACQ_SWEEP}
    mild_arm = f"{default_arm}_MILD"

    def _cells(pred) -> List[str]:
        return [f"{r['arm_id']}/seed{r['seed']}" for r in rows if pred(r)]

    return {
        # C1 also reads r_acq from every cell (its acquisition denominator).
        "C1": _cells(lambda r: r["arm_id"] == default_arm),
        "C2": _cells(lambda r: r["arm_id"] in sweep_default),
        "C3": _cells(lambda r: not r["retention"] or r["arm_id"] == default_arm),
        "C4": _cells(lambda r: r["arm_id"] in sweep_short),
        "C5": _cells(lambda r: r["arm_id"] in (mild_arm, default_arm)),
    }


def _label_for(analysis: Dict[str, Any]) -> str:
    m = analysis["metrics"]
    if not analysis["criteria_non_degenerate"]["C1"]:
        return "reacquisition_dv_unmeasured"
    if analysis["overall_pass"]:
        return "rapid_reacquisition_supported"
    if m["c3_pass"] and not m["c1_pass"] and not m["c2_pass"]:
        return "retention_real_but_rapid_reacquisition_falsified"
    if not m["c3_pass"]:
        return "retention_did_not_discriminate_erasure"
    return "rapid_reacquisition_partially_falsified"


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    arms = ARMS if not dry_run else ARMS
    rows: List[Dict[str, Any]] = []
    arm_gates: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []

    for arm in arms:
        for seed in seeds:
            # Boundary line: resets the runner's episodes_in_run counter.
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
                  f"bar={row['forced_bar']} state={row['final_state']}", flush=True)
            # One verdict per (arm, seed) cell -- seeds x conditions total.
            print(f"verdict: {'PASS' if rr is not None else 'FAIL'}", flush=True)

    analysis = _analyse(rows)
    per_arm = aggregate_arm_gates(arm_gates)

    # Fold the readiness gate into per-criterion non-degeneracy. Each criterion is
    # a CROSS-ARM aggregate, so it is not owned by a single arm the way
    # arm_criteria_non_degenerate assumes -- a criterion is non-degenerate only if
    # its own degeneracy test passed AND every cell it reads cleared its gate.
    # Done explicitly rather than by forcing the helper's one-arm-per-criterion
    # shape, which would silently keep only the last arm's verdict.
    green_cells = set(per_arm.get("green_arms", []))
    owning_cells = _criterion_owning_cells(rows)
    for name, cells in owning_cells.items():
        gate_ok = bool(cells) and all(c in green_cells for c in cells)
        analysis["criteria_non_degenerate"][name] = (
            analysis["criteria_non_degenerate"][name] and gate_ok)

    outcome = "PASS" if analysis["overall_pass"] else "FAIL"
    m = analysis["metrics"]

    # Per-claim direction. MECH-324's own falsifier text defines the FALSIFIED
    # condition, so a C1/C2 failure is a genuine `weakens`, not an `unknown`.
    if analysis["overall_pass"]:
        mech324_dir = "supports"
    elif not analysis["criteria_non_degenerate"]["C1"]:
        mech324_dir = "unknown"
    elif m["c3_pass"]:
        # Retention IS real (an erasure operator cannot produce C3) but the
        # quantitative rapid-reacquisition prediction is falsified.
        mech324_dir = "mixed"
    else:
        mech324_dir = "weakens"

    # MECH-323: P0 measures repetitions-to-formation against R_min under the joint
    # gate. Formation landing at R_min is that operator behaving as specified.
    med_acq = m["median_r_acq_form"]
    mech323_dir = ("supports" if med_acq is not None and med_acq <= float(R_MIN)
                   else ("mixed" if med_acq is not None else "unknown"))

    non_degenerate = bool(analysis["criteria_non_degenerate"]["C1"])
    degeneracy_reason = (
        "" if non_degenerate else
        "C1's arm produced no uncensored r_reacq, so the load-bearing DV was never "
        "measured; the run cannot speak to the rapid-reacquisition prediction."
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

    # Design-time refusal: a precondition no arm could satisfy from its
    # PRE-REGISTERED config must stop the run before compute is spent.
    arm_contexts = [
        {"arm_id": a["arm_id"], "retention": a["retention"],
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
        "dissolve_mild": list(DISSOLVE_MILD),
        "f_reacq_sweep": F_REACQ_SWEEP,
        "f_reacq_default": F_REACQ_DEFAULT,
        "window_default": WINDOW_DEFAULT,
        "window_short": WINDOW_SHORT,
        "scaling_rho_floor": SCALING_RHO_FLOOR,
        "mild_tolerance_frac": MILD_TOLERANCE_FRAC,
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
            "unit_scale_dissolution_unreachable": {
                "finding": (
                    "MECH-323 specifies outcome variance on a 0-1 normalised "
                    "scale. Max attainable POPULATION variance on a span-S range "
                    "is S^2/4, so on the unit scale it is 0.25 -- BELOW the "
                    "F_high = 0.45 dissolution gate. On the claim's own specified "
                    "scale, CRYSTALLISED -> DISSOLVING is arithmetically "
                    "unreachable, and the whole relapse-falsifier family (a)/(b)/"
                    "(c) has no entry point at the pre-registered defaults."
                ),
                "unit_scale_max_variance": 0.25,
                "variance_high": VARIANCE_HIGH,
                "this_run_outcome_span": OUTCOME_SPAN,
                "this_run_max_variance": OUTCOME_MAX_VARIANCE,
                "note": (
                    "Reported regardless of this run's outcome. F_high was NOT "
                    "altered; the harness outcome SCALE was pre-registered so the "
                    "claim's own thresholds are both reachable."
                ),
            },
            "structural_contamination_bound": {
                "note": (
                    "Reaching DISSOLVED requires T_dissolve trials of supra-F_high "
                    "variance, so at least dissolve_trials entries of the "
                    "window_trials-long variance window are contaminated at the "
                    "moment of dissolution -- 50 of 100 at the claim defaults. "
                    "This bound is structural, not a property of this driver's "
                    "dissolution stream; the MILD arm checks that empirically."
                ),
                "dissolve_trials": DISSOLVE_TRIALS,
                "window_default": WINDOW_DEFAULT,
                "min_contaminated_fraction_default_window": (
                    DISSOLVE_TRIALS / float(WINDOW_DEFAULT)),
            },
            "transfer_limit": (
                "Barnes 2005 is rodent T-maze procedural learning; Bouton 2012 is "
                "single-response operant contingencies. Both extinguish an object "
                "FAR SIMPLER than a multi-element composed chunked primitive. "
                "Whether a SEQUENCE extinguishes with the same asymmetry is "
                "EXTRAPOLATION, NOT EVIDENCE. Neither source quantifies a "
                "threshold, so f_reacq = 0.25 is an UNCALIBRATED ENGINEERING "
                "DEFAULT on the same footing as F_high = 0.45: this run tests the "
                "DIRECTION and SCALING of the prediction, never the value."
            ),
            "driver_rationale": (
                "Operator-level driver, not an agent run: V3-EXQ-810 measured the "
                "accumulator SILENT under agent control (c1_form_seed_frac 0.333, "
                "label chunk_accumulator_silent), so an agent-level reacquisition "
                "run would measure that readiness gap rather than MECH-324."
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
          f"median r_reacq(default arm)={m0(result,'median_r_reacq_default_arm')} "
          f"R_min={R_MIN} forced_bar={m0(result,'forced_bar_default_arm')}",
          flush=True)
    print(f"C1={result['metrics']['c1_pass']} C2={result['metrics']['c2_pass']} "
          f"C3={result['metrics']['c3_pass']} C4={result['metrics']['c4_pass']} "
          f"C5={result['metrics']['c5_pass']} "
          f"rho_W{WINDOW_DEFAULT}={m0(result,'scaling_rho_default_window')} "
          f"rho_W{WINDOW_SHORT}={m0(result,'scaling_rho_short_window')}", flush=True)
    print(f"r_reacq/W mean={m0(result,'r_reacq_over_window_mean')} "
          f"sd={m0(result,'r_reacq_over_window_stdev')} "
          f"censored={result['metrics']['n_censored_cells']}/"
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
