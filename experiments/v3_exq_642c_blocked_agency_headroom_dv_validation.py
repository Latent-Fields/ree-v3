#!/opt/local/bin/python3
"""
V3-EXQ-642c -- MECH-353 blocked-agency floor validation, HEADROOM DV.

Same-question successor to V3-EXQ-642b under a new letter, exactly as ratified
at the /failure-autopsy Step 8 gate on 2026-09-01 (user present) in
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-642b_2026-09-01.md
section 8, and re-confirmed by governance-20260903T2013.

Supersedes: V3-EXQ-642b (run
v3_exq_642b_blocked_agency_calibrated_floor_validation_20260831T131011Z_v3,
outcome FAIL, evidence_direction non_contributory).

WHAT IS UNCHANGED (deliberately -- the autopsy's own instruction): the
protocol, environment, seeds (42, 43, 44), arms (ARM_BLOCK / ARM_CONTROL),
budgets, the P0a/P0b warmups, the per-seed P0 readiness gate, C0, C3, and the
CALIBRATION_CONFIG under validation (the baseline-relative
outcome_mismatch floor built in ree-v3 d49db86f3e64670, inherited verbatim
from 642b). The sole causal change is the READOUT.

THE DEFECT BEING REPAIRED. 642b's C1 and C2 both read `z_block_peak`, a max
over the run against a hard clamp (`z_block_cap = 1.5`,
ree_core/affect/blocked_agency.py:153). Both arms touched 1.500 on all three
seeds, so `z_block_peak(BLOCK) - z_block_peak(CONTROL)` was exactly 0.000 and
C1/C2 returned false BY ARITHMETIC, not by measurement. The substrate build
under validation actually WORKED: on the mean, CONTROL fell from 1.26-1.35
(642a, legacy absolute floor) to 0.064-0.177 (642b, baseline-relative floor),
and BLOCK-minus-CONTROL separation improved 4.1x-9.3x.

WHY A NEW LETTER RATHER THAN ADJUDICATING 642b's ALREADY-RECORDED MEAN. The
mean is present in 642b's manifest for both arms on all three seeds, so the
answer looks free. It is not: 642b's pre-registered margins (C1_MARGIN 0.20,
Z_BLOCK_MIN 0.20, C2_MARGIN 0.20) were calibrated for the PEAK statistic, and
applying them to a different statistic post hoc is exactly the threshold
transplant the anti-fabrication rule forbids. The user chose a new letter at
the gate for that reason. This driver therefore RE-DERIVES the margins for the
mean statistic by a stated a-priori rule (below) and pre-registers them here,
before the run.

=====================================================================
THE HEADROOM DV, AND HOW ITS BARS WERE DERIVED
=====================================================================

DV: `z_block_mean` -- the mean of the z_block integrator over the scripted P1a
measurement steps. This is the first of the three headroom statistics the
autopsy names (mean / area-under-the-accumulation-curve / time-to-threshold).
Over a fixed step budget the normalised AUC IS the mean, so `z_block_mean`
and `z_block_auc_norm` are the same number; both are recorded, and
`z_block_time_to_threshold_frac` is recorded alongside as the third option, so
a successor need not re-run to compare them.

Unlike the peak, the mean is NOT annihilated by a transient excursion to the
cap: a single clamped step moves a 2400-step mean by at most 1.5/2400.

BAR DERIVATION RULE, stated a priori and applied uniformly:

    every z_block margin in this run = 20% of the integrator's own hard cap
    (Z_BLOCK_CAP = 1.5)  ->  0.30

so C1_MEAN_MARGIN = Z_BLOCK_MEAN_MIN = C2_MEAN_MARGIN = 0.30. The rule is
NOT read off 642b's observed separations. Note the direction it moves the bar:
642b's peak-calibrated 0.20 was 13.3% of the same cap, so 20% is STRICTER as a
fraction of the DV's own nominal range -- this re-derivation TIGHTENS the bar,
which is the opposite of the failure mode the anti-fabrication rule guards
against. (It is a range-fraction statement, not a power statement.)

IS THE 0.30 BAR NON-TRIVIAL? A NEGATIVE CONTROL FROM THE RECORD. A bar derived
a priori can still land somewhere useless -- so low that PASS is guaranteed, or
so high that FAIL is. It does not, and the two predecessor manifests settle it
without any threshold fitting. The SAME statistic, measured on the SAME
protocol, under the two floor regimes:

  legacy absolute floor      (642a): mean separations 0.1650 / 0.1214 / 0.0831
  baseline-relative floor    (642b): mean separations 0.6805 / 0.5904 / 0.7709

0.30 sits in the gap: ~1.8x above the highest legacy separation and ~2.0x below
the lowest calibrated one, so C1 rejects the un-calibrated regime on 3/3 seeds
and accepts the calibrated one on 3/3. The bar was NOT chosen to achieve that --
it is 20% of Z_BLOCK_CAP by the rule above, written before either number was
looked up -- but the record confirms the criterion discriminates the two
regimes rather than passing (or failing) whatever it is shown. That is what
642b's peak criterion could not do: its peak separation was 0.000 in BOTH
regimes.

This was VERIFIED BY REPLAY at authoring time, not argued: `_evaluate_seed` was
run over both predecessor manifests' recorded per-seed arm dicts.

  over 642a (legacy floor):      C0 T, C1 F, C2 F, C3 T  on 3/3 seeds -> FAIL
  over 642b (calibrated floor):  C0 T, C1 T, C2 T, C3 T  on 3/3 seeds -> PASS

So C0-C3 are JOINTLY satisfiable at values this substrate actually reaches (the
right-hand row), and the criteria reject the regime the calibration was built to
fix (the left-hand row). Neither row is evidence FOR the calibration -- 642b's
data cannot validate 642b -- but together they establish that the criterion is
neither vacuous nor unmeetable before any compute is spent.

FEASIBILITY IS GATED, NOT ASSUMED -- two `dv_headroom` preconditions
(experiments/_metrics.dv_headroom_check, ree-v3 8e133d26ed):

  (1) `c1_mean_elevation_headroom_prior` -- PRE-REGISTERED, analytic, known
      before this run: the achievable elevation above the highest CONTROL mean
      the PREDECESSOR measured is Z_BLOCK_CAP - 0.1768 = 1.3232 (642b seed 43,
      the worst of 0.1313 / 0.1768 / 0.0636). Required at margin 2.0:
      2 x 0.30 = 0.60. 1.3232 >= 0.60, so the bar sits well inside the range.
      Contrast the peak DV it replaces, whose achievable elevation was
      Z_BLOCK_CAP - 1.500 = 0.000 against a 0.20 bar.
  (2) `c1_mean_elevation_headroom_measured` -- the same quantity measured on
      THIS run's own CONTROL arm (statistic ceiling_headroom, dv_bounds
      (0, 1.5), control_values = the per-seed CONTROL z_block_mean). This one
      can only be evaluated after the arms run, so it is a post-hoc
      feasibility check rather than a P0 abort: if it is unmet the run
      self-routes `substrate_not_ready_requeue` instead of recording a
      falsification the DV had no room to express.

SATURATION FRACTION IS RECORDED PER ARM, as both the 642a and 642b autopsies
asked. `z_block_saturation_frac` = fraction of measured steps within
SATURATION_EPS of the cap. The residual finding the substrate_queue entry
`sd_blocked_agency_mismatch_floor_calibration` keeps at severity `corrupting`
is precisely that the CONTROL arm still touches 1.500 transiently while
sitting at a mean of 0.064-0.177; that entry stays `corrupting` and this run
does not close it. `z_block_peak` and the (expected-zero) peak separation are
also still recorded, as the standing witness that the peak readout is dead in
this regime.

C2 IS DEMOTED TO NON-DEGENERACY-FLAGGED, NOT SILENTLY KEPT. C2 is
`(z_block_sep - z_harm_a_sep) >= margin`, and this environment pins
`num_hazards=0` so z_harm_a is flat by construction -- 642b measured a
z_harm_a separation of exactly 0.000 on all three seeds. The subtraction is
therefore inert and C2 is C1's separation wearing a second name, which is the
"two criteria that are algebraically the same quantity" trap the V3-EXQ-981
autopsy records as its learning 5. C2 keeps its formula (moved to the mean, so
nothing is hidden) and keeps gating -- but when the measured z_harm_a
separation is within Z_HARM_A_INERT_EPS of zero on every ready seed, this run
sets `interpretation.criteria_non_degenerate["C2"] = false` with a recorded
reason, so no reader can count C2 as independent confirmation of C1.

C0 and C3 are unchanged. Both passed in 642b and neither is degenerate.

GOAL PINNING IS DECLARED, NOT LEFT TO BE MISREAD. The protocol pins z_goal at
GOAL_PIN = 0.5 through base._pin_goal(), a direct write that bypasses the
counted update_z_goal writer; 642b's manifest consequently reported
`z_goal_stream.writer_defect: true` on writer_calls = 0 over 14400 active
ticks, which the 642b autopsy section 6 corrected to a mislabelled deliberate
control. This driver stamps the z_goal_stream block itself with
`goal_pinned=True` (ree-v3 209c00fb883), so `writer_defect` reads null --
"not assessable, goal deliberately pinned" -- rather than a false positive
that would resurface this family in pending_review.md as a data-quality
concern it is not. Neither C0 nor C3 reads a goal-dependent statistic, so this
is a recording correction, not an adjudication change.

DV-SYMMETRY INVARIANCE (mandatory per-arm declaration). The DV is a MEAN of a
non-negative accumulator over a fixed step budget, so its symmetry group is
permutations of the measured steps. The manipulation -- ARM_BLOCK's scheduled
action blocking (`scheduled_action_block_enabled`, interval BLOCK_INTERVAL,
prob 1.0) versus ARM_CONTROL's free actions -- is NOT invariant under it: it
changes the VALUE of z_block on blocked steps (through the comparator's
outcome_mismatch), not the order in which steps are visited. It is also not a
broadcast additive constant (the V3-EXQ-604c class): a constant offset applied
to both arms cancels in the BLOCK-minus-CONTROL difference the criteria read,
and the manipulation is not a driver-supplied score offset at all. The same
statement holds for ARM_CONTROL, which differs only in that the blocking
schedule is off. Both arms run the identical measurement code path.

CLAIMLESS. `claim_ids: []` -- post-build substrate validation, as 642a/642b.
`evidence_direction: non_contributory` on every outcome; a FAIL must not
weaken a claim through metadata inherited from the base module. MECH-353
remains v3_pending until this passes; governance, not this run, clears it.

GOV-REUSE-1. The decisive readout (`z_block_mean` per arm per seed) IS already
recorded in 642b's manifest -- but only under 642b's own pre-registered
thresholds, which were calibrated for the peak. Reanalysis would therefore
have to invent the bar it scores against, which is the transplant hazard the
user ruled out at the Step 8 gate. The run is required, and it is cheap
(642b: elapsed_seconds 1465.1 for the full 3-seed x 2-arm grid).

=====================================================================
RED-TEAM (Step 4.5, 2026-09-04, claude-opus-5): CONTESTED -- 9 findings,
every one dispositioned below. No BLOCKING finding survived verification.
=====================================================================

F1 ATTRIBUTION GATE UNRECORDED -- ACCEPTED, FIXED. z_block only increments
  when motor_agency >= attribution_motor_floor (blocked_agency.py:322, floor
  0.5 at :150), and the effective mismatch floor is what the mismatch is
  compared against. 642a/642b recorded NEITHER, so their FAILs could not be
  attributed between "mismatch below floor", "attribution gate shut" and
  "goal gate" -- `blocked_agency.get_state()` exposes all of it and is called
  nowhere in the family. This run records per arm: motor_agency mean/min,
  motor_gate_shut_frac, the live attribution_motor_floor,
  effective_mismatch_floor_mean, n_external_block_ticks and the final
  baseline_mismatch_ema. Zero extra compute -- every value was already on the
  output object the measurement loop reads.

F2 "TOROIDAL ENV MAKES THE SEPARATION GEOMETRIC, NOT ATTRIBUTIONAL"
  -- DISMISSED ON THE RECORD. The claim: CONTROL always moves on a toroidal
  grid, BLOCK stalls, so a trained world_forward alone guarantees the
  separation. If that held, 642a -- SAME toroidal env, SAME P0a/P0b training,
  SAME arms -- would have shown it. It did not: 642a's mean separations were
  0.1650 / 0.1214 / 0.0831, all below this run's 0.30 bar, because the legacy
  absolute floor suppressed them. The separation therefore depends on the
  floor calibration under test, not on env geometry. Recorded as a real
  limit on scope, not a defect: this design cannot separate "external block"
  from "move cancelled by any means", and does not claim to.

F3 C1 IMPLIES C2, SO ONE GRID BRANCH IS UNREACHABLE -- ACCEPTED, DISCLOSED.
  With num_hazards=0 the z_harm_a separation is identically 0, so C2 reduces
  to `sep >= 0.30` and C1 is C2 plus a second conjunct: C1 => C2. The base
  grid's `elif not c2_pass -> z_block_tracks_z_harm_a_not_dissociated` branch
  (642a:666-667) is therefore UNREACHABLE in this run. Already flagged via
  criteria_non_degenerate["C2"]=false; the dead branch is now named too.

F4 THE MEASURED HEADROOM GATE CERTIFIES ROOM, NOT SENSITIVITY -- ACCEPTED,
  SCOPE CORRECTED. `c1_mean_elevation_headroom_measured` is unmet only if a
  CONTROL mean exceeds 0.90, so the diagnostically important failure -- both
  arms near zero, comparator dead -- yields the MAXIMUM headroom reading. The
  gate is monotone in the direction that makes the substrate look testable
  and cannot catch a dead comparator. It is not claimed to: C0 is evaluated
  FIRST in the base grid, and a dead comparator routes to
  z_block_integrator_comparator_not_read (642a:663-664), not to a C1
  falsification. The headroom preconditions guard the CLAMP-DEGENERACY that
  killed 642b, and nothing else. Stated here rather than left to be
  over-read as a general feasibility gate.

F5 CLAMP CENSORING OF THE MEAN -- QUANTITATIVELY DISMISSED, DIAGNOSTIC KEPT.
  The claim: BLOCK sits pinned at the 1.5 cap so the mean is ~1.5 x
  saturation_frac rather than a graded elevation. 642b's measured BLOCK means
  were 0.8119 / 0.7671 / 0.8345 -- about 55% of the cap, not near it -- so the
  arm is not clamp-censored in this regime. The reviewer's other half stands:
  z_block_saturation_frac was recorded but read by nothing, so it is now
  surfaced per arm per seed in headroom_dv_diagnostics for exactly this check.

F6 SEED FLOOR DENOMINATED ON THE REALIZED n -- ACCEPTED, FLAGGED. See the
  comment at the `seed_base_sufficient` block below.

F7 ARM ORDER NOT COUNTERBALANCED -- ACCEPTED, DISCLOSED, NOT FIXED. Both arms
  of a seed run sequentially on ONE agent, ARM_BLOCK first. For C1 this is
  clean: agent.reset() calls blocked_agency.reset() (agent.py:3661), which
  nulls _baseline_mismatch_ema per episode (blocked_agency.py:461, the
  explicit 642a repair), so no floor state carries across. For C3 it is a
  genuine confound -- residue and hippocampal state are NOT reset
  (agent.py:3434) and C3's rates come from the policy arm, so ARM_CONTROL's
  policy runs on an agent carrying ARM_BLOCK's history. NOT fixed here
  because the autopsy's ratified instruction is "keep C0 and C3 as they
  stand"; changing C3 would break the same-protocol comparison this letter
  exists to make. A C3 difference is order-confounded and must be read that
  way.

F8 VACUITY DETECTOR KEYED ON THE PINNED PEAK -- ACCEPTED, FIXED. The base
  keys C1/C2/C3 non-degeneracy on any_nonzero_block_peak, and the peak is
  pinned at 1.5 here -- so the detector built BECAUSE C3 once passed
  vacuously reported non-degenerate unconditionally. All three are now keyed
  on z_block_mean. The zero-margin strict inequality in C3's assert_sig is
  inherited and left alone, per the same instruction as F7.

F9 CROSS-ARM RNG DESYNC -- DISMISSED AT SOURCE. `_build_env(seed, block)`
  (642a:237) constructs a FRESH CausalGridWorldV2 with the SAME `seed=seed`
  for each arm, so the arms share no RNG stream to desynchronise.

The reviewer's summary finding -- that a FAIL labelled
z_block_integrator_no_rise would collapse several causes into one label -- is
what F1 fixes, and is the single most valuable thing this pass produced.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments import v3_exq_642a_blocked_agency_zblock_discriminative as base  # noqa: E402
from experiments._metrics import dv_headroom_check, p0_readiness_gate, P0NotReady  # noqa: E402
from experiments._lib.z_goal_stream import (  # noqa: E402
    MANIFEST_KEY as Z_GOAL_STREAM_KEY,
    z_goal_stream_stats,
)
from ree_core.agent import REEAgent  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_642c_blocked_agency_headroom_dv_validation"
SUPERSEDES = "V3-EXQ-642b"

# Inherited VERBATIM from V3-EXQ-642b -- this is the built lever still under
# validation, and it is not what changed in this letter.
CALIBRATION_CONFIG = {
    "blocked_agency_outcome_mismatch_floor_mode": "baseline_relative",
    "blocked_agency_outcome_mismatch_baseline_alpha": 0.02,
    "blocked_agency_outcome_mismatch_floor_ratio": 1.5,
    "blocked_agency_outcome_mismatch_baseline_min_floor": 0.02,
}

# ---- Pre-registered thresholds for the HEADROOM DV (see module docstring
# "BAR DERIVATION RULE"). Every margin is 20% of Z_BLOCK_CAP. ----
Z_BLOCK_CAP = 1.5           # ree_core/affect/blocked_agency.py:153 (default);
                            # asserted against the live config at run time.
MARGIN_FRACTION_OF_CAP = 0.20
# round() only strips the binary-float residue (1.5 * 0.20 is
# 0.30000000000000004); the value IS the rule's output, not a hand-set literal.
_MARGIN = round(Z_BLOCK_CAP * MARGIN_FRACTION_OF_CAP, 6)
assert _MARGIN == 0.30, _MARGIN
C1_MEAN_MARGIN = _MARGIN      # 0.30 absolute BLOCK-minus-CONTROL separation
Z_BLOCK_MEAN_MIN = _MARGIN    # 0.30 absolute BLOCK-arm floor
C2_MEAN_MARGIN = _MARGIN      # 0.30 dissociation margin
HEADROOM_MARGIN = 2.0       # the bar must sit at most half the achievable
                            # range away, not merely be touchable
SATURATION_EPS = 1e-3       # |z_block - cap| < this counts as a clamped step
Z_HARM_A_INERT_EPS = 1e-6   # below this, C2's subtraction is inert
Z_BLOCK_RISE_THRESHOLD = Z_BLOCK_MEAN_MIN   # for time-to-threshold recording

# The PREDECESSOR's measured CONTROL z_block_mean, per seed, read from
# v3_exq_642b_blocked_agency_calibrated_floor_validation_20260831T131011Z_v3.json.
# Used only for the analytic, pre-registered half of the headroom gate.
PREDECESSOR_CONTROL_Z_BLOCK_MEAN = {42: 0.1313, 43: 0.1768, 44: 0.0636}
PREDECESSOR_ACHIEVABLE_ELEVATION = Z_BLOCK_CAP - max(
    PREDECESSOR_CONTROL_Z_BLOCK_MEAN.values()
)   # 1.3232

base.__doc__ = __doc__
base.__file__ = __file__
base.EXPERIMENT_TYPE = EXPERIMENT_TYPE
base.SUPERSEDES = SUPERSEDES
base.CFG_KWARGS = {**base.CFG_KWARGS, **CALIBRATION_CONFIG}

_base_build_manifest = base._build_manifest


def _scripted_measure(agent: REEAgent, seed: int, block: bool, episodes: int,
                      steps: int) -> Dict:
    """P1a, re-implemented from 642a's version with the headroom statistics added.

    Byte-for-byte the same measurement loop and the same returned keys as
    base._scripted_measure; the ONLY additions are z_block_saturation_frac,
    z_block_auc_norm, z_block_time_to_threshold_frac and n_z_block_steps. It is
    re-implemented rather than wrapped because the per-step z_block series is a
    local of the base function and is not returned.
    """
    env = base._build_env(seed, block=block)
    _, od = env.reset()
    agent.reset()
    base._pin_goal(agent)
    agent.sense(od["body_state"], od["world_state"])  # tick0, seeds caches
    rng = np.random.RandomState(seed + 1000)
    z_block_series: List[float] = []
    z_harm_a_series: List[float] = []
    mism_blocked: List[float] = []
    mism_free: List[float] = []
    # RED-TEAM F1 (2026-09-04): the ATTRIBUTION gate
    # (blocked_agency.py:322, motor_agency >= attribution_motor_floor) and the
    # effective floor are what decide whether a mismatch becomes z_block at
    # all. Neither was recorded by 642a/642b, so their FAILs could not be
    # attributed between "mismatch below floor", "attribution gate shut" and
    # "goal gate". Every one of these numbers is already on the output object
    # this loop reads and was being discarded -- collecting them is zero extra
    # compute. blocked_agency.get_state() exposes the same fields and is
    # called nowhere in this family.
    motor_series: List[float] = []
    floor_series: List[float] = []
    n_external_block_ticks = 0
    for _ep in range(episodes):
        for _ in range(steps):
            idx = int(rng.randint(0, 4))
            aoh = base._one_hot(idx)
            agent._last_action = aoh.clone()
            _, h, d, inf, od = env.step(aoh)
            lat = agent.sense(od["body_state"], od["world_state"])
            o = agent.blocked_agency.last_output()
            z_block_series.append(float(o.z_block))
            motor_series.append(float(o.motor_agency))
            floor_series.append(float(o.effective_outcome_mismatch_floor))
            if bool(o.external_block_this_tick):
                n_external_block_ticks += 1
            if lat.z_harm_a is not None:
                z_harm_a_series.append(float(lat.z_harm_a.detach().norm().item()))
            else:
                z_harm_a_series.append(0.0)
            if inf.get("action_blocked_this_step", False):
                mism_blocked.append(float(o.outcome_mismatch))
            else:
                mism_free.append(float(o.outcome_mismatch))
            if d:
                _, od = env.reset()
                agent.reset()
                base._pin_goal(agent)
                agent.sense(od["body_state"], od["world_state"])

    n = len(z_block_series)
    cap = float(getattr(agent.blocked_agency.config, "z_block_cap", Z_BLOCK_CAP))
    n_sat = sum(1 for v in z_block_series if v >= cap - SATURATION_EPS)
    ttt = -1.0
    for i, v in enumerate(z_block_series):
        if v >= Z_BLOCK_RISE_THRESHOLD:
            ttt = i / float(n) if n else -1.0
            break
    mean = float(np.mean(z_block_series)) if z_block_series else 0.0
    motor_floor = float(getattr(agent.blocked_agency.config,
                                "attribution_motor_floor", 0.5))
    ba_state = agent.blocked_agency.get_state()
    return {
        "z_block_peak": max(z_block_series) if z_block_series else 0.0,
        "z_block_mean": mean,
        # Over a fixed step budget the normalised area under the accumulation
        # curve IS the mean; recorded under both names so a successor comparing
        # the autopsy's three candidate headroom statistics need not re-run.
        "z_block_auc_norm": mean,
        "z_block_saturation_frac": (n_sat / n) if n else 0.0,
        "z_block_cap_observed": cap,
        "z_block_time_to_threshold_frac": ttt,
        "n_z_block_steps": n,
        "z_harm_a_mean": float(np.mean(z_harm_a_series)) if z_harm_a_series else 0.0,
        "blocked_step_mismatch_mean": float(np.mean(mism_blocked)) if mism_blocked else 0.0,
        "free_step_mismatch_mean": float(np.mean(mism_free)) if mism_free else 0.0,
        "n_blocked_steps": len(mism_blocked),
        "n_free_steps": len(mism_free),
        # Attribution-gate diagnostics (RED-TEAM F1). motor_floor is read off
        # the live config rather than re-typed, so it cannot drift.
        "motor_agency_mean": float(np.mean(motor_series)) if motor_series else 0.0,
        "motor_agency_min": min(motor_series) if motor_series else 0.0,
        "motor_gate_shut_frac": (
            sum(1 for v in motor_series if v < motor_floor) / len(motor_series)
        ) if motor_series else 0.0,
        "attribution_motor_floor": motor_floor,
        "effective_mismatch_floor_mean": float(np.mean(floor_series)) if floor_series else 0.0,
        "n_external_block_ticks": n_external_block_ticks,
        "baseline_mismatch_ema_final": ba_state.get("baseline_mismatch_ema"),
        "outcome_mismatch_floor_mode": ba_state.get("outcome_mismatch_floor_mode"),
        "n_external_blocks_counter": ba_state.get("n_external_blocks"),
    }


def _evaluate_seed(block: Dict, control: Dict) -> Dict:
    """C0 and C3 verbatim from 642a. C1 and C2 moved onto the headroom DV."""
    c0 = (block["blocked_step_mismatch_mean"]
          - block["free_step_mismatch_mean"]) >= base.C0_MARGIN

    z_block_sep = block["z_block_mean"] - control["z_block_mean"]
    z_harm_a_sep = block["z_harm_a_mean"] - control["z_harm_a_mean"]
    c1 = (z_block_sep >= C1_MEAN_MARGIN) and (block["z_block_mean"] >= Z_BLOCK_MEAN_MIN)
    c2 = (z_block_sep - z_harm_a_sep) >= C2_MEAN_MARGIN

    no_withdraw = block["action_rate"] >= (control["action_rate"] - base.C3_EPS)
    assert_sig = (
        block["action_rate"] > control["action_rate"]
        or block["alt_switch_rate"] > control["alt_switch_rate"]
    )
    no_suffering = (block["z_harm_a_mean_policy"]
                    <= (control["z_harm_a_mean_policy"] + base.C3_EPS))
    c3 = no_withdraw and assert_sig and no_suffering

    return {
        "C0_detector_readiness": bool(c0),
        "C1_z_block_rises": bool(c1),
        "C2_dissociation_from_z_harm_a": bool(c2),
        "C3_assert_not_withdraw": bool(c3),
        # The headroom DV the criteria now read.
        "z_block_mean_separation": z_block_sep,
        "z_harm_a_separation": z_harm_a_sep,
        # The dead peak readout, kept as the standing witness that it is dead.
        "z_block_peak_separation": block["z_block_peak"] - control["z_block_peak"],
        "z_block_separation": z_block_sep,   # base key name, now on the mean
    }


def _headroom_preconditions(result: Dict) -> List[Dict]:
    """The two dv_headroom feasibility checks (see module docstring)."""
    prior = dv_headroom_check(
        "c1_mean_elevation_headroom_prior",
        dv_name="z_block_mean",
        criterion_threshold=C1_MEAN_MARGIN,
        achievable=PREDECESSOR_ACHIEVABLE_ELEVATION,
        margin=HEADROOM_MARGIN,
        control=(
            "PRE-REGISTERED, analytic: Z_BLOCK_CAP minus the highest CONTROL "
            "z_block_mean V3-EXQ-642b measured (0.1768, seed 43) = 1.3232. This "
            "is the room the successor's absolute elevation bar has to work in, "
            "computed from the predecessor's own recorded data before this run. "
            "The peak DV it replaces had Z_BLOCK_CAP - 1.500 = 0.000 of the same "
            "room against a 0.20 bar, on all three seeds."
        ),
    )
    control_means = [
        s["ARM_CONTROL"]["z_block_mean"]
        for s in result.get("per_seed", [])
        if s.get("ARM_CONTROL") and s.get("ARM_BLOCK")
    ]
    checks = [prior]
    if control_means:
        checks.append(dv_headroom_check(
            "c1_mean_elevation_headroom_measured",
            dv_name="z_block_mean",
            criterion_threshold=C1_MEAN_MARGIN,
            control_values=control_means,
            statistic="ceiling_headroom",
            dv_bounds=(0.0, Z_BLOCK_CAP),
            margin=HEADROOM_MARGIN,
            control=(
                "The same quantity measured on THIS run's own ARM_CONTROL arm. "
                "Evaluated after the arms run (the DV is the run's output), so "
                "it is a feasibility check rather than a P0 abort: unmet routes "
                "the run to substrate_not_ready_requeue rather than recording a "
                "falsification the DV had no room to express."
            ),
        ))
    try:
        return p0_readiness_gate(checks)
    except P0NotReady as e:
        return e.preconditions


def _build_manifest(result: Dict, timestamp_utc: str, dry_run: bool) -> Dict:
    manifest = _base_build_manifest(result, timestamp_utc, dry_run)

    # Claimless post-build validation: outcome routes governance readiness only.
    manifest["evidence_direction"] = "non_contributory"
    manifest["evidence_direction_note"] = (
        "Claimless post-build substrate validation; outcome routes governance "
        "readiness only and does not update claim confidence."
    )
    manifest["calibration_under_test"] = dict(CALIBRATION_CONFIG)

    # GOAL_PINNED OPT-IN (ree-v3 209c00fb883, 2026-09-01). This protocol pins
    # z_goal at a fixed magnitude via base._pin_goal(), a direct
    # agent.goal_state._z_goal write that bypasses the counted update_z_goal
    # writer. V3-EXQ-642b's manifest therefore reported
    # z_goal_stream.writer_defect: true against writer_calls = 0 over 14400
    # active ticks -- which the 642b autopsy section 6 corrected to a
    # MISLABELLED DELIBERATE CONTROL, and which is exactly what the goal_pinned
    # flag was subsequently built to express. Stamped here, where the run's
    # agents are still on `result` (base.main() pops them only AFTER
    # _build_manifest returns), so the block carries goal_pinned: true and
    # writer_defect: null -- "not assessable", not a clean bill of health.
    # stamp_recording_core's _fill posture leaves an explicit author value
    # alone, so this is what reaches the manifest.
    _zg = z_goal_stream_stats(result.get("agents_for_z_goal") or [],
                              goal_pinned=True)
    if _zg:
        manifest[Z_GOAL_STREAM_KEY] = _zg

    manifest["pre_registered_thresholds"] = {
        "MARGIN_FRACTION_OF_CAP": MARGIN_FRACTION_OF_CAP,
        "Z_BLOCK_CAP": Z_BLOCK_CAP,
        "C0_MARGIN": base.C0_MARGIN,
        "C1_MEAN_MARGIN": C1_MEAN_MARGIN,
        "Z_BLOCK_MEAN_MIN": Z_BLOCK_MEAN_MIN,
        "C2_MEAN_MARGIN": C2_MEAN_MARGIN,
        "HEADROOM_MARGIN": HEADROOM_MARGIN,
        "C3_EPS": base.C3_EPS,
        "SEED_PASS_FRACTION": base.SEED_PASS_FRACTION,
        "SATURATION_EPS": SATURATION_EPS,
        "derivation_rule": (
            "every z_block margin = 20% of Z_BLOCK_CAP (1.5) = 0.30, stated "
            "a priori and NOT read off V3-EXQ-642b's observed separations. The "
            "peak-calibrated 0.20 it replaces was 13.3% of the same cap, so "
            "this re-derivation TIGHTENS the bar as a fraction of the DV's own "
            "nominal range."
        ),
    }

    interp = manifest.get("interpretation") or {}
    preconditions = list(interp.get("preconditions") or [])
    headroom = _headroom_preconditions(result)
    preconditions.extend(headroom)
    interp["preconditions"] = preconditions

    headroom_unmet = [p["name"] for p in headroom if not p.get("met")]
    if headroom_unmet and manifest.get("outcome") != "PASS":
        # A criterion the DV had no room to satisfy must not be recorded as a
        # falsification. This is the whole point of the dv_headroom class.
        interp["label"] = "substrate_not_ready_requeue"
        interp["headroom_unmet"] = headroom_unmet
    elif headroom_unmet:
        interp["headroom_unmet"] = headroom_unmet

    ready = [s for s in result.get("per_seed", [])
             if s.get("ARM_BLOCK") and s.get("ARM_CONTROL")]
    z_harm_a_inert = bool(ready) and all(
        abs(s["criteria"]["z_harm_a_separation"]) <= Z_HARM_A_INERT_EPS
        for s in ready if s.get("criteria")
    )
    cnd = dict(interp.get("criteria_non_degenerate") or {})

    # The base module self-reports C1/C2/C3 non-degeneracy off z_block_PEAK,
    # which in this regime is pinned at the 1.5 cap and therefore reports
    # "non-degenerate" for a criterion that no longer reads it. Re-key C1 onto
    # the statistic it now consumes -- the 642b autopsy's learning 3 ("a
    # validation run's success condition should be stated on the statistic its
    # criterion consumes") applied to the self-report as well as the criterion.
    # RED-TEAM F8: the base keys C1/C2/C3 non-degeneracy on
    # any_nonzero_block_peak, and the peak is pinned at the 1.5 cap in this
    # regime -- so the vacuity detector that exists BECAUSE C3 once passed
    # vacuously reports "non-degenerate" unconditionally. Re-key all three
    # onto the statistic actually in play.
    if ready:
        _mean_live = any(s_["ARM_BLOCK"]["z_block_mean"] > 1e-6 for s_ in ready)
        cnd["C1"] = _mean_live
        cnd["C2"] = _mean_live
        cnd["C3"] = _mean_live

    if z_harm_a_inert:
        cnd["C2"] = False
        interp["c2_degeneracy_reason"] = (
            "ENV_KWARGS pins num_hazards=0, so z_harm_a is flat by construction "
            "and the measured z_harm_a separation is 0 on every ready seed. C2's "
            "subtraction is therefore inert and C2 is C1's separation under a "
            "second name -- it is NOT independent confirmation of C1. Recorded "
            "rather than silently kept (V3-EXQ-981 autopsy, learning 5)."
        )
    interp["criteria_non_degenerate"] = cnd

    # RED-TEAM F6: base.run_experiment computes need = ceil(2/3 * n_READY),
    # denominated on the REALIZED ready-seed count, not the intended 3. With
    # two seeds refused by the P0 gate, n_ready = 1 and need = 1 -- one seed
    # would carry a validated_clear_v3_pending label while combination_rule
    # still reads ">= 2/3 readiness-cleared seeds". Recorded, not silently
    # relabelled: the outcome stays what the criteria said, but a reader is
    # told the seed base was too thin to call it a validation.
    n_ready = len(ready)
    interp["n_ready_seeds_effective"] = n_ready
    interp["seed_base_sufficient"] = bool(n_ready >= 2)
    if n_ready < 2:
        interp["seed_base_note"] = (
            f"Only {n_ready} readiness-cleared seed(s). base.run_experiment "
            "denominates the 2/3 seed floor on the REALIZED ready count, so "
            "the pass fraction was met against a seed base too thin to "
            "validate anything; treat this run as substrate_not_ready_requeue "
            "regardless of the criteria verdict."
        )

    manifest["interpretation"] = interp

    # Make C2's dependence visible in the criteria list too.
    for c in manifest.get("criteria") or []:
        if c.get("name") == "C2_dissociation_from_z_harm_a" and z_harm_a_inert:
            c["note"] = (
                "gating, but NOT independent: z_harm_a is structurally flat in "
                "this env (num_hazards=0), so C2 reduces to C1's separation. "
                "criteria_non_degenerate.C2 is false."
            )
        if c.get("name") == "C1_z_block_rises":
            c["note"] = (
                "moved onto the headroom DV z_block_mean; bar "
                f"{C1_MEAN_MARGIN} absolute separation AND BLOCK mean >= "
                f"{Z_BLOCK_MEAN_MIN}, both 20% of Z_BLOCK_CAP"
            )

    manifest["combination_rule"] = (
        "overall_pass = C0 AND C1 AND C2 AND C3, each on >= 2/3 "
        "readiness-cleared seeds (SEED_PASS_FRACTION). C1 and C2 now read the "
        "HEADROOM DV z_block_mean instead of the clamp-degenerate z_block_peak. "
        "Feasibility of C1's bar is gated by two dv_headroom preconditions (one "
        "pre-registered from V3-EXQ-642b's measured CONTROL means, one measured "
        "on this run's own CONTROL arm); an unmet headroom precondition on a "
        "non-PASS run relabels the interpretation substrate_not_ready_requeue "
        "rather than letting a starved criterion read as a falsification. C2 is "
        "flagged non-degenerate:false whenever the z_harm_a separation it "
        "subtracts is structurally zero."
    )

    # Saturation fraction and the dead peak readout, per arm per seed.
    manifest["headroom_dv_diagnostics"] = {
        "dv_name": "z_block_mean",
        "z_block_cap": Z_BLOCK_CAP,
        "per_seed": [
            {
                "seed": s.get("seed"),
                "block_z_block_mean": s["ARM_BLOCK"]["z_block_mean"],
                "control_z_block_mean": s["ARM_CONTROL"]["z_block_mean"],
                "mean_separation": (s["ARM_BLOCK"]["z_block_mean"]
                                    - s["ARM_CONTROL"]["z_block_mean"]),
                "block_z_block_peak": s["ARM_BLOCK"]["z_block_peak"],
                "control_z_block_peak": s["ARM_CONTROL"]["z_block_peak"],
                "peak_separation": (s["ARM_BLOCK"]["z_block_peak"]
                                    - s["ARM_CONTROL"]["z_block_peak"]),
                "block_saturation_frac": s["ARM_BLOCK"].get("z_block_saturation_frac"),
                "control_saturation_frac": s["ARM_CONTROL"].get("z_block_saturation_frac"),
                "block_time_to_threshold_frac": s["ARM_BLOCK"].get("z_block_time_to_threshold_frac"),
                "control_time_to_threshold_frac": s["ARM_CONTROL"].get("z_block_time_to_threshold_frac"),
                # RED-TEAM F1: what the attribution gate and the effective
                # floor were actually doing, per arm. Without these a FAIL
                # labelled z_block_integrator_no_rise collapses at least three
                # distinct causes into one label.
                "block_motor_agency_mean": s["ARM_BLOCK"].get("motor_agency_mean"),
                "control_motor_agency_mean": s["ARM_CONTROL"].get("motor_agency_mean"),
                "block_motor_gate_shut_frac": s["ARM_BLOCK"].get("motor_gate_shut_frac"),
                "control_motor_gate_shut_frac": s["ARM_CONTROL"].get("motor_gate_shut_frac"),
                "attribution_motor_floor": s["ARM_BLOCK"].get("attribution_motor_floor"),
                "block_effective_mismatch_floor_mean": s["ARM_BLOCK"].get("effective_mismatch_floor_mean"),
                "control_effective_mismatch_floor_mean": s["ARM_CONTROL"].get("effective_mismatch_floor_mean"),
                "block_n_external_block_ticks": s["ARM_BLOCK"].get("n_external_block_ticks"),
                "control_n_external_block_ticks": s["ARM_CONTROL"].get("n_external_block_ticks"),
                "block_baseline_mismatch_ema_final": s["ARM_BLOCK"].get("baseline_mismatch_ema_final"),
                "control_baseline_mismatch_ema_final": s["ARM_CONTROL"].get("baseline_mismatch_ema_final"),
            }
            for s in ready
        ],
        "note": (
            "The residual the sd_blocked_agency_mismatch_floor_calibration entry "
            "keeps at severity corrupting is that the CONTROL arm still touches "
            "the 1.5 cap transiently while sitting at a mean of 0.064-0.177. "
            "peak_separation is expected to be ~0.0 in both arms; it is recorded "
            "as the standing witness that a peak-shaped readout is dead in this "
            "regime, not as a criterion. This run does NOT close that entry."
        ),
    }

    manifest["predecessor_result"] = {
        "queue_id": "V3-EXQ-642b",
        "run_id": ("v3_exq_642b_blocked_agency_calibrated_floor_validation_"
                   "20260831T131011Z_v3"),
        "outcome": "FAIL",
        "evidence_direction": "non_contributory",
        "failure_signature": (
            "C1/C2 read z_block_peak, a max against a hard clamp; both arms "
            "reached 1.500 on all 3 seeds so the peak separation was exactly "
            "0.000 and both criteria were false by arithmetic. The same runs' "
            "z_block_mean separations were 0.681 / 0.590 / 0.771."
        ),
        "autopsy": "failure_autopsy_V3-EXQ-642b_2026-09-01.md (confirmed)",
    }
    manifest["notes"] = (
        "Post-build validation of the baseline-relative blocked-agency "
        "outcome-mismatch floor (ree-v3 d49db86f3e64670), re-run under a new "
        "letter with the load-bearing readout moved from the clamp-degenerate "
        "z_block_peak to the headroom DV z_block_mean, and the margins "
        "re-derived for that statistic by an a-priori rule (20% of "
        "Z_BLOCK_CAP) rather than transplanted from the peak calibration. "
        "Protocol, env, seeds, arms, budgets, C0 and C3 are unchanged from "
        "V3-EXQ-642a/642b. Saturation fraction is recorded per arm. PASS "
        "requires C0-C3 on >= 2/3 readiness-cleared seeds and permits "
        "governance to consider clearing MECH-353 v3_pending; this run does "
        "not clear it itself."
    )
    ci = dict(manifest.get("custom_information") or {})
    ci["red_team_disposition_note"] = (
        "See the V3-EXQ-642c queue entry note for the Step 4.5 verdict and the "
        "model that produced it."
    )
    ci["gov_reuse_1_note"] = (
        "GOV-REUSE-1: the decisive readout (per-arm z_block_mean) IS recorded in "
        "V3-EXQ-642b's manifest, but only against thresholds calibrated for the "
        "PEAK statistic. Scoring it by reanalysis would require inventing the "
        "bar, which is the threshold transplant the user ruled out at the "
        "2026-09-01 Step 8 gate. Run required; cost ~1465 s for the full grid."
    )
    manifest["custom_information"] = ci
    return manifest


base._scripted_measure = _scripted_measure
base._evaluate_seed = _evaluate_seed
base._build_manifest = _build_manifest


def main():
    return base.main()


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    _dry = "--dry-run" in sys.argv
    if _outcome is not None:
        emit_outcome(
            outcome=_outcome,
            manifest_path=_manifest_path,
            dry_run=_dry,
        )
    raise SystemExit(0)
