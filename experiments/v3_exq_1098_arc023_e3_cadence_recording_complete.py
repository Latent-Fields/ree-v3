#!/opt/local/bin/python3
"""V3-EXQ-1098 -- ARC-023 recording-complete E3 heartbeat-cadence measurement.

SLEEP DRIVER: not applicable (no sleep loop used in this driver).

RED-TEAM (Step 4.5, model fable, ONE pass, not iterated): BLOCKING. Six findings; all
six verified against source before acting, five fixed or already-recorded, and the
sixth is a defect in the CLAIM'S OWN pre-registered criterion that this driver has no
authority to change. Dispositions in RED_TEAM_DISPOSITIONS below.

  NOT QUEUED AS A RESULT. See "WHY THIS IS NOT QUEUED" at the end of this docstring.

WHAT THIS RUNS, AND WHY IT IS NOT A DISCRIMINATIVE PAIR
------------------------------------------------------
ARC-023 ("three BG-like loops operate at characteristic thalamic heartbeat rates")
had its falsifier TIGHTENED on 2026-09-24 under GFLAG-0443 option A. The tightened
form is a SINGLE-CONDITION measurement of production behaviour under ecological
load, not a contrast: it asks whether the realized E3 cadence tracks the
arousal-set period and stays below E2's configured share. There are therefore no
arms, and nothing is ablated -- an ablation would measure MECH-091, a different
claim. Everything this driver adds to a production run is RECORD-ONLY
(`_lib/baselines/arc023_e3_cadence.CadenceRecorder`: every wrapper records then
calls straight through, so behaviour is bit-identical to an uninstrumented run).

EXP-0548's gating_reason names exactly one deficiency, and it is a RECORDING one:
"design needs the REQUIRED RECORDING fields, which no current driver emits --
/queue-experiment owns the design." This driver is that design.

  V3-EXQ-942 IS NOT ADMISSIBLE and is not reused -- the claim says so in terms
  ("the falsifier was written with its numbers in view"). Its manifest
  (v3_exq_942_..._20260820T073245Z_v3.json) is tagged claim_ids ['INV-013'] and
  contains none of current_e3_steps / z_beta / phase_reset / trigger /
  reset_driven / e3_share. GOV-REUSE-1 check: reanalysis_query.py over 1077
  manifests found 0 carrying reset_driven_share / e3_share_realized /
  current_e3_steps / trigger_counts / clock_driven_e3_updates (instrument canary
  passed: readout "e3" unfiltered DOES return hits). Not recoverable -> run.

  EXQ-131 IS ALSO INADMISSIBLE, by the claim's own exclusion: its 2026-03-30
  diagnosis established that synchronous/time-multiplexed polling produces an "E3
  output freeze artifact", so var_harm_eval-style discriminative-pair metrics must
  NOT be used here. This driver reads no E3 OUTPUT at all -- only clock cadence
  counters and phase_reset call-site counts, which are not subject to that
  artifact (they are not readouts of E3 state between ticks).

THE PRE-REGISTERED CRITERIA ARE THE CLAIM'S, VERBATIM
-----------------------------------------------------
Copied from ARC-023's `what_would_answer`, not re-derived; the constants below are
the claim's numbers and are never computed from this run's own statistics.

  CONFIRMING, on >= 2 of 3 seeds, ALL of:
    (i)   clock-driven E3 updates track sum_t 1/_current_e3_steps(t) within
          +/-15% relative                                        -> C1
    (ii)  realized E3 update share (clock-driven plus reset-driven) is below E2's
          configured share 1/e2_steps_per_tick (1/3 at defaults) by >= 0.08
          absolute                                               -> C2
    (iii) reset-driven share is <= 0.10                           -> C3
  FALSIFYING, on >= 2 of 3 seeds:
    realized E3 update share from clock-driven plus ONSET-GATED reset triggers
    is >= E2's configured share. Reset ticks from a trigger with NO onset gate
    (the MECH-091 harm trigger) are reported separately; an excess attributable
    only to them is PARTIAL and routes to MECH-091, not a falsification.
  PARTIAL:
    (i) fails with (ii) holding, or reset-driven share > 0.10.

FOUR THINGS PRE-REGISTERED HERE THAT THE CLAIM DOES NOT SAY -- stated, not papered over
--------------------------------------------------------------------------------------
(a) OPERATING POINT. `beta_gate_bistable=True` is RATIFIED by decision chip
    chip-20260925-arc023-trigger-reachability-config option (2)
    (orchestrate-20260924-1707 under the user's standing delegation
    rec-20260924-fb429c72). The claim's tolerances (+/-15%, >=0.08, <=0.10) were
    calibrated with V3-EXQ-942's numbers in view, and 942 ran at
    `beta_gate_bistable=False`. So THE TOLERANCES ARE BEING APPLIED AT A DIFFERENT
    OPERATING POINT THAN THE ONE THEY WERE DERIVED FROM. The 942-config reference
    values from this lineage's 2026-09-25 probe are recorded in the manifest as a
    NON-GATING contrast under diagnostics.probe_reference_942_config
    (reset_driven_share 0.2625, e3_share_realized 0.3600, clock_tracking_ratio
    0.7047, vs 0.1175 / 0.2300 / 0.8580 at the ratified config).

(b) THE SPLIT IS 2-OF-4 LIVE, NOT 4-OF-4. `completion` and `ncl_reassert` cannot
    fire at default config -- for ARITHMETIC reasons, not "not trained":
      completion: compute_completion_signal = sigmoid(-best_score*0.5) has
        achievable range (0, 0.5] on non-negative residue, but release needs
        >= completion_release_threshold (default 0.75). Unreachable AS AN IDENTITY
        regardless of training and regardless of beta_gate_bistable. Registered as
        substrate_queue residue-completion-signal-threshold-unreachable; the
        inverted docstring was corrected under GFLAG-0344.
      ncl_reassert: needs `_ncl_hold_active`; measured 0 in the 2026-09-25 probe
        EVEN WITH use_natural_commit_latch_hold ON, matching config.py's own note
        that the latch-hold "NEVER armed (ncl_hold_reassert_total=0)".
    This CORRECTS the ratifying decision's stated reason, which held that option
    (2) makes completion reachable. It does not change the decision's outcome --
    its other two reasons stand (match the canonical MECH-091 lineage and
    V3-EXQ-944/944a/944b; route commit-entry through the CURRENT
    readiness-admission site rather than the legacy elevate site).
    Both classes are recorded with a THREE-VALUED reachability status so a zero can
    never be read as "measured zero" (CLAUDE.md "Negative instruments"), and the
    observed completion-signal MAX is recorded against its threshold so the
    unreachability claim is re-derivable from the manifest rather than trusted.

(c) LEG (i)'s OWN NON-DEGENERACY. If `_current_e3_steps` takes ONE value for a
    whole run, leg (i) reduces to the near-identity "a clock with period K fires
    N/K times" and a PASS on it is vacuous. substrate_queue
    mech005-endogenous-arousal-dynamic-range asserts exactly that pinning on a
    FULLY TRAINED agent (endogenous ||z_beta|| spread 0.91% of its mean), while
    this lineage's untrained probe measured 7 distinct periods and 72.2% spread.
    Which holds at THIS run's trained operating point is unknown in advance, so it
    is MEASURED: `arousal_period_varies` is a recorded precondition, and C1 is
    marked non-degenerate FALSE (and the run `non_degenerate: false`) whenever the
    period is pinned. No falsifier change -- the claim already REQUIRES per-step
    `_current_e3_steps` and `|z_beta|` precisely so a reader can see this.

(d) SAMPLE FLOOR. `MIN_STEPS_PER_SEED = 800`, recorded as a precondition, not
    invented post hoc: the tightest bar any criterion turns on is C3's 0.10, whose
    binomial SE at n=800 is sqrt(0.1*0.9/800) = 0.0106 -- an order below C2's 0.08
    gap. 942's worst seed recorded 899 steps (early episode termination), so this
    floor is reachable at this schedule; EVAL_EPISODES is 30 rather than 942's 25
    for headroom against that same termination.

KNOWN LIMITATION, recorded (not a finding of this run): ARC-023's tightened
FALSIFYING leg is measured to be effectively unreachable -- clock-driven share is
clamped at <= 1/beta_rate_min_steps = 0.20 by update_e3_rate_from_beta, so
onset-gated resets must supply >= 0.13, and they measure 0.0075-0.0125 per step.
Raised as GFLAG-0494; this driver reports the leg faithfully rather than adjusting
it. Practical consequence: this run discriminates CONFIRMING vs PARTIAL.

WHY THIS IS NOT QUEUED (2026-09-25)
----------------------------------
The driver is complete, smoke-passing and validator-clean, and it is landed so that it
can be queued the moment the falsifier is repaired. It is NOT queued, because ARC-023's
tightened falsifier now carries THREE independently measured defects, and together they
leave the run with one reachable verdict cell:

  1. CONFIRMING leg (i) is confounded with reset traffic (GFLAG-0495, F1 above). Since
     a CONFIRM needs C1 to PASS, and C1 passes only at reset share roughly <= 0.02-0.08,
     CONFIRMING is gated on the MECH-091 harm trigger being nearly SILENT -- in an
     environment this lineage deliberately configures with num_hazards > 0 to LOAD that
     trigger. The confirming region and the ecological-load premise are in direct
     tension.
  2. FALSIFYING is effectively unreachable (GFLAG-0494): clock-driven share is clamped
     at <= 1/beta_rate_min_steps = 0.20, so onset-gated resets must supply >= 0.13, and
     they measure 0.0075-0.0125 per step.
  3. The 4-way REQUIRED RECORDING split has only 2 live classes, because the completion
     release threshold is unreachable as an identity and the NCL latch never arms
     (docstring (b) above).

So the reachable outcome space is essentially {PARTIAL}, and a run with one reachable
cell does not discriminate. Spending the compute would produce a PARTIAL that routes to
MECH-091 -- which is knowable from the probe already recorded, at no compute cost. The
falsifier needs repair first; that is a governance decision, not this driver's to make.

u/h/s/n/w -- see the refusal record on EXP-0548 and GFLAG-0495.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import reset_all_rng  # noqa: E402
from experiments._lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.baselines import arc023_e3_cadence as lineage  # noqa: E402
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1098_arc023_e3_cadence_recording_complete"
QUEUE_ID = "V3-EXQ-1098"
CLAIM_IDS: List[str] = ["ARC-023"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
RED_TEAM_VERDICT = "BLOCKING (fable, 2026-09-25): C1 confounded with reset traffic -- see GFLAG-0495"
RED_TEAM_DISPOSITIONS = {
    "F1_C1_measures_reset_traffic_not_period_tracking": (
        "VERIFIED and independently reproduced. With MultiRateClock's period held FIXED at "
        "10 -- so tracking is perfect by construction and MECH-093 is not involved -- the "
        "pre-registered ratio is 0.878 at reset share 0.02, 0.702 at 0.057, 0.520 at 0.102 "
        "and 0.149 at 0.253 (independent resets); clustering shifts it up (0.860 at 0.051 "
        "burst 3, 0.911 at 0.051 burst 6). Mechanism: advance() zeroes _e3_phase_step on a "
        "reset tick (clock.py), so every step in a reset-terminated cycle contributes 1/K to "
        "the expectation while being structurally unable to produce a clock-driven tick. "
        "PART FIXED: the old phase_step_zeroed_by_reset counter was incremented in the same "
        "branch as reset_driven, so it was identically equal to it and carried no "
        "information; replaced by the cycle ledger "
        "(expected_fraction_lost_to_reset_truncation, "
        "clock_tracking_ratio_clock_terminated_only), which makes a C1 failure ATTRIBUTABLE. "
        "PART NOT FIXABLE HERE: C1 is the claim's pre-registered formula and is computed "
        "verbatim. Raised as GFLAG-0495, not silently adjusted."),
    "F2_confirming_fired_without_any_seed_satisfying_the_conjunction": (
        "VERIFIED, FIXED, regression-tested. _adjudicate took each criterion's seed-majority "
        "independently, so rows (T,T,F),(T,F,T),(F,T,T) returned "
        "arc023_confirmed_phase1_rate_separation / supports / PASS while every seed printed "
        "verdict: FAIL. The claim reads 'on >= 2 of 3 seeds ... ALL of (i)(ii)(iii)', so the "
        "conjunction is per seed. Now seeds_satisfying_all_three. Confirmed the new code "
        "returns PARTIAL/FAIL on that exact counterexample, still CONFIRMS on a genuine "
        "2-of-3 conjunction, and still tests FALSIFYING first."),
    "F3_non_degeneracy_gate_satisfied_by_the_clock_episode_reset_artefact": (
        "VERIFIED against clock.py reset() (_current_e3_steps = _e3_base_steps) and "
        "agent.py:3916 (agent.reset() calls it). K is read BEFORE advance() while its only "
        "writer runs AFTER, so each episode's first recorded K is the BASE period "
        "unconditionally -- 30 such samples per seed, enough to make n_distinct >= 2 read "
        "true on a run pinned at any other value. FIXED: the precondition now reads "
        "..._n_distinct_steady, which excludes those samples; the episode-first samples and "
        "a full per-value histogram are recorded so the gate is re-derivable."),
    "F4_dv_headroom_punished_seed_agreement": (
        "VERIFIED as an own-goal in this driver's own added precondition (not the claim's). "
        "A cross-SEED range with a 0.01 floor would read met:false on 0.200/0.204/0.208 -- "
        "the shape of a clean, precise confirming run -- and the indexer's numeric recompute "
        "would then flag precondition_unmet and block scoring. FIXED: the denominator is now "
        "the range of per-episode shares POOLED over all seeds, and an undetermined case "
        "(fewer than 2 pooled episodes) is scoped out as cannot-determine rather than "
        "reported as a failure."),
    "F5_readiness_does_not_move_evidence_direction": (
        "PARTLY ACCEPTED, no change. The manifest can carry weakens/supports alongside "
        "non_degenerate:false, but the indexer excludes a non_degenerate:false run from "
        "scoring (scoring_excluded: degenerate), so it is contained. The reviewer's second "
        "half -- that the FALSIFY leg is unreachable at this config, leaving reachable "
        "directions {supports, mixed} -- is accurate and was already measured and raised by "
        "this session as GFLAG-0494 before the review ran."),
    "F6_small_numerator_and_attribution_leaks": (
        "ACCEPTED, magnitude ~n_episodes/n. The per-episode cache-miss regeneration was "
        "already recorded (e3_invocations_without_clock_tick). The stale-_window leak across "
        "an episode boundary is FIXED (mark_episode_start clears it). A harm reset requested "
        "on an episode's final step stays counted in requests_by_class, which is correct -- "
        "it WAS requested; the tick it would have produced is correctly absent."),
}

SEEDS: Tuple[int, ...] = (11, 23, 37)

# --- Pre-registered thresholds. ARC-023's own numbers; never derived from this run.
TRACK_TOLERANCE = 0.15          # leg (i): |ratio - 1| <= 0.15
SHARE_GAP_MIN = 0.08            # leg (ii): e2_share - e3_share_realized >= 0.08
RESET_SHARE_MAX = 0.10          # leg (iii): reset_driven_share <= 0.10
SEED_MAJORITY = 2               # "on >= 2 of 3 seeds"
MIN_STEPS_PER_SEED = 800        # sample floor -- see docstring (d)
MIN_DISTINCT_PERIODS = 2        # leg (i) non-degeneracy -- see docstring (c)

DRY_WARMUP_EPISODES = 2
DRY_EVAL_EPISODES = 2
DRY_STEPS_PER_EPISODE = 20

# 2026-09-25 pre-authoring probe, ree-cloud-4, 400 untrained steps. NOT measured by
# this run -- recorded as the non-gating 942-config contrast required by the
# ratifying decision's item (a). Provenance in the manifest block itself.
PROBE_REFERENCE = {
    "provenance": (
        "2026-09-25 pre-authoring probe on ree-cloud-4, 400 UNTRAINED steps at this "
        "lineage's ENV_KWARGS, seed 11. NOT measured by this run; non-gating. Method: "
        "REE_assembly/evidence/planning/arc023_recording_design_preflight_staged_20260925.md"
    ),
    "beta_gate_bistable_false_942_config": {
        "reset_driven_share": 0.2625, "e3_share_realized": 0.3600,
        "clock_tracking_ratio": 0.7047, "clock_driven_share": 0.0850,
    },
    "beta_gate_bistable_true_ratified_config": {
        "reset_driven_share": 0.1175, "e3_share_realized": 0.2300,
        "clock_tracking_ratio": 0.8580, "clock_driven_share": 0.1050,
    },
}

_ZG = ZGoalStreamAccumulator()


def _worst_ct_ratio(rows: List[Dict[str, Any]]) -> float:
    """Worst (largest-deviation-from-1) clock-terminated-only tracking ratio.

    NON-GATING. Paired with the pre-registered ratio so a reader can attribute a
    C1 failure: if the pre-registered ratio fails while THIS sits near 1.0, the
    entire shortfall is reset truncation discarding partial cycles, not the clock
    failing to honour its arousal-set period.
    """
    worst, best_dev = 0.0, -1.0
    for r in rows:
        v = r.get("clock_tracking_ratio_clock_terminated_only")
        val = float(v) if v is not None else 0.0
        dev = abs(val - 1.0)
        if dev > best_dev:
            best_dev, worst = dev, val
    return worst


def _worst_ratio_deviation(rows: List[Dict[str, Any]]) -> float:
    """Largest |ratio - 1| across seeds. A seed with NO clock-driven tick has
    ratio 0.0 (deviation 1.0), which is a real reading -- heavy reset traffic
    zeroed the phase counter before it ever reached K -- not a missing value."""
    devs = []
    for r in rows:
        v = r["criteria"]["C1_measured_ratio"]
        devs.append(abs(float(v) - 1.0) if v is not None else 1.0)
    return max(devs) if devs else 1.0


def _worst_ratio_value(rows: List[Dict[str, Any]]) -> float:
    """The raw ratio of the seed furthest from 1.0 (paired with the deviation so a
    reader can see WHICH side of the band it fell)."""
    worst, best_dev = 0.0, -1.0
    for r in rows:
        v = r["criteria"]["C1_measured_ratio"]
        val = float(v) if v is not None else 0.0
        dev = abs(val - 1.0)
        if dev > best_dev:
            best_dev, worst = dev, val
    return worst


def _evaluate_seed(row: Dict[str, Any], e2_share: float) -> Dict[str, Any]:
    """Apply the pre-registered criteria to ONE seed's recorded row."""
    ratio = row.get("clock_tracking_ratio")
    c1 = (ratio is not None) and (abs(float(ratio) - 1.0) <= TRACK_TOLERANCE)
    gap = float(row["e3_share_gap_below_e2"])
    c2 = gap >= SHARE_GAP_MIN
    reset_share = float(row["reset_driven_share"])
    c3 = reset_share <= RESET_SHARE_MAX
    # FALSIFYING leg: clock-driven + ONSET-GATED resets only.
    share_gated = float(row["e3_share_gated"])
    falsify = share_gated >= e2_share
    # The all-triggers excess, which the claim routes to MECH-091 when it is
    # attributable only to the un-gated harm trigger.
    share_all = float(row["e3_share_realized"])
    excess_all = share_all >= e2_share
    return {
        "C1_clock_tracking": c1,
        "C1_measured_ratio": (float(ratio) if ratio is not None else None),
        "C1_threshold_tolerance": TRACK_TOLERANCE,
        "C2_e3_slower_than_e2": c2,
        "C2_measured_gap": gap,
        "C2_threshold_gap_min": SHARE_GAP_MIN,
        "C3_reset_share_bounded": c3,
        "C3_measured_reset_share": reset_share,
        "C3_threshold_reset_share_max": RESET_SHARE_MAX,
        "FALSIFY_gated_share_ge_e2": falsify,
        "FALSIFY_measured_gated_share": share_gated,
        "FALSIFY_threshold_e2_share": e2_share,
        "excess_all_triggers": excess_all,
        "measured_all_trigger_share": share_all,
        "steps_recorded": int(row["steps_recorded"]),
        # STEADY, not raw: the raw set always contains clock.reset()'s base period
        # (one sample per episode), which would satisfy this gate by artefact.
        "period_non_degenerate": (
            int(row["current_e3_steps_n_distinct_steady"]) >= MIN_DISTINCT_PERIODS),
    }


def _run_seed(seed: int, warmup_episodes: int, eval_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    reset_all_rng(seed)

    env_warm = lineage.build_env(seed)
    agent = lineage.build_agent(env_warm)

    print("Seed %d Condition production" % seed, flush=True)
    total_progress = warmup_episodes + eval_episodes
    warmup_train(
        agent, env_warm,
        num_episodes=int(warmup_episodes), steps_per_episode=int(steps_per_episode),
        label="cadence seed=%d" % seed, progress_total_episodes=int(total_progress),
    )

    env_eval = lineage.build_env(seed)
    harness = StepHarness(agent, env_eval, train_mode=False, seed=seed)
    agent.eval()

    sites = lineage.resolve_trigger_sites()
    lineage.assert_trigger_wiring(sites)
    reachability = lineage.trigger_reachability(agent)

    with lineage.CadenceRecorder(agent, sites=sites) as rec:
        for ep in range(int(eval_episodes)):
            _flat, obs_dict = env_eval.reset()
            agent.reset()
            harness.reset()
            rec.mark_episode_start()
            for _t in range(int(steps_per_episode)):
                with torch.no_grad():
                    result = harness.step(obs_dict)
                obs_dict = result.next_obs_dict
                if result.done:
                    break
            print("  [train] cadence seed=%d ep %d/%d"
                  % (seed, warmup_episodes + ep + 1, total_progress), flush=True)

    row = rec.summary()
    row["seed"] = seed
    row["trigger_reachability"] = reachability
    _ZG.observe_stats(harness.z_goal_stream_stats())

    e2_share = float(row["e2_configured_share"])
    row["criteria"] = _evaluate_seed(row, e2_share)
    print("  [seed %d] leg(i) attribution: expected_lost_to_reset_truncation=%.1f%% "
          "ratio_clock_terminated_only=%s"
          % (seed, 100.0 * float(row.get("expected_fraction_lost_to_reset_truncation") or 0.0),
             ("%.4f" % row["clock_tracking_ratio_clock_terminated_only"])
             if row.get("clock_tracking_ratio_clock_terminated_only") is not None else "n/a"),
          flush=True)
    seed_pass = (row["criteria"]["C1_clock_tracking"]
                 and row["criteria"]["C2_e3_slower_than_e2"]
                 and row["criteria"]["C3_reset_share_bounded"])
    print("  [seed %d] steps=%d K_distinct=%s clock=%.4f reset=%.4f realized=%.4f "
          "gated=%.4f ratio=%s"
          % (seed, row["steps_recorded"], row["current_e3_steps_distinct"],
             row["clock_driven_share"], row["reset_driven_share"],
             row["e3_share_realized"], row["e3_share_gated"],
             ("%.4f" % row["criteria"]["C1_measured_ratio"])
             if row["criteria"]["C1_measured_ratio"] is not None else "n/a"),
          flush=True)
    print("verdict: %s" % ("PASS" if seed_pass else "FAIL"), flush=True)
    return row


def _adjudicate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the claim's >= 2-of-3-seed verdict grid. Order matters: FALSIFYING is
    tested first because the claim makes it the terminal reading."""
    n = len(rows)
    crit = [r["criteria"] for r in rows]
    n_c1 = sum(1 for c in crit if c["C1_clock_tracking"])
    n_c2 = sum(1 for c in crit if c["C2_e3_slower_than_e2"])
    n_c3 = sum(1 for c in crit if c["C3_reset_share_bounded"])
    n_fals = sum(1 for c in crit if c["FALSIFY_gated_share_ge_e2"])
    n_excess = sum(1 for c in crit if c["excess_all_triggers"])
    n_period_ok = sum(1 for c in crit if c["period_non_degenerate"])
    # PER-SEED CONJUNCTION. The claim reads "on >= 2 of 3 seeds ... (i) AND (ii)
    # AND (iii)", so the conjunction is taken WITHIN a seed and the majority over
    # seeds. Counting each criterion's majority independently is NOT the same test
    # and is strictly weaker: rows (T,T,F), (T,F,T), (F,T,T) give every criterion a
    # 2-of-3 majority while NO seed satisfies all three, so the independent form
    # returns CONFIRMING/supports on a run where every seed printed verdict: FAIL.
    n_conf = sum(1 for c in crit
                 if c["C1_clock_tracking"] and c["C2_e3_slower_than_e2"]
                 and c["C3_reset_share_bounded"])

    if n_fals >= SEED_MAJORITY:
        label = "arc023_falsified_e3_not_realized_as_slowest_loop"
        direction, outcome = "weakens", "FAIL"
    elif n_conf >= SEED_MAJORITY:
        label = "arc023_confirmed_phase1_rate_separation"
        direction, outcome = "supports", "PASS"
    else:
        if n_excess >= SEED_MAJORITY:
            label = ("arc023_partial_excess_attributable_to_ungated_harm_trigger"
                     "__routes_mech091_onset_gate")
        elif n_c3 < SEED_MAJORITY:
            label = "arc023_partial_reset_driven_share_above_bound__routes_mech091_onset_gate"
        elif n_c1 < SEED_MAJORITY and n_c2 >= SEED_MAJORITY:
            label = "arc023_partial_e3_slowest_but_not_tracking_arousal_set_period"
        else:
            label = "arc023_partial_mixed"
        direction, outcome = "mixed", "FAIL"

    return {
        "label": label,
        "evidence_direction": direction,
        "outcome": outcome,
        "seed_counts": {
            "n_seeds": n, "seed_majority_required": SEED_MAJORITY,
            "C1_clock_tracking": n_c1, "C2_e3_slower_than_e2": n_c2,
            "C3_reset_share_bounded": n_c3,
            "seeds_satisfying_all_three": n_conf,
            "FALSIFY_gated_share_ge_e2": n_fals,
            "excess_all_triggers": n_excess,
            "period_non_degenerate": n_period_ok,
        },
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    warmup = DRY_WARMUP_EPISODES if dry_run else lineage.WARMUP_EPISODES
    evals = DRY_EVAL_EPISODES if dry_run else lineage.EVAL_EPISODES
    steps = DRY_STEPS_PER_EPISODE if dry_run else lineage.STEPS_PER_EPISODE
    seeds = SEEDS[:1] if dry_run else SEEDS

    rows = [_run_seed(s, warmup, evals, steps) for s in seeds]
    verdict = _adjudicate(rows)

    e2_share = float(rows[0]["e2_configured_share"])
    worst_steps = min(int(r["steps_recorded"]) for r in rows)
    worst_periods = min(int(r["current_e3_steps_n_distinct_steady"]) for r in rows)
    periods_distinct = 3  # asserted in build_agent; recorded for re-derivability
    # dv_headroom denominator. An earlier draft used the CROSS-SEED range of
    # e3_share_realized, which is an own-goal: three seeds converging tightly
    # (0.200/0.204/0.208) is PRECISION, not a pinned DV, yet it would read
    # range 0.008 < 0.01 -> met:false -> `precondition_unmet` at the indexer, which
    # blocks scoring on exactly the cleanest confirming shape. Pool the PER-EPISODE
    # shares instead: that measures whether the statistic can move at all, which is
    # the actual headroom question, and it is well defined at one seed.
    _ep_shares: List[float] = []
    for r in rows:
        _ep_shares.extend([float(x) for x in r.get("per_episode_e3_share_realized", [])])
    dv_range = (max(_ep_shares) - min(_ep_shares)) if len(_ep_shares) >= 2 else 0.0

    preconditions = [
        {
            "name": "distinct_configured_periods",
            "kind": "readiness",
            "description": ("ARC-023's own non-degeneracy precondition: E1/E2/E3 must be "
                            "configured at three DISTINCT periods or the test is vacuous"),
            "measured": float(periods_distinct), "threshold": 3.0,
            "direction": "lower",
            "control": "count of distinct e1/e2/e3 steps_per_tick, asserted in build_agent",
            "met": periods_distinct >= 3,
        },
        {
            "name": "arousal_period_varies",
            "kind": "readiness",
            "description": ("leg (i) non-degeneracy: _current_e3_steps must take >= 2 values "
                            "within a run, else 'tracks sum_t 1/K(t)' is a near-identity and a "
                            "PASS on C1 is vacuous"),
            "measured": float(worst_periods), "threshold": float(MIN_DISTINCT_PERIODS),
            "direction": "lower",
            "control": ("WORST seed's n_distinct(_current_e3_steps) EXCLUDING each episode's "
                        "first sample, which is clock.reset()'s base period by construction "
                        "and would otherwise satisfy this gate by artefact"),
            "offending_cell": min(
                rows, key=lambda r: r["current_e3_steps_n_distinct_steady"])["seed"],
            "met": worst_periods >= MIN_DISTINCT_PERIODS,
        },
        {
            # dv_headroom for C2 -- validate_experiments'
            # criterion-exceeds-achievable-range check. C2 puts an ABSOLUTE floor
            # (0.08) on a derived gap, so the bar is only meaningful if the DV
            # underneath it can actually move. There is no control ARM here (the
            # falsifier is single-condition), so the available headroom measure is
            # the DV's realised CROSS-SEED range. Recorded prior measurements
            # straddle the implied bar of 0.2533 on e3_share_realized -- 942 got
            # 0.153 / 0.169 / 0.394, and this lineage's probe got 0.2300 (ratified
            # config) and 0.3600 (942 config) -- so the bar is attainable in BOTH
            # directions, which is what this precondition asserts is still true.
            "name": "dv_headroom_e3_share_realized_range",
            "kind": "readiness",
            "description": ("C2's DV (e3_share_realized) must show non-trivial realised range "
                            "across measurement episodes, else the 0.08 absolute gap bar cuts "
                            "a pinned statistic"),
            "measured": float(dv_range), "threshold": 0.01,
            "direction": "lower",
            "control": ("realised range of e3_share_realized POOLED over all measurement "
                        "episodes of all seeds (NOT the cross-seed range -- seed agreement is "
                        "precision, not a pinned DV); bar-equivalent value is "
                        "e2_share - 0.08 = %.4f, which 942 (0.153/0.169/0.394) and this "
                        "lineage's probe (0.2300/0.3600) both straddle"
                        % (e2_share - SHARE_GAP_MIN)),
            "met": (dv_range >= 0.01) or (len(_ep_shares) < 2),
            "n_episodes_pooled": len(_ep_shares),
            "scoped_out": (len(_ep_shares) < 2),
            "applies_note": ("a realised RANGE is undefined with fewer than 2 pooled "
                             "episodes (--dry-run); scoped out as cannot-determine rather "
                             "than reported as a failed precondition"),
        },
        {
            "name": "min_steps_per_seed",
            "kind": "readiness",
            "description": ("sample floor: binomial SE at C3's 0.10 bar with n=800 is 0.0106, "
                            "an order below C2's 0.08 gap"),
            "measured": float(worst_steps), "threshold": float(MIN_STEPS_PER_SEED),
            "direction": "lower",
            "control": "WORST seed's recorded eval steps (episodes can terminate early)",
            "offending_cell": min(rows, key=lambda r: r["steps_recorded"])["seed"],
            "met": worst_steps >= MIN_STEPS_PER_SEED,
        },
    ]

    # C1 is degenerate wherever the period is pinned; C2/C3 wherever the sample floor fails.
    sample_ok = worst_steps >= MIN_STEPS_PER_SEED
    period_ok = worst_periods >= MIN_DISTINCT_PERIODS
    criteria_non_degenerate = {
        "C1_clock_tracking": bool(period_ok and sample_ok),
        "C2_e3_slower_than_e2": bool(sample_ok),
        "C3_reset_share_bounded": bool(sample_ok),
    }

    combination_rule = (
        "CONFIRMING requires C1 AND C2 AND C3 each on >= %d of %d seeds. FALSIFYING is "
        "tested FIRST and requires (clock-driven + ONSET-GATED reset) share >= E2's "
        "configured share on >= %d seeds. Anything else is PARTIAL; an all-trigger excess "
        "that the gated-only share does not reproduce is attributable to the un-gated "
        "MECH-091 harm trigger and routes to MECH-091's onset gate, NOT to a falsification "
        "of ARC-023." % (SEED_MAJORITY, len(seeds), SEED_MAJORITY)
    )

    criteria_list = [
        {"name": "CONFIRMING_conjunction_per_seed", "load_bearing": True,
         "passed": verdict["seed_counts"]["seeds_satisfying_all_three"] >= SEED_MAJORITY,
         "measured": verdict["seed_counts"]["seeds_satisfying_all_three"],
         "threshold": SEED_MAJORITY, "unit": "seeds",
         "note": "C1 AND C2 AND C3 within a single seed, then majority over seeds"},
        {"name": "C1_clock_tracking", "load_bearing": True,
         "passed": verdict["seed_counts"]["C1_clock_tracking"] >= SEED_MAJORITY,
         "measured": verdict["seed_counts"]["C1_clock_tracking"],
         "threshold": SEED_MAJORITY, "unit": "seeds",
         "per_seed_measured": [c["criteria"]["C1_measured_ratio"] for c in rows],
         "per_seed_threshold_tolerance": TRACK_TOLERANCE},
        {"name": "C2_e3_slower_than_e2", "load_bearing": True,
         "passed": verdict["seed_counts"]["C2_e3_slower_than_e2"] >= SEED_MAJORITY,
         "measured": verdict["seed_counts"]["C2_e3_slower_than_e2"],
         "threshold": SEED_MAJORITY, "unit": "seeds",
         "per_seed_measured": [c["criteria"]["C2_measured_gap"] for c in rows],
         "per_seed_threshold": SHARE_GAP_MIN},
        {"name": "C3_reset_share_bounded", "load_bearing": True,
         "passed": verdict["seed_counts"]["C3_reset_share_bounded"] >= SEED_MAJORITY,
         "measured": verdict["seed_counts"]["C3_reset_share_bounded"],
         "threshold": SEED_MAJORITY, "unit": "seeds",
         "per_seed_measured": [c["criteria"]["C3_measured_reset_share"] for c in rows],
         "per_seed_threshold": RESET_SHARE_MAX},
        {"name": "FALSIFY_gated_share_ge_e2", "load_bearing": True,
         "passed": verdict["seed_counts"]["FALSIFY_gated_share_ge_e2"] >= SEED_MAJORITY,
         "measured": verdict["seed_counts"]["FALSIFY_gated_share_ge_e2"],
         "threshold": SEED_MAJORITY, "unit": "seeds",
         "per_seed_measured": [c["criteria"]["FALSIFY_measured_gated_share"] for c in rows],
         "per_seed_threshold": e2_share},
    ]

    degeneracy_reason = None
    non_degenerate = True
    if not period_ok:
        non_degenerate = False
        degeneracy_reason = (
            "_current_e3_steps took %d distinct value(s) on the worst seed, so CONFIRMING "
            "leg (i) reduces to the near-identity 'a clock of period K fires N/K times' and "
            "a C1 PASS is vacuous. This is the pinning substrate_queue "
            "mech005-endogenous-arousal-dynamic-range predicts on a trained agent "
            "(endogenous ||z_beta|| spread 0.91%% of its mean)." % worst_periods
        )
    elif not sample_ok:
        non_degenerate = False
        degeneracy_reason = (
            "worst seed recorded %d steps, below the pre-registered %d floor, so the share "
            "estimates the criteria turn on are under-powered." % (worst_steps, MIN_STEPS_PER_SEED)
        )

    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    full_config = {
        "env": dict(lineage.ENV_KWARGS),
        "alpha_world": lineage.ALPHA_WORLD,
        "beta_gate_bistable": lineage.BETA_GATE_BISTABLE,
        "beta_gate_bistable_authority": lineage.BISTABLE_DECISION,
        "warmup_episodes": warmup, "eval_episodes": evals,
        "steps_per_episode": steps,
        "e2_configured_share": e2_share,
        "pre_registered": {
            "track_tolerance": TRACK_TOLERANCE, "share_gap_min": SHARE_GAP_MIN,
            "reset_share_max": RESET_SHARE_MAX, "seed_majority": SEED_MAJORITY,
            "min_steps_per_seed": MIN_STEPS_PER_SEED,
            "min_distinct_periods": MIN_DISTINCT_PERIODS,
        },
        "lineage_config_slice": lineage.config_slice(),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": list(CLAIM_IDS),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": verdict["outcome"],
        "evidence_direction": verdict["evidence_direction"],
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "non_degenerate": non_degenerate,
        "sleep_driver_pattern": "not_applicable",
        "red_team_verdict": RED_TEAM_VERDICT,
        "interpretation": {
            "label": verdict["label"],
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "routes_to": ("MECH-091 onset gate" if "mech091" in verdict["label"] else None),
        },
        "criteria": criteria_list,
        "combination_rule": combination_rule,
        "per_seed_results": rows,
        "diagnostics": {
            "trigger_reachability": rows[0]["trigger_reachability"],
            "trigger_classes_live": [
                c for c, v in rows[0]["trigger_reachability"].items()
                if v.get("status") == "live"
            ],
            "trigger_classes_structurally_unreachable": [
                c for c, v in rows[0]["trigger_reachability"].items()
                if v.get("status") == "structurally_unreachable"
            ],
            "probe_reference_942_config": PROBE_REFERENCE,
            "falsifying_leg_reachability_note": (
                "GFLAG-0494: clock-driven share is clamped at <= 1/beta_rate_min_steps = 0.20 "
                "by update_e3_rate_from_beta, so the FALSIFYING leg needs onset-gated reset "
                "share >= 0.13 (>= 0.21 at a realistic clock share); onset-gated resets "
                "measured 0.0075-0.0125 in the 2026-09-25 probe. Reported faithfully here, "
                "not adjusted."
            ),
        },
    }
    if degeneracy_reason is not None:
        manifest["degeneracy_reason"] = degeneracy_reason

    manifest["readout"] = flat_readout({
        "dv_headroom_e3_share_realized_range": float(dv_range),
        "n_seeds_satisfying_all_three": verdict["seed_counts"]["seeds_satisfying_all_three"],
        # leg (i) attribution -- non-gating, see the lineage cycle-ledger comment
        "expected_fraction_lost_to_reset_truncation_max": max(
            float(r.get("expected_fraction_lost_to_reset_truncation") or 0.0) for r in rows),
        "clock_tracking_ratio_clock_terminated_only_worst": _worst_ct_ratio(rows),
        "current_e3_steps_n_distinct_steady_worst": worst_periods,
        "n_seeds": len(rows),
        # WORST cell, not the mean and not the best: C1's verdict is a per-seed
        # worst-case claim, and the indexer recomputes `met` from the number we
        # report, so reporting min() here would mask an out-of-band seed.
        "clock_tracking_abs_deviation_worst": _worst_ratio_deviation(rows),
        "clock_tracking_ratio_worst_cell": _worst_ratio_value(rows),
        "e3_share_realized_max": max(float(r["e3_share_realized"]) for r in rows),
        "e3_share_gap_below_e2_min": min(float(r["e3_share_gap_below_e2"]) for r in rows),
        "reset_driven_share_max": max(float(r["reset_driven_share"]) for r in rows),
        "e3_share_gated_max": max(float(r["e3_share_gated"]) for r in rows),
        "clock_driven_share_mean": sum(float(r["clock_driven_share"]) for r in rows) / len(rows),
        "e2_configured_share": e2_share,
        "steps_recorded_worst": worst_steps,
        "completion_signal_max": max(float(r["completion_signal_max"]) for r in rows),
        "n_seeds_C1": verdict["seed_counts"]["C1_clock_tracking"],
        "n_seeds_C2": verdict["seed_counts"]["C2_e3_slower_than_e2"],
        "n_seeds_C3": verdict["seed_counts"]["C3_reset_share_bounded"],
        "n_seeds_FALSIFY": verdict["seed_counts"]["FALSIFY_gated_share_ge_e2"],
        "non_degenerate": non_degenerate,
    })

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=list(seeds),
        script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    return {"outcome": verdict["outcome"], "manifest_path": out_path,
            "label": verdict["label"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V3-EXQ-1098 ARC-023 E3-cadence recording-complete run")
    ap.add_argument("--dry-run", action="store_true", help="short smoke; manifest relocated")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_path = result["manifest_path"]
    print("label: %s" % result["label"], flush=True)
    print("manifest: %s" % out_path, flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
