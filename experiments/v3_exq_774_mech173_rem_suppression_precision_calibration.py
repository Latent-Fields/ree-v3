#!/opt/local/bin/python3
"""V3-EXQ-774 -- MECH-173: REM-suppression precision-recalibration confidence-accuracy probe.

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

experiment_purpose: DIAGNOSTIC. This probe does NOT assume MECH-173 is expressible
on the current substrate -- it MEASURES whether it is. The built MECH-204 REM
precision-recalibration consumer (sleep_substrate:GAP-1, F1 closure, V3-EXQ-541c
PASS) is a persistent-zero-point low-pass SETPOINT filter on E3._running_variance
(serotonin.compute_recalibration_target -> e3.recalibrate_precision_to at the
WRITEBACK phase). 541b/c found its per-cycle effect is small and its cross-arm
divergence sub-threshold (<5%). So a naive MECH-173 *evidence* run risks scoring a
substrate ceiling as "weakens MECH-173". A diagnostic (excluded from confidence
scoring) plus a P0 readiness gate self-routes an inadequate substrate to
substrate_not_ready_requeue instead of a false claim verdict.

MECH-173 (the claim under probe):
  REM-suppressing medications (anticholinergics, MAOIs, most antidepressants,
  benzodiazepines) selectively impair MECH-123 precision recalibration -- the
  most-downstream offline phase -- producing the earliest dementia prodrome:
  overconfident contextual attributions before overt memory loss (subjective
  normalcy, objective deficit). i.e. internal precision/commitment stays high
  while the accuracy of the agent's own uncertainty estimate degrades.

Substrate model of "REM-suppressing medication":
  A learning agent's instantaneous forward-model error DROPS over training. REM
  precision recalibration pulls E3's running_variance toward the accumulated
  persistent zero-point (a long-horizon uncertainty baseline), providing
  "epistemic humility": it keeps the internal uncertainty estimate anchored to
  the accumulated baseline rather than the transiently-low recent error.
  Suppressing REM removes that anchor, leaving precision at the mercy of recent
  (low) error -> the agent under-estimates its uncertainty -> OVERCONFIDENCE.

Arms (4; same env + seed across arms -> paired):
  ARM_FULL_SLEEP    sws=T rem=T recalib=T  -- unmedicated reference.
  ARM_REM_SUPPRESSED sws=T rem=F recalib=T(gated off by rem=F) -- biological
                     REM-suppressing-medication model (REM sleep abolished, SWS
                     intact). Removes REM recalibration AND REM attribution.
  ARM_RECALIB_OFF   sws=T rem=T recalib=F  -- isolates the MECH-123/204 precision
                     recalibration function ONLY (REM otherwise intact). The clean
                     mechanistic ablation for MECH-173's "impairs MECH-123" clause.
  ARM_SWS_SUPPRESSED sws=F rem=T recalib=T -- SELECTIVITY control (SWS abolished,
                     REM + recalibration intact). MECH-173 predicts SWS suppression
                     does NOT hit precision recalibration (that is MECH-174's lane),
                     so this arm should behave like FULL_SLEEP, not like the REM
                     arms.

Signals (real dynamics; NOT synthetic PE):
  StepHarness drives sense -> ... -> update_residue -> e3.post_action_update, which
  updates running_variance from the genuine E2 forward-model error
  (actual_z_world - predicted_world) and surfaces that error as
  residue_metrics["e3_prediction_error"] (mean squared PE this tick). So:
    confidence = e3.current_precision (= 1/running_variance; commitment fires when
                 running_variance < commit_threshold) -- includes recalibration.
    accuracy   = e3_prediction_error (real forward-model squared error) -- an
                 INDEPENDENT ground-truth of how well the world model actually
                 predicts; NOT modified by recalibration.

Load-bearing statistic (continuous, variance-bearing -- not a saturating binary,
per the 767/768 vacuous-pass lesson):
  true_error_ref   = mean over eval ticks of e3_prediction_error (the arm's true
                     typical forward-model error).
  overconfidence_index = (true_error_ref - mean(running_variance)) / true_error_ref
                     Positive => the agent's internal uncertainty sits BELOW its
                     true typical error = OVERCONFIDENT. For a recalibration-OFF
                     arm running_variance is a plain EMA of the error, so this
                     index sits near 0 by construction (clean baseline); the
                     recalibration nudge is what shifts it. The cross-arm DELTA
                     therefore isolates the recalibration effect.

Paired deltas (per seed, vs FULL_SLEEP):
  delta_recalib = overconf[RECALIB_OFF]   - overconf[FULL_SLEEP]  (clean ablation)
  delta_rem     = overconf[REM_SUPPRESSED]- overconf[FULL_SLEEP]  (medication model)
  delta_sws     = overconf[SWS_SUPPRESSED]- overconf[FULL_SLEEP]  (selectivity)

Significance gate (SD-of-delta + absolute floor, per feedback_effect_size_pass_gate):
  a delta is "significant positive" iff mean_delta > max(ABS_FLOOR, K_SD * sd_delta)
  and mean_delta > 0.

P0 readiness preconditions (measured; gate the self-routed LABEL):
  rv_live_all_arms          : every cell's rv_final differs from precision_init
                              (0.5) by > RV_LIVE_FLOOR (the Q-042 / V3-EXQ-530c
                              contract -- else the loop never drove precision and
                              the whole run is vacuous).
  recalib_engaged_full_sleep: FULL_SLEEP recalibration fired on >=1 cycle AND its
                              mean |rv_after - rv_before| per fired cycle >
                              RECALIB_MOVE_FLOOR (the lever is actually engaged --
                              guards against the F1 no-op the 541 finding warns of).

Self-route (diagnostic label; NOT a scored claim verdict):
  substrate_not_ready_requeue                        -- a readiness precondition unmet.
  substrate_ceiling_recalibration_subthreshold       -- readiness met but cross-arm
      overconfidence spread below NONDEGEN_FLOOR (the built F1 setpoint cannot move
      calibration enough to express MECH-173). Route: /implement-substrate (Phase 7
      / Option B per the REM-precision lit-pull SYNTHESIS, sleep_substrate_plan.md).
  supports_mech173_rem_precision_selective_overconfidence -- delta_recalib (and
      delta_rem) significant-positive AND selectivity holds (|delta_sws| below the
      REM-arm effect). Supports MECH-173; a follow-on may re-run as evidence.
  does_not_support_f1_setpoint_reverses_sign         -- delta_recalib significant
      NEGATIVE (the built F1 setpoint reduces overconfidence in the ablation
      direction opposite to MECH-173's prediction -> MECH-173 needs a different
      recalibration mechanism; route /implement-substrate Phase 7).
  mixed_inconclusive                                 -- otherwise.

claim_ids: ["MECH-173"]  (diagnostic -> excluded from confidence/conflict scoring)
backlog_id: EVB-0116  (IGW-20260717-207)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-774"
EXPERIMENT_TYPE = "v3_exq_774_mech173_rem_suppression_precision_calibration"
CLAIM_IDS = ["MECH-173"]
EXPERIMENT_PURPOSE = "diagnostic"
BACKLOG_ID = "EVB-0116"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"

N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_SEEDS = 3
GRID_SIZE = 12
STEPS_PER_EP = 200
LR = 5e-4

# Substrate-liveness / recalibration constants.
PRECISION_INIT_BASELINE = 0.5      # REEConfig precision_init default
RV_LIVE_FLOOR = 1e-6               # rv_final must differ from precision_init by more than this
RECALIB_MOVE_FLOOR = 1e-4          # FULL_SLEEP mean per-cycle |rv_after-rv_before| must exceed this

# Significance / degeneracy gates. Units are natural-log ratio (overconfidence_score
# = log(true_error_ref / mean_running_variance); >0 = absolutely overconfident,
# <0 = under-confident/humble, 0 = calibrated).
ABS_FLOOR = 0.10                   # minimum |mean paired delta| in log-ratio units (~10% rv shift)
K_SD = 1.0                         # delta must exceed K_SD * sd(paired delta)
NONDEGEN_FLOOR = 0.05              # cross-arm spread in overconfidence_score below this = degenerate
ABS_OVERCONF_MARGIN = 0.10         # a cell is ABSOLUTELY overconfident iff score > this (rv below true error)
ACCURACY_DISSOCIATION_TOL = 0.25   # max cross-arm relative spread in true_error_ref for a clean dissociation

# Sleep / precision substrate knobs (held constant across arms; arms vary only the
# sws_enabled / rem_enabled / use_rem_precision_recalibration switches).
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# (arm_id, sws_enabled, rem_enabled, use_rem_precision_recalibration)
ARMS = (
    ("ARM_FULL_SLEEP", True, True, True),
    ("ARM_REM_SUPPRESSED", True, False, True),
    ("ARM_RECALIB_OFF", True, True, False),
    ("ARM_SWS_SUPPRESSED", False, True, True),
)


def _make_env(seed: int, dry_run: bool = False) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=(8 if dry_run else GRID_SIZE),
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.10,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _arm_config_slice(sws: bool, rem: bool, recalib: bool) -> Dict:
    """The config the cell's build+collect path actually reads (arm-distinguishing
    switches plus the shared sleep/precision operating point)."""
    return {
        "grid_size": GRID_SIZE,
        "steps_per_ep": STEPS_PER_EP,
        "n_train_eps": N_TRAIN_EPS,
        "n_eval_eps": N_EVAL_EPS,
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_enabled": sws,
        "rem_enabled": rem,
        "use_rem_precision_recalibration": recalib,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "sleep_loop_episodes_K": 1,
    }


def _make_agent(env: CausalGridWorldV2, sws: bool, rem: bool, recalib: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        sws_enabled=sws,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=rem,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        use_rem_precision_recalibration=recalib,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=REM_PRECISION_RECALIBRATION_STEP,
    )
    # Tonic 5-HT must be enabled for compute_recalibration_target() to be meaningful
    # (the WRITEBACK recalibration is gated on serotonin.enabled).
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _read_recalib_metrics(agent: REEAgent) -> Optional[Dict[str, float]]:
    """Read the sleep cycle telemetry that the just-fired agent.reset() left in
    sleep_loop.state.last_metrics. Returns None when no sleep fired this boundary."""
    if agent.sleep_loop is None:
        return None
    state = agent.sleep_loop.state
    if state is None or not state.last_metrics:
        return None
    m = dict(state.last_metrics)
    out: Dict[str, float] = {}
    if "mech204_recalibration_fired" in m:
        out["fired"] = float(m.get("mech204_recalibration_fired", 0.0))
    if "mech204_running_variance_before" in m and "mech204_running_variance_after" in m:
        out["rv_before"] = float(m["mech204_running_variance_before"])
        out["rv_after"] = float(m["mech204_running_variance_after"])
    return out or None


def _run_arm_seed(arm, seed, n_train, n_eval, steps, dry_run=False) -> Dict:
    arm_label, sws, rem, recalib = arm
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = _make_env(seed, dry_run=dry_run)
    agent = _make_agent(env, sws, rem, recalib)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    print(f"Seed {seed} Condition {arm_label}", flush=True)

    # ---- Training phase (forward model learns -> instantaneous error drops) ----
    train_harness = StepHarness(agent, env, train_mode=True, seed=seed)
    for ep in range(n_train):
        agent.reset()                    # fires sleep for the prior episode (K=1)
        _, obs_dict = env.reset()
        train_harness.reset()
        for _ in range(steps):
            result = train_harness.step(obs_dict)
            optimizer.zero_grad()
            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            obs_dict = result.next_obs_dict
            if result.done:
                break
        if (ep + 1) % 5 == 0 or ep + 1 == n_train:
            print(
                f"  [train] arm={arm_label} seed={seed} ep {ep + 1}/{n_train} "
                f"rv={float(agent.e3._running_variance):.6f} "
                f"prec={float(agent.e3.current_precision):.4f}",
                flush=True,
            )
    rv_after_training = float(agent.e3._running_variance)

    # ---- Eval phase (no grad); capture confidence, accuracy, recalibration ----
    eval_harness = StepHarness(agent, env, train_mode=False, seed=seed + 10000)
    rv_vals: List[float] = []
    precision_vals: List[float] = []
    committed_vals: List[bool] = []
    pe_vals: List[float] = []
    recalib_fired_cycles = 0
    recalib_moves: List[float] = []

    for _ep in range(n_eval):
        agent.reset()                    # fires sleep for the prior episode
        rc = _read_recalib_metrics(agent)
        if rc is not None:
            if rc.get("fired", 0.0) >= 1.0:
                recalib_fired_cycles += 1
            if "rv_before" in rc and "rv_after" in rc:
                recalib_moves.append(abs(rc["rv_after"] - rc["rv_before"]))
        _, obs_dict = env.reset()
        eval_harness.reset()
        for _ in range(steps):
            result = eval_harness.step(obs_dict)
            rv_vals.append(float(agent.e3._running_variance))
            precision_vals.append(float(agent.e3.current_precision))
            committed_vals.append(bool(agent.beta_gate.is_elevated))
            pe = result.residue_metrics.get("e3_prediction_error")
            if pe is not None:
                pe_vals.append(float(pe.detach().item()) if torch.is_tensor(pe) else float(pe))
            obs_dict = result.next_obs_dict
            if result.done:
                break

    rv_final = float(agent.e3._running_variance)
    n_eval_ticks = len(rv_vals)
    n_pe = len(pe_vals)

    true_error_ref = float(sum(pe_vals) / n_pe) if n_pe else 0.0
    mean_rv = float(sum(rv_vals) / n_eval_ticks) if n_eval_ticks else 0.0
    mean_precision = float(sum(precision_vals) / n_eval_ticks) if n_eval_ticks else 0.0
    commit_rate = float(sum(1 for c in committed_vals if c) / n_eval_ticks) if n_eval_ticks else 0.0
    # calibration_ratio = internal uncertainty estimate / true forward-model error.
    #   >1 under-confident (humble), ~1 calibrated, <1 OVERCONFIDENT.
    # overconfidence_score = log(true_error_ref / mean_rv): >0 overconfident, <0 humble,
    #   0 calibrated. Stable, signed, dimensionless (avoids the tiny-denominator blow-up).
    if true_error_ref > 1e-9 and mean_rv > 1e-9:
        calibration_ratio = mean_rv / true_error_ref
        overconfidence_score = float(np.log(true_error_ref / mean_rv))
    else:
        calibration_ratio = 0.0
        overconfidence_score = 0.0
    recalib_move_mean = float(sum(recalib_moves) / len(recalib_moves)) if recalib_moves else 0.0

    passed = (
        abs(rv_final - PRECISION_INIT_BASELINE) > RV_LIVE_FLOOR
        and n_eval_ticks > 0
        and n_pe > 0
    )
    print(
        f"  arm={arm_label} seed={seed} overconf_score={overconfidence_score:.4f} "
        f"calib_ratio={calibration_ratio:.3f} true_err={true_error_ref:.6f} "
        f"mean_rv={mean_rv:.6f} rv_final={rv_final:.6f} "
        f"recalib_fired={recalib_fired_cycles} recalib_move={recalib_move_mean:.6f} "
        f"commit={commit_rate:.3f}",
        flush=True,
    )
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    return {
        "arm_id": arm_label,
        "seed": seed,
        "sws_enabled": sws,
        "rem_enabled": rem,
        "use_rem_precision_recalibration": recalib,
        "overconfidence_score": float(overconfidence_score),
        "calibration_ratio": float(calibration_ratio),
        "true_error_ref": true_error_ref,
        "mean_running_variance": mean_rv,
        "mean_precision": mean_precision,
        "commit_rate": commit_rate,
        "rv_after_training": rv_after_training,
        "rv_final": rv_final,
        "rv_live": bool(abs(rv_final - PRECISION_INIT_BASELINE) > RV_LIVE_FLOOR),
        "recalib_fired_cycles": recalib_fired_cycles,
        "recalib_move_mean": recalib_move_mean,
        "n_eval_ticks": n_eval_ticks,
        "n_pe_samples": n_pe,
    }


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _sd(xs: List[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def _adjudicate(cells: List[Dict], seeds: List[int]) -> Dict:
    by_arm: Dict[str, Dict[int, Dict]] = {}
    for c in cells:
        by_arm.setdefault(c["arm_id"], {})[c["seed"]] = c

    arm_score = {arm: _mean([by_arm[arm][s]["overconfidence_score"] for s in seeds])
                 for arm in by_arm}
    arm_calib_ratio = {arm: _mean([by_arm[arm][s]["calibration_ratio"] for s in seeds])
                       for arm in by_arm}
    arm_true_error = {arm: _mean([by_arm[arm][s]["true_error_ref"] for s in seeds])
                      for arm in by_arm}

    # Paired deltas of overconfidence_score vs FULL_SLEEP (positive = arm relatively
    # MORE overconfident than the unmedicated reference).
    def paired_delta(arm: str) -> List[float]:
        return [by_arm[arm][s]["overconfidence_score"] - by_arm["ARM_FULL_SLEEP"][s]["overconfidence_score"]
                for s in seeds]

    d_recalib = paired_delta("ARM_RECALIB_OFF")
    d_rem = paired_delta("ARM_REM_SUPPRESSED")
    d_sws = paired_delta("ARM_SWS_SUPPRESSED")
    m_recalib, sd_recalib = _mean(d_recalib), _sd(d_recalib)
    m_rem, sd_rem = _mean(d_rem), _sd(d_rem)
    m_sws, sd_sws = _mean(d_sws), _sd(d_sws)

    def sig_pos(m: float, sd: float) -> bool:
        return m > 0 and m > max(ABS_FLOOR, K_SD * sd)

    def sig_neg(m: float, sd: float) -> bool:
        return m < 0 and (-m) > max(ABS_FLOOR, K_SD * sd)

    # --- P0 readiness preconditions (measured) ---
    min_rv_diff = min(abs(c["rv_final"] - PRECISION_INIT_BASELINE) for c in cells)
    fs_cells = [by_arm["ARM_FULL_SLEEP"][s] for s in seeds]
    fs_fired = min(c["recalib_fired_cycles"] for c in fs_cells)
    fs_move = _mean([c["recalib_move_mean"] for c in fs_cells])

    rv_live_met = bool(min_rv_diff > RV_LIVE_FLOOR)
    recalib_engaged_met = bool(fs_fired >= 1 and fs_move > RECALIB_MOVE_FLOOR)

    preconditions = [
        {"name": "rv_live_all_arms",
         "description": "every cell rv_final differs from precision_init by > floor (Q-042/530c contract)",
         "measured": float(min_rv_diff), "threshold": RV_LIVE_FLOOR, "met": rv_live_met},
        {"name": "recalib_engaged_full_sleep",
         "description": "FULL_SLEEP mean per-cycle |rv_after-rv_before| exceeds floor (lever engaged)",
         "measured": float(fs_move), "threshold": RECALIB_MOVE_FLOOR,
         "control": "FULL_SLEEP arm cross-cycle WRITEBACK recalibration movement", "met": recalib_engaged_met},
        {"name": "recalib_fired_full_sleep_cycles",
         "description": "FULL_SLEEP recalibration fired on at least one sleep cycle in every seed",
         "measured": float(fs_fired), "threshold": 1.0, "met": bool(fs_fired >= 1)},
    ]
    readiness_ok = rv_live_met and recalib_engaged_met and fs_fired >= 1

    # --- Non-degeneracy: cross-arm spread in overconfidence_score (same statistic the
    #     load-bearing deltas route on) ---
    spread = max(arm_score.values()) - min(arm_score.values())
    non_degenerate = bool(spread > NONDEGEN_FLOOR)

    # --- MECH-173 direction + absolute-overconfidence tests ---
    recalib_positive = sig_pos(m_recalib, sd_recalib)   # ablation shifts confidence up (relatively)
    rem_positive = sig_pos(m_rem, sd_rem)
    recalib_negative = sig_neg(m_recalib, sd_recalib)   # ablation LOWERS confidence (F1 reverses sign)
    # Selectivity: SWS suppression must NOT reproduce the REM-arm confidence shift.
    selectivity_ok = (abs(m_sws) < max(ABS_FLOOR, 0.5 * abs(m_recalib))) if recalib_positive else True
    # Absolute overconfidence: is a REM/recalib-suppressed arm actually overconfident
    # (score > margin => internal uncertainty BELOW true forward-model error), the
    # specific MECH-173 prodrome claim (not merely "less humble than full sleep")?
    suppressed_absolutely_overconfident = bool(
        arm_score["ARM_REM_SUPPRESSED"] > ABS_OVERCONF_MARGIN
        or arm_score["ARM_RECALIB_OFF"] > ABS_OVERCONF_MARGIN
    )
    # Accuracy dissociation: forward-model error (accuracy) roughly constant across arms,
    # so any confidence shift is NOT an accuracy artifact.
    mean_true_err = _mean(list(arm_true_error.values()))
    acc_spread_rel = (
        (max(arm_true_error.values()) - min(arm_true_error.values())) / mean_true_err
        if mean_true_err > 1e-9 else 0.0
    )
    accuracy_dissociation = bool(acc_spread_rel <= ACCURACY_DISSOCIATION_TOL)

    criteria = [
        {"name": "delta_recalib_positive_significant", "load_bearing": True, "passed": bool(recalib_positive)},
        {"name": "selectivity_sws_null", "load_bearing": True, "passed": bool(selectivity_ok)},
        {"name": "suppressed_arm_absolutely_overconfident", "load_bearing": True,
         "passed": bool(suppressed_absolutely_overconfident)},
    ]
    criteria_non_degenerate = {"cross_arm_overconfidence_score_spread": non_degenerate}

    # --- Self-route ---
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome, direction = "FAIL", "non_contributory"
    elif not non_degenerate:
        label = "substrate_ceiling_recalibration_subthreshold"
        outcome, direction = "FAIL", "non_contributory"
    elif recalib_positive and selectivity_ok and rem_positive and suppressed_absolutely_overconfident:
        # Full MECH-173 signature: selective, direction-correct, AND crosses into
        # absolute overconfidence.
        label = "supports_mech173_rem_precision_selective_overconfidence"
        outcome, direction = "PASS", "supports"
    elif recalib_positive and selectivity_ok:
        # Direction-correct + selective, but the suppressed arm stays under-confident:
        # the built F1 setpoint anchors precision ABOVE forward-model accuracy, so its
        # loss reduces humility rather than causing overconfidence. MECH-173's overconfidence
        # prodrome needs a recalibration mechanism anchored to attribution ACCURACY, not to a
        # lagging precision setpoint. Route: /implement-substrate (Phase 7 / Option B).
        label = "partial_direction_no_absolute_overconfidence"
        outcome, direction = "FAIL", "does_not_support"
    elif recalib_negative:
        # F1 setpoint suppression LOWERS confidence: opposite of MECH-173's prediction.
        label = "does_not_support_f1_setpoint_reverses_sign"
        outcome, direction = "FAIL", "does_not_support"
    else:
        label = "mixed_inconclusive"
        outcome, direction = "FAIL", "inconclusive"

    return {
        "label": label,
        "outcome": outcome,
        "evidence_direction": direction,
        "preconditions": preconditions,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": ("" if non_degenerate else
                              f"cross-arm overconfidence_score spread {spread:.4f} <= NONDEGEN_FLOOR {NONDEGEN_FLOOR}"),
        "arm_overconfidence_score": arm_score,
        "arm_calibration_ratio": arm_calib_ratio,
        "arm_true_error_ref": arm_true_error,
        "suppressed_absolutely_overconfident": suppressed_absolutely_overconfident,
        "accuracy_dissociation": accuracy_dissociation,
        "accuracy_spread_relative": float(acc_spread_rel),
        "deltas": {
            "delta_recalib_off_vs_full": {"per_seed": d_recalib, "mean": m_recalib, "sd": sd_recalib,
                                          "significant_positive": bool(recalib_positive),
                                          "significant_negative": bool(recalib_negative)},
            "delta_rem_suppressed_vs_full": {"per_seed": d_rem, "mean": m_rem, "sd": sd_rem,
                                             "significant_positive": bool(rem_positive)},
            "delta_sws_suppressed_vs_full": {"per_seed": d_sws, "mean": m_sws, "sd": sd_sws},
        },
        "cross_arm_spread": float(spread),
        "readiness_ok": readiness_ok,
        "thresholds": {
            "ABS_FLOOR": ABS_FLOOR, "K_SD": K_SD, "NONDEGEN_FLOOR": NONDEGEN_FLOOR,
            "ABS_OVERCONF_MARGIN": ABS_OVERCONF_MARGIN,
            "ACCURACY_DISSOCIATION_TOL": ACCURACY_DISSOCIATION_TOL,
            "RV_LIVE_FLOOR": RV_LIVE_FLOOR, "RECALIB_MOVE_FLOOR": RECALIB_MOVE_FLOOR,
            "PRECISION_INIT_BASELINE": PRECISION_INIT_BASELINE,
        },
    }


def run_experiment(dry_run: bool = False) -> Dict:
    if dry_run:
        n_train, n_eval, n_seeds, steps = 1, 1, 2, 20
    else:
        n_train, n_eval, n_seeds, steps = N_TRAIN_EPS, N_EVAL_EPS, N_SEEDS, STEPS_PER_EP

    rng = np.random.default_rng(42)
    seeds = [int(rng.integers(1000, 9999)) for _ in range(n_seeds)]

    cells: List[Dict] = []
    t0 = time.time()
    t0_perf = time.perf_counter()

    for arm in ARMS:
        arm_label, sws, rem, recalib = arm
        for seed in seeds:
            with arm_cell(
                seed,
                config_slice=_arm_config_slice(sws, rem, recalib),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_arm_seed(arm, seed, n_train, n_eval, steps, dry_run=dry_run)
                cell.stamp(row)
            cells.append(row)

    elapsed = time.time() - t0
    adj = _adjudicate(cells, seeds)

    print("\nResults:", flush=True)
    for arm in adj["arm_overconfidence_score"]:
        print(f"  {arm}: overconf_score={adj['arm_overconfidence_score'][arm]:.4f} "
              f"calib_ratio={adj['arm_calibration_ratio'][arm]:.3f} "
              f"true_err={adj['arm_true_error_ref'][arm]:.6f}", flush=True)
    print(f"  delta_recalib_off_vs_full: mean={adj['deltas']['delta_recalib_off_vs_full']['mean']:.4f} "
          f"sd={adj['deltas']['delta_recalib_off_vs_full']['sd']:.4f}", flush=True)
    print(f"  delta_rem_suppressed_vs_full: mean={adj['deltas']['delta_rem_suppressed_vs_full']['mean']:.4f}", flush=True)
    print(f"  delta_sws_suppressed_vs_full: mean={adj['deltas']['delta_sws_suppressed_vs_full']['mean']:.4f}", flush=True)
    print(f"  suppressed_absolutely_overconfident={adj['suppressed_absolutely_overconfident']} "
          f"accuracy_dissociation={adj['accuracy_dissociation']}", flush=True)
    print(f"  readiness_ok={adj['readiness_ok']} non_degenerate={adj['non_degenerate']}", flush=True)
    print(f"  label={adj['label']} outcome={adj['outcome']}", flush=True)

    return {
        "cells": cells,
        "adjudication": adj,
        "seeds": seeds,
        "elapsed_seconds": elapsed,
        "t0": t0,
        "t0_perf": t0_perf,
        "config_n": {"n_train_eps": n_train, "n_eval_eps": n_eval, "steps_per_ep": steps, "n_seeds": n_seeds},
    }


def main(dry_run: bool = False) -> Dict:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    res = run_experiment(dry_run=dry_run)
    adj = res["adjudication"]
    outcome = adj["outcome"]

    if dry_run:
        print("DRY_RUN_COMPLETE", flush=True)
        return {"outcome": outcome, "manifest_path": None, "run_id": run_id}

    full_config = {
        "grid_size": GRID_SIZE,
        "steps_per_ep": res["config_n"]["steps_per_ep"],
        "n_train_eps": res["config_n"]["n_train_eps"],
        "n_eval_eps": res["config_n"]["n_eval_eps"],
        "n_seeds": res["config_n"]["n_seeds"],
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "arms": [{"arm_id": a[0], "sws_enabled": a[1], "rem_enabled": a[2],
                  "use_rem_precision_recalibration": a[3]} for a in ARMS],
        "env": {"num_hazards": 3, "num_resources": 3, "hazard_harm": 0.04,
                "proximity_harm_scale": 0.12, "proximity_benefit_scale": 0.10,
                "use_proxy_fields": True, "resource_respawn_on_consume": True},
        "seeds": res["seeds"],
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "backlog_id": BACKLOG_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": adj["evidence_direction"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "interpretation": {
            "label": adj["label"],
            "preconditions": adj["preconditions"],
            "criteria": adj["criteria"],
            "criteria_non_degenerate": adj["criteria_non_degenerate"],
        },
        "non_degenerate": adj["non_degenerate"],
        "degeneracy_reason": adj["degeneracy_reason"],
        "aggregates": {
            "arm_overconfidence_score": adj["arm_overconfidence_score"],
            "arm_calibration_ratio": adj["arm_calibration_ratio"],
            "arm_true_error_ref": adj["arm_true_error_ref"],
            "deltas": adj["deltas"],
            "cross_arm_spread": adj["cross_arm_spread"],
            "suppressed_absolutely_overconfident": adj["suppressed_absolutely_overconfident"],
            "accuracy_dissociation": adj["accuracy_dissociation"],
            "accuracy_spread_relative": adj["accuracy_spread_relative"],
            "readiness_ok": adj["readiness_ok"],
        },
        "thresholds": adj["thresholds"],
        "arm_results": res["cells"],
        "elapsed_seconds": res["elapsed_seconds"],
        "notes": (
            "DIAGNOSTIC probe of MECH-173 on the built MECH-204 REM precision-"
            "recalibration consumer (sleep_substrate:GAP-1). Models REM-suppressing "
            "medication as rem_enabled=False (ARM_REM_SUPPRESSED) and isolates the "
            "MECH-123/204 precision-recalibration function via use_rem_precision_"
            "recalibration=False (ARM_RECALIB_OFF); ARM_SWS_SUPPRESSED is the "
            "selectivity control (MECH-174's lane, not MECH-173's). Load-bearing "
            "statistic: overconfidence_index=(true_error_ref-mean_running_variance)/"
            "true_error_ref, where true_error_ref is the mean real forward-model "
            "squared error (residue_metrics['e3_prediction_error'], independent of "
            "the recalibration-modified running_variance). MECH-173 predicts REM/"
            "recalibration suppression raises overconfidence_index vs FULL_SLEEP with "
            "SWS suppression null. P0 readiness gate self-routes substrate_not_ready_"
            "requeue if rv is not live or the F1 recalibration lever never engages; a "
            "sub-threshold cross-arm spread self-routes substrate_ceiling (route "
            "/implement-substrate Phase 7 per 541c). Diagnostic => excluded from "
            "governance confidence/conflict scoring; a PASS is MECH-173 support that a "
            "follow-on may re-run as evidence, not itself a promotion."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=res["seeds"],
        script_path=Path(__file__),
        started_at=res["t0_perf"],
    )
    print(f"Result written to: {out_path}", flush=True)
    return {"outcome": outcome, "manifest_path": str(out_path), "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test (2 seeds, tiny).")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    _outcome = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
    sys.exit(0)
