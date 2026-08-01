"""V3-EXQ-864: SD-076 wci_symmetric_rv_ref/rv WITHIN-EPISODE trajectory diagnostic --
where/when does the inflation_lowers_rv sign flip (found in V3-EXQ-860) happen?

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every
episode) -- UNCHANGED from the 794a/850/853/860 lineage.

THE PUZZLE THIS DIAGNOSES (Finding B, failure_autopsy_V3-EXQ-860_2026-08-01.md section 2).
V3-EXQ-860 measured `inflation_lowers_rv` (= mean over seeds of
wci_symmetric_rv_ref_final - rv_final_after_training) WRONG-SIGNED in BOTH arms at
STEPS_PER_EP=1000 (LO -2.76e-4, HI -1.46e-4, vs a +1e-4 threshold) -- the inflated rv
ended up ABOVE the substrate's own live un-inflated counterfactual, opposite the
intended direction. Siblings V3-EXQ-850/853 (both at STEPS_PER_EP=200) showed small
POSITIVE-but-sub-threshold values for the SAME statistic, never negative. This sign
flip appeared only at 5x longer episodes and is new/undiagnosed.

THIS IS A PURE INSTRUMENTATION/MEASUREMENT DIAGNOSTIC, NOT NEW SD-076/MECH-204
EVIDENCE. It does NOT block the EVB-0454 SD-076 decision (deadline
2026-08-03T20:50:36Z) -- that decision proceeds on V3-EXQ-860's Finding A (the
closure-fraction comparison), which is unaffected by this puzzle (see the autopsy
section 2, "This does not undermine Finding A"). claim_ids=["SD-076"] only, matching
the 860/853/850-h2 convention (Phase 7 broadcast is not engaged; F1/REM recalibration
is on but MECH-204's corrective broadcast mechanism is not, so this diagnoses SD-076's
own EMA dynamics only, not MECH-204).

MECHANISM UNDER INSTRUMENTATION (ree_core/predictors/e3_selector.py). Every tick,
`update_running_variance` (line 626) advances BOTH the inflated path
`_running_variance` (asymmetric alpha: faster when error is improving relative to its
OWN current estimate, slower when worsening) and the counterfactual
`_wci_symmetric_rv_ref` (fixed symmetric `_ema_alpha`, same `error_var`) from the SAME
per-tick prediction error. `_apply_wci_rv_floor` (line 691) then bounds the inflated
path at `waking_confidence_rv_floor_relative_frac * _wci_symmetric_rv_ref` via a soft
floor. Both EMAs are advanced from `post_action_update` (line ~3658) on every
env-step's `select_action` -> `post_action_update` call, once per training tick --
confirmed by direct source read 2026-08-01, unchanged from 860's own re-verification.
The F1/REM precision-recalibration WRITEBACK (MECH-204, unrelated broadcast mechanism)
nudges `_running_variance` once per EPISODE BOUNDARY (agent.reset(), K=1). HYPOTHESIS:
at 5x more within-episode ticks before that once-per-episode correction, the inflated
and counterfactual EMAs may diverge in a way invisible at the original 200-step episode
length. This script does not test that hypothesis -- it INSTRUMENTS the trajectory so a
human/autopsy can see where/when (if at all) the diff = ref - rv crosses zero.

DESIGN: sweep STEPS_PER_EP in {200, 500, 1000} -- the three points that bracket the
observed transition (200 in the 850/853 lineage: small positive/sub-threshold; 1000 in
860: negative). Log (rv, wci_symmetric_rv_ref, diff=ref-rv) at ~40 evenly-spaced ticks
WITHIN EVERY episode (not just the end-of-training snapshot 860/853/850 recorded), so
both the within-episode trajectory and the across-episode trend are visible. Same
substrate operating point and same two inflation doses (ARM_INFL_LO=0.6,
ARM_INFL_HI=0.8) as 860, so a direct comparison against 860's own end-of-training
readout is possible as a replication sanity check.

CHEAPER THAN 860 ON PURPOSE (queue-experiment skill Step 2.4 GOV-REUSE-1 + "small,
cheap" instruction). N_TRAIN_EPS reduced from 860's 30 to 8 -- this diagnostic only
needs to see a handful of F1-recalibration cycles establish the within-episode
divergence pattern, not full training-scale statistical power (the closure-fraction
comparison, which DOES need that scale, was already answered decisively by 860's
Finding A). No eval phase at all -- eval never touches the training-phase EMA dynamics
this diagnostic exists to see (860's own module docstring: "the decisive readout
rv_final is read at the END of training, before eval begins"). N_SEEDS=2 (light
replication, not full statistical power -- this is exploratory instrumentation, not a
hypothesis test). Total train-only steps: (200+500+1000) * 8 episodes * 2 arms * 2
seeds = 54400, versus 860's 180000 train-only steps (~30%) with zero eval overhead on
top -- a materially cheaper run.

GOV-REUSE-1 CHECK (queue-experiment Step 2.4). Decisive readout: the WITHIN-EPISODE
trajectory of (rv, wci_symmetric_rv_ref, diff) at multiple STEPS_PER_EP values -- not
recorded by any manifest today. Checked via
REE_assembly/scripts/reanalysis_query.py query --readout rv_trajectory_within_episode
--claim SD-076 (2026-08-01): 5 manifests scanned (794/794a/850-h1/850-h2/860), 0 carry
any within-episode trajectory readout -- every one of them records only the
end-of-training snapshot (rv_final_after_training /
wci_symmetric_rv_ref_after_training), matching 860's own module docstring
confirmation ("NONE of the four vary STEPS_PER_EP... every recorded run held it at
200") and additionally confirming none record *any* intra-episode granularity at any
STEPS_PER_EP. Not recoverable -> run.

SUBSTRATE READINESS (queue-experiment Step 2.5). Identical substrate surface to
V3-EXQ-860, re-confirmed by direct source read 2026-08-01: `E3TrajectorySelector.
_running_variance` / `._wci_symmetric_rv_ref` (e3_selector.py:302,314),
`update_running_variance` (e3_selector.py:626, called from `post_action_update` at
line ~3658 on every training tick), `agent.sleep_loop` (agent.py), the MECH-204
writeback keys (`mech204_recalibration_fired` / `mech204_running_variance_{before,
after}`, phase_manager.py), and the `use_waking_confidence_inflation` /
`waking_confidence_inflation_asymmetry` / `waking_confidence_rv_floor*` REEConfig.e3
fields (utils/config.py) all confirmed present and unchanged. This script manipulates
NEITHER implementation, only STEPS_PER_EP and the sampling instrumentation -- no new
substrate build needed. claims.yaml: SD-076 is `candidate` / `epistemic_category:
standard`, no additional gate beyond what 860/853/850 already cleared.

RE-DERIVE BRAKE (queue-experiment Step 2.5b). This is not a re-test of the H1/H2/H3
discrimination at all (no dose-response criterion, no closure-fraction readout, no
substrate verdict) -- it is pure instrument/measurement diagnosis of a NEW puzzle
(the wrong-signed precondition) that only surfaced in 860 itself. 0 prior
`substrate_ceiling` autopsies exist for SD-076 (860's own re_derive_brake block:
fired=false, prior_substrate_ceiling_autopsies=[]). Brake does not fire.

CLAIM TAGGING (queue-experiment Step 3 "claim_ids accuracy rule"). CLAIM_IDS =
["SD-076"] only -- identical reasoning to 860/853/850-h2: Phase 7 broadcast
(`use_rem_precision_broadcast`) is OFF in every arm, so this produces no MECH-204
evidence. F1/REM recalibration is held ON/unmanipulated (same as 860), so its
presence yields no attributable evidence either way. This diagnostic characterizes
SD-076's own inflation-vs-counterfactual EMA dynamics, nothing else.

EXPERIMENT_PURPOSE = "diagnostic": pure instrumentation/measurement investigation of
an undiagnosed sign-flip puzzle, explicitly excluded from confidence/conflict scoring.
Does NOT feed a PASS/FAIL hypothesis about SD-076/MECH-204 -- the "verdict" this
script computes is about DATA QUALITY (did the instrumentation capture usable,
non-degenerate trajectories across the sweep), not about a claim.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per-arm; queue-experiment Step 3). The
decisive readouts are (a) diff_final = wci_symmetric_rv_ref_after_training -
rv_final_after_training, a scalar EMA-level snapshot with no permutation symmetry
(same declaration as 860, unchanged: the asymmetric EMA's settling level changes with
`inflation_asymmetry` and with the number of update ticks before the snapshot -- NOT
invariant under either manipulated axis, STEPS_PER_EP or arm dose), and (b) the
per-tick trajectory of diff = ref - rv within an episode, which is a raw time series,
not a set-aggregate -- there is no permutation/rescaling symmetry to be invariant
under; it is read directly, not computed via any aggregate statistic that a
manipulation could leave invariant.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-864"
EXPERIMENT_TYPE = "v3_exq_864_sd076_wci_rv_trajectory_crossover_diagnostic"
CLAIM_IDS = ["SD-076"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = (
    "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"
)
FANOUT_SOURCE = "failure_autopsy_V3-EXQ-860_2026-08-01.json"
PUZZLE_SOURCE_RUN_ID = "v3_exq_860_mech204_sd076_h2_steps_per_ep_probe_20260801T142420Z_v3"
PUZZLE_SOURCE_QUEUE_ID = "V3-EXQ-860"

# ---- Run shape. STEPS_PER_EP is swept; N_TRAIN_EPS is fixed (unconfounded across the
# sweep) and REDUCED from 860's 30 -- see module docstring "CHEAPER THAN 860". ----
N_TRAIN_EPS = 8            # enough F1 firings (8) to see the within-episode pattern
                            # settle across a few recalibration cycles; not intended to
                            # match 860's full-scale statistical power.
STEPS_PER_EP_SWEEP: Tuple[int, ...] = (200, 500, 1000)   # brackets 860's transition
N_SEEDS = 2
GRID_SIZE = 12
LR = 5e-4

# ---- Substrate operating point (held constant, identical to 794a/850/853/860). ----
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# ---- The two inflation doses, identical to 860. ----
INFLATION_ASYMMETRY_LO = 0.6
INFLATION_ASYMMETRY_HI = 0.8
INFLATION_RV_FLOOR = 0.01
INFLATION_RV_FLOOR_RELATIVE_FRAC = 0.2
INFLATION_RV_FLOOR_MODE = "soft"
INFLATION_RV_FLOOR_SOFTNESS = 0.25

# ---- Pre-registered thresholds (NOT derived from this run's own statistics). Same
# floors as 794a/850/853/860 for the two readiness preconditions this run inherits. ----
PRECISION_INIT_BASELINE = 0.5
RV_LIVE_FLOOR = 1e-6
RECALIB_MOVE_FLOOR = 1e-4
# This diagnostic's own non-degeneracy floor: the trajectory must show REAL EMA
# movement (diff varies across ticks), not a frozen/constant reading. Pre-registered,
# not derived from this run's own data -- a diff that never moves by more than this
# across an entire cell's trajectory means the instrumentation captured nothing.
TRAJECTORY_VARIATION_FLOOR = 1e-6

# Within-episode sampling: ~40 samples per episode, regardless of STEPS_PER_EP, so the
# three swept lengths are visually/comparably resolved rather than one being sparse.
SAMPLES_PER_EPISODE_TARGET = 40


def _sample_interval(steps_per_ep: int) -> int:
    return max(1, steps_per_ep // SAMPLES_PER_EPISODE_TARGET)


# (steps_per_ep, arm_id, asymmetry) triples -- the full sweep grid.
ARMS: Tuple[Tuple[str, float], ...] = (
    ("ARM_INFL_LO", INFLATION_ASYMMETRY_LO),
    ("ARM_INFL_HI", INFLATION_ASYMMETRY_HI),
)
CELLS: Tuple[Tuple[int, str, float], ...] = tuple(
    (steps, arm_id, asym)
    for steps in STEPS_PER_EP_SWEEP
    for (arm_id, asym) in ARMS
)

_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------- preconditions --
# Readiness-only (this diagnostic makes no claim-scoring verdict). Both inherited
# unmodified from the 860/853/850 lineage's own readiness gate, plus this script's own
# trajectory non-degeneracy check. Applies per (steps_per_ep, arm) cell, aggregated
# across its 2 seeds.
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="rv_live",
        description="rv_final differs from precision_init by more than the floor "
                    "(Q-042/530c substrate-liveness contract). Worst seed reported.",
        control="every seed of this cell; a dead rv makes the trajectory meaningless",
        threshold=RV_LIVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="f1_recalib_engaged",
        description="mean per-cycle |rv_after - rv_before| from the F1 WRITEBACK "
                    "recalibration exceeds the floor -- confirms the reduced "
                    "N_TRAIN_EPS=8 operating point still exercises the same "
                    "substrate mechanism 860 measured at N_TRAIN_EPS=30.",
        control="F1 recalibration is ON in every cell of this design",
        threshold=RECALIB_MOVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="trajectory_non_degenerate",
        description="max(diff) - min(diff) across this cell's sampled within-episode "
                    "trajectory exceeds the floor -- the instrumentation captured "
                    "real EMA movement, not a frozen/constant reading. Worst seed "
                    "reported.",
        control="the sampled diff series itself; a constant reading anywhere near "
                "zero variation means nothing was actually observed",
        threshold=TRAJECTORY_VARIATION_FLOOR,
        direction="lower",
    ),
)


def _cell_ctx(steps: int, arm_id: str, asym: float) -> Dict[str, object]:
    return {"id": f"{arm_id}::steps{steps}", "steps_per_ep": steps,
            "arm_id": arm_id, "asymmetry": asym}


CELL_CONTEXTS = [_cell_ctx(s, a, x) for (s, a, x) in CELLS]


# ------------------------------------------------------------------ build helpers --
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


def _make_agent(env: CausalGridWorldV2, asym: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_sleep_loop=True,
        sleep_loop_episodes_K=1,
        use_rem_precision_recalibration=True,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=REM_PRECISION_RECALIBRATION_STEP,
        use_rem_precision_broadcast=False,
        rem_precision_broadcast_gain=0.0,
    )
    cfg.e3.use_waking_confidence_inflation = True
    cfg.e3.waking_confidence_inflation_asymmetry = float(asym)
    cfg.e3.waking_confidence_rv_floor = INFLATION_RV_FLOOR
    cfg.e3.waking_confidence_rv_floor_relative_frac = INFLATION_RV_FLOOR_RELATIVE_FRAC
    cfg.e3.waking_confidence_rv_floor_mode = INFLATION_RV_FLOOR_MODE
    cfg.e3.waking_confidence_rv_floor_softness = INFLATION_RV_FLOOR_SOFTNESS
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _arm_config_slice(asym: float, steps_per_ep: int, n_train: int) -> Dict:
    return {
        "grid_size": GRID_SIZE,
        "train_steps_per_ep": steps_per_ep,
        "n_train_eps": n_train,
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_enabled": True,
        "rem_enabled": True,
        "use_rem_precision_recalibration": True,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "use_rem_precision_broadcast": False,
        "rem_precision_broadcast_gain": 0.0,
        "use_waking_confidence_inflation": True,
        "waking_confidence_inflation_asymmetry": float(asym),
        "waking_confidence_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "precision_init_baseline": PRECISION_INIT_BASELINE,
        "sample_interval": _sample_interval(steps_per_ep),
        "no_eval_phase": True,
        "trajectory_variation_floor": TRAJECTORY_VARIATION_FLOOR,
    }


def _read_recalib_metrics(agent: REEAgent) -> Optional[Dict[str, float]]:
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


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------- one cell --
def _run_cell(steps_per_ep: int, arm_id: str, asym: float, seed: int,
              n_train: int, dry_run: bool = False) -> Dict:
    interval = _sample_interval(steps_per_ep)

    with arm_cell(
        seed,
        config_slice=_arm_config_slice(asym, steps_per_ep, n_train),
        script_path=Path(__file__),
        include_driver_script_in_hash=False,  # mint-as-you-go: cross-driver reusable
    ) as cell:
        env = _make_env(seed, dry_run=dry_run)
        agent = _make_agent(env, asym)
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        print(f"Seed {seed} Condition {arm_id}_steps{steps_per_ep} n_train={n_train}",
              flush=True)

        trajectory: List[Dict[str, float]] = []
        recalib_moves: List[float] = []
        recalib_fired = 0
        global_tick = 0
        train_harness = StepHarness(agent, env, train_mode=True, seed=seed)
        for ep in range(n_train):
            agent.reset()  # fires the sleep cycle for the prior episode (K=1)
            rec = _read_recalib_metrics(agent)
            if rec is not None:
                if rec.get("fired", 0.0) > 0.0:
                    recalib_fired += 1
                if "rv_before" in rec and "rv_after" in rec:
                    recalib_moves.append(abs(rec["rv_after"] - rec["rv_before"]))
            _, obs_dict = env.reset()
            train_harness.reset()
            for t in range(steps_per_ep):
                result = train_harness.step(obs_dict)
                optimizer.zero_grad()
                loss = agent.compute_prediction_loss()
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                obs_dict = result.next_obs_dict
                if t % interval == 0 or t == steps_per_ep - 1:
                    rv = float(agent.e3._running_variance)
                    ref = float(agent.e3._wci_symmetric_rv_ref)
                    trajectory.append({
                        "episode_idx": ep,
                        "tick_in_episode": t,
                        "global_tick": global_tick,
                        "rv": rv,
                        "wci_symmetric_rv_ref": ref,
                        "diff": ref - rv,
                    })
                global_tick += 1
                if result.done:
                    break
            if (ep + 1) % 2 == 0 or ep + 1 == n_train:
                print(
                    f"  [train] arm={arm_id} steps_per_ep={steps_per_ep} seed={seed} "
                    f"ep {ep + 1}/{n_train} "
                    f"rv={float(agent.e3._running_variance):.6f} "
                    f"ref={float(agent.e3._wci_symmetric_rv_ref):.6f} "
                    f"diff={float(agent.e3._wci_symmetric_rv_ref) - float(agent.e3._running_variance):+.6f}",
                    flush=True,
                )

        rv_final = float(agent.e3._running_variance)
        ref_final = float(agent.e3._wci_symmetric_rv_ref)
        diff_final = ref_final - rv_final

        diff_series = [row["diff"] for row in trajectory]
        traj_variation = (max(diff_series) - min(diff_series)) if diff_series else 0.0

        # Find the first sampled tick (in global ordering) where the sign of diff
        # flips relative to the FIRST sampled tick's sign, if any -- a coarse,
        # descriptive crossover locator, not a statistical test.
        crossover_global_tick = None
        if diff_series:
            first_sign = diff_series[0] >= 0.0
            for row in trajectory:
                if (row["diff"] >= 0.0) != first_sign:
                    crossover_global_tick = row["global_tick"]
                    break

        _moved_from_first_sample = bool(
            diff_series and (diff_series[-1] >= 0.0) != (diff_series[0] >= 0.0))
        print(
            f"  [end]  arm={arm_id} steps_per_ep={steps_per_ep} seed={seed} "
            f"rv_final={rv_final:.6f} ref_final={ref_final:.6f} "
            f"diff_final={diff_final:+.6f} "
            f"crossover_seen={'Y' if _moved_from_first_sample else 'N'}",
            flush=True,
        )
        print(f"verdict: {'PASS' if traj_variation > TRAJECTORY_VARIATION_FLOOR else 'FAIL'}",
              flush=True)

        _ZG.observe(agent)

        row = {
            "steps_per_ep": steps_per_ep,
            "arm_id": arm_id,
            "seed": seed,
            "inflation_asymmetry": float(asym),
            "n_train_eps": n_train,
            "sample_interval": interval,
            "rv_final_after_training": rv_final,
            "wci_symmetric_rv_ref_after_training": ref_final,
            "diff_final": diff_final,
            "rv_delta_from_precision_init": abs(rv_final - PRECISION_INIT_BASELINE),
            "recalib_cycles_fired": recalib_fired,
            "recalib_mean_abs_move": _mean(recalib_moves),
            "trajectory_variation": traj_variation,
            "trajectory_n_samples": len(trajectory),
            "sign_flip_seen": _moved_from_first_sample,
            "crossover_global_tick": crossover_global_tick,
            "trajectory": trajectory,
        }
        cell.stamp(row)
    return row


# ---------------------------------------------------------------------- analysis --
def _analyse(cells: List[Dict], seeds: List[int]) -> Dict:
    by_cell: Dict[Tuple[int, str], Dict[int, Dict]] = {}
    for c in cells:
        by_cell.setdefault((c["steps_per_ep"], c["arm_id"]), {})[c["seed"]] = c

    # ---- readiness gates, one arm-gate per (steps_per_ep, arm_id) grid cell ----
    arm_gates = []
    for (steps, arm_id, asym) in CELLS:
        ctx = _cell_ctx(steps, arm_id, asym)
        rows = by_cell[(steps, arm_id)]
        measured: Dict[str, float] = {
            "rv_live": min(rows[s]["rv_delta_from_precision_init"] for s in seeds),
            "f1_recalib_engaged": _mean([rows[s]["recalib_mean_abs_move"] for s in seeds]),
            "trajectory_non_degenerate": min(rows[s]["trajectory_variation"] for s in seeds),
        }
        gate_id = f"{arm_id}::steps{steps}"
        arm_gates.append(
            evaluate_arm_gate(gate_id, ctx, list(PRECONDITION_SPECS), measured))

    gate = aggregate_arm_gates(arm_gates)

    # ---- per-cell descriptive summary (mean diff_final over seeds, and whether ANY
    # seed in this cell saw a within-episode sign flip) ----
    per_cell: Dict[str, Dict] = {}
    for (steps, arm_id, asym) in CELLS:
        rows = by_cell[(steps, arm_id)]
        diff_final_mean = _mean([rows[s]["diff_final"] for s in seeds])
        per_cell[f"{arm_id}_steps{steps}"] = {
            "steps_per_ep": steps,
            "arm_id": arm_id,
            "asymmetry": asym,
            "diff_final_mean": diff_final_mean,
            "diff_final_sign": "positive" if diff_final_mean >= 0.0 else "negative",
            "any_seed_sign_flip": any(rows[s]["sign_flip_seen"] for s in seeds),
            "n_seeds": len(rows),
        }

    # ---- per-arm crossover bracket across the STEPS_PER_EP sweep (descriptive) ----
    per_arm_bracket: Dict[str, Dict] = {}
    for (arm_id, asym) in ARMS:
        ordered = [per_cell[f"{arm_id}_steps{s}"] for s in STEPS_PER_EP_SWEEP]
        signs = [c["diff_final_sign"] for c in ordered]
        bracket = None
        for i in range(len(ordered) - 1):
            if signs[i] != signs[i + 1]:
                bracket = (STEPS_PER_EP_SWEEP[i], STEPS_PER_EP_SWEEP[i + 1])
                break
        per_arm_bracket[arm_id] = {
            "steps_per_ep_swept": list(STEPS_PER_EP_SWEEP),
            "diff_final_by_steps": {
                str(s): per_cell[f"{arm_id}_steps{s}"]["diff_final_mean"]
                for s in STEPS_PER_EP_SWEEP
            },
            "sign_sequence": signs,
            "crossover_bracket_steps_per_ep": list(bracket) if bracket else None,
        }

    # C0 (load-bearing, diagnostic-adjudication sense): the instrumentation captured
    # real, non-degenerate trajectories across the WHOLE sweep -- i.e. this run's data
    # is trustworthy to read at all. Independent of whether a crossover was found.
    c0_data_usable = bool(gate["all_green"])
    # C1 (non-load-bearing, descriptive): at least one arm's sweep brackets a sign
    # change within {200, 500, 1000} -- i.e. this run's chosen sweep actually located
    # the crossover, rather than only bracketing it outside the tested range.
    c1_crossover_bracketed = any(
        per_arm_bracket[a]["crossover_bracket_steps_per_ep"] is not None
        for (a, _) in ARMS
    )

    criteria = [
        {"name": "C0_trajectory_data_usable", "load_bearing": True,
         "passed": c0_data_usable},
        {"name": "C1_crossover_bracketed_in_swept_range", "load_bearing": False,
         "passed": c1_crossover_bracketed,
         "per_arm_bracket": per_arm_bracket},
    ]

    # C0/C1 both read ALL SIX cells jointly (the whole sweep), so they are only
    # non-degenerate if EVERY cell's gate is green -- a criterion owned by every
    # cell is degenerate if any is red. arm_criteria_non_degenerate keys per-arm
    # (here per-cell), so intersect by hand via both_green first (mirrors 860's own
    # "C1/C2/C3 all read BOTH arms jointly" comment) and pass a single representative
    # gate_id -- its own green/red status is redundant once raw_non_degen already
    # encodes the full intersection, exactly as in the 860 precedent.
    both_green = bool(gate["all_green"])
    raw_non_degen = {
        "C0_trajectory_data_usable": both_green,
        "C1_crossover_bracketed_in_swept_range": both_green,
    }
    _representative_gate_id = arm_gates[0]["arm"] if arm_gates else "?"
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {_representative_gate_id: list(raw_non_degen.keys())}, gate, raw_non_degen)

    readiness_ok = bool(gate["non_degenerate"]) and both_green
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif c1_crossover_bracketed:
        label = "wci_rv_crossover_bracketed_in_swept_range"
        outcome = "PASS"
    else:
        label = "wci_rv_trajectory_characterized_no_crossover_in_range"
        outcome = "PASS"

    per_claim = {"SD-076": "unknown"}  # instrumentation only -- see module docstring

    return {
        "outcome": outcome,
        "label": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": per_claim,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "gate": gate,
        "arm_gates": arm_gates,
        "per_cell": per_cell,
        "per_arm_bracket": per_arm_bracket,
        "readiness_ok": readiness_ok,
        "thresholds": {
            "RV_LIVE_FLOOR": RV_LIVE_FLOOR,
            "RECALIB_MOVE_FLOOR": RECALIB_MOVE_FLOOR,
            "TRAJECTORY_VARIATION_FLOOR": TRAJECTORY_VARIATION_FLOOR,
            "PRECISION_INIT_BASELINE": PRECISION_INIT_BASELINE,
        },
        "reference_values": {
            "v3_exq_860_inflation_lowers_rv": {"LO": -2.76e-4, "HI": -1.46e-4},
            "v3_exq_850_853_inflation_lowers_rv_context": "small positive, sub-threshold (exact values in those manifests)",
        },
    }


# -------------------------------------------------------------------------- main --
def run_experiment(dry_run: bool = False) -> Dict:
    t0 = time.perf_counter()
    n_train = 3 if dry_run else N_TRAIN_EPS
    n_seeds = 2 if dry_run else N_SEEDS
    steps_sweep = (20, 40) if dry_run else STEPS_PER_EP_SWEEP
    seeds = list(range(n_seeds))

    if not dry_run:
        assert N_TRAIN_EPS >= 4, (
            f"Need at least a few F1-recalibration cycles to see the within-episode "
            f"pattern settle; got N_TRAIN_EPS={N_TRAIN_EPS}"
        )
        assert list(STEPS_PER_EP_SWEEP) == sorted(STEPS_PER_EP_SWEEP), (
            "STEPS_PER_EP_SWEEP must be ascending for the crossover-bracket logic"
        )

    cells_ctx = [
        (steps, arm_id, asym)
        for steps in steps_sweep
        for (arm_id, asym) in ARMS
    ] if dry_run else list(CELLS)

    assert_no_structurally_unsatisfiable_gate(list(PRECONDITION_SPECS), [
        _cell_ctx(s, a, x) for (s, a, x) in cells_ctx
    ])

    cells: List[Dict] = []
    for (steps, arm_id, asym) in cells_ctx:
        for seed in seeds:
            cells.append(_run_cell(steps, arm_id, asym, seed, n_train, dry_run=dry_run))

    if dry_run:
        # Dry-run uses a different (smaller) steps sweep, so the full-sweep bracket
        # analysis (keyed to STEPS_PER_EP_SWEEP) doesn't apply -- just confirm cells ran.
        adj = {
            "outcome": "PASS" if all(c["trajectory_variation"] >= 0.0 for c in cells) else "FAIL",
            "label": "dry_run_smoke",
            "criteria": [],
            "readiness_ok": True,
        }
    else:
        adj = _analyse(cells, seeds)
    adj["cells"] = cells
    adj["seeds"] = seeds
    adj["elapsed_seconds"] = time.perf_counter() - t0
    adj["t0_perf"] = t0
    adj["config_n"] = {"n_train_eps": n_train, "n_seeds": n_seeds,
                       "steps_per_ep_sweep": list(steps_sweep)}
    return adj


def main(dry_run: bool = False) -> Dict:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    adj = run_experiment(dry_run=dry_run)
    outcome = adj["outcome"]

    print("", flush=True)
    print(f"label={adj['label']} outcome={outcome} readiness_ok={adj.get('readiness_ok')}",
          flush=True)
    if not dry_run:
        for arm_id, bracket in adj["per_arm_bracket"].items():
            print(f"  {arm_id}: diff_final_by_steps={bracket['diff_final_by_steps']} "
                  f"crossover_bracket={bracket['crossover_bracket_steps_per_ep']}",
                  flush=True)
        for c in adj["criteria"]:
            lb = " (load-bearing)" if c["load_bearing"] else ""
            print(f"  {c['name']}: {'PASS' if c['passed'] else 'FAIL'}{lb}", flush=True)

    if dry_run:
        print("DRY_RUN_COMPLETE", flush=True)
        return {"outcome": outcome, "manifest_path": None, "run_id": run_id}

    full_config = {
        "grid_size": GRID_SIZE,
        "n_train_eps": adj["config_n"]["n_train_eps"],
        "n_train_eps_v3_exq_860_baseline": 30,
        "steps_per_ep_sweep": adj["config_n"]["steps_per_ep_sweep"],
        "n_seeds": adj["config_n"]["n_seeds"],
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "use_rem_precision_broadcast": False,
        "inflation_asymmetry_lo": INFLATION_ASYMMETRY_LO,
        "inflation_asymmetry_hi": INFLATION_ASYMMETRY_HI,
        "inflation_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "samples_per_episode_target": SAMPLES_PER_EPISODE_TARGET,
        "arms": [{"arm_id": a[0], "inflation_asymmetry": float(a[1])} for a in ARMS],
        "env": {"num_hazards": 3, "num_resources": 3, "hazard_harm": 0.04,
                "proximity_harm_scale": 0.12, "proximity_benefit_scale": 0.10,
                "use_proxy_fields": True, "resource_respawn_on_consume": True},
        "seeds": adj["seeds"],
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": adj["evidence_direction"],
        "evidence_direction_per_claim": adj["evidence_direction_per_claim"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "fanout_source": FANOUT_SOURCE,
        "puzzle_source_run_id": PUZZLE_SOURCE_RUN_ID,
        "puzzle_source_queue_id": PUZZLE_SOURCE_QUEUE_ID,
        "interpretation": {
            "label": adj["label"],
            "preconditions": adj["gate"]["adjudication_preconditions"],
            "criteria": adj["criteria"],
            "criteria_non_degenerate": adj["criteria_non_degenerate"],
        },
        "per_arm_gate": adj["gate"]["per_arm_gate"],
        "non_degenerate": adj["gate"]["non_degenerate"],
        "degeneracy_reason": adj["gate"]["degeneracy_reason"],
        "aggregates": {
            "per_cell": adj["per_cell"],
            "per_arm_bracket": adj["per_arm_bracket"],
            "readiness_ok": adj["readiness_ok"],
        },
        "thresholds": adj["thresholds"],
        "reference_values": adj["reference_values"],
        "arm_results": adj["cells"],
        "per_seed_cells": adj["cells"],
        "elapsed_seconds": adj["elapsed_seconds"],
        "notes": (
            "PURE INSTRUMENTATION/MEASUREMENT DIAGNOSTIC (experiment_purpose=diagnostic, "
            "excluded from confidence/conflict scoring) investigating Finding B of "
            "failure_autopsy_V3-EXQ-860_2026-08-01.md: the wrong-signed "
            "inflation_lowers_rv precondition (rv ended up ABOVE the un-inflated "
            "counterfactual at STEPS_PER_EP=1000, opposite the intended direction, "
            "vs small-positive-but-sub-threshold at STEPS_PER_EP=200 in the 850/853 "
            "sibling runs). This run sweeps STEPS_PER_EP in {200, 500, 1000} at a "
            "REDUCED N_TRAIN_EPS=8 (vs 860's 30 -- cheaper on purpose, sufficient to "
            "see a few F1-recalibration cycles) and logs the (rv, "
            "wci_symmetric_rv_ref, diff) trajectory at ~40 samples per episode within "
            "EVERY training episode, not just the end-of-training snapshot 860/853/850 "
            "recorded. NOT a re-test of the H1/H2/H3 discrimination and NOT new "
            "SD-076/MECH-204 evidence -- claim_ids=[SD-076] reflects only that this "
            "characterizes SD-076's own EMA dynamics. DOES NOT BLOCK the EVB-0454 "
            "SD-076 decision (deadline 2026-08-03T20:50:36Z): that decision proceeds "
            "on V3-EXQ-860's Finding A (the closure-fraction comparison), which this "
            "diagnostic does not touch. See aggregates.per_arm_bracket for the "
            "STEPS_PER_EP interval (if any) in which each arm's diff_final crossed "
            "sign, and arm_results[*].trajectory for the raw within-episode series."
        ),
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=adj["seeds"],
        script_path=Path(__file__),
        started_at=adj["t0_perf"],
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Result written to: {out_path}", flush=True)
    return {"outcome": outcome, "manifest_path": str(out_path), "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test (tiny).")
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
