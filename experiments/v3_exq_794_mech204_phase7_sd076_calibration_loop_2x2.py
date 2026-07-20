"""V3-EXQ-794: MECH-204 Phase 7 (broadcast precision anchor) x SD-076 (waking
confidence inflation) -- 2x2 substrate-readiness validation. Successor to V3-EXQ-774.

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

WHY A 2x2 AND NOT A PHASE-7 ABLATION
------------------------------------
V3-EXQ-774 FAILed and `failure_autopsy_V3-EXQ-774_2026-07-17` (confirmed) adjudicated
`substrate_ceiling`: the built MECH-204 F1 consumer "cannot express absolute
overconfidence". The root cause is that E3.update_running_variance maintained
`_running_variance` (rv) as a SYMMETRIC EMA of true prediction error, so

    rv ~= true prediction error   BY CONSTRUCTION

and the load-bearing statistic

    overconfidence_score = log(true_error_ref / mean_rv)

is therefore pinned near zero NO MATTER WHAT IS ABLATED. 774 measured -0.000148
(ARM_REM_SUPPRESSED) and -0.000918 (ARM_RECALIB_OFF) -- both indistinguishable from
zero -- and its `suppressed_arm_absolutely_overconfident` criterion recorded False.

That is a TAUTOLOGY, not a null. MECH-204's corrective function presupposes a daytime
precision-drift source the V3 substrate did not have. So an ablation of Phase 7 ALONE
would retest to an identical near-zero result, and that null would read as a Phase-7
refutation when in fact nothing was measured. Both factors must be ablated together:

  A: MECH-204 Phase 7 / Option B  -- REEConfig.use_rem_precision_broadcast (the CORRECTION)
  B: SD-076 waking confidence inflation -- E3Config.use_waking_confidence_inflation (the SOURCE)

ARMS (2x3 factorial: broadcast {OFF,ON} x asymmetry {OFF, LO=0.6, HI=0.8}; all arms run
FULL sleep with F1 recalibration ON, because Phase 7 runs ALONGSIDE F1 and its read is a
no-op until a REM entry has populated serotonin._persistent_zero_point):

  ARM_OFF_OFF     bcast=F asym=OFF -- reproduces 774's ARM_FULL_SLEEP condition.
                                      Replication anchor + the tautology control.
  ARM_BCAST_ONLY  bcast=T asym=OFF -- correction with nothing to correct.
  ARM_INFL_LO     bcast=F asym=0.6 -- drift, no correction.
  ARM_BOTH_LO     bcast=T asym=0.6 -- drift plus correction.
  ARM_INFL_HI     bcast=F asym=0.8 -- stronger drift, no correction.
  ARM_BOTH_HI     bcast=T asym=0.8 -- stronger drift plus correction.

WHY THREE ASYMMETRY LEVELS AND NOT TWO. C1 is the load-bearing capability question
("can the substrate express absolute overconfidence at all?"), and clearing its +0.10
margin from 774's -0.2097 baseline needs a swing > 0.31 log units. A single
pre-registered asymmetry of 0.6 delivered ~0.34-0.44 in smoke -- too close to call, so a
C1 failure would have been uninterpretable (weak lever setting, or wrong mechanism?).
The HI level makes that distinguishable. C1's OPERATIVE LEVEL is the LOWEST asymmetry
that clears, so a result holding at LO is never attributed to HI; C2 is then evaluated
at that operative level, because a C2 measured where C1 never cleared is meaningless.
HI=0.8 is a 9:1 good-vs-bad-news update ratio -- at the EDGE of the defensible
optimism-bias band and a capability probe, NOT a proposed default. Consequently, if C1
fails at BOTH levels the route is NOT "sweep higher": it is that the asymmetric-EMA form
is the wrong drift source and a different SD-076 mechanism is owed.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per-arm; /queue-experiment Step 3)
-------------------------------------------------------------------------------
DV = overconfidence_score = log(true_error_ref / mean_rv), a scalar function of
`_running_variance` and the independently-measured forward-model error. Its symmetry
group: it is invariant under PERMUTATION of the eval ticks (both terms are means over
ticks). It is NOT invariant under any change to rv's LEVEL.

  ARM_BCAST_ONLY / ARM_BOTH_LO / ARM_BOTH_HI (factor A): broadcast_precision_pull writes
      rv directly (interpolates it toward the F1 cumulative zero-point). A level change,
      not a permutation. NOT invariant. OK
  ARM_INFL_LO / ARM_INFL_HI / ARM_BOTH_LO / ARM_BOTH_HI (factor B): the asymmetric EMA
      changes rv's update rule and hence its settling level. A level change, not a
      permutation. NOT invariant. OK
  ARM_OFF_OFF: no manipulation; it is the control the other five are read against.

Both manipulations write the SAME scalar the DV reads, which is exactly why the Phase 7
write-site was corrected from E3 score to precision space before this run was designed.
WORTH RECORDING: had Phase 7 been built at the 2026-05-09 spec'd site (additive bias on
the E3 score), factor A WOULD have been invariant under this DV -- a broadcast scalar
added uniformly across candidates cannot move an argmax or a softmax sample, so it would
have perturbed no selection, hence no rv trajectory, and the measured delta would have
been an arithmetic zero fixed before the run (the V3-EXQ-604c class). See
sleep_substrate_plan.md decision log 2026-07-20.

PURPOSE / SCOPE
---------------
DIAGNOSTIC substrate-readiness validation, NOT governance evidence. It asks whether the
substrate can now (a) express absolute overconfidence at all and (b) correct it via the
broadcast anchor. It deliberately does NOT tag MECH-173: MECH-173 is about REM-SUPPRESSION
raising overconfidence, which needs the rem_enabled manipulation this design does not run.
A PASS here makes a MECH-173 retest MEANINGFUL; it is not itself MECH-173 support.

GOV-REUSE-1: the decisive readout (overconfidence_score) is recorded only in V3-EXQ-774
(substrate_hash fff0845d547b05a7), on a substrate with neither Phase 7 nor SD-076 --
a different substrate_hash, so its value cannot answer this question. Not recoverable ->
run. Checked run_ids: v3_exq_774_mech173_rem_suppression_precision_calibration_20260717T152554Z_v3.
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-794"
EXPERIMENT_TYPE = "v3_exq_794_mech204_phase7_sd076_calibration_loop_2x2"
CLAIM_IDS = ["MECH-204", "SD-076"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = (
    "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"
)
SUPERSEDES = "v3_exq_774_mech173_rem_suppression_precision_calibration"

# ---- Run shape (mirrors 774 so the OFF_OFF arm is comparable to its ARM_FULL_SLEEP) ----
N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_SEEDS = 3
GRID_SIZE = 12
STEPS_PER_EP = 200
LR = 5e-4

# ---- Substrate operating point (held constant across arms) ----
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# ---- The two factors under test ----
# Phase 7 gain is PER WAKING STEP, so it must be far below the per-CYCLE F1 step (0.25).
BROADCAST_GAIN = 0.01
# SD-076 asymmetry levels. LO=0.6 is the substrate unit-smoke value (index -0.164 ->
# +0.273); HI=0.8 is the DE-RISKING level added because clearing C1's +0.10 margin from
# 774's -0.2097 baseline needs a swing > 0.31 log units and LO delivered only ~0.34-0.44
# in smoke -- too close to call. Ratio of good-news to bad-news update weight is
# (1+a)/(1-a): LO = 4:1, HI = 9:1. HI is at the EDGE of the defensible optimism-bias
# band and is included as a capability probe, not as a proposed default. A C1 that
# clears ONLY at HI is a materially WEAKER result than one clearing at LO, and the
# manifest records `operative_asymmetry` so that distinction cannot be lost by a reader.
INFLATION_ASYMMETRY_LO = 0.6
INFLATION_ASYMMETRY_HI = 0.8
INFLATION_RV_FLOOR = 0.01

# ---- Pre-registered thresholds (NOT derived from this run's own statistics) ----
PRECISION_INIT_BASELINE = 0.5    # REEConfig precision_init default
RV_LIVE_FLOOR = 1e-6             # rv_final must differ from precision_init by more than this
RECALIB_MOVE_FLOOR = 1e-4        # F1 mean per-cycle |rv_after - rv_before| floor
BROADCAST_MOVE_FLOOR = 1e-4      # broadcast-ON arm's rv must differ from its matched OFF arm
INFLATION_MOVE_FLOOR = 1e-4      # inflation-ON arm's rv must sit BELOW its matched OFF arm
ZERO_POINT_PRESENT_FLOOR = 0.99  # fraction of cells whose _persistent_zero_point is populated

# Units below are natural-log ratio (>0 = absolutely overconfident, <0 = under-confident).
# Carried over from 774 unchanged so the two runs are directly comparable.
ABS_OVERCONF_MARGIN = 0.10       # a cell is ABSOLUTELY overconfident iff score > this
ABS_FLOOR = 0.10                 # minimum |mean paired delta| for a real effect
K_SD = 1.0                       # delta must exceed K_SD * sd(paired delta)
NONDEGEN_FLOOR = 0.05            # arm-pair separation below this = degenerate criterion
MIN_SEEDS_OVERCONF = 2           # C1 needs >= 2/3 seeds absolutely overconfident

# (arm_id, use_rem_precision_broadcast, inflation_asymmetry_or_None)
# 2x3 factorial: broadcast {OFF, ON} x asymmetry {OFF, LO, HI}.
# asymmetry None means use_waking_confidence_inflation=False -- the flag-OFF path, which
# is BIT-IDENTICAL by construction. It is deliberately NOT expressed as asymmetry 0.0:
# the ON path additionally applies waking_confidence_rv_floor, so ON-at-0.0 is NOT the
# same computation as OFF and would be a subtly different control.
ARMS: Tuple[Tuple[str, bool, Optional[float]], ...] = (
    ("ARM_OFF_OFF", False, None),
    ("ARM_BCAST_ONLY", True, None),
    ("ARM_INFL_LO", False, INFLATION_ASYMMETRY_LO),
    ("ARM_BOTH_LO", True, INFLATION_ASYMMETRY_LO),
    ("ARM_INFL_HI", False, INFLATION_ASYMMETRY_HI),
    ("ARM_BOTH_HI", True, INFLATION_ASYMMETRY_HI),
)

# Inflation levels in ASCENDING order. C1's operative level is the LOWEST that clears,
# so a result that holds at LO is never attributed to HI.
INFLATION_LEVELS = (
    ("LO", INFLATION_ASYMMETRY_LO, "ARM_INFL_LO", "ARM_BOTH_LO"),
    ("HI", INFLATION_ASYMMETRY_HI, "ARM_INFL_HI", "ARM_BOTH_HI"),
)

# Matched-pair maps for the cross-arm same-statistic readiness preconditions.
# Each broadcast-ON arm pairs with the broadcast-OFF arm at the SAME asymmetry;
# each inflation-ON arm pairs with the inflation-OFF arm at the SAME broadcast level.
BCAST_OFF_PARTNER = {
    "ARM_BCAST_ONLY": "ARM_OFF_OFF",
    "ARM_BOTH_LO": "ARM_INFL_LO",
    "ARM_BOTH_HI": "ARM_INFL_HI",
}
INFL_OFF_PARTNER = {
    "ARM_INFL_LO": "ARM_OFF_OFF",
    "ARM_BOTH_LO": "ARM_BCAST_ONLY",
    "ARM_INFL_HI": "ARM_OFF_OFF",
    "ARM_BOTH_HI": "ARM_BCAST_ONLY",
}


# ---------------------------------------------------------------- preconditions --
# Regime-conditioned per the V3-EXQ-785 rule: a precondition that is not meaningful for
# an arm is SCOPED OUT of that arm, never failed by it, and never vacates another arm.
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="rv_live",
        description="rv_final differs from precision_init by more than the floor (the "
                    "Q-042/530c substrate-liveness contract). Worst cell reported.",
        control="every seed of this arm; a dead rv makes the DV meaningless",
        threshold=RV_LIVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="f1_recalib_engaged",
        description="mean per-cycle |rv_after - rv_before| from the F1 WRITEBACK "
                    "recalibration exceeds the floor, i.e. REM was entered and the "
                    "MECH-204 lever moved rv. Phase 7's broadcast read is a no-op until "
                    "a REM entry populates serotonin._persistent_zero_point.",
        control="F1 recalibration is ON in every arm of this design",
        threshold=RECALIB_MOVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="zero_point_populated",
        description="fraction of this arm's cells whose serotonin._persistent_zero_point "
                    "is populated at eval start. The broadcast returns its no-target "
                    "sentinel (0.0) and does nothing when it is None.",
        control="a REM entry must have occurred during training",
        threshold=ZERO_POINT_PRESENT_FLOOR,
        direction="lower",
        applies_to=lambda ctx: bool(ctx["broadcast"]),
        applies_note="the persistent zero-point is only READ by the Phase 7 broadcast; "
                     "in a broadcast-OFF arm nothing consumes it, so requiring it would "
                     "make the precondition non-meaningful rather than informative.",
    ),
    PreconditionSpec(
        name="broadcast_moves_rv",
        description="mean over seeds of |rv_final(this arm) - rv_final(matched "
                    "broadcast-OFF arm)|. SAME STATISTIC the DV routes on (rv level), "
                    "measured against a matched positive control differing only in the "
                    "broadcast flag -- not a magnitude proxy (the V3-EXQ-643 defect).",
        control="matched broadcast-OFF arm at the same inflation level, same seeds",
        threshold=BROADCAST_MOVE_FLOOR,
        direction="lower",
        applies_to=lambda ctx: bool(ctx["broadcast"]),
        applies_note="a broadcast-OFF arm has no broadcast to engage; asserting this "
                     "would be structurally unsatisfiable and would vacate the arm.",
    ),
    PreconditionSpec(
        name="inflation_lowers_rv",
        description="mean over seeds of (rv_final(matched inflation-OFF arm) - "
                    "rv_final(this arm)). SIGNED: SD-076 must push rv DOWN (below the "
                    "symmetric-EMA level) or it is not an inflation source. Same "
                    "statistic the DV routes on.",
        control="matched inflation-OFF arm at the same broadcast level, same seeds",
        threshold=INFLATION_MOVE_FLOOR,
        direction="lower",
        applies_to=lambda ctx: bool(ctx["inflation"]),
        applies_note="an inflation-OFF arm has no asymmetry to engage.",
    ),
)


def _arm_ctx(arm_id: str, bcast: bool, asym: Optional[float]) -> Dict[str, object]:
    return {"arm_id": arm_id, "broadcast": bcast,
            "inflation": asym is not None, "asymmetry": asym}


ARM_CONTEXTS = [_arm_ctx(a, b, x) for (a, b, x) in ARMS]


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


def _make_agent(env: CausalGridWorldV2, bcast: bool, asym: Optional[float]) -> REEAgent:
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
        # Factor A: MECH-204 Phase 7 / Option B broadcast anchor.
        use_rem_precision_broadcast=bcast,
        rem_precision_broadcast_gain=(BROADCAST_GAIN if bcast else 0.0),
    )
    # Factor B: SD-076 waking confidence inflation lives on E3Config. asym is None for
    # the OFF level, which leaves the flag False -> the bit-identical symmetric path.
    cfg.e3.use_waking_confidence_inflation = asym is not None
    cfg.e3.waking_confidence_inflation_asymmetry = (float(asym) if asym is not None else 0.0)
    cfg.e3.waking_confidence_rv_floor = INFLATION_RV_FLOOR
    # Tonic 5-HT must be on for compute_recalibration_target() to be meaningful (both the
    # F1 WRITEBACK and the Phase 7 broadcast read it).
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _arm_config_slice(bcast: bool, asym: Optional[float]) -> Dict:
    """The config the cell's build+collect path actually reads."""
    return {
        "grid_size": GRID_SIZE,
        "steps_per_ep": STEPS_PER_EP,
        "n_train_eps": N_TRAIN_EPS,
        "n_eval_eps": N_EVAL_EPS,
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
        "use_rem_precision_broadcast": bcast,
        "rem_precision_broadcast_gain": (BROADCAST_GAIN if bcast else 0.0),
        "use_waking_confidence_inflation": asym is not None,
        "waking_confidence_inflation_asymmetry": (float(asym) if asym is not None else 0.0),
        "waking_confidence_rv_floor": INFLATION_RV_FLOOR,
    }


def _read_recalib_metrics(agent: REEAgent) -> Optional[Dict[str, float]]:
    """Sleep-cycle telemetry left in sleep_loop.state.last_metrics by agent.reset()."""
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
def _run_arm_seed(arm, seed, n_train, n_eval, steps, dry_run=False) -> Dict:
    arm_label, bcast, asym = arm

    with arm_cell(
        seed,
        config_slice=_arm_config_slice(bcast, asym),
        script_path=Path(__file__),
        include_driver_script_in_hash=False,  # mint-as-you-go: cross-driver reusable
    ) as cell:
        env = _make_env(seed, dry_run=dry_run)
        agent = _make_agent(env, bcast, asym)
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        print(f"Seed {seed} Condition {arm_label}", flush=True)

        # ---- Training: forward model learns; F1 recalibration fires each boundary ----
        recalib_moves: List[float] = []
        recalib_fired = 0
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
        zero_point = agent.serotonin._persistent_zero_point
        zero_point_populated = zero_point is not None

        # ---- Eval: capture confidence (rv) and accuracy (real forward-model error) ----
        eval_harness = StepHarness(agent, env, train_mode=False, seed=seed + 10000)
        rv_vals: List[float] = []
        pe_vals: List[float] = []
        for ep in range(n_eval):
            agent.reset()
            _, obs_dict = env.reset()
            eval_harness.reset()
            for _ in range(steps):
                result = eval_harness.step(obs_dict)
                rv_vals.append(float(agent.e3._running_variance))
                pe = result.residue_metrics.get("e3_prediction_error")
                if pe is not None:
                    pe_vals.append(float(pe))
                obs_dict = result.next_obs_dict
                if result.done:
                    break

        mean_rv = _mean(rv_vals)
        true_error_ref = _mean(pe_vals)

        # overconfidence_score = log(true_error_ref / mean_rv). >0 = the agent believes
        # itself more accurate than it is. true_error_ref is measured from the REAL
        # forward-model error and is independent of the rv the two levers modify.
        if true_error_ref > 1e-9 and mean_rv > 1e-9:
            calibration_ratio = mean_rv / true_error_ref
            overconfidence_score = float(np.log(true_error_ref / mean_rv))
        else:
            calibration_ratio = float("nan")
            overconfidence_score = 0.0

        absolutely_overconfident = overconfidence_score > ABS_OVERCONF_MARGIN
        print(
            f"  [eval] arm={arm_label} seed={seed} score={overconfidence_score:+.4f} "
            f"calib_ratio={calibration_ratio:.3f} true_err={true_error_ref:.6f} "
            f"mean_rv={mean_rv:.6f} rv_final={rv_after_training:.6f} "
            f"zero_point={'set' if zero_point_populated else 'NONE'}",
            flush=True,
        )
        print(f"verdict: {'PASS' if absolutely_overconfident else 'FAIL'}", flush=True)

        row = {
            "arm_id": arm_label,
            "seed": seed,
            "use_rem_precision_broadcast": bcast,
            "use_waking_confidence_inflation": asym is not None,
            "inflation_asymmetry": (float(asym) if asym is not None else 0.0),
            "overconfidence_score": overconfidence_score,
            "calibration_ratio": calibration_ratio,
            "true_error_ref": true_error_ref,
            "mean_running_variance": mean_rv,
            "rv_final_after_training": rv_after_training,
            "rv_delta_from_precision_init": abs(rv_after_training - PRECISION_INIT_BASELINE),
            "persistent_zero_point": (float(zero_point) if zero_point_populated else None),
            "zero_point_populated": zero_point_populated,
            "recalib_cycles_fired": recalib_fired,
            "recalib_mean_abs_move": _mean(recalib_moves),
            "absolutely_overconfident": absolutely_overconfident,
            "n_eval_ticks": len(rv_vals),
            "n_pe_ticks": len(pe_vals),
        }
        cell.stamp(row)
    return row


# ---------------------------------------------------------------------- analysis --
def _paired_delta(by_arm, seeds, arm_a, arm_b) -> Dict:
    """Per-seed paired delta score[arm_a] - score[arm_b], plus its mean/sd."""
    per_seed = [
        by_arm[arm_a][s]["overconfidence_score"] - by_arm[arm_b][s]["overconfidence_score"]
        for s in seeds
    ]
    mean = _mean(per_seed)
    sd = float(statistics.pstdev(per_seed)) if len(per_seed) > 1 else 0.0
    return {
        "per_seed": per_seed,
        "mean": mean,
        "sd": sd,
        "significant": bool(abs(mean) > ABS_FLOOR and abs(mean) > K_SD * sd),
    }


def _analyse(cells: List[Dict], seeds: List[int]) -> Dict:
    by_arm: Dict[str, Dict[int, Dict]] = {}
    for c in cells:
        by_arm.setdefault(c["arm_id"], {})[c["seed"]] = c

    arm_score = {a: _mean([by_arm[a][s]["overconfidence_score"] for s in seeds])
                 for a in by_arm}
    arm_ratio = {a: _mean([by_arm[a][s]["calibration_ratio"] for s in seeds])
                 for a in by_arm}
    arm_true_err = {a: _mean([by_arm[a][s]["true_error_ref"] for s in seeds])
                    for a in by_arm}
    arm_rv = {a: _mean([by_arm[a][s]["rv_final_after_training"] for s in seeds])
              for a in by_arm}

    # ---- per-arm regime-conditioned readiness gates ----
    arm_gates = []
    for (arm_id, bcast, asym) in ARMS:
        infl = asym is not None
        ctx = _arm_ctx(arm_id, bcast, asym)
        measured: Dict[str, float] = {
            # worst cell, not the mean -- `met` is a per-seed claim
            "rv_live": min(by_arm[arm_id][s]["rv_delta_from_precision_init"] for s in seeds),
            "f1_recalib_engaged": _mean(
                [by_arm[arm_id][s]["recalib_mean_abs_move"] for s in seeds]),
        }
        if bcast:
            partner = BCAST_OFF_PARTNER[arm_id]
            measured["zero_point_populated"] = _mean(
                [1.0 if by_arm[arm_id][s]["zero_point_populated"] else 0.0 for s in seeds])
            measured["broadcast_moves_rv"] = _mean(
                [abs(by_arm[arm_id][s]["rv_final_after_training"]
                     - by_arm[partner][s]["rv_final_after_training"]) for s in seeds])
        if infl:
            partner = INFL_OFF_PARTNER[arm_id]
            # SIGNED: inflation must push rv DOWN relative to its matched symmetric arm.
            measured["inflation_lowers_rv"] = _mean(
                [by_arm[partner][s]["rv_final_after_training"]
                 - by_arm[arm_id][s]["rv_final_after_training"] for s in seeds])
        arm_gates.append(
            evaluate_arm_gate(arm_id, ctx, list(PRECONDITION_SPECS), measured))

    gate = aggregate_arm_gates(arm_gates)

    # ---- per-level readouts + the OPERATIVE-LEVEL rule ----
    # C1's operative level is the LOWEST asymmetry that clears the margin, so a result
    # that holds at LO is never attributed to HI, and a C1 that clears ONLY at HI is
    # visibly the weaker finding it is.
    per_level: Dict[str, Dict] = {}
    for (lvl, asym, infl_arm, both_arm) in INFLATION_LEVELS:
        n_over = sum(1 for s in seeds if by_arm[infl_arm][s]["absolutely_overconfident"])
        per_level[lvl] = {
            "asymmetry": asym,
            "infl_arm": infl_arm,
            "both_arm": both_arm,
            "n_seeds_overconfident": n_over,
            "clears_c1": bool(n_over >= MIN_SEEDS_OVERCONF),
            "infl_score": arm_score[infl_arm],
            "both_score": arm_score[both_arm],
            "d_inflation": _paired_delta(by_arm, seeds, infl_arm, "ARM_OFF_OFF"),
            "d_broadcast_under_drift": _paired_delta(by_arm, seeds, both_arm, infl_arm),
        }

    operative = next((lvl for (lvl, _a, _i, _b) in INFLATION_LEVELS
                      if per_level[lvl]["clears_c1"]), None)

    # The broadcast's effect with NO drift to correct (the interaction's other cell).
    d_nodrift = _paired_delta(by_arm, seeds, "ARM_BCAST_ONLY", "ARM_OFF_OFF")

    # C1 (load-bearing): SD-076 makes ABSOLUTE overconfidence expressible at SOME
    # defensible asymmetry. This is a CAPABILITY question -- can the substrate express
    # it at all -- so clearing at either level answers it, with `operative_asymmetry`
    # recording the strength of the answer.
    c1 = operative is not None
    if c1:
        d_drift = per_level[operative]["d_broadcast_under_drift"]
        n_overconf_operative = per_level[operative]["n_seeds_overconfident"]
    else:
        d_drift = {"per_seed": [], "mean": 0.0, "sd": 0.0, "significant": False}
        n_overconf_operative = max(per_level[l]["n_seeds_overconfident"] for l in per_level)

    # C2 (load-bearing): at the operative level, the broadcast REDUCES overconfidence.
    c2 = bool(c1 and d_drift["mean"] < 0.0 and d_drift["significant"])
    # C3: the interaction -- the correction does more work when there IS drift.
    c3 = bool(c1 and abs(d_drift["mean"]) > abs(d_nodrift["mean"]))
    # C4: the OFF_OFF arm reproduces 774's ceiling (NOT absolutely overconfident).
    c4 = bool(arm_score["ARM_OFF_OFF"] <= ABS_OVERCONF_MARGIN)
    # C5 (dose-response, diagnostic): more asymmetry -> more overconfidence. A
    # non-monotone reading means the lever is not behaving as a graded drift source and
    # a PASS at HI should be read with suspicion even if C1 clears.
    c5 = bool(per_level["HI"]["infl_score"] > per_level["LO"]["infl_score"])

    criteria = [
        {"name": "C1_inflation_creates_absolute_overconfidence", "load_bearing": True,
         "passed": c1, "operative_asymmetry": (per_level[operative]["asymmetry"]
                                               if operative else None),
         "operative_level": operative,
         "n_seeds_overconfident": n_overconf_operative,
         "min_required": MIN_SEEDS_OVERCONF,
         "per_level_n_seeds": {l: per_level[l]["n_seeds_overconfident"] for l in per_level}},
        {"name": "C2_broadcast_corrects_under_drift", "load_bearing": True,
         "passed": c2, "mean_delta": d_drift["mean"], "sd": d_drift["sd"],
         "evaluated_at_level": operative},
        {"name": "C3_interaction_correction_larger_under_drift", "load_bearing": False,
         "passed": c3, "delta_drift": d_drift["mean"], "delta_nodrift": d_nodrift["mean"]},
        {"name": "C4_off_off_reproduces_774_ceiling", "load_bearing": False,
         "passed": c4, "off_off_score": arm_score["ARM_OFF_OFF"]},
        {"name": "C5_asymmetry_dose_response_monotone", "load_bearing": False,
         "passed": c5, "lo_score": per_level["LO"]["infl_score"],
         "hi_score": per_level["HI"]["infl_score"]},
    ]

    # ---- non-degeneracy, keyed to the owning arms' gates ----
    # Separation is evaluated at the OPERATIVE level when there is one; with no operative
    # level C1 is non-degenerate iff SOME level separated from OFF_OFF (i.e. the lever
    # moved the DV but not far enough -- a real negative), and degenerate iff no level
    # moved it at all (the lever did nothing -- uninformative).
    sep_by_level = {l: abs(per_level[l]["infl_score"] - arm_score["ARM_OFF_OFF"])
                    for l in per_level}
    if operative:
        sep_infl = sep_by_level[operative]
        sep_bcast = abs(per_level[operative]["both_score"] - per_level[operative]["infl_score"])
        c1_owner = per_level[operative]["infl_arm"]
        c2_owner = per_level[operative]["both_arm"]
    else:
        sep_infl = max(sep_by_level.values())
        sep_bcast = 0.0
        c1_owner = "ARM_INFL_HI"   # the strongest lever setting we tried
        c2_owner = "ARM_BOTH_HI"

    raw_non_degen = {
        "C1_inflation_creates_absolute_overconfidence": bool(sep_infl > NONDEGEN_FLOOR),
        # With no operative level C2 was never evaluated; mark it degenerate rather than
        # letting an unevaluated criterion read as a substantive negative.
        "C2_broadcast_corrects_under_drift": bool(
            operative is not None and sep_bcast > NONDEGEN_FLOOR),
        "C3_interaction_correction_larger_under_drift": bool(
            operative is not None and sep_infl > NONDEGEN_FLOOR
            and sep_bcast > NONDEGEN_FLOOR),
        "C4_off_off_reproduces_774_ceiling": bool(
            min(by_arm["ARM_OFF_OFF"][s]["rv_delta_from_precision_init"]
                for s in seeds) > RV_LIVE_FLOOR),
        "C5_asymmetry_dose_response_monotone": bool(
            abs(per_level["HI"]["infl_score"] - per_level["LO"]["infl_score"])
            > NONDEGEN_FLOOR),
    }
    # arm_id -> the criteria that arm OWNS (the arm whose gate most determines it).
    # A criterion owned by a RED arm is non_degenerate=False; `raw_non_degen` can
    # additionally fail a criterion whose green arm still lacks separation.
    criteria_by_arm = {
        c1_owner: ["C1_inflation_creates_absolute_overconfidence"],
        c2_owner: ["C2_broadcast_corrects_under_drift",
                   "C3_interaction_correction_larger_under_drift"],
        "ARM_OFF_OFF": ["C4_off_off_reproduces_774_ceiling"],
        "ARM_INFL_HI": ["C5_asymmetry_dose_response_monotone"],
    }
    # c1_owner can collide with ARM_INFL_HI when HI is operative; merge rather than clobber.
    if c1_owner == "ARM_INFL_HI":
        criteria_by_arm["ARM_INFL_HI"] = [
            "C1_inflation_creates_absolute_overconfidence",
            "C5_asymmetry_dose_response_monotone",
        ]
    criteria_non_degenerate = arm_criteria_non_degenerate(
        criteria_by_arm, gate, raw_non_degen)

    # ---- self-route ----
    readiness_ok = bool(gate["non_degenerate"])
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "inconclusive"
    elif c1 and c2:
        label = "phase7_sd076_calibration_loop_closed"
        outcome = "PASS"
        direction = "supports"
    elif c1 and not c2:
        # The DV is now measurable but the anchor did not correct -- a REAL negative
        # about Phase 7, and the first time that statement has been meaningful.
        label = "drift_expressible_broadcast_does_not_correct"
        outcome = "FAIL"
        direction = "does_not_support"
    else:
        # C1 failed at BOTH defensible asymmetries, so C2 is untestable and the DV is
        # still tautological. NOT a Phase-7 verdict. Because HI (9:1) is already at the
        # edge of the defensible optimism-bias band, the route here is NOT "sweep
        # higher" -- it is that the asymmetric-EMA form is the wrong drift source and a
        # different SD-076 mechanism is owed.
        label = "drift_source_insufficient_dv_still_tautological"
        outcome = "FAIL"
        direction = "inconclusive"

    per_claim = {
        "SD-076": ("supports" if c1 else "does_not_support") if readiness_ok else "unknown",
        "MECH-204": (("supports" if c2 else "does_not_support")
                     if (readiness_ok and c1) else "unknown"),
    }

    return {
        "outcome": outcome,
        "label": label,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "gate": gate,
        "arm_gates": arm_gates,
        "arm_overconfidence_score": arm_score,
        "arm_calibration_ratio": arm_ratio,
        "arm_true_error_ref": arm_true_err,
        "arm_rv_final": arm_rv,
        "deltas": {
            "d_broadcast_under_drift_at_operative": d_drift,
            "d_broadcast_no_drift": d_nodrift,
        },
        "per_level": per_level,
        "operative_level": operative,
        "operative_asymmetry": (per_level[operative]["asymmetry"] if operative else None),
        "n_seeds_overconfident": {
            arm_id: sum(1 for s in seeds if by_arm[arm_id][s]["absolutely_overconfident"])
            for arm_id in by_arm},
        "readiness_ok": readiness_ok,
        "thresholds": {
            "ABS_OVERCONF_MARGIN": ABS_OVERCONF_MARGIN,
            "ABS_FLOOR": ABS_FLOOR,
            "K_SD": K_SD,
            "NONDEGEN_FLOOR": NONDEGEN_FLOOR,
            "MIN_SEEDS_OVERCONF": MIN_SEEDS_OVERCONF,
            "RV_LIVE_FLOOR": RV_LIVE_FLOOR,
            "RECALIB_MOVE_FLOOR": RECALIB_MOVE_FLOOR,
            "BROADCAST_MOVE_FLOOR": BROADCAST_MOVE_FLOOR,
            "INFLATION_MOVE_FLOOR": INFLATION_MOVE_FLOOR,
            "ZERO_POINT_PRESENT_FLOOR": ZERO_POINT_PRESENT_FLOOR,
            "PRECISION_INIT_BASELINE": PRECISION_INIT_BASELINE,
        },
    }


# -------------------------------------------------------------------------- main --
def run_experiment(dry_run: bool = False) -> Dict:
    t0 = time.perf_counter()
    n_train = 2 if dry_run else N_TRAIN_EPS
    n_eval = 1 if dry_run else N_EVAL_EPS
    n_seeds = 2 if dry_run else N_SEEDS
    steps = 20 if dry_run else STEPS_PER_EP
    seeds = list(range(n_seeds))

    # Design-time proof: refuse before compute if any gate is structurally unsatisfiable.
    assert_no_structurally_unsatisfiable_gate(list(PRECONDITION_SPECS), ARM_CONTEXTS)

    cells: List[Dict] = []
    for arm in ARMS:
        for seed in seeds:
            cells.append(_run_arm_seed(arm, seed, n_train, n_eval, steps, dry_run=dry_run))

    adj = _analyse(cells, seeds)
    adj["cells"] = cells
    adj["seeds"] = seeds
    adj["elapsed_seconds"] = time.perf_counter() - t0
    adj["t0_perf"] = t0
    adj["config_n"] = {"steps_per_ep": steps, "n_train_eps": n_train,
                       "n_eval_eps": n_eval, "n_seeds": n_seeds}
    return adj


def main(dry_run: bool = False) -> Dict:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    adj = run_experiment(dry_run=dry_run)
    outcome = adj["outcome"]

    print("", flush=True)
    print(f"label={adj['label']} outcome={outcome} readiness_ok={adj['readiness_ok']} "
          f"operative_level={adj['operative_level']} "
          f"operative_asymmetry={adj['operative_asymmetry']}", flush=True)
    for arm_id in (a[0] for a in ARMS):
        print(f"  {arm_id:<16} score={adj['arm_overconfidence_score'][arm_id]:+.4f} "
              f"calib={adj['arm_calibration_ratio'][arm_id]:.3f} "
              f"rv={adj['arm_rv_final'][arm_id]:.6f}", flush=True)
    for c in adj["criteria"]:
        lb = " (load-bearing)" if c["load_bearing"] else ""
        print(f"  {c['name']}: {'PASS' if c['passed'] else 'FAIL'}{lb}", flush=True)

    if dry_run:
        print("DRY_RUN_COMPLETE", flush=True)
        return {"outcome": outcome, "manifest_path": None, "run_id": run_id}

    full_config = {
        "grid_size": GRID_SIZE,
        "steps_per_ep": adj["config_n"]["steps_per_ep"],
        "n_train_eps": adj["config_n"]["n_train_eps"],
        "n_eval_eps": adj["config_n"]["n_eval_eps"],
        "n_seeds": adj["config_n"]["n_seeds"],
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "precision_zero_point_ema_alpha": PRECISION_ZERO_POINT_EMA_ALPHA,
        "rem_precision_recalibration_step": REM_PRECISION_RECALIBRATION_STEP,
        "broadcast_gain": BROADCAST_GAIN,
        "inflation_asymmetry_lo": INFLATION_ASYMMETRY_LO,
        "inflation_asymmetry_hi": INFLATION_ASYMMETRY_HI,
        "inflation_rv_floor": INFLATION_RV_FLOOR,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "arms": [{"arm_id": a[0], "use_rem_precision_broadcast": a[1],
                  "use_waking_confidence_inflation": a[2] is not None,
                  "waking_confidence_inflation_asymmetry": (
                      float(a[2]) if a[2] is not None else 0.0)} for a in ARMS],
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
        "supersedes": SUPERSEDES,
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
            "arm_overconfidence_score": adj["arm_overconfidence_score"],
            "arm_calibration_ratio": adj["arm_calibration_ratio"],
            "arm_true_error_ref": adj["arm_true_error_ref"],
            "arm_rv_final": adj["arm_rv_final"],
            "deltas": adj["deltas"],
            "per_level": adj["per_level"],
            "operative_level": adj["operative_level"],
            "operative_asymmetry": adj["operative_asymmetry"],
            "n_seeds_overconfident": adj["n_seeds_overconfident"],
            "readiness_ok": adj["readiness_ok"],
        },
        "thresholds": adj["thresholds"],
        "arm_results": adj["cells"],
        "per_seed_cells": adj["cells"],
        "elapsed_seconds": adj["elapsed_seconds"],
        "notes": (
            "DIAGNOSTIC 2x2 substrate-readiness validation of MECH-204 Phase 7 / Option B "
            "(REEConfig.use_rem_precision_broadcast, the CORRECTION) x SD-076 "
            "(E3Config.use_waking_confidence_inflation, the SOURCE), both landed ree-v3 "
            "8ac193d7ed 2026-07-20. Successor to V3-EXQ-774, whose confirmed autopsy "
            "(failure_autopsy_V3-EXQ-774_2026-07-17) adjudicated substrate_ceiling: the "
            "symmetric precision EMA made running_variance track true prediction error BY "
            "CONSTRUCTION, pinning overconfidence_score near zero regardless of ablation "
            "(774 measured -0.000148 / -0.000918 on its suppressed arms). BOTH factors are "
            "ablated because a Phase-7-only ablation would retest to an identical null that "
            "would read as a Phase-7 refutation when nothing had been measured. C1 "
            "(load-bearing) asks whether SD-076 makes ABSOLUTE overconfidence expressible "
            "at all; C2 (load-bearing) asks whether the broadcast anchor CORRECTS it where "
            "drift exists; C3 is the 2x2 interaction; C4 checks that ARM_OFF_OFF reproduces "
            "774's ceiling. DV-SYMMETRY: overconfidence_score is invariant under permutation "
            "of eval ticks but NOT under changes to rv's level, and BOTH manipulations write "
            "rv directly -- which is why the Phase 7 write-site was corrected from E3 score "
            "to precision space (a broadcast scalar on the score would have been invariant "
            "under this DV, the V3-EXQ-604c class). Preconditions are REGIME-CONDITIONED per "
            "the V3-EXQ-785 rule: broadcast/inflation readiness checks are scoped to the arms "
            "where they are meaningful and never vacate a clean arm. Readiness failure "
            "self-routes substrate_not_ready_requeue, never a substrate verdict. DIAGNOSTIC "
            "=> excluded from governance confidence/conflict scoring. Deliberately does NOT "
            "tag MECH-173: that claim needs the rem_enabled manipulation this design does not "
            "run; a PASS here makes a MECH-173 retest meaningful, it is not MECH-173 support."
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
