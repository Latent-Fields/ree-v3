"""V3-EXQ-850: MECH-204 x SD-076 H1 discrimination probe -- does F1/REM precision
recalibration DAMP SD-076's waking-induced rv drift before the eval-phase
measurement? Single-axis (integration) leg of the GOV-FANOUT-1 portfolio raised by
failure_autopsy_V3-EXQ-794a_2026-07-31 (H1 of 3; H2 is the sibling exposure-budget
leg, a distinct script, queued separately -- do not conflate).

WHY THIS PROBE EXISTS
----------------------
V3-EXQ-794a's full behavioural loop measured SD-076's asymmetric-EMA waking-confidence
inflation reaching rv_final 0.003997733405264173 (LO, asymmetry=0.6) / 0.003870367272153275
(HI, asymmetry=0.8) at the SD-076 headroom repair's (ree-v3 452f99e367) own operating
scale. The SAME repaired mechanism's own validation smoke (recorded in
substrate_queue.json, sd_waking_confidence_inflation_headroom entry) reached rv_final
0.0025377 (LO) / 0.0021031 (HI) at the IDENTICAL measured error scale (true_error_ref
~0.0037) -- both flagged genuinely overconfident, roughly TWICE the reduction from the
un-inflated baseline that the full loop achieved. The autopsy's load-bearing new fact:
the smoke never exercises the sleep/F1 loop at all, while 794a's full loop runs F1/REM
precision recalibration ON in EVERY arm (rem_precision_recalibration_step=0.25) by design
necessity (Phase 7 -- MECH-204's broadcast correction under test in 794a -- runs alongside
F1 and its read is a no-op until a REM entry has populated
serotonin._persistent_zero_point, so F1 cannot be switched off in that design without
breaking Phase 7's own precondition).

H1 (this probe): F1/REM precision recalibration WRITEBACK (REEConfig
use_rem_precision_recalibration, step 0.25) nudges E3._running_variance back toward
1.0/target_precision every REM cycle, and that pull partially counteracts SD-076's
waking-induced downward drift on rv BEFORE the eval-phase measurement -- damping the
observable effect relative to the isolated (no-sleep-loop) smoke, which never engages
this pull at all.

DESIGN: single-axis re-run. This probe does NOT run Phase 7 (MECH-204's broadcast
correction is not under test here -- use_rem_precision_broadcast stays at its bit-identical
False default in every arm) and does NOT run the ARM_OFF_OFF / ARM_BCAST_ONLY / ARM_BOTH_LO
/ ARM_BOTH_HI cells from 794a's 2x3 factorial. It re-runs ONLY 794a's two INFL-only cells
(ARM_INFL_LO, ARM_INFL_HI: asymmetry {0.6, 0.8}, broadcast OFF) with EXACTLY ONE further
change from 794a: REEConfig.use_rem_precision_recalibration is set False instead of True.
Every other config value (grid size, training/eval episode counts, LR, alpha_world/self,
sws/rem attribution steps, the SD-076 repaired floor knobs, the seed set) is held identical
to 794a so the comparison isolates the F1 axis alone.

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per-arm; /queue-experiment Step 3)
-------------------------------------------------------------------------------
DV = rv_final_after_training (== E3._running_variance at end of training), read directly
off the substrate, and the derived overconfidence_score = log(true_error_ref / mean_rv),
a scalar function of that same rv trajectory and the independently-measured forward-model
error. Symmetry group: invariant under PERMUTATION of eval ticks (both terms of the score
are tick-means); NOT invariant under any change to rv's LEVEL.

  Both arms (ARM_INFL_LO, ARM_INFL_HI), THE manipulation under test: flipping
  use_rem_precision_recalibration False->True changes whether the WRITEBACK phase's
  MECH-204 consumer (ree_core/sleep/phase_manager.py ~line 427) ever nudges
  E3._running_variance toward 1.0/target_precision at all -- a LEVEL change to the exact
  quantity the DV reads, not a permutation. NOT invariant. OK: the manipulation can move
  the DV, so a null result is a real null, not an arithmetic identity fixed before the run.
  Confirmed by direct code read: ree_core/sleep/phase_manager.py's WRITEBACK block is
  gated on `self.use_rem_precision_recalibration` and writes NOTHING to
  writeback_metrics (no mech204_recalibration_fired / _before / _after keys at all) when
  the flag is False -- so this run's own recalib_cycles_fired / recalib_mean_abs_move
  readouts are a direct, code-level manipulation check that the ablation actually took
  effect, not merely a config-flag assertion (see the f1_recalib_disabled_confirmed
  precondition below).

GOV-REUSE-1 (existing-evidence check)
--------------------------------------
Decisive readout: rv_final_after_training for ARM_INFL_LO / ARM_INFL_HI under the SD-076
headroom-repaired floor (relative_frac=0.2, soft, softness=0.25) with
use_rem_precision_recalibration=False in a FULL behavioural (sleep-loop-engaged) run.
`REE_assembly/scripts/reanalysis_query.py query --readout rv_final --claim MECH-204` (run
2026-07-31) finds exactly two manifests carrying rv_final on a recoverable substrate_hash:
v3_exq_794_..._20260721T113848Z_v3 (substrate_hash 402e3f5a23a3a8e1..., PRE-repair, F1 ON)
and v3_exq_794a_..._20260724T063301Z_v3 (substrate_hash f569f39451e9746a..., POST-repair,
F1 ON). Neither has F1 disabled -- both are on the opposite side of exactly the axis this
probe manipulates, so neither can answer H1. The repair's own validation smoke
(substrate_queue.json sd_waking_confidence_inflation_headroom.implementation_note_update)
is an isolated synthetic-error-sequence computation with NO sleep loop at all (not a
tracked run_id / manifest, and not comparable on substrate_hash even if it were -- it never
engages CausalGridWorldV2, REEAgent training, or the sleep/F1 machinery this probe reads).
So the F1-disabled + full-behavioural-loop + repaired-floor combination is recorded
NOWHERE. Not recoverable -> run.

Re-derive brake (2.5b): re-ran the corpus count for MECH-204 and SD-076 (2026-07-31) --
0 confirmed substrate_ceiling autopsies for either claim (matches
failure_autopsy_V3-EXQ-794a_2026-07-31.json's own re_derive_brake.note). Brake does not
fire. This is also explicitly a GOV-FANOUT-1 discrimination leg on a fanout_recommendation,
not a sequential re-pose of the braked design, so the brake's scope exclusion applies even
were the count nonzero.

Substrate readiness (2.5): every feature this probe exercises is landed and IMPLEMENTED --
SD-076 waking confidence inflation + the headroom repair (ree-v3 452f99e367), MECH-204 F1
precision recalibration (ree-v3, landed well before 794/794a), sws/rem sleep loop
machinery. use_rem_precision_broadcast (Phase 7 / MECH-204 Option B) is likewise landed but
simply held at its False default here -- this probe does not exercise it.

WHY NO ARM_OFF_OFF / SYMMETRIC-EMA BASELINE IN THIS RUN
----------------------------------------------------------
794a's `inflation_lowers_rv` precondition compared each inflation arm's rv_final against a
matched inflation-OFF arm at the same broadcast level (its positive control that SD-076's
asymmetric EMA genuinely pulls rv below the symmetric-EMA path). This probe does not run
that matched OFF arm -- rerunning it under F1-disabled would answer a *different* question
(whether SD-076 has directional effect at all with F1 off, already established at the same
scale by both 774's OFF_OFF arm and the isolated smoke's own OFF path) and would double the
run's cost for a comparison the discrimination does not need. H1's decisive test is a
BETWEEN-RUN comparison against two already-recorded external reference points at the
IDENTICAL error scale and identical inflation levels: 794a's own rv_final (F1 ON, this
run's counterfactual) and the repair smoke's rv_final (F1 never engaged at all, the
isolated-mechanism ceiling). Both are pre-registered constants below (SMOKE_RV_FINAL,
REF_794A_RV_FINAL), not derived from this run's own statistics. This run supplies the
THIRD point on that same axis (F1 present in the full loop vs F1 absent in the full loop),
which is exactly what discriminates H1 from H2/H3 -- an in-run OFF/baseline arm doesn't
add information for that specific question and was dropped to keep this a genuine
single-axis probe (2 arms, not the full 2x3 factorial) per the task's own design spec.

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)

PURPOSE / SCOPE
-----------------
DIAGNOSTIC discrimination probe, NOT governance evidence for MECH-204 or SD-076's core
claims (C1/C2 of 794a's own design are not re-tested here -- this asks only which of
H1/H2/H3 explains 794a's full-loop-vs-smoke shortfall). A CONFIRMED H1 here is evidence
that MECH-204's F1 consumer and SD-076's drift source genuinely interact (informative for
BOTH claims: it explains 794a's C1 shortfall as an interaction effect, not a defect in
either mechanism). A REFUTED H1 (both doses stay near 794a's own F1-ON values) rules out
the F1-damping explanation and shifts weight toward H2 (training-exposure budget, the
sibling probe) and H3 (wrong drift-source mechanism form, per the autopsy WEAKENED but not
eliminated). Excluded from governance confidence/conflict scoring by EXPERIMENT_PURPOSE.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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

QUEUE_ID = "V3-EXQ-850"
EXPERIMENT_TYPE = "v3_exq_850_mech204_sd076_h1_f1_damping_probe"
CLAIM_IDS = ["MECH-204", "SD-076"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = (
    "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"
)
# This is a discrimination leg raised BY the 794a autopsy's fanout_recommendation, not a
# same-question re-pose of 794a itself -- 794a is not corrected or replaced by this run.
FANOUT_SOURCE = "failure_autopsy_V3-EXQ-794a_2026-07-31.json"
FANOUT_HYPOTHESIS = "H1-f1-recalibration-damping"

# ---- Run shape (identical to 794a's INFL-only cells -- only F1 changes) ----
N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_SEEDS = 3
GRID_SIZE = 12
STEPS_PER_EP = 200
LR = 5e-4

# ---- Substrate operating point (held constant vs 794a EXCEPT the H1 axis itself) ----
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25  # inert here: the consumer is disabled (below)
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# THE H1 AXIS. 794a held this True in every arm "by design necessity" (Phase 7's read is
# a no-op without a REM entry populating the zero-point). This probe does not run Phase 7,
# so nothing requires F1 to stay engaged, and disabling it is precisely H1's manipulation.
USE_REM_PRECISION_RECALIBRATION = False

# ---- SD-076 dose levels (unchanged from 794a) ----
INFLATION_ASYMMETRY_LO = 0.6
INFLATION_ASYMMETRY_HI = 0.8
INFLATION_RV_FLOOR = 0.01
# SD-076 headroom repair knobs (ree-v3 452f99e367) -- unchanged from 794a.
INFLATION_RV_FLOOR_RELATIVE_FRAC = 0.2
INFLATION_RV_FLOOR_MODE = "soft"
INFLATION_RV_FLOOR_SOFTNESS = 0.25

# ---- Pre-registered EXTERNAL reference points (NOT derived from this run) ----
# 794a's own rv_final for these exact two arms (F1 ON) -- this run's counterfactual.
# Source: REE_assembly/evidence/experiments/
#   v3_exq_794a_mech204_phase7_sd076_calibration_loop_2x2_20260724T063301Z_v3.json
#   aggregates.arm_rv_final.{ARM_INFL_LO,ARM_INFL_HI}
REF_794A_RV_FINAL = {"LO": 0.003997733405264173, "HI": 0.003870367272153275}
# The SD-076 headroom repair's own isolated-mechanism validation smoke (no sleep loop, F1
# never engaged) at the identical measured error scale (true_error_ref ~0.0037).
# Source: REE_assembly/evidence/planning/substrate_queue.json
#   sd_waking_confidence_inflation_headroom.implementation_note_update
SMOKE_RV_FINAL = {"LO": 0.0025377, "HI": 0.0021031}

# ---- Pre-registered thresholds (NOT derived from this run's own statistics) ----
PRECISION_INIT_BASELINE = 0.5    # REEConfig precision_init default
RV_LIVE_FLOOR = 1e-6             # rv_final must differ from precision_init by more than this
# Guards the SAME clamp risk 794a discovered, re-checked under F1-off (the repair's
# behaviour without F1's periodic pull has not itself been verified in a full loop before).
DOSE_SEPARATION_FLOOR = 1e-4
# The manipulation-check floor: F1's WRITEBACK consumer writes NOTHING to
# writeback_metrics when disabled (confirmed by direct code read, module docstring), so
# recalib_cycles_fired must read exactly 0 and recalib_mean_abs_move must read exactly
# 0.0 for every seed of every arm. Ceiling-direction preconditions below (met when
# measured < threshold).
F1_DISABLED_FIRED_CEILING = 0.5     # recalib_cycles_fired must be < this (i.e. == 0)
F1_DISABLED_MOVE_CEILING = 1e-9     # recalib_mean_abs_move must be < this (i.e. == 0.0)
# H1 decision rule: a level "supports" H1 iff its rv_final has closed at least HALF the
# gap between 794a's F1-ON reference and the smoke's F1-isolated reference (closer to the
# smoke than to 794a). Pre-registered midpoint split, not derived from this run's data.
H1_GAP_CLOSED_THRESHOLD = 0.5
# A level's rv_final must move at least this far from 794a's F1-ON reference for the
# level's H1 verdict (support OR refute) to be non-degenerate -- otherwise "no support"
# cannot be distinguished from "nothing moved at all". Same order of magnitude as
# DOSE_SEPARATION_FLOOR and roughly 7-14% of the LO/HI gap sizes below.
NONDEGEN_RV_MOVE_FLOOR = 1e-4
# C_MONO (informational, non-load-bearing): more asymmetry should still lower rv_final
# (more inflation), the same direction 794a's own C5 checked.
NONDEGEN_FLOOR = 0.00005  # rv units; LO vs HI rv_final separation floor for C_MONO

ARMS: Tuple[Tuple[str, float], ...] = (
    ("ARM_INFL_LO", INFLATION_ASYMMETRY_LO),
    ("ARM_INFL_HI", INFLATION_ASYMMETRY_HI),
)
LEVELS: Tuple[Tuple[str, float, str], ...] = (
    ("LO", INFLATION_ASYMMETRY_LO, "ARM_INFL_LO"),
    ("HI", INFLATION_ASYMMETRY_HI, "ARM_INFL_HI"),
)
# Sibling arm at the OTHER asymmetry level -- the matched positive control for
# dose_levels_separated (differs ONLY in the dose).
DOSE_SIBLING = {"ARM_INFL_LO": "ARM_INFL_HI", "ARM_INFL_HI": "ARM_INFL_LO"}


# ---------------------------------------------------------------- preconditions --
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
        name="f1_recalib_disabled_confirmed_fired",
        description="recalib_cycles_fired must be 0 for every seed -- the WRITEBACK "
                    "consumer (ree_core/sleep/phase_manager.py) writes "
                    "mech204_recalibration_fired only when "
                    "use_rem_precision_recalibration is True, so a nonzero count here "
                    "means the ablation did not actually take effect (a config/wiring "
                    "bug), not a scientific result.",
        control="use_rem_precision_recalibration=False in every arm of this design",
        threshold=F1_DISABLED_FIRED_CEILING,
        direction="upper",
    ),
    PreconditionSpec(
        name="f1_recalib_disabled_confirmed_move",
        description="recalib_mean_abs_move must be 0.0 for every seed -- same "
                    "manipulation check as f1_recalib_disabled_confirmed_fired, on the "
                    "movement statistic instead of the fired count.",
        control="use_rem_precision_recalibration=False in every arm of this design",
        threshold=F1_DISABLED_MOVE_CEILING,
        direction="upper",
    ),
    PreconditionSpec(
        name="dose_levels_separated",
        description="|rv_final(this arm) - rv_final(sibling arm at the OTHER asymmetry)|. "
                    "THE 794 GATE, re-checked under F1-off: two nominally different doses "
                    "producing the same rv is a SATURATION signature, not a null. Same "
                    "statistic the DV routes on, against a control differing ONLY in the "
                    "dose.",
        control="sibling inflation arm at the other asymmetry level, same seeds -- "
                "differs only in the dose",
        threshold=DOSE_SEPARATION_FLOOR,
        direction="lower",
    ),
)


def _arm_ctx(arm_id: str, asym: float) -> Dict[str, object]:
    return {"arm_id": arm_id, "asymmetry": asym}


ARM_CONTEXTS = [_arm_ctx(a, x) for (a, x) in ARMS]
_ZG = ZGoalStreamAccumulator()


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
        # THE H1 AXIS -- everything else below is unchanged from 794a.
        use_rem_precision_recalibration=USE_REM_PRECISION_RECALIBRATION,
        precision_zero_point_ema_alpha=PRECISION_ZERO_POINT_EMA_ALPHA,
        rem_precision_recalibration_step=REM_PRECISION_RECALIBRATION_STEP,
        # Phase 7 not exercised by this probe -- bit-identical False/0.0 defaults.
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


def _arm_config_slice(asym: float) -> Dict:
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
        "use_rem_precision_recalibration": USE_REM_PRECISION_RECALIBRATION,
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
        # Read by the cell's call graph (rv_delta_from_precision_init readout, and the
        # per-cell verdict print's gap_closed_frac) -- declared per the config_slice
        # under-approximation lint (validate_experiments.py CONFIG_SLICE-DECLARATION).
        "precision_init_baseline": PRECISION_INIT_BASELINE,
        "h1_gap_closed_threshold": H1_GAP_CLOSED_THRESHOLD,
    }


def _read_recalib_metrics(agent: REEAgent) -> Optional[Dict[str, float]]:
    """Sleep-cycle telemetry left in sleep_loop.state.last_metrics by agent.reset().

    When use_rem_precision_recalibration is False, phase_manager's WRITEBACK block
    never runs and writes none of these keys -- so this always returns None in this
    probe, by construction. Read anyway (rather than trusting the config flag) so a
    nonzero reading is a real, catchable bug signal rather than a silent assumption.
    """
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
    arm_label, asym = arm

    with arm_cell(
        seed,
        config_slice=_arm_config_slice(asym),
        script_path=Path(__file__),
        include_driver_script_in_hash=False,  # mint-as-you-go: cross-driver reusable
    ) as cell:
        env = _make_env(seed, dry_run=dry_run)
        agent = _make_agent(env, asym)
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        print(f"Seed {seed} Condition {arm_label}", flush=True)

        # ---- Training: forward model learns; F1 WRITEBACK is a no-op every boundary --
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

        _ZG.observe(agent)  # AFTER stepping is complete for this cell

        mean_rv = _mean(rv_vals)
        true_error_ref = _mean(pe_vals)

        if true_error_ref > 1e-9 and mean_rv > 1e-9:
            calibration_ratio = mean_rv / true_error_ref
            overconfidence_score = float(np.log(true_error_ref / mean_rv))
        else:
            calibration_ratio = float("nan")
            overconfidence_score = 0.0

        print(
            f"  [eval] arm={arm_label} seed={seed} score={overconfidence_score:+.4f} "
            f"calib_ratio={calibration_ratio:.3f} true_err={true_error_ref:.6f} "
            f"mean_rv={mean_rv:.6f} rv_final={rv_after_training:.6f} "
            f"recalib_fired={recalib_fired} recalib_move={_mean(recalib_moves):.8f}",
            flush=True,
        )
        # Diagnostic per-cell verdict: is THIS cell's rv_final closer to the smoke's
        # reference than to 794a's F1-ON reference? (final H1 verdict is level-level,
        # computed in _analyse over the seed mean -- this print is per-seed context.)
        level = "LO" if arm_label == "ARM_INFL_LO" else "HI"
        gap_total = REF_794A_RV_FINAL[level] - SMOKE_RV_FINAL[level]
        closer_to_smoke = (
            (REF_794A_RV_FINAL[level] - rv_after_training) / gap_total
            >= H1_GAP_CLOSED_THRESHOLD
        )
        print(f"verdict: {'PASS' if closer_to_smoke else 'FAIL'}", flush=True)

        row = {
            "arm_id": arm_label,
            "seed": seed,
            "inflation_asymmetry": float(asym),
            "overconfidence_score": overconfidence_score,
            "calibration_ratio": calibration_ratio,
            "true_error_ref": true_error_ref,
            "mean_running_variance": mean_rv,
            "rv_final_after_training": rv_after_training,
            "rv_delta_from_precision_init": abs(rv_after_training - PRECISION_INIT_BASELINE),
            "recalib_cycles_fired": recalib_fired,
            "recalib_mean_abs_move": _mean(recalib_moves),
            "n_eval_ticks": len(rv_vals),
            "n_pe_ticks": len(pe_vals),
        }
        cell.stamp(row)
    return row


# ---------------------------------------------------------------------- analysis --
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

    # ---- per-arm readiness gates ----
    arm_gates = []
    for (arm_id, asym) in ARMS:
        ctx = _arm_ctx(arm_id, asym)
        sibling = DOSE_SIBLING[arm_id]
        measured: Dict[str, float] = {
            "rv_live": min(by_arm[arm_id][s]["rv_delta_from_precision_init"] for s in seeds),
            "f1_recalib_disabled_confirmed_fired": max(
                by_arm[arm_id][s]["recalib_cycles_fired"] for s in seeds),
            "f1_recalib_disabled_confirmed_move": max(
                by_arm[arm_id][s]["recalib_mean_abs_move"] for s in seeds),
            "dose_levels_separated": abs(
                _mean([by_arm[arm_id][s]["rv_final_after_training"] for s in seeds])
                - _mean([by_arm[sibling][s]["rv_final_after_training"] for s in seeds])),
        }
        arm_gates.append(
            evaluate_arm_gate(arm_id, ctx, list(PRECONDITION_SPECS), measured))

    gate = aggregate_arm_gates(arm_gates)

    # ---- per-level H1 discrimination readout ----
    per_level: Dict[str, Dict] = {}
    for (lvl, asym, arm_id) in LEVELS:
        rv_final = arm_rv[arm_id]
        ref_794a = REF_794A_RV_FINAL[lvl]
        smoke = SMOKE_RV_FINAL[lvl]
        gap_total = ref_794a - smoke
        gap_closed = ref_794a - rv_final
        gap_closed_frac = gap_closed / gap_total if gap_total else float("nan")
        move_from_794a = abs(ref_794a - rv_final)
        non_degenerate = bool(move_from_794a > NONDEGEN_RV_MOVE_FLOOR)
        h1_supported = bool(non_degenerate and gap_closed_frac >= H1_GAP_CLOSED_THRESHOLD)
        per_level[lvl] = {
            "asymmetry": asym,
            "arm_id": arm_id,
            "rv_final": rv_final,
            "ref_794a_rv_final": ref_794a,
            "smoke_rv_final": smoke,
            "gap_total": gap_total,
            "gap_closed": gap_closed,
            "gap_closed_frac": gap_closed_frac,
            "move_from_794a": move_from_794a,
            "non_degenerate": non_degenerate,
            "h1_supported": h1_supported,
            "overconfidence_score": arm_score[arm_id],
            "calibration_ratio": arm_ratio[arm_id],
        }

    readiness_ok = bool(gate["non_degenerate"])
    lo, hi = per_level["LO"], per_level["HI"]
    both_non_degenerate = bool(lo["non_degenerate"] and hi["non_degenerate"])
    n_supporting = sum(1 for l in (lo, hi) if l["h1_supported"])

    # ---- C_MONO (informational, non-load-bearing): dose-response direction check ----
    c_mono = bool(hi["rv_final"] < lo["rv_final"])  # more asymmetry -> lower rv (more infl.)
    c_mono_non_degenerate = bool(abs(hi["rv_final"] - lo["rv_final"]) > NONDEGEN_FLOOR)

    criteria = [
        {"name": "C1_LO_h1_f1_damping_confirmed", "load_bearing": True,
         "passed": lo["h1_supported"], "gap_closed_frac": lo["gap_closed_frac"],
         "rv_final": lo["rv_final"], "non_degenerate": lo["non_degenerate"]},
        {"name": "C2_HI_h1_f1_damping_confirmed", "load_bearing": True,
         "passed": hi["h1_supported"], "gap_closed_frac": hi["gap_closed_frac"],
         "rv_final": hi["rv_final"], "non_degenerate": hi["non_degenerate"]},
        {"name": "C_MONO_dose_response_direction", "load_bearing": False,
         "passed": c_mono, "lo_rv_final": lo["rv_final"], "hi_rv_final": hi["rv_final"]},
    ]

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "inconclusive"
    elif not both_non_degenerate:
        label = "h1_result_indeterminate_insufficient_separation"
        outcome = "FAIL"
        direction = "inconclusive"
    elif n_supporting == 2:
        label = "h1_f1_damping_confirmed_both_doses"
        outcome = "PASS"
        direction = "supports"
    elif n_supporting == 0:
        label = "h1_f1_damping_not_supported_both_doses"
        outcome = "FAIL"
        direction = "inconclusive"
    else:
        label = "h1_f1_damping_partially_supported_mixed_dose"
        outcome = "FAIL"
        direction = "mixed"

    per_claim = {
        "MECH-204": direction if readiness_ok else "unknown",
        "SD-076": direction if readiness_ok else "unknown",
    }

    # ---- non-degeneracy, keyed to the owning arms' gates ----
    raw_non_degen = {
        "C1_LO_h1_f1_damping_confirmed": lo["non_degenerate"],
        "C2_HI_h1_f1_damping_confirmed": hi["non_degenerate"],
        "C_MONO_dose_response_direction": c_mono_non_degenerate,
    }
    criteria_by_arm = {
        "ARM_INFL_LO": ["C1_LO_h1_f1_damping_confirmed"],
        "ARM_INFL_HI": ["C2_HI_h1_f1_damping_confirmed", "C_MONO_dose_response_direction"],
    }
    criteria_non_degenerate = arm_criteria_non_degenerate(
        criteria_by_arm, gate, raw_non_degen)

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
        "per_level": per_level,
        "n_levels_supporting_h1": n_supporting,
        "readiness_ok": readiness_ok,
        "thresholds": {
            "H1_GAP_CLOSED_THRESHOLD": H1_GAP_CLOSED_THRESHOLD,
            "NONDEGEN_RV_MOVE_FLOOR": NONDEGEN_RV_MOVE_FLOOR,
            "NONDEGEN_FLOOR": NONDEGEN_FLOOR,
            "DOSE_SEPARATION_FLOOR": DOSE_SEPARATION_FLOOR,
            "RV_LIVE_FLOOR": RV_LIVE_FLOOR,
            "F1_DISABLED_FIRED_CEILING": F1_DISABLED_FIRED_CEILING,
            "F1_DISABLED_MOVE_CEILING": F1_DISABLED_MOVE_CEILING,
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

    assert_no_structurally_unsatisfiable_gate(
        list(PRECONDITION_SPECS), ARM_CONTEXTS, arm_id_key="arm_id")

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
          f"n_levels_supporting_h1={adj['n_levels_supporting_h1']}/2", flush=True)
    for arm_id in (a[0] for a in ARMS):
        print(f"  {arm_id:<16} score={adj['arm_overconfidence_score'][arm_id]:+.4f} "
              f"calib={adj['arm_calibration_ratio'][arm_id]:.3f} "
              f"rv={adj['arm_rv_final'][arm_id]:.6f}", flush=True)
    for lvl in ("LO", "HI"):
        pl = adj["per_level"][lvl]
        print(f"  level={lvl} rv_final={pl['rv_final']:.6f} "
              f"ref_794a={pl['ref_794a_rv_final']:.6f} smoke={pl['smoke_rv_final']:.6f} "
              f"gap_closed_frac={pl['gap_closed_frac']:.3f} "
              f"h1_supported={pl['h1_supported']}", flush=True)
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
        "use_rem_precision_recalibration": USE_REM_PRECISION_RECALIBRATION,
        "use_rem_precision_broadcast": False,
        "rem_precision_broadcast_gain": 0.0,
        "inflation_asymmetry_lo": INFLATION_ASYMMETRY_LO,
        "inflation_asymmetry_hi": INFLATION_ASYMMETRY_HI,
        "inflation_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "sleep_loop_episodes_K": 1,
        "tonic_5ht_enabled": True,
        "arms": [{"arm_id": a[0], "waking_confidence_inflation_asymmetry": float(a[1])}
                 for a in ARMS],
        "env": {"num_hazards": 3, "num_resources": 3, "hazard_harm": 0.04,
                "proximity_harm_scale": 0.12, "proximity_benefit_scale": 0.10,
                "use_proxy_fields": True, "resource_respawn_on_consume": True},
        "seeds": adj["seeds"],
        "reference_points": {
            "ref_794a_rv_final": REF_794A_RV_FINAL,
            "smoke_rv_final": SMOKE_RV_FINAL,
        },
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
        "fanout_hypothesis": FANOUT_HYPOTHESIS,
        "dose_key": "asymmetry",
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
            "per_level": adj["per_level"],
            "n_levels_supporting_h1": adj["n_levels_supporting_h1"],
            "readiness_ok": adj["readiness_ok"],
        },
        "thresholds": adj["thresholds"],
        "arm_results": adj["cells"],
        "per_seed_cells": adj["cells"],
        "elapsed_seconds": adj["elapsed_seconds"],
        "notes": (
            "DIAGNOSTIC single-axis discrimination probe -- GOV-FANOUT-1 leg H1 of the "
            "3-hypothesis portfolio raised by failure_autopsy_V3-EXQ-794a_2026-07-31.json "
            "(H2 = training-exposure budget, a separate sibling script/queue entry; H3 = "
            "wrong drift-source mechanism form, routes to /lit-pull, not an experiment). "
            "Re-runs ONLY 794a's ARM_INFL_LO/ARM_INFL_HI cells (broadcast/Phase-7 not "
            "exercised) with use_rem_precision_recalibration flipped False (794a: True), "
            "holding every other config value identical to 794a. Decisive readout: "
            "rv_final_after_training per dose level, compared against two pre-registered "
            "EXTERNAL reference points (not derived from this run) -- 794a's own F1-ON "
            "rv_final for the same two arms, and the SD-076 headroom repair's own isolated "
            "(no-sleep-loop) validation smoke's rv_final at the identical error scale. H1 "
            "is supported at a dose level iff this run's rv_final closes at least half the "
            "pre-registered gap between those two references (closer to the smoke than to "
            "794a) AND the movement from 794a's reference clears a non-degeneracy floor. "
            "NOT governance evidence for MECH-204 or SD-076's own C1/C2 (794a's own "
            "criteria) -- this asks only which of H1/H2/H3 explains 794a's full-loop-vs-"
            "smoke shortfall. EXPERIMENT_PURPOSE=diagnostic => excluded from governance "
            "confidence/conflict scoring."
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
