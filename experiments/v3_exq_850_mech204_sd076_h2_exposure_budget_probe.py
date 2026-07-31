"""V3-EXQ-853: MECH-204/SD-076 H2 discrimination probe -- does a substantially
longer training budget close the V3-EXQ-794a full-loop-vs-repair-smoke gap?

GOV-FANOUT-1 leg. Fanout source: failure_autopsy_V3-EXQ-794a_2026-07-31.json,
target run_id v3_exq_794a_mech204_phase7_sd076_calibration_loop_2x2_20260724T063301Z_v3,
hypothesis H2 ("insufficient training exposure") of a 3-way discrimination
(H1 F1-interaction damping / H2 insufficient exposure / H3 wrong mechanism form).
A sibling leg (different script, same autopsy) probes H1 in parallel; this script
probes H2 only, per GOV-FANOUT-1's "single-axis, do not sweep the braked design"
rule. H3 is a /lit-pull commission, not an experiment (autopsy fanout_recommendation
note).

THE GAP THIS PROBE EXISTS TO EXPLAIN
-------------------------------------
V3-EXQ-794a (a same-question re-run of 794 against the repaired SD-076 headroom,
ree-v3 452f99e367) ran the full 2x3 behavioural loop at N_TRAIN_EPS=30 and reached
rv_final 0.003998 (LO, asymmetry 0.6) / 0.003870 (HI, asymmetry 0.8). The SAME
repaired mechanism, exercised in the repair's own isolated-unit validation smoke
(ree-v3/tests/contracts/test_sd076_rv_floor_headroom.py, N_STEPS=4000 direct EMA
updates over a synthetic error sequence at the identical measured error scale,
true_error_ref ~0.0037) reached rv_final 0.0025377 (LO) / 0.0021031 (HI) -- roughly
TWICE the reduction from the un-inflated operating point (0.005420) that the full
loop achieved. Both readouts are recorded: 794a's manifest
(v3_exq_794a_mech204_phase7_sd076_calibration_loop_2x2_20260724T063301Z_v3.json,
aggregates.per_level.{LO,HI}.rv_final) and the smoke's own headroom-repair
implementation_note_update in substrate_queue.json's sd_waking_confidence_inflation_
headroom entry (cross-checked directly against the pinned contract's own
REPAIRED-config assertions in test_sd076_rv_floor_headroom.py).

H2 (this leg): the full-loop's training budget (N_TRAIN_EPS=30, K=1 single-fire
sleep -- SleepLoopManager fires the F1 REM recalibration cycle once per episode via
agent.reset()) may simply not give the asymmetric EMA enough exposure to accumulate
as much drift as the smoke's more direct 4000-step update sequence reached within
its own iteration count. This is an EXPOSURE/DOSE-DURATION hypothesis, not a
mechanism-form hypothesis (H3) or an interaction hypothesis (H1: the smoke never
runs the F1/REM loop at all, so it cannot see any damping from it -- that
possibility is this leg's sibling's job, not this script's).

WHAT "K=1 SLEEP CYCLE" GOVERNS, AND WHY THE BUDGET INCREASE TARGETS N_TRAIN_EPS.
sleep_loop_episodes_K=1 makes SleepLoopManager fire the sleep cycle (SWS
consolidation + REM attribution + the F1 precision-recalibration WRITEBACK) once
per training episode, at the agent.reset() call that starts the NEXT episode (see
_read_recalib_metrics: the fired/rv_before/rv_after telemetry left by the PRIOR
episode's sleep cycle is read at the next reset). So N_TRAIN_EPS jointly controls
(a) the total number of waking environment steps the asymmetric EMA updates on
(N_TRAIN_EPS * STEPS_PER_EP), and (b) the total number of F1 recalibration firings
(one per episode, K=1) that pull rv toward the REM-computed target and could reset
part of any within-episode drift accumulation. Raising N_TRAIN_EPS raises BOTH
together, which is the correct manipulation for H2 as stated (more total exposure,
same K=1 pattern as 794a -- unlike H1's probe, which would hold N_TRAIN_EPS fixed
and instead disable the F1 recalibration lever itself).

BUDGET CHOSEN: 5x (N_TRAIN_EPS 30 -> 150), following the autopsy's own suggested
H2 probe sketch ("N_TRAIN_EPS increased substantially, e.g. 3x") at the upper end
of the 3x-5x range the routing brief asked for. Reasoning: 794a's per-cell RAW
waking-step count (30 * 200 = 6000) already exceeds the smoke's 4000-step
continuous sequence, yet only reached half the smoke's reduction -- so a small
multiplier close to parity with the smoke's own step count would not meaningfully
change the exposure regime relative to what 794a already ran. A 5x multiplier
(150 * 200 = 30000 waking steps, 7.5x the smoke's 4000) gives five times as many
post-F1-reset within-episode accumulation windows as 794a, which is the generous
side of "substantially longer" without leaving the compute budget open-ended.
COMPUTE-TIME TRADEOFF (see the proposed queue entry `note` for the derivation):
scaling only the training phase (eval phase N_EVAL_EPS=20 stays fixed, since the
decisive readout rv_final is read at the END of training, before eval begins)
means the per-cell step count grows 3.4x (10000 -> 34000 total train+eval steps),
not 5x -- and this probe runs only 2 arms x 3 seeds = 6 cells versus 794a's 18,
so the ESTIMATED total wall time (see below) is actually shorter than 794a's own
6136.5s despite the longer per-cell budget.

WHY ONLY TWO ARMS (ARM_INFL_LO / ARM_INFL_HI), NOT 794a's 2x3 FACTORIAL.
This is a single-axis discrimination probe (GOV-FANOUT-1: "never a power-bump of
the braked design"; each leg attacks ONE design axis). The Phase 7 broadcast axis
(MECH-204's correction, `use_rem_precision_broadcast`) is not manipulated here at
all -- it stays OFF in both arms, exactly as it is absent from the repair's own
isolated smoke, so the comparison against the smoke is apples-to-apples on the
SD-076 drift-source axis alone. Dropping ARM_OFF_OFF, ARM_BCAST_ONLY, ARM_BOTH_LO,
ARM_BOTH_HI is therefore not merely cheaper: including any broadcast-ON arm would
reintroduce the correction as a confound on a probe designed to isolate the SOURCE
mechanism's own exposure sufficiency. F1/REM precision recalibration
(`use_rem_precision_recalibration`, rem_precision_recalibration_step=0.25) is
UNCONDITIONAL in 794a's `_make_agent` (always True, regardless of the Phase 7
broadcast flag) and stays exactly that way here, unchanged from 794a -- it is the
axis the SIBLING (H1) leg manipulates, not this one.

DECISIVE READOUT NAMED, GOV-REUSE-1 CHECK (queue-experiment Step 2.4). Decisive
readout: rv_final_after_training at the LO and HI inflation doses, from a FULL
behavioural loop on the SD-076-headroom-repaired substrate at a training budget
materially larger than 794a's 30 episodes. Checked via
REE_assembly/scripts/reanalysis_query.py query --readout rv_final --claim SD-076
(2026-07-31): exactly two manifests carry `rv_final`/`rv_final_after_training`,
v3_exq_794_..._20260721T113848Z_v3 (pre-repair, substrate_hash
402e3f5a23a3a8e1...) and v3_exq_794a_..._20260724T063301Z_v3 (post-repair,
substrate_hash f569f39451e9746a...) -- both at N_TRAIN_EPS=30. Neither answers
"what happens at a materially larger training budget", which is the question this
leg exists to ask. The repair-validation smoke is a pytest contract, not a
manifest with a substrate_hash, so it is cited as an external reference value
(see above) rather than treated as a queryable prior run. Not recoverable -> run.

SUBSTRATE READINESS (queue-experiment Step 2.5). Every feature this script
exercises is already IMPLEMENTED and consumed identically to 794a: the SD-076
headroom repair (ree-v3 452f99e367, sd_waking_confidence_inflation_headroom,
status "implemented" in substrate_queue.json) and MECH-204's F1 REM precision
recalibration (rem_precision_recalibration_step, already load-bearing in 794a).
This probe manipulates NEITHER's implementation, only the training-episode count
-- no new substrate build is needed. claims.yaml: SD-076 is `candidate` /
`epistemic_category: standard`, no `v3_pending` / `implementation_phase: v3` gate.

RE-DERIVE BRAKE (queue-experiment Step 2.5b). failure_autopsy_V3-EXQ-794a's own
`re_derive_brake` block reports SD-076 = 0 confirmed substrate_ceiling autopsies
across the corpus (recommended_epistemic_category here is
measurement_test_design_defect, not substrate_ceiling) -- brake does not fire, and
in any case this is explicitly a GOV-FANOUT-1 diagnostic discriminating WHY a
prior result reads the way it does, which the brake's own text exempts.

CLAIM TAGGING (queue-experiment Step 3 "claim_ids accuracy rule" + the autopsy's
own claim-tagging discussion). CLAIM_IDS = ["SD-076"] ONLY, not MECH-204. This run
never engages the Phase 7 broadcast (`use_rem_precision_broadcast=False` in every
arm), so it produces no evidence at all about MECH-204's corrective mechanism --
C2-type evidence (does the broadcast correct drift) simply cannot exist without a
broadcast-ON arm. What this run CAN speak to is SD-076's own drift-source
adequacy: whether the asymmetric-EMA mechanism, given enough exposure in the real
training loop (not just the isolated smoke), reaches the same overconfident regime
the smoke demonstrated is reachable at this error scale. F1/REM recalibration
(nominally a MECH-204 substrate component) is held constant/unmanipulated here, so
even though it is technically active, its presence cannot yield MECH-204 evidence
either way -- an unmanipulated factor cannot be attributed a direction. This is a
narrower tag than 794a's blanket ["MECH-204", "SD-076"], by design: this leg tests
one axis of one claim.

EXPERIMENT_PURPOSE = "diagnostic" (matches 794/794a): discriminates WHY 794a's
full-loop result fell short of the repair's own smoke, not a governance-scoring
evidence run. Excluded from confidence/conflict scoring per convention.

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires
every episode) -- UNCHANGED from 794a; this is the substrate operating point, not
the manipulated axis (that is N_TRAIN_EPS, the count of K=1 firings).

DV-SYMMETRY INVARIANCE DECLARATION (mandatory per-arm; queue-experiment Step 3).
DV = rv_final_after_training, a scalar level of `E3TrajectorySelector.
_running_variance` read at the end of the training phase (before any eval-phase
tick). It is a SNAPSHOT of an EMA's current level, not a statistic computed over a
permutable collection -- it has no permutation symmetry to be invariant or
non-invariant under. The only relevant question is whether the manipulations
change that level:
  ARM_INFL_LO / ARM_INFL_HI (the dose axis): the asymmetric EMA's update rule
      (and hence its settling level) changes with `inflation_asymmetry`. A level
      change, not a symmetry-preserving transform. NOT invariant. OK -- identical
      reasoning to 794a's factor B declaration.
  N_TRAIN_EPS (the exposure axis, this leg's manipulation): more update
      iterations before the snapshot is taken changes an EMA's level whenever it
      has not yet converged (which is exactly what H2 asks: has it converged by
      episode 30, or does it keep moving through episode 150?). NOT invariant by
      construction -- this is the entire premise of the probe, not a confound to
      declare against.
Both arms write the SAME scalar (rv, in precision units) the smoke and 794a both
read, so the three readouts (this run, 794a, the smoke) are on a directly
comparable scale.

Both arms also carry the internal comparator `_wci_symmetric_rv_ref` (the
E3TrajectorySelector's own tracked un-inflated counterfactual EMA -- see
ree_core/predictors/e3_selector.py update_running_variance /
_apply_wci_rv_floor), read per seed at end of training. This lets the
`inflation_lowers_rv` readiness precondition be evaluated WITHOUT running an
ARM_OFF_OFF comparator arm (which would reintroduce the broadcast-free-but-
inflation-free control this single-axis design deliberately omits) -- the
substrate already computes the counterfactual it would have reached with
inflation off, alongside the real (inflated) trajectory, on every tick.
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

QUEUE_ID = "V3-EXQ-853"
EXPERIMENT_TYPE = "v3_exq_850_mech204_sd076_h2_exposure_budget_probe"
CLAIM_IDS = ["SD-076"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = (
    "K=1 single-fire (SleepLoopManager, sleep_loop_episodes_K=1, fires every episode)"
)
FANOUT_SOURCE = "failure_autopsy_V3-EXQ-794a_2026-07-31.json"
FANOUT_HYPOTHESIS = "H2-insufficient-training-exposure"
FANOUT_TARGET_RUN_ID = "v3_exq_794a_mech204_phase7_sd076_calibration_loop_2x2_20260724T063301Z_v3"

# ---- Run shape. Identical to 794a EXCEPT N_TRAIN_EPS (the manipulated axis). ----
N_TRAIN_EPS_794A = 30            # 794a's own budget, kept for the ratio/provenance record
N_TRAIN_EPS = 150                # 5x -- see module docstring "BUDGET CHOSEN"
N_EVAL_EPS = 20
N_SEEDS = 3
GRID_SIZE = 12
STEPS_PER_EP = 200
LR = 5e-4

# ---- Substrate operating point (held constant, identical to 794a). ----
SWS_CONSOLIDATION_STEPS = 8
REM_ATTRIBUTION_STEPS = 6
PRECISION_ZERO_POINT_EMA_ALPHA = 0.1
REM_PRECISION_RECALIBRATION_STEP = 0.25
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# ---- The single factor under test: SD-076 asymmetry dose. Phase 7 broadcast is
# NOT a factor here -- it is held OFF in every arm (see module docstring). ----
INFLATION_ASYMMETRY_LO = 0.6
INFLATION_ASYMMETRY_HI = 0.8
INFLATION_RV_FLOOR = 0.01
INFLATION_RV_FLOOR_RELATIVE_FRAC = 0.2   # SD-076 headroom repair, identical to 794a
INFLATION_RV_FLOOR_MODE = "soft"
INFLATION_RV_FLOOR_SOFTNESS = 0.25

# ---- Pre-registered thresholds (NOT derived from this run's own statistics). ----
PRECISION_INIT_BASELINE = 0.5    # REEConfig precision_init default
RV_LIVE_FLOOR = 1e-6             # rv_final must differ from precision_init by more than this
RECALIB_MOVE_FLOOR = 1e-4        # F1 mean per-cycle |rv_after - rv_before| floor
INFLATION_MOVE_FLOOR = 1e-4      # inflation must push rv below the internal un-inflated ref
# Same statistic and same floor as 794a's dose_levels_separated gate -- the "still
# clamped" saturation signature this probe would ALSO need to rule out before its
# own H2 readout can be trusted.
DOSE_SEPARATION_FLOOR = 1e-4

# ---- H2-specific pre-registered EXTERNAL reference values (not derived from this
# run). Both cited precisely in the module docstring's "THE GAP THIS PROBE EXISTS
# TO EXPLAIN" section. ----
FULLLOOP_794A_RV_FINAL = {
    "LO": 0.003997733405264173,   # 794a manifest aggregates.per_level.LO.rv_final
    "HI": 0.003870367272153275,   # 794a manifest aggregates.per_level.HI.rv_final
}
SMOKE_RV_FINAL = {
    # ree-v3 452f99e367 sd_waking_confidence_inflation_headroom validation smoke
    # (test_sd076_rv_floor_headroom.py, REPAIRED config, at the 794-measured error
    # scale true_error_ref ~0.0037). 7-significant-figure values as recorded in
    # substrate_queue.json's implementation_note_update.
    "LO": 0.0025377,
    "HI": 0.0021031,
}
# Fraction of the (794a -> smoke) gap this run's rv_final must close, at BOTH LO
# and HI, to read H2 as SUPPORTED (more exposure moves the substrate meaningfully
# toward the smoke's demonstrated regime).
H2_CLOSURE_SUPPORT_FLOOR = 0.30
# Fraction below which, at BOTH LO and HI, reads as a PLATEAU despite the 5x
# budget -- H2 NOT supported (the gap is not an exposure/dose-duration artifact).
H2_CLOSURE_PLATEAU_CEILING = 0.10

# (arm_id, inflation_asymmetry). Both arms have Phase 7 broadcast OFF (see module
# docstring "WHY ONLY TWO ARMS").
ARMS: Tuple[Tuple[str, float], ...] = (
    ("ARM_INFL_LO", INFLATION_ASYMMETRY_LO),
    ("ARM_INFL_HI", INFLATION_ASYMMETRY_HI),
)

# Ascending order, mirroring 794a's operative-level convention (unused here for a
# capability gate since both arms are always evaluated, but kept so per-level
# reporting reads the same way as 794a's for a human cross-reading the two runs).
INFLATION_LEVELS = (
    ("LO", INFLATION_ASYMMETRY_LO, "ARM_INFL_LO"),
    ("HI", INFLATION_ASYMMETRY_HI, "ARM_INFL_HI"),
)

# Each arm's sibling at the OTHER asymmetry level -- the matched positive control
# for the dose-separation precondition (differs ONLY in the dose).
DOSE_SIBLING = {
    "ARM_INFL_LO": "ARM_INFL_HI",
    "ARM_INFL_HI": "ARM_INFL_LO",
}


# ---------------------------------------------------------------- preconditions --
# Both arms are inflation arms with no broadcast axis, so every precondition here
# applies unconditionally (no regime conditioning needed -- contrast 794a, where
# broadcast-scoped preconditions needed `applies_to`).
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="rv_live",
        description="rv_final differs from precision_init by more than the floor "
                    "(the Q-042/530c substrate-liveness contract). Worst cell "
                    "reported.",
        control="every seed of this arm; a dead rv makes the DV meaningless",
        threshold=RV_LIVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="f1_recalib_engaged",
        description="mean per-cycle |rv_after - rv_before| from the F1 WRITEBACK "
                    "recalibration exceeds the floor, i.e. REM was entered and the "
                    "MECH-204 lever moved rv at least once. F1 recalibration is ON "
                    "in every arm, unchanged from 794a -- confirms the substrate "
                    "operating point this probe holds constant is actually live "
                    "across the LONGER training budget too.",
        control="F1 recalibration is ON in every arm of this design",
        threshold=RECALIB_MOVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="inflation_lowers_rv",
        description="mean over seeds of (wci_symmetric_rv_ref_final - "
                    "rv_final_after_training). SIGNED: SD-076 must push rv DOWN "
                    "relative to the SUBSTRATE'S OWN internally-tracked un-inflated "
                    "counterfactual (E3TrajectorySelector._wci_symmetric_rv_ref) "
                    "or it is not an inflation source. Same statistic the DV "
                    "routes on (rv level), measured against a positive control "
                    "computed by the substrate itself on every tick -- no separate "
                    "ARM_OFF_OFF arm is needed for this comparison.",
        control="each seed's own _wci_symmetric_rv_ref, tracked in parallel by the "
                "substrate on every update_running_variance call",
        threshold=INFLATION_MOVE_FLOOR,
        direction="lower",
    ),
    PreconditionSpec(
        name="dose_levels_separated",
        description="|rv_final(this arm) - rv_final(sibling arm at the OTHER "
                    "asymmetry)|. THE 794 GATE, carried forward: two nominally "
                    "different doses producing the same rv is a SATURATION "
                    "signature, not a null. Must clear before this run's own H2 "
                    "closure readout can be trusted (a still-clamped lever would "
                    "read as 'no closure' for a reason unrelated to exposure).",
        control="sibling inflation arm at the other asymmetry level, same seeds "
                "-- differs only in the dose",
        threshold=DOSE_SEPARATION_FLOOR,
        direction="lower",
    ),
)


def _arm_ctx(arm_id: str, asym: float) -> Dict[str, object]:
    return {"arm_id": arm_id, "asymmetry": asym}


ARM_CONTEXTS = [_arm_ctx(a, x) for (a, x) in ARMS]


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
        # Phase 7 broadcast is NOT a factor in this probe -- held OFF in every arm.
        use_rem_precision_broadcast=False,
        rem_precision_broadcast_gain=0.0,
    )
    # SD-076 waking confidence inflation -- the single manipulated axis.
    cfg.e3.use_waking_confidence_inflation = True
    cfg.e3.waking_confidence_inflation_asymmetry = float(asym)
    cfg.e3.waking_confidence_rv_floor = INFLATION_RV_FLOOR
    cfg.e3.waking_confidence_rv_floor_relative_frac = INFLATION_RV_FLOOR_RELATIVE_FRAC
    cfg.e3.waking_confidence_rv_floor_mode = INFLATION_RV_FLOOR_MODE
    cfg.e3.waking_confidence_rv_floor_softness = INFLATION_RV_FLOOR_SOFTNESS
    # Tonic 5-HT must be on for compute_recalibration_target() to be meaningful (the
    # F1 WRITEBACK reads it every recalibration cycle).
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def _arm_config_slice(asym: float, n_train: int) -> Dict:
    """The config the cell's build+collect path actually reads."""
    return {
        "grid_size": GRID_SIZE,
        "steps_per_ep": STEPS_PER_EP,
        "n_train_eps": n_train,
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
        "use_rem_precision_broadcast": False,
        "rem_precision_broadcast_gain": 0.0,
        "use_waking_confidence_inflation": True,
        "waking_confidence_inflation_asymmetry": float(asym),
        "waking_confidence_rv_floor": INFLATION_RV_FLOOR,
        "waking_confidence_rv_floor_relative_frac": INFLATION_RV_FLOOR_RELATIVE_FRAC,
        "waking_confidence_rv_floor_mode": INFLATION_RV_FLOOR_MODE,
        "waking_confidence_rv_floor_softness": INFLATION_RV_FLOOR_SOFTNESS,
        "precision_init_baseline": PRECISION_INIT_BASELINE,
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
    arm_label, asym = arm

    with arm_cell(
        seed,
        config_slice=_arm_config_slice(asym, n_train),
        script_path=Path(__file__),
        include_driver_script_in_hash=False,  # mint-as-you-go: cross-driver reusable
    ) as cell:
        env = _make_env(seed, dry_run=dry_run)
        agent = _make_agent(env, asym)
        optimizer = optim.Adam(agent.parameters(), lr=LR)

        print(f"Seed {seed} Condition {arm_label} n_train={n_train}", flush=True)

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
            if (ep + 1) % 25 == 0 or ep + 1 == n_train:
                print(
                    f"  [train] arm={arm_label} seed={seed} ep {ep + 1}/{n_train} "
                    f"rv={float(agent.e3._running_variance):.6f} "
                    f"ref={float(agent.e3._wci_symmetric_rv_ref):.6f} "
                    f"prec={float(agent.e3.current_precision):.4f}",
                    flush=True,
                )

        rv_after_training = float(agent.e3._running_variance)
        # The substrate's own internally-tracked un-inflated counterfactual -- the
        # inflation_lowers_rv precondition's positive control (see module docstring).
        wci_symmetric_rv_ref_after_training = float(agent.e3._wci_symmetric_rv_ref)

        # ---- Eval: capture confidence (rv) and accuracy (real forward-model error),
        # recorded for context/comparability with 794a even though this probe's
        # decisive readout is rv_after_training, not the eval-phase overconfidence
        # score. ----
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

        if true_error_ref > 1e-9 and mean_rv > 1e-9:
            calibration_ratio = mean_rv / true_error_ref
            overconfidence_score = float(np.log(true_error_ref / mean_rv))
        else:
            calibration_ratio = float("nan")
            overconfidence_score = 0.0

        absolutely_overconfident = overconfidence_score > 0.10
        print(
            f"  [eval] arm={arm_label} seed={seed} score={overconfidence_score:+.4f} "
            f"calib_ratio={calibration_ratio:.3f} true_err={true_error_ref:.6f} "
            f"mean_rv={mean_rv:.6f} rv_final={rv_after_training:.6f} "
            f"wci_ref={wci_symmetric_rv_ref_after_training:.6f}",
            flush=True,
        )

        row = {
            "arm_id": arm_label,
            "seed": seed,
            "inflation_asymmetry": float(asym),
            "n_train_eps": n_train,
            "overconfidence_score": overconfidence_score,
            "calibration_ratio": calibration_ratio,
            "true_error_ref": true_error_ref,
            "mean_running_variance": mean_rv,
            "rv_final_after_training": rv_after_training,
            "wci_symmetric_rv_ref_after_training": wci_symmetric_rv_ref_after_training,
            "rv_delta_from_precision_init": abs(rv_after_training - PRECISION_INIT_BASELINE),
            "recalib_cycles_fired": recalib_fired,
            "recalib_mean_abs_move": _mean(recalib_moves),
            "absolutely_overconfident": absolutely_overconfident,
            "n_eval_ticks": len(rv_vals),
            "n_pe_ticks": len(pe_vals),
        }
        cell.stamp(row)
    return row


# ---------------------------------------------------------------------- analysis --
def _closure_fraction(this_run_rv: float, level: str) -> float:
    """Fraction of the (794a -> smoke) gap this run's rv_final has closed.

    0.0 = exactly reproduces 794a (no movement); 1.0 = exactly reaches the smoke's
    value; >1.0 = overshoots past the smoke's value; negative = moved AWAY from the
    smoke's value (rv rose relative to 794a). rv is a "lower = more overconfident"
    scale, so the gap is (794a_rv - smoke_rv), both positive since 794a's rv sat
    above the smoke's.
    """
    baseline = FULLLOOP_794A_RV_FINAL[level]
    target = SMOKE_RV_FINAL[level]
    gap = baseline - target
    if abs(gap) < 1e-12:
        return 0.0
    return float((baseline - this_run_rv) / gap)


def _analyse(cells: List[Dict], seeds: List[int]) -> Dict:
    by_arm: Dict[str, Dict[int, Dict]] = {}
    for c in cells:
        by_arm.setdefault(c["arm_id"], {})[c["seed"]] = c

    arm_rv = {a: _mean([by_arm[a][s]["rv_final_after_training"] for s in seeds])
              for a in by_arm}
    arm_score = {a: _mean([by_arm[a][s]["overconfidence_score"] for s in seeds])
                 for a in by_arm}
    arm_ratio = {a: _mean([by_arm[a][s]["calibration_ratio"] for s in seeds])
                 for a in by_arm}
    arm_true_err = {a: _mean([by_arm[a][s]["true_error_ref"] for s in seeds])
                    for a in by_arm}

    # ---- readiness gates (both arms unconditionally in scope; no regime
    # conditioning needed -- see PRECONDITION_SPECS comment). ----
    arm_gates = []
    for (arm_id, asym) in ARMS:
        ctx = _arm_ctx(arm_id, asym)
        sibling = DOSE_SIBLING[arm_id]
        measured: Dict[str, float] = {
            "rv_live": min(by_arm[arm_id][s]["rv_delta_from_precision_init"] for s in seeds),
            "f1_recalib_engaged": _mean(
                [by_arm[arm_id][s]["recalib_mean_abs_move"] for s in seeds]),
            "inflation_lowers_rv": _mean(
                [by_arm[arm_id][s]["wci_symmetric_rv_ref_after_training"]
                 - by_arm[arm_id][s]["rv_final_after_training"] for s in seeds]),
            "dose_levels_separated": abs(
                _mean([by_arm[arm_id][s]["rv_final_after_training"] for s in seeds])
                - _mean([by_arm[sibling][s]["rv_final_after_training"] for s in seeds])),
        }
        arm_gates.append(
            evaluate_arm_gate(arm_id, ctx, list(PRECONDITION_SPECS), measured))

    gate = aggregate_arm_gates(arm_gates)

    # ---- per-level H2 closure readout ----
    per_level: Dict[str, Dict] = {}
    for (lvl, asym, infl_arm) in INFLATION_LEVELS:
        rv_this_run = arm_rv[infl_arm]
        closure = _closure_fraction(rv_this_run, lvl)
        per_level[lvl] = {
            "asymmetry": asym,
            "infl_arm": infl_arm,
            "rv_final": rv_this_run,
            "rv_final_794a": FULLLOOP_794A_RV_FINAL[lvl],
            "rv_final_smoke": SMOKE_RV_FINAL[lvl],
            "closure_fraction": closure,
            "closes_meaningfully": bool(closure >= H2_CLOSURE_SUPPORT_FLOOR),
            "plateaus": bool(closure < H2_CLOSURE_PLATEAU_CEILING),
            "n_seeds_overconfident": sum(
                1 for s in seeds if by_arm[infl_arm][s]["absolutely_overconfident"]),
            "infl_score": arm_score[infl_arm],
        }

    # C1 (load-bearing): H2 is SUPPORTED iff the extended budget closes a
    # meaningful fraction of the 794a-vs-smoke gap at BOTH doses. A single dose
    # closing while the other does not is a genuinely mixed/ambiguous result, not
    # a clean H2 confirmation (per-level readouts remain fully visible either way).
    c1_h2_supported = bool(
        per_level["LO"]["closes_meaningfully"] and per_level["HI"]["closes_meaningfully"])
    # C2 (load-bearing): H2 is NOT SUPPORTED (a plateau) iff BOTH doses stay below
    # the plateau ceiling despite the 5x budget -- the gap is not exposure-driven.
    c2_h2_plateau = bool(
        per_level["LO"]["plateaus"] and per_level["HI"]["plateaus"])
    # C3 (diagnostic, non-load-bearing): dose-response direction preserved (more
    # asymmetry -> more overconfidence / lower rv), same check as 794a's C5.
    c3_dose_response_monotone = bool(
        per_level["HI"]["rv_final"] < per_level["LO"]["rv_final"])

    criteria = [
        {"name": "C1_extended_exposure_closes_smoke_gap", "load_bearing": True,
         "passed": c1_h2_supported,
         "closure_lo": per_level["LO"]["closure_fraction"],
         "closure_hi": per_level["HI"]["closure_fraction"],
         "support_floor": H2_CLOSURE_SUPPORT_FLOOR},
        {"name": "C2_extended_exposure_plateaus", "load_bearing": True,
         "passed": c2_h2_plateau,
         "closure_lo": per_level["LO"]["closure_fraction"],
         "closure_hi": per_level["HI"]["closure_fraction"],
         "plateau_ceiling": H2_CLOSURE_PLATEAU_CEILING},
        {"name": "C3_dose_response_monotone", "load_bearing": False,
         "passed": c3_dose_response_monotone,
         "lo_rv": per_level["LO"]["rv_final"], "hi_rv": per_level["HI"]["rv_final"]},
    ]

    # ---- non-degeneracy, keyed to the owning arm's readiness gate ----
    criteria_by_arm = {
        "ARM_INFL_LO": ["C1_extended_exposure_closes_smoke_gap",
                        "C2_extended_exposure_plateaus",
                        "C3_dose_response_monotone"],
        "ARM_INFL_HI": ["C1_extended_exposure_closes_smoke_gap",
                        "C2_extended_exposure_plateaus",
                        "C3_dose_response_monotone"],
    }
    # C1/C2/C3 all read BOTH arms jointly, so they are only non-degenerate if
    # BOTH arms are green (a criterion owned by two arms is degenerate if either
    # is red -- arm_criteria_non_degenerate keys per-arm, so intersect by hand).
    both_green = bool(gate["all_green"])
    raw_non_degen = {
        "C1_extended_exposure_closes_smoke_gap": both_green,
        "C2_extended_exposure_plateaus": both_green,
        "C3_dose_response_monotone": both_green,
    }
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {"ARM_INFL_LO": list(raw_non_degen.keys())}, gate, raw_non_degen)

    # ---- self-route ----
    readiness_ok = bool(gate["non_degenerate"]) and both_green
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "inconclusive"
    elif c1_h2_supported:
        label = "h2_training_exposure_closes_smoke_gap"
        outcome = "PASS"
        direction = "supports"
    elif c2_h2_plateau:
        label = "h2_training_exposure_insufficient_explanation_plateau"
        outcome = "FAIL"
        direction = "inconclusive"
    else:
        label = "h2_training_exposure_partial_ambiguous"
        outcome = "FAIL"
        direction = "inconclusive"

    per_claim = {"SD-076": direction if readiness_ok else "unknown"}

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
        "readiness_ok": readiness_ok,
        "thresholds": {
            "H2_CLOSURE_SUPPORT_FLOOR": H2_CLOSURE_SUPPORT_FLOOR,
            "H2_CLOSURE_PLATEAU_CEILING": H2_CLOSURE_PLATEAU_CEILING,
            "DOSE_SEPARATION_FLOOR": DOSE_SEPARATION_FLOOR,
            "RV_LIVE_FLOOR": RV_LIVE_FLOOR,
            "RECALIB_MOVE_FLOOR": RECALIB_MOVE_FLOOR,
            "INFLATION_MOVE_FLOOR": INFLATION_MOVE_FLOOR,
            "PRECISION_INIT_BASELINE": PRECISION_INIT_BASELINE,
        },
        "reference_values": {
            "fullloop_794a_rv_final": FULLLOOP_794A_RV_FINAL,
            "smoke_rv_final": SMOKE_RV_FINAL,
        },
    }


# -------------------------------------------------------------------------- main --
def run_experiment(dry_run: bool = False) -> Dict:
    t0 = time.perf_counter()
    n_train = 4 if dry_run else N_TRAIN_EPS
    n_eval = 1 if dry_run else N_EVAL_EPS
    n_seeds = 2 if dry_run else N_SEEDS
    steps = 20 if dry_run else STEPS_PER_EP
    seeds = list(range(n_seeds))

    # Design-time proof: this probe's whole point is a SUBSTANTIALLY larger budget
    # than 794a -- catch a copy-paste regression back to 794a's own N_TRAIN_EPS
    # before any compute is spent (the class of bug the queue-experiment skill's
    # "no copy-and-modify fast path" warning names: hardcoded loop denominators
    # that don't match the new run's intent).
    if not dry_run:
        assert N_TRAIN_EPS >= 3 * N_TRAIN_EPS_794A, (
            f"H2 probe requires N_TRAIN_EPS >= 3x 794a's budget "
            f"({3 * N_TRAIN_EPS_794A}); got {N_TRAIN_EPS}"
        )

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
    print(f"label={adj['label']} outcome={outcome} readiness_ok={adj['readiness_ok']}",
          flush=True)
    for lvl in ("LO", "HI"):
        pl = adj["per_level"][lvl]
        print(f"  {lvl:<3} asym={pl['asymmetry']:.2f} rv_final={pl['rv_final']:.6f} "
              f"(794a={pl['rv_final_794a']:.6f} smoke={pl['rv_final_smoke']:.6f}) "
              f"closure={pl['closure_fraction']:+.3f}", flush=True)
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
        "n_train_eps_794a_baseline": N_TRAIN_EPS_794A,
        "n_eval_eps": adj["config_n"]["n_eval_eps"],
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
        "dose_key": "asymmetry",
        "fanout_source": FANOUT_SOURCE,
        "fanout_hypothesis": FANOUT_HYPOTHESIS,
        "fanout_target_run_id": FANOUT_TARGET_RUN_ID,
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
            "readiness_ok": adj["readiness_ok"],
        },
        "thresholds": adj["thresholds"],
        "reference_values": adj["reference_values"],
        "arm_results": adj["cells"],
        "per_seed_cells": adj["cells"],
        "elapsed_seconds": adj["elapsed_seconds"],
        "notes": (
            "DIAGNOSTIC GOV-FANOUT-1 H2 leg (exposure/world axis) discriminating why "
            "V3-EXQ-794a's full-loop rv_final (0.003998 LO / 0.003870 HI) reached only "
            "about half the reduction from baseline that the SD-076 headroom repair's "
            "own isolated validation smoke demonstrated (rv_final 0.0025377 LO / "
            "0.0021031 HI) at the identical measured error scale. Tests whether a "
            "substantially longer training budget (N_TRAIN_EPS 30 -> 150, 5x) closes "
            "that gap -- an exposure/dose-duration hypothesis (H2), distinct from the "
            "sibling leg's F1-interaction-damping hypothesis (H1) and the driver's own "
            "pre-registered wrong-mechanism-form fallback (H3, weakest-supported per "
            "the parent autopsy). Single-axis: only ARM_INFL_LO/ARM_INFL_HI run, Phase "
            "7 broadcast held OFF throughout (not this leg's manipulated axis), F1/REM "
            "precision recalibration held ON unchanged from 794a. DIAGNOSTIC => "
            "excluded from governance confidence/conflict scoring. claim_ids=[SD-076] "
            "only -- MECH-204's broadcast correction is never engaged in this design, "
            "so it yields no MECH-204 evidence either way (see module docstring "
            "'CLAIM TAGGING')."
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
