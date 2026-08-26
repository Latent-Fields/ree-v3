"""
V3-EXQ-603v -- MECH-357 avoidance-efficacy ELIGIBILITY-TRACE REPAIR validation
(diagnostic; claim_ids=["MECH-357"]).

PURPOSE (diagnostic -- instrument repair, NOT a claim retest):
Validate substrate_queue entry `mech357-avoidance-efficacy-eligibility-trace-
imbalance` (status `implemented`, ree-v3 93d5d98b80, 2026-08-16) against its OWN
pre-registered failure_record target. That entry's implementation_note_update
states plainly: "NOT yet verified against this entry's own failure_record target
(a rerun of V3-EXQ-603u-shape scoring-window gate_live check on >= 2/3 INTACT
seeds) -- that empirical re-validation is separate work and the failure_record
below is deliberately left resolved=open until a real rerun confirms it." This
run IS that rerun. No such run exists: no 603v+ manifest, and the only later
MECH-357 work (chip-20260825-mech357-h2-reanalysis) reanalysed the EXISTING
603s/t/u data, i.e. the broken-trace substrate.

THE DEFECT BEING REPAIRED. `infralimbic_avoidance_gate.update()` credits
avoidance_efficacy only when a DIRECTED action under threat drops z_harm_a
(rare), while it decayed on every OTHER under-threat tick (common) -- including
freeze/no-op ticks, which are the ABSENCE of an avoidance attempt, not a failed
one. The per-event rates (learn_rate=0.05 vs leak_rate=0.02) are not the
problem; the TICK-COUNT asymmetry is. Measured on V3-EXQ-603u (this exact
config, 3 INTACT seeds): decay:credit 60.9:1 / 130.7:1 / 81.8:1 over 22-25k
decay events per cell, driving the learned trace from a healthy in-run peak of
0.659 / 0.936 / 0.832 down to a last-10-episode median of 6.5e-26 / 3.9e-29 /
3.6e-24 -- numerical zero, ~24 orders of magnitude below its own peak.

WHY THAT MATTERS AND IS NOT COSMETIC. The gate consumes
effective_efficacy = max(avoidance_efficacy, scaffold_floor), and the protective
scaffold floor ANNEALS 0.8 -> 0.0 across Stage-H. So the underflow is unmasked
EXACTLY as the scoring window opens: the INTACT arm is functionally LESIONED
where the DV measures it, while whole-run readiness aggregates still report the
gate engaged and suppressing at 1.0. Every MECH-357 Stage-H run to date
(603h/k/r/s/t/u) was scored through that window.

THE FIX UNDER TEST (unconditional, no flag -- so there is no within-run OFF arm
and none is needed). Credit-eligibility windowing, candidate (a) of the
substrate entry's implementation_hint: a freeze/no-op tick under threat no
longer decays avoidance_efficacy; directed attempts that FAIL still decay,
unchanged. A `mech357_n_freeze_noop` counter records how many ticks the old rule
would have charged as decay.

DESIGN -- 2 arms x 3 seeds [42,43,44], ABSOLUTE pre-registered trace gate.
Config is bit-for-bit 603u's (agent-directed pursuit: env_drift_interval=1,
env_drift_prob=0.6, hazard_agent_pursuit=0.9), so the trace comparison against
603u's recorded decay:credit ratios is apples-to-apples and the ONLY changed
variable is the substrate fix.

  ARM_INTACT_midline   (use_ia=True, driver=True, midline spawn) -- THE ARM THE
      pre-registered target names. Load-bearing criterion C1 is scored here.
  ARM_POSCTRL_reefspawn (use_ia=True, driver=True, reef spawn) -- an independent
      REPLICATION of the same trace readout in a different spawn regime (603u
      recorded 3.99e-22 / 1.85e-13 / 2.21e-27 there). Non-load-bearing.

ARM_LESION IS DELIBERATELY DROPPED, and this is a design decision worth stating
rather than an omission. The gate is OFF on that arm, so it carries no learned
trace at all (603u recorded per_seed_avoidance_efficacy = [0.0, 0.0, 0.0]) and
the fix cannot change it -- re-running it would spend ~6.4h of cloud compute to
reproduce a result that is invariant by construction. Its 603u values remain
citable as-is.

THIS RUN DOES NOT ATTEMPT THE MECH-357 DISCRIMINATION, ON PURPOSE. 603u measured
G_H_LESION_frac = 1.0 (3/3 seeds cleared the survival gate) alongside
G_H_INTACT_frac = 1.0, and self-routed
`pressure_insufficient_lesion_ceiling_requeue`: the negative control has NO
headroom from below, so an INTACT-vs-LESION contrast is structurally vacuous at
this pressure REGARDLESS of the trace. That is a separate, already-identified
pressure-calibration problem (603u's own route note: re-calibrate UP --
env_drift_prob / num_hazards). Folding a pressure recalibration into this run
would confound the trace repair with a changed threat regime and put a second
~19h run at risk of returning inconclusive on both questions. Trace repair
first; pressure recalibration is owed separately and is reported as a finding.

DV-SYMMETRY DECLARATION (per arm, per the 604c rule).
  ARM_INTACT / ARM_POSCTRL -- DV is the MEDIAN of the learned
  `mech357_avoidance_efficacy` over the last SCORING_WINDOW_EPISODES Stage-H
  episodes, a magnitude on a bounded [0,1] scale. Symmetry group of the DV:
  none under which the manipulation is invariant. The manipulation is a change
  to the very update rule that GENERATES the DV (it removes a class of decay
  events from the recurrence), so it is not a broadcast additive constant
  (the DV is not an argmax/softmax readout), not a monotone rescaling (the DV
  is a magnitude, not a rank), and not a permutation of interchangeable units
  (the DV is a time-ordered window, not a set aggregate). A no-op fix would
  leave the DV at its 603u value; there is no arithmetic identity forcing the
  measured delta either way.

PRE-REGISTERED GATES (constants below, fixed before the run).
  C1 (LOAD-BEARING, ARM_INTACT): median avoidance_efficacy over the LAST 10
     Stage-H episodes >= TRACE_LIVE_FLOOR (0.01) on >= 2/3 seeds. This is the
     substrate entry's failure_record target, made scoring-window-scoped rather
     than a whole-run aggregate exactly as that target demands. 603u measured
     ~1e-24..1e-29 here, so the floor sits ~22 orders of magnitude above the
     observed failure -- there is no ambiguous band.
  C2 (secondary, NOT load-bearing, ARM_INTACT): window median >=
     TRACE_PEAK_FRACTION (0.1) x the cell's own in-run peak -- the target's
     "within an order of magnitude of its early-episode range" clause, stated
     as a ratio so it is scale-free per seed.
  C1R (replication, NOT load-bearing, ARM_POSCTRL): C1 recomputed on POSCTRL.

READINESS (the same-statistic positive control the 643 rule requires).
  R3 asserts the SAME statistic C1 routes on -- avoidance_efficacy -- on a
  positive control: the cell's in-run PEAK must reach TRACE_PEAK_READY_FLOOR
  (0.05). 603u cleared this comfortably (0.659-0.970) on the BROKEN substrate,
  so it is known-achievable and independent of the fix. If even the peak is
  below floor the learner never moved the trace at all and the run self-routes
  `substrate_not_ready_requeue` -- never a verdict label.

The per-arm gate is evaluated with experiments/_lib/precondition_gate.py, so a
red POSCTRL can NEVER vacate a green INTACT (the V3-EXQ-785 whole-run-AND
defect).

SLEEP DRIVER: not applicable (waking goal-pipeline onboarding scheduler; no
sleep loop is enabled in any arm).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_603v_mech357_eligibility_trace_repair_validation"
QUEUE_ID = "V3-EXQ-603v"
CLAIM_IDS: List[str] = ["MECH-357"]
# DIAGNOSTIC: this run validates an INSTRUMENT REPAIR (the eligibility-trace
# credit/decay imbalance), not the MECH-357 hypothesis itself. Diagnostics are
# excluded from governance confidence/conflict scoring -- correct here, because a
# repaired instrument is a precondition for a MECH-357 verdict, not a verdict.
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror 603r/603h exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Budgets (mirror 603r/603h full budget for apples-to-apples comparison).
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

# Isolated hazard-avoidance Stage-H (density unchanged from 603r/603h so hazard
# MOBILITY is the isolated new variable).
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = False  # midline for LESION/INTACT; POSCTRL overrides
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- THE FREEZE-INCOMPATIBLE PRESSURE: mobile predators + AGENT-DIRECTED PURSUIT.
# 603u KEEPS 603s's mobile-predator drift (interval=1/prob=0.6, hazards attempt a
# drift EVERY step with 0.6 per-hazard probability) so hazards still move often,
# and ADDS agent-directed pursuit (below): the SINGLE change from 603s is that a
# moving hazard now biases toward the agent's current cell instead of drifting at
# random. The reef refuge (hazard-excluded under drift) remains reachable. ---
HAZARD_STAGE_ENV_DRIFT_INTERVAL = 1
HAZARD_STAGE_ENV_DRIFT_PROB = 0.6
# Agent-directed pursuit (V3-EXQ-603u): per-drift-tick probability a moving hazard
# sorts its step toward the agent's CURRENT cell (Manhattan-nearest) rather than a
# random shuffle. Built into CausalGridWorldV2 (39b5ca8), threaded through Stage-H
# this session. 0.9 = strong: when a hazard moves, it chases 90% of the time. This
# is what removes 603s's spawn-lottery confound (undirected drift never targeted
# the agent -> the exact tie); a pursuing hazard closes distance unless the agent
# actively escapes to reef, forcing the sustained Pavlovian-instrumental conflict.
HAZARD_STAGE_HAZARD_AGENT_PURSUIT = 0.9

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603r).
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (unchanged from 603r) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# --- FIX 1: SD-059/MECH-358 escape-affordance bridge (603j/603l/603q config,
# applied to ALL arms -- orthogonal to the ilPFC gate under test) ---
ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

# --- FIX 2: harm-pathway training amend (603k/603q config, applied to ALL
# arms -- all need a non-degenerate harm-cost signal regardless of the gate) ---
HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250

# Pre-registered gates (constants).
STAGE0_ZGOAL_GATE = 0.4
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
# Seed-fraction thresholds are compared with a STRICT ">" by
# precondition_gate.met_for and with ">=" by the indexer's own recompute. With 3
# seeds the attainable fractions are {0, 1/3, 2/3, 1}, so a bare MIN_FRACTION
# threshold would read "2 of 3 seeds" as UNMET under the strict form. Nudging the
# threshold below the bound makes both conventions agree. (603u used the same
# 1e-9 device on its headroom precondition.)
MIN_FRACTION_EPS = MIN_FRACTION - 1e-9

# --- PRE-REGISTERED trace-viability gates (fixed before the run) ---
# Scoring window = the last N Stage-H episodes, i.e. exactly where the annealing
# scaffold floor has reached ~0 and the LEARNED trace is what the gate consumes.
SCORING_WINDOW_EPISODES = 10
# C1 load-bearing floor. 603u measured window medians of ~1e-24..1e-29 on this
# same config, against in-run peaks of 0.659-0.936; 0.01 sits ~22 orders above
# the observed failure and ~1 order below the low end of 603u's early-episode
# range (0.018-0.074), so it satisfies the failure_record's "non-underflowed,
# ideally within an order of magnitude of its early-episode range" wording
# without resting on a knife-edge.
TRACE_LIVE_FLOOR = 0.01
# C2 secondary: window median as a fraction of the cell's OWN in-run peak --
# the "within an order of magnitude" clause, scale-free per seed.
TRACE_PEAK_FRACTION = 0.1
# R3 readiness positive control, asserted on the SAME statistic C1 routes on.
# 603u cleared this on the BROKEN substrate (peaks 0.659-0.970), so a below-floor
# reading here means the learner never moved the trace at all -> not-ready.
TRACE_PEAK_READY_FLOOR = 0.05

# 2 arms, both gate-ON (the only arms that carry a learned trace). ARM_LESION is
# deliberately dropped -- gate OFF => no trace => invariant under this fix; see
# the module docstring.
ARM_INTACT = "ARM_INTACT_midline_pag_gate_combined_fix"
ARM_POSCTRL = "ARM_POSCTRL_reefspawn_pag_gate_combined_fix"
ARMS = [
    {"label": ARM_INTACT, "use_ia": True, "driver": True, "reef_spawn": False},
    {"label": ARM_POSCTRL, "use_ia": True, "driver": True, "reef_spawn": True},
]

_ZG = ZGoalStreamAccumulator()


def _make_scaffold_cfg(dry_run: bool, arm: Dict[str, Any]) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # The isolated Stage-H (density unchanged from 603g/603h/603r).
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        # POSCTRL spawns in the reef refuge (adjacent to safety); LESION/INTACT
        # spawn at the midline and must navigate to it.
        scaffold_hazard_stage_spawn_in_reef_half=bool(arm["reef_spawn"]),
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # THE FREEZE-INCOMPATIBLE PRESSURE: mobile predators (603s) + agent-directed
        # pursuit (603u -- the single new variable over 603s).
        scaffold_hazard_stage_env_drift_interval=HAZARD_STAGE_ENV_DRIFT_INTERVAL,
        scaffold_hazard_stage_env_drift_prob=HAZARD_STAGE_ENV_DRIFT_PROB,
        scaffold_hazard_stage_hazard_agent_pursuit=HAZARD_STAGE_HAZARD_AGENT_PURSUIT,
        # SD-058 / MECH-357 avoidance-learning driver (INTACT + POSCTRL only).
        scaffold_avoidance_driver_enabled=bool(arm["driver"]),
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE (all arms): feed the env harm stream (unchanged from 603r).
        scaffold_feed_harm_stream=True,
        # FIX 2 (all arms): harm-pathway training amend (603k/603q).
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, use_ia: bool) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        # MECH-279 PAG freeze-gate (ALL arms) -- unchanged from 603r.
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        # SD-058 / MECH-357 instrumental-avoidance gate (INTACT + POSCTRL only).
        use_instrumental_avoidance=bool(use_ia),
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # FIX 1 (all arms): SD-059/MECH-358 escape-affordance bridge (603j/603l/603q).
        use_escape_affordance_bridge=True,
        use_escape_relief_credit=True,
        use_escape_safety_credit=True,
        escape_threat_floor=ESCAPE_THREAT_FLOOR,
        escape_threat_ref=ESCAPE_THREAT_REF,
        escape_approach_gain=ESCAPE_APPROACH_GAIN,
        escape_bias_scale=ESCAPE_BIAS_SCALE,
        escape_use_trained_safety_signal=True,
        escape_safety_signal_threshold=ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        use_suffering_derivative_comparator=True,
        use_conditioned_safety_store=True,
        use_contextual_safety_terrain=True,
        e2_action_contrastive_enabled=True,
        # FIX 2 (all arms): harm-pathway training needs SD-010 z_harm_s +
        # E2_harm_s forward wired at the REEConfig level (603k/603q).
        use_harm_stream=True,
        use_e2_harm_s_forward=True,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _config_slice(arm: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """Content-addressed config slice for the per-cell arm fingerprint."""
    return {
        "arm": arm["label"],
        "use_instrumental_avoidance": bool(arm["use_ia"]),
        "scaffold_avoidance_driver_enabled": bool(arm["driver"]),
        "spawn_in_reef_half": bool(arm["reef_spawn"]),
        "use_pag_freeze_gate": True,
        "pag_theta_freeze": PAG_THETA_FREEZE,
        "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
        "avoidance_threat_ref": AVOIDANCE_THREAT_REF,
        "feed_harm_stream": True,
        "avoidance_scaffold_floor_start": AVOIDANCE_SCAFFOLD_FLOOR_START,
        "avoidance_scaffold_floor_end": AVOIDANCE_SCAFFOLD_FLOOR_END,
        "freeze_incompatible_pressure": {
            "env_drift_interval": HAZARD_STAGE_ENV_DRIFT_INTERVAL,
            "env_drift_prob": HAZARD_STAGE_ENV_DRIFT_PROB,
            "hazard_agent_pursuit": HAZARD_STAGE_HAZARD_AGENT_PURSUIT,
        },
        "escape_bridge_fix": {
            "use_escape_affordance_bridge": True,
            "escape_threat_floor": ESCAPE_THREAT_FLOOR,
            "escape_threat_ref": ESCAPE_THREAT_REF,
            "escape_approach_gain": ESCAPE_APPROACH_GAIN,
            "escape_bias_scale": ESCAPE_BIAS_SCALE,
            "escape_use_trained_safety_signal": True,
        },
        "harm_pathway_fix": {
            "scaffold_train_harm_pathway": True,
            "scaffold_harm_pathway_lr": HARM_PATHWAY_LR,
            "scaffold_harm_pathway_encoder_lr": HARM_PATHWAY_ENCODER_LR,
            "scaffold_harm_pathway_warmup_steps": HARM_PATHWAY_WARMUP_STEPS,
            "use_harm_stream": True,
            "use_e2_harm_s_forward": True,
        },
        "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, P2_BUDGET, TRAIN_STEPS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "dry_run": bool(dry_run),
    }


def _aborted_record(arm_label: str, seed: int, stage: str, reason: str,
                    s0_peak: float = 0.0) -> Dict[str, Any]:
    return {
        "arm": arm_label, "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "hazard_stage_survival_pass": False,
        "hazard_stage_median_last_window": 0.0,
        "p1_survival_pass": False,
        "p2_contact_rate": 0.0,
        "g0_stage0_zgoal": bool(s0_peak > STAGE0_ZGOAL_GATE),
        "g1_p1_survival": False,
        "g2_p2_contact": False,
        "g_h_hazard_survival": False,
        "avoidance_gate_state": {},
        "avoidance_efficacy_trajectory": [],
        "pag_n_commits": 0,
        "pag_n_releases": 0,
        "reached_hazard_stage": stage not in ("stage0", "stage0b", "p0"),
        "reached_p1": False,
        "reached_p2": False,
        "seed_pass": False,
    }


def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool,
                  total_eps: int) -> Dict[str, Any]:
    """Full curriculum for one (arm, seed) cell. arm_cell resets all RNG on
    enter (order-independent) and stamps the fingerprint on the returned row."""
    with arm_cell(
        seed,
        config_slice=_config_slice(arm, dry_run),
        script_path=Path(__file__),
        config_slice_declared=True,
    ) as cell:
        scaffold_cfg = _make_scaffold_cfg(dry_run, arm)
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env, arm["use_ia"])).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

        print(f"Seed {seed} Condition {arm['label']}", flush=True)

        def _gate_state() -> Dict[str, Any]:
            g = getattr(agent, "instrumental_avoidance", None)
            return g.get_state() if g is not None else {}

        def _pag_state() -> Dict[str, Any]:
            p = getattr(agent, "pag_freeze_gate", None)
            return dict(p.diagnostics) if p is not None else {}

        # Stage 0 -- forced-benefit nursery (goal-formation positive control).
        s0 = scheduler.run_stage0_nursery(agent, device)
        done = s0.n_episodes
        print(f"  [train] stage0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0", s0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            _ZG.observe(agent)
            return rec

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0b", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0b", s0b.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            _ZG.observe(agent)
            return rec

        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
              flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=p0", flush=True)
            rec = _aborted_record(arm["label"], seed, "p0", p0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            _ZG.observe(agent)
            return rec

        # Stage-H -- ISOLATED HAZARD-AVOIDANCE under the agent-directed-pursuit field
        # (mobile drift + hazard_agent_pursuit; the SD-058/MECH-357 driver target).
        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        gate_after_h = _gate_state()
        pag_after_h = _pag_state()
        eff_traj = list(getattr(hz, "avoidance_efficacy_trajectory", []) or [])
        print(f"  [train] hazard_avoidance {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={hz.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
              f" pag_commits={pag_after_h.get('n_commits', 0)}"
              f" eff={gate_after_h.get('mech357_avoidance_efficacy', 0.0):.4f}"
              f" n_credit={gate_after_h.get('mech357_n_credit', 0)}"
              f" n_freeze_suppr={gate_after_h.get('mech357_n_freeze_suppressed', 0)}",
              flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=hazard", flush=True)
            rec = _aborted_record(arm["label"], seed, "hazard", hz.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            rec["avoidance_gate_state"] = gate_after_h
            rec["avoidance_efficacy_trajectory"] = eff_traj
            cell.stamp(rec)
            _ZG.observe(agent)
            return rec

        # P1 -- combined wean (transfer; goal live again for INTACT/POSCTRL).
        p1 = scheduler.run_p1(agent, device)
        done += p1.n_episodes
        print(f"  [train] p1 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={p1.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

        # P2 -- frozen-policy guarded measurement.
        p2 = scheduler.run_p2(agent, device)
        done += p2.n_episodes
        print(f"  [train] p2 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" contact_rate={p2.contact_rate:.4f}", flush=True)

        g0 = bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE)
        g1 = bool(p1.survival_gate_passed)
        g2 = bool(p2.contact_rate > CONTACT_GATE)
        g_h = bool(hz.survival_gate_passed)
        gate_final = _gate_state()
        seed_pass = bool(g_h)  # this run's per-seed pass is the Stage-H survival
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm['label']}"
              f" g_h={g_h} g0={g0} g1={g1} g2={g2}"
              f" eff_final={gate_final.get('mech357_avoidance_efficacy', 0.0):.4f}",
              flush=True)

        rec = {
            "arm": arm["label"],
            "seed": seed,
            "aborted_at": None,
            "abort_reason": "",
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "p0_mean_episode_length": float(p0.mean_episode_length),
            "hazard_stage_survival_pass": g_h,
            "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            "hazard_stage_n_episodes": int(hz.n_episodes),
            "hazard_avoidance_driver_enabled": bool(getattr(hz, "avoidance_driver_enabled", False)),
            "pag_n_commits": int(pag_after_h.get("n_commits", 0)),
            "pag_n_releases": int(pag_after_h.get("n_releases", 0)),
            "p1_survival_pass": g1,
            "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
            "p2_contact_rate": float(p2.contact_rate),
            "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
            "g0_stage0_zgoal": g0,
            "g1_p1_survival": g1,
            "g2_p2_contact": g2,
            "g_h_hazard_survival": g_h,
            "avoidance_gate_state": gate_final,
            "avoidance_efficacy_trajectory": eff_traj,
            "reached_hazard_stage": True,
            "reached_p1": True,
            "reached_p2": True,
            "seed_pass": seed_pass,
        }
        cell.stamp(rec)
        _ZG.observe(agent)
        return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _arm_summary(arm: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    g_h_flags = [bool(r.get("g_h_hazard_survival", False)) for r in rows]
    g1_flags = [bool(r.get("g1_p1_survival", False)) for r in rows]
    g0_flags = [bool(r.get("g0_stage0_zgoal", False)) for r in rows]
    g2_flags = [bool(r.get("g2_p2_contact", False)) for r in rows]
    engaged_flags, suppressed_flags = [], []
    for r in rows:
        gs = r.get("avoidance_gate_state", {}) or {}
        engaged_flags.append(
            (int(gs.get("mech357_n_credit", 0)) + int(gs.get("mech357_n_decay", 0))) > 0
        )
        suppressed_flags.append(int(gs.get("mech357_n_freeze_suppressed", 0)) > 0)
    pag_freeze_flags = [int(r.get("pag_n_commits", 0)) > 0 for r in rows]
    return {
        "arm": arm["label"],
        "use_instrumental_avoidance": bool(arm["use_ia"]),
        "scaffold_avoidance_driver_enabled": bool(arm["driver"]),
        "spawn_in_reef_half": bool(arm["reef_spawn"]),
        "g_h_frac": _frac(g_h_flags),
        "g0_frac": _frac(g0_flags),
        "g1_frac": _frac(g1_flags),
        "g2_frac": _frac(g2_flags),
        "gate_engaged_frac": _frac(engaged_flags),
        "gate_freeze_suppressed_frac": _frac(suppressed_flags),
        "pag_freeze_frac": _frac(pag_freeze_flags),
        "per_seed_g_h": g_h_flags,
        "per_seed_g1": g1_flags,
        "per_seed_pag_n_commits": [int(r.get("pag_n_commits", 0)) for r in rows],
        "per_seed_hazard_median_last_window": [
            r.get("hazard_stage_median_last_window", 0.0) for r in rows
        ],
        "per_seed_avoidance_efficacy": [
            (r.get("avoidance_gate_state", {}) or {}).get("mech357_avoidance_efficacy", 0.0)
            for r in rows
        ],
        "per_seed_avoidance_efficacy_trajectory": [
            r.get("avoidance_efficacy_trajectory", []) for r in rows
        ],
        "arm_fingerprint": [r.get("arm_fingerprint") for r in rows],
    }


def _median(values: List[float]) -> float:
    """Plain median. No numpy dependency in the scoring path."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _trace_window_stats(arm_summary: Dict[str, Any]) -> List[Any]:
    """Per-seed scoring-window readout of the LEARNED avoidance_efficacy trace.

    Reads `mech357_avoidance_efficacy` out of the per-episode trajectory that
    scaffolded_sd054_onboarding records for the hazard stage. That is the LEARNED
    component specifically -- NOT `effective_efficacy`, which is
    max(learned, scaffold_floor) and would report the annealing protective floor
    rather than the quantity under test. Reading the wrong one of those two is the
    exact confusion that let the underflow hide behind a gate reporting 1.0.

    Returns one dict per seed (None for a cell that never reached Stage-H).
    """
    out: List[Any] = []
    for traj in arm_summary.get("per_seed_avoidance_efficacy_trajectory") or []:
        entries = [e for e in (traj or []) if isinstance(e, dict)]
        eff = [float(e.get("avoidance_efficacy", 0.0)) for e in entries]
        if not eff:
            out.append(None)
            continue
        win = eff[-SCORING_WINDOW_EPISODES:]
        last = entries[-1]
        n_credit = int(last.get("n_credit", 0))
        n_decay = int(last.get("n_decay", 0))
        run_max = max(eff)
        med = _median(win)
        out.append({
            "n_stage_h_episodes": len(eff),
            "window_episodes": len(win),
            "window_median": med,
            "window_max": max(win),
            "window_min": min(win),
            "run_max": run_max,
            "window_median_over_run_max": (med / run_max) if run_max > 0.0 else 0.0,
            "n_credit": n_credit,
            "n_decay": n_decay,
            "decay_credit_ratio": (float(n_decay) / n_credit) if n_credit > 0 else None,
        })
    return out


def _frac_cells(stats: List[Any], pred) -> float:
    """Fraction of NON-NULL cells satisfying pred.

    `measured` and `met` are the same statistic by construction here: both are
    this fraction against MIN_FRACTION, so the indexer's recompute agrees with
    the author's flag rather than adjudicating a different quantity.
    """
    vals = [t for t in stats if t is not None]
    if not vals:
        return 0.0
    return float(sum(1 for t in vals if pred(t))) / float(len(vals))


def _worst_cell(stats: List[Any], key: str):
    """Return (seed_index, worst_value) for `key` -- the extremum, never a mean.

    A worst-case claim must be reported with the offending cell, not an average
    that can mask a single out-of-band seed (the V3-EXQ-779b recomputability rule).
    """
    vals = [(i, t) for i, t in enumerate(stats) if t is not None]
    if not vals:
        return (None, 0.0)
    idx, cell = min(vals, key=lambda pair: pair[1][key])
    return (idx, float(cell[key]))


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET
        )

    arm_results: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    for arm in ARMS:
        rows = [_run_seed_arm(arm, s, dry_run, total_eps) for s in seeds]
        per_seed.extend(rows)
        arm_results.append(_arm_summary(arm, rows))

    by_label = {a["arm"]: a for a in arm_results}
    on = by_label[ARM_INTACT]
    posctrl = by_label[ARM_POSCTRL]

    trace_by_arm = {a["arm"]: _trace_window_stats(a) for a in arm_results}
    on_trace = trace_by_arm[ARM_INTACT]
    posctrl_trace = trace_by_arm[ARM_POSCTRL]

    # --- C1 (LOAD-BEARING, ARM_INTACT): the pre-registered failure_record target.
    # Scoring-window-scoped, NOT a whole-run aggregate -- that scoping is the whole
    # point of the target, because a whole-run aggregate reports the gate "engaged"
    # at 1.0 while the learned trace it consumes has underflowed to ~1e-26.
    c1_frac = _frac_cells(on_trace, lambda t: t["window_median"] >= TRACE_LIVE_FLOOR)
    c1_pass = bool(c1_frac >= MIN_FRACTION)
    # --- C2 (secondary, NOT load-bearing): "within an order of magnitude of its
    # early-episode range", expressed scale-free against each cell's own peak.
    c2_frac = _frac_cells(
        on_trace, lambda t: t["window_median_over_run_max"] >= TRACE_PEAK_FRACTION
    )
    c2_pass = bool(c2_frac >= MIN_FRACTION)
    # --- C1R (replication, NOT load-bearing): the same readout on POSCTRL.
    c1r_frac = _frac_cells(posctrl_trace, lambda t: t["window_median"] >= TRACE_LIVE_FLOOR)
    c1r_pass = bool(c1r_frac >= MIN_FRACTION)

    # --- READINESS preconditions, evaluated PER ARM so a red POSCTRL can never
    # vacate a green INTACT (the V3-EXQ-785 whole-run-AND defect).
    def _gate_on(ctx: Dict[str, Any]) -> bool:
        return bool(ctx.get("gate_on"))

    specs = [
        PreconditionSpec(
            name="ilpfc_gate_engages_and_suppresses_freeze",
            description=(
                "The ilPFC gate must register efficacy updates under threat "
                "(n_credit+n_decay>0) AND suppress the PAG freeze "
                "(n_freeze_suppressed>0) on >= 2/3 seeds. Below floor => the gate "
                "never engaged, so the trace readout measures nothing => "
                "substrate_not_ready_requeue."
            ),
            control="Gate-ON arm with the protective-scaffold floor 0.8 annealing to 0.0.",
            threshold=MIN_FRACTION_EPS,
            direction="lower",
            applies_to=_gate_on,
            applies_note="Only a gate-ON arm has an ilPFC gate to engage.",
            structural_max=lambda ctx: 1.0,
        ),
        PreconditionSpec(
            name="stage0_forced_feed_lights_zgoal",
            description=(
                "Stage-0 forced supra-threshold benefit lights z_goal (>0.4) on "
                ">= 2/3 seeds -- the goal-FORMATION positive control, without which "
                "the agent never reaches a meaningful Stage-H."
            ),
            control="run_stage0_nursery forced-feed.",
            threshold=MIN_FRACTION_EPS,
            direction="lower",
            structural_max=lambda ctx: 1.0,
        ),
        PreconditionSpec(
            name="avoidance_trace_peak_nondegenerate",
            description=(
                "SAME-STATISTIC positive control for C1 (the 643 rule): the LEARNED "
                "mech357_avoidance_efficacy -- the exact quantity C1 routes on, not a "
                "proxy -- must reach TRACE_PEAK_READY_FLOOR at SOME point in the "
                "Stage-H run on >= 2/3 seeds. V3-EXQ-603u cleared this on the BROKEN "
                "substrate (in-run peaks 0.659-0.970), so it is known-achievable and "
                "independent of the repair under test. Below floor => the learner "
                "never moved the trace at all => substrate_not_ready_requeue, NEVER a "
                "verdict about whether the repair worked."
            ),
            control=(
                "V3-EXQ-603u, same config, pre-fix: in-run peaks 0.659/0.936/0.832 "
                "(INTACT) and 0.894/0.964/0.971 (POSCTRL)."
            ),
            threshold=MIN_FRACTION_EPS,
            direction="lower",
            applies_to=_gate_on,
            applies_note="Only a gate-ON arm carries a learned avoidance trace.",
            structural_max=lambda ctx: 1.0,
        ),
    ]

    arm_contexts = [
        {"id": ARM_INTACT, "gate_on": True},
        {"id": ARM_POSCTRL, "gate_on": True},
    ]
    assert_no_structurally_unsatisfiable_gate(specs, arm_contexts)

    def _measured_for(arm_label: str) -> Dict[str, float]:
        summ = by_label[arm_label]
        tr = trace_by_arm[arm_label]
        return {
            "ilpfc_gate_engages_and_suppresses_freeze": float(
                min(summ["gate_engaged_frac"], summ["gate_freeze_suppressed_frac"])
            ),
            "stage0_forced_feed_lights_zgoal": float(summ["g0_frac"]),
            "avoidance_trace_peak_nondegenerate": _frac_cells(
                tr, lambda t: t["run_max"] >= TRACE_PEAK_READY_FLOOR
            ),
        }

    arm_gates = [
        evaluate_arm_gate(ctx["id"], ctx, specs, _measured_for(ctx["id"]))
        for ctx in arm_contexts
    ]
    aggregate = aggregate_arm_gates(arm_gates)
    intact_green = ARM_INTACT in aggregate["green_arms"]

    # --- ROUTING. A below-floor readiness reading is ALWAYS a requeue, never a
    # verdict on the repair (the 642 rule).
    if not intact_green:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_note = (
            "ARM_INTACT failed its readiness gate ("
            + ", ".join(arm_gates[0]["failed_preconditions"])
            + "), so the scoring-window trace readout measures nothing about the "
              "eligibility-trace repair. Re-queue at an adequate P0; do NOT read this "
              "as evidence the repair failed."
        )
    elif c1_pass:
        outcome = "PASS"
        readiness_route = "eligibility_trace_repair_validated"
        route_note = (
            "Credit-eligibility windowing (ree-v3 93d5d98b80) holds the learned "
            "avoidance_efficacy above the pre-registered floor through the scoring "
            "window, where V3-EXQ-603u measured numerical zero (~1e-24..1e-29) on the "
            "identical config. The MECH-357 Stage-H instrument is repaired: the INTACT "
            "arm is no longer functionally lesioned where the DV is measured. "
            "substrate_queue mech357-avoidance-efficacy-eligibility-trace-imbalance "
            "failure_record target MET -> that failure_record may be resolved."
        )
    else:
        outcome = "FAIL"
        readiness_route = "eligibility_trace_repair_insufficient"
        route_note = (
            "Gate engaged and the trace demonstrably moved (readiness peak cleared), "
            "but the learned avoidance_efficacy still falls below the pre-registered "
            "floor across the scoring window. Credit-eligibility windowing alone is "
            "insufficient; candidates (b) leak/learn rebalance against realistic "
            "n_credit/n_decay ratios and (c) time-since-credit decay remain open in "
            "the substrate entry's implementation_hint."
        )

    preconditions = aggregate["adjudication_preconditions"]
    criteria = [
        {"name": "C1_trace_live_in_scoring_window_INTACT",
         "load_bearing": True, "passed": bool(c1_pass)},
        {"name": "C2_trace_within_order_of_magnitude_of_peak_INTACT",
         "load_bearing": False, "passed": bool(c2_pass)},
        {"name": "C1R_trace_live_in_scoring_window_POSCTRL",
         "load_bearing": False, "passed": bool(c1r_pass)},
    ]
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {
            ARM_INTACT: ["C1_trace_live_in_scoring_window_INTACT",
                         "C2_trace_within_order_of_magnitude_of_peak_INTACT"],
            ARM_POSCTRL: ["C1R_trace_live_in_scoring_window_POSCTRL"],
        },
        aggregate,
        extra={
            # Power check, independent of the gate: the window must actually exist.
            "C1_trace_live_in_scoring_window_INTACT": bool(
                _frac_cells(on_trace, lambda t: t["window_episodes"] >= SCORING_WINDOW_EPISODES)
                >= MIN_FRACTION
            ),
        },
    )

    evidence_direction = (
        "supports" if outcome == "PASS"
        else ("weakens" if intact_green else "non_contributory")
    )

    worst_seed_idx, worst_median = _worst_cell(on_trace, "window_median")
    print(
        f"[{EXPERIMENT_TYPE}] C1_frac={c1_frac:.2f} (floor {TRACE_LIVE_FLOOR})"
        f" C2_frac={c2_frac:.2f} C1R_frac={c1r_frac:.2f}"
        f" worst_INTACT_seed_idx={worst_seed_idx} worst_window_median={worst_median:.3g}"
        f" intact_green={intact_green}"
        f" -> outcome={outcome} route={readiness_route}"
        f" evidence_direction={evidence_direction}",
        flush=True,
    )

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "primary_pass": bool(c1_pass),
        "c1_trace_live_frac_intact": c1_frac,
        "c2_within_order_of_magnitude_frac_intact": c2_frac,
        "c1r_trace_live_frac_posctrl": c1r_frac,
        "intact_gate_green": intact_green,
        "readiness_met": bool(aggregate["non_degenerate"]),
        "non_degenerate": bool(aggregate["non_degenerate"]),
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "per_arm_gate": aggregate["per_arm_gate"],
        "arm_results": arm_results,
        "trace_window_stats": {
            "arm_intact": on_trace,
            "arm_posctrl": posctrl_trace,
        },
        "acceptance": {
            "pass_rule": (
                "PASS = ARM_INTACT readiness gate green AND median learned "
                "mech357_avoidance_efficacy over the LAST "
                f"{SCORING_WINDOW_EPISODES} Stage-H episodes >= {TRACE_LIVE_FLOOR} "
                "on >= 2/3 INTACT seeds"
            ),
            "min_fraction": MIN_FRACTION,
            "scoring_window_episodes": SCORING_WINDOW_EPISODES,
            "trace_live_floor": TRACE_LIVE_FLOOR,
            "trace_peak_fraction": TRACE_PEAK_FRACTION,
            "trace_peak_ready_floor": TRACE_PEAK_READY_FLOOR,
            "predecessor_measured_pre_fix": {
                "run_id": ("v3_exq_603u_instrumental_avoidance_agent_pursuit_"
                           "20260815T020607Z_v3"),
                "intact_window_medians": [6.5e-26, 3.92e-29, 3.63e-24],
                "intact_run_peaks": [0.6587, 0.9362, 0.8318],
                "intact_decay_credit_ratios": [60.9, 130.7, 81.8],
                "posctrl_window_medians": [3.99e-22, 1.85e-13, 2.21e-27],
                "note": ("Recorded from the 603u manifest's "
                         "per_seed_avoidance_efficacy_trajectory. Same config, "
                         "pre-fix substrate -- the comparison this run is against."),
            },
        },
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_note": route_note,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "preconditions_scope_note": aggregate["per_arm_gate"].get(
                "preconditions_scope_note", ""
            ),
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        "seeds": SEEDS,
        "budgets": {
            "stage0": STAGE0_BUDGET, "stage0b": STAGE0B_BUDGET, "p0": P0_BUDGET,
            "hazard": HAZARD_STAGE_BUDGET, "p1": P1_BUDGET, "p2": P2_BUDGET,
            "train_steps": TRAIN_STEPS,
        },
        "hazard_stage": {
            "num_hazards": HAZARD_STAGE_NUM_HAZARDS,
            "num_resources": HAZARD_STAGE_NUM_RESOURCES,
            "hazard_food_attraction": HAZARD_STAGE_HFA,
            "proximity_harm_scale": HAZARD_STAGE_PROXIMITY_HARM,
            "survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "stability_window": HAZARD_STAGE_STABILITY_WINDOW,
            "env_drift_interval": HAZARD_STAGE_ENV_DRIFT_INTERVAL,
            "env_drift_prob": HAZARD_STAGE_ENV_DRIFT_PROB,
            "hazard_agent_pursuit": HAZARD_STAGE_HAZARD_AGENT_PURSUIT,
        },
        "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
        "arms": [a["label"] for a in ARMS],
    }
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "SD-058 / MECH-357 instrumental-avoidance acquisition "
                     "(ree_core/pfc/infralimbic_avoidance_gate.py) WITH the "
                     "credit-eligibility-windowing repair (ree-v3 93d5d98b80, "
                     "2026-08-16: freeze/no-op ticks under threat no longer decay "
                     "avoidance_efficacy), driven in the scaffolded_sd054_onboarding "
                     "Stage-H under the SAME agent-directed-pursuit field as "
                     "V3-EXQ-603u (env_drift_interval=1/prob=0.6 + "
                     "hazard_agent_pursuit=0.9), with the SD-059/MECH-358 escape-"
                     "affordance bridge + harm-pathway-training amend both on",
        "validates": "substrate_queue mech357-avoidance-efficacy-eligibility-trace-"
                     "imbalance (status implemented, ree-v3 93d5d98b80) against its OWN "
                     "pre-registered failure_record target: a scoring-window-scoped "
                     "gate_live check that the LEARNED mech357_avoidance_efficacy is "
                     "non-underflowed across the last 10 Stage-H episodes on >= 2/3 "
                     "INTACT seeds. Instrument repair, not a MECH-357 retest.",
        "validates_substrate_queue_entry": "mech357-avoidance-efficacy-eligibility-trace-imbalance",
        "validates_substrate_commit": "93d5d98b80",
        "predecessor_run_id": "v3_exq_603u_instrumental_avoidance_agent_pursuit_20260815T020607Z_v3",
        "predecessor_queue_id": "V3-EXQ-603u",
        "supersedes": None,
        "reuse_check_note": "GOV-REUSE-1: decisive readout = median LEARNED "
                            "mech357_avoidance_efficacy over the last 10 Stage-H "
                            "episodes on the INTACT arm. Checked V3-EXQ-603u "
                            "(20260815T020607Z, the only run recording "
                            "per_seed_avoidance_efficacy_trajectory on this config) plus "
                            "the whole 603h/k/r/s/t series. The readout IS recorded there "
                            "-- but every recorded value predates the repair under test "
                            "(ree-v3 93d5d98b80), i.e. sits on a DIFFERENT substrate_hash "
                            "(603u: dcab912f6cbe9566...). A post-fix value exists in no "
                            "manifest and is not derivable by reprocessing, because the "
                            "fix changes the update recurrence that generates the "
                            "trajectory. Not reanalysis-recoverable -> ran fresh. "
                            "chip-20260825-mech357-h2-reanalysis reanalysed 603s/t/u "
                            "under a graded survival DV and likewise could not speak to "
                            "the post-fix trace.",
        "design_note": "Validation rerun of the V3-EXQ-603u shape against the "
                       "credit-eligibility-windowing repair. The substrate_queue entry "
                       "mech357-avoidance-efficacy-eligibility-trace-imbalance is status "
                       "`implemented` with its failure_record deliberately left "
                       "resolved=open and its implementation_note_update stating the "
                       "empirical re-validation is separate, still-owed work; this run "
                       "discharges it. Config is bit-for-bit 603u's so the ONLY changed "
                       "variable is the substrate fix. 2 arms x 3 seeds [42,43,44]: "
                       "ARM_INTACT (the arm the target names, load-bearing) and "
                       "ARM_POSCTRL (independent replication in a different spawn "
                       "regime). ARM_LESION is dropped deliberately -- gate OFF means no "
                       "learned trace (603u recorded [0.0,0.0,0.0]) so it is invariant "
                       "under this fix and re-running it would be ~6.4h of cloud compute "
                       "for a result fixed by construction. This run does NOT attempt the "
                       "MECH-357 INTACT-vs-LESION discrimination: 603u measured "
                       "G_H_LESION_frac=1.0 alongside G_H_INTACT_frac=1.0 and self-routed "
                       "pressure_insufficient_lesion_ceiling_requeue, so that contrast is "
                       "structurally vacuous at this pressure regardless of the trace. "
                       "Pressure recalibration is separately owed and is reported as a "
                       "finding rather than folded in here, which would confound the "
                       "repair with a changed threat regime. Clears the MECH-357 "
                       "re-derive brake (count 2: failure_autopsy_V3-EXQ-603s_2026-08-10, "
                       "failure_autopsy_V3-EXQ-603t_2026-08-13): the 603t autopsy's "
                       "recommended_substrate_queue_entry is an AMEND of "
                       "mech357-freeze-incompatible-pressure-mechanism titled 'Wire "
                       "agent-directed hazard pursuit into Stage-H onboarding; separately "
                       "investigate eligibility-trace leak:learn imbalance' -- the "
                       "eligibility-trace half is exactly the substrate this run "
                       "validates, and it is now IMPLEMENTED, so the brake is RELEASED. "
                       "experiment_purpose=diagnostic: this is instrument repair, "
                       "excluded from governance confidence scoring by design.",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_pass_rule": "ARM_INTACT readiness gate green AND median LEARNED "
                                 "mech357_avoidance_efficacy over the LAST 10 Stage-H "
                                 "episodes >= 0.01 on >= 2/3 INTACT seeds",
            "scoring_window_episodes": SCORING_WINDOW_EPISODES,
            "trace_live_floor": TRACE_LIVE_FLOOR,
            "trace_peak_fraction": TRACE_PEAK_FRACTION,
            "trace_peak_ready_floor": TRACE_PEAK_READY_FLOOR,
            "source_of_target": "substrate_queue mech357-avoidance-efficacy-eligibility-"
                                "trace-imbalance failure_record[0].target",
            "freeze_incompatible_pressure": {
                "env_drift_interval": HAZARD_STAGE_ENV_DRIFT_INTERVAL,
                "env_drift_prob": HAZARD_STAGE_ENV_DRIFT_PROB,
                "hazard_agent_pursuit": HAZARD_STAGE_HAZARD_AGENT_PURSUIT,
                "note": "identical to V3-EXQ-603u -- unchanged on purpose",
            },
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "min_fraction": MIN_FRACTION,
        },
        "avoidance_driver": {
            "scaffold_floor_start": AVOIDANCE_SCAFFOLD_FLOOR_START,
            "scaffold_floor_end": AVOIDANCE_SCAFFOLD_FLOOR_END,
            "feed_harm_stream": True,
            "avoidance_threat_ref": AVOIDANCE_THREAT_REF,
            "pag_theta_freeze": PAG_THETA_FREEZE,
            "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
        },
        "escape_bridge_fix": {
            "use_escape_affordance_bridge": True,
            "use_escape_relief_credit": True,
            "use_escape_safety_credit": True,
            "escape_threat_floor": ESCAPE_THREAT_FLOOR,
            "escape_threat_ref": ESCAPE_THREAT_REF,
            "escape_approach_gain": ESCAPE_APPROACH_GAIN,
            "escape_bias_scale": ESCAPE_BIAS_SCALE,
            "escape_use_trained_safety_signal": True,
            "escape_safety_signal_threshold": ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        },
        "harm_pathway_fix": {
            "scaffold_train_harm_pathway": True,
            "scaffold_harm_pathway_lr": HARM_PATHWAY_LR,
            "scaffold_harm_pathway_in_p0": True,
            "scaffold_harm_pathway_encoder_lr": HARM_PATHWAY_ENCODER_LR,
            "scaffold_harm_pathway_warmup_steps": HARM_PATHWAY_WARMUP_STEPS,
            "use_harm_stream": True,
            "use_e2_harm_s_forward": True,
        },
    }
    manifest.update(result)
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
            dry_run=bool(args.dry_run),
        )
