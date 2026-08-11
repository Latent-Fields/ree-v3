"""
V3-EXQ-603t -- SD-058 / MECH-357 instrumental-avoidance acquisition: SD-029
SCHEDULED-EXTERNAL-HAZARD Stage-H retest (claim-tagged, MECH-357).

PURPOSE (evidence; claim_ids=["MECH-357"]):
Direct successor to V3-EXQ-603s, per
mech357_freeze_incompatible_pressure_scoping_2026-08-10.md (a scoping spike
requested by failure_autopsy_V3-EXQ-603s_2026-08-10.md section 6 routing).

603s applied a mobile-predator field (env_drift_interval 5->1, env_drift_prob
0.3->0.6) to reintroduce a freeze-incompatible pressure, but FAILED
discrimination on an EXACT TIE (G_H_LESION_frac == G_H_INTACT_frac ==
0.6667). The scoping spike re-read 603s's own per-seed data and its own
_drift_hazards mechanism and found the cause: env_drift is UNDIRECTED
(_drift_hazards, causal_grid_world.py:4945 -- hazards move randomly, or
biased toward FOOD when hazard_food_attraction>0, NEVER toward the agent).
On a 10x10 grid with 4 fast-drifting hazards, this converts "hazards
eventually threaten the agent" into a SPAWN-POSITION LOTTERY: whether a
hazard's random walk happens to reach the agent early is close to
independent of the agent's policy (freeze/release vs directed escape), so
the outcome is bimodal (200-step survival ceiling or ~5.5-6-step death
matching the driver's own documented self-contamination death clock for a
pure freeze) rather than a graded Pavlovian-instrumental conflict.

THE FIX THIS RUN INTRODUCES -- SD-029 scheduled_external_hazard, an
already-built, currently-unused CausalGridWorldV2 mechanism
(causal_grid_world.py:260-270 constructor params /
_inject_external_hazard at causal_grid_world.py:4774-4843). On a schedule
(every scheduled_external_hazard_interval steps, with probability
scheduled_external_hazard_prob), it deterministically moves an existing
hazard (or spawns one) to a cell ADJACENT TO THE AGENT'S CURRENT POSITION --
not a random cell, not food-biased, but the agent's neighbourhood
specifically, wherever the agent currently is. This is qualitatively
different from env_drift in exactly the way the scoping note identifies as
missing:
  - AGENT-RELATIVE BY CONSTRUCTION (unlike env_drift, which is symmetric
    w.r.t. the agent and only coincidentally threatens it).
  - DISCRETE AND SCHEDULED, not continuous -- a genuine threat-onset event,
    closer to the active-avoidance paradigms (Moscarello & LeDoux) both
    603r/603s autopsies cite as the biological reference.
  - REMOVES THE SPAWN-POSITION LOTTERY: because injection targets the
    agent's cell AFTER the episode (and the trained policy) is already
    underway, the outcome is no longer dominated by an early random-walk
    trajectory relative to a fixed spawn point.
  - Zero new environment code -- the four SD-029 parameters already exist on
    CausalGridWorldV2.__init__ (default enabled=False) and are threaded here
    through ScaffoldedSD054OnboardingConfig / _build_env's "hazard" phase the
    same way env_drift_interval/_prob already were (V3-EXQ-603s).

env_drift is REVERTED to the CausalGridWorldV2 constructor defaults
(interval=5, prob=0.3 -- the pre-603s / pre-603r baseline) so the discrete
SD-029 injection is the ISOLATED new pressure variable, rather than stacking
on top of 603s's spawn-lottery-prone continuous drift. Everything else --
both combined fixes (escape-affordance bridge + harm-pathway training),
budgets, seeds, the ilPFC gate under test -- is IDENTICAL to 603s.

CALIBRATION. interval=20, prob=0.9: several discrete forced-proximity events
across a 200-step / 40-episode Stage-H budget (roughly one event per two
episodes), near-guaranteed on schedule so all three arms see comparable
event counts, adjacent_only=True (matches "an adjacent threat appears"
rather than "somewhere in the grid"). This is a FIRST calibration point --
per the scoping note, one discrete adjacency event might still be survivable
by luck or a single reactive move without requiring SUSTAINED directed
evasion, which is exactly why the two-sided headroom guard exists: a
mis-calibrated schedule self-routes to a requeue instead of a false MECH-357
verdict.

DESIGN -- 3 arms x 3 seeds [42,43,44], TWO-SIDED discriminative-headroom
guard (unchanged from 603s -- worked as designed there; the mechanism, not
the guard, is what changed):

  ARM_LESION_midline    (use_ia=False, driver=False, midline spawn, both fixes):
      the NEGATIVE CONTROL. For the field to be discriminative it MUST now
      FAIL the survival gate (G_H_LESION < 2/3) -- headroom from BELOW. If
      LESION still clears the gate, the scheduled-hazard pressure did not
      reintroduce the conflict -> substrate_not_ready_requeue.
  ARM_INTACT_midline    (use_ia=True,  driver=True,  midline spawn, both fixes):
      the TEST arm. The ilPFC gate suppresses the PAG freeze so the agent
      keeps fleeing under threat instead of being caught by the scheduled
      adjacency event mid-freeze.
  ARM_POSCTRL_reefspawn (use_ia=True,  driver=True,  REEF spawn,   both fixes):
      the POSITIVE CONTROL / upper-bound guard. Best-case (gated) agent
      starting ADJACENT to safety. It MUST clear the gate: if even a gated
      agent spawned in the refuge cannot survive, the pressure is impossible
      for ANY policy -> substrate_not_ready_requeue, NOT a MECH-357 weakens.

PRE-REGISTERED ACCEPTANCE (constants; identical structure to 603s):
  READINESS (non-vacuity; ALL must hold or -> substrate_not_ready_requeue):
    R1 pavlovian_freeze_reaction_present_on_lesion  (PAG commits on LESION >=2/3)
    R2 ilpfc_gate_engages_and_suppresses_freeze_on_intact (>=2/3 INTACT seeds)
    R3 stage0_forced_feed_lights_zgoal_on_intact    (goal-formation control >=2/3)
    R4 discriminative_headroom_below: LESION does NOT clear the gate
       (G_H_LESION < 2/3) -- the scheduled-hazard pressure reintroduced the
       conflict.
    R5 survivability_exists_above: POSCTRL clears the gate (G_H_POSCTRL >= 2/3)
       -- the pressure is not impossible for a gated agent from safety.
  PRIMARY (load-bearing): G_H_INTACT >= 2/3 AND G_H_INTACT_frac > G_H_LESION_frac.
  CONFIRMING: readiness (R1-R5) all met AND primary clears -> MECH-357 supports:
    the ilPFC freeze-suppression gate converts a non-survivable passive
    policy into a survivable directed-avoidance policy under a discrete,
    agent-relative threat-onset field.
  FALSIFYING: readiness (R1-R5) all met (conflict present, survivability
    exists, gate engaging) but INTACT still fails to clear 2/3 or fails to
    beat LESION -> MECH-357 weakens.
  NON-DISCRIMINATIVE (either headroom guard fails) -> substrate_not_ready_requeue,
    non_contributory: re-calibrate (interval down / prob up if LESION
    ceilinged; interval up / prob down if POSCTRL floored) before spending a
    verdict.

MECH-357's own avoidance_efficacy eligibility-trace credit event
instrumentation (per-episode Stage-H trajectory) carries over unchanged from
603s.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: evidence
claim_ids: ["MECH-357"]  (claim-tagged SD-029 successor of 603s)
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

EXPERIMENT_TYPE = "v3_exq_603t_instrumental_avoidance_scheduled_external_hazard"
QUEUE_ID = "V3-EXQ-603t"
CLAIM_IDS: List[str] = ["MECH-357"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror 603s/603r/603h exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Budgets (mirror 603s/603r/603h full budget for apples-to-apples comparison).
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

# Isolated hazard-avoidance Stage-H (density unchanged from 603s/603r/603h so
# the pressure mechanism is the isolated new variable).
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = False  # midline for LESION/INTACT; POSCTRL overrides
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- Ambient env_drift REVERTED to CausalGridWorldV2 constructor defaults
# (603s used interval=1/prob=0.6 "mobile predator" drift; the scoping spike
# traced 603s's exact tie to this being UNDIRECTED, a spawn-position
# lottery). Keeping background drift at its low default isolates the new
# SD-029 discrete injection below as the only pressure variable. ---
HAZARD_STAGE_ENV_DRIFT_INTERVAL = 5
HAZARD_STAGE_ENV_DRIFT_PROB = 0.3

# --- THE FREEZE-INCOMPATIBLE PRESSURE (603t): SD-029 scheduled external
# hazard. Deterministically places a hazard adjacent to the agent's CURRENT
# position every HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_INTERVAL steps with
# probability HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_PROB -- agent-relative
# and discrete, unlike 603s's undirected continuous drift. ---
HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ENABLED = True
HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_INTERVAL = 20
HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_PROB = 0.9
HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ADJACENT_ONLY = True

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603s/603r).
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (unchanged from 603s/603r) ---
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

# 3 arms: LESION (negative control, midline), INTACT (test, midline),
# POSCTRL (positive control / upper-bound guard, gated + reef spawn).
ARM_LESION = "ARM_LESION_midline_pag_no_gate_combined_fix"
ARM_INTACT = "ARM_INTACT_midline_pag_gate_combined_fix"
ARM_POSCTRL = "ARM_POSCTRL_reefspawn_pag_gate_combined_fix"
ARMS = [
    {"label": ARM_LESION, "use_ia": False, "driver": False, "reef_spawn": False},
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
        # The isolated Stage-H (density unchanged from 603g/603h/603r/603s).
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
        # Ambient drift REVERTED to defaults (603t isolates SD-029 as the
        # pressure variable; see module docstring).
        scaffold_hazard_stage_env_drift_interval=HAZARD_STAGE_ENV_DRIFT_INTERVAL,
        scaffold_hazard_stage_env_drift_prob=HAZARD_STAGE_ENV_DRIFT_PROB,
        # THE FREEZE-INCOMPATIBLE PRESSURE (603t): SD-029 scheduled external
        # hazard, agent-relative and discrete.
        scaffold_hazard_stage_scheduled_external_hazard_enabled=(
            HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ENABLED
        ),
        scaffold_hazard_stage_scheduled_external_hazard_interval=(
            HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_INTERVAL
        ),
        scaffold_hazard_stage_scheduled_external_hazard_prob=(
            HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_PROB
        ),
        scaffold_hazard_stage_scheduled_external_hazard_adjacent_only=(
            HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ADJACENT_ONLY
        ),
        # SD-058 / MECH-357 avoidance-learning driver (INTACT + POSCTRL only).
        scaffold_avoidance_driver_enabled=bool(arm["driver"]),
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE (all arms): feed the env harm stream (unchanged from 603s/603r).
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
        # MECH-279 PAG freeze-gate (ALL arms) -- unchanged from 603s/603r.
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
            "mechanism": "scheduled_external_hazard",
            "env_drift_interval": HAZARD_STAGE_ENV_DRIFT_INTERVAL,
            "env_drift_prob": HAZARD_STAGE_ENV_DRIFT_PROB,
            "scheduled_external_hazard_enabled": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ENABLED,
            "scheduled_external_hazard_interval": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_INTERVAL,
            "scheduled_external_hazard_prob": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_PROB,
            "scheduled_external_hazard_adjacent_only": (
                HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ADJACENT_ONLY
            ),
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

        # Stage-H -- ISOLATED HAZARD-AVOIDANCE under the SD-029 scheduled
        # external-hazard field (the SD-058/MECH-357 driver target).
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
    off = by_label[ARM_LESION]
    on = by_label[ARM_INTACT]
    posctrl = by_label[ARM_POSCTRL]

    # --- PRIMARY pre-registered acceptance (load-bearing) ---
    g_h_on_clears = bool(on["g_h_frac"] >= MIN_FRACTION)
    g_h_on_beats_off = bool(on["g_h_frac"] > off["g_h_frac"])
    primary_pass = bool(g_h_on_clears and g_h_on_beats_off)

    # --- READINESS PRECONDITIONS ---
    # R1/R2/R3 (unchanged from 603s/603r): the mechanism must actually engage.
    pavlovian_reaction_present = bool(off["pag_freeze_frac"] >= MIN_FRACTION)
    gate_engaged = bool(on["gate_engaged_frac"] >= MIN_FRACTION)
    gate_suppresses = bool(on["gate_freeze_suppressed_frac"] >= MIN_FRACTION)
    g0_on_ok = bool(on["g0_frac"] >= MIN_FRACTION)
    mechanism_ready = bool(
        pavlovian_reaction_present and gate_engaged and gate_suppresses
    )
    # R4 (headroom from BELOW): the scheduled-hazard pressure must make the
    # passive/undirected LESION policy FAIL the survival gate.
    headroom_below = bool(off["g_h_frac"] < MIN_FRACTION)
    # R5 (survivability from ABOVE): a gated agent spawned in the refuge
    # (best case) must clear the gate, else the pressure is impossible for anyone.
    survivability_exists = bool(posctrl["g_h_frac"] >= MIN_FRACTION)

    readiness_met = bool(mechanism_ready and headroom_below and survivability_exists)

    if not mechanism_ready:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_note = "mechanism did not engage (PAG freeze / gate engage+suppress / g0 below floor)"
    elif not survivability_exists:
        outcome = "FAIL"
        readiness_route = "pressure_impossible_requeue"
        route_note = ("scheduled-hazard pressure too lethal: even the gated positive-"
                      "control arm spawned in the reef refuge could not clear the "
                      "survival gate -> re-calibrate DOWN (raise scheduled_external_"
                      "hazard_interval / lower scheduled_external_hazard_prob)")
    elif not headroom_below:
        outcome = "FAIL"
        readiness_route = "pressure_insufficient_lesion_ceiling_requeue"
        route_note = ("scheduled-hazard pressure did not reintroduce the conflict: the "
                      "LESION negative control still cleared the survival gate -> "
                      "re-calibrate UP (lower scheduled_external_hazard_interval / "
                      "raise scheduled_external_hazard_prob)")
    elif primary_pass:
        outcome = "PASS"
        readiness_route = "avoidance_mechanism_trains_survival_leg_scheduled_external_hazard"
        route_note = ("ilPFC freeze-suppression converts a non-survivable passive policy "
                      "into a survivable directed-avoidance policy under a discrete, "
                      "agent-relative threat-onset field")
    else:
        outcome = "FAIL"
        readiness_route = "avoidance_gate_insufficient_scheduled_external_hazard"
        route_note = ("conflict present (LESION fails), survivability confirmed (POSCTRL "
                      "clears), gate engaging -- but INTACT still failed to clear 2/3 or "
                      "beat LESION: freeze-suppression insufficient to rescue survival")

    preconditions = [
        {
            "name": "pavlovian_freeze_reaction_present_on_lesion",
            "kind": "readiness",
            "description": "MECH-279 PAG freeze must commit on the LESION arm "
                           "(pag_n_commits > 0 on >= 2/3 seeds) -- the Pavlovian reaction "
                           "the ilPFC suppresses. Below-floor => no freeze to suppress => "
                           "vacuous contrast => substrate_not_ready_requeue.",
            "control": "ARM_LESION: PAG on, harm stream fed, both combined fixes on, "
                       "SD-029 scheduled hazard -- the freeze reaction must still fire.",
            "measured": float(off["pag_freeze_frac"]),
            "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(pavlovian_reaction_present),
        },
        {
            "name": "ilpfc_gate_engages_and_suppresses_freeze_on_intact",
            "kind": "readiness",
            "description": "On the INTACT arm the ilPFC gate must register efficacy updates "
                           "under threat (n_credit+n_decay>0) AND suppress the PAG freeze "
                           "(n_freeze_suppressed>0) on >= 2/3 seeds. Below-floor => gate "
                           "never engaged => substrate_not_ready_requeue.",
            "control": "ARM_INTACT: gate on + protective-scaffold floor 0.8 + both fixes.",
            "measured": float(min(on["gate_engaged_frac"], on["gate_freeze_suppressed_frac"])),
            "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(gate_engaged and gate_suppresses),
        },
        {
            "name": "stage0_forced_feed_lights_zgoal_on_intact",
            "kind": "readiness",
            "description": "Stage-0 forced supra-threshold benefit lights z_goal (>0.4) on "
                           ">=2/3 INTACT seeds -- the goal-FORMATION positive control.",
            "control": "run_stage0_nursery forced-feed.",
            "measured": float(on["g0_frac"]),
            "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(g0_on_ok),
        },
        {
            "name": "discriminative_headroom_below_lesion_fails_gate",
            "kind": "readiness",
            "description": "The SD-029 scheduled-hazard pressure must make the LESION "
                           "negative control FAIL the survival gate (G_H_LESION < 2/3) -- "
                           "i.e. it reintroduced the Pavlovian-instrumental conflict that "
                           "603s's spawn-lottery-confounded field could not discriminate. "
                           "Measured = 1 - G_H_LESION_frac so the floor semantics match: "
                           "below floor => LESION still ceilinged => "
                           "pressure_insufficient_lesion_ceiling_requeue.",
            "control": "ARM_LESION under SD-029 scheduled hazard "
                       "(interval=20, prob=0.9, adjacent_only=True).",
            "measured": float(1.0 - off["g_h_frac"]),
            "threshold": float(1.0 - MIN_FRACTION + 1e-9),
            "direction": "lower",
            "met": bool(headroom_below),
        },
        {
            "name": "survivability_exists_above_posctrl_clears_gate",
            "kind": "readiness",
            "description": "The pressure must not be impossible: a gated agent spawned in "
                           "the reef refuge (best case) must clear the survival gate "
                           "(G_H_POSCTRL >= 2/3). Below-floor => pressure too lethal for "
                           "ANY policy => pressure_impossible_requeue, NOT a MECH-357 "
                           "weakens.",
            "control": "ARM_POSCTRL: gate on + reef spawn + both fixes, SD-029 scheduled "
                       "hazard.",
            "measured": float(posctrl["g_h_frac"]),
            "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(survivability_exists),
        },
    ]
    criteria_non_degenerate = {
        "both_test_arms_reached_hazard_stage": bool(
            _frac([r.get("reached_hazard_stage", False) for r in per_seed]) >= MIN_FRACTION
        ),
        "pavlovian_freeze_present": bool(off["pag_freeze_frac"] > 0.0),
        "gate_active_on_intact": bool(on["gate_engaged_frac"] > 0.0),
        "discriminative_headroom_present": bool(headroom_below and survivability_exists),
    }
    criteria = [
        {"name": "G_H_INTACT_clears_2of3", "load_bearing": True, "passed": bool(g_h_on_clears)},
        {"name": "G_H_INTACT_beats_LESION", "load_bearing": True, "passed": bool(g_h_on_beats_off)},
    ]

    evidence_direction = (
        "supports" if outcome == "PASS"
        else ("weakens" if readiness_met else "non_contributory")
    )

    print(
        f"[{EXPERIMENT_TYPE}] G_H_INTACT={on['g_h_frac']:.2f} G_H_LESION={off['g_h_frac']:.2f}"
        f" G_H_POSCTRL={posctrl['g_h_frac']:.2f}"
        f" pag_freeze_LESION={off['pag_freeze_frac']:.2f}"
        f" gate_engaged_INTACT={on['gate_engaged_frac']:.2f}"
        f" gate_suppr_INTACT={on['gate_freeze_suppressed_frac']:.2f}"
        f" headroom_below={headroom_below} survivability_exists={survivability_exists}"
        f" -> outcome={outcome} route={readiness_route} evidence_direction={evidence_direction}",
        flush=True,
    )

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "primary_pass": primary_pass,
        "g_h_intact_clears": g_h_on_clears,
        "g_h_intact_beats_lesion": g_h_on_beats_off,
        "readiness_met": readiness_met,
        "mechanism_ready": mechanism_ready,
        "pavlovian_reaction_present": pavlovian_reaction_present,
        "gate_engaged": gate_engaged,
        "gate_suppresses": gate_suppresses,
        "discriminative_headroom_below": headroom_below,
        "survivability_exists": survivability_exists,
        "arm_results": arm_results,
        "acceptance": {
            "pass_rule": "PASS = mechanism_ready AND headroom_below (G_H_LESION < 2/3) AND "
                         "survivability_exists (G_H_POSCTRL >= 2/3) AND G_H_INTACT >= 2/3 "
                         "AND G_H_INTACT_frac > G_H_LESION_frac",
            "min_fraction": MIN_FRACTION,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "freeze_incompatible_pressure": {
                "mechanism": "SD-029 scheduled_external_hazard (agent-relative, discrete)",
                "scheduled_external_hazard_interval": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_INTERVAL,
                "scheduled_external_hazard_prob": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_PROB,
                "scheduled_external_hazard_adjacent_only": (
                    HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ADJACENT_ONLY
                ),
                "env_drift_interval": HAZARD_STAGE_ENV_DRIFT_INTERVAL,
                "env_drift_prob": HAZARD_STAGE_ENV_DRIFT_PROB,
                "env_defaults_for_comparison": {"env_drift_interval": 5, "env_drift_prob": 0.3},
            },
            "g_h_intact_frac": on["g_h_frac"],
            "g_h_lesion_frac": off["g_h_frac"],
            "g_h_posctrl_frac": posctrl["g_h_frac"],
            "pag_freeze_lesion_frac": off["pag_freeze_frac"],
            "gate_engaged_intact_frac": on["gate_engaged_frac"],
            "gate_freeze_suppressed_intact_frac": on["gate_freeze_suppressed_frac"],
            "g1_transfer_intact_frac": on["g1_frac"],
            "g1_transfer_lesion_frac": off["g1_frac"],
        },
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_note": route_note,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
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
            "scheduled_external_hazard_enabled": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ENABLED,
            "scheduled_external_hazard_interval": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_INTERVAL,
            "scheduled_external_hazard_prob": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_PROB,
            "scheduled_external_hazard_adjacent_only": (
                HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_ADJACENT_ONLY
            ),
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
                     "(ree_core/pfc/infralimbic_avoidance_gate.py) driven in the "
                     "scaffolded_sd054_onboarding Stage-H under an SD-029 SCHEDULED "
                     "EXTERNAL-HAZARD field (interval=20, prob=0.9, adjacent_only=True), "
                     "with ambient env_drift reverted to defaults (5, 0.3) and the "
                     "SD-059/MECH-358 escape-affordance bridge + harm-pathway-training "
                     "amend both on for ALL arms",
        "validates": "MECH-357 (ilPFC-analog freeze-suppression + instrumental-avoidance "
                     "action pathway + eligibility-trace efficacy learning) -- SD-029 "
                     "scheduled-external-hazard successor of V3-EXQ-603s's mobile-predator "
                     "design, replacing an undirected continuous-drift pressure (which "
                     "603s's own per-seed data showed collapses to a spawn-position "
                     "lottery) with a discrete, agent-relative threat-onset event, per "
                     "mech357_freeze_incompatible_pressure_scoping_2026-08-10.md",
        "predecessor_run_id": "v3_exq_603s_instrumental_avoidance_freeze_incompatible_hazard_20260809T161324Z_v3",
        "predecessor_queue_id": "V3-EXQ-603s",
        "supersedes": None,
        "reuse_check_note": "GOV-REUSE-1: the decisive readout (G_H_LESION_frac + the "
                            "INTACT-beats-LESION discrimination under an SD-029 SCHEDULED "
                            "EXTERNAL-HAZARD Stage-H) is a NEW manipulation "
                            "(scheduled_external_hazard_enabled=True, never previously used "
                            "in this lineage -- 603g/603h/603r ran the static default, 603s "
                            "ran undirected mobile-predator drift) present in no recorded "
                            "run. Not reanalysis-recoverable -> ran fresh.",
        "design_note": "SD-029 scheduled-external-hazard successor of V3-EXQ-603s, per "
                       "mech357_freeze_incompatible_pressure_scoping_2026-08-10.md. 603s's "
                       "mobile-predator field (env_drift interval=1/prob=0.6) FAILED "
                       "discrimination on an EXACT TIE (G_H_LESION_frac == G_H_INTACT_frac "
                       "== 0.6667). The scoping spike traced this to env_drift being "
                       "UNDIRECTED -- hazards never target the agent -- which on a 10x10 "
                       "grid with 4 fast hazards produces a spawn-position lottery rather "
                       "than a graded conflict (603s's own per-seed data is strictly "
                       "bimodal: 200-step ceiling or ~5.5-6-step self-contamination death, "
                       "nothing between). 603t replaces that with SD-029 "
                       "scheduled_external_hazard (interval=20, prob=0.9, "
                       "adjacent_only=True), an already-built, previously-unused "
                       "CausalGridWorldV2 mechanism that deterministically places a hazard "
                       "adjacent to the agent's CURRENT position on a schedule -- "
                       "agent-relative and discrete by construction, removing the "
                       "spawn-lottery confound. Ambient env_drift reverted to defaults "
                       "(5, 0.3) so the discrete injection is the isolated new pressure "
                       "variable rather than stacking on 603s's confounded drift. 3 arms x "
                       "3 seeds [42,43,44] with the same TWO-SIDED discriminative-headroom "
                       "guard as 603s (unchanged: it worked as designed there -- the "
                       "mechanism, not the guard, is what changed): ARM_LESION (negative "
                       "control) must FAIL the gate (headroom from below); ARM_POSCTRL "
                       "(gated + reef spawn, positive control) must CLEAR it (survivability "
                       "exists / pressure not impossible). A mis-calibrated schedule "
                       "self-routes to a requeue rather than a false MECH-357 verdict.",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_pass_rule": "mechanism_ready AND G_H_LESION < 2/3 (headroom below) AND "
                                 "G_H_POSCTRL >= 2/3 (survivability above) AND G_H_INTACT >= "
                                 "2/3 AND G_H_INTACT_frac > G_H_LESION_frac",
            "g_h_hazard_stage_survival": "median episode length over last 10 Stage-H episodes >= 75",
            "freeze_incompatible_pressure": {
                "mechanism": "SD-029 scheduled_external_hazard",
                "scheduled_external_hazard_interval": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_INTERVAL,
                "scheduled_external_hazard_prob": HAZARD_STAGE_SCHEDULED_EXTERNAL_HAZARD_PROB,
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
