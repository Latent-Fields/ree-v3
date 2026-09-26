"""
V3-EXQ-1109 -- DCD2 Probe F: freeze -> release -> veto, EARLIEST-EDGE probe on the
V3-EXQ-1107 trained harness. Three legs (F0 / F1 / F3). DIAGNOSTIC.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
RED-TEAM (Step 4.5, fable): CONTESTED, 7 findings verified against source and fixed
(details in the RED_TEAM constant and the queue entry note).

WHY THIS RUN EXISTS
-------------------
Plan of record: REE_assembly evidence/planning/dynamic_control_discrimination_plan_20260926.md
(41cfe94841) sec 2 Family F, sec 3 rows F0/F1/F3, sec 4.2. It ABSORBS
chip-20260926-pag-freeze-lock-confirmer (the freeze on/off confirmer the
V3-EXQ-1107 / 1106 / 1090 autopsies asked for) and keeps that chip's two arms
(use_pag_freeze_gate True vs False) and its decision rule verbatim (F1 below),
and adds the representation leg (F0) and the veto leg (F3).

THE PIVOTAL QUESTION (F0, the earliest edge). V3-EXQ-1107 recorded ||z_harm_a||
IDENTICAL on all 90 eval ticks (3.87566 at the hazard, p1 and p2 checkpoints, to 5
decimals). Is the harm encoder input-insensitive (H-SV: signal validity fails at the
representation edge), or was its input static because the agent was frozen next to
a hazard? F0 answers with the freeze OFF: per-tick z_harm_a under a SCRIPTED random
walk at eval (training untouched), under the trained agent's own policy, and at the
untrained `init` checkpoint (the untrained-encoder reference). The hazard oracle
(nearest-hazard distance) is used for SCORING ONLY; it is never fed to any
controller.

PREMISE CORRECTED AT AUTHORING (re-measured against source at ree-v3 436a988742)
------------------------------------------------------------------------------
In THIS harness harm_obs_a is NOT the hazard-proximity EMA the CeA re-probe
(cea_onset_input_reprobe_20260925.md) measured. The 1107/935a env sets
limb_damage_enabled=True, so causal_grid_world.py (~4144-4158, SD-022) re-sources
harm_obs_a to 7 BODY-DAMAGE dims (limb_damage[4], max, mean, residual pain); the
50-d proximity EMA is the limb-damage-OFF path only. So the PAG freeze gate
(agent.py ~11464-11548, reads ||z_harm_a||) here reads an encoding of accumulated
body damage, and z_harm_a need not track proximity even if the encoder is perfect.
F0 therefore scores THREE readouts, so "input-insensitive encoder" and "the stream
does not carry proximity" cannot alias:
  R2_dist  held-out ridge R^2 of z_harm_a (16-d) -> nearest-hazard Manhattan distance
           (capped at DIST_CAP)                         [PRIMARY, the plan's wording]
  R2_fid   held-out ridge R^2 of z_harm_a -> the encoder's own input content (mean limb
           damage, harm_obs_a[5])                        [encoder fidelity]
  R2_in    held-out ridge R^2 of harm_obs_a (7-d, the encoder's actual input) ->
           nearest-hazard distance                       [input ceiling]
plus R2_hf (z_harm_a -> env hazard_field at the agent, the env's own proximity
definition; distance is capped at DIST_CAP and the bipartite layout keeps the agent
far from hazards on many ticks, so the capped fraction is recorded), the
affective-encoder hidden ReLU dead-unit fraction over the walk (a forward hook; no
ree_core edit), the within-life relative SD of ||z_harm_a||, and
resting_norm_over_theta = min walk ||z_harm_a|| / theta_freeze.
Red-team correction (verified, causal_grid_world.py ~2898-2911): limb damage accrues
ONLY on a MOVE with harm < 0 (STAY uses no limb), so a frozen agent's harm_obs_a is
constant BY CONSTRUCTION and 1107's 90 identical ticks are a tautology of the STAY
override + SD-022, not evidence about encoder sensitivity. F0 therefore asks the
question that remains: when the agent MOVES, does z_harm_a carry proximity? And the
lock itself can be input-independent: if resting ||z_harm_a|| >= theta_freeze the
exit (z < theta*tone) is unreachable at any input (gate_input_independent readout).
Third correction (verified, ~2590-2649): a STATIONARY agent contaminates its own cell
and is then harmed as "agent_caused_hazard"; the 6-step 1107 lives are contamination
deaths from standing still. Contacts are therefore split into true hazard
(env_caused_hazard) and contamination (agent_caused_hazard); F3 uses true hazards.
Second correction: 1107's identical lives came partly from ENV-SEED PINNING (every
eval life on the same hazard-adjacent spawn, 1107 autopsy sec 1). This run builds a
FRESH env per eval episode with its own derived seed (paired across arms), so F1 is
not read on one spawn.

DESIGN
------
Seeds 47, 48, 49 and ENV_SEED_BASE 1107 = V3-EXQ-1107's, so training reproduces
1107's agents on the same curriculum (935a scaffolded_sd054_onboarding, budgets and
substrate config byte-identical; RNG saved/restored around probes). Probes at init
(untrained), p0, final (after p2). Arms (clones of the SAME trained weights; per
episode: identical env seed and identical RNG reset in every own-policy arm):
  ARM_FZ_ON     use_pag_freeze_gate=True  (1107 config)          own policy  [F1]
  ARM_FZ_OFF    use_pag_freeze_gate=False                        own policy  [F1, F0 secondary]
  ARM_WALK      freeze off, SCRIPTED uniform random walk over the 4 moves, sense()
                only (policy never consulted); fresh env per episode, runs until
                WALK_MIN_TICKS (600) ticks or WALK_MAX_EPS (120) episodes -- the smoke
                measured random-walk lives of only 11-17 steps in this env  [F0 primary]
  ARM_VETO_OFF  freeze off, 1090 gate chain ON (go/no-go constitution + F-eligibility
                demotion + adaptive floor), endogenous producer OFF      [F3, final only]
  ARM_VETO_ON   freeze off, same chain, MECH-449 endogenous producer ON (1090 knobs)
                                                                         [F3, final only]
Freeze-OFF arms tick a SHADOW PAGFreezeGate (copy of the trained agent's gate config)
on the same ||z_harm_a|| each step: observation-only, never touches the action. It
records whether the gate WOULD have locked on the freeze-off trajectory.

F3 IS Q-111 LEG-1 PRECONDITION DATA, NOT A Q-111 TEST (Q-111's own note: do not queue
a Q-111 test until ARC-155/ARC-156 corrections exist). It records whether, with the
freeze removed, the executed action varies (1090's missing action-diversity
precondition) and whether the veto then changes executed action and hazard contact.

PRE-REGISTERED DECISION RULES (thresholds are constants in this file)
---------------------------------------------------------------------
F0 [LOAD-BEARING], per seed at the final checkpoint, ARM_WALK, held-out split by
episode parity. Seed F0-ready iff n_heldout >= MIN_HELDOUT and heldout distance SD >=
DIST_SD_MIN (positive control: the walk samples proximity). Seed class:
  (best = max(R2_dist, R2_hf); best_in = max(R2_in, R2_in_hf))
  HOLDS               best >= R2_TRACK (0.25)
  ENCODER_INSENSITIVE best < R2_FAIL (0.10) AND damage SD >= DAMAGE_SD_MIN AND
                      R2_fid < R2_FAIL          (encoder does not follow its own input)
  STREAM_CHOICE       best < R2_FAIL AND damage SD >= DAMAGE_SD_MIN AND R2_fid >= R2_TRACK
                      AND best_in < R2_FAIL     (encoder faithful; the damage stream
                                                 itself does not carry proximity)
  HSV_UNSUBTYPED      best < R2_FAIL, subtype not decidable
  INDETERMINATE       otherwise (incl. faithful encoder, input carries proximity, z not)
Run verdict over >= MIN_READY_SEEDS ready seeds, majority (>= 2/3):
  HOLDS -> F0_REPRESENTATION_HOLDS (1107's constancy was frozen input; go to F1-F3,
           ARC-155/ARC-156 designs at the PAG become runnable);
  any H-SV class -> F0_HSV_<majority subtype> (the freeze/veto/CeA failures share an
           upstream cause; route to the harm-encoder/stream owner; MECH-279 amend is
           the wrong target);
  else F0_INDETERMINATE; < MIN_READY_SEEDS ready -> F0_NOT_READY.
F1 (the chip's rule, kept verbatim), final checkpoint. collapse(arm) := mean episode
length <= COLLAPSE_LEN (15) OR executed-STAY fraction >= STATIONARY_FRAC (0.90)
(wall-pressing is recorded as blocked_move_fraction, not counted as collapse).
Seed LOCK iff ARM_FZ_ON collapses with freeze_active fraction >= FREEZE_FRAC_LOCK
(0.90) AND ARM_FZ_OFF does not collapse; BOTH iff both collapse; NO_LOCK iff FZ_ON
does not collapse. Run (>= 2/3 of seeds):
  LOCK    -> F1_LOCK_CONFIRMED (MECH-279 amend under GFLAG-0508 lineage; levers
             pag_max_freeze_duration or theta recalibration);
  BOTH and F0 = ENCODER_INSENSITIVE -> F1_BOTH_COLLAPSE_SATURATED_LATENT (harm-encoder
             saturation is the root; the MECH-279 amend is the wrong target);
  BOTH otherwise -> F1_BOTH_COLLAPSE_<F0 verdict>;  NO_LOCK -> F1_NO_LOCK_REPRODUCED;
  else F1_MIXED.
F3 (precondition data, not load-bearing), final checkpoint. Seed:
  DIVERSITY_COLLAPSED  ARM_VETO_OFF executed-action entropy < H_MIN_BITS (1.0 bit of
                       log2(5)=2.32) -- record the collapse target (modal action and
                       fraction, E3 committed fraction, beta-elevated fraction);
  VETO_NOT_LIVE        producer never armed or never fired;
  EFFECTIVE            attributable paired divergence on >= DIVERGE_FRAC_MIN (0.5) of
                       episodes AND TRUE-hazard contact rate ON <= (1 - CONTACT_REDUCTION
                       0.20) x OFF, with >= MIN_OFF_HAZARD_CONTACTS OFF true contacts;
  ACTIONS_CONTACT_UNMEASURABLE  divergence met, OFF hazard contacts too few;
  VETO_NO_CANDIDATE_CONTRAST  within-tick candidate harm spread p50 < sd_floor x z_thr
                       (0.02) or every fired tick was all-fired: nothing to veto BETWEEN
                       (upstream candidate homogeneity, f_dominance lineage), not H3;
  NO_ORGANISM_EFFECT   otherwise (diversity present, veto live, no safety effect).
Run (>= 2/3): EFFECTIVE -> F3_H3_WEAKENED (1090's inert veto was substrate removal by
  the freeze); NO_ORGANISM_EFFECT -> F3_H3_LIVE (genuine composition / authority
  defect); DIVERSITY_COLLAPSED -> F3_STARVED_BY_FAMILY_C (repair order C then F);
  VETO_NOT_LIVE -> F3_VETO_NOT_LIVE; ACTIONS_CONTACT_UNMEASURABLE ->
  F3_CONTACT_UNMEASURABLE; VETO_NO_CANDIDATE_CONTRAST -> F3_NO_CANDIDATE_CONTRAST;
  else F3_MIXED.
OUTCOME: PASS iff F0 reached a decisive verdict (HOLDS or H-SV) AND F1 reached a
non-MIXED verdict -- i.e. "the earliest edge was adjudicated", NOT "REE behaves".
F3 never gates. combination_rule is recorded in the manifest.

DV-SYMMETRY (Step 3.5)
  F0 walk vs own policy: the manipulation changes WHICH states are sampled; the DV
    (held-out R^2) is not invariant under it (a frozen trajectory has ~0 target SD).
  F1 freeze ON vs OFF: removes the STAY override of the executed action; DVs (episode
    length, stationary fraction, freeze fraction) are not invariant under it.
  F3 veto ON vs OFF: a per-candidate safety axis changes the eligible set -- not a
    uniform additive constant or monotone transform of the scores; the DV is the
    executed-action sequence and contact rate, which it can change.

STEP 2.4 (GOV-REUSE-1). Decisive readout: held-out R^2 of z_harm_a on hazard distance
under a MOVING policy on a trained 935a-curriculum agent, plus freeze on/off arms on
current main. Checked V3-EXQ-1107 (only ||z_harm_a|| summary stats of a frozen agent),
1090 (freeze on, pre-STAY-fix code), 1106 (100% frozen), 935a (no z_harm_a), and the
CeA re-probe (untrained, limb-damage-OFF 50-d EMA stream, onset d' not R^2). Not
recoverable -> run.

claim_ids: [] (experiment_purpose diagnostic). Bears on MECH-279, SD-011, MECH-449,
ARC-155, ARC-156, Q-111 (recorded under `bears_on`, never scored).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.pag.freeze_gate import PAGFreezeGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _derive_env_seed,
    _sd049_kwargs,
    _sense_with_optional_harm,
    _benefit_and_drive,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.arm_fingerprint import reset_all_rng  # noqa: E402
from experiments._lib.manifest_core import pin_recording_substrate  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1109_pag_freeze_veto_earliest_edge"
QUEUE_ID = "V3-EXQ-1109"
CLAIM_IDS: List[str] = []
BEARS_ON = ["MECH-279", "SD-011", "MECH-449", "ARC-155", "ARC-156", "Q-111", "GFLAG-0508"]
EXPERIMENT_PURPOSE = "diagnostic"
SOURCE_RECORD = (
    "REE_assembly evidence/planning/dynamic_control_discrimination_plan_20260926.md "
    "(41cfe94841) sec 4.2; absorbs chip-20260926-pag-freeze-lock-confirmer"
)
RED_TEAM = ("fable, Step 4.5: CONTESTED (7 findings, all verified against source and fixed: "
            "#1 contamination vs true-hazard contacts split; #2 collapse on executed STAY not "
            "stationary (wall-press recorded); #3 resting_norm_over_theta readout; #4 docstring; "
            "#5 hazard_field target + R2_in in STREAM_CHOICE + cap fraction; #6 "
            "VETO_NO_CANDIDATE_CONTRAST class; #7 armed read before select)")

ARM_FINGERPRINT_EXEMPT = (
    "arms are eval-time clones of ONE trained agent per seed (paired design, no "
    "per-cell training); arm_results cells are not independently trained and are "
    "never reuse candidates."
)
ANCHOR_REACHABILITY_EXEMPT = (
    "readiness preconditions are plain seed fractions of directly measured quantities "
    "(the curriculum reached the final checkpoint; the scripted walk sampled a "
    "distance SD >= 0.75 cells over >= 100 held-out ticks) -- not signature anchors."
)

SEEDS = [47, 48, 49]
CONDITION_LABEL = "TRAINED_935A_CURRICULUM_FREEZE_VETO_EARLIEST_EDGE"
ENV_SEED_BASE = 1107          # = V3-EXQ-1107, so training reproduces its agents
EVAL_ENV_STREAM = 7           # eval envs: fresh derived seed per episode (not pinned)

ARM_FZ_ON = "ARM_FZ_ON"
ARM_FZ_OFF = "ARM_FZ_OFF"
ARM_WALK = "ARM_WALK"
ARM_VETO_OFF = "ARM_VETO_OFF"
ARM_VETO_ON = "ARM_VETO_ON"
PROBE_ARMS = [ARM_FZ_ON, ARM_FZ_OFF, ARM_WALK]
FINAL_ARMS = [ARM_FZ_ON, ARM_FZ_OFF, ARM_WALK, ARM_VETO_OFF, ARM_VETO_ON]
CHECKPOINTS = ["init", "p0", "final"]

# --- eval budgets ----------------------------------------------------------
PROBE_OWN_EPS = 3
FINAL_OWN_EPS = 12
WALK_EPS = 8                  # nominal progress unit for the walk block
WALK_STEPS = 150
WALK_MIN_TICKS = 600          # walk until this many ticks (smoke: random-walk lives
WALK_MAX_EPS = 120            # are ~11-17 steps in this env, so 8 fixed eps would
                              # leave < MIN_HELDOUT held-out ticks)
MOVE_ACTIONS = [0, 1, 2, 3]
STAY_ACTION = 4

# --- pre-registered thresholds (never derived from this run) ---------------
R2_TRACK = 0.25
R2_FAIL = 0.10
DIST_CAP = 6
DIST_SD_MIN = 0.75
MIN_HELDOUT = 100
DAMAGE_SD_MIN = 1e-3
MIN_READY_SEEDS = 2
RIDGE_LAMBDA = 1e-2
COLLAPSE_LEN = 15.0
STATIONARY_FRAC = 0.90
FREEZE_FRAC_LOCK = 0.90
H_MIN_BITS = 1.0
DIVERGE_FRAC_MIN = 0.5
CONTACT_REDUCTION = 0.20
MIN_OFF_HAZARD_CONTACTS = 5
HARM_RANGE_PRECONDITION = 0.02
MIN_FRACTION = 2.0 / 3.0

# --- MECH-449 endogenous producer knobs (= V3-EXQ-1090 / ree_core defaults) -
GNG_SAFETY_Z_THRESHOLD = 2.0
GNG_SAFETY_EMA_DECAY = 0.999
GNG_SAFETY_SD_FLOOR = 0.01
GNG_SAFETY_WARMUP_SAMPLES = 200
GNG_SAFETY_FLOOR = 0.5

HAZARD_TT = ("agent_caused_hazard", "env_caused_hazard")
TRUE_HAZARD_TT = ("env_caused_hazard",)          # a real hazard cell
CONTAMINATION_TT = ("agent_caused_hazard",)      # self-contaminated cell (red-team #1)
RESOURCE_TT = ("resource", "resource_contact")

_ZG = ZGoalStreamAccumulator()

# --- curriculum + substrate: byte-identical to V3-EXQ-1107 / 935a ----------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
AFFINITY_INPUT_CAP_TRAIN = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0


# --------------------------------------------------------------------------
# Harness: copied VERBATIM from V3-EXQ-1107 (which copied V3-EXQ-935a)
# --------------------------------------------------------------------------
def _make_scaffold_cfg(dry_run: bool,
                       env_seed: Optional[int] = None) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 3, 3, 3, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
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
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_env_seed=env_seed,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """V3-EXQ-935a / 1107 substrate config, unchanged (freeze gate ON)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
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
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=False,
        use_external_task_drive=True,
        external_task_drive_affinity_weight=3.0,
        external_task_drive_salience_weight=2.0,
        external_task_drive_commit_weight=1.0,
        external_task_drive_proximity_weight=1.0,
        salience_affinity_input_cap=AFFINITY_INPUT_CAP_TRAIN,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_dual_cue_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                        seed: Optional[int] = None) -> CausalGridWorldV2:
    """P2-config foraging env WITH the GAP-3 dual_cue primitive (935a/1107 verbatim)."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
        seed=seed,
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=p2_hfa,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(1, 2),
        **_sd049_kwargs(scaffold_cfg),
    )


# --------------------------------------------------------------------------
# Arm construction
# --------------------------------------------------------------------------
def _clone(trained: REEAgent, device: torch.device, freeze_on: bool) -> REEAgent:
    """Clone the SAME trained weights into a fresh agent; freeze_on toggles
    use_pag_freeze_gate on the deep-copied config BEFORE construction (the gate is
    not a parameterised module, so the state_dict is identical across arms)."""
    cfg = copy.deepcopy(trained.config)
    cfg.use_pag_freeze_gate = bool(freeze_on)
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained.state_dict().items()}
    try:
        agent.load_state_dict(state)
        agent._clone_load_report = {"strict": True, "missing": [], "unexpected": []}
    except RuntimeError:
        res = agent.load_state_dict(state, strict=False)
        agent._clone_load_report = {"strict": False,
                                    "missing": list(getattr(res, "missing_keys", []))[:50],
                                    "unexpected": list(getattr(res, "unexpected_keys", []))[:50]}
    agent.e3._running_variance = float(trained.e3._running_variance)
    if trained.goal_state is not None and agent.goal_state is not None:
        agent.goal_state.load_state_dict(trained.goal_state.state_dict())
    if (agent.pag_freeze_gate is not None) != bool(freeze_on):
        raise RuntimeError(f"arm wiring check failed: freeze_on={freeze_on} but gate "
                           f"present={agent.pag_freeze_gate is not None}")
    return agent


def _arm_veto_chain(agent: REEAgent, producer_on: bool) -> Dict[str, Any]:
    """V3-EXQ-1090's _arm_gate on a clone, with the endogenous producer ON or OFF.
    Both F3 arms carry the identical selection chain, so the ONLY difference is the
    MECH-449 endogenous safety axis."""
    e3c = agent.config.e3
    if agent.e3.config is not e3c:
        raise RuntimeError("E3 selector does not share agent.config.e3")
    e3c.use_go_nogo_constitution = True
    e3c.use_f_eligibility_demotion = True
    e3c.use_f_eligibility_adaptive_floor = True
    e3c.f_eligibility_adaptive_mean_factor = 1.0
    e3c.use_gng_endogenous_safety = bool(producer_on)
    e3c.gng_safety_z_threshold = GNG_SAFETY_Z_THRESHOLD
    e3c.gng_safety_ema_decay = GNG_SAFETY_EMA_DECAY
    e3c.gng_safety_sd_floor = GNG_SAFETY_SD_FLOOR
    e3c.gng_safety_warmup_samples = GNG_SAFETY_WARMUP_SAMPLES
    e3c.gng_safety_floor = GNG_SAFETY_FLOOR
    agent.reset_gng_safety_state()
    return {"use_go_nogo_constitution": True, "use_f_eligibility_demotion": True,
            "use_f_eligibility_adaptive_floor": True,
            "use_gng_endogenous_safety": bool(producer_on),
            "gng_safety_warmup_samples": GNG_SAFETY_WARMUP_SAMPLES}


def _build_arm(trained: REEAgent, device: torch.device, arm: str) -> REEAgent:
    agent = _clone(trained, device, freeze_on=(arm == ARM_FZ_ON))
    if arm == ARM_VETO_OFF:
        agent._veto_chain = _arm_veto_chain(agent, producer_on=False)
    elif arm == ARM_VETO_ON:
        agent._veto_chain = _arm_veto_chain(agent, producer_on=True)
    return agent


# --------------------------------------------------------------------------
# RNG save/restore so probes never perturb the training trajectory (1107)
# --------------------------------------------------------------------------
def _harness_modules() -> List[Any]:
    mods = []
    for name in ("experiments._harness", "_harness"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "_action_random") and m not in mods:
            mods.append(m)
    return mods


def _save_rng() -> Dict[str, Any]:
    return {"py": random.getstate(), "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "harness": [(m, m._action_random.getstate()) for m in _harness_modules()]}


def _restore_rng(st: Dict[str, Any]) -> None:
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    for m, s in st["harness"]:
        m._action_random.setstate(s)


# --------------------------------------------------------------------------
# Scoring helpers (oracle used for SCORING ONLY)
# --------------------------------------------------------------------------
def _nearest_hazard_distance(env: CausalGridWorldV2) -> float:
    ax, ay = int(env.agent_x), int(env.agent_y)
    hz = list(getattr(env, "hazards", []) or [])
    if not hz:
        return float(DIST_CAP)
    d = min(abs(ax - int(h[0])) + abs(ay - int(h[1])) for h in hz)
    return float(min(d, DIST_CAP))


def _hazard_at_agent(env: CausalGridWorldV2) -> Optional[float]:
    try:
        return float(np.clip(env.hazard_field[int(env.agent_x), int(env.agent_y)], 0.0, 1.0))
    except Exception:
        return None


def _ridge_r2(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> Dict[str, Any]:
    """Held-out R^2 of a standardized ridge fit (closed form, intercept)."""
    out = {"r2": None, "n_train": int(train.sum()), "n_test": int(test.sum()),
           "target_sd_test": None}
    if train.sum() < 10 or test.sum() < 10:
        return out
    Xtr, Xte, ytr, yte = X[train], X[test], y[train], y[test]
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    ym = ytr.mean()
    A = Ztr.T @ Ztr + RIDGE_LAMBDA * len(ytr) * np.eye(Ztr.shape[1])
    w = np.linalg.solve(A, Ztr.T @ (ytr - ym))
    pred = Zte @ w + ym
    sst = float(((yte - yte.mean()) ** 2).sum())
    out["target_sd_test"] = round(float(yte.std()), 5)
    if sst < 1e-12:
        return out
    out["r2"] = round(1.0 - float(((yte - pred) ** 2).sum()) / sst, 5)
    return out


def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not v:
        return {"n": 0, "mean": None, "sd": None, "p10": None, "p50": None, "p90": None,
                "min": None, "max": None}
    a = np.array(v, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 5), "sd": round(float(a.std()), 6),
            "p10": round(float(np.quantile(a, 0.1)), 5), "p50": round(float(np.quantile(a, 0.5)), 5),
            "p90": round(float(np.quantile(a, 0.9)), 5), "min": round(float(a.min()), 5),
            "max": round(float(a.max()), 5)}


def _entropy_bits(actions: List[int], n: int = 5) -> Optional[float]:
    if not actions:
        return None
    c = np.bincount(np.array(actions, dtype=int), minlength=n).astype(float)
    p = c / c.sum()
    p = p[p > 0]
    return round(float(-(p * np.log2(p)).sum()), 5)


def _within_life_rel_sd(norms_by_ep: List[List[float]]) -> Optional[float]:
    vals = []
    for ns in norms_by_ep:
        if len(ns) >= 3:
            a = np.array(ns, dtype=float)
            m = abs(float(a.mean()))
            vals.append(float(a.std()) / m if m > 1e-12 else 0.0)
    return round(float(np.mean(vals)), 6) if vals else None


def _f0_block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Held-out R^2 block from per-tick rows (split by episode parity)."""
    if not rows:
        return {"n_ticks": 0}
    X_z = np.array([r["z"] for r in rows], dtype=float)
    X_in = np.array([r["hoa"] for r in rows], dtype=float)
    norm = np.array([[r["zn"]] for r in rows], dtype=float)
    dist = np.array([r["dist"] for r in rows], dtype=float)
    hf = np.array([r.get("hf") if r.get("hf") is not None else np.nan for r in rows], dtype=float)
    hf_ok = ~np.isnan(hf)
    dmg = np.array([r["dmg"] for r in rows], dtype=float)
    ep = np.array([r["ep"] for r in rows], dtype=int)
    test = (ep % 2) == 1
    train = ~test
    by_ep: Dict[int, List[float]] = {}
    for r in rows:
        by_ep.setdefault(r["ep"], []).append(r["zn"])
    return {
        "n_ticks": int(len(rows)),
        "r2_dist": _ridge_r2(X_z, dist, train, test),
        "r2_fid": _ridge_r2(X_z, dmg, train, test),
        "r2_in": _ridge_r2(X_in, dist, train, test),
        "r2_norm_dist": _ridge_r2(norm, dist, train, test),
        "r2_hf": (_ridge_r2(X_z[hf_ok], hf[hf_ok], train[hf_ok], test[hf_ok])
                  if hf_ok.sum() >= 20 else {"r2": None}),
        "r2_in_hf": (_ridge_r2(X_in[hf_ok], hf[hf_ok], train[hf_ok], test[hf_ok])
                     if hf_ok.sum() >= 20 else {"r2": None}),
        "hazard_field_at_agent": _stats(hf[hf_ok].tolist()),
        "frac_ticks_distance_at_cap": round(float((dist >= DIST_CAP).mean()), 5),
        "distance": _stats(dist.tolist()),
        "distance_sd_heldout": round(float(dist[test].std()), 5) if test.any() else None,
        "damage_mean": _stats(dmg.tolist()),
        "damage_sd": round(float(dmg.std()), 6),
        "z_harm_a_l2": _stats(norm[:, 0].tolist()),
        "z_harm_a_within_life_rel_sd": _within_life_rel_sd(list(by_ep.values())),
        "n_distinct_z_norm_6dp": int(len({round(float(x), 6) for x in norm[:, 0]})),
    }


# --------------------------------------------------------------------------
# Scripted random walk (ARM_WALK): sense() only, policy never consulted
# --------------------------------------------------------------------------
def _eval_walk(agent: REEAgent, scaffold_cfg, device, env_seeds: List[Optional[int]],
               rng_base: int, steps: int, min_ticks: int) -> Dict[str, Any]:
    agent.eval()
    enc = getattr(agent.latent_stack, "affective_harm_encoder", None)
    hidden_active: Optional[np.ndarray] = None
    hidden_n = [0]
    handle = None
    if enc is not None and isinstance(getattr(enc, "encoder", None), torch.nn.Sequential):
        def _hook(_m, _i, out):
            nonlocal hidden_active
            a = (out.detach().reshape(-1, out.shape[-1]) > 0).any(dim=0).cpu().numpy()
            hidden_active = a if hidden_active is None else (hidden_active | a)
            hidden_n[0] += 1
        handle = enc.encoder[1].register_forward_hook(_hook)
    rows: List[Dict[str, Any]] = []
    ep_lengths: List[int] = []
    try:
        with torch.no_grad():
            for ep, es in enumerate(env_seeds):
                env = _build_dual_cue_env(scaffold_cfg, seed=es)
                _, od = env.reset()
                agent.reset()
                reset_all_rng(rng_base + ep)
                walker = random.Random(rng_base * 7919 + ep)
                n = 0
                for _t in range(steps):
                    lat = _sense_with_optional_harm(agent, od["body_state"].to(device),
                                                    od["world_state"].to(device), od, device, True)
                    zha = getattr(lat, "z_harm_a", None)
                    if zha is not None:
                        hoa = od.get("harm_obs_a")
                        hoa_l = ([float(x) for x in hoa.reshape(-1).tolist()]
                                 if hoa is not None else [0.0] * HARM_OBS_A_DIM)
                        rows.append({"ep": ep, "z": [float(x) for x in zha.detach().reshape(-1).tolist()],
                                     "zn": float(zha.detach().norm().item()),
                                     "hoa": hoa_l, "dmg": hoa_l[5] if len(hoa_l) > 5 else 0.0,
                                     "dist": _nearest_hazard_distance(env),
                                     "hf": _hazard_at_agent(env)})
                    a = walker.choice(MOVE_ACTIONS)
                    _, _h, done, _info, od = env.step(a)
                    n += 1
                    if done:
                        break
                ep_lengths.append(n)
                if len(rows) >= min_ticks:
                    break
    finally:
        if handle is not None:
            handle.remove()
    blk = _f0_block(rows)
    blk["arm"] = ARM_WALK
    blk["episode_lengths"] = ep_lengths
    blk["encoder_hidden_units"] = int(hidden_active.size) if hidden_active is not None else None
    blk["encoder_dead_unit_fraction"] = (
        round(1.0 - float(hidden_active.mean()), 5) if hidden_active is not None else None)
    blk["encoder_hook_calls"] = hidden_n[0]
    return blk


# --------------------------------------------------------------------------
# Own-policy eval (FZ_ON / FZ_OFF / VETO_OFF / VETO_ON)
# --------------------------------------------------------------------------
def _eval_policy(agent: REEAgent, arm: str, trained_pag_cfg, scaffold_cfg, device,
                 env_seeds: List[Optional[int]], rng_base: int, steps: int) -> Dict[str, Any]:
    agent.eval()
    world_dim = agent.config.latent.world_dim
    gate = agent.pag_freeze_gate
    ia = agent.instrumental_avoidance
    veto = arm in (ARM_VETO_OFF, ARM_VETO_ON)
    shadow = (PAGFreezeGate(copy.deepcopy(trained_pag_cfg))
              if (gate is None and trained_pag_cfg is not None) else None)
    rows: List[Dict[str, Any]] = []
    actions_all: List[int] = []
    per_ep: List[Dict[str, Any]] = []
    n_commits = n_releases = n_suppressed = 0
    tick_harm_means: List[float] = []
    within_tick_spreads: List[float] = []
    with torch.no_grad():
        for ep, es in enumerate(env_seeds):
            env = _build_dual_cue_env(scaffold_cfg, seed=es)
            _, od = env.reset()
            agent.reset()
            if shadow is not None:
                shadow.reset()
            reset_all_rng(rng_base + ep)
            c0 = (gate._n_commits, gate._n_releases) if gate is not None else (0, 0)
            s0 = int(getattr(ia, "_n_freeze_suppressed", 0)) if ia is not None else 0
            acts: List[int] = []
            armed_seq: List[bool] = []
            applied_seq: List[int] = []
            n = fz = sfz = stat = haz = thz = ctm = res = comm = beta = 0
            hsum = 0.0
            for _t in range(steps):
                pos0 = (int(env.agent_x), int(env.agent_y))
                dist = _nearest_hazard_distance(env)
                hfa = _hazard_at_agent(env)
                hoa = od.get("harm_obs_a")
                hoa_l = ([float(x) for x in hoa.reshape(-1).tolist()] if hoa is not None
                         else [0.0] * HARM_OBS_A_DIM)
                lat = _sense_with_optional_harm(agent, od["body_state"].to(device),
                                                od["world_state"].to(device), od, device, True)
                if agent.goal_state is not None:  # native drive_level feed, as 1107
                    _b, _drv = _benefit_and_drive(od["body_state"].to(device))
                    agent.goal_state._last_drive_level = float(_drv)
                ticks = agent.clock.advance()
                e1p = (agent._e1_tick(lat) if ticks.get("e1_tick")
                       else torch.zeros(1, world_dim, device=device))
                cand = agent.generate_trajectories(lat, e1p, ticks)
                if veto:
                    g0 = agent.gng_safety_diagnostics()
                    armed_seq.append(int(g0.get("n_samples", 0)) >= GNG_SAFETY_WARMUP_SAMPLES)
                    agent.e3.last_score_diagnostics = None
                act = agent.select_action(cand, ticks)
                ai = int(act.argmax(dim=-1).item())
                zha = getattr(lat, "z_harm_a", None)
                zn = float(zha.detach().norm().item()) if zha is not None else float("nan")
                if gate is not None and getattr(agent, "_pag_last_output", None) is not None:
                    fz += int(bool(agent._pag_last_output.freeze_active))
                if shadow is not None and math.isfinite(zn):
                    sfz += int(bool(shadow.tick(z_harm_a_norm=zn).freeze_active))
                comm += int(getattr(agent.e3, "_committed_trajectory", None) is not None)
                bg = getattr(agent, "beta_gate", None)
                beta += int(bool(getattr(bg, "is_elevated", False))) if bg is not None else 0
                if veto:
                    g1 = agent.gng_safety_diagnostics()
                    applied_seq.append(int(g1.get("n_safety_nogo_applied", 0)))
                    if int(g1.get("n_ticks_scored", 0)) > int(g0.get("n_ticks_scored", 0)):
                        last = g1.get("last") or {}
                        hv = last.get("harm") or []
                        if hv:
                            tick_harm_means.append(float(np.mean(hv)))
                            within_tick_spreads.append(float(max(hv) - min(hv)))
                if zha is not None:
                    rows.append({"ep": ep, "z": [float(x) for x in zha.detach().reshape(-1).tolist()],
                                 "zn": zn, "hoa": hoa_l,
                                 "dmg": hoa_l[5] if len(hoa_l) > 5 else 0.0, "dist": dist,
                                 "hf": hfa})
                acts.append(ai)
                _, h, done, info, od = env.step(ai)
                n += 1
                stat += int((int(env.agent_x), int(env.agent_y)) == pos0)
                tt = str((info or {}).get("transition_type", ""))
                haz += int(tt in HAZARD_TT)
                thz += int(tt in TRUE_HAZARD_TT)
                ctm += int(tt in CONTAMINATION_TT)
                res += int(tt in RESOURCE_TT)
                try:
                    hsum += float(h)
                except (TypeError, ValueError):
                    pass
                if done:
                    break
            if gate is not None:
                n_commits += gate._n_commits - c0[0]
                n_releases += gate._n_releases - c0[1]
            if ia is not None:
                n_suppressed += int(getattr(ia, "_n_freeze_suppressed", 0)) - s0
            actions_all.extend(acts)
            per_ep.append({"ep": ep, "env_seed": es, "len": n, "freeze_steps": fz,
                           "shadow_freeze_steps": sfz, "stationary_steps": stat,
                           "hazard_contacts": haz, "true_hazard_contacts": thz,
                           "contamination_contacts": ctm, "resource_contacts": res,
                           "committed_steps": comm, "beta_elevated_steps": beta,
                           "harm_sum": round(hsum, 5), "actions": acts,
                           "armed_seq": armed_seq if veto else None,
                           "applied_cum_seq": applied_seq if veto else None})
    total = sum(e["len"] for e in per_ep)
    counts = np.bincount(np.array(actions_all, dtype=int), minlength=5) if actions_all else np.zeros(5)
    modal = int(np.argmax(counts)) if total else None
    out = {
        "arm": arm, "n_episodes": len(per_ep), "total_steps": total,
        "mean_episode_length": round(total / len(per_ep), 3) if per_ep else 0.0,
        "freeze_gate_present": gate is not None,
        "freeze_active_fraction": round(sum(e["freeze_steps"] for e in per_ep) / total, 5) if (total and gate is not None) else None,
        "shadow_would_freeze_fraction": round(sum(e["shadow_freeze_steps"] for e in per_ep) / total, 5) if (total and shadow is not None) else None,
        "n_commits": int(n_commits), "n_releases": int(n_releases),
        "mech357_n_freeze_suppressed": int(n_suppressed),
        "stationary_fraction": round(sum(e["stationary_steps"] for e in per_ep) / total, 5) if total else None,
        "hazard_contacts": int(sum(e["hazard_contacts"] for e in per_ep)),
        "hazard_contact_rate": round(sum(e["hazard_contacts"] for e in per_ep) / total, 5) if total else None,
        "true_hazard_contacts": int(sum(e["true_hazard_contacts"] for e in per_ep)),
        "true_hazard_contact_rate": round(sum(e["true_hazard_contacts"] for e in per_ep) / total, 5) if total else None,
        "contamination_contacts": int(sum(e["contamination_contacts"] for e in per_ep)),
        "stay_fraction": round(float(counts[STAY_ACTION]) / total, 5) if total else None,
        "blocked_move_fraction": (round(max(0.0, sum(e["stationary_steps"] for e in per_ep) / total
                                            - float(counts[STAY_ACTION]) / total), 5) if total else None),
        "resource_contacts": int(sum(e["resource_contacts"] for e in per_ep)),
        "contact_rate": round(sum(e["resource_contacts"] for e in per_ep) / total, 5) if total else None,
        "executed_action_counts": [int(x) for x in counts],
        "executed_action_entropy_bits": _entropy_bits(actions_all),
        "modal_action": modal,
        "modal_action_fraction": round(float(counts.max()) / total, 5) if total else None,
        "e3_committed_fraction": round(sum(e["committed_steps"] for e in per_ep) / total, 5) if total else None,
        "beta_elevated_fraction": round(sum(e["beta_elevated_steps"] for e in per_ep) / total, 5) if total else None,
        "mean_episode_harm_sum": round(float(np.mean([e["harm_sum"] for e in per_ep])), 5) if per_ep else None,
        "clone_load_report": getattr(agent, "_clone_load_report", None),
        "f0_own_policy": _f0_block(rows),
        "per_episode": per_ep,
    }
    if veto:
        gd = agent.gng_safety_diagnostics()
        gd.pop("last", None)
        nc = int(gd.get("n_candidates_scored", 0))
        out["veto_chain"] = getattr(agent, "_veto_chain", None)
        out["producer_diagnostics"] = {k: (float(v) if isinstance(v, float) else v)
                                       for k, v in gd.items() if not isinstance(v, (list, dict))}
        out["veto_fire_rate"] = round(float(gd.get("n_signal_fired", 0)) / nc, 6) if nc else None
        out["producer_armed"] = bool(int(gd.get("n_samples", 0)) >= GNG_SAFETY_WARMUP_SAMPLES)
        out["tick_harm_mean"] = _stats(tick_harm_means)
        out["tick_harm_range"] = (round(max(tick_harm_means) - min(tick_harm_means), 6)
                                  if tick_harm_means else None)
        out["within_tick_harm_spread"] = _stats(within_tick_spreads)
    return out


def _paired_divergence(off: Dict[str, Any], on: Dict[str, Any]) -> Dict[str, Any]:
    """Per episode (same env seed, same RNG reset): first step where the executed
    actions differ. Attributable iff the ON producer was ARMED at that step (a
    divergence before arming means the pairing is broken, recorded, never credited)."""
    rows = []
    n_attr = n_unattr = 0
    for eo, en in zip(off["per_episode"], on["per_episode"]):
        a, b = eo["actions"], en["actions"]
        m = min(len(a), len(b))
        k = next((i for i in range(m) if a[i] != b[i]), None)
        if k is None and len(a) != len(b):
            k = m
        armed = None
        applied_before = None
        if k is not None:
            aseq = en.get("armed_seq") or []
            cseq = en.get("applied_cum_seq") or []
            idx = min(k, len(aseq) - 1) if aseq else None
            armed = bool(aseq[idx]) if idx is not None and idx >= 0 else False
            applied_before = int(cseq[idx]) if (cseq and idx is not None and idx >= 0) else 0
            if armed:
                n_attr += 1
            else:
                n_unattr += 1
        rows.append({"ep": eo["ep"], "first_divergence_step": k,
                     "armed_at_divergence": armed,
                     "cum_hard_veto_applied_at_divergence": applied_before})
    n = len(rows)
    return {"per_episode": rows, "n_episodes": n, "n_attributable_divergent": n_attr,
            "n_unattributed_divergent": n_unattr,
            "attributable_divergence_fraction": round(n_attr / n, 5) if n else None}


# --------------------------------------------------------------------------
# Probe / final evaluation
# --------------------------------------------------------------------------
def _env_seeds(seed: int, ck_idx: int, n: int) -> List[Optional[int]]:
    return [_derive_env_seed(ENV_SEED_BASE + seed, stream=EVAL_ENV_STREAM, idx=ck_idx * 1000 + e)
            for e in range(n)]


def _evaluate(trained: REEAgent, scaffold_cfg, device, seed: int, label: str,
              arms: List[str], own_eps: int, walk_eps: int, steps: int,
              walk_min_ticks: int = WALK_MIN_TICKS) -> Dict[str, Any]:
    ck = CHECKPOINTS.index(label)
    own_seeds = _env_seeds(seed, ck, own_eps)
    walk_seeds = _env_seeds(seed, ck + 10, walk_eps)  # walk_eps = MAX episodes
    rng_base = seed * 100000 + ck * 1000
    pag_cfg = copy.deepcopy(trained.pag_freeze_gate.config) if trained.pag_freeze_gate is not None else None
    out: Dict[str, Any] = {}
    for arm in arms:
        reset_all_rng(rng_base)
        agent = _build_arm(trained, device, arm)
        if arm == ARM_WALK:
            out[arm] = _eval_walk(agent, scaffold_cfg, device, walk_seeds, rng_base + 500,
                                  min(WALK_STEPS, steps) if steps >= 30 else steps,
                                  walk_min_ticks)
        else:
            out[arm] = _eval_policy(agent, arm, pag_cfg, scaffold_cfg, device, own_seeds,
                                    rng_base, steps)
        _ZG.observe(agent)
        del agent
    if ARM_VETO_OFF in out and ARM_VETO_ON in out:
        out["paired_divergence_veto"] = _paired_divergence(out[ARM_VETO_OFF], out[ARM_VETO_ON])
    return out


def _probe(trained, scaffold_cfg, device, seed, label, arms, own_eps, walk_eps, steps,
           walk_min_ticks):
    st = _save_rng()
    try:
        res = _evaluate(trained, scaffold_cfg, device, seed, label, arms, own_eps, walk_eps,
                        steps, walk_min_ticks)
    finally:
        _restore_rng(st)
    w = res.get(ARM_WALK, {})
    on = res.get(ARM_FZ_ON, {})
    off = res.get(ARM_FZ_OFF, {})
    print(f"  [probe] seed={seed} ckpt={label}"
          f" walk_r2_dist={(w.get('r2_dist') or {}).get('r2')} r2_fid={(w.get('r2_fid') or {}).get('r2')}"
          f" r2_in={(w.get('r2_in') or {}).get('r2')} dead={w.get('encoder_dead_unit_fraction')}"
          f" | FZ_ON len={on.get('mean_episode_length')} fz={on.get('freeze_active_fraction')}"
          f" | FZ_OFF len={off.get('mean_episode_length')} H={off.get('executed_action_entropy_bits')}"
          f" shadow_fz={off.get('shadow_would_freeze_fraction')}", flush=True)
    return {"checkpoint": label, "arms": res}


# --------------------------------------------------------------------------
# Per-seed classification (pre-registered)
# --------------------------------------------------------------------------
def _r2(block: Dict[str, Any], key: str) -> Optional[float]:
    v = (block.get(key) or {}).get("r2")
    return None if v is None else float(v)


def _f0_seed(walk: Dict[str, Any]) -> Dict[str, Any]:
    n_test = int((walk.get("r2_dist") or {}).get("n_test") or 0)
    dsd = walk.get("distance_sd_heldout")
    ready = bool(n_test >= MIN_HELDOUT and dsd is not None and dsd >= DIST_SD_MIN)
    rd, rf, rh = _r2(walk, "r2_dist"), _r2(walk, "r2_fid"), _r2(walk, "r2_hf")
    ri, rih = _r2(walk, "r2_in"), _r2(walk, "r2_in_hf")
    dmg_ok = bool((walk.get("damage_sd") or 0.0) >= DAMAGE_SD_MIN)
    best = max([x for x in (rd, rh) if x is not None], default=None)   # proximity: either target
    best_in = max([x for x in (ri, rih) if x is not None], default=None)
    if not ready or best is None:
        cls = "NOT_READY"
    elif best >= R2_TRACK:
        cls = "HOLDS"
    elif best < R2_FAIL:
        if dmg_ok and rf is not None and rf < R2_FAIL:
            cls = "ENCODER_INSENSITIVE"
        elif (dmg_ok and rf is not None and rf >= R2_TRACK
              and best_in is not None and best_in < R2_FAIL):
            cls = "STREAM_CHOICE"
        elif dmg_ok and rf is not None and rf >= R2_TRACK:
            cls = "INDETERMINATE"   # encoder faithful, input carries proximity, z does not
        else:
            cls = "HSV_UNSUBTYPED"
    else:
        cls = "INDETERMINATE"
    zmin = (walk.get("z_harm_a_l2") or {}).get("min")
    return {"ready": ready, "class": cls, "r2_dist": rd, "r2_hf": rh, "r2_fid": rf,
            "r2_in": ri, "r2_in_hf": rih, "r2_norm_dist": _r2(walk, "r2_norm_dist"),
            "frac_ticks_distance_at_cap": walk.get("frac_ticks_distance_at_cap"),
            "resting_norm_over_theta": (round(float(zmin) / PAG_THETA_FREEZE, 5)
                                        if zmin is not None else None),
            "n_heldout": n_test, "distance_sd_heldout": dsd, "damage_sd": walk.get("damage_sd"),
            "encoder_dead_unit_fraction": walk.get("encoder_dead_unit_fraction"),
            "z_harm_a_within_life_rel_sd": walk.get("z_harm_a_within_life_rel_sd")}


def _collapse(arm: Dict[str, Any]) -> bool:
    # executed-STAY fraction, not position-stationary (red-team #2: wall-pressing is
    # recorded as blocked_move_fraction, not counted as a freeze-like collapse)
    return bool((arm.get("mean_episode_length") or 0.0) <= COLLAPSE_LEN
                or (arm.get("stay_fraction") or 0.0) >= STATIONARY_FRAC)


def _f1_seed(on: Dict[str, Any], off: Dict[str, Any]) -> Dict[str, Any]:
    c_on, c_off = _collapse(on), _collapse(off)
    fz = on.get("freeze_active_fraction") or 0.0
    if c_on and fz >= FREEZE_FRAC_LOCK and not c_off:
        cls = "LOCK"
    elif c_on and c_off:
        cls = "BOTH"
    elif not c_on:
        cls = "NO_LOCK"
    else:
        cls = "OTHER"
    return {"class": cls, "on_collapse": c_on, "off_collapse": c_off,
            "on_freeze_active_fraction": on.get("freeze_active_fraction"),
            "on_mean_len": on.get("mean_episode_length"), "off_mean_len": off.get("mean_episode_length"),
            "on_stationary": on.get("stationary_fraction"), "off_stationary": off.get("stationary_fraction"),
            "on_stay_fraction": on.get("stay_fraction"), "off_stay_fraction": off.get("stay_fraction"),
            "off_blocked_move_fraction": off.get("blocked_move_fraction"),
            "on_true_hazard_contacts": on.get("true_hazard_contacts"),
            "on_contamination_contacts": on.get("contamination_contacts"),
            "off_shadow_would_freeze_fraction": off.get("shadow_would_freeze_fraction"),
            "freeze_engaged": bool(fz > 0.0)}


def _f3_seed(voff: Dict[str, Any], von: Dict[str, Any], pdiv: Dict[str, Any]) -> Dict[str, Any]:
    h = voff.get("executed_action_entropy_bits")
    div_ok = bool(h is not None and h >= H_MIN_BITS)
    live = bool(von.get("producer_armed") and int((von.get("producer_diagnostics") or {}).get("n_signal_fired", 0)) > 0)
    dfrac = pdiv.get("attributable_divergence_fraction") or 0.0
    off_haz = int(voff.get("true_hazard_contacts") or 0)
    r_off, r_on = voff.get("true_hazard_contact_rate"), von.get("true_hazard_contact_rate")
    spread = (von.get("within_tick_harm_spread") or {}).get("p50")
    pdg = von.get("producer_diagnostics") or {}
    no_contrast = bool((spread is not None and spread < GNG_SAFETY_SD_FLOOR * GNG_SAFETY_Z_THRESHOLD)
                       or (int(pdg.get("n_ticks_any_fired", 0)) > 0
                           and int(pdg.get("n_ticks_all_fired", 0)) == int(pdg.get("n_ticks_any_fired", 0))))
    contact_eval = bool(off_haz >= MIN_OFF_HAZARD_CONTACTS and r_off is not None and r_on is not None)
    reduced = bool(contact_eval and r_on <= (1.0 - CONTACT_REDUCTION) * r_off)
    if not div_ok:
        cls = "DIVERSITY_COLLAPSED"
    elif not live:
        cls = "VETO_NOT_LIVE"
    elif dfrac >= DIVERGE_FRAC_MIN and reduced:
        cls = "EFFECTIVE"
    elif dfrac >= DIVERGE_FRAC_MIN and not contact_eval:
        cls = "ACTIONS_CONTACT_UNMEASURABLE"
    elif no_contrast:
        cls = "VETO_NO_CANDIDATE_CONTRAST"   # red-team #6: nothing to veto between
    else:
        cls = "NO_ORGANISM_EFFECT"
    return {"class": cls, "no_candidate_contrast": no_contrast, "veto_off_entropy_bits": h, "diversity_ok": div_ok, "veto_live": live,
            "attributable_divergence_fraction": dfrac,
            "n_unattributed_divergent": pdiv.get("n_unattributed_divergent"),
            "off_hazard_contacts": off_haz, "off_hazard_contact_rate": r_off,
            "on_hazard_contact_rate": r_on, "contact_evaluable": contact_eval,
            "contact_reduced": reduced, "veto_fire_rate": von.get("veto_fire_rate"),
            "harm_range_ok": bool((von.get("tick_harm_range") or 0.0) >= HARM_RANGE_PRECONDITION),
            "tick_harm_range": von.get("tick_harm_range"),
            "within_tick_harm_spread_p50": (von.get("within_tick_harm_spread") or {}).get("p50"),
            "collapse_target": {"modal_action": voff.get("modal_action"),
                                "modal_action_fraction": voff.get("modal_action_fraction"),
                                "e3_committed_fraction": voff.get("e3_committed_fraction"),
                                "beta_elevated_fraction": voff.get("beta_elevated_fraction")}}


# --------------------------------------------------------------------------
# One seed
# --------------------------------------------------------------------------
def _aborted(seed, stage, reason, ckpts) -> Dict[str, Any]:
    return {"seed": seed, "aborted_at": stage, "abort_reason": reason, "checkpoints": ckpts,
            "final": None, "reached_final": False}


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_env_base = ENV_SEED_BASE + seed
    scaffold_cfg = _make_scaffold_cfg(dry_run, env_seed=seed_env_base)
    device = torch.device("cpu")
    steps = scaffold_cfg.scaffold_steps_per_episode
    probe_own = 1 if dry_run else PROBE_OWN_EPS
    final_own = 2 if dry_run else FINAL_OWN_EPS
    walk_eps = 2 if dry_run else WALK_EPS            # nominal progress unit
    walk_max = 10 if dry_run else WALK_MAX_EPS
    walk_min = 60 if dry_run else WALK_MIN_TICKS

    # identical construction to 1107 (the probe env draws RNG before the agent)
    probe_env = _build_dual_cue_env(scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=0))
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0
    ckpts: List[Dict[str, Any]] = []

    def _ck(label: str) -> None:
        nonlocal done
        ckpts.append(_probe(agent, scaffold_cfg, device, seed, label, PROBE_ARMS,
                            probe_own, walk_max, steps, walk_min))
        done += 2 * probe_own + walk_eps
        print(f"  [train] probe_{label} seed={seed} ep {done}/{total_eps}", flush=True)

    _ck("init")
    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0", flush=True)
        return _aborted(seed, "stage0", s0.abort_reason, ckpts)
    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b", flush=True)
        return _aborted(seed, "stage0b", s0b.abort_reason, ckpts)
    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0", flush=True)
        return _aborted(seed, "p0", p0.abort_reason, ckpts)
    _ck("p0")
    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard", flush=True)
        return _aborted(seed, "hazard", hz.abort_reason, ckpts)
    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}", flush=True)
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f}", flush=True)
    _ZG.observe(agent)

    final = _evaluate(agent, scaffold_cfg, device, seed, "final", FINAL_ARMS,
                      final_own, walk_max, steps, walk_min)
    done += 4 * final_own + walk_eps
    print(f"  [train] final_eval seed={seed} ep {done}/{total_eps}", flush=True)

    f0 = _f0_seed(final[ARM_WALK])
    f1 = _f1_seed(final[ARM_FZ_ON], final[ARM_FZ_OFF])
    f3 = _f3_seed(final[ARM_VETO_OFF], final[ARM_VETO_ON], final["paired_divergence_veto"])
    print(f"  [eval] seed={seed} F0={f0['class']} r2_dist={f0['r2_dist']} r2_fid={f0['r2_fid']}"
          f" r2_in={f0['r2_in']} | F1={f1['class']} on_len={f1['on_mean_len']}"
          f" on_fz={f1['on_freeze_active_fraction']} off_len={f1['off_mean_len']}"
          f" | F3={f3['class']} H_off={f3['veto_off_entropy_bits']}"
          f" div={f3['attributable_divergence_fraction']}", flush=True)
    decisive = f0["class"] in ("HOLDS", "ENCODER_INSENSITIVE", "STREAM_CHOICE", "HSV_UNSUBTYPED")
    print(f"verdict: {'PASS' if decisive else 'FAIL'} seed={seed} F0={f0['class']}"
          f" F1={f1['class']} F3={f3['class']}", flush=True)
    return {"seed": seed, "aborted_at": None, "abort_reason": "", "reached_final": True,
            "p2_contact_rate": float(p2.contact_rate),
            "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
            "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
            "p1_survival_pass": bool(p1.survival_gate_passed),
            "f0": f0, "f1": f1, "f3": f3, "checkpoints": ckpts, "final": final}


def _majority(classes: List[str]) -> str:
    if not classes:
        return "NONE"
    for c in sorted(set(classes)):
        if classes.count(c) / float(len(classes)) >= MIN_FRACTION:
            return c
    return "MIXED"


def _strip_actions(res: Dict[str, Any]) -> None:
    """Keep per-episode action sequences only for the F3 pair (bounded size)."""
    for arm, blk in (res or {}).items():
        if isinstance(blk, dict) and arm in (ARM_FZ_ON, ARM_FZ_OFF):
            for e in blk.get("per_episode", []):
                e["actions"] = None


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        probe_own, final_own, walk_eps = 1, 2, 2
        train = 2 + 2 + 3 + 3 + 3 + 2
    else:
        probe_own, final_own, walk_eps = PROBE_OWN_EPS, FINAL_OWN_EPS, WALK_EPS
        train = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET + P1_BUDGET + P2_BUDGET
    total_eps = train + 2 * (2 * probe_own + walk_eps) + (4 * final_own + walk_eps)

    per_seed = [_run_seed(s, dry_run, total_eps) for s in seeds]
    n = len(per_seed)
    reached = [r for r in per_seed if r.get("reached_final")]
    reached_frac = len(reached) / float(n) if n else 0.0

    # ---- F0 -------------------------------------------------------------
    f0_ready = [r for r in reached if r["f0"]["ready"]]
    f0_ready_frac = len(f0_ready) / float(n) if n else 0.0
    hsv = ("ENCODER_INSENSITIVE", "STREAM_CHOICE", "HSV_UNSUBTYPED")
    f0_classes = [r["f0"]["class"] for r in f0_ready]
    if len(f0_ready) < MIN_READY_SEEDS:
        f0_verdict = "F0_NOT_READY"
    else:
        maj = _majority(f0_classes)
        hsv_frac = sum(1 for c in f0_classes if c in hsv) / float(len(f0_classes))
        if maj == "HOLDS":
            f0_verdict = "F0_REPRESENTATION_HOLDS"
        elif hsv_frac >= MIN_FRACTION:
            sub = _majority([c for c in f0_classes if c in hsv])
            f0_verdict = "F0_HSV_" + (sub if sub not in ("MIXED", "NONE") else "MIXED_SUBTYPE")
        else:
            f0_verdict = "F0_INDETERMINATE"
    f0_decisive = f0_verdict == "F0_REPRESENTATION_HOLDS" or f0_verdict.startswith("F0_HSV_")

    # ---- F1 -------------------------------------------------------------
    f1_classes = [r["f1"]["class"] for r in reached]
    f1_maj = _majority(f1_classes) if len(reached) >= MIN_READY_SEEDS else "NOT_READY"
    if f1_maj == "LOCK":
        f1_verdict = "F1_LOCK_CONFIRMED"
    elif f1_maj == "BOTH":
        f1_verdict = ("F1_BOTH_COLLAPSE_SATURATED_LATENT"
                      if f0_verdict == "F0_HSV_ENCODER_INSENSITIVE"
                      else "F1_BOTH_COLLAPSE_" + f0_verdict)
    elif f1_maj == "NO_LOCK":
        f1_verdict = "F1_NO_LOCK_REPRODUCED"
    elif f1_maj == "NOT_READY":
        f1_verdict = "F1_NOT_READY"
    else:
        f1_verdict = "F1_MIXED"
    f1_decisive = f1_verdict not in ("F1_MIXED", "F1_NOT_READY")
    freeze_engaged_any = any(r["f1"]["freeze_engaged"] for r in reached)

    # ---- F3 -------------------------------------------------------------
    f3_classes = [r["f3"]["class"] for r in reached]
    f3_maj = _majority(f3_classes) if reached else "NONE"
    f3_map = {"EFFECTIVE": "F3_H3_WEAKENED", "NO_ORGANISM_EFFECT": "F3_H3_LIVE",
              "DIVERSITY_COLLAPSED": "F3_STARVED_BY_FAMILY_C", "VETO_NOT_LIVE": "F3_VETO_NOT_LIVE",
              "ACTIONS_CONTACT_UNMEASURABLE": "F3_CONTACT_UNMEASURABLE",
              "VETO_NO_CANDIDATE_CONTRAST": "F3_NO_CANDIDATE_CONTRAST"}
    f3_verdict = f3_map.get(f3_maj, "F3_MIXED")
    unattr_any = any(int(r["f3"].get("n_unattributed_divergent") or 0) > 0 for r in reached)

    gate_input_independent = bool(reached and sum(
        1 for r in reached if (r["f0"]["resting_norm_over_theta"] or 0.0) >= 1.0)
        / float(len(reached)) >= MIN_FRACTION)
    outcome = "PASS" if (f0_decisive and f1_decisive) else "FAIL"
    if reached_frac < MIN_FRACTION or f0_verdict == "F0_NOT_READY":
        label = "substrate_not_ready_requeue"
    else:
        label = f0_verdict.lower()

    combination_rule = (
        "PASS iff F0 (LOAD-BEARING: the earliest edge, harm representation validity) "
        "reached a decisive verdict (F0_REPRESENTATION_HOLDS or F0_HSV_*) AND F1 (the "
        "chip's freeze on/off lock rule) reached a non-MIXED verdict. PASS means the "
        "edge was adjudicated, not that REE behaves. F3 is Q-111 leg-1 PRECONDITION "
        "DATA, never in the gate. Each leg's verdict is a majority (>= 2/3) of per-seed "
        "classes; F1's BOTH-collapse branch reads F0's verdict to separate saturation "
        "from other collapse.")

    for r in per_seed:
        for ck in r.get("checkpoints") or []:
            _strip_actions(ck.get("arms"))
        _strip_actions(r.get("final"))

    preconditions = [
        {"name": "curriculum_reached_final_checkpoint", "kind": "readiness",
         "description": "fraction of seeds that completed the 935a curriculum (no stage abort).",
         "control": "V3-EXQ-1107 reached the final eval on 3/3 of these seeds on this curriculum.",
         "measured": round(reached_frac, 4), "threshold": MIN_FRACTION, "direction": "lower",
         "met": bool(reached_frac >= MIN_FRACTION)},
        {"name": "f0_walk_samples_proximity", "kind": "readiness",
         "description": (f"fraction of seeds whose scripted walk gave >= {MIN_HELDOUT} held-out ticks "
                         f"with nearest-hazard distance SD >= {DIST_SD_MIN} (the R^2 target must vary "
                         "or R^2 is undefined)."),
         "control": "a uniform random walk over the 4 moves in a 12x12 grid with 4 hazards "
                    "(positive control: the target varies by construction unless lives are ~1 step).",
         "measured": round(f0_ready_frac, 4), "threshold": MIN_FRACTION, "direction": "lower",
         "met": bool(f0_ready_frac >= MIN_FRACTION)},
    ]
    worst = min((r["f0"]["r2_dist"] for r in f0_ready if r["f0"]["r2_dist"] is not None), default=None)
    criteria = [
        {"name": "F0_representation_validity_decisive", "load_bearing": True, "passed": f0_decisive,
         "verdict": f0_verdict, "measured": worst, "threshold": R2_TRACK,
         "threshold_fail": R2_FAIL, "comparator": ">=",
         "per_seed_statistic": "held-out ridge R^2 of z_harm_a on capped nearest-hazard distance "
                               "(ARM_WALK, final); measured = worst ready seed"},
        {"name": "F1_freeze_lock_on_vs_off", "load_bearing": True, "passed": f1_decisive,
         "verdict": f1_verdict, "measured": round(sum(1 for c in f1_classes if c == "LOCK") / float(max(1, len(f1_classes))), 4),
         "threshold": MIN_FRACTION, "comparator": ">=",
         "per_seed_statistic": (f"LOCK: FZ_ON collapses (len <= {COLLAPSE_LEN} or stationary >= "
                                f"{STATIONARY_FRAC}) with freeze_active >= {FREEZE_FRAC_LOCK}, FZ_OFF does not")},
        {"name": "F3_veto_precondition_data", "load_bearing": False, "passed": f3_verdict == "F3_H3_WEAKENED",
         "verdict": f3_verdict, "threshold_not_applicable": "precondition data for Q-111 leg 1; classes, not a bar",
         "per_seed_statistic": f"VETO_OFF entropy >= {H_MIN_BITS} bits; attributable divergence >= "
                               f"{DIVERGE_FRAC_MIN}; hazard contact reduced >= {CONTACT_REDUCTION}"},
    ]
    criteria_non_degenerate = {
        "F0_representation_validity_decisive": bool(len(f0_ready) >= MIN_READY_SEEDS),
        "F1_freeze_lock_on_vs_off": bool(freeze_engaged_any and len(reached) >= MIN_READY_SEEDS),
        "F3_veto_precondition_data": bool(reached and not unattr_any),
    }

    def _mean(vals):
        v = [float(x) for x in vals if x is not None]
        return round(float(np.mean(v)), 5) if v else None

    readout = flat_readout({
        "f0_decisive": f0_decisive, "f1_decisive": f1_decisive,
        "f3_h3_weakened": f3_verdict == "F3_H3_WEAKENED",
        "f0_ready_fraction": f0_ready_frac, "reached_final_fraction": reached_frac,
        "f0_r2_dist_mean": _mean([r["f0"]["r2_dist"] for r in reached]),
        "f0_r2_fid_mean": _mean([r["f0"]["r2_fid"] for r in reached]),
        "f0_r2_hf_mean": _mean([r["f0"]["r2_hf"] for r in reached]),
        "f0_resting_norm_over_theta_min": (min([r["f0"]["resting_norm_over_theta"] for r in reached
                                                if r["f0"]["resting_norm_over_theta"] is not None], default=None)),
        "f0_frac_ticks_distance_at_cap_mean": _mean([r["f0"]["frac_ticks_distance_at_cap"] for r in reached]),
        "f0_r2_in_mean": _mean([r["f0"]["r2_in"] for r in reached]),
        "f0_r2_norm_dist_mean": _mean([r["f0"]["r2_norm_dist"] for r in reached]),
        "f0_encoder_dead_unit_fraction_mean": _mean([r["f0"]["encoder_dead_unit_fraction"] for r in reached]),
        "f0_init_r2_dist_mean": _mean([_r2(r["checkpoints"][0]["arms"][ARM_WALK], "r2_dist")
                                       for r in per_seed if r.get("checkpoints")]),
        "f1_lock_fraction": sum(1 for c in f1_classes if c == "LOCK") / float(max(1, len(f1_classes))),
        "f1_on_freeze_active_fraction_mean": _mean([r["f1"]["on_freeze_active_fraction"] for r in reached]),
        "f1_on_mean_len": _mean([r["f1"]["on_mean_len"] for r in reached]),
        "f1_off_mean_len": _mean([r["f1"]["off_mean_len"] for r in reached]),
        "f1_off_shadow_would_freeze_mean": _mean([r["f1"]["off_shadow_would_freeze_fraction"] for r in reached]),
        "f3_veto_off_entropy_mean": _mean([r["f3"]["veto_off_entropy_bits"] for r in reached]),
        "f3_attributable_divergence_mean": _mean([r["f3"]["attributable_divergence_fraction"] for r in reached]),
        "f3_veto_fire_rate_mean": _mean([r["f3"]["veto_fire_rate"] for r in reached]),
        "n_seeds": n, "r2_track": R2_TRACK, "r2_fail": R2_FAIL,
    })
    print(f"[{EXPERIMENT_TYPE}] F0={f0_verdict} F1={f1_verdict} F3={f3_verdict} -> outcome={outcome}", flush=True)
    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "readout": readout,
        "interpretation": {
            "label": label,
            "f0_verdict": f0_verdict, "f1_verdict": f1_verdict, "f3_verdict": f3_verdict,
            "f0_classes": f0_classes, "f1_classes": f1_classes, "f3_classes": f3_classes,
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "f3_scope": "Q-111 leg-1 PRECONDITION DATA, not a Q-111 test",
            "gate_input_independent": gate_input_independent,
            "gate_input_independent_rule": ("min over the final scripted walk of ||z_harm_a|| >= "
                                            "theta_freeze on >= 2/3 reached seeds: the freeze EXIT "
                                            "(z < theta*tone, tone 1.0) is unreachable at every sampled "
                                            "input, so the lock is an operating-point SCALE fact (ARC-155 "
                                            "H1 site) whatever F0's R^2 says; a HOLDS verdict then does "
                                            "NOT by itself make ARC-155/ARC-156 PAG designs runnable."),
            "dv_symmetry": {
                "F0_walk_vs_own": "NOT INVARIANT: policy changes the sampled states; R^2 needs target variance.",
                "F1_freeze_on_vs_off": "NOT INVARIANT: removes the STAY override of the executed action.",
                "F3_veto_on_vs_off": "NOT INVARIANT: per-candidate safety axis changes the eligible set.",
            },
            "premise_corrected": ("harm_obs_a in this harness is 7-d BODY DAMAGE (limb_damage_enabled=True, "
                                  "SD-022), not the proximity EMA the CeA re-probe measured; 1107 eval envs "
                                  "were seed-pinned to one spawn, this run uses a fresh paired seed per episode."),
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        "arms_probe": PROBE_ARMS, "arms_final": FINAL_ARMS, "checkpoints": CHECKPOINTS,
        "probe_own_eps": PROBE_OWN_EPS, "final_own_eps": FINAL_OWN_EPS, "walk_eps_nominal": WALK_EPS,
        "walk_min_ticks": WALK_MIN_TICKS, "walk_max_eps": WALK_MAX_EPS,
        "walk_steps": WALK_STEPS, "env_seed_base": ENV_SEED_BASE, "eval_env_stream": EVAL_ENV_STREAM,
        "seeds": SEEDS, "train_steps": TRAIN_STEPS,
        "thresholds": {"r2_track": R2_TRACK, "r2_fail": R2_FAIL, "dist_cap": DIST_CAP,
                       "dist_sd_min": DIST_SD_MIN, "min_heldout": MIN_HELDOUT,
                       "damage_sd_min": DAMAGE_SD_MIN, "ridge_lambda": RIDGE_LAMBDA,
                       "collapse_len": COLLAPSE_LEN, "stationary_frac": STATIONARY_FRAC,
                       "freeze_frac_lock": FREEZE_FRAC_LOCK, "h_min_bits": H_MIN_BITS,
                       "diverge_frac_min": DIVERGE_FRAC_MIN, "contact_reduction": CONTACT_REDUCTION,
                       "min_off_hazard_contacts": MIN_OFF_HAZARD_CONTACTS,
                       "harm_range_precondition": HARM_RANGE_PRECONDITION, "min_fraction": MIN_FRACTION},
        "mech449_producer": {"z_threshold": GNG_SAFETY_Z_THRESHOLD, "ema_decay": GNG_SAFETY_EMA_DECAY,
                             "sd_floor": GNG_SAFETY_SD_FLOOR, "warmup": GNG_SAFETY_WARMUP_SAMPLES,
                             "floor": GNG_SAFETY_FLOOR},
        "scaffold_curriculum": {"stage0": STAGE0_BUDGET, "stage0b": STAGE0B_BUDGET, "p0": P0_BUDGET,
                                "hazard": HAZARD_STAGE_BUDGET, "p1": P1_BUDGET, "p2": P2_BUDGET,
                                "config_basis": "V3-EXQ-1107 / 935a _make_config (freeze ON)"},
    }
    manifest = {
        "run_id": run_id, "experiment_type": EXPERIMENT_TYPE, "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS, "bears_on": BEARS_ON, "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1", "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "source_record": SOURCE_RECORD, "red_team": RED_TEAM,
        "predecessor": "V3-EXQ-1107 (harness + seeds; not superseded)", "supersedes": None,
        "chip_ref": "chip-20260926-pag-freeze-lock-confirmer",
        "condition": CONDITION_LABEL,
        "pre_registered_thresholds": full_config["thresholds"],
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "config": full_config, "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=dry_run, config=full_config,
        seeds=SEEDS[:1] if dry_run else SEEDS, script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(), started_at=t0,
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    pin_recording_substrate(script_path=Path(__file__))  # substrate identity at process START
    _res = main(dry_run=args.dry_run)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        dry_run=bool(args.dry_run),
    )
