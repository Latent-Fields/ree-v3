"""
V3-EXQ-1107 (SD-032a coordinator switching, trained agent): do within-life
operating-mode REVERSALS survive training, and do they still require an
independent external_task input (external_task_drive)? DIAGNOSTIC.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
RED-TEAM (fable, Step 4.5): BLOCKING as first written (F1: C2 was an arithmetic
identity of the frozen-drive eval harness) -> fixed: C2 demoted to a structural
control with a reachability counter, native drive_level fed live, PASS rests on
C1; F2 relative-spread floor on C3; F4 clone load report; F5 >= 2 testable
seeds; F6 cap-2 labels. F3 CLEAR. Round 2 (fable, one re-spawn after the
BLOCKING fix changed C2/the PASS gate): CLEAR; minor notes fixed (PINNED ->
non_contributory, counterfactual logit margin + beta-elevated fraction logged).

WHY THIS RUN EXISTS
-------------------
REE_assembly evidence/planning/mode_switch_cea_mechanism_trace_20260925.md
(e2bbd2a98e; GFLAG-0554) traced the SalienceCoordinator (ree_core/cingulate/
salience_coordinator.py) on UNTRAINED agents and found:
  * dACC ON, no external_task_drive (its arm A1, cap 2.0): one switch per life
    (external_task -> internal_planning after reset) and then locked. Within-life
    reversals 0/0/0 on seeds 11/12/13. Root: input TOPOLOGY -- dacc_pe is the only
    salience source AND the dominant internal_planning affinity source, so the
    ticks that could fire the trigger are the ticks that pull the argmax away from
    external_task.
  * + use_external_task_drive (its arm A4): within-life reversals 2/16/25,
    switches back INTO external_task 1/16/13.
  * NOT LIKE-FOR-LIKE with 935a: 935a's eval ran with drive_level frozen at 0
    (see C2 below); this run feeds the native drive live, which also reaches
    the MECH-295 liking-bridge cue side (agent.py:9183) identically in both
    arms. 935a's switch counts are a prior, not an expectation.
  * Its single named next action: "one trained-agent cloud run on 935a's
    curriculum, 3 seeds, A1 vs A4" -- does reversal survive training, and is the
    shared-upstream guess right (dacc_pe falls back to ||z_harm_a|| when
    use_e2_harm_a is off, dacc.py:213-218; the curriculum's harm_accum aux
    (scaffolded_sd054_onboarding.py:1436) trains the affective encoder and was
    measured to inflate |z_harm_a| ~100x offline).

PREMISE CORRECTED AT AUTHORING (re-measured against V3-EXQ-935a's source)
------------------------------------------------------------------------
The brief described the arms as "dACC+cap (935a's config) vs
dACC+cap+use_external_task_drive". 935a's config ALREADY sets
use_external_task_drive=True (affinity 3.0, salience 2.0,
v3_exq_935a_...py `_make_config`). So 935a's config is the A4-like ON arm, and
the NEW arm is the lesion: identical trained weights with the drive OFF. This
run therefore trains ONE 935a-curriculum agent per seed (drive ON during
training, as 935a) and evaluates two frozen-policy clones of it:
  ARM_DRIVE_OFF  -- config deep-copy with use_external_task_drive=False (the
                   modetrace A1 analogue: dACC + cap, no independent
                   external_task input). The drive signal is neither computed
                   nor registered (agent.py:2800, 8652).
  ARM_DRIVE_ON   -- 935a's config unchanged (modetrace A4 analogue, goal gate ON
                   as in 935a: engagement requires an active z_goal, which the
                   trained agent has).
Eval-time lesion (not separate training) is the paired design: both arms share
bit-identical weights, the same eval env seed and the same RNG state at cell
entry, so any difference is the drive's input to the coordinator. Stated
limitation: this asks whether the TRAINED agent's register needs the drive, not
whether an agent TRAINED without it would develop one (the mode register has no
known behavioural consumer at this config -- MECH-157/MECH-259 R4 audit -- so
training is not expected to depend on it; the behavioural readouts below record
whether that holds).

GOV-REUSE-1 (Step 2.4). Decisive readouts: within-life reversals and switches
INTO external_task, per arm, on a trained agent; dacc_pe and ||z_harm_a|| scale
across training. V3-EXQ-935a (run v3_exq_935a_..._20260916T095809Z_v3) is the
compatible trained-agent record: it has the ON arm only, at swept caps, and
records n_switches per cell (switches > episodes on seeds 47/49 at caps ~1.6-2.2;
exactly 15 switches / 15 episodes -- one-way lock -- on seeds 48/50/51 at caps
above ~1.05) but NO drive-OFF arm, no into-external counts, no dacc_pe and no
z_harm_a. Partially recoverable -> this run is the minimal targeted run for the
missing pieces: the OFF lesion, the into-external counts and the scale
trajectories.

DESIGN
------
Seeds 47, 48, 49 (the first three of 935a's fresh seeds, taken in order, not
selected by outcome). Per seed:
  1. Train with 935a's scaffolded_sd054_onboarding curriculum, budgets and
     substrate config byte-identical to 935a (stage0, stage0b, p0, hazard, p1,
     p2). Env seeds PINNED (ENV_SEED_BASE) so the two arms' eval layouts are
     identical (935a ran unpinned; this run is not bit-comparable to it).
  2. CHECKPOINT PROBES before training (init) and after each stage: both arms on
     clones, PROBE_EPISODES each, eval cap EVAL_CAP. Training RNG state (python,
     numpy, torch, harness) is saved and restored around every probe so probing
     does not perturb training. Records per checkpoint: dacc_pe distribution,
     ||z_harm_a||_2, mean|z_harm_a| (CeA low_freq statistic), their per-tick
     correlation, the affective_harm_encoder parameter hash and norm, and each
     arm's switch counts.
  3. FINAL EVAL: both arms, MODE_EVAL_EPISODES each, eval cap EVAL_CAP = 2.0
     (the training cap AFFINITY_INPUT_CAP_TRAIN, and the modetrace A1/A4 cap),
     symmetric rails (as 935a), fresh env with the same seed for both arms,
     same RNG reset for both arms.

DEPENDENT VARIABLES (per arm, per seed, final eval)
  n_out   = register switches OUT of external_task
  n_in    = register switches INTO external_task (never counts the per-episode
            reset(), which forces external_task outside tick())
  reversals_beyond_first = sum over lives of max(0, switches_in_life - 1)
  Per-seed arm class: REVERSING (n_in >= INTO_EXT_MIN), ONE_WAY (n_in <= 1 AND
  n_out >= LEAVE_FRAC * n_episodes -- left external_task and did not come back),
  PINNED_EXTERNAL (n_out < LEAVE_FRAC * n_episodes AND n_in <= 1 -- never left),
  INTERMEDIATE (anything else).

CRITERIA (pre-registered; thresholds are constants in this file)
  C1 REVERSALS_SURVIVE_TRAINING_WITH_DRIVE [load-bearing, the ONLY one]:
     ARM_DRIVE_ON is REVERSING on >= MIN_FRACTION (2/3) of testable seeds, with
     >= MIN_TESTABLE_SEEDS (2) testable seeds.
  C2 DRIVE_OFF_STAYS_ONE_WAY [STRUCTURAL CONTROL, NOT load-bearing]:
     ARM_DRIVE_OFF is ONE_WAY on >= MIN_FRACTION of testable seeds. Red-team
     (fable) F1, verified against source: with drive_level frozen, the OFF arm's
     external_task logit is exactly external_task_bias = 1.0, while any tick that
     clears the trigger (salience = dacc_pe + 0.5 x foraging > 1.0) has an
     internal_planning logit >= 1.0 -- a return is arithmetically impossible for
     ANY weights. So C2 cannot, by itself, say anything about training. Two
     repairs: (i) this driver feeds the NATIVE drive_level (SD-012
     clip(1 - energy), the default external_task affinity input) every eval
     step -- its only writer is update_z_goal (agent.py:12870), which a
     frozen-goal eval never calls, so 935a's harness also ran with it frozen at 0;
     z_goal itself stays frozen as in 935a, only the scalar is fed; (ii) the run
     counts, per arm, the ticks on which argmax(operating_mode) returned to
     external_task while the register was away from it
     (n_ticks_ext_argmax_while_away). Because salience >= dacc_pe >= 2.0 clears
     the 1.0 threshold on every trained tick, such a tick always fires, so
     off_return_reachable is equivalent to the OFF arm having returned at least
     once (red-team r2); the COUNTERFACTUAL distance-to-return is recorded
     separately as ext_logit_margin_while_away (exact log-odds of external_task
     vs the best other mode on away ticks; > 0 would mean a return). C2 is
     non-degenerate only when off_return_reachable holds on some testable seed.
     Expected: at cap 2.0 with a trained dacc_pe >= 2.0, the OFF ext logit
     (<= 1 + drive_level <= 2) still cannot beat the capped dacc_pe side, so C2
     is likely structural again -- the run records that as a measurement, not a
     finding about training.
  PASS iff C1 (`combination_rule` in the manifest).
  C3 SHARED_UPSTREAM [NOT load-bearing, separate hypothesis H-SHARED, recorded
     with measured/threshold]: at the final eval, on >= MIN_FRACTION of testable
     seeds, BOTH (a) Pearson corr over coordinator ticks between dacc_pe and
     ||z_harm_a||_2 >= CORR_MIN in ARM_DRIVE_OFF, AND (b) dacc_pe mean grew
     >= GROWTH_MIN x from the init checkpoint to the final eval AND
     ||z_harm_a||_2 mean grew >= GROWTH_MIN x over the same span.
     Stated openly (red-team F2): with use_e2_harm_a off and dacc_pe_cap None,
     dacc_pe = ||z_harm_a|| x (1 + min(precision/scale, 3)) x saturation
     (dacc.py:213-225), so (a) is corr(k x, x) ~ 1 by construction and (b)'s two
     ratios are one measurement counted twice. C3 therefore reads as "is the
     fallback branch live and did training inflate the shared magnitude". A
     per-seed relative-spread floor ((p90 - p10)/mean > REL_SPREAD_MIN on BOTH
     series) gates measurability, so a resting offset with numerical wobble
     routes shared_upstream_not_measurable, never supported. harm_obs_a_dim = 7
     here (the modetrace used 50); its ~100x offline inflation is not assumed.

ROUTING (declares both directions; diagnostic self-route is a hypothesis)
  readiness unmet, or < 2 testable seeds -> substrate_not_ready_requeue
  C1 PASS, labelled by the majority OFF class:
    OFF REVERSING                  -> trained_agent_reverses_with_or_without_external_task_drive_cap2
                                      (native drive_level is enough; the drive is
                                      not necessary)
    OFF ONE_WAY, return reachable  -> external_task_drive_needed_for_reversal_trained_cap2
    OFF ONE_WAY, not reachable     -> external_task_drive_sufficient_for_reversal_trained_cap2_off_arm_structurally_one_way
    otherwise                      -> external_task_drive_sufficient_for_reversal_trained_cap2
  C1 FAIL, labelled by the majority ON class:
    ON ONE_WAY                     -> trained_agent_locks_even_with_drive_cap2
                                      (cap-margin interplay: at the trained scale the
                                      capped dacc_pe side outweighs the drive; the
                                      untrained D2 result does not carry over)
    ON PINNED_EXTERNAL             -> drive_pins_external_task_no_reversal_cap2
                                      (engagement > ~0.33 throughout at affinity 3.0,
                                      cap 2.0; the dry-run smoke reached this shape)
    anything else                  -> drive_effect_not_resolved (classes recorded)
  H-SHARED is routed separately: shared_upstream_supported / _refuted /
  _not_measurable.
  SD-032a direction: supports on PASS; weakens if ready and the ON majority is
  ONE_WAY (competed and locked); non_contributory otherwise (incl. PINNED, an
  affinity-margin artefact at cap 2.0). Diagnostic -- excluded from confidence scoring.

BEHAVIOURAL READOUTS (red-team F3): the register has no live behavioural
consumer at this config, so the two arms are expected to be behaviourally
bit-paired (identical episode lengths / harm) BY CONSTRUCTION -- the smoke
showed exactly that. Identical behaviour is therefore not a null finding about
the mode register's reach; it confirms the arms differ only in coordinator
arithmetic on an identical input stream.

DV-SYMMETRY (Step 3.5). ARM_DRIVE_OFF vs ARM_DRIVE_ON: the DV is the sequence of
argmax(operating_mode) changes gated by salience_aggregate > threshold. The
manipulation removes one signed per-mode logit term (+3.0 x drive on
external_task only) and one salience term; that changes logit DIFFERENCES
between modes, not a common offset, and is not a monotone transform of the
logits -- the DV is not invariant under it (modetrace D2: 0/0/0 -> 2/16/25).
C3's DVs (Pearson r; mean ratios) are not manipulated by arm.

PRECONDITIONS (readiness; seed-scoped, aggregated as a fraction, never AND-ed)
  P1 foraging_contact_guard: 935a's guard (P2 contact_rate > 0 AND
     z_goal_norm_at_contact_peak > 0.4) on >= 2/3 seeds.
  P2 drive_engages_on_arm: mean external_task_drive input in ARM_DRIVE_ON final
     eval > DRIVE_ENGAGE_FLOOR on >= 2/3 guard seeds (C1's manipulation must be
     live; a goal-gated drive stuck at 0 starves C1, it does not falsify it).
  P3 dacc_salience_live: mean dacc_pe in ARM_DRIVE_OFF final eval >
     DACC_LIVE_FLOOR on >= 2/3 guard seeds (without the dACC alarm neither arm
     can switch at all -- the dACC-off wiring failure the modetrace named).

INCIDENTAL (consumer half, out of scope; owned by
chip-20260917-sd032a-operating-mode-no-consumer): per-arm mean episode length and
summed env harm are recorded so a reader can see whether the mode sequence
reached behaviour.

RE-DERIVE BRAKE (Step 2.5b): SD-032a's brake was RELEASED for 935a
(mode-governance-engagement LANDED); this is a different manipulation (drive
lesion + scale trajectories), not a lettered re-test. MECH-157 is tagged only
because this run measures its non-degeneracy "required event" (a within-life
coordinator-driven transition); MECH-157 is non_contributory on every branch.

claim_ids: SD-032a, MECH-157.  experiment_purpose: diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

EXPERIMENT_TYPE = "v3_exq_1107_sd032a_trained_mode_reversal_drive"
QUEUE_ID = "V3-EXQ-1107"
CLAIM_IDS: List[str] = ["SD-032a", "MECH-157"]
EXPERIMENT_PURPOSE = "diagnostic"
SOURCE_RECORD = (
    "REE_assembly evidence/planning/mode_switch_cea_mechanism_trace_20260925.md "
    "(e2bbd2a98e), GFLAG-0554"
)

SEEDS = [47, 48, 49]
CONDITION_LABEL = "TRAINED_935A_CURRICULUM_DRIVE_OFF_VS_ON"
ENV_SEED_BASE = 1107  # pinned: both arms see identical eval layouts

_ZG = ZGoalStreamAccumulator()

ANCHOR_REACHABILITY_EXEMPT = (
    "The three readiness preconditions are ordinary upstream gates (curriculum "
    "foraging guard; the ON arm's drive input is live; the dACC alarm is live), "
    "each a plain '> floor on >= 2/3 seeds' fraction over a directly-measured "
    "quantity -- not a hand-written degeneracy-reproduction predicate narrower "
    "than the state it anchors to. Reachability: the guard cleared 5/5 on this "
    "curriculum in V3-EXQ-935a; drive/dacc floors (0.05) sit far below the values "
    "measured in 935a cells (external_task margins 0.32-0.81) and the untrained "
    "modetrace (dacc_pe 0.19-1.27)."
)

MODE_NAMES = ["external_task", "internal_planning", "internal_replay", "offline_consolidation"]
EXT = "external_task"

ARM_OFF = "ARM_DRIVE_OFF"
ARM_ON = "ARM_DRIVE_ON"
ARMS = [ARM_OFF, ARM_ON]

# --- eval / criterion constants (pre-registered) --------------------------
AFFINITY_INPUT_CAP_TRAIN = 2.0     # 935a training cap
EVAL_CAP = 2.0                     # = training cap = modetrace A1/A4 cap
MODE_EVAL_EPISODES = 15            # as 935a
PROBE_EPISODES = 2                 # per arm per checkpoint
INTO_EXT_MIN = 3                   # REVERSING: >= 3 returns to external_task in 15 lives
LEAVE_FRAC = 0.5                   # ONE_WAY / PINNED split: left external_task in >= half the lives
MIN_FRACTION = 2.0 / 3.0
DRIVE_ENGAGE_FLOOR = 0.05
DACC_LIVE_FLOOR = 0.05
CORR_MIN = 0.8
GROWTH_MIN = 3.0
CEA_LOWFREQ_THRESHOLD = 0.5        # cea.py:337 default, for the tonic-fire readout only
REL_SPREAD_MIN = 1e-3
# Audit-only switch (never set on a queued run): skip checkpoint probes to show
# they do not perturb training (compare the p2 encoder hash + final eval).
import os as _os  # noqa: E402
NO_PROBES_AUDIT = _os.environ.get("REE_EXQ1107_NO_PROBES_AUDIT", "") == "1"              # C3 measurable only if (p90-p10)/mean exceeds this for both
MIN_TESTABLE_SEEDS = 2             # a verdict needs >= 2 testable seeds (red-team F5)

# --- curriculum + substrate: byte-identical to V3-EXQ-935a ----------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

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

CHECKPOINTS = ["init", "stage0", "stage0b", "p0", "hazard", "p1", "p2"]


# --------------------------------------------------------------------------
# Harness (scaffold cfg, substrate cfg, env build, clone, rails): REUSED
# VERBATIM from V3-EXQ-935a except _clone_for_arm's drive toggle.
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
    """V3-EXQ-935a's substrate config, unchanged (drive ON during training)."""
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
    """P2-config foraging env WITH the GAP-3 dual_cue primitive (935a verbatim)."""
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


def _clone_for_arm(trained_agent: REEAgent, device: torch.device, drive_on: bool) -> REEAgent:
    """Clone the SAME trained weights into a fresh agent (935a's clone incl. the
    goal_state fix). drive_on=False sets use_external_task_drive=False on the
    deep-copied config BEFORE construction, so the coordinator never registers the
    external_task_drive affinity/salience weights (agent.py:2800) and select_action
    never computes the signal (agent.py:8652). The drive has no parameters, so the
    state_dict is identical across arms."""
    cfg = copy.deepcopy(trained_agent.config)
    cfg.use_external_task_drive = bool(drive_on)
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
        agent._clone_load_report = {"strict": True, "missing": [], "unexpected": []}
    except RuntimeError:
        res = agent.load_state_dict(state, strict=False)
        agent._clone_load_report = {"strict": False,
                                    "missing": list(getattr(res, "missing_keys", []))[:50],
                                    "unexpected": list(getattr(res, "unexpected_keys", []))[:50]}
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    if trained_agent.goal_state is not None and agent.goal_state is not None:
        agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())
    registered = "external_task_drive" in agent.salience.config.affinity_weights
    if registered != bool(drive_on):
        raise RuntimeError(
            f"arm wiring check failed: drive_on={drive_on} but "
            f"external_task_drive registered={registered}")
    return agent


def _apply_symmetric(coord) -> None:
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}


# --------------------------------------------------------------------------
# RNG save/restore so checkpoint probes never perturb the training trajectory
# --------------------------------------------------------------------------
def _harness_modules() -> List[Any]:
    mods = []
    for name in ("experiments._harness", "_harness"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "_action_random") and m not in mods:
            mods.append(m)
    return mods


def _save_rng() -> Dict[str, Any]:
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "harness": [(m, m._action_random.getstate()) for m in _harness_modules()],
    }


def _restore_rng(st: Dict[str, Any]) -> None:
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    for m, s in st["harness"]:
        m._action_random.setstate(s)


# --------------------------------------------------------------------------
# Instrumented eval of ONE arm
# --------------------------------------------------------------------------
def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
    v = [x for x in vals if math.isfinite(x)]
    if not v:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None, "max": None}
    arr = np.array(v, dtype=float)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 5),
        "p10": round(float(np.quantile(arr, 0.10)), 5),
        "p50": round(float(np.quantile(arr, 0.50)), 5),
        "p90": round(float(np.quantile(arr, 0.90)), 5),
        "max": round(float(arr.max()), 5),
    }


def _classify(n_in: int, n_out: int, n_eps: int) -> str:
    if n_in >= INTO_EXT_MIN:
        return "REVERSING"
    left_enough = n_out >= LEAVE_FRAC * n_eps
    if n_in <= 1 and left_enough:
        return "ONE_WAY"
    if n_in <= 1 and not left_enough:
        return "PINNED_EXTERNAL"
    return "INTERMEDIATE"


def _eval_arm(agent: REEAgent, env: CausalGridWorldV2, cap: float, arm_label: str,
              scaffold_cfg: ScaffoldedSD054OnboardingConfig, device: torch.device,
              n_eps: int, steps_per_ep: int) -> Dict[str, Any]:
    """Frozen-policy eval of one arm. Per env step: committed register mode (as
    935a). Per coordinator tick (wrapped tick): trigger, salience vs threshold,
    raw input signals, and the CURRENT tick's z_harm_a (sense() runs before
    select_action(), which runs the coordinator tick)."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience
    coord.config.affinity_input_cap = float(cap)
    _apply_symmetric(coord)
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    tick_rows: List[tuple] = []
    orig_tick = coord.tick

    def _wrapped(*a, **k):
        prev_cur = coord.current_mode
        out = orig_tick(*a, **k)
        om = dict(coord.operating_mode)
        am = max(om.items(), key=lambda kv: kv[1])[0] if om else prev_cur
        ext_argmax_away = bool(prev_cur != EXT and am == EXT)
        # exact logit margin ext - best other (softmax temperature 1.0): the
        # counterfactual distance-to-return on ticks spent away from external_task
        if prev_cur != EXT and om.get(EXT, 0.0) > 0.0:
            _oth = max(v for m_, v in om.items() if m_ != EXT)
            ext_margin_away = (float(math.log(om[EXT]) - math.log(_oth))
                               if _oth > 0.0 else float("nan"))
        else:
            ext_margin_away = float("nan")
        beta_el = bool(getattr(getattr(agent, "beta_gate", None), "is_elevated", False))
        sig = coord._input_signals
        lat = agent._current_latent
        zha = getattr(lat, "z_harm_a", None) if lat is not None else None
        if zha is not None:
            z2 = float(zha.detach().norm().item())
            z1 = float(zha.detach().abs().mean().item())
        else:
            z2 = float("nan")
            z1 = float("nan")
        tick_rows.append((
            bool(out.get("mode_switch_trigger", False)),
            float(out.get("salience_aggregate", 0.0)),
            float(out.get("enter_threshold", 0.0)),
            float(sig.get("dacc_pe", 0.0)),
            float(sig.get("external_task_drive", 0.0)),
            float(sig.get("drive_level", 0.0)),
            float(sig.get("dacc_foraging", 0.0)),
            z2, z1, ext_argmax_away, ext_margin_away, beta_el,
        ))
        return out

    coord.tick = _wrapped

    per_life: List[Dict[str, int]] = []
    mode_steps = {m: 0 for m in MODE_NAMES}
    other_steps = 0
    ep_lengths: List[int] = []
    harm_sums: List[float] = []
    try:
        with torch.no_grad():
            for _ep in range(n_eps):
                _, obs_dict = env.reset()
                agent.reset()
                prev_mode = coord.current_mode
                sw = n_in = n_out = 0
                steps = 0
                hsum = 0.0
                for _ in range(steps_per_ep):
                    obs_body = obs_dict["body_state"].to(device)
                    obs_world = obs_dict["world_state"].to(device)
                    latent = _sense_with_optional_harm(
                        agent, obs_body, obs_world, obs_dict, device, feed_harm)
                    # NATIVE drive_level (the default-ON external_task affinity input,
                    # SD-012 clip(1 - energy)): its only writer is update_z_goal
                    # (agent.py:12870), which a frozen-goal eval never calls, so
                    # without this it is frozen at 0 and a return to external_task
                    # is arithmetically impossible in the OFF arm (red-team F1). Only
                    # the scalar is fed; z_goal itself stays frozen, as in 935a.
                    if agent.goal_state is not None:
                        _b, _drv = _benefit_and_drive(obs_body)
                        agent.goal_state._last_drive_level = float(_drv)
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, world_dim, device=device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks)
                    action_idx = int(action.argmax(dim=-1).item())
                    cur = coord.current_mode
                    if cur in mode_steps:
                        mode_steps[cur] += 1
                    else:
                        other_steps += 1
                    if cur != prev_mode:
                        sw += 1
                        if cur == EXT:
                            n_in += 1
                        if prev_mode == EXT:
                            n_out += 1
                        prev_mode = cur
                    steps += 1
                    _, harm, done, _info, obs_dict = env.step(action_idx)
                    try:
                        hsum += float(harm)
                    except (TypeError, ValueError):
                        pass
                    if done:
                        break
                per_life.append({"switches": sw, "into_ext": n_in, "out_ext": n_out, "steps": steps})
                ep_lengths.append(steps)
                harm_sums.append(hsum)
    finally:
        coord.tick = orig_tick

    total_steps = sum(ep_lengths)
    n_in_tot = sum(l["into_ext"] for l in per_life)
    n_out_tot = sum(l["out_ext"] for l in per_life)
    rev_beyond_first = sum(max(0, l["switches"] - 1) for l in per_life)
    dacc = [r[3] for r in tick_rows]
    drv = [r[4] for r in tick_rows]
    z2 = [r[7] for r in tick_rows]
    z1 = [r[8] for r in tick_rows]
    n_ticks = len(tick_rows)
    return {
        "arm": arm_label,
        "cap": float(cap),
        "n_episodes": n_eps,
        "total_steps": total_steps,
        "coord_n_ticks": n_ticks,
        "n_switches": sum(l["switches"] for l in per_life),
        "n_into_external": n_in_tot,
        "n_out_of_external": n_out_tot,
        "reversals_beyond_first": rev_beyond_first,
        "reversals_per_life": round(rev_beyond_first / float(n_eps), 4) if n_eps else 0.0,
        "arm_class": _classify(n_in_tot, n_out_tot, n_eps),
        "per_life": per_life,
        "fraction_in_external_task": round(mode_steps[EXT] / total_steps, 4) if total_steps else 0.0,
        "mode_step_counts": mode_steps,
        "other_mode_steps": other_steps,
        "n_trigger_ticks": sum(1 for r in tick_rows if r[0]),
        "n_ticks_ext_argmax_while_away": sum(1 for r in tick_rows if r[9]),
        "ext_logit_margin_while_away": _stats([r[10] for r in tick_rows]),
        "frac_ticks_beta_elevated": (round(sum(1 for r in tick_rows if r[11]) / n_ticks, 4)
                                     if n_ticks else None),
        "clone_load_report": getattr(agent, "_clone_load_report", None),
        "n_ticks_salience_above_threshold": sum(1 for r in tick_rows if r[1] > r[2]),
        "salience_aggregate": _stats([r[1] for r in tick_rows]),
        "dacc_pe": _stats(dacc),
        "frac_ticks_dacc_pe_above_cap": round(sum(1 for x in dacc if x > cap) / n_ticks, 4) if n_ticks else None,
        "external_task_drive": _stats(drv),
        "drive_level": _stats([r[5] for r in tick_rows]),
        "dacc_foraging": _stats([r[6] for r in tick_rows]),
        "z_harm_a_l2": _stats(z2),
        "z_harm_a_meanabs": _stats(z1),
        "frac_ticks_meanabs_above_cea_threshold": (
            round(sum(1 for x in z1 if math.isfinite(x) and x > CEA_LOWFREQ_THRESHOLD) / n_ticks, 4)
            if n_ticks else None),
        "corr_dacc_pe_vs_z_harm_a_l2": _pearson(dacc, z2),
        "mean_episode_length": round(float(np.mean(ep_lengths)), 3) if ep_lengths else 0.0,
        "mean_episode_harm_sum": round(float(np.mean(harm_sums)), 5) if harm_sums else 0.0,
    }


def _encoder_fingerprint(agent: REEAgent) -> Dict[str, Any]:
    enc = getattr(agent.latent_stack, "affective_harm_encoder", None)
    if enc is None:
        return {"present": False, "hash": None, "param_l2": None}
    h = hashlib.sha256()
    sq = 0.0
    for k, v in sorted(enc.state_dict().items()):
        t = v.detach().cpu().contiguous()
        h.update(k.encode("ascii", "ignore"))
        h.update(t.numpy().tobytes())
        sq += float((t.float() ** 2).sum().item())
    return {"present": True, "hash": h.hexdigest()[:16], "param_l2": round(math.sqrt(sq), 5)}


def _run_pair(agent: REEAgent, scaffold_cfg, device, seed: int, n_eps: int,
              steps_per_ep: int, env_stream: int, env_idx: int, rng_seed: int) -> Dict[str, Any]:
    """Evaluate both arms on clones of `agent`: same env seed, same RNG at entry."""
    out = {}
    env_seed = _derive_env_seed(ENV_SEED_BASE + seed, stream=env_stream, idx=env_idx)
    for arm in ARMS:
        # Seed BEFORE construction (weights are then overwritten by the trained
        # state_dict, but construction draws RNG) and again after the env build so
        # both arms enter the eval with the identical RNG state.
        reset_all_rng(rng_seed)
        clone = _clone_for_arm(agent, device, drive_on=(arm == ARM_ON))
        env = _build_dual_cue_env(scaffold_cfg, seed=env_seed)
        env.reset()
        reset_all_rng(rng_seed)
        out[arm] = _eval_arm(clone, env, EVAL_CAP, arm, scaffold_cfg, device,
                             n_eps, steps_per_ep)
        _ZG.observe(clone)
        del clone
    return out


def _probe(agent, scaffold_cfg, device, seed, label, idx, steps_per_ep, n_eps) -> Dict[str, Any]:
    st = _save_rng()
    try:
        pair = _run_pair(agent, scaffold_cfg, device, seed, n_eps, steps_per_ep,
                         env_stream=3, env_idx=idx, rng_seed=seed * 100 + idx)
    finally:
        _restore_rng(st)
    enc = _encoder_fingerprint(agent)
    off = pair[ARM_OFF]
    on = pair[ARM_ON]
    print(f"  [probe] seed={seed} ckpt={label}"
          f" dacc_pe_mean={off['dacc_pe']['mean']} zha_l2_mean={off['z_harm_a_l2']['mean']}"
          f" corr={off['corr_dacc_pe_vs_z_harm_a_l2']}"
          f" OFF in/out={off['n_into_external']}/{off['n_out_of_external']}"
          f" ON in/out={on['n_into_external']}/{on['n_out_of_external']}"
          f" drive_mean={on['external_task_drive']['mean']} enc={enc['hash']}", flush=True)
    return {"checkpoint": label, "encoder": enc, "arms": pair}


def _aborted(seed: int, stage: str, reason: str, checkpoints) -> Dict[str, Any]:
    return {"seed": seed, "aborted_at": stage, "abort_reason": reason,
            "guard_pass": False, "checkpoints": checkpoints, "final": None,
            "testable": False}


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_env_base = ENV_SEED_BASE + seed
    scaffold_cfg = _make_scaffold_cfg(dry_run, env_seed=seed_env_base)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else MODE_EVAL_EPISODES
    probe_eps = 1 if dry_run else PROBE_EPISODES

    probe_env = _build_dual_cue_env(
        scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=0))
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0
    ckpts: List[Dict[str, Any]] = []

    def _ck(label: str) -> None:
        nonlocal done
        if NO_PROBES_AUDIT:
            # audit mode only (red-team F4): prove probes do not perturb training
            enc = _encoder_fingerprint(agent)
            print(f"  [probe-skipped] seed={seed} ckpt={label} enc={enc['hash']}", flush=True)
            ckpts.append({"checkpoint": label, "encoder": enc, "arms": None})
        else:
            ckpts.append(_probe(agent, scaffold_cfg, device, seed, label,
                                CHECKPOINTS.index(label), steps_per_ep, probe_eps))
        done += 2 * probe_eps

    _ck("init")

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0", flush=True)
        return _aborted(seed, "stage0", s0.abort_reason, ckpts)
    _ck("stage0")

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b", flush=True)
        return _aborted(seed, "stage0b", s0b.abort_reason, ckpts)
    _ck("stage0b")

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0", flush=True)
        return _aborted(seed, "p0", p0.abort_reason, ckpts)
    _ck("p0")

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard", flush=True)
        return _aborted(seed, "hazard", hz.abort_reason, ckpts)
    _ck("hazard")

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)
    _ck("p1")

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)
    _ck("p2")

    guard_pass = bool(p2.contact_rate > CONTACT_GATE
                      and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE)
    _ZG.observe(agent)

    final = _run_pair(agent, scaffold_cfg, device, seed, eval_eps, steps_per_ep,
                      env_stream=4, env_idx=0, rng_seed=seed * 100 + 99)
    done += 2 * eval_eps
    off, on = final[ARM_OFF], final[ARM_ON]
    print(f"  [train] final_eval seed={seed} ep {done}/{total_eps}", flush=True)
    for arm in ARMS:
        c = final[arm]
        print(f"  [eval] seed={seed} {arm} class={c['arm_class']} in={c['n_into_external']}"
              f" out={c['n_out_of_external']} rev_beyond_first={c['reversals_beyond_first']}"
              f" occ={c['fraction_in_external_task']} dacc_pe_mean={c['dacc_pe']['mean']}"
              f" drive_mean={c['external_task_drive']['mean']}"
              f" ep_len={c['mean_episode_length']}", flush=True)

    init_arms = ckpts[0]["arms"] if ckpts and ckpts[0].get("arms") else None
    d0 = init_arms[ARM_OFF]["dacc_pe"]["mean"] if init_arms else None
    z0 = init_arms[ARM_OFF]["z_harm_a_l2"]["mean"] if init_arms else None
    d1 = off["dacc_pe"]["mean"]
    z1 = off["z_harm_a_l2"]["mean"]
    dacc_growth = (d1 / d0) if (d0 and d1 is not None and d0 > 1e-9) else None
    zha_growth = (z1 / z0) if (z0 and z1 is not None and z0 > 1e-9) else None
    corr = off["corr_dacc_pe_vs_z_harm_a_l2"]

    drive_engaged = bool((on["external_task_drive"]["mean"] or 0.0) > DRIVE_ENGAGE_FLOOR)
    dacc_live = bool((off["dacc_pe"]["mean"] or 0.0) > DACC_LIVE_FLOOR)
    on_reversing = on["arm_class"] == "REVERSING"
    off_one_way = off["arm_class"] == "ONE_WAY"
    def _relvar(st):
        m, p10, p90 = st.get("mean"), st.get("p10"), st.get("p90")
        if m is None or p10 is None or p90 is None or abs(m) < 1e-12:
            return 0.0
        return float((p90 - p10) / abs(m))
    varies = bool(_relvar(off["dacc_pe"]) > REL_SPREAD_MIN and _relvar(off["z_harm_a_l2"]) > REL_SPREAD_MIN)
    shared_measurable = bool(varies and corr is not None and dacc_growth is not None
                             and zha_growth is not None)
    shared_ok = bool(shared_measurable and corr >= CORR_MIN
                     and dacc_growth >= GROWTH_MIN and zha_growth >= GROWTH_MIN)

    print(f"verdict: {'PASS' if (guard_pass and on_reversing) else 'FAIL'}"
          f" seed={seed} guard_pass={guard_pass} drive_engaged={drive_engaged}"
          f" dacc_live={dacc_live} ON={on['arm_class']} OFF={off['arm_class']}"
          f" corr={corr} dacc_growth={dacc_growth} zha_growth={zha_growth}", flush=True)

    return {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "drive_engaged": drive_engaged,
        "dacc_live": dacc_live,
        "testable": bool(guard_pass and drive_engaged and dacc_live),
        "on_class": on["arm_class"],
        "off_class": off["arm_class"],
        "on_reversing": on_reversing,
        "off_one_way": off_one_way,
        "corr_dacc_pe_vs_z_harm_a_l2_off": corr,
        "dacc_pe_growth_init_to_final": None if dacc_growth is None else round(dacc_growth, 4),
        "z_harm_a_l2_growth_init_to_final": None if zha_growth is None else round(zha_growth, 4),
        "shared_upstream_measurable": shared_measurable,
        "shared_upstream_inputs_vary": varies,
        "off_ext_argmax_while_away_ticks": off["n_ticks_ext_argmax_while_away"],
        "shared_upstream_ok": shared_ok,
        "checkpoints": ckpts,
        "final": final,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _majority(classes: List[str]) -> str:
    """The class held by >= MIN_FRACTION of seeds, else 'MIXED' ('NONE' if empty)."""
    if not classes:
        return "NONE"
    for c in sorted(set(classes)):
        if classes.count(c) / float(len(classes)) >= MIN_FRACTION:
            return c
    return "MIXED"


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = (2 + 2 + 3 + 3 + 3 + 2) + len(CHECKPOINTS) * 2 * 1 + 2 * 2
    else:
        total_eps = (STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
                     + P1_BUDGET + P2_BUDGET
                     + len(CHECKPOINTS) * 2 * PROBE_EPISODES + 2 * MODE_EVAL_EPISODES)

    per_seed = [_run_seed(s, dry_run, total_eps) for s in seeds]
    n = len(per_seed)

    guard_flags = [bool(r["guard_pass"]) for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_seeds = [r for r in per_seed if r["guard_pass"]]
    drive_frac = _frac([bool(r.get("drive_engaged")) for r in guard_seeds])
    dacc_frac = _frac([bool(r.get("dacc_live")) for r in guard_seeds])
    p1_met = bool(guard_frac >= MIN_FRACTION)
    p2_met = bool(drive_frac >= MIN_FRACTION)
    p3_met = bool(dacc_frac >= MIN_FRACTION)
    ready = bool(p1_met and p2_met and p3_met)

    testable = [r for r in per_seed if r.get("testable")]
    n_t = len(testable)
    c1_frac = _frac([bool(r["on_reversing"]) for r in testable])
    c2_frac = _frac([bool(r["off_one_way"]) for r in testable])
    c1 = bool(n_t >= MIN_TESTABLE_SEEDS and c1_frac >= MIN_FRACTION)
    c2 = bool(n_t >= MIN_TESTABLE_SEEDS and c2_frac >= MIN_FRACTION)
    off_return_reachable = any(int(r.get("off_ext_argmax_while_away_ticks", 0)) > 0 for r in testable)
    shared_meas = [r for r in testable if r.get("shared_upstream_measurable")]
    c3_frac = _frac([bool(r["shared_upstream_ok"]) for r in shared_meas])
    c3_measurable = bool(len(shared_meas) >= max(1, math.ceil(MIN_FRACTION * max(n_t, 1))))
    c3 = bool(c3_measurable and c3_frac >= MIN_FRACTION)

    # Non-degeneracy: the arms must differ somewhere (manipulation reached the DV)
    # and the classes used by C1/C2 must not all be PINNED_EXTERNAL.
    arms_differ = any(
        (r["final"][ARM_ON]["n_into_external"] != r["final"][ARM_OFF]["n_into_external"])
        or (r["final"][ARM_ON]["n_switches"] != r["final"][ARM_OFF]["n_switches"])
        for r in testable)
    c1_nd = bool(ready and n_t >= MIN_TESTABLE_SEEDS)
    # C2 is a STRUCTURAL CONTROL (red-team F1): it discriminates only if the OFF
    # arm's affinity side could ever put external_task back on top while the
    # register was away from it. If that count is 0 on every seed, "stays one-way"
    # is arithmetic (ext logit <= 1 + drive_level <= 2 <= clamped dacc_pe side),
    # not a finding.
    c2_nd = bool(ready and n_t >= MIN_TESTABLE_SEEDS and off_return_reachable
                 and any(r["off_class"] != "PINNED_EXTERNAL" for r in testable))

    combination_rule = ("PASS iff C1_reversals_survive_training_with_drive (single load-bearing "
                        "criterion). C2_drive_off_stays_one_way is a STRUCTURAL CONTROL "
                        "(load_bearing False): it is informative only when "
                        "off_return_reachable (the OFF arm's argmax ever returned to "
                        "external_task while the register was away); otherwise it is "
                        "arithmetic and criteria_non_degenerate marks it false. "
                        "C3_shared_upstream is a separate hypothesis (H-SHARED), not in "
                        "the PASS gate, routing its own label.")

    on_maj = _majority([r["on_class"] for r in testable])
    off_maj = _majority([r["off_class"] for r in testable])
    cls = ("|on=" + ",".join(r["on_class"] for r in testable)
           + "|off=" + ",".join(r["off_class"] for r in testable)
           + f"|off_return_reachable={off_return_reachable}")
    if not p1_met:
        outcome, label, reason = "FAIL", "substrate_not_ready_requeue", "contact_guard_unmet"
    elif not p2_met:
        outcome, label, reason = "FAIL", "substrate_not_ready_requeue", "external_task_drive_not_engaging_on_arm"
    elif not p3_met:
        outcome, label, reason = "FAIL", "substrate_not_ready_requeue", "dacc_salience_not_live"
    elif n_t < MIN_TESTABLE_SEEDS:
        outcome, label, reason = ("FAIL", "substrate_not_ready_requeue",
                                  f"n_testable_seeds_{n_t}_below_{MIN_TESTABLE_SEEDS}")
    elif c1:
        outcome = "PASS"
        if off_maj == "REVERSING":
            label = "trained_agent_reverses_with_or_without_external_task_drive_cap2"
        elif off_maj == "ONE_WAY" and off_return_reachable:
            label = "external_task_drive_needed_for_reversal_trained_cap2"
        elif off_maj == "ONE_WAY":
            label = "external_task_drive_sufficient_for_reversal_trained_cap2_off_arm_structurally_one_way"
        else:
            label = "external_task_drive_sufficient_for_reversal_trained_cap2"
        reason = "criteria_passed:C1" + cls
    else:
        # FAIL: labelled by the MAJORITY per-seed ON class -- a failed C1 can mean
        # the ON register locked in internal_planning OR never left external_task.
        outcome = "FAIL"
        if on_maj == "ONE_WAY":
            label = "trained_agent_locks_even_with_drive_cap2"
        elif on_maj == "PINNED_EXTERNAL":
            label = "drive_pins_external_task_no_reversal_cap2"
        else:
            label = "drive_effect_not_resolved"
        reason = f"criteria_failed:C1|on_majority={on_maj}|off_majority={off_maj}" + cls

    if not ready or not c3_measurable:
        shared_label = "shared_upstream_not_measurable"
    else:
        shared_label = "shared_upstream_supported" if c3 else "shared_upstream_refuted"

    if not ready or n_t < MIN_TESTABLE_SEEDS:
        sd032a_dir = "non_contributory"
    elif c1:
        sd032a_dir = "supports"      # the register alternates within a life on a trained agent
    elif on_maj == "ONE_WAY":
        sd032a_dir = "weakens"       # competed, left external_task, never returned, drive live
    else:
        # PINNED_EXTERNAL / unresolved: an affinity-margin artefact at cap 2.0 (the
        # register never competed), not a failure of alternation (red-team r2 Q4)
        sd032a_dir = "non_contributory"
    direction_map = {"SD-032a": sd032a_dir, "MECH-157": "non_contributory"}

    print(f"[{EXPERIMENT_TYPE}] ready={ready} (P1 {guard_frac:.3f} P2 {drive_frac:.3f} P3 {dacc_frac:.3f})"
          f" n_testable={n_t}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] C1 {c1_frac:.3f} passed={c1} | C2 {c2_frac:.3f} passed={c2}"
          f" | C3 {c3_frac:.3f} passed={c3} measurable={c3_measurable}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] -> outcome={outcome} route={label} shared={shared_label}", flush=True)

    preconditions = [
        {"name": "foraging_contact_guard", "kind": "readiness",
         "description": "935a's P2 contact guard (contact_rate > 0 AND z_goal_norm_at_contact_peak > 0.4) "
                        "on >= 2/3 seeds; an agent that never became foraging-competent has no active "
                        "goal and the goal-gated drive cannot engage.",
         "control": "cleared 5/5 by V3-EXQ-935a on this curriculum (seeds 47-51).",
         "measured": round(guard_frac, 4), "threshold": MIN_FRACTION, "direction": "lower",
         "met": p1_met},
        {"name": "drive_engages_on_arm", "kind": "readiness",
         "description": "mean external_task_drive input in ARM_DRIVE_ON final eval > floor on >= 2/3 "
                        "guard seeds. C1 routes on ON-arm reversals, which need the drive input live; a "
                        "zero drive starves C1 rather than falsifying it.",
         "control": "935a ARM cells at the same config: external_task margins 0.32-0.81 on 5/5 seeds.",
         "measured": round(drive_frac, 4), "threshold": MIN_FRACTION, "direction": "lower",
         "met": p2_met},
        {"name": "dacc_salience_live", "kind": "readiness",
         "description": "mean dacc_pe in ARM_DRIVE_OFF final eval > floor on >= 2/3 guard seeds. The "
                        "dACC alarm is the only default salience source; without it neither arm can "
                        "switch (the dACC-off wiring failure), which is not the question here.",
         "control": "untrained modetrace A1: dacc_pe 0.19-1.27 on 3/3 seeds.",
         "measured": round(dacc_frac, 4), "threshold": MIN_FRACTION, "direction": "lower",
         "met": p3_met},
    ]
    criteria = [
        {"name": "C1_reversals_survive_training_with_drive", "load_bearing": True, "passed": c1,
         "measured": round(c1_frac, 4), "threshold": MIN_FRACTION, "comparator": ">=",
         "per_seed_statistic": f"ARM_DRIVE_ON n_into_external >= {INTO_EXT_MIN} over "
                               f"{MODE_EVAL_EPISODES} lives"},
        {"name": "C2_drive_off_stays_one_way", "load_bearing": False, "passed": c2,
         "role": "structural control; informative only if off_return_reachable",
         "off_return_reachable": off_return_reachable,
         "measured": round(c2_frac, 4), "threshold": MIN_FRACTION, "comparator": ">=",
         "per_seed_statistic": f"ARM_DRIVE_OFF n_into_external <= 1 AND n_out_of_external >= "
                               f"{LEAVE_FRAC} x lives"},
        {"name": "C3_shared_upstream", "load_bearing": False, "passed": c3,
         "measured": round(c3_frac, 4), "threshold": MIN_FRACTION, "comparator": ">=",
         "per_seed_statistic": f"corr(dacc_pe, ||z_harm_a||_2) >= {CORR_MIN} AND dacc_pe and "
                               f"||z_harm_a||_2 mean growth init->final >= {GROWTH_MIN}x"},
    ]
    criteria_non_degenerate = {
        "C1_reversals_survive_training_with_drive": c1_nd,
        "C2_drive_off_stays_one_way": c2_nd,
        "C3_shared_upstream": bool(ready and c3_measurable),
    }

    def _mean_final(arm: str, key: str) -> Optional[float]:
        vals = [float(r["final"][arm][key]) for r in testable
                if r.get("final") and r["final"][arm].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    def _mean_stat(arm: str, key: str) -> Optional[float]:
        vals = [r["final"][arm][key]["mean"] for r in testable
                if r.get("final") and r["final"][arm][key]["mean"] is not None]
        return round(float(np.mean(vals)), 5) if vals else None

    readout = flat_readout({
        "C1_reversals_survive_training_with_drive": c1,
        "C2_drive_off_stays_one_way": c2,
        "off_return_reachable": off_return_reachable,
        "C3_shared_upstream": c3,
        "c1_fraction": c1_frac, "c2_fraction": c2_frac, "c3_fraction": c3_frac,
        "min_fraction": MIN_FRACTION,
        "ready": ready, "guard_fraction": guard_frac, "drive_engaged_fraction": drive_frac,
        "dacc_live_fraction": dacc_frac, "n_seeds": n, "n_testable_seeds": n_t,
        "arms_differ": arms_differ,
        "into_ext_min": INTO_EXT_MIN, "eval_cap": EVAL_CAP,
        "on_mean_into_external": _mean_final(ARM_ON, "n_into_external"),
        "off_mean_into_external": _mean_final(ARM_OFF, "n_into_external"),
        "on_mean_reversals_per_life": _mean_final(ARM_ON, "reversals_per_life"),
        "off_mean_reversals_per_life": _mean_final(ARM_OFF, "reversals_per_life"),
        "on_mean_occupancy_external": _mean_final(ARM_ON, "fraction_in_external_task"),
        "off_mean_occupancy_external": _mean_final(ARM_OFF, "fraction_in_external_task"),
        "on_mean_episode_length": _mean_final(ARM_ON, "mean_episode_length"),
        "off_mean_episode_length": _mean_final(ARM_OFF, "mean_episode_length"),
        "off_dacc_pe_mean": _mean_stat(ARM_OFF, "dacc_pe"),
        "off_z_harm_a_l2_mean": _mean_stat(ARM_OFF, "z_harm_a_l2"),
        "on_external_task_drive_mean": _mean_stat(ARM_ON, "external_task_drive"),
    })

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": direction_map,
        "readout": readout,
        "interpretation": {
            "label": label,
            "route_reason": reason,
            "shared_upstream_label": shared_label,
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "arms_differ": arms_differ,
            "dv_symmetry": {
                "dv": "register switch sequence (argmax(operating_mode) changes gated by "
                      "salience_aggregate > threshold)",
                "ARM_DRIVE_OFF_vs_ON": "NOT INVARIANT: removes a signed external_task-only logit "
                                       "term and a salience term -- changes logit differences, "
                                       "not a common offset; untrained D2 moved 0/0/0 -> 2/16/25.",
            },
            "premise_corrected": "935a's config already had use_external_task_drive=True; the new "
                                 "arm is the drive-OFF lesion of the same trained weights.",
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
        "arms": ARMS, "eval_cap": EVAL_CAP, "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
        "affinity_bound_mode": "clamp (coordinator default; 935a-identical, not the 2026-09-19 squash)",
        "rails": "symmetric (enter/exit per-mode thresholds cleared) on every eval cell, as 935a",
        "mode_eval_episodes": MODE_EVAL_EPISODES, "probe_episodes": PROBE_EPISODES,
        "checkpoints": CHECKPOINTS, "into_ext_min": INTO_EXT_MIN, "leave_frac": LEAVE_FRAC,
        "min_fraction": MIN_FRACTION, "drive_engage_floor": DRIVE_ENGAGE_FLOOR,
        "dacc_live_floor": DACC_LIVE_FLOOR, "corr_min": CORR_MIN, "growth_min": GROWTH_MIN,
        "env_seed_base": ENV_SEED_BASE, "seeds": SEEDS, "train_steps": TRAIN_STEPS,
        "p2_zgoal_gate": P2_ZGOAL_GATE, "contact_gate": CONTACT_GATE,
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET, "config_basis": "V3-EXQ-935a",
        },
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
        "source_record": SOURCE_RECORD,
        "predecessor": "V3-EXQ-935a (curriculum + config source; not superseded)",
        "supersedes": None,
        "condition": CONDITION_LABEL,
        "pre_registered_thresholds": {
            "into_ext_min": INTO_EXT_MIN, "leave_frac": LEAVE_FRAC, "min_fraction": MIN_FRACTION,
            "drive_engage_floor": DRIVE_ENGAGE_FLOOR, "dacc_live_floor": DACC_LIVE_FLOOR,
            "corr_min": CORR_MIN, "growth_min": GROWTH_MIN, "eval_cap": EVAL_CAP,
        },
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "config": full_config,
        "stage_plan": stage_plan(),
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
    _res = main(dry_run=args.dry_run)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        dry_run=bool(args.dry_run),
    )
