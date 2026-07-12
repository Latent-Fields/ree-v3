"""
V3-EXQ-460l (SUPERSEDES V3-EXQ-460k): ARC-108 JOB-2 CONTROL-PLANE L0/L1/L2 falsifier --
the dopaminergic DRIVER pair (rho_t maintenance ramp + habenula negative-delta_t de-commit,
BUILT ree-v3 main c5614ab 2026-06-22) on the now-BUILT CLOSURE-EXCLUSIVE DE-COMMIT EVAL
substrate (closure_exclusive_decommit_eval, ree-v3 main e52158d 2026-06-22).

DISTINCT FROM V3-EXQ-700 (the JOB-1 sec-7.1 SELECTION 2x2 learned-gating falsifier,
use_learned_channel_gating x use_learned_settling_step). This is the JOB-2 sec-7.2
CONTROL-PLANE falsifier: does giving the commit/maintain/de-commit MACHINERY its missing
dopaminergic DRIVER (a) self-limit the flat-hold monopoly (B6 / 460h) and (b) supply a
content-driven, dissociable de-commit (the parked rung-6 non-dissociability) where the
hand-specified arithmetic plumbing alone could not? Design-of-record:
REE_assembly/evidence/planning/unified_dopamine_substrate_design_2026-06-22.md sec 7.2;
substrate doc REE_assembly/docs/architecture/arc_108_job2_control_plane.md.

WHY 460l SUPERSEDES 460k: 460k tested the rung-6 NaturalCommitUrgencyRelease lever (a
graded *duration* release with no dopaminergic driver) on the closure-exclusive eval
substrate. The ARC-108 JOB-2 design replaces that hand-tuned duration lever with the
biologically-faithful DRIVER pair: the rho_t maintenance ramp REPLACES the flat-hold
maintenance driver (a proximity-scaled DA ramp that peaks-then-declines, so it cannot
monopolise -- the structural B6 fix), and the habenula negative-delta_t abort ADDS a
content-driven de-commit input to the SD-034 closure operator (the rung-6 fix done
properly: fires on outcome valence, dissociable from the refractory clock). The rung-6
NaturalCommitUrgencyRelease is OFF in EVERY arm here -- the rho ramp is the maintenance
driver under test. Do NOT re-author V3-EXQ-460k.

SUBSTRATE (IDENTICAL to 460k: the 603n foraging-competent curriculum + the SD-034
commitment-closure-control-plane amend Legs A/B/C + beta-engagement coupling +
de-commit-authority magnitude lever + the natural-commit LATCH-HOLD +
closure_exclusive_decommit_eval ON in every arm). The closure-exclusive eval mode (a)
makes beta elevation closure-EXCLUSIVE (_commit_for_beta = _closure_commit_active only;
the fragile F-driven result.committed path is SUPPRESSED from beta) AND (b) arms the
natural-commit latch-hold on the closure-coupled commit, so the OFF baseline sustains a
natural-commit occupancy via the closure plane -- the only regime where natural-commit
occupancy is dissociable from closure-de-commit.

ARMS (one curriculum build per seed -- DRIVER pair OFF during training -- then THREE eval
arms, each a clone of the SAME trained weights re-configured with the arm's driver config;
the rho ramp + habenula carry no trainable parameters so the clone is exact, mirroring
460k/460h). closure_exclusive_decommit_eval + use_closure_commit_beta_coupling +
use_natural_commit_latch_hold are ON in EVERY arm; the variable is the JOB-2 DRIVER pair:
  L0 FLAT_LATCH    -- flat bistable hold (latch-hold re-asserts beta unconditionally),
                      refractory-timer de-commit only. rho ramp OFF, habenula OFF. The
                      monolithic-hold reference (the 460h B6 signature).
  L1 RHO_RAMP      -- rho_t proximity ramp ON (use_rho_maintenance_ramp=True, requires
                      use_natural_commit_latch_hold=True). The unconditional re-assert is
                      replaced by a ramp-gated one: the hold self-limits once rho_t declines
                      from its proximity peak. refractory-timer de-commit only (habenula OFF).
  L2 RHO_HABENULA  -- rho_t ramp ON + habenula negative-delta_t de-commit ON
                      (use_habenula_decommit=True, requires use_closure_operator=True <-
                      use_lateral_pfc_analog=True; reuses the JOB-1 signed-RPE
                      delta_t = R_t - V-hat_t already computed in e3_selector.post_action_update
                      via the E3Config mirror). habenula_decommit_delta_threshold = 0.0
                      (fire on any worse-than-expected outcome).

The eval loop CALLS agent.update_residue() each tick (unlike 460k's frozen-policy eval) so
the waking post-action path runs -- this is what computes delta_t and routes it into the
habenula abort. update_residue is called in ALL arms (identical residue dynamics; the only
variables are the rho ramp [L1/L2] and the habenula [L2]).

READINESS / NON-VACUITY (all must clear before D1/D2/D3 are scored; any unmet self-routes
substrate_not_ready_requeue -- NEVER a false weakens):
  (1) 603n foraging contact guard: per-seed P2 contact_rate > 0 AND
      z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.
  (2) rule_bias_head trained (anti-460d-bug gate): P1 rule_bias_pathway_enabled AND mean
      per-candidate |bias| > floor on >= 2/3 seeds (closures need a magnitude-bearing
      rule_state to form for the de-commit to act on).
  (3) CLOSURE-EXCLUSIVE EVAL ARMED THE HOLD (the 460k gate 2.5; = the "dissociable eval
      mode arms" non-vacuity): on L0 the closure-exclusive eval mode actually armed the
      closure-coupled latch-hold -- ncl_hold_closure_armed_total > 0 AND
      ncl_hold_reassert_total > 0 on >= 2/3 guard seeds. If ~0 the dissociable occupancy
      did not form -> substrate_not_ready_requeue.
  (4) L0 MONOLITHIC-HOLD BASELINE: L0 reproduces the flat-hold monopoly (mean per-commit
      hold length >= SUSTAINED_HOLD_MEAN_FLOOR) on >= 2/3 guard seeds -- the reference D1
      shows the ramp self-limiting AGAINST. If L0 does not monopolise there is nothing to
      self-limit -> substrate_not_ready_requeue.
  (5) RHO PROXIMITY VARIANCE (the JOB-2 ramp non-vacuity): on L1 the rho ramp had proximity
      to ramp on AND self-limited -- rho_peak_max > RHO_PEAK_FLOOR AND rho_n_releases_total
      > 0 on >= 2/3 guard seeds. If rho_t carries no proximity variance the ramp cannot
      peak-then-decline -> substrate_not_ready_requeue (NEVER a false weakens).
  (6) DELTA_T NEGATIVE VARIANCE (the JOB-2 habenula non-vacuity): on L2 the signed RPE
      delta_t carried worse-than-expected variance to abort on -- n_neg_delta_ticks >=
      MIN_NEG_DELTA_TICKS on >= 2/3 guard seeds. If delta_t never goes negative the
      habenula has nothing to fire on -> substrate_not_ready_requeue (NEVER a false weakens).

PRE-REGISTERED DISCRIMINATORS (constants; scored only once all six readiness gates clear).
The negative outcome is a ROUTE, not a falsification (this is a control-plane readiness
falsifier): a preconditions-met no-lift (the ramp varies + the habenula has negative
variance + the eval armed, but the hold still monopolises / behaviour does not lift) is the
genuine "maintenance is not the binding constraint" outcome -> route to JOB-1 / selection
(maintenance_not_binding_route_to_selection, non_contributory), NEVER a false weakening of
the ramp+habenula design.
  D1 (PRIMARY, load-bearing -- ramp self-limits where the flat latch monopolises): on
     >= 2/3 guard seeds, L1 produces a BOUNDED occupancy (max consecutive beta run strictly
     below L0 by >= D1_OCC_BOUND_FRAC, no single hold monopolising) AND committed-action-class
     diversity rises strict-above L0 (committed_class_entropy(L1) > committed_class_entropy(L0)
     + D1_ENTROPY_MARGIN). Readouts: rho_n_releases / rho_last_ticks_at_release +
     ncl_hold_reassert_total + max_consecutive_beta_run + committed_class_entropy.
  D2 (load-bearing -- the CORE discriminator: release content-driven, NOT a re-parameterised
     timer): on >= 2/3 guard seeds, L1 release is content-driven -- rho_n_releases_total > 0
     AND the per-episode release-tick dispersion std(rho_last_ticks_at_release) >
     RELEASE_TICK_STD_FLOOR (variable timing tracking proximity; a flat hold with a shorter
     FIXED duration would release at a constant tick = std ~ 0 = NOT the claim). The ramp
     releases ONLY at the proximity-decline crossing by construction (rho_last_decline_frac
     >= release_margin), so a non-trivial release-tick spread certifies the release tracks
     content, not a clock.
  D3 (load-bearing -- habenula de-commit dissociable): on >= 2/3 guard seeds, L2 fires the
     content-driven habenula abort -- n_habenula_aborts >= MIN_HABENULA_ABORTS -- and the
     negative-delta_t precondition (gate 6) held, so the aborts are dissociable from the
     latch refractory phase (they fired on outcome valence, not the clock). Readouts:
     closure_operator.get_state() n_habenula_aborts + the habenula_decommit_fired count from
     agent.update_residue + the L2-vs-L1 occupancy reduction (secondary).

OVERALL PASS = D1 AND D2 AND D3 on >= 2/3 guard seeds (the rho ramp self-limits the monopoly
AND the release is content-driven AND the habenula de-commit is dissociable). A
preconditions-met no-lift on D1 routes to selection (non_contributory).

claim_ids (the JOB-2 control-plane portion): MECH-090 (commitment latch -- the ramp drives
  its release; D1), MECH-342 (maintenance-release -- the rho ramp IS the maintenance driver;
  D1+D2), MECH-445 (closure->beta coupling engagement -- gate 3 + the de-commit; D3
  precondition), MECH-446 (de-commit-authority / occupancy drop -- the habenula de-commit;
  D3), ARC-108 (the unified dopamine substrate / JOB-2 driver pair; D1+D2+D3).
experiment_purpose: evidence.
supersedes: V3-EXQ-460k (the rung-6 duration-lever falsifier; superseded by the ARC-108
  JOB-2 DRIVER pair).

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_460l_job2_control_plane_ramp_habenula_falsifier"
QUEUE_ID = "V3-EXQ-460l"
# JOB-2 control-plane portion: MECH-090 (latch release, D1), MECH-342 (maintenance-release
# = the rho ramp, D1+D2), MECH-445 (closure->beta coupling, D3 precondition), MECH-446
# (de-commit occupancy drop = the habenula, D3), ARC-108 (the unified dopamine driver pair).
CLAIM_IDS: List[str] = ["MECH-090", "MECH-342", "MECH-445", "MECH-446", "ARC-108"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "v3_exq_460k_natural_commit_occupancy_release_decommit_falsifier"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_JOB2_CONTROL_PLANE_RAMP_HABENULA_L0_L1_L2"

# --- Goal-pipeline / encoder dims (mirror 603n / 460k exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C; mirror 460k) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- ARC-108 JOB-2 (c) rho_t maintenance ramp (REEConfig defaults; L1/L2) ---
RHO_HOLD_FLOOR = 0.05         # release when rho_t below floor
RHO_RELEASE_MARGIN = 0.5      # release when declined >= margin * peak past the proximity peak
RHO_ONSET_GRACE_TICKS = 3     # let the ramp rise to its peak before it can self-limit

# --- ARC-108 JOB-2 (d) habenula negative-delta_t de-commit (REEConfig defaults; L2) ---
HABENULA_DECOMMIT_DELTA_THRESHOLD = 0.0   # fire when delta_t < this (worse-than-expected)

# --- Within-arm around-closure window DV (secondary D3 readout; mirror 460k) ---
CLOSURE_WINDOW = 10
WINDOW_MIN_TICKS = 3
C2_MIN_WINDOW_EVENTS = 2
WITHIN_PRE_OCC_FLOOR = 0.1

# --- Curriculum budgets (mirror 603n / 460k exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
CLOSURE_EVAL_EPISODES = 15  # per arm (x3 arms)
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

# --- Pre-registered acceptance thresholds ---
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
RULE_BIAS_MEAN_FLOOR = 0.005

# Readiness floors.
SUSTAINED_HOLD_MEAN_FLOOR = 5.0   # L0 mean per-commit hold floor (the monolithic-hold reference)
RHO_PEAK_FLOOR = 1e-4             # L1 rho_t must reach a non-trivial proximity peak
MIN_NEG_DELTA_TICKS = 3           # L2 must observe >= this many worse-than-expected delta_t ticks

# Discriminator thresholds.
D1_OCC_BOUND_FRAC = 0.15         # L1 max consecutive beta run must drop >= this frac below L0
D1_ENTROPY_MARGIN = 0.05         # committed_class_entropy(L1) must exceed L0 by >= this (nats)
RELEASE_TICK_STD_FLOOR = 0.5     # L1 per-episode rho release-tick dispersion (content-driven, not a clock)
MIN_HABENULA_ABORTS = 1          # L2 content-driven habenula de-commits

# --- Eval-arm definitions (JOB-2 DRIVER pair config; closure-exclusive eval ON in every
#     arm; rung-6 NaturalCommitUrgencyRelease OFF in every arm). ---
L0_ARM = "L0_FLAT_LATCH"
L1_ARM = "L1_RHO_RAMP"
L2_ARM = "L2_RHO_HABENULA"
ARMS: List[Dict[str, Any]] = [
    {"key": L0_ARM, "driver": {"rho": False, "habenula": False}},
    {"key": L1_ARM, "driver": {"rho": True, "habenula": False}},
    {"key": L2_ARM, "driver": {"rho": True, "habenula": True}},
]


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
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
        scaffold_train_rule_bias_head=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate (mirror 460k) + the commitment control-plane +
    commitment-closure-control-plane amend Legs A/B/C + beta-engagement coupling +
    the closure-exclusive de-commit eval mode + the natural-commit latch-hold. The ARC-108
    JOB-2 DRIVER pair (rho ramp + habenula) is LEFT OFF here (the trained-base config); it
    is armed per-arm at eval by _clone_arm so all three arms share one trained substrate
    (the ramp + habenula carry no trainable parameters). The rung-6
    NaturalCommitUrgencyRelease is OFF in every arm (the rho ramp is the maintenance driver
    under test)."""
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
        use_closure_operator=True,
        # SD-034 commitment-closure-control-plane amend (Legs A/B/C + coupling):
        use_closure_env_completion_hook=True,          # Leg A
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D)
        use_closure_commit_beta_coupling=True,         # beta-engagement coupling
        # rung-6 natural-commit-occupancy-release lever: OFF in every arm (the rho ramp is
        # the JOB-2 maintenance driver under test, not the hand-tuned duration lever).
        use_natural_commit_urgency_release=False,
        # The natural-commit LATCH-HOLD is ARMED on the base config (carried into every arm
        # via _clone_arm's deepcopy). It re-asserts the beta latch so the OFF baseline
        # sustains via the closure plane. The rho ramp (L1/L2) REPLACES its unconditional
        # re-assert with a ramp-gated one (the B6 fix).
        use_natural_commit_latch_hold=True,
        # The CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode (BUILT ree-v3 e52158d 2026-06-22): beta
        # elevation closure-EXCLUSIVE + latch-hold arms on _closure_commit_active so the OFF
        # baseline sustains a natural-commit occupancy via the closure plane (the only regime
        # where natural-commit occupancy is dissociable from closure-de-commit). ON in EVERY
        # arm. Preconditions (loud ValueError at REEAgent.__init__):
        # use_closure_commit_beta_coupling AND use_natural_commit_latch_hold -- both set.
        closure_exclusive_decommit_eval=True,
        # ARC-108 JOB-2 DRIVER pair: OFF on the trained base; armed per-arm at eval.
        use_rho_maintenance_ramp=False,
        rho_hold_floor=RHO_HOLD_FLOOR,
        rho_release_margin=RHO_RELEASE_MARGIN,
        rho_onset_grace_ticks=RHO_ONSET_GRACE_TICKS,
        use_habenula_decommit=False,
        habenula_decommit_delta_threshold=HABENULA_DECOMMIT_DELTA_THRESHOLD,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_closure_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint
    tolerance-band completion so the SD-034 closure operator has completions to fire on
    (mirror 460k)."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
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
        subgoal_mode=True,
        num_waypoints=2,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.25,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
        **_sd049_kwargs(scaffold_cfg),
    )


def _clone_arm(trained_agent: REEAgent, device: torch.device, arm: Dict[str, Any]) -> REEAgent:
    """Clone the SAME trained weights into an agent built with this arm's JOB-2 DRIVER
    config (rho ramp x habenula). The closure-exclusive eval + latch-hold + closure operator
    stay ON in every arm (the DRIVER pair -- not the machinery -- is the variable). The rho
    ramp + habenula carry no trainable parameters, so the state_dict loads cleanly (mirrors
    460k's _clone_arm)."""
    cfg = copy.deepcopy(trained_agent.config)
    dv = arm["driver"]
    # (c) rho_t maintenance ramp (L1/L2). Requires use_natural_commit_latch_hold (carried).
    cfg.use_rho_maintenance_ramp = bool(dv["rho"])
    cfg.rho_hold_floor = RHO_HOLD_FLOOR
    cfg.rho_release_margin = RHO_RELEASE_MARGIN
    cfg.rho_onset_grace_ticks = RHO_ONSET_GRACE_TICKS
    # (d) habenula negative-delta_t de-commit (L2 only). Requires use_closure_operator
    # (carried). The E3Config mirror drives post_action_update to emit habenula_delta_t.
    cfg.use_habenula_decommit = bool(dv["habenula"])
    cfg.habenula_decommit_delta_threshold = HABENULA_DECOMMIT_DELTA_THRESHOLD
    cfg.e3.use_habenula_decommit = bool(dv["habenula"])
    # rung-6 lever stays OFF in every arm.
    cfg.use_natural_commit_urgency_release = False
    cfg.use_closure_operator = True
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        agent.load_state_dict(state, strict=False)
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    agent.beta_gate = BetaGate(completion_release_threshold=2.0)
    return agent


def _around_closure_windows(
    beta_history: List[bool], fire_ticks: List[int]
) -> List[Dict[str, float]]:
    """For each closure fire at tick t, compute the beta-latch occupancy FRACTION in the
    pre-closure window [t-W, t) and the post-closure window (t, t+W], requiring at least
    WINDOW_MIN_TICKS available ticks on each side (the paired within-arm de-commit datum;
    mirror 460k). For L2 these windows capture the habenula aborts (which fire the SD-034
    closure)."""
    n = len(beta_history)
    events: List[Dict[str, float]] = []
    for t in fire_ticks:
        pre_lo = max(0, t - CLOSURE_WINDOW)
        pre = beta_history[pre_lo:t]
        post_hi = min(n, t + 1 + CLOSURE_WINDOW)
        post = beta_history[t + 1:post_hi]
        if len(pre) < WINDOW_MIN_TICKS or len(post) < WINDOW_MIN_TICKS:
            continue
        pre_occ = sum(1 for b in pre if b) / float(len(pre))
        post_occ = sum(1 for b in post if b) / float(len(post))
        events.append({"pre_occ": pre_occ, "post_occ": post_occ})
    return events


def _max_consecutive_true(seq: List[bool]) -> int:
    """Longest run of consecutive True (beta-elevated) ticks -- the monolithic-hold proxy
    (L0 high; the rho ramp self-limit bounds it in L1/L2)."""
    best = 0
    cur = 0
    for b in seq:
        if b:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _committed_class_entropy(counts: Counter) -> float:
    """Shannon entropy (nats) of the committed first-action class distribution. Higher =
    more committed-action diversity (the D1 lift readout)."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _eval_arm_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Eval instrumented for the ARC-108 JOB-2 DRIVER pair. UNLIKE 460k's frozen-policy
    eval, this loop CALLS agent.update_residue() each tick so the waking post-action path
    runs -- that computes the signed RPE delta_t and routes it into the habenula abort (L2).
    Reads: the rho maintenance ramp get_state() (rho_n_releases / rho_last_ticks_at_release
    / rho_peak, per-episode before the next agent.reset() wipes them) + the closure operator
    get_state() n_habenula_aborts (cumulative diff) + the latch-hold occupancy + committed
    first-action class entropy + the L2 signed-RPE delta_t distribution."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_dacc = getattr(agent, "dacc", None) is not None
    has_rho = getattr(agent, "rho_maintenance_ramp", None) is not None
    hook_enabled = bool(getattr(agent.config, "use_closure_env_completion_hook", False))
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    habenula_pre = (
        int(agent.closure_operator.get_state().get("n_habenula_aborts", 0))
        if has_closure else 0
    )
    beta_release_events = 0
    nogo_installed_total = 0
    total_committed_steps = 0
    total_beta_elevated = 0
    n_sequence_completions = 0
    n_hook_fires = 0
    n_closure_commit_intent = 0
    n_closure_coupled_elevations = 0
    around_events: List[Dict[str, float]] = []
    max_consecutive_beta_run = 0
    ncl_hold_reassert_total = 0
    ncl_hold_closure_armed_total = 0
    # JOB-2 rho ramp accumulators (per-episode read).
    rho_n_releases_total = 0
    rho_peak_max = 0.0
    rho_peaks: List[float] = []
    rho_release_ticks: List[int] = []   # per-episode rho_last_ticks_at_release (when released)
    # JOB-2 habenula accumulators.
    habenula_decommit_fired_count = 0
    delta_t_values: List[float] = []
    n_neg_delta_ticks = 0
    # Committed first-action class histogram (D1 diversity readout).
    committed_class_counts: Counter = Counter()

    for _ in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        prev_beta = bool(agent.beta_gate.is_elevated)
        beta_history: List[bool] = []
        fire_ticks: List[int] = []

        for tick_idx in range(steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = _sense_with_optional_harm(
                agent, obs_body, obs_world, obs_dict, device, feed_harm
            )

            n_closures_before = (
                int(agent.closure_operator._n_closures) if has_closure else 0
            )
            dacc_hist_before = len(agent.dacc._action_history) if has_dacc else 0

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            if has_closure:
                fired_now = int(agent.closure_operator._n_closures) - n_closures_before
                if fired_now > 0 and has_dacc:
                    nogo_installed_total += (
                        len(agent.dacc._action_history) - dacc_hist_before
                    )

            cur_beta = bool(agent.beta_gate.is_elevated)
            beta_history.append(cur_beta)
            committed_now = agent.e3._committed_trajectory is not None
            if committed_now:
                total_committed_steps += 1
                # Committed first-action class (the D1 diversity readout): the action class
                # the agent is committed to executing this tick.
                committed_class_counts[action_idx] += 1
            if cur_beta:
                total_beta_elevated += 1
            if prev_beta and not cur_beta:
                beta_release_events += 1
            prev_beta = cur_beta

            _, _harm, done, info, obs_dict = env.step(action_idx)

            # JOB-2: drive the waking post-action path so delta_t is computed + routed into
            # the habenula abort (L2). update_residue runs post_action_update (running
            # variance + V-hat EMA + the signed RPE) and, when use_habenula_decommit is on,
            # the SD-034 habenula_tick. Called in ALL arms (identical residue dynamics; the
            # only variables are the rho ramp [L1/L2] + the habenula [L2]).
            resid_metrics = agent.update_residue(
                harm_signal=float(_harm), hypothesis_tag=False
            )
            _dt = resid_metrics.get("e3_habenula_delta_t")
            if _dt is not None:
                try:
                    dt_val = float(_dt.item()) if hasattr(_dt, "item") else float(_dt)
                except Exception:
                    dt_val = None
                if dt_val is not None:
                    delta_t_values.append(dt_val)
                    if dt_val < HABENULA_DECOMMIT_DELTA_THRESHOLD:
                        n_neg_delta_ticks += 1
            if "habenula_decommit_fired" in resid_metrics:
                habenula_decommit_fired_count += 1

            if info.get("transition_type") == "sequence_complete":
                n_sequence_completions += 1
                if has_closure and hook_enabled:
                    ev = agent.notify_env_completion(action_class=action_idx)
                    if ev is not None and getattr(ev, "fired", False):
                        n_hook_fires += 1
                        nogo_installed_total += int(getattr(ev, "nogo_pushed", 0))

            if has_closure and int(agent.closure_operator._n_closures) > n_closures_before:
                fire_ticks.append(tick_idx)
            if done:
                break

        around_events.extend(_around_closure_windows(beta_history, fire_ticks))
        _ep_run = _max_consecutive_true(beta_history)
        if _ep_run > max_consecutive_beta_run:
            max_consecutive_beta_run = _ep_run
        # Read per-episode counters BEFORE the next agent.reset() wipes them.
        ncl_hold_reassert_total += int(getattr(agent, "_ncl_hold_reassert_count", 0))
        ncl_hold_closure_armed_total += int(
            getattr(agent, "_ncl_hold_closure_armed_count", 0)
        )
        if has_rho:
            rstate = agent.rho_maintenance_ramp.get_state()
            ep_rel = int(rstate.get("rho_n_releases", 0))
            rho_n_releases_total += ep_rel
            ep_peak = float(rstate.get("rho_peak", 0.0))
            rho_peaks.append(ep_peak)
            if ep_peak > rho_peak_max:
                rho_peak_max = ep_peak
            if ep_rel > 0:
                rho_release_ticks.append(int(rstate.get("rho_last_ticks_at_release", 0)))
        _bstate = agent.beta_gate.get_state()
        n_closure_commit_intent += int(_bstate.get("sd034_n_closure_commit_intent", 0))
        n_closure_coupled_elevations += int(
            _bstate.get("sd034_n_closure_coupled_elevations", 0)
        )

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre if has_closure else 0
    )
    n_habenula_aborts = (
        int(agent.closure_operator.get_state().get("n_habenula_aborts", 0)) - habenula_pre
        if has_closure else 0
    )
    n_window_events = len(around_events)
    mean_pre_occ = (
        sum(e["pre_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    mean_post_occ = (
        sum(e["post_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    release_tick_std = (
        float(statistics.pstdev(rho_release_ticks)) if len(rho_release_ticks) >= 2 else 0.0
    )
    delta_t_std = (
        float(statistics.pstdev(delta_t_values)) if len(delta_t_values) >= 2 else 0.0
    )
    delta_t_min = float(min(delta_t_values)) if delta_t_values else 0.0
    return {
        "n_closures": n_closures,
        "n_habenula_aborts": n_habenula_aborts,
        "habenula_decommit_fired_count": habenula_decommit_fired_count,
        "sd034_n_closure_commit_intent": n_closure_commit_intent,
        "sd034_n_closure_coupled_elevations": n_closure_coupled_elevations,
        "n_hook_fires": n_hook_fires,
        "n_automatic_fires": max(0, n_closures - n_hook_fires),
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "mean_beta_elevated_steps": total_beta_elevated / max(1, n_eps),
        "mean_per_commit_hold": total_beta_elevated / max(1, beta_release_events),
        "max_consecutive_beta_run": max_consecutive_beta_run,
        "ncl_hold_reassert_total": ncl_hold_reassert_total,
        "ncl_hold_closure_armed_total": ncl_hold_closure_armed_total,
        "n_sequence_completions": n_sequence_completions,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        "env_hook_enabled": hook_enabled,
        "n_window_events": n_window_events,
        "mean_pre_closure_occ": mean_pre_occ,
        "mean_post_closure_occ": mean_post_occ,
        # JOB-2 rho ramp diagnostics (D1 / D2).
        "rho_present": has_rho,
        "rho_n_releases_total": rho_n_releases_total,
        "rho_peak_max": rho_peak_max,
        "rho_mean_peak": (sum(rho_peaks) / len(rho_peaks)) if rho_peaks else 0.0,
        "rho_n_release_episodes": len(rho_release_ticks),
        "rho_release_tick_std": release_tick_std,
        "rho_release_tick_mean": (
            sum(rho_release_ticks) / len(rho_release_ticks) if rho_release_ticks else 0.0
        ),
        # JOB-2 habenula diagnostics (D3 / non-vacuity gate 6).
        "n_neg_delta_ticks": n_neg_delta_ticks,
        "n_delta_ticks": len(delta_t_values),
        "delta_t_min": delta_t_min,
        "delta_t_std": delta_t_std,
        # Committed first-action class diversity (D1 lift readout).
        "committed_class_entropy": _committed_class_entropy(committed_class_counts),
        "n_committed_classes": len(committed_class_counts),
    }


# ---------------------------------------------------------------------------
# Readiness / non-vacuity gate helpers (all keyed on the relevant arm)
# ---------------------------------------------------------------------------
def _closure_exclusive_eval_armed(arm_l0: Dict[str, Any]) -> bool:
    """Gate 3 (= the 460k gate 2.5; the 'dissociable eval mode arms' non-vacuity): on L0 the
    closure-exclusive eval mode actually armed the closure-coupled latch-hold --
    ncl_hold_closure_armed_total > 0 AND ncl_hold_reassert_total > 0."""
    return bool(
        int(arm_l0.get("ncl_hold_closure_armed_total", 0)) > 0
        and int(arm_l0.get("ncl_hold_reassert_total", 0)) > 0
    )


def _l0_monolithic(arm_l0: Dict[str, Any]) -> bool:
    """Gate 4: L0 reproduces the flat-hold monopoly (mean per-commit hold >= floor) -- the
    reference the rho ramp self-limits against."""
    return bool(float(arm_l0.get("mean_per_commit_hold", 0.0)) >= SUSTAINED_HOLD_MEAN_FLOOR)


def _rho_proximity_varied(arm_l1: Dict[str, Any]) -> bool:
    """Gate 5 (rho non-vacuity): on L1 the rho ramp reached a non-trivial proximity peak AND
    self-limited at least once -- rho_peak_max > RHO_PEAK_FLOOR AND rho_n_releases_total > 0.
    If rho_t carries no proximity variance the ramp cannot peak-then-decline."""
    return bool(
        float(arm_l1.get("rho_peak_max", 0.0)) > RHO_PEAK_FLOOR
        and int(arm_l1.get("rho_n_releases_total", 0)) > 0
    )


def _delta_negative_varied(arm_l2: Dict[str, Any]) -> bool:
    """Gate 6 (habenula non-vacuity): on L2 the signed RPE delta_t carried worse-than-expected
    variance to abort on -- n_neg_delta_ticks >= MIN_NEG_DELTA_TICKS. If delta_t never goes
    negative the habenula has nothing to fire on."""
    return bool(int(arm_l2.get("n_neg_delta_ticks", 0)) >= MIN_NEG_DELTA_TICKS)


def _d1_ramp_self_limits(arm_l0: Dict[str, Any], arm_l1: Dict[str, Any]) -> bool:
    """D1 (PRIMARY): the rho ramp self-limits where the flat latch monopolises. L1 produces a
    BOUNDED occupancy (max consecutive beta run strictly below L0 by >= D1_OCC_BOUND_FRAC) AND
    committed-action-class diversity rises strict-above L0 (entropy margin)."""
    l0_run = int(arm_l0.get("max_consecutive_beta_run", 0))
    l1_run = int(arm_l1.get("max_consecutive_beta_run", 0))
    if l0_run <= 0:
        return False
    bounded = bool(l1_run < l0_run and (l0_run - l1_run) >= D1_OCC_BOUND_FRAC * l0_run)
    diversity_lift = bool(
        float(arm_l1.get("committed_class_entropy", 0.0))
        > float(arm_l0.get("committed_class_entropy", 0.0)) + D1_ENTROPY_MARGIN
    )
    return bool(bounded and diversity_lift)


def _d2_release_content_driven(arm_l1: Dict[str, Any]) -> bool:
    """D2 (CORE): the L1 release is content-driven, NOT a re-parameterised timer. The ramp
    self-limited (rho_n_releases_total > 0) AND the per-episode release-tick dispersion is
    non-trivial (std > RELEASE_TICK_STD_FLOOR -- variable timing tracking proximity; a fixed
    timer would release at a constant tick = std ~ 0). The ramp releases ONLY at the
    proximity-decline crossing by construction, so a non-trivial release-tick spread certifies
    content-driven release."""
    return bool(
        int(arm_l1.get("rho_n_releases_total", 0)) > 0
        and float(arm_l1.get("rho_release_tick_std", 0.0)) > RELEASE_TICK_STD_FLOOR
    )


def _d3_habenula_dissociable(arm_l2: Dict[str, Any], delta_negative_met: bool) -> bool:
    """D3: the habenula de-commit is dissociable. L2 fired the content-driven habenula abort
    (n_habenula_aborts >= MIN_HABENULA_ABORTS) AND the negative-delta_t precondition held, so
    the aborts fired on outcome valence (content), dissociable from the latch refractory
    phase (clock)."""
    return bool(
        int(arm_l2.get("n_habenula_aborts", 0)) >= MIN_HABENULA_ABORTS
        and delta_negative_met
    )


def _rule_bias_mean(p1) -> float:
    diag = getattr(p1, "rule_bias_diag", None) or {}
    n = int(diag.get("n_bias_samples", 0))
    s = float(diag.get("sum_bias_abs_mean", 0.0))
    return s / n if n > 0 else 0.0


def _empty_arm() -> Dict[str, Any]:
    return {
        "n_closures": 0, "n_habenula_aborts": 0, "habenula_decommit_fired_count": 0,
        "sd034_n_closure_commit_intent": 0, "sd034_n_closure_coupled_elevations": 0,
        "n_hook_fires": 0, "n_automatic_fires": 0, "beta_release_events": 0,
        "nogo_installed_total": 0, "total_committed_steps": 0, "total_beta_elevated": 0,
        "mean_beta_elevated_steps": 0.0, "mean_per_commit_hold": 0.0,
        "max_consecutive_beta_run": 0, "ncl_hold_reassert_total": 0,
        "ncl_hold_closure_armed_total": 0, "n_sequence_completions": 0, "n_eval_episodes": 0,
        "closure_present": False, "env_hook_enabled": False, "n_window_events": 0,
        "mean_pre_closure_occ": 0.0, "mean_post_closure_occ": 0.0, "rho_present": False,
        "rho_n_releases_total": 0, "rho_peak_max": 0.0, "rho_mean_peak": 0.0,
        "rho_n_release_episodes": 0, "rho_release_tick_std": 0.0, "rho_release_tick_mean": 0.0,
        "n_neg_delta_ticks": 0, "n_delta_ticks": 0, "delta_t_min": 0.0, "delta_t_std": 0.0,
        "committed_class_entropy": 0.0, "n_committed_classes": 0,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "rule_bias_pathway_enabled": False, "rule_bias_mean_abs": 0.0,
        "rule_bias_n_train_steps": 0, "rule_bias_trained": False,
        "arms": {a["key"]: _empty_arm() for a in ARMS},
        "closure_eval_armed": False,
        "l0_monolithic": False,
        "rho_proximity_varied": False,
        "delta_negative_varied": False,
        "d1_ramp_self_limits": False,
        "d2_release_content_driven": False,
        "d3_habenula_dissociable": False,
        "pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else CLOSURE_EVAL_EPISODES

    probe_env = _build_closure_env(scaffold_cfg)
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    rule_bias_enabled = bool(getattr(p1, "rule_bias_pathway_enabled", False))
    rule_bias_mean = _rule_bias_mean(p1)
    rule_bias_steps = int((getattr(p1, "rule_bias_diag", None) or {}).get("n_train_steps", 0))
    rule_bias_trained = bool(rule_bias_enabled and rule_bias_mean > RULE_BIAS_MEAN_FLOOR)
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
          f" rule_bias_enabled={rule_bias_enabled} rule_bias_mean={rule_bias_mean:.4f}"
          f" rule_bias_steps={rule_bias_steps} rule_bias_trained={rule_bias_trained}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    # Eval all three arms on the SAME trained substrate (clone per arm; DRIVER pair toggled).
    arms_out: Dict[str, Any] = {}
    for arm in ARMS:
        closure_env = _build_closure_env(scaffold_cfg)
        closure_env.reset()
        print(f"Seed {seed} Condition {arm['key']}", flush=True)
        agent_arm = _clone_arm(agent, device, arm)
        agent_arm.e3._running_variance = float(agent.e3._running_variance)
        arms_out[arm["key"]] = _eval_arm_behaviour(
            agent_arm, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
        )
        done += eval_eps

    arm_l0 = arms_out[L0_ARM]
    arm_l1 = arms_out[L1_ARM]
    arm_l2 = arms_out[L2_ARM]

    # Per-seed readiness gates.
    closure_eval_armed = _closure_exclusive_eval_armed(arm_l0)
    l0_monolithic = _l0_monolithic(arm_l0)
    rho_proximity_varied = _rho_proximity_varied(arm_l1)
    delta_negative_varied = _delta_negative_varied(arm_l2)

    # Per-seed discriminators (scored only when all readiness gates clear at the cohort level;
    # recorded per seed here for the >= 2/3 tally).
    d1 = _d1_ramp_self_limits(arm_l0, arm_l1)
    d2 = _d2_release_content_driven(arm_l1)
    d3 = _d3_habenula_dissociable(arm_l2, delta_negative_varied)
    seed_pass = bool(d1 and d2 and d3)

    print(f"  [train] arm_eval seed={seed} ep {done}/{total_eps}"
          f" | L0 run={arm_l0['max_consecutive_beta_run']} hold={arm_l0['mean_per_commit_hold']:.2f}"
          f" ent={arm_l0['committed_class_entropy']:.3f} armed={arm_l0['ncl_hold_closure_armed_total']}"
          f" reassert={arm_l0['ncl_hold_reassert_total']}"
          f" | L1 run={arm_l1['max_consecutive_beta_run']} ent={arm_l1['committed_class_entropy']:.3f}"
          f" rho_rel={arm_l1['rho_n_releases_total']} rho_peak={arm_l1['rho_peak_max']:.4f}"
          f" rel_tick_std={arm_l1['rho_release_tick_std']:.2f}"
          f" | L2 hab_aborts={arm_l2['n_habenula_aborts']} hab_fired={arm_l2['habenula_decommit_fired_count']}"
          f" n_neg_dt={arm_l2['n_neg_delta_ticks']} dt_min={arm_l2['delta_t_min']:.3f}"
          f" | eval_armed={closure_eval_armed} l0_mono={l0_monolithic}"
          f" rho_var={rho_proximity_varied} dt_var={delta_negative_varied}"
          f" D1={d1} D2={d2} D3={d3}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} eval_armed={closure_eval_armed} l0_mono={l0_monolithic}"
          f" rho_var={rho_proximity_varied} dt_var={delta_negative_varied}"
          f" rule_bias_trained={rule_bias_trained} D1={d1} D2={d2} D3={d3}"
          f" (contact_rate={p2.contact_rate:.4f} z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})",
          flush=True)

    return {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "rule_bias_pathway_enabled": rule_bias_enabled,
        "rule_bias_mean_abs": float(rule_bias_mean),
        "rule_bias_n_train_steps": rule_bias_steps,
        "rule_bias_trained": rule_bias_trained,
        "arms": arms_out,
        "closure_eval_armed": closure_eval_armed,
        "l0_monolithic": l0_monolithic,
        "rho_proximity_varied": rho_proximity_varied,
        "delta_negative_varied": delta_negative_varied,
        "d1_ramp_self_limits": d1,
        "d2_release_content_driven": d2,
        "d3_habenula_dissociable": d3,
        "pass": seed_pass,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_arms = len(ARMS)
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + n_arms * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + n_arms * CLOSURE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # Readiness gate (2): rule_bias_head trained.
    rb_flags = [bool(r.get("rule_bias_trained", False)) for r in guard_passing]
    rb_frac = _frac(rb_flags)
    rule_bias_trained_met = bool(rb_frac >= MIN_FRACTION)

    # Readiness gate (3): closure-exclusive eval armed the hold (the dissociable eval mode).
    ce_flags = [bool(r.get("closure_eval_armed", False)) for r in guard_passing]
    ce_frac = _frac(ce_flags)
    closure_eval_armed_met = bool(ce_frac >= MIN_FRACTION)

    # Readiness gate (4): L0 monolithic-hold baseline (the reference the ramp self-limits vs).
    l0_flags = [bool(r.get("l0_monolithic", False)) for r in guard_passing]
    l0_frac = _frac(l0_flags)
    l0_monolithic_met = bool(l0_frac >= MIN_FRACTION)

    # Readiness gate (5): rho proximity variance (the ramp had proximity to ramp on).
    rho_flags = [bool(r.get("rho_proximity_varied", False)) for r in guard_passing]
    rho_frac = _frac(rho_flags)
    rho_proximity_met = bool(rho_frac >= MIN_FRACTION)

    # Readiness gate (6): delta_t negative variance (the habenula had outcomes to abort on).
    dt_flags = [bool(r.get("delta_negative_varied", False)) for r in guard_passing]
    dt_frac = _frac(dt_flags)
    delta_negative_met = bool(dt_frac >= MIN_FRACTION)

    # Discriminators (the scored DVs).
    d1_flags = [bool(r.get("d1_ramp_self_limits", False)) for r in guard_passing]
    d1_frac = _frac(d1_flags)
    d1_met = bool(d1_frac >= MIN_FRACTION)
    d2_flags = [bool(r.get("d2_release_content_driven", False)) for r in guard_passing]
    d2_frac = _frac(d2_flags)
    d2_met = bool(d2_frac >= MIN_FRACTION)
    d3_flags = [bool(r.get("d3_habenula_dissociable", False)) for r in guard_passing]
    d3_frac = _frac(d3_flags)
    d3_met = bool(d3_frac >= MIN_FRACTION)

    overall_criteria_pass = bool(d1_met and d2_met and d3_met)

    readiness_all_met = bool(
        contact_non_vacuity_met and rule_bias_trained_met and closure_eval_armed_met
        and l0_monolithic_met and rho_proximity_met and delta_negative_met
    )

    # Routing. Negative outcome is a ROUTE, not a falsification (control-plane readiness
    # falsifier). Readiness-gate failures -> substrate_not_ready_requeue; a preconditions-met
    # no-lift on D1 -> maintenance_not_binding_route_to_selection (route to JOB-1, NEVER a
    # false weakening of the ramp+habenula design).
    if not contact_non_vacuity_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "contact_guard_unmet"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not rule_bias_trained_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "rule_bias_head_untrained"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not closure_eval_armed_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "closure_exclusive_eval_did_not_arm_hold"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not l0_monolithic_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "l0_baseline_not_monolithic"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not rho_proximity_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "rho_no_proximity_variance"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not delta_negative_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "delta_t_no_negative_variance"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        # All six readiness gates clear -> D1/D2/D3 are interpretable.
        if overall_criteria_pass:
            outcome = "PASS"
            readiness_route = "dopaminergic_driver_pair_self_limits_and_dissociates"
            route_reason = "d1_ramp_selflimit_and_d2_content_release_and_d3_habenula_dissociable_met"
            direction_map = {cid: "supports" for cid in CLAIM_IDS}
            overall_direction = "supports"
        elif not d1_met:
            # The ramp varies + the habenula has negative variance + the eval armed, but the
            # hold still monopolises / behaviour does not lift -> maintenance is NOT the
            # binding constraint -> route to JOB-1 / selection. NEVER a false weakening.
            outcome = "FAIL"
            readiness_route = "maintenance_not_binding_route_to_selection"
            route_reason = "preconditions_met_no_occupancy_bounding_or_diversity_lift_route_job1"
            direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
            overall_direction = "non_contributory"
        else:
            # D1 holds (the ramp self-limits + lifts diversity) but a HOW discriminator is
            # unmet: D2 (content-driven release) and/or D3 (habenula dissociable). Partial
            # support; still NOT a falsification of the driver pair.
            outcome = "FAIL"
            readiness_route = "ramp_self_limits_but_how_discriminator_unmet"
            route_reason = (
                "d1_met_but_"
                + ("d2_unmet_" if not d2_met else "")
                + ("d3_unmet" if not d3_met else "")
            ).rstrip("_")
            # MECH-090 supported by D1 (latch release); the rest split on D2/D3.
            direction_map = {
                "MECH-090": "supports",
                "MECH-342": "supports" if d2_met else "non_contributory",
                "MECH-445": "supports" if d3_met else "non_contributory",
                "MECH-446": "supports" if d3_met else "non_contributory",
                "ARC-108": "mixed",
            }
            overall_direction = "mixed"

    print(f"[{EXPERIMENT_TYPE}] contact={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) rule_bias_trained={rule_bias_trained_met} (frac={rb_frac:.3f})"
          f" closure_eval_armed={closure_eval_armed_met} (frac={ce_frac:.3f})"
          f" l0_monolithic={l0_monolithic_met} (frac={l0_frac:.3f})"
          f" rho_proximity={rho_proximity_met} (frac={rho_frac:.3f})"
          f" delta_negative={delta_negative_met} (frac={dt_frac:.3f})"
          f" | D1={d1_met}({d1_frac:.3f}) D2={d2_met}({d2_frac:.3f}) D3={d3_met}({d3_frac:.3f})"
          f" criteria_pass={overall_criteria_pass}"
          f" -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "rule_bias_trained_met": rule_bias_trained_met,
        "rule_bias_trained_fraction": rb_frac,
        "closure_eval_armed_met": closure_eval_armed_met,
        "closure_eval_armed_fraction": ce_frac,
        "l0_monolithic_met": l0_monolithic_met,
        "l0_monolithic_fraction": l0_frac,
        "rho_proximity_varied_met": rho_proximity_met,
        "rho_proximity_varied_fraction": rho_frac,
        "delta_negative_varied_met": delta_negative_met,
        "delta_negative_varied_fraction": dt_frac,
        "d1_ramp_self_limits_met": d1_met,
        "d1_fraction": d1_frac,
        "d2_release_content_driven_met": d2_met,
        "d2_fraction": d2_frac,
        "d3_habenula_dissociable_met": d3_met,
        "d3_fraction": d3_frac,
        "overall_pass": bool(readiness_all_met and overall_criteria_pass),
        "per_seed_guard_pass": guard_flags,
        "per_seed_d1": [bool(r.get("d1_ramp_self_limits", False)) for r in per_seed],
        "per_seed_d2": [bool(r.get("d2_release_content_driven", False)) for r in per_seed],
        "per_seed_d3": [bool(r.get("d3_habenula_dissociable", False)) for r in per_seed],
        "route_reason": route_reason,
    }

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": [
                {
                    "name": "foraging_contact_guard",
                    "description": "603n G2+G3: per-seed P2 contact_rate > 0 AND "
                                   "z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.",
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                },
                {
                    "name": "rule_bias_head_trained",
                    "description": "Leg C readiness (anti-460d-bug gate): P1 "
                                   "rule_bias_pathway_enabled AND mean per-candidate |bias| > "
                                   "floor on >= 2/3 seeds (closures need a magnitude-bearing "
                                   "rule_state to form). Below floor -> substrate_not_ready_"
                                   "requeue, NEVER a weakens.",
                    "control": "P1OnboardingResult.rule_bias_diag mean |bias|.",
                    "measured": rb_frac,
                    "threshold": MIN_FRACTION,
                    "met": rule_bias_trained_met,
                },
                {
                    "name": "closure_exclusive_eval_armed_hold",
                    "description": "The closure-exclusive de-commit eval mode "
                                   "(closure_exclusive_decommit_eval, BUILT ree-v3 e52158d) "
                                   "ACTUALLY armed the closure-coupled natural-commit latch-hold "
                                   "on L0 -- ncl_hold_closure_armed_total > 0 AND "
                                   "ncl_hold_reassert_total > 0 on >= 2/3 guard seeds (the "
                                   "'dissociable eval mode arms' non-vacuity). Below this -> "
                                   "substrate_not_ready_requeue (the dissociable occupancy did "
                                   "not form), NEVER a false weakening.",
                    "control": "L0 ncl_hold_closure_armed_total + ncl_hold_reassert_total under "
                               "closure_exclusive_decommit_eval.",
                    "measured": ce_frac,
                    "threshold": MIN_FRACTION,
                    "met": closure_eval_armed_met,
                },
                {
                    "name": "l0_monolithic_hold_baseline",
                    "description": "L0 (flat latch, rho + habenula OFF) reproduces the flat-hold "
                                   "monopoly -- mean per-commit hold length "
                                   "total_beta_elevated/max(1,beta_release_events) >= "
                                   "SUSTAINED_HOLD_MEAN_FLOOR -- on >= 2/3 guard seeds, the "
                                   "reference the rho ramp self-limits AGAINST (the 460h B6 "
                                   "signature). If L0 does not monopolise there is nothing to "
                                   "self-limit -> substrate_not_ready_requeue.",
                    "control": "L0 mean_per_commit_hold (+ max_consecutive_beta_run reported).",
                    "measured": l0_frac,
                    "threshold": MIN_FRACTION,
                    "met": l0_monolithic_met,
                },
                {
                    "name": "rho_proximity_variance",
                    "description": "JOB-2 ramp non-vacuity: on L1 the rho_t ramp reached a "
                                   "non-trivial proximity peak (rho_peak_max > RHO_PEAK_FLOOR) "
                                   "AND self-limited at least once (rho_n_releases_total > 0) on "
                                   ">= 2/3 guard seeds. If rho_t carries no proximity variance "
                                   "the ramp cannot peak-then-decline -> substrate_not_ready_"
                                   "requeue (NEVER a false weakens).",
                    "control": "L1 rho_maintenance_ramp.get_state() rho_peak_max + "
                               "rho_n_releases_total.",
                    "measured": rho_frac,
                    "threshold": MIN_FRACTION,
                    "met": rho_proximity_met,
                },
                {
                    "name": "delta_t_negative_variance",
                    "description": "JOB-2 habenula non-vacuity: on L2 the signed RPE "
                                   "delta_t = R_t - V-hat_t carried worse-than-expected variance "
                                   "to abort on -- n_neg_delta_ticks >= MIN_NEG_DELTA_TICKS on "
                                   ">= 2/3 guard seeds. If delta_t never goes negative the "
                                   "habenula has nothing to fire on -> substrate_not_ready_"
                                   "requeue (NEVER a false weakens).",
                    "control": "L2 e3_habenula_delta_t distribution (n_neg_delta_ticks + "
                               "delta_t_min + delta_t_std).",
                    "measured": dt_frac,
                    "threshold": MIN_FRACTION,
                    "met": delta_negative_met,
                },
            ],
            "criteria": [
                {"name": "D1_ramp_self_limits_where_flat_latch_monopolises",
                 "load_bearing": True, "passed": d1_met},
                {"name": "D2_release_content_driven_not_reparameterised_timer",
                 "load_bearing": True, "passed": d2_met},
                {"name": "D3_habenula_decommit_dissociable",
                 "load_bearing": True, "passed": d3_met},
            ],
            "criteria_non_degenerate": {
                # The D1/D2/D3 DVs are non-degenerate iff all six readiness gates cleared
                # (contact, trained head, closure-exclusive eval armed, L0 monolithic, rho
                # proximity variance, delta_t negative variance) -- otherwise the
                # occupancy/release/abort readouts are structurally uninterpretable.
                "D1_ramp_self_limits_where_flat_latch_monopolises": readiness_all_met,
                "D2_release_content_driven_not_reparameterised_timer": readiness_all_met,
                "D3_habenula_decommit_dissociable": readiness_all_met,
            },
            "discriminators": {
                "D1": {
                    "definition": "Per guard seed: L1 max_consecutive_beta_run < L0 by >= "
                                  "D1_OCC_BOUND_FRAC (bounded occupancy, no single hold "
                                  "monopolising) AND committed_class_entropy(L1) > "
                                  "committed_class_entropy(L0) + D1_ENTROPY_MARGIN "
                                  "(committed-action-class diversity rises). PASS on >= 2/3.",
                    "d1_occ_bound_frac": D1_OCC_BOUND_FRAC,
                    "d1_entropy_margin": D1_ENTROPY_MARGIN,
                    "fraction": d1_frac,
                    "met": d1_met,
                },
                "D2": {
                    "definition": "Per guard seed: L1 rho_n_releases_total > 0 AND per-episode "
                                  "release-tick dispersion std(rho_last_ticks_at_release) > "
                                  "RELEASE_TICK_STD_FLOOR (content-driven variable timing; a "
                                  "fixed shorter-duration timer releases at a CONSTANT tick = "
                                  "std ~ 0). The ramp releases ONLY at the proximity-decline "
                                  "crossing by construction. PASS on >= 2/3. THE CORE "
                                  "discriminator: 'releases sooner' is NOT the claim.",
                    "release_tick_std_floor": RELEASE_TICK_STD_FLOOR,
                    "fraction": d2_frac,
                    "met": d2_met,
                },
                "D3": {
                    "definition": "Per guard seed: L2 n_habenula_aborts >= MIN_HABENULA_ABORTS "
                                  "AND the negative-delta_t precondition held -> the aborts "
                                  "fired on outcome valence (content), dissociable from the "
                                  "latch refractory phase (clock). PASS on >= 2/3.",
                    "min_habenula_aborts": MIN_HABENULA_ABORTS,
                    "fraction": d3_frac,
                    "met": d3_met,
                },
            },
            "driver_pair_under_test": {
                "rho_maintenance_ramp": "ree_core/policy/rho_maintenance_ramp.py "
                    "RhoMaintenanceRamp (use_rho_maintenance_ramp; L1/L2). REPLACES the "
                    "flat-hold maintenance driver with a proximity-scaled peaks-then-declines "
                    "self-limit (the B6 fix).",
                "habenula_decommit": "ree_core/governance/closure_operator.py "
                    "ClosureOperator.habenula_tick (use_habenula_decommit; L2). ADDS a "
                    "content-driven negative-delta_t abort to the SD-034 closure operator "
                    "(the dissociable de-commit; reuses the JOB-1 signed-RPE delta_t).",
                "rho_hold_floor": RHO_HOLD_FLOOR,
                "rho_release_margin": RHO_RELEASE_MARGIN,
                "rho_onset_grace_ticks": RHO_ONSET_GRACE_TICKS,
                "habenula_decommit_delta_threshold": HABENULA_DECOMMIT_DELTA_THRESHOLD,
                "note": "closure_exclusive_decommit_eval + use_closure_commit_beta_coupling + "
                        "use_natural_commit_latch_hold ON in EVERY arm; the rung-6 "
                        "NaturalCommitUrgencyRelease is OFF in EVERY arm (the rho ramp is the "
                        "maintenance driver under test, NOT the hand-tuned duration lever).",
            },
            "route_to_selection_note": "A preconditions-met no-lift (the ramp varies + the "
                "habenula has negative variance + the eval armed, but the hold still "
                "monopolises / committed-action diversity does not lift on D1) is the genuine "
                "'maintenance is not the binding constraint' outcome -> route to JOB-1 / "
                "selection (maintenance_not_binding_route_to_selection, non_contributory), "
                "NEVER a false weakening of the rho-ramp + habenula design.",
            "mech261_note": "MECH-261 (mode-conditioning) NOT tagged -- the Leg-A "
                            "env-completion hook bypasses mode-conditioning (n_automatic_fires "
                            "reported as a diagnostic). MECH-260 (No-Go) NOT re-tagged "
                            "(nogo_installed reported only).",
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> "
                     "P0 -> Stage-H -> P1 -> P2; 603n config) + commitment control-plane "
                     "(bistable BetaGate + SD-034 ClosureOperator + SD-033a LateralPFC + "
                     "SD-032 dACC/salience) + subgoal_mode waypoint tolerance-band completion "
                     "+ commitment-closure-control-plane Legs A/B/C + beta-engagement coupling "
                     "+ the natural-commit LATCH-HOLD + the CLOSURE-EXCLUSIVE DE-COMMIT EVAL "
                     "mode (closure_exclusive_decommit_eval, ree-v3 e52158d) ARMED in ALL arms "
                     "+ the ARC-108 JOB-2 DRIVER pair (rho_t maintenance ramp + habenula "
                     "negative-delta_t de-commit, ree-v3 c5614ab) toggled per arm "
                     "(L0/L1/L2). rung-6 NaturalCommitUrgencyRelease OFF in every arm.",
        "condition": CONDITION_LABEL,
        "method_note": "ARC-108 JOB-2 control-plane L0/L1/L2 falsifier (unified_dopamine_"
                       "substrate_design_2026-06-22.md sec 7.2). SUPERSEDES V3-EXQ-460k (the "
                       "rung-6 duration-lever falsifier): the JOB-2 DRIVER pair is the "
                       "biologically-faithful successor -- the rho ramp REPLACES the flat-hold "
                       "maintenance driver (B6 fix), the habenula ADDS a content-driven "
                       "dissociable de-commit (the rung-6 fix done properly). DISTINCT from "
                       "V3-EXQ-700 (the JOB-1 sec-7.1 SELECTION 2x2). Three eval arms on one "
                       "trained substrate per seed (DRIVER pair toggled at eval via clone): "
                       "L0_FLAT_LATCH (monolithic-hold reference) / L1_RHO_RAMP (the ramp "
                       "self-limits the hold) / L2_RHO_HABENULA (+ content-driven dissociable "
                       "de-commit). The eval loop CALLS agent.update_residue() each tick so the "
                       "waking post-action path computes delta_t and fires the habenula. SIX "
                       "readiness gates self-route substrate_not_ready_requeue when unmet "
                       "(never a false weakens). D1 (ramp self-limits where the flat latch "
                       "monopolises) / D2 (release content-driven NOT a re-parameterised timer, "
                       "the CORE discriminator) / D3 (habenula de-commit dissociable). A "
                       "preconditions-met no-lift routes to JOB-1 / selection "
                       "(maintenance_not_binding_route_to_selection), NEVER a falsification. "
                       "claim_ids: MECH-090/MECH-342/MECH-445/MECH-446 + ARC-108.",
        "arm_note": "ARMS = " + ", ".join(a["key"] for a in ARMS) + ". closure-exclusive eval "
                    "+ beta-engagement coupling + natural-commit latch-hold ON in every arm; "
                    "the variable is the JOB-2 DRIVER pair (rho ramp x habenula). L0 is the "
                    "monolithic-hold reference; L1 adds the rho ramp (D1/D2); L2 adds the "
                    "habenula de-commit (D3).",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "rule_bias_mean_floor": RULE_BIAS_MEAN_FLOOR,
            "sustained_hold_mean_floor": SUSTAINED_HOLD_MEAN_FLOOR,
            "rho_peak_floor": RHO_PEAK_FLOOR,
            "min_neg_delta_ticks": MIN_NEG_DELTA_TICKS,
            "d1_occ_bound_frac": D1_OCC_BOUND_FRAC,
            "d1_entropy_margin": D1_ENTROPY_MARGIN,
            "release_tick_std_floor": RELEASE_TICK_STD_FLOOR,
            "min_habenula_aborts": MIN_HABENULA_ABORTS,
            "closure_window": CLOSURE_WINDOW,
            "window_min_ticks": WINDOW_MIN_TICKS,
            "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
            "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
            "use_natural_commit_latch_hold": True,
            "closure_exclusive_decommit_eval": True,
            "use_closure_commit_beta_coupling": True,
            "use_natural_commit_urgency_release": False,
            "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
            "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
            "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
            "rho_hold_floor": RHO_HOLD_FLOOR,
            "rho_release_margin": RHO_RELEASE_MARGIN,
            "rho_onset_grace_ticks": RHO_ONSET_GRACE_TICKS,
            "habenula_decommit_delta_threshold": HABENULA_DECOMMIT_DELTA_THRESHOLD,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "closure_eval_episodes_per_arm": CLOSURE_EVAL_EPISODES,
            "n_eval_arms": len(ARMS),
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "scaffold_train_rule_bias_head": True,
            "config_basis": "V3-EXQ-460k substrate (closure-exclusive de-commit eval, ree-v3 "
                            "main e52158d, 2026-06-22) + ARC-108 JOB-2 DRIVER pair (rho ramp + "
                            "habenula de-commit, ree-v3 main c5614ab, 2026-06-22)",
        },
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
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
        manifest_path=_res.get("manifest_path"),
        dry_run=args.dry_run,
    )
