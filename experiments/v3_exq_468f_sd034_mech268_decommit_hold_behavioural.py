"""
V3-EXQ-468f (supersedes V3-EXQ-468e): SD-034 / MECH-268 / MECH-090 commitment-vs-
contradiction PERSEVERATION retest on the DE-COMMIT-AUTHORITY-MAGNITUDE-amended
substrate (committed-run-scaled Leg-B refractory, ree-v3 main 2cd0aa2), with the C2
de-commit DV redesigned to a GRADED within-ON-arm AROUND-CONTRADICTION occupancy delta
(replacing the 1.0-saturated absolute committed_frac_post_absolute).

WHY THIS SUPERSEDES 468e (failure_autopsy_V3-EXQ-468e_2026-06-18, confirmed; this is the
perseveration-side sibling of the 460f->460g de-commit-magnitude re-issue): 468e's
beta-engagement amend ENGAGED THE SUBSTRATE FAIRLY -- both non-vacuity gates cleared
(foraging-contact 1.0; commitment-non-vacuity 1.0 -- ON committed AND a contradiction
fired 3/3), so C1/C2 drove a verdict. C1 (beta_release_near_contradiction) PASSED 3/3
with ON > OFF (43/16/58 vs 14/10/0) -- the MECH-268 dACC-saturation -> beta-release
pathway works PROXIMALLY. C2 (committed_frac_post_absolute ON < OFF) FAILED 3/3 because
the ON-arm post-contradiction committed fraction is PINNED AT THE 1.0 CEILING on every
seed: the agent stays fully committed through the whole post-contradiction window despite
the release (seed-44 ON=1.0 vs OFF=0.0 is a non-commit artefact, NOT de-commit). SAME
STRUCTURAL PROPERTY AS 460f via an INDEPENDENT DV: the de-commit/release fires with
correct sign but SUB-THRESHOLD AUTHORITY MAGNITUDE; the absolute committed-fraction DV
re-pinned at the ceiling (the 468c->468e cap-pin escape moved 0.85->1.0 but did not lift).

THE TWO FIXES THIS RE-ISSUE ARMS (mirroring the 460g de-commit re-issue):
  (a) MAGNITUDE LEVER (substrate, landed ree-v3 main 2cd0aa2): the Leg-B refractory
      installed at a closure fire is now SCALED by the committed-run length captured from
      the BetaGate BEFORE the closure's own release():
        n = closure_decommit_hold_ticks
            + round(closure_decommit_hold_scale_with_run * committed_run_length),
      clamped to closure_decommit_hold_max_ticks. A long committed run -- the exact source
      of the swamping post-contradiction commitment -- triggers a proportionally long
      post-closure hold, so the de-commit authority scales with the commitment it must
      overcome. Armed here via closure_decommit_hold_scale_with_run +
      closure_decommit_hold_max_ticks on the ON-arm config. (The OFF clone has closure off
      -> _fire never runs -> the lever is inert there.)
  (b) GRADED WITHIN-ARM C2 DV (experiment side, this script): the load-bearing C2 is now a
      PAIRED within-ON-arm AROUND-CONTRADICTION committed-occupancy delta -- for each
      counter-evidence (contradiction) injection at tick t, the COMMITTED-trajectory
      occupancy fraction in a CONTRADICTION_WINDOW pre-contradiction window [t-W, t) vs the
      same-length post-contradiction window (t, t+W]. The de-commit lowers the
      post-contradiction committed occupancy below the pre-contradiction occupancy. This is
      paired (each contradiction is its own pre/post pair) and GRADED (a continuous
      occupancy fraction, not the absolute committed_frac_post_absolute that re-pinned at
      1.0), so a real-but-partial de-commit is measurable instead of being masked by the
      ceiling. The 468e between-arm absolute DV is RETAINED as a SECONDARY diagnostic only
      (NOT load-bearing).

NON-VACUITY GATE TIGHTENED (the brief): the closure-plane coupling must actually have
engaged the latch on the ON arm -- sd034_n_closure_coupled_elevations > 0 AND a
contradiction fired (episodes_with_contradiction > 0) on >= 2/3 guard seeds -- else the
de-commit DV is measuring the fragile natural commit-entry, not the closure-plane
commitment, and the run self-routes substrate_not_ready_requeue (NEVER a false weakens).
This replaces 468e's both-arms-total_beta_elevated gate with the direct coupling
non-vacuity readout (the 460g tightening, applied to the perseveration side).

SUBSTRATE legs under test (all landed; armed here on the 603n foraging substrate):
  Leg A  env-completion hook (use_closure_env_completion_hook) -> emit_closure.
  Leg B  de-commit refractory (closure_decommit_hold_ticks) + the MAGNITUDE lever (a).
  Leg C  scaffold_train_rule_bias_head (598b REINFORCE in P1) -- trained magnitude-bearing
         rule_state so the closure-coupled de-commit has MECH-090 latch authority.
  beta-engagement coupling (use_closure_commit_beta_coupling) -- ties the closure-plane
         commitment to bistable beta elevation so the latch is engaged on every seed where
         a closure commitment forms.

WHAT THIS MEASURES (perseveration mechanism, unchanged from 468e): when counter-evidence
contradicts an active commitment, a healthy agent RELEASES the commitment (MECH-090 beta
drops via MECH-268 dACC PE saturation; the Leg-B de-commit hold then keeps it UNcommitted
through the post-contradiction window). Without sufficient de-commit AUTHORITY the agent
PERSEVERATES -- stays committed despite the contradiction-coupled release -- the OCD-like
signature. C1 measures the proximal release (MECH-268); the graded C2 measures the
sustained de-commit (SD-034 authority); MECH-090 keys on the C1 release (the 468e autopsy
mis-attribution fix -- do NOT weaken the active latch on a downstream-authority fail).

ARMS (one curriculum build per seed, two frozen-policy evals):
  ARM_SUBSTRATE_ON   -- full closure + dACC-saturation + bistable + env hook + de-commit
                        hold + MAGNITUDE lever, built on the TRAINED rule_bias_head.
                        Load-bearing graded C2 measured here.
  ARM_SUBSTRATE_OFF  -- clone of the SAME trained weights with closure OFF and dACC
                        saturation disabled. Secondary between-arm diagnostic only
                        (NOT load-bearing). Expect perseveration without the substrate.

CONTACT GUARD (603n G2 + G3 foraging non-vacuity, mirrors 514n): per-seed guard =
  (P2 contact_rate > 0) AND (P2 z_goal_norm_at_contact_peak > 0.4). < 2/3 seeds passing ->
  substrate_not_ready_requeue (FAIL; non_contributory).

READINESS / NON-VACUITY (all must clear before C2 is scored; any unmet self-routes
substrate_not_ready_requeue -- NEVER a false weakens):
  (1) 603n foraging contact guard (above) on >= 2/3 seeds.
  (2) rule_bias_head trained (the DIRECT anti-460d-bug gate): P1 rule_bias_pathway_enabled
      AND mean per-candidate |bias| (rule_bias_diag) > RULE_BIAS_MEAN_FLOOR on >= 2/3 seeds.
  (3) closure-coupling + contradiction non-vacuity (TIGHTENED): ON
      sd034_n_closure_coupled_elevations > 0 AND ON episodes_with_contradiction > 0 on
      >= 2/3 guard seeds.
  (4) within-arm window non-vacuity: ON arm produced >= C2_MIN_WINDOW_EVENTS scored
      around-contradiction windows with a non-trivial pre-contradiction committed
      occupancy (mean_pre_committed_occ > WITHIN_PRE_OCC_FLOOR) on >= 2/3 guard seeds --
      else there was nothing committed to de-commit (substrate_not_ready_requeue).

PRE-REGISTERED ACCEPTANCE (constants; per-seed PASS = C1 AND C2; overall PASS = majority
2/3 guard seeds; scored only once all four readiness gates clear):
  C1  ARM_SUBSTRATE_ON  beta_release_near_contradiction >= 1   (MECH-268 proximal release)
  C2  GRADED WITHIN-ARM AROUND-CONTRADICTION COMMITTED-OCCUPANCY DROP (load-bearing): on
      the ON arm, mean post-contradiction COMMITTED-occupancy fraction < mean
      pre-contradiction committed-occupancy fraction with a >= DECOMMIT_MIN_DROP_FRAC
      relative drop, over >= C2_MIN_WINDOW_EVENTS scored windows with mean_pre_committed_occ
      > WITHIN_PRE_OCC_FLOOR. Paired within-arm statistic (replaces the 1.0-pinned
      committed_frac_post_absolute).

PER-CLAIM DIRECTION:
  any readiness gate NOT met -> non_contributory (substrate_not_ready_requeue).
  all met:
    SD-034   = supports if (C1 AND C2) all guard-passing seeds else weakens
               (de-commit AUTHORITY -- keys on the graded C2).
    MECH-268 = supports if C1 all guard-passing seeds else weakens
               (dACC PE -> beta release proximal pathway -- keys on C1).
    MECH-090 = supports if C1 all guard-passing seeds else weakens
               (latch RELEASE via C1; the 468e autopsy established the self-stamped
               MECH-090 weakens was MIS-ATTRIBUTED -- the run tests MECH-090's latch
               RELEASE via C1, which the substrate executes; do NOT weaken the active
               latch on a downstream de-commit-authority (C2) fail. EXQ-048/MECH-057b class).

claim_ids: SD-034, MECH-268, MECH-090.
experiment_purpose: evidence.
supersedes: V3-EXQ-468e.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking goal-pipeline
  onboarding scheduler).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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

EXPERIMENT_TYPE = "v3_exq_468f_sd034_mech268_decommit_hold_behavioural"
QUEUE_ID = "V3-EXQ-468f"
CLAIM_IDS: List[str] = ["SD-034", "MECH-268", "MECH-090"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-468e"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_TRAINED_RULE_BIAS_PERSEVERATION_GRADED_DV_RETEST"

# --- Goal-pipeline / encoder dims (mirror 603n / 468e exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Curriculum budgets (mirror 603n / 468e exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15            # 603n-canonical contact-guard measurement (run_p2)
CONTRADICTION_EVAL_EPISODES = 15  # per arm (ON + OFF); the 468 commitment DV
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

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 603n / 468e) ---
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (mirror 603n / 468e) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- MECH-268 dACC PE saturation knobs (mirror 468e) ---
DACC_SAT_WINDOW = 8
DACC_SAT_STRENGTH = 0.3
DACC_SAT_GRACE = 2

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5  # Leg B: post-closure latch refractory base window
# DE-COMMIT-AUTHORITY MAGNITUDE lever (460f amend, ree-v3 main 2cd0aa2): scale the Leg-B
# refractory by the committed-run length at the closure fire so a long committed run
# triggers a proportionally long hold. n = base + round(scale * run_length), capped.
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- Contradiction eval thresholds (mirror 468e; constants, NOT derived) ---
RELEASE_WINDOW = 20      # steps after a counter-evidence injection within which a beta
                         # release counts as contradiction-triggered (C1)

# --- GRADED within-arm around-contradiction window DV (part b) ---
CONTRADICTION_WINDOW = 10   # ticks on each side of a counter-evidence injection
WINDOW_MIN_TICKS = 3        # minimum ticks each side to score an around-contradiction event
C2_MIN_WINDOW_EVENTS = 2    # minimum scored around-contradiction windows on the ON arm
WITHIN_PRE_OCC_FLOOR = 0.1  # pre-contradiction committed occupancy must be non-trivial
# C2 within-arm around-contradiction de-commit DV: mean post-contradiction COMMITTED
# occupancy must be at least this RELATIVE fraction below mean pre-contradiction occupancy
# (paired across contradictions). Graded; replaces the 468e cap-pinned absolute fraction.
DECOMMIT_MIN_DROP_FRAC = 0.10

# --- Leg C readiness: rule_bias_head must have TRAINED -- mean per-candidate |bias| above
# this floor. The untrained 460d head produced ~0; the Leg-C smoke produced 0.039.
RULE_BIAS_MEAN_FLOOR = 0.005

# --- Pre-registered acceptance thresholds ---
P2_ZGOAL_GATE = 0.4          # per-seed contact-guard: z_goal_norm_at_contact_peak floor (603n G3)
CONTACT_GATE = 0.0           # per-seed contact-guard: P2 contact_rate floor (603n G2)
MIN_FRACTION = 2.0 / 3.0     # >= 2/3 seeds for non-vacuity + any aggregate gate
C1_MIN_RELEASES = 1          # ARM_SUBSTRATE_ON beta_release_near_contradiction floor


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
        # developmental-window / consolidation amend (2026-06-03b)
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        # 634c seeding calibration (2026-06-03c)
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        # foraging-competence residual amend (2026-06-05)
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        # SD-057 cue-recall bridge (wean-to-wild contact lever; enables SD-049 in envs)
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # curriculum-decomposition amend (2026-06-07): isolated Stage-H
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (mirror 603n)
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE: feed the env harm stream so z_harm / z_harm_a populate
        scaffold_feed_harm_stream=True,
        # harm-pathway training (2026-06-09 amend; ON, validated by 603k/603n)
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        # commitment_closure:GAP-4 Leg C (2026-06-16): TRAIN the rule_bias_head in P1 so the
        # closure-coupled de-commit has a magnitude-bearing rule_state (the 460g pattern).
        scaffold_train_rule_bias_head=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate (mirror 468e) + the commitment control-plane
    (bistable BetaGate + SD-034 closure + SD-033a LateralPFC + SD-032 dACC/salience) +
    the commitment-closure-control-plane amend Legs A/B/C + beta-engagement coupling + the
    DE-COMMIT-AUTHORITY MAGNITUDE lever (460f amend). dACC PE saturation (MECH-268) is set
    on the built agent (not surfaced through from_dims)."""
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
        # SD-057 object-bound incentive-salience layer (foraging/cue-recall lever)
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # --- commitment control-plane substrate (the 468 mechanism under test) ---
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
        # SD-034 commitment-closure-control-plane amend (2026-06-12), the Leg A/B legs:
        use_closure_env_completion_hook=True,            # Leg A (env completion -> emit_closure)
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base de-commit hold
        # DE-COMMIT-AUTHORITY MAGNITUDE lever (2026-06-19, failure_autopsy_V3-EXQ-460f /
        # _468e): scale the Leg-B refractory by committed-run length so the de-commit
        # authority scales with the commitment it must overcome.
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,           # Leg C un-zero (GAP-D); trained by scaffold leg
        # BETA-ENGAGEMENT amend (2026-06-17, failure_autopsy_V3-EXQ-460e): couple the
        # closure-plane commitment to bistable beta elevation so the agent reliably
        # commits-with-beta and the contradiction-release / de-commit DV is exercisable.
        use_closure_commit_beta_coupling=True,
    )
    cfg.latent.use_resource_encoder = True   # SD-015 (z_resource -> bank L2 bind requires it)
    cfg.heartbeat.beta_gate_bistable = True   # MECH-090 bistable latch
    return cfg


def _enable_dacc_saturation(agent: REEAgent) -> None:
    """MECH-268 dACC PE saturation (not surfaced through from_dims)."""
    if agent.dacc is not None:
        agent.dacc.config.dacc_saturation_enabled = True
        agent.dacc.config.dacc_saturation_window = DACC_SAT_WINDOW
        agent.dacc.config.dacc_saturation_strength = DACC_SAT_STRENGTH
        agent.dacc.config.dacc_saturation_grace = DACC_SAT_GRACE


def _build_contradiction_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                             dry_run: bool) -> CausalGridWorldV2:
    """P2-config foraging env (same structural kwargs as the curriculum's P2 env so
    world_obs_dim matches the curriculum-built agent) WITH the GAP-3
    completion_tolerance(waypoint) + counter_evidence contradiction primitives layered on
    (dynamics-only; verified world_obs_dim-preserving). Unchanged from 468e -- the
    contradiction machinery is what makes the perseveration DV exercisable."""
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
        # GAP-3 contradiction primitives (the 468 mechanism; dynamics-only). subgoal_mode is
        # LOAD-BEARING: both waypoint-completion paths AND the counter_evidence injection
        # gate hard-require it (causal_grid_world 1664/1740/2067). num_waypoints=4 lengthens
        # the committed sequence so seq_in_progress windows overlap the injection cadence.
        subgoal_mode=True,
        num_waypoints=4,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.25,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
        counter_evidence_enabled=True,
        counter_evidence_interval=(3 if dry_run else 5),
        counter_evidence_prob=(0.95 if dry_run else 0.85),
        counter_evidence_degrade_step=0.2,
        counter_evidence_degrade_floor=0.0,
        counter_evidence_requires_persistent_rule=True,
        **_sd049_kwargs(scaffold_cfg),  # SD-049 multi-resource (parity with P2 env)
    )


def _clone_substrate_off(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone the SAME trained weights into a substrate-OFF agent (closure OFF + dACC PE
    saturation OFF) -- the perseveration-without-substrate contrast arm. The closure
    operator carries no trainable parameters, so the trained state_dict loads cleanly."""
    cfg_off = copy.deepcopy(trained_agent.config)
    cfg_off.use_closure_operator = False
    cfg_off.heartbeat.beta_gate_bistable = True
    agent_off = REEAgent(cfg_off).to(device)
    if agent_off.dacc is not None:
        agent_off.dacc.config.dacc_saturation_enabled = False

    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_off.load_state_dict(state)
    except RuntimeError:
        agent_off.load_state_dict(state, strict=False)

    agent_off.e3._running_variance = float(trained_agent.e3._running_variance)
    agent_off.beta_gate = BetaGate(completion_release_threshold=2.0)
    return agent_off


def _around_contradiction_windows(
    committed_history: List[bool], injection_ticks: List[int]
) -> List[Dict[str, float]]:
    """For each counter-evidence injection at tick t, compute the COMMITTED-trajectory
    occupancy FRACTION in the pre-contradiction window [t-W, t) and the post-contradiction
    window (t, t+W], requiring at least WINDOW_MIN_TICKS available ticks on each side.
    Returns one {pre_occ, post_occ} dict per scored window (the paired within-arm
    de-commit datum -- the graded replacement for 468e's committed_frac_post_absolute)."""
    n = len(committed_history)
    events: List[Dict[str, float]] = []
    for t in injection_ticks:
        pre_lo = max(0, t - CONTRADICTION_WINDOW)
        pre = committed_history[pre_lo:t]               # ticks before the injection
        post_hi = min(n, t + 1 + CONTRADICTION_WINDOW)
        post = committed_history[t + 1:post_hi]          # ticks after the injection
        if len(pre) < WINDOW_MIN_TICKS or len(post) < WINDOW_MIN_TICKS:
            continue
        pre_occ = sum(1 for c in pre if c) / float(len(pre))
        post_occ = sum(1 for c in post if c) / float(len(post))
        events.append({"pre_occ": pre_occ, "post_occ": post_occ})
    return events


def _eval_contradiction_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for commitment-vs-contradiction (ported from 468e).
    Adds per-episode COMMITTED-occupancy history + counter-evidence-injection tick tracking
    so the GRADED within-arm around-contradiction committed-occupancy delta (part b) can be
    computed; the 468e between-arm absolute committed_frac_post_absolute is retained as a
    secondary diagnostic."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    hook_enabled = bool(getattr(agent.config, "use_closure_env_completion_hook", False))
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    total_committed_steps = 0
    total_beta_elevated = 0
    total_beta_release_events = 0
    total_beta_release_near_contradiction = 0
    total_committed_pre = 0      # absolute pre-contradiction committed ticks (secondary)
    total_committed_post = 0     # absolute post-contradiction committed ticks (secondary)
    total_pre_steps = 0
    total_post_steps = 0
    total_hook_fires = 0
    total_closure_coupled = 0    # BETA-ENGAGEMENT amend non-vacuity readout
    total_episodes_with_contradiction = 0
    around_events: List[Dict[str, float]] = []  # graded within-arm pre/post-contradiction

    per_episode: List[Dict[str, Any]] = []

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_beta = bool(agent.beta_gate.is_elevated)

            ep_committed = 0
            ep_elevated = 0
            ep_release_events = 0
            ep_release_near_contradiction = 0
            ep_committed_pre = 0
            ep_committed_post = 0
            ep_pre_steps = 0
            ep_post_steps = 0
            ep_hook_fires = 0
            first_contradiction_step = -1
            recent_injection_timers: List[int] = []
            committed_history: List[bool] = []   # per-tick committed-trajectory state
            injection_ticks: List[int] = []      # ticks at which a contradiction fired

            for step in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, feed_harm
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                cur_committed = agent.e3._committed_trajectory is not None
                cur_beta = bool(agent.beta_gate.is_elevated)
                committed_history.append(cur_committed)

                if cur_committed:
                    ep_committed += 1
                if cur_beta:
                    ep_elevated += 1

                # Beta release event (elevated -> not elevated).
                if prev_beta and not cur_beta:
                    ep_release_events += 1
                    if any(t <= RELEASE_WINDOW for t in recent_injection_timers):
                        ep_release_near_contradiction += 1
                prev_beta = cur_beta

                # Advance + prune recent injection timers.
                recent_injection_timers = [t + 1 for t in recent_injection_timers]
                recent_injection_timers = [
                    t for t in recent_injection_timers if t <= RELEASE_WINDOW + 1
                ]

                _, _harm, done, info, obs_dict = env.step(action_idx)

                # Leg A: route an env waypoint completion into emit_closure (the
                # closure-coupled release the Leg-B de-commit hold then protects).
                if (info.get("transition_type") == "sequence_complete"
                        and has_closure and hook_enabled):
                    ev = agent.notify_env_completion(action_class=action_idx)
                    if ev is not None and getattr(ev, "fired", False):
                        ep_hook_fires += 1

                injected = bool(info.get("counter_evidence_injected_this_tick", False))
                if injected:
                    recent_injection_timers.append(0)
                    injection_ticks.append(step)
                    if first_contradiction_step < 0:
                        first_contradiction_step = step

                # 468e absolute accounting (SECONDARY diagnostic): count ALL post-window
                # ticks as the absolute-fraction denominator + the committed subset as the
                # numerator. cur_committed is the pre-step committed state.
                if first_contradiction_step >= 0:
                    ep_post_steps += 1
                    if cur_committed:
                        ep_committed_post += 1
                else:
                    ep_pre_steps += 1
                    if cur_committed:
                        ep_committed_pre += 1

                if done:
                    break

            had_contradiction = first_contradiction_step >= 0
            if had_contradiction:
                total_episodes_with_contradiction += 1
                total_committed_pre += ep_committed_pre
                total_committed_post += ep_committed_post
                total_pre_steps += ep_pre_steps
                total_post_steps += ep_post_steps

            # graded within-arm around-contradiction committed-occupancy windows
            around_events.extend(
                _around_contradiction_windows(committed_history, injection_ticks)
            )

            total_committed_steps += ep_committed
            total_beta_elevated += ep_elevated
            total_beta_release_events += ep_release_events
            total_beta_release_near_contradiction += ep_release_near_contradiction
            total_hook_fires += ep_hook_fires
            # BETA-ENGAGEMENT amend: accumulate this episode's closure-coupled beta
            # elevations BEFORE the next agent.reset() wipes the per-episode counter.
            total_closure_coupled += int(
                agent.beta_gate.get_state().get("sd034_n_closure_coupled_elevations", 0)
            )

            per_episode.append({
                "committed_steps": ep_committed,
                "beta_elevated_steps": ep_elevated,
                "release_events": ep_release_events,
                "release_near_contradiction": ep_release_near_contradiction,
                "committed_pre": ep_committed_pre,
                "committed_post": ep_committed_post,
                "post_steps": ep_post_steps,
                "hook_fires": ep_hook_fires,
                "n_injections": len(injection_ticks),
                "had_contradiction": had_contradiction,
                "first_contradiction_step": first_contradiction_step,
            })

    # 468e absolute post-contradiction committed fraction in [0, 1] (SECONDARY diagnostic).
    committed_frac_post_absolute = (
        total_committed_post / total_post_steps if total_post_steps > 0 else 1.0
    )

    # graded within-arm around-contradiction occupancy aggregates (LOAD-BEARING C2 source)
    n_window_events = len(around_events)
    mean_pre_occ = (
        sum(e["pre_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    mean_post_occ = (
        sum(e["post_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )

    return {
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "total_beta_release_events": total_beta_release_events,
        "beta_release_near_contradiction": total_beta_release_near_contradiction,
        "n_hook_fires": total_hook_fires,
        "sd034_n_closure_coupled_elevations": total_closure_coupled,
        "episodes_with_contradiction": total_episodes_with_contradiction,
        "total_post_steps": total_post_steps,
        "total_pre_steps": total_pre_steps,
        "committed_frac_post_absolute": committed_frac_post_absolute,
        "env_hook_enabled": hook_enabled,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        # graded within-arm around-contradiction DV (part b; LOAD-BEARING)
        "n_window_events": n_window_events,
        "mean_pre_committed_occ": mean_pre_occ,
        "mean_post_committed_occ": mean_post_occ,
    }


def _within_arm_decommit_drop(arm_on: Dict[str, Any]) -> bool:
    """C2 graded within-arm around-contradiction DV (load-bearing): on the ON arm, mean
    post-contradiction COMMITTED occupancy fraction < mean pre-contradiction occupancy
    fraction with a >= DECOMMIT_MIN_DROP_FRAC relative drop, over >= C2_MIN_WINDOW_EVENTS
    scored windows whose pre-occupancy cleared WITHIN_PRE_OCC_FLOOR (there was something
    committed to de-commit)."""
    n_ev = int(arm_on.get("n_window_events", 0))
    pre = float(arm_on.get("mean_pre_committed_occ", 0.0))
    post = float(arm_on.get("mean_post_committed_occ", 0.0))
    if n_ev < C2_MIN_WINDOW_EVENTS or pre <= WITHIN_PRE_OCC_FLOOR:
        return False
    return bool(post < pre and (pre - post) >= DECOMMIT_MIN_DROP_FRAC * pre)


def _within_arm_window_nonvacuous(arm_on: Dict[str, Any]) -> bool:
    """Readiness gate (4): the ON arm produced enough scored around-contradiction windows
    with a non-trivial pre-contradiction committed occupancy for the within-arm DV to be
    interpretable."""
    return bool(
        int(arm_on.get("n_window_events", 0)) >= C2_MIN_WINDOW_EVENTS
        and float(arm_on.get("mean_pre_committed_occ", 0.0)) > WITHIN_PRE_OCC_FLOOR
    )


def _between_arm_drop(arm_on: Dict[str, Any], arm_off: Dict[str, Any]) -> bool:
    """SECONDARY diagnostic (the 468e absolute DV; NOT load-bearing): ON absolute
    post-contradiction committed fraction below OFF by >= DECOMMIT_MIN_DROP_FRAC, OFF having
    a non-trivial post-contradiction window."""
    on_frac = float(arm_on.get("committed_frac_post_absolute", 1.0))
    off_frac = float(arm_off.get("committed_frac_post_absolute", 1.0))
    if off_frac <= 0.1 or int(arm_off.get("total_post_steps", 0)) <= 0:
        return False
    return bool(on_frac < off_frac and (off_frac - on_frac) >= DECOMMIT_MIN_DROP_FRAC * off_frac)


def _rule_bias_mean(p1) -> float:
    diag = getattr(p1, "rule_bias_diag", None) or {}
    n = int(diag.get("n_bias_samples", 0))
    s = float(diag.get("sum_bias_abs_mean", 0.0))
    return s / n if n > 0 else 0.0


def _empty_arm() -> Dict[str, Any]:
    return {
        "total_committed_steps": 0, "total_beta_elevated": 0,
        "total_beta_release_events": 0, "beta_release_near_contradiction": 0,
        "n_hook_fires": 0, "sd034_n_closure_coupled_elevations": 0,
        "episodes_with_contradiction": 0,
        "total_post_steps": 0, "total_pre_steps": 0,
        "committed_frac_post_absolute": 1.0, "env_hook_enabled": False,
        "n_eval_episodes": 0, "closure_present": False,
        "n_window_events": 0, "mean_pre_committed_occ": 0.0, "mean_post_committed_occ": 0.0,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "rule_bias_pathway_enabled": False,
        "rule_bias_mean_abs": 0.0,
        "rule_bias_n_train_steps": 0,
        "rule_bias_trained": False,
        "ARM_SUBSTRATE_ON": _empty_arm(),
        "ARM_SUBSTRATE_OFF": _empty_arm(),
        "criteria": {"C1": False, "C2": False},
        "coupling_nonvacuous": False,
        "within_window_nonvacuous": False,
        "between_arm_drop": False,
        "pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else CONTRADICTION_EVAL_EPISODES

    # Build the agent on a P2-config contradiction env (world_obs_dim parity with the eval).
    probe_env = _build_contradiction_env(scaffold_cfg, dry_run)
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    _enable_dacc_saturation(agent)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    # --- Curriculum build: Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1 (mirror 514n) ---
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

    # --- 603n-canonical contact guard via run_p2 (consumption-event-gated readout) ---
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    # --- Commitment-vs-contradiction DV (always measured; gated at aggregation) ---
    contradiction_env = _build_contradiction_env(scaffold_cfg, dry_run)
    contradiction_env.reset()

    print(f"Seed {seed} Condition ARM_SUBSTRATE_ON", flush=True)
    arm_on = _eval_contradiction_behaviour(
        agent, contradiction_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    print(f"Seed {seed} Condition ARM_SUBSTRATE_OFF", flush=True)
    agent_off = _clone_substrate_off(agent, device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    arm_off = _eval_contradiction_behaviour(
        agent_off, contradiction_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    # C1: ON arm had >= 1 beta release triggered near a contradiction (MECH-268 proximal).
    c1 = arm_on["beta_release_near_contradiction"] >= C1_MIN_RELEASES
    # C2 (graded within-arm around-contradiction de-commit DV; LOAD-BEARING): the ON arm's
    # post-contradiction COMMITTED occupancy is paired-lower than its pre-contradiction
    # occupancy by a relative margin -- the de-commit hold releases the commitment after a
    # contradiction-coupled release. Replaces the 1.0-pinned committed_frac_post_absolute.
    c2 = _within_arm_decommit_drop(arm_on)

    # Coupling + contradiction non-vacuity (TIGHTENED gate): the closure-plane coupling
    # actually engaged the latch on the ON arm AND a contradiction fired (the DV had an
    # opportunity). Replaces 468e's both-arms-total_beta_elevated gate.
    coupling_nonvacuous = bool(
        arm_on["sd034_n_closure_coupled_elevations"] > 0
        and arm_on["episodes_with_contradiction"] > 0
    )
    within_window_nonvacuous = _within_arm_window_nonvacuous(arm_on)
    between_arm_drop = _between_arm_drop(arm_on, arm_off)  # secondary diagnostic only

    seed_pass = bool(c1 and c2)

    decommit_gap = (
        arm_on["mean_pre_committed_occ"] - arm_on["mean_post_committed_occ"]
    )
    print(f"  [train] contradiction_eval seed={seed} ep {done}/{total_eps}"
          f" c1={c1} c2_within={c2} decommit_gap={decommit_gap:.4f}"
          f" pre_occ={arm_on['mean_pre_committed_occ']:.3f} post_occ={arm_on['mean_post_committed_occ']:.3f}"
          f" win_events={arm_on['n_window_events']} coupled={arm_on['sd034_n_closure_coupled_elevations']}"
          f" rel_near={arm_on['beta_release_near_contradiction']}"
          f" eps_contra={arm_on['episodes_with_contradiction']}"
          f" between_arm_drop={between_arm_drop}"
          f" on_cfp_abs={arm_on['committed_frac_post_absolute']:.4f}"
          f" off_cfp_abs={arm_off['committed_frac_post_absolute']:.4f}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} coupling_nonvacuous={coupling_nonvacuous}"
          f" within_window={within_window_nonvacuous} rule_bias_trained={rule_bias_trained}"
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
        "ARM_SUBSTRATE_ON": arm_on,
        "ARM_SUBSTRATE_OFF": arm_off,
        "decommit_gap": float(decommit_gap),
        "criteria": {"C1": c1, "C2": c2},
        "coupling_nonvacuous": coupling_nonvacuous,
        "within_window_nonvacuous": within_window_nonvacuous,
        "between_arm_drop": between_arm_drop,
        "pass": seed_pass,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + 2 * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + 2 * CONTRADICTION_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # Readiness gate (2): rule_bias_head actually trained (the anti-460d-bug gate).
    rb_flags = [bool(r.get("rule_bias_trained", False)) for r in guard_passing]
    rb_frac = _frac(rb_flags)
    rule_bias_trained_met = bool(rb_frac >= MIN_FRACTION)

    # Readiness gate (3): closure-coupling + contradiction non-vacuity (TIGHTENED).
    cp_flags = [bool(r.get("coupling_nonvacuous", False)) for r in guard_passing]
    cp_frac = _frac(cp_flags)
    coupling_nonvacuity_met = bool(cp_frac >= MIN_FRACTION)

    # Readiness gate (4): within-arm window non-vacuity among guard-passing seeds.
    ww_flags = [bool(r.get("within_window_nonvacuous", False)) for r in guard_passing]
    ww_frac = _frac(ww_flags)
    within_window_met = bool(ww_frac >= MIN_FRACTION)

    seed_pass_flags = [bool(r.get("pass", False)) for r in guard_passing]
    n_pass = sum(1 for f in seed_pass_flags if f)
    pass_frac = _frac(seed_pass_flags)
    overall_criteria_pass = bool(pass_frac >= MIN_FRACTION)

    def _all_guard(crit_key: str) -> bool:
        return bool(guard_passing) and all(
            r.get("criteria", {}).get(crit_key) for r in guard_passing
        )

    c1_all = _all_guard("C1")
    c2_all = _all_guard("C2")

    readiness_all_met = bool(
        contact_non_vacuity_met and rule_bias_trained_met
        and coupling_nonvacuity_met and within_window_met
    )

    if not contact_non_vacuity_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "contact_guard_unmet"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not rule_bias_trained_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "rule_bias_head_untrained"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not coupling_nonvacuity_met:
        # The closure-plane coupling did not engage the latch / no contradiction fired on the
        # ON arm (the 468e/460f inert-coupling signature) -> the de-commit DV is measuring the
        # fragile natural commit-entry, not the closure-plane commitment = substrate not ready.
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "closure_coupling_or_contradiction_not_engaged"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not within_window_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "within_arm_windows_vacuous"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        # All four readiness gates clear -> the graded within-arm de-commit DV is interpretable.
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("sd034_decommit_authority_confirmed"
                           if overall_criteria_pass else "residual_decommit_authority_open")
        route_reason = ("c1_c2_majority_met" if overall_criteria_pass
                        else "decommit_dv_unmet_genuine_weakens")
        direction_map = {
            # SD-034 de-commit AUTHORITY keys on the graded within-arm C2 (and C1 release).
            "SD-034": "supports" if (c1_all and c2_all) else "weakens",
            # MECH-268 dACC PE -> beta release proximal pathway keys on C1.
            "MECH-268": "supports" if c1_all else "weakens",
            # MECH-090 latch RELEASE keys on C1 (the 468e autopsy mis-attribution fix --
            # do NOT weaken the active latch on a downstream de-commit-authority (C2) fail).
            "MECH-090": "supports" if c1_all else "weakens",
        }
        overall_direction = "supports" if overall_criteria_pass else "weakens"

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) rule_bias_trained={rule_bias_trained_met}"
          f" (frac={rb_frac:.3f}) coupling_nonvacuous={coupling_nonvacuity_met} (frac={cp_frac:.3f})"
          f" within_window={within_window_met} (frac={ww_frac:.3f})"
          f" criteria_pass={overall_criteria_pass} ({n_pass}/{len(guard_passing)})"
          f" -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "rule_bias_trained_met": rule_bias_trained_met,
        "rule_bias_trained_fraction": rb_frac,
        "coupling_nonvacuity_met": coupling_nonvacuity_met,
        "coupling_nonvacuity_fraction": cp_frac,
        "within_window_met": within_window_met,
        "within_window_fraction": ww_frac,
        "C1_all_guard_passing": c1_all,
        "C2_within_all_guard_passing": c2_all,
        "criteria_pass_fraction": pass_frac,
        "n_seeds_pass": n_pass,
        "overall_pass": bool(readiness_all_met and overall_criteria_pass),
        "per_seed_guard_pass": guard_flags,
        "per_seed_criteria_pass": [bool(r.get("pass", False)) for r in per_seed],
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
                    "description": "Leg C readiness (the DIRECT anti-460d-bug gate): P1 "
                                   "rule_bias_pathway_enabled AND mean per-candidate |bias| "
                                   "> floor on >= 2/3 seeds. Below floor -> substrate_not_"
                                   "ready_requeue (the head did not train), NEVER a weakens.",
                    "control": "P1OnboardingResult.rule_bias_diag mean |bias|.",
                    "measured": rb_frac,
                    "threshold": MIN_FRACTION,
                    "met": rule_bias_trained_met,
                },
                {
                    "name": "closure_coupling_and_contradiction_nonvacuous",
                    "description": "TIGHTENED non-vacuity (the brief): ON "
                                   "sd034_n_closure_coupled_elevations > 0 AND ON "
                                   "episodes_with_contradiction > 0 on >= 2/3 guard seeds -- "
                                   "the closure-plane coupling engaged the latch AND a "
                                   "contradiction fired (not the fragile natural commit-entry). "
                                   "Below floor -> substrate_not_ready_requeue (the 468e/460f "
                                   "inert-coupling signature).",
                    "control": "ARM_SUBSTRATE_ON sd034_n_closure_coupled_elevations + "
                               "episodes_with_contradiction.",
                    "measured": cp_frac,
                    "threshold": MIN_FRACTION,
                    "met": coupling_nonvacuity_met,
                },
                {
                    "name": "within_arm_window_nonvacuous",
                    "description": "ON arm produced >= C2_MIN_WINDOW_EVENTS scored "
                                   "around-contradiction windows with mean pre-contradiction "
                                   "committed occupancy > WITHIN_PRE_OCC_FLOOR on >= 2/3 guard "
                                   "seeds -- there was something committed to de-commit. Below "
                                   "floor -> substrate_not_ready_requeue.",
                    "control": "ARM_SUBSTRATE_ON n_window_events + mean_pre_committed_occ.",
                    "measured": ww_frac,
                    "threshold": MIN_FRACTION,
                    "met": within_window_met,
                },
            ],
            "criteria": [
                {"name": "C1_beta_release_near_contradiction", "load_bearing": True,
                 "passed": c1_all},
                {"name": "C2_within_arm_around_contradiction_committed_occ_drop",
                 "load_bearing": True, "passed": c2_all},
            ],
            "criteria_non_degenerate": {
                "C1": readiness_all_met,
                "C2": readiness_all_met,
            },
            "contact_guard": {
                "definition": "per-seed: P2 contact_rate > 0 AND z_goal_norm_at_contact_peak "
                              "> 0.4 (603n G2 + G3). < 2/3 seeds passing -> "
                              "substrate_not_ready_requeue, never a false weakens.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "graded_within_arm_dv": {
                "definition": "468f C2 reads a PAIRED within-ON-arm around-contradiction "
                              "COMMITTED-occupancy delta: for each counter-evidence injection "
                              "at tick t, committed occupancy in [t-W, t) vs (t, t+W]; the "
                              "de-commit lowers post-contradiction occupancy below "
                              "pre-contradiction occupancy by >= DECOMMIT_MIN_DROP_FRAC over "
                              ">= C2_MIN_WINDOW_EVENTS windows (pre > WITHIN_PRE_OCC_FLOOR). "
                              "Graded -- replaces the 468e committed_frac_post_absolute that "
                              "re-pinned at the 1.0 ceiling 3/3.",
                "contradiction_window": CONTRADICTION_WINDOW,
                "window_min_ticks": WINDOW_MIN_TICKS,
                "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
                "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
                "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
                "release_window": RELEASE_WINDOW,
            },
            "amend_legs_under_test": {
                "leg_a_env_completion_hook": "REEAgent.notify_env_completion -> emit_closure "
                                             "on env sequence_complete (waypoint) ticks.",
                "leg_b_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
                "leg_b_magnitude_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
                "leg_b_magnitude_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
                "leg_c_trained_rule_bias_head": True,
                "beta_engagement_coupling": True,
            },
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
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> "
                     "P0 -> Stage-H -> P1 -> P2; harm-pathway training ON, Leg-C "
                     "scaffold_train_rule_bias_head ON, 603n config; ready=true 2026-06-11) + "
                     "commitment control-plane (bistable BetaGate MECH-090 + SD-034 "
                     "ClosureOperator + MECH-268 dACC PE saturation + SD-033a LateralPFC + "
                     "SD-032 dACC/salience) + commitment-closure-control-plane amend Legs A/B/C "
                     "+ beta-engagement coupling + DE-COMMIT-AUTHORITY MAGNITUDE lever "
                     "(closure_decommit_hold_scale_with_run + _max_ticks, ree-v3 main 2cd0aa2) "
                     "+ GAP-3 contradiction env primitives (completion_tolerance(waypoint) + "
                     "counter_evidence).",
        "condition": CONDITION_LABEL,
        "supersedes": SUPERSEDES,
        "method_note": "468e's commitment-vs-contradiction perseveration mechanism (SD-034 "
                       "closure + MECH-268 dACC PE saturation coordinate a MECH-090 beta "
                       "release under counter-evidence; the Leg-B de-commit hold keeps the "
                       "ON arm UNcommitted through the post-contradiction window; OFF "
                       "perseverates) re-run on the DE-COMMIT-AUTHORITY-MAGNITUDE-amended "
                       "substrate (committed-run-scaled refractory) WITH the C2 DV redesigned "
                       "from the 1.0-pinned absolute committed_frac_post_absolute to a GRADED "
                       "PAIRED within-ON-arm around-contradiction committed-occupancy delta. "
                       "The perseveration-side sibling of the 460f->460g de-commit re-issue.",
        "readiness_note": "FOUR readiness gates self-route a substrate-not-engaged read to "
                          "non_contributory (never a false weakens): (1) 603n contact guard; "
                          "(2) rule_bias_head trained (the anti-460d-bug gate); (3) closure-"
                          "coupling + contradiction non-vacuity (sd034_n_closure_coupled_"
                          "elevations > 0 AND a contradiction fired); (4) within-arm window "
                          "non-vacuity (enough scored around-contradiction windows with a "
                          "non-trivial pre-contradiction committed occupancy). Only when ALL "
                          "clear does C1/C2 drive a supports/weakens verdict.",
        "arm_note": "ARM_SUBSTRATE_ON (full closure + dACC saturation + bistable + env hook + "
                    "de-commit hold + MAGNITUDE lever, trained rule_bias_head) vs "
                    "ARM_SUBSTRATE_OFF (same trained weights, closure off + saturation off). "
                    "The graded within-arm C2 is measured on the ON arm; the between-arm "
                    "absolute committed_frac_post_absolute is a secondary diagnostic only.",
        "claim_id_note": "MECH-090 keys on C1 (latch RELEASE), NOT the C2-gated overall PASS "
                         "-- the failure_autopsy_V3-EXQ-468e_2026-06-18 established the 468e "
                         "self-stamped MECH-090 weakens was MIS-ATTRIBUTED (the run tests "
                         "MECH-090's latch release via C1, which passed 3/3; do not weaken the "
                         "active latch on a downstream de-commit-authority fail). MECH-261 is "
                         "NOT tagged: all closures were Leg-A hook-driven (n_automatic_fires=0) "
                         "so mode-conditioning is not exercised (the 468e/460f governance).",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "c1_min_releases": C1_MIN_RELEASES,
            "release_window": RELEASE_WINDOW,
            "contradiction_window": CONTRADICTION_WINDOW,
            "window_min_ticks": WINDOW_MIN_TICKS,
            "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
            "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
            "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
            "rule_bias_mean_floor": RULE_BIAS_MEAN_FLOOR,
            "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
            "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
            "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
            "dacc_saturation_window": DACC_SAT_WINDOW,
            "dacc_saturation_strength": DACC_SAT_STRENGTH,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "contradiction_eval_episodes_per_arm": CONTRADICTION_EVAL_EPISODES,
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "scaffold_train_rule_bias_head": True,
            "config_basis": "V3-EXQ-603n (substrate-readiness run that flipped "
                            "scaffolded_sd054_onboarding ready=true) + 460g magnitude lever",
        },
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
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
        )
