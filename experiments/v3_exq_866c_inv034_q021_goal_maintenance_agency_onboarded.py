#!/opt/local/bin/python3
"""
V3-EXQ-866c -- INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency,
MEASUREMENT-HARNESS CORRECTION of V3-EXQ-866a (driver-only re-run).

Supersedes V3-EXQ-866a (confirmed failure_autopsy_V3-EXQ-866a_2026-08-03).
866a inherits its whole three-condition design, curriculum config, seeds, and
pre-registered gates G0/C1-C5 UNCHANGED from 866. The ONE change here is how the
C6 z_goal mechanistic readout is measured -- see "WHY 866c: THE C6 MEASUREMENT-
HARNESS BUG" below. This is a measurement-harness correction, NOT a new
experimental design.

Routed from decision chip chip-20260808-scaffolded-c6-misdiagnosis-routing
(user-authorized 2026-08-08, option A accepted), itself from IGW-20260808-200's
diagnosis. Full diagnosis + empirical probe:
REE_assembly/evidence/planning/scaffolded_curriculum_hazard_rebalance_diagnosis_staged_2026-08-08.md
(committed REE_assembly 5521a31df5).

Claims: INV-034 (primary), Q-021 (companion open question -- unchanged from 866a).

WHY 866c: THE C6 MEASUREMENT-HARNESS BUG (driver-only; no substrate change)
--------------------------------------------------------------------------
866a FAILed with c6_pass=0 and zgoal_norm_mean_FULL=0.1198 (C6 needs > 0.4),
reading as "z_goal not maintained in FULL". That reading is a measurement
ARTIFACT of 866a's own P2 harness, not a substrate property:

- The substrate PRESERVES z_goal through the entire curriculum. An instrumented
  probe of the real ScaffoldedSD054OnboardingScheduler (866a's exact config)
  shows z_goal enters the P2 window at ~0.52 (Stage-0 0.48 -> Stage-0b/P0/Stage-H
  hold it flat -> P1 ecological contact grows it to ~0.53). The frozen stages
  never call update_z_goal, so they cannot decay z_goal; the scheduler's own
  run_p2 is contact-gated (scaffold_developmental_window_enabled +
  scaffold_contact_gated_goal_updates, both set in this config), so it SKIPS
  update_z_goal on unfed steps (n_decay_only == 0) and reports the maintained
  attractor peak (z_goal_norm_peak_max ~0.52), not a decaying mean.

- 866a's driver did NOT use run_p2. Its _measure_866_style rolled its own P2
  readout that called update_z_goal(benefit, drive) UNCONDITIONALLY every step
  and averaged ||z_goal|| over all steps. On the P2 env with
  resource_visit_rate ~0.003, ~all steps are unfed, so GoalState.update's
  unconditional decay (decay_goal=0.005; 0.995^200 ~ 0.37 per 200-step episode,
  and z_goal persists across episodes -- REEAgent.reset() does not reset
  goal_state) washes z_goal to near-zero, and the 30-episode mean lands at
  ~0.12. That is a decay-only washout, not a failure of goal maintenance.

THE FIX (this file): measure the C6 z_goal readout via the scheduler's OWN
contact-gated run_p2 (z_goal_norm_peak_max), instead of 866a's decay-only
unconditional-update mean. This stops the driver rolling its own z_goal
measurement -- the exact anti-pattern that caused the bug -- and delegates to the
scheduler's tested, contract-covered, contact-gated measurement.

HOW THE C6 READOUT IS COMPUTED (per trained arm, per seed)
----------------------------------------------------------
1. Train through the SAME curriculum as 866a: run_stage0_nursery ->
   run_stage0b_consolidation -> run_p0 -> run_hazard_avoidance -> run_p1.
2. Snapshot the post-P1 z_goal tensor (the maintained attractor, ~0.52).
3. Run _measure_866_style EXACTLY as 866a for the BEHAVIORAL metrics that gates
   G0/C1-C5 consume (transition_type resource_visit_rate / harm_rate,
   action-histogram policy_entropy, survival_rate). This pass is bit-identical
   to 866a: it runs first, from the pristine post-P1 state, on the same RNG
   stream, with update_z_goal still called every step. Its own zgoal_norm_mean
   (the decay-only washout) is RECORDED for the audit trail as
   zgoal_norm_mean_866a_style, but is NO LONGER used for C6.
4. Restore z_goal to the post-P1 snapshot (undoing step 3's decay), then call
   scheduler.run_p2(agent, device) -- the scheduler's contact-gated P2 phase --
   and take z_goal_norm_peak_max as the C6 readout. run_p2 also records
   z_goal_norm_at_contact_peak, n_decay_only (== 0 confirms the measurement is
   contact-gated), n_skipped_protected, and num_contact_events, all stamped into
   the manifest for transparency.

WHY z_goal_norm_peak_max, NOT z_goal_norm_at_contact_peak, FOR THE C6 BOOLEAN
----------------------------------------------------------------------------
C6 is a MECHANISTIC check: "is z_goal genuinely engaged in FULL and absent in
AVOIDANCE_ONLY?" -- a wanting-pathway-engagement question, orthogonal to whether
the agent forages well (that is G0/C4). z_goal_norm_peak_max reads the maintained
attractor and is robust to the foraging-competence confound. Because G0 fails on
this substrate (see below), the FULL arm makes almost no ecological contact in
P2, so z_goal_norm_at_contact_peak can read low/0.0 for lack of contact events --
which would spuriously FAIL C6 for a z_goal that IS mechanistically maintained.
Using the contact-gated peak as the C6 gate would conflate z_goal-maintenance
with foraging success, i.e. re-import exactly the G0 signal C6 is meant to be
independent of. So C6 gates on z_goal_norm_peak_max; z_goal_norm_at_contact_peak
is recorded but not gated. AVOIDANCE_ONLY has agent.goal_state is None, so run_p2
returns 0.0 for both z_goal readouts (all z_goal reads in _eval_episode are
guarded and update_z_goal early-returns) -- trivially < the C6 inactive ceiling.

THE PRIMARY BLOCKER IS G0, NOT C6 -- 866c IS NOT EXPECTED TO PASS OVERALL
------------------------------------------------------------------------
866a's actual primary FAIL is G0: the FULL arm forages BELOW the RANDOM baseline
(resource_visit_rate_FULL 0.0033 vs RANDOM 0.0103). G0 gates first, so even a
perfect C6 fix leaves the run FAILing. G0 is the long-running GAP-2 / Stage-H
foraging-competence ceiling this lineage has fought since ~2026-06; it is a
foraging/survival problem, NOT the z_goal-maintenance problem C6 measures, and it
is routed separately to the G0 foraging-competence thread (chip
chip-20260808-g0-foraging-competence-autopsy). 866c's contribution is a HONEST,
un-confounded C6 z_goal readout alongside a faithfully-reproduced G0 result;
governance can then read C6 as a clean mechanistic control rather than a
measurement artifact. Expect status FAIL (G0 gates), c6 now measured correctly.

RE-DERIVE BRAKE (Step 2.5b) -- WHY 866c IS NOT BRAKED
-----------------------------------------------------
The literal re-derive counter reads 2 for INV-034/Q-021 (866 + 866a autopsies),
but neither owes a substrate build: failure_autopsy_V3-EXQ-866 is
`substrate_not_ready_requeue` (action=none -- a re-run recommendation, a
false-positive of the non_contributory direction proxy), and
failure_autopsy_V3-EXQ-866a is `substrate_ceiling` with action=none (no build
owed), naming the Stage-H foraging path. The brake's remedy is "route to
/implement-substrate on the named substrate" -- inapplicable here because no
substrate build is owed and 866a's own ceiling reading was partly driven by the
C6 measurement artifact this fix corrects. 866c is a USER-AUTHORIZED
instrument-repair (measurement-harness) correction -- the Step 2.5b
instrument-defect carve-out -- NOT a re-attempt of the G0 ceiling. G0 is reported
honestly as still-failing and routed to its own thread.

DESIGN: SAME THREE CONDITIONS, SAME GATES (formulas/margins unchanged from 866a)
-------------------------------------------------------------------------------
  A. FULL          -- full goal/wanting pathway live (z_goal_enabled, MECH-295
                       liking bridge, MECH-307 conjunction, SD-057 incentive bank
                       + cue-recall, drive_weight + seeding calibration on;
                       validated V3-EXQ-603q ARM_BASE_IA_ONLY agent config).
  B. AVOIDANCE_ONLY -- the SAME harm-avoidance / NoGo channel as FULL (identical
                       harm streams, E2_harm_s forward, PAG freeze gate,
                       instrumental avoidance gate, safety terrain -- unchanged),
                       but z_goal_enabled=False, drive_weight=0, MECH-295/307 off,
                       incentive bank + cue-recall off. agent.goal_state is None,
                       so every goal-pipeline call no-ops or aborts cleanly.
                       BOTH trained arms share the identical
                       ScaffoldedSD054OnboardingConfig; only the REEConfig differs
                       -> single-variable ablation by construction.
  C. RANDOM        -- uniform action selection on the SAME P2-phase target env,
                       for episode/step parity (distinguishes "B is actively
                       quiescent" from "B can't find resources").

PRE-REGISTERED GATES (IDENTICAL formulas/margins to V3-EXQ-866/866a; in order,
G0 must pass first or the run self-routes to non_contributory):
  G0 non-degeneracy: resource_visit_rate_FULL >= resource_visit_rate_RANDOM
      + 0.05 on >= 2/3 seeds.
  C1 harm-avoidance parity: harm_rate_AVOIDANCE_ONLY <= harm_rate_FULL * 1.15
      on >= 2/3 seeds.
  C2 survival parity: episode_survival_rate_AVOIDANCE_ONLY >=
      episode_survival_rate_FULL - 0.1 on >= 2/3 seeds.
  C3 quiescence/flatness: resource_visit_rate_AVOIDANCE_ONLY <=
      resource_visit_rate_RANDOM + 0.05 on >= 2/3 seeds.
  C4 approach restored by goal maintenance: resource_visit_rate_FULL >=
      resource_visit_rate_AVOIDANCE_ONLY + 0.10 on >= 2/3 seeds.
  C5 entropy signature: policy_entropy_AVOIDANCE_ONLY <=
      policy_entropy_FULL - 0.10 on >= 2/3 seeds.
  C6 mechanistic check (CORRECTED READOUT): z_goal_norm_peak_max_FULL > 0.4 AND
      z_goal_norm_peak_max_AVOIDANCE_ONLY < 0.1 on >= 2/3 seeds. (866a gated the
      same thresholds on the decay-only zgoal_norm_mean; 866c gates them on the
      contact-gated run_p2 peak. Thresholds unchanged.)

PASS (supports INV-034 + Q-021) = G0 AND C1 AND C2 AND C3 AND C4 AND C5 AND C6.
G0 fails -> non_contributory. C1/C2 fail (with G0 passing) -> mixed/
non_contributory. G0+C1+C2 pass but C3/C4/C5/C6 don't show the predicted gap ->
weakens.

Biological basis (unchanged from 866): Bariselli 2018 (PMID 29481617) D1/D2
pathways evaluate the SAME proposals; without D1 (Go/wanting) only D2
(NoGo/avoidance) constrains selection -> quiescent default. Barch & Dowd 2010
(PMID 20868638) "wanting" (prospective) vs "liking" (reactive); avolition in
schizophrenia = intact liking, absent wanting.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

ethics_preflight:
  involves_negative_valence: false        # harm stream is pre-ethical instrumentation (SENT-0)
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# experiments package
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _benefit_and_drive,
    _sense_with_optional_harm,
)

# --------------------------------------------------------------------- #
# Experiment metadata
# --------------------------------------------------------------------- #
EXPERIMENT_TYPE = "v3_exq_866c_inv034_q021_goal_maintenance_agency_onboarded"
QUEUE_ID = "V3-EXQ-866c"
SUPERSEDES = "V3-EXQ-866a"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["INV-034", "Q-021"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
CONDITIONS = ["FULL", "AVOIDANCE_ONLY", "RANDOM"]

# --- Curriculum budgets (603q ARM_BASE_IA_ONLY Stage-0/0b/P0/Stage-H,
#     extended with P1/P2 at the scaffold's own class-default target env) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 30
STEPS_PER_EP = 200
P0_NUM_HAZARDS = 1
TOTAL_TRAIN_EPS = (
    STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
    + P1_BUDGET + P2_BUDGET
)  # 250 -- denominator for [train] ep N/M (trained arms)

# Dry-run (smoke) budgets.
DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_P1, DRY_P2, DRY_STEPS = 2, 2, 5, 5, 5, 8, 30
DRY_TOTAL_TRAIN_EPS = (
    DRY_STAGE0 + DRY_STAGE0B + DRY_P0 + DRY_HAZARD + DRY_P1 + DRY_P2
)

# --- Stage-H regime (603q's amend-validated anchor) -------------------------
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 603q) ------
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal ---------------------------
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# --- Harm-pathway training (603k) + stabilization (603q amend) --------------
HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250

# --- P2 measurement guard (near-universal across scaffolded_sd054_onboarding
#     consumers; unchanged from 866a -- see 866a docstring for the empirical
#     justification). 0.3 admits foraging contact during measurement while still
#     being harder than P0/P1's lower values. ---
P2_HFA_GUARD = 0.3

# --- Encoder / latent dims (mirror 603q exactly) ----------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
ALPHA_WORLD = 0.9

# Pre-registered thresholds -- IDENTICAL to V3-EXQ-866/866a (see module docstring).
MIN_SEEDS_PASS = 2  # of 3 -- ">= 2/3 seeds"
G0_MARGIN = 0.05
C1_HARM_TOLERANCE = 1.15
C2_SURVIVAL_MARGIN = 0.10
C3_MARGIN = 0.05
C4_MARGIN = 0.10
C5_ENTROPY_MARGIN = 0.10
C6_ZGOAL_ACTIVE_FLOOR = 0.4
C6_ZGOAL_INACTIVE_CEIL = 0.1


# --------------------------------------------------------------------- #
# Scaffold + agent config builders (IDENTICAL to 866a)
# --------------------------------------------------------------------- #

def build_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    """The scheduler config shared by BOTH trained arms (FULL and
    AVOIDANCE_ONLY). Identical env kwargs and curriculum budgets at every phase
    -- only the REEConfig each arm's agent is built with differs
    (build_agent_config). This is what makes the ablation single-variable by
    construction."""
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_P1, DRY_P2, DRY_STEPS)
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, STEPS_PER_EP)

    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_strict_goal_isolation=False,  # legacy read-path behaviour (landed default)
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # Stage-H isolated hazard-avoidance leg.
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver.
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # Harm stream + harm-pathway training.
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
        # P2 measurement guard -- see P2_HFA_GUARD comment above.
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
    )
    if steps < 75:
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def build_agent_config(env, condition: str) -> REEConfig:
    """REEConfig for the given condition. FULL and AVOIDANCE_ONLY share every
    harm/avoidance-channel flag (the C1/C2 control per 866's own invariant); only
    the wanting/goal-pathway flags differ. IDENTICAL to 866a."""
    full = (condition == "FULL")
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        # NoGo / harm-avoidance channel -- IDENTICAL in both trained arms.
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
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_escape_affordance_bridge=False,
        use_escape_relief_credit=False,
        use_escape_safety_credit=False,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
        # Encoder-quality fairness -- both arms.
        e2_action_contrastive_enabled=True,
        # Go / wanting channel -- FULL only.
        z_goal_enabled=full,
        drive_weight=(DRIVE_WEIGHT if full else 0.0),
        e1_goal_conditioned=full,
        use_mech295_liking_bridge=full,
        use_mech307_conjunction=full,
        use_incentive_token_bank=full,
        use_cue_recall=full,
        cue_recall_gain=(CUE_RECALL_GAIN if full else 0.0),
    )
    cfg.latent.use_resource_encoder = True  # both arms -- encoder-quality fairness
    return cfg


# --------------------------------------------------------------------- #
# Metrics helpers (identical formulas to V3-EXQ-866/866a)
# --------------------------------------------------------------------- #

def _action_entropy(action_counts: List[int]) -> float:
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _frac_seeds(pred_per_seed: List[bool]) -> float:
    return sum(1 for p in pred_per_seed if p) / max(1, len(pred_per_seed))


# --------------------------------------------------------------------- #
# 866-style BEHAVIORAL P2 measurement (G0/C1-C5) -- UNCHANGED from 866a.
# The zgoal_norm_mean it returns is the decay-only washout; 866c records it as
# zgoal_norm_mean_866a_style for the audit trail but NO LONGER gates C6 on it.
# --------------------------------------------------------------------- #

def _measure_866_style(
    agent: Optional[REEAgent],
    env,
    device: torch.device,
    cfg: ScaffoldedSD054OnboardingConfig,
    n_episodes: int,
    steps_per_ep: int,
    condition: str,
) -> Dict[str, Any]:
    """Frozen-policy (or random) BEHAVIORAL measurement using V3-EXQ-866's own
    metric definitions (transition_type-based resource_visit_rate / harm_rate,
    action-histogram policy_entropy, survival_rate). Bit-identical to 866a. The
    zgoal_norm_mean it computes is the decay-only unconditional-update washout;
    it is recorded but the C6 gate reads run_p2's contact-gated peak instead (see
    _run_trained_cell). `agent=None` -> uniform random (the RANDOM condition)."""
    import random as _random

    is_random = agent is None
    if not is_random:
        agent.eval()
    action_dim = env.action_dim

    resource_visits = 0
    harm_events = 0
    episodes_survived = 0
    action_counts = [0] * action_dim
    action_counts_fresh = [0] * action_dim
    total_steps = 0
    fresh_select_steps = 0
    zgoal_norms: List[float] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        if not is_random:
            agent.reset()
        ep_died = False

        for _step in range(steps_per_ep):
            if is_random:
                action_idx = _random.randint(0, action_dim - 1)
                action_counts[action_idx] += 1
                action_counts_fresh[action_idx] += 1
                fresh_select_steps += 1
            else:
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                with torch.no_grad():
                    latent = _sense_with_optional_harm(
                        agent, obs_body, obs_world, obs_dict, device,
                        cfg.scaffold_feed_harm_stream,
                    )
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent)
                        if ticks.get("e1_tick", True)
                        else torch.zeros(1, agent.config.latent.world_dim, device=device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=0.5)
                action_idx = int(action.argmax(dim=-1).item())
                action_counts[action_idx] += 1
                if ticks.get("e3_tick", False):
                    action_counts_fresh[action_idx] += 1
                    fresh_select_steps += 1

                benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(device))
                agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
                if condition == "FULL" and agent.goal_state is not None:
                    zgoal_norms.append(float(agent.goal_state.z_goal.norm().item()))

            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            harm_val = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0

            if ttype == "benefit_approach" or ttype == "resource":
                resource_visits += 1
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1
            total_steps += 1
            if done:
                if harm_val > 0.0:
                    ep_died = True
                break
        episodes_survived += 0 if ep_died else 1

    resource_visit_rate = resource_visits / max(1, total_steps)
    harm_rate = harm_events / max(1, total_steps)
    policy_entropy = _action_entropy(action_counts)
    policy_entropy_fresh_select = (
        _action_entropy(action_counts_fresh) if fresh_select_steps > 0 else 0.0
    )
    survival_rate = episodes_survived / max(1, n_episodes)
    zgoal_norm_mean = (
        sum(zgoal_norms) / len(zgoal_norms) if zgoal_norms else 0.0
    )

    return {
        "resource_visit_rate": resource_visit_rate,
        "harm_rate": harm_rate,
        "policy_entropy": policy_entropy,
        "policy_entropy_fresh_select": policy_entropy_fresh_select,
        "n_fresh_select": fresh_select_steps,
        "n_held": total_steps - fresh_select_steps,
        "survival_rate": survival_rate,
        # 866a-style decay-only washout -- RECORDED for the audit trail, NOT gated.
        "zgoal_norm_mean_866a_style": zgoal_norm_mean,
        "n_resource_events": resource_visits,
        "n_harm_events": harm_events,
        "total_steps": total_steps,
    }


# --------------------------------------------------------------------- #
# Trained cells (FULL / AVOIDANCE_ONLY)
# --------------------------------------------------------------------- #

def _cell_config_slice(condition: str, seed: int, dry_run: bool) -> Dict:
    return {
        "condition": condition,
        "seed": seed,
        "dry_run": dry_run,
        "stage0_budget": STAGE0_BUDGET,
        "stage0b_budget": STAGE0B_BUDGET,
        "p0_budget": P0_BUDGET,
        "hazard_budget": HAZARD_STAGE_BUDGET,
        "p1_budget": P1_BUDGET,
        "p2_budget": P2_BUDGET,
        "steps_per_ep": STEPS_PER_EP,
        "p0_num_hazards": P0_NUM_HAZARDS,
        "hazard_stage_regime": [
            HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
            HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
        ],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR, N_RESOURCE_TYPES],
        "harm_pathway_lr": [HARM_PATHWAY_LR, HARM_PATHWAY_ENCODER_LR, HARM_PATHWAY_WARMUP_STEPS],
        "p2_hfa_guard": P2_HFA_GUARD,
        "z_goal_enabled": (condition == "FULL"),
        "drive_weight": (DRIVE_WEIGHT if condition == "FULL" else 0.0),
        # 866c C6-readout-source marker (does not change the trained substrate;
        # only the readout path differs from 866a).
        "c6_readout_source": "scheduler_run_p2_contact_gated_peak",
    }


def _run_trained_cell(
    condition: str, seed: int, zg: ZGoalStreamAccumulator, dry_run: bool
) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {condition}", flush=True)
    device = torch.device("cpu")

    sched_cfg = build_scaffold_cfg(dry_run)
    scheduler = ScaffoldedSD054OnboardingScheduler(sched_cfg)

    probe_env = _build_env(sched_cfg, phase="p2", seed=None)
    torch.manual_seed(seed)
    agent_cfg = build_agent_config(probe_env, condition)
    agent = REEAgent(agent_cfg).to(device)

    ep_so_far = 0

    def _progress(phase_name: str, n_this_phase: int):
        nonlocal ep_so_far
        ep_so_far += n_this_phase
        total = DRY_TOTAL_TRAIN_EPS if dry_run else TOTAL_TRAIN_EPS
        print(
            f"  [train] seed={seed} {condition} ep {ep_so_far}/{total} phase={phase_name}",
            flush=True,
        )

    s0 = scheduler.run_stage0_nursery(agent, device)
    _progress("stage0", s0.n_episodes if not s0.aborted else 0)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    _progress("stage0b", s0b.n_episodes if not s0b.aborted else 0)

    p0 = scheduler.run_p0(agent, device)
    _progress("p0", p0.n_episodes)

    hz = scheduler.run_hazard_avoidance(agent, device)
    _progress("stage_h", hz.n_episodes)

    p1 = scheduler.run_p1(agent, device)
    _progress("p1", p1.n_episodes)

    steps_per = DRY_STEPS if dry_run else STEPS_PER_EP
    p2_eps = DRY_P2 if dry_run else P2_BUDGET

    # --- Snapshot the post-P1 z_goal (the maintained attractor, ~0.52) so the
    #     contact-gated run_p2 mechanistic readout below sees it, rather than the
    #     near-zero state _measure_866_style leaves behind after its unconditional
    #     per-step decay. Restoring ONLY z_goal keeps the behavioral pass
    #     bit-identical to 866a while giving run_p2 the correct entry state.
    saved_z_goal = None
    gs = getattr(agent, "goal_state", None)
    if gs is not None and hasattr(gs, "z_goal"):
        try:
            saved_z_goal = gs.z_goal.detach().clone()
        except Exception:
            saved_z_goal = None

    # --- BEHAVIORAL G0/C1-C5 measurement: UNCHANGED from 866a (runs first, from
    #     the pristine post-P1 state, on the same RNG stream). ---
    p2_env = _build_env(sched_cfg, phase="p2", seed=None)
    metrics = _measure_866_style(
        agent, p2_env, device, sched_cfg, p2_eps, steps_per, condition
    )

    # --- C6 measurement-harness CORRECTION (866c): restore the post-P1 z_goal,
    #     then read the mechanistic z_goal via the scheduler's OWN contact-gated
    #     run_p2 (z_goal_norm_peak_max; n_decay_only == 0 confirms contact-gating)
    #     instead of 866a's decay-only unconditional-update mean. AVOIDANCE_ONLY
    #     has goal_state is None, so run_p2 returns 0.0 (all z_goal reads guarded).
    if saved_z_goal is not None and gs is not None:
        gs._z_goal = saved_z_goal
    p2m = scheduler.run_p2(agent, device)
    _progress("p2", p2_eps)

    zg.observe(agent)

    z_goal_peak_max = float(getattr(p2m, "z_goal_norm_peak_max", 0.0) or 0.0)
    z_goal_contact_peak = float(getattr(p2m, "z_goal_norm_at_contact_peak", 0.0) or 0.0)

    print(
        f"  [eval] {condition} seed={seed} resource_rate={metrics['resource_visit_rate']:.4f} "
        f"harm_rate={metrics['harm_rate']:.4f} entropy={metrics['policy_entropy']:.4f} "
        f"survival={metrics['survival_rate']:.4f} "
        f"zgoal_peak_max={z_goal_peak_max:.4f} zgoal_contact_peak={z_goal_contact_peak:.4f} "
        f"(866a_mean={metrics['zgoal_norm_mean_866a_style']:.4f})",
        flush=True,
    )
    print("verdict: PASS", flush=True)  # cell ran to completion; scientific verdict is aggregate-level

    row: Dict[str, Any] = {
        "seed": seed,
        "condition": condition,
        "stage0_z_goal_norm_peak": s0.z_goal_norm_peak,
        "stage0_aborted": s0.aborted,
        "stage0b_retention_ratio": s0b.retention_ratio,
        "stage0b_aborted": s0b.aborted,
        "p0_mean_episode_length": p0.mean_episode_length,
        "hazard_median_last_window": hz.median_last_window_episode_length,
        "hazard_survival_gate_passed": hz.survival_gate_passed,
        "p1_median_last_window": p1.median_last_window_episode_length,
        "p1_survival_gate_passed": p1.survival_gate_passed,
        # --- 866c C6 readout (contact-gated run_p2) ---
        "z_goal_norm_peak_max": z_goal_peak_max,
        "z_goal_norm_at_contact_peak": z_goal_contact_peak,
        "p2_n_decay_only": int(getattr(p2m, "n_decay_only_updates", 0) or 0),
        "p2_n_skipped_protected": int(getattr(p2m, "n_skipped_protected_updates", 0) or 0),
        "p2_num_contact_events": int(getattr(p2m, "num_contact_events", 0) or 0),
        "p2_contact_rate": float(getattr(p2m, "contact_rate", 0.0) or 0.0),
        "p2_contact_gated": bool(getattr(p2m, "contact_gated", False)),
    }
    row.update(metrics)
    return row


def _run_random_cell(seed: int, dry_run: bool) -> Dict[str, Any]:
    print(f"Seed {seed} Condition RANDOM", flush=True)
    device = torch.device("cpu")

    import random as _random
    _random.seed(seed)
    torch.manual_seed(seed)

    sched_cfg = build_scaffold_cfg(dry_run)
    steps_per = DRY_STEPS if dry_run else STEPS_PER_EP
    p2_eps = DRY_P2 if dry_run else P2_BUDGET
    env = _build_env(sched_cfg, phase="p2", seed=None)

    metrics = _measure_866_style(None, env, device, sched_cfg, p2_eps, steps_per, "RANDOM")

    print(
        f"  [train] seed={seed} RANDOM ep {p2_eps}/{p2_eps} phase=p2",
        flush=True,
    )
    print(
        f"  [eval] RANDOM seed={seed} resource_rate={metrics['resource_visit_rate']:.4f} "
        f"harm_rate={metrics['harm_rate']:.4f} entropy={metrics['policy_entropy']:.4f} "
        f"survival={metrics['survival_rate']:.4f}",
        flush=True,
    )
    print("verdict: PASS", flush=True)

    row: Dict[str, Any] = {"seed": seed, "condition": "RANDOM"}
    # RANDOM has no z_goal; carry the C6-readout keys as 0.0 so the aggregate loop
    # below never KeyErrors (RANDOM is not used by any C6 comparison).
    row["z_goal_norm_peak_max"] = 0.0
    row["z_goal_norm_at_contact_peak"] = 0.0
    row["p2_n_decay_only"] = 0
    row["p2_n_skipped_protected"] = 0
    row["p2_num_contact_events"] = 0
    row["p2_contact_rate"] = 0.0
    row["p2_contact_gated"] = False
    row.update(metrics)
    return row


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria (formulas identical to V3-EXQ-866/866a;
# ONLY the C6 readout source changed -- see module docstring)
# --------------------------------------------------------------------- #

def run(dry_run: bool = False) -> tuple:
    print(f"\n[{QUEUE_ID}] INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency "
          f"(scaffolded_sd054_onboarding; C6 measurement-harness correction)", flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[str, Dict[int, Dict]] = {"FULL": {}, "AVOIDANCE_ONLY": {}, "RANDOM": {}}

    for seed in SEEDS:
        for condition in CONDITIONS:
            config_slice = _cell_config_slice(condition, seed, dry_run)
            with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
                if condition == "RANDOM":
                    row = _run_random_cell(seed, dry_run=dry_run)
                else:
                    row = _run_trained_cell(condition, seed, zg, dry_run=dry_run)
                cell.stamp(row)
            arm_results.append(row)
            per_seed[condition][seed] = row

    g0_per_seed, c1_per_seed, c2_per_seed = [], [], []
    c3_per_seed, c4_per_seed, c5_per_seed, c6_per_seed = [], [], [], []

    for seed in SEEDS:
        full = per_seed["FULL"][seed]
        avoid = per_seed["AVOIDANCE_ONLY"][seed]
        rand = per_seed["RANDOM"][seed]

        g0_per_seed.append(full["resource_visit_rate"] >= rand["resource_visit_rate"] + G0_MARGIN)
        c1_per_seed.append(avoid["harm_rate"] <= full["harm_rate"] * C1_HARM_TOLERANCE)
        c2_per_seed.append(avoid["survival_rate"] >= full["survival_rate"] - C2_SURVIVAL_MARGIN)
        c3_per_seed.append(avoid["resource_visit_rate"] <= rand["resource_visit_rate"] + C3_MARGIN)
        c4_per_seed.append(full["resource_visit_rate"] >= avoid["resource_visit_rate"] + C4_MARGIN)
        c5_per_seed.append(avoid["policy_entropy"] <= full["policy_entropy"] - C5_ENTROPY_MARGIN)
        # C6 CORRECTED READOUT (866c): gate on the contact-gated run_p2 peak, not
        # the decay-only 866a mean. Thresholds unchanged.
        c6_per_seed.append(
            full["z_goal_norm_peak_max"] > C6_ZGOAL_ACTIVE_FLOOR
            and avoid["z_goal_norm_peak_max"] < C6_ZGOAL_INACTIVE_CEIL
        )

    g0_frac = _frac_seeds(g0_per_seed)
    c1_frac = _frac_seeds(c1_per_seed)
    c2_frac = _frac_seeds(c2_per_seed)
    c3_frac = _frac_seeds(c3_per_seed)
    c4_frac = _frac_seeds(c4_per_seed)
    c5_frac = _frac_seeds(c5_per_seed)
    c6_frac = _frac_seeds(c6_per_seed)

    threshold = MIN_SEEDS_PASS / len(SEEDS)
    g0_pass = g0_frac >= threshold
    c1_pass = c1_frac >= threshold
    c2_pass = c2_frac >= threshold
    c3_pass = c3_frac >= threshold
    c4_pass = c4_frac >= threshold
    c5_pass = c5_frac >= threshold
    c6_pass = c6_frac >= threshold

    non_degenerate = True
    degeneracy_reason = None

    if not g0_pass:
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "non_degenerate_precondition_unmet"
        non_degenerate = False
        degeneracy_reason = (
            "G0 non-degeneracy gate failed on the scaffolded_sd054_onboarding "
            "curriculum: FULL arm did not clear random-baseline resource-visit "
            "rate by the pre-registered margin on >= 2/3 seeds. This is the "
            "GAP-2 / Stage-H foraging-competence ceiling (routed separately); it "
            "gates before C6, so the corrected C6 readout is a clean mechanistic "
            "control, NOT evidence for/against INV-034/Q-021 on its own."
        )
        interpretation = (
            "G0 non-degeneracy gate FAILED (foraging-competence ceiling), so the "
            "run is non_contributory to INV-034/Q-021 -- as expected and as "
            "diagnosed for 866a. 866c's contribution is the CORRECTED C6 z_goal "
            "readout (contact-gated run_p2 peak, n_decay_only=0), which replaces "
            "866a's decay-only washout artifact: see z_goal_norm_peak_max_mean_* "
            "vs zgoal_norm_mean_866a_style_mean_* in metrics. G0 (foraging) is "
            "routed to the GAP-2 / Stage-H foraging-competence thread; it is a "
            "foraging/survival problem, not a z_goal-maintenance problem."
        )
    elif not (c1_pass and c2_pass):
        status = "FAIL"
        evidence_direction = "non_contributory"
        route_reason = "ablation_not_clean_competence_confound"
        interpretation = (
            "G0 passed but the ablation is not clean: AVOIDANCE_ONLY differs from "
            "FULL in harm-avoidance (C1) and/or survival (C2), so a resource-rate "
            "gap cannot be attributed to agency specifically vs general competence "
            "loss (C1) or a MECH-180-style death-not-quiescence confound (C2)."
        )
    else:
        all_pass = c3_pass and c4_pass and c5_pass and c6_pass
        status = "PASS" if all_pass else "FAIL"
        evidence_direction = "supports" if all_pass else "weakens"
        route_reason = "clean_ablation_scored"
        if all_pass:
            interpretation = (
                "INV-034 + Q-021 SUPPORTED: with harm-avoidance and survival held "
                "comparable between arms (C1/C2), the avoidance-only architecture "
                "collapses to a quiescent/flat policy at-or-below random baseline "
                "(C3) with reduced action diversity (C5), while the goal-maintaining "
                "FULL architecture sustains systematic resource-directed approach "
                "(C4) with z_goal genuinely engaged only in FULL (C6, contact-gated "
                "run_p2 peak). Goal maintenance is necessary for committed "
                "goal-directed agency beyond harm-avoidance alone."
            )
        else:
            interpretation = (
                "INV-034 + Q-021 WEAKENED: the ablation is clean (C1/C2 PASS) but "
                "the predicted quiescence/entropy/z_goal-engagement gap did not "
                "clear threshold on >= 2/3 seeds, on the scaffolded_sd054_onboarding "
                "curriculum with the corrected C6 readout."
            )

    metrics: Dict[str, float] = {
        "g0_frac_seeds": g0_frac, "c1_frac_seeds": c1_frac, "c2_frac_seeds": c2_frac,
        "c3_frac_seeds": c3_frac, "c4_frac_seeds": c4_frac, "c5_frac_seeds": c5_frac,
        "c6_frac_seeds": c6_frac,
        "g0_pass": 1.0 if g0_pass else 0.0, "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0, "c3_pass": 1.0 if c3_pass else 0.0,
        "c4_pass": 1.0 if c4_pass else 0.0, "c5_pass": 1.0 if c5_pass else 0.0,
        "c6_pass": 1.0 if c6_pass else 0.0,
    }
    for cond in CONDITIONS:
        rvr = [per_seed[cond][s]["resource_visit_rate"] for s in SEEDS]
        hr = [per_seed[cond][s]["harm_rate"] for s in SEEDS]
        ent = [per_seed[cond][s]["policy_entropy"] for s in SEEDS]
        ent_fresh = [per_seed[cond][s]["policy_entropy_fresh_select"] for s in SEEDS]
        surv = [per_seed[cond][s]["survival_rate"] for s in SEEDS]
        zg_peak = [per_seed[cond][s]["z_goal_norm_peak_max"] for s in SEEDS]
        zg_contact = [per_seed[cond][s]["z_goal_norm_at_contact_peak"] for s in SEEDS]
        zg_866a = [per_seed[cond][s].get("zgoal_norm_mean_866a_style", 0.0) for s in SEEDS]
        metrics[f"resource_visit_rate_mean_{cond}"] = sum(rvr) / len(rvr)
        metrics[f"harm_rate_mean_{cond}"] = sum(hr) / len(hr)
        metrics[f"policy_entropy_mean_{cond}"] = sum(ent) / len(ent)
        metrics[f"policy_entropy_fresh_select_mean_{cond}"] = sum(ent_fresh) / len(ent_fresh)
        metrics[f"survival_rate_mean_{cond}"] = sum(surv) / len(surv)
        # 866c corrected C6 readout (primary) + contact-peak + 866a bug signature.
        metrics[f"z_goal_norm_peak_max_mean_{cond}"] = sum(zg_peak) / len(zg_peak)
        metrics[f"z_goal_norm_at_contact_peak_mean_{cond}"] = sum(zg_contact) / len(zg_contact)
        metrics[f"zgoal_norm_mean_866a_style_mean_{cond}"] = sum(zg_866a) / len(zg_866a)

    evidence_direction_per_claim = {"INV-034": evidence_direction, "Q-021": evidence_direction}

    summary_markdown = (
        f"# {QUEUE_ID} -- INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency "
        f"(onboarded; C6 measurement-harness correction)\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Route reason:** {route_reason}\n"
        f"**Supersedes:** {SUPERSEDES}\n"
        f"**Claims:** INV-034, Q-021\n\n"
        f"## C6 readout: 866c (contact-gated run_p2 peak) vs 866a (decay-only mean)\n\n"
        f"| Condition | z_goal_norm_peak_max (866c C6) | zgoal_norm_mean (866a bug) |\n"
        f"|---|---|---|\n"
        f"| FULL | {metrics.get('z_goal_norm_peak_max_mean_FULL', 0.0):.4f} | "
        f"{metrics.get('zgoal_norm_mean_866a_style_mean_FULL', 0.0):.4f} |\n"
        f"| AVOIDANCE_ONLY | {metrics.get('z_goal_norm_peak_max_mean_AVOIDANCE_ONLY', 0.0):.4f} | "
        f"{metrics.get('zgoal_norm_mean_866a_style_mean_AVOIDANCE_ONLY', 0.0):.4f} |\n\n"
        f"## Gates\n\n"
        f"| Gate | Frac seeds | Pass |\n|---|---|---|\n"
        f"| G0 non-degeneracy (foraging) | {g0_frac:.2f} | {g0_pass} |\n"
        f"| C1 harm parity | {c1_frac:.2f} | {c1_pass} |\n"
        f"| C2 survival parity | {c2_frac:.2f} | {c2_pass} |\n"
        f"| C3 quiescence (avoidance-only flat) | {c3_frac:.2f} | {c3_pass} |\n"
        f"| C4 approach restored (FULL > avoidance) | {c4_frac:.2f} | {c4_pass} |\n"
        f"| C5 entropy signature | {c5_frac:.2f} | {c5_pass} |\n"
        f"| C6 z_goal mechanistic (contact-gated peak) | {c6_frac:.2f} | {c6_pass} |\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "per_seed_results": per_seed,
        "c6_readout_source": "scheduler_run_p2_contact_gated_peak_max",
        "fatal_error_count": 0,
    }
    if not non_degenerate:
        result["non_degenerate"] = False
        result["degeneracy_reason"] = degeneracy_reason

    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "stage0_budget": STAGE0_BUDGET,
        "stage0b_budget": STAGE0B_BUDGET,
        "p0_budget": P0_BUDGET,
        "hazard_budget": HAZARD_STAGE_BUDGET,
        "p1_budget": P1_BUDGET,
        "p2_budget": P2_BUDGET,
        "steps_per_ep": STEPS_PER_EP,
        "p0_num_hazards": P0_NUM_HAZARDS,
        "hazard_stage_regime": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                                 HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR, N_RESOURCE_TYPES],
        "drive_weight": DRIVE_WEIGHT,
        "p2_hfa_guard": P2_HFA_GUARD,
        "c6_readout_source": "scheduler_run_p2_contact_gated_peak_max",
    }

    # write_flat_manifest stamps the always-record core internally (arm_results
    # is already assembled, so substrate_hash hoists from the per-cell
    # arm_fingerprints rather than being recomputed fresh).
    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
