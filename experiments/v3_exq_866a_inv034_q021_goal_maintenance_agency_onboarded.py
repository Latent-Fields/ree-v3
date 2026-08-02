#!/opt/local/bin/python3
"""
V3-EXQ-866a -- INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency,
re-run on the scaffolded_sd054_onboarding curriculum.

Supersedes V3-EXQ-866 (confirmed failure_autopsy_V3-EXQ-866_2026-08-02,
autopsied session infallible-villani-f280fa, user-approved 2026-08-02).
866's G0 non-degeneracy gate FAILED: on the lightweight direct-REEConfig
harness, the FULL (approach+avoidance) arm's resource_visit_rate (0.0046)
came in BELOW the RANDOM baseline (0.0217) -- the trained agent foraged
*worse* than chance. C1/C2/C3/C6 all passed cleanly (harm parity, survival
parity, quiescence, z_goal mechanistic check), so this is a competence-floor
issue in the harness, not evidence against INV-034/Q-021 -- it mirrors the
already-documented EXQ-072b failure mode. The autopsy's own routing (per
IGW-222 DESIGN.md 5.8, "Option B") is to re-run on the scaffolded_sd054_
onboarding curriculum (experiments/scaffolded_sd054_onboarding.py), which is
the substrate that actually cleared goal_pipeline GAP-2's contact-rate
ceiling (V3-EXQ-514o PASS, 2026-06-15) and, on this exact validated
configuration (V3-EXQ-603q ARM_BASE_IA_ONLY), produces a discriminative harm
landscape and an unsaturated Stage-H survival baseline (37.7/200 steps).
This script is that re-run: the SAME three-condition design and the SAME
pre-registered G0-C6 gates as 866, ported onto the scaffolded curriculum
instead of the lightweight harness.

Claims: INV-034 (primary), Q-021 (companion open question -- unchanged from
866; this design still directly implements Q-021's own pre-specified
two-arm ablation).

WHY THE SCAFFOLDED CURRICULUM, NOT A TWEAK TO THE LIGHTWEIGHT HARNESS
----------------------------------------------------------------------
866's own three-phase schedule (P0 warmup 100ep / P1 full-pipeline 150ep /
P2 frozen-eval 50ep, flat 10x10 grid, no isolated hazard-avoidance stage, no
harm-pathway training) is precisely the design the 603i/603k/603q autopsy
chain diagnosed as insufficient: without dedicated harm-pathway training
(E3.harm_eval_head + z_world discriminativeness), the harm cost that scores
every candidate trajectory is a near-constant random landscape, so the
agent cannot learn to navigate toward resources at all -- it dies or
wanders below chance regardless of whether the goal/wanting pathway is
enabled. More P0/P1 budget on that harness cannot fix a missing-mechanism
gap (nothing trains the head). scaffolded_sd054_onboarding.py's validated
803q-lineage configuration is the substrate that already closes this: a
graduated Stage-0 forced-feed nursery (z_goal FORMS unconditionally),
Stage-0b protected consolidation (the fragile Stage-0 trace is not washed
out by decay-only updates), an isolated Stage-H hazard-avoidance leg with
harm-pathway training (E3.harm_eval_head + z_harm sensory/affective +
E2_harm_s all trained, per 603k/603q), then P1 anneal into the full target
env and P2 frozen-eval measurement. This is a harness/environment
escalation per the autopsy's own diagnosis, not a substrate build --
scaffolded_sd054_onboarding.py already exists and is validated.

DESIGN: SAME THREE CONDITIONS, SAME GATES, DIFFERENT HARNESS
--------------------------------------------------------------
  A. FULL          -- full goal/wanting pathway live: z_goal_enabled,
                       MECH-295 liking bridge, MECH-307 conjunction,
                       SD-057 incentive-token bank + cue-recall, drive_weight
                       and goal-seeding calibration all on (the validated
                       V3-EXQ-603q ARM_BASE_IA_ONLY agent config).
  B. AVOIDANCE_ONLY -- the SAME harm-avoidance / NoGo channel as FULL
                       (identical harm stream, affective harm stream,
                       E2_harm_s forward, PAG freeze gate, instrumental
                       avoidance gate, escape/safety terrain -- all
                       unchanged), but z_goal_enabled=False, drive_weight=0,
                       MECH-295/307 off, incentive bank + cue-recall off.
                       agent.goal_state is None for this arm, so every
                       goal-pipeline call the scheduler makes (Stage-0
                       forced-feed, update_z_goal at P1/P2) safely no-ops or
                       aborts cleanly (verified against live source,
                       ree_core/agent.py:9544 `update_z_goal` returns
                       immediately when `self.goal_state is None`;
                       scaffolded_sd054_onboarding.py's run_stage0_nursery /
                       run_stage0b_consolidation abort cleanly with
                       abort_reason="goal_state_none_set_z_goal_enabled_true"
                       rather than raising). BOTH trained arms share the
                       identical ScaffoldedSD054OnboardingConfig (same env
                       kwargs at every phase, same curriculum budgets, same
                       harm-pathway training) -- only the REEConfig differs,
                       which is what makes the ablation single-variable by
                       construction rather than by inspection.
  C. RANDOM        -- uniform action selection, no training, no curriculum
                       -- measured on the SAME P2-phase target env
                       (_build_env(cfg, phase="p2")) that FULL/AVOIDANCE_ONLY
                       are measured on, for episode/step parity. Distinguishes
                       "B is actively quiescent" from "B just can't find
                       resources" exactly as in 866.

DIVERGENCE FROM 866's LITERAL IMPLEMENTATION, AND WHY IT IS THE RIGHT ONE:
866 additionally used a separate "benefit_eval" head (benefit_eval_enabled /
benefit_weight / compute_benefit_eval_loss) as part of its FULL arm's
wanting pathway. scaffolded_sd054_onboarding.py's own training loop
(_train_episode) never calls compute_benefit_eval_loss, and the validated
603q lineage config does not use it -- the scaffold's own native wanting
mechanism is z_goal/GoalState + MECH-295 liking bridge + SD-057 incentive
bank + cue-recall, which IS what this script severs for AVOIDANCE_ONLY.
Reusing 866's separate untested benefit_eval head on top of a curriculum
that never trains it would add an inert, non-representative pathway; the
substrate's own validated wanting mechanism is the correct thing to ablate.

P2 MEASUREMENT GUARD (found necessary during pre-queue verification,
2026-08-02): the class default `scaffold_p2_hazard_food_attraction_guard`
(-1.0, i.e. no guard) leaves the P2 measurement env at
`scaffold_p2_hazard_food_attraction=0.7`, which the scheduler's own docstring
already warns "can suppress contact and make z_goal=0 uninterpretable". A
single-seed, reduced-budget pre-queue check with the guard left at its class
default produced near-zero resource_visit_rate for ALL THREE conditions
(FULL 0.0045, AVOIDANCE_ONLY 0.0058, RANDOM 0.0058) -- the hard P2 env was
suppressing contact for a random policy too, so the measurement window
itself was non-discriminating, not the substrate. A grep of every landed
scaffolded_sd054_onboarding consumer (v3_exq_603*/793*/812*.py) confirms this
guard is set to 0.3 near-universally in practice -- none use the class
default. This script sets `P2_HFA_GUARD = 0.3` accordingly (see
build_scaffold_cfg), matching established practice rather than leaving a
degenerate class default in place.

PRE-REGISTERED GATES (identical formulas/margins to V3-EXQ-866; in order,
G0 must pass first or the run self-routes to non_contributory -- never a
false weakens of INV-034/Q-021):
  G0 non-degeneracy: resource_visit_rate_FULL >= resource_visit_rate_RANDOM
      + 0.05 on >= 2/3 seeds.
  C1 harm-avoidance parity: harm_rate_AVOIDANCE_ONLY <= harm_rate_FULL * 1.15
      on >= 2/3 seeds.
  C2 survival parity (MECH-180 death-vs-quiescence guard):
      episode_survival_rate_AVOIDANCE_ONLY >= episode_survival_rate_FULL - 0.1
      on >= 2/3 seeds.
  C3 quiescence/flatness: resource_visit_rate_AVOIDANCE_ONLY <=
      resource_visit_rate_RANDOM + 0.05 on >= 2/3 seeds.
  C4 approach restored by goal maintenance: resource_visit_rate_FULL >=
      resource_visit_rate_AVOIDANCE_ONLY + 0.10 on >= 2/3 seeds.
  C5 entropy signature: policy_entropy_AVOIDANCE_ONLY <=
      policy_entropy_FULL - 0.10 on >= 2/3 seeds.
  C6 mechanistic check: z_goal_norm_mean_FULL > 0.4 AND
      z_goal_norm_mean_AVOIDANCE_ONLY < 0.1 on >= 2/3 seeds.

PASS (supports INV-034 + Q-021) = G0 AND C1 AND C2 AND C3 AND C4 AND C5 AND C6.
G0 fails -> non_contributory (substrate/env non-degeneracy issue; this is
  now the SECOND-generation harness, so a further G0 failure here would be
  a genuine escalation-exhausted finding, not a routine re-queue).
C1/C2 fail (with G0 passing) -> mixed/non_contributory (ablation not clean).
G0+C1+C2 pass but C3/C4/C5/C6 don't show the predicted gap -> weakens.

Biological basis (unchanged from 866): Bariselli 2018 (PMID 29481617) D1/D2
pathways evaluate the SAME proposals; without D1 (Go/wanting) only D2
(NoGo/avoidance) constrains selection -> quiescent default. Barch & Dowd
2010 (PMID 20868638) "wanting" (prospective) vs "liking" (reactive);
avolition in schizophrenia = intact liking, absent wanting.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler, same as every other importer).

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
EXPERIMENT_TYPE = "v3_exq_866a_inv034_q021_goal_maintenance_agency_onboarded"
QUEUE_ID = "V3-EXQ-866a"
SUPERSEDES = "V3-EXQ-866"
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
#     consumers -- grep of experiments/v3_exq_603*/793*/812*.py shows every
#     landed measurement-phase script sets this to 0.3, none use the class
#     default -1.0/"no guard"/hfa=0.7). Without it the P2 window's hazard_food_
#     attraction stays at the class default 0.7, which the module's own
#     docstring warns "can suppress contact and make z_goal=0 uninterpretable"
#     -- confirmed empirically here: a single-seed reduced-budget check with
#     this left at -1.0 produced near-zero resource_visit_rate for EVERY
#     condition (FULL 0.0045, AVOIDANCE_ONLY 0.0058, RANDOM 0.0058) -- i.e. the
#     hard P2 env was suppressing contact for a random policy too, not
#     discriminating agency at all. P2_HFA_GUARD=0.3 admits foraging contact
#     during measurement while still being harder than P0/P1's lower values.
P2_HFA_GUARD = 0.3

# --- Encoder / latent dims (mirror 603q exactly) ----------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
ALPHA_WORLD = 0.9

# Pre-registered thresholds -- IDENTICAL to V3-EXQ-866 (see module docstring).
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
# Scaffold + agent config builders
# --------------------------------------------------------------------- #

def build_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    """The scheduler config shared by BOTH trained arms (FULL and
    AVOIDANCE_ONLY). Identical env kwargs and curriculum budgets at every
    phase -- only the REEConfig each arm's agent is built with differs
    (build_agent_config below). This is what makes the ablation
    single-variable by construction."""
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
        # Harm stream + harm-pathway training: without these the E3 harm cost
        # is a random constant (603i) and survival is unreachable at any
        # budget -- this is precisely the gap that made 866's lightweight
        # harness non-degenerate.
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
    harm/avoidance-channel flag (the C1/C2 control per 866's own invariant);
    only the wanting/goal-pathway flags differ."""
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
# Metrics helpers (identical formulas to V3-EXQ-866)
# --------------------------------------------------------------------- #

def _action_entropy(action_counts: List[int]) -> float:
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _frac_seeds(pred_per_seed: List[bool]) -> float:
    return sum(1 for p in pred_per_seed if p) / max(1, len(pred_per_seed))


# --------------------------------------------------------------------- #
# 866-style P2 measurement, on the scaffold's own P2-phase target env
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
    """Frozen-policy (or random) measurement using V3-EXQ-866's own metric
    definitions (transition_type-based resource_visit_rate / harm_rate,
    action-histogram policy_entropy, z_goal norm), run on whichever env is
    passed in. `agent=None` -> uniform random action selection (the RANDOM
    condition); otherwise the agent is stepped in eval mode with the SAME
    frozen-policy protocol scaffolded_sd054_onboarding.py's own run_p2 uses
    (temperature 0.5, update_z_goal called every step so FULL's goal stream
    stays live during measurement -- a no-op for AVOIDANCE_ONLY since
    agent.goal_state is None)."""
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
        "zgoal_norm_mean": zgoal_norm_mean,
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
    p2_env = _build_env(sched_cfg, phase="p2", seed=None)
    metrics = _measure_866_style(
        agent, p2_env, device, sched_cfg, p2_eps, steps_per, condition
    )
    _progress("p2", p2_eps)

    zg.observe(agent)

    print(
        f"  [eval] {condition} seed={seed} resource_rate={metrics['resource_visit_rate']:.4f} "
        f"harm_rate={metrics['harm_rate']:.4f} entropy={metrics['policy_entropy']:.4f} "
        f"survival={metrics['survival_rate']:.4f} zgoal_norm={metrics['zgoal_norm_mean']:.4f}",
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
    row.update(metrics)
    return row


# --------------------------------------------------------------------- #
# Aggregate + acceptance criteria (identical formulas to V3-EXQ-866)
# --------------------------------------------------------------------- #

def run(dry_run: bool = False) -> tuple:
    print(f"\n[{QUEUE_ID}] INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency "
          f"(scaffolded_sd054_onboarding re-run)", flush=True)

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
        c6_per_seed.append(
            full["zgoal_norm_mean"] > C6_ZGOAL_ACTIVE_FLOOR
            and avoid["zgoal_norm_mean"] < C6_ZGOAL_INACTIVE_CEIL
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
            "G0 non-degeneracy gate failed AGAIN on the scaffolded_sd054_onboarding "
            "curriculum: FULL arm did not clear random-baseline resource-visit rate "
            "by the pre-registered margin on >= 2/3 seeds."
        )
        interpretation = (
            "G0 non-degeneracy gate FAILED on the second-generation (scaffolded) "
            "harness. This is a genuine escalation-exhausted finding, not a routine "
            "re-queue: the validated 603q-lineage curriculum (Stage-0 forced-feed, "
            "Stage-0b consolidation, isolated Stage-H hazard-avoidance with harm-"
            "pathway training, P1 anneal, P2 frozen-eval) still could not get the "
            "FULL arm to forage above the RANDOM baseline. Still NOT evidence "
            "against INV-034/Q-021 -- it means the ceiling is deeper than the "
            "866-diagnosed harness gap. Route to a fresh substrate-readiness "
            "diagnostic on this exact configuration before a further lettered "
            "iteration."
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
                "(C4) with z_goal genuinely engaged only in FULL (C6), on the "
                "scaffolded_sd054_onboarding curriculum. Goal maintenance is "
                "necessary for the agent to exercise committed goal-directed agency "
                "beyond harm-avoidance alone."
            )
        else:
            interpretation = (
                "INV-034 + Q-021 WEAKENED: the ablation is clean (harm-avoidance/"
                "survival held comparable, C1/C2 PASS) but the predicted "
                "quiescence/entropy/z_goal-engagement gap did not clear threshold "
                "on >= 2/3 seeds, on the scaffolded_sd054_onboarding curriculum. "
                "Genuine evidence against the flatness prediction on this substrate "
                "configuration."
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
        zg_vals = [per_seed[cond][s]["zgoal_norm_mean"] for s in SEEDS]
        metrics[f"resource_visit_rate_mean_{cond}"] = sum(rvr) / len(rvr)
        metrics[f"harm_rate_mean_{cond}"] = sum(hr) / len(hr)
        metrics[f"policy_entropy_mean_{cond}"] = sum(ent) / len(ent)
        metrics[f"policy_entropy_fresh_select_mean_{cond}"] = sum(ent_fresh) / len(ent_fresh)
        metrics[f"survival_rate_mean_{cond}"] = sum(surv) / len(surv)
        metrics[f"zgoal_norm_mean_{cond}"] = sum(zg_vals) / len(zg_vals)

    evidence_direction_per_claim = {"INV-034": evidence_direction, "Q-021": evidence_direction}

    summary_markdown = (
        f"# {QUEUE_ID} -- INV-034 / Q-021 Goal-Maintenance-Necessary-for-Agency "
        f"(onboarded)\n\n"
        f"**Status:** {status}  **Evidence direction:** {evidence_direction}\n"
        f"**Route reason:** {route_reason}\n"
        f"**Supersedes:** {SUPERSEDES}\n"
        f"**Claims:** INV-034, Q-021\n\n"
        f"## Gates\n\n"
        f"| Gate | Frac seeds | Pass |\n|---|---|---|\n"
        f"| G0 non-degeneracy | {g0_frac:.2f} | {g0_pass} |\n"
        f"| C1 harm parity | {c1_frac:.2f} | {c1_pass} |\n"
        f"| C2 survival parity | {c2_frac:.2f} | {c2_pass} |\n"
        f"| C3 quiescence (avoidance-only flat) | {c3_frac:.2f} | {c3_pass} |\n"
        f"| C4 approach restored (FULL > avoidance) | {c4_frac:.2f} | {c4_pass} |\n"
        f"| C5 entropy signature | {c5_frac:.2f} | {c5_pass} |\n"
        f"| C6 z_goal mechanistic check | {c6_frac:.2f} | {c6_pass} |\n\n"
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
