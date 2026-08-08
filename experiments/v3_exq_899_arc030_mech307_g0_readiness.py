#!/opt/local/bin/python3
"""
V3-EXQ-899 -- ARC-030 / MECH-307 G0-READINESS DIAGNOSTIC.

experiment_purpose: diagnostic  (NOT scored evidence; excluded from confidence)
claim_ids: []                   (a substrate/curriculum READINESS gate -- tags no
            claim as evidence; the claims it GATES are ARC-030, Q-021, INV-034,
            named here and in the queue-entry note, not in claim_ids -- exactly
            the 603k "claim_ids=[], diagnostic, non_contributory" precedent)
gates_claims (informational only): ARC-030 (primary), Q-021, INV-034
regression/readiness_check_of: V3-EXQ-866a  (does NOT supersede it)
substrate_gate_shared_with: substrate_queue.json `scaffolded-curriculum-hazard-rebalance`

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler, same as every other importer -- inherited
verbatim from V3-EXQ-866a).

WHY THIS RUNS
-------------
MECH-307 (the D1/approach conjunction / incentive-salience master flag) was found
2026-08-07 to be UNREACHABLE through `REEConfig.from_dims()` -- all 12 of its
params were absent from the from_dims signature and silently swallowed by
**kwargs, so all 84 drivers requesting it via from_dims ran with it entirely OFF.
The scaffolded lineage ARC-030's retest is built on -- V3-EXQ-603q / 866a / 866b --
is in that affected set, so every behavioural run in it had MECH-307 OFF the whole
time (see evidence/planning/mech307_from_dims_unreachable_2026-08-07.md).

That from_dims wiring gap is now FIXED (chip-20260807-mech307-fromdims-wiring, DONE,
landed ree-v3 d63f13b7 + 2c682a91; contract test_mech307_from_dims_wiring.py, 19
tests). MECH-307 is now GENUINELY reachable via from_dims for the first time --
verified in-session before authoring this script: on origin/main,
`REEConfig.from_dims(..., use_mech307_conjunction=True)` flips all four conjunction
sub-flags (`use_mech307_split_surprise`, `use_mech307_schema_multichannel`,
`use_mech307_predicted_location_write`) to True; passing False keeps them False;
and the rest of the goal pathway (z_goal_enabled / drive_weight / e1_goal_conditioned
/ use_mech295_liking_bridge / use_incentive_token_bank / use_cue_recall) reaches
`cfg.goal.*` as before -- so MECH-307 is the SINGLE variable between the two FULL
arms below, by construction.

WHAT THIS IS -- A READINESS GATE, NOT ARC-030's DISCRIMINATIVE RETEST
--------------------------------------------------------------------
Before ARC-030's real COMBINED-vs-NOGO_ONLY discriminative retest (Go-channel
ablation; design in ARC-030's own what_would_answer) can be queued, the shared
non-degeneracy precondition (G0) must be shown to clear on the substrate MECH-307
is now GENUINELY on. This is not a formality:

  * V3-EXQ-866a already FAILED G0 on this exact scaffolded curriculum with
    MECH-307 OFF (FULL resource_visit_rate 0.0033 < RANDOM 0.0103, 2026-08-03),
    and its z_goal decayed from a Stage-0 peak ~0.5 to a P2 mean of only 0.12,
    below the 0.4 C6 floor.
  * V3-EXQ-866b (fresh, commit-verified 603q re-run, PASS 2026-08-03) confirmed
    the SUBSTRATE is sound -- 866a's G0 failure is CURRICULUM/harness (the
    intervening hazard-only stages starve goal maintenance before the P2 window),
    not a substrate regression. That gap is tracked as the substrate_queue stub
    `scaffolded-curriculum-hazard-rebalance` (unblocks Q-021/INV-034/ARC-030).

Turning MECH-307 GENUINELY on for the first time is itself a change to this
delicate baseline. It could (a) rescue G0 (the incentive-salience conjunction
lifts goal-directed foraging above chance), (b) leave G0 unchanged (the curriculum
gap dominates), or (c) PERTURB it downward (MECH-307-on, though mechanically
correct, destabilises a config that was tuned with it off). This diagnostic
measures which -- with a MECH-307-OFF arm run in the SAME harness / SAME seeds so
the difference is attributable to MECH-307 rather than to any harness drift vs the
separate 866a run.

DESIGN: THREE CONDITIONS (a MECH-307 A/B on the 866a FULL config + RANDOM)
--------------------------------------------------------------------------
  A. FULL_M307_ON  -- V3-EXQ-866a's FULL agent config (full goal/wanting pathway
                      + identical NoGo channel), with use_mech307_conjunction=True
                      now GENUINELY reaching the substrate.
  B. FULL_M307_OFF -- BYTE-IDENTICAL to A except use_mech307_conjunction=False.
                      This reproduces V3-EXQ-866a's ACTUAL FULL arm (which ran
                      with MECH-307 silently off) IN THIS HARNESS, as the paired
                      "before" control. MECH-307 is the single differing variable.
  C. RANDOM        -- uniform action selection, no training, measured on the SAME
                      P2-phase target env, for the G0 non-degeneracy denominator
                      (identical to 866a's RANDOM arm).

NOTE the AVOIDANCE_ONLY (Go-channel ablation) arm of 866a is deliberately NOT here:
that is the shape of the Q-021/INV-034/ARC-030 DISCRIMINATIVE ablation, which this
readiness gate is a precondition FOR, not an instance OF.

PRE-REGISTERED READINESS GATE (the load-bearing criterion):
  G0_ON non-degeneracy: resource_visit_rate_FULL_M307_ON
      >= resource_visit_rate_RANDOM + 0.05 on >= 2/3 seeds.
      (identical formula/margin to 866a's G0, applied to the MECH-307-ON arm)

MEASURABILITY PRECONDITIONS (is the G0 measurement even TAKEABLE? -- self-route
below-floor to substrate_not_ready_requeue, never a claim verdict):
  P1 curriculum_reached_p2 : FULL_M307_ON completed the curriculum through the P2
      measurement window (total_steps > 0) on >= 2/3 seeds.
  P2 p2_window_admits_contact : RANDOM resource_visit_rate > 0.005 on >= 2/3 seeds
      -- the SAME statistic G0 routes on, measured on the positive control (a
      random policy on the P2 target env). If a random policy cannot contact
      resources at all, the P2 window is non-discriminating and a FULL<RANDOM
      reading is a dead-window artifact, not a foraging finding (the 866a
      pre-queue P2_HFA_GUARD=-1.0 degeneracy -- guarded here with P2_HFA_GUARD=0.3,
      matching every landed scaffolded_sd054_onboarding consumer).

NON-GATING DIAGNOSTICS recorded for the reader / the eventual /failure-autopsy:
  * G0_OFF (reproduction): resource_visit_rate_FULL_M307_OFF vs RANDOM+0.05 --
    expected to FAIL, reproducing 866a in this harness.
  * z_goal survival per FULL arm: zgoal_norm_mean_FULL_M307_ON /
    zgoal_norm_mean_FULL_M307_OFF vs the 0.4 C6 floor (866a's OFF-equivalent was
    ~0.12). Localises whether MECH-307-on helps goal maintenance survive to P2.
  * MECH-307 A/B deltas: d_resource_visit_rate = ON - OFF and d_zgoal = ON - OFF,
    per seed + mean -- the attributable MECH-307 effect on this baseline.
  * Stage-boundary survival localisation (hazard_median_last_window, p1 window)
    per 866a's own row shape.

ROUTING (this is a diagnostic; claim_ids=[] so it weights NO governance score):
  preconditions unmet            -> outcome FAIL, label substrate_not_ready_requeue
                                    (the G0 measurement was not takeable).
  preconditions met + G0_ON pass -> outcome PASS, label
                                    readiness_pass_arc030_retest_unblocked
                                    (ARC-030's COMBINED-vs-NOGO_ONLY retest is
                                    unblocked; see ARC-030 what_would_answer).
  preconditions met + G0_ON fail -> outcome FAIL, label
                                    readiness_fail_curriculum_gate_blocks_retest
                                    (route to the `scaffolded-curriculum-hazard-
                                    rebalance` substrate fix FIRST, then re-run
                                    this readiness -- do NOT force through to the
                                    discriminative retest). If G0_OFF would have
                                    passed while G0_ON fails, the additional
                                    `mech307_perturbs_baseline` flag is set: that
                                    is the specific (c) outcome above.

DV-SYMMETRY (per-arm, per the queue-experiment skill's DV-symmetry-invariance rule):
  * FULL_M307_ON vs FULL_M307_OFF -- manipulation = MECH-307 conjunction, a
    GOAL-CONDITIONED per-candidate incentive-salience modulation of z_goal / z_beta
    (ree_core; NOT a broadcast scalar added uniformly across candidates). Both DVs
    are sensitive to it: resource_visit_rate is a function of the argmax action
    sequence, and MECH-307 modulates the per-candidate goal-proximity term that
    enters E3 candidate scoring, so it CAN move the argmax; zgoal_norm is modulated
    directly. The manipulation is NOT invariant under either DV's symmetry group.
  * RANDOM -- no manipulation (uniform reference for the G0 denominator); not an
    arm under a symmetry concern.

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
EXPERIMENT_TYPE = "v3_exq_899_arc030_mech307_g0_readiness"
QUEUE_ID = "V3-EXQ-899"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []            # readiness gate -- tags NO claim as evidence
GATES_CLAIMS = ["ARC-030", "Q-021", "INV-034"]  # informational only
EXPERIMENT_PURPOSE = "diagnostic"
READINESS_CHECK_OF = "V3-EXQ-866a"

SEEDS = [42, 43, 44]                  # same as 866a for direct comparability
CONDITIONS = ["FULL_M307_ON", "FULL_M307_OFF", "RANDOM"]

# The two readiness preconditions below (curriculum_reached_p2,
# p2_window_admits_contact) are plain MEASURABILITY guards, NOT the
# known-degenerate-signature anchors assert_anchor_reachable exists for (778d:
# "null_zero control reproduces a railed signature", scored by a hand-written
# predicate NARROWER than the degeneracy, unmeetable-by-construction). Their
# positive controls are reachable by construction and are demonstrably CLEARED by
# the reference run V3-EXQ-866a: RANDOM resource_visit_rate came in at 0.0103 and
# 0.0217, both well above the 0.005 window floor, and the FULL arm reached P2 with
# total_steps>0. Neither predicate is narrower than the state it anchors, so the
# 778d false-negative-forever defect cannot arise; the guard's frozen-degenerate-
# fixture machinery targets the opposite (known-degenerate) case and does not fit.
ANCHOR_REACHABILITY_EXEMPT = (
    "readiness preconditions are measurability guards (reached-P2, random-forages-"
    "above-floor), reachable by construction and cleared by the V3-EXQ-866a "
    "reference (RANDOM rvr 0.0103/0.0217 > 0.005 floor); not a known-degenerate-"
    "signature anchor"
)

# --- Curriculum budgets (IDENTICAL to V3-EXQ-866a) --------------------------
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

# --- Stage-H regime (866a / 603q amend-validated anchor) --------------------
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 866a) -------
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

# --- P2 measurement guard (near-universal across scaffolded consumers; see
#     the extended justification in V3-EXQ-866a. Left at the class default
#     (-1.0) the P2 window suppresses contact for a RANDOM policy too, which
#     collapses the very G0 non-degeneracy denominator this readiness gate
#     depends on -- so 0.3 here is load-bearing for P2 (p2_window_admits_contact),
#     not cosmetic.
P2_HFA_GUARD = 0.3

# --- Encoder / latent dims (mirror 866a exactly) ----------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
ALPHA_WORLD = 0.9

# Pre-registered thresholds.
MIN_SEEDS_PASS = 2                    # of 3 -- ">= 2/3 seeds"
G0_MARGIN = 0.05                      # identical to 866a G0
C6_ZGOAL_FLOOR = 0.4                  # 866a C6 z_goal floor (non-gating here)
P2_WINDOW_CONTACT_FLOOR = 0.005       # RANDOM must forage above this for a live window
SEED_THRESHOLD = MIN_SEEDS_PASS / len(SEEDS)


# --------------------------------------------------------------------- #
# Scaffold + agent config builders
# --------------------------------------------------------------------- #

def build_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    """The scheduler config shared by BOTH trained arms (FULL_M307_ON and
    FULL_M307_OFF). Identical env kwargs and curriculum budgets at every phase --
    only the REEConfig each arm's agent is built with differs (build_agent_config),
    and there the SINGLE difference is use_mech307_conjunction. This is what makes
    the MECH-307 A/B single-variable by construction. IDENTICAL to 866a's
    build_scaffold_cfg."""
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_P1, DRY_P2, DRY_STEPS)
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, STEPS_PER_EP)

    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_strict_goal_isolation=False,
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
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
    )
    if steps < 75:
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def build_agent_config(env, condition: str) -> REEConfig:
    """REEConfig for the given condition. BOTH FULL arms share EVERY flag -- the
    full goal/wanting pathway AND the harm/avoidance channel (exactly 866a's FULL
    arm) -- and differ ONLY in use_mech307_conjunction. That single-variable
    construction is the whole point of this A/B."""
    m307 = (condition == "FULL_M307_ON")
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        # NoGo / harm-avoidance channel -- IDENTICAL in both FULL arms.
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
        e2_action_contrastive_enabled=True,
        # Go / wanting channel -- FULL in BOTH arms.
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        e1_goal_conditioned=True,
        use_mech295_liking_bridge=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        # THE SINGLE A/B VARIABLE -- genuinely reachable via from_dims since d63f13b7.
        use_mech307_conjunction=m307,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


# --------------------------------------------------------------------- #
# Metrics helpers (identical formulas to V3-EXQ-866a)
# --------------------------------------------------------------------- #

def _action_entropy(action_counts: List[int]) -> float:
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _frac_seeds(pred_per_seed: List[bool]) -> float:
    return sum(1 for p in pred_per_seed if p) / max(1, len(pred_per_seed))


# --------------------------------------------------------------------- #
# 866a-style P2 measurement, on the scaffold's own P2-phase target env
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
    """Frozen-policy (or random) measurement using V3-EXQ-866a's own metric
    definitions. `agent=None` -> uniform random (the RANDOM condition). z_goal norm
    is recorded for BOTH FULL arms (any condition with a live goal_state), so the
    ON-vs-OFF z_goal survival comparison is available -- 866a recorded it for its
    single FULL arm only."""
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
                # Record z_goal for BOTH FULL arms (goal_state present) -- the
                # ON-vs-OFF survival comparison is the point of this readiness gate.
                if agent.goal_state is not None:
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
# Cells
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
        "use_mech307_conjunction": (condition == "FULL_M307_ON"),
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

    print(f"  [train] seed={seed} RANDOM ep {p2_eps}/{p2_eps} phase=p2", flush=True)
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
# Aggregate + readiness gate
# --------------------------------------------------------------------- #

def run(dry_run: bool = False) -> tuple:
    print(f"\n[{QUEUE_ID}] ARC-030 / MECH-307 G0-readiness diagnostic "
          f"(scaffolded_sd054_onboarding, MECH-307 A/B)", flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict] = []
    per_seed: Dict[str, Dict[int, Dict]] = {c: {} for c in CONDITIONS}

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

    # --- per-seed gate + diagnostics ---
    g0_on_per_seed, g0_off_per_seed = [], []
    zgoal_on_floor_per_seed, zgoal_off_floor_per_seed = [], []
    reached_p2_per_seed, window_live_per_seed = [], []
    d_rvr_per_seed, d_zgoal_per_seed = [], []

    for seed in SEEDS:
        on = per_seed["FULL_M307_ON"][seed]
        off = per_seed["FULL_M307_OFF"][seed]
        rand = per_seed["RANDOM"][seed]

        g0_on_per_seed.append(on["resource_visit_rate"] >= rand["resource_visit_rate"] + G0_MARGIN)
        g0_off_per_seed.append(off["resource_visit_rate"] >= rand["resource_visit_rate"] + G0_MARGIN)
        zgoal_on_floor_per_seed.append(on["zgoal_norm_mean"] > C6_ZGOAL_FLOOR)
        zgoal_off_floor_per_seed.append(off["zgoal_norm_mean"] > C6_ZGOAL_FLOOR)
        reached_p2_per_seed.append(on["total_steps"] > 0)
        window_live_per_seed.append(rand["resource_visit_rate"] > P2_WINDOW_CONTACT_FLOOR)
        d_rvr_per_seed.append(on["resource_visit_rate"] - off["resource_visit_rate"])
        d_zgoal_per_seed.append(on["zgoal_norm_mean"] - off["zgoal_norm_mean"])

    g0_on_frac = _frac_seeds(g0_on_per_seed)
    g0_off_frac = _frac_seeds(g0_off_per_seed)
    reached_p2_frac = _frac_seeds(reached_p2_per_seed)
    window_live_frac = _frac_seeds(window_live_per_seed)

    g0_on_pass = g0_on_frac >= SEED_THRESHOLD
    g0_off_pass = g0_off_frac >= SEED_THRESHOLD
    reached_p2_ok = reached_p2_frac >= SEED_THRESHOLD
    window_live_ok = window_live_frac >= SEED_THRESHOLD
    preconditions_met = reached_p2_ok and window_live_ok

    # G0 non-degeneracy: both arms measured with real steps AND the compared arms
    # are not bit-identical across every seed (a live, discriminating measurement).
    on_stepped = all(per_seed["FULL_M307_ON"][s]["total_steps"] > 0 for s in SEEDS)
    rand_stepped = all(per_seed["RANDOM"][s]["total_steps"] > 0 for s in SEEDS)
    not_identical = any(
        per_seed["FULL_M307_ON"][s]["resource_visit_rate"]
        != per_seed["RANDOM"][s]["resource_visit_rate"] for s in SEEDS
    )
    g0_non_degenerate = bool(on_stepped and rand_stepped and not_identical)

    # MECH-307 perturbation flag: OFF would clear G0 but ON does not.
    mech307_perturbs_baseline = bool(g0_off_pass and not g0_on_pass)

    # --- routing ---
    if not preconditions_met:
        status = "FAIL"
        label = "substrate_not_ready_requeue"
        interpretation_text = (
            "G0 measurement NOT TAKEABLE: the measurability preconditions were not "
            "met -- either the curriculum did not reach the P2 window on >= 2/3 "
            "seeds (reached_p2) or a RANDOM policy could not contact resources in "
            "the P2 window on >= 2/3 seeds (p2_window_admits_contact), so a "
            "FULL-vs-RANDOM comparison would be a dead-window artifact rather than a "
            "readiness verdict. Self-routes substrate_not_ready_requeue -- this is "
            "NOT a verdict on ARC-030/MECH-307."
        )
    elif g0_on_pass:
        status = "PASS"
        label = "readiness_pass_arc030_retest_unblocked"
        interpretation_text = (
            "READINESS PASS: with MECH-307 genuinely ON for the first time, the "
            "FULL arm clears the RANDOM baseline resource-visit rate by >= 0.05 on "
            ">= 2/3 seeds (G0_ON). The shared non-degeneracy precondition holds, so "
            "ARC-030's COMBINED-vs-NOGO_ONLY discriminative retest (design in "
            "ARC-030's what_would_answer) is unblocked. Read the non-gating z_goal "
            "survival and MECH-307 A/B deltas below to see whether MECH-307-on is "
            "what lifted G0 (ON clears while OFF does not) or whether both arms "
            "clear it."
        )
    else:
        status = "FAIL"
        label = "readiness_fail_curriculum_gate_blocks_retest"
        interpretation_text = (
            "READINESS FAIL (informative, NOT a claim verdict): with MECH-307 "
            "genuinely ON, the FULL arm still does not clear the RANDOM baseline by "
            ">= 0.05 on >= 2/3 seeds (G0_ON), on a live/discriminating P2 window. "
            "MECH-307 reachability alone does not restore the G0 gate -- consistent "
            "with the still-open curriculum gap (z_goal starved before P2). Route to "
            "the substrate_queue stub `scaffolded-curriculum-hazard-rebalance` FIRST, "
            "then re-run this readiness gate; do NOT force through to the "
            "discriminative retest. "
            + (
                "ADDITIONALLY, mech307_perturbs_baseline=TRUE: the MECH-307-OFF arm "
                "WOULD have cleared G0 while the ON arm does not -- MECH-307 being "
                "genuinely on, though mechanically correct, DEGRADES this baseline "
                "config's foraging competence. That interaction must be understood "
                "before the discriminative retest, independent of the curriculum fix."
                if mech307_perturbs_baseline else
                "mech307_perturbs_baseline=FALSE: the OFF arm also fails G0, so this "
                "is the pre-existing curriculum gap (866a reproduced), not a MECH-307 "
                "regression."
            )
        )

    # --- interpretation block (diagnostic adjudication gate) ---
    preconditions = [
        {
            "name": "curriculum_reached_p2",
            "description": "FULL_M307_ON completed the curriculum through the P2 "
                           "measurement window (total_steps > 0) on >= 2/3 seeds",
            "kind": "readiness",
            "measured": reached_p2_frac,
            "threshold": SEED_THRESHOLD,
            "direction": "lower",
            "control": "FULL_M307_ON trained cells; total_steps>0 means P2 measurement ran",
            "met": reached_p2_ok,
        },
        {
            "name": "p2_window_admits_contact",
            "description": "RANDOM resource_visit_rate > 0.005 on >= 2/3 seeds -- the "
                           "SAME statistic G0 routes on, on the positive control (a "
                           "random policy on the P2 target env). Guards the 866a "
                           "P2_HFA_GUARD=-1.0 dead-window degeneracy.",
            "kind": "readiness",
            "measured": window_live_frac,
            "threshold": SEED_THRESHOLD,
            "direction": "lower",
            "control": "RANDOM policy on the P2 env; a live window must let random forage above the floor",
            "met": window_live_ok,
        },
    ]
    criteria = [
        {"name": "G0_ON_resource_visit_rate_clears_random", "load_bearing": True, "passed": bool(g0_on_pass)},
    ]
    criteria_non_degenerate = {
        "G0_ON_resource_visit_rate_clears_random": g0_non_degenerate,
    }

    # --- metrics ---
    metrics: Dict[str, float] = {
        "g0_on_frac_seeds": g0_on_frac,
        "g0_off_frac_seeds": g0_off_frac,
        "g0_on_pass": 1.0 if g0_on_pass else 0.0,
        "g0_off_pass": 1.0 if g0_off_pass else 0.0,
        "reached_p2_frac_seeds": reached_p2_frac,
        "p2_window_live_frac_seeds": window_live_frac,
        "preconditions_met": 1.0 if preconditions_met else 0.0,
        "g0_non_degenerate": 1.0 if g0_non_degenerate else 0.0,
        "mech307_perturbs_baseline": 1.0 if mech307_perturbs_baseline else 0.0,
        "zgoal_on_clears_c6_floor_frac_seeds": _frac_seeds(zgoal_on_floor_per_seed),
        "zgoal_off_clears_c6_floor_frac_seeds": _frac_seeds(zgoal_off_floor_per_seed),
        "d_resource_visit_rate_mean": sum(d_rvr_per_seed) / len(d_rvr_per_seed),
        "d_zgoal_norm_mean": sum(d_zgoal_per_seed) / len(d_zgoal_per_seed),
    }
    for cond in CONDITIONS:
        rvr = [per_seed[cond][s]["resource_visit_rate"] for s in SEEDS]
        hr = [per_seed[cond][s]["harm_rate"] for s in SEEDS]
        ent = [per_seed[cond][s]["policy_entropy"] for s in SEEDS]
        surv = [per_seed[cond][s]["survival_rate"] for s in SEEDS]
        zg_vals = [per_seed[cond][s]["zgoal_norm_mean"] for s in SEEDS]
        metrics[f"resource_visit_rate_mean_{cond}"] = sum(rvr) / len(rvr)
        metrics[f"harm_rate_mean_{cond}"] = sum(hr) / len(hr)
        metrics[f"policy_entropy_mean_{cond}"] = sum(ent) / len(ent)
        metrics[f"survival_rate_mean_{cond}"] = sum(surv) / len(surv)
        metrics[f"zgoal_norm_mean_{cond}"] = sum(zg_vals) / len(zg_vals)

    summary_markdown = (
        f"# {QUEUE_ID} -- ARC-030 / MECH-307 G0-readiness diagnostic\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; gates ARC-030/Q-021/INV-034)\n"
        f"**Readiness check of:** {READINESS_CHECK_OF}\n\n"
        f"## Readiness gate\n\n"
        f"| Item | Frac seeds | Pass |\n|---|---|---|\n"
        f"| P1 curriculum_reached_p2 | {reached_p2_frac:.2f} | {reached_p2_ok} |\n"
        f"| P2 p2_window_admits_contact | {window_live_frac:.2f} | {window_live_ok} |\n"
        f"| **G0_ON (load-bearing)** | {g0_on_frac:.2f} | {g0_on_pass} |\n"
        f"| G0_OFF (reproduction, non-gating) | {g0_off_frac:.2f} | {g0_off_pass} |\n\n"
        f"## MECH-307 A/B (non-gating diagnostics)\n\n"
        f"| Metric | FULL_M307_ON | FULL_M307_OFF | RANDOM |\n|---|---|---|---|\n"
        f"| resource_visit_rate (mean) | {metrics['resource_visit_rate_mean_FULL_M307_ON']:.4f} "
        f"| {metrics['resource_visit_rate_mean_FULL_M307_OFF']:.4f} "
        f"| {metrics['resource_visit_rate_mean_RANDOM']:.4f} |\n"
        f"| zgoal_norm (mean; C6 floor 0.4) | {metrics['zgoal_norm_mean_FULL_M307_ON']:.4f} "
        f"| {metrics['zgoal_norm_mean_FULL_M307_OFF']:.4f} | -- |\n\n"
        f"d_resource_visit_rate (ON-OFF) = {metrics['d_resource_visit_rate_mean']:.4f}; "
        f"d_zgoal (ON-OFF) = {metrics['d_zgoal_norm_mean']:.4f}; "
        f"mech307_perturbs_baseline = {bool(mech307_perturbs_baseline)}\n\n"
        f"## Interpretation\n\n{interpretation_text}\n"
    )

    result: Dict[str, Any] = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "gates_claims": GATES_CLAIMS,
        "readiness_check_of": READINESS_CHECK_OF,
        "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "text": interpretation_text,
        },
        "per_seed_results": per_seed,
        "per_seed_diagnostics": {
            "g0_on_per_seed": g0_on_per_seed,
            "g0_off_per_seed": g0_off_per_seed,
            "d_resource_visit_rate_per_seed": d_rvr_per_seed,
            "d_zgoal_norm_per_seed": d_zgoal_per_seed,
            "zgoal_on_clears_c6_floor_per_seed": zgoal_on_floor_per_seed,
            "zgoal_off_clears_c6_floor_per_seed": zgoal_off_floor_per_seed,
        },
        "fatal_error_count": 0,
    }
    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
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
        "g0_margin": G0_MARGIN,
        "c6_zgoal_floor": C6_ZGOAL_FLOOR,
        "p2_window_contact_floor": P2_WINDOW_CONTACT_FLOOR,
        "ab_variable": "use_mech307_conjunction",
    }

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
