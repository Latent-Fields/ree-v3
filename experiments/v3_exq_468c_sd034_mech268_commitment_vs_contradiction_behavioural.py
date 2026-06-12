"""
V3-EXQ-468c (EXP-0164 behavioural): commitment vs contradiction, measured on a
foraging-competent agent built through the FULL scaffolded_sd054_onboarding
curriculum at the 603n config. Successor to V3-EXQ-468b (NOT a supersede).

ROUTING: the commitment_closure:GAP-4 *b cohort (460b/461b/464b/466b/467b/468b)
all RAN 2026-06-04 and were reclassified non_contributory + substrate_ceiling:
in the committed_mode_curriculum loop the agent committed (beta latch engaged)
but never tolerance-completed a waypoint / never hit a contradiction, so
n_closures = beta_release = n_windows = 0 in EVERY arm (the
substrate-not-engaged signature -- the agent could not forage competently enough
to reach a goal, so no completion / contradiction ever occurred). 468b was built
on the committed_mode_curriculum (P0 warmup -> P1 consolidation), which trains
commitment but NOT foraging competence, so it could not deliver runtime goal
completions. This successor REWIRES the training harness to the
scaffolded_sd054_onboarding curriculum, whose readiness V3-EXQ-603n PASSED
(substrate_queue.scaffolded_sd054_onboarding.ready flipped true 2026-06-11) --
the substrate that DELIVERS the foraging competence / goal completions the *b
cohort lacked. The commitment-vs-contradiction mechanism under test is unchanged
from 468b; only the agent-building curriculum changed (hence a NEW letter, not a
silent re-run).

WHAT THIS MEASURES (unchanged from 468b): when counter-evidence contradicts an
active commitment, a healthy agent RELEASES the commitment (MECH-090 beta drops,
coordinated by SD-034 closure + MECH-268 dACC PE saturation preventing runaway
conflict accumulation). Without the substrate the agent PERSEVERATES -- stays
committed despite contradiction -- the OCD-like signature.

SUBSTRATE: built through the FULL scaffolded_sd054_onboarding curriculum at the
603n config (Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1 -> P2), identical to
V3-EXQ-514n, WITH the commitment-control-plane substrate ALSO enabled on the same
agent (bistable BetaGate MECH-090 + SD-034 ClosureOperator + MECH-268 dACC PE
saturation + SD-033a LateralPFCAnalog + SD-032 dACC/salience). The foraging
substrate (harm streams, SP-CEM, z_goal, MECH-295 bridge, SD-057 cue-recall,
SD-058 instrumental avoidance, harm-pathway training) is identical to the
603n-validated config so the curriculum builds a foraging-competent agent; the
control-plane substrate is layered on top so commitment + contradiction-release
are exercisable. Then a frozen-policy commitment-vs-contradiction eval runs in a
P2-config env that ADDS the GAP-3 completion_tolerance(waypoint) +
counter_evidence primitives (dynamics-only; world_obs_dim-preserving, verified).

ARMS (one curriculum build per seed, two frozen-policy evals):
  ARM_SUBSTRATE_ON  -- full closure + dACC-saturation + bistable agent.
                       Counter-evidence fires DURING committed sequences; dACC PE
                       saturation attenuates the conflicting PE and closure
                       releases the MECH-090 beta latch.
  ARM_SUBSTRATE_OFF -- clone of the SAME trained weights with closure OFF and dACC
                       saturation disabled. Expect perseveration: committed steps
                       sustained despite counter-evidence.
  (The committed_mode_curriculum O-2 forced-rv contrast arm is dropped: it was a
   GAP-11 committed_mode_curriculum requirement and that curriculum is no longer
   used. The load-bearing dissociation is the ON-vs-OFF substrate contrast.)

CONTACT GUARD (603n G2 + G3 foraging non-vacuity, mirrors 514n): per-seed guard =
  (P2 contact_rate > 0) AND (P2 z_goal_norm_at_contact_peak > 0.4). A seed failing
  the guard is excluded from DV aggregation; < 2/3 seeds passing -> the run
  self-routes substrate_not_ready_requeue (FAIL; non_contributory). A
  foraging-incompetent seed is "could not forage", NOT a real commitment null.

COMMITMENT NON-VACUITY READINESS GATE (the EXACT gap that made the *b cohort
  non_contributory -- they checked nothing and scored n_closures=0 as evidence):
  before scoring C1/C2/C3, assert on guard-passing seeds that the ON-arm eval
  actually exercised the mechanism: total_beta_elevated > 0 (the agent COMMITTED)
  AND episodes_with_contradiction > 0 (counter-evidence FIRED during a committed
  window) on >= 2/3 guard-passing seeds. Below floor -> substrate_not_ready_requeue
  (FAIL; non_contributory), NEVER a false weakens. This is the V3-EXQ-643 /
  V3-EXQ-514n same-precondition lesson applied to the commitment DV: a contact-zero
  OR commitment-zero OR contradiction-zero read is "substrate not engaged", not
  "substrate falsified".

PRE-REGISTERED ACCEPTANCE (thresholds are constants, NOT post-hoc):
  C1  ARM_SUBSTRATE_ON  beta_release_near_contradiction >= 1
                        (>= 1 beta release within RELEASE_WINDOW steps of a
                         counter-evidence injection event)
  C2  ARM_SUBSTRATE_ON  committed_frac_post < C2_DROP_FACTOR
                        (committed-step fraction drops after first contradiction)
  C3  ARM_SUBSTRATE_OFF committed_frac_post >= C3_PERSIST_FACTOR
                        (OFF arm perseverates -- fraction is sustained)
  Per-seed PASS = C1 AND C2 AND C3. EXPERIMENT PASS = contact non-vacuity met AND
  commitment non-vacuity met AND >= 2/3 guard-passing seeds pass.

PER-CLAIM DIRECTION:
  contact-guard non-vacuity NOT met     -> non_contributory (substrate_not_ready_requeue).
  commitment non-vacuity NOT met        -> non_contributory (substrate_not_ready_requeue).
  both met:  SD-034 = supports if (C1 AND C2) all guard-passing seeds else weakens;
             MECH-268 = supports if C1 all guard-passing seeds else weakens;
             MECH-090 = supports if overall PASS else weakens (a GENUINE weakens:
             the agent forages, commits, and hits a contradiction, yet does not
             release the commitment).

claim_ids: SD-034, MECH-268, MECH-090.
experiment_purpose: evidence
predecessor: V3-EXQ-468b (NOT a supersede -- 468b is non_contributory/substrate_ceiling,
  not invalid-by-bug; the V3-EXQ-468 substrate-readiness diagnostic stands).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
  goal-pipeline onboarding scheduler).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_468c_sd034_mech268_commitment_vs_contradiction_behavioural"
QUEUE_ID = "V3-EXQ-468c"
CLAIM_IDS: List[str] = ["SD-034", "MECH-268", "MECH-090"]
EXPERIMENT_PURPOSE = "evidence"
PREDECESSOR = "V3-EXQ-468b (successor, NOT supersede)"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_COMMITMENT_VS_CONTRADICTION"

# --- Goal-pipeline / encoder dims (mirror 603n / 514n exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Curriculum budgets (mirror 603n / 514n exactly) ---
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

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 603n / 514n) ---
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (mirror 603n / 514n) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- MECH-268 dACC PE saturation knobs (mirror 468b) ---
DACC_SAT_WINDOW = 8
DACC_SAT_STRENGTH = 0.3
DACC_SAT_GRACE = 2

# --- Contradiction eval thresholds (mirror 468b; constants, NOT derived) ---
RELEASE_WINDOW = 20      # steps after a counter-evidence injection within which a
                         # beta release counts as contradiction-triggered (C1)
C2_DROP_FACTOR = 0.85    # C2: ON committed-frac post must be below this * pre
C3_PERSIST_FACTOR = 0.70 # C3: OFF committed-frac post must stay at/above this * pre

# --- Pre-registered acceptance thresholds ---
P2_ZGOAL_GATE = 0.4          # per-seed contact-guard: z_goal_norm_at_contact_peak floor (603n G3)
CONTACT_GATE = 0.0           # per-seed contact-guard: P2 contact_rate floor (603n G2)
MIN_FRACTION = 2.0 / 3.0     # >= 2/3 seeds for non-vacuity + any aggregate gate


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
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate (mirror 514n) + the commitment
    control-plane substrate (bistable BetaGate + SD-034 closure + SD-033a
    LateralPFC + SD-032 dACC/salience). dACC PE saturation (MECH-268) is set on
    the built agent (not surfaced through from_dims)."""
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
    """P2-config foraging env (same structural kwargs as _build_env(cfg, 'p2') so
    world_obs_dim matches the curriculum-built agent) WITH the GAP-3
    completion_tolerance(waypoint) + counter_evidence contradiction primitives
    layered on (dynamics-only; verified world_obs_dim-preserving)."""
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
        # GAP-3 contradiction primitives (the 468 mechanism; dynamics-only).
        # subgoal_mode=True is LOAD-BEARING: both waypoint-completion paths AND
        # the counter_evidence injection gate hard-require it (causal_grid_world
        # lines 1664/1740/2067). 468b never set it -> the entire GAP-3 waypoint/
        # counter-evidence machinery was inert there, so its contradictions=0 was
        # partly a wiring gap (not only the *b-cohort foraging-competence ceiling).
        # num_waypoints=4 lengthens the committed sequence so seq_in_progress
        # windows are long enough to overlap the injection cadence; the interval/
        # prob are tuned so counter-evidence fires DURING committed windows (probe
        # 2026-06-12: 23 injections / 16 beta-releases-near-contradiction).
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
    """Clone the SAME trained weights into a substrate-OFF agent (closure OFF +
    dACC PE saturation OFF) -- the perseveration-without-substrate contrast arm.
    The closure operator carries no trainable parameters, so the trained
    state_dict loads cleanly (strict, with a non-strict fallback)."""
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


def _eval_contradiction_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for commitment-vs-contradiction (ported
    from 468b, using _sense_with_optional_harm so z_harm_a populates). Tracks per
    episode: committed steps before/after the first counter-evidence injection;
    beta releases within RELEASE_WINDOW of any injection; total beta-elevated +
    committed steps."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    total_committed_steps = 0
    total_beta_elevated = 0
    total_beta_release_events = 0
    total_beta_release_near_contradiction = 0
    total_committed_pre = 0
    total_committed_post = 0
    total_episodes_with_contradiction = 0

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
            first_contradiction_step = -1
            recent_injection_timers: List[int] = []

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

                injected = bool(info.get("counter_evidence_injected_this_tick", False))
                if injected:
                    recent_injection_timers.append(0)
                    if first_contradiction_step < 0:
                        first_contradiction_step = step

                if first_contradiction_step >= 0:
                    if cur_committed:
                        ep_committed_post += 1
                else:
                    if cur_committed:
                        ep_committed_pre += 1

                if done:
                    break

            had_contradiction = first_contradiction_step >= 0
            if had_contradiction:
                total_episodes_with_contradiction += 1
                total_committed_pre += ep_committed_pre
                total_committed_post += ep_committed_post

            total_committed_steps += ep_committed
            total_beta_elevated += ep_elevated
            total_beta_release_events += ep_release_events
            total_beta_release_near_contradiction += ep_release_near_contradiction

            per_episode.append({
                "committed_steps": ep_committed,
                "beta_elevated_steps": ep_elevated,
                "release_events": ep_release_events,
                "release_near_contradiction": ep_release_near_contradiction,
                "committed_pre": ep_committed_pre,
                "committed_post": ep_committed_post,
                "had_contradiction": had_contradiction,
                "first_contradiction_step": first_contradiction_step,
            })

    eps_with_c = max(1, total_episodes_with_contradiction)
    mean_committed_pre = total_committed_pre / eps_with_c
    mean_committed_post = total_committed_post / eps_with_c
    committed_frac_post = (
        mean_committed_post / mean_committed_pre if mean_committed_pre > 0 else 1.0
    )

    return {
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "total_beta_release_events": total_beta_release_events,
        "beta_release_near_contradiction": total_beta_release_near_contradiction,
        "episodes_with_contradiction": total_episodes_with_contradiction,
        "mean_committed_pre": mean_committed_pre,
        "mean_committed_post": mean_committed_post,
        "committed_frac_post_vs_pre": committed_frac_post,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    empty_arm = {
        "total_committed_steps": 0, "total_beta_elevated": 0,
        "total_beta_release_events": 0, "beta_release_near_contradiction": 0,
        "episodes_with_contradiction": 0, "mean_committed_pre": 0.0,
        "mean_committed_post": 0.0, "committed_frac_post_vs_pre": 1.0,
        "n_eval_episodes": 0, "closure_present": False,
    }
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "ARM_SUBSTRATE_ON": dict(empty_arm),
        "ARM_SUBSTRATE_OFF": dict(empty_arm),
        "criteria": {"C1": False, "C2": False, "C3": False},
        "commitment_non_vacuity": False,
        "pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else CONTRADICTION_EVAL_EPISODES

    # Build the agent on a P2-config env (world_obs_dim parity with the eval env).
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
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

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

    # C1: ON arm had >= 1 beta release triggered near a contradiction.
    c1 = arm_on["beta_release_near_contradiction"] >= 1
    # C2: ON arm committed fraction dropped after the first contradiction.
    c2 = arm_on["committed_frac_post_vs_pre"] < C2_DROP_FACTOR
    # C3: OFF arm committed fraction sustained (perseveration).
    c3 = arm_off["committed_frac_post_vs_pre"] >= C3_PERSIST_FACTOR

    # Commitment non-vacuity: the ON arm actually committed AND a contradiction
    # fired during the eval (the exact gap the *b cohort scored as n_closures=0).
    commitment_non_vacuity = bool(
        arm_on["total_beta_elevated"] > 0
        and arm_on["episodes_with_contradiction"] > 0
    )

    seed_pass = bool(c1 and c2 and c3)

    print(f"  [train] wl_eval seed={seed} ep {done}/{total_eps}"
          f" c1={c1} c2={c2} c3={c3}"
          f" beta_elev={arm_on['total_beta_elevated']}"
          f" eps_contra={arm_on['episodes_with_contradiction']}"
          f" rel_near={arm_on['beta_release_near_contradiction']}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} commit_nonvacuity={commitment_non_vacuity}"
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
        "ARM_SUBSTRATE_ON": arm_on,
        "ARM_SUBSTRATE_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "commitment_non_vacuity": commitment_non_vacuity,
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

    # --- Commitment non-vacuity readiness gate over guard-passing seeds ---
    commit_flags = [bool(r.get("commitment_non_vacuity", False)) for r in guard_passing]
    commit_nonvacuity_frac = _frac(commit_flags)
    commitment_non_vacuity_met = bool(commit_nonvacuity_frac >= MIN_FRACTION)

    # --- Per-seed C1/C2/C3 over guard-passing seeds ONLY ---
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

    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not commitment_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "commitment_or_contradiction_not_engaged"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("commitment_vs_contradiction_release"
                           if overall_criteria_pass else "residual_perseveration_open")
        route_reason = "c1_c2_c3_majority_met" if overall_criteria_pass else "criteria_unmet_genuine_weakens"
        direction_map = {
            "SD-034": "supports" if (c1_all and c2_all) else "weakens",
            "MECH-268": "supports" if c1_all else "weakens",
            "MECH-090": "supports" if overall_criteria_pass else "weakens",
        }
        overall_direction = "supports" if overall_criteria_pass else "weakens"

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) commit_non_vacuity={commitment_non_vacuity_met}"
          f" (frac={commit_nonvacuity_frac:.3f}) criteria_pass={overall_criteria_pass}"
          f" ({n_pass}/{len(guard_passing)}) -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "commitment_non_vacuity_met": commitment_non_vacuity_met,
        "commitment_non_vacuity_fraction": commit_nonvacuity_frac,
        "C1_all_guard_passing": c1_all,
        "C2_all_guard_passing": c2_all,
        "criteria_pass_fraction": pass_frac,
        "n_seeds_pass": n_pass,
        "overall_pass": bool(contact_non_vacuity_met and commitment_non_vacuity_met
                             and overall_criteria_pass),
        "per_seed_guard_pass": guard_flags,
        "per_seed_criteria_pass": [bool(r.get("pass", False)) for r in per_seed],
        "route_reason": route_reason,
    }

    # criteria_non_degenerate: did C1/C2/C3 get a fair test (both non-vacuity gates met)?
    crit_non_degenerate = bool(contact_non_vacuity_met and commitment_non_vacuity_met)

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
                                   "z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds "
                                   "(the foraging competence the *b cohort lacked).",
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                },
                {
                    "name": "commitment_and_contradiction_engaged",
                    "description": "the ON-arm eval actually committed (total_beta_elevated "
                                   "> 0) AND a counter-evidence injection fired during the "
                                   "eval (episodes_with_contradiction > 0) on >= 2/3 "
                                   "guard-passing seeds -- the EXACT n_closures=0 gap the *b "
                                   "cohort scored as evidence (now self-routed, never a "
                                   "false weakens).",
                    "control": "fraction of guard-passing seeds where the ON-arm beta latch "
                               "engaged AND a contradiction fired.",
                    "measured": commit_nonvacuity_frac,
                    "threshold": MIN_FRACTION,
                    "met": commitment_non_vacuity_met,
                },
            ],
            "criteria": [
                {"name": "C1_beta_release_near_contradiction", "load_bearing": True,
                 "passed": c1_all},
                {"name": "C2_committed_frac_drop_on", "load_bearing": True,
                 "passed": c2_all},
                {"name": "C3_off_arm_perseverates", "load_bearing": True,
                 "passed": _all_guard("C3")},
            ],
            "criteria_non_degenerate": {
                "C1": crit_non_degenerate,
                "C2": crit_non_degenerate,
                "C3": crit_non_degenerate,
            },
            "contact_guard": {
                "definition": "per-seed: P2 contact_rate > 0 AND z_goal_norm_at_contact_peak "
                              "> 0.4 (603n G2 + G3). < 2/3 seeds passing -> "
                              "substrate_not_ready_requeue, never a false weakens.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "commitment_non_vacuity_gate": {
                "definition": "ON-arm total_beta_elevated > 0 AND episodes_with_contradiction "
                              "> 0 on >= 2/3 guard-passing seeds. Below floor -> "
                              "substrate_not_ready_requeue (non_contributory), NEVER a false "
                              "weakens (the V3-EXQ-643 / V3-EXQ-514n same-precondition lesson "
                              "applied to the commitment DV).",
                "min_fraction": MIN_FRACTION,
                "release_window": RELEASE_WINDOW,
                "c2_drop_factor": C2_DROP_FACTOR,
                "c3_persist_factor": C3_PERSIST_FACTOR,
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
                     "P0 -> Stage-H -> P1 -> P2; harm-pathway training ON, 603n config; "
                     "ready=true 2026-06-11) + commitment control-plane (bistable BetaGate "
                     "MECH-090 + SD-034 ClosureOperator + MECH-268 dACC PE saturation + "
                     "SD-033a LateralPFC + SD-032 dACC/salience) + GAP-3 contradiction env "
                     "primitives (completion_tolerance(waypoint) + counter_evidence).",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "method_note": "468b's commitment-vs-contradiction mechanism (SD-034 closure + "
                       "MECH-268 dACC PE saturation coordinate a MECH-090 beta release under "
                       "sustained counter-evidence; ON releases, OFF perseverates) re-run on a "
                       "foraging-competent agent BUILT THROUGH the scaffolded_sd054_onboarding "
                       "curriculum (whose readiness V3-EXQ-603n PASSED). 468b used the "
                       "committed_mode_curriculum, which trains commitment but NOT foraging "
                       "competence, so the agent committed but never tolerance-completed a "
                       "waypoint / never hit a contradiction (n_closures=0; the *b-cohort "
                       "substrate-not-engaged signature). The scaffolded curriculum delivers "
                       "the foraging competence / goal completions that gap lacked.",
        "readiness_note": "TWO non-vacuity gates self-route a substrate-not-engaged read to "
                          "non_contributory (never a false weakens): (1) the 603n contact guard "
                          "(foraging competence); (2) the commitment non-vacuity gate (the ON "
                          "arm actually committed AND a contradiction fired). Only when BOTH "
                          "are met does C1/C2/C3 drive a supports/weakens verdict. This applies "
                          "the V3-EXQ-643 / V3-EXQ-514n same-precondition lesson to the "
                          "commitment DV.",
        "arm_note": "ARM_SUBSTRATE_ON (full closure + dACC saturation + bistable) vs "
                    "ARM_SUBSTRATE_OFF (same trained weights, closure off + saturation off). "
                    "The committed_mode_curriculum O-2 forced-rv contrast arm is dropped (that "
                    "curriculum is no longer used).",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "release_window": RELEASE_WINDOW,
            "c2_drop_factor": C2_DROP_FACTOR,
            "c3_persist_factor": C3_PERSIST_FACTOR,
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
            "config_basis": "V3-EXQ-603n (substrate-readiness run that flipped "
                            "scaffolded_sd054_onboarding ready=true)",
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
