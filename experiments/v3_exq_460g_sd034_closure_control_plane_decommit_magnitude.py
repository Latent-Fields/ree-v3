"""
V3-EXQ-460g (supersedes V3-EXQ-460f): SD-034 closure-control-plane de-commit
authority retest on the DE-COMMIT-AUTHORITY-MAGNITUDE-amended substrate
(committed-run-scaled Leg-B refractory, ree-v3 main 2cd0aa2), with the C2 DV
redesigned to a WITHIN-ARM around-closure occupancy delta.

WHY THIS SUPERSEDES 460f (failure_autopsy_V3-EXQ-460f_2026-06-18, confirmed +
user-adjudicated 2026-06-18T08:04Z governance cycle): the BETA-ENGAGEMENT amend
WORKED -- all 4 readiness gates cleared and the C2 de-commit occupancy-drop DV ran for
the first time (PASS seed 42: ON 23.73 < OFF 35.67, -33.5%; FAIL 2/3). But on
strong-natural-commit seeds the closure->beta coupling was INERT
(sd034_n_closure_coupled_elevations 36/52 on seed 42 vs 0/0 on seeds 43/44), so the DV
reduced to the bare Leg-B 5-tick refractory whose magnitude (~20-35 tick-blocks) is
SWAMPED by the ~530-560 natural-commit elevated steps; the BETWEEN-arm unpaired
occupancy DV is also underpowered. NOT a falsification (seed 42 + 460e seed 44 are
existence proofs of the correct de-commit SIGN); the gap is de-commit-authority
MAGNITUDE + DV POWER.

THE TWO FIXES THIS RE-ISSUE ARMS:
  (a) MAGNITUDE LEVER (substrate, landed ree-v3 main 2cd0aa2): the Leg-B refractory
      installed at a closure fire is now SCALED by the committed-run length captured
      from the BetaGate BEFORE the closure's own release():
        n = closure_decommit_hold_ticks
            + round(closure_decommit_hold_scale_with_run * committed_run_length),
      clamped to closure_decommit_hold_max_ticks. A long committed run -- the exact
      source of the ~530-560 swamping elevated steps -- triggers a proportionally long
      post-closure hold, so the de-commit authority scales with the latch occupancy it
      must overcome. Armed here via closure_decommit_hold_scale_with_run +
      closure_decommit_hold_max_ticks on the ON-arm config. (The OFF clone has closure
      off -> _fire never runs -> the lever is inert there.)
  (b) WITHIN-ARM C2 DV (experiment side, this script): the load-bearing C2 is now a
      PAIRED within-ON-arm AROUND-CLOSURE occupancy delta -- for each closure fire, the
      beta-latch occupancy in a CLOSURE_WINDOW pre-closure window vs the same-length
      post-closure window. The de-commit lowers post-closure occupancy below pre-closure
      occupancy. This is paired (each closure is its own pre/post pair) and so is far
      more powerful than 460f's between-arm unpaired mean comparison (which is retained
      here as a SECONDARY diagnostic, NOT load-bearing).

NON-VACUITY GATE TIGHTENED (the brief): the coupling must actually have engaged the
latch on the ON arm -- sd034_n_closure_coupled_elevations > 0 on >= 2/3 guard seeds --
else the de-commit DV is measuring the fragile natural commit-entry, not the
closure-plane commitment, and the run self-routes substrate_not_ready_requeue (NEVER a
false weakens). This replaces 460f's both-arms-total_beta_elevated gate with the direct
coupling non-vacuity readout.

SUBSTRATE legs under test (all landed; this run arms them on the 603n foraging substrate):
  Leg A  env-completion hook (use_closure_env_completion_hook) -> emit_closure.
  Leg B  de-commit refractory (closure_decommit_hold_ticks) + the MAGNITUDE lever (a).
  Leg C  scaffold_train_rule_bias_head (598b REINFORCE in P1) -- trained magnitude-bearing
         rule_state so the closure-coupled de-commit has MECH-090 latch authority.
  beta-engagement coupling (use_closure_commit_beta_coupling) -- ties the closure-plane
         commitment to bistable beta elevation so the latch is engaged on every seed where
         a closure commitment forms.

ARMS (one curriculum build per seed, two frozen-policy evals):
  ARM_CLOSURE_ON   -- full closure + env hook + de-commit hold + MAGNITUDE lever + bistable,
                      built on the TRAINED rule_bias_head. Load-bearing C2 measured here.
  ARM_CLOSURE_OFF  -- clone of the SAME trained weights, closure OFF. Secondary between-arm
                      diagnostic only (NOT load-bearing).

READINESS / NON-VACUITY (all must clear before C2 is scored; any unmet self-routes
substrate_not_ready_requeue -- NEVER a false weakens):
  (1) 603n foraging contact guard: per-seed P2 contact_rate > 0 AND
      z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.
  (2) closure-coupling non-vacuity (TIGHTENED): ON sd034_n_closure_coupled_elevations > 0
      AND ON n_sequence_completions > 0 on >= 2/3 guard seeds.
  (3) closure-trigger available: ON n_closures > 0 on >= 2/3 guard seeds.
  (4) rule_bias_head trained (the DIRECT anti-460d-bug gate): P1 rule_bias_pathway_enabled
      AND mean per-candidate |bias| (rule_bias_diag) > RULE_BIAS_MEAN_FLOOR on >= 2/3 seeds.
  (5) within-arm window non-vacuity: ON arm produced >= C2_MIN_WINDOW_EVENTS scored
      around-closure windows with a non-trivial pre-closure occupancy (mean_pre_occ >
      WITHIN_PRE_OCC_FLOOR) on >= 2/3 guard seeds -- else there was nothing committed to
      de-commit (substrate_not_ready_requeue).

PRE-REGISTERED ACCEPTANCE (constants; per-seed PASS = C1 AND C2 AND C3; overall PASS =
majority 2/3 guard seeds; scored only once all five readiness gates clear):
  C1  ARM_CLOSURE_ON  n_closures >= 1                  (closure fires in a live loop)
  C2  WITHIN-ARM AROUND-CLOSURE OCCUPANCY DROP (load-bearing): on the ON arm, mean
      post-closure beta-latch occupancy fraction < mean pre-closure occupancy fraction
      with a >= DECOMMIT_MIN_DROP_FRAC relative drop, over >= C2_MIN_WINDOW_EVENTS scored
      windows with mean_pre_occ > WITHIN_PRE_OCC_FLOOR. Paired within-arm statistic.
  C3  ARM_CLOSURE_ON  nogo_installed_total >= 1        (targeted MECH-260 No-Go)

claim_ids: SD-034, MECH-260, MECH-261.
  - SD-034 direction keys on the within-arm C2 de-commit DV.
  - MECH-260 keys on C3 (No-Go installed).
  - MECH-261 (mode-conditioning) is tagged but its direction stays non_contributory
    unless the AUTOMATIC mode-conditioned detector actually fired (n_automatic_fires > 0):
    the 460f autopsy established all closures were Leg-A hook-driven (n_automatic_fires=0),
    so mode-conditioning was NOT exercised -- protecting the stable MECH-261 claim from a
    self-stamped weakens, exactly as the 460f governance applied.
experiment_purpose: evidence.
supersedes: V3-EXQ-460f.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
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

EXPERIMENT_TYPE = "v3_exq_460g_sd034_closure_control_plane_decommit_magnitude"
QUEUE_ID = "V3-EXQ-460g"
CLAIM_IDS: List[str] = ["SD-034", "MECH-260", "MECH-261"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-460f"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_TRAINED_RULE_BIAS_DECOMMIT_MAGNITUDE_RETEST"

# --- Goal-pipeline / encoder dims (mirror 603n / 460f exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5  # Leg B: post-closure latch refractory base window
# DE-COMMIT-AUTHORITY MAGNITUDE lever (460f amend, ree-v3 main 2cd0aa2): scale the
# Leg-B refractory by the committed-run length at the closure fire so a long committed
# run triggers a proportionally long hold. n = base + round(scale * run_length), capped.
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- Within-arm around-closure window DV (part b) ---
CLOSURE_WINDOW = 10        # ticks on each side of a closure fire
WINDOW_MIN_TICKS = 3       # minimum ticks each side to score an around-closure event
C2_MIN_WINDOW_EVENTS = 2   # minimum scored around-closure windows on the ON arm
WITHIN_PRE_OCC_FLOOR = 0.1  # pre-closure occupancy must be non-trivial (was committed)

# --- Curriculum budgets (mirror 603n / 460f exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
CLOSURE_EVAL_EPISODES = 15  # per arm (ON + OFF)
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
C1_MIN_CLOSURES = 1
C3_MIN_NOGO = 1
# C2 within-arm around-closure de-commit DV: mean post-closure occupancy must be at
# least this RELATIVE fraction below mean pre-closure occupancy (paired across closures).
DECOMMIT_MIN_DROP_FRAC = 0.10
# Leg C readiness: the rule_bias_head must have TRAINED -- mean per-candidate |bias|
# above this floor. The untrained 460d head produced ~0; the Leg-C smoke produced 0.039.
RULE_BIAS_MEAN_FLOOR = 0.005


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
        # commitment_closure:GAP-4 Leg C (2026-06-16): TRAIN the rule_bias_head in P1.
        scaffold_train_rule_bias_head=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate (mirror 460f) + the commitment control-plane +
    the commitment-closure-control-plane amend Legs A/B/C + beta-engagement coupling +
    the DE-COMMIT-AUTHORITY MAGNITUDE lever (460f amend)."""
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
        # SD-034 commitment-closure-control-plane amend (2026-06-12):
        use_closure_env_completion_hook=True,          # Leg A
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base
        # DE-COMMIT-AUTHORITY MAGNITUDE lever (2026-06-19, failure_autopsy_V3-EXQ-460f):
        # scale the Leg-B refractory by committed-run length so the de-commit authority
        # scales with the natural-commit latch occupancy it must overcome.
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D); trained by scaffold leg
        # BETA-ENGAGEMENT amend (2026-06-17): couple the closure-plane commitment to
        # bistable beta elevation so the de-commit DV is readable on every seed where a
        # closure commitment forms.
        use_closure_commit_beta_coupling=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_closure_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint
    tolerance-band completion so the SD-034 closure operator has completions to fire on."""
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


def _clone_closure_off(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone the SAME trained weights into a closure-OFF agent (closure carries no
    trainable parameters, so the state_dict loads cleanly)."""
    cfg_off = copy.deepcopy(trained_agent.config)
    cfg_off.use_closure_operator = False
    cfg_off.heartbeat.beta_gate_bistable = True
    agent_off = REEAgent(cfg_off).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_off.load_state_dict(state)
    except RuntimeError:
        agent_off.load_state_dict(state, strict=False)
    agent_off.e3._running_variance = float(trained_agent.e3._running_variance)
    agent_off.beta_gate = BetaGate(completion_release_threshold=2.0)
    return agent_off


def _around_closure_windows(
    beta_history: List[bool], fire_ticks: List[int]
) -> List[Dict[str, float]]:
    """For each closure fire at tick t, compute the beta-latch occupancy FRACTION in the
    pre-closure window [t-W, t) and the post-closure window (t, t+W], requiring at least
    WINDOW_MIN_TICKS available ticks on each side. Returns one {pre_occ, post_occ} dict
    per scored window (the paired within-arm de-commit datum)."""
    n = len(beta_history)
    events: List[Dict[str, float]] = []
    for t in fire_ticks:
        pre_lo = max(0, t - CLOSURE_WINDOW)
        pre = beta_history[pre_lo:t]               # ticks before the fire
        post_hi = min(n, t + 1 + CLOSURE_WINDOW)
        post = beta_history[t + 1:post_hi]          # ticks after the fire
        if len(pre) < WINDOW_MIN_TICKS or len(post) < WINDOW_MIN_TICKS:
            continue
        pre_occ = sum(1 for b in pre if b) / float(len(pre))
        post_occ = sum(1 for b in post if b) / float(len(post))
        events.append({"pre_occ": pre_occ, "post_occ": post_occ})
    return events


def _eval_closure_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for SD-034 closure behaviour. Adds per-episode
    beta-history + closure-fire-tick tracking so the WITHIN-ARM around-closure occupancy
    delta (part b) can be computed; the legacy between-arm aggregate (mean_beta_elevated_
    steps) is retained as a secondary diagnostic."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_dacc = getattr(agent, "dacc", None) is not None
    hook_enabled = bool(getattr(agent.config, "use_closure_env_completion_hook", False))
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    beta_release_events = 0
    nogo_installed_total = 0
    total_committed_steps = 0
    total_beta_elevated = 0
    n_sequence_completions = 0
    n_hook_fires = 0
    n_closure_coupled_elevations = 0  # BETA-ENGAGEMENT amend non-vacuity readout
    around_events: List[Dict[str, float]] = []  # within-arm pre/post-closure occupancy

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_beta = bool(agent.beta_gate.is_elevated)
            beta_history: List[bool] = []  # per-tick beta-elevated, this episode
            fire_ticks: List[int] = []     # ticks at which a closure fired, this episode

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

                # record beta-latch occupancy for this tick
                cur_beta = bool(agent.beta_gate.is_elevated)
                beta_history.append(cur_beta)
                if agent.e3._committed_trajectory is not None:
                    total_committed_steps += 1
                if cur_beta:
                    total_beta_elevated += 1
                if prev_beta and not cur_beta:
                    beta_release_events += 1
                prev_beta = cur_beta

                _, _harm, done, info, obs_dict = env.step(action_idx)
                if info.get("transition_type") == "sequence_complete":
                    n_sequence_completions += 1
                    if has_closure and hook_enabled:
                        ev = agent.notify_env_completion(action_class=action_idx)
                        if ev is not None and getattr(ev, "fired", False):
                            n_hook_fires += 1
                            nogo_installed_total += int(getattr(ev, "nogo_pushed", 0))

                # a closure fired this tick (automatic detector and/or env hook) ->
                # mark the tick for the around-closure window measurement.
                if has_closure and int(agent.closure_operator._n_closures) > n_closures_before:
                    fire_ticks.append(tick_idx)
                if done:
                    break

            # within-arm around-closure occupancy windows for this episode
            around_events.extend(_around_closure_windows(beta_history, fire_ticks))
            # BETA-ENGAGEMENT amend: accumulate this episode's closure-coupled beta
            # elevations BEFORE the next agent.reset() wipes the per-episode counter.
            n_closure_coupled_elevations += int(
                agent.beta_gate.get_state().get("sd034_n_closure_coupled_elevations", 0)
            )

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre if has_closure else 0
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
    return {
        "n_closures": n_closures,
        "sd034_n_closure_coupled_elevations": n_closure_coupled_elevations,
        "n_hook_fires": n_hook_fires,
        "n_automatic_fires": max(0, n_closures - n_hook_fires),
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "mean_beta_elevated_steps": total_beta_elevated / max(1, n_eps),
        "n_sequence_completions": n_sequence_completions,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        "env_hook_enabled": hook_enabled,
        # within-arm around-closure DV (part b)
        "n_window_events": n_window_events,
        "mean_pre_closure_occ": mean_pre_occ,
        "mean_post_closure_occ": mean_post_occ,
    }


def _within_arm_decommit_drop(arm_on: Dict[str, Any]) -> bool:
    """C2 within-arm around-closure DV (load-bearing): on the ON arm, mean post-closure
    occupancy fraction < mean pre-closure occupancy fraction with a >= DECOMMIT_MIN_DROP_
    FRAC relative drop, over >= C2_MIN_WINDOW_EVENTS scored windows whose pre-occupancy
    cleared WITHIN_PRE_OCC_FLOOR (there was something committed to de-commit)."""
    n_ev = int(arm_on.get("n_window_events", 0))
    pre = float(arm_on.get("mean_pre_closure_occ", 0.0))
    post = float(arm_on.get("mean_post_closure_occ", 0.0))
    if n_ev < C2_MIN_WINDOW_EVENTS or pre <= WITHIN_PRE_OCC_FLOOR:
        return False
    return bool(post < pre and (pre - post) >= DECOMMIT_MIN_DROP_FRAC * pre)


def _within_arm_window_nonvacuous(arm_on: Dict[str, Any]) -> bool:
    """Readiness gate (5): the ON arm produced enough scored around-closure windows with a
    non-trivial pre-closure occupancy for the within-arm DV to be interpretable."""
    return bool(
        int(arm_on.get("n_window_events", 0)) >= C2_MIN_WINDOW_EVENTS
        and float(arm_on.get("mean_pre_closure_occ", 0.0)) > WITHIN_PRE_OCC_FLOOR
    )


def _between_arm_drop(arm_on: Dict[str, Any], arm_off: Dict[str, Any]) -> bool:
    """SECONDARY diagnostic (the 460f between-arm DV; NOT load-bearing): ON mean occupancy
    below OFF mean occupancy by >= DECOMMIT_MIN_DROP_FRAC, OFF having committed."""
    on_occ = float(arm_on.get("mean_beta_elevated_steps", 0.0))
    off_occ = float(arm_off.get("mean_beta_elevated_steps", 0.0))
    if off_occ <= 0.5:
        return False
    return bool(on_occ < off_occ and (off_occ - on_occ) >= DECOMMIT_MIN_DROP_FRAC * off_occ)


def _rule_bias_mean(p1) -> float:
    diag = getattr(p1, "rule_bias_diag", None) or {}
    n = int(diag.get("n_bias_samples", 0))
    s = float(diag.get("sum_bias_abs_mean", 0.0))
    return s / n if n > 0 else 0.0


def _empty_arm() -> Dict[str, Any]:
    return {
        "n_closures": 0, "sd034_n_closure_coupled_elevations": 0, "n_hook_fires": 0,
        "n_automatic_fires": 0, "beta_release_events": 0, "nogo_installed_total": 0,
        "total_committed_steps": 0, "total_beta_elevated": 0,
        "mean_beta_elevated_steps": 0.0, "n_sequence_completions": 0,
        "n_eval_episodes": 0, "closure_present": False, "env_hook_enabled": False,
        "n_window_events": 0, "mean_pre_closure_occ": 0.0, "mean_post_closure_occ": 0.0,
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
        "ARM_CLOSURE_ON": _empty_arm(), "ARM_CLOSURE_OFF": _empty_arm(),
        "criteria": {"C1": False, "C2": False, "C3": False},
        "coupling_nonvacuous": False,
        "closure_trigger_available": False,
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

    closure_env = _build_closure_env(scaffold_cfg)
    closure_env.reset()

    print(f"Seed {seed} Condition ARM_CLOSURE_ON", flush=True)
    arm_on = _eval_closure_behaviour(
        agent, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    print(f"Seed {seed} Condition ARM_CLOSURE_OFF", flush=True)
    agent_off = _clone_closure_off(agent, device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    arm_off = _eval_closure_behaviour(
        agent_off, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    c1 = arm_on["n_closures"] >= C1_MIN_CLOSURES
    c2 = _within_arm_decommit_drop(arm_on)              # LOAD-BEARING within-arm DV
    c3 = arm_on["nogo_installed_total"] >= C3_MIN_NOGO
    # Coupling non-vacuity (TIGHTENED gate): the closure-plane coupling actually engaged
    # the latch on the ON arm AND a sequence completed (closure had an opportunity).
    coupling_nonvacuous = bool(
        arm_on["sd034_n_closure_coupled_elevations"] > 0
        and arm_on["n_sequence_completions"] > 0
    )
    closure_trigger_available = bool(arm_on["n_closures"] > 0)
    within_window_nonvacuous = _within_arm_window_nonvacuous(arm_on)
    between_arm_drop = _between_arm_drop(arm_on, arm_off)  # secondary diagnostic only
    seed_pass = bool(c1 and c2 and c3)

    print(f"  [train] closure_eval seed={seed} ep {done}/{total_eps}"
          f" c1={c1} c2_within={c2} c3={c3}"
          f" pre_occ={arm_on['mean_pre_closure_occ']:.3f} post_occ={arm_on['mean_post_closure_occ']:.3f}"
          f" win_events={arm_on['n_window_events']} coupled={arm_on['sd034_n_closure_coupled_elevations']}"
          f" between_arm_drop={between_arm_drop}"
          f" n_closures={arm_on['n_closures']} auto={arm_on['n_automatic_fires']} hook={arm_on['n_hook_fires']}"
          f" nogo={arm_on['nogo_installed_total']}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} coupling_nonvacuous={coupling_nonvacuous}"
          f" closure_trigger={closure_trigger_available} within_window={within_window_nonvacuous}"
          f" rule_bias_trained={rule_bias_trained}"
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
        "ARM_CLOSURE_ON": arm_on,
        "ARM_CLOSURE_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "coupling_nonvacuous": coupling_nonvacuous,
        "closure_trigger_available": closure_trigger_available,
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
            + P1_BUDGET + P2_BUDGET + 2 * CLOSURE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # Readiness gate (2): closure-coupling non-vacuity among guard-passing seeds (TIGHTENED).
    cp_flags = [bool(r.get("coupling_nonvacuous", False)) for r in guard_passing]
    cp_frac = _frac(cp_flags)
    coupling_nonvacuity_met = bool(cp_frac >= MIN_FRACTION)

    # Readiness gate (3): closure-trigger available among guard-passing seeds.
    ct_flags = [bool(r.get("closure_trigger_available", False)) for r in guard_passing]
    ct_frac = _frac(ct_flags)
    closure_trigger_available_met = bool(ct_frac >= MIN_FRACTION)

    # Readiness gate (4): rule_bias_head actually trained (the anti-460d-bug gate).
    rb_flags = [bool(r.get("rule_bias_trained", False)) for r in guard_passing]
    rb_frac = _frac(rb_flags)
    rule_bias_trained_met = bool(rb_frac >= MIN_FRACTION)

    # Readiness gate (5): within-arm window non-vacuity among guard-passing seeds.
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

    c2_all = _all_guard("C2")
    c3_all = _all_guard("C3")
    # MECH-261 mode-conditioning is only EXERCISED when the automatic mode-conditioned
    # detector fired (not the Leg-A hook). The 460f autopsy found all closures hook-driven
    # (n_automatic_fires=0) -> mode-conditioning unexercised -> MECH-261 non_contributory.
    mech261_exercised = bool(guard_passing) and any(
        int(r.get("ARM_CLOSURE_ON", {}).get("n_automatic_fires", 0)) > 0
        for r in guard_passing
    )

    readiness_all_met = bool(
        contact_non_vacuity_met and rule_bias_trained_met
        and coupling_nonvacuity_met and closure_trigger_available_met
        and within_window_met
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
        # The closure-plane coupling did not engage the latch on the ON arm (the 460f
        # inert-coupling signature) -> the de-commit DV is measuring the fragile natural
        # commit-entry, not the closure-plane commitment = substrate not ready.
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "closure_coupling_not_engaged"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not closure_trigger_available_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "closure_trigger_unavailable"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not within_window_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "within_arm_windows_vacuous"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        # All five readiness gates clear -> the within-arm de-commit DV is interpretable.
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("sd034_decommit_authority_confirmed"
                           if overall_criteria_pass else "residual_decommit_authority_open")
        route_reason = ("c1_c2_c3_majority_met" if overall_criteria_pass
                        else "decommit_dv_unmet_genuine_weakens")
        direction_map = {
            "SD-034": "supports" if overall_criteria_pass else "weakens",
            "MECH-260": "supports" if c3_all else "weakens",
            # MECH-261 protected: only a verdict when mode-conditioning was actually
            # exercised; hook-driven closures leave it non_contributory (460f autopsy).
            "MECH-261": (("supports" if overall_criteria_pass else "weakens")
                         if mech261_exercised else "non_contributory"),
        }
        overall_direction = "supports" if overall_criteria_pass else "weakens"

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) rule_bias_trained={rule_bias_trained_met}"
          f" (frac={rb_frac:.3f}) coupling_nonvacuous={coupling_nonvacuity_met} (frac={cp_frac:.3f})"
          f" closure_trigger={closure_trigger_available_met} (frac={ct_frac:.3f})"
          f" within_window={within_window_met} (frac={ww_frac:.3f})"
          f" mech261_exercised={mech261_exercised}"
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
        "closure_trigger_available_met": closure_trigger_available_met,
        "closure_trigger_fraction": ct_frac,
        "within_window_met": within_window_met,
        "within_window_fraction": ww_frac,
        "mech261_exercised": mech261_exercised,
        "C2_within_all_guard_passing": c2_all,
        "C3_all_guard_passing": c3_all,
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
                    "name": "closure_coupling_nonvacuous",
                    "description": "TIGHTENED non-vacuity (the brief): ON "
                                   "sd034_n_closure_coupled_elevations > 0 AND ON "
                                   "n_sequence_completions > 0 on >= 2/3 guard seeds -- the "
                                   "closure-plane coupling engaged the latch (not the fragile "
                                   "natural commit-entry). Below floor -> substrate_not_ready_"
                                   "requeue (the 460f inert-coupling signature).",
                    "control": "ARM_CLOSURE_ON sd034_n_closure_coupled_elevations.",
                    "measured": cp_frac,
                    "threshold": MIN_FRACTION,
                    "met": coupling_nonvacuity_met,
                },
                {
                    "name": "closure_trigger_available_count",
                    "description": "ON-arm n_closures > 0 reachable on >= 2/3 guard seeds. "
                                   "Below floor -> substrate_not_ready_requeue.",
                    "control": "ARM_CLOSURE_ON n_closures > 0 (Leg-A hook + trained head).",
                    "measured": ct_frac,
                    "threshold": MIN_FRACTION,
                    "met": closure_trigger_available_met,
                },
                {
                    "name": "within_arm_window_nonvacuous",
                    "description": "ON arm produced >= C2_MIN_WINDOW_EVENTS scored "
                                   "around-closure windows with mean_pre_occ > "
                                   "WITHIN_PRE_OCC_FLOOR on >= 2/3 guard seeds -- there was "
                                   "something committed to de-commit. Below floor -> "
                                   "substrate_not_ready_requeue (nothing to measure).",
                    "control": "ARM_CLOSURE_ON n_window_events + mean_pre_closure_occ.",
                    "measured": ww_frac,
                    "threshold": MIN_FRACTION,
                    "met": within_window_met,
                },
            ],
            "criteria": [
                {"name": "C1_n_closures", "load_bearing": False, "passed": _all_guard("C1")},
                {"name": "C2_within_arm_decommit_drop", "load_bearing": True, "passed": c2_all},
                {"name": "C3_nogo_installed", "load_bearing": False, "passed": c3_all},
            ],
            "criteria_non_degenerate": {
                # The within-arm de-commit DV is non-degenerate iff all five readiness gates
                # cleared (contact, trained head, coupling engaged, closure fired, scored
                # windows with committed pre-occupancy) -- otherwise the occupancy delta is
                # structurally uninterpretable and the run self-routes substrate_not_ready.
                "C1": readiness_all_met,
                "C2": readiness_all_met,
                "C3": readiness_all_met,
            },
            "decommit_dv": {
                "definition": "C2 (load-bearing, WITHIN-ARM, part b) = on ARM_CLOSURE_ON, "
                              "mean post-closure beta-latch occupancy fraction < mean "
                              "pre-closure occupancy fraction with a >= "
                              "DECOMMIT_MIN_DROP_FRAC relative drop, over >= "
                              "C2_MIN_WINDOW_EVENTS scored CLOSURE_WINDOW-tick windows with "
                              "mean_pre_occ > WITHIN_PRE_OCC_FLOOR. Paired within-arm "
                              "around-closure statistic -- replaces 460f's underpowered "
                              "between-arm unpaired mean comparison (retained as a "
                              "secondary diagnostic: between_arm_drop).",
                "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
                "closure_window": CLOSURE_WINDOW,
                "window_min_ticks": WINDOW_MIN_TICKS,
                "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
                "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
            },
            "magnitude_lever": {
                "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
                "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
                "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
                "note": "Leg-B refractory scales with committed-run length (460f amend, "
                        "ree-v3 main 2cd0aa2) so the de-commit authority scales with the "
                        "natural-commit latch occupancy it must overcome.",
            },
            "amend_legs_under_test": {
                "leg_a_env_completion_hook": "REEAgent.notify_env_completion -> emit_closure.",
                "leg_b_decommit_hold_magnitude": "committed-run-scaled refractory (460f amend).",
                "leg_c_trained_rule_bias_head": "scaffold_train_rule_bias_head (598b REINFORCE in P1).",
                "beta_engagement_coupling": "use_closure_commit_beta_coupling.",
            },
            "mech261_note": "MECH-261 mode-conditioning direction stays non_contributory "
                            "unless the AUTOMATIC mode-conditioned detector fired "
                            "(n_automatic_fires > 0); the 460f autopsy found all closures "
                            "hook-driven, so it is protected from a self-stamped weakens.",
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
                     "+ commitment-closure-control-plane Legs A/B (env-completion hook + "
                     "de-commit hold) + Leg C (scaffold_train_rule_bias_head) + beta-engagement "
                     "coupling + the DE-COMMIT-AUTHORITY MAGNITUDE lever (committed-run-scaled "
                     "Leg-B refractory, ree-v3 main 2cd0aa2).",
        "condition": CONDITION_LABEL,
        "method_note": "Supersedes 460f. 460f ran the C2 de-commit DV for the first time "
                       "(seed-42 PASS, 2/3 FAIL) but on strong-natural-commit seeds the "
                       "closure->beta coupling was inert so the bare 5-tick refractory was "
                       "swamped by ~530-560 natural-commit elevated steps; the between-arm "
                       "unpaired DV was also underpowered (failure_autopsy_V3-EXQ-460f). 460g "
                       "(a) arms the committed-run-scaled refractory MAGNITUDE lever (the de-"
                       "commit hold now scales with the run length it must overcome), (b) "
                       "redesigns C2 to a PAIRED within-ON-arm around-closure pre-vs-post "
                       "occupancy delta, and tightens the non-vacuity gate to require "
                       "sd034_n_closure_coupled_elevations > 0 on scored seeds. Five readiness "
                       "gates self-route substrate_not_ready_requeue when unmet -- never a "
                       "false weakens. MECH-261 stays non_contributory unless the automatic "
                       "mode-conditioned detector fires (460f autopsy protection).",
        "arm_note": "ARM_CLOSURE_ON (full closure + env hook + de-commit MAGNITUDE hold + "
                    "bistable + coupling, on the TRAINED rule_bias_head; load-bearing within-arm "
                    "C2) vs ARM_CLOSURE_OFF (same trained weights, closure off; between-arm "
                    "diagnostic only).",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "c1_min_closures": C1_MIN_CLOSURES,
            "c3_min_nogo": C3_MIN_NOGO,
            "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
            "closure_window": CLOSURE_WINDOW,
            "window_min_ticks": WINDOW_MIN_TICKS,
            "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
            "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
            "rule_bias_mean_floor": RULE_BIAS_MEAN_FLOOR,
            "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
            "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
            "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "closure_eval_episodes_per_arm": CLOSURE_EVAL_EPISODES,
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "scaffold_train_rule_bias_head": True,
            "config_basis": "V3-EXQ-603n + Leg C (2026-06-16) + de-commit-magnitude amend (2026-06-19)",
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
