"""
V3-EXQ-466e (EXP-0162 behavioural): SD-034 satisficing / residue discharge,
measured on a foraging-competent agent built through the FULL
scaffolded_sd054_onboarding curriculum at the 603n config. Supersedes
V3-EXQ-466d.

ROOT-CAUSE FIX (confirmed failure_autopsy_V3-EXQ-466d_2026-06-24, interactive
gate -- non_contributory, NOT a falsification, NOT a substrate ceiling):
466d had closures FORM (C1_n_closures PASS -- the 466c Leg-A env-completion hook
is fixed) but C2_discharge_events = 0 on 3/3 seeds because the residue field is
EMPTY for the whole run. ResidueField.discharge_domain (field.py:621) is fully
implemented and correctly wired into ClosureOperator._fire
(closure_operator.residue = self.residue_field, agent.py:1577) and fires on
every closure, but it returns 0 when active_mask is empty (field.py:671), and the
field is never populated: neither the scaffold curriculum nor the 466d eval loop
ever calls agent.update_residue() -- the SOLE path to add_residue / active RBF
centers (agent.py:7390 -> field.py:123). agent.reset() explicitly does NOT reset
residue (agent.py:2354), so the emptiness is the never-exercised accumulation
path, not a per-episode wipe. C2 was therefore DEGENERATE: pinned at 0 by test
construction, independent of closure behaviour (the V3-EXQ-642 invalid-precondition
pattern one leg past the 460c hook gap).

466e is a HARNESS fix (substrate_queue action = none; the discharge substrate
discharge_domain is built + wired + fires; nothing to enrich). Three changes:

  (1) WIRE agent.update_residue(harm_signal, world_delta=None) each step in BOTH
      a residue-priming foraging pass (_prime_residue_field -- the in-script
      analog of "scaffold P1/P2 training" residue accumulation, since the shared
      scaffold module is out of scope for a harness fix) AND in the
      _eval_residue_discharge eval loop, so the residue field accumulates active
      RBF centers reflecting the foraging trajectory BEFORE a closure fires.
      update_residue accumulates only on harm steps (harm_signal < 0, agent.py:7525),
      which the 603n hazard substrate produces; it also runs e3.post_action_update,
      completing the waking post-action loop 466d omitted. Residue persists across
      agent.reset() (invariant), so the priming pass and the early eval episodes
      populate the field for the closures that fire later.

  (2) CAPTURE notify_env_completion's returned ClosureEvent so the Leg-A-hook
      closures' discharge is counted. In 466d ALL closures came via
      notify_env_completion (n_closures == n_env_completion_hook_calls == 3/7/5),
      but discharge_events counted ONLY closures fired during select_action and
      DISCARDED the notify_env_completion event -- so even a populated field would
      have scored C2 = 0. notify_env_completion returns the ClosureEvent (agent.py:
      7206, returns _env_closure_evt) carrying residue_centers_discharged
      (closure_operator.py:216), so 466e reads it and increments discharge_events
      when its discharge >= 1. The select_action discharge check is retained for
      any auto-stability closure (general; that path was silent in 466d).

  (3) ADD a residue-field-populated NON-VACUITY GATE: on the ON arm of the
      guard-passing seeds, residue_active_peak (max active_mask.sum() over the
      eval) > 0 on >= 2/3 seeds. If unmet -> self-route substrate_not_ready_requeue
      (non_contributory, non_degenerate=False), NEVER a false weakens -- this is
      the precondition gate the 466d self-route lacked. It detects the 466d
      empty-field artifact directly: an empty field -> discharge_domain pinned at 0
      by construction -> re-queue at an adequate harness, not a SD-034 verdict.

CLAIM SCOPE: claim_ids = [SD-034] ONLY. MECH-094 is DROPPED (466d's autopsy:
the discharge is waking-only -- no simulation/replay path -- so MECH-094 (a stable
claim, conf 0.868, 23 supports vs 1 weakens) is NOT exercised; tagging it would
risk a spurious weakens). Re-add MECH-094 only with an explicit simulation-tagged
discharge control arm.

WHAT THIS MEASURES (unchanged from 466b/c/d): the ocd4 satisficing / over-checking
dissociation. With closure PRESENT, a satisficing completion discharges rule-domain
residue (multiplicative decay with a 1e-6 floor; discharge_events accumulate). With
closure ABSENT (same weights), residue accumulates unchecked (over-checking) -- zero
discharge.

ARMS (one curriculum build per seed, residue prime, two frozen-policy evals):
  ARM_CLOSURE_ON   -- full closure + bistable + Leg-A hook ON. Expect
                      n_closures > 0 AND discharge_events >= 1 (on a populated field).
  ARM_CLOSURE_OFF  -- same trained weights, closure OFF (hook also OFF). Zero
                      closure fires, zero residue discharge (C3) even though the
                      field is populated -- the strong negative control.

CONTACT GUARD (603n G2 + G3): per-seed contact_rate > 0 AND z_goal_norm_at_contact_peak
  > 0.4; < 2/3 seeds passing -> substrate_not_ready_requeue (non_contributory).

COMMITMENT / COMPLETION NON-VACUITY GATE: before scoring, assert the ON arm
  committed (total_beta_elevated > 0) AND reached completions
  (n_sequence_completions > 0) on >= 2/3 guard-passing seeds. Below floor ->
  substrate_not_ready_requeue (non_contributory), NEVER a false weakens.

RESIDUE NON-VACUITY GATE (the 466e addition): ON arm residue_active_peak > 0 on
  >= 2/3 guard-passing seeds. Below floor -> substrate_not_ready_requeue
  (non_contributory, non_degenerate=False), NEVER a false weakens.

PRE-REGISTERED ACCEPTANCE (constants; PASS = majority 2/3 guard seeds):
  C1  ARM_CLOSURE_ON  n_closures >= 1
  C2  ARM_CLOSURE_ON  discharge_events >= 1   (now counts notify-hook closures too)
  C3  ARM_CLOSURE_OFF n_closures == 0 AND discharge_events == 0
  Per-seed PASS = C1 AND C2 AND C3.

PER-CLAIM DIRECTION:
  any readiness precondition (contact / completion / residue) NOT met
      -> non_contributory (substrate_not_ready_requeue), non_degenerate=False.
  all met:  SD-034 = supports if overall PASS else weakens.

claim_ids: SD-034.
experiment_purpose: evidence
supersedes: V3-EXQ-466d

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_466e_sd034_satisficing_residue_discharge_behavioural"
QUEUE_ID = "V3-EXQ-466e"
CLAIM_IDS: List[str] = ["SD-034"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-466d"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_SATISFICING_RESIDUE_DISCHARGE_HARNESS_FIXED"

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
DISCHARGE_EVAL_EPISODES = 15
RESIDUE_PRIME_EPISODES = 5
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
MIN_FRACTION = 2.0 / 3.0
C1_MIN_CLOSURES = 1
C2_MIN_DISCHARGE_EVENTS = 1
RESIDUE_ACTIVE_FLOOR = 1   # residue non-vacuity: >= 1 active center at peak


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
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
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
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    # Leg A: explicit env-completion hook (the 466c n_closures=0 root-cause fix).
    cfg.use_closure_env_completion_hook = True
    return cfg


def _build_closure_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint
    tolerance-band completion so the SD-034 closure operator has completions to
    fire its residue discharge on. subgoal_mode=True is LOAD-BEARING."""
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
    cfg_off = copy.deepcopy(trained_agent.config)
    cfg_off.use_closure_operator = False
    # Explicitly disable the Leg A hook on the OFF clone so the eval harness
    # cannot accidentally call notify_env_completion into a None operator.
    cfg_off.use_closure_env_completion_hook = False
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


def _residue_active_count(agent: REEAgent) -> int:
    """Number of ACTIVE RBF centers in the residue field (the non-vacuity
    quantity: discharge_domain returns 0 when this is 0, field.py:671)."""
    try:
        return int(agent.residue_field.rbf_field.active_mask.sum().item())
    except Exception:
        return 0


def _prime_residue_field(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> int:
    """Residue-priming foraging pass -- the in-script analog of 'scaffold P1/P2
    training' residue accumulation (the shared scaffold module is out of scope
    for a HARNESS fix). Runs the trained agent through the closure env and calls
    agent.update_residue(harm_signal) each step so the persistent residue field
    accumulates active RBF centers reflecting the foraging trajectory BEFORE any
    closure fires in the measured eval. Does NOT call notify_env_completion (no
    Leg-A closure fires here) and does NOT measure discharge -- it only populates
    the field. agent.reset() does not reset residue (invariant, agent.py:2354),
    so this accumulation persists into the eval arms.

    Returns the active-center count after priming (diagnostic)."""
    agent.eval()
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream
    world_dim = agent.config.latent.world_dim
    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(steps_per_episode):
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
                _, _harm, done, info, obs_dict = env.step(action_idx)
                # Harm-gated residue accumulation (the populator 466d lacked).
                agent.update_residue(harm_signal=float(_harm), world_delta=None)
                if done:
                    break
    return _residue_active_count(agent)


def _eval_residue_discharge(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for SD-034 residue-discharge behaviour.
    Tracks closure fires, discharge events (closures with
    residue_centers_discharged >= 1 -- counted on BOTH the select_action
    auto-stability path AND the notify_env_completion Leg-A path), mean active-
    weight reduction, beta-elevated steps, sequence completions, the Leg-A hook
    call count, and the residue-field non-vacuity quantity (active-center peak).

    KEY 466e CHANGES vs 466d:
      (1) calls agent.update_residue(harm_signal) each step (after env.step) so
          the residue field accumulates active RBF centers -- the populator 466d
          omitted, which left discharge_domain pinned at 0 on an empty active_mask.
      (2) CAPTURES notify_env_completion's returned ClosureEvent and counts its
          residue_centers_discharged. 466d discarded that event, so the Leg-A-hook
          closures (which were ALL the closures in 466d) never contributed to
          discharge_events.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    discharge_events = 0
    weight_reductions: List[float] = []
    total_committed_steps = 0
    total_beta_elevated = 0
    n_sequence_completions = 0
    n_env_completion_hook_calls = 0
    n_closures_with_residue = 0
    residue_active_peak = _residue_active_count(agent)

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, feed_harm
                )

                n_closures_before = (
                    int(agent.closure_operator._n_closures) if has_closure else 0
                )

                def _w_sum() -> float:
                    if has_closure and agent.residue_field.rbf_field.active_mask.any():
                        return float(
                            agent.residue_field.rbf_field.weights.data[
                                agent.residue_field.rbf_field.active_mask
                            ].sum().item()
                        )
                    return 0.0

                w_sum_before = _w_sum()

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # --- discharge via the select_action auto-stability path ---
                if has_closure:
                    fired_now = int(agent.closure_operator._n_closures) - n_closures_before
                    if fired_now > 0 and agent.closure_operator._event_log:
                        last_event = agent.closure_operator._event_log[-1]
                        active_at_fire = _residue_active_count(agent)
                        if active_at_fire > 0:
                            n_closures_with_residue += 1
                        if last_event.residue_centers_discharged >= 1:
                            discharge_events += 1
                            w_sum_after = _w_sum()
                            weight_reductions.append(w_sum_before - w_sum_after)

                if agent.e3._committed_trajectory is not None:
                    total_committed_steps += 1
                if bool(agent.beta_gate.is_elevated):
                    total_beta_elevated += 1

                _, _harm, done, info, obs_dict = env.step(action_idx)
                if info.get("transition_type") == "sequence_complete":
                    n_sequence_completions += 1
                    # Leg A fix: explicitly route the env completion into
                    # emit_closure via the registered hook, AND capture the
                    # returned ClosureEvent so the discharge it performed is
                    # counted (466d discarded the event -> C2 could never fire
                    # for notify-hook closures even on a populated field).
                    if has_closure:
                        n_env_completion_hook_calls += 1
                        w_sum_pre_notify = _w_sum()
                        closure_evt = agent.notify_env_completion(action_class=action_idx)
                        if closure_evt is not None and closure_evt.fired:
                            active_at_fire = _residue_active_count(agent)
                            if active_at_fire > 0:
                                n_closures_with_residue += 1
                            if closure_evt.residue_centers_discharged >= 1:
                                discharge_events += 1
                                w_sum_post_notify = _w_sum()
                                weight_reductions.append(
                                    w_sum_pre_notify - w_sum_post_notify
                                )

                # Harm-gated residue accumulation (the populator 466d lacked).
                # Runs AFTER the closure checks so residue from THIS step feeds the
                # closures that fire on SUBSEQUENT steps; residue persists across
                # agent.reset() so the field stays populated for the whole eval.
                agent.update_residue(harm_signal=float(_harm), world_delta=None)
                residue_active_peak = max(residue_active_peak, _residue_active_count(agent))

                if done:
                    break

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre if has_closure else 0
    )
    mean_reduction = float(np.mean(weight_reductions)) if weight_reductions else 0.0
    return {
        "n_closures": n_closures,
        "discharge_events": discharge_events,
        "mean_residue_weight_reduction": mean_reduction,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "n_sequence_completions": n_sequence_completions,
        "n_env_completion_hook_calls": n_env_completion_hook_calls,
        "n_closures_with_residue": n_closures_with_residue,
        "residue_active_peak": residue_active_peak,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    empty = {
        "n_closures": 0, "discharge_events": 0, "mean_residue_weight_reduction": 0.0,
        "total_committed_steps": 0, "total_beta_elevated": 0,
        "n_sequence_completions": 0, "n_env_completion_hook_calls": 0,
        "n_closures_with_residue": 0, "residue_active_peak": 0,
        "n_eval_episodes": 0, "closure_present": False,
    }
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "residue_prime_active_count": 0,
        "residue_populated": False,
        "ARM_CLOSURE_ON": dict(empty), "ARM_CLOSURE_OFF": dict(empty),
        "criteria": {"C1": False, "C2": False, "C3": False},
        "commitment_completion_non_vacuity": False,
        "pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else DISCHARGE_EVAL_EPISODES
    prime_eps = 2 if dry_run else RESIDUE_PRIME_EPISODES

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
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

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

    # RESIDUE PRIME: populate the persistent residue field on the trained agent
    # via a foraging pass that calls update_residue each step (no closures fire),
    # so the field is non-empty from the FIRST closure of the measured eval.
    prime_active = _prime_residue_field(
        agent, closure_env, scaffold_cfg, device, prime_eps, steps_per_ep
    )
    done += prime_eps
    print(f"  [train] residue_prime seed={seed} ep {done}/{total_eps}"
          f" active_centers={prime_active}", flush=True)

    print(f"Seed {seed} Condition ARM_CLOSURE_ON", flush=True)
    arm_on = _eval_residue_discharge(
        agent, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    print(f"Seed {seed} Condition ARM_CLOSURE_OFF", flush=True)
    agent_off = _clone_closure_off(agent, device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    arm_off = _eval_residue_discharge(
        agent_off, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    c1 = arm_on["n_closures"] >= C1_MIN_CLOSURES
    c2 = arm_on["discharge_events"] >= C2_MIN_DISCHARGE_EVENTS
    c3 = bool(arm_off["n_closures"] == 0 and arm_off["discharge_events"] == 0)
    commitment_completion_non_vacuity = bool(
        arm_on["total_beta_elevated"] > 0 and arm_on["n_sequence_completions"] > 0
    )
    residue_populated = bool(arm_on["residue_active_peak"] >= RESIDUE_ACTIVE_FLOOR)
    seed_pass = bool(c1 and c2 and c3)

    print(f"  [train] discharge_eval seed={seed} ep {done}/{total_eps}"
          f" c1={c1} c2={c2} c3={c3}"
          f" n_closures={arm_on['n_closures']} discharge={arm_on['discharge_events']}"
          f" hook_calls={arm_on['n_env_completion_hook_calls']}"
          f" residue_active_peak={arm_on['residue_active_peak']}"
          f" closures_with_residue={arm_on['n_closures_with_residue']}"
          f" completions={arm_on['n_sequence_completions']} beta_elev={arm_on['total_beta_elevated']}",
          flush=True)
    print(f"verdict: {'PASS' if (guard_pass and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} complete_nonvacuity={commitment_completion_non_vacuity}"
          f" residue_populated={residue_populated}"
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
        "residue_prime_active_count": int(prime_active),
        "residue_populated": residue_populated,
        "ARM_CLOSURE_ON": arm_on,
        "ARM_CLOSURE_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "commitment_completion_non_vacuity": commitment_completion_non_vacuity,
        "pass": seed_pass,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + 2 + 2 * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + RESIDUE_PRIME_EPISODES + 2 * DISCHARGE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    cc_flags = [bool(r.get("commitment_completion_non_vacuity", False)) for r in guard_passing]
    cc_frac = _frac(cc_flags)
    completion_non_vacuity_met = bool(cc_frac >= MIN_FRACTION)

    res_flags = [bool(r.get("residue_populated", False)) for r in guard_passing]
    res_frac = _frac(res_flags)
    residue_non_vacuity_met = bool(res_frac >= MIN_FRACTION)

    seed_pass_flags = [bool(r.get("pass", False)) for r in guard_passing]
    n_pass = sum(1 for f in seed_pass_flags if f)
    pass_frac = _frac(seed_pass_flags)
    overall_criteria_pass = bool(pass_frac >= MIN_FRACTION)

    def _all_guard(crit_key: str) -> bool:
        return bool(guard_passing) and all(
            r.get("criteria", {}).get(crit_key) for r in guard_passing
        )

    non_degenerate = True
    degeneracy_reason = ""
    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
        overall_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "603n contact guard unmet on majority of seeds -- the discharge test "
            "never reached a foraging-competent agent."
        )
    elif not completion_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "commitment_or_completion_not_engaged"
        overall_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "ON arm did not commit (total_beta_elevated>0) AND complete "
            "(n_sequence_completions>0) on majority of guard seeds -- closure had no "
            "opportunity to fire."
        )
    elif not residue_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "residue_field_unpopulated"
        overall_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "Residue field stayed empty (residue_active_peak < 1) on majority of "
            "guard seeds -- the 466d empty-field artifact. discharge_domain is pinned "
            "at 0 by an empty active_mask (field.py:671), independent of closure "
            "behaviour, so C2 cannot fire. Re-queue at an adequate residue-accumulation "
            "harness; NOT an SD-034 verdict."
        )
    else:
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("sd034_satisficing_discharge_confirmed"
                           if overall_criteria_pass else "residual_discharge_open")
        route_reason = "c1_c2_c3_majority_met" if overall_criteria_pass else "criteria_unmet_genuine_weakens"
        overall_direction = "supports" if overall_criteria_pass else "weakens"

    direction_map = {"SD-034": overall_direction}

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) completion_non_vacuity={completion_non_vacuity_met}"
          f" (frac={cc_frac:.3f}) residue_non_vacuity={residue_non_vacuity_met}"
          f" (frac={res_frac:.3f}) criteria_pass={overall_criteria_pass}"
          f" ({n_pass}/{len(guard_passing)}) -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "completion_non_vacuity_met": completion_non_vacuity_met,
        "completion_non_vacuity_fraction": cc_frac,
        "residue_non_vacuity_met": residue_non_vacuity_met,
        "residue_non_vacuity_fraction": res_frac,
        "criteria_pass_fraction": pass_frac,
        "n_seeds_pass": n_pass,
        "overall_pass": bool(contact_non_vacuity_met and completion_non_vacuity_met
                             and residue_non_vacuity_met and overall_criteria_pass),
        "per_seed_guard_pass": guard_flags,
        "per_seed_criteria_pass": [bool(r.get("pass", False)) for r in per_seed],
        "per_seed_residue_populated": [bool(r.get("residue_populated", False)) for r in per_seed],
        "route_reason": route_reason,
    }

    all_preconditions_met = bool(
        contact_non_vacuity_met and completion_non_vacuity_met and residue_non_vacuity_met
    )

    result: Dict[str, Any] = {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "non_degenerate": non_degenerate,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": [
                {
                    "name": "foraging_contact_guard",
                    "description": "603n G2+G3 contact guard on >= 2/3 seeds.",
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                },
                {
                    "name": "commitment_and_completion_engaged",
                    "description": "ON arm committed (total_beta_elevated > 0) AND reached "
                                   "completions (n_sequence_completions > 0) on >= 2/3 "
                                   "guard-passing seeds, so closure had a hook-based "
                                   "opportunity to fire.",
                    "control": "fraction of guard-passing seeds with ON commitment AND completions.",
                    "measured": cc_frac,
                    "threshold": MIN_FRACTION,
                    "met": completion_non_vacuity_met,
                },
                {
                    "name": "residue_field_populated_at_closure",
                    "description": "ON arm residue_active_peak >= 1 (the residue field "
                                   "accumulated active RBF centers) on >= 2/3 guard-passing "
                                   "seeds, so discharge_domain has in-domain centers to act "
                                   "on. The 466d empty-field artifact reads residue_active_peak "
                                   "= 0 and self-routes substrate_not_ready_requeue here.",
                    "control": "update_residue wired into the residue-priming pass + eval loop; "
                               "active_mask.sum() is the SAME quantity discharge_domain gates "
                               "on (field.py:671).",
                    "measured": res_frac,
                    "threshold": MIN_FRACTION,
                    "met": residue_non_vacuity_met,
                },
            ],
            "criteria": [
                {"name": "C1_n_closures", "load_bearing": True, "passed": _all_guard("C1")},
                {"name": "C2_discharge_events", "load_bearing": True, "passed": _all_guard("C2")},
                {"name": "C3_off_no_closure_no_discharge", "load_bearing": True,
                 "passed": _all_guard("C3")},
            ],
            "criteria_non_degenerate": {
                "C1": all_preconditions_met,
                "C2": all_preconditions_met,
                "C3": all_preconditions_met,
            },
            "contact_guard": {
                "definition": "per-seed P2 contact_rate > 0 AND z_goal_norm_at_contact_peak > 0.4; "
                              "< 2/3 seeds -> substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "residue_non_vacuity_gate": {
                "definition": "ON residue_active_peak >= 1 on >= 2/3 guard-passing seeds. "
                              "Below floor -> substrate_not_ready_requeue (non_contributory, "
                              "non_degenerate=False), NEVER a false weakens. Detects the 466d "
                              "empty-field artifact: discharge_domain returns 0 on an empty "
                              "active_mask (field.py:671) regardless of closure behaviour.",
                "min_fraction": MIN_FRACTION,
                "residue_active_floor": RESIDUE_ACTIVE_FLOOR,
            },
            "completion_non_vacuity_gate": {
                "definition": "ON total_beta_elevated > 0 AND n_sequence_completions > 0 on "
                              ">= 2/3 guard-passing seeds. Below floor -> substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "c1_min_closures": C1_MIN_CLOSURES,
                "c2_min_discharge_events": C2_MIN_DISCHARGE_EVENTS,
            },
        },
        "per_seed": per_seed,
    }
    if not non_degenerate:
        result["degeneracy_reason"] = degeneracy_reason
    return result


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
        "supersedes": SUPERSEDES,
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": (
            "scaffolded_sd054_onboarding (full curriculum; 603n config; ready=true 2026-06-11) "
            "+ commitment-closure-control-plane Leg A env-completion hook "
            "(use_closure_env_completion_hook=True; IMPLEMENTED 2026-06-12) "
            "+ bistable BetaGate + SD-034 ClosureOperator + SD-033a LateralPFC "
            "+ SD-032 dACC/salience + subgoal_mode waypoint tolerance-band completion env primitive. "
            "Substrate UNCHANGED vs 466d -- 466e is a HARNESS fix (substrate_queue action=none)."
        ),
        "condition": CONDITION_LABEL,
        "method_note": (
            "466e supersedes 466d (confirmed failure_autopsy_V3-EXQ-466d_2026-06-24, "
            "non_contributory, interactive gate). 466d had C1_n_closures PASS but "
            "C2_discharge_events = 0 on 3/3 seeds because the residue field was EMPTY for "
            "the whole run: discharge_domain (field.py:621) is implemented + wired "
            "(closure_operator.residue = residue_field, agent.py:1577) and fires on every "
            "closure, but neither the scaffold curriculum nor the 466d eval ever called "
            "agent.update_residue() (the sole add_residue path), so discharge_domain returned "
            "0 from an empty active_mask (field.py:671) regardless of closure behaviour -- C2 "
            "was DEGENERATE (pinned by construction; the V3-EXQ-642 invalid-precondition "
            "pattern). 466e fixes the HARNESS: (1) wires agent.update_residue(harm_signal) into "
            "a residue-priming foraging pass (_prime_residue_field) + the _eval_residue_discharge "
            "eval loop so the field accumulates active RBF centers BEFORE closures fire; "
            "(2) CAPTURES notify_env_completion's returned ClosureEvent so the Leg-A-hook "
            "closures' residue_centers_discharged is counted -- in 466d ALL closures came via "
            "the notify hook (n_closures == n_env_completion_hook_calls), but discharge_events "
            "counted only select_action closures and discarded the notify event, so even a "
            "populated field would have scored C2 = 0; (3) adds a residue-field-populated "
            "non-vacuity gate that self-routes substrate_not_ready_requeue (non_degenerate=False) "
            "when residue stays empty. ARM_CLOSURE_OFF clone explicitly sets "
            "use_closure_env_completion_hook=False so the off-arm cannot route completions into "
            "a None closure_operator (C3 stays a strong negative control on a populated field)."
        ),
        "readiness_note": (
            "THREE non-vacuity gates self-route a substrate-not-ready read to non_contributory "
            "(never a false weakens): the 603n contact guard + the commitment+completion gate + "
            "the NEW residue-field-populated gate. Only when all three are met do C1/C2/C3 score "
            "SD-034 (PASS=supports / criteria-unmet=weakens)."
        ),
        "arm_note": (
            "ARM_CLOSURE_ON (use_closure_operator=True + use_closure_env_completion_hook=True + "
            "bistable) vs ARM_CLOSURE_OFF (same trained weights; closure=False; hook=False). "
            "Both arms share the trained curriculum build (one build per seed) + a primed, "
            "populated residue field; ARM_CLOSURE_OFF cannot discharge (no closure operator) "
            "so C3 holds even with a populated field. No separable expensive OFF baseline to "
            "mint (the curriculum build is shared, not per-arm)."
        ),
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "c1_min_closures": C1_MIN_CLOSURES,
            "c2_min_discharge_events": C2_MIN_DISCHARGE_EVENTS,
            "residue_active_floor": RESIDUE_ACTIVE_FLOOR,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "residue_prime_episodes": RESIDUE_PRIME_EPISODES,
            "discharge_eval_episodes_per_arm": DISCHARGE_EVAL_EPISODES,
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "config_basis": "V3-EXQ-603n",
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
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
            dry_run=args.dry_run,
        )
