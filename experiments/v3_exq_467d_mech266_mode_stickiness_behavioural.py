"""
V3-EXQ-467d (EXP-0163 behavioural): MECH-266 mode stickiness / Hold-decay
dose-response, measured on a foraging-competent agent built through the FULL
scaffolded_sd054_onboarding curriculum at the 603n config, WITH the new
mode-governance-engagement substrate (use_external_task_drive). Successor to
V3-EXQ-467c (NOT a supersede).

ROUTING: 467c landed on the foraging-competent 603n substrate but FAILed
non_contributory (substrate_ceiling): fraction_in_external_task = 0.0 / mean_dwell an
episode-length artifact (total_steps/n_runs) / n_switches == n_episodes for every
ratio, so the MECH-266 dose-response could not express (the contested mode was never
occupied; failure_autopsy_SD-034-closure-cluster-ext_2026-06-12 sub-cluster B). The
mode-governance-engagement substrate (landed 2026-06-13, ree-v3
REEConfig.use_external_task_drive) adds the external_task salience SOURCE the
SalienceCoordinator lacked: a committed-goal-pursuit engagement signal registered in
the coordinator's affinity (-> external_task) AND salience (switch-into-external_task
aggregate). 467d enables it.

TWO SUBSTRATE CHANGES vs 467c:
  (1) use_external_task_drive=True in the agent config (the ONLY agent-config change --
      everything else is bit-identical to 467c so the comparison isolates the drive).
  (2) the non-vacuity readiness gate is RE-STATED: the contested mode must actually be
      occupied across the sweep (min over ratio arms of fraction_in_external_task >
      OCCUPANCY_FLOOR) BEFORE the dose-response criteria are scored -- replacing the
      467c n_switches>=1 gate that certified episode-boundary settles.

WHAT THIS MEASURES (unchanged from 467c): the MECH-266 hysteresis dose-response. Sweep
the uniform exit-rail ratio across r in {0.10, 0.50, 1.00, 1.50, 2.00}. A lower exit
rail (r=0.10) makes modes harder to leave -> longer dwell + fewer switches; a higher
rail (r=2.00) makes them easy to leave -> shorter dwell. Mean dwell should be monotone
non-increasing in r, with the lowest-r dwell at least C2_MIN_DWELL_RATIO x the highest-r
dwell.

CONDITIONS (one curriculum build per seed; one frozen-policy eval per ratio on a clone):
  r in RATIOS, each a clone of the SAME trained weights with set_hysteresis_ratio(r).
  (The committed_mode_curriculum O-2 forced-rv contrast arm is dropped.)
  NOTE: use_closure_operator is OFF -- closure injects a confounding mode-switch signal.

CONTACT GUARD (603n G2 + G3): per-seed contact_rate > 0 AND z_goal_norm_at_contact_peak
  > 0.4; < 2/3 seeds passing -> substrate_not_ready_requeue (non_contributory).

EXTERNAL_TASK OCCUPANCY NON-VACUITY GATE (re-states the 467c n_switches gate -- the SAME
  occupancy statistic the dose-response operates over, per the V3-EXQ-643 same-statistic
  rule): before scoring C1/C2, assert the contested external_task mode is genuinely
  occupied across the sweep -- per-seed occupancy_min = min over ratio arms of
  fraction_in_external_task > OCCUPANCY_FLOOR -- on >= 2/3 guard-passing seeds. Below
  floor -> substrate_not_ready_requeue (non_contributory), NEVER a false weakens. A
  zero-occupancy read is "mode governance not engaged" (the unresolved
  scaffolded_sd054_onboarding nav-competence / Stage-H dependency may legitimately keep
  the agent from foraging long enough), not "MECH-266 falsified".

PRE-REGISTERED ACCEPTANCE (constants; PASS = majority 2/3 guard seeds):
  C1  mean_dwell is monotone non-increasing across the r sweep
  C2  dwell(r=0.10) >= C2_MIN_DWELL_RATIO x max(dwell(r=2.00), 1.0)
  Per-seed PASS = C1 AND C2 (scored only when the occupancy non-vacuity gate is met).

PER-CLAIM DIRECTION:
  contact / occupancy non-vacuity NOT met -> non_contributory (substrate_not_ready_requeue).
  both met:  MECH-266 = supports if overall PASS else weakens (the hysteresis dose-response);
             SD-032a = supports if overall PASS else weakens (the SalienceCoordinator
             mode register the ratio acts on).

claim_ids: MECH-266, SD-032a.
experiment_purpose: evidence
predecessor: V3-EXQ-467c (NOT a supersede).

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
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_467d_mech266_mode_stickiness_behavioural"
QUEUE_ID = "V3-EXQ-467d"
CLAIM_IDS: List[str] = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "evidence"
PREDECESSOR = "V3-EXQ-467c (successor, NOT supersede)"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_MODE_STICKINESS"

STICKY_MODE = "external_task"
RATIOS = [0.10, 0.50, 1.00, 1.50, 2.00]
C2_MIN_DWELL_RATIO = 2.0
# mode-governance-engagement: the contested external_task mode must be genuinely
# occupied across the sweep before the dose-response is scored. Replaces the 467c
# n_switches>=1 gate (which counted episode-boundary settles).
OCCUPANCY_FLOOR = 0.10

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
MODE_EVAL_EPISODES = 12
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
        use_closure_operator=False,
        # mode-governance-engagement substrate (2026-06-13): the external_task salience
        # SOURCE the SalienceCoordinator lacked. The ONLY agent-config change vs 467c.
        # Calibrated weights (substrate defaults 1.0; this experiment opts into a
        # stronger setting so a committed foraging step decisively contests external_task
        # vs the dACC internal_planning push -- verified non-vacuous: committed engagement
        # ~1.0 -> aggregate 3.2 > enter_threshold 1.0 -> external_task wins + switches;
        # uncommitted ~0.1 -> internal_planning wins).
        use_external_task_drive=True,
        external_task_drive_affinity_weight=3.0,
        external_task_drive_salience_weight=2.0,
        external_task_drive_commit_weight=1.0,
        external_task_drive_proximity_weight=1.0,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_dual_cue_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH the GAP-3 dual_cue primitive.
    dual_cue requires SD-049, which the scaffolded _sd049_kwargs enables."""
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
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(1, 2),
        **_sd049_kwargs(scaffold_cfg),
    )


def _clone_for_arm(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    cfg = copy.deepcopy(trained_agent.config)
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        agent.load_state_dict(state, strict=False)
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    return agent


def _eval_mode_dwell(
    agent: REEAgent,
    env: CausalGridWorldV2,
    ratio: float,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for MECH-266 mode-dwell measurement (ported
    from 467b, using _sense_with_optional_harm). Applies set_hysteresis_ratio(ratio)
    before running; tracks dwell runs + switches."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience
    coord.set_hysteresis_ratio(ratio)
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    all_run_lengths: List[int] = []
    total_switches = 0
    total_steps = 0
    external_task_steps = 0

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_mode = coord.current_mode
            current_run = 1

            for _ in range(steps_per_ep):
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

                new_mode = coord.current_mode
                if new_mode == STICKY_MODE:
                    external_task_steps += 1
                if new_mode != prev_mode:
                    all_run_lengths.append(current_run)
                    total_switches += 1
                    current_run = 1
                    prev_mode = new_mode
                else:
                    current_run += 1

                total_steps += 1
                _, _harm, done, _info, obs_dict = env.step(action_idx)
                if done:
                    all_run_lengths.append(current_run)
                    current_run = 0
                    break

            if current_run > 0:
                all_run_lengths.append(current_run)

    mean_dwell = (
        float(sum(all_run_lengths)) / len(all_run_lengths)
        if all_run_lengths else float(steps_per_ep)
    )
    frac_task = external_task_steps / total_steps if total_steps else 0.0
    return {
        "ratio": ratio,
        "mean_dwell": round(mean_dwell, 3),
        "n_switches": total_switches,
        "fraction_in_external_task": round(frac_task, 4),
        "n_runs": len(all_run_lengths),
        "total_steps": total_steps,
        "n_episodes": n_eps,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    empty_conds = [
        {"ratio": r, "mean_dwell": 0.0, "n_switches": 0,
         "fraction_in_external_task": 0.0, "n_runs": 0,
         "total_steps": 0, "n_episodes": 0}
        for r in RATIOS
    ]
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "condition_results": empty_conds,
        "occupancy_min": 0.0,
        "occupancy_non_vacuity": False,
        "criteria": {"C1": False, "C2": False},
        "pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else MODE_EVAL_EPISODES

    probe_env = _build_dual_cue_env(scaffold_cfg)
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

    dual_env = _build_dual_cue_env(scaffold_cfg)
    dual_env.reset()

    condition_results: List[Dict[str, Any]] = []
    for r in RATIOS:
        print(f"Seed {seed} Condition r={r}", flush=True)
        agent_cond = _clone_for_arm(agent, device)
        cond = _eval_mode_dwell(agent_cond, dual_env, r, scaffold_cfg, device, eval_eps, steps_per_ep)
        condition_results.append(cond)
        done += eval_eps

    dwells = [c["mean_dwell"] for c in condition_results]
    c1 = all(dwells[i] >= dwells[i + 1] for i in range(len(dwells) - 1))
    dwell_low = condition_results[0]["mean_dwell"]
    dwell_high = condition_results[-1]["mean_dwell"]
    c2 = dwell_low >= C2_MIN_DWELL_RATIO * max(dwell_high, 1.0)
    # mode-governance-engagement occupancy non-vacuity gate (replaces the 467c
    # n_switches gate): the contested mode must be occupied across the sweep.
    occupancy_min = min(c["fraction_in_external_task"] for c in condition_results)
    occupancy_non_vacuity = bool(occupancy_min > OCCUPANCY_FLOOR)
    seed_pass = bool(c1 and c2)

    print(f"  [train] dwell_sweep seed={seed} ep {done}/{total_eps}"
          f" c1={c1} c2={c2} occupancy_min={occupancy_min:.4f}"
          f" occ_nonvacuity={occupancy_non_vacuity} dwells={dwells}"
          f" fracs={[c['fraction_in_external_task'] for c in condition_results]}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and occupancy_non_vacuity and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} occupancy_nonvacuity={occupancy_non_vacuity}"
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
        "condition_results": condition_results,
        "occupancy_min": round(float(occupancy_min), 4),
        "occupancy_non_vacuity": occupancy_non_vacuity,
        "criteria": {"C1": c1, "C2": c2},
        "pass": seed_pass,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_ratios = len(RATIOS)
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + n_ratios * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + n_ratios * MODE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    occ_flags = [bool(r.get("occupancy_non_vacuity", False)) for r in guard_passing]
    occ_frac = _frac(occ_flags)
    occupancy_non_vacuity_met = bool(occ_frac >= MIN_FRACTION)

    seed_pass_flags = [bool(r.get("pass", False)) for r in guard_passing]
    n_pass = sum(1 for f in seed_pass_flags if f)
    pass_frac = _frac(seed_pass_flags)
    overall_criteria_pass = bool(pass_frac >= MIN_FRACTION)

    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not occupancy_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "external_task_mode_not_occupied"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("mech266_mode_stickiness_dose_response_confirmed"
                           if overall_criteria_pass else "residual_dose_response_open")
        route_reason = "c1_c2_c3_majority_met" if overall_criteria_pass else "criteria_unmet_genuine_weakens"
        direction_map = {
            "MECH-266": "supports" if overall_criteria_pass else "weakens",
            "SD-032a": "supports" if overall_criteria_pass else "weakens",
        }
        overall_direction = "supports" if overall_criteria_pass else "weakens"

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) occupancy_non_vacuity={occupancy_non_vacuity_met}"
          f" (frac={occ_frac:.3f}) criteria_pass={overall_criteria_pass}"
          f" ({n_pass}/{len(guard_passing)}) -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    def _all_guard(crit_key: str) -> bool:
        return bool(guard_passing) and all(
            r.get("criteria", {}).get(crit_key) for r in guard_passing
        )

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "occupancy_non_vacuity_met": occupancy_non_vacuity_met,
        "occupancy_non_vacuity_fraction": occ_frac,
        "criteria_pass_fraction": pass_frac,
        "n_seeds_pass": n_pass,
        "overall_pass": bool(contact_non_vacuity_met and occupancy_non_vacuity_met
                             and overall_criteria_pass),
        "ratios": RATIOS,
        "per_seed_guard_pass": guard_flags,
        "per_seed_criteria_pass": [bool(r.get("pass", False)) for r in per_seed],
        "route_reason": route_reason,
    }

    crit_non_degenerate = bool(contact_non_vacuity_met and occupancy_non_vacuity_met)

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
                    "description": "603n G2+G3 contact guard on >= 2/3 seeds.",
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                },
                {
                    "name": "external_task_occupancy_reachable",
                    "kind": "readiness",
                    "description": "the contested external_task mode is genuinely occupied across "
                                   "the sweep -- per-seed min over ratio arms of "
                                   "fraction_in_external_task > OCCUPANCY_FLOOR -- on >= 2/3 "
                                   "guard-passing seeds. The SAME occupancy statistic the "
                                   "dose-response operates over (V3-EXQ-643 same-statistic rule), "
                                   "re-stating the 467c n_switches>=1 gate. A zero-occupancy read is "
                                   "mode-governance-not-engaged (possibly the unresolved "
                                   "nav-competence dependency), NOT a MECH-266 falsification.",
                    "control": "fraction of guard-passing seeds with per-seed min-ratio-arm "
                               "fraction_in_external_task > OCCUPANCY_FLOOR.",
                    "measured": occ_frac,
                    "threshold": MIN_FRACTION,
                    "met": occupancy_non_vacuity_met,
                },
            ],
            "criteria": [
                {"name": "C1_dwell_monotone_non_increasing_in_r", "load_bearing": True,
                 "passed": _all_guard("C1")},
                {"name": "C2_low_r_dwell_ge_ratio_times_high_r", "load_bearing": True,
                 "passed": _all_guard("C2")},
            ],
            "criteria_non_degenerate": {
                "C1": crit_non_degenerate,
                "C2": crit_non_degenerate,
            },
            "contact_guard": {
                "definition": "per-seed P2 contact_rate > 0 AND z_goal_norm_at_contact_peak > 0.4; "
                              "< 2/3 seeds -> substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "occupancy_non_vacuity_gate": {
                "definition": "per-seed min over ratio arms of fraction_in_external_task > "
                              "OCCUPANCY_FLOOR on >= 2/3 guard-passing seeds (the SAME occupancy "
                              "statistic the dose-response operates over). Below floor -> "
                              "substrate_not_ready_requeue (non_contributory), NEVER a false "
                              "weakens. Re-states the 467c n_switches>=1 gate.",
                "min_fraction": MIN_FRACTION,
                "ratios": RATIOS,
                "c2_min_dwell_ratio": C2_MIN_DWELL_RATIO,
                "occupancy_floor": OCCUPANCY_FLOOR,
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
        "substrate": "scaffolded_sd054_onboarding (full curriculum; 603n config; ready=true "
                     "2026-06-11) + SalienceCoordinator (SD-032a) + MECH-266 uniform exit-rail "
                     "ratio sweep + mode-governance-engagement external_task drive "
                     "(use_external_task_drive=True, landed 2026-06-13) + GAP-3 dual_cue env "
                     "primitive. use_closure_operator OFF (closure would inject a confounding "
                     "mode-switch signal).",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "method_note": "467c's MECH-266 mode-stickiness dose-response (a lower uniform exit rail "
                       "-> longer dwell + fewer switches; mean dwell monotone non-increasing in r) "
                       "re-run WITH the mode-governance-engagement substrate enabled "
                       "(use_external_task_drive=True). 467c FAILed non_contributory because "
                       "fraction_in_external_task=0.0 / mean_dwell was an episode-length artifact "
                       "(no external_task salience SOURCE), so the dose-response had no contested "
                       "mode to operate over. The drive is the ONLY agent-config change vs 467c.",
        "readiness_note": "TWO non-vacuity gates self-route a substrate-not-engaged read to "
                          "non_contributory (never a false weakens): the 603n contact guard + the "
                          "external_task occupancy gate (min over ratio arms of "
                          "fraction_in_external_task > OCCUPANCY_FLOOR -- the SAME occupancy "
                          "statistic the dose-response operates over, replacing the 467c "
                          "n_switches>=1 gate).",
        "arm_note": "One eval per hysteresis ratio in RATIOS, each a clone of the same trained "
                    "weights. The committed_mode_curriculum O-2 forced-rv contrast arm is dropped.",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "ratios": RATIOS,
            "c2_min_dwell_ratio": C2_MIN_DWELL_RATIO,
            "occupancy_floor": OCCUPANCY_FLOOR,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "mode_eval_episodes_per_ratio": MODE_EVAL_EPISODES,
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "config_basis": "V3-EXQ-603n",
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
