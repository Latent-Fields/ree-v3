"""
V3-EXQ-461c (EXP-0157 behavioural): MECH-090 / SD-033a / SD-034 delayed-reward
persistence, measured on a foraging-competent agent built through the FULL
scaffolded_sd054_onboarding curriculum at the 603n config. Successor to
V3-EXQ-461b (NOT a supersede).

ROUTING: the commitment_closure:GAP-4 *b cohort all RAN 2026-06-04 and were
reclassified non_contributory + substrate_ceiling -- in the committed_mode_curriculum
loop the agent committed but never tolerance-completed a waypoint / never held a
delayed window long enough (n_windows=0). 461b was built on the committed_mode_curriculum,
which trains commitment but NOT foraging competence, AND never set subgoal_mode=True.
This rewires the TRAINING harness to the scaffolded_sd054_onboarding curriculum at the
603n config (the 514n pattern; ready=true 2026-06-11) and sets subgoal_mode=True so
delayed committed Hold windows + waypoint completions occur and closure can couple to
the delayed resolution.

WHAT THIS MEASURES (unchanged from 461b): the Hold-axis falsifier. A committed Hold
window (MECH-090 bistable latch) persists across a delay; the SD-033a rule_state norm
is retained across the window (persistence ratio release/entry >= floor); and an SD-034
closure fires within CLOSURE_WINDOW_TICKS of the delayed resolution.

ARMS (one curriculum build per seed, two frozen-policy evals):
  ARM_HOLD_ON      -- full bistable + closure agent. Expect delay windows with
                      retained rule_state and closure-coupled resolutions.
  ARM_NO_HOLD_OFF  -- clone of the SAME trained weights with beta_gate_bistable=False
                      (legacy per-tick beta, no Hold latch). Under-binding contrast:
                      short / no committed windows.
  (The committed_mode_curriculum O-2 forced-rv contrast arm is dropped.)

CONTACT GUARD (603n G2 + G3): per-seed contact_rate > 0 AND z_goal_norm_at_contact_peak
  > 0.4; < 2/3 seeds passing -> substrate_not_ready_requeue (non_contributory).

COMMITMENT NON-VACUITY GATE: before scoring, assert the ON arm actually produced
  committed windows (n_windows > 0) on >= 2/3 guard-passing seeds -- the EXACT
  n_windows=0 gap the *b cohort scored. Below floor -> substrate_not_ready_requeue
  (non_contributory), NEVER a false weakens.

PRE-REGISTERED ACCEPTANCE (constants; PASS = majority 2/3 guard seeds):
  C1  ARM_HOLD_ON  n_delay_windows >= 1 AND mean_persistence_ratio >= 0.5
                   (delay windows form AND SD-033a rule_state persists)
  C2  ARM_HOLD_ON  n_closure_coupled_resolutions >= 1 (SD-034 couples to resolution)
  C3  ARM_HOLD_ON  mean_window_len > ARM_NO_HOLD_OFF mean_window_len (Hold latch
                   extends the committed window vs the no-hold contrast)
  Per-seed PASS = C1 AND C2 AND C3.

PER-CLAIM DIRECTION:
  contact / commitment non-vacuity NOT met -> non_contributory (substrate_not_ready_requeue).
  both met:  MECH-090 = supports if overall PASS else weakens (the Hold latch);
             SD-033a = supports if (C1 all guard seeds) else weakens (rule persistence);
             SD-034 = supports if (C2 all guard seeds) else weakens (closure coupling).

claim_ids: MECH-090, SD-033a, SD-034.
experiment_purpose: evidence
predecessor: V3-EXQ-461b (NOT a supersede).

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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_461c_mech090_sd033a_delayed_reward_persistence_behavioural"
QUEUE_ID = "V3-EXQ-461c"
CLAIM_IDS: List[str] = ["MECH-090", "SD-033a", "SD-034"]
EXPERIMENT_PURPOSE = "evidence"
PREDECESSOR = "V3-EXQ-461b (successor, NOT supersede)"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_DELAYED_REWARD_PERSISTENCE"

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
PERSIST_EVAL_EPISODES = 15
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

# --- Delayed-persistence thresholds (mirror 461b; constants) ---
MIN_DELAY_TICKS = 3
CLOSURE_WINDOW_TICKS = 2
C1_MIN_WINDOWS = 1
C1_PERSIST_FLOOR = 0.5
C2_MIN_COUPLED = 1

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
        use_closure_operator=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_hold_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint
    tolerance-band completion so delayed committed Hold windows + completions occur
    and closure can couple to the delayed resolution. subgoal_mode=True is
    LOAD-BEARING (461b never set it)."""
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


def _clone_no_hold(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone the SAME trained weights into a no-Hold agent (beta_gate_bistable=False,
    legacy per-tick beta). The under-binding contrast for the Hold latch."""
    cfg_off = copy.deepcopy(trained_agent.config)
    cfg_off.heartbeat.beta_gate_bistable = False
    agent_off = REEAgent(cfg_off).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_off.load_state_dict(state)
    except RuntimeError:
        agent_off.load_state_dict(state, strict=False)
    agent_off.e3._running_variance = float(trained_agent.e3._running_variance)
    return agent_off


def _eval_delayed_persistence(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for delayed-reward persistence (ported from
    461b, using _sense_with_optional_harm). Detects committed Hold windows via beta
    transitions; measures window length, SD-033a rule_state persistence
    (release/entry), and SD-034 closure coupling to the delayed resolution."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    window_lengths: List[int] = []
    delay_windows: List[int] = []
    persistence_ratios: List[float] = []
    n_closure_coupled = 0
    n_resolutions = 0
    total_beta_elevated = 0
    global_tick = 0
    closure_fire_ticks: List[int] = []

    def _rule_norm() -> float:
        if not has_lpfc or agent.lateral_pfc.rule_state is None:
            return 0.0
        return float(agent.lateral_pfc.rule_state.norm().item())

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_beta = bool(agent.beta_gate.is_elevated)
            window_len = 0
            rule_norm_entry = 0.0

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, feed_harm
                )

                closures_before = (
                    int(agent.closure_operator._n_closures) if has_closure else 0
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                if has_closure and int(agent.closure_operator._n_closures) - closures_before > 0:
                    closure_fire_ticks.append(global_tick)

                cur_beta = bool(agent.beta_gate.is_elevated)
                if cur_beta:
                    total_beta_elevated += 1

                if cur_beta and not prev_beta:
                    window_len = 1
                    rule_norm_entry = _rule_norm()
                elif cur_beta and prev_beta:
                    window_len += 1
                elif (not cur_beta) and prev_beta:
                    n_resolutions += 1
                    rule_norm_release = _rule_norm()
                    window_lengths.append(window_len)
                    if window_len >= MIN_DELAY_TICKS:
                        delay_windows.append(window_len)
                        denom = max(rule_norm_entry, 1e-6)
                        persistence_ratios.append(rule_norm_release / denom)
                    if any(abs(global_tick - c) <= CLOSURE_WINDOW_TICKS for c in closure_fire_ticks):
                        n_closure_coupled += 1
                    window_len = 0

                prev_beta = cur_beta
                global_tick += 1

                _, _harm, done, _info, obs_dict = env.step(action_idx)
                if done:
                    break

    mean_window_len = (
        float(sum(window_lengths)) / len(window_lengths) if window_lengths else 0.0
    )
    mean_persistence = (
        float(sum(persistence_ratios)) / len(persistence_ratios) if persistence_ratios else 0.0
    )
    return {
        "n_windows": len(window_lengths),
        "n_delay_windows": len(delay_windows),
        "mean_window_len": round(mean_window_len, 3),
        "mean_persistence_ratio": round(mean_persistence, 4),
        "n_resolutions": n_resolutions,
        "n_closure_coupled_resolutions": n_closure_coupled,
        "total_beta_elevated": total_beta_elevated,
        "closure_present": has_closure,
        "n_eval_episodes": n_eps,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    empty = {
        "n_windows": 0, "n_delay_windows": 0, "mean_window_len": 0.0,
        "mean_persistence_ratio": 0.0, "n_resolutions": 0,
        "n_closure_coupled_resolutions": 0, "total_beta_elevated": 0,
        "closure_present": False, "n_eval_episodes": 0,
    }
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "ARM_HOLD_ON": dict(empty), "ARM_NO_HOLD_OFF": dict(empty),
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
    eval_eps = 2 if dry_run else PERSIST_EVAL_EPISODES

    probe_env = _build_hold_env(scaffold_cfg)
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

    hold_env = _build_hold_env(scaffold_cfg)
    hold_env.reset()

    print(f"Seed {seed} Condition ARM_HOLD_ON", flush=True)
    arm_on = _eval_delayed_persistence(
        agent, hold_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    print(f"Seed {seed} Condition ARM_NO_HOLD_OFF", flush=True)
    agent_off = _clone_no_hold(agent, device)
    arm_off = _eval_delayed_persistence(
        agent_off, hold_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    c1 = bool(
        arm_on["n_delay_windows"] >= C1_MIN_WINDOWS
        and arm_on["mean_persistence_ratio"] >= C1_PERSIST_FLOOR
    )
    c2 = arm_on["n_closure_coupled_resolutions"] >= C2_MIN_COUPLED
    c3 = arm_on["mean_window_len"] > arm_off["mean_window_len"]
    commitment_non_vacuity = bool(arm_on["n_windows"] > 0)
    seed_pass = bool(c1 and c2 and c3)

    print(f"  [train] persist_eval seed={seed} ep {done}/{total_eps}"
          f" c1={c1} c2={c2} c3={c3}"
          f" n_windows={arm_on['n_windows']} n_delay={arm_on['n_delay_windows']}"
          f" persist={arm_on['mean_persistence_ratio']} coupled={arm_on['n_closure_coupled_resolutions']}", flush=True)
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
        "ARM_HOLD_ON": arm_on,
        "ARM_NO_HOLD_OFF": arm_off,
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
            + P1_BUDGET + P2_BUDGET + 2 * PERSIST_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    cv_flags = [bool(r.get("commitment_non_vacuity", False)) for r in guard_passing]
    cv_frac = _frac(cv_flags)
    commitment_non_vacuity_met = bool(cv_frac >= MIN_FRACTION)

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
        route_reason = "commitment_windows_not_engaged"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("delayed_reward_persistence_confirmed"
                           if overall_criteria_pass else "residual_persistence_open")
        route_reason = "c1_c2_c3_majority_met" if overall_criteria_pass else "criteria_unmet_genuine_weakens"
        direction_map = {
            "MECH-090": "supports" if overall_criteria_pass else "weakens",
            "SD-033a": "supports" if c1_all else "weakens",
            "SD-034": "supports" if c2_all else "weakens",
        }
        overall_direction = "supports" if overall_criteria_pass else "weakens"

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) commit_non_vacuity={commitment_non_vacuity_met}"
          f" (frac={cv_frac:.3f}) criteria_pass={overall_criteria_pass}"
          f" ({n_pass}/{len(guard_passing)}) -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "commitment_non_vacuity_met": commitment_non_vacuity_met,
        "commitment_non_vacuity_fraction": cv_frac,
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
                    "description": "603n G2+G3 contact guard on >= 2/3 seeds.",
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                },
                {
                    "name": "committed_windows_engaged",
                    "description": "ON arm produced committed Hold windows (n_windows > 0) on "
                                   ">= 2/3 guard-passing seeds -- the exact n_windows=0 gap the "
                                   "*b cohort scored.",
                    "control": "fraction of guard-passing seeds with ON-arm committed windows.",
                    "measured": cv_frac,
                    "threshold": MIN_FRACTION,
                    "met": commitment_non_vacuity_met,
                },
            ],
            "criteria": [
                {"name": "C1_delay_windows_and_persistence", "load_bearing": True, "passed": c1_all},
                {"name": "C2_closure_coupled_resolutions", "load_bearing": True, "passed": c2_all},
                {"name": "C3_hold_extends_window_vs_nohold", "load_bearing": True,
                 "passed": _all_guard("C3")},
            ],
            "criteria_non_degenerate": {
                "C1": crit_non_degenerate,
                "C2": crit_non_degenerate,
                "C3": crit_non_degenerate,
            },
            "contact_guard": {
                "definition": "per-seed P2 contact_rate > 0 AND z_goal_norm_at_contact_peak > 0.4; "
                              "< 2/3 seeds -> substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "commitment_non_vacuity_gate": {
                "definition": "ON-arm n_windows > 0 on >= 2/3 guard-passing seeds. Below floor -> "
                              "substrate_not_ready_requeue (non_contributory), NEVER a false weakens.",
                "min_fraction": MIN_FRACTION,
                "min_delay_ticks": MIN_DELAY_TICKS,
                "closure_window_ticks": CLOSURE_WINDOW_TICKS,
                "c1_persist_floor": C1_PERSIST_FLOOR,
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
                     "2026-06-11) + commitment control-plane (bistable BetaGate MECH-090 + SD-033a "
                     "LateralPFC + SD-034 closure + SD-032 dACC/salience) + subgoal_mode waypoint "
                     "tolerance-band completion env primitive.",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "method_note": "461b's delayed-reward persistence Hold-axis falsifier (committed Hold "
                       "window persists across a delay with retained SD-033a rule_state, and SD-034 "
                       "closure couples to the delayed resolution) re-run on a foraging-competent "
                       "agent BUILT THROUGH the scaffolded_sd054_onboarding curriculum. 461b's "
                       "committed_mode_curriculum trained commitment but NOT foraging competence and "
                       "never set subgoal_mode=True, so committed windows never formed (n_windows=0).",
        "readiness_note": "TWO non-vacuity gates self-route a substrate-not-engaged read to "
                          "non_contributory (never a false weakens): the 603n contact guard + the "
                          "commitment non-vacuity gate (the ON arm actually produced committed "
                          "Hold windows).",
        "arm_note": "ARM_HOLD_ON (bistable) vs ARM_NO_HOLD_OFF (same trained weights, "
                    "beta_gate_bistable=False, legacy per-tick beta). The committed_mode_curriculum "
                    "O-2 forced-rv contrast arm is dropped.",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "min_delay_ticks": MIN_DELAY_TICKS,
            "closure_window_ticks": CLOSURE_WINDOW_TICKS,
            "c1_min_windows": C1_MIN_WINDOWS,
            "c1_persist_floor": C1_PERSIST_FLOOR,
            "c2_min_coupled": C2_MIN_COUPLED,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "persist_eval_episodes_per_arm": PERSIST_EVAL_EPISODES,
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
        )
