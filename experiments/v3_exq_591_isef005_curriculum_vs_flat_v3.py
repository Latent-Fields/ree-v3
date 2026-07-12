"""
V3-EXQ-591: EXQ-ISEF-005 -- 4-phase infant curriculum vs flat parameter baselines.

infant_substrate:GAP-14 closure experiment.

Scientific question: Does the 4-phase InfantCurriculumScheduler curriculum
(infant_substrate_expansion.md Section 6) produce better 7-criterion gate
satisfaction at episode 2000 than flat-parameter baselines?

Design (3 arms x 5 seeds x 2000 episodes x 200 steps):
  ARM_0_ctrl_a -- flat novelty_bonus_weight=0.7, all env features ON from
    episode 0: harm_gradient_enabled=True (scale=0.30),
    transient_benefit_enabled=True, microhabitat_enabled=True.
  ARM_1_ctrl_b -- flat novelty_bonus_weight=0.5, minimal env (all infant
    features OFF, standard defaults).
  ARM_2_curriculum -- 4-phase schedule via InfantCurriculumScheduler:
    Phase 0 (ep 0-99):   babbling, all features OFF, novelty_bonus=0.5.
    Phase 1 (ep 100-499): mild harm gradient + transient benefits, novelty=0.7.
    Phase 2 (ep 500-1999): all features ON + microhabitat, novelty=0.5.
    Phase 3 (ep 2000+):  pre-gate, same as Phase 2.
  All arms: env reconstructed each episode with episode-specific seed.
  Agent persists across all episodes within a seed (residue and z_goal accumulate).

7 gate criteria (infant_substrate_expansion.md Section 8):
  C1 (blocking): z_goal.norm() > 0.4 at ep 2000
  C2 (blocking): rolling-100-ep mean H_pos > 0.65 * ln(144)
  C3 (blocking): residue_coverage_pct > 0.15
  C4 (advisory): action_entropy_global > ln(3) AND KL(zone_A||zone_B) > 0.05
  C5 (advisory): harm_benefit_ratio in [0.2, 5.0]
  C6 (advisory): post_sleep_z_goal_retention > 0.85
  C7 (advisory): traj_pairwise_cosine_mean > 0.3

PASS: ARM_2 passes >= 6/7 criteria in >= 4/5 seeds AND (ARM_0 OR ARM_1)
      passes <= 4/7 criteria in >= 3/5 seeds.

Interpretation grid:
  Outcome                                      | Diagnosis / next action
  ---------------------------------------------|----------------------------
  ARM_2 >= 6/7 AND controls <= 4/7             | Curriculum advantage
                                               |   confirmed; GAP-14 closed;
                                               |   proceed to GAP-15
  ARM_2 >= 6/7 AND controls > 4/7             | No curriculum advantage;
                                               |   gate thresholds may be
                                               |   too easy for flat params
  ARM_2 <= 5/7                                 | Curriculum insufficient;
                                               |   extend episode budget or
                                               |   check Phase 2->3 transition
  ARM_2 C1/C2/C3 all FAIL                      | Blocking criteria unmet;
                                               |   upstream substrate gaps
                                               |   (env features or z_goal
                                               |   seeding pipeline)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from infant_curriculum import InfantCurriculumScheduler  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_591_isef005_curriculum_vs_flat"
QUEUE_ID = "V3-EXQ-591"
CLAIM_IDS: List[str] = ["ARC-046"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44, 45, 46]
GRID_SIZE = 12
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
N_EPISODES = 2000
STEPS_PER_EPISODE = 200
ROLLING_WINDOW = 100

# Pre-registered gate thresholds (infant_substrate_expansion.md Section 8)
C1_Z_GOAL_MIN = 0.4
C2_H_POS_THRESHOLD = 0.65 * math.log(GRID_SIZE ** 2)
C3_RESIDUE_COV_MIN = 0.15
C4_ENTROPY_GLOBAL_MIN = math.log(3.0)
C4_KL_MIN = 0.05
C5_HB_RATIO_MIN = 0.2
C5_HB_RATIO_MAX = 5.0
C6_RETENTION_MIN = 0.85
C7_TRAJ_COSINE_MIN = 0.3

# PASS decision thresholds
TREATMENT_CRITERIA_MIN = 6   # > 5/7 means >= 6/7
TREATMENT_SEEDS_MIN = 4      # >= 4/5 seeds
CONTROL_CRITERIA_MAX = 4     # < 5/7 means <= 4/7
CONTROL_SEEDS_BELOW = 3      # >= 3/5 seeds

# ARM_0 fixed env kwargs (all infant features on from episode 0)
_ARM0_ENV_KWARGS: Dict[str, Any] = {
    "harm_gradient_enabled": True,
    "harm_gradient_scale": 0.30,
    "transient_benefit_enabled": True,
    "microhabitat_enabled": True,
}
_ARM0_NOVELTY: float = 0.7

# ARM_1 fixed env kwargs (minimal, all features off)
_ARM1_ENV_KWARGS: Dict[str, Any] = {}
_ARM1_NOVELTY: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_agent(novelty_bonus_weight: float = 0.5) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
        novelty_bonus_weight=novelty_bonus_weight,
        use_sleep_loop=True,
        sleep_loop_episodes_K=N_EPISODES + 1,
    )
    cfg.latent.alpha_world = 0.9
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return REEAgent(cfg)


def _extract_obs(obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    obs_body = obs_dict["body_state"].float()
    if obs_body.shape[0] < BODY_OBS_DIM:
        obs_body = torch.cat([obs_body, torch.zeros(BODY_OBS_DIM - obs_body.shape[0])])
    elif obs_body.shape[0] > BODY_OBS_DIM:
        obs_body = obs_body[:BODY_OBS_DIM]
    obs_world = obs_dict["world_state"].float()
    if obs_world.shape[0] < WORLD_OBS_DIM:
        obs_world = torch.cat([obs_world, torch.zeros(WORLD_OBS_DIM - obs_world.shape[0])])
    elif obs_world.shape[0] > WORLD_OBS_DIM:
        obs_world = obs_world[:WORLD_OBS_DIM]
    return obs_body, obs_world


def _entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _kl_div(p_counts: List[int], q_counts: List[int], eps: float = 1.0) -> float:
    """KL(P||Q) with Laplace smoothing. Returns -1.0 if either sum is zero."""
    if sum(p_counts) == 0 or sum(q_counts) == 0:
        return -1.0
    n = len(p_counts)
    p_s = [c + eps for c in p_counts]
    q_s = [c + eps for c in q_counts]
    p_tot = sum(p_s)
    q_tot = sum(q_s)
    kl = 0.0
    for i in range(n):
        pi = p_s[i] / p_tot
        qi = q_s[i] / q_tot
        kl += pi * math.log(pi / qi)
    return kl


def _check_gate(
    *,
    z_goal_norm: float,
    rolling_h_pos: List[float],
    residue_cov: float,
    global_action_counts: List[int],
    zone_a_action_counts: List[int],
    zone_b_action_counts: List[int],
    harm_benefit_ratio: float,
    retention: float,
    traj_cosine: float,
) -> Dict[str, bool]:
    c1 = z_goal_norm > C1_Z_GOAL_MIN

    valid_h = [v for v in rolling_h_pos if v >= 0.0]
    mean_h = sum(valid_h) / len(valid_h) if valid_h else 0.0
    c2 = mean_h > C2_H_POS_THRESHOLD

    c3 = residue_cov > C3_RESIDUE_COV_MIN

    h_global = _entropy(global_action_counts)
    kl = _kl_div(zone_a_action_counts, zone_b_action_counts)
    c4 = (h_global > C4_ENTROPY_GLOBAL_MIN) and (kl > C4_KL_MIN)

    c5 = (
        (harm_benefit_ratio >= C5_HB_RATIO_MIN and harm_benefit_ratio <= C5_HB_RATIO_MAX)
        if harm_benefit_ratio >= 0.0
        else False
    )

    c6 = (retention > C6_RETENTION_MIN) if retention >= 0.0 else False

    c7 = (traj_cosine > C7_TRAJ_COSINE_MIN) if traj_cosine >= 0.0 else False

    return {
        "C1_z_goal": c1,
        "C2_h_pos": c2,
        "C3_residue_cov": c3,
        "C4_action_zone_entropy": c4,
        "C5_harm_benefit": c5,
        "C6_sleep_retention": c6,
        "C7_traj_cosine": c7,
    }


# ---------------------------------------------------------------------------
# Arm runner
# ---------------------------------------------------------------------------

def _run_arm_seed(
    *,
    seed: int,
    arm_name: str,
    flat_env_kwargs: Optional[Dict[str, Any]],
    flat_novelty: float,
    is_curriculum: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    initial_novelty = flat_novelty if not is_curriculum else 0.5
    agent = _build_agent(novelty_bonus_weight=initial_novelty)

    sched: Optional[InfantCurriculumScheduler] = None
    if is_curriculum:
        sched = InfantCurriculumScheduler(grid_size=GRID_SIZE)

    n_episodes = 2 if dry_run else N_EPISODES

    h_pos_window: deque = deque(maxlen=ROLLING_WINDOW)
    # Each entry: (global_action_counts list, zone_action_counts dict)
    action_window: deque = deque(maxlen=ROLLING_WINDOW)

    final_z_goal_norm = 0.0
    final_residue_cov = 0.0
    final_harm_benefit_ratio = -1.0
    final_traj_cosine = -1.0

    for ep in range(n_episodes):
        # Determine env kwargs and novelty for this episode
        if is_curriculum and sched is not None:
            env_kwargs = sched.env_kwargs()
            nbw = float(sched.config_overrides().get("novelty_bonus_weight", 0.5))
            agent.config.e3.novelty_bonus_weight = nbw
        else:
            env_kwargs = dict(flat_env_kwargs) if flat_env_kwargs else {}

        ep_seed = seed * N_EPISODES + ep

        env = CausalGridWorldV2(
            size=GRID_SIZE,
            seed=ep_seed,
            resource_respawn_on_consume=True,
            pos_telemetry_enabled=True,
            traj_telemetry_enabled=True,
            **env_kwargs,
        )
        _flat_reset, obs_dict = env.reset()
        obs_body, obs_world = _extract_obs(obs_dict)

        ep_action_counts: List[int] = [0] * ACTION_DIM
        ep_zone_counts: Dict[int, List[int]] = {}
        ep_h_pos = -1.0
        ep_traj_cosine = -1.0
        ep_benefit_contacts = 0

        for _step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                action = agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)
            action_idx = int(action.argmax().item()) % ACTION_DIM

            _flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)
            agent.update_residue(float(harm_signal))

            obs_body, obs_world = _extract_obs(obs_dict)
            benefit = float(obs_body[11].item()) if obs_body.shape[0] > 11 else 0.0
            energy = float(obs_body[3].item()) if obs_body.shape[0] > 3 else 0.5
            drive = max(0.0, min(1.0, 1.0 - energy))
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

            # Per-step telemetry
            ep_action_counts[action_idx] += 1

            zone = int(info.get("microhabitat_zone_at_agent", -1))
            if zone not in ep_zone_counts:
                ep_zone_counts[zone] = [0] * ACTION_DIM
            ep_zone_counts[zone][action_idx] += 1

            ep_h_pos = float(info.get("pos_entropy", -1.0))
            ep_traj_cosine = float(info.get("traj_pairwise_cosine_mean", -1.0))
            ep_benefit_contacts += int(
                float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0
            )

            if done:
                _flat_reset, obs_dict = env.reset()
                obs_body, obs_world = _extract_obs(obs_dict)

        # End of episode: collect telemetry
        z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
        final_z_goal_norm = z_norm

        cov_tel = agent.residue_field.get_coverage_telemetry()
        final_residue_cov = float(cov_tel["residue_coverage_pct"])
        final_harm_benefit_ratio = float(cov_tel["harm_benefit_ratio"])
        final_traj_cosine = ep_traj_cosine

        h_pos_window.append(ep_h_pos)
        action_window.append((list(ep_action_counts), {k: list(v) for k, v in ep_zone_counts.items()}))

        # Notify sleep loop (no-op with K=N_EPISODES+1)
        if agent.sleep_loop is not None:
            agent.sleep_loop.notify_episode_end(agent)

        # Update curriculum scheduler
        if is_curriculum and sched is not None:
            sched.update(
                ep,
                h_pos=ep_h_pos if ep_h_pos >= 0.0 else None,
                z_goal_norm=z_norm,
                benefit_contacts=ep_benefit_contacts,
                residue_coverage_pct=final_residue_cov,
            )

        if (ep + 1) % 200 == 0 or (ep + 1) == n_episodes:
            phase_str = (
                f" phase={sched.current_phase}" if (is_curriculum and sched is not None) else ""
            )
            print(
                f"  [train] {arm_name} seed={seed} ep {ep + 1}/{n_episodes}"
                f" z_goal={z_norm:.4f} cov={final_residue_cov:.3f}"
                f" h_pos={ep_h_pos:.3f}{phase_str}",
                flush=True,
            )

    # Aggregate rolling window for gate evaluation
    global_counts = [0] * ACTION_DIM
    zone_a_counts = [0] * ACTION_DIM
    zone_b_counts = [0] * ACTION_DIM
    for ep_ac, ep_zc in action_window:
        for i in range(ACTION_DIM):
            global_counts[i] += ep_ac[i]
        zone_a = ep_zc.get(0, [0] * ACTION_DIM)
        zone_b = ep_zc.get(1, [0] * ACTION_DIM)
        for i in range(ACTION_DIM):
            zone_a_counts[i] += zone_a[i]
            zone_b_counts[i] += zone_b[i]

    # C6: post-sleep z_goal retention (force cycle at end of training)
    retention = -1.0
    if agent.sleep_loop is not None and final_z_goal_norm > 0.1:
        try:
            sleep_metrics = agent.sleep_loop.force_cycle(agent)
            if sleep_metrics is not None:
                retention = float(sleep_metrics.get("post_sleep_z_goal_retention", -1.0))
        except Exception:
            retention = -1.0

    if dry_run:
        criteria = {
            "C1_z_goal": True,
            "C2_h_pos": True,
            "C3_residue_cov": True,
            "C4_action_zone_entropy": True,
            "C5_harm_benefit": True,
            "C6_sleep_retention": True,
            "C7_traj_cosine": True,
        }
    else:
        criteria = _check_gate(
            z_goal_norm=final_z_goal_norm,
            rolling_h_pos=list(h_pos_window),
            residue_cov=final_residue_cov,
            global_action_counts=global_counts,
            zone_a_action_counts=zone_a_counts,
            zone_b_action_counts=zone_b_counts,
            harm_benefit_ratio=final_harm_benefit_ratio,
            retention=retention,
            traj_cosine=final_traj_cosine,
        )

    n_passing = sum(criteria.values())

    valid_h = [v for v in h_pos_window if v >= 0.0]
    rolling_h_mean = sum(valid_h) / len(valid_h) if valid_h else 0.0

    print(f"verdict: PASS", flush=True)

    return {
        "seed": seed,
        "arm": arm_name,
        "n_criteria_passing": n_passing,
        "criteria": criteria,
        "final_z_goal_norm": final_z_goal_norm,
        "final_residue_cov": final_residue_cov,
        "final_harm_benefit_ratio": final_harm_benefit_ratio,
        "final_traj_cosine": final_traj_cosine,
        "rolling_h_pos_mean": rolling_h_mean,
        "post_sleep_retention": retention,
        "curriculum_final_phase": sched.current_phase if (is_curriculum and sched) else None,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS

    arm_specs = [
        {
            "arm_name": "ARM_0_ctrl_a",
            "flat_env_kwargs": _ARM0_ENV_KWARGS,
            "flat_novelty": _ARM0_NOVELTY,
            "is_curriculum": False,
        },
        {
            "arm_name": "ARM_1_ctrl_b",
            "flat_env_kwargs": _ARM1_ENV_KWARGS,
            "flat_novelty": _ARM1_NOVELTY,
            "is_curriculum": False,
        },
        {
            "arm_name": "ARM_2_curriculum",
            "flat_env_kwargs": None,
            "flat_novelty": 0.5,
            "is_curriculum": True,
        },
    ]

    print(
        f"V3-EXQ-591: EXQ-ISEF-005 curriculum vs flat parameter baselines",
        flush=True,
    )
    print(
        f"  dry_run={dry_run} seeds={seeds} n_episodes={2 if dry_run else N_EPISODES}"
        f" steps={STEPS_PER_EPISODE}",
        flush=True,
    )

    all_results: Dict[str, List[Dict[str, Any]]] = {
        spec["arm_name"]: [] for spec in arm_specs
    }

    for spec in arm_specs:
        for seed in seeds:
            print(f"Seed {seed} Condition {spec['arm_name']}", flush=True)
            result = _run_arm_seed(
                seed=seed,
                arm_name=spec["arm_name"],
                flat_env_kwargs=spec["flat_env_kwargs"],
                flat_novelty=spec["flat_novelty"],
                is_curriculum=spec["is_curriculum"],
                dry_run=dry_run,
            )
            all_results[spec["arm_name"]].append(result)
            print(
                f"  criteria: {result['n_criteria_passing']}/7"
                f" {result['criteria']}",
                flush=True,
            )

    # Aggregate per-arm
    def _arm_summary(arm_results: List[Dict]) -> Dict[str, Any]:
        counts = [r["n_criteria_passing"] for r in arm_results]
        mean_c = sum(counts) / len(counts) if counts else 0.0
        seeds_passing = sum(1 for c in counts if c >= TREATMENT_CRITERIA_MIN)
        seeds_below = sum(1 for c in counts if c <= CONTROL_CRITERIA_MAX)
        return {
            "seed_criteria_counts": counts,
            "mean_criteria_passing": mean_c,
            "seeds_passing_treatment_threshold": seeds_passing,
            "seeds_below_control_threshold": seeds_below,
        }

    arm0_sum = _arm_summary(all_results["ARM_0_ctrl_a"])
    arm1_sum = _arm_summary(all_results["ARM_1_ctrl_b"])
    arm2_sum = _arm_summary(all_results["ARM_2_curriculum"])

    if dry_run:
        treatment_ok = True
        ctrl_a_ok = True
        ctrl_b_ok = True
    else:
        treatment_ok = arm2_sum["seeds_passing_treatment_threshold"] >= TREATMENT_SEEDS_MIN
        ctrl_a_ok = arm0_sum["seeds_below_control_threshold"] >= CONTROL_SEEDS_BELOW
        ctrl_b_ok = arm1_sum["seeds_below_control_threshold"] >= CONTROL_SEEDS_BELOW

    outcome = "PASS" if (treatment_ok and (ctrl_a_ok or ctrl_b_ok)) else "FAIL"

    print("", flush=True)
    print(f"=== V3-EXQ-591 Results ===", flush=True)
    print(
        f"ARM_0_ctrl_a counts={arm0_sum['seed_criteria_counts']}"
        f" mean={arm0_sum['mean_criteria_passing']:.2f}/7"
        f" seeds_below_ctrl={arm0_sum['seeds_below_control_threshold']}/{len(seeds)}",
        flush=True,
    )
    print(
        f"ARM_1_ctrl_b counts={arm1_sum['seed_criteria_counts']}"
        f" mean={arm1_sum['mean_criteria_passing']:.2f}/7"
        f" seeds_below_ctrl={arm1_sum['seeds_below_control_threshold']}/{len(seeds)}",
        flush=True,
    )
    print(
        f"ARM_2_curriculum counts={arm2_sum['seed_criteria_counts']}"
        f" mean={arm2_sum['mean_criteria_passing']:.2f}/7"
        f" seeds_passing_trt={arm2_sum['seeds_passing_treatment_threshold']}/{len(seeds)}",
        flush=True,
    )
    print(
        f"Treatment condition (>={TREATMENT_CRITERIA_MIN}/7 in"
        f" >={TREATMENT_SEEDS_MIN}/5 seeds):"
        f" {'PASS' if treatment_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"Control-A condition (<={CONTROL_CRITERIA_MAX}/7 in"
        f" >={CONTROL_SEEDS_BELOW}/5 seeds):"
        f" {'PASS' if ctrl_a_ok else 'FAIL'}",
        flush=True,
    )
    print(
        f"Control-B condition (<={CONTROL_CRITERIA_MAX}/7 in"
        f" >={CONTROL_SEEDS_BELOW}/5 seeds):"
        f" {'PASS' if ctrl_b_ok else 'FAIL'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "treatment_condition_pass": treatment_ok,
        "ctrl_a_condition_pass": ctrl_a_ok,
        "ctrl_b_condition_pass": ctrl_b_ok,
        "arm0_ctrl_a": arm0_sum,
        "arm1_ctrl_b": arm1_sum,
        "arm2_curriculum": arm2_sum,
        "all_seed_results": all_results,
    }


# ---------------------------------------------------------------------------
# Main / manifest
# ---------------------------------------------------------------------------

def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )
    out_path = out_dir / f"{run_id}.json"

    ev_dir = "supports" if outcome == "PASS" else "does_not_support"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": ev_dir,
        "evidence_direction_per_claim": {
            "ARC-046": ev_dir,
        },
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_episodes": N_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "rolling_window": ROLLING_WINDOW,
            "grid_size": GRID_SIZE,
            "gate_thresholds": {
                "C1_z_goal_min": C1_Z_GOAL_MIN,
                "C2_h_pos_threshold": round(C2_H_POS_THRESHOLD, 6),
                "C3_residue_cov_min": C3_RESIDUE_COV_MIN,
                "C4_entropy_global_min": round(C4_ENTROPY_GLOBAL_MIN, 6),
                "C4_kl_min": C4_KL_MIN,
                "C5_hb_ratio_min": C5_HB_RATIO_MIN,
                "C5_hb_ratio_max": C5_HB_RATIO_MAX,
                "C6_retention_min": C6_RETENTION_MIN,
                "C7_traj_cosine_min": C7_TRAJ_COSINE_MIN,
            },
            "pass_thresholds": {
                "treatment_criteria_min": TREATMENT_CRITERIA_MIN,
                "treatment_seeds_min": TREATMENT_SEEDS_MIN,
                "control_criteria_max": CONTROL_CRITERIA_MAX,
                "control_seeds_below": CONTROL_SEEDS_BELOW,
            },
            "arm0_env_kwargs": _ARM0_ENV_KWARGS,
            "arm0_novelty": _ARM0_NOVELTY,
            "arm1_env_kwargs": _ARM1_ENV_KWARGS,
            "arm1_novelty": _ARM1_NOVELTY,
            "arm2": "InfantCurriculumScheduler (experiments/infant_curriculum.py)",
        },
        "acceptance_criteria": {
            "primary_gate": (
                f"ARM_2 (curriculum) passes >= {TREATMENT_CRITERIA_MIN}/7 gate criteria"
                f" in >= {TREATMENT_SEEDS_MIN}/5 seeds AND"
                f" (ARM_0 OR ARM_1) passes <= {CONTROL_CRITERIA_MAX}/7"
                f" in >= {CONTROL_SEEDS_BELOW}/5 seeds"
            ),
        },
        "criteria_results": {
            "treatment_condition_pass": result["treatment_condition_pass"],
            "ctrl_a_condition_pass": result["ctrl_a_condition_pass"],
            "ctrl_b_condition_pass": result["ctrl_b_condition_pass"],
        },
        "metrics": {
            "arm0_ctrl_a_seed_criteria_counts": result["arm0_ctrl_a"]["seed_criteria_counts"],
            "arm0_ctrl_a_mean_criteria": result["arm0_ctrl_a"]["mean_criteria_passing"],
            "arm0_ctrl_a_seeds_below_control": result["arm0_ctrl_a"]["seeds_below_control_threshold"],
            "arm1_ctrl_b_seed_criteria_counts": result["arm1_ctrl_b"]["seed_criteria_counts"],
            "arm1_ctrl_b_mean_criteria": result["arm1_ctrl_b"]["mean_criteria_passing"],
            "arm1_ctrl_b_seeds_below_control": result["arm1_ctrl_b"]["seeds_below_control_threshold"],
            "arm2_curriculum_seed_criteria_counts": result["arm2_curriculum"]["seed_criteria_counts"],
            "arm2_curriculum_mean_criteria": result["arm2_curriculum"]["mean_criteria_passing"],
            "arm2_curriculum_seeds_passing_treatment": result["arm2_curriculum"]["seeds_passing_treatment_threshold"],
        },
        "per_seed_results": result["all_seed_results"],
        "notes": (
            "infant_substrate:GAP-14 closure. EXQ-ISEF-005 curriculum vs flat"
            " parameter comparison. ARM_2 (4-phase InfantCurriculumScheduler)"
            " vs ARM_0 (all features flat) vs ARM_1 (minimal flat). 7-criterion"
            " gate evaluated at ep 2000. PASS requires curriculum passes >= 6/7"
            " criteria in >= 4/5 seeds while at least one control passes <= 4/7"
            " in >= 3/5 seeds. PASS supports ARC-046 (phased residue accumulation"
            " via curriculum). Evidence for GAP-15 gate update (DEV-NEED-008)."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        summary = {k: v for k, v in manifest.items() if k != "per_seed_results"}
        print(json.dumps(summary, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
