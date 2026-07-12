"""V3-EXQ-572: Intervention A harm-affect dual attractor experiment.

Tests whether Intervention A parameters (affective_harm_scale=0.5,
urgency_weight=0.3, goal_weight=1.0) produce context-sensitive strategy
switching (behavioral diversity / Rung 3) in the SD-054 bipartite reef
environment.

ARM_0_baseline: no Intervention A parameters (control)
ARM_1_harm_affect: affective_harm_scale + urgency_weight only
ARM_2_goal: goal_weight only
ARM_3_all: full Intervention A (all three parameters)

Acceptance criteria (ARM_3 must pass >= 2/3 seeds):
  C1: TV of action distributions between hazard and resource contexts > 0.20
      (behavioral switching between harm-threat and food contexts)
  C2: z_harm_a_norm-conditioned reef_presence_rate delta > 0.15
      (affective harm drives reef-proximity preference)
  C3: goal_active-conditioned resource_presence_rate delta > 0.15
      (goal state drives foraging behavior)
  PASS = C1 AND (C2 OR C3)

Supports ARC-065: context-sensitive behavioral attractors via harm-affect
integration.

EXPERIMENT_PURPOSE = "evidence"
"""

import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_572_intervention_a_dual_attractor"

SEEDS = [42, 123, 456]
WARMUP_EPISODES = 20
EVAL_EPISODES = 30
TOTAL_EPISODES = WARMUP_EPISODES + EVAL_EPISODES
STEPS_PER_EPISODE = 200

GRID_SIZE = 12
NUM_HAZARDS = 2

# Pre-registered thresholds
C1_TV_THRESHOLD = 0.20
C2_DELTA_THRESHOLD = 0.15
C3_DELTA_THRESHOLD = 0.15
MIN_PASS_SEEDS = 2

ARM_CONFIGS = {
    "ARM_0_baseline": {
        "affective_harm_scale": 0.0,
        "urgency_weight": 0.0,
        "goal_weight": 0.0,   # explicit -- from_dims default is 1.0
        "z_goal_enabled": True,
    },
    "ARM_1_harm_affect": {
        "affective_harm_scale": 0.5,
        "urgency_weight": 0.3,
        "goal_weight": 0.0,   # explicit -- from_dims default is 1.0
        "z_goal_enabled": True,
    },
    "ARM_2_goal": {
        "affective_harm_scale": 0.0,
        "urgency_weight": 0.0,
        "goal_weight": 1.0,
        "z_goal_enabled": True,
    },
    "ARM_3_all": {
        "affective_harm_scale": 0.5,
        "urgency_weight": 0.3,
        "goal_weight": 1.0,
        "z_goal_enabled": True,
    },
}


def make_env_and_agent(seed, arm_params):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        use_proxy_fields=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis="horizontal",
        reef_bipartite_agent_band_radius=1,
        hazard_food_attraction=0.3,
        harm_history_len=10,
    )
    _, obs_dict = env.reset()

    body_obs_dim = obs_dict["body_state"].shape[0]
    world_obs_dim = obs_dict["world_state"].shape[0]
    action_dim = env.action_dim

    cfg = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=action_dim,
        harm_obs_dim=51,
        alpha_world=0.9,
        drive_weight=2.0,
        z_goal_enabled=arm_params["z_goal_enabled"],
        affective_harm_scale=arm_params["affective_harm_scale"],
        urgency_weight=arm_params["urgency_weight"],
        goal_weight=arm_params["goal_weight"],
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_history_len=10,
    )
    agent = REEAgent(config=cfg)
    agent.eval()
    return agent, env, obs_dict


def run_arm(arm_name, arm_params, seed, dry_run=False):
    agent, env, obs_dict = make_env_and_agent(seed, arm_params)

    n_episodes = 3 if dry_run else TOTAL_EPISODES

    # Metrics accumulators (eval phase only)
    action_counts_hazard = np.zeros(env.action_dim)
    action_counts_resource = np.zeros(env.action_dim)
    c2_records = []  # (z_harm_a_norm, reef_present)
    c3_records = []  # (goal_active, resource_present)

    for ep in range(n_episodes):
        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  [train] label seed={seed} ep {ep+1}/{TOTAL_EPISODES} arm={arm_name}",
                flush=True,
            )

        in_eval = ep >= WARMUP_EPISODES

        _, obs_dict = env.reset()
        agent.reset()

        for _step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].float()
            obs_world = obs_dict["world_state"].float()
            obs_harm = obs_dict.get("harm_obs")
            obs_harm_a = obs_dict.get("harm_obs_a")
            obs_harm_history = obs_dict.get("harm_history")

            if obs_harm is not None:
                obs_harm = obs_harm.float()
            if obs_harm_a is not None:
                obs_harm_a = obs_harm_a.float()
            if obs_harm_history is not None:
                obs_harm_history = obs_harm_history.float()

            with torch.no_grad():
                latent = agent.sense(
                    obs_body=obs_body,
                    obs_world=obs_world,
                    obs_harm=obs_harm,
                    obs_harm_a=obs_harm_a,
                    obs_harm_history=obs_harm_history,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks["e1_tick"]
                    else torch.zeros(1, agent.config.latent.world_dim)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)

            drive = float(REEAgent.compute_drive_level(obs_body))
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                agent.update_z_goal(float(harm_signal), drive_level=drive)
            agent.update_residue(float(harm_signal))

            if in_eval and obs_harm is not None:
                hazard_proximity = obs_harm[0:25].max().item()
                resource_proximity = obs_harm[25:50].max().item()
                is_hazard_ctx = hazard_proximity > 0.3
                is_resource_ctx = resource_proximity > 0.3

                if is_hazard_ctx:
                    action_counts_hazard[action_idx] += 1
                if is_resource_ctx:
                    action_counts_resource[action_idx] += 1

                z_harm_a_norm = 0.0
                if latent.z_harm_a is not None:
                    z_harm_a_norm = float(latent.z_harm_a.norm().item())

                reef_fv = obs_dict.get("reef_field_view")
                reef_present = reef_fv is not None and float(reef_fv.max().item()) > 0.3
                c2_records.append((z_harm_a_norm, reef_present))

                goal_active = False
                if agent.goal_state is not None:
                    goal_active = bool(agent.goal_state.is_active())
                c3_records.append((goal_active, is_resource_ctx))

            if done:
                agent.reset()
                _, obs_dict = env.reset()

    # Compute metrics
    total_hazard = action_counts_hazard.sum() + 1e-9
    total_resource = action_counts_resource.sum() + 1e-9
    p_hazard = action_counts_hazard / total_hazard
    p_resource = action_counts_resource / total_resource
    c1_tv = 0.5 * float(np.abs(p_hazard - p_resource).sum())

    c2_delta = 0.0
    if len(c2_records) >= 10:
        norms = [r[0] for r in c2_records]
        med_norm = float(np.median(norms))
        high_reef = [r[1] for r in c2_records if r[0] > med_norm]
        low_reef = [r[1] for r in c2_records if r[0] <= med_norm]
        if high_reef and low_reef:
            c2_delta = float(np.mean(high_reef)) - float(np.mean(low_reef))

    c3_delta = 0.0
    if len(c3_records) >= 10:
        active_res = [r[1] for r in c3_records if r[0]]
        inactive_res = [r[1] for r in c3_records if not r[0]]
        if active_res and inactive_res:
            c3_delta = float(np.mean(active_res)) - float(np.mean(inactive_res))

    c1_pass = c1_tv > C1_TV_THRESHOLD
    c2_pass = c2_delta > C2_DELTA_THRESHOLD
    c3_pass = c3_delta > C3_DELTA_THRESHOLD
    overall_pass = c1_pass and (c2_pass or c3_pass)

    return {
        "arm": arm_name,
        "seed": seed,
        "c1_tv": c1_tv,
        "c2_delta": c2_delta,
        "c3_delta": c3_delta,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "overall_pass": overall_pass,
        "n_c2_records": len(c2_records),
        "n_c3_records": len(c3_records),
    }


def run_experiment(dry_run=False):
    all_results = {}

    for arm_name, arm_params in ARM_CONFIGS.items():
        arm_results = []
        for seed in SEEDS:
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            r = run_arm(arm_name, arm_params, seed, dry_run=dry_run)
            arm_results.append(r)
            passed = r["overall_pass"]
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
        all_results[arm_name] = arm_results

    arm3_results = all_results.get("ARM_3_all", [])
    arm3_c1_pass_count = sum(1 for r in arm3_results if r["c1_pass"])
    arm3_c2_pass_count = sum(1 for r in arm3_results if r["c2_pass"])
    arm3_c3_pass_count = sum(1 for r in arm3_results if r["c3_pass"])
    arm3_overall_pass_count = sum(1 for r in arm3_results if r["overall_pass"])

    arm0_results = all_results.get("ARM_0_baseline", [])
    arm0_c1_pass_count = sum(1 for r in arm0_results if r["c1_pass"])

    arm_summaries = {}
    for arm_name, arm_results in all_results.items():
        arm_summaries[arm_name] = {
            "mean_c1_tv": float(np.mean([r["c1_tv"] for r in arm_results])),
            "mean_c2_delta": float(np.mean([r["c2_delta"] for r in arm_results])),
            "mean_c3_delta": float(np.mean([r["c3_delta"] for r in arm_results])),
            "c1_pass_seeds": sum(1 for r in arm_results if r["c1_pass"]),
            "c2_pass_seeds": sum(1 for r in arm_results if r["c2_pass"]),
            "c3_pass_seeds": sum(1 for r in arm_results if r["c3_pass"]),
            "overall_pass_seeds": sum(1 for r in arm_results if r["overall_pass"]),
        }

    if arm3_overall_pass_count >= MIN_PASS_SEEDS:
        outcome = "PASS"
        outcome_note = (
            f"ARM_3 (full Intervention A) passed overall on "
            f"{arm3_overall_pass_count}/3 seeds "
            f"(C1 TV>{C1_TV_THRESHOLD} AND "
            f"(C2 delta>{C2_DELTA_THRESHOLD} OR C3 delta>{C3_DELTA_THRESHOLD})). "
            f"Supports ARC-065 dual attractor via harm-affect integration."
        )
    else:
        outcome = "FAIL"
        outcome_note = (
            f"ARM_3 (full Intervention A) passed overall on only "
            f"{arm3_overall_pass_count}/3 seeds "
            f"(C1 pass: {arm3_c1_pass_count}/3, "
            f"C2 pass: {arm3_c2_pass_count}/3, "
            f"C3 pass: {arm3_c3_pass_count}/3). "
            f"Behavioral diversity criterion not met."
        )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_summaries": arm_summaries,
        "per_seed_results": all_results,
        "arm3_c1_pass_count": arm3_c1_pass_count,
        "arm3_c2_pass_count": arm3_c2_pass_count,
        "arm3_c3_pass_count": arm3_c3_pass_count,
        "arm3_overall_pass_count": arm3_overall_pass_count,
        "arm0_c1_pass_count": arm0_c1_pass_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-572: Intervention A dual attractor"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Short run for smoke testing"
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_572_intervention_a_dual_attractor_{timestamp}_v3"

    out_dir = (
        Path(__file__).parent.parent.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["ARC-065"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": (
            "supports" if result["outcome"] == "PASS" else "weakens"
        ),
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "warmup_episodes": WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "total_episodes": TOTAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "arm_names": list(ARM_CONFIGS.keys()),
        "arm_summaries": result["arm_summaries"],
        "per_seed_results": {
            arm: [
                {
                    k: (
                        bool(v)
                        if isinstance(v, (bool, np.bool_))
                        else float(v)
                        if isinstance(v, (float, np.floating))
                        else int(v)
                        if isinstance(v, (int, np.integer))
                        else str(v)
                    )
                    for k, v in r.items()
                }
                for r in results
            ]
            for arm, results in result["per_seed_results"].items()
        },
        "thresholds": {
            "c1_tv": C1_TV_THRESHOLD,
            "c2_delta": C2_DELTA_THRESHOLD,
            "c3_delta": C3_DELTA_THRESHOLD,
            "min_pass_seeds": MIN_PASS_SEEDS,
        },
        "arm3_c1_pass_count": result["arm3_c1_pass_count"],
        "arm3_c2_pass_count": result["arm3_c2_pass_count"],
        "arm3_c3_pass_count": result["arm3_c3_pass_count"],
        "arm3_overall_pass_count": result["arm3_overall_pass_count"],
        "arm0_c1_pass_count": result["arm0_c1_pass_count"],
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        json_default=str,
    )
    print(f"Manifest written: {out_path}")

    if args.dry_run:
        print("DRY RUN complete.")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
