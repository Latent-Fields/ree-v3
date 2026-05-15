#!/usr/bin/env python3
"""
V3-EXQ-570: E2 Rollout Collapse Diagnostic

Diagnostic experiment testing whether E2 rollout collapse prevents candidate
diversity from becoming behavioural diversity in the ree-v3 architecture.

4 arms measure candidate and z_world diversity at different intervention points:
  ARM_0_baseline   -- standard agent + env, temperature=1.0
  ARM_1_high_temp  -- standard env, temperature=5.0
  ARM_2_bipartite  -- SD-054 bipartite env (categorically distinct niches), temperature=1.0
  ARM_3_oracle     -- bypass CEM; inject diverse synthetic action seqs into E2.rollout_with_world

PASS criterion (pre-registered threshold):
  ARM_3 mean oracle_z_world_pairwise_dist > 0.05 (across all seeds)

Interpretation grid (one row per plausible outcome -> next action):
  oracle FAIL                            -> E2/LatentStack z_world insensitive to action;
                                           root cause in substrate; check alpha_world, encoder
  oracle PASS, arm0 entropy < 0.3 nat   -> CEM/hippocampus collapses proposal diversity;
                                           check hippocampal.propose_trajectories() distribution
  oracle PASS, arm1 entropy > arm0      -> temperature lever effective;
                                           raise temperature or soften CEM scoring
  oracle PASS, arm2 entropy > arm0      -> env contrast lever sufficient;
                                           bipartite env or env-diversity injection recommended
  oracle PASS, all arms low entropy     -> scoring/selection collapses diverse proposals;
                                           check E3._running_variance and score sharpening
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent, REEConfig, candidate_support_preflight
from ree_core.environment.causal_grid_world import CausalGridWorldV2

QUEUE_ID = "V3-EXQ-570"
EXPERIMENT_TYPE = "v3_exq_570_e2_rollout_collapse_diagnostic"
EXPERIMENT_PURPOSE = "diagnostic"

# Pre-registered thresholds
THRESHOLD_ORACLE_Z_WORLD_DIST = 0.05   # ARM_3 PASS if mean pairwise L2 > this
THRESHOLD_ARM0_ENTROPY_DIAGNOSTIC = 0.3  # below this flags candidate collapse in ARM_0

# Run config
SEEDS = [42, 123]
EPISODES_PER_ARM = 20
STEPS_PER_EPISODE = 30
WARMUP_STEPS = 5          # steps before measuring to let latents initialise
ORACLE_ROLLOUT_HORIZON = 8

ARM_LABELS = [
    "ARM_0_baseline",
    "ARM_1_high_temp",
    "ARM_2_bipartite",
    "ARM_3_oracle",
]
ARM_TEMPERATURES = {
    "ARM_0_baseline": 1.0,
    "ARM_1_high_temp": 5.0,
    "ARM_2_bipartite": 1.0,
    "ARM_3_oracle": 1.0,
}
ARM_BIPARTITE = {
    "ARM_0_baseline": False,
    "ARM_1_high_temp": False,
    "ARM_2_bipartite": True,
    "ARM_3_oracle": True,
}


def make_env(arm_label, seed):
    kwargs = {"num_hazards": 1, "seed": seed}
    if ARM_BIPARTITE[arm_label]:
        kwargs.update({
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "reef_bipartite_axis": "horizontal",
            "reef_bipartite_agent_band_radius": 1,
        })
    return CausalGridWorldV2(**kwargs)


def make_agent(env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _flat, obs_dict = env.reset()
    obs_body_dim = int(obs_dict["body_state"].shape[0])
    obs_world_dim = int(obs_dict["world_state"].shape[0])
    action_dim = int(env.action_dim)
    config = REEConfig.from_dims(
        body_obs_dim=obs_body_dim,
        world_obs_dim=obs_world_dim,
        action_dim=action_dim,
        alpha_world=0.9,  # SD-008: required for z_world fidelity
    )
    agent = REEAgent(config)
    agent.eval()
    return agent, action_dim


def mean_pairwise_l2(vecs):
    """Mean pairwise L2 distance over a list of numpy vectors."""
    if len(vecs) < 2:
        return 0.0
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    return float(np.mean(dists)) if dists else 0.0


def first_action_seq_diversity(candidates):
    """Mean pairwise L2 of first-step actions across candidate trajectories."""
    if not candidates or len(candidates) < 2:
        return 0.0
    # Trajectory.actions: [batch=1, horizon, action_dim]
    first_actions = [c.actions[0, 0].detach().cpu().numpy() for c in candidates]
    return mean_pairwise_l2(first_actions)


def run_arm(arm_label, seed, episodes, dry_run=False):
    """Run one arm×seed combination and return aggregated metrics dict."""
    env = make_env(arm_label, seed)
    agent, action_dim = make_agent(env, seed)
    temperature = ARM_TEMPERATURES[arm_label]
    ep_count = 2 if dry_run else episodes

    arm_first_entropy = []
    arm_seq_diversity = []
    arm_e3_variance = []
    arm_oracle_z_world_dist = []

    for ep in range(ep_count):
        flat_obs, obs_dict = env.reset()
        obs_body_t = torch.as_tensor(obs_dict["body_state"], dtype=torch.float32).unsqueeze(0)
        obs_world_t = torch.as_tensor(obs_dict["world_state"], dtype=torch.float32).unsqueeze(0)

        ep_first_entropy = []
        ep_seq_diversity = []
        ep_e3_variance = []
        ep_oracle_z_world_dist = []

        for step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                action_t = agent.act_with_split_obs(
                    obs_body_t, obs_world_t, temperature=temperature
                )
            action_np = int(action_t.argmax(dim=-1).item())

            if step >= WARMUP_STEPS:
                if arm_label == "ARM_3_oracle":
                    latent = agent._current_latent
                    if latent is not None:
                        z_self = latent.z_self   # [1, self_dim]
                        z_world = latent.z_world  # [1, world_dim]
                        n_dirs = min(4, action_dim)

                        # Construct 2*n_dirs diverse action sequences (pos + neg per dim)
                        seqs = []
                        for d in range(n_dirs):
                            s = torch.zeros(1, ORACLE_ROLLOUT_HORIZON, action_dim)
                            s[0, :, d] = 1.0
                            seqs.append(s)
                        for d in range(n_dirs):
                            s = torch.zeros(1, ORACLE_ROLLOUT_HORIZON, action_dim)
                            s[0, :, d] = -1.0
                            seqs.append(s)

                        final_z_worlds = []
                        with torch.no_grad():
                            for s in seqs:
                                traj = agent.hippocampal.e2.rollout_with_world(
                                    z_self, z_world, s,
                                    compute_action_objects=False,
                                )
                                if traj.world_states and len(traj.world_states) > 0:
                                    zw = traj.world_states[-1][0].detach().cpu().numpy()
                                    final_z_worlds.append(zw)

                        if len(final_z_worlds) >= 2:
                            ep_oracle_z_world_dist.append(mean_pairwise_l2(final_z_worlds))
                else:
                    candidates = agent._committed_candidates
                    if candidates is not None:
                        try:
                            sup = candidate_support_preflight(candidates)
                            ep_first_entropy.append(
                                float(sup.get("candidate_first_action_entropy", 0.0))
                            )
                        except Exception:
                            pass
                        ep_seq_diversity.append(first_action_seq_diversity(candidates))

                    try:
                        ep_e3_variance.append(float(agent.e3._running_variance))
                    except Exception:
                        pass

            flat_obs, harm_signal, done, info, next_obs_dict = env.step(action_np)
            obs_dict = next_obs_dict
            obs_body_t = torch.as_tensor(obs_dict["body_state"], dtype=torch.float32).unsqueeze(0)
            obs_world_t = torch.as_tensor(obs_dict["world_state"], dtype=torch.float32).unsqueeze(0)
            if done:
                break

        if ep_first_entropy:
            arm_first_entropy.append(float(np.mean(ep_first_entropy)))
        if ep_seq_diversity:
            arm_seq_diversity.append(float(np.mean(ep_seq_diversity)))
        if ep_e3_variance:
            arm_e3_variance.append(float(np.mean(ep_e3_variance)))
        if ep_oracle_z_world_dist:
            arm_oracle_z_world_dist.append(float(np.mean(ep_oracle_z_world_dist)))

        if (ep + 1) % max(1, ep_count // 4) == 0 or ep + 1 == ep_count:
            print(
                f"  [train] {arm_label} seed={seed} ep {ep + 1}/{ep_count} ...",
                flush=True,
            )

    return {
        "arm": arm_label,
        "seed": seed,
        "episodes": ep_count,
        "mean_first_action_entropy": float(np.mean(arm_first_entropy)) if arm_first_entropy else None,
        "mean_action_seq_diversity": float(np.mean(arm_seq_diversity)) if arm_seq_diversity else None,
        "mean_e3_score_variance": float(np.mean(arm_e3_variance)) if arm_e3_variance else None,
        "mean_oracle_z_world_pairwise_dist": (
            float(np.mean(arm_oracle_z_world_dist)) if arm_oracle_z_world_dist else None
        ),
    }


def run_experiment(seeds, episodes, dry_run=False):
    all_results = {}
    oracle_passed_flags = []

    for seed in seeds:
        for arm_label in ARM_LABELS:
            print(f"Seed {seed} Condition {arm_label}", flush=True)
            result = run_arm(arm_label, seed, episodes, dry_run=dry_run)

            if arm_label == "ARM_3_oracle":
                dist = result.get("mean_oracle_z_world_pairwise_dist")
                arm_passed = dist is not None and dist > THRESHOLD_ORACLE_Z_WORLD_DIST
                oracle_passed_flags.append(arm_passed)
                result["arm_passed"] = arm_passed
            else:
                result["arm_passed"] = True  # non-oracle arms are diagnostic; don't gate outcome

            if arm_label == "ARM_0_baseline":
                entropy = result.get("mean_first_action_entropy")
                if entropy is not None and entropy < THRESHOLD_ARM0_ENTROPY_DIAGNOSTIC:
                    print(
                        f"  [diag] arm0 first_action_entropy={entropy:.4f} < "
                        f"{THRESHOLD_ARM0_ENTROPY_DIAGNOSTIC} -> candidate collapse flagged",
                        flush=True,
                    )

            all_results[f"seed{seed}_{arm_label}"] = result
            print(f"verdict: {'PASS' if result['arm_passed'] else 'FAIL'}", flush=True)

    overall_passed = bool(oracle_passed_flags) and all(oracle_passed_flags)
    return all_results, "PASS" if overall_passed else "FAIL", oracle_passed_flags


def main():
    parser = argparse.ArgumentParser(description="V3-EXQ-570 E2 Rollout Collapse Diagnostic")
    parser.add_argument("--dry-run", action="store_true", help="2 episodes per arm for fast validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    seeds = args.seeds or SEEDS
    dry_run = args.dry_run

    if dry_run:
        print("DRY RUN: 2 episodes per arm", flush=True)

    print(f"Starting {QUEUE_ID}: E2 Rollout Collapse Diagnostic", flush=True)
    print(
        f"Seeds: {seeds}, Episodes/arm: {2 if dry_run else EPISODES_PER_ARM}, "
        f"Arms: {len(ARM_LABELS)}, oracle threshold: {THRESHOLD_ORACLE_Z_WORLD_DIST}",
        flush=True,
    )

    all_results, outcome, oracle_flags = run_experiment(seeds, EPISODES_PER_ARM, dry_run=dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "outcome": outcome,
        "timestamp_utc": timestamp,
        "seeds": seeds,
        "episodes_per_arm": 2 if dry_run else EPISODES_PER_ARM,
        "steps_per_episode": STEPS_PER_EPISODE,
        "warmup_steps": WARMUP_STEPS,
        "oracle_rollout_horizon": ORACLE_ROLLOUT_HORIZON,
        "threshold_oracle_z_world_dist": THRESHOLD_ORACLE_Z_WORLD_DIST,
        "threshold_arm0_entropy_diagnostic": THRESHOLD_ARM0_ENTROPY_DIAGNOSTIC,
        "arm_labels": ARM_LABELS,
        "all_results": all_results,
        "oracle_passed_per_seed": oracle_flags,
        "dry_run": dry_run,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "REE_assembly", "evidence", "experiments",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {outcome}", flush=True)

    return outcome, out_path


if __name__ == "__main__":
    _outcome, _out_path = main()
    _outcome_raw = str(_outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
