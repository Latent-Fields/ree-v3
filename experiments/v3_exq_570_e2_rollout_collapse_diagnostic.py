"""V3-EXQ-570: E2 rollout collapse diagnostic.

Stage 3 of the behavioral diversity collapse chain: even when CEM produces
diverse first-action candidates (Stage 1 fixed by SP-CEM, EXQ-567), E2 dynamics
over H=30 rollout steps may converge all candidates to identical terminal z_world
states, eliminating diversity before E3 scoring.

Interpretation grid:
  ARM_1 diversity_preservation_ratio > 0.5  -> Stage 3 is NOT the bottleneck;
      z_world diversity survives rollout. Look at E3 scoring or post-commit collapse.
  ARM_1 ratio 0.3-0.5  -> Partial collapse; E2 attenuates diversity but does not
      eliminate it. Stage 3 is a contributing factor.
  ARM_1 ratio < 0.3    -> Stage 3 IS a bottleneck; E2 rollout collapses candidates.
      Needs a diversity-preserving rollout objective or shorter horizon.
  ARM_1 first_step_spread < 0.01 -> Class-anchored sequences failed to produce
      diverse first-step z_world; experiment is non-contributory at init.

EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence scoring.
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

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_570_e2_rollout_collapse_diagnostic"

# Pre-registered thresholds
PASS_THRESHOLD = 0.5
FAIL_THRESHOLD = 0.3
NON_CONTRIBUTORY_SPREAD_MIN = 0.01

SEEDS = [42, 123, 456]
N_PROBE_STEPS = 50
N_RANDOM_CANDIDATES = 32
GRID_SIZE = 8
NUM_HAZARDS = 1

ARM_NAMES = ["ARM_0_random_cem", "ARM_1_class_scaffold", "ARM_2_mixed"]


def make_env_and_agent(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        use_proxy_fields=True,
    )
    _, obs_dict = env.reset()

    body_obs_dim = obs_dict["body_state"].shape[0]
    world_obs_dim = obs_dict["world_state"].shape[0]
    action_dim = env.action_dim

    config = REEConfig().from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=action_dim,
        harm_obs_dim=51,
    )
    config.latent.alpha_world = 0.9  # SD-008: high fidelity z_world

    agent = REEAgent(config=config)
    agent.eval()

    horizon = config.e2.rollout_horizon
    return agent, env, obs_dict, action_dim, horizon


def build_candidates(arm_name, action_dim, horizon, n_random):
    """Build action sequences [N, horizon, action_dim] for each arm."""
    if arm_name == "ARM_0_random_cem":
        return torch.randn(n_random, horizon, action_dim)

    elif arm_name == "ARM_1_class_scaffold":
        seqs = []
        for ac in range(action_dim):
            seq = torch.zeros(horizon, action_dim)
            seq[0, ac] = 1.0  # one-hot first action, rest zeros
            seqs.append(seq)
        return torch.stack(seqs, dim=0)  # [4, horizon, action_dim]

    elif arm_name == "ARM_2_mixed":
        class_seqs = []
        for ac in range(action_dim):
            seq = torch.zeros(horizon, action_dim)
            seq[0, ac] = 1.0
            class_seqs.append(seq)
        n_fill = max(0, n_random - action_dim)
        if n_fill > 0:
            fill = torch.randn(n_fill, horizon, action_dim)
            return torch.cat([torch.stack(class_seqs, dim=0), fill], dim=0)
        return torch.stack(class_seqs, dim=0)

    else:
        raise ValueError(f"Unknown arm: {arm_name}")


def compute_spread(z_list):
    """Mean std across latent dims for a list of z tensors."""
    if len(z_list) < 2:
        return 0.0
    stacked = torch.stack(z_list, dim=0)
    return stacked.std(dim=0).mean().item()


def run_arm(arm_name, seed, dry_run=False):
    agent, env, obs_dict, action_dim, horizon = make_env_and_agent(seed)
    n_steps = 3 if dry_run else N_PROBE_STEPS

    first_step_spreads = []
    terminal_spreads = []

    for step_i in range(n_steps):
        if (step_i + 1) % 10 == 0 or step_i == 0:
            print(
                f"  [train] label seed={seed} ep {step_i+1}/{n_steps} arm={arm_name}",
                flush=True,
            )

        obs_body = torch.tensor(obs_dict["body_state"]).float()
        obs_world = torch.tensor(obs_dict["world_state"]).float()
        latent = agent.sense(obs_body=obs_body, obs_world=obs_world)

        z_world = latent.z_world
        z_self = latent.z_self
        if z_world is None:
            _, _, terminated, _, obs_dict = env.step(int(np.random.randint(env.action_dim)))
            if terminated:
                _, obs_dict = env.reset()
            continue
        if z_world.dim() == 2:
            z_world = z_world.squeeze(0)
        if z_self is not None and z_self.dim() == 2:
            z_self = z_self.squeeze(0)
        z_world = z_world.detach()
        z_self = z_self.detach() if z_self is not None else torch.zeros_like(z_world)

        candidates = build_candidates(arm_name, action_dim, horizon, N_RANDOM_CANDIDATES)

        first_zw = []
        terminal_zw = []

        with torch.no_grad():
            for i in range(candidates.shape[0]):
                seq = candidates[i].unsqueeze(0)  # [1, horizon, action_dim]
                traj = agent.e2.rollout_with_world(
                    initial_z_self=z_self.unsqueeze(0),
                    initial_z_world=z_world.unsqueeze(0),
                    action_sequence=seq,
                )
                ws = traj.world_states
                if ws is not None and len(ws) >= 2:
                    w1 = ws[1]
                    wt = ws[-1]
                    if w1.dim() == 2:
                        w1 = w1.squeeze(0)
                    if wt.dim() == 2:
                        wt = wt.squeeze(0)
                    first_zw.append(w1.detach())
                    terminal_zw.append(wt.detach())

        if len(first_zw) >= 2:
            first_step_spreads.append(compute_spread(first_zw))
            terminal_spreads.append(compute_spread(terminal_zw))

        action = int(np.random.randint(env.action_dim))
        _, _, terminated, _, obs_dict = env.step(action)
        if terminated:
            _, obs_dict = env.reset()

    mean_fs = float(np.mean(first_step_spreads)) if first_step_spreads else 0.0
    mean_ts = float(np.mean(terminal_spreads)) if terminal_spreads else 0.0
    ratio = mean_ts / (mean_fs + 1e-6)

    return {
        "arm": arm_name,
        "seed": seed,
        "n_valid": len(first_step_spreads),
        "mean_first_step_spread": mean_fs,
        "mean_terminal_spread": mean_ts,
        "diversity_preservation_ratio": ratio,
    }


def run_experiment(dry_run=False):
    all_results = {}

    for arm_name in ARM_NAMES:
        arm_results = []
        for seed in SEEDS:
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            r = run_arm(arm_name, seed, dry_run=dry_run)
            arm_results.append(r)
            passed_seed = r["diversity_preservation_ratio"] >= PASS_THRESHOLD
            print(f"verdict: {'PASS' if passed_seed else 'FAIL'}", flush=True)
        all_results[arm_name] = arm_results

    arm_summaries = {}
    for arm_name, arm_results in all_results.items():
        valid = [r for r in arm_results if r["n_valid"] > 0]
        if valid:
            arm_summaries[arm_name] = {
                "mean_first_step_spread": float(np.mean([r["mean_first_step_spread"] for r in valid])),
                "mean_terminal_spread": float(np.mean([r["mean_terminal_spread"] for r in valid])),
                "diversity_preservation_ratio": float(np.mean([r["diversity_preservation_ratio"] for r in valid])),
                "n_valid_seeds": len(valid),
            }
        else:
            arm_summaries[arm_name] = {
                "mean_first_step_spread": 0.0,
                "mean_terminal_spread": 0.0,
                "diversity_preservation_ratio": 0.0,
                "n_valid_seeds": 0,
            }

    arm1 = arm_summaries.get("ARM_1_class_scaffold", {})
    arm1_ratio = arm1.get("diversity_preservation_ratio", 0.0)
    arm1_first_step = arm1.get("mean_first_step_spread", 0.0)

    if arm1_first_step < NON_CONTRIBUTORY_SPREAD_MIN:
        outcome = "FAIL"
        outcome_note = (
            f"non_contributory: scaffold first_step_spread={arm1_first_step:.4f} < "
            f"{NON_CONTRIBUTORY_SPREAD_MIN}; z_world at random init lacks sensitivity "
            f"to first-action diversity"
        )
    elif arm1_ratio >= PASS_THRESHOLD:
        outcome = "PASS"
        outcome_note = (
            f"E2 rollout preserves diversity (ARM_1 ratio={arm1_ratio:.3f} >= {PASS_THRESHOLD}); "
            f"Stage 3 is NOT the bottleneck"
        )
    elif arm1_ratio < FAIL_THRESHOLD:
        outcome = "FAIL"
        outcome_note = (
            f"E2 rollout collapses diversity (ARM_1 ratio={arm1_ratio:.3f} < {FAIL_THRESHOLD}); "
            f"Stage 3 IS a bottleneck"
        )
    else:
        outcome = "FAIL"
        outcome_note = (
            f"Partial E2 collapse (ARM_1 ratio={arm1_ratio:.3f} in [{FAIL_THRESHOLD}, {PASS_THRESHOLD}]); "
            f"Stage 3 is a contributing factor"
        )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_summaries": arm_summaries,
        "per_seed_results": all_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-570: E2 rollout collapse diagnostic")
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_570_e2_rollout_collapse_diagnostic_{timestamp}_v3"

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
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "n_probe_steps": N_PROBE_STEPS,
        "n_random_candidates": N_RANDOM_CANDIDATES,
        "arm_names": ARM_NAMES,
        "arm_summaries": result["arm_summaries"],
        "per_seed_results": {
            arm: [
                {k: (v if not isinstance(v, float) else float(v))
                 for k, v in r.items()}
                for r in results
            ]
            for arm, results in result["per_seed_results"].items()
        },
        "thresholds": {
            "pass": PASS_THRESHOLD,
            "fail": FAIL_THRESHOLD,
            "non_contributory_spread_min": NON_CONTRIBUTORY_SPREAD_MIN,
        },
    }

    out_path = out_dir / f"{run_id}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Manifest written: {out_path}")

    if args.dry_run:
        print("DRY RUN complete.")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
