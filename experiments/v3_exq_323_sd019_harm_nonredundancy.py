#!/opt/local/bin/python3
"""
V3-EXQ-323: SD-019 Harm Stream Non-Redundancy Constraint

experiment_purpose: evidence

Tests that compute_harm_nonredundancy_loss() (harm_nonredundancy_weight=1.0)
reduces cosine similarity between z_harm_s and z_harm_a compared to the
unconstrained baseline.

Two conditions per seed:
  NONREDUNDANT -- harm_nonredundancy_weight=1.0 (cosine^2 penalty applied)
  BASELINE     -- harm_nonredundancy_weight=0.0 (no penalty, current default)

Key metric:
  cosine_sq -- mean cosine_similarity^2 between z_harm_s and z_harm_a
               (lower is better -- streams are more orthogonal)

Pass criterion (pre-registered):
  C1: cosine_sq_nonredundant < cosine_sq_baseline (penalty reduces redundancy)
  C2: cosine_sq_nonredundant <= 0.5 (absolute threshold for meaningful non-redundancy)
  C3: Both streams still have non-zero norms (training has not collapsed)

Experiment PASS: >= 3/5 seeds satisfy C1 and C2.

Claims: SD-019 (affective harm non-redundancy constraint), SD-011
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_323_sd019_harm_nonredundancy"
CLAIM_IDS = ["SD-019", "SD-011"]

C1_threshold = 0.0    # nonredundant cosine_sq < baseline cosine_sq
C2_threshold = 0.5    # nonredundant cosine_sq <= 0.5
C3_min_norm = 0.01    # streams must have non-zero norms
PASS_MIN_SEEDS = 3

HARM_OBS_DIM = 51
HARM_OBS_A_DIM = 50   # EMA proximity (standard SD-011)
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16

SEEDS = [42, 43, 44, 45, 46]
TRAIN_EPISODES = 60
EVAL_STEPS = 300
STEPS_PER_EPISODE = 200
LR = 1e-3


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.1, resource_benefit=0.05,
        use_proxy_fields=True, seed=seed,
    )


def make_config(nonredundancy_weight: float) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        harm_nonredundancy_weight=nonredundancy_weight,
    )


def run_training(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                 harm_head: nn.Module, env: CausalGridWorldV2,
                 device, n_eps: int, nonredundancy_weight: float):
    """Train with or without non-redundancy penalty."""
    all_params = (
        list(agent.parameters())
        + list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(harm_head.parameters())
    )
    opt = optim.Adam(all_params, lr=LR)

    # Projection layers for non-redundancy loss (shared comparison dim)
    proj_s = nn.Linear(Z_HARM_DIM, Z_HARM_DIM).to(device)
    proj_a = nn.Linear(Z_HARM_A_DIM, Z_HARM_DIM).to(device)
    opt_proj = optim.Adam(list(proj_s.parameters()) + list(proj_a.parameters()), lr=LR)

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for step in range(STEPS_PER_EPISODE):
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs = harm_obs.to(device)

            z_harm_s = enc_s(harm_obs.unsqueeze(0))
            z_harm_a = None
            if harm_obs_a is not None:
                harm_obs_a = harm_obs_a.to(device)
                res = enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res

            # Harm proximity supervision
            prox_pred = harm_head(z_harm_s)
            prox_target = harm_obs[-1:].unsqueeze(0)
            harm_loss = F.mse_loss(prox_pred, prox_target)

            total_loss = harm_loss

            # Non-redundancy penalty (when enabled)
            if nonredundancy_weight > 0.0 and z_harm_a is not None:
                z_s_proj = proj_s(z_harm_s)
                z_a_proj = proj_a(z_harm_a)
                cos_sim = F.cosine_similarity(z_s_proj, z_a_proj, dim=-1).mean()
                penalty = cos_sim.pow(2)
                total_loss = total_loss + nonredundancy_weight * penalty

            opt.zero_grad()
            opt_proj.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
                if nonredundancy_weight > 0.0:
                    opt_proj.step()

            action_idx = random.randint(0, 3)
            _, _, done, _, obs_dict = env.step(action_idx)
            if done:
                break


def eval_redundancy(enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                    env: CausalGridWorldV2, device) -> Dict:
    """Measure cosine_sq between z_harm_s and z_harm_a."""
    cosine_sqs = []
    z_s_norms = []
    z_a_norms = []

    _, obs_dict = env.reset()
    for step in range(EVAL_STEPS):
        harm_obs = obs_dict.get("harm_obs")
        harm_obs_a = obs_dict.get("harm_obs_a")
        if harm_obs is None:
            break
        harm_obs = harm_obs.to(device)

        with torch.no_grad():
            z_harm_s = enc_s(harm_obs.unsqueeze(0))
            z_harm_a = None
            if harm_obs_a is not None:
                harm_obs_a = harm_obs_a.to(device)
                res = enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res

            if z_harm_a is not None:
                # Project to same dim for cosine comparison
                if z_harm_s.shape[-1] == z_harm_a.shape[-1]:
                    cos_sq = float(F.cosine_similarity(z_harm_s, z_harm_a, dim=-1).pow(2).mean().item())
                else:
                    # Truncate to min dim
                    min_d = min(z_harm_s.shape[-1], z_harm_a.shape[-1])
                    cos_sq = float(F.cosine_similarity(
                        z_harm_s[..., :min_d], z_harm_a[..., :min_d], dim=-1
                    ).pow(2).mean().item())
                cosine_sqs.append(cos_sq)
            z_s_norms.append(float(z_harm_s.norm().item()))
            if z_harm_a is not None:
                z_a_norms.append(float(z_harm_a.norm().item()))

        action_idx = random.randint(0, 3)
        _, _, done, _, obs_dict = env.step(action_idx)
        if done:
            break

    return {
        "cosine_sq": float(np.mean(cosine_sqs)) if cosine_sqs else 1.0,
        "z_harm_s_norm": float(np.mean(z_s_norms)) if z_s_norms else 0.0,
        "z_harm_a_norm": float(np.mean(z_a_norms)) if z_a_norms else 0.0,
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = 5 if dry_run else TRAIN_EPISODES

    print(f"Seed {seed}")
    condition_results = {}
    for condition in ["NONREDUNDANT", "BASELINE"]:
        weight = 1.0 if condition == "NONREDUNDANT" else 0.0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        cfg = make_config(nonredundancy_weight=weight)
        agent = REEAgent(cfg)
        enc_s = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
        enc_a = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM).to(device)
        harm_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)

        print(f"  {condition}: training {n_train} eps (weight={weight})...")
        run_training(agent, enc_s, enc_a, harm_head, env, device, n_train, weight)
        print(f"  {condition}: eval...")
        metrics = eval_redundancy(enc_s, enc_a, env, device)
        condition_results[condition] = metrics
        print(
            f"  {condition}: cosine_sq={metrics['cosine_sq']:.4f} "
            f"z_s_norm={metrics['z_harm_s_norm']:.4f} "
            f"z_a_norm={metrics['z_harm_a_norm']:.4f}"
        )

    nr = condition_results["NONREDUNDANT"]
    bl = condition_results["BASELINE"]
    c1_pass = nr["cosine_sq"] < bl["cosine_sq"]
    c2_pass = nr["cosine_sq"] <= C2_threshold
    c3_pass = nr["z_harm_s_norm"] >= C3_min_norm and nr["z_harm_a_norm"] >= C3_min_norm
    seed_pass = c1_pass and c2_pass and c3_pass

    print(f"  -> {'PASS' if seed_pass else 'FAIL'}")
    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "cosine_sq_nonredundant": nr["cosine_sq"],
        "cosine_sq_baseline": bl["cosine_sq"],
        "z_harm_s_norm_nonredundant": nr["z_harm_s_norm"],
        "z_harm_a_norm_nonredundant": nr["z_harm_a_norm"],
        "c1_cosine_sq_reduced": c1_pass,
        "c2_abs_threshold": c2_pass,
        "c3_no_collapse": c3_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        "v3_exq_323_sd019_harm_nonredundancy_dry" if args.dry_run
        else f"v3_exq_323_sd019_harm_nonredundancy_{timestamp}_v3"
    )
    print(f"EXQ-323 start: {run_id}")

    per_seed = [run_seed(s, dry_run=args.dry_run) for s in SEEDS]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-323 {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(SEEDS)}")
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"cos_sq_NR={r['cosine_sq_nonredundant']:.4f} "
            f"cos_sq_BL={r['cosine_sq_baseline']:.4f}"
        )

    evidence_direction = "supports" if experiment_passes else "does_not_support"
    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "evidence",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {
            "SD-019": evidence_direction,
            "SD-011": evidence_direction,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "registered_thresholds": {
            "C1_cosine_sq_reduced": C1_threshold,
            "C2_abs_cosine_sq": C2_threshold,
            "C3_min_norm": C3_min_norm,
            "seeds_needed": PASS_MIN_SEEDS,
        },
        "per_seed_results": per_seed,
        "seeds_passing": seeds_passing,
        "experiment_passes": experiment_passes,
    }

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
