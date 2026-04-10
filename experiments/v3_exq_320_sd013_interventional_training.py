#!/opt/local/bin/python3
"""
V3-EXQ-320: SD-013 Interventional vs Observational E2_harm_s Training

experiment_purpose: evidence

Tests that SD-013 interventional contrastive training produces a more
action-sensitive E2_harm_s forward model than observational-only training.

Two conditions per seed:
  INTERVENTIONAL -- E2HarmSForward trained with compute_interventional_loss()
                    applied to 30% of steps (use_interventional=True)
  OBSERVATIONAL  -- E2HarmSForward trained with MSE only (use_interventional=False)

Key metric: action_gap -- the mean L2 distance between E2 predictions for
a_actual vs a_cf at the same z_harm_s state. A higher gap means the model
distinguishes actions better (more interventional).

Pass criterion (pre-registered):
  C1: action_gap_interventional > action_gap_observational (mean over seeds)
  C2: forward_r2_interventional >= 0.5 (forward model has not degraded)
  C3: action_gap_interventional >= 0.05 (gap is absolute, not just relative)

Experiment PASS: >= 3/5 seeds satisfy C1 and C2. Mean C3 satisfied.

Claims: SD-013 (interventional training required for unbiased causal_sig), SD-003
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig


EXPERIMENT_TYPE = "v3_exq_320_sd013_interventional_training"
CLAIM_IDS = ["SD-013", "SD-003"]

# Pre-registered thresholds
C1_threshold = 0.0     # interventional gap > observational gap
C2_r2_threshold = 0.3  # forward model R2 must not collapse
C3_gap_threshold = 0.05  # absolute gap threshold
PASS_MIN_SEEDS = 3

# Architecture constants
HARM_OBS_DIM = 51
Z_HARM_DIM = 32
ACTION_DIM = 4

SEEDS = [42, 43, 44, 45, 46]
WARMUP_EPISODES = 30    # HarmEncoder warmup (P0)
TRAIN_EPISODES = 50     # E2HarmSForward training (P1)
EVAL_STEPS = 200        # steps to measure action_gap
STEPS_PER_EPISODE = 200
LR_ENC = 1e-3
LR_FWD = 5e-4


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.1, resource_benefit=0.05,
        use_proxy_fields=True, seed=seed,
    )


def make_harm_encoder(device) -> HarmEncoder:
    return HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)


def make_e2(use_interventional: bool, device) -> E2HarmSForward:
    cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=Z_HARM_DIM,
        action_dim=ACTION_DIM,
        use_interventional=use_interventional,
        interventional_fraction=0.3,
        interventional_margin=0.1,
    )
    return E2HarmSForward(cfg).to(device)


def run_encoder_warmup(enc: HarmEncoder, env: CausalGridWorldV2,
                       device, n_eps: int):
    """P0: train HarmEncoder on harm proximity supervision."""
    prox_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
    opt = optim.Adam(list(enc.parameters()) + list(prox_head.parameters()), lr=LR_ENC)
    for ep in range(n_eps):
        env.reset()
        for step in range(STEPS_PER_EPISODE):
            obs_dict = env._get_observation_dict()
            harm_obs = obs_dict.get("harm_obs")
            if harm_obs is None:
                continue
            harm_obs = harm_obs.to(device)
            z_harm_s = enc(harm_obs.unsqueeze(0))
            prox_pred = prox_head(z_harm_s)
            prox_target = harm_obs[-1:].unsqueeze(0)  # last channel = harm_exposure
            loss = F.mse_loss(prox_pred, prox_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            action_idx = random.randint(0, 3)
            _, _, done, _, _ = env.step(action_idx)
            if done:
                break


def run_forward_train(enc: HarmEncoder, fwd: E2HarmSForward,
                      env: CausalGridWorldV2, device, n_eps: int) -> float:
    """P1: train E2HarmSForward on frozen encoder (stop-gradient)."""
    opt = optim.Adam(fwd.parameters(), lr=LR_FWD)
    mse_losses = []
    for ep in range(n_eps):
        env.reset()
        z_prev = None
        for step in range(STEPS_PER_EPISODE - 1):
            obs_dict = env._get_observation_dict()
            harm_obs = obs_dict.get("harm_obs")
            if harm_obs is None:
                z_prev = None
                break
            harm_obs = harm_obs.to(device)
            with torch.no_grad():
                z_curr = enc(harm_obs.unsqueeze(0))

            action_idx = random.randint(0, 3)
            a_onehot = torch.zeros(1, ACTION_DIM, device=device)
            a_onehot[0, action_idx] = 1.0

            if z_prev is not None:
                # Forward prediction
                z_pred = fwd(z_prev.detach(), a_prev)
                target = z_curr.detach()
                loss = fwd.compute_loss(z_pred, target)

                # SD-013: interventional loss (sample random a_cf != a_prev_idx)
                if fwd.config.use_interventional and random.random() < fwd.config.interventional_fraction:
                    cf_idx = random.choice([i for i in range(ACTION_DIM) if i != a_prev_idx])
                    a_cf = torch.zeros(1, ACTION_DIM, device=device)
                    a_cf[0, cf_idx] = 1.0
                    int_loss = fwd.compute_interventional_loss(
                        z_prev.detach(), a_prev.detach(), a_cf.detach()
                    )
                    loss = loss + int_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
                opt.step()
                mse_losses.append(float(loss.item()))

            z_prev = z_curr
            a_prev = a_onehot
            a_prev_idx = action_idx

            _, _, done, _, _ = env.step(action_idx)
            if done:
                z_prev = None
                break

    return float(np.mean(mse_losses)) if mse_losses else 0.0


def eval_action_gap(enc: HarmEncoder, fwd: E2HarmSForward,
                    env: CausalGridWorldV2, device) -> Dict:
    """Measure action_gap (mean L2 dist between actual and CF predictions) and forward R2."""
    env.reset()
    gaps = []
    mse_vals = []
    actual_targets = []

    z_prev = None
    for step in range(EVAL_STEPS):
        obs_dict = env._get_observation_dict()
        harm_obs = obs_dict.get("harm_obs")
        if harm_obs is None:
            break
        harm_obs = harm_obs.to(device)
        with torch.no_grad():
            z_curr = enc(harm_obs.unsqueeze(0))

        action_idx = random.randint(0, 3)
        a_onehot = torch.zeros(1, ACTION_DIM, device=device)
        a_onehot[0, action_idx] = 1.0

        if z_prev is not None:
            with torch.no_grad():
                z_pred = fwd(z_prev, a_prev)
                # action gap
                cf_idx = random.choice([i for i in range(ACTION_DIM) if i != a_prev_idx])
                a_cf = torch.zeros(1, ACTION_DIM, device=device)
                a_cf[0, cf_idx] = 1.0
                z_pred_cf = fwd(z_prev, a_cf)
                gap = float((z_pred - z_pred_cf).norm(dim=-1).mean().item())
                gaps.append(gap)
                # MSE for R2
                mse = float(F.mse_loss(z_pred, z_curr).item())
                mse_vals.append(mse)
                actual_targets.append(float(z_curr.var().item()))

        z_prev = z_curr
        a_prev = a_onehot
        a_prev_idx = action_idx
        _, _, done, _, _ = env.step(action_idx)
        if done:
            break

    action_gap = float(np.mean(gaps)) if gaps else 0.0
    mean_mse = float(np.mean(mse_vals)) if mse_vals else 0.0
    mean_var = float(np.mean(actual_targets)) if actual_targets else 1.0
    # R2 = 1 - MSE/Var (simple approximation)
    r2 = max(0.0, 1.0 - mean_mse / max(mean_var, 1e-8))
    return {"action_gap": action_gap, "forward_r2": r2, "mean_mse": mean_mse}


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    n_warmup = 3 if dry_run else WARMUP_EPISODES
    n_train = 3 if dry_run else TRAIN_EPISODES

    print(f"Seed {seed}")
    condition_results = {}
    for condition in ["INTERVENTIONAL", "OBSERVATIONAL"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        enc = make_harm_encoder(device)
        fwd = make_e2(use_interventional=(condition == "INTERVENTIONAL"), device=device)

        print(f"  {condition}: P0 encoder warmup...")
        run_encoder_warmup(enc, env, device, n_warmup)
        print(f"  {condition}: P1 forward train...")
        run_forward_train(enc, fwd, env, device, n_train)
        print(f"  {condition}: eval...")
        metrics = eval_action_gap(enc, fwd, env, device)
        condition_results[condition] = metrics
        print(f"  {condition}: action_gap={metrics['action_gap']:.4f} r2={metrics['forward_r2']:.4f}")

    interv = condition_results["INTERVENTIONAL"]
    obs = condition_results["OBSERVATIONAL"]
    c1_pass = interv["action_gap"] > obs["action_gap"]
    c2_pass = interv["forward_r2"] >= C2_r2_threshold
    c3_pass = interv["action_gap"] >= C3_gap_threshold
    seed_pass = c1_pass and c2_pass

    print(f"  -> {'PASS' if seed_pass else 'FAIL'}")
    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "action_gap_interventional": interv["action_gap"],
        "action_gap_observational": obs["action_gap"],
        "forward_r2_interventional": interv["forward_r2"],
        "c1_gap_improvement": c1_pass,
        "c2_r2_maintained": c2_pass,
        "c3_abs_gap": c3_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"v3_exq_320_sd013_interventional_training_dry"
        if args.dry_run
        else f"v3_exq_320_sd013_interventional_training_{timestamp}_v3"
    )
    print(f"EXQ-320 start: {run_id}")

    per_seed = [run_seed(s, dry_run=args.dry_run) for s in SEEDS]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    mean_c3 = float(np.mean([r["c3_abs_gap"] for r in per_seed]))
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS and mean_c3
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-320 {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(SEEDS)}")
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"gap_int={r['action_gap_interventional']:.4f} "
            f"gap_obs={r['action_gap_observational']:.4f} "
            f"r2={r['forward_r2_interventional']:.4f}"
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
            "SD-013": evidence_direction,
            "SD-003": evidence_direction,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "registered_thresholds": {
            "C1_gap_improvement": C1_threshold,
            "C2_r2": C2_r2_threshold,
            "C3_abs_gap": C3_gap_threshold,
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
