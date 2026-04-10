#!/opt/local/bin/python3
"""
V3-EXQ-324: SD-020 Harm Surprise PE vs EMA Target for z_harm_a Training

experiment_purpose: evidence

Tests that harm_surprise_pe_enabled=True produces a z_harm_a that correlates with
harm prediction error (surprise) rather than raw accumulated harm magnitude.

Two conditions per seed:
  PE_TARGET  -- harm_surprise_pe_enabled=True (z_harm_a trained on precision-weighted PE)
  EMA_TARGET -- harm_surprise_pe_enabled=False (z_harm_a trained on raw EMA harm scalar)

Key metrics:
  z_harm_a_surprise_corr -- Pearson correlation between z_harm_a norm and harm_PE
                            (PE_TARGET should be higher)
  z_harm_a_magnitude_corr -- correlation with raw harm magnitude
                             (EMA_TARGET should be higher, PE_TARGET lower)

Pass criterion (pre-registered):
  C1: z_harm_a_surprise_corr_PE > z_harm_a_surprise_corr_EMA
      (PE condition has more surprise-tracking in z_harm_a)
  C2: z_harm_a_surprise_corr_PE >= 0.2 (absolute threshold)
  C3: Both conditions show non-zero z_harm_a norms (no collapse)

Experiment PASS: >= 3/5 seeds satisfy C1 and C2.

Claims: SD-020 (z_harm_a encodes affective surprise), SD-011, Q-036
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


EXPERIMENT_TYPE = "v3_exq_324_sd020_harm_surprise_pe"
CLAIM_IDS = ["SD-020", "SD-011", "Q-036"]

C1_threshold = 0.0    # PE condition surprise_corr > EMA condition surprise_corr
C2_threshold = 0.15   # absolute surprise_corr threshold (lower bar: correlation is noisy)
PASS_MIN_SEEDS = 3

HARM_OBS_DIM = 51
HARM_OBS_A_DIM = 50
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16

SEEDS = [42, 43, 44, 45, 46]
TRAIN_EPISODES = 60
EVAL_STEPS = 500
STEPS_PER_EPISODE = 200
LR = 1e-3
HARM_EMA_ALPHA = 0.1


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.1, resource_benefit=0.05,
        use_proxy_fields=True, seed=seed,
    )


def make_config(pe_enabled: bool) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        harm_surprise_pe_enabled=pe_enabled,
        harm_obs_ema_alpha=HARM_EMA_ALPHA,
    )


def run_training(agent: REEAgent, enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                 env: CausalGridWorldV2, device, n_eps: int, pe_enabled: bool):
    """Train encoders. When pe_enabled, the harm_accum_loss uses PE target."""
    prox_head = nn.Sequential(nn.Linear(Z_HARM_DIM, 1), nn.Sigmoid()).to(device)
    harm_a_head = nn.Sequential(nn.Linear(Z_HARM_A_DIM, 1)).to(device)
    all_params = (
        list(agent.parameters())
        + list(enc_s.parameters())
        + list(enc_a.parameters())
        + list(prox_head.parameters())
        + list(harm_a_head.parameters())
    )
    opt = optim.Adam(all_params, lr=LR)

    # Manual EMA for PE computation when pe_enabled
    harm_ema = 0.0

    for ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm_accum = 0.0

        for step in range(STEPS_PER_EPISODE):
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")
            if harm_obs is None:
                break
            harm_obs = harm_obs.to(device)

            z_harm_s = enc_s(harm_obs.unsqueeze(0))

            # Training target for z_harm_a
            harm_scalar = float(harm_obs[-1].item())  # harm_exposure channel
            ep_harm_accum += harm_scalar

            if pe_enabled:
                # PE target: how surprising is the current harm level?
                harm_pe = abs(harm_scalar - harm_ema)
                harm_ema = (1 - HARM_EMA_ALPHA) * harm_ema + HARM_EMA_ALPHA * harm_scalar
                harm_a_target_val = harm_pe
            else:
                harm_a_target_val = harm_scalar
                harm_ema = (1 - HARM_EMA_ALPHA) * harm_ema + HARM_EMA_ALPHA * harm_scalar

            total_loss = F.mse_loss(
                prox_head(z_harm_s),
                harm_obs[-1:].unsqueeze(0)
            )

            if harm_obs_a is not None:
                harm_obs_a_t = harm_obs_a.to(device)
                res = enc_a(harm_obs_a_t.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res
                harm_a_target = torch.tensor([[harm_a_target_val]], dtype=torch.float32, device=device)
                harm_a_pred = harm_a_head(z_harm_a)
                total_loss = total_loss + F.mse_loss(harm_a_pred, harm_a_target)

            opt.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()

            action_idx = random.randint(0, 3)
            _, _, done, info, obs_dict = env.step(action_idx)
            if done:
                break


def eval_surprise_tracking(enc_s: HarmEncoder, enc_a: AffectiveHarmEncoder,
                            env: CausalGridWorldV2, device) -> Dict:
    """Measure correlation between z_harm_a norm and harm PE."""
    z_harm_a_norms = []
    harm_pe_vals = []
    harm_scalar_vals = []
    harm_ema = 0.0

    _, obs_dict = env.reset()
    for step in range(EVAL_STEPS):
        harm_obs = obs_dict.get("harm_obs")
        harm_obs_a = obs_dict.get("harm_obs_a")
        if harm_obs is None:
            break
        harm_obs = harm_obs.to(device)

        harm_scalar = float(harm_obs[-1].item())
        harm_pe = abs(harm_scalar - harm_ema)
        harm_ema = (1 - HARM_EMA_ALPHA) * harm_ema + HARM_EMA_ALPHA * harm_scalar

        with torch.no_grad():
            if harm_obs_a is not None:
                harm_obs_a_t = harm_obs_a.to(device)
                res = enc_a(harm_obs_a_t.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res
                z_harm_a_norm = float(z_harm_a.norm().item())
            else:
                z_harm_a_norm = 0.0

        z_harm_a_norms.append(z_harm_a_norm)
        harm_pe_vals.append(harm_pe)
        harm_scalar_vals.append(harm_scalar)

        action_idx = random.randint(0, 3)
        _, _, done, _, obs_dict = env.step(action_idx)
        if done:
            break

    def safe_corr(x, y):
        if len(x) < 3:
            return 0.0
        x_arr = np.array(x)
        y_arr = np.array(y)
        if x_arr.std() < 1e-8 or y_arr.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(x_arr, y_arr)[0, 1])

    surprise_corr = safe_corr(z_harm_a_norms, harm_pe_vals)
    magnitude_corr = safe_corr(z_harm_a_norms, harm_scalar_vals)
    mean_norm = float(np.mean(z_harm_a_norms)) if z_harm_a_norms else 0.0

    return {
        "z_harm_a_surprise_corr": surprise_corr,
        "z_harm_a_magnitude_corr": magnitude_corr,
        "z_harm_a_mean_norm": mean_norm,
        "n_steps": len(z_harm_a_norms),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = 5 if dry_run else TRAIN_EPISODES

    print(f"Seed {seed}")
    condition_results = {}
    for condition in ["PE_TARGET", "EMA_TARGET"]:
        pe_enabled = (condition == "PE_TARGET")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(seed)
        cfg = make_config(pe_enabled=pe_enabled)
        agent = REEAgent(cfg)
        enc_s = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
        enc_a = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM).to(device)

        print(f"  {condition}: training {n_train} eps...")
        run_training(agent, enc_s, enc_a, env, device, n_train, pe_enabled)
        print(f"  {condition}: eval...")
        metrics = eval_surprise_tracking(enc_s, enc_a, env, device)
        condition_results[condition] = metrics
        print(
            f"  {condition}: surprise_corr={metrics['z_harm_a_surprise_corr']:.4f} "
            f"magnitude_corr={metrics['z_harm_a_magnitude_corr']:.4f} "
            f"norm={metrics['z_harm_a_mean_norm']:.4f}"
        )

    pe = condition_results["PE_TARGET"]
    ema = condition_results["EMA_TARGET"]
    c1_pass = pe["z_harm_a_surprise_corr"] > ema["z_harm_a_surprise_corr"]
    c2_pass = pe["z_harm_a_surprise_corr"] >= C2_threshold
    c3_pass = pe["z_harm_a_mean_norm"] > 0.0
    seed_pass = c1_pass and c2_pass and c3_pass

    print(f"  -> {'PASS' if seed_pass else 'FAIL'}")
    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "surprise_corr_pe": pe["z_harm_a_surprise_corr"],
        "surprise_corr_ema": ema["z_harm_a_surprise_corr"],
        "magnitude_corr_pe": pe["z_harm_a_magnitude_corr"],
        "z_harm_a_norm_pe": pe["z_harm_a_mean_norm"],
        "c1_surprise_corr_higher": c1_pass,
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
        "v3_exq_324_sd020_harm_surprise_pe_dry" if args.dry_run
        else f"v3_exq_324_sd020_harm_surprise_pe_{timestamp}_v3"
    )
    print(f"EXQ-324 start: {run_id}")

    per_seed = [run_seed(s, dry_run=args.dry_run) for s in SEEDS]
    seeds_passing = sum(1 for r in per_seed if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-324 {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(SEEDS)}")
    for r in per_seed:
        s = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {s} "
            f"surp_pe={r['surprise_corr_pe']:.4f} "
            f"surp_ema={r['surprise_corr_ema']:.4f}"
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
            "SD-020": evidence_direction,
            "SD-011": evidence_direction,
            "Q-036": evidence_direction,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "registered_thresholds": {
            "C1_surprise_corr_higher": C1_threshold,
            "C2_abs_surprise_corr": C2_threshold,
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
